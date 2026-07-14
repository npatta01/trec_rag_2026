"""Raw-first, rate-limited retrieval for the frozen facet-aware fusion pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Mapping, Sequence

import requests

from .det_sparse_ledger import (
    RETRIEVER_VERSION,
    RawTransportResponse,
    RetrievalLedger,
    RetrievalLedgerError,
    RetrievalRequest,
    RetrievalTransport,
)
from .facet_aware_fusion_manifest import (
    ANALYZER_FINGERPRINT_SHA256,
    EXPERIMENT_ID,
    PRIOR_PILOT_TOPIC_IDS,
    PROTECTED_TOPIC_IDS,
    TOPIC_IDS,
    load_manifest,
    validate_manifest,
)
from .remote_client import rate_limited_session
from .remote_config import RemotePyseriniConfig
from .repo_env import find_repo_root, load_repo_env


ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
INDEX_ID = "climbmix-400b"
HITS = 100
MIN_INTERVAL_SECONDS = 3.0
MAX_EXTERNAL_REQUESTS = 24
MAX_EXTERNAL_REQUESTS_PER_TOPIC = 7
FORBIDDEN_TOPIC_IDS = PROTECTED_TOPIC_IDS | PRIOR_PILOT_TOPIC_IDS
SHARED_CACHE_DIR = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
)
LIMITER_STATE_PATH = SHARED_CACHE_DIR / "rate-limit.sqlite"
PREFLIGHT_SCHEMA_VERSION = "facet-aware-fusion-retrieval-preflight-v2"
SUMMARY_SCHEMA_VERSION = "facet-aware-fusion-retrieval-summary-v1"
CANDIDATE_SCHEMA_VERSION = "facet-aware-fusion-candidate-v1"
RECOVERY_SCHEMA_VERSION = "facet-aware-fusion-recovery-receipt-v2"
RECOVERY_CLAIM_SCHEMA_VERSION = "facet-aware-fusion-recovery-claim-v1"
RECOVERY_ROOT_CAUSE = "repo_env_not_loaded_before_token_read_v1"
RECOVERY_RECEIPT_NAME = "recovery_receipt.json"
RECOVERY_CLAIM_NAME = "recovery_claim.json"
_TIMEOUT_SECONDS = 30.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FACET_COUNTS = {"233": 3, "273": 7, "161": 7, "14": 7}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _runner_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _create_bytes(path: Path, payload: bytes) -> None:
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_fresh_final_outputs(run_dir: Path) -> None:
    for name in ("retrieval_summary.json", "candidates.jsonl"):
        path = Path(run_dir) / name
        if path.exists():
            raise FileExistsError(f"create-only output already exists: {path}")


def _facets(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = manifest.get("facets")
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("manifest facets must be a list of objects")
    return raw


def _reject_forbidden_topics(manifest: Mapping[str, object]) -> None:
    observed: list[str] = []
    topic_ids = manifest.get("topic_ids")
    if isinstance(topic_ids, list):
        observed.extend(row for row in topic_ids if isinstance(row, str))
    observed.extend(
        row.get("topic_id")
        for row in _facets(manifest)
        if isinstance(row.get("topic_id"), str)
    )
    for topic_id in observed:
        if topic_id in FORBIDDEN_TOPIC_IDS:
            raise ValueError(f"forbidden topic {topic_id} is not retrievable")


def _analyzer_fingerprint(manifest: Mapping[str, object]) -> str:
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("manifest hashes must be an object")
    value = hashes.get("analyzer_fingerprint_sha256")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("manifest analyzer fingerprint must be a lowercase SHA-256")
    return value


def _validated_facets(
    manifest: Mapping[str, object],
    *,
    cache_root: Path,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    _reject_forbidden_topics(manifest)
    facets = tuple(_facets(manifest))
    if manifest.get("qrels_opened") is not False:
        raise ValueError("retrieval requires qrels_opened=false")
    if manifest.get("topic_ids") != list(TOPIC_IDS):
        raise ValueError("retrieval requires the exact ordered held-out topic IDs")
    if len(facets) != MAX_EXTERNAL_REQUESTS:
        raise ValueError("facet retrieval requires exactly 24 manifest facets")
    observed_topics = [row.get("topic_id") for row in facets]
    expected_topics = [
        topic_id
        for topic_id in TOPIC_IDS
        for _ in range(_FACET_COUNTS[topic_id])
    ]
    if observed_topics != expected_topics:
        raise ValueError("facet retrieval topic counts/order differ from the frozen plan")
    if [row.get("manifest_order") for row in facets] != list(
        range(MAX_EXTERNAL_REQUESTS)
    ):
        raise ValueError("facet retrieval requires consecutive frozen manifest order")
    facet_ids = [row.get("facet_id") for row in facets]
    queries = [row.get("query") for row in facets]
    if any(not isinstance(value, str) or not value for value in facet_ids):
        raise ValueError("every facet requires a non-empty facet_id")
    if len(set(facet_ids)) != MAX_EXTERNAL_REQUESTS:
        raise ValueError("facet_id values must be unique")
    if any(not isinstance(value, str) or not value.strip() for value in queries):
        raise ValueError("every facet requires a non-empty query")
    _analyzer_fingerprint(manifest)
    validate_manifest(manifest, cache_root=Path(cache_root))
    return facets


def build_requests(
    manifest: Mapping[str, object],
    endpoint: str,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> tuple[RetrievalRequest, ...]:
    """Build the exact ordered 24-request allowlist from a validated manifest."""

    if endpoint != ENDPOINT:
        raise ValueError(f"facet retrieval requires the frozen endpoint {ENDPOINT}")
    facets = _validated_facets(manifest, cache_root=cache_root)
    analyzer_fingerprint = _analyzer_fingerprint(manifest)
    requests_to_run = tuple(
        RetrievalRequest.from_query(
            topic_id=str(facet["topic_id"]),
            variant_name=f"{EXPERIMENT_ID}:{facet['facet_id']}",
            query_text=str(facet["query"]),
            index_url=ENDPOINT,
            index_id=INDEX_ID,
            hits=HITS,
            analyzer_fingerprint_sha256=analyzer_fingerprint,
            retriever_version=RETRIEVER_VERSION,
        )
        for facet in facets
    )
    if len({row.identity.request_key for row in requests_to_run}) != len(
        requests_to_run
    ):
        raise ValueError("facet retrieval request identities must be unique")
    return requests_to_run


def build_live_config(*, api_token: str | None) -> RemotePyseriniConfig:
    """Return the sole live HTTP/limiter configuration authorized by the pilot."""

    if api_token is not None and (not isinstance(api_token, str) or not api_token):
        raise ValueError("api_token must be non-empty text or None")
    return RemotePyseriniConfig(
        index_url=ENDPOINT,
        api_token=api_token,
        hits=HITS,
        queries=(),
        min_interval_seconds=MIN_INTERVAL_SECONDS,
        burst=1,
        limiter_state_path=LIMITER_STATE_PATH,
    )


class RateLimitedFacetTransport:
    """One-shot HTTP transport bound to the persistent global rate limiter."""

    transport_version = "facet_aware_fusion_rate_limited_http_v1"
    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        config: RemotePyseriniConfig,
        allowed_requests: Sequence[RetrievalRequest],
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        expected_config = build_live_config(api_token=config.api_token)
        if config != expected_config:
            raise ValueError("facet transport configuration differs from frozen policy")
        if timeout_seconds != _TIMEOUT_SECONDS:
            raise ValueError("facet transport timeout is frozen at 30 seconds")
        allowed = tuple(allowed_requests)
        if len(allowed) != MAX_EXTERNAL_REQUESTS:
            raise ValueError("facet transport requires exactly 24 allowed requests")
        allowed_by_key = {row.identity.request_key: row for row in allowed}
        if len(allowed_by_key) != MAX_EXTERNAL_REQUESTS:
            raise ValueError("facet transport allowlist request keys must be unique")
        topic_ids = [row.identity.topic_id for row in allowed]
        expected_topics = [
            topic_id
            for topic_id in TOPIC_IDS
            for _ in range(_FACET_COUNTS[topic_id])
        ]
        if topic_ids != expected_topics:
            raise ValueError("facet transport allowlist topic order is not canonical")
        for request in allowed:
            identity = request.identity
            if identity.topic_id in FORBIDDEN_TOPIC_IDS:
                raise ValueError(f"forbidden topic {identity.topic_id} is not retrievable")
            if (
                identity.index_url != ENDPOINT
                or identity.index_id != INDEX_ID
                or identity.hits != HITS
                or identity.retriever_version != RETRIEVER_VERSION
                or identity.analyzer_fingerprint_sha256
                != ANALYZER_FINGERPRINT_SHA256
                or identity.bm25_k1 is not None
                or identity.bm25_b is not None
                or not identity.variant_name.startswith(f"{EXPERIMENT_ID}:")
            ):
                raise ValueError("facet transport allowlist violates frozen identity")
        self.config = config
        self.endpoint_url = ENDPOINT
        self.timeout_seconds = timeout_seconds
        self._allowed_requests = allowed_by_key
        self.session = session if session is not None else rate_limited_session(config)

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        if self._allowed_requests.get(request.identity.request_key) != request:
            raise ValueError("request is not in the frozen facet allowlist")
        if request.identity.topic_id in FORBIDDEN_TOPIC_IDS:
            raise ValueError(f"forbidden topic {request.identity.topic_id} is not retrievable")
        started = time.monotonic()
        response = self.session.get(
            ENDPOINT,
            params={"query": request.query_text, "hits": str(HITS)},
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token is not None
                    else {}
                ),
            },
            timeout=self.timeout_seconds,
            allow_redirects=False,
        )
        elapsed = getattr(response, "elapsed", None)
        elapsed_seconds = (
            float(elapsed.total_seconds())
            if elapsed is not None and hasattr(elapsed, "total_seconds")
            else time.monotonic() - started
        )
        body = response.content
        if not isinstance(body, bytes):
            body = bytes(body)
        # Return non-2xx bodies unchanged; RetrievalLedger commits bytes before
        # classifying the HTTP status and refuses automatic replay thereafter.
        return RawTransportResponse(
            status=int(response.status_code),
            headers=dict(response.headers),
            body=body,
            elapsed_seconds=elapsed_seconds,
        )


def preflight_retrieval(
    manifest: Mapping[str, object],
    ledger: RetrievalLedger,
    endpoint: str,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    """Validate the exact plan and cache identities without reserving a call."""

    cache_root = Path(cache_root)
    requests_to_run = build_requests(manifest, endpoint, cache_root=cache_root)
    if ledger.max_calls != MAX_EXTERNAL_REQUESTS:
        raise ValueError("facet retrieval ledger must freeze max_calls=24")
    if ledger.max_calls_per_topic != MAX_EXTERNAL_REQUESTS_PER_TOPIC:
        raise ValueError(
            "facet retrieval ledger must freeze max_calls_per_topic=7"
        )
    cache_status = tuple(ledger.has_verified_cache(row) for row in requests_to_run)
    existing_external_attempts = ledger.call_count()
    projected_external_attempts = (
        existing_external_attempts + len(requests_to_run) - sum(cache_status)
    )
    if projected_external_attempts > MAX_EXTERNAL_REQUESTS:
        raise ValueError(
            "facet retrieval would exceed the 24-call global budget; "
            f"projected {projected_external_attempts}"
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "authorization_mode": "normal",
        "experiment_id": EXPERIMENT_ID,
        "endpoint": ENDPOINT,
        "index_id": INDEX_ID,
        "hits": HITS,
        "manifest_sha256": _manifest_sha256(manifest),
        "runner_code_sha256": _runner_code_sha256(),
        "planned_requests": len(requests_to_run),
        "verified_cache_hits": sum(cache_status),
        "existing_external_attempts": existing_external_attempts,
        "projected_external_attempts": projected_external_attempts,
        "max_external_attempts": MAX_EXTERNAL_REQUESTS,
        "max_external_attempts_per_topic": MAX_EXTERNAL_REQUESTS_PER_TOPIC,
        "min_interval_seconds": MIN_INTERVAL_SECONDS,
        "minimum_start_span_seconds": MIN_INTERVAL_SECONDS
        * (MAX_EXTERNAL_REQUESTS - 1),
        "limiter_state_path": str(LIMITER_STATE_PATH),
        "source_cache_root": str(cache_root),
        "qrels_opened": False,
        "requests": [
            {
                "manifest_order": index,
                "request_key": request.identity.request_key,
                "identity": request.identity.canonical_dict(),
                "query_text": request.query_text,
                "verified_cache": verified_cache,
            }
            for index, (request, verified_cache) in enumerate(
                zip(requests_to_run, cache_status, strict=True),
                start=0,
            )
        ],
    }


def create_preflight(
    manifest: Mapping[str, object],
    ledger: RetrievalLedger,
    endpoint: str = ENDPOINT,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    """Create the one immutable preflight receipt and no retrieval calls."""

    _require_fresh_final_outputs(ledger.run_dir)
    payload = preflight_retrieval(
        manifest,
        ledger,
        endpoint,
        cache_root=cache_root,
    )
    _create_json(ledger.run_dir / "preflight.json", payload)
    return payload


def _failed_401_evidence(
    manifest: Mapping[str, object],
    failed_ledger: RetrievalLedger,
    endpoint: str,
    *,
    cache_root: Path,
) -> dict[str, object]:
    requests_to_run = build_requests(manifest, endpoint, cache_root=cache_root)
    report = failed_ledger.validate_run()
    if (
        report.reservations != 1
        or report.failures != 1
        or report.successes != 0
        or report.pending != 0
        or report.cache_hits != 0
    ):
        raise ValueError(
            "recovery requires exactly one terminal failed external attempt"
        )
    request = requests_to_run[0]
    request_key = request.identity.request_key
    outcome_path = failed_ledger.outcome_path(request_key)
    metadata_path = failed_ledger.raw_metadata_path(request_key)
    preflight_path = failed_ledger.run_dir / "preflight.json"
    if not outcome_path.is_file() or not metadata_path.is_file() or not preflight_path.is_file():
        raise ValueError("recovery source lacks the first request's immutable evidence")
    outcome = _read_json_object(outcome_path, "failed outcome")
    metadata = _read_json_object(metadata_path, "failed raw metadata")
    if (
        outcome.get("status") != "failure"
        or outcome.get("failure_type") != "response_validation_error"
        or outcome.get("message") != "HTTP status 401 is not successful"
        or outcome.get("raw_present") is not True
        or metadata.get("http_status") != 401
        or metadata.get("request_key") != request_key
        or outcome.get("response_sha256") != metadata.get("response_sha256")
    ):
        raise ValueError("recovery source is not the expected preserved HTTP 401")
    response_sha256 = outcome.get("response_sha256")
    if not isinstance(response_sha256, str) or _SHA256_RE.fullmatch(response_sha256) is None:
        raise ValueError("recovery source has an invalid response hash")
    return {
        "failed_run_dir": str(failed_ledger.run_dir.resolve()),
        "failed_request_key": request_key,
        "failed_response_sha256": response_sha256,
        "failed_http_status": 401,
        "failed_outcome_sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
        "failed_raw_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "failed_preflight_sha256": hashlib.sha256(
            preflight_path.read_bytes()
        ).hexdigest(),
    }


def _recovery_preflight(
    normal_preflight: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(normal_preflight)
    payload["authorization_mode"] = "recovery"
    payload["recovery_source"] = {
        "failed_run_dir": evidence["failed_run_dir"],
        "failed_request_key": evidence["failed_request_key"],
        "failed_response_sha256": evidence["failed_response_sha256"],
        "root_cause": RECOVERY_ROOT_CAUSE,
    }
    return payload


def _recovery_receipt(
    manifest: Mapping[str, object],
    failed_ledger: RetrievalLedger,
    recovery_ledger: RetrievalLedger,
    endpoint: str,
    preflight: Mapping[str, object],
    *,
    cache_root: Path,
) -> dict[str, object]:
    evidence = _failed_401_evidence(
        manifest,
        failed_ledger,
        endpoint,
        cache_root=cache_root,
    )
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "root_cause": RECOVERY_ROOT_CAUSE,
        "root_cause_detail": (
            "the runner read PYSERINI_API_TOKEN before loading the repository env"
        ),
        **evidence,
        "recovery_run_dir": str(recovery_ledger.run_dir.resolve()),
        "recovery_preflight_sha256": hashlib.sha256(
            _pretty_json_bytes(preflight)
        ).hexdigest(),
        "planned_requests": MAX_EXTERNAL_REQUESTS,
        "authorized_batches": 1,
        "failed_run_replay_allowed": False,
        "credentials_recorded": False,
    }


def _recovery_claim(
    receipt: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": RECOVERY_CLAIM_SCHEMA_VERSION,
        "root_cause": RECOVERY_ROOT_CAUSE,
        "failed_run_dir": receipt["failed_run_dir"],
        "failed_request_key": receipt["failed_request_key"],
        "failed_response_sha256": receipt["failed_response_sha256"],
        "recovery_run_dir": receipt["recovery_run_dir"],
        "recovery_preflight_sha256": hashlib.sha256(
            _pretty_json_bytes(preflight)
        ).hexdigest(),
        "recovery_receipt_sha256": hashlib.sha256(
            _pretty_json_bytes(receipt)
        ).hexdigest(),
        "authorized_batches": 1,
    }


def create_recovery_preflight(
    manifest: Mapping[str, object],
    failed_ledger: RetrievalLedger,
    recovery_ledger: RetrievalLedger,
    endpoint: str = ENDPOINT,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    """Bind one fresh preflight to an immutable failed-401 ledger receipt."""

    if failed_ledger.run_dir.resolve() == recovery_ledger.run_dir.resolve():
        raise ValueError("recovery output must differ from the failed run directory")
    _require_fresh_final_outputs(recovery_ledger.run_dir)
    receipt_path = recovery_ledger.run_dir / RECOVERY_RECEIPT_NAME
    preflight_path = recovery_ledger.run_dir / "preflight.json"
    for path in (receipt_path, preflight_path):
        if path.exists():
            raise FileExistsError(f"create-only output already exists: {path}")
    claim_path = failed_ledger.run_dir / RECOVERY_CLAIM_NAME
    if claim_path.exists():
        raise FileExistsError(f"create-only recovery claim already exists: {claim_path}")
    if recovery_ledger.validate_run().planned_requests != 0:
        raise ValueError("recovery output must have no prior retrieval invocation")
    normal_preflight = preflight_retrieval(
        manifest,
        recovery_ledger,
        endpoint,
        cache_root=cache_root,
    )
    evidence = _failed_401_evidence(
        manifest,
        failed_ledger,
        endpoint,
        cache_root=cache_root,
    )
    preflight = _recovery_preflight(normal_preflight, evidence)
    receipt = _recovery_receipt(
        manifest,
        failed_ledger,
        recovery_ledger,
        endpoint,
        preflight,
        cache_root=cache_root,
    )
    claim = _recovery_claim(receipt, preflight)
    # Every crash prefix is inert: consume the source first, then record the
    # destination receipt, and create the runnable recovery preflight last.
    _create_json(claim_path, claim)
    _create_json(receipt_path, receipt)
    _create_json(preflight_path, preflight)
    return receipt


def _validate_recovery_receipt_if_present(
    manifest: Mapping[str, object],
    recovery_ledger: RetrievalLedger,
    endpoint: str,
    preflight: Mapping[str, object],
    *,
    cache_root: Path,
) -> None:
    path = recovery_ledger.run_dir / RECOVERY_RECEIPT_NAME
    authorization_mode = preflight.get("authorization_mode")
    if authorization_mode == "normal":
        if path.exists():
            raise ValueError("normal preflight cannot carry a recovery receipt")
        return
    if authorization_mode != "recovery":
        raise ValueError("preflight lacks an explicit authorization mode")
    if not path.is_file():
        raise ValueError("recovery receipt is required by the recovery preflight")
    stored = _read_json_object(path, "recovery receipt")
    recovery_source = preflight.get("recovery_source")
    if not isinstance(recovery_source, Mapping):
        raise ValueError("recovery preflight lacks its source identity")
    failed_run_dir = recovery_source.get("failed_run_dir")
    if not isinstance(failed_run_dir, str) or not failed_run_dir:
        raise ValueError("recovery preflight has no failed run directory")
    failed_ledger = _ledger(Path(failed_run_dir), cache_root=cache_root)
    normal_preflight = preflight_retrieval(
        manifest,
        recovery_ledger,
        endpoint,
        cache_root=cache_root,
    )
    evidence = _failed_401_evidence(
        manifest,
        failed_ledger,
        endpoint,
        cache_root=cache_root,
    )
    expected_preflight = _recovery_preflight(normal_preflight, evidence)
    if dict(preflight) != expected_preflight:
        raise ValueError("recovery preflight differs from immutable source evidence")
    expected_receipt = _recovery_receipt(
        manifest,
        failed_ledger,
        recovery_ledger,
        endpoint,
        preflight,
        cache_root=cache_root,
    )
    if (
        stored != expected_receipt
        or path.read_bytes() != _pretty_json_bytes(stored)
    ):
        raise ValueError("recovery receipt differs from immutable failed-run evidence")
    claim_path = failed_ledger.run_dir / RECOVERY_CLAIM_NAME
    if not claim_path.is_file():
        raise ValueError("source recovery claim is required")
    stored_claim = _read_json_object(claim_path, "source recovery claim")
    expected_claim = _recovery_claim(expected_receipt, preflight)
    if (
        stored_claim != expected_claim
        or claim_path.read_bytes() != _pretty_json_bytes(stored_claim)
    ):
        raise ValueError("source recovery claim differs from authorized destination")


def _read_preflight(
    manifest: Mapping[str, object],
    ledger: RetrievalLedger,
    endpoint: str,
    *,
    cache_root: Path,
) -> tuple[dict[str, object], Path]:
    path = ledger.run_dir / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError(f"run requires the create-only preflight receipt: {path}")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"preflight receipt is unreadable: {path}") from exc
    if not isinstance(stored, dict):
        raise ValueError("preflight receipt must be a JSON object")
    authorization_mode = stored.get("authorization_mode")
    if authorization_mode not in {"normal", "recovery"}:
        raise ValueError("preflight lacks an explicit authorization mode")
    receipt_path = ledger.run_dir / RECOVERY_RECEIPT_NAME
    if authorization_mode == "recovery" and not receipt_path.is_file():
        raise ValueError("recovery receipt is required by the recovery preflight")
    if authorization_mode == "normal" and receipt_path.exists():
        raise ValueError("normal preflight cannot carry a recovery receipt")
    expected: dict[str, object] = preflight_retrieval(
        manifest,
        ledger,
        endpoint,
        cache_root=cache_root,
    )
    if authorization_mode == "recovery":
        recovery_source = stored.get("recovery_source")
        if (
            not isinstance(recovery_source, Mapping)
            or set(recovery_source)
            != {
                "failed_run_dir",
                "failed_request_key",
                "failed_response_sha256",
                "root_cause",
            }
            or recovery_source.get("root_cause") != RECOVERY_ROOT_CAUSE
        ):
            raise ValueError("recovery preflight has an invalid source identity")
        expected["authorization_mode"] = "recovery"
        expected["recovery_source"] = dict(recovery_source)
    if stored != expected:
        raise ValueError("stored preflight no longer matches the exact retrieval plan")
    return stored, path


def _candidate_bytes(
    requests_to_run: Sequence[RetrievalRequest],
    ledger: RetrievalLedger,
) -> tuple[bytes, int]:
    rows: list[bytes] = []
    for request in requests_to_run:
        facet_id = request.identity.variant_name.removeprefix(f"{EXPERIMENT_ID}:")
        result = ledger.load_verified_result(request)
        for candidate in result.candidates:
            row = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "topic_id": request.identity.topic_id,
                "facet_id": facet_id,
                "query_sha256": request.identity.query_sha256,
                "rank": candidate.rank,
                "docid": candidate.docid,
                "score": candidate.score,
                "text": candidate.text,
            }
            rows.append(_canonical_bytes(row) + b"\n")
    return b"".join(rows), len(rows)


def execute_retrieval(
    manifest: Mapping[str, object],
    ledger: RetrievalLedger,
    transport: RetrievalTransport,
    endpoint: str = ENDPOINT,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    """Execute the preflighted allowlist once, stopping at the first failure."""

    _require_fresh_final_outputs(ledger.run_dir)
    requests_to_run = build_requests(manifest, endpoint, cache_root=cache_root)
    preflight, preflight_path = _read_preflight(
        manifest,
        ledger,
        endpoint,
        cache_root=cache_root,
    )
    _validate_recovery_receipt_if_present(
        manifest,
        ledger,
        endpoint,
        preflight,
        cache_root=cache_root,
    )
    for request in requests_to_run:
        # Deliberately uncaught: one immutable failed attempt terminates the batch.
        ledger.retrieve(request, transport)

    report = ledger.validate_run()
    if (
        report.planned_requests != MAX_EXTERNAL_REQUESTS
        or report.successes + report.cache_hits != MAX_EXTERNAL_REQUESTS
        or report.failures != 0
        or report.pending != 0
    ):
        raise RetrievalLedgerError(
            "facet retrieval is incomplete after the authorized 24-request plan"
        )
    candidate_bytes, candidate_rows = _candidate_bytes(requests_to_run, ledger)
    candidate_path = ledger.run_dir / "candidates.jsonl"
    _create_bytes(candidate_path, candidate_bytes)
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "qrels_opened": False,
        "manifest_sha256": preflight["manifest_sha256"],
        "runner_code_sha256": preflight["runner_code_sha256"],
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_rows": candidate_rows,
        "planned_requests": report.planned_requests,
        "external_calls": report.external_calls,
        "successes": report.successes,
        "failures": report.failures,
        "pending": report.pending,
        "cache_hits": report.cache_hits,
        "per_topic_external_calls": report.per_topic_external_calls,
    }
    _create_json(ledger.run_dir / "retrieval_summary.json", summary)
    return summary


def _ledger(output: Path, *, cache_root: Path) -> RetrievalLedger:
    return RetrievalLedger(
        output,
        shared_cache_dir=cache_root,
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=MAX_EXTERNAL_REQUESTS_PER_TOPIC,
        min_results=50,
        required_text_results=50,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "recover-preflight"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--cache-root", type=Path, default=SHARED_CACHE_DIR)
        if command == "recover-preflight":
            child.add_argument("--failed-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = find_repo_root(Path.cwd())
    load_repo_env(repo_root)
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest, cache_root=args.cache_root)
    _validated_facets(manifest, cache_root=args.cache_root)
    ledger = _ledger(args.output, cache_root=args.cache_root)
    if args.command == "preflight":
        payload = create_preflight(manifest, ledger, cache_root=args.cache_root)
    elif args.command == "recover-preflight":
        failed_ledger = _ledger(args.failed_output, cache_root=args.cache_root)
        payload = create_recovery_preflight(
            manifest,
            failed_ledger,
            ledger,
            cache_root=args.cache_root,
        )
    else:
        _require_fresh_final_outputs(ledger.run_dir)
        preflight, _ = _read_preflight(
            manifest,
            ledger,
            ENDPOINT,
            cache_root=args.cache_root,
        )
        _validate_recovery_receipt_if_present(
            manifest,
            ledger,
            ENDPOINT,
            preflight,
            cache_root=args.cache_root,
        )
        config = build_live_config(
            api_token=os.environ.get("PYSERINI_API_TOKEN") or None
        )
        requests_to_run = build_requests(
            manifest,
            ENDPOINT,
            cache_root=args.cache_root,
        )
        transport = RateLimitedFacetTransport(config, requests_to_run)
        payload = execute_retrieval(
            manifest,
            ledger,
            transport,
            cache_root=args.cache_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

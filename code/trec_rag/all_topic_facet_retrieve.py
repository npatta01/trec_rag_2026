"""Raw-first exact-identity retrieval for the sealed all-topic facet plan.

The live ``run`` command is intentionally only a wiring surface.  Original
queries can never reach its transport: they must be authenticated top-1,000
cache hits.  Facet attempts are one-shot, spaced by the persistent limiter,
and represented by durable create-only ledger records.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import requests
from pyrate_limiter import FileLockSQLiteBucket, Limiter, RequestRate
from requests_ratelimiter import LimiterSession

from .all_topic_facet_contract import (
    ALL_TOPIC_IDS,
    EXPERIMENT_ID,
    FACET_DEPTH,
    ORIGINAL_DEPTH,
    ORIGINAL_INDEX,
    ORIGINAL_INDEX_URL,
    REQUEST_INTERVAL_SECONDS,
    REQUEST_PLAN_SCHEMA_VERSION,
    verify_planning,
)
from .deep_facet_candidate_run import (
    ENDPOINT,
    INDEX_ID,
    SHARED_CACHE_DIR,
    _load_verified_cache,
    _normalize_response,
    _store_cache,
    build_live_config,
)
from .det_sparse_ledger import RETRIEVER_VERSION, RawTransportResponse, RetrievalRequest
from .remote_config import RemotePyseriniConfig
from .repo_env import find_repo_root, load_repo_env


RETRIEVAL_PLAN_SCHEMA_VERSION = "all-topic-retrieval-plan-v1"
LEDGER_SCHEMA_VERSION = "all-topic-raw-first-ledger-v1"
STREAM_SCHEMA_VERSION = "all-topic-retrieval-stream-row-v1"
UNION_SCHEMA_VERSION = "all-topic-authenticated-union-row-v1"
SUMMARY_SCHEMA_VERSION = "all-topic-retrieval-summary-v1"
SEAL_SCHEMA_VERSION = "all-topic-retrieval-seal-v1"
EXPECTED_FACET_REQUEST_COUNT = 148
APPROVED_ORIGINAL_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/outputs/_retriever_cache/pyserini_remote"
)
_TIMEOUT_SECONDS = 60.0


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class RetrievalTransport(Protocol):
    one_shot_no_retry: bool

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse: ...


class _SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _write_exclusive(path, _pretty_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be objects")
    return rows


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_original_binding(row: object, topic_id: str) -> Mapping[str, object]:
    if not isinstance(row, Mapping) or row.get("topic_id") != topic_id:
        raise ValueError("original cache incomplete")
    identity = row.get("request_identity")
    provenance = row.get("raw_response_provenance")
    if (
        row.get("cache_hit") is not True
        or row.get("depth") != ORIGINAL_DEPTH
        or row.get("candidate_count") != ORIGINAL_DEPTH
        or not isinstance(identity, Mapping)
        or identity.get("topic_id") != topic_id
        or identity.get("variant_name") != "original"
        or identity.get("hits") != ORIGINAL_DEPTH
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("original cache incomplete")
    _require_sha(identity.get("query_sha256"), "original query hash")
    _require_sha(provenance.get("cache_content_sha256"), "original response hash")
    return row


def build_retrieval_plan(
    planning: Mapping[str, object],
    original_cache: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Validate the sealed request semantics against available original caches."""

    topic_ids = planning.get("topic_ids")
    if topic_ids != list(ALL_TOPIC_IDS):
        raise ValueError("topic drift from the authorized all-topic scope")
    if (
        planning.get("schema_version") != REQUEST_PLAN_SCHEMA_VERSION
        or planning.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("manifest drift from the sealed request plan")
    if (
        planning.get("original_depth") != ORIGINAL_DEPTH
        or planning.get("facet_depth") != FACET_DEPTH
        or planning.get("request_interval_seconds") != REQUEST_INTERVAL_SECONDS
    ):
        raise ValueError("depth drift from the sealed request plan")
    originals = planning.get("originals")
    if (
        planning.get("original_request_count") != 0
        or planning.get("original_cache_hit_count") != len(ALL_TOPIC_IDS)
        or not isinstance(originals, list)
        or len(originals) != len(ALL_TOPIC_IDS)
        or set(original_cache) != set(ALL_TOPIC_IDS)
    ):
        raise ValueError("original cache incomplete; network fallback is forbidden")
    original_by_topic = {
        str(row.get("topic_id")): row for row in originals if isinstance(row, Mapping)
    }
    if set(original_by_topic) != set(ALL_TOPIC_IDS):
        raise ValueError("original cache incomplete; network fallback is forbidden")
    for topic_id in ALL_TOPIC_IDS:
        sealed = _validate_original_binding(original_by_topic[topic_id], topic_id)
        supplied = original_cache.get(topic_id)
        if not isinstance(supplied, Mapping):
            raise ValueError("original cache incomplete; network fallback is forbidden")
        supplied_identity = supplied.get("request_identity")
        response_sha = supplied.get("response_sha256")
        if (
            supplied.get("topic_id") != topic_id
            or supplied.get("depth") != ORIGINAL_DEPTH
            or supplied.get("candidate_count") != ORIGINAL_DEPTH
            or supplied_identity != sealed.get("request_identity")
            or response_sha
            != sealed["raw_response_provenance"]["cache_content_sha256"]  # type: ignore[index]
        ):
            raise ValueError("original cache incomplete or exact identity mismatch")

    facets = planning.get("facet_requests")
    facet_count = planning.get("facet_request_count")
    if (
        not isinstance(facets, list)
        or isinstance(facet_count, bool)
        or not isinstance(facet_count, int)
        or facet_count != len(facets)
        or facet_count != EXPECTED_FACET_REQUEST_COUNT
        or planning.get("total_external_request_count") != facet_count
    ):
        raise ValueError("manifest drift in facet request count")
    seen_variants: set[str] = set()
    for order, row in enumerate(facets):
        if not isinstance(row, Mapping):
            raise ValueError("manifest drift in facet request schema")
        topic_id = row.get("topic_id")
        facet_id = row.get("facet_id")
        query = row.get("query")
        variant = row.get("variant_name")
        if topic_id not in ALL_TOPIC_IDS:
            raise ValueError("topic drift in facet requests")
        if (
            row.get("request_order") != order
            or row.get("depth") != FACET_DEPTH
            or not isinstance(facet_id, str)
            or not facet_id
            or not isinstance(query, str)
            or not query
            or variant != f"{EXPERIMENT_ID}:{facet_id}"
            or row.get("query_sha256") != _sha256_bytes(query.encode("utf-8"))
        ):
            raise ValueError("depth drift or manifest drift in facet requests")
        if variant in seen_variants:
            raise ValueError("manifest drift: facet request identities are not unique")
        seen_variants.add(str(variant))

    analyzer_sha = _require_sha(planning.get("analyzer_sha256"), "analyzer hash")
    root_sha = _require_sha(
        planning.get("planning_root_sha256"), "planning root hash"
    )
    return {
        "schema_version": RETRIEVAL_PLAN_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "topic_ids": list(ALL_TOPIC_IDS),
        "original_depth": ORIGINAL_DEPTH,
        "facet_depth": FACET_DEPTH,
        "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "original_request_count": 0,
        "originals": list(originals),
        "facet_requests": list(facets),
        "analyzer_sha256": analyzer_sha,
        "planning_root_sha256": root_sha,
        "planning_files": planning.get("planning_files"),
        "planning_seal_sha256": planning.get("planning_seal_sha256"),
        "approved_original_cache_root": planning.get("approved_original_cache_root"),
    }


def _build_facet_requests(plan: Mapping[str, object]) -> tuple[RetrievalRequest, ...]:
    analyzer_sha = str(plan["analyzer_sha256"])
    facets = plan["facet_requests"]
    assert isinstance(facets, list)
    requests_to_run = tuple(
        RetrievalRequest.from_query(
            topic_id=str(row["topic_id"]),
            variant_name=str(row["variant_name"]),
            query_text=str(row["query"]),
            index_url=ENDPOINT,
            index_id=INDEX_ID,
            hits=FACET_DEPTH,
            analyzer_fingerprint_sha256=analyzer_sha,
            retriever_version=RETRIEVER_VERSION,
        )
        for row in facets
        if isinstance(row, Mapping)
    )
    if len({request.identity.request_key for request in requests_to_run}) != len(
        requests_to_run
    ):
        raise ValueError("manifest drift: facet request keys are not unique")
    return requests_to_run


class RateLimitedFacetTransport:
    """Persistent one-start-per-three-seconds HTTP transport with no retry."""

    one_shot_no_retry = True
    redirects_allowed = False
    transport_version = "all_topic_rate_limited_requests_v1"

    def __init__(
        self,
        config: RemotePyseriniConfig,
        allowed_requests: Sequence[RetrievalRequest],
        *,
        session: requests.Session | object | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        if config != build_live_config(api_token=config.api_token):
            raise ValueError("transport config differs from the persistent 3s limiter policy")
        if timeout_seconds != _TIMEOUT_SECONDS:
            raise ValueError("transport timeout is frozen at 60 seconds")
        allowed = tuple(allowed_requests)
        self._allowed = {request.identity.request_key: request for request in allowed}
        if len(self._allowed) != len(allowed):
            raise ValueError("transport allowlist is not unique")
        for request in allowed:
            identity = request.identity
            if (
                identity.variant_name == "original"
                or not identity.variant_name.startswith(f"{EXPERIMENT_ID}:")
                or identity.hits != FACET_DEPTH
                or identity.index_url != ENDPOINT
                or identity.index_id != INDEX_ID
            ):
                raise ValueError("transport allowlist violates facet-only identity")
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.session = session if session is not None else _grant_timestamp_session(config)
        self.last_start_event: dict[str, object] | None = None

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        identity = request.identity
        if identity.variant_name == "original" or identity.hits != FACET_DEPTH:
            raise ValueError("original-query network request rejected")
        if self._allowed.get(identity.request_key) != request:
            raise ValueError("request is not in the frozen facet transport allowlist")
        started = time.monotonic()
        try:
            response = self.session.get(  # type: ignore[attr-defined]
                ENDPOINT,
                params={"query": request.query_text, "hits": str(FACET_DEPTH)},
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
        finally:
            granted = getattr(self.session, "last_grant_event", None)
            if isinstance(granted, Mapping):
                self.last_start_event = {
                    **dict(granted),
                    "request_key": request.identity.request_key,
                }
        elapsed = getattr(response, "elapsed", None)
        elapsed_seconds = (
            float(elapsed.total_seconds())
            if elapsed is not None and hasattr(elapsed, "total_seconds")
            else time.monotonic() - started
        )
        return RawTransportResponse(
            status=int(response.status_code),
            headers=dict(response.headers),
            body=bytes(response.content),
            elapsed_seconds=elapsed_seconds,
        )


class _GrantTimestampLimiterSession(LimiterSession):
    """LimiterSession that records wall time only after the SQLite grant."""

    last_grant_event: dict[str, object] | None = None

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def,override]
        with self.limiter.ratelimit(
            self._bucket_name(request), delay=True, max_delay=self.max_delay
        ):
            epoch = time.time()
            self.last_grant_event = {
                "started_at_epoch": epoch,
                "started_at_utc": datetime.fromtimestamp(
                    epoch, timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            }
            response = requests.Session.send(self, request, **kwargs)
            if response.status_code in self.limit_statuses:
                self._fill_bucket(request)
            return response


def _grant_timestamp_session(config: RemotePyseriniConfig) -> requests.Session:
    config.limiter_state_path.parent.mkdir(parents=True, exist_ok=True)
    limiter = Limiter(
        RequestRate(1, math.ceil(config.min_interval_seconds)),
        bucket_class=FileLockSQLiteBucket,
        bucket_kwargs={"path": config.limiter_state_path},
        time_function=time.time,
    )
    session = _GrantTimestampLimiterSession(
        limiter=limiter, per_host=True, max_delay=None
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_union(
    original_rows: Sequence[Mapping[str, object]],
    facet_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Deduplicate by exact topic/document identity while retaining every stream."""

    union: list[dict[str, object]] = []
    by_identity: dict[tuple[str, str], dict[str, object]] = {}
    provenance_keys = (
        "stream_id",
        "stream_rank",
        "request_identity",
        "query_sha256",
        "response_sha256",
    )
    for row in (*original_rows, *facet_rows):
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        text = row.get("text")
        if (
            not isinstance(topic_id, str)
            or not isinstance(document_id, str)
            or not document_id
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("candidate row identity/text is invalid")
        provenance = {key: row.get(key) for key in provenance_keys}
        if (
            not isinstance(provenance["stream_id"], str)
            or isinstance(provenance["stream_rank"], bool)
            or not isinstance(provenance["stream_rank"], int)
            or provenance["stream_rank"] < 1
            or not isinstance(provenance["request_identity"], Mapping)
        ):
            raise ValueError("candidate stream provenance is incomplete")
        _require_sha(provenance["query_sha256"], "stream query hash")
        _require_sha(provenance["response_sha256"], "stream response hash")
        if "score" in row:
            provenance["score"] = row["score"]
        key = (topic_id, document_id)
        existing = by_identity.get(key)
        if existing is None:
            existing = {
                "schema_version": UNION_SCHEMA_VERSION,
                "topic_id": topic_id,
                "document_id": document_id,
                "text": text,
                "stream_provenance": [],
            }
            by_identity[key] = existing
            union.append(existing)
        elif existing["text"] != text:
            raise ValueError("same topic/document has conflicting passage text")
        stream_provenance = existing["stream_provenance"]
        assert isinstance(stream_provenance, list)
        stream_provenance.append(provenance)
    return union


def _original_stream_rows(
    plan: Mapping[str, object], original_cache: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    sealed = {
        str(row["topic_id"]): row
        for row in plan["originals"]  # type: ignore[index]
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, object]] = []
    for topic_id in ALL_TOPIC_IDS:
        cache = original_cache[topic_id]
        candidates = cache.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != ORIGINAL_DEPTH:
            raise ValueError("original cache incomplete or candidate count mismatch")
        identity = sealed[topic_id]["request_identity"]
        assert isinstance(identity, Mapping)
        response_sha = str(cache["response_sha256"])
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError("original cache candidate schema mismatch")
            docid = candidate.get("docid")
            text = candidate.get("text") or candidate.get("doc")
            if (
                not isinstance(docid, str)
                or not docid
                or candidate.get("rank") != rank
                or not isinstance(text, str)
                or not text
            ):
                raise ValueError("original cache candidate identity/rank mismatch")
            rows.append(
                {
                    "schema_version": STREAM_SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "document_id": docid,
                    "text": text,
                    "score": candidate.get("score", 0.0),
                    "stream_id": "original",
                    "stream_rank": rank,
                    "request_identity": dict(identity),
                    "query_sha256": identity["query_sha256"],
                    "response_sha256": response_sha,
                }
            )
    return rows


def _approved_original_root(value: Path) -> Path:
    expected = APPROVED_ORIGINAL_CACHE_ROOT.resolve()
    supplied = Path(value).resolve()
    if supplied != expected:
        raise ValueError("approved original cache root differs from the fixed documented root")
    return supplied


def _load_original_caches(
    plan: Mapping[str, object], *, approved_root: Path
) -> dict[str, dict[str, object]]:
    root = _approved_original_root(approved_root)
    result: dict[str, dict[str, object]] = {}
    for sealed in plan.get("originals", []):
        if not isinstance(sealed, Mapping):
            raise ValueError("original cache incomplete")
        topic_id = str(sealed.get("topic_id"))
        path_value = sealed.get("cache_path")
        expected_sha = sealed.get("cache_sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha, str):
            raise ValueError("original cache incomplete")
        path = Path(path_value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("original cache path escapes the approved fixed root") from exc
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("original cache incomplete") from exc
        if _sha256_bytes(raw) != expected_sha or not isinstance(payload, Mapping):
            raise ValueError("original cache incomplete or content hash mismatch")
        response = payload.get("response")
        candidates = response.get("candidates") if isinstance(response, Mapping) else None
        if (
            payload.get("topic_id") != topic_id
            or payload.get("variant_name") != "original"
            or payload.get("hits") != ORIGINAL_DEPTH
            or payload.get("index") != ORIGINAL_INDEX
            or payload.get("index_url") != ORIGINAL_INDEX_URL
            or not isinstance(candidates, list)
            or len(candidates) != ORIGINAL_DEPTH
        ):
            raise ValueError("original cache incomplete or exact identity mismatch")
        result[topic_id] = {
            "topic_id": topic_id,
            "depth": ORIGINAL_DEPTH,
            "candidate_count": ORIGINAL_DEPTH,
            "request_identity": sealed["request_identity"],
            "response_sha256": expected_sha,
            "candidates": [
                {
                    "docid": candidate.get("docid"),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("score", 0.0),
                    "text": candidate.get("doc"),
                }
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ],
        }
    return result


def _load_sealed_planning(
    planning_dir: Path, *, approved_original_cache_root: Path
) -> dict[str, object]:
    planning_dir = Path(planning_dir)
    approved_root = _approved_original_root(approved_original_cache_root)
    plan = _read_json(planning_dir / "request_plan.json", "sealed request plan")
    originals = plan.get("originals")
    if not isinstance(originals, list) or not originals:
        raise ValueError("sealed request plan has no original cache bindings")
    verified = verify_planning(planning_dir, approved_cache_root=approved_root)
    manifest = _read_json(planning_dir / "manifest.json", "sealed facet manifest")
    analyzer = manifest.get("analyzer")
    if not isinstance(analyzer, Mapping):
        raise ValueError("sealed facet analyzer is invalid")
    enriched = dict(plan)
    enriched["analyzer_sha256"] = _sha256_bytes(_canonical_bytes(analyzer))
    enriched["planning_root_sha256"] = verified["root_sha256"]
    planning_seal_raw = (planning_dir / "SEALED.json").read_bytes()
    planning_seal = json.loads(planning_seal_raw)
    if not isinstance(planning_seal, Mapping) or not isinstance(
        planning_seal.get("files"), Mapping
    ):
        raise ValueError("sealed planning file bindings are invalid")
    enriched["planning_files"] = dict(planning_seal["files"])
    enriched["planning_seal_sha256"] = _sha256_bytes(planning_seal_raw)
    enriched["approved_original_cache_root"] = str(approved_root)
    return enriched


def _ledger_paths(output: Path, request_key: str) -> dict[str, Path]:
    root = output / "ledger"
    return {
        "reservation": root / "reservations" / f"{request_key}.json",
        "raw": root / "raw" / f"{request_key}.json",
        "metadata": root / "metadata" / f"{request_key}.json",
        "candidates": root / "candidates" / f"{request_key}.json",
        "outcome": root / "outcomes" / f"{request_key}.json",
    }


def _load_completed_attempt(
    paths: Mapping[str, Path], request: RetrievalRequest
) -> tuple[tuple[dict[str, object], ...], str] | None:
    if not paths["reservation"].exists():
        if any(path.exists() for name, path in paths.items() if name != "reservation"):
            raise ValueError("ledger artifacts exist without a durable reservation")
        return None
    if not paths["outcome"].exists():
        raise ValueError(f"pending attempt cannot be retried: {request.identity.request_key}")
    outcome = _read_json(paths["outcome"], "retrieval outcome")
    if outcome.get("status") == "failure":
        raise ValueError(
            f"immutable failed attempt cannot be retried: {request.identity.request_key}"
        )
    if outcome.get("status") not in {"success", "cache_hit"}:
        raise ValueError("retrieval outcome status is invalid")
    try:
        raw = paths["raw"].read_bytes()
        candidate_value = json.loads(paths["candidates"].read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("completed attempt artifacts are unreadable") from exc
    raw_sha = _sha256_bytes(raw)
    if outcome.get("raw_sha256") != raw_sha or not isinstance(candidate_value, list):
        raise ValueError("completed attempt hash mismatch")
    normalized = _normalize_response(raw)
    if list(normalized) != candidate_value:
        raise ValueError("completed attempt candidates mismatch raw response")
    return normalized, raw_sha


def _record_failure(
    path: Path,
    request: RetrievalRequest,
    *,
    failure_type: str,
    message: str,
    raw_sha256: str | None,
) -> None:
    _write_json(
        path,
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "status": "failure",
            "failure_type": failure_type,
            "message": message,
            "raw_sha256": raw_sha256,
        },
    )


def _shared_claim_paths(cache_root: Path, request_key: str) -> tuple[Path, Path]:
    root = Path(cache_root) / "all-topic-attempts-v1" / request_key[:2]
    return root / f"{request_key}.lock", root / f"{request_key}.json"


@contextmanager
def _exclusive_request_claim(cache_root: Path, request_key: str):
    lock_path, claim_path = _shared_claim_paths(cache_root, request_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield claim_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _transport_start_event(
    transport: RetrievalTransport, request: RetrievalRequest
) -> dict[str, object]:
    event = getattr(transport, "last_start_event", None)
    if (
        not isinstance(event, Mapping)
        or event.get("request_key") != request.identity.request_key
        or isinstance(event.get("started_at_epoch"), bool)
        or not isinstance(event.get("started_at_epoch"), (int, float))
        or not isinstance(event.get("started_at_utc"), str)
        or not str(event["started_at_utc"]).endswith("Z")
    ):
        raise ValueError("transport did not return a limiter-granted request-start event")
    return dict(event)


def _attempt_request(
    request: RetrievalRequest,
    transport: RetrievalTransport,
    output: Path,
    cache_root: Path,
    *,
    order: int,
    clock: Clock,
    last_start: float | None,
) -> tuple[tuple[dict[str, object], ...] | None, str | None, bool, float | None]:
    paths = _ledger_paths(output, request.identity.request_key)
    completed = _load_completed_attempt(paths, request)
    if completed is not None:
        return completed[0], completed[1], True, last_start
    _write_json(
        paths["reservation"],
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "identity": request.identity.canonical_dict(),
            "query_text": request.query_text,
            "request_order": order,
        },
    )
    with _exclusive_request_claim(cache_root, request.identity.request_key) as claim_path:
        # The cache is rechecked only after the cross-process identity lock.
        cached = _load_verified_cache(cache_root, request)
        shared_claim = (
            _read_json(claim_path, "shared request claim")
            if claim_path.exists()
            else None
        )
        if cached is not None:
            if (
                not isinstance(shared_claim, Mapping)
                or shared_claim.get("status") != "success"
                or shared_claim.get("request_key")
                != request.identity.request_key
                or shared_claim.get("identity")
                != request.identity.canonical_dict()
                or shared_claim.get("query_text") != request.query_text
            ):
                raise ValueError("exact cache lacks an authenticated shared attempt claim")
            event = shared_claim.get("start_event")
            if (
                not isinstance(event, Mapping)
                or event.get("request_key") != request.identity.request_key
            ):
                raise ValueError("shared attempt claim lacks limiter-granted start evidence")
            raw, candidates = cached
            raw_sha = _sha256_bytes(raw)
            if shared_claim.get("raw_sha256") != raw_sha:
                raise ValueError("shared attempt claim/cache hash mismatch")
            _write_exclusive(paths["raw"], raw)
            _write_json(
                paths["metadata"],
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "http_status": 200,
                    "elapsed_seconds": 0.0,
                    "raw_sha256": raw_sha,
                    "cache_hit": True,
                    "limiter_granted_request_key": event.get("request_key"),
                    "limiter_granted_start_epoch": event.get("started_at_epoch"),
                    "limiter_granted_start_utc": event.get("started_at_utc"),
                },
            )
            _write_exclusive(paths["candidates"], _pretty_bytes(list(candidates)))
            _write_json(
                paths["outcome"],
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "status": "cache_hit",
                    "candidate_count": len(candidates),
                    "raw_sha256": raw_sha,
                },
            )
            return candidates, raw_sha, True, last_start
        if shared_claim is not None:
            raise ValueError(
                f"immutable shared {shared_claim.get('status')} attempt cannot be retried: "
                f"{request.identity.request_key}"
            )
        _write_json(
            claim_path,
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "identity": request.identity.canonical_dict(),
                "query_text": request.query_text,
                "status": "pending",
            },
        )
        if last_start is not None:
            remaining = REQUEST_INTERVAL_SECONDS - (clock.monotonic() - last_start)
            if remaining > 0:
                clock.sleep(remaining)
        started = clock.monotonic()
        try:
            response = transport(request)
            event = _transport_start_event(transport, request)
        except Exception as exc:
            try:
                event = _transport_start_event(transport, request)
            except ValueError:
                event = None
            _replace_json(
                claim_path,
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "identity": request.identity.canonical_dict(),
                    "query_text": request.query_text,
                    "status": "failure",
                    "failure_type": "transport_exception",
                    "start_event": event,
                },
            )
            _record_failure(
                paths["outcome"],
                request,
                failure_type="transport_exception",
                message=f"{type(exc).__name__}: {exc}",
                raw_sha256=None,
            )
            return None, None, False, started
        raw = response.body
        _write_exclusive(paths["raw"], raw)
        raw_sha = _sha256_bytes(raw)
        _write_json(
            paths["metadata"],
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "http_status": response.status,
                "headers": dict(response.headers),
                "elapsed_seconds": float(response.elapsed_seconds),
                "raw_sha256": raw_sha,
                "cache_hit": False,
                "limiter_granted_request_key": event["request_key"],
                "limiter_granted_start_epoch": event["started_at_epoch"],
                "limiter_granted_start_utc": event["started_at_utc"],
            },
        )
        if response.status != 200:
            _replace_json(
                claim_path,
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "identity": request.identity.canonical_dict(),
                    "query_text": request.query_text,
                    "status": "failure",
                    "failure_type": "http_error",
                    "raw_sha256": raw_sha,
                    "start_event": event,
                },
            )
            _record_failure(
                paths["outcome"],
                request,
                failure_type="http_error",
                message=f"HTTP status {response.status} is not successful",
                raw_sha256=raw_sha,
            )
            return None, raw_sha, False, started
        try:
            candidates = _normalize_response(raw)
        except ValueError as exc:
            _replace_json(
                claim_path,
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "identity": request.identity.canonical_dict(),
                    "query_text": request.query_text,
                    "status": "failure",
                    "failure_type": "response_validation_error",
                    "raw_sha256": raw_sha,
                    "start_event": event,
                },
            )
            _record_failure(
                paths["outcome"],
                request,
                failure_type="response_validation_error",
                message=str(exc),
                raw_sha256=raw_sha,
            )
            return None, raw_sha, False, started
        _store_cache(cache_root, request, raw, candidates)
        _replace_json(
            claim_path,
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "identity": request.identity.canonical_dict(),
                "query_text": request.query_text,
                "status": "success",
                "raw_sha256": raw_sha,
                "start_event": event,
            },
        )
        _write_exclusive(paths["candidates"], _pretty_bytes(list(candidates)))
        _write_json(
            paths["outcome"],
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "status": "success",
                "candidate_count": len(candidates),
                "raw_sha256": raw_sha,
            },
        )
        return candidates, raw_sha, False, started


def _facet_stream_rows(
    plan: Mapping[str, object],
    requests_to_run: Sequence[RetrievalRequest],
    results: Sequence[tuple[tuple[dict[str, object], ...], str]],
) -> list[dict[str, object]]:
    facets = plan["facet_requests"]
    assert isinstance(facets, list)
    rows: list[dict[str, object]] = []
    for facet, request, (candidates, response_sha) in zip(
        facets, requests_to_run, results, strict=True
    ):
        assert isinstance(facet, Mapping)
        for candidate in candidates:
            rows.append(
                {
                    "schema_version": STREAM_SCHEMA_VERSION,
                    "topic_id": request.identity.topic_id,
                    "document_id": candidate["docid"],
                    "text": candidate["text"],
                    "score": candidate["score"],
                    "stream_id": facet["facet_id"],
                    "stream_rank": candidate["rank"],
                    "request_identity": request.identity.canonical_dict(),
                    "query_sha256": request.identity.query_sha256,
                    "response_sha256": response_sha,
                }
            )
    return rows


def _create_seal(output: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "RETRIEVAL_SEALED.json":
            raw = path.read_bytes()
            files[path.relative_to(output).as_posix()] = {
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
    seal: dict[str, object] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "root_sha256": _sha256_bytes(_canonical_bytes(files)),
    }
    _write_json(output / "RETRIEVAL_SEALED.json", seal)
    return seal


def run_retrieval(
    planning_dir: Path | Mapping[str, object],
    output_dir: Path,
    transport: RetrievalTransport,
    *,
    clock: Clock | None = None,
    cache_root: Path = SHARED_CACHE_DIR,
    original_cache: Mapping[str, Mapping[str, object]] | None = None,
    approved_original_cache_root: Path = APPROVED_ORIGINAL_CACHE_ROOT,
) -> dict[str, object]:
    """Run or resume the exact facet allowlist; original network calls are impossible."""

    if getattr(transport, "one_shot_no_retry", None) is not True:
        raise ValueError("retrieval transport must freeze one-shot no-retry")
    raw_plan = (
        dict(planning_dir)
        if isinstance(planning_dir, Mapping)
        else _load_sealed_planning(
            Path(planning_dir),
            approved_original_cache_root=approved_original_cache_root,
        )
    )
    cache_values = (
        dict(original_cache)
        if original_cache is not None
        else _load_original_caches(
            raw_plan, approved_root=approved_original_cache_root
        )
    )
    plan = build_retrieval_plan(raw_plan, cache_values)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "RETRIEVAL_SEALED.json").exists():
        if isinstance(planning_dir, Mapping):
            raise ValueError("sealed retrieval resume requires the original planning directory")
        verification = verify_retrieval(
            output,
            Path(planning_dir),
            approved_original_cache_root=approved_original_cache_root,
        )
        return _read_json(output / "retrieval_summary.json", "retrieval summary") | verification
    plan_path = output / "retrieval_plan.json"
    plan_raw = _pretty_bytes(plan)
    if plan_path.exists():
        if plan_path.read_bytes() != plan_raw:
            raise ValueError("manifest drift from the persisted retrieval plan")
    else:
        _write_exclusive(plan_path, plan_raw)

    requests_to_run = _build_facet_requests(plan)
    active_clock = clock if clock is not None else _SystemClock()
    last_start: float | None = None
    starts: list[float] = []
    results: list[tuple[tuple[dict[str, object], ...], str] | None] = []
    cache_hits = 0
    failures = 0
    for order, request in enumerate(requests_to_run):
        prior_start = last_start
        candidates, response_sha, cached, last_start = _attempt_request(
            request,
            transport,
            output,
            Path(cache_root),
            order=order,
            clock=active_clock,
            last_start=last_start,
        )
        if last_start is not None and last_start != prior_start:
            starts.append(last_start)
        cache_hits += int(cached)
        if candidates is None or response_sha is None:
            failures += 1
            results.append(None)
        else:
            results.append((candidates, response_sha))
    deltas = [round(right - left, 9) for left, right in zip(starts, starts[1:])]
    if failures:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "complete": False,
            "external_attempts": len(starts),
            "cache_hits": cache_hits,
            "failures": failures,
            "retry_count": 0,
            "original_network_requests": 0,
            "request_start_deltas": deltas,
        }

    complete_results = [result for result in results if result is not None]
    original_rows = _original_stream_rows(plan, cache_values)
    facet_rows = _facet_stream_rows(plan, requests_to_run, complete_results)
    union_rows = build_union(original_rows, facet_rows)
    _write_exclusive(output / "original_candidates.jsonl", _jsonl_bytes(original_rows))
    _write_exclusive(output / "facet_candidates.jsonl", _jsonl_bytes(facet_rows))
    _write_exclusive(output / "accepted_union.jsonl", _jsonl_bytes(union_rows))
    start_count, minimum_start_delta, actual_start_deltas = _verify_request_start_events(
        output, requests_to_run
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "topic_count": len(ALL_TOPIC_IDS),
        "facet_request_count": len(requests_to_run),
        "external_attempts": len(starts),
        "cache_hits": cache_hits,
        "failures": 0,
        "retry_count": 0,
        "original_network_requests": 0,
        "request_start_count": start_count,
        "minimum_request_start_delta_seconds": minimum_start_delta,
        "request_start_deltas": actual_start_deltas,
        "original_candidate_rows": len(original_rows),
        "facet_candidate_rows": len(facet_rows),
        "accepted_union_rows": len(union_rows),
        "planning_root_sha256": plan["planning_root_sha256"],
    }
    _write_json(output / "retrieval_summary.json", summary)
    _create_seal(output)
    return summary


def _verify_stream_provenance(
    output: Path,
    plan: Mapping[str, object],
    original_rows: Sequence[Mapping[str, object]],
    facet_rows: Sequence[Mapping[str, object]],
    source_original_cache: Mapping[str, Mapping[str, object]],
) -> None:
    if (
        plan.get("schema_version") != RETRIEVAL_PLAN_SCHEMA_VERSION
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("topic_ids") != list(ALL_TOPIC_IDS)
        or plan.get("original_depth") != ORIGINAL_DEPTH
        or plan.get("facet_depth") != FACET_DEPTH
        or plan.get("original_request_count") != 0
    ):
        raise ValueError("persisted retrieval plan semantics are invalid")
    sealed_originals = plan.get("originals")
    if not isinstance(sealed_originals, list):
        raise ValueError("persisted retrieval plan original bindings are invalid")
    original_by_topic = {
        str(row.get("topic_id")): row
        for row in sealed_originals
        if isinstance(row, Mapping)
    }
    observed_originals: dict[str, list[Mapping[str, object]]] = {
        topic_id: [] for topic_id in ALL_TOPIC_IDS
    }
    for row in original_rows:
        topic_id = row.get("topic_id")
        if topic_id not in observed_originals:
            raise ValueError("original stream provenance has topic drift")
        observed_originals[str(topic_id)].append(row)
    for topic_id in ALL_TOPIC_IDS:
        sealed = original_by_topic.get(topic_id)
        rows = observed_originals[topic_id]
        if not isinstance(sealed, Mapping) or len(rows) != ORIGINAL_DEPTH:
            raise ValueError("original stream provenance is incomplete")
        identity = sealed.get("request_identity")
        provenance = sealed.get("raw_response_provenance")
        if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("original stream provenance binding is invalid")
        source = source_original_cache.get(topic_id)
        source_candidates = source.get("candidates") if isinstance(source, Mapping) else None
        if not isinstance(source_candidates, list) or len(source_candidates) != ORIGINAL_DEPTH:
            raise ValueError("original cache candidate source is incomplete")
        for rank, (row, source_candidate) in enumerate(
            zip(rows, source_candidates, strict=True), start=1
        ):
            if not isinstance(source_candidate, Mapping):
                raise ValueError("original cache candidate source is invalid")
            if (
                row.get("schema_version") != STREAM_SCHEMA_VERSION
                or row.get("stream_id") != "original"
                or row.get("stream_rank") != rank
                or row.get("request_identity") != identity
                or row.get("query_sha256") != identity.get("query_sha256")
                or row.get("response_sha256")
                != provenance.get("cache_content_sha256")
                or row.get("document_id") != source_candidate.get("docid")
                or row.get("text") != source_candidate.get("text")
                or row.get("score") != source_candidate.get("score", 0.0)
                or source_candidate.get("rank") != rank
            ):
                raise ValueError(
                    "original cache candidate differs from sealed original stream provenance"
                )

    requests_to_run = _build_facet_requests(plan)
    facets = plan.get("facet_requests")
    if not isinstance(facets, list) or len(facets) != len(requests_to_run):
        raise ValueError("facet stream plan is invalid")
    rows_by_stream: dict[str, list[Mapping[str, object]]] = {}
    for row in facet_rows:
        stream_id = row.get("stream_id")
        if not isinstance(stream_id, str):
            raise ValueError("facet stream provenance has an invalid stream ID")
        rows_by_stream.setdefault(stream_id, []).append(row)
    expected_stream_ids = {
        str(facet.get("facet_id")) for facet in facets if isinstance(facet, Mapping)
    }
    if set(rows_by_stream) != expected_stream_ids:
        raise ValueError("facet stream provenance has stream drift")
    for order, (facet, request) in enumerate(zip(facets, requests_to_run, strict=True)):
        assert isinstance(facet, Mapping)
        stream_id = str(facet["facet_id"])
        rows = rows_by_stream[stream_id]
        paths = _ledger_paths(output, request.identity.request_key)
        reservation = _read_json(paths["reservation"], "facet reservation")
        metadata = _read_json(paths["metadata"], "facet response metadata")
        outcome = _read_json(paths["outcome"], "facet outcome")
        raw = paths["raw"].read_bytes()
        raw_sha = _sha256_bytes(raw)
        candidates = _normalize_response(raw)
        if (
            reservation.get("request_key") != request.identity.request_key
            or reservation.get("identity") != request.identity.canonical_dict()
            or reservation.get("query_text") != request.query_text
            or reservation.get("request_order") != order
            or metadata.get("raw_sha256") != raw_sha
            or outcome.get("status") not in {"success", "cache_hit"}
            or outcome.get("raw_sha256") != raw_sha
            or outcome.get("candidate_count") != FACET_DEPTH
            or len(rows) != FACET_DEPTH
        ):
            raise ValueError("facet stream provenance differs from its raw-first ledger")
        for candidate, row in zip(candidates, rows, strict=True):
            if (
                row.get("schema_version") != STREAM_SCHEMA_VERSION
                or row.get("topic_id") != request.identity.topic_id
                or row.get("document_id") != candidate["docid"]
                or row.get("text") != candidate["text"]
                or row.get("score") != candidate["score"]
                or row.get("stream_rank") != candidate["rank"]
                or row.get("request_identity") != request.identity.canonical_dict()
                or row.get("query_sha256") != request.identity.query_sha256
                or row.get("response_sha256") != raw_sha
            ):
                raise ValueError("facet stream provenance differs from its authenticated response")


def _verify_request_start_events(
    output: Path, requests_to_run: Sequence[RetrievalRequest]
) -> tuple[int, float | None, list[float]]:
    events: dict[str, tuple[float, str]] = {}
    for request in requests_to_run:
        metadata = _read_json(
            _ledger_paths(output, request.identity.request_key)["metadata"],
            "facet response metadata",
        )
        request_key = metadata.get("limiter_granted_request_key")
        epoch = metadata.get("limiter_granted_start_epoch")
        utc = metadata.get("limiter_granted_start_utc")
        if (
            request_key != request.identity.request_key
            or isinstance(epoch, bool)
            or not isinstance(epoch, (int, float))
            or not isinstance(utc, str)
            or not utc.endswith("Z")
        ):
            raise ValueError("limiter-granted request-start evidence is invalid")
        try:
            parsed = datetime.fromisoformat(utc.removesuffix("Z") + "+00:00").timestamp()
        except ValueError as exc:
            raise ValueError("limiter-granted request-start wall timestamp is invalid") from exc
        if abs(parsed - float(epoch)) > 0.001:
            raise ValueError("limiter-granted request-start epoch/UTC mismatch")
        previous = events.get(str(request_key))
        current = (float(epoch), utc)
        if previous is not None and previous != current:
            raise ValueError("request identity has conflicting limiter-granted starts")
        events[str(request_key)] = current
    if set(events) != {request.identity.request_key for request in requests_to_run}:
        raise ValueError("complete shared attempt ledger is missing request starts")
    chronological = sorted(epoch for epoch, _ in events.values())
    deltas = [right - left for left, right in zip(chronological, chronological[1:])]
    if any(delta < REQUEST_INTERVAL_SECONDS for delta in deltas):
        raise ValueError("shared request starts violate the 3-second limiter interval")
    return len(events), min(deltas) if deltas else None, deltas


def verify_retrieval(
    output_dir: Path,
    planning_dir: Path,
    *,
    approved_original_cache_root: Path = APPROVED_ORIGINAL_CACHE_ROOT,
) -> dict[str, object]:
    """Authenticate every sealed byte and recompute the accepted union."""

    output = Path(output_dir)
    trusted_raw_plan = _load_sealed_planning(
        Path(planning_dir),
        approved_original_cache_root=approved_original_cache_root,
    )
    source_original_cache = _load_original_caches(
        trusted_raw_plan, approved_root=approved_original_cache_root
    )
    trusted_plan = build_retrieval_plan(trusted_raw_plan, source_original_cache)
    seal = _read_json(output / "RETRIEVAL_SEALED.json", "retrieval seal")
    if (
        seal.get("schema_version") != SEAL_SCHEMA_VERSION
        or seal.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(seal.get("files"), Mapping)
    ):
        raise ValueError("retrieval seal identity mismatch")
    files = seal["files"]
    assert isinstance(files, Mapping)
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "RETRIEVAL_SEALED.json"
    }
    if observed != set(files):
        raise ValueError("retrieval seal mismatch: file set differs")
    for name, binding in files.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise ValueError("retrieval seal mismatch: invalid binding")
        raw = (output / name).read_bytes()
        if binding.get("bytes") != len(raw) or binding.get("sha256") != _sha256_bytes(raw):
            raise ValueError(f"retrieval seal mismatch for {name}")
    if seal.get("root_sha256") != _sha256_bytes(_canonical_bytes(files)):
        raise ValueError("retrieval seal mismatch: root hash differs")

    summary = _read_json(output / "retrieval_summary.json", "retrieval summary")
    if (
        summary.get("complete") is not True
        or summary.get("failures") != 0
        or summary.get("retry_count") != 0
        or summary.get("original_network_requests") != 0
    ):
        raise ValueError("retrieval summary is not a complete zero-retry run")
    original_rows = _read_jsonl(output / "original_candidates.jsonl", "original candidates")
    facet_rows = _read_jsonl(output / "facet_candidates.jsonl", "facet candidates")
    accepted = _read_jsonl(output / "accepted_union.jsonl", "accepted union")
    plan = _read_json(output / "retrieval_plan.json", "persisted retrieval plan")
    if plan != trusted_plan:
        raise ValueError("retrieval does not match the external sealed planning binding")
    requests_to_run = _build_facet_requests(trusted_plan)
    expected_original_rows = len(ALL_TOPIC_IDS) * ORIGINAL_DEPTH
    expected_facet_rows = len(requests_to_run) * FACET_DEPTH
    if (
        summary.get("topic_count") != len(ALL_TOPIC_IDS)
        or summary.get("facet_request_count") != len(requests_to_run)
        or summary.get("original_candidate_rows") != expected_original_rows
        or summary.get("facet_candidate_rows") != expected_facet_rows
        or len(original_rows) != expected_original_rows
        or len(facet_rows) != expected_facet_rows
        or not isinstance(summary.get("external_attempts"), int)
        or not isinstance(summary.get("cache_hits"), int)
        or summary["external_attempts"] + summary["cache_hits"]
        != len(requests_to_run)
    ):
        raise ValueError("retrieval summary facet count or candidate counts are invalid")
    _verify_stream_provenance(
        output, plan, original_rows, facet_rows, source_original_cache
    )
    start_count, minimum_delta, start_deltas = _verify_request_start_events(
        output, requests_to_run
    )
    if (
        summary.get("request_start_count") != start_count
        or summary.get("minimum_request_start_delta_seconds") != minimum_delta
        or summary.get("request_start_deltas") != start_deltas
    ):
        raise ValueError("retrieval summary request-start evidence is invalid")
    if accepted != build_union(original_rows, facet_rows):
        raise ValueError("accepted union does not match authenticated stream provenance")
    if summary.get("accepted_union_rows") != len(accepted):
        raise ValueError("accepted union row count differs from summary")
    topics = {str(row.get("topic_id")) for row in accepted}
    if topics != set(ALL_TOPIC_IDS):
        raise ValueError("accepted union topic coverage mismatch")
    return {
        "verified": True,
        "topic_count": len(topics),
        "accepted_union_rows": len(accepted),
        "root_sha256": seal["root_sha256"],
        "request_start_count": start_count,
        "minimum_request_start_delta_seconds": minimum_delta,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--planning", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--cache-root", type=Path, default=SHARED_CACHE_DIR)
    run_parser.add_argument(
        "--original-cache-root", type=Path, default=APPROVED_ORIGINAL_CACHE_ROOT
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--retrieval", type=Path, required=True)
    verify_parser.add_argument("--planning", type=Path, required=True)
    verify_parser.add_argument(
        "--original-cache-root", type=Path, default=APPROVED_ORIGINAL_CACHE_ROOT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = find_repo_root(Path.cwd())
    load_repo_env(repo_root)
    args = _parser().parse_args(argv)
    if args.command == "verify":
        payload = verify_retrieval(
            args.retrieval,
            args.planning,
            approved_original_cache_root=args.original_cache_root,
        )
    else:
        raw_plan = _load_sealed_planning(
            args.planning,
            approved_original_cache_root=args.original_cache_root,
        )
        original_cache = _load_original_caches(
            raw_plan, approved_root=args.original_cache_root
        )
        plan = build_retrieval_plan(raw_plan, original_cache)
        requests_to_run = _build_facet_requests(plan)
        transport = RateLimitedFacetTransport(
            build_live_config(api_token=os.environ.get("PYSERINI_API_TOKEN") or None),
            requests_to_run,
        )
        payload = run_retrieval(
            raw_plan,
            args.output,
            transport,
            cache_root=args.cache_root,
            original_cache=original_cache,
            approved_original_cache_root=args.original_cache_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

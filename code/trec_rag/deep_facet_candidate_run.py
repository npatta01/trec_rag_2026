"""Rate-limited, raw-first retrieval for the deep-facet candidate pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests

from .deep_facet_candidate_manifest import (
    EXCLUDED_TOPIC_IDS,
    EXPERIMENT_ID,
    FACET_HITS,
    TOPIC_IDS,
    assert_mutation_allowed,
    load_manifest,
    validate_manifest,
)
from .det_sparse_ledger import RETRIEVER_VERSION, RawTransportResponse, RetrievalRequest
from .remote_client import extract_text, rate_limited_session
from .remote_config import RemotePyseriniConfig
from .repo_env import find_repo_root, load_repo_env


ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
INDEX_ID = "climbmix-400b"
MIN_INTERVAL_SECONDS = 3.0
MAX_EXTERNAL_REQUESTS = 25
MAX_EXTERNAL_REQUESTS_PER_TOPIC = 7
LIMITER_STATE_PATH = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote/rate-limit.sqlite"
)
SHARED_CACHE_DIR = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
)
PREFLIGHT_SCHEMA_VERSION = "deep-facet-candidate-retrieval-preflight-v1"
CANDIDATE_SCHEMA_VERSION = "deep-facet-candidate-row-v1"
SUMMARY_SCHEMA_VERSION = "deep-facet-candidate-retrieval-summary-v1"
LEDGER_SCHEMA_VERSION = "deep-facet-candidate-raw-ledger-v1"
CACHE_SCHEMA_VERSION = "deep-facet-candidate-exact-cache-v1"
_FACET_COUNTS = {"219": 7, "72": 7, "300": 4, "84": 7}
_TIMEOUT_SECONDS = 60.0


class RetrievalTransport(Protocol):
    def __call__(self, request: RetrievalRequest) -> RawTransportResponse: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    _exclusive_bytes(path, _pretty_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_bytes(manifest))


def _analyzer_sha256(manifest: Mapping[str, object]) -> str:
    analyzer = manifest.get("analyzer")
    if not isinstance(analyzer, Mapping) or not analyzer:
        raise ValueError("manifest must freeze a non-empty analyzer object")
    return _sha256_bytes(_canonical_bytes(analyzer))


def _facets(manifest: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = manifest.get("facets")
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("manifest facets must be an array of objects")
    return tuple(raw)


def _validated_facets(
    manifest: Mapping[str, object], *, source_cache_root: Path
) -> tuple[Mapping[str, object], ...]:
    observed: list[str] = []
    topic_ids = manifest.get("topic_ids")
    if isinstance(topic_ids, list):
        observed.extend(str(value) for value in topic_ids)
    facets = _facets(manifest)
    observed.extend(str(row.get("topic_id")) for row in facets)
    if set(observed) & EXCLUDED_TOPIC_IDS:
        raise ValueError("retrieval manifest contains an excluded topic")
    if manifest.get("qrels_opened") is not False:
        raise ValueError("retrieval requires qrels_opened=false")
    if topic_ids != list(TOPIC_IDS):
        raise ValueError("retrieval requires the exact frozen topic order")
    if len(facets) != MAX_EXTERNAL_REQUESTS:
        raise ValueError("facet retrieval requires exactly 25 facets")
    expected_topics = [
        topic_id for topic_id in TOPIC_IDS for _ in range(_FACET_COUNTS[topic_id])
    ]
    if [row.get("topic_id") for row in facets] != expected_topics:
        raise ValueError("facet topic counts or order differ from the frozen plan")
    if [row.get("manifest_order") for row in facets] != list(
        range(MAX_EXTERNAL_REQUESTS)
    ):
        raise ValueError("facet manifest order is not consecutive")
    facet_ids = [row.get("facet_id") for row in facets]
    queries = [row.get("query") for row in facets]
    if len(set(facet_ids)) != MAX_EXTERNAL_REQUESTS or any(
        not isinstance(value, str) or not value for value in facet_ids
    ):
        raise ValueError("facet IDs must be unique non-empty text")
    if any(not isinstance(value, str) or not value.strip() for value in queries):
        raise ValueError("facet queries must be non-empty text")
    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, Mapping) or (
        retrieval.get("endpoint") != ENDPOINT
        or retrieval.get("index") != INDEX_ID
        or retrieval.get("facet_hits") != FACET_HITS
        or retrieval.get("max_external_requests") != MAX_EXTERNAL_REQUESTS
        or retrieval.get("minimum_interval_seconds") != 3
        or retrieval.get("retry_count") != 0
    ):
        raise ValueError("manifest retrieval policy differs from the frozen plan")
    _analyzer_sha256(manifest)
    validate_manifest(manifest, cache_root=Path(source_cache_root))
    return facets


def build_requests(
    manifest: Mapping[str, object],
    endpoint: str = ENDPOINT,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
) -> tuple[RetrievalRequest, ...]:
    """Build the exact ordered 25-request allowlist."""

    if endpoint != ENDPOINT:
        raise ValueError(f"facet retrieval requires the frozen endpoint {ENDPOINT}")
    facets = _validated_facets(manifest, source_cache_root=Path(cache_root))
    analyzer_sha256 = _analyzer_sha256(manifest)
    requests_to_run = tuple(
        RetrievalRequest.from_query(
            topic_id=str(facet["topic_id"]),
            variant_name=f"{EXPERIMENT_ID}:{facet['facet_id']}",
            query_text=str(facet["query"]),
            index_url=ENDPOINT,
            index_id=INDEX_ID,
            hits=FACET_HITS,
            analyzer_fingerprint_sha256=analyzer_sha256,
            retriever_version=RETRIEVER_VERSION,
        )
        for facet in facets
    )
    if len({row.identity.request_key for row in requests_to_run}) != len(requests_to_run):
        raise ValueError("facet request identities must be unique")
    return requests_to_run


def build_live_config(*, api_token: str | None) -> RemotePyseriniConfig:
    if api_token is not None and (not isinstance(api_token, str) or not api_token):
        raise ValueError("api_token must be non-empty text or None")
    return RemotePyseriniConfig(
        index_url=ENDPOINT,
        api_token=api_token,
        hits=FACET_HITS,
        queries=(),
        min_interval_seconds=MIN_INTERVAL_SECONDS,
        burst=1,
        limiter_state_path=LIMITER_STATE_PATH,
    )


class RateLimitedFacetTransport:
    """One-shot HTTP transport using the persistent requests-ratelimiter store."""

    one_shot_no_retry = True
    redirects_allowed = False
    transport_version = "deep_facet_rate_limited_requests_v1"

    def __init__(
        self,
        config: RemotePyseriniConfig,
        allowed_requests: Sequence[RetrievalRequest],
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        if config != build_live_config(api_token=config.api_token):
            raise ValueError("transport config differs from the frozen limiter policy")
        if timeout_seconds != _TIMEOUT_SECONDS:
            raise ValueError("transport timeout is frozen at 60 seconds")
        allowed = tuple(allowed_requests)
        if len(allowed) != MAX_EXTERNAL_REQUESTS:
            raise ValueError("transport requires exactly 25 allowed requests")
        self._allowed = {request.identity.request_key: request for request in allowed}
        if len(self._allowed) != MAX_EXTERNAL_REQUESTS:
            raise ValueError("transport request allowlist is not unique")
        for request in allowed:
            identity = request.identity
            if (
                identity.topic_id in EXCLUDED_TOPIC_IDS
                or identity.index_url != ENDPOINT
                or identity.index_id != INDEX_ID
                or identity.hits != FACET_HITS
                or identity.retriever_version != RETRIEVER_VERSION
                or identity.bm25_k1 is not None
                or identity.bm25_b is not None
                or not identity.variant_name.startswith(f"{EXPERIMENT_ID}:")
            ):
                raise ValueError("transport allowlist violates the frozen request identity")
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.session = session if session is not None else rate_limited_session(config)

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        if self._allowed.get(request.identity.request_key) != request:
            raise ValueError("request is not in the frozen transport allowlist")
        started = time.monotonic()
        response = self.session.get(
            ENDPOINT,
            params={"query": request.query_text, "hits": str(FACET_HITS)},
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
        return RawTransportResponse(
            status=int(response.status_code),
            headers=dict(response.headers),
            body=bytes(response.content),
            elapsed_seconds=elapsed_seconds,
        )


def _normalize_response(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not valid UTF-8 JSON") from exc
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, list) or len(candidates) != FACET_HITS:
        raise ValueError("response must contain exactly 200 unique text-bearing candidates")
    rows: list[dict[str, object]] = []
    seen_docids: set[str] = set()
    for position, value in enumerate(candidates, start=1):
        if not isinstance(value, Mapping):
            raise ValueError("response must contain exactly 200 unique text-bearing candidates")
        docid = value.get("docid") or value.get("id") or value.get("_id")
        rank = value.get("rank", position)
        score = value.get("score", 0.0)
        text = extract_text(value.get("doc") or value.get("contents") or value)
        if (
            not isinstance(docid, str)
            or not docid.strip()
            or docid.strip() in seen_docids
            or isinstance(rank, bool)
            or rank != position
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not text
        ):
            raise ValueError("response must contain exactly 200 unique text-bearing candidates")
        docid = docid.strip()
        seen_docids.add(docid)
        rows.append(
            {"docid": docid, "rank": position, "score": float(score), "text": text}
        )
    return tuple(rows)


def _cache_paths(cache_root: Path, request_key: str) -> tuple[Path, Path, Path]:
    prefix = Path(cache_root) / "deep-facet-v1" / request_key[:2]
    return (
        prefix / f"{request_key}.raw.json",
        prefix / f"{request_key}.candidates.json",
        prefix / f"{request_key}.manifest.json",
    )


def _load_verified_cache(
    cache_root: Path, request: RetrievalRequest
) -> tuple[bytes, tuple[dict[str, object], ...]] | None:
    raw_path, candidates_path, manifest_path = _cache_paths(
        cache_root, request.identity.request_key
    )
    exists = (raw_path.exists(), candidates_path.exists(), manifest_path.exists())
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError(f"partial exact cache exists for {request.identity.request_key}")
    manifest = _read_json(manifest_path, "exact cache manifest")
    raw = raw_path.read_bytes()
    candidate_bytes = candidates_path.read_bytes()
    try:
        candidates = json.loads(candidate_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("exact cache candidates are invalid JSON") from exc
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("identity") != request.identity.canonical_dict()
        or manifest.get("query_text") != request.query_text
        or manifest.get("raw_sha256") != _sha256_bytes(raw)
        or manifest.get("candidates_sha256") != _sha256_bytes(candidate_bytes)
        or manifest.get("candidate_count") != FACET_HITS
        or not isinstance(candidates, list)
    ):
        raise ValueError("exact cache identity or content hash mismatch")
    normalized = _normalize_response(raw)
    if list(normalized) != candidates:
        raise ValueError("exact cache normalized candidates mismatch")
    return raw, normalized


def _store_cache(
    cache_root: Path,
    request: RetrievalRequest,
    raw: bytes,
    candidates: Sequence[Mapping[str, object]],
) -> None:
    raw_path, candidates_path, manifest_path = _cache_paths(
        cache_root, request.identity.request_key
    )
    candidate_bytes = _pretty_bytes(list(candidates))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": request.identity.canonical_dict(),
        "query_text": request.query_text,
        "raw_sha256": _sha256_bytes(raw),
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "candidate_count": FACET_HITS,
    }
    _exclusive_bytes(raw_path, raw)
    _exclusive_bytes(candidates_path, candidate_bytes)
    _exclusive_json(manifest_path, manifest)


def _output_paths(output: Path, request_key: str) -> dict[str, Path]:
    ledger = Path(output) / "ledger"
    return {
        "reservation": ledger / "reservations" / f"{request_key}.json",
        "raw": ledger / "raw" / f"{request_key}.json",
        "metadata": ledger / "metadata" / f"{request_key}.json",
        "candidate": ledger / "candidates" / f"{request_key}.json",
        "outcome": ledger / "outcomes" / f"{request_key}.json",
    }


def _preflight_payload(
    manifest: Mapping[str, object],
    output: Path,
    *,
    cache_root: Path,
    source_cache_root: Path,
) -> dict[str, object]:
    requests_to_run = build_requests(manifest, ENDPOINT, cache_root=source_cache_root)
    cache_status = [
        _load_verified_cache(cache_root, request) is not None
        for request in requests_to_run
    ]
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "endpoint": ENDPOINT,
        "index_id": INDEX_ID,
        "hits": FACET_HITS,
        "manifest_sha256": _manifest_sha256(manifest),
        "runner_code_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "planned_requests": MAX_EXTERNAL_REQUESTS,
        "verified_cache_hits": sum(cache_status),
        "projected_external_attempts": MAX_EXTERNAL_REQUESTS - sum(cache_status),
        "max_external_attempts": MAX_EXTERNAL_REQUESTS,
        "max_external_attempts_per_topic": MAX_EXTERNAL_REQUESTS_PER_TOPIC,
        "min_interval_seconds": MIN_INTERVAL_SECONDS,
        "minimum_start_span_seconds": (MAX_EXTERNAL_REQUESTS - 1)
        * MIN_INTERVAL_SECONDS,
        "minimum_batch_window_seconds": MAX_EXTERNAL_REQUESTS
        * MIN_INTERVAL_SECONDS,
        "retry_count": 0,
        "limiter_state_path": str(LIMITER_STATE_PATH),
        "cache_root": str(Path(cache_root)),
        "source_cache_root": str(Path(source_cache_root)),
        "output": str(Path(output)),
        "qrels_opened": False,
        "requests": [
            {
                "manifest_order": order,
                "request_key": request.identity.request_key,
                "identity": request.identity.canonical_dict(),
                "query_text": request.query_text,
                "verified_cache": cached,
            }
            for order, (request, cached) in enumerate(
                zip(requests_to_run, cache_status, strict=True)
            )
        ],
    }


def _assert_upstream_mutable(output: Path) -> None:
    assert_mutation_allowed(output)
    assert_mutation_allowed(output.parent)


def create_preflight(
    manifest: Mapping[str, object],
    output: Path,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
    source_cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    output = Path(output)
    _assert_upstream_mutable(output)
    if (output / "retrieval_summary.json").exists() or (output / "candidates.jsonl").exists():
        raise FileExistsError("retrieval final output already exists")
    payload = _preflight_payload(
        manifest,
        output,
        cache_root=Path(cache_root),
        source_cache_root=Path(source_cache_root),
    )
    _exclusive_json(output / "preflight.json", payload)
    return payload


def _record_failure(
    path: Path,
    request: RetrievalRequest,
    *,
    failure_type: str,
    message: str,
    raw_sha256: str | None,
) -> None:
    _exclusive_json(
        path,
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "status": "failure",
            "failure_type": failure_type,
            "message": message,
            "raw_sha256": raw_sha256,
            "recorded_at": _utc_now(),
        },
    )


def _retrieve_one(
    request: RetrievalRequest,
    transport: RetrievalTransport,
    output: Path,
    *,
    order: int,
    cache_root: Path,
) -> tuple[tuple[dict[str, object], ...], bool]:
    paths = _output_paths(output, request.identity.request_key)
    if any(path.exists() for path in paths.values()):
        raise ValueError(f"retrieval replay refused for {request.identity.request_key}")
    _exclusive_json(
        paths["reservation"],
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "identity": request.identity.canonical_dict(),
            "query_text": request.query_text,
            "manifest_order": order,
            "reserved_at": _utc_now(),
        },
    )
    cached = _load_verified_cache(cache_root, request)
    if cached is not None:
        raw, candidates = cached
        cache_hit = True
        elapsed_seconds = 0.0
        http_status = 200
        headers: Mapping[str, object] = {}
    else:
        cache_hit = False
        try:
            response = transport(request)
        except Exception as exc:
            _record_failure(
                paths["outcome"],
                request,
                failure_type="transport_exception",
                message=f"{type(exc).__name__}: {exc}",
                raw_sha256=None,
            )
            raise
        raw = response.body
        elapsed_seconds = float(response.elapsed_seconds)
        http_status = int(response.status)
        headers = response.headers
        _exclusive_bytes(paths["raw"], raw)
        raw_sha256 = _sha256_bytes(raw)
        _exclusive_json(
            paths["metadata"],
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "http_status": http_status,
                "headers": dict(headers),
                "elapsed_seconds": elapsed_seconds,
                "raw_sha256": raw_sha256,
                "recorded_at": _utc_now(),
            },
        )
        if http_status != 200:
            message = f"HTTP status {http_status} is not successful"
            _record_failure(
                paths["outcome"],
                request,
                failure_type="http_error",
                message=message,
                raw_sha256=raw_sha256,
            )
            raise ValueError(message)
        try:
            candidates = _normalize_response(raw)
        except ValueError as exc:
            _record_failure(
                paths["outcome"],
                request,
                failure_type="response_validation_error",
                message=str(exc),
                raw_sha256=raw_sha256,
            )
            raise
        _store_cache(cache_root, request, raw, candidates)
    if cache_hit:
        _exclusive_bytes(paths["raw"], raw)
        _exclusive_json(
            paths["metadata"],
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": request.identity.request_key,
                "http_status": http_status,
                "headers": dict(headers),
                "elapsed_seconds": elapsed_seconds,
                "raw_sha256": _sha256_bytes(raw),
                "recorded_at": _utc_now(),
                "cache_hit": True,
            },
        )
    _exclusive_bytes(paths["candidate"], _pretty_bytes(list(candidates)))
    _exclusive_json(
        paths["outcome"],
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "status": "cache_hit" if cache_hit else "success",
            "candidate_count": len(candidates),
            "raw_sha256": _sha256_bytes(raw),
            "recorded_at": _utc_now(),
        },
    )
    return candidates, cache_hit


def execute_retrieval(
    manifest: Mapping[str, object],
    transport: RetrievalTransport,
    output: Path,
    *,
    cache_root: Path = SHARED_CACHE_DIR,
    source_cache_root: Path = SHARED_CACHE_DIR,
) -> dict[str, object]:
    """Execute the immutable allowlist once, stopping after the first failure."""

    output = Path(output)
    _assert_upstream_mutable(output)
    if getattr(transport, "one_shot_no_retry", None) is not True:
        raise ValueError("retrieval transport must explicitly freeze one-shot no-retry")
    for final in (output / "retrieval_summary.json", output / "candidates.jsonl"):
        if final.exists():
            raise FileExistsError(f"create-only output already exists: {final}")
    preflight_path = output / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("retrieval requires a create-only preflight receipt")
    stored_preflight = _read_json(preflight_path, "retrieval preflight")
    expected_preflight = _preflight_payload(
        manifest,
        output,
        cache_root=Path(cache_root),
        source_cache_root=Path(source_cache_root),
    )
    if stored_preflight != expected_preflight:
        raise ValueError("stored preflight no longer matches the exact retrieval plan")
    requests_to_run = build_requests(
        manifest, ENDPOINT, cache_root=Path(source_cache_root)
    )
    facet_by_id = {str(row["facet_id"]): row for row in _facets(manifest)}
    all_rows: list[dict[str, object]] = []
    cache_hits = 0
    for order, request in enumerate(requests_to_run):
        candidates, cached = _retrieve_one(
            request,
            transport,
            output,
            order=order,
            cache_root=Path(cache_root),
        )
        cache_hits += int(cached)
        facet_id = request.identity.variant_name.removeprefix(f"{EXPERIMENT_ID}:")
        facet = facet_by_id[facet_id]
        for candidate in candidates:
            all_rows.append(
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "request_key": request.identity.request_key,
                    "topic_id": request.identity.topic_id,
                    "facet_id": facet_id,
                    "manifest_order": facet["manifest_order"],
                    "query_sha256": request.identity.query_sha256,
                    **candidate,
                }
            )
    if len(all_rows) != MAX_EXTERNAL_REQUESTS * FACET_HITS:
        raise ValueError("complete retrieval must contain exactly 5,000 candidates")
    candidate_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in all_rows)
    _exclusive_bytes(output / "candidates.jsonl", candidate_bytes)
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "qrels_opened": False,
        "manifest_sha256": stored_preflight["manifest_sha256"],
        "runner_code_sha256": stored_preflight["runner_code_sha256"],
        "preflight_sha256": _sha256_bytes(preflight_path.read_bytes()),
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "candidate_rows": len(all_rows),
        "planned_requests": MAX_EXTERNAL_REQUESTS,
        "external_calls": MAX_EXTERNAL_REQUESTS - cache_hits,
        "successes": MAX_EXTERNAL_REQUESTS - cache_hits,
        "cache_hits": cache_hits,
        "failures": 0,
        "retry_count": 0,
    }
    _exclusive_json(output / "retrieval_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        child.add_argument("--cache-root", type=Path, default=SHARED_CACHE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = find_repo_root(Path.cwd())
    load_repo_env(repo_root)
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest, cache_root=args.cache_root)
    if args.command == "preflight":
        payload = create_preflight(
            manifest,
            args.output,
            cache_root=args.cache_root,
            source_cache_root=args.cache_root,
        )
    else:
        requests_to_run = build_requests(
            manifest, ENDPOINT, cache_root=args.cache_root
        )
        transport = RateLimitedFacetTransport(
            build_live_config(api_token=os.environ.get("PYSERINI_API_TOKEN") or None),
            requests_to_run,
        )
        payload = execute_retrieval(
            manifest,
            transport,
            args.output,
            cache_root=args.cache_root,
            source_cache_root=args.cache_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

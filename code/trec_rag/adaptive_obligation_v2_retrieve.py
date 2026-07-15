"""Plan and guard focused 1,000-hit retrieval for accepted v2 obligations.

The cache audit is deliberately transport-free.  Live execution is available only
through a separately approved, injected boundary and records exact response bytes
before decoding or normalization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_obligation_v2_contract import canonical_sha256
from .adaptive_obligation_v2_propose import (
    _capture_regular_file_no_symlinks,
    _fsync_directory,
    _open_directory_no_symlinks,
    _read_stable_regular_at,
)
from .det_sparse_ledger import RawTransportResponse
from .remote_client import extract_text, rate_limited_session
from .remote_config import RemotePyseriniConfig


RETRIEVAL_HITS = 1_000
MAX_ACCEPTED_O1_PER_TOPIC = 4
MAX_RETRIEVAL_REQUESTS = 16
TIMEOUT_SECONDS = 120
TRANSPORT_RETRY_COUNT = 0
REQUEST_START_INTERVAL_SECONDS = 3
INDEX_ID = "climbmix-400b"
RETRIEVER_VERSION = "adaptive_o1_pyserini_remote_raw_first_v1"
DEFAULT_ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
LIMITER_STATE_PATH = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/"
    "pyserini_remote/rate-limit.sqlite"
)
SHARED_CACHE_DIR = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
)
RATE_LIMITER_IDENTITY: dict[str, object] = {
    "implementation": "requests-ratelimiter+pyrate-limiter-filelock-sqlite",
    "state_path": str(LIMITER_STATE_PATH),
    "minimum_request_start_interval_seconds": REQUEST_START_INTERVAL_SECONDS,
    "burst": 1,
    "per_host": True,
    "max_delay": None,
}
PREFLIGHT_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-preflight-v1"
JOB_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-job-v1"
CACHE_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-cache-v1"
LEDGER_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-ledger-v1"
SUMMARY_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-summary-v1"
ESTIMATED_RAW_BYTES_PER_REQUEST = 8_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOPIC_ORDER = {topic_id: index for index, topic_id in enumerate(PILOT_TOPIC_IDS)}


class RetrievalTransport(Protocol):
    one_shot_no_retry: bool
    request_start_interval_seconds: int
    timeout_seconds: int

    def __call__(self, job: Mapping[str, object]) -> RawTransportResponse: ...


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return " ".join(value.split())


def remove_exact_duplicate_phrases(parts: Sequence[object]) -> list[str]:
    """Remove only case-insensitive, whitespace-normalized whole-part duplicates."""

    output: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = _require_text(part, "query part")
        normalized = text.casefold()
        if normalized not in seen:
            seen.add(normalized)
            output.append(text)
    return output


def render_o1_bm25_query(
    *,
    anchor_terms: Sequence[object],
    parent_text: object,
    o1_label: object,
    narrative: object,
) -> str:
    """Render anchors + complete O0 + O1 without copying the broad narrative."""

    del narrative
    if not isinstance(anchor_terms, Sequence) or isinstance(anchor_terms, (str, bytes)):
        raise TypeError("anchor_terms must be an array of text")
    if not anchor_terms:
        raise ValueError("anchor_terms must not be empty")
    if any(not isinstance(value, str) for value in anchor_terms):
        raise TypeError("anchor_terms must contain only text")
    parts = [*anchor_terms, parent_text, o1_label]
    return " ".join(remove_exact_duplicate_phrases(parts))


def _protected_precheck(rows: object) -> list[Mapping[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("accepted O1 rows must be an array")
    materialized: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("accepted O1 rows must contain objects")
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        materialized.append(row)
    if len(materialized) > MAX_RETRIEVAL_REQUESTS:
        raise ValueError(f"retrieval permits at most {MAX_RETRIEVAL_REQUESTS} requests")
    return materialized


def _validated_endpoint(value: object) -> str:
    endpoint = _require_text(value, "endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    return endpoint


def build_retrieval_jobs(
    accepted_o1_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
) -> list[dict[str, object]]:
    """Freeze one exact 1,000-hit request for every accepted O1."""

    rows = _protected_precheck(accepted_o1_rows)
    endpoint_text = _validated_endpoint(endpoint)
    index_text = _require_text(index_id, "index_id")
    topic_counts = Counter(str(row.get("topic_id")) for row in rows)
    if any(count > MAX_ACCEPTED_O1_PER_TOPIC for count in topic_counts.values()):
        raise ValueError("retrieval permits at most four accepted O1 records per topic")

    validated: list[tuple[tuple[object, ...], Mapping[str, object]]] = []
    seen_ids: set[str] = set()
    for row in rows:
        topic_id = str(row.get("topic_id"))
        if topic_id not in _TOPIC_ORDER:
            raise ValueError("accepted O1 topic is unknown")
        if row.get("accepted") is not True or row.get("decision") != "SUPPORTED":
            raise ValueError("retrieval requires an accepted SUPPORTED O1")
        proposal_id = _require_text(row.get("proposal_id"), "accepted O1 ID")
        if proposal_id in seen_ids:
            raise ValueError("accepted O1 IDs must be unique")
        seen_ids.add(proposal_id)
        parent_id = _require_text(row.get("parent_id"), "parent_id")
        manifest_order = row.get("parent_manifest_order")
        if type(manifest_order) is not int or manifest_order < 0:
            raise ValueError("parent_manifest_order must be a non-negative integer")
        _require_text(row.get("parent_text"), "parent_text")
        _require_text(row.get("label"), "label")
        anchors = row.get("anchor_terms")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise TypeError("anchor_terms must be an array of text")
        if not anchors or any(not isinstance(value, str) for value in anchors):
            raise ValueError("anchor_terms must contain non-empty text")
        validated.append(
            (
                (
                    _TOPIC_ORDER[topic_id],
                    manifest_order,
                    parent_id.casefold(),
                    proposal_id.casefold(),
                ),
                row,
            )
        )

    jobs: list[dict[str, object]] = []
    query_hashes: set[str] = set()
    for _key, row in sorted(validated, key=lambda pair: pair[0]):
        query_text = render_o1_bm25_query(
            anchor_terms=row["anchor_terms"],  # type: ignore[arg-type]
            parent_text=row["parent_text"],
            o1_label=row["label"],
            narrative=row.get("narrative"),
        )
        query_sha256 = _sha256(query_text.encode("utf-8"))
        if query_sha256 in query_hashes:
            raise ValueError("retrieval query hashes must be unique")
        query_hashes.add(query_sha256)
        accepted_o1_sha256 = canonical_sha256(dict(row))
        identity: dict[str, object] = {
            "topic_id": str(row["topic_id"]),
            "parent_id": str(row["parent_id"]),
            "accepted_o1_id": str(row["proposal_id"]),
            "accepted_o1_sha256": accepted_o1_sha256,
            "query_text": query_text,
            "query_sha256": query_sha256,
            "endpoint": endpoint_text,
            "index_id": index_text,
            "retriever_version": RETRIEVER_VERSION,
            "hits": RETRIEVAL_HITS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "transport_retry_count": TRANSPORT_RETRY_COUNT,
            "rate_limiter": dict(RATE_LIMITER_IDENTITY),
        }
        request_key = canonical_sha256(identity)
        jobs.append(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": request_key,
                "request_key": request_key,
                "topic_id": identity["topic_id"],
                "parent_id": identity["parent_id"],
                "accepted_o1_id": identity["accepted_o1_id"],
                "accepted_o1_sha256": accepted_o1_sha256,
                "query_text": query_text,
                "query_sha256": query_sha256,
                "endpoint": endpoint_text,
                "index_id": index_text,
                "retriever_version": RETRIEVER_VERSION,
                "hits": RETRIEVAL_HITS,
                "timeout_seconds": TIMEOUT_SECONDS,
                "transport_retry_count": TRANSPORT_RETRY_COUNT,
                "rate_limiter": dict(RATE_LIMITER_IDENTITY),
                "request_identity": identity,
            }
        )
    return jobs


def _cache_paths(cache_root: Path, request_key: str) -> tuple[Path, Path, Path]:
    prefix = Path(cache_root) / "adaptive-obligation-v2" / request_key[:2]
    return (
        prefix / f"{request_key}.raw.json",
        prefix / f"{request_key}.candidates.json",
        prefix / f"{request_key}.manifest.json",
    )


def _normalize_response(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not valid UTF-8 JSON") from exc
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, list) or len(candidates) != RETRIEVAL_HITS:
        raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, value in enumerate(candidates, start=1):
        if not isinstance(value, Mapping):
            raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
        docid = value.get("docid") or value.get("id") or value.get("_id")
        rank = value.get("rank", position)
        score = value.get("score", 0.0)
        text = extract_text(value.get("doc") or value.get("contents") or value)
        if (
            not isinstance(docid, str)
            or not docid.strip()
            or docid.strip() in seen
            or type(rank) is not int
            or rank != position
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not text
        ):
            raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
        docid = docid.strip()
        seen.add(docid)
        normalized.append(
            {"docid": docid, "rank": position, "score": float(score), "text": text}
        )
    return tuple(normalized)


def _capture_cache_file(path: Path) -> bytes:
    try:
        return _capture_regular_file_no_symlinks(path)
    except OSError as exc:
        raise ValueError("exact cache is missing or unsafe") from exc


def _load_verified_cache(
    cache_root: Path, job: Mapping[str, object]
) -> dict[str, object] | None:
    request_key = str(job.get("request_key"))
    raw_path, candidates_path, manifest_path = _cache_paths(cache_root, request_key)
    exists = tuple(os.path.lexists(path) for path in (raw_path, candidates_path, manifest_path))
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError(f"partial exact cache exists for {request_key}")
    raw = _capture_cache_file(raw_path)
    candidate_source = _capture_cache_file(candidates_path)
    manifest_source = _capture_cache_file(manifest_path)
    try:
        candidates = json.loads(candidate_source)
        manifest = json.loads(manifest_source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("exact cache JSON is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest_source != _pretty_bytes(manifest)
        or candidate_source != _pretty_bytes(candidates)
        or manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("request_key") != request_key
        or manifest.get("request_identity") != job.get("request_identity")
        or manifest.get("query_text") != job.get("query_text")
        or manifest.get("raw_sha256") != _sha256(raw)
        or manifest.get("candidates_sha256") != _sha256(candidate_source)
        or manifest.get("candidate_count") != RETRIEVAL_HITS
        or not isinstance(candidates, list)
    ):
        raise ValueError("exact cache identity or content hash mismatch")
    normalized = _normalize_response(raw)
    if list(normalized) != candidates:
        raise ValueError("exact cache normalized candidates mismatch")
    return {
        "raw": raw,
        "candidates": normalized,
        "hit": True,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "candidate_count": RETRIEVAL_HITS,
    }


def _cache_status(value: object, job: Mapping[str, object]) -> dict[str, object]:
    if value is None:
        return {"request_key": job["request_key"], "hit": False, "raw_bytes": 0}
    if not isinstance(value, Mapping):
        raise ValueError("cache probe must return an object or None")
    hit = value.get("hit")
    raw_bytes = value.get("raw_bytes")
    candidate_count = value.get("candidate_count")
    raw_sha256 = value.get("raw_sha256")
    if (
        hit is not True
        or type(raw_bytes) is not int
        or raw_bytes < 0
        or candidate_count != RETRIEVAL_HITS
        or not isinstance(raw_sha256, str)
        or not _SHA256_RE.fullmatch(raw_sha256)
    ):
        raise ValueError("cache probe returned an invalid verified-hit record")
    return {
        "request_key": job["request_key"],
        "hit": True,
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha256,
        "candidate_count": RETRIEVAL_HITS,
    }


def _verify_job(job: object) -> Mapping[str, object]:
    if not isinstance(job, Mapping):
        raise ValueError("retrieval job must be an object")
    identity = job.get("request_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("retrieval request identity is missing")
    request_key = job.get("request_key")
    if (
        job.get("schema_version") != JOB_SCHEMA_VERSION
        or not isinstance(request_key, str)
        or request_key != canonical_sha256(dict(identity))
        or job.get("job_id") != request_key
        or job.get("query_text") != identity.get("query_text")
        or job.get("query_sha256") != _sha256(str(job.get("query_text")).encode("utf-8"))
        or job.get("query_sha256") != identity.get("query_sha256")
    ):
        raise ValueError("retrieval request identity differs")
    topic_id = str(identity.get("topic_id"))
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    expected = {
        "topic_id": job.get("topic_id"),
        "parent_id": job.get("parent_id"),
        "accepted_o1_id": job.get("accepted_o1_id"),
        "accepted_o1_sha256": job.get("accepted_o1_sha256"),
        "query_text": job.get("query_text"),
        "query_sha256": job.get("query_sha256"),
        "endpoint": job.get("endpoint"),
        "index_id": job.get("index_id"),
        "retriever_version": RETRIEVER_VERSION,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": RATE_LIMITER_IDENTITY,
    }
    if dict(identity) != expected or topic_id not in _TOPIC_ORDER:
        raise ValueError("retrieval request identity differs")
    return job


def _verify_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise ValueError("retrieval preflight receipt must be an object")
    expected_fields = {
        "schema_version",
        "status",
        "topic_ids",
        "planned_request_count",
        "verified_cache_hits",
        "verified_cache_misses",
        "expected_raw_rows",
        "observed_cached_raw_bytes",
        "estimated_new_raw_bytes",
        "estimated_total_raw_bytes",
        "primary_external_attempts",
        "maximum_external_attempts",
        "hits",
        "timeout_seconds",
        "transport_retry_count",
        "rate_limiter",
        "cache_root",
        "cache_audit",
        "request_inventory_sha256",
        "requests",
        "qrels_opened",
        "network_call_count",
        "retrieval_call_count",
        "paid_call_count",
    }
    if set(receipt) != expected_fields:
        raise ValueError("retrieval preflight fields differ")
    if receipt.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError("retrieval preflight topic_ids differ")
    jobs = receipt.get("requests")
    if not isinstance(jobs, list):
        raise ValueError("retrieval preflight request inventory is missing")
    verified_jobs = [_verify_job(job) for job in jobs]
    if len(jobs) > MAX_RETRIEVAL_REQUESTS:
        raise ValueError("retrieval preflight exceeds the 16-request ceiling")
    counts = Counter(str(job["topic_id"]) for job in verified_jobs)
    if any(count > MAX_ACCEPTED_O1_PER_TOPIC for count in counts.values()):
        raise ValueError("retrieval preflight exceeds the per-topic ceiling")
    keys = [str(job["request_key"]) for job in verified_jobs]
    queries = [str(job["query_sha256"]) for job in verified_jobs]
    cache_audit = receipt.get("cache_audit")
    if (
        len(set(keys)) != len(keys)
        or len(set(queries)) != len(queries)
        or not isinstance(cache_audit, list)
        or len(cache_audit) != len(jobs)
    ):
        raise ValueError("retrieval preflight inventories differ")
    for job, status in zip(verified_jobs, cache_audit, strict=True):
        if not isinstance(status, Mapping) or status.get("request_key") != job["request_key"]:
            raise ValueError("retrieval preflight cache audit differs")
        _cache_status(status if status.get("hit") is True else None, job)
    hits = sum(status.get("hit") is True for status in cache_audit if isinstance(status, Mapping))
    misses = len(jobs) - hits
    cached_bytes = sum(
        int(status.get("raw_bytes", 0))
        for status in cache_audit
        if isinstance(status, Mapping) and status.get("hit") is True
    )
    expected_static = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "planned_request_count": len(jobs),
        "verified_cache_hits": hits,
        "verified_cache_misses": misses,
        "expected_raw_rows": len(jobs) * RETRIEVAL_HITS,
        "observed_cached_raw_bytes": cached_bytes,
        "estimated_new_raw_bytes": misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "estimated_total_raw_bytes": cached_bytes + misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "primary_external_attempts": misses,
        "maximum_external_attempts": misses,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": RATE_LIMITER_IDENTITY,
        "request_inventory_sha256": canonical_sha256(jobs),
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "paid_call_count": 0,
    }
    for name, expected in expected_static.items():
        if receipt.get(name) != expected:
            raise ValueError(f"retrieval preflight {name} differs")
    cache_root = receipt.get("cache_root")
    if not isinstance(cache_root, str) or not Path(cache_root).is_absolute():
        raise ValueError("retrieval preflight cache_root must be absolute")
    return receipt


def _exclusive_bytes(path: Path, source: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(source)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _exclusive_json(path: Path, value: object) -> None:
    _exclusive_bytes(path, _pretty_bytes(value))


def _publish_preflight(output_dir: Path, receipt: Mapping[str, object]) -> None:
    output = Path(output_dir)
    if not output.name or output.name in {".", ".."}:
        raise ValueError("preflight output path is unsafe")
    try:
        parent_fd = _open_directory_no_symlinks(output.parent)
    except OSError as exc:
        raise ValueError("preflight output parent is missing or unsafe") from exc
    staging_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    published = False
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"create-only preflight already exists: {output}")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        source = _pretty_bytes(receipt)
        file_fd = os.open(
            "receipt.json",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging_fd,
        )
        try:
            view = memoryview(source)
            while view:
                written = os.write(file_fd, view)
                view = view[written:]
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(staging_fd)
        os.rename(
            staging_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
    finally:
        if staging_fd is not None:
            if not published:
                try:
                    os.unlink("receipt.json", dir_fd=staging_fd)
                except FileNotFoundError:
                    pass
            os.close(staging_fd)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def audit_retrieval_cache(
    accepted_o1_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
    cache_root: Path = SHARED_CACHE_DIR,
    cache_loader: Callable[[dict[str, object]], object] | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Audit exact cache identities without constructing transport or sessions."""

    jobs = build_retrieval_jobs(
        accepted_o1_rows, endpoint=endpoint, index_id=index_id
    )
    root = Path(cache_root).absolute()
    statuses: list[dict[str, object]] = []
    for job in jobs:
        observed = (
            cache_loader(job)
            if cache_loader is not None
            else _load_verified_cache(root, job)
        )
        statuses.append(_cache_status(observed, job))
    hits = sum(status["hit"] is True for status in statuses)
    misses = len(statuses) - hits
    cached_bytes = sum(int(status["raw_bytes"]) for status in statuses if status["hit"] is True)
    receipt: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "planned_request_count": len(jobs),
        "verified_cache_hits": hits,
        "verified_cache_misses": misses,
        "expected_raw_rows": len(jobs) * RETRIEVAL_HITS,
        "observed_cached_raw_bytes": cached_bytes,
        "estimated_new_raw_bytes": misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "estimated_total_raw_bytes": cached_bytes + misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "primary_external_attempts": misses,
        "maximum_external_attempts": misses,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": dict(RATE_LIMITER_IDENTITY),
        "cache_root": str(root),
        "cache_audit": statuses,
        "request_inventory_sha256": canonical_sha256(jobs),
        "requests": jobs,
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "paid_call_count": 0,
    }
    _verify_receipt(receipt)
    if output_dir is not None:
        _publish_preflight(Path(output_dir), receipt)
    return receipt


def _capture_preflight_source(path: Path) -> bytes:
    expected = {"receipt.json"}
    try:
        descriptor = _open_directory_no_symlinks(path)
        try:
            before = os.fstat(descriptor)
            names_before = set(os.listdir(descriptor))
            if names_before != expected:
                raise OSError("preflight inventory differs")
            source = _read_stable_regular_at(
                descriptor, "receipt.json", require_single_link=True
            )
            names_after = set(os.listdir(descriptor))
            after = os.fstat(descriptor)
            if names_after != names_before or (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise OSError("preflight changed during capture")
            return source
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("retrieval preflight is missing or unsafe") from exc


def _parse_preflight_source(source: bytes) -> dict[str, object]:
    try:
        receipt = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("retrieval preflight receipt is invalid") from exc
    if not isinstance(receipt, dict) or source != _pretty_bytes(receipt):
        raise ValueError("retrieval preflight receipt is not canonical")
    return _verify_receipt(receipt)


def verify_retrieval_preflight(preflight_dir: Path) -> dict[str, object]:
    return _parse_preflight_source(_capture_preflight_source(Path(preflight_dir)))


def _capture_retrieval_approval(path: Path) -> tuple[dict[str, object], str]:
    try:
        source = _capture_regular_file_no_symlinks(Path(path))
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("retrieval approval required") from exc
    required = {
        "schema_version",
        "stage",
        "preflight_sha256",
        "planned_request_count",
        "primary_external_attempts",
        "maximum_external_attempts",
        "transport_retry_count",
        "approved",
    }
    if (
        not isinstance(value, dict)
        or source != _pretty_bytes(value)
        or not required <= set(value)
        or value.get("schema_version")
        != "adaptive-obligation-v2-retrieval-approval-v1"
        or value.get("stage") != "retrieval"
        or value.get("approved") is not True
        or not isinstance(value.get("preflight_sha256"), str)
        or not _SHA256_RE.fullmatch(str(value["preflight_sha256"]))
        or any(
            type(value.get(name)) is not int
            for name in (
                "planned_request_count",
                "primary_external_attempts",
                "maximum_external_attempts",
                "transport_retry_count",
            )
        )
        or value.get("transport_retry_count") != 0
    ):
        raise PermissionError("retrieval approval required")
    return value, _sha256(source)


def _capture_retrieval_preflight(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], bytes]:
    source = _capture_preflight_source(path)
    if _sha256(source) != expected_sha256:
        raise PermissionError("retrieval approval required")
    return _parse_preflight_source(source), source


def _verify_approval_against_preflight(
    approval: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    for name in (
        "planned_request_count",
        "primary_external_attempts",
        "maximum_external_attempts",
        "transport_retry_count",
    ):
        if approval.get(name) != receipt.get(name):
            raise PermissionError("retrieval approval required")


def _current_cache(
    receipt: Mapping[str, object], cache_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hits: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    observed: list[dict[str, object]] = []
    jobs = receipt["requests"]
    assert isinstance(jobs, list)
    for raw_job in jobs:
        job = dict(raw_job)
        cached = _load_verified_cache(cache_root, job)
        observed.append(_cache_status(cached, job))
        (hits if cached is not None else misses).append(
            {"job": job, "cache": cached} if cached is not None else {"job": job}
        )
    if observed != receipt.get("cache_audit"):
        raise ValueError("retrieval cache changed after frozen preflight")
    return hits, misses


def _claim_output_root(output: Path) -> None:
    output = Path(output)
    parent_fd = _open_directory_no_symlinks(output.parent)
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        raise FileExistsError(f"create-only retrieval output exists: {output}") from None
    finally:
        os.close(parent_fd)
    for name in ("attempts", "raw", "metadata", "candidates", "outcomes", "cache_hits"):
        (output / name).mkdir(mode=0o700)
    _fsync_directory(output)


def _store_cache(
    cache_root: Path,
    job: Mapping[str, object],
    raw: bytes,
    candidates: Sequence[Mapping[str, object]],
) -> None:
    raw_path, candidates_path, manifest_path = _cache_paths(
        cache_root, str(job["request_key"])
    )
    candidate_source = _pretty_bytes(list(candidates))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "request_key": job["request_key"],
        "request_identity": job["request_identity"],
        "query_text": job["query_text"],
        "raw_sha256": _sha256(raw),
        "candidates_sha256": _sha256(candidate_source),
        "candidate_count": RETRIEVAL_HITS,
    }
    _exclusive_bytes(raw_path, raw)
    _exclusive_bytes(candidates_path, candidate_source)
    _exclusive_json(manifest_path, manifest)


def _failure(
    output: Path,
    job: Mapping[str, object],
    *,
    failure_type: str,
    message: str,
    raw_sha256: str | None,
) -> None:
    _exclusive_json(
        output / "outcomes" / f"{job['request_key']}.json",
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": job["request_key"],
            "status": "failure",
            "failure_type": failure_type,
            "message": message,
            "raw_sha256": raw_sha256,
        },
    )


class RateLimitedO1Transport:
    """Tracked requests-ratelimiter transport with no hidden retry path."""

    one_shot_no_retry = True
    request_start_interval_seconds = REQUEST_START_INTERVAL_SECONDS
    timeout_seconds = TIMEOUT_SECONDS

    def __init__(
        self,
        allowed_jobs: Sequence[Mapping[str, object]],
        *,
        api_token: str | None,
        session: requests.Session | None = None,
    ) -> None:
        jobs = [dict(_verify_job(job)) for job in allowed_jobs]
        self._allowed = {str(job["request_key"]): job for job in jobs}
        if len(self._allowed) != len(jobs):
            raise ValueError("transport allowlist contains duplicate requests")
        endpoints = {str(job["endpoint"]) for job in jobs}
        if len(endpoints) != 1:
            raise ValueError("transport allowlist must use one endpoint")
        self.endpoint = next(iter(endpoints))
        self.api_token = api_token
        config = RemotePyseriniConfig(
            index_url=self.endpoint,
            api_token=api_token,
            hits=RETRIEVAL_HITS,
            queries=(),
            min_interval_seconds=REQUEST_START_INTERVAL_SECONDS,
            burst=1,
            limiter_state_path=LIMITER_STATE_PATH,
        )
        self.session = session if session is not None else rate_limited_session(config)

    def __call__(self, job: Mapping[str, object]) -> RawTransportResponse:
        allowed = self._allowed.get(str(job.get("request_key")))
        if allowed != dict(job):
            raise ValueError("request is outside the frozen transport allowlist")
        started = time.monotonic()
        response = self.session.get(
            self.endpoint,
            params={"query": job["query_text"], "hits": str(RETRIEVAL_HITS)},
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_token}"}
                    if self.api_token is not None
                    else {}
                ),
            },
            timeout=TIMEOUT_SECONDS,
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


def execute_retrieval(
    *,
    preflight_dir: Path,
    approval_path: Path,
    transport_factory: Callable[[Sequence[Mapping[str, object]]], RetrievalTransport],
    output_dir: Path | None = None,
    cache_root: Path | None = None,
) -> dict[str, object]:
    """Execute an approved immutable request inventory exactly once."""

    approval, approval_sha256 = _capture_retrieval_approval(Path(approval_path))
    receipt, receipt_source = _capture_retrieval_preflight(
        Path(preflight_dir), expected_sha256=str(approval["preflight_sha256"])
    )
    _verify_approval_against_preflight(approval, receipt)
    root = Path(cache_root or str(receipt["cache_root"])).absolute()
    cache_hits, cache_misses = _current_cache(receipt, root)
    miss_jobs = [row["job"] for row in cache_misses]
    transport: RetrievalTransport | None = None
    if miss_jobs:
        transport = transport_factory(miss_jobs)
        if getattr(transport, "one_shot_no_retry", None) is not True:
            raise ValueError("retrieval transport must be one-shot no-retry")
        if (
            getattr(transport, "request_start_interval_seconds", None)
            != REQUEST_START_INTERVAL_SECONDS
            or getattr(transport, "timeout_seconds", None) != TIMEOUT_SECONDS
        ):
            raise ValueError("retrieval transport limiter or timeout differs")
    output = Path(output_dir or (Path(preflight_dir).parent / "retrieval"))
    _claim_output_root(output)

    combined: list[dict[str, object]] = []
    external_attempts = 0
    cached_by_key = {
        str(row["job"]["request_key"]): row for row in cache_hits
    }
    jobs = receipt["requests"]
    assert isinstance(jobs, list)
    for order, raw_job in enumerate(jobs):
        job = dict(raw_job)
        key = str(job["request_key"])
        cached_row = cached_by_key.get(key)
        if cached_row is not None:
            cached = cached_row["cache"]
            assert isinstance(cached, Mapping)
            raw = cached["raw"]
            candidates = cached["candidates"]
            assert isinstance(raw, bytes) and isinstance(candidates, tuple)
            _exclusive_json(
                output / "cache_hits" / f"{key}.json",
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": key,
                    "request_identity": job["request_identity"],
                    "raw_sha256": cached["raw_sha256"],
                    "candidate_count": RETRIEVAL_HITS,
                    "external_attempts": 0,
                },
            )
            _exclusive_bytes(output / "raw" / f"{key}.body", raw)
            _exclusive_json(output / "candidates" / f"{key}.json", list(candidates))
            _exclusive_json(
                output / "outcomes" / f"{key}.json",
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": key,
                    "status": "cache_hit",
                    "candidate_count": RETRIEVAL_HITS,
                    "raw_sha256": cached["raw_sha256"],
                },
            )
        else:
            assert transport is not None
            _exclusive_json(
                output / "attempts" / f"{key}.json",
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": key,
                    "request_identity": job["request_identity"],
                    "query_text": job["query_text"],
                    "attempt_ordinal": 1,
                    "transport_retry_count": 0,
                    "manifest_order": order,
                },
            )
            external_attempts += 1
            try:
                response = transport(job)
            except Exception as exc:
                _failure(
                    output,
                    job,
                    failure_type="transport_exception",
                    message=f"{type(exc).__name__}: {exc}",
                    raw_sha256=None,
                )
                raise
            raw = response.body
            _exclusive_bytes(output / "raw" / f"{key}.body", raw)
            raw_sha256 = _sha256(raw)
            _exclusive_json(
                output / "metadata" / f"{key}.json",
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": key,
                    "http_status": response.status,
                    "headers": dict(response.headers),
                    "elapsed_seconds": float(response.elapsed_seconds),
                    "raw_sha256": raw_sha256,
                },
            )
            if response.status != 200:
                message = f"HTTP status {response.status} is not successful"
                _failure(
                    output,
                    job,
                    failure_type="http_error",
                    message=message,
                    raw_sha256=raw_sha256,
                )
                raise ValueError(message)
            try:
                candidates = _normalize_response(raw)
            except ValueError as exc:
                _failure(
                    output,
                    job,
                    failure_type="response_validation_error",
                    message=str(exc),
                    raw_sha256=raw_sha256,
                )
                raise
            _store_cache(root, job, raw, candidates)
            _exclusive_json(output / "candidates" / f"{key}.json", list(candidates))
            _exclusive_json(
                output / "outcomes" / f"{key}.json",
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "request_key": key,
                    "status": "success",
                    "candidate_count": RETRIEVAL_HITS,
                    "raw_sha256": raw_sha256,
                },
            )
        for candidate in candidates:
            combined.append(
                {
                    "request_key": key,
                    "topic_id": job["topic_id"],
                    "parent_id": job["parent_id"],
                    "accepted_o1_id": job["accepted_o1_id"],
                    **candidate,
                }
            )

    _exclusive_bytes(
        output / "candidates.jsonl",
        b"".join(_canonical_bytes(row) + b"\n" for row in combined),
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "preflight_sha256": _sha256(receipt_source),
        "approval_sha256": approval_sha256,
        "planned_request_count": len(jobs),
        "cache_hits": len(cache_hits),
        "external_attempts": external_attempts,
        "candidate_rows": len(combined),
        "transport_retry_count": 0,
        "qrels_opened": False,
    }
    _exclusive_json(output / "summary.json", summary)
    return summary


__all__ = [
    "INDEX_ID",
    "MAX_ACCEPTED_O1_PER_TOPIC",
    "MAX_RETRIEVAL_REQUESTS",
    "RATE_LIMITER_IDENTITY",
    "REQUEST_START_INTERVAL_SECONDS",
    "RETRIEVAL_HITS",
    "RETRIEVER_VERSION",
    "RateLimitedO1Transport",
    "TIMEOUT_SECONDS",
    "TRANSPORT_RETRY_COUNT",
    "audit_retrieval_cache",
    "build_retrieval_jobs",
    "execute_retrieval",
    "render_o1_bm25_query",
    "verify_retrieval_preflight",
]

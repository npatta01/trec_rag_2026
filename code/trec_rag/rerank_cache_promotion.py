"""Validate and transactionally promote remotely generated reranker caches.

The score builder intentionally uses the same schema-v2 artifact and global
content-cache interfaces on local and remote hardware.  This module is the
trust boundary between a downloaded staging directory and the shared local
cache: it reconciles the download against the local retrieval inputs, validates
the JSONL artifacts plus sealed SQLite score databases, and logically imports
scores before publishing a manifest-last promotion receipt.

Source databases are never copied as files.  Identical promotion attempts are
serialized, and bounded logical imports make retries safe after a process-local
failure, including ``KeyboardInterrupt``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.pipeline import pipeline_cache_dir
from trec_rag.pipeline_config import PipelineConfig, load_pipeline_config
from trec_rag.repo_env import load_repo_env
from trec_rag.rerank_score_cache import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_INDEX_URL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SCORE_REPRESENTATION,
    GlobalScoreCache,
    ScoreCacheContext,
    _queries_by_topic,
    _topic_candidates,
    _topics,
    build_arg_parser as build_score_cache_arg_parser,
    score_cache_key_from_hashes,
)


RAG25_DEV_TOPIC_IDS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "144",
    "161",
    "200",
    "213",
    "219",
    "224",
    "225",
    "233",
    "273",
    "300",
    "407",
    "477",
    "499",
    "515",
    "707",
    "897",
)

# Independently observed by the score-builder dry run for the 1,000-document
# RAG25 development candidate pools with cm=3500 and overlap=350.
RAG25_WINDOW_ROWS_PER_TOPIC = {
    "14": 6517,
    "31": 6071,
    "37": 10427,
    "58": 9538,
    "72": 7030,
    "84": 9585,
    "144": 12195,
    "161": 14425,
    "200": 6925,
    "213": 12664,
    "219": 12797,
    "224": 8758,
    "225": 5086,
    "233": 8103,
    "273": 15635,
    "300": 21688,
    "407": 8276,
    "477": 15736,
    "499": 5645,
    "515": 11533,
    "707": 9133,
    "897": 9389,
}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SQLITE_HEADER = b"SQLite format 3\x00"
_SCORE_CACHE_SCHEMA_VERSION = "score-cache-v2"
_FALSE_LEGACY_INPUT_POLICIES = frozenset(
    {"trec_rag_raw_v2", "extractive_sentence_pair_v1"}
)
_EFFECTIVE_INPUT_POLICY = "trec_rag_whitespace_v1"
_SCORE_CACHE_OBJECTS = {
    ("table", "cache_meta"),
    ("table", "scores"),
    ("table", "claims"),
    ("table", "imports"),
    ("index", "claims_expiry_idx"),
}
_SCORE_CACHE_OBJECT_SQL = {
    ("table", "cache_meta"): "CREATE TABLE cache_meta ( key TEXT PRIMARY KEY, value TEXT NOT NULL ) STRICT",
    ("table", "scores"): "CREATE TABLE scores ( key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32), query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32), text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32), score REAL NOT NULL CHECK(typeof(score) = 'real' AND score = score AND abs(score) <= 1.7976931348623157e+308) ) STRICT",
    ("table", "claims"): "CREATE TABLE claims ( key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32), query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32), text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32), owner TEXT NOT NULL, token BLOB NOT NULL CHECK(length(token) = 16), claimed_at REAL NOT NULL, lease_expires_at REAL NOT NULL ) STRICT",
    ("table", "imports"): "CREATE TABLE imports ( source_sha256 BLOB PRIMARY KEY CHECK(length(source_sha256) = 32), source_path TEXT NOT NULL, source_row_count INTEGER NOT NULL CHECK(source_row_count >= 0), inserted_count INTEGER NOT NULL CHECK(inserted_count >= 0), logical_digest BLOB NOT NULL CHECK(length(logical_digest) = 32), authorization_sha256 BLOB CHECK(authorization_sha256 IS NULL OR length(authorization_sha256) = 32) ) STRICT",
    ("index", "claims_expiry_idx"): "CREATE INDEX claims_expiry_idx ON claims(lease_expires_at)",
}


class CacheBundleValidationError(ValueError):
    """Raised when a staged cache bundle is unsafe to promote."""


@dataclass(frozen=True)
class ExpectedDocumentIdentity:
    rank: int
    query_sha256: str
    text_sha256: str


@dataclass(frozen=True)
class ExpectedWindowIdentity:
    chunk_count: int
    chunk_id: str
    start_char: int
    end_char: int
    query_sha256: str
    document_text_sha256: str
    text_sha256: str


@dataclass(frozen=True)
class BundleExpectations:
    """Expected scope and, when derived locally, exact input identities."""

    topic_ids: tuple[str, ...] = RAG25_DEV_TOPIC_IDS
    documents_per_topic: int = 1000
    window_rows_per_topic: Mapping[str, int] | None = None
    documents: Mapping[tuple[str, str], ExpectedDocumentIdentity] | None = None
    windows: Mapping[tuple[str, str, int], ExpectedWindowIdentity] | None = None
    document_context: ScoreCacheContext | None = None
    window_context: ScoreCacheContext | None = None


@dataclass(frozen=True)
class CacheBundlePaths:
    """Artifact files plus either a schema-v2 cache root or explicit cache files."""

    document_artifact: Path
    window_artifact: Path
    score_cache_root: Path | None = None
    document_score_cache: Path | None = None
    window_score_cache: Path | None = None


@dataclass(frozen=True)
class FileValidation:
    path: Path
    sha256: str
    row_count: int
    unique_key_count: int
    logical_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "path": str(self.path),
            "sha256": self.sha256,
            "row_count": self.row_count,
            "unique_key_count": self.unique_key_count,
        }
        if self.logical_digest is not None:
            payload["logical_digest"] = self.logical_digest
        return payload


@dataclass(frozen=True)
class RuntimeStatusValidation:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class BundleValidation:
    document_artifact: FileValidation
    window_artifact: FileValidation
    document_score_cache: FileValidation
    window_score_cache: FileValidation
    document_context: ScoreCacheContext
    window_context: ScoreCacheContext
    topic_document_counts: Mapping[str, int]
    topic_window_counts: Mapping[str, int]
    cache_extra_keys: Mapping[str, int]
    reconciled_to_local_inputs: bool
    matched_selected_context: bool
    runtime_status: RuntimeStatusValidation | None

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "reconciled_to_local_inputs": self.reconciled_to_local_inputs,
            "matched_selected_context": self.matched_selected_context,
            "files": {
                "document_artifact": self.document_artifact.to_dict(),
                "window_artifact": self.window_artifact.to_dict(),
                "document_score_cache": self.document_score_cache.to_dict(),
                "window_score_cache": self.window_score_cache.to_dict(),
            },
            "contexts": {
                "document": asdict(self.document_context),
                "window": asdict(self.window_context),
            },
            "topic_document_counts": dict(self.topic_document_counts),
            "topic_window_counts": dict(self.topic_window_counts),
            "cache_extra_keys": dict(self.cache_extra_keys),
            "runtime_status": (
                self.runtime_status.to_dict()
                if self.runtime_status is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PromotionResult:
    validation: BundleValidation
    archive_dir: Path
    destinations: Mapping[str, Path]
    backups: Mapping[str, Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": True,
            "archive_dir": str(self.archive_dir),
            "destinations": {
                key: str(value) for key, value in self.destinations.items()
            },
            "backups": {key: str(value) for key, value in self.backups.items()},
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class _DocumentRecord:
    rank: int
    query_sha256: str
    text_sha256: str


@dataclass(frozen=True)
class _CacheReference:
    query_sha256: str
    text_sha256: str
    score: float


@dataclass
class _WindowCoverage:
    chunk_count: int
    indices: set[int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise CacheBundleValidationError(f"missing staged JSONL file: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheBundleValidationError(
                    f"{path}:{line_number}: invalid JSONL row"
                ) from exc
            if not isinstance(row, dict):
                raise CacheBundleValidationError(
                    f"{path}:{line_number}: JSONL row must be an object"
                )
            yield line_number, row


def _first_row(path: Path) -> dict[str, Any]:
    try:
        _, row = next(_jsonl_rows(path))
    except StopIteration as exc:
        raise CacheBundleValidationError(f"empty staged JSONL file: {path}") from exc
    return row


def _required_string(
    row: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CacheBundleValidationError(
            f"{path}:{line_number}: {field} must be a non-empty string"
        )
    return value


def _required_int(
    row: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CacheBundleValidationError(
            f"{path}:{line_number}: {field} must be an integer"
        )
    return value


def _required_sha256(
    row: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> str:
    value = _required_string(row, field, path=path, line_number=line_number)
    if not _SHA256_PATTERN.fullmatch(value):
        raise CacheBundleValidationError(
            f"{path}:{line_number}: {field} must be a lowercase SHA-256 digest"
        )
    return value


def _finite_score(row: Mapping[str, Any], *, path: Path, line_number: int) -> float:
    value = row.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CacheBundleValidationError(
            f"{path}:{line_number}: score must be a JSON number"
        )
    score = float(value)
    if not math.isfinite(score):
        raise CacheBundleValidationError(
            f"{path}:{line_number}: raw-logit score must be finite"
        )
    return score


def _context_from_artifact(path: Path, *, kind: str) -> ScoreCacheContext:
    row = _first_row(path)
    line_number = 1
    if row.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CacheBundleValidationError(
            f"{path}:{line_number}: artifact_schema_version must be "
            f"{ARTIFACT_SCHEMA_VERSION}"
        )
    requested_max_length = _required_int(
        row, "requested_max_length", path=path, line_number=line_number
    )
    pair_buffer_tokens = _required_int(
        row, "pair_buffer_tokens", path=path, line_number=line_number
    )
    context = ScoreCacheContext(
        backend=_required_string(row, "backend", path=path, line_number=line_number),
        backend_version=_required_string(
            row, "backend_version", path=path, line_number=line_number
        ),
        model=_required_string(row, "model", path=path, line_number=line_number),
        model_revision=_required_string(
            row, "model_revision", path=path, line_number=line_number
        ),
        score_representation=_required_string(
            row, "score_representation", path=path, line_number=line_number
        ),
        inference_dtype=_required_string(
            row, "inference_dtype", path=path, line_number=line_number
        ),
        input_policy=_required_string(
            row, "input_policy", path=path, line_number=line_number
        ),
        max_length=_required_int(row, "max_length", path=path, line_number=line_number),
        score_kind=_required_string(
            row, "score_kind", path=path, line_number=line_number
        ),
        requested_max_length=requested_max_length,
        pair_buffer_tokens=pair_buffer_tokens,
        chunk_max_characters=(
            _required_int(
                row, "chunk_max_characters", path=path, line_number=line_number
            )
            if kind == "window"
            else None
        ),
        chunk_overlap_characters=(
            _required_int(
                row, "chunk_overlap_characters", path=path, line_number=line_number
            )
            if kind == "window"
            else None
        ),
    )
    if context.score_representation != "raw_logits":
        raise CacheBundleValidationError(
            f"{path}: score_representation must be raw_logits"
        )
    if context.max_length <= 0 or requested_max_length <= 0 or pair_buffer_tokens < 0:
        raise CacheBundleValidationError(f"{path}: invalid max-length metadata")
    if requested_max_length - context.max_length != pair_buffer_tokens:
        raise CacheBundleValidationError(
            f"{path}: requested_max_length - max_length must equal pair_buffer_tokens"
        )
    if kind == "document" and context.score_kind == "window":
        raise CacheBundleValidationError(
            f"{path}: document artifact uses window score_kind"
        )
    if kind == "window" and context.score_kind != "window":
        raise CacheBundleValidationError(
            f"{path}: window artifact score_kind must be window"
        )
    return context


def _validate_artifact_metadata(
    row: Mapping[str, Any],
    context: ScoreCacheContext,
    *,
    path: Path,
    line_number: int,
) -> None:
    for field, expected in context.artifact_metadata.items():
        if row.get(field) != expected:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: {field} must be {expected!r}; "
                f"found {row.get(field)!r}"
            )


def _cache_key_from_hashes(
    context: ScoreCacheContext, *, query_sha256: str, text_sha256: str
) -> str:
    return score_cache_key_from_hashes(
        context,
        query_sha256=query_sha256,
        text_sha256=text_sha256,
    )


def _record_cache_reference(
    references: dict[str, _CacheReference],
    cache_key: str,
    reference: _CacheReference,
    *,
    label: str,
) -> None:
    previous = references.get(cache_key)
    if previous is not None and previous != reference:
        raise CacheBundleValidationError(f"conflicting duplicate {label}: {cache_key}")
    references[cache_key] = reference


def _validate_expected_topics(expectations: BundleExpectations) -> set[str]:
    expected_topics = set(expectations.topic_ids)
    if not expected_topics or len(expected_topics) != len(expectations.topic_ids):
        raise CacheBundleValidationError(
            "expected topic IDs must be non-empty and unique"
        )
    if expectations.documents_per_topic <= 0:
        raise CacheBundleValidationError("documents_per_topic must be positive")
    if expectations.window_rows_per_topic is not None:
        count_topics = set(expectations.window_rows_per_topic)
        if count_topics != expected_topics:
            raise CacheBundleValidationError(
                "expected window-count topics do not exactly match expected topic IDs"
            )
    return expected_topics


def _require_expected_context(
    actual: ScoreCacheContext,
    expected: ScoreCacheContext | None,
    *,
    label: str,
    allow_legacy_policy_rebinding: bool = False,
) -> bool:
    if expected is None:
        return False
    if actual == expected:
        return True
    if allow_legacy_policy_rebinding and _is_legacy_policy_rebinding(
        actual, expected
    ):
        return True
    actual_fields = asdict(actual)
    expected_fields = asdict(expected)
    mismatches = [
        f"{field}={actual_fields[field]!r} (config: {expected_value!r})"
        for field, expected_value in expected_fields.items()
        if actual_fields[field] != expected_value
    ]
    raise CacheBundleValidationError(
        f"{label} artifact context differs from selected config: "
        + ", ".join(mismatches)
    )


def _is_legacy_policy_rebinding(
    legacy_context: ScoreCacheContext, target_context: ScoreCacheContext
) -> bool:
    if legacy_context.input_policy not in _FALSE_LEGACY_INPUT_POLICIES:
        return False
    if target_context.input_policy != _EFFECTIVE_INPUT_POLICY:
        return False
    return all(
        legacy_value == getattr(target_context, field)
        for field, legacy_value in asdict(legacy_context).items()
        if field != "input_policy"
    )


def _validate_document_artifact(
    path: Path,
    context: ScoreCacheContext,
    expectations: BundleExpectations,
) -> tuple[
    FileValidation,
    dict[tuple[str, str], _DocumentRecord],
    dict[str, _CacheReference],
    dict[str, int],
]:
    expected_topics = _validate_expected_topics(expectations)
    documents: dict[tuple[str, str], _DocumentRecord] = {}
    cache_references: dict[str, _CacheReference] = {}
    topic_counts: Counter[str] = Counter()
    ranks_by_topic: dict[str, set[int]] = defaultdict(set)
    row_count = 0

    for line_number, row in _jsonl_rows(path):
        row_count += 1
        _validate_artifact_metadata(row, context, path=path, line_number=line_number)
        topic_id = _required_string(row, "topic_id", path=path, line_number=line_number)
        docid = _required_string(row, "docid", path=path, line_number=line_number)
        if topic_id not in expected_topics:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: unexpected topic_id {topic_id}"
            )
        rank = _required_int(row, "rank", path=path, line_number=line_number)
        query_hash = _required_sha256(
            row, "query_sha256", path=path, line_number=line_number
        )
        text_hash = _required_sha256(
            row, "text_sha256", path=path, line_number=line_number
        )
        cache_key = _required_sha256(
            row, "score_cache_key", path=path, line_number=line_number
        )
        score = _finite_score(row, path=path, line_number=line_number)
        expected_cache_key = _cache_key_from_hashes(
            context, query_sha256=query_hash, text_sha256=text_hash
        )
        if cache_key != expected_cache_key:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: document score_cache_key does not match metadata/hashes"
            )
        key = (topic_id, docid)
        if key in documents:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: duplicate document artifact row for {key}"
            )
        record = _DocumentRecord(rank, query_hash, text_hash)
        if expectations.documents is not None:
            expected = expectations.documents.get(key)
            if expected is None or expected != ExpectedDocumentIdentity(
                rank, query_hash, text_hash
            ):
                raise CacheBundleValidationError(
                    f"{path}:{line_number}: document identity differs from local input for {key}"
                )
        documents[key] = record
        topic_counts[topic_id] += 1
        ranks_by_topic[topic_id].add(rank)
        _record_cache_reference(
            cache_references,
            cache_key,
            _CacheReference(query_hash, text_hash, score),
            label="document artifact cache key",
        )

    if set(topic_counts) != expected_topics:
        missing = sorted(expected_topics - set(topic_counts))
        raise CacheBundleValidationError(
            f"document artifact is missing topics: {missing}"
        )
    expected_ranks = set(range(1, expectations.documents_per_topic + 1))
    for topic_id in expectations.topic_ids:
        if topic_counts[topic_id] != expectations.documents_per_topic:
            raise CacheBundleValidationError(
                f"topic {topic_id}: expected {expectations.documents_per_topic} documents; "
                f"found {topic_counts[topic_id]}"
            )
        if ranks_by_topic[topic_id] != expected_ranks:
            raise CacheBundleValidationError(
                f"topic {topic_id}: document ranks must exactly cover "
                f"1..{expectations.documents_per_topic}"
            )
    if expectations.documents is not None and set(expectations.documents) != set(
        documents
    ):
        raise CacheBundleValidationError(
            "document artifact keys do not exactly match locally derived inputs"
        )
    queries_by_topic: dict[str, set[str]] = defaultdict(set)
    for (topic_id, _), record in documents.items():
        queries_by_topic[topic_id].add(record.query_sha256)
    if any(len(hashes) != 1 for hashes in queries_by_topic.values()):
        raise CacheBundleValidationError("each topic must use exactly one query hash")

    return (
        FileValidation(path, _sha256_file(path), row_count, len(cache_references)),
        documents,
        cache_references,
        dict(topic_counts),
    )


def _validate_window_artifact(
    path: Path,
    context: ScoreCacheContext,
    expectations: BundleExpectations,
    documents: Mapping[tuple[str, str], _DocumentRecord],
) -> tuple[FileValidation, dict[str, _CacheReference], dict[str, int]]:
    coverage: dict[tuple[str, str], _WindowCoverage] = {}
    cache_references: dict[str, _CacheReference] = {}
    topic_counts: Counter[str] = Counter()
    seen_windows: set[tuple[str, str, int]] = set()
    row_count = 0

    for line_number, row in _jsonl_rows(path):
        row_count += 1
        _validate_artifact_metadata(row, context, path=path, line_number=line_number)
        topic_id = _required_string(row, "topic_id", path=path, line_number=line_number)
        docid = _required_string(row, "docid", path=path, line_number=line_number)
        doc_key = (topic_id, docid)
        document = documents.get(doc_key)
        if document is None:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: window has no matching document artifact: {doc_key}"
            )
        rank = _required_int(row, "rank", path=path, line_number=line_number)
        chunk_index = _required_int(
            row, "chunk_index", path=path, line_number=line_number
        )
        chunk_count = _required_int(
            row, "chunk_count", path=path, line_number=line_number
        )
        chunk_id = _required_string(row, "chunk_id", path=path, line_number=line_number)
        start_char = _required_int(
            row, "start_char", path=path, line_number=line_number
        )
        end_char = _required_int(row, "end_char", path=path, line_number=line_number)
        query_hash = _required_sha256(
            row, "query_sha256", path=path, line_number=line_number
        )
        document_hash = _required_sha256(
            row, "document_text_sha256", path=path, line_number=line_number
        )
        text_hash = _required_sha256(
            row, "text_sha256", path=path, line_number=line_number
        )
        cache_key = _required_sha256(
            row, "score_cache_key", path=path, line_number=line_number
        )
        score = _finite_score(row, path=path, line_number=line_number)

        if rank != document.rank or query_hash != document.query_sha256:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: window rank/query hash differs from document artifact"
            )
        if document_hash != document.text_sha256:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: window document hash differs from document artifact"
            )
        if chunk_count <= 0 or not 0 <= chunk_index < chunk_count:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: invalid chunk_index/chunk_count"
            )
        if start_char < 0 or end_char <= start_char:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: invalid window character offsets"
            )
        if chunk_id != f"{docid}:{chunk_index:04d}":
            raise CacheBundleValidationError(
                f"{path}:{line_number}: chunk_id does not match docid/chunk_index"
            )
        expected_cache_key = _cache_key_from_hashes(
            context, query_sha256=query_hash, text_sha256=text_hash
        )
        if cache_key != expected_cache_key:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: window score_cache_key does not match metadata/hashes"
            )
        window_key = (topic_id, docid, chunk_index)
        if window_key in seen_windows:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: duplicate window artifact row for {window_key}"
            )
        if expectations.windows is not None:
            expected = expectations.windows.get(window_key)
            observed = ExpectedWindowIdentity(
                chunk_count,
                chunk_id,
                start_char,
                end_char,
                query_hash,
                document_hash,
                text_hash,
            )
            if expected is None or expected != observed:
                raise CacheBundleValidationError(
                    f"{path}:{line_number}: window identity differs from local input for "
                    f"{window_key}"
                )
        seen_windows.add(window_key)
        topic_counts[topic_id] += 1
        doc_coverage = coverage.setdefault(
            doc_key, _WindowCoverage(chunk_count=chunk_count, indices=set())
        )
        if doc_coverage.chunk_count != chunk_count:
            raise CacheBundleValidationError(
                f"{path}:{line_number}: inconsistent chunk_count for {doc_key}"
            )
        doc_coverage.indices.add(chunk_index)
        _record_cache_reference(
            cache_references,
            cache_key,
            _CacheReference(query_hash, text_hash, score),
            label="window artifact cache key",
        )

    if set(coverage) != set(documents):
        missing = sorted(set(documents) - set(coverage))[:5]
        raise CacheBundleValidationError(
            f"window artifact does not cover every document; first missing keys: {missing}"
        )
    derived_topic_counts: Counter[str] = Counter()
    for (topic_id, docid), doc_coverage in coverage.items():
        if len(doc_coverage.indices) != doc_coverage.chunk_count:
            raise CacheBundleValidationError(
                f"incomplete window coverage for {(topic_id, docid)}: expected "
                f"{doc_coverage.chunk_count}, found {len(doc_coverage.indices)}"
            )
        derived_topic_counts[topic_id] += doc_coverage.chunk_count
    if dict(topic_counts) != dict(derived_topic_counts):
        raise CacheBundleValidationError(
            "window row totals do not reconcile with per-document chunk_count metadata"
        )
    if expectations.window_rows_per_topic is not None:
        for topic_id in expectations.topic_ids:
            expected = expectations.window_rows_per_topic[topic_id]
            if topic_counts[topic_id] != expected:
                raise CacheBundleValidationError(
                    f"topic {topic_id}: expected {expected} window rows; "
                    f"found {topic_counts[topic_id]}"
                )
    if expectations.windows is not None and set(expectations.windows) != seen_windows:
        raise CacheBundleValidationError(
            "window artifact keys do not exactly match locally derived inputs"
        )
    return (
        FileValidation(path, _sha256_file(path), row_count, len(cache_references)),
        cache_references,
        dict(topic_counts),
    )


def _resolve_cache_paths(
    paths: CacheBundlePaths,
    document_context: ScoreCacheContext,
    window_context: ScoreCacheContext,
) -> tuple[Path, Path]:
    document_path = paths.document_score_cache
    window_path = paths.window_score_cache
    if document_path is None:
        if paths.score_cache_root is None:
            raise CacheBundleValidationError(
                "provide score_cache_root or an explicit document_score_cache path"
            )
        document_path = paths.score_cache_root.joinpath(*document_context.path_parts)
    if window_path is None:
        if paths.score_cache_root is None:
            raise CacheBundleValidationError(
                "provide score_cache_root or an explicit window_score_cache path"
            )
        window_path = paths.score_cache_root.joinpath(*window_context.path_parts)
    return document_path, window_path


def _normalize_schema_sql(value: str) -> str:
    normalized = value.lower().rstrip(";")
    for character in "(),=":
        normalized = normalized.replace(character, f" {character} ")
    return " ".join(normalized.split())


def _score_cache_sidecars(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    )


def _assert_sealed_score_cache(path: Path) -> None:
    sidecars = [
        sidecar
        for sidecar in _score_cache_sidecars(path)
        if sidecar.exists() or sidecar.is_symlink()
    ]
    if sidecars:
        raise CacheBundleValidationError(
            f"{path}: sealed score cache has an active or uncheckpointed WAL sidecar: "
            + ", ".join(str(sidecar) for sidecar in sidecars)
        )


def _verify_sealed_score_cache_identity(path: Path, expected_sha256: str) -> None:
    _assert_sealed_score_cache(path)
    if _sha256_file(path) != expected_sha256:
        raise CacheBundleValidationError(f"sealed score cache changed: {path}")
    _assert_sealed_score_cache(path)


def _open_score_cache(
    path: Path,
    context: ScoreCacheContext,
    *,
    sealed: bool,
) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise CacheBundleValidationError(
            f"{path}: score cache must be a regular SQLite database"
        )
    if sealed:
        _assert_sealed_score_cache(path)
    connection: sqlite3.Connection | None = None
    try:
        with path.open("rb") as source:
            header = source.read(20)
            if header[: len(_SQLITE_HEADER)] != _SQLITE_HEADER:
                raise CacheBundleValidationError(
                    f"{path}: score cache is not a SQLite database"
                )
            if sealed and header[18:20] != b"\x02\x02":
                raise CacheBundleValidationError(
                    f"{path}: score cache is not a sealed WAL database"
                )
        uri = (
            f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
            + ("&immutable=1" if sealed else "")
        )
        connection = sqlite3.connect(uri, uri=True, timeout=5, isolation_level=None)
        connection.execute("PRAGMA query_only = ON")
        journal_mode = str(
            connection.execute("PRAGMA journal_mode").fetchone()[0]
        ).lower()
        if not sealed and journal_mode != "wal":
            raise CacheBundleValidationError(
                f"{path}: score cache must use WAL journal mode; found {journal_mode!r}"
            )
        if sealed and str(
            connection.execute("PRAGMA locking_mode").fetchone()[0]
        ).lower() != "normal":
            raise CacheBundleValidationError(
                f"{path}: score cache has an active exclusive lock"
            )
        schema_rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        actual_objects = {(str(row[0]), str(row[1])) for row in schema_rows}
        if actual_objects != _SCORE_CACHE_OBJECTS:
            raise CacheBundleValidationError(
                f"{path}: score-cache-v2 schema objects do not match exactly; "
                f"found {sorted(actual_objects)!r}"
            )
        for row in schema_rows:
            identity = (str(row[0]), str(row[1]))
            if _normalize_schema_sql(str(row[2])) != _normalize_schema_sql(
                _SCORE_CACHE_OBJECT_SQL[identity]
            ):
                raise CacheBundleValidationError(
                    f"{path}: score-cache-v2 schema definition mismatch for {identity!r}"
                )
        metadata_rows = connection.execute(
            "SELECT key, value FROM cache_meta ORDER BY key"
        ).fetchall()
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata_rows
        ):
            raise CacheBundleValidationError(f"{path}: score-cache-v2 metadata is malformed")
        metadata = dict(metadata_rows)
        expected_metadata = {
            "schema_version": _SCORE_CACHE_SCHEMA_VERSION,
            "context_sha256": context.context_sha256,
            "context_json": context.context_json,
        }
        if metadata != expected_metadata or len(metadata_rows) != len(expected_metadata):
            raise CacheBundleValidationError(
                f"{path}: score-cache-v2 schema/context metadata mismatch"
            )
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity_rows != [("ok",)]:
            raise CacheBundleValidationError(
                f"{path}: SQLite integrity check failed: {integrity_rows!r}"
            )
        return connection
    except CacheBundleValidationError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error, ValueError) as exc:
        if connection is not None:
            connection.close()
        raise CacheBundleValidationError(
            f"{path}: invalid SQLite score cache: {exc}"
        ) from exc


def _score_cache_row(
    path: Path,
    context: ScoreCacheContext,
    row: Sequence[Any],
) -> tuple[str, str, str, float]:
    key, query_hash, text_hash, score, score_type = row
    if not all(
        isinstance(value, bytes) and len(value) == 32
        for value in (key, query_hash, text_hash)
    ):
        raise CacheBundleValidationError(
            f"{path}: score cache hashes must be 32-byte BLOBs"
        )
    if (
        score_type != "real"
        or isinstance(score, bool)
        or not isinstance(score, int | float)
    ):
        raise CacheBundleValidationError(
            f"{path}: score cache scores must be SQLite REAL values, not Boolean or other types"
        )
    value = float(score)
    if not math.isfinite(value):
        raise CacheBundleValidationError(f"{path}: score cache contains a nonfinite score")
    key_hex = bytes(key).hex()
    query_hex = bytes(query_hash).hex()
    text_hex = bytes(text_hash).hex()
    expected_key = _cache_key_from_hashes(
        context, query_sha256=query_hex, text_sha256=text_hex
    )
    if key_hex != expected_key:
        raise CacheBundleValidationError(
            f"{path}: score cache key does not match metadata/content hashes"
        )
    return key_hex, query_hex, text_hex, value


def _score_row_digest(
    key_hex: str, query_hex: str, text_hex: str, score: float
) -> bytes:
    return hashlib.sha256(
        bytes.fromhex(key_hex)
        + bytes.fromhex(query_hex)
        + bytes.fromhex(text_hex)
        + score.hex().encode("ascii")
    ).digest()


def _validate_score_cache(
    path: Path,
    context: ScoreCacheContext,
    artifact_references: Mapping[str, _CacheReference],
) -> tuple[FileValidation, int]:
    connection = _open_score_cache(path, context, sealed=True)
    covered_keys: set[str] = set()
    row_count = 0
    extra_count = 0
    logical_hasher = hashlib.sha256(bytes.fromhex(context.context_sha256))
    try:
        for row in connection.execute(
            "SELECT key_sha256, query_sha256, text_sha256, score, typeof(score) "
            "FROM scores ORDER BY key_sha256"
        ):
            cache_key, query_hash, text_hash, score = _score_cache_row(
                path, context, row
            )
            row_count += 1
            logical_hasher.update(
                _score_row_digest(cache_key, query_hash, text_hash, score)
            )
            artifact_reference = artifact_references.get(cache_key)
            if artifact_reference is None:
                extra_count += 1
                continue
            covered_keys.add(cache_key)
            if artifact_reference != _CacheReference(query_hash, text_hash, score):
                raise CacheBundleValidationError(
                    f"{path}: global cache conflicts with artifact key {cache_key}"
                )
        missing_count = len(artifact_references) - len(covered_keys)
        if missing_count:
            raise CacheBundleValidationError(
                f"{path}: global cache is missing {missing_count} artifact score keys"
            )
        return (
            FileValidation(
                path,
                _sha256_file(path),
                row_count,
                row_count,
                logical_digest=logical_hasher.hexdigest(),
            ),
            extra_count,
        )
    finally:
        connection.close()


def _runtime_count(
    value: Any,
    *,
    path: Path,
    field: str,
    expected: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CacheBundleValidationError(
            f"{path}: runtime status {field} must be an integer"
        )
    if value != expected:
        raise CacheBundleValidationError(
            f"{path}: runtime status {field}={value}; expected {expected}"
        )


def _runtime_sha256(
    value: Any,
    *,
    path: Path,
    field: str,
    expected: str,
) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise CacheBundleValidationError(
            f"{path}: runtime status {field} must be a lowercase SHA-256 digest"
        )
    if value != expected:
        raise CacheBundleValidationError(
            f"{path}: runtime status {field} does not match the staged artifact"
        )


def _validate_runtime_status(
    path: Path,
    *,
    document: FileValidation,
    window: FileValidation,
    topic_document_counts: Mapping[str, int],
    topic_window_counts: Mapping[str, int],
    document_context: ScoreCacheContext,
    window_context: ScoreCacheContext,
    expectations: BundleExpectations,
) -> RuntimeStatusValidation:
    if not path.is_file():
        raise CacheBundleValidationError(f"missing Modal runtime status: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CacheBundleValidationError(
            f"{path}: invalid runtime status JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise CacheBundleValidationError(
            f"{path}: runtime status must be a JSON object"
        )
    if payload.get("state") != "completed":
        raise CacheBundleValidationError(
            f"{path}: runtime status state must be 'completed'; "
            f"found {payload.get('state')!r}"
        )
    for field, summary in (
        ("document_rows", document),
        ("window_rows", window),
    ):
        if field not in payload:
            raise CacheBundleValidationError(
                f"{path}: runtime status is missing {field}"
            )
        _runtime_count(
            payload[field],
            path=path,
            field=field,
            expected=summary.row_count,
        )

    for label, summary, topic_counts in (
        ("document", document, topic_document_counts),
        ("window", window, topic_window_counts),
    ):
        digest_fields_found: list[str] = []
        validation_field = f"{label}_validation"
        nested = payload.get(validation_field)
        if nested is not None:
            if not isinstance(nested, dict):
                raise CacheBundleValidationError(
                    f"{path}: runtime status {validation_field} must be an object"
                )
            if "rows" in nested:
                _runtime_count(
                    nested["rows"],
                    path=path,
                    field=f"{validation_field}.rows",
                    expected=summary.row_count,
                )
            if "sha256" in nested:
                digest_fields_found.append(f"{validation_field}.sha256")
                _runtime_sha256(
                    nested["sha256"],
                    path=path,
                    field=f"{validation_field}.sha256",
                    expected=summary.sha256,
                )
            if "rows_by_topic" in nested:
                observed_counts = nested["rows_by_topic"]
                if not isinstance(observed_counts, dict) or observed_counts != dict(
                    topic_counts
                ):
                    raise CacheBundleValidationError(
                        f"{path}: runtime status {validation_field}.rows_by_topic "
                        "does not match the validated artifact"
                    )
        for sha_field in (f"{label}_sha256", f"{label}_artifact_sha256"):
            if sha_field in payload:
                digest_fields_found.append(sha_field)
                _runtime_sha256(
                    payload[sha_field],
                    path=path,
                    field=sha_field,
                    expected=summary.sha256,
                )
        if not digest_fields_found:
            raise CacheBundleValidationError(
                f"{path}: runtime status must include a {label} artifact SHA-256 "
                f"in {validation_field}.sha256 or a supported top-level field"
            )

    scoring_contract = payload.get("scoring_contract")
    if scoring_contract is not None:
        if not isinstance(scoring_contract, dict):
            raise CacheBundleValidationError(
                f"{path}: runtime status scoring_contract must be an object"
            )
        expected_contract = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "backend": document_context.backend,
            "backend_version": document_context.backend_version,
            "model": document_context.model,
            "model_revision": document_context.model_revision,
            "score_representation": document_context.score_representation,
            "inference_dtype": document_context.inference_dtype,
            "input_policy": document_context.input_policy,
            "candidate_limit": expectations.documents_per_topic,
            "topics": list(expectations.topic_ids),
            "document_max_length": document_context.requested_max_length,
            "document_pair_buffer_tokens": document_context.pair_buffer_tokens,
            "document_model_max_length": document_context.max_length,
            "window_max_length": window_context.max_length,
            "chunk_max_characters": window_context.chunk_max_characters,
            "chunk_overlap_characters": window_context.chunk_overlap_characters,
        }
        missing_contract_fields = [
            field for field in expected_contract if field not in scoring_contract
        ]
        if missing_contract_fields:
            raise CacheBundleValidationError(
                f"{path}: runtime scoring_contract is missing required fields: "
                + ", ".join(missing_contract_fields)
            )
        for field, expected in expected_contract.items():
            if scoring_contract[field] != expected:
                raise CacheBundleValidationError(
                    f"{path}: runtime scoring_contract.{field}="
                    f"{scoring_contract[field]!r}; expected {expected!r}"
                )
    input_sha256 = payload.get("input_sha256")
    if input_sha256 is not None and (
        not isinstance(input_sha256, str) or not _SHA256_PATTERN.fullmatch(input_sha256)
    ):
        raise CacheBundleValidationError(
            f"{path}: runtime status input_sha256 must be a lowercase SHA-256 digest"
        )
    return RuntimeStatusValidation(path, _sha256_file(path), payload)


def validate_cache_bundle(
    paths: CacheBundlePaths,
    *,
    expectations: BundleExpectations | None = None,
    runtime_status_path: Path | None = None,
    _allow_legacy_policy_rebinding: bool = False,
) -> BundleValidation:
    """Validate a staged document/window artifact and its schema-v2 caches."""

    expectations = expectations or BundleExpectations(
        window_rows_per_topic=RAG25_WINDOW_ROWS_PER_TOPIC
    )
    document_context = _context_from_artifact(paths.document_artifact, kind="document")
    window_context = _context_from_artifact(paths.window_artifact, kind="window")
    document_context_matched = _require_expected_context(
        document_context,
        expectations.document_context,
        label="document",
        allow_legacy_policy_rebinding=_allow_legacy_policy_rebinding,
    )
    window_context_matched = _require_expected_context(
        window_context,
        expectations.window_context,
        label="window",
        allow_legacy_policy_rebinding=_allow_legacy_policy_rebinding,
    )
    if (
        document_context.cache_identity_metadata
        != window_context.cache_identity_metadata
    ):
        raise CacheBundleValidationError(
            "document and window artifacts use different model/cache identity metadata"
        )
    (
        document_summary,
        documents,
        document_references,
        topic_document_counts,
    ) = _validate_document_artifact(
        paths.document_artifact, document_context, expectations
    )
    window_summary, window_references, topic_window_counts = _validate_window_artifact(
        paths.window_artifact, window_context, expectations, documents
    )
    document_cache_path, window_cache_path = _resolve_cache_paths(
        paths, document_context, window_context
    )
    document_cache_summary, document_extras = _validate_score_cache(
        document_cache_path, document_context, document_references
    )
    window_cache_summary, window_extras = _validate_score_cache(
        window_cache_path, window_context, window_references
    )
    runtime_status = (
        _validate_runtime_status(
            runtime_status_path,
            document=document_summary,
            window=window_summary,
            topic_document_counts=topic_document_counts,
            topic_window_counts=topic_window_counts,
            document_context=document_context,
            window_context=window_context,
            expectations=expectations,
        )
        if runtime_status_path is not None
        else None
    )
    reconciled = expectations.documents is not None and expectations.windows is not None
    return BundleValidation(
        document_artifact=document_summary,
        window_artifact=window_summary,
        document_score_cache=document_cache_summary,
        window_score_cache=window_cache_summary,
        document_context=document_context,
        window_context=window_context,
        topic_document_counts=topic_document_counts,
        topic_window_counts=topic_window_counts,
        cache_extra_keys={"document": document_extras, "window": window_extras},
        reconciled_to_local_inputs=reconciled,
        matched_selected_context=document_context_matched and window_context_matched,
        runtime_status=runtime_status,
    )


def _score_contexts_from_config(
    config: PipelineConfig,
) -> tuple[ScoreCacheContext, ScoreCacheContext]:
    reranker = config.ranking.reranker
    if reranker is None:
        raise ValueError("config must define a cached-artifact reranker")
    if reranker.artifact_schema_version not in {None, ARTIFACT_SCHEMA_VERSION}:
        raise ValueError(
            f"config artifact_schema_version must be {ARTIFACT_SCHEMA_VERSION}"
        )
    score_builder = build_score_cache_arg_parser()

    def configured(value: int | None, default_name: str) -> int:
        default = score_builder.get_default(default_name)
        if not isinstance(default, int) or isinstance(default, bool):
            raise RuntimeError(
                f"score builder has no integer default for {default_name}"
            )
        return value if value is not None else default

    document_max_length = configured(
        reranker.document_max_length, "document_max_length"
    )
    document_pair_buffer_tokens = configured(
        reranker.document_pair_buffer_tokens,
        "document_pair_buffer_tokens",
    )
    window_max_length = configured(reranker.window_max_length, "window_max_length")
    chunk_max_characters = configured(
        reranker.chunk_max_characters, "chunk_max_characters"
    )
    chunk_overlap_characters = configured(
        reranker.chunk_overlap_characters, "chunk_overlap_characters"
    )
    document_model_max_length = document_max_length - document_pair_buffer_tokens
    if document_model_max_length <= 0:
        raise ValueError(
            "config document_pair_buffer_tokens must be smaller than document_max_length"
        )
    score_representation = reranker.score_representation or DEFAULT_SCORE_REPRESENTATION
    if score_representation != DEFAULT_SCORE_REPRESENTATION:
        raise ValueError("selected config must require raw_logits")
    common = {
        "backend": "sentence-transformers-cross-encoder",
        "backend_version": reranker.backend_version or DEFAULT_BACKEND_VERSION,
        "model": reranker.model,
        "model_revision": reranker.model_revision or DEFAULT_MODEL_REVISION,
        "score_representation": score_representation,
        "inference_dtype": reranker.inference_dtype or DEFAULT_INFERENCE_DTYPE,
        "input_policy": reranker.input_policy or "trec_rag_raw_v2",
    }
    return (
        ScoreCacheContext(
            **common,
            max_length=document_model_max_length,
            requested_max_length=document_max_length,
            pair_buffer_tokens=document_pair_buffer_tokens,
            score_kind=(
                f"doc_max_{document_max_length}_buf{document_pair_buffer_tokens}"
            ),
        ),
        ScoreCacheContext(
            **common,
            max_length=window_max_length,
            requested_max_length=window_max_length,
            pair_buffer_tokens=0,
            score_kind="window",
            chunk_max_characters=chunk_max_characters,
            chunk_overlap_characters=chunk_overlap_characters,
        ),
    )


def derive_expectations_from_config(
    config_path: Path,
    *,
    candidate_limit: int = 1000,
    index_url: str | None = None,
    reconcile_known_rag25_counts: bool = True,
) -> BundleExpectations:
    """Derive exact document/window identities from the local retrieval cache."""

    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    config = load_pipeline_config(config_path)
    if len(config.retrievers) != 1 or config.ranking.reranker is None:
        raise ValueError(
            "config must define one retriever and a cached-artifact reranker"
        )
    document_context, window_context = _score_contexts_from_config(config)
    load_repo_env(config.root_dir)
    resolved_index_url = (
        index_url or os.environ.get("INDEX_URL") or DEFAULT_INDEX_URL
    ).rstrip("?")
    topics = _topics(config, [])
    queries = _queries_by_topic(config)
    retriever = config.retrievers[0]
    cache_dir = pipeline_cache_dir(config.root_dir, config.run_id)
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=window_context.chunk_max_characters or 0,
            overlap_characters=window_context.chunk_overlap_characters or 0,
        )
    )
    documents: dict[tuple[str, str], ExpectedDocumentIdentity] = {}
    windows: dict[tuple[str, str, int], ExpectedWindowIdentity] = {}
    topic_window_counts: Counter[str] = Counter()
    for topic in topics:
        candidates = _topic_candidates(
            config=config,
            topic=topic,
            query=queries[topic.id],
            retriever=retriever,
            cache_dir=cache_dir,
            index_url=resolved_index_url,
            limit=candidate_limit,
        )
        if len(candidates) != candidate_limit:
            raise CacheBundleValidationError(
                f"topic {topic.id}: local retrieval cache has {len(candidates)} candidates; "
                f"expected {candidate_limit}"
            )
        for candidate in candidates:
            doc_key = (topic.id, candidate.docid)
            if doc_key in documents:
                raise CacheBundleValidationError(
                    f"duplicate local candidate key: {doc_key}"
                )
            query_hash = _sha256_text(candidate.query_text)
            document_hash = _sha256_text(candidate.text)
            documents[doc_key] = ExpectedDocumentIdentity(
                candidate.rank, query_hash, document_hash
            )
            chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
            if not chunks:
                raise CacheBundleValidationError(
                    f"local input has no windows for {doc_key}"
                )
            for chunk_index, chunk in enumerate(chunks):
                window_key = (topic.id, candidate.docid, chunk_index)
                windows[window_key] = ExpectedWindowIdentity(
                    chunk_count=len(chunks),
                    chunk_id=chunk.chunk_id,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    query_sha256=query_hash,
                    document_text_sha256=document_hash,
                    text_sha256=_sha256_text(chunk.text),
                )
                topic_window_counts[topic.id] += 1
    topic_ids = tuple(topic.id for topic in topics)
    if (
        reconcile_known_rag25_counts
        and candidate_limit == 1000
        and set(topic_ids) == set(RAG25_DEV_TOPIC_IDS)
        and dict(topic_window_counts) != RAG25_WINDOW_ROWS_PER_TOPIC
    ):
        raise CacheBundleValidationError(
            "locally derived RAG25 window counts differ from the independently recorded "
            "1,000-document counts"
        )
    return BundleExpectations(
        topic_ids=topic_ids,
        documents_per_topic=candidate_limit,
        window_rows_per_topic=dict(topic_window_counts),
        documents=documents,
        windows=windows,
        document_context=document_context,
        window_context=window_context,
    )


def _copy_file_durable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)
        destination_file.flush()
        os.fsync(destination_file.fileno())
    shutil.copystat(source, destination, follow_symlinks=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_durable(
    path: Path,
    payload: Mapping[str, object],
    *,
    on_replace: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    if on_replace is not None:
        on_replace()
    _fsync_directory(path.parent)


def _atomic_replace(source: Path, destination: Path) -> None:
    """Small test seam around the same-filesystem atomic replacement primitive."""

    os.replace(source, destination)


def _validate_destination_score_cache(
    path: Path, context: ScoreCacheContext
) -> None:
    sidecars = _score_cache_sidecars(path)
    if not path.exists():
        if any(sidecar.exists() or sidecar.is_symlink() for sidecar in sidecars):
            raise CacheBundleValidationError(
                f"destination score cache is missing its main SQLite file: {path}"
            )
        return
    if path.is_symlink() or not path.is_file():
        raise CacheBundleValidationError(
            f"destination score cache must be a regular file: {path}"
        )
    connection = _open_score_cache(path, context, sealed=False)
    connection.close()


def _destination_cache(
    path: Path,
    root: Path | None,
    context: ScoreCacheContext,
) -> GlobalScoreCache:
    if root is None:
        suffix_length = len(context.path_parts)
        if len(path.parents) <= suffix_length - 1 or tuple(path.parts[-suffix_length:]) != context.path_parts:
            raise CacheBundleValidationError(
                "explicit destination score-cache paths must use the context path layout"
            )
        root = path.parents[suffix_length - 1]
    cache = GlobalScoreCache(root, context)
    if cache.path.resolve() != path.resolve():
        cache.close()
        raise CacheBundleValidationError(
            f"destination score cache path does not match its context: {path}"
        )
    return cache


def _import_sqlite_score_cache(
    source_path: Path,
    source_context: ScoreCacheContext,
    target_cache: GlobalScoreCache,
    target_context: ScoreCacheContext,
    *,
    expected_source_sha256: str,
    expected_source_row_count: int,
    expected_source_logical_digest: str,
) -> dict[str, object]:
    if _sha256_file(source_path) != expected_source_sha256:
        raise CacheBundleValidationError(
            f"staged score cache changed after validation: {source_path}"
        )
    logical_hasher = hashlib.sha256(bytes.fromhex(source_context.context_sha256))
    authorization_sha256 = _promotion_authorization_sha256(
        source_context,
        target_context,
    )

    def rows() -> Iterator[dict[str, object]]:
        connection = _open_score_cache(source_path, source_context, sealed=True)
        try:
            for row in connection.execute(
                "SELECT key_sha256, query_sha256, text_sha256, score, typeof(score) "
                "FROM scores ORDER BY key_sha256"
            ):
                cache_key, query_hash, text_hash, score = _score_cache_row(
                    source_path, source_context, row
                )
                logical_hasher.update(
                    _score_row_digest(cache_key, query_hash, text_hash, score)
                )
                yield {
                    "key": _cache_key_from_hashes(
                        target_context,
                        query_sha256=query_hash,
                        text_sha256=text_hash,
                    ),
                    "query_sha256": query_hash,
                    "text_sha256": text_hash,
                    "score": score,
                }
        finally:
            connection.close()

    receipt: dict[str, object] | None = None
    try:
        receipt = target_cache.import_scores(
            rows(),
            source_path=source_path,
            source_sha256=expected_source_sha256,
            authorization_sha256=authorization_sha256,
            expected_row_count=expected_source_row_count,
        )
        if logical_hasher.hexdigest() != expected_source_logical_digest:
            raise CacheBundleValidationError(
                f"staged score cache logical digest changed after validation: {source_path}"
            )
        _verify_sealed_score_cache_identity(source_path, expected_source_sha256)
        return receipt
    except BaseException:
        if receipt is not None:
            target_cache._rollback_score_import(receipt)
        raise


def _validate_destination_conflicts(
    source_path: Path,
    source_context: ScoreCacheContext,
    destination_path: Path,
    destination_context: ScoreCacheContext,
) -> None:
    """Check all existing destination rows before either cache is mutated."""

    if not destination_path.exists():
        return
    source_connection = _open_score_cache(source_path, source_context, sealed=True)
    destination_connection = _open_score_cache(
        destination_path, destination_context, sealed=False
    )
    try:
        for row in destination_connection.execute(
            "SELECT key_sha256, query_sha256, text_sha256, score, typeof(score) "
            "FROM scores ORDER BY key_sha256"
        ):
            _score_cache_row(destination_path, destination_context, row)

        batch: list[tuple[bytes, str, str, float]] = []

        def check_batch() -> None:
            if not batch:
                return
            expected_by_key = {
                key: (query_hash, text_hash, score)
                for key, query_hash, text_hash, score in batch
            }
            placeholders = ", ".join("?" for _ in batch)
            rows = destination_connection.execute(
                "SELECT key_sha256, query_sha256, text_sha256, score, typeof(score) "
                f"FROM scores WHERE key_sha256 IN ({placeholders})",
                tuple(key for key, _query, _text, _score in batch),
            )
            for row in rows:
                observed_key, query_hash, text_hash, score = _score_cache_row(
                    destination_path, destination_context, row
                )
                expected = expected_by_key[bytes.fromhex(observed_key)]
                if (query_hash, text_hash, score) != expected:
                    raise CacheBundleValidationError(
                        f"destination score cache conflicts with staged score key {observed_key}"
                    )
            batch.clear()

        for row in source_connection.execute(
            "SELECT key_sha256, query_sha256, text_sha256, score, typeof(score) "
            "FROM scores ORDER BY key_sha256"
        ):
            source_key, query_hash, text_hash, score = _score_cache_row(
                source_path, source_context, row
            )
            target_key = _cache_key_from_hashes(
                destination_context,
                query_sha256=query_hash,
                text_sha256=text_hash,
            )
            batch.append((bytes.fromhex(target_key), query_hash, text_hash, score))
            if len(batch) >= 512:
                check_batch()
        check_batch()
    finally:
        destination_connection.close()
        source_connection.close()


def _promotion_authorization_sha256(
    source_context: ScoreCacheContext,
    target_context: ScoreCacheContext,
) -> str:
    return _sha256_text(
        json.dumps(
            {
                "authorization_schema_version": 1,
                "legacy_context_sha256": source_context.context_sha256,
                "target_context_sha256": target_context.context_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _promotion_cache_receipt(
    validation: BundleValidation,
    *,
    label: str,
    target_context: ScoreCacheContext,
    import_receipt: Mapping[str, object],
) -> dict[str, object]:
    summary = (
        validation.document_score_cache
        if label == "document"
        else validation.window_score_cache
    )
    legacy_context = (
        validation.document_context
        if label == "document"
        else validation.window_context
    )
    authorization_sha256 = _promotion_authorization_sha256(
        legacy_context,
        target_context,
    )
    if import_receipt.get("source_sha256") != summary.sha256:
        raise CacheBundleValidationError("score-cache import source receipt mismatch")
    if import_receipt.get("authorization_sha256") != authorization_sha256:
        raise CacheBundleValidationError(
            "score-cache import authorization receipt mismatch"
        )
    return {
        "source_sha256": summary.sha256,
        "source_logical_digest": summary.logical_digest,
        "target_logical_digest": import_receipt["logical_digest"],
        "source_row_count": summary.row_count,
        "inserted_count": import_receipt["inserted_count"],
        "already_identical_count": import_receipt["already_identical_count"],
        "receipt_reused": import_receipt["receipt_reused"],
        "legacy_context_sha256": legacy_context.context_sha256,
        "target_context_sha256": target_context.context_sha256,
        "declared_input_policy": legacy_context.input_policy,
        "effective_input_policy": target_context.input_policy,
        "authorization_sha256": authorization_sha256,
    }


def _promotion_destination_paths(
    destination: CacheBundlePaths,
    expectations: BundleExpectations,
) -> tuple[Path, ...]:
    document_context = expectations.document_context
    window_context = expectations.window_context
    if document_context is None or window_context is None:
        raise CacheBundleValidationError(
            "promotion requires document/window score contexts derived from the selected config"
        )
    document_cache, window_cache = _resolve_cache_paths(
        destination,
        document_context,
        window_context,
    )
    return (
        destination.document_artifact,
        destination.window_artifact,
        document_cache,
        window_cache,
    )


def _acquire_promotion_locks(
    destination: CacheBundlePaths,
    expectations: BundleExpectations,
) -> list[Any]:
    lock_paths = sorted(
        {
            path.resolve().with_name(f".{path.name}.promotion.lock")
            for path in _promotion_destination_paths(destination, expectations)
        },
        key=str,
    )
    lock_files: list[Any] = []
    try:
        for lock_path in lock_paths:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = lock_path.open("a+b")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            lock_files.append(lock_file)
        return lock_files
    except BaseException:
        _release_promotion_locks(lock_files)
        raise


def _release_promotion_locks(lock_files: Sequence[Any]) -> None:
    for lock_file in reversed(lock_files):
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _promote_cache_bundle_unlocked(
    staged: CacheBundlePaths,
    destination: CacheBundlePaths,
    *,
    archive_root: Path,
    expectations: BundleExpectations,
    runtime_status_path: Path | None = None,
) -> PromotionResult:
    """Validate and promote artifacts while logically importing SQLite scores.

    Source score databases are sealed read-only inputs and are never copied.
    Their rows are imported into the context-bound destination caches in
    bounded transactions.  Artifact files are switched only after both logical
    imports succeed; the durable promotion manifest is written last.  A retry
    after an interrupted import is safe because ``GlobalScoreCache`` treats
    identical existing rows as idempotent and rejects contradictory rows.
    """

    validation = validate_cache_bundle(
        staged,
        expectations=expectations,
        runtime_status_path=runtime_status_path,
        _allow_legacy_policy_rebinding=True,
    )
    if (
        not validation.reconciled_to_local_inputs
        or not validation.matched_selected_context
    ):
        raise CacheBundleValidationError(
            "promotion requires document/window identities and score contexts derived "
            "from the selected config"
        )
    if any(validation.cache_extra_keys.values()):
        raise CacheBundleValidationError(
            "promotion requires exact artifact/cache key coverage; staged global "
            f"cache has extra keys: {dict(validation.cache_extra_keys)}"
        )
    target_document_context = expectations.document_context
    target_window_context = expectations.window_context
    if target_document_context is None or target_window_context is None:
        raise CacheBundleValidationError(
            "promotion requires document/window score contexts derived from the selected config"
        )
    staged_document_cache, staged_window_cache = _resolve_cache_paths(
        staged, validation.document_context, validation.window_context
    )
    destination_document_cache, destination_window_cache = _resolve_cache_paths(
        destination, target_document_context, target_window_context
    )
    sources = {
        "document_artifact": staged.document_artifact,
        "window_artifact": staged.window_artifact,
        "document_score_cache": staged_document_cache,
        "window_score_cache": staged_window_cache,
    }
    destinations = {
        "document_artifact": destination.document_artifact,
        "window_artifact": destination.window_artifact,
        "document_score_cache": destination_document_cache,
        "window_score_cache": destination_window_cache,
    }
    if len({path.resolve() for path in destinations.values()}) != len(destinations):
        raise CacheBundleValidationError("promotion destination paths must be distinct")
    for label, source in sources.items():
        if source.is_symlink():
            raise CacheBundleValidationError(f"staged {label} must not be a symlink")
        if source.resolve() == destinations[label].resolve():
            raise CacheBundleValidationError(
                f"staged and destination {label} paths must be different"
            )

    for label, target in destinations.items():
        if label.endswith("_artifact") and target.exists():
            if target.is_symlink() or not target.is_file():
                raise CacheBundleValidationError(
                    f"destination {label} must be a regular file: {target}"
                )
    _validate_destination_score_cache(
        destination_document_cache, target_document_context
    )
    _validate_destination_score_cache(destination_window_cache, target_window_context)

    _validate_destination_conflicts(
        staged_document_cache,
        validation.document_context,
        destination_document_cache,
        target_document_context,
    )
    _validate_destination_conflicts(
        staged_window_cache,
        validation.window_context,
        destination_window_cache,
        target_window_context,
    )

    if destination_document_cache.resolve() == destination_window_cache.resolve():
        raise CacheBundleValidationError("promotion destination paths must be distinct")
    if archive_root.exists() and not archive_root.is_dir():
        raise CacheBundleValidationError(f"promotion archive root must be a directory: {archive_root}")

    destination_document_cache_obj = _destination_cache(
        destination_document_cache,
        destination.score_cache_root,
        target_document_context,
    )
    destination_window_cache_obj = _destination_cache(
        destination_window_cache,
        destination.score_cache_root,
        target_window_context,
    )
    import_receipts: list[tuple[GlobalScoreCache, dict[str, object]]] = []
    manifest_published = False
    try:
        try:
            document_import_receipt = _import_sqlite_score_cache(
                staged_document_cache,
                validation.document_context,
                destination_document_cache_obj,
                target_document_context,
                expected_source_sha256=validation.document_score_cache.sha256,
                expected_source_row_count=validation.document_score_cache.row_count,
                expected_source_logical_digest=(
                    validation.document_score_cache.logical_digest or ""
                ),
            )
            import_receipts.append(
                (destination_document_cache_obj, document_import_receipt)
            )
            window_import_receipt = _import_sqlite_score_cache(
                staged_window_cache,
                validation.window_context,
                destination_window_cache_obj,
                target_window_context,
                expected_source_sha256=validation.window_score_cache.sha256,
                expected_source_row_count=validation.window_score_cache.row_count,
                expected_source_logical_digest=(
                    validation.window_score_cache.logical_digest or ""
                ),
            )
            import_receipts.append(
                (destination_window_cache_obj, window_import_receipt)
            )
        except (CacheBundleValidationError, OSError, sqlite3.Error, ValueError) as exc:
            raise CacheBundleValidationError(
                f"destination score-cache logical import failed: {exc}"
            ) from exc

        _verify_sealed_score_cache_identity(
            staged_document_cache,
            validation.document_score_cache.sha256,
        )
        _verify_sealed_score_cache_identity(
            staged_window_cache,
            validation.window_score_cache.sha256,
        )
        expected_artifact_digests = {
            "document_artifact": validation.document_artifact.sha256,
            "window_artifact": validation.window_artifact.sha256,
        }
        transaction_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        archive_dir = archive_root / transaction_id
        previous_dir = archive_dir / "previous"
        archive_root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(archive_root.parent)
        archive_dir.mkdir(exist_ok=False)
        _fsync_directory(archive_root)
        previous_dir.mkdir()
        _fsync_directory(archive_dir)

        artifact_labels = ("document_artifact", "window_artifact")
        incoming: dict[str, Path] = {}
        rollback: dict[str, Path] = {}
        backups: dict[str, Path] = {}
        installed: set[str] = set()
        old_moved: set[str] = set()
        unchanged: set[str] = set()
        manifest_path = archive_dir / "promotion_manifest.json"
        try:
            for label in artifact_labels:
                source = sources[label]
                target = destinations[label]
                target.parent.mkdir(parents=True, exist_ok=True)
                if (
                    target.is_file()
                    and _sha256_file(target) == expected_artifact_digests[label]
                ):
                    unchanged.add(label)
                    continue
                incoming_path = target.with_name(
                    f".{target.name}.{transaction_id}.incoming"
                )
                incoming[label] = incoming_path
                _copy_file_durable(source, incoming_path)
                _fsync_directory(target.parent)
                if _sha256_file(incoming_path) != expected_artifact_digests[label]:
                    raise CacheBundleValidationError(
                        f"staged {label} changed after validation; promotion aborted"
                    )
                if target.exists():
                    if target.is_symlink() or not target.is_file():
                        raise CacheBundleValidationError(
                            f"destination {label} must be a regular file: {target}"
                        )
                    backup = previous_dir / f"{label}__{target.name}"
                    _copy_file_durable(target, backup)
                    backups[label] = backup
                    rollback[label] = target.with_name(
                        f".{target.name}.{transaction_id}.rollback"
                    )

            # The archive hierarchy and every previous-file directory entry must be
            # durable before the first live destination is moved aside.
            _fsync_directory(previous_dir)
            _fsync_directory(archive_dir)
            _fsync_directory(archive_root)

            for label in artifact_labels:
                target = destinations[label]
                if label in unchanged:
                    continue
                if target.exists():
                    _atomic_replace(target, rollback[label])
                    old_moved.add(label)
                _atomic_replace(incoming[label], target)
                installed.add(label)
                _fsync_directory(target.parent)

            for label in artifact_labels:
                target = destinations[label]
                if _sha256_file(target) != expected_artifact_digests[label]:
                    raise CacheBundleValidationError(
                        f"promoted {label} checksum differs from validated staging file"
                    )
            manifest = {
                "transaction_id": transaction_id,
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {key: str(value) for key, value in sources.items()},
                "destinations": {
                    key: str(value) for key, value in destinations.items()
                },
                "backups": {key: str(value) for key, value in backups.items()},
                "sha256": {
                    **expected_artifact_digests,
                    "document_score_cache": validation.document_score_cache.sha256,
                    "window_score_cache": validation.window_score_cache.sha256,
                },
                "score_cache_receipts": {
                    "document": _promotion_cache_receipt(
                        validation,
                        label="document",
                        target_context=target_document_context,
                        import_receipt=document_import_receipt,
                    ),
                    "window": _promotion_cache_receipt(
                        validation,
                        label="window",
                        target_context=target_window_context,
                        import_receipt=window_import_receipt,
                    ),
                },
                "reviewed_modal_runtime_status": (
                    validation.runtime_status.to_dict()
                    if validation.runtime_status is not None
                    else None
                ),
                "validation": validation.to_dict(),
            }
            def mark_manifest_published() -> None:
                nonlocal manifest_published
                manifest_published = True

            _write_json_durable(
                manifest_path,
                manifest,
                on_replace=mark_manifest_published,
            )
        except BaseException:
            # The archive directory is unique to this attempt, so the marker can
            # only exist after our atomic rename. Rename is the publication
            # boundary even if its following directory fsync reports an error.
            if not manifest_published and manifest_path.is_file():
                manifest_published = True
            if not manifest_published:
                for label in reversed(artifact_labels):
                    target = destinations[label]
                    target_changed = False
                    if label in installed and target.exists():
                        target.unlink()
                        target_changed = True
                    rollback_path = rollback.get(label)
                    if label in old_moved and rollback_path and rollback_path.exists():
                        _atomic_replace(rollback_path, target)
                        target_changed = True
                    if target_changed:
                        _fsync_directory(target.parent)
            raise
        finally:
            incoming_parents: set[Path] = set()
            for path in incoming.values():
                if path.exists():
                    path.unlink()
                    incoming_parents.add(path.parent)
            for parent in incoming_parents:
                _fsync_directory(parent)

        rollback_parents: set[Path] = set()
        for path in rollback.values():
            if path.exists():
                path.unlink()
                rollback_parents.add(path.parent)
        for parent in rollback_parents:
            _fsync_directory(parent)
        return PromotionResult(validation, archive_dir, destinations, backups)
    except BaseException:
        if not manifest_published:
            for cache, receipt in reversed(import_receipts):
                cache._rollback_score_import(receipt)
        raise
    finally:
        destination_document_cache_obj.close()
        destination_window_cache_obj.close()


def promote_cache_bundle(
    staged: CacheBundlePaths,
    destination: CacheBundlePaths,
    *,
    archive_root: Path,
    expectations: BundleExpectations,
    runtime_status_path: Path | None = None,
) -> PromotionResult:
    lock_files = _acquire_promotion_locks(destination, expectations)
    try:
        return _promote_cache_bundle_unlocked(
            staged,
            destination,
            archive_root=archive_root,
            expectations=expectations,
            runtime_status_path=runtime_status_path,
        )
    finally:
        _release_promotion_locks(lock_files)


def _common_cli_arguments(
    parser: argparse.ArgumentParser, *, runtime_status_required: bool
) -> None:
    parser.add_argument("--staged-document-artifact", type=Path, required=True)
    parser.add_argument("--staged-window-artifact", type=Path, required=True)
    parser.add_argument("--staged-score-cache-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"),
    )
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--index-url", default=None)
    parser.add_argument(
        "--runtime-status",
        type=Path,
        required=runtime_status_required,
        help="Downloaded Modal runtime_status.json to reconcile with staged artifacts.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and promote staged schema-v2 reranker score caches."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    _common_cli_arguments(validate, runtime_status_required=False)
    promote = commands.add_parser(
        "promote",
        description=(
            "Promote a validated bundle with a logical SQLite import and a "
            "manifest-last receipt. Identical attempts are serialized."
        ),
    )
    _common_cli_arguments(promote, runtime_status_required=True)
    promote.add_argument("--destination-document-artifact", type=Path, required=True)
    promote.add_argument("--destination-window-artifact", type=Path, required=True)
    promote.add_argument("--destination-score-cache-root", type=Path, required=True)
    promote.add_argument("--archive-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    expectations = derive_expectations_from_config(
        args.config,
        candidate_limit=args.candidate_limit,
        index_url=args.index_url,
    )
    staged = CacheBundlePaths(
        document_artifact=args.staged_document_artifact,
        window_artifact=args.staged_window_artifact,
        score_cache_root=args.staged_score_cache_root,
    )
    if args.command == "validate":
        payload = validate_cache_bundle(
            staged,
            expectations=expectations,
            runtime_status_path=args.runtime_status,
        ).to_dict()
    else:
        destination = CacheBundlePaths(
            document_artifact=args.destination_document_artifact,
            window_artifact=args.destination_window_artifact,
            score_cache_root=args.destination_score_cache_root,
        )
        payload = promote_cache_bundle(
            staged,
            destination,
            archive_root=args.archive_root,
            expectations=expectations,
            runtime_status_path=args.runtime_status,
        ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

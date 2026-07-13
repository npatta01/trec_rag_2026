"""Pinned safe-model materialization and tokenizer-only MiniLM preflight.

This module deliberately separates the one approved Hugging Face download from
all later use.  Materialization downloads an exact safe-file allowlist without
constructing a tokenizer or model.  Preflight then verifies those bytes, loads
only the local tokenizer, and creates a bounded query/document window plan.  It
contains no model construction, forward pass, retrieval, or qrels interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .facet_local_minilm_manifest import (
    PROTECTED_TOPIC_IDS,
    FacetLocalManifest,
    load_facet_local_manifest,
    load_facet_local_source_snapshot,
)
from .rerank_score_cache import GlobalScoreCache, ScoreCacheContext


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

APPROVAL_SCHEMA_VERSION = "facet-local-minilm-model-download-approval-v1"
APPROVAL_SCOPE = "facet_local_minilm_model_materialization_v1"
MATERIALIZATION_SCHEMA_VERSION = "facet-local-minilm-materialization-v1"
WINDOW_SCHEMA_VERSION = "facet-local-minilm-window-plan-row-v1"
WINDOW_POLICY_VERSION = "facet-local-minilm-window-policy-v1"
PREFLIGHT_SCHEMA_VERSION = "facet-local-minilm-preflight-v1"

PAIR_MAX_TOKENS = 512
QUERY_MAX_TOKENS = 192
MIN_PASSAGE_TOKENS = 256
PASSAGE_OVERLAP_TOKENS = 64
MAX_WINDOWS_PER_DOCUMENT = 32
MAX_UNCACHED_PAIRS = 100_000

_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "approval_scope",
        "approved_by",
        "model_id",
        "revision",
        "allow_patterns",
        "allow_patterns_sha256",
        "acknowledged_network_download",
        "acknowledged_safe_files_only",
        "acknowledged_no_model_or_tokenizer_construction",
        "acknowledged_no_inference_qrels_retrieval_or_paid_calls",
    }
)
_MATERIALIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "model_id",
        "revision",
        "allow_patterns",
        "allow_patterns_sha256",
        "approval_file",
        "approval_sha256",
        "resolved_snapshot_path",
        "files",
        "snapshot_bytes",
        "snapshot_sha256",
    }
)
_FILE_FIELDS = frozenset({"path", "bytes", "sha256"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_compact_json_bytes(value: object) -> bytes:
    """Canonical compact JSON bytes used for identities, without a newline."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _loads_object_no_duplicates(content: bytes, label: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)


def approval_allow_patterns_sha256() -> str:
    """Hash of the exact ordered allowlist, independent of receipt formatting."""

    return _sha256_bytes(canonical_compact_json_bytes(list(ALLOW_PATTERNS)))


def _load_and_validate_approval(path: Path) -> tuple[dict[str, object], bytes]:
    approval_path = Path(path)
    try:
        source = approval_path.read_bytes()
    except OSError as exc:
        raise ValueError("model download approval receipt is required") from exc
    receipt = _loads_object_no_duplicates(source, "model download approval receipt")
    if _canonical_pretty_json_bytes(receipt) != source:
        raise ValueError("model download approval receipt must be canonical JSON bytes")
    if set(receipt) != _APPROVAL_FIELDS:
        raise ValueError("model download approval receipt fields mismatch")
    if receipt.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise ValueError("model download approval schema mismatch")
    if receipt.get("approval_scope") != APPROVAL_SCOPE:
        raise ValueError("model download approval scope mismatch")
    approved_by = receipt.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("model download approval approved_by must be nonempty")
    if receipt.get("model_id") != MODEL_ID:
        raise ValueError("model download approval model mismatch")
    if receipt.get("revision") != MODEL_REVISION:
        raise ValueError("model download approval revision mismatch")
    if receipt.get("allow_patterns") != list(ALLOW_PATTERNS):
        raise ValueError("model download approval allowlist mismatch")
    if receipt.get("allow_patterns_sha256") != approval_allow_patterns_sha256():
        raise ValueError("model download approval allowlist hash mismatch")
    for field in (
        "acknowledged_network_download",
        "acknowledged_safe_files_only",
        "acknowledged_no_model_or_tokenizer_construction",
        "acknowledged_no_inference_qrels_retrieval_or_paid_calls",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"model download approval must acknowledge {field}")
    return receipt, source


def _safe_snapshot_file_records(snapshot: Path) -> list[dict[str, object]]:
    if not snapshot.is_dir():
        raise ValueError("resolved model snapshot is not a directory")
    observed = sorted(
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    )
    pickle_suffixes = (".bin", ".pt", ".pth", ".pickle", ".pkl")
    unexpected = [
        name
        for name in observed
        if name not in ALLOW_PATTERNS or name.lower().endswith(pickle_suffixes)
    ]
    if unexpected:
        raise ValueError(
            "pickle or unexpected model files are forbidden: " + ", ".join(unexpected)
        )
    missing = [name for name in ALLOW_PATTERNS if name not in observed]
    if missing:
        raise ValueError("allowlisted model files are missing: " + ", ".join(missing))
    records: list[dict[str, object]] = []
    for name in ALLOW_PATTERNS:
        path = snapshot / name
        records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def materialize_model(
    *,
    model_id: str,
    revision: str,
    approval_path: Path,
    output_dir: Path,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, object]:
    """Download only the approved pinned files and write a create-only receipt."""

    if model_id != MODEL_ID:
        raise ValueError("requested model differs from the pinned model")
    if revision != MODEL_REVISION:
        raise ValueError("requested revision differs from the pinned revision")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only model output already exists: {destination}")
    _approval, approval_source = _load_and_validate_approval(Path(approval_path))
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    resolved = Path(
        snapshot_download_fn(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOW_PATTERNS),
        )
    ).resolve()
    files = _safe_snapshot_file_records(resolved)
    receipt: dict[str, object] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "pinned_safe_files_materialized",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "allow_patterns": list(ALLOW_PATTERNS),
        "allow_patterns_sha256": approval_allow_patterns_sha256(),
        "approval_file": str(Path(approval_path).resolve()),
        "approval_sha256": _sha256_bytes(approval_source),
        "resolved_snapshot_path": str(resolved),
        "files": files,
        "snapshot_bytes": sum(int(row["bytes"]) for row in files),
        "snapshot_sha256": _sha256_bytes(canonical_compact_json_bytes(files)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"create-only model output already exists: {destination}"
        ) from exc
    _exclusive_write(destination / "materialization.json", _canonical_pretty_json_bytes(receipt))
    return receipt


def verify_materialization_receipt(path: Path) -> Path:
    """Verify receipt, approval, exact snapshot file set, bytes, and hashes."""

    receipt_path = Path(path)
    try:
        source = receipt_path.read_bytes()
    except OSError as exc:
        raise ValueError("model materialization receipt is required") from exc
    receipt = _loads_object_no_duplicates(source, "model materialization receipt")
    if _canonical_pretty_json_bytes(receipt) != source:
        raise ValueError("model materialization receipt must be canonical JSON bytes")
    if set(receipt) != _MATERIALIZATION_FIELDS:
        raise ValueError("model materialization receipt fields mismatch")
    if (
        receipt.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or receipt.get("status") != "pinned_safe_files_materialized"
        or receipt.get("model_id") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("allow_patterns") != list(ALLOW_PATTERNS)
        or receipt.get("allow_patterns_sha256") != approval_allow_patterns_sha256()
    ):
        raise ValueError("model materialization receipt binding mismatch")
    approval_file = receipt.get("approval_file")
    if not isinstance(approval_file, str) or not approval_file:
        raise ValueError("model materialization approval path is invalid")
    _approval, approval_source = _load_and_validate_approval(Path(approval_file))
    if receipt.get("approval_sha256") != _sha256_bytes(approval_source):
        raise ValueError("model materialization approval hash mismatch")
    snapshot_value = receipt.get("resolved_snapshot_path")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("model materialization snapshot path is invalid")
    snapshot = Path(snapshot_value).resolve()
    files = _safe_snapshot_file_records(snapshot)
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list) or any(
        not isinstance(row, dict) or set(row) != _FILE_FIELDS for row in raw_files
    ):
        raise ValueError("model materialization file records are invalid")
    for expected, observed in zip(files, raw_files, strict=False):
        if expected["path"] != observed.get("path"):
            raise ValueError("materialized file path mismatch")
        if expected["bytes"] != observed.get("bytes"):
            raise ValueError("materialized file byte count mismatch")
        if expected["sha256"] != observed.get("sha256"):
            raise ValueError("materialized file hash mismatch")
    if len(files) != len(raw_files):
        raise ValueError("materialized file record count mismatch")
    if receipt.get("snapshot_bytes") != sum(int(row["bytes"]) for row in files):
        raise ValueError("materialized snapshot byte count mismatch")
    if receipt.get("snapshot_sha256") != _sha256_bytes(
        canonical_compact_json_bytes(files)
    ):
        raise ValueError("materialized snapshot hash mismatch")
    return snapshot


def load_verified_tokenizer(
    materialization_receipt_path: Path,
    *,
    auto_tokenizer_cls: object | None = None,
) -> object:
    """Load only a tokenizer from a freshly verified local snapshot."""

    snapshot = verify_materialization_receipt(materialization_receipt_path)
    if auto_tokenizer_cls is None:
        from transformers import AutoTokenizer

        auto_tokenizer_cls = AutoTokenizer
    return auto_tokenizer_cls.from_pretrained(  # type: ignore[attr-defined]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def score_cache_context() -> ScoreCacheContext:
    """Frozen cache identity shared by preflight and the later scorer."""

    return ScoreCacheContext(
        backend="transformers-auto-sequence-classification",
        backend_version=importlib.metadata.version("transformers"),
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        max_length=PAIR_MAX_TOKENS,
        score_kind="facet_local_window",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy=WINDOW_POLICY_VERSION,
        requested_max_length=PAIR_MAX_TOKENS,
        pair_buffer_tokens=0,
    )


def _score_cache_key(query: str, window_text: str) -> str:
    context = score_cache_context()
    payload = {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": context.backend,
        "model": context.model,
        "max_length": context.max_length,
        "score_kind": context.score_kind,
        **context.cache_identity_metadata,
        "query_sha256": _sha256_bytes(query.encode("utf-8")),
        "text_sha256": _sha256_bytes(window_text.encode("utf-8")),
    }
    return _sha256_bytes(canonical_compact_json_bytes(payload))


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    """Round a non-negative rational number to nearest, with halves upward."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("round_half_up_ratio requires non-negative/positive inputs")
    return (2 * numerator + denominator) // (2 * denominator)


def _candidate_value(candidate: Mapping[str, object] | object, field: str) -> object:
    if isinstance(candidate, Mapping):
        return candidate.get(field)
    return getattr(candidate, field, None)


def _required_candidate_text(
    candidate: Mapping[str, object] | object, field: str
) -> str:
    value = _candidate_value(candidate, field)
    if not isinstance(value, str) or (field != "text" and not value):
        raise ValueError(f"candidate {field} must be text")
    return value


def _reject_protected_candidates(candidates: Sequence[Mapping[str, object] | object]) -> None:
    for candidate in candidates:
        topic_id = _candidate_value(candidate, "topic_id")
        if str(topic_id) in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _selected_window_indices(window_count: int) -> tuple[int, ...]:
    if window_count <= MAX_WINDOWS_PER_DOCUMENT:
        return tuple(range(window_count))
    indices = tuple(
        round_half_up_ratio(index * (window_count - 1), MAX_WINDOWS_PER_DOCUMENT - 1)
        for index in range(MAX_WINDOWS_PER_DOCUMENT)
    )
    if (
        len(indices) != MAX_WINDOWS_PER_DOCUMENT
        or len(set(indices)) != MAX_WINDOWS_PER_DOCUMENT
        or tuple(sorted(indices)) != indices
        or indices[0] != 0
        or indices[-1] != window_count - 1
    ):
        raise ValueError("32-window half-up selection is not unique and increasing")
    return indices


def _coverage_fraction(spans: Sequence[tuple[int, int]], token_count: int) -> float:
    if token_count == 0:
        return 1.0
    covered = 0
    cursor = 0
    for start, end in sorted(spans):
        if end <= cursor:
            continue
        covered += end - max(start, cursor)
        cursor = max(cursor, end)
    return covered / token_count


def _decode_fitted_window(
    tokenizer: object,
    document_tokens: Sequence[object],
    *,
    start: int,
    proposed_end: int,
    passage_budget: int,
) -> tuple[int, str, int]:
    """Fit the serialized passage by its actual re-tokenized length."""

    end = proposed_end
    while True:
        window_text = tokenizer.decode(  # type: ignore[attr-defined]
            document_tokens[start:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        serialized_tokens = tokenizer.encode(  # type: ignore[attr-defined]
            window_text, add_special_tokens=False, truncation=False
        )
        serialized_count = len(serialized_tokens)
        if serialized_count <= passage_budget:
            return end, window_text, serialized_count
        original_count = end - start
        if original_count <= 1:
            raise ValueError(
                "one serialized document token exceeds the passage token budget"
            )
        fitted_count = max(1, original_count * passage_budget // serialized_count)
        if fitted_count >= original_count:
            fitted_count = original_count - 1
        end = start + fitted_count


@dataclass(frozen=True)
class WindowPlanRow:
    topic_id: str
    family: str
    variant: str
    rank: int
    document_id: str
    query: str
    query_sha256: str
    query_token_count: int
    document_sha256: str
    document_token_count: int
    passage_token_budget: int
    passage_overlap_tokens: int
    original_window_count: int
    original_window_index: int
    selected_window_count: int
    selected_window_index: int
    document_start_token: int
    document_end_token: int
    document_token_coverage_fraction: float
    window_text: str
    window_sha256: str
    pair_special_token_count: int
    pair_token_count: int
    window_id: str
    cache_key: str
    cache_hit: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": WINDOW_SCHEMA_VERSION, **self.__dict__}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WindowPlanRow":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected.union({"schema_version"}):
            raise ValueError("window plan row fields mismatch")
        if value.get("schema_version") != WINDOW_SCHEMA_VERSION:
            raise ValueError("window plan row schema mismatch")
        return cls(**{key: value[key] for key in expected})  # type: ignore[arg-type]


@dataclass(frozen=True)
class DocumentWindowDiagnostic:
    topic_id: str
    variant: str
    rank: int
    document_id: str
    document_token_count: int
    original_window_count: int
    selected_window_count: int
    capped: bool
    document_token_coverage_fraction: float

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PreflightPlan:
    windows: tuple[WindowPlanRow, ...]
    documents: tuple[DocumentWindowDiagnostic, ...]
    streams: tuple[dict[str, object], ...]
    topics: tuple[dict[str, object], ...]
    summary: dict[str, object]
    benchmark: dict[str, object]


def _window_id(row: Mapping[str, object]) -> str:
    identity = {
        "schema_version": WINDOW_SCHEMA_VERSION,
        "topic_id": row["topic_id"],
        "variant": row["variant"],
        "rank": row["rank"],
        "document_id": row["document_id"],
        "query_sha256": row["query_sha256"],
        "document_sha256": row["document_sha256"],
        "original_window_index": row["original_window_index"],
        "document_start_token": row["document_start_token"],
        "document_end_token": row["document_end_token"],
        "window_sha256": row["window_sha256"],
    }
    return _sha256_bytes(canonical_compact_json_bytes(identity))


def build_window_plan(
    candidate: Mapping[str, object] | object,
    tokenizer: object,
    *,
    query: str,
) -> tuple[WindowPlanRow, ...]:
    """Build one bounded query/document window plan without truncation."""

    topic_id = _required_candidate_text(candidate, "topic_id")
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    if not isinstance(query, str) or not query:
        raise ValueError("query must be nonempty text")
    family = _required_candidate_text(candidate, "family")
    variant = _required_candidate_text(candidate, "variant")
    document_id = _required_candidate_text(candidate, "document_id")
    text = _required_candidate_text(candidate, "text")
    rank = _candidate_value(candidate, "rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("candidate rank must be a positive integer")
    expected_query_hash = _sha256_bytes(query.encode("utf-8"))
    candidate_query = _candidate_value(candidate, "query")
    supplied_query_hash = _candidate_value(candidate, "query_sha256")
    if (
        candidate_query == query
        and supplied_query_hash is not None
        and supplied_query_hash != expected_query_hash
    ):
        raise ValueError("candidate query hash mismatch")
    expected_document_hash = _sha256_bytes(text.encode("utf-8"))
    supplied_document_hash = _candidate_value(candidate, "text_sha256")
    if supplied_document_hash is not None and supplied_document_hash != expected_document_hash:
        raise ValueError("candidate document hash mismatch")

    query_tokens = tokenizer.encode(  # type: ignore[attr-defined]
        query, add_special_tokens=False, truncation=False
    )
    if len(query_tokens) > QUERY_MAX_TOKENS:
        raise ValueError(f"query exceeds {QUERY_MAX_TOKENS} tokens")
    special_token_count = tokenizer.num_special_tokens_to_add(pair=True)  # type: ignore[attr-defined]
    if isinstance(special_token_count, bool) or not isinstance(special_token_count, int):
        raise ValueError("tokenizer pair special-token count must be an integer")
    passage_budget = PAIR_MAX_TOKENS - len(query_tokens) - special_token_count
    if passage_budget < MIN_PASSAGE_TOKENS:
        raise ValueError(
            f"passage budget is below {MIN_PASSAGE_TOKENS} tokens"
        )
    document_tokens = tokenizer.encode(  # type: ignore[attr-defined]
        text, add_special_tokens=False, truncation=False
    )
    document_token_count = len(document_tokens)
    serialized_windows: list[tuple[int, int, str, int]] = []
    if document_token_count == 0:
        end, window_text, serialized_count = _decode_fitted_window(
            tokenizer,
            document_tokens,
            start=0,
            proposed_end=0,
            passage_budget=passage_budget,
        )
        serialized_windows.append((0, end, window_text, serialized_count))
    else:
        start = 0
        while True:
            proposed_end = min(start + passage_budget, document_token_count)
            end, window_text, serialized_count = _decode_fitted_window(
                tokenizer,
                document_tokens,
                start=start,
                proposed_end=proposed_end,
                passage_budget=passage_budget,
            )
            serialized_windows.append((start, end, window_text, serialized_count))
            if end == document_token_count:
                break
            if end - start <= PASSAGE_OVERLAP_TOKENS:
                raise ValueError(
                    "serialized passage cannot advance beyond the 64-token overlap"
                )
            start = end - PASSAGE_OVERLAP_TOKENS
    selected_indices = _selected_window_indices(len(serialized_windows))
    selected_spans = [
        serialized_windows[index][:2] for index in selected_indices
    ]
    coverage = _coverage_fraction(selected_spans, document_token_count)
    rows: list[WindowPlanRow] = []
    for selected_position, original_index in enumerate(selected_indices):
        start, end, window_text, serialized_count = serialized_windows[original_index]
        window_hash = _sha256_bytes(window_text.encode("utf-8"))
        values: dict[str, object] = {
            "topic_id": topic_id,
            "family": family,
            "variant": variant,
            "rank": rank,
            "document_id": document_id,
            "query": query,
            "query_sha256": expected_query_hash,
            "query_token_count": len(query_tokens),
            "document_sha256": expected_document_hash,
            "document_token_count": document_token_count,
            "passage_token_budget": passage_budget,
            "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "original_window_count": len(serialized_windows),
            "original_window_index": original_index,
            "selected_window_count": len(selected_indices),
            "selected_window_index": selected_position,
            "document_start_token": start,
            "document_end_token": end,
            "document_token_coverage_fraction": coverage,
            "window_text": window_text,
            "window_sha256": window_hash,
            "pair_special_token_count": special_token_count,
            "pair_token_count": len(query_tokens) + serialized_count + special_token_count,
            "cache_key": _score_cache_key(query, window_text),
        }
        values["window_id"] = _window_id(values)
        rows.append(WindowPlanRow(**values))  # type: ignore[arg-type]
    result = tuple(rows)
    verify_window_plan(result)
    return result


def verify_window_plan(rows: Sequence[WindowPlanRow]) -> None:
    """Recompute all policy positions and content identities for one document."""

    if not rows:
        raise ValueError("window plan must contain at least one row")
    first = rows[0]
    invariant_fields = (
        "topic_id",
        "family",
        "variant",
        "rank",
        "document_id",
        "query",
        "query_sha256",
        "query_token_count",
        "document_sha256",
        "document_token_count",
        "passage_token_budget",
        "passage_overlap_tokens",
        "original_window_count",
        "selected_window_count",
        "document_token_coverage_fraction",
        "pair_special_token_count",
    )
    if any(
        getattr(row, field) != getattr(first, field)
        for row in rows
        for field in invariant_fields
    ):
        raise ValueError("window plan document invariants differ")
    if first.topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {first.topic_id} is forbidden")
    if first.query_sha256 != _sha256_bytes(first.query.encode("utf-8")):
        raise ValueError("window query hash mismatch")
    if not _is_sha256(first.document_sha256):
        raise ValueError("window document hash is invalid")
    if first.passage_overlap_tokens != PASSAGE_OVERLAP_TOKENS:
        raise ValueError("window overlap differs from 64 tokens")
    if first.passage_token_budget < MIN_PASSAGE_TOKENS:
        raise ValueError("window passage budget is below 256 tokens")
    if first.query_token_count > QUERY_MAX_TOKENS:
        raise ValueError("window query exceeds 192 tokens")
    if first.selected_window_count != len(rows) or len(rows) > MAX_WINDOWS_PER_DOCUMENT:
        raise ValueError("window selected count is invalid")
    expected_indices = _selected_window_indices(first.original_window_count)
    if tuple(row.original_window_index for row in rows) != expected_indices:
        raise ValueError("window original indices differ from the exact policy")
    if tuple(row.selected_window_index for row in rows) != tuple(range(len(rows))):
        raise ValueError("window selected indices are not contiguous")
    expected_spans = tuple(
        (row.document_start_token, row.document_end_token) for row in rows
    )
    if any(
        start < 0
        or end < start
        or end > first.document_token_count
        or end - start > first.passage_token_budget
        for start, end in expected_spans
    ):
        raise ValueError("window spans exceed document or passage bounds")
    if first.original_window_count <= MAX_WINDOWS_PER_DOCUMENT and any(
        current.document_start_token
        != previous.document_end_token - PASSAGE_OVERLAP_TOKENS
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("window spans differ from the exact 64-token overlap policy")
    coverage = _coverage_fraction(expected_spans, first.document_token_count)
    if first.document_token_coverage_fraction != coverage:
        raise ValueError("window document-token coverage mismatch")
    if rows[0].document_start_token != 0:
        raise ValueError("window plan does not cover the document start")
    if rows[-1].document_end_token != first.document_token_count:
        raise ValueError("window plan does not cover the document end")
    for row in rows:
        if row.window_sha256 != _sha256_bytes(row.window_text.encode("utf-8")):
            raise ValueError("window hash mismatch")
        if row.cache_key != _score_cache_key(row.query, row.window_text):
            raise ValueError("window cache key mismatch")
        if row.window_id != _window_id(row.__dict__):
            raise ValueError("stable window ID mismatch")
        if (
            row.pair_token_count
            > row.query_token_count
            + row.passage_token_budget
            + row.pair_special_token_count
            or row.pair_token_count > 512
        ):
            raise ValueError("window pair token count is invalid")
        if not isinstance(row.cache_hit, bool):
            raise ValueError("window cache_hit must be boolean")


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    return (0, int(topic_id)) if topic_id.isdigit() else (1, topic_id)


def _candidate_sort_key(candidate: Mapping[str, object] | object) -> tuple[object, ...]:
    topic_id = str(_candidate_value(candidate, "topic_id"))
    rank = _candidate_value(candidate, "rank")
    rank_value = rank if isinstance(rank, int) and not isinstance(rank, bool) else -1
    return (
        _topic_sort_key(topic_id),
        str(_candidate_value(candidate, "variant")),
        rank_value,
        str(_candidate_value(candidate, "document_id")),
    )


def _nearest_quantile(values: Sequence[float], numerator: int, denominator: int) -> float:
    ordered = sorted(values)
    index = round_half_up_ratio((len(ordered) - 1) * numerator, denominator)
    return ordered[index]


def _group_statistics(
    windows: Sequence[WindowPlanRow],
    documents: Sequence[DocumentWindowDiagnostic],
) -> dict[str, object]:
    cache_keys = {row.cache_key for row in windows}
    hit_keys = {row.cache_key for row in windows if row.cache_hit}
    miss_keys = cache_keys - hit_keys
    coverage = [row.document_token_coverage_fraction for row in documents]
    return {
        "document_count": len(documents),
        "window_count": len(windows),
        "cache_hit_count": sum(row.cache_hit for row in windows),
        "cache_miss_count": sum(not row.cache_hit for row in windows),
        "unique_cache_pair_count": len(cache_keys),
        "unique_cache_hit_count": len(hit_keys),
        "unique_uncached_pair_count": len(miss_keys),
        "capped_document_count": sum(row.capped for row in documents),
        "coverage_min": min(coverage),
        "coverage_median": _nearest_quantile(coverage, 1, 2),
        "coverage_p95": _nearest_quantile(coverage, 95, 100),
    }


def enforce_uncached_pair_ceiling(count: int) -> None:
    if count > MAX_UNCACHED_PAIRS:
        raise ValueError(
            f"uncached pairs exceed the 100,000-miss ceiling: {count:,}"
        )


def build_benchmark_plan(rows: Sequence[WindowPlanRow]) -> dict[str, object]:
    """Freeze the benchmark cache keys without executing a forward pass."""

    misses_by_key: dict[str, WindowPlanRow] = {}
    for row in sorted(rows, key=lambda item: (item.cache_key, item.window_id)):
        if not row.cache_hit:
            misses_by_key.setdefault(row.cache_key, row)
    misses = tuple(misses_by_key.values())
    if not misses:
        mode = "cache_complete"
        warmup: tuple[WindowPlanRow, ...] = ()
        timed: tuple[WindowPlanRow, ...] = ()
        repetitions = 0
    elif len(misses) >= 96:
        mode = "primary"
        warmup = tuple(
            sorted(misses, key=lambda row: (-row.pair_token_count, row.cache_key))[:32]
        )
        warmup_keys = {row.cache_key for row in warmup}
        remainder = tuple(
            sorted(
                (row for row in misses if row.cache_key not in warmup_keys),
                key=lambda row: (row.pair_token_count, row.cache_key),
            )
        )
        indices = tuple(
            round_half_up_ratio(index * (len(remainder) - 1), 63)
            for index in range(64)
        )
        if len(set(indices)) != 64:
            raise ValueError("benchmark 64-pair selection is not unique")
        timed = tuple(remainder[index] for index in indices)
        repetitions = 3
    else:
        mode = "small_sample"
        warmup_count = min(16, len(misses) // 4)
        warmup = tuple(
            sorted(misses, key=lambda row: (-row.pair_token_count, row.cache_key))[
                :warmup_count
            ]
        )
        warmup_keys = {row.cache_key for row in warmup}
        timed = tuple(
            sorted(
                (row for row in misses if row.cache_key not in warmup_keys),
                key=lambda row: (row.pair_token_count, row.cache_key),
            )
        )
        repetitions = 1
    warmup_cache_keys = [row.cache_key for row in warmup]
    timed_cache_keys = [row.cache_key for row in timed]
    identity = {
        "warmup_cache_keys": warmup_cache_keys,
        "timed_cache_keys": timed_cache_keys,
        "timed_repetitions": repetitions,
    }
    return {
        "mode": mode,
        "uncached_pair_count": len(misses),
        "warmup_pair_count": len(warmup),
        "timed_sample_pair_count": len(timed),
        "timed_repetitions": repetitions,
        "forward_pair_count": len(warmup) + len(timed) * repetitions,
        "warmup_cache_keys": warmup_cache_keys,
        "timed_cache_keys": timed_cache_keys,
        "sample_sha256": _sha256_bytes(canonical_compact_json_bytes(identity)),
    }


def build_preflight(
    candidates: Sequence[Mapping[str, object] | object],
    tokenizer: object,
    score_cache: object,
) -> PreflightPlan:
    """Create deterministic windows and cost diagnostics from frozen rows only."""

    materialized_candidates = tuple(candidates)
    _reject_protected_candidates(materialized_candidates)
    ordered = tuple(sorted(materialized_candidates, key=_candidate_sort_key))
    windows: list[WindowPlanRow] = []
    documents: list[DocumentWindowDiagnostic] = []
    observed_candidates: set[tuple[str, str, int, str]] = set()
    for candidate in ordered:
        topic_id = _required_candidate_text(candidate, "topic_id")
        variant = _required_candidate_text(candidate, "variant")
        document_id = _required_candidate_text(candidate, "document_id")
        rank = _candidate_value(candidate, "rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("candidate rank must be an integer")
        identity = (topic_id, variant, rank, document_id)
        if identity in observed_candidates:
            raise ValueError("duplicate preflight candidate identity")
        observed_candidates.add(identity)
        query = _required_candidate_text(candidate, "query")
        candidate_rows = build_window_plan(candidate, tokenizer, query=query)
        cached_rows: list[WindowPlanRow] = []
        for row in candidate_rows:
            cache_key = score_cache.cache_key(  # type: ignore[attr-defined]
                query_text=row.query, text=row.window_text
            )
            if cache_key != row.cache_key:
                raise ValueError("score cache identity differs from frozen preflight context")
            cached_rows.append(
                replace(row, cache_hit=cache_key in score_cache.scores)  # type: ignore[attr-defined]
            )
        verify_window_plan(cached_rows)
        windows.extend(cached_rows)
        first = cached_rows[0]
        documents.append(
            DocumentWindowDiagnostic(
                topic_id=first.topic_id,
                variant=first.variant,
                rank=first.rank,
                document_id=first.document_id,
                document_token_count=first.document_token_count,
                original_window_count=first.original_window_count,
                selected_window_count=first.selected_window_count,
                capped=first.original_window_count > MAX_WINDOWS_PER_DOCUMENT,
                document_token_coverage_fraction=first.document_token_coverage_fraction,
            )
        )
    if not documents:
        raise ValueError("preflight requires at least one candidate")

    window_tuple = tuple(windows)
    document_tuple = tuple(documents)
    by_stream_windows: dict[tuple[str, str], list[WindowPlanRow]] = defaultdict(list)
    by_stream_documents: dict[
        tuple[str, str], list[DocumentWindowDiagnostic]
    ] = defaultdict(list)
    by_topic_windows: dict[str, list[WindowPlanRow]] = defaultdict(list)
    by_topic_documents: dict[str, list[DocumentWindowDiagnostic]] = defaultdict(list)
    for row in window_tuple:
        by_stream_windows[(row.topic_id, row.variant)].append(row)
        by_topic_windows[row.topic_id].append(row)
    for row in document_tuple:
        by_stream_documents[(row.topic_id, row.variant)].append(row)
        by_topic_documents[row.topic_id].append(row)
    streams = tuple(
        {
            "topic_id": key[0],
            "variant": key[1],
            **_group_statistics(by_stream_windows[key], by_stream_documents[key]),
        }
        for key in sorted(by_stream_windows, key=lambda key: (_topic_sort_key(key[0]), key[1]))
    )
    topics = tuple(
        {
            "topic_id": topic_id,
            **_group_statistics(by_topic_windows[topic_id], by_topic_documents[topic_id]),
        }
        for topic_id in sorted(by_topic_windows, key=_topic_sort_key)
    )
    summary = {
        **_group_statistics(window_tuple, document_tuple),
        "stream_count": len(streams),
        "topic_count": len(topics),
        "max_query_token_count": max(row.query_token_count for row in window_tuple),
        "max_document_token_count": max(
            row.document_token_count for row in window_tuple
        ),
        "max_pair_token_count": max(row.pair_token_count for row in window_tuple),
        "inference_count": 0,
        "qrels_access_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
    }
    enforce_uncached_pair_ceiling(int(summary["unique_uncached_pair_count"]))
    benchmark = build_benchmark_plan(window_tuple)
    return PreflightPlan(
        windows=window_tuple,
        documents=document_tuple,
        streams=streams,
        topics=topics,
        summary=summary,
        benchmark=benchmark,
    )


def _window_jsonl_bytes(rows: Sequence[WindowPlanRow]) -> bytes:
    return b"".join(
        canonical_compact_json_bytes(row.to_dict()) + b"\n" for row in rows
    )


def _materialization_payload(path: Path) -> dict[str, object]:
    source = Path(path).read_bytes()
    return _loads_object_no_duplicates(source, "model materialization receipt")


def run_preflight(
    *,
    manifest_path: Path,
    source_dir: Path,
    materialization_receipt_path: Path,
    score_cache_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run and persist the complete tokenizer-only, qrels-free preflight."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output already exists: {destination}")
    manifest: FacetLocalManifest = load_facet_local_manifest(Path(manifest_path))
    rows, source_receipt = load_facet_local_source_snapshot(Path(source_dir), manifest)
    _reject_protected_candidates(rows)
    tokenizer = load_verified_tokenizer(Path(materialization_receipt_path))
    cache = GlobalScoreCache(Path(score_cache_root), score_cache_context())
    plan = build_preflight(rows, tokenizer, cache)
    windows_bytes = _window_jsonl_bytes(plan.windows)
    materialization = _materialization_payload(Path(materialization_receipt_path))
    context = score_cache_context()
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "manifest_file": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256_file(Path(manifest_path)),
        "source_dir": str(Path(source_dir).resolve()),
        "source_receipt_sha256": manifest.source_receipt_sha256,
        "source_candidates_sha256": manifest.candidates_sha256,
        "source_candidate_rows": manifest.candidate_rows,
        "model_materialization_receipt": str(
            Path(materialization_receipt_path).resolve()
        ),
        "model_materialization_receipt_sha256": _sha256_file(
            Path(materialization_receipt_path)
        ),
        "model_materialization": materialization,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        },
        "score_cache": {
            "root": str(Path(score_cache_root).resolve()),
            "path": str(cache.path.resolve()),
            "context": context.artifact_metadata,
        },
        "window_policy": {
            "version": WINDOW_POLICY_VERSION,
            "pair_max_tokens": PAIR_MAX_TOKENS,
            "query_max_tokens": QUERY_MAX_TOKENS,
            "minimum_passage_tokens": MIN_PASSAGE_TOKENS,
            "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "maximum_windows_per_document": MAX_WINDOWS_PER_DOCUMENT,
            "capped_index_rule": "round_half_up(j * (N - 1) / 31), j=0..31",
            "maximum_uncached_pairs": MAX_UNCACHED_PAIRS,
            "coverage_p95_rule": "nearest-rank half-up over sorted document fractions",
        },
        "windows_file": "windows.jsonl",
        "windows_bytes": len(windows_bytes),
        "windows_sha256": _sha256_bytes(windows_bytes),
        "documents": [row.to_dict() for row in plan.documents],
        "streams": list(plan.streams),
        "topics": list(plan.topics),
        "summary": plan.summary,
        "benchmark": plan.benchmark,
        "inference_authorized": False,
        "model_constructed": False,
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
    }
    # Retain the authenticated source receipt binding without reopening any ledger.
    if source_receipt.get("candidates_sha256") != payload["source_candidates_sha256"]:
        raise ValueError("preflight source receipt binding changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"create-only preflight output already exists: {destination}"
        ) from exc
    _exclusive_write(destination / "windows.jsonl", windows_bytes)
    _exclusive_write(destination / "preflight.json", _canonical_pretty_json_bytes(payload))
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize pinned MiniLM files or run tokenizer-only preflight"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--score-cache", type=Path)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--model-id", required=True)
    materialize.add_argument("--revision", required=True)
    materialize.add_argument("--approval", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "materialize":
        receipt = materialize_model(
            model_id=args.model_id,
            revision=args.revision,
            approval_path=args.approval,
            output_dir=args.output,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    required = {
        "--manifest": args.manifest,
        "--source": args.source,
        "--model-receipt": args.model_receipt,
        "--score-cache": args.score_cache,
        "--output": args.output,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error("preflight requires " + ", ".join(missing))
    payload = run_preflight(
        manifest_path=args.manifest,
        source_dir=args.source,
        materialization_receipt_path=args.model_receipt,
        score_cache_root=args.score_cache,
        output_dir=args.output,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

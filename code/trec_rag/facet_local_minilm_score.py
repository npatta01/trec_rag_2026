"""Approval-gated, local-only MiniLM benchmark and scoring runner.

The scorer accepts only the authenticated tokenizer preflight-v2 window plan.
Model construction and every forward pass remain behind an exact receipt.  It
has no retrieval, qrels, hosted-inference, or network interface.

This deliberately remains a frozen one-module runner so the approval, cache,
and artifact invariants can be audited together.  The trade-off is module-size
risk: future behavior should be added only through a separately reviewed file
map rather than allowing this safety boundary to grow without review.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import statistics
import struct
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .facet_local_minilm_manifest import PROTECTED_TOPIC_IDS
from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    PREFLIGHT_SCHEMA_VERSION,
    WINDOW_POLICY_VERSION,
    WindowPlanRow,
    build_benchmark_plan,
    canonical_compact_json_bytes,
    load_verified_materialization,
    score_cache_context,
    verify_window_plan_structure,
)
from .rerank_score_cache import GlobalScoreCache, ScoreCacheContext


BATCH_SIZE = 32
MAX_BENCHMARK_PAIRS = 256

BENCHMARK_APPROVAL_SCHEMA_VERSION = "facet-local-minilm-benchmark-approval-v2"
BENCHMARK_APPROVAL_SCOPE = "facet_local_minilm_rocm_benchmark_v2"
BENCHMARK_CPU_APPROVAL_SCHEMA_VERSION = (
    "facet-local-minilm-benchmark-cpu-fallback-approval-v2"
)
BENCHMARK_CPU_APPROVAL_SCOPE = "facet_local_minilm_cpu_benchmark_fallback_v2"
FULL_APPROVAL_SCHEMA_VERSION = "facet-local-minilm-full-inference-approval-v2"
FULL_APPROVAL_SCOPE = "facet_local_minilm_rocm_full_inference_v2"
FULL_CPU_APPROVAL_SCHEMA_VERSION = (
    "facet-local-minilm-full-inference-cpu-fallback-approval-v2"
)
FULL_CPU_APPROVAL_SCOPE = "facet_local_minilm_cpu_full_inference_fallback_v2"

BENCHMARK_TELEMETRY_SCHEMA_VERSION = "facet-local-minilm-benchmark-telemetry-v2"
FULL_REQUEST_SCHEMA_VERSION = "facet-local-minilm-full-inference-request-v2"
SCORE_ROW_SCHEMA_VERSION = "facet-local-minilm-score-row-v2"
SCORING_RECEIPT_SCHEMA_VERSION = "facet-local-minilm-scoring-receipt-v2"
RUN_RESERVATION_SCHEMA_VERSION = "facet-local-minilm-run-reservation-v1"
RUN_TERMINAL_SCHEMA_VERSION = "facet-local-minilm-run-terminal-v1"
CACHE_TRANSACTION_SCHEMA_VERSION = "facet-local-minilm-cache-transaction-v1"

BENCHMARK_ACTION = "benchmark"
FULL_SCORING_ACTION = "full_scoring"

_BENCHMARK_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "approval_scope",
        "action",
        "run_id",
        "output_path",
        "approved_by",
        "device",
        "execution_backend",
        "pair_limit",
        "preflight_sha256",
        "windows_sha256",
        "model_materialization_receipt_sha256",
        "benchmark_sample_sha256",
        "acknowledged_maximum_256_pairs",
        "acknowledged_local_files_only",
        "acknowledged_no_cpu_fallback",
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference",
    }
)
_BENCHMARK_CPU_APPROVAL_FIELDS = frozenset(
    (_BENCHMARK_APPROVAL_FIELDS - {"acknowledged_no_cpu_fallback"})
    | {"acknowledged_cpu_fallback"}
)
_FULL_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "approval_scope",
        "action",
        "run_id",
        "output_path",
        "approved_by",
        "device",
        "execution_backend",
        "preflight_sha256",
        "windows_sha256",
        "model_materialization_receipt_sha256",
        "benchmark_sample_sha256",
        "full_inference_request_file",
        "full_inference_request_sha256",
        "benchmark_telemetry_file",
        "benchmark_telemetry_sha256",
        "score_cache_path",
        "batch_size",
        "acknowledged_projected_runtime",
        "acknowledged_local_files_only",
        "acknowledged_no_cpu_fallback",
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference",
    }
)
_FULL_CPU_APPROVAL_FIELDS = frozenset(
    (_FULL_APPROVAL_FIELDS - {"acknowledged_no_cpu_fallback"})
    | {"acknowledged_cpu_fallback"}
)
_FULL_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "preflight_sha256",
        "windows_sha256",
        "model_materialization_receipt_sha256",
        "benchmark_telemetry_file",
        "benchmark_telemetry_sha256",
        "benchmark_sample_sha256",
        "device",
        "execution_backend",
        "batch_size",
        "uncached_pair_count",
        "score_cache_path",
        "projected_full_run_wall_seconds",
        "peak_device_memory_bytes",
        "peak_host_memory_bytes",
        "cpu_fallback_authorized",
        "qrels_path_supported",
        "retrieval_path_supported",
        "network_access_supported",
        "hosted_inference_supported",
    }
)

_BENCHMARK_TELEMETRY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "action",
        "run_id",
        "output_path",
        "preflight_sha256",
        "windows_sha256",
        "model_materialization_receipt_sha256",
        "benchmark_approval_sha256",
        "benchmark_sample_sha256",
        "mode",
        "device",
        "execution_backend",
        "device_probe",
        "batch_size",
        "uncached_pair_count",
        "warmup_pair_count",
        "timed_sample_pair_count",
        "forward_pair_count",
        "timed_scope",
        "timed_repetitions",
        "median_pairs_per_second",
        "projection_multiplier",
        "projected_unique_scoring_seconds",
        "fixed_setup_seconds",
        "fixed_finalize_seconds",
        "projected_full_run_wall_seconds",
        "peak_device_memory_bytes",
        "peak_host_memory_bytes",
    }
)

_CACHE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "backend",
        "backend_version",
        "model",
        "model_revision",
        "score_representation",
        "inference_dtype",
        "input_policy",
        "max_length",
        "score_kind",
        "cache_key",
        "query_sha256",
        "text_sha256",
        "score",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
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
        value = json.loads(
            content,
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains nonstandard JSON constant {constant}")
            ),
        )
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _score_cache_file_binding(path: Path) -> dict[str, object]:
    cache_path = Path(path).resolve()
    if not cache_path.exists():
        if os.path.lexists(cache_path):
            raise ValueError("score cache path is not a readable regular file")
        return {
            "state": "absent",
            "path": str(cache_path),
            "bytes": 0,
            "sha256": None,
        }
    if not cache_path.is_file() or not os.access(cache_path, os.R_OK):
        raise ValueError("score cache path is not a readable regular file")
    source = cache_path.read_bytes()
    return {
        "state": "present",
        "path": str(cache_path),
        "bytes": len(source),
        "sha256": _sha256_bytes(source),
    }


def _validate_score_cache_binding(
    value: object,
    *,
    expected_path: Path,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("preflight score-cache file binding is invalid")
    binding = dict(value)
    if set(binding) != {"state", "path", "bytes", "sha256"}:
        raise ValueError("preflight score-cache file binding fields mismatch")
    if binding.get("path") != str(expected_path.resolve()):
        raise ValueError("preflight score-cache file binding path mismatch")
    state = binding.get("state")
    if state == "absent":
        if binding.get("bytes") != 0 or binding.get("sha256") is not None:
            raise ValueError("preflight absent score-cache binding is invalid")
    elif state == "present":
        if (
            not isinstance(binding.get("bytes"), int)
            or isinstance(binding.get("bytes"), bool)
            or int(binding["bytes"]) < 0
            or not _is_sha256(binding.get("sha256"))
        ):
            raise ValueError("preflight present score-cache binding is invalid")
    else:
        raise ValueError("preflight score-cache binding state is invalid")
    return binding


def score_cache_key(
    context: ScoreCacheContext,
    *,
    query: str,
    window: str,
) -> str:
    """Return the exact GlobalScoreCache v2 content identity."""

    payload = {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": context.backend,
        "model": context.model,
        "max_length": context.max_length,
        "score_kind": context.score_kind,
        **context.cache_identity_metadata,
        "query_sha256": _sha256_bytes(query.encode("utf-8")),
        "text_sha256": _sha256_bytes(window.encode("utf-8")),
    }
    return _sha256_bytes(canonical_compact_json_bytes(payload))


@dataclass(frozen=True)
class InferenceRuntime:
    """Late-bound local inference dependencies; injectable at hardware edges."""

    torch: object
    auto_tokenizer_cls: object
    auto_model_cls: object
    clock: Callable[[], float]
    host_memory_bytes: Callable[[], int]
    rocm_probe: Callable[[object, str], Mapping[str, object]]


@dataclass(frozen=True)
class ScoringInputs:
    preflight: dict[str, object]
    preflight_path: Path | None
    preflight_source: bytes
    preflight_sha256: str
    windows_sha256: str
    materialization_receipt_sha256: str
    snapshot_path: Path
    windows: tuple[object, ...]
    benchmark: dict[str, object]
    context: ScoreCacheContext
    score_cache_root: Path
    score_cache_path: Path
    score_cache_binding: dict[str, object]


@dataclass(frozen=True)
class MiniLMScoreRow:
    topic_id: str
    family: str
    variant: str
    rank: int
    document_id: str
    window_id: str
    query_sha256: str
    window_sha256: str
    cache_key: str
    reservation_sha256: str
    disposition: str
    score: float
    elapsed_seconds: float
    peak_device_memory_bytes: int
    peak_host_memory_bytes: int
    model: str
    model_revision: str
    score_representation: str
    inference_dtype: str
    raw_output_sha256: str
    output_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": SCORE_ROW_SCHEMA_VERSION, **self.__dict__}


def _group_and_verify_windows(windows: Sequence[WindowPlanRow]) -> None:
    grouped: dict[tuple[object, ...], list[WindowPlanRow]] = defaultdict(list)
    for row in windows:
        grouped[
            (row.topic_id, row.variant, row.rank, row.document_id)
        ].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.selected_window_index)
        verify_window_plan_structure(rows)


def _validate_input_safety(inputs: ScoringInputs) -> None:
    preflight = inputs.preflight
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("canonical facet-local-minilm-preflight-v2 is required")
    if preflight.get("status") != "tokenizer_only_preflight_complete":
        raise ValueError("scoring preflight status mismatch")
    for field in (
        "inference_authorized",
        "model_constructed",
        "qrels_path_supported",
        "retrieval_path_supported",
    ):
        if preflight.get(field) is not False:
            raise ValueError(f"preflight {field} must remain false")
    summary = preflight.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("preflight summary is invalid")
    counters = {
        "qrels_access_count": "qrels access counter",
        "retrieval_call_count": "retrieval call counter",
        "hosted_inference_call_count": "hosted inference counter",
        "inference_count": "inference counter",
    }
    for field, label in counters.items():
        if summary.get(field) != 0:
            raise ValueError(f"preflight {label} must be zero")
    materialization = preflight.get("model_materialization")
    if not isinstance(materialization, Mapping):
        raise ValueError("preflight model materialization binding is invalid")
    if (
        materialization.get("model_id") != MODEL_ID
        or materialization.get("revision") != MODEL_REVISION
    ):
        raise ValueError("preflight model/revision differs from pinned MiniLM")
    if inputs.context != score_cache_context():
        raise ValueError("preflight score-cache context differs from pinned context")
    if inputs.context.inference_dtype != "float32":
        raise ValueError("MiniLM inference dtype must be float32")
    if inputs.context.score_representation != "raw_logits":
        raise ValueError("MiniLM score representation must be raw_logits")
    expected_path = inputs.score_cache_root.joinpath(*inputs.context.path_parts).resolve()
    if inputs.score_cache_path.resolve() != expected_path:
        raise ValueError("preflight score-cache path differs from pinned context")
    _validate_score_cache_binding(
        inputs.score_cache_binding,
        expected_path=inputs.score_cache_path,
    )

    for collection_name in ("topics", "streams"):
        collection = preflight.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"preflight {collection_name} are invalid")
        for row in collection:
            if isinstance(row, Mapping):
                topic_id = str(row.get("topic_id", ""))
                if topic_id in PROTECTED_TOPIC_IDS:
                    raise ValueError(f"protected topic {topic_id} is forbidden")
    for row in inputs.windows:
        topic_id = str(getattr(row, "topic_id", ""))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        query = getattr(row, "query", None)
        window = getattr(row, "window_text", None)
        if not isinstance(query, str) or not isinstance(window, str):
            raise ValueError("window query/text is invalid")
        if getattr(row, "cache_key", None) != score_cache_key(
            inputs.context, query=query, window=window
        ):
            raise ValueError("window cache key differs from pinned identity")
        if int(getattr(row, "pair_token_count", 0)) > PAIR_MAX_TOKENS:
            raise ValueError("window pair exceeds 512 tokens")

    if summary.get("window_count") != len(inputs.windows):
        raise ValueError("preflight window count mismatch")
    if preflight.get("benchmark") != inputs.benchmark:
        raise ValueError("preflight benchmark binding mismatch")


def load_scoring_inputs(preflight_path: Path) -> ScoringInputs:
    """Authenticate preflight-v2, windows, materialization, and cache identity."""

    path = Path(preflight_path).resolve()
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError("scoring preflight is required") from exc
    preflight = _loads_object_no_duplicates(source, "scoring preflight")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("canonical facet-local-minilm-preflight-v2 is required")
    if _pretty_json_bytes(preflight) != source:
        raise ValueError("scoring preflight must be canonical JSON bytes")

    windows_file = preflight.get("windows_file")
    if not isinstance(windows_file, str) or not windows_file:
        raise ValueError("preflight windows file is invalid")
    windows_path = Path(windows_file)
    if not windows_path.is_absolute():
        windows_path = path.parent / windows_path
    try:
        windows_source = windows_path.read_bytes()
    except OSError as exc:
        raise ValueError("preflight windows are required") from exc
    if (
        preflight.get("windows_bytes") != len(windows_source)
        or preflight.get("windows_sha256") != _sha256_bytes(windows_source)
    ):
        raise ValueError("preflight windows bytes/hash mismatch")
    windows: list[WindowPlanRow] = []
    for line_number, line in enumerate(windows_source.splitlines(), start=1):
        if not line:
            raise ValueError(f"preflight windows row {line_number} is empty")
        row = _loads_object_no_duplicates(line, f"preflight windows row {line_number}")
        if canonical_compact_json_bytes(row) != line:
            raise ValueError("preflight windows must be canonical JSONL bytes")
        windows.append(WindowPlanRow.from_dict(row))
    _group_and_verify_windows(windows)

    score_cache = preflight.get("score_cache")
    if not isinstance(score_cache, Mapping):
        raise ValueError("preflight score-cache binding is invalid")
    context = score_cache_context()
    if score_cache.get("context") != context.artifact_metadata:
        raise ValueError("preflight score-cache context mismatch")
    root_value = score_cache.get("root")
    path_value = score_cache.get("path")
    if not isinstance(root_value, str) or not isinstance(path_value, str):
        raise ValueError("preflight score-cache root/path is invalid")
    root = Path(root_value).resolve()
    cache_path = Path(path_value).resolve()
    if cache_path != root.joinpath(*context.path_parts).resolve():
        raise ValueError("preflight score-cache path differs from pinned context")
    cache_binding = _validate_score_cache_binding(
        score_cache.get("binding"),
        expected_path=cache_path,
    )

    receipt_value = preflight.get("model_materialization_receipt")
    receipt_sha256 = preflight.get("model_materialization_receipt_sha256")
    if not isinstance(receipt_value, str) or not _is_sha256(receipt_sha256):
        raise ValueError("preflight model materialization receipt binding is invalid")
    receipt_path = Path(receipt_value).resolve()
    if _sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("preflight model materialization receipt hash mismatch")
    verified = load_verified_materialization(receipt_path)
    if verified.sha256 != receipt_sha256:
        raise ValueError("verified model materialization receipt hash mismatch")
    if preflight.get("model_materialization") != verified.payload:
        raise ValueError("preflight model materialization payload mismatch")

    benchmark = preflight.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("preflight benchmark plan is invalid")
    recomputed = build_benchmark_plan(windows)
    if benchmark != recomputed:
        raise ValueError("benchmark sample hash differs from preflight windows")
    inputs = ScoringInputs(
        preflight=preflight,
        preflight_path=path,
        preflight_source=source,
        preflight_sha256=_sha256_bytes(source),
        windows_sha256=_sha256_bytes(windows_source),
        materialization_receipt_sha256=verified.sha256,
        snapshot_path=verified.snapshot,
        windows=tuple(windows),
        benchmark=dict(benchmark),
        context=context,
        score_cache_root=root,
        score_cache_path=cache_path,
        score_cache_binding=cache_binding,
    )
    _validate_input_safety(inputs)
    return inputs


def _coerce_inputs(value: ScoringInputs | Path | str) -> ScoringInputs:
    if isinstance(value, ScoringInputs):
        return value
    return load_scoring_inputs(Path(value))


def _load_receipt(
    value: Mapping[str, object] | Path | str | None,
    *,
    expected_sha256: str | None,
    label: str,
    required_message: str,
) -> tuple[dict[str, object], bytes, Path]:
    if value is None:
        raise ValueError(required_message)
    if isinstance(value, Mapping):
        raise ValueError(f"canonical file-backed {label} is required")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"out-of-band expected {label} SHA-256 is required")
    path = Path(value).resolve()
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(required_message) from exc
    if _sha256_bytes(source) != expected_sha256:
        raise ValueError(f"{label} differs from out-of-band expected SHA-256")
    receipt = _loads_object_no_duplicates(source, label)
    if source != _pretty_json_bytes(receipt):
        raise ValueError(f"{label} must be canonical JSON bytes")
    return receipt, source, path


def _require_run_binding(
    receipt: Mapping[str, object],
    *,
    label: str,
    action: str,
    output_dir: Path,
) -> str:
    if receipt.get("action") != action:
        raise ValueError(f"{label} action mismatch")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{label} run_id must be nonempty")
    if receipt.get("output_path") != str(Path(output_dir).resolve()):
        raise ValueError(f"{label} resolved output path mismatch")
    return run_id


def _require_nonempty_approver(receipt: Mapping[str, object], label: str) -> None:
    approved_by = receipt.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError(f"{label} approved_by must be nonempty")


def _validate_benchmark_approval(
    inputs: ScoringInputs,
    approval: Mapping[str, object] | Path | str | None,
    *,
    expected_approval_sha256: str | None,
    pair_limit: int,
    device: str,
    output_dir: Path,
) -> tuple[dict[str, object], str, Path]:
    receipt, source, approval_path = _load_receipt(
        approval,
        expected_sha256=expected_approval_sha256,
        label="benchmark approval receipt",
        required_message="explicit benchmark approval receipt is required",
    )
    if device == "cpu":
        fields = _BENCHMARK_CPU_APPROVAL_FIELDS
        schema = BENCHMARK_CPU_APPROVAL_SCHEMA_VERSION
        scope = BENCHMARK_CPU_APPROVAL_SCOPE
        acknowledgement = "acknowledged_cpu_fallback"
        backend = "cpu"
    else:
        fields = _BENCHMARK_APPROVAL_FIELDS
        schema = BENCHMARK_APPROVAL_SCHEMA_VERSION
        scope = BENCHMARK_APPROVAL_SCOPE
        acknowledgement = "acknowledged_no_cpu_fallback"
        backend = "rocm"
    if set(receipt) != fields:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("benchmark approval fields mismatch")
    if receipt.get("schema_version") != schema or receipt.get("approval_scope") != scope:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("benchmark approval schema/scope mismatch")
    _require_run_binding(
        receipt,
        label="benchmark approval",
        action=BENCHMARK_ACTION,
        output_dir=output_dir,
    )
    _require_nonempty_approver(receipt, "benchmark approval")
    if receipt.get("device") != device or receipt.get("execution_backend") != backend:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("benchmark approval device/backend mismatch")
    approved_limit = receipt.get("pair_limit")
    if (
        not isinstance(approved_limit, int)
        or isinstance(approved_limit, bool)
        or approved_limit > MAX_BENCHMARK_PAIRS
        or pair_limit > approved_limit
    ):
        raise ValueError("benchmark approval pair limit mismatch")
    bindings = {
        "preflight_sha256": "benchmark approval preflight hash mismatch",
        "windows_sha256": "benchmark approval windows hash mismatch",
        "model_materialization_receipt_sha256": (
            "benchmark approval model receipt hash mismatch"
        ),
        "benchmark_sample_sha256": "benchmark approval sample hash mismatch",
    }
    expected = {
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
    }
    for field, message in bindings.items():
        if receipt.get(field) != expected[field]:
            raise ValueError(message)
    for field in (
        "acknowledged_maximum_256_pairs",
        "acknowledged_local_files_only",
        acknowledgement,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"benchmark approval must acknowledge {field}")
    return receipt, _sha256_bytes(source), approval_path


def _reservation_payload(window: object, sequence_index: int) -> dict[str, object]:
    return {
        "sequence_index": sequence_index,
        "topic_id": str(getattr(window, "topic_id")),
        "family": str(getattr(window, "family")),
        "variant": str(getattr(window, "variant")),
        "rank": int(getattr(window, "rank")),
        "document_id": str(getattr(window, "document_id")),
        "window_id": str(getattr(window, "window_id")),
        "cache_key": str(getattr(window, "cache_key")),
    }


def _planned_reservation_records(
    windows: Sequence[object],
) -> tuple[tuple[dict[str, object], ...], str]:
    records: list[dict[str, object]] = []
    sequence_hashes: list[str] = []
    for index, window in enumerate(windows):
        payload = _reservation_payload(window, index)
        reservation_sha256 = _sha256_bytes(canonical_compact_json_bytes(payload))
        records.append({**payload, "reservation_sha256": reservation_sha256})
        sequence_hashes.append(reservation_sha256)
    root = _sha256_bytes(
        b"".join((value + "\n").encode("ascii") for value in sequence_hashes)
    )
    return tuple(records), root


@dataclass(frozen=True)
class _RunReservation:
    destination: Path
    action: str
    run_id: str
    approval_sha256: str
    reservation_sequence_root_sha256: str
    planned_row_count: int


def _reserve_run(
    *,
    output_dir: Path,
    action: str,
    run_id: str,
    approval_sha256: str,
    inputs: ScoringInputs,
    planned_windows: Sequence[object],
) -> _RunReservation:
    destination = Path(output_dir).resolve()
    try:
        destination.mkdir(parents=True)
    except FileExistsError as exc:
        output_label = "scoring" if action == FULL_SCORING_ACTION else action
        raise FileExistsError(
            f"create-only {output_label} output already exists: {destination}"
        ) from exc
    records, sequence_root = _planned_reservation_records(planned_windows)
    records_bytes = b"".join(
        canonical_compact_json_bytes(record) + b"\n" for record in records
    )
    _exclusive_write(destination / "planned_reservations.jsonl", records_bytes)
    payload = {
        "schema_version": RUN_RESERVATION_SCHEMA_VERSION,
        "status": "reserved",
        "action": action,
        "run_id": run_id,
        "output_path": str(destination),
        "approval_sha256": approval_sha256,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
        "planned_row_count": len(records),
        "planned_reservations_bytes": len(records_bytes),
        "planned_reservations_sha256": _sha256_bytes(records_bytes),
        "reservation_sequence_root_sha256": sequence_root,
    }
    _exclusive_write(destination / "run_reservation.json", _pretty_json_bytes(payload))
    _fsync_directory(destination)
    return _RunReservation(
        destination=destination,
        action=action,
        run_id=run_id,
        approval_sha256=approval_sha256,
        reservation_sequence_root_sha256=sequence_root,
        planned_row_count=len(records),
    )


def _consume_approval(
    inputs: ScoringInputs,
    reservation: _RunReservation,
    approval_path: Path,
) -> Path:
    registry = inputs.score_cache_root / ".facet-local-minilm-approval-consumptions"
    registry.mkdir(parents=True, exist_ok=True)
    marker = registry / f"{reservation.approval_sha256}.json"
    payload = {
        "schema_version": "facet-local-minilm-approval-consumption-v1",
        "status": "consumed",
        "action": reservation.action,
        "run_id": reservation.run_id,
        "approval_file": str(Path(approval_path).resolve()),
        "approval_sha256": reservation.approval_sha256,
        "output_path": str(reservation.destination),
    }
    try:
        _exclusive_write(marker, _pretty_json_bytes(payload))
    except FileExistsError as exc:
        raise ValueError("approval receipt replay rejected") from exc
    _fsync_directory(registry)
    return marker


def _seal_terminal(
    reservation: _RunReservation,
    *,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "schema_version": RUN_TERMINAL_SCHEMA_VERSION,
        "status": status,
        "action": reservation.action,
        "run_id": reservation.run_id,
        "approval_sha256": reservation.approval_sha256,
        "output_path": str(reservation.destination),
        **dict(details or {}),
    }
    _exclusive_write(
        reservation.destination / "run_terminal.json",
        _pretty_json_bytes(payload),
    )
    _fsync_directory(reservation.destination)


def _seal_failure(reservation: _RunReservation, exc: BaseException) -> None:
    try:
        _seal_terminal(
            reservation,
            status="failed",
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
    except BaseException:
        # Preserve the originating failure; an existing terminal is itself a
        # durable replay/error signal and must never be overwritten.
        pass


def _default_host_memory_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  This host contract is Linux/ROCm.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _probe_rocm(torch_module: object, device: str) -> Mapping[str, object]:
    if device != "cuda":
        raise ValueError("ROCm probe requires the torch cuda device spelling")
    version = getattr(torch_module, "version", None)
    hip = getattr(version, "hip", None)
    cuda = getattr(torch_module, "cuda", None)
    if not hip or cuda is None or not cuda.is_available():  # type: ignore[attr-defined]
        raise RuntimeError("ROCm probe failed before model load")
    if int(cuda.device_count()) < 1:  # type: ignore[attr-defined]
        raise RuntimeError("ROCm probe found no device before model load")
    return {
        "device_name": str(cuda.get_device_name(0)),  # type: ignore[attr-defined]
        "torch_hip_version": str(hip),
    }


def _default_runtime() -> InferenceRuntime:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return InferenceRuntime(
        torch=torch,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForSequenceClassification,
        clock=time.perf_counter,
        host_memory_bytes=_default_host_memory_bytes,
        rocm_probe=_probe_rocm,
    )


def _load_local_model(
    inputs: ScoringInputs,
    runtime: InferenceRuntime,
    *,
    device: str,
) -> tuple[object, object, dict[str, object]]:
    if device == "cuda":
        probe = dict(runtime.rocm_probe(runtime.torch, device))
    elif device == "cpu":
        probe = {"device_name": "cpu", "torch_hip_version": None}
    else:
        raise ValueError("device must be cuda or cpu")
    tokenizer = runtime.auto_tokenizer_cls.from_pretrained(  # type: ignore[attr-defined]
        inputs.snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    model = runtime.auto_model_cls.from_pretrained(  # type: ignore[attr-defined]
        inputs.snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=runtime.torch.float32,  # type: ignore[attr-defined]
    )
    model = model.float()  # type: ignore[attr-defined]
    model = model.eval()  # type: ignore[attr-defined]
    model = model.to(device)  # type: ignore[attr-defined]
    return tokenizer, model, probe


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _float32(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("MiniLM raw logit must be finite")
    return struct.unpack(">f", struct.pack(">f", converted))[0]


@dataclass(frozen=True)
class _CacheSnapshot:
    source: bytes
    binding: dict[str, object]
    rows_by_key: dict[str, dict[str, object]]
    scores: dict[str, float]


def _cache_transaction_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.facet-local-minilm-transaction.json")


def _derived_cache_key_from_row(
    context: ScoreCacheContext,
    row: Mapping[str, object],
) -> str:
    payload = {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": context.backend,
        "model": context.model,
        "max_length": context.max_length,
        "score_kind": context.score_kind,
        **context.cache_identity_metadata,
        "query_sha256": row.get("query_sha256"),
        "text_sha256": row.get("text_sha256"),
    }
    return _sha256_bytes(canonical_compact_json_bytes(payload))


def _load_bound_cache(inputs: ScoringInputs) -> _CacheSnapshot:
    transaction_path = _cache_transaction_path(inputs.score_cache_path)
    if transaction_path.exists() or os.path.lexists(transaction_path):
        raise ValueError("unsealed score-cache transaction is forbidden")
    current = _score_cache_file_binding(inputs.score_cache_path)
    if current != inputs.score_cache_binding:
        raise ValueError("score cache changed after preflight")
    if current["state"] == "absent":
        return _CacheSnapshot(
            source=b"",
            binding=current,
            rows_by_key={},
            scores={},
        )
    source = inputs.score_cache_path.read_bytes()
    if (
        len(source) != current["bytes"]
        or _sha256_bytes(source) != current["sha256"]
    ):
        raise ValueError("score cache changed while authenticating preflight binding")
    rows_by_key: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    expected_context = {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": inputs.context.backend,
        "backend_version": inputs.context.backend_version,
        "model": inputs.context.model,
        "model_revision": inputs.context.model_revision,
        "score_representation": inputs.context.score_representation,
        "inference_dtype": inputs.context.inference_dtype,
        "input_policy": inputs.context.input_policy,
        "max_length": inputs.context.max_length,
        "score_kind": inputs.context.score_kind,
    }
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            raise ValueError(f"score cache row {line_number} is empty")
        row = _loads_object_no_duplicates(line, f"score cache row {line_number}")
        if set(row) != _CACHE_ROW_FIELDS:
            raise ValueError(f"score cache row {line_number} fields mismatch")
        for field, expected in expected_context.items():
            if row.get(field) != expected:
                raise ValueError("score cache row context mismatch")
        if not _is_sha256(row.get("query_sha256")) or not _is_sha256(
            row.get("text_sha256")
        ):
            raise ValueError("score cache row query/text hash is invalid")
        cache_key = row.get("cache_key")
        if not _is_sha256(cache_key) or cache_key != _derived_cache_key_from_row(
            inputs.context, row
        ):
            raise ValueError("score cache row derived key mismatch")
        raw_score = row.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("score cache row score must be a finite float32")
        score = float(raw_score)
        if not math.isfinite(score) or _float32(score) != score:
            raise ValueError("score cache row score must be a finite float32")
        if cache_key in scores and scores[cache_key] != score:
            raise ValueError("score cache has conflicting duplicate score")
        scores[str(cache_key)] = score
        rows_by_key.setdefault(str(cache_key), row)
    return _CacheSnapshot(
        source=source,
        binding=current,
        rows_by_key=rows_by_key,
        scores=scores,
    )


def _cache_row(
    context: ScoreCacheContext,
    *,
    query: str,
    window: str,
    score: float,
) -> dict[str, object]:
    cache_key = score_cache_key(context, query=query, window=window)
    return {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": context.backend,
        "model": context.model,
        "max_length": context.max_length,
        "score_kind": context.score_kind,
        **context.cache_identity_metadata,
        "cache_key": cache_key,
        "query_sha256": _sha256_bytes(query.encode("utf-8")),
        "text_sha256": _sha256_bytes(window.encode("utf-8")),
        "score": _float32(score),
    }


class _CacheWriterLock:
    def __init__(self, cache_path: Path) -> None:
        self.path = cache_path.with_name(f"{cache_path.name}.lock")
        self.descriptor: int | None = None

    def __enter__(self) -> "_CacheWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_args: object) -> None:
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


def _write_cache_replacement(
    inputs: ScoringInputs,
    reservation: _RunReservation,
    additions: Sequence[tuple[str, str, float]],
) -> tuple[_CacheSnapshot, _CacheSnapshot, Path | None]:
    with _CacheWriterLock(inputs.score_cache_path):
        before = _load_bound_cache(inputs)
        if not additions:
            return before, before, None
        rows_by_key = dict(before.rows_by_key)
        for query, window, raw_score in additions:
            row = _cache_row(
                inputs.context,
                query=query,
                window=window,
                score=raw_score,
            )
            cache_key = str(row["cache_key"])
            existing = rows_by_key.get(cache_key)
            if existing is not None:
                if float(existing["score"]) != float(row["score"]):
                    raise ValueError("conflicting score for existing global cache key")
                continue
            rows_by_key[cache_key] = row
        content = b"".join(
            canonical_compact_json_bytes(rows_by_key[key]) + b"\n"
            for key in sorted(rows_by_key)
        )
        after_binding = {
            "state": "present",
            "path": str(inputs.score_cache_path),
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        transaction_path = _cache_transaction_path(inputs.score_cache_path)
        transaction = {
            "schema_version": CACHE_TRANSACTION_SCHEMA_VERSION,
            "status": "prepared",
            "action": reservation.action,
            "run_id": reservation.run_id,
            "approval_sha256": reservation.approval_sha256,
            "output_path": str(reservation.destination),
            "cache_before": before.binding,
            "cache_after": after_binding,
            "cache_before_count": len(before.rows_by_key),
            "cache_after_count": len(rows_by_key),
        }
        _exclusive_write(transaction_path, _pretty_json_bytes(transaction))
        staged = inputs.score_cache_path.with_name(
            f".{inputs.score_cache_path.name}.{reservation.run_id}.staged"
        )
        try:
            _exclusive_write(staged, content)
            os.replace(staged, inputs.score_cache_path)
            _fsync_directory(inputs.score_cache_path.parent)
        except BaseException:
            if staged.exists():
                staged.unlink()
            if _score_cache_file_binding(inputs.score_cache_path) == before.binding:
                transaction_path.unlink(missing_ok=True)
                _fsync_directory(inputs.score_cache_path.parent)
            raise
        after = _CacheSnapshot(
            source=content,
            binding=after_binding,
            rows_by_key=rows_by_key,
            scores={key: float(row["score"]) for key, row in rows_by_key.items()},
        )
        return before, after, transaction_path


def _extract_logits(output: object, expected: int) -> list[float]:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("sequence-classification output has no logits")
    shape = tuple(int(value) for value in getattr(logits, "shape", ()))
    if shape != (expected, 1):
        raise ValueError("MiniLM logits shape must be exactly (batch, 1)")
    values = logits.detach().float().cpu().tolist()  # type: ignore[attr-defined]
    if (
        not isinstance(values, list)
        or len(values) != expected
        or any(not isinstance(value, list) or len(value) != 1 for value in values)
    ):
        raise ValueError("MiniLM logits shape must be exactly (batch, 1)")
    return [_float32(value[0]) for value in values]


@dataclass(frozen=True)
class _ForwardResult:
    scores: tuple[float, ...]
    elapsed_seconds: tuple[float, ...]
    peak_device_memory_bytes: int
    peak_host_memory_bytes: int


def _forward_pairs(
    rows: Sequence[object],
    *,
    tokenizer: object,
    model: object,
    runtime: InferenceRuntime,
    device: str,
) -> _ForwardResult:
    scores: list[float] = []
    elapsed_per_pair: list[float] = []
    peak_host = runtime.host_memory_bytes()
    peak_device = 0
    cuda = getattr(runtime.torch, "cuda", None)
    if device == "cuda":
        cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        queries = [str(getattr(row, "query")) for row in batch]
        windows = [str(getattr(row, "window_text")) for row in batch]
        started = runtime.clock()
        encoded = tokenizer(  # type: ignore[operator]
            queries,
            windows,
            padding=True,
            truncation=False,
            max_length=PAIR_MAX_TOKENS,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer batch output must be a mapping")
        device_encoded = {
            key: tensor.to(device)  # type: ignore[attr-defined]
            for key, tensor in encoded.items()
        }
        if device == "cuda":
            cuda.synchronize()  # type: ignore[attr-defined]
        try:
            with runtime.torch.inference_mode():  # type: ignore[attr-defined]
                output = model(**device_encoded)  # type: ignore[operator]
        except BaseException as exc:
            if _is_oom(exc):
                raise RuntimeError(
                    "ROCm out of memory; fail closed without retry or CPU fallback; "
                    "a separate CPU approval receipt is required"
                ) from exc
            raise
        if device == "cuda":
            cuda.synchronize()  # type: ignore[attr-defined]
        elapsed = runtime.clock() - started
        if elapsed <= 0:
            raise RuntimeError("inference clock did not advance")
        batch_scores = _extract_logits(output, len(batch))
        scores.extend(batch_scores)
        elapsed_per_pair.extend([elapsed / len(batch)] * len(batch))
        peak_host = max(peak_host, runtime.host_memory_bytes())
        if device == "cuda":
            peak_device = max(
                peak_device,
                int(cuda.max_memory_allocated()),  # type: ignore[attr-defined]
            )
    return _ForwardResult(
        scores=tuple(scores),
        elapsed_seconds=tuple(elapsed_per_pair),
        peak_device_memory_bytes=peak_device,
        peak_host_memory_bytes=peak_host,
    )


def _rows_by_cache_key(rows: Sequence[object]) -> dict[str, object]:
    by_key: dict[str, object] = {}
    for row in rows:
        by_key.setdefault(str(getattr(row, "cache_key")), row)
    return by_key


def _validate_frozen_benchmark(inputs: ScoringInputs) -> dict[str, object]:
    recomputed = build_benchmark_plan(inputs.windows)  # type: ignore[arg-type]
    if recomputed != inputs.benchmark:
        raise ValueError("benchmark sample hash differs from preflight windows")
    return recomputed


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0 or (positive and converted <= 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return converted


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _load_benchmark_telemetry(
    inputs: ScoringInputs,
    telemetry_path: Path | str,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, object], bytes, Path]:
    if isinstance(telemetry_path, Mapping):
        raise ValueError("authenticated persisted benchmark telemetry is required")
    if not _is_sha256(expected_sha256):
        raise ValueError("authenticated persisted benchmark telemetry SHA-256 is required")
    path = Path(telemetry_path).resolve()
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError("authenticated persisted benchmark telemetry is required") from exc
    if _sha256_bytes(source) != expected_sha256:
        raise ValueError("benchmark telemetry differs from expected file hash")
    telemetry = _loads_object_no_duplicates(source, "benchmark telemetry")
    if source != _pretty_json_bytes(telemetry):
        raise ValueError("benchmark telemetry must be canonical JSON bytes")
    if set(telemetry) != _BENCHMARK_TELEMETRY_FIELDS:
        raise ValueError("benchmark telemetry fields mismatch")
    plan = _validate_frozen_benchmark(inputs)
    expected_values = {
        "schema_version": BENCHMARK_TELEMETRY_SCHEMA_VERSION,
        "action": BENCHMARK_ACTION,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": plan["sample_sha256"],
        "mode": plan["mode"],
        "batch_size": BATCH_SIZE,
        "uncached_pair_count": plan["uncached_pair_count"],
        "warmup_pair_count": plan["warmup_pair_count"],
        "timed_sample_pair_count": plan["timed_sample_pair_count"],
        "forward_pair_count": plan["forward_pair_count"],
        "timed_scope": [
            "tokenization",
            "device_transfer",
            "forward",
            "device_synchronization",
        ],
    }
    for field, expected in expected_values.items():
        if telemetry.get(field) != expected:
            raise ValueError(f"benchmark telemetry {field} mismatch")
    if not isinstance(telemetry.get("run_id"), str) or not str(
        telemetry["run_id"]
    ).strip():
        raise ValueError("benchmark telemetry run_id is invalid")
    if not _is_sha256(telemetry.get("benchmark_approval_sha256")):
        raise ValueError("benchmark telemetry approval hash is invalid")
    output_value = telemetry.get("output_path")
    if (
        not isinstance(output_value, str)
        or path != Path(output_value).resolve() / "benchmark_telemetry.json"
    ):
        raise ValueError("benchmark telemetry output path mismatch")
    device = telemetry.get("device")
    backend = telemetry.get("execution_backend")
    if (device, backend) not in {("cuda", "rocm"), ("cpu", "cpu")}:
        raise ValueError("benchmark telemetry device/backend mismatch")
    peak_device = _nonnegative_int(
        telemetry.get("peak_device_memory_bytes"),
        "benchmark telemetry peak device memory",
    )
    _nonnegative_int(
        telemetry.get("peak_host_memory_bytes"),
        "benchmark telemetry peak host memory",
    )
    if device == "cpu" and peak_device != 0:
        raise ValueError("CPU benchmark telemetry device memory must be zero")
    setup = _finite_number(
        telemetry.get("fixed_setup_seconds"),
        "benchmark telemetry fixed setup seconds",
    )
    finalize = _finite_number(
        telemetry.get("fixed_finalize_seconds"),
        "benchmark telemetry fixed finalize seconds",
    )
    repetitions = telemetry.get("timed_repetitions")
    if not isinstance(repetitions, list):
        raise ValueError("benchmark telemetry timed repetitions are invalid")
    forward_count = int(plan["forward_pair_count"])
    if forward_count == 0:
        if (
            telemetry.get("status") != "cache_complete_no_benchmark"
            or telemetry.get("device_probe") is not None
            or repetitions
            or telemetry.get("median_pairs_per_second") is not None
            or telemetry.get("projection_multiplier") != 0.0
            or telemetry.get("projected_unique_scoring_seconds") != 0.0
            or telemetry.get("projected_full_run_wall_seconds") != 0.0
            or setup != 0.0
            or finalize != 0.0
        ):
            raise ValueError("cache-complete benchmark telemetry is invalid")
        return telemetry, source, path
    if telemetry.get("status") != "benchmark_complete":
        raise ValueError("benchmark telemetry status mismatch")
    if not isinstance(telemetry.get("device_probe"), Mapping):
        raise ValueError("benchmark telemetry device probe is invalid")
    expected_repetitions = int(plan["timed_repetitions"])
    if len(repetitions) != expected_repetitions:
        raise ValueError("benchmark telemetry repetition count mismatch")
    throughputs: list[float] = []
    for index, repetition in enumerate(repetitions, start=1):
        if not isinstance(repetition, Mapping):
            raise ValueError("benchmark telemetry repetition is invalid")
        if set(repetition) != {
            "repetition",
            "pair_count",
            "elapsed_seconds",
            "pairs_per_second",
        }:
            raise ValueError("benchmark telemetry repetition fields mismatch")
        if repetition.get("repetition") != index or repetition.get(
            "pair_count"
        ) != plan["timed_sample_pair_count"]:
            raise ValueError("benchmark telemetry repetition binding mismatch")
        elapsed = _finite_number(
            repetition.get("elapsed_seconds"),
            "benchmark telemetry repetition elapsed seconds",
            positive=True,
        )
        throughput = _finite_number(
            repetition.get("pairs_per_second"),
            "benchmark telemetry repetition throughput",
            positive=True,
        )
        recomputed = int(plan["timed_sample_pair_count"]) / elapsed
        if not math.isclose(throughput, recomputed, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("benchmark telemetry repetition throughput mismatch")
        throughputs.append(throughput)
    median = _finite_number(
        telemetry.get("median_pairs_per_second"),
        "benchmark telemetry median throughput",
        positive=True,
    )
    if not math.isclose(
        median,
        statistics.median(throughputs),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("benchmark telemetry median throughput mismatch")
    multiplier = 1.25 if plan["mode"] == "primary" else 1.50
    if telemetry.get("projection_multiplier") != multiplier:
        raise ValueError("benchmark telemetry projection multiplier mismatch")
    projected_scoring = multiplier * int(plan["uncached_pair_count"]) / median
    recorded_scoring = _finite_number(
        telemetry.get("projected_unique_scoring_seconds"),
        "benchmark telemetry projected scoring seconds",
        positive=True,
    )
    recorded_wall = _finite_number(
        telemetry.get("projected_full_run_wall_seconds"),
        "benchmark telemetry projected wall seconds",
        positive=True,
    )
    if not math.isclose(
        recorded_scoring, projected_scoring, rel_tol=1e-12, abs_tol=0.0
    ) or not math.isclose(
        recorded_wall,
        setup + projected_scoring + finalize,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("benchmark telemetry projection mismatch")
    return telemetry, source, path


def build_full_inference_request(
    inputs: ScoringInputs,
    benchmark_telemetry: Path | str,
    *,
    expected_telemetry_sha256: str | None = None,
) -> dict[str, object]:
    """Build the second-gate request from authenticated persisted telemetry."""

    telemetry, telemetry_source, telemetry_path = _load_benchmark_telemetry(
        inputs,
        benchmark_telemetry,
        expected_sha256=expected_telemetry_sha256,
    )
    device = str(telemetry["device"])
    execution_backend = str(telemetry["execution_backend"])
    return {
        "schema_version": FULL_REQUEST_SCHEMA_VERSION,
        "status": "awaiting_explicit_full_inference_approval",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_telemetry_file": str(telemetry_path),
        "benchmark_telemetry_sha256": _sha256_bytes(telemetry_source),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
        "device": device,
        "execution_backend": execution_backend,
        "batch_size": BATCH_SIZE,
        "uncached_pair_count": inputs.benchmark.get("uncached_pair_count"),
        "score_cache_path": str(inputs.score_cache_path),
        "projected_full_run_wall_seconds": float(
            telemetry["projected_full_run_wall_seconds"]
        ),
        "peak_device_memory_bytes": int(
            telemetry["peak_device_memory_bytes"]
        ),
        "peak_host_memory_bytes": int(
            telemetry["peak_host_memory_bytes"]
        ),
        "cpu_fallback_authorized": device == "cpu",
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
    }


def _persist_benchmark_output(
    inputs: ScoringInputs,
    output_dir: Path,
    telemetry: Mapping[str, object],
 ) -> tuple[str, dict[str, object]]:
    destination = Path(output_dir).resolve()
    telemetry_path = destination / "benchmark_telemetry.json"
    telemetry_source = _pretty_json_bytes(telemetry)
    _exclusive_write(telemetry_path, telemetry_source)
    telemetry_sha256 = _sha256_bytes(telemetry_source)
    request = build_full_inference_request(
        inputs,
        telemetry_path,
        expected_telemetry_sha256=telemetry_sha256,
    )
    _exclusive_write(destination / "full_inference_request.json", _pretty_json_bytes(request))
    _fsync_directory(destination)
    return telemetry_sha256, request


def run_benchmark(
    preflight: ScoringInputs | Path | str,
    approval: Mapping[str, object] | Path | str | None,
    *,
    expected_approval_sha256: str | None = None,
    output_dir: Path,
    pair_limit: int = MAX_BENCHMARK_PAIRS,
    device: str = "cuda",
    runtime_factory: Callable[[], InferenceRuntime] = _default_runtime,
) -> dict[str, object]:
    """Run only the frozen, explicitly approved benchmark sample."""

    if (
        not isinstance(pair_limit, int)
        or isinstance(pair_limit, bool)
        or pair_limit < 1
        or pair_limit > MAX_BENCHMARK_PAIRS
    ):
        raise ValueError("256-pair benchmark ceiling exceeded")
    inputs = _coerce_inputs(preflight)
    _validate_input_safety(inputs)
    plan = _validate_frozen_benchmark(inputs)
    _load_bound_cache(inputs)
    forward_pairs = int(plan["forward_pair_count"])
    if forward_pairs > pair_limit:
        raise ValueError("frozen benchmark exceeds requested pair limit")
    approval_receipt, approval_sha256, approval_path = _validate_benchmark_approval(
        inputs,
        approval,
        expected_approval_sha256=expected_approval_sha256,
        pair_limit=pair_limit,
        device=device,
        output_dir=output_dir,
    )
    by_key = _rows_by_cache_key(inputs.windows)
    try:
        warmup = [by_key[str(key)] for key in plan["warmup_cache_keys"]]  # type: ignore[index]
        timed = [by_key[str(key)] for key in plan["timed_cache_keys"]]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError("frozen benchmark key is absent from windows") from exc
    planned = tuple(warmup) + tuple(
        row
        for _repetition in range(int(plan["timed_repetitions"]))
        for row in timed
    )
    reservation = _reserve_run(
        output_dir=output_dir,
        action=BENCHMARK_ACTION,
        run_id=str(approval_receipt["run_id"]),
        approval_sha256=approval_sha256,
        inputs=inputs,
        planned_windows=planned,
    )
    try:
        _consume_approval(inputs, reservation, approval_path)
        execution_backend = "rocm" if device == "cuda" else "cpu"
        if forward_pairs == 0:
            telemetry: dict[str, object] = {
                "schema_version": BENCHMARK_TELEMETRY_SCHEMA_VERSION,
                "status": "cache_complete_no_benchmark",
                "action": BENCHMARK_ACTION,
                "run_id": reservation.run_id,
                "output_path": str(reservation.destination),
                "preflight_sha256": inputs.preflight_sha256,
                "windows_sha256": inputs.windows_sha256,
                "model_materialization_receipt_sha256": (
                    inputs.materialization_receipt_sha256
                ),
                "benchmark_approval_sha256": approval_sha256,
                "benchmark_sample_sha256": plan["sample_sha256"],
                "mode": plan["mode"],
                "device": device,
                "execution_backend": execution_backend,
                "device_probe": None,
                "batch_size": BATCH_SIZE,
                "uncached_pair_count": plan["uncached_pair_count"],
                "warmup_pair_count": plan["warmup_pair_count"],
                "timed_sample_pair_count": plan["timed_sample_pair_count"],
                "forward_pair_count": 0,
                "timed_scope": [
                    "tokenization",
                    "device_transfer",
                    "forward",
                    "device_synchronization",
                ],
                "timed_repetitions": [],
                "median_pairs_per_second": None,
                "projection_multiplier": 0.0,
                "projected_unique_scoring_seconds": 0.0,
                "fixed_setup_seconds": 0.0,
                "fixed_finalize_seconds": 0.0,
                "projected_full_run_wall_seconds": 0.0,
                "peak_device_memory_bytes": 0,
                "peak_host_memory_bytes": 0,
            }
        else:
            runtime = runtime_factory()
            setup_started = runtime.clock()
            tokenizer, model, probe = _load_local_model(
                inputs, runtime, device=device
            )
            if device == "cuda":
                runtime.torch.cuda.synchronize()  # type: ignore[attr-defined]
            fixed_setup = runtime.clock() - setup_started
            if fixed_setup <= 0:
                raise RuntimeError("benchmark setup clock did not advance")
            peak_device = 0
            peak_host = runtime.host_memory_bytes()
            if warmup:
                warm = _forward_pairs(
                    warmup,
                    tokenizer=tokenizer,
                    model=model,
                    runtime=runtime,
                    device=device,
                )
                peak_device = max(peak_device, warm.peak_device_memory_bytes)
                peak_host = max(peak_host, warm.peak_host_memory_bytes)
            repetitions: list[dict[str, object]] = []
            throughputs: list[float] = []
            for repetition in range(int(plan["timed_repetitions"])):
                result = _forward_pairs(
                    timed,
                    tokenizer=tokenizer,
                    model=model,
                    runtime=runtime,
                    device=device,
                )
                elapsed = sum(result.elapsed_seconds)
                throughput = len(timed) / elapsed
                repetitions.append(
                    {
                        "repetition": repetition + 1,
                        "pair_count": len(timed),
                        "elapsed_seconds": elapsed,
                        "pairs_per_second": throughput,
                    }
                )
                throughputs.append(throughput)
                peak_device = max(peak_device, result.peak_device_memory_bytes)
                peak_host = max(peak_host, result.peak_host_memory_bytes)
            if not throughputs:
                raise RuntimeError("nonempty benchmark has no timed throughput sample")
            finalize_started = runtime.clock()
            if device == "cuda":
                runtime.torch.cuda.synchronize()  # type: ignore[attr-defined]
            peak_host = max(peak_host, runtime.host_memory_bytes())
            fixed_finalize = runtime.clock() - finalize_started
            if fixed_finalize <= 0:
                raise RuntimeError("benchmark finalize clock did not advance")
            median_throughput = statistics.median(throughputs)
            multiplier = 1.25 if plan["mode"] == "primary" else 1.50
            projected_scoring = (
                multiplier
                * int(plan["uncached_pair_count"])
                / median_throughput
            )
            projected_wall = fixed_setup + projected_scoring + fixed_finalize
            telemetry = {
                "schema_version": BENCHMARK_TELEMETRY_SCHEMA_VERSION,
                "status": "benchmark_complete",
                "action": BENCHMARK_ACTION,
                "run_id": reservation.run_id,
                "output_path": str(reservation.destination),
                "preflight_sha256": inputs.preflight_sha256,
                "windows_sha256": inputs.windows_sha256,
                "model_materialization_receipt_sha256": (
                    inputs.materialization_receipt_sha256
                ),
                "benchmark_approval_sha256": approval_sha256,
                "benchmark_sample_sha256": plan["sample_sha256"],
                "mode": plan["mode"],
                "device": device,
                "execution_backend": execution_backend,
                "device_probe": probe,
                "batch_size": BATCH_SIZE,
                "uncached_pair_count": plan["uncached_pair_count"],
                "warmup_pair_count": plan["warmup_pair_count"],
                "timed_sample_pair_count": plan["timed_sample_pair_count"],
                "forward_pair_count": forward_pairs,
                "timed_scope": [
                    "tokenization",
                    "device_transfer",
                    "forward",
                    "device_synchronization",
                ],
                "timed_repetitions": repetitions,
                "median_pairs_per_second": median_throughput,
                "projection_multiplier": multiplier,
                "projected_unique_scoring_seconds": projected_scoring,
                "fixed_setup_seconds": fixed_setup,
                "fixed_finalize_seconds": fixed_finalize,
                "projected_full_run_wall_seconds": projected_wall,
                "peak_device_memory_bytes": peak_device,
                "peak_host_memory_bytes": peak_host,
            }
        telemetry_sha256, request = _persist_benchmark_output(
            inputs,
            reservation.destination,
            telemetry,
        )
        _seal_terminal(
            reservation,
            status="complete",
            details={
                "benchmark_telemetry_sha256": telemetry_sha256,
                "full_inference_request_sha256": _sha256_bytes(
                    _pretty_json_bytes(request)
                ),
            },
        )
        return telemetry
    except BaseException as exc:
        _seal_failure(reservation, exc)
        raise


def _load_full_request(
    receipt: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    value = receipt.get("full_inference_request_file")
    if not isinstance(value, str) or not value:
        raise ValueError("full inference approval request path is invalid")
    path = Path(value)
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError("full inference request is required") from exc
    request = _loads_object_no_duplicates(source, "full inference request")
    if source != _pretty_json_bytes(request):
        raise ValueError("full inference request must be canonical JSON bytes")
    if set(request) != _FULL_REQUEST_FIELDS:
        raise ValueError("full inference request fields mismatch")
    return request, source


def _validate_full_approval(
    inputs: ScoringInputs,
    approval: Mapping[str, object] | Path | str | None,
    *,
    expected_approval_sha256: str | None,
    device: str,
    output_dir: Path,
) -> tuple[dict[str, object], str, Path, dict[str, object], dict[str, object]]:
    receipt, approval_source, approval_path = _load_receipt(
        approval,
        expected_sha256=expected_approval_sha256,
        label="full inference approval receipt",
        required_message="explicit full inference approval receipt is required",
    )
    if device == "cpu":
        fields = _FULL_CPU_APPROVAL_FIELDS
        schema = FULL_CPU_APPROVAL_SCHEMA_VERSION
        scope = FULL_CPU_APPROVAL_SCOPE
        acknowledgement = "acknowledged_cpu_fallback"
        backend = "cpu"
    else:
        fields = _FULL_APPROVAL_FIELDS
        schema = FULL_APPROVAL_SCHEMA_VERSION
        scope = FULL_APPROVAL_SCOPE
        acknowledgement = "acknowledged_no_cpu_fallback"
        backend = "rocm"
    if set(receipt) != fields:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("full inference approval fields mismatch")
    if receipt.get("schema_version") != schema or receipt.get("approval_scope") != scope:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("full inference approval schema/scope mismatch")
    _require_run_binding(
        receipt,
        label="full inference approval",
        action=FULL_SCORING_ACTION,
        output_dir=output_dir,
    )
    _require_nonempty_approver(receipt, "full inference approval")
    if receipt.get("device") != device or receipt.get("execution_backend") != backend:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval receipt")
        raise ValueError("full inference approval device/backend mismatch")
    expected = {
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
        "score_cache_path": str(inputs.score_cache_path),
        "batch_size": BATCH_SIZE,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"full inference approval {field} mismatch")
    for field in (
        "acknowledged_projected_runtime",
        "acknowledged_local_files_only",
        acknowledgement,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"full inference approval must acknowledge {field}")
    request, request_source = _load_full_request(receipt)
    if receipt.get("full_inference_request_sha256") != _sha256_bytes(request_source):
        raise ValueError("full inference approval request hash mismatch")
    telemetry_file = receipt.get("benchmark_telemetry_file")
    telemetry_sha256 = receipt.get("benchmark_telemetry_sha256")
    if (
        not isinstance(telemetry_file, str)
        or request.get("benchmark_telemetry_file") != str(Path(telemetry_file).resolve())
        or request.get("benchmark_telemetry_sha256") != telemetry_sha256
    ):
        raise ValueError("full inference approval telemetry binding mismatch")
    expected_request = build_full_inference_request(
        inputs,
        telemetry_file,
        expected_telemetry_sha256=(
            str(telemetry_sha256) if telemetry_sha256 is not None else None
        ),
    )
    if request != expected_request:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval request")
        raise ValueError("full inference request differs from authenticated telemetry")
    if request.get("device") != device or request.get("execution_backend") != backend:
        if device == "cpu":
            raise ValueError("CPU fallback requires a separate approval request")
        raise ValueError("full inference request device/backend mismatch")
    telemetry, _source, _path = _load_benchmark_telemetry(
        inputs,
        telemetry_file,
        expected_sha256=(
            str(telemetry_sha256) if telemetry_sha256 is not None else None
        ),
    )
    return (
        receipt,
        _sha256_bytes(approval_source),
        approval_path,
        request,
        telemetry,
    )


def _raw_output_sha256(cache_key: str, score: float) -> str:
    return _sha256_bytes(
        canonical_compact_json_bytes(
            {
                "cache_key": cache_key,
                "inference_dtype": "float32",
                "score_representation": "raw_logits",
                "raw_float32_be_sha256": _sha256_bytes(struct.pack(">f", score)),
            }
        )
    )


def _build_score_row(
    window: object,
    *,
    sequence_index: int,
    disposition: str,
    score: float,
    elapsed_seconds: float,
    peak_device_memory_bytes: int,
    peak_host_memory_bytes: int,
) -> MiniLMScoreRow:
    cache_key = str(getattr(window, "cache_key"))
    reservation = _reservation_payload(window, sequence_index)
    reservation_sha256 = _sha256_bytes(canonical_compact_json_bytes(reservation))
    raw_output_sha256 = _raw_output_sha256(cache_key, score)
    output_sha256 = _sha256_bytes(
        canonical_compact_json_bytes(
            {
                "reservation_sha256": reservation_sha256,
                "disposition": disposition,
                "raw_output_sha256": raw_output_sha256,
            }
        )
    )
    return MiniLMScoreRow(
        topic_id=str(getattr(window, "topic_id")),
        family=str(getattr(window, "family")),
        variant=str(getattr(window, "variant")),
        rank=int(getattr(window, "rank")),
        document_id=str(getattr(window, "document_id")),
        window_id=str(getattr(window, "window_id")),
        query_sha256=str(getattr(window, "query_sha256")),
        window_sha256=str(getattr(window, "window_sha256")),
        cache_key=cache_key,
        reservation_sha256=reservation_sha256,
        disposition=disposition,
        score=score,
        elapsed_seconds=elapsed_seconds,
        peak_device_memory_bytes=peak_device_memory_bytes,
        peak_host_memory_bytes=peak_host_memory_bytes,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        score_representation="raw_logits",
        inference_dtype="float32",
        raw_output_sha256=raw_output_sha256,
        output_sha256=output_sha256,
    )


def _persist_scoring_output(
    destination: Path,
    rows: Sequence[MiniLMScoreRow],
    *,
    inputs: ScoringInputs,
    reservation: _RunReservation,
    approval_sha256: str,
    request: Mapping[str, object],
    cache_before: _CacheSnapshot,
    cache_after: _CacheSnapshot,
) -> None:
    planned_records, reservation_sequence_root = _planned_reservation_records(
        inputs.windows
    )
    if (
        len(rows) != len(planned_records)
        or reservation.planned_row_count != len(planned_records)
        or reservation.reservation_sequence_root_sha256
        != reservation_sequence_root
        or any(
            row.reservation_sha256 != planned["reservation_sha256"]
            for row, planned in zip(rows, planned_records, strict=True)
        )
    ):
        raise RuntimeError("scoring rows differ from reserved preflight sequence")
    ledger_bytes = b"".join(
        canonical_compact_json_bytes(row.to_dict()) + b"\n" for row in rows
    )
    ledger_sequence_root = _sha256_bytes(
        b"".join((row.output_sha256 + "\n").encode("ascii") for row in rows)
    )
    unique_outputs: dict[str, str] = {}
    for row in rows:
        previous = unique_outputs.setdefault(row.cache_key, row.raw_output_sha256)
        if previous != row.raw_output_sha256:
            raise RuntimeError("one cache key produced conflicting raw scores")
    unique_score_root = _sha256_bytes(
        b"".join(
            f"{cache_key}:{unique_outputs[cache_key]}\n".encode("ascii")
            for cache_key in sorted(unique_outputs)
        )
    )
    receipt = {
        "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "full_inference_approval_sha256": approval_sha256,
        "full_inference_request_sha256": _sha256_bytes(_pretty_json_bytes(request)),
        "score_cache_path": str(inputs.score_cache_path),
        "cache_before_sha256": cache_before.binding["sha256"],
        "cache_after_sha256": cache_after.binding["sha256"],
        "cache_before_bytes": cache_before.binding["bytes"],
        "cache_after_bytes": cache_after.binding["bytes"],
        "cache_before_count": len(cache_before.rows_by_key),
        "cache_after_count": len(cache_after.rows_by_key),
        "planned_window_count": len(inputs.windows),
        "completed_window_count": len(rows),
        "cache_hit_count": sum(row.disposition == "cache_hit" for row in rows),
        "forward_pass_count": sum(row.disposition == "forward_pass" for row in rows),
        "same_run_reuse_count": sum(
            row.disposition == "same_run_reuse" for row in rows
        ),
        "unique_score_count": len(unique_outputs),
        "failed_window_count": 0,
        "pending_window_count": 0,
        "reservation_sequence_root_sha256": reservation_sequence_root,
        "ledger_bytes": len(ledger_bytes),
        "ledger_sha256": _sha256_bytes(ledger_bytes),
        "ledger_sequence_root_sha256": ledger_sequence_root,
        "unique_score_root_sha256": unique_score_root,
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    _exclusive_write(destination / "scoring_ledger.jsonl", ledger_bytes)
    _exclusive_write(destination / "scoring_receipt.json", _pretty_json_bytes(receipt))
    _fsync_directory(destination)


def run_full_scoring(
    preflight: ScoringInputs | Mapping[str, object] | Path | str,
    approval: Mapping[str, object] | Path | str | None,
    *,
    expected_approval_sha256: str | None = None,
    output_dir: Path,
    score_cache_root: Path | None = None,
    device: str = "cuda",
    score_cache_factory: Callable[[Path, ScoreCacheContext], object] = GlobalScoreCache,
    runtime_factory: Callable[[], InferenceRuntime] = _default_runtime,
) -> tuple[MiniLMScoreRow, ...]:
    """Score all exact misses once and materialize one row per planned window."""

    # Keep the public refusal contract useful to callers holding an in-memory
    # preflight fixture: absence of the second receipt never reaches file,
    # cache, runtime, or model boundaries.  Protected topics still take
    # precedence when they are visible in that fixture.
    if approval is None and isinstance(preflight, Mapping):
        topics = preflight.get("topics", [])
        if isinstance(topics, list):
            for row in topics:
                if isinstance(row, Mapping):
                    topic_id = str(row.get("topic_id", ""))
                    if topic_id in PROTECTED_TOPIC_IDS:
                        raise ValueError(f"protected topic {topic_id} is forbidden")
        raise ValueError("explicit full inference approval receipt is required")
    inputs = _coerce_inputs(preflight)
    _validate_input_safety(inputs)
    _validate_frozen_benchmark(inputs)
    root = Path(score_cache_root or inputs.score_cache_root).resolve()
    expected_path = root.joinpath(*inputs.context.path_parts).resolve()
    if expected_path != inputs.score_cache_path.resolve():
        raise ValueError("full scoring cache root differs from preflight binding")
    receipt, approval_sha256, approval_path, request, _telemetry = (
        _validate_full_approval(
            inputs,
            approval,
            expected_approval_sha256=expected_approval_sha256,
            device=device,
            output_dir=output_dir,
        )
    )
    reservation = _reserve_run(
        output_dir=output_dir,
        action=FULL_SCORING_ACTION,
        run_id=str(receipt["run_id"]),
        approval_sha256=approval_sha256,
        inputs=inputs,
        planned_windows=inputs.windows,
    )
    transaction_path: Path | None = None
    try:
        _consume_approval(inputs, reservation, approval_path)
        cache_snapshot = _load_bound_cache(inputs)
        cache = score_cache_factory(root, inputs.context)
        cache_path = Path(getattr(cache, "path")).resolve()
        if cache_path != expected_path:
            raise ValueError("GlobalScoreCache path differs from preflight binding")
        by_key = _rows_by_cache_key(inputs.windows)
        cached_scores: dict[str, float] = {}
        misses: list[object] = []
        for cache_key, window in by_key.items():
            query = str(getattr(window, "query"))
            text = str(getattr(window, "window_text"))
            if score_cache_key(inputs.context, query=query, window=text) != cache_key:
                raise ValueError("GlobalScoreCache identity differs from window plan")
            score = cache_snapshot.scores.get(cache_key)
            if bool(getattr(window, "cache_hit", False)) != (score is not None):
                raise ValueError("preflight cache-hit reservation differs from bound cache")
            if score is None:
                misses.append(window)
            else:
                cached_scores[cache_key] = score

        forward_scores: dict[str, tuple[float, float, int, int]] = {}
        additions: list[tuple[str, str, float]] = []
        if misses:
            runtime = runtime_factory()
            tokenizer, model, _probe = _load_local_model(
                inputs, runtime, device=device
            )
            result = _forward_pairs(
                misses,
                tokenizer=tokenizer,
                model=model,
                runtime=runtime,
                device=device,
            )
            for window, score, elapsed in zip(
                misses, result.scores, result.elapsed_seconds, strict=True
            ):
                cache_key = str(getattr(window, "cache_key"))
                forward_scores[cache_key] = (
                    score,
                    elapsed,
                    result.peak_device_memory_bytes,
                    result.peak_host_memory_bytes,
                )
                additions.append(
                    (
                        str(getattr(window, "query")),
                        str(getattr(window, "window_text")),
                        score,
                    )
                )

        rows: list[MiniLMScoreRow] = []
        seen_forward: set[str] = set()
        for sequence_index, window in enumerate(inputs.windows):
            cache_key = str(getattr(window, "cache_key"))
            if cache_key in cached_scores:
                rows.append(
                    _build_score_row(
                        window,
                        sequence_index=sequence_index,
                        disposition="cache_hit",
                        score=cached_scores[cache_key],
                        elapsed_seconds=0.0,
                        peak_device_memory_bytes=0,
                        peak_host_memory_bytes=0,
                    )
                )
                continue
            score, elapsed, device_memory, host_memory = forward_scores[cache_key]
            first = cache_key not in seen_forward
            seen_forward.add(cache_key)
            rows.append(
                _build_score_row(
                    window,
                    sequence_index=sequence_index,
                    disposition="forward_pass" if first else "same_run_reuse",
                    score=score,
                    elapsed_seconds=elapsed if first else 0.0,
                    peak_device_memory_bytes=device_memory if first else 0,
                    peak_host_memory_bytes=host_memory if first else 0,
                )
            )
        if len(rows) != len(inputs.windows):
            raise RuntimeError("scoring ledger has pending or failed rows")
        cache_before, cache_after, transaction_path = _write_cache_replacement(
            inputs,
            reservation,
            additions,
        )
        _persist_scoring_output(
            reservation.destination,
            rows,
            inputs=inputs,
            reservation=reservation,
            approval_sha256=approval_sha256,
            request=request,
            cache_before=cache_before,
            cache_after=cache_after,
        )
        _seal_terminal(
            reservation,
            status="complete",
            details={
                "scoring_receipt_sha256": _sha256_file(
                    reservation.destination / "scoring_receipt.json"
                ),
                "cache_after_sha256": cache_after.binding["sha256"],
            },
        )
        if transaction_path is not None:
            transaction_path.unlink()
            _fsync_directory(transaction_path.parent)
        return tuple(rows)
    except BaseException as exc:
        _seal_failure(reservation, exc)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run approval-gated local MiniLM benchmark or full scoring"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--preflight", type=Path, required=True)
    benchmark.add_argument("--approval", type=Path, required=True)
    benchmark.add_argument("--approval-sha256", required=True)
    benchmark.add_argument("--pair-limit", type=int, default=MAX_BENCHMARK_PAIRS)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run = subparsers.add_parser("run")
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--approval-sha256", required=True)
    run.add_argument("--score-cache", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    inputs = load_scoring_inputs(args.preflight)
    if args.command == "benchmark":
        result = run_benchmark(
            inputs,
            args.approval,
            expected_approval_sha256=args.approval_sha256,
            pair_limit=args.pair_limit,
            output_dir=args.output,
            device=args.device,
        )
    else:
        rows = run_full_scoring(
            inputs,
            args.approval,
            expected_approval_sha256=args.approval_sha256,
            score_cache_root=args.score_cache,
            output_dir=args.output,
            device=args.device,
        )
        result = {
            "status": "complete",
            "score_row_count": len(rows),
            "output": str(args.output.resolve()),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

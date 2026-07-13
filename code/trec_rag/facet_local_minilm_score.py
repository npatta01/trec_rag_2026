"""Approval-gated, local-only MiniLM benchmark and scoring runner.

The scorer accepts only the authenticated tokenizer preflight-v2 window plan.
Model construction and every forward pass remain behind an exact receipt.  It
has no retrieval, qrels, hosted-inference, or network interface.
"""

from __future__ import annotations

import argparse
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

BENCHMARK_APPROVAL_SCHEMA_VERSION = "facet-local-minilm-benchmark-approval-v1"
BENCHMARK_APPROVAL_SCOPE = "facet_local_minilm_rocm_benchmark_v1"
BENCHMARK_CPU_APPROVAL_SCHEMA_VERSION = (
    "facet-local-minilm-benchmark-cpu-fallback-approval-v1"
)
BENCHMARK_CPU_APPROVAL_SCOPE = "facet_local_minilm_cpu_benchmark_fallback_v1"
FULL_APPROVAL_SCHEMA_VERSION = "facet-local-minilm-full-inference-approval-v1"
FULL_APPROVAL_SCOPE = "facet_local_minilm_rocm_full_inference_v1"
FULL_CPU_APPROVAL_SCHEMA_VERSION = (
    "facet-local-minilm-full-inference-cpu-fallback-approval-v1"
)
FULL_CPU_APPROVAL_SCOPE = "facet_local_minilm_cpu_full_inference_fallback_v1"

BENCHMARK_TELEMETRY_SCHEMA_VERSION = "facet-local-minilm-benchmark-telemetry-v1"
FULL_REQUEST_SCHEMA_VERSION = "facet-local-minilm-full-inference-request-v1"
SCORE_ROW_SCHEMA_VERSION = "facet-local-minilm-score-row-v1"
SCORING_RECEIPT_SCHEMA_VERSION = "facet-local-minilm-scoring-receipt-v1"

_BENCHMARK_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "approval_scope",
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
        "approved_by",
        "device",
        "execution_backend",
        "preflight_sha256",
        "windows_sha256",
        "model_materialization_receipt_sha256",
        "benchmark_sample_sha256",
        "full_inference_request_file",
        "full_inference_request_sha256",
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
    label: str,
    required_message: str,
) -> tuple[dict[str, object], bytes]:
    if value is None:
        raise ValueError(required_message)
    if isinstance(value, Mapping):
        receipt = dict(value)
        return receipt, _pretty_json_bytes(receipt)
    path = Path(value)
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(required_message) from exc
    receipt = _loads_object_no_duplicates(source, label)
    if source != _pretty_json_bytes(receipt):
        raise ValueError(f"{label} must be canonical JSON bytes")
    return receipt, source


def _require_nonempty_approver(receipt: Mapping[str, object], label: str) -> None:
    approved_by = receipt.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError(f"{label} approved_by must be nonempty")


def _validate_benchmark_approval(
    inputs: ScoringInputs,
    approval: Mapping[str, object] | Path | str | None,
    *,
    pair_limit: int,
    device: str,
) -> tuple[dict[str, object], str]:
    receipt, source = _load_receipt(
        approval,
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
    return receipt, _sha256_bytes(source)


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


def _extract_logits(output: object, expected: int) -> list[float]:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("sequence-classification output has no logits")
    values = logits.detach().float().cpu().reshape(-1).tolist()  # type: ignore[attr-defined]
    flattened: list[object] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("MiniLM must produce one raw relevance logit per pair")
            flattened.append(value[0])
        else:
            flattened.append(value)
    if len(flattened) != expected:
        raise ValueError("MiniLM output count differs from input pair count")
    return [_float32(value) for value in flattened]


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
        started = runtime.clock()
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


def build_full_inference_request(
    inputs: ScoringInputs,
    benchmark_telemetry: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact second-gate request; this does not authorize inference."""

    device = str(benchmark_telemetry.get("device", "cuda"))
    execution_backend = str(
        benchmark_telemetry.get(
            "execution_backend", "rocm" if device == "cuda" else "cpu"
        )
    )
    return {
        "schema_version": FULL_REQUEST_SCHEMA_VERSION,
        "status": "awaiting_explicit_full_inference_approval",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_telemetry_sha256": _sha256_bytes(
            canonical_compact_json_bytes(dict(benchmark_telemetry))
        ),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
        "device": device,
        "execution_backend": execution_backend,
        "batch_size": BATCH_SIZE,
        "uncached_pair_count": inputs.benchmark.get("uncached_pair_count"),
        "score_cache_path": str(inputs.score_cache_path),
        "projected_full_run_wall_seconds": float(
            benchmark_telemetry.get("projected_full_run_wall_seconds", 0.0)
        ),
        "peak_device_memory_bytes": int(
            benchmark_telemetry.get("peak_device_memory_bytes", 0)
        ),
        "peak_host_memory_bytes": int(
            benchmark_telemetry.get("peak_host_memory_bytes", 0)
        ),
        "cpu_fallback_authorized": device == "cpu",
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
    }


def _persist_benchmark_output(
    output_dir: Path,
    telemetry: Mapping[str, object],
    request: Mapping[str, object],
) -> None:
    destination = Path(output_dir)
    try:
        destination.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(
            f"create-only benchmark output already exists: {destination}"
        ) from exc
    _exclusive_write(destination / "benchmark_telemetry.json", _pretty_json_bytes(telemetry))
    _exclusive_write(destination / "full_inference_request.json", _pretty_json_bytes(request))


def run_benchmark(
    preflight: ScoringInputs | Path | str,
    approval: Mapping[str, object] | Path | str | None,
    *,
    pair_limit: int = MAX_BENCHMARK_PAIRS,
    output_dir: Path | None = None,
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
    forward_pairs = int(plan["forward_pair_count"])
    if forward_pairs == 0:
        telemetry: dict[str, object] = {
            "schema_version": BENCHMARK_TELEMETRY_SCHEMA_VERSION,
            "status": "cache_complete_no_benchmark",
            "preflight_sha256": inputs.preflight_sha256,
            "windows_sha256": inputs.windows_sha256,
            "model_materialization_receipt_sha256": (
                inputs.materialization_receipt_sha256
            ),
            "benchmark_sample_sha256": plan["sample_sha256"],
            "device": device,
            "execution_backend": "rocm" if device == "cuda" else "cpu",
            "batch_size": BATCH_SIZE,
            "forward_pair_count": 0,
            "median_pairs_per_second": None,
            "projected_full_run_wall_seconds": 0.0,
            "peak_device_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
            "timed_repetitions": [],
        }
        request = build_full_inference_request(inputs, telemetry)
        if output_dir is not None:
            _persist_benchmark_output(output_dir, telemetry, request)
        return telemetry
    if forward_pairs > pair_limit:
        raise ValueError("frozen benchmark exceeds requested pair limit")
    _approval, approval_sha256 = _validate_benchmark_approval(
        inputs, approval, pair_limit=pair_limit, device=device
    )
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"create-only benchmark output already exists: {output_dir}")
    runtime = runtime_factory()
    tokenizer, model, probe = _load_local_model(inputs, runtime, device=device)
    by_key = _rows_by_cache_key(inputs.windows)
    try:
        warmup = [by_key[str(key)] for key in plan["warmup_cache_keys"]]  # type: ignore[index]
        timed = [by_key[str(key)] for key in plan["timed_cache_keys"]]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError("frozen benchmark key is absent from windows") from exc
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
    median_throughput = statistics.median(throughputs)
    multiplier = 1.25 if plan["mode"] == "primary" else 1.50
    projected = (
        multiplier * int(plan["uncached_pair_count"]) / median_throughput
    )
    telemetry = {
        "schema_version": BENCHMARK_TELEMETRY_SCHEMA_VERSION,
        "status": "benchmark_complete",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_approval_sha256": approval_sha256,
        "benchmark_sample_sha256": plan["sample_sha256"],
        "mode": plan["mode"],
        "device": device,
        "execution_backend": "rocm" if device == "cuda" else "cpu",
        "device_probe": probe,
        "batch_size": BATCH_SIZE,
        "warmup_pair_count": plan["warmup_pair_count"],
        "timed_sample_pair_count": plan["timed_sample_pair_count"],
        "forward_pair_count": forward_pairs,
        "timed_repetitions": repetitions,
        "median_pairs_per_second": median_throughput,
        "projection_multiplier": multiplier,
        "projected_full_run_wall_seconds": projected,
        "peak_device_memory_bytes": peak_device,
        "peak_host_memory_bytes": peak_host,
    }
    request = build_full_inference_request(inputs, telemetry)
    if output_dir is not None:
        _persist_benchmark_output(output_dir, telemetry, request)
    return telemetry


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
    device: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    receipt, approval_source = _load_receipt(
        approval,
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
    request_expected = {
        "schema_version": FULL_REQUEST_SCHEMA_VERSION,
        "status": "awaiting_explicit_full_inference_approval",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark.get("sample_sha256"),
        "device": device,
        "execution_backend": backend,
        "batch_size": BATCH_SIZE,
        "uncached_pair_count": inputs.benchmark.get("uncached_pair_count"),
        "score_cache_path": str(inputs.score_cache_path),
        "cpu_fallback_authorized": device == "cpu",
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
    }
    for field, value in request_expected.items():
        # CPU use always needs a newly generated CPU-specific request as well.
        if request.get(field) != value:
            if device == "cpu":
                raise ValueError("CPU fallback requires a separate approval request")
            raise ValueError(f"full inference request {field} mismatch")
    return receipt, _sha256_bytes(approval_source), request


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
    disposition: str,
    score: float,
    elapsed_seconds: float,
    peak_device_memory_bytes: int,
    peak_host_memory_bytes: int,
) -> MiniLMScoreRow:
    cache_key = str(getattr(window, "cache_key"))
    reservation = {
        "topic_id": str(getattr(window, "topic_id")),
        "variant": str(getattr(window, "variant")),
        "rank": int(getattr(window, "rank")),
        "document_id": str(getattr(window, "document_id")),
        "window_id": str(getattr(window, "window_id")),
        "cache_key": cache_key,
    }
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
    approval_sha256: str,
    request: Mapping[str, object],
) -> None:
    ledger_bytes = b"".join(
        canonical_compact_json_bytes(row.to_dict()) + b"\n" for row in rows
    )
    ledger_root = _sha256_bytes(
        b"".join((row.output_sha256 + "\n").encode("ascii") for row in rows)
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
        "planned_window_count": len(inputs.windows),
        "completed_window_count": len(rows),
        "cache_hit_count": sum(row.disposition == "cache_hit" for row in rows),
        "forward_pass_count": sum(row.disposition == "forward_pass" for row in rows),
        "failed_window_count": 0,
        "pending_window_count": 0,
        "ledger_bytes": len(ledger_bytes),
        "ledger_sha256": _sha256_bytes(ledger_bytes),
        "ledger_root_sha256": ledger_root,
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    _exclusive_write(destination / "scoring_ledger.jsonl", ledger_bytes)
    _exclusive_write(destination / "scoring_receipt.json", _pretty_json_bytes(receipt))


def run_full_scoring(
    preflight: ScoringInputs | Mapping[str, object] | Path | str,
    approval: Mapping[str, object] | Path | str | None,
    *,
    score_cache_root: Path | None = None,
    output_dir: Path | None = None,
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
    _receipt, approval_sha256, request = _validate_full_approval(
        inputs, approval, device=device
    )
    destination: Path | None = None
    if output_dir is not None:
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(
                f"create-only scoring output already exists: {destination}"
            )
        try:
            destination.mkdir(parents=True)
        except FileExistsError as exc:
            raise FileExistsError(
                f"create-only scoring output already exists: {destination}"
            ) from exc

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
        if cache.cache_key(query_text=query, text=text) != cache_key:  # type: ignore[attr-defined]
            raise ValueError("GlobalScoreCache identity differs from window plan")
        score = cache.get(query_text=query, text=text)  # type: ignore[attr-defined]
        if score is None:
            misses.append(window)
        else:
            cached_scores[cache_key] = _float32(score)

    forward_scores: dict[str, tuple[float, float, int, int]] = {}
    if misses:
        runtime = runtime_factory()
        tokenizer, model, _probe = _load_local_model(inputs, runtime, device=device)
        result = _forward_pairs(
            misses,
            tokenizer=tokenizer,
            model=model,
            runtime=runtime,
            device=device,
        )
        additions: list[tuple[str, str, float]] = []
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
        # Commit only after every approved miss completed; OOM writes nothing.
        cache.add_many(additions)  # type: ignore[attr-defined]

    rows: list[MiniLMScoreRow] = []
    for window in inputs.windows:
        cache_key = str(getattr(window, "cache_key"))
        if cache_key in cached_scores:
            rows.append(
                _build_score_row(
                    window,
                    disposition="cache_hit",
                    score=cached_scores[cache_key],
                    elapsed_seconds=0.0,
                    peak_device_memory_bytes=0,
                    peak_host_memory_bytes=0,
                )
            )
        else:
            score, elapsed, device_memory, host_memory = forward_scores[cache_key]
            rows.append(
                _build_score_row(
                    window,
                    disposition="forward_pass",
                    score=score,
                    elapsed_seconds=elapsed,
                    peak_device_memory_bytes=device_memory,
                    peak_host_memory_bytes=host_memory,
                )
            )
    if len(rows) != len(inputs.windows):
        raise RuntimeError("scoring ledger has pending or failed rows")
    if destination is not None:
        _persist_scoring_output(
            destination,
            rows,
            inputs=inputs,
            approval_sha256=approval_sha256,
            request=request,
        )
    return tuple(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run approval-gated local MiniLM benchmark or full scoring"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--preflight", type=Path, required=True)
    benchmark.add_argument("--approval", type=Path, required=True)
    benchmark.add_argument("--pair-limit", type=int, default=MAX_BENCHMARK_PAIRS)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run = subparsers.add_parser("run")
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
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
            pair_limit=args.pair_limit,
            output_dir=args.output,
            device=args.device,
        )
    else:
        rows = run_full_scoring(
            inputs,
            args.approval,
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

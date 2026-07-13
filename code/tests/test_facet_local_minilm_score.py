from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag import facet_local_minilm_score as scorer_module
from trec_rag.facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    build_benchmark_plan,
    score_cache_context,
)
from trec_rag.facet_local_minilm_score import (
    BATCH_SIZE,
    BENCHMARK_APPROVAL_SCHEMA_VERSION,
    BENCHMARK_APPROVAL_SCOPE,
    FULL_APPROVAL_SCHEMA_VERSION,
    FULL_APPROVAL_SCOPE,
    FULL_CPU_APPROVAL_SCHEMA_VERSION,
    FULL_CPU_APPROVAL_SCOPE,
    InferenceRuntime,
    MiniLMScoreRow,
    ScoringInputs,
    build_full_inference_request,
    load_scoring_inputs,
    run_benchmark,
    run_full_scoring,
    score_cache_key,
)
from trec_rag.rerank_score_cache import GlobalScoreCache


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


@dataclass(frozen=True)
class FakeWindow:
    cache_key: str
    query: str
    window_text: str
    pair_token_count: int
    topic_id: str = "200"
    family: str = "facet"
    variant: str = "facet:test"
    rank: int = 1
    document_id: str = "doc-1"
    query_sha256: str = "a" * 64
    window_sha256: str = "b" * 64
    window_id: str = "c" * 64
    cache_hit: bool = False


def _window(index: int, *, pair_tokens: int | None = None, topic_id: str = "200"):
    query = f"facet query {index}"
    text = f"window passage {index}"
    context = score_cache_context()
    return FakeWindow(
        cache_key=score_cache_key(context, query=query, window=text),
        query=query,
        window_text=text,
        pair_token_count=pair_tokens or 100 + index,
        topic_id=topic_id,
        document_id=f"doc-{index}",
        query_sha256=_sha256(query.encode()),
        window_sha256=_sha256(text.encode()),
        window_id=_sha256(f"window-{index}".encode()),
    )


def _inputs(
    *,
    windows=(),
    benchmark=None,
    score_cache_root: Path | None = None,
    preflight_overrides: dict[str, object] | None = None,
) -> ScoringInputs:
    if benchmark is None:
        benchmark = build_benchmark_plan(windows) if windows else {
            "mode": "cache_complete",
            "uncached_pair_count": 0,
            "warmup_pair_count": 0,
            "timed_sample_pair_count": 0,
            "timed_repetitions": 0,
            "forward_pair_count": 0,
            "warmup_cache_keys": [],
            "timed_cache_keys": [],
            "sample_sha256": _sha256(
                b'{"timed_cache_keys":[],"timed_repetitions":0,"warmup_cache_keys":[]}'
            ),
        }
    context = score_cache_context()
    root = Path(score_cache_root or "/test/facet-local-minilm-score-cache").resolve()
    path = root.joinpath(*context.path_parts).resolve()
    binding = _file_binding(path)
    topics = sorted({str(row.topic_id) for row in windows})
    payload: dict[str, object] = {
        "schema_version": "facet-local-minilm-preflight-v2",
        "status": "tokenizer_only_preflight_complete",
        "model_materialization_receipt_sha256": "3" * 64,
        "model_materialization": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "window_policy": {"version": "facet-local-minilm-window-policy-v1"},
        "score_cache": {
            "root": str(root),
            "path": str(path),
            "context": context.artifact_metadata,
            "binding": binding,
        },
        "summary": {
            "window_count": len(windows),
            "unique_uncached_pair_count": int(
                benchmark.get("uncached_pair_count", 0)
            ),
            "qrels_access_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "inference_count": 0,
        },
        "topics": [{"topic_id": topic_id} for topic_id in topics],
        "streams": [],
        "benchmark": dict(benchmark),
        "inference_authorized": False,
        "model_constructed": False,
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
    }
    if preflight_overrides:
        for key, value in preflight_overrides.items():
            if key == "summary" and isinstance(value, dict):
                payload["summary"] = {**payload["summary"], **value}  # type: ignore[arg-type]
            else:
                payload[key] = value
    return ScoringInputs(
        preflight=payload,
        preflight_path=None,
        preflight_source=b"",
        preflight_sha256="1" * 64,
        windows_sha256="2" * 64,
        materialization_receipt_sha256="3" * 64,
        snapshot_path=Path("/verified/pinned/snapshot"),
        windows=tuple(windows),
        benchmark=dict(benchmark),
        context=context,
        score_cache_root=root,
        score_cache_path=path,
        score_cache_binding=binding,
    )


def _benchmark_approval(
    inputs: ScoringInputs,
    output_dir: Path,
    *,
    device: str = "cuda",
    run_id: str = "benchmark-test-run",
    pair_limit: int = 256,
):
    payload = {
        "schema_version": (
            scorer_module.BENCHMARK_CPU_APPROVAL_SCHEMA_VERSION
            if device == "cpu"
            else BENCHMARK_APPROVAL_SCHEMA_VERSION
        ),
        "approval_scope": (
            scorer_module.BENCHMARK_CPU_APPROVAL_SCOPE
            if device == "cpu"
            else BENCHMARK_APPROVAL_SCOPE
        ),
        "action": "benchmark",
        "run_id": run_id,
        "output_path": str(output_dir.resolve()),
        "approved_by": "unit-test-user",
        "device": device,
        "execution_backend": "cpu" if device == "cpu" else "rocm",
        "pair_limit": pair_limit,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "acknowledged_maximum_256_pairs": True,
        "acknowledged_local_files_only": True,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference": True,
    }
    payload[
        "acknowledged_cpu_fallback"
        if device == "cpu"
        else "acknowledged_no_cpu_fallback"
    ] = True
    return payload


def _write_approval(
    directory: Path,
    name: str,
    payload: dict[str, object],
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    source = _pretty(payload)
    path.write_bytes(source)
    return path, _sha256(source)


def _write_benchmark_approval(
    tmp_path: Path,
    inputs: ScoringInputs,
    output_dir: Path,
    *,
    device: str = "cuda",
    run_id: str = "benchmark-test-run",
) -> tuple[Path, str]:
    return _write_approval(
        tmp_path / "approvals",
        f"{run_id}.json",
        _benchmark_approval(
            inputs,
            output_dir,
            device=device,
            run_id=run_id,
        ),
    )


def _write_full_request(
    tmp_path: Path,
    inputs: ScoringInputs,
    *,
    device: str = "cuda",
    approval_pair_limit: int = 256,
) -> tuple[Path, dict[str, object]]:
    plan = inputs.benchmark
    forward_count = int(plan["forward_pair_count"])
    output = tmp_path / f"persisted_benchmark_{device}"
    output.mkdir()
    backend = "rocm" if device == "cuda" else "cpu"
    if forward_count:
        throughput = 10.0
        multiplier = 1.25 if plan["mode"] == "primary" else 1.5
        projected_scoring = multiplier * int(plan["uncached_pair_count"]) / throughput
        repetitions = [
            {
                "repetition": index + 1,
                "pair_count": int(plan["timed_sample_pair_count"]),
                "elapsed_seconds": int(plan["timed_sample_pair_count"]) / throughput,
                "pairs_per_second": throughput,
            }
            for index in range(int(plan["timed_repetitions"]))
        ]
        status = "benchmark_complete"
        setup = 2.0
        finalize = 1.0
        projected_wall = setup + projected_scoring + finalize
        probe: object = {
            "device_name": "fake AMD" if device == "cuda" else "cpu",
            "torch_hip_version": "6.4" if device == "cuda" else None,
        }
        median: object = throughput
    else:
        multiplier = 0.0
        projected_scoring = 0.0
        repetitions = []
        status = "cache_complete_no_benchmark"
        setup = finalize = projected_wall = 0.0
        probe = None
        median = None
    telemetry = {
        "schema_version": scorer_module.BENCHMARK_TELEMETRY_SCHEMA_VERSION,
        "status": status,
        "action": "benchmark",
        "run_id": f"persisted-benchmark-{device}",
        "output_path": str(output.resolve()),
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "benchmark_approval_sha256": "pending",
        "mode": plan["mode"],
        "median_pairs_per_second": median,
        "projection_multiplier": multiplier,
        "projected_unique_scoring_seconds": projected_scoring,
        "fixed_setup_seconds": setup,
        "fixed_finalize_seconds": finalize,
        "projected_full_run_wall_seconds": projected_wall,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "device": device,
        "execution_backend": backend,
        "device_probe": probe,
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
        "timed_repetitions": repetitions,
        "peak_device_memory_bytes": 1234 if device == "cuda" and forward_count else 0,
        "peak_host_memory_bytes": 4567 if forward_count else 0,
    }
    approval_path, approval_sha256 = _write_approval(
        tmp_path / "approvals",
        f"persisted-benchmark-{device}.json",
        _benchmark_approval(
            inputs,
            output,
            device=device,
            run_id=f"persisted-benchmark-{device}",
            pair_limit=approval_pair_limit,
        ),
    )
    telemetry["benchmark_approval_sha256"] = approval_sha256
    telemetry_path = output / "benchmark_telemetry.json"
    telemetry_source = _pretty(telemetry)
    telemetry_path.write_bytes(telemetry_source)
    telemetry_sha256 = _sha256(telemetry_source)
    request = {
        "schema_version": scorer_module.FULL_REQUEST_SCHEMA_VERSION,
        "status": "awaiting_explicit_full_inference_approval",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_telemetry_file": str(telemetry_path.resolve()),
        "benchmark_telemetry_sha256": telemetry_sha256,
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "device": device,
        "execution_backend": backend,
        "batch_size": BATCH_SIZE,
        "uncached_pair_count": inputs.benchmark["uncached_pair_count"],
        "score_cache_path": str(inputs.score_cache_path),
        "projected_full_run_wall_seconds": projected_wall,
        "peak_device_memory_bytes": telemetry["peak_device_memory_bytes"],
        "peak_host_memory_bytes": telemetry["peak_host_memory_bytes"],
        "cpu_fallback_authorized": device == "cpu",
        "qrels_path_supported": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
    }
    path = output / "full_inference_request.json"
    request_source = _pretty(request)
    path.write_bytes(request_source)
    by_key = {}
    for window in inputs.windows:
        by_key.setdefault(window.cache_key, window)
    planned = tuple(
        [by_key[key] for key in inputs.benchmark["warmup_cache_keys"]]
        + [
            by_key[key]
            for _repetition in range(int(inputs.benchmark["timed_repetitions"]))
            for key in inputs.benchmark["timed_cache_keys"]
        ]
    )
    records, sequence_root = scorer_module._planned_reservation_records(planned)
    records_source = b"".join(
        scorer_module.canonical_compact_json_bytes(record) + b"\n"
        for record in records
    )
    (output / "planned_reservations.jsonl").write_bytes(records_source)
    consumption_source = _pretty(
        {
            "schema_version": "facet-local-minilm-approval-consumption-v1",
            "status": "consumed",
            "action": "benchmark",
            "run_id": telemetry["run_id"],
            "approval_file": str(approval_path.resolve()),
            "approval_sha256": approval_sha256,
            "output_path": str(output.resolve()),
        }
    )
    (output / "approval_consumption.json").write_bytes(consumption_source)
    registry = (
        inputs.score_cache_root / ".facet-local-minilm-approval-consumptions"
    )
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{approval_sha256}.json").write_bytes(consumption_source)
    (output / "run_reservation.json").write_bytes(
        _pretty(
            {
                "schema_version": scorer_module.RUN_RESERVATION_SCHEMA_VERSION,
                "status": "reserved",
                "action": "benchmark",
                "run_id": telemetry["run_id"],
                "output_path": str(output.resolve()),
                "approval_sha256": approval_sha256,
                "preflight_sha256": inputs.preflight_sha256,
                "windows_sha256": inputs.windows_sha256,
                "model_materialization_receipt_sha256": (
                    inputs.materialization_receipt_sha256
                ),
                "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
                "planned_row_count": len(records),
                "planned_reservations_bytes": len(records_source),
                "planned_reservations_sha256": _sha256(records_source),
                "reservation_sequence_root_sha256": sequence_root,
            }
        )
    )
    (output / "run_terminal.json").write_bytes(
        _pretty(
            {
                "schema_version": scorer_module.RUN_TERMINAL_SCHEMA_VERSION,
                "status": "complete",
                "action": "benchmark",
                "run_id": telemetry["run_id"],
                "approval_sha256": approval_sha256,
                "output_path": str(output.resolve()),
                "preflight_sha256": inputs.preflight_sha256,
                "windows_sha256": inputs.windows_sha256,
                "model_materialization_receipt_sha256": (
                    inputs.materialization_receipt_sha256
                ),
                "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
                "device": device,
                "execution_backend": backend,
                "forward_pair_count": inputs.benchmark["forward_pair_count"],
                "benchmark_telemetry_sha256": telemetry_sha256,
                "full_inference_request_sha256": _sha256(request_source),
            }
        )
    )
    assert build_full_inference_request(
        inputs,
        telemetry_path,
        expected_telemetry_sha256=telemetry_sha256,
    ) == request
    return path, request


def _full_approval(
    inputs: ScoringInputs,
    request_path: Path,
    output_dir: Path,
    *,
    device="cuda",
    run_id: str = "full-scoring-test-run",
):
    request = json.loads(request_path.read_text())
    return {
        "schema_version": FULL_APPROVAL_SCHEMA_VERSION,
        "approval_scope": FULL_APPROVAL_SCOPE,
        "action": "full_scoring",
        "run_id": run_id,
        "output_path": str(output_dir.resolve()),
        "approved_by": "unit-test-user",
        "device": device,
        "execution_backend": "rocm",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "full_inference_request_file": str(request_path.resolve()),
        "full_inference_request_sha256": _sha256(request_path.read_bytes()),
        "benchmark_telemetry_file": request["benchmark_telemetry_file"],
        "benchmark_telemetry_sha256": request["benchmark_telemetry_sha256"],
        "score_cache_path": str(inputs.score_cache_path),
        "batch_size": BATCH_SIZE,
        "acknowledged_projected_runtime": True,
        "acknowledged_local_files_only": True,
        "acknowledged_no_cpu_fallback": True,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference": True,
}


def _full_cpu_approval(
    inputs: ScoringInputs,
    request_path: Path,
    output_dir: Path,
):
    approval = _full_approval(inputs, request_path, output_dir, device="cpu")
    approval["schema_version"] = FULL_CPU_APPROVAL_SCHEMA_VERSION
    approval["approval_scope"] = FULL_CPU_APPROVAL_SCOPE
    approval["execution_backend"] = "cpu"
    approval.pop("acknowledged_no_cpu_fallback")
    approval["acknowledged_cpu_fallback"] = True
    return approval


def _write_full_approval(
    tmp_path: Path,
    inputs: ScoringInputs,
    request_path: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    run_id: str = "full-scoring-test-run",
) -> tuple[Path, str]:
    payload = (
        _full_cpu_approval(inputs, request_path, output_dir)
        if device == "cpu"
        else _full_approval(
            inputs,
            request_path,
            output_dir,
            run_id=run_id,
        )
    )
    return _write_approval(
        tmp_path / "approvals",
        f"{run_id}.json",
        payload,
    )


class FakeDeviceBatch:
    def __init__(self, count: int) -> None:
        self.count = count
        self.device_calls: list[str] = []

    def to(self, device: str):
        self.device_calls.append(device)
        return self


class FakeLogits:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.shape = (len(values), 1)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, _shape):
        return self

    def tolist(self):
        return [[value] for value in self.values]


class FakeTorch:
    float32 = "float32"

    def __init__(self) -> None:
        self.inference_depth = 0
        self.version = SimpleNamespace(hip="6.4", cuda=None)
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "fake AMD",
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1234,
        )

    @contextmanager
    def inference_mode(self):
        self.inference_depth += 1
        try:
            yield
        finally:
            self.inference_depth -= 1


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str], dict[str, object]]] = []

    def __call__(self, queries, windows, **kwargs):
        self.calls.append((list(queries), list(windows), kwargs))
        return {
            "input_ids": FakeDeviceBatch(len(queries)),
            "attention_mask": FakeDeviceBatch(len(queries)),
        }


class FakeModel:
    def __init__(self, torch: FakeTorch, *, oom: bool = False) -> None:
        self.torch = torch
        self.oom = oom
        self.calls: list[int] = []
        self.float_calls = 0
        self.eval_calls = 0
        self.device_calls: list[str] = []

    def float(self):
        self.float_calls += 1
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.device_calls.append(device)
        return self

    def __call__(self, **encoded):
        assert self.torch.inference_depth == 1
        count = encoded["input_ids"].count
        self.calls.append(count)
        if self.oom:
            raise RuntimeError("HIP out of memory")
        return SimpleNamespace(logits=FakeLogits([index + 0.25 for index in range(count)]))


class Loader:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def from_pretrained(self, path, **kwargs):
        self.calls.append((Path(path), kwargs))
        return self.result


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


@dataclass
class FakeRuntimeBundle:
    runtime: InferenceRuntime
    torch: FakeTorch
    tokenizer: FakeTokenizer
    model: FakeModel
    tokenizer_loader: Loader
    model_loader: Loader
    probes: list[str]


def _runtime(*, oom: bool = False) -> FakeRuntimeBundle:
    torch = FakeTorch()
    tokenizer = FakeTokenizer()
    model = FakeModel(torch, oom=oom)
    tokenizer_loader = Loader(tokenizer)
    model_loader = Loader(model)
    probes: list[str] = []
    runtime = InferenceRuntime(
        torch=torch,
        auto_tokenizer_cls=tokenizer_loader,
        auto_model_cls=model_loader,
        clock=IncrementingClock(),
        host_memory_bytes=lambda: 4567,
        rocm_probe=lambda _torch, device: probes.append(device) or {
            "device_name": "fake AMD",
            "torch_hip_version": "6.4",
        },
    )
    return FakeRuntimeBundle(
        runtime,
        torch,
        tokenizer,
        model,
        tokenizer_loader,
        model_loader,
        probes,
    )


def test_benchmark_refuses_more_than_256_pairs_before_runtime_creation(tmp_path):
    runtime_calls: list[object] = []

    with pytest.raises(ValueError, match="256-pair benchmark ceiling"):
        run_benchmark(
            _inputs(),
            approval={"not": "valid"},
            output_dir=tmp_path / "benchmark",
            pair_limit=257,
            runtime_factory=lambda: runtime_calls.append(object()),
        )

    assert runtime_calls == []


def test_full_run_requires_approval_before_runtime_or_cache_access(tmp_path):
    runtime_calls: list[object] = []
    cache_calls: list[object] = []

    with pytest.raises(ValueError, match="full inference approval"):
        run_full_scoring(
            _inputs(),
            approval=None,
            output_dir=tmp_path / "scoring",
            score_cache_factory=lambda *_args: cache_calls.append(object()),
            runtime_factory=lambda: runtime_calls.append(object()),
        )

    assert cache_calls == []
    assert runtime_calls == []


def test_missing_full_approval_refuses_plain_preflight_fixture_before_load(tmp_path):
    with pytest.raises(ValueError, match="full inference approval"):
        run_full_scoring(
            {"schema_version": "facet-local-minilm-preflight-v2", "topics": []},
            approval=None,
            output_dir=tmp_path / "scoring",
            score_cache_factory=lambda *_args: pytest.fail("cache access"),
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_cache_identity_binds_query_window_and_model():
    context = score_cache_context()
    left = score_cache_key(context, query="facet one", window="same text")
    right = score_cache_key(context, query="facet two", window="same text")
    other_model = replace(context, model="different/model")

    assert left != right
    assert left != score_cache_key(other_model, query="facet one", window="same text")
    assert left == GlobalScoreCache(Path("unused"), context).cache_key(
        query_text="facet one", text="same text"
    )
    assert context.model_revision == MODEL_REVISION
    assert context.inference_dtype == "float32"
    assert context.score_representation == "raw_logits"


def test_strict_benchmark_receipt_binds_every_frozen_hash_before_probe(tmp_path):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )
    output = tmp_path / "benchmark-one"
    approval = _benchmark_approval(inputs, output)
    approval["preflight_sha256"] = "9" * 64
    approval_path, approval_sha256 = _write_approval(
        tmp_path / "approvals", "wrong-hash.json", approval
    )
    bundle = _runtime()

    with pytest.raises(ValueError, match="benchmark approval preflight hash mismatch"):
        run_benchmark(
            inputs,
            approval_path,
            expected_approval_sha256=approval_sha256,
            output_dir=output,
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.probes == []
    assert bundle.model_loader.calls == []

    output = tmp_path / "benchmark-two"
    approval = _benchmark_approval(inputs, output, run_id="benchmark-two")
    approval["unexpected"] = True
    approval_path, approval_sha256 = _write_approval(
        tmp_path / "approvals", "unexpected.json", approval
    )
    with pytest.raises(ValueError, match="benchmark approval fields mismatch"):
        run_benchmark(
            inputs,
            approval_path,
            expected_approval_sha256=approval_sha256,
            output_dir=output,
            runtime_factory=lambda: bundle.runtime,
        )
    assert bundle.probes == []


def test_protected_topics_and_qrels_counters_fail_before_gate_or_cache(tmp_path):
    protected = _inputs(windows=(_window(1, topic_id="144"),))
    with pytest.raises(ValueError, match="protected topic 144"):
        run_full_scoring(
            protected,
            approval=None,
            output_dir=tmp_path / "scoring",
            score_cache_factory=lambda *_args: pytest.fail("cache access"),
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )

    qrels = _inputs(preflight_overrides={"summary": {"qrels_access_count": 1}})
    with pytest.raises(ValueError, match="qrels access counter"):
        run_benchmark(
            qrels,
            approval=None,
            output_dir=tmp_path / "benchmark",
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_primary_benchmark_replays_exact_frozen_224_pairs_and_local_model(tmp_path):
    windows = tuple(_window(index, pair_tokens=100 + index) for index in range(100))
    inputs = _inputs(windows=windows, score_cache_root=tmp_path / "cache")
    output = tmp_path / "benchmark"
    approval, approval_sha256 = _write_benchmark_approval(
        tmp_path, inputs, output
    )
    bundle = _runtime()

    telemetry = run_benchmark(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert bundle.probes == ["cuda"]
    assert sum(bundle.model.calls) == 224
    assert bundle.model.calls == [32] * 7
    assert bundle.model.float_calls == bundle.model.eval_calls == 1
    assert bundle.model.device_calls == ["cuda"]
    assert bundle.tokenizer_loader.calls == [
        (
            inputs.snapshot_path,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]
    assert bundle.model_loader.calls == [
        (
            inputs.snapshot_path,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "torch_dtype": bundle.torch.float32,
            },
        )
    ]
    assert all(len(queries) <= BATCH_SIZE for queries, _windows, _kwargs in bundle.tokenizer.calls)
    assert all(
        kwargs
        == {
            "padding": True,
            "truncation": False,
            "max_length": 512,
            "return_tensors": "pt",
        }
        for _queries, _windows, kwargs in bundle.tokenizer.calls
    )
    assert telemetry["benchmark_sample_sha256"] == inputs.benchmark["sample_sha256"]
    assert telemetry["forward_pair_count"] == 224
    assert telemetry["batch_size"] == 32
    assert telemetry["peak_device_memory_bytes"] == 1234
    assert telemetry["peak_host_memory_bytes"] == 4567


def test_benchmark_rejects_recomputed_sample_mismatch_before_model_load(tmp_path):
    windows = tuple(_window(index) for index in range(100))
    correct = build_benchmark_plan(windows)
    inputs = _inputs(windows=windows, benchmark={**correct, "sample_sha256": "f" * 64})
    bundle = _runtime()

    with pytest.raises(ValueError, match="benchmark sample hash differs from preflight"):
        run_benchmark(
            inputs,
            None,
            output_dir=tmp_path / "benchmark",
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.probes == []
    assert bundle.model_loader.calls == []


def test_cpu_fallback_requires_a_different_approval_before_runtime(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "cpu-rejected"
    payload = _full_approval(inputs, request_path, output)
    approval, approval_sha256 = _write_approval(
        tmp_path / "approvals", "cuda-full.json", payload
    )
    bundle = _runtime()

    with pytest.raises(ValueError, match="CPU fallback requires a separate approval"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            output_dir=output,
            score_cache_root=cache_root,
            device="cpu",
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.probes == []
    assert bundle.model_loader.calls == []


def test_separately_benchmarked_and_approved_cpu_run_is_permitted(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs, device="cpu")
    output = tmp_path / "cpu-scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path,
        inputs,
        request_path,
        output,
        device="cpu",
        run_id="cpu-full-run",
    )
    bundle = _runtime()

    rows = run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        output_dir=output,
        score_cache_root=cache_root,
        device="cpu",
        runtime_factory=lambda: bundle.runtime,
    )

    assert len(rows) == 1
    assert bundle.probes == []
    assert bundle.model.device_calls == ["cpu"]


def test_full_scoring_resumes_hits_scores_unique_misses_and_writes_ledger(tmp_path):
    cache_root = tmp_path / "cache"
    first, second = _window(1), _window(2)
    cache = GlobalScoreCache(cache_root, score_cache_context())
    cache.add_many([(first.query, first.window_text, 9.5)])
    inputs = _inputs(
        windows=(replace(first, cache_hit=True), second),
        score_cache_root=cache_root,
    )
    request_path, _request = _write_full_request(tmp_path, inputs)
    bundle = _runtime()
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    rows = run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        score_cache_root=cache_root,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert len(rows) == 2
    assert all(isinstance(row, MiniLMScoreRow) for row in rows)
    assert [row.disposition for row in rows] == ["cache_hit", "forward_pass"]
    assert rows[0].score == 9.5
    assert bundle.model.calls == [1]
    assert all(row.score_representation == "raw_logits" for row in rows)
    assert all(row.inference_dtype == "float32" for row in rows)
    assert all(row.model_revision == MODEL_REVISION for row in rows)
    assert (output / "scoring_ledger.jsonl").is_file()
    receipt = json.loads((output / "scoring_receipt.json").read_text())
    assert receipt["planned_window_count"] == receipt["completed_window_count"] == 2
    assert receipt["failed_window_count"] == receipt["pending_window_count"] == 0
    assert receipt["cache_hit_count"] == receipt["forward_pass_count"] == 1
    assert len(receipt["ledger_sequence_root_sha256"]) == 64
    reloaded = GlobalScoreCache(cache_root, score_cache_context())
    assert reloaded.get(query_text=first.query, text=first.window_text) == 9.5
    assert reloaded.get(query_text=second.query, text=second.window_text) == 0.25


def test_oom_fails_closed_without_cache_write_or_cpu_retry(tmp_path):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    bundle = _runtime(oom=True)
    output = tmp_path / "oom-scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    with pytest.raises(RuntimeError, match="out of memory.*fail closed.*CPU"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            output_dir=output,
            score_cache_root=cache_root,
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.model.calls == [1]
    assert bundle.model.device_calls == ["cuda"]
    assert GlobalScoreCache(cache_root, score_cache_context()).scores == {}


def test_zero_miss_benchmark_skips_runtime_and_projects_no_inference(tmp_path):
    inputs = _inputs(score_cache_root=tmp_path / "cache")
    output = tmp_path / "benchmark"
    approval, approval_sha256 = _write_benchmark_approval(
        tmp_path, inputs, output
    )
    telemetry = run_benchmark(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        output_dir=output,
        runtime_factory=lambda: pytest.fail("zero misses must not load runtime"),
    )

    assert telemetry["status"] == "cache_complete_no_benchmark"
    assert telemetry["forward_pair_count"] == 0
    assert telemetry["projected_full_run_wall_seconds"] == 0.0


def test_full_output_is_create_only_before_cache_or_runtime(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "existing"
    output.mkdir()
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    with pytest.raises(FileExistsError, match="create-only scoring output"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            score_cache_factory=lambda *_args: pytest.fail("cache access"),
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_canonical_v2_preflight_is_required_before_materialization_verification(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_bytes(
        _pretty(
            {
                "schema_version": "facet-local-minilm-preflight-v1",
                "status": "tokenizer_only_preflight_complete",
            }
        )
    )

    with pytest.raises(ValueError, match="preflight-v2"):
        load_scoring_inputs(preflight)


def _file_binding(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "state": "absent",
            "path": str(path.resolve()),
            "bytes": 0,
            "sha256": None,
        }
    source = path.read_bytes()
    return {
        "state": "present",
        "path": str(path.resolve()),
        "bytes": len(source),
        "sha256": _sha256(source),
    }


def _with_cache_binding(inputs: ScoringInputs) -> ScoringInputs:
    payload = dict(inputs.preflight)
    score_cache = dict(payload["score_cache"])  # type: ignore[arg-type]
    binding = _file_binding(inputs.score_cache_path)
    score_cache["binding"] = binding
    payload["score_cache"] = score_cache
    return replace(inputs, preflight=payload, score_cache_binding=binding)


def test_review_approval_contract_is_file_backed_hash_bound_and_output_bound():
    benchmark = inspect.signature(run_benchmark).parameters
    full = inspect.signature(run_full_scoring).parameters

    assert "expected_approval_sha256" in benchmark
    assert "expected_approval_sha256" in full
    assert benchmark["output_dir"].default is inspect.Parameter.empty
    assert full["output_dir"].default is inspect.Parameter.empty


def test_review_mapping_approval_is_rejected_even_when_benchmark_has_zero_misses(
    tmp_path,
):
    inputs = _with_cache_binding(_inputs(score_cache_root=tmp_path / "cache"))
    output = tmp_path / "benchmark"

    with pytest.raises(ValueError, match="canonical file-backed benchmark approval"):
        run_benchmark(
            inputs,
            _benchmark_approval(inputs, output),
            output_dir=output,
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_review_cache_appearing_after_preflight_is_rejected_before_runtime(tmp_path):
    cache_root = tmp_path / "cache"
    window = replace(_window(1), cache_hit=True)
    inputs = _with_cache_binding(_inputs(windows=(window,), score_cache_root=cache_root))
    cache = GlobalScoreCache(cache_root, score_cache_context())
    cache.add_many([(window.query, window.window_text, 9.5)])
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    with pytest.raises(ValueError, match="score cache changed after preflight"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_review_present_cache_rows_require_exact_context_key_and_float32(tmp_path):
    cache_root = tmp_path / "cache"
    window = replace(_window(1), cache_hit=True)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    inputs.score_cache_path.parent.mkdir(parents=True)
    malformed = {
        "schema_version": GlobalScoreCache.schema_version,
        "backend": inputs.context.backend,
        "backend_version": inputs.context.backend_version,
        "model": "wrong/model",
        "model_revision": inputs.context.model_revision,
        "score_representation": inputs.context.score_representation,
        "inference_dtype": inputs.context.inference_dtype,
        "input_policy": inputs.context.input_policy,
        "max_length": inputs.context.max_length,
        "score_kind": inputs.context.score_kind,
        "cache_key": window.cache_key,
        "query_sha256": _sha256(window.query.encode()),
        "text_sha256": _sha256(window.window_text.encode()),
        "score": 0.1,
    }
    inputs.score_cache_path.write_bytes(
        json.dumps(malformed, sort_keys=True).encode() + b"\n"
    )
    inputs = _with_cache_binding(inputs)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    with pytest.raises(ValueError, match="score cache row context mismatch"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_review_oom_leaves_durable_reservation_and_terminal_failure(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _with_cache_binding(
        _inputs(windows=(_window(1),), score_cache_root=cache_root)
    )
    request_path, _request = _write_full_request(tmp_path, inputs)
    bundle = _runtime(oom=True)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            runtime_factory=lambda: bundle.runtime,
        )

    reservation = json.loads((output / "run_reservation.json").read_text())
    terminal = json.loads((output / "run_terminal.json").read_text())
    assert reservation["planned_row_count"] == 1
    assert len(reservation["reservation_sequence_root_sha256"]) == 64
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "RuntimeError"
    assert not inputs.score_cache_path.exists()


def test_review_cache_write_is_full_atomic_replacement_and_records_hashes(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "cache"
    inputs = _with_cache_binding(
        _inputs(windows=(_window(1),), score_cache_root=cache_root)
    )
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    bundle = _runtime()
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = scorer_module.os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(scorer_module.os, "replace", recording_replace)
    run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        score_cache_root=cache_root,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert replacements and replacements[-1][1] == inputs.score_cache_path
    receipt = json.loads((output / "scoring_receipt.json").read_text())
    assert receipt["cache_before_sha256"] is None
    assert len(receipt["cache_after_sha256"]) == 64
    assert receipt["cache_before_bytes"] == 0
    assert receipt["cache_after_bytes"] == inputs.score_cache_path.stat().st_size
    assert receipt["cache_before_count"] == 0
    assert receipt["cache_after_count"] == 1


def test_review_full_request_rejects_in_memory_or_nonstandard_telemetry(tmp_path):
    inputs = _with_cache_binding(
        _inputs(windows=(_window(1),), score_cache_root=tmp_path / "cache")
    )
    telemetry = {
        "schema_version": "facet-local-minilm-benchmark-telemetry-v1",
        "status": "benchmark_complete",
        "median_pairs_per_second": float("nan"),
    }

    with pytest.raises(ValueError, match="authenticated persisted benchmark telemetry"):
        build_full_inference_request(inputs, telemetry)

    telemetry_path = tmp_path / "benchmark_telemetry.json"
    telemetry_path.write_text('{"median_pairs_per_second": NaN}\n')
    with pytest.raises(ValueError, match="nonstandard JSON constant"):
        scorer_module._loads_object_no_duplicates(
            telemetry_path.read_bytes(), "benchmark telemetry"
        )


def test_review_benchmark_reports_end_to_end_timing_and_fixed_overhead(tmp_path):
    windows = tuple(_window(index, pair_tokens=100 + index) for index in range(100))
    inputs = _with_cache_binding(
        _inputs(windows=windows, score_cache_root=tmp_path / "cache")
    )
    bundle = _runtime()
    output = tmp_path / "benchmark"
    approval, approval_sha256 = _write_benchmark_approval(
        tmp_path, inputs, output
    )
    telemetry = run_benchmark(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert telemetry["timed_scope"] == [
        "tokenization",
        "device_transfer",
        "forward",
        "device_synchronization",
    ]
    assert telemetry["fixed_setup_seconds"] > 0
    assert telemetry["fixed_finalize_seconds"] > 0
    expected = (
        telemetry["fixed_setup_seconds"]
        + telemetry["projected_unique_scoring_seconds"]
        + telemetry["fixed_finalize_seconds"]
    )
    assert telemetry["projected_full_run_wall_seconds"] == pytest.approx(expected)


def test_review_repeated_miss_uses_same_run_reuse_and_deterministic_roots(tmp_path):
    cache_root = tmp_path / "cache"
    first = _window(1)
    repeated = replace(first, rank=2, window_id="d" * 64)
    inputs = _with_cache_binding(
        _inputs(windows=(first, repeated), score_cache_root=cache_root)
    )
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    bundle = _runtime()
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )

    rows = run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        score_cache_root=cache_root,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert [row.disposition for row in rows] == ["forward_pass", "same_run_reuse"]
    assert bundle.model.calls == [1]
    receipt = json.loads((output / "scoring_receipt.json").read_text())
    assert receipt["forward_pass_count"] == 1
    assert receipt["same_run_reuse_count"] == 1
    assert len(receipt["reservation_sequence_root_sha256"]) == 64
    assert len(receipt["ledger_sequence_root_sha256"]) == 64
    assert len(receipt["unique_score_root_sha256"]) == 64
    assert "ledger_root_sha256" not in receipt


def test_review_logits_require_exact_batch_by_one_shape():
    logits = FakeLogits([0.25, 1.25])
    logits.shape = (2,)

    with pytest.raises(ValueError, match=r"exactly \(batch, 1\)"):
        scorer_module._extract_logits(SimpleNamespace(logits=logits), 2)


def test_review_model_load_requests_explicit_float32_dtype(tmp_path):
    inputs = _with_cache_binding(
        _inputs(windows=(_window(1),), score_cache_root=tmp_path / "cache")
    )
    bundle = _runtime()

    scorer_module._load_local_model(inputs, bundle.runtime, device="cuda")

    assert bundle.model_loader.calls[0][1]["torch_dtype"] == bundle.torch.float32


def test_review_out_of_band_hash_mismatch_and_approval_replay_are_rejected(tmp_path):
    inputs = _inputs(score_cache_root=tmp_path / "cache")
    output = tmp_path / "benchmark"
    approval, approval_sha256 = _write_benchmark_approval(
        tmp_path, inputs, output
    )

    with pytest.raises(ValueError, match="out-of-band expected SHA-256"):
        run_benchmark(
            inputs,
            approval,
            expected_approval_sha256="9" * 64,
            output_dir=output,
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )

    run_benchmark(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        output_dir=output,
        runtime_factory=lambda: pytest.fail("zero-miss runtime creation"),
    )
    assert json.loads((output / "run_terminal.json").read_text())["status"] == "complete"

    replay_approval = tmp_path / "copied-approval.json"
    replay_approval.write_bytes(approval.read_bytes())
    for artifact in output.iterdir():
        artifact.unlink()
    output.rmdir()
    with pytest.raises(ValueError, match="approval receipt replay rejected"):
        run_benchmark(
            inputs,
            replay_approval,
            expected_approval_sha256=approval_sha256,
            output_dir=output,
            runtime_factory=lambda: pytest.fail("replay runtime creation"),
        )
    assert json.loads((output / "run_terminal.json").read_text())["status"] == "failed"


def test_review_cache_is_revalidated_under_writer_lock(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )
    bundle = _runtime()
    original_call = FakeModel.__call__

    def mutate_cache_during_forward(model, **encoded):
        result = original_call(model, **encoded)
        GlobalScoreCache(cache_root, score_cache_context()).add_many(
            [(window.query, window.window_text, 9.5)]
        )
        return result

    monkeypatch.setattr(FakeModel, "__call__", mutate_cache_during_forward)

    with pytest.raises(ValueError, match="score cache changed after preflight"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            runtime_factory=lambda: bundle.runtime,
        )

    assert json.loads((output / "run_terminal.json").read_text())["status"] == "failed"
    assert GlobalScoreCache(cache_root, score_cache_context()).get(
        query_text=window.query, text=window.window_text
    ) == 9.5


def test_review_unsealed_cache_transaction_is_never_accepted(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )
    bundle = _runtime()

    monkeypatch.setattr(
        scorer_module,
        "_persist_scoring_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated ledger failure")
        ),
    )
    with pytest.raises(RuntimeError, match="simulated ledger failure"):
        run_full_scoring(
            inputs,
            approval,
            expected_approval_sha256=approval_sha256,
            score_cache_root=cache_root,
            output_dir=output,
            runtime_factory=lambda: bundle.runtime,
        )

    transaction = scorer_module._cache_transaction_path(inputs.score_cache_path)
    assert transaction.is_file()
    assert json.loads((output / "run_terminal.json").read_text())["status"] == "failed"
    rebound = _inputs(
        windows=(replace(window, cache_hit=True),),
        score_cache_root=cache_root,
    )
    with pytest.raises(ValueError, match="unsealed score-cache transaction"):
        scorer_module._load_bound_cache(rebound)


def test_review_persisted_telemetry_projection_and_memory_are_recomputed(tmp_path):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )
    _request_path, request = _write_full_request(tmp_path, inputs)
    telemetry_path = Path(request["benchmark_telemetry_file"])
    telemetry = json.loads(telemetry_path.read_text())
    telemetry["projected_full_run_wall_seconds"] += 1.0
    telemetry["peak_host_memory_bytes"] = -1
    source = _pretty(telemetry)
    telemetry_path.write_bytes(source)

    with pytest.raises(ValueError, match="peak host memory|projection mismatch"):
        build_full_inference_request(
            inputs,
            telemetry_path,
            expected_telemetry_sha256=_sha256(source),
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "absent_consumption",
        "mismatched_reservation",
        "failed_terminal",
        "telemetry_hash",
        "request_hash",
        "request_bytes",
    ),
)
def test_rereview_benchmark_provenance_rejects_incomplete_or_tampered_companions(
    tmp_path,
    corruption,
):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )
    _request_path, request = _write_full_request(tmp_path, inputs)
    telemetry_path = Path(request["benchmark_telemetry_file"])
    benchmark_output = telemetry_path.parent

    if corruption == "absent_consumption":
        (benchmark_output / "approval_consumption.json").unlink()
    elif corruption == "mismatched_reservation":
        path = benchmark_output / "run_reservation.json"
        payload = json.loads(path.read_text())
        payload["run_id"] = "different-run"
        path.write_bytes(_pretty(payload))
    elif corruption == "failed_terminal":
        path = benchmark_output / "run_terminal.json"
        payload = json.loads(path.read_text())
        payload["status"] = "failed"
        path.write_bytes(_pretty(payload))
    elif corruption == "telemetry_hash":
        path = benchmark_output / "run_terminal.json"
        payload = json.loads(path.read_text())
        payload["benchmark_telemetry_sha256"] = "9" * 64
        path.write_bytes(_pretty(payload))
    elif corruption == "request_hash":
        path = benchmark_output / "run_terminal.json"
        payload = json.loads(path.read_text())
        payload["full_inference_request_sha256"] = "9" * 64
        path.write_bytes(_pretty(payload))
    else:
        path = benchmark_output / "full_inference_request.json"
        payload = json.loads(path.read_text())
        payload["projected_full_run_wall_seconds"] += 1.0
        path.write_bytes(_pretty(payload))

    with pytest.raises(ValueError, match="benchmark provenance"):
        build_full_inference_request(
            inputs,
            telemetry_path,
            expected_telemetry_sha256=_sha256(telemetry_path.read_bytes()),
        )


def test_rereview_authentic_complete_benchmark_provenance_builds_request(tmp_path):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )
    _request_path, expected = _write_full_request(tmp_path, inputs)
    telemetry_path = Path(expected["benchmark_telemetry_file"])

    assert build_full_inference_request(
        inputs,
        telemetry_path,
        expected_telemetry_sha256=_sha256(telemetry_path.read_bytes()),
    ) == expected


@pytest.mark.parametrize("pair_limit", (-1, 0))
def test_final_review_provenance_rejects_nonpositive_approval_pair_limit(
    tmp_path,
    pair_limit,
):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )

    with pytest.raises(ValueError, match="benchmark provenance.*pair limit"):
        _write_full_request(
            tmp_path,
            inputs,
            approval_pair_limit=pair_limit,
        )


def test_final_review_provenance_rejects_forward_count_above_approval_limit(
    tmp_path,
):
    windows = tuple(_window(index) for index in range(100))
    inputs = _inputs(windows=windows, score_cache_root=tmp_path / "cache")
    assert inputs.benchmark["forward_pair_count"] == 224

    with pytest.raises(ValueError, match="benchmark provenance.*pair limit"):
        _write_full_request(tmp_path, inputs, approval_pair_limit=1)


@pytest.mark.parametrize("corruption", ("missing", "mismatched"))
def test_final_review_provenance_requires_matching_replay_registry_marker(
    tmp_path,
    corruption,
):
    inputs = _inputs(
        windows=(_window(1),), score_cache_root=tmp_path / "cache"
    )
    _request_path, request = _write_full_request(tmp_path, inputs)
    telemetry_path = Path(request["benchmark_telemetry_file"])
    telemetry = json.loads(telemetry_path.read_text())
    marker = (
        inputs.score_cache_root
        / ".facet-local-minilm-approval-consumptions"
        / f"{telemetry['benchmark_approval_sha256']}.json"
    )
    if corruption == "missing":
        marker.unlink()
    else:
        payload = json.loads(marker.read_text())
        payload["run_id"] = "different-run"
        marker.write_bytes(_pretty(payload))

    with pytest.raises(ValueError, match="benchmark provenance.*registry"):
        build_full_inference_request(
            inputs,
            telemetry_path,
            expected_telemetry_sha256=_sha256(telemetry_path.read_bytes()),
        )


def test_final_review_valid_224_of_256_benchmark_lineage_passes(tmp_path):
    windows = tuple(_window(index) for index in range(100))
    inputs = _inputs(windows=windows, score_cache_root=tmp_path / "cache")
    _request_path, expected = _write_full_request(
        tmp_path,
        inputs,
        approval_pair_limit=256,
    )
    telemetry_path = Path(expected["benchmark_telemetry_file"])

    assert inputs.benchmark["forward_pair_count"] == 224
    assert build_full_inference_request(
        inputs,
        telemetry_path,
        expected_telemetry_sha256=_sha256(telemetry_path.read_bytes()),
    ) == expected


def test_rereview_cleanup_failure_after_complete_terminal_does_not_fail_run(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )
    bundle = _runtime()
    transaction_path = scorer_module._cache_transaction_path(inputs.score_cache_path)
    real_unlink = Path.unlink

    def fail_transaction_cleanup(path, *args, **kwargs):
        if path == transaction_path:
            raise OSError("simulated committed-journal cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_transaction_cleanup)

    rows = run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        score_cache_root=cache_root,
        output_dir=output,
        runtime_factory=lambda: bundle.runtime,
    )

    assert len(rows) == 1
    assert json.loads((output / "run_terminal.json").read_text())["status"] == "complete"
    assert transaction_path.is_file()


def _completed_scoring_with_leftover_transaction(tmp_path):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    output = tmp_path / "scoring"
    approval, approval_sha256 = _write_full_approval(
        tmp_path, inputs, request_path, output
    )
    run_full_scoring(
        inputs,
        approval,
        expected_approval_sha256=approval_sha256,
        score_cache_root=cache_root,
        output_dir=output,
        runtime_factory=lambda: _runtime().runtime,
    )
    scoring_receipt = json.loads((output / "scoring_receipt.json").read_text())
    before = {
        "state": "absent",
        "path": str(inputs.score_cache_path),
        "bytes": scoring_receipt["cache_before_bytes"],
        "sha256": scoring_receipt["cache_before_sha256"],
    }
    after = _file_binding(inputs.score_cache_path)
    transaction = {
        "schema_version": scorer_module.CACHE_TRANSACTION_SCHEMA_VERSION,
        "status": "prepared",
        "action": "full_scoring",
        "run_id": "full-scoring-test-run",
        "approval_sha256": approval_sha256,
        "output_path": str(output.resolve()),
        "cache_before": before,
        "cache_after": after,
        "cache_before_count": scoring_receipt["cache_before_count"],
        "cache_after_count": scoring_receipt["cache_after_count"],
    }
    transaction_path = scorer_module._cache_transaction_path(inputs.score_cache_path)
    transaction_path.write_bytes(_pretty(transaction))
    rebound = _inputs(
        windows=(replace(window, cache_hit=True),),
        score_cache_root=cache_root,
    )
    return rebound, transaction_path, output


def test_rereview_restart_recovers_valid_committed_transaction(tmp_path):
    rebound, transaction_path, _output = _completed_scoring_with_leftover_transaction(
        tmp_path
    )

    snapshot = scorer_module._load_bound_cache(rebound)

    assert len(snapshot.scores) == 1
    assert not transaction_path.exists()


@pytest.mark.parametrize("corruption", ("journal", "cache", "terminal"))
def test_rereview_recovery_rejects_mismatched_transaction_cache_or_terminal(
    tmp_path,
    corruption,
):
    rebound, transaction_path, output = _completed_scoring_with_leftover_transaction(
        tmp_path
    )
    if corruption == "journal":
        transaction = json.loads(transaction_path.read_text())
        transaction["run_id"] = "different-run"
        transaction_path.write_bytes(_pretty(transaction))
    elif corruption == "cache":
        rebound.score_cache_path.write_bytes(
            rebound.score_cache_path.read_bytes() + b"\n"
        )
        rebound = _with_cache_binding(rebound)
    else:
        terminal_path = output / "run_terminal.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["cache_after_sha256"] = "9" * 64
        terminal_path.write_bytes(_pretty(terminal))

    with pytest.raises(ValueError, match="score-cache transaction"):
        scorer_module._load_bound_cache(rebound)

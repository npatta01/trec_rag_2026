from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    )


def _benchmark_approval(inputs: ScoringInputs, *, device: str = "cuda"):
    return {
        "schema_version": BENCHMARK_APPROVAL_SCHEMA_VERSION,
        "approval_scope": BENCHMARK_APPROVAL_SCOPE,
        "approved_by": "unit-test-user",
        "device": device,
        "execution_backend": "rocm",
        "pair_limit": 256,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "acknowledged_maximum_256_pairs": True,
        "acknowledged_local_files_only": True,
        "acknowledged_no_cpu_fallback": True,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference": True,
    }


def _write_full_request(
    tmp_path: Path,
    inputs: ScoringInputs,
    *,
    projected_wall_seconds: float = 60.0,
) -> tuple[Path, dict[str, object]]:
    telemetry = {
        "schema_version": "facet-local-minilm-benchmark-telemetry-v1",
        "status": "benchmark_complete",
        "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
        "median_pairs_per_second": 10.0,
        "projected_full_run_wall_seconds": projected_wall_seconds,
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": (
            inputs.materialization_receipt_sha256
        ),
        "device": "cuda",
        "execution_backend": "rocm",
        "batch_size": BATCH_SIZE,
    }
    request = build_full_inference_request(inputs, telemetry)
    path = tmp_path / "full_inference_request.json"
    path.write_bytes(_pretty(request))
    return path, request


def _full_approval(inputs: ScoringInputs, request_path: Path, *, device="cuda"):
    return {
        "schema_version": FULL_APPROVAL_SCHEMA_VERSION,
        "approval_scope": FULL_APPROVAL_SCOPE,
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
        "score_cache_path": str(inputs.score_cache_path),
        "batch_size": BATCH_SIZE,
        "acknowledged_projected_runtime": True,
        "acknowledged_local_files_only": True,
        "acknowledged_no_cpu_fallback": True,
        "acknowledged_no_qrels_retrieval_network_or_hosted_inference": True,
    }


def _full_cpu_approval(inputs: ScoringInputs, request_path: Path):
    approval = _full_approval(inputs, request_path, device="cpu")
    approval["schema_version"] = FULL_CPU_APPROVAL_SCHEMA_VERSION
    approval["approval_scope"] = FULL_CPU_APPROVAL_SCOPE
    approval["execution_backend"] = "cpu"
    approval.pop("acknowledged_no_cpu_fallback")
    approval["acknowledged_cpu_fallback"] = True
    return approval


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

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, _shape):
        return self

    def tolist(self):
        return list(self.values)


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


def test_benchmark_refuses_more_than_256_pairs_before_runtime_creation():
    runtime_calls: list[object] = []

    with pytest.raises(ValueError, match="256-pair benchmark ceiling"):
        run_benchmark(
            _inputs(),
            approval={"not": "valid"},
            pair_limit=257,
            runtime_factory=lambda: runtime_calls.append(object()),
        )

    assert runtime_calls == []


def test_full_run_requires_approval_before_runtime_or_cache_access():
    runtime_calls: list[object] = []
    cache_calls: list[object] = []

    with pytest.raises(ValueError, match="full inference approval"):
        run_full_scoring(
            _inputs(),
            approval=None,
            score_cache_factory=lambda *_args: cache_calls.append(object()),
            runtime_factory=lambda: runtime_calls.append(object()),
        )

    assert cache_calls == []
    assert runtime_calls == []


def test_missing_full_approval_refuses_plain_preflight_fixture_before_load():
    with pytest.raises(ValueError, match="full inference approval"):
        run_full_scoring(
            {"schema_version": "facet-local-minilm-preflight-v2", "topics": []},
            approval=None,
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


def test_strict_benchmark_receipt_binds_every_frozen_hash_before_probe():
    inputs = _inputs(windows=(_window(1),))
    approval = _benchmark_approval(inputs)
    approval["preflight_sha256"] = "9" * 64
    bundle = _runtime()

    with pytest.raises(ValueError, match="benchmark approval preflight hash mismatch"):
        run_benchmark(inputs, approval, runtime_factory=lambda: bundle.runtime)

    assert bundle.probes == []
    assert bundle.model_loader.calls == []

    approval = _benchmark_approval(inputs)
    approval["unexpected"] = True
    with pytest.raises(ValueError, match="benchmark approval fields mismatch"):
        run_benchmark(inputs, approval, runtime_factory=lambda: bundle.runtime)
    assert bundle.probes == []


def test_protected_topics_and_qrels_counters_fail_before_gate_or_cache():
    protected = _inputs(windows=(_window(1, topic_id="144"),))
    with pytest.raises(ValueError, match="protected topic 144"):
        run_full_scoring(
            protected,
            approval=None,
            score_cache_factory=lambda *_args: pytest.fail("cache access"),
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )

    qrels = _inputs(preflight_overrides={"summary": {"qrels_access_count": 1}})
    with pytest.raises(ValueError, match="qrels access counter"):
        run_benchmark(
            qrels,
            approval=None,
            runtime_factory=lambda: pytest.fail("runtime creation"),
        )


def test_primary_benchmark_replays_exact_frozen_224_pairs_and_local_model():
    windows = tuple(_window(index, pair_tokens=100 + index) for index in range(100))
    inputs = _inputs(windows=windows)
    approval = _benchmark_approval(inputs)
    bundle = _runtime()

    telemetry = run_benchmark(
        inputs,
        approval,
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


def test_benchmark_rejects_recomputed_sample_mismatch_before_model_load():
    windows = tuple(_window(index) for index in range(100))
    correct = build_benchmark_plan(windows)
    inputs = _inputs(windows=windows, benchmark={**correct, "sample_sha256": "f" * 64})
    bundle = _runtime()

    with pytest.raises(ValueError, match="benchmark sample hash differs from preflight"):
        run_benchmark(
            inputs,
            _benchmark_approval(inputs),
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.probes == []
    assert bundle.model_loader.calls == []


def test_cpu_fallback_requires_a_different_approval_before_runtime(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    approval = _full_approval(inputs, request_path)
    bundle = _runtime()

    with pytest.raises(ValueError, match="CPU fallback requires a separate approval"):
        run_full_scoring(
            inputs,
            approval,
            score_cache_root=cache_root,
            device="cpu",
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.probes == []
    assert bundle.model_loader.calls == []


def test_separately_benchmarked_and_approved_cpu_run_is_permitted(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    telemetry = {
        "device": "cpu",
        "execution_backend": "cpu",
        "projected_full_run_wall_seconds": 120.0,
    }
    request = build_full_inference_request(inputs, telemetry)
    request_path = tmp_path / "cpu_full_inference_request.json"
    request_path.write_bytes(_pretty(request))
    bundle = _runtime()

    rows = run_full_scoring(
        inputs,
        _full_cpu_approval(inputs, request_path),
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
    inputs = _inputs(windows=(first, second), score_cache_root=cache_root)
    cache = GlobalScoreCache(cache_root, score_cache_context())
    cache.add_many([(first.query, first.window_text, 9.5)])
    request_path, _request = _write_full_request(tmp_path, inputs)
    approval = _full_approval(inputs, request_path)
    bundle = _runtime()
    output = tmp_path / "scoring"

    rows = run_full_scoring(
        inputs,
        approval,
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
    assert len(receipt["ledger_root_sha256"]) == 64
    reloaded = GlobalScoreCache(cache_root, score_cache_context())
    assert reloaded.get(query_text=first.query, text=first.window_text) == 9.5
    assert reloaded.get(query_text=second.query, text=second.window_text) == 0.25


def test_oom_fails_closed_without_cache_write_or_cpu_retry(tmp_path):
    cache_root = tmp_path / "cache"
    window = _window(1)
    inputs = _inputs(windows=(window,), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    approval = _full_approval(inputs, request_path)
    bundle = _runtime(oom=True)

    with pytest.raises(RuntimeError, match="out of memory.*fail closed.*CPU"):
        run_full_scoring(
            inputs,
            approval,
            score_cache_root=cache_root,
            runtime_factory=lambda: bundle.runtime,
        )

    assert bundle.model.calls == [1]
    assert bundle.model.device_calls == ["cuda"]
    assert GlobalScoreCache(cache_root, score_cache_context()).scores == {}


def test_zero_miss_benchmark_skips_runtime_and_projects_no_inference():
    inputs = _inputs()
    telemetry = run_benchmark(
        inputs,
        approval=None,
        runtime_factory=lambda: pytest.fail("zero misses must not load runtime"),
    )

    assert telemetry["status"] == "cache_complete_no_benchmark"
    assert telemetry["forward_pair_count"] == 0
    assert telemetry["projected_full_run_wall_seconds"] == 0.0


def test_full_output_is_create_only_before_cache_or_runtime(tmp_path):
    cache_root = tmp_path / "cache"
    inputs = _inputs(windows=(_window(1),), score_cache_root=cache_root)
    request_path, _request = _write_full_request(tmp_path, inputs)
    approval = _full_approval(inputs, request_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="create-only scoring output"):
        run_full_scoring(
            inputs,
            approval,
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

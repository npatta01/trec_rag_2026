from __future__ import annotations

import asyncio
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from trec_rag.runpod_passage_scorer import (
    MAX_REQUEST_BYTES,
    REMOTE_MODEL_BATCH_SIZE,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    remote_endpoint_identity,
    remote_endpoint_identity_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_PATH = ROOT / "code" / "tools" / "runpod_flash_passage_scorer" / "main.py"


class FakeParameter:
    dtype = "torch.bfloat16"
    device = "cuda:0"


class FakeModel:
    def __init__(self, scores: list[object]) -> None:
        self.scores = scores
        self.model = self
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def parameters(self):
        return iter((FakeParameter(),))

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), dict(kwargs)))
        return list(self.scores)


class FailingModel(FakeModel):
    def predict(self, pairs, **kwargs):
        raise RuntimeError("private query and passage leaked by model")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(query: str, text: str) -> str:
    payload = {
        "identity_sha256": remote_endpoint_identity_sha256(),
        "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
        "passage_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }
    return sha256(_canonical(payload)).hexdigest()


def _request(*texts: str, query: str = "private query") -> dict[str, object]:
    passages = [
        {"content_id": _content_id(query, text), "text": text}
        for text in texts
    ]
    body: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "expected_identity_sha256": remote_endpoint_identity_sha256(),
        "query": query,
        "passages": passages,
    }
    return {**body, "request_id": sha256(_canonical(body)).hexdigest()}


@pytest.fixture
def endpoint_module(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeNetworkVolume:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            captured["volume"] = kwargs

    class FakeEndpoint:
        def __init__(self, **kwargs: object) -> None:
            captured["endpoint"] = kwargs

        def __call__(self, function):
            return function

    fake_flash = SimpleNamespace(
        DataCenter=SimpleNamespace(EU_RO_1="EU_RO_1", US_NC_2="US_NC_2"),
        Endpoint=FakeEndpoint,
        GpuType=SimpleNamespace(NVIDIA_GEFORCE_RTX_4090="RTX_4090"),
        NetworkVolume=FakeNetworkVolume,
    )
    monkeypatch.setitem(sys.modules, "runpod_flash", fake_flash)
    spec = importlib.util.spec_from_file_location(
        "test_runpod_flash_passage_endpoint_module",
        ENDPOINT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._TEST_CAPTURED_CONFIG = captured
    return module


def test_endpoint_configuration_scales_to_zero_and_persists_model_cache(
    endpoint_module,
) -> None:
    captured = endpoint_module._TEST_CAPTURED_CONFIG
    endpoint = captured["endpoint"]

    assert endpoint["name"] == "trec-rag-mixedbread-passage-scorer-v1"
    assert endpoint["gpu"] == "RTX_4090"
    assert endpoint["workers"] == (0, 3)
    assert endpoint["idle_timeout"] == 900
    assert endpoint["max_concurrency"] == 1
    assert endpoint["flashboot"] is True
    assert endpoint["execution_timeout_ms"] == 900_000
    assert endpoint["datacenter"] == "EU_RO_1"
    assert endpoint["env"] == {
        "HF_HUB_CACHE": "/runpod-volume/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert captured["volume"] == {
        "name": "trec-rag-mixedbread-model-cache-v1",
        "size": 50,
        "datacenter": "EU_RO_1",
    }


def test_endpoint_scores_ordered_passages_with_raw_logits(endpoint_module) -> None:
    model = FakeModel([0.75, -0.25])
    endpoint_module._MODEL = model
    request = _request("private first passage", "private second passage")

    response = asyncio.run(endpoint_module.score_batch(request))

    assert response["schema_version"] == RESPONSE_SCHEMA_VERSION
    assert response["request_id"] == request["request_id"]
    assert response["identity"] == remote_endpoint_identity()
    assert response["identity_sha256"] == remote_endpoint_identity_sha256()
    assert response["results"] == [
        {
            "content_id": request["passages"][0]["content_id"],
            "score": 0.75,
        },
        {
            "content_id": request["passages"][1]["content_id"],
            "score": -0.25,
        },
    ]
    pairs, kwargs = model.calls[0]
    assert pairs == [
        ("private query", "private first passage"),
        ("private query", "private second passage"),
    ]
    assert kwargs["batch_size"] == REMOTE_MODEL_BATCH_SIZE
    assert kwargs["show_progress_bar"] is False
    assert kwargs["convert_to_tensor"] is True
    assert kwargs["activation_fn"](3.25) == 3.25


def test_endpoint_loads_model_once_per_warm_worker(
    endpoint_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([0.5])
    load_calls: list[tuple[str, dict[str, object]]] = []

    def cross_encoder(model_name: str, **kwargs: object) -> FakeModel:
        load_calls.append((model_name, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(__version__="5.6.0", CrossEncoder=cross_encoder),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.9.1+cu128",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "NVIDIA GeForce RTX 4090",
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.13.0"),
    )
    if hasattr(endpoint_module, "_MODEL"):
        del endpoint_module._MODEL

    asyncio.run(endpoint_module.score_batch(_request("first")))
    asyncio.run(endpoint_module.score_batch(_request("second")))

    assert len(load_calls) == 1
    assert load_calls[0][1]["revision"] == (
        "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
    )
    assert load_calls[0][1]["max_length"] == 1024
    assert load_calls[0][1]["device"] == "cuda"
    assert load_calls[0][1]["local_files_only"] is False
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    (
        "sentence_transformers_version",
        "torch_version",
        "transformers_version",
        "gpu",
        "message",
    ),
    [
        (
            "5.6.1",
            "2.9.1+cu128",
            "5.13.0",
            "NVIDIA GeForce RTX 4090",
            "sentence-transformers version",
        ),
        (
            "5.6.0",
            "2.9.2+cu128",
            "5.13.0",
            "NVIDIA GeForce RTX 4090",
            "torch version",
        ),
        (
            "5.6.0",
            "2.9.1+cu128",
            "5.13.1",
            "NVIDIA GeForce RTX 4090",
            "transformers version",
        ),
        ("5.6.0", "2.9.1+cu128", "5.13.0", "NVIDIA RTX A6000", "CUDA device"),
    ],
)
def test_endpoint_rejects_runtime_that_differs_from_cache_identity(
    endpoint_module,
    monkeypatch: pytest.MonkeyPatch,
    sentence_transformers_version: str,
    torch_version: str,
    transformers_version: str,
    gpu: str,
    message: str,
) -> None:
    model = FakeModel([0.5])
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            __version__=sentence_transformers_version,
            CrossEncoder=lambda *_args, **_kwargs: model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__=torch_version,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: gpu,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__=transformers_version),
    )
    if hasattr(endpoint_module, "_MODEL"):
        del endpoint_module._MODEL

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(endpoint_module.score_batch(_request("private passage")))

    assert model.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda request: request.update(passages=[]), "passages"),
        (
            lambda request: request.update(expected_identity_sha256="0" * 64),
            "identity",
        ),
        (lambda request: request.update(request_id="0" * 64), "request_id"),
    ],
)
def test_endpoint_rejects_malformed_requests(
    endpoint_module,
    mutation,
    message: str,
) -> None:
    endpoint_module._MODEL = FakeModel([0.5])
    request = _request("private passage")
    mutation(request)

    with pytest.raises(ValueError, match=message):
        asyncio.run(endpoint_module.score_batch(request))


def test_endpoint_rejects_oversized_request_before_model_inference(endpoint_module) -> None:
    model = FakeModel([0.5])
    endpoint_module._MODEL = model
    request = _request("x" * MAX_REQUEST_BYTES)

    with pytest.raises(ValueError, match="6 MiB"):
        asyncio.run(endpoint_module.score_batch(request))

    assert model.calls == []


@pytest.mark.parametrize("score", [True, math.nan, math.inf, "0.5"])
def test_endpoint_rejects_invalid_model_scores(endpoint_module, score: object) -> None:
    endpoint_module._MODEL = FakeModel([score])

    with pytest.raises(ValueError, match="finite real"):
        asyncio.run(endpoint_module.score_batch(_request("private passage")))


def test_endpoint_suppresses_model_exception_details(endpoint_module) -> None:
    endpoint_module._MODEL = FailingModel([])

    with pytest.raises(RuntimeError, match="model inference failed") as captured:
        asyncio.run(endpoint_module.score_batch(_request("private passage")))

    assert "private" not in str(captured.value)
    assert captured.value.__suppress_context__ is True

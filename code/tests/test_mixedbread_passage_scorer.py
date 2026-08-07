from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag.chunking import TextChunk
from trec_rag.mixedbread_passage_scorer import (
    INPUT_POLICY,
    MAX_LENGTH,
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
    MixedbreadPassageScorer,
    load_pinned_cross_encoder,
)
from trec_rag.rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    ScoreCacheMiss,
)


class _Parameter:
    dtype = "torch.bfloat16"


class FakeModel:
    def __init__(self, scores: list[object]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []
        self.predict_kwargs: list[dict[str, object]] = []

    def parameters(self):
        return [_Parameter()]

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        self.predict_kwargs.append(kwargs)
        return self.scores


class WrongDtypeModel(FakeModel):
    def parameters(self):
        return [SimpleNamespace(dtype="torch.float32")]


def chunks(*texts: str) -> tuple[TextChunk, ...]:
    return tuple(
        TextChunk(
            document_id="doc-a",
            chunk_id=f"doc-a:{index:04d}",
            text=text,
            start_char=index,
            end_char=index + len(text),
        )
        for index, text in enumerate(texts)
    )


def fake_loader(scores: list[object]):
    model = FakeModel(scores)
    return model, (lambda **_kwargs: model)


def test_scorer_reuses_cached_passage_scores_without_loading_model(tmp_path: Path) -> None:
    first_model = FakeModel([0.1, 0.9])
    first = MixedbreadPassageScorer(tmp_path, model_loader=lambda **_: first_model)
    expected = first.rank("query", chunks("alpha", "beta"))
    replay = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    assert replay.rank("query", chunks("alpha", "beta")) == expected


def test_scorer_preserves_input_order_across_hits_and_misses(tmp_path: Path) -> None:
    model = FakeModel([0.4, 0.2])
    scorer = MixedbreadPassageScorer(tmp_path, model_loader=lambda **_: model)
    rows = scorer.rank("query", chunks("second", "first"))
    assert [row.chunk.text for row in rows] == ["second", "first"]


def test_scorer_exposes_cumulative_cache_and_model_batch_accounting(tmp_path: Path) -> None:
    seed = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: FakeModel([0.1]),
    )
    seed.rank("query", chunks("cached"))
    model = FakeModel([0.7, 0.8])
    scorer = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: model,
        batch_size=2,
    )

    scorer.rank("query", chunks("cached", "new-a", "new-b"))

    assert scorer.stats == {"cache_hits": 1, "cache_misses": 2, "model_batches": 1}
    assert model.calls == [[("query", "new-a"), ("query", "new-b")]]

    scorer.rank("query", chunks("cached", "new-a", "new-b"))
    assert scorer.stats == {"cache_hits": 4, "cache_misses": 2, "model_batches": 1}
    assert len(model.calls) == 1


def test_scorer_does_not_count_model_batch_when_model_loading_fails(tmp_path: Path) -> None:
    def failing_loader(**_kwargs: object) -> object:
        raise RuntimeError("model load failed")

    scorer = MixedbreadPassageScorer(tmp_path, model_loader=failing_loader)

    with pytest.raises(RuntimeError, match="model load failed"):
        scorer.rank("query", chunks("passage"))

    assert scorer.stats == {"cache_hits": 0, "cache_misses": 1, "model_batches": 0}


def test_scorer_read_only_mode_never_loads_model_and_accounts_for_misses(tmp_path: Path) -> None:
    seed = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: FakeModel([0.1]),
    )
    seed.rank("query", chunks("cached"))
    seed.score_cache.close()
    read_only = MixedbreadPassageScorer(
        tmp_path,
        read_only=True,
        model_loader=lambda **_: pytest.fail("read-only scorer loaded the model"),
    )

    assert read_only.rank("query", chunks("cached"))[0].relevance_score == 0.1
    with pytest.raises(ScoreCacheMiss) as captured:
        read_only.rank("query", chunks("missing"))

    assert captured.value.missing[0].cache_key == read_only.cache_key("query", "missing")
    assert read_only.stats == {"cache_hits": 1, "cache_misses": 1, "model_batches": 0}


def test_scorer_exposes_pinned_json_identity_and_exact_cache_key(tmp_path: Path) -> None:
    scorer = MixedbreadPassageScorer(tmp_path, device="cpu")

    json.dumps(scorer.identity, sort_keys=True)
    assert scorer.identity == {
        "backend": "sentence-transformers-cross-encoder",
        "backend_version": DEFAULT_BACKEND_VERSION,
        "model": MIXEDBREAD_MODEL,
        "model_revision": MIXEDBREAD_REVISION,
        "score_representation": "raw_logits",
        "inference_dtype": DEFAULT_INFERENCE_DTYPE,
        "max_length": MAX_LENGTH,
        "batch_size": 8,
        "device": "cpu",
        "input_policy": INPUT_POLICY,
        "implementation_version": 1,
    }
    assert scorer.cache_key("query", "passage") == scorer.score_cache.cache_key(
        query_text="query", text="passage"
    )


def test_scorer_loads_pinned_model_lazily_with_raw_logits(tmp_path: Path) -> None:
    model, loader = fake_loader([0.7])
    calls: list[dict[str, object]] = []

    def recording_loader(**kwargs: object):
        calls.append(kwargs)
        return loader(**kwargs)

    scorer = MixedbreadPassageScorer(
        tmp_path,
        device="cpu",
        model_loader=recording_loader,
        batch_size=4,
    )
    assert calls == []
    assert scorer.rank("query", chunks("passage"))[0].relevance_score == 0.7
    assert calls == [
        {
            "model_name": MIXEDBREAD_MODEL,
            "revision": MIXEDBREAD_REVISION,
            "max_length": MAX_LENGTH,
            "device": "cpu",
            "local_files_only": True,
        }
    ]
    assert model.predict_kwargs[0]["activation_fn"](3.25) == 3.25
    assert model.predict_kwargs[0]["batch_size"] == 4


def test_scorer_does_not_retain_model_that_fails_dtype_validation(tmp_path: Path) -> None:
    invalid = WrongDtypeModel([9.9])
    replacement = FakeModel([0.7])
    models = iter((invalid, replacement))
    loader_calls = 0

    def loader(**_kwargs: object) -> FakeModel:
        nonlocal loader_calls
        loader_calls += 1
        return next(models)

    scorer = MixedbreadPassageScorer(tmp_path, model_loader=loader)

    with pytest.raises(RuntimeError, match="model dtype does not match cache context"):
        scorer.rank("query", chunks("passage"))

    rows = scorer.rank("query", chunks("passage"))
    assert [row.relevance_score for row in rows] == [0.7]
    assert loader_calls == 2
    assert invalid.calls == []
    assert replacement.calls == [[("query", "passage")]]


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([True], "Boolean"),
        ([math.nan], "finite"),
        ([0.1, 0.2], "count"),
    ],
)
def test_scorer_rejects_invalid_model_outputs(
    tmp_path: Path, scores: list[object], message: str
) -> None:
    model = FakeModel(scores)
    scorer = MixedbreadPassageScorer(tmp_path, model_loader=lambda **_: model)

    with pytest.raises(ValueError, match=message):
        scorer.rank("query", chunks("passage"))
    assert scorer.score_cache.scores == {}


@pytest.mark.parametrize(
    ("cached_scores", "message"),
    [
        ([True], "finite"),
        ([math.inf], "finite"),
        ([], "count"),
    ],
)
def test_scorer_rejects_invalid_cached_outputs(
    tmp_path: Path,
    cached_scores: list[object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: pytest.fail("cache hit must not load the model"),
    )
    monkeypatch.setattr(scorer.score_cache, "score_many", lambda *_args, **_kwargs: cached_scores)

    with pytest.raises(ValueError, match=message):
        scorer.rank("query", chunks("passage"))


def test_load_pinned_cross_encoder_uses_offline_pinned_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel([0.1])
    calls: list[tuple[str, dict[str, object]]] = []

    def cross_encoder(model_name: str, **kwargs: object):
        calls.append((model_name, kwargs))
        return model

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=cross_encoder),
    )

    assert load_pinned_cross_encoder(device="cpu") is model
    assert calls == [
        (
            MIXEDBREAD_MODEL,
            {
                "revision": MIXEDBREAD_REVISION,
                "max_length": MAX_LENGTH,
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]

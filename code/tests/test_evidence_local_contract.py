from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import trec_rag.evidence_local as evidence_local
from trec_rag.evidence_local import (
    LocalCacheAccounting,
    MINILM_MODEL,
    MINILM_REVISION,
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
    LocalMiniLMSimilarity,
    MixedbreadSentencePairScorer,
    _load_local_cross_encoder,
)
from trec_rag.facet_evidence import SentencePair


class _Parameter:
    dtype = "torch.bfloat16"


class _RawLogitModel:
    def __init__(self, scores=(1.25, -2.5)) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def parameters(self):
        return [_Parameter()]

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        assert kwargs["activation_fn"](2.5) == 2.5
        return self.scores


def test_mixedbread_constructor_requires_the_pinned_offline_snapshot(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def cross_encoder(name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=cross_encoder),
    )

    assert _load_local_cross_encoder(
        MIXEDBREAD_MODEL,
        revision=MIXEDBREAD_REVISION,
        max_length=512,
        device="cpu",
        local_files_only=True,
    ) is sentinel
    assert calls == [(MIXEDBREAD_MODEL, {
        "revision": MIXEDBREAD_REVISION,
        "max_length": 512,
        "device": "cpu",
        "local_files_only": True,
    })]
    with pytest.raises(ValueError, match="local_files_only=True"):
        _load_local_cross_encoder(
            MIXEDBREAD_MODEL,
            revision=MIXEDBREAD_REVISION,
            max_length=512,
            device="cpu",
            local_files_only=False,
        )


def test_mixedbread_real_cache_deduplicates_and_restores_pair_order(tmp_path) -> None:
    model = _RawLogitModel()
    loads = []

    def loader(name, **kwargs):
        loads.append((name, kwargs))
        return model

    scorer = MixedbreadSentencePairScorer(
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=loader,
        batch_size=8,
    )
    duplicate = SentencePair(
        "224", "doc-a", "safety", "Safety evidence", "First sentence."
    )
    other = SentencePair(
        "224", "doc-a", "safety", "Safety evidence", "Second sentence."
    )

    assert scorer.score_pairs((duplicate, other, duplicate)) == (1.25, -2.5, 1.25)
    assert model.calls == [[
        ("Safety evidence", "First sentence."),
        ("Safety evidence", "Second sentence."),
    ]]
    assert loads == [(MIXEDBREAD_MODEL, {
        "revision": MIXEDBREAD_REVISION,
        "max_length": 512,
        "device": "cpu",
        "local_files_only": True,
    })]

    assert scorer.score_pairs((other, duplicate)) == (-2.5, 1.25)
    assert len(model.calls) == 1


def test_mixedbread_rejects_boolean_cache_and_model_scores(tmp_path) -> None:
    cached = MixedbreadSentencePairScorer(
        score_cache_root=tmp_path / "cached",
        model_loader=lambda *_args, **_kwargs: pytest.fail("model must stay lazy"),
    )
    with pytest.raises(ValueError, match="bool"):
        cached.score_cache.add_many((("query", "text", True),))
    assert cached.score_cache.connection.execute(
        "SELECT count(*) FROM scores"
    ).fetchone()[0] == 0

    model = _RawLogitModel(scores=(True,))
    scorer = MixedbreadSentencePairScorer(
        score_cache_root=tmp_path / "model",
        model_loader=lambda *_args, **_kwargs: model,
    )
    pair = SentencePair("224", "doc-a", "safety", "Safety evidence", "Sentence.")
    with pytest.raises(ValueError, match="Boolean"):
        scorer.score_pairs((pair,))
    assert scorer.score_cache.connection.execute(
        "SELECT count(*) FROM scores"
    ).fetchone()[0] == 0


def test_minilm_resolves_device_and_computes_pinned_offline_cosine(monkeypatch) -> None:
    device_calls = []
    loader_calls = []

    class Model:
        def __init__(self) -> None:
            self.encode_calls = []

        def encode(self, texts, **kwargs):
            self.encode_calls.append((tuple(texts), kwargs))
            return ((3.0, 4.0), (0.0, 2.0))

    model = Model()
    monkeypatch.setattr(
        evidence_local,
        "_choose_device",
        lambda value: device_calls.append(value) or "cpu",
    )

    def loader(model_name, **kwargs):
        loader_calls.append((model_name, kwargs))
        return model

    similarity = LocalMiniLMSimilarity(loader=loader, device="auto", batch_size=7)
    assert loader_calls == []
    assert similarity.identity == {
        "backend": "sentence-transformers",
        "embedding_representation": "l2_normalized_float",
        "local_files_only": True,
        "model": MINILM_MODEL,
        "model_revision": MINILM_REVISION,
        "score_kind": "normalized_vector_cosine",
        "batch_size": 7,
        "device": "cpu",
    }

    matrix = similarity.cosine_matrix(("first", "second"))

    assert device_calls == ["auto"]
    assert loader_calls == [(MINILM_MODEL, {
        "revision": MINILM_REVISION,
        "local_files_only": True,
        "device": "cpu",
    })]
    assert model.encode_calls == [(('first', 'second'), {
        "batch_size": 7,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    })]
    assert matrix[0] == pytest.approx((1.0, 0.8))
    assert matrix[1] == pytest.approx((0.8, 1.0))


def test_minilm_similarity_cache_replays_without_loading_model(tmp_path) -> None:
    """Catches a warm portable matrix cache loading the local embedding model."""
    class Model:
        def encode(self, texts, **_kwargs):
            assert tuple(texts) == ("first", "second")
            return ((1.0, 0.0), (0.0, 1.0))

    online = LocalMiniLMSimilarity(
        loader=lambda *_args, **_kwargs: Model(),
        device="cpu",
        batch_size=1,
        cache_root=tmp_path / "cache",
    )
    first = online.cosine_matrix(("first", "second"))

    offline = LocalMiniLMSimilarity(
        loader=lambda *_args, **_kwargs: pytest.fail("cache hit must not load model"),
        device="cpu",
        batch_size=7,
        cache_root=tmp_path / "cache",
        cache_only=True,
    )
    second = offline.cosine_matrix(("first", "second"))

    assert second == pytest.approx(first)
    assert online.accounting == LocalCacheAccounting(
        cache_hits=0, cache_misses=1, model_batches=2
    )
    assert offline.accounting == LocalCacheAccounting(
        cache_hits=1, cache_misses=0, model_batches=0
    )


def test_minilm_cache_only_miss_is_side_effect_free_and_model_lazy(tmp_path) -> None:
    """Catches a missing offline matrix creating cache state or loading a model."""
    from trec_rag.similarity_cache import SimilarityCacheMiss

    root = tmp_path / "absent"
    similarity = LocalMiniLMSimilarity(
        loader=lambda *_args, **_kwargs: pytest.fail("offline miss must not load model"),
        device="cpu",
        cache_root=root,
        cache_only=True,
    )

    with pytest.raises(SimilarityCacheMiss, match="similarity cache miss"):
        similarity.cosine_matrix(("first", "second"))

    assert similarity.accounting == LocalCacheAccounting(
        cache_hits=0, cache_misses=1, model_batches=0
    )
    assert not root.exists()

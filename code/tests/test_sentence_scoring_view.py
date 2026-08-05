from __future__ import annotations

import math

import pytest

from trec_rag.evidence_local import MixedbreadSentencePairScorer
from trec_rag.facet_evidence import SCORING_NORMALIZATION_VERSION, SentencePair


class _Parameter:
    dtype = "torch.bfloat16"


class _RecordingModel:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = [] if responses is None else list(responses)
        self.calls: list[list[tuple[str, str]]] = []

    def parameters(self):
        return [_Parameter()]

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        assert kwargs["activation_fn"](2.5) == 2.5
        if self.responses:
            return self.responses.pop(0)
        return tuple(float(index + 1) for index in range(len(self.calls[-1])))


def _scorer(tmp_path, model, *, batch_size: int = 32, loads=None):
    if loads is None:
        loads = []

    def loader(name, **kwargs):
        loads.append((name, kwargs))
        return model

    return MixedbreadSentencePairScorer(
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=loader,
        batch_size=batch_size,
    )


def _pair(sentence_text: str, *, query_text: str = "query") -> SentencePair:
    return SentencePair("topic", "document", "subnarrative", query_text, sentence_text)


def test_scoring_view_is_the_only_model_and_cache_sentence_text(tmp_path) -> None:
    model = _RecordingModel()
    exact_sentence = "Alpha\t\r\nBeta\u2003Gamma"
    pair = _pair(exact_sentence, query_text="query\ttext")
    equivalent = _pair("Alpha Beta Gamma", query_text="query\ttext")
    scorer = _scorer(tmp_path, model)

    assert scorer.score_pairs((pair, equivalent, pair)) == (1.0, 1.0, 1.0)
    assert pair == _pair(exact_sentence, query_text="query\ttext")
    assert pair.sentence_text == exact_sentence
    assert model.calls == [[("query\ttext", "Alpha Beta Gamma")]]


def test_warm_cache_hit_does_not_load_or_call_model(tmp_path) -> None:
    first_model = _RecordingModel(responses=[(4.25,)])
    first = _scorer(tmp_path, first_model)
    pair = _pair("Warm\t\r\ncache")
    assert first.score_pairs((pair,)) == (4.25,)

    cold_model = _RecordingModel()
    loads = []
    second = _scorer(tmp_path, cold_model, loads=loads)

    assert second.score_pairs((pair,)) == (4.25,)
    assert loads == []
    assert cold_model.calls == []


def test_partial_cache_hit_batches_only_unique_misses(tmp_path) -> None:
    warm_model = _RecordingModel(responses=[(9.0,)])
    warm = _scorer(tmp_path, warm_model)
    warm_pair = _pair("already\tcached")
    assert warm.score_pairs((warm_pair,)) == (9.0,)

    model = _RecordingModel()
    scorer = _scorer(tmp_path, model, batch_size=2)
    miss_one = _pair("miss one")
    miss_two = _pair("miss\u2003two")
    miss_three = _pair("miss three")

    scores = scorer.score_pairs((warm_pair, miss_one, miss_two, miss_three))

    assert scores == (9.0, 1.0, 2.0, 1.0)
    assert model.calls == [
        [("query", "miss one"), ("query", "miss two")],
        [("query", "miss three")],
    ]


def test_score_pairs_makes_one_global_cache_request_and_restores_order(tmp_path, monkeypatch) -> None:
    model = _RecordingModel()
    scorer = _scorer(tmp_path, model, batch_size=2)
    original_score_many = scorer.score_cache.score_many
    requests = []

    def score_many(pairs, compute_batch, batch_size, *args, **kwargs):
        materialized = tuple(pairs)
        requests.append((materialized, batch_size))
        return original_score_many(materialized, compute_batch, batch_size, *args, **kwargs)

    monkeypatch.setattr(scorer.score_cache, "score_many", score_many)
    first = _pair("first\ttext")
    second = _pair("second\r\ntext")

    assert scorer.score_pairs((second, first, second)) == (1.0, 2.0, 1.0)
    assert requests == [
        ((
            ("query", "second text"),
            ("query", "first text"),
            ("query", "second text"),
        ), 2)
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    (
        ((True,), "Boolean"),
        ((math.nan,), "finite"),
        ((), "count"),
    ),
)
def test_invalid_model_scores_are_rejected_by_cache_boundary(tmp_path, response, message) -> None:
    model = _RecordingModel(responses=[response])
    scorer = _scorer(tmp_path, model)

    with pytest.raises(ValueError, match=message):
        scorer.score_pairs((_pair("invalid score"),))


def test_empty_scoring_view_is_rejected_before_cache_or_model_access(tmp_path, monkeypatch) -> None:
    model = _RecordingModel()
    scorer = _scorer(tmp_path, model)
    monkeypatch.setattr(
        scorer.score_cache,
        "score_many",
        lambda *_args, **_kwargs: pytest.fail("cache must not be accessed"),
    )

    with pytest.raises(ValueError, match="empty"):
        scorer.score_pairs((_pair("\t\r\n\u2003"),))

    assert model.calls == []


def test_identity_names_effective_scoring_normalization_policy(tmp_path) -> None:
    scorer = _scorer(tmp_path, _RecordingModel())

    assert scorer.identity["input_policy"] == SCORING_NORMALIZATION_VERSION
    assert scorer.score_cache.context.input_policy == SCORING_NORMALIZATION_VERSION

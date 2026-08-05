from __future__ import annotations

import hashlib
import json

import pytest

from trec_rag.facet_evidence import SCORING_NORMALIZATION_VERSION, ScoringView
from trec_rag.facet_retrieval import (
    DOCUMENT_MAX_LENGTH,
    DOCUMENT_PAIR_BUFFER_TOKENS,
    WINDOW_MAX_LENGTH,
    FacetRetrievalLane,
    MixedbreadCoverageScorer,
)
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.topics import Topic


class _Parameter:
    dtype = "torch.bfloat16"


class _CapturingModel:
    def __init__(self, calls: list[list[tuple[str, str]]], score: float) -> None:
        self._calls = calls
        self._score = score

    def parameters(self):
        return [_Parameter()]

    def predict(self, pairs, **_kwargs):
        materialized = [tuple(pair) for pair in pairs]
        self._calls.append(materialized)
        return [self._score] * len(materialized)


def _lane(topic: Topic) -> FacetRetrievalLane:
    query = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    return FacetRetrievalLane(query, query, None)


def _candidate(topic: Topic, source: str, *, docid: str = "doc-a", rank: int = 1):
    return RetrievedCandidate(
        topic.id,
        "original",
        "climbmix_bm25",
        topic.narrative,
        docid,
        rank,
        float(100 - rank),
        source,
    )


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_coverage_scorer_uses_whitespace_view_but_returns_exact_source(tmp_path) -> None:
    topic = Topic("14", "", "Exact scoring-view test")
    lane = _lane(topic)
    source = "  Café\tBeta\r\nGamma\u2003Δelta  "
    view = ScoringView(source)
    model_calls: dict[int, list[list[tuple[str, str]]]] = {
        DOCUMENT_MAX_LENGTH - DOCUMENT_PAIR_BUFFER_TOKENS: [],
        WINDOW_MAX_LENGTH: [],
    }

    def loader(_model, *, revision, max_length, device):
        assert revision
        assert device == "cpu"
        return _CapturingModel(model_calls[max_length], 2.0 if max_length != WINDOW_MAX_LENGTH else 3.0)

    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "ledger",
        score_cache_root=tmp_path / "cache",
        device="cpu",
        model_loader=loader,
    )
    result = scorer.score_lane(topic, lane, [_candidate(topic, source)])

    assert scorer.document_cache.context.input_policy == SCORING_NORMALIZATION_VERSION
    assert scorer.window_cache.context.input_policy == SCORING_NORMALIZATION_VERSION
    assert scorer.identity["input_policy"] == SCORING_NORMALIZATION_VERSION
    assert model_calls[DOCUMENT_MAX_LENGTH - DOCUMENT_PAIR_BUFFER_TOKENS] == [
        [(topic.narrative, view.scoring_text)]
    ]
    assert model_calls[WINDOW_MAX_LENGTH] == [[(topic.narrative, view.scoring_text)]]
    assert len(result) == 1
    assert result[0].text == source
    assert result[0].text != view.scoring_text
    passage = result[0].winning_passages[0]
    assert (passage.start_char, passage.end_char) == (
        source.index("C"),
        source.index("Δ") + len("Δelta"),
    )
    assert source[passage.start_char : passage.end_char] == "Café\tBeta\r\nGamma\u2003Δelta"

    document_row = _rows(scorer.document_score_path)[0]
    window_row = _rows(scorer.window_score_path)[0]
    assert document_row["text_sha256"] == view.scoring_text_sha256
    assert window_row["document_text_sha256"] == view.scoring_text_sha256
    assert (window_row["start_char"], window_row["end_char"]) == (
        0,
        len(view.scoring_text),
    )
    assert window_row["text_sha256"] == hashlib.sha256(
        view.scoring_text.encode("utf-8")
    ).hexdigest()


def test_same_scoring_text_shares_scores_but_projects_each_exact_source(tmp_path) -> None:
    topic = Topic("31", "", "Duplicate scoring-view test")
    lane = _lane(topic)
    sources = ("alpha\tbeta", "  alpha\r\n beta  ")
    assert ScoringView(sources[0]).scoring_text == ScoringView(sources[1]).scoring_text
    calls: list[tuple[int, int]] = []

    class Model(_CapturingModel):
        def predict(self, pairs, **kwargs):
            calls.append((len(pairs), 1))
            return super().predict(pairs, **kwargs)

    def loader(_model, *, revision, max_length, device):
        assert revision and device == "cpu"
        return Model([], 1.0)

    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "first-ledger",
        score_cache_root=tmp_path / "cache",
        device="cpu",
        model_loader=loader,
    )
    result = scorer.score_lane(
        topic,
        lane,
        [
            _candidate(topic, sources[0], docid="doc-a", rank=1),
            _candidate(topic, sources[1], docid="doc-b", rank=2),
        ],
    )

    assert calls == [(1, 1), (1, 1)]
    by_doc = {row.docid: row for row in result}
    assert by_doc["doc-a"].text == sources[0]
    assert by_doc["doc-b"].text == sources[1]
    for docid, source in zip(("doc-a", "doc-b"), sources, strict=True):
        passage = by_doc[docid].winning_passages[0]
        assert ScoringView(
            source[passage.start_char : passage.end_char]
        ).scoring_text == "alpha beta"

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("warm normalized score cache must avoid model loading")

    warm = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "warm-ledger",
        score_cache_root=tmp_path / "cache",
        model_loader=forbidden_loader,
    )
    assert [row.text for row in warm.score_lane(
        topic,
        lane,
        [
            _candidate(topic, sources[0], docid="doc-a", rank=1),
            _candidate(topic, sources[1], docid="doc-b", rank=2),
        ],
    )] == [row.text for row in result]


def test_coverage_scorer_rejects_empty_scoring_view_before_model(tmp_path) -> None:
    topic = Topic("37", "", "Empty scoring-view test")

    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "ledger",
        score_cache_root=tmp_path / "cache",
        model_loader=lambda *_args, **_kwargs: pytest.fail("model must remain lazy"),
    )

    with pytest.raises(ValueError, match="scoring view"):
        scorer.score_lane(topic, _lane(topic), [_candidate(topic, " \t\r\n")])

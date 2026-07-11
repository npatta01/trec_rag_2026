from __future__ import annotations

import math

import pytest

from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.ranking import reciprocal_rank_fusion


def _candidate(
    topic_id: str,
    variant_name: str,
    retriever_name: str,
    docid: str,
    rank: int,
    score: float,
    text: str = "",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id=topic_id,
        variant_name=variant_name,
        retriever_name=retriever_name,
        query_text=f"{variant_name} query",
        docid=docid,
        rank=rank,
        score=score,
        text=text,
    )


def test_rrf_fuses_per_topic_dedupes_streams_and_preserves_full_provenance():
    candidates = [
        _candidate("31", "original", "bm25", "doc-a", 1, 9.0),
        _candidate("31", "original", "bm25", "doc-a", 3, 100.0, "A duplicate"),
        _candidate("31", "original", "bm25", "doc-b", 3, 7.0, "B original"),
        _candidate("31", "facet", "bm25", "doc-a", 2, 8.0, "A facet"),
        _candidate("31", "facet", "bm25", "doc-b", 1, 6.0, "B facet"),
        _candidate("32", "original", "bm25", "doc-z", 1, 5.0, "Z"),
        _candidate("32", "facet", "bm25", "doc-y", 1, 4.0, "Y"),
    ]

    ranked = reciprocal_rank_fusion(candidates)

    assert ranked == reciprocal_rank_fusion(list(reversed(candidates)))
    assert [(row.topic_id, row.docid, row.rank) for row in ranked] == [
        ("31", "doc-a", 1),
        ("31", "doc-b", 2),
        ("32", "doc-y", 1),
        ("32", "doc-z", 2),
    ]

    doc_a = ranked[0]
    assert doc_a.score == pytest.approx((1 / 61) + (1 / 62))
    assert doc_a.text == "A facet"
    assert [(item["variant_name"], item["source_rank"]) for item in doc_a.provenance] == [
        ("facet", 2),
        ("original", 1),
    ]
    assert {item["source_score"] for item in doc_a.provenance} == {8.0, 9.0}
    for item in doc_a.provenance:
        assert item["ranker"] == "reciprocal_rank_fusion"
        assert item["retriever_name"] == "bm25"
        assert item["query_text"].endswith(" query")
        assert item["rrf_k"] == 60
        assert item["rrf_weight"] == 1.0
        assert item["rrf_contribution"] == pytest.approx(1 / (60 + item["source_rank"]))


def test_rrf_applies_explicit_weights_and_limit_per_topic():
    candidates = [
        _candidate("31", "original", "bm25", "doc-original", 1, 9.0, "Original"),
        _candidate("31", "facet", "bm25", "doc-facet", 1, 1.0, "Facet"),
        _candidate("32", "original", "bm25", "doc-original-2", 1, 8.0, "Original 2"),
        _candidate("32", "facet", "bm25", "doc-facet-2", 1, 2.0, "Facet 2"),
    ]

    ranked = reciprocal_rank_fusion(
        candidates,
        stream_weights={
            ("original", "bm25"): 1.0,
            ("facet", "bm25"): 2.0,
            ("unused", "bm25"): 3.0,
        },
        limit=1,
    )

    assert [(row.topic_id, row.docid, row.rank) for row in ranked] == [
        ("31", "doc-facet", 1),
        ("32", "doc-facet-2", 1),
    ]
    assert all(row.score == pytest.approx(2 / 61) for row in ranked)
    assert all(row.provenance[0]["rrf_weight"] == 2.0 for row in ranked)


def test_rrf_dedupe_uses_best_rank_then_best_score_and_recovers_nonempty_text():
    candidates = [
        _candidate("31", "original", "bm25", "doc-a", 1, 2.0),
        _candidate("31", "original", "bm25", "doc-a", 1, 7.0),
        _candidate("31", "original", "bm25", "doc-a", 2, 100.0, "Recovered text"),
    ]

    [ranked] = reciprocal_rank_fusion(candidates)

    assert ranked.score == pytest.approx(1 / 61)
    assert ranked.text == "Recovered text"
    assert len(ranked.provenance) == 1
    assert ranked.provenance[0]["source_rank"] == 1
    assert ranked.provenance[0]["source_score"] == 7.0


@pytest.mark.parametrize("k", [0, -1, math.inf, math.nan])
def test_rrf_rejects_invalid_k(k: float):
    with pytest.raises(ValueError, match="rrf k.*at least 1"):
        reciprocal_rank_fusion([], k=k)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_limit(limit: object):
    with pytest.raises(ValueError, match="rrf limit.*at least 1"):
        reciprocal_rank_fusion([], limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("rank", [0, -1, math.inf, math.nan])
def test_rrf_rejects_invalid_source_rank(rank: float):
    candidate = _candidate("31", "original", "bm25", "doc-a", rank, 1.0)

    with pytest.raises(ValueError, match="candidate ranks.*at least 1"):
        reciprocal_rank_fusion([candidate])


@pytest.mark.parametrize("weight", [0, -1, math.inf, math.nan, "invalid"])
def test_rrf_rejects_nonpositive_or_nonfinite_weights(weight: object):
    candidate = _candidate("31", "original", "bm25", "doc-a", 1, 1.0)

    with pytest.raises(ValueError, match="stream weights.*finite and positive"):
        reciprocal_rank_fusion(
            [candidate],
            stream_weights={("original", "bm25"): weight},  # type: ignore[dict-item]
        )


def test_rrf_requires_every_observed_stream_when_weights_are_explicit():
    candidates = [
        _candidate("31", "original", "bm25", "doc-a", 1, 2.0),
        _candidate("31", "facet", "bm25", "doc-b", 1, 1.0),
    ]

    with pytest.raises(ValueError, match="missing rrf stream weights.*facet.*bm25"):
        reciprocal_rank_fusion(
            candidates,
            stream_weights={("original", "bm25"): 1.0},
        )


def test_rrf_validates_even_unused_configured_weights():
    candidate = _candidate("31", "original", "bm25", "doc-a", 1, 1.0)

    with pytest.raises(ValueError, match="stream weights.*finite and positive"):
        reciprocal_rank_fusion(
            [candidate],
            stream_weights={
                ("original", "bm25"): 1.0,
                ("unused", "bm25"): 0.0,
            },
        )

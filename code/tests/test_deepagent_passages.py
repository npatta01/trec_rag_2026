import pytest

from trec_rag.chunking import TextChunk
from trec_rag.deepagent_passages import (
    PassageSelectionConfig,
    PassageSelectionError,
    ScoredPassage,
    group_by_document,
    select_diverse_passages,
    selection_summary,
)


def passage(document_id, chunk_index, score, document_rank=1):
    chunk_id = f"{document_id}::{chunk_index}"
    return ScoredPassage(
        document_id=document_id,
        document_rank=document_rank,
        chunk=TextChunk(
            document_id=document_id,
            chunk_id=chunk_id,
            text=f"text for {chunk_id}",
            start_char=chunk_index * 100,
            end_char=chunk_index * 100 + 100,
        ),
        relevance_score=score,
    )


def test_a_single_document_cannot_take_the_whole_result_set():
    """The defect this module exists to fix: one document, every passage."""
    passages = [passage("D1", i, 10.0 - i * 0.01) for i in range(50)]
    passages += [passage("D2", 0, 1.0), passage("D3", 0, 0.9)]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=10, per_document_cap=3, min_distinct_documents=2),
    )

    per_document = {}
    for row in selected:
        per_document[row.document_id] = per_document.get(row.document_id, 0) + 1
    assert per_document["D1"] == 3
    assert set(per_document) == {"D1", "D2", "D3"}


def test_breadth_phase_admits_a_weaker_document_over_a_stronger_second_passage():
    """Breadth runs before depth, which a plain global top-K would not do."""
    passages = [
        passage("D1", 0, 10.0),
        passage("D1", 1, 9.0),
        passage("D2", 0, 1.0),
    ]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=2, per_document_cap=2, min_distinct_documents=2),
    )

    assert [row.document_id for row in selected] == ["D1", "D2"]
    assert [row.chunk_id for row in selected] == ["D1::0", "D2::0"]


def test_global_top_k_alone_would_have_collapsed_onto_one_document():
    """Guards the specific alternative the design rules out."""
    passages = [passage("D1", i, 10.0 - i) for i in range(5)]
    passages += [passage("D2", 0, 1.0)]

    plain_top_k = sorted(passages, key=lambda p: -p.relevance_score)[:3]
    assert {row.document_id for row in plain_top_k} == {"D1"}

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=3, per_document_cap=3, min_distinct_documents=2),
    )
    assert {row.document_id for row in selected} == {"D1", "D2"}


def test_breadth_target_degrades_gracefully_when_documents_run_out():
    passages = [passage("D1", 0, 5.0), passage("D1", 1, 4.0)]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=4, per_document_cap=2, min_distinct_documents=4),
    )

    assert len(selected) == 2
    assert {row.document_id for row in selected} == {"D1"}


def test_selection_never_exceeds_top_k():
    passages = [passage(f"D{d}", i, 10.0 - d - i * 0.1) for d in range(10) for i in range(5)]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=7, per_document_cap=2, min_distinct_documents=3),
    )

    assert len(selected) == 7


def test_selection_is_deterministic_regardless_of_input_order():
    passages = [passage(f"D{d}", i, 5.0 - i * 0.1) for d in range(4) for i in range(4)]
    config = PassageSelectionConfig(top_k=6, per_document_cap=2, min_distinct_documents=3)

    forward = select_diverse_passages(passages, config)
    backward = select_diverse_passages(list(reversed(passages)), config)

    assert [row.chunk_id for row in forward] == [row.chunk_id for row in backward]


def test_equal_scores_break_ties_by_retrieval_rank_then_chunk_id():
    passages = [
        passage("D2", 0, 1.0, document_rank=2),
        passage("D1", 0, 1.0, document_rank=1),
    ]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=2, per_document_cap=1, min_distinct_documents=2),
    )

    assert [row.document_id for row in selected] == ["D1", "D2"]


def test_empty_input_returns_empty_selection():
    assert select_diverse_passages([]) == ()


def test_no_passage_is_returned_twice():
    passages = [passage("D1", 0, 5.0), passage("D2", 0, 4.0), passage("D1", 1, 3.0)]

    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=3, per_document_cap=2, min_distinct_documents=2),
    )

    chunk_ids = [row.chunk_id for row in selected]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool_hits": 0},
        {"rerank_depth": -1},
        {"top_k": 0},
        {"per_document_cap": 0},
        {"min_distinct_documents": 0},
        {"rerank_depth": 2000, "pool_hits": 1000},
        {"per_document_cap": 20, "top_k": 10},
        {"min_distinct_documents": 20, "top_k": 10},
        {"top_k": True},
    ],
)
def test_unusable_configurations_are_rejected(kwargs):
    with pytest.raises(PassageSelectionError):
        PassageSelectionConfig(**kwargs)


def test_defaults_retrieve_the_full_pool():
    """The spec's floor: retrieving shallower than this is the measured defect."""
    config = PassageSelectionConfig()
    assert config.pool_hits == 1000
    assert config.min_distinct_documents > 1
    assert config.per_document_cap < config.top_k


def test_summary_reports_what_was_dropped_not_only_what_was_kept():
    passages = [passage("D1", 0, 5.0), passage("D2", 0, 4.0)]
    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=2, per_document_cap=1, min_distinct_documents=2),
    )

    summary = selection_summary(selected, scored_total=500, documents_scored=100)

    assert summary["returned_passages"] == 2
    assert summary["distinct_documents"] == 2
    assert summary["chunks_scored"] == 500
    assert summary["documents_scored"] == 100
    assert summary["chunks_not_returned"] == 498
    assert summary["passages_per_document"] == {"D1": 1, "D2": 1}


def test_grouping_preserves_document_provenance_for_the_ledger():
    passages = [
        passage("D1", 0, 5.0),
        passage("D2", 0, 4.0),
        passage("D1", 1, 3.0),
    ]
    selected = select_diverse_passages(
        passages,
        PassageSelectionConfig(top_k=3, per_document_cap=2, min_distinct_documents=2),
    )

    grouped = group_by_document(selected)

    assert [document_id for document_id, _ in grouped] == ["D1", "D2"]
    assert sum(len(rows) for _, rows in grouped) == len(selected)
    for _, rows in grouped:
        assert [row.chunk_id for row in rows] == sorted(
            (row.chunk_id for row in rows),
            key=lambda cid: [r.relevance_score for r in rows if r.chunk_id == cid][0],
            reverse=True,
        )


class FakeChunker:
    def split_text(self, text, *, document_id):
        parts = text.split("|")
        return [
            TextChunk(
                document_id=document_id,
                chunk_id=f"{document_id}::{i}",
                text=part,
                start_char=i * 10,
                end_char=i * 10 + len(part),
            )
            for i, part in enumerate(parts)
            if part
        ]


class FakeRanker:
    """Scores by chunk length so ordering is predictable but not input order."""

    def __init__(self):
        self.calls = []

    def rank(self, focus_query, chunks):
        self.calls.append((focus_query, tuple(c.chunk_id for c in chunks)))
        return tuple(
            ScoredChunkStub(chunk, float(len(chunk.text)))
            for chunk in chunks
        )


class ScoredChunkStub:
    def __init__(self, chunk, relevance_score):
        self.chunk = chunk
        self.relevance_score = relevance_score


def test_pool_scoring_scores_every_chunk_in_one_batched_call():
    """Cross-document comparability requires a single scale, so one call."""
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    docs = [
        PooledDocument("D1", 1, "aaa|bb"),
        PooledDocument("D2", 2, "cccc"),
    ]
    ranker = FakeRanker()

    result = score_document_pool(
        docs, "focus", chunker=FakeChunker(), ranker=ranker,
        config=PassageSelectionConfig(rerank_depth=10, pool_hits=10),
    )

    assert len(ranker.calls) == 1
    assert result.chunks_scored == 3
    assert result.documents_scored == 2
    assert result.documents_skipped == 0
    assert {p.document_id for p in result.passages} == {"D1", "D2"}


def test_pool_scoring_reports_documents_beyond_rerank_depth_as_skipped():
    """A bounded scan must not read as an exhaustive one."""
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    docs = [PooledDocument(f"D{i}", i, "aa") for i in range(1, 11)]

    result = score_document_pool(
        docs, "focus", chunker=FakeChunker(), ranker=FakeRanker(),
        config=PassageSelectionConfig(rerank_depth=3, pool_hits=10),
    )

    assert result.documents_scored == 3
    assert result.documents_skipped == 7


def test_pool_scoring_uses_retrieval_rank_order_not_input_order():
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    docs = [PooledDocument("D9", 9, "aa"), PooledDocument("D1", 1, "bb")]

    result = score_document_pool(
        docs, "focus", chunker=FakeChunker(), ranker=FakeRanker(),
        config=PassageSelectionConfig(rerank_depth=1, pool_hits=10),
    )

    assert {p.document_id for p in result.passages} == {"D1"}


def test_pool_scoring_carries_document_rank_onto_each_passage():
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    docs = [PooledDocument("D5", 5, "aaa")]

    result = score_document_pool(
        docs, "focus", chunker=FakeChunker(), ranker=FakeRanker(),
        config=PassageSelectionConfig(rerank_depth=5, pool_hits=10),
    )

    assert [p.document_rank for p in result.passages] == [5]


def test_pool_scoring_skips_empty_documents_without_failing():
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    docs = [PooledDocument("D1", 1, ""), PooledDocument("D2", 2, "   "), PooledDocument("D3", 3, "aa")]

    result = score_document_pool(
        docs, "focus", chunker=FakeChunker(), ranker=FakeRanker(),
        config=PassageSelectionConfig(rerank_depth=10, pool_hits=10),
    )

    assert result.chunks_scored == 1
    assert {p.document_id for p in result.passages} == {"D3"}


def test_pool_scoring_with_no_usable_text_returns_empty_without_ranking():
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    ranker = FakeRanker()
    result = score_document_pool(
        [PooledDocument("D1", 1, "")], "focus", chunker=FakeChunker(), ranker=ranker,
        config=PassageSelectionConfig(rerank_depth=10, pool_hits=10),
    )

    assert result.passages == ()
    assert ranker.calls == []


def test_pool_scoring_rejects_a_blank_focus_query():
    from trec_rag.deepagent_passages import PooledDocument, score_document_pool

    with pytest.raises(PassageSelectionError):
        score_document_pool(
            [PooledDocument("D1", 1, "aa")], "  ",
            chunker=FakeChunker(), ranker=FakeRanker(),
        )

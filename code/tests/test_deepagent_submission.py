from types import SimpleNamespace

import pytest

from trec_rag.deepagent_evidence import EvidenceCoverageState
from trec_rag.deepagent_retrieval import (
    AgentCandidateProvenance,
    AgentRankedCandidate,
    AgentSearch,
)
from trec_rag.deepagent_snippets import RelevantSnippet, SnippetPage
from trec_rag.deepagent_submission import (
    AgenticDocumentRank,
    document_usefulness,
    rank_agentic_documents,
    rank_for_submission,
    selection_summary,
    submission_rows,
)
from trec_rag.pipeline_models import RankedCandidate, RetrievedCandidate


NARRATIVE = "Why do people migrate and what challenges do they face?"


def _state(documents=("doc-a", "doc-b", "doc-c")):
    state = EvidenceCoverageState(NARRATIVE)
    for index, document_id in enumerate(documents):
        state.record_snippet_page(
            SnippetPage(
                document_id=document_id,
                focus_query="migration drivers",
                snippets=(
                    RelevantSnippet(
                        f"{document_id}:000{index}",
                        0,
                        80,
                        "Conflict displaces people across borders every year.",
                        0.9,
                    ),
                ),
                next_cursor=None,
                page_index=0,
                residual_count=0,
                residual_top_score=None,
                returned_min_score=0.9,
                pages_estimated=1,
            )
        )
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why do people migrate?",
                },
                {
                    "need_id": "n2",
                    "narrative_span": "what challenges do they face",
                    "question": "What challenges do they face?",
                },
            ]
        }
    )
    return state


def _add(state, nugget_id, cite, need_ids=("n1",)):
    return state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": nugget_id,
                    "text": "Conflict displaces people across borders.",
                    "need_ids": list(need_ids),
                    "facet_ids": [],
                    "evidence": [{"cite": cite}],
                    "contradicts": [],
                }
            ]
        }
    )


def test_only_documents_that_supplied_evidence_are_submitted() -> None:
    """A document that was read and produced nothing is not evidence."""
    state = _state()
    _add(state, "g1", "S1")

    ranked = rank_for_submission(state.report())

    assert [row.document_id for row in ranked] == ["doc-a"]


def test_a_drafted_document_outranks_an_undrafted_one() -> None:
    """Being selected into a need's bounded draft set is the importance signal."""
    state = _state()
    _add(state, "drafted", "S1")
    _add(state, "vital", "S2")
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "answerable",
                    "remaining_gap": "",
                    "draft_answer": "People migrate because conflict displaces them.",
                    "draft_nugget_ids": ["drafted"],
                }
            ]
        }
    )

    ranked = rank_for_submission(state.report())

    assert [row.document_id for row in ranked][0] == "doc-a"


def test_a_superseded_claim_stops_vouching_for_its_document() -> None:
    state = _state()
    _add(state, "old", "S1")
    _add(state, "new", "S2")
    state.apply_delta(
        {"supersede_nuggets": [{"nugget_id": "old", "superseded_by": "new"}]}
    )

    ranked = rank_for_submission(state.report())

    assert [row.document_id for row in ranked] == ["doc-b"]


def test_ranking_is_deterministic_and_ties_break_by_document_id() -> None:
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    first = [row.document_id for row in rank_for_submission(state.report())]
    second = [row.document_id for row in rank_for_submission(state.report())]

    assert first == second == sorted(first)


def test_an_empty_ledger_submits_nothing_rather_than_padding() -> None:
    """The task says do not pad to a conventional depth."""
    assert rank_for_submission(_state().report()) == ()


def test_max_documents_is_a_valve_not_a_default() -> None:
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    assert len(rank_for_submission(state.report())) == 3
    assert len(rank_for_submission(state.report(), max_documents=2)) == 2
    with pytest.raises(ValueError):
        rank_for_submission(state.report(), max_documents=0)


def test_run_rows_are_ranked_from_one_with_non_increasing_scores() -> None:
    state = _state()
    _add(state, "vital1", "S1")
    _add(state, "okay1", "S2")

    rows = submission_rows(state.report(), topic_id="rag2026-1", run_id="deepagent")

    assert [row["rank"] for row in rows] == [1, 2]
    assert [row["q0"] for row in rows] == ["Q0", "Q0"]
    assert all(row["topic_id"] == "rag2026-1" for row in rows)
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_usefulness_records_need_breadth_and_support() -> None:
    state = _state()
    _add(state, "g1", "S1", need_ids=("n1", "n2"))

    rows = {row.document_id: row for row in document_usefulness(state.report())}

    assert rows["doc-a"].need_ids == ("n1", "n2")
    assert rows["doc-a"].nugget_count == 1
    assert rows["doc-a"].mean_support_ratio > 0.0


def test_summary_reports_what_the_submission_leans_on() -> None:
    """Vital is now the drafted selection, so the summary counts that."""
    state = _state()
    _add(state, "v", "S1")
    _add(state, "o", "S2")
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "more needed",
                    "draft_nugget_ids": ["v"],
                }
            ]
        }
    )

    summary = selection_summary(rank_for_submission(state.report()))

    assert summary["submitted_documents"] == 2
    assert summary["documents_behind_a_vital_nugget"] == 1
    assert summary["documents_behind_a_drafted_nugget"] == 1
    assert summary["single_nugget_documents"] == 2


# --- variable-depth agentic document order -----------------------------------
#
# The agentic run submits document order rather than a tuned score: documents
# that supplied live evidence, ordered by the retrieval evidence that produced
# them, with a depth that falls out of the ledger.


def _retrieved(docid: str, rank: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id="rag2026-1",
        variant_name="original",
        retriever_name="climbmix",
        query_text="migration drivers",
        docid=docid,
        rank=rank,
        score=1.0 / rank,
        text="Conflict displaces people across borders every year.",
    )


def _search(*docids_and_ranks: tuple[str, int], query: str = "migration drivers"):
    return AgentSearch(
        query=query,
        kind="original",
        candidates=tuple(_retrieved(docid, rank) for docid, rank in docids_and_ranks),
        cache_status="hit",
    )


def _fused(*docids: str) -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(
            topic_id="rag2026-1",
            docid=docid,
            rank=rank,
            score=1.0 / rank,
            text="Conflict displaces people across borders every year.",
            provenance=[],
        )
        for rank, docid in enumerate(docids, start=1)
    )


def test_agentic_document_order_submits_only_live_supporting_documents() -> None:
    """Read-but-uncited and superseded-only documents are not evidence."""
    state = _state()
    _add(state, "live", "S1")
    _add(state, "old", "S2")
    _add(state, "new", "S3")
    state.apply_delta(
        {"supersede_nuggets": [{"nugget_id": "old", "superseded_by": "new"}]}
    )

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-a", "doc-b", "doc-c"),
        searches=(_search(("doc-a", 1), ("doc-b", 2), ("doc-c", 3)),),
    )

    # doc-b is retrieved and fused but only backed a superseded claim.
    assert [row.document_id for row in ranked] == ["doc-a", "doc-c"]


def test_agentic_document_order_keeps_final_fused_order() -> None:
    """Fused/RRF order is the ranking signal for documents that survived it."""
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    ranked = rank_agentic_documents(
        state.report(),
        # Neither document-id order nor search order.
        fused_candidates=_fused("doc-c", "doc-a", "doc-b"),
        searches=(_search(("doc-a", 1), ("doc-b", 2), ("doc-c", 3)),),
    )

    assert [row.document_id for row in ranked] == ["doc-c", "doc-a", "doc-b"]
    assert {row.order_source for row in ranked} == {"fused"}


def test_agentic_document_order_falls_back_to_earliest_search_position() -> None:
    """A document fusion dropped keeps the earliest place any search gave it."""
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=(),
        searches=(
            _search(("doc-c", 1), ("doc-b", 2)),
            # doc-b reappears later and higher; the earliest sighting still wins.
            _search(("doc-b", 1), ("doc-a", 2), query="displacement"),
        ),
    )

    assert [row.document_id for row in ranked] == ["doc-c", "doc-b", "doc-a"]
    assert {row.order_source for row in ranked} == {"search"}


def test_agentic_document_order_uses_passage_rank_when_search_retains_passages() -> None:
    """Production searches retain reranked passages; that is the fallback signal."""
    state = _state(documents=("doc-a", "doc-b"))
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    search = SimpleNamespace(
        # Remote document order disagrees with the local passage reranker.
        candidates=(
            SimpleNamespace(docid="doc-a", rank=1),
            SimpleNamespace(docid="doc-b", rank=2),
        ),
        passages=(
            SimpleNamespace(docid="doc-b", rank=1),
            SimpleNamespace(docid="doc-a", rank=8),
        ),
    )

    ranked = rank_agentic_documents(
        state.report(), fused_candidates=(), searches=(search,)
    )

    assert [row.document_id for row in ranked] == ["doc-b", "doc-a"]


def test_agentic_document_order_breaks_search_ties_by_document_id() -> None:
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=(),
        searches=(_search(("doc-b", 1), ("doc-a", 1)),),
    )

    assert [row.document_id for row in ranked] == ["doc-a", "doc-b"]


def test_agentic_document_order_puts_fused_documents_before_search_only_ones() -> None:
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-c"),
        searches=(_search(("doc-a", 1), ("doc-b", 2), ("doc-c", 3)),),
    )

    assert [row.document_id for row in ranked] == ["doc-c", "doc-a", "doc-b"]
    assert [row.order_source for row in ranked] == ["fused", "search", "search"]


def test_agentic_document_order_places_unseen_documents_last_by_id() -> None:
    """A cited document neither fused nor in any search still has to be ranked."""
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")
    _add(state, "c", "S3")

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-c"),
        searches=(),
    )

    assert [row.document_id for row in ranked] == ["doc-c", "doc-a", "doc-b"]
    assert [row.order_source for row in ranked] == ["fused", "unplaced", "unplaced"]


def test_agentic_document_order_collapses_duplicates() -> None:
    state = _state()
    _add(state, "a", "S1")
    _add(state, "a2", "S1")
    _add(state, "b", "S2")

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-a", "doc-b", "doc-a"),
        searches=(
            _search(("doc-a", 1), ("doc-b", 2)),
            _search(("doc-a", 1), query="displacement"),
        ),
    )

    assert [row.document_id for row in ranked] == ["doc-a", "doc-b"]


def test_agentic_document_order_ranks_are_contiguous_with_integer_scores() -> None:
    """Ranks start at 1 with no gaps; scores are ``document_count - rank + 1``."""
    state = _state(documents=("doc-a", "doc-b", "doc-c", "doc-d"))
    for index, handle in enumerate(("S1", "S2", "S3", "S4")):
        _add(state, f"g{index}", handle)

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-d", "doc-b"),
        searches=(_search(("doc-c", 1), ("doc-a", 2)),),
    )

    assert [row.rank for row in ranked] == [1, 2, 3, 4]
    scores = [row.score for row in ranked]
    assert scores == [4, 3, 2, 1]
    assert all(
        isinstance(row.score, int) and not isinstance(row.score, bool)
        for row in ranked
    )
    assert all(row.score > 0 for row in ranked)
    assert all(earlier > later for earlier, later in zip(scores, scores[1:]))
    assert all(row.score == len(ranked) - row.rank + 1 for row in ranked)
    assert ranked[0] == AgenticDocumentRank(
        document_id="doc-d", rank=1, score=4, order_source="fused"
    )


def test_agentic_document_order_is_empty_without_grounded_evidence() -> None:
    """Fused candidates never inject a document the ledger did not ground."""
    state = _state()

    ranked = rank_agentic_documents(
        state.report(),
        fused_candidates=_fused("doc-a", "doc-b", "doc-c"),
        searches=(_search(("doc-a", 1), ("doc-b", 2), ("doc-c", 3)),),
    )

    assert ranked == ()


def test_agentic_document_order_accepts_agent_ranked_candidates() -> None:
    """The projector takes the runner's own immutable candidate objects."""
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")

    fused = tuple(
        AgentRankedCandidate(
            topic_id="rag2026-1",
            docid=docid,
            rank=rank,
            score=1.0 / rank,
            text="Conflict displaces people across borders every year.",
            provenance=(
                AgentCandidateProvenance(
                    query="migration drivers",
                    query_kind="original",
                    variant_name="original",
                    retriever_name="climbmix",
                    source_rank=rank,
                    source_score=1.0 / rank,
                    cache_status="hit",
                ),
            ),
        )
        for rank, docid in enumerate(("doc-b", "doc-a"), start=1)
    )

    ranked = rank_agentic_documents(state.report(), fused_candidates=fused)

    assert [(row.document_id, row.rank, row.score) for row in ranked] == [
        ("doc-b", 1, 2),
        ("doc-a", 2, 1),
    ]


def test_agentic_document_order_leaves_legacy_ranking_untouched() -> None:
    """The usefulness ranking keeps its own float-score semantics."""
    state = _state()
    _add(state, "a", "S1")
    _add(state, "b", "S2")

    legacy = rank_for_submission(state.report())

    assert [row.document_id for row in legacy] == ["doc-a", "doc-b"]
    assert all(isinstance(row.score, float) for row in legacy)

import pytest

from trec_rag.deepagent_evidence import EvidenceCoverageState
from trec_rag.deepagent_snippets import RelevantSnippet, SnippetPage
from trec_rag.deepagent_submission import (
    document_usefulness,
    rank_for_submission,
    selection_summary,
    submission_rows,
)


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

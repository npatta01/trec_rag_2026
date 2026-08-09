from __future__ import annotations

import json

import pytest
from typing_extensions import TypedDict
from trec_rag.deepagent_budget import ResearchTaskContext
from trec_rag.deepagent_evidence import (
    MAX_DRAFT_NUGGETS_PER_NEED,
    MAX_FRONTIER_CHARACTERS,
    SATURATION_ZERO_YIELD_PAGES,
    DocumentObservation,
    EvidenceCoverageState,
    RetrievalStateDelta,
)
from trec_rag.deepagent_snippets import RelevantSnippet, SnippetPage


def test_model_facing_delta_uses_backported_typed_dict() -> None:
    """Pydantic 2 requires typing_extensions.TypedDict on Python 3.11."""
    assert RetrievalStateDelta.__orig_bases__[0] is TypedDict


def _state_with_snippet() -> EvidenceCoverageState:
    state = EvidenceCoverageState("Why do people migrate and what challenges do they face?")
    state.record_search(
        query="Why do people migrate and what challenges do they face?",
        kind="original",
        documents=(DocumentObservation("doc-a", 1),),
    )
    snippet_text = "Conflict and persecution force people to flee."
    page = SnippetPage(
        document_id="doc-a",
        focus_query="migration drivers",
        snippets=(
            RelevantSnippet(
                chunk_id="doc-a:0001",
                start_char=0,
                end_char=len(snippet_text),
                text=snippet_text,
                relevance_score=0.9,
            ),
        ),
        next_cursor=None,
        page_index=0,
        residual_count=0,
        residual_top_score=None,
        returned_min_score=0.9,
        pages_estimated=1,
    )
    state.record_snippet_page(page)
    return state


def _state_with_sentences() -> EvidenceCoverageState:
    """One snippet whose text splits into exactly three sentences."""
    state = EvidenceCoverageState("Why do people migrate?")
    state.record_search(
        query="Why do people migrate?",
        kind="original",
        documents=(DocumentObservation("doc-a", 1),),
    )
    text = (
        "Conflict displaces people. Persecution forces flight. "
        "Drought removes livelihoods."
    )
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="migration drivers",
            snippets=(RelevantSnippet("doc-a:0001", 0, len(text), text, 0.9),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.9,
            pages_estimated=1,
        )
    )
    return state


def _add_need_and_facet(state: EvidenceCoverageState) -> None:
    result = state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why do people migrate?",
                }
            ],
            "add_facets": [
                {
                    "facet_id": "f1",
                    "need_ids": ["n1"],
                    "dimension": "driver",
                    "value": "conflict",
                    "origin": "narrative",
                    "origin_snippet_id": None,
                }
            ],
        }
    )
    assert result.accepted_ids == ("n1", "f1")


def _grounded_nugget_delta(nugget_id: str = "g1") -> dict[str, object]:
    return {
        "add_nuggets": [
            {
                "nugget_id": nugget_id,
                "text": "Conflict and persecution force displacement.",
                "need_ids": ["n1"],
                "facet_ids": ["f1"],
                "evidence": [{"cite": "S1"}],
                "contradicts": [],
            }
        ]
    }


def test_record_retrieval_action_records_actual_arguments_atomically() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "N1",
                    "narrative_span": "Why do people migrate?",
                    "question": "What evidence explains migration?",
                }
            ]
        }
    )
    context = ResearchTaskContext("R1-N1", 1, "focused", ("N1",))

    error = state.record_retrieval_action(
        action="search",
        target="actual refined query",
        focus_query=None,
        motivating_ids=["N1"],
        rationale="N1 has no grounded driver evidence",
        context=context,
    )

    assert error is None
    action = state.report().actions[-1]
    assert action.target == "actual refined query"
    assert action.motivating_ids == ("N1",)
    assert action.state == "consumed"
    assert action.research_task_id == "R1-N1"
    assert action.round_index == 1
    assert action.depth == "focused"


def test_record_retrieval_action_rejects_unknown_motivation_without_an_action() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))

    error = state.record_retrieval_action(
        action="search",
        target="query",
        focus_query=None,
        motivating_ids=["UNKNOWN"],
        rationale="find evidence",
        context=context,
    )

    assert error == "UNKNOWN_MOTIVATION"
    assert state.report().actions == ()


def test_needs_must_be_anchored_in_the_untouched_narrative() -> None:
    state = _state_with_snippet()

    result = state.apply_delta(
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
                    "question": "What challenges do migrants face?",
                },
            ]
        }
    )

    assert result.accepted_ids == ("n1", "n2")
    assert result.rejected == ()
    assert tuple(need.need_id for need in state.report().needs) == ("n1", "n2")


def test_unknown_only_delta_is_rejected() -> None:
    state = _state_with_snippet()

    result = state.apply_delta({"make_up_a_section": []})

    assert result.accepted_ids == ()
    assert [(item.section, item.index, item.code) for item in result.rejected] == [
        ("make_up_a_section", 0, "UNKNOWN_SECTION")
    ]


def test_unknown_handle_rejects_only_its_nugget() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    delta = _grounded_nugget_delta()
    delta["add_nuggets"] = [
        *delta["add_nuggets"],
        {
            "nugget_id": "g2",
            "text": "A fabricated claim.",
            "need_ids": ["n1"],
            "facet_ids": ["f1"],
            "evidence": [{"cite": "S9"}],
            "contradicts": [],
        },
    ]

    result = state.apply_delta(delta)

    assert result.accepted_ids == ("g1",)
    assert result.rejected[0].code == "UNKNOWN_CITATION"
    assert tuple(nugget.nugget_id for nugget in state.report().nuggets) == ("g1",)


def test_frontier_counts_grounded_nuggets_per_need() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)

    empty = json.loads(state.view("frontier"))["needs"]
    assert [item["grounded_nugget_count"] for item in empty] == [0]

    state.apply_delta(_grounded_nugget_delta())

    covered = json.loads(state.view("frontier"))["needs"]
    assert [item["grounded_nugget_count"] for item in covered] == [1]


def test_pending_closeout_needs_only_include_live_evidence_without_drafts() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())

    assert state.pending_closeout_need_ids() == ("n1",)

    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "More detail would help.",
                    "draft_nugget_ids": ["g1"],
                }
            ]
        }
    )

    assert state.pending_closeout_need_ids() == ()


def test_pending_closeout_ignores_superseded_only_evidence() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n2",
                    "narrative_span": "what challenges do they face",
                    "question": "What challenges do migrants face?",
                }
            ]
        }
    )
    state.apply_delta(_grounded_nugget_delta("g1"))
    state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "g2",
                    "text": "Challenges follow displacement.",
                    "need_ids": ["n2"],
                    "facet_ids": [],
                    "evidence": [{"cite": "S1"}],
                }
            ]
        }
    )
    state.apply_delta(
        {"supersede_nuggets": [{"nugget_id": "g1", "superseded_by": "g2"}]}
    )

    assert state.pending_closeout_need_ids() == ("n2",)


def test_pending_closeout_reopens_when_every_selected_draft_is_superseded() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta("g1"))
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "More detail would help.",
                    "draft_nugget_ids": ["g1"],
                }
            ],
            "add_nuggets": [
                {
                    "nugget_id": "g2",
                    "text": "A replacement claim.",
                    "need_ids": ["n1"],
                    "facet_ids": [],
                    "evidence": [{"cite": "S1"}],
                }
            ],
        }
    )
    state.apply_delta(
        {"supersede_nuggets": [{"nugget_id": "g1", "superseded_by": "g2"}]}
    )

    assert state.pending_closeout_need_ids() == ("n1",)


def test_pending_closeout_reopens_when_any_selected_draft_is_superseded() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta("g1"))
    state.apply_delta(_grounded_nugget_delta("g2"))
    state.apply_delta(_grounded_nugget_delta("g3"))
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "More detail would help.",
                    "draft_nugget_ids": ["g1", "g2"],
                }
            ]
        }
    )
    state.apply_delta(
        {"supersede_nuggets": [{"nugget_id": "g1", "superseded_by": "g3"}]}
    )

    assert state.pending_closeout_need_ids() == ("n1",)

    recovery = state.recover_grounded_drafts()

    assert recovery.recovered_need_ids == ("n1",)
    assert state.report().needs[0].draft_nugget_ids == ("g2", "g3")


def test_grounded_recovery_selects_live_nuggets_in_need_order_and_caps() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    nugget_ids = [
        f"g{index}" for index in range(MAX_DRAFT_NUGGETS_PER_NEED + 2)
    ]
    for nugget_id in nugget_ids:
        result = state.apply_delta(_grounded_nugget_delta(nugget_id))
        assert result.accepted_ids == (nugget_id,)
    state.apply_delta(
        {
            "supersede_nuggets": [
                {"nugget_id": "g1", "superseded_by": nugget_ids[-1]}
            ]
        }
    )

    recovery = state.recover_grounded_drafts()

    expected = ("g0", "g2", "g3", "g4", "g5")
    assert recovery.recovered_need_ids == ("n1",)
    assert recovery.selected_nugget_ids == expected
    assert "g1" not in recovery.live_grounded_nugget_ids
    need = state.report().needs[0]
    assert need.status == "partial"
    assert need.draft_nugget_ids == expected
    assert state.pending_closeout_need_ids() == ()


def test_grounded_recovery_retains_valid_selection_and_uses_only_owned_nuggets() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n2",
                    "narrative_span": "what challenges do they face",
                    "question": "What challenges do migrants face?",
                }
            ]
        }
    )
    state.apply_delta(_grounded_nugget_delta("owned-by-n1"))
    state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "owned-by-n2",
                    "text": "Displacement creates practical challenges.",
                    "need_ids": ["n2"],
                    "facet_ids": [],
                    "evidence": [{"cite": "S1"}],
                    "contradicts": [],
                }
            ],
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "More detail would help.",
                    "draft_nugget_ids": ["owned-by-n1"],
                }
            ],
        }
    )

    recovery = state.recover_grounded_drafts()

    needs = {need.need_id: need for need in state.report().needs}
    assert recovery.recovered_need_ids == ("n2",)
    assert recovery.selected_nugget_ids == ("owned-by-n2",)
    assert needs["n1"].draft_nugget_ids == ("owned-by-n1",)
    assert needs["n2"].draft_nugget_ids == ("owned-by-n2",)


def test_grounded_recovery_reports_zero_without_mutating_empty_state() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why do people migrate?",
                }
            ]
        }
    )
    before = state.report()

    recovery = state.recover_grounded_drafts()

    after = state.report()
    assert recovery.live_grounded_nugget_ids == ()
    assert recovery.recovered_need_ids == ()
    assert recovery.selected_nugget_ids == ()
    assert after.state_version == before.state_version
    assert after.state_hash == before.state_hash
    assert after.needs[0].draft_nugget_ids == ()


def test_atomic_completion_rejects_live_evidence_without_draft_selection() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())

    result = state.complete_retrieval()

    assert result["ok"] is False
    assert result["code"] == "INCOMPLETE_CLOSEOUT"
    assert result["need_ids"] == ["n1"]
    assert state.report().terminal_reason is None


def test_atomic_completion_reports_open_need_ids() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why do people migrate?",
                }
            ]
        }
    )

    result = state.complete_retrieval()

    assert result["ok"] is False
    assert result["code"] == "COMPLETION_OPEN_NEEDS"
    assert result["need_ids"] == ["n1"]


def test_every_rejected_completion_includes_deterministic_need_ids() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why do people migrate?",
                }
            ]
        }
    )
    assert state.choose_action(
        action="search",
        target="migration",
        focus_query=None,
        motivating_ids=["n1"],
        rationale="find evidence",
    )["ok"] is True

    result = state.complete_retrieval()

    assert result["ok"] is False
    assert result["code"] == "PENDING_ACTION_EXISTS"
    assert result["need_ids"] == ["n1"]


def test_frontier_count_ignores_a_status_the_agent_got_wrong() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "unaddressed",
                    "remaining_gap": "claimed empty despite grounded evidence",
                }
            ]
        }
    )

    need = json.loads(state.view("frontier"))["needs"][0]

    assert need["status"] == "unaddressed"
    assert need["grounded_nugget_count"] == 1, (
        "the count comes from the ledger, not from the status an agent set"
    )


def test_citations_resolve_to_stored_sentences() -> None:
    state = _state_with_sentences()
    _add_need_and_facet(state)

    def claim(nugget_id: str, cite: str) -> dict[str, object]:
        return {
            "add_nuggets": [
                {
                    "nugget_id": nugget_id,
                    "text": "A claim.",
                    "need_ids": ["n1"],
                    "facet_ids": ["f1"],
                    "evidence": [{"cite": cite}],
                    "contradicts": [],
                }
            ]
        }

    assert state.apply_delta(claim("a", "S1.2")).accepted_ids == ("a",)
    assert state.apply_delta(claim("b", "S1.2-3")).accepted_ids == ("b",)
    assert state.apply_delta(claim("c", "S1")).accepted_ids == ("c",)

    quotes = {item.nugget_id: item.evidence[0].quote for item in state.report().nuggets}
    assert quotes["a"] == "Persecution forces flight."
    assert quotes["b"] == "Persecution forces flight. Drought removes livelihoods."
    assert quotes["c"] == (
        "Conflict displaces people. Persecution forces flight. "
        "Drought removes livelihoods."
    )
    for reference in state.report().nuggets[0].evidence:
        assert reference.document_id == "doc-a"
        assert reference.snippet_id == "doc-a:0001"
        assert reference.page_index == 0


@pytest.mark.parametrize("cite", ["S1.0", "S1.4", "S1.3-2", "S1.2.3", "SS1", "1.2", ""])
def test_malformed_or_out_of_range_citations_are_rejected(cite: str) -> None:
    state = _state_with_sentences()
    _add_need_and_facet(state)

    result = state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "g9",
                    "text": "A claim.",
                    "need_ids": ["n1"],
                    "facet_ids": ["f1"],
                    "evidence": [{"cite": cite}],
                    "contradicts": [],
                }
            ]
        }
    )

    assert result.accepted_ids == ()
    assert result.rejected[0].code == "INVALID_CITATION"


def test_handles_are_invocation_scoped_and_stable_across_pages() -> None:
    state = _state_with_snippet()

    second = state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="migration drivers",
            snippets=(
                RelevantSnippet("doc-a:0001", 0, 45, "Conflict and persecution force people to flee.", 0.9),
                RelevantSnippet("doc-a:0002", 0, 16, "Another support.", 0.8),
            ),
            next_cursor=None,
            page_index=1,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.8,
            pages_estimated=2,
        )
    )

    assert [item.handle for item in second] == ["S1", "S2"], (
        "a re-observed snippet keeps its handle; a new one continues the run's count"
    )
    assert second[0].snippet_id == "doc-a:0001"
    assert second[1].snippet_id == "doc-a:0002"


def test_a_snippet_without_terminators_is_still_one_citable_sentence() -> None:
    state = EvidenceCoverageState("Why do people migrate?")
    state.record_search(
        query="migration",
        kind="original",
        documents=(DocumentObservation("doc-a", 1),),
    )

    handles = state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="migration drivers",
            snippets=(RelevantSnippet("doc-a:0001", 0, 20, "no terminator here", 0.9),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.9,
            pages_estimated=1,
        )
    )

    assert handles[0].sentences == ("no terminator here",)


def test_answerable_requires_a_grounded_draft() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)

    rejected = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "answerable",
                    "remaining_gap": "",
                    "draft_answer": "Conflict is a driver.",
                    "draft_nugget_ids": ["missing"],
                }
            ]
        }
    )
    state.apply_delta(_grounded_nugget_delta())
    accepted = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "answerable",
                    "remaining_gap": "",
                    "draft_answer": "Conflict and persecution can force displacement.",
                    "draft_nugget_ids": ["g1"],
                }
            ]
        }
    )

    assert rejected.rejected[0].code == "MISSING_GROUNDED_DRAFT"
    assert accepted.accepted_ids == ("n1",)
    assert state.report().needs[0].status == "answerable"


def test_support_tracks_documents_not_claimed_independence() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="another focus",
            snippets=(
                RelevantSnippet("doc-a:0002", 0, 15, "Another support.", 0.8),
            ),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.8,
            pages_estimated=1,
        )
    )
    state.apply_delta(
        {
            "add_evidence": [
                {
                    "nugget_id": "g1",
                    "cite": "S2",
                }
            ]
        }
    )
    assert state.report().nuggets[0].support == "single_document"

    state.record_snippet_page(
        SnippetPage(
            document_id="doc-b",
            focus_query="migration drivers",
            snippets=(RelevantSnippet("doc-b:0001", 0, 16, "Outside support.", 0.7),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.7,
            pages_estimated=1,
        )
    )
    state.apply_delta(
        {
            "add_evidence": [
                {
                    "nugget_id": "g1",
                    "cite": "S3",
                }
            ]
        }
    )

    assert state.report().nuggets[0].support == "multi_document"


def test_snippet_facet_requires_its_origin_snippet_id() -> None:
    state = _state_with_snippet()
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n1",
                    "narrative_span": "Why do people migrate",
                    "question": "Why?",
                }
            ],
            "add_facets": [
                {
                    "facet_id": "f1",
                    "need_ids": ["n1"],
                    "dimension": "driver",
                    "value": "conflict",
                    "origin": "snippet",
                    "origin_snippet_id": None,
                }
            ],
        }
    )

    assert state.report().facets == ()


def test_contradictions_are_symmetric_and_supersession_keeps_both_nuggets() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta("g1"))
    state.apply_delta(_grounded_nugget_delta("g2"))
    state.apply_delta(
        {
            "add_evidence": [
                {
                    "nugget_id": "g2",
                    "cite": "S1",
                }
            ],
            "supersede_nuggets": [{"nugget_id": "g1", "superseded_by": "g2"}],
        }
    )
    state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "g3",
                    "text": "A linked assertion.",
                    "need_ids": ["n1"],
                    "facet_ids": ["f1"],
                    "evidence": [
                        {
                            "cite": "S1",
                        }
                    ],
                    "contradicts": ["g1"],
                }
            ]
        }
    )

    nuggets = {nugget.nugget_id: nugget for nugget in state.report().nuggets}
    assert nuggets["g1"].contradicts == ("g3",)
    assert nuggets["g3"].contradicts == ("g1",)
    assert nuggets["g1"].superseded_by == "g2"
    assert tuple(nuggets) == ("g1", "g2", "g3")


def test_conflicted_requires_an_explicit_link_between_grounded_nuggets() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta("g1"))
    state.apply_delta(_grounded_nugget_delta("g2"))

    rejected = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "conflicted",
                    "remaining_gap": "Claims disagree.",
                    "draft_answer": None,
                    "draft_nugget_ids": ["g1", "g2"],
                }
            ]
        }
    )
    state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "g3",
                    "text": "Linked contrary assertion.",
                    "need_ids": ["n1"],
                    "facet_ids": ["f1"],
                    "evidence": [
                        {
                            "cite": "S1",
                        }
                    ],
                    "contradicts": ["g1"],
                }
            ]
        }
    )
    accepted = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "conflicted",
                    "remaining_gap": "Claims disagree.",
                    "draft_answer": None,
                    "draft_nugget_ids": ["g1", "g3"],
                }
            ]
        }
    )

    assert rejected.rejected[0].code == "MISSING_CONTRADICTION_LINK"
    assert accepted.accepted_ids == ("n1",)


def test_views_are_compact_scoped_and_safe() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "Challenges remain uncovered.",
                    "draft_answer": None,
                    "draft_nugget_ids": [],
                }
            ]
        }
    )

    frontier = state.view("frontier")
    need = json.loads(state.view("need:n1"))
    unknown = json.loads(state.view("need:unknown"))

    assert len(frontier) <= MAX_FRONTIER_CHARACTERS
    assert "Challenges remain uncovered." in frontier
    assert "Conflict and persecution force people to flee." not in frontier
    assert json.loads(frontier)["state_hash"] == state.report().state_hash
    assert [item["nugget_id"] for item in need["nuggets"]] == ["g1"]
    assert need["nuggets"][0]["evidence"][0]["snippet_id"] == "doc-a:0001"
    assert unknown["code"] == "UNKNOWN_SCOPE"


def test_actions_require_open_motivation_and_matching_consumption() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)

    assert state.choose_action(
        action="search", target="query", focus_query=None, motivating_ids=[], rationale="gap"
    )["code"] == "MISSING_MOTIVATION"
    accepted = state.choose_action(
        action="search", target="query", focus_query=None, motivating_ids=["n1"], rationale="gap"
    )
    assert accepted["state"] == "pending"
    assert state.choose_action(
        action="extract", target="doc-a", focus_query="drivers", motivating_ids=["n1"], rationale="gap"
    )["code"] == "PENDING_ACTION_EXISTS"
    assert state.require_pending_action(action="search", target="wrong", focus_query=None)["code"] == "ACTION_MISMATCH"
    assert state.require_pending_action(action="search", target="query", focus_query=None) is None

    state.apply_delta(_grounded_nugget_delta())
    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "answerable",
                    "remaining_gap": "",
                    "draft_answer": "Conflict is a driver.",
                    "draft_nugget_ids": ["g1"],
                }
            ]
        }
    )
    assert state.choose_action(
        action="search", target="query", focus_query=None, motivating_ids=["n1"], rationale="closed"
    )["code"] == "CLOSED_MOTIVATION"
    assert state.choose_action(
        action="search", target="query", focus_query=None, motivating_ids=["unknown"], rationale="unknown"
    )["code"] == "UNKNOWN_MOTIVATION"


def test_stop_rules_include_page_yield_saturation_not_action_count() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)

    assert state.choose_action(
        action="stop", target="completion", focus_query=None, motivating_ids=["n1"], rationale="done"
    )["code"] == "COMPLETION_OPEN_NEEDS"
    assert state.choose_action(
        action="stop", target="saturation", focus_query=None, motivating_ids=[], rationale="stalled"
    )["code"] == "MISSING_MOTIVATION"
    assert state.choose_action(
        action="stop", target="saturation", focus_query=None, motivating_ids=["n1"], rationale="stalled"
    )["code"] == "INSUFFICIENT_ZERO_YIELD_PAGES"

    for page_index in range(1, SATURATION_ZERO_YIELD_PAGES + 1):
        state.record_snippet_page(
            SnippetPage(
                document_id="doc-a",
                focus_query="migration drivers",
                snippets=(),
                next_cursor=None,
                page_index=page_index,
                residual_count=0,
                residual_top_score=None,
                returned_min_score=None,
                pages_estimated=SATURATION_ZERO_YIELD_PAGES + 1,
            )
        )

    accepted = state.choose_action(
        action="stop", target="saturation", focus_query=None, motivating_ids=["n1"], rationale="stalled"
    )
    assert accepted["state"] == "terminal"
    assert state.report().terminal_reason == "saturation"


def test_duplicate_evidence_is_a_version_preserving_noop() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())
    version_before_duplicate = state.report().state_version

    duplicate = state.apply_delta(
        {
            "add_evidence": [
                {
                    "nugget_id": "g1",
                    "cite": "S1",
                }
            ]
        }
    )

    assert duplicate.accepted_ids == ()
    assert duplicate.rejected[0].code == "DUPLICATE_EVIDENCE"
    assert duplicate.state_version == version_before_duplicate


def test_evidence_attachment_does_not_reset_zero_yield_saturation() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())
    for page_index in (1, 2):
        state.record_snippet_page(
            SnippetPage(
                document_id="doc-a",
                focus_query="migration drivers",
                snippets=(),
                next_cursor=None,
                page_index=page_index,
                residual_count=0,
                residual_top_score=None,
                returned_min_score=None,
                pages_estimated=4,
            )
        )
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-b",
            focus_query="migration drivers",
            snippets=(RelevantSnippet("doc-b:0001", 0, 16, "Outside support.", 0.7),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.7,
            pages_estimated=1,
        )
    )
    state.apply_delta(
        {
            "add_evidence": [
                {
                    "nugget_id": "g1",
                    "cite": "S2",
                }
            ]
        }
    )

    stopped = state.choose_action(
        action="stop",
        target="saturation",
        focus_query=None,
        motivating_ids=["n1"],
        rationale="three pages produced no new nuggets",
    )

    assert stopped["state"] == "terminal"


def test_frontier_renders_recent_yield_for_each_document_focus() -> None:
    state = _state_with_snippet()
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-b",
            focus_query="migration drivers",
            snippets=(RelevantSnippet("doc-b:0001", 0, 16, "Outside support.", 0.7),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.7,
            pages_estimated=1,
        )
    )
    _add_need_and_facet(state)
    state.apply_delta(_grounded_nugget_delta())

    documents = {
        item["document_id"]: item for item in json.loads(state.view("frontier"))["documents"]
    }

    assert documents["doc-a"]["recent_yield"] is True
    assert documents["doc-b"]["recent_yield"] is False


def test_terminal_stop_rejects_pending_retrieval_action() -> None:
    state = _state_with_snippet()
    _add_need_and_facet(state)
    state.choose_action(
        action="search",
        target="targeted migration query",
        focus_query=None,
        motivating_ids=["n1"],
        rationale="an open need remains",
    )
    for page_index in range(1, SATURATION_ZERO_YIELD_PAGES + 1):
        state.record_snippet_page(
            SnippetPage(
                document_id="doc-a",
                focus_query="migration drivers",
                snippets=(),
                next_cursor=None,
                page_index=page_index,
                residual_count=0,
                residual_top_score=None,
                returned_min_score=None,
                pages_estimated=SATURATION_ZERO_YIELD_PAGES + 1,
            )
        )

    stopped = state.choose_action(
        action="stop",
        target="saturation",
        focus_query=None,
        motivating_ids=["n1"],
        rationale="pages were empty",
    )

    assert stopped["code"] == "PENDING_ACTION_EXISTS"


def test_terminal_stop_rejects_all_later_action_operations() -> None:
    state = EvidenceCoverageState("Why do people migrate and what challenges do they face?")
    state.choose_action(
        action="stop",
        target="completion",
        focus_query=None,
        motivating_ids=[],
        rationale="no needs were recorded",
    )

    later_action = state.choose_action(
        action="search",
        target="query",
        focus_query=None,
        motivating_ids=["n1"],
        rationale="should not be accepted",
    )
    later_consumption = state.require_pending_action(
        action="search", target="query", focus_query=None
    )

    assert later_action["code"] == "TERMINAL_STATE"
    assert later_consumption["code"] == "TERMINAL_STATE"


def test_paginate_action_requires_a_recorded_page_for_exact_document_focus() -> None:
    state = EvidenceCoverageState("Explain migration drivers.")

    assert (
        state.expected_snippet_action(
            document_id="doc-a",
            focus_query="migration drivers",
            cursor="opaque-page-two",
        )
        is None
    )

    state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="migration drivers",
            snippets=(),
            next_cursor="opaque-page-two",
            page_index=0,
            residual_count=1,
            residual_top_score=0.6,
            returned_min_score=None,
            pages_estimated=2,
        )
    )

    assert (
        state.expected_snippet_action(
            document_id="doc-a",
            focus_query="migration drivers",
            cursor="opaque-page-two",
        )
        == "paginate"
    )
    assert (
        state.expected_snippet_action(
            document_id="doc-a",
            focus_query="unseen focus",
            cursor="opaque-page-two",
        )
        is None
    )


def _nugget_with(cite: str, text: str = "A claim.", **extra):
    row = {
        "nugget_id": extra.pop("nugget_id", "x1"),
        "text": text,
        "need_ids": ["n1"],
        "facet_ids": ["f1"],
        "evidence": [{"cite": cite}],
        "contradicts": [],
    }
    row.update(extra)
    return {"add_nuggets": [row]}


def _state_with_degenerate_sentence():
    """A snippet whose splitter output includes a list marker and a case name.

    Both are what the live run actually produced: the sentence splitter breaks
    on "v." and on numbered list markers, minting citable spans that carry no
    information.
    """
    state = EvidenceCoverageState("Why do people migrate and what challenges do they face?")
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="immigration law",
            snippets=(
                RelevantSnippet(
                    "doc-a:0001",
                    0,
                    120,
                    "Plyler v. Doe concerned schooling. 2. States may not charge tuition.",
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
    return state


def test_the_splitter_rejoins_fragments_it_split_off_a_real_sentence() -> None:
    """The live failure at source: "Plyler v." was its own citable span."""
    state = _state_with_degenerate_sentence()
    observation = next(iter(state._snippets.values()))
    spans = [
        observation.text[start:end] for start, end in observation.sentence_spans
    ]

    assert spans == [
        "Plyler v. Doe concerned schooling.",
        "2. States may not charge tuition.",
    ]


def test_every_span_the_splitter_emits_is_citable() -> None:
    state = _state_with_degenerate_sentence()
    _add_need_and_facet(state)

    for index in range(1, 3):
        result = state.apply_delta(
            _nugget_with(f"S1.{index}", nugget_id=f"n{index}")
        )
        assert result.accepted_ids == (f"n{index}",), result.rejected


def test_a_wholly_degenerate_snippet_is_still_refused_as_a_backstop() -> None:
    """Merging cannot save a snippet that is nothing but a marker."""
    state = EvidenceCoverageState(
        "Why do people migrate and what challenges do they face?"
    )
    state.record_snippet_page(
        SnippetPage(
            document_id="doc-a",
            focus_query="immigration law",
            snippets=(RelevantSnippet("doc-a:0001", 0, 2, "2.", 0.9),),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=0.9,
            pages_estimated=1,
        )
    )
    _add_need_and_facet(state)

    result = state.apply_delta(_nugget_with("S1"))

    assert result.accepted_ids == ()
    assert [r.code for r in result.rejected] == ["DEGENERATE_CITATION"]


def test_whole_snippet_citation_still_works() -> None:
    state = _state_with_degenerate_sentence()
    _add_need_and_facet(state)

    assert state.apply_delta(_nugget_with("S1")).accepted_ids == ("x1",)


def test_support_ratio_separates_a_grounded_claim_from_a_recited_one() -> None:
    """The measured failure mode: a claim whose words are not in what it cites."""
    state = _state_with_sentences()
    _add_need_and_facet(state)

    state.apply_delta(
        _nugget_with("S1.2", text="Persecution forces flight.", nugget_id="grounded")
    )
    state.apply_delta(
        _nugget_with(
            "S1.2",
            text=(
                "The Supreme Court held that the Equal Protection Clause bars "
                "states from denying enrollment to undocumented children."
            ),
            nugget_id="recited",
        )
    )

    ratios = {n.nugget_id: n.support_ratio for n in state.report().nuggets}
    assert ratios["grounded"] > ratios["recited"]
    assert ratios["recited"] < 0.5


def test_support_ratio_never_rejects_a_claim() -> None:
    """A lexical screen must not discard grounded evidence."""
    state = _state_with_sentences()
    _add_need_and_facet(state)

    result = state.apply_delta(
        _nugget_with("S1.2", text="Entirely unrelated vocabulary about spacecraft.")
    )

    assert result.accepted_ids == ("x1",)
    assert state.report().nuggets[0].support_ratio < 0.5


def test_importance_is_derived_from_selection_not_asserted() -> None:
    """An asserted label collapsed to 94% vital on a live run; a selection cannot."""
    state = _state_with_sentences()
    _add_need_and_facet(state)
    state.apply_delta(_nugget_with("S1", nugget_id="chosen"))
    state.apply_delta(_nugget_with("S1.2", nugget_id="passed_over"))

    assert all(n.importance == "okay" for n in state.report().nuggets)

    state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "more needed",
                    "draft_nugget_ids": ["chosen"],
                }
            ]
        }
    )

    importance = {n.nugget_id: n.importance for n in state.report().nuggets}
    assert importance == {"chosen": "vital", "passed_over": "okay"}


def test_a_nugget_delta_cannot_claim_its_own_importance() -> None:
    state = _state_with_sentences()
    _add_need_and_facet(state)

    state.apply_delta(_nugget_with("S1", importance="vital"))

    assert state.report().nuggets[0].importance == "okay"


def test_the_draft_selection_is_capped_so_selection_actually_selects() -> None:
    """Unbounded, a live run drafted all 145 nuggets and the signal went flat."""
    from trec_rag.deepagent_evidence import MAX_DRAFT_NUGGETS_PER_NEED

    state = _state_with_sentences()
    _add_need_and_facet(state)
    ids = []
    for index in range(MAX_DRAFT_NUGGETS_PER_NEED + 1):
        nugget_id = f"g{index}"
        ids.append(nugget_id)
        state.apply_delta(_nugget_with("S1", nugget_id=nugget_id))

    def draft(selected):
        return state.apply_delta(
            {
                "set_need_status": [
                    {
                        "need_id": "n1",
                        "status": "partial",
                        "remaining_gap": "more needed",
                        "draft_nugget_ids": selected,
                    }
                ]
            }
        )

    too_many = draft(ids)
    assert too_many.accepted_ids == ()
    assert [r.code for r in too_many.rejected] == ["TOO_MANY_DRAFT_NUGGETS"]

    assert draft(ids[:MAX_DRAFT_NUGGETS_PER_NEED]).accepted_ids == ("n1",)
    # A need with fewer good nuggets must not have to pad to the cap.
    assert draft(ids[:2]).accepted_ids == ("n1",)


def test_a_partial_status_cannot_draft_a_nugget_that_does_not_exist_yet() -> None:
    """Reproduces a reviewed defect: phantom ids minting vital labels.

    Grounding used to be enforced only for terminal statuses, so a "partial"
    row could pre-list ids that arrived later and be labelled vital on arrival,
    inflating the submission ranker's top-weighted feature with no real
    selection behind it.
    """
    state = _state_with_sentences()
    _add_need_and_facet(state)

    phantom = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "more needed",
                    "draft_nugget_ids": ["G1", "G2"],
                }
            ]
        }
    )

    assert phantom.accepted_ids == ()
    assert [r.code for r in phantom.rejected] == ["MISSING_GROUNDED_DRAFT"]

    state.apply_delta(_nugget_with("S1", nugget_id="G1"))
    assert state.report().nuggets[0].importance == "okay"


def test_a_partial_status_cannot_draft_another_needs_nugget() -> None:
    state = _state_with_sentences()
    _add_need_and_facet(state)
    state.apply_delta(
        {
            "add_needs": [
                {
                    "need_id": "n2",
                    "narrative_span": "people migrate",
                    "question": "Where do people migrate to?",
                }
            ]
        }
    )
    state.apply_delta(_nugget_with("S1", nugget_id="owned_by_n1"))

    result = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n2",
                    "status": "partial",
                    "remaining_gap": "more needed",
                    "draft_nugget_ids": ["owned_by_n1"],
                }
            ]
        }
    )

    assert [r.code for r in result.rejected] == ["MISSING_GROUNDED_DRAFT"]


def test_a_partial_status_with_a_real_grounded_nugget_is_still_accepted() -> None:
    state = _state_with_sentences()
    _add_need_and_facet(state)
    state.apply_delta(_nugget_with("S1", nugget_id="real"))

    result = state.apply_delta(
        {
            "set_need_status": [
                {
                    "need_id": "n1",
                    "status": "partial",
                    "remaining_gap": "more needed",
                    "draft_nugget_ids": ["real"],
                }
            ]
        }
    )

    assert result.accepted_ids == ("n1",)
    assert state.report().nuggets[0].importance == "vital"

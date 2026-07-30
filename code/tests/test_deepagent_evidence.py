from __future__ import annotations

import json

from trec_rag.deepagent_evidence import (
    MAX_FRONTIER_CHARACTERS,
    SATURATION_ZERO_YIELD_PAGES,
    DocumentObservation,
    EvidenceCoverageState,
)
from trec_rag.deepagent_snippets import RelevantSnippet, SnippetPage


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
                "evidence": [
                    {
                        "snippet_id": "doc-a:0001",
                        "quote": "Conflict and persecution force people to flee.",
                    }
                ],
                "contradicts": [],
            }
        ]
    }


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


def test_ungrounded_quote_rejects_only_its_nugget() -> None:
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
            "evidence": [{"snippet_id": "doc-a:0001", "quote": "Invented quotation."}],
            "contradicts": [],
        },
    ]

    result = state.apply_delta(delta)

    assert result.accepted_ids == ("g1",)
    assert result.rejected[0].code == "UNGROUNDED_QUOTE"
    assert tuple(nugget.nugget_id for nugget in state.report().nuggets) == ("g1",)


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
                    "snippet_id": "doc-a:0002",
                    "quote": "Another support.",
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
                    "snippet_id": "doc-b:0001",
                    "quote": "Outside support.",
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
                    "snippet_id": "doc-a:0001",
                    "quote": "Conflict and persecution force people to flee.",
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
                            "snippet_id": "doc-a:0001",
                            "quote": "Conflict and persecution force people to flee.",
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
                            "snippet_id": "doc-a:0001",
                            "quote": "Conflict and persecution force people to flee.",
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
                    "snippet_id": "doc-a:0001",
                    "quote": "Conflict and persecution force people to flee.",
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
                    "snippet_id": "doc-b:0001",
                    "quote": "Outside support.",
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

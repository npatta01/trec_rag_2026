"""Executable contract for the code-owned query-planner v2 invariants.

These tests intentionally use a separate v2 API while the v1 planner remains
available for its immutable diagnostic record.  They specify the boundary
between model proposals and deterministic Python validation/rendering.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from trec_rag.query_planner import (
    MAX_ANCHOR_TOKENS,
    MAX_COVERAGE_ITEMS,
    MAX_COVERAGE_ITEMS_PER_FACET,
    MAX_FACET_EXPANSION_TERMS,
    MAX_FACET_NEW_TOKENS,
    MAX_FACETS,
    MAX_GLOBAL_ANCHORS,
    MAX_GLOBAL_EXPANSION_TERMS,
    TOKENIZER_VERSION,
    V2_SCHEMA_VERSION,
    QueryPlanValidationError,
    TokenRange,
    global_expansion_new_token_cap,
    merge_adjacent_token_ranges,
    parse_query_plan_v2,
    plan_or_original_fallback,
    render_query_plan_v2,
    resolve_token_ranges,
    tokenize_narrative,
)
from trec_rag.topics import Topic


def _topic() -> Topic:
    return Topic(
        id="scope",
        title="Scoped housing comparison",
        narrative="Compare housing costs in Dubai with housing costs in the UK.",
    )


def _term(
    term: str,
    *anchor_refs: str,
    relation: str = "common_variant",
) -> dict[str, object]:
    return {
        "term": term,
        "relation": relation,
        "anchor_refs": list(anchor_refs or ("a_subject",)),
    }


def _valid_payload() -> dict[str, object]:
    # Token IDs for the narrative in _topic():
    # 0 Compare | 1 housing | 2 costs | 3 in | 4 Dubai | 5 with |
    # 6 housing | 7 costs | 8 in | 9 the | 10 UK | 11 .
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "topic_id": "scope",
        "anchors": [
            {
                "anchor_id": "a_subject",
                "range": {"start_token": 1, "end_token": 3},
                "kind": "topic",
                "scope": "global",
                "coverage_refs": [],
            },
            {
                "anchor_id": "a_dubai",
                "range": {"start_token": 4, "end_token": 5},
                "kind": "geography",
                "scope": "coverage",
                "coverage_refs": ["c_dubai"],
            },
            {
                "anchor_id": "a_uk",
                "range": {"start_token": 10, "end_token": 11},
                "kind": "geography",
                "scope": "coverage",
                "coverage_refs": ["c_uk"],
            },
        ],
        "coverage_items": [
            {
                "coverage_id": "c_dubai",
                "source_span_refs": [{"start_token": 1, "end_token": 5}],
            },
            {
                "coverage_id": "c_uk",
                "source_span_refs": [{"start_token": 6, "end_token": 11}],
            },
        ],
        "facets": [
            {
                "facet_id": "f_dubai",
                "coverage_refs": ["c_dubai"],
                "expansion_terms": [_term("rental prices", "a_subject")],
            },
            {
                "facet_id": "f_uk",
                "coverage_refs": ["c_uk"],
                "expansion_terms": [_term("rental prices", "a_subject")],
            },
        ],
        "global_expansion": {"terms": []},
    }


def _parse(
    payload: dict[str, object] | None = None,
    *,
    topic: Topic | None = None,
):
    topic = topic or _topic()
    tape = tokenize_narrative(topic.narrative)
    return parse_query_plan_v2(payload or _valid_payload(), topic=topic, token_tape=tape)


def test_token_tape_freezes_unicode_codepoint_offsets_and_token_boundaries():
    narrative = "O’Reilly's co-op—costs €20?"

    tape = tokenize_narrative(narrative)

    assert TOKENIZER_VERSION == "narrative_token_tape_v1"
    assert tape.version == TOKENIZER_VERSION
    assert tape.normalization == "none"
    assert tape.offset_unit == "unicode_code_points"
    assert tape.narrative_sha256 == hashlib.sha256(narrative.encode("utf-8")).hexdigest()
    assert tape.token_count == 7
    assert [(row.text, row.start_char, row.end_char) for row in tape.tokens] == [
        ("O’Reilly's", 0, 10),
        ("co-op", 11, 16),
        ("—", 16, 17),
        ("costs", 17, 22),
        ("€", 23, 24),
        ("20", 24, 26),
        ("?", 26, 27),
    ]
    assert tape.to_dict()["tokens"][-1] == {
        "text": "?",
        "start_char": 26,
        "end_char": 27,
    }


def test_final_token_end_is_a_valid_exclusive_range_boundary():
    narrative = "O’Reilly's co-op—costs €20?"
    tape = tokenize_narrative(narrative)

    (resolved,) = resolve_token_ranges(
        narrative,
        tape,
        (TokenRange(start_token=5, end_token=tape.token_count),),
    )

    assert resolved.token_range == TokenRange(5, 7)
    assert (resolved.start_char, resolved.end_char, resolved.text) == (24, 27, "20?")


def test_adjacent_ranges_merge_but_discontinuous_ranges_remain_separate():
    adjacent = merge_adjacent_token_ranges(
        (TokenRange(3, 5), TokenRange(1, 3), TokenRange(5, 6))
    )
    discontinuous = merge_adjacent_token_ranges(
        (TokenRange(1, 3), TokenRange(4, 6))
    )

    assert adjacent == (TokenRange(1, 6),)
    assert discontinuous == (TokenRange(1, 3), TokenRange(4, 6))

    narrative = "zero alpha beta and gamma delta"
    tape = tokenize_narrative(narrative)
    resolved = resolve_token_ranges(narrative, tape, discontinuous)
    assert [row.text for row in resolved] == ["alpha beta", "gamma delta"]
    assert "and" not in tuple(row.text for row in resolved)


@pytest.mark.parametrize(
    "token_range",
    [
        TokenRange(-1, 1),
        TokenRange(2, 2),
        TokenRange(3, 2),
        TokenRange(0, 13),
    ],
)
def test_range_resolution_rejects_negative_empty_reversed_and_out_of_bounds(token_range):
    topic = _topic()
    tape = tokenize_narrative(topic.narrative)

    with pytest.raises(QueryPlanValidationError, match="range|token|bound|empty"):
        resolve_token_ranges(topic.narrative, tape, (token_range,))


def test_range_resolution_rejects_duplicate_ranges():
    topic = _topic()
    tape = tokenize_narrative(topic.narrative)
    duplicate = TokenRange(1, 3)

    with pytest.raises(QueryPlanValidationError, match="duplicate"):
        resolve_token_ranges(topic.narrative, tape, (duplicate, duplicate))


@pytest.mark.parametrize("target", ["anchor", "coverage"])
def test_plan_rejects_ranges_without_analyzed_content(target):
    payload = _valid_payload()
    punctuation = {"start_token": 11, "end_token": 12}
    if target == "anchor":
        payload["anchors"][0]["range"] = punctuation
    else:
        payload["coverage_items"][0]["source_span_refs"] = [punctuation]

    with pytest.raises(QueryPlanValidationError, match="content-bearing|content token"):
        _parse(payload)


def test_anchor_range_is_limited_to_eight_analyzed_tokens():
    topic = Topic(
        id="long-anchor",
        title="Long anchor",
        narrative="alpha bravo charlie delta echo foxtrot golf hotel india juliet.",
    )
    payload = {
        "schema_version": V2_SCHEMA_VERSION,
        "topic_id": topic.id,
        "anchors": [
            {
                "anchor_id": "a_long",
                "range": {"start_token": 0, "end_token": 9},
                "kind": "topic",
                "scope": "global",
                "coverage_refs": [],
            }
        ],
        "coverage_items": [
            {
                "coverage_id": "c_all",
                "source_span_refs": [{"start_token": 0, "end_token": 10}],
            }
        ],
        "facets": [
            {"facet_id": "f_all", "coverage_refs": ["c_all"], "expansion_terms": []}
        ],
        "global_expansion": {"terms": []},
    }

    with pytest.raises(QueryPlanValidationError, match="anchor|8|eight"):
        _parse(payload, topic=topic)


@pytest.mark.parametrize("partition_error", ["missing", "duplicate", "empty_facet"])
def test_every_coverage_item_partitions_into_exactly_one_nonempty_facet(partition_error):
    payload = _valid_payload()
    if partition_error == "missing":
        payload["coverage_items"].append(
            {
                "coverage_id": "c_unreferenced",
                "source_span_refs": [{"start_token": 0, "end_token": 1}],
            }
        )
    elif partition_error == "duplicate":
        payload["facets"][0]["coverage_refs"].append("c_uk")
    else:
        payload["facets"][1]["coverage_refs"] = []

    with pytest.raises(
        QueryPlanValidationError,
        match="exactly one facet|partition|nonempty|coverage",
    ):
        _parse(payload)


def test_coverage_item_count_is_capped_at_sixteen_before_partitioning():
    payload = _valid_payload()
    for index in range(15):
        payload["coverage_items"].append(
            {
                "coverage_id": f"c_extra_{index}",
                "source_span_refs": [{"start_token": 1, "end_token": 3}],
            }
        )

    with pytest.raises(QueryPlanValidationError, match="coverage|16|sixteen"):
        _parse(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("global_has_coverage", "global|coverage_refs"),
        ("coverage_has_no_refs", "coverage|coverage_refs"),
        ("no_global_anchor", "global anchor|global anchors"),
        ("no_global_subject", "entity|topic"),
    ],
)
def test_scoped_anchor_contract_is_mechanically_enforced(mutation, message):
    payload = _valid_payload()
    if mutation == "global_has_coverage":
        payload["anchors"][0]["coverage_refs"] = ["c_dubai"]
    elif mutation == "coverage_has_no_refs":
        payload["anchors"][1]["coverage_refs"] = []
    elif mutation == "no_global_anchor":
        payload["anchors"][0]["scope"] = "coverage"
        payload["anchors"][0]["coverage_refs"] = ["c_dubai", "c_uk"]
    else:
        payload["anchors"][0]["kind"] = "relation"

    with pytest.raises(QueryPlanValidationError, match=message):
        _parse(payload)


def test_no_more_than_four_global_anchors_are_accepted():
    payload = _valid_payload()
    payload["anchors"].extend(
        [
            {
                "anchor_id": "a_compare",
                "range": {"start_token": 0, "end_token": 1},
                "kind": "relation",
                "scope": "global",
                "coverage_refs": [],
            },
            {
                "anchor_id": "a_uk_housing",
                "range": {"start_token": 6, "end_token": 7},
                "kind": "topic",
                "scope": "global",
                "coverage_refs": [],
            },
            {
                "anchor_id": "a_uk_costs",
                "range": {"start_token": 7, "end_token": 8},
                "kind": "metric",
                "scope": "global",
                "coverage_refs": [],
            },
            {
                "anchor_id": "a_compare_with",
                "range": {"start_token": 0, "end_token": 3},
                "kind": "comparison",
                "scope": "global",
                "coverage_refs": [],
            },
        ]
    )

    with pytest.raises(QueryPlanValidationError, match="global anchor|at most 4|maximum.*4"):
        _parse(payload)


def test_renderer_inherits_global_and_applicable_scoped_anchors_without_leakage():
    topic = _topic()
    tape = tokenize_narrative(topic.narrative)
    plan = parse_query_plan_v2(_valid_payload(), topic=topic, token_tape=tape)

    rows = render_query_plan_v2(topic, plan, token_tape=tape)
    by_variant = {row.variant_name: row for row in rows}

    dubai = by_variant["facet:f_dubai"]
    uk = by_variant["facet:f_uk"]
    assert "housing costs" in dubai.query_text
    assert "Dubai" in dubai.query_text
    assert "UK" not in dubai.query_text
    assert "housing costs" in uk.query_text
    assert "UK" in uk.query_text
    assert "Dubai" not in uk.query_text


def test_facet_expansion_refs_must_be_a_subset_of_inherited_anchor_refs():
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [_term("rental prices", "a_uk")]

    with pytest.raises(
        QueryPlanValidationError,
        match="expansion|anchor_refs|inherited|scope",
    ):
        _parse(payload)


def test_python_rejects_duplicate_id_references_without_decoder_unique_items():
    payload = _valid_payload()
    payload["facets"][0]["coverage_refs"] = ["c_dubai", "c_dubai"]

    with pytest.raises(QueryPlanValidationError, match="duplicate references"):
        _parse(payload)


@pytest.mark.parametrize(
    ("unsafe_term", "message"),
    [
        ("２０", "numeric|number|Unicode"),
        ("site:gov.uk", "operator|field syntax|query syntax"),
        ("site：gov.uk", "operator|field syntax|query syntax"),
        ("rent AND prices", "operator|query syntax"),
        ("rent ＡＮＤ prices", "operator|query syntax"),
        ("［rent］", "operator|query syntax"),
    ],
)
def test_proposed_expansions_reject_absent_unicode_numbers_and_query_operators(
    unsafe_term,
    message,
):
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [_term(unsafe_term, "a_subject")]

    with pytest.raises(QueryPlanValidationError, match=message):
        _parse(payload)


def test_numeric_runs_require_exact_membership_not_substring_membership():
    topic = Topic(
        id="scope",
        title="Dated housing comparison",
        narrative=(
            "Compare housing costs in Dubai with housing costs in the UK in 2020."
        ),
    )
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [_term("20", "a_subject")]

    with pytest.raises(QueryPlanValidationError, match="numeric|number|Unicode"):
        _parse(payload, topic=topic)


def test_expansion_word_limit_counts_repeated_analyzer_tokens():
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [
        _term("rent rent rent rent", "a_subject")
    ]

    with pytest.raises(QueryPlanValidationError, match="at most 3 analyzed words"):
        _parse(payload)


def test_anchor_token_limit_counts_repeated_analyzer_tokens():
    topic = Topic(
        id="scope",
        title="Repeated anchor",
        narrative=(
            "Compare bank bank bank bank bank bank bank bank bank in Dubai "
            "with housing costs in the UK."
        ),
    )
    payload = _valid_payload()
    payload["anchors"][0]["range"] = {"start_token": 1, "end_token": 10}

    with pytest.raises(QueryPlanValidationError, match="anchor maximum of 8"):
        _parse(payload, topic=topic)


def test_unsafe_proposal_is_a_hard_failure_even_when_the_filter_rejects_it():
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [
        _term("rental prices", "a_subject"),
        _term("site:gov.uk", "a_subject"),
    ]

    with pytest.raises(QueryPlanValidationError) as raised:
        _parse(payload)

    assert raised.value.hard_failure is True
    assert any(
        row.term == "site:gov.uk"
        and row.status == "rejected"
        and row.rejection_reason in {"query_operator", "field_syntax"}
        for row in raised.value.expansion_audit
    )


def test_invalid_plan_falls_back_to_the_exact_original_only_and_retains_failure():
    topic = Topic(
        id="scope",
        title="Scoped housing comparison",
        narrative="Compare  housing costs in Dubai with housing costs in the UK.\n",
    )
    tape = tokenize_narrative(topic.narrative)
    payload = _valid_payload()
    payload["facets"][0]["coverage_refs"] = ["does_not_exist"]

    outcome = plan_or_original_fallback(topic, payload, token_tape=tape)

    assert outcome.used_fallback is True
    assert outcome.plan is None
    assert outcome.failure is not None
    assert outcome.failure.status == "plan_validation_error"
    assert len(outcome.rendered_queries) == 1
    (fallback,) = outcome.rendered_queries
    assert fallback.variant_name == "original"
    assert fallback.source_type == "original"
    assert fallback.query_text == topic.narrative
    assert fallback.components == (topic.narrative,)


def test_frozen_planner_caps_are_public_and_exact():
    assert MAX_ANCHOR_TOKENS == 8
    assert MAX_COVERAGE_ITEMS == 16
    assert MAX_COVERAGE_ITEMS_PER_FACET == 2
    assert MAX_FACETS == 8
    assert MAX_GLOBAL_ANCHORS == 4
    assert MAX_GLOBAL_EXPANSION_TERMS == 8
    assert MAX_FACET_EXPANSION_TERMS == 3
    assert MAX_FACET_NEW_TOKENS == 6


def test_facet_count_is_capped_at_eight_before_partitioning():
    payload = _valid_payload()
    template = payload["facets"][0]
    payload["facets"] = [
        {**deepcopy(template), "facet_id": f"f_extra_{index}"}
        for index in range(9)
    ]

    with pytest.raises(QueryPlanValidationError, match="facet|at most 8|maximum.*8"):
        _parse(payload)


def test_global_new_token_cap_is_ten_percent_rounded_up_and_capped_at_eight():
    ten = " ".join(f"lexeme{index}" for index in range(10))
    eleven = " ".join(f"lexeme{index}" for index in range(11))
    hundred = " ".join(f"lexeme{index}" for index in range(100))

    assert global_expansion_new_token_cap("") == 0
    assert global_expansion_new_token_cap(ten) == 1
    assert global_expansion_new_token_cap(eleven) == 2
    assert global_expansion_new_token_cap(hundred) == 8


@pytest.mark.parametrize(
    ("target", "terms", "message"),
    [
        (
            "global",
            [
                "rental",
                "leasing",
                "tenancy",
                "rents",
                "pricing",
                "affordability",
                "accommodation",
                "dwelling",
                "residence",
            ],
            "global|8|eight",
        ),
        (
            "facet",
            ["rental", "leasing", "tenancy", "affordability"],
            "facet|3|three",
        ),
    ],
)
def test_expansion_object_caps_are_enforced_before_term_filtering(target, terms, message):
    payload = _valid_payload()
    objects = [_term(term, "a_subject") for term in terms]
    if target == "global":
        payload["global_expansion"]["terms"] = objects
    else:
        payload["facets"][0]["expansion_terms"] = objects

    with pytest.raises(QueryPlanValidationError, match=message):
        _parse(payload)


def test_each_coverage_item_and_facet_obeys_range_and_grouping_caps():
    payload = _valid_payload()
    payload["coverage_items"][0]["source_span_refs"] = [
        {"start_token": 1, "end_token": 2},
        {"start_token": 4, "end_token": 5},
        {"start_token": 10, "end_token": 11},
    ]

    with pytest.raises(QueryPlanValidationError, match="range|2|two"):
        _parse(payload)

    payload = _valid_payload()
    payload["coverage_items"].append(
        {
            "coverage_id": "c_third",
            "source_span_refs": [{"start_token": 0, "end_token": 1}],
        }
    )
    payload["facets"][0]["coverage_refs"] = ["c_dubai", "c_uk", "c_third"]
    with pytest.raises(QueryPlanValidationError, match="at most 2|one or two|maximum.*2"):
        _parse(payload)


def test_each_expansion_term_is_limited_to_three_analyzed_words():
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [
        _term("rental housing price index", "a_subject")
    ]

    with pytest.raises(QueryPlanValidationError, match="at most 3|three.*word|3.*word"):
        _parse(payload)


def test_rendering_rejects_facet_expansions_over_six_new_unique_tokens():
    topic = _topic()
    tape = tokenize_narrative(topic.narrative)
    payload = _valid_payload()
    payload["facets"][0]["expansion_terms"] = [
        _term("rental prices leasing", "a_subject"),
        _term("tenancy payments affordability", "a_subject"),
        _term("accommodation dwellings", "a_subject"),
    ]
    plan = parse_query_plan_v2(payload, topic=topic, token_tape=tape)

    with pytest.raises(QueryPlanValidationError, match="facet|6|six|new token"):
        render_query_plan_v2(topic, plan, token_tape=tape)


def test_rendering_rejects_global_expansion_over_dynamic_new_token_cap():
    topic = _topic()
    tape = tokenize_narrative(topic.narrative)
    payload = deepcopy(_valid_payload())
    # This narrative has five unique analyzed content terms, hence a one-token
    # global allowance. One object can still exceed that separate token budget.
    payload["global_expansion"]["terms"] = [_term("rental prices", "a_subject")]
    plan = parse_query_plan_v2(payload, topic=topic, token_tape=tape)

    with pytest.raises(QueryPlanValidationError, match="global|1|one|new token"):
        render_query_plan_v2(topic, plan, token_tape=tape)

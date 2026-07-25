from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from trec_rag.facet_query_planning import (
    Anchor,
    CoverageItem,
    Expansion,
    Facet,
    FacetPlan,
    FacetPlanValidationError,
    TokenRange,
    render_facet_queries,
    tokenize_narrative,
    validate_facet_plan,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


NARRATIVE = "Café owners compare rent increases and zoning changes."


def _topic(*, topic_id: str = "housing-1", narrative: str = NARRATIVE) -> Topic:
    return Topic(id=topic_id, title="A title that must not matter", narrative=narrative)


def _plan(
    *,
    topic_id: str = "housing-1",
    narrative: str = NARRATIVE,
    anchors: tuple[Anchor, ...] | None = None,
    coverage_items: tuple[CoverageItem, ...] | None = None,
    facets: tuple[Facet, ...] | None = None,
) -> FacetPlan:
    return FacetPlan(
        topic_id=topic_id,
        narrative_sha256=sha256(narrative.encode("utf-8")).hexdigest(),
        anchors=anchors
        if anchors is not None
        else (
            Anchor(
                anchor_id="housing-topic",
                token_range=TokenRange(0, 2),
                kind="topic",
                scope="global",
                coverage_refs=(),
            ),
            Anchor(
                anchor_id="rent-anchor",
                token_range=TokenRange(3, 5),
                kind="aspect",
                scope="coverage",
                coverage_refs=("rent",),
            ),
        ),
        coverage_items=coverage_items
        if coverage_items is not None
        else (
            CoverageItem(
                coverage_id="rent",
                source_ranges=(TokenRange(3, 5),),
            ),
        ),
        facets=facets
        if facets is not None
        else (
            Facet(
                facet_id="rent-effects",
                coverage_refs=("rent",),
                expansions=(
                    Expansion(
                        term="rental market",
                        relation="neutral_search_term",
                        anchor_refs=("rent-anchor",),
                    ),
                ),
            ),
        ),
    )


HOUSING_NARRATIVE = "Housing tenants compare rent increases and zoning changes."


def _housing_topic() -> Topic:
    return Topic(
        id="housing-comparison",
        title="Irrelevant title text must never enter a query",
        narrative=HOUSING_NARRATIVE,
    )


def _housing_plan(*, facets: tuple[Facet, ...] | None = None) -> FacetPlan:
    return _plan(
        topic_id="housing-comparison",
        narrative=HOUSING_NARRATIVE,
        anchors=(
            Anchor("housing", TokenRange(0, 2), "topic", "global", ()),
            Anchor("comparison", TokenRange(2, 3), "context", "global", ()),
            Anchor("rent", TokenRange(3, 5), "aspect", "coverage", ("rent",)),
            Anchor("zoning", TokenRange(6, 8), "aspect", "coverage", ("zoning",)),
        ),
        coverage_items=(
            CoverageItem("rent", (TokenRange(4, 5), TokenRange(3, 4))),
            CoverageItem("zoning", (TokenRange(6, 8),)),
        ),
        facets=facets
        if facets is not None
        else (
            Facet(
                "rent-effects",
                ("rent",),
                (
                    Expansion("rent increases", "neutral_search_term", ("rent",)),
                    Expansion("affordability", "neutral_search_term", ("rent",)),
                    Expansion("lease costs", "neutral_search_term", ("rent",)),
                ),
            ),
            Facet(
                "zoning-effects",
                ("zoning",),
                (Expansion("land use", "neutral_search_term", ("zoning",)),),
            ),
        ),
    )


def test_rendering_uses_narrative_order_scoped_anchors_and_declared_expansions() -> None:
    topic = _housing_topic()
    plan = _housing_plan()

    first_result = render_facet_queries(topic, plan)
    second_result = render_facet_queries(topic, plan)

    assert first_result == second_result
    assert first_result.used_fallback is False
    assert first_result.error is None
    assert first_result.queries == (
        QueryVariant(
            topic_id="housing-comparison",
            variant_name="facet:rent-effects",
            query_text="rent increases Housing tenants compare affordability lease costs",
            source_type="structured_facet",
        ),
        QueryVariant(
            topic_id="housing-comparison",
            variant_name="facet:zoning-effects",
            query_text="zoning changes. Housing tenants compare land use",
            source_type="structured_facet",
        ),
    )
    assert all(topic.title not in query.query_text for query in first_result.queries)


@pytest.mark.parametrize(
    "relation",
    ("alias", "acronym", "technical_term", "common_variant", "neutral_search_term"),
)
def test_validation_accepts_each_allowed_lexical_expansion_relation(relation: str) -> None:
    plan = _housing_plan(
        facets=(
            Facet(
                "rent-effects",
                ("rent",),
                (Expansion("affordability", relation, ("rent",)),),
            ),
            Facet("zoning-effects", ("zoning",), ()),
        )
    )

    assert validate_facet_plan(_housing_topic(), plan).plan == plan


@pytest.mark.parametrize("relation", ("related", "synonym", "semantic"))
def test_validation_rejects_nonlexical_expansion_relations(relation: str) -> None:
    plan = _housing_plan(
        facets=(
            Facet(
                "rent-effects",
                ("rent",),
                (Expansion("affordability", relation, ("rent",)),),
            ),
            Facet("zoning-effects", ("zoning",), ()),
        )
    )

    with pytest.raises(FacetPlanValidationError, match="unsupported expansion relation"):
        validate_facet_plan(_housing_topic(), plan)


def test_validation_rejects_an_expansion_without_anchor_provenance() -> None:
    plan = _housing_plan(
        facets=(
            Facet(
                "rent-effects",
                ("rent",),
                (Expansion("affordability", "alias", ()),),
            ),
            Facet("zoning-effects", ("zoning",), ()),
        )
    )

    with pytest.raises(FacetPlanValidationError, match="at least one anchor"):
        validate_facet_plan(_housing_topic(), plan)


@pytest.mark.parametrize(
    ("expansions", "description"),
    [
        ((Expansion("neighborhood", "neutral_search_term", ("zoning",)),), "inherited"),
        ((Expansion("tenant burden", "synonym", ("rent",)),), "relation"),
        ((Expansion("tenant\nburden", "neutral_search_term", ("rent",)),), "control"),
        ((Expansion("title:rent", "neutral_search_term", ("rent",)),), "field"),
        ((Expansion("rent AND zoning", "neutral_search_term", ("rent",)),), "operator"),
        ((Expansion("rent || zoning", "neutral_search_term", ("rent",)),), "operator"),
        ((Expansion("rent 2026", "neutral_search_term", ("rent",)),), "numeric"),
        ((Expansion("rent ٢٠٢٦", "neutral_search_term", ("rent",)),), "numeric"),
        ((Expansion("one two three four", "neutral_search_term", ("rent",)),), "analyzed words"),
        (
            (
                Expansion("renters facing eviction", "neutral_search_term", ("rent",)),
                Expansion("legal aid support", "neutral_search_term", ("rent",)),
                Expansion("subsidy", "neutral_search_term", ("rent",)),
            ),
            "new unique content tokens",
        ),
        (
            (
                Expansion("tenant", "neutral_search_term", ("rent",)),
                Expansion("burden", "neutral_search_term", ("rent",)),
                Expansion("affordability", "neutral_search_term", ("rent",)),
                Expansion("lease", "neutral_search_term", ("rent",)),
            ),
            "too many expansions",
        ),
    ],
)
def test_validation_rejects_unsafe_expansions(
    expansions: tuple[Expansion, ...], description: str
) -> None:
    plan = _housing_plan(
        facets=(
            Facet("rent-effects", ("rent",), expansions),
            Facet("zoning-effects", ("zoning",), ()),
        )
    )

    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(_housing_topic(), plan)


def test_rendering_discards_every_facet_and_falls_back_when_a_late_facet_is_invalid() -> None:
    topic = _housing_topic()
    plan = _housing_plan(
        facets=(
            Facet(
                "rent-effects",
                ("rent",),
                (Expansion("affordability", "neutral_search_term", ("rent",)),),
            ),
            Facet(
                "zoning-effects",
                ("zoning",),
                (Expansion("zoning AND permits", "neutral_search_term", ("zoning",)),),
            ),
        )
    )

    result = render_facet_queries(topic, plan)

    assert result.queries == (
        QueryVariant(
            topic_id=topic.id,
            variant_name="original",
            query_text=topic.narrative,
            source_type="original_topic",
        ),
    )
    assert result.used_fallback is True
    assert result.error == "expansion term contains a query operator"


def test_token_tape_preserves_unicode_offsets_and_half_open_range_text() -> None:
    tape = tokenize_narrative("Café\towners\ncompare  rent")

    assert [(token.text, token.start_offset, token.end_offset) for token in tape.tokens] == [
        ("Café", 0, 4),
        ("owners", 5, 11),
        ("compare", 12, 19),
        ("rent", 21, 25),
    ]
    assert tape.resolve(TokenRange(1, 4)) == "owners\ncompare  rent"


def test_records_are_frozen_and_validation_accepts_a_complete_plan() -> None:
    plan = _plan()

    with pytest.raises(FrozenInstanceError):
        plan.topic_id = "other"  # type: ignore[misc]

    validated = validate_facet_plan(_topic(), plan)

    assert validated.plan == plan


@pytest.mark.parametrize(
    ("source_ranges", "description"),
    [
        ((TokenRange(-1, 1),), "negative"),
        ((TokenRange(2, 2),), "empty"),
        ((TokenRange(4, 2),), "reversed"),
        ((TokenRange(3, 5), TokenRange(3, 5)), "duplicate"),
        ((TokenRange(3, 9),), "out-of-bounds"),
    ],
)
def test_validation_rejects_invalid_coverage_ranges(
    source_ranges: tuple[TokenRange, ...], description: str
) -> None:
    plan = _plan(
        coverage_items=(
            CoverageItem(coverage_id="rent", source_ranges=source_ranges),
        )
    )

    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(_topic(), plan)


@pytest.mark.parametrize(
    "token_range",
    [
        TokenRange("0", 1),  # type: ignore[arg-type]
        TokenRange(0, "1"),  # type: ignore[arg-type]
        TokenRange(0.0, 1),  # type: ignore[arg-type]
        TokenRange(0, 1.0),  # type: ignore[arg-type]
        TokenRange(False, 1),
        TokenRange(0, True),
    ],
)
def test_validation_rejects_non_integer_token_range_indices(
    token_range: TokenRange,
) -> None:
    plan = _plan(
        coverage_items=(
            CoverageItem(coverage_id="rent", source_ranges=(token_range,)),
        )
    )

    with pytest.raises(FacetPlanValidationError, match="integer token indices"):
        validate_facet_plan(_topic(), plan)


@pytest.mark.parametrize(
    ("topic", "plan", "description"),
    [
        (_topic(topic_id="other"), _plan(), "topic ID"),
        (
            _topic(),
            replace(_plan(), narrative_sha256="0" * 64),
            "SHA-256",
        ),
    ],
)
def test_validation_rejects_topic_identity_mismatches(
    topic: Topic, plan: FacetPlan, description: str
) -> None:
    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(topic, plan)


@pytest.mark.parametrize(
    ("plan", "description"),
    [
        (
            _plan(
                anchors=(
                    Anchor("topic", TokenRange(0, 1), "topic", "global", ()),
                    Anchor("topic", TokenRange(1, 2), "entity", "global", ()),
                )
            ),
            "duplicate anchor ID",
        ),
        (
            _plan(
                coverage_items=(
                    CoverageItem("rent", (TokenRange(3, 5),)),
                    CoverageItem("rent", (TokenRange(5, 7),)),
                )
            ),
            "duplicate coverage ID",
        ),
        (
            _plan(
                facets=(
                    Facet("rent-effects", ("rent",), ()),
                    Facet("rent-effects", ("rent",), ()),
                )
            ),
            "duplicate facet ID",
        ),
        (
            _plan(
                facets=(Facet("rent-effects", ("missing",), ()),),
            ),
            "unknown coverage reference",
        ),
        (
            _plan(
                anchors=(
                    Anchor("topic", TokenRange(0, 2), "topic", "global", ()),
                    Anchor(
                        "rent-anchor",
                        TokenRange(3, 5),
                        "aspect",
                        "coverage",
                        ("missing",),
                    ),
                )
            ),
            "unknown coverage reference",
        ),
    ],
)
def test_validation_rejects_duplicate_ids_and_dangling_references(
    plan: FacetPlan, description: str
) -> None:
    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(_topic(), plan)


@pytest.mark.parametrize(
    ("anchors", "description"),
    [
        (
            (
                Anchor("topic", TokenRange(0, 2), "topic", "global", ("rent",)),
            ),
            "global anchor",
        ),
        (
            (
                Anchor("topic", TokenRange(0, 2), "topic", "global", ()),
                Anchor("rent", TokenRange(3, 5), "aspect", "coverage", ()),
            ),
            "coverage anchor",
        ),
        (
            (
                Anchor("topic", TokenRange(0, 2), "topic", "other", ()),
            ),
            "anchor scope",
        ),
        (
            (
                Anchor("relation", TokenRange(0, 2), "aspect", "global", ()),
            ),
            "topic or entity",
        ),
    ],
)
def test_validation_rejects_invalid_anchor_scopes_and_missing_global_subject(
    anchors: tuple[Anchor, ...], description: str
) -> None:
    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(_topic(), _plan(anchors=anchors))


@pytest.mark.parametrize(
    ("facets", "description"),
    [
        ((), "at least one facet"),
        ((Facet("empty", (), ()),), "must reference coverage"),
        (
            (
                Facet("rent-one", ("rent",), ()),
                Facet("rent-two", ("rent",), ()),
            ),
            "exactly one facet",
        ),
    ],
)
def test_validation_requires_a_nonempty_exact_coverage_partition(
    facets: tuple[Facet, ...], description: str
) -> None:
    with pytest.raises(FacetPlanValidationError, match=description):
        validate_facet_plan(_topic(), _plan(facets=facets))


def test_validation_rejects_coverage_assigned_to_zero_facets() -> None:
    plan = _plan(
        coverage_items=(
            CoverageItem("rent", (TokenRange(3, 5),)),
            CoverageItem("zoning", (TokenRange(5, 7),)),
        ),
        facets=(Facet("rent-only", ("rent",), ()),),
    )

    with pytest.raises(FacetPlanValidationError, match="exactly one facet"):
        validate_facet_plan(_topic(), plan)


@pytest.mark.parametrize(
    "plan",
    [
        _plan(coverage_items=("not-a-coverage-item",)),  # type: ignore[arg-type]
        _plan(anchors=("not-an-anchor",)),  # type: ignore[arg-type]
        _plan(facets=("not-a-facet",)),  # type: ignore[arg-type]
    ],
)
def test_validation_reports_malformed_nested_records_as_plan_errors(plan: FacetPlan) -> None:
    with pytest.raises(FacetPlanValidationError):
        validate_facet_plan(_topic(), plan)


def test_validation_rejects_an_expansion_with_a_dangling_anchor_reference() -> None:
    plan = _plan(
        facets=(
            Facet(
                facet_id="rent-effects",
                coverage_refs=("rent",),
                expansions=(
                    Expansion(
                        term="rental market",
                        relation="neutral_search_term",
                        anchor_refs=("missing-anchor",),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(FacetPlanValidationError, match="unknown anchor reference"):
        validate_facet_plan(_topic(), plan)

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
    tokenize_narrative,
    validate_facet_plan,
)
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
                        relation="related",
                        anchor_refs=("rent-anchor",),
                    ),
                ),
            ),
        ),
    )


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
                        relation="related",
                        anchor_refs=("missing-anchor",),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(FacetPlanValidationError, match="unknown anchor reference"):
        validate_facet_plan(_topic(), plan)

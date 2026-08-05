from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

import trec_rag.facet_evidence as facet_evidence
from trec_rag.facet_evidence import SCORING_NORMALIZATION_VERSION


ScoringView = getattr(facet_evidence, "ScoringView", None)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _byte_offsets(source: str) -> tuple[int, ...]:
    return tuple(
        len(source[:index].encode("utf-8")) for index in range(len(source) + 1)
    )


def _view(source: str):
    assert ScoringView is not None, "ScoringView is not implemented"
    return ScoringView(source)


def test_scoring_view_binds_exact_source_hashes_and_canonical_boundaries() -> None:
    source = "  Café\tBeta\r\nGamma\u2003Δelta  "

    view = _view(source)

    assert view.source == source
    assert view.source_sha256 == _digest(source)
    assert view.normalization_version == SCORING_NORMALIZATION_VERSION
    assert view.scoring_text == "Café Beta Gamma Δelta"
    assert view.scoring_text_sha256 == _digest(view.scoring_text)
    assert view.scoring_boundaries == (
        source.index("C"),
        source.index("C") + 1,
        source.index("C") + 2,
        source.index("C") + 3,
        source.index("C") + 4,
        source.index("B"),
        source.index("B") + 1,
        source.index("B") + 2,
        source.index("B") + 3,
        source.index("B") + 4,
        source.index("G"),
        source.index("G") + 1,
        source.index("G") + 2,
        source.index("G") + 3,
        source.index("G") + 4,
        source.index("G") + 5,
        source.index("Δ"),
        source.index("Δ") + 1,
        source.index("Δ") + 2,
        source.index("Δ") + 3,
        source.index("Δ") + 4,
        source.index("Δ") + 5,
    )


def test_scoring_view_projects_edge_and_interior_spans_to_exact_chars_and_bytes() -> None:
    source = "\tCafé  \tΩmega\r\n第三段\u2003終わり  "
    view = _view(source)
    offsets = _byte_offsets(source)

    first = view.project(0, len("Café"))
    interior_start = view.scoring_text.index("Ωmega")
    interior = view.project(interior_start, interior_start + len("Ωmega"))
    whole = view.project(0, len(view.scoring_text))

    assert isinstance(first, facet_evidence.ScoringSpanProjection)
    assert first.scoring_start_char == 0
    assert first.scoring_end_char == len("Café")
    assert first.source_start_char == source.index("C")
    assert first.source_end_char == source.index("C") + len("Café")
    assert first.source_start_byte == offsets[first.source_start_char]
    assert first.source_end_byte == offsets[first.source_end_char]
    assert first.source_text == "Café"

    assert interior.scoring_start_char == interior_start
    assert interior.scoring_end_char == interior_start + len("Ωmega")
    assert interior.source_text == "Ωmega"
    assert interior.source_start_byte == len(
        source[: source.index("Ω")].encode("utf-8")
    )
    assert interior.source_end_byte == len(
        source[: source.index("Ω") + len("Ωmega")].encode("utf-8")
    )

    assert whole.source_start_char == source.index("C")
    assert whole.source_end_char == len(source) - 2
    assert whole.source_text == source[whole.source_start_char : whole.source_end_char]
    assert source.encode("utf-8")[whole.source_start_byte : whole.source_end_byte].decode(
        "utf-8"
    ) == whole.source_text


def test_scoring_view_preserves_coordinate_identity_for_duplicate_normalized_sequences() -> None:
    source = "same  token same\t token"
    view = _view(source)
    first_end = len("same token")
    second_start = len("same token ")

    first = view.project(0, first_end)
    second = view.project(second_start, len(view.scoring_text))

    assert first.source_text == "same  token"
    assert second.source_text == "same\t token"
    assert first.scoring_start_char == 0
    assert second.scoring_start_char == second_start
    assert (first.source_start_char, first.source_end_char) != (
        second.source_start_char,
        second.source_end_char,
    )


def test_scoring_view_can_project_a_nonempty_collapsed_whitespace_span() -> None:
    source = "left\t\r\nright"
    view = _view(source)

    projection = view.project(len("left"), len("left "))

    assert projection.source_text == "\t\r\n"
    assert projection.source_start_char == source.index("\t")
    assert projection.source_end_char == source.index("r")
    assert projection.source_text != view.scoring_text[
        projection.scoring_start_char : projection.scoring_end_char
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    (
        (True, 1),
        (0, True),
        (False, 1),
        (0, False),
        (-1, 1),
        (0, 0),
        (2, 1),
        (0, 1_000),
        (1_000, 1_001),
        (0.0, 1),
        (0, 1.0),
        ("0", 1),
    ),
)
def test_scoring_view_rejects_invalid_scoring_spans(start: object, end: object) -> None:
    view = _view("one two")

    with pytest.raises(ValueError, match="scoring span"):
        view.project(start, end)


@pytest.mark.parametrize("source", ("", " \t\r\n\u2003"))
def test_scoring_view_makes_whitespace_only_sources_safe(source: str) -> None:
    view = _view(source)

    assert view.source == source
    assert view.source_sha256 == _digest(source)
    assert view.scoring_text == ""
    assert view.scoring_text_sha256 == _digest("")
    assert view.scoring_boundaries == (0,)

    with pytest.raises(ValueError, match="scoring span"):
        view.project(0, 1)


def test_scoring_view_is_immutable_and_rejects_tampered_derived_state() -> None:
    view = _view("one two")

    with pytest.raises(FrozenInstanceError):
        view.scoring_text = "tampered"  # type: ignore[misc]

    object.__setattr__(view, "scoring_boundaries", (0, 1))
    with pytest.raises(ValueError, match="scoring view"):
        view.project(0, 1)


def test_scoring_span_projection_rejects_inconsistent_character_offsets() -> None:
    with pytest.raises(ValueError, match="character offsets"):
        facet_evidence.ScoringSpanProjection(
            scoring_start_char=0,
            scoring_end_char=1,
            source_start_char=4,
            source_end_char=7,
            source_start_byte=4,
            source_end_byte=5,
            source_text="x",
        )

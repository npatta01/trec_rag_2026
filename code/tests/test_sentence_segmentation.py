"""Characterization of the shared sentence segmenter.

These pin the behaviour both retrieval lanes depend on: exact source slices,
non-overlapping spans inside their paragraph, and the boundary decisions that
previously shredded evidence into headings and half-sentences.
"""

from __future__ import annotations

import pytest

from trec_rag.chunking import Sentence, sentence_segmenter
from trec_rag.deepagent_evidence import _sentence_spans
from trec_rag.facet_evidence import (
    MAX_PARAGRAPH_CHARACTERS,
    _byte_offsets,
    _segmented_source,
    _sentences_in_paragraph,
    _source_spans,
)


def _texts(source: str) -> list[str]:
    return [source[row.start_char : row.end_char] for row in sentence_segmenter().segment(source)]


# The boundary decisions a hand-rolled splitter kept getting wrong. Each of
# these produced a junk evidence candidate in the 2025 non-agentic run.
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("The U.S. entered the war in 1950. Truman acted quickly.",
         ["The U.S. entered the war in 1950.", "Truman acted quickly."]),
        ("Plyler v. Doe concerned schooling. It was decided in 1982.",
         ["Plyler v. Doe concerned schooling.", "It was decided in 1982."]),
        ("Costs include rent, food, etc. Wages have not kept pace.",
         ["Costs include rent, food, etc.", "Wages have not kept pace."]),
        ("Prices rose 20.5 percent last year. That is a lot.",
         ["Prices rose 20.5 percent last year.", "That is a lot."]),
        ("Dr. J. R. Smith wrote it. He was clear.",
         ["Dr. J. R. Smith wrote it.", "He was clear."]),
    ],
)
def test_segmenter_keeps_abbreviations_and_initials_inside_their_sentence(
    source: str, expected: list[str]
) -> None:
    assert _texts(source) == expected


def test_segmenter_keeps_a_sentence_wrapped_across_two_lines_whole() -> None:
    """A hard line wrap is not a sentence boundary; it once truncated evidence."""
    source = "The rental market has surged\nover the past two years. Prices rose."

    assert _texts(source) == [
        "The rental market has surged\nover the past two years.",
        "Prices rose.",
    ]


def test_blank_line_splits_even_when_the_model_finds_no_boundary() -> None:
    """spaCy reads 'Gamma.\\n\\nDelta.' as one sentence; the blank line still wins.

    Without this a missed boundary silently merges two unrelated blocks into one
    candidate, and the paragraph derived from it spans both.
    """
    assert _texts("Gamma.\n\nDelta.") == ["Gamma.", "Delta."]


def test_headings_are_marked_as_not_standing_alone() -> None:
    source = "Grocery Costs:\n\nIn Dubai, groceries are reasonably priced."
    rows = sentence_segmenter().segment(source)

    assert [row.is_complete for row in rows] == [False, True]
    assert source[rows[0].start_char : rows[0].end_char] == "Grocery Costs:"


def test_paragraphs_group_sentences_and_break_on_blank_lines() -> None:
    source = "Housing Costs:\nThe market surged\nover two years. Prices rose.\n\nTail sentence."
    offsets = _byte_offsets(source)

    paragraphs = _source_spans(source, offsets)

    assert [span.text for span in paragraphs] == [
        "Housing Costs:\nThe market surged\nover two years. Prices rose.",
        "Tail sentence.",
    ]


def test_every_sentence_is_an_exact_non_overlapping_slice_inside_its_paragraph() -> None:
    source = (
        "Cost of Living in Dubai\nHousing Costs:\nThe rental market has surged\n"
        "over two years. Prices rose 20%.\n\nGrocery Costs:\n\nGroceries are "
        "reasonable. However, imports cost more.\n"
    )
    offsets = _byte_offsets(source)

    for paragraph in _source_spans(source, offsets):
        assert source[paragraph.start_char : paragraph.end_char] == paragraph.text
        assert paragraph.end_byte - paragraph.start_byte == len(paragraph.text.encode("utf-8"))
        previous_end = paragraph.start_char
        for sentence in _sentences_in_paragraph(source, paragraph, offsets):
            assert source[sentence.start_char : sentence.end_char] == sentence.text
            assert sentence.end_byte - sentence.start_byte == len(sentence.text.encode("utf-8"))
            assert paragraph.start_char <= sentence.start_char
            assert sentence.end_char <= paragraph.end_char
            assert sentence.start_char >= previous_end
            previous_end = sentence.end_char


def test_offsets_stay_exact_across_multibyte_source() -> None:
    source = "Café\tΩmega.\r\nSecond line?  \n\n第三段.\t終わり!"
    offsets = _byte_offsets(source)

    for paragraph in _source_spans(source, offsets):
        assert source[paragraph.start_char : paragraph.end_char] == paragraph.text
        assert paragraph.end_byte - paragraph.start_byte == len(paragraph.text.encode("utf-8"))


def test_a_paragraph_exceeds_the_ceiling_only_to_keep_one_sentence_whole() -> None:
    """Blank-line-free source must not collapse into one giant paragraph.

    The ceiling is enforced between sentences, so the one case that may exceed
    it is a single sentence longer than the ceiling: splitting that would cut a
    sentence in half, which the exact-span contract does not allow.
    """
    source = ". ".join(f"Line {index} carries real words" for index in range(4_000))

    groups = _segmented_source(source)

    assert len(groups) > 1
    for (start, end), sentences in groups:
        assert end - start <= MAX_PARAGRAPH_CHARACTERS or len(sentences) == 1


def test_segmentation_is_stable_across_repeated_calls() -> None:
    source = "Housing Costs:\nThe market surged. Prices rose.\n\nTail."

    assert _segmented_source(source) == _segmented_source(source)


def test_deepagent_joins_a_heading_to_the_sentence_it_introduces() -> None:
    """The lane that once produced a citable span reading only 'Plyler v.'."""
    text = "Grocery Costs:\n\nGroceries are reasonably priced. Imports cost more."

    spans = _sentence_spans(text)

    assert [text[start:end] for start, end in spans] == [
        "Grocery Costs:\n\nGroceries are reasonably priced.",
        "Imports cost more.",
    ]


def test_deepagent_never_returns_an_empty_span_set() -> None:
    for text in ("", "   ", "\n\n"):
        assert _sentence_spans(text) == ((0, len(text)),)


def test_both_lanes_agree_on_sentence_boundaries() -> None:
    """One segmenter, so a citation resolved in either lane covers the same text."""
    source = "The U.S. acted in 1950. Truman was clear. Eisenhower disagreed."
    offsets = _byte_offsets(source)

    competition = [
        span.text
        for paragraph in _source_spans(source, offsets)
        for span in _sentences_in_paragraph(source, paragraph, offsets)
    ]
    deepagent = [source[start:end] for start, end in _sentence_spans(source)]

    assert competition == deepagent


def test_sentence_rejects_an_empty_range() -> None:
    with pytest.raises(ValueError):
        Sentence(start_char=5, end_char=5, is_complete=True)


def test_an_unpunctuated_clause_with_a_finite_verb_stands_alone() -> None:
    """Completeness must consult POS, not punctuation alone.

    Excluding spaCy's attribute_ruler leaves ``token.pos_`` empty on every
    token, which silently reduced this to a punctuation check and refused real
    sentences as if they were headings.
    """
    rows = sentence_segmenter().segment("The committee approved the new safety rules")

    assert [row.is_complete for row in rows] == [True]


def test_a_noun_phrase_heading_does_not_stand_alone() -> None:
    assert [row.is_complete for row in sentence_segmenter().segment("Vaccine Safety Data")] == [False]


@pytest.mark.parametrize(
    "heading",
    [
        "How to Apply",
        "Getting Started",
        "Learn More",
        "Ways to Save",
        "Frequently Asked Questions",
    ],
)
def test_an_unpunctuated_nonfinite_heading_does_not_stand_alone(heading: str) -> None:
    assert [row.is_complete for row in sentence_segmenter().segment(heading)] == [False]


def test_a_document_past_spacy_max_length_still_segments() -> None:
    """One oversized document must not abort a topic's candidate generation."""
    block = "The committee approved the rules. Prices rose sharply that year.\n\n"
    source = block * 20_000  # ~1.3M characters, past spaCy's 1,000,000 ceiling

    rows = sentence_segmenter().segment(source)

    assert len(rows) > 1
    assert all(source[row.start_char : row.end_char].strip() for row in rows)
    previous_end = 0
    for row in rows:
        assert row.start_char >= previous_end
        previous_end = row.end_char


def test_bounded_blocks_tile_the_source_without_dropping_text() -> None:
    from trec_rag.chunking import _bounded_blocks

    source = "".join(f"Sentence {index} here.\n\n" for index in range(500))

    blocks = _bounded_blocks(source, 200)

    assert blocks[0][0] == 0
    assert blocks[-1][1] == len(source)
    for (_, end), (next_start, _) in zip(blocks, blocks[1:]):
        assert end == next_start


def test_the_recorded_splitter_version_matches_the_live_segmenter() -> None:
    """Fail closed when the model or pipeline changes without a version bump.

    The version is compared on resume to decide whether cached candidates and
    their ~383k sentence scores may be reused. If it drifts from the segmenter
    that actually produced them, scores are reused against different text.
    """
    from trec_rag.facet_evidence import SENTENCE_SPLITTER_VERSION

    identity = sentence_segmenter().identity

    assert identity["backend"] in SENTENCE_SPLITTER_VERSION
    assert str(identity["model"]) in SENTENCE_SPLITTER_VERSION
    assert str(identity["model_version"]) in SENTENCE_SPLITTER_VERSION
    assert sorted(identity["disabled"]) == ["lemmatizer", "ner"], (
        "changing the enabled pipeline changes sentence boundaries and "
        "completeness; bump SENTENCE_SPLITTER_VERSION with it"
    )

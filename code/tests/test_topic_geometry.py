from __future__ import annotations

from hashlib import sha256

import pytest

from trec_rag.facet_evidence import SourceSpan
from trec_rag.topic_geometry import (
    DigestMismatchError,
    DocumentGeometryIndex,
    GeometryConflictError,
)


def _digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def test_geometry_preserves_unicode_offsets_and_scoring_boundaries() -> None:
    source = "Café\tΩmega.\r\nSecond line?  \n\n第三段.\t終わり!"
    index = DocumentGeometryIndex()

    geometry = index.admit(_digest(source), source)

    assert geometry.content_sha256 == _digest(source)
    assert geometry.source == source
    assert geometry.byte_offsets == tuple(
        len(source[:character].encode("utf-8"))
        for character in range(len(source) + 1)
    )
    assert geometry.byte_offsets[source.index("Ω")] == len("Café\t".encode("utf-8"))
    assert geometry.byte_offsets[source.index("終わり") + len("終わり")] == len(
        "Café\tΩmega.\r\nSecond line?  \n\n第三段.\t終わり".encode("utf-8")
    )

    assert geometry.scoring_text == "Café Ωmega. Second line? 第三段. 終わり!"
    assert geometry.scoring_text_sha256 == _digest(geometry.scoring_text)
    assert len(geometry.scoring_boundaries) == len(geometry.scoring_text) + 1
    assert geometry.scoring_boundaries[4] == 4
    assert geometry.scoring_boundaries[5] == source.index("Ω")
    assert geometry.scoring_boundaries[10] == source.index(".")
    assert geometry.scoring_boundaries[12] == source.index("Second")
    assert geometry.scoring_boundaries[-1] == len(source)

    assert tuple(span.text for span in geometry.paragraphs) == (
        "Café\tΩmega.",
        "Second line?  ",
        "第三段.\t終わり!",
    )
    assert all(
        source[span.start_char : span.end_char] == span.text
        and span.end_byte - span.start_byte
        == len(span.text.encode("utf-8"))
        for span in geometry.paragraphs
    )


def test_geometry_provides_constant_time_membership_and_adjacency_lookups() -> None:
    source = "First sentence. Second sentence.\nThird sentence!\nFourth."
    geometry = DocumentGeometryIndex().admit(_digest(source), source)

    first, second, third = geometry.paragraphs
    assert geometry.paragraph_ordinal((first.start_char, first.end_char)) == 0
    assert geometry.paragraph_ordinal((second.start_char, second.end_char)) == 1
    assert geometry.paragraph_neighbor((first.start_char, first.end_char), 1) == second
    assert geometry.paragraph_neighbor((second.start_char, second.end_char), -1) == first
    assert geometry.paragraph_neighbor((third.start_char, third.end_char), 1) is None
    forged_paragraph = SourceSpan(
        first.text + " forged",
        first.start_char,
        first.end_char,
        first.start_byte,
        first.end_byte,
        first.text_sha256,
    )
    assert geometry.paragraph_ordinal(forged_paragraph) is None

    first_sentences = geometry.sentences_for((first.start_char, first.end_char))
    assert tuple(sentence.text for sentence in first_sentences) == (
        "First sentence.",
        "Second sentence.",
    )
    first_sentence, second_sentence = first_sentences
    assert geometry.sentence_ordinal(
        (first.start_char, first.end_char),
        (first_sentence.start_char, first_sentence.end_char),
    ) == 0
    assert geometry.sentence_ordinal(
        (first.start_char, first.end_char),
        (second_sentence.start_char, second_sentence.end_char),
    ) == 1
    assert geometry.sentence_neighbor(
        (first.start_char, first.end_char),
        (first_sentence.start_char, first_sentence.end_char),
        1,
    ) == second_sentence
    assert geometry.sentence_neighbor(
        (first.start_char, first.end_char),
        (second_sentence.start_char, second_sentence.end_char),
        1,
    ) is None
    forged_sentence = SourceSpan(
        first_sentence.text,
        first_sentence.start_char,
        first_sentence.end_char,
        first_sentence.start_byte,
        first_sentence.end_byte,
        "0" * 64,
    )
    assert geometry.sentence_ordinal(first, forged_sentence) is None

    assert geometry.paragraph_index[(first.start_char, first.end_char)] == 0
    assert geometry.sentence_indices_by_paragraph[(first.start_char, first.end_char)][
        (second_sentence.start_char, second_sentence.end_char)
    ] == 1


def test_index_derives_two_documents_once_each_and_reuses_exact_sources() -> None:
    first_source = "Alpha.\nBeta."
    second_source = "Gamma.\nDelta."
    index = DocumentGeometryIndex()

    first = index.admit(_digest(first_source), first_source)
    assert index.admit(_digest(first_source), first_source) is first
    second = index.admit(_digest(second_source), second_source)
    assert index.admit(_digest(second_source), second_source) is second

    counters = index.counters
    assert len(index) == 2
    assert counters.documents_indexed == 2
    assert counters.geometry_builds == 2
    assert counters.cache_hits == 2
    assert counters.paragraph_spans_built == 4
    assert counters.sentence_spans_built == 4
    assert index.get(_digest(first_source)) is first
    assert index.get(_digest(second_source)) is second


def test_index_invokes_canonical_derivation_helpers_once_per_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.facet_evidence as facet_evidence
    import trec_rag.topic_geometry as topic_geometry

    counts = {"bytes": 0, "scoring": 0, "paragraphs": 0, "sentences": 0}
    original_bytes = facet_evidence._byte_offsets
    original_scoring = topic_geometry._scoring_text_and_boundaries
    original_paragraphs = facet_evidence._source_spans
    original_sentences = topic_geometry._sentences_in_paragraph

    def byte_offsets(source: str):
        counts["bytes"] += 1
        return original_bytes(source)

    def scoring(source: str):
        counts["scoring"] += 1
        return original_scoring(source)

    def paragraphs(source: str, offsets: tuple[int, ...]):
        counts["paragraphs"] += 1
        return original_paragraphs(source, offsets)

    def sentences(source: str, paragraph: SourceSpan, offsets: tuple[int, ...]):
        counts["sentences"] += 1
        return original_sentences(source, paragraph, offsets)

    monkeypatch.setattr(facet_evidence, "_byte_offsets", byte_offsets)
    monkeypatch.setattr(topic_geometry, "_scoring_text_and_boundaries", scoring)
    monkeypatch.setattr(facet_evidence, "_source_spans", paragraphs)
    monkeypatch.setattr(topic_geometry, "_sentences_in_paragraph", sentences)

    sources = ("Alpha.\nBeta.", "Gamma.\nDelta.")
    index = DocumentGeometryIndex()
    for source in sources:
        digest = _digest(source)
        first = index.admit(digest, source)
        assert index.admit(digest, source) is first

    assert counts == {
        "bytes": 2,
        "scoring": 2,
        "paragraphs": 2,
        "sentences": 4,
    }


def test_index_rejects_digest_mismatch_conflict_and_invalid_lookups() -> None:
    source = "Exact source."
    digest = _digest(source)
    index = DocumentGeometryIndex()

    with pytest.raises(DigestMismatchError):
        index.admit("0" * 64, source)
    with pytest.raises(DigestMismatchError):
        index.admit(digest.upper(), source)

    index.admit(digest, source)
    with pytest.raises(GeometryConflictError):
        index.admit(digest, "Different source.")
    with pytest.raises(ValueError):
        index.get("not-a-digest")

    geometry = index.get(digest)
    assert geometry is not None
    with pytest.raises(ValueError):
        geometry.sentences_for((999, 1000))
    assert geometry.paragraph_ordinal((999, 1000)) is None
    assert geometry.sentence_ordinal((0, 1), (0, 1)) is None


def test_geometry_rejects_duplicate_derived_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_rag.facet_evidence as facet_evidence
    import trec_rag.topic_geometry as topic_geometry

    source = "One.\nTwo."
    original = facet_evidence._source_spans

    def duplicate_spans(value: str, byte_offsets: tuple[int, ...]) -> tuple[SourceSpan, ...]:
        spans = original(value, byte_offsets)
        return spans + (spans[0],)

    monkeypatch.setattr(facet_evidence, "_source_spans", duplicate_spans)

    with pytest.raises(ValueError, match="duplicate paragraph"):
        DocumentGeometryIndex().admit(_digest(source), source)

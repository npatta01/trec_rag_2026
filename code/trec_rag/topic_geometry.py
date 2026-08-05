"""Content-addressed exact source geometry for document validation.

The module owns only immutable, in-process geometry.  It deliberately does not
know about topics, candidates, persistence, models, or network services.  All
text parsing delegates to the canonical helpers in :mod:`facet_evidence` so
that validation and extraction share the same byte, whitespace, paragraph, and
sentence semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from re import fullmatch
from types import MappingProxyType
from typing import TypeAlias

from .facet_evidence import (
    SourceSpan,
    _scoring_text_and_boundaries,
    _sentences_in_paragraph,
    _source_validation_cache,
)


SpanKey: TypeAlias = tuple[int, int]
_DIGEST_PATTERN = r"[0-9a-f]{64}"


class TopicGeometryError(ValueError):
    """Base error for invalid or inconsistent geometry input."""


class DigestMismatchError(TopicGeometryError):
    """The declared content digest does not identify the supplied source."""


class GeometryConflictError(TopicGeometryError):
    """A digest already identifies different source text in one index."""


class GeometryInvariantError(TopicGeometryError):
    """Canonical geometry violated an exact-source invariant."""


@dataclass(frozen=True)
class GeometryCounters:
    """Small observable counters for derivation and reuse tests."""

    documents_indexed: int = 0
    geometry_builds: int = 0
    cache_hits: int = 0
    paragraph_spans_built: int = 0
    sentence_spans_built: int = 0


@dataclass(frozen=True)
class DocumentGeometry:
    """All exact derived coordinates for one content-addressed source."""

    content_sha256: str
    source: str
    byte_offsets: tuple[int, ...]
    scoring_text: str
    scoring_boundaries: tuple[int, ...]
    scoring_text_sha256: str
    paragraphs: tuple[SourceSpan, ...]
    paragraph_index: Mapping[SpanKey, int]
    sentences_by_paragraph: Mapping[SpanKey, tuple[SourceSpan, ...]]
    sentence_indices_by_paragraph: Mapping[SpanKey, Mapping[SpanKey, int]]

    @property
    def character_count(self) -> int:
        return len(self.source)

    @property
    def byte_count(self) -> int:
        return self.byte_offsets[-1]

    def paragraph_ordinal(self, span: SourceSpan | SpanKey) -> int | None:
        """Return the exact paragraph ordinal, or ``None`` if not a member."""
        ordinal = self.paragraph_index.get(_span_key(span))
        if (
            ordinal is not None
            and isinstance(span, SourceSpan)
            and self.paragraphs[ordinal] != span
        ):
            return None
        return ordinal

    def sentences_for(self, paragraph: SourceSpan | SpanKey) -> tuple[SourceSpan, ...]:
        """Return all exact sentences for a known paragraph in source order."""
        key = _span_key(paragraph)
        try:
            return self.sentences_by_paragraph[key]
        except KeyError as exc:
            raise TopicGeometryError("paragraph span is not in this document") from exc

    def sentence_ordinal(
        self,
        paragraph: SourceSpan | SpanKey,
        sentence: SourceSpan | SpanKey,
    ) -> int | None:
        """Return an exact sentence ordinal within a paragraph in O(1)."""
        paragraph_key = _span_key(paragraph)
        sentence_key = _span_key(sentence)
        sentence_index = self.sentence_indices_by_paragraph.get(paragraph_key)
        if sentence_index is None:
            return None
        ordinal = sentence_index.get(sentence_key)
        if ordinal is None:
            return None
        if isinstance(paragraph, SourceSpan):
            paragraph_ordinal = self.paragraph_index.get(paragraph_key)
            if (
                paragraph_ordinal is None
                or self.paragraphs[paragraph_ordinal] != paragraph
            ):
                return None
        if (
            isinstance(sentence, SourceSpan)
            and self.sentences_by_paragraph[paragraph_key][ordinal] != sentence
        ):
            return None
        return ordinal

    def paragraph_neighbor(
        self,
        paragraph: SourceSpan | SpanKey,
        offset: int,
    ) -> SourceSpan | None:
        """Return a paragraph at a relative ordinal, or ``None`` at an edge."""
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TopicGeometryError("paragraph adjacency offset must be an integer")
        ordinal = self.paragraph_ordinal(paragraph)
        if ordinal is None:
            return None
        target = ordinal + offset
        if target < 0 or target >= len(self.paragraphs):
            return None
        return self.paragraphs[target]

    def sentence_neighbor(
        self,
        paragraph: SourceSpan | SpanKey,
        sentence: SourceSpan | SpanKey,
        offset: int,
    ) -> SourceSpan | None:
        """Return a sentence at a relative ordinal, or ``None`` at an edge."""
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TopicGeometryError("sentence adjacency offset must be an integer")
        paragraph_key = _span_key(paragraph)
        sentence_ordinal = self.sentence_ordinal(paragraph_key, sentence)
        if sentence_ordinal is None:
            return None
        sentences = self.sentences_by_paragraph.get(paragraph_key)
        if sentences is None:
            return None
        target = sentence_ordinal + offset
        if target < 0 or target >= len(sentences):
            return None
        return sentences[target]


class DocumentGeometryIndex:
    """Unbounded in-memory geometry index keyed by full source digest.

    ``admit`` is the only derivation entry point.  A repeated digest/source
    returns the same immutable object, while a digest collision or a digest
    mismatch is rejected before any inconsistent object can enter the index.
    """

    def __init__(self) -> None:
        self._by_digest: dict[str, DocumentGeometry] = {}
        self._counters = GeometryCounters()

    def __len__(self) -> int:
        return len(self._by_digest)

    @property
    def counters(self) -> GeometryCounters:
        """Return a snapshot of derivation and reuse counters."""
        return self._counters

    def get(self, content_sha256: str) -> DocumentGeometry | None:
        """Get geometry by a validated full lowercase SHA-256 digest."""
        _validate_digest(content_sha256, "content_sha256")
        return self._by_digest.get(content_sha256)

    def admit(self, content_sha256: str, source: str) -> DocumentGeometry:
        """Validate, derive, and retain exact geometry for ``source``.

        The digest is part of the admission identity rather than a hint.  This
        makes callers prove the source bytes they intend to validate and keeps
        source conflicts fail-closed even when a caller reuses a document key.
        """
        _validate_digest(content_sha256, "content_sha256")
        if not isinstance(source, str):
            raise TopicGeometryError("source must be text")

        existing = self._by_digest.get(content_sha256)
        if existing is not None:
            if existing.source != source:
                raise GeometryConflictError(
                    "content_sha256 is already bound to different source text"
                )
            self._counters = _replace_counters(
                self._counters, cache_hits=self._counters.cache_hits + 1
            )
            return existing

        actual_digest = _digest(source)
        if actual_digest != content_sha256:
            raise DigestMismatchError("content_sha256 does not match UTF-8 source bytes")

        geometry = _derive_geometry(content_sha256, source)
        self._by_digest[content_sha256] = geometry
        self._counters = _replace_counters(
            self._counters,
            documents_indexed=self._counters.documents_indexed + 1,
            geometry_builds=self._counters.geometry_builds + 1,
            paragraph_spans_built=self._counters.paragraph_spans_built
            + len(geometry.paragraphs),
            sentence_spans_built=self._counters.sentence_spans_built
            + sum(len(rows) for rows in geometry.sentences_by_paragraph.values()),
        )
        return geometry


def _replace_counters(counters: GeometryCounters, **changes: int) -> GeometryCounters:
    values = {
        "documents_indexed": counters.documents_indexed,
        "geometry_builds": counters.geometry_builds,
        "cache_hits": counters.cache_hits,
        "paragraph_spans_built": counters.paragraph_spans_built,
        "sentence_spans_built": counters.sentence_spans_built,
    }
    values.update(changes)
    return GeometryCounters(**values)


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or fullmatch(_DIGEST_PATTERN, value) is None:
        raise DigestMismatchError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def _span_key(value: SourceSpan | SpanKey) -> SpanKey:
    if isinstance(value, SourceSpan):
        return value.start_char, value.end_char
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
    ):
        raise TopicGeometryError("span lookup must be a (start_char, end_char) tuple")
    return value


def _derive_geometry(content_sha256: str, source: str) -> DocumentGeometry:
    source_cache = _source_validation_cache(source)
    byte_offsets = source_cache.byte_offsets
    _validate_byte_offsets(source, byte_offsets)

    scoring_text, scoring_boundaries = _scoring_text_and_boundaries(source)
    _validate_scoring_boundaries(source, scoring_text, scoring_boundaries)

    paragraphs = source_cache.paragraphs
    paragraph_index = _index_paragraphs(source, byte_offsets, paragraphs)
    if paragraph_index != source_cache.paragraph_index:
        raise GeometryInvariantError("source validation paragraph index is inconsistent")
    sentences_by_paragraph: dict[SpanKey, tuple[SourceSpan, ...]] = {}
    sentence_indices: dict[SpanKey, Mapping[SpanKey, int]] = {}
    for paragraph in paragraphs:
        key = (paragraph.start_char, paragraph.end_char)
        sentences = source_cache.sentences_by_paragraph.get(key)
        if sentences is None:
            sentences = _sentences_in_paragraph(source, paragraph, byte_offsets)
            source_cache.sentences_by_paragraph[key] = sentences
        _validate_sentences(source, byte_offsets, paragraph, sentences)
        sentence_keys = [(sentence.start_char, sentence.end_char) for sentence in sentences]
        if len(sentence_keys) != len(set(sentence_keys)):
            raise GeometryInvariantError("duplicate sentence span")
        sentences_by_paragraph[key] = sentences
        sentence_indices[key] = MappingProxyType(
            {sentence_key: index for index, sentence_key in enumerate(sentence_keys)}
        )

    return DocumentGeometry(
        content_sha256=content_sha256,
        source=source,
        byte_offsets=byte_offsets,
        scoring_text=scoring_text,
        scoring_boundaries=scoring_boundaries,
        scoring_text_sha256=_digest(scoring_text),
        paragraphs=paragraphs,
        paragraph_index=MappingProxyType(paragraph_index),
        sentences_by_paragraph=MappingProxyType(sentences_by_paragraph),
        sentence_indices_by_paragraph=MappingProxyType(sentence_indices),
    )


def _validate_byte_offsets(source: str, offsets: tuple[int, ...]) -> None:
    if len(offsets) != len(source) + 1 or offsets[0] != 0:
        raise GeometryInvariantError("byte offsets do not match exact UTF-8 source")
    expected = 0
    for index, character in enumerate(source):
        expected += len(character.encode("utf-8"))
        if offsets[index + 1] != expected:
            raise GeometryInvariantError("byte offsets do not match exact UTF-8 source")


def _validate_scoring_boundaries(
    source: str,
    scoring_text: str,
    boundaries: tuple[int, ...],
) -> None:
    if len(boundaries) != len(scoring_text) + 1:
        raise GeometryInvariantError("scoring boundary length is inconsistent")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > len(source)
        for value in boundaries
    ):
        raise GeometryInvariantError("scoring boundary is outside source")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise GeometryInvariantError("scoring boundaries are not strictly ordered")
    if scoring_text and boundaries[-1] != boundaries[-2] + 1:
        raise GeometryInvariantError("scoring boundary end does not match source text")
    for index, character in enumerate(scoring_text):
        source_index = boundaries[index]
        if source[source_index] == character:
            continue
        if (
            character == " "
            and index > 0
            and source_index == boundaries[index - 1] + 1
            and source_index < len(source)
            and source[source_index].isspace()
        ):
            continue
        raise GeometryInvariantError("scoring boundary does not identify source text")


def _index_paragraphs(
    source: str,
    byte_offsets: tuple[int, ...],
    paragraphs: tuple[SourceSpan, ...],
) -> dict[SpanKey, int]:
    result: dict[SpanKey, int] = {}
    previous_end = -1
    for index, paragraph in enumerate(paragraphs):
        _validate_span(source, byte_offsets, paragraph, "paragraph")
        key = (paragraph.start_char, paragraph.end_char)
        if key in result:
            raise GeometryInvariantError("duplicate paragraph span")
        if paragraph.start_char < previous_end:
            raise GeometryInvariantError("overlapping paragraph spans")
        result[key] = index
        previous_end = paragraph.end_char
    return result


def _validate_sentences(
    source: str,
    byte_offsets: tuple[int, ...],
    paragraph: SourceSpan,
    sentences: tuple[SourceSpan, ...],
) -> None:
    previous_end = paragraph.start_char
    for sentence in sentences:
        _validate_span(source, byte_offsets, sentence, "sentence")
        if not (
            paragraph.start_char <= sentence.start_char
            and sentence.end_char <= paragraph.end_char
        ):
            raise GeometryInvariantError("sentence is outside its paragraph")
        if sentence.start_char < previous_end:
            raise GeometryInvariantError("overlapping sentence spans")
        previous_end = sentence.end_char


def _validate_span(
    source: str,
    byte_offsets: tuple[int, ...],
    span: SourceSpan,
    kind: str,
) -> None:
    if not isinstance(span, SourceSpan):
        raise GeometryInvariantError(f"{kind} geometry is not a SourceSpan")
    if span.start_char < 0 or span.end_char <= span.start_char or span.end_char > len(source):
        raise GeometryInvariantError(f"{kind} span is outside source")
    if source[span.start_char : span.end_char] != span.text:
        raise GeometryInvariantError(f"{kind} text does not match source span")
    if (
        span.start_byte != byte_offsets[span.start_char]
        or span.end_byte != byte_offsets[span.end_char]
    ):
        raise GeometryInvariantError(f"{kind} byte offsets do not match source")
    if span.text_sha256 != _digest(span.text):
        raise GeometryInvariantError(f"{kind} digest does not match source text")

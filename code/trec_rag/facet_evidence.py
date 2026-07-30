"""Exact, provenance-complete extraction and selection for facet evidence.

This module deliberately has no model import: callers inject the local sentence
pair scorer and similarity provider through typed protocols.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real
import re
from typing import Protocol


SCHEMA_VERSION = "extractive_candidate_nugget_v1"
SENTENCE_SPLITTER_VERSION = "exact_rules_v1"
SCORING_NORMALIZATION_VERSION = "trec_rag_whitespace_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKENS = re.compile(r"\S+")
_WHITESPACE_RUN = re.compile(r"\s+")
_CLOSERS = "\"'”’)]}"
_ABBREVIATIONS = frozenset({
    "adj", "apr", "aug", "ave", "blvd", "capt", "cf", "co", "dec", "dr",
    "e.g", "ed", "etc", "feb", "fig", "fri", "gen", "i.e", "inc", "jan",
    "jr", "jul", "jun", "lt", "mar", "mr", "mrs", "ms", "mt", "no", "nov",
    "oct", "prof", "rd", "sep", "sept", "sr", "st", "thu", "tues", "vs",
    "wed",
})
_DEPENDENCY_CUES = (
    "this", "that", "these", "those", "it", "they", "such", "the former",
    "the latter", "therefore", "thus", "consequently", "as a result",
)
_DEPENDENCY_BOUNDARY_PUNCTUATION = frozenset(",;:\u2013\u2014-")


def _hash(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return sha256(data).hexdigest()


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hash_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class CandidateSubnarrative:
    subnarrative_id: str
    text: str
    text_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty_string(self.subnarrative_id, "subnarrative_id")
        _nonempty_string(self.text, "subnarrative text")
        object.__setattr__(self, "text_sha256", _hash(self.text))


@dataclass(frozen=True)
class ScoredPassage:
    passage_id: str
    lane_id: str
    query_id: str
    scoring_start_char: int
    scoring_end_char: int
    scoring_text_sha256: str
    chunk_text_sha256: str
    cross_encoder_score: float
    cross_encoder_rank: int

    def __post_init__(self) -> None:
        for name in ("passage_id", "lane_id", "query_id"):
            _nonempty_string(getattr(self, name), name)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.scoring_start_char, self.scoring_end_char)):
            raise ValueError("scoring passage offsets must be integers")
        if self.scoring_start_char < 0 or self.scoring_end_char <= self.scoring_start_char:
            raise ValueError("scoring passage offsets must be a non-empty range")
        _hash_value(self.scoring_text_sha256, "scoring_text_sha256")
        _hash_value(self.chunk_text_sha256, "chunk_text_sha256")
        object.__setattr__(self, "cross_encoder_score", _finite_score(self.cross_encoder_score, "passage score"))
        if isinstance(self.cross_encoder_rank, bool) or not isinstance(self.cross_encoder_rank, int) or self.cross_encoder_rank <= 0:
            raise ValueError("cross_encoder_rank must be positive")


@dataclass(frozen=True)
class ExtractiveCandidateRequest:
    topic_id: str
    document_id: str
    source: str
    document_sha256: str
    scoring_text_sha256: str
    subnarratives: tuple[CandidateSubnarrative, ...]
    passages: tuple[ScoredPassage, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.topic_id, "topic_id")
        _nonempty_string(self.document_id, "document_id")
        if not isinstance(self.source, str):
            raise ValueError("source must be text")
        _hash_value(self.document_sha256, "document_sha256")
        _hash_value(self.scoring_text_sha256, "scoring_text_sha256")
        if not isinstance(self.subnarratives, tuple) or not self.subnarratives or any(not isinstance(row, CandidateSubnarrative) for row in self.subnarratives):
            raise ValueError("subnarratives must be a non-empty tuple of CandidateSubnarrative")
        if len({row.subnarrative_id for row in self.subnarratives}) != len(self.subnarratives):
            raise ValueError("subnarrative IDs must be unique")
        if not isinstance(self.passages, tuple) or not self.passages or any(not isinstance(row, ScoredPassage) for row in self.passages):
            raise ValueError("passages must be a non-empty tuple of ScoredPassage")
        passages_by_id: dict[str, ScoredPassage] = {}
        for passage in self.passages:
            previous = passages_by_id.setdefault(passage.passage_id, passage)
            if previous != passage:
                raise ValueError(f"conflicting passage_id {passage.passage_id!r}")


@dataclass(frozen=True)
class SentencePair:
    topic_id: str
    document_id: str
    subnarrative_id: str
    query_text: str
    sentence_text: str

    def __post_init__(self) -> None:
        for name in ("topic_id", "document_id", "subnarrative_id", "query_text", "sentence_text"):
            _nonempty_string(getattr(self, name), name)


class SentencePairScorer(Protocol):
    def score_pairs(self, pairs: Sequence[SentencePair]) -> Sequence[float]: ...


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    text_sha256: str


@dataclass(frozen=True)
class SentenceEvidence(SourceSpan):
    cross_encoder_score: float


@dataclass(frozen=True)
class PassageProvenance:
    passage_id: str
    lane_id: str
    query_id: str
    scoring_start_char: int
    scoring_end_char: int
    source_start_char: int
    source_end_char: int
    source_start_byte: int
    source_end_byte: int
    source_text: str
    source_text_sha256: str
    scoring_text_sha256: str
    chunk_text_sha256: str
    normalization_version: str
    cross_encoder_score: float
    cross_encoder_rank: int


@dataclass(frozen=True)
class ExtractiveCandidate:
    schema_version: str
    topic_id: str
    docid: str
    subnarrative_id: str
    candidate_nugget_id: str
    nugget_type: str
    candidate_kind: str
    text: str
    evidence_sentences: tuple[SentenceEvidence, ...]
    matched_paragraph: SourceSpan
    context_before: SourceSpan | None
    context_after: SourceSpan | None
    passages: tuple[PassageProvenance, ...]
    sentence_cross_encoder_score: float
    rank_within_document_subnarrative: int
    document_sha256: str
    scoring_text_sha256: str
    subnarrative_sha256: str
    sentence_splitter_version: str


@dataclass(frozen=True)
class _TranslatedPassage:
    passage: ScoredPassage
    source_start_char: int
    source_end_char: int


def _scoring_text_and_boundaries(source: str) -> tuple[str, tuple[int, ...]]:
    tokens = tuple(_TOKENS.finditer(source))
    if not tokens:
        return "", (0,)
    parts: list[str] = []
    boundaries = [tokens[0].start()]
    for index, token in enumerate(tokens):
        if index:
            parts.append(" ")
            boundaries.append(token.start())
        word = token.group()
        parts.append(word)
        boundaries.extend(token.start() + offset for offset in range(1, len(word) + 1))
    text = "".join(parts)
    if len(boundaries) != len(text) + 1:
        raise AssertionError("scoring boundary construction is inconsistent")
    return text, tuple(boundaries)


def project_source_span(
    source: str,
    raw_start: int,
    raw_end: int,
) -> tuple[str, int, int, str]:
    """Project one raw-source span into ``trec_rag_whitespace_v1`` coordinates.

    Peripheral source whitespace is not evidence. Interior whitespace is
    collapsed exactly as it is for sentence scoring. The returned tuple is the
    complete scoring text, projected start/end offsets, and exact projected
    chunk text.
    """
    scoring_text, projected = project_source_spans(source, ((raw_start, raw_end),))
    scoring_start, scoring_end, chunk = projected[0]
    return scoring_text, scoring_start, scoring_end, chunk


def project_source_spans(
    source: str,
    raw_spans: Sequence[tuple[int, int]],
) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    """Project several raw spans while normalizing the document only once."""
    if not isinstance(source, str):
        raise TypeError("source must be text")
    spans = tuple(raw_spans)
    scoring_text, boundaries = _scoring_text_and_boundaries(source)
    result: list[tuple[int, int, str]] = []
    for raw_start, raw_end in spans:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (raw_start, raw_end)
        ):
            raise ValueError("raw source offsets must be integers")
        if raw_start < 0 or raw_end <= raw_start or raw_end > len(source):
            raise ValueError("raw source offsets must be a non-empty in-range span")

        trimmed_start = raw_start
        trimmed_end = raw_end
        while trimmed_start < trimmed_end and source[trimmed_start].isspace():
            trimmed_start += 1
        while trimmed_end > trimmed_start and source[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_start == trimmed_end:
            raise ValueError("raw source span contains only whitespace")

        scoring_start = bisect_left(boundaries, trimmed_start)
        scoring_end = bisect_left(boundaries, trimmed_end)
        if (
            scoring_start >= len(boundaries)
            or scoring_end >= len(boundaries)
            or boundaries[scoring_start] != trimmed_start
            or boundaries[scoring_end] != trimmed_end
            or scoring_end <= scoring_start
        ):
            raise ValueError("raw source span cannot be projected into scoring coordinates")
        chunk = scoring_text[scoring_start:scoring_end]
        expected = " ".join(source[trimmed_start:trimmed_end].split())
        if not chunk or chunk != expected:
            raise ValueError("projected scoring span differs from normalized source")
        result.append((scoring_start, scoring_end, chunk))
    return scoring_text, tuple(result)


def _normalize_scoring_slice(source: str) -> str:
    """Collapse whitespace without stripping a selected synthetic separator."""
    return _WHITESPACE_RUN.sub(" ", source)


def _byte_offsets(source: str) -> tuple[int, ...]:
    offsets = [0]
    for char in source:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    return tuple(offsets)


def _source_spans(source: str, byte_offsets: tuple[int, ...]) -> tuple[SourceSpan, ...]:
    result: list[SourceSpan] = []
    cursor = 0
    for line in source.splitlines(keepends=True):
        line_end = cursor + len(line.rstrip("\r\n"))
        if source[cursor:line_end].strip():
            result.append(_make_span(source, byte_offsets, cursor, line_end))
        cursor += len(line)
    if cursor < len(source) and source[cursor:].strip():
        result.append(_make_span(source, byte_offsets, cursor, len(source)))
    if not source and cursor == 0:
        return ()
    return tuple(result)


def _make_span(source: str, byte_offsets: tuple[int, ...], start: int, end: int) -> SourceSpan:
    text = source[start:end]
    if start < 0 or end <= start or end > len(source) or not text.strip():
        raise ValueError("source span must be non-empty exact source text")
    return SourceSpan(text, start, end, byte_offsets[start], byte_offsets[end], _hash(text))


def _word_before(text: str, index: int) -> str:
    reversed_word: list[str] = []
    cursor = index
    while cursor > 0:
        char = text[cursor - 1]
        if char != "." and not ("A" <= char <= "Z" or "a" <= char <= "z"):
            break
        reversed_word.append(char)
        cursor -= 1
    return "".join(reversed(reversed_word)).casefold()


def _is_terminal(text: str, index: int) -> bool:
    char = text[index]
    cursor = index + 1
    while cursor < len(text) and text[cursor] in _CLOSERS:
        cursor += 1
    at_paragraph_end = cursor == len(text)
    if char == ".":
        if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
        word = _word_before(text, index)
        if not at_paragraph_end and (
            word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())
        ):
            return False
    return at_paragraph_end or text[cursor].isspace()


def _sentences_in_paragraph(source: str, paragraph: SourceSpan, byte_offsets: tuple[int, ...]) -> tuple[SourceSpan, ...]:
    text = paragraph.text
    result: list[SourceSpan] = []
    start = 0
    for index, char in enumerate(text):
        if char not in ".?!" or not _is_terminal(text, index):
            continue
        end = index + 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        left = start
        while left < end and text[left].isspace():
            left += 1
        right = end
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            result.append(_make_span(source, byte_offsets, paragraph.start_char + left, paragraph.start_char + right))
        start = end
    left = start
    right = len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        result.append(
            _make_span(
                source,
                byte_offsets,
                paragraph.start_char + left,
                paragraph.start_char + right,
            )
        )
    return tuple(result)


def _dependency_cue(sentence: str) -> bool:
    value = sentence.lstrip().casefold()
    for cue in _DEPENDENCY_CUES:
        if not value.startswith(cue):
            continue
        if len(value) == len(cue):
            return True
        boundary = value[len(cue)]
        if boundary.isspace() or boundary in _DEPENDENCY_BOUNDARY_PUNCTUATION:
            return True
    return False


def _candidate_id(request: ExtractiveCandidateRequest, subnarrative: CandidateSubnarrative, kind: str, sentences: tuple[SentenceEvidence, ...]) -> str:
    identity = {
        "candidate_kind": kind,
        "document_id": request.document_id,
        "document_sha256": request.document_sha256,
        "schema_version": SCHEMA_VERSION,
        "sentences": [
            {
                "end_byte": row.end_byte,
                "end_char": row.end_char,
                "start_byte": row.start_byte,
                "start_char": row.start_char,
                "text_sha256": row.text_sha256,
            }
            for row in sentences
        ],
        "subnarrative_id": subnarrative.subnarrative_id,
        "subnarrative_sha256": subnarrative.text_sha256,
        "topic_id": request.topic_id,
    }
    return "ecn1_" + _hash(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _validate_and_translate(request: ExtractiveCandidateRequest) -> tuple[str, tuple[int, ...], tuple[_TranslatedPassage, ...]]:
    if _hash(request.source) != request.document_sha256:
        raise ValueError("document_sha256 does not match source")
    scoring_text, boundaries = _scoring_text_and_boundaries(request.source)
    if _hash(scoring_text) != request.scoring_text_sha256:
        raise ValueError("scoring_text_sha256 does not match normalized source")
    translated: list[_TranslatedPassage] = []
    for passage in request.passages:
        if passage.scoring_text_sha256 != request.scoring_text_sha256:
            raise ValueError("passage scoring_text_sha256 does not match request")
        if passage.scoring_end_char > len(scoring_text):
            raise ValueError("scoring passage is outside normalized source")
        chunk = scoring_text[passage.scoring_start_char : passage.scoring_end_char]
        if _hash(chunk) != passage.chunk_text_sha256:
            raise ValueError("passage chunk_text_sha256 does not match scoring slice")
        source_start = boundaries[passage.scoring_start_char]
        source_end = boundaries[passage.scoring_end_char]
        if source_end <= source_start or _normalize_scoring_slice(request.source[source_start:source_end]) != chunk:
            raise ValueError("scoring passage cannot translate to an exact source slice")
        translated.append(_TranslatedPassage(passage, source_start, source_end))
    return scoring_text, boundaries, tuple(translated)


def validate_candidate_request(request: ExtractiveCandidateRequest) -> None:
    """Validate every source/scoring coordinate before a request is serialized."""
    if not isinstance(request, ExtractiveCandidateRequest):
        raise TypeError("request must be an ExtractiveCandidateRequest")
    _validate_and_translate(request)


def validate_extractive_candidate_source(
    candidate: ExtractiveCandidate,
    *,
    source: str,
    subnarrative_text: str,
) -> None:
    """Revalidate one decoded candidate against its exact source and plan text."""
    if not isinstance(candidate, ExtractiveCandidate):
        raise TypeError("candidate must be an ExtractiveCandidate")
    if not isinstance(source, str) or not source:
        raise ValueError("candidate source must be non-empty text")
    subnarrative = CandidateSubnarrative(
        candidate.subnarrative_id, subnarrative_text
    )
    passages = tuple(
        ScoredPassage(
            passage_id=row.passage_id,
            lane_id=row.lane_id,
            query_id=row.query_id,
            scoring_start_char=row.scoring_start_char,
            scoring_end_char=row.scoring_end_char,
            scoring_text_sha256=row.scoring_text_sha256,
            chunk_text_sha256=row.chunk_text_sha256,
            cross_encoder_score=row.cross_encoder_score,
            cross_encoder_rank=row.cross_encoder_rank,
        )
        for row in candidate.passages
    )
    request = ExtractiveCandidateRequest(
        topic_id=candidate.topic_id,
        document_id=candidate.docid,
        source=source,
        document_sha256=candidate.document_sha256,
        scoring_text_sha256=candidate.scoring_text_sha256,
        subnarratives=(subnarrative,),
        passages=passages,
    )
    _scoring_text, _boundaries, translated = _validate_and_translate(request)
    if (
        candidate.schema_version != SCHEMA_VERSION
        or candidate.nugget_type != "extractive"
        or candidate.candidate_kind
        not in {"exact_sentence", "exact_sentence_pair"}
        or candidate.sentence_splitter_version != SENTENCE_SPLITTER_VERSION
        or candidate.subnarrative_sha256 != subnarrative.text_sha256
        or isinstance(candidate.rank_within_document_subnarrative, bool)
        or not isinstance(candidate.rank_within_document_subnarrative, int)
        or candidate.rank_within_document_subnarrative <= 0
        or not candidate.passages
        or not candidate.evidence_sentences
    ):
        raise ValueError("candidate schema or plan identity is inconsistent")
    byte_offsets = _byte_offsets(source)
    if len({row.passage_id for row in candidate.passages}) != len(candidate.passages):
        raise ValueError("candidate passage IDs must be unique")
    for row, projected in zip(candidate.passages, translated, strict=True):
        if (
            row.normalization_version != SCORING_NORMALIZATION_VERSION
            or row.source_start_char != projected.source_start_char
            or row.source_end_char != projected.source_end_char
            or row.source_start_byte != byte_offsets[projected.source_start_char]
            or row.source_end_byte != byte_offsets[projected.source_end_char]
            or row.source_text
            != source[projected.source_start_char : projected.source_end_char]
            or row.source_text_sha256 != _hash(row.source_text)
        ):
            raise ValueError("candidate passage projection is inconsistent")
    expected_sentence_count = (
        1 if candidate.candidate_kind == "exact_sentence" else 2
    )
    if len(candidate.evidence_sentences) != expected_sentence_count:
        raise ValueError("candidate kind differs from its exact sentence spans")
    def validate_span(span: SourceSpan, label: str) -> None:
        if (
            isinstance(span.start_char, bool)
            or not isinstance(span.start_char, int)
            or isinstance(span.end_char, bool)
            or not isinstance(span.end_char, int)
            or span.start_char < 0
            or span.end_char <= span.start_char
            or span.end_char > len(source)
            or source[span.start_char : span.end_char] != span.text
            or byte_offsets[span.start_char] != span.start_byte
            or byte_offsets[span.end_char] != span.end_byte
            or _hash(span.text) != span.text_sha256
        ):
            raise ValueError(f"candidate {label} source span is inconsistent")

    validate_span(candidate.matched_paragraph, "paragraph")
    for sentence in candidate.evidence_sentences:
        validate_span(sentence, "sentence")
        _finite_score(sentence.cross_encoder_score, "candidate sentence score")
    for label, context in (
        ("context_before", candidate.context_before),
        ("context_after", candidate.context_after),
    ):
        if context is not None:
            validate_span(context, label)
    paragraphs = _source_spans(source, byte_offsets)
    try:
        paragraph_index = paragraphs.index(candidate.matched_paragraph)
    except ValueError as exc:
        raise ValueError("candidate paragraph is not an exact source paragraph") from exc
    expected_before = paragraphs[paragraph_index - 1] if paragraph_index else None
    expected_after = (
        paragraphs[paragraph_index + 1]
        if paragraph_index + 1 < len(paragraphs)
        else None
    )
    if candidate.context_before != expected_before or candidate.context_after != expected_after:
        raise ValueError("candidate paragraph context is inconsistent")
    source_sentences = _sentences_in_paragraph(
        source, candidate.matched_paragraph, byte_offsets
    )
    if any(
        not any(
            sentence.text == source_sentence.text
            and sentence.start_char == source_sentence.start_char
            and sentence.end_char == source_sentence.end_char
            and sentence.start_byte == source_sentence.start_byte
            and sentence.end_byte == source_sentence.end_byte
            and sentence.text_sha256 == source_sentence.text_sha256
            for source_sentence in source_sentences
        )
        for sentence in candidate.evidence_sentences
    ):
        raise ValueError("candidate evidence is not an exact source sentence")
    if candidate.candidate_kind == "exact_sentence_pair":
        first, second = candidate.evidence_sentences
        indices = [
            next(
                index
                for index, source_sentence in enumerate(source_sentences)
                if source_sentence.start_char == sentence.start_char
                and source_sentence.end_char == sentence.end_char
            )
            for sentence in (first, second)
        ]
        if indices[1] != indices[0] + 1 or not _dependency_cue(second.text):
            raise ValueError("candidate sentence pair is not an admitted adjacent pair")
    expected_text = source[
        candidate.evidence_sentences[0].start_char
        : candidate.evidence_sentences[-1].end_char
    ]
    if (
        candidate.text != expected_text
        or _finite_score(
            candidate.sentence_cross_encoder_score, "candidate score"
        )
        != min(row.cross_encoder_score for row in candidate.evidence_sentences)
        or candidate.candidate_nugget_id
        != _candidate_id(
            request,
            subnarrative,
            candidate.candidate_kind,
            candidate.evidence_sentences,
        )
    ):
        raise ValueError("candidate text, score, or content identity is inconsistent")
    _recheck_candidates(request, (candidate,))


def _passage_provenance(
    translated: Sequence[_TranslatedPassage],
    source: str,
    byte_offsets: tuple[int, ...],
    paragraph: SourceSpan,
) -> tuple[PassageProvenance, ...]:
    selected = [
        row for row in translated
        if row.source_start_char < paragraph.end_char and row.source_end_char > paragraph.start_char
    ]
    unique = {
        (
            row.passage.passage_id, row.passage.lane_id, row.passage.query_id,
            row.passage.scoring_start_char, row.passage.scoring_end_char,
            row.source_start_char, row.source_end_char, row.passage.scoring_text_sha256,
            row.passage.chunk_text_sha256, row.passage.cross_encoder_score, row.passage.cross_encoder_rank,
        ): row
        for row in selected
    }
    return tuple(
        PassageProvenance(
            passage_id=row.passage.passage_id,
            lane_id=row.passage.lane_id,
            query_id=row.passage.query_id,
            scoring_start_char=row.passage.scoring_start_char,
            scoring_end_char=row.passage.scoring_end_char,
            source_start_char=row.source_start_char,
            source_end_char=row.source_end_char,
            source_start_byte=byte_offsets[row.source_start_char],
            source_end_byte=byte_offsets[row.source_end_char],
            source_text=source[row.source_start_char : row.source_end_char],
            source_text_sha256=_hash(source[row.source_start_char : row.source_end_char]),
            scoring_text_sha256=row.passage.scoring_text_sha256,
            chunk_text_sha256=row.passage.chunk_text_sha256,
            normalization_version=SCORING_NORMALIZATION_VERSION,
            cross_encoder_score=row.passage.cross_encoder_score,
            cross_encoder_rank=row.passage.cross_encoder_rank,
        )
        for _, row in sorted(unique.items())
    )


def extract_document_candidates(request: ExtractiveCandidateRequest, scorer: SentencePairScorer) -> tuple[ExtractiveCandidate, ...]:
    """Extract exact complete source sentences for every supplied subnarrative."""
    if not isinstance(request, ExtractiveCandidateRequest):
        raise TypeError("request must be ExtractiveCandidateRequest")
    if not hasattr(scorer, "score_pairs"):
        raise TypeError("scorer must provide score_pairs")
    _, _, translated = _validate_and_translate(request)
    byte_offsets = _byte_offsets(request.source)
    paragraphs = _source_spans(request.source, byte_offsets)
    if any(
        not any(
            passage.source_start_char < paragraph.end_char
            and passage.source_end_char > paragraph.start_char
            for paragraph in paragraphs
        )
        for passage in translated
    ):
        raise ValueError("every passage must intersect a non-empty source paragraph")
    matched = tuple(
        (index, paragraph, _passage_provenance(translated, request.source, byte_offsets, paragraph))
        for index, paragraph in enumerate(paragraphs)
        if _passage_provenance(translated, request.source, byte_offsets, paragraph)
    )
    sentence_rows = tuple(
        (index, paragraph, provenance, _sentences_in_paragraph(request.source, paragraph, byte_offsets))
        for index, paragraph, provenance in matched
    )
    located_sentences = tuple(
        (paragraph_index, paragraph, provenance, sentence)
        for paragraph_index, paragraph, provenance, sentences in sentence_rows
        for sentence in sentences
    )
    pairs = tuple(
        SentencePair(request.topic_id, request.document_id, sub.subnarrative_id, sub.text, sentence.text)
        for sub in request.subnarratives
        for _, _, _, sentence in located_sentences
    )
    scores = tuple(scorer.score_pairs(pairs))
    if len(scores) != len(pairs):
        raise ValueError("scorer response length must match sentence pairs")
    scores = tuple(_finite_score(score, "sentence score") for score in scores)
    score_lookup = {
        (sub.subnarrative_id, sentence.start_char, sentence.end_char): score
        for sub_index, sub in enumerate(request.subnarratives)
        for (_, _, _, sentence), score in zip(
            located_sentences,
            scores[sub_index * len(located_sentences) : (sub_index + 1) * len(located_sentences)],
            strict=True,
        )
    }
    candidates: list[ExtractiveCandidate] = []
    for sub in request.subnarratives:
        singleton_rows: list[tuple[SourceSpan, SourceSpan, int, tuple[PassageProvenance, ...], float]] = []
        for paragraph_index, paragraph, provenance, sentence in located_sentences:
            score = score_lookup[(sub.subnarrative_id, sentence.start_char, sentence.end_char)]
            singleton_rows.append((sentence, paragraph, paragraph_index, provenance, score))
        singleton_rows.sort(key=lambda row: (-row[4], row[0].start_char, row[0].end_char))
        for rank, (sentence, paragraph, paragraph_index, provenance, score) in enumerate(singleton_rows, start=1):
            evidence = SentenceEvidence(**asdict(sentence), cross_encoder_score=score)
            candidates.append(_build_candidate(request, sub, "exact_sentence", (evidence,), paragraph, paragraphs, paragraph_index, provenance, score, rank))
        pair_rows: list[tuple[tuple[SentenceEvidence, SentenceEvidence], SourceSpan, int, tuple[PassageProvenance, ...], float]] = []
        for paragraph_index, paragraph, provenance, sentences in sentence_rows:
            for first, second in zip(sentences, sentences[1:]):
                if not _dependency_cue(second.text):
                    continue
                first_score = score_lookup[(sub.subnarrative_id, first.start_char, first.end_char)]
                second_score = score_lookup[(sub.subnarrative_id, second.start_char, second.end_char)]
                pair_rows.append((
                    (SentenceEvidence(**asdict(first), cross_encoder_score=first_score), SentenceEvidence(**asdict(second), cross_encoder_score=second_score)),
                    paragraph, paragraph_index, provenance, min(first_score, second_score),
                ))
        for rank, (evidence, paragraph, paragraph_index, provenance, score) in enumerate(sorted(pair_rows, key=lambda row: (-row[4], row[0][0].start_char)), start=1):
            candidates.append(_build_candidate(request, sub, "exact_sentence_pair", evidence, paragraph, paragraphs, paragraph_index, provenance, score, rank))
    _recheck_candidates(request, candidates)
    return tuple(candidates)


def _build_candidate(request: ExtractiveCandidateRequest, sub: CandidateSubnarrative, kind: str, evidence: tuple[SentenceEvidence, ...], paragraph: SourceSpan, paragraphs: tuple[SourceSpan, ...], paragraph_index: int, provenance: tuple[PassageProvenance, ...], score: float, rank: int) -> ExtractiveCandidate:
    text = request.source[evidence[0].start_char : evidence[-1].end_char]
    return ExtractiveCandidate(
        schema_version=SCHEMA_VERSION,
        topic_id=request.topic_id,
        docid=request.document_id,
        subnarrative_id=sub.subnarrative_id,
        candidate_nugget_id=_candidate_id(request, sub, kind, evidence),
        nugget_type="extractive",
        candidate_kind=kind,
        text=text,
        evidence_sentences=evidence,
        matched_paragraph=paragraph,
        context_before=paragraphs[paragraph_index - 1] if paragraph_index else None,
        context_after=paragraphs[paragraph_index + 1] if paragraph_index + 1 < len(paragraphs) else None,
        passages=provenance,
        sentence_cross_encoder_score=score,
        rank_within_document_subnarrative=rank,
        document_sha256=request.document_sha256,
        scoring_text_sha256=request.scoring_text_sha256,
        subnarrative_sha256=sub.text_sha256,
        sentence_splitter_version=SENTENCE_SPLITTER_VERSION,
    )


def _recheck_candidates(request: ExtractiveCandidateRequest, candidates: Sequence[ExtractiveCandidate]) -> None:
    byte_offsets = _byte_offsets(request.source)
    for candidate in candidates:
        if not candidate.passages or candidate.document_sha256 != _hash(request.source):
            raise ValueError("candidate provenance or document hash is inconsistent")
        if candidate.text != request.source[candidate.evidence_sentences[0].start_char : candidate.evidence_sentences[-1].end_char]:
            raise ValueError("candidate text is not an exact source slice")
        for sentence in candidate.evidence_sentences:
            if (request.source[sentence.start_char : sentence.end_char] != sentence.text or
                byte_offsets[sentence.start_char] != sentence.start_byte or
                byte_offsets[sentence.end_char] != sentence.end_byte or
                _hash(sentence.text) != sentence.text_sha256):
                raise ValueError("candidate sentence provenance is inconsistent")
        for passage in candidate.passages:
            if (
                passage.source_start_char < 0
                or passage.source_end_char <= passage.source_start_char
                or passage.source_end_char > len(request.source)
                or request.source[passage.source_start_char : passage.source_end_char] != passage.source_text
                or byte_offsets[passage.source_start_char] != passage.source_start_byte
                or byte_offsets[passage.source_end_char] != passage.source_end_byte
                or _hash(passage.source_text) != passage.source_text_sha256
            ):
                raise ValueError("candidate passage source provenance is inconsistent")

SELECTION_SCHEMA_VERSION = "subnarrative_selection_v1"
DEFAULT_SEMANTIC_THRESHOLD = 0.92
DEFAULT_MMR_LAMBDA = 0.7
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])")
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s+\d{4})?\b|"
    r"\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|neither|nor|none|without|cannot|can't|won't|"
    r"didn't|doesn't|isn't|wasn't|weren't|aren't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _canonical_identity(value: object) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping) or not value or any(not isinstance(key, str) or not key for key in value):
        raise ValueError("similarity identity must be a non-empty canonical immutable mapping")
    rows: list[tuple[str, object]] = []
    for key, item in value.items():
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError("similarity identity values must be canonical immutable JSON scalars")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("similarity identity values must be finite")
        rows.append((key, item))
    return tuple(sorted(rows))


@dataclass(frozen=True)
class SubnarrativeContext:
    topic_id: str
    official_narrative: str
    subnarrative_id: str
    subnarrative_text: str
    official_narrative_sha256: str = field(init=False)
    subnarrative_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("topic_id", "official_narrative", "subnarrative_id", "subnarrative_text"):
            _nonempty(getattr(self, name), name)
        object.__setattr__(self, "official_narrative_sha256", _digest(self.official_narrative))
        object.__setattr__(self, "subnarrative_sha256", _digest(self.subnarrative_text))


@dataclass(frozen=True)
class SelectionCandidate:
    topic_id: str
    subnarrative_id: str
    candidate_nugget_id: str
    candidate_kind: str
    text: str
    docid: str
    document_sha256: str
    raw_logit: float

    def __post_init__(self) -> None:
        for name in ("topic_id", "subnarrative_id", "candidate_nugget_id", "candidate_kind", "text", "docid"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.document_sha256, str) or not _SHA256.fullmatch(self.document_sha256):
            raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "raw_logit", _finite(self.raw_logit, "raw_logit"))


@dataclass(frozen=True)
class EvidenceMember:
    candidate_nugget_id: str
    candidate_kind: str
    text: str
    docid: str
    document_sha256: str
    raw_logit: float

    def __post_init__(self) -> None:
        for name in ("candidate_nugget_id", "candidate_kind", "text", "docid"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.document_sha256, str) or not _SHA256.fullmatch(self.document_sha256):
            raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "raw_logit", _finite(self.raw_logit, "raw_logit"))


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: str
    representative_candidate_nugget_id: str
    representative_text: str
    representative_raw_logit: float
    members: tuple[EvidenceMember, ...]
    supports: tuple[EvidenceMember, ...]
    support_document_count: int

    def __post_init__(self) -> None:
        _nonempty(self.cluster_id, "cluster_id")
        _nonempty(self.representative_candidate_nugget_id, "representative_candidate_nugget_id")
        _nonempty(self.representative_text, "representative_text")
        object.__setattr__(
            self, "representative_raw_logit", _finite(self.representative_raw_logit, "representative_raw_logit")
        )
        if (
            not isinstance(self.members, tuple)
            or any(not isinstance(member, EvidenceMember) for member in self.members)
            or not isinstance(self.supports, tuple)
            or any(not isinstance(member, EvidenceMember) for member in self.supports)
        ):
            raise ValueError("cluster members and supports must be tuples of EvidenceMember")
        if not self.members or not self.supports:
            raise ValueError("clusters must retain members and selected supports")
        member_by_id = {member.candidate_nugget_id: member for member in self.members}
        if len(member_by_id) != len(self.members):
            raise ValueError("cluster member IDs must be unique")
        representative = member_by_id.get(self.representative_candidate_nugget_id)
        if (
            representative is None
            or representative.text != self.representative_text
            or representative.raw_logit != self.representative_raw_logit
        ):
            raise ValueError("cluster representative must identify an exact retained member")
        if any(member_by_id.get(member.candidate_nugget_id) != member for member in self.supports):
            raise ValueError("cluster supports must be retained members")
        if representative not in self.supports:
            raise ValueError("cluster primary representative must be one of its supports")
        if len(self.supports) > 3:
            raise ValueError("a cluster may carry at most three support documents")
        if (
            isinstance(self.support_document_count, bool)
            or not isinstance(self.support_document_count, int)
        ):
            raise ValueError("cluster support_document_count must be a non-Boolean integer")
        support_documents = {member.document_sha256 for member in self.supports}
        if len(support_documents) != len(self.supports) or self.support_document_count != len(support_documents):
            raise ValueError("cluster supports must come from distinct documents")


@dataclass(frozen=True)
class BudgetSnapshot:
    budget: int
    cluster_ids: tuple[str, ...]
    exhausted: bool

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget <= 0:
            raise ValueError("snapshot budget must be strictly positive")
        if (
            not isinstance(self.cluster_ids, tuple)
            or any(not isinstance(cluster_id, str) or not cluster_id for cluster_id in self.cluster_ids)
        ):
            raise ValueError("snapshot cluster_ids must be a tuple of non-empty strings")
        if len(set(self.cluster_ids)) != len(self.cluster_ids):
            raise ValueError("snapshot cluster IDs must be unique")
        if len(self.cluster_ids) > self.budget:
            raise ValueError("snapshot cannot contain more clusters than its budget")
        if not isinstance(self.exhausted, bool):
            raise ValueError("snapshot exhausted must be boolean")


@dataclass(frozen=True)
class SelectionPolicy:
    budgets: tuple[int, ...] = (40, 80, 120)
    precluster_limit: int | None = None
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    mmr_lambda: float = DEFAULT_MMR_LAMBDA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.budgets, tuple)
            or not self.budgets
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.budgets)
            or tuple(sorted(set(self.budgets))) != self.budgets
        ):
            raise ValueError("budgets must be a strictly increasing tuple of positive integers")
        limit = 10 * max(self.budgets) if self.precluster_limit is None else self.precluster_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("precluster_limit must be positive")
        threshold = _finite(self.semantic_threshold, "semantic_threshold")
        mmr_lambda = _finite(self.mmr_lambda, "mmr_lambda")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("semantic_threshold must be between -1 and 1")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        object.__setattr__(self, "precluster_limit", limit)
        object.__setattr__(self, "semantic_threshold", threshold)
        object.__setattr__(self, "mmr_lambda", mmr_lambda)


@dataclass(frozen=True)
class SubnarrativeSelection:
    schema_version: str
    context: SubnarrativeContext
    policy: SelectionPolicy
    similarity_identity: tuple[tuple[str, object], ...]
    candidate_count: int
    exact_group_count: int
    precluster_count: int
    semantic_cluster_count: int
    clusters: tuple[SemanticCluster, ...]
    snapshots: tuple[BudgetSnapshot, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported selection schema version")
        if not isinstance(self.context, SubnarrativeContext) or not isinstance(self.policy, SelectionPolicy):
            raise ValueError("selection context and policy must be validated records")
        try:
            if (
                not isinstance(self.similarity_identity, tuple)
                or self.similarity_identity != _canonical_identity(dict(self.similarity_identity))
            ):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("selection similarity_identity must be a canonical immutable tuple structure") from None
        counts = (
            self.candidate_count, self.exact_group_count, self.precluster_count,
            self.semantic_cluster_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("selection counts must be non-negative integers")
        if not (
            self.candidate_count >= self.exact_group_count >= self.precluster_count
            >= self.semantic_cluster_count >= len(self.clusters)
        ):
            raise ValueError("selection counts must preserve pipeline-stage exhaustion")
        if (
            not isinstance(self.clusters, tuple)
            or any(not isinstance(cluster, SemanticCluster) for cluster in self.clusters)
            or not isinstance(self.snapshots, tuple)
            or any(not isinstance(snapshot, BudgetSnapshot) for snapshot in self.snapshots)
        ):
            raise ValueError("selection clusters and snapshots must be validated tuples")
        ordered_ids = tuple(cluster.cluster_id for cluster in self.clusters)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("selection cluster IDs must be unique")
        if tuple(snapshot.budget for snapshot in self.snapshots) != self.policy.budgets:
            raise ValueError("selection snapshots must match policy budgets")
        for snapshot in self.snapshots:
            if snapshot.cluster_ids != ordered_ids[: snapshot.budget]:
                raise ValueError("selection snapshots must be prefixes of one cluster order")
            if snapshot.exhausted != (len(ordered_ids) < snapshot.budget):
                raise ValueError("selection snapshot exhaustion is inconsistent")


class SimilarityProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, object]: ...

    def cosine_matrix(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

@dataclass(frozen=True)
class _ExactGroup:
    representative: EvidenceMember
    candidate_kind: str
    text: str
    members: tuple[EvidenceMember, ...]
    guard_signature: tuple[object, ...]
    rank: int
    relevance: float


def _member(candidate: SelectionCandidate) -> EvidenceMember:
    return EvidenceMember(
        candidate_nugget_id=candidate.candidate_nugget_id,
        candidate_kind=candidate.candidate_kind,
        text=candidate.text,
        docid=candidate.docid,
        document_sha256=candidate.document_sha256,
        raw_logit=candidate.raw_logit,
    )


def _guard_signature(kind: str, text: str) -> tuple[object, ...]:
    numbers = tuple(match.group().replace(",", "").casefold() for match in _NUMBER.finditer(text))
    dates = tuple(match.group().casefold() for match in _DATE.finditer(text))
    negated = bool(_NEGATION.search(text))
    entities = tuple(sorted({match.group().casefold() for match in _ENTITY.finditer(text)}))
    return kind, numbers, dates, negated, entities


def _exact_groups(candidates: Sequence[SelectionCandidate]) -> tuple[_ExactGroup, ...]:
    grouped: dict[tuple[str, str], list[EvidenceMember]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.candidate_kind, candidate.text), []).append(_member(candidate))
    rows: list[tuple[EvidenceMember, str, str, tuple[EvidenceMember, ...]]] = []
    for (kind, text), members in grouped.items():
        ordered = tuple(sorted(members, key=lambda row: (-row.raw_logit, row.candidate_nugget_id)))
        rows.append((ordered[0], kind, text, ordered))
    rows.sort(key=lambda row: (-row[0].raw_logit, row[0].candidate_nugget_id))
    return tuple(
        _ExactGroup(
            representative=representative,
            candidate_kind=kind,
            text=text,
            members=members,
            guard_signature=_guard_signature(kind, text),
            rank=index,
            relevance=1.0 / index,
        )
        for index, (representative, kind, text, members) in enumerate(rows, start=1)
    )


def _validated_matrix(value: object, size: int) -> object:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != size:  # type: ignore[arg-type]
        raise ValueError("cosine matrix must be square and match the requested texts")
    for row in value:  # type: ignore[union-attr]
        if isinstance(row, (str, bytes)) or not hasattr(row, "__len__") or len(row) != size:
            raise ValueError("cosine matrix must be square and match the requested texts")
        finite_row = tuple(_finite(item, "cosine matrix value") for item in row)
        if any(item < -1.0 or item > 1.0 for item in finite_row):
            raise ValueError("cosine matrix values must be between -1 and 1")
    for left in range(size):
        for right in range(left + 1, size):
            if not math.isclose(value[left][right], value[right][left], rel_tol=0.0, abs_tol=1e-12):  # type: ignore[index]
                raise ValueError("cosine matrix must be symmetric")
    return value


def _cluster_groups(
    groups: tuple[_ExactGroup, ...],
    matrix: object,
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    clusters: list[list[int]] = []
    for member_index, member in enumerate(groups):
        for cluster in clusters:
            representative_index = cluster[0]
            representative = groups[representative_index]
            if (
                member.guard_signature == representative.guard_signature
                and matrix[member_index][representative_index] >= threshold  # type: ignore[index]
            ):
                cluster.append(member_index)
                break
        else:
            clusters.append([member_index])
    return tuple(tuple(cluster) for cluster in clusters)


def _cluster_id(context: SubnarrativeContext, representative: EvidenceMember) -> str:
    identity = {
        "topic_id": context.topic_id,
        "subnarrative_id": context.subnarrative_id,
        "representative_candidate_nugget_id": representative.candidate_nugget_id,
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "esc1_" + _digest(payload)


def _selected_supports(groups: tuple[_ExactGroup, ...], indexes: tuple[int, ...]) -> tuple[EvidenceMember, ...]:
    ordered = (
        member
        for index in indexes
        for member in groups[index].members
    )
    supports: list[EvidenceMember] = []
    documents: set[str] = set()
    for member in ordered:
        if member.document_sha256 in documents:
            continue
        supports.append(member)
        documents.add(member.document_sha256)
        if len(supports) == 3:
            break
    return tuple(supports)


def _mmr_order(
    groups: tuple[_ExactGroup, ...],
    clustered: tuple[tuple[int, ...], ...],
    matrix: object,
    mmr_lambda: float,
    context: SubnarrativeContext,
    limit: int,
) -> tuple[int, ...]:
    remaining = set(range(len(clustered)))
    selected: list[int] = []
    ids = tuple(_cluster_id(context, groups[indexes[0]].representative) for indexes in clustered)
    while remaining and len(selected) < limit:
        def key(cluster_index: int) -> tuple[float, str]:
            representative_index = clustered[cluster_index][0]
            relevance = groups[representative_index].relevance
            novelty_penalty = max(
                (matrix[representative_index][clustered[selected_index][0]] for selected_index in selected),  # type: ignore[index]
                default=0.0,
            )
            score = mmr_lambda * relevance - (1.0 - mmr_lambda) * novelty_penalty
            return -score, ids[cluster_index]

        chosen = min(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def select_subnarrative_candidates(
    context: SubnarrativeContext,
    candidates: Sequence[SelectionCandidate],
    similarity: SimilarityProvider,
    policy: SelectionPolicy = SelectionPolicy(),
) -> SubnarrativeSelection:
    """Collapse, safely cluster, and diversify candidates for exactly one subnarrative."""
    if not isinstance(context, SubnarrativeContext):
        raise TypeError("context must be SubnarrativeContext")
    if not isinstance(policy, SelectionPolicy):
        raise TypeError("policy must be SelectionPolicy")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of SelectionCandidate")
    identity = _canonical_identity(getattr(similarity, "identity", None))
    unique_by_id: dict[str, SelectionCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, SelectionCandidate):
            raise TypeError("candidates must contain SelectionCandidate records")
        if candidate.topic_id != context.topic_id:
            raise ValueError("candidate topic_id does not match context")
        if candidate.subnarrative_id != context.subnarrative_id:
            raise ValueError("candidate subnarrative_id does not match context")
        previous = unique_by_id.setdefault(candidate.candidate_nugget_id, candidate)
        if previous != candidate:
            raise ValueError(f"conflicting duplicate candidate ID {candidate.candidate_nugget_id!r}")
    unique_candidates = tuple(unique_by_id.values())
    all_groups = _exact_groups(unique_candidates)
    groups = all_groups[: policy.precluster_limit]
    if groups:
        unique_texts = tuple(dict.fromkeys(group.text for group in groups))
        unique_matrix = _validated_matrix(similarity.cosine_matrix(unique_texts), len(unique_texts))
        text_index = {text: index for index, text in enumerate(unique_texts)}
        matrix = unique_matrix if len(unique_texts) == len(groups) else tuple(
            tuple(unique_matrix[text_index[left.text]][text_index[right.text]] for right in groups)  # type: ignore[index]
            for left in groups
        )
        clustered = _cluster_groups(groups, matrix, policy.semantic_threshold)
        order = _mmr_order(
            groups, clustered, matrix, policy.mmr_lambda, context, max(policy.budgets)
        )
    else:
        matrix = ()
        clustered = ()
        order = ()
    clusters: list[SemanticCluster] = []
    for cluster_index in order[:max(policy.budgets)]:
        group_indexes = clustered[cluster_index]
        representative = groups[group_indexes[0]].representative
        members = tuple(member for index in group_indexes for member in groups[index].members)
        supports = _selected_supports(groups, group_indexes)
        clusters.append(SemanticCluster(
            cluster_id=_cluster_id(context, representative),
            representative_candidate_nugget_id=representative.candidate_nugget_id,
            representative_text=representative.text,
            representative_raw_logit=representative.raw_logit,
            members=members,
            supports=supports,
            support_document_count=len(supports),
        ))
    cluster_ids = tuple(cluster.cluster_id for cluster in clusters)
    snapshots = tuple(
        BudgetSnapshot(
            budget=budget,
            cluster_ids=cluster_ids[:budget],
            exhausted=len(cluster_ids) < budget,
        )
        for budget in policy.budgets
    )
    return SubnarrativeSelection(
        schema_version=SELECTION_SCHEMA_VERSION,
        context=context,
        policy=policy,
        similarity_identity=identity,
        candidate_count=len(unique_candidates),
        exact_group_count=len(all_groups),
        precluster_count=len(groups),
        semantic_cluster_count=len(clustered),
        clusters=tuple(clusters),
        snapshots=snapshots,
    )

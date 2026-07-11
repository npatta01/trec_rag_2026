"""Deterministic sparse query decomposition and conservative PRF.

``det_sparse_v1`` is deliberately independent from the model-backed query
planner.  It operates only on exact source text, a caller-supplied frozen query
analyzer, and (for PRF) caller-supplied retrieval rows.  There are no model or
network clients in this module.

Offsets in this module are Python string offsets: zero-based, half-open Unicode
code-point offsets into the exact narrative supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, QueryAnalyzer
from trec_rag.query_planner import (
    TOKENIZER_VERSION as NARRATIVE_TOKENIZER_VERSION,
    NarrativeTokenTape,
    tokenize_narrative,
)


PLANNER_VERSION = "det_sparse_v1"
SPLITTER_VERSION = "det_sparse_exact_span_splitter_v1"
TOKENIZER_VERSION = NARRATIVE_TOKENIZER_VERSION
PRF_VERSION = "det_sparse_prf_v1"
MAX_FACETS = 4
MIN_FACET_UNIQUE_TOKENS = 3

PRF_FOREGROUND_RANKS = range(1, 6)
PRF_BACKGROUND_RANKS = range(6, 51)
PRF_MAX_TERMS = 2

# This tuple is part of the frozen v1 contract.  In particular, ordinary comma
# separated noun lists are not boundaries.
REQUEST_CUES = (
    "i",
    "could",
    "can",
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "whether",
    "which",
)
_COORDINATORS = ("and", "or", "also")
_REQUEST_CUE_RE = re.compile(
    rf"(?:(?:{'|'.join(_COORDINATORS)})\s+)?"
    rf"(?:{'|'.join(REQUEST_CUES)})(?=\W|$)",
    flags=re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SourceSpan:
    """An exact half-open Unicode code-point span in the source narrative."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class LexicalUnit:
    """One conservatively split source unit and its frozen analyzer tape."""

    unit_id: str
    source_span: SourceSpan
    token_tape_indices: tuple[int, ...]
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_tokens: tuple[str, ...]


@dataclass(frozen=True)
class FacetMergeAudit:
    action: Literal["analyzer_identical", "adjacent_min_tokens"]
    retained_group: tuple[str, ...]
    merged_group: tuple[str, ...]
    result_group: tuple[str, ...]
    combined_unique_analyzed_tokens: int


@dataclass(frozen=True)
class DeterministicFacet:
    facet_id: str
    variant_name: str
    coverage_unit_ids: tuple[str, ...]
    query_component_unit_ids: tuple[str, ...]
    coverage_source_spans: tuple[SourceSpan, ...]
    query_source_spans: tuple[SourceSpan, ...]
    query_text: str
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_tokens: tuple[str, ...]


@dataclass(frozen=True)
class PlannerFailure:
    code: str
    message: str


@dataclass(frozen=True)
class DeterministicSparsePlan:
    planner_version: str
    splitter_version: str
    tokenizer_version: str
    topic_id: str
    status: Literal["ok", "fallback"]
    narrative_sha256: str
    token_tape_sha256: str | None
    analyzer_token_sha256: str | None
    analyzer_fingerprint_sha256: str | None
    original_query_text: str
    lexical_units: tuple[LexicalUnit, ...]
    facets: tuple[DeterministicFacet, ...]
    merge_audit: tuple[FacetMergeAudit, ...]
    failure: PlannerFailure | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def query_variants(self) -> tuple[QueryVariant, ...]:
        """Return the permanent original stream first, then valid facets."""

        original = QueryVariant(
            topic_id=self.topic_id,
            variant_name=f"{PLANNER_VERSION}:original",
            query_text=self.original_query_text,
            source_type=f"{PLANNER_VERSION}_original",
        )
        if self.status == "fallback":
            return (original,)
        facets = tuple(
            QueryVariant(
                topic_id=self.topic_id,
                variant_name=facet.variant_name,
                query_text=facet.query_text,
                source_type=f"{PLANNER_VERSION}_facet",
            )
            for facet in self.facets
        )
        return (original, *facets)


@dataclass(frozen=True)
class PrfTermAudit:
    surface: str
    analyzed_tokens: tuple[str, ...]
    surface_foreground_document_frequency: int
    surface_top50_document_frequency: int
    foreground_document_frequency: int
    background_document_frequency: int
    top50_document_frequency: int
    foreground_occurrence_frequency: int
    top50_occurrence_frequency: int
    score: float | None
    disposition: Literal["selected", "rejected"]
    reasons: tuple[str, ...]
    selection_rank: int | None


@dataclass(frozen=True)
class PrfFailure:
    code: str
    message: str


@dataclass(frozen=True)
class PrfExpansion:
    planner_version: str
    prf_version: str
    tokenizer_version: str
    status: Literal["ok", "no_expansion", "failure"]
    topic_id: str | None
    base_query_text: str
    query_text: str | None
    selected_terms: tuple[str, ...]
    base_source_sha256: str
    base_query_sha256: str
    rendered_query_sha256: str | None
    token_tape_sha256: str | None
    analyzer_token_sha256: str | None
    analyzer_fingerprint_sha256: str | None
    raw_response_sha256: str
    retrieval_request_key: str | None
    retrieval_candidates_sha256: str | None
    retriever_version: str | None
    provenance_verified: bool
    term_audit: tuple[PrfTermAudit, ...]
    failure: PrfFailure | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def query_variant(self, *, variant_name: str | None = None) -> QueryVariant | None:
        """Return the expansion only when it is distinct from the base query."""

        if self.status != "ok" or self.query_text is None or self.topic_id is None:
            return None
        return QueryVariant(
            topic_id=self.topic_id,
            variant_name=variant_name or f"{PLANNER_VERSION}:prf",
            query_text=self.query_text,
            source_type=f"{PLANNER_VERSION}_prf",
        )


@dataclass
class _FacetGroup:
    coverage_indices: list[int]
    component_indices: list[int]


@dataclass
class _SurfaceStats:
    surface: str
    foreground_docs: set[int]
    background_docs: set[int]
    foreground_occurrences: int = 0
    top50_occurrences: int = 0
    saw_lowercase_source: bool = False


class _PlannerInvalid(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint_sha256(fingerprint: AnalyzerFingerprint) -> str:
    return _canonical_sha256(fingerprint.to_dict())


def _token_tape_sha256(token_tape: NarrativeTokenTape) -> str:
    """Hash the complete exact-offset tape, including tokenizer provenance."""

    return _canonical_sha256(token_tape.to_dict())


def _analyzer_token_sha256(analyzed: AnalyzedQuery) -> str:
    """Hash analyzer occurrences separately from the source token tape."""

    return _canonical_sha256(list(analyzed.tokens))


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _trimmed_span(narrative: str, start: int, end: int) -> SourceSpan | None:
    while start < end and narrative[start].isspace():
        start += 1
    while end > start and narrative[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    return SourceSpan(start=start, end=end, text=narrative[start:end])


def _cue_after_comma(narrative: str, comma_end: int) -> bool:
    continuation = narrative[comma_end:].lstrip()
    return _REQUEST_CUE_RE.match(continuation) is not None


def _coordinator_continuation_end(narrative: str, index: int) -> int | None:
    """Return the end of a coordinator starting a request-cue continuation."""

    if index > 0 and not narrative[index - 1].isspace():
        return None
    for coordinator in _COORDINATORS:
        end = index + len(coordinator)
        if narrative[index:end].lower() != coordinator:
            continue
        if end >= len(narrative) or not narrative[end].isspace():
            continue
        following = narrative[end:].lstrip()
        cue_match = re.match(
            rf"(?:{'|'.join(REQUEST_CUES)})(?=\W|$)",
            following,
            flags=re.IGNORECASE,
        )
        if cue_match is not None:
            return end
    return None


def split_deterministic_units(narrative: str) -> tuple[SourceSpan, ...]:
    """Apply the frozen ``det_sparse_v1`` source-only boundary rules.

    A frozen punctuation boundary is recognized only at end of text or before
    whitespace.  This deterministic guard avoids splitting decimals and most
    embedded acronym punctuation.
    """

    if not isinstance(narrative, str):
        raise TypeError("narrative must be text")
    spans: list[SourceSpan] = []
    unit_start = 0
    index = 0
    length = len(narrative)
    while index < length:
        character = narrative[index]
        boundary_end: int | None = None
        if character in ".?!;:":
            punctuation_end = index + 1
            while (
                punctuation_end < length
                and narrative[punctuation_end] in ".?!;:"
            ):
                punctuation_end += 1
            if punctuation_end == length or narrative[punctuation_end].isspace():
                boundary_end = punctuation_end
        elif character == "," and _cue_after_comma(narrative, index + 1):
            boundary_end = index + 1

        # A coordinator-led request continuation is a boundary even without a
        # comma: ``explain X and how Y`` -> ``explain X`` / ``and how Y``.
        coordinator_end = _coordinator_continuation_end(narrative, index)
        if boundary_end is None and coordinator_end is not None:
            span = _trimmed_span(narrative, unit_start, index)
            if span is not None:
                spans.append(span)
            unit_start = index
            index = coordinator_end
            continue

        if boundary_end is not None:
            span = _trimmed_span(narrative, unit_start, boundary_end)
            if span is not None:
                spans.append(span)
            unit_start = boundary_end
            index = boundary_end
            continue
        index += 1

    final_span = _trimmed_span(narrative, unit_start, length)
    if final_span is not None:
        spans.append(final_span)
    return tuple(spans)


def _analyze_checked(
    analyzer: QueryAnalyzer,
    text: str,
    fingerprint: AnalyzerFingerprint,
) -> AnalyzedQuery:
    analyzed = analyzer.analyze(text)
    if analyzed.fingerprint != fingerprint:
        raise _PlannerInvalid(
            "analyzer_fingerprint_changed",
            "query analyzer fingerprint changed during deterministic planning",
        )
    return analyzed


def _component_indices(groups: Sequence[_FacetGroup]) -> list[int]:
    result: list[int] = []
    for group in groups:
        for index in group.component_indices:
            if index not in result:
                result.append(index)
    return sorted(result)


def _render_group_text(group: _FacetGroup, units: Sequence[LexicalUnit]) -> str:
    components: list[str] = []
    for index in group.component_indices:
        text = units[index].source_span.text
        if text not in components:
            components.append(text)
    return " ".join(components)


def _analyze_group(
    group: _FacetGroup,
    units: Sequence[LexicalUnit],
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
) -> AnalyzedQuery:
    return _analyze_checked(analyzer, _render_group_text(group, units), fingerprint)


def _merge_identical_groups(
    groups: list[_FacetGroup],
    units: Sequence[LexicalUnit],
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
    audit: list[FacetMergeAudit],
) -> None:
    index = 0
    while index < len(groups):
        retained_tokens = _analyze_group(
            groups[index], units, analyzer, fingerprint
        ).tokens
        candidate_index = index + 1
        while candidate_index < len(groups):
            candidate_tokens = _analyze_group(
                groups[candidate_index], units, analyzer, fingerprint
            ).tokens
            if candidate_tokens != retained_tokens:
                candidate_index += 1
                continue
            retained = groups[index]
            duplicate = groups.pop(candidate_index)
            old_retained_ids = tuple(
                units[item].unit_id for item in retained.coverage_indices
            )
            duplicate_ids = tuple(
                units[item].unit_id for item in duplicate.coverage_indices
            )
            retained.coverage_indices = sorted(
                set(retained.coverage_indices + duplicate.coverage_indices)
            )
            # Keep the earliest literal rendering.  Appending an
            # analyzer-identical alternate would recreate a duplicate query.
            audit.append(
                FacetMergeAudit(
                    action="analyzer_identical",
                    retained_group=old_retained_ids,
                    merged_group=duplicate_ids,
                    result_group=tuple(
                        units[item].unit_id for item in retained.coverage_indices
                    ),
                    combined_unique_analyzed_tokens=len(set(retained_tokens)),
                )
            )
        index += 1


def _merge_to_limit(
    groups: list[_FacetGroup],
    units: Sequence[LexicalUnit],
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
    audit: list[FacetMergeAudit],
) -> None:
    while len(groups) > MAX_FACETS:
        choices: list[tuple[int, int, _FacetGroup]] = []
        for index in range(len(groups) - 1):
            merged = _FacetGroup(
                coverage_indices=sorted(
                    set(
                        groups[index].coverage_indices
                        + groups[index + 1].coverage_indices
                    )
                ),
                component_indices=_component_indices(
                    (groups[index], groups[index + 1])
                ),
            )
            unique_count = len(
                _analyze_group(merged, units, analyzer, fingerprint).unique_tokens
            )
            choices.append((unique_count, index, merged))
        unique_count, merge_index, merged = min(
            choices, key=lambda row: (row[0], row[1])
        )
        left = groups[merge_index]
        right = groups[merge_index + 1]
        audit.append(
            FacetMergeAudit(
                action="adjacent_min_tokens",
                retained_group=tuple(
                    units[item].unit_id for item in left.coverage_indices
                ),
                merged_group=tuple(
                    units[item].unit_id for item in right.coverage_indices
                ),
                result_group=tuple(
                    units[item].unit_id for item in merged.coverage_indices
                ),
                combined_unique_analyzed_tokens=unique_count,
            )
        )
        groups[merge_index : merge_index + 2] = [merged]


def _fallback_plan(
    *,
    topic_id: str,
    narrative: str,
    token_tape_sha256: str | None,
    analyzer_token_sha256: str | None,
    analyzer_fingerprint_sha256: str | None,
    lexical_units: Sequence[LexicalUnit],
    merge_audit: Sequence[FacetMergeAudit],
    code: str,
    message: str,
) -> DeterministicSparsePlan:
    return DeterministicSparsePlan(
        planner_version=PLANNER_VERSION,
        splitter_version=SPLITTER_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        topic_id=topic_id,
        status="fallback",
        narrative_sha256=_text_sha256(narrative),
        token_tape_sha256=token_tape_sha256,
        analyzer_token_sha256=analyzer_token_sha256,
        analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
        original_query_text=narrative,
        lexical_units=tuple(lexical_units),
        facets=(),
        merge_audit=tuple(merge_audit),
        failure=PlannerFailure(code=code, message=message),
    )


def build_deterministic_sparse_plan(
    *,
    topic_id: str,
    narrative: str,
    query_analyzer: QueryAnalyzer,
) -> DeterministicSparsePlan:
    """Build an auditable model-free facet plan or an explicit fallback."""

    if not isinstance(topic_id, str) or not topic_id:
        raise ValueError("topic_id must be non-empty text")
    if not isinstance(narrative, str):
        raise TypeError("narrative must be text")

    units: list[LexicalUnit] = []
    merge_audit: list[FacetMergeAudit] = []
    token_tape = tokenize_narrative(narrative)
    token_hash: str | None = _token_tape_sha256(token_tape)
    analyzer_token_hash: str | None = None
    fingerprint_hash: str | None = None
    try:
        if not narrative.strip():
            raise _PlannerInvalid("empty_narrative", "narrative is empty")
        base_analysis = query_analyzer.analyze(narrative)
        fingerprint = base_analysis.fingerprint
        analyzer_token_hash = _analyzer_token_sha256(base_analysis)
        fingerprint_hash = _fingerprint_sha256(fingerprint)
        if not base_analysis.tokens:
            raise _PlannerInvalid(
                "empty_token_tape", "narrative has no analyzer tokens"
            )

        spans = split_deterministic_units(narrative)
        if not spans:
            raise _PlannerInvalid("no_source_units", "no nonempty source unit found")
        token_owners: dict[int, list[int]] = {
            token_index: [] for token_index in range(token_tape.token_count)
        }
        for unit_index, span in enumerate(spans):
            for token_index, token in enumerate(token_tape.tokens):
                if span.start <= token.start_char and token.end_char <= span.end:
                    token_owners[token_index].append(unit_index)
        invalid_token_owners = {
            token_index: owners
            for token_index, owners in token_owners.items()
            if len(owners) != 1
        }
        if invalid_token_owners:
            raise _PlannerInvalid(
                "token_tape_coverage_mismatch",
                "every exact-offset narrative token must belong to exactly one source unit",
            )
        for index, span in enumerate(spans, start=1):
            if narrative[span.start : span.end] != span.text:
                raise _PlannerInvalid(
                    "source_span_mismatch", "source span does not resolve exactly"
                )
            analyzed = _analyze_checked(
                query_analyzer, span.text, fingerprint
            )
            units.append(
                LexicalUnit(
                    unit_id=f"u{index:03d}",
                    source_span=span,
                    token_tape_indices=tuple(
                        token_index
                        for token_index, owners in token_owners.items()
                        if owners == [index - 1]
                    ),
                    analyzed_tokens=analyzed.tokens,
                    unique_analyzed_tokens=analyzed.unique_tokens,
                )
            )

        partitioned_tape = tuple(
            token for unit in units for token in unit.analyzed_tokens
        )
        if partitioned_tape != base_analysis.tokens:
            raise _PlannerInvalid(
                "lexical_coverage_mismatch",
                "split source units do not partition the narrative analyzer tape exactly",
            )

        # The first unit is both its own facet and shared parent context.
        groups = [_FacetGroup([0], [0])]
        groups.extend(_FacetGroup([index], [0, index]) for index in range(1, len(units)))
        _merge_identical_groups(
            groups, units, query_analyzer, fingerprint, merge_audit
        )
        _merge_to_limit(
            groups, units, query_analyzer, fingerprint, merge_audit
        )
        # An adjacent merge can itself create an analyzer-identical rendering;
        # apply the frozen duplicate rule once more before validation.
        _merge_identical_groups(
            groups, units, query_analyzer, fingerprint, merge_audit
        )

        assigned_units = [
            index for group in groups for index in group.coverage_indices
        ]
        if sorted(assigned_units) != list(range(len(units))) or len(
            assigned_units
        ) != len(set(assigned_units)):
            raise _PlannerInvalid(
                "facet_coverage_partition",
                "every source unit must belong to exactly one final facet",
            )

        base_unique = set(base_analysis.unique_tokens)
        facets: list[DeterministicFacet] = []
        rendered_token_tapes: set[tuple[str, ...]] = set()
        for index, group in enumerate(groups, start=1):
            rendered = _render_group_text(group, units)
            analyzed = _analyze_group(
                group, units, query_analyzer, fingerprint
            )
            if len(analyzed.unique_tokens) < MIN_FACET_UNIQUE_TOKENS:
                raise _PlannerInvalid(
                    "facet_too_short",
                    f"facet {index} has fewer than three unique analyzer tokens",
                )
            if not set(analyzed.unique_tokens).issubset(base_unique):
                raise _PlannerInvalid(
                    "facet_outside_narrative",
                    f"facet {index} contains an analyzer token absent from the narrative",
                )
            if analyzed.tokens in rendered_token_tapes:
                raise _PlannerInvalid(
                    "duplicate_facet_query",
                    "analyzer-identical facet queries remain after deterministic merging",
                )
            rendered_token_tapes.add(analyzed.tokens)
            facet_id = f"f{index:02d}"
            facets.append(
                DeterministicFacet(
                    facet_id=facet_id,
                    variant_name=f"{PLANNER_VERSION}:facet:{facet_id}",
                    coverage_unit_ids=tuple(
                        units[item].unit_id for item in group.coverage_indices
                    ),
                    query_component_unit_ids=tuple(
                        units[item].unit_id for item in group.component_indices
                    ),
                    coverage_source_spans=tuple(
                        units[item].source_span for item in group.coverage_indices
                    ),
                    query_source_spans=tuple(
                        units[item].source_span for item in group.component_indices
                    ),
                    query_text=rendered,
                    analyzed_tokens=analyzed.tokens,
                    unique_analyzed_tokens=analyzed.unique_tokens,
                )
            )

        return DeterministicSparsePlan(
            planner_version=PLANNER_VERSION,
            splitter_version=SPLITTER_VERSION,
            tokenizer_version=TOKENIZER_VERSION,
            topic_id=topic_id,
            status="ok",
            narrative_sha256=_text_sha256(narrative),
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_token_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_query_text=narrative,
            lexical_units=tuple(units),
            facets=tuple(facets),
            merge_audit=tuple(merge_audit),
            failure=None,
        )
    except _PlannerInvalid as error:
        return _fallback_plan(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_token_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            lexical_units=units,
            merge_audit=merge_audit,
            code=error.code,
            message=str(error),
        )
    except Exception as error:  # Analyzer failures are represented, not hidden.
        return _fallback_plan(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_token_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            lexical_units=units,
            merge_audit=merge_audit,
            code="analyzer_error",
            message=f"{type(error).__name__}: {error}",
        )


# Short alias for callers that already use planner-style ``plan_*`` names.
plan_deterministic_sparse_query = build_deterministic_sparse_plan


def _strip_edge_punctuation(raw: str) -> str:
    start = 0
    end = len(raw)
    while start < end and unicodedata.category(raw[start])[0] in {"P", "S"}:
        start += 1
    while end > start and unicodedata.category(raw[end - 1])[0] in {"P", "S"}:
        end -= 1
    return raw[start:end]


def _surface_occurrences(text: str) -> Iterable[tuple[str, bool]]:
    for raw in text.split():
        stripped = _strip_edge_punctuation(raw)
        if not stripped:
            continue
        lowered = stripped.lower()
        # A lowercase source occurrence is intentionally stricter than merely
        # lowercasing for comparison; it suppresses uppercase-only names.
        yield lowered, stripped == lowered


def _valid_raw_response_sha256(value: str) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _prf_failure(
    *,
    base_query_text: str,
    raw_response_sha256: str,
    code: str,
    message: str,
    topic_id: str | None = None,
    token_tape_sha256: str | None = None,
    analyzer_token_sha256: str | None = None,
    analyzer_fingerprint_sha256: str | None = None,
    term_audit: Sequence[PrfTermAudit] = (),
) -> PrfExpansion:
    if token_tape_sha256 is None:
        token_tape_sha256 = _token_tape_sha256(
            tokenize_narrative(base_query_text)
        )
    return PrfExpansion(
        planner_version=PLANNER_VERSION,
        prf_version=PRF_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        status="failure",
        topic_id=topic_id,
        base_query_text=base_query_text,
        query_text=None,
        selected_terms=(),
        base_source_sha256=_text_sha256(base_query_text),
        base_query_sha256=_text_sha256(base_query_text),
        rendered_query_sha256=None,
        token_tape_sha256=token_tape_sha256,
        analyzer_token_sha256=analyzer_token_sha256,
        analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
        raw_response_sha256=raw_response_sha256,
        retrieval_request_key=None,
        retrieval_candidates_sha256=None,
        retriever_version=None,
        provenance_verified=False,
        term_audit=tuple(term_audit),
        failure=PrfFailure(code=code, message=message),
    )


def build_prf_expansion(
    *,
    base_query_text: str,
    candidates: Sequence[RetrievedCandidate],
    query_analyzer: QueryAnalyzer,
    raw_response_sha256: str,
    existing_query_texts: Sequence[str] = (),
) -> PrfExpansion:
    """Select up to two conservative contrastive PRF terms.

    The function consumes exactly ranks 1--50 from one retrieval stream.  It
    never emits the unchanged base query or a query already supplied through
    ``existing_query_texts``.
    """

    if not isinstance(base_query_text, str) or not base_query_text.strip():
        raise ValueError("base_query_text must be non-empty text")
    if not _valid_raw_response_sha256(raw_response_sha256):
        raise ValueError("raw_response_sha256 must be 64 lowercase hex characters")

    top_by_rank: dict[int, RetrievedCandidate] = {}
    for candidate in candidates:
        if 1 <= candidate.rank <= 50:
            if candidate.rank in top_by_rank:
                return _prf_failure(
                    base_query_text=base_query_text,
                    raw_response_sha256=raw_response_sha256,
                    code="duplicate_rank",
                    message=f"retrieval stream contains duplicate rank {candidate.rank}",
                )
            top_by_rank[candidate.rank] = candidate
    missing_ranks = sorted(set(range(1, 51)) - set(top_by_rank))
    if missing_ranks:
        return _prf_failure(
            base_query_text=base_query_text,
            raw_response_sha256=raw_response_sha256,
            code="insufficient_top50",
            message="retrieval rows do not contain every rank from 1 through 50",
        )

    top50 = [top_by_rank[rank] for rank in range(1, 51)]
    stream_keys = {
        (
            row.topic_id,
            row.variant_name,
            row.retriever_name,
            row.query_text,
        )
        for row in top50
    }
    if len(stream_keys) != 1:
        return _prf_failure(
            base_query_text=base_query_text,
            raw_response_sha256=raw_response_sha256,
            code="mixed_retrieval_streams",
            message="PRF ranks 1-50 must come from one retrieval stream",
        )
    if top50[0].query_text != base_query_text:
        return _prf_failure(
            base_query_text=base_query_text,
            raw_response_sha256=raw_response_sha256,
            topic_id=top50[0].topic_id,
            code="base_query_mismatch",
            message="retrieval stream query text does not equal the PRF base query",
        )
    if len({row.docid for row in top50}) != 50:
        return _prf_failure(
            base_query_text=base_query_text,
            raw_response_sha256=raw_response_sha256,
            code="duplicate_top50_docid",
            message="PRF ranks 1-50 must contain 50 distinct documents",
        )

    topic_id = top50[0].topic_id
    token_hash: str | None = _token_tape_sha256(
        tokenize_narrative(base_query_text)
    )
    analyzer_token_hash: str | None = None
    fingerprint_hash: str | None = None
    audits: list[PrfTermAudit] = []
    try:
        base_analysis = query_analyzer.analyze(base_query_text)
        fingerprint = base_analysis.fingerprint
        analyzer_token_hash = _analyzer_token_sha256(base_analysis)
        fingerprint_hash = _fingerprint_sha256(fingerprint)
        base_tokens = set(base_analysis.unique_tokens)

        stats_by_surface: dict[str, _SurfaceStats] = {}
        for row in top50:
            document_seen: set[str] = set()
            for surface, is_lowercase_source in _surface_occurrences(row.text):
                stats = stats_by_surface.setdefault(
                    surface,
                    _SurfaceStats(surface, set(), set()),
                )
                if is_lowercase_source:
                    stats.saw_lowercase_source = True
                else:
                    # Uppercase occurrences are audited but never contribute to
                    # PRF frequency estimates.
                    continue
                stats.top50_occurrences += 1
                if row.rank <= 5:
                    stats.foreground_occurrences += 1
                document_seen.add(surface)
            for surface in document_seen:
                stats = stats_by_surface[surface]
                if row.rank <= 5:
                    stats.foreground_docs.add(row.rank)
                else:
                    stats.background_docs.add(row.rank)

        intermediate: list[dict[str, object]] = []
        for surface in sorted(stats_by_surface):
            stats = stats_by_surface[surface]
            reasons: list[str] = []
            analyzed_tokens: tuple[str, ...] = ()
            if not stats.saw_lowercase_source:
                reasons.append("no_lowercase_source_occurrence")
            if not 3 <= len(surface) <= 24:
                reasons.append("surface_length_outside_3_24")
            if not surface or not all(character.isalpha() for character in surface):
                reasons.append("surface_not_unicode_letters")
            if not reasons:
                analyzed = _analyze_checked(query_analyzer, surface, fingerprint)
                analyzed_tokens = analyzed.tokens
                if len(analyzed_tokens) != 1:
                    reasons.append("analyzer_token_count_not_one")
            if len(analyzed_tokens) == 1 and analyzed_tokens[0] in base_tokens:
                reasons.append("analyzed_form_present_in_base")
            intermediate.append(
                {
                    "surface": surface,
                    "analyzed_tokens": analyzed_tokens,
                    "surface_foreground_docs": set(stats.foreground_docs),
                    "surface_background_docs": set(stats.background_docs),
                    "foreground_df": 0,
                    "background_df": 0,
                    "top50_df": 0,
                    "foreground_occurrences": stats.foreground_occurrences,
                    "top50_occurrences": stats.top50_occurrences,
                    "score": None,
                    "reasons": reasons,
                }
            )

        # Score analyzed forms, not literal spellings: document-frequency sets
        # are unioned across all structurally valid surfaces for that form.
        by_analyzed_form: dict[str, list[dict[str, object]]] = {}
        for row in intermediate:
            analyzed_tokens = row["analyzed_tokens"]
            reasons = row["reasons"]
            assert isinstance(analyzed_tokens, tuple)
            assert isinstance(reasons, list)
            if len(analyzed_tokens) == 1 and not reasons:
                by_analyzed_form.setdefault(analyzed_tokens[0], []).append(row)
        for rows in by_analyzed_form.values():
            foreground_docs: set[int] = set()
            background_docs: set[int] = set()
            for row in rows:
                foreground_docs.update(row["surface_foreground_docs"])
                background_docs.update(row["surface_background_docs"])
            foreground_df = len(foreground_docs)
            background_df = len(background_docs)
            top50_df = foreground_df + background_df
            score = math.log((foreground_df + 0.5) / 6.0) - math.log(
                (background_df + 0.5) / 46.0
            )
            preferred = min(
                rows,
                key=lambda row: (
                    -len(row["surface_foreground_docs"]),
                    -int(row["top50_occurrences"]),
                    str(row["surface"]),
                ),
            )
            for row in rows:
                row["foreground_df"] = foreground_df
                row["background_df"] = background_df
                row["top50_df"] = top50_df
                row["score"] = score
                reasons = row["reasons"]
                assert isinstance(reasons, list)
                if foreground_df < 2:
                    reasons.append("foreground_df_below_2")
                if top50_df >= 40:
                    reasons.append("top50_df_at_least_40")
                if score <= 0.0:
                    reasons.append("nonpositive_log_contrast")
                if row is not preferred:
                    reasons.append("surface_not_preferred_for_analyzed_form")

        # Rows which never formed a valid single analyzed form still receive
        # their literal surface counts and explicit frequency-gate audit.
        for row in intermediate:
            if row["score"] is not None:
                continue
            surface_foreground_docs = row["surface_foreground_docs"]
            surface_background_docs = row["surface_background_docs"]
            assert isinstance(surface_foreground_docs, set)
            assert isinstance(surface_background_docs, set)
            row["foreground_df"] = len(surface_foreground_docs)
            row["background_df"] = len(surface_background_docs)
            row["top50_df"] = len(surface_foreground_docs) + len(
                surface_background_docs
            )
            reasons = row["reasons"]
            assert isinstance(reasons, list)
            if len(surface_foreground_docs) < 2:
                reasons.append("foreground_df_below_2")
            if int(row["top50_df"]) >= 40:
                reasons.append("top50_df_at_least_40")

        eligible = [row for row in intermediate if not row["reasons"]]
        eligible.sort(
            key=lambda row: (
                -float(row["score"]),
                -int(row["foreground_df"]),
                str(row["analyzed_tokens"][0]),
            )
        )

        existing = {" ".join(base_query_text.split())}
        existing.update(" ".join(query.split()) for query in existing_query_texts)
        selected: list[dict[str, object]] = []
        for row in eligible:
            reasons = row["reasons"]
            assert isinstance(reasons, list)
            if len(selected) >= PRF_MAX_TERMS:
                reasons.append("selection_limit_2")
                continue
            proposed_terms = [str(item["surface"]) for item in selected]
            proposed_terms.append(str(row["surface"]))
            proposed_query = " ".join(
                (base_query_text.strip(), *proposed_terms)
            )
            if " ".join(proposed_query.split()) in existing:
                reasons.append("duplicate_query")
                continue
            selected.append(row)

        selection_positions = {
            str(row["surface"]): index
            for index, row in enumerate(selected, start=1)
        }
        for row in sorted(intermediate, key=lambda item: str(item["surface"])):
            reasons = tuple(str(reason) for reason in row["reasons"])
            surface = str(row["surface"])
            audits.append(
                PrfTermAudit(
                    surface=surface,
                    analyzed_tokens=tuple(row["analyzed_tokens"]),
                    surface_foreground_document_frequency=len(
                        row["surface_foreground_docs"]
                    ),
                    surface_top50_document_frequency=(
                        len(row["surface_foreground_docs"])
                        + len(row["surface_background_docs"])
                    ),
                    foreground_document_frequency=int(row["foreground_df"]),
                    background_document_frequency=int(row["background_df"]),
                    top50_document_frequency=int(row["top50_df"]),
                    foreground_occurrence_frequency=int(
                        row["foreground_occurrences"]
                    ),
                    top50_occurrence_frequency=int(row["top50_occurrences"]),
                    score=(float(row["score"]) if row["score"] is not None else None),
                    disposition=("selected" if surface in selection_positions else "rejected"),
                    reasons=reasons,
                    selection_rank=selection_positions.get(surface),
                )
            )

        selected_terms = tuple(str(row["surface"]) for row in selected)
        query_text = (
            " ".join((base_query_text.strip(), *selected_terms))
            if selected_terms
            else None
        )
        if query_text is not None and " ".join(query_text.split()) in existing:
            raise _PlannerInvalid(
                "duplicate_prf_query", "PRF emitted a duplicate query"
            )
        return PrfExpansion(
            planner_version=PLANNER_VERSION,
            prf_version=PRF_VERSION,
            tokenizer_version=TOKENIZER_VERSION,
            status="ok" if query_text is not None else "no_expansion",
            topic_id=topic_id,
            base_query_text=base_query_text,
            query_text=query_text,
            selected_terms=selected_terms,
            base_source_sha256=_text_sha256(base_query_text),
            base_query_sha256=_text_sha256(base_query_text),
            rendered_query_sha256=(
                _text_sha256(query_text) if query_text is not None else None
            ),
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_token_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            raw_response_sha256=raw_response_sha256,
            retrieval_request_key=None,
            retrieval_candidates_sha256=None,
            retriever_version=None,
            provenance_verified=False,
            term_audit=tuple(audits),
            failure=None,
        )
    except Exception as error:
        code = error.code if isinstance(error, _PlannerInvalid) else "analyzer_error"
        return _prf_failure(
            base_query_text=base_query_text,
            raw_response_sha256=raw_response_sha256,
            topic_id=topic_id,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_token_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            term_audit=audits,
            code=code,
            message=f"{type(error).__name__}: {error}",
        )


expand_query_with_prf = build_prf_expansion

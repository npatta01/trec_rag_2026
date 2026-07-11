"""Deterministic recurrent-anchor sparse decomposition (v3).

This module is deliberately offline and source-only.  It consumes an exact
narrative plus a caller-supplied, fingerprinted Lucene-equivalent analyzer.  It
does not contain topic data, retrieval, qrels, model, reranker, cache, or agent
clients.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Literal, Mapping, Sequence

from trec_rag.deterministic_sparse import (
    SPLITTER_VERSION,
    TOKENIZER_VERSION,
    SourceSpan,
    split_deterministic_units,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, QueryAnalyzer
from trec_rag.query_planner import NarrativeTokenTape, tokenize_narrative


PLANNER_VERSION = "det_sparse_v3"
RENDERER_VERSION = "det_sparse_recurrent_anchor_renderer_v3"
ANCHOR_SELECTOR_VERSION = "det_sparse_cross_unit_recurrent_anchor_v1"
SELECTION_VERSION = "det_sparse_anchor_critical_quantile_selection_v3"
CONVERSATIONAL_SURFACE_VERSION = "det_sparse_conversational_surfaces_v1"
NORMALIZATION_VERSION = "unicode_nfkc_casefold_v1"
MAX_FACETS = 4
MIN_PARENT_UNIQUE_TERMS = 4


_CONVERSATIONAL_TEXT = """
a, about, also, an, and, answer, are, ask, asking, at, be, been, being, but,
by, can, could, curious, deeper, describe, description, detailed, did, discuss,
discussion, do, does, explain, explanation, finally, for, from, gain, had, has,
have, he, her, hers, him, his, hope, hoping, how, i, i'd, i’d, i'm, i’m,
identify, in, information, interest, interested, is, it, its, know, knowing,
learn, learning, like, liked, list, look, looking, may, me, might, must, my, of,
on, or, our, ours, overview, please, provide, question, report, say, shall, she,
should, tell, that, the, their, theirs, them, these, they, this, those, to,
understand, understanding, want, wanted, wants, was, we, we're, we’re, were,
what, why, will, with, would, you, your, yours
"""
CONVERSATIONAL_SURFACES = tuple(
    item.strip()
    for item in _CONVERSATIONAL_TEXT.replace("\n", " ").split(",")
    if item.strip()
)
if len(CONVERSATIONAL_SURFACES) != 114 or len(set(CONVERSATIONAL_SURFACES)) != 114:
    raise AssertionError("v3 conversational surface inventory must contain 114 rows")


BM25Signature = tuple[tuple[str, int], ...]
CoreScore = tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class PlannerFailureV3:
    code: str
    message: str


@dataclass(frozen=True)
class AlignedOccurrenceV3:
    occurrence_id: str
    analyzer_term: str
    token_id: int
    exact_surface: str
    source_span: SourceSpan
    unit_id: str
    conversational: bool


@dataclass(frozen=True)
class AlignedTokenV3:
    token_id: int
    exact_surface: str
    source_span: SourceSpan
    unit_id: str
    analyzer_tokens: tuple[str, ...]
    conversational_surface: bool


@dataclass(frozen=True)
class LexicalUnitV3:
    unit_id: str
    source_span: SourceSpan
    token_ids: tuple[int, ...]
    analyzer_tokens: tuple[str, ...]
    unique_analyzer_terms: tuple[str, ...]
    bm25_signature: BM25Signature
    aligned_occurrence_ids: tuple[str, ...]
    eligible_unique_terms: tuple[str, ...]


@dataclass(frozen=True)
class ConversationalSurfaceProjectionV3:
    source_surface: str
    normalized_surface: str
    analyzer_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ConversationalInventoryAuditV3:
    version: str
    normalization_version: str
    source_surfaces: tuple[str, ...]
    normalized_unique_surfaces: tuple[str, ...]
    source_surfaces_sha256: str
    normalized_surfaces_sha256: str
    projections: tuple[ConversationalSurfaceProjectionV3, ...]
    projection_sha256: str
    projected_analyzer_terms: tuple[str, ...]
    analyzer_fingerprint_sha256: str
    excluded_occurrence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecurrenceTermAuditV3:
    analyzer_term: str
    parent_tf: int
    child_df: int
    child_tf: int
    parent_occurrence_ids: tuple[str, ...]
    child_occurrence_ids: tuple[str, ...]
    supporting_child_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvidenceIdentityV3:
    narrative_sha256: str
    start: int
    end: int
    text_sha256: str
    token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        # Field names and value types are the frozen hash payload.
        return {
            "end": self.end,
            "narrative_sha256": self.narrative_sha256,
            "start": self.start,
            "text_sha256": self.text_sha256,
            "token_ids": list(self.token_ids),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_compact_json(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class RecurrentCoreHullV3:
    evidence: CandidateEvidenceIdentityV3
    evidence_sha256: str
    source_span: SourceSpan
    exact_text: str
    token_ids: tuple[int, ...]
    recurrent_terms: tuple[str, ...]


@dataclass(frozen=True)
class AnchorCandidateV3:
    evidence: CandidateEvidenceIdentityV3
    evidence_sha256: str
    source_span: SourceSpan
    exact_text: str
    token_ids: tuple[int, ...]
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_terms: tuple[str, ...]
    bm25_signature: BM25Signature
    recurrent_terms: tuple[str, ...]
    joint_child_df: int
    recurrence_mass: int
    eligible_nonrecurrent_occurrence_count: int
    conversational_occurrence_count: int
    core_score: CoreScore
    admissible: bool
    rejection_reasons: tuple[str, ...]
    minimal_hulls: tuple[RecurrentCoreHullV3, ...]
    selected: bool


@dataclass(frozen=True)
class RecurrentCoreAuditV3:
    terms: tuple[str, ...]
    core_sha256: str
    candidate_evidence_sha256s: tuple[str, ...]
    minimal_hulls: tuple[RecurrentCoreHullV3, ...]
    maximal: bool


@dataclass(frozen=True)
class SelectedAnchorV3:
    candidate_evidence_sha256: str
    evidence_sha256: str
    core_sha256: str
    source_span: SourceSpan
    token_ids: tuple[int, ...]
    query_text: str
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_terms: tuple[str, ...]
    bm25_signature: BM25Signature
    core_terms: tuple[str, ...]
    core_score: CoreScore


@dataclass(frozen=True)
class CriticalityAuditV3:
    owner_id: str
    coverage_unit_ids: tuple[str, ...]
    source_span: SourceSpan
    full_analyzed_tokens: tuple[str, ...]
    full_unique_terms: tuple[str, ...]
    eligible_unique_terms: tuple[str, ...]
    full_core_intersection: tuple[str, ...]
    eligible_core_intersection: tuple[str, ...]
    missing_full_core_terms: tuple[str, ...]
    label: Literal["anchorless", "partial", "complete"]


@dataclass(frozen=True)
class ChildMergeAuditV3:
    action: Literal["adjacent_child_min_unique_terms"]
    left_unit_ids: tuple[str, ...]
    right_unit_ids: tuple[str, ...]
    result_unit_ids: tuple[str, ...]
    left_source_span: SourceSpan
    right_source_span: SourceSpan
    combined_source_span: SourceSpan
    combined_analyzed_tokens: tuple[str, ...]
    combined_unique_analyzed_terms: tuple[str, ...]


@dataclass(frozen=True)
class DeterministicFacetV3:
    facet_id: str
    variant_name: str
    coverage_unit_ids: tuple[str, ...]
    coverage_source_span: SourceSpan
    query_source_spans: tuple[SourceSpan, ...]
    query_text: str
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_terms: tuple[str, ...]
    bm25_signature: BM25Signature
    child_payload_terms: tuple[str, ...]

    @property
    def unique_analyzed_tokens(self) -> tuple[str, ...]:
        """Compatibility spelling used by earlier sparse plan consumers."""

        return self.unique_analyzed_terms


@dataclass(frozen=True)
class FacetInvariantAuditV3:
    facet_id: str
    exact_distinct_from_original: bool
    signature_distinct_from_original: bool
    occurrence_count: int
    original_occurrence_count: int
    strict_original_submultiset: bool
    reconstruction_exact: bool


@dataclass(frozen=True)
class DeterministicSparsePlanV3:
    planner_version: str
    renderer_version: str
    anchor_selector_version: str
    selection_version: str
    splitter_version: str
    tokenizer_version: str
    topic_id: str
    status: Literal["ok", "fallback"]
    narrative_sha256: str
    token_tape_sha256: str | None
    analyzer_token_sha256: str | None
    analyzer_fingerprint_sha256: str | None
    original_query_text: str
    original_bm25_signature: BM25Signature
    aligned_tokens: tuple[AlignedTokenV3, ...]
    aligned_occurrences: tuple[AlignedOccurrenceV3, ...]
    lexical_units: tuple[LexicalUnitV3, ...]
    conversational_inventory: ConversationalInventoryAuditV3 | None
    recurrence_audit: tuple[RecurrenceTermAuditV3, ...]
    candidate_audit: tuple[AnchorCandidateV3, ...]
    core_audit: tuple[RecurrentCoreAuditV3, ...]
    anchor: SelectedAnchorV3 | None
    raw_criticality: tuple[CriticalityAuditV3, ...]
    final_criticality: tuple[CriticalityAuditV3, ...]
    merge_audit: tuple[ChildMergeAuditV3, ...]
    facets: tuple[DeterministicFacetV3, ...]
    invariant_audit: tuple[FacetInvariantAuditV3, ...]
    failure: PlannerFailureV3 | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def query_variants(self) -> tuple[QueryVariant, ...]:
        original = QueryVariant(
            topic_id=self.topic_id,
            variant_name=f"{PLANNER_VERSION}:original",
            query_text=self.original_query_text,
            source_type=f"{PLANNER_VERSION}_original",
        )
        if self.status == "fallback":
            return (original,)
        return (
            original,
            *(
                QueryVariant(
                    topic_id=self.topic_id,
                    variant_name=facet.variant_name,
                    query_text=facet.query_text,
                    source_type=f"{PLANNER_VERSION}_facet",
                )
                for facet in self.facets
            ),
        )


@dataclass
class _ChildGroup:
    unit_indices: list[int]


class _PlannerInvalid(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _compact_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_compact_json(value)).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_sha256(fingerprint: AnalyzerFingerprint) -> str:
    return _canonical_sha256(fingerprint.to_dict())


def _signature(tokens: Sequence[str]) -> BM25Signature:
    return tuple(sorted(Counter(tokens).items()))


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def normalize_conversational_surface(surface: str) -> str:
    return unicodedata.normalize("NFKC", surface).casefold()


NORMALIZED_CONVERSATIONAL_SURFACES = tuple(
    sorted({normalize_conversational_surface(value) for value in CONVERSATIONAL_SURFACES})
)
if len(NORMALIZED_CONVERSATIONAL_SURFACES) != 114:
    raise AssertionError("v3 normalized conversational inventory must stay unique")
_NORMALIZED_CONVERSATIONAL_SET = frozenset(NORMALIZED_CONVERSATIONAL_SURFACES)


def precision_admitted(recurrent_count: int, unique_count: int) -> bool:
    return recurrent_count >= 2 and 3 * recurrent_count >= 2 * unique_count


def recurrence_mass(
    recurrent_terms: Sequence[str],
    recurrence: Mapping[str, RecurrenceTermAuditV3],
) -> int:
    return sum(
        recurrence[term].child_df * min(recurrence[term].parent_tf, 3)
        for term in recurrent_terms
    )


def _core_score(
    *,
    joint_child_df: int,
    recurrence_mass_value: int,
    recurrent_term_count: int,
    support_union_count: int,
    eligible_nonrecurrent_occurrence_count: int,
    conversational_occurrence_count: int,
    total_analyzed_occurrence_count: int,
    source_token_record_count: int,
) -> CoreScore:
    """Construct the frozen position-free lexicographic score."""

    return (
        joint_child_df,
        recurrence_mass_value,
        recurrent_term_count,
        support_union_count,
        -eligible_nonrecurrent_occurrence_count,
        -conversational_occurrence_count,
        -total_analyzed_occurrence_count,
        -source_token_record_count,
    )


def candidate_tie_key(candidate: AnchorCandidateV3) -> tuple[object, ...]:
    """Ascending key after the maximal lexicographic core score."""

    return (
        -candidate.source_span.start,
        candidate.source_span.end - candidate.source_span.start,
        candidate.exact_text.encode("utf-8"),
    )


def _strict_submultiset(child: Sequence[str], parent: Sequence[str]) -> bool:
    child_counts = Counter(child)
    parent_counts = Counter(parent)
    return child_counts != parent_counts and all(
        count <= parent_counts[term] for term, count in child_counts.items()
    )


def _analyze_checked(
    analyzer: QueryAnalyzer,
    text: str,
    fingerprint: AnalyzerFingerprint,
) -> AnalyzedQuery:
    analyzed = analyzer.analyze(text)
    if analyzed.fingerprint != fingerprint:
        raise _PlannerInvalid(
            "analyzer_fingerprint_changed",
            "query analyzer fingerprint changed during v3 planning",
        )
    return analyzed


def build_conversational_inventory_audit(
    query_analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
    *,
    excluded_occurrence_ids: Sequence[str] = (),
) -> ConversationalInventoryAuditV3:
    projections = tuple(
        ConversationalSurfaceProjectionV3(
            source_surface=surface,
            normalized_surface=normalize_conversational_surface(surface),
            analyzer_tokens=_analyze_checked(query_analyzer, surface, fingerprint).tokens,
        )
        for surface in CONVERSATIONAL_SURFACES
    )
    return ConversationalInventoryAuditV3(
        version=CONVERSATIONAL_SURFACE_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        source_surfaces=CONVERSATIONAL_SURFACES,
        normalized_unique_surfaces=NORMALIZED_CONVERSATIONAL_SURFACES,
        source_surfaces_sha256=_canonical_sha256(list(CONVERSATIONAL_SURFACES)),
        normalized_surfaces_sha256=_canonical_sha256(
            list(NORMALIZED_CONVERSATIONAL_SURFACES)
        ),
        projections=projections,
        projection_sha256=_canonical_sha256([asdict(row) for row in projections]),
        projected_analyzer_terms=tuple(
            sorted({term for row in projections for term in row.analyzer_tokens})
        ),
        analyzer_fingerprint_sha256=_fingerprint_sha256(fingerprint),
        excluded_occurrence_ids=tuple(excluded_occurrence_ids),
    )


def candidate_evidence_identity(
    *,
    narrative_sha256: str,
    start: int,
    end: int,
    exact_text: str,
    token_ids: Sequence[int],
) -> CandidateEvidenceIdentityV3:
    return CandidateEvidenceIdentityV3(
        narrative_sha256=narrative_sha256,
        start=start,
        end=end,
        text_sha256=_text_sha256(exact_text),
        token_ids=tuple(token_ids),
    )


def _align_source(
    narrative: str,
    *,
    token_tape: NarrativeTokenTape,
    query_analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
    base_analysis: AnalyzedQuery,
) -> tuple[
    list[LexicalUnitV3],
    list[AlignedTokenV3],
    list[AlignedOccurrenceV3],
]:
    spans = split_deterministic_units(narrative)
    if len(spans) < 2:
        raise _PlannerInvalid(
            "insufficient_source_units",
            "det_sparse_v3 requires at least two exact source units",
        )
    owners: dict[int, list[int]] = {
        token_id: [] for token_id in range(token_tape.token_count)
    }
    for unit_index, span in enumerate(spans):
        if narrative[span.start : span.end] != span.text:
            raise _PlannerInvalid("source_span_mismatch", "source span is not exact")
        for token_id, token in enumerate(token_tape.tokens):
            if span.start <= token.start_char and token.end_char <= span.end:
                owners[token_id].append(unit_index)
    if any(len(owner) != 1 for owner in owners.values()):
        raise _PlannerInvalid(
            "token_tape_coverage_mismatch",
            "every narrative token record must have exactly one source-unit owner",
        )

    aligned_tokens: list[AlignedTokenV3] = []
    occurrences: list[AlignedOccurrenceV3] = []
    for token_id, token in enumerate(token_tape.tokens):
        unit_index = owners[token_id][0]
        unit_id = f"u{unit_index + 1:03d}"
        surface = narrative[token.start_char : token.end_char]
        if surface != token.text:
            raise _PlannerInvalid(
                "token_surface_mismatch",
                "narrative token record does not resolve to its exact source surface",
            )
        analyzed = _analyze_checked(query_analyzer, surface, fingerprint)
        conversational = (
            normalize_conversational_surface(surface)
            in _NORMALIZED_CONVERSATIONAL_SET
        )
        token_span = SourceSpan(token.start_char, token.end_char, surface)
        aligned_tokens.append(
            AlignedTokenV3(
                token_id=token_id,
                exact_surface=surface,
                source_span=token_span,
                unit_id=unit_id,
                analyzer_tokens=analyzed.tokens,
                conversational_surface=conversational,
            )
        )
        occurrences.extend(
            AlignedOccurrenceV3(
                occurrence_id=f"{unit_id}:t{token_id:04d}:o{position:02d}",
                analyzer_term=term,
                token_id=token_id,
                exact_surface=surface,
                source_span=token_span,
                unit_id=unit_id,
                conversational=conversational,
            )
            for position, term in enumerate(analyzed.tokens)
        )

    units: list[LexicalUnitV3] = []
    for unit_index, span in enumerate(spans):
        unit_id = f"u{unit_index + 1:03d}"
        token_ids = tuple(
            token_id for token_id, owner in owners.items() if owner == [unit_index]
        )
        aligned_unit_tokens = tuple(
            term
            for token_id in token_ids
            for term in aligned_tokens[token_id].analyzer_tokens
        )
        whole = _analyze_checked(query_analyzer, span.text, fingerprint)
        if aligned_unit_tokens != whole.tokens:
            raise _PlannerInvalid(
                "token_analyzer_alignment_mismatch",
                f"per-token analyzer tape does not reconstruct {unit_id}",
            )
        unit_occurrences = [row for row in occurrences if row.unit_id == unit_id]
        units.append(
            LexicalUnitV3(
                unit_id=unit_id,
                source_span=span,
                token_ids=token_ids,
                analyzer_tokens=whole.tokens,
                unique_analyzer_terms=whole.unique_tokens,
                bm25_signature=_signature(whole.tokens),
                aligned_occurrence_ids=tuple(
                    row.occurrence_id for row in unit_occurrences
                ),
                eligible_unique_terms=_stable_unique(
                    row.analyzer_term
                    for row in unit_occurrences
                    if not row.conversational
                ),
            )
        )
    if tuple(term for unit in units for term in unit.analyzer_tokens) != base_analysis.tokens:
        raise _PlannerInvalid(
            "unit_analyzer_alignment_mismatch",
            "exact unit analyzer tapes do not reconstruct the narrative tape",
        )
    return units, aligned_tokens, occurrences


def _build_recurrence(
    units: Sequence[LexicalUnitV3],
    occurrences: Sequence[AlignedOccurrenceV3],
) -> tuple[RecurrenceTermAuditV3, ...]:
    parent_id = units[0].unit_id
    parent_rows = [
        row
        for row in occurrences
        if row.unit_id == parent_id and not row.conversational
    ]
    child_rows = [
        row
        for row in occurrences
        if row.unit_id != parent_id and not row.conversational
    ]
    result: list[RecurrenceTermAuditV3] = []
    for term in _stable_unique(row.analyzer_term for row in parent_rows):
        in_parent = [row for row in parent_rows if row.analyzer_term == term]
        in_children = [row for row in child_rows if row.analyzer_term == term]
        child_units = tuple(sorted({row.unit_id for row in in_children}))
        result.append(
            RecurrenceTermAuditV3(
                analyzer_term=term,
                parent_tf=len(in_parent),
                child_df=len(child_units),
                child_tf=len(in_children),
                parent_occurrence_ids=tuple(row.occurrence_id for row in in_parent),
                child_occurrence_ids=tuple(row.occurrence_id for row in in_children),
                supporting_child_unit_ids=child_units,
            )
        )
    return tuple(result)


def _candidate_hulls(
    candidate: AnchorCandidateV3,
    *,
    narrative: str,
    narrative_sha256: str,
    aligned_tokens: Sequence[AlignedTokenV3],
    recurrence_by_term: Mapping[str, RecurrenceTermAuditV3],
) -> tuple[RecurrentCoreHullV3, ...]:
    token_ids = candidate.token_ids
    matching: list[tuple[int, int, tuple[int, ...], tuple[str, ...]]] = []
    target = set(candidate.recurrent_terms)
    for left in range(len(token_ids)):
        for right in range(left + 1, len(token_ids) + 1):
            window_ids = token_ids[left:right]
            recurrent = {
                term
                for token_id in window_ids
                if not aligned_tokens[token_id].conversational_surface
                for term in aligned_tokens[token_id].analyzer_tokens
                if recurrence_by_term.get(term) is not None
                and recurrence_by_term[term].child_df >= 1
            }
            if recurrent == target:
                matching.append((left, right, window_ids, tuple(sorted(recurrent))))
    minimal = [
        row
        for row in matching
        if not any(
            other[0] >= row[0]
            and other[1] <= row[1]
            and (other[0], other[1]) != (row[0], row[1])
            for other in matching
        )
    ]
    hulls: list[RecurrentCoreHullV3] = []
    for _left, _right, ids, recurrent in minimal:
        start = aligned_tokens[ids[0]].source_span.start
        end = aligned_tokens[ids[-1]].source_span.end
        text = narrative[start:end]
        evidence = candidate_evidence_identity(
            narrative_sha256=narrative_sha256,
            start=start,
            end=end,
            exact_text=text,
            token_ids=ids,
        )
        hulls.append(
            RecurrentCoreHullV3(
                evidence=evidence,
                evidence_sha256=evidence.sha256,
                source_span=SourceSpan(start, end, text),
                exact_text=text,
                token_ids=ids,
                recurrent_terms=recurrent,
            )
        )
    return tuple(hulls)


def _enumerate_candidates(
    narrative: str,
    *,
    narrative_sha256: str,
    parent: LexicalUnitV3,
    child_units: Sequence[LexicalUnitV3],
    aligned_tokens: Sequence[AlignedTokenV3],
    occurrences: Sequence[AlignedOccurrenceV3],
    recurrence_audit: Sequence[RecurrenceTermAuditV3],
    query_analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
) -> tuple[AnchorCandidateV3, ...]:
    recurrence_by_term = {row.analyzer_term: row for row in recurrence_audit}
    child_eligible = {
        unit.unit_id: set(unit.eligible_unique_terms) for unit in child_units
    }
    parent_occurrence_cap = min(6, len(parent.analyzer_tokens) // 2)
    candidates: list[AnchorCandidateV3] = []
    parent_ids = parent.token_ids
    for left in range(len(parent_ids)):
        for right in range(left + 1, min(len(parent_ids), left + 6) + 1):
            token_ids = parent_ids[left:right]
            first = aligned_tokens[token_ids[0]]
            last = aligned_tokens[token_ids[-1]]
            start = first.source_span.start
            end = last.source_span.end
            exact_text = narrative[start:end]
            evidence = candidate_evidence_identity(
                narrative_sha256=narrative_sha256,
                start=start,
                end=end,
                exact_text=exact_text,
                token_ids=token_ids,
            )
            analyzed = _analyze_checked(query_analyzer, exact_text, fingerprint)
            aligned_window = tuple(
                term
                for token_id in token_ids
                for term in aligned_tokens[token_id].analyzer_tokens
            )
            reasons: list[str] = []
            if not first.analyzer_tokens or not last.analyzer_tokens:
                reasons.append("endpoint_has_no_analyzer_occurrence")
            if aligned_window != analyzed.tokens:
                reasons.append("candidate_token_alignment_mismatch")
            unique_terms = analyzed.unique_tokens
            if not 2 <= len(unique_terms) <= 4:
                reasons.append("unique_terms_outside_two_four")
            candidate_occurrences = [
                row for row in occurrences if row.token_id in set(token_ids)
            ]
            eligible_terms = {
                row.analyzer_term
                for row in candidate_occurrences
                if not row.conversational
            }
            recurrent_terms = tuple(
                sorted(
                    term
                    for term in eligible_terms
                    if recurrence_by_term.get(term) is not None
                    and recurrence_by_term[term].child_df >= 1
                )
            )
            if len(recurrent_terms) < 2:
                reasons.append("fewer_than_two_recurrent_terms")
            if not precision_admitted(len(recurrent_terms), len(unique_terms)):
                reasons.append("recurrence_precision_below_two_thirds")
            joint_child_df = sum(
                set(recurrent_terms).issubset(terms)
                for terms in child_eligible.values()
            ) if recurrent_terms else 0
            if joint_child_df < 1:
                reasons.append("no_joint_child_support")
            if len(analyzed.tokens) > parent_occurrence_cap:
                reasons.append("candidate_exceeds_half_parent_occurrence_cap")
            if not _strict_submultiset(analyzed.tokens, parent.analyzer_tokens):
                reasons.append("candidate_not_strict_parent_submultiset")
            mass = recurrence_mass(recurrent_terms, recurrence_by_term)
            eligible_nonrecurrent = sum(
                not row.conversational and row.analyzer_term not in recurrent_terms
                for row in candidate_occurrences
            )
            conversational_count = sum(
                row.conversational for row in candidate_occurrences
            )
            support_union = {
                unit_id
                for term in recurrent_terms
                for unit_id in recurrence_by_term[term].supporting_child_unit_ids
            }
            score = _core_score(
                joint_child_df=joint_child_df,
                recurrence_mass_value=mass,
                recurrent_term_count=len(recurrent_terms),
                support_union_count=len(support_union),
                eligible_nonrecurrent_occurrence_count=eligible_nonrecurrent,
                conversational_occurrence_count=conversational_count,
                total_analyzed_occurrence_count=len(analyzed.tokens),
                source_token_record_count=len(token_ids),
            )
            candidate = AnchorCandidateV3(
                evidence=evidence,
                evidence_sha256=evidence.sha256,
                source_span=SourceSpan(start, end, exact_text),
                exact_text=exact_text,
                token_ids=token_ids,
                analyzed_tokens=analyzed.tokens,
                unique_analyzed_terms=unique_terms,
                bm25_signature=_signature(analyzed.tokens),
                recurrent_terms=recurrent_terms,
                joint_child_df=joint_child_df,
                recurrence_mass=mass,
                eligible_nonrecurrent_occurrence_count=eligible_nonrecurrent,
                conversational_occurrence_count=conversational_count,
                core_score=score,
                admissible=not reasons,
                rejection_reasons=tuple(reasons),
                minimal_hulls=(),
                selected=False,
            )
            if candidate.admissible:
                candidate = replace(
                    candidate,
                    minimal_hulls=_candidate_hulls(
                        candidate,
                        narrative=narrative,
                        narrative_sha256=narrative_sha256,
                        aligned_tokens=aligned_tokens,
                        recurrence_by_term=recurrence_by_term,
                    ),
                )
            candidates.append(candidate)
    return tuple(candidates)


def _core_sha256(terms: Sequence[str]) -> str:
    return _canonical_sha256(list(terms))


def _build_cores(
    candidates: Sequence[AnchorCandidateV3],
) -> tuple[RecurrentCoreAuditV3, ...]:
    by_core: dict[tuple[str, ...], list[AnchorCandidateV3]] = {}
    for candidate in candidates:
        if candidate.admissible:
            by_core.setdefault(candidate.recurrent_terms, []).append(candidate)
    term_sets = {terms: set(terms) for terms in by_core}
    cores: list[RecurrentCoreAuditV3] = []
    for terms in sorted(by_core):
        rows = by_core[terms]
        unique_hulls: dict[CandidateEvidenceIdentityV3, RecurrentCoreHullV3] = {}
        for row in rows:
            for hull in row.minimal_hulls:
                unique_hulls[hull.evidence] = hull
        maximal = not any(
            term_sets[terms] < other_set
            for other_terms, other_set in term_sets.items()
            if other_terms != terms
        )
        cores.append(
            RecurrentCoreAuditV3(
                terms=terms,
                core_sha256=_core_sha256(terms),
                candidate_evidence_sha256s=tuple(
                    sorted(row.evidence_sha256 for row in rows)
                ),
                minimal_hulls=tuple(
                    sorted(
                        unique_hulls.values(),
                        key=lambda row: (
                            row.source_span.start,
                            row.source_span.end,
                            row.evidence_sha256,
                        ),
                    )
                ),
                maximal=maximal,
            )
        )
    return tuple(cores)


def _ambiguous_maximal_cores(cores: Sequence[RecurrentCoreAuditV3]) -> bool:
    maximal = [row for row in cores if row.maximal]
    for index, left in enumerate(maximal):
        for right in maximal[index + 1 :]:
            if set(left.terms).intersection(right.terms):
                continue
            if any(
                left_hull.source_span.end <= right_hull.source_span.start
                or right_hull.source_span.end <= left_hull.source_span.start
                for left_hull in left.minimal_hulls
                for right_hull in right.minimal_hulls
            ):
                return True
    return False


def _select_anchor_candidate(
    candidates: Sequence[AnchorCandidateV3],
    cores: Sequence[RecurrentCoreAuditV3],
) -> tuple[tuple[AnchorCandidateV3, ...], AnchorCandidateV3]:
    maximal_terms = {row.terms for row in cores if row.maximal}
    selectable = [
        row
        for row in candidates
        if row.admissible and row.recurrent_terms in maximal_terms
    ]
    if not selectable:
        raise _PlannerInvalid(
            "recurrent_anchor_unavailable",
            "no coherent recurrent topical anchor is admissible",
        )
    best_score = max(row.core_score for row in selectable)
    best = min(
        (row for row in selectable if row.core_score == best_score),
        key=candidate_tie_key,
    )
    selected_rows = tuple(
        replace(row, selected=row.evidence == best.evidence)
        for row in candidates
    )
    return selected_rows, replace(best, selected=True)


def _exact_group_span(
    narrative: str,
    units: Sequence[LexicalUnitV3],
    indices: Sequence[int],
) -> SourceSpan:
    if not indices or list(indices) != list(range(indices[0], indices[-1] + 1)):
        raise _PlannerInvalid(
            "noncontiguous_child_group",
            "child groups must remain exact and adjacent",
        )
    start = units[indices[0]].source_span.start
    end = units[indices[-1]].source_span.end
    return SourceSpan(start, end, narrative[start:end])


def _merge_children(
    narrative: str,
    *,
    units: Sequence[LexicalUnitV3],
    query_analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
) -> tuple[list[_ChildGroup], tuple[ChildMergeAuditV3, ...]]:
    groups = [_ChildGroup([index]) for index in range(1, len(units))]
    audit: list[ChildMergeAuditV3] = []
    while 1 + len(groups) > MAX_FACETS:
        choices: list[tuple[int, int, SourceSpan, AnalyzedQuery]] = []
        for index in range(len(groups) - 1):
            combined_ids = groups[index].unit_indices + groups[index + 1].unit_indices
            span = _exact_group_span(narrative, units, combined_ids)
            analysis = _analyze_checked(query_analyzer, span.text, fingerprint)
            choices.append((len(analysis.unique_tokens), index, span, analysis))
        _count, index, combined_span, combined_analysis = min(
            choices, key=lambda row: (row[0], row[1])
        )
        left = groups[index]
        right = groups[index + 1]
        merged = _ChildGroup(left.unit_indices + right.unit_indices)
        audit.append(
            ChildMergeAuditV3(
                action="adjacent_child_min_unique_terms",
                left_unit_ids=tuple(units[item].unit_id for item in left.unit_indices),
                right_unit_ids=tuple(units[item].unit_id for item in right.unit_indices),
                result_unit_ids=tuple(units[item].unit_id for item in merged.unit_indices),
                left_source_span=_exact_group_span(narrative, units, left.unit_indices),
                right_source_span=_exact_group_span(narrative, units, right.unit_indices),
                combined_source_span=combined_span,
                combined_analyzed_tokens=combined_analysis.tokens,
                combined_unique_analyzed_terms=combined_analysis.unique_tokens,
            )
        )
        groups[index : index + 2] = [merged]
    return groups, tuple(audit)


def _criticality(
    *,
    owner_id: str,
    coverage_unit_ids: Sequence[str],
    source_span: SourceSpan,
    core_terms: Sequence[str],
    full_analyzed_tokens: Sequence[str],
    eligible_terms: Iterable[str],
) -> CriticalityAuditV3:
    full_unique = _stable_unique(full_analyzed_tokens)
    eligible_unique = _stable_unique(eligible_terms)
    full = set(full_unique)
    eligible = set(eligible_unique)
    full_intersection = tuple(term for term in core_terms if term in full)
    eligible_intersection = tuple(term for term in core_terms if term in eligible)
    missing = tuple(term for term in core_terms if term not in full)
    label: Literal["anchorless", "partial", "complete"]
    if not full_intersection:
        label = "anchorless"
    elif len(full_intersection) == len(core_terms):
        label = "complete"
    else:
        label = "partial"
    return CriticalityAuditV3(
        owner_id=owner_id,
        coverage_unit_ids=tuple(coverage_unit_ids),
        source_span=source_span,
        full_analyzed_tokens=tuple(full_analyzed_tokens),
        full_unique_terms=full_unique,
        eligible_unique_terms=eligible_unique,
        full_core_intersection=full_intersection,
        eligible_core_intersection=eligible_intersection,
        missing_full_core_terms=missing,
        label=label,
    )


def _empty_plan(
    *,
    topic_id: str,
    narrative: str,
    token_tape_sha256: str | None,
    analyzer_token_sha256: str | None,
    analyzer_fingerprint_sha256: str | None,
    original_signature: BM25Signature,
    aligned_tokens: Sequence[AlignedTokenV3],
    occurrences: Sequence[AlignedOccurrenceV3],
    units: Sequence[LexicalUnitV3],
    inventory: ConversationalInventoryAuditV3 | None,
    recurrence_audit: Sequence[RecurrenceTermAuditV3],
    candidate_audit: Sequence[AnchorCandidateV3],
    core_audit: Sequence[RecurrentCoreAuditV3],
    anchor: SelectedAnchorV3 | None,
    raw_criticality: Sequence[CriticalityAuditV3],
    final_criticality: Sequence[CriticalityAuditV3],
    merge_audit: Sequence[ChildMergeAuditV3],
    invariant_audit: Sequence[FacetInvariantAuditV3],
    code: str,
    message: str,
) -> DeterministicSparsePlanV3:
    return DeterministicSparsePlanV3(
        planner_version=PLANNER_VERSION,
        renderer_version=RENDERER_VERSION,
        anchor_selector_version=ANCHOR_SELECTOR_VERSION,
        selection_version=SELECTION_VERSION,
        splitter_version=SPLITTER_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        topic_id=topic_id,
        status="fallback",
        narrative_sha256=_text_sha256(narrative),
        token_tape_sha256=token_tape_sha256,
        analyzer_token_sha256=analyzer_token_sha256,
        analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
        original_query_text=narrative,
        original_bm25_signature=original_signature,
        aligned_tokens=tuple(aligned_tokens),
        aligned_occurrences=tuple(occurrences),
        lexical_units=tuple(units),
        conversational_inventory=inventory,
        recurrence_audit=tuple(recurrence_audit),
        candidate_audit=tuple(candidate_audit),
        core_audit=tuple(core_audit),
        anchor=anchor,
        raw_criticality=tuple(raw_criticality),
        final_criticality=tuple(final_criticality),
        merge_audit=tuple(merge_audit),
        facets=(),
        invariant_audit=tuple(invariant_audit),
        failure=PlannerFailureV3(code, message),
    )


def build_deterministic_sparse_v3_plan(
    topic_id: str,
    narrative: str,
    query_analyzer: QueryAnalyzer,
) -> DeterministicSparsePlanV3:
    """Build a source-exact recurrent-anchor plan or original-only fallback."""

    if not isinstance(topic_id, str) or not topic_id:
        raise ValueError("topic_id must be non-empty text")
    if not isinstance(narrative, str):
        raise TypeError("narrative must be text")
    token_tape = tokenize_narrative(narrative)
    token_hash: str | None = _canonical_sha256(token_tape.to_dict())
    analyzer_hash: str | None = None
    fingerprint_hash: str | None = None
    original_signature: BM25Signature = ()
    aligned_tokens: list[AlignedTokenV3] = []
    occurrences: list[AlignedOccurrenceV3] = []
    units: list[LexicalUnitV3] = []
    inventory: ConversationalInventoryAuditV3 | None = None
    recurrence_audit: tuple[RecurrenceTermAuditV3, ...] = ()
    candidate_audit: tuple[AnchorCandidateV3, ...] = ()
    core_audit: tuple[RecurrentCoreAuditV3, ...] = ()
    anchor: SelectedAnchorV3 | None = None
    raw_criticality: tuple[CriticalityAuditV3, ...] = ()
    final_criticality: tuple[CriticalityAuditV3, ...] = ()
    merge_audit: tuple[ChildMergeAuditV3, ...] = ()
    invariant_audit: list[FacetInvariantAuditV3] = []
    try:
        if not narrative.strip():
            raise _PlannerInvalid("empty_narrative", "narrative is empty")
        base = query_analyzer.analyze(narrative)
        fingerprint = base.fingerprint
        analyzer_hash = _canonical_sha256(list(base.tokens))
        fingerprint_hash = _fingerprint_sha256(fingerprint)
        original_signature = _signature(base.tokens)
        if not base.tokens:
            raise _PlannerInvalid("empty_analyzer_tape", "narrative has no analyzer tokens")
        units, aligned_tokens, occurrences = _align_source(
            narrative,
            token_tape=token_tape,
            query_analyzer=query_analyzer,
            fingerprint=fingerprint,
            base_analysis=base,
        )
        if len(units[0].unique_analyzer_terms) < MIN_PARENT_UNIQUE_TERMS:
            raise _PlannerInvalid(
                "parent_unique_terms_below_four",
                "first source unit must contain at least four unique analyzer terms",
            )
        inventory = build_conversational_inventory_audit(
            query_analyzer,
            fingerprint,
            excluded_occurrence_ids=tuple(
                row.occurrence_id for row in occurrences if row.conversational
            ),
        )
        recurrence_audit = _build_recurrence(units, occurrences)
        candidate_audit = _enumerate_candidates(
            narrative,
            narrative_sha256=_text_sha256(narrative),
            parent=units[0],
            child_units=units[1:],
            aligned_tokens=aligned_tokens,
            occurrences=occurrences,
            recurrence_audit=recurrence_audit,
            query_analyzer=query_analyzer,
            fingerprint=fingerprint,
        )
        core_audit = _build_cores(candidate_audit)
        if not any(row.admissible for row in candidate_audit):
            raise _PlannerInvalid(
                "recurrent_anchor_unavailable",
                "no coherent recurrent topical anchor is admissible",
            )
        if _ambiguous_maximal_cores(core_audit):
            raise _PlannerInvalid(
                "ambiguous_recurrent_anchor",
                "distinct disjoint maximal recurrent cores have nonoverlapping hulls",
            )
        candidate_audit, selected = _select_anchor_candidate(
            candidate_audit, core_audit
        )
        anchor = SelectedAnchorV3(
            candidate_evidence_sha256=selected.evidence_sha256,
            evidence_sha256=selected.evidence_sha256,
            core_sha256=_core_sha256(selected.recurrent_terms),
            source_span=selected.source_span,
            token_ids=selected.token_ids,
            query_text=selected.exact_text,
            analyzed_tokens=selected.analyzed_tokens,
            unique_analyzed_terms=selected.unique_analyzed_terms,
            bm25_signature=selected.bm25_signature,
            core_terms=selected.recurrent_terms,
            core_score=selected.core_score,
        )

        raw_criticality = tuple(
            _criticality(
                owner_id=unit.unit_id,
                coverage_unit_ids=(unit.unit_id,),
                source_span=unit.source_span,
                core_terms=anchor.core_terms,
                full_analyzed_tokens=unit.analyzer_tokens,
                eligible_terms=unit.eligible_unique_terms,
            )
            for unit in units[1:]
        )
        child_groups, merge_audit = _merge_children(
            narrative,
            units=units,
            query_analyzer=query_analyzer,
            fingerprint=fingerprint,
        )
        groups = [[0], *(row.unit_indices for row in child_groups)]
        assigned = [index for group in groups for index in group]
        if assigned != list(range(len(units))) or len(set(assigned)) != len(assigned):
            raise _PlannerInvalid(
                "facet_coverage_partition",
                "every source unit must have exactly one final coverage owner",
            )

        final_rows: list[CriticalityAuditV3] = []
        final_group_analysis: dict[
            tuple[int, ...], tuple[AnalyzedQuery, tuple[str, ...]]
        ] = {}
        for group_index, indices in enumerate(groups[1:], start=2):
            span = _exact_group_span(narrative, units, indices)
            exact_analysis = _analyze_checked(query_analyzer, span.text, fingerprint)
            group_token_ids = tuple(
                token_id for index in indices for token_id in units[index].token_ids
            )
            aligned_group_tape = tuple(
                term
                for token_id in group_token_ids
                for term in aligned_tokens[token_id].analyzer_tokens
            )
            if aligned_group_tape != exact_analysis.tokens:
                raise _PlannerInvalid(
                    "final_group_token_alignment_mismatch",
                    f"exact final group f{group_index:02d} does not replay from aligned token occurrences",
                )
            group_unit_ids = {units[index].unit_id for index in indices}
            eligible_terms = _stable_unique(
                row.analyzer_term
                for row in occurrences
                if row.unit_id in group_unit_ids and not row.conversational
            )
            final_group_analysis[tuple(indices)] = (exact_analysis, eligible_terms)
            final_rows.append(
                _criticality(
                    owner_id=f"f{group_index:02d}",
                    coverage_unit_ids=tuple(units[index].unit_id for index in indices),
                    source_span=span,
                    core_terms=anchor.core_terms,
                    full_analyzed_tokens=exact_analysis.tokens,
                    eligible_terms=eligible_terms,
                )
            )
        final_criticality = tuple(final_rows)

        facets: list[DeterministicFacetV3] = []
        exact_seen: set[str] = set()
        signature_seen: set[BM25Signature] = set()
        anchor_term_set = set(anchor.unique_analyzed_terms)
        for facet_index, indices in enumerate(groups, start=1):
            coverage_span = _exact_group_span(narrative, units, indices)
            if facet_index == 1:
                query_text = units[0].source_span.text
                expected_tape = units[0].analyzer_tokens
                query_spans = (units[0].source_span,)
                payload: tuple[str, ...] = ()
            else:
                query_text = f"{anchor.query_text} {coverage_span.text}"
                child_analysis, eligible_child_terms = final_group_analysis[
                    tuple(indices)
                ]
                child_tape = child_analysis.tokens
                expected_tape = (*anchor.analyzed_tokens, *child_tape)
                query_spans = (anchor.source_span, coverage_span)
                payload = tuple(
                    term for term in eligible_child_terms if term not in anchor_term_set
                )
                if len(payload) < 2:
                    raise _PlannerInvalid(
                        "insufficient_child_payload",
                        f"f{facet_index:02d} has fewer than two eligible child terms outside anchor",
                    )
                if not (
                    anchor.source_span.end <= coverage_span.start
                    or coverage_span.end <= anchor.source_span.start
                ):
                    raise _PlannerInvalid(
                        "anchor_child_span_overlap",
                        "anchor and child source spans must be disjoint",
                    )
            analyzed = _analyze_checked(query_analyzer, query_text, fingerprint)
            reconstruction_exact = analyzed.tokens == tuple(expected_tape)
            signature = _signature(analyzed.tokens)
            facet_id = f"f{facet_index:02d}"
            invariant = FacetInvariantAuditV3(
                facet_id=facet_id,
                exact_distinct_from_original=query_text != narrative,
                signature_distinct_from_original=signature != original_signature,
                occurrence_count=len(analyzed.tokens),
                original_occurrence_count=len(base.tokens),
                strict_original_submultiset=_strict_submultiset(
                    analyzed.tokens, base.tokens
                ),
                reconstruction_exact=reconstruction_exact,
            )
            invariant_audit.append(invariant)
            if not reconstruction_exact:
                raise _PlannerInvalid(
                    "rendered_query_reconstruction_mismatch",
                    f"{facet_id} analyzer tape does not reconstruct from exact source tapes",
                )
            if len(analyzed.unique_tokens) < 3:
                raise _PlannerInvalid(
                    "facet_unique_terms_below_three",
                    f"{facet_id} has fewer than three unique analyzer terms",
                )
            if not invariant.exact_distinct_from_original:
                raise _PlannerInvalid(
                    "facet_exactly_equals_original",
                    f"{facet_id} exactly equals O",
                )
            if not invariant.signature_distinct_from_original:
                raise _PlannerInvalid(
                    "facet_signature_equals_original",
                    f"{facet_id} has O's BM25 signature",
                )
            if (
                invariant.occurrence_count >= invariant.original_occurrence_count
                or not invariant.strict_original_submultiset
            ):
                raise _PlannerInvalid(
                    "facet_not_strict_original_submultiset",
                    f"{facet_id} is not occurrence-strictly narrower than O",
                )
            if query_text in exact_seen:
                raise _PlannerInvalid(
                    "duplicate_exact_facet_query", "facet exact texts collide"
                )
            if signature in signature_seen:
                raise _PlannerInvalid(
                    "duplicate_facet_signature", "facet BM25 signatures collide"
                )
            exact_seen.add(query_text)
            signature_seen.add(signature)
            facets.append(
                DeterministicFacetV3(
                    facet_id=facet_id,
                    variant_name=f"{PLANNER_VERSION}:facet:{facet_id}",
                    coverage_unit_ids=tuple(units[index].unit_id for index in indices),
                    coverage_source_span=coverage_span,
                    query_source_spans=query_spans,
                    query_text=query_text,
                    analyzed_tokens=analyzed.tokens,
                    unique_analyzed_terms=analyzed.unique_tokens,
                    bm25_signature=signature,
                    child_payload_terms=payload,
                )
            )
        if not 2 <= len(facets) <= MAX_FACETS:
            raise _PlannerInvalid(
                "facet_count_outside_two_four",
                "successful v3 plans require two through four facets",
            )
        return DeterministicSparsePlanV3(
            planner_version=PLANNER_VERSION,
            renderer_version=RENDERER_VERSION,
            anchor_selector_version=ANCHOR_SELECTOR_VERSION,
            selection_version=SELECTION_VERSION,
            splitter_version=SPLITTER_VERSION,
            tokenizer_version=TOKENIZER_VERSION,
            topic_id=topic_id,
            status="ok",
            narrative_sha256=_text_sha256(narrative),
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_query_text=narrative,
            original_bm25_signature=original_signature,
            aligned_tokens=tuple(aligned_tokens),
            aligned_occurrences=tuple(occurrences),
            lexical_units=tuple(units),
            conversational_inventory=inventory,
            recurrence_audit=recurrence_audit,
            candidate_audit=candidate_audit,
            core_audit=core_audit,
            anchor=anchor,
            raw_criticality=raw_criticality,
            final_criticality=final_criticality,
            merge_audit=merge_audit,
            facets=tuple(facets),
            invariant_audit=tuple(invariant_audit),
            failure=None,
        )
    except _PlannerInvalid as error:
        return _empty_plan(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_signature=original_signature,
            aligned_tokens=aligned_tokens,
            occurrences=occurrences,
            units=units,
            inventory=inventory,
            recurrence_audit=recurrence_audit,
            candidate_audit=candidate_audit,
            core_audit=core_audit,
            anchor=anchor,
            raw_criticality=raw_criticality,
            final_criticality=final_criticality,
            merge_audit=merge_audit,
            invariant_audit=invariant_audit,
            code=error.code,
            message=str(error),
        )
    except Exception as error:
        return _empty_plan(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_signature=original_signature,
            aligned_tokens=aligned_tokens,
            occurrences=occurrences,
            units=units,
            inventory=inventory,
            recurrence_audit=recurrence_audit,
            candidate_audit=candidate_audit,
            core_audit=core_audit,
            anchor=anchor,
            raw_criticality=raw_criticality,
            final_criticality=final_criticality,
            merge_audit=merge_audit,
            invariant_audit=invariant_audit,
            code="analyzer_error",
            message=f"{type(error).__name__}: {error}",
        )


plan_deterministic_sparse_v3 = build_deterministic_sparse_v3_plan

"""Qrels-blind deterministic sparse decomposition with bounded parent context.

``det_sparse_v2`` is a separate planner arm.  It reuses the frozen v1 exact
splitter and narrative token tape, but renders every child from a strict prefix
of the first unit plus the child's exact contiguous source coverage.  It has no
model, retrieval, reranking, qrels, or network dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

from trec_rag.deterministic_sparse import (
    SPLITTER_VERSION,
    TOKENIZER_VERSION,
    LexicalUnit,
    SourceSpan,
    split_deterministic_units,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, QueryAnalyzer
from trec_rag.query_planner import NarrativeTokenTape, tokenize_narrative


PLANNER_VERSION = "det_sparse_v2"
RENDERER_VERSION = "det_sparse_bounded_parent_renderer_v2"
SELECTION_VERSION = "det_sparse_structural_selection_v2"
MAX_FACETS = 4
MIN_FACET_UNIQUE_TERMS = 3
MIN_PARENT_UNIQUE_TERMS = 4
MIN_CONTEXT_UNIQUE_TERMS = 2
MAX_CONTEXT_UNIQUE_TERMS = 6
CONTEXT_ALLOCATION = "floor_half_u1_unique_terms"


BM25Signature = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PlannerFailureV2:
    code: str
    message: str


@dataclass(frozen=True)
class ContextCandidateAuditV2:
    source_span: SourceSpan
    token_tape_indices: tuple[int, ...]
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_tokens: tuple[str, ...]
    bm25_signature: BM25Signature
    admissible: bool
    rejection_reasons: tuple[str, ...]
    selected: bool


@dataclass(frozen=True)
class BoundedParentContextV2:
    source_span: SourceSpan
    token_tape_indices: tuple[int, ...]
    query_text: str
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_tokens: tuple[str, ...]
    bm25_signature: BM25Signature
    parent_unique_term_count: int
    unique_term_ceiling: int


@dataclass(frozen=True)
class ChildMergeAuditV2:
    action: Literal["adjacent_child_min_unique_terms"]
    left_unit_ids: tuple[str, ...]
    right_unit_ids: tuple[str, ...]
    result_unit_ids: tuple[str, ...]
    left_source_span: SourceSpan
    right_source_span: SourceSpan
    combined_source_span: SourceSpan
    combined_analyzed_tokens: tuple[str, ...]
    combined_unique_analyzed_tokens: tuple[str, ...]


@dataclass(frozen=True)
class FacetInvariantAuditV2:
    facet_id: str
    exact_text_distinct_from_original: bool
    signature_distinct_from_original: bool
    occurrence_count: int
    original_occurrence_count: int
    strict_occurrence_reduction: bool
    strict_original_submultiset: bool
    nonparent_unit_novel_terms: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class DeterministicFacetV2:
    facet_id: str
    variant_name: str
    coverage_unit_ids: tuple[str, ...]
    coverage_source_span: SourceSpan
    query_source_spans: tuple[SourceSpan, ...]
    query_text: str
    analyzed_tokens: tuple[str, ...]
    unique_analyzed_tokens: tuple[str, ...]
    bm25_signature: BM25Signature


@dataclass(frozen=True)
class DeterministicSparsePlanV2:
    planner_version: str
    renderer_version: str
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
    lexical_units: tuple[LexicalUnit, ...]
    facets: tuple[DeterministicFacetV2, ...]
    context: BoundedParentContextV2 | None
    context_selection_audit: tuple[ContextCandidateAuditV2, ...]
    merge_audit: tuple[ChildMergeAuditV2, ...]
    invariant_audit: tuple[FacetInvariantAuditV2, ...]
    failure: PlannerFailureV2 | None

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
    def __init__(
        self,
        code: str,
        message: str,
        *,
        context_selection_audit: Sequence[ContextCandidateAuditV2] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context_selection_audit = tuple(context_selection_audit)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint_sha256(fingerprint: AnalyzerFingerprint) -> str:
    return _canonical_sha256(fingerprint.to_dict())


def _token_tape_sha256(tape: NarrativeTokenTape) -> str:
    return _canonical_sha256(tape.to_dict())


def _analyzer_token_sha256(analysis: AnalyzedQuery) -> str:
    return _canonical_sha256(list(analysis.tokens))


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _signature(tokens: Sequence[str]) -> BM25Signature:
    return tuple(sorted(Counter(tokens).items()))


def _strict_submultiset(child: Sequence[str], parent: Sequence[str]) -> bool:
    child_counts = Counter(child)
    parent_counts = Counter(parent)
    return child_counts != parent_counts and all(
        count <= parent_counts[token] for token, count in child_counts.items()
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
            "query analyzer fingerprint changed during deterministic planning",
        )
    return analyzed


def _exact_group_span(
    narrative: str,
    units: Sequence[LexicalUnit],
    indices: Sequence[int],
) -> SourceSpan:
    if not indices or list(indices) != list(range(indices[0], indices[-1] + 1)):
        raise _PlannerInvalid(
            "noncontiguous_child_group",
            "child merge groups must remain adjacent and contiguous",
        )
    start = units[indices[0]].source_span.start
    end = units[indices[-1]].source_span.end
    return SourceSpan(start=start, end=end, text=narrative[start:end])


def _build_lexical_units(
    narrative: str,
    *,
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
    token_tape: NarrativeTokenTape,
    base_analysis: AnalyzedQuery,
) -> list[LexicalUnit]:
    spans = split_deterministic_units(narrative)
    if len(spans) < 2:
        raise _PlannerInvalid(
            "insufficient_source_units",
            "det_sparse_v2 requires at least two exact source units",
        )
    owners: dict[int, list[int]] = {
        index: [] for index in range(token_tape.token_count)
    }
    for unit_index, span in enumerate(spans):
        if narrative[span.start : span.end] != span.text:
            raise _PlannerInvalid("source_span_mismatch", "source span is not exact")
        for token_index, token in enumerate(token_tape.tokens):
            if span.start <= token.start_char and token.end_char <= span.end:
                owners[token_index].append(unit_index)
    if any(len(indices) != 1 for indices in owners.values()):
        raise _PlannerInvalid(
            "token_tape_coverage_mismatch",
            "every narrative token must belong to exactly one exact source unit",
        )
    units: list[LexicalUnit] = []
    for unit_index, span in enumerate(spans):
        analyzed = _analyze_checked(analyzer, span.text, fingerprint)
        units.append(
            LexicalUnit(
                unit_id=f"u{unit_index + 1:03d}",
                source_span=span,
                token_tape_indices=tuple(
                    index for index, owner in owners.items() if owner == [unit_index]
                ),
                analyzed_tokens=analyzed.tokens,
                unique_analyzed_tokens=analyzed.unique_tokens,
            )
        )
    if tuple(token for unit in units for token in unit.analyzed_tokens) != base_analysis.tokens:
        raise _PlannerInvalid(
            "lexical_coverage_mismatch",
            "source units do not partition the narrative analyzer tape exactly",
        )
    return units


def _select_context(
    narrative: str,
    *,
    parent: LexicalUnit,
    token_tape: NarrativeTokenTape,
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
) -> tuple[BoundedParentContextV2, tuple[ContextCandidateAuditV2, ...]]:
    parent_unique_count = len(parent.unique_analyzed_tokens)
    if parent_unique_count < MIN_PARENT_UNIQUE_TERMS:
        raise _PlannerInvalid(
            "parent_unique_terms_below_four",
            "first source unit must contain at least four unique analyzer terms",
        )
    ceiling = min(MAX_CONTEXT_UNIQUE_TERMS, parent_unique_count // 2)
    parent_indices = parent.token_tape_indices
    if not parent_indices:
        raise _PlannerInvalid("parent_has_no_source_tokens", "first unit has no source tokens")

    provisional: list[dict[str, object]] = []
    for last_index in parent_indices:
        end = token_tape.tokens[last_index].end_char
        span = SourceSpan(
            start=parent.source_span.start,
            end=end,
            text=narrative[parent.source_span.start:end],
        )
        analyzed = _analyze_checked(analyzer, span.text, fingerprint)
        reasons: list[str] = []
        if end >= parent.source_span.end:
            reasons.append("not_strict_source_prefix")
        unique_count = len(analyzed.unique_tokens)
        if not MIN_CONTEXT_UNIQUE_TERMS <= unique_count <= ceiling:
            reasons.append("unique_terms_outside_bounded_range")
        if not _strict_submultiset(analyzed.tokens, parent.analyzed_tokens):
            reasons.append("not_strict_parent_analyzed_submultiset")
        provisional.append(
            {
                "span": span,
                "token_indices": tuple(
                    index
                    for index in parent_indices
                    if token_tape.tokens[index].end_char <= end
                ),
                "analysis": analyzed,
                "reasons": reasons,
            }
        )
    def materialize_audit(
        selected_span: SourceSpan | None,
    ) -> tuple[ContextCandidateAuditV2, ...]:
        audits: list[ContextCandidateAuditV2] = []
        for row in provisional:
            span = row["span"]
            analyzed = row["analysis"]
            token_indices = row["token_indices"]
            reasons = row["reasons"]
            assert isinstance(span, SourceSpan)
            assert isinstance(analyzed, AnalyzedQuery)
            assert isinstance(token_indices, tuple)
            assert isinstance(reasons, list)
            audits.append(
                ContextCandidateAuditV2(
                    source_span=span,
                    token_tape_indices=token_indices,
                    analyzed_tokens=analyzed.tokens,
                    unique_analyzed_tokens=analyzed.unique_tokens,
                    bm25_signature=_signature(analyzed.tokens),
                    admissible=not reasons,
                    rejection_reasons=tuple(reasons),
                    selected=span == selected_span,
                )
            )
        return tuple(audits)

    eligible = [row for row in provisional if not row["reasons"]]
    if not eligible:
        raise _PlannerInvalid(
            "bounded_parent_context_unavailable",
            "no strict token-boundary parent prefix satisfies the bounded context rule",
            context_selection_audit=materialize_audit(None),
        )
    selected = min(
        eligible,
        key=lambda row: (
            -len(row["analysis"].unique_tokens),  # type: ignore[union-attr]
            row["span"].end,  # type: ignore[union-attr]
        ),
    )
    selected_span = selected["span"]
    selected_analysis = selected["analysis"]
    assert isinstance(selected_span, SourceSpan)
    assert isinstance(selected_analysis, AnalyzedQuery)
    audits = materialize_audit(selected_span)
    return (
        BoundedParentContextV2(
            source_span=selected_span,
            token_tape_indices=selected["token_indices"],  # type: ignore[arg-type]
            query_text=selected_span.text,
            analyzed_tokens=selected_analysis.tokens,
            unique_analyzed_tokens=selected_analysis.unique_tokens,
            bm25_signature=_signature(selected_analysis.tokens),
            parent_unique_term_count=parent_unique_count,
            unique_term_ceiling=ceiling,
        ),
        audits,
    )


def _merge_children_to_limit(
    narrative: str,
    *,
    units: Sequence[LexicalUnit],
    analyzer: QueryAnalyzer,
    fingerprint: AnalyzerFingerprint,
) -> tuple[list[_ChildGroup], tuple[ChildMergeAuditV2, ...]]:
    groups = [_ChildGroup([index]) for index in range(1, len(units))]
    audit: list[ChildMergeAuditV2] = []
    # U1 is a protected standalone group, so at most three child groups remain.
    while 1 + len(groups) > MAX_FACETS:
        choices: list[tuple[int, int, SourceSpan, AnalyzedQuery]] = []
        for index in range(len(groups) - 1):
            combined_indices = groups[index].unit_indices + groups[index + 1].unit_indices
            span = _exact_group_span(narrative, units, combined_indices)
            analyzed = _analyze_checked(analyzer, span.text, fingerprint)
            choices.append((len(analyzed.unique_tokens), index, span, analyzed))
        _count, merge_index, span, analyzed = min(
            choices, key=lambda row: (row[0], row[1])
        )
        left = groups[merge_index]
        right = groups[merge_index + 1]
        merged = _ChildGroup(left.unit_indices + right.unit_indices)
        left_span = _exact_group_span(narrative, units, left.unit_indices)
        right_span = _exact_group_span(narrative, units, right.unit_indices)
        audit.append(
            ChildMergeAuditV2(
                action="adjacent_child_min_unique_terms",
                left_unit_ids=tuple(units[index].unit_id for index in left.unit_indices),
                right_unit_ids=tuple(units[index].unit_id for index in right.unit_indices),
                result_unit_ids=tuple(units[index].unit_id for index in merged.unit_indices),
                left_source_span=left_span,
                right_source_span=right_span,
                combined_source_span=span,
                combined_analyzed_tokens=analyzed.tokens,
                combined_unique_analyzed_tokens=analyzed.unique_tokens,
            )
        )
        groups[merge_index : merge_index + 2] = [merged]
    return groups, tuple(audit)


def _fallback(
    *,
    topic_id: str,
    narrative: str,
    token_tape_sha256: str | None,
    analyzer_token_sha256: str | None,
    analyzer_fingerprint_sha256: str | None,
    original_bm25_signature: BM25Signature,
    units: Sequence[LexicalUnit],
    context: BoundedParentContextV2 | None,
    context_audit: Sequence[ContextCandidateAuditV2],
    merge_audit: Sequence[ChildMergeAuditV2],
    invariant_audit: Sequence[FacetInvariantAuditV2],
    code: str,
    message: str,
) -> DeterministicSparsePlanV2:
    return DeterministicSparsePlanV2(
        planner_version=PLANNER_VERSION,
        renderer_version=RENDERER_VERSION,
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
        original_bm25_signature=original_bm25_signature,
        lexical_units=tuple(units),
        facets=(),
        context=context,
        context_selection_audit=tuple(context_audit),
        merge_audit=tuple(merge_audit),
        invariant_audit=tuple(invariant_audit),
        failure=PlannerFailureV2(code=code, message=message),
    )


def build_deterministic_sparse_v2_plan(
    topic_id: str,
    narrative: str,
    query_analyzer: QueryAnalyzer,
) -> DeterministicSparsePlanV2:
    """Build the frozen bounded-parent v2 plan or exact original fallback."""

    if not isinstance(topic_id, str) or not topic_id:
        raise ValueError("topic_id must be non-empty text")
    if not isinstance(narrative, str):
        raise TypeError("narrative must be text")

    token_tape = tokenize_narrative(narrative)
    token_hash: str | None = _token_tape_sha256(token_tape)
    analyzer_hash: str | None = None
    fingerprint_hash: str | None = None
    original_signature: BM25Signature = ()
    units: list[LexicalUnit] = []
    context: BoundedParentContextV2 | None = None
    context_audit: tuple[ContextCandidateAuditV2, ...] = ()
    merge_audit: tuple[ChildMergeAuditV2, ...] = ()
    invariant_audit: list[FacetInvariantAuditV2] = []
    try:
        if not narrative.strip():
            raise _PlannerInvalid("empty_narrative", "narrative is empty")
        base_analysis = query_analyzer.analyze(narrative)
        fingerprint = base_analysis.fingerprint
        analyzer_hash = _analyzer_token_sha256(base_analysis)
        fingerprint_hash = _fingerprint_sha256(fingerprint)
        original_signature = _signature(base_analysis.tokens)
        if not base_analysis.tokens:
            raise _PlannerInvalid("empty_analyzer_tape", "narrative has no analyzer tokens")
        units = _build_lexical_units(
            narrative,
            analyzer=query_analyzer,
            fingerprint=fingerprint,
            token_tape=token_tape,
            base_analysis=base_analysis,
        )
        context, context_audit = _select_context(
            narrative,
            parent=units[0],
            token_tape=token_tape,
            analyzer=query_analyzer,
            fingerprint=fingerprint,
        )
        child_groups, merge_audit = _merge_children_to_limit(
            narrative,
            units=units,
            analyzer=query_analyzer,
            fingerprint=fingerprint,
        )
        coverage_groups = [[0], *(group.unit_indices for group in child_groups)]
        assigned = [index for group in coverage_groups for index in group]
        if assigned != list(range(len(units))) or len(assigned) != len(set(assigned)):
            raise _PlannerInvalid(
                "facet_coverage_partition",
                "each exact source unit must be covered once in source order",
            )

        facets: list[DeterministicFacetV2] = []
        exact_queries: set[str] = set()
        signatures: set[BM25Signature] = set()
        context_terms = set(context.unique_analyzed_tokens)
        for facet_index, group_indices in enumerate(coverage_groups, start=1):
            coverage_span = _exact_group_span(narrative, units, group_indices)
            if facet_index == 1:
                query_text = units[0].source_span.text
                query_spans = (units[0].source_span,)
            else:
                query_text = f"{context.query_text} {coverage_span.text}"
                query_spans = (context.source_span, coverage_span)
            analyzed = _analyze_checked(query_analyzer, query_text, fingerprint)
            signature = _signature(analyzed.tokens)
            facet_id = f"f{facet_index:02d}"
            novel_by_unit: list[tuple[str, tuple[str, ...]]] = []
            for unit_index in group_indices:
                if unit_index == 0:
                    continue
                novel = tuple(
                    token
                    for token in units[unit_index].unique_analyzed_tokens
                    if token not in context_terms
                )
                novel_by_unit.append((units[unit_index].unit_id, novel))
                if not novel:
                    raise _PlannerInvalid(
                        "child_unit_has_no_term_outside_context",
                        f"{units[unit_index].unit_id} contributes no term absent bounded context",
                    )
            audit = FacetInvariantAuditV2(
                facet_id=facet_id,
                exact_text_distinct_from_original=query_text != narrative,
                signature_distinct_from_original=signature != original_signature,
                occurrence_count=len(analyzed.tokens),
                original_occurrence_count=len(base_analysis.tokens),
                strict_occurrence_reduction=len(analyzed.tokens) < len(base_analysis.tokens),
                strict_original_submultiset=_strict_submultiset(
                    analyzed.tokens, base_analysis.tokens
                ),
                nonparent_unit_novel_terms=tuple(novel_by_unit),
            )
            invariant_audit.append(audit)
            if len(analyzed.unique_tokens) < MIN_FACET_UNIQUE_TERMS:
                raise _PlannerInvalid(
                    "facet_unique_terms_below_three",
                    f"{facet_id} has fewer than three unique analyzer terms",
                )
            if not audit.exact_text_distinct_from_original:
                raise _PlannerInvalid(
                    "facet_exactly_equals_original",
                    f"{facet_id} exactly reconstructs the original query",
                )
            if not audit.signature_distinct_from_original:
                raise _PlannerInvalid(
                    "facet_signature_equals_original",
                    f"{facet_id} has the original BM25 term-frequency signature",
                )
            if not audit.strict_occurrence_reduction:
                raise _PlannerInvalid(
                    "facet_occurrences_not_below_original",
                    f"{facet_id} has no strict occurrence-count reduction",
                )
            if not audit.strict_original_submultiset:
                raise _PlannerInvalid(
                    "facet_not_original_submultiset",
                    f"{facet_id} is not a strict analyzer-token submultiset of original",
                )
            if query_text in exact_queries:
                raise _PlannerInvalid(
                    "duplicate_exact_facet_query",
                    "final facets contain an exact-text duplicate",
                )
            if signature in signatures:
                raise _PlannerInvalid(
                    "duplicate_facet_signature",
                    "final facets contain a duplicate BM25 term-frequency signature",
                )
            exact_queries.add(query_text)
            signatures.add(signature)
            facets.append(
                DeterministicFacetV2(
                    facet_id=facet_id,
                    variant_name=f"{PLANNER_VERSION}:facet:{facet_id}",
                    coverage_unit_ids=tuple(units[index].unit_id for index in group_indices),
                    coverage_source_span=coverage_span,
                    query_source_spans=query_spans,
                    query_text=query_text,
                    analyzed_tokens=analyzed.tokens,
                    unique_analyzed_tokens=analyzed.unique_tokens,
                    bm25_signature=signature,
                )
            )
        if not 2 <= len(facets) <= MAX_FACETS:
            raise _PlannerInvalid(
                "facet_count_outside_two_four",
                "successful v2 plans require between two and four final facets",
            )
        return DeterministicSparsePlanV2(
            planner_version=PLANNER_VERSION,
            renderer_version=RENDERER_VERSION,
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
            lexical_units=tuple(units),
            facets=tuple(facets),
            context=context,
            context_selection_audit=context_audit,
            merge_audit=merge_audit,
            invariant_audit=tuple(invariant_audit),
            failure=None,
        )
    except _PlannerInvalid as error:
        if error.context_selection_audit:
            context_audit = error.context_selection_audit
        return _fallback(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_bm25_signature=original_signature,
            units=units,
            context=context,
            context_audit=context_audit,
            merge_audit=merge_audit,
            invariant_audit=invariant_audit,
            code=error.code,
            message=str(error),
        )
    except Exception as error:
        return _fallback(
            topic_id=topic_id,
            narrative=narrative,
            token_tape_sha256=token_hash,
            analyzer_token_sha256=analyzer_hash,
            analyzer_fingerprint_sha256=fingerprint_hash,
            original_bm25_signature=original_signature,
            units=units,
            context=context,
            context_audit=context_audit,
            merge_audit=merge_audit,
            invariant_audit=invariant_audit,
            code="analyzer_error",
            message=f"{type(error).__name__}: {error}",
        )


plan_deterministic_sparse_v2 = build_deterministic_sparse_v2_plan

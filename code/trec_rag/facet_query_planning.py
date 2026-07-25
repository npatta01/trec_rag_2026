"""Typed contracts, validation, and rendering for narrative facet plans.

This module deliberately accepts only typed records.  It performs no plan
generation, parsing, file access, or retrieval integration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import re
import unicodedata

from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


MAX_COVERAGE_ITEMS = 16
MAX_FACETS = 8
MAX_GLOBAL_ANCHORS = 4
MAX_RANGES_PER_COVERAGE_ITEM = 2
MAX_COVERAGE_ITEMS_PER_FACET = 2
MAX_CONTENT_TOKENS_PER_ANCHOR = 8
MAX_EXPANSIONS_PER_FACET = 3
MAX_ANALYZED_WORDS_PER_EXPANSION = 3
MAX_NEW_EXPANSION_TOKENS_PER_FACET = 6
ALLOWED_LEXICAL_RELATIONS = frozenset(
    {
        "alias",
        "acronym",
        "technical_term",
        "common_variant",
        "neutral_search_term",
    }
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_BOOLEAN_OR_QUERY_OPERATOR_PATTERN = re.compile(
    r'\b(?:AND|OR|NOT)\b|&&|\|\||[!(){}\[\]^~*?"\\\\]|(?:^|\s)[+-](?=\S)',
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TokenRange:
    """A half-open interval over the narrative's non-whitespace token tape."""

    start_token: int
    end_token: int


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    token_range: TokenRange
    kind: str
    scope: str
    coverage_refs: tuple[str, ...]


@dataclass(frozen=True)
class Expansion:
    term: str
    relation: str
    anchor_refs: tuple[str, ...]


@dataclass(frozen=True)
class CoverageItem:
    coverage_id: str
    source_ranges: tuple[TokenRange, ...]


@dataclass(frozen=True)
class Facet:
    facet_id: str
    coverage_refs: tuple[str, ...]
    expansions: tuple[Expansion, ...]


@dataclass(frozen=True)
class FacetPlan:
    topic_id: str
    narrative_sha256: str
    anchors: tuple[Anchor, ...]
    coverage_items: tuple[CoverageItem, ...]
    facets: tuple[Facet, ...]


@dataclass(frozen=True)
class FacetPlanningResult:
    queries: tuple[QueryVariant, ...]
    used_fallback: bool
    error: str | None


class FacetPlanValidationError(ValueError):
    """Raised when a supplied facet plan cannot safely be rendered."""


@dataclass(frozen=True)
class NarrativeToken:
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class NarrativeTokenTape:
    """Non-whitespace narrative tokens with Unicode code-point offsets."""

    narrative: str
    tokens: tuple[NarrativeToken, ...]

    def resolve(self, token_range: TokenRange) -> str:
        """Return the exact source text covered by a validated token range."""
        _validate_range_bounds(token_range, len(self.tokens), label="token")
        return self.narrative[
            self.tokens[token_range.start_token].start_offset : self.tokens[
                token_range.end_token - 1
            ].end_offset
        ]


@dataclass(frozen=True)
class _ValidatedFacetPlan:
    plan: FacetPlan
    token_tape: NarrativeTokenTape


def tokenize_narrative(narrative: str) -> NarrativeTokenTape:
    """Tokenize a narrative without normalizing it or losing source offsets."""
    if not isinstance(narrative, str):
        raise TypeError("narrative must be a string")
    return NarrativeTokenTape(
        narrative=narrative,
        tokens=tuple(
            NarrativeToken(
                text=match.group(),
                start_offset=match.start(),
                end_offset=match.end(),
            )
            for match in re.finditer(r"\S+", narrative)
        ),
    )


def validate_facet_plan(topic: Topic, plan: FacetPlan) -> _ValidatedFacetPlan:
    """Validate all mechanical plan invariants and return its tokenized view."""
    if not isinstance(topic, Topic):
        raise FacetPlanValidationError("topic must be a Topic")
    if not isinstance(plan, FacetPlan):
        raise FacetPlanValidationError("plan must be a FacetPlan")
    if plan.topic_id != topic.id:
        raise FacetPlanValidationError("plan topic ID does not match topic ID")
    narrative_sha256 = sha256(topic.narrative.encode("utf-8")).hexdigest()
    if plan.narrative_sha256 != narrative_sha256:
        raise FacetPlanValidationError("plan narrative SHA-256 does not match topic narrative")

    tape = tokenize_narrative(topic.narrative)
    if not tape.tokens:
        raise FacetPlanValidationError("topic narrative must contain non-whitespace text")

    _validate_collection(plan.coverage_items, "coverage items", minimum=1, maximum=MAX_COVERAGE_ITEMS)
    _validate_collection(plan.facets, "facets", minimum=1, maximum=MAX_FACETS)
    _validate_collection(plan.anchors, "anchors", minimum=1)
    _require_records(plan.coverage_items, CoverageItem, "coverage items")
    _require_records(plan.anchors, Anchor, "anchors")
    _require_records(plan.facets, Facet, "facets")

    coverage_ids = _validate_unique_ids(
        (item.coverage_id for item in plan.coverage_items), "coverage"
    )
    anchor_ids = _validate_unique_ids(
        (anchor.anchor_id for anchor in plan.anchors), "anchor"
    )
    _validate_unique_ids((facet.facet_id for facet in plan.facets), "facet")

    _validate_coverage_items(plan.coverage_items, tape)
    _validate_anchors(plan.anchors, coverage_ids, tape)
    _validate_facets(
        plan.facets,
        coverage_ids,
        anchor_ids,
        plan.anchors,
        plan.coverage_items,
        tape,
    )
    _validate_coverage_partition(plan.coverage_items, plan.facets)

    return _ValidatedFacetPlan(plan=plan, token_tape=tape)


def render_facet_queries(topic: Topic, plan: FacetPlan) -> FacetPlanningResult:
    """Render one deterministic structured query for every validated facet."""
    try:
        validated = validate_facet_plan(topic, plan)
    except FacetPlanValidationError as exc:
        return FacetPlanningResult(
            queries=(
                QueryVariant(
                    topic_id=topic.id,
                    variant_name="original",
                    query_text=topic.narrative,
                    source_type="original_topic",
                ),
            ),
            used_fallback=True,
            error=str(exc),
        )
    coverage_by_id = {
        coverage.coverage_id: coverage for coverage in validated.plan.coverage_items
    }
    anchors = validated.plan.anchors

    queries = tuple(
        QueryVariant(
            topic_id=topic.id,
            variant_name=f"facet:{facet.facet_id}",
            query_text=_render_facet_query(
                facet,
                coverage_by_id,
                anchors,
                validated.token_tape,
            ),
            source_type="structured_facet",
        )
        for facet in validated.plan.facets
    )
    return FacetPlanningResult(queries=queries, used_fallback=False, error=None)


def _render_facet_query(
    facet: Facet,
    coverage_by_id: dict[str, CoverageItem],
    anchors: tuple[Anchor, ...],
    tape: NarrativeTokenTape,
) -> str:
    coverage = sorted(
        (coverage_by_id[coverage_id] for coverage_id in facet.coverage_refs),
        key=lambda item: (
            min(source_range.start_token for source_range in item.source_ranges),
            item.coverage_id,
        ),
    )
    components = [
        " ".join(
            tape.resolve(source_range)
            for source_range in _source_ranges_in_narrative_order(item)
        )
        for item in coverage
    ]
    components.extend(
        tape.resolve(anchor.token_range)
        for anchor in sorted(
            _inherited_anchors(facet, anchors),
            key=lambda anchor: (
                anchor.token_range.start_token,
                anchor.token_range.end_token,
                anchor.anchor_id,
            ),
        )
    )
    components.extend(expansion.term for expansion in facet.expansions)

    seen_components: set[str] = set()
    rendered_components: list[str] = []
    for component in components:
        normalized = " ".join(component.split()).casefold()
        if normalized and normalized not in seen_components:
            seen_components.add(normalized)
            rendered_components.append(" ".join(component.split()))
    return " ".join(rendered_components)


def _validate_collection(
    value: object, label: str, *, minimum: int = 0, maximum: int | None = None
) -> None:
    if not isinstance(value, tuple):
        raise FacetPlanValidationError(f"{label} must be tuples")
    if len(value) < minimum:
        if label == "facets":
            raise FacetPlanValidationError("plan must contain at least one facet")
        raise FacetPlanValidationError(f"plan must contain at least {minimum} {label}")
    if maximum is not None and len(value) > maximum:
        raise FacetPlanValidationError(f"plan has too many {label}")


def _require_records(
    records: tuple[object, ...], record_type: type[object], label: str
) -> None:
    if not all(isinstance(record, record_type) for record in records):
        raise FacetPlanValidationError(f"{label} must contain {record_type.__name__} records")


def _validate_unique_ids(values: Iterable[object], label: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
            raise FacetPlanValidationError(f"invalid {label} ID")
        if value in seen:
            raise FacetPlanValidationError(f"duplicate {label} ID: {value}")
        seen.add(value)
    return seen


def _validate_coverage_items(
    coverage_items: tuple[CoverageItem, ...], tape: NarrativeTokenTape
) -> None:
    for item in coverage_items:
        _validate_collection(
            item.source_ranges,
            "source ranges",
            minimum=1,
            maximum=MAX_RANGES_PER_COVERAGE_ITEM,
        )
        seen_ranges: set[TokenRange] = set()
        for token_range in item.source_ranges:
            _validate_range_bounds(token_range, len(tape.tokens), label="coverage")
            if token_range in seen_ranges:
                raise FacetPlanValidationError("duplicate coverage range")
            seen_ranges.add(token_range)
            if not _range_has_content(token_range, tape):
                raise FacetPlanValidationError("coverage range must contain content")


def _validate_anchors(
    anchors: tuple[Anchor, ...], coverage_ids: set[str], tape: NarrativeTokenTape
) -> None:
    global_anchors = 0
    has_subject_anchor = False
    for anchor in anchors:
        _validate_range_bounds(anchor.token_range, len(tape.tokens), label="anchor")
        content_token_count = sum(
            _token_has_content(token.text)
            for token in tape.tokens[
                anchor.token_range.start_token : anchor.token_range.end_token
            ]
        )
        if content_token_count == 0:
            raise FacetPlanValidationError("anchor range must contain content")
        if content_token_count > MAX_CONTENT_TOKENS_PER_ANCHOR:
            raise FacetPlanValidationError("anchor range has too many content tokens")
        if not isinstance(anchor.kind, str) or not _ID_PATTERN.fullmatch(anchor.kind):
            raise FacetPlanValidationError("invalid anchor kind")
        _validate_reference_tuple(anchor.coverage_refs, coverage_ids, "coverage")
        if anchor.scope == "global":
            global_anchors += 1
            if anchor.coverage_refs:
                raise FacetPlanValidationError("global anchor cannot reference coverage")
            has_subject_anchor = has_subject_anchor or anchor.kind in {"topic", "entity"}
        elif anchor.scope == "coverage":
            if not anchor.coverage_refs:
                raise FacetPlanValidationError("coverage anchor must reference coverage")
        else:
            raise FacetPlanValidationError("invalid anchor scope")
    if global_anchors > MAX_GLOBAL_ANCHORS:
        raise FacetPlanValidationError("plan has too many global anchors")
    if global_anchors < 1:
        raise FacetPlanValidationError("plan requires at least one global anchor")
    if not has_subject_anchor:
        raise FacetPlanValidationError("plan requires a topic or entity global anchor")


def _validate_facets(
    facets: tuple[Facet, ...],
    coverage_ids: set[str],
    anchor_ids: set[str],
    anchors: tuple[Anchor, ...],
    coverage_items: tuple[CoverageItem, ...],
    tape: NarrativeTokenTape,
) -> None:
    coverage_by_id = {
        coverage_item.coverage_id: coverage_item for coverage_item in coverage_items
    }
    for facet in facets:
        _validate_reference_tuple(facet.coverage_refs, coverage_ids, "coverage")
        if not facet.coverage_refs:
            raise FacetPlanValidationError("facet must reference coverage")
        if len(facet.coverage_refs) > MAX_COVERAGE_ITEMS_PER_FACET:
            raise FacetPlanValidationError("facet references too many coverage items")
        _validate_collection(
            facet.expansions,
            "expansions",
            maximum=MAX_EXPANSIONS_PER_FACET,
        )
        for expansion in facet.expansions:
            if not isinstance(expansion, Expansion):
                raise FacetPlanValidationError("expansions must be Expansion records")
            _validate_reference_tuple(expansion.anchor_refs, anchor_ids, "anchor")
        _validate_facet_expansions(facet, anchors, coverage_by_id, tape)


def _validate_facet_expansions(
    facet: Facet,
    anchors: tuple[Anchor, ...],
    coverage_by_id: dict[str, CoverageItem],
    tape: NarrativeTokenTape,
) -> None:
    inherited_anchors = _inherited_anchors(facet, anchors)
    inherited_anchor_ids = {anchor.anchor_id for anchor in inherited_anchors}
    inherited_content_tokens = {
        token.casefold()
        for coverage_id in facet.coverage_refs
        for source_range in _source_ranges_in_narrative_order(coverage_by_id[coverage_id])
        for token in _analyzed_words(tape.resolve(source_range))
    }
    inherited_content_tokens.update(
        token.casefold()
        for anchor in inherited_anchors
        for token in _analyzed_words(tape.resolve(anchor.token_range))
    )
    narrative_numeric_runs = set(_numeric_runs(tape.narrative))
    expansion_content_tokens: set[str] = set()

    for expansion in facet.expansions:
        if not expansion.anchor_refs:
            raise FacetPlanValidationError("expansion must reference at least one anchor")
        if not set(expansion.anchor_refs).issubset(inherited_anchor_ids):
            raise FacetPlanValidationError("expansion references an anchor not inherited by its facet")
        if (
            not isinstance(expansion.relation, str)
            or expansion.relation not in ALLOWED_LEXICAL_RELATIONS
        ):
            raise FacetPlanValidationError("unsupported expansion relation")
        if not isinstance(expansion.term, str):
            raise FacetPlanValidationError("expansion term must be a string")
        if any(unicodedata.category(character) == "Cc" for character in expansion.term):
            raise FacetPlanValidationError("expansion term contains a control character")
        if ":" in expansion.term:
            raise FacetPlanValidationError("expansion term contains field syntax")
        if _BOOLEAN_OR_QUERY_OPERATOR_PATTERN.search(expansion.term):
            raise FacetPlanValidationError("expansion term contains a query operator")
        if not set(_numeric_runs(expansion.term)).issubset(narrative_numeric_runs):
            raise FacetPlanValidationError("expansion term contains an absent numeric run")

        analyzed_words = _analyzed_words(expansion.term)
        if not 1 <= len(analyzed_words) <= MAX_ANALYZED_WORDS_PER_EXPANSION:
            raise FacetPlanValidationError("expansion term has too many analyzed words")
        expansion_content_tokens.update(word.casefold() for word in analyzed_words)

    new_content_tokens = expansion_content_tokens - inherited_content_tokens
    if len(new_content_tokens) > MAX_NEW_EXPANSION_TOKENS_PER_FACET:
        raise FacetPlanValidationError("facet has too many new unique content tokens")


def _inherited_anchors(facet: Facet, anchors: tuple[Anchor, ...]) -> tuple[Anchor, ...]:
    facet_coverage = set(facet.coverage_refs)
    return tuple(
        anchor
        for anchor in anchors
        if anchor.scope == "global"
        or set(anchor.coverage_refs).issubset(facet_coverage)
    )


def _source_ranges_in_narrative_order(coverage: CoverageItem) -> tuple[TokenRange, ...]:
    return tuple(
        sorted(
            coverage.source_ranges,
            key=lambda source_range: (
                source_range.start_token,
                source_range.end_token,
            ),
        )
    )


def _analyzed_words(text: str) -> tuple[str, ...]:
    words: list[str] = []
    current_word: list[str] = []
    for character in text:
        if unicodedata.category(character)[0] in {"L", "N"}:
            current_word.append(character)
        elif current_word:
            words.append("".join(current_word))
            current_word = []
    if current_word:
        words.append("".join(current_word))
    return tuple(words)


def _numeric_runs(text: str) -> tuple[str, ...]:
    runs: list[str] = []
    current_run: list[str] = []
    for character in text:
        if unicodedata.category(character)[0] == "N":
            current_run.append(character)
        elif current_run:
            runs.append("".join(current_run))
            current_run = []
    if current_run:
        runs.append("".join(current_run))
    return tuple(runs)


def _validate_coverage_partition(
    coverage_items: tuple[CoverageItem, ...], facets: tuple[Facet, ...]
) -> None:
    assignment_count = {item.coverage_id: 0 for item in coverage_items}
    for facet in facets:
        for coverage_id in facet.coverage_refs:
            assignment_count[coverage_id] += 1
    for coverage_id, count in assignment_count.items():
        if count != 1:
            raise FacetPlanValidationError(
                f"coverage {coverage_id} must appear in exactly one facet"
            )


def _validate_reference_tuple(
    references: object, valid_ids: set[str], target: str
) -> None:
    if not isinstance(references, tuple):
        raise FacetPlanValidationError(f"{target} references must be tuples")
    seen: set[str] = set()
    for reference in references:
        if not isinstance(reference, str) or reference not in valid_ids:
            raise FacetPlanValidationError(f"unknown {target} reference")
        if reference in seen:
            raise FacetPlanValidationError(f"duplicate {target} reference")
        seen.add(reference)


def _validate_range_bounds(token_range: object, token_count: int, *, label: str) -> None:
    if not isinstance(token_range, TokenRange):
        raise FacetPlanValidationError(f"{label} range must be a TokenRange")
    if type(token_range.start_token) is not int or type(token_range.end_token) is not int:
        raise FacetPlanValidationError(
            f"{label} range must use integer token indices"
        )
    if token_range.start_token < 0:
        raise FacetPlanValidationError(f"{label} range has a negative start")
    if token_range.end_token <= token_range.start_token:
        if token_range.end_token == token_range.start_token:
            raise FacetPlanValidationError(f"{label} range is empty")
        raise FacetPlanValidationError(f"{label} range is reversed")
    if token_range.end_token > token_count:
        raise FacetPlanValidationError(f"{label} range is out-of-bounds")


def _range_has_content(token_range: TokenRange, tape: NarrativeTokenTape) -> bool:
    return any(
        _token_has_content(token.text)
        for token in tape.tokens[token_range.start_token : token_range.end_token]
    )


def _token_has_content(token: str) -> bool:
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in token)

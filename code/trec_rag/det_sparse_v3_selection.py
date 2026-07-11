"""Qrels-blind critical-first structural topic selection for v3.

The selector consumes only the frozen candidate narratives, their deterministic
plans, and analyzer-derived shape fields.  It has no retrieval, qrels, model,
reranker, metric, or replacement path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from trec_rag.det_sparse_v3_config import (
    CANDIDATE_TOPIC_IDS,
    PILOT_TOPIC_COUNT,
    QUANTILE_BIN_COUNT,
    SELECTION_SEED,
    SELECTION_VERSION,
)
from trec_rag.deterministic_sparse_v3 import (
    DeterministicSparsePlanV3,
    build_deterministic_sparse_v3_plan,
)
from trec_rag.query_analyzer import QueryAnalyzer
from trec_rag.topics import Topic


SELECTION_SCHEMA_VERSION = "det_sparse_anchor_critical_quantile_selection_v3"
CRITICAL_LABEL = "anchorless"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selection_digest(*, topic_id: str, narrative_sha256: str, seed: str) -> str:
    """Apply the frozen seed-NUL-topic-NUL-narrative digest encoding."""

    if not isinstance(topic_id, str) or not topic_id or not topic_id.isascii():
        raise ValueError("topic_id must be non-empty ASCII text")
    if (
        not isinstance(narrative_sha256, str)
        or len(narrative_sha256) != 64
        or any(character not in "0123456789abcdef" for character in narrative_sha256)
    ):
        raise ValueError("narrative_sha256 must be 64 lowercase hexadecimal bytes")
    if not isinstance(seed, str) or not seed:
        raise ValueError("selection seed must be non-empty text")
    payload = (
        seed.encode("utf-8")
        + b"\0"
        + topic_id.encode("utf-8")
        + b"\0"
        + narrative_sha256.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def semantic_plan_sha256(plan: object) -> str:
    """Hash the canonical meaning returned by the v3 plan."""

    to_dict = getattr(plan, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("v3 plan must expose to_dict()")
    return _canonical_sha256(to_dict())


@dataclass(frozen=True)
class CriticalityCounts:
    anchorless: int
    partial: int
    complete: int

    @property
    def total(self) -> int:
        return self.anchorless + self.partial + self.complete


@dataclass(frozen=True)
class CandidateScreen:
    topic_id: str
    narrative_sha256: str
    token_tape_sha256: str | None
    analyzer_token_sha256: str | None
    analyzer_fingerprint_sha256: str | None
    unit_count: int
    original_unique_term_count: int
    facet_count: int
    merge_count: int
    anchor_core_term_count: int | None
    anchor_text_sha256: str | None
    anchor_evidence_sha256: str | None
    anchor_core_sha256: str | None
    raw_criticality: CriticalityCounts
    final_criticality: CriticalityCounts
    plan_status: str
    plan_semantic_sha256: str
    eligible: bool
    failure_code: str | None
    failure_message: str | None
    selection_digest: str
    remaining_sort_key: tuple[int, int, int, int, int] | None
    quantile_bin: int | None
    selection_role: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuantileBinAudit:
    bin_index: int
    start: int
    end: int
    topic_ids: tuple[str, ...]
    winner_topic_id: str


@dataclass(frozen=True)
class SelectionFailure:
    code: str
    message: str


@dataclass(frozen=True)
class StructuralSelectionV3:
    schema_version: str
    selection_version: str
    seed: str
    status: str
    candidate_topic_ids: tuple[str, ...]
    screens: tuple[CandidateScreen, ...]
    critical_pool_topic_ids: tuple[str, ...]
    critical_topic_id: str | None
    remaining_ordered_topic_ids: tuple[str, ...]
    quantile_bins: tuple[QuantileBinAudit, ...]
    provisional_selected_topic_ids: tuple[str, ...]
    selected_topic_ids: tuple[str, ...]
    failure: SelectionFailure | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionOutcomeV3:
    selection: StructuralSelectionV3
    plans_by_topic: Mapping[str, DeterministicSparsePlanV3]


def _build_plan(
    topic_id: str,
    narrative: str,
    query_analyzer: QueryAnalyzer,
) -> DeterministicSparsePlanV3:
    """Call the sole v3 production planner (a seam only for synthetic tests)."""

    return build_deterministic_sparse_v3_plan(topic_id, narrative, query_analyzer)


def _criticality_counts(rows: object, owner: str) -> CriticalityCounts:
    if not isinstance(rows, (tuple, list)):
        raise ValueError(f"{owner} must be an ordered criticality sequence")
    counts = {"anchorless": 0, "partial": 0, "complete": 0}
    for row in rows:
        label = getattr(row, "label", None)
        if label not in counts:
            raise ValueError(f"{owner} has an unknown criticality label {label!r}")
        counts[label] += 1
    return CriticalityCounts(**counts)


def _optional_hash(plan: object, name: str) -> str | None:
    value = getattr(plan, name, None)
    if value is not None and (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"plan {name} must be lowercase SHA-256 or null")
    return value


def _screen_topic(
    topic: Topic,
    *,
    query_analyzer: QueryAnalyzer,
    fingerprint: object,
    seed: str,
) -> tuple[CandidateScreen, DeterministicSparsePlanV3]:
    plan = _build_plan(topic.id, topic.narrative, query_analyzer)
    if getattr(plan, "topic_id", None) != topic.id:
        raise ValueError("v3 plan topic ID differs from its source topic")
    if getattr(plan, "original_query_text", None) != topic.narrative:
        raise ValueError("v3 plan original query differs from the exact narrative")
    narrative_sha256 = _text_sha256(topic.narrative)
    if getattr(plan, "narrative_sha256", None) != narrative_sha256:
        raise ValueError("v3 plan narrative hash differs from its source narrative")

    analyzed = query_analyzer.analyze(topic.narrative)
    if analyzed.fingerprint != fingerprint or query_analyzer.fingerprint != fingerprint:
        raise ValueError("selection analyzer fingerprint changed")
    plan_status = getattr(plan, "status", None)
    if plan_status not in {"ok", "fallback"}:
        raise ValueError(f"v3 plan has unknown status {plan_status!r}")
    eligible = plan_status == "ok"
    token_tape_sha256 = _optional_hash(plan, "token_tape_sha256")
    analyzer_token_sha256 = _optional_hash(plan, "analyzer_token_sha256")
    analyzer_fingerprint_sha256 = _optional_hash(
        plan, "analyzer_fingerprint_sha256"
    )
    if eligible:
        if token_tape_sha256 is None:
            raise ValueError("eligible v3 plan lacks its token-tape hash")
        expected_analyzer_token_sha256 = _canonical_sha256(list(analyzed.tokens))
        if analyzer_token_sha256 != expected_analyzer_token_sha256:
            raise ValueError("eligible v3 plan analyzer-token hash drifted")
        fingerprint_to_dict = getattr(fingerprint, "to_dict", None)
        if not callable(fingerprint_to_dict):
            raise TypeError("selection analyzer fingerprint must expose to_dict()")
        expected_fingerprint_sha256 = _canonical_sha256(fingerprint_to_dict())
        if analyzer_fingerprint_sha256 != expected_fingerprint_sha256:
            raise ValueError("eligible v3 plan analyzer-fingerprint hash drifted")

    units = getattr(plan, "lexical_units", None)
    facets = getattr(plan, "facets", None)
    merge_audit = getattr(plan, "merge_audit", None)
    raw_rows = getattr(plan, "raw_criticality", None)
    final_rows = getattr(plan, "final_criticality", None)
    if not isinstance(units, (tuple, list)):
        raise ValueError("v3 plan lexical_units must be an ordered sequence")
    if not isinstance(facets, (tuple, list)):
        raise ValueError("v3 plan facets must be an ordered sequence")
    if not isinstance(merge_audit, (tuple, list)):
        raise ValueError("v3 plan merge_audit must be an ordered sequence")
    raw_criticality = _criticality_counts(raw_rows, "raw_criticality")
    final_criticality = _criticality_counts(final_rows, "final_criticality")

    anchor = getattr(plan, "anchor", None)
    if eligible and anchor is None:
        raise ValueError("eligible v3 plan lacks a selected anchor")
    if anchor is None:
        anchor_core_term_count = None
        anchor_text_sha256 = None
        anchor_evidence_sha256 = None
        anchor_core_sha256 = None
    else:
        core_terms = getattr(anchor, "core_terms", None)
        query_text = getattr(anchor, "query_text", None)
        if not isinstance(core_terms, (tuple, list)) or not all(
            isinstance(term, str) and term for term in core_terms
        ):
            raise ValueError("selected anchor core_terms must be a non-empty sequence")
        if not isinstance(query_text, str) or not query_text:
            raise ValueError("selected anchor query_text must be non-empty text")
        anchor_core_term_count = len(core_terms)
        anchor_text_sha256 = _text_sha256(query_text)
        anchor_evidence_sha256 = getattr(anchor, "evidence_sha256", None)
        anchor_core_sha256 = getattr(anchor, "core_sha256", None)
        for name, value in (
            ("anchor.evidence_sha256", anchor_evidence_sha256),
            ("anchor.core_sha256", anchor_core_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")

    if eligible:
        if not 2 <= len(facets) <= 4:
            raise ValueError("eligible v3 plan must contain two through four facets")
        if anchor_core_term_count is None or anchor_core_term_count < 2:
            raise ValueError("eligible v3 plan must retain at least two core terms")
        if raw_criticality.total != len(units) - 1:
            raise ValueError("raw criticality must contain one row per child source unit")
        if final_criticality.total != len(facets) - 1:
            raise ValueError("final criticality must contain one row per child facet")

    failure = getattr(plan, "failure", None)
    if eligible and failure is not None:
        raise ValueError("eligible v3 plan unexpectedly carries a failure")
    if not eligible and failure is None:
        raise ValueError("fallback v3 plan must carry an exact failure")
    failure_code = getattr(failure, "code", None) if failure is not None else None
    failure_message = getattr(failure, "message", None) if failure is not None else None
    if failure is not None and (
        not isinstance(failure_code, str)
        or not failure_code
        or not isinstance(failure_message, str)
        or not failure_message
    ):
        raise ValueError("v3 plan failure must have non-empty code and message")

    digest = selection_digest(
        topic_id=topic.id,
        narrative_sha256=narrative_sha256,
        seed=seed,
    )
    sort_key = (
        min(len(units), 4),
        len(analyzed.unique_tokens),
        len(facets),
        int(anchor_core_term_count or 0),
        int(topic.id),
    ) if eligible else None
    return (
        CandidateScreen(
            topic_id=topic.id,
            narrative_sha256=narrative_sha256,
            token_tape_sha256=token_tape_sha256,
            analyzer_token_sha256=analyzer_token_sha256,
            analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
            unit_count=len(units),
            original_unique_term_count=len(analyzed.unique_tokens),
            facet_count=len(facets),
            merge_count=len(merge_audit),
            anchor_core_term_count=anchor_core_term_count,
            anchor_text_sha256=anchor_text_sha256,
            anchor_evidence_sha256=anchor_evidence_sha256,
            anchor_core_sha256=anchor_core_sha256,
            raw_criticality=raw_criticality,
            final_criticality=final_criticality,
            plan_status=plan_status,
            plan_semantic_sha256=semantic_plan_sha256(plan),
            eligible=eligible,
            failure_code=failure_code,
            failure_message=failure_message,
            selection_digest=digest,
            remaining_sort_key=sort_key,
            quantile_bin=None,
            selection_role=None,
        ),
        plan,
    )


def _selection_result(
    *,
    seed: str,
    status: str,
    screens: Sequence[CandidateScreen],
    plans: Mapping[str, DeterministicSparsePlanV3],
    critical_pool_topic_ids: Sequence[str] = (),
    critical_topic_id: str | None = None,
    remaining_ordered_topic_ids: Sequence[str] = (),
    quantile_bins: Sequence[QuantileBinAudit] = (),
    provisional_selected_topic_ids: Sequence[str] = (),
    selected_topic_ids: Sequence[str] = (),
    failure: SelectionFailure | None = None,
) -> SelectionOutcomeV3:
    return SelectionOutcomeV3(
        selection=StructuralSelectionV3(
            schema_version=SELECTION_SCHEMA_VERSION,
            selection_version=SELECTION_VERSION,
            seed=seed,
            status=status,
            candidate_topic_ids=CANDIDATE_TOPIC_IDS,
            screens=tuple(screens),
            critical_pool_topic_ids=tuple(critical_pool_topic_ids),
            critical_topic_id=critical_topic_id,
            remaining_ordered_topic_ids=tuple(remaining_ordered_topic_ids),
            quantile_bins=tuple(quantile_bins),
            provisional_selected_topic_ids=tuple(provisional_selected_topic_ids),
            selected_topic_ids=tuple(selected_topic_ids),
            failure=failure,
        ),
        plans_by_topic=dict(plans),
    )


def _eligible_remaining_sort_key(
    row: CandidateScreen,
) -> tuple[int, int, int, int, int]:
    if not row.eligible or row.remaining_sort_key is None:
        raise AssertionError("only eligible audited rows may enter v3 quantile sorting")
    return row.remaining_sort_key


def screen_and_select_structural_topics_v3(
    topics: Sequence[Topic],
    *,
    query_analyzer: QueryAnalyzer,
    candidate_topic_ids: Sequence[str] = CANDIDATE_TOPIC_IDS,
    seed: str = SELECTION_SEED,
) -> SelectionOutcomeV3:
    """Screen exactly nine candidates and apply critical-first three-bin selection."""

    if tuple(candidate_topic_ids) != CANDIDATE_TOPIC_IDS:
        raise ValueError("v3 candidate universe/order differs from the frozen set")
    if seed != SELECTION_SEED:
        raise ValueError("v3 structural selection seed differs from the frozen seed")
    if tuple(topic.id for topic in topics) != CANDIDATE_TOPIC_IDS:
        raise ValueError("candidate topic records differ from the frozen ID order")

    fingerprint = query_analyzer.fingerprint
    plans: dict[str, DeterministicSparsePlanV3] = {}
    screens: list[CandidateScreen] = []
    for topic in topics:
        screen, plan = _screen_topic(
            topic,
            query_analyzer=query_analyzer,
            fingerprint=fingerprint,
            seed=seed,
        )
        screens.append(screen)
        plans[topic.id] = plan

    eligible = [row for row in screens if row.eligible]
    if len(eligible) < PILOT_TOPIC_COUNT:
        return _selection_result(
            seed=seed,
            status="failure",
            screens=screens,
            plans=plans,
            failure=SelectionFailure(
                "fewer_than_four_eligible_topics",
                "fewer than four candidates satisfy frozen v3 admission",
            ),
        )

    critical_pool = [
        row for row in eligible if row.raw_criticality.anchorless >= 1
    ]
    critical_pool_ids = tuple(row.topic_id for row in critical_pool)
    if not critical_pool:
        return _selection_result(
            seed=seed,
            status="failure",
            screens=screens,
            plans=plans,
            critical_pool_topic_ids=critical_pool_ids,
            failure=SelectionFailure(
                "no_anchor_critical_topic",
                "no eligible topic has a raw full-set anchorless child",
            ),
        )
    critical = min(
        critical_pool,
        key=lambda row: (row.selection_digest, int(row.topic_id)),
    )

    remaining = sorted(
        (row for row in eligible if row.topic_id != critical.topic_id),
        key=_eligible_remaining_sort_key,
    )
    remaining_ids = tuple(row.topic_id for row in remaining)
    r = len(remaining)
    if r < QUANTILE_BIN_COUNT:
        return _selection_result(
            seed=seed,
            status="failure",
            screens=screens,
            plans=plans,
            critical_pool_topic_ids=critical_pool_ids,
            critical_topic_id=critical.topic_id,
            remaining_ordered_topic_ids=remaining_ids,
            failure=SelectionFailure(
                "fewer_than_three_remaining_topics",
                "critical removal left fewer than three eligible topics",
            ),
        )

    bins: list[QuantileBinAudit] = []
    bin_by_topic: dict[str, int] = {}
    winners: list[str] = []
    for bin_index in range(QUANTILE_BIN_COUNT):
        start = (bin_index * r) // QUANTILE_BIN_COUNT
        end = ((bin_index + 1) * r) // QUANTILE_BIN_COUNT
        members = remaining[start:end]
        if not members:
            raise AssertionError("approved v3 half-open quantile bin is empty")
        winner = min(
            members,
            key=lambda row: (row.selection_digest, int(row.topic_id)),
        )
        member_ids = tuple(row.topic_id for row in members)
        bins.append(
            QuantileBinAudit(
                bin_index=bin_index,
                start=start,
                end=end,
                topic_ids=member_ids,
                winner_topic_id=winner.topic_id,
            )
        )
        winners.append(winner.topic_id)
        for member in members:
            bin_by_topic[member.topic_id] = bin_index

    provisional = (critical.topic_id, *winners)
    if len(provisional) != PILOT_TOPIC_COUNT or len(set(provisional)) != PILOT_TOPIC_COUNT:
        raise AssertionError("v3 selection did not produce four distinct topics")
    winner_roles = {
        critical.topic_id: "critical",
        **{topic_id: f"bin{index}" for index, topic_id in enumerate(winners)},
    }
    annotated_screens = tuple(
        replace(
            row,
            quantile_bin=bin_by_topic.get(row.topic_id),
            selection_role=winner_roles.get(row.topic_id),
        )
        for row in screens
    )

    if critical.final_criticality.anchorless < 1:
        return _selection_result(
            seed=seed,
            status="failure",
            screens=annotated_screens,
            plans=plans,
            critical_pool_topic_ids=critical_pool_ids,
            critical_topic_id=critical.topic_id,
            remaining_ordered_topic_ids=remaining_ids,
            quantile_bins=bins,
            provisional_selected_topic_ids=provisional,
            failure=SelectionFailure(
                "selected_critical_merge_masked",
                "selected critical topic lacks a final full-set anchorless witness",
            ),
        )

    return _selection_result(
        seed=seed,
        status="ok",
        screens=annotated_screens,
        plans=plans,
        critical_pool_topic_ids=critical_pool_ids,
        critical_topic_id=critical.topic_id,
        remaining_ordered_topic_ids=remaining_ids,
        quantile_bins=bins,
        provisional_selected_topic_ids=provisional,
        selected_topic_ids=provisional,
        failure=None,
    )

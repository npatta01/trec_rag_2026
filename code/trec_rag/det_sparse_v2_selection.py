"""Qrels-blind structural screening and deterministic topic selection for v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from trec_rag.det_sparse_v2_config import (
    CANDIDATE_TOPIC_IDS,
    PILOT_TOPIC_COUNT,
    SELECTION_SEED,
)
from trec_rag.deterministic_sparse_v2 import (
    SELECTION_VERSION,
    DeterministicSparsePlanV2,
    build_deterministic_sparse_v2_plan,
)
from trec_rag.query_analyzer import QueryAnalyzer
from trec_rag.topics import Topic


SELECTION_SCHEMA_VERSION = "det_sparse_structural_selection_v2"
STRATUM_ORDER = ("A", "B", "C", "D")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_plan_sha256(plan: DeterministicSparsePlanV2) -> str:
    """Hash canonical plan meaning independently of JSON artifact formatting."""

    return _canonical_sha256(plan.to_dict())


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selection_digest(
    *,
    topic_id: str,
    narrative_sha256: str,
    seed: str,
) -> str:
    payload = (
        seed.encode("utf-8")
        + b"\0"
        + topic_id.encode("utf-8")
        + b"\0"
        + narrative_sha256.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CandidateScreen:
    topic_id: str
    narrative_sha256: str
    unit_count: int
    original_unique_term_count: int
    plan_status: str
    plan_semantic_sha256: str
    eligible: bool
    failure_code: str | None
    failure_message: str | None
    stratum: str | None
    selection_digest: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionFailure:
    code: str
    message: str


@dataclass(frozen=True)
class StructuralSelection:
    schema_version: str
    selection_version: str
    seed: str
    status: str
    candidate_topic_ids: tuple[str, ...]
    screens: tuple[CandidateScreen, ...]
    selected_topic_ids: tuple[str, ...]
    failure: SelectionFailure | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionOutcome:
    selection: StructuralSelection
    plans_by_topic: Mapping[str, DeterministicSparsePlanV2]


def _failure(
    *,
    seed: str,
    screens: Sequence[CandidateScreen],
    plans: Mapping[str, DeterministicSparsePlanV2],
    code: str,
    message: str,
) -> SelectionOutcome:
    return SelectionOutcome(
        selection=StructuralSelection(
            schema_version=SELECTION_SCHEMA_VERSION,
            selection_version=SELECTION_VERSION,
            seed=seed,
            status="failure",
            candidate_topic_ids=CANDIDATE_TOPIC_IDS,
            screens=tuple(screens),
            selected_topic_ids=(),
            failure=SelectionFailure(code, message),
        ),
        plans_by_topic=dict(plans),
    )


def screen_and_select_structural_topics(
    topics: Sequence[Topic],
    *,
    query_analyzer: QueryAnalyzer,
    candidate_topic_ids: Sequence[str] = CANDIDATE_TOPIC_IDS,
    seed: str = SELECTION_SEED,
) -> SelectionOutcome:
    """Screen the exact candidate universe and choose one topic per stratum."""

    if tuple(candidate_topic_ids) != CANDIDATE_TOPIC_IDS:
        raise ValueError("v2 candidate universe/order differs from the frozen set")
    if seed != SELECTION_SEED:
        raise ValueError("v2 structural selection seed differs from the frozen seed")
    if tuple(topic.id for topic in topics) != CANDIDATE_TOPIC_IDS:
        raise ValueError("candidate topic records differ from the frozen ID order")

    plans: dict[str, DeterministicSparsePlanV2] = {}
    screens: list[CandidateScreen] = []
    fingerprint = query_analyzer.fingerprint
    for topic in topics:
        plan = build_deterministic_sparse_v2_plan(
            topic_id=topic.id,
            narrative=topic.narrative,
            query_analyzer=query_analyzer,
        )
        plans[topic.id] = plan
        analyzed = query_analyzer.analyze(topic.narrative)
        if analyzed.fingerprint != fingerprint:
            raise ValueError("selection analyzer fingerprint changed")
        failure = plan.failure
        narrative_hash = _text_sha256(topic.narrative)
        screens.append(
            CandidateScreen(
                topic_id=topic.id,
                narrative_sha256=narrative_hash,
                unit_count=len(plan.lexical_units),
                original_unique_term_count=len(analyzed.unique_tokens),
                plan_status=plan.status,
                plan_semantic_sha256=semantic_plan_sha256(plan),
                eligible=plan.status == "ok",
                failure_code=failure.code if failure is not None else None,
                failure_message=failure.message if failure is not None else None,
                stratum=None,
                selection_digest=None,
            )
        )

    eligible_two = sorted(
        (row for row in screens if row.eligible and row.unit_count == 2),
        key=lambda row: (row.original_unique_term_count, int(row.topic_id)),
    )
    lower_size = math.ceil(len(eligible_two) / 2)
    strata_by_topic: dict[str, str] = {}
    for row in eligible_two[:lower_size]:
        strata_by_topic[row.topic_id] = "A"
    for row in eligible_two[lower_size:]:
        strata_by_topic[row.topic_id] = "B"
    for row in screens:
        if not row.eligible:
            continue
        if row.unit_count == 3:
            strata_by_topic[row.topic_id] = "C"
        elif row.unit_count >= 4:
            strata_by_topic[row.topic_id] = "D"

    screened_with_strata = [
        replace(
            row,
            stratum=strata_by_topic.get(row.topic_id),
            selection_digest=(
                _selection_digest(
                    topic_id=row.topic_id,
                    narrative_sha256=row.narrative_sha256,
                    seed=seed,
                )
                if row.topic_id in strata_by_topic
                else None
            ),
        )
        for row in screens
    ]
    if sum(row.eligible for row in screened_with_strata) < PILOT_TOPIC_COUNT:
        return _failure(
            seed=seed,
            screens=screened_with_strata,
            plans=plans,
            code="fewer_than_four_eligible_topics",
            message="fewer than four candidates satisfy the frozen v2 admission rule",
        )

    chosen: list[str] = []
    for stratum in STRATUM_ORDER:
        candidates = [
            row
            for row in screened_with_strata
            if row.eligible and row.stratum == stratum
        ]
        if not candidates:
            return _failure(
                seed=seed,
                screens=screened_with_strata,
                plans=plans,
                code=f"empty_stratum_{stratum}",
                message=f"selection stratum {stratum} has no eligible topic",
            )
        chosen.append(
            min(
                candidates,
                key=lambda row: (str(row.selection_digest), int(row.topic_id)),
            ).topic_id
        )

    if len(chosen) != PILOT_TOPIC_COUNT or len(set(chosen)) != PILOT_TOPIC_COUNT:
        raise AssertionError("v2 selection did not produce four distinct topics")
    selection = StructuralSelection(
        schema_version=SELECTION_SCHEMA_VERSION,
        selection_version=SELECTION_VERSION,
        seed=seed,
        status="ok",
        candidate_topic_ids=CANDIDATE_TOPIC_IDS,
        screens=tuple(screened_with_strata),
        selected_topic_ids=tuple(chosen),
        failure=None,
    )
    return SelectionOutcome(selection=selection, plans_by_topic=plans)

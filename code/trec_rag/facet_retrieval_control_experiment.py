"""Deterministic replacement and family-balanced fusion for facet controls."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .evaluation import evaluate_ranked
from .facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    ControlManifest,
    _validate_manifest,
)
from .pipeline_models import RankedCandidate, RetrievedCandidate
from .ranking import reciprocal_rank_fusion


RRF_K = 60
RANKING_DEPTH = 100
ORIGINAL_FAMILY_WEIGHT = 0.5
FACET_FAMILY_WEIGHT = 0.5
_EXPECTED_TOPIC_IDS = ("200", "225", "707", "897")
_ARM_IDS = ("B0", "W0", "W1", "W2")
_ARM_PREFERENCE = {arm_id: index for index, arm_id in enumerate(_ARM_IDS)}
EXPECTED_ALTERNATIVE_NAMES = (
    *(f"R2:200:{arm}" for arm in _ARM_IDS),
    *(
        f"R2:225:{first}-{second}"
        for first in _ARM_IDS
        for second in _ARM_IDS
    ),
    *(f"R2:707:{arm}" for arm in _ARM_IDS),
    "R2:897:B0",
)


@dataclass(frozen=True)
class StreamArmEvaluation:
    """Qrels-backed contribution metrics plus frozen qrels-free diagnostics."""

    arm_id: str
    overlap_with_original_top100: int
    relevant_at_10: int
    relevant_at_100: int
    graded_recall_at_100: float
    ndcg_at_10: float
    unique_relevant_contribution: int
    unique_graded_gain: int
    domain_drift_top10_count: int
    content_quality_top10_count: int
    coherence_failed: bool
    recall_at_100: float = 0.0
    precision_at_10: float = 0.0
    judged_rate_at_10: float = 0.0
    judged_rate_at_100: float = 0.0
    relevant_denominator: int = 0
    graded_recall_denominator: int = 0

    @property
    def top10_noise(self) -> int:
        return self.domain_drift_top10_count + self.content_quality_top10_count


def _validate_metric_rows(
    rows: Sequence[RetrievedCandidate],
    *,
    label: str,
    expected_topic_id: str | None = None,
) -> tuple[RetrievedCandidate, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.rank, row.docid, -row.score)))
    if len(ordered) != RANKING_DEPTH:
        raise ValueError(f"{label} must contain exactly 100 rows")
    if [row.rank for row in ordered] != list(range(1, RANKING_DEPTH + 1)):
        raise ValueError(f"{label} must contain canonical ranks 1 through 100")
    if len({row.docid for row in ordered}) != RANKING_DEPTH:
        raise ValueError(f"{label} must contain 100 distinct document IDs")
    topics = {row.topic_id for row in ordered}
    if len(topics) != 1:
        raise ValueError(f"{label} must contain exactly one topic")
    topic_id = next(iter(topics))
    if expected_topic_id is not None and topic_id != expected_topic_id:
        raise ValueError(f"{label} topic differs from the original stream")
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    return ordered


def _inspection_value(inspection: object, field: str) -> object:
    if isinstance(inspection, Mapping):
        if field not in inspection:
            raise ValueError(f"inspection is missing {field}")
        return inspection[field]
    if not hasattr(inspection, field):
        raise ValueError(f"inspection is missing {field}")
    return getattr(inspection, field)


def evaluate_stream_arm(
    arm_id: str,
    facet_rows: Sequence[RetrievedCandidate],
    original_rows: Sequence[RetrievedCandidate],
    topic_qrels: Mapping[str, int],
    inspection: object,
) -> StreamArmEvaluation:
    """Evaluate one frozen facet stream against its unchanged original top 100.

    Relevance uses grade >= 2. Graded recall divides the non-negative grade mass
    retrieved by the total non-negative grade mass for the topic. Both recall
    families are defined as zero when their denominator is zero.
    """

    if arm_id not in _ARM_PREFERENCE:
        raise ValueError(f"unknown control arm: {arm_id}")
    original = _validate_metric_rows(original_rows, label="original stream")
    topic_id = original[0].topic_id
    facet = _validate_metric_rows(
        facet_rows,
        label=f"facet arm {arm_id}",
        expected_topic_id=topic_id,
    )
    if not isinstance(topic_qrels, Mapping) or not all(
        isinstance(docid, str)
        and docid
        and isinstance(grade, int)
        and not isinstance(grade, bool)
        for docid, grade in topic_qrels.items()
    ):
        raise ValueError("topic qrels must map document IDs to integer grades")

    ranked = [
        RankedCandidate(
            topic_id=row.topic_id,
            docid=row.docid,
            rank=row.rank,
            score=row.score,
            text=row.text,
            provenance=[],
        )
        for row in facet
    ]
    evaluated = evaluate_ranked(
        ranked,
        {topic_id: dict(topic_qrels)},
        metric_names=(
            "relevant_count@10",
            "relevant_count@100",
            "recall@100",
            "graded_recall@100",
            "ndcg@10",
            "precision@10",
            "judged_rate@10",
            "judged_rate@100",
        ),
        relevance_threshold=2,
        topic_ids=(topic_id,),
    )["per_topic"][topic_id]
    original_docids = {row.docid for row in original}
    facet_docids = [row.docid for row in facet]
    unique_relevant = [
        docid
        for docid in facet_docids
        if docid not in original_docids and topic_qrels.get(docid, 0) >= 2
    ]
    drift = _inspection_value(inspection, "domain_drift_top10_count")
    content = _inspection_value(inspection, "content_quality_top10_count")
    coherence = _inspection_value(inspection, "coherence_failed")
    if (
        isinstance(drift, bool)
        or not isinstance(drift, int)
        or drift < 0
        or isinstance(content, bool)
        or not isinstance(content, int)
        or content < 0
        or not isinstance(coherence, bool)
    ):
        raise ValueError("inspection contains invalid coherence or top-10 noise values")
    relevant_denominator = sum(grade >= 2 for grade in topic_qrels.values())
    graded_denominator = sum(max(grade, 0) for grade in topic_qrels.values())
    return StreamArmEvaluation(
        arm_id=arm_id,
        overlap_with_original_top100=len(set(facet_docids) & original_docids),
        relevant_at_10=int(evaluated["relevant_count@10"]),
        relevant_at_100=int(evaluated["relevant_count@100"]),
        recall_at_100=float(evaluated["recall@100"]),
        graded_recall_at_100=float(evaluated["graded_recall@100"]),
        ndcg_at_10=float(evaluated["ndcg@10"]),
        precision_at_10=float(evaluated["precision@10"]),
        judged_rate_at_10=float(evaluated["judged_rate@10"]),
        judged_rate_at_100=float(evaluated["judged_rate@100"]),
        unique_relevant_contribution=len(unique_relevant),
        unique_graded_gain=sum(topic_qrels[docid] for docid in unique_relevant),
        domain_drift_top10_count=drift,
        content_quality_top10_count=content,
        coherence_failed=coherence,
        relevant_denominator=relevant_denominator,
        graded_recall_denominator=graded_denominator,
    )


def select_stream_arm(
    arms: Mapping[str, StreamArmEvaluation],
) -> StreamArmEvaluation:
    """Apply the frozen eligibility filter and lexicographic selection rule."""

    if set(arms) != set(_ARM_IDS):
        raise ValueError("stream selection requires exactly B0, W0, W1, and W2")
    if any(key != value.arm_id for key, value in arms.items()):
        raise ValueError("stream selection keys must match arm records")
    baseline = arms["B0"]
    eligible = [
        arm
        for arm in arms.values()
        if not arm.coherence_failed
        and not (
            arm.domain_drift_top10_count > baseline.domain_drift_top10_count
            and arm.content_quality_top10_count
            > baseline.content_quality_top10_count
        )
    ]
    if not eligible:
        raise ValueError("stream selection has no eligible arms")
    for arm in eligible:
        if not all(
            math.isfinite(value)
            for value in (arm.graded_recall_at_100, arm.ndcg_at_10)
        ):
            raise ValueError("stream selection metrics must be finite")
    return max(
        eligible,
        key=lambda arm: (
            arm.unique_graded_gain,
            arm.graded_recall_at_100,
            arm.ndcg_at_10,
            -arm.top10_noise,
            -_ARM_PREFERENCE[arm.arm_id],
        ),
    )


def selected_ranking_references(selected: Mapping[str, str]) -> dict[str, str]:
    """Resolve selected stream arms to the exact pre-qrels ranking names."""

    expected = {"200/f07a", "225/f02", "225/f04", "707/f02"}
    if set(selected) != expected or any(arm not in _ARM_IDS for arm in selected.values()):
        raise ValueError("selected stream arms differ from the exact four-stream boundary")
    references = {
        "200": f"R2:200:{selected['200/f07a']}",
        "225": f"R2:225:{selected['225/f02']}-{selected['225/f04']}",
        "707": f"R2:707:{selected['707/f02']}",
        "897": "R2:897:B0",
    }
    if not set(references.values()) <= set(EXPECTED_ALTERNATIVE_NAMES):
        raise AssertionError("selected ranking is outside the frozen 25 alternatives")
    return references


def retrieval_repair_decision(
    *,
    r2_graded_recall: float,
    r1_graded_recall: float,
    r2_ndcg: float,
    r1_ndcg: float,
    per_topic_ndcg_deltas: Sequence[float],
    selected_noise: int,
    b0_noise: int,
) -> str:
    """Return the preregistered binary retrieval-repair classification."""

    if len(per_topic_ndcg_deltas) != 4:
        raise ValueError("decision requires exactly four per-topic nDCG deltas")
    numeric = (
        r2_graded_recall,
        r1_graded_recall,
        r2_ndcg,
        r1_ndcg,
        *per_topic_ndcg_deltas,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
        raise ValueError("decision metrics must be finite numbers")
    if (
        isinstance(selected_noise, bool)
        or not isinstance(selected_noise, int)
        or selected_noise < 0
        or isinstance(b0_noise, bool)
        or not isinstance(b0_noise, int)
        or b0_noise < 0
    ):
        raise ValueError("decision noise counts must be non-negative integers")
    success = (
        r2_graded_recall > r1_graded_recall
        and r2_ndcg >= r1_ndcg - 0.02
        and min(per_topic_ndcg_deltas) >= -0.10
        and selected_noise <= b0_noise
    )
    return "retrieval_repair_success" if success else "retrieval_repair_failed"


def _candidate_key(row: RetrievedCandidate) -> tuple[object, ...]:
    return (
        row.topic_id,
        row.variant_name,
        row.retriever_name,
        row.rank,
        row.docid,
        -row.score,
        row.query_text,
        row.text,
    )


def _reject_protected(
    baseline_r1: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
) -> None:
    topics = {
        *(stream.topic_id for stream in manifest.streams),
        *(row.topic_id for row in baseline_r1),
        *(row.topic_id for row in control_rows),
    }
    protected = sorted(topics & set(PROTECTED_TOPIC_IDS))
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")


def _require_depth(rows: Sequence[RetrievedCandidate], label: str) -> None:
    if len(rows) != RANKING_DEPTH:
        raise ValueError(f"{label} must contain exactly 100 candidates")
    ranks = [row.rank for row in rows]
    if sorted(ranks) != list(range(1, RANKING_DEPTH + 1)):
        raise ValueError(f"{label} must contain each source rank 1 through 100")
    if len({row.docid for row in rows}) != RANKING_DEPTH:
        raise ValueError(f"{label} must contain 100 distinct document IDs")


def index_control_streams(
    baseline_r1: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
) -> dict[tuple[str, str, str], tuple[RetrievedCandidate, ...]]:
    """Index exact B0/W0/W1/W2 rows for each of the four registered streams."""

    _reject_protected(baseline_r1, control_rows, manifest)
    _validate_manifest(manifest)
    indexed_rows: dict[tuple[str, str, str], list[RetrievedCandidate]] = defaultdict(list)
    expected_control_variants: dict[tuple[str, str], tuple[str, str, str]] = {}
    for stream in manifest.streams:
        baseline_variant = f"sparse_relevance_v1:R1:{stream.stream_id}"
        baseline = [
            row
            for row in baseline_r1
            if row.topic_id == stream.topic_id and row.variant_name == baseline_variant
        ]
        indexed_rows[(stream.topic_id, stream.stream_id, "B0")].extend(baseline)
        for arm_id in ("W0", "W1", "W2"):
            expected_control_variants[(stream.topic_id, f"facet_control_v1:{arm_id}:{stream.stream_id}")] = (
                stream.topic_id,
                stream.stream_id,
                arm_id,
            )

    for row in control_rows:
        key = expected_control_variants.get((row.topic_id, row.variant_name))
        if key is None:
            raise ValueError(
                "control candidates contain a row outside the registered W0/W1/W2 streams"
            )
        indexed_rows[key].append(row)

    result: dict[tuple[str, str, str], tuple[RetrievedCandidate, ...]] = {}
    for stream in manifest.streams:
        for arm_id in ("B0", "W0", "W1", "W2"):
            key = (stream.topic_id, stream.stream_id, arm_id)
            rows = tuple(sorted(indexed_rows.get(key, ()), key=_candidate_key))
            _require_depth(rows, f"{stream.topic_id}/{stream.stream_id}/{arm_id}")
            if len({(row.variant_name, row.retriever_name) for row in rows}) != 1:
                raise ValueError(f"{key!r} must identify exactly one retrieval stream")
            result[key] = rows
    return result


def _is_original_stream(stream: tuple[str, str]) -> bool:
    variant_name, _retriever_name = stream
    return variant_name == "prompt_lab_v1:original" or variant_name.endswith(":original")


def _family_balanced_topic_fusion(
    topic_id: str,
    candidates: Sequence[RetrievedCandidate],
) -> list[RankedCandidate]:
    streams = sorted({(row.variant_name, row.retriever_name) for row in candidates})
    original_streams = [stream for stream in streams if _is_original_stream(stream)]
    facet_streams = [stream for stream in streams if not _is_original_stream(stream)]
    if len(original_streams) != 1:
        raise ValueError(f"topic {topic_id} must contain exactly one original stream")
    if not facet_streams:
        raise ValueError(f"topic {topic_id} must contain at least one facet stream")
    weights = {
        original_streams[0]: ORIGINAL_FAMILY_WEIGHT,
        **{
            stream: FACET_FAMILY_WEIGHT / len(facet_streams)
            for stream in facet_streams
        },
    }
    ranked = reciprocal_rank_fusion(
        sorted(candidates, key=_candidate_key),
        k=RRF_K,
        stream_weights=weights,
        limit=RANKING_DEPTH,
    )
    retained = [row for row in ranked if row.topic_id == topic_id]
    if len(retained) != RANKING_DEPTH or any(
        row.rank != expected for expected, row in enumerate(retained, start=1)
    ):
        raise ValueError(f"topic {topic_id} fusion did not produce exact depth 100")
    return retained


def build_topic_alternatives(
    r1_arm: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
) -> dict[str, list[RankedCandidate]]:
    """Freeze-ready 4+16+4+1 topic alternatives in deterministic arm order."""

    # This namespace firewall intentionally precedes validation, indexing, and fusion.
    _reject_protected(r1_arm, control_rows, manifest)
    _validate_manifest(manifest)
    controls = index_control_streams(r1_arm, control_rows, manifest)
    observed_topics = tuple(sorted({row.topic_id for row in r1_arm}))
    if observed_topics != _EXPECTED_TOPIC_IDS:
        raise ValueError(
            f"R1 arm topics must be exactly {_EXPECTED_TOPIC_IDS!r}; found {observed_topics!r}"
        )

    controlled_by_topic: dict[str, list[str]] = defaultdict(list)
    baseline_variants: dict[tuple[str, str], str] = {}
    for stream in manifest.streams:
        controlled_by_topic[stream.topic_id].append(stream.stream_id)
        baseline_variants[(stream.topic_id, stream.stream_id)] = (
            f"sparse_relevance_v1:R1:{stream.stream_id}"
        )

    baseline_by_topic: dict[str, list[RetrievedCandidate]] = defaultdict(list)
    for row in sorted(r1_arm, key=_candidate_key):
        baseline_by_topic[row.topic_id].append(row)
    alternatives: dict[str, list[RankedCandidate]] = {}
    for topic_id in _EXPECTED_TOPIC_IDS:
        stream_ids = controlled_by_topic.get(topic_id, [])
        combinations = (
            itertools.product(_ARM_IDS, repeat=len(stream_ids))
            if stream_ids
            else [("B0",)]
        )
        for selected_arms in combinations:
            label = "-".join(selected_arms)
            pool = list(baseline_by_topic[topic_id])
            if stream_ids:
                removed_variants = {
                    baseline_variants[(topic_id, stream_id)] for stream_id in stream_ids
                }
                pool = [row for row in pool if row.variant_name not in removed_variants]
                for stream_id, arm_id in zip(stream_ids, selected_arms, strict=True):
                    pool.extend(controls[(topic_id, stream_id, arm_id)])
            alternatives[f"R2:{topic_id}:{label}"] = _family_balanced_topic_fusion(
                topic_id, pool
            )
    if tuple(alternatives) != EXPECTED_ALTERNATIVE_NAMES:
        raise AssertionError("topic alternatives differ from the exact frozen 25-name set")
    return alternatives

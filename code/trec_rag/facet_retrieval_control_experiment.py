"""Deterministic replacement and family-balanced fusion for facet controls."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Sequence

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
    arm_ids = ("B0", "W0", "W1", "W2")
    alternatives: dict[str, list[RankedCandidate]] = {}
    for topic_id in _EXPECTED_TOPIC_IDS:
        stream_ids = controlled_by_topic.get(topic_id, [])
        combinations = itertools.product(arm_ids, repeat=len(stream_ids)) if stream_ids else [("B0",)]
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
    if len(alternatives) != 25:
        raise AssertionError(f"expected 25 topic alternatives, built {len(alternatives)}")
    return alternatives

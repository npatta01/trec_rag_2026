"""Evaluate facet-local MiniLM-B reranker candidates under the experiment firewall."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


PILOT_TOPIC_IDS = ("200", "225", "707", "897")
PILOT_TOPIC_ID_SET = frozenset(PILOT_TOPIC_IDS)
REPO_ROOT = Path(__file__).resolve().parents[2]
QRELS_CONSUMPTION_EXPERIMENT_ID = "rag25_facet_local_minilm_v1"
PREREGISTERED_RANKING_ARMS = (
    "R1_LEGACY",
    "C0_TOPIC_LOCAL",
    "BF100_TOPIC_LOCAL",
    "BF50_TOPIC_LOCAL",
    "BF20_TOPIC_LOCAL",
    "BO100_TOPIC_LOCAL",
    "BB100_TOPIC_LOCAL",
    "BF100_MAXP_TOPIC_LOCAL",
    "BF100_LEGACY_FUSION",
)
EVALUATION_ARTIFACT_NAMES = (
    "raw_union.json",
    "prefusion.json",
    "facet_retention.json",
    "systems.json",
    "gains_losses.json",
    "review_metrics.json",
    "representatives.json",
    "decision.json",
)


def _validate_topic_boundary(
    topic_ids: Iterable[Any], label: str, *, exact: bool = False
) -> set[str]:
    observed = {str(topic_id) for topic_id in topic_ids}
    outside = observed - PILOT_TOPIC_ID_SET
    if outside:
        raise ValueError(
            f"{label} topic boundary contains forbidden or out-of-scope topics: "
            + ", ".join(sorted(outside))
        )
    if exact and observed != PILOT_TOPIC_ID_SET:
        raise ValueError(
            f"{label} topic boundary must be exactly " + ", ".join(PILOT_TOPIC_IDS)
        )
    return observed


def _facet_topic_ids(
    facets: Mapping[tuple[str, str], Any], label: str
) -> set[str]:
    topics: list[str] = []
    for key in facets:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"{label} facet key is invalid")
        topics.append(str(key[0]))
    return _validate_topic_boundary(topics, label)


def _review_topic_ids(
    rows: Sequence[Mapping[str, Any]], label: str, *, exact: bool = False
) -> set[str]:
    topics: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row is invalid")
        if "topic_id" in row:
            topics.append(str(row["topic_id"]))
        memberships = row.get("memberships")
        if not isinstance(memberships, Sequence):
            raise ValueError(f"{label} memberships are invalid")
        for membership in memberships:
            if not isinstance(membership, Mapping):
                raise ValueError(f"{label} membership is invalid")
            facet = membership.get("facet")
            if not isinstance(facet, Mapping) or "topic_id" not in facet:
                raise ValueError(f"{label} facet membership is invalid")
            topics.append(str(facet["topic_id"]))
    return _validate_topic_boundary(topics, label, exact=exact)


def validate_frozen_input_topics(frozen_inputs: Mapping[str, Any]) -> None:
    """Reject incomplete or out-of-scope frozen inputs before qrels access."""

    original = frozen_inputs.get("original")
    rankings = frozen_inputs.get("rankings")
    control_facets = frozen_inputs.get("control_facets")
    bf_facets = frozen_inputs.get("bf_facets")
    stream_weights = frozen_inputs.get("stream_weights")
    review_rows = frozen_inputs.get("review_rows")
    if not all(
        isinstance(value, Mapping)
        for value in (original, rankings, control_facets, bf_facets, stream_weights)
    ) or not isinstance(review_rows, Sequence):
        raise ValueError("frozen evaluator inputs are incomplete")
    _validate_topic_boundary(original, "original ranking", exact=True)
    if set(rankings) != set(PREREGISTERED_RANKING_ARMS):
        raise ValueError("frozen evaluator inputs lack the exact preregistered system set")
    for arm in PREREGISTERED_RANKING_ARMS:
        per_topic = rankings[arm]
        if not isinstance(per_topic, Mapping):
            raise ValueError(f"final ranking {arm} is invalid")
        _validate_topic_boundary(per_topic, f"{arm} ranking", exact=True)
    control_topics = _facet_topic_ids(control_facets, "control facet streams")
    bf_topics = _facet_topic_ids(bf_facets, "BF facet streams")
    weight_topics = _facet_topic_ids(stream_weights, "facet stream weights")
    if (
        control_topics != PILOT_TOPIC_ID_SET
        or bf_topics != PILOT_TOPIC_ID_SET
        or weight_topics != PILOT_TOPIC_ID_SET
        or set(control_facets) != set(bf_facets)
        or set(control_facets) != set(stream_weights)
    ):
        raise ValueError("facet stream topic boundary or identity set is incomplete")
    _review_topic_ids(review_rows, "blinded review", exact=True)


@dataclass(frozen=True)
class DocidSetComparison:
    """Reconciled set-level gains and losses between two rankings."""

    baseline_count: int
    candidate_count: int
    gained: set[str]
    lost: set[str]

    @property
    def net_change(self) -> int:
        return self.candidate_count - self.baseline_count


def compare_docid_sets(
    baseline: Iterable[str], candidate: Iterable[str]
) -> DocidSetComparison:
    """Compare unique document identifiers and retain reconcilable counts."""

    baseline_set = {str(docid) for docid in baseline}
    candidate_set = {str(docid) for docid in candidate}
    return DocidSetComparison(
        baseline_count=len(baseline_set),
        candidate_count=len(candidate_set),
        gained=candidate_set - baseline_set,
        lost=baseline_set - candidate_set,
    )


def _deduplicate(docids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(docid) for docid in docids))


def _topic_sort_key(topic_id: str) -> tuple[int, str]:
    try:
        return (PILOT_TOPIC_IDS.index(topic_id), topic_id)
    except ValueError:
        return (len(PILOT_TOPIC_IDS), topic_id)


def _dcg(grades: Sequence[int | float]) -> float:
    return sum(
        (2 ** float(grade) - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
        if float(grade) > 0.0
    )


def evaluate_ranking(
    ranking: Sequence[str], qrels: Mapping[str, int | float]
) -> dict[str, int | float]:
    """Calculate the preregistered ranking, coverage, and judging metrics."""

    docids = _deduplicate(ranking)
    normalized_qrels = {str(docid): float(grade) for docid, grade in qrels.items()}

    top10 = docids[:10]
    top50 = docids[:50]
    top100 = docids[:100]
    binary_relevant = {
        docid for docid, grade in normalized_qrels.items() if grade >= 2.0
    }
    positive_grades = [grade for grade in normalized_qrels.values() if grade > 0.0]

    relevant_at_10 = sum(docid in binary_relevant for docid in top10)
    relevant_at_100 = sum(docid in binary_relevant for docid in top100)
    graded_gain_at_100 = sum(
        max(0.0, normalized_qrels.get(docid, 0.0)) for docid in top100
    )

    ranked_grades = [max(0.0, normalized_qrels.get(docid, 0.0)) for docid in top10]
    full_ideal_dcg = _dcg(sorted(positive_grades, reverse=True)[:10])

    def oracle_ndcg(pool: Sequence[str]) -> float:
        attainable = sorted(
            (
                max(0.0, normalized_qrels.get(docid, 0.0))
                for docid in pool
                if normalized_qrels.get(docid, 0.0) > 0.0
            ),
            reverse=True,
        )[:10]
        return _dcg(attainable) / full_ideal_dcg if full_ideal_dcg else 0.0

    return {
        "relevant_count@10": relevant_at_10,
        "relevant_count@100": relevant_at_100,
        "precision@10": relevant_at_10 / 10.0,
        "recall@100": (
            relevant_at_100 / len(binary_relevant) if binary_relevant else 0.0
        ),
        "graded_recall@100": (
            graded_gain_at_100 / sum(positive_grades) if positive_grades else 0.0
        ),
        "ndcg@10": _dcg(ranked_grades) / full_ideal_dcg if full_ideal_dcg else 0.0,
        "judged_rate@10": sum(docid in normalized_qrels for docid in top10) / 10.0,
        "judged_rate@100": (
            sum(docid in normalized_qrels for docid in top100) / 100.0
        ),
        "oracle_ndcg@10_from_top50": oracle_ndcg(top50),
        "oracle_ndcg@10_from_top100": oracle_ndcg(top100),
    }


def build_union_curves(
    original: Mapping[str, Sequence[str]],
    control_facets: Mapping[tuple[str, str], Sequence[str]],
    bf_facets: Mapping[tuple[str, str], Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    depths: Sequence[int] = (20, 50, 100),
) -> dict[str, dict[int, dict[str, Any]]]:
    """Build paired, deduplicated pre-fusion candidate-union curves."""

    original_topics = _validate_topic_boundary(original, "original union input")
    control_topics = _facet_topic_ids(control_facets, "control union facets")
    bf_topics = _facet_topic_ids(bf_facets, "BF union facets")
    _validate_topic_boundary(qrels, "union qrels")
    if original_topics != control_topics or original_topics != bf_topics:
        raise ValueError("union input topic boundary differs between sources")
    if set(control_facets) != set(bf_facets):
        raise ValueError("control and BF must contain the same facet streams")
    if 100 in depths:
        mismatches = [
            key
            for key in control_facets
            if set(_deduplicate(control_facets[key])[:100])
            != set(_deduplicate(bf_facets[key])[:100])
        ]
        if mismatches:
            raise ValueError("control and BF depth-100 streams must have the same raw candidates")

    topic_ids = {
        *(str(topic_id) for topic_id in original),
        *(str(topic_id) for topic_id, _ in control_facets),
        *(str(topic_id) for topic_id, _ in bf_facets),
    }
    ordered_topics = sorted(topic_ids, key=_topic_sort_key)
    systems = {
        "C0_TOPIC_LOCAL": control_facets,
        "BF100_TOPIC_LOCAL": bf_facets,
    }
    result: dict[str, dict[int, dict[str, Any]]] = {}

    for arm, facet_rankings in systems.items():
        arm_curves: dict[int, dict[str, Any]] = {}
        for depth in depths:
            per_topic: dict[str, dict[str, Any]] = {}
            for topic_id in ordered_topics:
                union = set(_deduplicate(original.get(topic_id, ())))
                for (facet_topic, _), ranking in facet_rankings.items():
                    if str(facet_topic) == topic_id:
                        union.update(_deduplicate(ranking)[:depth])
                docids = sorted(union)
                topic_qrels = qrels.get(topic_id, {})
                relevant_docids = sorted(
                    docid for docid in union if float(topic_qrels.get(docid, 0)) >= 2
                )
                positive_denominator = sum(
                    max(0.0, float(grade)) for grade in topic_qrels.values()
                )
                graded_gain = sum(
                    max(0.0, float(topic_qrels.get(docid, 0))) for docid in union
                )
                relevant_denominator = sum(
                    float(grade) >= 2 for grade in topic_qrels.values()
                )
                per_topic[topic_id] = {
                    "docids": docids,
                    "unique_candidate_documents": len(docids),
                    "relevant_docids": relevant_docids,
                    "relevant_documents": len(relevant_docids),
                    "graded_relevant_gain": graded_gain,
                    "recall": (
                        len(relevant_docids) / relevant_denominator
                        if relevant_denominator
                        else 0.0
                    ),
                    "graded_recall": (
                        graded_gain / positive_denominator
                        if positive_denominator
                        else 0.0
                    ),
                }
            arm_curves[int(depth)] = {
                "per_topic": per_topic,
                "macro": {
                    "recall": fmean(row["recall"] for row in per_topic.values())
                    if per_topic
                    else 0.0,
                    "graded_recall": fmean(
                        row["graded_recall"] for row in per_topic.values()
                    )
                    if per_topic
                    else 0.0,
                },
            }
        result[arm] = arm_curves
    return result


def build_facet_retention(
    facets_by_arm: Mapping[
        str, Mapping[tuple[str, str], Sequence[str]]
    ],
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    baselines: Mapping[str, Mapping[str, Iterable[str]]],
    depths: Sequence[int] = (20, 50, 100),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Report relevant retention and unique contribution for every facet."""

    facet_topics: set[str] = set()
    for arm, facets in facets_by_arm.items():
        facet_topics.update(_facet_topic_ids(facets, f"{arm} retention facets"))
    _validate_topic_boundary(qrels, "retention qrels")
    for baseline_name, per_topic in baselines.items():
        _validate_topic_boundary(per_topic, f"{baseline_name} retention baseline")
    if not facet_topics:
        raise ValueError("facet retention topic boundary is empty")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, facets in facets_by_arm.items():
        arm_rows: dict[str, dict[str, Any]] = {}
        for (raw_topic_id, variant_name), ranking in facets.items():
            topic_id = str(raw_topic_id)
            docids = _deduplicate(ranking)
            topic_qrels = qrels.get(topic_id, {})
            relevant = {
                docid for docid in docids if float(topic_qrels.get(docid, 0)) >= 2
            }
            arm_rows[f"{topic_id}/{variant_name}"] = {
                "relevant_retained": {
                    str(depth): len(
                        {
                            docid
                            for docid in docids[:depth]
                            if float(topic_qrels.get(docid, 0)) >= 2
                        }
                    )
                    for depth in depths
                },
                "unique_relevant_beyond": {
                    baseline_name: sorted(
                        relevant
                        - {
                            str(docid)
                            for docid in baseline_topics.get(topic_id, ())
                        }
                    )
                    for baseline_name, baseline_topics in baselines.items()
                },
            }
        result[arm] = arm_rows
    return result


def compute_prefusion_evidence(
    *,
    control_facets: Mapping[tuple[str, str], Sequence[str]],
    bf_facets: Mapping[tuple[str, str], Sequence[str]],
    stream_weights: Mapping[tuple[str, str], int | float],
    control_final: Mapping[str, Sequence[str]],
    bf_final: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    promotion_depth: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Keep raw candidates, local-rank evidence, and final fusion sets distinct."""

    control_topics = _facet_topic_ids(control_facets, "control prefusion facets")
    bf_topics = _facet_topic_ids(bf_facets, "BF prefusion facets")
    weight_topics = _facet_topic_ids(stream_weights, "prefusion stream weights")
    control_final_topics = _validate_topic_boundary(
        control_final, "control final ranking"
    )
    bf_final_topics = _validate_topic_boundary(bf_final, "BF final ranking")
    _validate_topic_boundary(qrels, "prefusion qrels")
    if not (
        control_topics
        == bf_topics
        == weight_topics
        == control_final_topics
        == bf_final_topics
    ) or set(control_facets) != set(bf_facets) or set(control_facets) != set(
        stream_weights
    ):
        raise ValueError("prefusion topic boundary or facet identity set differs")
    facet_keys = sorted(
        set(control_facets) | set(bf_facets),
        key=lambda key: (_topic_sort_key(str(key[0])), str(key[1])),
    )
    topic_ids = {
        *(str(topic_id) for topic_id, _ in facet_keys),
        *(str(topic_id) for topic_id in control_final),
        *(str(topic_id) for topic_id in bf_final),
    }
    ordered_topics = sorted(topic_ids, key=_topic_sort_key)

    raw_unions: dict[str, set[str]] = {topic_id: set() for topic_id in ordered_topics}
    best_control: dict[tuple[str, str], int] = {}
    best_bf: dict[tuple[str, str], int] = {}
    facet_rows: list[dict[str, Any]] = []

    for raw_key in facet_keys:
        raw_topic_id, variant_name = raw_key
        topic_id = str(raw_topic_id)
        control = _deduplicate(control_facets.get(raw_key, ()))
        bf = _deduplicate(bf_facets.get(raw_key, ()))
        raw_unions[topic_id].update(control)
        raw_unions[topic_id].update(bf)
        control_ranks = {docid: rank for rank, docid in enumerate(control, start=1)}
        bf_ranks = {docid: rank for rank, docid in enumerate(bf, start=1)}
        candidate_docids = sorted(
            set(control) | set(bf),
            key=lambda docid: (
                bf_ranks.get(docid, math.inf),
                control_ranks.get(docid, math.inf),
                docid,
            ),
        )

        for docid in candidate_docids:
            control_rank = control_ranks.get(docid)
            bf_rank = bf_ranks.get(docid)
            if control_rank is not None:
                best_control[(topic_id, docid)] = min(
                    best_control.get((topic_id, docid), control_rank), control_rank
                )
            if bf_rank is not None:
                best_bf[(topic_id, docid)] = min(
                    best_bf.get((topic_id, docid), bf_rank), bf_rank
                )
            weight = float(stream_weights.get(raw_key, 0.0))
            qrel_grade = int(qrels.get(topic_id, {}).get(docid, 0))
            facet_rows.append(
                {
                    "topic_id": topic_id,
                    "variant_name": str(variant_name),
                    "docid": docid,
                    "control_rank": control_rank,
                    "bf_rank": bf_rank,
                    "qrel_grade": qrel_grade,
                    "is_relevant": qrel_grade >= 2,
                    "stream_weight": weight,
                    "rrf_contribution": (
                        weight / (rrf_k + bf_rank) if bf_rank is not None else 0.0
                    ),
                }
            )

    control_final_sets = {
        topic_id: set(_deduplicate(control_final.get(topic_id, ())))
        for topic_id in ordered_topics
    }
    bf_final_sets = {
        topic_id: set(_deduplicate(bf_final.get(topic_id, ())))
        for topic_id in ordered_topics
    }
    promoted: list[str] = []
    for (topic_id, docid), bf_rank in sorted(best_bf.items()):
        if (
            float(qrels.get(topic_id, {}).get(docid, 0)) >= 2
            and docid not in control_final_sets[topic_id]
            and bf_rank <= promotion_depth
            and best_control.get((topic_id, docid), math.inf) > promotion_depth
        ):
            promoted.append(f"{topic_id}/{docid}")

    blocked = [
        qualified
        for qualified in promoted
        if qualified.split("/", 1)[1]
        not in bf_final_sets[qualified.split("/", 1)[0]]
    ]
    final_novel = sorted(
        f"{topic_id}/{docid}"
        for topic_id in ordered_topics
        for docid in bf_final_sets[topic_id] - control_final_sets[topic_id]
        if float(qrels.get(topic_id, {}).get(docid, 0)) >= 2
    )

    return {
        "raw_union_docids": {
            topic_id: sorted(raw_unions[topic_id]) for topic_id in ordered_topics
        },
        "facet_rows": facet_rows,
        "control_final_docids": {
            topic_id: sorted(control_final_sets[topic_id]) for topic_id in ordered_topics
        },
        "final_docids": {
            topic_id: sorted(bf_final_sets[topic_id]) for topic_id in ordered_topics
        },
        "pre_fusion_promoted_novel_docids": promoted,
        "fusion_blocked_novel_docids": blocked,
        "final_novel_docids": final_novel,
    }


def aggregate_review_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate blinded labels within facets, then macro-average by arm."""

    _review_topic_ids(rows, "blinded review")
    metric_names = (
        "direct_answer",
        "partial_or_related",
        "not_facet_relevant",
        "wrong_domain",
        "low_quality",
    )
    by_arm: dict[str, dict[str, int]] = defaultdict(
        lambda: {"denominator": 0, **{name: 0 for name in metric_names}}
    )
    by_facet: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"denominator": 0, **{name: 0 for name in metric_names}}
    )

    for row in rows:
        label = row["label"]
        relevance = str(label.get("relevance", ""))
        for membership in row.get("memberships", ()):
            arm = str(membership["arm"])
            facet = membership["facet"]
            facet_key = (
                arm,
                str(facet["topic_id"]),
                str(facet["variant_name"]),
            )
            for counts in (by_arm[arm], by_facet[facet_key]):
                counts["denominator"] += 1
                if relevance in counts:
                    counts[relevance] += 1
                counts["wrong_domain"] += int(bool(label.get("wrong_domain")))
                counts["low_quality"] += int(bool(label.get("low_quality")))

    per_facet_by_arm: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
    for (arm, topic_id, variant_name), counts in by_facet.items():
        denominator = counts["denominator"]
        per_facet_by_arm[arm][f"{topic_id}/{variant_name}"] = {
            **counts,
            **{
                f"{name}_rate": counts[name] / denominator
                for name in metric_names
            },
        }

    macro_by_arm: dict[str, dict[str, float]] = {}
    for arm, facets in per_facet_by_arm.items():
        macro_by_arm[arm] = {
            f"{name}_rate": fmean(
                float(facet[f"{name}_rate"]) for facet in facets.values()
            )
            for name in metric_names
        }

    return {
        "counts_by_arm": {arm: dict(counts) for arm, counts in by_arm.items()},
        "per_facet_by_arm": {arm: dict(facets) for arm, facets in per_facet_by_arm.items()},
        "macro_by_arm": macro_by_arm,
    }


def select_diagnostic_outcome(
    *,
    headroom: int,
    pre_fusion_promoted_novel: int,
    fusion_blocked_novel: int,
    final_novel: int,
    final_novel_vs_legacy_r1: int,
    net_relevant_change_vs_control: int,
    macro_deltas_vs_control: Mapping[str, float],
    per_topic_deltas_vs_control: Mapping[str, Mapping[str, float]],
    review_deltas: Mapping[str, float],
    macro_deltas_vs_legacy_r1: Mapping[str, float],
    per_topic_deltas_vs_legacy_r1: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the preregistered diagnostic table in its exact order."""

    def every_topic_at_least(
        deltas: Mapping[str, Mapping[str, float]], metric: str, floor: float
    ) -> bool:
        return all(float(topic[metric]) >= floor for topic in deltas.values())

    guards = (
        ("final_novel", final_novel >= 1),
        (
            "positive_net_relevant_change_vs_control",
            net_relevant_change_vs_control > 0,
        ),
        (
            "positive_macro_recall_vs_control",
            float(macro_deltas_vs_control["recall@100"]) > 0.0,
        ),
        (
            "nonnegative_macro_graded_recall_vs_control",
            float(macro_deltas_vs_control["graded_recall@100"]) >= 0.0,
        ),
        (
            "per_topic_recall_floor_vs_control",
            every_topic_at_least(
                per_topic_deltas_vs_control, "recall@100", -0.02
            ),
        ),
        (
            "per_topic_graded_recall_floor_vs_control",
            every_topic_at_least(
                per_topic_deltas_vs_control, "graded_recall@100", -0.02
            ),
        ),
        (
            "positive_direct_answer_rate_delta",
            float(review_deltas["direct_answer_rate"]) > 0.0,
        ),
        (
            "nonpositive_wrong_domain_rate_delta",
            float(review_deltas["wrong_domain_rate"]) <= 0.0,
        ),
        (
            "macro_ndcg_floor_vs_control",
            float(macro_deltas_vs_control["ndcg@10"]) >= -0.02,
        ),
        (
            "per_topic_ndcg_floor_vs_control",
            every_topic_at_least(per_topic_deltas_vs_control, "ndcg@10", -0.10),
        ),
        (
            "nonnegative_macro_recall_vs_legacy_r1",
            float(macro_deltas_vs_legacy_r1["recall@100"]) >= 0.0,
        ),
        (
            "nonnegative_macro_graded_recall_vs_legacy_r1",
            float(macro_deltas_vs_legacy_r1["graded_recall@100"]) >= 0.0,
        ),
        (
            "per_topic_recall_floor_vs_legacy_r1",
            every_topic_at_least(
                per_topic_deltas_vs_legacy_r1, "recall@100", -0.02
            ),
        ),
        (
            "per_topic_graded_recall_floor_vs_legacy_r1",
            every_topic_at_least(
                per_topic_deltas_vs_legacy_r1, "graded_recall@100", -0.02
            ),
        ),
        (
            "macro_ndcg_floor_vs_legacy_r1",
            float(macro_deltas_vs_legacy_r1["ndcg@10"]) >= -0.02,
        ),
        (
            "per_topic_ndcg_floor_vs_legacy_r1",
            every_topic_at_least(
                per_topic_deltas_vs_legacy_r1, "ndcg@10", -0.10
            ),
        ),
    )
    failed_guards = [name for name, passed in guards if not passed]
    macro_recall = float(macro_deltas_vs_control["recall@100"])
    macro_ndcg = float(macro_deltas_vs_control["ndcg@10"])

    if headroom == 0:
        outcome = "candidate_generation_gap"
    elif not failed_guards:
        outcome = "B_promotes_coverage"
    elif final_novel >= 1 or macro_recall > 0.0:
        outcome = "B_coverage_gain_with_regression"
    elif (
        headroom > 0
        and float(review_deltas["direct_answer_rate"]) > 0.0
        and float(review_deltas["wrong_domain_rate"]) <= 0.0
        and pre_fusion_promoted_novel >= 1
        and fusion_blocked_novel >= 1
        and macro_recall <= 0.0
    ):
        outcome = "B_filters_but_fusion_blocks"
    elif final_novel == 0 and macro_recall <= 0.0 and macro_ndcg > 0.0:
        outcome = "ranking_only_gain"
    else:
        outcome = "B_query_window_or_model_gap"

    return {
        "outcome": outcome,
        "stage_a_permitted": outcome == "B_promotes_coverage",
        "stage_a_executed": False,
        "failed_promotion_guards": failed_guards,
        "final_novel_vs_legacy_r1": final_novel_vs_legacy_r1,
    }


def _canonical_json_bytes(payload: Mapping[str, Any], *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def add_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical-content-bound copy of an evaluation artifact."""

    result = dict(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


def validate_self_hash(payload: Mapping[str, Any]) -> bool:
    """Reject an artifact whose claimed content hash no longer matches."""

    claimed = payload.get("artifact_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("artifact self-hash is missing or invalid")
    without_hash = dict(payload)
    without_hash.pop("artifact_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(without_hash)).hexdigest()
    if not hmac.compare_digest(claimed, actual):
        raise ValueError("artifact self-hash differs from its content")
    return True


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_json_object_bytes(source: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_jsonl_bytes(source: bytes, label: str) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in source.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{label} rows must be JSON objects")
    return values


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validated_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _read_authenticated_bytes(
    path: Path, expected_sha256: Any, label: str
) -> bytes:
    expected = _validated_sha256(expected_sha256, f"{label} hash")
    try:
        source = Path(path).read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    actual = hashlib.sha256(source).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"{label} authenticated artifact hash differs")
    return source


def _validate_prior_embedded_topics(value: Any) -> None:
    """Validate every explicit topic declaration in an authenticated prior."""

    def validate_collection(collection: Any, label: str) -> None:
        if isinstance(collection, Mapping):
            topic_ids = list(collection)
        elif isinstance(collection, Sequence) and not isinstance(
            collection, (str, bytes, bytearray)
        ):
            topic_ids = []
            for entry in collection:
                if isinstance(entry, Mapping):
                    if "topic_id" not in entry:
                        raise ValueError(f"{label} topic boundary shape is invalid")
                    topic_ids.append(entry["topic_id"])
                elif isinstance(entry, (str, int)):
                    topic_ids.append(entry)
                else:
                    raise ValueError(f"{label} topic boundary shape is invalid")
        else:
            raise ValueError(f"{label} topic boundary shape is invalid")
        observed = _validate_topic_boundary(topic_ids, label, exact=True)
        if len(topic_ids) != len(observed):
            raise ValueError(f"{label} topic boundary contains duplicate topics")

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            if "topic_id" in child:
                _validate_topic_boundary(
                    (child["topic_id"],), "prior evaluation row"
                )
            if "topic_ids" in child:
                declared = child["topic_ids"]
                if not isinstance(declared, Sequence) or isinstance(
                    declared, (str, bytes, bytearray)
                ):
                    raise ValueError(
                        "prior evaluation topic boundary declaration is invalid"
                    )
                _validate_topic_boundary(
                    declared, "prior evaluation declaration", exact=True
                )
                if [str(topic_id) for topic_id in declared] != list(PILOT_TOPIC_IDS):
                    raise ValueError(
                        "prior evaluation topic boundary must declare the exact pilot order"
                    )
            for key in ("topics", "per_topic"):
                if key in child:
                    validate_collection(
                        child[key], f"prior evaluation {key}"
                    )
            for nested in child.values():
                visit(nested)
        elif isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            for nested in child:
                visit(nested)

    visit(value)


def load_authenticated_prior_evaluation(
    prior_evaluation_manifest: Path,
    prior_evaluation: Path,
) -> dict[str, Any]:
    """Authenticate a pilot-only prior evaluation before returning parsed bytes."""

    manifest_path = Path(prior_evaluation_manifest)
    try:
        manifest_source = manifest_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("prior evaluation manifest is missing or invalid") from exc
    manifest = _parse_json_object_bytes(
        manifest_source, "prior evaluation manifest"
    )
    if set(manifest) != {
        "manifest_sha256",
        "prior_evaluation",
        "schema_version",
        "status",
        "topic_ids",
    } or (
        manifest.get("schema_version")
        != "facet-local-minilm-prior-evaluation-manifest-v1"
        or manifest.get("status") != "authenticated_prior_evaluation"
    ):
        raise ValueError("prior evaluation manifest contract differs")

    declared_topics = manifest.get("topic_ids")
    if not isinstance(declared_topics, Sequence) or isinstance(
        declared_topics, (str, bytes, bytearray)
    ):
        raise ValueError("prior evaluation manifest topic boundary is invalid")
    _validate_topic_boundary(
        declared_topics, "prior evaluation manifest", exact=True
    )
    if [str(topic_id) for topic_id in declared_topics] != list(PILOT_TOPIC_IDS):
        raise ValueError(
            "prior evaluation manifest topic boundary must declare exact pilot order"
        )

    claimed_manifest_hash = _validated_sha256(
        manifest.get("manifest_sha256"), "prior evaluation manifest self hash"
    )
    without_manifest_hash = dict(manifest)
    without_manifest_hash.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        _canonical_json_bytes(without_manifest_hash, newline=True)
    ).hexdigest()
    if not hmac.compare_digest(claimed_manifest_hash, actual_manifest_hash):
        raise ValueError("prior evaluation manifest self hash differs")

    record = manifest.get("prior_evaluation")
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "schema_version",
        "sha256",
    }:
        raise ValueError("prior evaluation manifest artifact binding differs")
    raw_relative = record.get("path")
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ValueError("prior evaluation path binding is missing")
    relative = Path(raw_relative)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_relative
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("prior evaluation path binding is invalid")
    declared_path = manifest_path.parent / relative
    provided_path = Path(prior_evaluation)
    if declared_path.resolve() != provided_path.resolve():
        raise ValueError("prior evaluation path binding differs")

    expected_schema = record.get("schema_version")
    if not isinstance(expected_schema, str) or not expected_schema:
        raise ValueError("prior evaluation schema binding is invalid")
    expected_hash = _validated_sha256(
        record.get("sha256"), "prior evaluation artifact hash"
    )
    source = _read_authenticated_bytes(
        provided_path, expected_hash, "prior evaluation"
    )
    prior = _parse_json_object_bytes(source, "prior evaluation")
    if prior.get("schema_version") != expected_schema:
        raise ValueError("prior evaluation schema binding differs")
    if "artifact_sha256" in prior:
        validate_self_hash(prior)
    _validate_prior_embedded_topics(prior)
    return {
        "prior_evaluation": prior,
        "prior_evaluation_manifest_sha256": hashlib.sha256(
            manifest_source
        ).hexdigest(),
        "prior_evaluation_sha256": expected_hash,
    }


def verify_ranking_freeze(freeze: Path) -> dict[str, Any]:
    """Verify the complete pre-qrels ranking freeze through its owner module."""

    freeze = Path(freeze)
    if not (freeze / "freeze.json").is_file():
        raise ValueError("ranking freeze is incomplete")
    from .facet_local_minilm_rank import verify_freeze

    try:
        return dict(verify_freeze(freeze))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("ranking freeze is incomplete") from exc


def verify_blinded_review_freeze(
    review_freeze: Path, ranking_freeze: Path
) -> dict[str, Any]:
    """Verify the qrels-blind review and its binding to the ranking freeze."""

    review_freeze = Path(review_freeze)
    if not (review_freeze / "review_freeze.json").is_file():
        raise ValueError("blinded-review freeze is incomplete")
    from .facet_local_minilm_review import verify_review_freeze

    try:
        return dict(
            verify_review_freeze(
                review_freeze,
                ranking_freeze_dir=Path(ranking_freeze),
            )
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("blinded-review freeze is incomplete") from exc


def load_frozen_inputs(
    ranking_freeze: Path,
    review_freeze: Path,
    prior_evaluation: Path,
    *,
    prior_evaluation_manifest: Path,
    ranking_freeze_sha256: str,
    review_freeze_sha256: str,
) -> dict[str, Any]:
    """Load frozen inputs from the exact bytes authenticated by both verifiers."""

    ranking_hash = _validated_sha256(
        ranking_freeze_sha256, "ranking freeze hash"
    )
    review_hash = _validated_sha256(review_freeze_sha256, "review freeze hash")
    ranking_root = Path(ranking_freeze)
    try:
        ranking_root_source = (ranking_root / "freeze.json").read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("ranking freeze is missing or invalid") from exc
    freeze = _parse_json_object_bytes(ranking_root_source, "ranking freeze")
    if (
        freeze.get("schema_version") != "facet-local-minilm-ranking-freeze-v1"
        or freeze.get("status") != "frozen_before_qrels"
        or freeze.get("qrels_opened") is not False
    ):
        raise ValueError("ranking freeze contract differs")
    if freeze.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError("ranking freeze topic boundary must be exactly the pilot topics")
    claimed_ranking_hash = _validated_sha256(
        freeze.get("freeze_sha256"), "ranking freeze self hash"
    )
    without_ranking_hash = dict(freeze)
    without_ranking_hash.pop("freeze_sha256", None)
    recomputed_ranking_hash = hashlib.sha256(
        _canonical_json_bytes(without_ranking_hash, newline=True)
    ).hexdigest()
    if not hmac.compare_digest(claimed_ranking_hash, recomputed_ranking_hash):
        raise ValueError("ranking freeze self hash differs")
    if not hmac.compare_digest(claimed_ranking_hash, ranking_hash):
        raise ValueError("ranking freeze changed after verification")

    raw_ranking_records = freeze.get("rankings")
    if not isinstance(raw_ranking_records, Mapping):
        raise ValueError("ranking freeze lacks final ranking records")
    if set(raw_ranking_records) != set(PREREGISTERED_RANKING_ARMS):
        raise ValueError("ranking freeze lacks the exact preregistered system set")
    stream_records = freeze.get("streams")
    if not isinstance(stream_records, Sequence):
        raise ValueError("ranking freeze lacks stream records")

    # The root declarations are the only data inspected before this complete
    # boundary check. No ranking or stream artifact may be opened first.
    declared_stream_topics: list[str] = []
    for raw_record in stream_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("ranking freeze has an invalid stream record")
        if "topic_id" not in raw_record:
            raise ValueError("stream topic boundary is missing")
        declared_stream_topics.append(str(raw_record["topic_id"]))
    _validate_topic_boundary(
        declared_stream_topics, "declared stream", exact=True
    )

    artifact_records = freeze.get("artifacts")
    if not isinstance(artifact_records, Mapping):
        raise ValueError("ranking freeze lacks artifact hash bindings")

    def bound_path(raw_path: Any, label: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{label} path is missing")
        relative = Path(raw_path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(f"{label} path escapes the verified ranking freeze")
        path = ranking_root / relative
        try:
            path.resolve().relative_to(ranking_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{label} path escapes the verified ranking freeze"
            ) from exc
        return path

    def read_bound_bytes(
        raw_path: Any, label: str, *, record_sha256: Any | None = None
    ) -> bytes:
        path = bound_path(raw_path, label)
        relative = path.relative_to(ranking_root).as_posix()
        raw_artifact = artifact_records.get(relative)
        if not isinstance(raw_artifact, Mapping):
            raise ValueError(f"{label} lacks an authenticated artifact binding")
        artifact_hash = _validated_sha256(
            raw_artifact.get("sha256"), f"{label} artifact hash"
        )
        if record_sha256 is not None:
            record_hash = _validated_sha256(
                record_sha256, f"{label} declared hash"
            )
            if not hmac.compare_digest(record_hash, artifact_hash):
                raise ValueError(f"{label} hash binding differs")
        return _read_authenticated_bytes(path, artifact_hash, label)

    def docid(row: Mapping[str, Any]) -> str:
        value = row.get("docid", row.get("document_id"))
        if not isinstance(value, str) or not value:
            raise ValueError("frozen ranking row lacks a document ID")
        return value

    rankings: dict[str, dict[str, list[str]]] = {}
    ranking_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    document_details: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in PREREGISTERED_RANKING_ARMS:
        raw_record = raw_ranking_records[arm]
        if not isinstance(raw_record, Mapping):
            raise ValueError("ranking freeze has an invalid final ranking record")
        rows = _parse_jsonl_bytes(
            read_bound_bytes(
                raw_record.get("path"),
                f"{arm} ranking",
                record_sha256=raw_record.get("file_sha256"),
            ),
            f"{arm} ranking",
        )
        _validate_topic_boundary(
            (row.get("topic_id") for row in rows),
            f"{arm} ranking",
            exact=True,
        )
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            topic_id = str(row.get("topic_id"))
            by_topic[topic_id].append(row)
            document_details.setdefault((topic_id, docid(row)), row)
        ordered = {
            topic_id: sorted(
                by_topic.get(topic_id, ()),
                key=lambda row: (int(row["rank"]), docid(row)),
            )
            for topic_id in PILOT_TOPIC_IDS
        }
        rankings[str(arm)] = {
            topic_id: [docid(row) for row in ordered[topic_id]]
            for topic_id in PILOT_TOPIC_IDS
        }
        ranking_rows[str(arm)] = ordered

    original: dict[str, list[str]] = {}
    control_facets: dict[tuple[str, str], list[str]] = {}
    bf_facets: dict[tuple[str, str], list[str]] = {}
    facet_stream_details: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw_record in stream_records:
        aggregation = str(raw_record.get("aggregation"))
        family = str(raw_record.get("family"))
        if aggregation not in {"bm25", "top4"}:
            continue
        topic_id = str(raw_record.get("topic_id"))
        variant = str(raw_record.get("variant_name"))
        rows = _parse_jsonl_bytes(
            read_bound_bytes(
                raw_record.get("path"),
                "stream ranking",
                record_sha256=raw_record.get("file_sha256"),
            ),
            "stream ranking",
        )
        row_topics = _validate_topic_boundary(
            (row.get("topic_id") for row in rows),
            "stream ranking",
        )
        if row_topics != {topic_id}:
            raise ValueError("stream ranking topic boundary differs from its declaration")
        ordered_rows = sorted(
            rows, key=lambda row: (int(row["rank"]), docid(row))
        )
        document_ids = [docid(row) for row in ordered_rows]
        for row in ordered_rows:
            document_details.setdefault((topic_id, docid(row)), row)
        if family == "original" and aggregation == "bm25":
            if topic_id in original:
                raise ValueError("ranking freeze contains duplicate original streams")
            original[topic_id] = document_ids
        elif family == "facet":
            key = (topic_id, variant)
            destination = control_facets if aggregation == "bm25" else bf_facets
            if key in destination:
                raise ValueError("ranking freeze contains duplicate facet streams")
            destination[key] = document_ids
            for row in ordered_rows:
                facet_stream_details[(topic_id, variant, aggregation, docid(row))] = row
    if set(original) != set(PILOT_TOPIC_IDS) or set(control_facets) != set(bf_facets):
        raise ValueError("ranking freeze stream population is incomplete")

    fusion_tables = freeze.get("fusion_tables")
    if not isinstance(fusion_tables, Mapping):
        raise ValueError("ranking freeze lacks fusion weight tables")
    weight_payload = _parse_json_object_bytes(
        read_bound_bytes(
            fusion_tables.get("family_rrf_topic_local_v2"),
            "topic-local fusion weights",
        ),
        "topic-local fusion weights",
    )
    raw_weights = weight_payload.get("weights")
    if not isinstance(raw_weights, Sequence):
        raise ValueError("topic-local fusion weights are invalid")
    _validate_topic_boundary(
        (
            row.get("topic_id")
            for row in raw_weights
            if isinstance(row, Mapping)
        ),
        "topic-local fusion weights",
        exact=True,
    )
    stream_weights = {
        (str(row["topic_id"]), str(row["variant_name"])): float(row["weight"])
        for row in raw_weights
        if isinstance(row, Mapping)
        and (str(row.get("topic_id")), str(row.get("variant_name")))
        in control_facets
    }
    if set(stream_weights) != set(control_facets):
        raise ValueError("facet streams lack exact topic-local fusion weights")

    prior_binding = load_authenticated_prior_evaluation(
        Path(prior_evaluation_manifest), Path(prior_evaluation)
    )
    review_root = Path(review_freeze)
    try:
        review_root_source = (review_root / "review_freeze.json").read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("blinded-review freeze is incomplete") from exc
    review = _parse_json_object_bytes(
        review_root_source, "blinded-review freeze"
    )
    if (
        review.get("schema_version") != "facet-local-minilm-review-freeze-v2"
        or review.get("status") != "review_frozen_before_qrels"
        or review.get("qrels_opened") is not False
    ):
        raise ValueError("blinded-review freeze contract differs")
    claimed_review_hash = _validated_sha256(
        review.get("review_freeze_sha256"), "review freeze self hash"
    )
    without_review_hash = dict(review)
    without_review_hash.pop("review_freeze_sha256", None)
    recomputed_review_hash = hashlib.sha256(
        _pretty_json_bytes(without_review_hash)
    ).hexdigest()
    if not hmac.compare_digest(claimed_review_hash, recomputed_review_hash):
        raise ValueError("review freeze self hash differs")
    if not hmac.compare_digest(claimed_review_hash, review_hash):
        raise ValueError("review freeze changed after verification")
    review_bindings = review.get("bindings")
    if not isinstance(review_bindings, Mapping):
        raise ValueError("blinded-review freeze bindings are invalid")
    bound_ranking_hash = _validated_sha256(
        review_bindings.get("ranking_freeze_sha256"),
        "review ranking freeze binding",
    )
    if not hmac.compare_digest(bound_ranking_hash, ranking_hash):
        raise ValueError("review freeze ranking binding differs")
    review_rows = _parse_jsonl_bytes(
        _read_authenticated_bytes(
            review_root / "unmasked_items.jsonl",
            review_bindings.get("unmasked_items_sha256"),
            "blinded-review unmasked items",
        ),
        "blinded-review unmasked items",
    )
    _review_topic_ids(review_rows, "blinded review", exact=True)
    result = {
        "bf_facets": bf_facets,
        "control_facets": control_facets,
        "document_details": document_details,
        "facet_stream_details": facet_stream_details,
        "original": original,
        **prior_binding,
        "ranking_rows": ranking_rows,
        "rankings": rankings,
        "review_rows": review_rows,
        "stream_weights": stream_weights,
    }
    validate_frozen_input_topics(result)
    return result


def validate_qrels_authorization(
    qrels_manifest: Path,
    qrels_approval: Path,
    *,
    ranking_freeze_sha256: str,
    review_freeze_sha256: str,
) -> dict[str, Any]:
    """Validate the exact projected-qrels authorization without opening qrels."""

    ranking_freeze_sha256 = _validated_sha256(
        ranking_freeze_sha256, "ranking freeze hash"
    )
    review_freeze_sha256 = _validated_sha256(
        review_freeze_sha256, "review freeze hash"
    )
    manifest_path = Path(qrels_manifest)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_value = json.loads(manifest_bytes)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("qrels projection manifest is missing or invalid") from exc
    if not isinstance(manifest_value, dict):
        raise ValueError("qrels projection manifest must be a JSON object")
    manifest = manifest_value
    if (
        manifest.get("schema_version") != "pilot-qrels-projection-v1"
        or manifest.get("status") != "authorized_projection"
    ):
        raise ValueError("qrels projection manifest contract differs")
    if manifest.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError(
            "qrels projection manifest must declare exactly topics "
            + ", ".join(PILOT_TOPIC_IDS)
        )

    projection_sha256 = _validated_sha256(
        manifest.get("projection_sha256"), "qrels projection hash"
    )
    projection_name = manifest.get("projection_path")
    if not isinstance(projection_name, str) or not projection_name:
        raise ValueError("qrels projection path is missing")
    declared_projection = Path(projection_name)
    if declared_projection.is_absolute() or ".." in declared_projection.parts:
        raise ValueError("qrels projection path must remain inside its authorized directory")
    projection_path = manifest_path.parent / declared_projection
    try:
        projection_path.resolve().relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(
            "qrels projection path must remain inside its authorized directory"
        ) from exc

    approval_path = Path(qrels_approval)
    try:
        approval_bytes = approval_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("qrels access approval is missing or invalid") from exc
    approval = _parse_json_object_bytes(approval_bytes, "qrels access approval")
    if (
        approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
    ):
        raise ValueError("qrels access approval contract differs")
    if approval.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError("qrels access approval must bind exactly topics")

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if approval.get("qrels_manifest_sha256") != manifest_sha256:
        raise ValueError("qrels manifest hash binding differs")
    if approval.get("qrels_projection_sha256") != projection_sha256:
        raise ValueError("qrels projection hash binding differs")
    if approval.get("ranking_freeze_sha256") != ranking_freeze_sha256:
        raise ValueError("ranking freeze hash binding differs")
    if approval.get("review_freeze_sha256") != review_freeze_sha256:
        raise ValueError("review freeze hash binding differs")

    return {
        "approval_path": approval_path,
        "approval_sha256": hashlib.sha256(approval_bytes).hexdigest(),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "projection_path": projection_path,
        "projection_sha256": projection_sha256,
        "topic_ids": list(PILOT_TOPIC_IDS),
    }


def trusted_qrels_consumption_dir() -> Path:
    """Return a registry namespace shared by every linked repository worktree."""

    completed = subprocess.run(
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(completed.stdout.strip()).resolve()
    if not common.is_dir():
        raise ValueError("git common directory is unavailable for qrels guard state")
    return (
        common
        / "trec-rag-qrels-consumptions"
        / QRELS_CONSUMPTION_EXPERIMENT_ID
    )


def qrels_consumption_identity(qrels_approval: Path) -> dict[str, str]:
    """Derive an immutable, caller-path-independent approval identity."""

    try:
        approval_bytes = Path(qrels_approval).read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("qrels access approval is missing or invalid") from exc
    approval = _parse_json_object_bytes(approval_bytes, "qrels access approval")
    if (
        approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
        or approval.get("topic_ids") != list(PILOT_TOPIC_IDS)
    ):
        raise ValueError("qrels access approval contract differs")
    identity = {
        "experiment_id": QRELS_CONSUMPTION_EXPERIMENT_ID,
        "qrels_approval_sha256": hashlib.sha256(approval_bytes).hexdigest(),
        "qrels_manifest_sha256": _validated_sha256(
            approval.get("qrels_manifest_sha256"), "approval qrels manifest hash"
        ),
        "qrels_projection_sha256": _validated_sha256(
            approval.get("qrels_projection_sha256"), "approval qrels projection hash"
        ),
        "ranking_freeze_sha256": _validated_sha256(
            approval.get("ranking_freeze_sha256"), "approval ranking freeze hash"
        ),
        "review_freeze_sha256": _validated_sha256(
            approval.get("review_freeze_sha256"), "approval review freeze hash"
        ),
    }
    return {
        **identity,
        "identity_sha256": hashlib.sha256(_canonical_json_bytes(identity)).hexdigest(),
    }


def _registry_path_for_identity(identity: Mapping[str, Any]) -> Path:
    identity_sha256 = _validated_sha256(
        identity.get("identity_sha256"), "qrels consumption identity hash"
    )
    return trusted_qrels_consumption_dir() / f"{identity_sha256}.json"


def qrels_consumption_registry_path(qrels_approval: Path) -> Path:
    """Return the trusted path for an approval's immutable identity."""

    return _registry_path_for_identity(qrels_consumption_identity(qrels_approval))


def _qrels_consumption_payload(
    *,
    identity: Mapping[str, Any],
    approval_sha256: str,
    qrels_manifest_sha256: str,
    qrels_projection_sha256: str,
    ranking_freeze_sha256: str,
    review_freeze_sha256: str,
    output: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    expected_identity_fields = {
        "qrels_approval_sha256": approval_sha256,
        "qrels_manifest_sha256": qrels_manifest_sha256,
        "qrels_projection_sha256": qrels_projection_sha256,
        "ranking_freeze_sha256": ranking_freeze_sha256,
        "review_freeze_sha256": review_freeze_sha256,
    }
    if any(
        identity.get(field) != _validated_sha256(value, field)
        for field, value in expected_identity_fields.items()
    ) or identity.get("experiment_id") != QRELS_CONSUMPTION_EXPERIMENT_ID:
        raise ValueError("qrels consumption identity binding differs")
    return add_self_hash(
        {
            "schema_version": "facet-local-minilm-qrels-consumption-v2",
            "status": "qrels_access_consumed",
            "registry_namespace": "git_common_dir_path_independent_identity",
            "experiment_id": QRELS_CONSUMPTION_EXPERIMENT_ID,
            "identity_sha256": _validated_sha256(
                identity.get("identity_sha256"), "qrels consumption identity hash"
            ),
            "topic_ids": list(PILOT_TOPIC_IDS),
            "qrels_approval_sha256": _validated_sha256(
                approval_sha256, "qrels approval hash"
            ),
            "qrels_manifest_sha256": _validated_sha256(
                qrels_manifest_sha256, "qrels manifest hash"
            ),
            "qrels_projection_sha256": _validated_sha256(
                qrels_projection_sha256, "qrels projection hash"
            ),
            "ranking_freeze_sha256": _validated_sha256(
                ranking_freeze_sha256, "ranking freeze hash"
            ),
            "review_freeze_sha256": _validated_sha256(
                review_freeze_sha256, "review freeze hash"
            ),
            "canonical_output_path": str(Path(output).resolve()),
            "output_receipt_path": str(Path(output_receipt).resolve()),
        }
    )


def _create_qrels_consumption_registry(
    registry_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically consume an approval before any output or qrels access."""

    registry = dict(payload)
    validate_self_hash(registry)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            registry_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError("qrels approval was already consumed") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_canonical_json_bytes(registry, newline=True))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return registry


def adopt_qrels_consumption_registry(
    qrels_approval: Path,
    *,
    qrels_access_receipt: Path,
    evaluation_output: Path,
) -> dict[str, Any]:
    """Adopt a verified completed evaluation without opening qrels again."""

    approval_path = Path(qrels_approval)
    identity = qrels_consumption_identity(approval_path)
    registry_path = _registry_path_for_identity(identity)
    if registry_path.exists() or registry_path.is_symlink():
        raise FileExistsError("qrels approval was already consumed")
    try:
        approval_bytes = approval_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("qrels access approval is missing or invalid") from exc
    approval = _parse_json_object_bytes(approval_bytes, "qrels access approval")
    if (
        approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
        or approval.get("topic_ids") != list(PILOT_TOPIC_IDS)
    ):
        raise ValueError("qrels access approval contract differs")

    output_path = Path(evaluation_output)
    receipt_path = Path(qrels_access_receipt)
    if receipt_path.resolve() != (output_path / "qrels_access_receipt.json").resolve():
        raise ValueError("qrels access receipt must belong to the evaluation output")
    receipt = _read_json_object(receipt_path, "qrels access receipt")
    validate_self_hash(receipt)
    if (
        receipt.get("schema_version")
        != "facet-local-minilm-qrels-access-receipt-v1"
        or receipt.get("status") != "qrels_access_consumed"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
    ):
        raise ValueError("qrels access receipt contract differs")
    for field in (
        "qrels_manifest_sha256",
        "qrels_projection_sha256",
        "ranking_freeze_sha256",
        "review_freeze_sha256",
    ):
        approval_value = _validated_sha256(approval.get(field), f"approval {field}")
        receipt_value = _validated_sha256(receipt.get(field), f"receipt {field}")
        if not hmac.compare_digest(approval_value, receipt_value):
            raise ValueError(f"qrels approval and receipt {field} binding differs")
    for name in EVALUATION_ARTIFACT_NAMES:
        artifact = _read_json_object(output_path / name, f"evaluation artifact {name}")
        validate_self_hash(artifact)
        bindings = artifact.get("bindings")
        if not isinstance(bindings, Mapping) or any(
            bindings.get(field) != receipt.get(field)
            for field in (
                "qrels_manifest_sha256",
                "qrels_projection_sha256",
                "ranking_freeze_sha256",
                "review_freeze_sha256",
            )
        ):
            raise ValueError("evaluation artifact qrels binding differs")

    payload = _qrels_consumption_payload(
        identity=identity,
        approval_sha256=hashlib.sha256(approval_bytes).hexdigest(),
        qrels_manifest_sha256=str(receipt["qrels_manifest_sha256"]),
        qrels_projection_sha256=str(receipt["qrels_projection_sha256"]),
        ranking_freeze_sha256=str(receipt["ranking_freeze_sha256"]),
        review_freeze_sha256=str(receipt["review_freeze_sha256"]),
        output=output_path,
        output_receipt=receipt_path,
    )
    return _create_qrels_consumption_registry(registry_path, payload)


def export_qrels_consumption_registry_mirror(
    qrels_approval: Path, *, output: Path
) -> dict[str, Any]:
    """Publish a create-only local mirror of the trusted global registry."""

    identity = qrels_consumption_identity(Path(qrels_approval))
    registry_path = _registry_path_for_identity(identity)
    try:
        registry_bytes = registry_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("trusted qrels consumption registry is missing") from exc
    registry = _parse_json_object_bytes(
        registry_bytes, "trusted qrels consumption registry"
    )
    validate_self_hash(registry)
    if (
        registry.get("schema_version")
        != "facet-local-minilm-qrels-consumption-v2"
        or registry.get("identity_sha256") != identity["identity_sha256"]
        or registry.get("qrels_approval_sha256")
        != identity["qrels_approval_sha256"]
    ):
        raise ValueError("trusted qrels consumption registry identity differs")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError("qrels consumption registry mirror already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(registry_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return registry


def read_qrels(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, dict[str, int]]:
    """Read the already-authorized four-topic projection in TREC qrels format."""

    result: dict[str, dict[str, int]] = defaultdict(dict)
    try:
        source = Path(path).read_bytes()
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise ValueError("authorized qrels projection is missing or invalid") from exc
    if expected_sha256 is not None:
        expected = _validated_sha256(expected_sha256, "authorized projection hash")
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("authorized qrels projection hash differs")
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("authorized qrels projection is missing or invalid") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid qrels row at line {line_number}")
        topic_id, _, docid, raw_grade = fields
        if topic_id not in PILOT_TOPIC_IDS:
            raise ValueError("qrels projection contains a topic outside the authorization")
        try:
            grade = int(raw_grade)
        except ValueError as exc:
            raise ValueError(f"invalid qrels grade at line {line_number}") from exc
        if docid in result[topic_id] and result[topic_id][docid] != grade:
            raise ValueError(f"conflicting duplicate qrels row at line {line_number}")
        result[topic_id][docid] = grade
    if list(topic_id for topic_id in PILOT_TOPIC_IDS if topic_id in result) != list(
        PILOT_TOPIC_IDS
    ):
        raise ValueError("qrels projection does not contain exactly the authorized topics")
    return {topic_id: result[topic_id] for topic_id in PILOT_TOPIC_IDS}


def _create_qrels_access_receipt(
    receipt_path: Path,
    *,
    authorization: Mapping[str, Any],
    ranking_freeze_sha256: str,
    review_freeze_sha256: str,
) -> dict[str, Any]:
    receipt = add_self_hash(
        {
            "schema_version": "facet-local-minilm-qrels-access-receipt-v1",
            "status": "qrels_access_consumed",
            "topic_ids": list(PILOT_TOPIC_IDS),
            "qrels_manifest_sha256": authorization["manifest_sha256"],
            "qrels_projection_sha256": authorization["projection_sha256"],
            "ranking_freeze_sha256": ranking_freeze_sha256,
            "review_freeze_sha256": review_freeze_sha256,
        }
    )
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json_bytes(receipt, newline=True).decode("utf-8"))
    except FileExistsError as exc:
        raise FileExistsError("repeat qrels access is refused") from exc
    return receipt


def _relevant_docids(
    ranking: Iterable[str], topic_qrels: Mapping[str, int | float]
) -> set[str]:
    return {
        str(docid)
        for docid in ranking
        if float(topic_qrels.get(str(docid), 0)) >= 2.0
    }


def _serialized_comparison(
    baseline: Iterable[str], candidate: Iterable[str]
) -> dict[str, Any]:
    comparison = compare_docid_sets(baseline, candidate)
    return {
        "baseline_count": comparison.baseline_count,
        "candidate_count": comparison.candidate_count,
        "gained": sorted(comparison.gained),
        "lost": sorted(comparison.lost),
        "net_change": comparison.net_change,
    }


def _bound_artifact(
    schema_version: str,
    bindings: Mapping[str, Any],
    content: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = json.loads(
        _canonical_json_bytes(
            {
                "schema_version": schema_version,
                "bindings": dict(bindings),
                **content,
            }
        )
    )
    return add_self_hash(normalized)


def _representative_provenance_row(
    qualified_docid: str,
    *,
    evidence_class: str,
    prefusion: Mapping[str, Any],
    frozen_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Select provenance from the facet event that explains a representative."""

    topic_id, document_id = qualified_docid.split("/", 1)
    _validate_topic_boundary((topic_id,), "representative provenance")
    raw_facet_rows = prefusion.get("facet_rows", ())
    candidate_events = [
        row
        for row in raw_facet_rows
        if isinstance(row, Mapping)
        and str(row.get("topic_id")) == topic_id
        and str(row.get("docid")) == document_id
    ] if isinstance(raw_facet_rows, Sequence) else []

    def event_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        control_rank = row.get("control_rank")
        bf_rank = row.get("bf_rank")
        control_value = int(control_rank) if control_rank is not None else 10**9
        bf_value = int(bf_rank) if bf_rank is not None else 10**9
        if evidence_class in {"promoted", "gained"}:
            qualifying = bf_value <= 20 and control_value > 20
            return (
                not qualifying,
                bf_value,
                -control_value,
                str(row.get("variant_name", "")),
            )
        demoting = control_value < bf_value
        return (
            not demoting,
            control_value,
            -bf_value,
            str(row.get("variant_name", "")),
        )

    event = min(candidate_events, key=event_key) if candidate_events else {}
    variant = str(event.get("variant_name", ""))
    facet_stream_details = frozen_inputs.get("facet_stream_details", {})
    raw_stream = (
        facet_stream_details.get((topic_id, variant, "top4", document_id), {})
        if isinstance(facet_stream_details, Mapping) and variant
        else {}
    )
    stream = raw_stream if isinstance(raw_stream, Mapping) else {}
    raw_windows = stream.get("selected_windows", ())
    windows = (
        [window for window in raw_windows if isinstance(window, Mapping)]
        if isinstance(raw_windows, Sequence)
        and not isinstance(raw_windows, (str, bytes, bytearray))
        else []
    )
    selected_window = (
        max(
            windows,
            key=lambda window: (
                float(window.get("score", float("-inf"))),
                str(window.get("window_id", "")),
            ),
        )
        if windows
        else None
    )
    details = frozen_inputs.get("document_details", {})
    raw_detail = (
        details.get((topic_id, document_id), {})
        if isinstance(details, Mapping)
        else {}
    )
    detail = raw_detail if isinstance(raw_detail, Mapping) else {}
    passage = (
        str(selected_window.get("window_text", ""))
        if selected_window is not None
        else str(
            stream.get(
                "passage",
                stream.get("text", detail.get("text", detail.get("passage", ""))),
            )
        )
    )
    stream_provenance = {
        key: stream[key]
        for key in (
            "aggregation",
            "family",
            "prior_rank",
            "query",
            "query_sha256",
            "rank",
            "retriever_name",
            "score",
            "source_score",
        )
        if key in stream
    }
    rankings = frozen_inputs.get("rankings", {})

    def final_rank(arm: str) -> int | None:
        raw_arm = rankings.get(arm, {}) if isinstance(rankings, Mapping) else {}
        raw_ranking = raw_arm.get(topic_id, ()) if isinstance(raw_arm, Mapping) else ()
        ranking = [str(value) for value in raw_ranking]
        try:
            return ranking.index(document_id) + 1
        except ValueError:
            return None

    return {
        "topic_id": topic_id,
        "document_id": document_id,
        "evidence_class": evidence_class,
        "facet_variant": variant or None,
        "facet_control_rank": event.get("control_rank"),
        "facet_bf_rank": event.get("bf_rank"),
        "c0_final_rank": final_rank("C0_TOPIC_LOCAL"),
        "bf_final_rank": final_rank("BF100_TOPIC_LOCAL"),
        "passage": passage,
        "selected_minilm_window": dict(selected_window)
        if selected_window is not None
        else None,
        "stream_provenance": stream_provenance,
        "provenance": [stream_provenance] if stream_provenance else detail.get(
            "provenance", detail.get("selected_windows", [])
        ),
    }


def validate_representative_provenance(
    representatives: Mapping[str, Any],
) -> bool:
    """Fail closed unless every saved representative proves its evidence class."""

    for evidence_class in ("promoted", "demoted", "gained", "lost"):
        raw_rows = representatives.get(evidence_class)
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            raise ValueError("representative provenance class rows are invalid")
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("representative provenance row is invalid")
            topic_id = str(raw_row.get("topic_id", ""))
            _validate_topic_boundary((topic_id,), "representative provenance")
            if raw_row.get("evidence_class") != evidence_class:
                raise ValueError("representative evidence class binding differs")
            window = raw_row.get("selected_minilm_window")
            if not isinstance(window, Mapping):
                raise ValueError("representative selected MiniLM window is missing")
            window_text = window.get("window_text")
            if not isinstance(window_text, str) or not window_text:
                raise ValueError("representative selected window text is missing")
            if raw_row.get("passage") != window_text:
                raise ValueError("representative passage differs from selected window text")
            stream_provenance = raw_row.get("stream_provenance")
            if not isinstance(stream_provenance, Mapping) or not stream_provenance:
                raise ValueError("representative stream provenance is empty")
            if evidence_class == "promoted":
                control_rank = raw_row.get("facet_control_rank")
                bf_rank = raw_row.get("facet_bf_rank")
                if (
                    isinstance(control_rank, bool)
                    or not isinstance(control_rank, int)
                    or isinstance(bf_rank, bool)
                    or not isinstance(bf_rank, int)
                    or not (bf_rank <= 20 < control_rank)
                ):
                    raise ValueError("representative promoted predicate differs")
            if evidence_class == "demoted":
                control_final_rank = raw_row.get("c0_final_rank")
                bf_final_rank = raw_row.get("bf_final_rank")
                if (
                    isinstance(control_final_rank, bool)
                    or not isinstance(control_final_rank, int)
                    or isinstance(bf_final_rank, bool)
                    or not isinstance(bf_final_rank, int)
                    or not (control_final_rank < bf_final_rank)
                ):
                    raise ValueError("representative demoted predicate differs")
    return True


def derive_representative_provenance_v2(
    freeze: Path,
    *,
    review_freeze: Path,
    prior_evaluation: Path,
    prior_evaluation_manifest: Path,
    prefusion: Path,
    representatives: Path,
    output: Path,
) -> dict[str, Any]:
    """Create authenticated representative provenance without opening qrels."""

    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("representative provenance output already exists")
    ranking = verify_ranking_freeze(Path(freeze))
    review = verify_blinded_review_freeze(Path(review_freeze), Path(freeze))
    ranking_hash = _validated_sha256(
        ranking.get("freeze_sha256"), "ranking freeze hash"
    )
    review_hash = _validated_sha256(
        review.get("review_freeze_sha256"), "review freeze hash"
    )
    frozen_inputs = load_frozen_inputs(
        Path(freeze),
        Path(review_freeze),
        Path(prior_evaluation),
        prior_evaluation_manifest=Path(prior_evaluation_manifest),
        ranking_freeze_sha256=ranking_hash,
        review_freeze_sha256=review_hash,
    )
    prefusion_payload = _read_json_object(Path(prefusion), "prefusion artifact")
    representatives_payload = _read_json_object(
        Path(representatives), "representatives artifact"
    )
    validate_self_hash(prefusion_payload)
    validate_self_hash(representatives_payload)
    for label, payload in (
        ("prefusion", prefusion_payload),
        ("representatives", representatives_payload),
    ):
        bindings = payload.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"{label} artifact lacks freeze bindings")
        if (
            bindings.get("ranking_freeze_sha256") != ranking_hash
            or bindings.get("review_freeze_sha256") != review_hash
        ):
            raise ValueError(f"{label} artifact freeze binding differs")

    derived_rows: dict[str, list[dict[str, Any]]] = {}
    for evidence_class in ("promoted", "demoted", "gained", "lost"):
        raw_rows = representatives_payload.get(evidence_class, ())
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            raise ValueError("representative class rows are invalid")
        qualified_docids: list[str] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("representative row is invalid")
            topic_id = str(raw_row.get("topic_id", ""))
            document_id = str(raw_row.get("document_id", ""))
            _validate_topic_boundary((topic_id,), "representative provenance")
            if not document_id:
                raise ValueError("representative row lacks a document ID")
            qualified_docids.append(f"{topic_id}/{document_id}")
        derived_rows[evidence_class] = [
            _representative_provenance_row(
                qualified_docid,
                evidence_class=evidence_class,
                prefusion=prefusion_payload,
                frozen_inputs=frozen_inputs,
            )
            for qualified_docid in qualified_docids
        ]

    payload = add_self_hash(
        {
            "schema_version": "facet-local-minilm-representative-provenance-v2",
            "status": "offline_derived_from_frozen_artifacts",
            "topic_ids": list(PILOT_TOPIC_IDS),
            "bindings": {
                "ranking_freeze_sha256": ranking_hash,
                "review_freeze_sha256": review_hash,
                "source_prefusion_artifact_sha256": prefusion_payload[
                    "artifact_sha256"
                ],
                "source_representatives_artifact_sha256": representatives_payload[
                    "artifact_sha256"
                ],
            },
            **derived_rows,
        }
    )
    validate_representative_provenance(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json_bytes(payload, newline=True).decode("utf-8"))
    except FileExistsError as exc:
        raise FileExistsError("representative provenance output already exists") from exc
    return payload


def build_evaluation_artifacts(
    frozen_inputs: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    bindings: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the eight deterministic, self-hashed Task 6 artifacts."""

    validate_frozen_input_topics(frozen_inputs)
    _validate_topic_boundary(qrels, "evaluation qrels", exact=True)
    if tuple(qrels) != PILOT_TOPIC_IDS:
        raise ValueError("evaluation qrels must contain exactly the pilot topics")
    original = frozen_inputs.get("original")
    control_facets = frozen_inputs.get("control_facets")
    bf_facets = frozen_inputs.get("bf_facets")
    rankings = frozen_inputs.get("rankings")
    stream_weights = frozen_inputs.get("stream_weights")
    review_rows = frozen_inputs.get("review_rows")
    if not all(
        isinstance(value, Mapping)
        for value in (original, control_facets, bf_facets, rankings, stream_weights)
    ) or not isinstance(review_rows, Sequence):
        raise ValueError("frozen evaluator inputs are incomplete")
    if set(rankings) != set(PREREGISTERED_RANKING_ARMS):
        raise ValueError("frozen evaluator inputs lack the exact preregistered system set")

    system_rankings: dict[str, dict[str, list[str]]] = {
        "O": {
            topic_id: _deduplicate(original.get(topic_id, ()))
            for topic_id in PILOT_TOPIC_IDS
        }
    }
    for arm, per_topic in rankings.items():
        if not isinstance(per_topic, Mapping):
            raise ValueError(f"final ranking {arm} is invalid")
        system_rankings[str(arm)] = {
            topic_id: _deduplicate(per_topic.get(topic_id, ()))
            for topic_id in PILOT_TOPIC_IDS
        }

    system_results: dict[str, dict[str, Any]] = {}
    for arm, per_topic_rankings in system_rankings.items():
        per_topic = {
            topic_id: evaluate_ranking(
                per_topic_rankings[topic_id], qrels[topic_id]
            )
            for topic_id in PILOT_TOPIC_IDS
        }
        metric_names = tuple(next(iter(per_topic.values())))
        system_results[arm] = {
            "docids": per_topic_rankings,
            "metrics": {
                metric: fmean(
                    float(per_topic[topic_id][metric])
                    for topic_id in PILOT_TOPIC_IDS
                )
                for metric in metric_names
            },
            "per_topic": per_topic,
        }

    baselines = {
        name: {
            topic_id: set(system_rankings[name][topic_id])
            for topic_id in PILOT_TOPIC_IDS
        }
        for name in ("O", "C0_TOPIC_LOCAL", "R1_LEGACY")
    }
    curves = build_union_curves(
        original,
        control_facets,
        bf_facets,
        qrels,
        depths=(20, 50, 100),
    )
    for arm_curves in curves.values():
        for curve in arm_curves.values():
            aggregate_candidates: set[str] = set()
            aggregate_baselines: dict[str, set[str]] = {
                name: set() for name in baselines
            }
            for topic_id in PILOT_TOPIC_IDS:
                topic_curve = curve["per_topic"][topic_id]
                candidate_relevant = _relevant_docids(
                    topic_curve["docids"], qrels[topic_id]
                )
                comparisons: dict[str, Any] = {}
                for baseline_name, baseline_topics in baselines.items():
                    baseline_relevant = _relevant_docids(
                        baseline_topics[topic_id], qrels[topic_id]
                    )
                    comparisons[baseline_name] = _serialized_comparison(
                        baseline_relevant, candidate_relevant
                    )
                    aggregate_baselines[baseline_name].update(
                        f"{topic_id}/{docid}" for docid in baseline_relevant
                    )
                topic_curve["comparisons"] = comparisons
                aggregate_candidates.update(
                    f"{topic_id}/{docid}" for docid in candidate_relevant
                )
            curve["comparisons"] = {
                baseline_name: _serialized_comparison(
                    baseline_relevant, aggregate_candidates
                )
                for baseline_name, baseline_relevant in aggregate_baselines.items()
            }
            curve["aggregate"] = {
                "unique_candidate_documents": sum(
                    int(curve["per_topic"][topic_id]["unique_candidate_documents"])
                    for topic_id in PILOT_TOPIC_IDS
                ),
                "relevant_documents": len(aggregate_candidates),
                "graded_relevant_gain": sum(
                    float(curve["per_topic"][topic_id]["graded_relevant_gain"])
                    for topic_id in PILOT_TOPIC_IDS
                ),
                "macro_recall": float(curve["macro"]["recall"]),
                "macro_graded_recall": float(curve["macro"]["graded_recall"]),
            }
    headroom_by_topic: dict[str, list[str]] = {}
    for topic_id in PILOT_TOPIC_IDS:
        raw_control = curves["C0_TOPIC_LOCAL"][100]["per_topic"][topic_id][
            "docids"
        ]
        raw_relevant = _relevant_docids(raw_control, qrels[topic_id])
        final_control = _relevant_docids(
            system_rankings["C0_TOPIC_LOCAL"][topic_id], qrels[topic_id]
        )
        headroom_by_topic[topic_id] = sorted(raw_relevant - final_control)
    headroom_docids = sorted(
        f"{topic_id}/{docid}"
        for topic_id in PILOT_TOPIC_IDS
        for docid in headroom_by_topic[topic_id]
    )

    prefusion = compute_prefusion_evidence(
        control_facets=control_facets,
        bf_facets=bf_facets,
        stream_weights=stream_weights,
        control_final=system_rankings["C0_TOPIC_LOCAL"],
        bf_final=system_rankings["BF100_TOPIC_LOCAL"],
        qrels=qrels,
        promotion_depth=20,
    )
    retention = build_facet_retention(
        {
            "C0_TOPIC_LOCAL": control_facets,
            "BF100_TOPIC_LOCAL": bf_facets,
        },
        qrels,
        baselines=baselines,
        depths=(20, 50, 100),
    )
    review_metrics = aggregate_review_metrics(review_rows)

    gains_losses: dict[str, dict[str, Any]] = {}
    for arm, per_topic_rankings in system_rankings.items():
        against: dict[str, Any] = {}
        for baseline_name in ("O", "C0_TOPIC_LOCAL", "R1_LEGACY"):
            per_topic_comparisons: dict[str, Any] = {}
            baseline_all: set[str] = set()
            candidate_all: set[str] = set()
            for topic_id in PILOT_TOPIC_IDS:
                baseline_relevant = _relevant_docids(
                    system_rankings[baseline_name][topic_id], qrels[topic_id]
                )
                candidate_relevant = _relevant_docids(
                    per_topic_rankings[topic_id], qrels[topic_id]
                )
                per_topic_comparisons[topic_id] = _serialized_comparison(
                    baseline_relevant, candidate_relevant
                )
                baseline_all.update(
                    f"{topic_id}/{docid}" for docid in baseline_relevant
                )
                candidate_all.update(
                    f"{topic_id}/{docid}" for docid in candidate_relevant
                )
            against[baseline_name] = {
                "aggregate": _serialized_comparison(baseline_all, candidate_all),
                "per_topic": per_topic_comparisons,
            }
        gains_losses[arm] = against

    def metric_deltas(
        candidate: str, baseline: str
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        metrics = ("recall@100", "graded_recall@100", "ndcg@10")
        macro = {
            metric: float(system_results[candidate]["metrics"][metric])
            - float(system_results[baseline]["metrics"][metric])
            for metric in metrics
        }
        per_topic = {
            topic_id: {
                metric: float(
                    system_results[candidate]["per_topic"][topic_id][metric]
                )
                - float(system_results[baseline]["per_topic"][topic_id][metric])
                for metric in metrics
            }
            for topic_id in PILOT_TOPIC_IDS
        }
        return macro, per_topic

    macro_vs_control, per_topic_vs_control = metric_deltas(
        "BF100_TOPIC_LOCAL", "C0_TOPIC_LOCAL"
    )
    macro_vs_legacy, per_topic_vs_legacy = metric_deltas(
        "BF100_TOPIC_LOCAL", "R1_LEGACY"
    )
    review_macro = review_metrics.get("macro_by_arm", {})
    if not isinstance(review_macro, Mapping) or not {
        "C0_TOPIC_LOCAL",
        "BF50_TOPIC_LOCAL",
    }.issubset(review_macro):
        raise ValueError("blinded review lacks both preregistered arms")
    review_deltas = {
        metric: float(review_macro["BF50_TOPIC_LOCAL"][metric])
        - float(review_macro["C0_TOPIC_LOCAL"][metric])
        for metric in ("direct_answer_rate", "wrong_domain_rate")
    }

    control_relevant_all = {
        f"{topic_id}/{docid}"
        for topic_id in PILOT_TOPIC_IDS
        for docid in _relevant_docids(
            system_rankings["C0_TOPIC_LOCAL"][topic_id], qrels[topic_id]
        )
    }
    legacy_relevant_all = {
        f"{topic_id}/{docid}"
        for topic_id in PILOT_TOPIC_IDS
        for docid in _relevant_docids(
            system_rankings["R1_LEGACY"][topic_id], qrels[topic_id]
        )
    }
    bf_relevant_all = {
        f"{topic_id}/{docid}"
        for topic_id in PILOT_TOPIC_IDS
        for docid in _relevant_docids(
            system_rankings["BF100_TOPIC_LOCAL"][topic_id], qrels[topic_id]
        )
    }
    final_novel_docids = sorted(bf_relevant_all - control_relevant_all)
    final_novel_vs_legacy_docids = sorted(bf_relevant_all - legacy_relevant_all)
    evidence = {
        "headroom": len(headroom_docids),
        "headroom_docids": headroom_docids,
        "pre_fusion_promoted_novel": len(
            prefusion["pre_fusion_promoted_novel_docids"]
        ),
        "fusion_blocked_novel": len(prefusion["fusion_blocked_novel_docids"]),
        "final_novel": len(final_novel_docids),
        "final_novel_docids": final_novel_docids,
        "final_novel_vs_legacy_r1": len(final_novel_vs_legacy_docids),
        "final_novel_vs_legacy_r1_docids": final_novel_vs_legacy_docids,
        "net_relevant_change_vs_control": len(bf_relevant_all)
        - len(control_relevant_all),
        "macro_deltas_vs_control": macro_vs_control,
        "per_topic_deltas_vs_control": per_topic_vs_control,
        "review_deltas": review_deltas,
        "macro_deltas_vs_legacy_r1": macro_vs_legacy,
        "per_topic_deltas_vs_legacy_r1": per_topic_vs_legacy,
    }
    decision = select_diagnostic_outcome(
        headroom=evidence["headroom"],
        pre_fusion_promoted_novel=evidence["pre_fusion_promoted_novel"],
        fusion_blocked_novel=evidence["fusion_blocked_novel"],
        final_novel=evidence["final_novel"],
        final_novel_vs_legacy_r1=evidence["final_novel_vs_legacy_r1"],
        net_relevant_change_vs_control=evidence[
            "net_relevant_change_vs_control"
        ],
        macro_deltas_vs_control=macro_vs_control,
        per_topic_deltas_vs_control=per_topic_vs_control,
        review_deltas=review_deltas,
        macro_deltas_vs_legacy_r1=macro_vs_legacy,
        per_topic_deltas_vs_legacy_r1=per_topic_vs_legacy,
    )

    control_ranks = {
        (topic_id, docid): rank
        for topic_id in PILOT_TOPIC_IDS
        for rank, docid in enumerate(
            system_rankings["C0_TOPIC_LOCAL"][topic_id], start=1
        )
    }
    bf_ranks = {
        (topic_id, docid): rank
        for topic_id in PILOT_TOPIC_IDS
        for rank, docid in enumerate(
            system_rankings["BF100_TOPIC_LOCAL"][topic_id], start=1
        )
    }
    demoted_docids = sorted(
        f"{topic_id}/{docid}"
        for (topic_id, docid), control_rank in control_ranks.items()
        if (topic_id, docid) in bf_ranks
        and bf_ranks[(topic_id, docid)] > control_rank
        and float(qrels[topic_id].get(docid, 0)) >= 2
    )
    representatives = {
        "promoted": [
            _representative_provenance_row(
                docid,
                evidence_class="promoted",
                prefusion=prefusion,
                frozen_inputs=frozen_inputs,
            )
            for docid in prefusion["pre_fusion_promoted_novel_docids"][:3]
        ],
        "demoted": [
            _representative_provenance_row(
                docid,
                evidence_class="demoted",
                prefusion=prefusion,
                frozen_inputs=frozen_inputs,
            )
            for docid in demoted_docids[:3]
        ],
        "gained": [
            _representative_provenance_row(
                docid,
                evidence_class="gained",
                prefusion=prefusion,
                frozen_inputs=frozen_inputs,
            )
            for docid in final_novel_docids[:3]
        ],
        "lost": [
            _representative_provenance_row(
                docid,
                evidence_class="lost",
                prefusion=prefusion,
                frozen_inputs=frozen_inputs,
            )
            for docid in sorted(control_relevant_all - bf_relevant_all)[:3]
        ],
    }
    validate_representative_provenance(representatives)

    artifacts = {
        "raw_union.json": _bound_artifact(
            "facet-local-minilm-raw-union-v1",
            bindings,
            {
                "curves": curves,
                "headroom_by_topic": headroom_by_topic,
                "headroom_docids": headroom_docids,
            },
        ),
        "prefusion.json": _bound_artifact(
            "facet-local-minilm-prefusion-v1", bindings, prefusion
        ),
        "facet_retention.json": _bound_artifact(
            "facet-local-minilm-facet-retention-v1",
            bindings,
            {"arms": retention},
        ),
        "systems.json": _bound_artifact(
            "facet-local-minilm-systems-v1",
            bindings,
            {"systems": system_results, "topic_ids": list(PILOT_TOPIC_IDS)},
        ),
        "gains_losses.json": _bound_artifact(
            "facet-local-minilm-gains-losses-v1",
            bindings,
            {"systems": gains_losses},
        ),
        "review_metrics.json": _bound_artifact(
            "facet-local-minilm-review-metrics-v1", bindings, review_metrics
        ),
        "representatives.json": _bound_artifact(
            "facet-local-minilm-representatives-v1", bindings, representatives
        ),
        "decision.json": _bound_artifact(
            "facet-local-minilm-decision-v1",
            bindings,
            {"decision": decision, "evidence": evidence},
        ),
    }
    if tuple(artifacts) != EVALUATION_ARTIFACT_NAMES:
        raise AssertionError("Task 6 artifact set differs from its fixed contract")
    return artifacts


def _publish_evaluation_artifacts(
    output: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    if set(artifacts) != set(EVALUATION_ARTIFACT_NAMES):
        raise ValueError("evaluation artifact set differs from the fixed contract")
    for name in EVALUATION_ARTIFACT_NAMES:
        try:
            with (output / name).open("x", encoding="utf-8") as stream:
                stream.write(
                    _canonical_json_bytes(artifacts[name], newline=True).decode(
                        "utf-8"
                    )
                )
        except FileExistsError as exc:
            raise FileExistsError(
                f"create-only evaluation output already exists: {name}"
            ) from exc


def evaluate(
    freeze: Path,
    *,
    review_freeze: Path | None = None,
    prior_evaluation: Path | None = None,
    prior_evaluation_manifest: Path | None = None,
    qrels_manifest: Path,
    qrels_approval: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Cross the qrels firewall once, only after every frozen input verifies."""

    output_path = Path(output) if output is not None else None
    receipt_path = output_path / "qrels_access_receipt.json" if output_path else None
    if receipt_path is not None and (
        receipt_path.exists() or receipt_path.is_symlink()
    ):
        raise FileExistsError("repeat qrels access is refused")
    if output_path is not None:
        collision = next(
            (
                name
                for name in EVALUATION_ARTIFACT_NAMES
                if (output_path / name).exists() or (output_path / name).is_symlink()
            ),
            None,
        )
        if collision is not None:
            raise FileExistsError(
                f"create-only evaluation output already exists: {collision}"
            )
    if (
        review_freeze is None
        or prior_evaluation is None
        or prior_evaluation_manifest is None
        or output_path is None
    ):
        raise ValueError("ranking and blinded-review freeze is incomplete")

    ranking = verify_ranking_freeze(Path(freeze))
    review = verify_blinded_review_freeze(Path(review_freeze), Path(freeze))
    ranking_hash = _validated_sha256(
        ranking.get("freeze_sha256"), "ranking freeze hash"
    )
    review_hash = _validated_sha256(
        review.get("review_freeze_sha256"), "review freeze hash"
    )
    frozen_inputs = load_frozen_inputs(
        Path(freeze),
        Path(review_freeze),
        Path(prior_evaluation),
        prior_evaluation_manifest=Path(prior_evaluation_manifest),
        ranking_freeze_sha256=ranking_hash,
        review_freeze_sha256=review_hash,
    )
    validate_frozen_input_topics(frozen_inputs)
    authorization = validate_qrels_authorization(
        Path(qrels_manifest),
        Path(qrels_approval),
        ranking_freeze_sha256=ranking_hash,
        review_freeze_sha256=review_hash,
    )
    consumption_identity = qrels_consumption_identity(Path(qrels_approval))
    consumption_path = _registry_path_for_identity(consumption_identity)
    if consumption_path.exists() or consumption_path.is_symlink():
        raise FileExistsError("qrels approval was already consumed")
    if any(
        consumption_identity.get(field) != authorization.get(auth_field)
        for field, auth_field in (
            ("qrels_approval_sha256", "approval_sha256"),
            ("qrels_manifest_sha256", "manifest_sha256"),
            ("qrels_projection_sha256", "projection_sha256"),
        )
    ) or (
        consumption_identity.get("ranking_freeze_sha256") != ranking_hash
        or consumption_identity.get("review_freeze_sha256") != review_hash
    ):
        raise ValueError("qrels consumption identity changed during authorization")

    projection_path = Path(authorization["projection_path"])
    consumption = _create_qrels_consumption_registry(
        consumption_path,
        _qrels_consumption_payload(
            identity=consumption_identity,
            approval_sha256=str(authorization["approval_sha256"]),
            qrels_manifest_sha256=str(authorization["manifest_sha256"]),
            qrels_projection_sha256=str(authorization["projection_sha256"]),
            ranking_freeze_sha256=ranking_hash,
            review_freeze_sha256=review_hash,
            output=output_path,
            output_receipt=receipt_path,
        ),
    )
    output_path.mkdir(parents=True, exist_ok=True)
    receipt = _create_qrels_access_receipt(
        receipt_path,
        authorization=authorization,
        ranking_freeze_sha256=ranking_hash,
        review_freeze_sha256=review_hash,
    )
    qrels = read_qrels(
        projection_path,
        expected_sha256=authorization["projection_sha256"],
    )
    bindings = {
        "ranking_freeze_sha256": ranking_hash,
        "review_freeze_sha256": review_hash,
        "prior_evaluation_manifest_sha256": _validated_sha256(
            frozen_inputs.get("prior_evaluation_manifest_sha256"),
            "prior evaluation manifest hash",
        ),
        "prior_evaluation_sha256": _validated_sha256(
            frozen_inputs.get("prior_evaluation_sha256"),
            "prior evaluation hash",
        ),
        "qrels_manifest_sha256": authorization["manifest_sha256"],
        "qrels_projection_sha256": authorization["projection_sha256"],
    }
    artifacts = build_evaluation_artifacts(
        frozen_inputs,
        qrels,
        bindings=bindings,
    )
    _publish_evaluation_artifacts(output_path, artifacts)
    return {
        "qrels": qrels,
        "qrels_access_receipt": receipt,
        "qrels_consumption_registry": consumption,
        "artifacts": artifacts,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen facet-local MiniLM pilot"
    )
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--review-freeze", type=Path, required=True)
    parser.add_argument("--prior-evaluation", type=Path, required=True)
    parser.add_argument("--prior-evaluation-manifest", type=Path, required=True)
    parser.add_argument("--qrels-manifest", type=Path, required=True)
    parser.add_argument("--qrels-approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_adoption_argument_parser() -> argparse.ArgumentParser:
    """Build the qrels-free CLI for adopting an already-completed evaluation."""

    parser = argparse.ArgumentParser(
        description="Adopt an existing qrels evaluation into the stable guard"
    )
    parser.add_argument("--qrels-approval", type=Path, required=True)
    parser.add_argument("--qrels-access-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    return parser


def adoption_main(argv: Sequence[str] | None = None) -> int:
    args = build_adoption_argument_parser().parse_args(argv)
    adopt_qrels_consumption_registry(
        args.qrels_approval,
        qrels_access_receipt=args.qrels_access_receipt,
        evaluation_output=args.evaluation_output,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    evaluate(
        args.freeze,
        review_freeze=args.review_freeze,
        prior_evaluation=args.prior_evaluation,
        prior_evaluation_manifest=args.prior_evaluation_manifest,
        qrels_manifest=args.qrels_manifest,
        qrels_approval=args.qrels_approval,
        output=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())

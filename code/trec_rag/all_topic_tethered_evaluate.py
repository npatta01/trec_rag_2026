"""Mechanically evaluate the sealed 22-topic tethered-facet rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .all_topic_facet_contract import ALL_TOPIC_IDS, EXPERIMENT_ID
from .all_topic_tethered_rank import (
    ARMS,
    CANONICAL_RANKING_ROOT_SHA256,
    PLANNING_ROOT_SHA256,
    RETRIEVAL_ROOT_SHA256,
    SCORE_PLAN_ROOT_SHA256,
    SCORING_ROOT_SHA256,
    verify_rankings,
)


SCHEMA_VERSION = "all-topic-tethered-evaluation-v2"
SEAL_SCHEMA_VERSION = "all-topic-tethered-evaluation-seal-v2"
DEPTHS = (100, 250, 500, 1000, 1500)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_SAMPLES = 100_000
PINNED_QRELS_SHA256 = "42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37"
PINNED_QRELS_SUFFIX = (
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)
PINNED_RANKINGS_PATH = "outputs/all_topic_tethered_facet_validation_v1/rankings_v2"
PINNED_UNION_PATH = "outputs/all_topic_tethered_facet_validation_v1/retrieval/accepted_union.jsonl"
PINNED_UPSTREAM_ROOTS = {
    "planning_root_sha256": PLANNING_ROOT_SHA256,
    "retrieval_root_sha256": RETRIEVAL_ROOT_SHA256,
    "score_plan_root_sha256": SCORE_PLAN_ROOT_SHA256,
    "scoring_root_sha256": SCORING_ROOT_SHA256,
}
FACET_RANK_BUCKETS = ((1, 50), (51, 100), (101, 150), (151, 200))
SELECTION_LADDER = (
    "RRF100-STATIC-DUAL",
    "RRF500-REINIT-DUAL",
    "RRF100-REINIT-DUAL",
    "RRF100-REINIT-DUAL-NR",
    "RRF100-STATIC-DUAL-NR",
)
CANONICAL_EVALUATION_ROOT_SHA256 = "1634e2d993d79d46b969a6bcdc5207a7c06bc881485fc091ae3b7904bc0bc72b"


def _compact(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _pretty(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _binding(content: bytes) -> dict[str, object]:
    return {"bytes": len(content), "sha256": _sha256(content)}


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _gain(grade: int) -> int:
    return 2**grade - 1 if grade >= 2 else 0


def _ndcg(ranking: Sequence[str], qrels: Mapping[str, int], depth: int) -> float:
    gains = [_gain(qrels.get(document, 0)) for document in ranking[:depth]]
    dcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_gains = sorted((_gain(grade) for grade in qrels.values()), reverse=True)[:depth]
    ideal = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    return dcg / ideal if ideal else 0.0


def _recall_auc(ranking: Sequence[str], relevant: set[str]) -> float:
    if not ranking or not relevant:
        return 0.0
    found = 0
    area = 0.0
    for document in ranking:
        found += document in relevant
        area += found / len(relevant)
    return area / len(ranking)


def paired_bootstrap(
    deltas: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, object]:
    """Return the deterministic percentile interval for paired topic means."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or samples <= 0:
        raise ValueError("paired bootstrap requires finite topic deltas and positive samples")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "seed": seed,
        "samples": samples,
        "mean_delta": float(values.mean()),
        "ci_95": [float(low), float(high)],
    }


def paired_sign_flip(deltas: Sequence[float]) -> dict[str, object]:
    """Enumerate the exact one-sided paired sign-flip distribution."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or len(values) > 22 or not np.isfinite(values).all():
        raise ValueError("exact sign flip requires one through 22 finite paired deltas")
    observed = float(values.mean())
    enumeration_count = 1 << len(values)
    extreme = 0
    # Chunked vectorization keeps exact enumeration practical without allocating
    # a 2^22 by 22 matrix.
    bit_positions = np.arange(len(values), dtype=np.uint64)
    for start in range(0, enumeration_count, 1 << 16):
        indices = np.arange(start, min(start + (1 << 16), enumeration_count), dtype=np.uint64)
        signs = (((indices[:, None] >> bit_positions) & 1) * 2 - 1).astype(np.int8)
        permuted = signs @ values / len(values)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return {
        "alternative": "greater",
        "observed_mean_delta": observed,
        "enumerations": enumeration_count,
        "extreme_enumerations": extreme,
        "p_value": extreme / enumeration_count,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise correction."""

    ordered = sorted((float(value), str(name)) for name, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (value, name) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be in [0, 1]")
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _normalize_inputs(
    rankings: Mapping[str, object], qrels: Mapping[str, object], provenance: Mapping[str, object]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, int]], dict[str, dict[str, list[dict[str, object]]]]]:
    exact_topics = set(ALL_TOPIC_IDS)
    if any(set(value) != exact_topics for value in (rankings, qrels, provenance)):
        raise ValueError("rankings, qrels, and provenance must contain the exact 22 topics")
    parsed_rankings: dict[str, dict[str, list[str]]] = {}
    parsed_qrels: dict[str, dict[str, int]] = {}
    parsed_provenance: dict[str, dict[str, list[dict[str, object]]]] = {}
    for topic in ALL_TOPIC_IDS:
        raw_arms = rankings[topic]
        if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARMS):
            raise ValueError(f"topic {topic} must contain the exact frozen arms")
        arms = {arm: [str(row.get("document_id")) if isinstance(row, Mapping) else str(row) for row in raw_arms[arm]] for arm in ARMS}
        population = set(arms["RRF"])
        if not population or any(len(order) != len(population) or set(order) != population for order in arms.values()):
            raise ValueError(f"topic {topic} arms are not complete equal-population permutations")
        raw_qrels, raw_provenance = qrels[topic], provenance[topic]
        if not isinstance(raw_qrels, Mapping) or not isinstance(raw_provenance, Mapping):
            raise ValueError("qrels and provenance topics must be mappings")
        grades: dict[str, int] = {}
        for document, grade in raw_qrels.items():
            if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4:
                raise ValueError("qrels grades must be integer values from zero through four")
            grades[str(document)] = grade
        entries: dict[str, list[dict[str, object]]] = {}
        if set(map(str, raw_provenance)) != population:
            raise ValueError(f"topic {topic} provenance differs from the ranked union")
        for document, raw_entries in raw_provenance.items():
            if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)) or not raw_entries:
                raise ValueError("stream provenance must be a nonempty sequence")
            entries[str(document)] = [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
            if len(entries[str(document)]) != len(raw_entries):
                raise ValueError("stream provenance entry is invalid")
        parsed_rankings[topic], parsed_qrels[topic], parsed_provenance[topic] = arms, grades, entries
    return parsed_rankings, parsed_qrels, parsed_provenance


def _topic_metrics(
    ranking: Sequence[str], qrels: Mapping[str, int], provenance: Mapping[str, Sequence[Mapping[str, object]]], depths: Sequence[int]
) -> dict[str, object]:
    relevant = {document for document, grade in qrels.items() if grade >= 2}
    total_gain = sum(_gain(qrels[document]) for document in relevant)
    facet_only = {
        document for document, entries in provenance.items()
        if entries and all(str(entry.get("stream_id")) != "original" for entry in entries) and document in relevant
    }
    result: dict[str, object] = {
        "known_relevant_total": len(relevant),
        "known_relevant_graded_gain_total": total_gain,
        "union_size": len(ranking),
        "normalized_recall_auc": _recall_auc(ranking, relevant),
        "facet_only_known_relevant_total": len(facet_only),
    }
    for depth in tuple(dict.fromkeys((*map(int, depths), 250, 500, 1000))):
        prefix = ranking[:depth]
        known = set(prefix) & relevant
        retrieved_gain = sum(_gain(qrels[document]) for document in known)
        judged = sum(document in qrels for document in prefix)
        result.update({
            f"known_relevant_count@{depth}": len(known),
            f"binary_recall@{depth}": _ratio(len(known), len(relevant)),
            f"graded_gain@{depth}": retrieved_gain,
            f"graded_recall@{depth}": _ratio(retrieved_gain, total_gain),
            f"ndcg@{depth}": _ndcg(ranking, qrels, depth),
            f"precision@{depth}": _ratio(len(known), len(prefix)),
            f"candidate_count@{depth}": len(prefix),
            f"judged_count@{depth}": judged,
            f"judged_rate@{depth}": _ratio(judged, len(prefix)),
            f"facet_only_known_relevant_retained@{depth}": len(set(prefix) & facet_only),
            f"facet_only_known_relevant_retention@{depth}": _ratio(len(set(prefix) & facet_only), len(facet_only)),
        })
    full_known = set(ranking) & relevant
    full_gain = sum(_gain(qrels[document]) for document in full_known)
    result.update({
        "known_relevant_count_full": len(full_known),
        "binary_recall_full": _ratio(len(full_known), len(relevant)),
        "graded_gain_full": full_gain,
        "graded_recall_full": _ratio(full_gain, total_gain),
        "ndcg_full": _ndcg(ranking, qrels, len(ranking)),
        "precision_full": _ratio(len(full_known), len(ranking)),
        "candidate_count_full": len(ranking),
        "judged_count_full": sum(document in qrels for document in ranking),
        "judged_rate_full": _ratio(sum(document in qrels for document in ranking), len(ranking)),
        "facet_only_known_relevant_retained_full": len(full_known & facet_only),
        "facet_only_known_relevant_retention_full": _ratio(len(full_known & facet_only), len(facet_only)),
        "full_union_recall_ceiling": _ratio(len(full_known), len(relevant)),
    })
    return result


def _aggregate(per_topic: Mapping[str, Mapping[str, object]], depths: Sequence[int]) -> dict[str, dict[str, object]]:
    pooled: dict[str, object] = {
        "known_relevant_total": sum(int(per_topic[t]["known_relevant_total"]) for t in ALL_TOPIC_IDS),
        "known_relevant_graded_gain_total": sum(int(per_topic[t]["known_relevant_graded_gain_total"]) for t in ALL_TOPIC_IDS),
        "union_size": sum(int(per_topic[t]["union_size"]) for t in ALL_TOPIC_IDS),
        "facet_only_known_relevant_total": sum(int(per_topic[t]["facet_only_known_relevant_total"]) for t in ALL_TOPIC_IDS),
    }
    macro: dict[str, object] = {}
    suffixes = [*map(str, depths), "full"]
    for suffix in suffixes:
        count_key = f"known_relevant_count@{suffix}" if suffix != "full" else "known_relevant_count_full"
        gain_key = f"graded_gain@{suffix}" if suffix != "full" else "graded_gain_full"
        candidate_key = f"candidate_count@{suffix}" if suffix != "full" else "candidate_count_full"
        judged_key = f"judged_count@{suffix}" if suffix != "full" else "judged_count_full"
        facet_key = f"facet_only_known_relevant_retained@{suffix}" if suffix != "full" else "facet_only_known_relevant_retained_full"
        metric_suffix = f"@{suffix}" if suffix != "full" else "_full"
        counts = sum(int(per_topic[t][count_key]) for t in ALL_TOPIC_IDS)
        gains = sum(int(per_topic[t][gain_key]) for t in ALL_TOPIC_IDS)
        candidates = sum(int(per_topic[t][candidate_key]) for t in ALL_TOPIC_IDS)
        judged = sum(int(per_topic[t][judged_key]) for t in ALL_TOPIC_IDS)
        facet = sum(int(per_topic[t][facet_key]) for t in ALL_TOPIC_IDS)
        pooled[count_key] = counts
        pooled[gain_key] = gains
        pooled[f"binary_recall{metric_suffix}"] = _ratio(counts, pooled["known_relevant_total"])
        pooled[f"graded_recall{metric_suffix}"] = _ratio(gains, pooled["known_relevant_graded_gain_total"])
        pooled[f"precision{metric_suffix}"] = _ratio(counts, candidates)
        pooled[f"judged_rate{metric_suffix}"] = _ratio(judged, candidates)
        pooled[f"facet_only_known_relevant_retention{metric_suffix}"] = _ratio(facet, pooled["facet_only_known_relevant_total"])
        for metric in ("binary_recall", "graded_recall", "ndcg", "precision", "judged_rate", "facet_only_known_relevant_retention"):
            key = f"{metric}{metric_suffix}"
            macro[key] = math.fsum(float(per_topic[t][key]) for t in ALL_TOPIC_IDS) / len(ALL_TOPIC_IDS)
    macro["normalized_recall_auc"] = math.fsum(float(per_topic[t]["normalized_recall_auc"]) for t in ALL_TOPIC_IDS) / len(ALL_TOPIC_IDS)
    macro["full_union_recall_ceiling"] = math.fsum(float(per_topic[t]["full_union_recall_ceiling"]) for t in ALL_TOPIC_IDS) / len(ALL_TOPIC_IDS)
    pooled["normalized_recall_auc"] = macro["normalized_recall_auc"]
    pooled["full_union_recall_ceiling"] = _ratio(
        sum(int(per_topic[t]["known_relevant_count_full"]) for t in ALL_TOPIC_IDS),
        pooled["known_relevant_total"],
    )
    # nDCG has no pooled denominator; repeat the explicitly macro topic mean in
    # the pooled presentation table and label its aggregation method.
    for suffix in suffixes:
        key = f"ndcg@{suffix}" if suffix != "full" else "ndcg_full"
        pooled[key] = macro[key]
    pooled["ndcg_aggregation"] = "macro_topic_mean"
    return {"pooled": pooled, "macro": macro}


def _arm_comparison(arm: str, per_topic: Mapping[str, Mapping[str, object]], baseline: Mapping[str, Mapping[str, object]], rankings: Mapping[str, Mapping[str, Sequence[str]]], depths: Sequence[int]) -> dict[str, object]:
    result: dict[str, object] = {"per_topic_deltas": {}}
    for topic in ALL_TOPIC_IDS:
        result["per_topic_deltas"][topic] = {
            key: float(value) - float(baseline[topic][key])
            for key, value in per_topic[topic].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key in baseline[topic]
        }
    for depth in depths:
        count_deltas = {topic: int(per_topic[topic][f"known_relevant_count@{depth}"]) - int(baseline[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS}
        recall_deltas = {topic: float(per_topic[topic][f"binary_recall@{depth}"]) - float(baseline[topic][f"binary_recall@{depth}"]) for topic in ALL_TOPIC_IDS}
        result[f"loss_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] < 0]
        result[f"win_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] > 0]
        result[f"tie_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] == 0]
        result[f"macro_delta_at_{depth}"] = math.fsum(recall_deltas.values()) / len(ALL_TOPIC_IDS)
        baseline_total = sum(int(baseline[topic]["known_relevant_total"]) for topic in ALL_TOPIC_IDS)
        result[f"pooled_delta_at_{depth}"] = _ratio(sum(int(per_topic[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS), baseline_total) - _ratio(sum(int(baseline[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS), baseline_total)
        worst_topic = min(ALL_TOPIC_IDS, key=lambda topic: (recall_deltas[topic], count_deltas[topic], ALL_TOPIC_IDS.index(topic)))
        result[f"worst_regression_at_{depth}"] = {
            "topic_id": worst_topic,
            "known_relevant_count_delta": count_deltas[worst_topic],
            "binary_recall_delta": recall_deltas[worst_topic],
        }
        result[f"wins_at_{depth}"] = len(result[f"win_topic_ids_at_{depth}"])
        result[f"ties_at_{depth}"] = len(result[f"tie_topic_ids_at_{depth}"])
        result[f"losses_at_{depth}"] = len(result[f"loss_topic_ids_at_{depth}"])
    result["loss_topic_ids"] = result["loss_topic_ids_at_1000"]
    judged_delta = math.fsum(float(per_topic[t]["judged_rate@1000"]) - float(baseline[t]["judged_rate@1000"]) for t in ALL_TOPIC_IDS) / len(ALL_TOPIC_IDS)
    result["macro_judged_rate_delta_at_1000"] = judged_delta
    protected = 500 if arm.startswith("RRF500") else 100
    result["protected_prefix_depth"] = protected
    result["protected_prefix_identical"] = all(list(rankings[t][arm])[:protected] == list(rankings[t]["RRF"])[:protected] for t in ALL_TOPIC_IDS)
    return result


def _statistics(arms: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    raw_p: dict[str, float] = {}
    for arm in SELECTION_LADDER:
        per_topic = arms[arm]["per_topic"]
        baseline = arms["RRF"]["per_topic"]
        deltas = [float(per_topic[t]["binary_recall@1000"]) - float(baseline[t]["binary_recall@1000"]) for t in ALL_TOPIC_IDS]
        bootstrap = paired_bootstrap(deltas)
        sign_flip = paired_sign_flip(deltas)
        result[arm] = {"topic_deltas": dict(zip(ALL_TOPIC_IDS, deltas, strict=True)), "bootstrap": bootstrap, "sign_flip": sign_flip}
        raw_p[arm] = float(sign_flip["p_value"])
    adjusted = holm_adjust(raw_p)
    for arm in SELECTION_LADDER:
        result[arm]["holm_adjusted_p"] = adjusted[arm]
        result[arm]["bootstrap_ci_95"] = result[arm]["bootstrap"]["ci_95"]
    return result


def apply_promotion_rules(metrics: Mapping[str, object], statistics: Mapping[str, object]) -> dict[str, object]:
    """Apply every preregistered guard to one non-baseline arm."""

    ci = statistics.get("bootstrap_ci_95")
    significant = (
        isinstance(ci, Sequence) and not isinstance(ci, (str, bytes)) and len(ci) == 2 and float(ci[0]) > 0
    ) or float(statistics.get("holm_adjusted_p", 1.0)) < 0.05
    checks = (
        ("zero_losses_at_250", not metrics.get("loss_topic_ids_at_250")),
        ("zero_losses_at_500", not metrics.get("loss_topic_ids_at_500")),
        ("zero_losses_at_1000", not metrics.get("loss_topic_ids")),
        ("positive_pooled_recall_at_1000", float(metrics.get("pooled_delta_at_1000", 0.0)) > 0),
        ("positive_macro_recall_at_1000", float(metrics.get("macro_delta_at_1000", 0.0)) > 0),
        ("at_least_eight_wins_at_1000", int(metrics.get("wins_at_1000", 0)) >= 8),
        ("corrected_significance", significant),
        ("protected_prefix_identity", metrics.get("protected_prefix_identical") is True),
    )
    failed = [name for name, passed in checks if not passed]
    return {"promoted": not failed, "failed_rules": failed, "rules": {name: passed for name, passed in checks}}


def _select(arms: Mapping[str, Mapping[str, object]], statistics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    decisions = {arm: apply_promotion_rules(arms[arm], statistics[arm]) for arm in SELECTION_LADDER}
    selected = next((arm for arm in SELECTION_LADDER if decisions[arm]["promoted"]), "RRF")
    return {"selected_arm": selected, "promoted": selected != "RRF", "selection_ladder": list(SELECTION_LADDER), "arm_decisions": decisions}


def evaluate_all_topics(
    rankings: Mapping[str, object], qrels: Mapping[str, object], provenance: Mapping[str, object], depths: Sequence[int] = DEPTHS
) -> dict[str, object]:
    """Evaluate exact all-topic complete permutations against known relevance."""

    parsed_rankings, parsed_qrels, parsed_provenance = _normalize_inputs(rankings, qrels, provenance)
    ordered_depths = tuple(dict.fromkeys(map(int, depths)))
    if not ordered_depths or any(depth <= 0 for depth in ordered_depths):
        raise ValueError("depths must be positive")
    metric_depths = tuple(dict.fromkeys((*ordered_depths, 250, 500, 1000)))
    arms: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        per_topic = {topic: _topic_metrics(parsed_rankings[topic][arm], parsed_qrels[topic], parsed_provenance[topic], metric_depths) for topic in ALL_TOPIC_IDS}
        arms[arm] = {"per_topic": per_topic, "aggregate": _aggregate(per_topic, metric_depths)}
    baseline = arms["RRF"]["per_topic"]
    for arm in ARMS:
        arms[arm].update(_arm_comparison(arm, arms[arm]["per_topic"], baseline, parsed_rankings, metric_depths))
    facet_yield: dict[str, dict[str, object]] = {}
    for topic in ALL_TOPIC_IDS:
        relevant = {document for document, grade in parsed_qrels[topic].items() if grade >= 2}
        buckets: dict[str, object] = {}
        for low, high in FACET_RANK_BUCKETS:
            documents = {
                document for document, entries in parsed_provenance[topic].items()
                if any(str(entry.get("stream_id")) != "original" and isinstance(entry.get("stream_rank"), int) and low <= int(entry["stream_rank"]) <= high for entry in entries)
            }
            buckets[f"{low}-{high}"] = {"unique_candidate_count": len(documents), "known_relevant_count": len(documents & relevant), "known_relevant_yield": _ratio(len(documents & relevant), len(documents))}
        facet_yield[topic] = buckets
    statistics = _statistics(arms)
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(ALL_TOPIC_IDS),
        "depths": [*metric_depths, "full"],
        "relevance_threshold": 2,
        "arms": arms,
        "statistics": statistics,
        "decision": _select(arms, statistics),
        "facet_rank_bucket_yield": facet_yield,
        "judged_rate_interpretation": {"promotion_threshold": None, "diagnostic_only": True, "unjudged_treated_as_nonrelevant": True},
    }


def _load_rankings(content: bytes) -> dict[str, dict[str, list[str]]]:
    grouped = {topic: {arm: [] for arm in ARMS} for topic in ALL_TOPIC_IDS}
    for number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rankings:{number} is invalid JSON") from exc
        topic, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic not in grouped or arm not in ARMS or row.get("rank") != len(grouped[topic][arm]) + 1:
            raise ValueError("ranking row scope or order differs")
        grouped[topic][arm].append(str(row.get("document_id")))
    return grouped


def _load_provenance(path: Path, expected: Mapping[str, object]) -> dict[str, dict[str, list[dict[str, object]]]]:
    result = {topic: {} for topic in ALL_TOPIC_IDS}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for number, line in enumerate(source, 1):
            digest.update(line); size += len(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"accepted union:{number} is invalid JSON") from exc
            topic, document = str(row.get("topic_id")), str(row.get("document_id"))
            raw = row.get("stream_provenance")
            if topic not in result or document in result[topic] or not isinstance(raw, list) or not raw:
                raise ValueError("accepted-union provenance identity differs")
            result[topic][document] = [dict(entry) for entry in raw if isinstance(entry, Mapping)]
            if len(result[topic][document]) != len(raw):
                raise ValueError("accepted-union provenance entry differs")
    if size != expected.get("bytes") or digest.hexdigest() != expected.get("sha256"):
        raise ValueError("accepted-union provenance differs from the frozen ranking binding")
    return result


def _load_qrels(content: bytes) -> dict[str, dict[str, int]]:
    result = {topic: {} for topic in ALL_TOPIC_IDS}
    observed: list[str] = []
    for number, line in enumerate(content.splitlines(), 1):
        fields = line.decode("utf-8").split()
        if len(fields) != 4:
            raise ValueError(f"qrels:{number} is not four-column TREC qrels")
        topic, _iteration, document, raw_grade = fields
        try:
            grade = int(raw_grade)
        except ValueError as exc:
            raise ValueError(f"qrels:{number} grade is not an integer") from exc
        if topic not in result or isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4 or document in result[topic]:
            raise ValueError("qrels scope, identity, or grade differs")
        if not observed or observed[-1] != topic:
            observed.append(topic)
        result[topic][document] = grade
    if observed != list(ALL_TOPIC_IDS) or any(not result[topic] for topic in ALL_TOPIC_IDS):
        raise ValueError("qrels must contain the exact contiguous 22-topic projection")
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_evaluation(result: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    metrics = {
        key: value for key, value in result.items()
        if key not in {"statistics", "decision", "facet_rank_bucket_yield"}
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "facet_rank_bucket_yield": result["facet_rank_bucket_yield"],
        "statistics": result["statistics"],
        "decision": result["decision"],
    }
    return metrics, diagnostics


def _summary(result: Mapping[str, object], artifacts: Mapping[str, bytes]) -> dict[str, object]:
    decision = result["decision"]
    assert isinstance(decision, Mapping)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "topic_ids": list(ALL_TOPIC_IDS),
        "arms": list(ARMS),
        "depths": result["depths"],
        "selected_arm": decision["selected_arm"],
        "qrels_opened": True,
        "network_calls": 0,
        "retrieval_calls": 0,
        "model_loads": 0,
        "inference_calls": 0,
        "artifacts": {name: _binding(content) for name, content in artifacts.items()},
    }


def _write_evaluation(result: Mapping[str, object], input_bindings: Mapping[str, object], output: Path) -> dict[str, object]:
    """Create one deterministic sealed evaluation from already-recomputed data."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    metrics, diagnostics = _split_evaluation(result)
    artifacts = {
        "metrics.json": _pretty(metrics),
        "diagnostics.json": _pretty(diagnostics),
        "input_bindings.json": _pretty(input_bindings),
    }
    summary = _summary(result, artifacts)
    artifacts["summary.json"] = _pretty(summary)
    seal_material = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed_evaluation",
        "files": {name: _binding(content) for name, content in artifacts.items()},
    }
    artifacts["SEALED.json"] = _pretty({
        **seal_material, "root_sha256": _sha256(_compact(seal_material))
    })
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        with (output / name).open("xb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    return summary


def _recompute_from_paths(
    rankings: Path,
    retrieval: Path,
    qrels: Path,
    *,
    qrels_reader: Callable[[Path], bytes] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate canonical source bytes, then independently recompute all results."""

    ranking_verification = verify_rankings(rankings)
    if ranking_verification.get("root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("ranking seal is not the canonical approved v2 root")
    seal_content = (rankings / "SEALED.json").read_bytes()
    seal = json.loads(seal_content)
    ranking_content = (rankings / "rankings.jsonl").read_bytes()
    if seal.get("files", {}).get("rankings.jsonl") != _binding(ranking_content):
        raise ValueError("ranking buffer differs from the verified ranking seal")
    ranking_orders = _load_rankings(ranking_content)
    frozen_bindings_content = (rankings / "input_bindings.json").read_bytes()
    if seal.get("files", {}).get("input_bindings.json") != _binding(frozen_bindings_content):
        raise ValueError("ranking input binding differs from the verified ranking seal")
    frozen_bindings = json.loads(frozen_bindings_content)
    if {key: frozen_bindings.get(key) for key in PINNED_UPSTREAM_ROOTS} != PINNED_UPSTREAM_ROOTS:
        raise ValueError("ranking upstream root bindings differ from the preregistered v2 inputs")
    accepted_binding = frozen_bindings.get("accepted_union")
    if not isinstance(accepted_binding, Mapping):
        raise ValueError("ranking accepted-union binding is missing")
    provenance = _load_provenance(retrieval / "accepted_union.jsonl", accepted_binding)
    if (rankings / "SEALED.json").read_bytes() != seal_content:
        raise ValueError("ranking seal changed during the authenticated snapshot")
    read = qrels_reader or (lambda path: path.read_bytes())
    qrels_content = read(qrels)
    if _sha256(qrels_content) != PINNED_QRELS_SHA256:
        raise ValueError("qrels SHA-256 differs from the pinned all-topic projection")
    result = evaluate_all_topics(ranking_orders, _load_qrels(qrels_content), provenance)
    bindings: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "ranking_verified_before_qrels_open": True,
        "ranking_root_sha256": CANONICAL_RANKING_ROOT_SHA256,
        "ranking_path": PINNED_RANKINGS_PATH,
        "rankings": {**_binding(ranking_content), "path": f"{PINNED_RANKINGS_PATH}/rankings.jsonl"},
        "ranking_seal": {**_binding(seal_content), "path": f"{PINNED_RANKINGS_PATH}/SEALED.json"},
        "accepted_union": {**dict(accepted_binding), "path": PINNED_UNION_PATH},
        "qrels": {**_binding(qrels_content), "path": PINNED_QRELS_SUFFIX},
        "upstream_roots": dict(PINNED_UPSTREAM_ROOTS),
    }
    return result, bindings


def _recompute_canonical() -> tuple[dict[str, object], dict[str, object]]:
    root = _repo_root()
    return _recompute_from_paths(
        root / PINNED_RANKINGS_PATH,
        root / Path(PINNED_UNION_PATH).parent,
        root / PINNED_QRELS_SUFFIX,
    )


def evaluate_frozen(
    rankings: Path,
    retrieval: Path,
    qrels: Path,
    output: Path,
    *,
    qrels_reader: Callable[[Path], bytes] | None = None,
) -> dict[str, object]:
    """Verify and snapshot all qrels-blind inputs before opening qrels once."""

    rankings, retrieval, qrels, output = map(Path, (rankings, retrieval, qrels, output))
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    root = _repo_root()
    expected = (
        (root / PINNED_RANKINGS_PATH).resolve(),
        (root / Path(PINNED_UNION_PATH).parent).resolve(),
        (root / PINNED_QRELS_SUFFIX).resolve(),
    )
    actual = (rankings.resolve(), retrieval.resolve(), qrels.resolve())
    if actual != expected:
        raise ValueError("evaluation inputs are not the exact canonical v2 paths")
    result, bindings = _recompute_from_paths(
        rankings, retrieval, qrels, qrels_reader=qrels_reader
    )
    return _write_evaluation(result, bindings, output)


def verify_evaluation(evaluation: Path) -> dict[str, object]:
    """Verify the pinned root and independently recompute from canonical sources."""

    evaluation = Path(evaluation)
    names = {"metrics.json", "diagnostics.json", "input_bindings.json", "summary.json", "SEALED.json"}
    if not evaluation.is_dir() or {path.name for path in evaluation.iterdir()} != names:
        raise ValueError("evaluation directory differs from the sealed contract")
    buffers = {name: (evaluation / name).read_bytes() for name in names}
    seal = json.loads(buffers["SEALED.json"])
    files = seal.get("files")
    material = {"schema_version": SEAL_SCHEMA_VERSION, "status": "sealed_evaluation", "files": files}
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION or seal.get("status") != "sealed_evaluation" or not isinstance(files, Mapping) or set(files) != names - {"SEALED.json"} or seal.get("root_sha256") != _sha256(_compact(material)):
        raise ValueError("evaluation seal contract differs")
    if CANONICAL_EVALUATION_ROOT_SHA256 is None or seal.get("root_sha256") != CANONICAL_EVALUATION_ROOT_SHA256:
        raise ValueError("canonical evaluation root differs or is not pinned")
    for name in names - {"SEALED.json"}:
        if files[name] != _binding(buffers[name]):
            raise ValueError(f"evaluation artifact differs: {name}")
    metrics, diagnostics, bindings, summary = (
        json.loads(buffers[name])
        for name in ("metrics.json", "diagnostics.json", "input_bindings.json", "summary.json")
    )
    recomputed, recomputed_bindings = _recompute_canonical()
    expected_metrics, expected_diagnostics = _split_evaluation(recomputed)
    if metrics != expected_metrics:
        raise ValueError("evaluation metrics differ from independently recomputed canonical metrics")
    if diagnostics != expected_diagnostics:
        raise ValueError("evaluation diagnostics differ from independently recomputed statistics or decision")
    if bindings != recomputed_bindings:
        raise ValueError("evaluation source binding differs from independently authenticated canonical binding")
    expected_summary = _summary(
        recomputed,
        {name: buffers[name] for name in ("metrics.json", "diagnostics.json", "input_bindings.json")},
    )
    if summary != expected_summary:
        raise ValueError("evaluation summary differs from independently recomputed canonical summary")
    return {**summary, "evaluation_root_sha256": seal["root_sha256"], "recomputed": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--rankings", type=Path, required=True)
    evaluate.add_argument("--retrieval", type=Path, required=True)
    evaluate.add_argument("--qrels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--evaluation", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        if not str(args.qrels).replace("\\", "/").endswith(PINNED_QRELS_SUFFIX):
            raise ValueError("qrels path is not the exact pinned all-topic projection")
        result = evaluate_frozen(args.rankings, args.retrieval, args.qrels, args.output)
    else:
        result = verify_evaluation(args.evaluation)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

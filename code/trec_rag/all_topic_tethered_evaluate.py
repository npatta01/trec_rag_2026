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
    verify_rankings,
)


SCHEMA_VERSION = "all-topic-tethered-evaluation-v1"
SEAL_SCHEMA_VERSION = "all-topic-tethered-evaluation-seal-v1"
DEPTHS = (100, 250, 500, 1000, 1500)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_SAMPLES = 100_000
PINNED_QRELS_SHA256 = "42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37"
PINNED_QRELS_SUFFIX = (
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)
FACET_RANK_BUCKETS = ((1, 50), (51, 100), (101, 150), (151, 200))
SELECTION_LADDER = (
    "RRF100-STATIC-DUAL",
    "RRF500-REINIT-DUAL",
    "RRF100-REINIT-DUAL",
    "RRF500-STATIC-DUAL",
    "DUAL",
)
JUDGED_RATE_MAX_DROP_AT_1000 = 0.02


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
        "graded_recall_full": _ratio(full_gain, total_gain),
        "ndcg_full": _ndcg(ranking, qrels, len(ranking)),
        "precision_full": _ratio(len(full_known), len(ranking)),
        "judged_rate_full": _ratio(sum(document in qrels for document in ranking), len(ranking)),
        "facet_only_known_relevant_retained_full": len(full_known & facet_only),
        "facet_only_known_relevant_retention_full": _ratio(len(full_known & facet_only), len(facet_only)),
        "full_union_recall_ceiling": _ratio(len(full_known), len(relevant)),
    })
    return result


def _arm_comparison(arm: str, per_topic: Mapping[str, Mapping[str, object]], baseline: Mapping[str, Mapping[str, object]], rankings: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for depth in (250, 500, 1000):
        count_deltas = {topic: int(per_topic[topic][f"known_relevant_count@{depth}"]) - int(baseline[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS}
        recall_deltas = {topic: float(per_topic[topic][f"binary_recall@{depth}"]) - float(baseline[topic][f"binary_recall@{depth}"]) for topic in ALL_TOPIC_IDS}
        result[f"loss_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] < 0]
        result[f"win_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] > 0]
        result[f"tie_topic_ids_at_{depth}"] = [topic for topic in ALL_TOPIC_IDS if count_deltas[topic] == 0]
        result[f"macro_delta_at_{depth}"] = math.fsum(recall_deltas.values()) / len(ALL_TOPIC_IDS)
        baseline_total = sum(int(baseline[topic]["known_relevant_total"]) for topic in ALL_TOPIC_IDS)
        result[f"pooled_delta_at_{depth}"] = _ratio(sum(int(per_topic[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS), baseline_total) - _ratio(sum(int(baseline[topic][f"known_relevant_count@{depth}"]) for topic in ALL_TOPIC_IDS), baseline_total)
        result[f"worst_regression_at_{depth}"] = min(recall_deltas.values())
    result["loss_topic_ids"] = result["loss_topic_ids_at_1000"]
    result["wins_at_1000"] = len(result["win_topic_ids_at_1000"])
    result["ties_at_1000"] = len(result["tie_topic_ids_at_1000"])
    result["losses_at_1000"] = len(result["loss_topic_ids_at_1000"])
    judged_delta = math.fsum(float(per_topic[t]["judged_rate@1000"]) - float(baseline[t]["judged_rate@1000"]) for t in ALL_TOPIC_IDS) / len(ALL_TOPIC_IDS)
    result["macro_judged_rate_delta_at_1000"] = judged_delta
    result["judged_rate_interpretable"] = judged_delta >= -JUDGED_RATE_MAX_DROP_AT_1000
    protected = 0 if arm == "DUAL" else 500 if arm.startswith("RRF500") else 100 if arm.startswith("RRF100") else 0
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
        ("interpretable_judged_rate", metrics.get("judged_rate_interpretable") is True),
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
    arms: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        per_topic = {topic: _topic_metrics(parsed_rankings[topic][arm], parsed_qrels[topic], parsed_provenance[topic], ordered_depths) for topic in ALL_TOPIC_IDS}
        arms[arm] = {"per_topic": per_topic}
    baseline = arms["RRF"]["per_topic"]
    for arm in ARMS:
        arms[arm].update(_arm_comparison(arm, arms[arm]["per_topic"], baseline, parsed_rankings))
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
        "depths": [*ordered_depths, "full"],
        "relevance_threshold": 2,
        "arms": arms,
        "statistics": statistics,
        "decision": _select(arms, statistics),
        "facet_rank_bucket_yield": facet_yield,
        "judged_rate_interpretation": {"maximum_allowed_macro_drop_at_1000": JUDGED_RATE_MAX_DROP_AT_1000, "unjudged_treated_as_nonrelevant": True},
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
    ranking_verification = verify_rankings(rankings)
    if ranking_verification.get("root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("ranking seal is not the canonical approved root")
    seal_content = (rankings / "SEALED.json").read_bytes()
    seal = json.loads(seal_content)
    ranking_content = (rankings / "rankings.jsonl").read_bytes()
    if seal.get("files", {}).get("rankings.jsonl") != _binding(ranking_content):
        raise ValueError("ranking buffer differs from the verified ranking seal")
    ranking_orders = _load_rankings(ranking_content)
    bindings_content = (rankings / "input_bindings.json").read_bytes()
    if seal.get("files", {}).get("input_bindings.json") != _binding(bindings_content):
        raise ValueError("ranking input binding differs from the verified ranking seal")
    frozen_bindings = json.loads(bindings_content)
    provenance = _load_provenance(retrieval / "accepted_union.jsonl", frozen_bindings["accepted_union"])
    if (rankings / "SEALED.json").read_bytes() != seal_content:
        raise ValueError("ranking seal changed during the authenticated snapshot")
    read = qrels_reader or (lambda path: path.read_bytes())
    qrels_content = read(qrels)
    if _sha256(qrels_content) != PINNED_QRELS_SHA256:
        raise ValueError("qrels SHA-256 differs from the pinned all-topic projection")
    metrics = evaluate_all_topics(ranking_orders, _load_qrels(qrels_content), provenance)
    statistics = metrics.pop("statistics")
    decision = metrics.pop("decision")
    facet_yield = metrics.pop("facet_rank_bucket_yield")
    diagnostics = {"schema_version": SCHEMA_VERSION, "facet_rank_bucket_yield": facet_yield, "statistics": statistics, "decision": decision}
    input_bindings = {
        "schema_version": SCHEMA_VERSION,
        "ranking_verified_before_qrels_open": True,
        "ranking_root_sha256": CANONICAL_RANKING_ROOT_SHA256,
        "rankings": {**_binding(ranking_content), "path": str((rankings / "rankings.jsonl").resolve())},
        "accepted_union": {**frozen_bindings["accepted_union"], "path": str((retrieval / "accepted_union.jsonl").resolve())},
        "qrels": {**_binding(qrels_content), "path": str(qrels.resolve())},
    }
    artifacts = {"metrics.json": _pretty(metrics), "diagnostics.json": _pretty(diagnostics), "input_bindings.json": _pretty(input_bindings)}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "topic_ids": list(ALL_TOPIC_IDS),
        "arms": list(ARMS),
        "depths": [*DEPTHS, "full"],
        "selected_arm": decision["selected_arm"],
        "qrels_opened": True,
        "network_calls": 0,
        "retrieval_calls": 0,
        "model_loads": 0,
        "inference_calls": 0,
        "artifacts": {name: _binding(content) for name, content in artifacts.items()},
    }
    artifacts["summary.json"] = _pretty(summary)
    seal_material = {"schema_version": SEAL_SCHEMA_VERSION, "status": "sealed_evaluation", "files": {name: _binding(content) for name, content in artifacts.items()}}
    artifacts["SEALED.json"] = _pretty({**seal_material, "root_sha256": _sha256(_compact(seal_material))})
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        with (output / name).open("xb") as sink:
            sink.write(content); sink.flush(); os.fsync(sink.fileno())
    return summary


def verify_evaluation(evaluation: Path) -> dict[str, object]:
    """Verify artifact hashes and recompute statistics and promotion exactly."""

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
    for name in names - {"SEALED.json"}:
        if files[name] != _binding(buffers[name]):
            raise ValueError(f"evaluation artifact differs: {name}")
    metrics, diagnostics, bindings, summary = (json.loads(buffers[name]) for name in ("metrics.json", "diagnostics.json", "input_bindings.json", "summary.json"))
    if metrics.get("schema_version") != SCHEMA_VERSION or metrics.get("topic_ids") != list(ALL_TOPIC_IDS) or metrics.get("relevance_threshold") != 2:
        raise ValueError("evaluation metric scope differs")
    arms = metrics.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS) or any(set(arms[arm]["per_topic"]) != set(ALL_TOPIC_IDS) for arm in ARMS):
        raise ValueError("evaluation per-topic scope differs")
    recomputed_statistics = _statistics(arms)
    recomputed_decision = _select(arms, recomputed_statistics)
    if diagnostics.get("statistics") != recomputed_statistics or diagnostics.get("decision") != recomputed_decision:
        raise ValueError("evaluation recomputation differs from saved statistics or promotion")
    if bindings.get("ranking_verified_before_qrels_open") is not True or bindings.get("ranking_root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("evaluation firewall binding differs")
    if summary.get("status") != "complete" or summary.get("selected_arm") != recomputed_decision["selected_arm"] or any(summary.get(name) != 0 for name in ("network_calls", "retrieval_calls", "model_loads", "inference_calls")):
        raise ValueError("evaluation summary differs")
    for name in ("metrics.json", "diagnostics.json", "input_bindings.json"):
        if summary.get("artifacts", {}).get(name) != _binding(buffers[name]):
            raise ValueError("evaluation summary artifact binding differs")
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

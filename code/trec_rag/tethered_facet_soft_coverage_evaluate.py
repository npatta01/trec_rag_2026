"""Evaluate sealed tethered-facet rankings without retrieval or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .tethered_facet_soft_coverage import ARMS, verify_soft_freeze


SCHEMA_VERSION = "tethered-facet-soft-coverage-evaluation-v1"
DEFAULT_DEPTHS = (100, 250, 500, 1000, 1500)
PINNED_QRELS_SHA256 = "03fc4bd18be36b7ea2d446975fec9fe17ac6698dcf068918c6bb228e9aab5e87"
_COUNTERS = (
    "network_call_count", "retrieval_call_count", "model_load_count",
    "inference_count", "hosted_inference_call_count", "paid_call_count",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl_bytes(content: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{number} must be an object")
        rows.append(row)
    return rows


def _reject_topic(topic_id: object) -> str:
    topic = str(topic_id)
    if topic in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic} is forbidden")
    if topic not in PILOT_TOPIC_IDS:
        raise ValueError(f"unexpected topic {topic}")
    return topic


def normalized_recall_auc(ranking: Sequence[str], relevant: set[str]) -> float:
    """Discrete mean of binary recall at every rank through full topic depth."""

    if not ranking or not relevant:
        return 0.0
    found = 0
    area = 0.0
    for document_id in ranking:
        found += document_id in relevant
        area += found / len(relevant)
    return area / len(ranking)


def _gain(grade: int) -> int:
    return 2**grade - 1 if grade >= 2 else 0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _ndcg(ranking: Sequence[str], qrels: Mapping[str, int], depth: int) -> float:
    gains = [_gain(int(qrels.get(document, 0))) for document in ranking[:depth]]
    dcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_gains = sorted((_gain(int(grade)) for grade in qrels.values()), reverse=True)[:depth]
    ideal = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    return dcg / ideal if ideal else 0.0


def _ranking_rows(value: object, *, topic: str, arm: str) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{topic}/{arm} must be a nonempty complete permutation")
    documents: list[str] = []
    rows: list[dict[str, object]] = []
    for raw in value:
        if isinstance(raw, Mapping):
            row = dict(raw)
            document = str(row.get("document_id"))
        else:
            document, row = str(raw), {"document_id": str(raw)}
        if not document or document == "None":
            raise ValueError("ranking document identity is invalid")
        documents.append(document)
        rows.append(row)
    if len(set(documents)) != len(documents):
        raise ValueError(f"{topic}/{arm} is not a complete permutation")
    return documents, rows


def _validate_inputs(
    rankings: Mapping[str, object],
    qrels: Mapping[str, object],
    provenance: Mapping[str, object],
) -> tuple[dict[str, dict[str, tuple[list[str], list[dict[str, object]]]]], dict[str, dict[str, int]], dict[str, dict[str, list[dict[str, object]]]]]:
    for mapping, label in ((rankings, "rankings"), (qrels, "qrels"), (provenance, "provenance")):
        for topic in mapping:
            _reject_topic(topic)
        if set(mapping) != set(PILOT_TOPIC_IDS):
            raise ValueError(f"{label} must contain the exact pilot topics")
    parsed_rankings: dict[str, dict[str, tuple[list[str], list[dict[str, object]]]]] = {}
    parsed_qrels: dict[str, dict[str, int]] = {}
    parsed_provenance: dict[str, dict[str, list[dict[str, object]]]] = {}
    for topic in PILOT_TOPIC_IDS:
        raw_arms = rankings[topic]
        raw_qrels = qrels[topic]
        raw_provenance = provenance[topic]
        if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARMS):
            raise ValueError(f"{topic} must contain the exact ranking arms")
        if not isinstance(raw_qrels, Mapping) or not isinstance(raw_provenance, Mapping):
            raise ValueError("qrels and provenance topics must be objects")
        topic_arms = {arm: _ranking_rows(raw_arms[arm], topic=topic, arm=arm) for arm in ARMS}
        population = set(topic_arms[ARMS[0]][0])
        if any(set(topic_arms[arm][0]) != population or len(topic_arms[arm][0]) != len(population) for arm in ARMS):
            raise ValueError(f"{topic} arms must be complete permutations of one population")
        if set(map(str, raw_provenance)) != population:
            raise ValueError(f"{topic} provenance differs from ranking population")
        topic_qrels: dict[str, int] = {}
        for document, raw_grade in raw_qrels.items():
            if isinstance(raw_grade, bool) or not isinstance(raw_grade, int) or not 0 <= raw_grade <= 4:
                raise ValueError("qrels grades must be integers from zero through four")
            topic_qrels[str(document)] = raw_grade
        topic_provenance: dict[str, list[dict[str, object]]] = {}
        for document, raw_entries in raw_provenance.items():
            if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)) or not raw_entries:
                raise ValueError("accepted-union provenance must be a nonempty sequence")
            entries = [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
            if len(entries) != len(raw_entries) or any(entry.get("family") not in {"original", "facet"} for entry in entries):
                raise ValueError("accepted-union provenance is invalid")
            topic_provenance[str(document)] = entries
        parsed_rankings[topic], parsed_qrels[topic], parsed_provenance[topic] = topic_arms, topic_qrels, topic_provenance
    return parsed_rankings, parsed_qrels, parsed_provenance


def evaluate_proxy(
    rankings: Mapping[str, object],
    qrels: Mapping[str, object],
    provenance: Mapping[str, object],
    depths: Sequence[int],
) -> dict[str, object]:
    """Compute qrels metrics over complete, equal-population permutations."""

    ordered_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not ordered_depths or any(depth <= 0 for depth in ordered_depths):
        raise ValueError("depths must be unique positive integers")
    parsed, parsed_qrels, parsed_provenance = _validate_inputs(rankings, qrels, provenance)
    arms_result: dict[str, dict[str, object]] = {}
    attribution: dict[str, dict[str, dict[str, int]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        per_topic: dict[str, dict[str, object]] = {}
        for topic in PILOT_TOPIC_IDS:
            ranking, rows = parsed[topic][arm]
            topic_qrels = parsed_qrels[topic]
            relevant = {document for document, grade in topic_qrels.items() if grade >= 2}
            total_gain = sum(_gain(topic_qrels[document]) for document in relevant)
            facet_only = {
                document for document, entries in parsed_provenance[topic].items()
                if {str(entry["family"]) for entry in entries} == {"facet"} and document in relevant
            }
            topic_metrics: dict[str, object] = {
                "total_relevant": len(relevant),
                "total_graded_gain": total_gain,
                "facet_only_relevant_total": len(facet_only),
                "recall_auc": normalized_recall_auc(ranking, relevant),
                "ndcg@10": _ndcg(ranking, topic_qrels, 10),
            }
            for depth in ordered_depths:
                prefix = ranking[:depth]
                prefix_relevant = set(prefix) & relevant
                retrieved_gain = sum(_gain(topic_qrels[document]) for document in prefix_relevant)
                facet_retained = len(set(prefix) & facet_only)
                topic_metrics.update({
                    f"binary_recall@{depth}": _ratio(len(prefix_relevant), len(relevant)),
                    f"graded_recall@{depth}": _ratio(retrieved_gain, total_gain),
                    f"relevant_count@{depth}": len(prefix_relevant),
                    f"judged_rate@{depth}": _ratio(sum(document in topic_qrels for document in prefix), len(prefix)),
                    f"facet_only_relevant_retained@{depth}": facet_retained,
                    f"facet_only_relevant_retention@{depth}": _ratio(facet_retained, len(facet_only)),
                    f"ndcg@{depth}": _ndcg(ranking, topic_qrels, depth),
                })
            full_relevant = set(ranking) & relevant
            full_gain = sum(_gain(topic_qrels[document]) for document in full_relevant)
            topic_metrics.update({
                "binary_recall_full": _ratio(len(full_relevant), len(relevant)),
                "graded_recall_full": _ratio(full_gain, total_gain),
                "relevant_count_full": len(full_relevant),
                "judged_rate_full": _ratio(sum(document in topic_qrels for document in ranking), len(ranking)),
                "facet_only_relevant_retained_full": len(set(ranking) & facet_only),
                "facet_only_relevant_retention_full": _ratio(len(set(ranking) & facet_only), len(facet_only)),
            })
            facet_counts: dict[str, int] = defaultdict(int)
            for document, row in zip(ranking, rows, strict=True):
                facet = row.get("coverage_facet")
                if facet is not None and document in relevant:
                    facet_counts[str(facet)] += 1
            attribution[arm][topic] = dict(sorted(facet_counts.items()))
            per_topic[topic] = topic_metrics
        aggregate: dict[str, object] = {}
        keys = [key for key in per_topic[PILOT_TOPIC_IDS[0]] if key not in {"total_relevant", "total_graded_gain", "facet_only_relevant_total"}]
        for key in keys:
            values = [per_topic[topic][key] for topic in PILOT_TOPIC_IDS]
            aggregate[key] = sum(values) if "count" in key or "retained" in key else math.fsum(float(value) for value in values) / len(values)
        aggregate["total_relevant"] = sum(int(per_topic[topic]["total_relevant"]) for topic in PILOT_TOPIC_IDS)
        aggregate["facet_only_relevant_total"] = sum(int(per_topic[topic]["facet_only_relevant_total"]) for topic in PILOT_TOPIC_IDS)
        arms_result[arm] = {"per_topic": per_topic, "aggregate": aggregate}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for topic in PILOT_TOPIC_IDS:
        control = arms_result["RRF"]["per_topic"][topic]
        deltas[topic] = {}
        for arm in ARMS:
            values = arms_result[arm]["per_topic"][topic]
            deltas[topic][arm] = {
                f"{key}_delta_vs_RRF": float(value) - float(control[key])
                for key, value in values.items()
                if key not in {"total_relevant", "total_graded_gain", "facet_only_relevant_total"}
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(PILOT_TOPIC_IDS),
        "arms": arms_result,
        "depths": [*ordered_depths, "full"],
        "relevance_threshold": 2,
        "per_topic_deltas": deltas,
        "coverage_proxy": {
            "name": "qrels-positive facet attribution",
            "is_true_nugget_coverage": False,
            "caveat": "A qrels-positive document attributed to a marginal facet gain does not prove that the document supports that facet.",
            "qrels_positive_facet_attribution": attribution,
        },
    }


def _load_verified_rankings(freeze: Path, summary: Mapping[str, object]) -> tuple[dict[str, dict[str, list[dict[str, object]]]], dict[str, str]]:
    rows = _read_jsonl_bytes((freeze / "rankings.jsonl").read_bytes(), "rankings")
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {topic: {arm: [] for arm in ARMS} for topic in PILOT_TOPIC_IDS}
    for row in rows:
        topic, arm = _reject_topic(row.get("topic_id")), str(row.get("arm"))
        if arm not in ARMS:
            raise ValueError("ranking arm is invalid")
        grouped[topic][arm].append(row)
    hashes: dict[str, str] = {}
    topic_summary = summary.get("topic_summary")
    for topic in PILOT_TOPIC_IDS:
        expected: set[str] | None = None
        for arm in ARMS:
            arm_rows = grouped[topic][arm]
            if [row.get("rank") for row in arm_rows] != list(range(1, len(arm_rows) + 1)):
                raise ValueError("ranking ranks are not contiguous")
            documents = [str(row.get("document_id")) for row in arm_rows]
            population = set(documents)
            if not documents or len(population) != len(documents) or (expected is not None and population != expected):
                raise ValueError("rankings are not complete permutations")
            expected = population
            hashes[f"{topic}/{arm}"] = _sha256(_canonical_bytes(documents))
        if not isinstance(topic_summary, Mapping) or not isinstance(topic_summary.get(topic), Mapping):
            raise ValueError("verified freeze lacks topic summary")
        complete = topic_summary[topic].get("complete_permutations")  # type: ignore[union-attr]
        if complete != {arm: True for arm in ARMS}:
            raise ValueError("verified freeze lacks complete-permutation attestations")
    return grouped, hashes


def _load_qrels(content: bytes) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {topic: {} for topic in PILOT_TOPIC_IDS}
    observed_topics: list[str] = []
    for row in _read_jsonl_bytes(content, "qrels projection"):
        topic = _reject_topic(row.get("topic_id"))
        if not observed_topics or observed_topics[-1] != topic:
            observed_topics.append(topic)
        document = str(row.get("document_id"))
        grade = row.get("grade")
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4 or document in result[topic]:
            raise ValueError("qrels projection identity or grade is invalid")
        result[topic][document] = grade
    if observed_topics != list(PILOT_TOPIC_IDS) or any(not result[topic] for topic in PILOT_TOPIC_IDS):
        raise ValueError("qrels projection must contain exact contiguous pilot topics")
    return result


def _load_union(content: bytes) -> dict[str, dict[str, list[dict[str, object]]]]:
    result: dict[str, dict[str, list[dict[str, object]]]] = {topic: {} for topic in PILOT_TOPIC_IDS}
    expected_rank = {topic: 1 for topic in PILOT_TOPIC_IDS}
    for row in _read_jsonl_bytes(content, "accepted union"):
        topic, document = _reject_topic(row.get("topic_id")), str(row.get("document_id"))
        if row.get("union_order") != expected_rank[topic] or document in result[topic]:
            raise ValueError("accepted-union order or identity is invalid")
        expected_rank[topic] += 1
        raw = row.get("provenance")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("accepted-union provenance is invalid")
        result[topic][document] = [dict(entry) for entry in raw if isinstance(entry, Mapping)]
        if len(result[topic][document]) != len(raw):
            raise ValueError("accepted-union provenance is invalid")
    return result


def _binding(path: Path, content: bytes, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path.resolve()), "bytes": len(content), "sha256": _sha256(content)}
    if rows is not None:
        value["rows"] = rows
    return value


def evaluate_frozen_proxy(
    freeze: Path,
    qrels: Path,
    union: Path,
    output: Path,
    *,
    expected_qrels_sha256: str = PINNED_QRELS_SHA256,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> dict[str, object]:
    """Verify rankings completely, then open the already-pinned qrels projection."""

    freeze, qrels, union, output = map(Path, (freeze, qrels, union, output))
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    freeze_summary = verify_soft_freeze(freeze)
    rankings, ranking_hashes = _load_verified_rankings(freeze, freeze_summary)
    union_content = union.read_bytes()
    provenance = _load_union(union_content)
    # This is the qrels access boundary: every ranking was sealed, hashed, and
    # checked as a complete permutation above.
    qrels_content = qrels.read_bytes()
    if _sha256(qrels_content) != expected_qrels_sha256:
        raise ValueError("qrels projection SHA-256 differs from the pinned projection")
    parsed_qrels = _load_qrels(qrels_content)
    metrics = evaluate_proxy(rankings, parsed_qrels, provenance, depths)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "ranking_sha256": ranking_hashes,
        "per_topic_deltas": metrics["per_topic_deltas"],
        "coverage_proxy": metrics["coverage_proxy"],
    }
    metrics = {key: value for key, value in metrics.items() if key not in {"per_topic_deltas", "coverage_proxy"}}
    ranking_content = (freeze / "rankings.jsonl").read_bytes()
    bindings = {
        "schema_version": SCHEMA_VERSION,
        "ranking_hashes_verified_before_qrels_open": True,
        "freeze": {"path": str(freeze.resolve()), "seal_root_sha256": freeze_summary.get("root_sha256")},
        "rankings": _binding(freeze / "rankings.jsonl", ranking_content, rows=len(_read_jsonl_bytes(ranking_content, "rankings"))),
        "qrels_projection": _binding(qrels, qrels_content, rows=len(_read_jsonl_bytes(qrels_content, "qrels projection"))),
        "accepted_union": _binding(union, union_content, rows=len(_read_jsonl_bytes(union_content, "accepted union"))),
    }
    artifacts = {"metrics.json": _pretty_bytes(metrics), "diagnostics.json": _pretty_bytes(diagnostics), "input_bindings.json": _pretty_bytes(bindings)}
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "qrels_opened": True,
        "topic_ids": list(PILOT_TOPIC_IDS), "arms": list(ARMS), "depths": [*map(int, depths), "full"],
        "relevance_threshold": 2, "protected_topic_count": 0, "external_cost_usd": 0.0,
        **{name: 0 for name in _COUNTERS},
        "artifacts": {name: {"bytes": len(content), "sha256": _sha256(content)} for name, content in artifacts.items()},
    }
    artifacts["summary.json"] = _pretty_bytes(summary)
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        path = output / name
        with path.open("xb") as sink:
            sink.write(content); sink.flush(); os.fsync(sink.fileno())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--qrels", required=True, type=Path)
    evaluate.add_argument("--union", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_frozen_proxy(args.freeze, args.qrels, args.union, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

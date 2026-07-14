"""Build and evaluate the fixed post-qrels RRF→GLOBAL→DUAL diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .deep_facet_candidate_manifest import TOPIC_IDS
from .deep_facet_candidate_evaluate import evaluate_ranking
from .deep_facet_candidate_rank import ARM_NAMES, verify_seal


CASCADE_ARM = "RRF-GLOBAL-DUAL"
RRF_HEAD_DEPTH = 10
GLOBAL_END_DEPTH = 100
NOVEL_RETENTION_MINIMUM_COUNT = 142
NOVEL_RELEVANT_TOTAL = 177
SCHEMA_VERSION = "deep-facet-candidate-cascade-v1"
SEAL_SCHEMA_VERSION = "deep-facet-candidate-cascade-seal-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_source_rankings(source_freeze: Path) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, list[tuple[int, str]]]] = {
        topic: {arm: [] for arm in ARM_NAMES} for topic in TOPIC_IDS
    }
    try:
        lines = (source_freeze / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("source rankings are unreadable") from exc
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"source rankings line {number} is invalid") from exc
        if not isinstance(row, Mapping):
            raise ValueError("source ranking row must be an object")
        topic_id, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic_id not in collected or arm not in collected[topic_id]:
            raise ValueError("source ranking has an unexpected topic or arm")
        collected[topic_id][arm].append((int(row["rank"]), str(row["document_id"])))
    result: dict[str, dict[str, list[str]]] = {}
    for topic_id in TOPIC_IDS:
        result[topic_id] = {}
        for arm in ARM_NAMES:
            ordered = sorted(collected[topic_id][arm])
            if [rank for rank, _ in ordered] != list(range(1, len(ordered) + 1)):
                raise ValueError("source ranking is not contiguous")
            result[topic_id][arm] = [document_id for _, document_id in ordered]
    return result


def _validate_permutation(values: Sequence[str], label: str) -> list[str]:
    result = [str(value) for value in values]
    if not result or any(not value for value in result):
        raise ValueError(f"{label} must be a non-empty permutation")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains a duplicate document ID")
    return result


def build_cascade(
    rrf: Sequence[str],
    global_arm: Sequence[str],
    dual: Sequence[str],
) -> list[str]:
    """Return the fixed complete cascade without inspecting scores or qrels."""

    rrf_ids = _validate_permutation(rrf, "RRF")
    global_ids = _validate_permutation(global_arm, "GLOBAL")
    dual_ids = _validate_permutation(dual, "DUAL")
    population = set(rrf_ids)
    if set(global_ids) != population or set(dual_ids) != population:
        raise ValueError("cascade arms must contain the same complete population")
    if len(population) < GLOBAL_END_DEPTH:
        raise ValueError("cascade population must contain at least 100 documents")

    selected = rrf_ids[:RRF_HEAD_DEPTH]
    selected_set = set(selected)
    for document_id in global_ids:
        if document_id not in selected_set:
            selected.append(document_id)
            selected_set.add(document_id)
            if len(selected) == GLOBAL_END_DEPTH:
                break
    selected.extend(document_id for document_id in dual_ids if document_id not in selected_set)
    if len(selected) != len(population) or set(selected) != population:
        raise ValueError("cascade failed to preserve the complete population")
    return selected


def decide_cascade(
    cascade: Mapping[str, object],
    baselines: Mapping[str, Mapping[str, object]],
    per_topic: Mapping[str, Mapping[str, object]],
    *,
    judged_coverage_defensible: bool | None,
) -> dict[str, object]:
    """Apply the advisor's fixed guards without inventing a coverage threshold."""

    rrf = baselines["RRF"]
    global_arm = baselines["GLOBAL"]
    guards = {
        "ndcg_10_exactly_preserves_rrf": float(cascade["ndcg@10"]) == float(rrf["ndcg@10"]),
        "graded_recall_500_beats_rrf": float(cascade["graded_recall@500"]) > float(rrf["graded_recall@500"]),
        "graded_recall_500_beats_global": float(cascade["graded_recall@500"]) > float(global_arm["graded_recall@500"]),
        "graded_recall_1000_not_below_rrf": float(cascade["graded_recall@1000"]) >= float(rrf["graded_recall@1000"]),
        "novel_retention_1000_at_least_142_of_177": (
            int(cascade["novel_retained@1000"]) >= NOVEL_RETENTION_MINIMUM_COUNT
            and float(cascade["novel_retention@1000"])
            >= NOVEL_RETENTION_MINIMUM_COUNT / NOVEL_RELEVANT_TOTAL
        ),
        "per_topic_graded_recall_500_floor": all(
            float(row["delta_vs_RRF"]) >= -0.02 for row in per_topic.values()
        ),
    }
    failed = [name for name, passed in guards.items() if not passed]
    mechanical_pass = not failed
    coverage_status = (
        "required" if judged_coverage_defensible is None
        else "defensible" if judged_coverage_defensible
        else "not_defensible"
    )
    return {
        "arm": CASCADE_ARM,
        "post_qrels_diagnostic": True,
        "production_promotion_authorized": False,
        "guards": guards,
        "failed_guards": failed,
        "mechanical_guards_pass": mechanical_pass,
        "coverage_review_status": coverage_status,
        "advance_to_fresh_validation": mechanical_pass and judged_coverage_defensible is True,
    }


def _mean(values: Sequence[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return math.fsum(present) / len(present) if present else None


def evaluate_cascade_payload(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    source_metrics: Mapping[str, object],
    *,
    judged_coverage_defensible: bool | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate an already frozen cascade against the existing qrels projection."""

    if list(rankings) != list(TOPIC_IDS) or list(qrels) != list(TOPIC_IDS):
        raise ValueError("cascade rankings and qrels must contain frozen topics in order")
    source_per_topic = source_metrics.get("per_topic")
    source_aggregate = source_metrics.get("aggregate")
    discovery = source_metrics.get("discovery")
    if not all(isinstance(value, Mapping) for value in (source_per_topic, source_aggregate, discovery)):
        raise ValueError("source metrics lack required evidence")
    assert isinstance(source_per_topic, Mapping)
    assert isinstance(source_aggregate, Mapping)
    assert isinstance(discovery, Mapping)

    per_topic: dict[str, dict[str, object]] = {}
    novel_by_topic: dict[str, set[str]] = {}
    for topic_id in TOPIC_IDS:
        ranking = [str(value) for value in rankings[topic_id]]
        if len(ranking) != len(set(ranking)):
            raise ValueError("cascade ranking contains duplicate documents")
        evaluated = evaluate_ranking(ranking, qrels[topic_id])
        discovery_row = discovery[topic_id]
        if not isinstance(discovery_row, Mapping) or not isinstance(discovery_row.get("novel_relevant_ids"), list):
            raise ValueError("source discovery lacks novel relevant IDs")
        novel_by_topic[topic_id] = set(map(str, discovery_row["novel_relevant_ids"]))
        row: dict[str, object] = {
            "ndcg@10": evaluated["ndcg@10"],
            "ndcg@100": evaluated["ndcg@100"],
        }
        for depth in (100, 500, 1000):
            depth_row = evaluated[str(depth)]
            assert isinstance(depth_row, Mapping)
            for field in ("recall", "graded_recall", "judged_rate", "relevant_count"):
                row[f"{field}@{depth}"] = depth_row[field]
            retained = len(novel_by_topic[topic_id].intersection(ranking[:depth]))
            row[f"novel_retained@{depth}"] = retained
            row[f"novel_retention@{depth}"] = (
                retained / len(novel_by_topic[topic_id]) if novel_by_topic[topic_id] else None
            )
        source_topic = source_per_topic[topic_id]
        if not isinstance(source_topic, Mapping) or not isinstance(source_topic.get("RRF"), Mapping):
            raise ValueError("source metrics lack topic RRF evidence")
        rrf_topic = source_topic["RRF"]
        assert isinstance(rrf_topic, Mapping)
        row["graded_recall@500_delta_vs_RRF"] = (
            float(row["graded_recall@500"]) - float(rrf_topic["graded_recall@500"])
        )
        per_topic[topic_id] = row

    aggregate: dict[str, object] = {}
    for field in ("ndcg@10", "ndcg@100"):
        aggregate[field] = _mean([per_topic[topic][field] for topic in TOPIC_IDS])
    for depth in (100, 500, 1000):
        for field in ("recall", "graded_recall", "judged_rate"):
            key = f"{field}@{depth}"
            aggregate[key] = _mean([per_topic[topic][key] for topic in TOPIC_IDS])
        retained = sum(int(per_topic[topic][f"novel_retained@{depth}"]) for topic in TOPIC_IDS)
        aggregate[f"novel_retained@{depth}"] = retained
        novel_total = sum(len(novel_by_topic[topic]) for topic in TOPIC_IDS)
        aggregate[f"novel_retention@{depth}"] = retained / novel_total if novel_total else None

    rrf = source_aggregate.get("RRF")
    global_arm = source_aggregate.get("GLOBAL")
    if not isinstance(rrf, Mapping) or not isinstance(global_arm, Mapping):
        raise ValueError("source metrics lack RRF or GLOBAL aggregate evidence")
    decision_rows = {
        topic: {
            "graded_recall@500": per_topic[topic]["graded_recall@500"],
            "delta_vs_RRF": per_topic[topic]["graded_recall@500_delta_vs_RRF"],
        }
        for topic in TOPIC_IDS
    }
    decision = decide_cascade(
        aggregate,
        {"RRF": rrf, "GLOBAL": global_arm},
        decision_rows,
        judged_coverage_defensible=judged_coverage_defensible,
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "post_qrels_diagnostic": True,
        "confirmatory_evidence": False,
        "topic_ids": list(TOPIC_IDS),
        "relevance_threshold": 2,
        "novel_relevant_count": sum(len(value) for value in novel_by_topic.values()),
        "aggregate": {CASCADE_ARM: aggregate},
        "per_topic": per_topic,
        "baseline": {"RRF": dict(rrf), "GLOBAL": dict(global_arm)},
    }
    return metrics, decision


def _artifact_record(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": _sha256(value)}


def freeze_cascade(source_freeze: Path, output: Path) -> dict[str, object]:
    """Create and seal the cascade without accepting or opening qrels."""

    source_freeze, output = Path(source_freeze), Path(output)
    if output.exists():
        raise FileExistsError(f"create-only cascade output already exists: {output}")
    source_seal = verify_seal(source_freeze)
    source_rankings = _load_source_rankings(source_freeze)
    rows: list[dict[str, object]] = []
    topic_summary: dict[str, object] = {}
    for topic_id in TOPIC_IDS:
        arms = source_rankings[topic_id]
        cascade = build_cascade(arms["RRF"], arms["GLOBAL"], arms["DUAL"])
        source_ranks = {
            arm: {document_id: rank for rank, document_id in enumerate(arms[arm], start=1)}
            for arm in ("RRF", "GLOBAL", "DUAL")
        }
        for rank, document_id in enumerate(cascade, start=1):
            source_arm = "RRF" if rank <= RRF_HEAD_DEPTH else (
                "GLOBAL" if rank <= GLOBAL_END_DEPTH else "DUAL"
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": CASCADE_ARM,
                    "rank": rank,
                    "document_id": document_id,
                    "source_arm": source_arm,
                    "source_rank": source_ranks[source_arm][document_id],
                }
            )
        topic_summary[topic_id] = {
            "candidate_count": len(cascade),
            "rrf_head_count": RRF_HEAD_DEPTH,
            "global_middle_count": GLOBAL_END_DEPTH - RRF_HEAD_DEPTH,
            "dual_tail_count": len(cascade) - GLOBAL_END_DEPTH,
        }

    output.mkdir(parents=True)
    parameters = {
        "schema_version": SCHEMA_VERSION,
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "topic_ids": list(TOPIC_IDS),
        "arm": CASCADE_ARM,
        "rrf_ranks": [1, RRF_HEAD_DEPTH],
        "global_ranks": [RRF_HEAD_DEPTH + 1, GLOBAL_END_DEPTH],
        "dual_start_rank": GLOBAL_END_DEPTH + 1,
        "skip_duplicates": True,
        "complete_permutation_required": True,
        "retrieval_calls": 0,
        "model_inference_calls": 0,
    }
    source_ranking_path = source_freeze / "rankings.jsonl"
    binding = {
        "schema_version": SCHEMA_VERSION,
        "source_freeze_path": str(source_freeze.resolve()),
        "source_seal_root_sha256": source_seal["root_sha256"],
        "source_rankings_sha256": _sha256(source_ranking_path.read_bytes()),
        "qrels_read": False,
    }
    ranking_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _exclusive_bytes(output / "parameters.json", _pretty_bytes(parameters))
    _exclusive_bytes(output / "input_binding.json", _pretty_bytes(binding))
    _exclusive_bytes(output / "rankings.jsonl", ranking_bytes)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "topic_ids": list(TOPIC_IDS),
        "arm": CASCADE_ARM,
        "ranking_row_count": len(rows),
        "topic_summary": topic_summary,
        "artifacts": {
            name: _artifact_record(output / name)
            for name in ("parameters.json", "input_binding.json", "rankings.jsonl")
        },
    }
    _exclusive_bytes(output / "summary.json", _pretty_bytes(summary))
    names = ("input_binding.json", "parameters.json", "rankings.jsonl", "summary.json")
    artifacts = {name: _artifact_record(output / name) for name in names}
    seal_material = {
        "topic_ids": list(TOPIC_IDS),
        "source_seal_root_sha256": source_seal["root_sha256"],
        "artifacts": artifacts,
    }
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "cascade_frozen_before_diagnostic_metrics",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        **seal_material,
        "root_sha256": _sha256(_canonical_bytes(seal_material)),
    }
    _exclusive_bytes(output / "SEALED.json", _pretty_bytes(seal))
    verify_cascade_freeze(output)
    return summary


def verify_cascade_freeze(output: Path) -> dict[str, object]:
    output = Path(output)
    seal = _read_object(output / "SEALED.json", "cascade seal")
    if (
        seal.get("schema_version") != SEAL_SCHEMA_VERSION
        or seal.get("status") != "cascade_frozen_before_diagnostic_metrics"
        or seal.get("qrels_read") is not False
        or seal.get("topic_ids") != list(TOPIC_IDS)
    ):
        raise ValueError("cascade seal contract differs")
    expected_names = {
        "SEALED.json",
        "input_binding.json",
        "parameters.json",
        "rankings.jsonl",
        "summary.json",
    }
    actual_names = {path.name for path in output.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("cascade freeze has missing or extra files")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("cascade seal artifacts are missing")
    actual = {
        name: _artifact_record(output / name)
        for name in ("input_binding.json", "parameters.json", "rankings.jsonl", "summary.json")
    }
    if actual != artifacts:
        raise ValueError("cascade artifact bytes were mutated")
    material = {
        "topic_ids": seal["topic_ids"],
        "source_seal_root_sha256": seal["source_seal_root_sha256"],
        "artifacts": artifacts,
    }
    if seal.get("root_sha256") != _sha256(_canonical_bytes(material)):
        raise ValueError("cascade seal root hash differs")
    return seal


def _load_cascade_rankings(output: Path) -> dict[str, list[str]]:
    collected: dict[str, list[tuple[int, str]]] = {topic: [] for topic in TOPIC_IDS}
    try:
        lines = (output / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("cascade rankings are unreadable") from exc
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cascade rankings line {number} is invalid") from exc
        if not isinstance(row, Mapping):
            raise ValueError("cascade ranking row must be an object")
        topic_id = str(row.get("topic_id"))
        if topic_id not in collected or row.get("arm") != CASCADE_ARM:
            raise ValueError("cascade ranking has an unexpected topic or arm")
        collected[topic_id].append((int(row["rank"]), str(row["document_id"])))
    result: dict[str, list[str]] = {}
    for topic_id in TOPIC_IDS:
        ordered = sorted(collected[topic_id])
        if [rank for rank, _ in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("cascade ranking is not contiguous")
        result[topic_id] = [document_id for _, document_id in ordered]
        if len(result[topic_id]) != len(set(result[topic_id])):
            raise ValueError("cascade ranking contains duplicate documents")
    return result


def verify_cascade_semantics(output: Path) -> dict[str, object]:
    """Recompute the fixed splice from its sealed source and compare every rank."""

    output = Path(output)
    cascade_seal = verify_cascade_freeze(output)
    binding = _read_object(output / "input_binding.json", "cascade input binding")
    source_freeze = Path(str(binding.get("source_freeze_path")))
    source_seal = verify_seal(source_freeze)
    source_rankings_path = source_freeze / "rankings.jsonl"
    if (
        source_seal.get("root_sha256") != cascade_seal.get("source_seal_root_sha256")
        or binding.get("source_seal_root_sha256") != source_seal.get("root_sha256")
        or binding.get("source_rankings_sha256")
        != _sha256(source_rankings_path.read_bytes())
    ):
        raise ValueError("cascade source seal binding differs")
    source_rankings = _load_source_rankings(source_freeze)
    actual = _load_cascade_rankings(output)
    for topic_id in TOPIC_IDS:
        arms = source_rankings[topic_id]
        expected = build_cascade(arms["RRF"], arms["GLOBAL"], arms["DUAL"])
        if actual[topic_id] != expected:
            raise ValueError(f"cascade semantic splice differs for topic {topic_id}")
    return {
        "status": "cascade_semantics_verified",
        "topic_ids": list(TOPIC_IDS),
        "cascade_seal_root_sha256": cascade_seal["root_sha256"],
        "source_seal_root_sha256": source_seal["root_sha256"],
    }


def _load_projected_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("qrels projection is unreadable") from exc
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"qrels projection line {number} is invalid") from exc
        if not isinstance(row, Mapping):
            raise ValueError("qrels projection row must be an object")
        topic_id, document_id = str(row.get("topic_id")), str(row.get("document_id"))
        if topic_id not in result:
            raise ValueError("qrels projection contains an unexpected topic")
        if document_id in result[topic_id]:
            raise ValueError("qrels projection contains a duplicate identity")
        result[topic_id][document_id] = int(row["grade"])
    if any(not result[topic] for topic in TOPIC_IDS):
        raise ValueError("qrels projection lacks a frozen topic")
    return result


def evaluate_cascade(
    cascade_freeze: Path,
    source_evaluation: Path,
    output: Path,
    *,
    judged_coverage_defensible: bool | None = None,
) -> dict[str, object]:
    """Evaluate the sealed diagnostic using only the existing qrels projection."""

    cascade_freeze, source_evaluation, output = (
        Path(cascade_freeze),
        Path(source_evaluation),
        Path(output),
    )
    semantic_receipt = verify_cascade_semantics(cascade_freeze)
    if output.exists():
        raise FileExistsError(f"create-only diagnostic output already exists: {output}")

    source_summary = _read_object(source_evaluation / "summary.json", "source evaluation summary")
    source_metrics_path = source_evaluation / "metrics.json"
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("qrels_opened") is not True
        or source_summary.get("metrics_sha256") != _sha256(source_metrics_path.read_bytes())
    ):
        raise ValueError("source evaluation is incomplete or mutated")
    projection_path = source_evaluation / "qrels_projection.jsonl"
    receipt = _read_object(
        source_evaluation / "qrels_access_receipt.json", "qrels access receipt"
    )
    if receipt.get("qrels_projection_sha256") != _sha256(projection_path.read_bytes()):
        raise ValueError("qrels projection differs from its access receipt")
    source_metrics = _read_object(source_metrics_path, "source evaluation metrics")
    rankings = _load_cascade_rankings(cascade_freeze)
    qrels = _load_projected_qrels(projection_path)
    metrics, decision = evaluate_cascade_payload(
        rankings,
        qrels,
        source_metrics,
        judged_coverage_defensible=judged_coverage_defensible,
    )

    output.mkdir(parents=True)
    metrics_bytes = _pretty_bytes(metrics)
    decision_bytes = _pretty_bytes(decision)
    _exclusive_bytes(output / "metrics.json", metrics_bytes)
    _exclusive_bytes(output / "decision.json", decision_bytes)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "confirmatory_evidence": False,
        "topic_ids": list(TOPIC_IDS),
        "cascade_seal_root_sha256": semantic_receipt["cascade_seal_root_sha256"],
        "source_qrels_projection_sha256": _sha256(projection_path.read_bytes()),
        "new_retrieval_calls": 0,
        "new_model_inference_calls": 0,
        "mechanical_guards_pass": decision["mechanical_guards_pass"],
        "coverage_review_status": decision["coverage_review_status"],
        "advance_to_fresh_validation": decision["advance_to_fresh_validation"],
        "metrics_sha256": _sha256(metrics_bytes),
        "decision_sha256": _sha256(decision_bytes),
    }
    _exclusive_bytes(output / "summary.json", _pretty_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source-freeze", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--freeze", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--source-evaluation", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_cascade(args.source_freeze, args.output)
    elif args.command == "verify":
        result = verify_cascade_semantics(args.freeze)
    else:
        result = evaluate_cascade(args.freeze, args.source_evaluation, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Projection-only evaluation for protected tethered facet rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .deep_facet_candidate_evaluate import evaluate_ranking
from .tethered_facet_two_basket import PILOT_TOPIC_IDS, verify_freeze


TOPIC_IDS = PILOT_TOPIC_IDS
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
NOVEL_RELEVANT_TOTAL = 177
SCHEMA_VERSION = "tethered-facet-evaluation-v1"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_object_bytes(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, content


def _exclusive_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _binding(path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": _sha256(content),
    }


def validate_protected_head(*, rrf: Sequence[object], arm: Sequence[object]) -> None:
    """Require exact ordered identity of the protected first 100 documents."""

    if len(rrf) < 100 or len(arm) < 100 or list(rrf[:100]) != list(arm[:100]):
        raise ValueError("arm top 100 must exactly match RRF top 100")


def _parse_projection(content: bytes) -> dict[str, dict[str, int]]:
    lines = content.splitlines()
    parsed: list[tuple[str, str, int]] = []
    observed_order: list[str] = []
    previous: str | None = None
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError
            raw_topic, raw_document, raw_grade = (
                row["topic_id"], row["document_id"], row["grade"]
            )
            if (
                not isinstance(raw_topic, str)
                or not isinstance(raw_document, str)
                or not isinstance(raw_grade, int)
                or isinstance(raw_grade, bool)
            ):
                raise TypeError
            topic_id, document_id, grade = raw_topic, raw_document, raw_grade
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"qrels projection line {line_number} is invalid") from exc
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"qrels projection contains protected topic {topic_id}")
        if topic_id not in TOPIC_IDS:
            raise ValueError(f"qrels projection contains unexpected topic {topic_id}")
        if not document_id:
            raise ValueError("qrels projection contains an empty document ID")
        if topic_id != previous:
            observed_order.append(topic_id)
            previous = topic_id
        parsed.append((topic_id, document_id, grade))
    if observed_order != list(TOPIC_IDS):
        raise ValueError("qrels projection must contain the exact topics in order")
    result: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    for topic_id, document_id, grade in parsed:
        if document_id in result[topic_id]:
            raise ValueError("qrels projection contains duplicate topic-document identities")
        result[topic_id][document_id] = grade
    if any(not result[topic] for topic in TOPIC_IDS):
        raise ValueError("qrels projection must contain the exact topics in order")
    return result


def load_projection(path: Path) -> dict[str, dict[str, int]]:
    """Read the sealed JSONL projection and enforce its topic firewall."""

    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"qrels projection is unreadable: {path}") from exc
    return _parse_projection(content)


def _require_exact_topics(value: Mapping[str, object], label: str) -> None:
    if list(value) != list(TOPIC_IDS):
        raise ValueError(f"{label} must contain the exact topics in order")


def derive_novel_set(
    qrels: Mapping[str, Mapping[str, int]],
    accepted_facet_candidates: Mapping[str, Sequence[str] | set[str]],
    original_at_1000: Mapping[str, Sequence[str]],
    *,
    expected_total: int = NOVEL_RELEVANT_TOTAL,
) -> dict[str, set[str]]:
    """Derive relevant accepted-facet candidates absent from original@1000."""

    for value, label in (
        (qrels, "qrels"),
        (accepted_facet_candidates, "accepted facet candidates"),
        (original_at_1000, "original rankings"),
    ):
        _require_exact_topics(value, label)
    result = {
        topic: {
            document_id
            for document_id in map(str, accepted_facet_candidates[topic])
            if int(qrels[topic].get(document_id, 0)) >= 2
            and document_id not in set(map(str, original_at_1000[topic][:1000]))
        }
        for topic in TOPIC_IDS
    }
    observed = sum(map(len, result.values()))
    if observed != expected_total:
        raise ValueError(
            f"frozen novel relevant set must contain exactly {expected_total} documents; "
            f"found {observed}"
        )
    return result


def _mean(values: Sequence[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return math.fsum(present) / len(present) if present else None


def evaluate_arm(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    novel_set: Mapping[str, set[str]],
    *,
    ranking_rows: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    depths: Sequence[int] = (500, 1000),
) -> dict[str, object]:
    """Evaluate one frozen arm with shared ranking metrics and diagnostics."""

    for value, label in ((rankings, "rankings"), (qrels, "qrels"), (novel_set, "novel set")):
        _require_exact_topics(value, label)
    ordered_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not ordered_depths or any(depth <= 0 for depth in ordered_depths):
        raise ValueError("evaluation depths must be positive")
    per_topic: dict[str, dict[str, object]] = {}
    for topic in TOPIC_IDS:
        ranking = list(map(str, rankings[topic]))
        metrics = evaluate_ranking(ranking, qrels[topic], depths=ordered_depths)
        topic_row: dict[str, object] = {
            "ndcg@10": metrics["ndcg@10"],
            "ndcg@100": metrics["ndcg@100"],
        }
        for depth in ordered_depths:
            values = metrics[str(depth)]
            if not isinstance(values, Mapping):
                raise ValueError("shared evaluator returned invalid depth metrics")
            retained = len(set(ranking[:depth]).intersection(novel_set[topic]))
            topic_row.update(
                {
                    f"recall@{depth}": values["recall"],
                    f"graded_recall@{depth}": values["graded_recall"],
                    f"judged_rate@{depth}": values["judged_rate"],
                    f"novel_retained@{depth}": retained,
                    f"novel_retention@{depth}": (
                        retained / len(novel_set[topic]) if novel_set[topic] else None
                    ),
                }
            )
        per_topic[topic] = topic_row
    novel_total = sum(map(len, novel_set.values()))
    aggregate: dict[str, object] = {
        "ndcg@10": _mean([per_topic[topic]["ndcg@10"] for topic in TOPIC_IDS]),
        "ndcg@100": _mean([per_topic[topic]["ndcg@100"] for topic in TOPIC_IDS]),
    }
    for depth in ordered_depths:
        for field in ("recall", "graded_recall", "judged_rate"):
            aggregate[f"{field}@{depth}"] = _mean(
                [per_topic[topic][f"{field}@{depth}"] for topic in TOPIC_IDS]
            )
        retained = sum(int(per_topic[topic][f"novel_retained@{depth}"]) for topic in TOPIC_IDS)
        aggregate[f"novel_retained@{depth}"] = retained
        aggregate[f"novel_retention@{depth}"] = retained / novel_total if novel_total else None

    basket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"selected_count": 0, "relevant_count": 0}
    )
    facets: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"selected_count": 0, "relevant_count": 0}
    )
    if ranking_rows is not None:
        _require_exact_topics(ranking_rows, "ranking rows")
        diagnostic_depth = 500 if 500 in ordered_depths else max(ordered_depths)
        for topic in TOPIC_IDS:
            for row in ranking_rows[topic]:
                rank = int(row.get("rank", 0))
                if not 1 <= rank <= diagnostic_depth:
                    continue
                document_id = str(row.get("document_id"))
                source = str(row.get("source"))
                relevant = int(qrels[topic].get(document_id, 0)) >= 2
                basket[source]["selected_count"] += 1
                basket[source]["relevant_count"] += int(relevant)
                facet_id = row.get("generating_facet")
                if facet_id is not None:
                    facet = facets[str(facet_id)]
                    facet["selected_count"] = int(facet["selected_count"]) + 1
                    facet["relevant_count"] = int(facet["relevant_count"]) + int(relevant)
        for facet in facets.values():
            facet["relevant_yield"] = int(facet["relevant_count"]) / int(facet["selected_count"])
    return {
        "per_topic": per_topic,
        "aggregate": aggregate,
        "basket_contributions": dict(sorted(basket.items())),
        "facet_yield": dict(sorted(facets.items())),
    }


def _at_least(value: object, *controls: object) -> bool:
    return all(float(value) >= float(control) for control in controls)


def decide(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the frozen diagnostic rule without discretionary interpretation."""

    aggregate = evidence["aggregate"]
    per_topic = evidence["per_topic"]
    if not isinstance(aggregate, Mapping) or not isinstance(per_topic, Mapping):
        raise ValueError("decision evidence lacks aggregate or per-topic metrics")
    rrf = aggregate["RRF"]
    facet = aggregate["FACET-2B"]
    tethered = aggregate["TETHERED-2B"]
    if not all(isinstance(value, Mapping) for value in (rrf, facet, tethered)):
        raise ValueError("decision evidence lacks required arms")
    if not (
        isinstance(rrf, Mapping)
        and isinstance(facet, Mapping)
        and isinstance(tethered, Mapping)
    ):
        raise ValueError("decision evidence lacks required arm metrics")
    recall500 = all(
        _at_least(tethered[f"{field}@500"], rrf[f"{field}@500"], facet[f"{field}@500"])
        for field in ("recall", "graded_recall")
    ) and any(
        float(tethered[f"{field}@500"]) > float(facet[f"{field}@500"])
        for field in ("recall", "graded_recall")
    )
    recall1000 = all(
        _at_least(tethered[f"{field}@1000"], rrf[f"{field}@1000"], facet[f"{field}@1000"])
        for field in ("recall", "graded_recall")
    )
    per_topic_loss = True
    for topic in TOPIC_IDS:
        row = per_topic.get(topic)
        if not isinstance(row, Mapping):
            raise ValueError("decision evidence lacks exact per-topic metrics")
        candidate = row["TETHERED-2B"]
        if not isinstance(candidate, Mapping):
            raise ValueError("decision evidence lacks TETHERED-2B per-topic metrics")
        for control_name in ("RRF", "FACET-2B"):
            control = row[control_name]
            if not isinstance(control, Mapping):
                raise ValueError("decision evidence lacks control per-topic metrics")
            for field in ("recall", "graded_recall"):
                if (
                    float(candidate[f"{field}@500"])
                    - float(control[f"{field}@500"])
                    < -0.02 - 1e-12
                ):
                    per_topic_loss = False
    judged_coverage = all(
        float(tethered[f"judged_rate@{depth}"])
        - float(control[f"judged_rate@{depth}"])
        >= -0.05 - 1e-12
        for depth in (500, 1000)
        for control in (rrf, facet)
    )
    guards = {
        "top100_identity": evidence.get("top100_identity") is True,
        "recall500": recall500,
        "novel500": int(tethered["novel_retained@500"]) >= 89,
        "recall1000": recall1000,
        "novel1000": int(tethered["novel_retained@1000"]) >= 142,
        "per_topic_loss": per_topic_loss,
        "judged_coverage": judged_coverage,
        "basket_capacity": evidence.get("documented_basket_shortage") is not True,
    }
    failed = [name for name, passed in guards.items() if not passed]
    if not failed:
        label = "mechanical_pass"
    elif set(failed).issubset({"judged_coverage", "basket_capacity"}):
        label = "inconclusive"
    else:
        label = "mechanical_fail"
    return {"label": label, "guards": guards, "failed_guards": failed}


def _load_frozen_rankings(
    freeze_dir: Path,
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, list[dict[str, object]]]],
    dict[str, object],
    bytes,
]:
    bindings, binding_bytes = _read_object_bytes(
        freeze_dir / "input_bindings.json", "freeze input bindings"
    )
    raw_topic_inputs = bindings.get("topic_inputs")
    if not isinstance(raw_topic_inputs, Mapping) or set(raw_topic_inputs) != {
        "FACET-2B",
        "TETHERED-2B",
    }:
        raise ValueError("freeze lacks exact arm semantic inputs")
    facet_inputs = raw_topic_inputs["FACET-2B"]
    if not isinstance(facet_inputs, Mapping) or set(facet_inputs) != set(TOPIC_IDS):
        raise ValueError("freeze semantic inputs lack exact topics in order")
    rrf: dict[str, list[str]] = {}
    for topic in TOPIC_IDS:
        raw = facet_inputs[topic]
        if not isinstance(raw, Mapping) or not isinstance(raw.get("rrf"), list):
            raise ValueError("freeze semantic topic input is invalid")
        rrf[topic] = list(map(str, raw["rrf"]))

    try:
        ranking_bytes = (freeze_dir / "rankings.jsonl").read_bytes()
    except OSError as exc:
        raise ValueError("frozen rankings are unreadable") from exc
    collected: dict[str, dict[str, list[dict[str, object]]]] = {
        topic: {"FACET-2B": [], "TETHERED-2B": []} for topic in TOPIC_IDS
    }
    for line_number, line in enumerate(ranking_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"frozen rankings line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError("frozen ranking rows must be objects")
        topic, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic in PROTECTED_TOPIC_IDS:
            raise ValueError("frozen rankings contain a protected topic")
        if topic not in collected or arm not in collected[topic]:
            raise ValueError("frozen rankings contain an unexpected topic or arm")
        collected[topic][arm].append(row)
    rankings: dict[str, dict[str, list[str]]] = {
        arm: {} for arm in ("FACET-2B", "TETHERED-2B")
    }
    for topic in TOPIC_IDS:
        for arm in ("FACET-2B", "TETHERED-2B"):
            rows = sorted(collected[topic][arm], key=lambda row: int(row["rank"]))
            if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("frozen ranking is not contiguous")
            ids = [str(row["document_id"]) for row in rows]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("frozen ranking is empty or contains duplicates")
            collected[topic][arm] = rows
            rankings[arm][topic] = ids
    return rankings, collected, bindings, binding_bytes


def _novel_inputs(
    bindings: Mapping[str, object], qrels: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    raw_arms = bindings["topic_inputs"]
    if not isinstance(raw_arms, Mapping):
        raise ValueError("freeze semantic arm inputs are invalid")
    raw_topics = raw_arms["FACET-2B"]
    if not isinstance(raw_topics, Mapping):
        raise ValueError("freeze semantic topic inputs are invalid")
    facets: dict[str, set[str]] = {}
    original: dict[str, list[str]] = {}
    for topic in TOPIC_IDS:
        raw = raw_topics[topic]
        if not isinstance(raw, Mapping) or not isinstance(raw.get("facets"), list):
            raise ValueError("freeze semantic facets are invalid")
        candidates: set[str] = set()
        for facet in raw["facets"]:
            if not isinstance(facet, Mapping) or not isinstance(facet.get("scores"), Mapping):
                raise ValueError("freeze semantic facet scores are invalid")
            candidates.update(map(str, facet["scores"]))
        facets[topic] = candidates
        rrf = raw.get("rrf")
        if not isinstance(rrf, list):
            raise ValueError("freeze semantic RRF ranking is invalid")
        original[topic] = list(map(str, rrf[:1000]))
    return derive_novel_set(
        qrels, facets, original, expected_total=NOVEL_RELEVANT_TOTAL
    ), original


def _validate_prior_novel_set(
    metrics: Mapping[str, object], novel: Mapping[str, set[str]]
) -> None:
    if metrics.get("topic_ids") != list(TOPIC_IDS):
        raise ValueError("prior metrics lack exact topics in order")
    discovery = metrics.get("discovery")
    if not isinstance(discovery, Mapping) or set(discovery) != set(TOPIC_IDS):
        raise ValueError("prior metrics lack frozen discovery evidence")
    for topic in TOPIC_IDS:
        row = discovery[topic]
        if not isinstance(row, Mapping) or not isinstance(row.get("novel_relevant_ids"), list):
            raise ValueError("prior metrics lack frozen novel-set evidence")
        if set(map(str, row["novel_relevant_ids"])) != novel[topic]:
            raise ValueError("derived novel set differs from prior frozen metrics")


def _per_topic_deltas(
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    fields = tuple(
        f"{metric}@{depth}"
        for depth in (500, 1000)
        for metric in ("recall", "graded_recall", "judged_rate")
    )
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for topic in TOPIC_IDS:
        result[topic] = {}
        for arm, control in (
            ("FACET-2B", "RRF"),
            ("TETHERED-2B", "RRF"),
            ("TETHERED-2B", "FACET-2B"),
        ):
            arm_topics = arms[arm]["per_topic"]
            control_topics = arms[control]["per_topic"]
            if not isinstance(arm_topics, Mapping) or not isinstance(control_topics, Mapping):
                raise ValueError("arm per-topic metrics are invalid")
            left, right = arm_topics[topic], control_topics[topic]
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                raise ValueError("arm topic metrics are invalid")
            result[topic][f"{arm}_vs_{control}"] = {
                field: (
                    float(left[field]) - float(right[field])
                    if left[field] is not None and right[field] is not None
                    else None
                )
                for field in fields
            }
    return result


def evaluate(freeze_dir: Path, projection: Path, output: Path) -> dict[str, object]:
    """Evaluate only the authenticated prior projection after Task 3 verification."""

    freeze_dir, projection, output = map(Path, (freeze_dir, projection, output))
    verify_freeze(freeze_dir)
    if output.exists():
        raise FileExistsError(f"create-only evaluation output already exists: {output}")
    if projection.name != "qrels_projection.jsonl" or projection.is_symlink():
        raise ValueError("only the sealed prior qrels projection is accepted")

    receipt_path = projection.parent / "qrels_access_receipt.json"
    prior_metrics_path = projection.parent / "metrics.json"
    prior_summary_path = projection.parent / "summary.json"
    receipt, receipt_bytes = _read_object_bytes(receipt_path, "qrels access receipt")
    prior_metrics, prior_metrics_bytes = _read_object_bytes(prior_metrics_path, "prior metrics")
    prior_summary, prior_summary_bytes = _read_object_bytes(prior_summary_path, "prior summary")
    if prior_summary.get("metrics_sha256") != _sha256(prior_metrics_bytes):
        raise ValueError("prior metrics SHA-256 differs from prior summary")
    if receipt.get("topic_ids") != list(TOPIC_IDS):
        raise ValueError("qrels access receipt lacks exact topics in order")
    try:
        projection_bytes = projection.read_bytes()
    except OSError as exc:
        raise ValueError("qrels projection is unreadable") from exc
    projection_sha256 = _sha256(projection_bytes)
    if receipt.get("qrels_projection_sha256") != projection_sha256:
        raise ValueError("qrels projection SHA-256 differs from sealed receipt")
    if receipt.get("qrels_projection_rows") != len(projection_bytes.splitlines()):
        raise ValueError("qrels projection row count differs from sealed receipt")
    qrels = _parse_projection(projection_bytes)

    rankings, ranking_rows, freeze_bindings, freeze_binding_bytes = _load_frozen_rankings(
        freeze_dir
    )
    novel, original = _novel_inputs(freeze_bindings, qrels)
    _validate_prior_novel_set(prior_metrics, novel)
    for topic in TOPIC_IDS:
        validate_protected_head(rrf=original[topic], arm=rankings["FACET-2B"][topic])
        validate_protected_head(rrf=original[topic], arm=rankings["TETHERED-2B"][topic])

    row_views = {
        arm: {topic: ranking_rows[topic][arm] for topic in TOPIC_IDS}
        for arm in ("FACET-2B", "TETHERED-2B")
    }
    arms = {
        "RRF": evaluate_arm(original, qrels, novel, depths=(100, 500, 1000)),
        "FACET-2B": evaluate_arm(
            rankings["FACET-2B"], qrels, novel,
            ranking_rows=row_views["FACET-2B"], depths=(100, 500, 1000),
        ),
        "TETHERED-2B": evaluate_arm(
            rankings["TETHERED-2B"], qrels, novel,
            ranking_rows=row_views["TETHERED-2B"], depths=(100, 500, 1000),
        ),
    }
    documented_shortage = any(
        int(row.get("shortage") or 0) > 0
        for topic in TOPIC_IDS
        for arm in ("FACET-2B", "TETHERED-2B")
        for row in ranking_rows[topic][arm]
    )
    aggregate = {arm: result["aggregate"] for arm, result in arms.items()}
    per_topic = {topic: {} for topic in TOPIC_IDS}
    for topic in TOPIC_IDS:
        for arm, result in arms.items():
            topics = result["per_topic"]
            if not isinstance(topics, Mapping):
                raise ValueError("arm per-topic metrics are invalid")
            per_topic[topic][arm] = topics[topic]
    decision = decide(
        {
            "top100_identity": True,
            "aggregate": aggregate,
            "per_topic": per_topic,
            "documented_basket_shortage": documented_shortage,
        }
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": sum(map(len, novel.values())),
        "arms": arms,
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "per_topic_deltas": _per_topic_deltas(arms),
        "basket_contributions": {
            arm: arms[arm]["basket_contributions"] for arm in ("FACET-2B", "TETHERED-2B")
        },
        "facet_yield": {
            arm: arms[arm]["facet_yield"] for arm in ("FACET-2B", "TETHERED-2B")
        },
        "novel_relevant_ids": {topic: sorted(novel[topic]) for topic in TOPIC_IDS},
        "documented_basket_shortage": documented_shortage,
    }
    seal_path = freeze_dir / "SEALED.json"
    try:
        seal_bytes = seal_path.read_bytes()
    except OSError as exc:
        raise ValueError("Task 3 seal is unreadable after verification") from exc
    try:
        seal = json.loads(seal_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Task 3 seal is invalid after verification") from exc
    if (
        not isinstance(seal, Mapping)
        or not isinstance(seal.get("root_sha256"), str)
        or len(seal["root_sha256"]) != 64
    ):
        raise ValueError("Task 3 seal root SHA-256 is invalid")
    input_bindings = {
        "schema_version": SCHEMA_VERSION,
        "task3_root_sha256": seal["root_sha256"],
        "task3_seal": _binding(seal_path, seal_bytes),
        "task3_input_bindings": _binding(freeze_dir / "input_bindings.json", freeze_binding_bytes),
        "qrels_projection": _binding(projection, projection_bytes),
        "qrels_access_receipt": _binding(receipt_path, receipt_bytes),
        "prior_metrics": _binding(prior_metrics_path, prior_metrics_bytes),
        "prior_summary": _binding(prior_summary_path, prior_summary_bytes),
        "original_qrels_opened": False,
    }
    payloads = {
        "metrics.json": _pretty_bytes(metrics),
        "diagnostics.json": _pretty_bytes(diagnostics),
        "decision.json": _pretty_bytes(decision),
        "input_bindings.json": _pretty_bytes(input_bindings),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "label": decision["label"],
        "topic_ids": list(TOPIC_IDS),
        "post_qrels_diagnostic": True,
        "production_validation": False,
        "no_new_retrieval": True,
        "original_qrels_opened": False,
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in payloads.items()
        },
    }
    payloads["summary.json"] = _pretty_bytes(summary)
    output.mkdir(parents=True)
    for name, content in payloads.items():
        _exclusive_bytes(output / name, content)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(args.freeze, args.projection, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

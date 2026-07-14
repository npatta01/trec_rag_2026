"""One-way qrels evaluation for the sealed deep-facet candidate pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .deep_facet_candidate_manifest import TOPIC_IDS, analyze_terms
from .deep_facet_candidate_rank import ARM_NAMES, PREFIX_DEPTHS, _jaccard, verify_seal


SCHEMA_VERSION = "deep-facet-candidate-evaluation-v1"
DEFAULT_QRELS = Path(
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
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


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _iter_jsonl(path: Path, label: str):
    try:
        source = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    with source:
        for number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label}:{number} must be an object")
            yield row


def _gain(grade: int) -> int:
    return (2**grade - 1) if grade >= 2 else 0


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def evaluate_set(document_ids: Sequence[str], qrels: Mapping[str, int]) -> dict[str, object]:
    unique = list(dict.fromkeys(map(str, document_ids)))
    relevant = {docid for docid, grade in qrels.items() if int(grade) >= 2}
    retrieved_relevant = relevant.intersection(unique)
    total_gain = sum(_gain(int(qrels[docid])) for docid in relevant)
    retrieved_gain = sum(_gain(int(qrels[docid])) for docid in retrieved_relevant)
    judged = sum(docid in qrels for docid in unique)
    return {
        "candidate_count": len(unique),
        "relevant_count": len(retrieved_relevant),
        "total_relevant": len(relevant),
        "recall": _ratio(len(retrieved_relevant), len(relevant)),
        "retrieved_gain": retrieved_gain,
        "total_gain": total_gain,
        "graded_recall": _ratio(retrieved_gain, total_gain),
        "judged_count": judged,
        "judged_rate": _ratio(judged, len(unique)),
    }


def _ndcg(ranking: Sequence[str], qrels: Mapping[str, int], depth: int) -> float | None:
    gains = [_gain(int(qrels.get(docid, 0))) for docid in ranking[:depth]]
    dcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_gains = sorted((_gain(int(value)) for value in qrels.values()), reverse=True)[:depth]
    ideal = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / ideal if ideal else None


def evaluate_ranking(
    ranking: Sequence[str], qrels: Mapping[str, int], *, depths: Sequence[int] = PREFIX_DEPTHS
) -> dict[str, object]:
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate document IDs")
    result: dict[str, object] = {
        "ndcg@10": _ndcg(ranking, qrels, 10),
        "ndcg@100": _ndcg(ranking, qrels, 100),
    }
    for depth in depths:
        result[str(depth)] = evaluate_set(ranking[:depth], qrels)
    return result


def create_qrels_sentinel(experiment_root: Path, *, seal_sha256: str) -> dict[str, object]:
    if len(seal_sha256) != 64:
        raise ValueError("seal SHA-256 is invalid")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "qrels_access_boundary_crossed",
        "qrels_opened": True,
        "seal_sha256": seal_sha256,
        "topic_ids": list(TOPIC_IDS),
        "upstream_mutation_forbidden": True,
    }
    _exclusive_bytes(Path(experiment_root) / "QRELS_ACCESSED", _pretty_bytes(payload))
    return payload


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"qrels are unreadable: {path}") from exc
    for number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"qrels line {number} must have four fields")
        topic_id, _iteration, document_id, raw_grade = parts
        if topic_id not in result:
            continue
        grade = int(raw_grade)
        if document_id in result[topic_id]:
            raise ValueError("qrels projection has duplicate topic-document identities")
        result[topic_id][document_id] = grade
    if any(not result[topic] for topic in TOPIC_IDS):
        raise ValueError("qrels projection lacks one or more frozen topics")
    return result


def _load_unions(gate_dir: Path) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, str]],
    dict[str, set[str]], dict[str, dict[str, list[str]]],
]:
    raw: dict[str, list[str]] = {topic: [] for topic in TOPIC_IDS}
    accepted: dict[str, list[str]] = {topic: [] for topic in TOPIC_IDS}
    texts: dict[str, dict[str, str]] = {topic: {} for topic in TOPIC_IDS}
    original: dict[str, set[str]] = {topic: set() for topic in TOPIC_IDS}
    provenance: dict[str, dict[str, list[str]]] = {topic: {} for topic in TOPIC_IDS}
    for kind, target in (("raw", raw), ("accepted", accepted)):
        for row in _iter_jsonl(gate_dir / f"u_{kind}.jsonl", f"U_{kind}"):
            topic_id = str(row.get("topic_id"))
            if topic_id not in target:
                raise ValueError("union contains an unexpected topic")
            document_id = str(row.get("document_id"))
            target[topic_id].append(document_id)
            texts[topic_id][document_id] = str(row.get("text"))
            sources = row.get("provenance")
            if not isinstance(sources, list):
                raise ValueError("union provenance must be an array")
            facets: list[str] = []
            for source in sources:
                if not isinstance(source, Mapping):
                    raise ValueError("union provenance entry must be an object")
                if source.get("family") == "original":
                    original[topic_id].add(document_id)
                elif source.get("family") == "facet" and source.get("accepted") is True:
                    facets.append(str(source.get("facet_id")))
            if kind == "accepted":
                provenance[topic_id][document_id] = sorted(set(facets))
    return raw, accepted, texts, original, provenance


def _load_rankings(freeze_dir: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[tuple[int, str]]]] = {
        topic: {arm: [] for arm in ARM_NAMES} for topic in TOPIC_IDS
    }
    for row in _iter_jsonl(freeze_dir / "rankings.jsonl", "rankings"):
        topic_id, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic_id not in result or arm not in result[topic_id]:
            raise ValueError("ranking contains an unexpected topic or arm")
        result[topic_id][arm].append((int(row["rank"]), str(row["document_id"])))
    ordered: dict[str, dict[str, list[str]]] = {}
    for topic_id in TOPIC_IDS:
        ordered[topic_id] = {}
        for arm in ARM_NAMES:
            rows = sorted(result[topic_id][arm])
            ids = [document_id for rank, document_id in rows]
            if [rank for rank, _document_id in rows] != list(range(1, len(rows) + 1)) or len(ids) != len(set(ids)):
                raise ValueError("ranking is not a contiguous permutation")
            ordered[topic_id][arm] = ids
    return ordered


def _macro(values: Sequence[float | None]) -> dict[str, object]:
    present = [float(value) for value in values if value is not None]
    return {
        "value": math.fsum(present) / len(present) if present else None,
        "included_topics": len(present),
        "excluded_topics": len(values) - len(present),
    }


def _duplicate_rates_by_depth(
    ranking: Sequence[str],
    texts: Mapping[str, str],
    token_cache: Mapping[str, frozenset[str]],
    depths: Sequence[int],
) -> dict[int, tuple[float | None, float | None]]:
    maximum = min(max(depths), len(ranking))
    if maximum == 0:
        return {depth: (None, None) for depth in depths}
    hashes: set[str] = set()
    exact = 0
    token_sets: list[frozenset[str]] = []
    near = 0
    result: dict[int, tuple[float | None, float | None]] = {}
    for rank, document_id in enumerate(ranking[:maximum], start=1):
        text = texts[document_id]
        digest = _sha256(text.encode("utf-8"))
        if digest in hashes:
            exact += 1
        hashes.add(digest)
        tokens = token_cache[document_id]
        if any(_jaccard(tokens, prior) >= 0.80 for prior in token_sets):
            near += 1
        token_sets.append(tokens)
        if rank in depths:
            result[rank] = (exact / rank, near / rank)
    for depth in depths:
        effective = min(depth, maximum)
        result.setdefault(depth, result.get(effective, (exact / maximum, near / maximum)))
    return result


def _macro_value(per_topic: Mapping[str, Mapping[str, object]], arm: str, field: str) -> float | None:
    return _macro([per_topic[topic][arm].get(field) for topic in TOPIC_IDS])["value"]  # type: ignore[return-value]


def _leave_one_out(per_topic: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for omitted in TOPIC_IDS:
        kept = [topic for topic in TOPIC_IDS if topic != omitted]
        values: dict[str, dict[str, float | None]] = {}
        for arm in ("RRF", "GLOBAL", "DUAL"):
            values[arm] = {
                "graded_recall@500": _macro([per_topic[t][arm].get("graded_recall@500") for t in kept])["value"],  # type: ignore[dict-item]
                "ndcg@10": _macro([per_topic[t][arm].get("ndcg@10") for t in kept])["value"],  # type: ignore[dict-item]
            }
        dual_gr = values["DUAL"]["graded_recall@500"]
        rrf_gr = values["RRF"]["graded_recall@500"]
        global_gr = values["GLOBAL"]["graded_recall@500"]
        dual_ndcg = values["DUAL"]["ndcg@10"]
        rrf_ndcg = values["RRF"]["ndcg@10"]
        result[omitted] = {
            "metrics": values,
            "dual_graded_recall_500_beats_rrf": dual_gr is not None and rrf_gr is not None and dual_gr > rrf_gr,
            "dual_graded_recall_500_beats_global": dual_gr is not None and global_gr is not None and dual_gr > global_gr,
            "dual_ndcg_10_within_002_rrf": dual_ndcg is not None and rrf_ndcg is not None and dual_ndcg >= rrf_ndcg - 0.02,
        }
    return result


def decide(evidence: Mapping[str, object]) -> dict[str, object]:
    aggregate = evidence["aggregate"]
    per_topic = evidence["per_topic"]
    leave_one_out = evidence["leave_one_out"]
    assert isinstance(aggregate, Mapping) and isinstance(per_topic, Mapping) and isinstance(leave_one_out, Mapping)
    rrf = aggregate["RRF"]
    global_arm = aggregate["GLOBAL"]
    dual = aggregate["DUAL"]
    assert isinstance(rrf, Mapping) and isinstance(global_arm, Mapping) and isinstance(dual, Mapping)
    guards = {
        "novel_discovery": int(evidence.get("novel_relevant_count", 0)) >= 5 and int(evidence.get("novel_relevant_topic_count", 0)) >= 2,
        "dual_graded_recall_500_beats_rrf": float(dual.get("graded_recall@500", -1)) > float(rrf.get("graded_recall@500", -1)),
        "dual_graded_recall_500_beats_global": float(dual.get("graded_recall@500", -1)) > float(global_arm.get("graded_recall@500", -1)),
        "dual_graded_recall_1000_not_below_rrf": float(dual.get("graded_recall@1000", -1)) >= float(rrf.get("graded_recall@1000", 0)),
        "dual_novel_retention_1000": float(dual.get("novel_retention@1000", 0)) >= 0.50,
        "dual_ndcg_10_within_002_rrf": float(dual.get("ndcg@10", -1)) >= float(rrf.get("ndcg@10", 0)) - 0.02,
        "per_topic_ndcg_floor": all(
            isinstance(row, Mapping) and float(row.get("DUAL_ndcg@10_delta_vs_RRF", -1)) >= -0.10
            for row in per_topic.values()
        ),
        "leave_one_out_stability": len(leave_one_out) == len(TOPIC_IDS) and all(
            isinstance(row, Mapping)
            and row.get("dual_graded_recall_500_beats_rrf") is True
            and row.get("dual_graded_recall_500_beats_global") is True
            and row.get("dual_ndcg_10_within_002_rrf") is True
            for row in leave_one_out.values()
        ),
    }
    failed = [name for name, passed in guards.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "advance_to_larger_validation": not failed,
        "production_promotion_authorized": False,
        "guards": guards,
        "failed_guards": failed,
        "aggregation": {
            "graded_recall": "macro_non_null",
            "ndcg": "macro_non_null",
            "novel_retention": "micro_topic_document",
        },
    }


def _evaluate_payload(
    freeze_dir: Path, qrels: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, object], dict[str, object]]:
    gate_dir = freeze_dir.parent / "gate_v1"
    raw, accepted, texts, original, provenance = _load_unions(gate_dir)
    rankings = _load_rankings(freeze_dir)
    prefix_unions = _read_json(gate_dir / "prefix_unions.json", "prefix unions")
    discovery: dict[str, object] = {}
    novel: dict[str, set[str]] = {}
    gate_loss: dict[str, set[str]] = {}
    attribution: dict[str, object] = {}
    for topic_id in TOPIC_IDS:
        topic_qrels = qrels[topic_id]
        relevant = {docid for docid, grade in topic_qrels.items() if int(grade) >= 2}
        novel[topic_id] = relevant.intersection(accepted[topic_id]).difference(original[topic_id])
        gate_loss[topic_id] = relevant.intersection(raw[topic_id]).difference(accepted[topic_id])
        prefix_metrics = {
            name: evaluate_set(ids, topic_qrels)
            for name, ids in prefix_unions[topic_id].items()  # type: ignore[union-attr]
        }
        discovery[topic_id] = {
            "original": evaluate_set(list(original[topic_id]), topic_qrels),
            "U_raw": evaluate_set(raw[topic_id], topic_qrels),
            "U_accepted": evaluate_set(accepted[topic_id], topic_qrels),
            "prefix_unions": prefix_metrics,
            "novel_relevant_ids": sorted(novel[topic_id]),
            "gate_lost_relevant_ids": sorted(gate_loss[topic_id]),
        }
        inclusive: dict[str, int] = defaultdict(int)
        exclusive: dict[str, int] = defaultdict(int)
        for document_id in novel[topic_id]:
            facets = provenance[topic_id].get(document_id, [])
            for facet_id in facets:
                inclusive[facet_id] += 1
            if len(facets) == 1:
                exclusive[facets[0]] += 1
        attribution[topic_id] = {"inclusive": dict(sorted(inclusive.items())), "exclusive": dict(sorted(exclusive.items()))}

    per_topic: dict[str, dict[str, object]] = {topic: {} for topic in TOPIC_IDS}
    for topic_id in TOPIC_IDS:
        token_cache = {
            document_id: frozenset(analyze_terms(text))
            for document_id, text in texts[topic_id].items()
        }
        for arm in ARM_NAMES:
            metrics = evaluate_ranking(rankings[topic_id][arm], qrels[topic_id])
            duplicate_rates = _duplicate_rates_by_depth(
                rankings[topic_id][arm], texts[topic_id], token_cache, PREFIX_DEPTHS
            )
            row: dict[str, object] = {
                "ndcg@10": metrics["ndcg@10"],
                "ndcg@100": metrics["ndcg@100"],
            }
            for depth in PREFIX_DEPTHS:
                depth_metrics = metrics[str(depth)]
                assert isinstance(depth_metrics, Mapping)
                for field in ("recall", "graded_recall", "judged_rate", "relevant_count"):
                    row[f"{field}@{depth}"] = depth_metrics[field]
                retained = len(novel[topic_id].intersection(rankings[topic_id][arm][:depth]))
                row[f"novel_retained@{depth}"] = retained
                row[f"novel_retention@{depth}"] = _ratio(retained, len(novel[topic_id]))
                exact, near = duplicate_rates[depth]
                row[f"exact_duplicate_rate@{depth}"] = exact
                row[f"near_duplicate_rate@{depth}"] = near
            per_topic[topic_id][arm] = row
        dual_ndcg = per_topic[topic_id]["DUAL"]["ndcg@10"]
        rrf_ndcg = per_topic[topic_id]["RRF"]["ndcg@10"]
        per_topic[topic_id]["DUAL_ndcg@10_delta_vs_RRF"] = (
            float(dual_ndcg) - float(rrf_ndcg) if dual_ndcg is not None and rrf_ndcg is not None else None
        )

    aggregate: dict[str, dict[str, object]] = {}
    novel_total = sum(len(value) for value in novel.values())
    for arm in ARM_NAMES:
        row: dict[str, object] = {}
        for field in ("ndcg@10", "ndcg@100"):
            macro = _macro([per_topic[t][arm][field] for t in TOPIC_IDS])  # type: ignore[list-item]
            row[field] = macro["value"]
            row[f"{field}_aggregation"] = macro
        for depth in PREFIX_DEPTHS:
            for field in ("recall", "graded_recall", "judged_rate", "exact_duplicate_rate", "near_duplicate_rate"):
                key = f"{field}@{depth}"
                macro = _macro([per_topic[t][arm][key] for t in TOPIC_IDS])  # type: ignore[list-item]
                row[key] = macro["value"]
                row[f"{key}_aggregation"] = macro
            retained = sum(int(per_topic[t][arm][f"novel_retained@{depth}"]) for t in TOPIC_IDS)
            row[f"novel_retained@{depth}"] = retained
            row[f"novel_retention@{depth}"] = _ratio(retained, novel_total)
        aggregate[arm] = row
    comparison: dict[str, object] = {
        "novel_relevant_count": novel_total,
        "novel_relevant_topic_count": sum(bool(value) for value in novel.values()),
        "gate_lost_relevant_count": sum(len(value) for value in gate_loss.values()),
        "aggregate": aggregate,
        "per_topic": {
            topic: {
                **{arm: per_topic[topic][arm] for arm in ARM_NAMES},
                "DUAL_ndcg@10_delta_vs_RRF": per_topic[topic]["DUAL_ndcg@10_delta_vs_RRF"],
            }
            for topic in TOPIC_IDS
        },
    }
    comparison["leave_one_out"] = _leave_one_out(comparison["per_topic"])  # type: ignore[arg-type]
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(TOPIC_IDS),
        "relevance_threshold": 2,
        "discovery": discovery,
        "novel_attribution": attribution,
        **comparison,
    }
    decision = decide(evidence)
    if novel_total == 0 and sum(len(relevant.intersection(raw[t]).difference(original[t])) for t, relevant in ((topic, {docid for docid, grade in qrels[topic].items() if grade >= 2}) for topic in TOPIC_IDS)) == 0:
        diagnosis = "retrieval"
    elif sum(len(value) for value in gate_loss.values()) and novel_total == 0:
        diagnosis = "gating"
    elif not decision["advance_to_larger_validation"]:
        diagnosis = "fusion"
    else:
        diagnosis = "advance"
    decision["diagnosis"] = diagnosis
    return evidence, decision


def evaluate(
    freeze_dir: Path,
    output: Path,
    *,
    qrels_loader: Callable[[], Mapping[str, Mapping[str, int]]] | None = None,
    qrels_path: Path = DEFAULT_QRELS,
) -> dict[str, object]:
    freeze_dir, output = Path(freeze_dir), Path(output)
    try:
        seal = verify_seal(freeze_dir)
    except (OSError, ValueError) as exc:
        raise ValueError("valid pre-qrels seal is required") from exc
    if output.exists():
        raise FileExistsError(f"create-only evaluation output already exists: {output}")
    sentinel_path = freeze_dir.parent / "QRELS_ACCESSED"
    if sentinel_path.exists():
        raise FileExistsError("qrels access sentinel already exists")
    seal_bytes = (freeze_dir / "SEALED.json").read_bytes()
    sentinel = create_qrels_sentinel(freeze_dir.parent, seal_sha256=_sha256(seal_bytes))
    qrels = qrels_loader() if qrels_loader is not None else _load_qrels(Path(qrels_path))
    if list(qrels) != list(TOPIC_IDS) or any(not qrels[topic] for topic in TOPIC_IDS):
        raise ValueError("qrels loader must return exactly the frozen topics in order")
    output.mkdir(parents=True)
    projection_rows = [
        {"topic_id": topic, "document_id": document_id, "grade": int(grade)}
        for topic in TOPIC_IDS
        for document_id, grade in sorted(qrels[topic].items())
    ]
    projection_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in projection_rows)
    _exclusive_bytes(output / "qrels_projection.jsonl", projection_bytes)
    access_receipt = {
        **sentinel,
        "seal_root_sha256": seal["root_sha256"],
        "qrels_source_name": Path(qrels_path).name if qrels_loader is None else "injected_test_loader",
        "qrels_projection_rows": len(projection_rows),
        "qrels_projection_sha256": _sha256(projection_bytes),
        "evaluator_code_sha256": _sha256(Path(__file__).read_bytes()),
    }
    _exclusive_bytes(output / "qrels_access_receipt.json", _pretty_bytes(access_receipt))
    evidence, decision = _evaluate_payload(freeze_dir, qrels)
    _exclusive_bytes(output / "metrics.json", _pretty_bytes(evidence))
    _exclusive_bytes(output / "decision.json", _pretty_bytes(decision))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "qrels_opened": True,
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": evidence["novel_relevant_count"],
        "gate_lost_relevant_count": evidence["gate_lost_relevant_count"],
        "diagnosis": decision["diagnosis"],
        "advance_to_larger_validation": decision["advance_to_larger_validation"],
        "failed_guards": decision["failed_guards"],
        "metrics_sha256": _sha256((output / "metrics.json").read_bytes()),
        "decision_sha256": _sha256((output / "decision.json").read_bytes()),
    }
    _exclusive_bytes(output / "summary.json", _pretty_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    args = parser.parse_args(argv)
    result = evaluate(args.freeze, args.output, qrels_path=args.qrels)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the offline, source-bound tethered-facet diagnostic report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import tethered_facet_evaluate as task4_evaluate


TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
ARMS = ("RRF", "FACET-2B", "TETHERED-2B")
DEPTHS = (500, 1000)
OUTPUT_FILES = ("artifact.json", "summary.json", "report_data.sqlite", "report.html")
SCHEMA_VERSION = "tethered-facet-diagnostic-report-v1"


@dataclass
class ReportSources:
    """Exact authenticated Task 1--4 inputs; paths are not emitted in the report."""

    topic_ids: list[str]
    task1_receipt: Path
    task2_receipt: Path
    task3_freeze: Path
    task4_evaluation: Path


@dataclass(frozen=True)
class BuiltReport:
    output_dir: Path
    artifact: dict[str, object]
    summary: dict[str, object]
    html: str
    artifact_bytes: bytes
    summary_bytes: bytes
    html_bytes: bytes


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _read_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        content = Path(path).read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, content


def _require_hash(actual_content: bytes, expected: object, label: str) -> None:
    if isinstance(expected, Mapping):
        expected = expected.get("sha256")
    if not isinstance(expected, str) or _sha256(actual_content) != expected:
        raise ValueError(f"{label} SHA-256 differs from its authenticated binding")


def _exact_binding(raw: object, path: Path, content: bytes, label: str) -> None:
    expected = {"path": str(path.resolve()), "bytes": len(content), "sha256": _sha256(content)}
    if raw != expected:
        raise ValueError(f"{label} exact source binding differs")


def _validate_topics(value: object, label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} topic IDs are invalid")
    topics = [str(topic) for topic in value]
    for topic in topics:
        if topic in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic} is forbidden")
    if topics != list(TOPIC_IDS):
        raise ValueError(f"{label} must contain the exact pilot topics in order")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _jsonl_rows(content: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} rows must be objects")
        rows.append(row)
    return rows


def _percentiles(scores: Mapping[str, object]) -> dict[str, float]:
    grouped: dict[float, list[str]] = defaultdict(list)
    for document, raw in scores.items():
        grouped[float(raw)].append(str(document))
    count = len(scores)
    output: dict[str, float] = {}
    first = 1
    for score in sorted(grouped, reverse=True):
        last = first + len(grouped[score]) - 1
        value = (count - ((first + last) / 2) + 1) / count
        for document in grouped[score]:
            output[document] = value
        first = last + 1
    return output


def _selected_windows(windows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    ordered = sorted(
        windows,
        key=lambda row: (-float(row["score"]), int(row["document_start_token"]), str(row["window_id"])),
    )
    selected: list[Mapping[str, object]] = []
    covered: list[tuple[int, int]] = []
    for row in ordered:
        start, end = int(row["document_start_token"]), int(row["document_end_token"])
        if start < 0 or end <= start:
            raise ValueError("representative scored window span is invalid")
        overlap = sum(
            max(0, min(end, right) - max(start, left)) for left, right in covered
        )
        if selected and end - start - overlap < 128:
            continue
        selected.append(row)
        covered.append((start, end))
        if len(selected) == 4:
            break
    return selected


def _bound_bytes(raw: object, label: str) -> tuple[Path, bytes]:
    binding = _mapping(raw, label)
    path = Path(str(binding.get("path")))
    if path.is_symlink():
        raise ValueError(f"{label} is unsafe")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    _exact_binding(binding, path, content, label)
    return path, content


def _authenticated_qrels_evidence(
    bindings: Mapping[str, object],
) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    """Verify the Task 4 historical leaf and derive grades/novel IDs from it."""

    anchor = task4_evaluate.HISTORICAL_PRIOR_EVALUATION_IDENTITY
    if bindings.get("historical_integrity_anchor") != anchor:
        raise ValueError("Task 4 historical qrels integrity anchor differs")
    if (
        bindings.get("historical_integrity_only") is not True
        or bindings.get("blind_generalization_evidence") is not False
        or bindings.get("original_qrels_opened") is not False
    ):
        raise ValueError("Task 4 historical qrels boundary differs")
    _prior_seal_path, prior_seal_bytes = _bound_bytes(
        bindings.get("prior_freeze_seal"), "Task 4 prior freeze seal"
    )
    _prior_rankings_path, _prior_rankings_bytes = _bound_bytes(
        bindings.get("prior_freeze_rankings"), "Task 4 prior freeze rankings"
    )
    try:
        prior_seal = json.loads(prior_seal_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Task 4 prior freeze seal is invalid") from exc
    if (
        not isinstance(prior_seal, Mapping)
        or bindings.get("prior_freeze_root_sha256") != prior_seal.get("root_sha256")
    ):
        raise ValueError("Task 4 prior freeze root differs")

    source_names = {
        "qrels_projection": "qrels_projection.jsonl",
        "qrels_access_receipt": "qrels_access_receipt.json",
        "prior_metrics": "metrics.json",
        "prior_decision": "decision.json",
        "prior_summary": "summary.json",
    }
    contents: dict[str, bytes] = {}
    for binding_name, filename in source_names.items():
        _path, content = _bound_bytes(
            bindings.get(binding_name), f"Task 4 {binding_name.replace('_', ' ')}"
        )
        contents[filename] = content
    expected_files = anchor.get("files") if isinstance(anchor, Mapping) else None
    if (
        anchor.get("schema_version") != "deep-facet-candidate-evaluation-v1"
        or anchor.get("topic_ids") != list(TOPIC_IDS)
        or not isinstance(expected_files, Mapping)
        or set(expected_files) != set(contents)
        or any(_sha256(contents[name]) != expected_files.get(name) for name in contents)
        or len(contents["qrels_projection.jsonl"].splitlines())
        != anchor.get("qrels_projection_rows")
    ):
        raise ValueError("Task 4 historical qrels integrity evidence differs")
    try:
        receipt = json.loads(contents["qrels_access_receipt.json"])
        prior_metrics = json.loads(contents["metrics.json"])
        prior_summary = json.loads(contents["summary.json"])
    except json.JSONDecodeError as exc:
        raise ValueError("Task 4 historical qrels evidence is invalid") from exc
    if not all(isinstance(value, Mapping) for value in (receipt, prior_metrics, prior_summary)):
        raise ValueError("Task 4 historical qrels evidence must be objects")
    projection_bytes = contents["qrels_projection.jsonl"]
    if (
        receipt.get("status") != "qrels_access_boundary_crossed"
        or receipt.get("qrels_opened") is not True
        or receipt.get("upstream_mutation_forbidden") is not True
        or receipt.get("topic_ids") != list(TOPIC_IDS)
        or receipt.get("seal_sha256") != _sha256(prior_seal_bytes)
        or receipt.get("seal_root_sha256") != prior_seal.get("root_sha256")
        or receipt.get("qrels_projection_sha256") != _sha256(projection_bytes)
        or receipt.get("qrels_projection_rows") != len(projection_bytes.splitlines())
        or prior_summary.get("metrics_sha256") != _sha256(contents["metrics.json"])
        or prior_summary.get("decision_sha256") != _sha256(contents["decision.json"])
    ):
        raise ValueError("Task 4 historical qrels trust chain differs")

    qrels = {topic: {} for topic in TOPIC_IDS}
    seen_topics: list[str] = []
    for row in _jsonl_rows(projection_bytes, "historical qrels projection"):
        topic, document, grade = str(row.get("topic_id")), str(row.get("document_id")), row.get("grade")
        if topic in PROTECTED_TOPIC_IDS or topic not in qrels:
            raise ValueError("historical qrels projection contains an unexpected topic")
        if type(grade) is not int or grade < 0 or grade > 3 or not document or document in qrels[topic]:
            raise ValueError("historical qrels projection row is invalid")
        if not seen_topics or seen_topics[-1] != topic:
            seen_topics.append(topic)
        qrels[topic][document] = grade
    if seen_topics != list(TOPIC_IDS):
        raise ValueError("historical qrels projection topic order differs")
    discovery = prior_metrics.get("discovery")
    if not isinstance(discovery, Mapping) or set(discovery) != set(TOPIC_IDS):
        raise ValueError("historical novel evidence is missing")
    novel: dict[str, set[str]] = {}
    for topic in TOPIC_IDS:
        record = discovery.get(topic)
        ids = record.get("novel_relevant_ids") if isinstance(record, Mapping) else None
        if not isinstance(ids, list) or len(ids) != len(set(map(str, ids))):
            raise ValueError("historical novel evidence is invalid")
        novel[topic] = set(map(str, ids))
        if any(qrels[topic].get(document, 0) < 2 for document in novel[topic]):
            raise ValueError("historical novel evidence differs from qrels grades")
    if (
        prior_metrics.get("novel_relevant_count") != sum(map(len, novel.values()))
        or prior_summary.get("novel_relevant_count") != sum(map(len, novel.values()))
    ):
        raise ValueError("historical novel count differs")
    return qrels, novel


def _verify_sources(sources: ReportSources) -> dict[str, object]:
    """Verify every upstream hash and semantic boundary before using evidence."""

    _validate_topics(sources.topic_ids, "requested")
    task1, task1_bytes = _read_object(sources.task1_receipt, "Task 1 receipt")
    task2, task2_bytes = _read_object(sources.task2_receipt, "Task 2 receipt")
    # Authenticate the upstream preflight before trusting any of its fields.
    _require_hash(task1_bytes, task2.get("preflight_sha256"), "Task 1 receipt")
    for payload, label in ((task1, "Task 1"), (task2, "Task 2")):
        if "topic_ids" in payload:
            _validate_topics(payload.get("topic_ids"), label)
    if (
        task1.get("schema_version") != "tethered-facet-minilm-preflight-v1"
        or task1.get("status") != "tokenizer_only_preflight_complete"
        or task1.get("qrels_opened", task1.get("qrels_read")) is not False
        or task1.get("retrieval_path_supported", task1.get("retrieval_performed")) is not False
    ):
        raise ValueError("Task 1 receipt is not an offline completed preflight")
    if (
        task2.get("schema_version")
        != "tethered-facet-minilm-scoring-receipt-v1"
        or task2.get("status") != "complete"
        or task2.get("qrels_opened") is not False
        or task2.get("network_access_supported") is not False
        or task2.get("hosted_inference_supported") is not False
    ):
        raise ValueError("Task 2 receipt is not a completed local-only scoring receipt")

    freeze = Path(sources.task3_freeze)
    seal, seal_bytes = _read_object(freeze / "SEALED.json", "Task 3 seal")
    if seal.get("status") not in {"sealed", "sealed_before_qrels"}:
        raise ValueError("Task 3 freeze is not sealed")
    if seal.get("schema_version") != "tethered-facet-two-basket-seal-v1":
        raise ValueError("Task 3 seal schema differs")
    sealed_files = _mapping(seal.get("files"), "Task 3 sealed files")
    if "root_sha256" in seal:
        material = {
            key: seal.get(key)
            for key in ("schema_version", "status", "qrels_opened", "files")
        }
        if seal.get("root_sha256") != _sha256(_canonical_bytes(material)):
            raise ValueError("Task 3 seal root SHA-256 differs")
    required_freeze_files = tuple(sorted(sealed_files))
    allowed_freeze_files = {
        "parameters.json",
        "input_bindings.json",
        "rankings.jsonl",
        "prefixes.json",
        "summary.json",
    }
    if not set(required_freeze_files) <= allowed_freeze_files:
        raise ValueError("Task 3 seal declares an unexpected artifact")
    if not {"input_bindings.json", "rankings.jsonl", "summary.json"} <= set(required_freeze_files):
        raise ValueError("Task 3 seal lacks required artifacts")
    freeze_bytes: dict[str, bytes] = {}
    for name in required_freeze_files:
        try:
            content = (freeze / name).read_bytes()
        except OSError as exc:
            raise ValueError(f"Task 3 {name} is unreadable") from exc
        _require_hash(content, sealed_files.get(name), f"Task 3 {name}")
        freeze_bytes[name] = content
    task3_bindings = json.loads(freeze_bytes["input_bindings.json"])
    task3_summary = json.loads(freeze_bytes["summary.json"])
    if not isinstance(task3_bindings, Mapping) or not isinstance(task3_summary, Mapping):
        raise ValueError("Task 3 authenticated metadata is invalid")
    if (
        task3_bindings.get("schema_version") != "tethered-facet-two-basket-freeze-v1"
        or task3_summary.get("schema_version") != "tethered-facet-two-basket-freeze-v1"
        or task3_summary.get("status") not in {"complete", "rankings_frozen_before_qrels"}
        or task3_summary.get("qrels_opened") is not False
    ):
        raise ValueError("Task 3 freeze is not a qrels-free completed freeze")
    _validate_topics(task3_summary.get("topic_ids"), "Task 3")
    task3_artifacts = task3_summary.get("artifacts")
    if isinstance(task3_artifacts, Mapping):
        _require_hash(freeze_bytes["rankings.jsonl"], task3_artifacts.get("rankings.jsonl"), "Task 3 rankings")
        _require_hash(freeze_bytes["input_bindings.json"], task3_artifacts.get("input_bindings.json"), "Task 3 input bindings")
    else:
        _require_hash(freeze_bytes["rankings.jsonl"], task3_summary.get("rankings_sha256"), "Task 3 rankings")
        _require_hash(freeze_bytes["input_bindings.json"], task3_summary.get("input_bindings_sha256"), "Task 3 input bindings")

    evaluation = Path(sources.task4_evaluation)
    evaluation_summary, evaluation_summary_bytes = _read_object(
        evaluation / "summary.json", "Task 4 summary"
    )
    if (
        evaluation_summary.get("status") != "complete"
        or evaluation_summary.get("post_qrels_diagnostic") is not True
        or evaluation_summary.get("production_validation") is not False
    ):
        raise ValueError("Task 4 evidence must be a completed post-qrels diagnostic")
    _validate_topics(evaluation_summary.get("topic_ids"), "Task 4")
    evaluation_payloads: dict[str, dict[str, object]] = {}
    evaluation_bytes: dict[str, bytes] = {}
    for stem in ("metrics", "diagnostics", "decision", "input_bindings"):
        payload, content = _read_object(evaluation / f"{stem}.json", f"Task 4 {stem}")
        summary_artifacts = evaluation_summary.get("artifacts")
        expected = (
            summary_artifacts.get(f"{stem}.json")
            if isinstance(summary_artifacts, Mapping)
            else evaluation_summary.get(f"{stem}_sha256")
        )
        _require_hash(content, expected, f"Task 4 {stem}")
        evaluation_payloads[stem] = payload
        evaluation_bytes[stem] = content
    for stem in ("metrics", "diagnostics"):
        if evaluation_payloads[stem].get("schema_version") != "tethered-facet-evaluation-v1":
            raise ValueError(f"Task 4 {stem} schema differs")
    evaluation_bindings = evaluation_payloads["input_bindings"]
    if evaluation_bindings.get("schema_version") != "tethered-facet-evaluation-v1":
        raise ValueError("Task 4 input binding schema differs")
    if evaluation_bindings.get("task3_root_sha256") != seal.get("root_sha256"):
        raise ValueError("Task 4 exact Task 3 root binding differs")
    authenticated_qrels, authenticated_novel = _authenticated_qrels_evidence(
        evaluation_bindings
    )
    _exact_binding(evaluation_bindings.get("task3_seal"), freeze / "SEALED.json", seal_bytes, "Task 4 Task 3 seal")
    _exact_binding(
        evaluation_bindings.get("task3_input_bindings"),
        freeze / "input_bindings.json",
        freeze_bytes["input_bindings.json"],
        "Task 4 Task 3 input bindings",
    )
    _exact_binding(
        evaluation_bindings.get("task3_rankings"),
        freeze / "rankings.jsonl",
        freeze_bytes["rankings.jsonl"],
        "Task 4 Task 3 rankings",
    )
    raw_inputs = task3_bindings.get("inputs")
    if not isinstance(raw_inputs, Mapping) or evaluation_bindings.get("task3_producer_sources") != raw_inputs:
        raise ValueError("Task 4 exact producer source bindings differ from Task 3")
    producer_names = (
        "facet_candidates", "facet_window_scores", "tethered_candidates",
        "tethered_window_scores", "tethered_document_scores",
        "tethered_preflight", "tethered_scoring_receipt",
    )
    producer_bytes: dict[str, bytes] = {}
    producer_hashes: dict[str, str] = {}
    for name in producer_names:
        raw = raw_inputs.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Task 3 exact producer binding is missing: {name}")
        path = Path(str(raw.get("path")))
        if path.is_symlink():
            raise ValueError(f"Task 3 producer source is unsafe: {name}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Task 3 producer source is unreadable: {name}") from exc
        _exact_binding(raw, path, content, f"Task 3 {name}")
        producer_bytes[name] = content
        producer_hashes[name] = _sha256(content)
    _exact_binding(raw_inputs["tethered_preflight"], Path(sources.task1_receipt), task1_bytes, "Task 1 direct receipt")
    _exact_binding(raw_inputs["tethered_scoring_receipt"], Path(sources.task2_receipt), task2_bytes, "Task 2 direct receipt")
    _validate_topics(evaluation_payloads["metrics"].get("topic_ids"), "Task 4 metrics")
    if "topic_ids" in evaluation_payloads["diagnostics"]:
        _validate_topics(evaluation_payloads["diagnostics"].get("topic_ids"), "Task 4 diagnostics")

    representative_source_hashes: dict[str, str] = {}
    for name in (
        "facet_candidates", "facet_window_scores", "tethered_candidates",
        "tethered_window_scores", "tethered_document_scores",
    ):
        raw = raw_inputs.get(name)
        digest = raw.get("sha256") if isinstance(raw, Mapping) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Task 3 representative input binding is invalid: {name}")
        representative_source_hashes[name] = digest
    ranking_rows: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for line in freeze_bytes["rankings.jsonl"].splitlines():
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("Task 3 ranking row is invalid")
        key = (str(row.get("topic_id")), str(row.get("document_id")), str(row.get("arm")))
        if key in ranking_rows:
            raise ValueError("Task 3 ranking identity is duplicated")
        ranking_rows[key] = row

    return {
        "task1": task1,
        "task2": task2,
        "task3_summary": dict(task3_summary),
        "metrics": evaluation_payloads["metrics"],
        "diagnostics": evaluation_payloads["diagnostics"],
        "decision": evaluation_payloads["decision"],
        "task4_summary": dict(evaluation_summary),
        "producer_bytes": producer_bytes,
        "task3_bindings": dict(task3_bindings),
        "authenticated_qrels": authenticated_qrels,
        "authenticated_novel": authenticated_novel,
        "task3_rankings_sha256": _sha256(freeze_bytes["rankings.jsonl"]),
        "task3_ranking_rows": ranking_rows,
        "representative_source_hashes": representative_source_hashes,
        "source_hashes": {
            "task1_receipt_sha256": _sha256(task1_bytes),
            "task2_receipt_sha256": _sha256(task2_bytes),
            "task3_seal_sha256": _sha256(seal_bytes),
            "task4_summary_sha256": _sha256(evaluation_summary_bytes),
            "task4_metrics_sha256": _sha256(evaluation_bytes["metrics"]),
            "task4_diagnostics_sha256": _sha256(evaluation_bytes["diagnostics"]),
            "task4_decision_sha256": _sha256(evaluation_bytes["decision"]),
        },
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _metric_rows(aggregate: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        values = _mapping(aggregate.get(arm), f"{arm} metrics")
        for depth in DEPTHS:
            rows.append(
                {
                    "arm": arm,
                    "depth": depth,
                    "binary_recall": _number(values.get(f"recall@{depth}"), f"{arm} recall@{depth}"),
                    "graded_recall": _number(
                        values.get(f"graded_recall@{depth}"), f"{arm} graded_recall@{depth}"
                    ),
                    "novel_retained": int(
                        _number(values.get(f"novel_retained@{depth}"), f"{arm} novel@{depth}")
                    ),
                    "judged_rate": (
                        _number(values["judged_rate@500"], f"{arm} judged rate")
                        if depth == 500 and "judged_rate@500" in values
                        else None
                    ),
                }
            )
    return rows


def _contribution_rows(diagnostics: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_topic = _mapping(diagnostics.get("per_topic_deltas"), "per-topic deltas")
    if set(per_topic) != set(TOPIC_IDS):
        raise ValueError("per-topic deltas must contain exact topics in order")
    topic_rows = []
    for topic in TOPIC_IDS:
        row = _mapping(per_topic[topic], f"topic {topic} deltas")
        if "recall@500_delta_tethered_vs_facet" in row:
            binary_delta = row["recall@500_delta_tethered_vs_facet"]
            graded_delta = row["graded_recall@500_delta_tethered_vs_facet"]
        else:
            comparison = _mapping(
                row.get("TETHERED-2B_vs_FACET-2B"),
                f"topic {topic} tethered comparison",
            )
            binary_delta = comparison.get("recall@500")
            graded_delta = comparison.get("graded_recall@500")
        topic_rows.append(
            {
                "topic_id": topic,
                "recall@500_delta_tethered_vs_facet": _number(
                    binary_delta, f"topic {topic} recall delta"
                ),
                "graded_recall@500_delta_tethered_vs_facet": _number(
                    graded_delta, f"topic {topic} graded recall delta"
                ),
            }
        )

    facet_yield = _mapping(diagnostics.get("facet_yield"), "facet yield")
    facet_rows: list[dict[str, object]] = []
    for arm in ("FACET-2B", "TETHERED-2B"):
        by_facet = _mapping(facet_yield.get(arm), f"{arm} facet yield")
        for facet_id in sorted(by_facet):
            values = _mapping(by_facet[facet_id], f"facet {facet_id}")
            selected = int(_number(values.get("selected_count"), "facet selected count"))
            relevant = int(_number(values.get("relevant_count"), "facet relevant count"))
            facet_rows.append(
                {
                    "arm": arm,
                    "facet_id": facet_id,
                    "selected_count": selected,
                    "relevant_count": relevant,
                    "relevant_yield": relevant / selected if selected else None,
                }
            )
    if not facet_rows:
        raise ValueError("facet contribution evidence is empty")
    return topic_rows, facet_rows


def _representatives(
    diagnostics: Mapping[str, object],
    *,
    task3_rankings_sha256: str,
    task3_ranking_rows: Mapping[tuple[str, str, str], Mapping[str, object]],
    source_hashes: Mapping[str, str],
    producer_bytes: Mapping[str, bytes],
    task3_bindings: Mapping[str, object],
    authenticated_qrels: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    raw = diagnostics.get("representatives")
    if not isinstance(raw, list) or not raw:
        raise ValueError("bounded representative evidence is missing")
    required = {
        "topic_id",
        "facet_id",
        "movement",
        "document_id",
        "narrative",
        "facet_query",
        "selected_passage",
        "facet_only_percentile",
        "tethered_percentile",
        "qrels_grade",
        "facet_only_final_rank",
        "tethered_final_rank",
        "prior_bm25_rank",
        "passage_provenance",
        "ranking_provenance",
    }
    rows: list[dict[str, object]] = []
    movements: set[str] = set()
    candidate_maps: dict[str, dict[tuple[str, str, str], dict[str, object]]] = {}
    for name in ("facet_candidates", "tethered_candidates"):
        parsed = _jsonl_rows(producer_bytes[name], name)
        candidate_maps[name] = {
            (str(item.get("topic_id")), str(item.get("facet_id", item.get("variant"))), str(item.get("document_id"))): item
            for item in parsed
        }
        if len(candidate_maps[name]) != len(parsed):
            raise ValueError(f"{name} contains duplicate identities")
    window_maps: dict[str, dict[tuple[str, str, str], list[dict[str, object]]]] = {}
    for name in ("facet_window_scores", "tethered_window_scores"):
        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for item in _jsonl_rows(producer_bytes[name], name):
            grouped[(str(item.get("topic_id")), str(item.get("facet_id", item.get("variant"))), str(item.get("document_id")))].append(item)
        window_maps[name] = grouped
    document_scores = {
        (str(item.get("topic_id")), str(item.get("facet_id")), str(item.get("document_id"))): item
        for item in _jsonl_rows(producer_bytes["tethered_document_scores"], "tethered document scores")
    }
    raw_topic_inputs = task3_bindings.get("topic_inputs")
    if not isinstance(raw_topic_inputs, Mapping):
        raise ValueError("Task 3 semantic inputs are missing")
    semantic: dict[tuple[str, str, str], tuple[dict[str, float], Mapping[str, object]]] = {}
    for arm in ("FACET-2B", "TETHERED-2B"):
        arm_inputs = raw_topic_inputs.get(arm)
        if not isinstance(arm_inputs, Mapping):
            raise ValueError("Task 3 semantic arm inputs are missing")
        for topic, topic_input in arm_inputs.items():
            facets = topic_input.get("facets") if isinstance(topic_input, Mapping) else None
            if not isinstance(facets, list):
                raise ValueError("Task 3 semantic facets are missing")
            for facet in facets:
                scores = facet.get("scores") if isinstance(facet, Mapping) else None
                if not isinstance(scores, Mapping):
                    raise ValueError("Task 3 semantic scores are missing")
                semantic[(arm, str(topic), str(facet.get("facet_id")))] = (_percentiles(scores), facet)
    for value in raw:
        row = _mapping(value, "representative row")
        if not required <= set(row):
            raise ValueError("representative row lacks required bounded evidence")
        topic = str(row["topic_id"])
        if topic in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic} is forbidden")
        if topic not in TOPIC_IDS:
            raise ValueError(f"representative contains unexpected topic {topic}")
        movement = str(row["movement"])
        if movement not in {"promoted", "demoted"}:
            raise ValueError("representative movement must be promoted or demoted")
        for field in ("narrative", "facet_query", "selected_passage", "document_id", "facet_id"):
            if not str(row[field]).strip():
                raise ValueError(f"representative {field} is blank")
        normalized = {field: row[field] for field in sorted(required)}
        normalized["facet_only_percentile"] = _number(
            row["facet_only_percentile"], "facet-only percentile"
        )
        normalized["tethered_percentile"] = _number(
            row["tethered_percentile"], "tethered percentile"
        )
        normalized["qrels_grade"] = int(_number(row["qrels_grade"], "qrels grade"))
        for field in ("facet_only_final_rank", "tethered_final_rank", "prior_bm25_rank"):
            normalized[field] = int(_number(row[field], field.replace("_", " ")))
            if normalized[field] <= 0:
                raise ValueError(f"representative {field} must be positive")
        passage = _mapping(row["passage_provenance"], "passage provenance")
        ranking = _mapping(row["ranking_provenance"], "ranking provenance")
        passage_required = {
            "candidate_source_sha256", "window_score_source_sha256",
            "document_score_source_sha256", "query_sha256", "text_sha256",
            "window_sha256", "window_id", "model", "model_revision",
            "document_start_token", "document_end_token", "rank_source",
        }
        ranking_required = {
            "task3_rankings_sha256", "facet_only_source", "tethered_source",
            "generating_facet", "percentile_method", "rank_source",
        }
        if set(passage) != passage_required or set(ranking) != ranking_required:
            raise ValueError("representative passage/ranking provenance schema differs")
        for field in ("candidate_source_sha256", "window_score_source_sha256", "query_sha256", "text_sha256", "window_sha256"):
            value = passage[field]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"representative passage provenance {field} is invalid")
        if passage["document_score_source_sha256"] is not None and (
            not isinstance(passage["document_score_source_sha256"], str)
            or len(passage["document_score_source_sha256"]) != 64
        ):
            raise ValueError("representative document score source SHA-256 is invalid")
        if not isinstance(ranking["task3_rankings_sha256"], str) or len(ranking["task3_rankings_sha256"]) != 64:
            raise ValueError("representative ranking provenance SHA-256 is invalid")
        if ranking["task3_rankings_sha256"] != task3_rankings_sha256:
            raise ValueError("representative ranking provenance does not match Task 3 rankings")
        if (
            passage["rank_source"] != "prior_bm25_rank"
            or ranking["percentile_method"] != "query_local_average_rank"
            or ranking["rank_source"] != "Task 3 sealed rankings.jsonl"
        ):
            raise ValueError("representative provenance constants differ")
        facet_row = task3_ranking_rows.get((topic, str(row["document_id"]), "FACET-2B"))
        tethered_row = task3_ranking_rows.get((topic, str(row["document_id"]), "TETHERED-2B"))
        generating_row = tethered_row if movement == "promoted" else facet_row
        if (
            not isinstance(facet_row, Mapping)
            or not isinstance(tethered_row, Mapping)
            or not isinstance(generating_row, Mapping)
            or normalized["facet_only_final_rank"] != facet_row.get("rank")
            or normalized["tethered_final_rank"] != tethered_row.get("rank")
            or normalized["prior_bm25_rank"] != generating_row.get("prior_bm25_rank")
            or ranking["facet_only_source"] != facet_row.get("source")
            or ranking["tethered_source"] != tethered_row.get("source")
            or ranking["generating_facet"] != row["facet_id"]
        ):
            raise ValueError("representative rank provenance does not match Task 3 rankings")
        prefix = "tethered" if movement == "promoted" else "facet"
        if (
            passage["candidate_source_sha256"] != source_hashes[f"{prefix}_candidates"]
            or passage["window_score_source_sha256"] != source_hashes[f"{prefix}_window_scores"]
            or (
                passage["document_score_source_sha256"] != source_hashes["tethered_document_scores"]
                if movement == "promoted"
                else passage["document_score_source_sha256"] is not None
            )
        ):
            raise ValueError("representative passage provenance does not match Task 3 inputs")
        identity = (topic, str(row["facet_id"]), str(row["document_id"]))
        if normalized["qrels_grade"] != authenticated_qrels[topic].get(str(row["document_id"]), 0):
            raise ValueError("representative qrels grade differs from authenticated projection")
        facet_candidate = candidate_maps["facet_candidates"].get(identity)
        tethered_candidate = candidate_maps["tethered_candidates"].get(identity)
        if not isinstance(facet_candidate, Mapping) or not isinstance(tethered_candidate, Mapping):
            raise ValueError("representative candidate identity is absent from exact producer sources")
        facet_query = tethered_candidate.get("facet_query")
        narrative = row["narrative"]
        exact_query = str(narrative) + "\n\nFocus: " + str(facet_query)
        if (
            facet_query != row["facet_query"]
            or tethered_candidate.get("query") != exact_query
            or tethered_candidate.get("query_sha256") != _sha256(exact_query.encode())
            or tethered_candidate.get("facet_query_sha256") != _sha256(str(facet_query).encode())
            or facet_candidate.get("query") != facet_query
            or facet_candidate.get("query_sha256") != tethered_candidate.get("facet_query_sha256")
            or facet_candidate.get("text_sha256") != tethered_candidate.get("text_sha256")
            or int(tethered_candidate.get("prior_bm25_rank", 0)) != normalized["prior_bm25_rank"]
        ):
            raise ValueError("representative narrative/facet/query/text/BM25 provenance differs")
        chosen_prefix = "tethered" if movement == "promoted" else "facet"
        chosen_windows = window_maps[f"{chosen_prefix}_window_scores"].get(identity, [])
        chosen = next((item for item in chosen_windows if item.get("window_id") == passage["window_id"]), None)
        selected_windows = _selected_windows(chosen_windows)
        semantic_percentiles, semantic_facet = semantic[("TETHERED-2B" if movement == "promoted" else "FACET-2B", topic, str(row["facet_id"]))]
        if (
            not isinstance(chosen, Mapping)
            or not selected_windows
            or chosen != selected_windows[0]
            or chosen.get("window_text") != row["selected_passage"]
            or chosen.get("window_sha256") != _sha256(str(row["selected_passage"]).encode())
            or chosen.get("window_sha256") != passage["window_sha256"]
            or chosen.get("query_sha256") != passage["query_sha256"]
            or chosen.get("document_sha256", chosen.get("text_sha256")) != passage["text_sha256"]
            or chosen.get("model") != passage["model"]
            or chosen.get("model_revision") != passage["model_revision"]
            or chosen.get("document_start_token") != passage["document_start_token"]
            or chosen.get("document_end_token") != passage["document_end_token"]
            or semantic_facet.get("model") != passage["model"]
            or semantic_facet.get("model_revision") != passage["model_revision"]
        ):
            raise ValueError("representative passage/model/span provenance differs from exact producer sources")
        facet_percentiles, _ = semantic[("FACET-2B", topic, str(row["facet_id"]))]
        tethered_percentiles, _ = semantic[("TETHERED-2B", topic, str(row["facet_id"]))]
        if (
            normalized["facet_only_percentile"] != facet_percentiles.get(str(row["document_id"]))
            or normalized["tethered_percentile"] != tethered_percentiles.get(str(row["document_id"]))
        ):
            raise ValueError("representative query-local percentiles differ from Task 3")
        if movement == "promoted":
            document = document_scores.get(identity)
            if (
                not isinstance(document, Mapping)
                or passage["document_score_source_sha256"] != source_hashes["tethered_document_scores"]
                or passage["window_sha256"] not in document.get("window_hashes", [])
                or document.get("window_hashes") != [item.get("window_sha256") for item in selected_windows]
                or document.get("model") != passage["model"]
                or document.get("model_revision") != passage["model_revision"]
            ):
                raise ValueError("representative document score provenance differs")
        normalized["passage_provenance"] = dict(passage)
        normalized["ranking_provenance"] = dict(ranking)
        rows.append(normalized)
        movements.add(movement)
    if movements != {"promoted", "demoted"}:
        raise ValueError("representatives must include promoted and demoted examples")
    return rows


def build_artifact(sources: ReportSources) -> dict[str, object]:
    """Build deterministic report data after authenticating all upstream sources."""

    verified = _verify_sources(sources)
    metrics = _mapping(verified["metrics"], "metrics")
    if isinstance(metrics.get("aggregate"), Mapping):
        aggregate = _mapping(metrics.get("aggregate"), "aggregate metrics")
    else:
        arms = _mapping(metrics.get("arms"), "arm metrics")
        aggregate = {
            arm: _mapping(_mapping(arms.get(arm), f"{arm} metrics").get("aggregate"), f"{arm} aggregate")
            for arm in ARMS
        }
    metric_rows = _metric_rows(aggregate)
    diagnostics = _mapping(verified["diagnostics"], "diagnostics")
    topic_rows, facet_rows = _contribution_rows(diagnostics)
    representatives = _representatives(
        diagnostics,
        task3_rankings_sha256=str(verified["task3_rankings_sha256"]),
        task3_ranking_rows=verified["task3_ranking_rows"],  # type: ignore[arg-type]
        source_hashes=verified["representative_source_hashes"],  # type: ignore[arg-type]
        producer_bytes=verified["producer_bytes"],  # type: ignore[arg-type]
        task3_bindings=verified["task3_bindings"],  # type: ignore[arg-type]
        authenticated_qrels=verified["authenticated_qrels"],  # type: ignore[arg-type]
    )
    decision = dict(_mapping(verified["decision"], "decision"))
    if decision.get("production_promotion_authorized") is True:
        raise ValueError("Task 4 decision does not forbid production promotion")

    def metric(arm: str, depth: int, field: str) -> float:
        return next(
            float(row[field])
            for row in metric_rows
            if row["arm"] == arm and row["depth"] == depth
        )

    facet_relevant = sum(
        int(row["relevant_count"]) for row in facet_rows if row["arm"] == "FACET-2B"
    )
    tethered_relevant = sum(
        int(row["relevant_count"]) for row in facet_rows if row["arm"] == "TETHERED-2B"
    )
    demoted_irrelevant = any(
        row["movement"] == "demoted"
        and int(row["qrels_grade"]) == 0
        and float(row["tethered_percentile"]) < float(row["facet_only_percentile"])
        for row in representatives
    )
    reduced_noise = tethered_relevant > facet_relevant and demoted_irrelevant
    recovered_novel = metric("TETHERED-2B", 500, "novel_retained") >= 89 and metric(
        "TETHERED-2B", 1000, "novel_retained"
    ) >= 142
    next_step = str(
        decision.get("next_step")
        or "Run a preregistered evaluation on fresh topics and untouched qrels."
    )
    novel_ids = _mapping(diagnostics.get("novel_relevant_ids"), "novel relevant IDs")
    if set(novel_ids) != set(TOPIC_IDS) or any(not isinstance(value, list) for value in novel_ids.values()):
        raise ValueError("diagnostic novel relevant IDs have invalid topic coverage")
    authenticated_novel = verified["authenticated_novel"]
    if any(
        set(map(str, novel_ids[topic])) != authenticated_novel[topic]  # type: ignore[index]
        for topic in TOPIC_IDS
    ):
        raise ValueError("diagnostic novel relevant IDs differ from authenticated qrels evidence")
    reconciled_novel_count = sum(len(set(map(str, value))) for value in novel_ids.values())  # type: ignore[arg-type]
    task4_summary = _mapping(verified["task4_summary"], "Task 4 summary")
    claimed_counts = (
        metrics.get("novel_relevant_total", metrics.get("novel_relevant_count")),
        task4_summary.get("novel_relevant_count"),
        decision.get("novel_relevant_count"),
    )
    if any(value != reconciled_novel_count for value in claimed_counts):
        raise ValueError("novel relevant count differs across authenticated Task 4 evidence")
    diagnostic_fields = (
        "noise_pattern_definitions", "noise_pattern_counts", "facet_yield_changes",
        "relevant_below_500", "duplicate_and_quota_pressure", "scoring_telemetry",
    )
    report_diagnostics = {field: diagnostics.get(field) for field in diagnostic_fields}
    if (
        not isinstance(report_diagnostics["noise_pattern_definitions"], Mapping)
        or any(not isinstance(report_diagnostics[field], list) for field in diagnostic_fields[1:5])
        or not isinstance(report_diagnostics["scoring_telemetry"], Mapping)
    ):
        raise ValueError("Task 4 complete diagnostic schema differs")
    patterns = report_diagnostics["noise_pattern_definitions"]
    assert isinstance(patterns, Mapping)
    for row in report_diagnostics["noise_pattern_counts"]:  # type: ignore[union-attr]
        if not isinstance(row, Mapping) or set(patterns) - set(row):
            raise ValueError("noise pattern count schema differs")
        if any(int(row[name]) < 0 or int(row[name]) > int(row["selected_count"]) for name in patterns):
            raise ValueError("noise pattern counts do not reconcile")
    for row in report_diagnostics["facet_yield_changes"]:  # type: ignore[union-attr]
        if not isinstance(row, Mapping) or row.get("classification") not in {"rose", "fell", "zero", "unchanged"}:
            raise ValueError("facet yield change schema differs")
        if int(row.get("tethered_relevant_count", 0)) - int(row.get("facet_only_relevant_count", 0)) != int(row.get("delta", 0)):
            raise ValueError("facet yield change counts do not reconcile")
    for row in report_diagnostics["relevant_below_500"]:  # type: ignore[union-attr]
        if not isinstance(row, Mapping) or row.get("reason") not in {
            "not_in_facet_candidate_pool", "facet_quota_exhausted",
            "facet_basket_capacity_exhausted",
        }:
            raise ValueError("relevant-below-500 reason schema differs")
        topic, document = str(row.get("topic_id")), str(row.get("document_id"))
        if (
            topic not in TOPIC_IDS
            or int(row.get("qrels_grade", -1))
            != verified["authenticated_qrels"][topic].get(document, 0)  # type: ignore[index]
        ):
            raise ValueError("relevant-below-500 qrels grade differs from authenticated projection")
    for row in report_diagnostics["duplicate_and_quota_pressure"]:  # type: ignore[union-attr]
        duplicates = row.get("duplicate_skip_totals") if isinstance(row, Mapping) else None
        shortages = row.get("shortage_counts") if isinstance(row, Mapping) else None
        if (
            not isinstance(duplicates, Mapping)
            or not isinstance(shortages, Mapping)
            or sum(map(int, duplicates.values())) != int(row.get("duplicate_skip_total", -1))
            or sum(map(int, shortages.values())) != int(row.get("shortage_total", -1))
        ):
            raise ValueError("duplicate/quota pressure counts do not reconcile")
    telemetry = _mapping(report_diagnostics["scoring_telemetry"], "scoring telemetry")
    task1 = _mapping(verified["task1"], "Task 1")
    task2 = _mapping(verified["task2"], "Task 2")
    task1_summary = _mapping(task1.get("summary"), "Task 1 summary")
    runtime = _mapping(task1.get("runtime_evidence"), "Task 1 runtime evidence")
    cache_hits = int(task2.get("cache_hit_count", -1))
    forward_pairs = int(task2.get("unique_forward_pair_count", -1))
    available_unique_pairs = task1_summary.get("unique_pair_count")
    unique_pairs = (
        int(available_unique_pairs)
        if available_unique_pairs is not None
        else cache_hits + forward_pairs
    )
    if min(cache_hits, forward_pairs, unique_pairs) < 0 or cache_hits + forward_pairs != unique_pairs:
        raise ValueError("Task 1/2 unique scoring pair accounting differs")
    expected_telemetry = {
        "preflight_source_sha256": verified["source_hashes"]["task1_receipt_sha256"],  # type: ignore[index]
        "scoring_receipt_source_sha256": verified["source_hashes"]["task2_receipt_sha256"],  # type: ignore[index]
        "model": task2.get("model"),
        "model_revision": task2.get("model_revision"),
        "query_document_pair_count": task1_summary.get("query_document_pair_count"),
        "planned_window_count": task2.get("planned_window_count"),
        "completed_window_count": task2.get("completed_window_count"),
        "document_score_count": task2.get("document_score_count"),
        "cache_hit_count": cache_hits,
        "cache_miss_count": forward_pairs,
        "unique_forward_pair_count": forward_pairs,
        "unique_scoring_pair_count": unique_pairs,
        "elapsed_seconds": task2.get("elapsed_seconds"),
        "projected_inference_seconds": runtime.get("projected_inference_seconds"),
        "peak_device_memory_bytes": task2.get("peak_device_memory_bytes"),
        "peak_host_memory_bytes": task2.get("peak_host_memory_bytes"),
    }
    if dict(telemetry) != expected_telemetry:
        raise ValueError("scoring telemetry differs from exact Task 1/2 sources")
    label = str(decision.get("label", "unknown"))
    title_label = label.replace("_", " ").title()
    novel_total_raw = reconciled_novel_count
    if type(novel_total_raw) is not int or novel_total_raw <= 0:
        raise ValueError("Task 4 metrics lack authenticated novel relevant total")
    return {
        "schema_version": SCHEMA_VERSION,
        "title": f"Narrative-tethered facet diagnostic — {title_label}",
        "mechanical_label": label,
        "scope": {
            "topic_ids": list(TOPIC_IDS),
            "post_qrels_diagnostic": True,
            "new_retrieval": False,
            "production_validation": False,
            "novel_relevant_total": novel_total_raw,
        },
        "answers": {
            "narrative_tether_reduced_noise": reduced_noise,
            "two_basket_recovered_novel_relevant": recovered_novel,
            "next_step": next_step,
            "noise_basis": {
                "facet_only_relevant_contributions": facet_relevant,
                "tethered_relevant_contributions": tethered_relevant,
                "irrelevant_example_demoted": demoted_irrelevant,
            },
        },
        "metric_deltas": {
            f"{field}@{depth}_tethered_vs_facet": metric("TETHERED-2B", depth, field)
            - metric("FACET-2B", depth, field)
            for depth in DEPTHS
            for field in ("binary_recall", "graded_recall", "novel_retained")
        },
        "metrics": metric_rows,
        "topic_contributions": topic_rows,
        "facet_contributions": facet_rows,
        "representatives": representatives,
        "diagnostics": report_diagnostics,
        "decision": decision,
        "source_hashes": verified["source_hashes"],
    }


def _pct(value: object) -> str:
    return f"{100 * float(value):.1f}%"


def _signed(value: object, *, percentage: bool = False) -> str:
    number = float(value) * (100 if percentage else 1)
    suffix = " pp" if percentage else ""
    return f"{number:+.1f}{suffix}"


def _render_html(artifact: Mapping[str, object]) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    metrics = artifact["metrics"]
    assert isinstance(metrics, list)
    deltas = _mapping(artifact["metric_deltas"], "metric deltas")
    answers = _mapping(artifact["answers"], "answers")
    decision = _mapping(artifact["decision"], "decision")
    diagnostics = _mapping(artifact["diagnostics"], "diagnostics")
    representatives = artifact["representatives"]
    topic_rows = artifact["topic_contributions"]
    facet_rows = artifact["facet_contributions"]
    assert isinstance(representatives, list) and isinstance(topic_rows, list) and isinstance(facet_rows, list)
    scope = _mapping(artifact["scope"], "scope")
    novel_total = int(scope["novel_relevant_total"])

    metric_body = "".join(
        "<tr>"
        f"<th scope='row'>{esc(row['arm'])}</th><td>{int(row['depth']):,}</td>"
        f"<td>{_pct(row['binary_recall'])}</td><td>{_pct(row['graded_recall'])}</td>"
        f"<td>{int(row['novel_retained'])} / {novel_total}</td>"
        "</tr>"
        for row in metrics
    )
    topic_body = "".join(
        "<tr>"
        f"<th scope='row'>{esc(row['topic_id'])}</th>"
        f"<td>{_signed(row['recall@500_delta_tethered_vs_facet'], percentage=True)}</td>"
        f"<td>{_signed(row['graded_recall@500_delta_tethered_vs_facet'], percentage=True)}</td>"
        "</tr>"
        for row in topic_rows
    )
    facet_body = "".join(
        "<tr>"
        f"<th scope='row'>{esc(row['facet_id'])}</th><td>{esc(row['arm'])}</td>"
        f"<td>{int(row['selected_count'])}</td><td>{int(row['relevant_count'])}</td>"
        f"<td>{_pct(row['relevant_yield'])}</td>"
        "</tr>"
        for row in facet_rows
    )
    evidence = "".join(
        "<article class='evidence' tabindex='0'>"
        f"<p class='eyebrow'>{esc(str(row['movement']).title())} · topic {esc(row['topic_id'])} · "
        f"qrels grade {int(row['qrels_grade'])}</p>"
        f"<h3>{esc(row['document_id'])}</h3>"
        f"<dl><dt>Full narrative</dt><dd>{esc(row['narrative'])}</dd>"
        f"<dt>Facet query</dt><dd>{esc(row['facet_query'])}</dd>"
        f"<dt>Selected passage</dt><dd>{esc(row['selected_passage'])}</dd>"
        f"<dt>Facet-only percentile</dt><dd>{_pct(row['facet_only_percentile'])}</dd>"
        f"<dt>Tethered percentile</dt><dd>{_pct(row['tethered_percentile'])}</dd>"
        f"<dt>Facet-only final rank</dt><dd>{int(row['facet_only_final_rank']):,}</dd>"
        f"<dt>Tethered final rank</dt><dd>{int(row['tethered_final_rank']):,}</dd>"
        f"<dt>Prior facet BM25 rank</dt><dd>{int(row['prior_bm25_rank']):,}</dd>"
        f"<dt>Passage provenance</dt><dd>window {esc(row['passage_provenance']['window_id'])}; "
        f"candidate source {esc(row['passage_provenance']['candidate_source_sha256'])}; window source {esc(row['passage_provenance']['window_score_source_sha256'])}; "
        f"document source {esc(row['passage_provenance']['document_score_source_sha256'])}; query {esc(row['passage_provenance']['query_sha256'])}; "
        f"text {esc(row['passage_provenance']['text_sha256'])}; window {esc(row['passage_provenance']['window_sha256'])}; model {esc(row['passage_provenance']['model'])} "
        f"revision {esc(row['passage_provenance']['model_revision'])}; tokens "
        f"{int(row['passage_provenance']['document_start_token'])}–{int(row['passage_provenance']['document_end_token'])}; rank source {esc(row['passage_provenance']['rank_source'])}</dd>"
        f"<dt>Ranking provenance</dt><dd>Task 3 SHA-256 {esc(row['ranking_provenance']['task3_rankings_sha256'])}; "
        f"facet source {esc(row['ranking_provenance']['facet_only_source'])}; tethered source {esc(row['ranking_provenance']['tethered_source'])}; "
        f"generating facet {esc(row['ranking_provenance']['generating_facet'])}; {esc(row['ranking_provenance']['percentile_method'])}; "
        f"rank source {esc(row['ranking_provenance']['rank_source'])}</dd></dl>"
        "</article>"
        for row in representatives
    )
    yes_noise = "Yes, within this diagnostic" if answers["narrative_tether_reduced_noise"] else "No clear reduction"
    yes_novel = "Yes, within this diagnostic" if answers["two_basket_recovered_novel_relevant"] else "No"
    noise_basis = (
        "Relevant facet-basket contributions increased and a judged-irrelevant high facet-only match was demoted when narrative context was restored."
        if answers["narrative_tether_reduced_noise"]
        else "The saved contribution counts and representative movements do not establish a clear reduction in facet noise."
    )
    noise_rows = diagnostics["noise_pattern_counts"]
    facet_changes = diagnostics["facet_yield_changes"]
    below_rows = diagnostics["relevant_below_500"]
    pressure_rows = diagnostics["duplicate_and_quota_pressure"]
    telemetry = _mapping(diagnostics["scoring_telemetry"], "scoring telemetry")
    assert all(isinstance(value, list) for value in (noise_rows, facet_changes, below_rows, pressure_rows))
    pattern_definitions = _mapping(
        diagnostics["noise_pattern_definitions"], "noise pattern definitions"
    )
    pattern_names = sorted(pattern_definitions)
    pattern_definition_html = "".join(
        f"<dt>{esc(name)}</dt><dd><code>{esc(pattern_definitions[name])}</code></dd>"
        for name in pattern_names
    )
    noise_body = "".join(
        f"<tr><th scope='row'>{esc(row['facet_id'])}</th><td>{esc(row['arm'])}</td><td>{int(row['selected_count'])}</td>"
        + "".join(f"<td>{int(row.get(name, 0))}</td>" for name in pattern_names)
        + "</tr>"
        for row in noise_rows
    )
    noise_headers = "".join(f'<th scope="col">{esc(name)}</th>' for name in pattern_names)
    change_body = "".join(
        f"<tr><th scope='row'>{esc(row['facet_id'])}</th><td>{int(row['facet_only_relevant_count'])}</td>"
        f"<td>{int(row['tethered_relevant_count'])}</td><td>{_signed(row['delta'])}</td>"
        f"<td>{esc(row['classification'])}</td></tr>"
        for row in facet_changes
    )
    below_body = "".join(
        f"<tr><th scope='row'>{esc(row['document_id'])}</th><td>{esc(row['arm'])}</td><td>{esc(row['topic_id'])}</td>"
        f"<td>{int(row['qrels_grade'])}</td><td>{int(row['final_rank'])}</td><td>{esc(row['best_facet'])}</td>"
        f"<td>{esc(row['best_facet_percentile'])}</td><td>{esc(row['prior_bm25_rank'])}</td>"
        f"<td>{esc(row['reason'])}</td></tr>"
        for row in below_rows
    ) or "<tr><td colspan='9'>No grade-2+ documents were below rank 500.</td></tr>"
    def map_text(value: object) -> str:
        mapping = _mapping(value, "diagnostic pressure map")
        return ", ".join(f"{key}: {int(mapping[key])}" for key in sorted(mapping)) or "None"

    pressure_body = "".join(
        f"<tr><th scope='row'>{esc(row['topic_id'])}</th><td>{esc(row['arm'])}</td>"
        f"<td>{esc(map_text(row['duplicate_skip_totals']))}</td><td>{int(row['duplicate_skip_total'])}</td>"
        f"<td>{esc(map_text(row['shortage_counts']))}</td><td>{int(row['shortage_total'])}</td></tr>"
        for row in pressure_rows
    )
    telemetry_html = "".join(
        f"<dt>{esc(key)}</dt><dd>{esc(telemetry[key])}</dd>" for key in telemetry
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(artifact['title'])}</title>
<style>
:root{{--ink:#172033;--muted:#536078;--paper:#f7f8fc;--card:#fff;--line:#c7cfdd;--accent:#234f9b;--good:#12613c;--warn:#7a3e00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
a{{color:#153f88}} a:focus-visible,[tabindex]:focus-visible{{outline:3px solid #e07800;outline-offset:4px}}
.skip{{position:absolute;left:-9999px}}.skip:focus{{left:1rem;top:1rem;background:#fff;padding:.75rem;z-index:2}}
header,main,footer{{width:min(1120px,calc(100% - 2rem));margin:auto}} header{{padding:3rem 0 1rem}} h1{{font-size:clamp(2rem,5vw,3.7rem);line-height:1.05;max-width:18ch}}
.label{{display:inline-block;background:#dce8ff;color:#173b78;border:1px solid #97b4e8;border-radius:999px;padding:.35rem .7rem;font-weight:700}}
.lede{{font-size:1.18rem;max-width:72ch}} section{{margin:2.25rem 0}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}}
.card,.evidence{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.1rem;box-shadow:0 3px 12px #1f33500c}}
.answer{{font-size:1.3rem;color:var(--good);font-weight:750}} .eyebrow{{color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em;font-size:.78rem}}
.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;list-style:none;padding:0}}.pipeline li{{background:#e8eef9;border:1px solid #a7b6ce;border-radius:10px;padding:.75rem;text-align:center;font-weight:650}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff}} table{{width:100%;border-collapse:collapse;min-width:650px}}caption{{text-align:left;font-weight:750;padding:1rem;background:#e8eef9}}th,td{{padding:.75rem;text-align:left;border-top:1px solid var(--line)}}thead th{{background:#f1f4fa}}tbody tr:nth-child(even){{background:#fafbfe}}
dl{{display:grid;grid-template-columns:minmax(9rem,1fr) 3fr;gap:.5rem 1rem}}dt{{font-weight:750}}dd{{margin:0}}.boundary{{border-left:6px solid #a75000;background:#fff4df;padding:1rem 1.2rem}}footer{{padding:2rem 0 4rem;color:var(--muted)}}
@media (max-width:760px){{.grid{{grid-template-columns:1fr}}.pipeline{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr}}header{{padding-top:2rem}}}}
@media (prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style></head><body><a class="skip" href="#main">Skip to report</a>
<header><span class="label">{esc(artifact['mechanical_label'])}</span><h1>{esc(artifact['title'])}</h1>
<p class="lede">The fixed comparison preserved the RRF head, then compared RRF, FACET-2B, and TETHERED-2B. Tethering changed Recall@500 by {_signed(deltas['binary_recall@500_tethered_vs_facet'], percentage=True)} and novel retention@500 by {_signed(deltas['novel_retained@500_tethered_vs_facet'])} documents versus FACET-2B.</p>
<p class="boundary"><strong>Interpretation boundary:</strong> post-qrels diagnostic · no new retrieval · not production validation.</p>
<p>The report builder reads only the exact historically anchored qrels projection already bound by Task 4, solely to authenticate qrels-derived diagnostics. It does not read original qrels or perform a new evaluation.</p></header>
<main id="main"><section aria-labelledby="questions"><h2 id="questions">Three decisions this report answers</h2><div class="grid">
<article class="card"><h3>Did narrative tethering reduce facet noise?</h3><p class="answer">{yes_noise}.</p><p>{noise_basis}</p></article>
<article class="card"><h3>Did two-basket fusion recover novel relevant documents?</h3><p class="answer">{yes_novel}.</p><p>TETHERED-2B retained {int(next(row['novel_retained'] for row in metrics if row['arm']=='TETHERED-2B' and row['depth']==500))}/{novel_total} at 500 and {int(next(row['novel_retained'] for row in metrics if row['arm']=='TETHERED-2B' and row['depth']==1000))}/{novel_total} at 1,000.</p></article>
<article class="card"><h3>What should happen next?</h3><p class="answer">Fresh validation.</p><p>{esc(answers['next_step'])}</p></article></div></section>
<section aria-labelledby="pipeline"><h2 id="pipeline">Fixed pipeline</h2><ol class="pipeline"><li>Sealed candidates</li><li>Facet + narrative MiniLM</li><li>Query-local percentiles</li><li>Protected two-basket fusion</li><li>Projection-only evaluation</li></ol></section>
<section aria-labelledby="metrics"><h2 id="metrics">Metric comparison</h2><p>Rows report Recall@500 and Recall@1000 alongside the corresponding graded recall and novel-document counts.</p><div class="table-wrap"><table><caption>RRF, FACET-2B, and TETHERED-2B at 500 and 1,000</caption><thead><tr><th scope="col">Arm</th><th scope="col">Depth</th><th scope="col">Recall</th><th scope="col">Graded recall</th><th scope="col">Novel relevant retained</th></tr></thead><tbody>{metric_body}</tbody></table></div></section>
<section aria-labelledby="topics"><h2 id="topics">Where the change came from</h2><div class="table-wrap"><table><caption>Per-topic TETHERED-2B deltas versus FACET-2B at 500</caption><thead><tr><th scope="col">Topic</th><th scope="col">Recall@500</th><th scope="col">Graded Recall@500</th></tr></thead><tbody>{topic_body}</tbody></table></div>
<div class="table-wrap" style="margin-top:1rem"><table><caption>Per-facet relevant contribution</caption><thead><tr><th scope="col">Facet</th><th scope="col">Arm</th><th scope="col">Selected</th><th scope="col">Relevant</th><th scope="col">Yield</th></tr></thead><tbody>{facet_body}</tbody></table></div></section>
<section aria-labelledby="examples"><h2 id="examples">Representative promoted and demoted passages</h2><p>These bounded rows come from authenticated diagnostics. The builder checks their grades against Task 4's exact historically anchored projection; it does not read original qrels or perform a new evaluation.</p><div class="grid">{evidence}</div></section>
<section aria-labelledby="noise"><h2 id="noise">Noise patterns</h2><dl>{pattern_definition_html}</dl><div class="table-wrap"><table><caption>Frozen regex pattern counts by arm and facet</caption><thead><tr><th scope="col">Facet</th><th scope="col">Arm</th><th scope="col">Selected</th>{noise_headers}</tr></thead><tbody>{noise_body}</tbody></table></div></section>
<section aria-labelledby="yield-changes"><h2 id="yield-changes">Facet yield changes</h2><div class="table-wrap"><table><caption>Relevant contribution changes</caption><thead><tr><th scope="col">Facet</th><th scope="col">Facet-only</th><th scope="col">Tethered</th><th scope="col">Delta</th><th scope="col">Classification</th></tr></thead><tbody>{change_body}</tbody></table></div></section>
<section aria-labelledby="below"><h2 id="below">Relevant below rank 500</h2><div class="table-wrap"><table><caption>Grade-2+ documents and explicit miss reasons</caption><thead><tr><th scope="col">Document</th><th scope="col">Arm</th><th scope="col">Topic</th><th scope="col">Qrels grade</th><th scope="col">Rank</th><th scope="col">Best facet</th><th scope="col">Best facet percentile</th><th scope="col">Prior BM25 rank</th><th scope="col">Reason</th></tr></thead><tbody>{below_body}</tbody></table></div></section>
<section aria-labelledby="pressure"><h2 id="pressure">Duplicate and quota pressure</h2><div class="table-wrap"><table><caption>Exact per-facet duplicate skips and shortages</caption><thead><tr><th scope="col">Topic</th><th scope="col">Arm</th><th scope="col">Duplicate map</th><th scope="col">Duplicate total</th><th scope="col">Shortage map</th><th scope="col">Shortage total</th></tr></thead><tbody>{pressure_body}</tbody></table></div></section>
<section aria-labelledby="telemetry"><h2 id="telemetry">Scoring telemetry</h2><dl>{telemetry_html}</dl></section>
<section aria-labelledby="limits"><h2 id="limits">What this does not establish</h2><p>The mechanical label is <strong>{esc(decision.get('label','unknown'))}</strong>, but the topics and judgments were already inspected. This is evidence for a fresh preregistered test, not authorization to promote a production system.</p></section></main>
<footer>Standalone offline artifact · source hashes are recorded in artifact.json · no external runtime dependencies.</footer></body></html>
"""


def _sqlite_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bytes)):
        return value
    if isinstance(value, bool):
        return int(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_database(path: Path, artifact: Mapping[str, object]) -> None:
    datasets = {
        "metrics": artifact["metrics"],
        "topic_contributions": artifact["topic_contributions"],
        "facet_contributions": artifact["facet_contributions"],
        "representatives": artifact["representatives"],
    }
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA page_size=4096")
        for name, values in datasets.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"report dataset {name} is empty")
            rows = [dict(_mapping(row, f"{name} row")) for row in values]
            columns = list(rows[0])
            if any(list(row) != columns for row in rows):
                raise ValueError(f"report dataset {name} has an unstable schema")
            declarations = [f'"{column}"' for column in columns]
            connection.execute(f'CREATE TABLE "{name}" ({", ".join(declarations)})')
            placeholders = ",".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({placeholders})',
                [tuple(_sqlite_value(row[column]) for column in columns) for row in rows],
            )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def build_report(sources: ReportSources, output: Path) -> BuiltReport:
    """Authenticate, build, and create the four deterministic report artifacts."""

    artifact = build_artifact(sources)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mechanical_label": artifact["mechanical_label"],
        "post_qrels_diagnostic": True,
        "new_retrieval": False,
        "production_validation": False,
        "topic_ids": list(TOPIC_IDS),
        "answers": artifact["answers"],
        "source_hashes": artifact["source_hashes"],
    }
    html_text = _render_html(artifact)
    artifact_bytes = _json_bytes(artifact)
    summary_bytes = _json_bytes(summary)
    html_bytes = html_text.encode("utf-8")

    output = Path(output)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"create-only output exists: {output}")
        existing = {entry.name for entry in output.iterdir()}
        if existing - {"README.md"} or any((output / name).exists() for name in OUTPUT_FILES):
            raise FileExistsError(f"create-only output is not empty: {output}")
    else:
        output.mkdir(parents=True)
    _exclusive_write(output / "artifact.json", artifact_bytes)
    _exclusive_write(output / "summary.json", summary_bytes)
    _write_database(output / "report_data.sqlite", artifact)
    _exclusive_write(output / "report.html", html_bytes)
    return BuiltReport(output, artifact, summary, html_text, artifact_bytes, summary_bytes, html_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the offline tethered-facet diagnostic report")
    parser.add_argument("--task1-receipt", required=True, type=Path)
    parser.add_argument("--task2-receipt", required=True, type=Path)
    parser.add_argument("--task3-freeze", required=True, type=Path)
    parser.add_argument("--task4-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_report(
        ReportSources(
            topic_ids=list(TOPIC_IDS),
            task1_receipt=args.task1_receipt,
            task2_receipt=args.task2_receipt,
            task3_freeze=args.task3_freeze,
            task4_evaluation=args.task4_evaluation,
        ),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

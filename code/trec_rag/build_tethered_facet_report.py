"""Build the offline, source-bound tethered-facet diagnostic report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


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


def _contains_hash(value: object, digest: str) -> bool:
    if value == digest:
        return True
    if isinstance(value, Mapping):
        return any(_contains_hash(item, digest) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_hash(item, digest) for item in value)
    return False


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
        task1.get("status") != "tokenizer_only_preflight_complete"
        or task1.get("qrels_opened", task1.get("qrels_read")) is not False
        or task1.get("retrieval_path_supported", task1.get("retrieval_performed")) is not False
    ):
        raise ValueError("Task 1 receipt is not an offline completed preflight")
    if (
        task2.get("status") != "complete"
        or task2.get("qrels_opened", task2.get("qrels_read")) is not False
        or task2.get("network_access_supported", task2.get("network_accessed")) is not False
    ):
        raise ValueError("Task 2 receipt is not a completed local-only scoring receipt")

    freeze = Path(sources.task3_freeze)
    seal, seal_bytes = _read_object(freeze / "SEALED.json", "Task 3 seal")
    if seal.get("status") not in {"sealed", "sealed_before_qrels"}:
        raise ValueError("Task 3 freeze is not sealed")
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
    if task3_summary.get("status") not in {"complete", "rankings_frozen_before_qrels"} or task3_summary.get("qrels_opened") is not False:
        raise ValueError("Task 3 freeze is not a qrels-free completed freeze")
    _validate_topics(task3_summary.get("topic_ids"), "Task 3")
    # Task 2 authenticates Task 1 above; Task 3 must in turn authenticate Task 2.
    if not _contains_hash(task3_bindings, _sha256(task2_bytes)):
        raise ValueError("Task 2 receipt SHA-256 is absent from Task 3 input bindings")
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
    evaluation_bindings = evaluation_payloads["input_bindings"]
    if not _contains_hash(evaluation_bindings, _sha256(seal_bytes)):
        raise ValueError("Task 3 seal SHA-256 is absent from Task 4 input bindings")
    _validate_topics(evaluation_payloads["metrics"].get("topic_ids"), "Task 4 metrics")
    if "topic_ids" in evaluation_payloads["diagnostics"]:
        _validate_topics(evaluation_payloads["diagnostics"].get("topic_ids"), "Task 4 diagnostics")

    raw_inputs = task3_bindings.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("Task 3 authenticated raw input bindings are missing")
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
    label = str(decision.get("label", "unknown"))
    title_label = label.replace("_", " ").title()
    novel_total_raw = metrics.get("novel_relevant_total", metrics.get("novel_relevant_count"))
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
        f"SHA-256 {esc(row['passage_provenance']['window_sha256'])}; model {esc(row['passage_provenance']['model'])} "
        f"revision {esc(row['passage_provenance']['model_revision'])}; tokens "
        f"{int(row['passage_provenance']['document_start_token'])}–{int(row['passage_provenance']['document_end_token'])}</dd>"
        f"<dt>Ranking provenance</dt><dd>Task 3 SHA-256 {esc(row['ranking_provenance']['task3_rankings_sha256'])}; "
        f"{esc(row['ranking_provenance']['percentile_method'])}</dd></dl>"
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
<p class="boundary"><strong>Interpretation boundary:</strong> post-qrels diagnostic · no new retrieval · not production validation.</p></header>
<main id="main"><section aria-labelledby="questions"><h2 id="questions">Three decisions this report answers</h2><div class="grid">
<article class="card"><h3>Did narrative tethering reduce facet noise?</h3><p class="answer">{yes_noise}.</p><p>{noise_basis}</p></article>
<article class="card"><h3>Did two-basket fusion recover novel relevant documents?</h3><p class="answer">{yes_novel}.</p><p>TETHERED-2B retained {int(next(row['novel_retained'] for row in metrics if row['arm']=='TETHERED-2B' and row['depth']==500))}/{novel_total} at 500 and {int(next(row['novel_retained'] for row in metrics if row['arm']=='TETHERED-2B' and row['depth']==1000))}/{novel_total} at 1,000.</p></article>
<article class="card"><h3>What should happen next?</h3><p class="answer">Fresh validation.</p><p>{esc(answers['next_step'])}</p></article></div></section>
<section aria-labelledby="pipeline"><h2 id="pipeline">Fixed pipeline</h2><ol class="pipeline"><li>Sealed candidates</li><li>Facet + narrative MiniLM</li><li>Query-local percentiles</li><li>Protected two-basket fusion</li><li>Projection-only evaluation</li></ol></section>
<section aria-labelledby="metrics"><h2 id="metrics">Metric comparison</h2><p>Rows report Recall@500 and Recall@1000 alongside the corresponding graded recall and novel-document counts.</p><div class="table-wrap"><table><caption>RRF, FACET-2B, and TETHERED-2B at 500 and 1,000</caption><thead><tr><th scope="col">Arm</th><th scope="col">Depth</th><th scope="col">Recall</th><th scope="col">Graded recall</th><th scope="col">Novel relevant retained</th></tr></thead><tbody>{metric_body}</tbody></table></div></section>
<section aria-labelledby="topics"><h2 id="topics">Where the change came from</h2><div class="table-wrap"><table><caption>Per-topic TETHERED-2B deltas versus FACET-2B at 500</caption><thead><tr><th scope="col">Topic</th><th scope="col">Recall@500</th><th scope="col">Graded Recall@500</th></tr></thead><tbody>{topic_body}</tbody></table></div>
<div class="table-wrap" style="margin-top:1rem"><table><caption>Per-facet relevant contribution</caption><thead><tr><th scope="col">Facet</th><th scope="col">Arm</th><th scope="col">Selected</th><th scope="col">Relevant</th><th scope="col">Yield</th></tr></thead><tbody>{facet_body}</tbody></table></div></section>
<section aria-labelledby="examples"><h2 id="examples">Representative promoted and demoted passages</h2><p>These bounded rows come from authenticated diagnostics; the report does not reopen canonical qrels or rankings.</p><div class="grid">{evidence}</div></section>
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

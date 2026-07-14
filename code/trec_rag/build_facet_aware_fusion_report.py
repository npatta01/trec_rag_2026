"""Build the canonical portable-report payload for the facet fusion pilot.

This module only reads already-frozen experiment artifacts.  It never opens
qrels, constructs rankings, retrieves documents, or performs inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .facet_aware_fusion_evaluate import validate_self_hash


PILOT_TOPICS = ("233", "273", "161", "14")
PROTECTED_TOPICS = frozenset(("144", "213", "224", "407", "515"))
GENERATED_AT = "2026-07-13T21:30:00Z"
REPORT_TITLE = "Facet-aware fusion found useful evidence, but xQuAD spent too much ranking budget"

CANONICAL_SOURCE_PATHS: dict[str, str] = {
    "pilot_manifest": "reports/experiments/facet_aware_fusion_pilot_v1/manifest.json",
    "retrieval_summary": "outputs/rag25_facet_aware_fusion_v1/retrieval_recovery_v1/retrieval_summary.json",
    "scoring_preflight": "outputs/rag25_facet_aware_fusion_v1/preflight_v1/preflight.json",
    "benchmark": "outputs/rag25_facet_aware_fusion_v1/benchmark_recovery_v1/benchmark_telemetry.json",
    "scoring_receipt": "outputs/rag25_facet_aware_fusion_v1/scoring_v1/scoring_receipt.json",
    "ranking_freeze": "outputs/rag25_facet_aware_fusion_v1/freeze_v1/freeze.json",
    "facet_gates": "outputs/rag25_facet_aware_fusion_v1/freeze_v1/gates.json",
    "cxq_provenance": "outputs/rag25_facet_aware_fusion_v1/freeze_v1/cxq_provenance.jsonl",
    "retrieval_candidates": "outputs/rag25_facet_aware_fusion_v1/retrieval_recovery_v1/candidates.jsonl",
    "metrics": "outputs/rag25_facet_aware_fusion_v1/evaluation_v1/metrics.json",
    "gains_losses": "outputs/rag25_facet_aware_fusion_v1/evaluation_v1/gains_losses.json",
    "decision": "outputs/rag25_facet_aware_fusion_v1/evaluation_v1/decision.json",
    "advisor_review": "reports/experiments/facet_aware_fusion_pilot_v1/advisor_review.md",
}


@dataclass(frozen=True)
class ReportInputs:
    """Immutable source bytes and their repository-relative identities."""

    source_bytes: Mapping[str, bytes]
    source_paths: Mapping[str, str]

    @classmethod
    def from_repo(cls, repo_root: Path) -> "ReportInputs":
        root = Path(repo_root)
        source_bytes: dict[str, bytes] = {}
        for source_id, relative in CANONICAL_SOURCE_PATHS.items():
            path = root / relative
            try:
                source_bytes[source_id] = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"required report source is missing: {relative}") from exc
        return cls(source_bytes=source_bytes, source_paths=CANONICAL_SOURCE_PATHS)


def _sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _json(inputs: ReportInputs, source_id: str) -> Any:
    try:
        return json.loads(inputs.source_bytes[source_id])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report source: {source_id}") from exc


def _jsonl(inputs: ReportInputs, source_id: str) -> list[dict[str, Any]]:
    try:
        lines = inputs.source_bytes[source_id].decode("utf-8").splitlines()
        values = [json.loads(line) for line in lines if line]
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report source: {source_id}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"invalid report rows: {source_id}")
    return values


def _validate_topic_boundary(*payloads: Any) -> None:
    expected = list(PILOT_TOPICS)
    for payload in payloads:
        if not isinstance(payload, Mapping) or payload.get("topic_ids") != expected:
            raise ValueError("report source violates the frozen topic boundary")
        observed = {str(value) for value in payload["topic_ids"]}
        if observed & PROTECTED_TOPICS:
            raise ValueError("report source violates the protected topic boundary")


def _paragraph_after_heading(markdown: str, heading: str) -> str:
    pattern = re.compile(
        rf"^#{{2,4}}\s+{re.escape(heading)}\s*$\n+(.*?)(?=^#{{2,4}}\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(markdown)
    if not match:
        return ""
    body = match.group(1).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    return paragraphs[0].replace("\n", " ") if paragraphs else ""


def _excerpt(text: str, limit: int = 500) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _source_specs(inputs: ReportInputs) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for source_id, relative in inputs.source_paths.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("report source paths must remain repository-relative")
        source = inputs.source_bytes.get(source_id)
        if source is None:
            raise ValueError(f"missing report source bytes: {source_id}")
        specs.append(
            {
                "id": source_id,
                "label": source_id.replace("_", " ").title(),
                "path": relative,
                "source_sha256": _sha256(source),
            }
        )
    return specs


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _materialize_sql_rows(
    rows: list[dict[str, Any]], columns: tuple[str, ...]
) -> tuple[str, list[dict[str, Any]]]:
    """Materialize reviewed rows through the exact SQLite query we expose."""

    if not rows:
        raise ValueError("report datasets must not be empty")
    selects: list[str] = []
    for index, row in enumerate(rows):
        values = []
        for column in columns:
            value = _sql_literal(row[column])
            values.append(f'{value} AS "{column}"' if index == 0 else value)
        selects.append("SELECT " + ", ".join(values))
    sql = "\nUNION ALL\n".join(selects)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        materialized = [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()
    return sql, materialized


def _derived_source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "source_sha256": _sha256(sql.encode("utf-8")),
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "filters": ["topics 233, 273, 161, and 14 only", "protected topics excluded"],
        },
    }


def _charts() -> list[dict[str, Any]]:
    return [
        {
            "id": "system_ndcg",
            "title": "nDCG@10 by retrieval and fusion arm",
            "subtitle": "Four held-out topics; higher values indicate better top-10 ranking quality.",
            "type": "bar",
            "intent": "comparison",
            "question": "Which frozen arm best ranks highly relevant documents in the top 10?",
            "rationale": "A categorical bar chart makes the six frozen-arm values directly comparable.",
            "comparisonContext": {"grain": "fusion arm", "unit": "nDCG@10", "baseline": "Original query"},
            "dataset": "system_metric_rows",
            "sourceId": "system_metrics_query",
            "encodings": {
                "x": {"field": "arm", "type": "nominal", "label": "Arm"},
                "y": {"field": "ndcg_at_10", "type": "quantitative", "format": "number", "label": "nDCG@10"},
                "tooltip": [
                    {"field": "graded_recall_at_100", "format": "number", "label": "Graded Recall@100"},
                    {"field": "judged_rate_at_100", "format": "percent", "label": "Judged@100"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "settings": {"sort": "none"},
            "surface": {"viewMode": "visualization"},
        },
        {
            "id": "novel_retention",
            "title": "Novel relevant facet documents retained at depth 100",
            "subtitle": "Share of 73 grade-≥2 accepted-facet candidates absent from the RRF top 100.",
            "type": "bar",
            "intent": "comparison",
            "question": "How much of the verified residual facet evidence does each alternative retain?",
            "rationale": "A bar chart exposes the coverage gain and its sharp tradeoff with ranking quality.",
            "comparisonContext": {"grain": "fusion arm", "unit": "fraction", "denominator": "73 residual relevant facet documents"},
            "dataset": "novel_retention_rows",
            "sourceId": "novel_retention_query",
            "encodings": {
                "x": {"field": "arm", "type": "nominal", "label": "Arm"},
                "y": {"field": "retention_fraction", "type": "quantitative", "format": "percent", "label": "Retention"},
                "tooltip": [
                    {"field": "retained_relevant", "format": "number", "label": "Retained"},
                    {"field": "eligible_relevant", "format": "number", "label": "Eligible"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "settings": {"sort": "none"},
            "surface": {"viewMode": "visualization"},
        },
    ]


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "per_topic_metrics",
            "title": "Per-topic metric audit",
            "subtitle": "All four held-out topics and all six frozen arms.",
            "dataset": "per_topic_metric_rows",
            "sourceId": "per_topic_metrics_query",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "topic_id", "direction": "asc"},
            "columns": [
                {"field": "topic_id", "label": "Topic", "type": "text"},
                {"field": "arm", "label": "Arm", "type": "text"},
                {"field": "ndcg_at_10", "label": "nDCG@10", "format": "number"},
                {"field": "graded_recall_at_100", "label": "Graded Recall@100", "format": "number"},
                {"field": "relevant_at_10", "label": "Relevant@10", "format": "number"},
                {"field": "judged_rate_at_100", "label": "Judged@100", "format": "percent"},
            ],
        },
        {
            "id": "facet_gates",
            "title": "Facet stream coherence gates",
            "subtitle": "Twenty streams passed; four failed frozen anchor or relation checks.",
            "dataset": "facet_gate_rows",
            "sourceId": "facet_gate_query",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "topic_id", "direction": "asc"},
            "columns": [
                {"field": "topic_id", "label": "Topic", "type": "text"},
                {"field": "facet_id", "label": "Facet", "type": "text"},
                {"field": "accepted", "label": "Accepted", "type": "text"},
                {"field": "anchor_top5_count", "label": "Anchor in top 5", "format": "number"},
                {"field": "anchor_relation_top5_count", "label": "Anchor + relation", "format": "number"},
                {"field": "failed_checks", "label": "Failed checks", "type": "text"},
            ],
        },
        {
            "id": "rejected_examples",
            "title": "Representative rejected-stream evidence",
            "subtitle": "Top BM25 passage from each rejected stream; diagnostic excerpts only.",
            "dataset": "rejected_facet_example_rows",
            "sourceId": "rejected_examples_query",
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "topic_id", "direction": "asc"},
            "columns": [
                {"field": "topic_id", "label": "Topic", "type": "text"},
                {"field": "facet_id", "label": "Rejected facet", "type": "text"},
                {"field": "failed_checks", "label": "Why rejected", "type": "text"},
                {"field": "passage_excerpt", "label": "Top passage excerpt", "type": "text"},
            ],
        },
    ]


def build_report(inputs: ReportInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a canonical report artifact and a compact reproducibility summary."""

    manifest = _json(inputs, "pilot_manifest")
    retrieval = _json(inputs, "retrieval_summary")
    preflight = _json(inputs, "scoring_preflight")
    benchmark = _json(inputs, "benchmark")
    scoring = _json(inputs, "scoring_receipt")
    freeze = _json(inputs, "ranking_freeze")
    gates = _json(inputs, "facet_gates")
    metrics = _json(inputs, "metrics")
    gains = _json(inputs, "gains_losses")
    decision = _json(inputs, "decision")

    _validate_topic_boundary(manifest, freeze, metrics, gains, decision)
    for payload in (metrics, gains, decision):
        validate_self_hash(payload)
    if not isinstance(gates, list) or len(gates) != 24:
        raise ValueError("facet gate artifact must contain exactly 24 streams")
    if {str(row.get("topic_id")) for row in gates} != set(PILOT_TOPICS):
        raise ValueError("facet gates violate the frozen topic boundary")

    accepted = sum(bool(row.get("accepted")) for row in gates)
    rejected = len(gates) - accepted
    if (accepted, rejected) != (20, 4):
        raise ValueError("facet gate counts differ from the frozen pilot")

    arm_order = ("O", "RRF", "TUS-C", "BI", "XQ", "CXQ")
    system_rows: list[dict[str, Any]] = []
    per_topic_rows: list[dict[str, Any]] = []
    for arm in arm_order:
        system = metrics["systems"][arm]
        aggregate = system["aggregate"]
        system_rows.append(
            {
                "arm": arm,
                "ndcg_at_10": aggregate["ndcg@10"],
                "graded_recall_at_100": aggregate["graded_recall@100"],
                "relevant_at_10": aggregate["relevant@10"],
                "judged_rate_at_10": aggregate["judged_rate@10"],
                "judged_rate_at_100": aggregate["judged_rate@100"],
            }
        )
        for topic_id in PILOT_TOPICS:
            row = system["per_topic"][topic_id]
            per_topic_rows.append(
                {
                    "topic_id": topic_id,
                    "arm": arm,
                    "ndcg_at_10": row["ndcg@10"],
                    "graded_recall_at_100": row["graded_recall@100"],
                    "relevant_at_10": row["relevant@10"],
                    "judged_rate_at_100": row["judged_rate@100"],
                }
            )

    retention_rows: list[dict[str, Any]] = []
    for arm in ("TUS-C", "BI", "XQ", "CXQ"):
        comparison = gains["comparisons"][arm]
        retention_rows.append(
            {
                "arm": arm,
                "retained_relevant": comparison["novel_relevant_retained"],
                "eligible_relevant": comparison["pre_fusion_novel_relevant"],
                "retention_fraction": comparison["novel_retention_fraction"],
                "topics_with_positive_retention": comparison["topics_with_positive_novel_retention"],
            }
        )
    if any(row["eligible_relevant"] != 73 for row in retention_rows):
        raise ValueError("novel relevant denominator differs across arms")

    gate_rows = [
        {
            "manifest_order": row["manifest_order"],
            "topic_id": str(row["topic_id"]),
            "facet_id": row["facet_id"],
            "accepted": bool(row["accepted"]),
            "anchor_top5_count": row["anchor_top5_count"],
            "anchor_relation_top5_count": row["anchor_relation_top5_count"],
            "wrong_domain_top5_count": row["wrong_domain_top5_count"],
            "content_warning_top5_count": row["content_warning_top5_count"],
            "failed_checks": ", ".join(row.get("failed_checks", [])) or "none",
        }
        for row in gates
    ]

    candidates = _jsonl(inputs, "retrieval_candidates")
    candidate_lookup = {
        (str(row.get("topic_id")), row.get("facet_id"), row.get("rank")): row
        for row in candidates
    }
    rejected_examples: list[dict[str, Any]] = []
    for gate in gates:
        if gate["accepted"]:
            continue
        candidate = candidate_lookup.get((str(gate["topic_id"]), gate["facet_id"], 1))
        if candidate is None:
            raise ValueError("rejected facet lacks its frozen rank-one passage")
        rejected_examples.append(
            {
                "topic_id": str(gate["topic_id"]),
                "facet_id": gate["facet_id"],
                "document_id": candidate["docid"],
                "failed_checks": ", ".join(gate["failed_checks"]),
                "passage_excerpt": _excerpt(candidate["text"]),
                "diagnostic_only": True,
            }
        )

    system_sql, system_rows = _materialize_sql_rows(
        system_rows,
        (
            "arm",
            "ndcg_at_10",
            "graded_recall_at_100",
            "relevant_at_10",
            "judged_rate_at_10",
            "judged_rate_at_100",
        ),
    )
    retention_sql, retention_rows = _materialize_sql_rows(
        retention_rows,
        (
            "arm",
            "retained_relevant",
            "eligible_relevant",
            "retention_fraction",
            "topics_with_positive_retention",
        ),
    )
    per_topic_sql, per_topic_rows = _materialize_sql_rows(
        per_topic_rows,
        (
            "topic_id",
            "arm",
            "ndcg_at_10",
            "graded_recall_at_100",
            "relevant_at_10",
            "judged_rate_at_100",
        ),
    )
    gate_sql, gate_rows = _materialize_sql_rows(
        gate_rows,
        (
            "manifest_order",
            "topic_id",
            "facet_id",
            "accepted",
            "anchor_top5_count",
            "anchor_relation_top5_count",
            "wrong_domain_top5_count",
            "content_warning_top5_count",
            "failed_checks",
        ),
    )
    examples_sql, rejected_examples = _materialize_sql_rows(
        rejected_examples,
        (
            "topic_id",
            "facet_id",
            "document_id",
            "failed_checks",
            "passage_excerpt",
            "diagnostic_only",
        ),
    )

    cxq_rows = _jsonl(inputs, "cxq_provenance")
    forced_count = sum(
        row.get("forced_facet") is not None or row.get("selection") != "xquad"
        for row in cxq_rows
    )
    artifacts = freeze.get("artifacts", {})
    xq_hash = artifacts.get("rankings/XQ.jsonl", {}).get("sha256")
    cxq_hash = artifacts.get("rankings/CXQ.jsonl", {}).get("sha256")
    xq_cxq_identical = bool(xq_hash and xq_hash == cxq_hash)

    advisor = inputs.source_bytes["advisor_review"].decode("utf-8").rstrip()
    advisor_verdict = _paragraph_after_heading(advisor, "Verdict")
    advisor_next = _paragraph_after_heading(advisor, "Recommended next experiment") or _paragraph_after_heading(
        advisor, "Single best next bounded experiment"
    )

    promoted = decision["decision"]["promoted_arm"]
    if promoted != "RRF":
        raise ValueError("the frozen mechanical decision no longer retains RRF")
    external_calls = retrieval["external_calls"]
    if external_calls != 24 or not retrieval["complete"]:
        raise ValueError("retrieval summary differs from the frozen completed run")

    headline = [{
        "rrf_ndcg_at_10": metrics["systems"]["RRF"]["aggregate"]["ndcg@10"],
        "rrf_graded_recall_at_100": metrics["systems"]["RRF"]["aggregate"]["graded_recall@100"],
        "novel_relevant_candidates": 73,
        "accepted_facets": accepted,
        "external_calls": external_calls,
        "local_cost_usd": 0,
    }]

    summary: dict[str, Any] = {
        "schema_version": "facet-aware-fusion-report-summary-v1",
        "topic_ids": list(PILOT_TOPICS),
        "decision": {"selected_arm": promoted, "no_new_arm_promoted": True, "failed_checks": decision["decision"]["failed_checks"]},
        "systems": {arm: metrics["systems"][arm]["aggregate"] for arm in arm_order},
        "facets": {"planned": 24, "accepted": accepted, "rejected": rejected},
        "novel_relevant": {
            "denominator": 73,
            "retained_by_xq": 66,
            "denominator_definition": "Unique grade-≥2 documents in accepted-facet MiniLM top-20 lists that are absent from RRF@100.",
        },
        "fusion": {"xq_cxq_byte_identical": xq_cxq_identical, "cxq_forced_selection_count": forced_count},
        "cost": {
            "external_calls": external_calls,
            "candidate_rows": retrieval["candidate_rows"],
            "scored_windows": scoring["completed_window_count"],
            "unique_local_pairs": scoring["forward_pass_count"],
            "projected_local_seconds": benchmark["projected_full_run_wall_seconds"],
            "monetary_usd": 0,
        },
        "advisor": {"verdict": advisor_verdict, "recommended_next_experiment": advisor_next},
        "bindings": {
            "ranking_freeze_sha256": metrics["bindings"]["ranking_freeze_sha256"],
            "metrics_self_sha256": metrics["self_sha256"],
            "decision_self_sha256": decision["self_sha256"],
        },
    }

    sources = _source_specs(inputs)
    sources.extend(
        [
            _derived_source(
                "system_metrics_query",
                "System metrics materialization",
                inputs.source_paths["metrics"],
                system_sql,
                "Exact aggregate metric rows materialized from the hash-verified metrics artifact.",
            ),
            _derived_source(
                "novel_retention_query",
                "Novel retention materialization",
                inputs.source_paths["gains_losses"],
                retention_sql,
                "Exact novel-retention rows materialized from the hash-verified gains/losses artifact.",
            ),
            _derived_source(
                "per_topic_metrics_query",
                "Per-topic metrics materialization",
                inputs.source_paths["metrics"],
                per_topic_sql,
                "Exact per-topic metric rows materialized from the hash-verified metrics artifact.",
            ),
            _derived_source(
                "facet_gate_query",
                "Facet gate materialization",
                inputs.source_paths["facet_gates"],
                gate_sql,
                "Exact deterministic gate rows materialized from the frozen gate artifact.",
            ),
            _derived_source(
                "rejected_examples_query",
                "Rejected passage materialization",
                inputs.source_paths["retrieval_candidates"],
                examples_sql,
                "Four bounded rank-one diagnostic excerpts materialized from frozen retrieval candidates.",
            ),
        ]
    )
    charts = _charts()
    tables = _tables()
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Technical summary\n\n"
                "**Decision: retain the current RRF; no new fusion arm is safe to promote.** The four-topic held-out pilot issued 24 rate-limited BM25 facet requests, locally scored 9,698 windows with MiniLM, and incurred **$0** in paid cost. RRF improved nDCG@10 from 0.472 to 0.589 with essentially flat graded Recall@100.\n\n"
                "**The facets worked as candidate generators.** Accepted facet lists contained relevant evidence missing from the original and RRF pools, and XQ/CXQ retained **66 of 73** residual relevant candidates. But xQuAD over-replaced the globally useful RRF list: it allocated 311 of 400 positions to facet-only documents and did not demonstrate safe improvement.\n\n"
                "**Advisor verdict:** retain facets, replace the fusion design. Protect the RRF head and use full-narrative MiniLM scores only for a small residual tail budget."
            ),
        },
        {
            "id": "facet_recall_finding",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Facets found relevant evidence that the main list missed\n\n"
                "The residual set contains 73 unique grade-≥2 documents in accepted-facet MiniLM top-20 lists that are **absent from RRF@100**. RRF's zero is true by construction, not an independent failure. The advisor additionally verified 83 relevant facet documents absent from the original top 100; RRF recovered some and left 73 outside both lists.\n\n"
                "The chart shows why the facet signal should be kept: XQ/CXQ retained 90.4% of that residual evidence. The unresolved problem is controlled insertion, not candidate discovery."
            ),
        },
        {"id": "novel_retention_chart", "type": "chart", "chartId": "novel_retention", "layout": "full"},
        {
            "id": "fusion_quality_finding",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## RRF improved the head; xQuAD consumed too much of the list\n\n"
                "RRF achieved the best nDCG@10 (0.589) and improved all four topics over the original query. XQ and CXQ fell to 0.284 because their relevance term treated the best document from every narrow facet as globally comparable to original rank 1. They replaced 290 RRF documents while only 51.5% of their top 100 was judged.\n\n"
                "The preregistered guardrails correctly block promotion. Because 187 of 290 XQ additions were unjudged, the exact size of the apparent loss is uncertain; the evidence supports ‘did not demonstrate safe improvement,’ not ‘facet documents were proven irrelevant.’"
            ),
        },
        {"id": "ndcg_chart", "type": "chart", "chartId": "system_ndcg", "layout": "full"},
        {"id": "per_topic_intro", "type": "markdown", "body": "## Every held-out topic improved at the top under RRF\n\nThe per-topic table preserves the exact metrics, judged coverage, and negative-result arms so aggregate gains cannot hide a topic regression.", "layout": "full"},
        {"id": "per_topic_table", "type": "table", "tableId": "per_topic_metrics", "layout": "full"},
        {"id": "facet_gate_intro", "type": "markdown", "body": "## The coherence gate removed four noisy streams before fusion\n\nTwenty of 24 facet streams passed. Four failed the frozen subject-anchor or anchor-plus-relation threshold. Content-quality warnings could prompt inspection but could not reject a stream by themselves.", "layout": "full"},
        {"id": "facet_gate_table", "type": "table", "tableId": "facet_gates", "layout": "full"},
        {"id": "rejected_example_intro", "type": "markdown", "body": "## Rejected passages show why facet-local filtering is necessary\n\nThese bounded excerpts are retrieval diagnostics, not relevance judgments. They show partial topic overlap without enough evidence for the requested relation.", "layout": "full"},
        {"id": "rejected_example_table", "type": "table", "tableId": "rejected_examples", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Scope and metric definitions\n\n"
                "**Scope.** Four deterministically selected, previously untouched development topics: 233, 273, 161, and 14. Protected topics 144, 213, 224, 407, and 515 and prior-pilot topics 200, 225, 707, and 897 were excluded.\n\n"
                "**nDCG@10** rewards placing higher-grade documents near the top. **Graded Recall@100** is the recovered relevance gain divided by all graded relevance gain in the projected qrels. **Judged@100** is the fraction of output documents present in the projection. **Novel retention** uses a deliberately RRF-relative denominator, so it compares alternative fusion arms only."
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## What ran and what stayed frozen\n\n"
                "The endpoint returned top-100 BM25 results for 24 narrative-tethered facet queries under a persistent one-start-per-three-seconds limiter. Local `cross-encoder/ms-marco-MiniLM-L6-v2` scored each document against its own facet; 20 coherent streams survived the deterministic gate.\n\n"
                "Six depth-100 arms were frozen before qrels access: original (O), family-balanced RRF, balanced interleaving (BI), xQuAD (XQ), constrained xQuAD (CXQ), and a TUS-style consensus diagnostic (TUS-C). XQ and CXQ are byte-identical and the CXQ deadline never fired, so this pilot contains no independent deadline-forcing result."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations and robustness\n\n"
                "This is a four-topic pilot, not production generalization. Judging coverage is differential: O is 100% judged at depth 100, RRF 98.25%, and XQ/CXQ 51.5%. Unjudged documents are scored as zero, so the measured xQuAD loss combines real displacement with non-random missing judgments.\n\n"
                "The ranking and evaluation files pass their native hash checks. No ranking was constructed after qrels access, no protected topic entered the pipeline, and no paid model or retrieval call was used."
            ),
        },
        {"id": "advisor_review", "type": "markdown", "body": advisor, "layout": "full"},
        {
            "id": "next_step",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Next experiment: protect RRF ranks 1–80\n\n"
                "On fresh held-out topics, score the complete candidate union once against the **full original narrative**, keep RRF ranks 1–80 unchanged, and let RRF ranks 81–100 compete with residual facet candidates for only 20 positions. Cap new documents at two per facet and never force insertion.\n\n"
                "This tests whether globally relevant facet evidence can replace only weak RRF-tail documents. It preserves nDCG@10 by construction, needs no new retrieval calls, and uses the already-cached cheap MiniLM model."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Does full-narrative MiniLM separate the 73 residual candidates from weak RRF-tail documents?\n"
                "- How many of the at-most-80 inserted documents require blind adjudication to remove differential-judging bias?\n"
                "- Is the two-per-facet cap sufficient across topics with three versus seven accepted facets?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "Held-out sparse retrieval and facet-aware fusion pilot.",
            "generatedAt": GENERATED_AT,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline_metrics": headline,
                "system_metric_rows": system_rows,
                "novel_retention_rows": retention_rows,
                "per_topic_metric_rows": per_topic_rows,
                "facet_gate_rows": gate_rows,
                "rejected_facet_example_rows": rejected_examples,
            },
        },
        "sources": sources,
    }
    return artifact, summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact, summary = build_report(ReportInputs.from_repo(args.repo_root))
    _write_json(args.artifact, artifact)
    _write_json(args.summary, summary)
    print(json.dumps({"artifact": str(args.artifact), "summary": str(args.summary), "selected_arm": summary["decision"]["selected_arm"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

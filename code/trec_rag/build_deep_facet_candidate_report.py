"""Build the canonical portable-report artifact for the deep-facet pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path


TITLE = "Deep facets find evidence; fusion loses precision"
TOPIC_IDS = ("219", "72", "300", "84")
ARM_ORDER = ("RRF", "GLOBAL", "FACET", "DUAL", "DUAL-NR")
CASCADE_ARM = "RRF-GLOBAL-DUAL"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def load_verified_evaluation(
    metrics_path: Path, decision_path: Path, summary_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    summary = _read_object(summary_path, "evaluation summary")
    if (
        summary.get("status") != "complete"
        or summary.get("qrels_opened") is not True
        or summary.get("metrics_sha256") != _sha256(metrics_path)
        or summary.get("decision_sha256") != _sha256(decision_path)
    ):
        raise ValueError("evaluation source hash or completion state differs")
    return (
        _read_object(metrics_path, "evaluation metrics"),
        _read_object(decision_path, "evaluation decision"),
        summary,
    )


def load_verified_cascade(
    metrics_path: Path, decision_path: Path, summary_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    summary = _read_object(summary_path, "cascade summary")
    if (
        summary.get("status") != "complete"
        or summary.get("post_qrels_diagnostic") is not True
        or summary.get("metrics_sha256") != _sha256(metrics_path)
        or summary.get("decision_sha256") != _sha256(decision_path)
    ):
        raise ValueError("cascade source hash or completion state differs")
    metrics = _read_object(metrics_path, "cascade metrics")
    decision = _read_object(decision_path, "cascade decision")
    if metrics.get("post_qrels_diagnostic") is not True:
        raise ValueError("cascade metrics are not labeled diagnostic")
    return metrics, decision, summary


def _pct(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _source(source_id: str, label: str, path: str, description: str) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "local-artifact",
            "language": "JSON/Markdown",
            "description": description,
            "tables_used": [path],
        },
    }


def _dataset_source(dataset: str, description: str) -> dict[str, object]:
    path = "reports/experiments/deep_facet_candidate_pilot_v1/report_data.sqlite"
    return {
        "id": f"{dataset}_sql",
        "label": f"{dataset.replace('_', ' ').title()} report dataset",
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "SQL",
            "sql": f"SELECT * FROM {dataset} ORDER BY row_order",
            "description": description,
            "tables_used": [dataset],
            "filters": ["Four frozen topics: 219, 72, 300, 84"],
        },
    }


def write_report_database(
    path: Path, datasets: Mapping[str, Sequence[Mapping[str, object]]]
) -> None:
    """Materialize and execute the exact SQLite sources declared by the artifact."""

    connection = sqlite3.connect(path)
    try:
        for name, raw_rows in datasets.items():
            if not name.replace("_", "").isalnum() or not raw_rows:
                raise ValueError("report datasets require safe names and non-empty rows")
            rows = [dict(row) for row in raw_rows]
            columns = list(rows[0])
            if any(list(row) != columns for row in rows):
                raise ValueError("report dataset rows must have a stable schema")
            connection.execute(f'DROP TABLE IF EXISTS "{name}"')
            declarations = ['"row_order" INTEGER PRIMARY KEY']
            for column in columns:
                if not column.replace("_", "").isalnum():
                    raise ValueError("report dataset column name is unsafe")
                values = [row[column] for row in rows if row[column] is not None]
                kind = "REAL" if any(isinstance(value, float) for value in values) else (
                    "INTEGER" if values and all(isinstance(value, (int, bool)) for value in values) else "TEXT"
                )
                declarations.append(f'"{column}" {kind}')
            connection.execute(f'CREATE TABLE "{name}" ({", ".join(declarations)})')
            placeholders = ",".join("?" for _ in range(len(columns) + 1))
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({placeholders})',
                [(index, *(row[column] for column in columns)) for index, row in enumerate(rows, start=1)],
            )
            fetched = connection.execute(
                f'SELECT * FROM "{name}" ORDER BY row_order'
            ).fetchall()
            if len(fetched) != len(rows):
                raise ValueError("report SQL source did not reproduce its dataset")
        connection.commit()
    finally:
        connection.close()


def build_artifact(
    metrics: Mapping[str, object],
    decision: Mapping[str, object],
    evaluation_summary: Mapping[str, object],
    *,
    advisor_memo: str,
    runtime_evidence: Mapping[str, object] | None = None,
    cascade_metrics: Mapping[str, object] | None = None,
    cascade_decision: Mapping[str, object] | None = None,
    cascade_advisor_memo: str = "",
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate = metrics.get("aggregate")
    discovery = metrics.get("discovery")
    per_topic = metrics.get("per_topic")
    if not isinstance(aggregate, Mapping) or not isinstance(discovery, Mapping) or not isinstance(per_topic, Mapping):
        raise ValueError("metrics lack aggregate, discovery, or per-topic evidence")
    if any(arm not in aggregate for arm in ARM_ORDER) or any(topic not in discovery for topic in TOPIC_IDS):
        raise ValueError("metrics lack a frozen arm or topic")
    novel_count = int(metrics["novel_relevant_count"])
    gate_lost = int(metrics["gate_lost_relevant_count"])
    novel_topics = int(metrics["novel_relevant_topic_count"])
    diagnosis = str(decision.get("diagnosis"))
    rrf = aggregate["RRF"]
    dual = aggregate["DUAL"]
    assert isinstance(rrf, Mapping) and isinstance(dual, Mapping)
    cascade: Mapping[str, object] | None = None
    cascade_per_topic: Mapping[str, object] | None = None
    if cascade_metrics is not None:
        cascade_aggregate = cascade_metrics.get("aggregate")
        cascade_per_topic = cascade_metrics.get("per_topic")  # type: ignore[assignment]
        if (
            cascade_metrics.get("post_qrels_diagnostic") is not True
            or not isinstance(cascade_aggregate, Mapping)
            or not isinstance(cascade_aggregate.get(CASCADE_ARM), Mapping)
            or not isinstance(cascade_per_topic, Mapping)
            or not isinstance(cascade_decision, Mapping)
        ):
            raise ValueError("cascade evidence is incomplete or not diagnostic")
        cascade = cascade_aggregate[CASCADE_ARM]  # type: ignore[assignment]
    cascade_guard_rows: list[dict[str, object]] = []
    if cascade is not None:
        assert isinstance(cascade_decision, Mapping)
        raw_guards = cascade_decision.get("guards")
        if not isinstance(raw_guards, Mapping) or not raw_guards:
            raise ValueError("cascade decision lacks mechanical guards")
        cascade_guard_rows = [
            {
                "guard": str(name).replace("_", " "),
                "outcome": "PASS" if passed is True else "FAIL",
            }
            for name, passed in raw_guards.items()
        ]
    runtime_value = runtime_evidence or {
        "external_calls": 0,
        "retry_count": 0,
        "phase1_seconds": 0.0,
        "phase2_seconds": 0.0,
        "peak_device_memory_bytes": 0,
    }
    runtime_row = {
        "external_calls": int(runtime_value["external_calls"]),
        "retry_count": int(runtime_value["retry_count"]),
        "phase1_seconds": float(runtime_value["phase1_seconds"]),
        "phase2_seconds": float(runtime_value["phase2_seconds"]),
        "local_scoring_seconds": float(runtime_value["phase1_seconds"]) + float(runtime_value["phase2_seconds"]),
        "peak_device_memory_mib": float(runtime_value["peak_device_memory_bytes"]) / (1024 * 1024),
        "paid_or_hosted_cost_usd": 0.0,
    }

    discovery_rows: list[dict[str, object]] = []
    topic_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for topic in TOPIC_IDS:
        row = discovery[topic]
        topic_metric = per_topic[topic]
        assert isinstance(row, Mapping) and isinstance(topic_metric, Mapping)
        for series, key in (("Original@1000", "original"), ("U_raw", "U_raw"), ("U_accepted", "U_accepted")):
            value = row[key]
            assert isinstance(value, Mapping)
            discovery_rows.append(
                {"topic": topic, "candidate_set": series, "relevant_documents": int(value["relevant_count"])}
            )
        novel_ids = row["novel_relevant_ids"]
        lost_ids = row["gate_lost_relevant_ids"]
        assert isinstance(novel_ids, list) and isinstance(lost_ids, list)
        topic_rows.append(
            {
                "topic": topic,
                "original_relevant": int(row["original"]["relevant_count"]),  # type: ignore[index]
                "accepted_relevant": int(row["U_accepted"]["relevant_count"]),  # type: ignore[index]
                "novel_relevant": len(novel_ids),
                "gate_lost": len(lost_ids),
                "rrf_ndcg10": float(topic_metric["RRF"]["ndcg@10"]),  # type: ignore[index]
                "dual_ndcg10": float(topic_metric["DUAL"]["ndcg@10"]),  # type: ignore[index]
                "dual_novel_500": int(topic_metric["DUAL"]["novel_retained@500"]),  # type: ignore[index]
                "dual_novel_1000": int(topic_metric["DUAL"]["novel_retained@1000"]),  # type: ignore[index]
                "cascade_ndcg10": (
                    float(cascade_per_topic[topic]["ndcg@10"])  # type: ignore[index]
                    if cascade_per_topic is not None else None
                ),
                "cascade_graded_recall500": (
                    float(cascade_per_topic[topic]["graded_recall@500"])  # type: ignore[index]
                    if cascade_per_topic is not None else None
                ),
                "cascade_graded_recall500_delta": (
                    float(cascade_per_topic[topic]["graded_recall@500_delta_vs_RRF"])  # type: ignore[index]
                    if cascade_per_topic is not None else None
                ),
            }
        )
        delta_rows.append(
            {"topic": topic, "ndcg10_delta": float(topic_metric["DUAL_ndcg@10_delta_vs_RRF"])}
        )

    arm_rows: list[dict[str, object]] = []
    retention_rows: list[dict[str, object]] = []
    report_arms = (*ARM_ORDER, CASCADE_ARM) if cascade is not None else ARM_ORDER
    for arm in report_arms:
        row = cascade if arm == CASCADE_ARM else aggregate[arm]
        assert isinstance(row, Mapping)
        arm_rows.append(
            {
                "arm": arm,
                "ndcg10": float(row["ndcg@10"]),
                "ndcg100": float(row["ndcg@100"]),
                "graded_recall500": float(row["graded_recall@500"]),
                "graded_recall1000": float(row["graded_recall@1000"]),
                "novel_retention500": float(row["novel_retention@500"]),
                "novel_retention1000": float(row["novel_retention@1000"]),
                "judged_rate100": float(row["judged_rate@100"]),
                "near_duplicate_rate500": (
                    float(row["near_duplicate_rate@500"])
                    if row.get("near_duplicate_rate@500") is not None else None
                ),
            }
        )
        retention_rows.extend(
            [
                {"arm": arm, "depth": "@500", "retention": float(row["novel_retention@500"])},
                {"arm": arm, "depth": "@1000", "retention": float(row["novel_retention@1000"])},
            ]
        )

    sources = [
        _dataset_source("headline", "Headline discovery, gate-loss, RRF, and DUAL metrics."),
        _dataset_source("discovery", "Relevant-document counts by topic and candidate set."),
        _dataset_source("arms", "Aggregate ranking and retention metrics by frozen arm."),
        _dataset_source("retention", "Novel relevant evidence retention by arm and depth."),
        _dataset_source("topic_deltas", "Per-topic DUAL nDCG@10 deltas versus RRF."),
        _dataset_source("topics", "Topic-level discovery and ranking tradeoff metrics."),
        _dataset_source("runtime", "External request and local MiniLM runtime/cost evidence."),
        _source(
            "evaluation_metrics",
            "Sealed four-topic evaluation metrics",
            "outputs/rag25_deep_facet_candidates_v1/evaluation_v1/metrics.json",
            "Qrels-backed discovery, ranking, judged-rate, duplication, and leave-one-out metrics.",
        ),
        _source(
            "evaluation_decision",
            "Mechanical pilot decision",
            "outputs/rag25_deep_facet_candidates_v1/evaluation_v1/decision.json",
            "Frozen advance guards and failure-stage diagnosis.",
        ),
        _source(
            "sealed_freeze",
            "Pre-qrels seal",
            "outputs/rag25_deep_facet_candidates_v1/freeze_v1/SEALED.json",
            "Hash inventory for all requests, candidates, scores, gates, and complete rankings frozen before qrels.",
        ),
        _source(
            "experiment_spec",
            "Preregistered experiment specification",
            "docs/superpowers/specs/2026-07-13-deep-facet-candidate-ranking-design.md",
            "Qrels-blind design, metrics, coefficients, tie-breaks, and stop rule.",
        ),
        _source(
            "advisor_review",
            "Independent advisor review",
            "reports/experiments/deep_facet_candidate_pilot_v1/advisor_review.md",
            "Post-results review that cannot change frozen candidates, rankings, or metrics.",
        ),
        _source(
            "runtime_receipt",
            "Local MiniLM scoring receipt",
            "outputs/rag25_deep_facet_candidates_v1/phase2_v1/scoring_receipt.json",
            "ROCm runtime, model revision, window count, memory, and output hashes.",
        ),
    ]
    if cascade is not None:
        sources.append(
            _dataset_source(
                "cascade_guards", "Mechanical pass/fail results for every fixed cascade stop guard."
            )
        )
        sources.extend(
            [
                _source(
                    "cascade_metrics",
                    "Sealed cascade diagnostic metrics",
                    "outputs/rag25_deep_facet_candidates_v1/post_qrels_cascade_v1/evaluation/metrics.json",
                    "Post-qrels diagnostic ranking, recall, novel-retention, and judged-rate metrics.",
                ),
                _source(
                    "cascade_decision",
                    "Cascade stop-rule decision",
                    "outputs/rag25_deep_facet_candidates_v1/post_qrels_cascade_v1/evaluation/decision.json",
                    "Mechanical guards for the fixed RRF–GLOBAL–DUAL cascade.",
                ),
                _source(
                    "cascade_freeze",
                    "Cascade diagnostic seal",
                    "outputs/rag25_deep_facet_candidates_v1/post_qrels_cascade_v1/freeze/SEALED.json",
                    "Hash inventory proving the cascade ranking was frozen before diagnostic metrics were computed.",
                ),
                _source(
                    "cascade_advisor",
                    "Independent cascade-results review",
                    "reports/experiments/deep_facet_candidate_pilot_v1/cascade_advisor_review.md",
                    "Independent interpretation of the fixed cascade outcome and next-step recommendation.",
                ),
            ]
        )

    cards = [
        {
            "id": "novel_card",
            "description": "Grade-2-or-higher documents in U_accepted but absent from original@1000.",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {"label": "Novel relevant documents", "field": "novel_relevant", "format": "number"},
                {"label": "Topics with additions", "field": "novel_topics", "format": "number"},
            ],
        },
        {
            "id": "gate_card",
            "description": "Relevant documents present in U_raw but removed with a rejected facet stream.",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [{"label": "Relevant documents lost by gate", "field": "gate_lost", "format": "number"}],
        },
        {
            "id": "rrf_card",
            "description": "Best aggregate early-ranking score among the frozen arms.",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [{"label": "RRF nDCG@10", "field": "rrf_ndcg10", "format": "number"}],
        },
        {
            "id": "dual_card",
            "description": "Fraction of all 177 novel relevant topic-document identities retained by DUAL@1000.",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [{"label": "DUAL novel retention@1000", "field": "dual_novel1000", "format": "percent"}],
        },
    ]
    if cascade is not None:
        cards.append(
            {
                "id": "cascade_card",
                "description": "The fixed cascade's macro graded Recall@500; it must beat both RRF and GLOBAL to pass.",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Cascade graded Recall@500", "field": "cascade_graded_recall500", "format": "percent"}
                ],
            }
        )

    charts = [
        {
            "id": "candidate_discovery",
            "title": "Relevant-document discovery by topic",
            "subtitle": "Grade ≥2 documents in original@1000, U_raw, and U_accepted.",
            "type": "bar",
            "dataset": "discovery",
            "sourceId": "discovery_sql",
            "encodings": {
                "x": {"field": "topic", "type": "nominal", "label": "Topic"},
                "y": {"field": "relevant_documents", "type": "quantitative", "label": "Relevant documents"},
                "color": {"field": "candidate_set", "type": "nominal", "label": "Candidate set"},
            },
            "layout": "full",
        },
        {
            "id": "early_precision",
            "title": "nDCG@10 by frozen ranking arm",
            "subtitle": "Macro mean across four topics; higher is better.",
            "type": "bar",
            "dataset": "arms",
            "sourceId": "arms_sql",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "arm", "type": "nominal", "label": "Ranking arm"},
                "y": {"field": "ndcg10", "type": "quantitative", "label": "nDCG@10"},
            },
            "layout": "full",
        },
        {
            "id": "novel_retention",
            "title": "Novel relevant evidence retained by depth",
            "subtitle": "Micro fraction of 177 novel relevant topic-document identities.",
            "type": "bar",
            "dataset": "retention",
            "sourceId": "retention_sql",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "arm", "type": "nominal", "label": "Ranking arm"},
                "y": {"field": "retention", "type": "quantitative", "label": "Novel retention", "format": "percent"},
                "color": {"field": "depth", "type": "nominal", "label": "Depth"},
            },
            "layout": "full",
        },
        {
            "id": "topic_delta",
            "title": "DUAL nDCG@10 delta versus RRF",
            "subtitle": "Per-topic signed difference; zero means no early-ranking change.",
            "type": "bar",
            "dataset": "topic_deltas",
            "sourceId": "topic_deltas_sql",
            "encodings": {
                "x": {"field": "topic", "type": "nominal", "label": "Topic"},
                "y": {"field": "ndcg10_delta", "type": "quantitative", "label": "nDCG@10 delta"},
            },
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "arm_table",
            "title": "Frozen arm metrics",
            "subtitle": "Macro ranking metrics and micro novel-evidence retention across four topics.",
            "dataset": "arms",
            "sourceId": "arms_sql",
            "defaultSort": {"field": "ndcg10", "direction": "desc"},
            "columns": [
                {"field": "arm", "label": "Arm", "type": "text"},
                {"field": "ndcg10", "label": "nDCG@10", "format": "number"},
                {"field": "graded_recall500", "label": "Graded Recall@500", "format": "percent"},
                {"field": "graded_recall1000", "label": "Graded Recall@1000", "format": "percent"},
                {"field": "novel_retention500", "label": "Novel retention@500", "format": "percent"},
                {"field": "novel_retention1000", "label": "Novel retention@1000", "format": "percent"},
                {"field": "judged_rate100", "label": "Judged rate@100", "format": "percent"},
            ],
        },
        {
            "id": "topic_table",
            "title": "Topic-level discovery and ranking tradeoff",
            "subtitle": "Exact grade≥2 discovery counts and early-ranking outcomes by topic.",
            "dataset": "topics",
            "sourceId": "topics_sql",
            "defaultSort": {"field": "novel_relevant", "direction": "desc"},
            "columns": [
                {"field": "topic", "label": "Topic", "type": "text"},
                {"field": "original_relevant", "label": "Original relevant", "format": "number"},
                {"field": "accepted_relevant", "label": "Accepted relevant", "format": "number"},
                {"field": "novel_relevant", "label": "Novel relevant", "format": "number"},
                {"field": "gate_lost", "label": "Gate lost", "format": "number"},
                {"field": "rrf_ndcg10", "label": "RRF nDCG@10", "format": "number"},
                {"field": "dual_ndcg10", "label": "DUAL nDCG@10", "format": "number"},
                {"field": "cascade_ndcg10", "label": "Cascade nDCG@10", "format": "number"},
                {"field": "cascade_graded_recall500", "label": "Cascade graded Recall@500", "format": "percent"},
                {"field": "cascade_graded_recall500_delta", "label": "Cascade GR@500 Δ vs RRF", "format": "number"},
            ],
        },
    ]
    if cascade is not None:
        tables.append(
            {
                "id": "cascade_guard_table",
                "title": "Fixed cascade stop guards",
                "subtitle": "Every preregistered mechanical condition; any FAIL stops the diagnostic arm.",
                "dataset": "cascade_guards",
                "sourceId": "cascade_guards_sql",
                "defaultSort": {"field": "outcome", "direction": "asc"},
                "columns": [
                    {"field": "guard", "label": "Guard", "type": "text"},
                    {"field": "outcome", "label": "Outcome", "type": "text"},
                ],
            }
        )

    advisor_clean = advisor_memo.strip() or "Advisor review was requested but no memo was available."
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "cascade_metrics" if cascade is not None else "evaluation_metrics",
            "body": (
                (
                    "## Technical summary\n\n"
                    "**The fixed cascade protected the first ten results, but did not fix relevance at candidate depth 500.** "
                    f"Its nDCG@10 exactly matches RRF at **{float(cascade['ndcg@10']):.3f}**, and it retains **{_pct(cascade['novel_retention@1000'])}** of the {novel_count} novel relevant documents by 1,000. "
                    f"But graded Recall@500 is only **{float(cascade['graded_recall@500']):.3f}**, below RRF (**{float(rrf['graded_recall@500']):.3f}**) and GLOBAL (**{float(aggregate['GLOBAL']['graded_recall@500']):.3f}**). "  # type: ignore[index]
                    "This post-qrels diagnostic fails the fixed stop rule. The candidate coverage is real; a positional splice of existing MiniLM/RRF rankings is not enough."
                )
                if cascade is not None
                else (
                    "## Technical summary\n\n"
                    f"**Facet decomposition succeeds at candidate discovery, but the tested fusion does not safely rank the additions.** "
                    f"U_accepted adds **{novel_count} grade≥2 documents** beyond original@1000 across **{novel_topics}/4 topics**; the stream gate loses only **{gate_lost}** relevant documents. "
                    f"However, RRF leads early precision at **{float(rrf['ndcg@10']):.3f} nDCG@10**, while DUAL falls to **{float(dual['ndcg@10']):.3f}** despite retaining **{_pct(dual['novel_retention@1000'])}** of novel evidence by rank 1,000. "
                    "The preregistered decision is therefore **stop; diagnose fusion**—not reject facet queries or accept pure BM25 as sufficient."
                )
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "discovery_finding",
            "type": "markdown",
            "sourceId": "evaluation_metrics",
            "body": (
                "## Facets add relevant evidence on every topic\n\n"
                "The important result is the gap between the original narrative pool and the accepted facet union. The additions are **+19** documents for topic 219, **+77** for 72, **+60** for 300, and **+21** for 84. "
                "U_raw and U_accepted are identical on three topics; the one rejected animal-vaccination stream costs two relevant documents on topic 84. "
                "This places the dominant failure after retrieval and mostly after gating."
            ),
        },
        {"id": "discovery_chart", "type": "chart", "chartId": "candidate_discovery", "layout": "full"},
        {
            "id": "fusion_finding",
            "type": "markdown",
            "sourceId": "evaluation_metrics",
            "body": (
                "## Coverage and early precision pull in opposite directions\n\n"
                f"RRF retains only **{_pct(rrf['novel_retention@500'])}** of novel evidence by 500 but has the strongest nDCG@10. DUAL raises novel retention@500 to **{_pct(dual['novel_retention@500'])}**, yet loses **{float(rrf['ndcg@10']) - float(dual['ndcg@10']):.3f}** aggregate nDCG@10. "
                "FACET is the extreme diagnostic: it retains almost all novel evidence by 1,000, but its nDCG@10 collapses. A useful next method must keep RRF-like head precision while admitting facet evidence deeper in the ranking."
            ),
        },
        {"id": "precision_chart", "type": "chart", "chartId": "early_precision", "layout": "full"},
        {"id": "retention_chart", "type": "chart", "chartId": "novel_retention", "layout": "full"},
        *(
            [
                {
                    "id": "cascade_finding",
                    "type": "markdown",
                    "sourceId": "cascade_metrics",
                    "body": (
                        "## The cascade answers the reranking question\n\n"
                        f"Ranks 1–10 remain exact RRF, ranks 11–100 use GLOBAL MiniLM order, and ranks 101 onward use DUAL. This keeps nDCG@10 unchanged and preserves **{int(cascade.get('novel_retained@1000', round(float(cascade['novel_retention@1000']) * novel_count)))}/{novel_count}** novel relevant documents by 1,000. "
                        f"It still reaches only **{_pct(cascade['graded_recall@500'])} graded Recall@500**, versus **{_pct(rrf['graded_recall@500'])}** for RRF and **{_pct(aggregate['GLOBAL']['graded_recall@500'])}** for GLOBAL. "  # type: ignore[index]
                        "Topic 219 loses more than the allowed 0.02, so the problem is not merely protecting the head. Existing MiniLM scores plus a fixed positional budget do not separate relevant facet candidates reliably enough."
                    ),
                }
            ]
            if cascade is not None
            else []
        ),
        *(
            [
                {
                    "id": "cascade_guard_table_block",
                    "type": "table",
                    "tableId": "cascade_guard_table",
                    "layout": "full",
                }
            ]
            if cascade is not None
            else []
        ),
        {
            "id": "topic_finding",
            "type": "markdown",
            "sourceId": "evaluation_metrics",
            "body": (
                "## The precision loss is concentrated but not isolated\n\n"
                "DUAL loses nDCG@10 on all four topics. The largest regressions are topic 300 (**−0.163**) and topic 84 (**−0.134**); topic 219 loses **−0.063** and topic 72 **−0.033**. "
                "Topic 300 also shows why the candidate union should be preserved: DUAL retains 50 novel relevant documents by 500 and slightly improves that topic's graded Recall@500, even while its first ten ranks worsen sharply."
            ),
        },
        {"id": "delta_chart", "type": "chart", "chartId": "topic_delta", "layout": "full"},
        {
            "id": "definitions",
            "type": "markdown",
            "sourceId": "experiment_spec",
            "body": (
                "## What was measured\n\n"
                "**Cohort.** Four fresh development topics (219, 72, 300, 84); protected and previously qrels-exposed pilot topics were excluded.\n\n"
                "**Relevant.** Projected qrel grade ≥2. **NovelRel** is a relevant topic-document identity absent from original@1000 but present in U_accepted. **Graded Recall** uses gain `2^grade−1` over grade≥2 documents. **nDCG** measures gain near the top of a ranking. **Judged rate** is reported separately; an unjudged document is not called irrelevant.\n\n"
                "**Candidate sets.** U_raw is original@1000 plus every successful facet@200. U_accepted removes only entire facet streams that failed the qrels-blind coherence gate. Every ranking arm permutes the same U_accepted set."
            ),
        },
        {"id": "arm_table_block", "type": "table", "tableId": "arm_table", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "sealed_freeze",
            "body": (
                "## Rankings were frozen before qrels\n\n"
                "Twenty-five depth-200 BM25 facet requests were retrieved through the persistent rate limiter. MiniLM scored each facet locally, one incoherent stream was rejected, and the accepted union received common-topic plus full-narrative MiniLM scores. "
                "RRF, GLOBAL, FACET, DUAL, and DUAL-NR then produced complete deterministic permutations. All requests, responses, candidates, model receipts, scores, gates, parameters, and rankings were sealed before the qrels sentinel was created. "
                "This is a deterministic two-stage retrieval experiment, not a free-running agent."
            ),
        },
        {
            "id": "runtime",
            "type": "markdown",
            "sourceId": "runtime_sql",
            "body": (
                "## Cost and runtime stayed bounded\n\n"
                f"Retrieval made **{runtime_row['external_calls']} external requests** through the persistent limiter with **{runtime_row['retry_count']} retries**. "
                f"Facet-local MiniLM took **{runtime_row['phase1_seconds']:.1f} seconds** and common/narrative scoring **{runtime_row['phase2_seconds']:.1f} seconds**, or **{runtime_row['local_scoring_seconds']:.1f} seconds** total local inference. "
                f"Peak device memory was **{runtime_row['peak_device_memory_mib']:.0f} MiB**. No model download, hosted inference, or paid call was used; measured hosted/paid cost was **$0**."
            ),
        },
        {"id": "topic_table_block", "type": "table", "tableId": "topic_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "evaluation_metrics",
            "body": (
                "## Limitations and robustness checks\n\n"
                "This is a four-topic pilot and cannot promote a production method. Semantic arms have much lower judged rates near the head than RRF, so their observed nDCG may penalize unjudged discoveries as zero; the report therefore does not label those documents irrelevant. "
                "The leave-one-topic-out requirement fails, so the DUAL result is not stable enough to advance. The DUAL-NR sensitivity arm changes little, which indicates that the 0.80 lexical redundancy penalty is not the primary cause of the precision loss. "
                "The qrels are projected development judgments rather than final task judgments."
            ),
        },
        {
            "id": "advisor",
            "type": "markdown",
            "sourceId": "advisor_review",
            "body": "## Independent advisor review\n\n" + advisor_clean,
        },
        *(
            [
                {
                    "id": "cascade_advisor",
                    "type": "markdown",
                    "sourceId": "cascade_advisor",
                    "body": (
                        "## Advisor review after the cascade\n\n"
                        + (cascade_advisor_memo.strip() or "Cascade review is pending.")
                    ),
                }
            ]
            if cascade is not None
            else []
        ),
        {
            "id": "next_step",
            "type": "markdown",
            "sourceId": "cascade_advisor" if cascade is not None else "evaluation_decision",
            "body": (
                (
                    "## Recommended next step\n\n"
                    "**Do not tune another positional cascade on these exposed topics.** The bounded mechanism diagnostic is one protected-head Mixedbread rerank: keep RRF ranks 1–10, form the residual union of RRF@500, GLOBAL@500, and DUAL@1000, and score it once with `mixedbread-ai/mxbai-rerank-base-v2`. "
                    "Use one identical structured query per topic containing the full narrative plus the complete accepted-facet obligation list, so scores are comparable within the topic. Fill ranks 11–500 from that score and append the remaining complete union in frozen DUAL order. Audit cached scores and freeze exact documents, windows, runtime, memory, and cost before any separately approved inference."
                )
                if cascade is not None
                else (
                    "## Recommended next step\n\n"
                    "Preserve the frozen U_accepted union and test **one deterministic RRF–GLOBAL–DUAL cascade**: exact RRF at ranks 1–10, GLOBAL at 11–100 while skipping selected documents, and DUAL from 101 onward, again skipping duplicates and eventually appending the complete union. "
                    "This uses only existing scores—no retrieval, inference, new weights, filtering, or iterative tuning. Freeze the cascade before evaluation. Stop if nDCG@10 differs from RRF, graded Recall@500 does not beat both RRF and GLOBAL, graded Recall@1000 falls below RRF, novel retention@1000 is below 80%, any topic loses more than 0.02 graded Recall@500 versus RRF, or judged coverage prevents a defensible comparison."
                )
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                (
                    "## Further questions\n\n"
                    "- How much of the frozen disagreement pool is already covered by authenticated Mixedbread cache entries?\n"
                    "- Would blind judgments of top-500 disagreements resolve the shallow judged-coverage imbalance?\n"
                    "- On fresh preregistered topics, can the protected-head reranker promote novel facet evidence without topic-level recall regressions?"
                )
                if cascade is not None
                else (
                    "## Further questions\n\n"
                    "- How large can the protected RRF head be while still admitting meaningful facet evidence by 500?\n"
                    "- Should insertion eligibility require both strong facet-local percentile and a minimum common/narrative coherence score?\n"
                    "- On a larger preregistered topic set, does the topic-300 recall gain persist without its early-precision regression?"
                )
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Qrels-backed technical report on deep facet candidate discovery and fusion.",
            "generatedAt": "2026-07-14T00:00:00-04:00",
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": row["id"], "label": row["label"], "path": row["path"]} for row in sources],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-07-14T00:00:00-04:00",
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "novel_relevant": novel_count,
                        "novel_topics": novel_topics,
                        "gate_lost": gate_lost,
                        "rrf_ndcg10": float(rrf["ndcg@10"]),
                        "dual_novel1000": float(dual["novel_retention@1000"]),
                        "cascade_graded_recall500": (
                            float(cascade["graded_recall@500"])
                            if cascade is not None else None
                        ),
                    }
                ],
                "discovery": discovery_rows,
                "arms": arm_rows,
                "retention": retention_rows,
                "topic_deltas": delta_rows,
                "topics": topic_rows,
                "runtime": [runtime_row],
                **({"cascade_guards": cascade_guard_rows} if cascade is not None else {}),
            },
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://deep-facet-candidate-pilot-v1",
            "controls": {"edit": False, "refresh": False},
        },
    }
    report_summary = {
        "title": TITLE,
        "status": evaluation_summary.get("status"),
        "qrels_opened": evaluation_summary.get("qrels_opened"),
        "diagnosis": diagnosis,
        "advance_to_larger_validation": decision.get("advance_to_larger_validation"),
        "novel_relevant_count": novel_count,
        "gate_lost_relevant_count": gate_lost,
        "rrf_ndcg10": float(rrf["ndcg@10"]),
        "dual_ndcg10": float(dual["ndcg@10"]),
        "dual_novel_retention1000": float(dual["novel_retention@1000"]),
        "cascade_mechanical_guards_pass": (
            cascade_decision.get("mechanical_guards_pass")
            if isinstance(cascade_decision, Mapping) else None
        ),
        "cascade_graded_recall500": (
            float(cascade["graded_recall@500"])
            if cascade is not None else None
        ),
    }
    return artifact, report_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--evaluation-summary", required=True, type=Path)
    parser.add_argument("--advisor", required=True, type=Path)
    parser.add_argument("--retrieval-summary", required=True, type=Path)
    parser.add_argument("--phase1-receipt", required=True, type=Path)
    parser.add_argument("--phase2-receipt", required=True, type=Path)
    parser.add_argument("--cascade-metrics", type=Path)
    parser.add_argument("--cascade-decision", type=Path)
    parser.add_argument("--cascade-summary", type=Path)
    parser.add_argument("--cascade-advisor", type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    metrics, decision, evaluation_summary = load_verified_evaluation(
        args.metrics, args.decision, args.evaluation_summary
    )
    advisor = args.advisor.read_text(encoding="utf-8")
    retrieval = _read_object(args.retrieval_summary, "retrieval summary")
    phase1 = _read_object(args.phase1_receipt, "phase-1 receipt")
    phase2 = _read_object(args.phase2_receipt, "phase-2 receipt")
    cascade_paths = (
        args.cascade_metrics,
        args.cascade_decision,
        args.cascade_summary,
        args.cascade_advisor,
    )
    if any(path is not None for path in cascade_paths) and not all(
        path is not None for path in cascade_paths
    ):
        raise ValueError("all cascade report inputs must be provided together")
    cascade_metrics = cascade_decision = None
    cascade_advisor = ""
    if all(path is not None for path in cascade_paths):
        cascade_metrics, cascade_decision, _cascade_summary = load_verified_cascade(
            args.cascade_metrics,
            args.cascade_decision,
            args.cascade_summary,
        )
        cascade_advisor = args.cascade_advisor.read_text(encoding="utf-8")
    if (
        retrieval.get("complete") is not True
        or retrieval.get("qrels_opened") is not False
        or phase1.get("status") != "complete"
        or phase1.get("qrels_opened") is not False
        or phase2.get("status") != "complete"
        or phase2.get("qrels_opened") is not False
    ):
        raise ValueError("runtime receipts are incomplete or not qrels-blind")
    artifact, summary = build_artifact(
        metrics,
        decision,
        evaluation_summary,
        advisor_memo=advisor,
        runtime_evidence={
            "external_calls": retrieval["external_calls"],
            "retry_count": retrieval["retry_count"],
            "phase1_seconds": phase1["elapsed_seconds"],
            "phase2_seconds": phase2["elapsed_seconds"],
            "peak_device_memory_bytes": max(
                int(phase1["peak_device_memory_bytes"]),
                int(phase2["peak_device_memory_bytes"]),
            ),
        },
        cascade_metrics=cascade_metrics,
        cascade_decision=cascade_decision,
        cascade_advisor_memo=cascade_advisor,
    )
    datasets = artifact["snapshot"]["datasets"]
    assert isinstance(datasets, Mapping)
    write_report_database(args.artifact.parent / "report_data.sqlite", datasets)  # type: ignore[arg-type]
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

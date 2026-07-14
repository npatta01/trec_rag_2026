from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.build_deep_facet_candidate_report import (
    build_artifact,
    load_verified_evaluation,
)


def _evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    aggregate = {
        arm: {
            "ndcg@10": value,
            "ndcg@100": value / 1.2,
            "graded_recall@500": value / 2,
            "graded_recall@1000": value / 1.5,
            "novel_retention@500": value,
            "novel_retention@1000": min(1.0, value + 0.2),
            "judged_rate@100": value,
            "near_duplicate_rate@500": value / 10,
        }
        for arm, value in {
            "RRF": 0.43,
            "GLOBAL": 0.36,
            "FACET": 0.12,
            "DUAL": 0.33,
            "DUAL-NR": 0.34,
        }.items()
    }
    discovery = {
        topic: {
            "original": {"relevant_count": 10},
            "U_raw": {"relevant_count": 15},
            "U_accepted": {"relevant_count": 14},
            "novel_relevant_ids": [f"{topic}-novel"],
            "gate_lost_relevant_ids": [f"{topic}-lost"],
        }
        for topic in ("219", "72", "300", "84")
    }
    per_topic = {
        topic: {
            "DUAL_ndcg@10_delta_vs_RRF": -0.03,
            **{arm: {"ndcg@10": row["ndcg@10"], "graded_recall@500": row["graded_recall@500"], "novel_retained@500": 1, "novel_retained@1000": 1} for arm, row in aggregate.items()},
        }
        for topic in discovery
    }
    metrics = {
        "novel_relevant_count": 177,
        "novel_relevant_topic_count": 4,
        "gate_lost_relevant_count": 2,
        "aggregate": aggregate,
        "discovery": discovery,
        "per_topic": per_topic,
    }
    decision = {
        "diagnosis": "fusion",
        "advance_to_larger_validation": False,
        "failed_guards": ["dual_ndcg_10_within_002_rrf"],
    }
    summary = {"status": "complete", "qrels_opened": True}
    return metrics, decision, summary


def test_artifact_is_answer_first_and_separates_failure_stages() -> None:
    metrics, decision, summary = _evidence()
    artifact, report_summary = build_artifact(
        metrics,
        decision,
        summary,
        advisor_memo="Advisor: preserve candidates.",
        runtime_evidence={
            "external_calls": 25,
            "retry_count": 0,
            "phase1_seconds": 113.7,
            "phase2_seconds": 456.0,
            "peak_device_memory_bytes": 489_715_712,
        },
    )
    encoded = json.dumps(artifact)
    blocks = artifact["manifest"]["blocks"]
    assert blocks[0]["body"].startswith("# Deep facets find evidence")
    assert "Technical summary" in blocks[1]["body"]
    assert "U_raw" in encoded and "U_accepted" in encoded
    assert "retrieval" in encoded and "gating" in encoded and "fusion" in encoded
    assert "569.7" in encoded and "25" in encoded
    assert report_summary["diagnosis"] == "fusion"
    assert report_summary["novel_relevant_count"] == 177


def test_artifact_has_native_charts_tables_sources_and_ready_snapshot() -> None:
    metrics, decision, summary = _evidence()
    artifact, _ = build_artifact(metrics, decision, summary, advisor_memo="review")
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["manifest"]["charts"]) >= 4
    assert len(artifact["manifest"]["tables"]) >= 2
    assert all(chart.get("sourceId") for chart in artifact["manifest"]["charts"])
    assert all(table.get("sourceId") for table in artifact["manifest"]["tables"])
    sources = {row["id"]: row for row in artifact["sources"]}
    for item in [
        *artifact["manifest"]["cards"],
        *artifact["manifest"]["charts"],
        *artifact["manifest"]["tables"],
    ]:
        assert sources[item["sourceId"]]["query"]["sql"].startswith("SELECT")


def test_mutated_evaluation_source_is_rejected(tmp_path: Path) -> None:
    metrics, decision, summary = _evidence()
    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.json"
    summary_path = tmp_path / "summary.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    summary.update(
        {
            "metrics_sha256": __import__("hashlib").sha256(metrics_path.read_bytes()).hexdigest(),
            "decision_sha256": __import__("hashlib").sha256(decision_path.read_bytes()).hexdigest(),
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metrics_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_verified_evaluation(metrics_path, decision_path, summary_path)

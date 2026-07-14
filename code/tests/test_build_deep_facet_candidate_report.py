from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.build_deep_facet_candidate_report import (
    build_artifact,
    load_verified_cascade,
    load_verified_evaluation,
    load_verified_mixedbread,
)


def _evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    aggregate = {
        arm: {
            "ndcg@10": value,
            "ndcg@100": value / 1.2,
            "graded_recall@500": value / 2,
            "recall@500": value / 2.1,
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


def test_artifact_leads_with_measured_cascade_failure_when_supplied() -> None:
    metrics, decision, summary = _evidence()
    per_topic = {
        topic: {
            "ndcg@10": metrics["per_topic"][topic]["RRF"]["ndcg@10"],
            "graded_recall@500": 0.18,
            "graded_recall@500_delta_vs_RRF": -0.03,
            "graded_recall@1000": 0.28,
            "novel_retained@500": 20,
            "novel_retained@1000": 38,
            "judged_rate@100": 0.47,
            "judged_rate@500": 0.29,
            "judged_rate@1000": 0.23,
        }
        for topic in ("219", "72", "300", "84")
    }
    cascade_metrics = {
        "post_qrels_diagnostic": True,
        "aggregate": {
            "RRF-GLOBAL-DUAL": {
                "ndcg@10": 0.43,
                "ndcg@100": 0.27,
                "graded_recall@500": 0.187,
                "graded_recall@1000": 0.280,
                "novel_retention@500": 114 / 177,
                "novel_retention@1000": 155 / 177,
                "judged_rate@100": 0.47,
                "judged_rate@500": 0.2875,
            }
        },
        "per_topic": per_topic,
    }
    cascade_decision = {
        "mechanical_guards_pass": False,
        "advance_to_fresh_validation": False,
        "failed_guards": ["graded_recall_500_beats_rrf"],
        "guards": {
            "ndcg_10_exactly_preserves_rrf": True,
            "graded_recall_500_beats_rrf": False,
        },
        "coverage_review_status": "required",
    }

    artifact, report_summary = build_artifact(
        metrics,
        decision,
        summary,
        advisor_memo="Initial advisor review.",
        cascade_metrics=cascade_metrics,
        cascade_decision=cascade_decision,
        cascade_advisor_memo="Cascade advisor: stronger reranking is warranted.",
    )

    encoded = json.dumps(artifact)
    assert "RRF-GLOBAL-DUAL" in encoded
    assert "did not fix" in encoded
    assert "post-qrels diagnostic" in encoded
    assert "stronger reranking is warranted" in encoded
    assert "cascade_guards" in artifact["snapshot"]["datasets"]
    assert len(artifact["snapshot"]["datasets"]["cascade_guards"]) == 2
    assert any(table["id"] == "cascade_guard_table" for table in artifact["manifest"]["tables"])
    assert any(block.get("tableId") == "cascade_guard_table" for block in artifact["manifest"]["blocks"])
    assert report_summary["cascade_mechanical_guards_pass"] is False
    assert report_summary["cascade_graded_recall500"] == pytest.approx(0.187)


def test_mutated_cascade_source_is_rejected(tmp_path: Path) -> None:
    import hashlib

    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.json"
    summary_path = tmp_path / "summary.json"
    metrics_path.write_text('{"post_qrels_diagnostic":true}\n', encoding="utf-8")
    decision_path.write_text('{"mechanical_guards_pass":false}\n', encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "post_qrels_diagnostic": True,
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cascade"):
        load_verified_cascade(metrics_path, decision_path, summary_path)


def _mixedbread_evidence() -> tuple[dict[str, object], dict[str, object]]:
    per_topic = {
        topic: {
            "candidate_count": count,
            "ndcg@10": 0.43,
            "ndcg@100": 0.40,
            "graded_recall@100": 0.13,
            "graded_recall@500": 0.24,
            "graded_recall@500_delta_vs_RRF": 0.03,
            "graded_recall@1000": 0.30,
            "graded_recall@full": 0.36,
            "judged_rate@100": 0.55,
            "judged_rate@500": 0.32,
            "judged_rate@1000": 0.24,
            "judged_rate@full": 0.18,
            "recall@100": 0.12,
            "recall@500": 0.22,
            "recall@1000": 0.28,
            "recall@full": 0.34,
            "novel_retained@100": 5,
            "novel_retained@500": 35,
            "novel_retained@1000": 42,
            "novel_retained@full": 44,
            "novel_retention@100": 0.1,
            "novel_retention@500": 0.75,
            "novel_retention@1000": 0.9,
            "novel_retention@full": 1.0,
        }
        for topic, count in {"219": 2182, "72": 2127, "300": 1712, "84": 2093}.items()
    }
    aggregate = {
        "ndcg@10": 0.43,
        "ndcg@100": 0.40,
        "graded_recall@100": 0.13,
        "graded_recall@500": 0.24,
        "recall@500": 0.22,
        "graded_recall@1000": 0.30,
        "graded_recall@full": 0.36,
        "judged_rate@100": 0.55,
        "judged_rate@500": 0.32,
        "judged_rate@1000": 0.24,
        "judged_rate@full": 0.18,
        "novel_retained@100": 20,
        "novel_retained@500": 140,
        "novel_retained@1000": 165,
        "novel_retained@full": 177,
        "novel_retention@100": 20 / 177,
        "novel_retention@500": 140 / 177,
        "novel_retention@1000": 165 / 177,
        "novel_retention@full": 1.0,
    }
    metrics = {
        "status": "complete",
        "post_qrels_diagnostic": True,
        "confirmatory_evidence": False,
        "aggregate": {"RRF-MIXEDBREAD-DUAL": aggregate},
        "per_topic": per_topic,
        "interpretation": {"complete_ranking": True},
    }
    receipt = {
        "status": "complete",
        "qrels_read": False,
        "window_count": 36435,
        "elapsed_seconds": 4200.0,
        "peak_device_memory_bytes": 2_700_000_000,
        "paid_cost_usd": 0,
    }
    return metrics, receipt


def test_artifact_explains_mixedbread_result_as_complete_rag_ranking() -> None:
    metrics, decision, summary = _evidence()
    cascade_metrics = {
        "post_qrels_diagnostic": True,
        "aggregate": {
            "RRF-GLOBAL-DUAL": {
                "ndcg@10": 0.43,
                "ndcg@100": 0.27,
                "graded_recall@500": 0.187,
                "graded_recall@1000": 0.280,
                "novel_retention@500": 0.64,
                "novel_retention@1000": 0.88,
                "judged_rate@100": 0.47,
                "judged_rate@500": 0.29,
            }
        },
        "per_topic": {
            topic: {
                "ndcg@10": 0.43,
                "graded_recall@500": 0.18,
                "graded_recall@500_delta_vs_RRF": -0.03,
            }
            for topic in ("219", "72", "300", "84")
        },
    }
    mixedbread, receipt = _mixedbread_evidence()

    artifact, report_summary = build_artifact(
        metrics,
        decision,
        summary,
        advisor_memo="review",
        cascade_metrics=cascade_metrics,
        cascade_decision={"mechanical_guards_pass": False, "guards": {"middle": False}},
        mixedbread_metrics=mixedbread,
        mixedbread_decision={
            "status": "complete",
            "mechanical_guards_pass": True,
            "advance_to_fresh_validation": False,
            "production_promotion_authorized": False,
        },
        mixedbread_receipt=receipt,
        mixedbread_advisor_memo="The complete ranking is suitable for fresh validation.",
    )

    encoded = json.dumps(artifact)
    assert "RRF-MIXEDBREAD-DUAL" in encoded
    assert "8,114" in encoded
    assert "no document is discarded" in encoded.lower()
    assert "36,435" in encoded
    assert "fresh validation" in encoded
    assert report_summary["mixedbread_complete_candidate_count"] == 8114


def test_mutated_mixedbread_source_is_rejected(tmp_path: Path) -> None:
    import hashlib

    metrics, receipt = _mixedbread_evidence()
    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.json"
    summary_path = tmp_path / "summary.json"
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    metrics["source"] = {
        "scoring_receipt": {
            "bytes": receipt_path.stat().st_size,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        }
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    decision_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "advance_to_fresh_validation": False,
                "production_promotion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "post_qrels_diagnostic": True,
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Mixedbread"):
        load_verified_mixedbread(metrics_path, decision_path, summary_path, receipt_path)

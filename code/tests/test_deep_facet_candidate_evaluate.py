from __future__ import annotations

from pathlib import Path

import pytest

from trec_rag.deep_facet_candidate_evaluate import (
    create_qrels_sentinel,
    decide,
    evaluate,
    evaluate_ranking,
    evaluate_set,
)


def test_unsealed_freeze_cannot_open_qrels(tmp_path: Path) -> None:
    calls: list[int] = []
    with pytest.raises(ValueError, match="seal"):
        evaluate(
            tmp_path / "freeze",
            tmp_path / "evaluation",
            qrels_loader=lambda: calls.append(1) or {},
        )
    assert calls == []


def test_qrels_sentinel_is_atomic_and_create_only(tmp_path: Path) -> None:
    receipt = create_qrels_sentinel(tmp_path, seal_sha256="a" * 64)
    assert receipt["qrels_opened"] is True
    assert (tmp_path / "QRELS_ACCESSED").exists()
    with pytest.raises(FileExistsError):
        create_qrels_sentinel(tmp_path, seal_sha256="a" * 64)


def test_set_and_ranking_metrics_use_grade_two_threshold() -> None:
    qrels = {"a": 3, "b": 2, "c": 1, "d": 0}
    set_metrics = evaluate_set(["a", "c"], qrels)
    assert set_metrics["relevant_count"] == 1
    assert set_metrics["recall"] == pytest.approx(0.5)
    assert set_metrics["graded_recall"] == pytest.approx(7 / 10)
    assert set_metrics["judged_rate"] == 1.0

    ranked = evaluate_ranking(["b", "x", "a", "c"], qrels, depths=(2, 4))
    assert ranked["2"]["recall"] == pytest.approx(0.5)
    assert ranked["4"]["recall"] == 1.0
    assert ranked["2"]["judged_rate"] == 0.5
    assert ranked["ndcg@10"] is not None


def test_advance_aggregation_and_guards() -> None:
    fixture = {
        "novel_relevant_count": 6,
        "novel_relevant_topic_count": 2,
        "aggregate": {
            "RRF": {"graded_recall@500": 0.40, "graded_recall@1000": 0.60, "ndcg@10": 0.50},
            "GLOBAL": {"graded_recall@500": 0.42, "graded_recall@1000": 0.61, "ndcg@10": 0.49},
            "DUAL": {"graded_recall@500": 0.50, "graded_recall@1000": 0.62, "ndcg@10": 0.49, "novel_retention@1000": 0.66},
        },
        "per_topic": {
            topic: {"DUAL_ndcg@10_delta_vs_RRF": -0.01}
            for topic in ("219", "72", "300", "84")
        },
        "leave_one_out": {
            topic: {"dual_graded_recall_500_beats_rrf": True, "dual_graded_recall_500_beats_global": True, "dual_ndcg_10_within_002_rrf": True}
            for topic in ("219", "72", "300", "84")
        },
    }
    result = decide(fixture)
    assert result["aggregation"]["graded_recall"] == "macro_non_null"
    assert result["aggregation"]["novel_retention"] == "micro_topic_document"
    assert result["advance_to_larger_validation"] is True


def test_failed_leave_one_out_stops_advance() -> None:
    fixture = {
        "novel_relevant_count": 6,
        "novel_relevant_topic_count": 2,
        "aggregate": {
            "RRF": {"graded_recall@500": 0.40, "graded_recall@1000": 0.60, "ndcg@10": 0.50},
            "GLOBAL": {"graded_recall@500": 0.42},
            "DUAL": {"graded_recall@500": 0.50, "graded_recall@1000": 0.62, "ndcg@10": 0.49, "novel_retention@1000": 0.66},
        },
        "per_topic": {
            topic: {"DUAL_ndcg@10_delta_vs_RRF": -0.01}
            for topic in ("219", "72", "300", "84")
        },
        "leave_one_out": {
            "219": {"dual_graded_recall_500_beats_rrf": False, "dual_graded_recall_500_beats_global": True, "dual_ndcg_10_within_002_rrf": True}
        },
    }
    result = decide(fixture)
    assert result["advance_to_larger_validation"] is False
    assert "leave_one_out_stability" in result["failed_guards"]

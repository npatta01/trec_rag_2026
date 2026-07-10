import csv
import json

import pytest

from trec_rag.pipeline_comparison import (
    _validate_candidate_pools,
    build_metric_comparison,
    write_comparison_outputs,
)
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.topics import Topic


def evaluation_payload(topic_values):
    metrics = {
        metric: sum(row[metric] for row in topic_values.values()) / len(topic_values)
        for metric in next(iter(topic_values.values()))
    }
    return {"metrics": metrics, "per_topic": topic_values}


def test_build_metric_comparison_tracks_topic_gains_losses_and_ties():
    baseline = evaluation_payload(
        {
            "1": {"ndcg@10": 0.2, "judged_rate@10": 1.0},
            "2": {"ndcg@10": 0.7, "judged_rate@10": 1.0},
            "3": {"ndcg@10": 0.4, "judged_rate@10": 0.8},
        }
    )
    candidate = evaluation_payload(
        {
            "1": {"ndcg@10": 0.5, "judged_rate@10": 1.0},
            "2": {"ndcg@10": 0.55, "judged_rate@10": 0.9},
            "3": {"ndcg@10": 0.4, "judged_rate@10": 0.8},
        }
    )

    comparison = build_metric_comparison(
        baseline,
        candidate,
        topics=[
            Topic("1", "First", "First topic"),
            Topic("2", "Second", "Second topic"),
            Topic("3", "Third", "Third topic"),
        ],
        metric_names=["ndcg@10", "judged_rate@10"],
        primary_metric="ndcg@10",
        baseline_id="bm25",
        candidate_id="reranker",
        big_regression_threshold=0.1,
    )

    assert comparison["primary_metric_summary"]["topic_counts"] == {
        "improved": 1,
        "degraded": 1,
        "tied": 1,
        "big_regressions": 1,
    }
    assert comparison["aggregate"]["ndcg@10"]["delta"] == pytest.approx(0.05)
    assert [row["primary_status"] for row in comparison["topics"]] == [
        "improved",
        "degraded",
        "tied",
    ]
    assert comparison["topics"][1]["is_big_regression"] is True


def test_write_comparison_outputs_writes_wide_and_long_topic_tables(tmp_path):
    baseline = evaluation_payload({"14": {"ndcg@10": 0.3, "judged_rate@10": 1.0}})
    candidate = evaluation_payload({"14": {"ndcg@10": 0.4, "judged_rate@10": 0.9}})
    comparison = build_metric_comparison(
        baseline,
        candidate,
        topics=[Topic("14", "Sports societal impact", "Narrative")],
        metric_names=["ndcg@10", "judged_rate@10"],
        primary_metric="ndcg@10",
        baseline_id="bm25",
        candidate_id="reranker",
    )

    write_comparison_outputs(comparison, tmp_path)

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    wide = list(csv.DictReader((tmp_path / "topic_metrics.csv").open()))
    long_rows = list(csv.DictReader((tmp_path / "topic_metric_deltas.csv").open()))
    assert saved["topic_count"] == 1
    assert wide[0]["topic_title"] == "Sports societal impact"
    assert float(wide[0]["delta_ndcg_at_10"]) == pytest.approx(0.1)
    assert [row["metric"] for row in long_rows] == ["ndcg@10", "judged_rate@10"]
    assert long_rows[0]["is_primary_metric"] == "True"


def test_candidate_pool_validation_rejects_same_docs_in_different_order():
    baseline = [
        RetrievedCandidate("14", "original", "bm25", "query", "doc-a", 1, 2.0, "A"),
        RetrievedCandidate("14", "original", "bm25", "query", "doc-b", 2, 1.0, "B"),
    ]
    candidate = [
        RetrievedCandidate("14", "original", "bm25", "query", "doc-b", 1, 2.0, "B"),
        RetrievedCandidate("14", "original", "bm25", "query", "doc-a", 2, 1.0, "A"),
    ]

    with pytest.raises(ValueError, match="candidate pools differ"):
        _validate_candidate_pools(baseline, candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        RetrievedCandidate("14", "original", "bm25", "different query", "doc-a", 1, 2.0, "A"),
        RetrievedCandidate("14", "original", "bm25", "query", "doc-a", 1, 2.0, "Changed A"),
    ],
)
def test_candidate_pool_validation_rejects_query_or_document_text_changes(candidate):
    baseline = [
        RetrievedCandidate("14", "original", "bm25", "query", "doc-a", 1, 2.0, "A")
    ]

    with pytest.raises(ValueError, match="candidate pools differ"):
        _validate_candidate_pools(baseline, [candidate])

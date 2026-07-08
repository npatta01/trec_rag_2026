import csv
import json

from trec_rag.cross_encoder_comparison import (
    collect_full_dev_rows,
    collect_prompt_probe_rows,
    write_comparison_report,
)


def test_collect_full_dev_rows_computes_regression_summary(tmp_path):
    artifact_path = tmp_path / "reranker.json"
    artifact_path.write_text(
        json.dumps(
            {
                "model": "example/model",
                "hits": 50,
                "max_length": 1024,
                "document_count": 3,
                "metrics": {
                    "bm25": {
                        "metrics": {"ndcg@10": 0.4},
                        "per_topic": {
                            "1": {"ndcg@10": 0.5},
                            "2": {"ndcg@10": 0.3},
                            "3": {"ndcg@10": 0.4},
                        },
                    },
                    "reranked": {
                        "metrics": {"ndcg@10": 0.5},
                        "per_topic": {
                            "1": {"ndcg@10": 0.7},
                            "2": {"ndcg@10": 0.1},
                            "3": {"ndcg@10": 0.4},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rows, topic_rows = collect_full_dev_rows([artifact_path])

    assert rows == [
        {
            "system_id": "example_model__reranked__reranker",
            "model": "example/model",
            "method": "reranked",
            "candidate_depth": "50",
            "input_limit": "1024",
            "document_count": "3",
            "chunk_count": "",
            "fit_count": "",
            "truncated_count": "",
            "ndcg_at_10": "0.5000000000",
            "delta_vs_bm25": "0.1000000000",
            "relative_lift_vs_bm25": "0.2500000000",
            "topic_losses": "1",
            "big_topic_losses": "1",
            "worst_topic": "2",
            "worst_delta": "-0.2000000000",
            "source_artifact": str(artifact_path),
            "notes": "",
        }
    ]
    assert topic_rows[1]["topic_id"] == "2"
    assert topic_rows[1]["delta_vs_bm25"] == "-0.2000000000"


def test_write_comparison_report_creates_summary_files(tmp_path):
    artifact_path = tmp_path / "reranker.json"
    artifact_path.write_text(
        json.dumps(
            {
                "model": "example/model",
                "hits": 50,
                "metrics": {
                    "bm25": {
                        "metrics": {"ndcg@10": 0.4},
                        "per_topic": {"1": {"ndcg@10": 0.4}},
                    },
                    "reranked": {
                        "metrics": {"ndcg@10": 0.6},
                        "per_topic": {"1": {"ndcg@10": 0.6}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "report"
    write_comparison_report([artifact_path], output_dir)

    system_rows = list(csv.DictReader((output_dir / "system_scores.csv").open()))
    topic_rows = list(csv.DictReader((output_dir / "topic_system_scores.csv").open()))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    notes = (output_dir / "notes.md").read_text(encoding="utf-8")

    assert system_rows[0]["system_id"] == "example_model__reranked__reranker"
    assert topic_rows[0]["system_id"] == "example_model__reranked__reranker"
    assert metrics["baseline"]["ndcg@10"] == 0.4
    assert metrics["best_full_dev_system"]["ndcg@10"] == 0.6
    assert "example/model" in notes


def test_collect_prompt_probe_rows_summarizes_rank_alignment(tmp_path):
    probe_path = tmp_path / "prompt_probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model": "Qwen/example",
                        "instructions": {
                            "default": [
                                {"topic_id": "1", "qrel_grade": 3, "score": 0.9},
                                {"topic_id": "1", "qrel_grade": 0, "score": 0.8},
                                {"topic_id": "2", "qrel_grade": 0, "score": 0.6},
                                {"topic_id": "2", "qrel_grade": 4, "score": 0.1},
                            ],
                            "strict": [
                                {"topic_id": "1", "qrel_grade": 3, "score": 0.9},
                                {"topic_id": "1", "qrel_grade": 0, "score": 0.1},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = collect_prompt_probe_rows(probe_path)

    assert rows == [
        {
            "model": "Qwen/example",
            "instruction": "default",
            "topic_count": "2",
            "document_count": "4",
            "mean_spearman": "0.0000000000",
            "worst_topic": "2",
            "worst_spearman": "-1.0000000000",
            "source_artifact": str(probe_path),
        },
        {
            "model": "Qwen/example",
            "instruction": "strict",
            "topic_count": "1",
            "document_count": "2",
            "mean_spearman": "1.0000000000",
            "worst_topic": "1",
            "worst_spearman": "1.0000000000",
            "source_artifact": str(probe_path),
        },
    ]

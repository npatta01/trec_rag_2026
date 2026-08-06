from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trec_rag.reranker_depth_evaluation import (
    evaluate_max_passage_systems,
    load_window_artifact,
    paired_bootstrap_interval,
    write_rankings,
    write_evaluation,
)


IDENTITY_FIELDS = {
    "artifact_schema_version": 2,
    "backend": "sentence-transformers-cross-encoder",
    "backend_version": "5.6.0",
    "model_revision": "revision-a",
    "score_representation": "raw_logits",
    "inference_dtype": "bfloat16",
    "input_policy": "trec_rag_raw_v2",
    "max_length": 1024,
    "score_kind": "window",
    "chunk_max_characters": 3500,
    "chunk_overlap_characters": 350,
}


def _window_rows(model: str, scores: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topic_id in ("1", "2"):
        for rank, docid in enumerate(("d1", "d2"), start=1):
            values = scores[f"{topic_id}:{docid}"]
            for chunk_index, score in enumerate(values):
                rows.append(
                    {
                        **IDENTITY_FIELDS,
                        "model": model,
                        "topic_id": topic_id,
                        "docid": docid,
                        "rank": rank,
                        "chunk_index": chunk_index,
                        "chunk_count": len(values),
                        "chunk_id": f"{docid}:{chunk_index:04d}",
                        "start_char": chunk_index * 10,
                        "end_char": chunk_index * 10 + 10,
                        "score": score,
                        "query_sha256": f"query-{topic_id}",
                        "document_text_sha256": f"text-{topic_id}-{docid}",
                        "text_sha256": f"chunk-{topic_id}-{docid}-{chunk_index}",
                        "score_cache_key": f"key-{model}-{topic_id}-{docid}-{chunk_index}",
                    }
                )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_qrels(path: Path) -> None:
    path.write_text(
        "1 0 d1 0\n"
        "1 0 d2 3\n"
        "2 0 d1 3\n"
        "2 0 d2 0\n",
        encoding="utf-8",
    )


def test_load_window_artifact_aggregates_max_and_validates_population(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    _write_jsonl(
        path,
        _window_rows(
            "example/a",
            {
                "1:d1": [0.1, 0.7],
                "1:d2": [0.6],
                "2:d1": [0.9],
                "2:d2": [0.2, 0.3],
            },
        ),
    )

    artifact = load_window_artifact(path, expected_depth=2, expected_topics=("1", "2"))

    assert artifact.model == "example/a"
    assert artifact.topic_ids == ("1", "2")
    assert artifact.documents["1"]["d1"].max_score == 0.7
    assert artifact.documents["1"]["d1"].max_chunk_index == 1
    assert artifact.documents["1"]["d1"].chunk_scores == (0.1, 0.7)
    assert artifact.documents["1"]["d1"].chunk_count == 2
    assert artifact.total_documents == 4
    assert artifact.total_chunks == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(1), "incomplete chunks"),
        (
            lambda rows: rows.__setitem__(
                slice(0, 2), [{**row, "rank": 2} for row in rows[0:2]]
            ),
            "BM25 ranks",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "model_revision": "other"}),
            "multiple artifact identities",
        ),
        (lambda rows: rows.append(dict(rows[0])), "duplicate chunk"),
    ],
)
def test_load_window_artifact_rejects_incomplete_or_conflicting_rows(
    tmp_path: Path, mutation, message: str
) -> None:
    rows = _window_rows(
        "example/a",
        {"1:d1": [0.1, 0.7], "1:d2": [0.6], "2:d1": [0.9], "2:d2": [0.2]},
    )
    mutation(rows)
    path = tmp_path / "scores.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        load_window_artifact(path, expected_depth=2, expected_topics=("1", "2"))


def test_evaluate_max_passage_systems_uses_same_candidates_and_repository_ndcg(
    tmp_path: Path,
) -> None:
    qrels = tmp_path / "qrels.txt"
    _write_qrels(qrels)
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    _write_jsonl(
        a_path,
        _window_rows(
            "example/a",
            {"1:d1": [0.9], "1:d2": [0.1], "2:d1": [0.8], "2:d2": [0.2]},
        ),
    )
    _write_jsonl(
        b_path,
        _window_rows(
            "example/b",
            {"1:d1": [0.1], "1:d2": [0.9], "2:d1": [0.2], "2:d2": [0.8]},
        ),
    )
    a = load_window_artifact(a_path, expected_depth=2, expected_topics=("1", "2"))
    b = load_window_artifact(b_path, expected_depth=2, expected_topics=("1", "2"))

    report = evaluate_max_passage_systems(
        {"a": a, "b": b},
        qrels_path=qrels,
        candidate_depths=(2,),
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    depth = report["depths"]["2"]
    assert depth["baseline"]["metrics"]["ndcg@10"] == pytest.approx(0.8154648768)
    assert depth["systems"]["a"]["metrics"]["ndcg@10"] == pytest.approx(0.8154648768)
    assert depth["systems"]["b"]["metrics"]["ndcg@10"] == pytest.approx(0.8154648768)
    assert depth["systems"]["a"]["metrics"]["precision@10"] == pytest.approx(0.1)
    assert depth["systems"]["b"]["metrics"]["precision@20"] == pytest.approx(0.05)
    assert depth["comparison"]["top_k_overlap"]["1"]["mean_overlap_fraction"] == 0.0
    assert depth["comparison"]["topic_wins"] == {"a": 1, "b": 1, "ties": 0}
    assert set(depth["comparison"]["per_topic_delta"]) == {"1", "2"}


def test_evaluation_rejects_cross_model_candidate_identity_mismatch(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.txt"
    _write_qrels(qrels)
    rows_a = _window_rows(
        "example/a",
        {"1:d1": [0.1], "1:d2": [0.2], "2:d1": [0.3], "2:d2": [0.4]},
    )
    rows_b = _window_rows(
        "example/b",
        {"1:d1": [0.1], "1:d2": [0.2], "2:d1": [0.3], "2:d2": [0.4]},
    )
    rows_b[0]["document_text_sha256"] = "different"
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a_path, rows_a)
    _write_jsonl(b_path, rows_b)

    with pytest.raises(ValueError, match="candidate identity mismatch"):
        evaluate_max_passage_systems(
            {
                "a": load_window_artifact(a_path, expected_depth=2),
                "b": load_window_artifact(b_path, expected_depth=2),
            },
            qrels_path=qrels,
            candidate_depths=(2,),
        )


def test_paired_bootstrap_constant_delta_has_exact_interval() -> None:
    assert paired_bootstrap_interval([0.125, 0.125, 0.125], samples=100, seed=9) == {
        "mean": 0.125,
        "lower_95": 0.125,
        "upper_95": 0.125,
        "samples": 100,
        "seed": 9,
    }


def test_write_evaluation_emits_metrics_and_topic_csv(tmp_path: Path) -> None:
    report = {
        "schema_version": "reranker-depth-evaluation-v1",
        "depths": {
            "2": {
                "comparison": {
                    "per_topic_delta": {
                        "1": {"a": 0.2, "b": 0.4, "b_minus_a": 0.2},
                        "2": {"a": 0.3, "b": 0.1, "b_minus_a": -0.2},
                    }
                }
            }
        },
    }

    write_evaluation(report, tmp_path / "out")

    assert json.loads((tmp_path / "out/metrics.json").read_text())["schema_version"] == (
        "reranker-depth-evaluation-v1"
    )
    rows = list(csv.DictReader((tmp_path / "out/topic_metrics.csv").open()))
    assert rows == [
        {"depth": "2", "topic_id": "1", "a": "0.2", "b": "0.4", "b_minus_a": "0.2"},
        {"depth": "2", "topic_id": "2", "a": "0.3", "b": "0.1", "b_minus_a": "-0.2"},
    ]


def test_write_rankings_emits_document_and_passage_rows(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    _write_jsonl(
        path,
        _window_rows(
            "example/a",
            {
                "1:d1": [0.1, 0.7],
                "1:d2": [0.6],
                "2:d1": [0.9],
                "2:d2": [0.2, 0.3],
            },
        ),
    )
    artifact = load_window_artifact(path, expected_depth=2)

    write_rankings({"a": artifact}, tmp_path / "out")

    document_rows = [
        json.loads(line)
        for line in (tmp_path / "out/a_document_rankings.jsonl").read_text().splitlines()
    ]
    passage_rows = [
        json.loads(line)
        for line in (tmp_path / "out/a_passage_rankings.jsonl").read_text().splitlines()
    ]
    assert document_rows[0] == {
        "bm25_rank": 1,
        "chunk_count": 2,
        "docid": "d1",
        "document_text_sha256": "text-1-d1",
        "max_passage_score": 0.7,
        "query_sha256": "query-1",
        "rerank_rank": 1,
        "system": "a",
        "topic_id": "1",
        "winning_chunk_index": 1,
    }
    assert passage_rows[:3] == [
        {
            "bm25_rank": 1,
            "chunk_count": 2,
            "chunk_index": 1,
            "docid": "d1",
            "is_document_max": True,
            "passage_rank": 1,
            "score": 0.7,
            "system": "a",
            "topic_id": "1",
        },
        {
            "bm25_rank": 2,
            "chunk_count": 1,
            "chunk_index": 0,
            "docid": "d2",
            "is_document_max": True,
            "passage_rank": 2,
            "score": 0.6,
            "system": "a",
            "topic_id": "1",
        },
        {
            "bm25_rank": 1,
            "chunk_count": 2,
            "chunk_index": 0,
            "docid": "d1",
            "is_document_max": False,
            "passage_rank": 3,
            "score": 0.1,
            "system": "a",
            "topic_id": "1",
        },
    ]

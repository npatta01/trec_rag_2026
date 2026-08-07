from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from trec_rag.competition_retrieval_evaluation import (
    evaluate_competition_retrieval_run,
    parse_trec_retrieval_run,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_trec_retrieval_run_accepts_strict_six_column_rows(tmp_path: Path) -> None:
    run_path = _write(
        tmp_path / "run.trec",
        "31 Q0 climb-1 1 9.5 fixed-run\n"
        "31 Q0 climb-2 2 -1.25 fixed-run\n"
        "37 Q0 climb-3 1 0 fixed-run\n",
    )

    ranked = parse_trec_retrieval_run(run_path, expected_topic_ids=("31", "37"))

    assert [
        (row.topic_id, row.docid, row.rank, row.score, row.text, row.provenance)
        for row in ranked
    ] == [
        ("31", "climb-1", 1, 9.5, "", [{"run_id": "fixed-run"}]),
        ("31", "climb-2", 2, -1.25, "", [{"run_id": "fixed-run"}]),
        ("37", "climb-3", 1, 0.0, "", [{"run_id": "fixed-run"}]),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("31 Q0 doc-a 2 2 run\n", "expected 1, got 2"),
        (
            "31 Q0 doc-a 1 3 run\n31 Q0 doc-b 3 2 run\n",
            "expected 2, got 3",
        ),
        (
            "31 Q0 doc-a 1 3 run\n31 Q0 doc-b 2 2 run\n31 Q0 doc-c 1 1 run\n",
            "expected 3, got 1",
        ),
        (
            "31 Q0 doc-a 1 2 run\n31 Q0 doc-b 2 3 run\n",
            "scores must be non-increasing",
        ),
    ],
)
def test_parse_trec_retrieval_run_requires_dense_rank_and_score_order(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    run_path = _write(tmp_path / "run.trec", body)

    with pytest.raises(ValueError, match=message):
        parse_trec_retrieval_run(run_path, expected_topic_ids=("31",))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("31 Q0 doc 1 1\n", "expected six TREC run columns"),
        ("31 Q1 doc 1 1 run\n", "column 2 must be Q0"),
        ("31 Q0 doc rank 1 run\n", "rank must be an integer"),
        ("31 Q0 doc 0 1 run\n", "rank must be positive"),
        ("31 Q0 doc 1 score run\n", "score must be a number"),
        ("31 Q0 doc 1 nan run\n", "score must be finite"),
        ("31 Q0 doc 1 inf run\n", "score must be finite"),
    ],
)
def test_parse_trec_retrieval_run_rejects_malformed_rows(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    run_path = _write(tmp_path / "run.trec", body)

    with pytest.raises(ValueError, match=message):
        parse_trec_retrieval_run(run_path, expected_topic_ids=("31",))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "31 Q0 doc-a 1 2 run\n31 Q0 doc-b 1 1 run\n",
            "duplicate rank 1 for topic 31",
        ),
        (
            "31 Q0 doc-a 1 2 run\n31 Q0 doc-a 2 1 run\n",
            "duplicate document ID doc-a for topic 31",
        ),
        (
            "31 Q0 doc-a 1 2 run-a\n31 Q0 doc-b 2 1 run-b\n",
            "conflicting run tags",
        ),
    ],
)
def test_parse_trec_retrieval_run_rejects_inconsistent_rows(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    run_path = _write(tmp_path / "run.trec", body)

    with pytest.raises(ValueError, match=message):
        parse_trec_retrieval_run(run_path, expected_topic_ids=("31",))


@pytest.mark.parametrize(
    ("expected_topic_ids", "message"),
    [
        (("31", "37"), "missing expected topics: 37"),
        (("31",), "topics outside the expected population: 37"),
    ],
)
def test_parse_trec_retrieval_run_requires_the_exact_topic_population(
    tmp_path: Path,
    expected_topic_ids: tuple[str, ...],
    message: str,
) -> None:
    body = "31 Q0 doc-a 1 2 run\n"
    if expected_topic_ids == ("31",):
        body += "37 Q0 doc-b 1 1 run\n"
    run_path = _write(tmp_path / "run.trec", body)

    with pytest.raises(ValueError, match=message):
        parse_trec_retrieval_run(run_path, expected_topic_ids=expected_topic_ids)


@pytest.mark.parametrize("expected_topic_ids", [(), ("",), ("31", "31")])
def test_parse_trec_retrieval_run_rejects_invalid_expected_topic_scope(
    tmp_path: Path,
    expected_topic_ids: tuple[str, ...],
) -> None:
    run_path = _write(tmp_path / "run.trec", "31 Q0 doc-a 1 2 run\n")

    with pytest.raises(ValueError, match="expected topic"):
        parse_trec_retrieval_run(run_path, expected_topic_ids=expected_topic_ids)


def test_evaluation_binds_scope_metrics_and_projected_qrels_provenance(tmp_path: Path) -> None:
    run_path = _write(
        tmp_path / "run.trec",
        "31 Q0 relevant-high 1 3 run\n"
        "31 Q0 unjudged 2 2 run\n"
        "31 Q0 relevant-medium 3 1 run\n",
    )
    qrels_path = _write(
        tmp_path / "projected.qrels",
        "31 0 relevant-high 3\n"
        "31 0 relevant-medium 2\n"
        "31 0 judged-low 1\n"
        "31 0 judged-zero 0\n",
    )

    result = evaluate_competition_retrieval_run(
        run_path,
        qrels_path,
        topic_ids=("31",),
        metric_names=(
            "ndcg@10",
            "judged_rate@10",
            "precision@10",
            "recall@100",
            "ideal_dcg_coverage@50",
        ),
        relevance_threshold=2,
        expected_qrels_sha256=sha256(qrels_path.read_bytes()).hexdigest(),
        expected_qrels_topic_ids=("31",),
        assessor_variant="tiny-projected-qrels-test-fixture",
    )

    assert result["schema_version"] == 1
    assert result["projected_development_qrels"] is True
    assert result["topic_ids"] == ["31"]
    assert result["run"] == {
        "path": str(run_path),
        "sha256": sha256(run_path.read_bytes()).hexdigest(),
    }
    assert result["qrels"] == {
        "path": str(qrels_path),
        "sha256": sha256(qrels_path.read_bytes()).hexdigest(),
    }
    assert result["assessor"] == {
        "variant": "tiny-projected-qrels-test-fixture",
        "path": str(qrels_path),
        "sha256": sha256(qrels_path.read_bytes()).hexdigest(),
    }
    assert result["relevance_threshold"] == 2
    assert result["metric_names"] == [
        "ndcg@10",
        "judged_rate@10",
        "precision@10",
        "recall@100",
        "ideal_dcg_coverage@50",
    ]
    assert result["metrics"] == pytest.approx(
        {
            "ndcg@10": 0.9049495058460971,
            "judged_rate@10": 0.2,
            "precision@10": 0.2,
            "recall@100": 1.0,
            "ideal_dcg_coverage@50": 0.9467667785528817,
        }
    )
    assert result["per_topic"]["31"] == pytest.approx(result["metrics"])


def test_evaluation_rejects_qrels_that_do_not_match_expected_digest(tmp_path: Path) -> None:
    run_path = _write(tmp_path / "run.trec", "31 Q0 relevant 1 1 run\n")
    qrels_path = _write(tmp_path / "projected.qrels", "31 0 relevant 3\n")

    with pytest.raises(ValueError, match="qrels SHA-256 does not match the pinned assessor"):
        evaluate_competition_retrieval_run(
            run_path,
            qrels_path,
            topic_ids=("31",),
            metric_names=("ndcg@10",),
            expected_qrels_sha256="0" * 64,
            expected_qrels_topic_ids=("31",),
        )


@pytest.mark.parametrize(
    ("qrels_body", "expected_topics", "message"),
    [
        ("31 Q0 relevant 3\n", ("31",), "qrels column 2 must be 0"),
        ("31 0 relevant 5\n", ("31",), "qrels grade must be in the range 0..4"),
        (
            "31 0 relevant 3\n31 0 relevant 2\n",
            ("31",),
            "duplicate qrels topic/document pair",
        ),
        ("31 0 relevant 3\n", ("31", "37"), "qrels missing expected topics: 37"),
        (
            "31 0 relevant 3\n37 0 other 2\n",
            ("31",),
            "qrels topics outside the expected population: 37",
        ),
    ],
)
def test_evaluation_strictly_validates_projected_qrels(
    tmp_path: Path,
    qrels_body: str,
    expected_topics: tuple[str, ...],
    message: str,
) -> None:
    run_path = _write(tmp_path / "run.trec", "31 Q0 relevant 1 1 run\n")
    qrels_path = _write(tmp_path / "projected.qrels", qrels_body)

    with pytest.raises(ValueError, match=message):
        evaluate_competition_retrieval_run(
            run_path,
            qrels_path,
            topic_ids=("31",),
            metric_names=("ndcg@10",),
            expected_qrels_sha256=sha256(qrels_path.read_bytes()).hexdigest(),
            expected_qrels_topic_ids=expected_topics,
        )


def test_cli_writes_canonical_json_with_default_metrics(tmp_path: Path) -> None:
    run_path = _write(tmp_path / "run.trec", "31 Q0 relevant 1 1 fixed-run\n")
    qrels_path = Path(
        "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
        "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
    )
    output_path = tmp_path / "evaluation.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.competition_retrieval_evaluation",
            "--run",
            str(run_path),
            "--qrels",
            str(qrels_path),
            "--topic",
            "31",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = output_path.read_bytes()
    result = json.loads(payload)
    assert payload == (
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert result["metric_names"] == [
        "ndcg@10",
        "judged_count@10",
        "judged_rate@10",
        "precision@10",
        "recall@10",
        "hit_rate@10",
        "relevant_count@10",
        "graded_recall@10",
        "judged_count@50",
        "judged_rate@50",
        "precision@50",
        "recall@50",
        "hit_rate@50",
        "relevant_count@50",
        "graded_recall@50",
        "ideal_dcg_coverage@50",
        "judged_count@100",
        "judged_rate@100",
        "recall@100",
    ]
    assert result["projected_development_qrels"] is True
    assert result["assessor"] == {
        "variant": "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1",
        "path": str(qrels_path),
        "sha256": "42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37",
    }

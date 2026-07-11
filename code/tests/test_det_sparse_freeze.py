import json

import pytest

from trec_rag.det_sparse_config import (
    ARM_NAMES,
    EVALUATION_METRICS,
    PILOT_TOPIC_IDS,
)
from trec_rag.det_sparse_freeze import (
    FrozenEvaluationResult,
    assess_pilot,
    create_freeze_manifest,
    evaluate_frozen_arms,
    rankings_sha256,
    verify_freeze_manifest,
)
from trec_rag.pipeline_models import RankedCandidate


def test_freeze_is_create_only_and_detects_tampering_without_reading_qrels(tmp_path):
    artifact = tmp_path / "plans.jsonl"
    artifact.write_text('{"plan":"frozen"}\n', encoding="utf-8")
    qrels = tmp_path / "qrels.txt"
    manifest_path = tmp_path / "freeze.json"

    manifest = create_freeze_manifest(
        manifest_path,
        artifact_root=tmp_path,
        artifacts=[artifact],
        qrels_path=qrels,
        metadata={"model_calls": 0, "reranker_calls": 0},
    )

    assert manifest["qrels_opened_before_freeze"] is False
    assert manifest["artifacts"][0]["path"] == "plans.jsonl"
    assert not qrels.exists()
    verify_freeze_manifest(
        manifest_path,
        artifact_root=tmp_path,
        expected_qrels_path=qrels,
    )
    with pytest.raises(FileExistsError):
        create_freeze_manifest(
            manifest_path,
            artifact_root=tmp_path,
            artifacts=[artifact],
            qrels_path=qrels,
            metadata={},
        )

    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_freeze_manifest(
            manifest_path,
            artifact_root=tmp_path,
            expected_qrels_path=qrels,
        )


def test_evaluation_does_not_read_qrels_until_freeze_verifies(tmp_path, monkeypatch):
    artifact = tmp_path / "rankings.jsonl"
    artifact.write_text("frozen\n", encoding="utf-8")
    qrels = tmp_path / "qrels.txt"
    qrels.write_text("200 0 doc-a 4\n", encoding="utf-8")
    manifest_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        manifest_path,
        artifact_root=tmp_path,
        artifacts=[artifact],
        qrels_path=qrels,
        metadata={},
    )
    artifact.write_text("changed\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze.parse_qrels_bytes",
        lambda content: calls.append(content) or {},
    )

    with pytest.raises(ValueError, match="mismatch"):
        evaluate_frozen_arms(
            manifest_path=manifest_path,
            artifact_root=tmp_path,
            qrels_path=qrels,
            rankings={arm: [] for arm in ARM_NAMES},
        )

    assert calls == []


def test_final_ranking_hash_requires_exact_depth_100():
    rankings = {
        arm: [
            RankedCandidate(topic_id, "doc-a", 1, 1.0, "text", [])
            for topic_id in PILOT_TOPIC_IDS
        ]
        for arm in ARM_NAMES
    }
    with pytest.raises(ValueError, match="exactly 100"):
        rankings_sha256(rankings)


def _metrics(recall_values, ndcg_values):
    per_topic = {
        topic_id: {
            metric: (
                recall
                if metric == "recall@100"
                else ndcg if metric == "ndcg@10" else 0.0
            )
            for metric in EVALUATION_METRICS
        }
        for topic_id, recall, ndcg in zip(PILOT_TOPIC_IDS, recall_values, ndcg_values)
    }
    return {
        "metrics": {
            metric: sum(row[metric] for row in per_topic.values()) / len(per_topic)
            for metric in EVALUATION_METRICS
        },
        "per_topic": per_topic,
    }


def test_preregistered_gate_prefers_combined_only_when_both_incremental_guards_pass(
    tmp_path,
    monkeypatch,
):
    metrics = {
        "O": _metrics([0.10] * 4, [0.50] * 4),
        "F": _metrics([0.112] * 4, [0.49] * 4),
        "E": _metrics([0.106] * 4, [0.495] * 4),
        "FE": _metrics([0.116] * 4, [0.49] * 4),
    }

    evidence = {"qrels_sha256": "a" * 64}
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze.evaluate_frozen_arms",
        lambda **_kwargs: FrozenEvaluationResult(metrics, evidence),
    )
    decision = assess_pilot(
        manifest_path=tmp_path / "freeze.json",
        artifact_root=tmp_path,
        qrels_path=tmp_path / "qrels.txt",
        rankings={},
    )

    assert decision.facets.passed
    assert decision.expansion.passed
    assert decision.combined.passed
    assert decision.preferred_arm == "FE"

    too_close = {
        **metrics,
        "FE": _metrics([0.114] * 4, [0.49] * 4),
    }
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze.evaluate_frozen_arms",
        lambda **_kwargs: FrozenEvaluationResult(too_close, evidence),
    )
    decision = assess_pilot(
        manifest_path=tmp_path / "freeze.json",
        artifact_root=tmp_path,
        qrels_path=tmp_path / "qrels.txt",
        rankings={},
    )
    assert not decision.combined.passed
    assert decision.preferred_arm == "F"
    assert decision.evaluation_evidence == evidence


def test_mechanical_failure_cannot_be_overridden_by_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze.evaluate_frozen_arms",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("freeze mismatch")),
    )

    with pytest.raises(ValueError, match="mismatch"):
        assess_pilot(
            manifest_path=tmp_path / "freeze.json",
            artifact_root=tmp_path,
            qrels_path=tmp_path / "qrels.txt",
            rankings={},
        )


def test_gate_rejects_nonfinite_or_incomplete_topic_metrics(tmp_path, monkeypatch):
    metrics = {arm: _metrics([0.1] * 4, [0.5] * 4) for arm in ARM_NAMES}
    metrics["FE"]["per_topic"]["200"]["recall@100"] = float("nan")
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze.evaluate_frozen_arms",
        lambda **_kwargs: FrozenEvaluationResult(metrics, {}),
    )
    with pytest.raises(ValueError, match="metric value"):
        assess_pilot(
            manifest_path=tmp_path / "freeze.json",
            artifact_root=tmp_path,
            qrels_path=tmp_path / "qrels.txt",
            rankings={},
        )

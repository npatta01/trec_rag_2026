from __future__ import annotations

from pathlib import Path

import pytest

from trec_rag.deep_facet_candidate_cascade import (
    build_cascade,
    decide_cascade,
    evaluate_cascade,
    evaluate_cascade_payload,
    freeze_cascade,
    verify_cascade_freeze,
    verify_cascade_semantics,
)
from trec_rag.deep_facet_candidate_manifest import TOPIC_IDS
from trec_rag.deep_facet_candidate_rank import ARM_NAMES, create_seal


def _permutations(size: int = 140) -> tuple[list[str], list[str], list[str]]:
    ids = [f"d{index:03d}" for index in range(size)]
    return ids, ids[5:] + ids[:5], list(reversed(ids))


def test_cascade_preserves_rrf_head_then_allocates_global_then_dual() -> None:
    rrf, global_arm, dual = _permutations()

    result = build_cascade(rrf, global_arm, dual)

    assert result[:10] == rrf[:10]
    expected_global = [docid for docid in global_arm if docid not in set(rrf[:10])][:90]
    assert result[10:100] == expected_global
    selected = set(result[:100])
    assert result[100:] == [docid for docid in dual if docid not in selected]
    assert len(result) == len(rrf)
    assert set(result) == set(rrf)
    assert len(result) == len(set(result))


def test_cascade_is_deterministic_and_rejects_mismatched_populations() -> None:
    rrf, global_arm, dual = _permutations()
    assert build_cascade(rrf, global_arm, dual) == build_cascade(rrf, global_arm, dual)

    with pytest.raises(ValueError, match="same complete population"):
        build_cascade(rrf, global_arm[:-1], dual)
    with pytest.raises(ValueError, match="duplicate"):
        build_cascade(rrf, global_arm, dual[:-1] + [dual[-2]])


def test_cascade_decision_applies_every_fixed_stop_guard() -> None:
    baseline = {
        "RRF": {"ndcg@10": 0.50, "graded_recall@500": 0.40, "graded_recall@1000": 0.60},
        "GLOBAL": {"graded_recall@500": 0.42},
    }
    per_topic = {
        topic: {"graded_recall@500": 0.45, "delta_vs_RRF": -0.01}
        for topic in ("219", "72", "300", "84")
    }
    cascade = {
        "ndcg@10": 0.50,
        "graded_recall@500": 0.45,
        "graded_recall@1000": 0.61,
        "novel_retained@1000": 142,
        "novel_retention@1000": 142 / 177,
    }

    pending = decide_cascade(cascade, baseline, per_topic, judged_coverage_defensible=None)
    assert pending["mechanical_guards_pass"] is True
    assert pending["advance_to_fresh_validation"] is False
    assert pending["coverage_review_status"] == "required"

    passed = decide_cascade(cascade, baseline, per_topic, judged_coverage_defensible=True)
    assert passed["advance_to_fresh_validation"] is True

    failed = decide_cascade(
        {**cascade, "graded_recall@500": 0.39, "novel_retained@1000": 141},
        baseline,
        {**per_topic, "84": {"graded_recall@500": 0.10, "delta_vs_RRF": -0.03}},
        judged_coverage_defensible=True,
    )
    assert failed["advance_to_fresh_validation"] is False
    assert set(failed["failed_guards"]) >= {
        "graded_recall_500_beats_rrf",
        "graded_recall_500_beats_global",
        "novel_retention_1000_at_least_142_of_177",
        "per_topic_graded_recall_500_floor",
    }


def test_builder_api_has_no_qrels_argument() -> None:
    import inspect

    parameters = inspect.signature(build_cascade).parameters
    assert "qrels" not in parameters
    assert "qrels_path" not in parameters


def _sealed_source(tmp_path: Path) -> Path:
    import json

    source = tmp_path / "source_freeze"
    source.mkdir()
    rows = []
    for topic in TOPIC_IDS:
        ids = [f"{topic}-d{index:03d}" for index in range(120)]
        orders = {
            "RRF": ids,
            "GLOBAL": ids[5:] + ids[:5],
            "FACET": ids[10:] + ids[:10],
            "DUAL": list(reversed(ids)),
            "DUAL-NR": ids[20:] + ids[:20],
        }
        for arm in ARM_NAMES:
            rows.extend(
                {
                    "topic_id": topic,
                    "arm": arm,
                    "rank": rank,
                    "document_id": document_id,
                }
                for rank, document_id in enumerate(orders[arm], start=1)
            )
    (source / "rankings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    create_seal(
        manifest_path=manifest,
        source_dirs=[],
        freeze_dir=source,
        topic_ids=TOPIC_IDS,
    )
    return source


def test_freeze_is_create_only_verifiable_and_rejects_mutation(tmp_path: Path) -> None:
    source = _sealed_source(tmp_path)
    output = tmp_path / "cascade"

    summary = freeze_cascade(source, output)

    assert summary["post_qrels_diagnostic"] is True
    assert summary["qrels_read"] is False
    assert summary["topic_ids"] == list(TOPIC_IDS)
    assert verify_cascade_freeze(output)["status"] == "cascade_frozen_before_diagnostic_metrics"
    with pytest.raises(FileExistsError, match="create-only"):
        freeze_cascade(source, output)

    with (output / "rankings.jsonl").open("a", encoding="utf-8") as sink:
        sink.write("{}\n")
    with pytest.raises(ValueError, match="mutated"):
        verify_cascade_freeze(output)


def test_semantic_verifier_rejects_wrong_but_self_consistently_resealed_splice(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    source = _sealed_source(tmp_path)
    output = tmp_path / "cascade"
    freeze_cascade(source, output)
    rows = [json.loads(line) for line in (output / "rankings.jsonl").read_text().splitlines()]
    first, second = rows[0], rows[1]
    first["document_id"], second["document_id"] = second["document_id"], first["document_id"]
    ranking_bytes = b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in rows
    )
    (output / "rankings.jsonl").write_bytes(ranking_bytes)
    seal_path = output / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    seal["artifacts"]["rankings.jsonl"] = {
        "bytes": len(ranking_bytes),
        "sha256": hashlib.sha256(ranking_bytes).hexdigest(),
    }
    material = {
        "topic_ids": seal["topic_ids"],
        "source_seal_root_sha256": seal["source_seal_root_sha256"],
        "artifacts": seal["artifacts"],
    }
    canonical = json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    seal["root_sha256"] = hashlib.sha256(canonical).hexdigest()
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")

    verify_cascade_freeze(output)
    with pytest.raises(ValueError, match="semantic splice"):
        verify_cascade_semantics(output)


def test_payload_evaluation_reports_metrics_novel_retention_and_pending_review() -> None:
    from trec_rag.deep_facet_candidate_evaluate import evaluate_ranking

    rankings: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, int]] = {}
    discovery: dict[str, object] = {}
    novel_counts = dict(zip(TOPIC_IDS, (45, 44, 44, 44), strict=True))
    for topic in TOPIC_IDS:
        ids = [f"{topic}-d{index:03d}" for index in range(120)]
        novel_start = len(ids) - novel_counts[topic]
        rankings[topic] = ids
        qrels[topic] = {
            document_id: (3 if index < 10 else 2)
            for index, document_id in enumerate(ids)
            if index < 10 or index >= novel_start
        }
        discovery[topic] = {"novel_relevant_ids": ids[novel_start:]}
    ndcg = sum(
        float(evaluate_ranking(rankings[topic], qrels[topic])["ndcg@10"])
        for topic in TOPIC_IDS
    ) / len(TOPIC_IDS)
    source_metrics = {
        "novel_relevant_count": 177,
        "aggregate": {
            "RRF": {
                "ndcg@10": ndcg,
                "graded_recall@500": 0.90,
                "graded_recall@1000": 0.95,
            },
            "GLOBAL": {"graded_recall@500": 0.89},
        },
        "per_topic": {
            topic: {"RRF": {"graded_recall@500": 0.90}}
            for topic in TOPIC_IDS
        },
        "discovery": discovery,
    }

    metrics, decision = evaluate_cascade_payload(
        rankings,
        qrels,
        source_metrics,
        judged_coverage_defensible=None,
    )

    cascade = metrics["aggregate"]["RRF-GLOBAL-DUAL"]
    assert cascade["ndcg@10"] == ndcg
    assert cascade["graded_recall@500"] == 1.0
    assert cascade["novel_retained@1000"] == 177
    assert cascade["novel_retention@1000"] == 1.0
    assert decision["mechanical_guards_pass"] is True
    assert decision["advance_to_fresh_validation"] is False
    assert decision["coverage_review_status"] == "required"


def _evaluation_source(tmp_path: Path) -> Path:
    import hashlib
    import json

    source = tmp_path / "source_evaluation"
    source.mkdir()
    projection_rows = []
    discovery: dict[str, object] = {}
    per_topic: dict[str, object] = {}
    novel_counts = dict(zip(TOPIC_IDS, (45, 44, 44, 44), strict=True))
    for topic in TOPIC_IDS:
        ids = [f"{topic}-d{index:03d}" for index in range(120)]
        novel_start = len(ids) - novel_counts[topic]
        qrels = {
            document_id: (3 if index < 10 else 2)
            for index, document_id in enumerate(ids)
            if index < 10 or index >= novel_start
        }
        projection_rows.extend(
            {"topic_id": topic, "document_id": document_id, "grade": grade}
            for document_id, grade in sorted(qrels.items())
        )
        discovery[topic] = {"novel_relevant_ids": ids[novel_start:]}
        per_topic[topic] = {"RRF": {"graded_recall@500": 0.90}}
    metrics = {
        "novel_relevant_count": 177,
        "aggregate": {
            "RRF": {"ndcg@10": 1.0, "graded_recall@500": 0.90, "graded_recall@1000": 0.95},
            "GLOBAL": {"graded_recall@500": 0.89},
        },
        "per_topic": per_topic,
        "discovery": discovery,
    }
    metrics_bytes = (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode()
    projection_bytes = b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in projection_rows
    )
    (source / "metrics.json").write_bytes(metrics_bytes)
    (source / "qrels_projection.jsonl").write_bytes(projection_bytes)
    (source / "qrels_access_receipt.json").write_text(
        json.dumps(
            {"qrels_projection_sha256": hashlib.sha256(projection_bytes).hexdigest()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "qrels_opened": True,
                "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_evaluator_requires_seal_then_writes_create_only_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seal"):
        evaluate_cascade(tmp_path / "not-sealed", tmp_path / "missing-evaluation", tmp_path / "out")
    assert not (tmp_path / "out").exists()

    source = _sealed_source(tmp_path)
    cascade = tmp_path / "cascade"
    freeze_cascade(source, cascade)
    evaluation = _evaluation_source(tmp_path)
    output = tmp_path / "diagnostic"

    summary = evaluate_cascade(cascade, evaluation, output)

    assert summary["status"] == "complete"
    assert summary["post_qrels_diagnostic"] is True
    assert summary["new_retrieval_calls"] == 0
    assert summary["new_model_inference_calls"] == 0
    assert (output / "metrics.json").exists()
    assert (output / "decision.json").exists()
    with pytest.raises(FileExistsError, match="create-only"):
        evaluate_cascade(cascade, evaluation, output)

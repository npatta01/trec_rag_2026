from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import trec_rag.tethered_facet_evaluate as evaluate_module
import trec_rag.tethered_facet_two_basket as rank_module
from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
from trec_rag.build_tethered_facet_report import ReportSources, build_artifact, build_report
from trec_rag.tethered_facet_evaluate import TOPIC_IDS, evaluate, evaluate_arm


def _pretty(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: object) -> bytes:
    content = _pretty(value)
    path.write_bytes(content)
    return content


def _prior_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "prior"
    freeze = root / "freeze_v1"
    evaluation = root / "evaluation_v1"
    freeze.mkdir(parents=True)
    evaluation.mkdir()
    seal_bytes = _pretty({"root_sha256": "b" * 64})
    (freeze / "SEALED.json").write_bytes(seal_bytes)
    qrels = {
        topic: {
            f"{topic}-d300": 2,
            f"{topic}-d349": 2,
            f"{topic}-d500": 2,
            f"{topic}-d549": 2,
        }
        for topic in TOPIC_IDS
    }
    projection_bytes = b"".join(
        json.dumps(
            {"topic_id": topic, "document_id": document, "grade": grade},
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        for topic in TOPIC_IDS
        for document, grade in qrels[topic].items()
    )
    (evaluation / "qrels_projection.jsonl").write_bytes(projection_bytes)
    prior_rows = []
    rankings = {}
    novel = {}
    for topic in TOPIC_IDS:
        documents = [f"{topic}-d{index:03d}" for index in range(650)]
        rankings[topic] = documents
        novel[topic] = set(qrels[topic])
        for rank, document in enumerate(documents, 1):
            index = rank - 1
            is_facet = 300 <= index < 550
            prior_rows.append({
                "topic_id": topic,
                "arm": "RRF",
                "rank": rank,
                "document_id": document,
                "original_rank": rank if index < 300 else None,
                "best_facet_rank": index - 299 if is_facet else 10**9,
                "facet_percentiles": {f"{topic}-facet": 1.0} if is_facet else {},
            })
    (freeze / "rankings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prior_rows),
        encoding="utf-8",
    )
    receipt_bytes = _write(evaluation / "qrels_access_receipt.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "status": "qrels_access_boundary_crossed",
        "qrels_opened": True,
        "upstream_mutation_forbidden": True,
        "topic_ids": list(TOPIC_IDS),
        "qrels_projection_rows": len(projection_bytes.splitlines()),
        "qrels_projection_sha256": _sha(projection_bytes),
        "seal_sha256": _sha(seal_bytes),
        "seal_root_sha256": "b" * 64,
        "qrels_source_name": "synthetic sealed projection",
        "evaluator_code_sha256": "c" * 64,
    })
    recomputed = evaluate_arm(rankings, qrels, novel, depths=(100, 500, 1000))
    metrics_bytes = _write(evaluation / "metrics.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": 16,
        "aggregate": {"RRF": recomputed["aggregate"]},
        "per_topic": {topic: {"RRF": recomputed["per_topic"][topic]} for topic in TOPIC_IDS},
        "discovery": {topic: {"novel_relevant_ids": sorted(novel[topic])} for topic in TOPIC_IDS},
    })
    decision_bytes = b"{}\n"
    (evaluation / "decision.json").write_bytes(decision_bytes)
    summary_bytes = _write(evaluation / "summary.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "status": "complete",
        "qrels_opened": True,
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": 16,
        "metrics_sha256": _sha(metrics_bytes),
        "decision_sha256": _sha(decision_bytes),
    })
    monkeypatch.setattr(evaluate_module, "NOVEL_RELEVANT_TOTAL", 16)
    monkeypatch.setattr(evaluate_module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64})
    monkeypatch.setattr(evaluate_module, "HISTORICAL_PRIOR_EVALUATION_IDENTITY", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "topic_ids": list(TOPIC_IDS),
        "qrels_projection_rows": len(projection_bytes.splitlines()),
        "files": {
            "qrels_access_receipt.json": _sha(receipt_bytes),
            "qrels_projection.jsonl": _sha(projection_bytes),
            "metrics.json": _sha(metrics_bytes),
            "decision.json": _sha(decision_bytes),
            "summary.json": _sha(summary_bytes),
        },
    })
    return evaluation


def test_task2_compatible_fixture_chain_without_inference_and_provenance_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep, tethered = _loader_sources(tmp_path)
    preflight = {
        "schema_version": "tethered-facet-minilm-preflight-v1",
        "status": "tokenizer_only_preflight_complete",
        "topic_ids": list(TOPIC_IDS),
        "qrels_read": False,
        "retrieval_performed": False,
        "model": "synthetic/minilm",
        "model_revision": "fixture-revision",
        "summary": {
            "query_document_pair_count": 1000,
            "window_count": 1000,
            "unique_pair_count": 1000,
            "cache_hit_window_count": 400,
            "cache_miss_window_count": 600,
        },
        "runtime_evidence": {"projected_inference_seconds": 12.5},
    }
    preflight_bytes = _write(tethered / "preflight.json", preflight)
    scoring = {
        "schema_version": "tethered-facet-minilm-scoring-v1",
        "status": "complete",
        "topic_ids": list(TOPIC_IDS),
        "preflight_sha256": _sha(preflight_bytes),
        "qrels_read": False,
        "network_accessed": False,
        "model": "synthetic/minilm",
        "model_revision": "fixture-revision",
        "planned_window_count": 1000,
        "completed_window_count": 1200,
        "document_score_count": 1000,
        "cache_hit_count": 400,
        "unique_forward_pair_count": 600,
        "elapsed_seconds": 9.5,
        "peak_device_memory_bytes": 1234,
        "peak_host_memory_bytes": 5678,
    }
    _write(tethered / "scoring_receipt.json", scoring)
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    task3 = tmp_path / "task3"
    assert rank_module.main([
        "freeze", "--deep-root", str(deep), "--tethered", str(tethered), "--output", str(task3)
    ]) == 0
    assert rank_module.main(["verify", "--freeze", str(task3)]) == 0

    prior = _prior_evaluation(tmp_path, monkeypatch)
    task4 = tmp_path / "task4"
    assert evaluate(task3, prior, task4)["status"] == "complete"
    diagnostics = json.loads((task4 / "diagnostics.json").read_text())
    assert set(diagnostics) >= {
        "noise_pattern_counts", "facet_yield_changes", "relevant_below_500",
        "duplicate_and_quota_pressure", "scoring_telemetry", "representatives",
    }
    assert {row["arm"] for row in diagnostics["noise_pattern_counts"]} == {"FACET-2B", "TETHERED-2B"}
    assert {
        arm: sum(row["selected_count"] for row in diagnostics["noise_pattern_counts"] if row["arm"] == arm)
        for arm in ("FACET-2B", "TETHERED-2B")
    } == {"FACET-2B": 800, "TETHERED-2B": 800}
    assert {row["classification"] for row in diagnostics["facet_yield_changes"]} <= {"rose", "fell", "zero", "unchanged"}
    assert all(row["reason"] in {"not_in_facet_candidate_pool", "facet_quota_exhausted", "facet_basket_capacity_exhausted"} for row in diagnostics["relevant_below_500"])
    assert all(row["reason"] != "duplicate_displaced" for row in diagnostics["relevant_below_500"])
    assert diagnostics["scoring_telemetry"]["cache_hit_count"] == 400
    assert diagnostics["scoring_telemetry"]["cache_miss_count"] == 600
    assert diagnostics["scoring_telemetry"]["unique_scoring_pair_count"] == 1000
    assert diagnostics["scoring_telemetry"]["model"] == "synthetic/minilm"
    assert len(diagnostics["duplicate_and_quota_pressure"]) == 8
    assert all(row["duplicate_skip_total"] == sum(row["duplicate_skip_totals"].values()) for row in diagnostics["duplicate_and_quota_pressure"])
    assert all(
        set(row["shortage_counts"]) == {f"{row['topic_id']}-facet"}
        for row in diagnostics["duplicate_and_quota_pressure"]
    )
    task5 = tmp_path / "task5"
    built = build_report(ReportSources(
        topic_ids=list(TOPIC_IDS),
        task1_receipt=tethered / "preflight.json",
        task2_receipt=tethered / "scoring_receipt.json",
        task3_freeze=task3,
        task4_evaluation=task4,
    ), task5)
    assert {row["movement"] for row in built.artifact["representatives"]} == {"promoted", "demoted"}
    assert (task5 / "report.html").exists()

    diagnostics_path = task4 / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["representatives"][0]["passage_provenance"]["window_sha256"] = "tampered"
    diagnostics_bytes = _pretty(diagnostics)
    diagnostics_path.write_bytes(diagnostics_bytes)
    summary_path = task4 / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["artifacts"]["diagnostics.json"] = {"bytes": len(diagnostics_bytes), "sha256": _sha(diagnostics_bytes)}
    summary_path.write_bytes(_pretty(summary))
    with pytest.raises(ValueError, match="passage provenance"):
        build_artifact(ReportSources(
            topic_ids=list(TOPIC_IDS), task1_receipt=tethered / "preflight.json",
            task2_receipt=tethered / "scoring_receipt.json", task3_freeze=task3,
            task4_evaluation=task4,
        ))

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

import trec_rag.tethered_facet_evaluate as evaluate_module
from trec_rag.build_tethered_facet_report import (
    ReportSources,
    build_artifact,
    build_parser,
    build_report,
    main,
)


TOPICS = ["219", "72", "300", "84"]


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: object) -> bytes:
    content = _bytes(value)
    path.write_bytes(content)
    return content


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(content), "sha256": _sha(content)}


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportSources:
    source = tmp_path / "sources"
    task3 = source / "task3"
    task4 = source / "task4"
    task3.mkdir(parents=True)
    task4.mkdir()

    preflight = {
        "schema_version": "tethered-facet-minilm-preflight-v1",
        "status": "tokenizer_only_preflight_complete",
        "topic_ids": TOPICS,
        "qrels_read": False,
        "retrieval_performed": False,
        "model": "synthetic/minilm",
        "model_revision": "fixture-revision",
        "summary": {
            "query_document_pair_count": 4800,
            "window_count": 5200,
            "unique_pair_count": 5000,
        },
        "runtime_evidence": {"projected_inference_seconds": 12.5},
    }
    preflight_bytes = _write(source / "preflight.json", preflight)
    scoring = {
        "schema_version": "tethered-facet-minilm-scoring-v1",
        "status": "complete",
        "topic_ids": TOPICS,
        "preflight_sha256": _sha(preflight_bytes),
        "qrels_read": False,
        "network_accessed": False,
        "model": "synthetic/minilm",
        "model_revision": "fixture-revision",
        "planned_window_count": 5000,
        "completed_window_count": 5200,
        "document_score_count": 4800,
        "cache_hit_count": 1000,
        "unique_forward_pair_count": 4000,
        "elapsed_seconds": 9.5,
        "peak_device_memory_bytes": 1234,
        "peak_host_memory_bytes": 5678,
    }
    scoring_bytes = _write(source / "scoring_receipt.json", scoring)
    raw = source / "raw"
    raw.mkdir()
    promoted_narrative = "Explain the full policy narrative and its trade-offs."
    promoted_facet = "positive effects on rural communities"
    promoted_text = "The program increased access while preserving local services."
    demoted_narrative = "Assess benefits, safety risks, and regulatory responses."
    demoted_facet = "reported safety incidents"
    demoted_text = "A product name matched the facet but the passage concerned another domain."
    facet_candidates = [
        {"topic_id": "219", "facet_id": "219-positive", "document_id": "doc-promoted", "query": promoted_facet, "query_sha256": _sha(promoted_facet.encode()), "text": promoted_text, "text_sha256": _sha(promoted_text.encode()), "rank": 7},
        {"topic_id": "84", "facet_id": "84-safety", "document_id": "doc-demoted", "query": demoted_facet, "query_sha256": _sha(demoted_facet.encode()), "text": demoted_text, "text_sha256": _sha(demoted_text.encode()), "rank": 11},
    ]
    tethered_candidates = [
        {**row, "schema_version": "tethered-facet-minilm-candidate-v1", "query": narrative + "\n\nFocus: " + row["query"], "query_sha256": _sha((narrative + "\n\nFocus: " + row["query"]).encode()), "facet_query": row["query"], "facet_query_sha256": row["query_sha256"], "prior_bm25_rank": row["rank"]}
        for row, narrative in zip(facet_candidates, (promoted_narrative, demoted_narrative), strict=True)
    ]
    facet_windows = [
        {"schema_version": "deep-facet-candidate-minilm-score-v1", "topic_id": "84", "variant": "84-safety", "document_id": "doc-demoted", "window_id": "w-demoted", "window_text": demoted_text, "window_sha256": _sha(demoted_text.encode()), "query_sha256": _sha(demoted_facet.encode()), "document_sha256": _sha(demoted_text.encode()), "model": "synthetic/minilm", "model_revision": "fixture-revision", "document_start_token": 3, "document_end_token": 15, "score": 2.0},
    ]
    tethered_windows = [
        {"schema_version": "tethered-facet-minilm-score-v1", "topic_id": "219", "facet_id": "219-positive", "document_id": "doc-promoted", "window_id": "w-promoted", "window_text": promoted_text, "window_sha256": _sha(promoted_text.encode()), "query_sha256": tethered_candidates[0]["query_sha256"], "document_sha256": _sha(promoted_text.encode()), "model": "synthetic/minilm", "model_revision": "fixture-revision", "document_start_token": 0, "document_end_token": 12, "score": 2.0},
    ]
    document_scores = [
        {"schema_version": "tethered-facet-minilm-document-score-v1", "topic_id": "219", "facet_id": "219-positive", "document_id": "doc-promoted", "window_hashes": [_sha(promoted_text.encode())], "model": "synthetic/minilm", "model_revision": "fixture-revision"},
    ]
    raw_paths = {}
    for name, rows in (
        ("facet_candidates", facet_candidates), ("facet_window_scores", facet_windows),
        ("tethered_candidates", tethered_candidates), ("tethered_window_scores", tethered_windows),
        ("tethered_document_scores", document_scores),
    ):
        path = raw / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
        raw_paths[name] = path
    raw_paths["tethered_preflight"] = source / "preflight.json"
    raw_paths["tethered_scoring_receipt"] = source / "scoring_receipt.json"

    ranking_rows = [
        json.dumps(
            {
                "topic_id": topic,
                "arm": arm,
                "rank": 1,
                "document_id": f"{topic}-{arm}-d1",
                "source": "protected_head",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for topic in TOPICS
        for arm in ("FACET-2B", "TETHERED-2B")
    ]
    ranking_rows.extend([
        json.dumps({"topic_id": "219", "arm": "FACET-2B", "rank": 611, "document_id": "doc-promoted", "source": "dual_tail", "prior_bm25_rank": None}),
        json.dumps({"topic_id": "219", "arm": "TETHERED-2B", "rank": 202, "document_id": "doc-promoted", "source": "facet_basket", "prior_bm25_rank": 7}),
        json.dumps({"topic_id": "84", "arm": "FACET-2B", "rank": 204, "document_id": "doc-demoted", "source": "facet_basket", "prior_bm25_rank": 11}),
        json.dumps({"topic_id": "84", "arm": "TETHERED-2B", "rank": 612, "document_id": "doc-demoted", "source": "dual_tail", "prior_bm25_rank": None}),
    ])
    rankings = "".join(row + "\n" for row in ranking_rows).encode()
    (task3 / "rankings.jsonl").write_bytes(rankings)
    bindings_bytes = _write(
        task3 / "input_bindings.json",
        {
            "schema_version": "tethered-facet-two-basket-freeze-v1",
            "task1_preflight_sha256": _sha(preflight_bytes),
            "task2_scoring_receipt_sha256": _sha(scoring_bytes),
            "inputs": {name: _binding(path) for name, path in raw_paths.items()},
            "topic_inputs": {
                "FACET-2B": {
                    "219": {"facets": [{"facet_id": "219-positive", "scores": {"doc-promoted": 1.0, "filler": 2.0}, "model": "synthetic/minilm", "model_revision": "fixture-revision"}]},
                    "84": {"facets": [{"facet_id": "84-safety", "scores": {"doc-demoted": 2.0, "filler": 1.0}, "model": "synthetic/minilm", "model_revision": "fixture-revision"}]},
                    "72": {"facets": []}, "300": {"facets": []},
                },
                "TETHERED-2B": {
                    "219": {"facets": [{"facet_id": "219-positive", "scores": {"doc-promoted": 2.0, "filler": 1.0}, "model": "synthetic/minilm", "model_revision": "fixture-revision"}]},
                    "84": {"facets": [{"facet_id": "84-safety", "scores": {"doc-demoted": 1.0, "filler": 2.0}, "model": "synthetic/minilm", "model_revision": "fixture-revision"}]},
                    "72": {"facets": []}, "300": {"facets": []},
                },
            },
        },
    )
    task3_summary_bytes = _write(
        task3 / "summary.json",
        {
            "schema_version": "tethered-facet-two-basket-freeze-v1",
            "status": "complete",
            "topic_ids": TOPICS,
            "qrels_opened": False,
            "rankings_sha256": _sha(rankings),
            "input_bindings_sha256": _sha(bindings_bytes),
        },
    )
    seal_material = {
        "schema_version": "tethered-facet-two-basket-seal-v1",
        "status": "sealed_before_qrels",
        "qrels_opened": False,
        "files": {
            "input_bindings.json": {"bytes": len(bindings_bytes), "sha256": _sha(bindings_bytes)},
            "rankings.jsonl": {"bytes": len(rankings), "sha256": _sha(rankings)},
            "summary.json": {"bytes": len(task3_summary_bytes), "sha256": _sha(task3_summary_bytes)},
        },
    }
    seal_bytes = _write(task3 / "SEALED.json", {
        **seal_material,
        "root_sha256": _sha(json.dumps(seal_material, sort_keys=True, separators=(",", ":")).encode()),
    })

    prior_freeze = source / "prior_freeze"
    prior = source / "prior_evaluation"
    prior_freeze.mkdir()
    prior.mkdir()
    prior_seal_bytes = _write(prior_freeze / "SEALED.json", {"root_sha256": "b" * 64})
    (prior_freeze / "rankings.jsonl").write_text("{}\n")
    novel_ids = {
        topic: [f"{topic}-novel-{index}" for index in range(50 + (3 if topic == "84" else 0))]
        for topic in TOPICS
    }
    qrels = {
        **{(topic, document): 2 for topic, documents in novel_ids.items() for document in documents},
        ("219", "doc-promoted"): 2,
        ("84", "doc-demoted"): 0,
        ("219", "below-500"): 2,
    }
    projection_bytes = b"".join(
        json.dumps(
            {"topic_id": topic, "document_id": document, "grade": grade},
            sort_keys=True, separators=(",", ":"),
        ).encode() + b"\n"
        for (topic, document), grade in sorted(qrels.items(), key=lambda item: (TOPICS.index(item[0][0]), item[0][1]))
    )
    (prior / "qrels_projection.jsonl").write_bytes(projection_bytes)
    receipt_bytes = _write(prior / "qrels_access_receipt.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "status": "qrels_access_boundary_crossed",
        "qrels_opened": True,
        "upstream_mutation_forbidden": True,
        "topic_ids": TOPICS,
        "qrels_projection_rows": len(projection_bytes.splitlines()),
        "qrels_projection_sha256": _sha(projection_bytes),
        "seal_sha256": _sha(prior_seal_bytes),
        "seal_root_sha256": "b" * 64,
        "qrels_source_name": "synthetic historical projection",
        "evaluator_code_sha256": "c" * 64,
    })
    prior_metrics_bytes = _write(prior / "metrics.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "topic_ids": TOPICS,
        "novel_relevant_count": 203,
        "discovery": {
            topic: {"novel_relevant_ids": documents}
            for topic, documents in novel_ids.items()
        },
    })
    prior_decision_bytes = _write(prior / "decision.json", {})
    prior_summary_bytes = _write(prior / "summary.json", {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "status": "complete",
        "qrels_opened": True,
        "topic_ids": TOPICS,
        "novel_relevant_count": 203,
        "metrics_sha256": _sha(prior_metrics_bytes),
        "decision_sha256": _sha(prior_decision_bytes),
    })
    historical_anchor = {
        "schema_version": "deep-facet-candidate-evaluation-v1",
        "topic_ids": TOPICS,
        "qrels_projection_rows": len(projection_bytes.splitlines()),
        "files": {
            "qrels_access_receipt.json": _sha(receipt_bytes),
            "qrels_projection.jsonl": _sha(projection_bytes),
            "metrics.json": _sha(prior_metrics_bytes),
            "decision.json": _sha(prior_decision_bytes),
            "summary.json": _sha(prior_summary_bytes),
        },
    }
    monkeypatch.setattr(evaluate_module, "HISTORICAL_PRIOR_EVALUATION_IDENTITY", historical_anchor)

    aggregate = {
        "RRF": {
            "recall@500": 0.62,
            "graded_recall@500": 0.58,
            "recall@1000": 0.78,
            "graded_recall@1000": 0.74,
            "novel_retained@500": 70,
            "novel_retained@1000": 130,
            "judged_rate@500": 0.61,
        },
        "FACET-2B": {
            "recall@500": 0.64,
            "graded_recall@500": 0.60,
            "recall@1000": 0.79,
            "graded_recall@1000": 0.75,
            "novel_retained@500": 92,
            "novel_retained@1000": 145,
            "judged_rate@500": 0.58,
        },
        "TETHERED-2B": {
            "recall@500": 0.67,
            "graded_recall@500": 0.63,
            "recall@1000": 0.81,
            "graded_recall@1000": 0.77,
            "novel_retained@500": 101,
            "novel_retained@1000": 151,
            "judged_rate@500": 0.60,
        },
    }
    metrics_bytes = _write(
        task4 / "metrics.json",
        {"schema_version": "tethered-facet-evaluation-v1", "topic_ids": TOPICS, "aggregate": aggregate, "novel_relevant_total": 203},
    )
    representatives = [
        {
            "topic_id": "219",
            "facet_id": "219-positive",
            "movement": "promoted",
            "document_id": "doc-promoted",
            "narrative": "Explain the full policy narrative and its trade-offs.",
            "facet_query": "positive effects on rural communities",
            "selected_passage": "The program increased access while preserving local services.",
            "facet_only_percentile": 0.5,
            "tethered_percentile": 1.0,
            "qrels_grade": 2,
            "facet_only_final_rank": 611,
            "tethered_final_rank": 202,
            "prior_bm25_rank": 7,
            "passage_provenance": {
                "candidate_source_sha256": _sha(raw_paths["tethered_candidates"].read_bytes()),
                "window_score_source_sha256": _sha(raw_paths["tethered_window_scores"].read_bytes()),
                "document_score_source_sha256": _sha(raw_paths["tethered_document_scores"].read_bytes()),
                "query_sha256": tethered_candidates[0]["query_sha256"],
                "text_sha256": _sha(promoted_text.encode()),
                "window_sha256": _sha(promoted_text.encode()),
                "window_id": "w-promoted",
                "model": "synthetic/minilm",
                "model_revision": "fixture-revision",
                "document_start_token": 0,
                "document_end_token": 12,
                "rank_source": "prior_bm25_rank",
            },
            "ranking_provenance": {
                "task3_rankings_sha256": _sha(rankings),
                "facet_only_source": "dual_tail",
                "tethered_source": "facet_basket",
                "generating_facet": "219-positive",
                "percentile_method": "query_local_average_rank",
                "rank_source": "Task 3 sealed rankings.jsonl",
            },
        },
        {
            "topic_id": "84",
            "facet_id": "84-safety",
            "movement": "demoted",
            "document_id": "doc-demoted",
            "narrative": "Assess benefits, safety risks, and regulatory responses.",
            "facet_query": "reported safety incidents",
            "selected_passage": "A product name matched the facet but the passage concerned another domain.",
            "facet_only_percentile": 1.0,
            "tethered_percentile": 0.5,
            "qrels_grade": 0,
            "facet_only_final_rank": 204,
            "tethered_final_rank": 612,
            "prior_bm25_rank": 11,
            "passage_provenance": {
                "candidate_source_sha256": _sha(raw_paths["facet_candidates"].read_bytes()),
                "window_score_source_sha256": _sha(raw_paths["facet_window_scores"].read_bytes()),
                "document_score_source_sha256": None,
                "query_sha256": _sha(demoted_facet.encode()),
                "text_sha256": _sha(demoted_text.encode()),
                "window_sha256": _sha(demoted_text.encode()),
                "window_id": "w-demoted",
                "model": "synthetic/minilm",
                "model_revision": "fixture-revision",
                "document_start_token": 3,
                "document_end_token": 15,
                "rank_source": "prior_bm25_rank",
            },
            "ranking_provenance": {
                "task3_rankings_sha256": _sha(rankings),
                "facet_only_source": "facet_basket",
                "tethered_source": "dual_tail",
                "generating_facet": "84-safety",
                "percentile_method": "query_local_average_rank",
                "rank_source": "Task 3 sealed rankings.jsonl",
            },
        },
    ]
    diagnostics_bytes = _write(
        task4 / "diagnostics.json",
        {
            "schema_version": "tethered-facet-evaluation-v1",
            "topic_ids": TOPICS,
            "representatives": representatives,
            "per_topic_deltas": {
                topic: {
                    "recall@500_delta_tethered_vs_facet": 0.01 + index / 100,
                    "graded_recall@500_delta_tethered_vs_facet": 0.02,
                }
                for index, topic in enumerate(TOPICS)
            },
            "facet_yield": {
                "FACET-2B": {
                    "219-positive": {"selected_count": 50, "relevant_count": 8},
                    "84-safety": {"selected_count": 50, "relevant_count": 3},
                },
                "TETHERED-2B": {
                    "219-positive": {"selected_count": 50, "relevant_count": 12},
                    "84-safety": {"selected_count": 50, "relevant_count": 5},
                },
            },
            "basket_contributions": {
                "FACET-2B": {"facet_basket": {"selected_count": 800, "relevant_count": 92}},
                "TETHERED-2B": {"facet_basket": {"selected_count": 800, "relevant_count": 101}},
            },
            "novel_relevant_ids": novel_ids,
            "noise_pattern_definitions": {"wrong_domain": "wrong domain"},
            "noise_pattern_counts": [
                {"arm": arm, "facet_id": "219-positive", "selected_count": 50, "wrong_domain": int(arm == "FACET-2B")}
                for arm in ("FACET-2B", "TETHERED-2B")
            ],
            "facet_yield_changes": [{
                "facet_id": "219-positive", "facet_only_relevant_count": 8,
                "tethered_relevant_count": 12, "delta": 4, "classification": "rose",
            }],
            "relevant_below_500": [{
                "arm": "FACET-2B", "topic_id": "219", "document_id": "below-500",
                "qrels_grade": 2, "final_rank": 611, "best_facet": "219-positive",
                "best_facet_percentile": 0.7, "prior_bm25_rank": 9,
                "reason": "facet_quota_exhausted",
            }],
            "duplicate_and_quota_pressure": [{
                "topic_id": "219", "arm": "FACET-2B",
                "duplicate_skip_totals": {"219-positive": 2}, "duplicate_skip_total": 2,
                "shortage_counts": {"219-positive": 1}, "shortage_total": 1,
            }],
            "scoring_telemetry": {
                "preflight_source_sha256": _sha(preflight_bytes),
                "scoring_receipt_source_sha256": _sha(scoring_bytes),
                "model": "synthetic/minilm", "model_revision": "fixture-revision",
                "query_document_pair_count": 4800, "planned_window_count": 5000,
                "completed_window_count": 5200, "document_score_count": 4800,
                "cache_hit_count": 1000, "cache_miss_count": 4000,
                "unique_forward_pair_count": 4000, "unique_scoring_pair_count": 5000,
                "elapsed_seconds": 9.5,
                "projected_inference_seconds": 12.5,
                "peak_device_memory_bytes": 1234, "peak_host_memory_bytes": 5678,
            },
        },
    )
    decision_bytes = _write(
        task4 / "decision.json",
        {
            "label": "mechanical_pass",
            "guards": {"recall500": True, "novel500": True, "judged_coverage": True},
            "next_step": "Run a preregistered evaluation on fresh topics and untouched qrels.",
            "production_promotion_authorized": False,
            "novel_relevant_count": 203,
        },
    )
    evaluation_bindings_bytes = _write(
        task4 / "input_bindings.json",
        {
            "schema_version": "tethered-facet-evaluation-v1",
            "task3_root_sha256": json.loads(seal_bytes)["root_sha256"],
            "task3_seal": _binding(task3 / "SEALED.json"),
            "task3_input_bindings": _binding(task3 / "input_bindings.json"),
            "task3_rankings": _binding(task3 / "rankings.jsonl"),
            "task3_producer_sources": json.loads(bindings_bytes)["inputs"],
            "prior_freeze_root_sha256": "b" * 64,
            "prior_freeze_seal": _binding(prior_freeze / "SEALED.json"),
            "prior_freeze_rankings": _binding(prior_freeze / "rankings.jsonl"),
            "qrels_projection": _binding(prior / "qrels_projection.jsonl"),
            "qrels_access_receipt": _binding(prior / "qrels_access_receipt.json"),
            "prior_metrics": _binding(prior / "metrics.json"),
            "prior_decision": _binding(prior / "decision.json"),
            "prior_summary": _binding(prior / "summary.json"),
            "historical_integrity_anchor": historical_anchor,
            "historical_integrity_only": True,
            "blind_generalization_evidence": False,
            "original_qrels_opened": False,
        },
    )
    _write(
        task4 / "summary.json",
        {
            "schema_version": "tethered-facet-evaluation-v1",
            "status": "complete",
            "topic_ids": TOPICS,
            "post_qrels_diagnostic": True,
            "production_validation": False,
            "novel_relevant_count": 203,
            "metrics_sha256": _sha(metrics_bytes),
            "diagnostics_sha256": _sha(diagnostics_bytes),
            "decision_sha256": _sha(decision_bytes),
            "input_bindings_sha256": _sha(evaluation_bindings_bytes),
        },
    )
    return ReportSources(
        topic_ids=TOPICS.copy(),
        task1_receipt=source / "preflight.json",
        task2_receipt=source / "scoring_receipt.json",
        task3_freeze=task3,
        task4_evaluation=task4,
    )


@pytest.fixture
def built(tmp_path: Path, sources: ReportSources):
    return build_report(sources, tmp_path / "report")


def test_report_answers_the_three_user_questions(built) -> None:
    html = built.html.lower()
    assert "did narrative tethering reduce facet noise?" in html
    assert "did two-basket fusion recover novel relevant documents?" in html
    assert "what should happen next?" in html


def test_report_exposes_narrative_facet_and_document_evidence(built) -> None:
    row = built.artifact["representatives"][0]
    assert row["narrative"]
    assert row["facet_query"]
    assert row["facet_only_percentile"] is not None
    assert row["tethered_percentile"] is not None
    assert row["selected_passage"]
    assert row["qrels_grade"] >= 0
    assert row["facet_only_final_rank"] > 0
    assert row["tethered_final_rank"] > 0
    assert row["prior_bm25_rank"] > 0
    assert row["passage_provenance"]["window_sha256"]
    assert row["ranking_provenance"]["task3_rankings_sha256"]
    assert {row["movement"] for row in built.artifact["representatives"]} == {
        "promoted",
        "demoted",
    }


def test_report_is_deterministic_and_create_only(tmp_path: Path, sources: ReportSources) -> None:
    first = build_report(sources, tmp_path / "one")
    second = build_report(sources, tmp_path / "two")
    assert first.artifact_bytes == second.artifact_bytes
    assert first.summary_bytes == second.summary_bytes
    assert first.html_bytes == second.html_bytes
    assert (tmp_path / "one" / "report_data.sqlite").read_bytes() == (
        tmp_path / "two" / "report_data.sqlite"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        build_report(sources, tmp_path / "one")


def test_report_rejects_unbound_or_protected_sources(sources: ReportSources) -> None:
    sources.topic_ids.append("144")
    with pytest.raises(ValueError, match="protected topic 144"):
        build_artifact(sources)


def test_source_hashes_fail_closed(tmp_path: Path, sources: ReportSources) -> None:
    sources.task1_receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Task 1.*SHA-256"):
        build_report(sources, tmp_path / "report")


def test_task4_decoy_hash_cannot_replace_exact_task3_binding(
    sources: ReportSources,
) -> None:
    bindings_path = sources.task4_evaluation / "input_bindings.json"
    bindings = json.loads(bindings_path.read_text())
    correct = bindings["task3_seal"]["sha256"]
    bindings["task3_seal"]["sha256"] = "0" * 64
    bindings["decoy_nested_hash"] = {"sha256": correct}
    bindings_bytes = _bytes(bindings)
    bindings_path.write_bytes(bindings_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["input_bindings_sha256"] = _sha(bindings_bytes)
    summary_path.write_bytes(_bytes(summary))

    with pytest.raises(ValueError, match="exact source binding"):
        build_artifact(sources)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.__setitem__("facet_only_final_rank", 999), "rank provenance"),
        (
            lambda row: row["ranking_provenance"].__setitem__("task3_rankings_sha256", "0" * 64),
            "ranking provenance",
        ),
        (
            lambda row: row["passage_provenance"].__setitem__("candidate_source_sha256", "0" * 64),
            "passage provenance",
        ),
    ],
)
def test_restamped_representative_provenance_tamper_fails_closed(
    sources: ReportSources, mutate, message: str
) -> None:
    diagnostics_path = sources.task4_evaluation / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    mutate(diagnostics["representatives"][0])
    diagnostics_bytes = _bytes(diagnostics)
    diagnostics_path.write_bytes(diagnostics_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["diagnostics_sha256"] = _sha(diagnostics_bytes)
    summary_path.write_bytes(_bytes(summary))

    with pytest.raises(ValueError, match=message):
        build_artifact(sources)


def test_report_is_standalone_accessible_and_explicitly_bounded(built) -> None:
    html = built.html.lower()
    for phrase in ("post-qrels diagnostic", "no new retrieval", "not production validation"):
        assert phrase in html
    assert "<script src=" not in html and "<link rel=" not in html
    assert "<main" in html and "<h1" in html and "<caption" in html
    assert ":focus-visible" in html and "@media (max-width:" in html
    assert "rrf" in html and "facet-2b" in html and "tethered-2b" in html
    assert "recall@500" in html and "recall@1000" in html
    assert built.artifact["decision"]["label"] == "mechanical_pass"
    assert "/203" in html or "/ 203" in html
    assert "/177" not in html and "/ 177" not in html
    assert "facet-only final rank" in html and "tethered final rank" in html
    assert "prior facet bm25 rank" in html
    assert "passage provenance" in html and "ranking provenance" in html
    for phrase in ("noise patterns", "facet yield changes", "relevant below rank 500", "duplicate and quota pressure", "scoring telemetry"):
        assert phrase in html


def test_report_title_never_asserts_improvement_for_false_or_inconclusive_result(
    sources: ReportSources,
) -> None:
    decision_path = sources.task4_evaluation / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["label"] = "inconclusive"
    decision_bytes = _bytes(decision)
    decision_path.write_bytes(decision_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["decision_sha256"] = _sha(decision_bytes)
    summary_path.write_bytes(_bytes(summary))

    artifact = build_artifact(sources)

    assert "improve" not in artifact["title"].lower()
    assert "inconclusive" in artifact["title"].lower()


def test_report_rejects_reconciled_novel_count_mismatch_after_restamp(
    sources: ReportSources,
) -> None:
    metrics_path = sources.task4_evaluation / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["novel_relevant_total"] = 202
    metrics_bytes = _bytes(metrics)
    metrics_path.write_bytes(metrics_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics_sha256"] = _sha(metrics_bytes)
    summary_path.write_bytes(_bytes(summary))

    with pytest.raises(ValueError, match="novel relevant count"):
        build_artifact(sources)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda diagnostics: diagnostics["representatives"][0].__setitem__("qrels_grade", 3),
        lambda diagnostics: diagnostics["relevant_below_500"][0].__setitem__("qrels_grade", 1),
        lambda diagnostics: diagnostics["novel_relevant_ids"]["219"].__setitem__(0, "219-invented-novel"),
    ],
)
def test_report_rejects_restamped_qrels_derived_evidence(
    sources: ReportSources, mutate,
) -> None:
    diagnostics_path = sources.task4_evaluation / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    mutate(diagnostics)
    diagnostics_bytes = _bytes(diagnostics)
    diagnostics_path.write_bytes(diagnostics_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["diagnostics_sha256"] = _sha(diagnostics_bytes)
    summary_path.write_bytes(_bytes(summary))

    with pytest.raises(ValueError, match="qrels|novel"):
        build_artifact(sources)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("passage_provenance", "rank_source", "invented"),
        ("ranking_provenance", "percentile_method", "global_rank"),
        ("ranking_provenance", "rank_source", "invented.jsonl"),
    ],
)
def test_report_rejects_restamped_provenance_constants(
    sources: ReportSources, section: str, field: str, bad_value: str,
) -> None:
    diagnostics_path = sources.task4_evaluation / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["representatives"][0][section][field] = bad_value
    diagnostics_bytes = _bytes(diagnostics)
    diagnostics_path.write_bytes(diagnostics_bytes)
    summary_path = sources.task4_evaluation / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["diagnostics_sha256"] = _sha(diagnostics_bytes)
    summary_path.write_bytes(_bytes(summary))

    with pytest.raises(ValueError, match="provenance"):
        build_artifact(sources)


def test_task4_exact_producer_schema_matches_task5_consumer(sources: ReportSources) -> None:
    artifact = build_artifact(sources)
    assert set(artifact["diagnostics"]) == {
        "noise_pattern_definitions", "noise_pattern_counts", "facet_yield_changes",
        "relevant_below_500", "duplicate_and_quota_pressure", "scoring_telemetry",
    }
    assert set(artifact["representatives"][0]) == {
        "topic_id", "facet_id", "movement", "document_id", "narrative",
        "facet_query", "selected_passage", "facet_only_percentile",
        "tethered_percentile", "qrels_grade", "facet_only_final_rank",
        "tethered_final_rank", "prior_bm25_rank", "passage_provenance",
        "ranking_provenance",
    }


def test_html_renders_complete_authenticated_diagnostics_with_column_scopes(built) -> None:
    html = built.html
    lower = html.lower()
    for value in (
        "wrong_domain", "+4", "qrels grade", "best facet percentile",
        "prior bm25 rank", "219-positive: 2", "219-positive: 1",
        "preflight_source_sha256", "scoring_receipt_source_sha256",
        "unique_scoring_pair_count", "document_score_count",
    ):
        assert value.lower() in lower
    headers = re.findall(r"<thead><tr>(.*?)</tr></thead>", html, flags=re.DOTALL)
    assert headers
    assert all(
        all("scope=\"col\"" in tag or "scope='col'" in tag for tag in re.findall(r"<th\b[^>]*>", header))
        for header in headers
    )


def test_sqlite_companion_contains_exact_bounded_datasets(built) -> None:
    with sqlite3.connect(built.output_dir / "report_data.sqlite") as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert tables == {"metrics", "topic_contributions", "facet_contributions", "representatives"}
        assert connection.execute("SELECT count(*) FROM representatives").fetchone()[0] == 2


def test_existing_destination_may_contain_only_tracked_readme(
    tmp_path: Path, sources: ReportSources
) -> None:
    destination = tmp_path / "report"
    destination.mkdir()
    (destination / "README.md").write_text("tracked instructions\n", encoding="utf-8")
    result = build_report(sources, destination)
    assert result.output_dir == destination
    assert (destination / "README.md").read_text() == "tracked instructions\n"


def test_cli_has_only_offline_source_and_output_arguments(tmp_path: Path, sources: ReportSources) -> None:
    help_text = build_parser().format_help().lower()
    for forbidden in ("qrels", "endpoint", "server", "publish", "download"):
        assert forbidden not in help_text
    output = tmp_path / "cli-report"
    assert main([
        "--task1-receipt", str(sources.task1_receipt),
        "--task2-receipt", str(sources.task2_receipt),
        "--task3-freeze", str(sources.task3_freeze),
        "--task4-evaluation", str(sources.task4_evaluation),
        "--output", str(output),
    ]) == 0
    assert (output / "report.html").exists()


def test_readme_reproduces_offline_stages_without_server_or_publish_commands() -> None:
    readme = (
        Path(__file__).parents[2]
        / "reports/experiments/tethered_facet_minilm_diagnostic_v1/README.md"
    ).read_text(encoding="utf-8")
    assert ".venv/bin/python-rocm -m trec_rag.tethered_facet_minilm_score score" in readme
    assert ".venv/bin/python -m trec_rag.tethered_facet_two_basket freeze" in readme
    assert ".venv/bin/python -m trec_rag.tethered_facet_two_basket verify" in readme
    assert "--deep-root outputs/rag25_deep_facet_candidates_v1" in readme
    assert ".venv/bin/python -m trec_rag.tethered_facet_evaluate" in readme
    assert ".venv/bin/python -m trec_rag.build_tethered_facet_report" in readme
    lowered = readme.lower()
    assert "python-rocm -m trec_rag.build_tethered_facet_report" not in lowered
    assert "http.server" not in lowered and "publish" not in lowered

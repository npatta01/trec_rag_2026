from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

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


@pytest.fixture
def sources(tmp_path: Path) -> ReportSources:
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
    }
    scoring_bytes = _write(source / "scoring_receipt.json", scoring)

    rankings = b"".join(
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
        ).encode()
        + b"\n"
        for topic in TOPICS
        for arm in ("FACET-2B", "TETHERED-2B")
    )
    (task3 / "rankings.jsonl").write_bytes(rankings)
    bindings_bytes = _write(
        task3 / "input_bindings.json",
        {
            "task1_preflight_sha256": _sha(preflight_bytes),
            "task2_scoring_receipt_sha256": _sha(scoring_bytes),
        },
    )
    task3_summary_bytes = _write(
        task3 / "summary.json",
        {
            "status": "complete",
            "topic_ids": TOPICS,
            "qrels_opened": False,
            "rankings_sha256": _sha(rankings),
            "input_bindings_sha256": _sha(bindings_bytes),
        },
    )
    seal_bytes = _write(
        task3 / "SEALED.json",
        {
            "schema_version": "tethered-facet-two-basket-seal-v1",
            "status": "sealed",
            "files": {
                "input_bindings.json": _sha(bindings_bytes),
                "rankings.jsonl": _sha(rankings),
                "summary.json": _sha(task3_summary_bytes),
            },
        },
    )

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
        {"topic_ids": TOPICS, "aggregate": aggregate, "novel_relevant_total": 177},
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
            "facet_only_percentile": 0.73,
            "tethered_percentile": 0.94,
            "qrels_grade": 2,
        },
        {
            "topic_id": "84",
            "facet_id": "84-safety",
            "movement": "demoted",
            "document_id": "doc-demoted",
            "narrative": "Assess benefits, safety risks, and regulatory responses.",
            "facet_query": "reported safety incidents",
            "selected_passage": "A product name matched the facet but the passage concerned another domain.",
            "facet_only_percentile": 0.96,
            "tethered_percentile": 0.31,
            "qrels_grade": 0,
        },
    ]
    diagnostics_bytes = _write(
        task4 / "diagnostics.json",
        {
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
        },
    )
    decision_bytes = _write(
        task4 / "decision.json",
        {
            "label": "mechanical_pass",
            "guards": {"recall500": True, "novel500": True, "judged_coverage": True},
            "next_step": "Run a preregistered evaluation on fresh topics and untouched qrels.",
            "production_promotion_authorized": False,
        },
    )
    evaluation_bindings_bytes = _write(
        task4 / "input_bindings.json",
        {
            "task3_seal_sha256": _sha(seal_bytes),
            "task2_scoring_receipt_sha256": _sha(scoring_bytes),
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
    assert ".venv/bin/python -c" in readme
    assert "trec_rag.tethered_facet_two_basket import verify_freeze" in readme
    assert ".venv/bin/python -m trec_rag.tethered_facet_evaluate" in readme
    assert ".venv/bin/python -m trec_rag.build_tethered_facet_report" in readme
    lowered = readme.lower()
    assert "python-rocm -m trec_rag.build_tethered_facet_report" not in lowered
    assert "http.server" not in lowered and "publish" not in lowered

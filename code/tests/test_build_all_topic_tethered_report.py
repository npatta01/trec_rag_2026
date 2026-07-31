from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

import trec_rag.build_all_topic_tethered_report as report_module
from trec_rag.build_all_topic_tethered_report import (
    CANONICAL_EVALUATION_ROOT_SHA256,
    CANONICAL_RANKING_ROOT_SHA256,
    VERIFICATION_BUNDLE_SHA256,
    _readme,
    build_report,
    main,
    verify_report,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "outputs/all_topic_tethered_facet_validation_v1"
pytestmark = pytest.mark.skipif(
    not (SOURCE_ROOT / "rankings_v3/SEALED.json").is_file(),
    reason=(
        "private sealed fixture unavailable; restore "
        "cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst "
        "at the repository root with `tar --zstd -xf "
        "cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst "
        "-C .` after verifying "
        "the pinned SHA-256 in reports/experiments/all_topic_tethered_facet_validation_v1/README.md"
    ),
)
TOPICS = {
    "14", "31", "37", "58", "72", "84", "144", "161", "200", "213",
    "219", "224", "225", "233", "273", "300", "407", "477", "499",
    "515", "707", "897",
}


def test_report_names_every_topic_and_regression() -> None:
    built = build_report(SOURCE_ROOT)
    assert set(built.summary["topic_ids"]) == TOPICS
    assert built.summary["decision"]["selected_arm"] == "RRF"
    assert built.summary["decision"]["promoted"] is False
    assert built.summary["decision"]["primary_loss_topic_ids"] == ["31", "300"]
    assert "Topic 31" in built.html
    assert "Topic 300" in built.html


def test_report_uses_only_portable_canonical_roots_and_rejects_v1_v2() -> None:
    built = build_report(SOURCE_ROOT)
    assert built.summary["provenance"]["ranking_root_sha256"] == CANONICAL_RANKING_ROOT_SHA256
    assert built.summary["provenance"]["evaluation_root_sha256"] == CANONICAL_EVALUATION_ROOT_SHA256
    assert "v1/v2 ranking/evaluation rejected" in built.html

    with pytest.raises(ValueError, match="superseded v1/v2"):
        build_report(SOURCE_ROOT, ranking_dir_name="rankings", evaluation_dir_name="evaluation")


def test_report_states_retrospective_known_relevant_and_generation_limits() -> None:
    html = build_report(SOURCE_ROOT).html
    assert "retrospective full-development stress test" in html
    assert "known-relevant" in html
    assert "not evidence of generalization" in html
    assert "downstream RAG answer generation is out of scope" in html


def test_readme_documents_external_bundle_restore_and_identity() -> None:
    readme = _readme(build_report(SOURCE_ROOT).summary)
    assert VERIFICATION_BUNDLE_SHA256 in readme
    assert "tar --zstd -xf cache/experiments/" in readme
    assert "not stored in Git" in readme
    assert "--root outputs/all_topic_tethered_facet_validation_v1" in readme


def test_report_contains_requested_evidence_and_accessibility_contract() -> None:
    built = build_report(SOURCE_ROOT)
    compact = built.html.replace(" ", "")
    assert 'name="viewport"' in built.html
    assert "overflow-x:auto" in compact
    assert "Recall and retained facet evidence by depth" in built.html
    assert "Known-relevant yield falls with facet depth" in built.html
    assert "Exact preregistered selection ladder" in built.html
    assert "Method and evidence boundary" in built.html
    assert "Costs and execution accounting" in built.html
    assert "https://cdn" not in built.html
    assert "<script src=" not in built.html
    assert "<details" in built.html
    assert "<svg" in built.html


def test_report_separates_binary_and_graded_capture() -> None:
    built = build_report(SOURCE_ROOT)

    assert built.summary["capture"]["known_relevant_total"] == 12984
    assert built.summary["capture"]["graded_gain_total"] == 84560
    assert built.summary["capture"]["rrf_at_20"] == {
        "metric_source": "postmortem diagnostic",
        "known_relevant": 316,
        "binary_recall": pytest.approx(316 / 12984),
        "graded_gain": 2492,
        "graded_recall": pytest.approx(2492 / 84560),
    }
    assert built.summary["capture"]["rrf_at_1000"]["known_relevant"] == 3953
    assert built.summary["capture"]["rrf_at_1000"]["graded_gain"] == 28815
    assert built.summary["capture"]["full_union"]["binary_recall"] == pytest.approx(
        0.3866296980899569
    )

    full_union = next(
        row
        for row in built.datasets["depth_metrics"]
        if row["arm"] == "RRF" and row["depth"] == "full"
    )
    assert full_union == {
        "arm": "RRF",
        "depth": "full",
        "metric_source": "sealed evaluation",
        "binary_recall": pytest.approx(0.3866296980899569),
        "known_relevant_count": 5020,
        "known_relevant_total": 12984,
        "graded_gain": 35840,
        "graded_gain_total": 84560,
        "graded_recall": pytest.approx(0.423841059602649),
        "facet_only_retention": 1.0,
        "judged_rate": pytest.approx(0.1779638490164806),
    }
    diagnostic = next(
        row
        for row in built.datasets["depth_metrics"]
        if row["arm"] == "RRF" and row["depth"] == 20
    )
    assert diagnostic["metric_source"] == "postmortem diagnostic"
    assert diagnostic["known_relevant_count"] == 316
    assert diagnostic["graded_gain"] == 2492
    assert diagnostic["facet_only_retention"] is None


def test_report_recomputes_depth_20_capture_from_authenticated_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = report_module.POSTMORTEM_PATH
    forged = json.loads(source.read_text())
    capture = forged["rrf_capture_diagnostic_at_20"]
    capture["known_relevant_count"] = 315
    capture["binary_recall"] = 315 / capture["known_relevant_total"]
    capture["graded_gain"] = 2491
    capture["graded_recall"] = 2491 / capture["graded_gain_total"]
    forged_path = tmp_path / "postmortem.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True))
    monkeypatch.setattr(report_module, "POSTMORTEM_PATH", forged_path)

    with pytest.raises(ValueError, match="authenticated ranking and qrels"):
        build_report(SOURCE_ROOT)


def test_report_answers_capture_failure_and_rag_questions_progressively() -> None:
    built = build_report(SOURCE_ROOT)

    assert "How much graded evidence do we capture?" in built.html
    assert "Judgment-pool dependent" in built.html
    assert "Source-diverse evidence selection" in built.html
    assert "Topic 300 facet-tail replay" in built.html
    assert "Topic 31 cutoff mechanics" in built.html
    assert "Topic 300 cutoff mechanics" in built.html
    assert "cap 100" in built.html
    assert "+2" in built.html
    assert "316 / 12,984" in built.html
    assert "2,492 / 84,560" in built.html
    assert "Postmortem diagnostic" in built.html
    assert "Facet retrieval-stream memberships" in built.html
    assert "DUAL selection-coverage facets" in built.html
    assert "global warming climate change effective strategies prevent reduce" in built.html
    assert built.html.index("How much graded evidence do we capture?") < built.html.index(
        "What this result means—and what it does not"
    )
    assert built.html.count("<details") >= 3
    assert built.html.count("<summary") == built.html.count("<details")

    failures = {row["topic_id"]: row for row in built.datasets["failure_evidence"]}
    assert failures["31"]["primary_at_1000"]["known_relevant_delta"] == -7
    assert failures["300"]["primary_at_1000"]["known_relevant_delta"] == -3
    assert failures["300"]["recovery_replay"]["known_relevant_delta"] == 2
    assert failures["300"]["facet_bucket_yield"]["101-150"] == pytest.approx(0.06)
    incoming = failures["300"]["primary_boundary_attribution"]["incoming"]
    assert incoming["facet_retrieval_stream_memberships"]["300-strategies"]["total"] == 54
    assert incoming["dual_selection_coverage_facets"]["300-strategies"] == 276
    assert "primary_selection_coverage_facet" not in incoming


def test_written_json_sqlite_artifact_and_html_are_consistent(tmp_path: Path) -> None:
    built = write_report(SOURCE_ROOT, tmp_path)
    receipt = verify_report(tmp_path)
    assert receipt["verified"] is True

    summary = json.loads((tmp_path / "summary.json").read_text())
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert summary == built.summary
    assert artifact["summary_sha256"] == built.artifact["summary_sha256"]
    assert artifact["source_roots"] == summary["provenance"]

    with sqlite3.connect(tmp_path / "report_data.sqlite") as db:
        topics = {row[0] for row in db.execute("SELECT topic_id FROM topic_metrics")}
        arms = {row[0] for row in db.execute("SELECT arm FROM arm_metrics")}
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        depth_columns = {
            row[1] for row in db.execute("PRAGMA table_info(depth_metrics)")
        }
        rrf_full = db.execute(
            "SELECT known_relevant_total, graded_gain, graded_gain_total, graded_recall "
            "FROM depth_metrics WHERE arm = 'RRF' AND depth = 'full'"
        ).fetchone()
        rrf_20 = db.execute(
            "SELECT metric_source, known_relevant_count, known_relevant_total, "
            "graded_gain, graded_gain_total FROM depth_metrics "
            "WHERE arm = 'RRF' AND depth = '20'"
        ).fetchone()
    assert topics == TOPICS
    assert arms == set(summary["arms"])
    assert metadata["ranking_root_sha256"] == CANONICAL_RANKING_ROOT_SHA256
    assert metadata["evaluation_root_sha256"] == CANONICAL_EVALUATION_ROOT_SHA256
    assert {
        "known_relevant_total",
        "graded_gain",
        "graded_gain_total",
        "graded_recall",
    } <= depth_columns
    assert rrf_full == pytest.approx((12984, 35840, 84560, 0.423841059602649))
    assert rrf_20 == ("postmortem diagnostic", 316, 12984, 2492, 84560)


def test_write_subcommand_uses_canonical_report_flag(tmp_path: Path) -> None:
    assert main([
        "write",
        "--root",
        str(SOURCE_ROOT),
        "--report",
        str(tmp_path),
    ]) == 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.html").is_file()


def _copy_report(destination: Path) -> None:
    write_report(SOURCE_ROOT, destination)


def test_verify_rebuilds_from_sources_and_rejects_coordinated_forgery(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _copy_report(report)
    summary_path = report / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["decision"]["recommendation"] = "promote DUAL"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    html_path = report / "report.html"
    html_path.write_text(html_path.read_text().replace("Retain RRF.", "Promote DUAL."))
    artifact_path = report / "artifact.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    artifact["html_sha256"] = hashlib.sha256(html_path.read_bytes()).hexdigest()
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="canonical rebuild"):
        verify_report(report, SOURCE_ROOT)


def test_verify_rejects_sqlite_metric_forgery(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _copy_report(report)
    with sqlite3.connect(report / "report_data.sqlite") as db:
        db.execute("UPDATE topic_metrics SET payload_json = '{}' WHERE topic_id = '31'")
    with pytest.raises(ValueError, match="SQLite"):
        verify_report(report, SOURCE_ROOT)


def test_cost_sources_are_authenticated_before_reporting(tmp_path: Path) -> None:
    root = tmp_path / "source"
    for relative in ("rankings_v3", "evaluation_v3"):
        shutil.copytree(SOURCE_ROOT / relative, root / relative)
    for relative, names in {
        "planning": ("SEALED.json",),
        "retrieval": ("RETRIEVAL_SEALED.json", "retrieval_summary.json"),
        "scoring": ("SCORE_PLAN_SEALED.json", "SCORING_SEALED.json", "scoring_receipt.json"),
    }.items():
        (root / relative).mkdir(parents=True)
        for name in names:
            shutil.copy2(SOURCE_ROOT / relative / name, root / relative / name)
    path = root / "retrieval/retrieval_summary.json"
    value = json.loads(path.read_text()); value["facet_request_count"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="retrieval_summary"):
        build_report(root)


def test_tables_are_keyboard_regions_and_statistics_are_exact() -> None:
    html = build_report(SOURCE_ROOT).html
    assert html.count('class="table-wrap" tabindex="0" role="region" aria-label=') == 3
    assert ".table-wrap:focus-visible" in html
    assert "paired exact sign-flip test" in html
    assert "+3.394 percentage points" in html
    assert "95% bootstrap CI +2.118 to +5.052" in html
    assert "raw p=0.000002623" in html
    assert "Holm-adjusted p=0.000006676" in html


def test_recall_chart_domain_is_derived_from_observed_data() -> None:
    built = build_report(SOURCE_ROOT)
    html = built.html
    expected_max = max(
        row["binary_recall"]
        for row in built.datasets["depth_metrics"]
        if row["arm"] in {"RRF", "RRF100-STATIC-DUAL", "RRF500-REINIT-DUAL"}
    ) * 1.05
    assert 'data-y-max="0.36"' not in html
    assert f'data-y-max="{expected_max:.4f}"' in html

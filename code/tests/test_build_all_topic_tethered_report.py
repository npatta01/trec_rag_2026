from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from trec_rag.build_all_topic_tethered_report import (
    CANONICAL_EVALUATION_ROOT_SHA256,
    CANONICAL_RANKING_ROOT_SHA256,
    build_report,
    verify_report,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "outputs/all_topic_tethered_facet_validation_v1"
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


def test_report_uses_only_corrected_canonical_roots_and_rejects_v1() -> None:
    built = build_report(SOURCE_ROOT)
    assert built.summary["provenance"]["ranking_root_sha256"] == CANONICAL_RANKING_ROOT_SHA256
    assert built.summary["provenance"]["evaluation_root_sha256"] == CANONICAL_EVALUATION_ROOT_SHA256
    assert "v1 ranking/evaluation rejected" in built.html

    with pytest.raises(ValueError, match="superseded v1"):
        build_report(SOURCE_ROOT, ranking_dir_name="rankings", evaluation_dir_name="evaluation")


def test_report_states_retrospective_known_relevant_and_generation_limits() -> None:
    html = build_report(SOURCE_ROOT).html
    assert "retrospective full-development stress test" in html
    assert "known-relevant" in html
    assert "not evidence of generalization" in html
    assert "downstream RAG answer generation is out of scope" in html


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
    assert topics == TOPICS
    assert arms == set(summary["arms"])
    assert metadata["ranking_root_sha256"] == CANONICAL_RANKING_ROOT_SHA256
    assert metadata["evaluation_root_sha256"] == CANONICAL_EVALUATION_ROOT_SHA256

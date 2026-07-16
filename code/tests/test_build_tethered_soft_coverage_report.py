from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from trec_rag.build_tethered_soft_coverage_report import (
    build_report_payload,
    render_report,
    write_report,
)


ROOT = Path(__file__).resolve().parents[2]
FREEZE = (
    ROOT
    / "outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze"
)
EVALUATION = (
    ROOT
    / "outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/evaluation"
)
PRIOR_SUMMARY = (
    ROOT / "reports/experiments/tethered_facet_minilm_diagnostic_v2/summary.json"
)


def _payload() -> dict[str, object]:
    return build_report_payload(FREEZE, EVALUATION, PRIOR_SUMMARY)


def test_report_separates_proxy_from_final_rag_quality() -> None:
    html = render_report(_payload())
    assert "This is not answer-generation evaluation" in html
    assert "qrels-positive facet exposure is only a proxy" in html
    assert "31.1% answer coverage" in html
    assert "separate answer-generation worktree" in html


def test_report_has_no_sensitive_paths_or_raw_documents() -> None:
    html = render_report(_payload())
    assert "/home/" not in html
    assert "window_text" not in html
    assert "API_KEY" not in html
    assert "document_id" not in html


def test_report_contains_required_comparisons_and_sources() -> None:
    html = render_report(_payload())
    for value in (
        "RRF",
        "NARRATIVE",
        "FIXED-O0",
        "TETHERED-DUAL",
        "RRF100-TETHERED-DUAL",
    ):
        assert value in html
    assert "Sources and reproducibility" in html
    assert "712" in html
    assert "764" in html
    assert "+52" in html
    assert "875 / 2,817" in html


def test_payload_preserves_metric_semantics_and_exact_findings() -> None:
    payload = _payload()
    findings = payload["findings"]
    assert findings["rrf_relevant_at_1000"] == 712
    assert findings["protected_relevant_at_1000"] == 764
    assert findings["relevant_delta_at_1000"] == 52
    assert findings["rrf_facet_only_at_1000"] == 95
    assert findings["protected_facet_only_at_1000"] == 160
    assert findings["full_union_relevant"] == 875
    assert findings["total_relevant"] == 2817
    assert findings["full_union_recall"] == 875 / 2817
    assert payload["metric_definitions"]["binary_recall"]["aggregation"] == "pooled_micro"
    assert payload["metric_definitions"]["ndcg"]["aggregation"] == "macro_topic_mean"
    assert payload["metric_definitions"]["recall_auc"]["aggregation"] == "macro_topic_mean"


def test_payload_names_every_direct_source_without_machine_paths() -> None:
    sources = _payload()["source_hashes"]
    assert "accepted_union_jsonl" in sources
    assert "qrels_projection_jsonl" in sources
    assert sources["accepted_union_jsonl"]["sha256"] == (
        "1f4c732a8499fa4e6cef965837dc4e39959ba84a0b7199978b1f27fcf617784a"
    )
    assert sources["qrels_projection_jsonl"]["sha256"] == (
        "03fc4bd18be36b7ea2d446975fec9fe17ac6698dcf068918c6bb228e9aab5e87"
    )
    assert "/home/" not in json.dumps(sources)


def test_overlap_decomposition_is_exact_and_nonadditive() -> None:
    payload = _payload()
    rows = payload["overlap_decomposition"]
    assert len(rows) == 4
    assert sum(row["union_relevant"] for row in rows) == 875
    assert sum(row["facet_only_relevant"] for row in rows) == 177
    assert all(
        row["union_relevant"]
        == row["original_only_relevant"]
        + row["overlap_relevant"]
        + row["facet_only_relevant"]
        for row in rows
    )
    html = render_report(payload)
    assert "<table" in html
    assert "nonadditive" in html
    assert "stacked chart" not in html.lower()


def test_write_report_emits_four_canonical_artifacts(tmp_path: Path) -> None:
    receipt = write_report(FREEZE, EVALUATION, PRIOR_SUMMARY, tmp_path)
    assert set(path.name for path in tmp_path.iterdir()) == {
        "artifact.json",
        "summary.json",
        "report_data.sqlite",
        "report.html",
    }
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert artifact["chart_omissions"][0]["reason"].startswith(
        "Exact overlap counts are more audit-friendly"
    )
    assert summary["external_calls"] == {
        "retrieval": 0,
        "inference": 0,
        "model_load": 0,
        "hosted_inference": 0,
        "network": 0,
        "paid": 0,
        "cost_usd": 0.0,
    }
    assert receipt["status"] == "complete"
    assert receipt["files"]["report.html"]["sha256"]
    with sqlite3.connect(tmp_path / "report_data.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM arm_metrics").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM overlap_decomposition").fetchone() == (4,)


def test_rendered_report_has_accessible_document_structure() -> None:
    html = render_report(_payload())
    assert '<a class="skip-link" href="#main">Skip to main content</a>' in html
    assert '<main id="main"' in html
    assert "<caption>" in html
    assert 'scope="col"' in html
    assert 'aria-label="Recall-depth comparison"' in html
    assert ":focus-visible" in html
    assert "prefers-reduced-motion" in html

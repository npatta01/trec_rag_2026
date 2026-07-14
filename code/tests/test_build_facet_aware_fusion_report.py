from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from trec_rag.build_facet_aware_fusion_report import (
    CANONICAL_SOURCE_PATHS,
    PILOT_TOPICS,
    PROTECTED_TOPICS,
    ReportInputs,
    build_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ADVISOR_REVIEW = """## Independent advisor review

### Verdict

TEST FIXTURE: Retain RRF. The facet candidate expansion is real, but the tested xQuAD weighting is too aggressive.

### Recommended next experiment

TEST FIXTURE: Freeze a capped facet-insertion arm that preserves the RRF backbone.
"""


@pytest.fixture(scope="module")
def report_inputs() -> ReportInputs:
    source_bytes = {
        source_id: (REPO_ROOT / relative_path).read_bytes()
        for source_id, relative_path in CANONICAL_SOURCE_PATHS.items()
        if source_id != "advisor_review"
    }
    source_bytes["advisor_review"] = TEST_ADVISOR_REVIEW.encode()
    return ReportInputs(
        source_bytes=source_bytes,
        source_paths=dict(CANONICAL_SOURCE_PATHS),
    )


@pytest.fixture(scope="module")
def built(report_inputs):
    return build_report(report_inputs)


def _report_text(artifact: dict[str, object]) -> str:
    return "\n".join(
        str(block.get("body", ""))
        for block in artifact["manifest"]["blocks"]
        if block["type"] == "markdown"
    )


def _replace_json_source(
    inputs: ReportInputs,
    source_id: str,
    transform,
) -> ReportInputs:
    payload = json.loads(inputs.source_bytes[source_id])
    transform(payload)
    changed = dict(inputs.source_bytes)
    changed[source_id] = json.dumps(
        payload, indent=2, sort_keys=True
    ).encode() + b"\n"
    return replace(inputs, source_bytes=changed)


def test_report_is_answer_first_and_reproduces_the_mechanical_decision(built):
    artifact, summary = built
    blocks = artifact["manifest"]["blocks"]
    text = _report_text(artifact)

    assert blocks[0]["body"].startswith("# ")
    assert blocks[1]["id"] == "technical_summary"
    assert summary["decision"]["selected_arm"] == "RRF"
    assert summary["decision"]["no_new_arm_promoted"] is True
    assert "retain the current RRF" in text
    assert "66 of 73" in text
    assert "over-replaced" in text
    assert "24" in blocks[1]["body"]
    assert "$0" in blocks[1]["body"]


def test_report_reproduces_metrics_without_recomputing_rankings(built):
    artifact, summary = built
    systems = {
        row["arm"]: row
        for row in artifact["snapshot"]["datasets"]["system_metric_rows"]
    }

    assert list(systems) == ["O", "RRF", "TUS-C", "BI", "XQ", "CXQ"]
    assert systems["RRF"]["ndcg_at_10"] == pytest.approx(0.5894638651596213)
    assert systems["RRF"]["graded_recall_at_100"] == pytest.approx(
        0.11098237307135614
    )
    assert systems["XQ"]["ndcg_at_10"] == pytest.approx(0.2835604646367565)
    assert systems["CXQ"] == {
        **systems["XQ"],
        "arm": "CXQ",
    }
    assert summary["systems"]["RRF"]["ndcg@10"] == pytest.approx(
        systems["RRF"]["ndcg_at_10"]
    )


def test_native_charts_show_ndcg_and_applicable_novel_retention(built):
    artifact, _ = built
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert list(charts) == ["system_ndcg", "novel_retention"]
    assert charts["system_ndcg"]["type"] == "bar"
    assert charts["system_ndcg"]["dataset"] == "system_metric_rows"
    assert charts["novel_retention"]["type"] == "bar"
    assert charts["novel_retention"]["dataset"] == "novel_retention_rows"

    retention = artifact["snapshot"]["datasets"]["novel_retention_rows"]
    assert [row["arm"] for row in retention] == ["TUS-C", "BI", "XQ", "CXQ"]
    assert retention[-1]["retained_relevant"] == 66
    assert retention[-1]["eligible_relevant"] == 73
    assert retention[-1]["retention_fraction"] == pytest.approx(
        0.9041095890410958
    )


def test_novel_denominator_caveat_prevents_false_rrf_failure_claim(built):
    artifact, summary = built
    text = _report_text(artifact)

    assert summary["novel_relevant"]["denominator"] == 73
    assert summary["novel_relevant"]["denominator_definition"]
    assert "absent from RRF@100" in text
    assert "RRF's zero is true by construction" in text
    assert "RRF failed to retain" not in text


def test_gate_and_representative_evidence_is_bounded_and_auditable(built):
    artifact, summary = built
    datasets = artifact["snapshot"]["datasets"]

    assert summary["facets"] == {"planned": 24, "accepted": 20, "rejected": 4}
    assert len(datasets["facet_gate_rows"]) == 24
    assert sum(row["accepted"] for row in datasets["facet_gate_rows"]) == 20
    assert len(datasets["rejected_facet_example_rows"]) == 4
    assert all(len(row["passage_excerpt"]) <= 520 for row in datasets["rejected_facet_example_rows"])
    assert all(row["diagnostic_only"] for row in datasets["rejected_facet_example_rows"])


def test_xq_and_cxq_identity_and_inactive_deadline_are_explicit(built):
    artifact, summary = built
    text = _report_text(artifact)

    assert summary["fusion"]["xq_cxq_byte_identical"] is True
    assert summary["fusion"]["cxq_forced_selection_count"] == 0
    assert "byte-identical" in text
    assert "deadline never fired" in text


def test_advisor_review_is_verbatim_and_verdict_is_extracted(built):
    artifact, summary = built
    advisor_block = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "advisor_review"
    )

    assert advisor_block["body"] == TEST_ADVISOR_REVIEW.rstrip()
    assert summary["advisor"]["verdict"] == (
        "TEST FIXTURE: Retain RRF. The facet candidate expansion is real, "
        "but the tested xQuAD weighting is too aggressive."
    )
    assert summary["advisor"]["recommended_next_experiment"].startswith(
        "TEST FIXTURE: Freeze"
    )


def test_sources_are_canonical_relative_and_every_native_visual_is_sourced(built):
    artifact, _ = built
    sources = {source["id"]: source for source in artifact["sources"]}
    assert sources
    for source in sources.values():
        assert not Path(source["path"]).is_absolute()
        assert ".." not in Path(source["path"]).parts
        assert source["source_sha256"]

    declared = {source["id"] for source in artifact["manifest"]["sources"]}
    assert declared == set(sources)
    for collection in ("charts", "tables"):
        for item in artifact["manifest"][collection]:
            assert item["sourceId"] in sources


def test_protected_or_mismatched_topics_fail_closed(report_inputs):
    def add_protected(payload):
        payload["topic_ids"][0] = sorted(PROTECTED_TOPICS)[0]

    tampered = _replace_json_source(report_inputs, "metrics", add_protected)
    with pytest.raises(ValueError, match="topic boundary"):
        build_report(tampered)

    def reorder_topics(payload):
        payload["topic_ids"] = list(reversed(PILOT_TOPICS))

    tampered = _replace_json_source(report_inputs, "decision", reorder_topics)
    with pytest.raises(ValueError, match="topic boundary"):
        build_report(tampered)


def test_report_input_contract_has_no_raw_qrels_source(report_inputs):
    assert set(report_inputs.source_bytes) == set(CANONICAL_SOURCE_PATHS)
    assert all("projection" not in path for path in report_inputs.source_paths.values())
    assert all(not path.endswith(".qrels") for path in report_inputs.source_paths.values())

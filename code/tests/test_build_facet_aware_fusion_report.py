from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from trec_rag.build_facet_aware_fusion_report import (
    CANONICAL_SOURCE_PATHS,
    PILOT_TOPICS,
    PROTECTED_TOPICS,
    REPORT_TITLE,
    ReportInputs,
    build_report,
    main,
)
from trec_rag.facet_aware_fusion_evaluate import add_self_hash


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
    assert list(charts) == ["system_ndcg", "novel_retention", "document_flow"]
    assert charts["system_ndcg"]["type"] == "bar"
    assert charts["system_ndcg"]["dataset"] == "system_metric_rows"
    assert charts["novel_retention"]["type"] == "bar"
    assert charts["novel_retention"]["dataset"] == "novel_retention_rows"
    assert charts["document_flow"]["type"] == "funnel"
    assert charts["document_flow"]["dataset"] == "document_flow_rows"

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


def test_advisor_only_counts_are_not_duplicated_as_first_party_findings(built):
    artifact, _ = built
    first_party = "\n".join(
        block.get("body", "")
        for block in artifact["manifest"]["blocks"]
        if block["type"] == "markdown" and block["id"] != "advisor_review"
    )
    for advisor_only_count in ("83 judged-relevant", "311 / 400", "replaced 290", "187 were unjudged"):
        assert advisor_only_count not in first_party
    assert "original and RRF pools" not in first_party
    assert "because their relevance term" not in first_party


def test_retrieval_receipt_must_confirm_qrels_were_closed(report_inputs):
    tampered = _replace_json_source(
        report_inputs,
        "retrieval_summary",
        lambda payload: payload.__setitem__("qrels_opened", True),
    )
    with pytest.raises(ValueError, match="pre-freeze qrels state"):
        build_report(tampered)


def test_evaluation_schema_binding_fails_closed(report_inputs):
    def change_schema(payload):
        payload["schema_version"] = "unexpected-schema"
        repaired = add_self_hash(payload)
        payload.clear()
        payload.update(repaired)

    tampered = _replace_json_source(report_inputs, "metrics", change_schema)
    with pytest.raises(ValueError, match="metrics schema binding"):
        build_report(tampered)


def test_preflight_lineage_binding_fails_closed(report_inputs):
    tampered = _replace_json_source(
        report_inputs,
        "scoring_preflight",
        lambda payload: payload.__setitem__("windows_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="pipeline lineage hash binding"):
        build_report(tampered)


def test_document_flow_and_visible_provenance_are_present(built):
    artifact, _ = built
    datasets = artifact["snapshot"]["datasets"]
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    blocks = {block["id"]: block for block in artifact["manifest"]["blocks"]}

    assert datasets["document_flow_rows"] == [
        {"stage": "Residual relevant facet candidates", "document_count": 73},
        {"stage": "Retained by XQ/CXQ", "document_count": 66},
    ]
    assert tables["source_provenance"]["dataset"] == "source_provenance_rows"
    assert blocks["provenance_table"]["tableId"] == "source_provenance"
    assert all(len(row["sha256"]) == 64 for row in datasets["source_provenance_rows"])


def test_mixed_run_evaluation_binding_fails_closed(report_inputs):
    def change_freeze_binding(payload):
        payload["bindings"]["ranking_freeze_sha256"] = "0" * 64
        repaired = add_self_hash(payload)
        payload.clear()
        payload.update(repaired)

    tampered = _replace_json_source(report_inputs, "gains_losses", change_freeze_binding)
    with pytest.raises(ValueError, match="mixed-run evaluation binding"):
        build_report(tampered)


@pytest.mark.parametrize(
    ("source_id", "mutation"),
    [
        (
            "facet_gates",
            lambda rows: rows[1].__setitem__(
                "content_warning_top5_count",
                rows[1]["content_warning_top5_count"] + 1,
            ),
        ),
        (
            "cxq_provenance",
            lambda rows: rows[0].__setitem__("deadline", 999),
        ),
        (
            "retrieval_candidates",
            lambda rows: rows[0].__setitem__("text", rows[0]["text"] + " tampered"),
        ),
    ],
)
def test_frozen_non_json_artifact_bindings_fail_closed(
    report_inputs, source_id, mutation
):
    if source_id == "facet_gates":
        rows = json.loads(report_inputs.source_bytes[source_id])
        mutation(rows)
        changed_bytes = json.dumps(rows, indent=2, sort_keys=True).encode() + b"\n"
    else:
        rows = [
            json.loads(line)
            for line in report_inputs.source_bytes[source_id].decode().splitlines()
            if line
        ]
        mutation(rows)
        changed_bytes = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
    changed = dict(report_inputs.source_bytes)
    changed[source_id] = changed_bytes
    with pytest.raises(ValueError, match="artifact hash binding"):
        build_report(replace(report_inputs, source_bytes=changed))


def test_cli_can_reproduce_html_through_configured_portable_packager(tmp_path):
    packager = tmp_path / "fake_packager.mjs"
    packager.write_text(
        """import fs from 'node:fs';
const input = process.argv[process.argv.indexOf('--input') + 1];
const output = process.argv[process.argv.indexOf('--output') + 1];
const artifact = JSON.parse(fs.readFileSync(input, 'utf8'));
fs.writeFileSync(output, `<html><title>${artifact.manifest.title}</title></html>`);
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    summary = tmp_path / "summary.json"
    html = tmp_path / "report.html"
    temp = tmp_path / "temp"

    assert main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--artifact",
            str(artifact),
            "--summary",
            str(summary),
            "--output",
            str(html),
            "--portable-delivery-script",
            str(packager),
            "--tmpdir",
            str(temp),
        ]
    ) == 0
    assert REPORT_TITLE in html.read_text(encoding="utf-8")

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.build_facet_local_minilm_report import (
    build_artifact,
    main,
    render_html_create_only,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports/experiments/facet_local_minilm_pilot_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs/rag25_facet_local_minilm_v1"
PILOT_TOPICS = {"200", "225", "707", "897"}
PROTECTED_TOPICS = {"144", "213", "224", "407", "515"}

SOURCE_FILES = {
    "manifest": REPORT_DIR / "manifest.json",
    "preflight": OUTPUT_ROOT / "preflight_v2/preflight.json",
    "scoring_receipt": OUTPUT_ROOT / "full_scoring_v1/scoring_receipt.json",
    "benchmark": OUTPUT_ROOT / "benchmark_v1/benchmark_telemetry.json",
    "model_download_approval": OUTPUT_ROOT / "approvals/model_download_v1.json",
    "benchmark_approval": OUTPUT_ROOT / "approvals/benchmark_v1.json",
    "full_scoring_approval": OUTPUT_ROOT / "approvals/full_scoring_v1.json",
    "ranking_freeze": OUTPUT_ROOT / "freeze_v1/freeze.json",
    "legacy_corrected_diff": OUTPUT_ROOT / "freeze_v1/legacy_corrected_diff.json",
    "review_freeze": OUTPUT_ROOT / "review_v3/review_freeze.json",
    "review_create_receipt": OUTPUT_ROOT / "review_v3/create_receipt.json",
    "raw_union": OUTPUT_ROOT / "evaluation_v1/raw_union.json",
    "prefusion": OUTPUT_ROOT / "evaluation_v1/prefusion.json",
    "facet_retention": OUTPUT_ROOT / "evaluation_v1/facet_retention.json",
    "systems": OUTPUT_ROOT / "evaluation_v1/systems.json",
    "gains_losses": OUTPUT_ROOT / "evaluation_v1/gains_losses.json",
    "review_metrics": OUTPUT_ROOT / "evaluation_v1/review_metrics.json",
    "representatives": OUTPUT_ROOT / "evaluation_v1/representatives.json",
    "representative_provenance": OUTPUT_ROOT
    / "derived_v2/representative_provenance_v2.json",
    "decision": OUTPUT_ROOT / "evaluation_v1/decision.json",
    "qrels_access_approval": OUTPUT_ROOT / "approvals/qrels_access_v1.json",
    "qrels_consumption_registry": OUTPUT_ROOT
    / "approvals/qrels_access_v1.json.consumed.json",
    "qrels_access_receipt": OUTPUT_ROOT
    / "evaluation_v1/qrels_access_receipt.json",
}

MATERIAL_INPUTS = {
    "decision": (
        "decision",
        "raw_union",
        "prefusion",
        "systems",
        "legacy_corrected_diff",
        "review_metrics",
    ),
    "qrels_access_receipt": (
        "qrels_access_receipt",
        "qrels_access_approval",
        "qrels_consumption_registry",
        "ranking_freeze",
        "review_freeze",
        "review_create_receipt",
    ),
    "representative_provenance": (
        "representative_provenance",
        "representatives",
        "prefusion",
        "ranking_freeze",
    ),
}


def _loaded_sources() -> tuple[dict[str, object], dict[str, bytes]]:
    payloads: dict[str, object] = {}
    raw: dict[str, bytes] = {}
    for key, path in SOURCE_FILES.items():
        raw[key] = path.read_bytes()
        payloads[key] = json.loads(raw[key])
    return payloads, raw


@pytest.fixture(scope="module")
def inputs():
    artifacts, artifact_bytes = _loaded_sources()
    return {
        "artifacts": artifacts,
        "artifact_bytes": artifact_bytes,
        "decision": artifacts["decision"],
    }


@pytest.fixture(scope="module")
def artifact(inputs):
    return build_artifact(**inputs)


def _report_text(artifact: dict[str, object]) -> str:
    return "\n".join(
        block.get("body", "") for block in artifact["manifest"]["blocks"]
    )


def test_report_leads_with_candidate_headroom_and_diagnostic_outcome(inputs):
    artifact = build_artifact(**inputs)
    text = _report_text(artifact)
    assert "raw union" in text.lower()
    assert inputs["decision"]["decision"]["outcome"] in text
    assert "nDCG@10 is a guardrail" in text


def test_report_cannot_hide_relevant_losses(inputs):
    artifact = build_artifact(**inputs)
    rows = artifact["snapshot"]["datasets"]["gain_loss_rows"]
    assert rows
    assert all("gained" in row and "lost" in row and "net_change" in row for row in rows)


def test_report_accounts_for_all_exact_streams_and_narratives(artifact, inputs):
    datasets = artifact["snapshot"]["datasets"]
    query_rows = datasets["query_rows"]
    narrative_rows = datasets["narrative_rows"]
    streams = inputs["artifacts"]["manifest"]["streams"]
    assert len(query_rows) == 31
    assert sum(row["family"] == "original" for row in query_rows) == 4
    assert sum(row["family"] == "facet" for row in query_rows) == 27
    assert sum(row["candidate_rows"] for row in query_rows) == 3100
    assert [(row["topic_id"], row["variant"], row["full_query"]) for row in query_rows] == [
        (row["topic_id"], row["variant"], row["query"]) for row in streams
    ]
    assert len(narrative_rows) == 4
    assert {row["topic_id"] for row in narrative_rows} == PILOT_TOPICS
    assert all(row["full_narrative"] for row in narrative_rows)


def test_every_source_hashes_the_loaded_saved_artifact(artifact, inputs):
    sources = {source["id"]: source for source in artifact["sources"]}
    assert set(sources) == set(SOURCE_FILES)
    for source_id, source in sources.items():
        material_ids = MATERIAL_INPUTS.get(source_id, (source_id,))
        expected_inputs = {
            str(SOURCE_FILES[input_id].relative_to(REPO_ROOT)): hashlib.sha256(
                inputs["artifact_bytes"][input_id]
            ).hexdigest()
            for input_id in material_ids
        }
        assert source["query"]["input_sha256"] == expected_inputs
        if len(expected_inputs) == 1:
            expected_query_hash = next(iter(expected_inputs.values()))
        else:
            encoded = json.dumps(
                expected_inputs, separators=(",", ":"), sort_keys=True
            ).encode()
            expected_query_hash = hashlib.sha256(encoded).hexdigest()
        assert source["query"]["id"] == f"sha256:{expected_query_hash}"
        assert not Path(source["path"]).is_absolute()
        assert ".." not in Path(source["path"]).parts
        assert source["query"]["tables_used"] == list(expected_inputs)
    declared = {source["id"] for source in artifact["manifest"]["sources"]}
    assert declared == set(sources)


def test_model_download_runtime_and_capped_window_evidence_are_visible(artifact):
    rows = artifact["snapshot"]["datasets"]["runtime_rows"]
    assert rows == [
        {
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
            "safe_file_count": 6,
            "safe_files_only": True,
            "execution_backend": "rocm",
            "device": "Radeon 8060S Graphics",
            "inference_dtype": "float32",
            "planned_windows": 14720,
            "completed_windows": 14720,
            "unique_scored_pairs": 14459,
            "capped_documents": 30,
            "maximum_windows_per_document": 32,
            "coverage_min_percent": pytest.approx(29.52450621799561),
            "coverage_median_percent": 100.0,
            "coverage_p95_percent": 100.0,
        }
    ]
    text = _report_text(artifact)
    assert "tokenizer-only preflight" in text
    assert "safe-file allowlist" in text
    assert "bounded windows" in text


def test_qrels_receipt_review_caveat_and_shared_attribution_are_explicit(artifact):
    text = _report_text(artifact)
    audit = artifact["snapshot"]["datasets"]["firewall_rows"]
    assert audit[0]["qrels_access_status"] == "qrels_access_consumed"
    assert audit[0]["qrels_access_count"] == 1
    assert audit[0]["review_frozen_before_qrels"] is True
    assert audit[0]["ranking_frozen_before_qrels"] is True
    assert audit[0]["review_unique_items"] == 101
    assert audit[0]["review_memberships"] == 108
    assert audit[0]["shared_items_attributed_to_both_arms"] == 7
    assert "pooled and non-exhaustive" in text
    assert "shared passages are attributed to both arm memberships" in text


def test_retention_and_firewall_evidence_are_reader_visible(artifact):
    datasets = artifact["snapshot"]["datasets"]
    retention = datasets["facet_retention_rows"]
    assert len(retention) == 54
    assert {row["arm"] for row in retention} == {"C0", "BF100"}
    assert all(
        {"relevant_at_k20", "relevant_at_k50", "relevant_at_k100"} <= row.keys()
        for row in retention
    )
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    blocks = artifact["manifest"]["blocks"]
    table_refs = {block.get("tableId") for block in blocks if block["type"] == "table"}
    assert tables["facet_retention"]["sourceId"] == "facet_retention"
    assert tables["evaluation_firewall"]["sourceId"] == "qrels_access_receipt"
    firewall_fields = {
        column["field"] for column in tables["evaluation_firewall"]["columns"]
    }
    assert firewall_fields == set(datasets["firewall_rows"][0])
    assert "stage_a_permitted" not in firewall_fields
    assert "stage_a_executed" not in firewall_fields
    assert {"facet_retention", "evaluation_firewall"} <= table_refs
    assert "one-time qrels access" in _report_text(artifact).lower()


def test_pilot_only_language_and_stage_diagnosis_are_unambiguous(artifact):
    text = _report_text(artifact)
    assert "four-topic descriptive pilot" in text
    assert "no inferential claim" in text
    assert "candidate absence" in text
    assert "local reranking" in text
    assert "fusion block" in text
    assert "Stage A was not permitted and was not executed" in text


def test_raw_prefusion_and_final_sets_remain_separate(artifact):
    rows = artifact["snapshot"]["datasets"]["stage_set_rows"]
    assert [(row["stage"], row["relevant_documents"]) for row in rows] == [
        ("Raw-union headroom", 220),
        ("Pre-fusion promoted novel", 22),
        ("Final novel vs corrected C0", 0),
    ]
    assert all(len(row["document_set_sha256"]) == 64 for row in rows)
    assert len({row["document_set_sha256"] for row in rows}) == 3


def test_promotion_window_and_candidate_union_are_defined_exactly(artifact):
    text = _report_text(artifact)
    assert "BF best facet rank ≤20 and C0 best facet rank >20" in text
    assert "original O@100 plus facet streams at K" in text
    chart = next(
        chart
        for chart in artifact["manifest"]["charts"]
        if chart["id"] == "candidate_union_retention"
    )
    assert "O@100" in chart["subtitle"]
    assert {row["arm"] for row in artifact["snapshot"]["datasets"]["candidate_union_rows"]} == {
        "C0",
        "BF100",
    }


def test_report_excludes_protected_topics_and_raw_qrels_rows(artifact):
    datasets = artifact["snapshot"]["datasets"]
    topic_values = {
        str(row["topic_id"])
        for rows in datasets.values()
        for row in rows
        if isinstance(row, dict) and row.get("topic_id") not in (None, "All four")
    }
    assert topic_values <= PILOT_TOPICS
    assert topic_values.isdisjoint(PROTECTED_TOPICS)
    serialized = json.dumps(artifact)
    for forbidden_key in (
        '"qrel_grade"',
        '"headroom_docids"',
        '"raw_union_docids"',
        '"final_docids"',
        '"protected_topic_ids"',
        '"secret_map"',
    ):
        assert forbidden_key not in serialized
    assert "/home/" not in serialized


def test_representative_passages_are_bounded_and_keep_provenance(artifact):
    rows = artifact["snapshot"]["datasets"]["representative_rows"]
    assert {row["evidence_class"] for row in rows} == {
        "Promoted before fusion",
        "Demoted in final ranking",
        "Gained in final ranking",
        "Lost from final ranking",
    }
    assert all("provenance" in row for row in rows)
    assert all(len(row["passage"]) <= 700 for row in rows)
    assert all(
        row["source_artifact"] == "representative_provenance_v2.json"
        for row in rows
    )
    promoted = [row for row in rows if row["evidence_class"] == "Promoted before fusion"]
    assert promoted
    assert all(row["facet"] != "n/a" for row in promoted)
    assert all(row["bf_facet_rank"] <= 20 for row in promoted)
    assert all(
        row["c0_facet_rank"] is None or row["c0_facet_rank"] > 20
        for row in promoted
    )


def test_firewall_exposes_approval_scoped_consumption_registry(artifact):
    row = artifact["snapshot"]["datasets"]["firewall_rows"][0]
    assert row["approval_scoped_consumption_registry"] is True
    assert row["qrels_access_status"] == "qrels_access_consumed"


def test_native_chart_contracts_match_the_four_required_visuals(artifact):
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert set(charts) == {
        "facet_review_rates",
        "candidate_union_retention",
        "promotion_funnel",
        "topic_ndcg_delta",
    }
    assert charts["facet_review_rates"]["encodings"]["x"]["field"] == "label"
    assert charts["facet_review_rates"]["encodings"]["y"]["field"] == "rate_percent"
    assert charts["facet_review_rates"]["encodings"]["color"]["field"] == "arm"
    assert charts["candidate_union_retention"]["encodings"]["x"]["field"] == "depth"
    assert charts["candidate_union_retention"]["encodings"]["color"]["field"] == "arm"
    assert "color" not in charts["promotion_funnel"]["encodings"]
    assert "color" not in charts["topic_ndcg_delta"]["encodings"]
    assert charts["topic_ndcg_delta"]["encodings"]["y"]["field"] == "ndcg_delta"


def test_tables_are_full_width_and_keep_exact_audit_detail(artifact):
    tables = artifact["manifest"]["tables"]
    assert all(table["layout"] == "full" for table in tables)
    assert all(table["defaultSort"]["field"] in {column["field"] for column in table["columns"]} for table in tables)
    assert all(table["columns"] for table in tables)
    assert {table["id"] for table in tables} >= {
        "narratives",
        "query_streams",
        "system_primary",
        "system_diagnostics",
        "facet_review",
        "facet_retention",
        "evaluation_firewall",
        "representatives",
    }


def test_reader_used_sources_have_exact_transformations_and_material_inputs(artifact):
    sources = {source["id"]: source for source in artifact["sources"]}
    native_source_ids = {
        item["sourceId"]
        for collection in ("charts", "tables")
        for item in artifact["manifest"][collection]
    }
    for source_id in native_source_ids:
        query = sources[source_id]["query"]
        assert query["sql"].strip() != (
            f"SELECT * FROM read_json_auto('{sources[source_id]['path']}');"
        )
        assert query["metric_definitions"]
    assert sources["decision"]["query"]["tables_used"] == [
        str(SOURCE_FILES[source_id].relative_to(REPO_ROOT))
        for source_id in MATERIAL_INPUTS["decision"]
    ]
    assert sources["qrels_access_receipt"]["query"]["tables_used"] == [
        str(SOURCE_FILES[source_id].relative_to(REPO_ROOT))
        for source_id in MATERIAL_INPUTS["qrels_access_receipt"]
    ]
    summary = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "technical_summary"
    )
    assert summary["sourceId"] == "decision"


def test_mixed_method_and_limitation_claims_are_split_by_source(artifact):
    blocks = {block["id"]: block for block in artifact["manifest"]["blocks"]}
    assert "definitions" not in blocks
    assert "limitations" not in blocks
    expected_sources = {
        "scope_heading": "manifest",
        "scope_isolation_note": "ranking_freeze",
        "definition_population": "manifest",
        "definition_baseline": "decision",
        "definition_relevance_review": "qrels_access_receipt",
        "definition_metrics": "decision",
        "definition_isolation": "ranking_freeze",
        "scoring_note": "scoring_receipt",
        "ranking_freeze_note": "ranking_freeze",
        "limitation_pilot": "manifest",
        "limitation_judgments": "qrels_access_receipt",
        "limitation_review": "qrels_access_receipt",
        "limitation_window": "preflight",
        "limitation_fusion": "decision",
    }
    for block_id, source_id in expected_sources.items():
        assert blocks[block_id]["sourceId"] == source_id
    assert "original-query order" not in blocks["scope_heading"]["body"]
    assert "No retrieval" not in blocks["scoring_note"]["body"]


def test_verified_findings_are_represented_exactly(artifact):
    datasets = artifact["snapshot"]["datasets"]
    review = {(row["arm"], row["label"]): row for row in datasets["review_rate_rows"]}
    assert review[("BF50", "Direct answer")]["rate_percent"] == pytest.approx(75.92592592592592)
    assert review[("C0", "Direct answer")]["rate_percent"] == pytest.approx(59.25925925925925)
    assert review[("BF50", "Wrong domain")]["rate_percent"] == pytest.approx(1.8518518518518516)
    assert review[("C0", "Wrong domain")]["rate_percent"] == pytest.approx(7.4074074074074066)
    union = {(row["arm"], row["depth"]): row["relevant_documents"] for row in datasets["candidate_union_rows"]}
    assert union == {
        ("C0", "K20"): 292,
        ("BF100", "K20"): 295,
        ("C0", "K50"): 352,
        ("BF100", "K50"): 360,
        ("C0", "K100"): 462,
        ("BF100", "K100"): 462,
    }
    deltas = {row["topic_id"]: row["ndcg_delta"] for row in datasets["topic_ndcg_rows"]}
    assert deltas == pytest.approx(
        {"200": 0.009798697787694877, "225": -0.02166115264654017, "707": 0.022109152967867918, "897": 0.0}
    )


def test_artifact_output_is_deterministic(inputs):
    assert build_artifact(**inputs) == build_artifact(**copy.deepcopy(inputs))


def test_builder_rejects_mutated_authenticated_source(inputs):
    tampered = copy.deepcopy(inputs)
    tampered["artifacts"]["decision"]["evidence"]["headroom"] += 1
    with pytest.raises(ValueError, match="decision payload differs from loaded bytes"):
        build_artifact(**tampered)


def test_builder_rejects_control_diff_not_bound_by_ranking_freeze(inputs):
    tampered = copy.deepcopy(inputs)
    tampered_diff = tampered["artifacts"]["legacy_corrected_diff"]
    tampered_diff["qrels_opened"] = True
    tampered["artifact_bytes"]["legacy_corrected_diff"] = (
        json.dumps(tampered_diff, separators=(",", ":"), sort_keys=True).encode()
    )
    with pytest.raises(ValueError, match="legacy corrected diff authentication differs"):
        build_artifact(**tampered)


def test_html_publication_is_atomic_create_only(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"surface":"report"}\n')
    renderer = tmp_path / "renderer.mjs"
    renderer.write_text("// exercised through the injected runner\n")
    output = tmp_path / "report.html"
    commands = []

    def fake_runner(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"<html>verified</html>")

    render_html_create_only(
        artifact_path=artifact_path,
        output_path=output,
        renderer_path=renderer,
        runner=fake_runner,
    )
    assert output.read_bytes() == b"<html>verified</html>"
    assert commands[0][:4] == [
        "node",
        str(renderer),
        "--input",
        str(artifact_path),
    ]

    output.write_bytes(b"existing html")
    commands.clear()
    with pytest.raises(FileExistsError, match="create-only HTML report"):
        render_html_create_only(
            artifact_path=artifact_path,
            output_path=output,
            renderer_path=renderer,
            runner=fake_runner,
        )
    assert output.read_bytes() == b"existing html"
    assert commands == []


def test_cli_rejects_noncanonical_source_paths(tmp_path):
    copied_manifest = tmp_path / "manifest.json"
    copied_manifest.write_bytes(SOURCE_FILES["manifest"].read_bytes())
    args = [
        "--manifest",
        str(copied_manifest),
        "--preflight",
        str(OUTPUT_ROOT / "preflight_v2"),
        "--scoring",
        str(OUTPUT_ROOT / "full_scoring_v1"),
        "--freeze",
        str(OUTPUT_ROOT / "freeze_v1"),
        "--review",
        str(OUTPUT_ROOT / "review_v3"),
        "--evaluation",
        str(OUTPUT_ROOT / "evaluation_v1"),
        "--output",
        str(tmp_path / "artifact.json"),
    ]
    with pytest.raises(ValueError, match="noncanonical manifest path"):
        main(args)


def test_cli_is_create_only(tmp_path):
    output = tmp_path / "artifact.json"
    args = [
        "--manifest",
        str(SOURCE_FILES["manifest"]),
        "--preflight",
        str(OUTPUT_ROOT / "preflight_v2"),
        "--scoring",
        str(OUTPUT_ROOT / "full_scoring_v1"),
        "--freeze",
        str(OUTPUT_ROOT / "freeze_v1"),
        "--review",
        str(OUTPUT_ROOT / "review_v3"),
        "--evaluation",
        str(OUTPUT_ROOT / "evaluation_v1"),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert json.loads(output.read_text())["surface"] == "report"
    with pytest.raises(FileExistsError, match="create-only report source"):
        main(args)

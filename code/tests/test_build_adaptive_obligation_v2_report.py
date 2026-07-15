from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import trec_rag.build_adaptive_obligation_v2_report as report_module
from trec_rag.build_adaptive_obligation_v2_report import (
    build_report_artifact,
    build_report_payload,
    deliver_html_create_only,
    write_artifact_create_only,
)


def _verified_sources() -> dict[str, object]:
    return {
        "contract": {
            "status": "complete",
            "topic_ids": ["219", "72", "300", "84"],
            "parent_count": 24,
            "reservoir_count": 48,
            "unit_count": 8_247,
            "protected_topic_count": 0,
            "qrels_opened": False,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "paid_call_count": 0,
            "model_load_count": 0,
            "inference_count": 0,
            "external_cost_usd": 0.0,
        },
        "proposal_preflight": {
            "status": "complete",
            "topic_ids": ["219", "72", "300", "84"],
            "job_count": 48,
            "primary_call_count": 48,
            "retry_call_ceiling": 48,
            "worst_case_call_ceiling": 96,
            "tokenizer_load_count": 1,
            "model_load_count": 0,
            "inference_count": 0,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "qrels_opened": False,
            "paid_call_count": 0,
            "external_cost_usd": 0.0,
        },
        "baseline_rankings": {
            "status": "complete",
            "topic_ids": ["219", "72", "300", "84"],
            "document_count": 8_114,
            "rankings": {
                "NARRATIVE": {"rows": 8_114, "sha256": "7" * 64},
                "FIXED-O0": {"rows": 8_114, "sha256": "6" * 64},
            },
            "qrels_opened": False,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "paid_call_count": 0,
            "model_load_count": 0,
            "inference_count": 0,
            "external_cost_usd": 0.0,
        },
        "v1_discovery": {
            "status": "discovery_unavailable",
            "reason": "corrected_pass_schema_json_truncation",
            "qrels_opened": False,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "paid_call_count": 0,
            "external_cost_usd": 0.0,
        },
        "source_paths": {
            "contract": "outputs/v2/contract/receipt.json",
            "proposal_preflight": "outputs/v2/proposal_preflight/receipt.json",
            "baseline_rankings": "outputs/v1/rankings/receipt.json",
            "v1_discovery": "outputs/v1/discovery/receipt.json",
        },
        "source_hashes": {
            "contract": "a" * 64,
            "proposal_preflight": "b" * 64,
            "baseline_rankings": "c" * 64,
            "v1_discovery": "d" * 64,
        },
        "supporting_paths": {
            "retrieval_implementation": "code/trec_rag/adaptive_obligation_v2_retrieve.py",
            "adaptive_plan": "docs/superpowers/plans/2026-07-15-adaptive-obligation-search-v2.md",
            "adaptive_design": "docs/superpowers/specs/2026-07-15-adaptive-obligation-search-v2-design.md",
            "report_builder": "code/trec_rag/build_adaptive_obligation_v2_report.py",
        },
        "supporting_hashes": {
            "retrieval_implementation": "1" * 64,
            "adaptive_plan": "2" * 64,
            "adaptive_design": "3" * 64,
            "report_builder": "4" * 64,
        },
        "output_inventory": {
            "v2_root": "outputs/v2",
            "observed_children": ["contract", "proposal_preflight"],
            "later_artifacts_present": [],
            "approval_present": False,
        },
        "later_artifacts_present": [],
        "approval_present": False,
    }


def test_report_states_exact_current_status() -> None:
    payload = build_report_payload(_verified_sources())
    assert payload["status"] == "proposal_preflight_ready"
    assert payload["parents"] == 24
    assert payload["reservoirs"] == 48
    assert payload["units"] == 8_247
    assert payload["proposal_jobs"] == 48
    assert payload["proposal_retry_call_ceiling"] == 48
    assert payload["proposal_worst_case_calls"] == 96
    assert payload["retrieval_hits_per_accepted_o1"] == 1_000
    assert payload["maximum_retrieval_requests"] == 16
    assert payload["qwen_calls_completed"] == 0
    assert payload["validation_calls_completed"] == 0
    assert payload["retrieval_calls_completed"] == 0
    assert payload["minilm_calls_completed"] == 0
    assert payload["qrels_opened"] is False


def test_artifact_leads_with_truth_and_marks_missing_adaptive_result() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "partial"
    assert artifact["snapshot"]["accessIssues"]
    bodies = "\n".join(
        block.get("body", "") for block in artifact["manifest"]["blocks"]
    )
    assert "fixed baselines are sealed; adaptive v2 has not run" in bodies
    assert "No adaptive relevance result exists yet" in bodies
    assert "ADAPTIVE-V2 improved" not in bodies


def test_artifact_has_required_technical_section_order() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    ids = [block["id"] for block in artifact["manifest"]["blocks"]]
    expected = [
        "title",
        "technical_summary",
        "key_evidence",
        "scope_definitions",
        "methodology",
        "limitations",
        "recommended_next_step",
        "further_questions",
    ]
    assert all(section in ids for section in expected)
    assert [ids.index(section) for section in expected] == sorted(
        ids.index(section) for section in expected
    )


def test_artifact_is_deterministic_and_uses_only_binary_readiness_chart() -> None:
    payload = build_report_payload(_verified_sources())
    build_timestamp = "2026-07-15T17:42:31Z"
    assert build_report_artifact(
        payload, build_timestamp=build_timestamp
    ) == build_report_artifact(
        copy.deepcopy(payload), build_timestamp=build_timestamp
    )
    artifact = build_report_artifact(payload, build_timestamp=build_timestamp)
    assert len(artifact["manifest"]["charts"]) == 1
    chart = artifact["manifest"]["charts"][0]
    assert chart["id"] == "stage_readiness"
    assert chart["type"] == "horizontalBar"
    assert chart["encodings"]["x"]["field"] == "stage"
    assert chart["encodings"]["y"]["field"] == "canonical_artifact_ready"
    assert chart["settings"]["orientation"] == "horizontal"
    assert "not relevance quality or progress" in chart["subtitle"]
    values = {
        row["canonical_artifact_ready"]
        for row in artifact["snapshot"]["datasets"]["stage_readiness"]
    }
    assert values == {0, 1}
    assert "color" not in chart["encodings"]
    assert "false comparison" in artifact["manifest"]["description"]


def test_one_injected_utc_build_instant_is_used_everywhere() -> None:
    timestamp = "2026-07-15T17:42:31Z"
    artifact = build_report_artifact(
        build_report_payload(_verified_sources()), build_timestamp=timestamp
    )
    assert artifact["manifest"]["generatedAt"] == timestamp
    assert artifact["snapshot"]["generatedAt"] == timestamp


def test_source_provenance_is_repo_relative_and_exact() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    sources = {source["id"]: source for source in artifact["sources"]}
    for source_id, expected_hash in _verified_sources()["source_hashes"].items():
        source = sources[source_id]
        assert not source["path"].startswith("/")
        assert ".." not in Path(source["path"]).parts
        assert source["query"]["id"] == f"sha256:{expected_hash}"
        assert source["query"]["tables_used"] == [source["path"]]
    receipt_rows = artifact["snapshot"]["datasets"]["source_receipts"]
    assert {row["sha256"] for row in receipt_rows} == set(
        _verified_sources()["source_hashes"].values()
    )


def test_design_and_stage_claims_have_reproducible_single_sources() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    sources = {source["id"]: source for source in artifact["sources"]}
    assert {"retrieval_design", "filesystem_inventory"} <= set(sources)
    assert sources["retrieval_design"]["query"]["tables_used"] == list(
        _verified_sources()["supporting_paths"].values()
    )[:3]
    inventory_sql = sources["filesystem_inventory"]["query"]["sql"]
    for value in (
        "Fixed baselines",
        "Focused BM25 retrieval",
        "unavailable",
        "hits=1,000",
        "outputs/v2/contract/receipt.json",
    ):
        assert value in inventory_sql

    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert tables["stage_status"]["sourceId"] == "filesystem_inventory"
    assert charts["stage_readiness"]["sourceId"] == "filesystem_inventory"

    blocks = {block["id"]: block for block in artifact["manifest"]["blocks"]}
    expected_sources = {
        "technical_summary": "filesystem_inventory",
        "technical_summary_proposal": "proposal_preflight",
        "technical_summary_retrieval": "retrieval_design",
        "stage_status_interpretation": "filesystem_inventory",
        "scope_definitions": "retrieval_design",
        "methodology_contract": "contract",
        "methodology_proposal": "proposal_preflight",
        "methodology_retrieval": "retrieval_design",
        "limitations": "filesystem_inventory",
        "limitations_retriever": "retrieval_design",
        "limitations_v1": "v1_discovery",
        "recommended_next_step": "proposal_preflight",
    }
    for block_id, source_id in expected_sources.items():
        assert blocks[block_id]["sourceId"] == source_id


def test_stage_table_never_confuses_ceiling_with_completed_work() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    stages = artifact["snapshot"]["datasets"]["stage_status"]
    assert len(stages) == 8
    assert all(row["calls_completed"] == 0 for row in stages)
    proposal = next(row for row in stages if row["stage"] == "O1 proposal preflight")
    assert proposal["status"] == "ready"
    assert "48 primary" in proposal["evidence"]
    retrieval = next(row for row in stages if row["stage"] == "Focused BM25 retrieval")
    assert retrieval["status"] == "unavailable"
    assert "hits=1,000" in retrieval["evidence"]


def test_report_defines_queries_folds_and_raw_first() -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    blocks = {block["id"]: block.get("body", "") for block in artifact["manifest"]["blocks"]}
    definitions = blocks["scope_definitions"]
    for phrase in ("**O0**", "**O1**", "**fold**", "**Raw-first**", "**Query-local**"):
        assert phrase in definitions
    method = "\n".join(
        body for block_id, body in blocks.items() if block_id.startswith("methodology")
    )
    assert "ordered parent anchors + complete O0 + accepted O1" in method
    assert "unchanged narrative + complete O0 + accepted O1" in method
    assert "top-100 and top-1,000" in method
    assert "There is no recursive" in method


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["contract"].__setitem__("qrels_opened", True), "qrels"),
        (
            lambda value: value["proposal_preflight"].__setitem__(
                "inference_count", 1
            ),
            "inference",
        ),
        (
            lambda value: value.__setitem__(
                "later_artifacts_present", ["proposals"]
            ),
            "later adaptive",
        ),
        (lambda value: value.__setitem__("approval_present", True), "approval"),
        (
            lambda value: value["source_paths"].__setitem__(
                "contract", "/tmp/receipt.json"
            ),
            "repo-relative",
        ),
    ],
)
def test_payload_fails_closed_on_truth_boundary(mutation, message) -> None:
    sources = _verified_sources()
    mutation(sources)
    with pytest.raises(ValueError, match=message):
        build_report_payload(sources)


def test_artifact_and_html_publication_are_create_only(tmp_path, monkeypatch) -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    artifact_path = tmp_path / "artifact.json"
    write_artifact_create_only(artifact_path, artifact)
    assert json.loads(artifact_path.read_text()) == artifact
    with pytest.raises(FileExistsError, match="create-only report artifact"):
        write_artifact_create_only(artifact_path, artifact)

    report_path = tmp_path / "report.html"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        temporary_path = Path(command[-1])
        temporary_path.write_text("<html>verified</html>")
        return report_module.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "html": str(temporary_path.resolve()),
                    "stages": {
                        "validation": "passed",
                        "package": "passed",
                        "verification": "passed",
                    },
                    "counts": report_module._expected_portable_counts(artifact),
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(report_module, "_run_canonical_delivery", fake_run)
    receipt = deliver_html_create_only(
        artifact_path=artifact_path,
        output_path=report_path,
    )
    assert receipt["stages"]["verification"] == "passed"
    assert report_path.read_text() == "<html>verified</html>"
    assert len(calls) == 1
    assert calls[0][0][:4] == [
        "node",
        str(report_module.CANONICAL_RENDERER_PATH),
        "--input",
        str(artifact_path),
    ]
    with pytest.raises(FileExistsError, match="create-only HTML report"):
        deliver_html_create_only(
            artifact_path=artifact_path,
            output_path=report_path,
        )
    assert len(calls) == 1


def test_failed_delivery_receipt_does_not_publish_html(tmp_path, monkeypatch) -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    artifact_path = tmp_path / "artifact.json"
    write_artifact_create_only(artifact_path, artifact)
    report_path = tmp_path / "report.html"
    temporary_paths: list[Path] = []

    def fake_run(command, **kwargs):
        del kwargs
        temporary_path = Path(command[-1])
        temporary_paths.append(temporary_path)
        temporary_path.write_text("<html>must not publish</html>")
        return report_module.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": False,
                    "stages": {
                        "validation": "passed",
                        "package": "passed",
                        "verification": "failed",
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(report_module, "_run_canonical_delivery", fake_run)
    with pytest.raises(RuntimeError, match="receipt"):
        deliver_html_create_only(
            artifact_path=artifact_path,
            output_path=report_path,
        )
    assert not report_path.exists()
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


@pytest.mark.parametrize("receipt_kind", ["malformed", "failed", "custom"])
def test_untrusted_delivery_receipts_never_publish_html(
    receipt_kind, tmp_path, monkeypatch
) -> None:
    artifact = build_report_artifact(build_report_payload(_verified_sources()))
    artifact_path = tmp_path / "artifact.json"
    write_artifact_create_only(artifact_path, artifact)
    report_path = tmp_path / "report.html"
    temporary_paths: list[Path] = []

    def fake_run(command, **kwargs):
        del kwargs
        temporary_path = Path(command[-1])
        temporary_paths.append(temporary_path)
        temporary_path.write_text("<html>must not publish</html>")
        if receipt_kind == "malformed":
            stdout = "{not-json"
        else:
            receipt = {
                "ok": True,
                "html": str(temporary_path.resolve()),
                "stages": {
                    "validation": "passed",
                    "package": "passed",
                    "verification": "passed",
                },
                "counts": report_module._expected_portable_counts(artifact),
            }
            if receipt_kind == "failed":
                receipt["stages"]["verification"] = "failed"
            else:
                receipt["counts"] = {**receipt["counts"], "blocks": 999}
            stdout = json.dumps(receipt)
        return report_module.subprocess.CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(report_module, "_run_canonical_delivery", fake_run)
    with pytest.raises(RuntimeError, match="receipt"):
        deliver_html_create_only(
            artifact_path=artifact_path,
            output_path=report_path,
        )
    assert not report_path.exists()
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_public_cli_does_not_accept_renderer_override() -> None:
    with pytest.raises(SystemExit):
        report_module._parser().parse_args(
            [
                "--contract",
                "contract",
                "--proposal-preflight",
                "proposal",
                "--baseline-rankings",
                "rankings",
                "--v1-discovery",
                "discovery",
                "--output",
                "report.html",
                "--renderer",
                "custom.mjs",
            ]
        )

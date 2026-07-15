from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import trec_rag.build_adaptive_obligation_v2_r2_report as report_module
from trec_rag.build_adaptive_obligation_v2_r2_report import (
    build_r2_report_artifact,
    deliver_html_create_only,
    load_verified_sources,
    write_artifact_create_only,
)


R2_PREFLIGHT_SHA256 = "4" * 64
R2_LEDGER_DIR = (
    "/home/npatta01/.codex/worktrees/41f9/trec_rag_2026/outputs/"
    "rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/"
    "proposal_ledger_r2"
)


def _verified_sources() -> dict[str, object]:
    return {
        "contract": {
            "schema_version": "adaptive-obligation-v2-contract-v1",
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
        "baseline_rankings": {
            "schema_version": "adaptive-evidence-baseline-rankings-v1",
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
        "r1_incident": {
            "schema_version": "adaptive-obligation-v2-proposal-incident-r1",
            "status": "aborted",
            "reason_code": "scope_rationale_length_exceeded",
            "attempted_job_count": 1,
            "terminal_schema_error_count": 1,
            "uncalled_job_count": 47,
            "observed_rationale_characters": 245,
            "accepted_rationale_maximum": 240,
            "output_token_count": 186,
            "preflight": {
                "path": "outputs/v2/proposal_preflight/receipt.json",
                "bytes": 16_452,
                "sha256": "1" * 64,
            },
            "approval": {
                "path": "outputs/v2/proposal_approval.json",
                "bytes": 365,
                "sha256": "2" * 64,
            },
            "ledger": {
                "path": "outputs/v2/proposal_ledger",
                "event_count": 2,
                "anchor": {
                    "path": "outputs/v2/proposal_ledger/anchor.json",
                    "bytes": 571_327,
                    "sha256": "3" * 64,
                },
                "events": {
                    "path": "outputs/v2/proposal_ledger/events.jsonl",
                    "bytes": 1_217,
                    "sha256": "5" * 64,
                },
                "head": {
                    "path": "outputs/v2/proposal_ledger/head.json",
                    "bytes": 159,
                    "sha256": "8" * 64,
                },
            },
            "raw_completion": {
                "path": "raw/job.1.completion",
                "bytes": 546,
                "sha256": "9" * 64,
            },
            "qrels_opened": False,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "paid_call_count": 0,
        },
        "r2_preflight": {
            "schema_version": "adaptive-obligation-v2-proposal-preflight-r2",
            "status": "complete",
            "topic_ids": ["219", "72", "300", "84"],
            "job_count": 48,
            "primary_call_count": 48,
            "retry_call_ceiling": 48,
            "worst_case_call_ceiling": 96,
            "primary_max_new_tokens": 256,
            "retry_max_new_tokens": 512,
            "prompt_revision": "tail-contract-r2",
            "prompt_sha256": "a" * 64,
            "schema_sha256": "b" * 64,
            "tokenizer_identity_sha256": "c" * 64,
            "prompt_token_counts": {
                "count": 48,
                "minimum": 33_610,
                "maximum": 64_264,
                "total": 2_300_662,
                "by_job": [],
            },
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "model_snapshot": {"manifest_sha256": "d" * 64},
            "code_sha256": {
                "adaptive_obligation_v2_contract.py": "e" * 64,
                "adaptive_obligation_v2_propose.py": "f" * 64,
                "adaptive_obligation_v2_propose_r2.py": "0" * 64,
            },
            "artifacts": {
                "jobs.jsonl": {"rows": 48, "sha256": "1" * 64},
                "prompt.json": {"rows": 1, "sha256": "2" * 64},
                "schema.json": {"rows": 1, "sha256": "3" * 64},
            },
            "ledger_dir": R2_LEDGER_DIR,
            "proposal_dir": R2_LEDGER_DIR.replace("proposal_ledger", "proposals"),
            "tokenizer_load_count": 1,
            "model_load_count": 0,
            "inference_count": 0,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "hosted_inference_call_count": 0,
            "qrels_opened": False,
            "paid_call_count": 0,
            "external_cost_usd": 0.0,
            "generation_allowed": False,
            "model_construction_allowed": False,
        },
        "source_paths": {
            "contract": "outputs/v2/contract/receipt.json",
            "baseline_rankings": "outputs/v1/rankings/receipt.json",
            "r1_incident": "outputs/v2/proposal_run_r1_incident/receipt.json",
            "r2_preflight": "outputs/v2/proposal_preflight_r2/receipt.json",
        },
        "source_hashes": {
            "contract": "a" * 64,
            "baseline_rankings": "b" * 64,
            "r1_incident": "c" * 64,
            "r2_preflight": R2_PREFLIGHT_SHA256,
        },
        "supporting_paths": {
            "recovery_plan": (
                "docs/superpowers/plans/"
                "2026-07-15-adaptive-obligation-proposal-r2-recovery.md"
            ),
            "recovery_design": (
                "docs/superpowers/specs/"
                "2026-07-15-adaptive-obligation-proposal-r2-recovery-design.md"
            ),
            "incident_implementation": (
                "code/trec_rag/adaptive_obligation_v2_proposal_incident.py"
            ),
            "r2_implementation": "code/trec_rag/adaptive_obligation_v2_propose_r2.py",
            "r2_model_implementation": (
                "code/trec_rag/adaptive_obligation_v2_local_model_r2.py"
            ),
            "report_builder": (
                "code/trec_rag/build_adaptive_obligation_v2_r2_report.py"
            ),
        },
        "supporting_hashes": {
            "recovery_plan": "4" * 64,
            "recovery_design": "5" * 64,
            "incident_implementation": "6" * 64,
            "r2_implementation": "7" * 64,
            "r2_model_implementation": "8" * 64,
            "report_builder": "9" * 64,
        },
        "r2_approval_present": False,
        "r2_ledger_present": False,
        "r2_proposals_present": False,
    }


def test_report_distinguishes_r1_failure_from_r2_readiness() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    text = json.dumps(artifact)
    assert "R1 stopped after exactly one schema-invalid call" in text
    assert "R2 proposal inference has not run" in text
    assert "R2 validation integration is unavailable and deferred" in text
    assert "No adaptive relevance result exists" in text
    assert '"calls_completed": 1' in text
    assert '"r2_calls_completed": 0' in text


def test_report_records_exact_r2_preflight_hash_counts_and_destinations() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    rows = artifact["snapshot"]["datasets"]["proposal_runs"]
    assert rows[1]["preflight_sha256"] == R2_PREFLIGHT_SHA256
    assert rows[1]["primary_call_count"] == 48
    assert rows[1]["retry_call_ceiling"] == 48
    assert rows[1]["worst_case_call_ceiling"] == 96
    assert rows[1]["prompt_tokens_minimum"] == 33_610
    assert rows[1]["prompt_tokens_maximum"] == 64_264
    assert rows[1]["prompt_tokens_total"] == 2_300_662
    assert rows[1]["ledger_dir"] == R2_LEDGER_DIR
    assert rows[1]["model_revision"] == (
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )


def test_rendered_run_table_exposes_exact_r2_tokens_without_unsourced_r1_ceilings() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    rows = artifact["snapshot"]["datasets"]["proposal_runs"]
    for field in (
        "primary_call_count",
        "retry_call_ceiling",
        "worst_case_call_ceiling",
        "model",
        "model_revision",
    ):
        assert rows[0][field] is None

    table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "proposal_runs"
    )
    visible_fields = {column["field"] for column in table["columns"]}
    assert {
        "prompt_tokens_minimum",
        "prompt_tokens_maximum",
        "prompt_tokens_total",
    } <= visible_fields
    bodies = "\n".join(
        block.get("body", "") for block in artifact["manifest"]["blocks"]
    )
    assert "33,610–64,264" in bodies
    assert "2,300,662 total" in bodies


def test_runtime_is_only_an_operator_estimate() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    text = json.dumps(artifact, ensure_ascii=False)
    assert "Operator estimate: 2–3 hours" in text
    assert "approximately three-minute, 54,402-token R1 call" in text
    assert "not a receipt-verified runtime metric" in text
    assert artifact["snapshot"]["datasets"]["runtime_estimate"] == [
        {
            "basis": "approximately three-minute, 54,402-token R1 call",
            "estimate_low_hours": 2,
            "estimate_high_hours": 3,
            "receipt_verified": False,
        }
    ]


@pytest.mark.parametrize(
    "field", ["r2_approval_present", "r2_ledger_present", "r2_proposals_present"]
)
def test_report_refuses_any_r2_execution_artifact(field: str) -> None:
    sources = _verified_sources()
    sources[field] = True
    with pytest.raises(ValueError, match="R2 inference must remain unopened"):
        build_r2_report_artifact(sources)


def test_report_uses_native_binary_chart_with_reviewed_rows_and_sources() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "partial"
    assert len(artifact["manifest"]["charts"]) == 1
    chart = artifact["manifest"]["charts"][0]
    assert chart["type"] == "horizontalBar"
    assert chart["dataset"] == "stage_readiness"
    assert chart["sourceId"] == "verified_status_inventory"
    assert "color" not in chart["encodings"]
    readiness = artifact["snapshot"]["datasets"]["stage_readiness"]
    assert {row["canonical_artifact_ready"] for row in readiness} == {0, 1}
    assert all({"stage", "status", "evidence", "calls_completed"} <= set(row) for row in readiness)

    source_ids = {source["id"] for source in artifact["sources"]}
    assert {
        "contract",
        "baseline_rankings",
        "r1_incident",
        "r2_preflight",
        "verified_status_inventory",
        "recovery_contract",
    } <= source_ids
    for source in artifact["sources"]:
        assert source["query"]["tables_used"]


def test_report_has_required_technical_sections_in_order() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    ids = [block["id"] for block in artifact["manifest"]["blocks"]]
    expected = [
        "title",
        "technical_summary",
        "key_findings",
        "scope_definitions",
        "methodology",
        "limitations",
        "recommended_next_step",
        "further_questions",
    ]
    assert [ids.index(section) for section in expected] == sorted(
        ids.index(section) for section in expected
    )
    next_step = next(
        block for block in artifact["manifest"]["blocks"]
        if block["id"] == "recommended_next_step"
    )["body"]
    assert "Approve only this exact R2 preflight" in next_step
    assert "Do not approve validation, BM25, MiniLM, or qrels" in next_step


def test_artifact_is_deterministic_with_one_injected_timestamp() -> None:
    timestamp = "2026-07-15T17:42:31Z"
    first = build_r2_report_artifact(_verified_sources(), build_timestamp=timestamp)
    second = build_r2_report_artifact(
        copy.deepcopy(_verified_sources()), build_timestamp=timestamp
    )
    assert first == second
    assert first["manifest"]["generatedAt"] == timestamp
    assert first["snapshot"]["generatedAt"] == timestamp


def test_parser_exposes_report_inputs_only() -> None:
    options = {
        option
        for action in report_module._parser()._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--contract",
        "--baseline-rankings",
        "--r1-incident",
        "--r2-preflight",
        "--artifact",
        "--output",
    }
    assert "--approval" not in options


def test_artifact_and_html_delivery_are_create_only(tmp_path: Path, monkeypatch) -> None:
    artifact = build_r2_report_artifact(
        _verified_sources(), build_timestamp="2026-07-15T17:42:31Z"
    )
    artifact_path = tmp_path / "artifact.json"
    write_artifact_create_only(artifact_path, artifact)
    original = artifact_path.read_bytes()
    with pytest.raises(FileExistsError, match="create-only report artifact"):
        write_artifact_create_only(artifact_path, artifact)
    assert artifact_path.read_bytes() == original

    output_path = tmp_path / "report.html"
    output_path.write_text("existing", encoding="utf-8")
    touched: list[object] = []
    monkeypatch.setattr(
        report_module,
        "_run_canonical_delivery",
        lambda *_args, **_kwargs: touched.append(True),
    )
    with pytest.raises(FileExistsError, match="create-only HTML report"):
        deliver_html_create_only(
            artifact_path=artifact_path,
            output_path=output_path,
        )
    assert touched == []
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_loader_preserves_incident_bound_relative_paths_for_byte_replay(
    monkeypatch,
) -> None:
    incident = copy.deepcopy(_verified_sources()["r1_incident"])
    root = Path(
        "outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2"
    )
    incident["preflight"]["path"] = str(root / "proposal_preflight/receipt.json")
    incident["approval"]["path"] = str(root / "proposal_approval.json")
    incident["ledger"]["path"] = str(root / "proposal_ledger")
    captured: dict[str, Path] = {}

    monkeypatch.setattr(report_module, "_read_json_object", lambda *_args: incident)
    monkeypatch.setattr(report_module, "verify_v2_contract", lambda *_args: _verified_sources()["contract"])
    monkeypatch.setattr(
        report_module,
        "verify_baseline_rankings",
        lambda *_args: _verified_sources()["baseline_rankings"],
    )
    monkeypatch.setattr(
        report_module,
        "verify_r2_proposal_preflight",
        lambda *_args: _verified_sources()["r2_preflight"],
    )

    def verify_incident(**kwargs):
        captured.update(kwargs)
        return incident

    monkeypatch.setattr(report_module, "verify_r1_incident", verify_incident)
    monkeypatch.setattr(report_module, "_repo_relative", lambda path: str(path))
    monkeypatch.setattr(report_module, "_sha256_file", lambda _path: "a" * 64)

    load_verified_sources(
        contract_dir=Path("outputs/v2/contract"),
        baseline_rankings_dir=Path("outputs/v1/rankings"),
        r1_incident_dir=Path("outputs/v2/proposal_run_r1_incident"),
        r2_preflight_dir=Path("outputs/v2/proposal_preflight_r2"),
    )

    assert captured["preflight_dir"] == root / "proposal_preflight"
    assert captured["approval_path"] == root / "proposal_approval.json"
    assert captured["ledger_dir"] == root / "proposal_ledger"

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight


def test_offline_preflight_report_is_non_inference_and_binds_artifacts():
    report = preflight.build_offline_preflight_report()

    assert report["schema_version"] == "semantic_anchor_offline_preflight_report_v1"
    assert report["status"] == "offline_preflight_pass"
    assert report["inference_authorized"] is False
    assert report["external_cost_authorized"] is False
    assert report["cost_counters"] == preflight.ZERO_COST_COUNTERS
    assert report["source_audit"]["checker"] == "det_sparse_v4_static_direct_source_audit_v1"
    assert report["source_audit"]["status"] == "pass"
    assert report["source_audit"]["audited_source_count"] == 3
    assert report["source_audit"]["unexpected_import_roots"] == []
    assert report["source_audit"]["unexpected_trec_rag_modules"] == []
    assert report["source_audit"]["denied_import_issues"] == []
    assert report["source_audit"]["denied_path_fragment_issues"] == {}
    assert report["source_audit"]["observed_trec_rag_modules"] == [
        "trec_rag.det_sparse_v4_contract",
        "trec_rag.query_schema_compat",
    ]
    assert report["schema_compatibility"] == {
        "checker": "vllm_0_24_xgrammar_unsupported_feature_lint",
        "case_count": 24,
        "status": "pass",
        "unsupported_feature_issues": [],
    }
    request_identity = report["request_identity"]
    assert request_identity["checker"] == "semantic_anchor_request_identity_v1"
    assert request_identity["status"] == "pass"
    assert request_identity["case_count"] == 24
    assert request_identity["first_case_id"] == "synthetic-case-001"
    assert len(request_identity["case_order_sha256"]) == 64
    assert list(request_identity["request_sha256"]) == [
        f"synthetic-case-{index:03d}" for index in range(1, 25)
    ]
    assert list(request_identity["request_body_size_bytes"]) == [
        f"synthetic-case-{index:03d}" for index in range(1, 25)
    ]
    assert all(len(value) == 64 for value in request_identity["request_sha256"].values())
    assert all(value > 0 for value in request_identity["request_body_size_bytes"].values())
    assert report["denied_topic_ids"] == list(contract.DENIED_TOPIC_IDS)
    assert report["runner_visible_artifacts"] == list(contract.runner_visible_artifacts())
    assert report["scorer_only_artifacts"] == list(contract.scorer_only_artifacts())
    assert report["artifact_count"] == len(report["artifact_sha256"])
    assert report["artifact_sha256"] == contract.validate_artifact_bundle()

    preflight.validate_offline_preflight_report(report)


def test_offline_preflight_report_rejects_inference_or_cost_authorization():
    report = preflight.build_offline_preflight_report()

    report["inference_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize inference"):
        preflight.validate_offline_preflight_report(report)

    report = preflight.build_offline_preflight_report()
    report["cost_counters"] = dict(preflight.ZERO_COST_COUNTERS, model_calls=1)
    with pytest.raises(ValueError, match="cost counters"):
        preflight.validate_offline_preflight_report(report)

    report = preflight.build_offline_preflight_report()
    schema_compatibility = dict(report["schema_compatibility"])
    schema_compatibility["unsupported_feature_issues"] = [
        {
            "case_id": "synthetic-case-001",
            "json_path": "$.properties.bad",
            "keyword": "uniqueItems",
            "reason": "unsupported",
        }
    ]
    schema_compatibility["status"] = "fail"
    report["schema_compatibility"] = schema_compatibility
    with pytest.raises(ValueError, match="schema compatibility"):
        preflight.validate_offline_preflight_report(report)

    report = preflight.build_offline_preflight_report()
    request_identity = dict(report["request_identity"])
    request_identity["request_sha256"] = dict(request_identity["request_sha256"])
    request_identity["request_sha256"]["synthetic-case-001"] = "not-a-sha"
    report["request_identity"] = request_identity
    with pytest.raises(ValueError, match="request hash"):
        preflight.validate_offline_preflight_report(report)


def test_offline_preflight_fails_on_denied_imports(tmp_path: Path):
    bad_source = tmp_path / "bad_v4_runner.py"
    bad_source.write_text(
        "from trec_rag.topics import load_topics\n"
        "from trec_rag.remote_pyserini import SearchClient\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="denied import audit failed"):
        preflight.build_offline_preflight_report(source_paths=[bad_source])


def test_offline_preflight_fails_on_denied_path_fragments(tmp_path: Path):
    bad_source = tmp_path / "bad_paths.py"
    bad_source.write_text(
        "ARTIFACT = 'cache/retrieval/topic-run.jsonl'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="denied path-fragment audit failed"):
        preflight.build_offline_preflight_report(source_paths=[bad_source])


def test_offline_preflight_fails_on_unallowlisted_source_import(tmp_path: Path):
    bad_source = tmp_path / "bad_import_root.py"
    bad_source.write_text("import subprocess\n", encoding="utf-8")

    summary = preflight.build_source_audit_summary([bad_source])

    assert summary["status"] == "fail"
    assert summary["unexpected_import_roots"] == ["subprocess"]
    with pytest.raises(ValueError, match="source audit failed"):
        preflight.build_offline_preflight_report(source_paths=[bad_source])


def test_offline_preflight_fails_on_missing_required_artifact(tmp_path: Path):
    copied = tmp_path / "artifacts"
    shutil.copytree(contract.ARTIFACT_DIR, copied)
    (copied / "semantic_anchor_case_registry_v1.json").unlink()

    with pytest.raises(ValueError, match="artifact missing"):
        preflight.build_offline_preflight_report(copied)


def test_offline_preflight_schema_compatibility_reports_unsupported_features(tmp_path: Path):
    copied = tmp_path / "artifacts"
    shutil.copytree(contract.ARTIFACT_DIR, copied)
    request_path = copied / "semantic_anchor_request_fixtures_v1.jsonl"
    records = [
        json.loads(line)
        for line in request_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["request"]["response_format"]["json_schema"]["schema"]["properties"][
        "bad_array"
    ] = {
        "type": "array",
        "items": {"type": "string"},
        "uniqueItems": True,
    }
    request_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = preflight.build_schema_compatibility_summary(copied)

    assert summary["status"] == "fail"
    assert summary["unsupported_feature_issues"][0]["case_id"] == "synthetic-case-001"
    assert summary["unsupported_feature_issues"][0]["keyword"] == "uniqueItems"


def test_offline_preflight_request_identity_binds_fixed_case_order_and_request_hashes(tmp_path: Path):
    copied = tmp_path / "artifacts"
    shutil.copytree(contract.ARTIFACT_DIR, copied)

    summary = preflight.build_request_identity_summary(copied)

    assert summary["status"] == "pass"
    assert summary["first_case_id"] == "synthetic-case-001"
    assert len(summary["request_sha256"]) == 24
    assert summary["request_sha256"]["synthetic-case-001"] != summary["request_sha256"][
        "synthetic-case-002"
    ]

    case_order_path = copied / "semantic_anchor_case_order_v1.json"
    case_order = json.loads(case_order_path.read_text(encoding="utf-8"))
    case_order["case_order"][0], case_order["case_order"][1] = (
        case_order["case_order"][1],
        case_order["case_order"][0],
    )
    case_order_path.write_text(json.dumps(case_order), encoding="utf-8")
    with pytest.raises(ValueError, match="case order drifted"):
        preflight.build_request_identity_summary(copied)


def test_offline_preflight_request_identity_rejects_request_order_drift(tmp_path: Path):
    copied = tmp_path / "artifacts"
    shutil.copytree(contract.ARTIFACT_DIR, copied)
    request_path = copied / "semantic_anchor_request_fixtures_v1.jsonl"
    records = [
        json.loads(line)
        for line in request_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0], records[1] = records[1], records[0]
    request_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="request fixture order"):
        preflight.build_request_identity_summary(copied)


def test_offline_preflight_cli_writes_create_only_json(tmp_path: Path):
    output = tmp_path / "preflight.json"

    assert preflight.main(["--output", output.as_posix(), "--pretty"]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    preflight.validate_offline_preflight_report(report)

    with pytest.raises(FileExistsError):
        preflight.main(["--output", output.as_posix()])

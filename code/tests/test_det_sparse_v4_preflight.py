from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight


def _offset_health():
    return {
        "schema_version": "lucene_whole_unit_offsets_health_v1",
        "status": "ok",
        "legacy_analyzer_port": 18081,
        "offset_analyzer_port": 18082,
    }


def _offset_fingerprint():
    return {
        "legacy_term_chain_fingerprint_sha256": "a" * 64,
        "offset_contract_version": "lucene_whole_unit_offsets_v1",
        "offset_server_class_sha256": "b" * 64,
        "offset_lucene_jar_sha256": "c" * 64,
        "offset_runtime_image_digest": "sha256:" + "d" * 64,
    }


def _offset_response(text: str, *, occurrences: list[dict[str, object]] | None = None):
    if occurrences is None:
        occurrences = [
            {
                "ordinal": 0,
                "term": "anchor",
                "start_codepoint": 0,
                "end_codepoint": 6,
                "position_increment": 1,
            }
        ]
    return {
        "schema_version": "lucene_whole_unit_offsets_response_v1",
        "text_sha256": contract.text_sha256(text),
        "offset_unit": "unicode_code_points",
        "fingerprint": _offset_fingerprint(),
        "occurrences": occurrences,
    }


def _offset_parity_fixtures():
    fixtures = []
    for surface_class in preflight.OFFSET_PARITY_SURFACE_CLASSES:
        for unit_position in preflight.OFFSET_PARITY_UNIT_POSITIONS:
            for punctuation_context in preflight.OFFSET_PARITY_PUNCTUATION_CONTEXTS:
                fixture_id = f"{surface_class}-{unit_position}-{punctuation_context}"
                text = preflight._offset_parity_expected_text(
                    surface_class=surface_class,
                    unit_position=unit_position,
                    punctuation_context=punctuation_context,
                )
                if surface_class == "analyzer_zero_stopword":
                    response = _offset_response(text, occurrences=[])
                else:
                    response = _offset_response(text)
                _request, _body, request_sha256 = contract.build_offset_request(text)
                fixtures.append(
                    {
                        "fixture_id": fixture_id,
                        "surface_class": surface_class,
                        "unit_position": unit_position,
                        "punctuation_context": punctuation_context,
                        "text": text,
                        "request_sha256": request_sha256,
                        "response_sha256": contract.sha256_bytes(
                            contract.canonical_json_bytes(response)
                        ),
                        "response": response,
                    }
                )
    return {
        "schema_version": "semantic_anchor_offset_parity_fixture_v1",
        "health": _offset_health(),
        "fixtures": fixtures,
    }


def _write_offset_parity_review(tmp_path: Path) -> Path:
    index = 0
    while (tmp_path / f"offset-parity-review-{index}.json").exists():
        index += 1
    fixtures_path = tmp_path / f"offset-parity-fixtures-{index}.json"
    review_path = tmp_path / f"offset-parity-review-{index}.json"
    fixtures_path.write_bytes(
        contract.canonical_json_bytes(_offset_parity_fixtures()) + b"\n"
    )
    review = preflight.build_offset_parity_review(fixtures_path)
    review_path.write_bytes(preflight.canonical_report_bytes(review))
    return review_path


def _schema_compiler_attestation(request_identity):
    return {
        "schema_version": "semantic_anchor_schema_compiler_attestation_v1",
        "compiler": {
            "vllm_version": "0.24.0",
            "xgrammar_version": "0.2.3",
            "structured_outputs_backend": "xgrammar",
        },
        "case_order_sha256": request_identity["case_order_sha256"],
        "cases": [
            {
                "case_id": case_id,
                "request_sha256": request_sha256,
                "schema_sha256": f"{index:064x}"[-64:],
                "xgrammar_strict": "pass",
            }
            for index, (case_id, request_sha256) in enumerate(
                request_identity["request_sha256"].items(),
                start=1,
            )
        ],
    }


def _model_runtime_attestation(model_inventory_sha256: str):
    return {
        "schema_version": "semantic_anchor_live_model_runtime_attestation_v1",
        "attestation_id": "runtime-attestation-001",
        "served_model": "gpt-oss-local",
        "repository": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "vllm_version": "0.24.0",
        "xgrammar_version": "0.2.3",
        "structured_outputs_backend": "xgrammar",
        "loopback_only": True,
        "egress_denied": True,
        "read_only_model_mount": True,
        "model_inventory_sha256": model_inventory_sha256,
    }


def _live_attestation_bundle(model_inventory_sha256: str):
    request_identity = preflight.build_request_identity_summary(contract.ARTIFACT_DIR)
    return {
        "schema_version": "semantic_anchor_live_attestation_bundle_v1",
        "offset_health": _offset_health(),
        "schema_compiler": _schema_compiler_attestation(request_identity),
        "model_runtime": _model_runtime_attestation(model_inventory_sha256),
    }


def test_offline_preflight_report_is_non_inference_and_binds_artifacts():
    report = preflight.build_offline_preflight_report()

    assert report["schema_version"] == "semantic_anchor_offline_preflight_report_v1"
    assert report["status"] == "offline_preflight_pass"
    assert report["inference_authorized"] is False
    assert report["external_cost_authorized"] is False
    assert report["cost_counters"] == preflight.ZERO_COST_COUNTERS
    assert report["source_audit"]["checker"] == "det_sparse_v4_static_direct_source_audit_v1"
    assert report["source_audit"]["status"] == "pass"
    assert report["source_audit"]["audited_source_count"] == 5
    assert report["source_audit"]["unexpected_import_roots"] == []
    assert report["source_audit"]["unexpected_trec_rag_modules"] == []
    assert report["source_audit"]["denied_import_issues"] == []
    assert report["source_audit"]["denied_path_fragment_issues"] == {}
    assert report["source_audit"]["observed_trec_rag_modules"] == [
        "trec_rag.det_sparse_v4_contract",
        "trec_rag.det_sparse_v4_preflight",
        "trec_rag.det_sparse_v4_scorer",
        "trec_rag.query_schema_compat",
    ]
    runtime_file_access = report["runtime_file_access"]
    assert runtime_file_access["checker"] == "det_sparse_v4_runtime_file_access_audit_v1"
    assert runtime_file_access["status"] == "pass"
    assert runtime_file_access["observed_open_count"] >= report["artifact_count"]
    assert runtime_file_access["observed_read_path_count"] >= report["artifact_count"]
    assert runtime_file_access["observed_write_path_count"] == 0
    assert runtime_file_access["denied_path_fragment_issues"] == {}
    assert runtime_file_access["denied_write_paths"] == []
    observed_paths = {record["path"] for record in runtime_file_access["observed_paths"]}
    assert str(contract.ARTIFACT_MANIFEST.resolve()) in observed_paths
    assert str(Path(preflight.__file__).resolve()) in observed_paths
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

    report = preflight.build_offline_preflight_report()
    runtime_file_access = dict(report["runtime_file_access"])
    runtime_file_access["observed_paths"] = [
        {
            "path": str((Path.cwd() / "cache" / "retrieval" / "bad.jsonl").resolve()),
            "accesses": ["read"],
        }
    ]
    runtime_file_access["denied_path_fragment_issues"] = {
        runtime_file_access["observed_paths"][0]["path"]: ["cache/retrieval"]
    }
    runtime_file_access["status"] = "fail"
    report["runtime_file_access"] = runtime_file_access
    with pytest.raises(ValueError, match="runtime file access"):
        preflight.validate_offline_preflight_report(report)


def test_runtime_file_access_summary_catches_denied_reads_and_writes(tmp_path: Path):
    denied_dir = tmp_path / "qrels"
    denied_dir.mkdir()
    denied_file = denied_dir / "synthetic.txt"
    denied_file.write_text("do not read through preflight\n", encoding="utf-8")
    write_file = tmp_path / "runtime-write.txt"

    def builder():
        denied_file.read_text(encoding="utf-8")
        write_file.write_text("write should fail the audit\n", encoding="utf-8")
        return {"status": "placeholder"}

    report = preflight.build_runtime_file_access_summary(builder)
    summary = report["runtime_file_access"]

    assert summary["status"] == "fail"
    assert str(denied_file.resolve()) in summary["denied_path_fragment_issues"]
    assert summary["denied_path_fragment_issues"][str(denied_file.resolve())] == ["qrels"]
    assert str(write_file.resolve()) in summary["denied_write_paths"]


def test_runtime_file_access_summary_catches_model_snapshot_reads(tmp_path: Path):
    model_file = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--gpt-oss-20b"
        / "snapshots"
        / "6cee5e81ee83917806bbde320786a8fb61efebee"
        / "model-00001-of-00003.safetensors"
    )
    model_file.parent.mkdir(parents=True)
    model_file.write_text("not a real model\n", encoding="utf-8")

    report = preflight.build_runtime_file_access_summary(
        lambda: {"status": model_file.read_text(encoding="utf-8")}
    )
    summary = report["runtime_file_access"]

    assert summary["status"] == "fail"
    assert summary["denied_path_fragment_issues"][str(model_file.resolve())] == [
        ".cache/huggingface",
        "huggingface/hub",
        "models--openai--gpt-oss-20b",
        ".safetensors",
    ]


def test_live_attestation_review_validates_bundle_without_authorizing_dispatch(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    model_inventory_sha256 = artifact_hashes[
        "semantic_anchor_model_inventory_attestation_v1.json"
    ]
    bundle_path = tmp_path / "live-attestation-bundle.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(_live_attestation_bundle(model_inventory_sha256))
        + b"\n"
    )
    offset_review_path = _write_offset_parity_review(tmp_path)

    report = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=offset_review_path,
    )

    assert report["schema_version"] == "semantic_anchor_live_attestation_review_v1"
    assert report["status"] == "live_attestation_review_pass"
    assert report["bundle_path"] == str(bundle_path.resolve())
    assert report["bundle_sha256"] == contract.sha256_file(bundle_path)
    assert report["bundle_canonical"] is True
    assert report["offset_parity_review_path"] == str(offset_review_path.resolve())
    assert report["offset_parity_review_sha256"] == contract.sha256_file(
        offset_review_path
    )
    assert report["offset_parity_review_status"] == "offset_parity_review_pass"
    assert report["offset_fingerprint_sha256"] == contract.sha256_bytes(
        contract.canonical_json_bytes(_offset_fingerprint())
    )
    assert report["model_inventory_sha256"] == model_inventory_sha256
    assert report["dispatch_authorized"] is False
    assert report["inference_authorized"] is False
    assert report["external_cost_authorized"] is False
    assert report["next_gate"] == "advisor_go_before_model_dispatch"
    assert report["pre_dispatch_attestation"] == {
        "schema_version": "semantic_anchor_pre_dispatch_attestation_v1",
        "attestation_id": "runtime-attestation-001",
        "served_model": "gpt-oss-local",
        "egress_denied": True,
        "read_only_model_mount": True,
        "model_inventory_sha256": model_inventory_sha256,
    }
    runtime_file_access = report["runtime_file_access"]
    assert runtime_file_access["status"] == "pass"
    assert runtime_file_access["observed_write_path_count"] == 0
    observed_paths = {record["path"] for record in runtime_file_access["observed_paths"]}
    assert str(bundle_path.resolve()) in observed_paths
    assert str(offset_review_path.resolve()) in observed_paths
    preflight.validate_live_attestation_review(report)


def test_offset_parity_review_validates_48_canonical_fixture_rows(tmp_path: Path):
    fixtures_path = tmp_path / "offset-parity-fixtures.json"
    fixtures = _offset_parity_fixtures()
    fixtures_path.write_bytes(contract.canonical_json_bytes(fixtures) + b"\n")

    review = preflight.build_offset_parity_review(fixtures_path)

    assert review["schema_version"] == "semantic_anchor_offset_parity_review_v1"
    assert review["status"] == "offset_parity_review_pass"
    assert review["fixture_path"] == str(fixtures_path.resolve())
    assert review["fixture_sha256"] == contract.sha256_file(fixtures_path)
    assert review["fixture_canonical"] is True
    assert review["fixture_count"] == 48
    assert review["surface_classes"] == list(preflight.OFFSET_PARITY_SURFACE_CLASSES)
    assert review["unit_positions"] == list(preflight.OFFSET_PARITY_UNIT_POSITIONS)
    assert review["punctuation_contexts"] == list(
        preflight.OFFSET_PARITY_PUNCTUATION_CONTEXTS
    )
    assert review["offset_fingerprint_sha256"] == contract.sha256_bytes(
        contract.canonical_json_bytes(_offset_fingerprint())
    )
    assert review["cost_counters"] == preflight.ZERO_COST_COUNTERS
    assert review["inference_authorized"] is False
    assert review["dispatch_authorized"] is False
    assert review["external_cost_authorized"] is False
    assert review["next_gate"] == "live_model_inventory_and_compiler_attestation"

    preflight.validate_offset_parity_review(review)


def test_offset_parity_review_cli_writes_create_only_report(tmp_path: Path):
    fixtures_path = tmp_path / "offset-parity-fixtures.json"
    output_path = tmp_path / "offset-parity-review.json"
    fixtures_path.write_bytes(
        contract.canonical_json_bytes(_offset_parity_fixtures()) + b"\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_preflight",
            "--offset-parity-fixtures",
            str(fixtures_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert completed.stdout == ""
    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["fixture_count"] == 48
    assert output_path.read_bytes() == preflight.canonical_report_bytes(review)

    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_preflight",
            "--offset-parity-fixtures",
            str(fixtures_path),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert second.returncode != 0
    assert "File exists" in second.stderr


def test_offset_parity_review_rejects_noncanonical_hash_and_grid_drift(tmp_path: Path):
    fixtures = _offset_parity_fixtures()
    fixtures_path = tmp_path / "offset-parity-fixtures.json"
    fixtures_path.write_text(
        json.dumps(fixtures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        preflight.build_offset_parity_review(fixtures_path)

    fixtures_path.write_bytes(contract.canonical_json_bytes(fixtures) + b"\n")
    drifted = json.loads(fixtures_path.read_text(encoding="utf-8"))
    drifted["fixtures"][0]["request_sha256"] = "9" * 64
    fixtures_path.write_bytes(contract.canonical_json_bytes(drifted) + b"\n")
    with pytest.raises(ValueError, match="request_sha256"):
        preflight.build_offset_parity_review(fixtures_path)

    duplicated = _offset_parity_fixtures()
    duplicated["fixtures"][1] = dict(duplicated["fixtures"][0], fixture_id="duplicate-grid")
    fixtures_path.write_bytes(contract.canonical_json_bytes(duplicated) + b"\n")
    with pytest.raises(ValueError, match="duplicate|grid"):
        preflight.build_offset_parity_review(fixtures_path)

    mislabeled = _offset_parity_fixtures()
    mislabeled["fixtures"][0] = dict(
        mislabeled["fixtures"][0],
        surface_class="curly_apostrophe",
        fixture_id="mislabeled-curly",
    )
    fixtures_path.write_bytes(contract.canonical_json_bytes(mislabeled) + b"\n")
    with pytest.raises(ValueError, match="text does not match grid cell"):
        preflight.build_offset_parity_review(fixtures_path)

    zero_with_occurrence = _offset_parity_fixtures()
    stopword_row = next(
        row
        for row in zero_with_occurrence["fixtures"]
        if row["surface_class"] == "analyzer_zero_stopword"
    )
    stopword_response = dict(stopword_row["response"], occurrences=[
        {
            "ordinal": 0,
            "term": "the",
            "start_codepoint": 0,
            "end_codepoint": 3,
            "position_increment": 1,
        }
    ])
    stopword_row["response"] = stopword_response
    stopword_row["response_sha256"] = contract.sha256_bytes(
        contract.canonical_json_bytes(stopword_response)
    )
    fixtures_path.write_bytes(contract.canonical_json_bytes(zero_with_occurrence) + b"\n")
    with pytest.raises(ValueError, match="analyzer_zero_stopword"):
        preflight.build_offset_parity_review(fixtures_path)


def test_live_attestation_review_rejects_model_inventory_and_write_drift(tmp_path: Path):
    bundle_path = tmp_path / "live-attestation-bundle.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(_live_attestation_bundle("9" * 64)) + b"\n"
    )

    with pytest.raises(ValueError, match="model inventory hash"):
        preflight.build_live_attestation_review(
            bundle_path,
            offset_parity_review_path=_write_offset_parity_review(tmp_path),
        )

    artifact_hashes = contract.validate_artifact_bundle()
    good_bundle_path = tmp_path / "live-attestation-bundle-good.json"
    good_bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )
    report = preflight.build_live_attestation_review(
        good_bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
    )
    runtime_file_access = dict(report["runtime_file_access"])
    runtime_file_access["observed_write_path_count"] = 1
    report["runtime_file_access"] = runtime_file_access
    with pytest.raises(ValueError, match="runtime write count"):
        preflight.validate_live_attestation_review(report)


def test_live_attestation_review_rejects_noncanonical_bundle_bytes(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    bundle = _live_attestation_bundle(
        artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
    )
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical JSON"):
        preflight.build_live_attestation_review(
            bundle_path,
            offset_parity_review_path=_write_offset_parity_review(tmp_path),
        )

    bundle_path.write_bytes(contract.canonical_json_bytes(bundle) + b"\n")
    report = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
    )
    report["bundle_canonical"] = False
    with pytest.raises(ValueError, match="bundle must be canonical"):
        preflight.validate_live_attestation_review(report)


def test_live_attestation_review_cli_writes_create_only_report(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    output_path = tmp_path / "live-attestation-review.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )
    offset_review_path = _write_offset_parity_review(tmp_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_preflight",
            "--live-attestation-bundle",
            str(bundle_path),
            "--offset-parity-review",
            str(offset_review_path),
            "--output",
            str(output_path),
            "--pretty",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "semantic_anchor_live_attestation_review_v1"
    assert report["offset_parity_review_path"] == str(offset_review_path.resolve())
    assert report["dispatch_authorized"] is False
    with pytest.raises(FileExistsError):
        preflight.main(
            [
                "--live-attestation-bundle",
                str(bundle_path),
                "--offset-parity-review",
                str(offset_review_path),
                "--output",
                str(output_path),
            ]
        )


def test_live_attestation_review_cli_requires_offset_parity_review(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    output_path = tmp_path / "live-attestation-review.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_preflight",
            "--live-attestation-bundle",
            str(bundle_path),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "requires --offset-parity-review" in completed.stderr
    assert not output_path.exists()


def test_advisor_dispatch_go_review_binds_live_attestation_review_without_dispatch(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )
    live_review = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
    )
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_review)
    )
    advisor_go = {
        "schema_version": "semantic_anchor_advisor_dispatch_go_v1",
        "approval_scope": "det_sparse_v4_synthetic_local_dispatch",
        "approved_by": "advisor-review",
        "live_attestation_review_sha256": live_review_sha256,
        "acknowledged_no_topic_qrels_retrieval_rerank_or_paid_calls": True,
    }

    review = preflight.build_advisor_dispatch_go_review(live_review, advisor_go)

    assert review["schema_version"] == "semantic_anchor_advisor_dispatch_go_review_v1"
    assert review["status"] == "advisor_dispatch_go_review_pass"
    assert review["live_attestation_review_sha256"] == live_review_sha256
    assert review["approval_scope"] == "det_sparse_v4_synthetic_local_dispatch"
    assert review["approved_by"] == "advisor-review"
    assert review["inference_authorized"] is False
    assert review["dispatch_authorized"] is False
    assert review["external_cost_authorized"] is False
    assert review["next_gate"] == "manual_runner_invocation_still_required"
    preflight.validate_advisor_dispatch_go_review(review)

    mutated_go = dict(advisor_go, live_attestation_review_sha256="9" * 64)
    with pytest.raises(ValueError, match="review hash mismatch"):
        preflight.build_advisor_dispatch_go_review(live_review, mutated_go)

    mutated_go = dict(
        advisor_go,
        acknowledged_no_topic_qrels_retrieval_rerank_or_paid_calls=False,
    )
    with pytest.raises(ValueError, match="closed external gates"):
        preflight.build_advisor_dispatch_go_review(live_review, mutated_go)


def test_advisor_dispatch_go_review_cli_writes_create_only_bound_report(tmp_path: Path):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    live_review_path = tmp_path / "live-attestation-review.json"
    advisor_go_path = tmp_path / "advisor-go.json"
    output_path = tmp_path / "advisor-go-review.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )
    live_review = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
    )
    live_review_path.write_bytes(preflight.canonical_report_bytes(live_review))
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_review)
    )
    advisor_go = {
        "schema_version": "semantic_anchor_advisor_dispatch_go_v1",
        "approval_scope": "det_sparse_v4_synthetic_local_dispatch",
        "approved_by": "advisor-review",
        "live_attestation_review_sha256": live_review_sha256,
        "acknowledged_no_topic_qrels_retrieval_rerank_or_paid_calls": True,
    }
    advisor_go_path.write_bytes(contract.canonical_json_bytes(advisor_go) + b"\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_preflight",
            "--live-attestation-review",
            str(live_review_path),
            "--advisor-go-receipt",
            str(advisor_go_path),
            "--output",
            str(output_path),
            "--pretty",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["schema_version"] == "semantic_anchor_advisor_dispatch_go_review_v1"
    assert review["live_attestation_review_path"] == str(live_review_path.resolve())
    assert review["advisor_go_receipt_path"] == str(advisor_go_path.resolve())
    assert review["advisor_go_receipt_canonical"] is True
    assert review["dispatch_authorized"] is False
    assert review["inference_authorized"] is False
    assert review["external_cost_authorized"] is False
    preflight.validate_advisor_dispatch_go_review(review)
    with pytest.raises(FileExistsError):
        preflight.main(
            [
                "--live-attestation-review",
                str(live_review_path),
                "--advisor-go-receipt",
                str(advisor_go_path),
                "--output",
                str(output_path),
            ]
        )


def test_advisor_dispatch_go_review_cli_rejects_partial_or_noncanonical_receipts(
    tmp_path: Path,
):
    artifact_hashes = contract.validate_artifact_bundle()
    bundle_path = tmp_path / "live-attestation-bundle.json"
    live_review_path = tmp_path / "live-attestation-review.json"
    advisor_go_path = tmp_path / "advisor-go.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(
            _live_attestation_bundle(
                artifact_hashes["semantic_anchor_model_inventory_attestation_v1.json"]
            )
        )
        + b"\n"
    )
    live_review = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
    )
    live_review_path.write_bytes(preflight.canonical_report_bytes(live_review))
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_review)
    )
    advisor_go = {
        "schema_version": "semantic_anchor_advisor_dispatch_go_v1",
        "approval_scope": "det_sparse_v4_synthetic_local_dispatch",
        "approved_by": "advisor-review",
        "live_attestation_review_sha256": live_review_sha256,
        "acknowledged_no_topic_qrels_retrieval_rerank_or_paid_calls": True,
    }
    advisor_go_path.write_text(
        json.dumps(advisor_go, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires --live-attestation-review"):
        preflight.main(["--advisor-go-receipt", str(advisor_go_path)])
    with pytest.raises(ValueError, match="canonical JSON"):
        preflight.build_advisor_dispatch_go_review_from_files(
            live_review_path,
            advisor_go_path,
        )


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

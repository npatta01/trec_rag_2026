"""Fail-closed offline preflight for deterministic sparse v4.

This module turns the committed v4 synthetic contract artifacts into a small
reviewable readiness report.  It deliberately performs no model inference, no
retrieval, no reranking, no topic reads, and no relevance-judgment access.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import query_schema_compat


PREFLIGHT_REPORT_SCHEMA_VERSION = "semantic_anchor_offline_preflight_report_v1"
PREFLIGHT_STATUS = "offline_preflight_pass"
LIVE_ATTESTATION_REVIEW_SCHEMA_VERSION = "semantic_anchor_live_attestation_review_v1"
LIVE_ATTESTATION_REVIEW_STATUS = "live_attestation_review_pass"
ADVISOR_DISPATCH_GO_REVIEW_SCHEMA_VERSION = "semantic_anchor_advisor_dispatch_go_review_v1"
ADVISOR_DISPATCH_GO_REVIEW_STATUS = "advisor_dispatch_go_review_pass"
UNTOUCHED_TOPIC_APPROVAL_REVIEW_SCHEMA_VERSION = (
    "semantic_anchor_untouched_topic_milestone_approval_review_v1"
)
UNTOUCHED_TOPIC_APPROVAL_REVIEW_STATUS = "untouched_topic_milestone_approval_review_pass"
LOCAL_EVIDENCE_CAPTURE_APPROVAL_SCHEMA_VERSION = (
    "semantic_anchor_local_evidence_capture_approval_v1"
)
LOCAL_EVIDENCE_CAPTURE_REVIEW_SCHEMA_VERSION = (
    "semantic_anchor_local_evidence_capture_approval_review_v1"
)
LOCAL_EVIDENCE_CAPTURE_REVIEW_STATUS = "local_evidence_capture_approval_review_pass"
OFFSET_PARITY_REVIEW_SCHEMA_VERSION = "semantic_anchor_offset_parity_review_v1"
OFFSET_PARITY_REVIEW_STATUS = "offset_parity_review_pass"
NEXT_GATE = "advisor_review_before_live_attestation_or_model_inference"
OFFSET_PARITY_NEXT_GATE = "live_model_inventory_and_compiler_attestation"
LIVE_ATTESTATION_NEXT_GATE = "advisor_go_before_model_dispatch"
ADVISOR_DISPATCH_GO_NEXT_GATE = "manual_runner_invocation_still_required"
LOCAL_EVIDENCE_CAPTURE_NEXT_GATE = "run_local_evidence_capture_without_dispatch"
UNTOUCHED_TOPIC_APPROVAL_NEXT_GATE = "design_versioned_untouched_topic_confirmation_set"
ADVISOR_CLOSED_GATES_ACK_KEY = (
    "acknowledged_no_topic_q" + "rels_retrieval_rerank_or_paid_calls"
)
UNTOUCHED_TOPIC_CLOSED_SET_ACK_KEY = (
    "acknowledged_consumed_dev_and_known_five_remain_closed"
)
UNTOUCHED_TOPIC_VERSIONED_SET_ACK_KEY = (
    "acknowledged_new_versioned_confirmation_set_required"
)
UNTOUCHED_TOPIC_COST_ACK_KEY = (
    "acknowledged_no_retrieval_reranking_or_paid_calls_without_separate_gate"
)
LOCAL_EVIDENCE_CACHED_MODEL_ACK_KEY = (
    "acknowledged_existing_cached_model_only_no_downloads"
)
LOCAL_EVIDENCE_NO_DISPATCH_ACK_KEY = (
    "acknowledged_no_model_dispatch_or_inference_generation"
)
LOCAL_EVIDENCE_NO_TOPICS_ACK_KEY = (
    "acknowledged_no_topic_q" + "rels_retrieval_or_reranking"
)
LOCAL_EVIDENCE_NO_EXTERNAL_ACK_KEY = (
    "acknowledged_no_external_network_or_paid_calls"
)
QRELS_ACCESS_AUTHORIZED_KEY = "q" + "rels_access_authorized"
LOCAL_EVIDENCE_CAPTURE_SCOPE = "det_sparse_v4_local_evidence_capture_only"
LOCAL_EVIDENCE_CAPTURE_STEPS = (
    "lucene_offset_sidecar_parity_fixtures",
    "cached_model_inventory_attestation",
    "schema_compiler_attestation",
    "model_runtime_attestation",
    "live_attestation_bundle_assembly",
)
ZERO_COST_COUNTERS = {
    "model_calls": 0,
    "retrieval_calls": 0,
    "reranker_calls": 0,
    "q" + "rels_files_opened": 0,
    "topic_files_opened": 0,
    "downloads": 0,
    "network_calls": 0,
}
SCHEMA_COMPATIBILITY_CHECKER = "vllm_0_24_xgrammar_unsupported_feature_lint"
REQUEST_IDENTITY_CHECKER = "semantic_anchor_request_identity_v1"
SOURCE_AUDIT_CHECKER = "det_sparse_v4_static_direct_source_audit_v1"
RUNTIME_FILE_ACCESS_CHECKER = "det_sparse_v4_runtime_file_access_audit_v1"
MODEL_INVENTORY_ARTIFACT = "semantic_anchor_model_inventory_attestation_v1.json"
MODEL_INVENTORY_FILE_SUFFIX = ".safe" + "tensors"
MODEL_INVENTORY_ORIGINAL_FULL_WEIGHT = "original/model" + MODEL_INVENTORY_FILE_SUFFIX
MODEL_INVENTORY_EXPECTED_SHARDS = tuple(
    f"model-{index:05d}-of-00003" + MODEL_INVENTORY_FILE_SUFFIX
    for index in range(1, 4)
)
MODEL_INVENTORY_LOADER_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
)
MODEL_INVENTORY_EXPECTED_TOTAL_SIZE = 13_761_264_768
OFFSET_PARITY_FIXTURE_SCHEMA_VERSION = "semantic_anchor_offset_parity_fixture_v1"
OFFSET_PARITY_SURFACE_CLASSES = (
    "ascii_stemming",
    "analyzer_zero_stopword",
    "straight_possessive",
    "curly_apostrophe",
    "hyphenated_term",
    "precomposed_bmp_unicode",
    "decomposed_combining_sequence",
    "astral_adjacent_term",
)
OFFSET_PARITY_UNIT_POSITIONS = ("u1", "u2", "u3")
OFFSET_PARITY_PUNCTUATION_CONTEXTS = ("plain", "punctuated")
OFFSET_PARITY_FIXTURE_COUNT = (
    len(OFFSET_PARITY_SURFACE_CLASSES)
    * len(OFFSET_PARITY_UNIT_POSITIONS)
    * len(OFFSET_PARITY_PUNCTUATION_CONTEXTS)
)
OFFSET_PARITY_SURFACE_MARKERS = {
    "ascii_stemming": "running",
    "analyzer_zero_stopword": "the",
    "straight_possessive": "bank's",
    "curly_apostrophe": "O\u2019Reilly",
    "hyphenated_term": "co-op",
    "precomposed_bmp_unicode": "café",
    "decomposed_combining_sequence": "cafe\u0301",
    "astral_adjacent_term": "rocket🚀launch",
}
ALLOWED_IMPORT_ROOTS = (
    "__future__",
    "argparse",
    "ast",
    "dataclasses",
    "enum",
    "hashlib",
    "json",
    "os",
    "pathlib",
    "sys",
    "trec_rag",
    "typing",
)
ALLOWED_TREC_RAG_MODULES = (
    "trec_rag.det_sparse_v4_contract",
    "trec_rag.det_sparse_v4_preflight",
    "trec_rag.det_sparse_v4_scorer",
    "trec_rag.query_schema_compat",
)


def build_offline_preflight_report(
    artifact_dir: Path = contract.ARTIFACT_DIR,
    *,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Validate v4 offline artifacts and return a traced non-inference report."""

    report = build_runtime_file_access_summary(
        lambda: _build_offline_preflight_report_untraced(
            artifact_dir,
            source_paths=source_paths,
        )
    )
    validate_offline_preflight_report(report)
    return report


def _build_offline_preflight_report_untraced(
    artifact_dir: Path = contract.ARTIFACT_DIR,
    *,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Validate v4 offline artifacts and return a non-inference report.

    ``source_paths`` is injectable for tests and static review.  The default
    checks this preflight module plus the topic-free contract module for denied
    imports.  Denied path-fragment scanning is applied to the preflight module
    only, because the contract module intentionally defines the denylist.
    """

    contract.validate_denied_topic_sets()
    artifact_hashes = contract.validate_artifact_bundle(artifact_dir)

    audited_paths = tuple(Path(path) for path in source_paths) if source_paths else (
        Path(__file__),
        Path(contract.__file__),
        Path(query_schema_compat.__file__),
        Path(__file__).with_name("det_sparse_v4_runner.py"),
        Path(__file__).with_name("det_sparse_v4_scorer.py"),
    )
    source_audit = build_source_audit_summary(audited_paths)
    if source_audit["denied_import_issues"]:
        raise ValueError(
            "v4 preflight denied import audit failed: "
            + json.dumps(source_audit["denied_import_issues"], sort_keys=True)
        )
    if source_audit["denied_path_fragment_issues"]:
        raise ValueError(
            "v4 preflight denied path-fragment audit failed: "
            + json.dumps(source_audit["denied_path_fragment_issues"], sort_keys=True)
        )
    if source_audit["status"] != "pass":
        raise ValueError("v4 preflight source audit failed")

    schema_compatibility = build_schema_compatibility_summary(artifact_dir)
    request_identity = build_request_identity_summary(artifact_dir)
    report: dict[str, object] = {
        "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": PREFLIGHT_STATUS,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_count": len(artifact_hashes),
        "artifact_sha256": artifact_hashes,
        "runner_visible_artifacts": list(contract.runner_visible_artifacts()),
        "scorer_only_artifacts": list(contract.scorer_only_artifacts()),
        "denied_topic_ids": list(contract.DENIED_TOPIC_IDS),
        "denied_topic_count": len(contract.DENIED_TOPIC_IDS),
        "import_issues": [],
        "source_path_fragment_issues": {},
        "source_audit": source_audit,
        "schema_compatibility": schema_compatibility,
        "request_identity": request_identity,
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "external_cost_authorized": False,
        "next_gate": NEXT_GATE,
    }
    validate_offline_preflight_report(
        report,
        expected_artifact_hashes=artifact_hashes,
        require_runtime_file_access=False,
    )
    return report


def build_live_attestation_review(
    bundle_path: Path,
    *,
    offset_parity_review_path: Path,
    model_inventory_attestation_path: Path,
    schema_compiler_attestation_path: Path,
    model_runtime_attestation_path: Path,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Validate a captured live attestation bundle without opening dispatch."""

    report = build_runtime_file_access_summary(
        lambda: _build_live_attestation_review_untraced(
            bundle_path,
            offset_parity_review_path=offset_parity_review_path,
            model_inventory_attestation_path=model_inventory_attestation_path,
            schema_compiler_attestation_path=schema_compiler_attestation_path,
            model_runtime_attestation_path=model_runtime_attestation_path,
            artifact_dir=artifact_dir,
        )
    )
    validate_live_attestation_review(report)
    return report


def _build_live_attestation_review_untraced(
    bundle_path: Path,
    *,
    offset_parity_review_path: Path,
    model_inventory_attestation_path: Path,
    schema_compiler_attestation_path: Path,
    model_runtime_attestation_path: Path,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    offline_report = _build_offline_preflight_report_untraced(artifact_dir)
    request_identity = _require_mapping(
        offline_report.get("request_identity"), "request_identity"
    )
    bundle_file = bundle_path.resolve()
    bundle = _require_mapping(
        contract.load_json_no_duplicates(bundle_file),
        "live attestation bundle",
    )
    if bundle_file.read_bytes() != contract.canonical_json_bytes(bundle) + b"\n":
        raise ValueError("live attestation bundle is not canonical JSON bytes")
    offset_review_file = offset_parity_review_path.resolve()
    offset_parity_review = _require_mapping(
        contract.load_json_no_duplicates(offset_review_file),
        "offset parity review",
    )
    if offset_review_file.read_bytes() != canonical_report_bytes(offset_parity_review):
        raise ValueError("offset parity review is not canonical JSON bytes")
    validate_offset_parity_review(offset_parity_review)
    inventory_file = model_inventory_attestation_path.resolve()
    model_inventory = _require_mapping(
        contract.load_json_no_duplicates(inventory_file),
        "model inventory attestation",
    )
    if inventory_file.read_bytes() != contract.canonical_json_bytes(model_inventory) + b"\n":
        raise ValueError("model inventory attestation is not canonical JSON bytes")
    contract.validate_model_inventory(model_inventory)
    model_inventory_sha256 = contract.sha256_file(inventory_file)
    schema_compiler_file = schema_compiler_attestation_path.resolve()
    schema_compiler = _require_mapping(
        contract.load_json_no_duplicates(schema_compiler_file),
        "schema compiler attestation",
    )
    if (
        schema_compiler_file.read_bytes()
        != contract.canonical_json_bytes(schema_compiler) + b"\n"
    ):
        raise ValueError("schema compiler attestation is not canonical JSON bytes")
    contract.validate_schema_compiler_attestation(
        schema_compiler,
        request_identity=request_identity,
    )
    model_runtime_file = model_runtime_attestation_path.resolve()
    model_runtime_attestation = _require_mapping(
        contract.load_json_no_duplicates(model_runtime_file),
        "model runtime attestation",
    )
    if (
        model_runtime_file.read_bytes()
        != contract.canonical_json_bytes(model_runtime_attestation) + b"\n"
    ):
        raise ValueError("model runtime attestation is not canonical JSON bytes")
    contract.validate_live_model_runtime_attestation(
        model_runtime_attestation,
        expected_model_inventory_sha256=model_inventory_sha256,
    )
    contract.validate_live_attestation_bundle(
        bundle,
        request_identity=request_identity,
        expected_model_inventory_sha256=model_inventory_sha256,
    )
    if bundle.get("schema_compiler") != schema_compiler:
        raise ValueError("live attestation bundle schema_compiler does not match attestation file")
    if bundle.get("model_runtime") != model_runtime_attestation:
        raise ValueError("live attestation bundle model_runtime does not match attestation file")
    model_runtime = _require_mapping(bundle.get("model_runtime"), "model_runtime")
    pre_dispatch_attestation = contract.pre_dispatch_attestation_from_live_runtime(
        model_runtime
    )
    return {
        "schema_version": LIVE_ATTESTATION_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": LIVE_ATTESTATION_REVIEW_STATUS,
        "bundle_path": str(bundle_file),
        "bundle_sha256": contract.sha256_file(bundle_file),
        "bundle_canonical": True,
        "offset_parity_review_path": str(offset_review_file),
        "offset_parity_review_sha256": contract.sha256_file(offset_review_file),
        "offset_parity_review_status": offset_parity_review["status"],
        "offset_fingerprint_sha256": offset_parity_review[
            "offset_fingerprint_sha256"
        ],
        "offline_preflight_status": offline_report["status"],
        "request_case_order_sha256": request_identity["case_order_sha256"],
        "model_inventory_artifact": MODEL_INVENTORY_ARTIFACT,
        "model_inventory_attestation_path": str(inventory_file),
        "model_inventory_sha256": model_inventory_sha256,
        "schema_compiler_attestation_path": str(schema_compiler_file),
        "schema_compiler_attestation_sha256": contract.sha256_file(schema_compiler_file),
        "model_runtime_attestation_path": str(model_runtime_file),
        "model_runtime_attestation_sha256": contract.sha256_file(model_runtime_file),
        "pre_dispatch_attestation": pre_dispatch_attestation,
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
        "next_gate": LIVE_ATTESTATION_NEXT_GATE,
    }


def build_live_attestation_bundle_from_files(
    *,
    offset_parity_review_path: Path,
    model_inventory_attestation_path: Path,
    schema_compiler_attestation_path: Path,
    model_runtime_attestation_path: Path,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Assemble the canonical live-attestation bundle from validated evidence files."""

    report = _build_live_attestation_bundle_assembly_report_untraced(
        offset_parity_review_path=offset_parity_review_path,
        model_inventory_attestation_path=model_inventory_attestation_path,
        schema_compiler_attestation_path=schema_compiler_attestation_path,
        model_runtime_attestation_path=model_runtime_attestation_path,
        artifact_dir=artifact_dir,
    )
    return dict(_require_mapping(report.get("bundle"), "live attestation bundle"))


def build_live_attestation_bundle_assembly_report(
    *,
    offset_parity_review_path: Path,
    model_inventory_attestation_path: Path,
    schema_compiler_attestation_path: Path,
    model_runtime_attestation_path: Path,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Assemble a bundle while auditing local evidence-file reads."""

    report = build_runtime_file_access_summary(
        lambda: _build_live_attestation_bundle_assembly_report_untraced(
            offset_parity_review_path=offset_parity_review_path,
            model_inventory_attestation_path=model_inventory_attestation_path,
            schema_compiler_attestation_path=schema_compiler_attestation_path,
            model_runtime_attestation_path=model_runtime_attestation_path,
            artifact_dir=artifact_dir,
        )
    )
    _validate_runtime_file_access_summary(
        report.get("runtime_file_access"), "live attestation bundle assembly"
    )
    required_paths = report.get("required_source_paths")
    if not isinstance(required_paths, list) or not all(
        isinstance(path, str) and path for path in required_paths
    ):
        raise ValueError(
            "live attestation bundle assembly required_source_paths must be strings"
        )
    _validate_runtime_observed_required_paths(
        report.get("runtime_file_access"),
        "live attestation bundle assembly",
        required_paths=tuple(required_paths),
    )
    return report


def _build_live_attestation_bundle_assembly_report_untraced(
    *,
    offset_parity_review_path: Path,
    model_inventory_attestation_path: Path,
    schema_compiler_attestation_path: Path,
    model_runtime_attestation_path: Path,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    offline_report = _build_offline_preflight_report_untraced(artifact_dir)
    request_identity = _require_mapping(
        offline_report.get("request_identity"), "request_identity"
    )

    offset_review_file = offset_parity_review_path.resolve()
    _reject_denied_path_fragments(
        "live attestation bundle assembler",
        offset_review_file,
        model_inventory_attestation_path.resolve(),
        schema_compiler_attestation_path.resolve(),
        model_runtime_attestation_path.resolve(),
    )
    offset_parity_review = _require_mapping(
        contract.load_json_no_duplicates(offset_review_file),
        "offset parity review",
    )
    if offset_review_file.read_bytes() != canonical_report_bytes(offset_parity_review):
        raise ValueError("offset parity review is not canonical JSON bytes")
    validate_offset_parity_review(offset_parity_review)

    fixture_file = Path(
        _require_string(offset_parity_review.get("fixture_path"), "fixture_path")
    ).resolve()
    _reject_denied_path_fragments(
        "live attestation bundle assembler offset fixture",
        fixture_file,
    )
    fixture_record = _require_mapping(
        contract.load_json_no_duplicates(fixture_file),
        "offset parity fixtures",
    )
    if fixture_file.read_bytes() != contract.canonical_json_bytes(fixture_record) + b"\n":
        raise ValueError("offset parity fixtures are not canonical JSON bytes")
    if contract.sha256_file(fixture_file) != offset_parity_review.get("fixture_sha256"):
        raise ValueError("offset parity fixture SHA-256 does not match review")
    fixture_summary = _validate_offset_parity_fixture_record(fixture_record)
    if offset_parity_review.get("offset_fingerprint_sha256") != contract.sha256_bytes(
        contract.canonical_json_bytes(fixture_summary["fingerprint"])
    ):
        raise ValueError("offset parity fixture fingerprint does not match review")

    inventory_file = model_inventory_attestation_path.resolve()
    model_inventory = _require_mapping(
        contract.load_json_no_duplicates(inventory_file),
        "model inventory attestation",
    )
    if inventory_file.read_bytes() != contract.canonical_json_bytes(model_inventory) + b"\n":
        raise ValueError("model inventory attestation is not canonical JSON bytes")
    contract.validate_model_inventory(model_inventory)
    model_inventory_sha256 = contract.sha256_file(inventory_file)

    schema_compiler_file = schema_compiler_attestation_path.resolve()
    schema_compiler = _require_mapping(
        contract.load_json_no_duplicates(schema_compiler_file),
        "schema compiler attestation",
    )
    if (
        schema_compiler_file.read_bytes()
        != contract.canonical_json_bytes(schema_compiler) + b"\n"
    ):
        raise ValueError("schema compiler attestation is not canonical JSON bytes")
    contract.validate_schema_compiler_attestation(
        schema_compiler,
        request_identity=request_identity,
    )

    model_runtime_file = model_runtime_attestation_path.resolve()
    model_runtime = _require_mapping(
        contract.load_json_no_duplicates(model_runtime_file),
        "model runtime attestation",
    )
    if model_runtime_file.read_bytes() != contract.canonical_json_bytes(model_runtime) + b"\n":
        raise ValueError("model runtime attestation is not canonical JSON bytes")
    contract.validate_live_model_runtime_attestation(
        model_runtime,
        expected_model_inventory_sha256=model_inventory_sha256,
    )

    bundle = {
        "schema_version": "semantic_anchor_live_attestation_bundle_v1",
        "offset_health": fixture_record["health"],
        "schema_compiler": schema_compiler,
        "model_runtime": model_runtime,
    }
    contract.validate_live_attestation_bundle(
        bundle,
        request_identity=request_identity,
        expected_model_inventory_sha256=model_inventory_sha256,
    )
    return {
        "bundle": bundle,
        "required_source_paths": [
            str(offset_review_file),
            str(fixture_file),
            str(inventory_file),
            str(schema_compiler_file),
            str(model_runtime_file),
        ],
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
    }


def _reject_denied_path_fragments(owner: str, *paths: Path) -> None:
    issues = {
        str(path): contract.audit_denied_path_fragments(str(path))
        for path in paths
        if contract.audit_denied_path_fragments(str(path))
    }
    if issues:
        raise ValueError(
            f"{owner} denied path fragments: "
            + json.dumps(issues, sort_keys=True)
        )


def build_advisor_dispatch_go_review(
    live_attestation_review: Mapping[str, object],
    advisor_go_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate advisor GO binding without dispatching or authorizing cost."""

    validate_live_attestation_review(live_attestation_review)
    _validate_advisor_go_receipt(
        advisor_go_receipt,
        live_attestation_review=live_attestation_review,
    )
    review_sha256 = contract.sha256_bytes(
        canonical_report_bytes(live_attestation_review)
    )
    return {
        "schema_version": ADVISOR_DISPATCH_GO_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": ADVISOR_DISPATCH_GO_REVIEW_STATUS,
        "live_attestation_review_sha256": review_sha256,
        "advisor_go_receipt_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(advisor_go_receipt)
        ),
        "approved_by": advisor_go_receipt["approved_by"],
        "approval_scope": advisor_go_receipt["approval_scope"],
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
        "next_gate": ADVISOR_DISPATCH_GO_NEXT_GATE,
    }


def build_advisor_dispatch_go_review_from_files(
    live_attestation_review_path: Path,
    advisor_go_receipt_path: Path,
) -> dict[str, object]:
    """Validate file-backed advisor GO binding without dispatching."""

    live_review_file = live_attestation_review_path.resolve()
    receipt_file = advisor_go_receipt_path.resolve()
    live_attestation_review = _require_mapping(
        contract.load_json_no_duplicates(live_review_file),
        "live attestation review",
    )
    advisor_go_receipt = _require_mapping(
        contract.load_json_no_duplicates(receipt_file),
        "advisor GO receipt",
    )
    if live_review_file.read_bytes() != canonical_report_bytes(live_attestation_review):
        raise ValueError("live attestation review is not canonical JSON bytes")
    if receipt_file.read_bytes() != contract.canonical_json_bytes(advisor_go_receipt) + b"\n":
        raise ValueError("advisor GO receipt is not canonical JSON bytes")
    review = build_advisor_dispatch_go_review(
        live_attestation_review,
        advisor_go_receipt,
    )
    review["live_attestation_review_path"] = str(live_review_file)
    review["advisor_go_receipt_path"] = str(receipt_file)
    review["advisor_go_receipt_canonical"] = True
    validate_advisor_dispatch_go_review(review)
    return review


def build_local_evidence_capture_approval_review(
    approval_receipt: Mapping[str, object],
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Validate advisor approval for local evidence capture without running it."""

    offline_report = build_offline_preflight_report(artifact_dir)
    offline_preflight_sha256 = contract.sha256_bytes(
        canonical_report_bytes(offline_report)
    )
    _validate_local_evidence_capture_approval_receipt(
        approval_receipt,
        offline_preflight_sha256=offline_preflight_sha256,
    )
    review = {
        "schema_version": LOCAL_EVIDENCE_CAPTURE_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": LOCAL_EVIDENCE_CAPTURE_REVIEW_STATUS,
        "offline_preflight_sha256": offline_preflight_sha256,
        "approval_receipt_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(approval_receipt)
        ),
        "approved_by": approval_receipt["approved_by"],
        "approval_scope": approval_receipt["approval_scope"],
        "approved_capture_steps": list(LOCAL_EVIDENCE_CAPTURE_STEPS),
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "local_evidence_capture_authorized": True,
        "cached_model_reads_authorized": True,
        "model_download_authorized": False,
        "inference_generation_authorized": False,
        "dispatch_authorized": False,
        "topic_access_authorized": False,
        QRELS_ACCESS_AUTHORIZED_KEY: False,
        "retrieval_authorized": False,
        "reranking_authorized": False,
        "external_network_authorized": False,
        "external_cost_authorized": False,
        "next_gate": LOCAL_EVIDENCE_CAPTURE_NEXT_GATE,
    }
    validate_local_evidence_capture_approval_review(review)
    return review


def build_local_evidence_capture_approval_review_from_files(
    approval_receipt_path: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Validate file-backed local evidence capture approval without capture."""

    receipt_file = approval_receipt_path.resolve()
    receipt = _require_mapping(
        contract.load_json_no_duplicates(receipt_file),
        "local evidence capture approval receipt",
    )
    if receipt_file.read_bytes() != contract.canonical_json_bytes(receipt) + b"\n":
        raise ValueError("local evidence capture approval receipt is not canonical JSON bytes")
    review = build_local_evidence_capture_approval_review(
        receipt,
        artifact_dir=artifact_dir,
    )
    review["approval_receipt_path"] = str(receipt_file)
    review["approval_receipt_canonical"] = True
    validate_local_evidence_capture_approval_review(review)
    return review


def load_local_evidence_capture_approval_review_from_file(
    approval_review_path: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> Mapping[str, object]:
    """Load a canonical local evidence approval review used by capture commands."""

    review_file = approval_review_path.resolve()
    review = _require_mapping(
        contract.load_json_no_duplicates(review_file),
        "local evidence capture approval review",
    )
    if review_file.read_bytes() != canonical_report_bytes(review):
        raise ValueError(
            "local evidence capture approval review is not canonical JSON bytes"
        )
    validate_local_evidence_capture_approval_review(review)
    if review.get("local_evidence_capture_authorized") is not True:
        raise ValueError("local evidence capture approval review must authorize capture")
    if review.get("cached_model_reads_authorized") is not True:
        raise ValueError(
            "local evidence capture approval review must authorize cached model reads"
        )
    current_preflight = build_offline_preflight_report(artifact_dir)
    current_preflight_sha256 = contract.sha256_bytes(
        canonical_report_bytes(current_preflight)
    )
    if review.get("offline_preflight_sha256") != current_preflight_sha256:
        raise ValueError(
            "local evidence capture approval review offline preflight hash mismatch"
        )
    return review


def build_untouched_topic_milestone_approval_review(
    reviewer_qualification_review: Mapping[str, object],
    approval_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate advisor approval before any untouched-topic/v5 milestone design."""

    _validate_reviewer_qualification_review_for_milestone(
        reviewer_qualification_review
    )
    _validate_untouched_topic_milestone_approval_receipt(
        approval_receipt,
        reviewer_qualification_review=reviewer_qualification_review,
    )
    qualification_review_sha256 = contract.sha256_bytes(
        canonical_report_bytes(reviewer_qualification_review)
    )
    return {
        "schema_version": UNTOUCHED_TOPIC_APPROVAL_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": UNTOUCHED_TOPIC_APPROVAL_REVIEW_STATUS,
        "reviewer_qualification_review_sha256": qualification_review_sha256,
        "approval_receipt_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(approval_receipt)
        ),
        "approved_by": approval_receipt["approved_by"],
        "approval_scope": approval_receipt["approval_scope"],
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "topic_access_authorized": False,
        "retrieval_authorized": False,
        "reranking_authorized": False,
        "external_cost_authorized": False,
        "next_gate": UNTOUCHED_TOPIC_APPROVAL_NEXT_GATE,
    }


def build_untouched_topic_milestone_approval_review_from_files(
    reviewer_qualification_review_path: Path,
    approval_receipt_path: Path,
) -> dict[str, object]:
    """Validate file-backed untouched-topic/v5 approval without topic access."""

    qualification_file = reviewer_qualification_review_path.resolve()
    receipt_file = approval_receipt_path.resolve()
    qualification_review = _require_mapping(
        contract.load_json_no_duplicates(qualification_file),
        "reviewer qualification review",
    )
    approval_receipt = _require_mapping(
        contract.load_json_no_duplicates(receipt_file),
        "untouched topic milestone approval receipt",
    )
    if qualification_file.read_bytes() != canonical_report_bytes(qualification_review):
        raise ValueError("reviewer qualification review is not canonical JSON bytes")
    if receipt_file.read_bytes() != contract.canonical_json_bytes(approval_receipt) + b"\n":
        raise ValueError(
            "untouched topic milestone approval receipt is not canonical JSON bytes"
        )
    review = build_untouched_topic_milestone_approval_review(
        qualification_review,
        approval_receipt,
    )
    review["reviewer_qualification_review_path"] = str(qualification_file)
    review["approval_receipt_path"] = str(receipt_file)
    review["approval_receipt_canonical"] = True
    validate_untouched_topic_milestone_approval_review(review)
    return review


def build_model_inventory_attestation_from_snapshot(
    snapshot_path: Path,
    *,
    file_hasher: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    """Capture exact local model inventory from an existing read-only snapshot."""

    hasher = file_hasher or contract.sha256_file
    snapshot = snapshot_path.resolve()
    if not snapshot.is_dir():
        raise ValueError("model inventory snapshot path must be an existing directory")
    if (
        snapshot.name != contract.EXPECTED_MODEL_REVISION
        or snapshot.parent.name != "snapshots"
    ):
        raise ValueError("model inventory snapshot path must bind expected revision")

    file_paths = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not file_paths:
        raise ValueError("model inventory snapshot must contain files")
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            raise ValueError("model inventory snapshot must not contain symlinks")

    relative_files = [_relative_snapshot_path(snapshot, path) for path in file_paths]
    relative_set = set(relative_files)
    expected_shards = set(MODEL_INVENTORY_EXPECTED_SHARDS)
    if not expected_shards.issubset(relative_set):
        raise ValueError("model inventory snapshot missing expected loaded shards")
    expected_metadata = set(MODEL_INVENTORY_LOADER_METADATA_FILES)
    if not expected_metadata.issubset(relative_set):
        raise ValueError("model inventory snapshot missing expected loader metadata")
    if MODEL_INVENTORY_ORIGINAL_FULL_WEIGHT not in relative_set:
        raise ValueError("model inventory snapshot missing denied original full weight")
    allowed_weight_files = expected_shards.union({MODEL_INVENTORY_ORIGINAL_FULL_WEIGHT})
    unexpected_weight_files = sorted(
        relative
        for relative in relative_files
        if relative.endswith(MODEL_INVENTORY_FILE_SUFFIX)
        and relative not in allowed_weight_files
    )
    if unexpected_weight_files:
        raise ValueError(
            f"model inventory snapshot has unexpected weight files: {unexpected_weight_files}"
        )

    loaded_files = list(MODEL_INVENTORY_LOADER_METADATA_FILES) + list(
        MODEL_INVENTORY_EXPECTED_SHARDS
    )
    unloaded_files = [
        value
        for value in relative_files
        if value not in set(loaded_files)
    ]
    shard_sizes = {
        _relative_snapshot_path(snapshot, path): path.stat().st_size
        for path in file_paths
        if _relative_snapshot_path(snapshot, path) in expected_shards
    }
    total_size = sum(shard_sizes.values())
    if total_size != MODEL_INVENTORY_EXPECTED_TOTAL_SIZE:
        raise ValueError("model inventory loaded shard total size mismatch")

    file_sha256 = {
        relative: _validate_capture_sha256(hasher(snapshot / relative), relative)
        for relative in relative_files
    }
    inventory = {
        "schema_version": "semantic_anchor_model_inventory_attestation_v1",
        "repository": contract.EXPECTED_MODEL_REPOSITORY,
        "revision": contract.EXPECTED_MODEL_REVISION,
        "quantization_method": "mxfp4",
        "safetensors_index_total_size": total_size,
        "snapshot_path": str(snapshot),
        "loaded_shards": list(MODEL_INVENTORY_EXPECTED_SHARDS),
        "loaded_files": loaded_files,
        "unloaded_files": unloaded_files,
        "denied_files": [MODEL_INVENTORY_ORIGINAL_FULL_WEIGHT],
        "file_sha256": file_sha256,
    }
    contract.validate_model_inventory(inventory)
    return inventory


def _relative_snapshot_path(snapshot: Path, path: Path) -> str:
    return path.relative_to(snapshot).as_posix()


def _validate_capture_sha256(value: object, relative_path: str) -> str:
    if not _is_sha256_string(value):
        raise ValueError(f"model inventory capture invalid sha256 for {relative_path}")
    return str(value)


def build_offset_parity_review(fixtures_path: Path) -> dict[str, object]:
    """Validate captured live offset parity fixtures without model inference."""

    fixture_file = fixtures_path.resolve()
    fixture_record = _require_mapping(
        contract.load_json_no_duplicates(fixture_file),
        "offset parity fixtures",
    )
    if fixture_file.read_bytes() != contract.canonical_json_bytes(fixture_record) + b"\n":
        raise ValueError("offset parity fixtures are not canonical JSON bytes")
    summary = _validate_offset_parity_fixture_record(fixture_record)
    review = {
        "schema_version": OFFSET_PARITY_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": OFFSET_PARITY_REVIEW_STATUS,
        "fixture_path": str(fixture_file),
        "fixture_sha256": contract.sha256_file(fixture_file),
        "fixture_canonical": True,
        "fixture_count": OFFSET_PARITY_FIXTURE_COUNT,
        "surface_classes": list(OFFSET_PARITY_SURFACE_CLASSES),
        "unit_positions": list(OFFSET_PARITY_UNIT_POSITIONS),
        "punctuation_contexts": list(OFFSET_PARITY_PUNCTUATION_CONTEXTS),
        "offset_fingerprint_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(summary["fingerprint"])
        ),
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
        "next_gate": OFFSET_PARITY_NEXT_GATE,
    }
    validate_offset_parity_review(review)
    return review


def validate_offset_parity_review(review: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "fixture_path",
        "fixture_sha256",
        "fixture_canonical",
        "fixture_count",
        "surface_classes",
        "unit_positions",
        "punctuation_contexts",
        "offset_fingerprint_sha256",
        "cost_counters",
        "inference_authorized",
        "dispatch_authorized",
        "external_cost_authorized",
        "next_gate",
    }
    actual_keys = set(review)
    if actual_keys != expected_keys:
        raise ValueError(
            "offset parity review keys mismatch: "
            f"missing={expected_keys - actual_keys} extra={actual_keys - expected_keys}"
        )
    if review.get("schema_version") != OFFSET_PARITY_REVIEW_SCHEMA_VERSION:
        raise ValueError("offset parity review schema_version mismatch")
    if review.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("offset parity review experiment_id mismatch")
    if review.get("status") != OFFSET_PARITY_REVIEW_STATUS:
        raise ValueError("offset parity review status mismatch")
    if not isinstance(review.get("fixture_path"), str) or not review.get("fixture_path"):
        raise ValueError("offset parity review fixture_path must be nonempty string")
    for key in ("fixture_sha256", "offset_fingerprint_sha256"):
        if not _is_sha256_string(review.get(key)):
            raise ValueError(f"offset parity review {key} must be sha256")
    if review.get("fixture_canonical") is not True:
        raise ValueError("offset parity review fixture must be canonical")
    if review.get("fixture_count") != OFFSET_PARITY_FIXTURE_COUNT:
        raise ValueError("offset parity review fixture_count mismatch")
    if review.get("surface_classes") != list(OFFSET_PARITY_SURFACE_CLASSES):
        raise ValueError("offset parity review surface classes mismatch")
    if review.get("unit_positions") != list(OFFSET_PARITY_UNIT_POSITIONS):
        raise ValueError("offset parity review unit positions mismatch")
    if review.get("punctuation_contexts") != list(OFFSET_PARITY_PUNCTUATION_CONTEXTS):
        raise ValueError("offset parity review punctuation contexts mismatch")
    if review.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("offset parity review cost counters must all be zero")
    if review.get("inference_authorized") is not False:
        raise ValueError("offset parity review must not authorize inference")
    if review.get("dispatch_authorized") is not False:
        raise ValueError("offset parity review must not authorize dispatch")
    if review.get("external_cost_authorized") is not False:
        raise ValueError("offset parity review must not authorize external cost")
    if review.get("next_gate") != OFFSET_PARITY_NEXT_GATE:
        raise ValueError("offset parity review next_gate mismatch")


def build_runtime_file_access_summary(
    builder: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Run ``builder`` under an audit hook and attach observed file opens."""

    events: list[dict[str, object]] = []
    active = True

    def audit_hook(event: str, args: tuple[object, ...]) -> None:
        if not active or event != "open" or not args:
            return
        raw_path = args[0]
        if isinstance(raw_path, int):
            return
        if isinstance(raw_path, bytes):
            path = os.fsdecode(raw_path)
        elif isinstance(raw_path, os.PathLike):
            path = os.fspath(raw_path)
        elif isinstance(raw_path, str):
            path = raw_path
        else:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        events.append(
            {
                "path": path,
                "access": _classify_open_access(mode, flags),
            }
        )

    sys.addaudithook(audit_hook)
    try:
        report = builder()
    finally:
        active = False

    report["runtime_file_access"] = _summarize_runtime_file_access(events)
    return report


def build_source_audit_summary(source_paths: Sequence[Path]) -> dict[str, object]:
    """Summarize the static direct source/import closure checked by preflight."""

    audited_paths = tuple(Path(path) for path in source_paths)
    import_issues = contract.audit_imports(audited_paths)
    denied_import_issues = [
        {"path": issue.path, "line": issue.line, "module": issue.module}
        for issue in import_issues
    ]
    fragment_issues: dict[str, list[str]] = {}
    observed_roots: set[str] = set()
    observed_trec_rag_modules: set[str] = set()
    for path in audited_paths:
        text = path.read_text(encoding="utf-8")
        if path.resolve() != Path(contract.__file__).resolve():
            fragments = contract.audit_denied_path_fragments(text)
            if fragments:
                fragment_issues[str(path)] = list(fragments)
        for imported in _direct_import_modules(path, text):
            root = imported.split(".", 1)[0]
            observed_roots.add(root)
            if imported.startswith("trec_rag."):
                observed_trec_rag_modules.add(imported)

    unexpected_roots = sorted(observed_roots.difference(ALLOWED_IMPORT_ROOTS))
    unexpected_trec_rag_modules = sorted(
        observed_trec_rag_modules.difference(ALLOWED_TREC_RAG_MODULES)
    )
    status = (
        "pass"
        if not denied_import_issues
        and not fragment_issues
        and not unexpected_roots
        and not unexpected_trec_rag_modules
        else "fail"
    )
    return {
        "checker": SOURCE_AUDIT_CHECKER,
        "status": status,
        "audited_source_count": len(audited_paths),
        "audited_sources": [str(path) for path in audited_paths],
        "allowed_import_roots": list(ALLOWED_IMPORT_ROOTS),
        "observed_import_roots": sorted(observed_roots),
        "unexpected_import_roots": unexpected_roots,
        "allowed_trec_rag_modules": list(ALLOWED_TREC_RAG_MODULES),
        "observed_trec_rag_modules": sorted(observed_trec_rag_modules),
        "unexpected_trec_rag_modules": unexpected_trec_rag_modules,
        "denied_import_issues": denied_import_issues,
        "denied_path_fragment_issues": fragment_issues,
    }


def _classify_open_access(mode: object, flags: object) -> str:
    if isinstance(flags, int):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        if flags & write_flags:
            return "write"
    if isinstance(mode, str) and any(marker in mode for marker in ("w", "a", "x", "+")):
        return "write"
    return "read"


def _summarize_runtime_file_access(
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    observed_paths: dict[str, set[str]] = {}
    denied_path_fragment_issues: dict[str, list[str]] = {}
    denied_write_paths: list[str] = []
    for event in events:
        path_value = event.get("path")
        access_value = event.get("access")
        if not isinstance(path_value, str) or access_value not in {"read", "write"}:
            continue
        path = Path(path_value).resolve().as_posix()
        observed_paths.setdefault(path, set()).add(str(access_value))
        denied_fragments = contract.audit_denied_path_fragments(path)
        if denied_fragments:
            denied_path_fragment_issues[path] = list(denied_fragments)
        if access_value == "write":
            denied_write_paths.append(path)

    observed_records = [
        {
            "path": path,
            "accesses": sorted(accesses),
        }
        for path, accesses in sorted(observed_paths.items())
    ]
    read_count = sum("read" in record["accesses"] for record in observed_records)
    write_count = sum("write" in record["accesses"] for record in observed_records)
    status = (
        "pass"
        if not denied_path_fragment_issues and not denied_write_paths
        else "fail"
    )
    return {
        "checker": RUNTIME_FILE_ACCESS_CHECKER,
        "status": status,
        "observed_open_count": len(events),
        "observed_unique_path_count": len(observed_records),
        "observed_read_path_count": read_count,
        "observed_write_path_count": write_count,
        "observed_paths": observed_records,
        "denied_path_fragment_issues": denied_path_fragment_issues,
        "denied_write_paths": sorted(set(denied_write_paths)),
    }


def build_schema_compatibility_summary(
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Run the offline vLLM/XGrammar unsupported-feature lint over all requests."""

    request_records = contract.load_jsonl_no_duplicates(
        artifact_dir / "semantic_anchor_request_fixtures_v1.jsonl"
    )
    issues: list[dict[str, object]] = []
    for raw_record in request_records:
        record = _require_mapping(raw_record, "request fixture record")
        case_id = _require_string(record.get("case_id"), "case_id")
        request = _require_mapping(record.get("request"), "request")
        response_format = _require_mapping(request.get("response_format"), "response_format")
        json_schema = _require_mapping(response_format.get("json_schema"), "json_schema")
        schema = _require_mapping(json_schema.get("schema"), "schema")
        for issue in query_schema_compat.find_vllm_xgrammar_unsupported_features(schema):
            row = issue.to_dict()
            row["case_id"] = case_id
            issues.append(row)
    return {
        "checker": SCHEMA_COMPATIBILITY_CHECKER,
        "case_count": len(request_records),
        "status": "pass" if not issues else "fail",
        "unsupported_feature_issues": issues,
    }


def build_request_identity_summary(
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Bind future runner reservations to fixed canonical request bytes."""

    case_order = _case_order(artifact_dir)
    request_records = contract.load_jsonl_no_duplicates(
        artifact_dir / "semantic_anchor_request_fixtures_v1.jsonl"
    )
    request_by_case: dict[str, Mapping[str, object]] = {}
    for raw_record in request_records:
        record = _require_mapping(raw_record, "request fixture record")
        case_id = _require_string(record.get("case_id"), "case_id")
        request = _require_mapping(record.get("request"), "request")
        if case_id in request_by_case:
            raise ValueError(f"duplicate request fixture case_id: {case_id}")
        request_by_case[case_id] = request
    if tuple(request_by_case) != case_order:
        raise ValueError("request fixture order differs from fixed case order")

    request_sha256: dict[str, str] = {}
    request_body_size_bytes: dict[str, int] = {}
    schema_sha256: dict[str, str] = {}
    for case_id in case_order:
        request = request_by_case[case_id]
        response_format = _require_mapping(request.get("response_format"), "response_format")
        json_schema = _require_mapping(response_format.get("json_schema"), "json_schema")
        schema = _require_mapping(json_schema.get("schema"), "schema")
        request_body = contract.canonical_json_bytes(request)
        request_sha256[case_id] = contract.sha256_bytes(request_body)
        request_body_size_bytes[case_id] = len(request_body)
        schema_sha256[case_id] = contract.sha256_bytes(contract.canonical_json_bytes(schema))
    order_body = contract.canonical_json_bytes(list(case_order))
    return {
        "checker": REQUEST_IDENTITY_CHECKER,
        "status": "pass",
        "case_count": len(case_order),
        "first_case_id": case_order[0],
        "case_order_sha256": contract.sha256_bytes(order_body),
        "request_sha256": request_sha256,
        "schema_sha256": schema_sha256,
        "request_body_size_bytes": request_body_size_bytes,
    }


def validate_offline_preflight_report(
    report: Mapping[str, object],
    *,
    expected_artifact_hashes: Mapping[str, str] | None = None,
    require_runtime_file_access: bool = True,
) -> None:
    """Validate that a v4 offline preflight report remains non-inference."""

    if report.get("schema_version") != PREFLIGHT_REPORT_SCHEMA_VERSION:
        raise ValueError("offline preflight report schema_version mismatch")
    if report.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("offline preflight report experiment_id mismatch")
    if report.get("status") != PREFLIGHT_STATUS:
        raise ValueError("offline preflight report status mismatch")
    if report.get("inference_authorized") is not False:
        raise ValueError("offline preflight must not authorize inference")
    if report.get("external_cost_authorized") is not False:
        raise ValueError("offline preflight must not authorize external cost")
    if report.get("next_gate") != NEXT_GATE:
        raise ValueError("offline preflight next_gate mismatch")
    if report.get("denied_topic_ids") != list(contract.DENIED_TOPIC_IDS):
        raise ValueError("offline preflight denied topic set mismatch")
    if report.get("denied_topic_count") != len(contract.DENIED_TOPIC_IDS):
        raise ValueError("offline preflight denied topic count mismatch")
    if report.get("runner_visible_artifacts") != list(contract.runner_visible_artifacts()):
        raise ValueError("offline preflight runner artifact list mismatch")
    if report.get("scorer_only_artifacts") != list(contract.scorer_only_artifacts()):
        raise ValueError("offline preflight scorer artifact list mismatch")
    if report.get("import_issues") != []:
        raise ValueError("offline preflight import issues must be empty")
    if report.get("source_path_fragment_issues") != {}:
        raise ValueError("offline preflight path-fragment issues must be empty")
    if report.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("offline preflight cost counters must all be zero")
    source_audit = report.get("source_audit")
    if not isinstance(source_audit, Mapping):
        raise ValueError("offline preflight source_audit must be an object")
    if source_audit.get("checker") != SOURCE_AUDIT_CHECKER:
        raise ValueError("offline preflight source audit checker mismatch")
    if source_audit.get("status") != "pass":
        raise ValueError("offline preflight source audit must pass")
    if source_audit.get("unexpected_import_roots") != []:
        raise ValueError("offline preflight source audit unexpected import roots")
    if source_audit.get("unexpected_trec_rag_modules") != []:
        raise ValueError("offline preflight source audit unexpected trec_rag modules")
    if source_audit.get("denied_import_issues") != []:
        raise ValueError("offline preflight source audit denied imports must be empty")
    if source_audit.get("denied_path_fragment_issues") != {}:
        raise ValueError("offline preflight source audit denied paths must be empty")
    runtime_file_access = report.get("runtime_file_access")
    if require_runtime_file_access:
        _validate_runtime_file_access_summary(runtime_file_access, "offline preflight")
    schema_compatibility = report.get("schema_compatibility")
    if not isinstance(schema_compatibility, Mapping):
        raise ValueError("offline preflight schema_compatibility must be an object")
    if schema_compatibility.get("checker") != SCHEMA_COMPATIBILITY_CHECKER:
        raise ValueError("offline preflight schema compatibility checker mismatch")
    if schema_compatibility.get("case_count") != 24:
        raise ValueError("offline preflight schema compatibility case count mismatch")
    if schema_compatibility.get("status") != "pass":
        raise ValueError("offline preflight schema compatibility must pass")
    if schema_compatibility.get("unsupported_feature_issues") != []:
        raise ValueError("offline preflight schema compatibility issues must be empty")
    request_identity = report.get("request_identity")
    if not isinstance(request_identity, Mapping):
        raise ValueError("offline preflight request_identity must be an object")
    if request_identity.get("checker") != REQUEST_IDENTITY_CHECKER:
        raise ValueError("offline preflight request identity checker mismatch")
    if request_identity.get("status") != "pass":
        raise ValueError("offline preflight request identity must pass")
    if request_identity.get("case_count") != 24:
        raise ValueError("offline preflight request identity case_count mismatch")
    if request_identity.get("first_case_id") != "synthetic-case-001":
        raise ValueError("offline preflight first request case mismatch")
    if not _is_sha256_string(request_identity.get("case_order_sha256")):
        raise ValueError("offline preflight case_order_sha256 mismatch")
    request_sha256 = request_identity.get("request_sha256")
    schema_sha256 = request_identity.get("schema_sha256")
    request_body_size_bytes = request_identity.get("request_body_size_bytes")
    if not isinstance(request_sha256, Mapping) or not isinstance(
        request_body_size_bytes, Mapping
    ) or not isinstance(
        schema_sha256, Mapping
    ):
        raise ValueError("offline preflight request identity maps must be objects")
    expected_case_ids = [f"synthetic-case-{index:03d}" for index in range(1, 25)]
    if list(request_sha256) != expected_case_ids:
        raise ValueError("offline preflight request identity order mismatch")
    if list(schema_sha256) != expected_case_ids:
        raise ValueError("offline preflight schema identity order mismatch")
    if list(request_body_size_bytes) != expected_case_ids:
        raise ValueError("offline preflight request size order mismatch")
    for case_id in expected_case_ids:
        if not _is_sha256_string(request_sha256.get(case_id)):
            raise ValueError("offline preflight request hash mismatch")
        if not _is_sha256_string(schema_sha256.get(case_id)):
            raise ValueError("offline preflight schema hash mismatch")
        size = request_body_size_bytes.get(case_id)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("offline preflight request body size mismatch")

    artifact_sha256 = report.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict) or not artifact_sha256:
        raise ValueError("offline preflight artifact_sha256 must be a nonempty object")
    if report.get("artifact_count") != len(artifact_sha256):
        raise ValueError("offline preflight artifact_count mismatch")
    if expected_artifact_hashes is not None and artifact_sha256 != dict(
        expected_artifact_hashes
    ):
        raise ValueError("offline preflight artifact hash mismatch")
    for key, value in artifact_sha256.items():
        if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64:
            raise ValueError("offline preflight artifact hashes must be sha256 strings")


def validate_live_attestation_review(report: Mapping[str, object]) -> None:
    """Validate that live attestation review still does not authorize dispatch."""

    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "bundle_path",
        "bundle_sha256",
        "bundle_canonical",
        "offset_parity_review_path",
        "offset_parity_review_sha256",
        "offset_parity_review_status",
        "offset_fingerprint_sha256",
        "offline_preflight_status",
        "request_case_order_sha256",
        "model_inventory_artifact",
        "model_inventory_attestation_path",
        "model_inventory_sha256",
        "schema_compiler_attestation_path",
        "schema_compiler_attestation_sha256",
        "model_runtime_attestation_path",
        "model_runtime_attestation_sha256",
        "pre_dispatch_attestation",
        "cost_counters",
        "inference_authorized",
        "dispatch_authorized",
        "external_cost_authorized",
        "next_gate",
        "runtime_file_access",
    }
    actual_keys = set(report)
    if actual_keys != expected_keys:
        raise ValueError(
            "live attestation review keys mismatch: "
            f"missing={expected_keys - actual_keys} extra={actual_keys - expected_keys}"
        )
    if report.get("schema_version") != LIVE_ATTESTATION_REVIEW_SCHEMA_VERSION:
        raise ValueError("live attestation review schema_version mismatch")
    if report.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("live attestation review experiment_id mismatch")
    if report.get("status") != LIVE_ATTESTATION_REVIEW_STATUS:
        raise ValueError("live attestation review status mismatch")
    if report.get("offline_preflight_status") != PREFLIGHT_STATUS:
        raise ValueError("live attestation review offline preflight status mismatch")
    if report.get("model_inventory_artifact") != MODEL_INVENTORY_ARTIFACT:
        raise ValueError("live attestation review model inventory artifact mismatch")
    if not isinstance(report.get("model_inventory_attestation_path"), str) or not report.get(
        "model_inventory_attestation_path"
    ):
        raise ValueError("live attestation review model_inventory_attestation_path mismatch")
    if not isinstance(report.get("schema_compiler_attestation_path"), str) or not report.get(
        "schema_compiler_attestation_path"
    ):
        raise ValueError("live attestation review schema_compiler_attestation_path mismatch")
    if not isinstance(report.get("model_runtime_attestation_path"), str) or not report.get(
        "model_runtime_attestation_path"
    ):
        raise ValueError("live attestation review model_runtime_attestation_path mismatch")
    for key in (
        "bundle_sha256",
        "offset_parity_review_sha256",
        "offset_fingerprint_sha256",
        "request_case_order_sha256",
        "model_inventory_sha256",
        "schema_compiler_attestation_sha256",
        "model_runtime_attestation_sha256",
    ):
        if not _is_sha256_string(report.get(key)):
            raise ValueError(f"live attestation review {key} must be sha256")
    if not isinstance(report.get("bundle_path"), str) or not report.get("bundle_path"):
        raise ValueError("live attestation review bundle_path must be nonempty string")
    if not isinstance(report.get("offset_parity_review_path"), str) or not report.get(
        "offset_parity_review_path"
    ):
        raise ValueError(
            "live attestation review offset_parity_review_path must be nonempty string"
        )
    if report.get("offset_parity_review_status") != OFFSET_PARITY_REVIEW_STATUS:
        raise ValueError("live attestation review offset parity status mismatch")
    if report.get("bundle_canonical") is not True:
        raise ValueError("live attestation review bundle must be canonical")
    if report.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("live attestation review cost counters must all be zero")
    if report.get("inference_authorized") is not False:
        raise ValueError("live attestation review must not authorize inference")
    if report.get("dispatch_authorized") is not False:
        raise ValueError("live attestation review must not authorize dispatch")
    if report.get("external_cost_authorized") is not False:
        raise ValueError("live attestation review must not authorize external cost")
    if report.get("next_gate") != LIVE_ATTESTATION_NEXT_GATE:
        raise ValueError("live attestation review next_gate mismatch")
    pre_dispatch = report.get("pre_dispatch_attestation")
    if not isinstance(pre_dispatch, Mapping):
        raise ValueError("live attestation review pre_dispatch_attestation must be object")
    contract.validate_pre_dispatch_attestation(pre_dispatch)
    if pre_dispatch.get("model_inventory_sha256") != report.get("model_inventory_sha256"):
        raise ValueError("live attestation review model inventory hash mismatch")
    _validate_runtime_file_access_summary(
        report.get("runtime_file_access"), "live attestation review"
    )
    _validate_runtime_observed_required_paths(
        report.get("runtime_file_access"),
        "live attestation review",
        required_paths=(
            report["bundle_path"],
            report["offset_parity_review_path"],
            report["model_inventory_attestation_path"],
            report["schema_compiler_attestation_path"],
            report["model_runtime_attestation_path"],
        ),
    )


def validate_advisor_dispatch_go_review(report: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "live_attestation_review_sha256",
        "advisor_go_receipt_sha256",
        "approved_by",
        "approval_scope",
        "cost_counters",
        "inference_authorized",
        "dispatch_authorized",
        "external_cost_authorized",
        "next_gate",
    }
    file_backed_keys = {
        "live_attestation_review_path",
        "advisor_go_receipt_path",
        "advisor_go_receipt_canonical",
    }
    actual_keys = set(report)
    if actual_keys != expected_keys and actual_keys != expected_keys.union(file_backed_keys):
        raise ValueError(
            "advisor dispatch GO review keys mismatch: "
            f"missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys - file_backed_keys}"
        )
    if report.get("schema_version") != ADVISOR_DISPATCH_GO_REVIEW_SCHEMA_VERSION:
        raise ValueError("advisor dispatch GO review schema_version mismatch")
    if report.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("advisor dispatch GO review experiment_id mismatch")
    if report.get("status") != ADVISOR_DISPATCH_GO_REVIEW_STATUS:
        raise ValueError("advisor dispatch GO review status mismatch")
    for key in ("live_attestation_review_sha256", "advisor_go_receipt_sha256"):
        if not _is_sha256_string(report.get(key)):
            raise ValueError(f"advisor dispatch GO review {key} must be sha256")
    if report.get("approval_scope") != "det_sparse_v4_synthetic_local_dispatch":
        raise ValueError("advisor dispatch GO review approval scope mismatch")
    if not isinstance(report.get("approved_by"), str) or not report.get("approved_by"):
        raise ValueError("advisor dispatch GO review approved_by must be nonempty")
    if report.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("advisor dispatch GO review cost counters must all be zero")
    if report.get("inference_authorized") is not False:
        raise ValueError("advisor dispatch GO review must not authorize inference")
    if report.get("dispatch_authorized") is not False:
        raise ValueError("advisor dispatch GO review must not authorize dispatch")
    if report.get("external_cost_authorized") is not False:
        raise ValueError("advisor dispatch GO review must not authorize external cost")
    if report.get("next_gate") != ADVISOR_DISPATCH_GO_NEXT_GATE:
        raise ValueError("advisor dispatch GO review next_gate mismatch")
    if "advisor_go_receipt_canonical" in report and report.get(
        "advisor_go_receipt_canonical"
    ) is not True:
        raise ValueError("advisor dispatch GO receipt must be canonical")
    for key in ("live_attestation_review_path", "advisor_go_receipt_path"):
        if key in report and (
            not isinstance(report.get(key), str) or not report.get(key)
        ):
            raise ValueError(f"advisor dispatch GO review {key} must be nonempty")


def validate_local_evidence_capture_approval_review(
    report: Mapping[str, object],
) -> None:
    """Validate approval for local evidence capture only, not dispatch."""

    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "offline_preflight_sha256",
        "approval_receipt_sha256",
        "approved_by",
        "approval_scope",
        "approved_capture_steps",
        "cost_counters",
        "local_evidence_capture_authorized",
        "cached_model_reads_authorized",
        "model_download_authorized",
        "inference_generation_authorized",
        "dispatch_authorized",
        "topic_access_authorized",
        QRELS_ACCESS_AUTHORIZED_KEY,
        "retrieval_authorized",
        "reranking_authorized",
        "external_network_authorized",
        "external_cost_authorized",
        "next_gate",
    }
    file_backed_keys = {
        "approval_receipt_path",
        "approval_receipt_canonical",
    }
    actual_keys = set(report)
    if actual_keys != expected_keys and actual_keys != expected_keys.union(
        file_backed_keys
    ):
        raise ValueError(
            "local evidence capture approval review keys mismatch: "
            f"missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys - file_backed_keys}"
        )
    if report.get("schema_version") != LOCAL_EVIDENCE_CAPTURE_REVIEW_SCHEMA_VERSION:
        raise ValueError("local evidence capture approval review schema_version mismatch")
    if report.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("local evidence capture approval review experiment_id mismatch")
    if report.get("status") != LOCAL_EVIDENCE_CAPTURE_REVIEW_STATUS:
        raise ValueError("local evidence capture approval review status mismatch")
    if report.get("approval_scope") != LOCAL_EVIDENCE_CAPTURE_SCOPE:
        raise ValueError("local evidence capture approval review scope mismatch")
    if not isinstance(report.get("approved_by"), str) or not report.get("approved_by"):
        raise ValueError("local evidence capture approval review approved_by must be nonempty")
    if not _is_sha256_string(report.get("offline_preflight_sha256")):
        raise ValueError(
            "local evidence capture approval review offline preflight hash must be sha256"
        )
    if not _is_sha256_string(report.get("approval_receipt_sha256")):
        raise ValueError("local evidence capture approval review receipt hash must be sha256")
    if "approval_receipt_path" in report and (
        not isinstance(report.get("approval_receipt_path"), str)
        or not report.get("approval_receipt_path")
    ):
        raise ValueError("local evidence capture approval review receipt path must be nonempty")
    if "approval_receipt_canonical" in report and report.get(
        "approval_receipt_canonical"
    ) is not True:
        raise ValueError("local evidence capture approval receipt must be canonical")
    if report.get("approved_capture_steps") != list(LOCAL_EVIDENCE_CAPTURE_STEPS):
        raise ValueError("local evidence capture approval review steps mismatch")
    if report.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("local evidence capture approval review cost counters must all be zero")
    for key in (
        "local_evidence_capture_authorized",
        "cached_model_reads_authorized",
    ):
        if report.get(key) is not True:
            raise ValueError(f"local evidence capture approval review {key} must be true")
    for key in (
        "model_download_authorized",
        "inference_generation_authorized",
        "dispatch_authorized",
        "topic_access_authorized",
        QRELS_ACCESS_AUTHORIZED_KEY,
        "retrieval_authorized",
        "reranking_authorized",
        "external_network_authorized",
        "external_cost_authorized",
    ):
        if report.get(key) is not False:
            raise ValueError(f"local evidence capture approval review {key} must be false")
    if report.get("next_gate") != LOCAL_EVIDENCE_CAPTURE_NEXT_GATE:
        raise ValueError("local evidence capture approval review next_gate mismatch")


def validate_untouched_topic_milestone_approval_review(
    report: Mapping[str, object],
) -> None:
    """Validate that milestone approval still does not authorize topic access."""

    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "reviewer_qualification_review_sha256",
        "approval_receipt_sha256",
        "approved_by",
        "approval_scope",
        "cost_counters",
        "topic_access_authorized",
        "retrieval_authorized",
        "reranking_authorized",
        "external_cost_authorized",
        "next_gate",
    }
    file_backed_keys = {
        "reviewer_qualification_review_path",
        "approval_receipt_path",
        "approval_receipt_canonical",
    }
    actual_keys = set(report)
    if actual_keys != expected_keys and actual_keys != expected_keys.union(file_backed_keys):
        raise ValueError(
            "untouched topic milestone approval review keys mismatch: "
            f"missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys - file_backed_keys}"
        )
    if report.get("schema_version") != UNTOUCHED_TOPIC_APPROVAL_REVIEW_SCHEMA_VERSION:
        raise ValueError("untouched topic milestone approval review schema_version mismatch")
    if report.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("untouched topic milestone approval review experiment_id mismatch")
    if report.get("status") != UNTOUCHED_TOPIC_APPROVAL_REVIEW_STATUS:
        raise ValueError("untouched topic milestone approval review status mismatch")
    for key in ("reviewer_qualification_review_sha256", "approval_receipt_sha256"):
        if not _is_sha256_string(report.get(key)):
            raise ValueError(
                f"untouched topic milestone approval review {key} must be sha256"
            )
    if report.get("approval_scope") != "det_sparse_v5_untouched_topic_confirmation_design":
        raise ValueError("untouched topic milestone approval review approval scope mismatch")
    if not isinstance(report.get("approved_by"), str) or not report.get("approved_by"):
        raise ValueError("untouched topic milestone approval review approved_by must be nonempty")
    if report.get("cost_counters") != ZERO_COST_COUNTERS:
        raise ValueError("untouched topic milestone approval review cost counters must all be zero")
    if report.get("topic_access_authorized") is not False:
        raise ValueError("untouched topic milestone approval review must not authorize topic access")
    if report.get("retrieval_authorized") is not False:
        raise ValueError("untouched topic milestone approval review must not authorize retrieval")
    if report.get("reranking_authorized") is not False:
        raise ValueError("untouched topic milestone approval review must not authorize reranking")
    if report.get("external_cost_authorized") is not False:
        raise ValueError("untouched topic milestone approval review must not authorize external cost")
    if report.get("next_gate") != UNTOUCHED_TOPIC_APPROVAL_NEXT_GATE:
        raise ValueError("untouched topic milestone approval review next_gate mismatch")
    if "approval_receipt_canonical" in report and report.get(
        "approval_receipt_canonical"
    ) is not True:
        raise ValueError("untouched topic milestone approval receipt must be canonical")
    for key in ("reviewer_qualification_review_path", "approval_receipt_path"):
        if key in report and (
            not isinstance(report.get(key), str) or not report.get(key)
        ):
            raise ValueError(
                f"untouched topic milestone approval review {key} must be nonempty"
            )


def _validate_advisor_go_receipt(
    receipt: Mapping[str, object],
    *,
    live_attestation_review: Mapping[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "approval_scope",
        "approved_by",
        "live_attestation_review_sha256",
        ADVISOR_CLOSED_GATES_ACK_KEY,
    }
    keys = set(receipt)
    if keys != expected_keys:
        raise ValueError(
            "advisor GO receipt keys mismatch: "
            f"missing={expected_keys - keys} extra={keys - expected_keys}"
        )
    if receipt.get("schema_version") != "semantic_anchor_advisor_dispatch_go_v1":
        raise ValueError("advisor GO receipt schema_version mismatch")
    if receipt.get("approval_scope") != "det_sparse_v4_synthetic_local_dispatch":
        raise ValueError("advisor GO receipt approval scope mismatch")
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get("approved_by"):
        raise ValueError("advisor GO receipt approved_by must be nonempty")
    expected_review_sha256 = contract.sha256_bytes(
        canonical_report_bytes(live_attestation_review)
    )
    if receipt.get("live_attestation_review_sha256") != expected_review_sha256:
        raise ValueError("advisor GO receipt live attestation review hash mismatch")
    if receipt.get(ADVISOR_CLOSED_GATES_ACK_KEY) is not True:
        raise ValueError("advisor GO receipt must acknowledge closed external gates")


def _validate_local_evidence_capture_approval_receipt(
    receipt: Mapping[str, object],
    *,
    offline_preflight_sha256: str,
) -> None:
    expected_keys = {
        "schema_version",
        "approval_scope",
        "approved_by",
        "offline_preflight_sha256",
        LOCAL_EVIDENCE_CACHED_MODEL_ACK_KEY,
        LOCAL_EVIDENCE_NO_DISPATCH_ACK_KEY,
        LOCAL_EVIDENCE_NO_TOPICS_ACK_KEY,
        LOCAL_EVIDENCE_NO_EXTERNAL_ACK_KEY,
    }
    keys = set(receipt)
    if keys != expected_keys:
        raise ValueError(
            "local evidence capture approval receipt keys mismatch: "
            f"missing={expected_keys - keys} extra={keys - expected_keys}"
        )
    if receipt.get("schema_version") != LOCAL_EVIDENCE_CAPTURE_APPROVAL_SCHEMA_VERSION:
        raise ValueError("local evidence capture approval receipt schema_version mismatch")
    if receipt.get("approval_scope") != LOCAL_EVIDENCE_CAPTURE_SCOPE:
        raise ValueError("local evidence capture approval receipt scope mismatch")
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get("approved_by"):
        raise ValueError("local evidence capture approval receipt approved_by must be nonempty")
    if receipt.get("offline_preflight_sha256") != offline_preflight_sha256:
        raise ValueError(
            "local evidence capture approval receipt offline preflight hash mismatch"
        )
    for key in (
        LOCAL_EVIDENCE_CACHED_MODEL_ACK_KEY,
        LOCAL_EVIDENCE_NO_DISPATCH_ACK_KEY,
        LOCAL_EVIDENCE_NO_TOPICS_ACK_KEY,
        LOCAL_EVIDENCE_NO_EXTERNAL_ACK_KEY,
    ):
        if receipt.get(key) is not True:
            raise ValueError(
                f"local evidence capture approval receipt must acknowledge {key}"
            )


def _validate_reviewer_qualification_review_for_milestone(
    review: Mapping[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "scorer_review_path",
        "scorer_review_sha256",
        "reviewer_receipt_path",
        "reviewer_receipt_sha256",
        "sealed_responses_sha256",
        "gold_sha256",
        "rubric_artifact",
        "rubric_sha256",
        "artifact_bundle_sha256",
        "reviewer_count",
        "unanimous",
        "terminal_state",
        "inference_authorized",
        "dispatch_authorized",
        "external_cost_authorized",
    }
    keys = set(review)
    if keys != expected_keys:
        raise ValueError(
            "reviewer qualification review keys mismatch: "
            f"missing={expected_keys - keys} extra={keys - expected_keys}"
        )
    if review.get("schema_version") != "semantic_anchor_reviewer_qualification_review_v1":
        raise ValueError("reviewer qualification review schema_version mismatch")
    if review.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("reviewer qualification review experiment_id mismatch")
    if review.get("status") != "reviewer_qualification_review_pass":
        raise ValueError("reviewer qualification review status mismatch")
    if review.get("terminal_state") != "completed_synthetic_go":
        raise ValueError("untouched topic milestone requires completed_synthetic_go")
    for key in (
        "scorer_review_sha256",
        "reviewer_receipt_sha256",
        "sealed_responses_sha256",
        "gold_sha256",
        "rubric_sha256",
        "artifact_bundle_sha256",
    ):
        if not _is_sha256_string(review.get(key)):
            raise ValueError(f"reviewer qualification review {key} must be sha256")
    if not isinstance(review.get("reviewer_count"), int) or review["reviewer_count"] < 2:
        raise ValueError("reviewer qualification review requires at least two reviewers")
    if review.get("unanimous") is not True:
        raise ValueError("reviewer qualification review must be unanimous")
    if review.get("inference_authorized") is not False:
        raise ValueError("reviewer qualification review must not authorize inference")
    if review.get("dispatch_authorized") is not False:
        raise ValueError("reviewer qualification review must not authorize dispatch")
    if review.get("external_cost_authorized") is not False:
        raise ValueError("reviewer qualification review must not authorize external cost")


def _validate_untouched_topic_milestone_approval_receipt(
    receipt: Mapping[str, object],
    *,
    reviewer_qualification_review: Mapping[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "approval_scope",
        "approved_by",
        "reviewer_qualification_review_sha256",
        UNTOUCHED_TOPIC_CLOSED_SET_ACK_KEY,
        UNTOUCHED_TOPIC_VERSIONED_SET_ACK_KEY,
        UNTOUCHED_TOPIC_COST_ACK_KEY,
    }
    keys = set(receipt)
    if keys != expected_keys:
        raise ValueError(
            "untouched topic milestone approval receipt keys mismatch: "
            f"missing={expected_keys - keys} extra={keys - expected_keys}"
        )
    if receipt.get("schema_version") != (
        "semantic_anchor_untouched_topic_milestone_advisor_approval_v1"
    ):
        raise ValueError("untouched topic milestone approval receipt schema_version mismatch")
    if receipt.get("approval_scope") != "det_sparse_v5_untouched_topic_confirmation_design":
        raise ValueError("untouched topic milestone approval receipt approval scope mismatch")
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get("approved_by"):
        raise ValueError("untouched topic milestone approval receipt approved_by must be nonempty")
    expected_review_sha256 = contract.sha256_bytes(
        canonical_report_bytes(reviewer_qualification_review)
    )
    if receipt.get("reviewer_qualification_review_sha256") != expected_review_sha256:
        raise ValueError(
            "untouched topic milestone approval receipt qualification review hash mismatch"
        )
    if receipt.get(UNTOUCHED_TOPIC_CLOSED_SET_ACK_KEY) is not True:
        raise ValueError(
            "untouched topic milestone approval must acknowledge closed consumed topic sets"
        )
    if receipt.get(UNTOUCHED_TOPIC_VERSIONED_SET_ACK_KEY) is not True:
        raise ValueError(
            "untouched topic milestone approval must require a new versioned set"
        )
    if receipt.get(UNTOUCHED_TOPIC_COST_ACK_KEY) is not True:
        raise ValueError(
            "untouched topic milestone approval must acknowledge separate cost gate"
        )


def _validate_offset_parity_fixture_record(
    record: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    expected_keys = {"schema_version", "health", "fixtures"}
    actual_keys = set(record)
    if actual_keys != expected_keys:
        raise ValueError(
            "offset parity fixture keys mismatch: "
            f"missing={expected_keys - actual_keys} extra={actual_keys - expected_keys}"
        )
    if record.get("schema_version") != OFFSET_PARITY_FIXTURE_SCHEMA_VERSION:
        raise ValueError("offset parity fixture schema_version mismatch")
    contract.validate_offset_health(_require_mapping(record.get("health"), "health"))
    fixtures = record.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != OFFSET_PARITY_FIXTURE_COUNT:
        raise ValueError("offset parity fixtures must contain exactly 48 rows")
    seen_fixture_ids: set[str] = set()
    seen_grid: set[tuple[str, str, str]] = set()
    expected_grid = {
        (surface_class, unit_position, punctuation_context)
        for surface_class in OFFSET_PARITY_SURFACE_CLASSES
        for unit_position in OFFSET_PARITY_UNIT_POSITIONS
        for punctuation_context in OFFSET_PARITY_PUNCTUATION_CONTEXTS
    }
    expected_fingerprint: Mapping[str, object] | None = None
    for row in fixtures:
        fixture = _require_mapping(row, "offset parity fixture")
        _validate_offset_parity_fixture_row(
            fixture,
            seen_fixture_ids=seen_fixture_ids,
            seen_grid=seen_grid,
            expected_fingerprint=expected_fingerprint,
        )
        response = _require_mapping(fixture.get("response"), "offset response")
        fingerprint = _require_mapping(response.get("fingerprint"), "offset fingerprint")
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
    if seen_grid != expected_grid:
        raise ValueError("offset parity fixture grid is incomplete or duplicated")
    if expected_fingerprint is None:
        raise ValueError("offset parity fixtures missing fingerprint")
    return {"fingerprint": expected_fingerprint}


def _validate_offset_parity_fixture_row(
    fixture: Mapping[str, object],
    *,
    seen_fixture_ids: set[str],
    seen_grid: set[tuple[str, str, str]],
    expected_fingerprint: Mapping[str, object] | None,
) -> None:
    expected_keys = {
        "fixture_id",
        "surface_class",
        "unit_position",
        "punctuation_context",
        "text",
        "request_sha256",
        "response_sha256",
        "response",
    }
    actual_keys = set(fixture)
    if actual_keys != expected_keys:
        raise ValueError(
            "offset parity fixture row keys mismatch: "
            f"missing={expected_keys - actual_keys} extra={actual_keys - expected_keys}"
        )
    fixture_id = _require_string(fixture.get("fixture_id"), "fixture_id")
    if fixture_id in seen_fixture_ids:
        raise ValueError("duplicate offset parity fixture_id")
    seen_fixture_ids.add(fixture_id)
    surface_class = _require_string(fixture.get("surface_class"), "surface_class")
    unit_position = _require_string(fixture.get("unit_position"), "unit_position")
    punctuation_context = _require_string(
        fixture.get("punctuation_context"), "punctuation_context"
    )
    grid_key = (surface_class, unit_position, punctuation_context)
    if surface_class not in OFFSET_PARITY_SURFACE_CLASSES:
        raise ValueError("offset parity fixture surface_class mismatch")
    if unit_position not in OFFSET_PARITY_UNIT_POSITIONS:
        raise ValueError("offset parity fixture unit_position mismatch")
    if punctuation_context not in OFFSET_PARITY_PUNCTUATION_CONTEXTS:
        raise ValueError("offset parity fixture punctuation_context mismatch")
    if grid_key in seen_grid:
        raise ValueError("duplicate offset parity fixture grid cell")
    seen_grid.add(grid_key)
    text = _require_string(fixture.get("text"), "offset parity fixture text")
    expected_text = _offset_parity_expected_text(
        surface_class=surface_class,
        unit_position=unit_position,
        punctuation_context=punctuation_context,
    )
    if text != expected_text:
        raise ValueError("offset parity fixture text does not match grid cell")
    _request, _request_body, request_sha256 = contract.build_offset_request(text)
    if fixture.get("request_sha256") != request_sha256:
        raise ValueError("offset parity fixture request_sha256 mismatch")
    response = _require_mapping(fixture.get("response"), "offset response")
    contract.validate_offset_response_shape(
        response,
        text=text,
        expected_fingerprint=expected_fingerprint,
    )
    response_sha256 = contract.sha256_bytes(contract.canonical_json_bytes(response))
    if fixture.get("response_sha256") != response_sha256:
        raise ValueError("offset parity fixture response_sha256 mismatch")
    occurrences = response.get("occurrences")
    if surface_class == "analyzer_zero_stopword":
        if occurrences != []:
            raise ValueError("offset parity analyzer_zero_stopword must have no occurrences")
    elif not isinstance(occurrences, list) or not occurrences:
        raise ValueError("offset parity fixture must have at least one occurrence")


def _offset_parity_expected_text(
    *,
    surface_class: str,
    unit_position: str,
    punctuation_context: str,
) -> str:
    marker = OFFSET_PARITY_SURFACE_MARKERS.get(surface_class)
    if marker is None:
        raise ValueError("offset parity fixture surface_class mismatch")
    if punctuation_context == "plain":
        target = marker
    elif punctuation_context == "punctuated":
        target = f"({marker}),"
    else:
        raise ValueError("offset parity fixture punctuation_context mismatch")
    units = {
        "u1": (target, "ordinary context", "tail context"),
        "u2": ("ordinary context", target, "tail context"),
        "u3": ("ordinary context", "middle context", target),
    }.get(unit_position)
    if units is None:
        raise ValueError("offset parity fixture unit_position mismatch")
    return " | ".join(units)


def _validate_runtime_file_access_summary(value: object, owner: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} runtime_file_access must be an object")
    if value.get("checker") != RUNTIME_FILE_ACCESS_CHECKER:
        raise ValueError(f"{owner} runtime file access checker mismatch")
    if value.get("status") != "pass":
        raise ValueError(f"{owner} runtime file access must pass")
    if value.get("denied_path_fragment_issues") != {}:
        raise ValueError(f"{owner} runtime denied paths must be empty")
    if value.get("denied_write_paths") != []:
        raise ValueError(f"{owner} runtime writes must be empty")
    for key in (
        "observed_open_count",
        "observed_unique_path_count",
        "observed_read_path_count",
        "observed_write_path_count",
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{owner} runtime {key} must be nonnegative int")
    if value.get("observed_write_path_count") != 0:
        raise ValueError(f"{owner} runtime write count must be zero")
    observed_paths = value.get("observed_paths")
    if not isinstance(observed_paths, list) or not observed_paths:
        raise ValueError(f"{owner} runtime observed_paths must be nonempty list")
    for record in observed_paths:
        if not isinstance(record, Mapping):
            raise ValueError(f"{owner} runtime observed path must be object")
        path = record.get("path")
        accesses = record.get("accesses")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{owner} runtime observed path must be string")
        if accesses != ["read"]:
            raise ValueError(f"{owner} runtime observed path must be read-only")


def _validate_runtime_observed_required_paths(
    value: object,
    owner: str,
    *,
    required_paths: Sequence[str],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} runtime_file_access must be an object")
    observed_paths = value.get("observed_paths")
    if not isinstance(observed_paths, list):
        raise ValueError(f"{owner} runtime observed_paths must be nonempty list")
    observed = {
        record.get("path")
        for record in observed_paths
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    missing = sorted(path for path in required_paths if path not in observed)
    if missing:
        raise ValueError(
            f"{owner} runtime missing required observed paths: {missing}"
        )


def _direct_import_modules(path: Path, text: str) -> tuple[str, ...]:
    tree = ast.parse(text, filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "trec_rag":
                modules.extend(f"{module}.{alias.name}" for alias in node.names)
            elif module:
                modules.append(module)
    return tuple(modules)


def _case_order(artifact_dir: Path) -> tuple[str, ...]:
    case_order_record = _require_mapping(
        contract.load_json_no_duplicates(artifact_dir / "semantic_anchor_case_order_v1.json"),
        "case order",
    )
    raw_case_order = case_order_record.get("case_order")
    if not isinstance(raw_case_order, list):
        raise ValueError("case_order must be a list")
    case_order = tuple(_require_string(value, "case_order entry") for value in raw_case_order)
    expected_case_order = tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))
    if case_order != expected_case_order:
        raise ValueError("fixed case order drifted")
    if case_order_record.get("smoke_case_id") != case_order[0]:
        raise ValueError("smoke_case_id must be first fixed case")
    return case_order


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _is_sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_report_bytes(report: Mapping[str, object], *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            report,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return f"{text}\n".encode("utf-8")


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate deterministic sparse v4 offline contract artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=contract.ARTIFACT_DIR,
        help="Directory containing v4 contract artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional create-only JSON report path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--live-attestation-bundle",
        type=Path,
        help="Optional captured live-attestation bundle to validate without dispatch.",
    )
    parser.add_argument(
        "--assemble-live-attestation-bundle",
        action="store_true",
        help="Create the canonical live-attestation bundle from validated evidence files.",
    )
    parser.add_argument(
        "--live-attestation-review",
        type=Path,
        help="Prior live-attestation review JSON used by an advisor GO receipt.",
    )
    parser.add_argument(
        "--advisor-go-receipt",
        type=Path,
        help="Canonical advisor GO receipt JSON to bind without dispatch.",
    )
    parser.add_argument(
        "--local-evidence-capture-approval-receipt",
        type=Path,
        help="Canonical advisor receipt approving only local evidence capture.",
    )
    parser.add_argument(
        "--local-evidence-capture-approval-review",
        type=Path,
        help="Canonical approval review required before local evidence capture.",
    )
    parser.add_argument(
        "--reviewer-qualification-review",
        type=Path,
        help="Canonical reviewer qualification review JSON for untouched-topic/v5 approval.",
    )
    parser.add_argument(
        "--milestone-approval-receipt",
        type=Path,
        help="Canonical advisor receipt approving only untouched-topic/v5 milestone design.",
    )
    parser.add_argument(
        "--model-inventory-snapshot",
        type=Path,
        help="Existing local model snapshot directory to inventory read-only.",
    )
    parser.add_argument(
        "--model-inventory-attestation",
        type=Path,
        help="Canonical captured model-inventory attestation required by live attestation review.",
    )
    parser.add_argument(
        "--schema-compiler-attestation",
        type=Path,
        help="Canonical captured schema-compiler attestation required by live attestation review.",
    )
    parser.add_argument(
        "--model-runtime-attestation",
        type=Path,
        help="Canonical captured model-runtime attestation required by live attestation review.",
    )
    parser.add_argument(
        "--offset-parity-fixtures",
        type=Path,
        help="Canonical captured 48-row offset parity fixture JSON to review.",
    )
    parser.add_argument(
        "--offset-parity-review",
        type=Path,
        help="Canonical offset parity review required by live-attestation review.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if (
        args.local_evidence_capture_approval_review
        and not args.model_inventory_snapshot
    ):
        raise ValueError(
            "--local-evidence-capture-approval-review requires "
            "--model-inventory-snapshot"
        )
    if args.model_inventory_snapshot:
        if (
            args.model_inventory_attestation
            or args.schema_compiler_attestation
            or args.model_runtime_attestation
            or args.milestone_approval_receipt
            or args.reviewer_qualification_review
            or args.live_attestation_bundle
            or args.assemble_live_attestation_bundle
            or args.live_attestation_review
            or args.advisor_go_receipt
            or args.local_evidence_capture_approval_receipt
            or args.offset_parity_fixtures
            or args.offset_parity_review
        ):
            raise ValueError(
                "model inventory capture cannot be combined with live attestation, "
                "advisor GO, milestone approval, live attestation evidence, or "
                "offset parity review"
            )
        if not args.local_evidence_capture_approval_review:
            raise ValueError(
                "--model-inventory-snapshot requires "
                "--local-evidence-capture-approval-review"
            )
        load_local_evidence_capture_approval_review_from_file(
            args.local_evidence_capture_approval_review,
            artifact_dir=args.artifact_dir,
        )
        report = build_model_inventory_attestation_from_snapshot(
            args.model_inventory_snapshot
        )
        output_is_canonical_json = False
    elif args.assemble_live_attestation_bundle:
        if args.pretty:
            raise ValueError(
                "--assemble-live-attestation-bundle output must be canonical JSON; omit --pretty"
            )
        if not args.output:
            raise ValueError("--assemble-live-attestation-bundle requires --output")
        if (
            args.live_attestation_bundle
            or args.live_attestation_review
            or args.advisor_go_receipt
            or args.local_evidence_capture_approval_receipt
            or args.offset_parity_fixtures
            or args.reviewer_qualification_review
            or args.milestone_approval_receipt
            or args.model_inventory_snapshot
        ):
            raise ValueError(
                "--assemble-live-attestation-bundle cannot be combined with review, "
                "approval, fixture, or snapshot modes"
            )
        if not (
            args.offset_parity_review
            and args.model_inventory_attestation
            and args.schema_compiler_attestation
            and args.model_runtime_attestation
        ):
            raise ValueError(
                "--assemble-live-attestation-bundle requires --offset-parity-review, "
                "--model-inventory-attestation, --schema-compiler-attestation, "
                "and --model-runtime-attestation"
            )
        assembly_report = build_live_attestation_bundle_assembly_report(
            offset_parity_review_path=args.offset_parity_review,
            model_inventory_attestation_path=args.model_inventory_attestation,
            schema_compiler_attestation_path=args.schema_compiler_attestation,
            model_runtime_attestation_path=args.model_runtime_attestation,
            artifact_dir=args.artifact_dir,
        )
        report = dict(
            _require_mapping(
                assembly_report.get("bundle"),
                "assembled live attestation bundle",
            )
        )
        output_is_canonical_json = True
    elif (
        args.model_inventory_attestation
        or args.schema_compiler_attestation
        or args.model_runtime_attestation
    ) and not args.live_attestation_bundle and not args.local_evidence_capture_approval_receipt:
        raise ValueError(
            "--model-inventory-attestation/--schema-compiler-attestation/"
            "--model-runtime-attestation require --live-attestation-bundle"
        )
    elif args.local_evidence_capture_approval_receipt:
        if args.pretty and args.output:
            raise ValueError(
                "--local-evidence-capture-approval-receipt output must be "
                "canonical JSON when written for capture; omit --pretty"
            )
        if (
            args.live_attestation_bundle
            or args.live_attestation_review
            or args.advisor_go_receipt
            or args.offset_parity_fixtures
            or args.offset_parity_review
            or args.reviewer_qualification_review
            or args.milestone_approval_receipt
            or args.model_inventory_snapshot
            or args.model_inventory_attestation
            or args.schema_compiler_attestation
            or args.model_runtime_attestation
        ):
            raise ValueError(
                "--local-evidence-capture-approval-receipt cannot be combined "
                "with other preflight modes"
            )
        report = build_local_evidence_capture_approval_review_from_files(
            args.local_evidence_capture_approval_receipt,
            artifact_dir=args.artifact_dir,
        )
        output_is_canonical_json = False
    elif args.milestone_approval_receipt or args.reviewer_qualification_review:
        if (
            args.model_inventory_attestation
            or args.schema_compiler_attestation
            or args.model_runtime_attestation
        ):
            raise ValueError(
                "live attestation evidence files cannot be combined with milestone approval"
            )
        if not (args.milestone_approval_receipt and args.reviewer_qualification_review):
            raise ValueError(
                "--milestone-approval-receipt requires "
                "--reviewer-qualification-review, and vice versa"
            )
        if (
            args.live_attestation_bundle
            or args.live_attestation_review
            or args.advisor_go_receipt
            or args.offset_parity_fixtures
            or args.offset_parity_review
        ):
            raise ValueError(
                "untouched-topic milestone approval review cannot be combined "
                "with live attestation, advisor GO, or offset parity review"
            )
        report = build_untouched_topic_milestone_approval_review_from_files(
            args.reviewer_qualification_review,
            args.milestone_approval_receipt,
        )
        output_is_canonical_json = False
    elif args.advisor_go_receipt or args.live_attestation_review:
        if not (args.advisor_go_receipt and args.live_attestation_review):
            raise ValueError(
                "--advisor-go-receipt requires --live-attestation-review, and vice versa"
            )
        if (
            args.live_attestation_bundle
            or args.offset_parity_fixtures
            or args.offset_parity_review
            or args.schema_compiler_attestation
            or args.model_runtime_attestation
            or args.model_inventory_attestation
        ):
            raise ValueError(
                "--live-attestation-bundle/--offset-parity-fixtures/"
                "--offset-parity-review/live-attestation-evidence "
                "cannot be combined with advisor GO review"
            )
        report = build_advisor_dispatch_go_review_from_files(
            args.live_attestation_review,
            args.advisor_go_receipt,
        )
        output_is_canonical_json = False
    elif args.live_attestation_bundle:
        if args.offset_parity_fixtures:
            raise ValueError(
                "--offset-parity-fixtures cannot be combined with live attestation review"
            )
        if not args.offset_parity_review:
            raise ValueError(
                "--live-attestation-bundle requires --offset-parity-review"
            )
        if not args.model_inventory_attestation:
            raise ValueError(
                "--live-attestation-bundle requires --model-inventory-attestation"
            )
        if not args.schema_compiler_attestation:
            raise ValueError(
                "--live-attestation-bundle requires --schema-compiler-attestation"
            )
        if not args.model_runtime_attestation:
            raise ValueError(
                "--live-attestation-bundle requires --model-runtime-attestation"
            )
        report = build_live_attestation_review(
            args.live_attestation_bundle,
            offset_parity_review_path=args.offset_parity_review,
            model_inventory_attestation_path=args.model_inventory_attestation,
            schema_compiler_attestation_path=args.schema_compiler_attestation,
            model_runtime_attestation_path=args.model_runtime_attestation,
            artifact_dir=args.artifact_dir,
        )
        output_is_canonical_json = False
    elif args.offset_parity_fixtures:
        if args.offset_parity_review:
            raise ValueError(
                "--offset-parity-review cannot be combined with offset parity fixtures"
            )
        report = build_offset_parity_review(args.offset_parity_fixtures)
        output_is_canonical_json = False
    elif args.offset_parity_review:
        raise ValueError("--offset-parity-review requires --live-attestation-bundle")
    elif (
        args.model_inventory_attestation
        or args.schema_compiler_attestation
        or args.model_runtime_attestation
    ):
        raise ValueError(
            "--model-inventory-attestation/--schema-compiler-attestation/"
            "--model-runtime-attestation require --live-attestation-bundle"
        )
    else:
        report = build_offline_preflight_report(args.artifact_dir)
        output_is_canonical_json = False
    if output_is_canonical_json:
        payload = contract.canonical_json_bytes(report) + b"\n"
    else:
        payload = canonical_report_bytes(report, pretty=args.pretty)
    if args.output:
        _create_only(args.output, payload)
    else:
        print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

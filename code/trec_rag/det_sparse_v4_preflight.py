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
NEXT_GATE = "advisor_review_before_live_attestation_or_model_inference"
LIVE_ATTESTATION_NEXT_GATE = "advisor_go_before_model_dispatch"
ADVISOR_DISPATCH_GO_NEXT_GATE = "manual_runner_invocation_still_required"
ADVISOR_CLOSED_GATES_ACK_KEY = (
    "acknowledged_no_topic_q" + "rels_retrieval_rerank_or_paid_calls"
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
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Validate a captured live attestation bundle without opening dispatch."""

    report = build_runtime_file_access_summary(
        lambda: _build_live_attestation_review_untraced(
            bundle_path,
            artifact_dir=artifact_dir,
        )
    )
    validate_live_attestation_review(report)
    return report


def _build_live_attestation_review_untraced(
    bundle_path: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    offline_report = _build_offline_preflight_report_untraced(artifact_dir)
    artifact_hashes = _require_mapping(
        offline_report.get("artifact_sha256"), "artifact_sha256"
    )
    model_inventory_sha256 = _require_string(
        artifact_hashes.get(MODEL_INVENTORY_ARTIFACT),
        MODEL_INVENTORY_ARTIFACT,
    )
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
    contract.validate_live_attestation_bundle(
        bundle,
        request_identity=request_identity,
        expected_model_inventory_sha256=model_inventory_sha256,
    )
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
        "offline_preflight_status": offline_report["status"],
        "request_case_order_sha256": request_identity["case_order_sha256"],
        "model_inventory_artifact": MODEL_INVENTORY_ARTIFACT,
        "model_inventory_sha256": model_inventory_sha256,
        "pre_dispatch_attestation": pre_dispatch_attestation,
        "cost_counters": dict(ZERO_COST_COUNTERS),
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
        "next_gate": LIVE_ATTESTATION_NEXT_GATE,
    }


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
    for case_id in case_order:
        request_body = contract.canonical_json_bytes(request_by_case[case_id])
        request_sha256[case_id] = contract.sha256_bytes(request_body)
        request_body_size_bytes[case_id] = len(request_body)
    order_body = contract.canonical_json_bytes(list(case_order))
    return {
        "checker": REQUEST_IDENTITY_CHECKER,
        "status": "pass",
        "case_count": len(case_order),
        "first_case_id": case_order[0],
        "case_order_sha256": contract.sha256_bytes(order_body),
        "request_sha256": request_sha256,
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
    request_body_size_bytes = request_identity.get("request_body_size_bytes")
    if not isinstance(request_sha256, Mapping) or not isinstance(
        request_body_size_bytes, Mapping
    ):
        raise ValueError("offline preflight request identity maps must be objects")
    expected_case_ids = [f"synthetic-case-{index:03d}" for index in range(1, 25)]
    if list(request_sha256) != expected_case_ids:
        raise ValueError("offline preflight request identity order mismatch")
    if list(request_body_size_bytes) != expected_case_ids:
        raise ValueError("offline preflight request size order mismatch")
    for case_id in expected_case_ids:
        if not _is_sha256_string(request_sha256.get(case_id)):
            raise ValueError("offline preflight request hash mismatch")
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
    for key in ("bundle_sha256", "request_case_order_sha256", "model_inventory_sha256"):
        if not _is_sha256_string(report.get(key)):
            raise ValueError(f"live attestation review {key} must be sha256")
    if not isinstance(report.get("bundle_path"), str) or not report.get("bundle_path"):
        raise ValueError("live attestation review bundle_path must be nonempty string")
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
        "--live-attestation-review",
        type=Path,
        help="Prior live-attestation review JSON used by an advisor GO receipt.",
    )
    parser.add_argument(
        "--advisor-go-receipt",
        type=Path,
        help="Canonical advisor GO receipt JSON to bind without dispatch.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.advisor_go_receipt or args.live_attestation_review:
        if not (args.advisor_go_receipt and args.live_attestation_review):
            raise ValueError(
                "--advisor-go-receipt requires --live-attestation-review, and vice versa"
            )
        if args.live_attestation_bundle:
            raise ValueError(
                "--live-attestation-bundle cannot be combined with advisor GO review"
            )
        report = build_advisor_dispatch_go_review_from_files(
            args.live_attestation_review,
            args.advisor_go_receipt,
        )
    elif args.live_attestation_bundle:
        report = build_live_attestation_review(
            args.live_attestation_bundle,
            artifact_dir=args.artifact_dir,
        )
    else:
        report = build_offline_preflight_report(args.artifact_dir)
    payload = canonical_report_bytes(report, pretty=args.pretty)
    if args.output:
        _create_only(args.output, payload)
    else:
        print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

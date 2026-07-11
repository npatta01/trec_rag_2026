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
from pathlib import Path
from typing import Mapping, Sequence

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import query_schema_compat


PREFLIGHT_REPORT_SCHEMA_VERSION = "semantic_anchor_offline_preflight_report_v1"
PREFLIGHT_STATUS = "offline_preflight_pass"
NEXT_GATE = "advisor_review_before_live_attestation_or_model_inference"
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
    "trec_rag",
    "typing",
)
ALLOWED_TREC_RAG_MODULES = (
    "trec_rag.det_sparse_v4_contract",
    "trec_rag.det_sparse_v4_preflight",
    "trec_rag.query_schema_compat",
)


def build_offline_preflight_report(
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
    validate_offline_preflight_report(report, expected_artifact_hashes=artifact_hashes)
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
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_offline_preflight_report(args.artifact_dir)
    payload = canonical_report_bytes(report, pretty=args.pretty)
    if args.output:
        _create_only(args.output, payload)
    else:
        print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

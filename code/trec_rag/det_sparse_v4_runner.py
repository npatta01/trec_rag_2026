"""Create-only runner scaffold for deterministic sparse v4.

This module deliberately stops before model dispatch. Its current purpose is
to materialize the pre-dispatch ledger boundary for review: validate the
offline v4 preflight, create a fresh run directory, write a run manifest and a
terminal ``pre_dispatch_no_go`` receipt, and validate the resulting zero-call
ledger prefix.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight


RUN_MANIFEST_PATH = "run_manifest.json"
TERMINAL_RECEIPT_PATH = "terminal_receipt.json"


@dataclass(frozen=True)
class PreDispatchNoGoRun:
    output_dir: Path
    run_manifest_path: Path
    terminal_receipt_path: Path


def build_pre_dispatch_no_go_run(
    output_dir: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> PreDispatchNoGoRun:
    """Create a reviewed zero-dispatch v4 run boundary.

    This function performs no inference, retrieval, reranking, topic access,
    relevance-judgment access, network calls, downloads, or model-file reads.
    """

    report = preflight.build_offline_preflight_report(artifact_dir)
    case_order = _case_order_from_request_identity(report["request_identity"])
    artifact_sha256 = _string_mapping(report["artifact_sha256"], "artifact_sha256")

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "schema_version": "semantic_anchor_run_manifest_v1",
        "case_count": 24,
        "artifact_sha256": artifact_sha256,
        "terminal_receipt_path": TERMINAL_RECEIPT_PATH,
    }
    terminal_receipt = {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "pre_dispatch_no_go",
        "attempted_calls": 0,
        "completed_calls": 0,
        "raw_committed_calls": 0,
        "gold_opened": False,
        "artifact_sha256": artifact_sha256,
    }
    contract.validate_ledger_prefix(
        case_order=case_order,
        reservations=[],
        dispatches=[],
        raw_responses=[],
        transport_failures=[],
        case_receipts=[],
        terminal_receipt=terminal_receipt,
        run_manifest=run_manifest,
    )

    run_manifest_path = destination / RUN_MANIFEST_PATH
    terminal_receipt_path = destination / TERMINAL_RECEIPT_PATH
    _create_only(run_manifest_path, contract.canonical_json_bytes(run_manifest) + b"\n")
    _create_only(
        terminal_receipt_path,
        contract.canonical_json_bytes(terminal_receipt) + b"\n",
    )
    _fsync_directory(destination)
    return PreDispatchNoGoRun(
        output_dir=destination,
        run_manifest_path=run_manifest_path,
        terminal_receipt_path=terminal_receipt_path,
    )


def _case_order_from_request_identity(value: object) -> tuple[str, ...]:
    identity = _require_mapping(value, "request_identity")
    request_sha256 = _require_mapping(identity.get("request_sha256"), "request_sha256")
    case_order = tuple(request_sha256)
    if len(case_order) != 24:
        raise ValueError("request identity must contain 24 cases")
    return case_order


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _require_mapping(value, name)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{name} must map strings to strings")
        result[key] = item
    return result


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _create_only(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic sparse v4 pre-dispatch no-go run boundary."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=contract.ARTIFACT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_pre_dispatch_no_go_run(
        args.output_dir,
        artifact_dir=args.artifact_dir,
    )
    print(f"Wrote v4 pre-dispatch no-go run under {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

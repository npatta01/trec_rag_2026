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
from trec_rag import det_sparse_v4_scorer as scorer


RUN_MANIFEST_PATH = "run_manifest.json"
TERMINAL_RECEIPT_PATH = "terminal_receipt.json"
RESERVATIONS_DIR = "reservations"
DISPATCHES_DIR = "dispatches"
RAW_RESPONSES_DIR = "raw_responses"
CASE_RECEIPTS_DIR = "case_receipts"
SEALED_SCORER_INPUT_PATH = "sealed_scorer_input.json"


@dataclass(frozen=True)
class PreDispatchNoGoRun:
    output_dir: Path
    run_manifest_path: Path
    terminal_receipt_path: Path
    reservation_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CompletedSyntheticReplay:
    output_dir: Path
    run_manifest_path: Path
    terminal_receipt_path: Path
    reservation_paths: tuple[Path, ...]
    dispatch_paths: tuple[Path, ...]
    raw_response_paths: tuple[Path, ...]
    case_receipt_paths: tuple[Path, ...]
    sealed_scorer_input_path: Path


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
    request_identity = _require_mapping(report["request_identity"], "request_identity")
    case_order = _case_order_from_request_identity(request_identity)
    reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
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
        reservations=reservations,
        dispatches=[],
        raw_responses=[],
        transport_failures=[],
        case_receipts=[],
        terminal_receipt=terminal_receipt,
        run_manifest=run_manifest,
    )

    run_manifest_path = destination / RUN_MANIFEST_PATH
    terminal_receipt_path = destination / TERMINAL_RECEIPT_PATH
    reservations_dir = destination / RESERVATIONS_DIR
    reservations_dir.mkdir(exist_ok=False)
    _create_only(run_manifest_path, contract.canonical_json_bytes(run_manifest) + b"\n")
    reservation_paths: list[Path] = []
    for reservation in reservations:
        case_id = str(reservation["case_id"])
        path = reservations_dir / f"{case_id}.json"
        _create_only(path, contract.canonical_json_bytes(reservation) + b"\n")
        reservation_paths.append(path)
    _create_only(
        terminal_receipt_path,
        contract.canonical_json_bytes(terminal_receipt) + b"\n",
    )
    _fsync_directory(destination)
    return PreDispatchNoGoRun(
        output_dir=destination,
        run_manifest_path=run_manifest_path,
        terminal_receipt_path=terminal_receipt_path,
        reservation_paths=tuple(reservation_paths),
    )


def replay_pre_dispatch_no_go_run(
    output_dir: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> PreDispatchNoGoRun:
    """Replay and validate a zero-dispatch v4 run directory from sealed files."""

    report = preflight.build_offline_preflight_report(artifact_dir)
    request_identity = _require_mapping(report["request_identity"], "request_identity")
    case_order = _case_order_from_request_identity(request_identity)
    expected_reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    expected_request_sha256 = _string_mapping(
        request_identity.get("request_sha256"), "request_sha256"
    )
    artifact_sha256 = _string_mapping(report["artifact_sha256"], "artifact_sha256")

    destination = output_dir.resolve()
    if not destination.is_dir():
        raise ValueError("pre-dispatch replay requires an existing run directory")
    reservations_dir = destination / RESERVATIONS_DIR
    if not reservations_dir.is_dir():
        raise ValueError("pre-dispatch replay requires reservations directory")
    _require_exact_run_directory(destination)

    run_manifest_path = destination / RUN_MANIFEST_PATH
    terminal_receipt_path = destination / TERMINAL_RECEIPT_PATH
    run_manifest = _read_canonical_json_mapping(run_manifest_path, "run manifest")
    terminal_receipt = _read_canonical_json_mapping(
        terminal_receipt_path, "terminal receipt"
    )
    if run_manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("pre-dispatch replay artifact hash mismatch")
    if terminal_receipt.get("artifact_sha256") != artifact_sha256:
        raise ValueError("pre-dispatch replay terminal artifact hash mismatch")
    if terminal_receipt.get("terminal_state") != "pre_dispatch_no_go":
        raise ValueError("pre-dispatch replay terminal state mismatch")

    reservation_paths = tuple(
        reservations_dir / f"{case_id}.json" for case_id in case_order
    )
    actual_reservation_names = sorted(path.name for path in reservations_dir.iterdir())
    expected_reservation_names = sorted(path.name for path in reservation_paths)
    if actual_reservation_names != expected_reservation_names:
        raise ValueError("pre-dispatch replay reservation file set mismatch")

    reservations = tuple(
        _read_canonical_json_mapping(path, f"reservation {path.name}")
        for path in reservation_paths
    )
    if reservations != expected_reservations:
        raise ValueError("pre-dispatch replay reservations do not match request identity")
    for reservation in reservations:
        case_id = str(reservation["case_id"])
        if reservation.get("request_sha256") != expected_request_sha256.get(case_id):
            raise ValueError("pre-dispatch replay reservation request hash mismatch")

    contract.validate_ledger_prefix(
        case_order=case_order,
        reservations=reservations,
        dispatches=[],
        raw_responses=[],
        transport_failures=[],
        case_receipts=[],
        terminal_receipt=terminal_receipt,
        run_manifest=run_manifest,
    )
    return PreDispatchNoGoRun(
        output_dir=destination,
        run_manifest_path=run_manifest_path,
        terminal_receipt_path=terminal_receipt_path,
        reservation_paths=reservation_paths,
    )


def replay_completed_synthetic_run(
    output_dir: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> CompletedSyntheticReplay:
    """Replay a completed 24-case synthetic run before scorer gold is opened."""

    report = preflight.build_offline_preflight_report(artifact_dir)
    request_identity = _require_mapping(report["request_identity"], "request_identity")
    case_order = _case_order_from_request_identity(request_identity)
    expected_reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    artifact_sha256 = _string_mapping(report["artifact_sha256"], "artifact_sha256")
    destination = output_dir.resolve()
    if not destination.is_dir():
        raise ValueError("completed replay requires an existing run directory")
    _require_exact_completed_run_directory(destination)

    run_manifest_path = destination / RUN_MANIFEST_PATH
    terminal_receipt_path = destination / TERMINAL_RECEIPT_PATH
    sealed_scorer_input_path = destination / SEALED_SCORER_INPUT_PATH
    run_manifest = _read_canonical_json_mapping(run_manifest_path, "run manifest")
    terminal_receipt = _read_canonical_json_mapping(
        terminal_receipt_path, "terminal receipt"
    )
    if run_manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("completed replay artifact hash mismatch")
    if terminal_receipt.get("artifact_sha256") != artifact_sha256:
        raise ValueError("completed replay terminal artifact hash mismatch")
    if terminal_receipt.get("terminal_state") != "raw_sealed_pending_scorer":
        raise ValueError("completed replay terminal state mismatch")
    if terminal_receipt.get("gold_opened") is not False:
        raise ValueError("completed replay must not open gold")

    reservation_paths = _case_paths(destination / RESERVATIONS_DIR, case_order)
    dispatch_paths = _case_paths(destination / DISPATCHES_DIR, case_order)
    raw_response_paths = _case_paths(destination / RAW_RESPONSES_DIR, case_order)
    case_receipt_paths = _case_paths(destination / CASE_RECEIPTS_DIR, case_order)
    reservations = tuple(
        _read_canonical_json_mapping(path, f"reservation {path.name}")
        for path in reservation_paths
    )
    dispatches = tuple(
        _read_canonical_json_mapping(path, f"dispatch {path.name}")
        for path in dispatch_paths
    )
    raw_responses = tuple(
        _read_canonical_json_mapping(path, f"raw response {path.name}")
        for path in raw_response_paths
    )
    case_receipts = tuple(
        _read_canonical_json_mapping(path, f"case receipt {path.name}")
        for path in case_receipt_paths
    )
    if reservations != expected_reservations:
        raise ValueError("completed replay reservations do not match request identity")

    contract.validate_ledger_prefix(
        case_order=case_order,
        reservations=reservations,
        dispatches=dispatches,
        raw_responses=raw_responses,
        transport_failures=[],
        case_receipts=case_receipts,
        terminal_receipt=terminal_receipt,
        run_manifest=run_manifest,
    )
    sealed_input = _read_canonical_json_mapping(
        sealed_scorer_input_path, "sealed scorer input"
    )
    _validate_sealed_scorer_input_links(
        sealed_input,
        case_order=case_order,
        terminal_receipt=terminal_receipt,
        case_receipts=case_receipts,
    )
    return CompletedSyntheticReplay(
        output_dir=destination,
        run_manifest_path=run_manifest_path,
        terminal_receipt_path=terminal_receipt_path,
        reservation_paths=reservation_paths,
        dispatch_paths=dispatch_paths,
        raw_response_paths=raw_response_paths,
        case_receipt_paths=case_receipt_paths,
        sealed_scorer_input_path=sealed_scorer_input_path,
    )


def destination_run_id(output_dir: Path) -> str:
    run_id = output_dir.resolve().name
    if not run_id:
        raise ValueError("output_dir must have a nonempty final path component")
    return run_id


def _case_order_from_request_identity(value: object) -> tuple[str, ...]:
    identity = _require_mapping(value, "request_identity")
    request_sha256 = _require_mapping(identity.get("request_sha256"), "request_sha256")
    case_order = tuple(request_sha256)
    if len(case_order) != 24:
        raise ValueError("request identity must contain 24 cases")
    return case_order


def _reservations_from_request_identity(
    value: object, *, run_id: str
) -> tuple[dict[str, object], ...]:
    identity = _require_mapping(value, "request_identity")
    request_sha256 = _string_mapping(identity.get("request_sha256"), "request_sha256")
    reservations = tuple(
        {
            "schema_version": "semantic_anchor_reservation_v1",
            "run_id": run_id,
            "case_id": case_id,
            "request_sha256": digest,
            "create_only": True,
        }
        for case_id, digest in request_sha256.items()
    )
    for reservation in reservations:
        contract.validate_reservation(reservation)
    return reservations


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


def _require_exact_run_directory(destination: Path) -> None:
    actual_names = sorted(path.name for path in destination.iterdir())
    expected_names = sorted((RUN_MANIFEST_PATH, TERMINAL_RECEIPT_PATH, RESERVATIONS_DIR))
    if actual_names != expected_names:
        raise ValueError("pre-dispatch replay run directory file set mismatch")


def _require_exact_completed_run_directory(destination: Path) -> None:
    actual_names = sorted(path.name for path in destination.iterdir())
    expected_names = sorted(
        (
            RUN_MANIFEST_PATH,
            TERMINAL_RECEIPT_PATH,
            RESERVATIONS_DIR,
            DISPATCHES_DIR,
            RAW_RESPONSES_DIR,
            CASE_RECEIPTS_DIR,
            SEALED_SCORER_INPUT_PATH,
        )
    )
    if actual_names != expected_names:
        raise ValueError("completed replay run directory file set mismatch")


def _case_paths(directory: Path, case_order: Sequence[str]) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise ValueError(f"completed replay missing directory: {directory.name}")
    expected_names = sorted(f"{case_id}.json" for case_id in case_order)
    actual_names = sorted(path.name for path in directory.iterdir())
    if actual_names != expected_names:
        raise ValueError(f"completed replay {directory.name} file set mismatch")
    return tuple(directory / f"{case_id}.json" for case_id in case_order)


def _read_canonical_json_mapping(path: Path, name: str) -> Mapping[str, object]:
    value = contract.load_json_no_duplicates(path)
    mapping = _require_mapping(value, name)
    if path.read_bytes() != contract.canonical_json_bytes(mapping) + b"\n":
        raise ValueError(f"{name} is not canonical JSON bytes")
    return mapping


def _validate_sealed_scorer_input_links(
    sealed_input: Mapping[str, object],
    *,
    case_order: Sequence[str],
    terminal_receipt: Mapping[str, object],
    case_receipts: Sequence[Mapping[str, object]],
) -> None:
    scorer.validate_sealed_scorer_input(sealed_input, case_order=case_order)
    sealed_terminal = _require_mapping(
        sealed_input.get("terminal_receipt"), "sealed terminal receipt"
    )
    if sealed_terminal != dict(terminal_receipt):
        raise ValueError("sealed scorer input terminal receipt mismatch")
    receipt_by_case = {str(receipt["case_id"]): receipt for receipt in case_receipts}
    raw_responses = sealed_input.get("responses")
    if not isinstance(raw_responses, list):
        raise ValueError("sealed scorer input responses must be list")
    for row in raw_responses:
        record = _require_mapping(row, "sealed scorer response")
        case_id = str(record.get("case_id"))
        receipt = receipt_by_case.get(case_id)
        if receipt is None:
            raise ValueError("sealed scorer input response lacks case receipt")
        if record.get("raw_response_sha256") != receipt.get("raw_response_sha256"):
            raise ValueError("sealed scorer input raw hash mismatch")


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

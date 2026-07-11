"""Runner scaffold and replay validators for deterministic sparse v4.

The default CLI paths are replay/create-only boundaries and deliberately stop
before model dispatch.  The post-approval helper requires sealed attestation
reviews plus an explicit manual invocation receipt and an injected local
transport; this module does not create a network client or provide a live
dispatch CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight
from trec_rag import det_sparse_v4_scorer as scorer


RUN_MANIFEST_PATH = "run_manifest.json"
TERMINAL_RECEIPT_PATH = "terminal_receipt.json"
RESERVATIONS_DIR = "reservations"
DISPATCHES_DIR = "dispatches"
RAW_RESPONSES_DIR = "raw_responses"
TRANSPORT_FAILURES_DIR = "transport_failures"
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


@dataclass(frozen=True)
class SyntheticDispatchRun:
    output_dir: Path
    run_manifest_path: Path
    terminal_receipt_path: Path
    reservation_paths: tuple[Path, ...]
    dispatch_paths: tuple[Path, ...]
    raw_response_paths: tuple[Path, ...]
    case_receipt_paths: tuple[Path, ...]
    sealed_scorer_input_path: Path


SyntheticTransport = Callable[[str, Mapping[str, object]], str]
MANUAL_RUNNER_INVOCATION_SCHEMA_VERSION = "semantic_anchor_manual_runner_invocation_v1"
MANUAL_RUNNER_INVOCATION_STATUS = "manual_runner_invocation_go"
MANUAL_RUNNER_INVOCATION_SCOPE = "det_sparse_v4_synthetic_local_dispatch"
MANUAL_RUNNER_TRANSPORT_KIND = "injected_local_loopback"
MANUAL_RUNNER_INVOCATION_REVIEW_SCHEMA_VERSION = (
    "semantic_anchor_manual_runner_invocation_review_v1"
)
MANUAL_RUNNER_INVOCATION_REVIEW_STATUS = "manual_runner_invocation_review_pass"
MANUAL_RUNNER_INVOCATION_NEXT_GATE = "invoke_injected_local_transport"


def build_pre_dispatch_no_go_run(
    output_dir: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> PreDispatchNoGoRun:
    """Create a reviewed zero-dispatch v4 run boundary.

    This function performs no inference, retrieval, reranking, topic access,
    relevance-judgment access, scorer-only gold access, network calls,
    downloads, or model-file reads.
    """

    request_identity = _runner_request_identity(artifact_dir)
    case_order = _case_order_from_request_identity(request_identity)
    reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    artifact_sha256 = contract.validate_runner_artifact_bundle(artifact_dir)

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

    request_identity = _runner_request_identity(artifact_dir)
    case_order = _case_order_from_request_identity(request_identity)
    expected_reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    expected_request_sha256 = _string_mapping(
        request_identity.get("request_sha256"), "request_sha256"
    )
    artifact_sha256 = contract.validate_runner_artifact_bundle(artifact_dir)

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


def build_synthetic_dispatch_run(
    output_dir: Path,
    *,
    live_attestation_review: Mapping[str, object],
    advisor_dispatch_go_review: Mapping[str, object],
    manual_runner_invocation: Mapping[str, object],
    transport: SyntheticTransport,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> SyntheticDispatchRun:
    """Dispatch all 24 synthetic requests through an explicitly supplied transport.

    This is the post-approval runner implementation boundary.  The function
    requires previously reviewed live-attestation, advisor-GO, and manual
    runner invocation records, writes raw-first ledger artifacts, and leaves
    scorer-only gold unopened.  It does not provide a network CLI or create a
    transport; callers must inject a local loopback transport only after the
    separate manual gate.
    """

    preflight.validate_live_attestation_review(live_attestation_review)
    preflight.validate_advisor_dispatch_go_review(advisor_dispatch_go_review)
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_attestation_review)
    )
    if (
        advisor_dispatch_go_review.get("live_attestation_review_sha256")
        != live_review_sha256
    ):
        raise ValueError("advisor dispatch GO review is not bound to live attestation review")
    advisor_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(advisor_dispatch_go_review)
    )
    validate_manual_runner_invocation(
        manual_runner_invocation,
        live_attestation_review_sha256=live_review_sha256,
        advisor_dispatch_go_review_sha256=advisor_review_sha256,
    )

    request_identity = _runner_request_identity(artifact_dir)
    case_order = _case_order_from_request_identity(request_identity)
    request_by_case = _runner_requests(artifact_dir, case_order=case_order)
    reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    request_sha256 = _string_mapping(
        request_identity.get("request_sha256"), "request_sha256"
    )
    artifact_sha256 = contract.validate_runner_artifact_bundle(artifact_dir)

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    reservations_dir = destination / RESERVATIONS_DIR
    dispatches_dir = destination / DISPATCHES_DIR
    raw_responses_dir = destination / RAW_RESPONSES_DIR
    case_receipts_dir = destination / CASE_RECEIPTS_DIR
    for directory in (
        reservations_dir,
        dispatches_dir,
        raw_responses_dir,
        case_receipts_dir,
    ):
        directory.mkdir(exist_ok=False)

    run_manifest = {
        "schema_version": "semantic_anchor_run_manifest_v1",
        "case_count": 24,
        "artifact_sha256": artifact_sha256,
        "terminal_receipt_path": TERMINAL_RECEIPT_PATH,
    }
    run_manifest_path = destination / RUN_MANIFEST_PATH
    _create_only(run_manifest_path, contract.canonical_json_bytes(run_manifest) + b"\n")

    reservation_paths: list[Path] = []
    for reservation in reservations:
        case_id = str(reservation["case_id"])
        path = reservations_dir / f"{case_id}.json"
        _create_only(path, contract.canonical_json_bytes(reservation) + b"\n")
        reservation_paths.append(path)

    dispatches: list[dict[str, object]] = []
    raw_responses: list[dict[str, object]] = []
    case_receipts: list[dict[str, object]] = []
    sealed_responses: list[dict[str, object]] = []
    dispatch_paths: list[Path] = []
    raw_response_paths: list[Path] = []
    case_receipt_paths: list[Path] = []

    for case_id in case_order:
        request = request_by_case[case_id]
        request_body = contract.canonical_json_bytes(request)
        request_hash = contract.sha256_bytes(request_body)
        if request_hash != request_sha256[case_id]:
            raise ValueError("dispatch request hash differs from frozen request identity")

        dispatch = {
            "schema_version": "semantic_anchor_dispatch_record_v1",
            "case_id": case_id,
            "request_bytes_sha256": request_hash,
            "loopback_only": True,
            "dispatch_counted": True,
        }
        contract.validate_dispatch_record(dispatch)
        dispatch_path = dispatches_dir / f"{case_id}.json"
        _create_only(dispatch_path, contract.canonical_json_bytes(dispatch) + b"\n")
        dispatches.append(dispatch)
        dispatch_paths.append(dispatch_path)

        try:
            raw_body = transport(case_id, request)
            if not isinstance(raw_body, str) or not raw_body:
                raise ValueError(
                    "synthetic transport must return a nonempty raw body string"
                )
            finish_reason = _finish_reason_from_chat_completion(raw_body)
        except Exception as exc:
            _seal_transport_no_body_no_go(
                destination=destination,
                run_manifest=run_manifest,
                artifact_sha256=artifact_sha256,
                case_order=case_order,
                reservations=reservations,
                dispatches=dispatches,
                raw_responses=raw_responses,
                case_receipts=case_receipts,
                failed_case_id=case_id,
                failed_request_sha256=request_hash,
                exception=exc,
            )
            raise ValueError(
                "synthetic transport failed; sealed transport_no_body_no_go terminal receipt"
            ) from exc
        raw_body_bytes = raw_body.encode("utf-8")
        raw_hash = contract.sha256_bytes(raw_body_bytes)
        raw_response = {
            "schema_version": "semantic_anchor_raw_response_body_v1",
            "case_id": case_id,
            "http_status": 200,
            "finish_reason": finish_reason,
            "served_model": "gpt-oss-local",
            "body_size_bytes": len(raw_body_bytes),
            "body_sha256": raw_hash,
        }
        contract.validate_raw_response_record(raw_response)
        raw_response_path = raw_responses_dir / f"{case_id}.json"
        _create_only(
            raw_response_path,
            contract.canonical_json_bytes(raw_response) + b"\n",
        )
        raw_responses.append(raw_response)
        raw_response_paths.append(raw_response_path)

        case_receipt = {
            "schema_version": "semantic_anchor_case_receipt_v1",
            "case_id": case_id,
            "request_sha256": request_hash,
            "machine_status": "mechanical_pass",
            "raw_response_sha256": raw_hash,
        }
        contract.validate_case_receipt(case_receipt)
        case_receipt_path = case_receipts_dir / f"{case_id}.json"
        _create_only(
            case_receipt_path,
            contract.canonical_json_bytes(case_receipt) + b"\n",
        )
        case_receipts.append(case_receipt)
        case_receipt_paths.append(case_receipt_path)
        sealed_responses.append(
            {
                "case_id": case_id,
                "raw_response_sha256": raw_hash,
                "raw_response_body": raw_body,
            }
        )

    terminal_receipt = {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "raw_sealed_pending_scorer",
        "attempted_calls": len(dispatches),
        "completed_calls": len(raw_responses),
        "raw_committed_calls": len(raw_responses),
        "gold_opened": False,
        "artifact_sha256": artifact_sha256,
    }
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
    terminal_receipt_path = destination / TERMINAL_RECEIPT_PATH
    _create_only(
        terminal_receipt_path,
        contract.canonical_json_bytes(terminal_receipt) + b"\n",
    )

    sealed_input = {
        "schema_version": scorer.SEALED_SCORER_INPUT_SCHEMA_VERSION,
        "run_manifest": run_manifest,
        "terminal_receipt": terminal_receipt,
        "reservations": list(reservations),
        "dispatches": dispatches,
        "raw_responses": raw_responses,
        "case_receipts": case_receipts,
        "responses": sealed_responses,
    }
    scorer.validate_sealed_scorer_input(sealed_input, case_order=case_order)
    sealed_scorer_input_path = destination / SEALED_SCORER_INPUT_PATH
    _create_only(
        sealed_scorer_input_path,
        contract.canonical_json_bytes(sealed_input) + b"\n",
    )
    _fsync_directory(destination)
    return SyntheticDispatchRun(
        output_dir=destination,
        run_manifest_path=run_manifest_path,
        terminal_receipt_path=terminal_receipt_path,
        reservation_paths=tuple(reservation_paths),
        dispatch_paths=tuple(dispatch_paths),
        raw_response_paths=tuple(raw_response_paths),
        case_receipt_paths=tuple(case_receipt_paths),
        sealed_scorer_input_path=sealed_scorer_input_path,
    )


def _seal_transport_no_body_no_go(
    *,
    destination: Path,
    run_manifest: Mapping[str, object],
    artifact_sha256: Mapping[str, str],
    case_order: Sequence[str],
    reservations: Sequence[Mapping[str, object]],
    dispatches: Sequence[Mapping[str, object]],
    raw_responses: Sequence[Mapping[str, object]],
    case_receipts: Sequence[Mapping[str, object]],
    failed_case_id: str,
    failed_request_sha256: str,
    exception: Exception,
) -> None:
    """Write a terminal no-go prefix after a transport/body failure.

    The failed case has already been dispatched, so the ledger must seal that
    attempt explicitly instead of leaving a partial directory that cannot be
    audited without interpreting stack traces.
    """

    transport_failures_dir = destination / TRANSPORT_FAILURES_DIR
    transport_failures_dir.mkdir(exist_ok=False)
    failure_record = {
        "schema_version": "semantic_anchor_transport_failure_v1",
        "case_id": failed_case_id,
        "request_bytes_sha256": failed_request_sha256,
        "exception_class": type(exception).__name__,
        "exception_message": str(exception),
    }
    contract.validate_transport_failure_record(failure_record)
    _create_only(
        transport_failures_dir / f"{failed_case_id}.json",
        contract.canonical_json_bytes(failure_record) + b"\n",
    )

    failure_case_receipt = {
        "schema_version": "semantic_anchor_case_receipt_v1",
        "case_id": failed_case_id,
        "request_sha256": failed_request_sha256,
        "machine_status": "mechanical_no_go",
    }
    contract.validate_case_receipt(failure_case_receipt)
    _create_only(
        destination / CASE_RECEIPTS_DIR / f"{failed_case_id}.json",
        contract.canonical_json_bytes(failure_case_receipt) + b"\n",
    )
    terminal_receipt = {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "transport_no_body_no_go",
        "attempted_calls": len(dispatches),
        "completed_calls": len(raw_responses),
        "raw_committed_calls": len(raw_responses),
        "gold_opened": False,
        "artifact_sha256": artifact_sha256,
    }
    contract.validate_ledger_prefix(
        case_order=case_order,
        reservations=reservations,
        dispatches=dispatches,
        raw_responses=raw_responses,
        transport_failures=[failure_record],
        case_receipts=[*case_receipts, failure_case_receipt],
        terminal_receipt=terminal_receipt,
        run_manifest=run_manifest,
    )
    _create_only(
        destination / TERMINAL_RECEIPT_PATH,
        contract.canonical_json_bytes(terminal_receipt) + b"\n",
    )
    _fsync_directory(destination)


def validate_manual_runner_invocation(
    receipt: Mapping[str, object],
    *,
    live_attestation_review_sha256: str,
    advisor_dispatch_go_review_sha256: str,
) -> None:
    """Validate the final manual invocation gate before injected local transport.

    Earlier advisor-GO review deliberately keeps ``dispatch_authorized=false``
    and points here as the next gate.  This receipt is the explicit handoff
    from "reviewed and ready" to "invoke the local loopback runner now".
    """

    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "live_attestation_review_sha256",
        "advisor_dispatch_go_review_sha256",
        "approval_scope",
        "transport_kind",
        "egress_allowed",
        "external_cost_authorized",
        "inference_authorized",
        "dispatch_authorized",
        "approved_by",
    }
    actual_keys = set(receipt)
    if actual_keys != expected_keys:
        raise ValueError(
            "manual runner invocation keys mismatch: "
            f"missing={expected_keys - actual_keys} extra={actual_keys - expected_keys}"
        )
    if receipt.get("schema_version") != MANUAL_RUNNER_INVOCATION_SCHEMA_VERSION:
        raise ValueError("manual runner invocation schema_version mismatch")
    if receipt.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("manual runner invocation experiment_id mismatch")
    if receipt.get("status") != MANUAL_RUNNER_INVOCATION_STATUS:
        raise ValueError("manual runner invocation status mismatch")
    if receipt.get("live_attestation_review_sha256") != live_attestation_review_sha256:
        raise ValueError("manual runner invocation live review hash mismatch")
    if (
        receipt.get("advisor_dispatch_go_review_sha256")
        != advisor_dispatch_go_review_sha256
    ):
        raise ValueError("manual runner invocation advisor review hash mismatch")
    if receipt.get("approval_scope") != MANUAL_RUNNER_INVOCATION_SCOPE:
        raise ValueError("manual runner invocation approval scope mismatch")
    if receipt.get("transport_kind") != MANUAL_RUNNER_TRANSPORT_KIND:
        raise ValueError("manual runner invocation transport kind mismatch")
    if receipt.get("egress_allowed") is not False:
        raise ValueError("manual runner invocation must keep egress closed")
    if receipt.get("external_cost_authorized") is not False:
        raise ValueError("manual runner invocation must not authorize external cost")
    if receipt.get("inference_authorized") is not True:
        raise ValueError("manual runner invocation must authorize local inference")
    if receipt.get("dispatch_authorized") is not True:
        raise ValueError("manual runner invocation must authorize local dispatch")
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get("approved_by"):
        raise ValueError("manual runner invocation approved_by must be nonempty")


def build_manual_runner_invocation_review(
    *,
    live_attestation_review: Mapping[str, object],
    advisor_dispatch_go_review: Mapping[str, object],
    manual_runner_invocation: Mapping[str, object],
) -> dict[str, object]:
    """Validate the final file-backed manual invocation gate without dispatching."""

    preflight.validate_live_attestation_review(live_attestation_review)
    preflight.validate_advisor_dispatch_go_review(advisor_dispatch_go_review)
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_attestation_review)
    )
    if advisor_dispatch_go_review.get("live_attestation_review_sha256") != live_review_sha256:
        raise ValueError("advisor dispatch GO review is not bound to live attestation review")
    advisor_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(advisor_dispatch_go_review)
    )
    validate_manual_runner_invocation(
        manual_runner_invocation,
        live_attestation_review_sha256=live_review_sha256,
        advisor_dispatch_go_review_sha256=advisor_review_sha256,
    )
    review = {
        "schema_version": MANUAL_RUNNER_INVOCATION_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": MANUAL_RUNNER_INVOCATION_REVIEW_STATUS,
        "live_attestation_review_sha256": live_review_sha256,
        "advisor_dispatch_go_review_sha256": advisor_review_sha256,
        "manual_runner_invocation_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(manual_runner_invocation)
        ),
        "approved_by": manual_runner_invocation["approved_by"],
        "approval_scope": manual_runner_invocation["approval_scope"],
        "transport_kind": manual_runner_invocation["transport_kind"],
        "egress_allowed": False,
        "external_cost_authorized": False,
        "inference_authorized": True,
        "dispatch_authorized": True,
        "next_gate": MANUAL_RUNNER_INVOCATION_NEXT_GATE,
    }
    validate_manual_runner_invocation_review(review)
    return review


def build_manual_runner_invocation_review_from_files(
    *,
    live_attestation_review_path: Path,
    advisor_dispatch_go_review_path: Path,
    manual_runner_invocation_path: Path,
) -> dict[str, object]:
    """Validate file-backed manual runner invocation without creating transport."""

    live_review_file = live_attestation_review_path.resolve()
    advisor_review_file = advisor_dispatch_go_review_path.resolve()
    invocation_file = manual_runner_invocation_path.resolve()
    live_attestation_review = _require_mapping(
        contract.load_json_no_duplicates(live_review_file),
        "live attestation review",
    )
    advisor_dispatch_go_review = _require_mapping(
        contract.load_json_no_duplicates(advisor_review_file),
        "advisor dispatch GO review",
    )
    manual_runner_invocation = _require_mapping(
        contract.load_json_no_duplicates(invocation_file),
        "manual runner invocation",
    )
    if invocation_file.read_bytes() != (
        contract.canonical_json_bytes(manual_runner_invocation) + b"\n"
    ):
        raise ValueError("manual runner invocation is not canonical JSON bytes")
    review = build_manual_runner_invocation_review(
        live_attestation_review=live_attestation_review,
        advisor_dispatch_go_review=advisor_dispatch_go_review,
        manual_runner_invocation=manual_runner_invocation,
    )
    review["live_attestation_review_path"] = str(live_review_file)
    review["advisor_dispatch_go_review_path"] = str(advisor_review_file)
    review["manual_runner_invocation_path"] = str(invocation_file)
    review["manual_runner_invocation_canonical"] = True
    validate_manual_runner_invocation_review(review)
    return review


def validate_manual_runner_invocation_review(review: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "live_attestation_review_sha256",
        "advisor_dispatch_go_review_sha256",
        "manual_runner_invocation_sha256",
        "approved_by",
        "approval_scope",
        "transport_kind",
        "egress_allowed",
        "external_cost_authorized",
        "inference_authorized",
        "dispatch_authorized",
        "next_gate",
    }
    file_backed_keys = {
        "live_attestation_review_path",
        "advisor_dispatch_go_review_path",
        "manual_runner_invocation_path",
        "manual_runner_invocation_canonical",
    }
    actual_keys = set(review)
    if actual_keys != expected_keys and actual_keys != expected_keys.union(
        file_backed_keys
    ):
        raise ValueError(
            "manual runner invocation review keys mismatch: "
            f"missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys - file_backed_keys}"
        )
    if review.get("schema_version") != MANUAL_RUNNER_INVOCATION_REVIEW_SCHEMA_VERSION:
        raise ValueError("manual runner invocation review schema_version mismatch")
    if review.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("manual runner invocation review experiment_id mismatch")
    if review.get("status") != MANUAL_RUNNER_INVOCATION_REVIEW_STATUS:
        raise ValueError("manual runner invocation review status mismatch")
    for key in (
        "live_attestation_review_sha256",
        "advisor_dispatch_go_review_sha256",
        "manual_runner_invocation_sha256",
    ):
        value = review.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or not all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"manual runner invocation review {key} must be sha256")
    if review.get("approval_scope") != MANUAL_RUNNER_INVOCATION_SCOPE:
        raise ValueError("manual runner invocation review approval scope mismatch")
    if review.get("transport_kind") != MANUAL_RUNNER_TRANSPORT_KIND:
        raise ValueError("manual runner invocation review transport kind mismatch")
    if not isinstance(review.get("approved_by"), str) or not review.get("approved_by"):
        raise ValueError("manual runner invocation review approved_by must be nonempty")
    if review.get("egress_allowed") is not False:
        raise ValueError("manual runner invocation review must keep egress closed")
    if review.get("external_cost_authorized") is not False:
        raise ValueError(
            "manual runner invocation review must not authorize external cost"
        )
    if review.get("inference_authorized") is not True:
        raise ValueError("manual runner invocation review must authorize local inference")
    if review.get("dispatch_authorized") is not True:
        raise ValueError("manual runner invocation review must authorize local dispatch")
    if review.get("next_gate") != MANUAL_RUNNER_INVOCATION_NEXT_GATE:
        raise ValueError("manual runner invocation review next_gate mismatch")
    if "manual_runner_invocation_canonical" in review and review.get(
        "manual_runner_invocation_canonical"
    ) is not True:
        raise ValueError("manual runner invocation must be canonical")
    for key in (
        "live_attestation_review_path",
        "advisor_dispatch_go_review_path",
        "manual_runner_invocation_path",
    ):
        if key in review and (
            not isinstance(review.get(key), str) or not review.get(key)
        ):
            raise ValueError(f"manual runner invocation review {key} must be nonempty")


def replay_completed_synthetic_run(
    output_dir: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> CompletedSyntheticReplay:
    """Replay a completed 24-case synthetic run without opening scorer-only gold."""

    request_identity = _runner_request_identity(artifact_dir)
    case_order = _case_order_from_request_identity(request_identity)
    expected_reservations = _reservations_from_request_identity(
        request_identity,
        run_id=destination_run_id(output_dir),
    )
    artifact_sha256 = contract.validate_runner_artifact_bundle(artifact_dir)
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


def _runner_request_identity(artifact_dir: Path) -> Mapping[str, object]:
    """Build runner request identity without opening scorer-only artifacts."""

    case_order = _read_case_order(artifact_dir)
    request_records = contract.load_jsonl_no_duplicates(
        artifact_dir / "semantic_anchor_request_fixtures_v1.jsonl"
    )
    request_by_case: dict[str, Mapping[str, object]] = {}
    for raw_record in request_records:
        record = _require_mapping(raw_record, "request fixture record")
        case_id = record.get("case_id")
        request = _require_mapping(record.get("request"), "request")
        if not isinstance(case_id, str):
            raise ValueError("request fixture case_id must be string")
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
        "checker": "semantic_anchor_runner_request_identity_v1",
        "status": "pass",
        "case_count": len(case_order),
        "first_case_id": case_order[0],
        "case_order_sha256": contract.sha256_bytes(order_body),
        "request_sha256": request_sha256,
        "request_body_size_bytes": request_body_size_bytes,
    }


def _runner_requests(
    artifact_dir: Path,
    *,
    case_order: Sequence[str],
) -> dict[str, Mapping[str, object]]:
    request_records = contract.load_jsonl_no_duplicates(
        artifact_dir / "semantic_anchor_request_fixtures_v1.jsonl"
    )
    request_by_case: dict[str, Mapping[str, object]] = {}
    for raw_record in request_records:
        record = _require_mapping(raw_record, "request fixture record")
        case_id = record.get("case_id")
        request = _require_mapping(record.get("request"), "request")
        if not isinstance(case_id, str):
            raise ValueError("request fixture case_id must be string")
        if case_id in request_by_case:
            raise ValueError(f"duplicate request fixture case_id: {case_id}")
        request_by_case[case_id] = request
    if tuple(request_by_case) != tuple(case_order):
        raise ValueError("request fixture order differs from fixed case order")
    return request_by_case


def _read_case_order(artifact_dir: Path) -> tuple[str, ...]:
    case_order = _require_mapping(
        contract.load_json_no_duplicates(
            artifact_dir / "semantic_anchor_case_order_v1.json"
        ),
        "case order",
    )
    raw_order = case_order.get("case_order")
    if not isinstance(raw_order, list) or not all(
        isinstance(case_id, str) for case_id in raw_order
    ):
        raise ValueError("case_order must be a string list")
    contract.validate_fixed_case_order(raw_order)
    return tuple(raw_order)


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


def _finish_reason_from_chat_completion(raw_body: str) -> str:
    payload = _loads_object_no_duplicates(raw_body, "raw response body")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("raw response body must contain exactly one choice")
    choice = _require_mapping(choices[0], "raw response choice")
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise ValueError("raw response finish_reason must be string")
    if finish_reason != "stop":
        raise ValueError("raw response finish_reason must be stop")
    return finish_reason


def _loads_object_no_duplicates(text: str, name: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    return _require_mapping(value, name)


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
        description=(
            "Create or replay deterministic sparse v4 run-directory boundaries "
            "without model dispatch."
        )
    )
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--artifact-dir", type=Path, default=contract.ARTIFACT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--replay-pre-dispatch-no-go",
        action="store_true",
        help="Replay an existing zero-dispatch no-go run directory.",
    )
    mode.add_argument(
        "--replay-completed-synthetic",
        action="store_true",
        help=(
            "Replay an existing 24-case sealed synthetic run directory and "
            "validate its scorer input without opening gold."
        ),
    )
    mode.add_argument(
        "--validate-manual-runner-invocation",
        action="store_true",
        help=(
            "Validate the final manual runner invocation receipt and write a "
            "review without creating a transport or dispatching."
        ),
    )
    parser.add_argument(
        "--live-attestation-review",
        type=Path,
        help="Live-attestation review JSON for manual invocation validation.",
    )
    parser.add_argument(
        "--advisor-dispatch-go-review",
        type=Path,
        help="Advisor dispatch-GO review JSON for manual invocation validation.",
    )
    parser.add_argument(
        "--manual-runner-invocation",
        type=Path,
        help="Canonical manual runner invocation receipt JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Create-only output path for manual invocation review JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.validate_manual_runner_invocation:
        if args.output_dir is not None:
            raise ValueError(
                "--validate-manual-runner-invocation does not take output_dir"
            )
        if not (
            args.live_attestation_review
            and args.advisor_dispatch_go_review
            and args.manual_runner_invocation
            and args.output
        ):
            raise ValueError(
                "--validate-manual-runner-invocation requires "
                "--live-attestation-review, --advisor-dispatch-go-review, "
                "--manual-runner-invocation, and --output"
            )
        review = build_manual_runner_invocation_review_from_files(
            live_attestation_review_path=args.live_attestation_review,
            advisor_dispatch_go_review_path=args.advisor_dispatch_go_review,
            manual_runner_invocation_path=args.manual_runner_invocation,
        )
        _create_only(args.output, contract.canonical_json_bytes(review) + b"\n")
        print(f"Wrote v4 manual runner invocation review to {args.output.resolve()}")
    elif args.replay_pre_dispatch_no_go:
        if args.output_dir is None:
            raise ValueError("--replay-pre-dispatch-no-go requires output_dir")
        result = replay_pre_dispatch_no_go_run(
            args.output_dir,
            artifact_dir=args.artifact_dir,
        )
        print(
            "Replayed v4 pre-dispatch no-go run under "
            f"{result.output_dir}: reservations={len(result.reservation_paths)}"
        )
    elif args.replay_completed_synthetic:
        if args.output_dir is None:
            raise ValueError("--replay-completed-synthetic requires output_dir")
        result = replay_completed_synthetic_run(
            args.output_dir,
            artifact_dir=args.artifact_dir,
        )
        print(
            "Replayed v4 completed synthetic run under "
            f"{result.output_dir}: reservations={len(result.reservation_paths)} "
            f"dispatches={len(result.dispatch_paths)} "
            f"raw_responses={len(result.raw_response_paths)}"
        )
    else:
        if args.output_dir is None:
            raise ValueError("output_dir is required")
        result = build_pre_dispatch_no_go_run(
            args.output_dir,
            artifact_dir=args.artifact_dir,
        )
        print(f"Wrote v4 pre-dispatch no-go run under {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

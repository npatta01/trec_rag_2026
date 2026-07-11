from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight
from trec_rag import det_sparse_v4_runner as runner


def test_pre_dispatch_no_go_runner_writes_create_only_reservations_and_zero_call_boundary(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"

    result = runner.build_pre_dispatch_no_go_run(output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_manifest_path == output_dir.resolve() / "run_manifest.json"
    assert result.terminal_receipt_path == output_dir.resolve() / "terminal_receipt.json"
    assert len(result.reservation_paths) == 24
    assert result.reservation_paths[0] == (
        output_dir.resolve() / "reservations" / "synthetic-case-001.json"
    )
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(result.terminal_receipt_path.read_text(encoding="utf-8"))
    reservations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in result.reservation_paths
    ]
    request_identity = preflight.build_offline_preflight_report()["request_identity"]
    assert manifest["schema_version"] == "semantic_anchor_run_manifest_v1"
    assert manifest["case_count"] == 24
    assert manifest["terminal_receipt_path"] == "terminal_receipt.json"
    assert terminal["schema_version"] == "semantic_anchor_terminal_receipt_v1"
    assert terminal["terminal_state"] == "pre_dispatch_no_go"
    assert terminal["attempted_calls"] == 0
    assert terminal["completed_calls"] == 0
    assert terminal["raw_committed_calls"] == 0
    assert terminal["gold_opened"] is False
    assert terminal["artifact_sha256"] == manifest["artifact_sha256"]
    assert [reservation["case_id"] for reservation in reservations] == [
        f"synthetic-case-{index:03d}" for index in range(1, 25)
    ]
    assert [reservation["request_sha256"] for reservation in reservations] == list(
        request_identity["request_sha256"].values()
    )
    assert {reservation["run_id"] for reservation in reservations} == {"v4-no-go"}
    assert all(reservation["create_only"] is True for reservation in reservations)
    contract.validate_ledger_prefix(
        case_order=tuple(f"synthetic-case-{index:03d}" for index in range(1, 25)),
        reservations=reservations,
        dispatches=[],
        raw_responses=[],
        transport_failures=[],
        case_receipts=[],
        terminal_receipt=terminal,
        run_manifest=manifest,
    )
    replayed = runner.replay_pre_dispatch_no_go_run(output_dir)
    assert replayed == result


def test_pre_dispatch_no_go_runner_refuses_existing_output_dir(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        runner.build_pre_dispatch_no_go_run(output_dir)


def test_pre_dispatch_no_go_replay_rejects_drift_and_noncanonical_files(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"
    result = runner.build_pre_dispatch_no_go_run(output_dir)

    terminal = json.loads(result.terminal_receipt_path.read_text(encoding="utf-8"))
    result.terminal_receipt_path.write_bytes(
        contract.canonical_json_bytes(dict(terminal, attempted_calls=1)) + b"\n"
    )
    with pytest.raises(ValueError, match="terminal state|zero attempted"):
        runner.replay_pre_dispatch_no_go_run(output_dir)
    result.terminal_receipt_path.write_bytes(
        contract.canonical_json_bytes(terminal) + b"\n"
    )

    first_reservation = result.reservation_paths[0]
    reservation = json.loads(first_reservation.read_text(encoding="utf-8"))
    first_reservation.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        runner.replay_pre_dispatch_no_go_run(output_dir)
    first_reservation.write_bytes(contract.canonical_json_bytes(reservation) + b"\n")

    (output_dir / "dispatches").mkdir()
    with pytest.raises(ValueError, match="run directory file set"):
        runner.replay_pre_dispatch_no_go_run(output_dir)


def test_pre_dispatch_no_go_replay_rejects_reservation_hash_drift(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"
    result = runner.build_pre_dispatch_no_go_run(output_dir)
    first_reservation = result.reservation_paths[0]
    reservation = json.loads(first_reservation.read_text(encoding="utf-8"))
    reservation["request_sha256"] = "9" * 64
    first_reservation.write_bytes(contract.canonical_json_bytes(reservation) + b"\n")

    with pytest.raises(ValueError, match="reservations do not match request identity"):
        runner.replay_pre_dispatch_no_go_run(output_dir)


def _gold_driven_response(case: dict[str, object]) -> dict[str, object]:
    if case["decision"] == "select":
        acceptable = case["acceptable_ranges"][0]
        return {
            "schema_version": "semantic_anchor_response_v1",
            "case_id": case["case_id"],
            "decision": "select",
            "start_token": acceptable["start_token"],
            "end_token": acceptable["end_token"],
        }
    return {
        "schema_version": "semantic_anchor_response_v1",
        "case_id": case["case_id"],
        "decision": "abstain",
        "start_token": -1,
        "end_token": -1,
    }


def _raw_body_for_response(response: dict[str, object]) -> str:
    assistant_content = json.dumps(response, separators=(",", ":"), sort_keys=True)
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": assistant_content,
                },
            }
        ]
    }
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def _write_record(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _build_completed_run(output_dir: Path) -> None:
    preflight_report = preflight.build_offline_preflight_report()
    request_identity = preflight_report["request_identity"]
    request_sha256 = request_identity["request_sha256"]
    artifact_sha256 = preflight_report["artifact_sha256"]
    case_order = tuple(request_sha256)
    run_id = output_dir.resolve().name
    output_dir.mkdir()
    _write_record(
        output_dir / "run_manifest.json",
        {
            "schema_version": "semantic_anchor_run_manifest_v1",
            "case_count": 24,
            "artifact_sha256": artifact_sha256,
            "terminal_receipt_path": "terminal_receipt.json",
        },
    )
    terminal = {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "raw_sealed_pending_scorer",
        "attempted_calls": 24,
        "completed_calls": 24,
        "raw_committed_calls": 24,
        "gold_opened": False,
        "artifact_sha256": artifact_sha256,
    }
    _write_record(output_dir / "terminal_receipt.json", terminal)
    gold = contract.load_json_no_duplicates(
        contract.ARTIFACT_DIR / "semantic_anchor_gold_labels_v1.json"
    )
    gold_by_case = {case["case_id"]: case for case in gold["cases"]}
    reservations = []
    dispatches = []
    raw_responses = []
    case_receipts = []
    sealed_responses = []
    for index, case_id in enumerate(case_order, start=1):
        request_hash = request_sha256[case_id]
        response = _gold_driven_response(gold_by_case[case_id])
        raw_body = _raw_body_for_response(response)
        raw_hash = contract.sha256_bytes(raw_body.encode("utf-8"))
        reservation = {
            "schema_version": "semantic_anchor_reservation_v1",
            "run_id": run_id,
            "case_id": case_id,
            "request_sha256": request_hash,
            "create_only": True,
        }
        dispatch = {
            "schema_version": "semantic_anchor_dispatch_record_v1",
            "case_id": case_id,
            "request_bytes_sha256": request_hash,
            "loopback_only": True,
            "dispatch_counted": True,
        }
        raw_response = {
            "schema_version": "semantic_anchor_raw_response_body_v1",
            "case_id": case_id,
            "http_status": 200,
            "finish_reason": "stop",
            "served_model": "gpt-oss-local",
            "body_size_bytes": len(raw_body.encode("utf-8")),
            "body_sha256": raw_hash,
        }
        case_receipt = {
            "schema_version": "semantic_anchor_case_receipt_v1",
            "case_id": case_id,
            "request_sha256": request_hash,
            "machine_status": "mechanical_pass",
            "raw_response_sha256": raw_hash,
        }
        reservations.append(reservation)
        dispatches.append(dispatch)
        raw_responses.append(raw_response)
        case_receipts.append(case_receipt)
        _write_record(
            output_dir / "reservations" / f"{case_id}.json",
            reservation,
        )
        _write_record(
            output_dir / "dispatches" / f"{case_id}.json",
            dispatch,
        )
        _write_record(
            output_dir / "raw_responses" / f"{case_id}.json",
            raw_response,
        )
        _write_record(
            output_dir / "case_receipts" / f"{case_id}.json",
            case_receipt,
        )
        sealed_responses.append(
            {
                "case_id": case_id,
                "raw_response_sha256": raw_hash,
                "raw_response_body": raw_body,
            }
        )
    _write_record(
        output_dir / "sealed_scorer_input.json",
        {
            "schema_version": "semantic_anchor_sealed_scorer_input_v1",
            "run_manifest": {
                "schema_version": "semantic_anchor_run_manifest_v1",
                "case_count": 24,
                "artifact_sha256": artifact_sha256,
                "terminal_receipt_path": "terminal_receipt.json",
            },
            "terminal_receipt": terminal,
            "reservations": reservations,
            "dispatches": dispatches,
            "raw_responses": raw_responses,
            "case_receipts": case_receipts,
            "responses": sealed_responses,
        },
    )


def test_completed_synthetic_replay_validates_24_case_run_and_sealed_scorer_input(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)

    replayed = runner.replay_completed_synthetic_run(output_dir)

    assert replayed.output_dir == output_dir.resolve()
    assert len(replayed.reservation_paths) == 24
    assert len(replayed.dispatch_paths) == 24
    assert len(replayed.raw_response_paths) == 24
    assert len(replayed.case_receipt_paths) == 24
    assert replayed.sealed_scorer_input_path == output_dir.resolve() / "sealed_scorer_input.json"


def test_completed_synthetic_replay_rejects_extra_dispatch_artifact_and_sealed_hash_drift(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    (output_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="run directory file set"):
        runner.replay_completed_synthetic_run(output_dir)
    (output_dir / "unexpected.json").unlink()

    sealed_path = output_dir / "sealed_scorer_input.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed["responses"][0]["raw_response_sha256"] = "9" * 64
    _write_record(sealed_path, sealed)
    with pytest.raises(ValueError, match="hash differs from raw record|raw hash mismatch"):
        runner.replay_completed_synthetic_run(output_dir)

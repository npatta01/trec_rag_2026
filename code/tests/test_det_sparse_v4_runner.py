from __future__ import annotations

import json
import subprocess
import sys
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


def test_pre_dispatch_no_go_runner_does_not_open_scorer_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _deny_gold_json_open(monkeypatch)

    result = runner.build_pre_dispatch_no_go_run(tmp_path / "v4-no-go")
    replayed = runner.replay_pre_dispatch_no_go_run(result.output_dir)

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


def test_runner_cli_replays_pre_dispatch_no_go_without_dispatch(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"
    runner.build_pre_dispatch_no_go_run(output_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_runner",
            "--replay-pre-dispatch-no-go",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert "Replayed v4 pre-dispatch no-go run" in completed.stdout
    assert "reservations=24" in completed.stdout
    assert not (output_dir / "dispatches").exists()


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


def _write_offset_parity_review(tmp_path: Path) -> Path:
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
    fixtures_path = tmp_path / "offset-parity-fixtures.json"
    review_path = tmp_path / "offset-parity-review.json"
    fixtures_path.write_bytes(
        contract.canonical_json_bytes(
            {
                "schema_version": "semantic_anchor_offset_parity_fixture_v1",
                "health": _offset_health(),
                "fixtures": fixtures,
            }
        )
        + b"\n"
    )
    review = preflight.build_offset_parity_review(fixtures_path)
    review_path.write_bytes(preflight.canonical_report_bytes(review))
    return review_path


def _write_model_inventory_attestation(tmp_path: Path) -> tuple[Path, str]:
    inventory_path = tmp_path / "model-inventory-attestation.json"
    inventory = contract.load_json_no_duplicates(
        contract.ARTIFACT_DIR / "semantic_anchor_model_inventory_attestation_v1.json"
    )
    inventory_path.write_bytes(contract.canonical_json_bytes(inventory) + b"\n")
    return inventory_path, contract.sha256_file(inventory_path)


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
                "schema_sha256": request_identity["schema_sha256"][case_id],
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


def _approved_reviews(tmp_path: Path):
    inventory_path, model_inventory_sha256 = _write_model_inventory_attestation(tmp_path)
    bundle_path = tmp_path / "live-attestation-bundle.json"
    bundle_path.write_bytes(
        contract.canonical_json_bytes(_live_attestation_bundle(model_inventory_sha256))
        + b"\n"
    )
    live_review = preflight.build_live_attestation_review(
        bundle_path,
        offset_parity_review_path=_write_offset_parity_review(tmp_path),
        model_inventory_attestation_path=inventory_path,
    )
    live_review_sha256 = contract.sha256_bytes(
        preflight.canonical_report_bytes(live_review)
    )
    advisor_go = {
        "schema_version": "semantic_anchor_advisor_dispatch_go_v1",
        "approval_scope": "det_sparse_v4_synthetic_local_dispatch",
        "approved_by": "advisor",
        "live_attestation_review_sha256": live_review_sha256,
        preflight.ADVISOR_CLOSED_GATES_ACK_KEY: True,
    }
    advisor_review = preflight.build_advisor_dispatch_go_review(
        live_review,
        advisor_go,
    )
    return live_review, advisor_review


def _manual_runner_invocation(live_review, advisor_review):
    return {
        "schema_version": runner.MANUAL_RUNNER_INVOCATION_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": runner.MANUAL_RUNNER_INVOCATION_STATUS,
        "live_attestation_review_sha256": contract.sha256_bytes(
            preflight.canonical_report_bytes(live_review)
        ),
        "advisor_dispatch_go_review_sha256": contract.sha256_bytes(
            preflight.canonical_report_bytes(advisor_review)
        ),
        "approval_scope": runner.MANUAL_RUNNER_INVOCATION_SCOPE,
        "transport_kind": runner.MANUAL_RUNNER_TRANSPORT_KIND,
        "egress_allowed": False,
        "external_cost_authorized": False,
        "inference_authorized": True,
        "dispatch_authorized": True,
        "approved_by": "advisor",
    }


def test_synthetic_dispatch_run_writes_raw_first_ledger_with_fake_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    seen: list[str] = []

    def fake_transport(case_id: str, request: dict[str, object]) -> str:
        seen.append(case_id)
        assert request["model"] == "gpt-oss-local"
        return _raw_body_for_response(
            {
                "schema_version": "semantic_anchor_response_v1",
                "case_id": case_id,
                "decision": "abstain",
                "start_token": -1,
                "end_token": -1,
            }
        )

    output_dir = tmp_path / "dispatch-v4"
    _deny_gold_json_open(monkeypatch)

    result = runner.build_synthetic_dispatch_run(
        output_dir,
        live_attestation_review=live_review,
        advisor_dispatch_go_review=advisor_review,
        manual_runner_invocation=manual_invocation,
        transport=fake_transport,
    )

    assert seen == [f"synthetic-case-{index:03d}" for index in range(1, 25)]
    assert len(result.dispatch_paths) == 24
    assert len(result.raw_response_paths) == 24
    assert len(result.case_receipt_paths) == 24
    terminal = json.loads(result.terminal_receipt_path.read_text(encoding="utf-8"))
    assert terminal["terminal_state"] == "raw_sealed_pending_scorer"
    assert terminal["attempted_calls"] == 24
    assert terminal["completed_calls"] == 24
    assert terminal["raw_committed_calls"] == 24
    assert terminal["gold_opened"] is False
    sealed = json.loads(result.sealed_scorer_input_path.read_text(encoding="utf-8"))
    assert sealed["terminal_receipt"] == terminal
    assert [row["case_id"] for row in sealed["responses"]] == seen
    replayed = runner.replay_completed_synthetic_run(output_dir)
    assert replayed.output_dir == result.output_dir


def test_synthetic_dispatch_run_rejects_unbound_advisor_review_before_writing(
    tmp_path: Path,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    advisor_review = dict(advisor_review, live_attestation_review_sha256="9" * 64)
    output_dir = tmp_path / "dispatch-v4"

    with pytest.raises(ValueError, match="not bound"):
        runner.build_synthetic_dispatch_run(
            output_dir,
            live_attestation_review=live_review,
            advisor_dispatch_go_review=advisor_review,
            manual_runner_invocation=manual_invocation,
            transport=lambda _case_id, _request: "{}",
        )

    assert not output_dir.exists()


def test_synthetic_dispatch_run_requires_manual_invocation_before_transport(
    tmp_path: Path,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = dict(
        _manual_runner_invocation(live_review, advisor_review),
        dispatch_authorized=False,
    )
    output_dir = tmp_path / "dispatch-v4"
    seen: list[str] = []

    def fake_transport(case_id: str, request: dict[str, object]) -> str:
        seen.append(case_id)
        return "{}"

    with pytest.raises(ValueError, match="must authorize local dispatch"):
        runner.build_synthetic_dispatch_run(
            output_dir,
            live_attestation_review=live_review,
            advisor_dispatch_go_review=advisor_review,
            manual_runner_invocation=manual_invocation,
            transport=fake_transport,
        )

    assert seen == []
    assert not output_dir.exists()


def test_manual_runner_invocation_review_cli_validates_file_backed_gate(
    tmp_path: Path,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    live_review_path = tmp_path / "live-attestation-review.json"
    advisor_review_path = tmp_path / "advisor-dispatch-go-review.json"
    invocation_path = tmp_path / "manual-runner-invocation.json"
    output_path = tmp_path / "manual-runner-invocation-review.json"
    live_review_path.write_bytes(
        contract.canonical_json_bytes(live_review) + b"\n"
    )
    advisor_review_path.write_bytes(
        contract.canonical_json_bytes(advisor_review) + b"\n"
    )
    invocation_path.write_bytes(
        contract.canonical_json_bytes(manual_invocation) + b"\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_runner",
            "--validate-manual-runner-invocation",
            "--live-attestation-review",
            str(live_review_path),
            "--advisor-dispatch-go-review",
            str(advisor_review_path),
            "--manual-runner-invocation",
            str(invocation_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert "Wrote v4 manual runner invocation review" in completed.stdout
    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["schema_version"] == (
        "semantic_anchor_manual_runner_invocation_review_v1"
    )
    assert review["status"] == "manual_runner_invocation_review_pass"
    assert review["live_attestation_review_sha256"] == contract.sha256_bytes(
        preflight.canonical_report_bytes(live_review)
    )
    assert review["advisor_dispatch_go_review_sha256"] == contract.sha256_bytes(
        preflight.canonical_report_bytes(advisor_review)
    )
    assert review["manual_runner_invocation_sha256"] == contract.sha256_bytes(
        contract.canonical_json_bytes(manual_invocation)
    )
    assert review["transport_kind"] == "injected_local_loopback"
    assert review["egress_allowed"] is False
    assert review["external_cost_authorized"] is False
    assert review["inference_authorized"] is True
    assert review["dispatch_authorized"] is True
    assert review["next_gate"] == "invoke_injected_local_transport"
    assert review["manual_runner_invocation_canonical"] is True
    runner.validate_manual_runner_invocation_review(review)
    assert output_path.read_bytes() == contract.canonical_json_bytes(review) + b"\n"


def test_manual_runner_invocation_review_rejects_noncanonical_receipt(
    tmp_path: Path,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    live_review_path = tmp_path / "live-attestation-review.json"
    advisor_review_path = tmp_path / "advisor-dispatch-go-review.json"
    invocation_path = tmp_path / "manual-runner-invocation.json"
    output_path = tmp_path / "manual-runner-invocation-review.json"
    live_review_path.write_bytes(
        contract.canonical_json_bytes(live_review) + b"\n"
    )
    advisor_review_path.write_bytes(
        contract.canonical_json_bytes(advisor_review) + b"\n"
    )
    invocation_path.write_text(
        json.dumps(manual_invocation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_runner",
            "--validate-manual-runner-invocation",
            "--live-attestation-review",
            str(live_review_path),
            "--advisor-dispatch-go-review",
            str(advisor_review_path),
            "--manual-runner-invocation",
            str(invocation_path),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "manual runner invocation is not canonical JSON bytes" in completed.stderr
    assert not output_path.exists()


def test_synthetic_dispatch_run_seals_transport_no_body_no_go_on_bad_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    output_dir = tmp_path / "dispatch-v4"
    seen: list[str] = []

    def fake_transport(case_id: str, request: dict[str, object]) -> str:
        seen.append(case_id)
        assert request["model"] == "gpt-oss-local"
        return ""

    _deny_gold_json_open(monkeypatch)

    with pytest.raises(ValueError, match="transport_no_body_no_go"):
        runner.build_synthetic_dispatch_run(
            output_dir,
            live_attestation_review=live_review,
            advisor_dispatch_go_review=advisor_review,
            manual_runner_invocation=manual_invocation,
            transport=fake_transport,
        )

    assert seen == ["synthetic-case-001"]
    assert not (output_dir / "sealed_scorer_input.json").exists()
    terminal = json.loads(
        (output_dir / "terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["terminal_state"] == "transport_no_body_no_go"
    assert terminal["attempted_calls"] == 1
    assert terminal["completed_calls"] == 0
    assert terminal["raw_committed_calls"] == 0
    assert terminal["gold_opened"] is False
    failure = json.loads(
        (
            output_dir
            / "transport_failures"
            / "synthetic-case-001.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["exception_class"] == "ValueError"
    assert failure["request_bytes_sha256"] == json.loads(
        (
            output_dir
            / "dispatches"
            / "synthetic-case-001.json"
        ).read_text(encoding="utf-8")
    )["request_bytes_sha256"]
    case_receipt = json.loads(
        (
            output_dir
            / "case_receipts"
            / "synthetic-case-001.json"
        ).read_text(encoding="utf-8")
    )
    assert case_receipt["machine_status"] == "mechanical_no_go"


def test_synthetic_dispatch_run_seals_transport_no_body_no_go_on_non_stop_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    live_review, advisor_review = _approved_reviews(tmp_path)
    manual_invocation = _manual_runner_invocation(live_review, advisor_review)
    output_dir = tmp_path / "dispatch-v4"
    seen: list[str] = []

    def fake_transport(case_id: str, request: dict[str, object]) -> str:
        seen.append(case_id)
        assert request["model"] == "gpt-oss-local"
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "{}",
                        },
                    }
                ]
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    _deny_gold_json_open(monkeypatch)

    with pytest.raises(ValueError, match="transport_no_body_no_go"):
        runner.build_synthetic_dispatch_run(
            output_dir,
            live_attestation_review=live_review,
            advisor_dispatch_go_review=advisor_review,
            manual_runner_invocation=manual_invocation,
            transport=fake_transport,
        )

    assert seen == ["synthetic-case-001"]
    assert not (output_dir / "sealed_scorer_input.json").exists()
    terminal = json.loads(
        (output_dir / "terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["terminal_state"] == "transport_no_body_no_go"
    assert terminal["attempted_calls"] == 1
    assert terminal["completed_calls"] == 0
    assert terminal["raw_committed_calls"] == 0
    failure = json.loads(
        (
            output_dir
            / "transport_failures"
            / "synthetic-case-001.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["exception_message"] == "raw response finish_reason must be stop"
    case_receipt = json.loads(
        (
            output_dir
            / "case_receipts"
            / "synthetic-case-001.json"
        ).read_text(encoding="utf-8")
    )
    assert case_receipt["machine_status"] == "mechanical_no_go"


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


def _deny_gold_json_open(monkeypatch: pytest.MonkeyPatch) -> None:
    original = contract.load_json_no_duplicates

    def guarded(path: Path):
        if Path(path).name == "semantic_anchor_gold_labels_v1.json":
            raise AssertionError("runner must not open scorer-only gold labels")
        return original(path)

    monkeypatch.setattr(contract, "load_json_no_duplicates", guarded)


def _build_completed_run(output_dir: Path) -> None:
    preflight_report = preflight.build_offline_preflight_report()
    request_identity = preflight_report["request_identity"]
    request_sha256 = request_identity["request_sha256"]
    artifact_sha256 = contract.validate_runner_artifact_bundle()
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


def test_completed_synthetic_replay_does_not_open_scorer_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    _deny_gold_json_open(monkeypatch)

    replayed = runner.replay_completed_synthetic_run(output_dir)

    assert len(replayed.raw_response_paths) == 24


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


def test_completed_synthetic_replay_rejects_gold_opened_terminal(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    terminal_path = output_dir / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["gold_opened"] = True
    _write_record(terminal_path, terminal)

    with pytest.raises(ValueError, match="must not open gold"):
        runner.replay_completed_synthetic_run(output_dir)


def test_completed_synthetic_replay_rejects_terminal_counter_drift(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    terminal_path = output_dir / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["raw_committed_calls"] = 23
    _write_record(terminal_path, terminal)

    with pytest.raises(ValueError, match="requires 24 sealed raw responses"):
        runner.replay_completed_synthetic_run(output_dir)


def test_completed_synthetic_replay_rejects_missing_raw_response_file(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    (output_dir / "raw_responses" / "synthetic-case-001.json").unlink()

    with pytest.raises(ValueError, match="raw_responses file set mismatch"):
        runner.replay_completed_synthetic_run(output_dir)


def test_completed_synthetic_replay_rejects_sealed_response_order_drift(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)
    sealed_path = output_dir / "sealed_scorer_input.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed["responses"][0], sealed["responses"][1] = (
        sealed["responses"][1],
        sealed["responses"][0],
    )
    _write_record(sealed_path, sealed)

    with pytest.raises(ValueError, match="case order"):
        runner.replay_completed_synthetic_run(output_dir)


def test_runner_cli_replays_completed_synthetic_run_without_scoring(tmp_path: Path):
    output_dir = tmp_path / "completed-v4"
    _build_completed_run(output_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_runner",
            "--replay-completed-synthetic",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert "Replayed v4 completed synthetic run" in completed.stdout
    assert "reservations=24" in completed.stdout
    assert "dispatches=24" in completed.stdout
    assert "raw_responses=24" in completed.stdout
    terminal = json.loads((output_dir / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert terminal["gold_opened"] is False

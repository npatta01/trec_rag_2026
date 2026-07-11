from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_scorer as scorer


def _case_order() -> tuple[str, ...]:
    return tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))


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


def _sealed_scorer_input(*, terminal_state: str = "raw_sealed_pending_scorer"):
    gold = contract.load_json_no_duplicates(
        contract.ARTIFACT_DIR / "semantic_anchor_gold_labels_v1.json"
    )
    artifact_sha256 = contract.validate_artifact_bundle()
    run_id = "synthetic-scorer-run"
    reservations = []
    dispatches = []
    raw_responses = []
    case_receipts = []
    responses = []
    for index, case in enumerate(gold["cases"], start=1):
        case_id = str(case["case_id"])
        request_sha256 = f"{index:064x}"[-64:]
        response = _gold_driven_response(case)
        raw_body = _raw_body_for_response(response)
        raw_body_sha256 = contract.sha256_bytes(raw_body.encode("utf-8"))
        reservations.append(
            {
                "schema_version": "semantic_anchor_reservation_v1",
                "run_id": run_id,
                "case_id": case_id,
                "request_sha256": request_sha256,
                "create_only": True,
            }
        )
        dispatches.append(
            {
                "schema_version": "semantic_anchor_dispatch_record_v1",
                "case_id": case_id,
                "request_bytes_sha256": request_sha256,
                "loopback_only": True,
                "dispatch_counted": True,
            }
        )
        raw_responses.append(
            {
                "schema_version": "semantic_anchor_raw_response_body_v1",
                "case_id": case_id,
                "http_status": 200,
                "finish_reason": "stop",
                "served_model": "gpt-oss-local",
                "body_size_bytes": len(raw_body.encode("utf-8")),
                "body_sha256": raw_body_sha256,
            }
        )
        case_receipts.append(
            {
                "schema_version": "semantic_anchor_case_receipt_v1",
                "case_id": case_id,
                "request_sha256": request_sha256,
                "machine_status": "mechanical_pass",
                "raw_response_sha256": raw_body_sha256,
            }
        )
        responses.append(
            {
                "case_id": case_id,
                "raw_response_sha256": raw_body_sha256,
                "raw_response_body": raw_body,
            }
        )
    return {
        "schema_version": "semantic_anchor_sealed_scorer_input_v1",
        "run_manifest": {
            "schema_version": "semantic_anchor_run_manifest_v1",
            "case_count": 24,
            "artifact_sha256": artifact_sha256,
            "terminal_receipt_path": "terminal_receipt.json",
        },
        "terminal_receipt": {
            "schema_version": "semantic_anchor_terminal_receipt_v1",
            "terminal_state": terminal_state,
            "attempted_calls": 24,
            "completed_calls": 24,
            "raw_committed_calls": 24,
            "gold_opened": False,
            "artifact_sha256": artifact_sha256,
        },
        "reservations": reservations,
        "dispatches": dispatches,
        "raw_responses": raw_responses,
        "case_receipts": case_receipts,
        "responses": responses,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(contract.canonical_json_bytes(payload) + b"\n")


def _reviewer_receipt_for_scorer_review(
    scorer_review_path: Path,
    scorer_review: dict[str, object],
    *,
    reviewer_count: int = 2,
    unanimous: bool = True,
) -> dict[str, object]:
    artifact_hashes = contract.validate_artifact_bundle()
    return {
        "schema_version": "semantic_anchor_reviewer_receipt_v1",
        "reviewer_count": reviewer_count,
        "unanimous": unanimous,
        "scorer_review_sha256": contract.sha256_file(scorer_review_path),
        "sealed_responses_sha256": scorer_review["sealed_responses_sha256"],
        "gold_sha256": scorer_review["gold_sha256"],
        "rubric_sha256": artifact_hashes["semantic_anchor_reviewer_rubric_v1.md"],
        "artifact_bundle_sha256": contract.sha256_bytes(
            contract.canonical_json_bytes(artifact_hashes)
        ),
    }


def test_scorer_review_builds_case_ordered_receipt_after_seal_checks(tmp_path: Path):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    _write_json(sealed_path, _sealed_scorer_input())

    review = scorer.build_scorer_review(sealed_path)

    scorer.validate_scorer_review(review, case_order=_case_order())
    assert review["schema_version"] == "semantic_anchor_scorer_review_v1"
    assert review["gold_opened_after_seal"] is True
    assert review["inference_authorized"] is False
    assert review["dispatch_authorized"] is False
    classifications = [
        row["classification"]
        for row in review["scorer_receipt"]["case_results"]
    ]
    assert classifications[:18] == ["correct_select"] * 18
    assert classifications[18:] == ["safe_abstain"] * 6


def test_scorer_review_rejects_incomplete_or_reordered_sealed_inputs(tmp_path: Path):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    incomplete = _sealed_scorer_input(terminal_state="pre_dispatch_no_go")
    incomplete["terminal_receipt"]["attempted_calls"] = 0
    incomplete["terminal_receipt"]["completed_calls"] = 0
    incomplete["terminal_receipt"]["raw_committed_calls"] = 0
    incomplete["terminal_receipt"]["gold_opened"] = False
    _write_json(sealed_path, incomplete)
    with pytest.raises(ValueError, match="terminal attempted_calls differs"):
        scorer.build_scorer_review(sealed_path)

    reordered = _sealed_scorer_input()
    reordered["responses"][0], reordered["responses"][1] = (
        reordered["responses"][1],
        reordered["responses"][0],
    )
    _write_json(sealed_path, reordered)
    with pytest.raises(ValueError, match="case order mismatch"):
        scorer.build_scorer_review(sealed_path)


def test_scorer_review_does_not_open_gold_when_seal_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sealed = _sealed_scorer_input()
    sealed["terminal_receipt"]["attempted_calls"] = 23
    sealed["terminal_receipt"]["completed_calls"] = 23
    sealed["terminal_receipt"]["raw_committed_calls"] = 23
    sealed_path = tmp_path / "sealed-scorer-input.json"
    _write_json(sealed_path, sealed)
    opened: list[str] = []
    original_loader = scorer.contract.load_json_no_duplicates

    def recording_loader(path: Path):
        opened.append(path.name)
        return original_loader(path)

    monkeypatch.setattr(scorer.contract, "load_json_no_duplicates", recording_loader)

    with pytest.raises(ValueError, match="raw_sealed_pending_scorer requires 24"):
        scorer.build_scorer_review(sealed_path)

    assert "sealed-scorer-input.json" in opened
    assert "semantic_anchor_gold_labels_v1.json" not in opened


def test_scorer_review_rejects_noncanonical_sealed_input_before_gold_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    sealed_path.write_text(
        json.dumps(_sealed_scorer_input(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    opened: list[str] = []
    original_loader = scorer.contract.load_json_no_duplicates

    def recording_loader(path: Path):
        opened.append(path.name)
        return original_loader(path)

    monkeypatch.setattr(scorer.contract, "load_json_no_duplicates", recording_loader)

    with pytest.raises(ValueError, match="canonical JSON"):
        scorer.build_scorer_review(sealed_path)

    assert "sealed-scorer-input.json" in opened
    assert "semantic_anchor_gold_labels_v1.json" not in opened


def test_scorer_review_classifies_malformed_sealed_response(tmp_path: Path):
    sealed = _sealed_scorer_input()
    bad_body = _raw_body_for_response(
        {
            "schema_version": "semantic_anchor_response_v1",
            "case_id": "synthetic-case-001",
            "decision": "select",
            "start_token": True,
            "end_token": 2,
        }
    )
    bad_hash = contract.sha256_bytes(bad_body.encode("utf-8"))
    sealed["responses"][0]["raw_response_body"] = bad_body
    sealed["responses"][0]["raw_response_sha256"] = bad_hash
    sealed["raw_responses"][0]["body_size_bytes"] = len(bad_body.encode("utf-8"))
    sealed["raw_responses"][0]["body_sha256"] = bad_hash
    sealed["case_receipts"][0]["raw_response_sha256"] = bad_hash
    sealed_path = tmp_path / "sealed-scorer-input.json"
    _write_json(sealed_path, sealed)

    review = scorer.build_scorer_review(sealed_path)

    assert review["scorer_receipt"]["case_results"][0] == {
        "case_id": "synthetic-case-001",
        "classification": "mechanical_failure",
    }
    assert review["scorer_receipt"]["case_results"][1]["classification"] == "correct_select"


def test_scorer_review_rejects_parsed_response_drift_from_raw_body_hash(tmp_path: Path):
    sealed = _sealed_scorer_input()
    drifted_response = _gold_driven_response(
        {
            "case_id": "synthetic-case-001",
            "decision": "select",
            "acceptable_ranges": [{"start_token": 2, "end_token": 4}],
        }
    )
    sealed["responses"][0]["raw_response_body"] = _raw_body_for_response(drifted_response)
    sealed_path = tmp_path / "sealed-scorer-input.json"
    _write_json(sealed_path, sealed)

    with pytest.raises(ValueError, match="body hash mismatch"):
        scorer.build_scorer_review(sealed_path)


def test_scorer_cli_writes_create_only_review_without_dispatch(tmp_path: Path):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    output_path = tmp_path / "scorer-review.json"
    _write_json(sealed_path, _sealed_scorer_input())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_scorer",
            str(sealed_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    review = contract.load_json_no_duplicates(output_path)
    scorer.validate_scorer_review(review, case_order=_case_order())
    assert review["dispatch_authorized"] is False

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "trec_rag.det_sparse_v4_scorer",
                str(sealed_path),
                "--output",
                str(output_path),
            ],
            check=True,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
        )


def test_reviewer_qualification_review_binds_consensus_to_scorer_review(
    tmp_path: Path,
):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    scorer_review_path = tmp_path / "scorer-review.json"
    reviewer_receipt_path = tmp_path / "reviewer-receipt.json"
    _write_json(sealed_path, _sealed_scorer_input())
    scorer_review = scorer.build_scorer_review(sealed_path)
    _write_json(scorer_review_path, scorer_review)
    _write_json(
        reviewer_receipt_path,
        _reviewer_receipt_for_scorer_review(scorer_review_path, scorer_review),
    )

    review = scorer.build_reviewer_qualification_review(
        scorer_review_path,
        reviewer_receipt_path,
    )

    scorer.validate_reviewer_qualification_review(review)
    assert review["schema_version"] == "semantic_anchor_reviewer_qualification_review_v1"
    assert review["terminal_state"] == "completed_synthetic_go"
    assert review["dispatch_authorized"] is False


@pytest.mark.parametrize(
    "binding_field",
    [
        "scorer_review_sha256",
        "sealed_responses_sha256",
        "gold_sha256",
        "rubric_sha256",
        "artifact_bundle_sha256",
    ],
)
def test_reviewer_qualification_review_rejects_floating_or_wrong_bindings(
    tmp_path: Path,
    binding_field: str,
):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    scorer_review_path = tmp_path / "scorer-review.json"
    reviewer_receipt_path = tmp_path / "reviewer-receipt.json"
    _write_json(sealed_path, _sealed_scorer_input())
    scorer_review = scorer.build_scorer_review(sealed_path)
    _write_json(scorer_review_path, scorer_review)
    receipt = _reviewer_receipt_for_scorer_review(scorer_review_path, scorer_review)
    receipt[binding_field] = "0" * 64
    _write_json(reviewer_receipt_path, receipt)

    with pytest.raises(ValueError, match=f"{binding_field} binding mismatch"):
        scorer.build_reviewer_qualification_review(
            scorer_review_path,
            reviewer_receipt_path,
        )


def test_scorer_cli_writes_reviewer_qualification_review(tmp_path: Path):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    scorer_review_path = tmp_path / "scorer-review.json"
    reviewer_receipt_path = tmp_path / "reviewer-receipt.json"
    qualification_path = tmp_path / "qualification-review.json"
    _write_json(sealed_path, _sealed_scorer_input())
    scorer_review = scorer.build_scorer_review(sealed_path)
    _write_json(scorer_review_path, scorer_review)
    _write_json(
        reviewer_receipt_path,
        _reviewer_receipt_for_scorer_review(scorer_review_path, scorer_review),
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_scorer",
            str(scorer_review_path),
            "--qualification-review",
            "--reviewer-receipt",
            str(reviewer_receipt_path),
            "--output",
            str(qualification_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    review = contract.load_json_no_duplicates(qualification_path)
    scorer.validate_reviewer_qualification_review(review)
    assert review["external_cost_authorized"] is False


def test_documented_scorer_then_qualification_cli_path_uses_canonical_files(
    tmp_path: Path,
):
    sealed_path = tmp_path / "sealed-scorer-input.json"
    scorer_review_path = tmp_path / "scorer-review.json"
    reviewer_receipt_path = tmp_path / "reviewer-receipt.json"
    qualification_path = tmp_path / "qualification-review.json"
    _write_json(sealed_path, _sealed_scorer_input())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_scorer",
            str(sealed_path),
            "--output",
            str(scorer_review_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    scorer_review = contract.load_json_no_duplicates(scorer_review_path)
    assert scorer_review_path.read_bytes() == (
        contract.canonical_json_bytes(scorer_review) + b"\n"
    )
    _write_json(
        reviewer_receipt_path,
        _reviewer_receipt_for_scorer_review(scorer_review_path, scorer_review),
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trec_rag.det_sparse_v4_scorer",
            str(scorer_review_path),
            "--qualification-review",
            "--reviewer-receipt",
            str(reviewer_receipt_path),
            "--output",
            str(qualification_path),
        ],
        check=True,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    review = contract.load_json_no_duplicates(qualification_path)
    scorer.validate_reviewer_qualification_review(review)
    assert review["terminal_state"] == "completed_synthetic_go"

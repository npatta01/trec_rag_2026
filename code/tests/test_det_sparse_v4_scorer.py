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


def _sealed_scorer_input(*, terminal_state: str = "raw_sealed_pending_scorer"):
    gold = contract.load_json_no_duplicates(
        contract.ARTIFACT_DIR / "semantic_anchor_gold_labels_v1.json"
    )
    artifact_sha256 = contract.validate_artifact_bundle()
    responses = []
    for index, case in enumerate(gold["cases"], start=1):
        response = _gold_driven_response(case)
        responses.append(
            {
                "case_id": case["case_id"],
                "raw_response_sha256": f"{index:064x}"[-64:],
                "response": response,
            }
        )
    return {
        "schema_version": "semantic_anchor_sealed_scorer_input_v1",
        "terminal_receipt": {
            "schema_version": "semantic_anchor_terminal_receipt_v1",
            "terminal_state": terminal_state,
            "attempted_calls": 24,
            "completed_calls": 24,
            "raw_committed_calls": 24,
            "gold_opened": False,
            "artifact_sha256": artifact_sha256,
        },
        "responses": responses,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(contract.canonical_json_bytes(payload) + b"\n")


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
    with pytest.raises(ValueError, match="raw_sealed_pending_scorer"):
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


def test_scorer_review_classifies_malformed_sealed_response(tmp_path: Path):
    sealed = _sealed_scorer_input()
    sealed["responses"][0]["response"]["start_token"] = True
    sealed_path = tmp_path / "sealed-scorer-input.json"
    _write_json(sealed_path, sealed)

    review = scorer.build_scorer_review(sealed_path)

    assert review["scorer_receipt"]["case_results"][0] == {
        "case_id": "synthetic-case-001",
        "classification": "mechanical_failure",
    }
    assert review["scorer_receipt"]["case_results"][1]["classification"] == "correct_select"


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

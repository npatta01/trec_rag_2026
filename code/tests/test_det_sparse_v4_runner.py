from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_runner as runner


def test_pre_dispatch_no_go_runner_writes_create_only_zero_call_boundary(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"

    result = runner.build_pre_dispatch_no_go_run(output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_manifest_path == output_dir.resolve() / "run_manifest.json"
    assert result.terminal_receipt_path == output_dir.resolve() / "terminal_receipt.json"
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(result.terminal_receipt_path.read_text(encoding="utf-8"))
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
    contract.validate_ledger_prefix(
        case_order=tuple(f"synthetic-case-{index:03d}" for index in range(1, 25)),
        reservations=[],
        dispatches=[],
        raw_responses=[],
        transport_failures=[],
        case_receipts=[],
        terminal_receipt=terminal,
        run_manifest=manifest,
    )


def test_pre_dispatch_no_go_runner_refuses_existing_output_dir(tmp_path: Path):
    output_dir = tmp_path / "v4-no-go"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        runner.build_pre_dispatch_no_go_run(output_dir)

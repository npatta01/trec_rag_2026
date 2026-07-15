from __future__ import annotations

from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_local_model_r2 as local_model_r2
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
from trec_rag.adaptive_obligation_v2_local_model_r2 import (
    R2_APPROVAL_SCHEMA_VERSION,
    R2LocalJsonModel,
    verify_r2_inference_approval,
)
from trec_rag.adaptive_obligation_v2_propose import MODEL_ID, MODEL_REVISION
from trec_rag.adaptive_obligation_v2_propose_r2 import R2_PROPOSAL_SCHEMA


def _preflight(ledger_dir: Path) -> dict[str, object]:
    jobs = []
    for index in range(48):
        messages = [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": f"frozen R2 job {index}"},
        ]
        jobs.append(
            {
                "job_id": f"{index + 1:064x}",
                "messages": messages,
                "messages_sha256": canonical_sha256(messages),
                "prompt_token_count": index + 11,
            }
        )
    return {
        "receipt_sha256": "a" * 64,
        "ledger_dir": str(ledger_dir),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "model_snapshot": {"manifest_sha256": "b" * 64},
        "jobs": jobs,
    }


def _approval(
    preflight: dict[str, object], ledger_dir: Path
) -> dict[str, object]:
    return {
        "schema_version": R2_APPROVAL_SCHEMA_VERSION,
        "stage": "proposal_r2",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "ledger_dir": str(ledger_dir),
        "approved": True,
    }


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_json_bytes(
        self,
        messages: object,
        schema: object,
        **kwargs: object,
    ) -> tuple[bytes, int]:
        self.calls.append(
            {"messages": messages, "schema": schema, **kwargs}
        )
        return (
            b'{"status":"UNSUPPORTED","reason_code":"NO_ABSTRACT_CHILD","o1":null}',
            17,
        )


def test_r2_approval_binds_absolute_ledger_destination(tmp_path: Path) -> None:
    approved_ledger = tmp_path / "approved-ledger"
    preflight = _preflight(approved_ledger)
    with pytest.raises(PermissionError, match="ledger"):
        verify_r2_inference_approval(
            _approval(preflight, approved_ledger),
            preflight,
            ledger_dir=tmp_path / "different-ledger",
        )


def test_r2_approval_rejects_relative_or_symlinked_ledger_destination(
    tmp_path: Path,
) -> None:
    relative = Path("relative-ledger")
    preflight = _preflight(relative)
    with pytest.raises(PermissionError, match="absolute safe ledger"):
        verify_r2_inference_approval(
            _approval(preflight, relative),
            preflight,
            ledger_dir=relative,
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_ledger = linked_parent / "ledger"
    linked_preflight = _preflight(linked_ledger)
    with pytest.raises(PermissionError, match="absolute safe ledger"):
        verify_r2_inference_approval(
            _approval(linked_preflight, linked_ledger),
            linked_preflight,
            ledger_dir=linked_ledger,
        )


def test_r1_approval_mapping_cannot_authorize_r2(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    preflight = _preflight(ledger_dir)
    approval = _approval(preflight, ledger_dir)
    approval.update(
        {
            "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
            "stage": "proposal",
        }
    )
    approval.pop("ledger_dir")
    with pytest.raises(PermissionError, match="R2 proposal approval required"):
        verify_r2_inference_approval(
            approval, preflight, ledger_dir=ledger_dir
        )


def test_r2_model_rejects_duplicate_message_hash_before_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    ledger_dir = tmp_path / "ledger"
    preflight = _preflight(ledger_dir)
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    jobs[1]["messages"] = jobs[0]["messages"]
    jobs[1]["messages_sha256"] = jobs[0]["messages_sha256"]
    monkeypatch.setattr(
        local_model_r2,
        "load_pinned_local_runtime",
        lambda **_kwargs: touched.append("runtime"),
    )
    with pytest.raises(ValueError, match="unique frozen message hashes"):
        R2LocalJsonModel(
            approval=_approval(preflight, ledger_dir),
            preflight=preflight,
            ledger_dir=ledger_dir,
        )
    assert touched == []


def test_r2_model_passes_exact_frozen_prompt_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(
        local_model_r2,
        "load_pinned_local_runtime",
        lambda **_kwargs: runtime,
    )
    ledger_dir = tmp_path / "ledger"
    preflight = _preflight(ledger_dir)
    model = R2LocalJsonModel(
        approval=_approval(preflight, ledger_dir),
        preflight=preflight,
        ledger_dir=ledger_dir,
    )
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    job = jobs[0]
    model.generate(job["messages"], R2_PROPOSAL_SCHEMA, max_new_tokens=256)
    assert runtime.calls[0]["expected_prompt_tokens"] == job[
        "prompt_token_count"
    ]
    assert runtime.calls[0]["do_sample"] is False


def test_r2_model_rejects_unfrozen_schema_and_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(
        local_model_r2,
        "load_pinned_local_runtime",
        lambda **_kwargs: runtime,
    )
    ledger_dir = tmp_path / "ledger"
    preflight = _preflight(ledger_dir)
    model = R2LocalJsonModel(
        approval=_approval(preflight, ledger_dir),
        preflight=preflight,
        ledger_dir=ledger_dir,
    )
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    with pytest.raises(ValueError, match="schema differs"):
        model.generate(jobs[0]["messages"], {}, max_new_tokens=256)
    with pytest.raises(ValueError, match="messages differ"):
        model.generate([], R2_PROPOSAL_SCHEMA, max_new_tokens=256)
    assert runtime.calls == []

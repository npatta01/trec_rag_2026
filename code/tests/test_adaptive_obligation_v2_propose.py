from __future__ import annotations

import json
import hashlib
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import trec_rag.adaptive_obligation_v2_propose as propose_module
import trec_rag.adaptive_obligation_v2_local_model as local_model_module
from trec_rag.adaptive_evidence_contract import PILOT_TOPIC_IDS
from trec_rag.adaptive_obligation_v2_contract import (
    PARENT_SCHEMA_VERSION,
    RESERVOIR_SCHEMA_VERSION,
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    UNIT_SCHEMA_VERSION,
    canonical_sha256,
    sha256_text,
)
from trec_rag.adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    PROPOSAL_INVENTORY_SCHEMA_VERSION,
    PROPOSAL_RECEIPT_SCHEMA_VERSION,
    PROPOSAL_SCHEMA,
    _snapshot_inventory,
    _load_local_tokenizer,
    build_proposal_jobs,
    build_proposal_preflight,
    execute_proposals,
    finalize_proposal_inventory,
    main,
    run_job_with_retry,
    verify_proposal_preflight,
)
from trec_rag.adaptive_obligation_v2_validate import build_validation_preflight


def _contract_fixture() -> dict[str, object]:
    parents: list[dict[str, object]] = []
    reservoirs: list[dict[str, object]] = []
    units: list[dict[str, object]] = []
    for index in range(24):
        topic_id = PILOT_TOPIC_IDS[index % len(PILOT_TOPIC_IDS)]
        parent_id = f"{topic_id}-parent-{index:02d}"
        text = f"Complete O0 obligation {index}."
        narrative = f"Unchanged narrative for topic {topic_id}."
        query = f"{narrative}\n\nExplicit obligation:\n{text}"
        parents.append(
            {
                "schema_version": PARENT_SCHEMA_VERSION,
                "topic_id": topic_id,
                "parent_id": parent_id,
                "manifest_order": index,
                "text": text,
                "text_sha256": sha256_text(text),
                "query": query,
                "query_sha256": sha256_text(query),
            }
        )
        for fold in (0, 1):
            unit_ids: list[str] = []
            documents: list[dict[str, object]] = []
            for document_index in range(10):
                document_id = f"{parent_id}-f{fold}-d{document_index}"
                window_id = f"{document_id}-window"
                unit_text = f"Exact evidence {index} {fold} {document_index}."
                identity = {
                    "topic_id": topic_id,
                    "parent_id": parent_id,
                    "fold": fold,
                    "document_id": document_id,
                    "window_id": window_id,
                    "start": 0,
                    "end": len(unit_text),
                    "text": unit_text,
                }
                unit_id = canonical_sha256(identity)
                unit_ids.append(unit_id)
                units.append(
                    {
                        "schema_version": UNIT_SCHEMA_VERSION,
                        "unit_id": unit_id,
                        **identity,
                        "text_sha256": sha256_text(unit_text),
                    }
                )
                documents.append(
                    {
                        "rank": document_index + 1,
                        "document_id": document_id,
                        "document_sha256": "d" * 64,
                        "window_id": window_id,
                        "window_text": unit_text,
                        "window_sha256": sha256_text(unit_text),
                        "document_start_token": 0,
                        "document_end_token": 8,
                        "score": float(10 - document_index),
                        "unit_ids": [unit_id],
                    }
                )
            reservoirs.append(
                {
                    "schema_version": RESERVOIR_SCHEMA_VERSION,
                    "reservoir_id": canonical_sha256(
                        {
                            "topic_id": topic_id,
                            "parent_id": parent_id,
                            "fold": fold,
                            "document_ids": [row["document_id"] for row in documents],
                            "window_ids": [row["window_id"] for row in documents],
                        }
                    ),
                    "topic_id": topic_id,
                    "parent_id": parent_id,
                    "fold": fold,
                    "document_count": 10,
                    "document_ids": [row["document_id"] for row in documents],
                    "window_ids": [row["window_id"] for row in documents],
                    "documents": documents,
                }
            )
    receipt = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "parent_count": 24,
        "reservoir_count": 48,
        "unit_count": len(units),
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "inference_count": 0,
        "external_cost_usd": 0.0,
    }
    return {
        "parents": parents,
        "reservoirs": reservoirs,
        "units": units,
        "receipt": receipt,
    }


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.calls += 1
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(len(json.dumps(messages).split())))


def _fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _snapshot_fixture() -> dict[str, object]:
    files = [
        {
            "name": name,
            "bytes": index + 1,
            "blob_id": f"blob-{index}",
            "content_sha256": f"{index:x}" * 64,
        }
        for index, name in enumerate(
            ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
        )
    ]
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _fixture_source(
    tmp_path: Path, contract: dict[str, object], *, name: str
) -> tuple[Path, str]:
    root = tmp_path / name
    root.mkdir()
    receipt_bytes = _pretty(contract["receipt"])
    (root / "receipt.json").write_bytes(receipt_bytes)
    return root, hashlib.sha256(receipt_bytes).hexdigest()


def _build_fixture_preflight(
    tmp_path: Path,
    *,
    tokenizer: object,
    model_factory: object = None,
    name: str = "preflight",
    contract: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    source = contract or _contract_fixture()
    source_dir, receipt_sha256 = _fixture_source(
        tmp_path, source, name=f"{name}-contract"
    )
    output = tmp_path / name
    with patch.object(propose_module, "_load_verified_contract", return_value=source):
        receipt = build_proposal_preflight(
            source,
            tokenizer=tokenizer,
            model_factory=model_factory,  # type: ignore[arg-type]
            output_dir=output,
            contract_dir=source_dir,
            contract_receipt_sha256=receipt_sha256,
            model_snapshot=_snapshot_fixture(),
        )
    return receipt, source, source_dir, output


def _patch_fixture_verifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: dict[str, object],
    receipt: dict[str, object],
    tokenizer: object,
) -> None:
    monkeypatch.setattr(propose_module, "_load_verified_contract", lambda _path: contract)
    monkeypatch.setattr(
        propose_module, "_snapshot_inventory", lambda: receipt["model_snapshot"]
    )
    monkeypatch.setattr(
        propose_module, "_tokenizer_contract", lambda: receipt["tokenizer_contract"]
    )
    monkeypatch.setattr(propose_module, "_load_local_tokenizer", lambda: tokenizer)


def _expected_pairs() -> set[tuple[str, int]]:
    contract = _contract_fixture()
    return {
        (str(row["parent_id"]), int(row["fold"]))
        for row in contract["reservoirs"]  # type: ignore[index]
    }


def test_schema_is_one_o1_or_unsupported() -> None:
    assert PROPOSAL_SCHEMA["additionalProperties"] is False
    assert set(PROPOSAL_SCHEMA["properties"]["status"]["enum"]) == {
        "SUPPORTED",
        "UNSUPPORTED",
    }
    assert "n1" not in json.dumps(PROPOSAL_SCHEMA).casefold()


def test_jobs_are_exactly_parent_by_fold() -> None:
    jobs = build_proposal_jobs(_contract_fixture())
    assert len(jobs) == 48
    assert len({row["job_id"] for row in jobs}) == 48
    assert {(row["parent_id"], row["fold"]) for row in jobs} == _expected_pairs()


def test_preflight_never_constructs_a_model(tmp_path: Path) -> None:
    touched: list[str] = []
    receipt, _contract, _source, _output = _build_fixture_preflight(
        tmp_path,
        tokenizer=_fake_tokenizer(),
        model_factory=lambda: touched.append("model"),
    )
    assert touched == []
    assert receipt["primary_call_count"] == 48
    assert receipt["retry_call_ceiling"] == 48
    assert receipt["worst_case_call_ceiling"] == 96
    assert receipt["tokenizer_load_count"] == 1
    assert receipt["model_load_count"] == 0
    assert receipt["inference_count"] == 0


def test_messages_bind_narrative_complete_o0_units_and_schema() -> None:
    contract = _contract_fixture()
    job = build_proposal_jobs(contract)[0]
    messages = job["messages"]
    assert isinstance(messages, list)
    prompt = "\n".join(str(row["content"]) for row in messages)
    parent = contract["parents"][0]  # type: ignore[index]
    assert "Unchanged narrative for topic" in prompt
    assert parent["text"] in prompt
    assert job["input_unit_ids"][0] in prompt
    assert json.dumps(PROPOSAL_SCHEMA, separators=(",", ":"), sort_keys=True) in prompt
    folded = prompt.casefold()
    assert "outside knowledge" in folded
    assert "answer facts" in folded
    assert "support_unit_ids" in folded
    assert "unsupported" in folded


def test_preflight_seals_snapshot_tokenizer_prompt_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _load_local_tokenizer()
    receipt, contract, _source, output = _build_fixture_preflight(
        tmp_path,
        tokenizer=_load_local_tokenizer(),
        model_factory=lambda: pytest.fail("model factory must remain unreachable"),
    )
    assert set(path.name for path in output.iterdir()) == {
        "jobs.jsonl",
        "schema.json",
        "prompt.json",
        "receipt.json",
    }
    assert receipt["model_snapshot"]["model"] == MODEL_ID
    assert receipt["model_snapshot"]["revision"] == MODEL_REVISION
    assert len(receipt["model_snapshot"]["manifest_sha256"]) == 64
    assert {row["name"] for row in receipt["tokenizer_files"]} >= {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    tokenizer_contract = receipt["tokenizer_contract"]
    assert tokenizer_contract["loader"] == "tokenizers.Tokenizer.from_file"
    assert len(tokenizer_contract["tokenizer_json_sha256"]) == 64
    assert len(tokenizer_contract["tokenizer_config_sha256"]) == 64
    assert len(tokenizer_contract["chat_template_sha256"]) == 64
    assert hashlib.sha256(tokenizer_contract["chat_template"].encode()).hexdigest() == tokenizer_contract["chat_template_sha256"]
    assert receipt["prompt_token_counts"]["count"] == 48
    assert set(receipt["code_sha256"]) == {
        "adaptive_obligation_v2_contract.py",
        "adaptive_obligation_v2_propose.py",
    }
    _patch_fixture_verifier(
        monkeypatch, contract=contract, receipt=receipt, tokenizer=tokenizer
    )
    verified = verify_proposal_preflight(output)
    assert verified == receipt


def test_protected_topic_fails_before_tokenizer_or_output(tmp_path: Path) -> None:
    contract = _contract_fixture()
    contract["parents"][0]["topic_id"] = "144"  # type: ignore[index]
    tokenizer = _fake_tokenizer()
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="protected"):
        build_proposal_preflight(
            contract,
            tokenizer=tokenizer,
            model_factory=lambda: pytest.fail("model factory must remain unreachable"),
            output_dir=output,
            contract_dir=tmp_path / "must-not-open",
            contract_receipt_sha256="a" * 64,
            model_snapshot=_snapshot_fixture(),
        )
    assert tokenizer.calls == 0
    assert not output.exists()


def test_preflight_is_create_only_before_retokenizing(tmp_path: Path) -> None:
    first = _fake_tokenizer()
    _receipt, contract, source, output = _build_fixture_preflight(
        tmp_path, tokenizer=first
    )
    second = _fake_tokenizer()
    with pytest.raises(FileExistsError, match="create-only"):
        build_proposal_preflight(
            contract,
            tokenizer=second,
            model_factory=None,
            output_dir=output,
            contract_dir=source,
            contract_receipt_sha256=hashlib.sha256(
                (source / "receipt.json").read_bytes()
            ).hexdigest(),
            model_snapshot=_snapshot_fixture(),
        )
    assert first.calls == 48
    assert second.calls == 0


def test_verifier_rejects_job_tampering(tmp_path: Path) -> None:
    _receipt, _contract, _source, output = _build_fixture_preflight(
        tmp_path,
        tokenizer=_load_local_tokenizer(),
    )
    with (output / "jobs.jsonl").open("ab") as sink:
        sink.write(b"{}\n")
    with pytest.raises(ValueError, match="jobs.jsonl"):
        verify_proposal_preflight(output)


def test_verifier_recomputes_resealed_prompt_token_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _load_local_tokenizer()
    receipt, contract, _source, output = _build_fixture_preflight(
        tmp_path, tokenizer=tokenizer
    )
    rows = [json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]
    rows[0]["prompt_token_count"] += 1
    jobs_bytes = b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
        for row in rows
    )
    (output / "jobs.jsonl").write_bytes(jobs_bytes)
    receipt = json.loads((output / "receipt.json").read_bytes())
    receipt["artifacts"]["jobs.jsonl"].update(
        {"bytes": len(jobs_bytes), "sha256": hashlib.sha256(jobs_bytes).hexdigest()}
    )
    counts = [row["prompt_token_count"] for row in rows]
    receipt["prompt_token_counts"] = {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "total": sum(counts),
        "by_job": [
            {"job_id": row["job_id"], "prompt_token_count": row["prompt_token_count"]}
            for row in rows
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _patch_fixture_verifier(
        monkeypatch, contract=contract, receipt=receipt, tokenizer=tokenizer
    )
    with pytest.raises(ValueError, match="recomputed"):
        verify_proposal_preflight(output)


def test_execute_rejects_before_model_or_output_without_approval(
    tmp_path: Path,
) -> None:
    touched: list[str] = []
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        execute_proposals(
            preflight_dir=tmp_path / "missing",
            approval_path=tmp_path / "missing-approval.json",
            output_dir=tmp_path / "out",
            model_factory=lambda: touched.append("model"),
        )
    assert touched == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("approval_kind", ["malformed", "symlink", "wrong-stage"])
def test_bad_approval_fails_before_preflight_model_or_output_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_kind: str,
) -> None:
    approval_path = tmp_path / "approval.json"
    if approval_kind == "malformed":
        approval_path.write_bytes(b"not JSON")
    else:
        approval = {
            "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
            "stage": "validation" if approval_kind == "wrong-stage" else "proposal",
            "preflight_sha256": "a" * 64,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "primary_call_count": 48,
            "retry_call_ceiling": 48,
            "approved": True,
        }
        target = tmp_path / "approval-target.json"
        target.write_bytes(_pretty(approval))
        if approval_kind == "symlink":
            approval_path.symlink_to(target)
        else:
            approval_path.write_bytes(target.read_bytes())
    touched: list[str] = []
    monkeypatch.setattr(
        propose_module,
        "verify_proposal_preflight",
        lambda _path: touched.append("preflight"),
    )
    monkeypatch.setattr(
        propose_module,
        "_load_local_tokenizer",
        lambda: touched.append("tokenizer"),
    )
    monkeypatch.setattr(
        propose_module,
        "_path_present",
        lambda _path: touched.append("output"),
    )
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        execute_proposals(
            preflight_dir=tmp_path / "preflight-sentinel",
            approval_path=approval_path,
            output_dir=tmp_path / "output-sentinel",
            model_factory=lambda: touched.append("model"),
        )
    assert touched == []
    assert not (tmp_path / "output-sentinel").exists()


def _approval_record(preflight_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
        "stage": "proposal",
        "preflight_sha256": preflight_sha256,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "approved": True,
    }


def test_approval_capture_accepts_audit_metadata_but_rejects_required_field_changes(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    approval = real / "approval.json"
    approval.write_bytes(_pretty(_approval_record("a" * 64)))
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        propose_module._capture_inference_approval(linked_parent / "approval.json")

    audited = {
        **_approval_record("a" * 64),
        "created_by": "independent-approver",
        "audit": {"ticket": "RAG-2026"},
    }
    approval.write_bytes(_pretty(audited))
    captured = propose_module._capture_inference_approval(approval)
    assert captured.value == audited
    assert captured.sha256 == hashlib.sha256(_pretty(audited)).hexdigest()

    for wrong_required in (
        {"primary_call_count": 48.0},
        {"retry_call_ceiling": True},
        {"approved": 1},
        {"approved": False},
        {"model": "wrong/model"},
    ):
        changed = {**_approval_record("a" * 64), **wrong_required}
        approval.write_bytes(_pretty(changed))
        with pytest.raises(PermissionError, match="proposal inference approval required"):
            propose_module._capture_inference_approval(approval)


def _captured_preflight_fixture(tmp_path: Path) -> tuple[Path, bytes, list[dict[str, object]]]:
    root = tmp_path / "preflight"
    root.mkdir()
    jobs = [
        {
            "job_id": f"{index:064x}",
            "messages": [],
            "messages_sha256": canonical_sha256([]),
            "input_unit_ids": [],
            "primary_max_new_tokens": 256,
            "retry_max_new_tokens": 512,
        }
        for index in range(48)
    ]
    jobs_bytes = b"".join(
        (
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        for row in jobs
    )
    receipt_bytes = _pretty({"marker": "captured"})
    (root / "jobs.jsonl").write_bytes(jobs_bytes)
    (root / "schema.json").write_bytes(_pretty(PROPOSAL_SCHEMA))
    (root / "prompt.json").write_bytes(_pretty({"prompt": "captured"}))
    (root / "receipt.json").write_bytes(receipt_bytes)
    return root, receipt_bytes, jobs


def test_preflight_capture_validates_isolated_bytes_and_never_reopens_live_jobs(
    tmp_path: Path,
) -> None:
    root, receipt_bytes, jobs = _captured_preflight_fixture(tmp_path)
    original_jobs = (root / "jobs.jsonl").read_bytes()

    def verify_isolated(path: Path) -> dict[str, object]:
        (root / "jobs.jsonl").write_bytes(b'{"job_id":"live-race"}\n')
        assert (path / "jobs.jsonl").read_bytes() == original_jobs
        return {"marker": "captured"}

    captured = propose_module._capture_and_verify_preflight(
        root,
        expected_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        verifier=verify_isolated,
    )
    assert captured.jobs == jobs
    assert captured.receipt_sha256 == hashlib.sha256(receipt_bytes).hexdigest()
    assert (root / "jobs.jsonl").read_bytes() != original_jobs


def test_preflight_capture_rejects_a_symlinked_parent_before_verifier(
    tmp_path: Path,
) -> None:
    root, receipt_bytes, _jobs = _captured_preflight_fixture(tmp_path)
    linked = tmp_path / "linked-preflight"
    linked.symlink_to(root, target_is_directory=True)
    touched: list[str] = []
    with pytest.raises(ValueError, match="preflight.*unsafe"):
        propose_module._capture_and_verify_preflight(
            linked,
            expected_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            verifier=lambda _path: touched.append("verify"),
        )
    assert touched == []


def _execution_capture(tmp_path: Path) -> tuple[object, object]:
    _root, _receipt_bytes, jobs = _captured_preflight_fixture(tmp_path)
    preflight_sha256 = "d" * 64
    approval_value = _approval_record(preflight_sha256)
    approval_source = _pretty(approval_value)
    approval = propose_module._CapturedApproval(
        value=approval_value,
        source=approval_source,
        sha256=hashlib.sha256(approval_source).hexdigest(),
    )
    receipt = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "model_snapshot": {"manifest_sha256": "f" * 64},
    }
    preflight = propose_module._CapturedPreflight(
        receipt=receipt,
        receipt_sha256=preflight_sha256,
        jobs=jobs,
        contents={},
    )
    return approval, preflight


def _sealed_proposal_handoff_fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    object,
    object,
    Path,
    list[dict[str, object]],
]:
    contract = _contract_fixture()
    jobs = build_proposal_jobs(contract)
    preflight_sha256 = "d" * 64
    approval_value = _approval_record(preflight_sha256)
    approval_source = _pretty(approval_value)
    approval = propose_module._CapturedApproval(
        value=approval_value,
        source=approval_source,
        sha256=hashlib.sha256(approval_source).hexdigest(),
    )
    jobs_source = b"".join(propose_module._compact_bytes(job) for job in jobs)
    preflight_receipt = {
        "schema_version": propose_module.SCHEMA_VERSION,
        "status": "complete",
        "job_count": 48,
        "schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "contract_receipt": {"sha256": "c" * 64},
        "artifacts": {
            "jobs.jsonl": {
                "path": "jobs.jsonl",
                "rows": 48,
                "bytes": len(jobs_source),
                "sha256": hashlib.sha256(jobs_source).hexdigest(),
            }
        },
    }
    preflight = propose_module._CapturedPreflight(
        receipt=preflight_receipt,
        receipt_sha256=preflight_sha256,
        jobs=jobs,
        contents={"jobs.jsonl": jobs_source},
    )
    anchor = propose_module._build_run_anchor(
        jobs,
        preflight_sha256=preflight_sha256,
        approval_sha256=approval.sha256,
    )
    ledger_root = tmp_path / "proposal-ledger"
    ledger = propose_module.AppendOnlyAttemptLedger(
        ledger_root,
        expected_anchor=anchor,
        create_only=True,
    )
    first_support = str(jobs[0]["input_unit_ids"][0])  # type: ignore[index]
    for index, job in enumerate(jobs):
        value = (
            {
                "status": "SUPPORTED",
                "reason_code": "SUPPORTED",
                "o1": {
                    "label": "demographic screening audit need",
                    "scope_rationale": "Independent evidence names a reusable need.",
                    "support_unit_ids": [first_support],
                },
            }
            if index == 0
            else {
                "status": "UNSUPPORTED",
                "reason_code": "NO_ABSTRACT_CHILD",
                "o1": None,
            }
        )
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        run_job_with_retry(
            job,
            generate=lambda _ceiling, result=raw: (result, 20),
            ledger=ledger,
        )
    ledger.seal_completion()
    return contract, approval, preflight, ledger_root, jobs


def test_sealed_ledger_finalizer_feeds_authenticated_task4_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, approval, preflight, ledger_root, jobs = (
        _sealed_proposal_handoff_fixture(tmp_path)
    )
    monkeypatch.setattr(
        propose_module, "_capture_inference_approval", lambda _path: approval
    )
    monkeypatch.setattr(
        propose_module,
        "_capture_and_verify_preflight",
        lambda _path, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        local_model_module, "verify_inference_approval", lambda _approval, _preflight: None
    )
    inventory_root = tmp_path / "proposal-inventory"
    receipt = finalize_proposal_inventory(
        preflight_dir=tmp_path / "captured-preflight",
        approval_path=tmp_path / "captured-approval",
        ledger_dir=ledger_root,
        output_dir=inventory_root,
    )
    proposals = [
        json.loads(line)
        for line in (inventory_root / "proposals.jsonl").read_bytes().splitlines()
    ]
    assert set(path.name for path in inventory_root.iterdir()) == {
        "proposals.jsonl",
        "receipt.json",
    }
    assert receipt["schema_version"] == PROPOSAL_RECEIPT_SCHEMA_VERSION
    assert receipt["proposal_count"] == 48
    assert receipt["supported_count"] == 1
    assert receipt["run_anchor_sha256"] == json.loads(
        (ledger_root / "completion.json").read_bytes()
    )["anchor_sha256"]
    assert proposals[0]["schema_version"] == PROPOSAL_INVENTORY_SCHEMA_VERSION
    assert proposals[0]["job_id"] == jobs[0]["job_id"]
    assert proposals[0]["topic_id"] == jobs[0]["topic_id"]
    assert proposals[0]["parent_id"] == jobs[0]["parent_id"]
    assert proposals[0]["proposal_fold"] == jobs[0]["fold"]
    assert proposals[0]["parent_manifest_order"] == jobs[0]["parent_manifest_order"]
    assert proposals[0]["label"] == "demographic screening audit need"
    assert proposals[0]["support_unit_ids"] == [jobs[0]["input_unit_ids"][0]]
    assert proposals[0]["support_units"][0]["text_sha256"]
    assert proposals[0]["raw_completion_sha256"]

    with pytest.raises(FileExistsError, match="create-only proposal inventory"):
        finalize_proposal_inventory(
            preflight_dir=tmp_path / "captured-preflight",
            approval_path=tmp_path / "captured-approval",
            ledger_dir=ledger_root,
            output_dir=inventory_root,
        )

    real_parent = tmp_path / "real-inventory-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-inventory-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent.*unsafe"):
        finalize_proposal_inventory(
            preflight_dir=tmp_path / "captured-preflight",
            approval_path=tmp_path / "captured-approval",
            ledger_dir=ledger_root,
            output_dir=linked_parent / "inventory",
        )
    assert not (real_parent / "inventory").exists()

    validation = build_validation_preflight(
        contract,
        proposal_inventory_dir=inventory_root,
        proposal_preflight_dir=tmp_path / "captured-preflight",
        proposal_ledger_dir=ledger_root,
    )
    assert validation["job_count"] == 1
    assert validation["proposal_receipt"]["sha256"] == hashlib.sha256(
        (inventory_root / "receipt.json").read_bytes()
    ).hexdigest()

    linked_inventory = tmp_path / "linked-proposal-inventory"
    linked_inventory.symlink_to(inventory_root, target_is_directory=True)
    with pytest.raises(ValueError, match="inventory.*unsafe"):
        build_validation_preflight(
            contract,
            proposal_inventory_dir=linked_inventory,
            proposal_preflight_dir=tmp_path / "captured-preflight",
            proposal_ledger_dir=ledger_root,
        )

    forged_root = tmp_path / "forged-inventory"
    shutil.copytree(inventory_root, forged_root)
    forged = [dict(row) for row in proposals]
    forged[0]["label"] = "attacker changed label"
    identity = {
        key: value
        for key, value in forged[0].items()
        if key not in {"schema_version", "proposal_id"}
    }
    forged[0]["proposal_id"] = canonical_sha256(identity)
    forged_source = b"".join(propose_module._compact_bytes(row) for row in forged)
    (forged_root / "proposals.jsonl").write_bytes(forged_source)
    forged_receipt = json.loads((forged_root / "receipt.json").read_bytes())
    forged_receipt["proposals_sha256"] = hashlib.sha256(forged_source).hexdigest()
    forged_receipt["artifacts"]["proposals.jsonl"] = {
        "path": "proposals.jsonl",
        "rows": len(forged),
        "bytes": len(forged_source),
        "sha256": hashlib.sha256(forged_source).hexdigest(),
    }
    (forged_root / "receipt.json").write_bytes(_pretty(forged_receipt))
    with pytest.raises(ValueError, match="sealed ledger"):
        build_validation_preflight(
            contract,
            proposal_inventory_dir=forged_root,
            proposal_preflight_dir=tmp_path / "captured-preflight",
            proposal_ledger_dir=ledger_root,
        )

    missing_job_preflight = propose_module._CapturedPreflight(
        receipt=preflight.receipt,
        receipt_sha256=preflight.receipt_sha256,
        jobs=preflight.jobs[:-1],
        contents=preflight.contents,
    )
    monkeypatch.setattr(
        propose_module,
        "_capture_and_verify_preflight",
        lambda _path, **_kwargs: missing_job_preflight,
    )
    with pytest.raises(ValueError, match="exactly 48"):
        finalize_proposal_inventory(
            preflight_dir=tmp_path / "captured-preflight",
            approval_path=tmp_path / "captured-approval",
            ledger_dir=ledger_root,
            output_dir=tmp_path / "missing-job-inventory",
        )

    reordered_preflight = propose_module._CapturedPreflight(
        receipt=preflight.receipt,
        receipt_sha256=preflight.receipt_sha256,
        jobs=list(reversed(preflight.jobs)),
        contents=preflight.contents,
    )
    monkeypatch.setattr(
        propose_module,
        "_capture_and_verify_preflight",
        lambda _path, **_kwargs: reordered_preflight,
    )
    with pytest.raises(ValueError, match="anchor"):
        finalize_proposal_inventory(
            preflight_dir=tmp_path / "captured-preflight",
            approval_path=tmp_path / "captured-approval",
            ledger_dir=ledger_root,
            output_dir=tmp_path / "reordered-job-inventory",
        )


def test_finalize_cli_requires_approval_before_output(tmp_path: Path) -> None:
    output = tmp_path / "proposal-inventory"
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        main(
            [
                "finalize",
                "--preflight",
                str(tmp_path / "missing-preflight"),
                "--approval",
                str(tmp_path / "missing-approval"),
                "--ledger",
                str(tmp_path / "missing-ledger"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_execute_atomically_claims_output_before_model_and_second_executor_loses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval, preflight = _execution_capture(tmp_path)
    monkeypatch.setattr(
        propose_module, "_capture_inference_approval", lambda _path: approval
    )
    monkeypatch.setattr(
        propose_module,
        "_capture_and_verify_preflight",
        lambda _path, **_kwargs: preflight,
    )
    output = tmp_path / "proposals"
    entered = threading.Event()
    release = threading.Event()
    first_failures: list[BaseException] = []

    def first_factory() -> object:
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("stop after atomic claim")

    def run_first() -> None:
        try:
            execute_proposals(
                preflight_dir=tmp_path / "unused-preflight",
                approval_path=tmp_path / "unused-approval",
                output_dir=output,
                model_factory=first_factory,
            )
        except BaseException as exc:
            first_failures.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=5)
    assert output.is_dir()
    second_model: list[str] = []
    with pytest.raises(FileExistsError, match="create-only proposal output"):
        execute_proposals(
            preflight_dir=tmp_path / "unused-preflight",
            approval_path=tmp_path / "unused-approval",
            output_dir=output,
            model_factory=lambda: second_model.append("model"),
        )
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(first_failures) == 1
    assert isinstance(first_failures[0], RuntimeError)
    assert second_model == []


def test_execute_rejects_symlinked_output_parent_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval, preflight = _execution_capture(tmp_path)
    monkeypatch.setattr(
        propose_module, "_capture_inference_approval", lambda _path: approval
    )
    monkeypatch.setattr(
        propose_module,
        "_capture_and_verify_preflight",
        lambda _path, **_kwargs: preflight,
    )
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    touched: list[str] = []
    with pytest.raises(ValueError, match="output.*unsafe"):
        execute_proposals(
            preflight_dir=tmp_path / "unused-preflight",
            approval_path=tmp_path / "unused-approval",
            output_dir=linked_parent / "proposals",
            model_factory=lambda: touched.append("model"),
        )
    assert touched == []
    assert not (real_parent / "proposals").exists()


def test_only_ceiling_truncation_gets_one_retry() -> None:
    calls: list[int] = []
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": [],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    valid = b'{"status":"UNSUPPORTED","reason_code":"NO_ABSTRACT_CHILD","o1":null}'
    result = run_job_with_retry(
        job,
        generate=lambda ceiling: calls.append(ceiling)
        or ((b'{"status":', 256) if ceiling == 256 else (valid, 9)),
    )
    assert calls == [256, 512]
    assert result["status"] == "UNSUPPORTED"


def test_schema_error_is_not_retried() -> None:
    calls: list[int] = []
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": [],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    with pytest.raises(ValueError, match="schema"):
        run_job_with_retry(
            job,
            generate=lambda ceiling: calls.append(ceiling)
            or (b'{"status":"BAD"}', 4),
        )
    assert calls == [256]


def test_under_ceiling_incomplete_json_is_not_retried() -> None:
    calls: list[int] = []
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": [],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    with pytest.raises(ValueError, match="parse"):
        run_job_with_retry(
            job,
            generate=lambda ceiling: calls.append(ceiling) or (b'{"status":', 255),
        )
    assert calls == [256]


def test_semantic_error_is_not_retried() -> None:
    calls: list[int] = []
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": ["b" * 64],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    raw = json.dumps(
        {
            "status": "SUPPORTED",
            "reason_code": "SUPPORTED",
            "o1": {
                "label": "Abstract need",
                "scope_rationale": "Supported only by an out-of-job unit.",
                "support_unit_ids": ["c" * 64],
            },
        },
        separators=(",", ":"),
    ).encode()
    with pytest.raises(ValueError, match="semantic"):
        run_job_with_retry(
            job,
            generate=lambda ceiling: calls.append(ceiling) or (raw, 100),
        )
    assert calls == [256]


def test_retry_happens_at_most_once() -> None:
    calls: list[int] = []
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": [],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    with pytest.raises(ValueError, match="retry ceiling"):
        run_job_with_retry(
            job,
            generate=lambda ceiling: calls.append(ceiling) or (b'{"status":', ceiling),
        )
    assert calls == [256, 512]


@pytest.mark.parametrize(
    "generated",
    [
        b'{"status":"UNSUPPORTED","reason_code":"NO_ABSTRACT_CHILD","o1":null}',
        (b"{}", True),
        (b"{}", 1.0),
    ],
)
def test_injected_generation_requires_raw_bytes_and_exact_nonbool_token_count(
    generated: object,
) -> None:
    job = {
        "job_id": "a" * 64,
        "messages": [],
        "input_unit_ids": [],
        "primary_max_new_tokens": 256,
        "retry_max_new_tokens": 512,
    }
    with pytest.raises(TypeError, match="raw bytes.*exact token count"):
        run_job_with_retry(job, generate=lambda _ceiling: generated)


def test_local_tokenizer_loader_does_not_import_the_rocm_torch_runtime() -> None:
    before = set(sys.modules)
    tokenizer = _load_local_tokenizer()
    token_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "{}"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    imported = set(sys.modules) - before
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(
        name == "transformers" or name.startswith("transformers.") for name in imported
    )


def test_builder_rejects_missing_contract_path_before_tokenizer(tmp_path: Path) -> None:
    tokenizer = _fake_tokenizer()
    with pytest.raises(ValueError, match="contract.*path"):
        build_proposal_preflight(
            _contract_fixture(),
            tokenizer=tokenizer,
            model_factory=None,
            output_dir=tmp_path / "preflight",
            contract_dir=None,
            contract_receipt_sha256=None,
            model_snapshot=_snapshot_fixture(),
        )
    assert tokenizer.calls == 0
    assert not (tmp_path / "preflight").exists()


def test_builder_rejects_nonexistent_contract_path_before_tokenizer(
    tmp_path: Path,
) -> None:
    tokenizer = _fake_tokenizer()
    with pytest.raises(ValueError, match="contract.*path"):
        build_proposal_preflight(
            _contract_fixture(),
            tokenizer=tokenizer,
            model_factory=None,
            output_dir=tmp_path / "preflight",
            contract_dir=tmp_path / "missing",
            contract_receipt_sha256="a" * 64,
            model_snapshot=_snapshot_fixture(),
        )
    assert tokenizer.calls == 0


def test_builder_rejects_nonexact_relative_contract_path_before_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract_fixture()
    source, receipt_sha256 = _fixture_source(tmp_path, contract, name="contract")
    tokenizer = _fake_tokenizer()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute and exact"):
        build_proposal_preflight(
            contract,
            tokenizer=tokenizer,
            model_factory=None,
            output_dir=tmp_path / "preflight",
            contract_dir=Path(source.name),
            contract_receipt_sha256=receipt_sha256,
            model_snapshot=_snapshot_fixture(),
        )
    assert tokenizer.calls == 0


def test_builder_rejects_supplied_rows_that_differ_from_verified_contract(
    tmp_path: Path,
) -> None:
    verified = _contract_fixture()
    supplied = json.loads(json.dumps(verified))
    supplied["units"][0]["text"] = "resealed forged evidence"
    source, receipt_sha256 = _fixture_source(tmp_path, verified, name="contract")
    tokenizer = _fake_tokenizer()
    with patch.object(
        propose_module, "_load_verified_contract", return_value=verified
    ), pytest.raises(ValueError, match="verified contract rows"):
        build_proposal_preflight(
            supplied,
            tokenizer=tokenizer,
            model_factory=None,
            output_dir=tmp_path / "preflight",
            contract_dir=source,
            contract_receipt_sha256=receipt_sha256,
            model_snapshot=_snapshot_fixture(),
        )
    assert tokenizer.calls == 0


def test_verifier_rejects_resealed_null_contract_path_before_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _contract, _source, output = _build_fixture_preflight(
        tmp_path, tokenizer=_fake_tokenizer()
    )
    receipt["contract_receipt"]["path"] = None  # type: ignore[index]
    (output / "receipt.json").write_bytes(_pretty(receipt))
    touched: list[str] = []
    monkeypatch.setattr(
        propose_module,
        "_snapshot_inventory",
        lambda: touched.append("snapshot"),
    )
    monkeypatch.setattr(
        propose_module,
        "_load_local_tokenizer",
        lambda: touched.append("tokenizer"),
    )
    with pytest.raises(ValueError, match="contract.*path"):
        verify_proposal_preflight(output)
    assert touched == []


def test_verifier_rejects_resealed_contract_hash_before_source_or_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _contract, _source, output = _build_fixture_preflight(
        tmp_path, tokenizer=_fake_tokenizer()
    )
    receipt["contract_receipt"]["sha256"] = "0" * 64  # type: ignore[index]
    (output / "receipt.json").write_bytes(_pretty(receipt))
    touched: list[str] = []
    monkeypatch.setattr(
        propose_module,
        "_load_verified_contract",
        lambda _path: touched.append("source"),
    )
    monkeypatch.setattr(
        propose_module,
        "_snapshot_inventory",
        lambda: touched.append("snapshot"),
    )
    monkeypatch.setattr(
        propose_module,
        "_load_local_tokenizer",
        lambda: touched.append("tokenizer"),
    )
    with pytest.raises(ValueError, match="receipt hash"):
        verify_proposal_preflight(output)
    assert touched == []


def test_verifier_authenticates_source_before_snapshot_or_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _receipt, _contract, _source, output = _build_fixture_preflight(
        tmp_path, tokenizer=_fake_tokenizer()
    )
    touched: list[str] = []

    def fail_source(_path: Path) -> object:
        touched.append("source")
        raise ValueError("source authentication failed")

    monkeypatch.setattr(propose_module, "_load_verified_contract", fail_source)
    monkeypatch.setattr(
        propose_module,
        "_snapshot_inventory",
        lambda: touched.append("snapshot"),
    )
    monkeypatch.setattr(
        propose_module,
        "_load_local_tokenizer",
        lambda: touched.append("tokenizer"),
    )
    with pytest.raises(ValueError, match="source authentication failed"):
        verify_proposal_preflight(output)
    assert touched == ["source"]


def test_frozen_builder_hash_survives_future_module_edits_and_rejects_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _fake_tokenizer()
    receipt, contract, _source, output = _build_fixture_preflight(
        tmp_path, tokenizer=tokenizer
    )
    assert receipt["code_sha256"]["adaptive_obligation_v2_propose.py"] == (
        "f6c7a9db448d391a2ed7f1ad3eceba0bc8805e825ed4110ac6a7112985032988"
    )
    _patch_fixture_verifier(
        monkeypatch, contract=contract, receipt=receipt, tokenizer=tokenizer
    )
    assert verify_proposal_preflight(output) == receipt
    receipt["code_sha256"]["adaptive_obligation_v2_propose.py"] = "0" * 64
    (output / "receipt.json").write_bytes(_pretty(receipt))
    with pytest.raises(ValueError, match="builder.*hash"):
        verify_proposal_preflight(output)


def test_snapshot_inventory_hashes_bytes_even_for_64_hex_blob_name(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    target = blobs / ("f" * 64)
    target.write_bytes(b"corrupt bytes do not match the target name")
    required = (
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    for name in required:
        path = snapshot / name
        if name == "model-00001-of-00003.safetensors":
            path.symlink_to(target)
        else:
            path.write_bytes(name.encode())
    inventory = _snapshot_inventory(snapshot)
    row = next(
        item
        for item in inventory["files"]
        if item["name"] == "model-00001-of-00003.safetensors"
    )
    assert row["blob_id"] == "f" * 64
    assert row["content_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert row["content_sha256"] != row["blob_id"]


def test_verifier_docstring_discloses_tokenizer_loading() -> None:
    assert "tokenizer" in (verify_proposal_preflight.__doc__ or "").casefold()
    assert "without loading a tokenizer" not in (
        verify_proposal_preflight.__doc__ or ""
    ).casefold()

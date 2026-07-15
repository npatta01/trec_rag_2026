from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import trec_rag.adaptive_obligation_v2_propose as propose_module
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
    PROPOSAL_SCHEMA,
    _snapshot_inventory,
    _load_local_tokenizer,
    build_proposal_jobs,
    build_proposal_preflight,
    execute_proposals,
    main,
    run_job_with_retry,
    verify_proposal_preflight,
)


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
        or (b'{"status":' if ceiling == 256 else valid),
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
            generate=lambda ceiling: calls.append(ceiling) or b'{"status":"BAD"}',
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

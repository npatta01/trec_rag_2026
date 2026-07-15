from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

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
    _load_local_tokenizer,
    build_proposal_jobs,
    build_proposal_preflight,
    main,
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
    receipt = build_proposal_preflight(
        _contract_fixture(),
        tokenizer=_fake_tokenizer(),
        model_factory=lambda: touched.append("model"),
        output_dir=tmp_path / "preflight",
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


def test_preflight_seals_snapshot_tokenizer_prompt_and_code(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    receipt = build_proposal_preflight(
        _contract_fixture(),
        tokenizer=_load_local_tokenizer(),
        model_factory=lambda: pytest.fail("model factory must remain unreachable"),
        output_dir=output,
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
        )
    assert tokenizer.calls == 0
    assert not output.exists()


def test_preflight_is_create_only_before_retokenizing(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    first = _fake_tokenizer()
    build_proposal_preflight(
        _contract_fixture(), tokenizer=first, model_factory=None, output_dir=output
    )
    second = _fake_tokenizer()
    with pytest.raises(FileExistsError, match="create-only"):
        build_proposal_preflight(
            _contract_fixture(), tokenizer=second, model_factory=None, output_dir=output
        )
    assert first.calls == 48
    assert second.calls == 0


def test_verifier_rejects_job_tampering(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    build_proposal_preflight(
        _contract_fixture(),
        tokenizer=_load_local_tokenizer(),
        model_factory=None,
        output_dir=output,
    )
    with (output / "jobs.jsonl").open("ab") as sink:
        sink.write(b"{}\n")
    with pytest.raises(ValueError, match="jobs.jsonl"):
        verify_proposal_preflight(output)


def test_verifier_recomputes_resealed_prompt_token_counts(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    build_proposal_preflight(
        _contract_fixture(),
        tokenizer=_load_local_tokenizer(),
        model_factory=None,
        output_dir=output,
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
    with pytest.raises(ValueError, match="recomputed"):
        verify_proposal_preflight(output)


def test_cli_has_no_execute_action() -> None:
    with pytest.raises(SystemExit):
        main(["execute"])


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

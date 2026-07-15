from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_propose_r2 as r2_module
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
    build_proposal_jobs,
)
from trec_rag.adaptive_obligation_v2_propose_r2 import (
    R2_OUTPUT_CONTRACT,
    R2_PROPOSAL_SCHEMA,
    _build_authenticated_r2_preflight,
    _ordered_compact,
    build_r2_proposal_jobs,
    build_r2_proposal_preflight,
    publish_r2_proposal_preflight,
    render_r2_proposal_messages,
    verify_r2_proposal_preflight,
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


def _parent() -> dict[str, object]:
    return _contract_fixture()["parents"][0]  # type: ignore[index,return-value]


def _reservoir() -> dict[str, object]:
    return _contract_fixture()["reservoirs"][0]  # type: ignore[index,return-value]


def _evidence_units() -> list[dict[str, object]]:
    contract = _contract_fixture()
    unit_by_id = {
        row["unit_id"]: row for row in contract["units"]  # type: ignore[index]
    }
    reservoir = contract["reservoirs"][0]  # type: ignore[index]
    return [
        unit_by_id[unit_id]
        for document in reservoir["documents"]
        for unit_id in document["unit_ids"]
    ]


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
        return list(range(max(1, len(json.dumps(messages).split()))))


def _fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


def _snapshot_fixture() -> dict[str, object]:
    names = (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    files = [
        {
            "name": name,
            "bytes": index + 1,
            "blob_id": f"blob-{index}",
            "content_sha256": f"{index:x}" * 64,
        }
        for index, name in enumerate(names)
    ]
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _tokenizer_contract_fixture() -> dict[str, object]:
    return {
        "loader": "tokenizers.Tokenizer.from_file",
        "tokenizer_json_sha256": "1" * 64,
        "tokenizer_config_sha256": "2" * 64,
        "chat_template": "{{ messages }}",
        "chat_template_sha256": "3" * 64,
        "torch_imported": False,
        "transformers_imported": False,
    }


def _absolute_destination_bindings(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_dir": tmp_path / "preflight-r2",
        "ledger_dir": tmp_path / "ledger-r2",
        "proposal_dir": tmp_path / "proposals-r2",
    }


def _create_destination_leaf(path: Path, leaf_kind: str) -> None:
    if leaf_kind == "directory":
        path.mkdir()
    elif leaf_kind == "file":
        path.write_bytes(b"occupied")
    elif leaf_kind == "symlink":
        path.symlink_to(path.with_name("missing-target"))
    else:  # pragma: no cover - parametrization controls this branch
        raise AssertionError(leaf_kind)


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _source_binding(tmp_path: Path, contract: dict[str, object]) -> tuple[Path, str]:
    root = tmp_path / "contract"
    root.mkdir()
    source = _pretty(contract["receipt"])
    (root / "receipt.json").write_bytes(source)
    return root, hashlib.sha256(source).hexdigest()


def test_r2_output_contract_is_last_user_payload_field() -> None:
    messages = render_r2_proposal_messages(_parent(), _reservoir(), _evidence_units())
    payload = json.loads(messages[1]["content"])
    assert list(payload)[-1] == "output_contract"
    assert payload["output_contract"] == {
        "return": "exactly one JSON object and no prose",
        "label": {"max_unicode_characters": 80, "max_words": 10},
        "scope_rationale": {
            "exact_sentences": 1,
            "max_unicode_characters": 160,
            "max_words": 25,
        },
        "support_unit_ids_for_supported": {
            "minimum_items": 1,
            "maximum_items": 2,
            "source": "supplied evidence_units only",
        },
        "self_check": "silently verify every limit before emitting JSON",
        "on_failure": "return a schema-valid UNSUPPORTED object",
    }
    assert messages[1]["content"].endswith(
        '"output_contract":' + _ordered_compact(R2_OUTPUT_CONTRACT) + "}"
    )


def test_r2_preserves_schema_and_all_48_coverage_boundaries() -> None:
    contract = _contract_fixture()
    r1 = build_proposal_jobs(contract)
    r2 = build_r2_proposal_jobs(contract)
    assert len(r2) == 48
    assert [(j["topic_id"], j["parent_id"], j["fold"]) for j in r2] == [
        (j["topic_id"], j["parent_id"], j["fold"]) for j in r1
    ]
    assert [j["input_unit_ids"] for j in r2] == [
        j["input_unit_ids"] for j in r1
    ]
    assert R2_PROPOSAL_SCHEMA == PROPOSAL_SCHEMA
    assert R2_PROPOSAL_SCHEMA is not PROPOSAL_SCHEMA


def test_r2_preflight_is_tokenizer_only_and_versioned(tmp_path: Path) -> None:
    tokenizer = _fake_tokenizer()
    receipt = _build_authenticated_r2_preflight(
        _contract_fixture(),
        tokenizer=tokenizer,
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        **_absolute_destination_bindings(tmp_path),
    )
    assert receipt["schema_version"] == (
        "adaptive-obligation-v2-proposal-preflight-r2"
    )
    assert receipt["prompt_revision"] == "tail-contract-r2"
    assert receipt["job_count"] == 48
    assert receipt["tokenizer_load_count"] == 1
    assert receipt["model_load_count"] == 0
    assert receipt["inference_count"] == 0
    assert tokenizer.calls == 48
    assert not (tmp_path / "preflight-r2").exists()


def test_public_preflight_has_no_tokenizer_or_model_injection() -> None:
    assert list(inspect.signature(build_r2_proposal_preflight).parameters) == [
        "contract_dir",
        "output_dir",
        "ledger_dir",
        "proposal_dir",
    ]


def test_r2_rejects_protected_topic_before_tokenizer_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    contract = _contract_fixture()
    contract["parents"][0]["topic_id"] = "144"  # type: ignore[index]
    monkeypatch.setattr(r2_module, "_load_verified_contract", lambda _path: contract)
    monkeypatch.setattr(r2_module, "_snapshot_inventory", _snapshot_fixture)
    monkeypatch.setattr(
        r2_module,
        "_load_pinned_tokenizer_after_auth",
        lambda **_kwargs: touched.append("tokenizer"),
    )
    with pytest.raises(ValueError, match="protected topic"):
        build_r2_proposal_preflight(
            contract_dir=tmp_path / "contract",
            **_absolute_destination_bindings(tmp_path),
        )
    assert touched == []


def test_r2_preflight_rejects_relative_destinations(tmp_path: Path) -> None:
    for changed in ("ledger_dir", "proposal_dir"):
        bindings = _absolute_destination_bindings(tmp_path)
        bindings[changed] = Path("relative-output")
        with pytest.raises(ValueError, match="absolute safe destination"):
            _build_authenticated_r2_preflight(
                _contract_fixture(),
                tokenizer=_fake_tokenizer(),
                model_snapshot=_snapshot_fixture(),
                tokenizer_contract=_tokenizer_contract_fixture(),
                **bindings,
            )


@pytest.mark.parametrize("changed", ["ledger_dir", "proposal_dir"])
@pytest.mark.parametrize("leaf_kind", ["directory", "file", "symlink"])
def test_r2_preflight_requires_absent_create_only_destination_leaves(
    tmp_path: Path, changed: str, leaf_kind: str
) -> None:
    bindings = _absolute_destination_bindings(tmp_path)
    _create_destination_leaf(bindings[changed], leaf_kind)
    tokenizer = _fake_tokenizer()
    with pytest.raises(FileExistsError, match="create-only destination exists"):
        _build_authenticated_r2_preflight(
            _contract_fixture(),
            tokenizer=tokenizer,
            model_snapshot=_snapshot_fixture(),
            tokenizer_contract=_tokenizer_contract_fixture(),
            **bindings,
        )
    assert tokenizer.calls == 0


def test_r2_preflight_rejects_symlinked_destination_parent_before_tokenizer(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    bindings = _absolute_destination_bindings(tmp_path)
    bindings["ledger_dir"] = linked_parent / "ledger-r2"
    tokenizer = _fake_tokenizer()
    with pytest.raises(ValueError, match="absolute safe destination"):
        _build_authenticated_r2_preflight(
            _contract_fixture(),
            tokenizer=tokenizer,
            model_snapshot=_snapshot_fixture(),
            tokenizer_contract=_tokenizer_contract_fixture(),
            **bindings,
        )
    assert tokenizer.calls == 0


def test_private_fake_tokenizer_material_cannot_be_published(tmp_path: Path) -> None:
    material = _build_authenticated_r2_preflight(
        _contract_fixture(),
        tokenizer=_fake_tokenizer(),
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        **_absolute_destination_bindings(tmp_path),
    )
    with pytest.raises(PermissionError, match="production-authenticated"):
        publish_r2_proposal_preflight(
            material,
            output_dir=tmp_path / "preflight-r2",
        )
    assert not (tmp_path / "preflight-r2").exists()


def test_publication_verifies_staging_before_no_replace_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract_fixture()
    source_dir, _source_sha256 = _source_binding(tmp_path, contract)
    tokenizer = _fake_tokenizer()
    calls: list[Path] = []
    bindings = _absolute_destination_bindings(tmp_path)

    monkeypatch.setattr(r2_module, "_load_verified_contract", lambda _path: contract)
    monkeypatch.setattr(r2_module, "_snapshot_inventory", _snapshot_fixture)
    monkeypatch.setattr(
        r2_module, "_tokenizer_contract", _tokenizer_contract_fixture
    )
    monkeypatch.setattr(
        r2_module,
        "_load_pinned_tokenizer_after_auth",
        lambda **_kwargs: tokenizer,
    )

    def reject_staging(staging: Path) -> dict[str, object]:
        calls.append(staging)
        assert staging.is_dir()
        assert staging.parent == bindings["output_dir"].parent
        assert not bindings["output_dir"].exists()
        raise ValueError("staged verification rejected")

    monkeypatch.setattr(r2_module, "verify_r2_proposal_preflight", reject_staging)
    with pytest.raises(ValueError, match="staged verification rejected"):
        build_r2_proposal_preflight(
            contract_dir=source_dir,
            **bindings,
        )
    assert len(calls) == 1
    assert not bindings["output_dir"].exists()
    assert not calls[0].exists()
    assert list(tmp_path.glob(".preflight-r2.staging-*")) == []


def test_verifier_reconstructs_r2_jobs_and_recounts_with_pinned_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract_fixture()
    contract_dir, receipt_sha256 = _source_binding(tmp_path, contract)
    tokenizer = _fake_tokenizer()
    material = _build_authenticated_r2_preflight(
        contract,
        tokenizer=tokenizer,
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        contract_dir=contract_dir,
        contract_receipt_sha256=receipt_sha256,
        _publication_authorized=True,
        **_absolute_destination_bindings(tmp_path),
    )
    root = tmp_path / "captured-preflight"
    root.mkdir()
    for name, content in material.contents.items():
        (root / name).write_bytes(content)

    recount = _fake_tokenizer()
    monkeypatch.setattr(r2_module, "_load_verified_contract", lambda _path: contract)
    monkeypatch.setattr(r2_module, "_snapshot_inventory", _snapshot_fixture)
    monkeypatch.setattr(
        r2_module, "_tokenizer_contract", _tokenizer_contract_fixture
    )
    monkeypatch.setattr(
        r2_module,
        "_load_pinned_tokenizer_after_auth",
        lambda **_kwargs: recount,
    )
    verified = verify_r2_proposal_preflight(root)
    assert verified == dict(material)
    assert recount.calls == 48


def test_verifier_rejects_r1_preflight_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract_fixture()
    contract_dir, receipt_sha256 = _source_binding(tmp_path, contract)
    material = _build_authenticated_r2_preflight(
        contract,
        tokenizer=_fake_tokenizer(),
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        contract_dir=contract_dir,
        contract_receipt_sha256=receipt_sha256,
        _publication_authorized=True,
        **_absolute_destination_bindings(tmp_path),
    )
    root = tmp_path / "wrong-version"
    root.mkdir()
    for name, content in material.contents.items():
        (root / name).write_bytes(content)
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["schema_version"] = "adaptive-obligation-v2-proposal-preflight-v1"
    (root / "receipt.json").write_bytes(_pretty(receipt))
    monkeypatch.setattr(
        r2_module,
        "_snapshot_inventory",
        lambda: pytest.fail("R1 version must fail before snapshot access"),
    )
    with pytest.raises(ValueError, match="R2 preflight schema"):
        verify_r2_proposal_preflight(root)

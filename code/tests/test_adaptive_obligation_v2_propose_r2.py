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
    _execute_authenticated_r2_jobs,
    _build_authenticated_r2_preflight,
    _ordered_compact,
    build_r2_proposal_jobs,
    build_r2_proposal_preflight,
    execute_r2_proposals,
    finalize_r2_proposal_inventory,
    load_authenticated_r2_proposal_inventory,
    publish_r2_proposal_preflight,
    render_r2_proposal_messages,
    run_r2_job_with_retry,
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
    for name in ("parents", "reservoirs", "units"):
        rows = contract[name]
        assert isinstance(rows, list)
        source_rows = b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
        (root / f"{name}.jsonl").write_bytes(source_rows)
    source = _pretty(contract["receipt"])
    (root / "receipt.json").write_bytes(source)
    return root, hashlib.sha256(source).hexdigest()


def _load_minimally_authenticated_contract_source(
    root: Path,
) -> dict[str, object]:
    source_root = Path(root)
    receipt_source = (source_root / "receipt.json").read_bytes()
    receipt = json.loads(receipt_source)
    if (
        receipt_source != _pretty(receipt)
        or receipt.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
    ):
        raise ValueError("fake contract receipt authentication failed")

    def load_rows(name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for line in (source_root / f"{name}.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            canonical = json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if line != canonical:
                raise ValueError("fake contract row authentication failed")
            rows.append(row)
        return rows

    return {
        "parents": load_rows("parents"),
        "reservoirs": load_rows("reservoirs"),
        "units": load_rows("units"),
        "receipt": receipt,
    }


def _captured_material_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], object, Path]:
    contract = _contract_fixture()
    contract_dir, receipt_sha256 = _source_binding(tmp_path, contract)
    authenticated_contract = _load_minimally_authenticated_contract_source(
        contract_dir
    )
    material = _build_authenticated_r2_preflight(
        authenticated_contract,
        tokenizer=_fake_tokenizer(),
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        contract_dir=contract_dir,
        contract_receipt_sha256=receipt_sha256,
        **_absolute_destination_bindings(tmp_path),
    )
    root = tmp_path / "captured-preflight"
    root.mkdir()
    for name, content in material.contents.items():
        (root / name).write_bytes(content)
    return authenticated_contract, material, root


def _patch_verifier_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: dict[str, object],
    tokenizer: object | None = None,
) -> _FakeTokenizer:
    recount = tokenizer if tokenizer is not None else _fake_tokenizer()
    monkeypatch.setattr(
        r2_module,
        "_load_verified_contract",
        _load_minimally_authenticated_contract_source,
    )
    monkeypatch.setattr(r2_module, "_snapshot_inventory", _snapshot_fixture)
    monkeypatch.setattr(
        r2_module, "_tokenizer_contract", _tokenizer_contract_fixture
    )
    monkeypatch.setattr(
        r2_module,
        "_load_pinned_tokenizer_after_auth",
        lambda **_kwargs: recount,
    )
    return recount  # type: ignore[return-value]


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


def test_private_fake_tokenizer_seam_has_no_publication_authorization_argument(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(_build_authenticated_r2_preflight).parameters
    assert "_publication_authorized" not in parameters
    assert not any("publication" in name for name in parameters)
    with pytest.raises(TypeError):
        _build_authenticated_r2_preflight(
            _contract_fixture(),
            tokenizer=_fake_tokenizer(),
            model_snapshot=_snapshot_fixture(),
            tokenizer_contract=_tokenizer_contract_fixture(),
            _publication_authorized=True,
            **_absolute_destination_bindings(tmp_path),
        )


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
    assert not hasattr(material, "publication_authorized")
    with pytest.raises(TypeError):
        publish_r2_proposal_preflight(
            preflight=material,
            output_dir=tmp_path / "preflight-r2",
        )
    assert not (tmp_path / "preflight-r2").exists()


def test_publication_has_no_module_level_material_authority() -> None:
    assert not hasattr(r2_module, "_PRODUCTION_PUBLICATION_CAPABILITY")
    assert not hasattr(r2_module, "_R2PublishableMaterial")
    assert not hasattr(r2_module, "_build_production_r2_preflight_material")
    assert list(inspect.signature(publish_r2_proposal_preflight).parameters) == [
        "contract_dir",
        "output_dir",
        "ledger_dir",
        "proposal_dir",
    ]


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
    contract, material, root = _captured_material_fixture(tmp_path)
    recount = _patch_verifier_dependencies(monkeypatch, contract=contract)
    verified = verify_r2_proposal_preflight(root)
    assert verified == dict(material)
    assert recount.calls == 48


@pytest.mark.parametrize("artifact_name", ["prompt.json", "schema.json"])
def test_verifier_rejects_json_bool_integer_substitution_in_frozen_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    contract, _material, root = _captured_material_fixture(tmp_path)
    value = json.loads((root / artifact_name).read_bytes())
    if artifact_name == "prompt.json":
        assert value["decoding"]["seed"] == 0
        value["decoding"]["seed"] = False
        receipt_hash_name = "prompt_sha256"
    else:
        assert value["additionalProperties"] is False
        value["additionalProperties"] = 0
        receipt_hash_name = "schema_sha256"
    content = _pretty(value)
    (root / artifact_name).write_bytes(content)
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt[receipt_hash_name] = canonical_sha256(value)
    receipt["artifacts"][artifact_name].update(
        {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    (root / "receipt.json").write_bytes(_pretty(receipt))
    _patch_verifier_dependencies(monkeypatch, contract=contract)
    with pytest.raises(ValueError, match="schema or prompt"):
        verify_r2_proposal_preflight(root)


def test_verifier_rejects_json_bool_integer_substitution_in_reconstructed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _material, root = _captured_material_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "jobs.jsonl").read_bytes().splitlines()
    ]
    assert rows[0]["parent_manifest_order"] == 0
    rows[0]["parent_manifest_order"] = False
    jobs_bytes = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    (root / "jobs.jsonl").write_bytes(jobs_bytes)
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["artifacts"]["jobs.jsonl"].update(
        {
            "bytes": len(jobs_bytes),
            "sha256": hashlib.sha256(jobs_bytes).hexdigest(),
        }
    )
    (root / "receipt.json").write_bytes(_pretty(receipt))
    _patch_verifier_dependencies(monkeypatch, contract=contract)
    with pytest.raises(ValueError, match="authenticated contract reconstruction"):
        verify_r2_proposal_preflight(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top_level",
        "tokenizer_loading_local_files_only",
        "expected_runtime_proposal_calls",
        "planned_storage_job_rows",
    ],
)
def test_verifier_rejects_any_frozen_receipt_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contract, _material, root = _captured_material_fixture(tmp_path)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if mutation == "unknown_top_level":
        receipt["unreviewed_extension"] = True
    elif mutation == "tokenizer_loading_local_files_only":
        receipt["tokenizer_loading"]["local_files_only"] = False
    elif mutation == "expected_runtime_proposal_calls":
        receipt["expected_runtime"]["proposal_calls_executed"] = 1
    elif mutation == "planned_storage_job_rows":
        receipt["planned_storage"]["job_rows"] = 47
    else:  # pragma: no cover - parametrization controls this branch
        raise AssertionError(mutation)
    (root / "receipt.json").write_bytes(_pretty(receipt))
    _patch_verifier_dependencies(monkeypatch, contract=contract)
    with pytest.raises(ValueError, match="R2 preflight metadata"):
        verify_r2_proposal_preflight(root)


def test_code_contract_freezes_inherited_r1_builder_bytes(tmp_path: Path) -> None:
    receipt = _build_authenticated_r2_preflight(
        _contract_fixture(),
        tokenizer=_fake_tokenizer(),
        model_snapshot=_snapshot_fixture(),
        tokenizer_contract=_tokenizer_contract_fixture(),
        **_absolute_destination_bindings(tmp_path),
    )
    r1_path = Path(r2_module.r1.__file__)
    assert set(receipt["code_sha256"]) == {
        "adaptive_obligation_v2_contract.py",
        "adaptive_obligation_v2_propose.py",
        "adaptive_obligation_v2_propose_r2.py",
    }
    assert receipt["code_sha256"]["adaptive_obligation_v2_propose.py"] == (
        hashlib.sha256(r1_path.read_bytes()).hexdigest()
    )


def test_verifier_rejects_inherited_r1_builder_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _material, root = _captured_material_fixture(tmp_path)
    _patch_verifier_dependencies(monkeypatch, contract=contract)
    original = r2_module.r1._sha256_file

    def changed_r1_hash(path: Path) -> str:
        if Path(path).name == "adaptive_obligation_v2_propose.py":
            return "f" * 64
        return original(path)

    monkeypatch.setattr(r2_module.r1, "_sha256_file", changed_r1_hash)
    with pytest.raises(ValueError, match="builder code hashes"):
        verify_r2_proposal_preflight(root)


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


def _runtime_preflight(material: object) -> dict[str, object]:
    contents = material.contents
    jobs = [
        json.loads(line) for line in contents["jobs.jsonl"].splitlines()
    ]
    receipt_source = contents["receipt.json"]
    return {
        **dict(material),
        "receipt_sha256": hashlib.sha256(receipt_source).hexdigest(),
        "jobs": jobs,
    }


def _runtime_approval(
    preflight: dict[str, object], ledger_dir: Path
) -> dict[str, object]:
    return {
        "schema_version": "adaptive-obligation-v2-proposal-approval-r2",
        "stage": "proposal_r2",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "ledger_dir": str(ledger_dir),
        "approved": True,
    }


def _runtime_fixture(tmp_path: Path) -> tuple[dict[str, object], object, Path]:
    _contract, material, root = _captured_material_fixture(tmp_path)
    return _runtime_preflight(material), material, root


def _runtime_ledger(
    preflight: dict[str, object],
    ledger_dir: Path,
    *,
    approval_sha256: str = "e" * 64,
) -> object:
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    anchor = r2_module.r1._build_run_anchor(
        jobs,
        preflight_sha256=str(preflight["receipt_sha256"]),
        approval_sha256=approval_sha256,
    )
    return r2_module.r1.AppendOnlyAttemptLedger(
        ledger_dir,
        expected_anchor=anchor,
        create_only=True,
    )


def _reopen_runtime_ledger(
    preflight: dict[str, object],
    ledger_dir: Path,
    *,
    approval_sha256: str = "e" * 64,
) -> object:
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    anchor = r2_module.r1._build_run_anchor(
        jobs,
        preflight_sha256=str(preflight["receipt_sha256"]),
        approval_sha256=approval_sha256,
    )
    return r2_module.r1.AppendOnlyAttemptLedger(
        ledger_dir,
        expected_anchor=anchor,
        create_only=False,
    )


def _valid_completion_value() -> dict[str, object]:
    return {
        "status": "UNSUPPORTED",
        "reason_code": "NO_ABSTRACT_CHILD",
        "o1": None,
    }


def _valid_completion() -> bytes:
    return json.dumps(
        _valid_completion_value(), separators=(",", ":"), sort_keys=True
    ).encode()


class _SequenceModel:
    def __init__(self, completions: list[tuple[bytes, int]]) -> None:
        self.completions = list(completions)
        self.calls = 0

    def generate(
        self,
        *args: object,
        max_new_tokens: int | None = None,
    ) -> tuple[bytes, int]:
        if max_new_tokens is None:
            assert len(args) == 1 and isinstance(args[0], int)
        else:
            assert len(args) == 2
        self.calls += 1
        return self.completions.pop(0)


class _ValidFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        _messages: object,
        _schema: object,
        *,
        max_new_tokens: int,
    ) -> tuple[bytes, int]:
        assert max_new_tokens == 256
        self.calls += 1
        return _valid_completion(), 17


class _OffsetTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        ) + [999]


def test_r1_approval_cannot_authorize_r2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval_path = tmp_path / "approval-r1.json"
    approval_path.write_bytes(
        _pretty(
            {
                "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
                "stage": "proposal",
                "preflight_sha256": "a" * 64,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "primary_call_count": 48,
                "retry_call_ceiling": 48,
                "approved": True,
            }
        )
    )
    touched: list[str] = []
    monkeypatch.setattr(
        r2_module,
        "_load_r2_model_after_approval",
        lambda **_kwargs: touched.append("model"),
        raising=False,
    )
    ledger_dir = tmp_path / "r2-ledger"
    with pytest.raises(PermissionError, match="R2 proposal approval required"):
        execute_r2_proposals(
            preflight_dir=tmp_path / "r2-preflight",
            approval_path=approval_path,
            ledger_dir=ledger_dir,
        )
    assert touched == []
    assert not ledger_dir.exists()


def test_first_schema_error_leaves_47_jobs_uncalled(tmp_path: Path) -> None:
    preflight, _material, _root = _runtime_fixture(tmp_path)
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    support_unit_id = jobs[0]["input_unit_ids"][0]
    invalid = {
        "status": "SUPPORTED",
        "reason_code": "SUPPORTED",
        "o1": {
            "label": "bounded abstract need",
            "scope_rationale": "x" * 241,
            "support_unit_ids": [support_unit_id],
        },
    }
    model = _SequenceModel(
        [
            (
                json.dumps(invalid, separators=(",", ":"), sort_keys=True).encode(),
                80,
            )
        ]
    )
    ledger = _runtime_ledger(
        preflight, Path(str(preflight["ledger_dir"]))
    )
    with pytest.raises(ValueError, match="scope_rationale"):
        _execute_authenticated_r2_jobs(
            preflight, ledger=ledger, model=model
        )
    assert model.calls == 1


def test_only_exact_ceiling_incomplete_json_retries(tmp_path: Path) -> None:
    preflight, _material, _root = _runtime_fixture(tmp_path)
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    model = _SequenceModel(
        [(b'{"status":', 256), (_valid_completion(), 143)]
    )
    result = run_r2_job_with_retry(
        jobs[0],
        generate=model.generate,
        ledger=_runtime_ledger(
            preflight, Path(str(preflight["ledger_dir"]))
        ),
    )
    assert result == _valid_completion_value()
    assert model.calls == 2


def test_incomplete_json_below_ceiling_is_terminal(tmp_path: Path) -> None:
    preflight, _material, _root = _runtime_fixture(tmp_path)
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    model = _SequenceModel([(b'{"status":', 255)])
    with pytest.raises(ValueError, match="JSON"):
        run_r2_job_with_retry(
            jobs[0],
            generate=model.generate,
            ledger=_runtime_ledger(
                preflight, Path(str(preflight["ledger_dir"]))
            ),
        )
    assert model.calls == 1


def test_public_executor_exposes_no_runtime_injection() -> None:
    assert list(inspect.signature(execute_r2_proposals).parameters) == [
        "preflight_dir",
        "approval_path",
        "ledger_dir",
    ]


def test_complete_fake_r2_run_seals_48_results(tmp_path: Path) -> None:
    preflight, _material, _root = _runtime_fixture(tmp_path)
    ledger_dir = Path(str(preflight["ledger_dir"]))
    model = _ValidFakeModel()
    result = _execute_authenticated_r2_jobs(
        preflight,
        ledger=_runtime_ledger(preflight, ledger_dir),
        model=model,
    )
    assert result["job_count"] == 48
    assert len(result["results"]) == 48
    assert model.calls == 48
    assert _reopen_runtime_ledger(
        preflight, ledger_dir
    ).read_sealed_results()["results"] == result["results"]


def _complete_r2_run_fixture(tmp_path: Path) -> dict[str, object]:
    preflight, _material, preflight_dir = _runtime_fixture(tmp_path)
    ledger_dir = Path(str(preflight["ledger_dir"]))
    proposal_dir = Path(str(preflight["proposal_dir"]))
    approval = _runtime_approval(preflight, ledger_dir)
    approval_source = _pretty(approval)
    approval_path = tmp_path / "approval-r2.json"
    approval_path.write_bytes(approval_source)
    approval_sha256 = hashlib.sha256(approval_source).hexdigest()
    _execute_authenticated_r2_jobs(
        preflight,
        ledger=_runtime_ledger(
            preflight,
            ledger_dir,
            approval_sha256=approval_sha256,
        ),
        model=_ValidFakeModel(),
    )
    return {
        "preflight": preflight,
        "preflight_dir": preflight_dir,
        "approval_path": approval_path,
        "ledger_dir": ledger_dir,
        "proposal_dir": proposal_dir,
    }


def _install_post_execution_drift(
    frozen: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    preflight = frozen["preflight"]
    assert isinstance(preflight, dict)
    source_binding = preflight["contract_receipt"]
    assert isinstance(source_binding, dict)
    contract_root = Path(str(source_binding["path"]))
    if drift == "missing_contract_receipt":
        (contract_root / "receipt.json").unlink()
    elif drift == "changed_contract_receipt":
        receipt_path = contract_root / "receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt["unit_count"] += 1
        receipt_path.write_bytes(_pretty(receipt))
    elif drift == "changed_contract_content":
        units_path = contract_root / "units.jsonl"
        rows = [json.loads(line) for line in units_path.read_bytes().splitlines()]
        rows[0]["text"] += " Changed after execution."
        rows[0]["text_sha256"] = sha256_text(rows[0]["text"])
        units_path.write_bytes(
            b"".join(
                (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                for row in rows
            )
        )
    elif drift == "model_snapshot":
        changed = _snapshot_fixture()
        changed["manifest_sha256"] = "f" * 64
        monkeypatch.setattr(r2_module, "_snapshot_inventory", lambda: changed)
    elif drift == "tokenizer_files":
        changed_files = list(
            r2_module._tokenizer_file_inventory(_snapshot_fixture())
        )
        changed_files[0] = {**changed_files[0], "content_sha256": "f" * 64}
        monkeypatch.setattr(
            r2_module,
            "_tokenizer_file_inventory",
            lambda _snapshot: changed_files,
        )
    elif drift == "tokenizer_contract":
        changed = _tokenizer_contract_fixture()
        changed["chat_template_sha256"] = "f" * 64
        monkeypatch.setattr(r2_module, "_tokenizer_contract", lambda: changed)
    elif drift == "code":
        changed = dict(preflight["code_sha256"])
        changed["adaptive_obligation_v2_propose.py"] = "f" * 64
        monkeypatch.setattr(r2_module, "_code_contract", lambda: changed)
    elif drift == "prompt_token_count":
        tokenizer = _OffsetTokenizer()
        monkeypatch.setattr(
            r2_module,
            "_load_pinned_tokenizer_after_auth",
            lambda **_kwargs: tokenizer,
        )
    else:  # pragma: no cover - parametrization controls this branch
        raise AssertionError(drift)


_POST_EXECUTION_SOURCE_DRIFTS = (
    "missing_contract_receipt",
    "changed_contract_receipt",
    "changed_contract_content",
    "model_snapshot",
    "tokenizer_files",
    "tokenizer_contract",
    "code",
    "prompt_token_count",
)


@pytest.mark.parametrize("drift", _POST_EXECUTION_SOURCE_DRIFTS)
def test_r2_finalizer_replays_and_rejects_post_execution_source_chain_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    frozen = _complete_r2_run_fixture(tmp_path)
    preflight = frozen["preflight"]
    assert isinstance(preflight, dict)
    _patch_verifier_dependencies(
        monkeypatch,
        contract=_load_minimally_authenticated_contract_source(
            Path(str(preflight["contract_receipt"]["path"]))  # type: ignore[index]
        ),
    )
    _install_post_execution_drift(frozen, monkeypatch, drift)

    with pytest.raises((OSError, ValueError)):
        finalize_r2_proposal_inventory(
            preflight_dir=frozen["preflight_dir"],
            approval_path=frozen["approval_path"],
            ledger_dir=frozen["ledger_dir"],
            output_dir=frozen["proposal_dir"],
        )
    assert not Path(frozen["proposal_dir"]).exists()


@pytest.mark.parametrize("drift", _POST_EXECUTION_SOURCE_DRIFTS)
def test_r2_loader_replays_and_rejects_post_execution_source_chain_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    frozen = _complete_r2_run_fixture(tmp_path)
    preflight = frozen["preflight"]
    assert isinstance(preflight, dict)
    contract = _load_minimally_authenticated_contract_source(
        Path(str(preflight["contract_receipt"]["path"]))  # type: ignore[index]
    )
    _patch_verifier_dependencies(monkeypatch, contract=contract)
    finalize_r2_proposal_inventory(
        preflight_dir=frozen["preflight_dir"],
        approval_path=frozen["approval_path"],
        ledger_dir=frozen["ledger_dir"],
        output_dir=frozen["proposal_dir"],
    )
    _install_post_execution_drift(frozen, monkeypatch, drift)

    with pytest.raises((OSError, ValueError)):
        load_authenticated_r2_proposal_inventory(
            output_dir=frozen["proposal_dir"],
            preflight_dir=frozen["preflight_dir"],
            ledger_dir=frozen["ledger_dir"],
        )


def test_finalizer_rejects_nonfrozen_proposal_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _complete_r2_run_fixture(tmp_path)
    preflight = frozen["preflight"]
    assert isinstance(preflight, dict)
    _patch_verifier_dependencies(
        monkeypatch,
        contract=_load_minimally_authenticated_contract_source(
            Path(str(preflight["contract_receipt"]["path"]))  # type: ignore[index]
        ),
    )
    with pytest.raises(ValueError, match="frozen proposal destination"):
        finalize_r2_proposal_inventory(
            preflight_dir=frozen["preflight_dir"],
            approval_path=frozen["approval_path"],
            ledger_dir=frozen["ledger_dir"],
            output_dir=tmp_path / "different-proposals-r2",
        )
    assert not (tmp_path / "different-proposals-r2").exists()


def test_r2_finalizer_and_authenticated_loader_replay_sealed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _complete_r2_run_fixture(tmp_path)
    preflight = frozen["preflight"]
    assert isinstance(preflight, dict)
    _patch_verifier_dependencies(
        monkeypatch,
        contract=_load_minimally_authenticated_contract_source(
            Path(str(preflight["contract_receipt"]["path"]))  # type: ignore[index]
        ),
    )
    receipt = finalize_r2_proposal_inventory(
        preflight_dir=frozen["preflight_dir"],
        approval_path=frozen["approval_path"],
        ledger_dir=frozen["ledger_dir"],
        output_dir=frozen["proposal_dir"],
    )
    assert receipt["schema_version"] == (
        "adaptive-obligation-v2-proposal-receipt-r2"
    )
    assert receipt["proposal_count"] == 48
    loaded = load_authenticated_r2_proposal_inventory(
        output_dir=frozen["proposal_dir"],
        preflight_dir=frozen["preflight_dir"],
        ledger_dir=frozen["ledger_dir"],
    )
    assert len(loaded["proposals"]) == 48
    assert loaded["receipt"] == receipt


def test_r2_finalizer_requires_approval_before_any_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "proposals-r2"
    with pytest.raises(PermissionError, match="R2 proposal approval required"):
        finalize_r2_proposal_inventory(
            preflight_dir=tmp_path / "missing-preflight",
            approval_path=tmp_path / "missing-approval",
            ledger_dir=tmp_path / "missing-ledger",
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_r2_cli_has_no_approval_creation_or_implicit_execute_action() -> None:
    actions = set(r2_module._parser()._subparsers._group_actions[0].choices)
    assert actions == {
        "build-preflight",
        "verify-preflight",
        "execute",
        "finalize",
    }
    assert "approve" not in actions


def test_r2_build_cli_resolves_relative_preflight_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Path] = {}

    def fake_build(**arguments: Path) -> dict[str, object]:
        observed.update(arguments)
        return {
            "status": "complete",
            "job_count": 48,
            "primary_call_count": 48,
            "retry_call_ceiling": 48,
            "worst_case_call_ceiling": 96,
            "prompt_token_counts": {"minimum": 1, "maximum": 2, "total": 3},
            "tokenizer_load_count": 1,
            "model_load_count": 0,
            "inference_count": 0,
            "network_call_count": 0,
            "retrieval_call_count": 0,
            "qrels_opened": False,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(r2_module, "build_r2_proposal_preflight", fake_build)
    ledger = tmp_path / "ledger-r2"
    proposals = tmp_path / "proposals-r2"

    assert r2_module.main(
        [
            "build-preflight",
            "--contract",
            "contract",
            "--output",
            "preflight-r2",
            "--ledger-destination",
            str(ledger),
            "--proposal-destination",
            str(proposals),
        ]
    ) == 0

    assert observed == {
        "contract_dir": Path("contract"),
        "output_dir": tmp_path / "preflight-r2",
        "ledger_dir": ledger,
        "proposal_dir": proposals,
    }

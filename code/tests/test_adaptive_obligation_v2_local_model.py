from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.adaptive_obligation_v2_local_model as model_module
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
from trec_rag.adaptive_obligation_v2_local_model import (
    V2LocalJsonModel,
    V2ValidationLocalJsonModel,
    verify_inference_approval,
    verify_validation_inference_approval,
)
from trec_rag.adaptive_obligation_v2_propose import MODEL_ID, MODEL_REVISION


def _model_manifest(files: dict[str, bytes]) -> dict[str, object]:
    rows = [
        {
            "name": name,
            "bytes": len(content),
            "blob_id": name,
            "content_sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(files.items())
    ]
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": rows}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _write_snapshot(root: Path, files: dict[str, bytes]) -> dict[str, object]:
    root.mkdir()
    for name, content in files.items():
        (root / name).write_bytes(content)
    return _model_manifest(files)


def _preflight() -> dict[str, object]:
    manifest = _model_manifest({"config.json": b"config"})
    return {
        "receipt_sha256": "a" * 64,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "model_snapshot": manifest,
    }


def _approval() -> dict[str, object]:
    return {
        "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
        "stage": "proposal",
        "preflight_sha256": "a" * 64,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "approved": True,
    }


def _validation_schema() -> dict[str, object]:
    return {"type": "object", "additionalProperties": False}


def _validation_preflight() -> dict[str, object]:
    manifest = _model_manifest({"config.json": b"config"})
    messages = [{"role": "user", "content": "validate this obligation"}]
    return {
        "receipt_sha256": "b" * 64,
        "stage": "validation",
        "validator_role": "opposite_fold_semantic_validator",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": manifest,
        "model_snapshot_manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_identity_sha256": "c" * 64,
        "schema_sha256": canonical_sha256(_validation_schema()),
        "job_count": 1,
        "primary_call_count": 1,
        "retry_call_ceiling": 1,
        "worst_case_call_ceiling": 2,
        "jobs": [
            {
                "job_id": "validation:job-1",
                "messages": messages,
                "prompt_token_count": 17,
            }
        ],
    }


def _validation_approval() -> dict[str, object]:
    preflight = _validation_preflight()
    return {
        "schema_version": "adaptive-obligation-v2-validation-approval-v1",
        "stage": "validation",
        "validator_role": "opposite_fold_semantic_validator",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": preflight[
            "model_snapshot_manifest_sha256"
        ],
        "tokenizer_identity_sha256": preflight["tokenizer_identity_sha256"],
        "job_count": 1,
        "primary_call_count": 1,
        "retry_call_ceiling": 1,
        "worst_case_call_ceiling": 2,
        "ledger_dir": "/tmp/adaptive-obligation-v2-validation-ledger",
        "approved": True,
    }


def test_v2_local_json_model_interface_exists() -> None:
    assert V2LocalJsonModel.__name__ == "V2LocalJsonModel"


def test_approval_requires_exact_stage_and_preflight_binding() -> None:
    assert verify_inference_approval(_approval(), _preflight()) == _approval()
    audited = {
        **_approval(),
        "created_by": "independent-approver",
        "audit": {"ticket": "RAG-2026"},
    }
    assert verify_inference_approval(audited, _preflight()) == audited
    wrong = {**_approval(), "stage": "validation"}
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        verify_inference_approval(wrong, _preflight())
    null_hash_approval = {**_approval(), "preflight_sha256": None}
    null_hash_preflight = {**_preflight(), "receipt_sha256": None}
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        verify_inference_approval(null_hash_approval, null_hash_preflight)


@pytest.mark.parametrize(
    ("approval_change", "preflight_change"),
    [
        ({"primary_call_count": 48.0}, {}),
        ({"retry_call_ceiling": True}, {}),
        ({"approved": 1}, {}),
        ({"approved": False}, {}),
        ({"model_revision": "0" * 40}, {}),
        ({}, {"primary_call_count": 48.0}),
        ({}, {"retry_call_ceiling": True}),
    ],
)
def test_approval_and_preflight_counts_require_exact_json_types(
    approval_change: dict[str, object], preflight_change: dict[str, object]
) -> None:
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        verify_inference_approval(
            {**_approval(), **approval_change},
            {**_preflight(), **preflight_change},
        )


def test_adapter_gates_before_injected_runtime_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[dict[str, object]] = []
    fake_runtime = SimpleNamespace(generate_json_bytes=lambda *args, **kwargs: (b"{}", 1))

    def load(**kwargs: object) -> object:
        touched.append(dict(kwargs))
        return fake_runtime

    monkeypatch.setattr(model_module, "load_pinned_local_runtime", load)
    wrong = {**_approval(), "model_revision": "0" * 40}
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        V2LocalJsonModel(approval=wrong, preflight=_preflight())
    assert touched == []

    adapter = V2LocalJsonModel(approval=_approval(), preflight=_preflight())
    assert touched == [
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "model_manifest": _preflight()["model_snapshot"],
        }
    ]
    assert adapter.generate([], {}, max_new_tokens=256) == (b"{}", 1)


@pytest.mark.parametrize(
    ("approval_change", "preflight_change"),
    [
        ({"model_revision": "0" * 40}, {}),
        ({"model_snapshot_manifest_sha256": "0" * 64}, {}),
        ({"tokenizer_identity_sha256": "0" * 64}, {}),
        ({"worst_case_call_ceiling": 1}, {}),
        ({}, {"model_snapshot_manifest_sha256": "0" * 64}),
    ],
)
def test_validation_approval_binds_exact_model_tokenizer_and_call_ceiling(
    approval_change: dict[str, object], preflight_change: dict[str, object]
) -> None:
    with pytest.raises(PermissionError, match="validation inference approval required"):
        verify_validation_inference_approval(
            {**_validation_approval(), **approval_change},
            {**_validation_preflight(), **preflight_change},
        )


def test_validation_adapter_rechecks_approved_prompt_count_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_json_bytes(
            self,
            messages: object,
            schema: object,
            **kwargs: object,
        ) -> tuple[bytes, int]:
            calls.append(
                {
                    "messages": messages,
                    "schema": schema,
                    **kwargs,
                }
            )
            return b'{"decision":"NO_EVIDENCE","support_unit_ids":[]}', 8

    loader_calls: list[dict[str, object]] = []

    def load(**kwargs: object) -> object:
        loader_calls.append(dict(kwargs))
        return Runtime()

    monkeypatch.setattr(model_module, "load_pinned_local_runtime", load)
    preflight = _validation_preflight()
    adapter = V2ValidationLocalJsonModel(
        approval=_validation_approval(), preflight=preflight
    )
    messages = preflight["jobs"][0]["messages"]  # type: ignore[index]
    assert adapter.generate(
        messages, _validation_schema(), max_new_tokens=256
    ) == (b'{"decision":"NO_EVIDENCE","support_unit_ids":[]}', 8)
    assert loader_calls == [
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "model_manifest": preflight["model_snapshot"],
        }
    ]
    assert calls == [
        {
            "messages": messages,
            "schema": _validation_schema(),
            "max_new_tokens": 256,
            "do_sample": False,
            "expected_prompt_tokens": 17,
        }
    ]
    with pytest.raises(ValueError, match="prompt identity"):
        adapter.generate(
            [{"role": "user", "content": "different"}],
            _validation_schema(),
            max_new_tokens=256,
        )
    with pytest.raises(ValueError, match="schema"):
        adapter.generate(messages, {"type": "array"}, max_new_tokens=256)


def test_validation_preflight_loader_constructs_only_the_pinned_tokenizer(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    files = {
        "config.json": b"exact config",
        "tokenizer.json": b"exact tokenizer",
    }
    manifest = _write_snapshot(snapshot, files)
    calls: list[tuple[Path, dict[str, object]]] = []

    class Tokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> list[int]:
            return [1, 2, 3]

    class TokenizerClass:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> object:
            private = Path(path)
            assert private != snapshot
            assert {child.name for child in private.iterdir()} == set(files)
            calls.append((private, dict(kwargs)))
            return Tokenizer()

    tokenizer = model_module.load_pinned_local_tokenizer(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        snapshot_dir=snapshot,
        model_manifest=manifest,
        runtime_modules=SimpleNamespace(auto_tokenizer_cls=TokenizerClass),
    )
    assert tokenizer.apply_chat_template([]) == [1, 2, 3]  # type: ignore[attr-defined]
    assert calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }


def test_pinned_runtime_rejects_prompt_count_drift_before_inference() -> None:
    class Tensor:
        shape = (1, 3)

        def to(self, _device: str) -> "Tensor":
            return self

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *_args, **_kwargs: {"input_ids": Tensor()}
    )
    model = SimpleNamespace(
        generate=lambda **_kwargs: pytest.fail("inference must remain unreachable")
    )
    runtime = model_module._PinnedLocalRuntime(
        modules=SimpleNamespace(torch=pytest.fail),
        tokenizer=tokenizer,
        model=model,
        private_snapshot=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="prompt token count"):
        runtime.generate_json_bytes(
            [{"role": "user", "content": "prompt"}],
            {"type": "object"},
            max_new_tokens=256,
            do_sample=False,
            expected_prompt_tokens=4,
        )


def test_injected_loader_uses_exact_offline_safetensor_bfloat16_rocm_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate_temp = tmp_path / "alternate-temp"
    alternate_temp.mkdir()
    monkeypatch.setattr(model_module.tempfile, "tempdir", str(alternate_temp))
    snapshot = tmp_path / MODEL_REVISION
    files = {
        "config.json": b"exact config bytes",
        "model.safetensors": b"small injected weights",
        "tokenizer.json": b"exact tokenizer bytes",
    }
    manifest = _write_snapshot(snapshot, files)
    tokenizer_calls: list[tuple[object, dict[str, object]]] = []
    model_calls: list[tuple[object, dict[str, object]]] = []

    class TokenizerClass:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> object:
            private = Path(path)
            assert private != snapshot
            assert {item.name for item in private.iterdir()} == set(files)
            assert {name: (private / name).read_bytes() for name in files} == files
            assert all(
                stat.S_IMODE((private / name).stat().st_mode) & 0o222 == 0
                for name in files
            )
            tokenizer_calls.append((path, dict(kwargs)))
            return object()

    class FakeModel:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def eval(self) -> "FakeModel":
            self.actions.append("eval")
            return self

        def to(self, device: str) -> "FakeModel":
            self.actions.append(f"to:{device}")
            return self

    fake_model = FakeModel()

    class ModelClass:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> object:
            model_calls.append((path, dict(kwargs)))
            return fake_model

    modules = SimpleNamespace(
        torch=SimpleNamespace(
            bfloat16="bf16-sentinel",
            version=SimpleNamespace(hip="7.2.1"),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        ),
        auto_tokenizer_cls=TokenizerClass,
        auto_model_cls=ModelClass,
    )
    runtime = model_module.load_pinned_local_runtime(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        snapshot_dir=snapshot,
        model_manifest=manifest,
        runtime_modules=modules,
    )
    assert runtime is not None
    private_path = Path(tokenizer_calls[0][0])
    assert private_path.is_dir()
    assert private_path.parent.parent == Path("/var/tmp")
    assert tokenizer_calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert model_calls == [
        (
            private_path,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "torch_dtype": "bf16-sentinel",
            },
        )
    ]
    assert fake_model.actions == ["eval", "to:cuda"]
    (snapshot / "config.json").write_bytes(b"mutated source")
    assert (private_path / "config.json").read_bytes() == files["config.json"]


def test_injected_loader_rejects_non_rocm_before_tokenizer_or_model(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    manifest = _write_snapshot(snapshot, {"config.json": b"config"})
    touched: list[str] = []
    loader = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: touched.append("load")
    )
    modules = SimpleNamespace(
        torch=SimpleNamespace(
            bfloat16="bf16",
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        ),
        auto_tokenizer_cls=loader,
        auto_model_cls=loader,
    )
    with pytest.raises(RuntimeError, match="ROCm"):
        model_module.load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            snapshot_dir=snapshot,
            model_manifest=manifest,
            runtime_modules=modules,
        )
    assert touched == []


def test_private_snapshot_rejects_hash_or_inventory_mismatch_before_runtime_access(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    manifest = _write_snapshot(snapshot, {"config.json": b"approved"})
    (snapshot / "config.json").write_bytes(b"tampered")
    touched: list[str] = []
    modules = SimpleNamespace(
        torch=SimpleNamespace(
            bfloat16="bf16",
            version=SimpleNamespace(hip="7.2.1"),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        ),
        auto_tokenizer_cls=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: touched.append("tokenizer")
        ),
        auto_model_cls=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: touched.append("model")
        ),
    )
    with pytest.raises(RuntimeError, match="private model snapshot"):
        model_module.load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            snapshot_dir=snapshot,
            model_manifest=manifest,
            runtime_modules=modules,
        )
    assert touched == []


def test_private_snapshot_binds_the_opened_source_blob_identity(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    manifest = _write_snapshot(snapshot, {"config.json": b"approved"})
    files = [dict(row) for row in manifest["files"]]  # type: ignore[index]
    files[0]["blob_id"] = "substituted-blob"
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    substituted = {**payload, "manifest_sha256": canonical_sha256(payload)}
    with pytest.raises(RuntimeError, match="private model snapshot.*blob"):
        model_module.load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            snapshot_dir=snapshot,
            model_manifest=substituted,
            runtime_modules=pytest.fail,
        )

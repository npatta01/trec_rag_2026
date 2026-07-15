from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_local_model as model_module
from trec_rag.adaptive_obligation_v2_local_model import (
    V2LocalJsonModel,
    verify_inference_approval,
)
from trec_rag.adaptive_obligation_v2_propose import MODEL_ID, MODEL_REVISION


def _preflight() -> dict[str, object]:
    return {
        "receipt_sha256": "a" * 64,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
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


def test_v2_local_json_model_interface_exists() -> None:
    assert V2LocalJsonModel.__name__ == "V2LocalJsonModel"


def test_approval_requires_exact_stage_and_preflight_binding() -> None:
    assert verify_inference_approval(_approval(), _preflight()) == _approval()
    audited = {**_approval(), "created_by": "independent-approver"}
    assert verify_inference_approval(audited, _preflight()) == audited
    wrong = {**_approval(), "stage": "validation"}
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        verify_inference_approval(wrong, _preflight())
    null_hash_approval = {**_approval(), "preflight_sha256": None}
    null_hash_preflight = {**_preflight(), "receipt_sha256": None}
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        verify_inference_approval(null_hash_approval, null_hash_preflight)


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
        }
    ]
    assert adapter.generate([], {}, max_new_tokens=256) == (b"{}", 1)


def test_injected_loader_uses_exact_offline_safetensor_bfloat16_rocm_contract(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    tokenizer_calls: list[tuple[object, dict[str, object]]] = []
    model_calls: list[tuple[object, dict[str, object]]] = []

    class TokenizerClass:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> object:
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
        runtime_modules=modules,
    )
    assert runtime is not None
    assert tokenizer_calls == [
        (snapshot, {"local_files_only": True, "trust_remote_code": False})
    ]
    assert model_calls == [
        (
            snapshot,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "torch_dtype": "bf16-sentinel",
            },
        )
    ]
    assert fake_model.actions == ["eval", "to:cuda"]


def test_injected_loader_rejects_non_rocm_before_tokenizer_or_model(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
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
            runtime_modules=modules,
        )
    assert touched == []

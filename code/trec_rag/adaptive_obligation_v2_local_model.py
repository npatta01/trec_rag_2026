"""Separately approval-gated, pinned local Qwen JSON generation adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from .adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
)


APPROVAL_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-approval-v1"


def verify_inference_approval(
    approval: object, preflight: Mapping[str, object]
) -> dict[str, object]:
    """Require the exact proposal-stage approval bound to this preflight."""

    if not isinstance(approval, Mapping) or not isinstance(preflight, Mapping):
        raise PermissionError("proposal inference approval required")
    receipt_sha256 = preflight.get("receipt_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
    ):
        raise PermissionError("proposal inference approval required")
    required = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "stage": "proposal",
        "preflight_sha256": receipt_sha256,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "approved": True,
    }
    if not set(required) <= set(approval) or any(
        approval.get(name) != value for name, value in required.items()
    ):
        raise PermissionError("proposal inference approval required")
    if (
        preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
        or preflight.get("primary_call_count") != 48
        or preflight.get("retry_call_ceiling") != 48
    ):
        raise PermissionError("proposal inference approval required")
    return dict(approval)


def _default_runtime_modules() -> object:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return SimpleNamespace(
        torch=torch,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForCausalLM,
    )


class _PinnedLocalRuntime:
    def __init__(self, *, modules: object, tokenizer: object, model: object) -> None:
        self._modules = modules
        self._tokenizer = tokenizer
        self._model = model

    def generate_json_bytes(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[bytes, int]:
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("proposal JSON schema is required")
        if do_sample is not False:
            raise ValueError("proposal generation must be deterministic")
        tokenizer = self._tokenizer
        encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [dict(row) for row in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("local tokenizer returned invalid model inputs")
        device_inputs = {
            name: tensor.to("cuda")  # type: ignore[attr-defined]
            for name, tensor in encoded.items()
        }
        input_ids = device_inputs["input_ids"]
        prompt_tokens = int(input_ids.shape[-1])  # type: ignore[attr-defined]
        torch_module = self._modules.torch  # type: ignore[attr-defined]
        with torch_module.inference_mode():
            output = self._model.generate(  # type: ignore[attr-defined]
                **device_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output[0][prompt_tokens:]
        output_token_count = int(generated.shape[-1])
        text = tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[attr-defined]
        if not isinstance(text, str):
            raise ValueError("local tokenizer decode did not return text")
        return text.encode("utf-8"), output_token_count


def load_pinned_local_runtime(
    *,
    model_id: str,
    revision: str,
    local_files_only: bool,
    snapshot_dir: Path = MODEL_SNAPSHOT,
    runtime_modules: object | None = None,
) -> object:
    """Construct the exact local safetensor model on a verified ROCm device."""

    if (
        model_id != MODEL_ID
        or revision != MODEL_REVISION
        or local_files_only is not True
    ):
        raise RuntimeError("pinned local proposal model identity differs")
    snapshot = Path(snapshot_dir)
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != revision:
        raise RuntimeError("pinned local proposal model snapshot is missing or unsafe")
    modules = runtime_modules or _default_runtime_modules()
    torch_module = modules.torch  # type: ignore[attr-defined]
    cuda = getattr(torch_module, "cuda", None)
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    if (
        not hip
        or cuda is None
        or not cuda.is_available()
        or int(cuda.device_count()) < 1
    ):
        raise RuntimeError("proposal inference requires an available ROCm device")
    tokenizer = modules.auto_tokenizer_cls.from_pretrained(  # type: ignore[attr-defined]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = modules.auto_model_cls.from_pretrained(  # type: ignore[attr-defined]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch_module.bfloat16,
    )
    model = model.eval()  # type: ignore[attr-defined]
    model = model.to("cuda")  # type: ignore[attr-defined]
    return _PinnedLocalRuntime(modules=modules, tokenizer=tokenizer, model=model)


class V2LocalJsonModel:
    """Approval-gated facade over the pinned local JSON runtime."""

    def __init__(self, *, approval: object, preflight: Mapping[str, object]) -> None:
        verify_inference_approval(approval, preflight)
        self._runtime = load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
    ) -> tuple[bytes, int]:
        return self._runtime.generate_json_bytes(  # type: ignore[attr-defined,no-any-return]
            messages,
            schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

"""Pinned local JSON-only adapter for the adaptive evidence discovery model."""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    / MODEL_REVISION
)


_O1_PROPERTIES: dict[str, object] = {
    "label": {"type": "string", "minLength": 1},
    "scope_rationale": {"type": "string", "minLength": 1},
    "subject": {"type": "string", "minLength": 1},
    "population": {
        "type": "string",
        "minLength": 1,
        "description": "Population already named by the frozen parent O0 scope.",
    },
    "domain": {
        "type": "string",
        "minLength": 1,
        "description": "Domain already named by the frozen parent O0 scope.",
    },
    "relation": {"type": "string", "minLength": 1},
    "support_document_id": {"type": "string", "minLength": 1},
    "support_span": {"type": "string", "minLength": 1},
}
_O1_REQUIRED = list(_O1_PROPERTIES)
_N1_PROPERTIES: dict[str, object] = {
    "subject": {
        "type": "string",
        "minLength": 1,
        "description": "One atomic subject; no coordination or list.",
    },
    "relation": {
        "type": "string",
        "minLength": 1,
        "description": "One atomic relation; no coordination or second clause.",
    },
    "object": {
        "type": "string",
        "minLength": 1,
        "description": "One atomic object; no coordination or list.",
    },
    "support_document_id": {"type": "string", "minLength": 1},
    "support_span": {
        "type": "string",
        "minLength": 1,
        "description": "One exact single-sentence span supporting only this SRO fact.",
    },
}
_N1_REQUIRED = list(_N1_PROPERTIES)

DISCOVERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "o1", "n1"],
    "properties": {
        "status": {"type": "string", "enum": ["supported", "unsupported"]},
        "o1": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": _O1_REQUIRED,
                "properties": _O1_PROPERTIES,
            },
        },
        "n1": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": _N1_REQUIRED,
                "properties": _N1_PROPERTIES,
            },
        },
    },
}

def validate_json_schema(value: object, schema: Mapping[str, object], path: str = "$") -> None:
    """Validate the frozen, deliberately small JSON-schema subset used here."""

    expected_type = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
    }
    if expected_type in type_checks and not type_checks[str(expected_type)](value):
        raise ValueError(f"JSON schema {path} must be {expected_type}")
    raw_enum = schema.get("enum")
    if isinstance(raw_enum, list) and value not in raw_enum:
        raise ValueError(f"JSON schema {path} is outside its enum")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ValueError(f"JSON schema {path} is shorter than minLength")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [name for name in required if name not in value]
            if missing:
                raise ValueError(f"JSON schema {path} is missing {missing[0]}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"JSON schema {path} properties are invalid")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"JSON schema {path} has extra property {extra[0]}")
        for name, child in properties.items():
            if name in value:
                if not isinstance(child, Mapping):
                    raise ValueError(f"JSON schema {path}.{name} is invalid")
                validate_json_schema(value[name], child, f"{path}.{name}")
    if isinstance(value, list):
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ValueError(f"JSON schema {path} exceeds maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_json_schema(item, item_schema, f"{path}[{index}]")


def _host_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _default_runtime() -> object:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return SimpleNamespace(
        torch=torch,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForCausalLM,
        clock=time.perf_counter,
        host_memory_bytes=_host_memory_bytes,
    )


class LocalJsonModel:
    """Load the authenticated local Qwen snapshot once and return schema-valid JSON."""

    def __init__(self, *, runtime: object | None = None) -> None:
        self.runtime = runtime or _default_runtime()
        torch_module = self.runtime.torch  # type: ignore[attr-defined]
        cuda = getattr(torch_module, "cuda", None)
        hip = getattr(getattr(torch_module, "version", None), "hip", None)
        if (
            cuda is None
            or not cuda.is_available()
            or int(cuda.device_count()) < 1
            or (runtime is None and not hip)
        ):
            raise RuntimeError("local JSON discovery requires an available ROCm cuda device")
        if not MODEL_SNAPSHOT.is_dir():
            raise RuntimeError(f"pinned local model snapshot is missing: {MODEL_SNAPSHOT}")
        torch_module.manual_seed(0)
        cuda.manual_seed_all(0)
        cuda.reset_peak_memory_stats()
        self.device_name = str(cuda.get_device_name(0))
        self.started = float(self.runtime.clock())  # type: ignore[attr-defined]
        self.initial_host_memory = int(
            self.runtime.host_memory_bytes()  # type: ignore[attr-defined]
        )
        tokenizer_loader = self.runtime.auto_tokenizer_cls  # type: ignore[attr-defined]
        self.tokenizer = tokenizer_loader.from_pretrained(
            MODEL_SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = self.runtime.auto_model_cls.from_pretrained(  # type: ignore[attr-defined]
            MODEL_SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch_module.bfloat16,
        )
        self.model = self.model.eval()
        self.model = self.model.to("cuda")
        self.generation_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int = 1200,
    ) -> dict[str, object]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = int(inputs.input_ids.shape[1])
        with self.runtime.torch.inference_mode():  # type: ignore[attr-defined]
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        try:
            completion_tokens = output[0, input_length:]
        except TypeError:
            completion_tokens = output[0][input_length:]
        completion = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=True,
        )
        try:
            value = json.loads(completion)
        except json.JSONDecodeError as exc:
            raise ValueError("local model completion is not one exact JSON value") from exc
        if not isinstance(value, dict):
            raise ValueError("local model completion must be a JSON object")
        validate_json_schema(value, schema)
        self.generation_count += 1
        self.input_token_count += input_length
        try:
            self.output_token_count += int(completion_tokens.shape[-1])
        except AttributeError:
            self.output_token_count += len(completion_tokens)
        return value

    def execution_receipt(self) -> dict[str, object]:
        torch_module = self.runtime.torch  # type: ignore[attr-defined]
        return {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_snapshot": str(MODEL_SNAPSHOT),
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": "bfloat16",
            "eval_mode": True,
            "device": "cuda",
            "execution_backend": "rocm",
            "device_name": self.device_name,
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "torch_hip_version": str(
                getattr(getattr(torch_module, "version", None), "hip", "unknown")
            ),
            "seed": 0,
            "do_sample": False,
            "generation_count": self.generation_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "elapsed_seconds": max(
                0.0, float(self.runtime.clock()) - self.started  # type: ignore[attr-defined]
            ),
            "peak_device_memory_bytes": int(
                torch_module.cuda.max_memory_allocated()
            ),
            "peak_host_memory_bytes": max(
                self.initial_host_memory,
                int(self.runtime.host_memory_bytes()),  # type: ignore[attr-defined]
            ),
        }

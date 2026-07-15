"""Disabled v1 local-model adapter retained only for terminal-history verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    / MODEL_REVISION
)
DISCOVERY_V1_TERMINAL_ONLY = "discovery v1 is terminal-only"


_O1_PROPERTIES: dict[str, object] = {
    "label": {
        "type": "string",
        "minLength": 1,
        "description": (
            "Short nominal category ending in an abstract head noun; never a "
            "sentence, assertion, number, date, currency, or quantity."
        ),
    },
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

class LocalJsonModel:
    """Reject all v1 model access; future discovery requires a versioned v2 API."""

    def __init__(self, *, runtime: object | None = None) -> None:
        raise RuntimeError(DISCOVERY_V1_TERMINAL_ONLY)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int = 1200,
    ) -> dict[str, object]:
        raise RuntimeError(DISCOVERY_V1_TERMINAL_ONLY)

    def execution_receipt(self) -> dict[str, object]:
        raise RuntimeError(DISCOVERY_V1_TERMINAL_ONLY)

"""Strict provider-independent filtering for validated splice operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trec_rag.bounded_splice import SpliceOperation


class OperationScreenValidationError(ValueError):
    """Raised when an operation-screen response is incomplete or malformed."""


@dataclass(frozen=True)
class ScreenDecision:
    operation_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class OperationScreenResult:
    decisions: tuple[ScreenDecision, ...]
    accepted_operations: tuple[SpliceOperation, ...]


_GATE_FIELDS = (
    "fully_supported",
    "atomic",
    "material",
    "nonredundant",
    "replacement_safe",
)
_DECISION_FIELDS = frozenset({"operation_id", *_GATE_FIELDS})


def _operation_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("operation_count must be a positive integer")
    return value


def operation_screen_response_schema(operation_count: int) -> dict[str, object]:
    """Return a strict decision-only schema sized to one frozen operation set."""

    count = _operation_count(operation_count)
    ids = [f"op{index:03d}" for index in range(1, count + 1)]
    decision: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation_id", *_GATE_FIELDS],
        "properties": {
            "operation_id": {"type": "string", "enum": ids},
            **{field: {"type": "boolean"} for field in _GATE_FIELDS},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": decision,
            }
        },
    }


def operation_ids(operations: Sequence[SpliceOperation]) -> tuple[str, ...]:
    """Return stable one-based aliases for an ordered validated operation set."""

    if not operations:
        raise OperationScreenValidationError("operation screen requires at least one operation")
    if any(not isinstance(operation, SpliceOperation) for operation in operations):
        raise OperationScreenValidationError(
            "operation screen requires validated SpliceOperation values"
        )
    return tuple(f"op{index:03d}" for index in range(1, len(operations) + 1))


def validate_operation_screen_payload(
    payload: object,
    operations: Sequence[SpliceOperation],
) -> OperationScreenResult:
    """Validate every decision and derive the accepted operation subset."""

    expected_ids = operation_ids(operations)
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise OperationScreenValidationError(
            "operation screen payload must contain exactly decisions"
        )
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list):
        raise OperationScreenValidationError("operation screen decisions must be an array")
    if len(raw_decisions) != len(expected_ids):
        raise OperationScreenValidationError(
            "operation screen must decide every operation exactly once"
        )

    actual_ids: list[str] = []
    decisions: list[ScreenDecision] = []
    accepted: list[SpliceOperation] = []
    for index, (raw, operation) in enumerate(zip(raw_decisions, operations, strict=True)):
        if not isinstance(raw, dict) or set(raw) != _DECISION_FIELDS:
            raise OperationScreenValidationError(
                f"operation screen decision[{index}] has invalid fields"
            )
        operation_id = raw["operation_id"]
        if not isinstance(operation_id, str):
            raise OperationScreenValidationError(
                f"operation screen decision[{index}] operation_id must be a string"
            )
        actual_ids.append(operation_id)
        for field in _GATE_FIELDS:
            if type(raw[field]) is not bool:
                raise OperationScreenValidationError(
                    f"operation screen decision[{index}] {field} must be a boolean"
                )
        rejection_reasons = tuple(field for field in _GATE_FIELDS if not raw[field])
        is_accepted = not rejection_reasons
        decisions.append(
            ScreenDecision(
                operation_id=operation_id,
                accepted=is_accepted,
                rejection_reasons=rejection_reasons,
            )
        )
        if is_accepted:
            accepted.append(operation)

    if tuple(actual_ids) != expected_ids:
        raise OperationScreenValidationError(
            "operation screen IDs must match the frozen operation order exactly"
        )
    return OperationScreenResult(
        decisions=tuple(decisions),
        accepted_operations=tuple(accepted),
    )

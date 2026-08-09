"""Strict bounded splice validation and deterministic application helpers.

The splice response is intentionally small: a provider may keep a validated draft or
describe a bounded set of edits against the draft's original answer indexes.  This
module contains no provider or organizer integration; callers validate the complete
patch before asking :func:`apply_splice_operations` to assemble a candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from collections.abc import Sequence
from typing import Any


class SpliceValidationError(ValueError):
    """Raised when a provider splice cannot be authenticated or applied safely."""


@dataclass(frozen=True)
class SpliceOperation:
    """One immutable edit against an original answer array."""

    start_index: int
    delete_count: int
    text: str
    citations: tuple[str, ...]
    audit_card_ids: tuple[str, ...]


def splice_response_schema() -> dict[str, object]:
    """Return the strict JSON schema used for the provider splice response."""

    new_object: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "citations"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "citations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string"},
            },
        },
    }
    operation: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["start_index", "delete_count", "new_object", "audit_card_ids"],
        "properties": {
            "start_index": {"type": "integer", "minimum": 0},
            "delete_count": {"type": "integer", "enum": [0, 1, 2, 3]},
            "new_object": new_object,
            "audit_card_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "operations"],
        "properties": {
            "decision": {"type": "string", "enum": ["keep_draft", "edit"]},
            "operations": {
                "type": "array",
                "minItems": 0,
                "maxItems": 6,
                "items": operation,
            },
        },
    }


_ROOT_FIELDS = frozenset({"decision", "operations"})
_OPERATION_FIELDS = frozenset(
    {"start_index", "delete_count", "new_object", "audit_card_ids"}
)
_ANSWER_FIELDS = frozenset({"text", "citations"})
_DECISIONS = frozenset({"keep_draft", "edit"})
_DELETE_COUNTS = frozenset({0, 1, 2, 3})
_MAX_OPERATIONS = 6
_MAX_INSERTIONS = 4
_MAX_TOUCHED_OBJECTS = 8
_MAX_WORDS = 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_exact_fields(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpliceValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise SpliceValidationError(f"{label} has invalid fields ({'; '.join(details)})")
    return value


def _draft_answers(draft: object) -> list[dict[str, Any]]:
    if not isinstance(draft, dict):
        raise SpliceValidationError("draft must be an object")
    answer = draft.get("answer")
    if not isinstance(answer, list):
        raise SpliceValidationError("draft answer must be an array")
    checked: list[dict[str, Any]] = []
    for index, item in enumerate(answer):
        if not isinstance(item, dict):
            raise SpliceValidationError(f"draft answer[{index}] must be an object")
        text = item.get("text")
        if not isinstance(text, str):
            raise SpliceValidationError(f"draft answer[{index}] text must be a string")
        checked.append(item)
    return checked


def _nonempty_strings(value: object, *, label: str, minimum: int, maximum: int | None = None) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SpliceValidationError(f"{label} must be an array")
    if len(value) < minimum or (maximum is not None and len(value) > maximum):
        bound = f"at most {maximum}" if maximum is not None else f"at least {minimum}"
        raise SpliceValidationError(f"{label} must contain {bound} items")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SpliceValidationError(f"{label} must contain non-empty strings")
        result.append(item)
    if len(result) != len(set(result)):
        raise SpliceValidationError(f"{label} contains duplicate values")
    return tuple(result)


def _parse_operation(
    raw: object,
    *,
    operation_index: int,
    allowed_docids: tuple[str, ...],
    allowed_audit_card_ids: tuple[str, ...],
) -> SpliceOperation:
    operation = _require_exact_fields(
        raw,
        _OPERATION_FIELDS,
        f"operation[{operation_index}]",
    )
    start_index = operation["start_index"]
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise SpliceValidationError(f"operation[{operation_index}] start_index must be an integer")
    if start_index < 0:
        raise SpliceValidationError(f"operation[{operation_index}] start_index must be non-negative")

    delete_count = operation["delete_count"]
    if isinstance(delete_count, bool) or not isinstance(delete_count, int):
        raise SpliceValidationError(f"operation[{operation_index}] delete_count must be an integer")
    if delete_count not in _DELETE_COUNTS:
        raise SpliceValidationError(
            f"operation[{operation_index}] delete_count must be one of 0, 1, 2, or 3"
        )

    answer = _require_exact_fields(
        operation["new_object"],
        _ANSWER_FIELDS,
        f"operation[{operation_index}] new_object",
    )
    text = answer["text"]
    if not isinstance(text, str) or not text.strip():
        raise SpliceValidationError(f"operation[{operation_index}] new_object text must be non-empty")
    citations = _nonempty_strings(
        answer["citations"],
        label=f"operation[{operation_index}] citations",
        minimum=1,
        maximum=3,
    )
    allowed_docs = set(allowed_docids)
    foreign_docs = [docid for docid in citations if docid not in allowed_docs]
    if foreign_docs:
        raise SpliceValidationError(
            f"operation[{operation_index}] citations contain unauthenticated docid(s): "
            + ", ".join(foreign_docs)
        )

    audit_card_ids = _nonempty_strings(
        operation["audit_card_ids"],
        label=f"operation[{operation_index}] audit_card_ids",
        minimum=1,
    )
    allowed_cards = set(allowed_audit_card_ids)
    unknown_cards = [card_id for card_id in audit_card_ids if card_id not in allowed_cards]
    if unknown_cards:
        raise SpliceValidationError(
            f"operation[{operation_index}] has unknown audit-card ID(s): "
            + ", ".join(unknown_cards)
        )
    return SpliceOperation(
        start_index=start_index,
        delete_count=delete_count,
        text=text,
        citations=citations,
        audit_card_ids=audit_card_ids,
    )


def _validate_operation_geometry(
    operations: tuple[SpliceOperation, ...],
    *,
    answer_count: int,
    answer: Sequence[dict[str, Any]],
) -> None:
    starts = [operation.start_index for operation in operations]
    if len(starts) != len(set(starts)):
        raise SpliceValidationError("operations must have unique start indexes")

    insertion_count = sum(operation.delete_count == 0 for operation in operations)
    if insertion_count > _MAX_INSERTIONS:
        raise SpliceValidationError("at most four insertions are allowed")
    touched_count = sum(operation.delete_count for operation in operations)
    if touched_count > _MAX_TOUCHED_OBJECTS:
        raise SpliceValidationError(
            "at most eight touched original answer objects are allowed"
        )

    for index, operation in enumerate(operations):
        start = operation.start_index
        end = start + operation.delete_count
        if start > answer_count:
            raise SpliceValidationError(f"operation[{index}] start index is out of range")
        if operation.delete_count == 0:
            continue
        if start == answer_count or end > answer_count:
            raise SpliceValidationError(f"operation[{index}] replacement range is out of range")

    positive = [operation for operation in operations if operation.delete_count > 0]
    for left_index, left in enumerate(positive):
        left_end = left.start_index + left.delete_count
        for right in positive[left_index + 1 :]:
            right_end = right.start_index + right.delete_count
            if left.start_index < right_end and right.start_index < left_end:
                raise SpliceValidationError("replacement ranges overlap")
    for insertion in (operation for operation in operations if operation.delete_count == 0):
        for replacement in positive:
            replacement_end = replacement.start_index + replacement.delete_count
            if replacement.start_index <= insertion.start_index < replacement_end:
                raise SpliceValidationError(
                    "insertion is inside or at the start of a replacement range"
                )

    original_words = sum(len(str(item["text"]).split()) for item in answer)
    assembled_words = original_words
    for operation in operations:
        if operation.delete_count:
            assembled_words -= sum(
                len(str(item["text"]).split())
                for item in answer[operation.start_index : operation.start_index + operation.delete_count]
            )
        assembled_words += len(operation.text.split())
    if assembled_words > _MAX_WORDS:
        raise SpliceValidationError(f"assembled answer exceeds {_MAX_WORDS} words")


def validate_splice_payload(
    draft: dict[str, Any],
    payload: object,
    allowed_docids: tuple[str, ...],
    allowed_audit_card_ids: tuple[str, ...],
) -> tuple[SpliceOperation, ...] | None:
    """Validate a provider response against an immutable draft.

    The returned tuple contains only typed operations.  ``None`` is the valid
    ``keep_draft`` response.  All checks happen before a caller can apply or
    copy anything, so malformed patches cannot partially affect the draft.
    """

    answer = _draft_answers(draft)
    root = _require_exact_fields(payload, _ROOT_FIELDS, "splice payload")
    decision = root["decision"]
    if not isinstance(decision, str) or decision not in _DECISIONS:
        raise SpliceValidationError("decision must be keep_draft or edit")
    raw_operations = root["operations"]
    if not isinstance(raw_operations, list):
        raise SpliceValidationError("operations must be an array")
    if len(raw_operations) > _MAX_OPERATIONS:
        raise SpliceValidationError("at most six operations are allowed")
    if decision == "keep_draft":
        if raw_operations:
            raise SpliceValidationError("keep_draft requires an empty operations array")
        return None
    if not raw_operations:
        raise SpliceValidationError("edit requires at least one operation")

    parsed = tuple(
        _parse_operation(
            raw,
            operation_index=index,
            allowed_docids=allowed_docids,
            allowed_audit_card_ids=allowed_audit_card_ids,
        )
        for index, raw in enumerate(raw_operations)
    )
    seen_audit_card_ids: set[str] = set()
    for operation in parsed:
        repeated = seen_audit_card_ids.intersection(operation.audit_card_ids)
        if repeated:
            raise SpliceValidationError(
                "audit-card IDs must be unique across operations: " + ", ".join(sorted(repeated))
            )
        seen_audit_card_ids.update(operation.audit_card_ids)
    _validate_operation_geometry(parsed, answer_count=len(answer), answer=answer)
    return parsed


def apply_splice_operations(
    draft: dict[str, Any],
    operations: Sequence[SpliceOperation],
) -> dict[str, Any]:
    """Deep-copy ``draft`` and apply validated operations deterministically."""

    _draft_answers(draft)
    if not isinstance(draft.get("references"), list):
        raise SpliceValidationError("draft references must be an array")
    result = deepcopy(draft)
    references = result["references"]
    reference_indexes = {docid: index for index, docid in enumerate(references)}
    for operation in operations:
        if not isinstance(operation, SpliceOperation):
            raise SpliceValidationError("apply_splice_operations requires SpliceOperation values")
        for docid in operation.citations:
            if docid not in reference_indexes:
                reference_indexes[docid] = len(references)
                references.append(docid)

    assembled_answer = result["answer"]
    for operation in sorted(operations, key=lambda item: item.start_index, reverse=True):
        new_object = {
            "text": operation.text,
            "citations": [reference_indexes[docid] for docid in operation.citations],
        }
        assembled_answer[operation.start_index : operation.start_index + operation.delete_count] = [
            new_object
        ]
    return result


def validate_repaired_splice_payload(
    draft: dict[str, Any],
    repaired_payload: object,
    initial_payload: object,
    allowed_docids: tuple[str, ...],
    allowed_audit_card_ids: tuple[str, ...],
) -> tuple[SpliceOperation, ...] | None:
    """Validate a repair while preserving the initial response's answer objects."""

    # ``keep_draft`` has no answer object to authorize and can be accepted as
    # the repair's explicit fallback.  For edits, authorize every proposed
    # object before running normal geometry, citation, and budget validation.
    repaired_root = _require_exact_fields(repaired_payload, _ROOT_FIELDS, "repaired splice payload")
    if repaired_root["decision"] == "edit":
        allowed_objects: set[str] = set()
        if isinstance(initial_payload, dict) and isinstance(initial_payload.get("operations"), list):
            for index, raw in enumerate(initial_payload["operations"]):
                if isinstance(raw, dict) and "new_object" in raw:
                    try:
                        allowed_objects.add(_canonical_json(raw["new_object"]))
                    except (TypeError, ValueError) as error:
                        raise SpliceValidationError(
                            f"initial operation[{index}] has an unserializable new_object"
                        ) from error
        repaired_operations = repaired_root["operations"]
        if not isinstance(repaired_operations, list):
            raise SpliceValidationError("repaired operations must be an array")
        for index, raw in enumerate(repaired_operations):
            if not isinstance(raw, dict) or "new_object" not in raw:
                raise SpliceValidationError(
                    f"repaired operation[{index}] new_object is not from the initial payload"
                )
            try:
                candidate = _canonical_json(raw["new_object"])
            except (TypeError, ValueError) as error:
                raise SpliceValidationError(
                    f"repaired operation[{index}] new_object is not from the initial payload"
                ) from error
            if candidate not in allowed_objects:
                raise SpliceValidationError(
                    f"repaired operation[{index}] new_object differs from the initial payload"
                )
    return validate_splice_payload(
        draft,
        repaired_payload,
        allowed_docids,
        allowed_audit_card_ids,
    )

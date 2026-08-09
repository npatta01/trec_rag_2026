"""Focused tests for strict operation-level Luna screening."""

from __future__ import annotations

from copy import deepcopy

import pytest

from trec_rag.bounded_splice import SpliceOperation
from trec_rag.operation_screen import (
    OperationScreenValidationError,
    operation_ids,
    operation_screen_response_schema,
    validate_operation_screen_payload,
)


def _operations() -> tuple[SpliceOperation, ...]:
    return (
        SpliceOperation(
            start_index=1,
            delete_count=0,
            text="A material supported insertion.",
            citations=("DOC-INSERT",),
            audit_card_ids=("a001",),
        ),
        SpliceOperation(
            start_index=2,
            delete_count=1,
            text="A proposed replacement.",
            citations=("DOC-REPLACE",),
            audit_card_ids=("a002",),
        ),
    )


def _passing_payload() -> dict[str, object]:
    return {
        "decisions": [
            {
                "operation_id": operation_id,
                "fully_supported": True,
                "atomic": True,
                "material": True,
                "nonredundant": True,
                "replacement_safe": True,
            }
            for operation_id in ("op001", "op002")
        ]
    }


def test_schema_and_validator_derive_acceptance_from_every_gate() -> None:
    operations = _operations()
    schema = operation_screen_response_schema(len(operations))

    decisions_schema = schema["properties"]["decisions"]
    assert decisions_schema["minItems"] == 2
    assert decisions_schema["maxItems"] == 2
    assert decisions_schema["items"]["additionalProperties"] is False
    assert set(decisions_schema["items"]["required"]) == {
        "operation_id",
        "fully_supported",
        "atomic",
        "material",
        "nonredundant",
        "replacement_safe",
    }

    result = validate_operation_screen_payload(
        {
            "decisions": [
                {
                    "operation_id": "op001",
                    "fully_supported": True,
                    "atomic": True,
                    "material": True,
                    "nonredundant": True,
                    "replacement_safe": True,
                },
                {
                    "operation_id": "op002",
                    "fully_supported": True,
                    "atomic": True,
                    "material": True,
                    "nonredundant": True,
                    "replacement_safe": False,
                },
            ]
        },
        operations,
    )

    assert result.accepted_operations == (operations[0],)
    assert result.decisions[0].accepted is True
    assert result.decisions[0].rejection_reasons == ()
    assert result.decisions[1].accepted is False
    assert result.decisions[1].rejection_reasons == ("replacement_safe",)


def test_validator_rejects_missing_duplicate_unknown_or_reordered_ids() -> None:
    operations = _operations()
    bad_payloads = []

    missing = _passing_payload()
    missing["decisions"] = missing["decisions"][:-1]  # type: ignore[index]
    bad_payloads.append(missing)

    duplicate = _passing_payload()
    duplicate["decisions"][1]["operation_id"] = "op001"  # type: ignore[index]
    bad_payloads.append(duplicate)

    unknown = _passing_payload()
    unknown["decisions"][1]["operation_id"] = "op999"  # type: ignore[index]
    bad_payloads.append(unknown)

    reordered = _passing_payload()
    reordered["decisions"] = list(reversed(reordered["decisions"]))  # type: ignore[arg-type]
    bad_payloads.append(reordered)

    for payload in bad_payloads:
        with pytest.raises(OperationScreenValidationError):
            validate_operation_screen_payload(payload, operations)


def test_validator_rejects_extra_fields_and_non_boolean_gates() -> None:
    operations = _operations()

    extra_root = {**_passing_payload(), "accept": True}
    with pytest.raises(OperationScreenValidationError, match="exactly decisions"):
        validate_operation_screen_payload(extra_root, operations)

    extra_decision = deepcopy(_passing_payload())
    extra_decision["decisions"][0]["accept"] = True  # type: ignore[index]
    with pytest.raises(OperationScreenValidationError, match="invalid fields"):
        validate_operation_screen_payload(extra_decision, operations)

    integer_gate = deepcopy(_passing_payload())
    integer_gate["decisions"][0]["fully_supported"] = 1  # type: ignore[index]
    with pytest.raises(OperationScreenValidationError, match="must be a boolean"):
        validate_operation_screen_payload(integer_gate, operations)


def test_operation_ids_require_nonempty_validated_operations() -> None:
    with pytest.raises(OperationScreenValidationError, match="at least one"):
        operation_ids(())

    with pytest.raises(OperationScreenValidationError, match="SpliceOperation"):
        operation_ids(("not-validated",))  # type: ignore[arg-type]

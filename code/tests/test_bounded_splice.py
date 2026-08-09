"""Focused tests for the bounded splice response and deterministic applier."""

from __future__ import annotations

from copy import deepcopy

import pytest

from trec_rag import bounded_splice
from trec_rag.bounded_splice import SpliceOperation, SpliceValidationError, splice_response_schema


ALLOWED_DOCIDS = ("doc-old-0", "doc-old-1", "doc-insert", "doc-replace", "doc-merge")
ALLOWED_AUDIT_IDS = ("a001", "a002", "a003", "a004", "a005", "a006")
AUDIT_CARD_DOCIDS = {
    card_id: ALLOWED_DOCIDS
    for card_id in ALLOWED_AUDIT_IDS
}


def _draft() -> dict[str, object]:
    return {
        "metadata": {
            "run_id": "draft-run",
            "narrative_id": "topic-1",
            "narrative": "The complete official narrative.",
            "stable": {"nested": [1, 2, 3]},
        },
        "references": ["doc-old-0", "doc-old-1"],
        "answer": [
            {"text": "Original zero.", "citations": [0]},
            {"text": "Original one.", "citations": [1]},
            {"text": "Original two.", "citations": [0, 1]},
        ],
    }


def _validate(
    draft: dict[str, object], payload: object,
    audit_card_docids: dict[str, tuple[str, ...]] | None = None,
) -> tuple[SpliceOperation, ...] | None:
    return bounded_splice.validate_splice_payload(
        draft,
        payload,
        ALLOWED_DOCIDS,
        AUDIT_CARD_DOCIDS if audit_card_docids is None else audit_card_docids,
    )


def test_splice_schema_is_strict() -> None:
    schema = splice_response_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"decision", "operations"}

    operation_schema = schema["properties"]["operations"]["items"]
    assert operation_schema["additionalProperties"] is False
    assert set(operation_schema["required"]) == {
        "start_index",
        "delete_count",
        "new_object",
        "audit_card_ids",
    }
    assert operation_schema["properties"]["delete_count"]["enum"] == [0, 1, 2, 3]

    answer_schema = operation_schema["properties"]["new_object"]
    citation_schema = answer_schema["properties"]["citations"]
    assert citation_schema["minItems"] == 1
    assert citation_schema["maxItems"] == 2
    assert citation_schema["items"] == {"type": "string"}


def test_new_object_must_be_one_terminal_sentence() -> None:
    draft = _draft()

    with pytest.raises(SpliceValidationError, match="one terminal sentence"):
        _validate(
            draft,
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {
                            "text": "First supported claim. Second supported claim.",
                            "citations": ["doc-insert"],
                        },
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
        )

    with pytest.raises(SpliceValidationError, match="terminal punctuation"):
        _validate(
            draft,
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {
                            "text": "A supported but unterminated claim",
                            "citations": ["doc-insert"],
                        },
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
        )


def test_new_object_citations_must_come_from_its_named_audit_cards() -> None:
    draft = _draft()
    payload = {
        "decision": "edit",
        "operations": [
            {
                "start_index": 0,
                "delete_count": 0,
                "new_object": {
                    "text": "A card-linked claim.",
                    "citations": ["doc-replace"],
                },
                "audit_card_ids": ["a001"],
            }
        ],
    }

    with pytest.raises(SpliceValidationError, match="linked audit-card evidence"):
        _validate(draft, payload, {"a001": ("doc-insert",)})

    payload["operations"][0]["audit_card_ids"] = ["a001", "a002"]
    operations = _validate(
        draft,
        payload,
        {"a001": ("doc-insert",), "a002": ("doc-replace",)},
    )
    assert operations is not None
    assert operations[0].citations == ("doc-replace",)


def test_provider_schema_uses_only_supported_keywords() -> None:
    schema = splice_response_schema()

    assert "allOf" not in schema
    assert "if" not in schema
    assert "then" not in schema


def test_keep_draft_accepts_only_an_empty_operation_array() -> None:
    draft = _draft()

    assert _validate(draft, {"decision": "keep_draft", "operations": []}) is None

    with pytest.raises(SpliceValidationError, match="keep_draft"):
        _validate(
            draft,
            {
                "decision": "keep_draft",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {"text": "Inserted.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
        )


def test_insert_replace_and_merge_use_original_indexes_and_preserve_untouched_object() -> None:
    draft = {
        "metadata": {"run_id": "draft-run", "narrative_id": "topic-1"},
        "references": ["doc-old-0", "doc-old-1"],
        "answer": [
            {"text": "Original zero.", "citations": [0]},
            {"text": "Original one.", "citations": [1]},
            {"text": "Original two.", "citations": [0, 1]},
            {"text": "Untouched literal.", "citations": [1], "extra": {"x": [1, 2]}},
        ],
    }
    original_untouched = deepcopy(draft["answer"][3])
    payload = {
        "decision": "edit",
        "operations": [
            {
                "start_index": 0,
                "delete_count": 2,
                "new_object": {
                    "text": "Merged original material.",
                    "citations": ["doc-merge", "doc-old-0"],
                },
                "audit_card_ids": ["a001"],
            },
            {
                "start_index": 2,
                "delete_count": 1,
                "new_object": {
                    "text": "Replaced original material.",
                    "citations": ["doc-replace"],
                },
                "audit_card_ids": ["a002"],
            },
            {
                "start_index": 4,
                "delete_count": 0,
                "new_object": {
                    "text": "Inserted after the untouched object.",
                    "citations": ["doc-insert"],
                },
                "audit_card_ids": ["a003"],
            },
        ],
    }
    operations = _validate(draft, payload)
    assert operations is not None

    assembled = bounded_splice.apply_splice_operations(draft, operations)

    assert assembled["references"] == [
        "doc-old-0",
        "doc-old-1",
        "doc-merge",
        "doc-replace",
        "doc-insert",
    ]
    assert assembled["answer"] == [
        {"text": "Merged original material.", "citations": [2, 0]},
        {"text": "Replaced original material.", "citations": [3]},
        original_untouched,
        {"text": "Inserted after the untouched object.", "citations": [4]},
    ]
    assert draft["references"] == ["doc-old-0", "doc-old-1"]
    assert draft["answer"][3] == original_untouched
    assert assembled["metadata"] == draft["metadata"]
    assert assembled["metadata"] is not draft["metadata"]


def test_single_operation_indexes_are_relative_to_three_object_literal_draft() -> None:
    draft = _draft()
    original_answer = deepcopy(draft["answer"])

    insertion = _validate(
        draft,
        {
            "decision": "edit",
            "operations": [
                {
                    "start_index": 1,
                    "delete_count": 0,
                    "new_object": {"text": "Inserted.", "citations": ["doc-insert"]},
                    "audit_card_ids": ["a001"],
                }
            ],
        },
    )
    assert insertion is not None
    assert bounded_splice.apply_splice_operations(draft, insertion)["answer"] == [
        original_answer[0],
        {"text": "Inserted.", "citations": [2]},
        original_answer[1],
        original_answer[2],
    ]

    replacement = _validate(
        draft,
        {
            "decision": "edit",
            "operations": [
                {
                    "start_index": 1,
                    "delete_count": 1,
                    "new_object": {"text": "Replaced.", "citations": ["doc-replace"]},
                    "audit_card_ids": ["a002"],
                }
            ],
        },
    )
    assert replacement is not None
    assert bounded_splice.apply_splice_operations(draft, replacement)["answer"] == [
        original_answer[0],
        {"text": "Replaced.", "citations": [2]},
        original_answer[2],
    ]

    merge = _validate(
        draft,
        {
            "decision": "edit",
            "operations": [
                {
                    "start_index": 0,
                    "delete_count": 2,
                    "new_object": {"text": "Merged.", "citations": ["doc-merge"]},
                    "audit_card_ids": ["a003"],
                }
            ],
        },
    )
    assert merge is not None
    assert bounded_splice.apply_splice_operations(draft, merge)["answer"] == [
        {"text": "Merged.", "citations": [2]},
        original_answer[2],
    ]


@pytest.mark.parametrize(
    ("failure_name", "payload", "message"),
    [
        (
            "out_of_range_replacement_span",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 2,
                        "delete_count": 2,
                        "new_object": {"text": "Bad span.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
            "range",
        ),
        (
            "overlapping_replacement_ranges",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 2,
                        "new_object": {"text": "First.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a001"],
                    },
                    {
                        "start_index": 1,
                        "delete_count": 1,
                        "new_object": {"text": "Second.", "citations": ["doc-replace"]},
                        "audit_card_ids": ["a002"],
                    },
                ],
            },
            "overlap",
        ),
        (
            "six_operation_budget",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": index,
                        "delete_count": 0,
                        "new_object": {"text": f"Insert {index}.", "citations": ["doc-insert"]},
                        "audit_card_ids": [f"a00{index + 1}"],
                    }
                    for index in range(7)
                ],
            },
            "six operations",
        ),
        (
            "four_insertion_budget",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": index,
                        "delete_count": 0,
                        "new_object": {"text": f"Insert {index}.", "citations": ["doc-insert"]},
                        "audit_card_ids": [f"a00{index + 1}"],
                    }
                    for index in range(5)
                ],
            },
            "four insertions",
        ),
        (
            "eight_touched_object_budget",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 3,
                        "new_object": {"text": "A.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a001"],
                    },
                    {
                        "start_index": 3,
                        "delete_count": 3,
                        "new_object": {"text": "B.", "citations": ["doc-replace"]},
                        "audit_card_ids": ["a002"],
                    },
                    {
                        "start_index": 6,
                        "delete_count": 3,
                        "new_object": {"text": "C.", "citations": ["doc-merge"]},
                        "audit_card_ids": ["a003"],
                    },
                ],
            },
            "eight touched",
        ),
        (
            "unknown_audit_card",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {"text": "Unknown card.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a999"],
                    }
                ],
            },
            "audit",
        ),
        (
            "duplicate_raw_citations",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {
                            "text": "Duplicate citations.",
                            "citations": ["doc-insert", "doc-insert"],
                        },
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
            "duplicate",
        ),
        (
            "three_citation_limit",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {
                            "text": "Too many citations.",
                            "citations": ["doc-insert", "doc-replace", "doc-merge"],
                        },
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
            "at most 2",
        ),
        (
            "unauthenticated_raw_docid",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": 0,
                        "delete_count": 0,
                        "new_object": {"text": "Foreign citation.", "citations": ["doc-foreign"]},
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
            "docid",
        ),
        (
            "boolean_start_index",
            {
                "decision": "edit",
                "operations": [
                    {
                        "start_index": True,
                        "delete_count": 0,
                        "new_object": {"text": "Boolean index.", "citations": ["doc-insert"]},
                        "audit_card_ids": ["a001"],
                    }
                ],
            },
            "integer",
        ),
    ],
    ids=[
        "out_of_range_replacement_span",
        "overlapping_replacement_ranges",
        "six_operation_budget",
        "four_insertion_budget",
        "eight_touched_object_budget",
        "unknown_audit_card",
        "duplicate_raw_citations",
        "three_citation_limit",
        "unauthenticated_raw_docid",
        "boolean_start_index",
    ],
)
def test_invalid_splice_payload_is_rejected_atomically(
    failure_name: str,
    payload: object,
    message: str,
) -> None:
    del failure_name
    draft = _draft()
    before = deepcopy(draft)

    with pytest.raises(SpliceValidationError, match=message):
        _validate(draft, payload)

    assert draft == before


def test_repaired_payload_can_only_reuse_initial_new_objects() -> None:
    draft = _draft()
    initial_payload = {
        "decision": "edit",
        "operations": [
            {
                "start_index": 1,
                "delete_count": 1,
                "new_object": {"text": "Initial replacement.", "citations": ["doc-replace"]},
                "audit_card_ids": ["a001"],
            },
            {
                "start_index": 3,
                "delete_count": 0,
                "new_object": {"text": "Initial insertion.", "citations": ["doc-insert"]},
                "audit_card_ids": ["a002"],
            },
        ],
    }
    repaired = {
        "decision": "edit",
        "operations": [
            {
                "start_index": 3,
                "delete_count": 0,
                "new_object": {"citations": ["doc-insert"], "text": "Initial insertion."},
                "audit_card_ids": ["a002"],
            }
        ],
    }

    operations = bounded_splice.validate_repaired_splice_payload(
        draft,
        repaired,
        initial_payload,
        ALLOWED_DOCIDS,
        AUDIT_CARD_DOCIDS,
    )
    assert operations == (
        SpliceOperation(
            start_index=3,
            delete_count=0,
            text="Initial insertion.",
            citations=("doc-insert",),
            audit_card_ids=("a002",),
        ),
    )

    changed_text = deepcopy(repaired)
    changed_text["operations"][0]["new_object"]["text"] = "New prose is forbidden."
    with pytest.raises(SpliceValidationError, match="initial"):
        bounded_splice.validate_repaired_splice_payload(
            draft,
            changed_text,
            initial_payload,
            ALLOWED_DOCIDS,
            AUDIT_CARD_DOCIDS,
        )

    changed_citations = deepcopy(repaired)
    changed_citations["operations"][0]["new_object"]["citations"] = ["doc-old-0"]
    with pytest.raises(SpliceValidationError, match="initial"):
        bounded_splice.validate_repaired_splice_payload(
            draft,
            changed_citations,
            initial_payload,
            ALLOWED_DOCIDS,
            AUDIT_CARD_DOCIDS,
        )

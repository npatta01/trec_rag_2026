"""Offline guards for vLLM 0.24 structured-output schema compatibility."""

from __future__ import annotations

import pytest

from trec_rag.query_planner import query_plan_v2_json_schema
from trec_rag.query_schema_compat import (
    find_vllm_xgrammar_unsupported_features,
    require_vllm_xgrammar_compatible,
)


def test_exact_v2_schema_has_no_vllm_xgrammar_unsupported_features():
    for token_count in (1, 9, 24, 128, 10_000):
        schema = query_plan_v2_json_schema(
            topic_id=f"compat_{token_count}", token_count=token_count
        )
        assert find_vllm_xgrammar_unsupported_features(schema) == ()
        assert "uniqueItems" not in repr(schema)


@pytest.mark.parametrize(
    ("fragment", "keyword", "path_suffix"),
    [
        ({"type": "array", "uniqueItems": True}, "uniqueItems", ".uniqueItems"),
        ({"type": "array", "contains": {"type": "string"}}, "contains", ".contains"),
        ({"type": "array", "minContains": 1}, "minContains", ".minContains"),
        ({"type": "array", "maxContains": 2}, "maxContains", ".maxContains"),
        ({"type": "number", "multipleOf": 0.5}, "multipleOf", ".multipleOf"),
        (
            {"type": "object", "patternProperties": {"^x": {}}},
            "patternProperties",
            ".patternProperties",
        ),
        ({"type": "object", "propertyNames": {}}, "propertyNames", ".propertyNames"),
        ({"type": "string", "format": "regex"}, "format", ".format"),
    ],
)
def test_linter_reports_unsupported_keyword_with_json_path(
    fragment,
    keyword,
    path_suffix,
):
    schema = {
        "type": "object",
        "properties": {"nested": fragment},
    }

    (issue,) = find_vllm_xgrammar_unsupported_features(schema)

    assert issue.keyword == keyword
    assert issue.json_path == "$.properties.nested" + path_suffix
    with pytest.raises(ValueError, match=keyword):
        require_vllm_xgrammar_compatible(schema)


def test_linter_accepts_vllm_supported_string_format():
    schema = {"type": "string", "format": "date-time"}

    require_vllm_xgrammar_compatible(schema)

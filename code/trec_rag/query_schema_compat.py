"""Offline compatibility checks for vLLM 0.24 XGrammar JSON schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


VLLM_XGRAMMAR_SUPPORTED_STRING_FORMATS = frozenset(
    {
        "email",
        "date",
        "time",
        "date-time",
        "duration",
        "ipv4",
        "ipv6",
        "hostname",
        "uuid",
        "uri",
        "uri-reference",
        "uri-template",
        "json-pointer",
        "relative-json-pointer",
    }
)
VLLM_XGRAMMAR_UNSUPPORTED_ARRAY_KEYS = (
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
)
VLLM_XGRAMMAR_UNSUPPORTED_OBJECT_KEYS = (
    "patternProperties",
    "propertyNames",
)


@dataclass(frozen=True)
class SchemaCompatibilityIssue:
    json_path: str
    keyword: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"


def find_vllm_xgrammar_unsupported_features(
    schema: Mapping[str, object],
) -> tuple[SchemaCompatibilityIssue, ...]:
    """Mirror vLLM 0.24's unsupported-feature gate with JSON paths."""

    issues: list[SchemaCompatibilityIssue] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            schema_type = value.get("type")
            if schema_type in {"integer", "number"} and "multipleOf" in value:
                issues.append(
                    SchemaCompatibilityIssue(
                        _child_path(path, "multipleOf"),
                        "multipleOf",
                        "vLLM 0.24 XGrammar does not support numeric multipleOf",
                    )
                )
            if schema_type == "array":
                for keyword in VLLM_XGRAMMAR_UNSUPPORTED_ARRAY_KEYS:
                    if keyword in value:
                        issues.append(
                            SchemaCompatibilityIssue(
                                _child_path(path, keyword),
                                keyword,
                                f"vLLM 0.24 XGrammar does not support array {keyword}",
                            )
                        )
            if schema_type == "object":
                for keyword in VLLM_XGRAMMAR_UNSUPPORTED_OBJECT_KEYS:
                    if keyword in value:
                        issues.append(
                            SchemaCompatibilityIssue(
                                _child_path(path, keyword),
                                keyword,
                                f"vLLM 0.24 XGrammar does not support object {keyword}",
                            )
                        )
            if schema_type == "string" and "format" in value:
                string_format = value.get("format")
                if string_format not in VLLM_XGRAMMAR_SUPPORTED_STRING_FORMATS:
                    issues.append(
                        SchemaCompatibilityIssue(
                            _child_path(path, "format"),
                            "format",
                            f"unsupported vLLM 0.24 XGrammar string format: {string_format!r}",
                        )
                    )
            for key, child in value.items():
                visit(child, _child_path(path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(schema, "$")
    return tuple(issues)


def require_vllm_xgrammar_compatible(schema: Mapping[str, object]) -> None:
    issues = find_vllm_xgrammar_unsupported_features(schema)
    if not issues:
        return
    detail = "; ".join(
        f"{issue.json_path}: {issue.reason}" for issue in issues
    )
    raise ValueError("schema is not vLLM 0.24 XGrammar compatible: " + detail)

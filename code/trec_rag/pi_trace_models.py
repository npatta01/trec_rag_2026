"""Immutable trace-tree values and strict durable JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias


AttributeScalar: TypeAlias = str | bool | int | float
AttributeValue: TypeAlias = AttributeScalar | tuple[AttributeScalar, ...]

DEFAULT_BUNDLE_MAX_BYTES = 256 * 1024 * 1024


def _require_non_empty_string(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must be non-empty")


def _immutable(*args: object, **kwargs: object) -> None:
    raise TypeError("trace payloads are immutable")


class _FrozenDict(dict):
    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class _FrozenList(list):
    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _freeze_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict((key, _freeze_payload(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze_payload(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _freeze_attribute(value: object) -> AttributeValue:
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("trace attributes must contain only finite numbers")
        return value
    if isinstance(value, list | tuple):
        frozen = tuple(_freeze_attribute(item) for item in value)
        if any(isinstance(item, tuple) for item in frozen):
            raise TypeError("nested trace attribute sequences are not supported")
        return frozen
    raise TypeError(f"unsupported trace attribute value: {type(value).__name__}")


@dataclass(frozen=True)
class SpanSpec:
    name: str
    kind: str
    start_ns: int
    end_ns: int
    attributes: Mapping[str, AttributeValue]
    input_value: object | None
    output_value: object | None
    status: Literal["OK", "ERROR"]
    status_message: str | None = None
    children: tuple["SpanSpec", ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "span name")
        _require_non_empty_string(self.kind, "span kind")
        if isinstance(self.start_ns, bool) or not isinstance(self.start_ns, int):
            raise TypeError("span start_ns must be an integer")
        if isinstance(self.end_ns, bool) or not isinstance(self.end_ns, int):
            raise TypeError("span end_ns must be an integer")
        if self.end_ns < self.start_ns:
            raise ValueError("span end_ns must not precede start_ns")
        if not isinstance(self.status, str) or self.status not in {"OK", "ERROR"}:
            raise ValueError(f"unsupported span status: {self.status!r}")
        if self.status_message is not None and not isinstance(self.status_message, str):
            raise TypeError("span status_message must be a string or None")
        frozen_attributes = {
            key: _freeze_attribute(value) for key, value in self.attributes.items()
        }
        if any(not isinstance(key, str) for key in frozen_attributes):
            raise TypeError("trace attribute names must be strings")
        object.__setattr__(self, "attributes", MappingProxyType(frozen_attributes))
        object.__setattr__(self, "input_value", _freeze_payload(self.input_value))
        object.__setattr__(self, "output_value", _freeze_payload(self.output_value))
        object.__setattr__(self, "children", tuple(self.children))
        if any(not isinstance(child, SpanSpec) for child in self.children):
            raise TypeError("span children must be SpanSpec values")


@dataclass(frozen=True)
class TraceBundle:
    project_name: str
    session_id: str
    topic_id: str
    baseline: Literal["piika-agentic", "ragnarok-fixed"]
    root: SpanSpec

    def __post_init__(self) -> None:
        _require_non_empty_string(self.project_name, "trace project_name")
        _require_non_empty_string(self.session_id, "trace session_id")
        _require_non_empty_string(self.topic_id, "trace topic_id")
        if not isinstance(self.baseline, str) or self.baseline not in {
            "piika-agentic",
            "ragnarok-fixed",
        }:
            raise ValueError(f"unsupported trace baseline: {self.baseline!r}")
        if not isinstance(self.root, SpanSpec):
            raise TypeError("trace root must be a SpanSpec")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _span_to_dict(span: SpanSpec) -> dict[str, object]:
    return {
        "name": span.name,
        "kind": span.kind,
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "attributes": _json_value(span.attributes),
        "input_value": _json_value(span.input_value),
        "output_value": _json_value(span.output_value),
        "status": span.status,
        "status_message": span.status_message,
        "children": [_span_to_dict(child) for child in span.children],
    }


def _bundle_to_dict(bundle: TraceBundle) -> dict[str, object]:
    return {
        "project_name": bundle.project_name,
        "session_id": bundle.session_id,
        "topic_id": bundle.topic_id,
        "baseline": bundle.baseline,
        "root": _span_to_dict(bundle.root),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> object:
    raise ValueError(f"non-finite JSON value: {token}")


def _strict_json_loads(body: str) -> object:
    try:
        return json.loads(
            body,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid strict JSON") from error


def write_trace_bundle(bundle: TraceBundle, path: Path) -> Path:
    """Atomically persist a trace bundle as deterministic strict JSON."""
    if not isinstance(bundle, TraceBundle):
        raise TypeError("bundle must be a TraceBundle")
    destination = Path(path)
    body = json.dumps(
        _bundle_to_dict(bundle),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    value: dict[str, object], fields: set[str], label: str
) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} has an invalid field set")


def _span_from_dict(value: object) -> SpanSpec:
    record = _require_object(value, "span")
    _require_exact_fields(
        record,
        {
            "name",
            "kind",
            "start_ns",
            "end_ns",
            "attributes",
            "input_value",
            "output_value",
            "status",
            "status_message",
            "children",
        },
        "span",
    )
    attributes = _require_object(record["attributes"], "span attributes")
    children = record["children"]
    if not isinstance(children, list):
        raise ValueError("span children must be a JSON array")
    try:
        return SpanSpec(
            name=record["name"],
            kind=record["kind"],
            start_ns=record["start_ns"],
            end_ns=record["end_ns"],
            attributes=attributes,
            input_value=record["input_value"],
            output_value=record["output_value"],
            status=record["status"],
            status_message=record["status_message"],
            children=tuple(_span_from_dict(child) for child in children),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid span value") from error


def read_trace_bundle(
    path: Path, *, max_bytes: int = DEFAULT_BUNDLE_MAX_BYTES
) -> TraceBundle:
    """Read a strictly encoded trace bundle without accepting JSON extensions."""
    source = Path(path)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if source.stat().st_size > max_bytes:
        raise ValueError("trace bundle exceeds configured byte ceiling")
    try:
        body = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("trace bundle must be UTF-8") from error
    record = _require_object(_strict_json_loads(body), "trace bundle")
    _require_exact_fields(
        record,
        {"project_name", "session_id", "topic_id", "baseline", "root"},
        "trace bundle",
    )
    try:
        return TraceBundle(
            project_name=record["project_name"],
            session_id=record["session_id"],
            topic_id=record["topic_id"],
            baseline=record["baseline"],
            root=_span_from_dict(record["root"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid trace bundle") from error

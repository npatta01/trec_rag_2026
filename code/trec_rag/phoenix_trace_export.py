"""Credential-safe recursive export of trace bundles to Phoenix."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace.status import Status, StatusCode

from trec_rag.pi_trace_models import SpanSpec, TraceBundle


_JSON_MIME_TYPE = "application/json"
_TEXT_MIME_TYPE = "text/plain"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_WORKSPACE_PATH = re.compile(r"/s/[^/]+/?")


@dataclass(frozen=True, repr=False)
class SecretStr:
    """A string wrapper whose normal display forms never expose its value."""

    _value: str

    def __post_init__(self) -> None:
        if not isinstance(self._value, str):
            raise TypeError("secret value must be a string")
        if not self._value:
            raise ValueError("secret value must be non-empty")

    def reveal(self) -> str:
        """Return the secret only for an authenticated client boundary."""
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    __str__ = __repr__


def _normalize_endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("PHOENIX_COLLECTOR_ENDPOINT must be a string")
    endpoint = value.strip()
    if not endpoint:
        raise ValueError("PHOENIX_COLLECTOR_ENDPOINT is required")
    parsed = urlsplit(endpoint)
    if parsed.query or parsed.fragment:
        raise ValueError(
            "PHOENIX_COLLECTOR_ENDPOINT must not contain a query or fragment"
        )
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("PHOENIX_COLLECTOR_ENDPOINT must be an absolute URL")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("PHOENIX_COLLECTOR_ENDPOINT must use HTTP or HTTPS")
    if parsed.scheme != "https" and parsed.hostname.lower() not in _LOCAL_HOSTS:
        raise ValueError("PHOENIX_COLLECTOR_ENDPOINT must use HTTPS for hosted services")

    path = parsed.path.rstrip("/")
    if (
        parsed.hostname.lower() == "app.phoenix.arize.com"
        and (not path or _WORKSPACE_PATH.fullmatch(path))
    ):
        path = f"{path}/v1/traces"
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return normalized.rstrip("/")


@dataclass(frozen=True)
class PhoenixSettings:
    api_key: SecretStr
    collector_endpoint: str
    project_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, SecretStr):
            raise TypeError("api_key must be a SecretStr")
        if not isinstance(self.project_name, str):
            raise TypeError("PHOENIX_PROJECT_NAME must be a string")
        project_name = self.project_name.strip()
        if not project_name:
            raise ValueError("PHOENIX_PROJECT_NAME is required")
        object.__setattr__(self, "collector_endpoint", _normalize_endpoint(self.collector_endpoint))
        object.__setattr__(self, "project_name", project_name)

    @classmethod
    def from_env(cls) -> "PhoenixSettings":
        """Read all required Phoenix values, failing closed on absence."""
        values: dict[str, str] = {}
        for name in (
            "PHOENIX_API_KEY",
            "PHOENIX_COLLECTOR_ENDPOINT",
            "PHOENIX_PROJECT_NAME",
        ):
            value = os.environ.get(name)
            if value is None or not value.strip():
                raise ValueError(f"{name} is required")
            values[name] = value
        return cls(
            api_key=SecretStr(values["PHOENIX_API_KEY"]),
            collector_endpoint=values["PHOENIX_COLLECTOR_ENDPOINT"],
            project_name=values["PHOENIX_PROJECT_NAME"],
        )


@dataclass(frozen=True)
class ExportReceipt:
    project_name: str
    trace_id: str
    root_span_id: str
    exported_span_count: int


class _ExportMonitor:
    """Record synchronous exporter outcomes hidden by SimpleSpanProcessor."""

    def __init__(self, export: Callable[[Sequence[Any]], SpanExportResult]) -> None:
        self._export = export
        self.failed = False
        self.successful_span_count = 0

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        try:
            result = self._export(spans)
        except BaseException:
            self.failed = True
            raise
        if result is not SpanExportResult.SUCCESS:
            self.failed = True
        else:
            self.successful_span_count += len(spans)
        return result


def _install_export_monitor(provider: Any) -> _ExportMonitor | None:
    active_processor = getattr(provider, "_active_span_processor", None)
    processors = getattr(active_processor, "_span_processors", ())
    if len(processors) != 1:
        return None
    processor = processors[0]
    exporter = getattr(processor, "span_exporter", None)
    if exporter is None:
        batch_processor = getattr(processor, "_batch_processor", None)
        exporter = getattr(batch_processor, "_exporter", None)
    original_export = getattr(exporter, "export", None)
    if not callable(original_export):
        return None
    monitor = _ExportMonitor(original_export)
    exporter.export = monitor.export
    return monitor


def _forbidden_tokens(values: Iterable[object]) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in values:
        if isinstance(value, SecretStr):
            value = value.reveal()
        if isinstance(value, str) and len(value) >= 8:
            tokens.append(value)
    return tuple(tokens)


def _serialized_leaf(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, bool | int | float):
        return json.dumps(value, allow_nan=False)
    return None


def assert_no_secrets(bundle: TraceBundle, forbidden_values: Iterable[object]) -> None:
    """Reject credential values found anywhere in serialized trace content."""
    if not isinstance(bundle, TraceBundle):
        raise TypeError("bundle must be a TraceBundle")
    tokens = _forbidden_tokens(forbidden_values)
    if not tokens:
        return

    seen: set[int] = set()

    def inspect(value: object) -> None:
        leaf = _serialized_leaf(value)
        if leaf is not None:
            if any(token in leaf for token in tokens):
                raise ValueError("trace content contains a credential-like value")
            return

        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                inspect(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            for item in value:
                inspect(item)

    inspect(bundle)


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_compatible(item) for item in value]
    return value


def _payload_attributes(prefix: str, value: object | None) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        encoded = value
        mime_type = _TEXT_MIME_TYPE
    else:
        encoded = json.dumps(
            _json_compatible(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        mime_type = _JSON_MIME_TYPE
    return {
        getattr(SpanAttributes, f"{prefix}_VALUE"): encoded,
        getattr(SpanAttributes, f"{prefix}_MIME_TYPE"): mime_type,
    }


def _export_span(
    tracer: Any,
    spec: SpanSpec,
    *,
    parent_span: Any | None,
    root_attributes: Mapping[str, str] | None,
) -> tuple[Any, int]:
    attributes = dict(spec.attributes)
    attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = spec.kind
    if root_attributes is not None:
        attributes.update(root_attributes)
    context = (
        None if parent_span is None else trace.set_span_in_context(parent_span)
    )
    span = tracer.start_span(
        spec.name,
        context=context,
        start_time=spec.start_ns,
        attributes=attributes,
    )
    exported_count = 1
    try:
        for name, value in _payload_attributes("INPUT", spec.input_value).items():
            span.set_attribute(name, value)
        for name, value in _payload_attributes("OUTPUT", spec.output_value).items():
            span.set_attribute(name, value)
        if spec.status == "ERROR":
            span.set_status(Status(StatusCode.ERROR, spec.status_message))
        else:
            span.set_status(Status(StatusCode.OK))
        for child in spec.children:
            _, child_count = _export_span(
                tracer,
                child,
                parent_span=span,
                root_attributes=None,
            )
            exported_count += child_count
    finally:
        span.end(end_time=spec.end_ns)
    return span, exported_count


def export_trace(
    bundle: TraceBundle,
    settings: PhoenixSettings,
    *,
    provider_factory: Callable[..., Any] | None = None,
) -> ExportReceipt:
    """Synchronously export a trace tree and flush all provider resources."""
    if not isinstance(bundle, TraceBundle):
        raise TypeError("bundle must be a TraceBundle")
    if not isinstance(settings, PhoenixSettings):
        raise TypeError("settings must be PhoenixSettings")
    assert_no_secrets(bundle, (settings.api_key,))

    if provider_factory is None:
        from phoenix.otel import register

        provider_factory = register
    provider = provider_factory(
        project_name=settings.project_name,
        endpoint=settings.collector_endpoint,
        api_key=settings.api_key.reveal(),
        batch=False,
        verbose=False,
        set_global_tracer_provider=False,
    )
    flush_completed: bool | None = None
    monitor: _ExportMonitor | None = None
    try:
        monitor = _install_export_monitor(provider)
        if monitor is None:
            raise RuntimeError("Phoenix exporter result verification is unavailable")
        tracer = provider.get_tracer(__name__)
        root_span, exported_count = _export_span(
            tracer,
            bundle.root,
            parent_span=None,
            root_attributes={
                SpanAttributes.SESSION_ID: bundle.session_id,
                "topic.id": bundle.topic_id,
                "baseline": bundle.baseline,
            },
        )
    finally:
        try:
            flush_completed = provider.force_flush()
        finally:
            provider.shutdown()

    if (
        flush_completed is False
        or monitor.failed
        or monitor.successful_span_count != exported_count
    ):
        raise RuntimeError("Phoenix export did not complete successfully")

    context = root_span.get_span_context()
    return ExportReceipt(
        project_name=settings.project_name,
        trace_id=f"{context.trace_id:032x}",
        root_span_id=f"{context.span_id:016x}",
        exported_span_count=exported_count,
    )


__all__ = [
    "ExportReceipt",
    "PhoenixSettings",
    "SecretStr",
    "assert_no_secrets",
    "export_trace",
]

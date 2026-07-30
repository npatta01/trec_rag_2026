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
from trec_rag.openai_trace_semantics import (
    openai_llm_attributes,
    openai_request_envelope,
    openai_response_envelope,
)


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


def _json_string(value: object) -> str:
    return json.dumps(
        _json_compatible(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _messages(value: object | None, *, input_side: bool) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return []
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes):
        return [message for message in messages if isinstance(message, Mapping)]
    if isinstance(value.get("role"), str):
        return [value]
    if input_side and isinstance(value.get("system_prompt"), str) and isinstance(
        value.get("user_prompt"), str
    ):
        return [
            {"role": "system", "content": value["system_prompt"]},
            {"role": "user", "content": value["user_prompt"]},
        ]
    if input_side and isinstance(value.get("narrative"), str):
        return [{"role": "user", "content": value["narrative"]}]
    return []


def _message_attributes(
    prefix: str, messages: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for message_index, message in enumerate(messages):
        base = f"{prefix}.{message_index}.message"
        role = message.get("role")
        if isinstance(role, str):
            attributes[f"{base}.role"] = "tool" if role == "toolResult" else role
        tool_call_id = message.get("toolCallId", message.get("tool_call_id"))
        if isinstance(tool_call_id, str):
            attributes[f"{base}.tool_call_id"] = tool_call_id
        content = message.get("content")
        if isinstance(content, str):
            attributes[f"{base}.content"] = content
            continue
        if not isinstance(content, Sequence) or isinstance(content, str | bytes):
            continue
        content_index = 0
        tool_index = 0
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"} and isinstance(
                item.get("text"), str
            ):
                content_base = f"{base}.contents.{content_index}.message_content"
                attributes[f"{content_base}.type"] = "text"
                attributes[f"{content_base}.text"] = item["text"]
                content_index += 1
            elif item_type in {"thinking", "reasoning"}:
                reasoning = item.get("thinking", item.get("text"))
                if isinstance(reasoning, str):
                    content_base = f"{base}.contents.{content_index}.message_content"
                    attributes[f"{content_base}.type"] = "reasoning"
                    attributes[f"{content_base}.text"] = reasoning
                    content_index += 1
            elif item_type in {"toolCall", "tool_call"}:
                tool_base = f"{base}.tool_calls.{tool_index}.tool_call"
                tool_id = item.get("id", item.get("toolCallId"))
                tool_name = item.get("name", item.get("toolName"))
                arguments = item.get("arguments", item.get("args"))
                if isinstance(tool_id, str):
                    attributes[f"{tool_base}.id"] = tool_id
                if isinstance(tool_name, str):
                    attributes[f"{tool_base}.function.name"] = tool_name
                if arguments is not None:
                    attributes[f"{tool_base}.function.arguments"] = _json_string(
                        arguments
                    )
                tool_index += 1
    return attributes


def _llm_attributes(spec: SpanSpec) -> dict[str, object]:
    attributes: dict[str, object] = {}
    attributes.update(
        _message_attributes(
            SpanAttributes.LLM_INPUT_MESSAGES,
            _messages(spec.input_value, input_side=True),
        )
    )
    output_messages = _messages(spec.output_value, input_side=False)
    attributes.update(
        _message_attributes(SpanAttributes.LLM_OUTPUT_MESSAGES, output_messages)
    )
    if isinstance(spec.output_value, Mapping):
        output = spec.output_value
        for source, target in (
            ("model", SpanAttributes.LLM_MODEL_NAME),
            ("provider", SpanAttributes.LLM_PROVIDER),
            ("stopReason", SpanAttributes.LLM_FINISH_REASON),
        ):
            value = output.get(source)
            if isinstance(value, str):
                attributes[target] = value
        usage = output.get("usage")
        if isinstance(usage, Mapping):
            prompt = usage.get(
                "inputTokens", usage.get("prompt_tokens", usage.get("input"))
            )
            completion = usage.get(
                "outputTokens", usage.get("completion_tokens", usage.get("output"))
            )
            total = usage.get("totalTokens", usage.get("total_tokens"))
            if isinstance(prompt, int) and not isinstance(prompt, bool):
                attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt
            if isinstance(completion, int) and not isinstance(completion, bool):
                attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion
            if isinstance(total, int) and not isinstance(total, bool):
                attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = total
            elif isinstance(prompt, int) and isinstance(completion, int):
                attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = prompt + completion
            for source, target in (
                ("cacheRead", SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ),
                ("cacheWrite", SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE),
            ):
                value = usage.get(source)
                if isinstance(value, int) and not isinstance(value, bool):
                    attributes[target] = value
            costs = usage.get("cost")
            if isinstance(costs, Mapping):
                for source, target in (
                    ("input", SpanAttributes.LLM_COST_PROMPT),
                    ("output", SpanAttributes.LLM_COST_COMPLETION),
                    ("total", SpanAttributes.LLM_COST_TOTAL),
                    ("cacheRead", SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_READ),
                    ("cacheWrite", SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_WRITE),
                ):
                    value = costs.get(source)
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        attributes[target] = value
        invocation = output.get(
            "invocationParameters", output.get("invocation_parameters")
        )
        if isinstance(invocation, Mapping):
            attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = _json_string(
                invocation
            )
    return attributes


def _retrieval_attributes(output_value: object | None) -> dict[str, object]:
    if not isinstance(output_value, Mapping):
        return {}
    documents = output_value.get("documents")
    if not isinstance(documents, Sequence) or isinstance(documents, str | bytes):
        return {}
    attributes: dict[str, object] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            continue
        base = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.document"
        doc_id = document.get("docid", document.get("id"))
        content = document.get("text", document.get("content"))
        score = document.get("score")
        if isinstance(doc_id, str):
            attributes[f"{base}.id"] = doc_id
        if isinstance(content, str):
            attributes[f"{base}.content"] = content
        if isinstance(score, int | float) and not isinstance(score, bool):
            attributes[f"{base}.score"] = score
    return attributes


def _semantic_attributes(spec: SpanSpec) -> dict[str, object]:
    attributes: dict[str, object] = {}
    if spec.kind == "LLM":
        attributes.update(openai_llm_attributes(spec))
    if spec.kind == "RETRIEVER":
        attributes.update(_retrieval_attributes(spec.output_value))
    tool_name = spec.attributes.get("pi.tool.name")
    if isinstance(tool_name, str) and tool_name:
        attributes[SpanAttributes.TOOL_NAME] = tool_name
        tool_id = spec.attributes.get("pi.tool_call.id")
        if isinstance(tool_id, str) and tool_id:
            attributes[SpanAttributes.TOOL_ID] = tool_id
        if spec.input_value is not None:
            attributes[SpanAttributes.TOOL_PARAMETERS] = _json_string(spec.input_value)
    return attributes


def _export_span(
    tracer: Any,
    spec: SpanSpec,
    *,
    parent_span: Any | None,
    root_attributes: Mapping[str, str] | None,
) -> tuple[Any, int]:
    attributes = dict(spec.attributes)
    attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = spec.kind
    attributes.update(_semantic_attributes(spec))
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
        input_value = (
            openai_request_envelope(spec) if spec.kind == "LLM" else spec.input_value
        )
        output_value = (
            openai_response_envelope(spec) if spec.kind == "LLM" else spec.output_value
        )
        if spec.kind == "LLM":
            if spec.input_value is not None:
                span.set_attribute("pi.native.input_json", _json_string(spec.input_value))
            if spec.output_value is not None:
                span.set_attribute("pi.native.output_json", _json_string(spec.output_value))
        for name, value in _payload_attributes("INPUT", input_value).items():
            span.set_attribute(name, value)
        for name, value in _payload_attributes("OUTPUT", output_value).items():
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

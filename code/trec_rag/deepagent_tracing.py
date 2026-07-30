"""Optional Phoenix/OpenInference tracing for the Deep Agent retriever."""

from __future__ import annotations

from contextlib import contextmanager
import os
from threading import Lock
from typing import Iterator, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from phoenix.otel import register as phoenix_register


DEFAULT_PHOENIX_PROJECT = "trec-rag-deepagent-retrieval"
REDACTED_CONTENT = "[REDACTED]"
_CLOUD_HOST = "app.phoenix.arize.com"
_INSTRUMENTED_PROVIDERS: dict[int, object] = {}
_INSTRUMENTATION_LOCK = Lock()
_INVALID_ENDPOINT_MESSAGE = (
    "PHOENIX_COLLECTOR_ENDPOINT must be an absolute HTTP(S) URL without "
    "credentials, query, or fragment"
)


class _TracerProvider(Protocol):
    def get_tracer(self, instrumenting_module_name: str, *args: object, **kwargs: object) -> object: ...

    def force_flush(self, *args: object, **kwargs: object) -> bool: ...


class _MaskingSpan:
    """Limited facade for non-sensitive retrieval metadata."""

    _SAFE_ATTRIBUTE_KEYS = frozenset({"retrieval.document_count"})

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def set_attribute(self, key: str, value: object) -> None:
        if key not in self._SAFE_ATTRIBUTE_KEYS:
            raise ValueError(f"attribute {key!r} is not safe for retrieval tracing")
        self._wrapped.set_attribute(key, value)  # type: ignore[attr-defined]


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Return an explicit OTLP/HTTP trace endpoint without disclosing bad input."""
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError:
        raise ValueError(_INVALID_ENDPOINT_MESSAGE) from None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(_INVALID_ENDPOINT_MESSAGE)

    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        path = f"{path}/v1/traces"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _settings(environ: Mapping[str, str]) -> tuple[str | None, str, bool]:
    configured_endpoint = environ.get("PHOENIX_COLLECTOR_ENDPOINT") or None
    endpoint = (
        _normalize_otlp_http_endpoint(configured_endpoint)
        if configured_endpoint is not None
        else None
    )
    project = environ.get("PHOENIX_PROJECT_NAME") or DEFAULT_PHOENIX_PROJECT
    if (
        endpoint
        and urlsplit(endpoint).hostname == _CLOUD_HOST
        and not environ.get("PHOENIX_API_KEY")
    ):
        raise ValueError("PHOENIX_API_KEY is required for Phoenix Cloud")
    return endpoint, project, endpoint is not None


class RetrievalTracing:
    """Manual retrieval spans backed by an optional OpenTelemetry provider."""

    def __init__(
        self,
        *,
        enabled: bool,
        tracer: OITracer,
        provider: _TracerProvider | None,
        trace_content: bool,
    ) -> None:
        self.enabled = enabled
        self._tracer = tracer
        self._provider = provider
        self._trace_content = trace_content

    def _content(self, value: str) -> str:
        return value if self._trace_content else REDACTED_CONTENT

    @contextmanager
    def _span(
        self,
        name: str,
        content: str,
        kind: OpenInferenceSpanKindValues,
    ) -> Iterator[_MaskingSpan]:
        with self._tracer.start_as_current_span(
            name,
            attributes={SpanAttributes.INPUT_VALUE: self._content(content)},
            openinference_span_kind=kind,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield _MaskingSpan(span)
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", type(exc).__name__)
                raise

    @contextmanager
    def agent_span(self, narrative: str) -> Iterator[_MaskingSpan]:
        with self._span(
            "deepagent.retrieve", narrative, OpenInferenceSpanKindValues.AGENT
        ) as span:
            yield span

    @contextmanager
    def retriever_span(self, query: str) -> Iterator[_MaskingSpan]:
        with self._span(
            "climbmix.retrieve", query, OpenInferenceSpanKindValues.RETRIEVER
        ) as span:
            yield span

    def force_flush(self) -> bool:
        """Flush a configured exporter without raising from an optional sink."""
        if self._provider is None:
            return True
        try:
            return bool(self._provider.force_flush())
        except Exception:
            return False


def _instrument_langchain(provider: _TracerProvider, trace_content: bool) -> None:
    """Instrument a provider once, avoiding duplicate callbacks in notebooks."""
    with _INSTRUMENTATION_LOCK:
        provider_id = id(provider)
        if provider_id in _INSTRUMENTED_PROVIDERS:
            return
        LangChainInstrumentor().instrument(
            tracer_provider=provider,
            config=TraceConfig(
                hide_input_text=not trace_content,
                hide_output_text=not trace_content,
            ),
        )
        _INSTRUMENTED_PROVIDERS[provider_id] = provider


def create_retrieval_tracing(
    *,
    environ: Mapping[str, str] | None = None,
    tracer_provider: _TracerProvider | None = None,
    trace_content: bool = True,
) -> RetrievalTracing:
    """Create optional tracing without registering an exporter for injected providers."""
    resolved_environ = os.environ if environ is None else environ
    endpoint, project_name, enabled_by_endpoint = _settings(resolved_environ)

    provider = tracer_provider
    if provider is None and enabled_by_endpoint:
        provider = phoenix_register(
            endpoint=endpoint,
            project_name=project_name,
            protocol="http/protobuf",
            batch=True,
            api_key=resolved_environ.get("PHOENIX_API_KEY"),
            verbose=False,
        )

    if provider is None:
        noop = trace.get_tracer(__name__)
        return RetrievalTracing(
            enabled=False,
            tracer=OITracer(noop, TraceConfig()),
            provider=None,
            trace_content=trace_content,
        )

    _instrument_langchain(provider, trace_content)
    return RetrievalTracing(
        enabled=True,
        tracer=OITracer(provider.get_tracer(__name__), TraceConfig()),
        provider=provider,
        trace_content=trace_content,
    )

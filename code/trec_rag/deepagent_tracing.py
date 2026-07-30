"""Optional Phoenix/OpenInference tracing for the Deep Agent retriever."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from hashlib import sha256
from math import isfinite
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
MAX_TRACE_DOCUMENTS = 1_000
MAX_TRACE_DOCUMENT_ID_CHARACTERS = 512
MAX_TRACE_EXCERPT_CHARACTERS = 1_000
_CLOUD_HOST = "app.phoenix.arize.com"
_LIVE_PROVIDER_CACHE: dict[tuple[str, str, str | None], _TracerProvider] = {}
_ACTIVE_INSTRUMENTATION: tuple[object, ...] | None = None
_CONFIGURATION_LOCK = Lock()
_CONFLICTING_CONFIGURATION_MESSAGE = (
    "retrieval tracing is already configured differently in this process"
)
_INVALID_ENDPOINT_MESSAGE = (
    "PHOENIX_COLLECTOR_ENDPOINT must be an absolute HTTP(S) URL without "
    "credentials, query, or fragment"
)


class _TracerProvider(Protocol):
    def get_tracer(self, instrumenting_module_name: str, *args: object, **kwargs: object) -> object: ...

    def force_flush(self, *args: object, **kwargs: object) -> bool: ...


class _SafeSpan:
    """Base for purpose-specific span facades with no generic mutation API."""

    def __init__(self, wrapped: object, *, trace_content: bool) -> None:
        self._wrapped = wrapped
        self._trace_content = trace_content

    def _set(self, key: str, value: object) -> None:
        self._wrapped.set_attribute(key, value)  # type: ignore[attr-defined]

    def _content(self, value: str) -> str:
        return value if self._trace_content else REDACTED_CONTENT


def _validated_strings(
    name: str, values: Sequence[str], *, maximum_items: int, maximum_characters: int
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if len(values) > maximum_items:
        raise ValueError(f"{name} exceeds the safe tracing item limit")
    normalized = tuple(values)
    if any(not isinstance(value, str) for value in normalized):
        raise TypeError(f"{name} must contain only strings")
    if name in {"document_ids", "fused_document_ids"} and any(
        not value for value in normalized
    ):
        raise ValueError(f"{name} must not contain blank values")
    if any(len(value) > maximum_characters for value in normalized):
        raise ValueError(f"{name} contains an oversized value")
    return normalized


class _RetrieverSpan(_SafeSpan):
    """Allow only the bounded, typed evidence for one retrieval operation."""

    _CACHE_STATUSES = frozenset({"hit", "miss", "bypass", "not_reported"})

    def record_search(
        self,
        *,
        document_ids: Sequence[str],
        source_ranks: Sequence[int],
        source_scores: Sequence[float],
        cache_status: str,
        latency_ms: float,
        excerpts: Sequence[str],
    ) -> None:
        ids = _validated_strings(
            "document_ids",
            document_ids,
            maximum_items=MAX_TRACE_DOCUMENTS,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )
        bounded_excerpts = _validated_strings(
            "excerpts",
            excerpts,
            maximum_items=MAX_TRACE_DOCUMENTS,
            maximum_characters=MAX_TRACE_EXCERPT_CHARACTERS,
        )
        ranks = tuple(source_ranks)
        scores = tuple(source_scores)
        if not (len(ids) == len(ranks) == len(scores) == len(bounded_excerpts)):
            raise ValueError("retrieval trace evidence arrays must have equal lengths")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            for rank in ranks
        ):
            raise TypeError("source_ranks must contain only positive integers")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(float(score))
            for score in scores
        ):
            raise TypeError("source_scores must contain only finite numbers")
        if cache_status not in self._CACHE_STATUSES:
            raise ValueError("cache_status is not safe for retrieval tracing")
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not isfinite(float(latency_ms))
            or latency_ms < 0
        ):
            raise TypeError("latency_ms must be a finite non-negative number")

        self._set("retrieval.document_count", len(ids))
        self._set("retrieval.document_ids", ids)
        self._set("retrieval.source_ranks", ranks)
        self._set("retrieval.source_scores", tuple(float(score) for score in scores))
        self._set("retrieval.cache_status", cache_status)
        self._set("retrieval.latency_ms", float(latency_ms))
        self._set(
            "retrieval.document_excerpts",
            tuple(self._content(excerpt) for excerpt in bounded_excerpts),
        )


class _AgentSpan(_SafeSpan):
    """Allow only bounded final retrieval evidence on the root span."""

    _STOPPING_REASONS = frozenset({"agent_completed", "search_budget_exhausted"})

    def record_result(
        self, *, fused_document_ids: Sequence[str], stopping_reason: str
    ) -> None:
        ids = _validated_strings(
            "fused_document_ids",
            fused_document_ids,
            maximum_items=MAX_TRACE_DOCUMENTS,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )
        if stopping_reason not in self._STOPPING_REASONS:
            raise ValueError("stopping_reason is not safe for retrieval tracing")
        self._set("retrieval.fused_document_ids", ids)
        self._set("retrieval.stopping_reason", stopping_reason)


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
        or any(
            character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
            for character in endpoint
        )
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
    api_key = environ.get("PHOENIX_API_KEY")
    cloud_hostname = (
        (urlsplit(endpoint).hostname or "").lower().rstrip(".") if endpoint else ""
    )
    if (
        endpoint
        and cloud_hostname == _CLOUD_HOST
        and (api_key is None or not api_key.strip())
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
    ) -> Iterator[object]:
        with self._tracer.start_as_current_span(
            name,
            attributes={SpanAttributes.INPUT_VALUE: self._content(content)},
            openinference_span_kind=kind,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", type(exc).__name__)
                raise

    @contextmanager
    def agent_span(self, narrative: str) -> Iterator[_AgentSpan]:
        with self._span(
            "deepagent.retrieve", narrative, OpenInferenceSpanKindValues.AGENT
        ) as span:
            yield _AgentSpan(span, trace_content=self._trace_content)

    @contextmanager
    def retriever_span(self, query: str) -> Iterator[_RetrieverSpan]:
        with self._span(
            "climbmix.retrieve", query, OpenInferenceSpanKindValues.RETRIEVER
        ) as span:
            yield _RetrieverSpan(span, trace_content=self._trace_content)

    def force_flush(self) -> bool:
        """Flush a configured exporter without raising from an optional sink."""
        if self._provider is None:
            return True
        try:
            return bool(self._provider.force_flush())
        except Exception:
            return False


def _credential_fingerprint(api_key: str | None) -> str | None:
    """Return a non-reversible cache-key component without retaining a credential."""
    return sha256(api_key.encode()).hexdigest() if api_key is not None else None


def _instrument_langchain(provider: _TracerProvider, trace_content: bool) -> None:
    """Install the one process-global LangChain instrumentor for a resolved provider."""
    LangChainInstrumentor().instrument(
        tracer_provider=provider,
        config=TraceConfig(
            hide_input_text=not trace_content,
            hide_output_text=not trace_content,
        ),
    )


def create_retrieval_tracing(
    *,
    environ: Mapping[str, str] | None = None,
    tracer_provider: _TracerProvider | None = None,
    trace_content: bool = True,
) -> RetrievalTracing:
    """Create optional tracing without registering an exporter for injected providers."""
    global _ACTIVE_INSTRUMENTATION

    resolved_environ = os.environ if environ is None else environ
    endpoint, project_name, enabled_by_endpoint = _settings(resolved_environ)
    api_key = resolved_environ.get("PHOENIX_API_KEY")

    with _CONFIGURATION_LOCK:
        provider = tracer_provider
        if provider is not None:
            requested_configuration: tuple[object, ...] = (
                "injected",
                id(provider),
                trace_content,
            )
        elif enabled_by_endpoint:
            assert endpoint is not None
            live_key = (
                endpoint,
                project_name,
                _credential_fingerprint(api_key),
            )
            requested_configuration = ("live", *live_key, trace_content)
        else:
            if _ACTIVE_INSTRUMENTATION is not None:
                raise ValueError(_CONFLICTING_CONFIGURATION_MESSAGE)
            noop = trace.get_tracer(__name__)
            return RetrievalTracing(
                enabled=False,
                tracer=OITracer(noop, TraceConfig()),
                provider=None,
                trace_content=trace_content,
            )

        if (
            _ACTIVE_INSTRUMENTATION is not None
            and _ACTIVE_INSTRUMENTATION != requested_configuration
        ):
            raise ValueError(_CONFLICTING_CONFIGURATION_MESSAGE)

        if provider is None:
            provider = _LIVE_PROVIDER_CACHE.get(live_key)
            if provider is None:
                provider = phoenix_register(
                    endpoint=endpoint,
                    project_name=project_name,
                    protocol="http/protobuf",
                    batch=True,
                    api_key=api_key,
                    verbose=False,
                )
                _LIVE_PROVIDER_CACHE[live_key] = provider

        if _ACTIVE_INSTRUMENTATION is None:
            _instrument_langchain(provider, trace_content)
            _ACTIVE_INSTRUMENTATION = requested_configuration

    return RetrievalTracing(
        enabled=True,
        tracer=OITracer(provider.get_tracer(__name__), TraceConfig()),
        provider=provider,
        trace_content=trace_content,
    )

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
from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from phoenix.otel import register as phoenix_register


DEFAULT_PHOENIX_PROJECT = "trec-rag-deepagent-retrieval"
REDACTED_CONTENT = "[REDACTED]"
MAX_TRACE_DOCUMENTS = 1_000
MAX_TRACE_DOCUMENT_ID_CHARACTERS = 512
MAX_TRACE_SNIPPETS = 10
MAX_TRACE_SNIPPET_TEXT_CHARACTERS = 3_500
MAX_TRACE_RANKER_BACKEND_CHARACTERS = 512
MAX_TRACE_RESEARCH_TASK_ID_CHARACTERS = 128
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
    if name in {"document_ids", "fused_document_ids", "chunk_ids"} and any(
        not value for value in normalized
    ):
        raise ValueError(f"{name} must not contain blank values")
    if any(len(value) > maximum_characters for value in normalized):
        raise ValueError(f"{name} contains an oversized value")
    return normalized


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _optional_finite_score(name: str, value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise TypeError(f"{name} must be a finite number or None")
    return float(value)


_RESEARCH_DEPTHS = frozenset({"survey", "focused", "deep"})
_BUDGET_CODES = frozenset(
    {
        "OK",
        "SOFT_DEADLINE_REACHED",
        "HARD_DEADLINE_REACHED",
        "TASK_BUDGET_EXHAUSTED",
        "ROUND_BUDGET_EXHAUSTED",
        "CONCURRENCY_BUDGET_EXHAUSTED",
        "RETRIEVAL_BUDGET_EXHAUSTED",
        "TASK_TOOL_BUDGET_EXHAUSTED",
        "ROUND_RESEARCH_REQUIRED",
        "ROUND_SEQUENCE_INVALID",
        "NO_YIELD_STOP",
        "NO_PROGRESS_STOP",
    }
)
_RESEARCH_SNAPSHOT_COUNT_FIELDS = (
    "remaining_researchers",
    "remaining_rounds",
    "remaining_retrieval_calls",
)


def _research_identity_attributes(
    *,
    research_task_id: object,
    round_index: object,
    depth: object,
) -> dict[str, object]:
    if (
        not isinstance(research_task_id, str)
        or not research_task_id
        or len(research_task_id) > MAX_TRACE_RESEARCH_TASK_ID_CHARACTERS
        or any(not (character.isalnum() or character in "._:-") for character in research_task_id)
    ):
        raise ValueError("research_task_id is not safe for retrieval tracing")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
        raise TypeError("round_index must be a positive integer")
    if depth not in _RESEARCH_DEPTHS:
        raise ValueError("research depth is not safe for retrieval tracing")
    return {
        "deepagent.research_task_id": research_task_id,
        "deepagent.research_round_index": round_index,
        "deepagent.research_depth": depth,
    }


def _research_context_attributes(
    *,
    research_task_id: object,
    round_index: object,
    depth: object,
    code: object,
    must_stop: object,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    identity = _research_identity_attributes(
        research_task_id=research_task_id,
        round_index=round_index,
        depth=depth,
    )
    if code not in _BUDGET_CODES:
        raise ValueError("budget code is not safe for retrieval tracing")
    if not isinstance(must_stop, bool):
        raise TypeError("budget must_stop must be a boolean")
    counts = {
        name: _non_negative_integer(name, snapshot.get(name))
        for name in _RESEARCH_SNAPSHOT_COUNT_FIELDS
    }
    stop_code = snapshot.get("stop_code")
    if stop_code is not None and stop_code not in _BUDGET_CODES:
        raise ValueError("budget stop_code is not safe for retrieval tracing")
    return {
        **identity,
        "deepagent.budget_code": code,
        "deepagent.budget_must_stop": must_stop,
        **{f"deepagent.{name}": value for name, value in counts.items()},
        **(
            {"deepagent.budget_stop_code": stop_code}
            if stop_code is not None
            else {}
        ),
    }


class _RetrieverSpan(_SafeSpan):
    """Allow only the bounded, typed evidence for one retrieval operation."""

    _CACHE_STATUSES = frozenset({"hit", "miss", "bypass", "not_reported"})

    def record_research_context(self, **kwargs: object) -> None:
        for key, value in _research_context_attributes(**kwargs).items():
            self._set(key, value)

    def record_search(
        self,
        *,
        document_ids: Sequence[str],
        source_ranks: Sequence[int],
        source_scores: Sequence[float],
        cache_status: str,
        latency_ms: float,
        text_lengths: Sequence[int],
    ) -> None:
        ids = _validated_strings(
            "document_ids",
            document_ids,
            maximum_items=MAX_TRACE_DOCUMENTS,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )
        ranks = tuple(source_ranks)
        scores = tuple(source_scores)
        lengths = tuple(text_lengths)
        if not (len(ids) == len(ranks) == len(scores) == len(lengths)):
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
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length < 0
            for length in lengths
        ):
            raise TypeError("text_lengths must contain only non-negative integers")
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
        self._set("retrieval.document_text_lengths", lengths)


class _SnippetSpan(_SafeSpan):
    """Allow only bounded, typed evidence for one snippet extraction page."""

    _CACHE_STATUSES = frozenset({"hit", "miss"})

    def __init__(
        self, wrapped: object, *, trace_content: bool, document_id: str
    ) -> None:
        super().__init__(wrapped, trace_content=trace_content)
        self._document_id = document_id

    def record_research_context(self, **kwargs: object) -> None:
        for key, value in _research_context_attributes(**kwargs).items():
            self._set(key, value)

    def record_page(
        self,
        *,
        chunk_ids: Sequence[str],
        start_chars: Sequence[int],
        end_chars: Sequence[int],
        relevance_scores: Sequence[float],
        texts: Sequence[str],
        cache_status: str,
        ranker_backend: str,
        latency_ms: float,
        page_offset: int,
        has_next_page: bool,
        page_index: int,
        residual_count: int,
        residual_top_score: float | None,
        returned_min_score: float | None,
        pages_estimated: int,
    ) -> None:
        document_id = _validated_strings(
            "document_ids",
            (self._document_id,),
            maximum_items=1,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )[0]
        ids = _validated_strings(
            "chunk_ids",
            chunk_ids,
            maximum_items=MAX_TRACE_SNIPPETS,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )
        bounded_texts = _validated_strings(
            "texts",
            texts,
            maximum_items=MAX_TRACE_SNIPPETS,
            maximum_characters=MAX_TRACE_SNIPPET_TEXT_CHARACTERS,
        )
        starts = tuple(start_chars)
        ends = tuple(end_chars)
        scores = tuple(relevance_scores)
        if not (
            len(ids) == len(starts) == len(ends) == len(scores) == len(bounded_texts)
        ):
            raise ValueError("snippet trace evidence arrays must have equal lengths")
        if any(
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            for offset in (*starts, *ends)
        ):
            raise TypeError("snippet offsets must contain only non-negative integers")
        if any(end < start for start, end in zip(starts, ends, strict=True)):
            raise ValueError("snippet end offsets must not precede start offsets")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(float(score))
            for score in scores
        ):
            raise TypeError("relevance_scores must contain only finite numbers")
        if cache_status not in self._CACHE_STATUSES:
            raise ValueError("cache_status is not safe for snippet tracing")
        if (
            not isinstance(ranker_backend, str)
            or not ranker_backend.strip()
            or len(ranker_backend) > MAX_TRACE_RANKER_BACKEND_CHARACTERS
        ):
            raise ValueError("ranker_backend is not safe for snippet tracing")
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not isfinite(float(latency_ms))
            or latency_ms < 0
        ):
            raise TypeError("latency_ms must be a finite non-negative number")
        if (
            isinstance(page_offset, bool)
            or not isinstance(page_offset, int)
            or page_offset < 0
        ):
            raise TypeError("page_offset must be a non-negative integer")
        if not isinstance(has_next_page, bool):
            raise TypeError("has_next_page must be a boolean")
        validated_page_index = _non_negative_integer("page_index", page_index)
        validated_residual_count = _non_negative_integer(
            "residual_count", residual_count
        )
        validated_pages_estimated = _non_negative_integer(
            "pages_estimated", pages_estimated
        )
        validated_residual_top_score = _optional_finite_score(
            "residual_top_score", residual_top_score
        )
        validated_returned_min_score = _optional_finite_score(
            "returned_min_score", returned_min_score
        )
        if (validated_residual_count == 0) != (
            validated_residual_top_score is None
        ):
            raise ValueError("residual_count and residual_top_score must agree")

        self._set("snippet.document_id", document_id)
        self._set("snippet.chunk_ids", ids)
        self._set("snippet.start_chars", starts)
        self._set("snippet.end_chars", ends)
        self._set("snippet.relevance_scores", tuple(float(score) for score in scores))
        self._set("snippet.texts", tuple(self._content(text) for text in bounded_texts))
        self._set("snippet.cache_status", cache_status)
        self._set("snippet.ranker_backend", ranker_backend)
        self._set("snippet.latency_ms", float(latency_ms))
        self._set("snippet.page_offset", page_offset)
        self._set("snippet.has_next_page", has_next_page)
        self._set("snippet.page_index", validated_page_index)
        self._set("snippet.residual_count", validated_residual_count)
        if validated_residual_top_score is not None:
            self._set("snippet.residual_top_score", validated_residual_top_score)
        if validated_returned_min_score is not None:
            self._set("snippet.returned_min_score", validated_returned_min_score)
        self._set("snippet.pages_estimated", validated_pages_estimated)


class _AgentSpan(_SafeSpan):
    """Allow only bounded final retrieval evidence on the root span."""

    _STOPPING_REASONS = frozenset(
        {
            "agent_completed",
            "search_budget_exhausted",
            "budget_exhausted",
            "completion",
            "saturation",
            "coverage_complete",
            "evidence_saturated",
        }
    ).union(_BUDGET_CODES)
    _BUDGET_STOP_CODES = frozenset(
        {
            "HARD_DEADLINE_REACHED",
            "TASK_BUDGET_EXHAUSTED",
            "ROUND_BUDGET_EXHAUSTED",
            "RETRIEVAL_BUDGET_EXHAUSTED",
            "NO_PROGRESS_STOP",
        }
    )

    def record_result(
        self,
        *,
        fused_document_ids: Sequence[str],
        stopping_reason: str,
        coverage_state_hash: str,
        need_count: int,
        answerable_need_count: int,
        conflicted_need_count: int,
        unresolved_need_count: int,
        nugget_count: int,
        action_count: int,
        researcher_invocation_count: int,
        research_round_count: int,
        retrieval_call_count: int,
        budget_stop_code: str | None,
    ) -> None:
        ids = _validated_strings(
            "fused_document_ids",
            fused_document_ids,
            maximum_items=MAX_TRACE_DOCUMENTS,
            maximum_characters=MAX_TRACE_DOCUMENT_ID_CHARACTERS,
        )
        if stopping_reason not in self._STOPPING_REASONS:
            raise ValueError("stopping_reason is not safe for retrieval tracing")
        if not (
            isinstance(coverage_state_hash, str)
            and len(coverage_state_hash) == 64
            and all(
                "0" <= character <= "9" or "a" <= character <= "f"
                for character in coverage_state_hash
            )
        ):
            raise ValueError(
                "coverage_state_hash must be 64 lowercase hexadecimal characters"
            )
        counts = {
            "need_count": _non_negative_integer("need_count", need_count),
            "answerable_need_count": _non_negative_integer(
                "answerable_need_count", answerable_need_count
            ),
            "conflicted_need_count": _non_negative_integer(
                "conflicted_need_count", conflicted_need_count
            ),
            "unresolved_need_count": _non_negative_integer(
                "unresolved_need_count", unresolved_need_count
            ),
            "nugget_count": _non_negative_integer("nugget_count", nugget_count),
            "action_count": _non_negative_integer("action_count", action_count),
            "researcher_invocation_count": _non_negative_integer(
                "researcher_invocation_count", researcher_invocation_count
            ),
            "research_round_count": _non_negative_integer(
                "research_round_count", research_round_count
            ),
            "retrieval_call_count": _non_negative_integer(
                "retrieval_call_count", retrieval_call_count
            ),
        }
        if budget_stop_code is not None and budget_stop_code not in self._BUDGET_STOP_CODES:
            raise ValueError("budget_stop_code is not safe for retrieval tracing")
        self._set("retrieval.fused_document_ids", ids)
        self._set("retrieval.stopping_reason", stopping_reason)
        self._set("coverage.state_hash", coverage_state_hash)
        for name, count in counts.items():
            prefix = "deepagent" if name.startswith(("researcher_", "research_", "retrieval_")) else "coverage"
            self._set(f"{prefix}.{name}", count)
        if budget_stop_code is not None:
            self._set("deepagent.budget_stop_code", budget_stop_code)


class _ResearchTaskSpan(_SafeSpan):
    """Allow compact task admission/outcome evidence without task content."""

    def __init__(
        self,
        wrapped: object,
        *,
        trace_content: bool,
        research_task_id: str,
        round_index: int,
        depth: str,
    ) -> None:
        super().__init__(wrapped, trace_content=trace_content)
        self._research_task_id = research_task_id
        self._round_index = round_index
        self._depth = depth
        self._set("deepagent.role", "researcher")
        self._set("deepagent.phase", "execution")

    def record_budget_outcome(
        self, *, code: str, must_stop: bool, snapshot: Mapping[str, object]
    ) -> None:
        for key, value in _research_context_attributes(
            research_task_id=self._research_task_id,
            round_index=self._round_index,
            depth=self._depth,
            code=code,
            must_stop=must_stop,
            snapshot=snapshot,
        ).items():
            self._set(key, value)


class _ResearchDispatchSpan(_SafeSpan):
    """Expose a safe, visible outcome for every attempted researcher dispatch."""

    _OUTCOMES = frozenset({"completed", "failed", "refused", "rejected"})
    _CODES = _BUDGET_CODES.union(
        {"INVALID_RESEARCH_TASK", "RESEARCH_TASK_FAILED", "RESEARCHER_TYPE_DENIED"}
    )

    def __init__(self, wrapped: object, *, trace_content: bool) -> None:
        super().__init__(wrapped, trace_content=trace_content)
        self._set("deepagent.role", "researcher")
        self._set("deepagent.phase", "dispatch")

    def record_outcome(
        self,
        *,
        code: str,
        outcome: str,
        research_task_id: str | None,
        round_index: int | None,
        depth: str | None,
    ) -> None:
        if outcome not in self._OUTCOMES:
            raise ValueError("research dispatch outcome is not safe for tracing")
        if code not in self._CODES:
            raise ValueError("research dispatch code is not safe for tracing")
        identity = (research_task_id, round_index, depth)
        if any(value is None for value in identity):
            if any(value is not None for value in identity):
                raise ValueError("research dispatch identity must be complete or absent")
        else:
            attributes = _research_identity_attributes(
                research_task_id=research_task_id,
                round_index=round_index,
                depth=depth,
            )
            for key in (
                "deepagent.research_task_id",
                "deepagent.research_round_index",
                "deepagent.research_depth",
            ):
                self._set(key, attributes[key])
        self._set("deepagent.dispatch_outcome", outcome)
        self._set("deepagent.dispatch_code", code)
        if outcome != "completed":
            self._wrapped.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]


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

    @contextmanager
    def snippet_span(
        self, document_id: str, focus_query: str
    ) -> Iterator[_SnippetSpan]:
        with self._span(
            "deepagent.extract_relevant_snippets",
            focus_query,
            OpenInferenceSpanKindValues.RETRIEVER,
        ) as span:
            yield _SnippetSpan(
                span, trace_content=self._trace_content, document_id=document_id
            )

    @contextmanager
    def researcher_task_span(
        self, research_task_id: str, round_index: int, depth: str
    ) -> Iterator[_ResearchTaskSpan]:
        identity = _research_identity_attributes(
            research_task_id=research_task_id,
            round_index=round_index,
            depth=depth,
        )
        with using_attributes(
            metadata={"deepagent.role": "researcher", **identity},
            tags=[
                "deepagent:researcher",
                f"research-task:{research_task_id}",
                f"research-round:{round_index}",
                f"research-depth:{depth}",
            ],
        ):
            with self._span(
                f"deepagent.researcher.{research_task_id}",
                "",
                OpenInferenceSpanKindValues.AGENT,
            ) as span:
                yield _ResearchTaskSpan(
                    span,
                    trace_content=self._trace_content,
                    research_task_id=research_task_id,
                    round_index=round_index,
                    depth=depth,
                )

    @contextmanager
    def researcher_dispatch_span(self) -> Iterator[_ResearchDispatchSpan]:
        with self._span(
            "deepagent.researcher.dispatch", "", OpenInferenceSpanKindValues.TOOL
        ) as span:
            yield _ResearchDispatchSpan(span, trace_content=self._trace_content)

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
            # Content mode is intended for interactive Phoenix debugging. The
            # caller can still select metadata-only tracing explicitly.
            hide_llm_invocation_parameters=not trace_content,
            hide_inputs=not trace_content,
            hide_outputs=not trace_content,
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

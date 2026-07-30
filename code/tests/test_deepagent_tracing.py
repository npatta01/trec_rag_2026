from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode
from openinference.instrumentation.langchain import LangChainInstrumentor

import trec_rag.deepagent_tracing as deepagent_tracing
from trec_rag.deepagent_tracing import (
    REDACTED_CONTENT,
    create_retrieval_tracing,
)


def _clear_process_tracing_state() -> None:
    instrumentor = LangChainInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumented_providers = getattr(
        deepagent_tracing, "_INSTRUMENTED_PROVIDERS", None
    )
    if instrumented_providers is not None:
        instrumented_providers.clear()
    live_cache = getattr(deepagent_tracing, "_LIVE_PROVIDER_CACHE", None)
    if live_cache is not None:
        live_cache.clear()
    if hasattr(deepagent_tracing, "_ACTIVE_INSTRUMENTATION"):
        deepagent_tracing._ACTIVE_INSTRUMENTATION = None


@pytest.fixture(autouse=True)
def isolated_process_tracing_state() -> None:
    _clear_process_tracing_state()
    yield
    _clear_process_tracing_state()


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


def test_no_phoenix_endpoint_returns_disabled_tracing() -> None:
    tracing = create_retrieval_tracing(environ={})

    assert tracing.enabled is False


def test_phoenix_cloud_requires_key() -> None:
    with pytest.raises(ValueError, match="PHOENIX_API_KEY"):
        create_retrieval_tracing(
            environ={
                "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/example",
            }
        )


@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("https://APP.PHOENIX.ARIZE.COM./s/example", None),
        ("https://app.phoenix.arize.com/s/example", " \t "),
    ],
)
def test_phoenix_cloud_requires_nonblank_key_for_canonical_host(
    endpoint: str, api_key: str | None
) -> None:
    environ = {"PHOENIX_COLLECTOR_ENDPOINT": endpoint}
    if api_key is not None:
        environ["PHOENIX_API_KEY"] = api_key

    with (
        patch("trec_rag.deepagent_tracing.phoenix_register") as register,
        patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument"),
    ):
        with pytest.raises(ValueError, match="PHOENIX_API_KEY"):
            create_retrieval_tracing(environ=environ)

    register.assert_not_called()


def test_injected_provider_captures_agent_and_retriever_hierarchy(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(
        environ={"PHOENIX_PROJECT_NAME": "trec-rag-deepagent-retrieval"},
        tracer_provider=provider,
    )

    with tracing.agent_span("the narrative"):
        with tracing.retriever_span("follow-up query") as span:
            span.record_search(
                document_ids=("doc-a", "doc-b"),
                source_ranks=(1, 2),
                source_scores=(9.5, 8.0),
                cache_status="hit",
                latency_ms=12.5,
                text_lengths=(10, 20),
            )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["climbmix.retrieve", "deepagent.retrieve"]
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == spans[1].context.span_id
    assert spans[0].attributes["openinference.span.kind"] == "RETRIEVER"
    assert spans[1].attributes["openinference.span.kind"] == "AGENT"
    assert spans[0].attributes["input.value"] == "follow-up query"
    assert spans[1].attributes["input.value"] == "the narrative"
    assert spans[0].attributes["retrieval.document_count"] == 2
    assert spans[0].attributes["retrieval.document_text_lengths"] == (10, 20)
    assert "retrieval.document_excerpts" not in spans[0].attributes


@pytest.mark.parametrize("trace_content", [True, False])
def test_injected_provider_captures_bounded_redacted_snippet_page(
    span_exporter: InMemorySpanExporter,
    provider: TracerProvider,
    trace_content: bool,
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=trace_content
    )

    with tracing.snippet_span("doc-a", "private focus query") as span:
        span.record_page(
            chunk_ids=("doc-a:0007",),
            start_chars=(8100,),
            end_chars=(9310,),
            relevance_scores=(0.91,),
            texts=("bounded snippet text",),
            cache_status="hit",
            ranker_backend="sentence_transformers_cross_encoder",
            latency_ms=7.5,
            page_offset=10,
            has_next_page=True,
        )

    exported = span_exporter.get_finished_spans()[0]
    assert exported.name == "deepagent.extract_relevant_snippets"
    assert exported.attributes["input.value"] == (
        "private focus query" if trace_content else REDACTED_CONTENT
    )
    assert exported.attributes["snippet.document_id"] == "doc-a"
    assert exported.attributes["snippet.chunk_ids"] == ("doc-a:0007",)
    assert exported.attributes["snippet.start_chars"] == (8100,)
    assert exported.attributes["snippet.end_chars"] == (9310,)
    assert exported.attributes["snippet.relevance_scores"] == (0.91,)
    assert exported.attributes["snippet.texts"] == (
        ("bounded snippet text",) if trace_content else (REDACTED_CONTENT,)
    )
    assert exported.attributes["snippet.cache_status"] == "hit"
    assert (
        exported.attributes["snippet.ranker_backend"]
        == "sentence_transformers_cross_encoder"
    )
    assert exported.attributes["snippet.latency_ms"] == 7.5
    assert exported.attributes["snippet.page_offset"] == 10
    assert exported.attributes["snippet.has_next_page"] is True
    assert not {
        "snippet.cache_path",
        "snippet.cache_key",
        "snippet.cursor",
        "snippet.scratch_path",
        "snippet.document_text",
    } & set(exported.attributes)


def test_snippet_trace_api_rejects_more_than_ten_chunks_before_export(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)
    eleven_chunk_ids = tuple(f"doc-a:{index:04d}" for index in range(11))

    with tracing.snippet_span("doc-a", "safe focus query") as span:
        with pytest.raises(ValueError, match="chunk_ids exceeds.*item limit"):
            span.record_page(
                chunk_ids=eleven_chunk_ids,
                start_chars=tuple(range(11)),
                end_chars=tuple(range(1, 12)),
                relevance_scores=tuple(0.5 for _ in range(11)),
                texts=tuple("text" for _ in range(11)),
                cache_status="miss",
                ranker_backend="test-ranker",
                latency_ms=1.0,
                page_offset=0,
                has_next_page=False,
            )

    exported = span_exporter.get_finished_spans()[0]
    assert not any(key.startswith("snippet.") for key in exported.attributes)


def test_metadata_only_masks_manual_span_content(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=False
    )

    with tracing.agent_span("private narrative"):
        with tracing.retriever_span("private query"):
            pass

    spans = span_exporter.get_finished_spans()
    assert [span.attributes["input.value"] for span in spans] == [
        REDACTED_CONTENT,
        REDACTED_CONTENT,
    ]


@pytest.mark.parametrize("trace_content", [False, True])
def test_real_langchain_instrumentation_respects_content_mode_in_fresh_process(
    trace_content: bool,
) -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableLambda
        from langchain_core.tools import tool
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from trec_rag.deepagent_tracing import create_retrieval_tracing

        trace_content = sys.argv[1] == "true"
        sentinels = {
            "narrative": "PRIVATE-NARRATIVE-f0ef50",
            "cursor": "PRIVATE-CURSOR-a741b9",
            "snippet": "PRIVATE-SNIPPET-9a6cd2",
            "scratch": "/scratch/PRIVATE-NOTE-44d172.txt",
        }
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracing = create_retrieval_tracing(
            environ={}, tracer_provider=provider, trace_content=trace_content
        )

        @tool
        def inspect_document(payload: str) -> str:
            \"\"\"Inspect one retrieved document.\"\"\"
            return sentinels["snippet"] + " " + sentinels["scratch"]

        chain = (
            RunnableLambda(
                lambda narrative: {
                    "narrative": narrative,
                    "cursor": sentinels["cursor"],
                }
            ).with_config(run_name="agent")
            | RunnableLambda(
                lambda payload: inspect_document.invoke(json.dumps(payload))
            ).with_config(run_name="scratch")
        )
        chain.invoke(sentinels["narrative"])
        FakeListChatModel(
            responses=[sentinels["snippet"] + " " + sentinels["scratch"]]
        ).invoke(
            [
                HumanMessage(
                    content=sentinels["narrative"] + " " + sentinels["cursor"]
                )
            ]
        )
        with tracing.agent_span(sentinels["narrative"]):
            pass
        with tracing.snippet_span("doc-a", "safe focus query") as span:
            span.record_page(
                chunk_ids=("doc-a:0000",),
                start_chars=(0,),
                end_chars=(len(sentinels["snippet"]),),
                relevance_scores=(1.0,),
                texts=(sentinels["snippet"],),
                cache_status="miss",
                ranker_backend="test",
                latency_ms=1.0,
                page_offset=0,
                has_next_page=False,
            )
        spans = [
            {
                "attributes": dict(span.attributes),
                "events": [
                    {
                        "name": event.name,
                        "attributes": dict(event.attributes),
                    }
                    for event in span.events
                ],
            }
            for span in exporter.get_finished_spans()
        ]
        print(json.dumps({"sentinels": sentinels, "spans": spans}, sort_keys=True))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(trace_content).lower()],
        check=True,
        capture_output=True,
        text=True,
    )
    exported = json.loads(completed.stdout)
    serialized_spans = json.dumps(exported["spans"], sort_keys=True)

    assert exported["spans"]
    if trace_content:
        assert all(
            value in serialized_spans for value in exported["sentinels"].values()
        )
    else:
        assert all(
            value not in serialized_spans for value in exported["sentinels"].values()
        )


@pytest.mark.parametrize("span_factory", ["agent_span", "retriever_span"])
@pytest.mark.parametrize("trace_content", [False, True])
def test_span_facade_has_no_generic_attribute_mutation_path(
    span_exporter: InMemorySpanExporter,
    provider: TracerProvider,
    span_factory: str,
    trace_content: bool,
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=trace_content
    )

    with getattr(tracing, span_factory)("safe supplied content") as span:
        with pytest.raises(AttributeError):
            span.set_attribute("credentials.api_key", "private-api-key")

    exported = span_exporter.get_finished_spans()[0]
    assert "private-api-key" not in str(exported.attributes)


def test_metadata_only_exposes_only_safe_attribute_mutation(
    provider: TracerProvider,
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=False
    )

    with tracing.retriever_span("private query") as span:
        with pytest.raises(AttributeError):
            span.set_attributes({"retrieval.document_excerpt": "private excerpt"})
        with pytest.raises(AttributeError):
            span.add_event("private excerpt")


def test_retriever_trace_api_rejects_blank_document_ids_before_export(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    with tracing.retriever_span("safe query") as span:
        with pytest.raises(ValueError, match="document_ids.*blank"):
            span.record_search(
                document_ids=("",),
                source_ranks=(1,),
                source_scores=(1.0,),
                cache_status="miss",
                latency_ms=1.0,
                text_lengths=(12,),
            )

    exported = span_exporter.get_finished_spans()[0]
    assert "retrieval.document_ids" not in exported.attributes


def test_retriever_trace_api_rejects_negative_text_lengths_before_export(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    with tracing.retriever_span("safe query") as span:
        with pytest.raises(TypeError, match="text_lengths.*non-negative integers"):
            span.record_search(
                document_ids=("doc-a",),
                source_ranks=(1,),
                source_scores=(1.0,),
                cache_status="miss",
                latency_ms=1.0,
                text_lengths=(-1,),
            )

    exported = span_exporter.get_finished_spans()[0]
    assert "retrieval.document_text_lengths" not in exported.attributes


@pytest.mark.parametrize("span_factory", ["agent_span", "retriever_span"])
def test_metadata_only_exception_does_not_export_secret_details(
    span_exporter: InMemorySpanExporter, provider: TracerProvider, span_factory: str
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=False
    )

    with pytest.raises(RuntimeError, match="private exception secret"):
        with getattr(tracing, span_factory)("private supplied content"):
            raise RuntimeError("private exception secret")

    span = span_exporter.get_finished_spans()[0]
    assert "private exception secret" not in str(span.attributes)
    assert "private exception secret" not in str(span.events)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"


@pytest.mark.parametrize("trace_content", [False, True])
def test_instrumentation_is_idempotent_per_provider_identity(
    provider: TracerProvider, trace_content: bool
) -> None:
    with patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument") as instrument:
        create_retrieval_tracing(
            environ={}, tracer_provider=provider, trace_content=trace_content
        )
        create_retrieval_tracing(
            environ={}, tracer_provider=provider, trace_content=trace_content
        )

    assert instrument.call_count == 1
    assert instrument.call_args.kwargs["tracer_provider"] is provider
    config = instrument.call_args.kwargs["config"]
    assert config.hide_llm_invocation_parameters is (not trace_content)
    assert config.hide_inputs is (not trace_content)
    assert config.hide_outputs is (not trace_content)
    assert config.hide_input_text is (not trace_content)
    assert config.hide_output_text is (not trace_content)


def test_equivalent_live_setup_reuses_registration_and_real_instrumentation(
    provider: TracerProvider,
) -> None:
    first_environ = {
        "PHOENIX_COLLECTOR_ENDPOINT": "https://collector.example/base/",
        "PHOENIX_PROJECT_NAME": "safe-project",
        "PHOENIX_API_KEY": "private-live-key",
    }
    equivalent_environ = {
        **first_environ,
        "PHOENIX_COLLECTOR_ENDPOINT": "https://collector.example/base/v1/traces",
    }

    with patch(
        "trec_rag.deepagent_tracing.phoenix_register", return_value=provider
    ) as register:
        first = create_retrieval_tracing(environ=first_environ)
        installed_tracer = LangChainInstrumentor()._tracer
        second = create_retrieval_tracing(environ=equivalent_environ)

    assert register.call_count == 1
    assert first._provider is provider
    assert second._provider is provider
    assert LangChainInstrumentor()._tracer is installed_tracer
    assert "private-live-key" not in repr(deepagent_tracing._LIVE_PROVIDER_CACHE)


def test_distinct_injected_provider_is_rejected_by_real_instrumentor_lifecycle(
    provider: TracerProvider,
) -> None:
    other_provider = TracerProvider()
    create_retrieval_tracing(environ={}, tracer_provider=provider)
    installed_tracer = LangChainInstrumentor()._tracer

    with pytest.raises(ValueError) as exc_info:
        create_retrieval_tracing(environ={}, tracer_provider=other_provider)

    assert str(exc_info.value) == (
        "retrieval tracing is already configured differently in this process"
    )
    assert LangChainInstrumentor()._tracer is installed_tracer


@pytest.mark.parametrize("conflict", ["endpoint", "project"])
def test_conflicting_live_endpoint_or_project_is_rejected_before_registration(
    provider: TracerProvider, conflict: str
) -> None:
    base = {
        "PHOENIX_COLLECTOR_ENDPOINT": "https://collector.example/base",
        "PHOENIX_API_KEY": "private-live-key",
    }
    with patch(
        "trec_rag.deepagent_tracing.phoenix_register", return_value=provider
    ) as register:
        create_retrieval_tracing(
            environ={**base, "PHOENIX_PROJECT_NAME": "first-project"}
        )
        conflicting = {
            **base,
            "PHOENIX_PROJECT_NAME": "first-project",
        }
        private_value = "private-second-project"
        if conflict == "endpoint":
            private_value = "https://private-collector.example/base"
            conflicting["PHOENIX_COLLECTOR_ENDPOINT"] = private_value
        else:
            conflicting["PHOENIX_PROJECT_NAME"] = private_value
        with pytest.raises(ValueError) as exc_info:
            create_retrieval_tracing(environ=conflicting)

    assert str(exc_info.value) == (
        "retrieval tracing is already configured differently in this process"
    )
    assert private_value not in str(exc_info.value)
    register.assert_called_once()


def test_conflicting_content_mode_is_rejected_without_reinstrumenting(
    provider: TracerProvider,
) -> None:
    create_retrieval_tracing(environ={}, tracer_provider=provider, trace_content=True)
    installed_tracer = LangChainInstrumentor()._tracer

    with pytest.raises(
        ValueError,
        match="retrieval tracing is already configured differently in this process",
    ):
        create_retrieval_tracing(
            environ={}, tracer_provider=provider, trace_content=False
        )

    assert LangChainInstrumentor()._tracer is installed_tracer


def test_force_flush_delegates_to_injected_provider(provider: TracerProvider) -> None:
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    assert tracing.force_flush() is True


def test_force_flush_returns_false_when_provider_raises(provider: TracerProvider) -> None:
    def fail_flush(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("exporter unavailable")

    provider.force_flush = fail_flush  # type: ignore[method-assign]
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    assert tracing.force_flush() is False


@pytest.mark.parametrize(
    ("configured_endpoint", "expected_otlp_endpoint"),
    [
        (
            "https://app.phoenix.arize.com/s/example",
            "https://app.phoenix.arize.com/s/example/v1/traces",
        ),
        (
            "https://collector.example/v1/traces",
            "https://collector.example/v1/traces",
        ),
        (
            "http://phoenix.internal:6006/custom/base/",
            "http://phoenix.internal:6006/custom/base/v1/traces",
        ),
    ],
)
def test_live_registration_uses_normalized_otlp_http_endpoint(
    provider: TracerProvider,
    configured_endpoint: str,
    expected_otlp_endpoint: str,
) -> None:
    environ = {
        "PHOENIX_COLLECTOR_ENDPOINT": configured_endpoint,
        "PHOENIX_PROJECT_NAME": "custom-project",
        "PHOENIX_API_KEY": "  test-key  ",
    }
    with (
        patch("trec_rag.deepagent_tracing.phoenix_register", return_value=provider) as register,
        patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument"),
    ):
        tracing = create_retrieval_tracing(environ=environ)

    assert tracing.enabled is True
    register.assert_called_once_with(
        endpoint=expected_otlp_endpoint,
        project_name="custom-project",
        protocol="http/protobuf",
        batch=True,
        api_key="  test-key  ",
        verbose=False,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "collector.example/private-token",
        "ftp://collector.example/private-token",
        "https://user:private-token@collector.example",
        "https://collector.example/traces?api_key=private-token",
        "https://collector.example/traces#private-token",
        "https://collector.example/traces\x01private-token",
        "https://collector.example/traces\x7fprivate-token",
    ],
)
def test_malformed_collector_endpoint_is_rejected_without_disclosure(
    endpoint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("trec_rag.deepagent_tracing.phoenix_register") as register,
        patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument"),
    ):
        with pytest.raises(ValueError) as exc_info:
            create_retrieval_tracing(
                environ={
                    "PHOENIX_COLLECTOR_ENDPOINT": endpoint,
                    "PHOENIX_API_KEY": "test-key",
                }
            )

    register.assert_not_called()
    assert str(exc_info.value) == (
        "PHOENIX_COLLECTOR_ENDPOINT must be an absolute HTTP(S) URL without "
        "credentials, query, or fragment"
    )
    captured = capsys.readouterr()
    assert endpoint not in captured.out
    assert endpoint not in captured.err


def test_injected_provider_never_registers_with_phoenix(provider: TracerProvider) -> None:
    with (
        patch("trec_rag.deepagent_tracing.phoenix_register") as register,
        patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument"),
    ):
        create_retrieval_tracing(
            environ={"PHOENIX_COLLECTOR_ENDPOINT": "https://collector.example/v1/traces"},
            tracer_provider=provider,
        )

    register.assert_not_called()

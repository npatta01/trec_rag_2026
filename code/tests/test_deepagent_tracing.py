from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from trec_rag.deepagent_tracing import REDACTED_CONTENT, create_retrieval_tracing


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


def test_injected_provider_captures_agent_and_retriever_hierarchy(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(
        environ={"PHOENIX_PROJECT_NAME": "trec-rag-deepagent-retrieval"},
        tracer_provider=provider,
    )

    with tracing.agent_span("the narrative"):
        with tracing.retriever_span("follow-up query") as span:
            span.set_attribute("retrieval.document_count", 2)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["climbmix.retrieve", "deepagent.retrieve"]
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == spans[1].context.span_id
    assert spans[0].attributes["openinference.span.kind"] == "RETRIEVER"
    assert spans[1].attributes["openinference.span.kind"] == "AGENT"
    assert spans[0].attributes["input.value"] == "follow-up query"
    assert spans[1].attributes["input.value"] == "the narrative"
    assert spans[0].attributes["retrieval.document_count"] == 2


def test_metadata_only_masks_manual_span_content(
    span_exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=False
    )

    with tracing.agent_span("private narrative"):
        with tracing.retriever_span("private query") as span:
            span.set_attribute("retrieval.document_excerpt", "private excerpt")
            pass

    spans = span_exporter.get_finished_spans()
    assert [span.attributes["input.value"] for span in spans] == [
        REDACTED_CONTENT,
        REDACTED_CONTENT,
    ]
    assert spans[0].attributes["retrieval.document_excerpt"] == REDACTED_CONTENT


def test_instrumentation_is_idempotent_per_provider_identity(provider: TracerProvider) -> None:
    with patch("trec_rag.deepagent_tracing.LangChainInstrumentor.instrument") as instrument:
        create_retrieval_tracing(environ={}, tracer_provider=provider)
        create_retrieval_tracing(environ={}, tracer_provider=provider)

    assert instrument.call_count == 1
    assert instrument.call_args.kwargs["tracer_provider"] is provider
    config = instrument.call_args.kwargs["config"]
    assert config.hide_input_text is False
    assert config.hide_output_text is False


def test_force_flush_delegates_to_injected_provider(provider: TracerProvider) -> None:
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    assert tracing.force_flush() is True

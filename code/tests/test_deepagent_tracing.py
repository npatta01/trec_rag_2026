from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

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
        with tracing.retriever_span("private query"):
            pass

    spans = span_exporter.get_finished_spans()
    assert [span.attributes["input.value"] for span in spans] == [
        REDACTED_CONTENT,
        REDACTED_CONTENT,
    ]


@pytest.mark.parametrize("span_factory", ["agent_span", "retriever_span"])
@pytest.mark.parametrize("trace_content", [False, True])
@pytest.mark.parametrize(
    ("key", "secret_value"),
    [
        ("http.request.header.authorization", "Bearer private-token"),
        ("credentials.api_key", "private-api-key"),
        ("retrieval.raw_response", "private raw response"),
        ("retrieval.continuation_ticket", "private continuation ticket"),
        ("cache.path", "/private/cache/path"),
    ],
)
def test_span_facade_rejects_forbidden_attributes_without_exporting_values(
    span_exporter: InMemorySpanExporter,
    provider: TracerProvider,
    span_factory: str,
    trace_content: bool,
    key: str,
    secret_value: str,
) -> None:
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=trace_content
    )

    with getattr(tracing, span_factory)("safe supplied content") as span:
        with pytest.raises(ValueError, match="not safe for retrieval tracing"):
            span.set_attribute(key, secret_value)

    exported = span_exporter.get_finished_spans()[0]
    assert secret_value not in str(exported.attributes)


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

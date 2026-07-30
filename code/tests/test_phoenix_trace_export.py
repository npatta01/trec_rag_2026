from __future__ import annotations

import pytest

from trec_rag.phoenix_trace_export import (
    ExportReceipt,
    PhoenixSettings,
    SecretStr,
    assert_no_secrets,
    export_trace,
)
from trec_rag.pi_trace_models import SpanSpec, TraceBundle
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Span
from opentelemetry.trace.status import StatusCode


SYNTHETIC_KEY = "synthetic-phoenix-key"


def _span(
    *,
    name: str = "root",
    kind: str = "CHAIN",
    start_ns: int = 10,
    end_ns: int = 40,
    attributes: dict[str, object] | None = None,
    input_value: object | None = None,
    output_value: object | None = None,
    status: str = "OK",
    status_message: str | None = None,
    children: tuple[SpanSpec, ...] = (),
) -> SpanSpec:
    return SpanSpec(
        name=name,
        kind=kind,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes=attributes or {},
        input_value=input_value,
        output_value=output_value,
        status=status,
        status_message=status_message,
        children=children,
    )


def _bundle(root: SpanSpec | None = None) -> TraceBundle:
    return TraceBundle(
        project_name="trace-project",
        session_id="shared-session",
        topic_id="rag2026-1",
        baseline="piika-agentic",
        root=root or _span(),
    )


def _settings() -> PhoenixSettings:
    return PhoenixSettings(
        api_key=SecretStr(SYNTHETIC_KEY),
        collector_endpoint="https://app.phoenix.arize.com",
        project_name="trace-project",
    )


@pytest.mark.parametrize(
    "missing_name",
    ["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_PROJECT_NAME"],
)
def test_settings_fail_closed_when_a_required_environment_value_is_missing(
    monkeypatch, missing_name
):
    configured = {
        "PHOENIX_API_KEY": SYNTHETIC_KEY,
        "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com",
        "PHOENIX_PROJECT_NAME": "trace-project",
    }
    for name, value in configured.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing_name, raising=False)

    with pytest.raises(ValueError, match=f"{missing_name} is required"):
        PhoenixSettings.from_env()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            "https://app.phoenix.arize.com/s/npatta01",
            "https://app.phoenix.arize.com",
        ),
        (
            "https://app.phoenix.arize.com/v1/traces/",
            "https://app.phoenix.arize.com",
        ),
        ("https://phoenix.example.test/custom", "https://phoenix.example.test/custom"),
        (
            "https://phoenix.example.test/s/team",
            "https://phoenix.example.test/s/team",
        ),
    ],
)
def test_settings_normalize_only_supported_collector_suffixes(
    monkeypatch, configured, expected
):
    monkeypatch.setenv("PHOENIX_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", configured)
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "trace-project")

    assert PhoenixSettings.from_env().collector_endpoint == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://app.phoenix.arize.com",
        "https://app.phoenix.arize.com?token=not-allowed",
        "https://app.phoenix.arize.com#fragment-not-allowed",
    ],
)
def test_settings_reject_insecure_or_ambiguous_hosted_endpoints(monkeypatch, endpoint):
    monkeypatch.setenv("PHOENIX_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "trace-project")

    with pytest.raises(ValueError, match="PHOENIX_COLLECTOR_ENDPOINT"):
        PhoenixSettings.from_env()


def test_secret_string_never_reveals_its_value_through_repr_or_str():
    secret = SecretStr(SYNTHETIC_KEY)

    assert SYNTHETIC_KEY not in repr(secret)
    assert SYNTHETIC_KEY not in str(secret)
    assert secret.reveal() == SYNTHETIC_KEY


@pytest.mark.parametrize("location", ["attribute", "input", "output"])
def test_secret_scan_rejects_a_forbidden_value_anywhere_in_span_content(location):
    token = "forbidden-synthetic-token"
    changes = {
        "attributes": {"nested.token": f"prefix-{token}-suffix"}
        if location == "attribute"
        else {},
        "input_value": {"outer": ["safe", {"inner": token}]}
        if location == "input"
        else None,
        "output_value": {"outer": ("safe", {"inner": token})}
        if location == "output"
        else None,
    }
    child = _span(name="nested", start_ns=11, end_ns=12, **changes)

    with pytest.raises(ValueError, match="credential-like value") as caught:
        assert_no_secrets(_bundle(_span(children=(child,))), [token])

    assert token not in str(caught.value)


def test_secret_scan_ignores_values_too_short_to_match_safely():
    bundle = _bundle(_span(input_value={"ordinary": "contains-short"}))

    assert_no_secrets(bundle, ["short", "", None])


class _FakeSpanContext:
    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id


class _FakeSpan(Span):
    def __init__(
        self,
        name,
        parent,
        start_time,
        attributes,
        ordinal,
        on_end,
        fail_status=False,
    ):
        self.name = name
        self.parent = parent
        self.start_time = start_time
        self.attributes = dict(attributes or {})
        self.status = None
        self.end_time = None
        self.fail_status = fail_status
        self.context = _FakeSpanContext(0x1234, ordinal)
        self.on_end = on_end

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def set_status(self, status):
        if self.fail_status:
            raise RuntimeError("synthetic span failure")
        self.status = status

    def end(self, end_time=None):
        self.end_time = end_time
        self.on_end(self)

    def get_span_context(self):
        return self.context

    def add_event(self, name, attributes=None, timestamp=None):
        pass

    def update_name(self, name):
        self.name = name

    def is_recording(self):
        return self.end_time is None

    def record_exception(
        self, exception, attributes=None, timestamp=None, escaped=False
    ):
        pass


class _FakeTracer:
    def __init__(self, *, on_end, fail_name=None):
        self.spans = []
        self.fail_name = fail_name
        self.on_end = on_end

    def start_span(self, name, context=None, start_time=None, attributes=None):
        parent = trace.get_current_span(context) if context is not None else None
        span = _FakeSpan(
            name,
            parent,
            start_time,
            attributes,
            len(self.spans) + 1,
            self.on_end,
            fail_status=name == self.fail_name,
        )
        self.spans.append(span)
        return span


class _FakeExporter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def export(self, spans):
        self.calls.append(tuple(spans))
        return self.result


class _FakeProcessor:
    def __init__(self, exporter):
        self.span_exporter = exporter


class _FakeMultiProcessor:
    def __init__(self, processor):
        self._span_processors = (processor,)


class _FakeProvider:
    def __init__(
        self,
        *,
        fail_name=None,
        export_result=SpanExportResult.SUCCESS,
        flush_result=True,
        suppress_export=False,
    ):
        self.exporter = _FakeExporter(export_result)
        self._active_span_processor = _FakeMultiProcessor(
            _FakeProcessor(self.exporter)
        )
        self.tracer = _FakeTracer(
            on_end=(
                (lambda span: None)
                if suppress_export
                else (lambda span: self.exporter.export((span,)))
            ),
            fail_name=fail_name,
        )
        self.tracer_name = None
        self.flush_calls = 0
        self.shutdown_calls = 0
        self.flush_result = flush_result

    def get_tracer(self, name):
        self.tracer_name = name
        return self.tracer

    def force_flush(self):
        self.flush_calls += 1
        return self.flush_result

    def shutdown(self):
        self.shutdown_calls += 1


class _ProviderFactory:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.provider


def test_export_recursively_preserves_trace_semantics_and_returns_public_ids():
    grandchild = _span(
        name="generation",
        kind="LLM",
        start_ns=20,
        end_ns=30,
        attributes={"model.name": "synthetic-model"},
        input_value={"messages": [{"role": "user", "content": "question"}]},
        output_value={"answer": "response"},
        status="ERROR",
        status_message="synthetic failure",
    )
    child = _span(
        name="retrieval",
        kind="RETRIEVER",
        start_ns=15,
        end_ns=35,
        children=(grandchild,),
    )
    root = _span(
        start_ns=10,
        end_ns=40,
        attributes={"content.capture": "full"},
        children=(child,),
    )
    provider = _FakeProvider()
    factory = _ProviderFactory(provider)

    receipt = export_trace(_bundle(root), _settings(), provider_factory=factory)

    assert factory.calls == [
        {
            "project_name": "trace-project",
            "endpoint": "https://app.phoenix.arize.com",
            "api_key": SYNTHETIC_KEY,
            "batch": False,
            "verbose": False,
            "set_global_tracer_provider": False,
        }
    ]
    assert [span.name for span in provider.tracer.spans] == [
        "root",
        "retrieval",
        "generation",
    ]
    exported_root, exported_child, exported_grandchild = provider.tracer.spans
    assert exported_root.parent is None
    assert exported_child.parent is exported_root
    assert exported_grandchild.parent is exported_child
    assert [span.start_time for span in provider.tracer.spans] == [10, 15, 20]
    assert [span.end_time for span in provider.tracer.spans] == [40, 35, 30]
    assert exported_root.attributes == {
        "content.capture": "full",
        "openinference.span.kind": "CHAIN",
        "session.id": "shared-session",
        "topic.id": "rag2026-1",
        "baseline": "piika-agentic",
    }
    assert exported_child.attributes["openinference.span.kind"] == "RETRIEVER"
    assert exported_grandchild.attributes["openinference.span.kind"] == "LLM"
    assert exported_grandchild.attributes["input.mime_type"] == "application/json"
    assert exported_grandchild.attributes["input.value"] == (
        '{"messages":[{"content":"question","role":"user"}]}'
    )
    assert exported_grandchild.attributes["output.mime_type"] == "application/json"
    assert exported_grandchild.attributes["output.value"] == '{"answer":"response"}'
    assert exported_root.status.status_code is StatusCode.OK
    assert exported_grandchild.status.status_code is StatusCode.ERROR
    assert exported_grandchild.status.description == "synthetic failure"
    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1
    assert receipt == ExportReceipt(
        project_name="trace-project",
        trace_id="00000000000000000000000000001234",
        root_span_id="0000000000000001",
        exported_span_count=3,
    )


def test_export_rejects_secret_content_before_constructing_provider():
    factory = _ProviderFactory(_FakeProvider())
    bundle = _bundle(_span(input_value={"leak": f"prefix-{SYNTHETIC_KEY}"}))

    with pytest.raises(ValueError, match="credential-like value"):
        export_trace(bundle, _settings(), provider_factory=factory)

    assert factory.calls == []


def test_export_ends_open_spans_flushes_and_shuts_down_after_export_error():
    child = _span(name="fails", start_ns=20, end_ns=30)
    bundle = _bundle(_span(start_ns=10, end_ns=40, children=(child,)))
    provider = _FakeProvider(fail_name="fails")

    with pytest.raises(RuntimeError, match="synthetic span failure"):
        export_trace(bundle, _settings(), provider_factory=_ProviderFactory(provider))

    root_span, failed_span = provider.tracer.spans
    assert root_span.end_time == 40
    assert failed_span.end_time == 30
    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1


def test_export_rejects_a_failed_synchronous_export_result_and_shuts_down():
    provider = _FakeProvider(export_result=SpanExportResult.FAILURE)

    with pytest.raises(RuntimeError, match="Phoenix export did not complete"):
        export_trace(_bundle(), _settings(), provider_factory=_ProviderFactory(provider))

    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1


def test_export_rejects_a_false_flush_result_and_shuts_down():
    provider = _FakeProvider(flush_result=False)

    with pytest.raises(RuntimeError, match="Phoenix export did not complete"):
        export_trace(_bundle(), _settings(), provider_factory=_ProviderFactory(provider))

    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1


def test_export_rejects_when_sampling_suppresses_every_export_call():
    provider = _FakeProvider(suppress_export=True)

    with pytest.raises(RuntimeError, match="Phoenix export did not complete"):
        export_trace(_bundle(), _settings(), provider_factory=_ProviderFactory(provider))

    assert provider.exporter.calls == []
    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1


def test_export_rejects_an_unverifiable_provider_and_still_cleans_it_up():
    provider = _FakeProvider()
    del provider._active_span_processor

    with pytest.raises(RuntimeError, match="result verification is unavailable"):
        export_trace(_bundle(), _settings(), provider_factory=_ProviderFactory(provider))

    assert provider.tracer.spans == []
    assert provider.flush_calls == 1
    assert provider.shutdown_calls == 1

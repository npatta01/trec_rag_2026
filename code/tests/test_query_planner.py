import json
import urllib.error
import urllib.request
from copy import deepcopy
from io import BytesIO

import pytest

from trec_rag.query_planner import (
    ANALYZER_VERSION,
    DEVELOPER_PROMPT,
    PROMPT_VERSION,
    RENDERER_VERSION,
    SCHEMA_VERSION,
    Anchor,
    CoverageItem,
    ExpansionTerm,
    Facet,
    GlobalExpansion,
    QueryPlan,
    QueryPlanGenerationError,
    QueryPlanGenerator,
    QueryPlanHttpResponseError,
    QueryPlanHttpTransportError,
    QueryPlanValidationError,
    UrllibJsonTransport,
    analyze_content_terms,
    developer_prompt,
    global_expansion_term_cap,
    parse_query_plan,
    query_plan_json_schema,
    render_query_plan,
)
from trec_rag.topics import Topic


def _topic() -> Topic:
    return Topic(
        id="31",
        title="E-waste impacts",
        narrative=(
            "Explain e-waste health risks, battery recycling innovations, "
            "and local sustainability benefits."
        ),
    )


def _expansion(term: str, anchor_ref: str, *, relation: str = "alias") -> dict[str, object]:
    return {
        "term": term,
        "relation": relation,
        "anchor_refs": [anchor_ref],
        "provenance": "model_alias",
    }


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_id": "31",
        "complexity_class": "compound",
        "facet_count": 2,
        "anchors": [
            {"anchor_id": "a_topic", "source_span": "e-waste", "kind": "topic"},
            {
                "anchor_id": "a_health",
                "source_span": "health risks",
                "kind": "relation",
            },
            {
                "anchor_id": "a_recycling",
                "source_span": "battery recycling innovations",
                "kind": "topic",
            },
            {
                "anchor_id": "a_local",
                "source_span": "local sustainability benefits",
                "kind": "constraint",
            },
        ],
        "parent_anchor_refs": ["a_topic"],
        "global_expansion": {
            "terms": [
                _expansion("electronic waste", "a_topic"),
                _expansion("waste management", "a_topic", relation="neutral_search_term"),
                _expansion("electronic recycling", "a_recycling", relation="common_variant"),
            ]
        },
        "facets": [
            {
                "facet_id": "f_health",
                "priority": "core",
                "information_need": "Health effects of discarded electronics",
                "source_spans": ["health risks"],
                "anchor_refs": ["a_topic", "a_health"],
                "expansion_terms": [
                    _expansion("toxic exposure", "a_health", relation="technical_term"),
                    _expansion("public health", "a_health", relation="neutral_search_term"),
                ],
                "depends_on": [],
            },
            {
                "facet_id": "f_recycling",
                "priority": "core",
                "information_need": "Recycling innovation and local sustainability benefits",
                "source_spans": ["battery recycling innovations"],
                "anchor_refs": ["a_topic", "a_recycling", "a_local"],
                "expansion_terms": [
                    _expansion("circular economy", "a_recycling", relation="technical_term")
                ],
                "depends_on": [],
            },
        ],
        "coverage_map": [
            {
                "coverage_id": "c_health",
                "source_span": "health risks",
                "facet_refs": ["f_health"],
            },
            {
                "coverage_id": "c_recycling",
                "source_span": "battery recycling innovations",
                "facet_refs": ["f_recycling"],
            },
            {
                "coverage_id": "c_local",
                "source_span": "local sustainability benefits",
                "facet_refs": ["f_recycling"],
            },
        ],
    }


def _completion(payload: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl-plan-1",
        "model": "planner-served-model",
        "usage": {"prompt_tokens": 100, "completion_tokens": 75, "total_tokens": 175},
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
    }


class RecordingTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def post_json(self, url, payload, headers, timeout):
        self.calls.append((url, payload, headers, timeout))
        return self.response


def test_public_schema_and_request_payload_are_versioned_and_constrained():
    topic = _topic()
    schema = query_plan_json_schema(
        topic_id=topic.id,
        max_facets=3,
        max_global_terms=global_expansion_term_cap(topic.narrative),
    )

    assert SCHEMA_VERSION == "query_plan_v1"
    assert PROMPT_VERSION
    assert RENDERER_VERSION
    assert ANALYZER_VERSION
    assert "Use 1-8 facets" in DEVELOPER_PROMPT
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["properties"]["topic_id"]["const"] == topic.id
    assert schema["properties"]["facets"]["minItems"] == 1
    assert schema["properties"]["facets"]["maxItems"] == 3

    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1/",
        model="planner-local",
        max_tokens=777,
        temperature=0.25,
        reasoning_effort="low",
        max_facets=3,
    )
    request = generator.request_payload(topic)

    assert request["model"] == "planner-local"
    assert request["max_tokens"] == 777
    assert request["temperature"] == 0.25
    assert request["reasoning_effort"] == "low"
    assert request["messages"][0] == {
        "role": "developer",
        "content": developer_prompt(schema),
    }
    assert topic.narrative in request["messages"][1]["content"]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == SCHEMA_VERSION
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema

    with pytest.raises(ValueError, match="between 1 and 8"):
        query_plan_json_schema(topic_id=topic.id, max_facets=9)


def test_parse_plan_preserves_literal_spans_and_normalizes_generated_text():
    payload = _valid_payload()
    payload["facets"][0]["information_need"] = "  Health   effects of discarded electronics  "
    payload["global_expansion"]["terms"][0]["term"] = "  electronic   waste  "

    plan = parse_query_plan(payload, topic=_topic())

    assert plan.anchors[0].source_span == "e-waste"
    assert plan.facets[0].source_spans == ("health risks",)
    assert plan.facets[0].information_need == "Health effects of discarded electronics"
    assert plan.global_expansion.terms[0].term == "electronic waste"


def test_parse_plan_rejects_non_literal_span_even_when_only_case_differs():
    payload = _valid_payload()
    payload["anchors"][0]["source_span"] = "E-waste"

    with pytest.raises(QueryPlanValidationError, match="exact narrative substring"):
        parse_query_plan(payload, topic=_topic())


def test_parse_plan_rejects_whitespace_only_literal_span():
    payload = _valid_payload()
    payload["anchors"][0]["source_span"] = "   "

    with pytest.raises(QueryPlanValidationError, match="must be non-empty text"):
        parse_query_plan(payload, topic=_topic())


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("anchor", "unknown_anchor"),
        ("facet", "unknown_facet"),
        ("dependency", "unknown_dependency"),
    ],
)
def test_parse_plan_rejects_dangling_references(mutation, error):
    payload = _valid_payload()
    if mutation == "anchor":
        payload["facets"][0]["anchor_refs"].append(error)
    elif mutation == "facet":
        payload["coverage_map"][0]["facet_refs"].append(error)
    else:
        payload["facets"][0]["depends_on"].append(error)

    with pytest.raises(QueryPlanValidationError, match=error):
        parse_query_plan(payload, topic=_topic())


def test_global_expansion_cap_adapts_to_narrative_content():
    topic = _topic()
    assert analyze_content_terms(topic.narrative) == (
        "explain",
        "e-waste",
        "health",
        "risks",
        "battery",
        "recycling",
        "innovations",
        "local",
        "sustainability",
        "benefits",
    )
    assert global_expansion_term_cap(topic.narrative) == 3

    payload = _valid_payload()
    valid_plan = parse_query_plan(payload, topic=topic)
    rendered_global = render_query_plan(topic, valid_plan)[0]
    original_terms = set(analyze_content_terms(topic.narrative))
    assert tuple(
        term for term in rendered_global.unique_content_tokens if term not in original_terms
    ) == ("electronic", "waste", "management")

    payload["global_expansion"]["terms"].append(
        _expansion("resource recovery", "a_recycling", relation="technical_term")
    )
    with pytest.raises(QueryPlanValidationError, match="narrative-dependent cap of 3"):
        parse_query_plan(payload, topic=topic)


def test_renderer_rejects_multiword_global_terms_that_exceed_unique_token_cap():
    payload = _valid_payload()
    payload["global_expansion"]["terms"] = [
        _expansion("electronic waste", "a_topic"),
        _expansion("circular economy", "a_recycling", relation="technical_term"),
        _expansion("community jobs", "a_local", relation="neutral_search_term"),
    ]
    plan = parse_query_plan(payload, topic=_topic())

    with pytest.raises(QueryPlanValidationError, match="more than 3 unique added"):
        render_query_plan(_topic(), plan)


def test_renderer_is_deterministic_and_deduplicates_nested_components():
    payload = _valid_payload()
    payload["global_expansion"]["terms"][2] = _expansion(
        "health risks", "a_health", relation="neutral_search_term"
    )
    plan = parse_query_plan(payload, topic=_topic())

    first = render_query_plan(_topic(), plan)
    second = render_query_plan(_topic(), plan)

    assert first == second
    assert first[0].variant_name == "global_expansion"
    assert first[0].components[0] == _topic().narrative
    assert first[0].components.count("health risks") == 0
    assert first[0].query_text.casefold().count("health risks") == 1
    assert first[1].components == (
        "health risks",
        "e-waste",
        "toxic exposure",
        "public health",
    )
    assert len(first[1].unique_content_tokens) >= 5
    assert first[1].renderer_budget_exception is False


def test_renderer_orders_source_spans_and_missing_anchors_by_narrative_position():
    payload = _valid_payload()
    payload["facets"][1]["source_spans"] = [
        "local sustainability benefits",
        "battery recycling innovations",
    ]
    payload["facets"][1]["anchor_refs"] = ["a_local", "a_topic", "a_recycling"]
    plan = parse_query_plan(payload, topic=_topic())

    rendered = render_query_plan(_topic(), plan)[2]

    assert rendered.components[:3] == (
        "battery recycling innovations",
        "local sustainability benefits",
        "e-waste",
    )


def test_renderer_rejects_facets_below_five_content_terms():
    payload = _valid_payload()
    payload["facets"][0]["priority"] = "supporting"
    payload["facets"][0]["anchor_refs"] = ["a_health"]
    payload["facets"][0]["expansion_terms"] = []
    plan = parse_query_plan(payload, topic=_topic())

    with pytest.raises(QueryPlanValidationError, match="fewer than 5"):
        render_query_plan(_topic(), plan)


def _long_plan(*, one_source_span: bool) -> tuple[Topic, QueryPlan]:
    first = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar"
    second = "papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber bronze copper dune"
    topic = Topic(id="long", title="Long", narrative=f"{first} {second}")
    source_spans = (topic.narrative,) if one_source_span else (first, second)
    facet = Facet(
        facet_id="f_long",
        priority="core",
        information_need="Long diagnostic facet",
        source_spans=source_spans,
        anchor_refs=("a_long",),
        expansion_terms=(),
        depends_on=(),
    )
    plan = QueryPlan(
        schema_version=SCHEMA_VERSION,
        topic_id=topic.id,
        complexity_class="simple",
        facet_count=1,
        anchors=(Anchor("a_long", "alpha", "topic"),),
        parent_anchor_refs=("a_long",),
        global_expansion=GlobalExpansion(terms=()),
        facets=(facet,),
        coverage_map=(CoverageItem("c_long", first, ("f_long",)),),
    )
    return topic, plan


def test_renderer_rejects_splittable_facets_above_25_content_terms():
    topic, plan = _long_plan(one_source_span=False)

    with pytest.raises(QueryPlanValidationError, match="more than 25"):
        render_query_plan(topic, plan)


def test_renderer_marks_indivisible_source_span_budget_exception():
    topic, plan = _long_plan(one_source_span=True)

    facet_query = render_query_plan(topic, plan)[1]

    assert len(facet_query.unique_content_tokens) > 25
    assert facet_query.renderer_budget_exception is True


def test_renderer_rejects_short_sole_span_when_added_components_exceed_budget():
    source_span = "alpha bravo charlie delta echo"
    added_anchor_spans = (
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
    )
    topic = Topic(
        id="expanded-long",
        title="Expanded long facet",
        narrative=f"{source_span} {' '.join(added_anchor_spans)}",
    )
    anchors = (Anchor("a_topic", "alpha", "topic"),) + tuple(
        Anchor(f"a_{index}", span, "constraint")
        for index, span in enumerate(added_anchor_spans, start=1)
    )
    facet = Facet(
        facet_id="f_expanded",
        priority="core",
        information_need="Short source span plus many added components",
        source_spans=(source_span,),
        anchor_refs=tuple(anchor.anchor_id for anchor in anchors),
        expansion_terms=(
            ExpansionTerm(
                "silver gold",
                "neutral_search_term",
                ("a_topic",),
                "model_alias",
            ),
            ExpansionTerm(
                "bronze copper",
                "neutral_search_term",
                ("a_topic",),
                "model_alias",
            ),
        ),
        depends_on=(),
    )
    plan = QueryPlan(
        schema_version=SCHEMA_VERSION,
        topic_id=topic.id,
        complexity_class="broad",
        facet_count=1,
        anchors=anchors,
        parent_anchor_refs=("a_topic",),
        global_expansion=GlobalExpansion(terms=()),
        facets=(facet,),
        coverage_map=(CoverageItem("c_expanded", source_span, ("f_expanded",)),),
    )

    with pytest.raises(QueryPlanValidationError, match="more than 25"):
        render_query_plan(topic, plan)


def test_urllib_transport_serializes_request_and_requires_object_response(monkeypatch):
    captured = {}

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response(b'{"choices": []}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = UrllibJsonTransport()
    assert transport.post_json(
        "http://planner.test/v1/chat/completions",
        {"model": "local"},
        {"Content-Type": "application/json"},
        12.5,
    ) == {"choices": []}
    assert captured == {
        "url": "http://planner.test/v1/chat/completions",
        "body": {"model": "local"},
        "timeout": 12.5,
    }

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"[]"))
    with pytest.raises(ValueError, match="JSON object"):
        transport.post_json("http://planner.test", {}, {}, 1)


def test_urllib_transport_surfaces_http_error_body(monkeypatch):
    def fake_urlopen(_request, timeout):
        assert timeout == 3
        raise urllib.error.HTTPError(
            url="http://planner.test/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b'{"error":"model failed"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="model failed"):
        UrllibJsonTransport().post_json("http://planner.test", {}, {}, 3)


def test_urllib_raw_hook_precedes_utf8_decode_and_preserves_invalid_bytes_once(
    monkeypatch,
):
    raw_body = b'\xff{"unfinished":'

    class Response:
        status = 200
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": "invalid-wire-1",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return raw_body

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, timeout: Response()
    )
    observed = []

    with pytest.raises(QueryPlanHttpResponseError) as error:
        UrllibJsonTransport().post_json_with_raw_hook(
            "http://planner.test/v1/chat/completions",
            {},
            {},
            3,
            observed.append,
        )

    assert len(observed) == 1
    assert observed[0].raw_body == raw_body
    assert observed[0].http_status == 200
    assert observed[0].response_headers["X-Request-ID"] == "invalid-wire-1"
    assert error.value.raw_body == raw_body
    assert error.value.http_status == 200
    assert error.value.response_headers["X-Request-ID"] == "invalid-wire-1"


def test_urllib_successful_raw_hook_fires_once(monkeypatch):
    raw_body = b'{"choices":[]}'

    class Response:
        status = 200
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": "successful-wire-1",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return raw_body

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, timeout: Response()
    )
    observed = []

    response = UrllibJsonTransport().post_json_with_raw_hook(
        "http://planner.test/v1/chat/completions",
        {},
        {},
        3,
        observed.append,
    )

    assert response == {"choices": []}
    assert response.raw_body == raw_body
    assert len(observed) == 1
    assert observed[0].raw_body == raw_body
    assert observed[0].http_status == 200
    assert observed[0].response_headers["X-Request-ID"] == "successful-wire-1"


def test_urllib_http_error_is_raw_hooked_before_typed_transport_error(monkeypatch):
    raw_body = b'{"error":"model failed"}'

    def fake_urlopen(_request, timeout):
        assert timeout == 3
        raise urllib.error.HTTPError(
            url="http://planner.test/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs={
                "Content-Type": "application/json",
                "X-Request-ID": "failed-wire-1",
            },
            fp=BytesIO(raw_body),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    events = []

    def raw_hook(response):
        events.append(("hook", response))

    with pytest.raises(QueryPlanHttpTransportError) as error:
        UrllibJsonTransport().post_json_with_raw_hook(
            "http://planner.test/v1/chat/completions",
            {},
            {},
            3,
            raw_hook,
        )
    events.append(("raised", error.value))

    assert [name for name, _value in events] == ["hook", "raised"]
    hooked = events[0][1]
    assert hooked.raw_body == raw_body
    assert hooked.http_status == 503
    assert hooked.response_headers["X-Request-ID"] == "failed-wire-1"
    assert error.value.raw_body == raw_body
    assert error.value.http_status == 503
    assert error.value.response_headers["X-Request-ID"] == "failed-wire-1"


def test_urllib_raw_hook_persistence_failure_aborts_before_json_parsing(
    monkeypatch,
):
    class PersistenceFailure(RuntimeError):
        pass

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[]}'

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, timeout: Response()
    )
    parse_calls = []

    def forbidden_json_loads(*_args, **_kwargs):
        parse_calls.append(True)
        raise AssertionError("response JSON must not be parsed after persistence fails")

    monkeypatch.setattr(json, "loads", forbidden_json_loads)

    def failing_hook(_response):
        raise PersistenceFailure("could not persist raw response")

    with pytest.raises(PersistenceFailure, match="could not persist"):
        UrllibJsonTransport().post_json_with_raw_hook(
            "http://planner.test/v1/chat/completions",
            {},
            {},
            3,
            failing_hook,
        )

    assert parse_calls == []


def test_malformed_http_json_is_preserved_before_invalid_json_error(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Type": "application/json", "X-Request-ID": "wire-1"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"unfinished":'

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, timeout: Response()
    )
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        transport=UrllibJsonTransport(),
    )
    observed = []

    with pytest.raises(QueryPlanGenerationError) as error:
        generator.generate(
            _topic(),
            cache=False,
            response_hook=lambda response, elapsed: observed.append((response, elapsed)),
        )

    assert error.value.status == "invalid_json"
    assert error.value.content == '{"unfinished":'
    assert len(observed) == 1
    response, elapsed = observed[0]
    assert elapsed >= 0
    assert response == {}
    assert response.raw_body == b'{"unfinished":'
    assert response.http_status == 200
    assert response.response_headers["X-Request-ID"] == "wire-1"


def test_generation_error_rejects_unknown_status():
    with pytest.raises(ValueError, match="unsupported query-plan generation status"):
        QueryPlanGenerationError("bad status", status="not_in_the_taxonomy")


def test_generator_rejects_malformed_chat_completion_envelope():
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        transport=RecordingTransport({"choices": []}),
    )

    with pytest.raises(QueryPlanGenerationError, match=r"choices\[0\].message.content"):
        generator.generate(_topic(), cache=False)


def test_response_hook_runs_before_invalid_json_error_and_records_status():
    response = {
        "id": "chatcmpl-invalid-json",
        "model": "planner-served-model",
        "choices": [{"message": {"role": "assistant", "content": "not JSON"}}],
    }
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        transport=RecordingTransport(response),
    )
    events: list[tuple[str, object]] = []

    def response_hook(received, elapsed):
        events.append(("response_hook", (received, elapsed)))

    try:
        generator.generate(_topic(), cache=False, response_hook=response_hook)
    except QueryPlanGenerationError as exc:
        events.append(("error", exc.status))
        assert exc.status == "invalid_json"
        assert exc.response == response
        assert exc.content == "not JSON"
    else:
        pytest.fail("invalid JSON unexpectedly produced a query plan")

    assert [name for name, _ in events] == ["response_hook", "error"]
    hooked_response, elapsed = events[0][1]
    assert hooked_response == response
    assert elapsed >= 0


def test_response_hook_runs_before_plan_validation_error_and_records_status():
    payload = deepcopy(_valid_payload())
    payload["facet_count"] = 3
    response = _completion(payload)
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        transport=RecordingTransport(response),
    )
    events: list[str] = []

    def response_hook(_response, _elapsed):
        events.append("response_hook")

    try:
        generator.generate(_topic(), cache=False, response_hook=response_hook)
    except QueryPlanGenerationError as exc:
        events.append("error")
        assert exc.status == "plan_validation_error"
        assert exc.response == response
        assert json.loads(exc.content) == payload
    else:
        pytest.fail("invalid plan unexpectedly passed validation")

    assert events == ["response_hook", "error"]


def test_generator_distinguishes_render_validation_status_after_response_hook():
    payload = deepcopy(_valid_payload())
    payload["facets"][0]["expansion_terms"] = []
    response = _completion(payload)
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        transport=RecordingTransport(response),
    )
    events: list[str] = []

    def response_hook(_response, _elapsed):
        events.append("response_hook")

    try:
        generator.generate(_topic(), cache=False, response_hook=response_hook)
    except QueryPlanGenerationError as exc:
        events.append("error")
        assert exc.status == "render_validation_error"
        assert exc.response == response
        assert json.loads(exc.content) == payload
    else:
        pytest.fail("underfilled rendered query unexpectedly passed validation")

    assert events == ["response_hook", "error"]


def test_generator_caches_plan_and_reuses_provenance(tmp_path):
    transport = RecordingTransport(_completion(_valid_payload()))
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1/",
        model="planner-local",
        api_key="secret",
        model_revision="revision-1",
        cache_dir=tmp_path,
        timeout=15,
        max_tokens=900,
        temperature=0.5,
        reasoning_effort="low",
        transport=transport,
    )

    first = generator.generate(_topic())
    second = generator.generate(_topic())

    assert len(transport.calls) == 1
    url, request, headers, timeout = transport.calls[0]
    assert url == "http://planner.test/v1/chat/completions"
    assert request == generator.request_payload(_topic())
    assert headers["Authorization"] == "Bearer secret"
    assert timeout == 15
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.plan == second.plan
    assert first.provenance == second.provenance
    assert first.cache_path == second.cache_path

    cache = json.loads(first.cache_path.read_text(encoding="utf-8"))
    assert cache["topic"] == {
        "id": _topic().id,
        "title": _topic().title,
        "narrative": _topic().narrative,
    }
    assert cache["plan"]["schema_version"] == SCHEMA_VERSION
    assert cache["provenance"]["base_url"] == "http://planner.test/v1"
    assert cache["provenance"]["model"] == "planner-local"
    assert cache["provenance"]["model_revision"] == "revision-1"
    assert cache["provenance"]["schema_version"] == SCHEMA_VERSION
    assert cache["provenance"]["prompt_version"] == PROMPT_VERSION
    assert cache["provenance"]["renderer_version"] == RENDERER_VERSION
    assert cache["provenance"]["analyzer_version"] == ANALYZER_VERSION
    assert cache["provenance"]["reasoning_effort"] == "low"
    assert cache["provenance"]["usage"]["total_tokens"] == 175
    assert cache["provenance"]["elapsed_seconds"] >= 0
    assert cache["provenance"]["response_id"] == "chatcmpl-plan-1"
    assert cache["provenance"]["response_model"] == "planner-served-model"
    assert [row["variant_name"] for row in cache["rendered_queries"]] == [
        "global_expansion",
        "facet:f_health",
        "facet:f_recycling",
    ]


def test_cache_rejects_topic_and_provenance_tampering(tmp_path):
    transport = RecordingTransport(_completion(_valid_payload()))
    generator = QueryPlanGenerator(
        base_url="http://planner.test/v1",
        model="planner-local",
        cache_dir=tmp_path,
        transport=transport,
    )
    result = generator.generate(_topic())
    original = json.loads(result.cache_path.read_text(encoding="utf-8"))

    bad_topic = deepcopy(original)
    bad_topic["topic"]["narrative"] = "different narrative"
    result.cache_path.write_text(json.dumps(bad_topic), encoding="utf-8")
    with pytest.raises(QueryPlanValidationError, match="cache topic"):
        generator.generate(_topic())

    bad_provenance = deepcopy(original)
    bad_provenance["provenance"]["model"] = "different-model"
    result.cache_path.write_text(json.dumps(bad_provenance), encoding="utf-8")
    with pytest.raises(QueryPlanValidationError, match="provenance mismatch.*model"):
        generator.generate(_topic())

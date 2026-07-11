"""Contract tests for the v2 model boundary, with no live model calls."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Mapping

import pytest

from trec_rag import query_planner
from trec_rag.query_analyzer import (
    AnalyzedQuery,
    AnalyzerFingerprint,
    stable_unique,
)
from trec_rag.query_planner import (
    TOKENIZER_VERSION,
    V2_DEVELOPER_PROMPT,
    V2_PROMPT_VERSION,
    V2_SCHEMA_VERSION,
    HttpJsonResponse,
    QueryPlanHttpResponseError,
    QueryPlanGeneratorV2,
    developer_prompt_v2,
    query_plan_v2_json_schema,
    tokenize_narrative,
)
from trec_rag.topics import Topic


def _topic() -> Topic:
    return Topic(
        id="boundary",
        title="Battery recycling",
        narrative="Explain battery recycling health risks and local benefits.",
    )


def _valid_plan() -> dict[str, object]:
    # 0 Explain | 1 battery | 2 recycling | 3 health | 4 risks |
    # 5 and | 6 local | 7 benefits | 8 .
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "topic_id": "boundary",
        "anchors": [
            {
                "anchor_id": "a_battery",
                "range": {"start_token": 1, "end_token": 3},
                "kind": "topic",
                "scope": "global",
                "coverage_refs": [],
            },
            {
                "anchor_id": "a_health",
                "range": {"start_token": 3, "end_token": 5},
                "kind": "relation",
                "scope": "coverage",
                "coverage_refs": ["c_health"],
            },
            {
                "anchor_id": "a_local",
                "range": {"start_token": 6, "end_token": 8},
                "kind": "constraint",
                "scope": "coverage",
                "coverage_refs": ["c_local"],
            },
        ],
        "coverage_items": [
            {
                "coverage_id": "c_health",
                "source_span_refs": [{"start_token": 3, "end_token": 5}],
            },
            {
                "coverage_id": "c_local",
                "source_span_refs": [{"start_token": 6, "end_token": 8}],
            },
        ],
        "facets": [
            {
                "facet_id": "f_health",
                "coverage_refs": ["c_health"],
                "expansion_terms": [
                    {
                        "term": "toxic exposure",
                        "relation": "technical_term",
                        "anchor_refs": ["a_health"],
                    }
                ],
            },
            {
                "facet_id": "f_local",
                "coverage_refs": ["c_local"],
                "expansion_terms": [
                    {
                        "term": "community gains",
                        "relation": "common_variant",
                        "anchor_refs": ["a_local"],
                    }
                ],
            },
        ],
        "global_expansion": {"terms": []},
    }


FINGERPRINT = AnalyzerFingerprint(
    contract_version="lucene_query_analyzer_v1",
    implementation="test.fake_exact_analyzer",
    lucene_version="9.12.1",
    analyzer_class="org.apache.lucene.analysis.standard.StandardAnalyzer",
    tokenizer="StandardTokenizer",
    filters=("LowerCaseFilter",),
    stopword_sha256=hashlib.sha256(b"").hexdigest(),
    unicode_version=None,
    index_id="test-index-revision",
)


class FakeExactAnalyzer:
    fingerprint = FINGERPRINT

    def analyze(self, text: str) -> AnalyzedQuery:
        tokens = tuple(
            match.group(0).casefold()
            for match in re.finditer(r"[^\W_]+(?:[-'’][^\W_]+)*", text)
        )
        return AnalyzedQuery(tokens, stable_unique(tokens), self.fingerprint)


class RecordingTransport:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def post_json(self, url, payload, headers, timeout):
        self.calls.append((url, payload, headers, timeout))
        return self.response


def _response(
    content: str,
    *,
    response_id: str = "chatcmpl-v2",
    finish_reason: str = "stop",
) -> HttpJsonResponse:
    payload = {
        "id": response_id,
        "model": "served-v2-model",
        "usage": {"prompt_tokens": 300, "completion_tokens": 200, "total_tokens": 500},
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    raw_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return HttpJsonResponse(
        payload,
        raw_body=raw_body,
        http_status=200,
        response_headers={"Content-Type": "application/json", "X-Request-ID": "wire-v2"},
    )


def _walk_schema(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_schema(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_schema(child)


def _all_property_names(schema: dict[str, object]) -> set[str]:
    names: set[str] = set()
    for node in _walk_schema(schema):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
    return names


def _generator(response: Mapping[str, object], **overrides) -> QueryPlanGeneratorV2:
    kwargs = {
        "base_url": "http://planner.test/v1/",
        "model": "planner-v2-local",
        "model_revision": "revision-v2",
        "query_analyzer": FakeExactAnalyzer(),
        "transport": RecordingTransport(response),
    }
    kwargs.update(overrides)
    return QueryPlanGeneratorV2(**kwargs)


def test_v2_schema_is_strict_minimal_and_uses_bounded_integer_ranges():
    schema = query_plan_v2_json_schema(topic_id="boundary", token_count=9)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["schema_version"] == {
        "type": "string",
        "const": V2_SCHEMA_VERSION,
    }
    assert schema["properties"]["topic_id"] == {
        "type": "string",
        "const": "boundary",
    }

    for node in _walk_schema(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])

    properties = _all_property_names(schema)
    assert properties.isdisjoint(
        {
            "information_need",
            "priority",
            "facet_count",
            "complexity_class",
            "depends_on",
            "source_span",
            "source_spans",
            "text",
            "label",
            "provenance",
        }
    )

    root = schema["properties"]
    assert set(root) == {
        "schema_version",
        "topic_id",
        "anchors",
        "coverage_items",
        "facets",
        "global_expansion",
    }
    anchor = root["anchors"]["items"]
    coverage = root["coverage_items"]["items"]
    facet = root["facets"]["items"]
    expansion = facet["properties"]["expansion_terms"]["items"]
    assert set(anchor["properties"]) == {
        "anchor_id",
        "range",
        "kind",
        "scope",
        "coverage_refs",
    }
    assert set(coverage["properties"]) == {"coverage_id", "source_span_refs"}
    assert set(facet["properties"]) == {
        "facet_id",
        "coverage_refs",
        "expansion_terms",
    }
    assert set(expansion["properties"]) == {"term", "relation", "anchor_refs"}

    token_range = anchor["properties"]["range"]
    assert token_range == coverage["properties"]["source_span_refs"]["items"]
    assert token_range["properties"]["start_token"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 8,
    }
    assert token_range["properties"]["end_token"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 9,
    }
    assert root["coverage_items"]["maxItems"] == 16
    assert coverage["properties"]["source_span_refs"]["maxItems"] == 2
    assert root["facets"]["maxItems"] == 8
    assert facet["properties"]["coverage_refs"]["maxItems"] == 2
    assert facet["properties"]["expansion_terms"]["maxItems"] == 3
    assert root["global_expansion"]["properties"]["terms"]["maxItems"] == 8

    with pytest.raises(ValueError, match="token_count"):
        query_plan_v2_json_schema(topic_id="boundary", token_count=0)


def test_v2_prompt_freezes_atomicity_scope_lexical_safety_and_exact_schema():
    schema = query_plan_v2_json_schema(topic_id="boundary", token_count=9)
    prompt = developer_prompt_v2(schema)
    lowered = prompt.casefold()

    assert V2_DEVELOPER_PROMPT in prompt
    assert "one atomic coverage item" in lowered
    assert "coordinated" in lowered
    assert "indivisible comparison" in lowered or "indivisible relationship" in lowered
    assert "scope=\"global\"" in lowered or "scope=global" in lowered
    assert "scope=\"coverage\"" in lowered or "scope=coverage" in lowered
    assert "coverage_refs" in prompt
    assert "do not answer" in lowered
    assert "candidate answer" in lowered
    assert "new entity" in lowered or "new referent" in lowered
    assert "integer token positions" in lowered
    assert "start_token" in prompt and "end_token" in prompt
    assert "never copy" in lowered or "do not copy" in lowered
    assert "narrative" in lowered and "data" in lowered
    assert json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) in prompt


def test_v2_request_contains_exact_narrative_token_tape_and_constrained_schema():
    topic = _topic()
    generator = _generator(
        _response(json.dumps(_valid_plan())),
        instruction_role="system",
        reasoning_effort=None,
    )

    request = generator.request_payload(topic)
    tape = tokenize_narrative(topic.narrative)
    schema = query_plan_v2_json_schema(
        topic_id=topic.id,
        token_count=tape.token_count,
    )

    assert request["model"] == "planner-v2-local"
    assert request["max_tokens"] == 6400
    assert "reasoning_effort" not in request
    assert request["messages"][0] == {
        "role": "system",
        "content": developer_prompt_v2(schema),
    }
    user_content = request["messages"][1]["content"]
    assert f"<topic_id>{topic.id}</topic_id>" in user_content
    narrative = user_content.split("<narrative>\n", 1)[1].split(
        "\n</narrative>", 1
    )[0]
    tape_json = user_content.split("<token_tape>\n", 1)[1].split(
        "\n</token_tape>", 1
    )[0]
    assert narrative == topic.narrative
    assert json.loads(tape_json) == tape.to_dict()
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": V2_SCHEMA_VERSION,
            "strict": True,
            "schema": schema,
        },
    }

    reasoning_request = _generator(
        _response(json.dumps(_valid_plan())),
        instruction_role="developer",
        reasoning_effort="low",
        max_tokens=3200,
    ).request_payload(topic)
    assert reasoning_request["messages"][0]["role"] == "developer"
    assert reasoning_request["reasoning_effort"] == "low"
    assert reasoning_request["max_tokens"] == 3200

    with pytest.raises(ValueError, match="6400"):
        _generator(_response("{}"), max_tokens=6401)
    with pytest.raises(ValueError, match="instruction_role"):
        _generator(_response("{}"), instruction_role="assistant")


def test_raw_response_hook_runs_before_assistant_json_parsing(monkeypatch):
    response = _response("not JSON")
    generator = _generator(response)
    events: list[str] = []
    observed: list[HttpJsonResponse] = []
    real_loads = json.loads

    def recording_loads(value, *args, **kwargs):
        events.append("parse")
        return real_loads(value, *args, **kwargs)

    def response_hook(received, elapsed):
        events.append("hook")
        assert elapsed >= 0
        observed.append(received)

    monkeypatch.setattr(query_planner.json, "loads", recording_loads)
    result = generator.generate(_topic(), response_hook=response_hook)

    assert events[:2] == ["hook", "parse"]
    assert observed == [response]
    assert observed[0].raw_body == response.raw_body
    assert observed[0].response_headers["X-Request-ID"] == "wire-v2"
    assert result.outcome.used_fallback is True
    assert result.outcome.failure.status == "invalid_json"


def test_v2_generator_uses_transport_raw_hook_once_before_assistant_json_parsing(
    monkeypatch,
):
    response = _response(json.dumps(_valid_plan()))
    events: list[str] = []

    class RawHookTransport:
        def post_json(self, *_args, **_kwargs):
            raise AssertionError("generator must use post_json_with_raw_hook")

        def post_json_with_raw_hook(
            self, _url, _payload, _headers, _timeout, raw_response_hook
        ):
            events.append("transport")
            raw_response_hook(response)
            events.append("transport_return")
            return response

    real_loads = json.loads

    def recording_loads(value, *args, **kwargs):
        events.append("assistant_json_parse")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(query_planner.json, "loads", recording_loads)
    observed: list[HttpJsonResponse] = []

    def response_hook(received, elapsed):
        events.append("response_hook")
        assert elapsed >= 0
        observed.append(received)

    result = _generator(response, transport=RawHookTransport()).generate(
        _topic(), response_hook=response_hook
    )

    assert result.outcome.used_fallback is False
    assert events[:4] == [
        "transport",
        "response_hook",
        "transport_return",
        "assistant_json_parse",
    ]
    assert events.count("response_hook") == 1
    assert observed == [response]


def test_v2_generator_preserves_invalid_transport_bytes_once_via_raw_hook():
    raw_body = b'\xff{"choices":'
    raw_response = HttpJsonResponse(
        {},
        raw_body=raw_body,
        http_status=200,
        response_headers={
            "Content-Type": "application/json",
            "X-Request-ID": "invalid-wire-v2",
        },
    )

    class InvalidRawHookTransport:
        def post_json(self, *_args, **_kwargs):
            raise AssertionError("generator must use post_json_with_raw_hook")

        def post_json_with_raw_hook(
            self, _url, _payload, _headers, _timeout, raw_response_hook
        ):
            raw_response_hook(raw_response)
            raise QueryPlanHttpResponseError(
                "response was not valid UTF-8 JSON",
                raw_body=raw_body,
                http_status=200,
                response_headers=raw_response.response_headers,
            )

    observed: list[HttpJsonResponse] = []
    result = _generator(response={}, transport=InvalidRawHookTransport()).generate(
        _topic(),
        response_hook=lambda received, _elapsed: observed.append(received),
    )

    assert len(observed) == 1
    assert observed[0] is raw_response
    assert observed[0].raw_body == raw_body
    assert observed[0].response_headers["X-Request-ID"] == "invalid-wire-v2"
    assert result.outcome.used_fallback is True
    assert result.outcome.failure.status == "invalid_json"


def test_v2_generator_raw_hook_persistence_failure_aborts_before_parsing(
    monkeypatch,
):
    class PersistenceFailure(RuntimeError):
        pass

    response = _response(json.dumps(_valid_plan()))

    class RawHookTransport:
        def post_json(self, *_args, **_kwargs):
            raise AssertionError("generator must use post_json_with_raw_hook")

        def post_json_with_raw_hook(
            self, _url, _payload, _headers, _timeout, raw_response_hook
        ):
            raw_response_hook(response)
            raise AssertionError("transport must stop when persistence fails")

    parse_calls = []

    def forbidden_json_loads(*_args, **_kwargs):
        parse_calls.append(True)
        raise AssertionError("assistant JSON must not be parsed after persistence fails")

    monkeypatch.setattr(query_planner.json, "loads", forbidden_json_loads)

    def failing_hook(_received, _elapsed):
        raise PersistenceFailure("could not persist raw response")

    with pytest.raises(PersistenceFailure, match="could not persist"):
        _generator(response, transport=RawHookTransport()).generate(
            _topic(), response_hook=failing_hook
        )

    assert parse_calls == []


def test_malformed_http_json_is_hooked_and_falls_back_with_auditable_status():
    raw_body = b'{"choices":[{"message":'

    class InvalidHttpJsonTransport:
        def post_json(self, _url, _payload, _headers, _timeout):
            raise QueryPlanHttpResponseError(
                "response was not JSON",
                raw_body=raw_body,
                http_status=200,
                response_headers={"Content-Type": "application/json", "X-Request-ID": "bad-v2"},
            )

    generator = _generator(_response("{}"), transport=InvalidHttpJsonTransport())
    observed: list[HttpJsonResponse] = []

    result = generator.generate(
        _topic(),
        response_hook=lambda response, _elapsed: observed.append(response),
    )

    assert len(observed) == 1
    assert observed[0].raw_body == raw_body
    assert observed[0].http_status == 200
    assert observed[0].response_headers["X-Request-ID"] == "bad-v2"
    assert result.outcome.used_fallback is True
    assert result.outcome.failure.status == "invalid_json"
    assert result.outcome.rendered_queries[0].query_text == _topic().narrative


def test_v2_provenance_freezes_tokenizer_narrative_analyzer_schema_and_prompt():
    topic = _topic()
    response = _response(json.dumps(_valid_plan()), response_id="chatcmpl-provenance")
    generator = _generator(
        response,
        instruction_role="developer",
        reasoning_effort=None,
    )

    result = generator.generate(topic)

    assert result.outcome.used_fallback is False
    tape = tokenize_narrative(topic.narrative)
    schema = query_plan_v2_json_schema(topic_id=topic.id, token_count=tape.token_count)
    prompt = developer_prompt_v2(schema)
    provenance = result.provenance
    assert result.token_tape == tape
    assert provenance.schema_version == V2_SCHEMA_VERSION
    assert provenance.prompt_version == V2_PROMPT_VERSION
    assert provenance.tokenizer_version == TOKENIZER_VERSION
    assert provenance.token_count == tape.token_count
    assert provenance.narrative_sha256 == tape.narrative_sha256
    assert provenance.analyzer_fingerprint == FINGERPRINT.to_dict()
    assert provenance.schema_sha256 == hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert provenance.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert provenance.instruction_role == "developer"
    assert provenance.reasoning_effort is None
    assert provenance.max_tokens == 6400
    assert provenance.response_id == "chatcmpl-provenance"
    assert provenance.response_model == "served-v2-model"
    assert provenance.finish_reason == "stop"


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("invalid_envelope", "invalid_json"),
        ("invalid_json", "invalid_json"),
        ("invalid_plan", "plan_validation_error"),
        ("invalid_render", "render_validation_error"),
    ],
)
def test_invalid_model_outputs_keep_auditable_status_and_original_only_fallback(
    case,
    expected_status,
):
    if case == "invalid_envelope":
        response = HttpJsonResponse(
            {"id": "chatcmpl-invalid", "model": "served-v2-model", "choices": []},
            raw_body=b'{"id":"chatcmpl-invalid","choices":[]}',
            http_status=200,
            response_headers={"Content-Type": "application/json"},
        )
        expected_content = None
    elif case == "invalid_json":
        expected_content = "not JSON"
        response = _response(expected_content, response_id="chatcmpl-invalid")
    else:
        payload = deepcopy(_valid_plan())
        if case == "invalid_plan":
            payload["facets"][0]["coverage_refs"] = ["missing_coverage"]
        else:
            payload["facets"][0]["expansion_terms"] = []
        expected_content = json.dumps(payload)
        response = _response(expected_content, response_id="chatcmpl-invalid")

    observed: list[HttpJsonResponse] = []
    result = _generator(response).generate(
        _topic(),
        response_hook=lambda received, _elapsed: observed.append(received),
    )

    outcome = result.outcome
    assert outcome.used_fallback is True
    assert outcome.plan is None
    assert outcome.failure.status == expected_status
    assert outcome.failure.error
    assert result.provenance.response_id == "chatcmpl-invalid"
    assert observed == [response]
    assert observed[0].raw_body == response.raw_body
    if expected_content is not None:
        wire = json.loads(observed[0].raw_body.decode("utf-8"))
        assert wire["choices"][0]["message"]["content"] == expected_content
    assert len(outcome.rendered_queries) == 1
    (fallback,) = outcome.rendered_queries
    assert fallback.variant_name == "original"
    assert fallback.source_type == "original"
    assert fallback.query_text == _topic().narrative
    assert fallback.components == (_topic().narrative,)


def test_non_stop_finish_reason_is_a_mechanical_failure_even_if_json_is_valid():
    response = _response(
        json.dumps(_valid_plan()),
        response_id="chatcmpl-truncated",
        finish_reason="length",
    )

    result = _generator(response).generate(_topic())

    assert result.outcome.used_fallback is True
    assert result.outcome.plan is None
    assert result.outcome.failure.status == "invalid_json"
    assert "finish_reason='length'" in result.outcome.failure.error
    assert result.provenance.finish_reason == "length"
    assert result.outcome.rendered_queries[0].query_text == _topic().narrative


def test_analyzer_preflight_failure_happens_before_the_model_request():
    events: list[str] = []

    class FailingAnalyzer:
        @property
        def fingerprint(self):
            events.append("fingerprint")
            return FINGERPRINT

        def analyze(self, _text):
            events.append("analyze")
            raise RuntimeError("reference analyzer unavailable")

    class MustNotCallTransport:
        def post_json(self, *_args, **_kwargs):
            events.append("model")
            raise AssertionError("model request must follow analyzer preflight")

    generator = QueryPlanGeneratorV2(
        base_url="http://planner.test/v1",
        model="planner-v2-local",
        query_analyzer=FailingAnalyzer(),
        transport=MustNotCallTransport(),
    )

    with pytest.raises(RuntimeError, match="reference analyzer unavailable"):
        generator.generate(_topic())

    assert events == ["fingerprint", "analyze"]

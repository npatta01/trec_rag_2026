"""Public contract for fail-closed facet planning and hosted admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import pytest

from trec_rag.facet_extraction import (
    BackendReply,
    FacetPlanningResult,
    FacetResponse,
    GeneratedQueryPlan,
    OpenAICompatibleFacetBackend,
    OpenRouterDeepSeekFacetBackend,
    Subnarrative,
    extract_facets,
    plan_facet_queries,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


def _topic(*, title: str = "Private organizer title") -> Topic:
    return Topic(
        id="housing-1",
        title=title,
        narrative="Housing tenants compare rent increases and zoning changes.",
    )


def _plan() -> dict[str, object]:
    return {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": "housing-1",
        "subnarratives": [
            {
                "subnarrative": "How rent increases affect housing tenants",
                "bm25_queries": [
                    "rent increases housing tenant impacts",
                    "rising rents tenant affordability",
                ],
            },
            {
                "subnarrative": "How zoning changes affect housing tenants",
                "bm25_queries": ["zoning changes housing tenant impacts"],
            },
        ],
    }


def _envelope(plan: object | None = None) -> dict[str, object]:
    return {
        "id": "response-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-v4-flash-20260423",
        "provider": "DeepInfra",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(_plan() if plan is None else plan)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _StaticBackend:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def extract(self, topic: Topic) -> object:
        assert topic.id == "housing-1"
        return self.payload


class _Transport:
    def __init__(self, response: FacetResponse) -> None:
        self.response = response
        self.requests: list[object] = []

    def send(self, request: object) -> FacetResponse:
        self.requests.append(request)
        return self.response


def _original(topic: Topic) -> tuple[QueryVariant, ...]:
    return (QueryVariant(topic.id, "original", topic.narrative, "original_topic"),)


def _all_values(value: object):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _all_values(key)
            yield from _all_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_values(item)
    else:
        yield value


def test_narrative_only_request_keeps_title_outside_hosted_payload() -> None:
    transport = _Transport(FacetResponse(200, json.dumps(_envelope()).encode()))

    result = extract_facets(
        _topic(),
        OpenRouterDeepSeekFacetBackend(
            environ={"OPENROUTER_API_KEY": "test-only-key"}, transport=transport
        ),
    )

    assert result.used_fallback is False
    request_body = transport.requests[0].body
    assert _topic().narrative.encode() in request_body
    assert _topic().title.encode() not in request_body
    payload = json.loads(request_body)
    assert payload["model"] == "deepseek/deepseek-v4-flash-20260423"
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == "subnarrative_queries_v1"
    assert schema["strict"] is True
    assert schema["schema"]["type"] == "object"
    assert set(schema["schema"]["required"]) == {"schema_version", "topic_id", "subnarratives"}


def test_planning_types_render_original_then_ordered_canonical_lanes() -> None:
    result = plan_facet_queries(_topic(), _plan())

    assert isinstance(result, FacetPlanningResult)
    assert isinstance(result.plan, GeneratedQueryPlan)
    assert isinstance(result.subnarratives[0], Subnarrative)
    assert result.queries == (
        *_original(_topic()),
        QueryVariant("housing-1", "facet:subnarrative-1:q1", "rent increases housing tenant impacts", "generated_subnarrative_bm25"),
        QueryVariant("housing-1", "facet:subnarrative-1:q2", "rising rents tenant affordability", "generated_subnarrative_bm25"),
        QueryVariant("housing-1", "facet:subnarrative-2:q1", "zoning changes housing tenant impacts", "generated_subnarrative_bm25"),
    )
    assert BackendReply(b"content", b"raw", 200, {}) == BackendReply(b"content", b"raw", 200, {})


def test_title_changes_do_not_change_hosted_request_or_result() -> None:
    first_transport = _Transport(FacetResponse(200, json.dumps(_envelope()).encode()))
    second_transport = _Transport(FacetResponse(200, json.dumps(_envelope()).encode()))
    first = extract_facets(
        _topic(title="First private title"),
        OpenRouterDeepSeekFacetBackend(environ={"OPENROUTER_API_KEY": "test-only-key"}, transport=first_transport),
    )
    second = extract_facets(
        _topic(title="Second private title"),
        OpenRouterDeepSeekFacetBackend(environ={"OPENROUTER_API_KEY": "test-only-key"}, transport=second_transport),
    )

    assert first == second
    assert first_transport.requests[0].body == second_transport.requests[0].body


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("not-json", "generated query plan must be an object"),
        ({"schema_version": "wrong"}, "generated query plan has unexpected or missing fields"),
        ({"schema_version": "subnarrative_queries_v1", "topic_id": "housing-1", "subnarratives": []}, "subnarratives must contain between 1 and 8 items"),
    ],
)
def test_malformed_schema_and_semantic_plan_failures_return_exact_original(
    payload: object, expected_error: str
) -> None:
    result = extract_facets(_topic(), _StaticBackend(payload))

    assert result.used_fallback is True
    assert result.plan is None
    assert result.subnarratives == ()
    assert result.queries == _original(_topic())
    assert result.error == expected_error


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        (b'{"id":"one","id":"two"}', "strict JSON"),
        (b'{"id":NaN}', "strict JSON"),
        (json.dumps(_envelope()).replace('"created": 1', '"created": "one"').encode(), "created must be an integer"),
    ],
)
def test_hosted_decoder_rejects_duplicate_keys_nonfinite_and_wrong_types(
    body: bytes, expected_error: str
) -> None:
    result = extract_facets(
        _topic(),
        OpenRouterDeepSeekFacetBackend(
            environ={"OPENROUTER_API_KEY": "test-only-key"},
            transport=_Transport(FacetResponse(200, body)),
        ),
    )

    assert result.used_fallback is True
    assert result.queries == _original(_topic())
    assert expected_error in (result.error or "")


@pytest.mark.parametrize("base_url", ["http://model.example/v1", "http://192.0.2.1/v1"])
def test_credentialed_nonloopback_http_endpoint_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError, match="credentials require HTTPS"):
        OpenAICompatibleFacetBackend(base_url=base_url, model="facet-model", api_key="test-only-key")


@pytest.mark.parametrize(
    "response_body",
    [
        json.dumps(_envelope()).replace('"DeepInfra"', '"echoed-test-only-key"').encode(),
        json.dumps(_envelope()).replace('"DeepInfra"', '"echoed-test-only-\\u006bey"').encode(),
    ],
)
def test_openrouter_receipts_suppress_literal_and_unicode_escaped_credentials(
    response_body: bytes,
) -> None:
    secret = "test-only-key"
    receipts = []
    result = extract_facets(
        _topic(),
        OpenRouterDeepSeekFacetBackend(
            environ={"OPENROUTER_API_KEY": secret},
            transport=_Transport(FacetResponse(200, response_body)),
            receipt_hook=receipts.append,
        ),
    )

    assert result.used_fallback is False
    assert receipts[0].response_body is None
    assert all(
        secret not in value if isinstance(value, str) else secret.encode() not in value
        for value in _all_values(asdict(receipts[0]))
        if isinstance(value, (str, bytes))
    )


@pytest.mark.parametrize(
    ("response_body", "retained"),
    [
        (b'{"error":"test-only-\\u006bey"', False),
        (b'{"error":"ordinary malformed response"}', True),
    ],
    ids=("escaped-credential", "safe-malformed"),
)
def test_openrouter_malformed_receipt_preserves_digest_and_only_retains_safe_body(
    response_body: bytes, retained: bool
) -> None:
    secret = "test-only-key"
    receipts = []

    result = extract_facets(
        _topic(),
        OpenRouterDeepSeekFacetBackend(
            environ={"OPENROUTER_API_KEY": secret},
            transport=_Transport(FacetResponse(200, response_body)),
            receipt_hook=receipts.append,
        ),
    )

    assert result.used_fallback is True
    assert receipts[0].response_body == (response_body if retained else None)
    assert receipts[0].response_sha256 == sha256(response_body).hexdigest()
    assert secret not in repr(receipts[0])

from __future__ import annotations

import json

from trec_rag.tracing.openai_semantics import (
    openai_llm_attributes,
    openai_request_envelope,
    openai_response_envelope,
)
from trec_rag.experiments.organizer_pi.event_trace import piika_tool_schemas
from trec_rag.tracing.models import SpanSpec


def _llm_span(*, input_value, output_value):
    return SpanSpec(
        name="Pi assistant turn",
        kind="LLM",
        start_ns=1,
        end_ns=2,
        attributes={},
        input_value=input_value,
        output_value=output_value,
        status="OK",
    )


def _assistant_payload(*, content=None, **extra):
    return {
        "role": "assistant",
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "responseId": "resp-1",
        "stopReason": "toolUse",
        "usage": {"input": 12, "output": 7, "totalTokens": 19},
        "content": content if content is not None else [],
        **extra,
    }


def _tool_turn_spec():
    return _llm_span(
        input_value={
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "search",
                            "arguments": {
                                "reason": "find evidence",
                                "query": "evidence",
                                "hits": 8,
                            },
                        }
                    ],
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "content": [{"type": "text", "text": "ranked results"}],
                },
            ],
            "tools": list(piika_tool_schemas()),
        },
        output_value=_assistant_payload(
            content=[
                {
                    "type": "toolCall",
                    "id": "call-2",
                    "name": "read_document",
                    "arguments": {"reason": "verify", "docid": "d1"},
                }
            ]
        ),
    )


def test_pi_tool_turn_matches_openai_message_and_tool_result_contract():
    attrs = openai_llm_attributes(_tool_turn_spec())

    assert attrs["llm.system"] == "openai"
    assert attrs["llm.provider"] == "openai"
    assert attrs["llm.input_messages.1.message.tool_calls.0.tool_call.id"] == "call-1"
    assert attrs["llm.input_messages.2.message.role"] == "tool"
    assert attrs["llm.input_messages.2.message.tool_call_id"] == "call-1"
    assert attrs["llm.input_messages.2.message.content"] == "ranked results"
    assert json.loads(attrs["llm.tools.0.tool.json_schema"])["function"]["name"] == "search"
    assert (
        json.loads(attrs["llm.tools.1.tool.json_schema"])["function"]["name"]
        == "read_document"
    )


def test_parallel_tool_calls_and_ordered_reasoning_are_preserved():
    spec = _llm_span(
        input_value={"messages": [{"role": "user", "content": "question"}]},
        output_value=_assistant_payload(
            content=[
                {"type": "thinking", "thinking": "first reason"},
                {
                    "type": "toolCall",
                    "id": "call-a",
                    "name": "search",
                    "arguments": {"query": "a", "reason": "first"},
                },
                {"type": "thinking", "thinking": "second reason"},
                {
                    "type": "toolCall",
                    "id": "call-b",
                    "name": "search",
                    "arguments": {"query": "b", "reason": "second"},
                },
            ]
        ),
    )

    attrs = openai_llm_attributes(spec)
    assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call-a"
    assert attrs["llm.output_messages.0.message.tool_calls.1.tool_call.id"] == "call-b"
    assert attrs[
        "llm.output_messages.0.message.contents.0.message_content.text"
    ] == "first reason"
    assert attrs[
        "llm.output_messages.0.message.contents.1.message_content.text"
    ] == "second reason"


def test_extracted_reasoning_is_omitted_only_from_llm_presentation():
    spec = SpanSpec(
        name="Pi generation",
        kind="LLM",
        start_ns=1,
        end_ns=2,
        attributes={"pi.reasoning.extracted_to_children": True},
        input_value={"system_prompt": "system", "user_prompt": "question"},
        output_value=_assistant_payload(
            content=[
                {"type": "thinking", "thinking": "first native block"},
                {"type": "reasoning", "text": "second native block"},
                {"type": "text", "text": "final answer"},
            ]
        ),
        status="OK",
    )

    attrs = openai_llm_attributes(spec)
    assert attrs["llm.output_messages.0.message.content"] == "final answer"
    assert not any(
        "message_content.type" in key and value == "reasoning"
        for key, value in attrs.items()
    )
    response = openai_response_envelope(spec)
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "final answer",
    }
    assert spec.output_value["content"] == [
        {"type": "thinking", "thinking": "first native block"},
        {"type": "reasoning", "text": "second native block"},
        {"type": "text", "text": "final answer"},
    ]


def test_scalar_tool_result_content_is_preserved_as_scalar_openai_content():
    spec = _llm_span(
        input_value={
            "messages": [
                {"role": "toolResult", "toolCallId": "call-1", "content": "raw result"}
            ]
        },
        output_value=_assistant_payload(content=[{"type": "text", "text": "answer"}]),
    )

    assert openai_request_envelope(spec)["messages"] == [
        {"role": "tool", "tool_call_id": "call-1", "content": "raw result"}
    ]


def test_provider_identity_and_known_invocation_parameters_preserve_provenance():
    spec = _llm_span(
        input_value={
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "unrecognized": "do not export",
        },
        output_value=_assistant_payload(),
    )

    attrs = openai_llm_attributes(spec)
    assert attrs["llm.system"] == "openai"
    assert attrs["llm.provider"] == "openai"
    assert attrs["pi.original.llm.provider"] == "openai-codex"
    assert json.loads(attrs["llm.invocation_parameters"]) == {
        "temperature": 0.2,
        "top_p": 0.9,
    }


def test_pi_request_has_no_fabricated_system_message():
    request = openai_request_envelope(_tool_turn_spec())
    assert [message["role"] for message in request["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


def test_openai_request_and_response_envelopes_match_chat_completion_shapes():
    spec = _tool_turn_spec()

    assert openai_request_envelope(spec) == {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"hits":8,"query":"evidence","reason":"find evidence"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ranked results"},
        ],
        "tools": list(piika_tool_schemas()),
    }
    response = openai_response_envelope(spec)
    assert response["id"] == "resp-1"
    assert response["object"] == "chat.completion"
    assert response["model"] == "gpt-5.6-sol"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["finish_reason"] == "toolUse"


def test_fixed_request_keeps_exact_prompts_and_has_no_tools():
    spec = _llm_span(
        input_value={"system_prompt": "exact system\n", "user_prompt": "exact user\n"},
        output_value=_assistant_payload(content=[{"type": "text", "text": "answer"}]),
    )

    assert openai_request_envelope(spec) == {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "exact system\n"},
            {"role": "user", "content": "exact user\n"},
        ],
    }


def test_response_envelope_preserves_supported_pi_token_usage_aliases():
    spec = _llm_span(
        input_value={"messages": [{"role": "user", "content": "question"}]},
        output_value=_assistant_payload(
            usage={"inputTokens": 12, "outputTokens": 7, "totalTokens": 19}
        ),
    )

    assert openai_response_envelope(spec)["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }

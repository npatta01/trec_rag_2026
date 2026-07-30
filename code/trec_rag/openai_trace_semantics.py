"""Translate immutable Pi LLM payloads into OpenAI/OpenInference semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json

from openinference.semconv.trace import SpanAttributes

from trec_rag.pi_trace_models import SpanSpec


def _json_string(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def piika_tool_schemas() -> tuple[Mapping[str, object], ...]:
    """Return the two paginated pyserini-rest tools advertised in the capture."""
    return (
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the configured backend and return ranked hits directly. "
                    "The first argument must be reason, a brief rationale of at most 100 words."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Brief rationale for this search, maximum 100 words. Put the specific clue "
                                "or follow-up goal first."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": "Raw query string. Use concise lexical clues instead of long natural-language rewrites.",
                        },
                        "hits": {
                            "type": "number",
                            "description": "Maximum number of hits to return directly in this search call. Defaults to 5.",
                        },
                    },
                    "required": ["reason", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_document",
                "description": (
                    "Read a retrieved document by docid. Supports offset and limit for paginated "
                    "line-based reading, similar to the built-in read tool. The first argument must "
                    "be reason, a brief rationale of at most 100 words."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Brief rationale for opening this document, maximum 100 words. State the "
                                "candidate clue or fact you expect to verify in this doc."
                            ),
                        },
                        "docid": {
                            "type": "string",
                            "description": "Document id to retrieve",
                        },
                        "offset": {
                            "type": "number",
                            "description": "Line number to start reading from (1-indexed).",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of lines to read.",
                        },
                    },
                    "required": ["reason", "docid"],
                },
            },
        },
    )


def _text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, Sequence) or isinstance(content, bytes | bytearray):
        return []
    return [
        item["text"]
        for item in content
        if isinstance(item, Mapping)
        and item.get("type") in {"text", "input_text"}
        and isinstance(item.get("text"), str)
    ]


def _tool_calls(content: object) -> list[dict[str, object]]:
    if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
        return []
    calls: list[dict[str, object]] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") not in {"toolCall", "tool_call"}:
            continue
        call_id = item.get("id", item.get("toolCallId"))
        name = item.get("name", item.get("toolName"))
        arguments = item.get("arguments", item.get("args"))
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _json_string(arguments if arguments is not None else {}),
                },
            }
        )
    return calls


def normalize_openai_message(message: Mapping[str, object]) -> dict[str, object]:
    """Map one captured Pi message to its OpenAI chat-completions counterpart."""
    role = message.get("role")
    normalized: dict[str, object] = {"role": "tool" if role == "toolResult" else role}
    tool_call_id = message.get("toolCallId", message.get("tool_call_id"))
    if normalized["role"] == "tool" and isinstance(tool_call_id, str):
        normalized["tool_call_id"] = tool_call_id
    text = _text_parts(message.get("content"))
    if text:
        normalized["content"] = "\n".join(text)
    elif isinstance(message.get("content"), str):
        normalized["content"] = message["content"]
    calls = _tool_calls(message.get("content"))
    if calls:
        normalized["tool_calls"] = calls
    return {key: value for key, value in normalized.items() if isinstance(value, str) or value}


def _messages(value: object | None, *, input_side: bool) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return []
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes | bytearray):
        return [message for message in messages if isinstance(message, Mapping)]
    if isinstance(value.get("role"), str):
        return [value]
    if input_side and isinstance(value.get("system_prompt"), str) and isinstance(value.get("user_prompt"), str):
        return [
            {"role": "system", "content": value["system_prompt"]},
            {"role": "user", "content": value["user_prompt"]},
        ]
    return []


def _model(spec: SpanSpec) -> str | None:
    if isinstance(spec.output_value, Mapping) and isinstance(spec.output_value.get("model"), str):
        return spec.output_value["model"]
    return None


def _presented_output(spec: SpanSpec) -> Mapping[str, object]:
    """Return the LLM output used for OpenAI presentation without mutation."""
    if not isinstance(spec.output_value, Mapping):
        return {}
    output = dict(spec.output_value)
    if spec.attributes.get("pi.reasoning.extracted_to_children") is not True:
        return output
    content = output.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
        return output
    output["content"] = [
        item
        for item in content
        if not isinstance(item, Mapping)
        or item.get("type") not in {"thinking", "reasoning"}
    ]
    return output


def openai_request_envelope(spec: SpanSpec) -> Mapping[str, object]:
    request: dict[str, object] = {}
    model = _model(spec)
    if model is not None:
        request["model"] = model
    request["messages"] = [
        normalize_openai_message(message) for message in _messages(spec.input_value, input_side=True)
    ]
    if (
        isinstance(spec.input_value, Mapping)
        and isinstance(spec.input_value.get("tools"), Sequence)
        and not isinstance(spec.input_value.get("tools"), str | bytes | bytearray)
    ):
        request["tools"] = list(spec.input_value["tools"])
    return request


def openai_response_envelope(spec: SpanSpec) -> Mapping[str, object]:
    output = _presented_output(spec)
    message = normalize_openai_message(output) if output else {"role": "assistant"}
    response: dict[str, object] = {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": output.get("stopReason") if isinstance(output.get("stopReason"), str) else None,
            }
        ],
    }
    for source, target in (("responseId", "id"), ("model", "model")):
        value = output.get(source)
        if isinstance(value, str):
            response[target] = value
    usage = output.get("usage")
    if isinstance(usage, Mapping):
        normalized_usage: dict[str, int] = {}
        for source, aliases, target in (
            ("input", ("inputTokens", "prompt_tokens"), "prompt_tokens"),
            ("output", ("outputTokens", "completion_tokens"), "completion_tokens"),
            ("totalTokens", ("total_tokens",), "total_tokens"),
        ):
            value = usage.get(source)
            if value is None:
                value = next((usage[alias] for alias in aliases if alias in usage), None)
            if isinstance(value, int) and not isinstance(value, bool):
                normalized_usage[target] = value
        if normalized_usage:
            response["usage"] = normalized_usage
    return response


def _message_attributes(prefix: str, messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for message_index, message in enumerate(messages):
        base = f"{prefix}.{message_index}.message"
        normalized = normalize_openai_message(message)
        role = normalized.get("role")
        if isinstance(role, str):
            attributes[f"{base}.role"] = role
        content = normalized.get("content")
        if isinstance(content, str):
            attributes[f"{base}.content"] = content
        tool_call_id = normalized.get("tool_call_id")
        if isinstance(tool_call_id, str):
            attributes[f"{base}.tool_call_id"] = tool_call_id
        raw_content = message.get("content")
        if isinstance(raw_content, Sequence) and not isinstance(raw_content, str | bytes | bytearray):
            content_index = 0
            for item in raw_content:
                if not isinstance(item, Mapping) or item.get("type") not in {"thinking", "reasoning", "text", "input_text"}:
                    continue
                if item.get("type") in {"thinking", "reasoning"}:
                    text = item.get("thinking", item.get("text"))
                    content_type = "reasoning"
                else:
                    text = item.get("text")
                    content_type = "text"
                if isinstance(text, str):
                    content_base = f"{base}.contents.{content_index}.message_content"
                    attributes[f"{content_base}.type"] = content_type
                    attributes[f"{content_base}.text"] = text
                    content_index += 1
        calls = normalized.get("tool_calls")
        if isinstance(calls, list):
            for tool_index, call in enumerate(calls):
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    continue
                tool_base = f"{base}.tool_calls.{tool_index}.tool_call"
                if isinstance(call.get("id"), str):
                    attributes[f"{tool_base}.id"] = call["id"]
                function = call["function"]
                if isinstance(function.get("name"), str):
                    attributes[f"{tool_base}.function.name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    attributes[f"{tool_base}.function.arguments"] = function["arguments"]
    return attributes


_KNOWN_INVOCATION_PARAMETERS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    }
)


def openai_llm_attributes(spec: SpanSpec) -> Mapping[str, object]:
    """Return OpenInference attributes that match OpenAI chat instrumentation."""
    attributes: dict[str, object] = {
        SpanAttributes.LLM_SYSTEM: "openai",
        SpanAttributes.LLM_PROVIDER: "openai",
    }
    input_messages = _messages(spec.input_value, input_side=True)
    output = _presented_output(spec)
    output_messages = _messages(output, input_side=False)
    attributes.update(_message_attributes(SpanAttributes.LLM_INPUT_MESSAGES, input_messages))
    attributes.update(_message_attributes(SpanAttributes.LLM_OUTPUT_MESSAGES, output_messages))
    if isinstance(spec.input_value, Mapping):
        tools = spec.input_value.get("tools")
        if isinstance(tools, Sequence) and not isinstance(tools, str | bytes | bytearray):
            for index, tool in enumerate(tools):
                if isinstance(tool, Mapping):
                    attributes[f"{SpanAttributes.LLM_TOOLS}.{index}.tool.json_schema"] = _json_string(tool)
        invocation = {
            key: value
            for key, value in spec.input_value.items()
            if key in _KNOWN_INVOCATION_PARAMETERS
        }
        if invocation:
            attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = _json_string(invocation)
    model = output.get("model")
    if isinstance(model, str):
        attributes[SpanAttributes.LLM_MODEL_NAME] = model
    provider = output.get("provider")
    if isinstance(provider, str):
        attributes["pi.original.llm.provider"] = provider
    finish_reason = output.get("stopReason")
    if isinstance(finish_reason, str):
        attributes[SpanAttributes.LLM_FINISH_REASON] = finish_reason
    usage = output.get("usage")
    if not isinstance(usage, Mapping):
        return attributes
    prompt = usage.get("input", usage.get("inputTokens", usage.get("prompt_tokens")))
    completion = usage.get("output", usage.get("outputTokens", usage.get("completion_tokens")))
    total = usage.get("totalTokens")
    if isinstance(prompt, int) and not isinstance(prompt, bool):
        attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt
    if isinstance(completion, int) and not isinstance(completion, bool):
        attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion
    if isinstance(total, int) and not isinstance(total, bool):
        attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = total
    elif isinstance(prompt, int) and isinstance(completion, int):
        attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = prompt + completion
    for source, target in (("cacheRead", SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ), ("cacheWrite", SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)):
        value = usage.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            attributes[target] = value
    costs = usage.get("cost")
    if isinstance(costs, Mapping):
        for source, target in (("input", SpanAttributes.LLM_COST_PROMPT), ("output", SpanAttributes.LLM_COST_COMPLETION), ("total", SpanAttributes.LLM_COST_TOTAL), ("cacheRead", SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_READ), ("cacheWrite", SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_WRITE)):
            value = costs.get(source)
            if isinstance(value, int | float) and not isinstance(value, bool):
                attributes[target] = value
    return attributes

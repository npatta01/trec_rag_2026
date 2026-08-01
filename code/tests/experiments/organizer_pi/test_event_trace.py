from __future__ import annotations

import json
import os

import pytest

from trec_rag.experiments.organizer_pi.inputs import OrganizerTopic
from trec_rag.experiments.organizer_pi.event_trace import (
    build_fixed_trace,
    build_piika_trace,
    load_pi_events,
    piika_tool_schemas,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _assistant(text, timestamp, *, stop_reason="stop"):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "provider": "anthropic",
        "model": "model-id",
        "stopReason": stop_reason,
        "timestamp": timestamp,
    }


def test_piika_tool_schemas_preserve_captured_search_and_document_contract():
    schemas = piika_tool_schemas()

    assert [schema["function"]["name"] for schema in schemas] == [
        "search",
        "read_document",
    ]
    assert schemas[0]["function"]["parameters"]["required"] == ["reason", "query"]
    assert schemas[1]["function"]["parameters"]["required"] == ["reason", "docid"]


def test_piika_events_become_ordered_full_content_spans(tmp_path):
    first = _assistant("I will search and inspect the best result.", 1_800_000_000_000)
    final = _assistant("Final cited answer [1].", 1_800_000_000_500)
    event_path = _write_jsonl(
        tmp_path / "events.jsonl",
        [
            {"type": "message_start", "message": first},
            {"type": "message_end", "message": first},
            {
                "type": "tool_execution_start",
                "toolCallId": "search-1",
                "toolName": "search",
                "args": {
                    "reason": "find relevant evidence",
                    "query": "biological grief heart",
                },
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "search-1",
                "toolName": "search",
                "result": {
                    "content": [{"type": "text", "text": "two complete hits"}],
                    "details": {
                        "hits": [
                            {"docid": "d1", "score": 9.5, "text": "hit one"},
                            {"docid": "d2", "score": 8.5, "text": "hit two"},
                        ]
                    },
                },
                "isError": False,
            },
            {
                "type": "tool_execution_start",
                "toolCallId": "read-1",
                "toolName": "read_document",
                "args": {"reason": "read strongest hit", "docid": "d1"},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "read-1",
                "toolName": "read_document",
                "result": {
                    "content": [{"type": "text", "text": "full document text"}],
                    "details": {"docid": "d1", "text": "full document text"},
                },
                "isError": False,
            },
            {"type": "future_event", "payload": {"kept": "by count"}},
            {"type": "message_start", "message": final},
            {"type": "message_end", "message": final},
        ],
    )

    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(event_path),
        run_record={
            "status": "completed",
            "trec_rag_output": {"references": ["d1"], "answer": []},
        },
        session_id="rag2026-1-comparison",
    )

    assert bundle.project_name == "trec-rag-2026-pi-baselines"
    assert bundle.baseline == "piika-agentic"
    assert bundle.root.kind == "AGENT"
    assert bundle.root.attributes["pi.system_prompt.captured"] is False
    assert [child.kind for child in bundle.root.children] == [
        "LLM",
        "RETRIEVER",
        "TOOL",
        "LLM",
        "CHAIN",
    ]
    assert bundle.root.children[1].input_value["query"] == "biological grief heart"
    assert bundle.root.children[1].output_value["details"]["hits"][0]["text"] == "hit one"
    assert bundle.root.children[2].output_value["text"] == "full document text"
    assert bundle.root.children[3].output_value == final
    assert bundle.root.children[-1].output_value == {
        "status": "completed",
        "trec_rag_output": {"references": ["d1"], "answer": []},
    }
    assert bundle.root.attributes["pi.event.unknown_count"] == 1
    assert bundle.root.attributes["trace.timing_reconstructed"] is True


def test_piika_trace_preserves_prompt_bearing_native_events(tmp_path):
    agent_start = {
        "type": "agent_start",
        "systemPrompt": "full native system prompt",
        "prompt": "full native agent prompt",
    }
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": "full native user prompt"}],
        "timestamp": 1_800_000_000_000,
    }
    assistant = _assistant("answer", 1_800_000_000_100)
    user_start = {"type": "message_start", "message": user_message}
    user_end = {"type": "message_end", "message": user_message}
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "prompt-events.jsonl",
            [
                agent_start,
                user_start,
                user_end,
                {"type": "message_start", "message": assistant},
                {"type": "message_end", "message": assistant},
            ],
        )
    )

    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "official narrative"),
        events=events,
        run_record={"status": "completed"},
        session_id="session",
    )

    assert [child.kind for child in bundle.root.children] == ["CHAIN", "LLM", "CHAIN"]
    prompts = bundle.root.children[0]
    assert prompts.name == "Pi prompt construction"
    assert prompts.attributes["pi.prompt.event_count"] == 3
    assert prompts.input_value == {
        "topic_id": "rag2026-1",
        "narrative": "official narrative",
        "events": [agent_start, user_start, user_end],
    }


def test_piika_llm_with_no_prior_messages_still_advertises_both_tools(tmp_path):
    assistant = _assistant("answer", 1_800_000_000_000)
    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(
            _write_jsonl(
                tmp_path / "assistant-only.jsonl",
                [
                    {"type": "message_start", "message": assistant},
                    {"type": "message_end", "message": assistant},
                ],
            )
        ),
        run_record={"status": "completed"},
        session_id="session",
    )

    llm = next(span for span in bundle.root.children if span.kind == "LLM")
    assert llm.input_value["messages"] == []
    assert [tool["function"]["name"] for tool in llm.input_value["tools"]] == [
        "search",
        "read_document",
    ]


def test_piika_trace_uses_native_conversation_and_observed_completion_timing(tmp_path):
    session_ns = 1_800_000_000_000_000_000
    user_ms = 1_800_000_000_200
    first_ms = 1_800_000_000_300
    tool_result_ms = 1_800_000_012_900
    final_ms = 1_800_000_013_000
    completion_ns = 1_800_000_068_000_000_000
    user = {
        "role": "user",
        "content": [{"type": "text", "text": "question"}],
        "timestamp": user_ms,
    }
    first = _assistant("calling tool", first_ms)
    first["content"] = [
        {
            "type": "toolCall",
            "id": "call-1",
            "name": "search",
            "arguments": {"query": "q"},
        }
    ]
    tool_result = {
        "role": "toolResult",
        "toolCallId": "call-1",
        "toolName": "search",
        "content": [{"type": "text", "text": "result text"}],
        "timestamp": tool_result_ms,
    }
    final = _assistant("final answer", final_ms)
    path = _write_jsonl(
        tmp_path / "real-pattern.jsonl",
        [
            {"type": "session", "timestamp": "2027-01-15T08:00:00Z"},
            {"type": "message_end", "timestamp": user_ms, "message": user},
            {"type": "message_start", "timestamp": first_ms, "message": first},
            {"type": "message_end", "timestamp": first_ms + 1, "message": first},
            {
                "type": "tool_execution_start",
                "timestamp": first_ms + 2,
                "toolCallId": "call-1",
                "toolName": "search",
                "args": {"query": "q"},
            },
            {
                "type": "tool_execution_end",
                "timestamp": tool_result_ms - 1,
                "toolCallId": "call-1",
                "toolName": "search",
                "result": {
                    "content": "result text",
                    "details": {"timingMs": {"searchRpcMs": 2500.0}},
                },
                "isError": False,
            },
            {
                "type": "message_end",
                "timestamp": tool_result_ms,
                "message": tool_result,
            },
            {"type": "turn_end", "timestamp": tool_result_ms + 1, "message": first},
            {"type": "message_start", "timestamp": final_ms, "message": final},
            {"type": "message_end", "timestamp": final_ms + 1, "message": final},
            {"type": "turn_end", "timestamp": final_ms + 2, "message": final},
            {"type": "agent_end", "timestamp": final_ms + 3},
        ],
    )
    os.utime(path, ns=(completion_ns, completion_ns))

    events = load_pi_events(path)
    assert events.timing_reconstructed is False
    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "narrative must not be fabricated as every turn"),
        events=events,
        run_record={"status": "completed"},
        session_id="session",
    )

    llms = [span for span in bundle.root.children if span.kind == "LLM"]
    assert bundle.root.start_ns == session_ns
    assert bundle.root.end_ns == completion_ns
    assert llms[0].start_ns == first_ms * 1_000_000
    assert llms[0].end_ns == tool_result_ms * 1_000_000
    assert llms[0].attributes["trace.end_time.source"] == "next_tool_result_upper_bound"
    assert llms[0].input_value["messages"] == [user]
    assert [tool["function"]["name"] for tool in llms[0].input_value["tools"]] == [
        "search",
        "read_document",
    ]
    assert llms[1].start_ns == final_ms * 1_000_000
    assert llms[1].end_ns == completion_ns
    assert llms[1].attributes["trace.end_time.source"] == "event_file_mtime_upper_bound"
    assert llms[1].attributes["trace.timing_reconstructed"] is True
    assert bundle.root.attributes["trace.timing_reconstructed"] is True
    assert (
        bundle.root.children[-1].attributes["trace.timing_reconstructed"] is True
    )
    assert llms[1].input_value["messages"] == [user, first, tool_result]
    assert [tool["function"]["name"] for tool in llms[1].input_value["tools"]] == [
        "search",
        "read_document",
    ]
    tool = next(span for span in bundle.root.children if span.kind == "RETRIEVER")
    assert tool.end_ns - tool.start_ns == 2_500_000_000
    assert tool.attributes["trace.duration.source"] == "native_timing_ms"


def test_failed_tool_event_becomes_error_span_with_diagnostic(tmp_path):
    event_path = _write_jsonl(
        tmp_path / "failed.jsonl",
        [
            {
                "type": "tool_execution_start",
                "toolCallId": "search-failed",
                "toolName": "search",
                "args": {"query": "query"},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "search-failed",
                "toolName": "search",
                "result": {
                    "content": [{"type": "text", "text": "backend unavailable"}],
                    "details": {"status": 503},
                },
                "isError": True,
            },
        ],
    )

    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(event_path),
        run_record={"status": "failed", "error": "baseline stopped"},
        session_id="session",
    )

    failed = bundle.root.children[0]
    assert failed.kind == "RETRIEVER"
    assert failed.status == "ERROR"
    assert failed.status_message == "backend unavailable"
    assert failed.attributes["pi.tool_call.id"] == "search-failed"
    assert bundle.root.status == "ERROR"
    assert bundle.root.children[-1].status == "ERROR"
    assert bundle.root.children[-1].status_message == "baseline stopped"


@pytest.mark.parametrize(
    ("event", "expected_name", "expected_diagnostic"),
    [
        (
            {
                "type": "extension_error",
                "extensionPath": "/organizer/piika.ts",
                "event": "tool_call",
                "error": "extension backend failed",
            },
            "Pi extension_error",
            "extension backend failed",
        ),
        (
            {
                "type": "auto_retry_end",
                "success": False,
                "finalError": "model retries exhausted",
            },
            "Pi auto_retry_end",
            "model retries exhausted",
        ),
    ],
    ids=["extension-error", "retry-failed"],
)
def test_standalone_native_failure_event_becomes_error_span(
    tmp_path, event, expected_name, expected_diagnostic
):
    event_path = _write_jsonl(
        tmp_path / "extension-failed.jsonl",
        [event],
    )

    bundle = build_piika_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(event_path),
        run_record={"status": "completed"},
        session_id="session",
    )

    failed = bundle.root.children[0]
    assert failed.name == expected_name
    assert failed.kind == "CHAIN"
    assert failed.status == "ERROR"
    assert failed.status_message == expected_diagnostic
    assert failed.attributes["pi.event.type"] == event["type"]
    assert failed.output_value == event
    assert bundle.root.status == "ERROR"


def test_fixed_trace_keeps_100_ordered_documents_and_complete_prompts(tmp_path):
    final = _assistant("Fixed final payload [1].", 1_800_000_000_000)
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-events.jsonl",
            [
                {"type": "message_start", "message": final},
                {"type": "message_end", "message": final},
            ],
        )
    )
    documents = [
        {"rank": rank, "docid": f"d{rank}", "text": f"full text {rank}"}
        for rank in range(1, 101)
    ]

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={"status": "completed", "references": ["d1"]},
        system_prompt="complete system prompt",
        user_prompt="complete user prompt with all evidence",
        documents=documents,
        session_id="rag2026-1-comparison",
    )

    assert bundle.baseline == "ragnarok-fixed"
    assert bundle.root.kind == "AGENT"
    assert bundle.root.attributes["pi.system_prompt.captured"] is True
    assert [child.kind for child in bundle.root.children] == [
        "RETRIEVER",
        "CHAIN",
        "LLM",
        "CHAIN",
    ]
    assert bundle.root.children[0].output_value["documents"] == documents
    assert len(bundle.root.children[0].output_value["documents"]) == 100
    assert bundle.root.children[0].output_value["documents"][-1] == {
        "rank": 100,
        "docid": "d100",
        "text": "full text 100",
    }
    generation = bundle.root.children[2]
    assert generation.input_value == {
        "system_prompt": "complete system prompt",
        "user_prompt": "complete user prompt with all evidence",
    }
    assert generation.output_value == final
    assert bundle.root.children[-1].name == "organizer validation"
    assert sum(span.kind == "LLM" for span in bundle.root.children) == 1
    assert not any(child.kind == "LLM" for child in generation.children)


def test_fixed_trace_marks_every_derived_interval_as_reconstructed(tmp_path):
    native_timestamp_ms = 1_800_000_000_000
    aborted = _assistant(
        "partial attempt", native_timestamp_ms, stop_reason="aborted"
    )
    final = _assistant("fixed final", native_timestamp_ms + 100)
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-native-timing.jsonl",
            [
                {
                    "type": "message_end",
                    "message": aborted,
                },
                {"type": "message_end", "message": final},
                {
                    "type": "extension_error",
                    "timestamp": native_timestamp_ms + 200,
                    "error": "post-generation failure",
                },
            ],
        )
    )
    assert events.timing_reconstructed is False

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    evidence, prompts, generation, validation = bundle.root.children
    assert generation.start_ns == native_timestamp_ms * 1_000_000
    attempt, failure = generation.children
    assert attempt.start_ns == native_timestamp_ms * 1_000_000
    assert failure.start_ns == (native_timestamp_ms + 200) * 1_000_000
    assert bundle.root.attributes["trace.timing_reconstructed"] is True
    assert all(
        span.attributes["trace.timing_reconstructed"] is True
        for span in (evidence, prompts, generation, validation, attempt, failure)
    )


def _walk_spans(span):
    yield span
    for child in span.children:
        yield from _walk_spans(child)


def test_fixed_trace_extracts_native_reasoning_into_three_children(tmp_path):
    final = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "first native block"},
            {"type": "thinking", "thinking": "second native block"},
            {"type": "thinking", "thinking": "third native block"},
            {"type": "text", "text": "final answer"},
        ],
        "provider": "anthropic",
        "model": "model-id",
        "stopReason": "stop",
        "timestamp": 1_800_000_000_000,
    }
    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(
            _write_jsonl(
                tmp_path / "fixed-reasoning-events.jsonl",
                [
                    {"type": "message_start", "message": final},
                    {"type": "message_end", "message": final},
                ],
            )
        ),
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = bundle.root.children[2]
    assert generation.kind == "LLM"
    assert generation.attributes["pi.reasoning.extracted_to_children"] is True
    assert generation.attributes["pi.reasoning.block_count"] == 3
    assert [child.name for child in generation.children] == [
        "Reasoning summary 1",
        "Reasoning summary 2",
        "Reasoning summary 3",
    ]
    assert [child.output_value for child in generation.children] == [
        "first native block",
        "second native block",
        "third native block",
    ]
    assert all(child.kind == "CHAIN" for child in generation.children)
    assert all(
        child.attributes["trace.timing_reconstructed"] is True
        for child in generation.children
    )
    assert all(
        child.attributes["pi.reasoning.timing_method"] == "equal-partition"
        for child in generation.children
    )
    assert generation.children[0].start_ns == generation.start_ns
    assert generation.children[-1].end_ns == generation.end_ns
    assert all(
        left.end_ns == right.start_ns
        for left, right in zip(generation.children, generation.children[1:])
    )
    assert sum(span.kind == "LLM" for span in _walk_spans(bundle.root)) == 1
    assert sum(1 for _ in _walk_spans(bundle.root)) == 8


def test_fixed_trace_without_thinking_keeps_existing_topology(tmp_path):
    final = _assistant("final answer", 1_800_000_000_000)
    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(
            _write_jsonl(
                tmp_path / "fixed-no-reasoning-events.jsonl",
                [{"type": "message_end", "message": final}],
            )
        ),
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = bundle.root.children[2]
    assert generation.children == ()
    assert "pi.reasoning.extracted_to_children" not in generation.attributes
    assert sum(1 for _ in _walk_spans(bundle.root)) == 5


def test_fixed_trace_generation_uses_observed_file_completion_without_duplicate_llm(tmp_path):
    start_ms = 1_800_000_000_000
    completion_ns = 1_800_000_062_600_000_000
    final = _assistant("final", start_ms)
    path = _write_jsonl(
        tmp_path / "fixed-completion.jsonl",
        [
            {"type": "session", "timestamp": 1_799_999_999_000},
            {"type": "message_start", "message": final},
            {"type": "message_end", "message": final},
            {"type": "turn_end", "message": final},
            {"type": "agent_end"},
        ],
    )
    os.utime(path, ns=(completion_ns, completion_ns))

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(path),
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = next(span for span in bundle.root.children if span.kind == "LLM")
    assert bundle.root.end_ns == completion_ns
    assert generation.start_ns == start_ms * 1_000_000
    assert generation.end_ns == completion_ns
    assert generation.attributes["trace.end_time.source"] == "event_file_mtime_upper_bound"
    assert not any(child.kind == "LLM" for child in generation.children)


def test_fixed_trace_nests_native_failures_under_generation(tmp_path):
    final = _assistant("partial final payload", 1_800_000_000_000)
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-failed-events.jsonl",
            [
                {"type": "message_end", "message": final},
                {"type": "extension_error", "error": "fixed extension failed"},
            ],
        )
    )

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = bundle.root.children[2]
    assert generation.output_value == final
    assert generation.status == "ERROR"
    assert generation.status_message == "fixed extension failed"
    assert len(generation.children) == 1
    assert generation.children[0].name == "Pi extension_error"
    assert generation.children[0].output_value["error"] == "fixed extension failed"
    assert bundle.root.status == "ERROR"


def test_fixed_trace_early_native_failure_omits_unreached_success_spans(tmp_path):
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-early-failure.jsonl",
            [
                {
                    "type": "extension_error",
                    "timestamp": 1_800_000_000_000,
                    "error": "fixed extension failed before prompt construction",
                }
            ],
        )
    )

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={
            "status": "failed",
            "error": "fixed extension failed before prompt construction",
        },
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    assert [span.name for span in bundle.root.children] == [
        "Pi extension_error",
        "organizer validation",
    ]
    failure, validation = bundle.root.children
    assert failure.status == "ERROR"
    assert failure.output_value["error"] == (
        "fixed extension failed before prompt construction"
    )
    assert validation.status == "ERROR"
    assert validation.status_message == (
        "fixed extension failed before prompt construction"
    )
    assert bundle.root.status == "ERROR"


def test_fixed_trace_retains_every_assistant_attempt_and_final_output(tmp_path):
    aborted = _assistant(
        "partial assistant attempt", 1_800_000_000_000, stop_reason="aborted"
    )
    aborted["errorMessage"] = "first attempt aborted"
    final = _assistant("successful final output", 1_800_000_000_100)
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-attempts.jsonl",
            [
                {"type": "message_end", "message": aborted},
                {"type": "message_end", "message": final},
            ],
        )
    )

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = bundle.root.children[2]
    assert generation.output_value == final
    assert [child.output_value for child in generation.children] == [aborted]
    assert [child.kind for child in generation.children] == ["CHAIN"]
    assert [child.status for child in generation.children] == ["ERROR"]
    assert generation.children[0].status_message == "first attempt aborted"
    assert generation.status == "ERROR"
    assert generation.status_message == "first attempt aborted"


def test_fixed_generation_timing_encloses_all_native_children(tmp_path):
    final = _assistant("final output", 1_800_000_000_100)
    events = load_pi_events(
        _write_jsonl(
            tmp_path / "fixed-hierarchy.jsonl",
            [
                {
                    "type": "extension_error",
                    "timestamp": 1_800_000_000_000,
                    "error": "early extension failure",
                },
                {"type": "message_end", "message": final},
            ],
        )
    )

    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=events,
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )

    generation = bundle.root.children[2]
    assert generation.children[0].status_message == "early extension failure"
    assert generation.start_ns <= min(child.start_ns for child in generation.children)
    assert generation.end_ns >= max(child.end_ns for child in generation.children)


@pytest.mark.parametrize(
    "body",
    [
        "   \n\t\n",
        "[]\n",
        '{"type":"one","type":"two"}\n',
        '{"type":"event","value":NaN}\n',
        '{"type":"event","value":Infinity}\n',
        '{"type":"event","value":-Infinity}\n',
    ],
    ids=["blank-only", "non-object", "duplicate-key", "nan", "infinity", "negative-infinity"],
)
def test_load_pi_events_rejects_non_strict_jsonl(tmp_path, body):
    path = tmp_path / "invalid.jsonl"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        load_pi_events(path)


def test_load_pi_events_rejects_files_over_configured_ceiling(tmp_path):
    path = tmp_path / "too-large.jsonl"
    path.write_text('{"type":"event"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="byte ceiling"):
        load_pi_events(path, max_bytes=4)


def test_load_pi_events_preserves_order_and_reconstructs_monotonic_timing(tmp_path):
    path = _write_jsonl(
        tmp_path / "ordered.jsonl",
        [{"type": "first"}, {"type": "second"}, {"type": "third"}],
    )
    modification_ns = 1_700_000_000_123_456_789
    os.utime(path, ns=(modification_ns, modification_ns))

    events = load_pi_events(path)

    assert [event["type"] for event in events] == ["first", "second", "third"]
    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (
        modification_ns,
        modification_ns + 1,
        modification_ns + 2,
    )


def test_load_pi_events_uses_parseable_native_timestamps(tmp_path):
    path = _write_jsonl(
        tmp_path / "timed.jsonl",
        [
            {"type": "first", "timestamp": "2027-01-15T08:00:00Z"},
            {"type": "second", "timestamp": 1_800_000_001_000},
        ],
    )

    events = load_pi_events(path)

    assert events.timing_reconstructed is False
    assert events.timestamps_ns == (1_800_000_000_000_000_000, 1_800_000_001_000_000_000)


def test_load_pi_events_preserves_integer_nanosecond_timestamp_exactly(tmp_path):
    timestamp_ns = 1_800_000_000_000_000_123
    path = _write_jsonl(
        tmp_path / "nanoseconds.jsonl",
        [{"type": "event", "timestamp": timestamp_ns}],
    )

    events = load_pi_events(path)

    assert events.timing_reconstructed is False
    assert events.timestamps_ns == (timestamp_ns,)


def test_load_pi_events_preserves_native_timestamps_while_filling_gaps(tmp_path):
    first_ns = 1_800_000_000_000_000_000
    last_ns = first_ns + 100
    path = _write_jsonl(
        tmp_path / "mixed-timing.jsonl",
        [
            {"type": "first", "timestamp": first_ns},
            {"type": "missing"},
            {"type": "last", "timestamp": last_ns},
        ],
    )

    events = load_pi_events(path)

    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (first_ns, first_ns + 50, last_ns)


def test_load_pi_events_breaks_native_millisecond_timestamp_ties_without_losing_epoch_gaps(
    tmp_path,
):
    first_ms = 1_800_000_000_000
    last_ms = first_ms + 98_956
    path = _write_jsonl(
        tmp_path / "tied-millisecond-timing.jsonl",
        [
            {"type": "first", "timestamp": first_ms},
            {"type": "same-millisecond", "timestamp": first_ms},
            {"type": "same-millisecond-again", "timestamp": first_ms},
            {"type": "last", "timestamp": last_ms},
        ],
    )

    events = load_pi_events(path)

    first_ns = first_ms * 1_000_000
    last_ns = last_ms * 1_000_000
    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (first_ns, first_ns + 1, first_ns + 2, last_ns)
    assert events.timestamps_ns[-1] - events.timestamps_ns[0] == 98_956_000_000


def test_load_pi_events_repairs_backward_native_events_without_losing_later_gap(
    tmp_path,
):
    first_ms = 1_800_000_000_000
    path = _write_jsonl(
        tmp_path / "backward-native-event.jsonl",
        [
            {"type": "first", "timestamp": first_ms},
            {"type": "same-millisecond", "timestamp": first_ms},
            {"type": "late-emitted-old-event", "timestamp": first_ms - 12_714},
            {"type": "last", "timestamp": first_ms + 98_956},
        ],
    )

    events = load_pi_events(path)

    first_ns = first_ms * 1_000_000
    assert events.timestamps_ns == (
        first_ns,
        first_ns + 1,
        first_ns + 2,
        (first_ms + 98_956) * 1_000_000,
    )
    assert events.timestamps_ns[-1] - events.timestamps_ns[0] == 98_956_000_000


def test_load_pi_events_minimally_breaks_collisions_between_native_anchors(tmp_path):
    first_ns = 1_800_000_000_000_000_000
    path = _write_jsonl(
        tmp_path / "conflicting-timing.jsonl",
        [
            {"type": "first", "timestamp": first_ns},
            {"type": "missing"},
            {"type": "last", "timestamp": first_ns + 1},
        ],
    )

    events = load_pi_events(path)

    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (first_ns, first_ns + 1, first_ns + 2)


def test_load_pi_events_minimally_repairs_reversed_native_anchors(tmp_path):
    first_ns = 1_800_000_000_000_000_000
    path = _write_jsonl(
        tmp_path / "reversed-timing.jsonl",
        [
            {"type": "first", "timestamp": first_ns},
            {"type": "missing"},
            {"type": "last", "timestamp": first_ns - 1},
        ],
    )
    events = load_pi_events(path)

    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (
        first_ns,
        first_ns + 1,
        first_ns + 2,
    )

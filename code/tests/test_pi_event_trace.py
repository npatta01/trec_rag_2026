from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os

import pytest

from trec_rag.organizer_pi_inputs import OrganizerTopic
from trec_rag.pi_event_trace import (
    build_fixed_trace,
    build_piika_trace,
    load_pi_events,
)
from trec_rag.pi_trace_models import read_trace_bundle, write_trace_bundle


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
    assert bundle.root.kind == "CHAIN"
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
    assert len(generation.children) == 2
    assert generation.children[0].output_value == final
    assert generation.children[1].name == "Pi extension_error"
    assert generation.children[1].output_value["error"] == "fixed extension failed"
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
    assert [child.output_value for child in generation.children] == [aborted, final]
    assert [child.status for child in generation.children] == ["ERROR", "OK"]
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


@pytest.mark.parametrize("last_offset", [1, -1], ids=["collision", "reversed"])
def test_load_pi_events_reconstructs_when_native_anchors_cannot_be_monotonic(
    tmp_path, last_offset
):
    first_ns = 1_800_000_000_000_000_000
    path = _write_jsonl(
        tmp_path / "conflicting-timing.jsonl",
        [
            {"type": "first", "timestamp": first_ns},
            {"type": "missing"},
            {"type": "last", "timestamp": first_ns + last_offset},
        ],
    )
    modification_ns = 1_700_000_000_123_456_789
    os.utime(path, ns=(modification_ns, modification_ns))

    events = load_pi_events(path)

    assert events.timing_reconstructed is True
    assert events.timestamps_ns == (
        modification_ns,
        modification_ns + 1,
        modification_ns + 2,
    )


def test_trace_bundle_is_immutable_and_strict_json_round_trips_atomically(tmp_path):
    final = _assistant("final", 1_800_000_000_000)
    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(
            _write_jsonl(
                tmp_path / "events.jsonl",
                [{"type": "message_end", "message": final}],
            )
        ),
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )
    path = tmp_path / "trace.json"

    assert write_trace_bundle(bundle, path) == path
    restored = read_trace_bundle(path)

    assert restored == bundle
    with pytest.raises(FrozenInstanceError):
        restored.session_id = "changed"
    with pytest.raises(TypeError):
        restored.root.attributes["topic.id"] = "changed"
    documents = restored.root.children[0].output_value["documents"]
    with pytest.raises(TypeError):
        documents.append({"rank": 2, "docid": "d2", "text": "other"})
    with pytest.raises(TypeError):
        documents[0]["text"] = "changed"

    original_bytes = path.read_bytes()
    path.write_text('{"project_name":"first","project_name":"second"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace_bundle(path)
    path.write_bytes(original_bytes)

    generation = bundle.root.children[2]
    object.__setattr__(generation, "output_value", {"not_json": float("nan")})
    with pytest.raises(ValueError):
        write_trace_bundle(bundle, path)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("project_name", 17),
        ("session_id", ["session"]),
        ("topic_id", {"topic": "rag2026-1"}),
        ("baseline", ["piika-agentic"]),
        ("root.name", 17),
        ("root.kind", ["CHAIN"]),
        ("root.status", 17),
        ("root.status_message", {"message": "bad"}),
    ],
    ids=[
        "project-name",
        "session-id",
        "topic-id",
        "baseline",
        "span-name",
        "span-kind",
        "span-status",
        "span-status-message",
    ],
)
def test_read_trace_bundle_rejects_non_string_contract_fields(
    tmp_path, field, invalid_value
):
    final = _assistant("final", 1_800_000_000_000)
    bundle = build_fixed_trace(
        topic=OrganizerTopic("rag2026-1", "question"),
        events=load_pi_events(
            _write_jsonl(
                tmp_path / "strict-events.jsonl",
                [{"type": "message_end", "message": final}],
            )
        ),
        run_record={"status": "completed"},
        system_prompt="system",
        user_prompt="user",
        documents=[{"rank": 1, "docid": "d1", "text": "full text"}],
        session_id="session",
    )
    path = write_trace_bundle(bundle, tmp_path / "invalid-contract.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if field.startswith("root."):
        record["root"][field.removeprefix("root.")] = invalid_value
    else:
        record[field] = invalid_value
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid (trace bundle|span value)"):
        read_trace_bundle(path)

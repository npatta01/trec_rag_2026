"""Normalize native Pi JSONL events into durable trace trees."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

from trec_rag.organizer_pi_inputs import OrganizerTopic
from trec_rag.openai_trace_semantics import piika_tool_schemas
from trec_rag.pi_trace_models import (
    SpanSpec,
    TraceBundle,
    _strict_json_loads,
    read_trace_bundle,
    write_trace_bundle,
)


DEFAULT_EVENT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_PROJECT_NAME = "trec-rag-2026-pi-baselines"

_KNOWN_EVENT_TYPES = {
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "queue_update",
    "compaction_start",
    "compaction_end",
    "auto_retry_start",
    "auto_retry_end",
    "extension_error",
}

class LoadedPiEvents(tuple):
    """Ordered native event records plus loader-derived timing metadata."""

    timing_reconstructed: bool
    timestamps_ns: tuple[int, ...]
    observed_end_ns: int
    observed_end_source: str

    def __new__(
        cls,
        events: Iterable[Mapping[str, object]],
        *,
        timestamps_ns: Sequence[int],
        timing_reconstructed: bool,
        observed_end_ns: int,
        observed_end_source: str,
    ) -> "LoadedPiEvents":
        instance = super().__new__(cls, tuple(events))
        instance.timestamps_ns = tuple(timestamps_ns)
        instance.timing_reconstructed = timing_reconstructed
        instance.observed_end_ns = observed_end_ns
        instance.observed_end_source = observed_end_source
        return instance


def _parse_timestamp_ns(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0:
            return None
        magnitude = value
        if magnitude >= 100_000_000_000_000_000:
            multiplier = 1
        elif magnitude >= 100_000_000_000_000:
            multiplier = 1_000
        elif magnitude >= 100_000_000_000:
            multiplier = 1_000_000
        else:
            multiplier = 1_000_000_000
        return value * multiplier
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            return None
        if value >= 100_000_000_000_000_000:
            multiplier = 1
        elif value >= 100_000_000_000_000:
            multiplier = 1_000
        elif value >= 100_000_000_000:
            multiplier = 1_000_000
        else:
            multiplier = 1_000_000_000
        return int(value * multiplier)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = parsed - epoch
        return (
            delta.days * 86_400 * 1_000_000_000
            + delta.seconds * 1_000_000_000
            + delta.microseconds * 1_000
        )
    return None


def _native_timestamp_ns(event: Mapping[str, object]) -> int | None:
    direct = _parse_timestamp_ns(event.get("timestamp"))
    if direct is not None:
        return direct
    message = event.get("message")
    if isinstance(message, Mapping):
        return _parse_timestamp_ns(message.get("timestamp"))
    return None


def _resolve_timestamps(
    parsed: Sequence[int | None], *, fallback_ns: int
) -> tuple[tuple[int, ...], bool]:
    anchors = [(index, value) for index, value in enumerate(parsed) if value is not None]
    if not anchors:
        return tuple(fallback_ns + index for index in range(len(parsed))), True
    if anchors[0][1] < anchors[0][0]:
        return tuple(fallback_ns + index for index in range(len(parsed))), True

    resolved = list(parsed)
    index = 0
    while index < len(resolved):
        if resolved[index] is not None:
            index += 1
            continue
        start = index
        while index < len(resolved) and resolved[index] is None:
            index += 1
        stop = index
        count = stop - start
        previous = resolved[start - 1] if start else None
        following = resolved[stop] if stop < len(resolved) else None

        if previous is not None and following is not None and following > previous:
            distance = following - previous
            for offset in range(1, count + 1):
                resolved[start + offset - 1] = previous + (
                    distance * offset // (count + 1)
                )
        elif previous is not None:
            for offset in range(1, count + 1):
                resolved[start + offset - 1] = previous + offset
        elif following is not None:
            first = max(0, following - count)
            for offset in range(count):
                resolved[start + offset] = first + offset
        else:  # Covered by the all-missing branch; kept for type completeness.
            for offset in range(count):
                resolved[start + offset] = fallback_ns + start + offset

    reconstructed = len(anchors) != len(parsed)
    monotonic: list[int] = []
    for value in resolved:
        if value is None:  # Covered above; kept for type narrowing.
            continue
        if monotonic and value <= monotonic[-1]:
            value = monotonic[-1] + 1
            reconstructed = True
        monotonic.append(value)
    return tuple(monotonic), reconstructed


def load_pi_events(
    path: Path, *, max_bytes: int = DEFAULT_EVENT_MAX_BYTES
) -> LoadedPiEvents:
    """Load strict native Pi JSONL and attach monotonic nanosecond timing."""
    if isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    source = Path(path)
    stat = source.stat()
    if stat.st_size > max_bytes:
        raise ValueError("Pi event file exceeds configured byte ceiling")
    try:
        body = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Pi event file must be UTF-8") from error

    events: list[dict[str, object]] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = _strict_json_loads(line)
        except ValueError as error:
            raise ValueError(f"invalid Pi JSONL row {line_number}") from error
        if not isinstance(event, dict):
            raise ValueError(f"Pi JSONL row {line_number} must be an object")
        events.append(event)
    if not events:
        raise ValueError("Pi event file contains no event objects")

    timestamps, reconstructed = _resolve_timestamps(
        [_native_timestamp_ns(event) for event in events],
        fallback_ns=stat.st_mtime_ns,
    )
    observed_end_ns = max(stat.st_mtime_ns, timestamps[-1] + 1)
    return LoadedPiEvents(
        events,
        timestamps_ns=timestamps,
        timing_reconstructed=reconstructed,
        observed_end_ns=observed_end_ns,
        observed_end_source=(
            "event_file_mtime"
            if stat.st_mtime_ns == observed_end_ns
            else "last_native_event_upper_bound"
        ),
    )


def _events_with_timing(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], tuple[int, ...], bool, int, str]:
    records = tuple(events)
    if isinstance(events, LoadedPiEvents):
        return (
            records,
            events.timestamps_ns,
            events.timing_reconstructed,
            events.observed_end_ns,
            events.observed_end_source,
        )

    timestamps, reconstructed = _resolve_timestamps(
        tuple(_native_timestamp_ns(event) for event in records),
        fallback_ns=time.time_ns(),
    )
    return (
        records,
        timestamps,
        reconstructed,
        timestamps[-1] + 1,
        "last_native_event_upper_bound",
    )


def _message_role(event: Mapping[str, object]) -> object:
    message = event.get("message")
    return message.get("role") if isinstance(message, Mapping) else None


def _is_prompt_event(event: Mapping[str, object]) -> bool:
    event_type = event.get("type")
    if event_type == "agent_start":
        return bool(set(event) - {"type", "timestamp"})
    return event_type in {"message_start", "message_end"} and _message_role(
        event
    ) in {"user", "system"}


def _message_error(message: Mapping[str, object]) -> str | None:
    stop_reason = message.get("stopReason")
    if stop_reason not in {"error", "aborted"}:
        return None
    for key in ("errorMessage", "error", "message"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return str(stop_reason)


def _diagnostic(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, Mapping):
        return None
    content = value.get("content")
    if isinstance(content, list | tuple):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping)
            and isinstance(item.get("text"), str)
            and item.get("text")
        ]
        if texts:
            return "\n".join(texts)
    for key in (
        "error",
        "errorMessage",
        "finalError",
        "status_message",
        "message",
        "diagnostic",
        "reason",
    ):
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    details = value.get("details")
    if details is not value:
        return _diagnostic(details)
    return None


def _failure_event_span(
    event: Mapping[str, object], timestamp_ns: int
) -> SpanSpec:
    event_type = str(event.get("type", "unknown"))
    diagnostic = _diagnostic(event) or f"Pi reported {event_type}"
    return SpanSpec(
        name=f"Pi {event_type}",
        kind="CHAIN",
        start_ns=timestamp_ns,
        end_ns=timestamp_ns + 1,
        attributes={"pi.event.type": event_type},
        input_value=None,
        output_value=dict(event),
        status="ERROR",
        status_message=diagnostic,
    )


def _prompt_span(
    *,
    topic: OrganizerTopic,
    events: Sequence[tuple[int, Mapping[str, object], int]],
) -> SpanSpec:
    start_ns = events[0][2]
    end_ns = max(start_ns + 1, events[-1][2])
    native_events = [dict(event) for _, event, _ in events]
    return SpanSpec(
        name="Pi prompt construction",
        kind="CHAIN",
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={
            "pi.prompt.event_count": len(events),
            "pi.prompt.event_types": tuple(
                str(event.get("type", "unknown")) for _, event, _ in events
            ),
        },
        input_value={
            "topic_id": topic.topic_id,
            "narrative": topic.narrative,
            "events": native_events,
        },
        output_value=None,
        status="OK",
    )


def _tool_output(result: object) -> object:
    if not isinstance(result, Mapping):
        return {"result": result}
    output = dict(result)
    details = result.get("details")
    if isinstance(details, Mapping):
        for key, value in details.items():
            output.setdefault(key, value)
    return output


def _tool_span(
    start_event: Mapping[str, object],
    end_event: Mapping[str, object] | None,
    start_ns: int,
    end_ns: int,
) -> SpanSpec:
    tool_name = start_event.get("toolName")
    tool_call_id = start_event.get("toolCallId")
    kind = "RETRIEVER" if tool_name == "search" else "TOOL"
    failed = end_event is None or end_event.get("isError") is True
    result = end_event.get("result") if end_event is not None else None
    diagnostic = _diagnostic(result) if failed else None
    if failed and diagnostic is None:
        diagnostic = "tool execution did not produce a matching end event"
    timing_ms: float | None = None
    if isinstance(result, Mapping):
        details = result.get("details")
        if isinstance(details, Mapping):
            timings = details.get("timingMs")
            if isinstance(timings, Mapping):
                values = [
                    float(value)
                    for value in timings.values()
                    if isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and value >= 0
                ]
                if values:
                    timing_ms = max(values)
    attributes: dict[str, object] = {
        "pi.event.start_type": "tool_execution_start",
        "pi.event.end_type": (
            "tool_execution_end" if end_event is not None else "missing"
        ),
        "pi.tool_call.id": str(tool_call_id or ""),
        "pi.tool.name": str(tool_name or ""),
    }
    if timing_ms is not None:
        end_ns = start_ns + max(1, round(timing_ms * 1_000_000))
        attributes["trace.duration.source"] = "native_timing_ms"
    return SpanSpec(
        name=f"Pi {tool_name or 'tool'}",
        kind=kind,
        start_ns=start_ns,
        end_ns=max(start_ns + 1, end_ns),
        attributes=attributes,
        input_value=start_event.get("args"),
        output_value=_tool_output(result),
        status="ERROR" if failed else "OK",
        status_message=diagnostic,
    )


def _assistant_span(
    start_event: Mapping[str, object] | None,
    end_event: Mapping[str, object],
    start_ns: int,
    end_ns: int,
    *,
    input_messages: Sequence[Mapping[str, object]],
    end_time_source: str,
) -> SpanSpec:
    message = end_event.get("message")
    output = dict(message) if isinstance(message, Mapping) else message
    diagnostic = _message_error(message) if isinstance(message, Mapping) else None
    return SpanSpec(
        name="Pi assistant turn",
        kind="LLM",
        start_ns=start_ns,
        end_ns=max(start_ns + 1, end_ns),
        attributes={
            "pi.event.start_type": (
                "message_start" if start_event is not None else "missing"
            ),
            "pi.event.end_type": "message_end",
            "trace.end_time.source": end_time_source,
            "trace.end_time.upper_bound": True,
        },
        input_value={
            "messages": [dict(message) for message in input_messages],
            "tools": list(piika_tool_schemas()),
        },
        output_value=output,
        status="ERROR" if diagnostic is not None else "OK",
        status_message=diagnostic,
    )


def _reasoning_summary_children(
    message: object, *, start_ns: int, end_ns: int
) -> tuple[SpanSpec, ...]:
    """Create presentation-only children for captured native thinking blocks."""
    if not isinstance(message, Mapping):
        return ()
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
        return ()
    blocks = [
        item["thinking"]
        for item in content
        if isinstance(item, Mapping)
        and item.get("type") == "thinking"
        and isinstance(item.get("thinking"), str)
        and item["thinking"]
    ]
    if not blocks:
        return ()

    count = len(blocks)
    base_duration, remainder = divmod(end_ns - start_ns, count)
    children: list[SpanSpec] = []
    child_start = start_ns
    for index, block in enumerate(blocks, start=1):
        child_end = child_start + base_duration
        if index == count:
            child_end += remainder
        children.append(
            SpanSpec(
                name=f"Reasoning summary {index}",
                kind="CHAIN",
                start_ns=child_start,
                end_ns=child_end,
                attributes={
                    "pi.reasoning.block.index": index,
                    "pi.reasoning.block.count": count,
                    "pi.reasoning.source": "native-thinking-content",
                    "pi.reasoning.timing_method": "equal-partition",
                    "trace.timing_reconstructed": True,
                },
                input_value=None,
                output_value=block,
                status="OK",
            )
        )
        child_start = child_end
    return tuple(children)


def _validation_span(
    *,
    topic: OrganizerTopic,
    run_record: Mapping[str, object],
    timestamp_ns: int,
) -> SpanSpec:
    completed = run_record.get("status") == "completed"
    diagnostic = None if completed else _diagnostic(run_record)
    if not completed and diagnostic is None:
        diagnostic = f"organizer run status: {run_record.get('status', 'missing')}"
    return SpanSpec(
        name="organizer validation",
        kind="CHAIN",
        start_ns=timestamp_ns,
        end_ns=timestamp_ns + 1,
        attributes={"organizer.run.status": str(run_record.get("status", "missing"))},
        input_value={"topic_id": topic.topic_id},
        output_value=dict(run_record),
        status="OK" if completed else "ERROR",
        status_message=diagnostic,
    )


def _native_spans(
    *,
    topic: OrganizerTopic,
    events: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[int, SpanSpec]], tuple[int, ...], bool, int, int, int, str]:
    (
        records,
        timestamps,
        reconstructed,
        observed_end_ns,
        observed_end_source,
    ) = _events_with_timing(events)
    starts: dict[object, tuple[int, Mapping[str, object], int]] = {}
    assistant_start: tuple[int, Mapping[str, object], int] | None = None
    prompt_events: list[tuple[int, Mapping[str, object], int]] = []
    conversation: list[Mapping[str, object]] = []
    spans: list[tuple[int, SpanSpec]] = []

    for index, (event, timestamp_ns) in enumerate(zip(records, timestamps)):
        event_type = event.get("type")
        if _is_prompt_event(event):
            prompt_events.append((index, event, timestamp_ns))
        if event_type == "message_start" and _message_role(event) == "assistant":
            assistant_start = (index, event, timestamp_ns)
        elif event_type == "message_end" and _message_role(event) == "assistant":
            if assistant_start is None:
                start_index, start_event, start_ns = index, None, timestamp_ns
            else:
                start_index, start_event, start_ns = assistant_start
            next_boundary = observed_end_ns
            end_source = f"{observed_end_source}_upper_bound"
            for later_event, later_ns in zip(
                records[index + 1 :], timestamps[index + 1 :]
            ):
                later_role = _message_role(later_event)
                if (
                    later_event.get("type") in {"message_start", "message_end"}
                    and later_role == "toolResult"
                ):
                    next_boundary = later_ns
                    end_source = "next_tool_result_upper_bound"
                    break
                if later_event.get("type") == "message_start" and later_role == "assistant":
                    next_boundary = later_ns
                    end_source = "next_assistant_start_upper_bound"
                    break
            spans.append(
                (
                    start_index,
                    _assistant_span(
                        start_event,
                        event,
                        start_ns,
                        max(start_ns + 1, next_boundary),
                        input_messages=conversation,
                        end_time_source=end_source,
                    ),
                )
            )
            if isinstance(event.get("message"), Mapping):
                conversation.append(dict(event["message"]))
            assistant_start = None
        elif event_type == "tool_execution_start":
            starts[event.get("toolCallId")] = (index, event, timestamp_ns)
        elif event_type == "tool_execution_end":
            call_id = event.get("toolCallId")
            start = starts.pop(call_id, None)
            if start is None:
                synthetic_start = {
                    "type": "tool_execution_start",
                    "toolCallId": call_id,
                    "toolName": event.get("toolName"),
                    "args": None,
                }
                start = (index, synthetic_start, timestamp_ns)
            start_index, start_event, start_ns = start
            spans.append(
                (
                    start_index,
                    _tool_span(start_event, event, start_ns, timestamp_ns),
                )
            )
        elif event_type == "extension_error" or (
            event_type == "auto_retry_end"
            and (
                event.get("success") is False
                or event.get("error") is not None
                or event.get("finalError") is not None
            )
        ):
            spans.append((index, _failure_event_span(event, timestamp_ns)))
        elif event_type == "message_end" and _message_role(event) in {
            "user",
            "system",
            "toolResult",
        }:
            message = event.get("message")
            if isinstance(message, Mapping):
                conversation.append(dict(message))

    last_timestamp = timestamps[-1] if timestamps else time.time_ns()
    for _, (start_index, start_event, start_ns) in starts.items():
        spans.append(
            (
                start_index,
                _tool_span(start_event, None, start_ns, last_timestamp + 1),
            )
        )
    if prompt_events:
        spans.append(
            (prompt_events[0][0], _prompt_span(topic=topic, events=prompt_events))
        )
    spans.sort(key=lambda item: item[0])
    unknown_count = sum(event.get("type") not in _KNOWN_EVENT_TYPES for event in records)
    return (
        spans,
        timestamps,
        reconstructed,
        unknown_count,
        timestamps[0],
        observed_end_ns,
        observed_end_source,
    )


def _root_span(
    *,
    name: str,
    topic: OrganizerTopic,
    children: Sequence[SpanSpec],
    reconstructed: bool,
    unknown_count: int,
    observed_start_ns: int,
    observed_end_ns: int,
    observed_end_source: str,
    system_prompt_captured: bool | None = None,
) -> SpanSpec:
    start_ns = min(observed_start_ns, *(child.start_ns for child in children))
    end_ns = max(observed_end_ns, *(child.end_ns for child in children))
    failed_children = [child for child in children if child.status == "ERROR"]
    return SpanSpec(
        name=name,
        kind="AGENT",
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={
            "topic.id": topic.topic_id,
            "content.capture": "full",
            "pi.event.unknown_count": unknown_count,
            "trace.timing_reconstructed": reconstructed,
            "trace.root.end_time.source": observed_end_source,
            **(
                {"pi.system_prompt.captured": system_prompt_captured}
                if system_prompt_captured is not None
                else {}
            ),
        },
        input_value={"topic_id": topic.topic_id, "narrative": topic.narrative},
        output_value=None,
        status="ERROR" if failed_children else "OK",
        status_message=(failed_children[0].status_message if failed_children else None),
        children=tuple(children),
    )


def build_piika_trace(
    *,
    topic: OrganizerTopic,
    events: Sequence[Mapping[str, object]],
    run_record: Mapping[str, object],
    session_id: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> TraceBundle:
    """Build an agentic Piika trace from paired native Pi events."""
    (
        native_spans,
        timestamps,
        reconstructed,
        unknown_count,
        observed_start_ns,
        observed_end_ns,
        observed_end_source,
    ) = _native_spans(
        topic=topic, events=events
    )
    validation = _validation_span(
        topic=topic,
        run_record=run_record,
        timestamp_ns=max(observed_start_ns, observed_end_ns - 1),
    )
    children = [span for _, span in native_spans] + [validation]
    return TraceBundle(
        project_name=project_name,
        session_id=session_id,
        topic_id=topic.topic_id,
        baseline="piika-agentic",
        root=_root_span(
            name="Piika agentic baseline",
            topic=topic,
            children=children,
            reconstructed=reconstructed,
            unknown_count=unknown_count,
            observed_start_ns=observed_start_ns,
            observed_end_ns=observed_end_ns,
            observed_end_source=observed_end_source,
            system_prompt_captured=False,
        ),
    )


def build_fixed_trace(
    *,
    topic: OrganizerTopic,
    events: Sequence[Mapping[str, object]],
    run_record: Mapping[str, object],
    system_prompt: str,
    user_prompt: str,
    documents: Sequence[Mapping[str, object]],
    session_id: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> TraceBundle:
    """Build a fixed-retrieval trace without collapsing evidence or prompts."""
    (
        native_spans,
        timestamps,
        reconstructed,
        unknown_count,
        observed_start_ns,
        observed_end_ns,
        observed_end_source,
    ) = _native_spans(
        topic=topic, events=events
    )
    assistant_spans = [span for _, span in native_spans if span.kind == "LLM"]
    failure_children = [
        span
        for _, span in native_spans
        if span.kind != "LLM" and span.status == "ERROR"
    ]
    attempt_children = [
        replace(
            span,
            name="Pi assistant attempt",
            kind="CHAIN",
            attributes={**span.attributes, "pi.original.span.kind": "LLM"},
        )
        for span in assistant_spans[:-1]
    ]
    generation_children = attempt_children + failure_children
    base_ns = timestamps[0] if timestamps else time.time_ns()
    generation_start = (
        assistant_spans[-1].start_ns if assistant_spans else base_ns + 2
    )
    generation_end = (
        assistant_spans[-1].end_ns
        if assistant_spans
        else generation_start + 1
    )
    if generation_children:
        generation_start = min(
            generation_start, *(span.start_ns for span in generation_children)
        )
        generation_end = max(
            generation_end, *(span.end_ns for span in generation_children)
        )
    generation_message = assistant_spans[-1].output_value if assistant_spans else None
    failed_generation_children = [
        span for span in generation_children if span.status == "ERROR"
    ]
    generation_failed = bool(failed_generation_children)
    generation_status = "ERROR" if generation_failed or not assistant_spans else "OK"
    if failed_generation_children:
        generation_diagnostic = failed_generation_children[0].status_message
    else:
        generation_diagnostic = (
            None
            if assistant_spans
            else "native Pi events contain no final assistant message"
        )

    evidence_start = min(base_ns, generation_start) - 2
    evidence = SpanSpec(
        name="fixed evidence preparation",
        kind="RETRIEVER",
        start_ns=evidence_start,
        end_ns=evidence_start + 1,
        attributes={"retrieval.document_count": len(documents)},
        input_value={"topic_id": topic.topic_id, "narrative": topic.narrative},
        output_value={"documents": [dict(document) for document in documents]},
        status="OK",
    )
    prompts = SpanSpec(
        name="fixed prompt construction",
        kind="CHAIN",
        start_ns=evidence.end_ns,
        end_ns=max(evidence.end_ns + 1, generation_start),
        attributes={"content.capture": "full"},
        input_value={"documents": [dict(document) for document in documents]},
        output_value={"system_prompt": system_prompt, "user_prompt": user_prompt},
        status="OK",
    )
    final_generation_start = max(prompts.end_ns, generation_start)
    final_generation_end = max(prompts.end_ns + 1, generation_end)
    reasoning_children = _reasoning_summary_children(
        generation_message,
        start_ns=final_generation_start,
        end_ns=final_generation_end,
    )
    generation_attributes: dict[str, object] = {
        "pi.event.end_type": "message_end",
        "pi.assistant.attempt_count": len(assistant_spans),
        "trace.end_time.source": (
            assistant_spans[-1].attributes.get("trace.end_time.source", "unknown")
            if assistant_spans
            else "unknown"
        ),
        "trace.end_time.upper_bound": True,
    }
    if reasoning_children:
        generation_attributes.update(
            {
                "pi.reasoning.extracted_to_children": True,
                "pi.reasoning.block_count": len(reasoning_children),
            }
        )
    generation = SpanSpec(
        name="Pi generation",
        kind="LLM",
        start_ns=final_generation_start,
        end_ns=final_generation_end,
        attributes=generation_attributes,
        input_value={"system_prompt": system_prompt, "user_prompt": user_prompt},
        output_value=generation_message,
        status=generation_status,
        status_message=generation_diagnostic,
        children=tuple(generation_children) + reasoning_children,
    )
    validation = _validation_span(
        topic=topic,
        run_record=run_record,
        timestamp_ns=max(observed_start_ns, observed_end_ns - 1),
    )
    children = [evidence, prompts, generation, validation]
    return TraceBundle(
        project_name=project_name,
        session_id=session_id,
        topic_id=topic.topic_id,
        baseline="ragnarok-fixed",
        root=_root_span(
            name="Ragnarok fixed-retrieval baseline",
            topic=topic,
            children=children,
            reconstructed=reconstructed,
            unknown_count=unknown_count,
            observed_start_ns=observed_start_ns,
            observed_end_ns=observed_end_ns,
            observed_end_source=observed_end_source,
            system_prompt_captured=True,
        ),
    )


__all__ = [
    "LoadedPiEvents",
    "SpanSpec",
    "TraceBundle",
    "build_fixed_trace",
    "build_piika_trace",
    "load_pi_events",
    "read_trace_bundle",
    "write_trace_bundle",
]

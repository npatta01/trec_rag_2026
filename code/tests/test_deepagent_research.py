from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from contextlib import contextmanager

import pytest
from deepagents.middleware.subagents import TaskToolSchema
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from trec_rag.deepagent_budget import (
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
from trec_rag.deepagent_evidence import EvidenceCoverageState
from trec_rag.deepagent_research import (
    BundleEvidence,
    EvidenceBundle,
    MainToolFilterMiddleware,
    ResearchTaskBudgetMiddleware,
    ResearchTaskEnvelope,
    ResearcherToolFilterMiddleware,
    bind_research_task,
    build_research_subagent,
    current_research_task,
)


ALL_TOOLS = (
    "task",
    "search_climbmix",
    "extract_relevant_snippets",
    "view_retrieval_state",
    "update_retrieval_state",
    "complete_research_round",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",
    "write_todos",
)


def budget_payload() -> dict[str, object]:
    return BudgetSnapshot(
        elapsed_seconds=0.0,
        remaining_researchers=10,
        remaining_rounds=4,
        remaining_retrieval_calls=100,
        active_researchers=0,
        completed_researchers=0,
        completed_rounds=0,
        soft_deadline_reached=False,
        hard_deadline_reached=False,
        stop_code=None,
    ).as_dict()


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def visible_tools(
    middleware: object,
    names: Sequence[str],
    *,
    envelope: ResearchTaskEnvelope | None = None,
) -> set[str]:
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": name} for name in names],
        model_settings={"parallel_tool_calls": True},
    )
    seen: list[str] = []

    def handler(filtered: ModelRequest[object]) -> AIMessage:
        seen.extend(tool["name"] for tool in filtered.tools if isinstance(tool, dict))
        return AIMessage(content="ok")

    if envelope is None:
        middleware.wrap_model_call(request, handler)  # type: ignore[attr-defined]
    else:
        with bind_research_task(envelope):
            middleware.wrap_model_call(request, handler)  # type: ignore[attr-defined]
    return set(seen)


def task_request(
    *, description: str, subagent_type: str | None = "researcher"
) -> ToolCallRequest:
    args: dict[str, object] = {"description": description}
    if subagent_type is not None:
        args["subagent_type"] = subagent_type
    return ToolCallRequest(
        tool_call={
            "name": "task",
            "args": args,
            "id": "task-call",
        },
        tool=None,
        state={},
        runtime=None,
    )


def task_description(
    *,
    research_task_id: str = "R1-N1",
    depth: str = "survey",
) -> str:
    return json.dumps(
        {
            "research_task_id": research_task_id,
            "round_index": 1,
            "depth": depth,
            "motivating_ids": ["N1"],
            "goal": "Find evidence for N1.",
        }
    )


def test_research_bundle_carries_citations_not_transcribed_evidence() -> None:
    bundle = EvidenceBundle.model_validate(
        {
            "research_task_id": "R1-N1",
            "round_index": 1,
            "depth": "survey",
            "motivating_need_ids": ["N1"],
            "candidate_nuggets": [
                {
                    "claim": "Claim.",
                    "need_ids": ["N1"],
                    "facet_ids": [],
                    "evidence": [{"cite": "S3.2"}],
                    "contradicts_claims": [],
                }
            ],
            "conflicts": [],
            "unresolved_gaps": [],
            "suggested_followups": [],
            "stopping_reason": "goal_satisfied",
            "budget_snapshot": budget_payload(),
        }
    )

    assert bundle.candidate_nuggets[0].evidence[0].cite == "S3.2"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BundleEvidence.model_validate(
            {"cite": "S3.2", "quote": "a researcher may not transcribe evidence"}
        )


def test_strict_task_envelope_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResearchTaskEnvelope.model_validate(
            {**json.loads(task_description()), "query": "must be researcher-owned"}
        )


def test_role_filters_expose_only_approved_tools() -> None:
    assert visible_tools(MainToolFilterMiddleware(), ALL_TOOLS) == {
        "task",
        "view_retrieval_state",
        "update_retrieval_state",
        "complete_research_round",
        "read_file",
    }
    assert visible_tools(ResearcherToolFilterMiddleware(), ALL_TOOLS) == {
        "search_climbmix",
        "extract_relevant_snippets",
        "view_retrieval_state",
        "read_file",
    }


def test_main_filter_forces_research_after_empty_round_attempt() -> None:
    config = ResearchBudgetConfig()
    budget = ResearchBudget(config)
    assert budget.authorize_round_completion(1).code == "ROUND_RESEARCH_REQUIRED"
    middleware = MainToolFilterMiddleware(budget, config)
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": name} for name in ALL_TOOLS],
        model_settings={"parallel_tool_calls": True},
        state={"run_model_call_count": 2},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest) -> ModelResponse:
        observed["tools"] = [_tool["name"] for _tool in filtered.tools]
        observed["settings"] = filtered.model_settings
        observed["system"] = filtered.system_message
        observed["tool_choice"] = filtered.tool_choice
        return ModelResponse(result=[AIMessage(content="unused")])

    middleware.wrap_model_call(request, handler)

    assert observed["tools"] == ["task"]
    assert observed["settings"]["parallel_tool_calls"] is False
    assert observed["tool_choice"] == "task"
    assert "Round 1 cannot close" in str(observed["system"])


def observe_turn(
    middleware: MainToolFilterMiddleware, run_model_call_count: int = 2
) -> dict[str, object]:
    """Run one coordinator model turn and capture what the middleware allowed."""
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": name} for name in ALL_TOOLS],
        model_settings={"parallel_tool_calls": True},
        state={"run_model_call_count": run_model_call_count},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest) -> ModelResponse:
        observed["tools"] = [_tool["name"] for _tool in filtered.tools]
        observed["tool_choice"] = filtered.tool_choice
        observed["system"] = str(filtered.system_message)
        observed["settings"] = filtered.model_settings
        return ModelResponse(result=[AIMessage(content="unused")])

    middleware.wrap_model_call(request, handler)
    return observed


def researched_round_budget(
    config: ResearchBudgetConfig,
) -> ResearchBudget:
    """Return a budget whose round 1 has finished research but was never closed."""
    budget = ResearchBudget(config)
    context = ResearchTaskContext("R1-N1", 1, "focused", ("N1",))
    assert budget.reserve_task(context).ok
    budget.finish_task(context)
    return budget


def test_main_filter_compels_merge_then_close_for_a_researched_round() -> None:
    config = ResearchBudgetConfig()
    budget = researched_round_budget(config)
    middleware = MainToolFilterMiddleware(budget, config)

    merge_turn = observe_turn(middleware)

    assert merge_turn["tools"] == ["update_retrieval_state"]
    assert merge_turn["tool_choice"] == "update_retrieval_state"
    assert merge_turn["settings"]["parallel_tool_calls"] is False
    assert "Round 1 research has finished" in str(merge_turn["system"])

    close_turn = observe_turn(middleware)

    assert close_turn["tools"] == ["complete_research_round"]
    assert close_turn["tool_choice"] == "complete_research_round"
    assert "complete_research_round with round_index=1" in str(close_turn["system"])


def test_main_filter_releases_the_coordinator_once_the_round_is_closed() -> None:
    config = ResearchBudgetConfig()
    budget = researched_round_budget(config)
    middleware = MainToolFilterMiddleware(budget, config)
    observe_turn(middleware)
    observe_turn(middleware)

    assert budget.authorize_round_completion(1).ok
    assert budget.complete_round(1, EvidenceCoverageState("Why?").report()).ok

    released = observe_turn(middleware)

    assert set(released["tools"]) == {
        "task",
        "view_retrieval_state",
        "update_retrieval_state",
        "complete_research_round",
        "read_file",
    }
    assert released["tool_choice"] is None


def test_a_stopped_run_still_gets_to_merge_and_close_its_last_round() -> None:
    config = ResearchBudgetConfig(max_researcher_invocations=1)
    budget = researched_round_budget(config)
    middleware = MainToolFilterMiddleware(budget, config)

    assert budget.snapshot().stop_code == "TASK_BUDGET_EXHAUSTED"

    merge_turn = observe_turn(middleware)
    assert merge_turn["tools"] == ["update_retrieval_state"], (
        "a stopped run must still merge the bundles its researchers returned"
    )

    close_turn = observe_turn(middleware)
    assert close_turn["tools"] == ["complete_research_round"]

    assert budget.authorize_round_completion(1).ok
    budget.complete_round(1, EvidenceCoverageState("Why?").report())

    wrap_up = observe_turn(middleware)
    assert wrap_up["tools"] == [], "with nothing left to record, the run ends"


def test_main_filter_stops_compelling_closure_on_the_final_turn() -> None:
    config = ResearchBudgetConfig(max_main_models=4)
    budget = researched_round_budget(config)
    middleware = MainToolFilterMiddleware(budget, config)

    final = observe_turn(middleware, run_model_call_count=3)

    assert final["tools"] == []
    assert final["tool_choice"] is None
    assert "Return the grounded partial result" in str(final["system"])


def test_merge_turn_demands_need_status_updates() -> None:
    config = ResearchBudgetConfig()
    middleware = MainToolFilterMiddleware(researched_round_budget(config), config)

    merge_turn = observe_turn(middleware)

    system = str(merge_turn["system"])
    assert "set_need_status" in system
    assert "partial" in system
    assert "answerable" in system


def test_main_filter_records_the_model_ceiling_that_ends_the_run() -> None:
    config = ResearchBudgetConfig(max_main_models=4)
    budget = ResearchBudget(config)
    middleware = MainToolFilterMiddleware(budget, config)

    assert budget.snapshot().stop_code is None

    observe_turn(middleware, run_model_call_count=3)

    assert budget.snapshot().stop_code == "MAIN_MODEL_BUDGET_EXHAUSTED", (
        "a ceiling-terminated run must not report itself as agent_completed"
    )


def test_main_filter_leaves_stop_code_alone_below_the_model_ceiling() -> None:
    config = ResearchBudgetConfig(max_main_models=8)
    budget = ResearchBudget(config)
    middleware = MainToolFilterMiddleware(budget, config)

    observe_turn(middleware, run_model_call_count=2)

    assert budget.snapshot().stop_code is None


def test_task_schema_has_exactly_description_and_subagent_type() -> None:
    assert set(TaskToolSchema.model_fields) == {"description", "subagent_type"}


def test_task_middleware_binds_context_appends_snapshot_and_releases_task() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    middleware = ResearchTaskBudgetMiddleware(budget)
    seen: list[ResearchTaskEnvelope | None] = []

    def handler(_request: ToolCallRequest) -> ToolMessage:
        task = current_research_task()
        seen.append(task)
        return ToolMessage(content="bundle", tool_call_id="task-call")

    result = middleware.wrap_tool_call(task_request(description=task_description()), handler)

    assert isinstance(result, ToolMessage)
    assert seen == [ResearchTaskEnvelope.model_validate_json(task_description())]
    assert current_research_task() is None
    assert budget.snapshot().active_researchers == 0
    assert budget.snapshot().completed_researchers == 1
    assert '"active_researchers":1' in result.content


def test_task_middleware_records_compact_task_budget_outcome() -> None:
    records: list[dict[str, object]] = []

    class Span:
        def record_budget_outcome(self, **kwargs: object) -> None:
            records.append(kwargs)

    class Tracing:
        @contextmanager
        def researcher_task_span(
            self, task_id: str, round_index: int, depth: str
        ):
            assert (task_id, round_index, depth) == ("R1-N1", 1, "survey")
            yield Span()

    middleware = ResearchTaskBudgetMiddleware(
        ResearchBudget(ResearchBudgetConfig()), tracing=Tracing()
    )
    middleware.wrap_tool_call(
        task_request(description=task_description()),
        lambda _request: ToolMessage(content="bundle", tool_call_id="task-call"),
    )

    assert records[-1]["code"] == "OK"
    assert records[-1]["must_stop"] is False
    assert records[-1]["snapshot"]["completed_researchers"] == 1


@pytest.mark.parametrize("subagent_type", ["general-purpose", None])
def test_task_middleware_denies_a_non_researcher_without_killing_the_run(
    subagent_type: str | None,
) -> None:
    middleware = ResearchTaskBudgetMiddleware(ResearchBudget(ResearchBudgetConfig()))
    dispatched: list[str] = []

    message = middleware.wrap_tool_call(
        task_request(description=task_description(), subagent_type=subagent_type),
        lambda _request: dispatched.append("ran")
        or ToolMessage(content="unexpected", tool_call_id="task-call"),
    )

    payload = json.loads(message.content)
    assert dispatched == [], "the denied subagent must never execute"
    assert payload["code"] == "RESEARCHER_TYPE_DENIED"
    assert payload["ok"] is False
    assert payload["must_stop"] is False, "the coordinator can still retry"
    assert payload["required_subagent_type"] == "researcher"
    assert message.status == "error"


def test_denied_subagent_leaves_the_researcher_budget_untouched() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget)

    middleware.wrap_tool_call(
        task_request(description=task_description(), subagent_type="general-purpose"),
        lambda _request: ToolMessage(content="unexpected", tool_call_id="task-call"),
    )

    snapshot = budget.snapshot()
    assert snapshot.remaining_researchers == ResearchBudgetConfig().max_researcher_invocations
    assert snapshot.stop_code is None


@pytest.mark.parametrize("is_async", [False, True])
def test_task_middleware_returns_recoverable_error_for_prose_description(
    is_async: bool,
) -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget)
    request = task_request(description="Research N1 and find evidence.")

    if is_async:

        async def async_handler(_request: ToolCallRequest) -> ToolMessage:
            pytest.fail("invalid task must not run")

        result = asyncio.run(middleware.awrap_tool_call(request, async_handler))
    else:

        def sync_handler(_request: ToolCallRequest) -> ToolMessage:
            pytest.fail("invalid task must not run")

        result = middleware.wrap_tool_call(request, sync_handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["code"] == "INVALID_RESEARCH_TASK"
    assert payload["must_stop"] is False
    assert payload["description_format"]["research_task_id"] == "R1-N1"
    assert budget.snapshot().remaining_researchers == 10
    assert budget.snapshot().active_researchers == 0


def test_task_middleware_accepts_json_envelope_followed_by_research_instructions() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget)
    description = (
        task_description(depth="focused")
        + "\nResearch the stated gap. Rephrase the query if the first search is weak."
    )
    seen: list[ResearchTaskEnvelope | None] = []

    def handler(_request: ToolCallRequest) -> ToolMessage:
        seen.append(current_research_task())
        return ToolMessage(content="bundle", tool_call_id="task-call")

    result = middleware.wrap_tool_call(
        task_request(description=description),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert seen == [
        ResearchTaskEnvelope.model_validate_json(task_description(depth="focused"))
    ]
    assert budget.snapshot().completed_researchers == 1


def test_invalid_task_records_a_rejected_dispatch_without_task_content() -> None:
    outcomes: list[dict[str, object]] = []

    class Span:
        def record_outcome(self, **kwargs: object) -> None:
            outcomes.append(kwargs)

    class Tracing:
        @contextmanager
        def researcher_dispatch_span(self):
            yield Span()

    middleware = ResearchTaskBudgetMiddleware(
        ResearchBudget(ResearchBudgetConfig()), tracing=Tracing()
    )

    result = middleware.wrap_tool_call(
        task_request(description="Research N1 and find evidence."),
        lambda _request: pytest.fail("invalid task must not run"),
    )

    assert isinstance(result, ToolMessage)
    assert outcomes == [
        {
            "code": "INVALID_RESEARCH_TASK",
            "outcome": "rejected",
            "research_task_id": None,
            "round_index": None,
            "depth": None,
        }
    ]


def test_task_budget_refusal_preserves_a_transient_concurrency_decision() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    assert budget.reserve_task(ResearchTaskContext("existing", 1, "survey", ("N1",))).ok
    dispatch_outcomes: list[dict[str, object]] = []

    class DispatchSpan:
        def record_outcome(self, **kwargs: object) -> None:
            dispatch_outcomes.append(kwargs)

    class Tracing:
        @contextmanager
        def researcher_dispatch_span(self):
            yield DispatchSpan()

        @contextmanager
        def researcher_task_span(self, *_args: object):
            pytest.fail("refused dispatch must not create a researcher AGENT span")
            yield

    middleware = ResearchTaskBudgetMiddleware(budget, tracing=Tracing())

    result = middleware.wrap_tool_call(
        task_request(description=task_description()),
        lambda _request: pytest.fail("refused task must not invoke its handler"),
    )

    assert isinstance(result, ToolMessage)
    payload = json.loads(result.content)
    assert payload["code"] == "CONCURRENCY_BUDGET_EXHAUSTED"
    assert payload["must_stop"] is False
    assert dispatch_outcomes == [
        {
            "code": "CONCURRENCY_BUDGET_EXHAUSTED",
            "outcome": "refused",
            "research_task_id": "R1-N1",
            "round_index": 1,
            "depth": "survey",
        }
    ]


def test_task_middleware_leaves_non_task_calls_unchanged() -> None:
    middleware = ResearchTaskBudgetMiddleware(ResearchBudget(ResearchBudgetConfig()))
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {}, "id": "read-call"},
        tool=None,
        state={},
        runtime=None,
    )
    result = middleware.wrap_tool_call(
        request, lambda _request: ToolMessage(content="file", tool_call_id="read-call")
    )

    assert isinstance(result, ToolMessage)
    assert result.content == "file"


def test_async_task_middleware_matches_sync_behavior() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget)

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        assert current_research_task() is not None
        return ToolMessage(content="bundle", tool_call_id="task-call")

    result = asyncio.run(
        middleware.awrap_tool_call(task_request(description=task_description()), handler)
    )

    assert isinstance(result, ToolMessage)
    assert budget.snapshot().completed_researchers == 1


def test_task_middleware_refuses_new_survey_after_soft_deadline() -> None:
    clock = FakeClock()
    budget = ResearchBudget(ResearchBudgetConfig(soft_seconds=1), clock=clock)
    middleware = ResearchTaskBudgetMiddleware(budget)
    clock.advance(1)

    result = middleware.wrap_tool_call(
        task_request(description=task_description()),
        lambda _request: pytest.fail("soft-deadline survey must not run"),
    )

    assert isinstance(result, ToolMessage)
    payload = json.loads(result.content)
    assert payload["code"] == "SOFT_DEADLINE_REACHED"
    assert payload["must_stop"] is False
    assert budget.snapshot().active_researchers == 0
    assert budget.snapshot().stop_code is None


def test_async_task_middleware_allows_focused_work_after_soft_deadline() -> None:
    clock = FakeClock()
    budget = ResearchBudget(ResearchBudgetConfig(soft_seconds=1), clock=clock)
    middleware = ResearchTaskBudgetMiddleware(budget)
    clock.advance(1)

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        assert current_research_task() is not None
        return ToolMessage(content="bundle", tool_call_id="task-call")

    result = asyncio.run(
        middleware.awrap_tool_call(
            task_request(description=task_description(depth="focused")),
            handler,
        )
    )

    assert isinstance(result, ToolMessage)
    assert budget.snapshot().completed_researchers == 1
    assert budget.snapshot().stop_code is None


@pytest.mark.parametrize("is_async", [False, True])
def test_task_middleware_converts_ordinary_handler_failure_and_releases_slot(
    is_async: bool,
) -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    middleware = ResearchTaskBudgetMiddleware(budget)
    request = task_request(description=task_description())

    if is_async:

        async def async_handler(_request: ToolCallRequest) -> ToolMessage:
            raise RuntimeError("private provider detail")

        result = asyncio.run(middleware.awrap_tool_call(request, async_handler))
    else:

        def sync_handler(_request: ToolCallRequest) -> ToolMessage:
            raise RuntimeError("private provider detail")

        result = middleware.wrap_tool_call(request, sync_handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert set(payload) == {
        "budget_snapshot",
        "code",
        "must_stop",
        "research_task_id",
    }
    assert payload["code"] == "RESEARCH_TASK_FAILED"
    assert payload["must_stop"] is False
    assert payload["research_task_id"] == "R1-N1"
    assert payload["budget_snapshot"]["active_researchers"] == 1
    assert payload["budget_snapshot"]["completed_researchers"] == 0
    assert payload["budget_snapshot"]["stop_code"] is None
    assert "private provider detail" not in result.content
    assert current_research_task() is None
    assert budget.snapshot().active_researchers == 0
    assert budget.snapshot().completed_researchers == 1

    sibling = middleware.wrap_tool_call(
        task_request(
            description=task_description(
                research_task_id="R1-N2",
                depth="focused",
            )
        ),
        lambda _request: ToolMessage(content="bundle", tool_call_id="task-call"),
    )
    assert isinstance(sibling, ToolMessage)
    assert sibling.status == "success"
    assert str(sibling.content).startswith("bundle")


def test_task_middleware_does_not_catch_base_exception() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget)

    with pytest.raises(KeyboardInterrupt):
        middleware.wrap_tool_call(
            task_request(description=task_description()),
            lambda _request: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    assert current_research_task() is None
    assert budget.snapshot().active_researchers == 0


def test_researcher_filter_forces_bundle_after_task_local_no_yield_stop() -> None:
    config = ResearchBudgetConfig(no_yield_calls=3)
    budget = ResearchBudget(config)
    context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))
    envelope = ResearchTaskEnvelope.model_validate_json(task_description())
    assert budget.reserve_task(context).ok
    for _ in range(3):
        assert budget.reserve_retrieval(context, "search_climbmix").ok
        budget.record_yield(context, ())
    middleware = ResearcherToolFilterMiddleware(budget)
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": name} for name in ALL_TOOLS],
        model_settings={"parallel_tool_calls": True},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest[object]) -> AIMessage:
        observed["tools"] = filtered.tools
        observed["system"] = (
            filtered.system_message.content if filtered.system_message else ""
        )
        return AIMessage(content="bundle")

    with bind_research_task(envelope):
        middleware.wrap_model_call(request, handler)

    assert observed["tools"] == []
    assert "structured EvidenceBundle immediately" in str(observed["system"])
    assert "must_stop" in str(observed["system"])


def test_researcher_filter_mechanically_forces_search_then_snippets() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    context = ResearchTaskContext("R1-N1", 1, "focused", ("N1",))
    envelope = ResearchTaskEnvelope.model_validate_json(
        task_description(depth="focused")
    )
    assert budget.reserve_task(context).ok
    middleware = ResearcherToolFilterMiddleware(budget)
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": name} for name in ALL_TOOLS],
        model_settings={"parallel_tool_calls": True},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest) -> ModelResponse:
        observed["tools"] = [_tool["name"] for _tool in filtered.tools]
        observed["settings"] = filtered.model_settings
        observed["response_format"] = filtered.response_format
        observed["system"] = filtered.system_message
        observed["tool_choice"] = filtered.tool_choice
        return ModelResponse(result=[AIMessage(content="unused")])

    with bind_research_task(envelope):
        middleware.wrap_model_call(request, handler)

    assert observed["tools"] == ["search_climbmix"]
    assert observed["settings"]["parallel_tool_calls"] is False
    assert observed["response_format"] is None
    assert observed["tool_choice"] == "search_climbmix"
    assert "first action" in str(observed["system"])

    assert budget.reserve_retrieval(context, "search_climbmix").ok
    with bind_research_task(envelope):
        middleware.wrap_model_call(request, handler)
    assert observed["tools"] == ["extract_relevant_snippets"]
    assert observed["settings"]["parallel_tool_calls"] is False
    assert observed["response_format"] is None
    assert observed["tool_choice"] == "extract_relevant_snippets"
    assert "most relevant document" in str(observed["system"])

    assert budget.reserve_retrieval(context, "extract_relevant_snippets").ok
    with bind_research_task(envelope):
        middleware.wrap_model_call(request, handler)
    assert set(observed["tools"]) == {
        "search_climbmix",
        "extract_relevant_snippets",
        "view_retrieval_state",
        "read_file",
    }


def test_researcher_filter_stop_is_task_local_under_concurrent_contexts() -> None:
    config = ResearchBudgetConfig(no_yield_calls=1, max_concurrent=2)
    budget = ResearchBudget(config)
    stopped_context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))
    active_context = ResearchTaskContext("R1-N2", 1, "focused", ("N1",))
    assert budget.reserve_task(stopped_context).ok
    assert budget.reserve_task(active_context).ok
    assert budget.reserve_retrieval(stopped_context, "search_climbmix").ok
    budget.record_yield(stopped_context, ())
    middleware = ResearcherToolFilterMiddleware(budget)

    stopped = visible_tools(
        middleware,
        ALL_TOOLS,
        envelope=ResearchTaskEnvelope.model_validate_json(
            task_description(research_task_id="R1-N1")
        ),
    )
    active = visible_tools(
        middleware,
        ALL_TOOLS,
        envelope=ResearchTaskEnvelope.model_validate_json(
            task_description(research_task_id="R1-N2", depth="focused")
        ),
    )

    assert stopped == set()
    assert active == {"search_climbmix"}


def test_async_task_middleware_traces_compact_outcome_and_cleans_up() -> None:
    created: list[tuple[str, int, str]] = []
    outcomes: list[dict[str, object]] = []
    closed: list[bool] = []

    class Span:
        def record_budget_outcome(self, **kwargs: object) -> None:
            outcomes.append(kwargs)

    class Tracing:
        @contextmanager
        def researcher_task_span(
            self, task_id: str, round_index: int, depth: str
        ):
            created.append((task_id, round_index, depth))
            try:
                yield Span()
            finally:
                closed.append(True)

    budget = ResearchBudget(ResearchBudgetConfig())
    middleware = ResearchTaskBudgetMiddleware(budget, tracing=Tracing())

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        assert current_research_task() == ResearchTaskEnvelope.model_validate_json(
            task_description()
        )
        return ToolMessage(content="bundle", tool_call_id="task-call")

    result = asyncio.run(
        middleware.awrap_tool_call(task_request(description=task_description()), handler)
    )

    assert isinstance(result, ToolMessage)
    assert created == [("R1-N1", 1, "survey")]
    assert closed == [True]
    assert outcomes[-1]["code"] == "OK"
    assert outcomes[-1]["must_stop"] is False
    assert outcomes[-1]["snapshot"]["completed_researchers"] == 1
    assert current_research_task() is None
    assert budget.snapshot().active_researchers == 0


def test_researcher_spec_keeps_the_main_model_and_excludes_todos() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
    tools: list[Callable[..., object]] = []
    budget = ResearchBudget(ResearchBudgetConfig())
    spec = build_research_subagent(
        model=model,
        tools=tools,
        budget=budget,
        budget_config=ResearchBudgetConfig(),
    )

    assert spec["model"] is model
    assert spec["response_format"] is EvidenceBundle
    assert all(type(item).__name__ != "TodoListMiddleware" for item in spec["middleware"])
    assert "task" not in visible_tools(spec["middleware"][-1], ALL_TOOLS)

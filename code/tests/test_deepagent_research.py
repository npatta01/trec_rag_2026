from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from contextlib import contextmanager

import pytest
from deepagents.middleware.subagents import TaskToolSchema
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from trec_rag.deepagent_budget import (
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
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


def task_request(*, description: str, subagent_type: str = "researcher") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {"description": description, "subagent_type": subagent_type},
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


def test_research_bundle_requires_exact_evidence_coordinates() -> None:
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
                    "evidence": [
                        {
                            "document_id": "D1",
                            "snippet_id": "S1",
                            "page_index": 0,
                            "quote": "Exact quote.",
                        }
                    ],
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

    assert bundle.candidate_nuggets[0].evidence[0].snippet_id == "S1"
    with pytest.raises(ValidationError):
        BundleEvidence.model_validate(
            {"document_id": "D1", "snippet_id": "", "page_index": -1, "quote": ""}
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


def test_task_middleware_requires_the_researcher_subagent() -> None:
    middleware = ResearchTaskBudgetMiddleware(ResearchBudget(ResearchBudgetConfig()))

    with pytest.raises(PermissionError, match="researcher"):
        middleware.wrap_tool_call(
            task_request(description=task_description(), subagent_type="general-purpose"),
            lambda _request: ToolMessage(content="unexpected", tool_call_id="task-call"),
        )


def test_task_budget_refusal_preserves_a_transient_concurrency_decision() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    assert budget.reserve_task(ResearchTaskContext("existing", 1, "survey", ("N1",))).ok
    middleware = ResearchTaskBudgetMiddleware(budget)

    result = middleware.wrap_tool_call(
        task_request(description=task_description()),
        lambda _request: pytest.fail("refused task must not invoke its handler"),
    )

    assert isinstance(result, ToolMessage)
    payload = json.loads(result.content)
    assert payload["code"] == "CONCURRENCY_BUDGET_EXHAUSTED"
    assert payload["must_stop"] is False


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
    assert active == {
        "search_climbmix",
        "extract_relevant_snippets",
        "view_retrieval_state",
        "read_file",
    }


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

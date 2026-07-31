"""Bounded researcher schemas and middleware for Deep Agent retrieval."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
from typing import Any, Literal, cast

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from trec_rag.deepagent_budget import (
    BudgetDecision,
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
)


RESEARCHER_SYSTEM_PROMPT = """You are a bounded retrieval researcher.
Research only the stated gap. Generate and refine your own queries, then inspect
returned snippets for exact supporting quotes. Use only search_climbmix,
extract_relevant_snippets, view_retrieval_state, and read_file. Do not delegate,
write state, or use filesystem mutation tools. Return a compact EvidenceBundle:
each candidate claim must cite exact document, snippet, page, and quote
coordinates; report conflicts and remaining gaps rather than inventing support.
"""


class ResearchTaskEnvelope(BaseModel):
    """The validated compact task description passed to one researcher."""

    model_config = ConfigDict(extra="forbid")

    research_task_id: str = Field(min_length=1)
    round_index: int = Field(ge=1)
    depth: Literal["survey", "focused", "deep"]
    motivating_ids: list[str] = Field(min_length=1)
    goal: str = Field(min_length=1)
    known_evidence: str = ""
    remaining_gap: str = ""


class BudgetSnapshotModel(BaseModel):
    """JSON-compatible representation of the invocation budget."""

    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float
    remaining_researchers: int
    remaining_rounds: int
    remaining_retrieval_calls: int
    active_researchers: int
    completed_researchers: int
    completed_rounds: int
    soft_deadline_reached: bool
    hard_deadline_reached: bool
    stop_code: str | None


class BundleEvidence(BaseModel):
    """One exact snippet coordinate supporting a candidate nugget."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    snippet_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    quote: str = Field(min_length=1)


class CandidateNugget(BaseModel):
    """A compact candidate claim with the evidence needed to validate it."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    need_ids: list[str] = Field(min_length=1)
    facet_ids: list[str]
    evidence: list[BundleEvidence] = Field(min_length=1)
    contradicts_claims: list[str]


class EvidenceBundle(BaseModel):
    """The structured, evidence-grounded result returned by a researcher."""

    model_config = ConfigDict(extra="forbid")

    research_task_id: str
    round_index: int
    depth: Literal["survey", "focused", "deep"]
    motivating_need_ids: list[str]
    candidate_nuggets: list[CandidateNugget]
    conflicts: list[str]
    unresolved_gaps: list[str]
    suggested_followups: list[str]
    stopping_reason: Literal[
        "goal_satisfied", "evidence_saturated", "budget_exhausted", "failed"
    ]
    budget_snapshot: BudgetSnapshotModel


_research_task: ContextVar[ResearchTaskEnvelope | None] = ContextVar(
    "research_task", default=None
)


@contextmanager
def bind_research_task(envelope: ResearchTaskEnvelope):
    """Bind one validated task envelope for fixed tools during a task call."""

    token = _research_task.set(envelope)
    try:
        yield envelope
    finally:
        _research_task.reset(token)


def current_research_task() -> ResearchTaskEnvelope | None:
    """Return the task currently bound to this execution context, if any."""

    return _research_task.get()


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, Mapping):
        name = tool.get("name")
    else:
        name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class _RoleToolFilterMiddleware(AgentMiddleware):
    """Expose only a role's fixed tools and reject bypassed tool calls."""

    _ALLOWED_TOOLS: frozenset[str] = frozenset()
    _PARALLEL_TOOL_CALLS = False

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        return request.override(
            tools=[
                tool for tool in request.tools if _tool_name(tool) in self._ALLOWED_TOOLS
            ],
            model_settings={
                **(request.model_settings or {}),
                "parallel_tool_calls": self._PARALLEL_TOOL_CALLS,
            },
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return handler(self._filter_tools(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return await handler(self._filter_tools(request))

    def _require_allowed_tool(self, request: ToolCallRequest) -> None:
        if request.tool_call.get("name") not in self._ALLOWED_TOOLS:
            raise PermissionError("role-restricted tool access denied")

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        self._require_allowed_tool(request)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        self._require_allowed_tool(request)
        return await handler(request)


class MainToolFilterMiddleware(_RoleToolFilterMiddleware):
    """Limit the coordinator to delegation, semantic state, and spill reads."""

    _ALLOWED_TOOLS = frozenset(
        {
            "task",
            "view_retrieval_state",
            "update_retrieval_state",
            "complete_research_round",
            "read_file",
        }
    )
    _PARALLEL_TOOL_CALLS = True


class ResearcherToolFilterMiddleware(_RoleToolFilterMiddleware):
    """Limit a researcher to retrieval, compact state inspection, and spill reads."""

    _ALLOWED_TOOLS = frozenset(
        {
            "search_climbmix",
            "extract_relevant_snippets",
            "view_retrieval_state",
            "read_file",
        }
    )
    _PARALLEL_TOOL_CALLS = False


class ResearchTaskBudgetMiddleware(AgentMiddleware):
    """Reserve and bind each researcher task for one top-level invocation."""

    def __init__(self, budget: ResearchBudget) -> None:
        self._budget = budget

    @staticmethod
    def _envelope(request: ToolCallRequest) -> ResearchTaskEnvelope:
        args = request.tool_call.get("args")
        if not isinstance(args, Mapping):
            raise ValueError("task arguments must be an object")
        if args.get("subagent_type") != "researcher":
            raise PermissionError("only the researcher subagent is permitted")
        description = args.get("description")
        if not isinstance(description, str):
            raise ValueError("task description must be compact JSON")
        return ResearchTaskEnvelope.model_validate_json(description)

    @staticmethod
    def _context(envelope: ResearchTaskEnvelope):
        from trec_rag.deepagent_budget import ResearchTaskContext

        return ResearchTaskContext(
            research_task_id=envelope.research_task_id,
            round_index=envelope.round_index,
            depth=envelope.depth,
            motivating_ids=tuple(envelope.motivating_ids),
        )

    @staticmethod
    def _compact_snapshot(snapshot: BudgetSnapshot) -> str:
        return json.dumps(snapshot.as_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def _append_snapshot(cls, result: Any, snapshot: BudgetSnapshot) -> Any:
        if isinstance(result, ToolMessage):
            return result.model_copy(
                update={"content": cls._content_with_snapshot(result.content, snapshot)}
            )
        if isinstance(result, Command) and isinstance(result.update, Mapping):
            messages = result.update.get("messages")
            if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
                updated_messages = [
                    cls._append_snapshot(message, snapshot)
                    if isinstance(message, ToolMessage)
                    else message
                    for message in messages
                ]
                return replace(result, update={**result.update, "messages": updated_messages})
        return result

    @classmethod
    def _content_with_snapshot(
        cls, content: str | list[str | dict[Any, Any]], snapshot: BudgetSnapshot
    ) -> str | list[str | dict[Any, Any]]:
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return f"{content}\n\nbudget_snapshot={cls._compact_snapshot(snapshot)}"
            if isinstance(payload, dict):
                payload["budget_snapshot"] = snapshot.as_dict()
                return json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return content

    def _refusal(self, request: ToolCallRequest, decision: BudgetDecision) -> ToolMessage:
        tool_call_id = request.tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("task call requires an id")
        return ToolMessage(
            content=json.dumps(
                {
                    "budget_snapshot": decision.snapshot.as_dict(),
                    "code": decision.code,
                    "must_stop": decision.must_stop,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            tool_call_id=tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        if request.tool_call.get("name") != "task":
            return handler(request)
        envelope = self._envelope(request)
        context = self._context(envelope)
        decision = self._budget.reserve_task(context)
        if not decision.ok:
            return self._refusal(request, decision)
        try:
            with bind_research_task(envelope):
                result = handler(request)
            return self._append_snapshot(result, self._budget.snapshot())
        finally:
            self._budget.finish_task(context)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        if request.tool_call.get("name") != "task":
            return await handler(request)
        envelope = self._envelope(request)
        context = self._context(envelope)
        decision = self._budget.reserve_task(context)
        if not decision.ok:
            return self._refusal(request, decision)
        try:
            with bind_research_task(envelope):
                result = await handler(request)
            return self._append_snapshot(result, self._budget.snapshot())
        finally:
            self._budget.finish_task(context)


def build_research_subagent(
    *,
    model: BaseChatModel,
    tools: Sequence[Callable[..., object]],
    budget_config: ResearchBudgetConfig,
) -> SubAgent:
    """Build the only delegable, non-recursive retrieval researcher."""

    return cast(
        SubAgent,
        {
            "name": "researcher",
            "description": "Research one stated retrieval gap; refine queries autonomously.",
            "system_prompt": RESEARCHER_SYSTEM_PROMPT,
            "tools": list(tools),
            "model": model,
            "middleware": [
                ModelCallLimitMiddleware(
                    run_limit=budget_config.max_models_per_researcher,
                    exit_behavior="end",
                ),
                ToolCallLimitMiddleware(
                    run_limit=budget_config.max_tools_per_researcher,
                    exit_behavior="continue",
                ),
                ToolCallLimitMiddleware(
                    tool_name="search_climbmix",
                    run_limit=budget_config.max_searches_per_researcher,
                    exit_behavior="continue",
                ),
                ToolCallLimitMiddleware(
                    tool_name="extract_relevant_snippets",
                    run_limit=budget_config.max_snippets_per_researcher,
                    exit_behavior="continue",
                ),
                ResearcherToolFilterMiddleware(),
            ],
            "response_format": EvidenceBundle,
        },
    )

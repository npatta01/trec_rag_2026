"""Bounded researcher schemas and middleware for Deep Agent retrieval."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
from threading import Lock
from typing import Any, Literal, cast

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from trec_rag.deepagent_budget import (
    BudgetDecision,
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)


RESEARCHER_SYSTEM_PROMPT = """You are a bounded retrieval researcher.
Research only the stated gap. Generate and refine your own queries, then inspect
returned snippets for exact supporting quotes. Use only search_climbmix,
extract_relevant_snippets, view_retrieval_state, and read_file. Do not delegate,
write state, or use filesystem mutation tools. Return a compact EvidenceBundle.
Producing candidate nuggets is the job. Emit one for every claim the returned
snippets actually support, including partial ones that only cover part of your
gap. Use unresolved_gaps for what the snippets genuinely could not answer, not
as a substitute for reporting what they did. Returning an empty
candidate_nuggets list means the snippets supported nothing at all, which is
rare once a search has returned relevant passages.
Support each claim by citing snippet handles, never by writing out quotes.
extract_relevant_snippets gives each snippet a "cite" value such as S3 and
numbers its sentences. Cite "S3" for a whole snippet, "S3.2" for its sentence 2,
or "S3.2-4" for sentences 2 through 4. Cite the smallest range that carries the
claim. Handles stay valid for the rest of this run, so a handle from an earlier
page is still citable. The system fills in the document, page, and quote text.
Your first action must be search_climbmix. From its returned documents, your
next action must call extract_relevant_snippets on the most relevant document.
Do not inspect state, read spill files, or return EvidenceBundle until you have
attempted both steps, unless a budget response says must_stop=true. Afterward,
refine queries and inspect more documents or snippet pages as the evidence gap
requires; do not claim evidence that you have not inspected.
If any tool response says must_stop=true, call no more tools and immediately
return the structured EvidenceBundle with grounded work completed so far.
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
    """One citation into a snippet already returned during this invocation."""

    model_config = ConfigDict(extra="forbid")

    cite: str = Field(
        min_length=2,
        description=(
            'A snippet handle from extract_relevant_snippets: "S3" for the '
            'whole snippet, "S3.2" for its sentence 2, or "S3.2-4" for '
            "sentences 2 through 4. Never write out the quote itself."
        ),
    )


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

    def __init__(
        self,
        budget: ResearchBudget | None = None,
        budget_config: ResearchBudgetConfig | None = None,
    ) -> None:
        self._budget = budget
        self._budget_config = budget_config
        self._closure_lock = Lock()
        self._merge_turns_granted: set[int] = set()

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        filtered = super()._filter_tools(request)
        if self._budget is None or self._budget_config is None:
            return filtered
        run_count = request.state.get("run_model_call_count", 0)
        stop_code = self._budget.snapshot().stop_code
        model_limit_reached = run_count >= self._budget_config.max_main_models - 1
        if model_limit_reached and stop_code is None:
            # The ceiling is ending this run, so the result must not read as a
            # voluntary agent_completed.
            self._budget.note_main_model_exhausted()
        final_turn = stop_code is not None or model_limit_reached
        if not final_turn:
            required_round = self._budget.required_research_round()
            if required_round is not None:
                return self._directed(
                    filtered,
                    ["task"],
                    "task",
                    f"Round {required_round} cannot close without a completed "
                    'researcher. Your next action must call task with subagent_type='
                    f'"researcher" and round_index={required_round} in its JSON '
                    "description.",
                )
            pending_round = self._budget.pending_round_closure()
            if pending_round is not None:
                return self._directed(filtered, *self._closure_directive(pending_round))
            return filtered
        return self._directed(
            filtered,
            [],
            None,
            "Return the grounded partial result immediately. Do not call more tools.",
        )

    def _closure_directive(self, round_index: int) -> tuple[list[str], str, str]:
        """Grant one merge turn per open round, then compel its explicit close."""
        with self._closure_lock:
            merge_turn_used = round_index in self._merge_turns_granted
            self._merge_turns_granted.add(round_index)
        if not merge_turn_used:
            return (
                ["update_retrieval_state"],
                "update_retrieval_state",
                f"Round {round_index} research has finished. Merge every returned "
                "evidence bundle now with exactly one batched "
                "update_retrieval_state delta. That same delta must also carry a "
                "set_need_status row for every need whose evidence changed: use "
                '"partial" with an updated remaining_gap when grounded nuggets '
                'exist but the need is not fully answered, and "answerable" only '
                "with a draft_answer and grounded draft_nugget_ids. A need left "
                "unaddressed after its researchers returned is a reporting error.",
            )
        return (
            ["complete_research_round"],
            "complete_research_round",
            f"Round {round_index} is merged. Call complete_research_round with "
            f"round_index={round_index} now; no other action may precede it.",
        )

    def _directed(
        self,
        request: ModelRequest,
        tool_names: Sequence[str],
        tool_choice: str | None,
        instruction: str,
    ) -> ModelRequest:
        """Restrict this turn to one instructed action the coordinator cannot skip."""
        allowed = set(tool_names)
        existing = request.system_message
        content = f"{existing.content}\n\n{instruction}" if existing else instruction
        overrides: dict[str, Any] = {
            "tools": [
                tool for tool in request.tools if _tool_name(tool) in allowed
            ],
            "system_message": SystemMessage(content=content),
            "model_settings": {
                **(request.model_settings or {}),
                "parallel_tool_calls": False,
            },
        }
        if tool_choice is not None:
            overrides["tool_choice"] = tool_choice
        return request.override(**overrides)


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

    def __init__(self, budget: ResearchBudget | None = None) -> None:
        self._budget = budget

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        filtered = super()._filter_tools(request)
        envelope = current_research_task()
        if self._budget is None or envelope is None:
            return filtered
        context = ResearchTaskContext(
            research_task_id=envelope.research_task_id,
            round_index=envelope.round_index,
            depth=envelope.depth,
            motivating_ids=tuple(envelope.motivating_ids),
        )
        snapshot = self._budget.snapshot()
        task_stop_code = self._budget.task_stop_code(context)
        if task_stop_code is not None or snapshot.stop_code is not None:
            stop_code = task_stop_code or snapshot.stop_code
            instruction = (
                "A budget response now has must_stop=true "
                f"({stop_code}). Return the structured EvidenceBundle immediately "
                "with grounded work completed so far; do not call more tools."
            )
            existing = request.system_message
            content = (
                f"{existing.content}\n\n{instruction}" if existing else instruction
            )
            return filtered.override(
                tools=[],
                system_message=SystemMessage(content=content),
                model_settings={
                    **(filtered.model_settings or {}),
                    "parallel_tool_calls": False,
                },
            )
        if not self._budget.task_has_search_attempt(context):
            instruction = (
                "Your first action for this task must be search_climbmix. "
                "Do not view state, read files, or return EvidenceBundle yet."
            )
            existing = request.system_message
            content = (
                f"{existing.content}\n\n{instruction}" if existing else instruction
            )
            return filtered.override(
                tools=[
                    tool
                    for tool in filtered.tools
                    if _tool_name(tool) == "search_climbmix"
                ],
                response_format=None,
                system_message=SystemMessage(content=content),
                tool_choice="search_climbmix",
                model_settings={
                    **(filtered.model_settings or {}),
                    "parallel_tool_calls": False,
                },
            )
        if not self._budget.task_has_snippet_attempt(context):
            instruction = (
                "Use the search result you just received. Select its most relevant "
                "document_id and call extract_relevant_snippets now for the stated "
                "research gap. Do not return EvidenceBundle yet."
            )
            existing = request.system_message
            content = (
                f"{existing.content}\n\n{instruction}" if existing else instruction
            )
            return filtered.override(
                tools=[
                    tool
                    for tool in filtered.tools
                    if _tool_name(tool) == "extract_relevant_snippets"
                ],
                response_format=None,
                system_message=SystemMessage(content=content),
                tool_choice="extract_relevant_snippets",
                model_settings={
                    **(filtered.model_settings or {}),
                    "parallel_tool_calls": False,
                },
            )
        return filtered


class ResearchTaskBudgetMiddleware(AgentMiddleware):
    """Reserve and bind each researcher task for one top-level invocation."""

    def __init__(self, budget: ResearchBudget, *, tracing: object | None = None) -> None:
        self._budget = budget
        self._tracing = tracing

    @contextmanager
    def _trace_task(
        self, context: ResearchTaskContext
    ) -> Any:
        tracing = self._tracing
        if tracing is None:
            yield None
            return
        try:
            manager = tracing.researcher_task_span(
                context.research_task_id, context.round_index, context.depth
            )
            span = manager.__enter__()
        except Exception:
            yield None
            return
        try:
            yield span
        except BaseException as exc:
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass

    @contextmanager
    def _trace_dispatch(self) -> Any:
        tracing = self._tracing
        if tracing is None:
            yield None
            return
        try:
            manager = tracing.researcher_dispatch_span()
            span = manager.__enter__()
        except Exception:
            yield None
            return
        try:
            yield span
        except BaseException as exc:
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass

    @staticmethod
    def _record_trace_outcome(
        span: object | None, decision: BudgetDecision, snapshot: BudgetSnapshot
    ) -> None:
        if span is None:
            return
        try:
            span.record_budget_outcome(
                code=decision.code,
                must_stop=decision.must_stop or snapshot.stop_code is not None,
                snapshot=snapshot.as_dict(),
            )
        except Exception:
            pass

    @staticmethod
    def _record_dispatch_outcome(
        span: object | None,
        *,
        code: str,
        outcome: str,
        context: ResearchTaskContext | None = None,
    ) -> None:
        if span is None:
            return
        try:
            span.record_outcome(
                code=code,
                outcome=outcome,
                research_task_id=(context.research_task_id if context else None),
                round_index=(context.round_index if context else None),
                depth=(context.depth if context else None),
            )
        except Exception:
            pass

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
        stripped = description.lstrip()
        try:
            payload, end_index = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            raise ValueError("task description must begin with compact JSON") from None
        suffix = stripped[end_index:]
        if suffix and not suffix[0].isspace():
            raise ValueError("task instructions must follow JSON after whitespace")
        return ResearchTaskEnvelope.model_validate(payload)

    @staticmethod
    def _context(envelope: ResearchTaskEnvelope):
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

    def _invalid_task(self, request: ToolCallRequest) -> ToolMessage:
        tool_call_id = request.tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("task call requires an id")
        return ToolMessage(
            content=json.dumps(
                {
                    "budget_snapshot": self._budget.snapshot().as_dict(),
                    "code": "INVALID_RESEARCH_TASK",
                    "description_format": {
                        "research_task_id": "R1-N1",
                        "round_index": 1,
                        "depth": "focused",
                        "motivating_ids": ["N1"],
                        "goal": "Find grounded evidence for N1",
                        "known_evidence": "",
                        "remaining_gap": "No grounded evidence yet",
                    },
                    "description_protocol": (
                        "Begin with the JSON object; optional research instructions "
                        "may follow after a newline"
                    ),
                    "must_stop": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            tool_call_id=tool_call_id,
            status="error",
        )

    @staticmethod
    def _task_failure(
        request: ToolCallRequest,
        envelope: ResearchTaskEnvelope,
        snapshot: BudgetSnapshot,
    ) -> ToolMessage:
        tool_call_id = request.tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("task call requires an id")
        return ToolMessage(
            content=json.dumps(
                {
                    "budget_snapshot": snapshot.as_dict(),
                    "code": "RESEARCH_TASK_FAILED",
                    "must_stop": False,
                    "research_task_id": envelope.research_task_id,
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
        with self._trace_dispatch() as dispatch_span:
            try:
                envelope = self._envelope(request)
            except PermissionError:
                self._record_dispatch_outcome(
                    dispatch_span, code="RESEARCHER_TYPE_DENIED", outcome="rejected"
                )
                raise
            except ValueError:
                self._record_dispatch_outcome(
                    dispatch_span, code="INVALID_RESEARCH_TASK", outcome="rejected"
                )
                return self._invalid_task(request)
            context = self._context(envelope)
            decision = self._budget.reserve_task(context)
            if not decision.ok:
                self._record_dispatch_outcome(
                    dispatch_span,
                    code=decision.code,
                    outcome="refused",
                    context=context,
                )
                return self._refusal(request, decision)
            with self._trace_task(context) as span:
                self._record_trace_outcome(span, decision, decision.snapshot)
                try:
                    with bind_research_task(envelope):
                        try:
                            result = handler(request)
                        except Exception:
                            self._record_dispatch_outcome(
                                dispatch_span,
                                code="RESEARCH_TASK_FAILED",
                                outcome="failed",
                                context=context,
                            )
                            return self._task_failure(
                                request, envelope, self._budget.snapshot()
                            )
                    completed = self._append_snapshot(result, self._budget.snapshot())
                    self._record_dispatch_outcome(
                        dispatch_span,
                        code=decision.code,
                        outcome="completed",
                        context=context,
                    )
                    return completed
                finally:
                    self._budget.finish_task(context)
                    self._record_trace_outcome(span, decision, self._budget.snapshot())

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        if request.tool_call.get("name") != "task":
            return await handler(request)
        with self._trace_dispatch() as dispatch_span:
            try:
                envelope = self._envelope(request)
            except PermissionError:
                self._record_dispatch_outcome(
                    dispatch_span, code="RESEARCHER_TYPE_DENIED", outcome="rejected"
                )
                raise
            except ValueError:
                self._record_dispatch_outcome(
                    dispatch_span, code="INVALID_RESEARCH_TASK", outcome="rejected"
                )
                return self._invalid_task(request)
            context = self._context(envelope)
            decision = self._budget.reserve_task(context)
            if not decision.ok:
                self._record_dispatch_outcome(
                    dispatch_span,
                    code=decision.code,
                    outcome="refused",
                    context=context,
                )
                return self._refusal(request, decision)
            with self._trace_task(context) as span:
                self._record_trace_outcome(span, decision, decision.snapshot)
                try:
                    with bind_research_task(envelope):
                        try:
                            result = await handler(request)
                        except Exception:
                            self._record_dispatch_outcome(
                                dispatch_span,
                                code="RESEARCH_TASK_FAILED",
                                outcome="failed",
                                context=context,
                            )
                            return self._task_failure(
                                request, envelope, self._budget.snapshot()
                            )
                    completed = self._append_snapshot(result, self._budget.snapshot())
                    self._record_dispatch_outcome(
                        dispatch_span,
                        code=decision.code,
                        outcome="completed",
                        context=context,
                    )
                    return completed
                finally:
                    self._budget.finish_task(context)
                    self._record_trace_outcome(span, decision, self._budget.snapshot())


def build_research_subagent(
    *,
    model: BaseChatModel,
    tools: Sequence[Callable[..., object]],
    budget: ResearchBudget,
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
                ResearcherToolFilterMiddleware(budget),
            ],
            "response_format": EvidenceBundle,
        },
    )

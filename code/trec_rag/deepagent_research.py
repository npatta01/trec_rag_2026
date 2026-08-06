"""Bounded researcher schemas and middleware for Deep Agent retrieval."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
from threading import Lock
from typing import Any, Literal, NoReturn, cast

from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain.agents.structured_output import ToolStrategy
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
returned snippets for exact supporting quotes. Use only search_passages,
view_retrieval_state, and read_file. Do not delegate,
write state, or use filesystem mutation tools. Return a compact EvidenceBundle.
Producing candidate nuggets is the job. Emit one for every claim the returned
snippets actually support, including partial ones that only cover part of your
gap. Use unresolved_gaps for what the snippets genuinely could not answer, not
as a substitute for reporting what they did. Returning an empty
candidate_nuggets list means the snippets supported nothing at all, which is
rare once a search has returned relevant passages.
Return candidate_nuggets in your own order of importance, most important
first. Ordering is all that is asked of you: which claims are essential is
decided later, by the coordinator, which can compare across every need.
Support each claim by citing snippet handles, never by writing out quotes.
search_passages gives each passage a "cite"
value such as S3 and number its sentences. Cite "S3" for a whole snippet, "S3.2" for its sentence 2,
or "S3.2-4" for sentences 2 through 4. Cite the smallest range that carries the
claim. Handles stay valid for the rest of this run, so a handle from an earlier
page is still citable. The system fills in the document, page, and quote text.
Your first action must be search_passages. It returns the best passages from
across many documents, already selected for you: the passages you receive were
chosen by relevance and spread across sources before you saw them, so you do
not pick which document to read. Cite from them directly.
Do not inspect state, read spill files, or return EvidenceBundle until you have
searched, unless a budget response says must_stop=true. Afterward, refine
queries and search again as the evidence gap requires.
A claim supported by two independent documents is worth more than one supported
by two passages of the same document, so prefer citing across sources.
If any tool response says must_stop=true, call no more tools and immediately
return the structured EvidenceBundle with grounded work completed so far.
"""


_CLOSEOUT_DIRECTIVE = (
    "Research is over; no further researchers can run. Write up what was "
    "actually found, in one update_retrieval_state delta. Give every need "
    "with grounded nuggets a set_need_status row carrying draft_nugget_ids "
    "listing the few best of them: use \"answerable\" with a draft_answer "
    "where the evidence answers the need, and \"partial\" with an honest "
    "remaining_gap where it does not. The selection is what matters and is "
    "capped, so choose deliberately. Claim nothing the nuggets do not support. "
    "After recording the write-up, call complete_retrieval as the only terminal "
    "action."
)


class CloseoutAttemptsExhausted(RuntimeError):
    """The coordinator provider failed on every directed closeout attempt."""


def is_operational_provider_stop(exc: Exception) -> bool:
    """Classify retryable provider/transport stops without swallowing code bugs."""
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code == 429 or status_code >= 500
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    module = type(exc).__module__.split(".", 1)[0]
    name = type(exc).__name__.lower()
    if module in {
        "httpcore",
        "httpx",
        "langchain_openrouter",
        "openai",
        "openrouter",
        "requests",
    } and any(
        marker in name
        for marker in (
            "connect",
            "connection",
            "network",
            "noresponse",
            "ratelimit",
            "serviceunavailable",
            "timeout",
            "transport",
        )
    ):
        return True
    if isinstance(exc, RuntimeError):
        detail = str(exc).lower()
        return any(
            marker in detail
            for marker in (
                "provider unavailable",
                "provider timeout",
                "rate limit",
                "service unavailable",
            )
        )
    return False


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
            'A snippet handle from search_passages: "S3" for the '
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


def _has_tool_calls(response: object) -> bool:
    """Whether a model response asked for any tool."""
    inner = getattr(response, "model_response", response)
    messages = getattr(inner, "result", None)
    if messages is None:
        messages = [inner]
    return any(getattr(message, "tool_calls", None) for message in messages)


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, Mapping):
        name = tool.get("name")
    else:
        name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _failure_reason(failure: BaseException | None) -> str:
    """Classify a researcher failure without echoing provider text.

    The coordinator needs to tell an outage from a silent model, but raw
    exception messages carry provider internals into agent context and traces,
    so the message is read to classify and never repeated.
    """
    if failure is None:
        return "no_bundle_returned"
    if isinstance(failure, Exception) and is_operational_provider_stop(failure):
        return "retrieval_unavailable"
    name = type(failure).__name__
    detail = str(failure)
    if name in {"RemotePyseriniThrottled", "RetrievalTransportError"} or (
        "continuation required" in detail
    ):
        return "retrieval_unavailable"
    if "StructuredOutput" in name or "Validation" in name:
        return "invalid_structured_output"
    return "researcher_error"


def _is_recoverable_researcher_failure(failure: Exception) -> bool:
    """Keep only expected outage/structured-output failures inside the graph."""
    return _failure_reason(failure) in {
        "retrieval_unavailable",
        "invalid_structured_output",
    }


class _RoleToolFilterMiddleware(AgentMiddleware):
    """Expose only a role's fixed tools and reject bypassed tool calls."""

    _ALLOWED_TOOLS: frozenset[str] = frozenset()
    _PARALLEL_TOOL_CALLS = False
    _DENIED_MESSAGE = "role-restricted tool access denied"

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
            raise PermissionError(self._DENIED_MESSAGE)

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
            "complete_retrieval",
            "complete_research_round",
            "read_file",
        }
    )
    _PARALLEL_TOOL_CALLS = True

    def __init__(
        self,
        budget: ResearchBudget | None = None,
        budget_config: ResearchBudgetConfig | None = None,
        *,
        closeout_pending: Callable[[], bool] | None = None,
    ) -> None:
        self._budget = budget
        # None disables the bounce, so a bare middleware behaves as before.
        self._closeout_pending = closeout_pending
        self._budget_config = budget_config
        self._closure_lock = Lock()
        self._merge_turns_granted: set[int] = set()
        self._closeout_lock = Lock()
        # Voluntary-exit bounces and end-of-budget synthesis are two entrances
        # to the same closeout. They therefore share one two-attempt budget.
        self._closeout_attempts = 0

    def wrap_model_call(self, request: ModelRequest, handler):
        """Give an unfinished closeout at most two directed provider calls.

        A coordinator that replies with no tool call ends the run, and nothing
        guarded that path: two live runs finished with every need holding
        grounded nuggets and an empty draft selection, which zeroes the
        submission ranker's highest-weighted feature. Retrying the handler is
        the documented contract for this middleware hook, and this middleware
        is innermost, so the discarded reply is invisible to everything else.

        Voluntary-exit bounces and budget-directed synthesis share the same
        two attempts. A tool-calling response returns immediately; the live
        predicate decides on the next graph turn whether that update succeeded.
        """
        filtered = self._filter_tools(request)
        if self._is_directed_closeout(filtered):
            return self._invoke_closeout(filtered, handler)
        response = handler(filtered)
        bounced = self._closeout_bounce(filtered, response)
        if bounced is None:
            return response
        return self._invoke_closeout(bounced, handler)

    async def awrap_model_call(self, request: ModelRequest, handler):
        filtered = self._filter_tools(request)
        if self._is_directed_closeout(filtered):
            return await self._ainvoke_closeout(filtered, handler)
        response = await handler(filtered)
        bounced = self._closeout_bounce(filtered, response)
        if bounced is None:
            return response
        return await self._ainvoke_closeout(bounced, handler)

    def _invoke_closeout(self, request: ModelRequest, handler: Callable) -> Any:
        """Run one claimed closeout attempt and its sole allowed retry."""
        try:
            response = handler(request)
        except Exception as exc:
            retry = self._retry_after_closeout_failure(request, exc)
        else:
            retry = self._retry_after_invalid_closeout(request, response)
            if retry is None:
                return response
        try:
            response = handler(retry)
        except Exception as exc:
            self._raise_failed_closeout(exc)
        if not _has_tool_calls(response):
            self._note_closeout_refused()
        return response

    async def _ainvoke_closeout(
        self, request: ModelRequest, handler: Callable
    ) -> Any:
        """Async equivalent of _invoke_closeout."""
        try:
            response = await handler(request)
        except Exception as exc:
            retry = self._retry_after_closeout_failure(request, exc)
        else:
            retry = self._retry_after_invalid_closeout(request, response)
            if retry is None:
                return response
        try:
            response = await handler(retry)
        except Exception as exc:
            self._raise_failed_closeout(exc)
        if not _has_tool_calls(response):
            self._note_closeout_refused()
        return response

    def _retry_after_closeout_failure(
        self, request: ModelRequest, exc: Exception
    ) -> ModelRequest:
        if not is_operational_provider_stop(exc):
            raise exc
        retry = self._closeout_retry(request)
        if retry is None:
            self._raise_failed_closeout(exc)
        return retry

    def _retry_after_invalid_closeout(
        self, request: ModelRequest, response: object
    ) -> ModelRequest | None:
        if _has_tool_calls(response):
            return None
        retry = self._closeout_retry(request)
        if retry is None:
            self._note_closeout_refused()
        return retry

    def _raise_failed_closeout(self, exc: Exception) -> NoReturn:
        if not is_operational_provider_stop(exc):
            raise exc
        self._note_closeout_refused()
        raise CloseoutAttemptsExhausted(
            "directed closeout provider attempts exhausted"
        ) from exc

    def _is_directed_closeout(self, request: ModelRequest) -> bool:
        """Recognize only requests constructed from the closeout directive."""
        message = request.system_message
        content = getattr(message, "content", None)
        return (
            request.tool_choice == "update_retrieval_state"
            and isinstance(content, str)
            and _CLOSEOUT_DIRECTIVE in content
        )

    def _note_closeout_refused(self) -> None:
        if self._budget is not None:
            self._budget.note_closeout_refused()

    def _claim_closeout_attempt(self) -> int | None:
        """Claim one of the two closeout attempts shared by every entrance."""
        with self._closeout_lock:
            if self._closeout_attempts >= 2:
                return None
            self._closeout_attempts += 1
            return self._closeout_attempts

    def _closeout_retry(self, request: ModelRequest) -> ModelRequest | None:
        attempt = self._claim_closeout_attempt()
        if attempt is None:
            return None
        return self._directed(
            request,
            ["update_retrieval_state"],
            "update_retrieval_state",
            self._closeout_instruction(attempt),
        )

    @staticmethod
    def _closeout_instruction(attempt: int) -> str:
        if attempt == 1:
            return _CLOSEOUT_DIRECTIVE
        return (
            _CLOSEOUT_DIRECTIVE
            + " The first closeout attempt failed or was not accepted: needs "
            "that hold grounded evidence still have no recorded selection. "
            "Record it now with valid nugget ids; this is the final attempt."
        )

    def _closeout_bounce(
        self, filtered: ModelRequest, response: object
    ) -> ModelRequest | None:
        """Decide whether this reply is an unaccounted exit worth bouncing."""
        if self._closeout_pending is None:
            return None
        # Only the free phase. Every directed branch sets tool_choice and the
        # terminal branch empties the tool list; a silent reply on either is
        # not a voluntary exit.
        if filtered.tool_choice is not None or not filtered.tools:
            return None
        if _has_tool_calls(response):
            return None
        try:
            if not self._closeout_pending():
                return None
        except Exception:
            return None
        attempt = self._claim_closeout_attempt()
        if attempt is None:
            return None
        if self._budget is not None:
            self._budget.note_exit_bounced()
        return self._directed(
            filtered,
            ["update_retrieval_state"],
            "update_retrieval_state",
            self._closeout_instruction(attempt)
            + " You replied without calling a tool while needs that hold "
            "grounded evidence still have no recorded selection. Record that "
            "closeout now; the run continues afterwards.",
        )

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        filtered = super()._filter_tools(request)
        if self._budget is None or self._budget_config is None:
            return filtered
        run_count = request.state.get("run_model_call_count", 0)
        stop_code = self._budget.snapshot().stop_code
        model_limit_reached = run_count >= self._budget_config.max_main_models - 1
        # Research stops early enough to leave the reserve for writing up, so
        # synthesis is reached by running out of research turns and not only by
        # a stop code arriving first.
        research_turns_spent = run_count >= max(
            0,
            self._budget_config.max_main_models
            - 1
            - self._budget_config.synthesis_reserve_turns,
        )
        if model_limit_reached and stop_code is None:
            # The ceiling is ending this run, so the result must not read as a
            # voluntary agent_completed.
            self._budget.note_main_model_exhausted()
        if not model_limit_reached:
            if stop_code is None:
                required_round = self._budget.required_research_round()
                if required_round is not None:
                    return self._directed(
                        filtered,
                        ["task"],
                        "task",
                        f"Round {required_round} cannot close without a completed "
                        "researcher. Your next action must call task with "
                        f'subagent_type="researcher" and round_index='
                        f"{required_round} in its JSON description.",
                    )
            # Finalising an already-researched round outranks the run stop, so a
            # stopped run still merges and records its last batch.
            pending_round = self._budget.pending_round_closure()
            if pending_round is not None:
                return self._directed(filtered, *self._closure_directive(pending_round))
            if stop_code is None and not research_turns_spent:
                return filtered
            synthesis = self._synthesis_directive()
            if synthesis is not None:
                return self._directed(
                    filtered,
                    ["update_retrieval_state"],
                    "update_retrieval_state",
                    synthesis,
                )
        return self._directed(
            filtered,
            [],
            None,
            "Return the grounded partial result immediately. Do not call more tools.",
        )

    def _synthesis_directive(self) -> str | None:
        """Claim one shared closeout attempt, or None once both are spent.

        The first attempt is unconditional. Later attempts are spent only while
        needs holding grounded evidence still have no recorded selection, so a
        closeout that landed is never asked for twice. The configured reserve
        determines when synthesis starts; the shared closeout bound stays two.
        """
        if self._closeout_attempts and not self._closeout_still_pending():
            return None
        attempt = self._claim_closeout_attempt()
        if attempt is None:
            return None
        return self._closeout_instruction(attempt)

    def _closeout_still_pending(self) -> bool:
        """Whether grounded needs remain unselected; unknown counts as settled."""
        if self._closeout_pending is None:
            return False
        try:
            return bool(self._closeout_pending())
        except Exception:
            return False

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
            "search_passages",
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
                "Your first action for this task must be search_passages. "
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
                    if _tool_name(tool) == "search_passages"
                ],
                response_format=None,
                system_message=SystemMessage(content=content),
                tool_choice="search_passages",
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

    def _denied_subagent(self, request: ToolCallRequest) -> ToolMessage:
        """Refuse a non-researcher dispatch recoverably rather than fatally."""
        tool_call_id = request.tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("task call requires an id")
        return ToolMessage(
            content=json.dumps(
                {
                    "budget_snapshot": self._budget.snapshot().as_dict(),
                    "code": "RESEARCHER_TYPE_DENIED",
                    "must_stop": False,
                    "ok": False,
                    "required_subagent_type": "researcher",
                    "retry_guidance": (
                        'Pass subagent_type exactly as "researcher". No other '
                        "subagent exists, and omitting the field is the same as "
                        "requesting one that does not. This dispatch did not run; "
                        "reissue it with the correct subagent_type."
                    ),
                },
                sort_keys=True,
            ),
            name="task",
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
        failure: BaseException | None = None,
    ) -> ToolMessage:
        """Return an empty but well-formed bundle naming why research failed.

        A bare error tells the coordinator nothing, so an unreachable index and
        a model that produced nothing look identical and its next dispatch is a
        guess. The bundle shape also lets the merge step treat this like any
        other empty result.
        """
        tool_call_id = request.tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("task call requires an id")
        reason = _failure_reason(failure)
        return ToolMessage(
            content=json.dumps(
                {
                    "budget_snapshot": snapshot.as_dict(),
                    "candidate_nuggets": [],
                    "code": "RESEARCH_TASK_FAILED",
                    "failure_reason": reason,
                    "must_stop": False,
                    "research_task_id": envelope.research_task_id,
                    "retry_guidance": (
                        "This researcher returned nothing because it failed, not "
                        "because the need has no evidence. Treat the need as "
                        "still uncovered and dispatch it again."
                    ),
                    "unresolved_gaps": [],
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
                # The dispatch is still denied, but denying it must not destroy
                # the run: a mistyped or omitted subagent_type would otherwise
                # discard every grounded nugget collected so far.
                return self._denied_subagent(request)
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
                        except Exception as exc:
                            if not _is_recoverable_researcher_failure(exc):
                                raise
                            if _failure_reason(exc) == "retrieval_unavailable":
                                self._budget.note_retrieval_unavailable()
                            self._record_dispatch_outcome(
                                dispatch_span,
                                code="RESEARCH_TASK_FAILED",
                                outcome="failed",
                                context=context,
                            )
                            return self._task_failure(
                                request, envelope, self._budget.snapshot(), exc
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
                # The dispatch is still denied, but denying it must not destroy
                # the run: a mistyped or omitted subagent_type would otherwise
                # discard every grounded nugget collected so far.
                return self._denied_subagent(request)
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
                        except Exception as exc:
                            if not _is_recoverable_researcher_failure(exc):
                                raise
                            if _failure_reason(exc) == "retrieval_unavailable":
                                self._budget.note_retrieval_unavailable()
                            self._record_dispatch_outcome(
                                dispatch_span,
                                code="RESEARCH_TASK_FAILED",
                                outcome="failed",
                                context=context,
                            )
                            return self._task_failure(
                                request, envelope, self._budget.snapshot(), exc
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
                    tool_name="search_passages",
                    run_limit=budget_config.max_passage_searches_per_researcher,
                    exit_behavior="continue",
                ),
                ResearcherToolFilterMiddleware(budget),
            ],
            # A researcher whose searches misbehaved has returned an empty
            # completion, which native structured output cannot parse and which
            # kills the task. ToolStrategy feeds the parse failure back so the
            # model can answer again, and says what an honest empty result looks
            # like so "nothing to report" is expressible rather than silent.
            "response_format": ToolStrategy(
                EvidenceBundle,
                handle_errors=(
                    "Your EvidenceBundle did not parse. Return it again as valid "
                    "JSON matching the schema. If you found nothing worth "
                    "claiming, that is a valid answer: return candidate_nuggets "
                    "as an empty list with your unresolved_gaps and a "
                    "stopping_reason. Never return an empty response."
                ),
            ),
        },
    )

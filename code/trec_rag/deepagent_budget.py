"""Invocation-local admission control for bounded Deep Agent research."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from math import isfinite
from threading import Lock
from time import monotonic
from typing import Literal, Protocol

from trec_rag.deepagent_evidence import FacetReport, NeedReport, NuggetReport

ResearchDepth = Literal["survey", "focused", "deep"]
BudgetCode = Literal[
    "OK",
    "SOFT_DEADLINE_REACHED",
    "HARD_DEADLINE_REACHED",
    "TASK_BUDGET_EXHAUSTED",
    "CONCURRENCY_BUDGET_EXHAUSTED",
    "RETRIEVAL_BUDGET_EXHAUSTED",
    "TASK_TOOL_BUDGET_EXHAUSTED",
    "ROUND_RESEARCH_REQUIRED",
    "ROUND_SEQUENCE_INVALID",
    "MAIN_MODEL_BUDGET_EXHAUSTED",
    "RETRIEVAL_UNAVAILABLE",
    "NO_YIELD_STOP",
    "NO_PROGRESS_STOP",
]

_SEARCH_TOOL = "search_climbmix"
_SNIPPET_TOOL = "extract_relevant_snippets"
# A pooled passage search both searches and yields citable snippets, so it
# satisfies both first-action requirements while spending neither budget.
_PASSAGE_TOOL = "search_passages"
_RUN_STOP_PRIORITY: dict[BudgetCode, int] = {
    "TASK_BUDGET_EXHAUSTED": 20,
    "RETRIEVAL_BUDGET_EXHAUSTED": 30,
    "MAIN_MODEL_BUDGET_EXHAUSTED": 40,
    "NO_PROGRESS_STOP": 90,
    "HARD_DEADLINE_REACHED": 100,
}


@dataclass(frozen=True)
class ResearchBudgetConfig:
    """Budget limits, including model limits enforced by later middleware."""

    # Researcher invocations are the only limit on how much research happens.
    # A round cannot close without a finished researcher, so rounds are already
    # bounded by researchers and need no separate cap of their own.
    # Researchers are the binding constraint, measured. The runs that reached
    # full need coverage both ended on TASK_BUDGET_EXHAUSTED with retrieval and
    # wall clock to spare: 10 of 10 researchers spent, 7 retrieval calls left,
    # and the hard deadline never approached. Raising retrieval depth or the
    # deadlines instead does nothing, which one run demonstrated by scoring
    # worse with a longer deadline.
    max_researcher_invocations: int = 20
    # Concurrency is per topic, not across topics. Hosted-search throttling has
    # come from running whole topics back to back, which this does not govern.
    max_concurrent: int = 3
    max_retrieval_calls: int = 100
    max_tools_per_researcher: int = 20
    max_searches_per_researcher: int = 8
    max_snippets_per_researcher: int = 16
    # Its own unit: one pooled scoring pass costs materially more than one
    # snippet page, so it must not draw down the per-snippet allowance.
    max_passage_searches_per_researcher: int = 8
    max_models_per_researcher: int = 30
    # The coordinator spends roughly three turns per round (dispatch, merge,
    # close) plus decomposition and recovery, so this must clear one round
    # per researcher with margin or a productive run is cut off mid-round.
    # Scaled with the researcher count to hold the ratio the 10-researcher runs
    # were measured at. A tracked invariant requires at least three turns per
    # researcher, since a round costs dispatch, merge and close; the remainder
    # is decomposition, recovery and the synthesis reserve. Setting this below
    # the researchers it must serve strands them mid-round.
    max_main_models: int = 80
    # Turns held back so synthesis always gets one. Without a reserve,
    # whether answers get written is a race: a run that hits a stop code
    # first is granted a synthesis turn, and a run that exhausts its
    # coordinator turns first is told to return immediately and writes
    # none. That is why answerable came out 5-of-7 in one run and 0-of-5
    # in the next, with the second holding more evidence than the first.
    synthesis_reserve_turns: int = 2
    # Pacing hosted search every six seconds puts a full run near eleven
    # minutes before any thinking, so the old ten-minute soft deadline fired
    # on healthy runs. These bound a runaway, not a normal one.
    soft_seconds: float = 1800.0
    hard_seconds: float = 3600.0
    no_yield_calls: int = 3
    no_progress_rounds: int = 2

    def __post_init__(self) -> None:
        positive_int_fields = (
            "max_researcher_invocations",
            "max_concurrent",
            "max_retrieval_calls",
            "max_tools_per_researcher",
            "max_searches_per_researcher",
            "max_snippets_per_researcher",
            "max_passage_searches_per_researcher",
            "max_models_per_researcher",
            "max_main_models",
            "synthesis_reserve_turns",
            "no_yield_calls",
            "no_progress_rounds",
        )
        # A reserve at or above the ceiling leaves no research turns at all,
        # and the middleware would compel synthesis on the very first
        # coordinator turn before anything had been researched.
        if self.synthesis_reserve_turns >= self.max_main_models:
            raise ValueError(
                "synthesis_reserve_turns must leave at least one research turn"
            )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in ("soft_seconds", "hard_seconds"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative number")
        if self.hard_seconds < self.soft_seconds:
            raise ValueError("hard_seconds must be greater than or equal to soft_seconds")


@dataclass(frozen=True)
class ResearchTaskContext:
    research_task_id: str
    round_index: int
    depth: ResearchDepth
    motivating_ids: tuple[str, ...]


@dataclass(frozen=True)
class BudgetSnapshot:
    elapsed_seconds: float
    remaining_researchers: int
    remaining_retrieval_calls: int
    active_researchers: int
    completed_researchers: int
    completed_rounds: int
    soft_deadline_reached: bool
    hard_deadline_reached: bool
    stop_code: BudgetCode | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetDecision:
    ok: bool
    code: BudgetCode
    snapshot: BudgetSnapshot
    must_stop: bool = False


class _CoverageReport(Protocol):
    """The accepted-only report surface needed for semantic progress."""

    nuggets: tuple[NuggetReport, ...]
    needs: tuple[NeedReport, ...]
    facets: tuple[FacetReport, ...]


class ResearchBudget:
    """One thread-safe budget instance for a single top-level invocation."""

    def __init__(
        self,
        config: ResearchBudgetConfig,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self._started_at = clock()
        self._lock = Lock()
        self._reserved_researchers = 0
        self._reserved_retrieval_calls = 0
        self._active_tasks: dict[str, ResearchTaskContext] = {}
        self._completed_researchers = 0
        self._completed_researcher_rounds: set[int] = set()
        self._required_research_round: int | None = None
        self._completed_round_indexes: set[int] = set()
        self._task_tool_counts: dict[str, int] = {}
        self._task_search_counts: dict[str, int] = {}
        self._task_snippet_counts: dict[str, int] = {}
        self._task_passage_counts: dict[str, int] = {}
        self._seen_yield_ids: dict[str, set[str]] = {}
        self._no_yield_streaks: dict[str, int] = {}
        self._task_stop_codes: dict[str, BudgetCode] = {}
        self._round_progress: dict[int, tuple[frozenset[str], ...]] = {}
        self._round_decisions: dict[int, BudgetDecision] = {}
        self._no_progress_streak = 0
        self._retrieval_unavailable = False
        self._exit_bounced = False
        self._closeout_refused = False
        self._stop_code: BudgetCode | None = None

    def reserve_task(self, context: ResearchTaskContext) -> BudgetDecision:
        """Reserve one researcher slot if all global admission checks allow it."""
        with self._lock:
            stopping = self._global_stop_code()
            if stopping is not None:
                return self._refusal(stopping, must_stop=True)
            if context.research_task_id in self._active_tasks:
                return self._refusal("CONCURRENCY_BUDGET_EXHAUSTED")
            if (
                context.depth == "survey"
                and self._elapsed_seconds() >= self._config.soft_seconds
            ):
                return self._refusal("SOFT_DEADLINE_REACHED")
            if self._reserved_researchers >= self._config.max_researcher_invocations:
                return self._refusal("TASK_BUDGET_EXHAUSTED", must_stop=True)
            if len(self._active_tasks) >= self._config.max_concurrent:
                return self._refusal("CONCURRENCY_BUDGET_EXHAUSTED")

            self._reserved_researchers += 1
            self._active_tasks[context.research_task_id] = context
            self._task_tool_counts.setdefault(context.research_task_id, 0)
            self._task_search_counts.setdefault(context.research_task_id, 0)
            self._task_snippet_counts.setdefault(context.research_task_id, 0)
            self._task_passage_counts.setdefault(context.research_task_id, 0)
            self._seen_yield_ids.setdefault(context.research_task_id, set())
            self._no_yield_streaks.setdefault(context.research_task_id, 0)
            return self._admission()

    def reserve_retrieval(
        self, context: ResearchTaskContext, tool_name: str
    ) -> BudgetDecision:
        """Reserve a call before downstream tool-argument validation occurs."""
        with self._lock:
            task_id = context.research_task_id
            if self._active_tasks.get(task_id) != context:
                raise ValueError("retrieval requires the exact active task context")
            stopping = self._global_stop_code()
            if stopping is not None:
                return self._refusal(stopping, must_stop=True)
            task_stopping = self._task_stop_codes.get(task_id)
            if task_stopping is not None:
                return self._task_refusal(task_stopping)
            if self._reserved_retrieval_calls >= self._config.max_retrieval_calls:
                return self._refusal(
                    "RETRIEVAL_BUDGET_EXHAUSTED", must_stop=True
                )
            if self._task_tool_counts.get(task_id, 0) >= self._config.max_tools_per_researcher:
                return self._refusal("TASK_TOOL_BUDGET_EXHAUSTED")
            if (
                tool_name == _SEARCH_TOOL
                and self._task_search_counts.get(task_id, 0)
                >= self._config.max_searches_per_researcher
            ):
                return self._refusal("TASK_TOOL_BUDGET_EXHAUSTED")
            if (
                tool_name == _SNIPPET_TOOL
                and self._task_snippet_counts.get(task_id, 0)
                >= self._config.max_snippets_per_researcher
            ):
                return self._refusal("TASK_TOOL_BUDGET_EXHAUSTED")
            if (
                tool_name == _PASSAGE_TOOL
                and self._task_passage_counts.get(task_id, 0)
                >= self._config.max_passage_searches_per_researcher
            ):
                return self._refusal("TASK_TOOL_BUDGET_EXHAUSTED")

            self._reserved_retrieval_calls += 1
            self._task_tool_counts[task_id] = self._task_tool_counts.get(task_id, 0) + 1
            if tool_name == _SEARCH_TOOL:
                self._task_search_counts[task_id] = (
                    self._task_search_counts.get(task_id, 0) + 1
                )
            if tool_name == _SNIPPET_TOOL:
                self._task_snippet_counts[task_id] = (
                    self._task_snippet_counts.get(task_id, 0) + 1
                )
            if tool_name == _PASSAGE_TOOL:
                self._task_passage_counts[task_id] = (
                    self._task_passage_counts.get(task_id, 0) + 1
                )
            if self._reserved_retrieval_calls >= self._config.max_retrieval_calls:
                return self._terminal_admission("RETRIEVAL_BUDGET_EXHAUSTED")
            return self._admission()

    def finish_task(self, context: ResearchTaskContext) -> None:
        """Release a previously reserved task slot; safe to call from ``finally``."""
        with self._lock:
            task_id = context.research_task_id
            active_context = self._active_tasks.get(task_id)
            if active_context is None:
                return
            if active_context != context:
                raise ValueError("finish requires the exact active task context")
            del self._active_tasks[task_id]
            self._completed_researchers += 1
            self._completed_researcher_rounds.add(context.round_index)
            if self._required_research_round == context.round_index:
                self._required_research_round = None
            if (
                self._reserved_researchers
                >= self._config.max_researcher_invocations
                and not self._active_tasks
            ):
                self._persist_stop("TASK_BUDGET_EXHAUSTED")

    def record_yield(
        self, context: ResearchTaskContext, identifiers: Iterable[str]
    ) -> None:
        """Record novel evidence identifiers or extend the zero-yield streak."""
        with self._lock:
            task_id = context.research_task_id
            seen = self._seen_yield_ids.setdefault(task_id, set())
            novel_identifiers = set(identifiers).difference(seen)
            if novel_identifiers:
                seen.update(novel_identifiers)
                self._no_yield_streaks[task_id] = 0
            else:
                self._no_yield_streaks[task_id] = self._no_yield_streaks.get(task_id, 0) + 1
                if self._no_yield_streaks[task_id] >= self._config.no_yield_calls:
                    self._task_stop_codes[task_id] = "NO_YIELD_STOP"

    def task_stop_code(self, context: ResearchTaskContext) -> BudgetCode | None:
        """Return a researcher-local stop without changing the run-wide snapshot."""
        with self._lock:
            return self._task_stop_codes.get(context.research_task_id)

    def task_has_search_attempt(self, context: ResearchTaskContext) -> bool:
        """Return whether this active researcher has attempted its first search."""
        with self._lock:
            if self._active_tasks.get(context.research_task_id) != context:
                return False
            return (
                self._task_search_counts.get(context.research_task_id, 0) > 0
                or self._task_passage_counts.get(context.research_task_id, 0) > 0
            )

    def task_has_snippet_attempt(self, context: ResearchTaskContext) -> bool:
        """Return whether this active researcher has attempted snippet extraction.

        A pooled passage search already returned citable passages, so requiring
        a separate per-document extraction after it would force a step that
        cannot add anything the researcher does not already hold.
        """
        with self._lock:
            if self._active_tasks.get(context.research_task_id) != context:
                return False
            return (
                self._task_snippet_counts.get(context.research_task_id, 0) > 0
                or self._task_passage_counts.get(context.research_task_id, 0) > 0
            )

    def authorize_round_completion(self, round_index: int) -> BudgetDecision:
        """Require one finished researcher before a coordinator can close a round."""
        with self._lock:
            existing_decision = self._round_decisions.get(round_index)
            if existing_decision is not None:
                return self._admission()
            expected_round_index = len(self._round_decisions) + 1
            if round_index != expected_round_index:
                return self._refusal("ROUND_SEQUENCE_INVALID")
            if round_index in self._completed_researcher_rounds:
                # Closing records research that already finished. A run stop
                # forbids starting new work; it must not discard completed work,
                # or the last batch's evidence is lost with the round.
                return self._admission()
            stopping = self._global_stop_code()
            if stopping is not None:
                return self._refusal(stopping, must_stop=True)
            self._required_research_round = round_index
            return self._refusal("ROUND_RESEARCH_REQUIRED")

    def required_research_round(self) -> int | None:
        """Return the round whose empty completion attempt requires delegation."""
        with self._lock:
            return self._required_research_round

    def note_retrieval_unavailable(self) -> None:
        """Record that the index could not be reached, not that nothing was found.

        Deliberately not a run stop. One transport blip must not cancel the
        remaining retrieval; a sustained outage still ends the run through the
        existing no-yield and no-progress guards.
        """
        with self._lock:
            self._retrieval_unavailable = True

    def retrieval_unavailable(self) -> bool:
        """Whether any search failed to reach the index during this invocation."""
        with self._lock:
            return self._retrieval_unavailable

    def note_exit_bounced(self) -> None:
        """Record that a voluntary exit was sent back once for a closeout."""
        with self._lock:
            self._exit_bounced = True

    def exit_bounced(self) -> bool:
        with self._lock:
            return self._exit_bounced

    def note_closeout_refused(self) -> None:
        """Record that the compelled closeout turn also produced no tool call.

        A flag rather than a stop code: it must not cancel or outrank anything,
        only tell downstream that the ledger was never written up.
        """
        with self._lock:
            self._closeout_refused = True

    def closeout_refused(self) -> bool:
        with self._lock:
            return self._closeout_refused

    def note_main_model_exhausted(self) -> None:
        """Record that the coordinator's model-call ceiling, not the agent, ended it."""
        with self._lock:
            self._persist_stop("MAIN_MODEL_BUDGET_EXHAUSTED")

    def pending_round_closure(self) -> int | None:
        """Return the finished-research round the coordinator has not closed yet."""
        with self._lock:
            # Deliberately not gated on the run stop: a stopped run must still
            # record the round its finished researchers already produced.
            if self._active_tasks:
                return None
            expected_round_index = len(self._round_decisions) + 1
            if expected_round_index not in self._completed_researcher_rounds:
                return None
            return expected_round_index

    def complete_round(
        self, round_index: int, report: _CoverageReport
    ) -> BudgetDecision:
        """Record semantic coverage progress and enforce the no-progress stop."""
        with self._lock:
            existing_decision = self._round_decisions.get(round_index)
            if existing_decision is not None:
                return existing_decision
            expected_round_index = len(self._round_decisions) + 1
            if round_index != expected_round_index:
                return self._refusal("ROUND_SEQUENCE_INVALID")

            progress = self._semantic_progress(report)
            previous = self._round_progress.get(round_index - 1, (frozenset(),) * 3)
            made_progress = any(current > prior for current, prior in zip(progress, previous))
            self._round_progress[round_index] = progress
            self._completed_round_indexes.add(round_index)
            if made_progress:
                self._no_progress_streak = 0
            else:
                self._no_progress_streak += 1
            if self._no_progress_streak >= self._config.no_progress_rounds:
                decision = self._refusal("NO_PROGRESS_STOP", must_stop=True)
            else:
                decision = self._admission()
            self._round_decisions[round_index] = decision
            return decision

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable view of the current invocation-local state."""
        with self._lock:
            return self._snapshot()

    def _semantic_progress(
        self, report: _CoverageReport
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        return (
            frozenset(str(item.nugget_id) for item in report.nuggets),
            frozenset(
                str(item.need_id)
                for item in report.needs
                if item.status != "unaddressed"
            ),
            frozenset(
                str(item.facet_id)
                for item in report.facets
                if item.status != "open"
            ),
        )

    def _global_stop_code(self) -> BudgetCode | None:
        if self._elapsed_seconds() >= self._config.hard_seconds:
            self._persist_stop("HARD_DEADLINE_REACHED")
        return self._stop_code

    def _persist_stop(self, code: BudgetCode) -> None:
        """Persist the highest-priority true run-stop decision."""

        current_priority = _RUN_STOP_PRIORITY.get(self._stop_code, -1)
        if _RUN_STOP_PRIORITY.get(code, -1) > current_priority:
            self._stop_code = code

    def _admission(self) -> BudgetDecision:
        code: BudgetCode = "OK"
        if self._elapsed_seconds() >= self._config.soft_seconds:
            code = "SOFT_DEADLINE_REACHED"
        return BudgetDecision(ok=True, code=code, snapshot=self._snapshot())

    def _refusal(self, code: BudgetCode, *, must_stop: bool = False) -> BudgetDecision:
        if must_stop:
            self._persist_stop(code)
        return BudgetDecision(
            ok=False,
            code=code,
            snapshot=self._snapshot(),
            must_stop=must_stop,
        )

    def _task_refusal(self, code: BudgetCode) -> BudgetDecision:
        return BudgetDecision(
            ok=False,
            code=code,
            snapshot=self._snapshot(),
            must_stop=True,
        )

    def _terminal_admission(self, code: BudgetCode) -> BudgetDecision:
        self._persist_stop(code)
        return BudgetDecision(
            ok=True,
            code=code,
            snapshot=self._snapshot(),
            must_stop=True,
        )

    def _snapshot(self) -> BudgetSnapshot:
        elapsed_seconds = self._elapsed_seconds()
        hard_deadline_reached = elapsed_seconds >= self._config.hard_seconds
        if hard_deadline_reached:
            self._persist_stop("HARD_DEADLINE_REACHED")
        return BudgetSnapshot(
            elapsed_seconds=elapsed_seconds,
            remaining_researchers=max(
                0, self._config.max_researcher_invocations - self._reserved_researchers
            ),
            remaining_retrieval_calls=max(
                0, self._config.max_retrieval_calls - self._reserved_retrieval_calls
            ),
            active_researchers=len(self._active_tasks),
            completed_researchers=self._completed_researchers,
            completed_rounds=len(self._completed_round_indexes),
            soft_deadline_reached=elapsed_seconds >= self._config.soft_seconds,
            hard_deadline_reached=hard_deadline_reached,
            stop_code=self._stop_code,
        )

    def _elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

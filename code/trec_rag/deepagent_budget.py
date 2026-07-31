"""Invocation-local admission control for bounded Deep Agent research."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from math import isfinite
from threading import Lock
from time import monotonic
from typing import Literal, Protocol

ResearchDepth = Literal["survey", "focused", "deep"]
BudgetCode = Literal[
    "OK",
    "SOFT_DEADLINE_REACHED",
    "HARD_DEADLINE_REACHED",
    "TASK_BUDGET_EXHAUSTED",
    "ROUND_BUDGET_EXHAUSTED",
    "CONCURRENCY_BUDGET_EXHAUSTED",
    "RETRIEVAL_BUDGET_EXHAUSTED",
    "TASK_TOOL_BUDGET_EXHAUSTED",
    "NO_YIELD_STOP",
    "NO_PROGRESS_STOP",
]

_SEARCH_TOOL = "search_climbmix"
_SNIPPET_TOOL = "extract_relevant_snippets"


@dataclass(frozen=True)
class ResearchBudgetConfig:
    max_researcher_invocations: int = 10
    max_rounds: int = 4
    max_concurrent: int = 3
    max_retrieval_calls: int = 100
    max_tools_per_researcher: int = 20
    max_searches_per_researcher: int = 8
    max_snippets_per_researcher: int = 16
    max_models_per_researcher: int = 30
    max_main_models: int = 25
    soft_seconds: float = 600.0
    hard_seconds: float = 1800.0
    no_yield_calls: int = 3
    no_progress_rounds: int = 2

    def __post_init__(self) -> None:
        positive_int_fields = (
            "max_researcher_invocations",
            "max_rounds",
            "max_concurrent",
            "max_retrieval_calls",
            "max_tools_per_researcher",
            "max_searches_per_researcher",
            "max_snippets_per_researcher",
            "max_models_per_researcher",
            "max_main_models",
            "no_yield_calls",
            "no_progress_rounds",
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
    remaining_rounds: int
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
    nuggets: Iterable[object]
    needs: Iterable[object]
    facets: Iterable[object]


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
        self._active_task_ids: set[str] = set()
        self._completed_researchers = 0
        self._completed_round_indexes: set[int] = set()
        self._task_tool_counts: dict[str, int] = {}
        self._task_search_counts: dict[str, int] = {}
        self._task_snippet_counts: dict[str, int] = {}
        self._seen_yield_ids: dict[str, set[str]] = {}
        self._no_yield_streaks: dict[str, int] = {}
        self._round_progress: dict[int, tuple[frozenset[str], ...]] = {}
        self._no_progress_streak = 0

    def reserve_task(self, context: ResearchTaskContext) -> BudgetDecision:
        """Reserve one researcher slot if all global admission checks allow it."""
        with self._lock:
            stopping = self._global_stop_code()
            if stopping is not None:
                return self._refusal(stopping, must_stop=True)
            if self._reserved_researchers >= self._config.max_researcher_invocations:
                return self._refusal("TASK_BUDGET_EXHAUSTED")
            if context.round_index > self._config.max_rounds:
                return self._refusal("ROUND_BUDGET_EXHAUSTED")
            if len(self._active_task_ids) >= self._config.max_concurrent:
                return self._refusal("CONCURRENCY_BUDGET_EXHAUSTED")

            self._reserved_researchers += 1
            self._active_task_ids.add(context.research_task_id)
            self._task_tool_counts.setdefault(context.research_task_id, 0)
            self._task_search_counts.setdefault(context.research_task_id, 0)
            self._task_snippet_counts.setdefault(context.research_task_id, 0)
            self._seen_yield_ids.setdefault(context.research_task_id, set())
            self._no_yield_streaks.setdefault(context.research_task_id, 0)
            return self._admission()

    def reserve_retrieval(
        self, context: ResearchTaskContext, tool_name: str
    ) -> BudgetDecision:
        """Reserve one retrieval call and its task-local tool allowance."""
        with self._lock:
            stopping = self._global_stop_code()
            if stopping is not None:
                return self._refusal(stopping, must_stop=True)
            task_id = context.research_task_id
            if self._no_yield_streaks.get(task_id, 0) >= self._config.no_yield_calls:
                return self._refusal("NO_YIELD_STOP", must_stop=True)
            if self._reserved_retrieval_calls >= self._config.max_retrieval_calls:
                return self._refusal("RETRIEVAL_BUDGET_EXHAUSTED")
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
            return self._admission()

    def finish_task(self, context: ResearchTaskContext) -> None:
        """Release a previously reserved task slot; safe to call from ``finally``."""
        with self._lock:
            if context.research_task_id in self._active_task_ids:
                self._active_task_ids.remove(context.research_task_id)
                self._completed_researchers += 1

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

    def complete_round(
        self, round_index: int, report: _CoverageReport
    ) -> BudgetDecision:
        """Record semantic coverage progress and enforce the no-progress stop."""
        with self._lock:
            if round_index > self._config.max_rounds:
                return self._refusal("ROUND_BUDGET_EXHAUSTED")

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
                return self._refusal("NO_PROGRESS_STOP", must_stop=True)
            return self._admission()

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
            return "HARD_DEADLINE_REACHED"
        if self._no_progress_streak >= self._config.no_progress_rounds:
            return "NO_PROGRESS_STOP"
        return None

    def _admission(self) -> BudgetDecision:
        code: BudgetCode = "OK"
        if self._elapsed_seconds() >= self._config.soft_seconds:
            code = "SOFT_DEADLINE_REACHED"
        return BudgetDecision(ok=True, code=code, snapshot=self._snapshot())

    def _refusal(self, code: BudgetCode, *, must_stop: bool = False) -> BudgetDecision:
        return BudgetDecision(
            ok=False,
            code=code,
            snapshot=self._snapshot(),
            must_stop=must_stop,
        )

    def _snapshot(self) -> BudgetSnapshot:
        elapsed_seconds = self._elapsed_seconds()
        hard_deadline_reached = elapsed_seconds >= self._config.hard_seconds
        return BudgetSnapshot(
            elapsed_seconds=elapsed_seconds,
            remaining_researchers=max(
                0, self._config.max_researcher_invocations - self._reserved_researchers
            ),
            remaining_rounds=max(
                0, self._config.max_rounds - len(self._completed_round_indexes)
            ),
            remaining_retrieval_calls=max(
                0, self._config.max_retrieval_calls - self._reserved_retrieval_calls
            ),
            active_researchers=len(self._active_task_ids),
            completed_researchers=self._completed_researchers,
            completed_rounds=len(self._completed_round_indexes),
            soft_deadline_reached=elapsed_seconds >= self._config.soft_seconds,
            hard_deadline_reached=hard_deadline_reached,
            stop_code=(
                "HARD_DEADLINE_REACHED"
                if hard_deadline_reached
                else "NO_PROGRESS_STOP"
                if self._no_progress_streak >= self._config.no_progress_rounds
                else None
            ),
        )

    def _elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

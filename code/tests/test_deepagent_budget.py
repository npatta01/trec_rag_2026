from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from trec_rag.deepagent_budget import (
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
from trec_rag.deepagent_evidence import DocumentObservation, EvidenceCoverageState


class FakeClock:
    def __init__(self) -> None:
        self._seconds = 0.0

    def __call__(self) -> float:
        return self._seconds

    def advance(self, seconds: float) -> None:
        self._seconds += seconds


def active_budget() -> tuple[ResearchBudget, ResearchTaskContext]:
    budget = ResearchBudget(ResearchBudgetConfig())
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok
    return budget, context


def test_parallel_task_reservations_never_exceed_concurrency() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(max_researcher_invocations=10, max_concurrent=3)
    )
    contexts = [ResearchTaskContext(f"T{i}", 1, "survey", ("N1",)) for i in range(4)]

    decisions = [budget.reserve_task(context) for context in contexts]

    assert [decision.ok for decision in decisions] == [True, True, True, False]
    assert decisions[-1].code == "CONCURRENCY_BUDGET_EXHAUSTED"


def test_simultaneous_task_reservations_never_exceed_concurrency() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=3))
    barrier = Barrier(4)
    contexts = [ResearchTaskContext(f"T{index}", 1, "survey", ("N1",)) for index in range(4)]

    def reserve(context: ResearchTaskContext) -> bool:
        barrier.wait()
        return budget.reserve_task(context).ok

    with ThreadPoolExecutor(max_workers=4) as executor:
        admitted = list(executor.map(reserve, contexts))

    assert admitted.count(True) == 3
    assert admitted.count(False) == 1
    assert budget.snapshot().active_researchers == 3


def test_duplicate_active_task_is_refused_without_consuming_an_invocation() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=2))
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok

    duplicate = budget.reserve_task(context)
    budget.finish_task(context)

    assert duplicate.ok is False
    assert duplicate.snapshot.remaining_researchers == 9
    assert budget.snapshot().completed_researchers == 1


def test_hard_deadline_refuses_work_but_preserves_snapshot() -> None:
    clock = FakeClock()
    budget = ResearchBudget(
        ResearchBudgetConfig(soft_seconds=600, hard_seconds=1800),
        clock=clock,
    )
    clock.advance(1800)

    decision = budget.reserve_task(ResearchTaskContext("T1", 1, "survey", ("N1",)))

    assert decision.code == "HARD_DEADLINE_REACHED"
    assert decision.snapshot.hard_deadline_reached is True


def test_three_no_yield_calls_block_a_fourth_retrieval() -> None:
    budget, context = active_budget()

    for _ in range(3):
        assert budget.reserve_retrieval(context, "search_climbmix").ok
        budget.record_yield(context, ())

    decision = budget.reserve_retrieval(context, "search_climbmix")

    assert decision.code == "NO_YIELD_STOP"
    assert decision.must_stop is True


def test_tenth_task_is_admitted_and_eleventh_is_refused() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(max_researcher_invocations=10, max_concurrent=10)
    )

    decisions = [
        budget.reserve_task(ResearchTaskContext(f"T{index}", 1, "survey", ("N1",)))
        for index in range(1, 12)
    ]

    assert [decision.ok for decision in decisions] == [True] * 10 + [False]
    assert decisions[-1].code == "TASK_BUDGET_EXHAUSTED"


def test_fourth_round_is_admitted_and_fifth_is_refused() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    fourth = ResearchTaskContext("T4", 4, "deep", ("N1",))

    assert budget.reserve_task(fourth).ok
    budget.finish_task(fourth)
    decision = budget.reserve_task(ResearchTaskContext("T5", 5, "deep", ("N1",)))

    assert decision.code == "ROUND_BUDGET_EXHAUSTED"


def test_hundredth_retrieval_is_admitted_and_next_is_refused() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(
            max_retrieval_calls=100,
            max_tools_per_researcher=101,
            max_searches_per_researcher=101,
        )
    )
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok

    decisions = [budget.reserve_retrieval(context, "search_climbmix") for _ in range(101)]

    assert all(decision.ok for decision in decisions[:100])
    assert decisions[-1].code == "RETRIEVAL_BUDGET_EXHAUSTED"


def test_retrieval_requires_the_exact_active_task_context() -> None:
    budget, context = active_budget()
    different_context = ResearchTaskContext("T1", 2, "focused", ("N2",))

    with pytest.raises(ValueError, match="active task context"):
        budget.reserve_retrieval(different_context, "search_climbmix")
    budget.finish_task(context)
    with pytest.raises(ValueError, match="active task context"):
        budget.reserve_retrieval(context, "search_climbmix")

    assert budget.snapshot().remaining_retrieval_calls == 100


def test_admitted_retrieval_is_charged_before_downstream_validation() -> None:
    budget, context = active_budget()

    decision = budget.reserve_retrieval(context, "search_climbmix")

    assert decision.ok
    assert decision.snapshot.remaining_retrieval_calls == 99
    with pytest.raises(ValueError, match="malformed query"):
        raise ValueError("malformed query")
    assert budget.snapshot().remaining_retrieval_calls == 99


@pytest.mark.parametrize(
    ("tool_name", "config"),
    [
        ("search_climbmix", ResearchBudgetConfig(max_searches_per_researcher=8)),
        (
            "extract_relevant_snippets",
            ResearchBudgetConfig(max_snippets_per_researcher=16),
        ),
        ("other_retrieval_tool", ResearchBudgetConfig(max_tools_per_researcher=20)),
    ],
)
def test_per_task_tool_cap_refuses_first_call_after_limit(
    tool_name: str, config: ResearchBudgetConfig
) -> None:
    budget = ResearchBudget(config)
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok
    limit = {
        "search_climbmix": 8,
        "extract_relevant_snippets": 16,
        "other_retrieval_tool": 20,
    }[tool_name]

    decisions = [budget.reserve_retrieval(context, tool_name) for _ in range(limit + 1)]

    assert all(decision.ok for decision in decisions[:limit])
    assert decisions[-1].code == "TASK_TOOL_BUDGET_EXHAUSTED"


def test_soft_deadline_warns_but_allows_work() -> None:
    clock = FakeClock()
    budget = ResearchBudget(ResearchBudgetConfig(), clock=clock)
    clock.advance(600)

    decision = budget.reserve_task(ResearchTaskContext("T1", 1, "survey", ("N1",)))

    assert decision.ok is True
    assert decision.code == "SOFT_DEADLINE_REACHED"
    assert decision.snapshot.active_researchers == 1


def test_finishing_task_releases_its_concurrency_slot() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1))
    first = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(first).ok

    try:
        pass
    finally:
        budget.finish_task(first)

    assert budget.reserve_task(ResearchTaskContext("T2", 1, "survey", ("N1",))).ok


def test_new_yield_resets_the_no_yield_streak() -> None:
    budget, context = active_budget()
    for _ in range(2):
        assert budget.reserve_retrieval(context, "search_climbmix").ok
        budget.record_yield(context, ())

    assert budget.reserve_retrieval(context, "search_climbmix").ok
    budget.record_yield(context, ("nugget-1",))

    assert budget.reserve_retrieval(context, "search_climbmix").ok


def test_state_hash_change_without_semantic_progress_does_not_reset_stop_streak() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    state = EvidenceCoverageState("What changed?")
    first_report = state.report()
    state.record_search(
        query="What changed?",
        kind="original",
        documents=(DocumentObservation("doc-1", 1),),
    )
    second_report = state.report()

    first = budget.complete_round(1, first_report)
    second = budget.complete_round(2, second_report)
    decision = budget.reserve_task(ResearchTaskContext("T3", 3, "focused", ("N1",)))

    assert first_report.state_hash != second_report.state_hash
    assert first.ok
    assert second.code == "NO_PROGRESS_STOP"
    assert second.must_stop is True
    assert decision.code == "NO_PROGRESS_STOP"


def test_report_excludes_rejected_nuggets_from_semantic_progress() -> None:
    state = EvidenceCoverageState("What changed?")

    result = state.apply_delta(
        {
            "add_nuggets": [
                {
                    "nugget_id": "rejected",
                    "text": "Unsupported claim.",
                    "need_ids": [],
                    "facet_ids": [],
                    "evidence": [],
                    "contradicts": [],
                }
            ]
        }
    )

    assert result.accepted_ids == ()
    assert tuple(nugget.nugget_id for nugget in state.report().nuggets) == ()


def test_duplicate_round_completion_is_idempotent() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    report = EvidenceCoverageState("What changed?").report()

    first = budget.complete_round(1, report)
    duplicate = budget.complete_round(1, report)

    assert duplicate == first
    assert budget.snapshot().completed_rounds == 1


def test_round_completion_requires_the_next_round_index() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    report = EvidenceCoverageState("What changed?").report()

    with pytest.raises(ValueError, match="next incomplete round"):
        budget.complete_round(2, report)
    assert budget.complete_round(1, report).ok
    with pytest.raises(ValueError, match="next incomplete round"):
        budget.complete_round(3, report)


@pytest.mark.parametrize(
    "config",
    [
        {"max_concurrent": 0},
        {"max_retrieval_calls": True},
        {"soft_seconds": -1.0},
        {"soft_seconds": float("nan")},
        {"hard_seconds": float("inf")},
        {"soft_seconds": 10.0, "hard_seconds": 9.0},
    ],
)
def test_configuration_rejects_invalid_budget_values(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ResearchBudgetConfig(**config)  # type: ignore[arg-type]

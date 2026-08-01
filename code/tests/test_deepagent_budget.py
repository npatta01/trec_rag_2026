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
    assert decisions[-1].must_stop is False
    assert budget.snapshot().stop_code is None


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
    assert decision.snapshot.stop_code is None


def test_third_no_yield_stops_only_that_researcher() -> None:
    budget, context = active_budget()

    for _ in range(3):
        assert budget.reserve_retrieval(context, "search_climbmix").ok
        budget.record_yield(context, ())

    budget.finish_task(context)
    other = ResearchTaskContext("T2", 1, "survey", ("N2",))

    assert budget.snapshot().stop_code is None
    assert budget.reserve_task(other).ok
    assert budget.reserve_retrieval(other, "search_climbmix").ok


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
    assert decisions[-1].must_stop is True
    assert decisions[-1].snapshot.stop_code == "TASK_BUDGET_EXHAUSTED"


def test_last_finished_researcher_persists_run_wide_exhaustion() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(max_researcher_invocations=1, max_concurrent=1)
    )
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok

    budget.finish_task(context)

    assert budget.snapshot().stop_code == "TASK_BUDGET_EXHAUSTED"


def test_last_round_is_admitted_and_the_next_is_refused() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=1, max_rounds=4))
    fourth = ResearchTaskContext("T4", 4, "deep", ("N1",))

    assert budget.reserve_task(fourth).ok
    budget.finish_task(fourth)
    decision = budget.reserve_task(ResearchTaskContext("T5", 5, "deep", ("N1",)))

    assert decision.code == "ROUND_BUDGET_EXHAUSTED"
    assert decision.must_stop is True
    assert decision.snapshot.stop_code == "ROUND_BUDGET_EXHAUSTED"


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
    assert decisions[-1].must_stop is True
    assert decisions[-1].snapshot.stop_code == "RETRIEVAL_BUDGET_EXHAUSTED"


def test_last_combined_retrieval_is_admitted_with_terminal_snapshot() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(
            max_retrieval_calls=1,
            max_tools_per_researcher=2,
            max_searches_per_researcher=2,
        )
    )
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok

    decision = budget.reserve_retrieval(context, "search_climbmix")

    assert decision.ok is True
    assert decision.code == "RETRIEVAL_BUDGET_EXHAUSTED"
    assert decision.must_stop is True
    assert decision.snapshot.stop_code == "RETRIEVAL_BUDGET_EXHAUSTED"


def test_last_round_completion_persists_run_wide_exhaustion() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_rounds=1))
    report = EvidenceCoverageState("What changed?").report()

    decision = budget.complete_round(1, report)

    assert decision.code == "ROUND_BUDGET_EXHAUSTED"
    assert decision.must_stop is True
    assert decision.snapshot.stop_code == "ROUND_BUDGET_EXHAUSTED"


def test_empty_round_requires_a_finished_researcher_before_completion() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())

    refused = budget.authorize_round_completion(1)

    assert refused.ok is False
    assert refused.code == "ROUND_RESEARCH_REQUIRED"
    assert refused.must_stop is False
    assert budget.required_research_round() == 1
    assert budget.snapshot().completed_rounds == 0

    context = ResearchTaskContext("R1-N1", 1, "focused", ("N1",))
    assert budget.reserve_task(context).ok
    budget.finish_task(context)

    assert budget.required_research_round() is None
    assert budget.authorize_round_completion(1).ok is True


def test_hard_deadline_and_no_progress_override_global_cap_stop_codes() -> None:
    clock = FakeClock()
    budget = ResearchBudget(
        ResearchBudgetConfig(
            max_retrieval_calls=1,
            max_tools_per_researcher=2,
            max_searches_per_researcher=2,
            no_progress_rounds=1,
            soft_seconds=0,
            hard_seconds=1,
        ),
        clock=clock,
    )
    context = ResearchTaskContext("T1", 1, "focused", ("N1",))
    assert budget.reserve_task(context).ok

    assert (
        budget.reserve_retrieval(context, "search_climbmix").snapshot.stop_code
        == "RETRIEVAL_BUDGET_EXHAUSTED"
    )
    assert (
        budget.complete_round(
            1, EvidenceCoverageState("What changed?").report()
        ).snapshot.stop_code
        == "NO_PROGRESS_STOP"
    )
    clock.advance(1)

    assert budget.snapshot().stop_code == "HARD_DEADLINE_REACHED"


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
    assert decisions[-1].must_stop is False
    assert budget.snapshot().stop_code is None


def test_soft_deadline_refuses_new_survey_but_allows_focused_and_deep_work() -> None:
    clock = FakeClock()
    budget = ResearchBudget(ResearchBudgetConfig(max_concurrent=3), clock=clock)
    clock.advance(600)

    survey = budget.reserve_task(ResearchTaskContext("T1", 1, "survey", ("N1",)))
    focused = budget.reserve_task(ResearchTaskContext("T2", 1, "focused", ("N1",)))
    deep = budget.reserve_task(ResearchTaskContext("T3", 1, "deep", ("N1",)))

    assert survey.ok is False
    assert survey.code == "SOFT_DEADLINE_REACHED"
    assert survey.must_stop is False
    assert focused.ok is True
    assert focused.code == "SOFT_DEADLINE_REACHED"
    assert deep.ok is True
    assert deep.code == "SOFT_DEADLINE_REACHED"
    assert deep.snapshot.active_researchers == 2
    assert deep.snapshot.stop_code is None


def test_survey_already_active_at_soft_deadline_can_finish_and_use_retrieval() -> None:
    clock = FakeClock()
    budget = ResearchBudget(ResearchBudgetConfig(), clock=clock)
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok
    clock.advance(600)

    decision = budget.reserve_retrieval(context, "search_climbmix")
    budget.finish_task(context)

    assert decision.ok is True
    assert decision.code == "SOFT_DEADLINE_REACHED"
    assert budget.snapshot().completed_researchers == 1


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
    budget = ResearchBudget(ResearchBudgetConfig(no_progress_rounds=2))
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


def test_round_completion_refuses_an_out_of_order_round_index() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    report = EvidenceCoverageState("What changed?").report()

    skipped = budget.complete_round(2, report)
    assert not skipped.ok
    assert skipped.code == "ROUND_SEQUENCE_INVALID"
    assert not skipped.must_stop
    assert budget.snapshot().completed_rounds == 0

    assert budget.complete_round(1, report).ok
    assert budget.complete_round(3, report).code == "ROUND_SEQUENCE_INVALID"
    assert budget.snapshot().completed_rounds == 1


def test_traceable_budget_codes_match_the_budget_code_literal() -> None:
    """deepagent_tracing duplicates BudgetCode because it may not import budget."""
    from typing import get_args

    from trec_rag.deepagent_budget import BudgetCode
    from trec_rag.deepagent_tracing import _BUDGET_CODES

    assert set(get_args(BudgetCode)) == set(_BUDGET_CODES)


def test_main_model_exhaustion_is_recorded_as_a_run_stop() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())

    assert budget.snapshot().stop_code is None

    budget.note_main_model_exhausted()

    assert budget.snapshot().stop_code == "MAIN_MODEL_BUDGET_EXHAUSTED"


def test_main_model_exhaustion_never_outranks_a_harder_stop() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(hard_seconds=0.0, soft_seconds=0.0))

    assert budget.snapshot().stop_code == "HARD_DEADLINE_REACHED"

    budget.note_main_model_exhausted()

    assert budget.snapshot().stop_code == "HARD_DEADLINE_REACHED"


def test_pending_round_closure_tracks_the_unclosed_researched_round() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    report = EvidenceCoverageState("What changed?").report()
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))

    assert budget.pending_round_closure() is None

    assert budget.reserve_task(context).ok
    assert budget.pending_round_closure() is None, "a running batch is not closable"

    budget.finish_task(context)
    assert budget.pending_round_closure() == 1

    assert budget.complete_round(1, report).ok
    assert budget.pending_round_closure() is None, "a closed round stays closed"


def test_pending_round_closure_ignores_rounds_without_finished_research() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())
    report = EvidenceCoverageState("What changed?").report()
    first = ResearchTaskContext("T1", 1, "survey", ("N1",))

    assert budget.reserve_task(first).ok
    budget.finish_task(first)
    assert budget.complete_round(1, report).ok

    # Round 2 has been opened by nobody, so nothing is owed a close.
    assert budget.pending_round_closure() is None

    second = ResearchTaskContext("T2", 2, "focused", ("N2",))
    assert budget.reserve_task(second).ok
    budget.finish_task(second)
    assert budget.pending_round_closure() == 2


def test_pending_round_closure_yields_once_the_run_must_stop() -> None:
    budget = ResearchBudget(ResearchBudgetConfig(max_researcher_invocations=1))
    context = ResearchTaskContext("T1", 1, "survey", ("N1",))

    assert budget.reserve_task(context).ok
    budget.finish_task(context)

    assert budget.snapshot().stop_code == "TASK_BUDGET_EXHAUSTED"
    assert budget.pending_round_closure() is None


def test_round_authorization_refuses_an_out_of_order_round_index() -> None:
    budget = ResearchBudget(ResearchBudgetConfig())

    decision = budget.authorize_round_completion(2)

    assert not decision.ok
    assert decision.code == "ROUND_SEQUENCE_INVALID"
    assert budget.required_research_round() is None


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

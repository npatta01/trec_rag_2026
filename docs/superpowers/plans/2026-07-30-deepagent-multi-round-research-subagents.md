# Deep Agent Multi-Round Research Subagents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the main agent's brittle direct retrieval loop with bounded
synchronous researcher subagents that refine queries, return grounded evidence
bundles, and cannot exceed invocation-local tool, round, task, or time budgets.

**Architecture:** Keep the existing need/facet/nugget state and cache-first
retrieval implementations. Add one concurrency-safe budget module and one
researcher-contract module, then make the main Deep Agent coordinate a single
restricted custom researcher type through `task`. Fixed retrieval tools record
their actual arguments atomically; the main agent alone applies semantic state
deltas and closes rounds.

**Tech Stack:** Python 3.12, `deepagents==0.7.0`, `langchain==1.3.14`,
LangGraph, `langchain-openrouter==0.2.7`, Pydantic, pytest, Phoenix/OpenTelemetry.

## Global Constraints

- Python SDK only; do not add an interactive CLI.
- Use the same configurable `openrouter:<model-id>` model for main and researcher.
- Keep `ChatOpenRouter(max_retries=0)` and set a 120-second request timeout.
- Keep retrieval and reranker caches tool-owned and cache-first.
- The original untouched narrative search remains deterministic and is not an
  agent-budgeted follow-up call.
- Default safety limits: 10 researcher invocations, 4 rounds, 3 concurrent
  researchers, 100 combined researcher search/snippet attempts, 20 total tool
  calls per researcher, 8 searches per researcher, 16 snippet calls per
  researcher, 30 model calls per researcher, 25 main model calls, 10-minute
  soft threshold, and 30-minute hard admission deadline.
- After 3 consecutive no-yield retrieval calls, a researcher may only return
  its bundle. After 2 consecutive no-progress rounds, the main agent must stop.
- Disable the default general-purpose subagent; researchers cannot recursively
  call `task`.
- Do not install `TodoListMiddleware` or expose `write_todos`.
- Expose only `read_file` for automatic oversized-result spill; do not expose
  write/edit/delete/glob/grep/execute.
- Use `.venv/bin/python-rocm` for live snippet/reranker execution on this host.
- Keep test work proportional to a POC: targeted deterministic contracts plus
  one topic-224 smoke run.

---

## File Structure

- Create `code/trec_rag/deepagent_budget.py`: budget configuration, atomic
  reservations, snapshots, no-yield counters, and round-progress stopping.
- Create `code/trec_rag/deepagent_research.py`: task envelope, evidence-bundle
  schemas, task-context propagation, role-specific tool filtering, and agent
  construction.
- Modify `code/trec_rag/deepagent_evidence.py`: validate motivating need/facet
  IDs and record actual retrieval actions without pending string authorization.
- Modify `code/trec_rag/deepagent_retrieval.py`: bind budgeted fixed tools,
  construct the coordinator/researcher harness, and expose budget results.
- Modify `code/trec_rag/deepagent_tracing.py`: record researcher and budget
  outcome fields on Phoenix spans.
- Modify `code/trec_rag/README.md`: document the SDK behavior and budget knobs.
- Create `code/tests/test_deepagent_budget.py` and
  `code/tests/test_deepagent_research.py`.
- Modify `code/tests/test_deepagent_evidence.py`,
  `code/tests/test_deepagent_retrieval.py`, and
  `code/tests/test_deepagent_tracing.py`.

### Task 1: Invocation-Local Research Budget

**Files:**
- Create: `code/trec_rag/deepagent_budget.py`
- Create: `code/tests/test_deepagent_budget.py`

**Interfaces:**
- Produces:
  `ResearchBudgetConfig`, `ResearchTaskContext`, `BudgetSnapshot`,
  `BudgetDecision`, and `ResearchBudget`.
- `ResearchBudget.reserve_task(context)` atomically enforces task, round,
  concurrency, and elapsed-time admission.
- `ResearchBudget.reserve_retrieval(context, tool_name)` atomically enforces
  shared retrieval attempts, per-task search/snippet counts, no-yield stops,
  and elapsed time.
- `ResearchBudget.finish_task(context)` releases concurrency in `finally`.
- `ResearchBudget.record_yield(context, identifiers)` maintains the mechanical
  three-call no-yield streak.
- `ResearchBudget.complete_round(round_index, report)` maintains the
  two-round semantic no-progress streak.

- [ ] **Step 1: Write budget contract tests**

```python
def test_parallel_task_reservations_never_exceed_concurrency() -> None:
    budget = ResearchBudget(
        ResearchBudgetConfig(max_researcher_invocations=10, max_concurrent=3)
    )
    contexts = [ResearchTaskContext(f"T{i}", 1, "survey", ("N1",)) for i in range(4)]
    decisions = [budget.reserve_task(context) for context in contexts]
    assert [decision.ok for decision in decisions] == [True, True, True, False]
    assert decisions[-1].code == "CONCURRENCY_BUDGET_EXHAUSTED"


def test_hard_deadline_refuses_work_but_preserves_snapshot() -> None:
    clock = FakeClock()
    budget = ResearchBudget(
        ResearchBudgetConfig(soft_seconds=600, hard_seconds=1800),
        clock=clock,
    )
    clock.advance(1800)
    decision = budget.reserve_task(
        ResearchTaskContext("T1", 1, "survey", ("N1",))
    )
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
```

- [ ] **Step 2: Run the new tests and confirm the module is absent**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_deepagent_budget.py -q
```

Expected: collection fails because `trec_rag.deepagent_budget` does not exist.

- [ ] **Step 3: Implement immutable configuration and atomic state**

```python
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
```

All validation occurs in `ResearchBudgetConfig.__post_init__`. Protect counters,
active task IDs, per-task counters, seen yield IDs, and round snapshots with one
`threading.Lock`. Count attempted calls before argument validation. Use an
injected monotonic clock for deterministic threshold tests.

The test module defines `FakeClock` with `__call__()` and `advance(seconds)`,
plus `active_budget()` that reserves one `ResearchTaskContext` before returning
the `(budget, context)` pair used above.

- [ ] **Step 4: Add tests for all approved hard caps and adaptive stopping**

Cover exact boundary behavior for 10 tasks, round 4 vs. 5, 100 retrieval calls,
8 searches, 16 snippets, 20 total task tools, soft warning, task release in
`finally`, a yield reset, and two unchanged `EvidenceCoverageReport` snapshots.
Round progress is the tuple of accepted nugget IDs, needs whose status is no
longer `unaddressed`, and facets whose status is no longer `open`. A round
counts as progress only when one of those sets grows; a new state hash alone
does not reset the no-progress streak.

- [ ] **Step 5: Run and lint Task 1**

```bash
.venv/bin/python -m pytest code/tests/test_deepagent_budget.py -q
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_budget.py \
  code/tests/test_deepagent_budget.py
```

Expected: all Task 1 tests pass and Ruff is clean.

- [ ] **Step 6: Commit Task 1**

```bash
git add code/trec_rag/deepagent_budget.py code/tests/test_deepagent_budget.py
git commit -m "Add bounded Deep Agent research budget"
```

### Task 2: Evidence Bundles and Researcher Middleware

**Files:**
- Create: `code/trec_rag/deepagent_research.py`
- Create: `code/tests/test_deepagent_research.py`

**Interfaces:**
- Consumes: all Task 1 types.
- Produces:
  `ResearchTaskEnvelope`, `BundleEvidence`, `CandidateNugget`,
  `EvidenceBundle`, `bind_research_task`, `current_research_task`,
  `ResearchTaskBudgetMiddleware`, `MainToolFilterMiddleware`,
  `ResearcherToolFilterMiddleware`, and `build_research_subagent`.
- Task descriptions use compact JSON matching `ResearchTaskEnvelope`; queries
  remain free for the researcher to generate and refine.
- A context variable binds the validated envelope around each synchronous or
  asynchronous `task` handler, so fixed tools obtain task/round/depth metadata
  without asking the model to repeat it on every call.

- [ ] **Step 1: Write schema and middleware tests**

```python
def test_research_bundle_requires_exact_evidence_coordinates() -> None:
    bundle = EvidenceBundle.model_validate(
        {
            "research_task_id": "R1-N1",
            "round_index": 1,
            "depth": "survey",
            "motivating_need_ids": ["N1"],
            "candidate_nuggets": [{
                "claim": "Claim.",
                "need_ids": ["N1"],
                "facet_ids": [],
                "evidence": [{
                    "document_id": "D1",
                    "snippet_id": "S1",
                    "page_index": 0,
                    "quote": "Exact quote.",
                }],
                "contradicts_claims": [],
            }],
            "conflicts": [],
            "unresolved_gaps": [],
            "suggested_followups": [],
            "stopping_reason": "goal_satisfied",
            "budget_snapshot": budget_payload(),
        }
    )
    assert bundle.candidate_nuggets[0].evidence[0].snippet_id == "S1"


def test_role_filters_expose_only_approved_tools() -> None:
    assert visible_tools(MainToolFilterMiddleware(), ALL_TOOLS) == {
        "task", "view_retrieval_state", "update_retrieval_state",
        "complete_research_round", "read_file",
    }
    assert visible_tools(ResearcherToolFilterMiddleware(), ALL_TOOLS) == {
        "search_climbmix", "extract_relevant_snippets",
        "view_retrieval_state", "read_file",
    }
```

- [ ] **Step 2: Run the new tests and confirm failure**

```bash
.venv/bin/python -m pytest code/tests/test_deepagent_research.py -q
```

Expected: collection fails because `trec_rag.deepagent_research` is absent.

- [ ] **Step 3: Implement strict Pydantic bundle and task schemas**

```python
class ResearchTaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research_task_id: str = Field(min_length=1)
    round_index: int = Field(ge=1)
    depth: Literal["survey", "focused", "deep"]
    motivating_ids: list[str] = Field(min_length=1)
    goal: str = Field(min_length=1)
    known_evidence: str = ""
    remaining_gap: str = ""


class BudgetSnapshotModel(BaseModel):
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
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1)
    snippet_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    quote: str = Field(min_length=1)


class CandidateNugget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    need_ids: list[str] = Field(min_length=1)
    facet_ids: list[str]
    evidence: list[BundleEvidence] = Field(min_length=1)
    contradicts_claims: list[str]


class EvidenceBundle(BaseModel):
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
```

The test module defines `budget_payload()` from a default `BudgetSnapshot`, an
`ALL_TOOLS` tuple containing every main/researcher/Deep-Agents built-in name,
and `visible_tools(middleware, names)` by constructing a `ModelRequest` and
capturing the filtered tool names.

- [ ] **Step 4: Implement task binding and role filters**

`ResearchTaskBudgetMiddleware.wrap_tool_call` must:

1. ignore non-`task` calls;
2. parse `request.tool_call["args"]["description"]` as
   `ResearchTaskEnvelope`;
3. require `subagent_type == "researcher"`;
4. reserve the task;
5. bind the context with `ContextVar`;
6. call the handler;
7. append the final compact budget snapshot to the returned `ToolMessage`; and
8. release the task in `finally`.

The async wrapper mirrors the same sequence. Main filtering allows parallel
tool calls so several `task` calls can run together. Researcher filtering sets
`parallel_tool_calls=False`.

- [ ] **Step 5: Build one restricted researcher spec**

```python
def build_research_subagent(
    *,
    model: BaseChatModel,
    tools: Sequence[Callable[..., object]],
    budget_config: ResearchBudgetConfig,
) -> SubAgent:
    return {
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
    }
```

Test that no `TodoListMiddleware` is present, `task` is invisible to the
researcher, and the custom researcher uses the same model object as the main
agent.

- [ ] **Step 6: Run and lint Task 2**

```bash
.venv/bin/python -m pytest \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py -q
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_budget.py \
  code/trec_rag/deepagent_research.py \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py
```

- [ ] **Step 7: Commit Task 2**

```bash
git add code/trec_rag/deepagent_research.py code/tests/test_deepagent_research.py
git commit -m "Define bounded retrieval researcher"
```

### Task 3: Atomic Retrieval Tools and Coordinator Integration

**Files:**
- Modify: `code/trec_rag/deepagent_evidence.py`
- Modify: `code/trec_rag/deepagent_retrieval.py`
- Modify: `code/tests/test_deepagent_evidence.py`
- Modify: `code/tests/test_deepagent_retrieval.py`

**Interfaces:**
- Consumes: Task 1 budget and Task 2 research contracts.
- Produces:
  `EvidenceCoverageState.record_retrieval_action`,
  budgeted `search_climbmix` and `extract_relevant_snippets`,
  `complete_research_round`, a main coordinator agent, and
  `AgentRetrievalResult.budget_snapshot`.
- Ordinary retrieval calls no longer consume a pending
  `choose_next_action` authorization.

- [ ] **Step 1: Replace pending-action tests with atomic-action tests**

```python
def test_search_records_the_actual_query_and_motivation_in_one_call() -> None:
    toolset, state, context = seeded_toolset()
    with bind_research_task(context):
        payload = json.loads(
            toolset.search_climbmix(
                "actual refined query",
                ["N1"],
                "N1 has no grounded driver evidence",
            )
        )
    assert "error" not in payload
    action = state.report().actions[-1]
    assert action.target == "actual refined query"
    assert action.motivating_ids == ("N1",)
    assert action.research_task_id == context.research_task_id


def test_invalid_motivation_is_rejected_before_retrieval() -> None:
    with bind_research_task(context()):
        payload = json.loads(
            toolset.search_climbmix("query", ["UNKNOWN"], "reason")
        )
    assert payload["code"] == "UNKNOWN_MOTIVATION"
    assert fake_retriever.queries == ["untouched narrative"]
```

The retrieval test module extends its existing fake-retriever and fake-agent
fixtures with `seeded_toolset()`, which seeds need `N1`, creates an active
`ResearchTaskContext`, and returns the toolset plus its state. `context()`
returns that same valid context shape; no production-only helper is referenced
by the tests.

Delete tests whose only requirement is exact string equality between
`choose_next_action` and the following tool call. Retain pagination cursor,
quote grounding, cache, RRF, trace isolation, and narrative-first tests.

- [ ] **Step 2: Add atomic action recording to evidence state**

Extend `ActionReport` and `_Action` with:

```python
research_task_id: str | None
round_index: int | None
depth: Literal["survey", "focused", "deep"] | None
```

Implement:

```python
def record_retrieval_action(
    self,
    *,
    action: Literal["search", "extract", "paginate", "refocus"],
    target: str,
    focus_query: str | None,
    motivating_ids: Sequence[str],
    rationale: str,
    context: ResearchTaskContext,
) -> str | None:
    """Return a narrow error code, or append one consumed mechanical action."""
```

Validate nonblank actual arguments, existing open needs/facets, and exact
context values. Append the action directly as `consumed`; never create a
pending action. Keep `choose_action` only for terminal completion/saturation
until all old report compatibility tests are updated.

- [ ] **Step 3: Make fixed retrieval tools cache-first and budget-first**

The researcher-facing signatures become:

```python
def search_climbmix(
    query: str,
    motivating_ids: list[str],
    rationale: str,
) -> str: ...


def extract_relevant_snippets(
    document_id: str,
    focus_query: str,
    motivating_ids: list[str],
    rationale: str,
    cursor: str | None = None,
) -> str: ...
```

Each tool:

1. gets `current_research_task()` from Task 2;
2. reserves its attempted call before argument validation;
3. validates motivations and records the actual action;
4. runs the existing cache-first implementation;
5. records returned mechanical search/page state;
6. records unique document/snippet identifiers as yield;
7. returns the existing payload plus `budget_snapshot` and `must_stop`.

For pagination, retain exact cursor validation in the snippet extractor but
remove exact duplicated focus/target authorization. Do not expose cache
arguments to the model.

- [ ] **Step 4: Add main-only round completion**

```python
def complete_research_round(round_index: int) -> str:
    decision = budget.complete_round(round_index, coverage_state.report())
    return json.dumps(
        {
            "ok": decision.ok,
            "code": decision.code,
            "must_stop": decision.must_stop,
            "budget_snapshot": decision.snapshot.as_dict(),
        },
        sort_keys=True,
    )
```

The main prompt requires exactly one batched semantic delta after a parallel
task batch, followed by one `complete_research_round` call.

- [ ] **Step 5: Construct the coordinator and restricted researcher**

In `_create_agent`:

- initialize `ChatOpenRouter(model=model_id, max_retries=0, timeout=120)`;
- register an OpenRouter harness profile that disables the auto-added
  general-purpose subagent;
- pass one `build_research_subagent(...)`;
- add main `ModelCallLimitMiddleware(run_limit=25, exit_behavior="end")`;
- add main `ToolCallLimitMiddleware(tool_name="task", run_limit=10,
  exit_behavior="continue")`;
- add `ResearchTaskBudgetMiddleware`, then `MainToolFilterMiddleware`;
- retain `StateBackend()` for automatic ephemeral spill; and
- expose only main state/round tools directly.

The main system prompt seeds needs before delegation, emits JSON task envelopes,
batches independent `task` calls, merges returned bundles via one
`update_retrieval_state` delta, completes the round, and stops whenever the
budget response says `must_stop`.

`MainToolFilterMiddleware` also provides the reserved finalization turn. When
the shared budget has a stop code, or `run_model_call_count` reaches
`max_main_models - 1`, its next model request contains no tools and appends a
short instruction to return the grounded partial result immediately. The
framework's final model-call limit remains the absolute backstop.

- [ ] **Step 6: Expose deterministic budget outcome**

Add `budget_snapshot: BudgetSnapshot` to `AgentRetrievalResult`. Prefer terminal
coverage reasons, then budget stop reasons, then `agent_completed`. Budget
exhaustion must preserve completed searches, grounded nuggets, and open gaps.

- [ ] **Step 7: Run the full targeted retrieval suite**

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_budget.py \
  code/trec_rag/deepagent_research.py \
  code/trec_rag/deepagent_evidence.py \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py
```

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  code/trec_rag/deepagent_evidence.py \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py
git commit -m "Delegate bounded retrieval to researchers"
```

### Task 4: Phoenix Visibility, SDK Documentation, and Topic-224 Pass

**Files:**
- Modify: `code/trec_rag/deepagent_tracing.py`
- Modify: `code/tests/test_deepagent_tracing.py`
- Modify: `code/trec_rag/README.md`
- Verify only: topic-224 cache, OpenRouter, ClimbMix, and Phoenix integration.

**Interfaces:**
- Consumes: completed Tasks 1–3.
- Produces: trace-visible task/round/depth/budget outcome, updated SDK docs, one
  normal topic-224 result, and one deliberately tiny-budget stop result.

- [ ] **Step 1: Add trace assertions before trace implementation**

```python
def test_agent_result_trace_records_research_budget() -> None:
    span.record_result(
        fused_document_ids=("D1",),
        stopping_reason="budget_exhausted",
        coverage_state_hash="abc",
        need_count=5,
        answerable_need_count=1,
        conflicted_need_count=0,
        unresolved_need_count=4,
        nugget_count=2,
        action_count=3,
        researcher_invocation_count=4,
        research_round_count=2,
        retrieval_call_count=11,
        budget_stop_code="NO_PROGRESS_STOP",
    )
    assert attributes["deepagent.researcher_invocation_count"] == 4
    assert attributes["deepagent.budget_stop_code"] == "NO_PROGRESS_STOP"
```

- [ ] **Step 2: Record bounded researcher/budget attributes**

Add only compact identifiers/counts to Phoenix attributes. Keep actual
queries, focus queries, snippets, and existing configured trace content on
their current spans. Never add cache paths, secrets, full documents, or direct
personal data.

- [ ] **Step 3: Update the README**

Replace the three-follow-up-search description with:

- main/researcher capability matrix;
- fixed default budget table;
- query refinement and multi-round semantics;
- explicit no-`write_todos` decision;
- `budget_snapshot` result field;
- `.venv/bin/python-rocm` live-run requirement; and
- the fact that budget exhaustion is a grounded partial result, not coverage.

- [ ] **Step 4: Run all targeted tests and Ruff**

```bash
.venv/bin/python-rocm -m pytest -q \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py
.venv/bin/python-rocm -m ruff check \
  code/trec_rag/deepagent_snippets.py \
  code/trec_rag/deepagent_budget.py \
  code/trec_rag/deepagent_research.py \
  code/trec_rag/deepagent_evidence.py \
  code/trec_rag/deepagent_retrieval.py \
  code/trec_rag/deepagent_tracing.py \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_budget.py \
  code/tests/test_deepagent_research.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py
```

- [ ] **Step 5: Run topic 224 once through the normal Python SDK**

```bash
PYTHONPATH=code .venv/bin/python-rocm - <<'PY'
from pathlib import Path
import json

from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.topics import load_topic_narrative

topic_file = Path(
    "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
)
narrative = load_topic_narrative("224", topic_file)
result = DeepAgentRetriever.from_env(root=Path.cwd()).retrieve(narrative)
print(json.dumps({
    "stopping_reason": result.stopping_reason,
    "need_count": len(result.coverage_report.needs),
    "nugget_count": len(result.coverage_report.nuggets),
    "unresolved_need_ids": result.coverage_report.unresolved_need_ids,
    "search_queries": [search.query for search in result.searches],
    "budget": result.budget_snapshot.as_dict(),
    "trace_flush_succeeded": result.trace_flush_succeeded,
}, indent=2, sort_keys=True))
PY
```

Expected: five narrative-anchored needs; at least one completed researcher
bundle; no `matching_action` error; actual refined queries in the ledger;
grounded nugget evidence or explicit gaps; a completion, saturation, or honest
budget stopping reason; and successful trace flush when Phoenix is configured.

- [ ] **Step 6: Run one tiny-budget termination diagnostic**

Invoke the same narrative with:

```python
ResearchBudgetConfig(
    max_researcher_invocations=1,
    max_rounds=1,
    max_concurrent=1,
    max_retrieval_calls=1,
    max_tools_per_researcher=2,
    max_searches_per_researcher=1,
    max_snippets_per_researcher=1,
    max_models_per_researcher=5,
    max_main_models=8,
    soft_seconds=30,
    hard_seconds=120,
)
```

Expected: termination with `budget_exhausted`, no extra research call after the
limit, and a valid partial coverage report.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  code/trec_rag/deepagent_tracing.py \
  code/tests/test_deepagent_tracing.py \
  code/trec_rag/README.md
git commit -m "Trace and document bounded researcher retrieval"
```

- [ ] **Step 8: Record live verification evidence**

Append the sanitized result summary, Phoenix trace identifier when available,
elapsed time, model, package versions, exact test commands, and any remaining
unknowns under `## Verification Evidence` in this plan. Do not record API keys,
raw full documents, or cache contents.

## Verification Evidence

### Task 4 — 2026-07-30

- Added bounded Phoenix root-span attributes only: researcher invocation count,
  completed research-round count, combined retrieval-call count, and a validated
  terminal budget stop code. The attributes contain no task payloads, queries,
  snippets, documents, cache paths, or credentials.
- Updated the SDK documentation with the coordinator/researcher capability
  matrix, fixed budget defaults, query-refinement and round semantics,
  intentional no-`write_todos` boundary, `budget_snapshot`, ROCm interpreter
  requirement, and the grounded-partial exhaustion rule.
- Fresh targeted verification completed with the ROCm helper:

  ```console
  $ .venv/bin/python-rocm -m pytest -q \
      code/tests/test_deepagent_snippets.py \
      code/tests/test_deepagent_budget.py \
      code/tests/test_deepagent_research.py \
      code/tests/test_deepagent_evidence.py \
      code/tests/test_deepagent_retrieval.py \
      code/tests/test_deepagent_tracing.py
  272 passed in 9.78s

  $ /home/npatta01/anaconda3/bin/ruff check \
      code/trec_rag/deepagent_snippets.py \
      code/trec_rag/deepagent_budget.py \
      code/trec_rag/deepagent_research.py \
      code/trec_rag/deepagent_evidence.py \
      code/trec_rag/deepagent_retrieval.py \
      code/trec_rag/deepagent_tracing.py \
      code/tests/test_deepagent_snippets.py \
      code/tests/test_deepagent_budget.py \
      code/tests/test_deepagent_research.py \
      code/tests/test_deepagent_evidence.py \
      code/tests/test_deepagent_retrieval.py \
      code/tests/test_deepagent_tracing.py
  All checks passed!
  ```

- `git diff --check` completed with no output.
- The normal topic-224 SDK run was started once through `from_env` with the
  configured OpenRouter, ClimbMix, and Phoenix integration. Its practical outer
  cap interrupted it safely at `retrieve` after 718.296 seconds. It produced no
  result, so no topic outcome, trace flush status, or Phoenix trace identifier
  is available. It was not retried.
- The specified tiny-budget topic-224 diagnostic was started once with the
  requested `ResearchBudgetConfig` and a separate 180-second outer cap. It was
  likewise interrupted safely at `retrieve` after 178.388 seconds. Therefore
  live graceful-budget termination remains unverified; the deterministic budget
  and retrieval regression contracts above remain the available evidence.
- Non-secret local configuration evidence: model
  `openrouter:deepseek/deepseek-v4-flash`; `deepagents` 0.7.0; `langchain`
  1.3.14; `langchain-openrouter` 0.2.7; `arize-phoenix-client` 2.13.0;
  `arize-phoenix-otel` 0.16.1; `openinference-instrumentation-langchain`
  0.1.67.

### Task 4 fix round 1 — Phoenix precedence and compact task context

- The Phoenix root span now preserves the exact returned `stopping_reason`.
  `budget_stop_code` is a separate compact attribute and does not override a
  coverage-terminal reason.
- Existing manual ClimbMix, snippet, and researcher-task spans now receive
  only sanitized task ID, round, depth, budget decision/must-stop state,
  remaining researcher/round/retrieval counts, and an optional terminal budget
  code. No task description, query, snippet, document, cache path, or secret is
  added.
- No live provider run was performed for this fix round.

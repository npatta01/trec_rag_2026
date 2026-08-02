# DeepAgent Explicit Retrieval Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make explicit, validated `complete_retrieval` the only successful coordinator completion path while retaining a safe silent-exit guard.

**Architecture:** Put the live-evidence/draft-selection invariant in `EvidenceCoverageState`. Expose a no-argument `complete_retrieval` tool through `AgentToolset`, reuse the existing `stop`/`completion` transition after validation, and reject the legacy direct completion route. Wire the existing middleware guard to the same invariant and classify any final incomplete closeout as `closeout_refused`.

**Tech Stack:** Python 3.12 project environment, Pydantic state models, `deepagents==0.7.0`, `langchain==1.3.14`, `langchain-core==1.5.2`, pytest, and `FakeMessagesListChatModel`.

## Global Constraints

- The invariant must require selected `draft_nugget_ids` only for needs with live, non-superseded nuggets.
- Completion rejection must return stable bounded JSON and must not raise or mutate terminal state.
- Successful completion must reuse the existing `EvidenceCoverageState.choose_action(..., action="stop", target="completion", ...)` transition.
- `complete_retrieval` must be the only model-facing successful completion route; direct `choose_next_action` completion must be rejected.
- Preserve the one-shot silent-exit guard and make it use the shared invariant.
- `budget_exhausted` outranks `closeout_refused`; explicit coverage terminal reasons outrank both.
- Register every new stopping reason in `deepagent_tracing.py` because trace recording swallows exceptions.
- Do not wire this experimental SDK into competition runners or submission artifacts.

---

### Task 1: Add the shared completion invariant and explicit tool contract

**Files:**
- Modify: `code/trec_rag/deepagent_evidence.py` near `EvidenceCoverageState.report()` and terminal validation.
- Modify: `code/trec_rag/deepagent_retrieval.py` near `AgentToolset`, `DeepAgentRetriever.retrieve`, and `choose_next_action`.
- Test: `code/tests/test_deepagent_evidence.py` and `code/tests/test_deepagent_retrieval.py`.

**Interfaces:**
- Produce `EvidenceCoverageState.pending_closeout_need_ids() -> tuple[str, ...]` (or an equivalently named public helper) returning deterministic need IDs with live nuggets and no drafts.
- Produce `AgentToolset.complete_retrieval: Callable[[], str]`.
- The retrieval-local `complete_retrieval()` returns JSON with `ok: false`, stable code `INCOMPLETE_CLOSEOUT`, and `need_ids` on invariant failure; on valid input it returns the existing completion transition result.

- [ ] **Step 1: Write the failing invariant tests.** Add cases proving a live nugget with no draft selection is pending, a selected live nugget is not pending, and a superseded-only nugget is not pending.
- [ ] **Step 2: Run the focused invariant tests.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_evidence.py -k 'closeout or draft' -q`

Expected: FAIL because the shared helper does not yet exist.

- [ ] **Step 3: Write the failing completion-tool tests.** Build the existing retrieval test fixture around `AgentToolset`, assert incomplete completion returns structured `INCOMPLETE_CLOSEOUT` with need IDs and does not set `coverage_report.terminal_reason`, and assert a valid state reaches `completion`.
- [ ] **Step 4: Run the focused retrieval tests.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_retrieval.py -k 'complete_retrieval or choose_next_action' -q`

Expected: FAIL because `AgentToolset.complete_retrieval` and the retrieval-local tool do not yet exist.

- [ ] **Step 5: Implement the minimal shared helper and tool.** Compute live nugget IDs from the immutable report, return sorted offending need IDs, add the optional dataclass field, implement structured rejection, and reuse `choose_action` after validation. Reject only direct `target="completion"` calls in `choose_next_action` with `COMPLETE_RETRIEVAL_REQUIRED`.
- [ ] **Step 6: Run the focused tests to verify green.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_evidence.py -k 'closeout or draft' code/tests/test_deepagent_retrieval.py -k 'complete_retrieval or choose_next_action' -q`

Expected: PASS with no failures.

- [ ] **Step 7: Commit the task.**

```bash
git add code/trec_rag/deepagent_evidence.py code/trec_rag/deepagent_retrieval.py code/tests/test_deepagent_evidence.py code/tests/test_deepagent_retrieval.py
git commit -m "feat: add validated retrieval completion tool"
```

### Task 2: Wire the guard, terminal classification, and trace whitelist

**Files:**
- Modify: `code/trec_rag/deepagent_retrieval.py` near `_create_agent`, `AgentToolset` construction, and final stopping-reason calculation.
- Modify: `code/trec_rag/deepagent_research.py` near `MainToolFilterMiddleware` and `_ALLOWED_TOOLS`/closeout directive.
- Modify: `code/trec_rag/deepagent_tracing.py` in `_AgentSpan._STOPPING_REASONS`.
- Test: `code/tests/test_deepagent_research.py`, `code/tests/test_deepagent_retrieval.py`, and `code/tests/test_deepagent_tracing.py`.

**Interfaces:**
- `MainToolFilterMiddleware(..., closeout_pending=...)` receives the retrieval-local predicate through `AgentToolset`.
- The coordinator tool list includes `complete_retrieval`.
- Final stopping reason uses `closeout_refused` when the final report still has pending closeout needs and no higher-precedence reason exists.

- [ ] **Step 1: Write the failing middleware wiring and stopping-reason tests.** Assert `_create_agent` exposes `complete_retrieval`, a pending predicate activates the existing one-shot redirect, a final incomplete result is `closeout_refused`, and the trace span accepts that reason.
- [ ] **Step 2: Run the focused tests to verify red.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_research.py -k 'closeout' code/tests/test_deepagent_retrieval.py -k 'closeout or stopping_reason' code/tests/test_deepagent_tracing.py -k 'stopping_reason' -q`

Expected: FAIL because production construction does not pass the predicate, the tool is absent, and tracing does not whitelist the reason.

- [ ] **Step 3: Implement the minimal wiring.** Add `complete_retrieval` to coordinator tools and the middleware allowed set, pass the shared predicate from `retrieve`, keep the existing write-up bounce, classify pending final closeout as `closeout_refused` beneath budget exhaustion, and register the whitelist value.
- [ ] **Step 4: Run focused middleware, retrieval, and tracing tests.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_research.py -k 'closeout' code/tests/test_deepagent_retrieval.py -k 'closeout or stopping_reason' code/tests/test_deepagent_tracing.py -k 'stopping_reason' -q`

Expected: PASS with no failures.

- [ ] **Step 5: Commit the task.**

```bash
git add code/trec_rag/deepagent_research.py code/trec_rag/deepagent_retrieval.py code/trec_rag/deepagent_tracing.py code/tests/test_deepagent_research.py code/tests/test_deepagent_retrieval.py code/tests/test_deepagent_tracing.py
git commit -m "fix: guard incomplete retrieval closeout"
```

### Task 3: Prove the real LangGraph loop and regression surface

**Files:**
- Modify: `code/tests/test_deepagent_research.py` or `code/tests/test_deepagent_retrieval.py`, using the existing pinned LangGraph imports and fake model utilities.
- Modify: any source file only if the failing real-loop characterization identifies a production control-flow defect; keep the change limited to the explicit completion path.

**Interfaces:**
- The scripted fake model must exercise the actual `langchain.agents.create_agent` loop, not only a direct middleware handler stub.
- Scenarios: valid `complete_retrieval` terminates; incomplete completion returns a tool rejection and the loop continues; silent exit is redirected; a refused closeout is not labeled `agent_completed`.

- [ ] **Step 1: Write the real-loop tests first.** Use a tool that records invocation, a scripted `FakeMessagesListChatModel`, and the production middleware contract. Assert tool invocation count and final graph state rather than only handler calls.
- [ ] **Step 2: Run the real-loop tests to verify red.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_research.py -k 'real_loop or explicit_completion' -q`

Expected: FAIL until the coordinator tool and guard are correctly exposed to the real graph.

- [ ] **Step 3: Implement only the missing loop wiring.** Do not replace validation with prompt text or automatic draft selection. Preserve budget accounting and one-shot behavior.
- [ ] **Step 4: Run the real-loop tests to verify green.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_deepagent_research.py -k 'real_loop or explicit_completion' -q`

Expected: PASS with no failures.

- [ ] **Step 5: Run the complete test suite.**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/ -q`

Expected: all existing tests pass and only the repository's known skips remain; no new failures.

- [ ] **Step 6: Review the final diff and working tree.**

Run: `git diff --check && git status --short --branch && git diff --stat HEAD~2..HEAD`

Expected: no whitespace errors, only scoped source/tests/docs commits, and no secrets or generated artifacts.

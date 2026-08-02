# DeepAgent Retrieval — Handover to Codex

**Date:** 2026-08-02

**PR:** <https://github.com/npatta01/trec_rag_2026/pull/35>

**Branch:** `claude/deepagent-retrieval-handover-df766f` (pushed to `origin`)

**Worktree:** `/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/deepagent-retrieval-handover-df766f`

**State:** 1229 tests pass, 19 skip. 0 behind `master`. Tree clean.

Read `docs/superpowers/handoffs/2026-08-01-deepagent-retrieval-poc-handover.md`
first for what the system *is* and for the earlier defect history. This file is
what changed on 2026-08-02 and what to do next.

## Merge safety

The PR is safe to merge whenever the operator wants. `trec_rag.deepagent_*` is
an experimental SDK and is **not** wired into `run_pipeline`, `official_run`,
organizer exports, or any submission artifact, so nothing in it can affect a
competition run. Nothing is blocking.

## Environment

```bash
cd /home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/deepagent-retrieval-handover-df766f
bash code/tools/setup_env.sh          # builds .venv, auto-detects ROCm
PYTHONPATH=code .venv/bin/python -m pytest code/tests/ -q
```

Use `.venv/bin/python-rocm` for anything touching the reranker (GPU).

Secrets come from two files, both private, never printed:

- `.env` in this worktree, copied from the shared checkout: `INDEX_URL`,
  `PYSERINI_API_TOKEN`, `OPENROUTER_API_KEY`.
- **Phoenix tracing lives only in the Codex worktree:**
  `/home/npatta01/.codex/worktrees/a743/trec_rag_2026/.env` — `PHOENIX_API_KEY`,
  `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_PROJECT_NAME`. You already have this
  one. Source **both** or the run is silently untraced, which cost this session
  several untraced runs:

```bash
set -a; source .env; source /home/npatta01/.codex/worktrees/a743/trec_rag_2026/.env; set +a
```

## What landed

- **Passage-first retrieval** (`deepagent_passages.py`, `search_passages`).
  Fixes the measured collapse of 63 nuggets from 5 documents, one document per
  need. Live: 145 nuggets from 215 documents on topic 224, held on topic 897.
  Breadth is enforced in code — a breadth phase then a per-document cap.
- **Citation integrity.** The sentence splitter broke on `"v."` and list
  markers, so a Supreme Court holding was cited to a span reading `"Plyler v."`.
  Fragments now merge back into their sentence. Degenerate cited spans: 11/548
  → **0/518**. Claim/citation content-word overlap: median 0.62 → **0.84**.
- **Reranker window.** `_SNIPPET_MAX_LENGTH` was our own constant at 512; the
  model handles 32,768. Raised to 1024 (covers every chunk we produce; 2048
  measured byte-identical).
- **Synthesis reserve.** `answerable` was a race between two limits, not a
  quality problem. Reserving coordinator turns produced 7 of 7 on a run that
  stopped at a budget wall — the path that previously guaranteed zero.
- **Importance is a derived selection**, not a claimed label. A nugget is vital
  exactly when the coordinator puts it in a need's `draft_nugget_ids`, capped
  at 5 per need and enforced in the validator.
- **Ledger-driven submission ranking** (`deepagent_submission.py`). Imported
  **only by tests** — deliberately, on advice from two reviewers.
- **Middleware spans hidden from traces** by default; set
  `DEEPAGENT_TRACE_MIDDLEWARE=1` to keep them. Filtering happens at export, so
  execution is unchanged.
- **Seven defects fixed** from two independent code reviews, five of them
  unit-test-invisible. See commits `ebced2e` and `4f2d641`.

## The next task

**The coordinator can end a run by replying with no tool call.** Nothing
guarded that path. Two live runs finished with every need holding grounded
nuggets and an empty `draft_nugget_ids`, which zeroes `drafted_count` — the
submission ranker's highest-weighted feature — for every document. A third run
on the same topic reached 7 of 7, so it is non-deterministic.

The mechanism is built and tested: `MainToolFilterMiddleware.wrap_model_call`
bounces such an exit **once** into a forced closeout, reusing the existing
`_CLOSEOUT_DIRECTIVE` and `_directed`. It is currently **inert** because its
`closeout_pending` predicate defaults to `None`.

Three steps activate it:

1. In `DeepAgentRetriever.retrieve` (`deepagent_retrieval.py`), build a closure
   over `coverage_state` returning True when any need with live,
   non-superseded nuggets has an empty `draft_nugget_ids`. Thread it through a
   new optional `AgentToolset` field into
   `MainToolFilterMiddleware(..., closeout_pending=...)` at `_create_agent`.
   The live-nugget filter matters: a need whose only nuggets are superseded
   would otherwise trigger a bounce whose forced call cannot legally succeed.
2. Add a `closeout_refused` rung to the stopping-reason chain in `retrieve`,
   below `budget_exhausted` and above `agent_completed`, fed by
   `budget.closeout_refused()`.
3. Register `"closeout_refused"` in `_STOPPING_REASONS` in
   `deepagent_tracing.py`. **Omitting it drops the trace silently**, because
   `record_result` is wrapped in `try/except`.

Then the load-bearing test: drive `langchain.agents.create_agent` (not
`_create_agent`, which hardwires `ChatOpenRouter`) with a scripted
`FakeMessagesListChatModel` — no tool call, then a tool call, then no tool call
— and assert the forced tool ran once and the graph terminated. This proves the
handler retry works inside real LangGraph execution rather than against a stub.

**Highest risk:** the double `handler()` call is safe *because*
`MainToolFilterMiddleware` is innermost in the middleware list
(`deepagent_retrieval.py`, the `create_deep_agent` call). Reordering that list
would change what the retry re-executes. The real-loop test above is the guard.

### Also open, lower priority

- `search_passages` holds `followup_lock` across the hosted search *and* the
  ~13s GPU scoring pass, serialising researchers at ~20s each and converting
  the concurrency budget into wall-clock.
- `support_ratio` is computed at nugget creation and never recomputed when
  `add_evidence` appends a better citation.
- `vital_count` in `deepagent_submission.py` is provably dead — it always
  equals `drafted_count` now.
- `claim_support_ratio` uses substring matching, so "cat" matches
  "concatenate", inflating a tiebreaker term.
- Wiring the submission ranker into the runner. Both reviewers said shadow-run
  and ablate first; it is degenerate on both measured runs in opposite
  directions (everything drafted on one, nothing on the other).

## Two things to carry, learned the hard way

**Prompts do not produce separation here; validators do.** A researcher told
explicitly "do not mark everything vital: the label only helps if it separates"
returned **94% vital**, uniformly across every need. Every fix that held this
session was enforced in code. Assume the same of anything you are tempted to
solve with prompt wording.

**Do not trust n=1.** A chunk-size result that looked decisive on one topic
(P@5 0.800 vs 0.400) completely evaporated across 22 topics — 3500 vs 2000 came
out at 11 wins to 11 losses. It is written up as retracted in
`reports/2026-08-01-chunk-size-and-reranker-window.md`. Most numbers in this
session are one run per configuration on two topics.

And the older lesson still applies: when touching this code, check what a
refusal *costs*, not just that it refuses. Two of the seven review defects were
exactly that shape again.

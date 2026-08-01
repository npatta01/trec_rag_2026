# DeepAgent Retrieval POC — Handover

**Date:** 2026-08-01

**Branch:** `claude/deepagent-poc`, pushed to `origin`, tip `e34fad5`

**Worktree:** `/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/deepagent-retrieval-poc-handover-3c89a5`

Supersedes `docs/superpowers/handoffs/2026-07-31-deepagent-retrieval-poc-claude-handover.md`,
which is still the reference for the original architecture and design intent.
Read that first for what the system *is*; read this for what changed and what to
do next.

## Start here

The next piece of work is specified in full:
`docs/superpowers/specs/2026-08-01-deepagent-passage-first-retrieval-design.md`.
It carries the problem, the measurements, the recall curve, the recommendation,
an explicit "what not to do", and the open questions. Nothing else needs
deciding before implementation.

## Git state

**Updated 2026-08-01 (later session): master is merged in.** The branch is now
0 behind and 99 ahead. The reconcile was far cheaper than the numbers below
suggested: of 38 files changed on this side and 64 on master, only
`pyproject.toml` and `uv.lock` conflicted, and `code/trec_rag/README.md`
auto-merged. Both openinference packages were kept, since
`deepagent_tracing.py` imports both and a master test requires the
semantic-conventions pin in `project.dependencies` specifically.

Original state, for the record: 87 commits ahead of `master`, 123 behind,
merge-base at PR #20. Master had landed PRs 28/29/30. 74 commits were inherited
from the Codex worktree; the rest were from that session.

Branch is pushed, so the work is no longer only on local disk.

## What this session found

One defect pattern recurred five times: **a mechanism correct about refusing an
action, but wrong about what refusing should cost.** Each looked like normal
operation from the outside.

1. `authorize_round_completion` raised `ValueError` on a model-supplied round
   index, aborting the whole retrieval.
2. A `task` call with a missing or wrong `subagent_type` raised `PermissionError`
   out of the tool, discarding every grounded nugget collected so far.
3. Hitting the coordinator's model-call ceiling reported `agent_completed`, so a
   truncated run claimed a voluntary finish.
4. `finish_task` latched `TASK_BUDGET_EXHAUSTED` the moment the last researcher
   returned, which then blocked closing the round that researcher had just
   completed. The final batch's evidence was discarded on every exhausted run.
5. A single transient `ConnectionError` minted a continuation ticket that
   blocked *every* later query. Because the agent invents fresh queries, it
   could never reissue the failed one, so retrieval stayed wedged. Two full runs
   burned against a closed door while reporting `NO_PROGRESS_STOP` —
   indistinguishable from genuinely finding nothing.

All five are fixed. When touching this code, assume the same class of bug is
still present somewhere: check what a refusal *costs*, not just that it refuses.

## Changes, in order

| Commit | Change |
| --- | --- |
| `0dbd070` | Force research rounds to close; round-sequence errors return a code instead of raising |
| `aa6768d` | `MAIN_MODEL_BUDGET_EXHAUSTED`; require `set_need_status`; surface rejected evidence |
| `a00c5fb` | Model-facing snippet key renamed `chunk_id` → `snippet_id` |
| `f63c88e`, `9892639`, `2382d7f` | Citation handles: evidence is `S3` / `S3.2` / `S3.2-4`, quotes derived not transcribed |
| `89a4d3b` | A denied subagent dispatch no longer ends the run |
| `0436dd0` | `grounded_nugget_count` in the frontier; dispatch empty needs first |
| `9ef5eaf`, `69a66cd` | Concurrency to 1, then reverted to 3 (see "Mistakes") |
| `637dd64` | A run stop no longer discards the last round's finished research |
| `dd33d5d` | `max_rounds` removed entirely; synthesis turn added |
| `a44e7f3` | Only throttling latches retrieval; `trec_rag.continuation` CLI |
| `22a45d6` | Short throttles waited out in place; failed research returns a classified reason |
| `b1fa8b5` | Hosted search paced at 6s |
| `5bb3f22` | Soft/hard deadlines to 30/60 min |
| `bb375d5` | `ToolStrategy` repairs an unparsable evidence bundle |
| `b9d5237`, `e34fad5` | Passage-first design and the recall measurement behind it |

### The citation-handle change is the largest

Researchers used to retype the passage they had just read. The ledger then
validated the copy against text it already owned, and discarded transcription
errors as if they were bad evidence — 15 nuggets lost to a colon becoming an
underscore in one round, 2 more to reworded quotes.

Evidence is now a citation into stored sentence spans. `UNGROUNDED_QUOTE`,
`UNKNOWN_SNIPPET`, `INVALID_EVIDENCE` and `MISSING_EVIDENCE` no longer exist;
`UNKNOWN_CITATION` and `INVALID_CITATION` replace them. A nugget stores the
span and `report()` derives the quote, so a reported quote cannot disagree with
its snippet.

Verified live: 30 citations submitted, 30 resolved, zero citation failures.

## Current defaults

```
researchers=10  concurrent=3  main_models=40  retrieval_calls=100
tools/searches/snippets per researcher = 20/8/16   models per researcher = 30
no_yield_calls=3  no_progress_rounds=2
soft=1800s  hard=3600s
hosted search pacing = 6s per request per host (file-locked SQLite, cross-process)
```

`max_rounds` is deliberately gone. A round cannot close without a finished
researcher, so rounds are already bounded by researchers; a separate cap could
only strand them. A test asserts the field does not exist.

## Topic-224 run history

Every run is a single sample and the spread is large. Treat any one number as
weak evidence.

| Run | Nuggets | Docs | Needs w/ evidence | Answerable | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline (`f9d00a71`) | 6 | 1 | 1 | 0 | 0 rounds closed |
| after round fix | 6 | 1 | 1 | 0 | 3 rounds |
| after honesty fixes | 10 | 2 | 2 | 0 | |
| after citation handles | 16 | 4 | 5 | 0 | zero citation failures |
| three-run batch | 35 / 24 / 0 | 4 / 2 / 0 | — | 0 | third run blocked by the latch |
| after synthesis | 20 | 3 | 2 | **2** | first non-zero `answerable` |
| latest | **63** | **5** | 5 of 8 | 0 | zero retrieval failures |

The latest run grounded 63 nuggets in 5 documents, with per-need counts (9, 5,
23, 14, 12) exactly equal to per-document counts: one document per need. That
1:1 pattern is what the passage-first design exists to fix.

## Verified vs unverified

**Verified live:** citation handles resolve (30/30); rounds close; the honest
stop reasons fire; retrieval survived a run with zero transport failures after
the pacing and short-retry changes.

**Verified by audit, not by run:** the two draft answers from the
`answerable = 2` run are faithful aggregations of their cited nuggets, with no
fabrication. Two caveats found — one nugget (`G4`) garbles its source and the
answer repeats it faithfully, and several N2 nuggets come from a Denver-area
local document while the answer states them universally. Grounding guarantees
the answer reflects the nuggets; nothing checks the nuggets against the world.

**Not verified at all:** the short-throttle retry, the classified failure
reasons, and the `ToolStrategy` bundle repair have never been exercised live,
because they need a run that hits a throttle or an empty completion. They pass
unit tests only.

## Mistakes made this session, so they are not repeated

- **Concurrency.** Dropped `max_concurrent` to 1 to stop throttling, then found
  the throttle came from running whole topics back to back, which concurrency
  does not govern. Reverted. The rate limiter is already cross-process.
- **Rerank depth.** Chose depth 50 because it fit inside the 6s pacing window —
  a real-time constraint that does not apply to an offline submission.
- **Recall threshold.** Measured recall at UMBRELA grade >= 1, which means
  "related but does not answer" and covers 86% of judgements. That made BM25
  look saturated at depth 10. At grade >= 3 it captures 1.6%. Always check what
  a judgement level means before using it.
- **Attribution.** Concluded the empty-structured-output failure was only a
  symptom of blocked retrieval. A later run with no outage reproduced it in 2 of
  10 researchers, so it is an independent recurring failure.

## Environment

```bash
code/tools/setup_env.sh                       # builds .venv, detects ROCm
.venv/bin/python-rocm                         # for GPU work (reranker)
```

Environment variables come from two files, both private, never printed:

- `.env` in this worktree (copied from the shared checkout): `PYSERINI_API_TOKEN`,
  `INDEX_URL`, `OPENROUTER_API_KEY`
- `/home/npatta01/.codex/worktrees/a743/trec_rag_2026/.env`: the Phoenix tracing
  variables, which exist **only** there

Live run:

```bash
set -a
source .env
source /home/npatta01/.codex/worktrees/a743/trec_rag_2026/.env
set +a
PYTHONPATH=code .venv/bin/python-rocm your_script.py
```

If hosted retrieval refuses every query:

```bash
python -m trec_rag.continuation            # what is pending, and when it may retry
python -m trec_rag.continuation --resume   # reissue it; one hosted call
python -m trec_rag.continuation --discard  # drop it, no hosted call
```

## Verification

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/ -q
```

**Updated 2026-08-01 (later session): the 19 failures are gone.** Master had
already converted them to skips, so merging master fixed them. The baseline is
now **1194 passed, 19 skipped, 0 failed**. Treat any failure as yours.

Original note, for the record: 789 passed with 19 pre-existing environmental
failures in `test_all_topic_tethered_rank.py` and
`test_build_all_topic_tethered_report.py`, on a missing private `outputs/`
directory.

## Next work

Items 1, 2 and 4 were done in a later session on 2026-08-01. Item 3 is open.

1. ~~**Implement the passage-first tool.**~~ Done: `trec_rag.deepagent_passages`
   plus `search_passages`, which is now the researcher's forced first action.
   Diversity is enforced in code by a breadth phase and a per-document cap; the
   document ledger, pagination and citation handles are unchanged. **Never run
   live.** It passes unit and wiring tests only, and writing the wiring test
   caught a bug that would have failed every live passage search, so assume
   more of that class remains.
2. ~~**Measure recall on agent-reformulated queries.**~~ Done, in
   `reports/2026-08-01-reformulated-query-recall.md`. At depth 10, 65% of the
   agent's own reformulations return no answering document, which confirms the
   operator's observation. At depth 1000 that is 2%, so it is a depth problem,
   not a query-quality problem. Pooled across 151 reformulations, recall of
   answering documents reaches 0.952 at depth 1000 against 0.345 for the
   untouched narrative: reformulations are worse per query and much better in
   aggregate. The spec's worry was directionally wrong, and depth 1000 is
   better supported than it argued. One topic; see the report's caveats.
3. **Make `answerable` reliable, or drop it.** Still open. The synthesis turn
   produced answers in one run out of two. It forces which tool the coordinator
   calls but not what it puts in the delta.
4. ~~**Decide the integration target.**~~ Done: master is merged in, 0 behind.

### Corrections to the passage-first spec

The spec's cost table came from a 300-document sample with median 7,336
characters and assumed 4.33 chunks per document at 25ms each. A real depth-1000
pool has median 22,776 characters and yields ~12.6 chunks per document, while
this host scores at ~10ms per chunk. The two errors nearly cancel at depth
1000 — measured 113s against the spec's estimated 108s — but the spec is
increasingly optimistic in between, and "depth 50 is close to free" is wrong:
it costs 6.1s, not the sub-6s that would hide inside the pacing window.

## What not to do

- Do not trust a single run. Counts have ranged 0 to 63 across identical code.
- Do not fuzzy-repair citations, quotes, or identifiers.
- Do not echo raw exception text into agent context or traces; classify it. A
  test enforces this.
- Do not raise `hits` without the passage-first surface. Alone it is wasted.
- Do not reintroduce `max_rounds`.
- Do not run topics back to back without spacing; that is what triggered the
  throttle, not concurrency.
- Do not treat `single_document` vs `multi_document` as a source-independence
  claim. It records observed support only.

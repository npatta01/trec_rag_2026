# Task 3 implementation report

## Scope

Task 3 makes hosted RAGDoll support judging bounded, deterministic, and resumable.
Only these owned files were changed:

- code/trec_rag/offline_evaluation.py
- code/trec_rag/competition_evaluation_report.py
- code/tests/test_offline_evaluation.py
- this report

Accepted submission artifacts, retrieval/RAG documentation, and portal files were
not modified. No hosted judge was run; all judge callables used in tests are
synthetic fixtures.

## Changes

_resolve_judgments now:

- validates positive judge_workers and judge_limit values;
- scans tasks in declared order and resolves cache hits before scheduling any
  hosted work;
- de-duplicates misses by content-addressed judge identity, so repeated tasks
  consume one hosted call;
- selects no more than judge_limit unique misses before creating the worker
  pool, making a one-task probe strict even when more workers are configured;
- uses a bounded ThreadPoolExecutor only for selected misses;
- collects outcomes in task-selection order and performs every cache write on
  the controller thread;
- preserves task order in support_judgments.jsonl, counts failed calls once,
  never caches failures or invalid labels, and retries only uncached failures on
  a later invocation.

The evaluation CLI now exposes --judge-workers N, defaults it to 1, rejects
non-positive values, and rejects non-default values unless --run-judge is
present. The effective worker count is recorded in the private evaluation
manifest and stdout receipt. It is not included in friendly-report presentation
data, so rendered HTML does not expose volatile scheduling details and remains
independent of the worker count.

## TDD evidence

### RED

Added concurrency, probe-bound, cache-hit, failure-resume, worker-validation,
controller-write, receipt, and render-determinism regressions to
test_offline_evaluation.py. The initial focused run failed for the expected
missing implementation: the existing resolver ran sequentially
(max_active == 1), the manifest had no judge_workers field, and the
failure-resume assertion exposed the missing worker-aware path. The test
fixture's one incorrect hard-coded cache-hit assumption was corrected before
implementation; no production code was retained from that red phase.

### GREEN

After implementation:

    10 passed, 98 deselected

for the focused worker/probe/resume/controller suite.

## Verification

Completed Task 3 focused evaluator suite:

    PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py -q
    108 passed, 28 subtests passed

Completed Task 1–3 regression suite:

    PYTHONPATH=code .venv/bin/python -m pytest \
      code/tests/test_offline_evaluation.py \
      code/tests/test_ragdoll_io.py \
      code/tests/test_accepted_rag_evaluation.py -q
    169 passed, 28 subtests passed

Compile and whitespace checks:

    PYTHONPATH=code .venv/bin/python -m compileall -q \
      code/trec_rag/offline_evaluation.py \
      code/trec_rag/competition_evaluation_report.py \
      code/tests/test_offline_evaluation.py
    git diff --check

Both commands completed successfully.

The first repository-wide run completed 3,225 tests and 19 skips, but four
unrelated test_retrieval_cache_shard_workflow.py cases failed in
_clean_launcher_checkout while cloning the workspace (before URL validation).
Rerunning exactly those four cases independently passed:

    4 passed, 44 deselected

No Task 3 test failed in the repository-wide run.

## Self-review

- Cache reads and writes are controller-owned; worker threads invoke only the
  injected judge callable.
- A probe cannot overschedule: misses are selected before submission and
  duplicate identities do not consume extra quota.
- Results are assembled by original task index, not future completion order.
- Exceptions, failed outcomes, invalid labels, and cache conflicts remain
  explicit in the private receipt and do not silently become completed labels.
- Existing cache-only behavior remains zero-hosted-call by default, and the
  CLI's default worker count remains inert without --run-judge.
- Friendly presentation does not read the new worker field, preserving output
  determinism across worker counts.

## Commit

Planned commit message: feat: bound resumable RAGDoll judging

Committed as the Task 3 commit in this worktree with the message above.

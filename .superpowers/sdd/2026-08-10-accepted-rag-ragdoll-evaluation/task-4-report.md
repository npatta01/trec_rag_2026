# Task 4 report — preflight the accepted RAG evaluation workflow

## Scope

Task 4 documents and contract-tests the accepted Evalbase RAGDoll support-evaluation
workflow. Only these owned files were changed:

- `.agents/skills/trec-rag-competition-debug-report/SKILL.md`
- `code/trec_rag/README.md`
- `code/tests/test_offline_evaluation.py`
- this report

The five accepted organizer artifacts, report hubs, `AGENTS.md`, portal files, and
private evaluation outputs were not modified. No hosted judge, retrieval, generation,
or other provider call was made.

## Changes

- Added `AcceptedWorkflowDocumentationTests.test_accepted_workflow_documentation`, a
  contract test that requires both operator-facing documents to carry the accepted CLI
  flags, immutable repository-relative JSONL/metadata paths, private handoff and
  multi-stage source-identity paths, probe/full worker bounds, receipt gates, payload
  minimization, unavailable qrels/gold semantics, partial-output rule, and tailnet-only
  report rule.
- Added the canonical four-phase sequence to both documents:
  cache-only inventory → one-task `--judge-limit 1` probe with
  `--judge-workers 1` → resumable full judging with bounded
  `--judge-workers 4` → fresh-work-directory cache-only replay.
- Documented the exact accepted files:
  `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl`,
  `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/multistage/rag_output_trec_rag_2026.jsonl`,
  and `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/metadata.json`.
- Documented the private handoff
  `/home/npatta01/data/competitions/trec_rag_2026/outputs/facet-deepseek-b40-v3/generation_handoff_manifest.json`
  and preserved multi-stage identity
  `/home/npatta01/.codex/worktrees/rag26-ms1-full-run-1786308971/outputs/rag26-ms1-multistage-final/work/multistage_generation_identity.json`.
  Single-pass missing generation identity remains explicitly unavailable.
- Required pre-probe receipts to capture `completed_judgments`,
  `missing_judgments`, `failed_judgments`, `conflicting_judgments`,
  `reused_from_cache`, `hosted_calls`, and `fully_judged`. A partial output cannot be
  called complete; completion requires a cache-only replay with zero missing, failed,
  and conflicting judgments and `fully_judged: true`.
- Limited the described judge payload to one generated statement, cited selected-
  evidence text, and narrative/source metadata for that citation task. Retrieval TSVs,
  full-text archives, qrels, gold nuggets, generated claims as gold, credentials,
  unrelated topic data, and provider events remain excluded/private.
- Preserved explicit `Unavailable` semantics for qrels-based retrieval metrics and
  nugget coverage when qrels, released gold nuggets, or complete assignments are absent.
  Only privacy-scanned friendly HTML may use the existing tailnet-only portal; adjacent
  reports, manifests, judgments, raw events, and JSONL remain private.

## TDD evidence

### RED

Added the documentation contract test before changing either document. The focused run
failed for the intended missing-contract reason:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
32 failed, 1 passed, 109 deselected, 24 subtests passed
```

Failures were missing accepted flags, immutable paths, worker/probe/replay wording,
receipt fields, payload constraints, and tailnet-only documentation.

### GREEN

After the two documentation updates and the test-only case-insensitive text comparison:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
1 passed, 109 deselected, 56 subtests passed
```

## Verification

Focused and documentation checks:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py \
  code/tests/test_accepted_rag_evaluation.py \
  code/tests/test_ragdoll_io.py -q
171 passed, 84 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report_skill.py -q
12 passed, 10 subtests passed
```

The actual CLI help was inspected and includes `--accepted-rag`,
`--accepted-bundle-metadata`, `--handoff-manifest`, `--source-identity`,
`--run-judge`, `--judge-limit`, `--judge-workers`, `--work-dir`, `--cache-dir`,
and `--output`. No repository-specific Markdown/link checker was present; the
skill regression is the available documentation check.

Clean full repository suite:

```text
PYTHONPATH=code .venv/bin/python -m pytest -q
3233 passed, 19 skipped, 117 subtests passed in 112.95s
```

Final hygiene checks:

```text
git diff --check
PYTHONPATH=code .venv/bin/python -m compileall -q \
  code/tests/test_offline_evaluation.py
```

Both completed successfully. During cleanup, stale pytest-owned temporary trees and
the fixed retrieval-shard test cache were moved to the user trash rather than deleted;
they are recoverable under `/home/npatta01/.local/share/Trash/files/`.

## Self-review

- Both docs use the implemented `trec_rag.competition_evaluation_report` CLI and its
  accepted-mode flags; no unsupported command or legacy judgment import is described.
- The probe is explicitly one call and one worker. Full judging is resumable and bounded
  at four workers, with the same cache and cached-task reuse requirements.
- The final replay is cache-only and is the only completion gate; partial receipts remain
  incomplete and unavailable metrics are never represented as zero.
- Accepted artifact immutability, missing single-pass generation identity, payload
  minimization, private detail artifacts, and tailnet-only HTML serving are stated in
  both operator-facing documents and guarded by the contract test.
- No organizer artifact, report hub, portal file, or unrelated worktree change was
  touched. No hosted calls were made.

## Fix Round 1

### Reviewer findings addressed

- Replaced the broad document keyword scan with section-anchored semantic checks for
  the accepted-workflow sections in both operator documents. The test now requires the
  cache-only inventory and final replay to assert `manifest.scope.topic_ids` against the
  exact ordered 119-topic handoff scope, `judgment_tasks: 3155` for `rag26-ss1`, and
  `judgment_tasks: 7008` for `rag26-ms1-final`. The final replay additionally requires
  zero missing, failed, and conflicting judgments plus `fully_judged: true`.
- Removed the incorrect requirement that hosted calls equal raw remaining misses. Both
  documents now state that task counts can exceed unique judge prompt identities, one
  hosted success can fill multiple tasks, and `hosted_calls` is bounded by selected
  unique uncached identities. The fresh cache-only replay remains the exact completion
  authority.
- Corrected egress language: local private task objects retain narrative/source metadata
  under `metadata.source`, while the provider prompt contains only one generated statement
  and cited selected-evidence text. `task_id`, `evaluator`, and rendered `instruction` are
  identified as transport/evaluator envelope fields; no narrative/source metadata or
  unrelated topic data is in the provider prompt.

### RED and GREEN evidence

The first fix-round contract run failed because the existing docs lacked the anchored
scope/task gates, deduplication semantics, and exact egress distinction:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
2 failed, 1 passed, 109 deselected, 24 subtests passed
```

After updating both documents and the contract test:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
1 passed, 109 deselected, 60 subtests passed
```

### Fix-round verification

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py \
  code/tests/test_accepted_rag_evaluation.py \
  code/tests/test_ragdoll_io.py -q
171 passed, 88 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report_skill.py -q
12 passed, 10 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest -q
3233 passed, 19 skipped, 121 subtests passed in 110.98s
```

`git diff --check` and targeted test compilation passed. No hosted calls were made.
The fresh full suite started after moving only stale pytest-owned temporary fixtures to
the recoverable user Trash.

## Fix Round 2

### Reviewer precision findings addressed

- Corrected the scope receipt distinction in both operator documents. The stdout receipt's
  top-level `topic_ids` is checked first; the JSON manifest named by that receipt's
  `manifest_path` is then checked at `scope.topic_ids`. The contract test rejects the
  incorrect `manifest.scope.topic_ids` claim in either accepted-workflow section.
- Added the accepted identity totals: `3155 judgment tasks / 3148 unique judge identities`
  for `rag26-ss1`, and `7008 judgment tasks / 6974 unique judge identities` for
  `rag26-ms1-final`. After one successful new identity probe, the bounded resume hard
  maxima are `hosted_calls <= 3147` and `hosted_calls <= 6973`, respectively. Inventory
  and probe receipts determine actual calls; these maxima do not assume an empty cache,
  and the fresh cache-only replay remains the exact completion authority.
- Replaced the transport-envelope claim with the three implemented data layers: private
  `support_input.jsonl` retains the accepted narrative in row `metadata`; private
  `support_tasks.jsonl` retains only the listed provenance fields in `metadata.source`
  (including same-topic `sentence_context`, not narrative), while `task_id`, `evaluator`,
  and `instruction` are local `run_prompt` arguments/bookkeeping; the provider-visible
  rendered prompt contains only the generated statement and cited selected-evidence text.

### TDD RED and GREEN evidence

The round-2 contract test failed against the old scope wording, missing identity totals and
bounds, and missing three-layer payload contract:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
36 failed, 1 passed, 109 deselected, 42 subtests passed
```

After updating both operator documents and tightening the section-anchored assertions:

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
1 passed, 109 deselected, 78 subtests passed
```

### Round-2 verification

```text
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py \
  code/tests/test_accepted_rag_evaluation.py \
  code/tests/test_ragdoll_io.py -q
171 passed, 106 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report_skill.py -q
12 passed, 10 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest -q
3233 passed, 19 skipped, 139 subtests passed in 109.09s
```

`git diff --check` and targeted compilation passed after the final edits. The full suite
started from a clean test-temp state; generated pytest and retrieval-shard trees were moved
to recoverable user Trash under `/home/npatta01/.local/share/Trash/files/`. No hosted calls,
accepted-artifact edits, report-hub edits, portal edits, or unrelated worktree changes were
made.

## Commit

Fix-round 2 commit message: `docs: clarify accepted RAG evaluation precision`.

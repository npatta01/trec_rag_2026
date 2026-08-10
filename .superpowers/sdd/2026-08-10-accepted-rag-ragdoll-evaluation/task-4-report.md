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

## Commit

Planned commit message: `docs: preflight accepted RAG evaluations`.

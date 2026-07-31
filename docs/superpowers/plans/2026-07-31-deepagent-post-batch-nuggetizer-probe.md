# DeepAgent Post-Batch Nuggetizer Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with one bounded topic-224 experiment, whether a centralized
post-batch Nuggetizer stage usefully canonicalizes the existing researcher
claims without losing or inventing evidence.

**Architecture:** A throwaway Python runner reads one existing Phoenix trace,
reconstructs the researcher bundles and snippet observations in memory, and
passes a validated alias-based request through the repository's existing
Nuggetizer/OpenRouter adapter. Pure helper functions build and compare paths A
and B; the command-line shell performs Phoenix and hosted I/O. Nothing is wired
into the live DeepAgent graph.

**Tech Stack:** Python 3.12, Phoenix client, `nuggetizer==0.0.5`, the existing
canonical-nugget contract, OpenRouter DeepSeek V4 Flash, standard-library JSON.

## Global Constraints

- Reuse trace `f9d00a71e02ac6f940deb75a5090db5b` and topic `224`; do not retrieve again.
- Stop before cloud unless all six baseline nuggets and their evidence can be
  reconstructed and every quote is contained in its observed snippet.
- Make at most one hosted transport attempt, with the existing 120-second
  timeout and no retry.
- Do not persist trace payloads, snippets, cloud responses, or credentials.
- Print only a sanitized comparison: claims, aliases, counts, mappings,
  grounding status, model, call count, and latency.
- This is a POC: no production integration and no new test suite.

---

## Task 1: Build and Preflight the Probe

**Files:**
- Create: `code/trec_rag/post_batch_nuggetizer_probe.py`
- Modify: `docs/superpowers/plans/2026-07-31-deepagent-post-batch-nuggetizer-probe.md`

- [x] Add typed in-memory records and pure helpers that locate the five
  researcher task outputs, reconstruct the final six ledger nuggets, join
  evidence references to snippet observations, and build stable aliases.
- [x] Build one `CanonicalNuggetRequest` with the exact topic narrative,
  provisional claim text, and grounded evidence. Preserve each provisional
  nugget as fallback path A.
- [x] Add comparison helpers for normalized duplicates, evidence retention,
  provisional-to-canonical mapping, distinct documents, and grounding errors.
- [x] Add a thin runner with `--dry-run`. It must query Phoenix but make no
  hosted call in dry-run mode.
- [x] Run the dry-run and require exactly five bundles, six baseline nuggets,
  sixteen observed snippets, zero grounding failures, and zero hosted calls.

  Result: the structural counts passed and hosted calls remained zero, but the
  safety gate failed because `R1-N1:p003:e1` and `R1-N1:p004:e1` cite quotes
  not contained in their stated snippets. Per the approved failure policy, the
  hosted stage stopped before constructing the OpenRouter backend.

## Task 2: Execute One Hosted Comparison

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-deepagent-post-batch-nuggetizer-probe.md`

- [ ] Run the probe once through `NuggetizerCanonicalNuggetBackend`, bounded by
  the adapter's 120-second timeout and an outer process timeout, with no retry.
- [ ] Record the sanitized A/B output in the terminal only and assess whether
  path B reduced genuine redundancy without evidence loss or over-merging.
- [ ] Re-run a no-cloud syntax/import check, inspect the diff and git status,
  and commit only the probe and plan.

## Task 3: Grounded-Ledger Continuation

**Files:**
- Modify: `code/trec_rag/post_batch_nuggetizer_probe.py`
- Modify: `docs/superpowers/plans/2026-07-31-deepagent-post-batch-nuggetizer-probe.md`

- [ ] Add an explicit `--input-source ledger` mode that builds the canonical
  request from the same trace's six accepted ledger nuggets. Preserve
  `--input-source researcher` as the failing diagnostic path; never repair or
  accept its two invalid citations.
- [ ] Dry-run ledger mode and require 5 bundles, 6 inputs, 6 baseline nuggets,
  16 snippets, zero grounding failures, and zero hosted calls.
- [ ] Source the existing OpenRouter environment and run ledger mode once with
  a 150-second outer process limit. Require at most one adapter transport call;
  do not retry.
- [ ] Inspect the sanitized mapping for claim loss, evidence orphans, exact
  duplicates, and materially incorrect merges. Record the result below.
- [ ] Run a fresh syntax/import check and `git diff --check`, then commit only
  the approved POC files.

## Verification Evidence

- Dry-run: 5 bundles, 6 provisional nuggets, 6 baseline nuggets, 16 snippets,
  2 ungrounded researcher citations, and 0 hosted calls.
- Hosted comparison: deliberately not run; blocked by the pre-cloud grounding
  gate in the approved design.
- Final syntax/import check: passed with `.venv/bin/python`; `git diff --check`
  also passed. The fresh dry-run reproduced the same two grounding failures in
  5.1 seconds with zero hosted calls.

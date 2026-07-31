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

- [ ] Add typed in-memory records and pure helpers that locate the five
  researcher task outputs, reconstruct the final six ledger nuggets, join
  evidence references to snippet observations, and build stable aliases.
- [ ] Build one `CanonicalNuggetRequest` with the exact topic narrative,
  provisional claim text, and grounded evidence. Preserve each provisional
  nugget as fallback path A.
- [ ] Add comparison helpers for normalized duplicates, evidence retention,
  provisional-to-canonical mapping, distinct documents, and grounding errors.
- [ ] Add a thin runner with `--dry-run`. It must query Phoenix but make no
  hosted call in dry-run mode.
- [ ] Run the dry-run and require exactly five bundles, six baseline nuggets,
  sixteen observed snippets, zero grounding failures, and zero hosted calls.

## Task 2: Execute One Hosted Comparison

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-deepagent-post-batch-nuggetizer-probe.md`

- [ ] Run the probe once through `NuggetizerCanonicalNuggetBackend`, bounded by
  the adapter's 120-second timeout and an outer process timeout, with no retry.
- [ ] Record the sanitized A/B output in the terminal only and assess whether
  path B reduced genuine redundancy without evidence loss or over-merging.
- [ ] Re-run a no-cloud syntax/import check, inspect the diff and git status,
  and commit only the probe and plan.

## Verification Evidence

- Dry-run: pending.
- Hosted comparison: pending.
- Final syntax/import check: pending.

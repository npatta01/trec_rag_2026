# Post-qrels Cascade Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the advisor's fixed RRF→GLOBAL→DUAL positional cascade using only the sealed candidate rankings and the existing qrels projection.

**Architecture:** A qrels-blind builder creates and hashes a complete cascade permutation for every frozen topic. A separate post-qrels evaluator verifies that freeze, joins the existing qrels projection, computes the same ranking metrics as the original evaluation, and applies the advisor's fixed stop rules. The existing pre-qrels freeze and evaluation remain immutable.

**Tech Stack:** Python 3.12, JSON/JSONL artifacts, pytest, the repository's existing ranking metric functions, and the portable HTML report builder.

## Global Constraints

- Read only `outputs/rag25_deep_facet_candidates_v1/freeze_v1/` when building the cascade.
- Use exact RRF ranks 1–10, then 90 unseen documents in GLOBAL order, then all remaining documents in DUAL order.
- Preserve a complete, duplicate-free permutation of `U_accepted` for each of topics `219`, `72`, `300`, and `84`.
- Do not retrieve, call an endpoint, run model inference, tune boundaries, add weights, or filter candidates.
- Do not touch protected topics `144`, `213`, `224`, `407`, or `515`.
- Label all cascade findings post-qrels diagnostic evidence, not confirmatory evidence or production promotion.
- Create new downstream outputs; never modify `freeze_v1` or `evaluation_v1`.

---

### Task 1: Build and seal the cascade

**Files:**
- Create: `code/trec_rag/deep_facet_candidate_cascade.py`
- Create: `code/tests/test_deep_facet_candidate_cascade.py`

**Interfaces:**
- Consumes: the verified `freeze_v1/SEALED.json` and `freeze_v1/rankings.jsonl`.
- Produces: a create-only directory containing `rankings.jsonl`, `parameters.json`, `input_binding.json`, and `SEALED.json`.

- [x] Write failing tests proving exact head preservation, GLOBAL allocation through rank 100, DUAL completion, determinism, complete-permutation validation, and the absence of qrels inputs.
- [x] Run the focused test and confirm it fails because the module is absent.
- [x] Implement the minimum builder and seal verifier required by the tests.
- [x] Run the focused test and the existing ranking tests.

### Task 2: Evaluate fixed guards and update the report

**Files:**
- Modify: `code/trec_rag/deep_facet_candidate_cascade.py`
- Modify: `code/tests/test_deep_facet_candidate_cascade.py`
- Modify: `code/trec_rag/build_deep_facet_candidate_report.py`
- Modify: `code/tests/test_build_deep_facet_candidate_report.py`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/README.md`
- Update generated: `reports/experiments/deep_facet_candidate_pilot_v1/artifact.json`
- Update generated: `reports/experiments/deep_facet_candidate_pilot_v1/report_data.sqlite`
- Update generated: `reports/experiments/deep_facet_candidate_pilot_v1/report.html`
- Update generated: `reports/experiments/deep_facet_candidate_pilot_v1/summary.json`

**Interfaces:**
- Consumes: the verified cascade freeze, `evaluation_v1/qrels_projection.jsonl`, and the verified original metrics.
- Produces: create-only diagnostic `metrics.json`, `decision.json`, and `summary.json`, plus an updated rendered report.

- [x] Write failing tests for metric parity at nDCG@10 and all fixed stop guards.
- [x] Implement evaluation from the projected qrels and verified original metrics.
- [x] Run the builder, freeze the rankings, then run the evaluator exactly once.
- [x] Write failing report tests for the cascade arm, result narrative, guard table, and diagnostic labeling.
- [x] Extend the report builder, rebuild its canonical artifact/database, and render standalone HTML.
- [x] Verify targeted tests, the broader deep-facet test suite, immutable source hashes, `git diff --check`, and desktop/mobile rendering.
- [x] Ask the advisor to review the measured result and record that review in the report.

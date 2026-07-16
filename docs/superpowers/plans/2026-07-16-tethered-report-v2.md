# Tethered Facet Diagnostic Report v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit an authenticated v2 diagnostic report whose JSON, summary, and accessible HTML state only evidence-supported conclusions and include a derived full-union recall ceiling.

**Architecture:** Keep authentication in `_verify_sources`, add small pure derivation helpers in `build_tethered_facet_report.py`, and make `build_artifact` the single v2 evidence model consumed by HTML/summary/database writers. Tests use a real-shaped synthetic trust chain; no upstream artifact is rewritten.

**Tech Stack:** Python 3.12, pytest, standalone HTML/CSS, JSON/JSONL, SQLite.

## Global Constraints

- Preserve existing generated report v1; builder reruns emit `tethered-facet-diagnostic-report-v2`.
- Derive every count and metric from already-authenticated Task 3/4 evidence; do not hardcode production values.
- Do not read original qrels, retrieve documents, run evaluation/inference, use network access, or mutate outputs.

---

### Task 1: Real-shaped v2 regression fixture

**Files:**
- Modify: `code/tests/test_build_tethered_facet_report.py`

**Interfaces:**
- Consumes: existing `sources` fixture and `build_artifact`/`build_report`.
- Produces: authenticated Task 3 arm-identical ranking identities, 177 novel IDs in the union, aggregate/per-topic metrics with protected `ndcg@100`, full warning definitions, and a mechanical-fail decision.

- [ ] Add failing assertions for schema v2, `not_established`, false compatibility boolean, exact answer copy, derived warning totals, novel-pool wording/membership, numerical mechanical failure, next-step depths/AUC, full-union per-topic and micro arithmetic, representative grade distributions, and accessible table captions/headers.
- [ ] Add fail-closed tests that restamp an arm-only ranking identity, a duplicate ranking identity, a warning aggregate source row, and a novel ID absent from the accepted union.
- [ ] Run `TMPDIR=/dev/shm .venv/bin/python -m pytest code/tests/test_build_tethered_facet_report.py -q`; expected RED on missing v2 fields and old positive noise claim.

### Task 2: Authenticated v2 evidence derivation

**Files:**
- Modify: `code/trec_rag/build_tethered_facet_report.py`

**Interfaces:**
- Consumes: `verified["task3_ranking_rows"]`, `verified["authenticated_qrels"]`, `verified["authenticated_novel"]`, Task 4 metrics/diagnostics/decision.
- Produces: `_warning_pattern_totals`, `_representative_grade_distribution`, `_full_union_ceiling`, `_mechanical_explanation`, and v2 artifact fields.

- [ ] Set `SCHEMA_VERSION = "tethered-facet-diagnostic-report-v2"`.
- [ ] Implement warning totals by summing validated diagnostic rows for each arm/pattern; reject absent arms or count disagreement.
- [ ] Implement representative grade histograms overall/by movement from every displayed row.
- [ ] Implement full-union identity validation: exact arms/topics, positive unique ranks and documents per arm/topic, arm set equality, per-topic grade-2+ intersection/denominator/recall, and micro reconciliation.
- [ ] Verify every authenticated novel ID is in the shared union, then emit the no-discovery provenance statement.
- [ ] Derive aggregate TETHERED/RRF Recall@500 values/delta, topic-84 delta, and exact protected `ndcg@100` equality from metrics; reject missing/inconsistent evidence.
- [ ] Emit explicit noise conclusion/false boolean, mandated next-step experiment, warning caveat, full-union ceiling, mechanical explanation, and representative distribution.
- [ ] Run the focused report tests; expected GREEN.

### Task 3: Accessible v2 HTML and durable outputs

**Files:**
- Modify: `code/trec_rag/build_tethered_facet_report.py`
- Modify: `code/tests/test_build_tethered_facet_report.py`

**Interfaces:**
- Consumes: v2 artifact fields from Task 2.
- Produces: corrected standalone HTML and matching summary/artifact schema.

- [ ] Replace the noise answer with “Judged-relevant yield improved; net noise reduction not established.” and remove example-based aggregate inference.
- [ ] Render arm-level warning totals with the crude-pattern/wrong-domain caveat.
- [ ] Render novel provenance, numerical mechanical-fail explanation, full-union ceiling table/micro row, representative grade-distribution disclosure, and the offline soft-coverage next step.
- [ ] Preserve captions, scoped headers, keyboard focus, mobile table overflow, and no external dependencies.
- [ ] Run report tests and inspect deterministic HTML assertions; expected GREEN.

### Task 4: Verification and commit

**Files:**
- Verify only the files above plus design/plan documents.

**Interfaces:**
- Consumes: completed v2 builder/tests.
- Produces: one scoped implementation commit and verification evidence.

- [ ] Run `TMPDIR=/dev/shm .venv/bin/python -m pytest code/tests/test_build_tethered_facet_report.py -q`.
- [ ] Run the five-file full tethered suite; expect all tests pass.
- [ ] Run the three-file legacy compatibility suite; expect all tests pass.
- [ ] Run `git diff --check` and `py_compile` for changed Python files.
- [ ] Stage only scoped files and commit with a concise v2 report message.


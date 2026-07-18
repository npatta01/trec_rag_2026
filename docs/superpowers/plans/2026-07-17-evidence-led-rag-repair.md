# Evidence-Led RAG Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible Topic 31/300 failure postmortem, a bounded offline recovery result, a canonical experiment history, and an expanded verified HTML report that guides the transition into RAG.

**Architecture:** A pure postmortem module reads immutable canonical ranking evidence and optional full local feature/provenance sources, then writes one sanitized analysis record consumed by both Markdown and the existing report builder. Repository indexes and plans become navigation/status surfaces around the per-experiment records. The existing report generator remains the only HTML source, and a separate browser-QA wrapper supplies disk-backed Chrome state on hosts where `/tmp` cannot support SQLite WAL.

**Tech Stack:** Python 3.12, pytest, NumPy, standard-library JSON/CSV/HTML/SQLite/subprocess, Node smoke tests, headless Google Chrome, existing Tailscale Serve portal.

## Global Constraints

- Work on `codex/evidence-led-rag-repair`, based on `origin/master` commit `75e3bf1`.
- Preserve all sealed ranking/evaluation artifacts and canonical root hashes.
- Treat relevance as qrels grade 2 or higher and graded gain as `2**grade - 1` for grades 2 through 4.
- Treat every unjudged document as unknown, never nonrelevant.
- The offline recovery analysis may read existing local artifacts but must issue no retrieval, inference, download, or paid call.
- Keep active/uncommitted work in worktrees `41f9` and `f8e9` unchanged.
- Extend the existing report; do not create a second HTML report surface.
- Keep rendered artifacts private through the existing tailnet-only Serve mapping; never enable Funnel or a public listener.
- Browser QA must use disk-backed temporary/profile/cache directories and clean only directories it creates.
- Every code or behavior change follows red-green TDD; documentation-only generated outputs must be covered by a smoke or contract test.

## Verification evidence (2026-07-17)

- Tasks 1 through 3 completed their red-green cycles and independent reviews. The retained commits are `e8e71a7` plus seal/portability fix `2cdd8c6`, `7978150` plus receipt-status fix `9b877d7`, and `f4339c1` plus URL-hardening fix `9cdfd22`.
- The final integration suite passed `120` tests, including `test_experiment_records.py`; `node reports/index.test.js` passed and `git diff --check` was clean.
- The canonical report verifier passed for `22` topics and `6` arms. The tracked and live report SHA-256 is `1464a026979cd14eceea7bcd5edc751dbd0f34dc6f96124f1447dfab82fc6b06`; the summary SHA-256 is `640e20e0a02942278a87f3aea5948e51e4ae97dbf3c87912a053bf3724d4ec0e`.
- Sanitized JSON/report checks found no credentials, absolute home paths, raw document text/identifier fields, or external runtime dependencies. `artifact.json` verification passed.
- The private portal and direct report returned HTTP 200. The live report body matched the tracked SHA-256, required title/capture/Topic 300 signals were present, and `tailscale serve status --json` showed the tailnet HTTPS file handler at `/home/npatta01/codex-rendered` with no Funnel configuration.
- Whole-branch independent re-review approved remediation commit `28f094c`; no Critical or Important findings remain.

---

### Task 1: Reproducible Topic 31/300 postmortem and bounded recovery replay

**Files:**
- Create: `code/trec_rag/topic_failure_postmortem.py`
- Create: `code/tests/test_topic_failure_postmortem.py`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.json`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.md`
- Modify: `reports/experiments/all_topic_tethered_facet_validation_v1/README.md`

**Interfaces:**
- Consumes: canonical `rankings_v3/rankings.jsonl`, `rankings_v3/audit.jsonl`, UMBRELA qrels, tracked `facet_manifest.json`, and the authenticated full `retrieval/accepted_union.jsonl` plus `scoring/features.jsonl` when replaying ablations.
- Produces: typed boundary-analysis, recovery-replay, and Markdown-rendering functions plus sanitized deterministic `postmortem.json`/`postmortem.md`.

- [x] **Step 1: Write failing fixture tests for cutoff mechanics and unknown labels**

```python
def test_boundary_analysis_keeps_unjudged_separate() -> None:
    result = analyze_boundary_changes(
        baseline=["a", "b", "c", "d"],
        candidate=["a", "x", "y", "d"],
        qrels={"a": 4, "b": 2, "c": 1, "x": 2},
        depth=3,
    )
    assert result["outgoing"] == {"total": 2, "known_relevant": 1, "judged_below_2": 1, "unjudged": 0}
    assert result["incoming"] == {"total": 2, "known_relevant": 1, "judged_below_2": 0, "unjudged": 1}


def test_rank_cap_keeps_original_candidates_and_rejects_deep_facet_only() -> None:
    provenance = {
        "original": {"original_rank": 900, "facet_ranks": []},
        "shallow": {"original_rank": None, "facet_ranks": [87, 130]},
        "deep": {"original_rank": None, "facet_ranks": [101, 140]},
    }
    assert eligible_documents_for_facet_cap(provenance, 100) == {"original", "shallow"}
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m pytest -q code/tests/test_topic_failure_postmortem.py
```

Expected: collection fails because `trec_rag.topic_failure_postmortem` does not exist.

- [x] **Step 3: Implement pure boundary, attribution, and replay helpers**

Implement four public functions with these typed parameters and returns:

- `analyze_boundary_changes(baseline: Sequence[str], candidate: Sequence[str], qrels: Mapping[str, int], depth: int) -> dict[str, object]`
- `eligible_documents_for_facet_cap(provenance: Mapping[str, Mapping[str, object]], cap: int) -> set[str]`
- `replay_recovery_arms(topic_input: Mapping[str, object], qrels: Mapping[str, int], *, facet_rank_cap: int = 100) -> dict[str, object]`
- `render_postmortem(analysis: Mapping[str, object]) -> str`

The replay must preserve the complete accepted-union permutation. Documents
that are original-stream candidates remain eligible. Facet-only documents are
eligible for early promotion only when at least one facet retrieval rank is at
or above the cap; ineligible documents are appended in canonical RRF order.
Also report a no-narrative-score diagnostic using the unchanged remaining DUAL
weights, clearly labeled post-hoc and not promotion-eligible.

- [x] **Step 4: Add CLI validation and canonical regression assertions**

The CLI accepts explicit `--source-root`, `--qrels`, `--facet-manifest`, and
`--output-dir` paths. It verifies canonical seals before analysis and asserts:

```python
assert analysis["topics"]["31"]["primary_at_1000"]["known_relevant_delta"] == -7
assert analysis["topics"]["300"]["primary_at_1000"]["known_relevant_delta"] == -3
assert analysis["topics"]["300"]["facet_bucket_yield"] == {
    "1-50": 0.36180904522613067,
    "51-100": 0.29292929292929293,
    "101-150": 0.06,
    "151-200": 0.08,
}
```

- [x] **Step 5: Run the canonical offline replay and write sanitized outputs**

Set `ALL_TOPIC_SOURCE_ROOT` to the authenticated local all-topic output directory,
then run:

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m trec_rag.topic_failure_postmortem \
  --source-root "$ALL_TOPIC_SOURCE_ROOT" \
  --qrels trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
  --facet-manifest reports/experiments/all_topic_tethered_facet_validation_v1/facet_manifest.json \
  --output-dir reports/experiments/all_topic_tethered_facet_validation_v1
```

Expected: deterministic `postmortem.json` and `postmortem.md`; no document text,
credentials, absolute paths, or raw model scores are written.

- [x] **Step 6: Verify and commit Task 1**

Run:

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m pytest -q \
  code/tests/test_topic_failure_postmortem.py \
  code/tests/test_all_topic_tethered_rank.py \
  code/tests/test_all_topic_tethered_evaluate.py
git diff --check
```

Expected: all focused tests pass and the diff has no whitespace errors.

Commit:

```bash
git add code/trec_rag/topic_failure_postmortem.py \
  code/tests/test_topic_failure_postmortem.py \
  reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.json \
  reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.md \
  reports/experiments/all_topic_tethered_facet_validation_v1/README.md
git commit -m "analyze topic retrieval failures"
```

---

### Task 2: Canonical experiment history and discoverability

**Files:**
- Modify: `experiment.md`
- Modify: `reports/experiments/README.md`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/manifest.yaml`
- Modify: `reports/experiments/runs.csv`
- Modify: `reports/index.html`
- Modify: `reports/index.test.js`
- Modify: `docs/superpowers/plans/2026-07-16-all-topic-tethered-facet-validation.md`
- Create: `code/tests/test_experiment_history.py`

**Interfaces:**
- Consumes: merged experiment records, Task 1 postmortem, and explicitly labeled branch-only/uncommitted evidence.
- Produces: one human state ledger, one machine run row for the merged all-topic experiment, and report navigation that exposes the latest evidence.

- [x] **Step 1: Write failing history and reports-index contract tests**

```python
def test_history_names_merged_decisions_and_separates_unmerged_work() -> None:
    text = (REPO_ROOT / "experiment.md").read_text()
    assert "all_topic_tethered_facet_validation_v1" in text
    assert "Retained" in text and "Rejected" in text and "Active / unmerged" in text
    assert "Topic 31/300 postmortem" in text


def test_runs_index_contains_all_topic_record_once() -> None:
    generated_rows, _ = build_experiment_indexes(EXPERIMENTS)
    rows = list(csv.DictReader((EXPERIMENTS / "runs.csv").open()))
    assert sum(row["experiment_id"] == "all_topic_tethered_facet_validation_v1" for row in rows) == 1
    assert rows == generated_rows
```

Extend `reports/index.test.js` so it fails until the report card links
`experiments/all_topic_tethered_facet_validation_v1/report.html` and exposes an
`Experiment history` link.

- [x] **Step 2: Run contract tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_experiment_history.py
node reports/index.test.js
```

Expected: both commands fail on missing history/report signals.

- [x] **Step 3: Rewrite the human ledger and update machine indexes**

`experiment.md` must contain, in order:

1. current retained configuration and immediate recommendation;
2. merged chronological experiment table;
3. tried/learned/rejected/retained synthesis;
4. Topic 31/300 failure summary and judgment-pool caveat;
5. active/unmerged and branch-only work;
6. next actions: bounded recovery plus source-diverse RAG evidence selection.

Create the all-topic experiment's `manifest.yaml` with `split=dev`,
`topic_count=22`, `retriever=pyserini_remote`, `index=climbmix-400b`,
`ranking=family_balanced_rrf_and_preregistered_dual_arms`, canonical qrels,
sealed-root provenance in notes, and only repository-relative tracked artifact
paths. Correct the old coverage manifest's machine-specific cache path, then
regenerate `runs.csv` through `trec_rag.experiment_records`; do not hand-edit
the generated CSV.

- [x] **Step 4: Reconcile the completed all-topic plan**

Add a dated completion summary linking canonical roots, tests, report, private
bundle, and retained decision. Mark only steps proven by committed artifacts or
sealed receipts as complete; leave any genuinely unperformed administrative
step unchecked with a short reason.

- [x] **Step 5: Verify and commit Task 2**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_experiment_history.py
node reports/index.test.js
git diff --check
```

Expected: Python and Node contracts pass.

Commit:

```bash
git add experiment.md reports/experiments/README.md reports/experiments/runs.csv \
  reports/experiments/all_topic_tethered_facet_validation_v1/manifest.yaml \
  reports/experiments/bm25_candidate_pool_coverage_v1/manifest.yaml \
  reports/index.html reports/index.test.js \
  docs/superpowers/plans/2026-07-16-all-topic-tethered-facet-validation.md \
  code/tests/test_experiment_history.py
git commit -m "document experiment decisions and status"
```

---

### Task 3: Expand the canonical report and make browser QA reliable

**Files:**
- Modify: `code/trec_rag/build_all_topic_tethered_report.py`
- Modify: `code/tests/test_build_all_topic_tethered_report.py`
- Create: `code/tools/run_headless_chrome.py`
- Create: `code/tests/test_run_headless_chrome.py`
- Modify generated: `reports/experiments/all_topic_tethered_facet_validation_v1/summary.json`
- Modify generated: `reports/experiments/all_topic_tethered_facet_validation_v1/report_data.sqlite`
- Modify generated: `reports/experiments/all_topic_tethered_facet_validation_v1/artifact.json`
- Modify generated: `reports/experiments/all_topic_tethered_facet_validation_v1/report.html`

**Interfaces:**
- Consumes: canonical evaluation metrics plus Task 1 `postmortem.json`.
- Produces: graded/binary capture datasets, failure-evidence datasets, expanded progressive HTML, and a reusable `run_headless_chrome.py` command.

- [x] **Step 1: Write failing report assertions**

Add tests requiring:

```python
assert built.summary["capture"]["known_relevant_total"] == 12984
assert built.summary["capture"]["graded_gain_total"] == 84560
assert built.summary["capture"]["rrf_at_1000"]["known_relevant"] == 3953
assert built.summary["capture"]["rrf_at_1000"]["graded_gain"] == 28815
assert built.summary["capture"]["full_union"]["binary_recall"] == pytest.approx(0.3866296980899569)
assert "How much graded evidence do we capture?" in built.html
assert "Judgment-pool dependent" in built.html
assert "Source-diverse evidence selection" in built.html
assert "Topic 300 facet-tail replay" in built.html
```

Add a wrapper test with a fake Chrome executable that records its environment
and arguments. It must prove that `TMPDIR`, `--user-data-dir`, and
`--disk-cache-dir` are all inside the wrapper-created disk-backed root.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m pytest -q \
  code/tests/test_build_all_topic_tethered_report.py \
  code/tests/test_run_headless_chrome.py
```

Expected: failures identify missing capture/postmortem fields and wrapper.

- [x] **Step 3: Extend report datasets, summary, SQLite, and HTML**

Add `graded_recall`, `graded_gain`, `known_relevant_total`, and
`graded_gain_total` to depth rows and SQLite. Add one answer-first section with
sealed depths 100 through full union, followed by a Topic 31/300 section whose
detailed facet attribution and incoming/outgoing counts live inside native
`details/summary` disclosure. Keep all existing sections, ids, source hashes,
tables, charts, and caveats unless directly dependent on the new fields.

- [x] **Step 4: Implement the disk-backed Chrome wrapper**

The wrapper CLI is:

```bash
.venv/bin/python code/tools/run_headless_chrome.py \
  --url file:///absolute/report.html \
  --output /absolute/report.png \
  --width 1440 --height 1100
```

It locates `google-chrome` unless `--browser` is supplied, creates its private
root below `--scratch-root` or `$HOME/.cache/trec-rag/browser-qa`, exports
`TMPDIR` and `XDG_CACHE_HOME`, passes private profile/cache flags, checks the PNG
exists and is nonempty, and removes only the private root in a `finally` block.

- [x] **Step 5: Rebuild and verify the canonical report**

Run:

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m trec_rag.build_all_topic_tethered_report write \
  --root outputs/all_topic_tethered_facet_validation_v1 \
  --report reports/experiments/all_topic_tethered_facet_validation_v1
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m trec_rag.build_all_topic_tethered_report verify \
  --root outputs/all_topic_tethered_facet_validation_v1 \
  --report reports/experiments/all_topic_tethered_facet_validation_v1
```

- [x] **Step 6: Verify desktop/mobile rendering and commit Task 3**

Run focused tests, generate 1440×1100 and 390×844 screenshots through the
wrapper, inspect both screenshots, confirm `scrollWidth <= clientWidth`, and
confirm native `summary` plus focusable labelled table regions remain present.

Commit:

```bash
git add code/trec_rag/build_all_topic_tethered_report.py \
  code/tests/test_build_all_topic_tethered_report.py \
  code/tools/run_headless_chrome.py code/tests/test_run_headless_chrome.py \
  reports/experiments/all_topic_tethered_facet_validation_v1/{summary.json,report_data.sqlite,artifact.json,report.html}
git commit -m "expand retrieval evidence report"
```

---

### Task 4: Integration verification and private artifact handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-17-evidence-led-rag-repair.md`
- Replace derived private copy: `/home/npatta01/codex-rendered/plans/trec-2026-all-topic-tethered-facet-validation-v1.html`
- Modify private portal index description: `/home/npatta01/codex-rendered/index.html`

**Interfaces:**
- Consumes: reviewed Tasks 1 through 3.
- Produces: verified repository state, exact rendered copy, current tailnet-only link, and completed progress evidence.

- [x] **Step 1: Run the full relevant test suite**

```bash
TMPDIR="$HOME/.cache/trec-rag/tmp" .venv/bin/python -m pytest -q \
  code/tests/test_topic_failure_postmortem.py \
  code/tests/test_experiment_history.py \
  code/tests/test_build_all_topic_tethered_report.py \
  code/tests/test_run_headless_chrome.py \
  code/tests/test_all_topic_facet_contract.py \
  code/tests/test_all_topic_tethered_rank.py \
  code/tests/test_all_topic_tethered_evaluate.py
node reports/index.test.js
git diff --check
```

- [x] **Step 2: Run secret, portability, and artifact checks**

Verify no tracked diff contains `.env`, bearer tokens, API tokens, absolute
active-worktree paths, raw document text, or non-tailnet URLs. Verify the report
has no external runtime dependency and matches `artifact.json`.

- [x] **Step 3: Update the authorized private rendered copy**

Copy the exact verified `report.html` to the existing Tailscale Serve document
root and change the portal description from stale canonical v2 wording to
portable canonical v3 plus postmortem/graded-capture wording.

- [x] **Step 4: Verify live delivery**

Confirm:

- portal and direct report URLs return HTTP 200;
- the live body SHA-256 equals tracked `report.html`;
- `tailscale serve status --json` contains the HTTPS file handler and no Funnel
  enablement;
- the live document title, graded-capture section, and Topic 300 disclosure are
  present.

- [x] **Step 5: Complete the plan ledger and final review**

Mark this plan's verified steps complete, record exact test counts/hashes, run a
whole-branch code review from merge base to HEAD, resolve all Critical or
Important findings, and rerun affected tests.

Whole-branch review remediation evidence (2026-07-17):

- [x] Restored the postmortem-only RRF depth-20 diagnostic from the
  authenticated ranking and pinned qrels: 316 / 12,984 known relevant
  (2.4337646334%) and 2,492 / 84,560 graded gain (2.9470198675%). The report
  independently recomputes it from authenticated sources, reconciles both
  denominators to the sealed evaluation totals, labels it separately from
  sealed depths, and rejects a coordinated postmortem-count forgery.
- [x] Separated Topic 300 facet retrieval-stream membership from greedy DUAL
  selection-coverage audit state. The sanitized JSON, Markdown, and HTML name
  each tracked facet id/query formulation and keep known relevant,
  judged-below-2, and unjudged (unknown) counts separate.
- [x] Rebuilt and verified the canonical report for 22 topics / 6 arms. Exact
  SHA-256 values: postmortem JSON
  `4642dd17887ef043ed75e8677909291d0dcf4b0110ed856a5caa4ecc9c7775a8`,
  HTML `1464a026979cd14eceea7bcd5edc751dbd0f34dc6f96124f1447dfab82fc6b06`,
  summary `640e20e0a02942278a87f3aea5948e51e4ae97dbf3c87912a053bf3724d4ec0e`,
  and SQLite
  `5e5d5a653d22e7ea8bdcb06cd9e8ed17756bafcf1e3bbe8fb127fd3619304c28`.
- [x] Fresh relevant integration suite: 120 passed in 22.96 seconds; report
  index smoke test and `git diff --check` passed. Desktop 1440×1100 and mobile
  390×844 screenshots were captured and inspected.
- [x] Replaced the authorized private rendered copy. Portal/direct HTTP status
  is 200, the live body exactly matches the tracked HTML SHA-256 above, and
  Tailscale Serve remains an HTTPS file handler with no Funnel configuration.
- [x] Independent whole-branch re-review from merge base through remediation
  commit `28f094c` passed. It independently reproduced the depth-20 diagnostic,
  rebuilt the postmortem from authenticated full sources, verified the separate
  Topic 300 retrieval-stream and DUAL coverage attribution, reran 120 integration
  tests, and found no remaining Critical or Important issue.

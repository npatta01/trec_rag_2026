# Evidence-Led RAG Repair Design

**Date:** 2026-07-17  
**Status:** Approved  
**Audience:** Technical contributors preparing the TREC RAG 2026 system

## Objective

Turn the repository's fragmented experiment evidence into one dependable
decision trail, complete the Topic 31/300 retrieval-failure analysis, and make
the existing progressive-disclosure HTML report answer the two immediate
questions: how much graded evidence is captured, and whether to keep improving
retrieval or move into end-to-end RAG.

## Decision

Use an evidence-led repair rather than a link-only cleanup or a new experiment
registry framework.

1. Preserve `experiment.md` as the canonical human entrypoint.
2. Keep `reports/experiments/runs.csv` as the machine-readable merged-run
   index and label branch-only or uncommitted work separately in prose.
3. Add a reproducible offline Topic 300 replay that tests the observed
   deep-facet tail failure without retrieval, inference, or new judgments.
4. Add a source-backed Topic 31/300 postmortem beside the existing report.
5. Extend the existing report generator and HTML artifact rather than creating
   a second report surface.
6. Make browser QA reliable on this host with a disk-backed temporary profile.

## Evidence Baseline

The canonical evidence is `origin/master` commit `75e3bf1` plus the sealed
private bundle whose SHA-256 is
`3b726dcff28f8e67e6d4a5cf330e390c150b0a601091af17ae59b5870bc46e59`.
The report must continue to reject superseded v1/v2 ranking and evaluation
roots.

UMBRELA relevance is grade 2 or higher. Graded gain is `2**grade - 1` for
grades 2 through 4 and zero otherwise. Baseline RRF captures:

| Depth | Known relevant | Binary recall | Graded gain | Graded recall |
|---:|---:|---:|---:|---:|
| 20 | 316 / 12,984 | 2.43% | 2,492 / 84,560 | 2.95% |
| 100 | 1,400 / 12,984 | 10.78% | 10,532 / 84,560 | 12.46% |
| 1,000 | 3,953 / 12,984 | 30.45% | 28,815 / 84,560 | 34.08% |
| Full union | 5,020 / 12,984 | 38.66% | 35,840 / 84,560 | 42.38% |

The canonical evaluation already stores depths 100, 250, 500, 1,000, 1,500,
and full-union metrics. The tracked report will expose those sealed metrics.
The depth-20 diagnostic will remain in the postmortem with explicit qrels and
ranking provenance; it will not be inserted into the sealed evaluation root.

## Components

### Offline recovery analyzer

Add `trec_rag.topic_failure_postmortem` with pure functions that:

- load frozen ranking, audit, feature, provenance, qrels, and facet-manifest
  records;
- summarize incoming, outgoing, judged, unjudged, and graded-relevant counts;
- attribute boundary movement to original and facet streams;
- replay Topic 300 using the existing frozen features while excluding facet
  evidence below a configurable per-stream rank cap;
- emit a deterministic JSON analysis artifact and a Markdown postmortem.

The replay must not issue network requests, load a model, mutate sealed
artifacts, or describe unjudged documents as nonrelevant. Its primary test is a
rank-100 cap because Topic 300 facet yield falls from 36.2% and 29.3% in the
first two buckets to 6.0% and 8.0% in the last two.

### Canonical experiment ledger

Rewrite `experiment.md` as an answer-first state ledger with:

- retained baseline and current decision;
- chronological merged experiments and their outcomes;
- active/unmerged and branch-only work clearly separated;
- tried, learned, rejected, retained, and next-action fields;
- direct links to experiment records and the latest report.

Append the all-topic experiment to `runs.csv`, update the tracker convention,
add the latest report to `reports/index.html`, and reconcile the completed
all-topic implementation plan so its status is no longer misleading.

### Existing HTML report

Extend `build_all_topic_tethered_report.py` and its tests to show:

- known-relevant denominators and graded-gain denominators;
- binary and graded recall by sealed evaluation depth;
- the full-union ceiling;
- Topic 31/300 cutoff mechanics and judgment-pool limitations;
- Topic 300 facet-tail attribution and offline replay outcome;
- a progressive-disclosure section for detailed failure evidence;
- the recommendation to run a bounded recovery lane while starting RAG with
  source-diverse evidence selection.

The report remains self-contained, sanitized, keyboard-readable, responsive,
dark-mode aware, and tailnet-only.

### Browser QA wrapper

Add a small terminal-friendly wrapper that creates `TMPDIR`, Chrome profile,
and cache directories on a normal disk-backed filesystem before executing
Chrome. This addresses the verified host failure: SQLite WAL operations return
`disk I/O error` on the `/tmp` tmpfs and cause Chrome screenshot workers to
trap. The wrapper must clean only its own temporary directory.

## Retrieval And RAG Recommendation

Do not build a local dense or SPLADE first-stage index for ClimbMix-400b before
the 2026 deadline. The collection contains about 553 million rows and roughly
400 billion tokens; the official remote interface documents BM25 rather than a
neural index. Preserve this feasibility conclusion in the ledger, not as a new
retrieval experiment.

Run two bounded lanes:

1. **Recovery lane:** replay the Topic 300 facet rank-100 cap, document the
   Topic 31/300 pool bias, and target the weakest RRF top-20 topics without a
   full-corpus index.
2. **RAG lane:** start from retained RRF, select a source-diverse evidence set
   from a deeper candidate pool, then evaluate nugget coverage, sentence-level
   citations, and support. Do not pass the protected top 20 unchanged and call
   that facet-aware RAG.

## Validation

- New analyzer behavior follows red-green TDD with fixture-sized frozen data.
- Recomputed counts must reconcile to canonical metrics at depths 100 through
  1,500 and to qrels-derived depth-20 totals.
- Existing canonical-root, forgery-rejection, and accessibility tests remain
  green.
- Reports-index smoke tests assert the new report link and experiment-history
  entrypoint.
- Browser QA runs desktop and 390-pixel mobile screenshots through the wrapper,
  verifies no horizontal page overflow, exercises native `details` disclosure,
  and checks keyboard-focusable table regions.
- The live tailnet URL must return the exact rebuilt HTML hash with Funnel
  disabled before handoff.

## Non-Goals

- No paid or hosted inference.
- No new corpus retrieval.
- No full local dense or SPLADE index.
- No claim that heuristic text triage is ground-truth relevance.
- No mutation of sealed ranking/evaluation evidence.
- No public deployment, Funnel, Codex Sites, or sharing-permission changes.

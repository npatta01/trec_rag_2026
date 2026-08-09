# Cached Segmentation Rescore and Validation

Date: 2026-08-09
Status: Approved in conversation; written review pending
Scope: the 22 RAG 2025 development topics and the fixed/non-agentic retrieval pipeline

## Objective

Prove whether sentence-aware source segmentation improves selected evidence and
canonical retrieval nuggets without paying for planning, document retrieval, or
passage reranking again. Run the complete 22-topic validation on one dstack GPU
machine, publish private immutable result artifacts, and promote the code only
if the run passes both cache-safety and quality gates.

The selected topics are:

`14, 31, 37, 58, 72, 84, 144, 161, 200, 213, 219, 224, 225, 233, 273, 300,
407, 477, 499, 515, 707, 897`.

## Existing Evidence

The old line-oriented splitter divides hard-wrapped sentences and headings into
short candidate fragments. Across the old 22-topic p4 replay it produced
383,078 candidates. Topic 407 had a median physical-line length of 11
characters and 60.4% of physical lines were shorter than 40 characters.

The private Hugging Face Bucket contains one verified portable cache bundle for
each of the 22 topics. The archives total approximately 1.30 GiB compressed.
The previous all-topic replay proved that these bundles can reconstruct a fresh
cache and yield zero planning, retrieval, passage-scoring, similarity, and
canonicalization work under the old segmentation identity.

The sentence splitter changes candidate and downstream cache identities but
does not change query, document-retrieval, or passage-score identities.
Therefore a global offline replay cannot perform this validation: sentence
scoring, similarity, and usually canonicalization must be recomputed.

## Decisions

### Fail-closed upstream reuse

Add a distinct `cached-upstream-rescore` execution mode. It is not an alias for
online mode or global offline mode.

In this mode:

| Stage | Policy | Permitted work |
| --- | --- | --- |
| Planning | cache-only | cache reads only; a miss aborts before a provider call |
| Document retrieval | cache-only | cache reads only; a miss aborts before a network request |
| Document materialization | cache-only | restored document bytes only; a miss aborts |
| Passage scoring | read-only | restored scores only; a miss aborts before model inference |
| Sentence segmentation | recompute | local CPU/spaCy work |
| Sentence scoring | writable | local GPU inference and cache writes |
| Similarity/MMR features | writable | local inference and cache writes |
| Canonicalization | cache-backed online | cached responses may be reused; new selected groups may make hosted calls |

The mode must be represented in topic jobs and operation receipts rather than
inferred from credentials. A deliberately invalid API token is not a safety
mechanism because it still permits an attempted retrieval request.

Every completed topic must report zero planning misses/provider calls, zero
retrieval misses/network calls, and zero passage-score misses/model batches.
Any nonzero value fails the task before its result can be promoted or
published as valid.

### One machine, fastest useful GPU

Use one on-demand dstack machine so repository setup, bundle download, cache
merge, and model downloads are paid once. Preview live offers immediately
before launch and select the fastest available GPU whose offer the user
approves. Prefer H100-class compute when an acceptable offer exists; otherwise
prefer L40S over the previously proven A40. Do not choose extra VRAM alone: the
selected GPU must improve transformer inference throughput for this workload.

The task remains single-GPU. Sentence-scoring workers share that GPU; they do
not use distributed training or multiple machines.

### Bounded multi-topic concurrency

Topic 407 runs alone first on the leased machine because it has the clearest
known fragmentation baseline. The wrapper validates its cache counters,
artifacts, and structural before/after metrics before continuing.

After the canary passes, run multiple topics concurrently on the same GPU. Use
two workers as the proven floor and permit four only on a 48 GB or larger GPU
when a pre-compute memory check and the canary show sufficient headroom. The
wrapper records worker count, GPU identity, elapsed time, and peak memory. It
must fail rather than silently reduce validation or restart retrieval after an
out-of-memory error.

The concurrency goal is throughput, not maximum process count. Four model
copies that saturate memory without improving topics/hour are worse than two.
The final choice is recorded from the canary and a short two-worker throughput
probe on the same lease; the remaining topics use the faster safe setting.

## Remote Data Flow

1. Transport a clean, committed-only source snapshot through the existing
   dstack launcher pattern.
2. Verify required secret presence without printing values and confirm the
   configured Hugging Face Bucket is private.
3. Require the new result prefix to be empty.
4. Download exactly the 22 existing archive/marker pairs from the pinned
   private source prefix.
5. Verify each archive and merge all bundles into a fresh task-local cache.
   Never copy a live SQLite database between machines.
6. Run topic 407 in `cached-upstream-rescore` mode and apply the canary gates.
7. Select safe two- or four-topic concurrency and run the remaining 21 topics.
8. Validate all topic receipts and the aggregate retrieval export/handoff.
9. Produce a private before/after validation report.
10. Pack, locally verify, upload, download, byte-compare, and semantically
    reverify the new immutable result bundle. Upload its completion marker last.

The destination is a new run-specific private prefix beneath
`trec_rag_2026/experiments/`. Existing bundles are never deleted or
overwritten.

## Quality Validation

### Deterministic structural comparison

Compare old and fixed artifacts for the same 22 narratives and the same
upstream query, document, and passage identities. Report per-topic and macro
values for:

- candidate count, median character length, and fraction shorter than 40
  characters;
- selected-evidence count, median character length, and fragment proxy rate
  (`length < 40` or lacking terminal sentence punctuation);
- heading-only selected evidence;
- selected group provenance, full-passage containment, selection budget, and
  extractive-fallback rate;
- canonical nugget count, median length, empty/fallback rate, and source-group
  coverage.

Topic 407 must improve from the known 11-character median and 60.4% sub-40
baseline. Across all 22 topics, the fixed run must materially reduce candidate
fragmentation and must not worsen selected-evidence fragmentation or invalidate
any authority/provenance check.

### Paired semantic nugget comparison

Structural improvements alone do not prove better information coverage. Use a
paired evaluation for each topic:

1. derive or reuse one frozen answer-obligation plan from the unchanged
   narrative;
2. judge the old canonical retrieval nuggets against that plan;
3. judge the fixed canonical retrieval nuggets against the exact same plan and
   judge contract;
4. record required coverage, strict-full rate, obligation labels, regressions,
   and newly covered obligations.

The comparison must verify identical narrative and plan hashes before scoring.
It must use the same pinned model and prompt identities for both arms. This is a
planner-derived diagnostic, not organizer ground truth, and the report must say
so. Organizer gold nuggets and qrels remain outside the retrieval/generation
runner; a separate read-only development diagnostic may be added only if its
input and interpretation are explicit.

The fixed arm passes the semantic gate when topic-macro required coverage and
strict-full rate do not regress, no topic has an unexplained material coverage
loss, and the report shows either a positive aggregate change or clearly better
faithfulness/readability at equal coverage. Any loss is inspected at the
obligation and source-passage level before promotion.

## Promotion Gate

Promote only when all of the following are true:

- all 22 topics complete on one dstack machine;
- every upstream no-work counter is exactly zero;
- all checkpoint, provenance, handoff, archive, and round-trip validators pass;
- deterministic fragmentation metrics improve and selected evidence does not
  regress;
- paired semantic nugget coverage is better or equal under an identical plan;
- the tracked worktree is clean apart from the intended commits;
- targeted and full relevant tests pass.

Promotion means pushing `codex/fix-sentence-segmentation` and opening a draft
pull request. It does not mean merging to the default branch or publishing any
private output.

## Cost and Call Disclosure

Before launch, report the exact 22 topics, 1.30 GiB source-bundle download,
chosen GPU offer, hourly price, maximum duration/exposure, worker count, and
private output prefix.

Expected remote/hosted work is:

- planning: 0 provider calls;
- document retrieval: 0 network calls;
- passage scoring: 0 model batches;
- sentence scoring and similarity: local GPU work;
- canonicalization: cache hits plus bounded hosted calls for changed groups;
- paired nugget validation: one frozen-plan call and two judge calls per topic
  when no compatible plan cache exists, at most 66 hosted calls across 22
  topics.

Provider responses, caches, selected evidence, canonical nuggets, and reports
remain private and outside git.

## Failure Handling

- Missing or invalid source bundle: stop before GPU scoring.
- Any upstream cache miss: stop before the forbidden provider/network/model
  operation and preserve a private diagnostic receipt.
- Topic 407 quality gate failure: stop before the remaining 21 topics.
- Worker/OOM failure: preserve completed topic artifacts; diagnose on the same
  lease if possible, then resume only incomplete topics at lower concurrency.
- Canonicalization/provider failure: retain resumable state and do not label the
  topic complete.
- Remote publication mismatch: stop; never overwrite the destination prefix.
- Semantic regression: preserve the paired evidence for inspection and do not
  promote the branch.

## Non-goals

- No RAG answer generation.
- No new BM25/Pyserini retrieval.
- No passage reranking.
- No Modal implementation or execution.
- No public artifact or service.
- No merge to the default branch.

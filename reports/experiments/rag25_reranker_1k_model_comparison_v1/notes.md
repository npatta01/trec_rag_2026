# RAG25 top-1,000 reranker comparison

Run date: 2026-08-06  
Split: RAG25 development topics (22 topics)  
Record provenance: generated on branch `codex/rag25-reranker-1k`; treat as learned
evidence, not a promotion decision, until held-out validation.

## Technical summary

If forced to choose between these two full-depth max-passage orders, Mixedbread
has higher observed effectiveness than GTE: it reaches `0.406589` nDCG@10
versus `0.337349`, while taking 71m12s versus 42m35s on an NVIDIA A40. GTE is
about 1.67x faster on that full workload, but neither reranker beats retaining
BM25 order at depth 1,000 under these projected judgments (`0.4140` baseline).
The models are not drop-in replacements: their full-depth top-10 overlap is only
25.9%, and their raw logits have different scales.

The result is depth-dependent. At a 100-document reranking prefix, GTE is
slightly ahead on nDCG@10 (`0.531979` vs `0.525054`), with a paired bootstrap
95% interval for GTE−Mixedbread of `[-0.047888, +0.062718]`; the observed
difference is not distinguishable from zero in this 22-topic sample. The best
observed operating point in this post-hoc depth sweep is therefore a shallow
100-document prefix, with GTE a plausible latency choice—not a demonstrated
winner. This experiment does not establish that a GTE-first/Mixedbread-second
cascade would work; that remains a follow-up.

> **Read this comparison with caution:** the qrels do not cover every document
> in the 1,000-document candidate pools. Unjudged documents are treated as
> nonrelevant by this evaluator, so the reported nDCG/precision values are
> pool-based observations, not definitive judgments of model quality. In
> particular, the depth-1,000 decline is partly entangled with judgment
> coverage; use the depth-100 result as a hypothesis to validate, not as a
> settled winner.

## Effectiveness by candidate depth

Candidate depth means the first `N` documents in the same BM25 pool are scored
and reranked. Each document receives the maximum score of its scored passage
windows. nDCG is graded; `precision@k` counts qrel grade `>=2` as relevant.

| scored depth | BM25 nDCG@10 / @20 | Mixedbread nDCG@10 / @20 | GTE nDCG@10 / @20 | Mixedbread Δ@10 | GTE Δ@10 |
|---:|---:|---:|---:|---:|---:|
| 50 | .414 / .413 | .519 / .472 | .503 / .465 | +.105 | +.089 |
| 100 | .414 / .413 | .525 / .507 | **.532 / .506** | +.111 | +.118 |
| 200 | .414 / .413 | .482 / .438 | .451 / .430 | +.068 | +.037 |
| 500 | .414 / .413 | .429 / .388 | .357 / .342 | +.015 | −.057 |
| 1,000 | .414 / .413 | **.407 / .369** | .337 / .300 | −.007 | −.077 |

The corresponding precision values are: P@10/P@20 of Mixedbread/GTE =
`.759/.725` / `.814/.755` at depth 50; `.773/.766` / `.818/.809` at 100;
`.691/.630` / `.664/.648` at 200; `.614/.557` / `.523/.523` at 500; and
`.564/.509` / `.495/.452` at 1,000.

At depth 1,000, Mixedbread wins nDCG@10 on 14 topics and GTE wins on 8, but
both are below the BM25 mean. At depth 100, GTE wins 12 topics and Mixedbread
10; the observed difference is not distinguishable from zero. The falloff
beyond depth 100 is a warning against assuming that a shallow-prefix result
transfers to full-pool reranking.

## Judgment coverage changes with depth

The projected qrels do not judge every document in the 1,000-document pool.
Unjudged documents are scored as nonrelevant by the evaluator, so effectiveness
and pool coverage move together. Top-10 judged rates for Mixedbread/GTE are
100%/100% at depths 50 and 100, 84.1%/81.4% at 200, 71.4%/63.2% at 500, and
66.8%/58.2% at 1,000. The apparent full-depth deficit—especially GTE's—cannot
be interpreted as pure model quality without a denser or held-out judgment
pool.

## Runtime and cost

The local timing is a clean top-100 gate over 27,433 passage windows/model on
the Radeon host. The A40 timing is the complete 237,089-window/model workload;
both remote passes used BF16 and the same pinned software versions.

| environment | Mixedbread | GTE ModernBERT | relative GTE speed |
|---|---:|---:|---:|
| Local Radeon, top 100 | 16m54.71s | 8m53.60s | 1.90x |
| NVIDIA A40, top 1,000 | 71m12s | 42m35s | 1.67x |

The A40 task used `torch 2.9.1+cu128`, `transformers 5.13.0`,
`sentence-transformers 5.6.0`, `semantic-text-splitter 0.32.0`, and an A40
GPU. The successful dstack run cost `$0.9032`; including the failed preflight,
the dstack total was `$0.9518`.

Retrieval itself remained a model-independent remote Pyserini/API step. The
GPU was used for local cross-encoder passage scoring, not for first-stage
retrieval.

## Are the scores or rankings interchangeable?

No. Raw logits are model-specific and should not be thresholded or compared
across models without calibration:

| statistic over 22,000 document max scores | Mixedbread | GTE ModernBERT |
|---|---:|---:|
| range | -4.750 to 10.500 | -0.660 to 3.313 |
| mean | 5.366 | 1.396 |
| standard deviation | 1.961 | .524 |

The Pearson correlation of document max scores is `0.779`, but the mean
per-topic Spearman correlation of the induced document rankings is only
`0.637`. Full-depth top-1 overlap is 4.5%, top-10 overlap is 25.9%, and
top-20 overlap is 29.8%. Use each model's own ordering and evaluate its own
effectiveness; do not merge raw scores directly.

## Scope, method, and saved evidence

- Retrieval: original narrative only, BM25 over `climbmix-400b`, exactly 1,000
  unique documents/topic, 22,000 rows total.
- Passage policy: 3,500-character chunks with 350-character overlap, max input
  length 1,024, maximum passage score as the document score.
- Models: `mixedbread-ai/mxbai-rerank-base-v2` revision
  `3ea9d4dffa7d12a4f366be8e275c349de9fc9865` and
  `Alibaba-NLP/gte-reranker-modernbert-base` revision
  `f7481e6055501a30fb19d090657df9ec1f79ab2c`.
- Evaluation: projected RAG25 qrels
  `rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels`; paired
  bootstrap uses 10,000 topic resamples with seed `20260806`.
- Source score artifacts and full document/passage rankings are private and
  ignored. The tracked record contains only aggregate metrics and topic-level
  deltas; no corpus text, qrels-derived raw rows, secrets, or private bucket
  contents are committed.

The evaluator output includes 22,000 document-ranking rows and 237,089
passage-ranking rows per model. The private source artifacts retain the full
window identity, score-cache key, chunk offsets, and model provenance.

## Limitations and next steps

This is a 22-topic development-set result, not a 2026 test-set result. The
qrels are projected and judgment coverage is not a substitute for held-out
validation. Depth was selected post hoc from five values, so the apparent 100
document operating point must be preregistered and validated elsewhere. We
tested depth 1,000, not 5,000; one chunking policy; one raw-logit
maximum-passage aggregation rule; and no end-to-end answer or citation-quality
metric. More candidates also create more opportunities for spurious passage
maxima; this result should not be generalized to other aggregators or cascades.

The next low-cost experiment should preregister a shallow-prefix policy (for
example 100 or 200), compare it against BM25 and full-depth Mixedbread, and
validate it on held-out/test topics with explicit judgment-coverage reporting.
A GTE-first/Mixedbread-second cascade should be treated as a new experiment
rather than inferred from these rankings.

## Source and evidence pointers

- Tracked aggregate record: `manifest.yaml`, `config.yaml`, `metrics.json`,
  and `topic_metrics.csv` in this directory.
- Private dstack run: `rag25-reranker-1k-a40-20260806-v2`
- Public, privacy-scanned result prefix:
  `hf://buckets/Npatta01/trec-rag-2026-artifacts/trec_rag_2026/experiments/rag25-reranker-1k-a40-20260806-v2`
- The checksumed score artifacts contain identifiers, hashes, scores, and
  runtime metadata, but no raw document text. The ignored experiment workspace
  remains private and is not part of this report PR.

Tables are used instead of charts because this is a two-model, five-depth
audit comparison where exact values, denominators, and caveats are more useful
than visual interpolation.

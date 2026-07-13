# Facet-Local MiniLM B-First Pilot Design

## Objective

Determine whether a small cross-encoder can rescue useful documents buried in
the existing facet candidate pools by reranking every stream against its own
information need before cross-stream fusion. The experiment isolates facet
reranking first. A lexical Stage A is not tested unless MiniLM Stage B first
demonstrates useful facet-local ranking and coverage promotion.

The primary decision is not merely whether nDCG@10 rises. It is whether
facet-local MiniLM promotes topic-relevant documents that are absent from the
original narrative top 100 or current R1 top 100, while preserving acceptable
top-rank quality.

## Evidence status and interpretation

This is a descriptive four-topic pilot, not a blind generalization claim.
Topics `200`, `225`, `707`, and `897` have already influenced R1 development
and prior diagnostics. The rankings, review packet, model policy, and decision
rules must nevertheless freeze before projected qrels reopen.

Topic qrels identify overall topic relevance. They do not prove that a
document answers the particular facet that retrieved it. The experiment
therefore combines qrels-backed topic metrics with a small qrels-blind,
system-masked facet-relevance review.

## Immutable inputs and exact accounting

Reuse only the verified retrieval artifacts that produced frozen R1. Do not
make a new search request.

- Four original streams: one for each pilot topic.
- Twenty-two repaired R1 facet streams from
  `reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json`.
- Five retained first-pass facets already included in R1:
  - `225/prompt_lab_v1:facet:f06`
  - `225/prompt_lab_v1:facet:f07`
  - `707/prompt_lab_v1:facet:f01`
  - `707/prompt_lab_v1:facet:f03`
  - `897/prompt_lab_v1:facet:f01`
- Exactly 31 streams and 3,100 depth-100 stream rows before deduplication.

Per-topic stream counts are frozen as follows:

| Topic | Original | Facets | Total streams | Candidate rows |
|---|---:|---:|---:|---:|
| `200` | 1 | 9 | 10 | 1,000 |
| `225` | 1 | 7 | 8 | 800 |
| `707` | 1 | 3 | 4 | 400 |
| `897` | 1 | 8 | 9 | 900 |

Bind the new experiment manifest to the exact prior R1 freeze, source
manifests, ledgers, candidate texts, query strings, ranks, and document IDs.
Materialize the authenticated leaf snapshot once at
`outputs/rag25_facet_local_minilm_v1/source_v1/candidates.jsonl`, with its
create-only receipt beside it. The durable manifest records the snapshot hash,
row count, source bindings, and schema. After source freezing, every downstream
stage reads only this snapshot and its manifest; it must not reopen a live
retrieval ledger or cache.

Hard-reject protected topics `144`, `213`, `224`, `407`, and `515` before any
source join, tokenization, cache lookup, inference, fusion, review, evaluation,
or report construction.

## Model and query policy

Use the actual cross-encoder, not the similarly named embedding model:

- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- backend: Hugging Face Transformers sequence classification
- score: the single raw relevance logit
- maximum pair length: 512 tokens
- inference dtype: float32
- execution mode: evaluation, gradients disabled

The official model repository contains several alternative formats. A separate
gated materialization step may download only `config.json`,
`model.safetensors`, `special_tokens_map.json`, `tokenizer.json`,
`tokenizer_config.json`, and `vocab.txt` at the pinned revision. It requires an
approval receipt bound to the model, revision, and allowlist, then writes a
materialization receipt containing the resolved snapshot path, exact files,
bytes, and SHA-256 hashes before the tokenizer preflight. Reject pickle weights
and unexpected files. The 90.9 MB safetensors weight file and full resolved
local snapshot size must be recorded before inference.

Original candidates use the exact full narrative as their reranking query.
Repaired facets use the exact context-tethered R1 query. Retained facets use
their exact verified first-pass query text. Query serialization is the UTF-8
query string without a prompt wrapper. Every query hash is frozen.

## Tokenization and bounded document coverage

Tokenization is query-specific and deterministic:

- maximum query tokens: 192; fail rather than silently truncate;
- pair maximum: 512 including model special tokens;
- passage budget: the remaining pair tokens after the full query and special
  tokens, with a required minimum of 256 passage tokens;
- passage overlap: 64 document tokens;
- maximum windows per query-document pair: 32;
- if a document produces `N > 32` windows, retain indices
  `round_half_up(j * (N - 1) / 31)` for `j = 0..31`; assert 32 unique,
  increasing indices, with first index `0` and last index `N - 1`;
- maximum total uncached inference pairs: 100,000.

The tokenizer-only preflight materializes every query/window hash and reports
exact document, window, cache-hit, and cache-miss counts by topic and stream.
It also reports capped-document counts and selected-window document-token
coverage fractions by stream (minimum, median, and p95), so sampling loss is
visible rather than silently blamed on the model.
It reads no qrels and performs no model forward pass.

The benchmark sample is also frozen by the preflight. For at least 96 cache
misses, select the 32 longest pairs for one warm-up pass, then select 64 pairs
from the remainder after sorting by `(pair_token_count, cache_key)`. If `M` is
that remainder's size, use indices `round_half_up(j * (M - 1) / 63)` for
`j = 0..63`. Time that 64-pair
sample three times at batch size 32 and use median throughput; total benchmark
forward pairs are 224. For 1--95 misses, warm up the longest
`min(16, floor(M/4))` pairs and time every remaining miss once. Record peak
memory across warm-up and timed passes. Project full-run time as exact misses
divided by measured throughput, multiplied by `1.25` in the primary case or
`1.50` in the small-sample case. Zero misses require no benchmark or inference.

## Passage aggregation

Primary document score: top-four weighted span-distinct support.

1. Sort windows by descending raw logit, then start token and window ID.
2. Accept the best window.
3. Accept later windows only when they contribute at least 128 previously
   uncovered document tokens.
4. Retain at most four accepted windows.
5. Apply weights `0.55`, `0.25`, `0.13`, and `0.07`, renormalized over the
   number of accepted windows.

Sensitivity only: MaxP, the maximum window logit. Do not tune either
aggregation on these four topics. Documents tie-break by prior stream rank and
then document ID.

## Experiment matrix

### Frozen fusion-correction amendment

Pre-ranking reconstruction found that the historical R1 implementation built
topic-specific facet weights in a global mapping keyed only by
`(variant_name, retriever_name)`. Repeated facet names therefore inherited
weights from later topics. Preserve that exact artifact for continuity, but do
not treat it as the intended topic-local 0.5/0.5 family-balanced control.

The primary matrix uses `family_rrf_topic_local_v2`, whose stream identity is
`(topic_id, variant_name, retriever_name)`. For every topic, the original
stream has weight `0.5`, the facet family totals `0.5`, and every active facet
has weight `0.5 / topic_facet_count`. The weights must sum to exactly `1.0`
per topic and be invariant to input ordering.

Historical R1 metrics remain valid descriptions of `R1_LEGACY`, but they are
not evidence about the intended topic-local family-balanced RRF. Any corrected
qrels-backed result is reported as a retrospective fusion correction rather
than silently replacing the historical artifact. Audit the historical uniform
RRF sensitivity separately because it used the same topic-free key pattern;
it is not a promotion arm in this experiment.

The primary family reranks facet streams while leaving the original stream in
its prior BM25 order. This isolates facet-local value despite the original
family's 0.5 fusion weight.

| Arm | Original stream | Facet streams | Role |
|---|---|---|---|
| `R1_LEGACY` | BM25 | BM25 | Byte-identical historical R1; legacy global-key fusion |
| `C0_TOPIC_LOCAL` | BM25 | BM25 | Primary corrected control |
| `BF100_TOPIC_LOCAL` | BM25 | MiniLM top-four aggregation, retain 100 | Primary B test |
| `BF50_TOPIC_LOCAL` | BM25 | MiniLM top-four aggregation, retain 50 | Retention sensitivity |
| `BF20_TOPIC_LOCAL` | BM25 | MiniLM top-four aggregation, retain 20 | Retention sensitivity |
| `BO100_TOPIC_LOCAL` | MiniLM | BM25 | Original-only diagnostic |
| `BB100_TOPIC_LOCAL` | MiniLM | MiniLM | Both-families diagnostic |
| `BF100_MAXP_TOPIC_LOCAL` | BM25 | MiniLM MaxP, retain 100 | Aggregation sensitivity |
| `BF100_LEGACY_FUSION` | BM25 | MiniLM top-four aggregation, retain 100 | Secondary 2x2 decomposition diagnostic |

All B arms reuse one content-addressed score cache. The depth and aggregation
sensitivities are derived offline from the same scores. Do not run inference
separately for an arm.

`R1_LEGACY` is the exact prior frozen `R1__family_rrf.jsonl`, copied byte for
byte. Independently reconstruct its canonical `(topic_id, rank, document_id)`
rows with the frozen `family_rrf_global_key_v1` weight table and require the
hash to match the prior R1 canonical hash. Independently reconstruct
`C0_TOPIC_LOCAL` from the same 31 BM25 streams with the corrected v2 weights.
Save a qrels-free per-topic rank/set diff between these controls. The primary
causal comparison is `BF100_TOPIC_LOCAL - C0_TOPIC_LOCAL`. Report
`C0_TOPIC_LOCAL - R1_LEGACY` as the fusion-correction effect and
`BF100_TOPIC_LOCAL - R1_LEGACY` as the total operational change, not as a pure
MiniLM effect. `BF100_LEGACY_FUSION - R1_LEGACY` is a secondary diagnostic.

Fuse every arm with the existing family-balanced weighted RRF:

- `k = 60`;
- output depth `100`;
- original-family weight `0.5`;
- facet-family weight `0.5`, divided equally across that topic's active facet
  streams;
- deduplicate by document ID;
- preserve every stream rank, query, score, passage, and fusion contribution;
- never compare raw MiniLM scores across different queries.

The unequal facet counts by topic are part of the frozen design and must be
reported rather than normalized away.

## Blinded facet-local review

Before qrels access, pool the top two documents per facet from the BM25 facet
order used by `C0_TOPIC_LOCAL` and the MiniLM facet order used by
`BF50_TOPIC_LOCAL`, deduplicated within facet. This yields at most 108
facet/document items.
For every pooled facet/document pair, present the same frozen highest-scoring
MiniLM window regardless of whether the corrected control, BF50, or both
contributed the item.
Reviewers judge whether that displayed passage answers the facet; they are not
asked to infer relevance from unseen document text. Mask arm, rank, score, and
document ID. Deterministically shuffle using the experiment-manifest hash. The
secret unmasking map records every `(arm, facet, source rank, document ID,
passage-selection provenance)` membership. A shared item counts in both arm
denominators after labels freeze.

Two independent reviewers label each item:

- `direct_answer`
- `partial_or_related`
- `not_facet_relevant`
- `wrong_domain` flag
- `low_quality` flag

Disagreements on the three-way relevance label are adjudicated by a third
reviewer. Freeze label and adjudication hashes before unmasking systems. These
labels are pilot diagnostics, not official qrels.

## Freeze and qrels firewall

Before qrels access, save and hash:

- experiment manifest and protected-topic attestation;
- exact 31-stream/3,100-row candidate snapshot;
- model, tokenizer, runtime, device, dtype, and download provenance;
- tokenizer/window preflight and approval receipts;
- every raw MiniLM score and cache identity;
- every aggregated stream ranking;
- all nine fused system rankings;
- the legacy global-key weight table, corrected topic-qualified weight table,
  per-topic sum checks, and qrels-free legacy-versus-corrected diff audit;
- fusion definitions and tie-breaks;
- blinded review packet, labels, adjudication, and unmasking map;
- exact evaluated topic IDs.

The evaluator accepts only a self-contained verified freeze. It cannot open a
qrels path while any required hash, score, ranking, or blinded-review artifact
is missing.

The all-topic qrels file is not a supported evaluator input. Evaluation
requires a separately authorized, hash-pinned qrels projection whose sidecar
manifest declares exactly topics `200`, `225`, `707`, and `897`. Verify that
sidecar before opening the projected judgments. An atomic create-only access
receipt binds the projection and sidecar hashes, allowed topics, ranking-freeze
hash, and review-freeze hash; refuse repeat or mismatched access. This plan does
not create the projection from the all-topic qrels. If it is unavailable after
the freezes complete, stop and request explicit direction.

## Measurements

### Candidate-generation headroom

For facet depths `K = 20, 50, 100`, report paired pre-fusion curves:

- `O@100 + C0_TOPIC_LOCAL facets@K`;
- `O@100 + BF_TOPIC_LOCAL facets@K`.

At `K=100` both contain the same raw candidates, so union recall must be
identical. MiniLM can improve retention at `K=20/50` and within-stream ranks
used by final fusion, but it cannot create candidates. For every curve, report:

- unique candidate documents;
- relevant and graded-relevant documents;
- Recall and graded Recall;
- relevant documents absent from original `O@100`;
- relevant documents absent from corrected `C0_TOPIC_LOCAL@100`;
- relevant documents absent from historical `R1_LEGACY@100`;
- relevant documents gained and lost relative to those baselines;
- net relevant-document change;
- per-topic union-at-K curves.

### Facet-local filtering

Report by arm, topic, and facet:

- blinded direct-answer, partial, irrelevant, wrong-domain, and low-quality
  counts;
- topic-relevant documents retained at 100, 50, and 20;
- unique topic-relevant documents contributed beyond `O@100`,
  `C0_TOPIC_LOCAL@100`, and `R1_LEGACY@100`;
- representative promoted, demoted, gained, and lost passages.

### Final systems

Report aggregate and per-topic:

- Recall@100 and graded Recall@100 as primary metrics;
- novel relevant documents promoted versus O and R1;
- relevant gains, losses, and net change;
- nDCG@10 and precision@10 as guardrails;
- relevant documents at 10;
- judged rates at 10 and 100;
- oracle nDCG@10 from frozen top-50 and top-100 pools.

## Diagnostic outcomes

Do not reduce the result to an unexplained pass/fail. All aggregate metric
deltas are unweighted means over the four topics. Blinded rates are first
computed within each facet and then macro-averaged over 27 facets; shared
review items count in both arms. Equality is non-improvement except where a
rule explicitly says non-worse.

Define:

- `headroom`: count of qrels-relevant documents in the raw `O@100 +
  C0_TOPIC_LOCAL facets@100` union but absent from `C0_TOPIC_LOCAL@100`;
- `pre_fusion_promoted_novel`: count of relevant documents absent from
  `C0_TOPIC_LOCAL@100` whose best BF facet rank is at most 20 and whose best
  control facet rank is greater than 20;
- `final_novel`: relevant documents in `BF100_TOPIC_LOCAL@100` but absent from
  `C0_TOPIC_LOCAL@100`;
- `final_novel_vs_legacy_R1`: relevant documents in
  `BF100_TOPIC_LOCAL@100` but absent from `R1_LEGACY@100`.

Apply this ordered, mutually exclusive decision table:

1. `candidate_generation_gap` when `headroom == 0`. Reranking cannot create
   missing candidates; return to retrieval controls or one bounded feedback
   pass.
2. `B_promotes_coverage` when `final_novel >= 1`, `BF100_TOPIC_LOCAL` has
   positive net relevant-document change, positive macro Recall@100 delta
   versus `C0_TOPIC_LOCAL`, non-negative macro graded Recall@100 delta, no
   per-topic Recall@100 or graded Recall@100 delta below `-0.02`, positive
   macro direct-answer-rate delta, non-positive wrong-domain-rate delta, macro
   nDCG@10 delta at least `-0.02`, and no topic nDCG@10 delta below `-0.10`.
   The operational comparison versus `R1_LEGACY` must also have non-negative
   macro Recall@100 and graded Recall@100 deltas, no per-topic Recall@100 or
   graded Recall@100 delta below `-0.02`, macro nDCG@10 delta at least `-0.02`,
   and no topic nDCG@10 delta below `-0.10`. Only this outcome permits a
   separately designed Stage A efficiency experiment.
3. `B_coverage_gain_with_regression` when `final_novel >= 1` or macro
   Recall@100 improves, but any promotion guard above fails. Preserve the gain,
   diagnose the exact loss, and do not begin Stage A.
4. `B_filters_but_fusion_blocks` when `headroom > 0`, macro direct-answer rate
   improves without a wrong-domain-rate increase,
   `pre_fusion_promoted_novel >= 1`, and macro final Recall@100 versus
   `C0_TOPIC_LOCAL` does not improve.
   The three saved sets prove that candidates existed, MiniLM gave at least one
   novel relevant document a fusion-eligible local rank, and final RRF omitted
   the coverage gain. Diagnose fusion weights, truncation, and quotas offline.
5. `ranking_only_gain` when macro nDCG@10 improves but `final_novel == 0` and
   macro Recall@100 does not improve. Record the ordering gain only.
6. `B_query_window_or_model_gap` otherwise. Inspect query fit, capped-window
   coverage, aggregation, and model errors, then require advisor review before
   another run.

## Cost and authorization gates

1. Model-materialization gate: the user approved the actual recommended MiniLM
   model; require the approval receipt, download only pinned allowlisted files,
   and record exact files, bytes, and hashes without constructing the model.
2. Tokenizer-only preflight: verify the materialization receipt; no model
   inference, qrels, or network retrieval; publish exact query-window,
   coverage, and cache-miss counts.
3. Benchmark gate: obtain explicit approval for a maximum 256-pair ROCm
   benchmark; record throughput and peak device/host memory.
4. Full-inference gate: calculate runtime from benchmark throughput and exact
   misses, then obtain explicit approval before scoring.
5. Qrels gate: after every ranking and blinded-review artifact is frozen,
   accept only a separately authorized, hash-pinned, exact four-topic qrels
   projection and write the one-time access receipt. Never open and filter the
   all-topic qrels.

No paid API, hosted inference, new retrieval, recursive repair, or dense primary
retrieval is part of this experiment.

## Deliverables

- Versioned manifest and source-binding receipt.
- Tokenizer/window preflight with exact cost and cache coverage.
- Benchmark and full-inference approval receipts.
- Content-addressed MiniLM score cache and immutable run ledger.
- Frozen `R1_LEGACY`, `C0_TOPIC_LOCAL`, all six corrected topic-local B
  rankings, and `BF100_LEGACY_FUSION`.
- Blinded facet-local review packet and adjudicated labels.
- Qrels-backed candidate-headroom, gain/loss, recall, and guardrail report.
- Rendered local HTML that distinguishes retrieval absence, facet-filtering
  errors, and fusion loss and explains whether Stage A is warranted.

## Verification requirements

Tests must enforce exact topic, stream, candidate, model, window, cache,
ranking, review, and qrels-firewall contracts. Protected topics must fail at
every boundary. Raw MiniLM scores must never cross query boundaries. Every
arm must be reproducible from saved candidates and scores under input
reordering. The report must reproduce every metric and diagnostic outcome from
the frozen artifacts.

# Cache-First Candidate-Core Retrieval Runs — Design

**Date:** 2026-08-09

**Status:** approved in conversation; awaiting review of this written specification

**Source artifact:** `/home/npatta01/data/competitions/trec_rag_2026/outputs/facet-deepseek-b40-v3`

**Shared score cache:** `/home/npatta01/data/competitions/trec_rag_2026/cache/reranker`

## Decision summary

Create three variable-depth Retrieval submissions without rerunning retrieval.
Use authenticated scores from the finalized retrieval artifact to select a
small, evidence-backed candidate core for each topic. Complete the narrative
and subnarrative score matrix only within that core. Check the shared score
cache before every model call and infer only genuine misses.

Run inference on one fast remote GPU after an exact declined dstack preview.
Remote spending has a hard limit of USD 10. Write every new score to the
worker's isolated cache. Only after complete remote and local validation,
transactionally merge those scores into the shared local reranker cache and
replay all topics cache-only.

This specification supersedes the complete-union execution design in
`2026-08-09-three-retrieval-baseline-runs-design.md`. The complete-union design
remains documented as a more expensive alternative.

## Verified source identity

The input is the latest finalized retrieval artifact:

- run ID: `facet-deepseek-b40-v3`;
- 119 selected topics, `rag2026-0` through `rag2026-118`;
- all 119 topic receipts are complete with stopping reason
  `coverage_sufficient`;
- official retrieval row count: 21,727;
- export manifest SHA-256:
  `cb5a81608c48a5c43692ada0c9121cd96b7160589f08f5bbaf4e20612001463a`;
- export code commit:
  `05fdf35d858bf52bec2843c699a4bfd85d4b8c61`, equal to the current
  `origin/master` at design time.

The source artifact and shared cache are read through their absolute paths from
the isolated implementation worktree. The source artifact is never modified.

## Semantic units and source lanes

- The untouched narrative is one semantic unit.
- Each valid subnarrative text is one semantic unit.
- Each authenticated `facet:<subnarrative>:text` lane is the existing pooled
  retrieval source for that subnarrative.
- Planned BM25 query strings without their own authenticated result lanes are
  excluded.
- A topic with no valid subnarrative fails closed.

## Candidate-core gate

For every authenticated lane in `scoring/lane_scores.jsonl`, let `L_l(d)` be
the finite existing `aggregate_score` for document `d` in lane `l`. Compute the
median and median absolute deviation over the documents present in that lane.

- if `MAD_l > 0`, lane `l` admits `d` when
  `L_l(d) >= median_l + 2.5 * 1.4826 * MAD_l`;
- if `MAD_l = 0`, lane `l` admits only documents with
  `L_l(d) > median_l`.

The candidate core `C_t` is the deduplicated union of documents admitted by the
original lane or any subnarrative lane. If the union is empty, admit exactly
the highest-scoring original-lane document, breaking ties by best retrieval
rank and then UTF-8 bytewise document ID.

`C_t` is the final eligible document set shared by all three runs, and
`k_t = |C_t|` is their common variable depth. Do not apply another eligibility
cutoff after targeted scoring. Do not pad or truncate a topic to a fixed depth.

The gate uses only scores already produced by retrieval. It does not impute a
score for a document absent from a lane. The complete retrieval union remains
the provenance universe, but only `C_t` proceeds to targeted cross-scoring.

## Observed size reduction

A read-only dry analysis of all 119 source topics found:

- 4,246 candidate-core documents total;
- median 31 documents per topic;
- minimum 1, maximum 121, and 90th percentile 67 documents;
- 30,579 document–semantic-unit pairs before passage chunking.

The two-topic check on `rag2026-1` and `rag2026-18` reduced the candidate pools
from 7,395 and 7,856 documents to 61 and 41 documents. Their targeted matrices
contained 4,652 unique query–passage pairs: 1,055 cache hits and 3,597 misses,
compared with 695,368 misses for complete-union scoring. Extrapolating the
observed passages per document gives roughly 166,000 all-topic query–passage
pairs; the exact hit and miss counts must be computed by the sealed preflight.

## Targeted score matrix

For every document in `C_t`, score every competition chunk against:

- the untouched narrative;
- every subnarrative text.

Use the pinned `mixedbread-ai/mxbai-rerank-base-v2` scorer and the competition
chunker (`max_characters: 3500`, `overlap_characters: 350`). Preserve the
existing model revision, bfloat16 inference, maximum length, input policy, and
raw-logit representation. Reject non-finite scores.

Before inference, export only matching existing query–passage pairs from the
shared local score cache into an isolated task cache. The task must account for
every expected cache key as exactly one hit or miss. It may call the model only
for misses. No scoring process writes directly to the shared cache.

For one document and semantic unit:

1. sort passages by raw score descending, then span offsets;
2. greedily suppress a passage whose overlap coefficient with a retained
   passage is at least `0.5`;
3. aggregate the best four retained passages with weights
   `0.55, 0.25, 0.13, 0.07`, renormalized when fewer than four survive.

Normalize the resulting document aggregates to tied percentiles separately
within each topic and semantic unit over `C_t`. For `N > 1`, use
`P = (L + (T - 1) / 2) / (N - 1)`; for `N = 1`, use `P = 1`.

## Run 1: narrative

Rank `C_t` by:

1. narrative percentile descending;
2. best retrieval rank ascending;
3. UTF-8 bytewise document ID ascending.

## Run 2: narrative plus subnarrative

For every document, let `best` and `second` be its highest two subnarrative
percentiles after targeted completion.

- with at least two subnarratives:
  `facet = 0.7 * best + 0.3 * second`;
- with one subnarrative: `facet = best`.

Set `combo = 0.5 * narrative + 0.5 * facet` and rank `C_t` by:

1. combo descending;
2. best retrieval rank ascending;
3. UTF-8 bytewise document ID ascending.

## Run 3: breadth and passage depth

For each document and subnarrative, overlap-suppress passages and retain at
most the best three. For each subnarrative separately, compute a robust raw
passage threshold over all retained passages from `C_t`, using the same
median/MAD rule as the candidate gate.

A retained passage is strong when it passes its subnarrative threshold. For
each document compute:

- `supported_subnarrative_count`: distinct subnarratives with at least one
  strong passage;
- `strong_passage_count`: total strong retained passages across those
  subnarratives.

Rank `C_t` by:

1. supported subnarrative count descending;
2. strong passage count descending;
3. run-2 combo descending;
4. best retrieval rank ascending;
5. UTF-8 bytewise document ID ascending.

This replaces an arbitrary global top-100 passage limit and directly rewards
documents with repeated strong passage support across subnarratives.

## Remote execution gate

The sealed preflight computes exact all-topic cache hits and misses without
model inference and prepares one authenticated candidate-core input. Preview
one on-demand H200 task with a two-hour maximum duration and a price ceiling
that keeps its worst-case cost at or below USD 10. If H200 has no compatible
offers, preview H100 under the same duration and aggregate cost ceiling. The
task launches only after the exact declined preview is shown and confirmed.

Within the one remote task, score `rag2026-1` and `rag2026-18` first as the
canary phase. Immediately verify their candidate cores, pair accounting,
matrices, cache export, and zero-model-batch cache-only replay. Abort and
publish failure evidence if either canary fails. If both pass, continue through
the remaining 117 topics on the same loaded model and isolated cache. This
avoids a second machine startup while retaining a real two-topic cloud gate.

## Local cache merge

The remote result uses this local merge protocol:

1. finalize an immutable publication containing matrices, portable raw-score
   cache data, source identities, pair accounting, and file hashes;
2. download the publication and verify every declared byte and hash locally;
3. import the portable scores into a fresh local replay cache;
4. rebuild every selected topic matrix cache-only and require byte-identical
   matrices and zero model batches;
5. acquire the shared-cache writer lock and transactionally import the verified
   portable scores into the shared local reranker cache;
6. reject any identity mismatch or conflicting value; never overwrite or
   delete an existing score;
7. record before/after row counts, inserted/existing counts, context identity,
   source hashes, and the portable-cache hash in a merge receipt;
8. replay all 119 topics cache-only from the merged shared cache and verify the
   three final runs.

The merge always occurs locally. Remote workers never write the shared local
cache directly.

## Output and verification contract

Write three standard six-column TREC run files covering all 119 topics. Within
each topic, ranks start at one and scores are strictly non-increasing. Use
`k_t - rank + 1` as the organizer-facing score. Run IDs are stable and distinct.

Each output manifest records the source export hash, candidate-core hashes,
scorer and chunker identities, matrix hashes, topic depths, cache accounting,
execution location, and local merge receipt. It contains no document text,
cache contents, or secrets.

Tests must cover candidate-gate ties, zero-MAD and empty-core fallback,
candidate-core immutability across runs, exact cache hit/miss accounting,
overlap suppression at `0.5`, aggregation weights, percentile ties,
one-subnarrative behavior, strong-passage breadth, deterministic ordering,
portable-cache conflict rejection, cache-only replay, all 119 topic IDs,
variable depths, and standard TREC formatting.

Before remote inference, the implementation receives an independent Sol
review, the sealed all-topic preflight and real wrapper preflight pass, and the
exact dstack offer is previewed and confirmed.

## Explicit tradeoff

This design can transfer evidence across semantic units for every document
that has strong existing retrieval evidence. It cannot rescue a document that
is weak in every authenticated source lane. That loss of exhaustive transfer
is accepted to reuse prior computation, meet the time budget, and avoid the
complete-union matrix's disproportionate cost.

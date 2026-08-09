# Three Retrieval Baseline Runs — Design

**Date:** 2026-08-09

**Status:** superseded for execution by `2026-08-09-cache-first-candidate-core-retrieval-design.md`; retained as the complete-union alternative
**Input:** `/home/npatta01/data/competitions/trec_rag_2026/outputs/facet-deepseek-b40-v3`

## Goal

Create three organizer-valid retrieval submissions from the complete union of
documents already retrieved for each 2026 topic. Do not repeat BM25 retrieval.
Score missing query–passage pairs with the pinned reranker, cache those raw
scores, and derive all run variants locally from one immutable score matrix.

The three runs share the same topic-specific eligible document set and variable
cutoff. They differ only in ordering:

1. narrative relevance;
2. narrative plus subnarrative relevance;
3. breadth of strong passage evidence across subnarrative queries.

## Inputs and units

- A topic's document pool is the deduplicated union in
  `scoring/selection.json:union_pool`.
- The original narrative is one semantic unit.
- Each valid subnarrative is one semantic unit for document relevance and
  breadth evidence, even when the plan contains multiple BM25 query variants.
- The existing authenticated `facet:<subnarrative>:text` lane is the pooled
  retrieval source. Planned query strings without retrieval-result lanes are
  excluded because using them would require new retrieval.
- A topic with zero valid subnarratives is invalid and must fail closed.
- A document's best retrieval rank is the minimum source rank over every
  occurrence of that document in the retrieval audit.

## Immutable score matrix

For every topic, score every document in the complete union against:

- the untouched narrative;
- every subnarrative text.

Use the pinned `mixedbread-ai/mxbai-rerank-base-v2` scorer and the competition
chunker (`chunk_max_characters: 3500`, `chunk_overlap_characters: 350`). Reject
NaN and infinite model scores. Reuse existing cache entries where scorer,
query, document text, and chunking identities match exactly. Persist raw finite
scores and provenance, not cutoff or ranking decisions, so new local ranking
variants never require another GPU run.

Remote workers produce per-topic score-matrix shards plus cache deltas. Local
collection verifies hashes and complete pair accounting before merging cache
deltas. Collection must be idempotent and must not mutate the source retrieval
output.

## Passage aggregation

Passage spans are half-open: `[start_char, end_char)`.

For scores associated with one document and one semantic unit or query:

1. sort passages by descending raw score, then start offset, end offset;
2. retain a passage unless its overlap coefficient with any already retained
   passage is at least `0.5`;
3. overlap coefficient is intersection length divided by the shorter span
   length;
4. aggregate the best four retained passage scores using weights
   `0.55, 0.25, 0.13, 0.07`, renormalized when fewer than four survive.

The resulting finite number is the document's raw relevance score for that
semantic unit or query.

## Percentile normalization

Normalize document scores separately within each topic and semantic unit.
For `N > 1`, a tied value receives:

`P = (L + (T - 1) / 2) / (N - 1)`

where `L` is the number of lower values and `T` is the tied group size. For
`N = 1`, set `P = 1`. Equal raw values must receive equal percentiles.

## Shared topic-specific cutoff

Let `A_u(d)` be the finite top-four weighted raw passage aggregate for document
`d` and semantic unit `u`. For the narrative and for each subnarrative
separately, compute the median and MAD over the complete document union.

- if `MAD_u > 0`, unit `u` admits `d` when
  `A_u(d) >= median_u + 2.5 * 1.4826 * MAD_u`;
- if `MAD_u = 0`, unit `u` admits only documents with
  `A_u(d) > median_u`;
- the topic eligible set is the union of documents admitted by the narrative
  or any subnarrative;
- if that union is empty, admit exactly the document with highest narrative raw
  aggregate, breaking ties by best retrieval rank and then UTF-8 bytewise
  document ID.

Percentiles are used only for run ordering; they never affect eligibility.

This yields one eligible set `E_t` and one variable depth `k_t = |E_t|` shared
by all three runs. Never pad or truncate a topic to 1,000 documents: 1,000 is a
per-query retrieval ceiling, not a fixed final run depth.

## Run 1: narrative

Rank `E_t` by:

1. narrative percentile descending;
2. best retrieval rank ascending;
3. bytewise document ID ascending.

## Run 2: narrative plus subnarrative

For each document, take the best and second-best subnarrative percentiles.

- with at least two subnarratives:
  `facet = 0.7 * best + 0.3 * second_best`;
- with one subnarrative: `facet = best`.

Then compute `combo = 0.5 * narrative + 0.5 * facet` and rank `E_t` by:

1. combo descending;
2. best retrieval rank ascending;
3. bytewise document ID ascending.

## Run 3: breadth of passage support

For every document/subnarrative pair, perform overlap suppression and retain at
most the top three passages. Then, for each subnarrative, select the globally
best 100 retained passage hits across the complete topic union, before applying
`E_t`.
Ties use raw score descending, best retrieval rank ascending, document ID
ascending, and span offsets ascending.

For each eligible document:

- `supported_subnarrative_count` is the number of distinct subnarratives
  represented by at least one selected global hit;
- `admitted_hit_count` is the total number of its selected global hits.

Rank `E_t` by:

1. supported query count descending;
2. admitted hit count descending;
3. run-2 combo score descending;
4. best retrieval rank ascending;
5. bytewise document ID ascending.

## Output contract

Write three TREC run files. For a topic containing `k_t` documents, ranks are
one through `k_t` and run-file scores are `k_t - rank + 1`. Every line must use
the configured team/run tag. Topics must be naturally ordered by numeric topic
suffix. Writes are atomic.

Each run directory also contains a manifest with source hashes, code identity,
scorer/chunker identity, matrix hashes, per-topic `k_t`, and aggregate cache
hit/miss accounting. It contains no document text or secrets.

## Verification gates

Golden tests cover percentile ties, exact `0.5` overlap, zero-MAD behavior,
empty-set fallback, one-subnarrative renormalization, top-three-before-global-
top-100 breadth semantics, finite-score rejection, deterministic ties, and the
shared eligible set across all runs.

Before the full run:

1. an independent Sol reviewer approves the implementation;
2. a cloud smoke run for `rag2026-0` and `rag2026-37` verifies topic isolation,
   pinned identities, complete pair accounting, hashes, common `k_t`/set, no
   padding, deterministic ordering, and cache-only replay;
3. the dstack preview and maximum spend are reviewed.

The 119-topic run starts only after those gates pass.

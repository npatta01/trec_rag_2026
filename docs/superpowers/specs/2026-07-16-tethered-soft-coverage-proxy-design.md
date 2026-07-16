# Tethered Soft-Coverage Retrieval Proxy Design

Date: 2026-07-16

## Decision

Evaluate whether narrative-tethered facet scores can improve the ordering of the
existing 8,114-document accepted union without hard facet quotas. Reuse the
repository's already frozen `DUAL` and `DUAL-NR` greedy objectives, replacing
only the naked-facet MiniLM score maps with the newer narrative-plus-facet
MiniLM score maps.

This is a retrieval-side proxy for future RAG answer quality. It does not run an
answer generator and does not claim nugget coverage, citation correctness, or
final answer quality.

## Scope

The diagnostic uses only topics `219`, `72`, `300`, and `84`. Topic IDs `144`,
`213`, `224`, `407`, and `515` remain forbidden at every input, ranking,
evaluation, and reporting boundary.

The experiment performs no retrieval, model inference, model download, paid
call, hosted call, or network request. It consumes existing authenticated
artifacts and writes a new post-qrels diagnostic identity. Existing v2.1,
ranking, score, evaluation, and report artifacts remain immutable.

## Why this experiment is necessary

The sealed `FIXED-O0` ranking is already a complete facet-aware continuation,
but it assigns coverage by a token-deficit rotation. Its first offline
evaluation showed that hard coverage damages ordering:

- narrative-only known-relevant recall is `0.1590` at 500 and `0.2407` at 1,000;
- `FIXED-O0` known-relevant recall is `0.1179` at 500 and `0.1672` at 1,000; and
- both reach the same `0.3106` only after exhausting the full union.

The failure therefore does not show that facets are useless. It shows that
equalized facet allocation can promote weak streams too aggressively.

## Inputs

Authenticate and bind:

- the accepted union and retrieval provenance from
  `outputs/rag25_deep_facet_candidates_v1/gate_v1/`;
- the frozen RRF inputs and prior full-union rank features from
  `outputs/rag25_deep_facet_candidates_v1/freeze_v1/` and
  `outputs/rag25_deep_facet_candidates_v1/phase2_v1/`;
- the complete narrative-tethered MiniLM document scores from
  `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/scoring/`;
- the sealed `NARRATIVE` and `FIXED-O0` complete permutations from
  `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings/`;
  and
- the already opened projected development qrels from
  `outputs/rag25_deep_facet_candidates_v1/evaluation_v1/qrels_projection.jsonl`.

All source hashes and row counts are recorded before evaluation output is
written. Because the pilot topics and qrels were previously inspected, every
result is explicitly post-qrels diagnostic evidence and cannot promote a
production method.

## Ranking arms

### Controls

- `RRF`: existing family-balanced sparse-retrieval fusion.
- `NARRATIVE`: existing full-union order by best full-narrative MiniLM passage.
- `FIXED-O0`: existing token-deficit facet coverage continuation.

### Primary proxy

- `TETHERED-DUAL`: reuse the exact existing DUAL objective:

  `0.35*G + 0.15*N + 0.15*R + 0.25*L + 0.10*B - 0.15*D`

  where `G` is the common-query percentile, `N` is the full-narrative
  percentile, `R` is the family-balanced RRF percentile, `L` is the strongest
  narrative-tethered facet percentile, `B` is diminishing uncovered-facet
  gain, and `D` is the existing lexical redundancy penalty.

### Sensitivity control

- `TETHERED-DUAL-NR`: the same formula with `D=0`. This identifies whether the
  existing Jaccard redundancy penalty removes useful evidence.

### Protected-head sensitivity

- `RRF100-TETHERED-DUAL`: preserve the existing RRF ranks 1--100, then append
  the complete `TETHERED-DUAL` permutation while skipping duplicates. This is
  not a 100-document output limit: every union document remains present. The arm
  tests whether soft facet coverage is useful below the strong sparse-retrieval
  head without repeating the earlier hard 200-document facet basket.

No coefficient is tuned on these four topics. Raw cross-encoder or BM25 scores
never cross query boundaries; only within-query average-rank percentiles enter
the objective. Every arm is a complete permutation of the same union.

## Evaluation

Report for every arm:

- binary known-relevant Recall at 100, 250, 500, 1,000, 1,500, and full depth;
- graded Recall at the same depths;
- nDCG at 10, 100, 500, 1,000, and 1,500;
- relevant-document counts and per-topic deltas;
- normalized area under the binary recall-depth curve through full depth;
- facet-only relevant-document retention at each depth; and
- qrels-positive facet exposure as a weak coverage proxy.

`Qrels-positive facet exposure` means that a selected document is globally
relevant to the topic and the ranking attributed its marginal coverage gain to
that facet. It does **not** prove that the document supports that facet. The
rendered report must keep this limitation adjacent to the metric.

The ranking calculation is qrels-blind. Evaluation joins qrels only after all
four complete ranking hashes exist.

## Interpretation

The proxy succeeds directionally when `TETHERED-DUAL` improves the
recall-depth curve or early relevant-document retention over both `NARRATIVE`
and `FIXED-O0` without a material nDCG collapse. This diagnostic cannot promote
the method. It decides only whether soft facet-aware evidence ordering is worth
carrying into the separate end-to-end RAG worktree.

If neither soft arm nor the protected-head sensitivity improves the useful
recall curve over its controls, retain the strongest existing ordering and treat
candidate-generation/query-repair as the bottleneck. Reranking cannot raise the
full-union `0.3106` recall ceiling.

## Deliverables

- reusable tethered soft-ranking code and focused tests;
- create-only ranking and evaluation artifacts under
  `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/`;
- a canonical v3 report under
  `reports/experiments/tethered_facet_minilm_diagnostic_v3/`; and
- a sanitized rendered copy in the existing tailnet-only artifact portal.

The v3 report explains the distinction among candidate recall, candidate
ordering, the retrieval-side coverage proxy, and final answer-generation
quality. Full answer generation remains out of scope.

## Verification

Tests must prove protected-topic rejection before source access, authenticated
input binding, complete-permutation preservation, deterministic output under
input reordering, exact reuse of the frozen DUAL parameters, qrels isolation
until ranking hashes exist, correct metric denominators, explicit proxy caveats,
and zero network/retrieval/model-inference counters.

The final HTML is checked for source traceability, sensitive paths or secrets,
desktop/mobile layout, and exact parity between canonical and served bytes.

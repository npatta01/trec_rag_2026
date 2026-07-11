# `det_sparse_v2` preregistration and offline admission

Date: 2026-07-11

Status: sealed offline preflight is mechanically valid but received a
post-preflight **scientific NO-GO**. V2 is retired without retrieval. No qrels,
external retrieval, model, reranker, agent, or paid API call was made.

## Post-preflight archive review

The canonical preflight was created from clean commit
`b4d8dceaf616af98e0957a8ca55ea2a91ea76f83` and tree
`ad420c45b4a79d1e02bf7f74cced6ac6cf0e423a`. Its freeze SHA-256 is
`18f9e4363c5ff0e00fea5a950636e37e82dbbf5d4db668e5ddfe7bc099369530`.
All 13 candidates were mechanically eligible. The frozen rule selected topics
`84`, `37`, `300`, and `161`, one from each structural stratum. Their four
plans contain no fallback or alias, 15 base queries, and a derived maximum of
26 requests. External, model, and reranker calls are zero; `qrels_opened` is
false.

The required qrels-blind query-shape review found two fatal facets:

- Topic `300` facet `f02` renders “I'm interested in learning about effective
  strategies I’d also like to know what global measures can be taken”. It
  contains neither global warming nor climate change, so “global measures” has
  no bounded subject.
- Topic `161` facet `f02` renders “I want to understand and why people hold
  such different views on it.” It contains no abortion-related term, leaving
  “it” unresolved; the analyzer removes even that pronoun from the sparse
  signature.

The remaining facets are interpretable, narrower sparse requests. These two
failures are nevertheless decisive under the frozen all-plans gate. They expose
a new scientific weakness: the strict-prefix context can preserve generic
conversational boilerplate instead of a topic-bearing referent. BM25 signature
distinctness, exact-span provenance, and coverage completeness cannot detect
that semantic loss.

V2 must not run retrieval or expansion and cannot be repaired or resampled.
Topics `84`, `37`, `300`, and `161` are burned. Any continuation requires a new
version, the remaining fresh topics, and an admission rule that makes every
subject-dependent child retain an explicit source-backed topic anchor.

## Why v2 exists

The sealed `det_sparse_v1` preflight was mechanically valid but failed its
post-preflight scientific review. For three two-unit narratives, copying the
complete first unit in front of the child reconstructed the exact original
query, leaving no distinct child retrieval path. V1 was archived without any
retrieval and is not repaired in place.

V2 is a fresh version with new planner, renderer, selection, experiment,
output, plan, and freeze identities. It uses fresh topics and never imports a
v1 plan, output, response, cache entry, or ticket.

## Frozen question

On a qrels-blind, structurally selected four-topic diagnostic, do distinct
exact-source facets (`F`), conservative corpus-derived keyword expansion (`E`),
or both (`FE`) improve over the permanent exact original sparse query (`O`)?

The comparison deliberately tests decomposition and expansion separately and
together. It cannot establish general effectiveness from four topics, and it
cannot prove that an agent is or is not needed. An agent is a later exception
strategy only if the cheaper arms show a specific unresolved vocabulary or
coverage failure.

## Frozen offline protocol

- Candidate universe: `14, 31, 37, 58, 72, 84, 161, 219, 233, 273, 300,
  477, 499`.
- Exclusions: known-five planner topics `144, 213, 224, 407, 515` and burned v1
  topics `200, 225, 707, 897`.
- Screening reads only the 13 exact candidate narratives and the pinned local
  Lucene analyzer. It does not read qrels, baseline scores, prior metrics,
  retrieval results, model output, or reranker output.
- Eligible topics are divided into two lower/upper two-unit strata, one
  three-unit stratum, and one four-or-more-unit stratum. A frozen SHA-256 rule
  selects one topic per stratum. An empty stratum is a no-go with no manual
  replacement.
- `f01` is the exact first unit and cannot merge. Each child is a bounded strict
  exact prefix of the first unit, one ASCII space, and its exact contiguous
  child coverage span.
- Every successful facet is exact-text and BM25-signature distinct from the
  original, pairwise signature-distinct, strictly shorter in analyzer-token
  occurrences, and a strict token multiset of the original. Every source unit
  has exactly one non-original facet path.
- There are two through four facets. The canonical base-request count is
  exactly `1 + M`; a future expanded run is bounded by `1 + 2M <= 9` per topic
  and 36 total.
- The canonical preflight is create-only, binds a clean Git commit and runtime,
  reserves the attempt before screening, records zero external/model/reranker
  calls and `qrels_opened=false`, and freezes all artifacts before any future
  retrieval decision. Its formal API constructs the loopback analyzer and
  provenance internally, then performs a fresh semantic replay before success.
- Any future executor is frozen to exactly 100 result rows, one attempt, no
  retry, no redirect, a fresh-run namespace of
  `det_sparse_v2_fresh_run_local`, and the distinct global ticket namespace
  `rag25_det_sparse_structural4_v2`.

## Admission checkpoints

1. Unit, integration, provenance, and semantic replay tests pass.
2. A clean-tree canonical preflight selects exactly four valid topics, with no
   fallback, aliases, or missing coverage path.
3. An advisor inspects the exact qrels-blind query shapes. Every child must be
   interpretable as a search request, narrower than the original, and protected
   from the v1 collapse.
4. Only after those gates may a separate proposal justify retrieval. It still
   requires an immutable hosted-index identity and explicit user authorization.

Failure at checkpoints 1–3 archives v2 without retrieval. Selected topic
shapes cannot be repaired or resampled under the same version.

## Cost and model boundary

Offline admission uses no generative model. This is intentional: larger local
or hosted query-planning models are not justified unless a controlled smaller
method fails and model size shows incremental value. The external gate is hard
closed while the hosted collection revision is
`hosted_climbmix_unknown_revision`.

The full frozen contract and dark-mode-safe workflow are in
`docs/superpowers/det_sparse_v2_design.md`; the executable contract is
`configs/det_sparse_v2.yaml`.

## Verification before canonical preflight

- Focused v2 suite: 68 passed.
- Repository suite: 370 passed, 9 skipped.
- Live pinned loopback Lucene integration: 11 passed.
- Python compilation: passed.
- `git diff --check`: passed.
- Independent implementation reviewer: GO for commit and exactly one
  zero-cost canonical preflight; all P1/P2/P3 findings closed.
- IR advisor: GO for the same offline-only milestone and NO-GO for retrieval,
  reranking, model inference, or an agent.
- External retrieval/model/reranker calls: zero.
- Qrels opened: no.

The canonical identities and post-preflight review are recorded above. Both
independent reviewers issued the final scientific NO-GO after inspecting only
the frozen qrels-blind query shapes.

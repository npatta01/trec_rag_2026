# `det_sparse_v1` preregistration and offline validation

Date: 2026-07-11

Status: sealed offline preflight is mechanically valid but received a
post-preflight **scientific NO-GO**. The arm is retired without retrieval.
No qrels, retrieval, model, reranker, or paid API call was made while producing
or reviewing this record.

## Post-preflight archive review

The canonical preflight was created from clean commit
`dcfcb2dd8ed11f519f53a791bf27690ac158198a` and tree
`fe230fb771851f44fc4cf23f87dc92cd2d098795`. Its freeze SHA-256 is
`22a2008b630833268b5c1463504d5186b4c9c5292eaf3d6480eabaf7ee74bf29`.
All four plan files are hash-valid and mechanically `ok`, with no fallback,
11 base unique requests, a derived maximum of 21 requests, and zero external,
model, or reranker calls. The qrels-opened flag is false.

Exact query-shape review then exposed a scientific admission failure:

- Topic `225` facet `f02` is byte-identical to the original query.
- Topic `707` facet `f02` is byte-identical to the original query.
- Topic `897` facet `f02` is byte-identical to the original query.

For those two-unit narratives, rendering the child as `P + u002` reconstructs
the original. Canonical alias dedup works correctly, but the only genuinely new
facet stream is the first-unit prefix; there is no child-specific PRF stream.
Only topic `200` exercises substantive multi-facet retrieval and expansion, so
this four-topic arm cannot answer the intended decomposition-plus-expansion
question.

Topic `200` also clarifies an ambiguity in the frozen prose. The parent-only
`P` facet is a pre-merge seed, not a final invariant: the unrestricted
adjacent-min cap merge absorbed `u002` through `u004` into final `f01`. That is
consistent with the executable merge rule and audit but inconsistent with a
literal reading that final `f01` always equals only `P`. Because `f01` then
contains substantive units, the prose rationale for excluding it from PRF is
also incomplete; exclusion is the frozen policy/cost choice, not proof that it
is parent-only.

The archive is preserved as evidence that exact-span/mechanical validity is not
sufficient for useful decomposition. It must not be used for retrieval, and it
must not be repaired in place. Any continuation requires a new version, fresh
topics, an analyzer-distinct-from-original facet admission rule, a coverage-unit
non-original-path invariant, and an unambiguous parent-merge policy.

## Question and interpretation boundary

This diagnostic asks whether one frozen deterministic facet policy (`F`), one
conservative corpus-derived PRF policy (`E`), or their operational combination
(`FE`) improves over the exact original sparse query (`O`). It uses exactly
topics `200`, `225`, `707`, and `897`, selected as difficult development topics
outside the five locked planner topics. Because that selection used prior qrel
performance, results can diagnose these four cases but cannot establish that
decomposition or expansion works generally, or that an agent is or is not
needed.

The full frozen contract and dark-mode-safe workflow diagram are in
`docs/superpowers/det_sparse_v1_design.md`. The executable configuration is
`configs/det_sparse_v1.yaml`.

## Frozen experiment

- Original narrative is permanent in every arm.
- At most four exact-source parent-context facets are created without a model.
- PRF uses ranks 1–5 versus 6–50 and emits at most two audited lowercase
  single-word terms for the original and at most three non-parent facets.
- Arms are `O`, `F`, `E`, and `FE`; fusion is weighted RRF with `k=60`.
- Exact request ceilings are derived from the plans and are no more than nine
  per topic or 36 overall. Model and reranker call ceilings are zero.
- Query construction and the final semantic replay are qrels-blind. Evaluation
  hashes and parses one qrels byte snapshot only after the final freeze passes.
- Ungraded diagnostics report exact top-100 novelty/Jaccard by logical stream
  and arm plus verified calls, cache hits, and elapsed seconds. Graded
  diagnostics additionally report new relevant documents versus `O`; none of
  these descriptive values changes a promotion gate.

## Safety boundary

The canonical executor reloads the only accepted config path, fixes the output
and HTTPS endpoint, binds imported source files to a clean Git tree, verifies
the loopback analyzer, uses a no-retry/no-redirect transport, reserves both a
run-local raw-first ledger and a Git-common experiment ticket, and reconstructs
PRF, arm membership, and rankings from frozen evidence before qrels access.

The hosted collection currently identifies itself only as
`hosted_climbmix_unknown_revision`. Therefore the external authorization gate
raises before analyzer access, plan replay, ticket creation, output-directory
creation, or transport entry. The production transport independently refuses
network entry. Opening retrieval requires a separately versioned protocol,
immutable hosted-index identity, fresh advisor review, and explicit user
authorization.

## Verification before freeze

- Repository suite: 302 passed, 9 skipped.
- Live pinned localhost Lucene sidecar: 11 passed.
- Python compilation: passed.
- `git diff --check`: passed.
- Independent IR advisor: GO to commit and run only the offline preflight;
  post-preflight semantic review changed the retrieval verdict to NO-GO.
- External retrieval: NO-GO by design.

The canonical offline preflight is run only after this source is committed so
its manifest can bind a clean commit and tree. Passing it is not retrieval
authorization.

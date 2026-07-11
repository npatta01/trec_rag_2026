# Deterministic sparse v4 advisor review packet

Date: 2026-07-11

Status: offline review packet. This document does not authorize model
inference, topic access, relevance-judgment access, retrieval, reranking,
downloads, hosted APIs, paid calls, or an agent loop.

Baseline implementation reviewed by this packet:
`0e267af5af6622f2119ede48593fc1d636390e08`

Primary references:

- `docs/superpowers/det_sparse_v4_synthetic_design.md`
- `docs/superpowers/det_sparse_v4_contract_artifacts/`
- `code/trec_rag/det_sparse_v4_contract.py`
- `code/trec_rag/det_sparse_v4_preflight.py`
- `code/tests/test_det_sparse_v4_contract.py`
- `code/tests/test_det_sparse_v4_preflight.py`

## Review decision requested

Please review whether v4 is the right next strategy after the sealed v3 no-go:
use a deterministic sparse pipeline, but replace recurrent lexical anchor
selection with a single local structured-output exact-span selector on a
synthetic-only qualification set.

The requested decision is not “may we run on real topics?” The requested
decision is only:

1. Is the v4 synthetic contract a sound enough offline gate to proceed to live
   local runtime attestation?
2. Are the synthetic cases and abstention categories adequate to test the
   vocabulary-gap failure mode without touching consumed development topics?
3. Are the stop conditions strict enough that a failed first case or failed
   prefix cannot leak into semantic scoring?
4. Is the small cached local model contract acceptable, or should we revise the
   prompt/schema before any model dispatch?

If the answer is yes, the next authorized work should be live local analyzer
and model attestation only. It should still not include retrieval, reranking,
real topic access, relevance judgments, paid calls, downloads, or a larger
model.

## Why v4 exists

V3 is archived as a scientific no-go because exact lexical recurrence produced
zero admissible anchors. That failure does not prove that anchor+facet rendering
is the wrong retrieval tactic; it proves that the v3 anchor admission condition
was too lexically brittle for vocabulary gaps.

The proposed v4 keeps the conservative sparse-retrieval shape:

- exact first-unit source anchoring;
- exact child facet rendering;
- original-only fallback on any failed gate;
- qrels-blind freezing;
- no dense retrieval requirement;
- no free-running agent.

It changes only the anchor-discovery step. A local structured-output model is
asked to choose or abstain on one exact U1 token span. Python then validates the
range, analyzer evidence, renderer constraints, and ledger mechanics
deterministically.

## Issue-to-evidence matrix

| Identified issue | v4 offline evidence | Current status |
| --- | --- | --- |
| V2.1 must remain immutable | V4 uses new experiment ID `det_sparse_exact_span_synthetic_v4`; no v2.1 files are required by the preflight module. | Preserved in this slice. |
| Consumed development topics must stay closed | `DENIED_TOPIC_IDS` freezes the 22 consumed IDs; preflight reports `denied_topic_count=22` and zero topic-file opens. | Enforced by tests and preflight. |
| Known-five topics must not be touched | Known-five IDs are part of the denied set: `144, 213, 224, 407, 515`. | Enforced by tests and preflight. |
| Query decomposition/facet strategy needs vocabulary-gap handling | V4 tests a semantic exact-span selector only on synthetic cases, then renders deterministic sparse facets. | Proposed; no model call yet. |
| Dense retrieval is not a practical main option at collection scale | V4 remains sparse/query-expansion oriented and does not require dense indexing or dense ranking. | Strategy-level choice, not yet empirically validated. |
| Free-running agents could drift or spend | Design and preflight both disallow an agent loop; all counters remain zero. | Enforced as an offline gate. |
| Schema/runtime compatibility must be explicit | `semantic_anchor_response_v1` is a five-field strict schema; request fixtures embed per-case JSON schemas; all 24 schemas pass the offline vLLM 0.24/XGrammar unsupported-feature linter. | Offline fixtures and schema lint validated; live compiler attestation still pending. |
| Anchor typing/scope must be unambiguous | Synthetic registry covers exact `3 x 3 x 2` select grid plus six mandatory abstentions. | Offline fixtures validated; model behavior unknown. |
| Ledger integrity must be raw-first and fail-closed | Ledger schemas, terminal receipt schema, replay mutation registry, and terminal-state validator are committed. | Shapes validated; real runner/replay still pending. |
| Evaluation protocol must avoid qrels leakage | Gold labels are scorer-only artifacts; runner-visible artifact validation rejects gold/scorer leakage. | Offline separation validated; scorer execution still pending. |
| External cost must remain zero | Preflight reports zero model, retrieval, reranker, network, download, topic-file, and relevance-judgment counters. | Enforced for the offline packet. |

## Offline preflight evidence

The current offline command is:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight --pretty
```

Observed report summary from the committed tree:

- `schema_version`: `semantic_anchor_offline_preflight_report_v1`
- `experiment_id`: `det_sparse_exact_span_synthetic_v4`
- `status`: `offline_preflight_pass`
- `artifact_count`: 29
- `denied_topic_count`: 22
- `import_issues`: `[]`
- `source_path_fragment_issues`: `{}`
- `inference_authorized`: `false`
- `external_cost_authorized`: `false`
- `next_gate`: `advisor_review_before_live_attestation_or_model_inference`

The report binds every committed v4 contract artifact by SHA-256. The full hash
map is emitted by the command rather than copied here, so reviewers can rerun
the command against the exact current checkout.

The offline regression command used for this packet was:

```bash
.venv/bin/python -m pytest \
  code/tests/test_det_sparse_v3_config.py \
  code/tests/test_det_sparse_v3_preflight.py \
  code/tests/test_det_sparse_v4_contract.py \
  code/tests/test_det_sparse_v4_preflight.py \
  code/tests/test_det_sparse_v4_advisor_packet.py \
  -q
```

Result after adding packet and offline schema-compatibility checks: `92 passed`.

## What is not yet proven

This packet should not be read as a completed v4 implementation. The following
remain open gates:

1. Live Lucene offset sidecar attestation and parity fixtures.
2. Live local model inventory attestation from the exact cached snapshot.
3. XGrammar/vLLM compiler compatibility for all 24 per-case schemas.
4. Create-only runner implementation with raw-first dispatch receipts.
5. Replay mutation execution against an actual run directory.
6. Scorer-only gold opening after all 24 raw responses are sealed.
7. Advisor approval before any untouched-topic or v5 confirmation milestone.

Until those are complete, v4 is an offline contract and review packet, not an
inference-authorizing system.

## Advisor questions

1. Is the local exact-span selector a reasonable conservative exception handler
   for vocabulary gaps, given that dense retrieval is not a practical primary
   option here?
2. Should the synthetic set add more abstention cases before model dispatch, or
   is the current 18 select + 6 abstain design adequate for a first gate?
3. Are the admission thresholds too strict, too loose, or correctly fail-closed
   for a sparse system that must avoid wrong referents?
4. Should any part of the local model request contract change before live
   compiler/runtime attestation?
5. If v4 passes synthetic qualification, should the next milestone be a new
   separately versioned untouched-topic confirmation set rather than a broader
   rerun?

## Recommendation

Proceed only to the next offline-to-local gate: live local runtime attestation
and compiler checks for the already cached small model. Do not run retrieval,
reranking, real topics, relevance judgments, a larger model, downloads, paid
services, or an agent loop unless the advisor explicitly approves a later gate.

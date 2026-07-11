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
| Source/import closure must be fail-closed | Preflight reports a static direct source audit over the v4 preflight, contract, schema-compat, and no-dispatch runner modules; denied imports, denied path fragments, unexpected import roots, and unexpected `trec_rag` modules must all be empty. It also wraps offline preflight in a runtime file-open audit that must observe read-only access, no denied topic/qrels/cache/model path fragments, and zero writes. | Offline direct-source closure and runtime file-open tracing validated, including model-cache and safetensors path denial. |
| Schema/runtime compatibility must be explicit | `semantic_anchor_response_v1` is a five-field strict schema; request fixtures embed per-case JSON schemas; all 24 schemas pass the offline vLLM 0.24/XGrammar unsupported-feature linter; offset health/error/response schemas are bound to executable Python validators; offset responses validate shape, hash, fingerprint, and monotonicity; chat completions have a strict assistant-content extractor that rejects non-JSON, duplicate keys, tool calls, wrong roles, wrong case IDs, and malformed ranges; captured live schema-compiler evidence must bind vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`, the fixed case-order hash, and all 24 request hashes. | Offline fixtures, schema lint, offset sidecar schema validation, assistant-content extraction, and live-evidence validators covered; actual live compiler capture and live Java parity are still pending. |
| Local model inventory must be exact and cost-free | The committed inventory fixture is validated for schema version, repository, revision-bound snapshot path, quantization, total safetensors size, three loaded shards, loaded/unloaded/denied disjointness, and lowercase SHA-256 coverage for every listed loaded, unloaded, and denied file. Captured runtime evidence must bind served model `gpt-oss-local`, repository `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, vLLM/XGrammar versions, backend `xgrammar`, loopback-only serving, egress denial, read-only model mount, and the expected model-inventory SHA-256 before it can be projected into a ledger pre-dispatch attestation. | Offline fixture validation and live-evidence validators strengthened; actual live local inventory capture still pending. |
| Live attestation records must be exact before dispatch | Offline validators now define the required schema compiler attestation, live model runtime attestation, and combined live-attestation bundle. They bind vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`, served model `gpt-oss-local`, repository/revision, request identity hashes, loopback-only serving, egress denial, read-only model mount, and model-inventory SHA-256. | Attestation record shapes are validated offline; actual live collection still pending. |
| Anchor typing/scope must be unambiguous | Synthetic registry covers exact `3 x 3 x 2` select grid plus six mandatory abstentions. | Offline fixtures validated; model behavior unknown. |
| Ledger integrity must be raw-first and fail-closed | Ledger schemas, terminal receipt schema, replay mutation registry, terminal-state validator, executable ledger-prefix validator, and offline replay mutation oracles are committed. The prefix validator checks reservation/dispatch/raw/receipt hash links, terminal counters, manifest drift, transport-failure no-body handling, and completed 24-case terminal states. The mutation oracles execute all eight registered failure-code surfaces in memory. The pre-dispatch replay verifier rereads an actual create-only run directory from sealed files and rejects noncanonical bytes, artifact-hash drift, reservation hash drift, extra dispatch artifacts, and nonzero terminal counters. | Offline shapes, prefix mechanics, mutation failure surfaces, and actual zero-dispatch run-directory replay validated; post-dispatch replay still pending. |
| Runner entrypoint must not dispatch before approval | A create-only runner scaffold writes 24 request-hash reservation records, a fresh run manifest, and a `pre_dispatch_no_go` terminal receipt, validates the zero-call ledger prefix, refuses existing output directories, and performs no model, retrieval, reranking, topic, relevance-judgment, network, download, or model-file access. The replay verifier accepts only that exact file set and terminal state. | Offline no-dispatch runner boundary, reservation materialization, and no-go replay validated; live dispatch remains closed. |
| Runner request identities must be fixed before dispatch | Preflight reports a `request_identity` block with fixed 24-case order, first scored case, canonical request-body SHA-256s, and request byte sizes. | Reservation and dispatch identities are offline-bound; no runner dispatch yet. |
| Evaluation protocol must avoid qrels leakage | Gold labels are scorer-only artifacts; runner-visible artifact validation rejects gold/scorer leakage. The scorer/gold linter validates case order, U1-bounded acceptable and wrong-referent ranges, abstain reasons, and deterministic response classifications (`correct_select`, `safe_abstain`, `wrong_referent`, `wrong_abstain`, `mechanical_failure`). | Offline separation and scorer mechanics validated; scorer execution still pending. |
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
- `source_audit`: 4 source files, `status=pass`,
  `unexpected_import_roots=[]`, `unexpected_trec_rag_modules=[]`,
  `denied_import_issues=[]`, `denied_path_fragment_issues={}`
- `runtime_file_access`: `status=pass`, read-only observed paths,
  `observed_write_path_count=0`, `denied_path_fragment_issues={}`,
  `denied_write_paths=[]`; denied fragments include topic/qrels/cache paths,
  Hugging Face model cache paths, `models--openai--gpt-oss-20b`, model
  snapshots, and `.safetensors`
- `schema_compatibility`: 24 cases, offline
  `vllm_0_24_xgrammar_unsupported_feature_lint`, `status=pass`,
  `unsupported_feature_issues=[]`
- `request_identity`: 24 cases, first case `synthetic-case-001`, fixed
  case-order hash, canonical request SHA-256s, and request byte sizes
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
  code/tests/test_det_sparse_v4_runner.py \
  code/tests/test_det_sparse_v4_advisor_packet.py \
  code/tests/test_query_schema_compat.py \
  -q
```

Result after adding packet, source/import audit, offline schema-compatibility
checks, request identity binding, offset response validation, ledger-prefix
validation, offline replay mutation oracles, and scorer/gold classification
validation, assistant-content extraction, offset health/error schema checks,
runtime file-open tracing, live attestation evidence validators, and
pre-dispatch run-directory replay:
`145 passed`.

## What is not yet proven

This packet should not be read as a completed v4 implementation. The following
remain open gates:

1. Live Lucene offset sidecar attestation and parity fixtures.
2. Live local model inventory capture from the exact cached snapshot.
3. Live XGrammar/vLLM compiler capture for all 24 per-case schemas.
4. Post-approval live dispatch runner implementation with raw-first dispatch receipts.
5. Replay mutation execution against an actual post-dispatch run directory.
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

# Deterministic sparse v4 advisor review packet

Date: 2026-07-11

Status: offline review packet. This document does not authorize model
inference, topic access, relevance-judgment access, retrieval, reranking,
downloads, hosted APIs, paid calls, or an agent loop.

Baseline implementation reviewed by this packet: the current
`codex/structured-query-planner` checkout. The packet avoids embedding its own
commit hash so the review text remains stable across final commit/amend steps.

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
| Source/import closure must be fail-closed | Preflight reports a static direct source audit over the v4 preflight, contract, schema-compat, default no-dispatch runner CLI paths, and the gated injected dispatch helper; denied imports, denied path fragments, unexpected import roots, and unexpected `trec_rag` modules must all be empty. It also wraps offline preflight in a runtime file-open audit that must observe read-only access, no denied topic/qrels/cache/model path fragments, and zero writes. | Offline direct-source closure and runtime file-open tracing validated, including model-cache and safetensors path denial. |
| Schema/runtime compatibility must be explicit | `semantic_anchor_response_v1` is a five-field strict schema; request fixtures embed per-case JSON schemas; all 24 schemas pass the offline vLLM 0.24/XGrammar unsupported-feature linter; offset health/error/response schemas are bound to executable Python validators; offset responses validate shape, hash, fingerprint, and monotonicity; a file-backed offset parity review requires the full 48-row live fixture grid and stable offset fingerprint before the model gate; chat completions have a strict assistant-content extractor that rejects non-JSON, duplicate keys, tool calls, wrong roles, wrong case IDs, and malformed ranges; captured live schema-compiler evidence must bind vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`, the fixed case-order hash, all 24 request hashes, and all 24 per-case response-schema hashes. | Offline fixtures, schema lint, offset sidecar schema validation, offset parity review mechanics, assistant-content extraction, request/schema identity binding, and live-evidence validators covered; actual live compiler capture and live Java parity fixture capture are still pending. |
| Local model inventory must be exact and cost-free | The committed inventory fixture is validated for schema version, repository, revision-bound snapshot path, quantization, total safetensors size, three loaded shards, loaded/unloaded/denied disjointness, and lowercase SHA-256 coverage for every listed loaded, unloaded, and denied file. A read-only local snapshot inventory capture command now emits the same attestation shape from an existing caller-supplied cached snapshot, rejects wrong revisions, missing/extra loader-shard shape, symlinks, wrong shard total size, and original full-weight loading, and performs no model inference, network, or download. Captured runtime evidence must bind served model `gpt-oss-local`, repository `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, vLLM/XGrammar versions, backend `xgrammar`, loopback-only serving, egress denial, read-only model mount, and the expected model-inventory SHA-256 before it can be projected into a ledger pre-dispatch attestation. | Offline fixture validation, read-only capture mechanics, and live-evidence validators strengthened; actual capture against the real local snapshot is still pending. |
| Live attestation records must be exact before dispatch | Offline validators now define the required schema compiler attestation, live model runtime attestation, and combined live-attestation bundle. They bind vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`, served model `gpt-oss-local`, repository/revision, request identity hashes, loopback-only serving, egress denial, read-only model mount, and model-inventory SHA-256. The preflight CLI can validate a captured canonical bundle file into a `semantic_anchor_live_attestation_review_v1` report only after a canonical `semantic_anchor_offset_parity_review_v1` is supplied; it binds the offset review SHA-256 and offset fingerprint SHA-256, binds the bundle SHA-256 with `bundle_canonical=true`, projects the pre-dispatch ledger attestation, traces read-only file access, and still leaves `dispatch_authorized=false`. | Attestation record shapes, offset-parity-to-live-attestation binding, canonical bundle sealing, and bundle-review path are validated offline; actual live collection still pending. |
| Anchor typing/scope must be unambiguous | Synthetic registry covers exact `3 x 3 x 2` select grid plus six mandatory abstentions. | Offline fixtures validated; model behavior unknown. |
| Ledger integrity must be raw-first and fail-closed | Ledger schemas, terminal receipt schema, replay mutation registry, terminal-state validator, executable ledger-prefix validator, and offline replay mutation oracles are committed. The prefix validator checks reservation/dispatch/raw/receipt hash links, terminal counters, manifest drift, transport-failure no-body handling, and completed 24-case terminal states. The mutation oracles execute all eight registered failure-code surfaces in memory. The pre-dispatch replay verifier rereads an actual create-only run directory from sealed files and rejects noncanonical bytes, artifact-hash drift, reservation hash drift, extra dispatch artifacts, and nonzero terminal counters. The completed-run replay verifier rereads a 24-case synthetic run directory from sealed files, rejects extra artifacts, missing raw files, terminal counter drift, premature `gold_opened=true`, sealed response order drift, validates reservation/dispatch/raw/receipt ledger links, requires `raw_sealed_pending_scorer` with `gold_opened=false`, and validates the sealed scorer input before scorer-only gold can open. | Offline shapes, prefix mechanics, mutation failure surfaces, actual zero-dispatch run-directory replay, and offline completed synthetic run-directory replay validated; replay against an actual live post-dispatch run still pending. |
| Runner entrypoint must not dispatch before approval | A create-only runner scaffold writes 24 request-hash reservation records, a fresh run manifest, and a `pre_dispatch_no_go` terminal receipt, validates the zero-call ledger prefix, refuses existing output directories, and performs no model, retrieval, reranking, topic, relevance-judgment, network, download, or model-file access. The replay verifier accepts only that exact file set and terminal state. The post-approval synthetic dispatch function requires a validated live-attestation review, advisor-GO review, and explicit manual runner invocation receipt, accepts only an explicitly injected transport, writes reservation/dispatch/raw-response/case-receipt records, creates a canonical sealed scorer input, leaves scorer-only gold unopened, and seals bad/no-body transport output as `transport_no_body_no_go` before raising. | Offline no-dispatch runner boundary, reservation materialization, no-go replay, fake-transport dispatch implementation, and failed-transport terminal sealing validated; actual live model invocation remains closed. |
| Runner request identities must be fixed before dispatch | Preflight reports a `request_identity` block with fixed 24-case order, first scored case, canonical request-body SHA-256s, per-case response-schema SHA-256s, and request byte sizes. | Reservation and dispatch identities are offline-bound; live compiler attestations and fake-transport dispatch validate the same hashes; actual live invocation pending. |
| Evaluation protocol must avoid qrels leakage | Gold labels are scorer-only artifacts; runner-visible artifact validation rejects gold/scorer leakage. The scorer/gold linter validates case order, U1-bounded acceptable and wrong-referent ranges, abstain reasons, and deterministic response classifications (`correct_select`, `safe_abstain`, `wrong_referent`, `wrong_abstain`, `mechanical_failure`). A scorer-only CLI now accepts only canonical sealed synthetic responses, validates a `raw_sealed_pending_scorer` 24-response terminal state with `gold_opened=false`, validates ledger-prefix evidence, verifies parsed-response raw body hashes against raw response records and case receipts before opening gold, emits `semantic_anchor_scorer_review_v1`, and keeps inference/dispatch/external-cost authorization false. Reviewer receipts require at least two unanimous reviewers and are bound to the exact scorer review, sealed responses, gold artifact, rubric, and artifact bundle hashes. A reviewer-qualification review CLI maps only that bound consensus plus the scorer receipt to `completed_synthetic_go`; any scorer miss maps to `completed_qualification_no_go`. | Offline separation, scorer mechanics, canonical sealed-input binding, raw-body binding, scorer-only gold-opening boundary, bound reviewer consensus, and reviewer GO/NO-GO mapping validated; scorer execution on a real post-dispatch run remains pending. |
| Untouched-topic/v5 milestone must require advisor approval | A file-backed untouched-topic milestone approval review requires a canonical `semantic_anchor_reviewer_qualification_review_v1` with `completed_synthetic_go`, a canonical `semantic_anchor_untouched_topic_milestone_advisor_approval_v1` receipt, and explicit acknowledgments that consumed dev topics and known-five topics remain closed, a new separately versioned confirmation set is required, and retrieval/reranking/paid calls need a later gate. | Executable offline gate validated; it does not authorize topic access, retrieval, reranking, or external cost. |
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
- `source_audit`: 5 source files, `status=pass`,
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
  case-order hash, canonical request SHA-256s, canonical per-case
  response-schema SHA-256s, and request byte sizes
- `inference_authorized`: `false`
- `external_cost_authorized`: `false`
- `next_gate`: `advisor_review_before_live_attestation_or_model_inference`

The report binds every committed v4 contract artifact by SHA-256. The full hash
map is emitted by the command rather than copied here, so reviewers can rerun
the command against the exact current checkout.

Once offset parity, model inventory, schema-compiler, and model-runtime evidence
files are separately produced, the canonical live-attestation bundle is assembled
create-only with:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --assemble-live-attestation-bundle \
  --offset-parity-review path/to/offset-parity-review.json \
  --model-inventory-attestation path/to/model-inventory-attestation.json \
  --schema-compiler-attestation path/to/schema-compiler-attestation.json \
  --model-runtime-attestation path/to/model-runtime-attestation.json \
  --output path/to/live-attestation-bundle.json
```

That assembler rereads the canonical offset fixture path named by the offset
review, checks the fixture SHA-256 and offset fingerprint against the review,
validates all four evidence files against the frozen request/model identities,
and writes only canonical compact JSON because the next review gate rejects
noncanonical bundle bytes. It performs no dispatch, model inference, retrieval,
reranking, topic/qrels access, download, or network work.

Assembled live attestation bundles are then reviewed with:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --live-attestation-bundle path/to/live-attestation-bundle.json \
  --offset-parity-review path/to/offset-parity-review.json \
  --model-inventory-attestation path/to/model-inventory-attestation.json \
  --schema-compiler-attestation path/to/schema-compiler-attestation.json \
  --model-runtime-attestation path/to/model-runtime-attestation.json \
  --output path/to/live-attestation-review.json \
  --pretty
```

That review validates the bundle against the frozen request identity and the
canonical file-backed model-inventory, schema-compiler, and model-runtime
attestation SHA-256s, requires the bundle to exactly embed those same compiler
and runtime records, validates and binds the canonical offset parity review plus
offset fingerprint SHA-256, emits a pre-dispatch ledger attestation, requires
canonical JSON bundle bytes, reports `bundle_canonical=true`, traces file opens
for all five evidence files, and explicitly keeps `dispatch_authorized=false`.

Captured live offset parity fixtures, once produced by the loopback sidecar, are
reviewed separately before model inventory/compiler gates:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --offset-parity-fixtures path/to/offset-parity-fixtures.json \
  --output path/to/offset-parity-review.json \
  --pretty
```

That review requires canonical fixture bytes, exact 48-row coverage for the
eight surface classes by three unit positions by two punctuation contexts, the
exact fixture text contract for every grid cell, analyzer-zero rows with no
occurrences, nonzero rows with at least one occurrence, unique fixture IDs,
stable offset fingerprint across every response, canonical request/response
SHA-256 binding, executable offset response validation, zero cost counters, and
no inference or dispatch authorization.

An already-cached local model snapshot can be inventoried read-only without
loading the model:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --model-inventory-snapshot path/to/gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
  --output path/to/model-inventory-attestation.json
```

That capture hashes only files already present under the supplied snapshot,
requires the expected revision path, three expected local shard files with the
expected combined size, keeps `original/model.safetensors` denied/unloaded, and
does not authorize model inference, network, download, retrieval, reranking, or
topic/qrels access. Running this command against the real local model cache is
still a separately approved local cache-read gate because it hashes large model
files.

A later advisor/user GO receipt is also reviewed offline before any runner
entrypoint can be invoked. The receipt must use
`semantic_anchor_advisor_dispatch_go_v1`, name the
`det_sparse_v4_synthetic_local_dispatch` scope, bind the exact
`semantic_anchor_live_attestation_review_v1` canonical SHA-256, and acknowledge
that topic, qrels, retrieval, reranking, and paid-call gates remain closed. The
file-backed review rejects noncanonical live-attestation review or advisor-GO
receipt bytes before deriving any GO review. The
derived `semantic_anchor_advisor_dispatch_go_review_v1` report still keeps
`inference_authorized=false`, `dispatch_authorized=false`, and
`external_cost_authorized=false`; its next gate is
`manual_runner_invocation_still_required`.

The file-backed CLI path is:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --live-attestation-review path/to/live-attestation-review.json \
  --advisor-go-receipt path/to/advisor-go.json \
  --output path/to/advisor-go-review.json \
  --pretty
```

It requires the advisor GO receipt to be canonical JSON bytes, writes the review
create-only, and still does not invoke the runner.

The final manual runner invocation gate is also file-backed before any transport
is created. The receipt must use `semantic_anchor_manual_runner_invocation_v1`,
bind both the live-attestation review SHA-256 and the advisor-dispatch-GO review
SHA-256, keep `egress_allowed=false` and `external_cost_authorized=false`, name
the `injected_local_loopback` transport kind, and explicitly set local-only
`inference_authorized=true` plus `dispatch_authorized=true`. The review command
validates the canonical receipt and emits
`semantic_anchor_manual_runner_invocation_review_v1`; it still does not create a
network client, invoke a transport, or dispatch the model:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_runner \
  --validate-manual-runner-invocation \
  --live-attestation-review path/to/live-attestation-review.json \
  --advisor-dispatch-go-review path/to/advisor-go-review.json \
  --manual-runner-invocation path/to/manual-runner-invocation.json \
  --output path/to/manual-runner-invocation-review.json
```

Sealed synthetic responses are scored only after run-directory replay proves
all 24 raw responses are committed. The offline completed-run replay verifier
already exercises that boundary on a sealed synthetic run directory; an actual
live post-dispatch run remains a later gate. The completed-run replay verifier is
invoked with:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_runner \
  --replay-completed-synthetic path/to/run-directory
```

The same runner module can also replay a zero-dispatch no-go directory:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_runner \
  --replay-pre-dispatch-no-go path/to/run-directory
```

Both replay modes are file-backed validators; neither dispatches the model,
opens scorer-only gold, or changes the terminal receipt.
They bind runner-visible artifact hashes without opening scorer-only gold; the
scorer-only artifact bundle is opened only by the scorer after the raw ledger
seal is validated.

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_scorer \
  path/to/sealed-scorer-input.json \
  --output path/to/scorer-review.json
```

The scorer requires canonical sealed-input bytes, validates
`raw_sealed_pending_scorer` with all 24 raw responses committed and
`gold_opened=false`, validates ledger-prefix evidence, verifies each raw body
hash against raw response records and case receipts, then opens scorer-only gold
labels, emits `semantic_anchor_scorer_review_v1`, and does not authorize
inference, dispatch, or external cost.

The reviewer qualification gate is executable offline: reviewer receipts require
at least two unanimous reviewers and must bind to the exact scorer review, sealed
responses, gold artifact, rubric, and artifact bundle hashes. The terminal-state
mapper returns `completed_synthetic_go` only for a valid bound reviewer receipt
plus all `correct_select` / `safe_abstain` scorer classifications. Any scorer
miss maps to `completed_qualification_no_go`.

The qualification review is then written create-only from the canonical scorer
review and canonical reviewer receipt:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_scorer \
  path/to/scorer-review.json \
  --qualification-review \
  --reviewer-receipt path/to/reviewer-receipt.json \
  --output path/to/reviewer-qualification-review.json
```

Before any future untouched-topic or v5 confirmation milestone can even be
designed, a separate advisor approval receipt is reviewed create-only:

```bash
.venv/bin/python -m trec_rag.det_sparse_v4_preflight \
  --reviewer-qualification-review path/to/reviewer-qualification-review.json \
  --milestone-approval-receipt path/to/milestone-approval-receipt.json \
  --output path/to/milestone-approval-review.json \
  --pretty
```

That receipt must bind the exact reviewer-qualification review SHA-256, require
`completed_synthetic_go`, acknowledge that consumed development topics and the
known-five topics remain closed, require a new separately versioned confirmation
set, and acknowledge that retrieval, reranking, and paid calls remain behind a
later gate. The emitted review does not authorize topic access, retrieval,
reranking, or external cost.

The offline regression command used for this packet was:

```bash
.venv/bin/python -m pytest \
  code/tests/test_det_sparse_v3_config.py \
  code/tests/test_det_sparse_v3_preflight.py \
  code/tests/test_det_sparse_v4_contract.py \
  code/tests/test_det_sparse_v4_preflight.py \
  code/tests/test_det_sparse_v4_runner.py \
  code/tests/test_det_sparse_v4_scorer.py \
  code/tests/test_det_sparse_v4_advisor_packet.py \
  code/tests/test_query_schema_compat.py \
  -q
```

Result after adding packet, source/import audit, offline schema-compatibility
checks, request/schema identity binding, offset response validation, ledger-prefix
validation, offline replay mutation oracles, and scorer/gold classification
validation, assistant-content extraction, offset health/error schema checks,
file-backed offset parity review, offset-parity-to-live-attestation binding,
runtime file-open tracing, live attestation evidence validators,
captured-bundle review, canonical live-attestation bundle sealing, advisor
dispatch-GO binding, file-backed advisor-GO review CLI,
file-backed manual runner invocation review, pre-dispatch run-directory replay,
offline completed synthetic run-directory replay, runner replay CLI modes,
completed replay mutation coverage, runner-visible artifact hashing without
scorer-only gold opens, reviewed fake-transport dispatch implementation,
failed-transport terminal sealing, canonical sealed scorer input,
ledger-bound scorer-only sealed-response review, bound reviewer qualification
review, reviewer qualification terminal-state mapping, executable
untouched-topic/v5 milestone approval gating, and read-only model inventory
capture mechanics:
`199 passed`.

## What is not yet proven

This packet should not be read as a completed v4 implementation. The following
remain open gates:

1. Live Lucene offset sidecar attestation and parity fixtures.
2. Live local model inventory capture from the exact cached snapshot.
3. Live XGrammar/vLLM compiler capture for all 24 per-case schemas.
4. Actual post-approval live invocation of the dispatch runner against the local model.
5. Replay mutation execution against an actual live post-dispatch run directory.
6. Scorer-only gold opening against an actual post-dispatch run.
7. Actual advisor approval before any untouched-topic or v5 confirmation milestone.

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

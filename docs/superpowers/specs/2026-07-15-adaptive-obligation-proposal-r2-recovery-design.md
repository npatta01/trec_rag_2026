# Adaptive Obligation Proposal R2 Recovery Design

## Status and objective

Proposal run R1 stopped correctly after its first immutable attempt. The local
Qwen model returned valid JSON with valid support IDs, but its
`scope_rationale` contained 245 characters while the frozen schema allowed at
most 240. The ledger recorded one `started` event, the exact 546-byte raw
completion, and one terminal `schema_error` event. The remaining 47 jobs were
not called.

R2 must recover without editing, reclassifying, retrying, or accepting the R1
completion. It must preserve the same evidence boundary, model snapshot,
accepted proposal schema, and truncation-only retry rule while making the
length contract salient at the end of each long prompt.

## Rejected alternatives

- Do not widen the accepted schema. A larger limit weakens concision and does
  not solve instruction adherence.
- Do not truncate or normalize the R1 output. That would transform a terminal
  model result after inference.
- Do not reinterpret the R1 schema error as truncation. The output ended
  naturally at 186 tokens, below the 256-token ceiling.
- Do not add grammar-constrained decoding yet. The JSON structure, enums, and
  support IDs were already correct; exact Unicode string-length enforcement
  would add disproportionate runtime and dependency complexity.

## Immutable R1 incident record

The existing R1 preflight, approval, ledger, event chain, and raw completion
remain byte-immutable. A separate create-only incident directory records the
aborted state without writing inside the ledger root:

`outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident/`

Its canonical receipt binds:

- the R1 preflight receipt path and SHA-256;
- the R1 approval path and SHA-256;
- the ledger path, anchor SHA-256, head/event-chain SHA-256, and event count;
- every preserved raw completion path, byte count, and SHA-256;
- one attempted job, one terminal schema error, and 47 uncalled jobs;
- reason code `scope_rationale_length_exceeded`;
- observed length 245 and accepted maximum 240; and
- zero network, retrieval, hosted inference, paid, and qrels access.

The incident builder independently replays and verifies these facts. It cannot
seal, repair, delete, or append to the R1 ledger.

## Versioned R2 contract

R2 uses distinct schema identities for its prompt, job, preflight, approval,
and run outputs. It lives beside R1 under new create-only destinations:

- `proposal_preflight_r2/`
- `proposal_approval_r2.json`
- `proposal_ledger_r2/`
- `proposals_r2/`

The R1 verifier and artifacts remain independently usable and unchanged. The
R2 implementation is isolated behind versioned public entry points and may
reuse only stable low-level hashing, snapshot, ledger, and local-runtime
primitives.

R2 preserves exactly:

- pilot topics `219`, `72`, `300`, and `84`;
- protected-topic rejection for `144`, `213`, `224`, `407`, and `515`;
- 24 O0 parents and two folds per parent, producing 48 jobs;
- the complete R1 evidence units and parent/fold coverage boundaries;
- model `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- deterministic decoding;
- the accepted schema limits of 120 label characters and 240 rationale
  characters;
- 256 primary output tokens and one 512-token retry only for incomplete JSON
  that reaches the primary ceiling; and
- 48 primary calls with a maximum 48 truncation retries.

No R1 raw output or derived proposal is copied into R2. All 48 R2 jobs must run
under one homogeneous R2 prompt and preflight.

## R2 prompt rendering

The original scope instructions remain in the system message. Each user
message retains the unchanged narrative, complete parent O0, exact evidence
units, source fold, and full response schema.

A compact `output_contract` is serialized after the long evidence and response
schema so it is tail-adjacent to generation. It states:

- return exactly one JSON object and no prose;
- label target: at most 80 characters and at most 10 words;
- rationale target: exactly one short sentence, at most 160 characters and at
  most 25 words;
- cite one or two supplied support unit IDs for `SUPPORTED`;
- return `UNSUPPORTED` rather than violating the contract; and
- silently self-check the limits before emitting JSON.

The tighter targets are generation guidance, not a change to the accepted
120/240-character schema. Tail placement is part of the hashed R2 prompt
contract and is covered by tests.

## Preflight and approval boundary

The inference-free R2 preflight authenticates the R1 source contract, rebuilds
all 48 jobs, loads only the pinned tokenizer, and records exact per-job prompt
token counts. It freezes the model snapshot, tokenizer identity, prompt, schema,
jobs, code hashes, output ceilings, destinations, and all safety counters.

R2 inference cannot start until a new create-only approval binds the exact R2
preflight SHA-256, R2 approval schema, model and revision, 48 primary calls, 48
possible truncation retries, and the absolute `proposal_ledger_r2` destination.
The prior R1 approval cannot authorize R2.

The executor processes jobs sequentially. Job one is therefore a natural
fail-fast canary: any terminal schema or semantic error stops the run before
job two. A successful first job is not a separate tuned phase and does not
change the frozen remaining jobs.

## Finalization and downstream compatibility

Only a fully sealed R2 ledger with 48 valid results may produce
`proposals_r2/`. Finalization preserves raw hashes, support-unit bindings,
source contract identity, prompt and schema hashes, attempt ordinals, and output
token counts. The existing opposite-fold validation stage may consume R2 only
after its authenticated proposal loader explicitly recognizes the R2 receipt
and independently replays the complete R2 source chain.

Validation, BM25 retrieval, MiniLM scoring, qrels, and promotion remain outside
the R2 proposal-run approval.

## Tests and acceptance

Tests must prove:

- the R1 incident receipt reproduces the exact observed failure and rejects any
  changed approval, anchor, event, raw byte, count, or reason;
- R1 artifacts remain byte-identical and pass their existing verifier;
- all R2 messages end with the frozen compact output contract;
- R2 retains identical topic, parent, fold, evidence-unit, and accepted-schema
  boundaries;
- R2 prompt token counts and all source/code/model hashes are exact;
- R1 approval is rejected by R2 before model or output access;
- R2 approval is rejected if its preflight hash or ledger destination differs;
- schema errors remain terminal and only exact-ceiling incomplete JSON may
  retry;
- the public executor has no model, ledger, prompt, or destination injection;
- an invalid first result leaves jobs 2 through 48 uncalled;
- a complete fake-runtime run seals 48 results through the real ledger; and
- no protected topic, network, retrieval, paid service, or qrels access occurs
  during incident capture or preflight construction.

After the code and tests pass, build the R1 incident receipt and R2 preflight
only. Report the exact R2 hash, token counts, estimated runtime, and call ceiling
to the user. Do not create R2 approval or begin R2 inference until the user
separately approves that exact frozen preflight.

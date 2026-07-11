# Query planner v2 synthetic transport smoke 001

Date: 2026-07-11

## Decision

**Mechanical failure; stop and preserve.** The one preregistered synthetic
transport/schema request failed before inference because vLLM 0.24.0's grammar
compiler does not implement the JSON Schema keyword `uniqueItems`. This result
is excluded from every semantic diagnostic, holdout, and retrieval analysis.
It was not retried.

## Frozen boundary

- Run ID: `query_plan_v2_synthetic_smoke_001`
- Run kind: `synthetic_transport_smoke`
- Delivered schema version: `query_plan_v2`
- Delivered prompt version: `sparse_query_planner_v5`
- Topic: built-in `synthetic_transport_v2`, outside the development topics
- Model endpoint: loopback-only `gpt-oss-local`
- Model repository/revision: `openai/gpt-oss-20b` at
  `6cee5e81ee83917806bbde320786a8fb61efebee`
- vLLM: 0.24.0, local ROCm container
- Model image digest:
  `sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`
- Reference-analyzer fingerprint SHA-256:
  `f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4`
- Context preflight: 1,967 prompt + 5,400 maximum output = 7,367 of
  8,192 tokens, leaving 825
- Execution boundary: serial, local-only; zero retrieval, reranker, and paid API
  calls

The separately captured runtime record is
`docs/superpowers/query_planner_v2_local_runtime.json`.

## Observed outcome

- Top-level status: `http_error`
- HTTP status: 400
- Request elapsed time: 0.0115 seconds
- Successful plans: 0
- Failed outcomes: 1
- Missing outcomes: 0
- Model completion tokens: none; grammar compilation rejected the request
  before generation
- Exact response body:

  ```json
  {"error":{"message":"Grammar error: Unimplemented keys: [\"uniqueItems\"]","type":"BadRequestError","param":null,"code":400}}
  ```

- Raw body length: 125 bytes
- Raw body SHA-256:
  `939bd0eb124619d6812e66dadc3084c3b778b5a8f175ae1f9d92fa8353fa0afe`
- Raw base64 round trip and SHA verification: pass
- Offline `--rebuild-only` integrity validation: pass; expected nonzero exit
  because the preserved run contains one failure

Authoritative local artifacts are under
`outputs/query_planner_v2_synthetic_smoke_001/`. The raw response was fsynced
with create-only semantics before UTF-8 or JSON decoding, followed by one
immutable per-topic terminal outcome.

## Interpretation and next gate

`uniqueItems` constrains duplicate ID references in the model-facing schema.
Accepted plans already pass Python's `_v2_ids` duplicate-reference validation,
so decoder-side removal can retain the accepted-plan invariant. It would still
be a schema compatibility change and must be versioned and reviewed before any
new request.

No compatibility patch or second call is authorized by this result alone. The
IR advisor is reviewing whether to preregister a distinct one-shot compatibility
smoke or stop this planner arm. The formal five-topic diagnostic remains
NO-GO.

# Runpod Flash Batch Passage Scoring Design

Date: 2026-08-09

Status: Approved for implementation

## Objective

Move the expensive Mixedbread passage-scoring misses in agentic retrieval to a
scale-to-zero Runpod Flash endpoint while preserving the competition runner's
current content-addressed score-cache guarantees. The endpoint must improve
throughput through request batching and GPU inference without becoming an
authority for retrieval state, topic state, or published evidence.

The first implementation targets many crawler processes on one orchestrator
host. Those processes continue to share the repository's existing SQLite score
cache and its claim/lease protocol. Multiple independent crawler machines and a
distributed canonical score cache are intentionally outside this first cut.

## Selected Approach

Implement a queue-based, class-form Runpod Flash `@Endpoint` and an optional
remote prediction backend behind `MixedbreadPassageScorer`.

The class-form endpoint loads the pinned CrossEncoder once per worker. Each job
contains one query and an ordered batch of passages. The local scorer checks and
claims cache misses before submitting a job, validates the complete response,
and commits accepted scores through `GlobalScoreCache`. Cache hits never contact
Runpod.

This approach is preferred over two alternatives:

- A load-balanced REST service favors low-latency interactive calls but gives
  up the queue semantics that fit durable batch inference.
- An endpoint-owned distributed score cache would deduplicate across multiple
  crawler machines, but it adds a second claims/leases implementation and a
  shared-storage consistency boundary before that capability is needed.

## Architecture

```text
agentic topic processes
        |
        v
MixedbreadPassageScorer
        |
        +--> GlobalScoreCache hit ----------> scored passage
        |
        +--> claimed cache misses
                  |
                  v
          Flash batch client
                  |
                  v
       Runpod queue-based endpoint
       (0-3 GPU workers, model warm)
                  |
                  v
       validated ordered raw logits
                  |
                  v
       GlobalScoreCache commit
```

The endpoint is stateless with respect to score results. Runpod persistent
storage is used only for model files. Topic records, passages selected for
publication, score-cache claims, and canonical score rows remain local to the
competition runner.

## Flash Worker Configuration

The initial deployment contract is:

- Queue-based class `@Endpoint`.
- `workers=(0, 3)` so no GPU worker remains active indefinitely.
- `idle_timeout=900` seconds so bursts of adaptive searches reuse a warm model.
- One GPU per worker from the 24 GB Ada group for the initial benchmark.
- `max_concurrency=1` because one worker should execute one GPU scoring job at a
  time; request-level batching provides utilization.
- FlashBoot enabled.
- One Runpod Network Volume mounted for the Hugging Face model cache.
- A finite execution timeout large enough for cold initialization plus one
  maximum-size scoring request.
- Pinned Python dependencies matching the repository's CUDA scoring contract.

The maximum worker count is a cost and concurrency guard, not a target. The
first benchmark starts with one submitted job at a time and increases crawler
concurrency only after correctness and throughput are established.

## Model and Score Identity

The endpoint pins all inference-relevant values:

- model: `mixedbread-ai/mxbai-rerank-base-v2`
- revision: `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`
- maximum sequence length: 1,024 tokens
- score representation: raw logits with identity activation
- inference dtype: bfloat16
- backend: `sentence-transformers` CrossEncoder
- backend version: 5.6.0
- fixed internal microbatch policy
- CUDA device family and endpoint implementation version

Remote CUDA results initially use a distinct `ScoreCacheContext` from local
ROCm results. They must not be written under the existing local passage-cache
identity merely because the model revision and dtype match. A later parity
study may authorize promotion or identity unification only after measuring
score and ranking differences on fixed real and synthetic inputs.

The endpoint returns its complete identity with every response. The caller
compares it with the configured expected identity before accepting any score.

## Batch Contract

Each request is JSON with:

- a schema version;
- an idempotency/request identifier;
- the expected endpoint identity digest;
- one nonblank query string;
- an ordered list of passage records containing a caller-generated content ID
  and nonblank passage text.

The first request ceiling is 256 passages and an encoded payload safely below
Flash's 10 MB request limit. Query text appears once rather than once per pair.
The client may lower the request size without changing score identity.

Each response contains:

- the same schema version and request identifier;
- the actual endpoint identity and digest;
- one ordered result per input content ID;
- one finite real-valued raw logit per result;
- server-observed scoring and batch counts for diagnostics.

The caller rejects the entire response when identifiers are missing,
duplicated, reordered, or unexpected; counts differ; identity differs; or any
score is Boolean, non-numeric, NaN, or infinite. No partial response is
committed to the canonical cache.

## Caching

Caching is deliberately layered:

1. `GlobalScoreCache` is the canonical scored-pair cache. Its content hashes,
   SQLite claims, heartbeat leases, and atomic commits deduplicate work across
   local topic processes. Cache hits bypass the endpoint.
2. A Runpod Network Volume stores the pinned Hugging Face model snapshot. A
   worker that starts after scale-down reuses the downloaded snapshot.
3. The endpoint class holds one loaded CrossEncoder in memory for the life of a
   warm worker. The 15-minute idle timeout amortizes model initialization across
   nearby adaptive searches.

The endpoint does not store raw queries, passages, or score-result records on
the Network Volume. Request and worker logs must not print query or passage
text. Diagnostic fields use counts, timings, request IDs, and content hashes.

## Local Scorer Integration

`MixedbreadPassageScorer` gains an injected batch-prediction boundary while its
public `rank(query_text, chunks)` behavior remains unchanged. The local backend
continues to lazy-load the pinned model. The remote backend submits cache misses
to Flash.

The cache claim batch size and the endpoint's internal GPU microbatch size are
separate parameters:

- Request batch size controls how many claimed pairs cross the network in one
  job.
- GPU microbatch size controls `CrossEncoder.predict` memory use and throughput
  inside a worker.

This separation is required because the current local scorer uses one
`batch_size` value for both operations. Remote scoring should send hundreds of
pairs per request while the model may process only 8, 16, or 32 at a time.

Remote scoring is opt-in through ignored/local configuration or explicit
environment-backed settings. The canonical full-run config is not changed
until a benchmark and two-topic smoke run validate the remote path. Secrets and
endpoint IDs are not written into tracked configuration.

## Failure and Retry Behavior

Transport retries and semantic validation are separate:

- Retry bounded transient transport failures, queue timeouts, HTTP 429, and
  retryable 5xx responses with backoff.
- Do not retry malformed responses, identity mismatches, invalid scores, or
  permanent authorization errors as though they were transport failures.
- Use idempotent request identifiers so retrying a submitted batch is
  diagnosable and safe.
- Keep the existing `GlobalScoreCache` heartbeat active while remote inference
  is pending so another local process does not steal a live claim.
- If a batch ultimately fails, release its claims and fail that scoring call;
  do not substitute local inference unless an explicit fallback policy is
  configured and recorded.

The first implementation defaults to fail-closed remote behavior. This makes
benchmark and smoke-run evidence attributable and prevents an unnoticed mix of
remote and local score identities.

## Security and Privacy

- Resolve the Runpod API key from the environment or local Flash credentials.
- Never include the key in tracked files, process output, exception messages,
  or request payloads.
- Do not log raw narrative, query, passage, or model response content.
- Return only scores and identity metadata; the endpoint creates no retrieval
  or publication artifacts.
- Keep all benchmark outputs under ignored/private paths.
- Do not deploy or invoke a paid remote endpoint without an explicit run
  authorization that states the requested topic/input scope.

## Verification

Implementation is test-first and has four gates:

1. Contract tests cover request validation, response ordering, finite-score
   validation, identity matching, payload ceilings, and diagnostic redaction.
2. Scorer integration tests prove cache hits make no remote call, concurrent
   local callers deduplicate misses, failed batches commit no scores, and local
   scoring behavior remains unchanged.
3. A local fake-transport benchmark validates batching and accounting without
   Runpod access or cost.
4. After separate authorization, `flash dev` runs a remote parity and throughput
   benchmark. It compares fixed CUDA outputs with the local ROCm scorer and
   records cold-start time, warm throughput, queue delay, payload size, internal
   microbatch size, GPU identity, and estimated cost per million pairs.

Only after those gates pass should an ignored two-topic agentic smoke config
select the remote backend. The canonical 119-topic config remains untouched
until the smoke run verifies sealed outputs and expected cache reuse.

## Expected Repository Changes

- A focused Flash endpoint module under `code/tools/`.
- A reusable remote batch client and prediction protocol under
  `code/trec_rag/`.
- A small refactor of `MixedbreadPassageScorer` to separate request and model
  microbatching while preserving its public ranking API.
- Targeted unit and integration tests under `code/tests/`.
- README instructions for local testing, Flash development, deployment,
  teardown, cache identity, and the paid-run authorization boundary.

No public service, production deployment, canonical full-run config change, or
multi-machine distributed score cache is part of this design.

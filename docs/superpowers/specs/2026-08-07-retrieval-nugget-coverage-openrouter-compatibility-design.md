# Retrieval Nugget Coverage OpenRouter Compatibility Design

**Date:** 2026-08-07

## Objective

Restore live Retrieval Nugget Coverage evaluation on OpenRouter and use the
current accuracy-first model selected by the user. The change must preserve the
evaluator's bounded two-stage architecture, private-data boundary, strict
structured outputs, and resumable artifact integrity.

## Verified failure

The v1 provider request includes `temperature: 0` together with
`provider.require_parameters: true`. OpenRouter's current endpoint metadata for
the configured OpenAI models does not advertise `temperature`, so strict
routing cannot select an endpoint. Topic `14` created only its authenticated
`input.json`; no planner output was admitted and no other topic made a hosted
call.

## Approved decisions

1. Remove `temperature` from both planner and judge provider requests.
2. Keep `provider.require_parameters: true` and
   `provider.data_collection: "deny"`; OpenRouter must continue to reject a
   route that cannot honor every remaining request parameter or the data policy.
3. Keep strict JSON Schema output, `seed: 0`, reasoning disabled, non-streaming
   transport, bounded transport retries, and zero semantic retries.
4. Change both default model identities from `openai/gpt-5` to
   `openai/gpt-5.6-sol`. The first 22-topic evaluation will use the same model
   for planning and judging so its results do not mix model families.
5. Advance the evaluator identity from `retrieval_nugget_coverage_v1` to
   `retrieval_nugget_coverage_v2`. Prompt versions remain unchanged because the
   semantic planner and judge instructions are unchanged.
6. Do not reuse the failed v1 topic-14 namespace. Start all 22 topics in a new,
   private v2 work namespace. Preserve the old private input artifact for audit;
   do not delete or reinterpret it.

## Request and artifact behavior

The planner still receives only one narrative. The judge still receives that
narrative, the frozen obligation plan, and the ordered canonical nugget text.
Selected passages, retrieval, reranking, generation, qrels, and gold nuggets
remain outside the evaluator.

The exact serialized provider request remains hash-bound in each stage
artifact. Changing the model defaults, request body, and evaluator identity
therefore changes the persisted identity deliberately. Resume must reject v1
or differently modeled stage artifacts rather than silently reuse them.

The skill and README will state the new defaults and continue to require an
explicit opt-in before hosted calls. The existing authorization for this run is
limited to OpenRouter, `openai/gpt-5.6-sol`, the 22 named 2025 topics, and the
same narrative/plan/canonical-nugget payload categories.

## Error handling

Fresh cache-only create remains write-free. Hosted create writes the bound
input before the planner call, and any failure leaves a resumable partial
namespace. A failed planner still prevents the judge. No command may retry a
semantic failure automatically or continue to later topics after a probe
failure without diagnosis.

## Testing and verification

Test-first coverage will prove that the serialized planner and judge requests:

- omit `temperature`;
- retain strict provider/data-collection routing, reasoning disabled, seed
  zero, strict JSON Schema, and non-streaming behavior;
- use `openai/gpt-5.6-sol` by default;
- carry the v2 evaluator identity into persisted artifacts and reject stale
  identities on resume.

After targeted and full repository tests pass, run one new-namespace topic as
an end-to-end probe. Continue the remaining topics only if the receipt is
complete with exactly two hosted calls and validated artifacts. Aggregate
scores only from authenticated complete manifests and keep every input,
response, and report private and outside git.

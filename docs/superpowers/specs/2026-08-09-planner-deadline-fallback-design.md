# Planner Deadline Fallback Design

## Goal

Complete the existing `rag26-ms1` production run by reusing the six already-consumed Luna
planner responses, without changing temperature, prompts, models, evidence, citation authority,
or the authenticated run identity.

## Observed Failure Boundary

All six responses passed the provider's strict response schema and were saved as authenticated
planner receipts. They failed only the local blueprint semantic validator:

- one response allocates 1,000 planning words instead of 850–950;
- four obligations contain an anchor that is not an exact substring of the official narrative;
- two responses repeat a normalized narrative-anchor set across obligations.

These checks occur before evidence projection. Claim aliases remain locally resolved against the
authenticated handoff, and the organizer answer validator independently enforces the 1,024-word
limit, citation domain, exact-hint rules, and schema.

## Decision

Keep `validate_blueprint` strict. Add a separate deadline recovery function that clones a
provider payload and performs only these deterministic transformations:

1. When total `target_words` is 951–1,024, proportionally rescale the obligation allocations to
   exactly 950 positive integer words. Totals below 850 or above 1,024 remain invalid.
2. Replace every non-matching narrative span with the complete authenticated official narrative.
3. If normalized anchor sets are duplicated after replacement, append the first distinct,
   already-authenticated span from another obligation. If no such span exists or the four-span
   ceiling prevents disambiguation, fail closed.
4. Pass the transformed payload through the unchanged strict validator. No transformed payload
   may proceed unless that validator accepts it.

The final answer may still contain up to 1,024 words; the 950-word transformed total is planning
headroom, not an answer truncation.

## Resume And Audit Behavior

The bounded runner invokes recovery only after strict validation rejects a reusable planner
payload. It writes the resulting ordinary authenticated `blueprint.state.json`, so all existing
state loading, projection, hashing, and resume checks remain unchanged. The topic state and
manifest record a sanitized `planner_fallback` entry containing the original validator category
and deterministic actions. No planner reservation or provider call is added.

The full-run orchestrator publishes the organizer JSONL only when all 119 topics have valid final
records. It reports one durable `planner_deadline_fallback` warning for each recovered topic.

## Safety Boundaries

- The fallback is limited to the three observed blueprint conditions.
- Unknown aliases, invalid priorities or modes, malformed structures, missing must obligations,
  unsafe word totals, and any other validator error still fail closed.
- Replacement anchors come only from the official narrative; provider text that is not in the
  narrative never enters the writer context as an anchor.
- Evidence projection remains derived only from authenticated claim aliases and selected evidence.
- Final organizer validation remains unchanged and authoritative.
- The existing `rag26-ms1` identity and 113 sealed finals are reused unchanged.

## Verification

Tests must demonstrate the three transformations, fail-closed behavior for unrelated errors,
resume without another planner call, durable warnings, and unchanged strict validation. After a
clean commit, resume `rag26-ms1`, require 119 final topics and six fallback warnings, then run the
repository submission validator/AutoJudge plus the production citation, narrative, word-limit,
reference-uniqueness, and run-ID checks.

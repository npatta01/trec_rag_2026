# DeepAgent Post-Batch Nuggetizer Probe

## Objective

Test whether one centralized post-batch Nuggetizer call improves the existing
topic-224 researcher output by canonicalizing duplicate or paraphrased
provisional claims while preserving grounded evidence references.

This is a throwaway logic probe, not production integration.

## Fixed input

- Phoenix trace: `f9d00a71e02ac6f940deb75a5090db5b`.
- Topic: `224`.
- Existing run: five completed researchers, sixteen returned snippets, and six
  ledger nuggets.
- No new retrieval run.
- No fabricated or inferred claims when the trace lacks reconstructible data.

The probe reads only the configured Phoenix project and reconstructs exact
snippet coordinates, provisional claims, and evidence references in memory. It
must validate every quote against its observed snippet before any hosted call.

## Compared paths

### A — Current direct merge

Use the six grounded nuggets already produced by the completed DeepAgent run as
the baseline. Preserve claim text and evidence references exactly.

### B — Explicit post-batch canonicalization

Send all reconstructible provisional claims from the completed researcher batch
to the existing bounded Nuggetizer adapter in one request. Include stable
evidence aliases and a compact existing-ledger view. The hosted creator may
merge, split, retain, or reject provisional wording, but its output must refer
only to supplied aliases.

Use the repository's configured OpenRouter DeepSeek V4 Flash canonical backend:
temperature zero, strict JSON, no reasoning, one transport attempt, and the
existing 120-second timeout. Do not invoke the raw Nuggetizer network handler.

## Output and success criteria

Print a sanitized comparison containing:

- input and output nugget counts;
- exact normalized duplicates;
- candidate paraphrase merges;
- provisional-to-canonical mappings;
- evidence aliases retained, combined, or orphaned;
- distinct supporting documents per canonical nugget;
- mechanical grounding failures;
- hosted call count and latency.

The stage is promising when it produces a non-redundant canonical set, retains
all evidence needed by accepted claims, creates no unknown evidence references,
does not merge materially distinct claims, uses exactly one hosted call, and
finishes within 120 seconds.

If there are no duplicate or paraphrased input claims, the valid conclusion is
that topic 224 does not exercise the proposed mechanism; the stage is not
declared effective from claim-count reduction alone.

## Failure policy

- Missing or redacted trace claim/evidence data: stop before the hosted call.
- Unknown snippet, mismatched document, or quote not contained in the snippet:
  stop before the hosted call.
- Malformed, timed-out, or provenance-invalid hosted output: report failure and
  retain path A unchanged.
- Do not retry the hosted call.
- Do not write snippets, raw trace payloads, credentials, or cloud responses to
  the repository.

## Approved grounded-ledger continuation

The raw researcher-bundle preflight found two quote/snippet mismatches and
therefore made zero hosted calls. The approved continuation changes only path
B's input boundary: use the six mechanically accepted, grounded ledger nuggets
from the same trace as the provisional batch. This tests centralized
canonicalization after grounding rather than trusting researcher-authored
coordinates. Keep the same evidence aliases, one-call limit, model, timeout,
sanitized output, and no-persistence policy. Do not repair, infer, or silently
accept the two invalid researcher citations.

## Scope exclusions

- Entailment verification.
- Local-model comparison.
- Changes to live DeepAgent orchestration or ledger schemas.
- Another topic-224 retrieval run.
- Production cache, retry, concurrency, or deployment design.

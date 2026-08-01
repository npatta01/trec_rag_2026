# DeepAgent Citation Handles — Design

**Date:** 2026-08-01

**Status:** proposed

**Supersedes the evidence contract in:**
`docs/superpowers/specs/2026-07-30-deepagent-evidence-coverage-map-design.md`

## Problem

A researcher today proves a claim by retyping the passage it just read:

```json
{
  "document_id": "shard_00796_30325",
  "snippet_id": "shard_00796_30325:0002",
  "page_index": 0,
  "quote": "Push factors include poverty, war, and persecution."
}
```

The SDK then checks that typing against the passage it already holds, and
discards anything that does not match exactly.

Every field above is already known to the SDK before the model speaks. The
model is being asked to copy data back to its owner, and copying is where it
fails. Two live runs on topic 224 lost evidence to transcription alone:

| Failure | Count | Cause |
| --- | ---: | --- |
| `UNKNOWN_SNIPPET` | 15 | `:` retyped as `_`, on a document genuinely inspected |
| `UNGROUNDED_QUOTE` | 2 | quote reworded or stitched across two passages |

Neither was a reasoning error. Both were transport corruption, and in both
cases the underlying evidence was real.

Grep confirms the redundancy: outside its own validation in `_ground_evidence`,
`quote` is read nowhere in `code/trec_rag/` except when the nuggetizer probe
reads it back out. It exists to be checked against its own source.

## Principle

Separate **generated** content from **referenced** content.

- A claim is generated. It cannot be verified by construction, which is exactly
  why a grounding stage exists at all.
- Evidence is a **reference**. It should be a handle and nothing else. Today it
  is a handle *plus a copy of what the handle points at*, and the SDK validates
  the copy against the original.

Anything the SDK already owns must never make a round trip through the model.

## Design

### Handles

When a snippet is first returned to any agent during an invocation, the ledger
assigns it a short handle: `S1`, `S2`, `S3`, … Assignment is monotonic and
**invocation-scoped**, not page-scoped, so:

- paginating the same document never reuses `S1`–`S10`;
- three concurrent researchers never collide;
- a handle means one thing for the whole run, including across rounds.

Snippet text is split once, at observation time, into sentences. The resulting
character spans are stored on the observation. Resolution never re-splits, so a
handle cannot drift.

### Citation syntax

| Form | Means |
| --- | --- |
| `S3` | the whole of snippet 3 |
| `S3.2` | sentence 2 of snippet 3 |
| `S3.2-4` | sentences 2 through 4 of snippet 3, contiguous |

Sentence indices are 1-based. Ranges must be contiguous and ascending;
non-contiguous selections are rejected rather than silently joined.

### Researcher output

`BundleEvidence` collapses to one field:

```python
class BundleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cite: str = Field(min_length=2)   # "S3", "S3.2", "S3.2-4"
```

`document_id`, `snippet_id`, `page_index`, and `quote` are removed from the
model's output and **derived** by the SDK. `EvidenceReference` in the report
keeps all four, so the coverage report, the nuggetizer probe, and canonical
evidence aliases are unaffected. They simply become SDK-authored rather than
model-authored.

### State delta

`EvidenceDelta` becomes `{"cite": "S3.2"}` for both `add_nuggets` and
`add_evidence`. `_ground_evidence` is replaced by `_resolve_citations`, which
looks up the handle and slices the stored spans.

Rejection codes shrink:

| Code | When | Replaces |
| --- | --- | --- |
| `UNKNOWN_CITATION` | handle never observed this invocation | `UNKNOWN_SNIPPET`, `UNKNOWN_DOCUMENT` |
| `INVALID_CITATION` | malformed syntax, or sentence index out of range | `INVALID_EVIDENCE`, `MISSING_EVIDENCE` |

`UNGROUNDED_QUOTE` is **deleted**. There is no copy left to mismatch, so the
whitespace-normalising containment check and the "never fuzzy-repair a quote"
rule both go with it.

`DUPLICATE_EVIDENCE` stays, now compared on resolved spans.

## What this buys

- The largest observed failure class disappears by construction, not by
  validation.
- `S3.2` has no internal punctuation that looks like a mistake worth tidying,
  which is precisely what broke `shard_00796_30325:0002`.
- Researcher bundles stop carrying long quote strings, cutting output tokens,
  latency, and cost on the calls we make most.
- Less code: one resolver replaces a validator plus four rejection codes.
- A small local model becomes viable. Emitting long verbatim strings is the
  hard part for a weak model; emitting `S3.2` is not. This change is a
  precondition for that experiment, not an alternative to it.

## What this does not fix

The model can still cite the *wrong* sentence. Pointing at real text that does
not support the claim remains possible and remains undetected. Entailment
verification stays out of scope and stays a separate later stage, exactly as
the prototype design records.

This design narrows the untrusted surface to two things: which snippet was
chosen, and what claim was written. It does not shrink it to zero.

## Risks and mitigations

**Sentence splitting on messy web text.** ClimbMix is web text; abbreviations,
URLs, and list fragments will produce imperfect sentences. Mitigation: the
splitter is deterministic and length-capped, over-long segments sub-split at a
fallback boundary, and `S3` remains available to cite a whole snippet.

The important property holds regardless of split quality: whatever the SDK
slices is *real stored text*. A bad split yields an imprecise quote, never an
ungrounded one. Precision degrades; provenance cannot.

**Character offsets were considered and rejected.** Having the model emit
`[start, end]` positions looks similar but is much worse: models cannot count
characters, so wrong-but-in-range offsets would attach silently incorrect
evidence. Sentence indices are read off a numbered list rather than computed,
which is why they survive contact with a language model. A loud rejection beats
a quiet error in a provenance system.

**Payload size.** Numbering sentences adds tokens to each snippet response.
Researcher output shrinks by more, since quotes leave it entirely. Expected net
win, to be measured rather than assumed.

**Thread safety.** Handle assignment happens while up to three researchers run
concurrently. The registry must be allocated under the same lock that guards
snippet observation; verify during implementation.

## Cache impact

None. Sentence splitting and handle assignment happen at the tool boundary,
after the cached page is read — the same place `snippet_id` is already exposed.
`SnippetPage.as_dict()` is what the snippet result cache stores, and it is not
touched, so existing cached pages stay readable and no re-fetch is triggered.

Handles are invocation-local, so nothing about them needs to be stable across
runs and nothing needs a cache-identity bump.

## Migration

This is a POC. The old evidence shape is removed rather than dual-supported;
accepting both would reintroduce the translation step this design exists to
delete.

Touched: `deepagent_research.py` (schema, researcher prompt),
`deepagent_evidence.py` (registry, resolver, codes),
`deepagent_retrieval.py` (payload, coordinator prompt), plus tests.

## Test plan

- handle assignment is monotonic, invocation-scoped, and survives pagination
- concurrent observation from three researchers assigns unique handles
- `S3`, `S3.2`, and `S3.2-4` resolve to exactly the stored text
- out-of-range sentence, reversed range, non-contiguous range, and malformed
  syntax each reject with `INVALID_CITATION`
- an unobserved handle rejects with `UNKNOWN_CITATION`
- resolved evidence carries the correct derived `document_id`, `snippet_id`,
  and `page_index`
- a snippet whose text contains no sentence terminator still yields one
  citable sentence
- cached snippet pages written before this change still load

# Selected-Evidence Fixed Generation Design

**Status:** Approved by the user on 2026-08-04.

## Objective

Make the fixed one-shot DeepSeek and Sol runs consume the exact topic-owned passages selected
by retrieval, and eliminate citation ambiguity before either model can produce organizer output.

## Root Cause

The v1 generation runner reconstructs generation input from a TREC run plus full-document text.
That duplicates retrieval's source-of-truth, can expose arbitrary document heads instead of the
ranked passages, and makes topic/document binding depend on a second join. Its numeric citation
contract also lets the model define both the `references` order and the indexes into that order,
so the model can silently bind a sentence to the wrong allowed document.

## Boundary

Retrieval publishes one sealed `generation_handoff_manifest_v1` after all per-topic projections
have validated. Each topic contains the exact official narrative, selected clusters, exact
source-bound passages, optional advisory canonical nuggets, a topic semantic seal, and an ordered
citation-document domain. Full documents and document-head windows are not generation inputs.

`competition_rag_config_v2` accepts only the handoff path. DeepSeek and Sol use the same pinned
one-shot prompt, schema, local validation, resume identity, and post-processing. Their model
settings are the only intended differences.

```text
per-topic TopicRecords v4 + sealed selection + canonical nuggets
                          |
                          v
          canonical/generation-projection.json
                          |
              validate against paired organizer projection
                          |
                          v
          generation_handoff_manifest.json (manifest-last)
                          |
             +------------+------------+
             |                         |
          DeepSeek                    Sol
             |                         |
             +------ raw docids -------+
                          |
            validate against that topic's citation domain
                          |
            deterministic organizer index conversion
                          |
                          v
             rag_output_trec_rag_2026.jsonl
```

## Handoff Contract

- The root and every topic use canonical UTF-8 JSON with SHA-256 seals and duplicate-key
  rejection.
- The producer source contract is `topic_records_v4`.
- Selected passage text is reconstructed from authenticated document-store offsets before the
  per-topic projection is sealed.
- Every evidence row belongs to exactly one selected cluster and retains document ID, document
  rank, document hash, character offsets, and byte offsets.
- Canonical nuggets are advisory claim hints. They cannot introduce evidence or document IDs.
- Incomplete retrieval topics may be projected when selected evidence exists; status remains in
  the paired retrieval projection. A topic with no selected evidence fails projection loudly.
- The root retrieval export manifest is written last and authenticates the organizer run,
  full-text archive, and generation handoff.

## Citation Contract

The provider schema requires one to three exact ClimbMix document-ID strings per answer object.
The model may not emit numeric citations. Code verifies every citation and every model-supplied
reference against the current topic's sealed `citation_docids`, removes duplicate citations
without changing their order, derives the final references list from first citation use, and
maps those document IDs to zero-based organizer indexes.

The same path is mandatory for Sol and DeepSeek. Sol's stronger schema adherence may reduce
failures, but it does not make model-created reference numbering trustworthy.

## Retry and Failure Semantics

- Transport retries remain bounded by `transport_max_attempts` inside one provider call.
- Generation allows at most two semantic attempts per topic: the initial completion and one
  fresh completion after malformed JSON, retryable incomplete provider metadata, foreign
  document IDs, numeric citations, or local schema/evidence validation failure.
- Truncation, policy/content filtering, terminal HTTP errors, and exhausted transport retries do
  not trigger another semantic attempt.
- Every raw response or sanitized failure record is persisted under a monotonic attempt number.
- If both semantic attempts fail, no topic row or final submission is published. The topic error
  remains visible and `resume` can try again with new monotonic attempt numbers.
- There is no fallback model, evidence substitution, citation deletion, citation fabrication, or
  partial-answer publication.

## Scope

Included:

- strict handoff records and selected-evidence renderer;
- TopicRecords v3 projection and paired per-topic publication;
- manifest-last root handoff publication;
- v2 fixed one-shot runner and DeepSeek/Sol configs;
- raw-document-ID citation validation and bounded semantic retry;
- migration of organizer document readers only as needed to keep evaluation utilities separate.

Excluded:

- paired agentic writer/critic/repair experiments;
- RAGDoll scoring behavior changes;
- live hosted model calls;
- legacy v1 generation compatibility.

## Verification

Tests must prove strict deterministic handoff loading, exact UTF-8 passage geometry, no full-doc
sentinel leakage, paired organizer/generation document identity, incomplete-with-evidence
projection, v2-only config parsing, raw-document-ID conversion, rejection of numeric/foreign
citations, two-attempt exhaustion, monotonic resume artifacts, DeepSeek/Sol shared code paths,
resume identity coverage, and manifest-last publication. No hosted calls are permitted during
implementation verification.

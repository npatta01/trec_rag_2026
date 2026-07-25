# Topic 213 Evidence Handover Design

**Date:** 2026-07-25
**Status:** Approved in conversation
**Audience:** A colleague developing TREC RAG response-generation methods

## Objective

Produce a fixed, inspectable evidence handover for RAG25 development Topic 213
(the Korean War). For each of the ten released nugget-derived sub-narratives,
the handover will identify five ClimbMix documents and summarize the claims
each document supports.

The handover isolates response generation from retrieval. It is development
evidence, not hidden-test evidence and not a claim about official 2026 test
performance.

## Inputs

- Topic 213 from the pinned `trec-rag-data` submodule.
- Topic 213 nuggets and their exact `mapped_sub_narrative` values.
- The recommended Codex UMBRELA qrels.
- The authenticated all-topic accepted union containing ClimbMix document text.

Only documents with organizer-projected qrel grade 2, 3, or 4 are eligible.
The expected eligible population is 173 documents.

## Scoring

Every eligible document is scored against every sub-narrative with the pinned
Mixedbread reranker. The model score is a shortlisting signal, not a calibrated
probability and not an organizer judgment.

The final reviewed records keep two scores separate:

- `topic_qrel_grade`: organizer-projected whole-topic relevance, 2 through 4.
- `support_score`: reviewer judgment for the specific sub-narrative:
  - `0`: does not support the sub-narrative;
  - `1`: related or mentions it, but lacks usable detail;
  - `2`: partially supports at least one usable claim;
  - `3`: directly and substantially supports one or more usable claims.

The final top five for each sub-narrative must have `support_score >= 2`.
Mixedbread score orders candidates before review; the reviewer may reorder the
final five based on directness, detail, and non-redundant claim coverage.

## Output

The durable handover lives under:

`reports/experiments/rag25_topic213_evidence_handover_v1/`

It contains:

- `README.md`: provenance, limitations, and colleague instructions.
- `handover.json`: narrative, sub-narratives, and five reviewed documents per
  sub-narrative.
- `handover.md`: approachable rendering of the same records.
- `manifest.yaml`: source revisions, scoring policy, model identity, and
  artifact hashes.

Raw document text, credentials, absolute home paths, and passage offsets are
excluded from tracked outputs. The user requested whole-document claim mapping,
so the handover names supported claims without recording their locations.

## Review and limitations

The released qrels do not map documents to sub-narratives. This handover is a
derived mapping produced by model shortlisting and evidence review.

Nuggets may contain duplicates, spelling errors, or questionable assertions.
They define coverage areas but are not copied into supported claims unless the
selected document independently supports them.

The final artifact must verify that:

- all ten exact sub-narratives appear once;
- every sub-narrative has exactly five unique documents;
- every selected document belongs to the 173-document eligible population;
- every selected record has `support_score` 2 or 3 and at least one claim;
- no raw document text, secrets, or absolute machine paths are present.


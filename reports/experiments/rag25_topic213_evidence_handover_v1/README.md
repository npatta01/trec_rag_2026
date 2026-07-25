# Topic 213 evidence handover

This directory is the durable, text-safe colleague handover for Topic 213 of
the TREC RAG 2026 development set. It identifies five reviewed ClimbMix
documents for each of the ten released sub-narratives and states only the
concise claims that each whole document supports.

## Files

- `handover.json` is the canonical machine-readable mapping.
- `handover.md` is a deterministic rendering of the same mapping.
- `COLLEAGUE_TASK.md` defines the downstream passage-extraction,
  summarization, and response-generation assignment.
- `manifest.yaml` records source identity, population and review counts,
  scoring definitions, limitations, and artifact hashes.

No file in this directory contains source document text, passage excerpts,
passage offsets, credentials, model caches, or machine-local absolute paths.
Document text used during review remains in ignored workspace outputs.

## How the evidence was chosen

The eligible population is the 173 Topic 213 documents with organizer qrel
grade 2, 3, or 4. A pinned Mixedbread cross-encoder ranked all 1,730
document/sub-narrative pairs using raw logits and the strongest semantic chunk
score for each whole document. Review then covered all 12 ranked candidates
for every sub-narrative, not only the final five.

Candidate generation was entirely local: every one of the 173 eligible
documents was scored for each sub-narrative before the leading 12 were
shortlisted. The remote document endpoint was used only to acquire missing
eligible document bodies by known ID; it was not used for top-12 retrieval.

Review support scores mean:

- `0`: the document does not support the sub-narrative;
- `1`: the document is related but insufficient or indirect;
- `2`: the document supports at least one useful concrete claim;
- `3`: the document provides direct, detailed support.

Final documents have support score 2 or 3. Selection favors direct evidence,
specificity, and complementary coverage; the organizer qrel grade remains a
separate topic-level provenance field and is not a sub-narrative score.

The source-text-free reviewer reports and final adjudication records remain in
the ignored review workspace. `manifest.yaml` pins their basenames, roles, and
SHA-256 hashes; raw-text packets and machine-local paths are not tracked.

## Reproduce the tracked contracts

From the repository root:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py -k canonical
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py
```

The canonical tests validate the exact ten released labels, five unique
eligible documents per label, qrel-grade provenance, minimum support score,
sanitization, and byte-for-byte Markdown rendering through
`render_handover_markdown`.

## Limitations

- The shortlist is a review aid, not a calibrated relevance probability.
- Only the leading 12 model-ranked documents per sub-narrative received this
  evidence review; documents below rank 12 remain unknown.
- Five documents are a concise handover, not an exhaustive bibliography.
- A document may appear under more than one sub-narrative when it supports
  distinct claims.
- Source quality varies, so claims are deliberately narrower than some source
  documents' broader interpretations.
- The armistice ended open hostilities but was not a peace treaty; wording in
  the handover preserves that distinction.

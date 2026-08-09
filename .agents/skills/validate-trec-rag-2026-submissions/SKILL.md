---
name: validate-trec-rag-2026-submissions
description: Use when checking, preflighting, auditing, or diagnosing TREC RAG 2026 Retrieval TSV or RAG JSONL submission artifacts before delivery.
---

# Validate TREC RAG 2026 Submissions

Validate each artifact in its native task format. Use the bundled structural
validator for Retrieval runs and the organizer's AutoJudge for RAG reports.

## Workflow

1. Read the canonical contract at
   `trec-rag-skills/skills/trec-rag-2026-track-guidelines/SKILL.md` and its
   Retrieval or RAG reference. A newer official release overrides this skill.
2. Start at the main repository root.
3. Use the official topics TSV or AutoJudge Request JSONL. Name every submission
   artifact explicitly.
4. Run one combined preflight when both task outputs exist:

```bash
.venv/bin/python .agents/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
  --topics path/to/trec_rag_2026_queries.tsv \
  --retrieval path/to/r_output_trec_rag_2026.tsv \
  --rag path/to/rag_output_trec_rag_2026.jsonl
```

Repeat `--retrieval` or `--rag` to check multiple files. Add `--strict-rag` only
when the user wants AutoJudge smells promoted by its strict mode.

## Routing and Results

| Artifact | Authoritative check used here | Key result |
|---|---|---|
| Retrieval TSV | Bundled standard-library validator | Six fields, exact topic population, dense per-topic ranks, finite non-increasing scores, unique ClimbMix document IDs, stable run ID |
| RAG JSONL | `autojudge-base>=0.4.3` with `check --spec rag26 --topics` | Organizer schema, topic, narrative, citation, and word-limit checks |

The script prefers a compatible local AutoJudge and otherwise uses an isolated
`uv` environment. `PASS` is ready for the checks performed. `PASS WITH WARNINGS`
means AutoJudge emitted a `SMELL`; report it and let the user decide whether to
use strict mode. `FAIL` blocks submission, and the process exits nonzero if any
artifact fails. Preserve the printed counts and validator detail in the verdict.

Retrieval depth is narrative-specific: report minimum and maximum depth, but do
not require equal depths, a fixed cutoff, padding, or topic ordering. Never send
a Retrieval TSV to AutoJudge or adapt it into synthetic empty-answer RAG JSONL.

## Privacy and Project Checks

Read only the topics and submission paths named for validation. Keep outputs
private; no qrels, gold nuggets, RAGDoll scores, corpus text, provider responses,
`.env` files, or credentials are needed.

In repositories with an authenticated Retrieval-to-Generation handoff, also
apply its additive checks after this portable preflight: exact topic IDs and
narratives, allowed citation domain, sealed-manifest integrity, and organizer
filenames. Do not replace those project invariants with this structural check.

## Common Mistakes

- Manually rebuilding TSV-to-Request JSONL conversion; the script does it in a
  temporary directory.
- Reporting only an AutoJudge traceback; lead with the normalized artifact
  status and keep the detail underneath.
- Treating warnings as success without mentioning them, or treating variable
  Retrieval depth as malformed.

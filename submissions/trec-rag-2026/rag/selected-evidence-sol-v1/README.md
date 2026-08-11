# TREC RAG 2026 selected-evidence RAG runs

This private repository bundle contains two organizer-facing Retrieval-Augmented
Generation (`RAG`) submission files generated from the same authenticated
selected-evidence handoff. Retrieval was not rerun between generation
strategies. Both exact files were uploaded and accepted by Evalbase.

## Accepted files

| Suggested priority | Strategy | Run ID | File |
|---:|---|---|---|
| 1 | Bounded multi-stage Sol + Luna | `rag26-ms1-final` | [`multistage/rag_output_trec_rag_2026.jsonl`](multistage/rag_output_trec_rag_2026.jsonl) |
| 2 | Single-pass Sol | `rag26-ss1` | [`singlepass/rag_output_trec_rag_2026.jsonl`](singlepass/rag_output_trec_rag_2026.jsonl) |

Start with the repository-wide [`submission ledger`](../../SUBMISSION_LEDGER.md).
It records the exact accepted-file hashes and links to the archived portal
notes for both runs. The [final architecture report](../../../../reports/2026-competition-architecture.html)
shows how both strategies consume the sealed handoff without reading the
organizer Retrieval TSV.

Both upload files use `NP`, the registered Evalbase short identifier for the
`NP Labs` organization, in every record's required `metadata.team_id` field.

## Shared retrieval and evidence boundary

Both runs consume the authenticated 119-topic handoff produced by
`facet-deepseek-b40-v3`, with manifest SHA-256
`31dc1b3578741339101b8199c0ea027e2d8405e828532aa58f86bf74c0542a0d`.
Generation reads only that sealed selected-evidence handoff. It does not read
the TREC retrieval run, full-text ZIP, qrels, gold nuggets, or RAGDoll scores.

## Generation strategies

- **Multi-stage:** Luna creates a bounded narrative blueprint and performs
  evidence audits and operation screening; Sol writes the draft and performs a
  bounded revision or repair when required. Six already-consumed planner
  responses required deterministic deadline normalization and were strictly
  revalidated before generation continued.
- **Single-pass:** Sol receives the complete selected-evidence context and
  writes the organizer record directly. Two topics used the configured second
  semantic attempt; all 119 topics produced valid final rows.

Both strategies used OpenRouter, medium reasoning effort, no explicit
temperature, strict structured output, a 12,000-token response ceiling, and a
four-topic concurrency limit.

## Validation

Both accepted files independently pass the organizer AutoJudge for `rag26` with
all 119 official narratives, no extras, and no duplicates. Project validation
also confirms exact narrative order and text, allowed ClimbMix citation domains,
exact-hint citation rules, unique references, and the 1,024-word limit.

The bundle intentionally excludes generation work state, raw provider
responses, prompts, evidence passages, retrieval artifacts, credentials, qrels,
gold nuggets, and evaluation scores. The organizer JSONLs necessarily retain
the official narratives, generated answers, reference document IDs, and
citations required for submission.

---
name: trec-rag-competition-debug-report
description: Use when a user asks to inspect, explain, visualize, audit, or debug the stages of a completed TREC RAG competition retrieval or RAG run.
---

# TREC RAG Competition Debug Report

Use the repository's post-run report CLI as the single implementation. This workflow reads completed artifacts and creates a private, standalone HTML explanation; requests to execute an incomplete run belong to the competition retrieval or RAG workflow.

Never run retrieval, reranking, models, or hosted APIs, and never copy, serve, or publish the report.

## Workflow

1. Use the supplied repository root, or walk upward from the current directory to the first root with the repository markers `pyproject.toml` and `code/trec_rag/competition_debug_report.py`; if none exists, ask for the repository path. Use supplied config paths. Otherwise, use the sole compatible completed retrieval run and, when requested, its matching standard RAG config. If multiple completed retrieval runs remain genuinely ambiguous, list them concisely and ask one short question for the retrieval-config path; do not guess, run every candidate, or ask for information that repository evidence already resolves.
2. Confirm the configured run is complete by checking that its sealed retrieval export exists and that the configured RAG output exists when a RAG config is included. If either requested run is incomplete, stop and identify the missing post-run artifact.
3. Invoke only the repository CLI. Pass each requested topic with a separate `--topic`; omit topic flags to include all exported topics.

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --topic TOPIC_ID
```

Omit `--rag-config` for retrieval-only reports. Omit `--topic` when the user requests every topic. Let the CLI choose its private default output unless the user explicitly requests an in-repository HTML path.

## Examples

Retrieval-only:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml
```

Retrieval plus RAG:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --rag-config configs/rag26_competition_rag_gpt_sol_v2.yaml
```

4. Read the single JSON receipt from stdout. Confirm it contains `schema_version`, an absolute `output_path`, `topic_ids`, `rag_included`, and `source_sha256s`, and that its topics and RAG status match the request.
5. Return the absolute report path, the JSON receipt, and a concise validation summary. Warn that the local HTML contains private corpus text, generated claims, answers, and document identifiers and must remain private.

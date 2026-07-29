# Competition Retrieval and RAG Design

## Objective

Integrate the answer-generation runner from PR #24 with the current
`origin/master` retrieval pipeline while keeping retrieval and answer generation
as two independently runnable competition paths. The paths communicate only
through the retrieval export files, not through Python imports or subprocess
chaining.

## Public commands

Retrieval has one public module and one versioned configuration:

```bash
uv run --no-sync python -m trec_rag.competition_retrieval \
  configs/rag26_competition_retrieval_v1.yaml
```

The current `trec_rag.official_run` module is renamed to
`trec_rag.competition_retrieval`. No compatibility alias remains. Tests,
documentation, and internal references move to the new name.

RAG answer generation remains a separate command with a strict, versioned
configuration:

```bash
uv run --no-sync python -m trec_rag.competition_rag \
  --config configs/rag26_competition_rag_gpt_sol_v1.yaml
```

Neither command invokes the other. A user may run retrieval alone, generate
answers from any compatible frozen retrieval export, or run both commands in
sequence.

## Environment and command convention

Environment creation and dependency synchronization use the repository's uv
workflow:

```bash
code/tools/setup_env.sh
```

The setup script selects the standard dependency set or the ROCm dependency
group for the active host. Subsequent commands use `uv run --no-sync python` so
uv executes inside that prepared `.venv` without replacing the hardware-aware
dependency selection. Documentation and verification commands use this form
consistently rather than calling `.venv/bin/python` directly.

## Configuration convention

Checked-in competition configurations follow
`rag<year>_<purpose>_<variant>_v<version>.yaml`:

- `configs/rag26_competition_retrieval_v1.yaml`
- `configs/rag26_competition_rag_gpt_sol_v1.yaml`

The retrieval configuration replaces `configs/facet_pilot_v1.yaml`; all tracked
references are updated. Its existing strict schema and experiment identity are
preserved: `schema_version` remains `facet_pilot_config_v1`, and `experiment.id`
remains `facet-deepseek-b40-v1`.

The generation configuration gains the explicit schema version
`competition_rag_config_v1` and uses the same strict-YAML behavior as retrieval:
reject unknown sections, unknown fields, duplicate keys, missing required
values, invalid types, and unsupported schema versions. Its input paths point
to `outputs/facet-deepseek-b40-v1/` rather than placeholder filenames.

## File contract between paths

RAG generation consumes exactly two retrieval artifacts plus the official topic
file:

- `r_output_trec_rag_2026.tsv`: six-column TREC run in official topic order.
- `retrieval_with_text.jsonl.zip`: one query record per topic with candidate
  objects containing raw `docid` and document text.
- `trec_rag_2026_queries.tsv`: authoritative topic ID and narrative source.

The generation loader validates that every selected topic exists in the TREC
run, ranks and document IDs are unique, every selected document is present in
the full-text archive, and only selected run documents can appear in the final
references. The existing retrieval export remains the producer of the TSV and
ZIP; answer generation does not depend on retrieval manifests or internal
checkpoint layouts.

## Generation behavior

PR #24's fixed-retrieval GPT Sol runner is applied on top of current
`origin/master`. It retains:

- a CLI that accepts only `--config`;
- repository `.env` loading for the configured OpenRouter API-key variable;
- create, resume, and overwrite modes;
- bounded concurrency and identical-request transport retries;
- exact organizer `metadata`, `references`, and `answer` JSONL output;
- zero-based integer citations and the 1,024-word limit;
- per-topic raw responses, validated resumable rows, and atomic final output.

Two review findings are part of the integration rather than deferred work:

1. `overwrite` deletes the configured generation output file and its dedicated
   `work/` directory before replacement work starts. It does not delete the
   retrieval inputs or any other experiment directory. A failed overwrite
   followed by resume cannot reuse a row from the pre-overwrite run.
2. The parsed model object must contain exactly `references` and `answer`.
   Unexpected model fields are rejected before deterministic metadata is added.

## Failure and recovery semantics

Create mode refuses any existing output or work artifact for the experiment.
Resume mode reuses only rows that still validate against the configured topic,
narrative, retrieval document IDs, and submission identity. Overwrite mode
starts with no reusable rows from an earlier run.

A topic failure records its error and, when available, the raw provider
response. The consolidated submission is published only after every official
topic has one valid row. Transport retries repeat the same request; malformed or
semantically invalid completions do not trigger a model repair call.

## Compatibility and migration

The integration branch starts at the current `origin/master`, then applies the
generation functionality rather than merging an outdated base. Retrieval's
current behavior, artifact schemas, seals, ordering, and standalone tests remain
unchanged except for the public module and config filenames.

Documentation presents the workflow as two commands with the file handoff made
explicit. References to `trec_rag.official_run` and `configs/facet_pilot_v1.yaml`
are removed so the repository has one advertised retrieval path.

## Verification

Implementation is test-driven and includes:

- a retrieval CLI test for the renamed `competition_retrieval` entrypoint;
- strict generation-config tests, including schema version and duplicate YAML
  keys;
- a contract test feeding retrieval-shaped TREC and ZIP artifacts into RAG
  generation;
- a regression test proving malformed model objects with extra fields fail;
- a regression test covering overwrite, failed replacement, then resume;
- the existing competition-generation tests from PR #24;
- the current retrieval, export, topic, and run-integrity tests from
  `origin/master`;
- module compilation and `git diff --check`.

No live model call is required for the automated suite. A live OpenRouter smoke
test remains optional and requires explicit use of the configured credentials.

## Scope boundaries

This change does not combine retrieval and generation into one orchestrator,
change retrieval ranking or evidence selection, add evaluation or UI code,
publish generated competition outputs, expose a service, or make live provider
calls during tests.

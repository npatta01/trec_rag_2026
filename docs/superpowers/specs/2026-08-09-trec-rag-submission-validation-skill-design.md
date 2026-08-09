# TREC RAG 2026 Submission Validation Skill Design

**Status:** Approved conversational design; implementation remains gated on review of this written specification.

## Objective

Add a reusable, versioned skill under `trec-rag-skills/skills/` that validates official TREC RAG 2026 Retrieval and Retrieval-Augmented Generation submission artifacts. The skill must choose the correct validator for each task, produce an evidence-backed pass/fail summary, and avoid treating a RAG report validator as a Retrieval runfile validator.

## Context

The organizer-provided command uses `autojudge-base>=0.4.3` and `autojudge_base.report_tool check --spec rag26`. That command validates RAG report JSONL. A Retrieval submission is instead a six-column TREC runfile, so passing it directly to `report_tool` fails during JSON parsing and does not validate Retrieval ranks or scores.

The existing `trec-rag-2026-track-guidelines` skill remains the canonical task contract. The new skill is an execution-oriented companion, not a replacement or duplicate specification.

## Approaches Considered

1. **Standalone skill with a deterministic script — selected.** One skill routes Retrieval TSV through a self-contained validator and RAG JSONL through AutoJudge. This is portable, repeatable, and keeps task selection explicit.
2. **Instructions-only skill.** This is smaller, but future agents would repeatedly reconstruct adapters and validation snippets, recreating the TSV/JSONL mismatch risk.
3. **Expand `trec-rag-2026-track-guidelines`.** This would mix canonical reference material with executable environment and dependency behavior, making the frequently used track skill larger and less focused.

## Placement and Files

Create one skill named `validate-trec-rag-2026-submissions`:

```text
trec-rag-skills/skills/validate-trec-rag-2026-submissions/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── scripts/
│   └── validate_submission.py
└── tests/
    └── test_validate_submission.py
```

No README, changelog, copied track specification, or static fixture corpus will be added. Tests will create minimal temporary fixtures.

## Skill Contract

The skill triggers when an agent is asked to validate, check, preflight, audit, or diagnose a TREC RAG 2026 Retrieval TSV or RAG JSONL submission. It requires the `trec-rag-2026-track-guidelines` skill as the canonical background contract.

The skill must enforce this routing rule:

- Retrieval TSV is validated as a TREC runfile.
- RAG JSONL is validated with `autojudge-base>=0.4.3` and `--spec rag26`.
- A Retrieval run is never converted into a synthetic empty-answer RAG report and presented as an authoritative Retrieval validation.

The skill reports the artifact paths, selected task, validator/version, topic and row counts, hard failures, warnings, and final status. It distinguishes official submission checks from optional repository-specific provenance or handoff checks.

## Command Interface

The bundled script exposes one entry point:

```bash
python scripts/validate_submission.py \
  --topics path/to/trec_rag_2026_queries.tsv \
  --retrieval path/to/r_output_trec_rag_2026.tsv \
  --rag path/to/rag_output_trec_rag_2026.jsonl
```

`--retrieval` and `--rag` are independently optional but at least one is required. Each option is repeatable so multiple candidate runs can be checked in one invocation. `--topics` accepts the official two-column TSV or AutoJudge Request JSONL. `--strict-rag` optionally promotes AutoJudge smells to failures; default behavior follows the organizer-provided command and preserves smells as warnings.

The script returns zero only when every requested artifact has no hard failure. Dependency/setup failures and malformed inputs return nonzero. It does not modify submissions.

## Retrieval Validation

Use only the Python standard library so Retrieval validation has no package or network dependency. For every runfile, check:

- UTF-8 text with exactly six whitespace-separated fields on every nonblank row;
- literal `Q0` in column two;
- topic IDs drawn exactly from the supplied topics, with no missing or extra topic;
- positive integer ranks starting at 1 and increasing densely for each topic in file order;
- finite numeric scores that are non-increasing within each topic;
- no duplicate topic/document pair;
- ClimbMix document IDs matching `shard_\d+_\d+`;
- one nonempty, stable run ID across the complete file; and
- at least one submitted document per represented topic.

Do not impose a fixed depth, maximum depth, equal depth, padding requirement, topic ordering rule, or unstated run-ID length limit. Report the observed row count, topic count, and minimum/maximum per-topic depth.

## RAG Validation

Delegate RAG structural and semantic submission checks to `autojudge_base.report_tool check` with `--spec rag26` and the supplied topics. Require AutoJudge version 0.4.3 or newer.

If the running Python environment already has a compatible AutoJudge, use it. Otherwise, invoke it in an isolated environment through `uv run --isolated --no-project --with 'autojudge-base>=0.4.3'`. If neither path is available, fail with a concise setup command rather than mutating the active project environment.

When `--topics` is TSV, convert it in a temporary directory to AutoJudge Request JSONL containing only `request_id` and `title`. Remove the temporary material when validation ends. Forward AutoJudge diagnostics and exit semantics without reclassifying hard failures as warnings.

AutoJudge is authoritative for the official RAG JSONL surface: required metadata, exact narrative matching, answer structure, references/citations, ClimbMix identifier shape, word limits, coverage, and duplicate topics. Repository-specific authenticated-handoff or manifest validation remains an additive gate when a repository requires it.

## Privacy and Scope Boundaries

- Read only the requested submissions and official topics for the generic validation pass.
- Do not read qrels, gold nuggets, RAGDoll scores, raw provider responses, `.env` files, tokens, or full corpus text.
- Do not publish, upload, serve, or copy submission contents.
- Keep temporary converted topics local and short-lived.
- Summaries may contain counts and paths; avoid echoing answer text or corpus text.

## Error Handling

Collect Retrieval violations across files and report paths plus line/topic context. A malformed row must not prevent other requested files from being checked. Preserve AutoJudge's distinction between `PROBLEM` and `SMELL`. A final summary must list each artifact as pass, pass-with-warnings, or fail and return a process status consistent with the worst result.

## Testing Strategy

Follow skill TDD:

1. Run a baseline agent scenario without the new skill and record the observed validator-routing failure.
2. Write script tests before implementation and verify they fail because the script is absent.
3. Cover valid variable-depth Retrieval; malformed columns; wrong `Q0`; missing/extra topics; bad ranks; duplicate documents; increasing/nonfinite scores; bad ClimbMix IDs; conflicting run IDs; and multiple input files.
4. Test TSV-to-Request conversion and AutoJudge command construction without network access.
5. Run a real isolated AutoJudge integration check against minimal valid and invalid RAG fixtures.
6. Validate `SKILL.md` and `agents/openai.yaml` with the skill-creator tooling.
7. Forward-test a fresh agent on Retrieval-only, RAG-only, and combined requests using raw fixtures and the finished skill.

## Completion Criteria

The work is complete when:

- the skill is discoverable from validation/preflight language;
- Retrieval and RAG artifacts are routed to their correct validators;
- all unit and integration checks pass;
- the skill folder passes `quick_validate.py`;
- forward-testing produces accurate, evidence-backed results without using prohibited data; and
- the `trec-rag-skills` submodule commit and the superproject submodule pointer contain only this requested change.

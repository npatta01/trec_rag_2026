# Task 3 report: generation execution, validation, and recovery

## Status

Implemented and verified in `/tmp/trec-rag-competition-paths`.

## Files

- `code/trec_rag/competition_rag.py`
  - Adds strict OpenRouter JSON transport, organizer-record construction and
    validation, fixed-retrieval generation, resumable per-topic artifacts,
    atomic final JSONL consolidation, and `--config`-only CLI execution.
  - Leaves Task 2 loaders and their strict contracts intact.
  - Rejects any generated root other than exactly `references` and `answer`
    before metadata injection.
  - In overwrite mode, removes only `config.resolved_work_dir` and
    `config.output_path`; retrieval inputs are read-only throughout.
- `code/tests/test_competition_rag.py`
  - Ports the generation, transport, resume, atomic-output, and CLI behavior
    tests using independent literal fixtures.
  - Adds reviewed regressions for generated-root validation, overwrite followed
    by failure then resume, and shared concurrency limiting.

## Red evidence

`uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

Before implementation, collection failed with:

```text
ImportError: cannot import name 'OpenRouterJsonGenerator' from 'trec_rag.competition_rag'
```

The shared-semaphore regression was then added before its fix and failed with:

```text
assert 2 == 1
```

for `concurrency=1`, proving topic calls were not sharing a semaphore.

## Green evidence

```text
uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q
68 passed in 0.14s

uv run --no-sync python -m compileall -q code/trec_rag/competition_rag.py
exit 0

git diff --check -- code/trec_rag/competition_rag.py code/tests/test_competition_rag.py
exit 0
```

## Commit

`Add fixed-retrieval RAG generation` (this Task 3 commit)

## Self-review

- Provider requests set strict JSON Schema, `provider.require_parameters`, and
  reasoning exclusion. Only request/429/5xx transport is retried, with an
  unchanged request body; malformed successful completions are captured without
  a repair call.
- Submission validation enforces exact five-field metadata, nonempty unique
  selected references, exact answer-object fields, one to three unique
  zero-based integer citations, every reference cited, and the 1,024-word cap.
- Topics are selected in canonical TSV order; final consolidation is written to
  a sibling temporary file and atomically replaced only after every topic row
  validates.
- Overwrite recovery is covered end-to-end: failed replacement clears the old
  final output and old work rows, and resume regenerates both topics without
  touching the run or document inputs.

## Concerns

- The broad `git status` command triggers an unrelated Git LFS filter failure
  in the nested `trec-rag-data` submodule. Targeted status with
  `--ignore-submodules=all`, diff checks, compilation, and the focused suite
  completed successfully.

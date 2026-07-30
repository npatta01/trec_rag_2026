# Tracing Package Split Design

**Date:** 2026-07-30

## Objective

Separate reusable trace representation and Phoenix export code from the temporary
organizer Pi experiment. The installed package should make the durable tracing
interface obvious without presenting Pi-specific parsing and reconstruction as
part of the main `trec_rag` module surface.

## Package layout

```text
code/trec_rag/
  tracing/
    __init__.py
    models.py
    openai_semantics.py
    phoenix_export.py
  experiments/
    __init__.py
    organizer_pi/
      __init__.py
      inputs.py
      event_trace.py
      cli.py
```

Tests will mirror those seams:

```text
code/tests/
  tracing/
    test_models.py
    test_openai_semantics.py
    test_phoenix_export.py
  experiments/
    organizer_pi/
      test_inputs.py
      test_event_trace.py
      test_cli.py
```

## Module interfaces

### `trec_rag.tracing`

This is the reusable module. Its interface consists of immutable trace records,
strict bundle serialization, OpenAI/OpenInference presentation helpers, Phoenix
configuration, bundle export, and export receipts. It must not import from
`trec_rag.experiments`.

`models.py` owns `SpanSpec`, `TraceSpec`, and `TraceBundle`, including strict
JSON reading and atomic writing. `phoenix_export.py` consumes those records and
exports them without knowing how a source run was captured. Generic OpenAI
message/envelope normalization stays in `openai_semantics.py`.

Source-specific names and payloads may remain inside opaque span attributes and
native JSON fields. The reusable module transports them but does not construct
or interpret organizer Pi events.

### `trec_rag.experiments.organizer_pi`

This module retains the runnable reproduction harness. It owns one-topic input
preparation, native Pi event parsing, organizer prompt reconstruction, Piika
tool schemas, conversion into reusable trace records, validation spans, and the
build/export command-line workflow.

The supported command becomes:

```bash
python -m trec_rag.experiments.organizer_pi.cli build ...
python -m trec_rag.experiments.organizer_pi.cli export ...
```

This code may import `trec_rag.tracing`; the reverse dependency is forbidden.

## Migration rules

- Move implementations rather than copying them. There will be one owner for
  every function and record.
- Do not retain compatibility modules at the old top-level paths. These modules
  have not landed on `master`, so aliases would add a second interface without
  protecting an established caller.
- Update all imports, tests, README commands, design records, and implementation
  plans that describe the current executable path.
- Preserve trace bundle schemas and hosted Phoenix rendering. This is a package
  organization change, not a trace-format migration.
- Preserve the Pi reproduction harness as experimental runnable code; do not
  turn captured private artifacts into tracked test data.

## Error handling and dependency direction

Existing strict JSON, size limits, atomic writes, secret scanning, ignored-path
checks, exporter flush behavior, and receipt semantics remain unchanged. Import
tests will enforce that `trec_rag.tracing` has no dependency on the experiment
subpackage.

The optional Phoenix libraries remain in the `observability` dependency group.
Importing models or OpenAI semantic helpers must not require Phoenix to be
installed; only exporter execution may require those libraries.

## Verification

1. Run the reorganized focused tracing and organizer Pi test suites.
2. Compile every moved Python module.
3. Assert that no file under `trec_rag/tracing` imports an experiment module or
   contains organizer event reconstruction.
4. Run the full repository test suite and distinguish pre-existing missing
   external-fixture failures from regressions.
5. Run `git diff --check` and scan tracked files for configured Phoenix
   credentials before updating the PR.

## Success criteria

- A reader finds reusable tracing functionality only under `trec_rag.tracing`.
- Pi-specific capture and reconstruction are clearly labeled experimental and
  live only under `trec_rag.experiments.organizer_pi`.
- The focused suite remains green and trace bundle/rendering behavior is
  unchanged.
- PR #29 describes the new package paths and does not imply Pi is a primary
  production path.

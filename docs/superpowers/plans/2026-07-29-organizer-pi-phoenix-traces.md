# Organizer Pi Phoenix Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the organizer-provided Piika agentic and fixed-retrieval Pi baselines for `rag2026-1`, preserve their native artifacts, and export two full-content traces to one hosted Phoenix project.

**Architecture:** Do not patch either organizer runtime. Convert their durable Pi JSONL event streams and validated run records into a small internal trace tree, reject credential leakage, then export the tree with explicit OpenInference span kinds through Phoenix's OpenTelemetry endpoint. A separate input-preparation module creates byte-faithful one-topic TSV and TREC files and verifies published hashes before paid calls.

**Tech Stack:** Python 3.12, pytest, `arize-phoenix-otel>=0.16.0`, `arize-phoenix-client`, OpenTelemetry/OpenInference, organizer Piika TypeScript runner, organizer `ragnarok_style_ag.py`, Pi CLI.

## Global Constraints

- Use topic `rag2026-1` for both paths.
- Use Piika revision `1a29d8fb4f71c9711e0cb435db26ae5649be8cf0` and the organizer's current `ragnarok_style_ag.py` bytes.
- Do not modify either organizer runtime.
- Capture full narrative, prompts, search queries/results, retrieved passages, opened documents, assistant output, citations, validation, timings, and errors.
- Send traces to Phoenix Cloud project `trec-rag-2026-pi-baselines`.
- Read credentials only from environment or ignored `.env`; never print or serialize them.
- Retain native Pi events as the audit source and treat Phoenix as a derived view.
- Keep generated inputs, cloned organizer repositories, run outputs, and trace receipts out of git.
- A failed baseline still produces an error-status trace from its partial native artifacts.
- Do not repeat a paid model call solely because Phoenix export fails.

---

### Task 1: Single-topic organizer input preparation

**Files:**
- Create: `code/trec_rag/organizer_pi_inputs.py`
- Create: `code/tests/test_organizer_pi_inputs.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: official two-column query TSV and six-column TREC run.
- Produces: `select_topic(query_path: Path, topic_id: str) -> OrganizerTopic`, `write_topic_tsv(topic: OrganizerTopic, path: Path) -> Path`, `write_topic_run(run_path: Path, topic_id: str, path: Path, *, expected_depth: int) -> Path`, and `sha256_file(path: Path) -> str`.
- `OrganizerTopic` is `@dataclass(frozen=True)` with `topic_id: str` and `narrative: str`.

- [ ] **Step 1: Write the failing input-contract tests**

```python
def test_select_topic_preserves_exact_narrative(tmp_path):
    source = tmp_path / "queries.tsv"
    source.write_text("rag2026-0\tfirst\nrag2026-1\tline one\tline two\n", encoding="utf-8")
    topic = select_topic(source, "rag2026-1")
    assert topic == OrganizerTopic("rag2026-1", "line one\tline two")


def test_write_topic_run_keeps_rank_order_and_exactly_expected_depth(tmp_path):
    source = tmp_path / "run.trec"
    source.write_text(
        "rag2026-1 Q0 d2 2 8.0 tag\n"
        "rag2026-0 Q0 x 1 9.0 tag\n"
        "rag2026-1 Q0 d1 1 9.0 tag\n",
        encoding="utf-8",
    )
    output = write_topic_run(source, "rag2026-1", tmp_path / "one.trec", expected_depth=2)
    assert output.read_text(encoding="utf-8").splitlines() == [
        "rag2026-1 Q0 d1 1 9.0 tag",
        "rag2026-1 Q0 d2 2 8.0 tag",
    ]
```

Also cover missing/duplicate topic IDs, duplicate/non-positive/non-contiguous ranks, duplicate docids, wrong depth, malformed columns, and atomic output replacement.

- [ ] **Step 2: Run the focused test and confirm the red state**

Run: `.venv/bin/python -m pytest code/tests/test_organizer_pi_inputs.py -q`

Expected: FAIL during import because `trec_rag.organizer_pi_inputs` does not exist.

- [ ] **Step 3: Implement the strict input module**

```python
@dataclass(frozen=True)
class OrganizerTopic:
    topic_id: str
    narrative: str


def select_topic(query_path: Path, topic_id: str) -> OrganizerTopic:
    matches = []
    with Path(query_path).open(encoding="utf-8", newline="") as source:
        for fields in csv.reader(source, delimiter="\t"):
            if fields and fields[0].strip() == topic_id:
                matches.append(OrganizerTopic(topic_id, "\t".join(fields[1:]).strip()))
    if len(matches) != 1 or not matches[0].narrative:
        raise ValueError(f"expected exactly one non-empty topic {topic_id}")
    return matches[0]
```

Write outputs through a same-directory temporary file and `Path.replace()`. For the run, validate every matching line before sorting by rank, require ranks `1..expected_depth`, require unique docids, and preserve the original six fields byte-for-byte apart from sorted line order.

- [ ] **Step 4: Run the input tests**

Run: `.venv/bin/python -m pytest code/tests/test_organizer_pi_inputs.py -q`

Expected: PASS.

- [ ] **Step 5: Add adjacent usage documentation**

Document that the module creates a one-row topic file for both organizer paths and a 100-row fixed-retrieval run, while source hashes are checked before filtering. State that it does not copy or transform document text.

- [ ] **Step 6: Commit the input-preparation unit**

```bash
git add code/trec_rag/organizer_pi_inputs.py code/tests/test_organizer_pi_inputs.py code/trec_rag/README.md
git commit -m "add organizer single-topic input preparation"
```

### Task 2: Native Pi event normalization and trace trees

**Files:**
- Create: `code/trec_rag/pi_trace_models.py`
- Create: `code/trec_rag/pi_event_trace.py`
- Create: `code/tests/test_pi_event_trace.py`

**Interfaces:**
- Consumes: native Pi JSONL events, normalized organizer run record, topic narrative, and for the fixed path the rendered system/user prompts plus ordered document records.
- Produces: immutable `SpanSpec` and `TraceBundle` values plus `build_piika_trace(...) -> TraceBundle`, `build_fixed_trace(...) -> TraceBundle`, `write_trace_bundle(...) -> Path`, and `read_trace_bundle(...) -> TraceBundle`.

```python
@dataclass(frozen=True)
class SpanSpec:
    name: str
    kind: str
    start_ns: int
    end_ns: int
    attributes: Mapping[str, AttributeValue]
    input_value: object | None
    output_value: object | None
    status: Literal["OK", "ERROR"]
    status_message: str | None = None
    children: tuple["SpanSpec", ...] = ()


@dataclass(frozen=True)
class TraceBundle:
    project_name: str
    session_id: str
    topic_id: str
    baseline: Literal["piika-agentic", "ragnarok-fixed"]
    root: SpanSpec
```

- [ ] **Step 1: Write representative failing event-conversion tests**

Build compact synthetic JSONL fixtures containing `message_start`, assistant content, `tool_execution_start`, `tool_execution_end`, final `message_end`, and a failed tool call. Assert:

```python
bundle = build_piika_trace(
    topic=OrganizerTopic("rag2026-1", "question"),
    events=load_pi_events(events_path),
    run_record={"status": "completed", "trec_rag_output": {"references": ["d1"], "answer": []}},
    session_id="rag2026-1-comparison",
)
assert bundle.root.kind == "CHAIN"
assert [child.kind for child in bundle.root.children] == ["LLM", "RETRIEVER", "TOOL", "LLM", "CHAIN"]
assert bundle.root.children[1].input_value["query"] == "biological grief heart"
assert bundle.root.children[2].output_value["text"] == "full document text"
```

For `build_fixed_trace`, assert one `RETRIEVER` span contains all 100 ordered full-text documents, one `LLM` span contains both full prompts and the final assistant payload, and validation is a terminal `CHAIN` child. Assert failed native events map to `ERROR` with their diagnostic text. Assert unknown events are retained under `pi.event.unknown_count` rather than crashing conversion.

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run: `.venv/bin/python -m pytest code/tests/test_pi_event_trace.py -q`

Expected: FAIL during import because the trace modules do not exist.

- [ ] **Step 3: Implement strict JSONL loading and immutable trace models**

`load_pi_events(path)` must reject blank-only files, non-object rows, duplicate JSON keys, non-finite values, and files over a configurable byte ceiling. Preserve event order. Use event timestamps when parseable; otherwise allocate monotonically increasing nanoseconds from the file modification time and set `trace.timing_reconstructed=True`. Implement strict, atomic JSON serialization for `TraceBundle` so the exact derived trace can be retained before network export and retried without another model call.

- [ ] **Step 4: Implement Piika event pairing**

Pair tool start/end events by native call ID. Map `search` to `RETRIEVER`, `read_document` to `TOOL`, and assistant turns to `LLM`. Put tool arguments in `input_value`, full tool result details in `output_value`, and native event types/IDs in attributes. Add a final validation child from the normalized run record.

- [ ] **Step 5: Implement fixed-path trace construction**

Build children in this order: fixed evidence `RETRIEVER`, prompt construction `CHAIN`, Pi generation `LLM`, and validation `CHAIN`. Keep ordered documents as `[{"rank": 1, "docid": "...", "text": "..."}]`; do not collapse or hash their text.

- [ ] **Step 6: Run trace-tree tests**

Run: `.venv/bin/python -m pytest code/tests/test_pi_event_trace.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the event-normalization unit**

```bash
git add code/trec_rag/pi_trace_models.py code/trec_rag/pi_event_trace.py code/tests/test_pi_event_trace.py
git commit -m "normalize organizer Pi events into trace trees"
```

### Task 3: Credential-safe Phoenix Cloud exporter

**Files:**
- Create: `code/trec_rag/phoenix_trace_export.py`
- Create: `code/tests/test_phoenix_trace_export.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `TraceBundle` and environment variables `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT`, and `PHOENIX_PROJECT_NAME`.
- Produces: `PhoenixSettings.from_env()`, `assert_no_secrets(bundle, forbidden_values) -> None`, and `export_trace(bundle, settings, *, provider_factory=None) -> ExportReceipt`.

```python
@dataclass(frozen=True)
class PhoenixSettings:
    api_key: SecretStr
    collector_endpoint: str
    project_name: str


@dataclass(frozen=True)
class ExportReceipt:
    project_name: str
    trace_id: str
    root_span_id: str
    exported_span_count: int
```

- [ ] **Step 1: Write failing settings, secret-scan, and exporter tests**

Assert that missing keys fail closed, the browser workspace URL `https://app.phoenix.arize.com/s/npatta01` normalizes to the collector base `https://app.phoenix.arize.com`, non-HTTPS hosted endpoints are rejected, and `SecretStr.__repr__` never returns its value. Test that a forbidden token nested in any attribute/input/output raises before provider construction.

Use a fake tracer/provider to assert recursive parent-child relationships, explicit start/end times, OpenInference kinds, `session.id`, topic/baseline attributes, status mapping, provider `force_flush()`, and `shutdown()` even on export error.

- [ ] **Step 2: Run the exporter tests and confirm the red state**

Run: `.venv/bin/python -m pytest code/tests/test_phoenix_trace_export.py -q`

Expected: FAIL during import because `trec_rag.phoenix_trace_export` does not exist.

- [ ] **Step 3: Add the observability dependency group**

```toml
observability = [
    "arize-phoenix-client",
    "arize-phoenix-otel>=0.16.0",
]
```

Run: `uv lock && uv sync --group dev --group observability`

Expected: the lockfile resolves and `.venv/bin/python -c 'import phoenix.otel'` exits 0.

- [ ] **Step 4: Implement fail-closed settings and recursive secret inspection**

Normalize only the known Phoenix Cloud workspace suffix `/s/<workspace>` to the documented cloud base, remove a trailing `/v1/traces` before passing the base to `phoenix.otel.register`, and reject query strings or fragments. Traverse dataclasses, mappings, sequences, and strings; compare every configured credential value of length at least eight against serialized trace content.

- [ ] **Step 5: Implement recursive OpenTelemetry export**

Register with `project_name=settings.project_name`, `endpoint=settings.collector_endpoint`, `api_key=settings.api_key.reveal()`, and `batch=False`. Use `trace.set_span_in_context(parent_span)` for children, `openinference.span.kind` for each `SpanSpec.kind`, JSON MIME types for structured input/output, and explicit start/end nanoseconds. End every span in `finally`, force-flush, then shut down the provider.

- [ ] **Step 6: Run exporter tests and the combined focused suite**

Run: `.venv/bin/python -m pytest code/tests/test_phoenix_trace_export.py code/tests/test_pi_event_trace.py code/tests/test_organizer_pi_inputs.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the Phoenix exporter unit**

```bash
git add code/trec_rag/phoenix_trace_export.py code/tests/test_phoenix_trace_export.py pyproject.toml uv.lock
git commit -m "export organizer Pi traces to Phoenix Cloud"
```

### Task 4: Reusable trace import CLI

**Files:**
- Create: `code/trec_rag/organizer_pi_trace.py`
- Create: `code/tests/test_organizer_pi_trace.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: `--baseline`, `--topic-tsv`, native events, normalized output record, and baseline-specific full-content inputs.
- Produces: `python -m trec_rag.organizer_pi_trace build ...`, a local strict-JSON `TraceBundle`; and `python -m trec_rag.organizer_pi_trace export --bundle ...`, a JSON receipt written only after successful Phoenix flush.

- [ ] **Step 1: Write failing CLI contract tests**

Use injected loaders/exporters to assert:

```python
exit_code = main([
    "build", "--baseline", "piika-agentic",
    "--topic", "rag2026-1", "--topic-tsv", str(topics),
    "--events", str(events), "--record", str(record),
    "--bundle", str(bundle_path),
])
assert exit_code == 0

exit_code = main([
    "export", "--bundle", str(bundle_path), "--receipt", str(receipt),
], exporter=fake_exporter)
assert json.loads(receipt.read_text())["trace_id"] == "abc123"
```

Cover mutually required fixed-path arguments (`--ranked-run`, `--documents`, `--organizer-script`), topic mismatch, output-record mismatch, non-ignored bundle/receipt targets, preserved bundle on export failure, no receipt on export failure, and secret-free stderr.

- [ ] **Step 2: Run the CLI tests and confirm the red state**

Run: `.venv/bin/python -m pytest code/tests/test_organizer_pi_trace.py -q`

Expected: FAIL during import because the CLI module does not exist.

- [ ] **Step 3: Implement the CLI and atomic receipt**

Load repository `.env` with the existing `load_repo_env()` helper. `build` constructs and atomically saves the selected `TraceBundle` without requiring Phoenix credentials. `export` reloads the bundle, calls `assert_no_secrets` with credential values, exports, and writes only public receipt fields. For fixed retrieval, stream the ZIP and retain only the 100 docids from the filtered run, independently truncating each to the organizer's 1,000-word rule. Load the unmodified organizer script with `importlib.util.spec_from_file_location`, use its `SYSTEM_PROMPT` and `prompt()` to reconstruct the exact rendered prompts, and compare the rendered user prompt with the native Pi user-message event when that event is present.

- [ ] **Step 4: Add exact commands and data warnings to README**

Document both CLI forms, the Phoenix project name, full-content transmission, ignored output requirements, and the rule that native events are authoritative. Do not include a real trace ID, API key, or workspace-specific URL.

- [ ] **Step 5: Run the CLI and regression tests**

Run: `.venv/bin/python -m pytest code/tests/test_organizer_pi_trace.py code/tests/test_phoenix_trace_export.py code/tests/test_pi_event_trace.py code/tests/test_organizer_pi_inputs.py code/tests/test_remote_pyserini.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the import CLI unit**

```bash
git add code/trec_rag/organizer_pi_trace.py code/tests/test_organizer_pi_trace.py code/trec_rag/README.md
git commit -m "add organizer Pi trace import CLI"
```

### Task 5: Pin, preflight, and run both organizer paths for `rag2026-1`

**Files:**
- Create ignored artifacts under: `outputs/organizer-pi-phoenix/rag2026-1/`
- Do not modify tracked source files.

**Interfaces:**
- Consumes: Tasks 1-4, organizer inputs, Pi and Pyserini authentication, Phoenix authentication.
- Produces: two native run directories, two export receipts, and two hosted Phoenix traces sharing session `rag2026-1-comparison`.

- [ ] **Step 1: Perform cheap environment and secret preflight**

Run checks that print presence only for `PYSERINI_API_TOKEN` and `PHOENIX_API_KEY`; run `pi --version`, `node --version`, `npm --version`, and `.venv/bin/python --version`. Confirm `git check-ignore outputs/organizer-pi-phoenix/probe` succeeds. Do not run with `set -x`.

Expected: both credentials present, Pi authenticated, Node/npm available, Python 3.12, and output root ignored.

- [ ] **Step 2: Fetch exact organizer sources into the ignored output root**

Clone `nourj98/piika`, detach at `1a29d8fb4f71c9711e0cb435db26ae5649be8cf0`, verify `git rev-parse HEAD`, and run `npm ci`. Clone current `TREC-RAG/trec-rag-data`, record its commit, initialize Git LFS content, and verify the organizer-published SHA-256 values:

```text
queries: 72dc2fd358d3eeda973397ccd7a8775545b19a6deaefc67709167eee6a9f8a2c
ranked run: fff3b9a68cf8325dfc092237123a9fd2255b8135401e08c69b20b5ed312a5546
documents: cebbeb313065572ad69aaf3c9f311546bb09cf6d2085fc0160c81f9e6627c35c
```

Stop before any model call if a hash differs.

- [ ] **Step 3: Create and validate one-topic inputs**

Use Task 1 to write `inputs/rag2026-1.tsv` and `inputs/rag2026-1.top100.trec`. Assert one TSV row, 100 TREC rows, ranks 1-100, and 100 unique docids. Stream the document archive once to confirm all 100 texts exist.

- [ ] **Step 4: Authenticate Phoenix without exporting content**

Use `arize-phoenix-client` against the base Cloud endpoint to list or resolve project access, logging only HTTP status and account-safe project metadata. Do not put the API key in argv or output.

Expected: authenticated request succeeds before either paid run.

- [ ] **Step 5: Run Piika agentic search on the one-row TSV**

Use the organizer's `run:benchmark:query-set:sharded-shared-bm25` entry point with `--query-file` set to the one-row TSV, `--shard-count 1`, the documented model/thinking/output metadata, authenticated Pyserini REST configuration, and a dedicated output root. Preserve `merged/raw-events/rag2026-1.jsonl`, stderr, the normalized record, and final JSONL.

Expected: one validated `rag2026-1` record and non-empty native events. If it fails, retain partial events and continue to trace export with error status.

- [ ] **Step 6: Run fixed-retrieval generation on the same topic**

Invoke the organizer's unmodified `ragnarok_style_ag.py` with the one-row TSV, filtered 100-row run, published document ZIP, `--top-k 100`, `--max-document-words 1000`, `--model openai-codex/gpt-5.6-sol`, `--thinking medium`, `--concurrency 1`, documented metadata, and `--overwrite`.

Expected: one validated answer row plus `raw/rag2026-1.events.jsonl`. If validation fails, retain raw events and export an error trace.

- [ ] **Step 7: Export both traces to Phoenix Cloud**

Run the Task 4 `build` command once per baseline to save two local trace bundles, then run `export` for each bundle with `PHOENIX_PROJECT_NAME=trec-rag-2026-pi-baselines` and shared session `rag2026-1-comparison`. Never place the API key on the command line. Save public receipts under each native run directory.

Expected: two different non-empty trace IDs, both with successful flush receipts.

- [ ] **Step 8: Verify hosted traces and native parity**

Query Phoenix by project, session, topic, and baseline. Assert exactly two root traces; verify Piika has `RETRIEVER`, `TOOL`, `LLM`, and validation children; verify fixed has 100 full-text documents, complete prompts, `LLM`, and validation children. Compare trace final answers and reference arrays byte-for-byte with native normalized records. Scan tracked files, ignored receipts, and serialized spans for the configured credential values.

- [ ] **Step 9: Record operational evidence without secrets**

Write `outputs/organizer-pi-phoenix/rag2026-1/verification.json` containing source commits, published input hashes, native artifact hashes, trace IDs, span counts, statuses, and parity results. Exclude prompts/passages because Phoenix and native events already hold them, and exclude all headers and credentials.

### Task 6: Final verification and review

**Files:**
- Inspect all files changed by Tasks 1-4.
- Inspect ignored verification artifacts from Task 5.

**Interfaces:**
- Consumes: completed implementation and run evidence.
- Produces: reviewed, tested code and a user handoff with hosted Phoenix links/IDs.

- [ ] **Step 1: Run formatting-neutral static checks**

Run: `git diff --check HEAD~4..HEAD`

Expected: no whitespace errors.

- [ ] **Step 2: Run the full Python test suite**

Run: `.venv/bin/python -m pytest code/tests -q`

Expected: PASS. If unrelated environment-dependent tests cannot run, record each exact test and reason while keeping all new focused tests passing.

- [ ] **Step 3: Run a final secret scan**

Search tracked files and generated public receipts for the exact configured token values via an in-memory scanner that prints only file paths on a match. Also search for credential-shaped `Authorization`, `api_key`, and JWT fields. Delete and regenerate any contaminated ignored artifact; amend tracked files before handoff.

Expected: zero matches.

- [ ] **Step 4: Request independent code review**

Use `superpowers:requesting-code-review` against the implementation commits, with special attention to event fidelity, span parenting/timing, full-content completeness, secret handling, and paid-call non-repetition.

- [ ] **Step 5: Address findings and rerun focused verification**

For each accepted finding, add a regression test, observe failure, implement the fix, and rerun the affected focused suite plus the final secret scan.

- [ ] **Step 6: Commit final review fixes if needed**

```bash
git add code pyproject.toml uv.lock
git commit -m "verify organizer Pi Phoenix traces"
```

- [ ] **Step 7: Hand off the comparison**

Report both run statuses, native artifact paths, Phoenix project/session, trace links or trace IDs, span counts, input/source hashes, test results, and any remaining limitation. Do not include credential values or full corpus text in the handoff.

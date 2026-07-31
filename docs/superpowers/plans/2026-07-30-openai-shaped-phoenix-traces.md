# OpenAI-shaped Phoenix Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-export the two preserved `rag2026-1` runs with the same message, tool, request, response, provider, and invocation conventions as OpenAI auto-instrumentation.

**Architecture:** Add a pure translation module between immutable Pi `SpanSpec` payloads and OpenAI/OpenInference attributes. The Phoenix exporter delegates LLM payload and semantic construction to that module; the event normalizer only declares trace topology and authoritative tool availability. A new hosted session is built and exported only after offline red/green tests and review.

**Tech Stack:** Python 3.12, OpenInference semantic conventions, OpenTelemetry/Phoenix OTLP, pytest, Phoenix Python client.

## Global Constraints

- Keep exactly two traces: one Piika agent trace and one fixed-retrieval agent trace.
- Keep one LLM span per observed provider response and no presentation-only spans.
- Do not rerun the organizer, Pi, retrieval, or any model.
- Preserve full captured content and native payloads; never fabricate the missing Piika system prompt.
- Keep credentials in the ignored environment and out of commands, receipts, logs, and verification artifacts.
- Leave all previously hosted traces intact.

---

### Task 1: OpenAI/OpenInference semantic adapter

**Files:**
- Create: `code/trec_rag/tracing/openai_semantics.py`
- Create: `code/tests/tracing/test_openai_semantics.py`
- Modify: `code/trec_rag/experiments/organizer_pi/event_trace.py`
- Modify: `code/trec_rag/tracing/phoenix_export.py`
- Modify: `code/tests/experiments/organizer_pi/test_event_trace.py`
- Modify: `code/tests/tracing/test_phoenix_export.py`

**Interfaces:**
- Consumes: immutable `SpanSpec` input/output values produced by `trec_rag.experiments.organizer_pi.event_trace`.
- Produces: `piika_tool_schemas() -> tuple[Mapping[str, object], ...]`, `openai_request_envelope(spec: SpanSpec) -> Mapping[str, object]`, `openai_response_envelope(spec: SpanSpec) -> Mapping[str, object]`, and `openai_llm_attributes(spec: SpanSpec) -> Mapping[str, object]`.
- `phoenix_trace_export._export_span` uses OpenAI envelopes for LLM `input.value`/`output.value` and keeps strict JSON copies under `pi.native.input_json` and `pi.native.output_json`.

- [ ] **Step 1: Write failing message/tool compatibility tests**

Create literal fixtures resembling the saved Pi messages:

```python
def test_pi_tool_turn_matches_openai_message_and_tool_result_contract():
    spec = llm_span(
        input_value={"messages": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{
                "type": "toolCall", "id": "call-1", "name": "search",
                "arguments": {"reason": "find evidence", "query": "evidence", "hits": 8},
            }]},
            {"role": "toolResult", "toolCallId": "call-1", "content": [
                {"type": "text", "text": "ranked results"},
            ]},
        ], "tools": list(piika_tool_schemas())},
        output_value=assistant_payload_with_tool_call(),
    )
    attrs = openai_llm_attributes(spec)
    assert attrs["llm.system"] == "openai"
    assert attrs["llm.provider"] == "openai"
    assert attrs["llm.input_messages.1.message.tool_calls.0.tool_call.id"] == "call-1"
    assert attrs["llm.input_messages.2.message.role"] == "tool"
    assert attrs["llm.input_messages.2.message.tool_call_id"] == "call-1"
    assert attrs["llm.input_messages.2.message.content"] == "ranked results"
    assert json.loads(attrs["llm.tools.0.tool.json_schema"])["function"]["name"] == "search"
    assert json.loads(attrs["llm.tools.1.tool.json_schema"])["function"]["name"] == "read_document"
```

Add separate tests for parallel tool calls, reasoning plus tool calls, scalar tool-result content, recognized provider identity, known-only invocation parameters, original-provider provenance, and the absence of a fabricated Piika system message.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/tracing/test_openai_semantics.py -q
```

Expected: collection or assertion failures because the adapter and OpenAI-shaped attributes do not exist.

- [ ] **Step 3: Write failing generic request/response envelope tests**

Assert hand-written OpenAI-style shapes:

```python
request = openai_request_envelope(spec)
assert request == {
    "model": "gpt-5.6-sol",
    "messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "search", "arguments": '{"hits":8,"query":"evidence","reason":"find evidence"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ranked results"},
    ],
    "tools": list(piika_tool_schemas()),
}
response = openai_response_envelope(spec)
assert response["object"] == "chat.completion"
assert response["choices"][0]["message"]["role"] == "assistant"
assert response["choices"][0]["finish_reason"] == "toolUse"
```

For the fixed fixture, assert exactly two request messages containing the byte-exact system and user prompts and no tools. Assert native envelopes remain available as namespaced JSON attributes.

- [ ] **Step 4: Run envelope tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/tracing/test_openai_semantics.py -q
```

Expected: failures specifically naming missing OpenAI request/response envelopes.

- [ ] **Step 5: Implement the pure adapter**

Implement these boundaries in `trec_rag.tracing.openai_semantics`:

```python
def piika_tool_schemas() -> tuple[Mapping[str, object], ...]: ...
def normalize_openai_message(message: Mapping[str, object]) -> dict[str, object]: ...
def openai_request_envelope(spec: SpanSpec) -> Mapping[str, object]: ...
def openai_response_envelope(spec: SpanSpec) -> Mapping[str, object]: ...
def openai_llm_attributes(spec: SpanSpec) -> Mapping[str, object]: ...
```

The schemas must encode the organizer source contracts for this captured `pyserini-rest-2tool` run: `search` requires `reason` and `query`, with optional numeric `hits`; paginated `read_document` requires `reason` and `docid`, with optional numeric `offset`/`limit`. Preserve the source descriptions and required lists faithfully. Normalize only captured values. Join text-only tool-result parts into scalar `message.content`; retain ordered reasoning in OpenInference content attributes; keep tool calls in `message.tool_calls` with canonical JSON argument strings.

- [ ] **Step 6: Connect trace topology and exporter**

In `trec_rag.experiments.organizer_pi.event_trace`, change the two baseline roots from `CHAIN` to `AGENT`, attach `piika_tool_schemas()` to each Pi LLM input, and record `pi.system_prompt.captured=False`. Do not add a system message.

In `trec_rag.tracing.phoenix_export`, delegate LLM semantic attributes to `openai_llm_attributes`. For LLM spans only, serialize `openai_request_envelope` and `openai_response_envelope` into generic input/output fields, while retaining the original strict JSON under `pi.native.input_json` and `pi.native.output_json`. Non-LLM payload behavior remains unchanged.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/tracing/test_openai_semantics.py \
  code/tests/tracing/test_phoenix_export.py \
  code/tests/experiments/organizer_pi/test_event_trace.py \
  code/tests/experiments/organizer_pi/test_cli.py -q
```

Expected: all tests pass with no warnings.

- [ ] **Step 8: Inspect preserved captures offline**

Rebuild both bundles to temporary ignored paths and assert: Pi has one AGENT root, 10 LLM spans, 13 retriever spans, 13 tool spans, linked tool-call/result IDs, two advertised tool schemas per LLM, and no fabricated system message; fixed has one AGENT root, one LLM span, exact prompts, and 100 documents. Assert the immutable source-event hashes remain stable and every original per-span native payload is retained byte-for-byte under its namespaced attribute; OpenAI-shaped bundle/span hashes are expected to change.

- [ ] **Step 9: Commit**

```bash
git add code/trec_rag/tracing/openai_semantics.py code/tests/tracing/test_openai_semantics.py \
  code/trec_rag/experiments/organizer_pi/event_trace.py code/trec_rag/tracing/phoenix_export.py \
  code/tests/experiments/organizer_pi/test_event_trace.py code/tests/tracing/test_phoenix_export.py
git commit -m "match OpenAI Phoenix trace semantics"
```

---

### Task 2: Hosted v3 export and verification

**Files:**
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/piika/rag2026-1.openai-v3.trace.json`
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/fixed/rag2026-1.openai-v3.trace.json`
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/*/phoenix-receipt.openai-v3.json`
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/verification.openai-v3.json`
- Update ignored: `.superpowers/sdd/2026-07-29-organizer-pi-phoenix-traces/task-5-report.md`

**Interfaces:**
- Consumes: reviewed Task 1 exporter, preserved native events/records, ignored Phoenix environment.
- Produces: two public hosted trace IDs in session `rag2026-1-comparison-openai-v3` plus sanitized hashes and per-span schema evidence.

- [ ] **Step 1: Independent code review**

Review Task 1 against the design and official OpenAI request/response extractors. Treat fabricated messages, broken tool-call correlation, duplicated LLM cost, lost native content, or credential exposure as blockers.

- [ ] **Step 2: Rebuild without executing organizers/models**

Run the existing `organizer_pi_trace build` command once for each preserved event/record pair, targeting the v3 bundle paths and session `rag2026-1-comparison-openai-v3`.

- [ ] **Step 3: Offline schema gate**

Load both bundles and run the adapter. Assert literal provider/system values, exact message roles, tool schemas, tool-call/result linkage, model/usage fields, root/span counts, timings, 100 fixed documents, exact fixed prompts, and native payload preservation.

- [ ] **Step 4: Export saved bundles only**

Set only `PHOENIX_PROJECT_NAME=trec-rag-2026-pi-baselines` on the process; load endpoint/key from ignored `.env`. Export the two saved v3 bundles and write public receipts. Do not invoke Pi or a model.

- [ ] **Step 5: Query Phoenix and verify renderer inputs**

Query project/session through the Phoenix client. Assert exactly two root traces and 44 hosted spans. For every LLM span assert `llm.system=openai`, `llm.provider=openai`, structured input/output messages, invocation parameters, and—on Pi—two `llm.tools` schemas. Verify every assistant tool-call ID has a corresponding role-tool input message and tool span ID. Verify fixed retains one LLM and 100 document IDs/contents.

- [ ] **Step 6: Verify payload, timing, and secret safety**

Compare generic OpenAI request/response envelopes and namespaced native payloads with the local adapter byte-for-byte. Re-run focused tests, `py_compile`, `git diff --check`, direct trace-link HTTP checks, and configured-secret scans over tracked and generated task artifacts.

- [ ] **Step 7: Write sanitized evidence and hand off**

Write query timestamp, public trace/root/span IDs, counts, latency, bundle hashes, per-span semantic/native payload hashes, and secret-scan manifest to `verification.openai-v3.json`. Update the operational report and provide direct Phoenix redirect URLs. Do not include prompts, passages, headers, or credentials.

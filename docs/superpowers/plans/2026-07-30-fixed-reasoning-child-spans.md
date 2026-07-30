# Fixed Reasoning Child Spans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three visible, non-cost-bearing reasoning child spans to the saved fixed-retrieval generation while retaining one real LLM call and the complete native response.

**Architecture:** Fixed trace construction extracts only native `thinking` items from the final captured assistant response and creates deterministic CHAIN children inside the existing LLM span. The OpenAI semantic adapter uses an explicit span attribute to omit those extracted blocks from the rich/generic LLM presentation while the exporter continues retaining the unchanged native response. A reviewed saved-artifact rebuild produces and exports one additional eight-span fixed trace.

**Tech Stack:** Python 3.12, immutable `SpanSpec` trace trees, OpenInference semantic conventions, OpenTelemetry/Phoenix OTLP, pytest, Phoenix Python client.

## Global Constraints

- Preserve exactly one cost-bearing LLM span and one provider response in the revised fixed trace.
- Create exactly three reasoning children for the three captured native `thinking` items; do not split the seven headings.
- Preserve each thinking string byte-for-byte and in native order.
- Preserve the complete original response byte-for-byte under `pi.native.output_json`.
- Keep prompts, 100 documents, final answer, model, usage, cost, finish reason, and validation unchanged.
- Mark child timing as reconstructed equal partitions contained within the LLM interval.
- Apply the behavior only to fixed trace construction; do not change the Pi agentic trace.
- Do not invoke the organizer runtime, retrieval, Pi, or a model.
- Export one additional fixed trace; leave every existing hosted trace untouched.
- Keep credentials in ignored environment files and out of commands, receipts, logs, and tracked files.

---

### Task 1: Fixed reasoning extraction and presentation

**Files:**
- Modify: `code/trec_rag/pi_event_trace.py`
- Modify: `code/trec_rag/openai_trace_semantics.py`
- Modify: `code/tests/test_pi_event_trace.py`
- Modify: `code/tests/test_openai_trace_semantics.py`
- Modify: `code/tests/test_phoenix_trace_export.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: a fixed generation `SpanSpec` whose `output_value` is the unchanged captured assistant mapping and whose LLM start/end nanoseconds are known.
- Produces: `_reasoning_summary_children(message: object, *, start_ns: int, end_ns: int) -> tuple[SpanSpec, ...]` in `pi_event_trace.py` and presentation filtering keyed by `pi.reasoning.extracted_to_children=True` in `openai_trace_semantics.py`.
- Each reasoning child has `name="Reasoning summary <index>"`, `kind="CHAIN"`, exact string `output_value`, and attributes `pi.reasoning.block.index`, `pi.reasoning.block.count`, `pi.reasoning.source`, `pi.reasoning.timing_method`, and `trace.timing_reconstructed`.

- [ ] **Step 1: Write failing trace-construction tests**

Add a literal fixed assistant fixture with three native thinking items and one text item. Assert:

```python
def _walk_spans(span):
    yield span
    for child in span.children:
        yield from _walk_spans(child)

generation = bundle.root.children[2]
assert generation.kind == "LLM"
assert generation.attributes["pi.reasoning.extracted_to_children"] is True
assert generation.attributes["pi.reasoning.block_count"] == 3
assert [child.name for child in generation.children] == [
    "Reasoning summary 1",
    "Reasoning summary 2",
    "Reasoning summary 3",
]
assert [child.output_value for child in generation.children] == [
    "first native block",
    "second native block",
    "third native block",
]
assert all(child.kind == "CHAIN" for child in generation.children)
assert all(child.attributes["trace.timing_reconstructed"] is True for child in generation.children)
assert all(child.attributes["pi.reasoning.timing_method"] == "equal-partition" for child in generation.children)
assert generation.children[0].start_ns == generation.start_ns
assert generation.children[-1].end_ns == generation.end_ns
assert all(
    left.end_ns == right.start_ns
    for left, right in zip(generation.children, generation.children[1:], strict=True)
)
assert sum(span.kind == "LLM" for span in _walk_spans(bundle.root)) == 1
assert sum(1 for _ in _walk_spans(bundle.root)) == 8
```

Add a second test whose final message contains only text and assert no reasoning children, no extraction attribute, and the existing five-span trace.

- [ ] **Step 2: Run construction tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_pi_event_trace.py::test_fixed_trace_extracts_native_reasoning_into_three_children \
  code/tests/test_pi_event_trace.py::test_fixed_trace_without_thinking_keeps_existing_topology -q
```

Expected: assertions fail because fixed generation currently has no reasoning children or extraction attributes.

- [ ] **Step 3: Write failing semantic-presentation tests**

Create an LLM `SpanSpec` with `pi.reasoning.extracted_to_children=True` and native content containing thinking/text blocks. Assert:

```python
attrs = openai_llm_attributes(spec)
assert attrs["llm.output_messages.0.message.content"] == "final answer"
assert not any("message_content.type" in key and value == "reasoning" for key, value in attrs.items())
response = openai_response_envelope(spec)
assert response["choices"][0]["message"] == {
    "role": "assistant",
    "content": "final answer",
}
```

Extend the exporter test to assert `pi.native.output_json` still equals strict JSON for the complete response, including all thinking items, while `output.value` contains only the final assistant content.

- [ ] **Step 4: Run semantic/export tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_openai_trace_semantics.py::test_extracted_reasoning_is_omitted_only_from_llm_presentation \
  code/tests/test_phoenix_trace_export.py::test_export_separates_reasoning_children_without_changing_native_llm_payload -q
```

Expected: failures show thinking content remains in the rich attributes or the expected reasoning children do not exist.

- [ ] **Step 5: Implement deterministic reasoning child construction**

In `pi_event_trace.py`, implement `_reasoning_summary_children`. Accept only non-empty string values from ordered content mappings whose type is `thinking`. Compute `base_duration, remainder = divmod(end_ns - start_ns, count)` and create consecutive child intervals; add the remainder to the final child. Return no children for a non-mapping message, missing content, or zero captured thinking blocks.

In `build_fixed_trace`, construct the final LLM start/end first, create reasoning children from `generation_message`, append them after any observed attempt/failure children, and add these LLM attributes only when reasoning children exist:

```python
"pi.reasoning.extracted_to_children": True,
"pi.reasoning.block_count": len(reasoning_children),
```

Keep `generation.output_value=generation_message` unchanged.

- [ ] **Step 6: Implement presentation-only reasoning filtering**

In `openai_trace_semantics.py`, add a private helper that shallow-copies the output mapping and removes only content items of type `thinking` or `reasoning` when `spec.attributes.get("pi.reasoning.extracted_to_children") is True`. Use that presented mapping in both `openai_response_envelope` and the output-message branch of `openai_llm_attributes`. Do not mutate `SpanSpec.output_value`; model, provider, response ID, stop reason, usage, text, and tool calls remain available.

- [ ] **Step 7: Document the behavior**

Add a concise README paragraph stating that fixed traces with captured native thinking expose one CHAIN child per native block, retain one LLM/cost span, use reconstructed equal-partition timing, omit extracted reasoning from the LLM presentation to avoid duplication, and preserve the complete response under `pi.native.output_json`.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_openai_trace_semantics.py \
  code/tests/test_phoenix_trace_export.py \
  code/tests/test_pi_event_trace.py \
  code/tests/test_organizer_pi_trace.py -q
.venv/bin/python -m py_compile \
  code/trec_rag/openai_trace_semantics.py \
  code/trec_rag/pi_event_trace.py \
  code/trec_rag/phoenix_trace_export.py \
  code/trec_rag/organizer_pi_trace.py
git diff --check
```

Expected: all focused tests pass, compilation exits zero, and `git diff --check` emits no output.

- [ ] **Step 9: Rebuild the saved fixed bundle offline and inspect it**

Build an ignored bundle at `outputs/organizer-pi-phoenix/rag2026-1/fixed/rag2026-1.reasoning-v4.trace.json` using the existing fixed build command and session `rag2026-1-comparison-openai-v3`. Assert eight spans, one LLM, three direct CHAIN children named in order, exact thinking block hashes, contained timing, 100 documents, exact prompts, unchanged final text/usage/cost/validation, presentation-only final text, and byte-for-byte native output retention. Do not export in this task.

- [ ] **Step 10: Commit**

```bash
git add code/trec_rag/pi_event_trace.py code/trec_rag/openai_trace_semantics.py \
  code/tests/test_pi_event_trace.py code/tests/test_openai_trace_semantics.py \
  code/tests/test_phoenix_trace_export.py code/trec_rag/README.md
git commit -m "separate fixed reasoning into child spans"
```

---

### Task 2: Review, hosted export, and live verification

**Files:**
- Consume ignored: `outputs/organizer-pi-phoenix/rag2026-1/fixed/rag2026-1.reasoning-v4.trace.json`
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/fixed/phoenix-receipt.reasoning-v4.json`
- Create ignored: `outputs/organizer-pi-phoenix/rag2026-1/verification.reasoning-v4.json`

**Interfaces:**
- Consumes: the reviewed eight-span saved fixed bundle from Task 1 and Phoenix endpoint/key from ignored `.env`.
- Produces: one additional hosted fixed trace ID and sanitized evidence proving its eight-span topology, single LLM cost, three reasoning children, final-only LLM presentation, and complete native response.

- [ ] **Step 1: Independently review Task 1**

Review the complete Task 1 diff against the design. Treat altered native content, duplicated LLM usage/cost, fabricated timestamps without reconstruction flags, reasoning order loss, Pi-agentic changes, or credentials as blockers.

- [ ] **Step 2: Run the final offline gate**

Load the saved v4 bundle and assert all Task 1 Step 9 conditions. Compare its source event/record hashes with the previously verified immutable artifacts. Scan tracked files and the bundle for configured secret values without printing them.

- [ ] **Step 3: Export the saved bundle exactly once**

Load Phoenix settings from ignored `.env`, set only `PHOENIX_PROJECT_NAME=trec-rag-2026-pi-baselines` in the process, and export the v4 bundle to `phoenix-receipt.reasoning-v4.json`. Do not invoke build, organizer, retrieval, Pi, or model commands during this step. Leave existing traces untouched.

- [ ] **Step 4: Query the new trace by receipt ID**

Use the Phoenix client and internal project ID resolved from the project name. Assert the receipt identifies exactly eight exported spans and the hosted trace contains:

```text
1 AGENT root
1 RETRIEVER evidence span
2 top-level CHAIN spans (prompt and validation)
1 LLM generation span
3 direct CHAIN reasoning children
```

Assert the three children have the exact parent LLM span ID, ordered indexes, exact output hashes, reconstruction attributes, and no LLM token/cost attributes. Assert the LLM has recognized OpenAI provider/system attributes, final-only rich/generic output, unchanged usage/cost, and complete `pi.native.output_json`.

- [ ] **Step 5: Verify safety and write evidence**

Check the corrected direct Phoenix route using the internal project ID, not the project name. Write only trace/root/span IDs, counts, hashes, timestamps, semantic assertions, and secret-scan results to `verification.reasoning-v4.json`; exclude prompts, reasoning text, document text, headers, and credentials.

- [ ] **Step 6: Final focused verification and handoff**

Run the Task 1 focused test/compile/diff-check commands fresh. Report the new corrected Phoenix link, explicitly state that existing traces were not deleted, and note that the three child durations are reconstructed equal partitions rather than native timing.

# OpenAI-shaped Phoenix traces

## Objective

Re-export the preserved `rag2026-1` Piika and fixed-retrieval runs so Phoenix
receives the same OpenInference shape produced by OpenAI client
auto-instrumentation. Do not rerun either organizer, retrieval, Pi, or model.

## Trace topology

Keep exactly two traces in a new comparison session:

- Piika: one `AGENT` root containing the 10 observed LLM calls and the observed
  search/read-document operations in event order.
- Fixed retrieval: one `AGENT` root containing the 100-document retriever and
  exactly one observed LLM call.

Do not add presentation-only traces or duplicate LLM spans. Existing validation
and prompt-construction spans remain only where they represent real pipeline
work.

## OpenAI-compatible LLM representation

For every LLM span:

- Set `openinference.span.kind=LLM`, `llm.system=openai`, and
  `llm.provider=openai`.
- Preserve the original Pi provider under a namespaced provenance attribute.
- Emit `llm.model_name`, finish reason, token counts, cache details, and costs.
- Emit known request parameters through `llm.invocation_parameters`; never
  invent unknown sampling parameters.
- Emit observed input and output messages using indexed OpenInference message
  attributes.
- Normalize assistant tool requests to OpenAI `tool_calls`, including ID,
  function name, and JSON arguments.
- Normalize tool results to `role=tool`, scalar text content, and the matching
  `tool_call_id`.
- Emit the authoritative search/read-document JSON schemas through
  `llm.tools.<index>.tool.json_schema` on Piika calls where those tools were
  available.
- Use OpenAI-style request and response envelopes for generic `input.value` and
  `output.value`. Retain the complete native Pi payload in namespaced JSON
  attributes so no captured content is lost.

The fixed call includes its exact captured system and user prompts. The saved
Piika run did not persist its system prompt because prompt dumping was disabled;
the re-export must not fabricate it. It will include every observed user,
assistant, and tool message and record the system-prompt capture limitation as
provenance.

## Rendering contract

Phoenix should receive the fields its current renderer uses for the Messages,
Tools, invocation-parameter, usage, and provider views. Success is determined by
querying the hosted spans and verifying:

- all LLM spans have recognized OpenAI system/provider values;
- every Pi assistant tool call is linked to its tool result;
- Pi LLM spans advertise both authoritative tool schemas;
- fixed has one LLM span with exact system/user content;
- message, tool, usage, timing, and full generic/native payload parity holds;
- trace/span counts remain 2 roots and 39/5 spans unless a test-backed removal
  of a duplicate real-work span is required.

## Testing and safety

Use red-green TDD against the exporter and event normalizer. Add fixtures that
match OpenAI instrumentor request/response output, parallel tool calls, linked
tool results, reasoning plus tool calls, fixed full prompts, and missing Piika
system prompt. Rebuild bundles under a new session and export only after focused
tests, offline parity, independent review, and configured-secret scanning pass.

Old hosted traces remain untouched. Store only public IDs and hashes in the
verification record.

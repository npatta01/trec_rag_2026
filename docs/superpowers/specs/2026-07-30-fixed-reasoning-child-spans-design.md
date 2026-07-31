# Fixed Reasoning Child Spans Design

## Objective

Make the three reasoning-summary blocks captured in the fixed-retrieval model
response individually visible in Phoenix without misrepresenting them as three
model calls.

## Current package location

The immutable records, OpenAI presentation, and Phoenix export belong to
`trec_rag.tracing.{models,openai_semantics,phoenix_export}`. Fixed-run event
construction belongs to `trec_rag.experiments.organizer_pi.event_trace`, part
of a temporary experimental reproduction harness rather than a main production
path.

## Verified Source Shape

The saved fixed `Pi generation` response is one assistant response from one
provider call. Its ordered native `content` array contains three `thinking`
items followed by one final `text` item. The three thinking items contain seven
human-readable headings in total, but the native boundary is three blocks.

Phoenix currently renders the three reasoning content items contiguously inside
one assistant message. That presentation makes the blocks look like one long
reasoning section even though their native boundaries were preserved in the
saved response.

## Trace Topology

Build a new fixed trace with the existing five-span topology plus three ordered
children beneath the single `Pi generation` LLM span:

```text
Ragnarok fixed-retrieval baseline (AGENT)
├── fixed prompt construction (CHAIN)
├── fixed evidence preparation (RETRIEVER)
├── Pi generation (LLM; the only cost-bearing model call)
│   ├── Reasoning summary 1 (CHAIN)
│   ├── Reasoning summary 2 (CHAIN)
│   └── Reasoning summary 3 (CHAIN)
└── organizer validation (CHAIN)
```

The revised fixed trace therefore has eight spans. It still has exactly one LLM
span, one provider response, and one set of model usage/cost attributes. The Pi
agentic trace is unchanged.

## Reasoning Span Contract

Create one child span per native `thinking` item; do not split the seven bold
headings into additional spans. Each child:

- has OpenInference kind `CHAIN`;
- contains the exact captured thinking text as its output;
- records its one-based index and total block count;
- records `pi.reasoning.source=native-thinking-content`;
- carries no LLM model, usage, token, or cost attributes; and
- records that its timing is reconstructed.

No reasoning text is invented, summarized, rewritten, or reordered.

## LLM Rendering and Fidelity

For the revised fixed trace, the rich LLM output message and generic OpenAI
response envelope show only the final assistant text. This prevents the same
reasoning summaries from appearing both in the LLM panel and in the child
spans.

The original response, including all three native thinking items, remains
byte-for-byte available under `pi.native.output_json`. The child spans provide a
second, presentation-oriented view of those exact captured blocks. Input
messages, prompts, 100 documents, final answer, model metadata, usage, cost,
finish reason, and validation output remain unchanged.

## Reconstructed Timing

No per-block timestamps were captured. Divide the `Pi generation` interval into
three deterministic, ordered, non-overlapping child intervals of equal duration,
assigning any integer-nanosecond remainder to the final child. Every child
records:

- `trace.timing_reconstructed=true`; and
- `pi.reasoning.timing_method=equal-partition`.

These durations are presentation estimates, not claims about actual reasoning
latency. All child intervals remain within the real LLM parent interval.

## Scope and Hosted Export

Apply this behavior only to fixed-trace construction when native thinking blocks
are present. Existing saved bundles and hosted traces remain untouched. Rebuild
the fixed trace from saved artifacts only and export one additional fixed trace
to the existing project/session. Do not invoke the organizer runtime, retrieval,
Pi, or a model. The user will clear any older trace themselves.

## Validation

Tests and offline inspection must prove:

- three native thinking items produce exactly three ordered CHAIN children;
- zero thinking items produce no reasoning children;
- the fixed trace has eight spans and exactly one LLM span;
- child outputs match the native thinking strings exactly;
- child timing partitions are ordered, non-overlapping, contained by the LLM,
  and marked reconstructed;
- the rich/generic LLM output contains only the final assistant text;
- `pi.native.output_json` retains the original complete response byte-for-byte;
- prompts, 100 documents, final answer, usage, cost, and validation remain
  unchanged; and
- the hosted session gains exactly one new eight-span fixed trace with the
  expected parent-child relationships.

Run secret scans before export and keep credentials out of commands, receipts,
logs, and tracked files.

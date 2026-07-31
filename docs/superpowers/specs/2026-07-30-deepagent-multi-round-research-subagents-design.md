# Deep Agent Multi-Round Research Subagents POC Design

## Objective

Move search and snippet exploration out of the main retrieval agent into
specialized researcher subagents. The main agent should retain a compact view
of narrative coverage, run several independent researchers when useful, merge
their grounded findings, and launch later rounds for unresolved gaps.

This replaces the brittle `choose_next_action` followed by an exact-string
search/extraction handshake. The fixed tools will record the arguments they
actually execute in the mechanical retrieval ledger.

## Approved POC Shape

The POC uses Deep Agents' synchronous custom subagents:

- the main agent is the only agent allowed to invoke `task`;
- a custom `researcher` subagent receives one evidence goal and returns one
  structured `EvidenceBundle`;
- the researcher can call only the fixed search, snippet, and compact state-view
  tools; it cannot spawn another subagent;
- one main-agent model turn may launch several independent researcher tasks;
- the main agent may launch more rounds after merging the returned bundles;
- each researcher invocation is ephemeral, so every later round receives the
  relevant known evidence and remaining gap in its task description;
- the existing OpenRouter model remains configurable and is used by default for
  both the main agent and researchers.

This is intentionally not a durable job system. If one researcher fails, its
failure is returned without corrupting the other bundles or canonical state.

## Responsibilities

### Main agent

The main agent:

1. decomposes the untouched narrative into needs and facets;
2. selects independent research goals and an automatic depth for each;
3. fans those goals out to researcher subagents, concurrently when independent;
4. validates and serially merges returned bundles into the semantic state;
5. recomputes the frontier of gaps, conflicts, and weakly supported claims;
6. launches focused later rounds when they can improve the answer; and
7. decides when the narrative is answerable or further retrieval is saturated.

Only the main agent mutates the canonical need map and nugget store. Bundle
merges are serialized even when researchers ran concurrently.

### Researcher subagent

The researcher receives an evidence goal rather than a frozen query. It may:

- generate an initial query;
- inspect the returned candidate documents;
- broaden, narrow, rephrase, or refocus a weak query;
- extract several relevant snippets from one document;
- paginate a productive document/focus ranking; and
- triangulate a claim or investigate a conflict at deeper research depth.

The researcher does not update coverage, declare a narrative need answerable,
or write directly to the canonical nugget store. It returns grounded candidate
findings for the main agent to merge.

## Research Depth and Rounds

Depth is selected automatically by the main agent per task:

- `survey`: map the available evidence for a broad or untouched need;
- `focused`: close a specific remaining facet or weakly supported claim;
- `deep`: triangulate important claims, investigate contradictions, or continue
  productive snippet pagination.

Depth controls the researcher's success criteria and prompt, not a hard number
of searches, documents, pages, or snippets. There is no global semantic action
budget. A runtime timeout and maximum concurrent researcher count may remain
configuration safeguards, but they do not determine coverage or saturation.

A typical run is:

1. Round 1 launches survey tasks for independent uncovered needs.
2. The main agent serially merges the bundles and recomputes coverage.
3. Later rounds launch focused or deep tasks only for recorded gaps, conflicts,
   or promising residual evidence.
4. The run stops when every required need is answerable/conflicted or when the
   frontier has no plausible action that is still producing novel evidence.

The current global three-follow-up-search limit is removed from semantic
stopping because it is incompatible with independent multi-round researchers.

## Atomic Fixed-Tool Contract

The fixed tools are cache-first and own all mechanical logging. The model never
receives cache paths, cache keys, or cache-control arguments.

Every search call supplies:

- the actual `query` that will be executed;
- one or more `motivating_ids` for open needs/facets;
- a concise `rationale`;
- `research_task_id`, `round_index`, and `depth` context.

The search tool validates that the motivations are open, records the exact
actual arguments, performs the cache-first retrieval, and records returned
documents in one call.

Every snippet call similarly supplies the actual `document_id`, `focus_query`,
optional `cursor`, motivations, rationale, and task context. It validates and
records the exact page it returns. The configured page size remains fixed and
normally returns 5–10 ranked snippets; productive pagination may continue.

The tools reject invalid motivations before execution. Ordinary search,
extraction, refocus, and pagination no longer require a preceding
`choose_next_action` call or exact duplicate target string. Stop/saturation
validation remains a separate main-agent decision.

## EvidenceBundle

Each researcher returns a structured bundle containing:

```json
{
  "research_task_id": "round-1-need-3",
  "round_index": 1,
  "depth": "survey",
  "motivating_need_ids": ["need-3"],
  "candidate_nuggets": [
    {
      "claim": "...",
      "need_ids": ["need-3"],
      "facet_ids": ["facet-2"],
      "evidence": [
        {
          "document_id": "...",
          "snippet_id": "...",
          "page_index": 0,
          "quote": "exact text from the returned snippet"
        }
      ],
      "contradicts_claims": []
    }
  ],
  "conflicts": [],
  "unresolved_gaps": ["..."],
  "suggested_followups": ["..."],
  "stopping_reason": "goal_satisfied"
}
```

The bundle contains compact grounded findings, not entire documents or raw
snippet pages. A bundle may contain multiple relevant snippets from one
document and multiple documents for one claim.

The main agent validates each evidence quote against snippets already recorded
by the fixed tools. Valid bundle rows are merged even if another row is
rejected. Equivalent claims may be merged as one nugget with additional
evidence; contradictions and supersessions remain explicit.

## Context and State Boundaries

Researcher isolation prevents search results and long documents from polluting
the main agent context. A researcher may use its ephemeral scratch space when
tool output spills out of context, but scratch is not canonical state.

The main agent passes only the goal, motivating IDs, depth, relevant known
evidence summary, and unresolved gap to a researcher. It receives only the
bundle. Mechanical query/page history stays in the retrieval ledger, while
needs, facets, nuggets, and coverage stay in the main semantic stores.

Because synchronous researcher calls are stateless, a later round explicitly
receives the prior round's useful findings rather than relying on hidden
subagent memory.

## Trace and Failure Behavior

Phoenix spans identify the agent role, `research_task_id`, round, depth, and
actual fixed-tool arguments. Existing trace-content configuration determines
whether inputs and outputs are visible; the POC does not intentionally redact
them. Trace flushing remains unchanged.

Expected failure behavior:

- an invalid fixed-tool request returns a narrow, self-correctable error;
- a failed researcher produces a structured task failure and no semantic write;
- invalid bundle rows are rejected individually when possible;
- the main agent may launch a corrective later round without replaying
  successful researchers; and
- concurrent researchers never mutate shared semantic state directly.

## POC Constraints

- Python SDK only; no interactive CLI.
- Same OpenRouter configuration as the existing prototype.
- Live PyTorch/reranker runs use `.venv/bin/python-rocm` on this host.
- Retrieval and reranker caches remain tool-owned and cache-first.
- The untouched narrative remains the primary input; narrative lookup is an
  optional helper rather than a required bundled topic record.
- ClimbMix titles are not invented or added when the endpoint does not return
  them.
- Researchers cannot recursively delegate.
- No global limit on evidence, snippets, documents, pages, or research rounds.

## Implementation Boundary

For this POC, the built-in synchronous `task` tool is used to launch the custom
researcher. The main-agent prompt must place `research_task_id`, motivations,
round, depth, known evidence, and the goal in the task description. The
researcher's structured response format enforces `EvidenceBundle` output.

The main middleware allows `task` only for the main agent. Researcher
middleware exposes the fixed research tools but removes `task`, filesystem,
shell, and unrelated tools. The existing three stores are retained; only the
ordinary action authorization path and orchestration layer change.

## Verification

Keep verification proportional to the POC:

- targeted unit tests for atomic search/snippet logging and motivation
  validation;
- one test that researchers cannot access `task` or unrelated tools;
- one test for concurrent independent bundles followed by serialized merge;
- one test for a later autonomous query-rephrasing round;
- bundle grounding tests for exact quotes and partial-row rejection;
- existing Deep Agent retrieval tests and Ruff; and
- one instrumented topic-224 SDK run using `.venv/bin/python-rocm`.

The live pass succeeds when it initializes the five topic-224 needs, completes
at least one researcher bundle, records the queries actually executed without
action-mismatch errors, merges grounded nuggets, reports remaining gaps or a
grounded stopping reason, flushes its Phoenix trace, and returns within the
bounded diagnostic run.

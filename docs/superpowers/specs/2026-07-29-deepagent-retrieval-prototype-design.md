# Deep Agent Retrieval Prototype Design

## Objective

Build a small Python SDK prototype that answers one design question:

> Can a LangChain Deep Agent improve retrieval coverage by issuing a bounded
> sequence of follow-up ClimbMix searches from a supplied narrative while
> preserving an auditable record of every query and retrieved document?

The prototype is tool-first. Deep Agents controls search planning and stopping;
the repository's existing hosted Pyserini client remains the only corpus-search
implementation.

## Scope

The prototype will:

- accept an arbitrary non-empty narrative string as its only required retrieval
  input;
- always search the untouched narrative before the agent may request follow-up
  searches;
- expose the existing ClimbMix/Pyserini retriever as a bounded agent tool;
- return the original search, follow-up searches, per-search candidates, a
  deterministic deduplicated candidate set, the agent's rationale, its stopping
  reason, and available cache provenance;
- provide a separate helper that loads a narrative by topic ID from an explicit
  TREC topic file;
- use the same OpenRouter provider and DeepSeek model family as the repository's
  existing facet-planning work;
- reuse the shared retrieval cache, rate limiter, and explicit-continuation
  policy already implemented by `PyseriniRemoteRetriever`;
- remain importable and usable without an interactive terminal.

The prototype will not:

- require a topic ID in the primary retrieval API;
- add a chat REPL, `:reset`, or `:quit` commands;
- change `run_pipeline`, `run_official`, organizer export, or sealed experiment
  artifacts;
- create a local vector index or introduce a second retrieval implementation;
- use organizer nuggets, qrels, or hidden topic metadata at retrieval time;
- persist conversation state across SDK calls;
- silently retry hosted model or retrieval calls.

## Public API

The primary API requires only the narrative:

```python
from trec_rag.deepagent_retrieval import DeepAgentRetriever

retriever = DeepAgentRetriever.from_env()
result = retriever.retrieve(narrative)
```

Topic lookup is a separate convenience operation:

```python
from pathlib import Path

from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.topics import load_topic_narrative

narrative = load_topic_narrative(
    "224",
    Path("trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"),
)
result = DeepAgentRetriever.from_env().retrieve(narrative)
```

`load_topic_narrative` requires an explicit topic file so development and test
collections cannot be confused. It returns the narrative text and rejects a
missing or duplicate topic ID. `DeepAgentRetriever.retrieve` neither accepts
nor fabricates a TREC topic ID.

Constructor options may override the model, per-search depth, maximum number of
follow-up searches, and final result depth. Defaults remain deliberately small
for interactive experimentation from a notebook or Python shell.

## Components

### Deep Agent boundary

`trec_rag.deepagent_retrieval` owns all imports from `deepagents` and
`langchain-openrouter`. It constructs the agent with an explicit model string,
a retrieval-specific system prompt, the search tool, and the default ephemeral
state backend. It does not grant host filesystem or shell access.

The default model is the repository's OpenRouter DeepSeek alias and may be
overridden with `DEEPAGENT_MODEL` or a constructor argument. The SDK reports a
clear setup error when `OPENROUTER_API_KEY` is absent.

The dependency boundary is pinned to:

- `deepagents==0.7.0`;
- `langchain-openrouter==0.2.7`.

The narrow factory is intentional: Deep Agents 0.7.0 changed default prompts,
todo middleware, filesystem behavior, and backend contracts. Later upgrades
should require changes at this boundary rather than throughout retrieval code.

### Search tool

The custom `search_climbmix(query: str)` tool adapts each agent query to the
existing `QueryVariant` and `PyseriniRemoteRetriever` interface. It uses a
stable narrative hash as the internal cache topic component and a stable query
hash as the follow-up variant component. These values are cache identifiers,
not public TREC topic IDs.

The SDK performs the untouched-narrative search before invoking the agent. The
agent receives those initial results and may call `search_climbmix` only for
additional coverage. A per-call budget rejects excess searches. Every tool
result contains bounded excerpts plus document ID, source rank, and source
score; the complete normalized candidate remains in the SDK-side trace rather
than being copied into the model context repeatedly.

### Result records

Focused dataclasses describe the observable result:

- `AgentSearch`: exact query, query kind (`original` or `followup`), ordered
  normalized candidates, and cache information available from the retriever;
- `AgentRetrievalResult`: untouched narrative, ordered `AgentSearch` records,
  fused final candidates, agent rationale, and stopping reason.

The final candidate set is computed by deterministic reciprocal-rank fusion
across the searches, followed by document-ID deduplication. Raw BM25 scores are
retained per search but are not compared directly across different queries.
Ties use stable document-ID ordering. The agent chooses follow-up queries and
when to stop; deterministic code chooses the returned ordering.

## Data Flow

1. Validate and preserve the supplied narrative exactly.
2. Load repository environment values without logging secrets.
3. Search the untouched narrative through `PyseriniRemoteRetriever`.
4. Give the Deep Agent the narrative and a bounded view of the original
   results.
5. Let the agent issue zero or more bounded follow-up queries through
   `search_climbmix`.
6. Record each query and normalized candidate list in order.
7. Fuse and deduplicate candidates deterministically.
8. Return the typed result without writing official pipeline artifacts.

## Bounds and Safety

Initial defaults are:

- one mandatory original-narrative search;
- at most three agent-generated follow-up searches;
- ten candidates exposed per search;
- twenty candidates in the fused result;
- bounded document excerpts in model-visible tool results;
- no persistent checkpointer or store.

The default Deep Agents state backend is in-memory and ephemeral. The system
prompt directs the model to use the retrieval tool directly, preserve the
original narrative's intent, avoid unsupported factual conclusions, and stop
when follow-up searches no longer target a specific uncovered aspect.

## Failure Handling

- Empty or non-string narratives fail before any external call.
- A missing OpenRouter key produces a setup error naming
  `OPENROUTER_API_KEY` without revealing environment contents.
- Missing Pyserini endpoint/token configuration retains the existing explicit
  configuration errors.
- Retrieval throttling is surfaced with its continuation ticket and delay; the
  agent and SDK do not sleep or retry automatically.
- Malformed cached responses retain the existing identity and hash validation
  failures.
- An OpenRouter failure preserves completed retrieval traces in the raised SDK
  error so the original search is still inspectable.
- Exceeding the search budget returns a tool-visible budget error and records
  `search_budget_exhausted` as the stopping condition.

## Prototype Verification

This is intentionally a throwaway prototype rather than a production pipeline
feature. Verification will therefore use focused smoke checks instead of a new
large test surface:

- construct the SDK with a fake model and fake existing retriever, verifying
  that the untouched narrative is searched first and follow-up calls are
  recorded;
- exercise the deterministic fusion helper with repeated document IDs;
- confirm the pinned Deep Agents 0.7.0 factory imports and constructs without a
  live OpenRouter call;
- run the repository's existing targeted topic/retriever tests;
- run one live retrieval only when the required OpenRouter credential is
  available, without printing secrets or raw logs;
- run `git diff --check` and a source scan for credentials and placeholder
  markers.

If the prototype validates the agentic search behavior, the reusable result
records and tool adapter can be promoted into a separately designed pipeline
integration. The prototype itself remains clearly named and isolated until
that decision is made.

## Deliverables

- `code/trec_rag/deepagent_retrieval.py` — prototype SDK, Deep Agents factory,
  search-tool adapter, result records, and deterministic fusion;
- `code/trec_rag/topics.py` — separate `load_topic_narrative` convenience
  helper;
- `code/trec_rag/README.md` — SDK usage, inputs, outputs, credentials, bounds,
  and verification command;
- `pyproject.toml` and `uv.lock` — exact prototype dependency pins.

# DeepAgent Retrieval POC — Claude Handover

**Date:** 2026-07-31

**Repository:** `trec_rag_2026`

**Current worktree:** `/home/npatta01/.codex/worktrees/a743/trec_rag_2026`

**Current HEAD:** `9b14fce` (`Test nuggetizer on grounded ledger`)

**Git state at handover:** detached HEAD in a Codex-managed linked worktree;
pre-existing untracked `.superpowers/` is user-owned and must not be touched.

This is the canonical handover for continuing the experimental DeepAgent
retrieval work. Read `AGENTS.md` first, then this document, then the linked
specs and source files. Do not restart the design from scratch.

## Executive summary

The repository now contains a narrative-only Python SDK prototype that:

1. searches the untouched user narrative once through ClimbMix;
2. asks a main DeepAgent coordinator to decompose the narrative into needs;
3. lets the coordinator launch multiple bounded researcher subagents over
   multiple intended rounds;
4. lets each researcher autonomously formulate/refine queries, search ClimbMix,
   and request relevance-ranked snippets from retrieved documents;
5. keeps search and snippet caching inside the fixed tools, invisible to the
   agents;
6. records needs, retrieval activity, coverage, and grounded nuggets in a
   universal invocation-local state/ledger;
7. exports detailed Phoenix traces with explicit researcher span names; and
8. returns an immutable Python `AgentRetrievalResult`.

The latest experiment tested a separate post-batch Nuggetizer stage on topic
224. The important conclusion is:

> Treat researcher `EvidenceBundle`s as untrusted proposals. Mechanically
> validate their `(snippet_id, quote)` references first, admit only grounded
> claims to the ledger, and run any semantic nugget canonicalizer after that
> grounding boundary.

The one hosted canonicalization call safely preserved all six grounded ledger
nuggets and evidence aliases, but merged nothing because the six claims were
not duplicates. Therefore topic 224 validates the provenance mechanics but
does not demonstrate that canonicalization improves redundancy.

The live topic-224 retrieval itself is also not a quality success: Phoenix
reports five needs, zero answerable needs, five unresolved needs, six nuggets,
five researcher invocations, ten retrieval calls, zero completed research
rounds, and `agent_completed` as the stop reason. Only one researcher produced
nuggets, mostly about push/pull migration factors. Investigate why the
coordinator ended without closing a round or covering the rest of the
narrative before treating this prototype as effective retrieval.

## User intent and fixed preferences

These decisions were made explicitly during the design conversation:

- This is a POC. Prefer fast representative experiments over a large TDD or
  production-hardening effort.
- The primary product surface is a Python SDK, not an interactive CLI.
- The SDK accepts a narrative directly. Topic ID is not bundled into the main
  `retrieve()` API. `load_topic_narrative()` is a separate convenience helper.
- Use the existing OpenRouter configuration and DeepSeek V4 Flash unless an
  experiment explicitly changes the model.
- The main agent should be able to spawn multiple researcher subagents in a
  batch, and there may be multiple research rounds.
- Researchers should autonomously rephrase/refine poor queries and search
  again when evidence gaps remain.
- Researchers should search and request snippets themselves. They should not
  blindly summarize search-result prefixes.
- A document may yield multiple relevant snippets. Snippet pages default to
  ten results and can be paginated until exhausted.
- Do not impose a separate global cap on the number of relevant snippets
  simply because a document is long. Safety is enforced through researcher,
  tool, round, and time budgets instead.
- Search and snippet tools must be cache-first. Agents do not manage caches or
  receive cache arguments, paths, keys, or bypass controls.
- Scratch space is acceptable only for invocation-local overflow/notes. It is
  not the ledger and not a persistent cache.
- ClimbMix does not return a title field. Do not invent or derive titles for
  the agent payload.
- Phoenix traces should show content for debugging; do not redact normal model
  and tool content unless the user asks to change that policy. Credentials,
  headers, cache paths, raw provider responses, and scratch paths remain
  excluded.
- The user does not want the model to own cache policy.
- The user likes the three-store ledger design: narrative concepts/needs,
  mechanical retrieval history, and grounded nuggets/coverage.
- The user agreed that researchers may return provisional nuggets, but a
  centralized post-batch stage should own canonical nugget identity.
- Entailment verification is currently out of scope for researchers. If added,
  it should be a later deterministic/model-assisted validation stage.
- Tool/runtime limits are important because the agent must not run indefinitely.
- Keep OpenRouter requests at 120 seconds and disable hidden retries.
- The user stated that the exposed API key will be rotated at the end of the
  project. Never copy any credential from conversation history into source,
  docs, logs, or commands printed to the user.

## Current architecture

```text
provided narrative
       │
       ├─ deterministic original ClimbMix search (exact narrative)
       │       └─ full documents stay in a private invocation registry
       │
       ▼
main DeepAgent coordinator
       ├─ update_retrieval_state(add_needs/...)
       ├─ task(researcher A) ─┐
       ├─ task(researcher B) ─┼─ may execute as a bounded batch
       └─ task(researcher C) ─┘
                  │
                  ├─ search_climbmix(refined query)
                  ├─ extract_relevant_snippets(docid, focus, cursor?)
                  ├─ optionally paginate/refine/search again
                  └─ return EvidenceBundle with provisional claims
       │
       ▼
mechanical quote/snippet grounding
       │ rejects unknown IDs, wrong pages/docs, and non-contained quotes
       ▼
invocation-local need/retrieval/nugget ledger
       │
       ├─ intended: complete_research_round and repeat on remaining gaps
       │
       └─ proposed explicit post-batch canonicalization
              └─ one bounded Nuggetizer/OpenRouter creator call
                 after grounding, never before it
       │
       ▼
immutable AgentRetrievalResult
```

### Why the original search precedes the first model call

The exact narrative search is deterministic and intentional. Earlier, the
first model prompt included large search excerpts, which polluted context and
made it appear that the first LLM call had somehow searched already. The SDK
now sends only document metadata to the agent while keeping complete text in a
private registry. Researchers must call the snippet tool to inspect bounded,
relevance-ranked passages.

## Agent roles and tools

### Main coordinator

Available:

- `task` with only `subagent_type="researcher"`;
- `view_retrieval_state`;
- `update_retrieval_state`;
- `complete_research_round` / terminal state controls;
- `read_file` for automatically spilled oversized results.

Unavailable by design:

- direct `search_climbmix` or `extract_relevant_snippets`;
- recursive general-purpose subagents;
- shell/execute and host filesystem mutation;
- `write_todos` / `TodoListMiddleware`.

The retrieval ledger replaces a generic todo list for this workflow.

### Researcher subagent

Available:

- `search_climbmix`;
- `extract_relevant_snippets`;
- compact `view_retrieval_state`;
- `read_file` for automatic spill.

Unavailable:

- `task` or recursive delegation;
- semantic state updates;
- filesystem mutation and shell;
- generic todo tools.

Middleware mechanically forces the first researcher action to be
`search_climbmix` and the next required action to attempt
`extract_relevant_snippets` on a retrieved document. After those actions, the
researcher may refine queries, inspect other documents, and paginate snippets.

### Researcher output contract

`EvidenceBundle` contains:

- `research_task_id`, `round_index`, `depth`, and motivating need IDs;
- zero or more provisional `CandidateNugget`s;
- conflicts, unresolved gaps, suggested follow-ups, stopping reason, and a
  budget snapshot.

Each `CandidateNugget` contains a claim, need/facet IDs, contradictions, and
one or more `BundleEvidence` objects with:

- `document_id`;
- `snippet_id`;
- `page_index`;
- exact `quote`.

Pydantic validates shape and types only. It cannot establish that the model's
quote actually occurs in the referenced snippet. That semantic/provenance
check must remain mechanical and authoritative.

## The three invocation-local stores

All three are owned by `EvidenceCoverageState` and exposed through compact
views/deltas. They are not persistent caches.

1. **Need map**
   - exact narrative spans;
   - questions/needs;
   - discovered facets/concepts;
   - status (`unaddressed`, `partial`, `answerable`, `conflicted`);
   - remaining gaps and optional draft answers.

2. **Mechanical retrieval ledger**
   - original/follow-up searches;
   - actual query arguments;
   - returned document IDs/ranks;
   - inspected document/focus pairs;
   - snippet pages, pagination continuity, residual counts/scores;
   - actions and document states.

3. **Grounded nugget store**
   - concise claim text;
   - linked need/facet IDs;
   - exact evidence references;
   - contradictions;
   - single- vs multi-document observed support;
   - supersession relationships.

`update_retrieval_state(delta)` is the main append/update interface. It accepts
the sections `add_needs`, `add_facets`, `add_nuggets`, `add_evidence`,
`set_facet_status`, `set_need_status`, `supersede_nuggets`, and
`abandon_documents`. It returns accepted IDs plus row-level rejection codes,
state version, and state hash.

Important grounding rule: a nugget is admitted only when its submitted quote,
after whitespace normalization, occurs in the referenced snippet text already
observed by the state. Unknown snippets and non-contained quotes are rejected.

## Search and snippet behavior

### `search_climbmix`

- The only agent argument is exact `query`.
- Returns at most ten metadata-only candidates by default.
- Model-visible rows contain document ID, rank, score, and text length.
- Complete document text remains private to the SDK registry/result.
- Fixed remote retriever cache identity includes the real request arguments
  and configuration; no cache controls are exposed to the model.

### `extract_relevant_snippets`

```python
extract_relevant_snippets(
    document_id: str,
    focus_query: str,
    cursor: str | None = None,
) -> str
```

- `document_id` must have been returned during the same invocation.
- Default page size is ten snippets; the agent cannot override it.
- One document can return multiple relevant passages.
- `next_cursor` continues the stable ranking until exhausted.
- Long documents are chunked; full documents are never blindly injected.
- Default chunking is semantic, with 3,500-character maximum and
  350-character overlap.
- Default ranker is local
  `mixedbread-ai/mxbai-rerank-base-v2`, pinned revision
  `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`.
- The model must already be present in the local Hugging Face cache because the
  adapter uses `local_files_only=True`.
- Cursor/cache identities include every tool argument, document-text hash,
  chunk/ranker configuration, and schema versions.
- A small hosted/local LLM ranker remains an optional injected backend, not the
  default.

## Safety budgets

Current defaults from `ResearchBudgetConfig`:

| Limit | Default |
| --- | ---: |
| Researcher invocations | 10 |
| Research rounds | 4 |
| Concurrent researchers | 3 |
| Combined researcher search + snippet calls | 100 |
| Tool calls per researcher | 20 |
| Searches per researcher | 8 |
| Snippet calls per researcher | 16 |
| Model calls per researcher | 30 |
| Main model calls | 25 |
| Soft warning | 600 seconds |
| Hard admission deadline | 1,800 seconds |
| Consecutive no-yield retrieval calls | 3 |
| Consecutive no-progress rounds | 2 |

OpenRouter requests use `max_retries=0` and a 120,000 ms request timeout. The
remote search transport also disables automatic retries. Preserve these
properties unless the user explicitly changes them.

## Models currently used

| Function | Backend/model |
| --- | --- |
| Main coordinator | default `openrouter:deepseek/deepseek-v4-flash` |
| Researcher subagents | same configured OpenRouter model as coordinator |
| Snippet relevance ranking | local Mixedbread cross-encoder above |
| Post-batch canonical creator | versioned `deepseek/deepseek-v4-flash-20260423` through OpenRouter |
| Nuggetizer scoring in adapter | local deterministic scorer shim |
| Nuggetizer hosted scoring/assignment | deliberately bypassed |

The installed upstream `nuggetizer==0.0.5` defaults to hosted GPT-4o-family
components when used directly. Do not instantiate its default network stack.
Use `NuggetizerCanonicalNuggetBackend`, which preserves the repository's fixed
OpenRouter model, strict JSON schema, temperature zero, disabled reasoning,
one-shot transport, provenance aliases, and extractive fallback.

Local generative nuggetization has not been tested. This AMD host supports ROCm,
and future local-model experiments should prefer `.venv/bin/python-rocm`, but
do not download a large model or expose a service without user authorization.

## Phoenix tracing

Configured project: `trec-rag-deepagent-retrieval`.

The key trace for this handover is:

```text
trace_id = f9d00a71e02ac6f940deb75a5090db5b
topic_id = 224
```

Tracing is optional in the SDK but enabled in the current local environment.
Default `trace_content=True` means LangChain model/tool input and output content
is visible for debugging. Manual spans still exclude credentials and unsafe
runtime internals.

Researcher tasks have explicit spans named:

```text
deepagent.researcher.<research_task_id>
```

and carry role/task/round/depth metadata so the Phoenix tree distinguishes
subagents from the main graph. Earlier traces were difficult to read because
automatic `task`/`tools` spans looked generic; the explicit dynamic spans are
the intended UI marker. Phoenix does not currently provide a custom color API
through this instrumentation, so naming and attributes are used instead.

Never add raw API credentials to span attributes. A Phoenix API key was shared
in conversation history; it is intentionally absent from this document.

## Topic-224 live retrieval result

Narrative:

> I want to understand why people immigrate or become refugees, the challenges
> they face, and how laws and different groups shape immigration policies.
> Additionally, I'm interested in how various countries and religions view
> immigrants, and what options migrant workers have to improve their lives.

Verified root-span metrics:

| Metric | Value |
| --- | ---: |
| Need count | 5 |
| Answerable needs | 0 |
| Unresolved needs | 5 |
| Grounded ledger nuggets | 6 |
| Researcher invocations | 5 |
| Retrieval calls | 10 |
| Recorded completed rounds | 0 |
| Actions | 10 |
| Stop reason | `agent_completed` |

Researcher-bundle reconstruction found:

- five `task` outputs;
- one productive bundle (`R1-N1`) with six candidate nuggets;
- four empty bundles (`R1-N2`, `R1-N4`, `R1-N1-fix`, `R1-N1-READ`);
- four successful manual snippet-page spans;
- sixteen total observed snippets;
- six final grounded ledger nuggets, all supported by document
  `shard_00467_78505`.

The result over-focused on push/pull causes of migration and basic refugee
facts. It did not make the five narrative needs answerable and did not cover
the requested laws, policy-shaping groups, country/religion views, worker
options, and challenges sufficiently. The next quality investigation should
explain why five dispatched researchers yielded only one productive bundle and
why `research_round_count=0` despite five completed task spans.

## The invalid-citation finding

Two raw researcher candidates failed the mechanical evidence rule:

```text
R1-N1:p003:e1 -> ungrounded_quote
R1-N1:p004:e1 -> ungrounded_quote
```

This does not mean the general claims were necessarily false. It means the
model-authored `(snippet_id, quote)` association was not supported by the
authoritative snippet-tool output.

- Candidate `p003` cited `shard_00467_78505:0001`. Its first 332 normalized
  characters matched that snippet, but the generated “quote” then appended
  text not contiguous in the snippet. None of the sixteen returned snippets
  contained the complete submitted quote.
- Candidate `p004` also cited `shard_00467_78505:0001`, but none of the sixteen
  returned snippets contained its submitted pull-factor quote. The best
  contiguous overlap was only two characters.

Plausible mechanism: the researcher saw multiple tool-returned passages and
then generated a polished quote-shaped string while attaching the wrong or
insufficient snippet ID. A Pydantic schema catches missing/wrong types, not
this referential error. The ledger validator correctly rejected or replaced
those attempts with grounded versions.

Do not “repair” these citations silently using fuzzy matching. Either reject
the candidate or have an explicit retrieval/validation step obtain correct
evidence.

## Post-batch Nuggetizer probe

### Purpose

The user asked whether researchers should return nuggets or whether nugget
creation/canonicalization should be a separate stage. The agreed design is:

- researchers may return provisional, topic-aware claims;
- mechanical validation establishes trusted evidence;
- one explicit central post-batch stage owns canonical nugget identity and
  cross-researcher deduplication;
- the main agent should not independently improvise canonical identity.

### Probe implementation

File: `code/trec_rag/post_batch_nuggetizer_probe.py`.

The probe reads the existing Phoenix trace in memory and has two modes:

- `--input-source researcher`: diagnose the raw researcher bundle. Its dry-run
  reconstructs all expected records but reports the two grounding failures.
  Hosted execution is blocked before the backend is created.
- `--input-source ledger`: use the same trace's six mechanically accepted
  grounded ledger nuggets. This is the safe canonicalization boundary.

No raw trace payloads, snippets, cloud responses, or credentials are persisted.

### Hosted result

One ledger-mode call was made; do not repeat it merely to reproduce the number.

| Field | Result |
| --- | --- |
| State | `complete` |
| Model | `deepseek/deepseek-v4-flash-20260423` |
| Provider | AtlasCloud through OpenRouter |
| Hosted calls | 1, no retry |
| Latency | 3.104 seconds |
| Tokens | 585 prompt + 217 completion = 802 |
| Reported cost | `$0.00014266` |
| Input/output claims | 6 → 6 |
| Claim text | all six unchanged |
| Evidence aliases | all six retained |
| Orphaned/unknown aliases | 0 / 0 |
| Exact duplicates | 0 |
| Paraphrase merges | 0 |

Interpretation: provenance preservation worked, but the topic had no duplicate
grounded claims. The result is neutral on deduplication value. A useful next
probe needs a deliberately duplicate/paraphrased grounded batch or another
trace known to contain cross-researcher overlap.

## Environment and commands

### Environment layout

- Python version: `3.12.13` from `.python-version`.
- Active environment: `.venv/`.
- Use `.venv/bin/python-rocm` for live snippet/reranker work on this AMD host.
- Worktree `.env` currently contains Phoenix settings.
- Shared checkout environment
  `/home/npatta01/data/competitions/trec_rag_2026/.env` contains the existing
  OpenRouter and ClimbMix settings.
- Both files are ignored/private. Never print their values.
- Shared caches live under the main checkout's `cache/` directory and resolve
  correctly from linked worktrees.

If the environment must be rebuilt, run:

```bash
code/tools/setup_env.sh
```

### Main Python SDK

```python
from pathlib import Path

from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.topics import load_topic_narrative

narrative = load_topic_narrative(
    "224",
    Path("trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"),
)

result = DeepAgentRetriever.from_env().retrieve(narrative)
print(result.stopping_reason)
print(result.budget_snapshot.as_dict())
print(result.coverage_report.as_dict())
```

Run live retrieval with:

```bash
set -a
source .env
source /home/npatta01/data/competitions/trec_rag_2026/.env
set +a
PYTHONPATH=code .venv/bin/python-rocm your_script.py
```

Do not paste credentials directly into a command or notebook.

### Nuggetizer probe: safe local diagnostics

Raw-researcher diagnostic; expected to report two grounding failures and zero
hosted calls:

```bash
set -a
source .env
set +a
PYTHONPATH=code .venv/bin/python \
  code/trec_rag/post_batch_nuggetizer_probe.py \
  --input-source researcher \
  --dry-run
```

Grounded-ledger preflight; expected to pass with six inputs, sixteen snippets,
zero grounding failures, and zero hosted calls:

```bash
set -a
source .env
set +a
PYTHONPATH=code .venv/bin/python \
  code/trec_rag/post_batch_nuggetizer_probe.py \
  --input-source ledger \
  --dry-run
```

The non-dry ledger command makes a hosted call. Run it only for a deliberate
new experiment, source both environment files, and retain the existing outer
timeout/no-retry discipline.

### Verification commands

The latest POC verification passed:

```bash
PYTHONPATH=code .venv/bin/python -m py_compile \
  code/trec_rag/post_batch_nuggetizer_probe.py

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_canonical_nugget_contract.py -q
# 21 passed
```

The larger SDK validation command documented in `code/trec_rag/README.md` is:

```bash
.venv/bin/python -m pytest \
  code/tests/test_topics.py \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_pipeline.py \
  code/tests/test_remote_pyserini.py -q
```

Do not claim the full suite passes based only on the 21 canonical-contract
tests; run the appropriate fresh command after future code changes.

## Key source files

| File | Responsibility |
| --- | --- |
| `code/trec_rag/deepagent_retrieval.py` | Public SDK, coordinator prompt, tool binding, narrative-first search, RRF, result assembly |
| `code/trec_rag/deepagent_research.py` | Researcher schemas, role-specific middleware, forced search/snippet actions, task envelope recovery/tracing |
| `code/trec_rag/deepagent_budget.py` | Invocation-local task/tool/model/round/time budgets and adaptive stopping |
| `code/trec_rag/deepagent_evidence.py` | Need map, mechanical retrieval ledger, grounded nugget store, quote validation, immutable report |
| `code/trec_rag/deepagent_snippets.py` | Semantic chunking, local/optional ranker adapters, pagination, cursor/cache validation |
| `code/trec_rag/deepagent_tracing.py` | Phoenix/OpenTelemetry setup, safe attributes, dynamic researcher spans, redaction mode |
| `code/trec_rag/canonical_nuggets.py` | Typed canonical request/result contracts, one-shot backend seam, validation/fallback |
| `code/trec_rag/nuggetizer_adapter.py` | Bounded bridge to `nuggetizer==0.0.5` using one OpenRouter creator call and local scorer |
| `code/trec_rag/post_batch_nuggetizer_probe.py` | Throwaway trace reconstruction and researcher-vs-ledger canonicalization probe |
| `code/trec_rag/topics.py` | Separate explicit topic-file narrative lookup helper |
| `code/trec_rag/README.md` | Current user-facing SDK behavior, contracts, configuration, tracing, and validation |

## Design and plan documents

Read these in order for the history of the POC:

1. `docs/superpowers/specs/2026-07-29-deepagent-retrieval-prototype-design.md`
2. `docs/superpowers/specs/2026-07-30-deepagent-relevant-snippets-poc-design.md`
3. `docs/superpowers/specs/2026-07-30-deepagent-evidence-coverage-map-design.md`
4. `docs/superpowers/specs/2026-07-30-deepagent-multi-round-research-subagents-design.md`
5. `docs/superpowers/specs/2026-07-31-deepagent-post-batch-nuggetizer-probe-design.md`
6. `docs/superpowers/plans/2026-07-31-deepagent-post-batch-nuggetizer-probe.md`

Matching implementation plans for the earlier stages are under
`docs/superpowers/plans/` with the same date/topic names.

## Recent commits and git handoff

Relevant latest commits, newest first:

```text
9b14fce Test nuggetizer on grounded ledger
3693d70 Use grounded ledger for nuggetizer continuation
5515111 Probe post-batch nuggetization from Phoenix trace
e77a5e7 Plan post-batch nuggetizer probe
e74ab5a Design post-batch nuggetizer probe
13e7acb Require researcher snippet inspection
d555730 Recover researcher tasks and clarify Phoenix traces
448585b Force retrieval before closing research rounds
e66497b Bound OpenRouter requests and recover task envelopes
06f3c59 Fix final bounded research review findings
a2fc5dc Cover async researcher task tracing
e9b3944 Preserve Deep Agent trace stop precedence
a1eeb85 Trace and document bounded researcher retrieval
17ad7e7 Separate local and global research stops
```

The worktree is detached and externally managed. No branch was pushed and no PR
was created. Do not merge, push, delete the worktree, or clean the untracked
`.superpowers/` directory without explicit user direction. Continue from the
current HEAD or create a named branch only after confirming the intended
integration target.

## Known limitations and unresolved questions

### 1. Round completion/control flow is not behaving as intended

The key trace shows five researcher invocations but zero recorded completed
rounds, followed by `agent_completed`. This is the highest-priority diagnostic.
Determine whether:

- completed task outputs were not recognized as a single batch;
- the coordinator failed to issue the required one batched semantic update;
- `complete_research_round` was never called or was rejected;
- task recovery/failure messages caused the coordinator to spawn repair tasks
  instead of closing the round;
- the main model simply stopped despite middleware expectations.

Use the Phoenix tree and `deepagent.researcher.*` spans before changing prompts.
Prefer a mechanical middleware/state fix if the invariant can be enforced.

### 2. Raw EvidenceBundle citations are untrusted

Researchers can return schema-valid but provenance-invalid evidence. Add an
explicit validation boundary after each task/batch. A promising design is:

```text
raw EvidenceBundle
  -> validate every evidence reference against authoritative snippet state
  -> validated/rejected candidate records with stable rejection codes
  -> canonicalizer receives validated candidates only
```

Do not let the main LLM repair quotes by rewriting them.

### 3. Canonicalization is not integrated

The Nuggetizer work is a throwaway probe only. It is not wired into
`DeepAgentRetriever`. If integrated, preserve:

- one explicit post-grounding call per completed batch or deliberate round;
- stable evidence aliases;
- no retries;
- fallback to the already grounded ledger;
- no mutation on malformed/unknown provenance;
- compact existing-ledger context so cross-round duplicates can be detected;
- need IDs, facet IDs, conflicts, and contradiction links, which the current
  canonical contract/probe does not fully propagate.

Decide explicitly whether canonicalization is batch-local, round-local, or
global across the invocation. The user's “universal ledger” preference suggests
new candidates should be compared with the compact existing canonical ledger,
not only with siblings in the current batch.

### 4. No deduplication-positive test exists

Topic 224 contained no exact or paraphrased duplicates after grounding. Before
integrating the stage, construct a tiny controlled grounded batch containing:

- one exact duplicate;
- one paraphrase supported by a different document;
- two related but materially distinct claims that must not merge;
- a contradiction pair;
- stable evidence aliases from multiple documents.

One bounded call should demonstrate the intended merge/no-merge behavior and
provenance retention. Keep this smaller than another full live retrieval run.

### 5. Retrieval quality remains poor on the single live topic

The architecture worked mechanically, but coverage did not. Diagnose why four
researchers returned empty bundles and why only N1 yielded nuggets. Likely areas
to inspect include:

- coordinator task goals and remaining-gap text;
- researcher query refinement after low-yield first results;
- forced action middleware and whether it returns too early after one snippet
  attempt;
- document selection from metadata-only search results;
- pagination decisions when `next_cursor` exists;
- task/round budgets and repair-task churn;
- whether the main coordinator understands empty bundles and dispatches a
  genuinely different follow-up.

Do not increase budgets first. Establish why the existing ten retrieval calls
produced so little coverage.

### 6. Entailment is not verified

Grounding proves only that a quote occurs in the cited snippet. It does not
prove that the quote entails the summarized claim, that the source is reliable,
or that multiple documents are independent. If added later, entailment should
be a separate stage after exact citation grounding, ideally with deterministic
rejection/uncertainty states and retained source provenance.

### 7. Local generative nuggetization is untested

The local Mixedbread model is a reranker, not a claim generator. A local
generative model might work on this ROCm machine, but this was intentionally not
tested. Start with a tiny offline model/runtime compatibility probe before any
large download or service.

## Recommended next actions

Do these in order unless the user redirects:

1. **Diagnose the topic-224 round/control-flow failure without changing code.**
   Explain why five tasks resulted in zero completed rounds and early
   `agent_completed`.
2. **Add a mechanical EvidenceBundle validation boundary.** Raw researcher
   citations should never be described as grounded until checked against the
   authoritative snippet registry.
3. **Create one tiny deduplication-positive canonicalization fixture/probe.**
   Prove exact duplicate merge, paraphrase merge, distinct-claim preservation,
   contradiction handling, and evidence retention with one hosted call at most.
4. **Only if that succeeds, design the production seam.** Preferred flow is
   researcher proposals → validator → central canonicalizer against existing
   ledger → one deterministic ledger update.
5. **Run another single-topic retrieval only after the control-flow fix.** Keep
   the 120-second request limit and current budgets; compare coverage, number of
   productive researchers, completed rounds, snippet pagination, and stop
   reason to trace `f9d00a71e02ac6f940deb75a5090db5b`.

## What not to do

- Do not rerun expensive/live retrieval merely to reproduce known trace facts.
- Do not rerun the successful hosted Nuggetizer call unless testing a changed
  input or implementation.
- Do not trust model-authored citation coordinates without mechanical checks.
- Do not fuzzy-repair quotes silently.
- Do not expose cache policy or cache arguments to agents.
- Do not inject full retrieved documents into the first model context.
- Do not invent ClimbMix titles.
- Do not add `TodoListMiddleware` unless the user reverses the ledger decision.
- Do not give researchers recursive `task`, shell, or filesystem mutation.
- Do not hide subagents inside generic Phoenix spans; preserve explicit
  `deepagent.researcher.<id>` span names and role metadata.
- Do not enable retries or raise the 120-second OpenRouter timeout casually.
- Do not include credentials or raw private trace payloads in committed files.
- Do not push, publish, deploy, or alter sharing permissions without explicit
  user authorization.

## Definition of a good next milestone

A strong next handoff should show, on one bounded topic run:

- at least one completed research round recorded mechanically;
- multiple productive researchers covering different narrative needs;
- exact citation validation before claims enter the nugget ledger;
- clear rejected-candidate reasons rather than silent repair;
- no recursive agents or budget overruns;
- explicit Phoenix researcher spans;
- a final result whose need statuses and stopping reason agree with the trace;
- no hosted retries and no secret leakage;
- fresh targeted verification evidence.

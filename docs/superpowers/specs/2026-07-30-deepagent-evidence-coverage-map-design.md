# Deep Agent Evidence/Coverage Map POC Design

## Objective

Add invocation-local retrieval state that lets the Deep Agent answer two
questions throughout one retrieval run:

1. What parts of the supplied narrative remain unanswered?
2. Which search, document, focus query, or snippet page is most likely to close
   one of those gaps?

The POC will preserve the existing narrative-first ClimbMix search,
cache-first tools, relevance-ranked snippet pagination, OpenRouter model, and
Phoenix tracing. It will not turn the agent into a cache manager or persist
semantic memory across retrieval calls.

## Scope and Decisions

The approved design uses three connected invocation-local stores:

- a **need map** for narrative information needs, facets, and model-asserted
  coverage;
- a **retrieval ledger** for mechanically observed searches, documents,
  focus queries, snippet pages, cursors, scores, and yield;
- a **nugget store** for atomic claims and their grounded evidence.

The stores remain separate because their facts have different owners and
different levels of verifiability. They may share one SDK state container, but
they do not share one free-form model-authored record.

The POC will:

- seed needs from explicit clauses in the untouched narrative;
- produce five primary needs for topic 224, matching its five explicit
  information requests;
- permit evidence-discovered facets that attach to one or more seeded needs;
- represent several relevant snippets from one document and several documents
  supporting one nugget without treating them as independent nuggets;
- expose compact, task-specific state views rather than injecting the full
  stores into every model call;
- enrich snippet pages with enough residual-ranking information for the agent
  to make an informed pagination decision;
- preserve the existing configured follow-up-search bound;
- preserve a fixed SDK-configured snippet count per page;
- add no global cap on inspected documents, snippet pages, snippets, or agent
  actions;
- stop on semantic completion or evidence-yield saturation instead of a global
  count budget.

The POC will not add cross-invocation memory, embedding-based nugget merging, a
small-model controller, source-authority scoring, recursive concept graphs,
automatic contradiction adjudication, or official-run integration.

## Why Pagination Needs Its Own Signal

The current `next_cursor` is opaque. It tells the model that another page
exists but not whether that page is likely to contain useful evidence. The
snippet extractor already ranks the complete de-duplicated chunk sequence for
one `(document_id, focus_query)` and slices that stable sequence into pages.
Scores are therefore comparable between pages of that same extraction, though
not across different documents or focus queries.

Each snippet page will retain its existing fields and add:

```json
{
  "page_index": 0,
  "residual_count": 17,
  "residual_top_score": 0.78,
  "returned_min_score": 0.74,
  "pages_estimated": 3
}
```

`residual_top_score` is the next withheld ranked chunk's score, or `null` when
the page exhausts the ranking. `returned_min_score` is the lowest score on the
current non-empty page. `pages_estimated` uses the configured fixed page size.
These are mechanical tool outputs and participate in the exact cache response.
No score is interpreted across a different document or focus query.

The POC will omit a residual text preview initially. Score, count, and observed
nugget yield are enough to test the decision policy without leaking another
snippet outside the configured page.

## Domain Model

### Need map

An information need is an explicit question demanded by the narrative. Each
need stores:

- stable `need_id`;
- exact `narrative_span` copied from the untouched narrative;
- one atomic `question`;
- `status`: `unaddressed`, `partial`, `answerable`, or `conflicted`;
- concise `remaining_gap`;
- linked facet and nugget IDs;
- optional `draft_answer` and its supporting nugget IDs.

The model proposes the initial need decomposition before document inspection.
Topic 224 is expected to produce five needs: migration/refugee drivers,
challenges faced, law/group influence on policy, religious views of migrants,
and migrant-worker options. The SDK does not rewrite the supplied narrative.

A facet is a scoped dimension/value needed to answer one or more needs, such
as `population/refugee`, `actor/legislature`, `religion/christianity`, or
`mechanism/legal-status`. A facet discovered from evidence must cite its
originating snippet. Facets are cross-links, not a parent/child concept tree.

### Nugget store

A nugget is one concise, assertive claim supported by returned snippets. It
stores:

- stable `nugget_id`;
- claim text;
- linked need and facet IDs;
- zero or more contradiction links;
- one or more evidence references;
- derived document-support keys and support multiplicity;
- append step and optional `superseded_by` link.

An evidence reference stores `document_id`, `snippet_id`, `page_index`, and an
exact supporting quote. The quote must occur in the referenced cached snippet.
Several evidence references from one document may support one nugget but remain
one document. Evidence from another document is added to the existing nugget
when it supports the same claim. Because ClimbMix provides no title, publisher,
domain, or upstream-origin metadata, the POC reports only `single_document` or
`multi_document` support. It does not label distinct documents as independent
sources and does not invent a title or source identity.

Nuggets are append-only. Corrections create a new nugget and link the old one
through `superseded_by`; contradictions remain visible rather than being
deleted or silently resolved.

### Retrieval ledger

The SDK and tools append mechanical records for:

- original and follow-up searches, exact query text, result rank, and document
  IDs;
- each document/focus-query pair inspected;
- page index, snippet IDs, relevance scores, and cursor availability;
- residual count and residual score fields;
- nugget IDs proposed from the page;
- novel-nugget yield;
- document state: `unexamined`, `productive`, `exhausted`, or `abandoned`, with
  a reason.

The model does not copy these facts into the ledger and does not handle cache
status, keys, paths, or reuse decisions.

## Ownership and Validation

The model proposes semantic changes through one structured state-update tool:

- initial needs;
- facets discovered from the narrative or snippets;
- new nuggets and evidence links;
- additional evidence for an existing nugget;
- contradiction and supersession links;
- need status, remaining gap, and supported draft answer;
- explicit document abandonment reason;
- the motivating need/facet for its next action.

Deterministic SDK code validates only facts it can prove:

- referenced needs, facets, documents, snippets, and nuggets exist;
- referenced snippets were returned during this invocation;
- an evidence quote is a whitespace-normalized substring of that snippet;
- cursors and page indices match the stored extraction;
- distinct-document counts and page-yield counts are derived from stored
  evidence;
- an `answerable` need has a non-empty draft answer and grounded supporting
  nuggets.

The SDK does not claim to determine whether two paraphrases are the same claim,
whether a nugget truly answers a need, whether two claims genuinely conflict,
or whether coverage is semantically complete. Those remain reversible model
judgments with explicit evidence and reasons. Invalid structured writes are
rejected with narrow reason codes rather than silently repaired.

## Agent-Facing State Tools

The agent receives three semantic-control operations alongside the existing
search and snippet tools:

1. `view_retrieval_state(scope="frontier")` returns a compact projection.
   Scopes may select one need or document when detailed evidence is required.
2. `update_retrieval_state(delta)` submits a batched semantic change. The SDK
   validates it and returns accepted IDs plus rejected entries and reasons.
3. `choose_next_action(action, target, motivating_ids, rationale)` records why
   the agent selected a search, extraction, pagination, refocus, or stop action.

The full state stays in invocation-local SDK/LangGraph state. The default
frontier projection targets roughly 500 tokens and contains only need statuses,
remaining gaps, open facets, documents with residual snippets, recent nugget
yield, and pending actions. Nugget text and exact evidence are fetched only for
a selected need or document. Existing Deep Agent state scratch remains
available for automatic oversized-result spill or temporary notes, but it is
not the canonical semantic state.

## Coverage Rules

Need status is model-asserted and reversible:

- `unaddressed`: no grounded nugget answers the need;
- `partial`: grounded evidence contributes but the recorded gap remains;
- `answerable`: the model supplies a draft answer supported by grounded nugget
  IDs and records no unresolved required gap;
- `conflicted`: grounded nuggets disagree or cannot be reconciled at their
  current scope.

A topical mention alone cannot advance coverage. The draft-answer requirement
is the POC's false-coverage gate. Evidence count never substitutes for semantic
coverage, and ten snippets from one document never imply ten independent
sources.

## Retrieval Decision Policy

Every non-stop action records a motivating open need or facet. A completion
stop records no motivating IDs because no open need remains. A saturation stop
records the unresolved need or facet IDs whose eligible actions stopped adding
grounded nuggets.

1. Prefer an explicit narrative need that remains `unaddressed`; do not keep
   deepening an already productive need while another has no grounded nugget.
2. Inspect an unseen candidate document when its search query/rank indicates it
   targets the selected gap.
3. Paginate the current `(document_id, focus_query)` when:
   - `next_cursor` exists;
   - the previous page produced at least one novel grounded nugget for an open
     need or facet;
   - `residual_count` is non-zero;
   - `residual_top_score` does not show a material relevance cliff relative to
     `returned_min_score`; and
   - the document still plausibly addresses the recorded remaining gap.
4. Refocus the same document with a new focus query when it appears relevant to
   an open gap but the current focus yields no useful nugget.
5. Issue a targeted follow-up search when no retrieved document plausibly
   addresses the selected gap. The existing configured follow-up-search bound
   remains enforced by the search tool.
6. Abandon a document/focus pair after two consecutive pages or refocuses yield
   no novel grounded nugget, when remaining scores show a material cliff, or
   when all gaps it plausibly addresses are closed. Record the reason.
7. Stop when every primary need is `answerable` or explicitly `conflicted`, or
   when repeated eligible actions add no novel grounded nuggets and no open gap
   has a plausible unseen document, residual page, refocus, or available search.

The material score-cliff tolerance is an SDK configuration value used only
within one document/focus ranking. It is not exposed as an agent argument.
There is no page-count, snippet-count, document-count, or total-action stop.

## Context, Cache, and Trace Boundaries

Search and snippet caches remain owned by their tools. Every effective snippet
argument and the enriched response fields remain bound to the existing exact
cache identity. The model never receives cache controls or paths.

Phoenix will show:

- compact semantic-state reads and accepted/rejected state deltas;
- action type, target, motivating need/facet IDs, and state version/hash;
- mechanical page signals and novel-nugget yield;
- grounded snippet content when `trace_content=True`;
- redacted content when `trace_content=False`.

No credential, local cache path, opaque cursor value, or scratch-file content
is added to manual span attributes.

## Failure Behavior

- Reject ungrounded evidence and return the rejected delta item and reason.
- Reject unknown or cross-invocation IDs.
- Reject `answerable` without a draft answer and grounded supporting nuggets.
- Reject pagination against a stale or mismatched cursor.
- Keep prior accepted state unchanged when one batched delta item fails; report
  accepted and rejected items independently.
- Keep retrieval available if a semantic-state update fails so the agent can
  correct the delta.
- Surface tracing export failure without altering retrieval or semantic state.
- On final saturation, return unresolved needs and gaps instead of fabricating
  coverage.

## POC Verification

Keep verification targeted:

1. A snippet-page test proves residual metadata matches the stable ranked
   sequence and changes correctly across pages.
2. A grounding test accepts an exact snippet quote and rejects an invented or
   cross-invocation quote.
3. A support-multiplicity test proves several snippets from one document remain
   `single_document` while another document changes the nugget to
   `multi_document` without claiming true source independence.
4. A coverage test rejects `answerable` without a grounded draft and allows
   reversible status changes.
5. A compact-view test proves the default projection omits full snippet text
   and remains bounded while need/document views expose requested evidence.
6. An action-record test proves every search, extraction, pagination, and
   refocus decision names an open motivating need or facet at decision time;
   completion-stop and saturation-stop follow their explicit rules above.
7. A topic-224 smoke run compares the previous trace with the new run for need
   coverage, nugget grounding, documented abandonment, pagination decisions,
   unresolved gaps, and follow-up queries.

The POC succeeds when its topic-224 trace explains every retrieval action in
terms of the current coverage state, contains no accepted ungrounded evidence,
and either covers or explicitly reports each of the five narrative needs. It
does not require that every available cursor be followed.

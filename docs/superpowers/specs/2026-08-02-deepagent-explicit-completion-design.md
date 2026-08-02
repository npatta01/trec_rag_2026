# DeepAgent Explicit Retrieval Completion Design

## Problem

The coordinator can currently terminate when the model returns no tool call.
That framework-level exit is not a validated retrieval completion. Two live
runs ended with grounded nuggets but no selected `draft_nugget_ids`, flattening
the downstream submission-ranking signal. The existing one-shot middleware
bounce is useful defense in depth, but it is inert in production and does not
make completion an explicit state transition.

## Goal

Make `complete_retrieval` the only successful completion transition. A
completion attempt must reject any need that has live, non-superseded nuggets
but no selected draft nuggets. Silent exits remain guarded during migration and
are reported as incomplete closeouts rather than being mistaken for a valid
completion.

## Design

### 1. One coverage-derived invariant

`EvidenceCoverageState` exposes a small, lock-protected helper that returns the
need IDs whose nugget set contains at least one non-superseded nugget while
`draft_nugget_ids` is empty. This is the single source of truth used by:

- the `complete_retrieval` coordinator tool;
- the middleware's silent-exit predicate; and
- final stopping-reason computation.

The helper does not require a need's only nuggets to be live: superseded-only
evidence must not force a closeout call that cannot legally select it.

### 2. Explicit coordinator terminal tool

Add a no-argument `complete_retrieval` callable to `AgentToolset` and expose it
as a coordinator tool. The tool first checks the shared invariant. On failure,
it returns bounded structured JSON containing `ok: false`, a stable rejection
code, and the offending need IDs; it does not raise and does not mutate the
terminal state. It records the existing `closeout_refused` budget flag so the
final result and trace can distinguish this path from a voluntary completion.

On success, it reuses `EvidenceCoverageState.choose_action` with the existing
`stop`/`completion` transition. This preserves the established terminal-state
representation and its validation of open needs, pending actions, and
completion shape.

The legacy `choose_next_action(..., target="completion", ...)` route is rejected
with a stable guidance code so the new tool is the only model-facing successful
completion route. Non-terminal action behavior remains unchanged.

### 3. Silent-exit guard and final edge

Wire the existing predicate from `DeepAgentRetriever.retrieve` through
`AgentToolset` into `MainToolFilterMiddleware`. The one-shot guard remains a
closeout write-up redirect, then tells the coordinator to call
`complete_retrieval`; it does not create a second terminal definition.

Before final result construction, derive the same coverage invariant. If the
run has no terminal completion and still has pending closeout needs, record
`closeout_refused`. Stopping-reason precedence remains:

1. retrieval unavailable;
2. an explicit coverage terminal reason;
3. budget exhausted;
4. closeout refused/incomplete;
5. framework agent completed.

This ensures a second silent response, an un-wired test agent, or another
incomplete termination cannot be mislabeled as an ordinary completed run.

### 4. Trace safety

Register `closeout_refused` in the trace stopping-reason whitelist. The existing
trace recording boundary swallows exceptions, so this registration is required
for the new result to appear in Phoenix.

## Error handling

Completion rejection is a normal tool result, not an exception. The result is
bounded and names only need IDs and a stable code. The budget flag is an
observation, not a budget stop code; it must not outrank a later successful
explicit completion or an actual budget exhaustion.

## Tests

- Coverage helper: live evidence with no drafts triggers; selected drafts and
  superseded-only evidence do not.
- Completion tool: valid state reaches the existing completion terminal state;
  incomplete state returns the structured rejection and leaves the graph open.
- Alternate route: `choose_next_action` cannot create completion directly.
- Real LangGraph loop on pinned dependencies using
  `FakeMessagesListChatModel`: successful explicit completion terminates;
  incomplete completion returns to the loop; a silent exit is redirected; and
  a refusal is surfaced as `closeout_refused` rather than `agent_completed`.
- Existing retrieval, middleware, tracing, and full test suites remain green.

## Non-goals

- No automatic selection of draft nuggets; selection remains coordinator state
  and must be grounded and bounded by existing validators.
- No wiring into competition runners or submission artifacts.
- No changes to the lower-priority researcher lock, support-ratio calculation,
  dead vital-count field, or ranker integration.

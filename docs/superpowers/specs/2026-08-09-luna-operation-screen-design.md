# Luna Operation Screen Design

## Objective

Add the smallest reliable quality gate to the hardened bounded-splice flow: one inexpensive Luna
call that accepts or rejects each already-validated Sol operation without writing or rewriting
answer prose. Test it by replaying the frozen operations for development topics `233`, `300`, and
`499`; make zero new Sol calls and leave the production competition runner unchanged.

The screen succeeds if it rejects topic `233`'s known weak replacement while retaining the useful
insertions on `233`, `300`, and `499`. A structurally valid operation is not automatically useful;
the decision must consider the complete narrative and surviving draft.

## Approaches Considered

### One whole-answer decision vector — selected

Give Luna the complete narrative context, indexed draft, validated operation set, named audit cards,
and authenticated evidence catalog. It returns exactly one strict decision record for every stable
operation ID. Local code derives acceptance and applies the accepted subset.

This uses one cheap call per topic, lets Luna detect redundancy across the whole answer, and keeps
all mutation deterministic.

### One call per operation — rejected

This isolates judgments but creates three to five calls per tested topic, makes cross-operation and
whole-answer redundancy harder to see, and conflicts with the preference to avoid call
proliferation. It offers no clear benefit for these small operation sets.

### Deterministic lexical or score thresholds — rejected

Local rules can validate syntax, citation domains, and edit geometry, but they cannot reliably
decide whether a sentence is semantically redundant, whether a replacement loses a caveat, or
whether a passage fully supports the complete claim. Those are exactly the observed failure modes.

## Pure Screening Contract

The reusable `trec_rag.operation_screen` module receives a validated tuple of `SpliceOperation`
values and returns the accepted subset plus structured rejection reasons. Operations receive stable
IDs `op001`, `op002`, and so on in original Sol order.

The strict provider response contains one record per operation with exactly these fields:

- `operation_id`;
- `fully_supported`: every cited document's selected passages fully support the complete new
  sentence without outside knowledge;
- `atomic`: the sentence expresses one coherent claim at the competition's citation unit;
- `material`: it materially improves the answer to the complete narrative;
- `nonredundant`: surviving draft prose does not already convey the same point;
- `replacement_safe`: true for insertions; for replacements, the new object is more useful than
  everything removed and loses no distinct caveat, qualification, tradeoff, or relevant detail.

Local acceptance is derived, not model-selected: all five booleans must be true. The model cannot
return replacement text, citations, indexes, or an `accept` override. The payload must contain each
expected operation ID exactly once and no unknown ID. Any malformed or incomplete payload rejects
the entire screen and preserves the validated draft.

## Prompt and Evidence Boundary

The screen prompt contains:

- the untouched official narrative and authenticated blueprint;
- every advisory claim hint and selected evidence passage;
- the complete indexed draft;
- every validated Sol operation, labeled with its stable operation ID;
- only the merged audit cards named by those operations.

It contains no gold nuggets, qrels, prior qualitative verdicts, RAGDoll judgments, old final answer,
or topic-specific expected decision. The prompt explicitly states that the evidence catalog is
factual authority, audit cards are advisory, all cited documents must fully support the complete
object, and Luna must judge but never rewrite.

## Authenticated Replay

The `trec_rag.luna_operation_screen_replay` runner reads a completed hardened bounded-revision
source root without modifying it. Before any hosted call it verifies:

- the bounded state contract and every registered source-file hash;
- handoff and topic-context digests;
- the planner state and original validated draft;
- the registered merged-audit file;
- a semantic-success Sol revision receipt whose model, prompt hash, schema hash, reservation, and
  accepted payload agree with the frozen state;
- normal splice validation of the complete frozen operation set.

The replay identity binds all source hashes, screen prompt/schema hashes, model settings, and run
identity. One Luna semantic reservation is allowed. Resume may reuse a prompt/schema-matching
receipt or retry a terminal transport failure, but it must not repeat an ambiguous or completed
semantic request.

After screening, local code applies only accepted typed operations to the original draft, compacts
references through the existing normalizer, and runs the existing organizer and exact-hint citation
validators. Invalid screen output or invalid final assembly falls back atomically to the validated
draft. There is no model repair call.

## Artifacts and Privacy

Each topic uses a unique ignored config and output namespace. Private state includes the
prompt/schema-bound Luna receipt, source hashes, draft/final submissions, identities, and a
manifest-last aggregate with accepted/rejected counts and non-sensitive reason counts. Prompts,
answers, evidence, operation text, model rationales, and evaluator judgments remain private.

## Frozen Three-Topic Test

Run topics sequentially in this order:

1. `233`: negative probe; inspect whether the weak replacement is rejected and the three useful
   edits survive.
2. `300`: insertion-only positive probe; inspect whether all meaningful additions survive.
3. `499`: mixed insertion/replacement positive probe; inspect whether the substantive gains survive
   without retaining partially supported or redundant material.

Do not tune the prompt between topics. Use Luna medium thinking, exactly one call per topic, and zero
new Sol calls. Judge the sealed results from actual prose and authenticated passages, using existing
post-hoc support artifacts only after generation. Do not rerun the slow nuggetizer unless the
qualitative result is ambiguous.

## Promotion Rule

Promote the operation-screen module for integration only if the frozen screen rejects the known bad
topic-`233` replacement and retains the clearly useful operation sets on `300` and `499` without a
new grounding or coherence failure. If it rejects most useful edits or accepts the weak replacement,
keep the existing hardened output and do not add the screen to the production path.

The initial implementation remains a replay seam; `competition_rag.py` and checked-in competition
configs are outside this change.

## Focused Verification

- Pure schema, exact-ID, derived-acceptance, and fail-closed tests.
- Source receipt/file authentication and resume-safety tests.
- Existing splice and narrative-blueprint focused suites.
- Ruff, Python compilation, and `git diff --check`.
- Three sequential zero-Sol live replays with manifest-last validation.

Do not create a broad new test harness for this bounded experiment.

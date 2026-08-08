# Bounded Splice Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the throwaway full-answer revision with an atomic deterministic splice contract, then run and inspect fresh full-extraction topic `707` only.

**Architecture:** Put the strict splice schema, local validation, and deterministic answer applier in a focused `bounded_splice.py` module. Keep provider orchestration and durable reservation state in `narrative_blueprint_trial.py`, which supplies the immutable draft, known audit-card IDs, and authenticated citation domain. The valid draft remains the fallback for every failed patch or repair.

**Tech Stack:** Python 3.12, pytest, the existing strict OpenRouter JSON generator, existing generation handoff and organizer validators.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-08-bounded-splice-revision-design.md` exactly.
- Work only in the throwaway narrative-blueprint trial path; do not change the supported competition runner or checked-in full-run configs.
- Run topic `707` only during this plan. It must remain sequential and single-topic.
- Use at most 6 splice operations, 4 insertions, and 8 touched original answer objects.
- Use only `delete_count` values `0`, `1`, `2`, or `3`; pure deletion is not supported.
- Preserve every untargeted draft answer object exactly and keep existing reference positions stable.
- Validate patches atomically; never silently drop an invalid operation.
- Permit at most two routine Sol reservations and one deterministic validation-repair reservation per topic.
- A repaired operation may reuse only `new_object` values proposed by the initial splice response.
- Gold, qrels, nuggets, support judgments, and evaluator outputs remain post-hoc and never enter generation prompts.
- Keep live narratives, passages, prompts, answers, provider responses, and judgments under ignored private outputs.
- Use focused red-green tests only; do not add broad production hardening during this experiment.

---

### Task 1: Strict Splice Contract and Deterministic Applier

**Files:**
- Create: `code/trec_rag/bounded_splice.py`
- Create: `code/tests/test_bounded_splice.py`

**Interfaces:**
- Consumes: one validated draft submission record, a parsed provider payload, `allowed_docids: tuple[str, ...]`, and `allowed_audit_card_ids: tuple[str, ...]`.
- Produces: `splice_response_schema() -> dict[str, object]`, `validate_splice_payload(...) -> tuple[SpliceOperation, ...] | None`, `apply_splice_operations(...) -> dict[str, Any]`, and `validate_repaired_splice_payload(...) -> tuple[SpliceOperation, ...] | None`.
- `None` means the model selected `keep_draft`. Validation failures raise `SpliceValidationError` without mutating the draft.

- [ ] **Step 1: Write the failing strict-schema test**

Add a test that imports `splice_response_schema`, asserts the top-level required fields are exactly `decision` and `operations`, asserts each operation requires `start_index`, `delete_count`, `new_object`, and `audit_card_ids`, and asserts raw docid citations have `minItems: 1`, `maxItems: 3`.

- [ ] **Step 2: Run the schema test and verify red**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py::test_splice_schema_is_strict -q
```

Expected: collection fails because `trec_rag.bounded_splice` does not exist.

- [ ] **Step 3: Implement the minimal schema and immutable operation type**

Create `SpliceValidationError(ValueError)` and frozen `SpliceOperation` fields:

```python
start_index: int
delete_count: int
text: str
citations: tuple[str, ...]
audit_card_ids: tuple[str, ...]
```

Implement the strict provider schema with decisions `keep_draft` and `edit`, zero to six operations, `delete_count` enum `[0, 1, 2, 3]`, one non-empty text string, one to three raw string citations, and at least one audit-card ID.

- [ ] **Step 4: Run the schema test and verify green**

Run the command from Step 2. Expected: one passing test.

- [ ] **Step 5: Write failing validation and application tests**

Use a literal draft with three answer objects and two existing references. Add focused tests proving:

1. `keep_draft` accepts only an empty operation array.
2. One insertion, one replacement, and one two-object merge apply against original indexes, append a newly cited authenticated docid once, and leave the one untouched literal answer object equal to its original dictionary.
3. Validation atomically rejects an out-of-range span, overlapping operations, more than six operations, more than four insertions, more than eight touched objects, an unknown audit-card ID, duplicate citations, and a docid outside `allowed_docids`.
4. Repair validation accepts a reordered/subset payload only when each repaired `new_object` exactly equals a `new_object` from the initial parsed payload; changed prose or citations are rejected.

Name each parametrized failure case after the production break it catches. Derive all expected answer arrays and reference positions as literals rather than using applier helpers.

- [ ] **Step 6: Run the focused tests and verify red**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py -q
```

Expected: schema test passes and behavior tests fail because validation/application functions are missing.

- [ ] **Step 7: Implement validation and deterministic application**

Implement these exact behaviors:

- require exact payload and operation field sets;
- reject `bool` where an integer is required;
- `start_index == len(answer)` is valid only for insertion;
- replacement ranges must remain within the original answer;
- reject duplicate start indexes and any insertion positioned inside or at the start of a replacement range;
- count touched objects as the sum of positive `delete_count` values;
- validate non-empty unique citations and audit-card IDs against the supplied allowlists;
- validate the complete operation set before copying or applying anything;
- deep-copy the draft, append newly used docids to its `references` in deterministic operation/input order, convert raw docids to numeric reference indexes, then apply operations from the highest start index downward;
- leave metadata untouched so the orchestration layer can rebind the final run ID;
- verify repaired `new_object` dictionaries against the initial payload by canonical JSON equality before normal validation.

- [ ] **Step 8: Run focused tests and refactor only after green**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py -q
```

Expected: all splice tests pass with no warnings.

- [ ] **Step 9: Commit Task 1**

Stage only the new module and its focused tests, then commit:

```bash
git add code/trec_rag/bounded_splice.py code/tests/test_bounded_splice.py
git commit -m "prototype: add deterministic bounded splice applier"
```

### Task 2: Integrate Splice Revision Into the Throwaway Trial

**Files:**
- Modify: `code/trec_rag/narrative_blueprint_trial.py`
- Modify: `code/tests/test_narrative_blueprint.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes from Task 1: `splice_response_schema`, `validate_splice_payload`, `validate_repaired_splice_payload`, `apply_splice_operations`, and `SpliceValidationError`.
- Produces: one splice-specific revision prompt, deterministic audit-card IDs `a001...`, splice validation/application in `_run_bounded_revision`, and the existing sealed draft/final evaluation artifacts.

- [ ] **Step 1: Write failing prompt and contract-version tests**

Add focused tests that call public or intentionally imported trial helpers and prove:

- merged audit cards are assigned stable IDs `a001`, `a002`, ... without changing their ranking;
- the splice prompt contains indexed draft objects and those card IDs;
- the splice prompt asks for operations and does not include the full-answer output instruction `Return a complete replacement organizer JSON object`;
- the trial contract version differs from `bounded_narrative_revision_trial_v1`.

- [ ] **Step 2: Run the focused integration tests and verify red**

Run the selected new tests with:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_narrative_blueprint.py -k 'splice or audit_card_ids' -q
```

Expected: tests fail because splice prompt/card helpers are absent and the contract remains v1.

- [ ] **Step 3: Separate common evidence context from the full-answer output contract**

Refactor `_render_hybrid_writer_prompt` so a private common-context helper renders through the full citation domain. Keep the existing draft writer prompt byte-for-byte equivalent by appending the existing full-answer instructions in `_render_hybrid_writer_prompt`. Use only the common context in the splice revision prompt.

- [ ] **Step 4: Add stable audit-card IDs and the splice prompt**

After `merge_audit_cards`, add `card_id` values `a001...` in ranked order for the private saved merge and revision prompt. The prompt must present the immutable draft answer with explicit zero-based indexes, describe the splice budgets and semantics, treat cards as advisory, prefer merge/replace to unnecessary insertion, preserve caveats and balance, and permit `keep_draft`.

- [ ] **Step 5: Replace full-answer revision handling with splice handling**

Increment `TRIAL_CONTRACT_VERSION`. Call the revision provider with `splice_response_schema()`. Validate the payload with the topic citation domain and merged card IDs. For `keep_draft`, rebind and seal the draft. For `edit`, apply the patch, rebind the run ID, then run `_validate_generated_submission_record` and `_validate_exact_hint_citations` on the complete candidate.

On `SpliceValidationError` or a complete-candidate validation error, reserve the existing repair role once and request a corrected wrapper/subset using the same splice schema. Require repaired `new_object` values to come from the initial parsed payload. If repair fails, seal the draft. If no parsed initial payload exists, do not spend repair and seal the draft.

Keep the existing reservation ceiling, receipt privacy, state hash chain, and draft/final artifact layout.

- [ ] **Step 6: Run focused integration tests and make them green**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_narrative_blueprint.py -k 'splice or audit' -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Add the minimal README trial command**

Document that bounded revision now uses a splice response, that old v1 state cannot resume, and that each experiment must use a new ignored config/output namespace. Include only the existing one-topic dry-run/create/resume pattern; do not add a production command.

- [ ] **Step 8: Run the focused regression set**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py -q
PYTHONPATH=code .venv/bin/python -m ruff check code/trec_rag/bounded_splice.py code/trec_rag/narrative_blueprint_trial.py code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py
```

Expected: both commands exit zero.

- [ ] **Step 9: Commit Task 2**

Stage only the integration, focused tests, and README, then commit:

```bash
git add code/trec_rag/narrative_blueprint_trial.py code/tests/test_narrative_blueprint.py code/trec_rag/README.md
git commit -m "prototype: use bounded splice revision"
```

### Task 3: Offline Gate and Fresh Topic 707

**Files:**
- Create ignored: `configs/local/bounded-splice-707.yaml`
- Produce ignored: `outputs/rag26-bounded-splice-707-20260808/`
- Modify after evaluation: `docs/superpowers/reports/2026-08-08-bounded-splice-revision-results.md`

**Interfaces:**
- Consumes: the existing sealed generation handoff at `outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/generation_handoff_manifest.json` and the Task 2 trial CLI.
- Produces: one private paired draft/final run for topic `707`, one private call/cost ledger, and one aggregate privacy-reviewed report.

- [ ] **Step 1: Create the ignored single-topic config**

Copy the bounded-revision config shape into `configs/local/bounded-splice-707.yaml` with:

```yaml
experiment:
  id: rag26-bounded-splice-707-20260808
  output_dir: outputs/rag26-bounded-splice-707-20260808
  mode: create
  topic_ids: ["707"]
```

Keep the existing handoff path, team, OpenRouter model, reasoning effort, strict-schema mode,
12,000 maximum output tokens, 900-second timeout, one transport attempt, and concurrency one.

- [ ] **Step 2: Run the dry-run gate**

Run:

```bash
set -a
source .env
set +a
PYTHONPATH=code .venv/bin/python -m trec_rag.narrative_blueprint_trial \
  --config configs/local/bounded-splice-707.yaml --topic 707 \
  --bounded-revision --state-mode create --dry-run
```

Expected: exactly one topic, 4 generated groups, 5 Luna calls total (one planner plus four audits), two routine Sol reservations, one repair-only reservation, and no provider call.

- [ ] **Step 3: Report the live-run scope before spending**

State: topic `707` only; full extraction coverage `1.0`; 4 groups, 58 claim hints, 190 selected passages, and 149 citation-domain documents; expected hosted calls are 5 Luna and 2 Sol, plus at most one validation-only Sol repair; retrieval is reused from the sealed handoff; private output is `outputs/rag26-bounded-splice-707-20260808/`.

- [ ] **Step 4: Run and seal topic 707**

Run the Step 2 command without `--dry-run`. If interrupted after state creation, change the ignored config mode to `resume` and use `--state-mode resume`; do not overwrite or reuse a semantic reservation.

- [ ] **Step 5: Verify the private manifest before evaluation**

Confirm the manifest-last receipt and hash chain, draft/final topic identity, word ceiling, citation domain, call ledger, and splice preservation/application result. Do not open gold or evaluator artifacts during this verification.

- [ ] **Step 6: Perform blinded qualitative comparison before metrics**

Create a private randomized A/B packet containing only the official narrative and the two cited answers. Judge complete-narrative responsiveness, prioritization, explanatory flow, qualifications, redundancy, seam coherence, and grounding. Record the verdict before opening development nugget or support results.

- [ ] **Step 7: Run the existing paired post-hoc evaluation**

Use the same RAGDoll/DeepSeek nuggetizer and citation-support path, model, prompt, and settings used for the preceding three-topic trial. Run one judge pass initially; repeat only if its direction conflicts with qualitative inspection or the pair is close. Keep every raw judgment private.

- [ ] **Step 8: Record aggregate result and decision**

Create `docs/superpowers/reports/2026-08-08-bounded-splice-revision-results.md` with draft-to-final word/object counts, strict and partial vital/all coverage, citation support, qualitative verdict, exact hosted calls and provider cost, validation/fallback behavior, seam observations, and the decision to continue, revise, or stop before topics two and three. Include no narrative, answer, passage, audit-card, prompt, or per-nugget text.

- [ ] **Step 9: Run final verification and commit the aggregate report**

Run the focused tests and Ruff command from Task 2 again, validate the ignored run manifest, run `git diff --check`, and stage only the aggregate report if it contains no private material. Commit:

```bash
git add docs/superpowers/reports/2026-08-08-bounded-splice-revision-results.md
git commit -m "docs: record first bounded splice trial"
```

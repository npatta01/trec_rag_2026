# Planner Deadline Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the six consumed but locally invalid production planner payloads without another planner call, then finish and validate the 119-topic organizer submission.

**Architecture:** Add one deterministic payload normalizer beside the strict blueprint validator. The bounded runner uses it only after strict planner validation fails, persists a normal strict-valid blueprint plus sanitized fallback metadata, and the full-run orchestrator surfaces that metadata as durable warnings while retaining all-or-nothing publication.

**Tech Stack:** Python 3.12, dataclasses/mappings, existing authenticated handoff and bounded-resume state, pytest, AutoJudge.

## Global Constraints

- Reuse the exact six authenticated planner payloads already stored under `outputs/rag26-ms1/`.
- Make no new planner calls and do not change temperature, prompts, models, config identity, or the 113 sealed finals.
- Normalize only 951–1,024 total planning words, non-narrative spans, and duplicate normalized span sets.
- Require the unchanged strict validator to accept every normalized payload before downstream generation.
- Keep the 1,024-word organizer cap, citation domain, exact-hint rules, and all-or-nothing publication unchanged.
- Record every fallback as a durable warning in topic state/manifest and the full-run failure report.

---

### Task 1: Deterministic Strict-Valid Blueprint Recovery

**Files:**
- Modify: `code/trec_rag/narrative_blueprint.py`
- Test: `code/tests/test_narrative_blueprint.py`

**Interfaces:**
- Consumes: `GenerationTopic` plus a provider planner payload that failed `validate_blueprint`.
- Produces: `recover_blueprint_for_deadline(topic, payload) -> tuple[NarrativeBlueprint, tuple[str, ...]]`.

- [ ] **Step 1: Write failing unit tests**

Add tests that require:

```python
blueprint, actions = recover_blueprint_for_deadline(topic, payload)
assert sum(item.target_words for item in blueprint.obligations) == 950
assert "target_words_scaled" in actions
assert validate_blueprint(topic, {"obligations": [o.to_payload() for o in blueprint.obligations]}) == blueprint
```

Add a combined non-narrative/duplicate-anchor case asserting every stored span normalizes to an
official-narrative substring and every normalized span set is unique. Add an unknown-claim-alias
case asserting `BlueprintValidationError` is still raised.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_narrative_blueprint.py -k 'recover_blueprint_for_deadline' -q
```

Expected: collection fails because `recover_blueprint_for_deadline` does not exist.

- [ ] **Step 3: Implement the minimal recovery helper**

Clone the payload. For a total in 951–1,024, allocate exactly 950 words proportionally with
positive integer values and deterministic remainder distribution. Replace non-matching spans
with `topic.narrative`. Disambiguate repeated normalized sets by appending the first exact donor
span from another obligation that is absent from the repeated set. Fail when no donor fits within
the four-span limit. Return only after `validate_blueprint(topic, normalized_payload)` passes.

- [ ] **Step 4: Run unit tests and verify GREEN**

Run the Step 2 command and require all selected tests to pass.

- [ ] **Step 5: Commit the independently tested normalizer**

```bash
git add code/trec_rag/narrative_blueprint.py code/tests/test_narrative_blueprint.py
git commit -m "fix: normalize consumed planner payloads safely"
```

### Task 2: Resume Integration And Durable Warning

**Files:**
- Modify: `code/trec_rag/narrative_blueprint_trial.py`
- Modify: `code/trec_rag/competition_rag_multistage.py`
- Test: `code/tests/test_narrative_blueprint.py`
- Test: `code/tests/test_competition_rag_multistage.py`

**Interfaces:**
- Consumes: the Task 1 helper and an existing `semantic_returned` planner reservation.
- Produces: strict-valid `blueprint.state.json`, `state["planner_fallback"]`, manifest metadata, and one `planner_deadline_fallback` full-run warning per recovered topic.

- [ ] **Step 1: Write failing resume and warning tests**

Create a resume fixture with a recovered invalid planner payload and assert the planner generator
is never invoked, the planner stage seals, and `planner_fallback.actions` is recorded. Extend the
multistage fake completed-topic helper with planner-fallback metadata and assert `failures.json`
contains one warning with kind `planner_deadline_fallback` while the organizer JSONL publishes.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_competition_rag_multistage.py -q
```

Expected: the new fallback-state and warning assertions fail.

- [ ] **Step 3: Integrate recovery without changing identity**

Catch `BlueprintValidationError` only around planner payload validation, invoke
`recover_blueprint_for_deadline`, store the original sanitized validator message plus action list,
then continue through the existing projection/state serialization path. Include
`planner_fallback` in the per-topic manifest. In the multistage loader, translate that state field
to a durable warning without treating it as a failure.

- [ ] **Step 4: Run focused and static verification**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_competition_rag_multistage.py \
  code/tests/test_generation_handoff.py \
  code/tests/test_competition_rag.py -q
git diff --check
```

Require zero failures and no unexpected warnings.

- [ ] **Step 5: Commit the resume integration**

```bash
git add code/trec_rag/narrative_blueprint_trial.py \
  code/trec_rag/competition_rag_multistage.py \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_competition_rag_multistage.py
git commit -m "fix: resume deadline planner fallbacks"
```

### Task 3: Production Resume And Submission Validation

**Files:**
- Read ignored: `configs/local/rag26-ms1-multistage.yaml`
- Read/write private: `outputs/rag26-ms1/`
- Validate: `outputs/rag26-ms1/rag_output_trec_rag_2026.jsonl`

**Interfaces:**
- Consumes: the clean committed fallback implementation and existing identity-bound state.
- Produces: the complete organizer JSONL and validation evidence.

- [ ] **Step 1: Verify clean identity-compatible preflight**

Require a clean tracked tree, config mode `resume`, all 119 authenticated handoff topics, 113
existing finals, six planner payloads awaiting recovery, and unchanged `rag26-ms1-final` identity.

- [ ] **Step 2: Resume and monitor the six topics**

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.competition_rag_multistage \
  --config configs/local/rag26-ms1-multistage.yaml
```

Require 119 final states, six planner fallback warnings, zero failures, and atomic organizer JSONL publication.

- [ ] **Step 3: Run official and project validation**

```bash
.venv/bin/python .agents/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
  --topics trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv \
  --rag outputs/rag26-ms1/rag_output_trec_rag_2026.jsonl
```

Also re-read all 119 rows against the authenticated handoff and require exact order/narratives,
allowed citation domains, exact-hint rules, at most 1,024 words, unique references, run ID
`rag26-ms1-final` of length 15, zero failures, and exactly the six declared fallback warnings.

- [ ] **Step 4: Report the private submission path and final provider ledger**

Report Luna/Sol call counts, provider-reported cost, warning/fallback count, validator status, and
the exact organizer JSONL path without exposing prompts, evidence, responses, or credentials.

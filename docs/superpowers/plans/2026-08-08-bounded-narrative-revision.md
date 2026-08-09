# Bounded Narrative Revision Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway, resumable one-topic driver and use it sequentially on topics `233`, `300`, and `499` to measure whether a cheap evidence-only audit plus one Sol revision improves the first Sol draft.

**Architecture:** The existing throwaway blueprint-trial module gains one bounded-revision mode,
without changing the production runner. It persists every stage and reserves Sol calls before
sending them, keeps the routine path at two Sol calls, and permits a third Sol call only for
deterministic validation repair. After each final candidate is sealed, the existing RAGDoll
adapters score the retained draft and final answer independently.

**Tech Stack:** Python 3.12, existing `GenerationTopic` handoff records, `OpenRouterJsonGenerator`, GPT-5.6 Luna medium, GPT-5.6 Sol, RAGDoll, DeepSeek Flash, JSON/JSONL, Ruff.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-08-bounded-narrative-revision-design.md` verbatim.
- Run only topics `233`, `300`, and `499`, one topic to completion at a time in that order.
- Do not change prompts or contracts between those topics.
- Make at most three Sol semantic reservations per topic across create/resume attempts. The first two roles are draft and revision; the third role is validation repair only.
- Use one Luna-medium planner call and exactly one Luna-medium audit call per authenticated generated group.
- Generation-side code and models never open the TREC run, full-text ZIP, qrels, gold nuggets, RAGDoll output, prior answers, or prior evaluations for these topics.
- Open gold and RAGDoll data only after the topic's final candidate is sealed; never feed them back into generation.
- Do not silently tail-trim an over-limit candidate. More than 1,024 whitespace-separated answer words is a hard validation failure.
- Keep prompts, passages, answers, provider responses, and per-nugget judgments in ignored private output directories.
- Keep the prototype lean: no broad production integration and no new broad test suite. Verify with dry runs, targeted static checks, deterministic candidate validation, and the live experiment.
- Preserve unrelated untracked files and do not push or publish anything.

---

### Task 1: Extend the one-topic blueprint prototype with bounded revision

**Files:**
- Modify: `code/trec_rag/narrative_blueprint_trial.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: one existing RAG YAML config, one authenticated `GenerationTopic`, and `OPENROUTER_API_KEY` loaded through the existing repository environment helper.
- Produces: private stage receipts, a durable state/manifest, separate draft/final evaluation
  submissions, and arm-specific generation identities usable by `trec_rag.ragdoll_io`.
- CLI: `.venv/bin/python -m trec_rag.narrative_blueprint_trial --config CONFIG --topic TOPIC --bounded-revision --state-mode create|resume [--dry-run]`.

- [ ] **Step 1: Define the prototype question and contracts**

Extend the existing throwaway module's question to cover whether a post-draft, evidence-only
omission audit plus one bounded revision improves paired nugget coverage. Define:

```python
TRIAL_CONTRACT_VERSION = "bounded_narrative_revision_trial_v1"
LUNA_MODEL = "openai/gpt-5.6-luna"
LUNA_REASONING_EFFORT = "medium"
MAX_SOL_RESERVATIONS = 3
MAX_AUDIT_CARDS_PER_GROUP = 6
MAX_MERGED_AUDIT_CARDS = 24
```

Add a strict audit response schema and pure helpers with these interfaces:

```python
def audit_response_schema() -> dict[str, object]: ...
def render_group_audit_prompt(
    topic: GenerationTopic,
    *,
    group_id: str,
    draft: dict[str, Any],
) -> str: ...
def validate_group_audit(
    topic: GenerationTopic,
    *,
    group_id: str,
    payload: object,
) -> tuple[dict[str, Any], ...]: ...
def merge_audit_cards(
    topic: GenerationTopic,
    cards_by_group: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[dict[str, Any], ...]: ...
```

The audit schema contains the group alias, concise missing detail, one or more local evidence aliases, `must|should|could`, one omission type from the approved spec, rationale, and an optional zero-based replacement answer index. Validate every alias locally and reject extra keys. Ranking is deterministic: importance, then omission-type specificity, then handoff group/evidence order; exact duplicate normalized details collapse to the earliest card.

- [ ] **Step 2: Render draft, audit, revision, and repair prompts**

Reuse the existing blueprint planner and hybrid full-evidence writer rendering. The revision prompt contains the untouched narrative, authenticated blueprint, every advisory hint and selected passage exactly once, the validated draft, and merged audit cards. It instructs Sol to replace generic or redundant prose rather than append blindly and to return a complete organizer JSON object at or below 1,024 words.

The repair prompt contains the same authenticated writer context, the invalid full candidate, and exact local validator errors. It asks for the smallest full-answer correction and cannot include quality judgments or post-hoc metrics.

- [ ] **Step 3: Add durable stage state and the Sol reservation guard**

Persist state beneath `config.resolved_work_dir / "bounded_revision" / safe_topic_name`. Bind it to the trial contract, handoff manifest digest, topic context digest, config/run identity, model identities, prompt/schema hashes, and topic ID. Write state atomically.

Before each Sol request, append a pending reservation containing a monotonic ordinal and role from
exactly `draft`, `revision`, or `repair`. Refuse a reservation when three already exist; refuse a
`repair` reservation unless deterministic validation produced recorded errors; refuse any other
role after the second Sol reservation. Revalidate all persisted stage hashes before reuse in
`resume` mode. An unresolved pending reservation is consumed after a crash unless a terminal
transport-failure receipt proves that no semantic response was returned.

- [ ] **Step 4: Implement the state machine**

The normal path is:

```text
Luna plan -> Sol draft -> validate/retain draft -> one Luna audit per group
-> deterministic merge -> Sol revision -> validate -> seal final
```

If the draft fails validation, reserve one repair call, repair once, validate, seal if valid, and skip audits plus quality revision. If the revision fails, reserve the third Sol call for repair, validate once, and otherwise fall back to the retained valid draft. If no valid candidate exists, persist a resumable failure and emit no final submission.

Candidate construction may use `build_submission_record` and `normalize_generated_record`; it
must not call `trim_to_word_limit`. Validate with `_validate_generated_submission_record` and
`_validate_exact_hint_citations`. Write the first valid draft and sealed final as separate
single-row evaluation submissions with distinct run IDs ending in `-draft` and `-final`. Write one
compatible generation identity per arm, binding its run ID to the exact handoff and selected topic
context; do not forge or weaken the adapter's identity checks.

- [ ] **Step 5: Capture private call receipts and a manifest-last summary**

For every call persist stage, model, reasoning effort, prompt/schema digest, redacted raw response, usage, latency, reservation ordinal when applicable, and semantic/transport outcome. The manifest reports stage completion, Luna call count, Sol reservations by role, HTTP transport outcomes, words, answer-object counts, output digests, and total provider-reported cost. It must not embed narratives, passages, answers, or raw provider content.

- [ ] **Step 6: Document the one-command usage**

Add a short prototype section to `code/trec_rag/README.md` with the dry-run, create, and resume commands, the private outputs, and the no-gold boundary. State that topics must be run separately and that the driver is not the production competition path.

- [ ] **Step 7: Run lean static and dry-run verification**

Run:

```bash
ruff check code/trec_rag/narrative_blueprint_trial.py
.venv/bin/python -m py_compile code/trec_rag/narrative_blueprint_trial.py
PYTHONPATH=code .venv/bin/python -m trec_rag.narrative_blueprint_trial --config configs/local/bounded-revision-233.yaml --topic 233 --bounded-revision --state-mode create --dry-run
PYTHONPATH=code .venv/bin/python -m trec_rag.narrative_blueprint_trial --config configs/local/bounded-revision-300.yaml --topic 300 --bounded-revision --state-mode create --dry-run
PYTHONPATH=code .venv/bin/python -m trec_rag.narrative_blueprint_trial --config configs/local/bounded-revision-499.yaml --topic 499 --bounded-revision --state-mode create --dry-run
```

Each dry run must show one topic, the expected group/hint/evidence counts, one planner call, one audit call per group, two routine Sol reservations, one repair-only reserve, and no provider call.

- [ ] **Step 8: Commit the prototype**

```bash
git add code/trec_rag/narrative_blueprint_trial.py code/trec_rag/README.md
git commit -m "prototype: add bounded narrative revision trial"
```

---

### Task 2: Run the three topics sequentially

**Files:**
- Create ignored: `configs/local/bounded-revision-233.yaml`
- Create ignored: `configs/local/bounded-revision-300.yaml`
- Create ignored: `configs/local/bounded-revision-499.yaml`
- Produce ignored: `outputs/rag26-bounded-revision-{233,300,499}-20260808/`

**Interfaces:**
- Consumes: the approved development handoff and three unique experiment configs.
- Produces: one sealed draft/final pair and cost ledger per topic.

- [ ] **Step 1: Create isolated local configs**

Copy the current Sol v2 generation settings into three ignored local configs. Give every config a unique experiment ID, output path, work directory, run ID, and `topic_ids` containing exactly its one topic. Point all three at the same authenticated development handoff. Keep the writer model `openai/gpt-5.6-sol`; the prototype owns Luna-medium planner/auditor construction.

- [ ] **Step 2: Perform the preflight**

Confirm the branch, submodule revisions, selected topic ID, empty output namespace, required API key presence without printing it, and the dry-run budget. Report before calling providers:

```text
233: 4 groups, 5 Luna calls, 2 routine + 1 repair-only Sol reserve
300: 4 groups, 5 Luna calls, 2 routine + 1 repair-only Sol reserve
499: 7 groups, 8 Luna calls, 2 routine + 1 repair-only Sol reserve
```

- [ ] **Step 3: Run topic 233 and inspect the seal**

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.narrative_blueprint_trial \
  --config configs/local/bounded-revision-233.yaml --topic 233 \
  --bounded-revision --state-mode create
```

If interrupted, rerun the same command with `--state-mode resume`. Before advancing, read only the
private manifest/validation/cost receipts—not gold or RAGDoll—and confirm a sealed final candidate,
the Sol reservation count, word cap, citation-domain validity, and cost.

- [ ] **Step 4: Run and inspect topic 300**

Repeat Step 3 with the topic-300 config only after topic 233 is sealed. Do not change code, prompts, schemas, or model settings.

- [ ] **Step 5: Run and inspect topic 499**

Repeat Step 3 with the topic-499 config only after topic 300 is sealed. Do not change code, prompts, schemas, or model settings.

---

### Task 3: Evaluate each sealed draft/final pair

**Files:**
- Produce ignored: each topic output's `evaluation/draft/` and `evaluation/final/` trees.

**Interfaces:**
- Consumes: sealed submission pairs only.
- Produces: RAGDoll nugget metrics, selected-evidence citation-support metrics, paired nugget deltas, and evaluation cost receipts.

- [ ] **Step 1: Derive validated RAGDoll inputs after sealing**

For each topic and each arm (`draft`, `final`), run `trec_rag.ragdoll_io` with the authoritative development topics, released gold nuggets, authenticated handoff, and prototype generation identity. Write answers, nuggets, and selected-evidence support inputs into that arm's private evaluation directory. This is the first step allowed to open gold.

- [ ] **Step 2: Run nuggetizer evaluation with one fixed cheap judge**

Run `ragdoll nuggetizer eval` for draft and final using `openrouter/deepseek/deepseek-v4-flash` with `--thinking minimal`. Use the identical pinned RAGDoll revision, model, and settings for all six arms. Require a non-empty topic join and zero failed assignments.

- [ ] **Step 3: Run comparable semantic citation-support judging**

Run the existing RAGDoll support judge against each arm's selected-evidence support input using one fixed inexpensive judge configuration. Validate that every expected task has exactly one completed `FS`, `PS`, or `NS` result, then compute the existing weighted and hard support metrics. Do not substitute full documents or a different evidence view for one arm.

- [ ] **Step 4: Build paired metrics and cost tables**

For each topic calculate draft, final, and final-minus-draft values for strict vital, strict all, partial-credit vital, partial-credit all, words, answer objects, and citation-support metrics. Classify every gained/lost gold nugget using the approved gap taxonomy. Sum planner, audit, Sol draft/revision/repair, nuggetizer, and support-judge calls and provider-reported costs separately.

---

### Task 4: Record the verdict

**Files:**
- Create: `docs/superpowers/reports/2026-08-08-bounded-narrative-revision-results.md`
- Modify: `docs/superpowers/plans/2026-08-08-narrative-blueprint-generation.md`

**Interfaces:**
- Consumes: aggregate/private-safe metrics and inspected paired deltas.
- Produces: the experiment verdict, exact cost ledger, residual-gap taxonomy, and next brainstorm question.

- [ ] **Step 1: Write the aggregate report**

Record the three per-topic comparisons, macro metrics, call counts, costs, validation/repair behavior, semantic citation-support comparison, and whether the approved success rule passed. Include only short privacy-reviewed examples; do not copy passages, full answers, raw prompts, gold files, or provider responses.

- [ ] **Step 2: Update the canonical blueprint plan**

Append the bounded-revision result and the next decision. State explicitly that prompts were frozen across topics and whether any third Sol repair call was used.

- [ ] **Step 3: Verify documentation and repository scope**

Run:

```bash
git diff --check
git status --short
```

Confirm only the prototype, README, aggregate report, and canonical plan are tracked changes; private configs and outputs remain ignored.

- [ ] **Step 4: Commit the result**

```bash
git add docs/superpowers/reports/2026-08-08-bounded-narrative-revision-results.md docs/superpowers/plans/2026-08-08-narrative-blueprint-generation.md
git commit -m "docs: record bounded revision trial results"
```

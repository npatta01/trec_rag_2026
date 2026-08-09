# Luna Whole-Answer Splice Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a throwaway replay that replaces per-group Luna audits plus the second Sol revision with one whole-answer Luna splice call over an authenticated existing draft.

**Architecture:** Add one isolated `trec_rag.luna_splice_replay` module that imports the existing authenticated blueprint loader, provider receipt helper, splice validator/applier, candidate normalizer, and organizer validator. It reads but never mutates a completed bounded-revision source root, writes a new private replay namespace, and falls back atomically to the validated source draft on any unacceptable Luna result.

**Tech Stack:** Python 3.12, asyncio, existing OpenRouter structured-output client, existing bounded-splice primitives, pytest.

## Global Constraints

- Run only development topics `233` and `499`, sequentially.
- Reuse the existing authenticated planner state and validated Sol draft; make zero new Sol calls.
- Make at most one semantic Luna splice call per topic.
- Pass no gold nuggets, qrels, old final answer, old operations, or evaluator output to generation.
- Return at most three operations and preserve the existing atomicity, evidence-link, geometry, word-limit, reference-normalization, and fallback guarantees.
- Keep generated answers, evidence, prompts, responses, evaluations, and caches private and ignored.
- Do not change `competition_rag.py`, checked-in competition configs, or the production runner.
- Add only focused prototype tests.

---

### Task 1: Direct-Evidence Luna Replay Contract

**Files:**
- Create: `code/trec_rag/luna_splice_replay.py`
- Create: `code/tests/test_luna_splice_replay.py`

**Interfaces:**
- Consumes: `GenerationTopic`, authenticated `NarrativeBlueprint`, `BlueprintProjection`, and validated draft record.
- Produces: `luna_splice_response_schema()`, `render_luna_splice_prompt(topic, blueprint, projection, *, draft)`, `_evidence_alias_docids(topic)`, and `assemble_luna_splice_candidate(topic, draft, payload)`.

- [x] **Step 1: Write failing prompt and validation tests**

Create a minimal topic/blueprint/draft fixture. Assert that the prompt contains the full narrative,
selected evidence, immutable draft indexes, `at most three`, `prefer insertion`,
`clearly more useful than everything removed`, and the explicit rule that `audit_card_ids` contains
evidence aliases. Assert it contains neither old audit cards nor an instruction to return a full
answer.

Add a valid insertion using `audit_card_ids: ["e001"]` and its exact linked docid. Assert the
assembled candidate adds the object. Add one payload with four otherwise valid operations and assert
it fails with `at most three operations`.

- [x] **Step 2: Run the selected tests and verify red**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_luna_splice_replay.py -q
```

Expected: collection fails because `trec_rag.luna_splice_replay` does not exist.

- [x] **Step 3: Implement the minimal pure contract**

In `luna_splice_replay.py`, define:

```python
LUNA_SPLICE_REPLAY_CONTRACT_VERSION = "luna_whole_answer_splice_replay_v1"
MAX_REPLAY_OPERATIONS = 3

def luna_splice_response_schema() -> dict[str, object]:
    schema = deepcopy(splice_response_schema())
    schema["properties"]["operations"]["maxItems"] = MAX_REPLAY_OPERATIONS
    return schema

def _evidence_alias_docids(topic: GenerationTopic) -> dict[str, tuple[str, ...]]:
    return {
        f"e{index:03d}": (row.docid,)
        for index, row in enumerate(topic.evidence, start=1)
    }
```

`render_luna_splice_prompt(topic, blueprint, projection, *, draft)` reuses
`_render_hybrid_common_context(topic, blueprint, projection)` and
`_bounded_draft_objects_prompt(draft)`, then appends the frozen direct-splice instructions.

`assemble_luna_splice_candidate(topic, draft, payload)` rejects more than three raw operations
before calling `validate_splice_payload(draft, payload, topic.citation_docids,
_evidence_alias_docids(topic))`. It applies returned operations through
`apply_splice_operations(draft, operations)` and returns the unchanged draft for `keep_draft`. It
does not catch validation errors.

- [x] **Step 4: Run the focused tests and verify green**

Run the Task 1 command. Expected: all new tests pass.

- [x] **Step 5: Commit the contract**

```bash
git add code/trec_rag/luna_splice_replay.py code/tests/test_luna_splice_replay.py
git commit -m "prototype: add Luna whole-answer splice contract"
```

### Task 2: Authenticated One-Call Replay Runner

**Files:**
- Modify: `code/trec_rag/luna_splice_replay.py`
- Modify: `code/tests/test_luna_splice_replay.py`

**Interfaces:**
- Consumes: `--config`, `--topic`, `--source-root`, `--state-mode create|resume`, and optional `--dry-run`.
- Produces: private `work/luna_splice_replay/<topic>/` state, receipt, draft/final submissions, generation identities, and manifest.

- [x] **Step 1: Write failing source-authentication and fallback tests**

Build a temporary source root with current bounded state, registered blueprint state, and draft.
Assert the loader rejects a mismatched handoff digest, topic-context digest, or registered file hash.
Assert malformed replay payload handling returns the rebound validated draft and records draft
fallback without a second provider reservation.

- [x] **Step 2: Run the new tests and verify red**

Run the Task 1 pytest command. Expected: failures identify the missing source loader and fallback
helpers.

- [x] **Step 3: Implement source loading, durable state, and CLI**

Add two public experiment interfaces. `load_replay_source(source_root: Path, *, config: Any,
handoff: GenerationHandoff, topic: GenerationTopic) -> tuple[NarrativeBlueprint,
BlueprintProjection, dict[str, Any], dict[str, str]]` returns the authenticated blueprint,
projection, rebound draft, and immutable source hashes. `run_luna_splice_replay(config: Any,
handoff: GenerationHandoff, topic: GenerationTopic, *, source_root: Path, api_key: str,
state_mode: str) -> Path` writes and returns the private replay root.

The loader calls the existing bounded state/file and blueprint validators, compares handoff and
topic digests, rebinds and validates the draft under the replay identity, and returns immutable
source hashes.

The runner reserves one `whole-answer-splice` Luna stage before `_bounded_provider_call`, binds
recovered payload reuse to prompt/schema hashes, calls `assemble_luna_splice_candidate`,
normalizes/revalidates the candidate, and falls back to the rebound draft on semantic or local
validation error. It writes the final manifest last.

The CLI dry run prints topic/source/counts and
`calls=luna_whole_answer_splice:1,sol:0,provider:0` without creating output directories.

- [x] **Step 4: Run focused verification**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_luna_splice_replay.py \
  code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py -q
ruff check code/trec_rag/luna_splice_replay.py code/tests/test_luna_splice_replay.py
/home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m py_compile code/trec_rag/luna_splice_replay.py
git diff --check
```

- [x] **Step 5: Commit the runner**

```bash
git add code/trec_rag/luna_splice_replay.py code/tests/test_luna_splice_replay.py
git commit -m "prototype: run authenticated Luna splice replay"
```

### Task 3: Two-Topic Replay and Decision

**Files:**
- Create ignored: `configs/local/luna-splice-replay-{233,499}-20260809.yaml`
- Produce ignored: `outputs/rag26-luna-splice-replay-{233,499}-20260809/`
- Create: `docs/superpowers/reports/2026-08-09-luna-splice-replay-results.md`

**Interfaces:**
- Consumes: completed hardened source roots for topics `233` and `499` plus their authenticated handoff.
- Produces: two sealed replay candidates and an aggregate privacy-reviewed verdict.

- [x] **Step 1: Create unique ignored configs and run both dry runs**

Copy the matching hardened local configs, assign replay-specific experiment IDs/output directories,
and keep one topic per config. Confirm each dry run reports exactly one Luna call and zero Sol calls.

- [x] **Step 2: Run topic 233 and inspect before continuing**

Run create mode once. Verify source hashes, receipt outcome, operation count/types, citation linkage,
word count, normalized references, and fallback status. Perform a gold-blind review of whether the
known weak replacement was avoided; only then continue.

- [x] **Step 3: Run and inspect topic 499**

Repeat the same checks and judge whether the replay retains meaningful omissions from the known
positive case without visible seams or weak replacements.

- [x] **Step 4: Evaluate sealed candidates post hoc**

Use the same DeepSeek V4 Flash nuggetizer and citation-support settings as the frozen batch. Compare
draft, replay final, and current full-flow final. Keep all raw artifacts private.

- [x] **Step 5: Record and verify the verdict**

Write the aggregate report with calls, provider cost, validity, qualitative decision, coverage,
citation support, and the final choice among the Luna replay, draft-only flow, or another bounded
investigation. Rerun the focused verification command before committing the report.

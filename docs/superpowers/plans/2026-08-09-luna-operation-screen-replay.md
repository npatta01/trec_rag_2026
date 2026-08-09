# Luna Operation Screen Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test a one-call Luna gate that filters already-validated Sol splice operations, then replay it on frozen topics `233`, `300`, and `499` with zero new Sol calls.

**Status:** Implementation and frozen evaluation complete; PR integration gate blocked by the
repository's existing linked-worktree environment failures. The v2 replay passed all three probes
with 3 Luna calls, 0 Sol calls, and $0.007576 provider cost. One earlier topic-`233` diagnostic cost
$0.002224 and led to the coherent-subset prompt correction documented in the design and result
report.

**Architecture:** Put the strict decision schema and fail-closed operation filtering in a small provider-independent `operation_screen` module. Put authenticated frozen-source loading, prompt rendering, one-call OpenRouter execution, resume safety, final assembly, and manifest writing in a separate replay module. The replay reads but never mutates the source runs and does not change the production competition runner.

**Tech Stack:** Python 3.12, dataclasses, existing OpenRouter strict JSON client, existing bounded-splice and blueprint primitives, pytest.

## Global Constraints

- Use exactly one frozen-v2 Luna medium-thinking semantic call per tested topic and zero new Sol
  calls; preserve the earlier topic-`233` diagnostic as a separately reported iteration cost.
- Run only development topics `233`, `300`, and `499`, sequentially, without prompt changes between topics.
- Luna returns decisions only; it cannot return or modify operation prose, citations, indexes, or ranges.
- Accept an operation only when all five gates are true: fully supported, atomic, material, nonredundant, and replacement-safe.
- Require one decision for every expected operation ID and reject the entire screen payload on missing, duplicate, unknown, malformed, or extra fields.
- Fall back to the validated draft on any invalid screen payload or invalid assembled final.
- Pass no gold nuggets, qrels, old qualitative verdicts, RAGDoll judgments, or old final answer to generation.
- Keep prompts, evidence, answers, provider responses, and evaluator artifacts private and ignored.
- Do not change `competition_rag.py`, checked-in competition configs, or the production runner.
- Add only focused tests.

---

### Task 1: Pure Operation-Screen Contract

**Files:**
- Create: `code/trec_rag/operation_screen.py`
- Create: `code/tests/test_operation_screen.py`

**Interfaces:**
- Consumes: `Sequence[SpliceOperation]` and one provider payload.
- Produces: `operation_screen_response_schema(operation_count: int)`, `operation_ids(operations: Sequence[SpliceOperation])`, `validate_operation_screen_payload(payload: object, operations: Sequence[SpliceOperation]) -> OperationScreenResult`, `ScreenDecision`, and `OperationScreenResult`.

- [x] **Step 1: Write the failing schema and filtering tests**

Create two literal `SpliceOperation` values: one insertion and one replacement. Assert the dynamic
schema requires exactly two strict decision objects. Submit a payload that passes all insertion
gates but marks `replacement_safe` false for the replacement. Assert only the insertion is returned
and the replacement decision records `replacement_safe` as its sole reason.

```python
result = validate_operation_screen_payload(
    {
        "decisions": [
            {
                "operation_id": "op001",
                "fully_supported": True,
                "atomic": True,
                "material": True,
                "nonredundant": True,
                "replacement_safe": True,
            },
            {
                "operation_id": "op002",
                "fully_supported": True,
                "atomic": True,
                "material": True,
                "nonredundant": True,
                "replacement_safe": False,
            },
        ]
    },
    operations,
)
assert result.accepted_operations == (operations[0],)
assert result.decisions[1].rejection_reasons == ("replacement_safe",)
```

- [x] **Step 2: Run the pure tests and verify red**

Run:

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_operation_screen.py -q
```

Expected: collection fails because `trec_rag.operation_screen` does not exist.

- [x] **Step 3: Implement the minimal strict contract**

Define immutable records:

```python
@dataclass(frozen=True)
class ScreenDecision:
    operation_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]

@dataclass(frozen=True)
class OperationScreenResult:
    decisions: tuple[ScreenDecision, ...]
    accepted_operations: tuple[SpliceOperation, ...]
```

`operation_ids` assigns `op001...` by input order. The schema requires exactly
`operation_count` decision objects with no extra fields and boolean gate values. The validator
requires exact root/item fields, exact one-to-one IDs, real booleans, and input values that are
already `SpliceOperation` instances. It derives `accepted = all(gates)` and ordered reason names for
false gates; it never trusts a provider-supplied accept flag.

- [x] **Step 4: Add red tests for malformed identity and types**

Assert that missing, duplicate, unknown, and reordered IDs fail closed; assert extra fields and an
integer masquerading as a boolean fail. Run each new test and confirm the expected validator failure
before adding the corresponding branch.

- [x] **Step 5: Run focused green tests and commit**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_operation_screen.py code/tests/test_bounded_splice.py -q
/home/npatta01/anaconda3/bin/ruff check \
  code/trec_rag/operation_screen.py code/tests/test_operation_screen.py
git diff --check
git add code/trec_rag/operation_screen.py code/tests/test_operation_screen.py
git commit -m "prototype: add strict Luna operation screen"
```

### Task 2: Authenticated One-Call Replay

**Files:**
- Create: `code/trec_rag/luna_operation_screen_replay.py`
- Create: `code/tests/test_luna_operation_screen_replay.py`

**Interfaces:**
- Consumes: `--config`, `--topic`, completed `--source-root`, `--state-mode create|resume`, and optional `--dry-run`.
- Produces: `OperationScreenSource`, `load_operation_screen_source(...)`, `render_operation_screen_prompt(...)`, `finalize_operation_screen(...)`, `run_luna_operation_screen_replay(...)`, private replay state/receipt/evaluation arms, and a manifest-last aggregate.

- [x] **Step 1: Write failing prompt and source-authentication tests**

Build one real temporary bounded source fixture with planner state, draft, merged audit cards, state
reservation/call records, and a revision receipt. Assert the rendered prompt contains the untouched
narrative, indexed draft, stable operation IDs, candidate operations, named audit cards, full
evidence, and the five gates, while instructing decisions-only output.

Assert the loader rejects a changed registered audit file, wrong handoff/topic digest, receipt that
is not `semantic_success`, mismatched revision prompt/schema hash, mismatched revision reservation,
or an invalid frozen splice payload.

- [x] **Step 2: Run replay tests and verify red**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_luna_operation_screen_replay.py -q
```

Expected: collection fails because `trec_rag.luna_operation_screen_replay` does not exist.

- [x] **Step 3: Implement source loading and prompt rendering**

`load_operation_screen_source` authenticates state and registered files, reloads the planner state,
loads the original draft and merged audit cards, recomputes the original Sol revision prompt/schema
hashes, validates the receipt/reservation/call identity, validates its splice payload, then rebinds
the draft to the replay run identity. Return source hashes for state, blueprint, draft, merged audit,
and revision receipt.

`render_operation_screen_prompt` uses the existing full hybrid context and indexed draft renderer,
then serializes each typed operation with its stable ID and only its named audit cards. It states
that every cited document must fully support the complete object, insertions set replacement safety
true, replacements must preserve every removed detail, and Luna must return decisions only.

- [x] **Step 4: Write failing finalization and resume tests**

Assert a valid mixed decision payload applies only accepted operations and preserves rejected draft
objects. Assert malformed decision output returns the rebound draft with a recorded fallback error.
Assert resume reuses a matching semantic-success receipt, retries only a terminal transport failure,
and refuses an ambiguous pending request without a receipt.

- [x] **Step 5: Implement the one-call runner and CLI**

Use the existing `_bounded_provider_call`, candidate writers, generation identities, and local
validators. Reserve one `operation-screen` Luna call, bind the replay identity to prompt/schema and
source hashes, apply only `OperationScreenResult.accepted_operations`, normalize/rebind the final,
and write `manifest.json` last. Report accepted/rejected counts and aggregate reason counts without
copying operation text or evidence. Dry run prints source counts plus
`calls=luna_operation_screen:1,sol:0,provider:0` and creates no output directory.

- [x] **Step 6: Run focused green tests and commit**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python \
  -m pytest code/tests/test_operation_screen.py \
  code/tests/test_luna_operation_screen_replay.py \
  code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py -q
/home/npatta01/anaconda3/bin/ruff check \
  code/trec_rag/operation_screen.py code/trec_rag/luna_operation_screen_replay.py \
  code/tests/test_operation_screen.py code/tests/test_luna_operation_screen_replay.py
/home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m py_compile \
  code/trec_rag/operation_screen.py code/trec_rag/luna_operation_screen_replay.py
git diff --check
git add code/trec_rag/luna_operation_screen_replay.py \
  code/tests/test_luna_operation_screen_replay.py
git commit -m "prototype: replay Luna operation screening"
```

### Task 3: Frozen Three-Topic Test and Verdict

**Files:**
- Create ignored: `configs/local/luna-operation-screen-{233,300,499}-20260809.yaml`
- Produce ignored: `outputs/rag26-luna-operation-screen-{233,300,499}-20260809/`
- Create: `docs/superpowers/reports/2026-08-09-luna-operation-screen-results.md`
- Modify: `docs/superpowers/plans/2026-08-09-luna-operation-screen-replay.md`

**Interfaces:**
- Consumes: the frozen hardened source roots and authenticated shared handoff.
- Produces: three sealed screened candidates plus an aggregate privacy-reviewed decision.

- [x] **Step 1: Create unique ignored configs and dry-run all three topics**

Copy each matching hardened local config, assign a new operation-screen experiment/output ID, and
select exactly the matching topic. Confirm the source root, operation count, prompt size, one Luna
call, zero Sol calls, and no output creation.

- [x] **Step 2: Run topic 233 and inspect before continuing**

Run create mode once. Require a semantic-success manifest, valid source hashes, no fallback, and
organizer-valid final. Inspect the private decision vector and exact prose/evidence. Continue only
after deciding whether the known weak replacement was rejected and the retained subset is a
coherent, defensible improvement; do not require retaining edits whose usefulness depends on an
unsafe replacement.

- [x] **Step 3: Run topic 300 and inspect before continuing**

Repeat under the frozen prompt. Judge whether the insertion-only operation set remains meaningful,
supported, coherent, and materially useful. Do not modify code or prompt based on the result.

- [x] **Step 4: Run topic 499 and inspect**

Repeat under the same prompt. Judge whether the screen retains the substantive full-flow gains and
rejects any operation that is only partially supported, redundant, or replacement-unsafe.

- [x] **Step 5: Perform bounded post-hoc comparison**

After all three candidates seal, compare accepted/rejected operations with the prior gold-blind
qualitative review and existing support judgments. Run the cheap support judge only when a screened
answer creates a combination not already represented by cached exact answer-object tasks. Do not
rerun the nuggetizer unless qualitative judgment is genuinely ambiguous.

- [x] **Step 6: Record, verify, and commit the verdict**

Write aggregate calls, cost, decisions, qualitative outcomes, support results, and promotion
decision without private text. Mark the plan complete. Rerun the focused Task 2 verification,
inspect manifests and git status, then commit only the plan/report files.

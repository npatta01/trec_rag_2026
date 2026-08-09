# Bounded Splice Citation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent bounded splice operations from emitting compound answer objects or citing documents unrelated to their named omission-audit cards.

**Architecture:** Tighten the provider schema and pure validator in `bounded_splice.py`, then have the trial orchestrator derive an authenticated `card_id -> docids` mapping from merged audit cards and topic evidence. Preserve atomic fallback, repair immutability, and the three-Sol ceiling. Evaluate the already frozen topic-897 pair separately after implementation.

**Tech Stack:** Python 3.12, pytest, existing strict OpenRouter schema, existing RAGDoll CLI.

## Global Constraints

- Keep changes inside the throwaway bounded-splice path; do not modify the supported competition runner.
- Do not merge or copy PR #53's priority-aware planner.
- Every splice object is one terminal sentence intended to express one atomic claim.
- Allow one or two citations; one strongest citation is the prompt default.
- Every citation must be in both the topic-wide authenticated domain and the evidence union of the operation's named audit cards.
- Preserve atomic patch rejection, validated-draft fallback, and at most three Sol reservations.
- Keep tests focused; no broad production hardening.
- Frozen topic-897 evaluation is post-hoc and may not enter generation prompts.

---

### Task 1: Harden the Pure Splice Contract

**Files:**
- Modify: `code/trec_rag/bounded_splice.py`
- Modify: `code/tests/test_bounded_splice.py`

**Interfaces:**
- Consumes: immutable draft, parsed splice payload, authenticated topic docids, and `Mapping[str, tuple[str, ...]]` from audit-card ID to linked docids.
- Produces: the existing `SpliceOperation` tuple or `None`, with sentence, citation-count, and card-routing guarantees.

- [ ] **Step 1: Write the failing schema and routing tests**

Add literal tests asserting `splice_response_schema()` caps citations at two and that
`validate_splice_payload` rejects:

```python
{
    "text": "First claim. Second claim.",
    "citations": ["doc-a"],
}
```

as a compound object, rejects three citations, and rejects authenticated `doc-b` when the named
card maps only to `("doc-a",)`. Add one green-path fixture where two named cards map to `doc-a`
and `doc-b` and both citations are accepted.

- [ ] **Step 2: Run the selected tests and verify red**

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py \
  -k 'schema or sentence or card_linked or citation_limit' -q
```

Expected: failures show the current three-citation schema and flat card-ID interface.

- [ ] **Step 3: Implement the minimal validator changes**

Import `Mapping`, change both validator signatures to accept
`audit_card_docids: Mapping[str, tuple[str, ...]]`, and pass that mapping into `_parse_operation`.
After validating card IDs, compute the union of their mapped docids and reject any citation outside
it. Reject mapping entries with no linked docids. Change schema and runtime citation maximum to two.
Add a focused helper that requires one terminal sentence and rejects an internal sentence boundary.

- [ ] **Step 4: Run the focused suite and verify green**

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_bounded_splice.py -q
```

Expected: every splice test passes.

- [ ] **Step 5: Commit Task 1**

```bash
git add code/trec_rag/bounded_splice.py code/tests/test_bounded_splice.py
git commit -m "prototype: harden splice citation routing"
```

### Task 2: Bind Merged Audit Cards to Evidence

**Files:**
- Modify: `code/trec_rag/narrative_blueprint_trial.py`
- Modify: `code/tests/test_narrative_blueprint.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: `GenerationTopic`, merged cards containing `card_id` and `evidence_aliases`, and the existing deterministic audit alias order.
- Produces: `_audit_card_docids(topic, audit_cards) -> dict[str, tuple[str, ...]]`, used identically by initial and repaired splice validation.

- [ ] **Step 1: Write failing routing and prompt tests**

Use the existing three-evidence topic fixture and merged-card helper. Assert the mapping resolves
each card's aliases to exact topic docids in card order, rejects an unknown alias, and rejects an
empty alias list. Extend the revision-prompt test to require the phrases `one atomic claim`,
`one strongest citation`, `at most two`, `complete object`, and `linked to`.

- [ ] **Step 2: Run the selected tests and verify red**

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_narrative_blueprint.py \
  -k 'audit_card_docids or splice_revision_prompt' -q
```

Expected: mapping helper is absent and prompt assertions fail.

- [ ] **Step 3: Implement mapping and orchestration**

Build alias→docid from `_audit_aliases(topic)`. For each merged card, require a nonempty tuple of
known aliases and create an ordered unique docid tuple. Compute the mapping once after
`merge_audit_cards`, then pass it to both `validate_splice_payload` and
`validate_repaired_splice_payload`. Update initial and repair prompt clauses. Increment the trial
contract and prompt-contract versions because resume identity semantics changed.

- [ ] **Step 4: Document the narrower splice contract**

Update only the bounded-revision README paragraph: one atomic sentence/claim, one strongest
citation by default, maximum two, and card-linked document routing.

- [ ] **Step 5: Run both focused suites**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py -q
```

Expected: all tests pass without warnings.

- [ ] **Step 6: Commit Task 2**

```bash
git add code/trec_rag/narrative_blueprint_trial.py code/tests/test_narrative_blueprint.py code/trec_rag/README.md
git commit -m "prototype: bind splice edits to audit evidence"
```

### Task 3: Measure the Frozen Topic-897 Citation Effect

**Files:**
- Produce ignored: topic-897 draft/final support judgments and metrics under the existing private evaluation directories.
- Modify: `docs/superpowers/reports/2026-08-08-bounded-splice-revision-results.md`

**Interfaces:**
- Consumes: frozen draft/final `support_input.jsonl` files only.
- Produces: paired support metrics, evaluator call/cost receipt, and a privacy-reviewed report addendum.

- [ ] **Step 1: Run the identical cheap support judge for each arm**

For draft and final, run `ragdoll support judge` with
`openrouter/deepseek/deepseek-v4-flash`, minimal thinking, separate raw-event/output paths, and the
existing bounded retry behavior. Do not regenerate answers.

- [ ] **Step 2: Validate complete judgments and compute metrics**

Rerun `trec_rag.ragdoll_io` with `--support-judgments` for each arm, then run
`ragdoll support metrics`. Require exactly one completed result per expected citation task and no
conflicts. Use `ragdoll cost` for provider-reported evaluation cost.

- [ ] **Step 3: Record the paired verdict**

Append draft/final/delta values for weighted first-citation precision, weighted all-citation
precision, hard precision, task counts, failures, and cost. Explicitly state that this diagnoses
the old frozen output and does not validate the newly hardened contract.

- [ ] **Step 4: Run final verification**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py -q
/home/npatta01/anaconda3/bin/ruff check \
  code/trec_rag/bounded_splice.py code/trec_rag/narrative_blueprint_trial.py \
  code/tests/test_bounded_splice.py code/tests/test_narrative_blueprint.py
.venv/bin/python -m py_compile \
  code/trec_rag/bounded_splice.py code/trec_rag/narrative_blueprint_trial.py
git diff --check
```

- [ ] **Step 5: Commit the report**

```bash
git add docs/superpowers/reports/2026-08-08-bounded-splice-revision-results.md
git commit -m "docs: record splice citation-support result"
```

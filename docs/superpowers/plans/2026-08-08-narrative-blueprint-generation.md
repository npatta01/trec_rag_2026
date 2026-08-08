# Narrative Blueprint Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and exercise an opt-in narrative-blueprint generation strategy that prioritizes the full official narrative, protects must-level evidence coverage, and never exceeds three Sol semantic completions per topic.

**Architecture:** A new `trec_rag.narrative_blueprint` deep module owns compact planner rendering, strict validation, authenticated alias resolution, must-group widening, deterministic evidence projection, state serialization, and writer-context rendering. `competition_rag` owns provider orchestration, atomic private state, resume identity, and a reservation-before-request semantic-call ledger. The existing one-shot strategy remains byte-for-byte prompt compatible when the optional strategy setting is omitted.

**Tech Stack:** Python 3.12, frozen dataclasses, canonical JSON/SHA-256, asyncio/threaded hosted calls, pytest, OpenRouter strict JSON Schema, authenticated `GenerationTopic` handoff records.

## Bounded revision result — 2026-08-08

The sequential post-draft audit/revision trial completed for topics `233`, `300`, and
`499`; aggregate results and exact spend are in
[`docs/superpowers/reports/2026-08-08-bounded-narrative-revision-results.md`](../reports/2026-08-08-bounded-narrative-revision-results.md).
The first evaluator improved on `233` and `300` and regressed on `499`, but three-run
`499` repeats reversed the apparent direction often enough to invalidate tuning to the
first pass. The current conclusion is uncertainty, not production readiness.

The gold-free structural prototype allowed `233` (anchor margin `+0.062`) and `300`
(`+0.000`) and rejected `499` only for replacement index `37` outside the 36-object
draft range; anchors otherwise accepted all three. Use replacement-index validation and
by-construction deterministic edit operations, with protected anchors as diagnostics or
fallbacks. Keep gold/evaluation data out of generation, cap routine Sol work at draft plus
revision with one optional validation-repair call, build the smallest throwaway edit-operation
contract/applier, and then run one fresh sequential three-topic shadow batch before any
production hardening.

## Prototype-first status — 2026-08-08

Task 1 produced the pure blueprint/projection module and passed its task review. The user then corrected the execution strategy: do not over-engineer production tests before learning whether the approach improves answers. Tasks 2–4 below remain deferred. The bounded revision result above is the current prototype evidence; only behavior that wins a fresh, judge-stable paired evaluation will be solidified in the production runner and tests.

### Three-topic prototype result

The sequential pilot completed on development topics `72`, `200`, and `31`. Each valid generation used one planner plus one writer completion; topic `31` additionally consumed one rejected planner completion in its first namespace, reaching but not exceeding the three-Sol-call topic cap. Coverage was scored afterward with the same fixed-gold RAGDoll/DeepSeek setup as the baseline.

| Topic | Baseline strict vital | Blueprint strict vital | Delta | Baseline strict all | Blueprint strict all | Delta | Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `31` | 0.458333 | 0.541667 | +0.083334 | 0.470588 | 0.558824 | +0.088236 | 351 → 811 |
| `72` | 0.277778 | 0.486111 | +0.208333 | 0.244444 | 0.422222 | +0.177778 | 408 → 855 |
| `200` | 0.524590 | 0.655738 | +0.131148 | 0.477273 | 0.590909 | +0.113636 | 649 → 1,012 |
| **Macro** | **0.420234** | **0.561172** | **+0.140938** | **0.397435** | **0.523985** | **+0.126550** | **469 → 893 avg.** |

Interpretation: controlled length plus explicit narrative obligations is promising across all three topics. The trial does not isolate the extra planner call from the longer answer, although topic `200` improved from an already longer and stronger baseline. Every planner marked every direct narrative obligation `must`, so projection retained the full selected-evidence handoff; the measured gain is from planning and synthesis behavior, not evidence pruning.

The repeated residual gap is loss of high-specificity facts: quantities, named examples, and members of long enumerations. Examples include numeric environmental statistics, named Holocaust camps and Einsatzgruppen, and concrete waste/economy figures. Some of these details regressed from the baseline even as overall coverage improved. The prototype prompt now asks the planner for complementary high-specificity claim aliases and asks the writer to preserve supported lists. It also requires distinct anchors after topic `31` exposed duplicate anchors for compound narrative phrases.

### Matched one-call control

A no-planner control was then run on topics `72` and `200`. It used one writer call, the same controlled-length/detail instruction, every advisory hint once, and every selected passage once. Its prompt size was comparable to the blueprint writer context, so this isolates the obligation plan more fairly than the original duplicated one-shot renderer.

| Topic | Blueprint strict vital | Control strict vital | Blueprint − control | Blueprint strict all | Control strict all | Blueprint − control | Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `72` | 0.486111 | 0.361111 | +0.125000 | 0.422222 | 0.322222 | +0.100000 | 855 / 853 |
| `200` | 0.655738 | 0.737705 | −0.081967 | 0.590909 | 0.625000 | −0.034091 | 1,012 / 972 |
| **Two-topic macro** | **0.570925** | **0.549408** | **+0.021516** | **0.506566** | **0.473611** | **+0.032954** | **934 / 913 avg.** |

The opposing winners explain the mechanism. On topic `72`, planner obligations improved cross-facet mechanisms, human/ecological effects, and policy coverage. On topic `200`, the control preserved concrete enumerations and named facts—most visibly all six named death camps and the Einsatzgruppen—that the selected blueprint claims omitted. The planner's small two-topic macro advantage does not justify treating either current strategy as ideal, especially with only two matched topics.

The next prototype should therefore be a hybrid, still limited to one planner plus one writer: keep the obligation map and word allocation, but render every advisory claim hint once in a global grouped catalog instead of hiding unselected hints. Selected aliases remain the core checklist; unselected aliases are optional specificity candidates. Continue rendering each selected passage once. This directly combines the observed breadth benefit with the observed list/detail benefit without adding a call. Do not productionize or expand tests until that hybrid wins a small held-out trial. Before scaling further, obtain a comparable semantic citation-support sample; current prototype outputs have organizer-valid citations but no paired semantic support judgments.

### Held-out topic `84`: cheap-planner/full-evidence hybrid

Topic `84` was selected from handoff structure without reading its gold nugget text or prior
answers. It contains seven generated groups, 108 advisory claim hints, 355 selected passages,
and 257 citation-domain documents. Gold was opened only after all answer arms were complete;
the topic has 54 gold nuggets, 41 marked vital.

The hybrid writer received every hint and every selected passage exactly once. Planner-selected
aliases were emphasis cues only and did not prune the factual input or citation domain. The
matched control received the same complete hint/evidence inputs without a blueprint.

| Arm | Cheap planner | Selected hints | Planner projection | Words | Strict vital | Strict all | Partial-credit vital | Partial-credit all |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| No-planner control | none | — | — | 1,019 | 0.634146 | 0.574074 | **0.695122** | **0.620370** |
| Mini-plan hybrid | GPT-5 Mini | 64 / 108 | 239 / 355 passages | 970 | **0.634146** | **0.592593** | 0.670732 | **0.620370** |
| Luna-plan hybrid | GPT-5.6 Luna | 82 / 108 | 355 / 355 passages | 1,017 | 0.609756 | 0.500000 | 0.670732 | 0.564815 |

The Mini planner cost `$0.003568` and needed one deterministic local repair: it added a trailing
comma to an otherwise exact narrative anchor. Luna passed the strict planner contract on its
first real request, cost `$0.000751925`, allocated 900 words across seven obligations, and
selected more hints. Two earlier Luna attempts and one Sol attempt failed before a provider
response because the restricted sandbox could not resolve `openrouter.ai`; they are transport
failures, not semantic/model failures. The final topic budget was exactly three successful Sol
writer completions: Mini hybrid, control, and Luna hybrid. No further Sol calls may be made for
topic `84`.

Recorded provider cost for the complete held-out comparison:

| Arm | Planner calls / cost | Sol writer calls / cost | DeepSeek eval windows / cost | Arm total |
|---|---:|---:|---:|---:|
| No-planner control | 0 / `$0.000000000` | 1 / `$0.214967500` | 6 / `$0.003534300` | `$0.218501800` |
| Mini-plan hybrid | 1 / `$0.003568000` | 1 / `$0.248176250` | 6 / `$0.002697800` | `$0.254442050` |
| Luna-plan hybrid | 1 / `$0.000751925` | 1 / `$0.247291250` | 6 / `$0.003136560` | `$0.251179735` |
| **Total** | **2 / `$0.004319925`** | **3 / `$0.710435000`** | **18 / `$0.009368660`** | **`$0.724123585`** |

The ledger counts 23 successful semantic calls. Three additional attempts failed during DNS
resolution before any model response (two Luna planner transports and one Sol writer transport),
so they have no recorded provider usage or cost and are not counted as semantic completions.

The Mini/control delta came from only five gold nuggets. Mini gained full support for childhood
hepatitis B (vital) and toxoid vaccines (non-vital), but lost adult shingles (vital) and partial
support for benefit-versus-rare-adverse-events and hesitancy-linked disease resurgence. The
Mini plan selected the hepatitis B hint and labeled shingles optional, demonstrating that the
`selected core` / `optional` distinction creates zero-sum attention even when all evidence is
present.

Luna's broader selection restored full support for hepatitis B, shingles, and hesitancy-linked
disease resurgence and added partial support for perceived low disease risk and benefit/risk.
It nevertheless lost exact coverage elsewhere: recombinant, polysaccharide, and conjugate
vaccine mechanisms became partial; several safety statements disappeared; toxoid and named
animal non-core examples disappeared. Selecting more claims before drafting therefore does not
solve the compression problem; it changes which facts are sacrificed.

Nine vital nuggets were unsupported by both Mini and control. A literal selected-passage scan
found no obvious support for eight of them (complacency, perceived low disease risk, polio near
eradication, the 2–3 million annual-deaths figure, the 99% U.S. measles reduction, and three
specific animal-vaccine facts). One passage weakly mentioned vaccination inconvenience. This
scan is not a semantic evidence judge, but it indicates that part of the residual gap is upstream
retrieval/canonicalization rather than generation. Generation must not invent those facts.

**Decision:** do not productionize either hybrid yet. On the next fresh held-out topic, use:

1. one cheap Luna planner for narrative obligations and word allocation, with claim selections
   treated as non-exclusive focus suggestions rather than a selected/optional partition;
2. one Sol draft over every hint and every selected passage;
3. one cheap post-draft omission audit against the narrative, all hints, and selected evidence
   (never gold), returning only evidence-backed missing-detail cards;
4. one Sol revision that replaces generic or redundant prose with the best audited details while
   remaining at or below 1,024 words.

This design uses two Sol calls per topic, stays below the three-call cap, and attacks the observed
failure at the point where it is measurable: after seeing what the draft actually omitted. Keep
tests light until this revision loop wins on fresh topics. Semantic citation-support comparison
remains outstanding; all three topic-`84` outputs passed structural citation validation only.

## Global Constraints

- Prototype only one development topic at a time; completed topics are `31`, `72`, `200`, and
  held-out `84`. Never start an all-topic run, and make no further Sol calls for topic `84`.
- Use at most one planner and two writer semantic reservations per topic across every resume; never make a fourth Sol semantic call.
- Count and report HTTP transport attempts separately from semantic reservations.
- Planner inputs are limited to the untouched narrative, generated group text, and existing advisory claim-hint text with local aliases.
- Planner and writer never open the TREC run, full-text ZIP, qrels, gold nuggets, RAGDoll scores, or prior answers.
- Selected passages are factual authority; claim hints and blueprint labels are advisory.
- A `must` obligation widens to all selected passages in every represented group; `should` and `could` remain claim-linked.
- The existing `selected_evidence_one_shot_v1` path and checked-in full-run configs remain unchanged by default.
- Persist validated blueprint state and semantic reservations privately under the experiment's dedicated `work/` directory; keep outputs and provider responses out of git.
- Preserve all unrelated user changes and untracked files.
- Follow red-green-refactor: every production behavior is preceded by a test observed failing for the intended reason.

---

### Task 1: Build the narrative-blueprint deep module

**Files:**
- Create: `code/trec_rag/narrative_blueprint.py`
- Create: `code/tests/test_narrative_blueprint.py`

**Interfaces:**
- Consumes: authenticated `trec_rag.generation_handoff.GenerationTopic`.
- Produces:
  - `BLUEPRINT_CONTRACT_VERSION: str`
  - `BlueprintValidationError(ValueError)`
  - `NarrativeBlueprint` and `BlueprintProjection` frozen dataclasses
  - `planner_response_schema() -> dict[str, object]`
  - `render_planner_prompt(topic: GenerationTopic) -> str`
  - `validate_blueprint(topic: GenerationTopic, payload: object) -> NarrativeBlueprint`
  - `project_blueprint(topic: GenerationTopic, blueprint: NarrativeBlueprint) -> BlueprintProjection`
  - `render_blueprint_writer_context(topic: GenerationTopic, blueprint: NarrativeBlueprint, projection: BlueprintProjection) -> str`
  - `serialize_blueprint_state(topic: GenerationTopic, blueprint: NarrativeBlueprint, projection: BlueprintProjection, *, planner_prompt_sha256: str, writer_context_sha256: str) -> dict[str, object]`
  - `load_blueprint_state(topic: GenerationTopic, payload: object, *, planner_prompt_sha256: str) -> tuple[NarrativeBlueprint, BlueprintProjection]`

- [ ] **Step 1: Write failing tests for compact rendering and aliases**

Create a real two-group `GenerationTopic` fixture with three claims, linked and unlinked passages, and recognizable sentinel strings. Assert deterministic aliases and absence of every authority-bearing sentinel:

```python
def test_planner_prompt_is_compact_and_contains_no_authoritative_evidence() -> None:
    topic = _topic_fixture()

    prompt = render_planner_prompt(topic)

    assert topic.narrative in prompt
    assert "g001" in prompt and "g002" in prompt
    assert "c001" in prompt and "c003" in prompt
    assert topic.groups[0].text in prompt
    assert topic.claim_hints[0].text in prompt
    for evidence in topic.evidence:
        assert evidence.text not in prompt
        assert evidence.evidence_id not in prompt
        assert evidence.docid not in prompt
    for claim in topic.claim_hints:
        assert claim.claim_id not in prompt
```

- [ ] **Step 2: Run the compact-rendering test and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_narrative_blueprint.py::test_planner_prompt_is_compact_and_contains_no_authoritative_evidence -q
```

Expected: collection/import failure because `trec_rag.narrative_blueprint` does not exist.

- [ ] **Step 3: Implement only aliases, prompt rendering, and provider schema**

Define stable handoff-order aliases and a strict schema with 3–8 obligations and these exact fields:

```python
ANSWER_MODES = frozenset({"describe", "explain", "compare", "evaluate", "recommend", "enumerate"})
PRIORITIES = frozenset({"must", "should", "could"})
MIN_ALLOCATED_WORDS = 850
MAX_ALLOCATED_WORDS = 950
```

The renderer must serialize only narrative/group/claim text with `gNNN` and `cNNN` aliases.

- [ ] **Step 4: Run the compact-rendering test and verify GREEN**

Run the command from Step 2. Expected: one passing test.

- [ ] **Step 5: Write failing table-driven validation tests**

Cover one valid composite obligation set plus literal invalid mutations:

```python
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(obligations=p["obligations"][:2]), "3-8"),
        (lambda p: [o.update(priority="should") for o in p["obligations"]], "must"),
        (lambda p: p["obligations"][0].update(answer_mode="invent"), "answer_mode"),
        (lambda p: p["obligations"][0].update(narrative_spans=["not present"]), "narrative"),
        (lambda p: p["obligations"][0].update(selected_claim_aliases=["c999"]), "claim alias"),
        (lambda p: p["obligations"][0].update(selected_claim_aliases=[]), "claim"),
        (lambda p: p["obligations"][0].update(target_words=1), "850-950"),
    ],
)
def test_blueprint_validation_fails_closed(mutate, message) -> None:
    payload = copy.deepcopy(_valid_payload())
    mutate(payload)
    with pytest.raises(BlueprintValidationError, match=message):
        validate_blueprint(_topic_fixture(), payload)
```

Add a positive test proving a list of disjoint spans matches after NFKC normalization, case folding, and whitespace collapse. Add a negative test for duplicate normalized span sets and duplicate aliases.

- [ ] **Step 6: Run validation tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_narrative_blueprint.py -q
```

Expected: failures because `validate_blueprint` and dataclasses are not implemented.

- [ ] **Step 7: Implement strict validation and immutable blueprint records**

Use exact-key validation, `unicodedata.normalize("NFKC", text).casefold()`, and whitespace collapse. Validate every invariant from the design, derive native claim IDs locally, and retain the advisory label without treating it as evidence.

- [ ] **Step 8: Run validation tests and verify GREEN**

Run the command from Step 6. Expected: all current blueprint tests pass.

- [ ] **Step 9: Write failing projection and persistence tests**

Assert the exact evidence IDs for these behaviors:

```python
def test_must_widens_to_full_selected_groups_but_should_stays_claim_linked() -> None:
    topic = _topic_fixture()
    blueprint = validate_blueprint(topic, _valid_payload())

    projection = project_blueprint(topic, blueprint)

    assert projection.obligations[0].evidence_ids == (
        "g1-linked", "g1-unlinked"
    )
    assert projection.obligations[1].evidence_ids == ("g2-linked",)
    assert projection.evidence_ids == (
        "g1-linked", "g1-unlinked", "g2-linked"
    )
```

Also test stable deduplication, derived citation docids, obligation mapping in the writer context, no duplicated passage blocks, canonical state round-trip, and rejection after changing topic context, planner prompt hash, obligation content, evidence mapping, or writer-context hash.

- [ ] **Step 10: Run projection tests and verify RED**

Run the full new test file. Expected: failures because projection/rendering/state functions are missing.

- [ ] **Step 11: Implement projection, writer context, and authenticated state round-trip**

Derive all claim/evidence/docid mappings from the supplied `GenerationTopic`; never accept those identifiers from provider payloads. Preserve topic evidence order in the global catalog, render each passage once, and include per-obligation evidence aliases.

- [ ] **Step 12: Run Task 1 tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_narrative_blueprint.py -q
```

Expected: all tests pass with no warnings.

- [ ] **Step 13: Commit Task 1**

```bash
git add code/trec_rag/narrative_blueprint.py code/tests/test_narrative_blueprint.py
git commit -m "feat: add narrative blueprint planning module"
```

---

### Task 2: Add opt-in configuration, identity, and durable call accounting

**Files:**
- Modify: `code/trec_rag/competition_rag.py`
- Modify: `code/tests/test_competition_rag.py`

**Interfaces:**
- Consumes: Task 1 constants, planner schema/prompt, and blueprint state functions.
- Produces:
  - `RagGenerationConfig.strategy: str`
  - optional YAML `generation.strategy`
  - private per-topic ledger at `work/call_ledger/<safe-topic>.json`
  - `_reserve_semantic_call(path: Path, *, topic_id: str, strategy: str, context_sha256: str, stage: Literal["planner", "writer"]) -> int`
  - `_finish_semantic_call(path: Path, *, reservation: int, outcome: Literal["completed", "failed"]) -> None`

- [ ] **Step 1: Write failing strict-config tests**

Add tests beside existing config cases:

```python
def test_generation_strategy_defaults_to_existing_one_shot(tmp_path: Path) -> None:
    assert load_rag_generation_config(_write_config(tmp_path, _config_text())).strategy == "selected_evidence_one_shot_v1"

def test_generation_strategy_accepts_only_blueprint_v1(tmp_path: Path) -> None:
    config = load_rag_generation_config(_write_config(
        tmp_path,
        _config_text(generation_extra="  strategy: narrative_blueprint_v1\n"),
    ))
    assert config.strategy == "narrative_blueprint_v1"
```

Reject booleans, unknown strings, and duplicate strategy keys.

- [ ] **Step 2: Run strict-config tests and verify RED**

Run the named tests. Expected: unknown-key failures or missing `strategy` attribute.

- [ ] **Step 3: Implement the optional strategy setting**

Add exact constants:

```python
ONE_SHOT_STRATEGY = "selected_evidence_one_shot_v1"
BLUEPRINT_STRATEGY = "narrative_blueprint_v1"
GENERATION_STRATEGIES = frozenset({ONE_SHOT_STRATEGY, BLUEPRINT_STRATEGY})
```

Default only when the key is absent; otherwise require a nonempty supported string.

- [ ] **Step 4: Run config tests and verify GREEN**

Run all config-loading tests in `test_competition_rag.py`.

- [ ] **Step 5: Write failing call-ledger tests**

Use a real temporary path and assert reservation-before-request behavior, exact stage caps, cross-resume persistence, strict payload validation, and no fourth reservation:

```python
def test_blueprint_call_ledger_caps_total_semantic_reservations_at_three(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    identity = _ledger_identity()
    assert _reserve_semantic_call(ledger, identity=identity, stage="planner") == 1
    assert _reserve_semantic_call(ledger, identity=identity, stage="writer") == 2
    assert _reserve_semantic_call(ledger, identity=identity, stage="writer") == 3
    with pytest.raises(ValueError, match="semantic.*budget"):
        _reserve_semantic_call(ledger, identity=identity, stage="writer")
```

Mutate topic ID, strategy, context hash, reservation ordinal, stage, and status to prove fail-closed loading.

- [ ] **Step 6: Run ledger tests and verify RED**

Expected: missing helper failures.

- [ ] **Step 7: Implement atomic semantic reservations**

The JSON record contains exact keys `schema_version`, `topic_id`, `strategy`, `context_sha256`, and `reservations`. Each reservation contains exact keys `ordinal`, `stage`, and `outcome`; `outcome` starts as `reserved`. Write the reservation atomically before scheduling `generator.complete_json`. Planner count may not exceed one, writer count may not exceed two, and total may not exceed three.

- [ ] **Step 8: Run ledger tests and verify GREEN**

Run only the ledger tests, then the existing interruption/resume tests to ensure one-shot behavior is unchanged.

- [ ] **Step 9: Write failing identity tests**

Assert that enabling blueprint mode changes generation identity; Task 1's prompt/schema/contract hashes appear only for blueprint mode; and changing any of them refuses resume. Assert the one-shot topic prompt hash remains the existing `render_prompt(topic)` hash.

- [ ] **Step 10: Implement strategy-aware generation identity**

Bump `identity_version` once. Preserve the handoff `PROMPT_CONTRACT_VERSION` for one-shot/RAGDoll compatibility, and add Task 1's separate blueprint contract identity. Do not include generated blueprint bytes in the pre-call identity; authenticate those through per-topic blueprint state.

- [ ] **Step 11: Run Task 2 tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_competition_rag.py -q
```

Expected: the complete existing runner test file and new tests pass.

- [ ] **Step 12: Commit Task 2**

```bash
git add code/trec_rag/competition_rag.py code/tests/test_competition_rag.py
git commit -m "feat: persist blueprint generation call budget"
```

---

### Task 3: Integrate planner, narrowed writer, and resume behavior

**Files:**
- Modify: `code/trec_rag/competition_rag.py`
- Modify: `code/tests/test_competition_rag.py`

**Interfaces:**
- Consumes: Task 1 blueprint functions and Task 2 strategy/ledger.
- Produces: blueprint state at `work/blueprints/<safe-topic>.json`; stage-specific sanitized provider receipts; organizer rows generated from the derived citation domain.

- [ ] **Step 1: Write a sequence generator and failing happy-path integration test**

Use a test generator whose outputs are ordered per topic rather than keyed to one payload. Assert planner call precedes writer, schemas differ, the writer prompt contains obligation/evidence mapping, and one valid planner plus one valid writer produces the organizer row and private blueprint state.

```python
assert [call["response_schema"] for call in generator.calls] == [
    planner_response_schema(), output_schema()
]
assert generator.calls[0]["topic_id"] == "rag2026-1"
assert "FROZEN SELECTED RETRIEVAL EVIDENCE" not in generator.calls[0]["user_prompt"]
assert "NARRATIVE BLUEPRINT" in generator.calls[1]["user_prompt"]
```

- [ ] **Step 2: Run the happy-path test and verify RED**

Expected: only the existing writer is called.

- [ ] **Step 3: Implement one planner call and the narrowed writer context**

In blueprint mode:

1. load a valid persisted blueprint when present;
2. otherwise reserve the sole planner call, call the generator with planner system/user prompts and schema, validate/project, persist state, and finish the reservation;
3. render the writer prompt from that state;
4. restrict generated citation validation to `projection.citation_docids`;
5. keep `build_submission_record()` and organizer output shape unchanged.

Reuse the existing semaphore, executor, redaction, atomic JSON writer, and semantic retry instruction.

- [ ] **Step 4: Run the happy-path test and verify GREEN**

Expected: two calls, one private blueprint, one valid organizer row.

- [ ] **Step 5: Write failing failure/resume/budget tests**

Add behavior tests proving:

- invalid planner output reserves one call, writes a sanitized failure receipt, makes no writer call, and a resume makes no new planner call;
- valid persisted blueprint plus missing row resumes with writer only;
- tampered/mismatched blueprint state refuses resume without any provider call;
- planner plus one invalid writer plus one valid writer uses exactly three reservations;
- planner plus two invalid writers cannot make a fourth call on resume;
- a process-interruption simulation after reservation still consumes that slot;
- a saved organizer row in blueprint mode is reusable only with its valid blueprint state and projected citation domain;
- foreign docids or evidence outside the projection fail local validation;
- configured API-key text is absent from planner raw/error/state/ledger artifacts.

- [ ] **Step 6: Run the new integration tests and verify RED**

Expected: resume/budget/state failures expose the missing integration behavior.

- [ ] **Step 7: Implement fail-closed resume and global per-topic caps**

Count existing reservations rather than resetting budgets on every invocation. If a planner reservation exists without valid state, return a precise error. The existing one-shot invocation-local retry behavior stays unchanged; the hard cross-resume ledger applies to blueprint mode.

- [ ] **Step 8: Run all runner and handoff tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_competition_rag.py \
  code/tests/test_generation_handoff.py -q
```

Expected: all tests pass, including unchanged one-shot/resume tests.

- [ ] **Step 9: Commit Task 3**

```bash
git add code/trec_rag/competition_rag.py code/tests/test_competition_rag.py
git commit -m "feat: add narrative blueprint generation strategy"
```

---

### Task 4: Document and dry-validate the three-topic experiment

**Files:**
- Modify: `code/trec_rag/README.md`
- Create locally but do not commit: `configs/local/rag26_competition_rag_gpt_sol_blueprint-31-72-200-20260808.yaml`

**Interfaces:**
- Consumes: completed blueprint strategy and the existing sealed handoff.
- Produces: reproducible private trial command and a zero-call dry-validation receipt.

- [ ] **Step 1: Add README instructions beside the existing competition RAG smoke workflow**

Document the optional strategy, private artifact layout, hard semantic reservation budget, fail-closed resume rules, and an explicit warning that blueprint mode is experimental and must use a subset config before broader validation.

- [ ] **Step 2: Create the ignored local trial config**

Copy the existing three-topic baseline config and change exactly:

```yaml
experiment:
  id: rag26_competition_rag_gpt_sol_blueprint-31-72-200-20260808
  output_dir: outputs/rag26_competition_rag_gpt_sol_blueprint-31-72-200-20260808
  mode: create
  topic_ids: ["31", "72", "200"]
generation:
  strategy: narrative_blueprint_v1
```

Keep the sealed handoff path, Sol model, reasoning effort, schema mode, token limit, timeout, and transport policy fixed to the paired baseline.

- [ ] **Step 3: Run zero-call config and prompt validation**

Use `.venv/bin/python` to load the local config and handoff, select topics, render each planner prompt, and print only topic ID, group count, hint count, prompt bytes, estimated tokens, and expected maximum semantic reservations. Do not print narratives, hints, passages, IDs, secrets, or full prompts.

Expected:

- exactly topics `31`, `72`, `200`;
- compact planner prompts around 2–3k tokens;
- no network access or provider invocation;
- maximum three semantic reservations per topic.

- [ ] **Step 4: Run targeted and portability tests**

```bash
.venv/bin/python -m pytest \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_competition_rag.py \
  code/tests/test_generation_handoff.py \
  code/tests/test_environment_portability.py -q
```

- [ ] **Step 5: Verify the tracked diff excludes private artifacts**

```bash
git status --short
git diff --check
git diff -- code/trec_rag/README.md
```

Confirm no `outputs/`, `configs/local/`, provider response, corpus passage, gold nugget, qrel, RAGDoll, `.env`, or user-owned untracked file is staged or added.

- [ ] **Step 6: Commit Task 4 documentation only**

```bash
git add code/trec_rag/README.md
git commit -m "docs: document narrative blueprint trial"
```

---

### Task 5: Execute and evaluate the authorized three-topic trial

**Files:**
- Private generated artifacts only under `outputs/rag26_competition_rag_gpt_sol_blueprint-31-72-200-20260808/`
- No tracked file changes.

**Interfaces:**
- Consumes: ignored local config, authenticated handoff, `OPENROUTER_API_KEY`, and existing development-only evaluation tooling.
- Produces: private organizer-shaped answers, private call ledgers/blueprints, and paired development diagnostics.

- [ ] **Step 1: Preflight without printing secrets**

Verify the selected topic tuple, unique nonexisting output namespace, clean relevant tracked diff, initialized submodules at recorded commits, and presence—not value—of `OPENROUTER_API_KEY`.

- [ ] **Step 2: Report the run budget before starting**

Report to the user:

- 3 selected topics;
- zero retrieval/reranking/canonicalization calls and full sealed-handoff reuse;
- expected 6 Sol semantic reservations, hard maximum 9;
- up to three HTTP transport attempts per semantic reservation under the existing policy;
- exact private output directory;
- no all-topic run.

- [ ] **Step 3: Run blueprint generation**

```bash
.venv/bin/python -m trec_rag.competition_rag \
  --config configs/local/rag26_competition_rag_gpt_sol_blueprint-31-72-200-20260808.yaml
```

Do not use overwrite. If interrupted, inspect authenticated state and use `mode: resume`; never delete or regenerate a consumed planner reservation.

- [ ] **Step 4: Validate organizer output and call budgets**

Load the output through the repository validator. Confirm exact topic IDs/narratives, projected citation domains, 1–3 unique citations under the repository's strict generation profile, and at most 1,024 words. Inspect each private ledger and report semantic reservations by stage plus sanitized transport-attempt counts.

- [ ] **Step 5: Run paired development-only evaluation**

Use the same pinned RAGDoll assignment and selected-passage support workflow and settings used by the baseline. Evaluation may read development gold only after generation has finished; generation itself never reads it. Reuse validated judge caches where identities match, and report every hosted evaluation cache miss and call before authorizing it.

- [ ] **Step 6: Produce the paired comparison**

Report per topic and macro:

- answer word count;
- strict-vital and strict-all;
- weighted first-citation and all-judged precision;
- Full/Partial/No Support counts;
- retained passage/docid/word counts;
- planner obligations and priority counts without exposing private passage text;
- actual planner/writer semantic reservations and transport attempts.

Evaluate every mechanism-success gate from the design. Trace newly matched nuggets to blueprint obligations for mechanism diagnosis. A positive result authorizes only a proposal for broader development validation; it does not authorize another run.

# Retrieval Nugget Coverage Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a year-neutral, two-call evaluator that scores how completely one sealed topic's canonical retrieval nuggets cover a frozen narrative-derived obligation plan.

**Architecture:** One deep module owns typed records, strict planner/judge schemas, scoring, OpenRouter transport, authenticated handoff loading, private persistence, resume, and its CLI. The existing competition debug-report skill remains a thin cache-first router to that CLI. All behavior is developed with injected fake backends; hosted calls are opt-in and never needed by tests.

**Tech Stack:** Python 3.13, standard-library dataclasses/typing/json/hashlib/argparse/pathlib/urllib, existing `trec_rag.generation_handoff` authentication, pytest/unittest, Markdown agent skill.

## Global Constraints

- Implement the approved spec at `docs/superpowers/specs/2026-08-07-retrieval-nugget-coverage-evaluator-design.md`; do not broaden it.
- The evaluator is year-neutral: no 2025 data, gold nuggets, qrels, topic constants, calibration paths, or hidden calibration behavior.
- Read only the authenticated generation handoff's topic narrative and ordered `claim_hints`; never read selected passages, evidence IDs, docids, query text, generation output, gold, or qrels.
- Version 1 assumes canonical retrieval nuggets faithfully represent selected passages and must state that limitation in `report.json`.
- Exactly one narrative-only planner call and one all-nugget judge call per topic; no second planner, semantic repair, sharding, truncation, or ranking.
- Hosted calls require `--allow-hosted-calls`; default execution is cache-only.
- Use a 1,000,000-byte rendered judge-request ceiling and fail before calling the backend when exceeded.
- Keep all work artifacts private and outside git. Stdout receipts and errors must contain no narrative, nugget, obligation, span, or credential text.
- Use `create` and `resume` modes only. Never implement `overwrite` or delete state.
- Preserve unrelated work. The assigned worker owns only the files listed below.

## File Structure

- Create `code/trec_rag/retrieval_nugget_coverage.py`: the complete deep module and `python -m` CLI.
- Create `code/tests/test_retrieval_nugget_coverage.py`: domain, prompt, backend, persistence, handoff, and CLI tests.
- Create `code/tests/test_retrieval_nugget_coverage_skill.py`: agent-skill routing and privacy contract tests.
- Modify `.agents/skills/trec-rag-competition-debug-report/SKILL.md`: add the year-neutral retrieval nugget coverage route.
- Modify `code/trec_rag/README.md`: document inputs, outputs, execution modes, limitations, and validation.

---

### Task 1: Pure plan, judgment, and scoring domain

**Files:**
- Create: `code/trec_rag/retrieval_nugget_coverage.py`
- Create: `code/tests/test_retrieval_nugget_coverage.py`

**Interfaces:**
- Produces: `CoverageNugget`, `EvaluatorIdentity`, `FrozenPlan`, `CoverageJudgment`, `CoverageReport`, `NuggetCoverageError`.
- Produces: `validate_and_freeze_plan(narrative: str, payload: object) -> FrozenPlan`.
- Produces: `validate_judgments(plan: FrozenPlan, nuggets: Sequence[CoverageNugget], payload: object) -> tuple[CoverageJudgment, ...]`.
- Produces: `score_coverage(plan: FrozenPlan, nuggets: Sequence[CoverageNugget], judgments: Sequence[CoverageJudgment], identity: EvaluatorIdentity) -> CoverageReport`.

- [ ] **Step 1: Add failing tests for planner validation and local identifiers**

Create fixtures whose valid planner payload follows this exact shape:

```python
VALID_PLAN = {
    "schema_version": "retrieval_nugget_plan_v1",
    "facets": [
        {
            "title": "Costs and assumptions",
            "obligations": [
                {
                    "requirement": "Describe the projected cost and its assumptions.",
                    "support_test": "A figure and the assumptions used to derive it are present.",
                    "kind": "required_explicit",
                    "narrative_spans": ["cost and assumptions"],
                },
                {
                    "requirement": "Add useful historical context.",
                    "support_test": "A relevant historical comparison is present.",
                    "kind": "supplemental_inferred",
                    "narrative_spans": [],
                },
            ],
        }
    ],
    "unmapped_narrative_spans": [],
}
```

Assert exact-key validation, facet/obligation bounds, the 40-obligation cap, exact narrative-substring spans, empty supplemental spans, one required obligation, locally assigned `f001`/`f001-o001` IDs, deterministic canonical bytes, and deterministic SHA-256.

- [ ] **Step 2: Run the planner tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k 'plan or span or identifier' -q
```

Expected: collection/import failures because the module and symbols do not exist.

- [ ] **Step 3: Implement the typed domain and plan validator**

Start with immutable records and one typed exception:

```python
class NuggetCoverageError(ValueError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class CoverageNugget:
    nugget_id: str
    text: str


@dataclass(frozen=True)
class EvaluatorIdentity:
    schema_version: str
    planner_prompt_version: str
    judge_prompt_version: str
    planner_model: str
    judge_model: str
```

Add private exact-mapping, stripped-text, control-character, canonical-JSON, and SHA-256 helpers. Validate the approved `retrieval_nugget_plan_v1` schema and assign local IDs only after the model payload passes validation.

- [ ] **Step 4: Add failing judgment and scoring tests**

Use this exact valid judgment payload:

```python
VALID_JUDGMENTS = {
    "schema_version": "retrieval_nugget_judgment_v1",
    "judgments": [
        {
            "obligation_id": "f001-o001",
            "label": "partial",
            "supporting_nugget_aliases": ["n001"],
            "missing_elements": "The assumptions are missing.",
        },
        {
            "obligation_id": "f001-o002",
            "label": "full",
            "supporting_nugget_aliases": ["n002"],
            "missing_elements": "",
        },
    ],
}
```

Assert one judgment per obligation, no unknown/duplicate IDs or aliases, full/partial support requirements, unsupported empty support, `missing_elements` consistency, required-facet macro averaging, `strict_full_rate`, separate nullable supplemental coverage, resolved canonical `nugget_id`s, and uncited aliases that do not lower the score.

- [ ] **Step 5: Run the new tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k 'judgment or score or uncited' -q
```

Expected: failures for undefined validation and scoring functions.

- [ ] **Step 6: Implement judgment validation and deterministic scoring**

Use the approved mapping and facet macro-average:

```python
_LABEL_VALUE = {"full": 1.0, "partial": 0.5, "unsupported": 0.0}

required_facet_scores = [
    sum(_LABEL_VALUE[row.label] for row in required_rows) / len(required_rows)
    for required_rows in required_rows_by_facet
]
required_coverage = sum(required_facet_scores) / len(required_facet_scores)
```

Resolve local aliases to canonical retrieval nugget IDs in Python before constructing `CoverageReport`. Store full-precision floats; rounding belongs only in the CLI receipt.

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k 'plan or span or identifier or judgment or score or uncited' -q
git add code/trec_rag/retrieval_nugget_coverage.py code/tests/test_retrieval_nugget_coverage.py
git commit -m "Build retrieval nugget coverage domain"
```

Expected: selected tests pass and the commit contains only the two Task 1 files.

---

### Task 2: Structured prompts, injected backends, and the two-call evaluator

**Files:**
- Modify: `code/trec_rag/retrieval_nugget_coverage.py`
- Modify: `code/tests/test_retrieval_nugget_coverage.py`

**Interfaces:**
- Consumes: Task 1 domain records and validators.
- Produces: `CoverageModelRequest`, `CoverageModelBackend.complete(request) -> BackendReply` protocol.
- Produces: `render_planner_request(...)`, `render_judge_request(...)`.
- Produces: `evaluate_nugget_coverage(...) -> tuple[FrozenPlan, tuple[CoverageJudgment, ...], CoverageReport, EvaluationCallMetadata]`.

- [ ] **Step 1: Add failing tests for request isolation and strict schemas**

Implement a recording fake:

```python
class RecordingBackend:
    def __init__(self, replies: list[BackendReply]) -> None:
        self.replies = list(replies)
        self.requests: list[CoverageModelRequest] = []

    def complete(self, request: CoverageModelRequest) -> BackendReply:
        self.requests.append(request)
        return self.replies.pop(0)
```

Assert the planner request contains the narrative but no nugget ID/text; the judge request contains the narrative, frozen obligation IDs, and every `nNNN: claim text` in input order; neither request contains canonical claim IDs, passages, docids, ranks, or importance. Assert the response JSON schemas use exact keys and alias/obligation `enum`s.

- [ ] **Step 2: Run prompt tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k 'request or schema or recording' -q
```

Expected: failures for missing request/backend interfaces.

- [ ] **Step 3: Implement requests and the backend protocol**

Define a backend-neutral request record:

```python
@dataclass(frozen=True)
class CoverageModelRequest:
    stage: Literal["planner", "judge"]
    model: str
    messages: tuple[Mapping[str, str], ...]
    response_schema_name: str
    response_schema: Mapping[str, object]
    max_tokens: int


class CoverageModelBackend(Protocol):
    def complete(self, request: CoverageModelRequest) -> BackendReply: ...
```

Keep fixed system prompts as versioned module constants. Render compact deterministic JSON in user messages. Enforce the 1,000,000-byte judge-request cap on the actual serialized provider request body before invoking `complete`.

- [ ] **Step 4: Add failing orchestration tests**

Assert a successful evaluation calls the planner once and judge once; a malformed planner prevents the judge call; an oversized judge payload prevents the judge call; and no semantic repair call occurs for either malformed response. Check safe metadata is returned separately from raw response bodies.

- [ ] **Step 5: Implement the two-call pure evaluator**

Expose the spec signature:

```python
def evaluate_nugget_coverage(
    *,
    narrative: str,
    nuggets: Sequence[CoverageNugget],
    planner: CoverageModelBackend,
    judge: CoverageModelBackend,
    identity: EvaluatorIdentity,
) -> EvaluationResult:
    ...
```

Validate inputs before calls, run the exact two-stage sequence, preserve safe `BackendReply.metadata`, and reject empty nuggets before calling either backend.

- [ ] **Step 6: Run Task 2 tests and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k 'request or schema or recording or orchestration or oversized' -q
git add code/trec_rag/retrieval_nugget_coverage.py code/tests/test_retrieval_nugget_coverage.py
git commit -m "Add structured nugget coverage evaluation"
```

Expected: Task 2 tests pass with exactly two fake backend calls on the success path.

---

### Task 3: Authenticated handoff adapter, private persistence, resume, OpenRouter, and CLI

**Files:**
- Modify: `code/trec_rag/retrieval_nugget_coverage.py`
- Modify: `code/tests/test_retrieval_nugget_coverage.py`

**Interfaces:**
- Consumes: `load_generation_handoff`, `select_generation_topics`, Task 2 evaluator.
- Produces: `coverage_input_from_handoff(path: Path, topic_id: str) -> BoundCoverageInput`.
- Produces: `run_coverage_evaluation(config: CoverageRunConfig, planner: CoverageModelBackend | None = None, judge: CoverageModelBackend | None = None) -> CoverageRunReceipt`.
- Produces: `OpenRouterCoverageBackend` and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Add failing handoff-adapter tests**

Build or reuse a canonical `GenerationHandoff` fixture and write it with `write_generation_handoff`. Assert `load_generation_handoff` authentication is used, exactly one selected topic is required, claim-hint order becomes stable `n001…` aliases, and only `claim_id` plus `text` enter `CoverageNugget`. Assert unknown topic, altered manifest, empty narrative, and empty `claim_hints` fail before model calls.

- [ ] **Step 2: Implement the thin authenticated adapter**

The adapter must remain visibly narrow:

```python
handoff = load_generation_handoff(manifest_path)
topic, = select_generation_topics(handoff, [topic_id])
nuggets = tuple(
    CoverageNugget(nugget_id=hint.claim_id, text=hint.text)
    for hint in topic.claim_hints
)
```

Bind `handoff.manifest_sha256`, `topic.narrative_sha256`, topic ID, and ordered nugget text hashes into `BoundCoverageInput`. Do not read any other `ClaimHint` field.

- [ ] **Step 3: Add failing create/resume and redaction tests**

Assert artifact order and contents:

```text
input.json
plan.json
judgments.json
report.json
manifest.json  # last
```

Test that the omitted `--work-dir` resolves beneath the handoff manifest's parent as specified; `create` refuses a non-empty work directory; cache-only names missing planner/judge stages and calls no backend; resume reuses valid stages with zero backend calls; changed handoff/narrative/nugget order, prompt version, model, or artifact bytes fail closed; report and manifest hashes reproduce; stdout receipt/error contains none of the fixture narrative, nugget, obligation, span, or fake credential strings.

- [ ] **Step 4: Implement atomic canonical persistence and resume**

Use canonical JSON and no-replace atomic writes modeled after `generation_handoff.py`:

```python
def _publish_once(path: Path, payload: Mapping[str, object]) -> str:
    body = _canonical_json_bytes(payload)
    # Write a same-directory temporary file, fsync it, link/no-replace publish,
    # verify identical existing bytes, then fsync the directory.
    return sha256(body).hexdigest()
```

Write `manifest.json` only after every prior artifact validates. Persist raw model response bodies only in the private work directory if the repository pattern requires them; never include them in stdout or the final safe manifest. Cache-only errors are typed expected-state errors, not transport failures.

- [ ] **Step 5: Add failing OpenRouter and CLI tests**

Inject a fake transport and assert HTTPS OpenRouter endpoint, bearer credential redaction, strict `json_schema` response format, safe provider metadata, accepted `finish_reason == "stop"`, bounded transient transport retries only, no semantic retries, and reflection detection. Exercise `main()` with `capsys` for cache-only, create, resume, and safe non-zero errors.

- [ ] **Step 6: Implement OpenRouter backend and CLI**

Follow the existing `facet_extraction.py` transport/envelope helpers where their contracts fit. The CLI parser must expose only:

```text
--handoff-manifest PATH
--topic TOPIC_ID
--work-dir PATH
--planner-model NAME
--judge-model NAME
--mode create|resume
--allow-hosted-calls
```

Load `OPENROUTER_API_KEY` only when a cache miss plus `--allow-hosted-calls` actually requires a hosted stage. Print exactly one safe JSON receipt or safe JSON error and return `0`/non-zero accordingly.

- [ ] **Step 7: Run Task 3 and full module tests, then commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -q
.venv/bin/python -m trec_rag.retrieval_nugget_coverage --help
git add code/trec_rag/retrieval_nugget_coverage.py code/tests/test_retrieval_nugget_coverage.py
git commit -m "Add resumable retrieval nugget coverage CLI"
```

Expected: all module tests pass without network, secrets, or GPU; help exits zero.

---

### Task 4: Agent-skill route, documentation, and regression verification

**Files:**
- Modify: `.agents/skills/trec-rag-competition-debug-report/SKILL.md`
- Create: `code/tests/test_retrieval_nugget_coverage_skill.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: the Task 3 CLI and safe receipt contract.
- Produces: a year-neutral, cache-first operator workflow discoverable from the existing skill.

- [ ] **Step 1: Write failing skill-contract tests**

Assert the new section includes:

```python
for required in (
    "Retrieval Nugget Coverage",
    "-m trec_rag.retrieval_nugget_coverage",
    "--handoff-manifest",
    "--topic",
    "--allow-hosted-calls",
    "cache-only",
    "one narrative",
    "canonical retrieval nugget text",
    "maximum of two hosted calls",
):
    assert required in section
```

Also isolate the new section and assert it contains none of `2025`, `gold`, `qrels`, retrieval/reranking/generation commands, passage upload, serving, or publishing permission.

- [ ] **Step 2: Run skill test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage_skill.py -q
```

Expected: failure because the route has not been added.

- [ ] **Step 3: Add the narrow cache-first skill workflow**

Document exactly two commands: cache-only first, then the identical command with `--allow-hosted-calls` after explicit authorization. Require the skill to state provider, planner/judge model identities, maximum two calls, and payload categories before egress. Explicitly forbid retrieval, reranking, generation, passage egress, other topics/models/providers, publication, or serving.

- [ ] **Step 4: Add the README section**

Document the CLI flags, manifest input, private artifact bundle, create/resume semantics, cache-only default, scoring formula, core assumption, honest limits, and targeted test command. Do not include real narrative/nugget text or year-specific examples.

- [ ] **Step 5: Run targeted and neighboring regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_generation_handoff.py -q
```

Expected: all tests pass without hosted calls.

- [ ] **Step 6: Run hygiene checks and commit documentation**

Run:

```bash
rg -n '2025|gold|qrels' code/trec_rag/retrieval_nugget_coverage.py code/tests/test_retrieval_nugget_coverage.py
git diff --check
git status --short
git add .agents/skills/trec-rag-competition-debug-report/SKILL.md \
  code/tests/test_retrieval_nugget_coverage_skill.py code/trec_rag/README.md
git commit -m "Document retrieval nugget coverage workflow"
```

Expected: the first command returns no year/gold/qrels coupling in evaluator code or its domain tests; the skill-contract test may name those forbidden terms only to assert their absence from the isolated workflow section. Diff check is clean; only Task 4 files enter the commit.

---

### Task 5: Final implementation verification

**Files:**
- Verify only; modify implementation-owned files solely to fix verified failures.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: evidence that the implementation satisfies the spec before independent Claude review.

- [ ] **Step 1: Re-read the spec and map every acceptance criterion to a passing test**

Record the mapping in the plan's verification notes or final worker report. Do not add speculative features for deferred items.

- [ ] **Step 2: Run the complete targeted suite**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_generation_handoff.py -q
```

Expected: all pass.

- [ ] **Step 3: Run static repository checks**

Run:

```bash
.venv/bin/python -m trec_rag.retrieval_nugget_coverage --help
git diff --check origin/master...HEAD
git status --short --branch --ignore-submodules=all
```

Expected: help exits zero, diff check is clean, and the worktree is clean on `codex/retrieval-nugget-coverage-eval`.

- [ ] **Step 4: Commit only if verification required a fix**

```bash
git add code/trec_rag/retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  .agents/skills/trec-rag-competition-debug-report/SKILL.md code/trec_rag/README.md
git commit -m "Fix retrieval nugget coverage verification gaps"
```

Expected: no commit when verification found no defect; otherwise one focused fix commit.

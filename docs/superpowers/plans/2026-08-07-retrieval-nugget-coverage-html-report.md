# Retrieval Nugget Coverage HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private, year-neutral, self-contained HTML report with a run overview and deep-linkable detailed topic views for completed retrieval nugget coverage evaluations.

**Architecture:** Add a read-only completed-bundle loader to the evaluator, then build a separate report module that validates decomposition context, aggregates topics, and renders deterministic HTML. Keep hosted evaluation behavior unchanged; the report consumes only authenticated handoff text, validated coverage artifacts, and hash-checked decomposition text.

**Tech Stack:** Python 3.12, frozen dataclasses, stdlib JSON/HTML/hash/path/tempfile tooling, pytest, standalone semantic HTML/CSS/JavaScript, headless Chromium.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-07-retrieval-nugget-coverage-html-report-design.md` exactly.
- Use strict test-driven development: write each behavior test first, run it to observe the expected failure, then add the minimum production code.
- The renderer makes zero hosted calls and performs no searches, reranking, generation, or network access.
- Coverage judgments use only canonical nugget text; retrieval subnarratives and BM25 queries are labeled unevaluated context.
- Never include passages, document IDs, ranks, scores, provider responses, credentials, qrels, organizer gold nuggets, or RAG output in HTML.
- The HTML is standalone and deterministic: no external assets, current timestamp, random ID, or data-derived `innerHTML`.
- The report supports System/Light/Dark themes, accessible contrast and focus, keyboard operation, mobile reflow, browser history, and light print output.
- Preserve exact narrative, subnarrative, query, obligation, gap, and canonical nugget text; escape it only at the HTML boundary.
- Keep source under `code/`, tests under `code/tests/`, report documentation beside `code/trec_rag/README.md`, and skill guidance in the existing competition debug-report skill.
- Workers are not alone in the codebase: preserve unrelated edits and adapt to already committed work; never revert another task's changes.

---

### Task 1: Read-only completed coverage bundle loader

**Files:**
- Modify: `code/trec_rag/retrieval_nugget_coverage.py`
- Modify: `code/tests/test_retrieval_nugget_coverage.py`

**Interfaces:**
- Consumes: existing `coverage_input_from_handoff`, `_load_plan_artifact`, `_load_judgments_artifact`, `score_coverage`, `_report_payload`, `_validate_manifest`, and canonical artifact helpers.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class CompletedCoverageEvaluation:
    bound_input: BoundCoverageInput
    identity: EvaluatorIdentity
    plan: FrozenPlan
    judgments: tuple[CoverageJudgment, ...]
    report: CoverageReport
    artifact_hashes: Mapping[str, str]
    manifest_sha256: str


def load_completed_coverage_evaluation(
    *,
    handoff_manifest_path: Path,
    topic_id: str,
    work_dir: Path,
) -> CompletedCoverageEvaluation:
    """Validate and load a complete coverage bundle without writing or calling a backend."""
```

- The function is the only report-facing entry point into evaluator persistence.

- [ ] **Step 1: Write failing happy-path and read-only tests**

Build a complete bundle with the existing injected fake backends, snapshot the
five artifact bytes and directory entries, call
`load_completed_coverage_evaluation`, and assert:

```python
assert loaded.bound_input.topic_id == "topic-coverage"
assert loaded.plan.obligations == expected_plan.obligations
assert loaded.judgments == expected_judgments
assert loaded.report.required_coverage == 0.5
assert loaded.artifact_hashes.keys() == {
    "input.json", "plan.json", "judgments.json", "report.json"
}
assert len(loaded.manifest_sha256) == 64
assert snapshot_after == snapshot_before
```

The test must make both fake backends raise if invoked after fixture setup; the
loader receives no backend parameter and must not call one.

- [ ] **Step 2: Run the happy-path test and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py \
  -k 'load_completed_coverage_evaluation and read_only' -q
```

Expected: collection or assertion failure because the public loader does not
exist.

- [ ] **Step 3: Add the immutable result record and identity decoder**

Add `CompletedCoverageEvaluation` next to the existing run receipt. Add a private
manifest identity decoder that accepts exactly:

```python
{
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "planner_prompt_version": PLANNER_PROMPT_VERSION,
    "judge_prompt_version": JUDGE_PROMPT_VERSION,
    "planner_model": nonblank_text,
    "judge_model": nonblank_text,
}
```

Reject missing/extra keys, stale schema/prompt identities, unsafe controls, and
blank models with `NuggetCoverageError("persistence", ...)`.

- [ ] **Step 4: Implement the minimal read-only validation path**

The loader must:

1. derive `BoundCoverageInput` from the authenticated handoff;
2. reject a symlink/non-directory work root and any missing or symlinked required
   artifact;
3. load canonical `manifest.json` and reconstruct the identity;
4. validate `input.json`, planner, judgments, and recomputed report;
5. compare `report.json` to `_report_payload(score_coverage(...))`;
6. recompute the four artifact hashes and call `_validate_manifest` with exactly
   two completed stages and persisted safe provider metadata; and
7. return the frozen record with a mapping proxy and manifest file digest.

Do not call `_publish_once`, `run_coverage_evaluation`, or any backend constructor.

- [ ] **Step 5: Run the happy-path test and verify GREEN**

Run the Step 2 command. Expected: selected tests pass.

- [ ] **Step 6: Add failing corruption and path-safety tests**

Add parameterized cases for:

- changed `input.json`, `plan.json`, `judgments.json`, and `report.json`;
- changed planner or judge request SHA;
- stale evaluator/prompt identity;
- changed manifest artifact hash or completed-stage count;
- missing artifact;
- unknown supporting nugget ID; and
- symlinked work directory or required artifact.

Each case snapshots the directory and asserts failure leaves every byte and entry
unchanged.

- [ ] **Step 7: Run corruption tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py \
  -k 'load_completed_coverage_evaluation' -q
```

Expected: new edge cases fail until all invariants are enforced.

- [ ] **Step 8: Complete the validator and verify GREEN**

Add only the checks needed by the failing cases, reusing existing validators.
Run the Step 7 command and then:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -q
```

Expected: all evaluator tests pass.

- [ ] **Step 9: Self-review and commit**

Verify no persistence path writes and no evaluator run behavior changed. Commit:

```bash
git add code/trec_rag/retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage.py
git commit -m "Load completed coverage evaluations"
```

---

### Task 2: Report data model, topic discovery, decomposition validation, and aggregation

**Files:**
- Create: `code/trec_rag/retrieval_nugget_coverage_report.py`
- Create: `code/tests/test_retrieval_nugget_coverage_report.py`

**Interfaces:**
- Consumes: `CompletedCoverageEvaluation`,
  `load_completed_coverage_evaluation`, `load_generation_handoff`,
  `select_generation_topics`, `Topic`, and `load_validated_decomposition`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class RetrievalSubnarrativeContext:
    subnarrative_id: str
    text: str
    bm25_queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalPlanContext:
    used_fallback: bool
    subnarratives: tuple[RetrievalSubnarrativeContext, ...]
    planner_identity: Mapping[str, object]
    manifest_sha256: str
    result_sha256: str


@dataclass(frozen=True, slots=True)
class CoverageReportTopic:
    evaluation: CompletedCoverageEvaluation
    retrieval_plan: RetrievalPlanContext


@dataclass(frozen=True, slots=True)
class CoverageRunSummary:
    topic_count: int
    nugget_count: int
    required_obligation_count: int
    supplemental_obligation_count: int
    topic_macro_required_coverage: float
    topic_macro_strict_full_rate: float
    label_counts: Mapping[str, int]
    perfect_required_topic_count: int


@dataclass(frozen=True, slots=True)
class CoverageReportData:
    topics: tuple[CoverageReportTopic, ...]
    summary: CoverageRunSummary


def load_coverage_report_data(
    *,
    handoff_manifest_path: Path,
    coverage_root: Path,
    topic_ids: Sequence[str] = (),
) -> CoverageReportData: ...
```

- [ ] **Step 1: Write failing literal aggregation tests**

Construct two `CoverageReportTopic` values with hand-derived scores and counts.
The expected summary is literal, not computed with production helpers:

```python
assert summary.topic_count == 2
assert summary.nugget_count == 5
assert summary.required_obligation_count == 3
assert summary.supplemental_obligation_count == 1
assert summary.topic_macro_required_coverage == 0.75
assert summary.topic_macro_strict_full_rate == 0.625
assert dict(summary.label_counts) == {
    "full": 2, "partial": 1, "unsupported": 1
}
assert summary.perfect_required_topic_count == 1
```

- [ ] **Step 2: Run aggregation test and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py \
  -k 'summary' -q
```

Expected: failure because the report module does not exist.

- [ ] **Step 3: Add the frozen report records and pure summarizer**

Add the records above plus:

```python
def summarize_coverage_topics(
    topics: Sequence[CoverageReportTopic],
) -> CoverageRunSummary: ...
```

Reject an empty sequence. Count obligation kinds from `plan.obligations`, sum
the report's three label counts, and use `math.fsum` for topic-macro means.
Wrap mappings in `MappingProxyType`.

- [ ] **Step 4: Run aggregation test and verify GREEN**

Run Step 2. Expected: summary tests pass.

- [ ] **Step 5: Write failing decomposition-validation tests**

Use a real saved decomposition fixture and checkpoint manifest. Cover:

- valid subnarratives preserve exact text, IDs, order, and BM25 queries;
- original-only fallback returns zero subnarratives with `used_fallback=True`;
- noncanonical JSON;
- manifest/result filename, byte-count, or SHA disagreement;
- missing/extra manifest fields or malformed planner identity;
- wrong topic or exact narrative;
- symlinked topic, manifest, or result; and
- a topic root that escapes the retrieval output root.

- [ ] **Step 6: Run decomposition tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py \
  -k 'decomposition or retrieval_plan' -q
```

Expected: failures because no report-context loader exists.

- [ ] **Step 7: Implement strict decomposition loading**

Add a private canonical-JSON reader capped at 2 MiB. Resolve topic paths beneath
the handoff manifest parent, reject symlinks and non-regular files, validate
checkpoint manifest fields and canonical bytes, recompute result byte count and
SHA-256, require a JSON-object planner identity with only safe scalar/list/map
data, then call `load_validated_decomposition(Topic(...), result_path)`.

Project only subnarrative ID, exact text, and exact BM25 query strings. Do not
load retrieval, scoring, canonical, passage, document, or provider-response
artifacts.

- [ ] **Step 8: Run decomposition tests and verify GREEN**

Run Step 6. Expected: all decomposition tests pass.

- [ ] **Step 9: Write failing discovery and selection tests**

Build a handoff with topic order `("t2", "t1")` and matching complete coverage
directories. Assert:

- no selectors returns `("t2", "t1")`;
- repeated selectors preserve explicit order;
- unknown or duplicate selectors fail;
- an unknown directory containing a coverage manifest fails;
- any known topic directory containing some coverage artifacts but no manifest
  fails as incomplete;
- an empty discovery fails; and
- unrelated regular files are ignored.

- [ ] **Step 10: Implement `load_coverage_report_data` and verify GREEN**

Authenticate the handoff once for ordering and topic validation. Discover only
safe child directories. For every selected topic call the Task 1 loader and the
strict decomposition loader, then build the frozen tuple and summary.

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py -q
```

Expected: all report-data tests pass.

- [ ] **Step 11: Self-review and commit**

Confirm no forbidden artifact path is opened in production or tests. Commit:

```bash
git add code/trec_rag/retrieval_nugget_coverage_report.py \
  code/tests/test_retrieval_nugget_coverage_report.py
git commit -m "Load retrieval coverage report data"
```

---

### Task 3: Deterministic standalone HTML and interactive topic views

**Files:**
- Modify: `code/trec_rag/retrieval_nugget_coverage_report.py`
- Modify: `code/tests/test_retrieval_nugget_coverage_report.py`

**Interfaces:**
- Consumes: `CoverageReportData` from Task 2.
- Produces:

```python
def render_coverage_report_html(data: CoverageReportData) -> bytes:
    """Render deterministic, standalone UTF-8 HTML for validated report data."""
```

- [ ] **Step 1: Write failing semantic-content and escaping tests**

Render a controlled topic containing `<script>`, `&`, quotes, U+2028, and U+2029
in every user/model-derived text field. Assert:

- each exact value is visible after HTML parsing as text;
- no unescaped hostile tag exists;
- every obligation and canonical nugget appears exactly once in its canonical
  inventory and every mapped nugget appears under its obligation;
- the output contains no document ID, passage fixture sentinel, provider body,
  external `src=`, external `href=`, fetch/XHR/WebSocket/EventSource call, or
  data-derived `innerHTML` assignment; and
- two renders of the same data are byte-identical.

- [ ] **Step 2: Run renderer tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py \
  -k 'render or escape or deterministic' -q
```

Expected: failure because the renderer is absent.

- [ ] **Step 3: Implement the static semantic document**

Render with `html.escape(..., quote=True)` and stable iteration order. Include:

- `<!doctype html>`, UTF-8 and viewport metadata, descriptive title, skip link,
  and a main landmark;
- run metric cards and the four explicit limitations from the coverage report;
- search, status filter, and sort controls;
- one overview button per topic, default-ordered by required coverage then topic
  ID;
- one hidden detail section per topic with summary, narrative disclosure,
  separately labeled retrieval-plan disclosure, ordered facets/obligations,
  exact support tests/spans/gaps, mapped local nugget aliases/text, collapsed
  complete nugget inventory, and provenance hashes;
- the first partial or unsupported obligation open, all others closed; and
- unique, topic-prefixed HTML IDs.

Do not embed raw JSON; server-render all text so JavaScript only changes state.

- [ ] **Step 4: Add theme and responsive CSS**

Define every color through shared custom properties. Implement light defaults,
`prefers-color-scheme: dark`, `[data-theme="light"]`, and
`[data-theme="dark"]`. Add:

- status text plus shape/icon, not color alone;
- visible `:focus-visible` outlines;
- one-column mobile layout without fixed widths;
- reduced-motion fallback; and
- `@media print` forcing the light palette, hiding controls, and showing all
  disclosures and the selected detail.

- [ ] **Step 5: Add state-only JavaScript**

Use `textContent`, DOM attributes, `hidden`, and node reordering only. Implement:

- guarded early theme restore from local storage;
- Light/System/Dark buttons with `aria-pressed` and guarded persistence;
- search and filters (`all`, `has gaps`, `perfect`, `unsupported`);
- sort (`required coverage`, `strict full`, `topic ID`);
- `#topic=<encoded id>` deep links;
- `history.pushState` for selection and `popstate`/`hashchange` restoration;
- invalid-fragment fallback to overview with a live-region message; and
- back-to-overview control.

No script may initiate network I/O or inject data-derived HTML.

- [ ] **Step 6: Run renderer tests and verify GREEN**

Run Step 2 and then the complete report test file. Expected: all pass.

- [ ] **Step 7: Add failing static contract tests for theme, navigation, and accessibility**

Assert semantic controls/labels, live region, skip target, details summaries,
theme tokens for every surface, dark media query, manual theme selectors, print
rules, reduced-motion rule, pushState/popstate/hash handling, and absence of
duplicate IDs using an HTML parser.

- [ ] **Step 8: Complete the contract and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py -q
```

Expected: all report tests pass.

- [ ] **Step 9: Self-review and commit**

Mentally mutate escaping, topic selection, status filters, theme state, and
forbidden-network guards; ensure a test fails for each. Commit:

```bash
git add code/trec_rag/retrieval_nugget_coverage_report.py \
  code/tests/test_retrieval_nugget_coverage_report.py
git commit -m "Render retrieval coverage HTML report"
```

---

### Task 4: CLI publication, documentation, skill route, and integration checks

**Files:**
- Modify: `code/trec_rag/retrieval_nugget_coverage_report.py`
- Modify: `code/tests/test_retrieval_nugget_coverage_report.py`
- Modify: `code/tests/test_retrieval_nugget_coverage_skill.py`
- Modify: `code/trec_rag/README.md`
- Modify: `.agents/skills/trec-rag-competition-debug-report/SKILL.md`

**Interfaces:**
- Consumes: `load_coverage_report_data` and `render_coverage_report_html`.
- Produces:

```python
def publish_coverage_report(path: Path, body: bytes) -> Path: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

- CLI flags: required `--handoff-manifest`, `--coverage-root`, and `--output`;
  optional repeated `--topic`.

- [ ] **Step 1: Write failing CLI and publication tests**

Invoke `main([...])` against controlled complete fixtures and assert:

- a valid command returns zero and writes exact renderer bytes;
- repeated selectors preserve order;
- unknown, duplicate, empty, incomplete, and contradictory topic state returns a
  nonzero machine-safe error without a traceback;
- non-`.html`, symlinked output, output below a symlinked parent, and unwritable
  targets fail without changing an existing file;
- publication is atomic and leaves no temporary file after success or injected
  failure; and
- replacing an existing regular report with new deterministic bytes succeeds.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_report.py \
  -k 'cli or publish' -q
```

Expected: failures because CLI/publication functions are absent.

- [ ] **Step 3: Implement safe atomic publication and CLI**

Use a safe `argparse.ArgumentParser` subclass, structured stderr errors, and
`tempfile.NamedTemporaryFile` in the target directory followed by flush, fsync,
`os.replace`, and parent-directory fsync. Validate the full existing parent chain
for symlinks, reject a symlink/non-regular destination, require `.html`, and
always clean the temporary path.

The CLI loads, renders, publishes, and prints a compact JSON receipt containing
status, selected topic count, output path, output SHA-256, and `hosted_calls: 0`.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Establish the failing skill-route behavior test**

Extend `test_retrieval_nugget_coverage_skill.py` so the documented report route
is parsed into the exact command arguments and executed against controlled
fixtures. Assert the receipt has `hosted_calls == 0`, the HTML exists, and no
backend/network fake is invoked. Run it before editing the skill:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  -k 'html_report_route' -q
```

Expected: failure because the skill has no executable HTML report route.

- [ ] **Step 6: Add the minimal existing-skill route and README section**

Enhance the existing skill; do not create a new skill. Document:

- trigger words: view, render, browse, summarize, or inspect retrieval nugget
  coverage results;
- the exact zero-hosted-call CLI command;
- subnarratives/BM25 queries are retrieval-plan context and canonical nuggets
  are the judgment evidence representation;
- the report allowlist and forbidden private inputs; and
- privacy review plus the existing tailnet-only portal requirement before
  serving a presentation copy.

Add README inputs, outputs, validation, navigation, theme, and test commands.

- [ ] **Step 7: Verify skill GREEN and targeted integration**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage_report.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_generation_handoff.py -q

.venv/bin/python -m compileall -q \
  code/trec_rag/retrieval_nugget_coverage.py \
  code/trec_rag/retrieval_nugget_coverage_report.py

.venv/bin/python -m trec_rag.retrieval_nugget_coverage_report --help
```

Expected: all tests pass, compileall exits zero, and help lists the four flags.

- [ ] **Step 8: Self-review and commit**

Confirm skill guidance names no year-specific path/model/data and the report code
contains no hosted backend import/use. Commit:

```bash
git add code/trec_rag/retrieval_nugget_coverage_report.py \
  code/tests/test_retrieval_nugget_coverage_report.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/trec_rag/README.md \
  .agents/skills/trec-rag-competition-debug-report/SKILL.md
git commit -m "Document retrieval coverage HTML reports"
```

---

## Final Integration and Browser Verification

After all task reviews are clean, the controller must:

1. run the full project test suite fresh;
2. render the real 22-topic 2025 evaluation under its private output directory;
3. compare its six aggregate values to the validated report JSON values;
4. run headless Chromium at desktop and mobile widths in forced light and dark
   modes, checking console errors, network requests, horizontal overflow,
   overview/topic/back-forward behavior, filters, sorting, disclosures, theme
   overrides, focus visibility, and print CSS;
5. inspect screenshots for overview and detailed topic views in both themes;
6. scan the HTML for forbidden passage/document/provider/gold data and secrets;
7. copy only the privacy-reviewed derivative to the existing private tailnet
   portal, verify the live HTTPS response and tailnet-only mapping, and provide
   both the live URL and canonical private source path; and
8. run Claude Code review over the completed branch diff, fix confirmed findings
   with Luna, and ask Claude for one scoped re-review.

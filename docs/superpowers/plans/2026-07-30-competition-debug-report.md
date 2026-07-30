# Competition Debug Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only CLI and generic agent skill that explain every stored stage of a completed competition retrieval and optional RAG run in one private, self-contained HTML file.

**Architecture:** `trec_rag.competition_debug_report` loads standard retrieval and optional RAG configs, validates only bounded post-run artifacts, builds immutable topic-stage view records, renders deterministic standalone HTML, writes it atomically, and emits a compact JSON receipt. A concise repo-local skill under `.agents/skills/` discovers compatible completed runs and invokes this CLI without starting retrieval or generation. The skill ships in this parent-repository PR and does not depend on an unpublished organizer-submodule revision.

**Tech Stack:** Python 3.12, standard-library `argparse`, `dataclasses`, `hashlib`, `html`, `json`, `pathlib`, and `tempfile`; existing PyYAML-backed competition config loaders and validators; pytest; self-contained HTML5/CSS; Playwright when available; Agent Skills `SKILL.md` plus `agents/openai.yaml`.

## Global Constraints

- The report is post-run and read-only; it must never call retrieval, reranking, DeepSeek, canonicalization, OpenRouter, or another hosted/model path.
- Use `uv run --no-sync .venv/bin/python` for Python commands. Do not use `uv run` without the pinned interpreter.
- Accept `--retrieval-config` and optional `--rag-config`; do not accept a raw RAG-output path.
- Do not expose `--deep-verify` or a passage-count option.
- Never open `canonical/candidates.jsonl` or unsealed retrieval-provider caches.
- Render every stored subnarrative passage ranking; show the first five entries immediately and the remainder through progressive disclosure.
- Define a new document as a union-pool row whose memberships exclude `original`; do not infer newness from `selected_from_lane`.
- Show metadata for every new document and excerpts only for rows also present in `selected_documents.jsonl`.
- Keep the HTML self-contained, deterministic, accessible without JavaScript, responsive, and safe for hostile source strings.
- Write the report atomically under the ignored run output by default. Do not publish or copy it to the rendered portal because it contains raw corpus text and docids.
- Keep organizer output and checkpoint artifacts byte-for-byte unchanged.
- Implement code and the skill test-first. Commit focused changes frequently.

---

## File Structure

- Create `code/trec_rag/competition_debug_report.py`: typed report records, bounded artifact loading, cross-artifact validation, HTML rendering, atomic write, receipt, and CLI.
- Create `code/tests/test_competition_debug_report.py`: compact sealed-run fixtures and report/data/CLI contract tests.
- Modify `code/trec_rag/README.md`: post-run debugging command, privacy warning, output contract, and skill-friendly receipt.
- Create `.agents/skills/trec-rag-competition-debug-report/SKILL.md`: generic invocation workflow and safety/privacy contract.
- Create `.agents/skills/trec-rag-competition-debug-report/agents/openai.yaml`: generated Codex UI metadata.
- Create `code/tests/test_competition_debug_report_skill.py`: parent-local static skill contracts and discovery-path assertion.
- Keep `trec-rag-skills` at the official organizer revision; it remains source provenance only.

### Public interfaces

```python
@dataclass(frozen=True)
class DebugReportReceipt:
    schema_version: str
    output_path: Path
    topic_ids: tuple[str, ...]
    rag_included: bool
    source_sha256s: Mapping[str, str]

def load_debug_report_data(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
) -> DebugReportData: ...

def render_debug_report(data: DebugReportData) -> str: ...

def build_debug_report(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> DebugReportReceipt: ...

def main(argv: Sequence[str] | None = None) -> int: ...
```

The CLI success receipt is the compact JSON serialization of
`DebugReportReceipt`, with `output_path` rendered as an absolute string,
`topic_ids` as a JSON array, and `source_sha256s` sorted by portable relative
artifact label.

---

### Task 1: Bounded artifact model and topic identity

**Files:**
- Create: `code/trec_rag/competition_debug_report.py`
- Create: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: `load_facet_pilot_config(Path) -> FacetPilotConfig`, `select_configured_topics(FacetPilotConfig, topic_ids=...) -> tuple[Topic, ...]`, and bounded JSON/JSONL files under `FacetPilotConfig.output_dir`.
- Produces: immutable `DebugReportData`, `TopicReport`, `SubnarrativeReport`, `NewDocumentReport`, and `SelectedDocumentReport` records plus `load_debug_report_data(...)`.

- [ ] **Step 1: Build one minimal sealed-run fixture and write failing identity tests**

Create `_write_debug_run(tmp_path: Path) -> tuple[Path, Path]` in the test file. It writes a standard retrieval YAML, two official topics, root export artifacts, and one compact topic tree. Use literal records with hostile text so expectations do not reuse production logic:

```python
selection = {
    "schema_version": "facet_pilot_selection_v2",
    "topic_id": "rag2026-0",
    "selected_order": ["doc-original", "doc-facet"],
    "union_pool": [
        {"docid": "doc-original", "first_seen_lane": "original", "memberships": ["original"]},
        {"docid": "doc-both", "first_seen_lane": "original", "memberships": ["original", "facet-1"]},
        {"docid": "doc-facet", "first_seen_lane": "facet-1", "memberships": ["facet-1"]},
        {"docid": "doc-discarded", "first_seen_lane": "facet-1", "memberships": ["facet-1"]},
    ],
    "memberships": [
        {"docid": "doc-original", "lanes": [{"lane_name": "original", "aggregate_rank": 1, "aggregate_score": 9.0, "bm25_rank": 1, "bm25_score": 8.0}]},
        {"docid": "doc-facet", "lanes": [{"lane_name": "facet-1", "aggregate_rank": 1, "aggregate_score": 7.0, "bm25_rank": 2, "bm25_score": 6.0}]},
    ],
    "trace": [
        {"slot": 1, "docid": "doc-original", "lane_name": "original", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
        {"slot": 2, "docid": "doc-facet", "lane_name": "facet-1", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
    ],
}
```

Tests must assert official topic order, exact narrative/subnarrative/query values,
strict rejection of a decomposition topic mismatch, and rejection of a selected
document absent from `selected_order`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'identity or topic_order' -q
```

Expected: collection/import failure because `trec_rag.competition_debug_report`
does not exist.

- [ ] **Step 3: Implement strict bounded JSON helpers and immutable records**

In `competition_debug_report.py`, implement duplicate-key-safe JSON object and
JSONL readers, SHA-256 streaming for bounded source receipts, safe path checks,
and frozen report dataclasses. `load_debug_report_data` must:

```python
config = load_facet_pilot_config(retrieval_config_path)
topics = select_configured_topics(
    config,
    topic_ids=() if topic_ids is None else tuple(topic_ids),
)
```

Then require the root export manifest and per-topic `decomposition.json`,
`scoring/selection.json`, and `scoring/selected_documents.jsonl`. Validate topic
identity, uniqueness, stored order, ranks, text hashes, subnarrative identity,
and joins. Do not import or call `validate_retrieval_topic_checkpoints`.

- [ ] **Step 4: Implement the exact new-document classification**

Construct `NewDocumentReport` for every union row where:

```python
is_new = "original" not in tuple(row["memberships"])
excerpt = selected_text_by_docid.get(row["docid"])
```

Preserve union order within each `first_seen_lane`. Include every row's docid,
first-seen lane, memberships, and sealed text hash when available from retrieval
audit candidates. Only selected rows may carry an excerpt.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'identity or topic_order or new_document' -q
```

Expected: all selected tests pass, including `doc-facet` and `doc-discarded` as
new, `doc-both` as not new, and no excerpt for `doc-discarded`.

- [ ] **Step 6: Commit the bounded data foundation**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Add competition debug report data model"
```

---

### Task 2: Stored passages, selected evidence, and final retrieval projection

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Modify: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: Task 1 topic records and bounded `selected_subnarrative_scores.jsonl`, `subnarrative-selections.jsonl`, `canonical-nuggets.jsonl`, root TREC run, `retrieval_provenance.jsonl`, full-text ZIP, and export manifest.
- Produces: `PassageRankingReport`, `EvidenceClusterReport`, `CanonicalNuggetReport`, and `RetrievalOutputReport` fields on each `TopicReport`.

- [ ] **Step 1: Write failing passage and nugget join tests**

Extend the fixture with two stored subnarrative-score rows, a selected snapshot,
one evidence cluster, one final canonical nugget, and two final retrieval rows.
Use a selected source string with a hand-derived exact span:

```python
source = "Intro. Exact evidence sentence. Tail."
start = 7
end = 31
assert source[start:end] == "Exact evidence sentence."
```

Assert that the loader preserves every stored `aggregate_rank`, labels
`raw_logit` as a logit field, reconstructs the exact span, joins selected
cluster IDs at the configured budget, joins canonical evidence by docid, and
explains selected depth 2 versus final supported depth 1.

- [ ] **Step 2: Write a candidate-ledger tripwire test**

Create `canonical/candidates.jsonl` as a directory or monkeypatch
`Path.open`/`Path.read_bytes` to raise only for that resolved path. Also
monkeypatch `socket.socket.connect` to raise if any network connection is
attempted. Call `load_debug_report_data` and assert success. This test catches
future candidate-ledger or network access while asserting the observable report
still builds from bounded artifacts.

- [ ] **Step 3: Run the new tests and verify RED**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'passage or nugget or candidate_ledger or retrieval_projection' -q
```

Expected: failures because the Task 2 report fields are absent.

- [ ] **Step 4: Implement complete stored passage loading**

Group score rows by decomposition subnarrative order, require continuous
`aggregate_rank`, resolve each docid against selected text, and validate every
winning passage as:

```python
if not (0 <= start < end <= len(source)):
    raise ValueError("winning passage offsets are outside selected document")
text = source[start:end]
```

Retain all stored ranks and passages. Do not cap or recompute rankings.

- [ ] **Step 5: Implement selection, canonical nugget, and retrieval joins**

Parse the selected budget snapshot from each subnarrative selection, require
referenced clusters to exist, and build selected cluster records. Parse final
canonical results in stored subnarrative order and validate their topic,
subnarrative, state, budgets, evidence docids, and caps against the retrieval
config. Parse the root TREC run with the production RAG run loader, then join
`retrieval_provenance.jsonl` and full-text ZIP by `(topic_id, docid)` and require
exact coverage.

- [ ] **Step 6: Run the full report-data tests and verify GREEN**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -q
```

Expected: all current report tests pass, including candidate-ledger tripwire and
invalid-offset rejection.

- [ ] **Step 7: Commit stage joins**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Explain competition retrieval stages"
```

---

### Task 3: Self-contained accessible HTML renderer

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Modify: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: complete `DebugReportData` from Tasks 1–2.
- Produces: `render_debug_report(data: DebugReportData) -> str`.

- [ ] **Step 1: Write failing renderer structure and escaping tests**

Assert one `<!doctype html>`, `lang="en"`, viewport and color-scheme metadata,
semantic `main`/`nav`/topic `section` elements, all eight stage headings, table
captions/scoped headers, visible focus CSS, reduced-motion CSS, and no external
`http://`, `https://`, `<script src=`, stylesheet, image, or font dependency.

Insert literal hostile source strings:

```python
narrative = '<script>alert("narrative")</script>'
claim = '</script><img src=x onerror=alert(1)>'
```

Assert the raw strings do not appear and escaped text does. Assert the first
five passage ranking entries are outside the collapsed remainder `<details>` and
every later rank remains present inside it.

- [ ] **Step 2: Run renderer tests and verify RED**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -k 'html or escaping or passage_disclosure' -q
```

Expected: failures because `render_debug_report` is absent.

- [ ] **Step 3: Implement deterministic semantic rendering**

Use `html.escape(..., quote=True)` at the final rendering boundary for every
source-derived string. Render inline CSS only. Use topic/stage anchor IDs built
from validated safe topic IDs, `<details>` for excerpts and long collections,
and horizontally scrollable table wrappers. Render statuses with text plus
color. Do not embed raw JSON or require JavaScript.

The visible topic flow is exactly:

```text
Narrative → Subnarratives → New documents → Selected documents
→ Top passages → Final selected nuggets → Final retrieval → Final RAG
```

- [ ] **Step 4: Run renderer tests and verify GREEN**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -q
```

Expected: all report tests pass and hostile strings are text, never markup.

- [ ] **Step 5: Commit the renderer**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Render competition pipeline debug report"
```

---

### Task 4: Standard RAG config, atomic output, CLI receipt, and operator docs

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Modify: `code/tests/test_competition_debug_report.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: `load_rag_generation_config`, `load_queries`, `select_queries`, `load_trec_run`, and `validate_submission_record` from `competition_rag`.
- Produces: `build_debug_report(...) -> DebugReportReceipt` and `main(argv) -> int` with `--retrieval-config`, optional `--rag-config`, repeatable `--topic`, and optional `--output`.

- [ ] **Step 1: Write failing optional and validated RAG tests**

Add fixture RAG config/output records. Assert omission renders “RAG output not
supplied.” With the config supplied, assert exact topic order, answer items,
reference docids, zero-based citation-to-docid mapping, word count, and output
SHA-256. Mutate one RAG run path or topic narrative and assert a compatibility
error before output is written.

- [ ] **Step 2: Write failing atomic-output and CLI tests**

Call `main([...])` with a temporary `--output`. Capture stdout and assert the
exact receipt keys:

```python
assert set(receipt) == {
    "schema_version", "output_path", "topic_ids",
    "rag_included", "source_sha256s",
}
```

Assert the absolute path, selected topic list, stable sorted hashes, compact
single-line JSON, and existing output preservation when rendering is forced to
raise before replacement.

- [ ] **Step 3: Run CLI/RAG tests and verify RED**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -k 'rag or cli or atomic or receipt' -q
```

Expected: failures because RAG integration, build, and CLI seams are absent.

- [ ] **Step 4: Implement RAG config compatibility and validation**

Load the standard RAG config, require its canonical run and documents paths to
resolve to the retrieval export paths, select the same topic IDs, parse each
compact output row, and call `validate_submission_record` with production
metadata and allowed docids. Build immutable RAG topic records with resolved
citations and word counts.

- [ ] **Step 5: Implement atomic build and CLI**

Render fully before creating a same-directory temporary file. Write, flush,
`fsync`, `os.replace`, and directory-`fsync`; delete only the explicit temporary
file after failure. Reject output paths outside the repository or retrieval run
output unless the caller explicitly supplies an existing parent under the repo.
Print one compact sorted JSON receipt only after successful replacement.

- [ ] **Step 6: Document the post-run command**

Add retrieval-only and retrieval-plus-RAG examples using:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/...yaml \
  --rag-config configs/...yaml
```

State that the command is read-only, does not make API/model calls, avoids
candidate ledgers, writes a private raw-text report, and emits a JSON receipt.

- [ ] **Step 7: Run targeted and neighboring tests**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  code/tests/test_competition_rag.py \
  code/tests/test_retrieval_export.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit CLI and docs**

```bash
git add code/trec_rag/competition_debug_report.py \
  code/tests/test_competition_debug_report.py code/trec_rag/README.md
git commit -m "Add competition debug report CLI"
```

---

### Task 5: Generic agent skill, metadata, and forward evaluation

**Files:**
- Create: `.agents/skills/trec-rag-competition-debug-report/SKILL.md`
- Create: `.agents/skills/trec-rag-competition-debug-report/agents/openai.yaml`
- Create: `code/tests/test_competition_debug_report_skill.py`

**Interfaces:**
- Consumes: Task 4 CLI contract and standard retrieval/RAG config paths.
- Produces: discoverable repo-local `trec-rag-competition-debug-report` skill with no bundled executable script; the repository CLI is the single implementation and the parent PR is self-contained.

- [ ] **Step 1: Run a no-skill baseline scenario (RED)**

Dispatch a fresh, minimum-context agent without the proposed skill:

```text
The run is already complete. Make me a private HTML explanation of every stage
for this retrieval config and its matching RAG config. Do not rerun models or
publish anything. Return the report path and validation summary.
```

Provide only the repository path and the two config paths. Record whether the
agent discovers the CLI, uses both standard configs, avoids hosted/model calls,
keeps output private, and returns the JSON receipt. The expected baseline gap is
at least one missed contract element; if the control already satisfies every
element, keep the skill as a concise discoverability reference and do not add
hypothetical prohibitions.

- [ ] **Step 2: Initialize the skill using the required scaffold tool**

Run from the parent repository root:

```bash
.venv/bin/python \
  /home/npatta01/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  trec-rag-competition-debug-report \
  --path .agents/skills \
  --interface 'display_name=Competition Debug Report' \
  --interface 'short_description=Explain a completed TREC RAG run' \
  --interface 'default_prompt=Use $trec-rag-competition-debug-report to create a private HTML explanation of this completed competition run.'
```

Do not request `scripts`, `references`, `assets`, or examples; the CLI already
provides deterministic execution and the workflow fits in one concise skill.

- [ ] **Step 3: Replace scaffold placeholders with the minimal skill**

Use frontmatter with exactly `name` and a trigger-only third-person description:

```yaml
---
name: trec-rag-competition-debug-report
description: Use when a user asks to inspect, explain, visualize, audit, or debug the stages of a completed TREC RAG competition retrieval or RAG run.
---
```

The body must instruct agents to locate one standard retrieval config and
optional matching RAG config, confirm the run is complete, invoke the exact CLI
through `uv run --no-sync .venv/bin/python`, pass requested topics only, read the
JSON receipt, return the absolute report path plus concise summary, and warn that
the local HTML contains private corpus text. State positively that this is a
post-run report workflow; requests to execute an incomplete run belong to the
competition retrieval/RAG path instead.

- [ ] **Step 4: Validate structure and metadata**

```bash
.venv/bin/python \
  /home/npatta01/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .agents/skills/trec-rag-competition-debug-report
wc -w .agents/skills/trec-rag-competition-debug-report/SKILL.md
.venv/bin/python -m pytest code/tests/test_competition_debug_report_skill.py -q
```

Expected: validation passes, frontmatter contains only required fields, UI
metadata strings are quoted, default prompt names the skill, and SKILL.md stays
under 500 words.

- [ ] **Step 5: Forward-test the skill (GREEN)**

Dispatch a fresh agent with only the new skill path, repository path, and a
realistic request equivalent to Step 1. Verify it reads the complete SKILL.md,
uses the standard configs, invokes the CLI without hosted/model calls, does not
publish the raw report, and reports the CLI receipt. If it misses an observable
contract element, revise only the instruction responsible and repeat the same
scenario.

- [ ] **Step 6: Include the skill in the parent feature commit**

```bash
git add .agents/skills/trec-rag-competition-debug-report \
  code/tests/test_competition_debug_report_skill.py
```

Deliver these parent-local files in the same parent-repository PR as the CLI.
Do not create an organizer branch or commit, and do not change the organizer
gitlink for this feature.

---

### Task 6: Real two-topic HTML, visual QA, and completion verification

**Files:**
- Generate ignored artifact: `outputs/facet-deepseek-b40-v1-two-topic-smoke/competition_debug_report.html`
- Modify only if verification finds a defect: report code/tests/skill files from Tasks 1–5.

**Interfaces:**
- Consumes: final CLI, skill, and existing sealed two-topic retrieval/RAG artifacts.
- Produces: verified private HTML report and completion evidence.

- [ ] **Step 1: Run the real post-run CLI**

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/local/rag26_competition_retrieval_two_topic_smoke.yaml \
  --rag-config configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml
```

Expected: one compact JSON receipt, no API/model calls, and the default HTML
under the retrieval output directory without scanning either candidate ledger.

- [ ] **Step 2: Validate real report coverage programmatically**

Use the production module and a small read-only assertion command to require:

```python
assert receipt.topic_ids == ("rag2026-0", "rag2026-1")
assert [topic.retrieval_output.final_supported_depth for topic in data.topics] == [71, 60]
assert all(topic.subnarratives for topic in data.topics)
assert all(topic.new_documents for topic in data.topics)
assert all(topic.passage_rankings for topic in data.topics)
assert all(topic.canonical_nuggets for topic in data.topics)
assert all(topic.rag_output is not None for topic in data.topics)
```

Also record output byte count and SHA-256.

- [ ] **Step 3: Inspect desktop and mobile rendering**

If Playwright is installed, open the local HTML at 1440×1000 and 390×844,
capture screenshots outside git, and check headings, `<details>`, horizontal
table scrolling, focus visibility, overflow, and readable excerpts. If
Playwright is unavailable, record that limitation and verify HTML semantics with
the pytest assertions; do not install a new browser dependency merely for this
check.

- [ ] **Step 4: Run final repository verification**

```bash
uv run --no-sync .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q code/trec_rag
git diff --check
git status --short --ignore-submodules=dirty
git -C trec-rag-skills status --short
git submodule status
```

Expected: full suite passes; compilation and diff checks exit 0; both parent and
submodule are clean; submodule status pins `trec-rag-skills` to the official
pre-feature revision `f281e88f61252662033c681df8b1ed2d0ceda97e`.

- [ ] **Step 5: Review the complete branch against the spec**

Review `origin/master...HEAD` along Standards and Spec axes. Treat any report
path that opens candidate ledgers, any rerun/network behavior, missing passage
rows, unsafe source rendering, nonstandard config input, or missing skill
privacy guard as blocking. Fix blocking findings test-first and repeat Steps 1–4.

- [ ] **Step 6: Hand off without publishing**

Report the CLI command, absolute private HTML path, receipt hashes, full-suite
result, repo-local skill path, official organizer submodule revision, and any
unavailable visual check. Do not publish the private report.

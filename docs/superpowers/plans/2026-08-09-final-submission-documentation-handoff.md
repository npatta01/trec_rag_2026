# Final Submission Documentation Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the completed TREC RAG 2026 repository self-explanatory through one root artifact hub, one accurate final architecture report, accepted-submission status, and consistent links across reader and agent documentation.

**Architecture:** Treat the checked-in submission bundles and their metadata as immutable final evidence. Update canonical authored documentation first, render only the Quarto-owned architecture HTML from its QMD source, and use static Node smoke tests plus the repo-local submission validator to keep links, privacy boundaries, artifact hashes, and final-system terminology aligned.

**Tech Stack:** Standalone HTML/CSS, Quarto 1.9.38, accessible SVG, CommonJS Node smoke tests, Markdown, the repo-local Python submission validator, Playwright/Chromium-compatible screenshot tooling.

## Global Constraints

- The five checked-in organizer files were uploaded and accepted by Evalbase; do not change their bytes.
- Record missing Evalbase IDs and timestamps as `Not recorded`; never invent them.
- The final Retrieval TSVs and RAG JSONLs are sibling products with shared source provenance; the organizer Retrieval TSV is not Generation input.
- Generation reads only the sealed selected-evidence handoff and never reads the Retrieval TSV, full-text ZIP, qrels, gold nuggets, or RAGDoll scores.
- Keep `reports/2026-competition-architecture.qmd` canonical and regenerate `reports/2026-competition-architecture.html`; never patch the generated HTML directly.
- Do not expose ignored `outputs/`, caches, private work state, raw provider material, prompts, passages, answers, credentials, or absent local-only paths.
- Do not run retrieval, reranking, generation, evaluation, or hosted models.
- Do not deploy, publish, alter sharing permissions, or configure GitHub Pages.
- Preserve historical experiment records, plans, and specifications.

## File Map

- `index.html`: reader-facing root artifact hub.
- `index.test.js`: root hub, cross-document link, acceptance-status, and privacy smoke test.
- `README.md`: concise completed-project orientation and key-artifact map.
- `AGENTS.md`: completed-project state, final architecture invariants, canonical artifacts, validation command, and preservation rules.
- `submissions/trec-rag-2026/SUBMISSION_LEDGER.md`: accepted status and exact organizer-file control sheet.
- `submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md`: submitted Retrieval bundle guide.
- `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md`: submitted RAG bundle guide.
- `reports/2026-competition-architecture.qmd`: canonical final architecture narrative.
- `reports/2026-competition-architecture.html`: Quarto-generated final architecture report.
- `reports/2026-competition-architecture/*.svg`: nine final architecture views.
- `reports/2026-competition-architecture/README.md`: diagram manifest and source mapping.
- `reports/2026-competition-architecture.test.js`: architecture source/render/accessibility contract.
- `reports/index.html`: reports collection navigation with root/submission links.
- `reports/index.test.js`: report-index copy and link contract.

---

### Task 1: Root Artifact Hub

**Files:**
- Create: `index.test.js`
- Create: `index.html`

**Interfaces:**
- Consumes: existing repository-relative paths to the architecture report, submission ledger, bundle READMEs, skills, report index, code README, configs, and official submodules.
- Produces: one dependency-free root page whose internal repository links resolve and whose content contains no private path or secret signal.

- [x] **Step 1: Write the failing root-page smoke test**

Create `index.test.js` with a small `assert` helper, require `index.html`, and check the exact content contract:

```js
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "index.html");
function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(htmlPath), "root index.html should exist");
const html = fs.readFileSync(htmlPath, "utf8");

for (const signal of [
  "TREC RAG 2026 Submission",
  "Completed project",
  "Accepted submissions",
  "Architecture",
  "Validation & skills",
  "Reports",
  "Code",
  "Official inputs",
  "submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "reports/2026-competition-architecture.html",
  ".agents/skills/validate-trec-rag-2026-submissions/SKILL.md",
  "trec-rag-skills/skills/trec-rag-2026-track-guidelines/SKILL.md",
  "reports/index.html",
  "code/trec_rag/README.md",
]) assert(html.includes(signal), `Missing root-page signal: ${signal}`);

for (const signal of ["<main", "<header", "<section", "<footer", "aria-label=", "focus-visible", "prefers-reduced-motion", "@media (max-width: 620px)"])
  assert(html.includes(signal), `Missing accessibility/responsive signal: ${signal}`);

for (const forbidden of ["outputs/", "cache/", "PYSERINI_API_TOKEN", "OPENROUTER_API_KEY", "Authorization", "Bearer ", "TO" + "DO", "place" + "holder"])
  assert(!html.includes(forbidden), `Root page should not expose ${forbidden}`);

console.log("root artifact hub smoke test passed");
```

Add a repository-relative link list in the test and resolve each path from the repository root with `fs.existsSync`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
node index.test.js
```

Expected: FAIL with `root index.html should exist`.

- [x] **Step 3: Implement the standalone artifact hub**

Create `index.html` with this semantic structure and exact card labels:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Accepted TREC RAG 2026 submissions, architecture, validation, reports, code, and official inputs.">
  <title>TREC RAG 2026 Submission</title>
  <style>
    :root { color-scheme: dark light; --bg: #07111f; --text: #edf5ff; --muted: #a9bdd3; --border: #29435f; --blue: #6ab8ff; }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); margin: 0; min-height: 100vh; font-family: system-ui, sans-serif; }
    main { margin: 0 auto; max-width: 1180px; padding: clamp(48px, 8vw, 100px) 28px 56px; }
    .link-grid { display: grid; gap: 22px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .link-card { border-top: 5px solid var(--blue); min-height: 230px; padding: 26px 22px 22px; }
    a:focus-visible { border-radius: 3px; outline: 3px solid #f4bd62; outline-offset: 5px; }
    @media (max-width: 980px) { .link-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 620px) { .link-grid { grid-template-columns: 1fr; } .link-card { min-height: 0; } }
    @media (prefers-color-scheme: light) { :root { --bg: #f7f9fc; --text: #10243a; --muted: #50667c; --border: #cad6e2; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .000001s !important; } }
    @media print { :root { --bg: #fff; --text: #111827; --muted: #374151; } .link-card { break-inside: avoid; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">TREC RAG 2026 · completed project</div>
      <h1>TREC RAG 2026 submission</h1>
      <p class="intro">The final architecture, five Evalbase-accepted files, validation contracts, reports, and implementation.</p>
    </header>
    <section class="link-grid" aria-label="Project artifacts">
      <article class="link-card architecture"><h2>Architecture</h2></article>
      <article class="link-card submissions"><h2>Accepted submissions</h2></article>
      <article class="link-card validation"><h2>Validation &amp; skills</h2></article>
      <article class="link-card reports"><h2>Reports</h2></article>
      <article class="link-card code"><h2>Code</h2></article>
      <article class="link-card inputs"><h2>Official inputs</h2></article>
    </section>
    <footer><span>Five accepted runs · 119 narratives</span></footer>
  </main>
</body>
</html>
```

Adapt the proven `music-crs-2026` visual grammar: six accent-colored cards, system fonts, a three/two/one-column responsive grid, visible keyboard focus, dark/light variables, reduced-motion handling, and print-safe layout. Populate every card with the exact relative links listed in the approved design.

- [x] **Step 4: Run the root smoke test and verify GREEN**

Run `node index.test.js` and require `root artifact hub smoke test passed`.

- [x] **Step 5: Commit the root hub**

```bash
git add index.html index.test.js
git commit -m "docs: add final submission artifact hub"
```

---

### Task 2: Accepted Status and Canonical Artifact Maps

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `submissions/trec-rag-2026/SUBMISSION_LEDGER.md`
- Modify: `submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md`
- Modify: `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md`
- Modify: `index.test.js`

**Interfaces:**
- Consumes: the five immutable organizer files, their committed metadata/hashes, the user-confirmed Evalbase acceptance, and the repo-local validator command.
- Produces: consistent accepted status, explicit unknown confirmation metadata, one key-artifact map for people, and one for future agents.

- [x] **Step 1: Extend the smoke test with failing final-status assertions**

In `index.test.js`, read the five Markdown files and assert:

```js
assert(read("README.md").includes("## Final Project Artifacts"), "README should lead with final artifacts");
assert(read("AGENTS.md").includes("## Completed Project and Key Artifacts"), "AGENTS should orient to final artifacts");
assert((read("submissions/trec-rag-2026/SUBMISSION_LEDGER.md").match(/Accepted by Evalbase/g) || []).length >= 10, "ledger should record five table statuses and five confirmations");
assert(read("submissions/trec-rag-2026/SUBMISSION_LEDGER.md").includes("Not recorded"), "unknown portal IDs/timestamps must be explicit");
assert(read("submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md").includes("uploaded and accepted"), "Retrieval README should record acceptance");
assert(read("submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md").includes("uploaded and accepted"), "RAG README should record acceptance");
```

Also assert that `AGENTS.md` links the root hub, final architecture source/render, submission ledger, both bundle READMEs, validation skill, official contract, and code README.

- [x] **Step 2: Run the test and verify RED**

Run `node index.test.js` and require failure on the new status/artifact-map assertions.

- [x] **Step 3: Update the submission ledger without fabricating portal metadata**

Change each run-table status from `Ready to upload` to `Accepted by Evalbase`. Replace the confirmation table with these exact rows:

```markdown
| `r26-narr-facet-v1` | Not recorded | Not recorded | Accepted by Evalbase | User-confirmed 2026-08-09; exact checked-in file independently revalidated. |
| `r26-facet-breadth-v1` | Not recorded | Not recorded | Accepted by Evalbase | User-confirmed 2026-08-09; exact checked-in file independently revalidated. |
| `r26-narrative-v1` | Not recorded | Not recorded | Accepted by Evalbase | User-confirmed 2026-08-09; exact checked-in file independently revalidated. |
| `rag26-ms1-final` | Not recorded | Not recorded | Accepted by Evalbase | User-confirmed 2026-08-09; exact checked-in file independently revalidated. |
| `rag26-ss1` | Not recorded | Not recorded | Accepted by Evalbase | User-confirmed 2026-08-09; exact checked-in file independently revalidated. |
```

Keep all file links, hashes, row/topic counts, priorities, and Evalbase response links unchanged. Retitle `Submission procedure` to `Submitted-file verification` and describe how to revalidate the exact accepted bytes rather than instructing a future upload.

- [x] **Step 4: Update the two bundle READMEs**

Lead each README with `All files in this bundle were uploaded and accepted by Evalbase.` Rename `Upload files` to `Accepted files`. Preserve the priority tables, architecture explanation, privacy boundary, and validation counts. Replace imperative upload language with archival verification language and link the root hub, final architecture report, and submission ledger.

- [x] **Step 5: Add the human and agent artifact maps**

Add `## Final Project Artifacts` immediately after the README introduction with links to:

```text
index.html
reports/2026-competition-architecture.html
reports/2026-competition-architecture.qmd
submissions/trec-rag-2026/SUBMISSION_LEDGER.md
submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md
submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md
.agents/skills/validate-trec-rag-2026-submissions/SKILL.md
reports/index.html
code/trec_rag/README.md
```

Add `## Completed Project and Key Artifacts` near the top of `AGENTS.md`. State that the project is complete, the five organizer files are accepted and immutable, historical plans are records rather than a backlog, the QMD is canonical, and future agents should start at the root hub/ledger/final architecture. Replace the stale sequential architecture orientation with the frozen-source two-branch invariants from the approved spec. Retain the existing safety and environment rules that remain useful for reproducibility.

- [x] **Step 6: Run the smoke test and verify GREEN**

Run `node index.test.js` and require success.

- [x] **Step 7: Commit accepted status and artifact maps**

```bash
git add README.md AGENTS.md index.test.js \
  submissions/trec-rag-2026/SUBMISSION_LEDGER.md \
  submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md \
  submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md
git commit -m "docs: record accepted TREC RAG submissions"
```

---

### Task 3: Final Submitted Architecture Report

**Files:**
- Modify: `reports/2026-competition-architecture.test.js`
- Modify: `reports/2026-competition-architecture.qmd`
- Modify: `reports/2026-competition-architecture/README.md`
- Modify: `reports/2026-competition-architecture/report.css`
- Keep: `reports/2026-competition-architecture/theme-light.scss`
- Keep: `reports/2026-competition-architecture/theme-dark.scss`
- Modify: `reports/2026-competition-architecture/01-whole-system.svg`
- Rename/replace: `reports/2026-competition-architecture/02-retrieval-system.svg` → `02-frozen-source-retrieval.svg`
- Keep/update: `reports/2026-competition-architecture/03-bounded-deepseek-planning.svg`
- Rename/replace: `reports/2026-competition-architecture/04-per-query-candidate-accounting.svg` → `04-variable-depth-candidate-core.svg`
- Rename/replace: `reports/2026-competition-architecture/05-evidence-and-nuggetizer.svg` → `05-targeted-scoring-and-runs.svg`
- Keep/update: `reports/2026-competition-architecture/06-generation-handoff-contract.svg`
- Rename/replace: `reports/2026-competition-architecture/07-sol-generation.svg` → `07-single-pass-rag.svg`
- Rename/replace: `reports/2026-competition-architecture/08-validation-and-retries.svg` → `08-multistage-rag.svg`
- Rename/replace: `reports/2026-competition-architecture/09-organizer-output-split.svg` → `09-accepted-submissions.svg`
- Regenerate: `reports/2026-competition-architecture.html`

**Interfaces:**
- Consumes: final Retrieval architecture/metadata, final RAG README/metadata, frozen source-run contracts, and only aggregate privacy-reviewed evidence.
- Produces: one canonical QMD and one self-contained rendered HTML report with nine accessible final-state diagrams and direct links to exact checked-in artifacts.

- [x] **Step 1: Rewrite the architecture test for the final story and verify RED**

Replace the old figure list with:

```js
const figures = [
  "01-whole-system.svg",
  "02-frozen-source-retrieval.svg",
  "03-bounded-deepseek-planning.svg",
  "04-variable-depth-candidate-core.svg",
  "05-targeted-scoring-and-runs.svg",
  "06-generation-handoff-contract.svg",
  "07-single-pass-rag.svg",
  "08-multistage-rag.svg",
  "09-accepted-submissions.svg",
];
```

Require the final concepts:

```js
const requiredConcepts = [
  "Frozen source, two submission branches",
  "facet-deepseek-b40-v3",
  "DeepSeek V4 Flash",
  "climbmix-400b",
  "mixedbread-ai/mxbai-rerank-base-v2",
  "median absolute deviation",
  "2.5 × 1.4826 × MAD",
  "variable depth",
  "1–121 documents",
  "4,246 rows",
  "r26-narr-facet-v1",
  "r26-facet-breadth-v1",
  "r26-narrative-v1",
  "generation_handoff_manifest.json",
  "rag26-ms1-final",
  "rag26-ss1",
  "openai/gpt-5.6-luna",
  "openai/gpt-5.6-sol",
  "691 evidence groups",
  "1,024-word ceiling",
  "Accepted by Evalbase",
  "The Retrieval TSV is not Generation input",
];
```

Retain tests for embedded/local runtime assets, dark-mode SVG support, `role="img"`, title/description IDs, vector-effect, nine text equivalents, nine source disclosures, private-data exclusions, canonical QMD links in README/AGENTS, and generated HTML containing every figure.

Run `node reports/2026-competition-architecture.test.js`; expect failure because renamed figures and final concepts do not yet exist.

- [x] **Step 2: Author the final QMD narrative**

Keep the existing Quarto YAML contract and use these exact top-level sections:

```markdown
# Frozen source, two submission branches {#whole-system}
# Source retrieval: bounded widening, fixed evidence {#source-retrieval}
# Planning stays one-shot {#planning}
# Retrieval submission: robust variable depth {#candidate-core}
# Three orderings, one candidate set {#retrieval-runs}
# The handoff is the RAG trust boundary {#handoff}
# Single-pass RAG {#single-pass}
# Multi-stage RAG {#multi-stage}
# Five accepted organizer files {#accepted-artifacts}
# Sources, validation, and scope {#sources}
```

For each of the nine visual sections, include in order: a one-sentence takeaway, figure with full alt description, direct SVG link, explanatory prose, `**Text equivalent:**`, and `<details><summary>Implementation notes and sources</summary>` with repository-relative source links. Include the exact run IDs and final aggregate counts from committed metadata. Do not include any official test narrative ID/text, passage, prompt, provider response, or generated answer.

- [x] **Step 3: Replace the nine diagram views**

Use the existing SVG accessibility/theme template. For example, the candidate-core figure begins with this complete accessibility and palette contract before its labeled shapes:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img"
     aria-labelledby="candidate-core-title candidate-core-desc">
  <title id="candidate-core-title">Robust variable-depth candidate core</title>
  <desc id="candidate-core-desc">Authenticated original and subnarrative lanes apply a median absolute deviation threshold, union admitted documents, and retain a narrative-specific candidate depth from one to 121 documents.</desc>
  <style>
    :root { --paper:#ffffff; --ink:#17201c; --line:#52615a; --retrieval:#0f766e; --retrieval-soft:#dff2ee; --advisory:#8b5e0a; --advisory-soft:#faedc9; }
    @media (prefers-color-scheme: dark) { :root { --paper:#121b17; --ink:#eef5f0; --line:#afbbb4; --retrieval:#67d4c7; --retrieval-soft:#183d38; --advisory:#efbd67; --advisory-soft:#433616; } }
    .edge { vector-effect="non-scaling-stroke"; }
  </style>
  <rect width="1200" height="720" fill="var(--paper)"/>
  <text x="600" y="64" text-anchor="middle" fill="var(--ink)" font-size="34" font-weight="700">Robust admission creates one shared candidate core</text>
</svg>
```

Give the nine diagrams these questions/content boundaries:

1. complete frozen-source split into Retrieval and RAG bundles;
2. one-shot planning, BM25 lanes, Mixedbread scoring, evidence selection, frozen source;
3. original lane plus bounded 0–8 subnarratives and no iterative replan loop;
4. per-lane robust admission, inclusive candidate union, variable `k`, empty-union fallback;
5. cache-first targeted matrix and the combo/breadth/narrative orderings over identical sets;
6. authenticated handoff contents and barred inputs;
7. handoff → Sol → local validation → `rag26-ss1`;
8. Luna blueprint/audits → Sol draft → bounded revision/screen/fallback → `rag26-ms1-final`;
9. three Retrieval TSVs plus two RAG JSONLs, all accepted, with validator/hash gates.

Delete only the five superseded SVG filenames listed as rename/replace targets. Update the asset README table with the new names, reader questions, exact source files, and concise text equivalents. Preserve its conceptual-geometry and privacy explanation.

- [x] **Step 4: Add small final-artifact report styles**

Extend `report.css` with responsive `.artifact-grid`, `.artifact-card`, `.status-accepted`, and `.branch-note` classes using existing guide variables. Keep print, reduced-motion, focus, dark/light, and figure rules intact.

- [x] **Step 5: Render the canonical source**

Run:

```bash
quarto render reports/2026-competition-architecture.qmd
```

Expected: exit 0 and a regenerated `reports/2026-competition-architecture.html` identifying Quarto as generator with all runtime dependencies embedded/local.

- [x] **Step 6: Run the architecture smoke test and verify GREEN**

Run:

```bash
node reports/2026-competition-architecture.test.js
```

Require `2026 competition architecture smoke test passed`.

- [x] **Step 7: Commit the final architecture source, assets, test, and render**

```bash
git add reports/2026-competition-architecture.qmd \
  reports/2026-competition-architecture.html \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture/
git commit -m "docs: document final TREC RAG architecture"
```

---

### Task 4: Reports Collection Navigation

**Files:**
- Modify: `reports/index.test.js`
- Modify: `reports/index.html`

**Interfaces:**
- Consumes: the root artifact hub, final architecture report, accepted submission ledger, and existing historical reports.
- Produces: a reports collection page that leads with the final architecture and offers obvious paths back to the whole-project hub and accepted files.

- [x] **Step 1: Update report-index expectations and verify RED**

Change the test to require:

```js
for (const signal of [
  "Final 2026 submission architecture",
  "Frozen source → Retrieval submissions + RAG submissions",
  "Accepted submissions",
  "../index.html",
  "../submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "2026-competition-architecture.html",
]) assert(html.includes(signal), `Missing final navigation signal: ${signal}`);
```

Retain the historical report links and forbidden secret/draft-marker checks. Run `node reports/index.test.js` and expect failure on the new copy/links.

- [x] **Step 2: Update the report index**

Change the hero primary action to the final architecture, add secondary actions for the root hub and accepted ledger, update the architecture card copy to the frozen-source two-branch story, and retain the briefing/2025/experiment cards as supporting context. Use relative links consistently so the page works from the repository root and a static host.

- [x] **Step 3: Run both navigation smoke tests**

```bash
node index.test.js
node reports/index.test.js
```

Require both pass.

- [x] **Step 4: Commit report navigation**

```bash
git add reports/index.html reports/index.test.js
git commit -m "docs: connect final reports and submissions"
```

---

### Task 5: Final Validation and Responsive QA

**Files:**
- Modify: `docs/superpowers/plans/2026-08-09-final-submission-documentation-handoff.md` (mark completed steps and append exact verification evidence only)

**Interfaces:**
- Consumes: the completed documentation tree and immutable organizer files.
- Produces: reproducible final validation evidence without model calls, publication, or private-data access.

- [x] **Step 1: Revalidate all five accepted files**

Run the exact combined validator:

```bash
python3 .agents/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
  --topics trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv \
  --retrieval submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/combo/r_output_trec_rag_2026.tsv \
  --retrieval submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/breadth/r_output_trec_rag_2026.tsv \
  --retrieval submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/narrative/r_output_trec_rag_2026.tsv \
  --rag submissions/trec-rag-2026/rag/selected-evidence-sol-v1/multistage/rag_output_trec_rag_2026.jsonl \
  --rag submissions/trec-rag-2026/rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl
```

Expected: three Retrieval `PASS` results with 4,246 rows, 119 topics, depth 1–121; two RAG `PASS` results with 119/119 valid reports and no extras/duplicates.

- [x] **Step 2: Verify immutable hashes and filenames**

Run `sha256sum` on the five exact organizer files and compare with ledger/metadata values:

```text
29bc0c29dd51a752d49c734db456926ef94ad5202102cabd92d7fbf3e9dd15e8  combo
f42a794418cf692721adcd53df232ca0a3536d13d9cd137d90eb9821562e199d  breadth
80985a42e43333085975c27d88a1da82cc11cba6dd576c3fd506839c016f2feb  narrative
72200f7a0e3be19f9c7e8f23d3845f894f5e95e26caff2986ae16f848b4dee00  multistage
a33c60325c198cf90178d5b4c6c1b04209f7d39de278736b6817a8c5666c02a0  singlepass
```

- [x] **Step 3: Run all documentation smoke tests**

```bash
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
node reports/trec-rag-briefing-report.test.js
node reports/2025-promising-rag-architecture.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
```

Require six passes.

- [x] **Step 4: Check the rendered pages at desktop and mobile widths**

Capture both self-contained pages through the repository's private-profile Chrome wrapper:

```bash
repo_root=$(pwd)
qa_dir=$(mktemp -d)
python3 code/tools/run_headless_chrome.py --url "file://$repo_root/index.html" --output "$qa_dir/root-desktop.png" --width 1440 --height 1000
python3 code/tools/run_headless_chrome.py --url "file://$repo_root/index.html" --output "$qa_dir/root-mobile.png" --width 390 --height 844
python3 code/tools/run_headless_chrome.py --url "file://$repo_root/reports/2026-competition-architecture.html" --output "$qa_dir/architecture-desktop.png" --width 1440 --height 1000
python3 code/tools/run_headless_chrome.py --url "file://$repo_root/reports/2026-competition-architecture.html" --output "$qa_dir/architecture-mobile.png" --width 390 --height 844
```

Inspect all four paths with the image-view tool for clipped text, horizontal overflow, unreadable cards/diagrams, missing content, or broken layout. Keep the temporary screenshots out of git.

- [x] **Step 5: Run final repository checks**

```bash
git diff --check
git status --short
git diff --stat 88a163bf..HEAD
```

Confirm only scoped documentation, tests, SVG assets, generated architecture HTML, the approved spec, and this plan changed. Confirm all five organizer-file hashes still match and no ignored/private path is staged.

- [x] **Step 6: Record verification evidence and commit the completed plan**

Append a `## Verification Evidence` section containing the exact commands, pass counts, hash comparison, Quarto version, and responsive QA result; mark plan checkboxes complete.

```bash
git add docs/superpowers/plans/2026-08-09-final-submission-documentation-handoff.md
git commit -m "docs: record final handoff verification"
```

## Verification Evidence

Verified from the final worktree on 2026-08-09 without retrieval, reranking,
generation, evaluation, hosted-model calls, or publication.

- Submission contract: pinned `trec-rag-data` revision
  `a6255c3af2b4595789640546c1c96bff02f46871` and pinned
  `trec-rag-skills` revision `f281e8800ae20088ebed4b85c075e71fc37d28b0`
  matched their upstream default-branch heads. The official topics file SHA-256
  was `72dc2f413a97ad7d9f5d57d6c67df6602a8431130a71d1b45e25e7714f17b6e6`
  and contained 119 narratives.
- Five-file preflight: the combined
  `.agents/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py`
  command in Task 5 returned three Retrieval `PASS` results, each with 4,246
  rows, 119 topics, and depth 1–121, plus two RAG `PASS` results, each with
  119/119 valid reports, no extras, and no duplicates.
- Immutable bytes: `sha256sum` returned, in ledger order,
  `29bc0c29dd51a752d49c734db456926ef94ad5202102cabd92d7fbf3e9dd15e8`,
  `f42a794418cf692721adcd53df232ca0a3536d13d9cd137d90eb9821562e199d`,
  `80985a42e43333085975c27d88a1da82cc11cba6dd576c3fd506839c016f2feb`,
  `72200f7a0e3be19f9c7e8f23d3845f894f5e95e26caff2986ae16f848b4dee00`,
  and `a33c60325c198cf90178d5b4c6c1b04209f7d39de278736b6817a8c5666c02a0`.
  `git diff --name-only 88a163bf -- 'submissions/**/*.tsv'
  'submissions/**/*.jsonl' 'submissions/**/metadata.json'` returned no paths.
- Documentation tests: the six `node` commands in Task 5 all passed. The root
  hub, report collection, final architecture, briefing, 2025 architecture, and
  interactive-writeup checks each emitted their expected pass message.
- Link integrity: a repository-relative HTML link check, excluding embedded
  script and style source, resolved all 195 checked links.
- Canonical render: `quarto render
  reports/2026-competition-architecture.qmd` completed with Quarto 1.9.38, and
  the architecture smoke test passed against the regenerated HTML.
- Responsive QA: Chrome captures at 1440×1000 and 390×844 were inspected for
  the root hub and architecture report. Additional tall desktop captures were
  inspected across all nine diagrams and the accepted-artifact cards. No
  clipped text, horizontal overflow, unreadable cards or diagrams, missing
  content, or broken layout was observed. Screenshots remained temporary and
  outside git.
- Repository hygiene: `git diff --check` passed. The final diff from
  `88a163bf` contains only the approved documentation, tests, diagram assets,
  generated architecture HTML, specification, and plan; no ignored/private
  output was added.

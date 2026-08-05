# TREC RAG 2026 Competition Architecture Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a friend-readable, source-backed Quarto guide that follows one fictional narrative through the supported TREC RAG 2026 retrieval, selected-evidence handoff, GPT-5.6 Sol generation, validation, and organizer outputs.

**Architecture:** Keep `reports/2026-competition-architecture.qmd` as the only authored report source and commit its self-contained Quarto HTML rendering beside it. Keep nine accessible, theme-aware SVG diagrams as reusable local assets, with one small stylesheet and no custom JavaScript unless browser evidence proves it necessary. Protect the report's architecture facts, asset contract, generated output, and discovery links with Node smoke tests, then inspect the rendered result in real headless Chrome.

**Tech Stack:** Quarto 1.9.38, Markdown, standalone HTML with embedded resources, SVG 1.1/2-compatible markup, CSS custom properties, Node.js standard-library smoke tests, headless Google Chrome.

## Global Constraints

- Start from merge commit `7e9d3b9159f5222a761ba58ad515b236dc93a423` in the clean worktree and branch already created for this feature.
- Cover only the supported 2026 competition path implemented by `trec_rag.competition_retrieval` and `trec_rag.competition_rag`.
- Keep `reports/2026-competition-architecture.qmd` canonical; never hand-edit the rendered HTML.
- Do not add `_quarto.yml` or convert the repository into a Quarto project.
- Commit a self-contained `reports/2026-competition-architecture.html` with no viewing-time network dependency.
- Keep each of the nine figures as an individually viewable, accessible, theme-aware SVG under `reports/2026-competition-architecture/`.
- Use conventional architecture semantics: solid arrows for data/evidence flow, dashed arrows for advisory or control flow, cylinders for stores, documents for artifacts, and a gate for validation.
- Use a conceptual selection funnel only; do not imply proportional area or throughput where no measured proportion exists.
- Use the fictional extreme-heat narrative only; include no real test narrative, corpus text, generated answer, cache, provider response, qrels, gold nugget, RAGDoll score, or private run artifact.
- Treat `≤1,000 documents / focused query` and `≤100 passages / focused query` as ceilings with a unit change, never as guaranteed counts or one global topic total.
- State that selected passages are factual authority and canonical claim hints are advisory.
- Show that Generation consumes only the authenticated `generation_handoff_manifest.json` and never opens the retrieval run, full-text ZIP, qrels, gold nuggets, or RAGDoll scores.
- Keep transport attempts (`3`) separate from semantic attempts (`2`).
- Support light, dark, mobile, desktop, print, keyboard, reduced-motion, and grayscale use without relying on color alone.
- Preserve all existing operational commands, run-safety rules, report links, tests, and unrelated work.

---

### Task 1: Quarto Source Contract and Minimal Render

**Files:**
- Create: `reports/2026-competition-architecture.test.js`
- Create: `reports/2026-competition-architecture.qmd`
- Create: `reports/2026-competition-architecture/report.css`
- Generate: `reports/2026-competition-architecture.html`
- Read: `docs/superpowers/specs/2026-08-05-trec-rag-2026-competition-architecture-guide-design.md`

**Interfaces:**
- Consumes: the approved design specification and the installed `quarto` executable.
- Produces: a canonical `.qmd`, a local style source, a generated self-contained HTML file, and a smoke-test harness reused by later tasks.

- [ ] **Step 1: Write the failing Quarto contract test**

Create `reports/2026-competition-architecture.test.js` with Node standard-library assertions. The first slice must verify:

```js
const fs = require("node:fs");
const path = require("node:path");

const reportsRoot = __dirname;
const sourcePath = path.join(reportsRoot, "2026-competition-architecture.qmd");
const htmlPath = path.join(reportsRoot, "2026-competition-architecture.html");
const assetRoot = path.join(reportsRoot, "2026-competition-architecture");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(sourcePath), "canonical Quarto source should exist");
assert(fs.existsSync(htmlPath), "rendered architecture HTML should exist");
assert(fs.existsSync(path.join(assetRoot, "report.css")), "local report CSS should exist");
assert(!fs.existsSync(path.join(reportsRoot, "..", "_quarto.yml")), "repo-wide Quarto config must not exist");

const qmd = fs.readFileSync(sourcePath, "utf8");
const html = fs.readFileSync(htmlPath, "utf8");

for (const signal of [
  "embed-resources: true",
  "light: flatly",
  "dark: darkly",
  "toc: true",
  "lightbox: true",
  "2026-competition-architecture/report.css",
]) {
  assert(qmd.includes(signal), `Missing Quarto source contract: ${signal}`);
}

assert(/<meta[^>]+name="generator"[^>]+quarto/i.test(html), "HTML should identify Quarto as generator");
assert(!/<(?:script|link|img)[^>]+(?:src|href)="https?:/i.test(html), "HTML runtime assets must be local or embedded");
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
node reports/2026-competition-architecture.test.js
```

Expected: FAIL with `canonical Quarto source should exist`.

- [ ] **Step 3: Add the minimal standalone Quarto source and style source**

Create the `.qmd` with this rendering contract and a temporary heading/body sufficient to render:

```yaml
---
title: "From one narrative to a TREC RAG 2026 submission"
subtitle: "The supported competition architecture, told as a guided story"
lang: en
format:
  html:
    embed-resources: true
    toc: true
    toc-depth: 2
    smooth-scroll: true
    theme:
      light: flatly
      dark: darkly
    css: 2026-competition-architecture/report.css
lightbox: true
---
```

Create `report.css` with scoped design tokens, readable line lengths, responsive figure sizing, visible focus, print rules that expose disclosure content, and `prefers-reduced-motion`. Use `light-dark(...)` only with a declared `color-scheme`; keep a conventional rectangular visual grammar and avoid decorative card grids.

- [ ] **Step 4: Render and verify GREEN**

Run:

```bash
quarto render reports/2026-competition-architecture.qmd --to html
node reports/2026-competition-architecture.test.js
```

Expected: Quarto exits 0 and the smoke test reaches the end without an assertion failure.

- [ ] **Step 5: Commit the Quarto shell**

```bash
git add reports/2026-competition-architecture.qmd \
  reports/2026-competition-architecture.html \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture/report.css
git commit -m "docs: establish Quarto architecture guide"
```

### Task 2: Nine Reusable Architecture Figures

**Files:**
- Modify: `reports/2026-competition-architecture.test.js`
- Create: `reports/2026-competition-architecture/01-whole-system.svg`
- Create: `reports/2026-competition-architecture/02-retrieval-system.svg`
- Create: `reports/2026-competition-architecture/03-bounded-deepseek-planning.svg`
- Create: `reports/2026-competition-architecture/04-per-query-candidate-accounting.svg`
- Create: `reports/2026-competition-architecture/05-evidence-and-nuggetizer.svg`
- Create: `reports/2026-competition-architecture/06-generation-handoff-contract.svg`
- Create: `reports/2026-competition-architecture/07-sol-generation.svg`
- Create: `reports/2026-competition-architecture/08-validation-and-retries.svg`
- Create: `reports/2026-competition-architecture/09-organizer-output-split.svg`
- Create: `reports/2026-competition-architecture/README.md`

**Interfaces:**
- Consumes: stable colors and shape semantics from the approved specification; source facts from the two checked-in configs and the competition modules.
- Produces: nine SVGs with stable file names, unique `title`/`desc` identifiers, internal theme-aware CSS, visible legends, and readable mobile compositions.

- [ ] **Step 1: Extend the smoke test for the asset contract**

Add the exact file list and checks before any SVG exists:

```js
const figures = [
  "01-whole-system.svg",
  "02-retrieval-system.svg",
  "03-bounded-deepseek-planning.svg",
  "04-per-query-candidate-accounting.svg",
  "05-evidence-and-nuggetizer.svg",
  "06-generation-handoff-contract.svg",
  "07-sol-generation.svg",
  "08-validation-and-retries.svg",
  "09-organizer-output-split.svg",
];

for (const figure of figures) {
  const figurePath = path.join(assetRoot, figure);
  assert(fs.existsSync(figurePath), `Missing architecture figure: ${figure}`);
  const svg = fs.readFileSync(figurePath, "utf8");
  assert(svg.includes('role="img"'), `${figure} should expose image semantics`);
  assert(/aria-labelledby="[^"]+ [^"]+"/.test(svg), `${figure} should reference title and description`);
  assert(/<title id="[^"]+">[^<]+<\/title>/.test(svg), `${figure} should have a titled accessible name`);
  assert(/<desc id="[^"]+">[^<]+<\/desc>/.test(svg), `${figure} should have an accessible description`);
  assert(svg.includes("prefers-color-scheme: dark"), `${figure} should support dark mode`);
  assert(svg.includes("vector-effect=\"non-scaling-stroke\""), `${figure} should retain line weight when enlarged`);
}
```

Also require the asset README and a row for every file.

- [ ] **Step 2: Run the test and verify RED**

Run `node reports/2026-competition-architecture.test.js`.

Expected: FAIL with `Missing architecture figure: 01-whole-system.svg`.

- [ ] **Step 3: Create the shared SVG grammar in every standalone figure**

Each SVG must define its own variables and accessibility structure:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720"
     role="img" aria-labelledby="figure-N-title figure-N-desc">
  <title id="figure-N-title">Concise figure title</title>
  <desc id="figure-N-desc">Complete plain-language description of nodes, direction, and boundary.</desc>
  <style>
    :root { color-scheme: light dark; --bg:#f7f8f6; --ink:#17201c; --muted:#52615a; --line:#53615b; --retrieval:#147d73; --generation:#5b67c8; --advisory:#a06712; --danger:#b5473b; --panel:#ffffff; }
    @media (prefers-color-scheme: dark) { :root { --bg:#0f1512; --ink:#eef5f0; --muted:#b8c5be; --line:#a7b5ad; --retrieval:#63c9bc; --generation:#9ca9ff; --advisory:#efbd67; --danger:#ff9187; --panel:#17201c; } }
    text { fill:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    .flow { fill:none; stroke:var(--line); stroke-width:2; vector-effect:non-scaling-stroke; marker-end:url(#arrow); }
    .advisory { stroke-dasharray:8 7; stroke:var(--advisory); }
  </style>
```

Use visible words plus shape or line differences for every color-coded meaning. Keep body labels at least 16 SVG units in the declared view box so they remain at least 11 screen pixels in the report's normal figure width.

- [ ] **Step 4: Draw the nine focused figures with these exact messages**

1. `01-whole-system.svg`: Narrative → Retrieval → authenticated selected-evidence handoff → GPT-5.6 Sol Generation → validation → organizer RAG JSONL, plus a separate Retrieval branch to TREC run and full-text ZIP. Put the Retrieval/Generation trust boundary at the handoff.
2. `02-retrieval-system.svg`: One original narrative lane plus `0–8` DeepSeek-planned subnarratives with `1–3` BM25 queries each, yielding `1–25` executable query lanes. Each query enters the same Pyserini → chunk → Mixedbread → top-passage boundary, then local selection and advisory canonicalization converge into retrieval outputs.
3. `03-bounded-deepseek-planning.svg`: One strict, one-shot DeepSeek request produces `0–8` focused subnarratives with `1–3` BM25 queries each while the untouched narrative always remains a lane. Show the `1–25` total query-lane range and failure returning to the original-only lane, not an open-ended tool loop.
4. `04-per-query-candidate-accounting.svg`: A non-proportional accounting strip/funnel labeled `one original or planned BM25 query`, `≤1,000 documents`, `chunks (3500 characters, 350 overlap)`, Mixedbread scoring, and `≤100 passages`. Explicitly mark the documents→passages unit change and repeat-per-query scope.
5. `05-evidence-and-nuggetizer.svg`: Top passages → local exact-span scoring/deduplication/diversity selection (`≤40 evidence items per subnarrative`) → factual selected passages. A dashed advisory lane sends the same selection through Nuggetizer to `≤20 claim hints`, `≤3 supporting documents per claim`; show fallback to exact extractive evidence.
6. `06-generation-handoff-contract.svg`: A document-shaped authenticated manifest containing narrative, selected evidence groups, advisory claim hints, citation-domain document IDs, source receipts, and hashes. Around it, show barred inputs: retrieval run, text ZIP, qrels, gold nuggets, and RAGDoll scores.
7. `07-sol-generation.svg`: GPT-5.6 Sol consumes only the handoff projection, uses model `openai/gpt-5.6-sol`, medium reasoning, strict structured output, and a 12,000-token ceiling, then returns answer objects with raw document-ID citations.
8. `08-validation-and-retries.svg`: Separate nested retry scopes: each hosted transport request has `≤3 transport attempts`; a topic has `≤2 semantic attempts`. After generation, validate schema, 1,024-word ceiling, 1–3 unique citations, allowed citation domain, and exact-hint compatibility; only then deterministically convert document IDs to organizer indexes.
9. `09-organizer-output-split.svg`: Retrieval publishes the official TREC evidence run, full-text ZIP, and manifest-last receipt; Generation publishes organizer RAG JSONL. Make it visually impossible to infer that retrieval export artifacts feed Generation.

- [ ] **Step 5: Add the asset provenance README**

Create a table with columns `Figure`, `Reader question`, `Primary source`, and `Text equivalent`. Cite repository-relative paths to the fixed configs and modules. State that the extreme-heat example is fictional and that diagram geometry is conceptual rather than measured throughput.

- [ ] **Step 6: Run the smoke test and verify GREEN**

Run `node reports/2026-competition-architecture.test.js`.

Expected: all asset-path, title, description, dark-mode, and line-scaling assertions pass.

- [ ] **Step 7: Commit the reusable figures**

```bash
git add reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture/README.md \
  reports/2026-competition-architecture/*.svg
git commit -m "docs: add 2026 architecture figures"
```

### Task 3: Guided Story and Source-Backed Technical Notes

**Files:**
- Modify: `reports/2026-competition-architecture.test.js`
- Modify: `reports/2026-competition-architecture.qmd`
- Modify: `reports/2026-competition-architecture/report.css`
- Generate: `reports/2026-competition-architecture.html`
- Read: `configs/rag26_competition_retrieval_v2.yaml`
- Read: `configs/rag26_competition_rag_gpt_sol_v2.yaml`
- Read: `code/trec_rag/README.md`
- Read: `code/trec_rag/facet_extraction.py`
- Read: `code/trec_rag/canonical_nuggets.py`
- Read: `code/trec_rag/generation_handoff.py`
- Read: `code/trec_rag/competition_retrieval.py`
- Read: `code/trec_rag/competition_rag.py`

**Interfaces:**
- Consumes: the nine figure paths and verified implementation/config facts.
- Produces: the complete visible story, glossary, text equivalents, source disclosures, direct SVG links, and rendered standalone HTML.

- [ ] **Step 1: Extend the report test with content and boundary invariants**

Add exact signal arrays covering:

```js
const requiredConcepts = [
  "Illustrative narrative — not a TREC test topic",
  "DeepSeek V4 Flash",
  "deepseek/deepseek-v4-flash-20260423",
  "0–8 focused subnarratives",
  "Pyserini",
  "ClimbMix",
  "≤1,000 documents / focused query",
  "3500 characters",
  "350-character overlap",
  "Mixedbread",
  "mixedbread-ai/mxbai-rerank-base-v2",
  "≤100 passages / focused query",
  "Nuggetizer",
  "Selected passages are factual authority",
  "Canonical claim hints are advisory",
  "generation_handoff_manifest.json",
  "openai/gpt-5.6-sol",
  "12,000-token ceiling",
  "≤3 transport attempts",
  "≤2 semantic attempts",
  "1,024-word ceiling",
  "organizer RAG JSONL",
];
```

Require all nine figure references in numeric order, nine visible `Text equivalent:` blocks, direct `.svg` links, `<details>` technical notes, the solid/dashed legend, and the forbidden-input names. Forbid secret names/values beyond the documented environment-variable names, real topic IDs such as `rag2026-`, external runtime URLs, `TODO`, and `placeholder`.

- [ ] **Step 2: Run the test and verify RED**

Run `node reports/2026-competition-architecture.test.js`.

Expected: FAIL on the first missing narrative or architecture signal.

- [ ] **Step 3: Author the complete vertical story in Quarto Markdown**

Use these top-level sections and stable IDs:

```markdown
# The whole system {#whole-system}
# Retrieval: widen, score, then narrow {#retrieval}
# Planning is bounded {#planning}
# One query changes units {#candidate-accounting}
# Evidence first; hints second {#evidence}
# The handoff is the trust boundary {#handoff}
# Generation with GPT-5.6 Sol {#generation}
# Validation has two retry clocks {#validation}
# Two organizer-facing outputs {#outputs}
# Sources and scope {#sources}
```

Open with this fictional example, clearly labeled illustrative: a friend asks how a city should protect outdoor workers and runners during an extreme-heat week, including hydration, schedule changes, warning signs, and air-quality interactions. Carry the same example through queries, candidate documents, evidence passages, advisory hints, answer objects, and citations without quoting any corpus text or inventing a completed model answer.

For every figure use Quarto figure syntax with an ID and caption, immediately followed by an ordinary Markdown link to the source SVG, one short explanation, a visible `Text equivalent:` paragraph, and a native `<details><summary>Implementation notes and sources</summary>…</details>` block. The notes must distinguish human-friendly product names in the figure from exact model/config identifiers.

- [ ] **Step 4: Add reader aids without adding dashboard chrome**

Add a compact opening glossary for narrative, query lane, document, chunk, passage, selected evidence, canonical hint, handoff, and citation. Add one stable legend for solid evidence flow, dashed advisory/control flow, artifact documents, stores, gates, and the Retrieval/Generation color families. Keep the 1,000/100 values as small inline badges near their explanations, not as KPI cards.

- [ ] **Step 5: Render and run the focused test**

Run:

```bash
quarto render reports/2026-competition-architecture.qmd --to html
node reports/2026-competition-architecture.test.js
```

Expected: render succeeds and every content, order, boundary, safety, and local-link assertion passes.

- [ ] **Step 6: Commit the complete guide**

```bash
git add reports/2026-competition-architecture.qmd \
  reports/2026-competition-architecture.html \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture/report.css
git commit -m "docs: tell the 2026 competition architecture story"
```

### Task 4: Agent Orientation and Report Discovery

**Files:**
- Modify: `reports/2026-competition-architecture.test.js`
- Modify: `reports/index.test.js`
- Modify: `reports/index.html`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: stable paths to the canonical `.qmd`, generated `.html`, and SVG directory.
- Produces: concise contributor guidance and friend-facing discovery without duplicating the full guide.

- [ ] **Step 1: Extend discovery tests before changing documentation**

In `reports/index.test.js`, require these signals and link target:

```js
"TREC RAG 2026 Competition Architecture",
"Narrative → Retrieval → Handoff → Generation",
"2026-competition-architecture.html",
```

Add `"2026-competition-architecture.html"` to `linkedFiles`.

In the focused architecture test, read `../AGENTS.md` and `../README.md` and require both report paths, the `Architecture Orientation` heading, `Retrieval → handoff → Generation`, the per-query unit-change warning, selected-passages authority, authenticated-handoff-only rule, forbidden Generation inputs, and separate retry scopes.

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
node reports/2026-competition-architecture.test.js
node reports/index.test.js
```

Expected: both fail on their first missing discovery/orientation signal.

- [ ] **Step 3: Add the concise AGENTS orientation**

Add `## Architecture Orientation` immediately before the existing competition-run section. Link the rendered walkthrough for reading and the `.qmd` for editing. State the seven invariants from the design specification in one compact list; keep the existing commands and operational cautions unchanged.

- [ ] **Step 4: Add friend-facing README and index links**

Add the architecture guide to `README.md` under Current Contents, naming both the generated reader view and canonical Quarto source. Add one report card to the existing reports index using its established `.report-card` pattern and linking to `2026-competition-architecture.html`.

- [ ] **Step 5: Run discovery and regression tests**

Run:

```bash
node reports/2026-competition-architecture.test.js
node reports/index.test.js
node reports/trec-rag-briefing-report.test.js
node reports/2025-promising-rag-architecture.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
```

Expected: all five commands exit 0.

- [ ] **Step 6: Commit discovery and agent documentation**

```bash
git add AGENTS.md README.md reports/index.html reports/index.test.js \
  reports/2026-competition-architecture.test.js
git commit -m "docs: link the 2026 architecture guide"
```

### Task 5: Browser QA, Accessibility Polish, and Reproducible Render

**Files:**
- Modify only if a failing check requires it: `reports/2026-competition-architecture.qmd`
- Modify only if a failing check requires it: `reports/2026-competition-architecture/report.css`
- Modify only if a failing check requires it: `reports/2026-competition-architecture/*.svg`
- Regenerate after source fixes: `reports/2026-competition-architecture.html`
- Store temporary QA output only beneath a newly created `/tmp/trec-rag-architecture-qa-*` directory.

**Interfaces:**
- Consumes: the rendered report and nine direct SVGs.
- Produces: fresh evidence that the generated artifact is reproducible and legible across required rendering modes; no committed QA screenshots or browser state.

- [ ] **Step 1: Prove deterministic rendering from authored source**

Run:

```bash
quarto --version
quarto render reports/2026-competition-architecture.qmd --to html
git diff --exit-code -- reports/2026-competition-architecture.html
quarto render reports/2026-competition-architecture.qmd --to html
git diff --exit-code -- reports/2026-competition-architecture.html
```

Expected: Quarto reports `1.9.38`; both diffs exit 0.

- [ ] **Step 2: Capture fresh desktop and mobile light/dark evidence**

Create a new scratch root with `mktemp -d /tmp/trec-rag-architecture-qa-XXXXXX`. Use `code/tools/run_headless_chrome.py` with the scratch root explicitly set for 1440×1000 and 390×844 light screenshots. Use a fresh Chrome profile beneath the same scratch root and `--force-dark-mode` for matching dark screenshots. Load the generated report by absolute `file://` URL so no service is exposed.

Expected: four nonempty PNGs, no browser-state reuse, and no network dependency.

- [ ] **Step 3: Inspect full-page and focused figure crops**

Inspect all four screenshots plus figures 01, 02, 04, 06, and 08 at original detail. Verify no page-level horizontal overflow, clipped labels, overlaps, unreadably scaled text, low-contrast annotations, or ambiguous boundary arrows. Convert one desktop screenshot to grayscale when ImageMagick is available and verify the solid/dashed/shape encodings still carry the meaning.

- [ ] **Step 4: Verify keyboard, disclosure, lightbox, and print behavior**

Use headless Chrome DOM output or an available browser automation interface to verify all section anchors exist, nine disclosures are operable, nine figure-enlargement links are present, and no failed local asset request occurs. Print the page to a PDF in the scratch root and inspect representative pages to confirm figures, captions, text equivalents, and open disclosure content remain visible.

- [ ] **Step 5: Fix each observed defect test-first**

For every defect, add the smallest static assertion that would catch it when possible, run it to observe failure, patch only the `.qmd`, CSS source, or relevant SVG, re-render, and rerun the focused test. Never patch the generated HTML directly.

- [ ] **Step 6: Commit verified visual fixes if any**

If source fixes were required:

```bash
git add reports/2026-competition-architecture.qmd \
  reports/2026-competition-architecture.html \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture/report.css \
  reports/2026-competition-architecture/*.svg
git commit -m "docs: polish architecture guide rendering"
```

If no source fix was required, make no empty commit.

### Task 6: Independent Review and Final Verification

**Files:**
- Review: all changes from `7e9d3b9159f5222a761ba58ad515b236dc93a423` through the feature head.
- Modify only to resolve validated review findings.

**Interfaces:**
- Consumes: the approved design spec, this plan, the complete branch diff, and fresh test/browser evidence.
- Produces: an independently reviewed, clean branch ready for user handoff.

- [ ] **Step 1: Request an independent substantive review**

Provide the reviewer the base SHA, head SHA, spec path, plan path, and this scope: architecture-fact accuracy, Quarto source/generated-output discipline, SVG semantics, accessibility, mobile/dark rendering, link integrity, private-data exclusion, and regression risk.

- [ ] **Step 2: Resolve every Critical or Important finding**

For a valid finding, add or strengthen a failing test first, patch the authored source or SVG, re-render, and rerun the focused and affected regression tests. Explain and reject a finding only when repository evidence proves it incorrect.

- [ ] **Step 3: Run the complete final verification from a clean state**

Run:

```bash
git diff --check 7e9d3b9159f5222a761ba58ad515b236dc93a423..HEAD
quarto render reports/2026-competition-architecture.qmd --to html
git diff --exit-code -- reports/2026-competition-architecture.html
node reports/2026-competition-architecture.test.js
node reports/index.test.js
node reports/trec-rag-briefing-report.test.js
node reports/2025-promising-rag-architecture.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
git status --short --branch
```

Expected: no whitespace errors, render exits 0, generated HTML has no diff, all five test commands pass, and the worktree has no uncommitted changes.

- [ ] **Step 4: Check the privacy and artifact boundary**

Run a tracked-file scan for `.env`, `cache/`, `outputs/`, raw JSONL/ZIP artifacts, provider payloads, token-shaped values, and real `rag2026-` topic identifiers. Confirm the branch contains only documentation/report sources, generated report output, reusable SVGs, tests, and intended discovery edits.

- [ ] **Step 5: Record completion evidence**

Update this plan's checkboxes and add a short final evidence note containing the Quarto version, focused/regression test results, browser viewports/themes inspected, independent-review disposition, and final head SHA. Commit the updated plan only after all evidence is current.

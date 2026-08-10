# Final Analysis Navigation and Dark-Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put three direct final-analysis links on the completed-project hub, repair inline-code contrast in the Quarto architecture report, and refresh the tailnet-only portal without exposing private RAGDoll material.

**Architecture:** Keep the existing tracked 119-topic Retrieval report repository-relative, while linking the two completed RAGDoll detail reports to stable private tailnet URLs and labeling that access boundary. Fix dark mode only in canonical report CSS, rerender the QMD, update reader/agent entry points, then publish sanitized derived HTML copies through the existing Tailscale Serve portal.

**Tech Stack:** Standalone HTML/CSS, Quarto 1.9.38, Node.js smoke tests, headless Chrome/DevTools Protocol, Tailscale Serve.

## Global Constraints

- The hub contains exactly three direct analysis links: Retrieval Quality Analysis, RAG Analysis `rag26-ss1`, and RAG Analysis `rag26-ms1-final`; do not add a comparison landing page.
- Retrieval targets `reports/2026-retrieval-nugget-coverage.html`; RAG targets are the verified tailnet URLs from `2026-08-10-accepted-rag-ragdoll-evaluation.md`.
- Mark both RAG links `Private / tailnet`; do not imply public GitHub Pages visitors can open them.
- Edit `reports/2026-competition-architecture.qmd` and `reports/2026-competition-architecture/report.css`, then rerender; never patch generated architecture HTML directly.
- Dark inline code must not use Bootstrap light gray `rgb(248, 249, 250)` and must meet WCAG AA 4.5:1 contrast; source-code blocks remain unchanged.
- Do not copy raw evaluation inputs, passages, tasks, events, judgments, assignments, manifests, private paths, or secrets into git or the portal.
- Do not modify any accepted organizer file or enable/deploy GitHub Pages, Funnel, public listeners, or Codex Sites.

---

### Task 1: Repair canonical Quarto inline-code theming

**Files:**
- Modify: `reports/2026-competition-architecture/report.css`
- Modify: `reports/2026-competition-architecture.test.js`
- Regenerate: `reports/2026-competition-architecture.html`

**Interfaces:**
- Produces CSS variables `--guide-code-ink`, `--guide-code-bg`, and `--guide-code-border` in light/dark scopes.
- Produces an inline-code selector that overrides Bootstrap while excluding `pre code` and `.sourceCode` blocks.

- [ ] **Step 1: Add a failing source/render contract test**

```javascript
for (const token of ["--guide-code-ink", "--guide-code-bg", "--guide-code-border"]) {
  assert(css.includes(token), `report CSS should define ${token}`);
}
assert(
  css.includes("body.quarto-dark :not(pre) > code:not(.sourceCode)"),
  "dark mode should explicitly theme inline code without changing code blocks",
);
assert(html.includes("--guide-code-bg"), "rendered HTML should contain canonical code tokens");
```

- [ ] **Step 2: Run the smoke test and verify failure**

```bash
node reports/2026-competition-architecture.test.js
```

Expected: failure for missing code-background tokens.

- [ ] **Step 3: Add explicit accessible inline-code tokens**

```css
:root {
  --guide-code-ink: #6f3488;
  --guide-code-bg: #f3e8f7;
  --guide-code-border: #d8b9e3;
}

body.quarto-dark {
  --guide-code-ink: #f0cfff;
  --guide-code-bg: #2c2332;
  --guide-code-border: #765484;
}

:not(pre) > code:not(.sourceCode) {
  border: 1px solid var(--guide-code-border);
  background: var(--guide-code-bg);
  color: var(--guide-code-ink);
}

body.quarto-dark :not(pre) > code:not(.sourceCode) {
  border-color: var(--guide-code-border);
  background: var(--guide-code-bg);
  color: var(--guide-code-ink);
}
```

Keep `code { overflow-wrap: anywhere; }`; do not add backgrounds to `pre`, `.sourceCode`, or syntax-token selectors.

- [ ] **Step 4: Re-render from QMD and run tests**

```bash
quarto render reports/2026-competition-architecture.qmd
node reports/2026-competition-architecture.test.js
```

Expected: successful standalone HTML regeneration and passing test.

- [ ] **Step 5: Verify computed contrast in toggled dark mode**

Open the rendered file in headless Chrome, click `.quarto-color-scheme-toggle`, select `facet:<subnarrative>:text`, capture `getComputedStyle`, and compute relative luminance/contrast. Inspect a source-code block and 390-pixel mobile viewport.

Expected: `quarto-dark`; inline foreground `rgb(240, 207, 255)`; background `rgb(44, 35, 50)`; contrast at least 4.5:1; code-block background unchanged; no clipping.

- [ ] **Step 6: Commit the dark-mode repair**

```bash
git add reports/2026-competition-architecture/report.css \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture.html
git commit -m "fix: make architecture code dark-mode readable"
```

---

### Task 2: Add the three direct reader-facing analysis links

**Files:**
- Modify: `index.html`
- Modify: `index.test.js`
- Modify: `reports/index.html`
- Modify: `reports/index.test.js`
- Modify: `reports/2026-competition-architecture.qmd`
- Modify: `reports/2026-competition-architecture.test.js`
- Regenerate: `reports/2026-competition-architecture.html`

**Interfaces:**
- Consumes: verified final RAG report URLs from the evaluation plan.
- Produces identical link labels/access wording across root hub, report index, and architecture artifact map.

- [ ] **Step 1: Add failing three-link navigation tests**

```javascript
const analysisLinks = [
  ["Retrieval Quality Analysis", "reports/2026-retrieval-nugget-coverage.html"],
  ["RAG Analysis: rag26-ss1", "https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html"],
  ["RAG Analysis: rag26-ms1-final", "https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html"],
];
```

For `reports/index.html`, use `2026-retrieval-nugget-coverage.html`. Assert both RAG entries have nearby `Private / tailnet` text and no comparison-report link.

- [ ] **Step 2: Run tests and verify failure**

```bash
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
```

Expected: failures for missing analysis labels/targets.

- [ ] **Step 3: Add one analysis group to the root hub**

Add `Final analyses` near architecture/submissions. Use exact labels/links above. Give private links an accessible `<span class="access-note">Private / tailnet</span>` suffix; do not make access text the only link text.

- [ ] **Step 4: Add the same direct choices to the reports index**

Place the analyses before historical 2025 material. Describe Retrieval as the existing 119-topic coverage analysis and each RAG page as a separate 119-topic RAGDoll citation-support report. Avoid winner language.

- [ ] **Step 5: Add evaluation artifacts to the QMD**

State RAGDoll measures citation support, not official TREC correctness; qrel/gold metrics are unavailable; and evaluation did not influence accepted priority. Link all three reports and mark RAG detail private.

- [ ] **Step 6: Re-render, test, and commit**

```bash
quarto render reports/2026-competition-architecture.qmd
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
git add index.html index.test.js reports/index.html reports/index.test.js \
  reports/2026-competition-architecture.qmd \
  reports/2026-competition-architecture.test.js \
  reports/2026-competition-architecture.html
git commit -m "docs: link final retrieval and RAG analyses"
```

Expected: all tests pass and rendered architecture contains all three targets.

---

### Task 3: Orient future agents to the final analyses

**Files:**
- Modify: `AGENTS.md`
- Modify: `reports/2026-competition-architecture.test.js`

**Interfaces:**
- Consumes: three canonical links and evaluation limitations.
- Produces: completed-project orientation distinguishing tracked Retrieval analysis, private RAG detail, accepted immutable inputs, and evaluator skill.

- [ ] **Step 1: Add failing AGENTS orientation assertions**

```javascript
const evaluationSignals = [
  "reports/2026-retrieval-nugget-coverage.html",
  "rag26-ss1",
  "rag26-ms1-final",
  "Private / tailnet",
  ".agents/skills/trec-rag-competition-debug-report/SKILL.md",
  "citation support",
  "not an official TREC score",
];
```

- [ ] **Step 2: Run the architecture test and verify failure**

```bash
node reports/2026-competition-architecture.test.js
```

Expected: failure for missing orientation.

- [ ] **Step 3: Update completed-project/key-artifacts orientation**

Add the tracked Retrieval report and two stable private URLs. Point to accepted RAG JSONLs as immutable evaluated inputs and to the competition debug-report skill for reproduction. State raw evaluation work stays outside git and RAGDoll results neither changed accepted files nor establish official correctness.

- [ ] **Step 4: Run test and commit**

```bash
node reports/2026-competition-architecture.test.js
git add AGENTS.md reports/2026-competition-architecture.test.js
git commit -m "docs: orient agents to final analyses"
```

Expected: test passes and only assigned files are committed.

---

### Task 4: Refresh and verify the private rendered portal

**Files:**
- Update derived/private: `/home/npatta01/codex-rendered/plans/trec-rag-2026-final-project.html`
- Update derived/private: `/home/npatta01/codex-rendered/plans/trec-rag-2026-competition-architecture.html`
- Reuse derived/private: `/home/npatta01/codex-rendered/plans/trec-rag-2026-retrieval-nugget-coverage.html`
- Update derived/private: `/home/npatta01/codex-rendered/index.html`

**Interfaces:**
- Consumes: tracked hub/architecture/Retrieval HTML and two privacy-scanned RAG reports.
- Produces: five-page private final-project viewing set with three direct analysis links and no broken repository-relative targets.

- [ ] **Step 1: Re-run tracked verification**

```bash
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
git diff --check
```

Run the five-file validator/hashes. Expected: pass and accepted bytes match ledger.

- [ ] **Step 2: Refresh only sanitized derived HTML**

Copy root hub, architecture, and Retrieval report to existing portal filenames. Rewrite repository-relative links only in derived flat-directory copies; do not modify tracked HTML during copy. Confirm both RAG report files exist from the evaluation plan.

- [ ] **Step 3: Update the private portal index**

Group final hub, architecture, Retrieval quality, and both RAG analyses. Label RAG reports by exact run ID. Remove no unrelated entry unless it is a superseded copy owned by this task.

- [ ] **Step 4: Browser-test the live portal**

At desktop and 390-pixel mobile widths, open the hub, follow all three analysis links, toggle architecture dark mode, and inspect glossary code. Check focus, overflow, run IDs, task totals, and back navigation.

Expected: single-pass shows 3,155 judgments; multi-stage shows 7,008; dark inline code has explicit dark background; no raw/private artifact list appears.

- [ ] **Step 5: Verify HTTPS and tailnet-only exposure**

Require HTTP 200 for hub, architecture, Retrieval, and both RAG URLs. Inspect `tailscale serve status` for existing tailnet HTTPS and no Funnel/public listener. Scan for credentials, `/home/` paths, selected passage strings, raw-event fields, and adjacent JSON/JSONL links.

- [ ] **Step 6: Run final suite and cleanliness checks**

```bash
PYTHONPATH=code .venv/bin/python -m pytest -q
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
git diff --check
git status --short
```

Expected: full suite passes, tracked status is clean after commits, accepted hashes are unchanged, and only approved portal files changed outside git.

---

## Plan Completion Checks

- Root hub, reports index, architecture, and `AGENTS.md` use the same labels/destinations.
- Retrieval is directly reachable; neither RAG report is hidden behind a comparison page.
- Public pages mark RAG detail `Private / tailnet` and make no official-score claim.
- Actual toggled dark mode has WCAG AA inline-code contrast and unchanged source-code blocks.
- All five portal pages return HTTP 200 through tailnet-only Serve.
- Full tests, link tests, privacy scans, accepted validation, `git diff --check`, and final cleanliness pass.

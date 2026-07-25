# Promising TREC RAG 2025 Architecture HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and privately deliver a complete, accessible HTML reading copy of the source-backed 2025 architecture report.

**Architecture:** Keep `reports/2025-promising-rag-architecture.md` as the canonical research record and create a standalone HTML article beside it. The HTML uses semantic landmarks, native progressive disclosure, small optional JavaScript enhancements, and repository-relative local sources. A focused smoke test protects content, accessibility, safety, and link integrity; the existing report index provides discovery.

**Tech Stack:** Standalone HTML5, embedded CSS, vanilla JavaScript, Node.js smoke tests, Playwright through the repository headless-Chrome helper, Tailscale Serve for the already-authorized private portal.

## Global Constraints

- Preserve every substantive Markdown section, claim, caveat, score, comparison, source link, and verified/inference/unknown distinction.
- Keep `reports/2025-promising-rag-architecture.md` as the canonical source; do not delete or replace it.
- Use no external runtime dependencies, analytics, remote fonts, or third-party scripts.
- Do not publish through Codex Sites, Tailscale Funnel, a public listener, or any other public service.
- Keep all substantive content readable without JavaScript.
- Use semantic headings and landmarks, a skip link, visible keyboard focus, strong contrast, reduced-motion handling, descriptive image text, scoped table headers, and print styles.
- Keep the page mobile-friendly without horizontal page overflow.
- Expose only sanitized derived report assets through the existing tailnet-only portal.

---

### Task 1: Standalone Accessible HTML Report

**Files:**
- Create: `reports/2025-promising-rag-architecture.test.js`
- Create: `reports/2025-promising-rag-architecture.html`
- Read: `reports/2025-promising-rag-architecture.md`
- Read: `reports/trec-rag-2025-writeups/interactive-writeup.html`
- Read: `reports/trec-rag-2025-writeups/figures/2025-promising-composite-architecture.png`

**Interfaces:**
- Consumes: canonical Markdown headings, prose, tables, local PDF links, and the architecture PNG.
- Produces: `reports/2025-promising-rag-architecture.html`, a standalone page whose stable section IDs are `start`, `pipeline`, `definitions`, `team-evidence`, `recommended-build`, `confidence`, and `sources`.

- [ ] **Step 1: Write the failing HTML smoke test**

Create `reports/2025-promising-rag-architecture.test.js` with Node standard-library assertions:

```js
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "2025-promising-rag-architecture.html");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(htmlPath), "HTML report should exist");
const html = fs.readFileSync(htmlPath, "utf8");

for (const signal of [
  '<a class="skip-link" href="#main-content">',
  '<main id="main-content">',
  'id="start"',
  'id="pipeline"',
  'id="definitions"',
  'id="team-evidence"',
  'id="recommended-build"',
  'id="confidence"',
  'id="sources"',
  "UTokyo-HitU",
  "NC State LAS",
  "MITLL",
  "CFDA",
  "Tokyo University of Science",
  "WaterlooClarke",
  "GenAIus",
  "0.6934",
  "0.37",
  "0.65",
  "claims / nuggets + source IDs",
  'alt="Seven-stage recommended RAG architecture',
  'scope="col"',
  "@media print",
  "prefers-reduced-motion",
]) {
  assert(html.includes(signal), `Missing report signal: ${signal}`);
}

for (const forbidden of [
  "PYSERINI_API_TOKEN",
  "Authorization",
  "Bearer ",
  "https://cdn.",
  "fonts.googleapis.com",
  "Tailscale Funnel",
  "TODO",
  "placeholder",
]) {
  assert(!html.includes(forbidden), `Report should not expose ${forbidden}`);
}

const localTargets = [...html.matchAll(/(?:href|src)="([^"]+)"/g)]
  .map((match) => match[1])
  .filter((target) => !target.startsWith("#") && !target.startsWith("data:"));

for (const target of localTargets) {
  assert(
    !/^https?:/.test(target) || target.startsWith("https://trec.nist.gov/"),
    `Unexpected external runtime/source target: ${target}`,
  );
  if (!/^https?:/.test(target)) {
    assert(fs.existsSync(path.join(root, target)), `Missing local target: ${target}`);
  }
}

console.log("2025 architecture HTML smoke test passed");
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
node reports/2025-promising-rag-architecture.test.js
```

Expected: FAIL with `HTML report should exist`.

- [ ] **Step 3: Implement the complete standalone HTML**

Create `reports/2025-promising-rag-architecture.html` with:

- inline favicon, CSS, and optional lightbox JavaScript;
- `<header>`, `<nav aria-label="Report sections">`, `<main>`, `<article>`,
  `<section>`, `<aside>`, `<figure>`, `<table>`, and `<footer>` landmarks;
- a visible-by-keyboard skip link to `#main-content`;
- a hero that states the recommended composite and clearly labels it an
  inference, not a submitted 2025 run;
- the full architecture PNG at
  `trec-rag-2025-writeups/figures/2025-promising-composite-architecture.png`;
- a pipeline list that keeps the original narrative as the global contract;
- a comparison table with the columns `Role`, `Team`, `Verified evidence`, and
  `What to reuse`;
- native `<details>` team sections whose `<summary>` names the team and lesson;
- all Markdown sections and wording, including the distinction between
  organizer evaluation artifacts and participant runtime inputs;
- separate `.confidence-card.verified`, `.confidence-card.inference`, and
  `.confidence-card.unknown` blocks;
- repository-relative links to every cited local PDF;
- a no-JavaScript fallback where the only lost feature is image lightboxing;
- print CSS that opens details content with
  `details > * { display: block !important; }` and prints source URLs;
- responsive CSS at `760px` and `520px` breakpoints;
- focus-visible, reduced-motion, and high-contrast-safe styles.

Use this JavaScript interface only for the optional figure lightbox:

```js
const lightbox = document.getElementById("figure-lightbox");
const openFigure = document.getElementById("open-figure");
const closeFigure = document.getElementById("close-figure");

function setLightbox(open) {
  lightbox.hidden = !open;
  document.body.classList.toggle("lightbox-open", open);
  if (open) closeFigure.focus();
  else openFigure.focus();
}

openFigure.addEventListener("click", () => setLightbox(true));
closeFigure.addEventListener("click", () => setLightbox(false));
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) setLightbox(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) setLightbox(false);
});
```

- [ ] **Step 4: Run the focused smoke test**

Run:

```bash
node reports/2025-promising-rag-architecture.test.js
```

Expected: `2025 architecture HTML smoke test passed`.

- [ ] **Step 5: Verify complete Markdown-to-HTML coverage**

Run:

```bash
rg '^## |^### ' reports/2025-promising-rag-architecture.md
rg '<h2|<h3' reports/2025-promising-rag-architecture.html
```

Expected: every Markdown section has a corresponding visible HTML section or
subsection; team names and the Facts/Inference/Unknown sections all appear.

- [ ] **Step 6: Commit the standalone report**

```bash
git add reports/2025-promising-rag-architecture.md \
  reports/2025-promising-rag-architecture.html \
  reports/2025-promising-rag-architecture.test.js \
  reports/trec-rag-2025-writeups/figures/2025-promising-composite-architecture.png
git commit -m "add accessible 2025 architecture report"
```

### Task 2: Reports Index Integration

**Files:**
- Modify: `reports/index.html`
- Modify: `reports/index.test.js`
- Test: `reports/index.test.js`

**Interfaces:**
- Consumes: the stable report path
  `reports/2025-promising-rag-architecture.html`.
- Produces: one discovery card and one direct link from the shared reports
  index, protected by the existing index smoke test.

- [ ] **Step 1: Extend the failing index test**

Add these required signals to `requiredSignals` in `reports/index.test.js`:

```js
"Promising 2025 RAG Architecture",
"Narrative to verified answer",
"2025-promising-rag-architecture.html",
```

Add the target to `linkedFiles`:

```js
"2025-promising-rag-architecture.html",
```

- [ ] **Step 2: Run the index test to verify it fails**

Run:

```bash
node reports/index.test.js
```

Expected: FAIL with `Missing index signal: Promising 2025 RAG Architecture`.

- [ ] **Step 3: Add the report card**

Add a report card to `reports/index.html` using the existing `.report-card`
markup. The card must contain:

```html
<span class="label gold">architecture synthesis</span>
<h3>Promising 2025 RAG Architecture</h3>
<p>Narrative to verified answer: a source-backed synthesis of the strongest 2025 retrieval, coverage, evidence, and citation patterns.</p>
<a class="button primary" href="2025-promising-rag-architecture.html">Open architecture report</a>
```

Place it near the existing 2025 writeups card so the two reports read as a
pair.

- [ ] **Step 4: Run the index and report tests**

Run:

```bash
node reports/index.test.js
node reports/2025-promising-rag-architecture.test.js
```

Expected: both smoke tests print their pass messages.

- [ ] **Step 5: Commit the index integration**

```bash
git add reports/index.html reports/index.test.js
git commit -m "link 2025 architecture report"
```

### Task 3: Browser and Accessibility Verification

**Files:**
- Inspect: `reports/2025-promising-rag-architecture.html`
- Create temporarily and remove after inspection:
  `/tmp/trec-rag-2025-architecture-desktop.png`
  `/tmp/trec-rag-2025-architecture-mobile.png`

**Interfaces:**
- Consumes: the standalone HTML from Task 1.
- Produces: fresh desktop/mobile visual evidence and console/overflow checks;
  no new tracked artifact.

- [ ] **Step 1: Discover the headless browser interface**

Run:

```bash
.venv/bin/python code/tools/run_headless_chrome.py --help
```

Expected: usage text describing screenshot or browser invocation parameters.

- [ ] **Step 2: Render desktop and mobile views**

Use `code/tools/run_headless_chrome.py` or installed Playwright to load the
local HTML file at:

```text
/home/npatta01/.codex/worktrees/673c/trec_rag_2026/reports/2025-promising-rag-architecture.html
```

Capture:

- desktop: 1440×1000;
- mobile: 390×844.

Expected: the helper exits successfully and writes both screenshots.

- [ ] **Step 3: Inspect both screenshots**

Check both images for:

- no clipped navigation, headings, table cells, or architecture figure;
- no horizontal page overflow;
- readable type and adequate spacing;
- visible inference/verified/unknown distinction with text labels;
- a usable single-column mobile reading order.

If any check fails, adjust the HTML/CSS and repeat Steps 2–3.

- [ ] **Step 4: Verify interactive behavior**

With Playwright or equivalent browser automation:

```js
await page.click("#open-figure");
await expect(page.locator("#figure-lightbox")).toBeVisible();
await page.keyboard.press("Escape");
await expect(page.locator("#figure-lightbox")).toBeHidden();
await page.locator("details").first().focus();
await page.keyboard.press("Enter");
```

Expected: lightbox opens and closes with keyboard, and the first team details
element toggles from the keyboard without console errors.

- [ ] **Step 5: Run the complete local report test set**

Run:

```bash
node reports/2025-promising-rag-architecture.test.js
node reports/index.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
git diff --check
```

Expected: all tests pass and `git diff --check` prints no errors.

### Task 4: Private Tailnet Portal Delivery

**Files:**
- Create derived copy:
  `/home/npatta01/codex-rendered/plans/trec-rag-2025-promising-architecture.html`
- Create derived asset:
  `/home/npatta01/codex-rendered/plans/trec-rag-2025-promising-architecture.png`
- Modify:
  `/home/npatta01/codex-rendered/index.html`

**Interfaces:**
- Consumes: verified standalone report and PNG from Tasks 1–3.
- Produces: a sanitized HTML/PNG pair reachable through the existing
  tailnet-only Tailscale Serve portal.

- [ ] **Step 1: Create a portal-safe derived HTML copy**

Copy the verified report and architecture PNG under the explicit paths above.
In the derived HTML only:

- change the image path to `trec-rag-2025-promising-architecture.png`;
- retain descriptive source citations;
- replace repository-relative PDF links with their official
  `https://trec.nist.gov/pubs/trec34/papers/...` URLs from
  `reports/trec-rag-2025-writeups/manifest.tsv`;
- keep the Reports Home link pointed at `../index.html`;
- do not copy PDFs, source Markdown, raw logs, datasets, or repository trees.

- [ ] **Step 2: Add the portal index entry**

Add a concise entry in `/home/npatta01/codex-rendered/index.html` linking to:

```text
plans/trec-rag-2025-promising-architecture.html
```

Use the visible title `Promising TREC RAG 2025 Architecture`.

- [ ] **Step 3: Scan the derived artifacts**

Run:

```bash
rg -n "PYSERINI_API_TOKEN|Authorization|Bearer |OPENAI_API_KEY|TODO|placeholder" \
  /home/npatta01/codex-rendered/plans/trec-rag-2025-promising-architecture.html \
  /home/npatta01/codex-rendered/index.html
```

Expected: no matches.

Confirm the derived HTML references only its PNG, `../index.html`, fragment
links, and official NIST HTTPS PDF URLs.

- [ ] **Step 4: Verify the private URL and Serve policy**

Run:

```bash
curl -fsSI \
  https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2025-promising-architecture.html
tailscale serve status
```

Expected:

- HTTPS returns `200`;
- the active mapping serves `/home/npatta01/codex-rendered`;
- no Funnel/public exposure is enabled.

- [ ] **Step 5: Final repository and artifact verification**

Run:

```bash
node reports/2025-promising-rag-architecture.test.js
node reports/index.test.js
git status --short
git log -3 --oneline
```

Expected: the focused tests pass; only intentional files are changed or
committed; recent commits describe the report and index work.

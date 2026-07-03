# Report Index And Briefing Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a common reports index and restyle the 2026 briefing so it matches the richer 2025 interactive report experience.

**Architecture:** Keep the repo as static HTML reports under `reports/`. Add one shared entry page at `reports/index.html`, restyle `reports/trec-rag-briefing-report.html` with the 2025 report's visual system, and add smoke tests that assert required sections, links, interactions, and source references are present.

**Tech Stack:** Static HTML, inline CSS/JS, Node.js smoke tests using `fs` and string assertions, Python HTTP server for manual serving.

---

### Task 1: Common Reports Index

**Files:**
- Create: `reports/index.html`
- Create: `reports/index.test.js`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Create `reports/index.test.js` with assertions that `reports/index.html` exists, links to both reports, uses the 2025-style topbar/hero/card vocabulary, and does not expose local secrets.

- [ ] **Step 2: Run the test to verify it fails**

Run: `node reports/index.test.js`

Expected: fails because `reports/index.html` does not exist yet.

- [ ] **Step 3: Implement the index**

Create a static `reports/index.html` with shared top navigation, a large hero, report cards for the 2026 briefing and 2025 writeups, and a small "how to read" route grid.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node reports/index.test.js`

Expected: `reports/index smoke test passed`

### Task 2: Restyle The 2026 Briefing

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`
- Create: `reports/trec-rag-briefing-report.test.js`

- [ ] **Step 1: Write the failing test**

Create `reports/trec-rag-briefing-report.test.js` with assertions for the new 2025-style shell: `topbar`, `hero`, `shell`, route cards, glossary chips, pipeline stage buttons, ClimbMix sample section, implementation kit, failure diagnosis, source references, and no leaked token text.

- [ ] **Step 2: Run the test to verify it fails**

Run: `node reports/trec-rag-briefing-report.test.js`

Expected: fails because the current briefing still uses the old sidebar layout.

- [ ] **Step 3: Implement the restyle**

Rewrite the briefing as a richer static guide using the 2025 report palette and component vocabulary: topbar, hero, route cards, plain-English section, interactive glossary, ClimbMix document sample, 2025-to-2026 bridge, pipeline stage selector, implementation kit, evaluation/failure diagnosis, and appendix/source details.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node reports/trec-rag-briefing-report.test.js`

Expected: `trec-rag-briefing-report smoke test passed`

### Task 3: End-To-End Verification

**Files:**
- Verify: `reports/index.html`
- Verify: `reports/trec-rag-briefing-report.html`
- Verify: `reports/trec-rag-2025-writeups/interactive-writeup.html`

- [ ] **Step 1: Run all smoke tests**

Run:
`node reports/index.test.js && node reports/trec-rag-briefing-report.test.js && node reports/trec-rag-2025-writeups/interactive-writeup.test.js`

Expected: all three tests pass.

- [ ] **Step 2: Verify served pages**

Run a local HTTP request against the existing server for:
- `http://127.0.0.1:8801/reports/index.html`
- `http://127.0.0.1:8801/reports/trec-rag-briefing-report.html`

Expected: both return HTML containing the new title/hero text.

- [ ] **Step 3: Inspect git status**

Run: `git status -sb`

Expected: tracked changes are the report/index/test/README files only, with existing untracked local data untouched.

### Self-Review

Spec coverage: the plan includes a common index, 2025-style briefing restyle, README update, smoke tests, served-page checks, and no-token verification.

Placeholder scan: no unresolved placeholders or deferred tasks remain.

Type consistency: all paths and test commands match the repo's static-report layout.

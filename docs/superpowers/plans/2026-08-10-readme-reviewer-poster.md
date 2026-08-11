# README Reviewer Poster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the top of the GitHub README into a self-contained reviewer poster with the problem, whole-system architecture, verified counts, and four explicit report links.

**Architecture:** Reuse the checked-in whole-system SVG and standard GitHub-flavored Markdown. Extend the existing Node contract test before changing README content, then run the surrounding hub and architecture smoke tests.

**Tech Stack:** Markdown, SVG, Node.js contract tests, GitHub Pages links.

## Global Constraints

- Reuse `reports/2026-competition-architecture/01-whole-system.svg`; do not create a second architecture.
- Preserve the artifact hub as the primary top-level action.
- Support GitHub desktop, mobile, light mode, dark mode, and image-free reading.
- Keep all claims aligned with the submission ledger and architecture report.
- Do not expose private caches, raw evaluation data, credentials, or unpublished artifacts.

---

### Task 1: Reviewer-facing README poster

**Files:**
- Modify: `README.test.js`
- Modify: `README.md`

**Interfaces:**
- Consumes: the existing whole-system SVG and the four published report paths.
- Produces: a GitHub-renderable README whose reviewer-facing content is protected by `README.test.js`.

- [ ] **Step 1: Extend the failing README contract**

Add required assertions for `## The problem`, the linked whole-system image,
`## Project at a glance`, all four result counts, and these explicit labels:
`Architecture Report`, `Retrieval Quality Report`,
`RAGDoll Evaluation — Single-pass RAG`, and
`RAGDoll Evaluation — Multi-stage RAG`.

- [ ] **Step 2: Run the contract and verify failure**

Run: `node README.test.js`

Expected: FAIL because the existing README does not contain `## The problem`.

- [ ] **Step 3: Implement the poster content**

Insert the problem statement, linked whole-system SVG, verified-count table,
and explicit report list above repository references. Keep the artifact-hub link
above all detailed content and retain developer notes below it.

- [ ] **Step 4: Run focused verification**

Run:

```bash
node README.test.js
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Commit the implementation**

```bash
git add README.md README.test.js
git commit -m "docs: add reviewer poster to README"
```

- [ ] **Step 6: Publish and verify**

Push `codex/readme-reviewer-poster`, open a PR against `master`, merge it, then
verify the raw public README contains the four explicit report labels and the
GitHub repository and Pages hub both return HTTP 200.

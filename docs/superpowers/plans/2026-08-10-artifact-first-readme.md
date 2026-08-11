# Artifact-first README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public repository lead newcomers directly to the NP Labs TREC RAG 2026 artifact hub.

**Architecture:** `README.md` becomes a reviewer-first index with one dominant GitHub Pages link and compact repository references. GitHub repository metadata uses the same identity and canonical homepage so the About panel and README agree.

**Tech Stack:** Markdown, Node.js link assertions, GitHub CLI.

## Global Constraints

- Project identity is exactly `NP Labs submission for TREC RAG 2026`.
- Canonical homepage is exactly `https://npatta01.github.io/trec_rag_2026/`.
- Preserve developer setup, testing, and agent workflow below the reviewer-facing material.
- Do not duplicate the full artifact-hub navigation or enumerate five submissions in the opening.

---

### Task 1: Reviewer-first README

**Files:**
- Modify: `README.md`
- Create: `README.test.js`

**Interfaces:**
- Consumes: tracked artifact, ledger, architecture, skill, and implementation paths.
- Produces: the repository landing copy and an executable link/identity contract.

- [ ] **Step 1: Write the failing README contract**

Create `README.test.js` with this contract:

```js
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const source = fs.readFileSync(path.join(root, "README.md"), "utf8");
const hubLabel = "View the project artifact hub";
const hubUrl = "https://npatta01.github.io/trec_rag_2026/";
const required = [
  "# NP Labs · TREC RAG 2026",
  "NP Labs submission for TREC RAG 2026",
  `[${hubLabel}](${hubUrl})`,
  "submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "reports/2026-competition-architecture.qmd",
  ".agents/skills/validate-trec-rag-2026-submissions/SKILL.md",
  "code/trec_rag/README.md",
  "## Developer notes",
];
for (const value of required) {
  if (!source.includes(value)) throw new Error(`README missing: ${value}`);
}
if (source.indexOf(hubLabel) > source.indexOf("SUBMISSION_LEDGER.md")) {
  throw new Error("primary hub link must precede repository detail links");
}
for (const match of source.matchAll(/\[[^\]]+\]\((?!https?:|#)([^)]+)\)/g)) {
  if (!fs.existsSync(path.join(root, match[1]))) {
    throw new Error(`README target missing: ${match[1]}`);
  }
}
console.log("README artifact-first contract passed");
```

- [ ] **Step 2: Run the contract to verify it fails**

Run: `node README.test.js`

Expected: FAIL because the current README lacks the approved title and primary public hub link.

- [ ] **Step 3: Rewrite the README hierarchy**

Replace the opening with:

```markdown
# NP Labs · TREC RAG 2026

This repository contains the NP Labs submission for TREC RAG 2026.

## [View the project artifact hub](https://npatta01.github.io/trec_rag_2026/)
```

Follow it with a short hub overview, compact direct repository links, and a lower `Developer notes` section containing the existing setup and test guidance.

- [ ] **Step 4: Run the README contract**

Run: `node README.test.js`

Expected: PASS with all local targets present and the exact hub URL bound to the primary label.

- [ ] **Step 5: Commit**

```bash
git add README.md README.test.js
git commit -m "docs: make repository landing artifact-first"
```

### Task 2: GitHub repository identity

**Files:**
- Modify externally: GitHub repository description and homepage for `npatta01/trec_rag_2026`

**Interfaces:**
- Consumes: the identity and homepage constants from the approved spec.
- Produces: a consistent GitHub About panel pointing to the artifact hub.

- [ ] **Step 1: Update repository metadata**

Run:

```bash
gh repo edit npatta01/trec_rag_2026 \
  --description "NP Labs submission for TREC RAG 2026." \
  --homepage "https://npatta01.github.io/trec_rag_2026/"
```

- [ ] **Step 2: Verify repository metadata and live homepage**

Run:

```bash
gh repo view npatta01/trec_rag_2026 --json description,homepageUrl
curl -L --fail --silent --show-error --output /dev/null \
  https://npatta01.github.io/trec_rag_2026/
```

Expected: exact approved description and homepage, followed by HTTP success.

- [ ] **Step 3: Publish README branch**

Push `codex/readme-artifact-first`, open a pull request against `master`, merge it after checks, and verify the rendered GitHub README and Pages hub both return successfully.

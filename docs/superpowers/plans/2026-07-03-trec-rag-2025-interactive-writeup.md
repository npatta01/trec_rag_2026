# TREC RAG 2025 Interactive Writeup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable beginner-friendly but technically detailed interactive HTML writeup for official TREC RAG 2025 approaches.

**Architecture:** One static HTML file contains semantic content, embedded CSS, embedded team data, and lightweight vanilla JavaScript. One Node smoke test validates the expected document structure, interaction hooks, team count, tags, and local PDF links.

**Tech Stack:** HTML, CSS, vanilla JavaScript, Node built-ins for smoke testing.

---

### Task 1: Smoke Test

**Files:**
- Create: `reports/trec-rag-2025-writeups/interactive-writeup.test.js`

- [ ] **Step 1: Write the failing test**

Create a Node script that reads `interactive-writeup.html`, verifies required sections, counts team entries, checks approach tags, verifies interaction functions, and confirms local PDF links exist.

- [ ] **Step 2: Run the test and verify it fails**

Run: `node reports/trec-rag-2025-writeups/interactive-writeup.test.js`

Expected: failure because `interactive-writeup.html` does not exist yet.

### Task 2: Static Interactive Page

**Files:**
- Create: `reports/trec-rag-2025-writeups/interactive-writeup.html`
- Modify: `reports/trec-rag-2025-writeups/README.md`

- [ ] **Step 1: Build the HTML page**

Create the static page with beginner definitions, the TREC task setup, an interactive pipeline, approach filters, a cross-team synthesis section, searchable deep team dossiers, source links, and practical takeaways.

- Each team card must be marked `data-depth="deep-dossier"`.
- Each team dossier must include: Architecture, What they tried, Models and tools, Results and observations, Caveats, and What we can reuse.
- Results should distinguish official metrics from author-side ablations and qualitative observations.

- [ ] **Step 2: Link it from the README**

Add a short README entry pointing to `interactive-writeup.html`.

- [ ] **Step 3: Run the smoke test**

Run: `node reports/trec-rag-2025-writeups/interactive-writeup.test.js`

Expected: pass.

### Task 3: Browser Verification

**Files:**
- Verify: `reports/trec-rag-2025-writeups/interactive-writeup.html`

- [ ] **Step 1: Open with a local static server**

Run: `python3 -m http.server 8765 --directory reports/trec-rag-2025-writeups`

Expected: page available at `http://127.0.0.1:8765/interactive-writeup.html`.

- [ ] **Step 2: Verify rendering and interactions**

Use browser automation or text-level checks to confirm no console-breaking syntax errors and that filters/search/expansion/pipeline/glossary handlers are present.

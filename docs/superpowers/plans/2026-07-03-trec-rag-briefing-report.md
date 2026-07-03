# TREC RAG Briefing Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained interactive HTML briefing that explains TREC RAG 2026, the ClimbMix collection, what happened in TREC RAG 2025, and beginner-friendly retrieval/RAG techniques.

**Architecture:** Create one static HTML file at the repository root with embedded CSS and JavaScript. The page is structured as a scrollable report with slide-like sections, sticky navigation, expandable technique cards, worked-example tabs, and an appendix. Verification uses local browser loading and lightweight DOM checks rather than a build pipeline.

**Tech Stack:** HTML5, CSS3, vanilla JavaScript, local browser, optional Python HTTP server for inspection.

---

## File Structure

- Create: `reports/trec-rag-briefing-report.html`
  - Responsible for all report content, layout, interactivity, sources, and appendix sections.
- No framework, package manager, build artifacts, or external assets.
- No changes to the checked-in TREC data repositories.

## Source Inputs

Use these sources while writing the content:

- Local 2026 retrieval task guide: `trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/retrieval-task.md`
- Local 2026 RAG task guide: `trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/rag-task.md`
- Local 2026 development-data guide: `trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/development-data.md`
- Local 2026 qrels README: `trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/README.md`
- Official 2025 track page: `https://trec-rag.github.io/trec25/`
- TREC RAG 2025 overview paper: `https://arxiv.org/abs/2603.09891`
- NIST 2025 retrieval appendix: `https://trec.nist.gov/pubs/trec34/appendices/trec2025-rag-retrieval.html`
- TREC 2025 proceedings browser: `https://pages.nist.gov/trec-browser/trec34/rag/proceedings/`
- TREC 2025 runs browser: `https://pages.nist.gov/trec-browser/trec34/rag/runs/`
- UTokyo 2025 paper: `https://trec.nist.gov/pubs/trec34/papers/UTokyo.rag.pdf`

---

### Task 1: Reconfirm Source Facts Before Writing

**Files:**
- Read: source files and URLs listed above
- No file changes

- [ ] **Step 1: Read local 2026 source files**

Run:

```bash
sed -n '1,240p' trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/retrieval-task.md
sed -n '1,260p' trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/rag-task.md
sed -n '1,260p' trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/development-data.md
sed -n '1,220p' trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/README.md
```

Expected: commands show the official retrieval format, RAG JSONL format, development-data descriptions, and qrels grading explanation.

- [ ] **Step 2: Reconfirm 2025 facts from downloaded or web sources**

Use existing cached text when available:

```bash
rg -n "MS MARCO|105|22|Retrieval|Augmented Generation|Relevance Judgment|4method_merge|Kun-Third|nDCG|HyDE|Reciprocal" tmp/trec2025-rag-overview.txt tmp/trec2025_papers/UTokyo.rag.txt tmp/trec2025-runs.html tmp/trec2025-proceedings.html
```

Expected: output confirms the 2025 corpus, tasks, assessed narratives, top retrieval run, and technique descriptions.

- [ ] **Step 3: Write down the exact caveats to preserve**

Use these caveats in the report:

```text
UTokyo's 4method_merge is documented as the top 2025 retrieval run in the NIST retrieval appendix and participant paper.
The top RAG answer run should be described only as a top-scoring run from overview tables unless a detailed participant method writeup is found.
Development qrels are judged pools, not exhaustive truth over the whole collection.
Local chunks can be used internally, but submitted references should map back to official document IDs.
```

Expected: these caveats appear in the final source/caveat appendix and are reflected in the 2025 technique section.

---

### Task 2: Create Static Report Shell

**Files:**
- Create: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Create the initial HTML shell**

Use `apply_patch` to add `reports/trec-rag-briefing-report.html` with this structure:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TREC RAG 2026 Briefing</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1b2430;
      --muted: #5e6a78;
      --line: #d7dde5;
      --accent: #256f7b;
      --accent-2: #8b5e34;
      --retrieval: #2d6cdf;
      --generation: #b54c7a;
      --eval: #5c7c2f;
      --code: #111827;
      --code-bg: #eef2f6;
      --shadow: 0 10px 30px rgba(24, 33, 46, 0.08);
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }

    a { color: var(--accent); }

    .layout {
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 24px 18px;
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      overflow-y: auto;
    }

    .brand {
      margin-bottom: 24px;
    }

    .brand h1 {
      margin: 0 0 8px;
      font-size: 1.25rem;
      line-height: 1.2;
    }

    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .progress {
      height: 8px;
      margin: 18px 0 24px;
      border-radius: 999px;
      background: #e5e9ef;
      overflow: hidden;
    }

    .progress span {
      display: block;
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--generation));
      transition: width 120ms linear;
    }

    .nav {
      display: grid;
      gap: 6px;
    }

    .nav a {
      display: block;
      padding: 9px 10px;
      border-radius: 7px;
      color: var(--ink);
      text-decoration: none;
      font-size: 0.94rem;
    }

    .nav a.active,
    .nav a:hover {
      background: #e9f3f5;
      color: #174f58;
    }

    main {
      min-width: 0;
      padding: 32px min(5vw, 64px) 64px;
    }

    section {
      max-width: 1120px;
      margin: 0 auto 34px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .section-kicker {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h2 {
      margin: 0 0 12px;
      font-size: clamp(1.75rem, 2.5vw, 2.6rem);
      line-height: 1.08;
      letter-spacing: 0;
    }

    h3 {
      margin: 24px 0 10px;
      font-size: 1.18rem;
      line-height: 1.25;
    }

    p { margin: 0 0 14px; }

    .lede {
      max-width: 780px;
      color: #334155;
      font-size: 1.1rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    .card,
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 16px;
    }

    .card h3,
    details h3 {
      margin-top: 0;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #edf3f7;
      color: #365063;
      font-size: 0.78rem;
      font-weight: 700;
    }

    .table-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 680px;
      background: #fff;
    }

    th,
    td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #f0f4f8;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    code,
    pre {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    code {
      padding: 2px 5px;
      border-radius: 5px;
      background: var(--code-bg);
    }

    pre {
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 8px;
      background: var(--code);
      color: #f8fafc;
      overflow-x: auto;
      white-space: pre-wrap;
    }

    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 18px 0;
    }

    .tabs button,
    .stepper button {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 9px 12px;
    }

    .tabs button.active,
    .stepper button.active {
      border-color: var(--accent);
      background: #e6f3f5;
      color: #174f58;
    }

    .tab-panel[hidden],
    .example-step[hidden] {
      display: none;
    }

    summary {
      cursor: pointer;
      font-weight: 800;
    }

    .source-list {
      display: grid;
      gap: 8px;
      padding-left: 18px;
    }

    @media (max-width: 860px) {
      .layout { display: block; }
      .sidebar {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .nav {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      main {
        padding: 22px 14px 44px;
      }
      section {
        padding: 20px;
      }
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar" aria-label="Report navigation">
      <div class="brand">
        <h1>TREC RAG Briefing</h1>
        <p>A newcomer-friendly guide to the 2026 track, ClimbMix, and lessons from 2025.</p>
      </div>
      <div class="progress" aria-label="Reading progress"><span id="progress-bar"></span></div>
      <nav class="nav" id="page-nav">
        <a href="#orientation">Orientation</a>
        <a href="#climbmix">ClimbMix</a>
        <a href="#year-2025">2025 Context</a>
        <a href="#techniques">Technique Gallery</a>
        <a href="#changes">2025 to 2026</a>
        <a href="#strategy">Solving 2026</a>
        <a href="#evaluation">Evaluation</a>
        <a href="#appendix">Appendix</a>
      </nav>
    </aside>
    <main>
      <section id="orientation">
        <p class="section-kicker">Orientation</p>
        <h2>What Is This Competition About?</h2>
        <p class="lede">TREC RAG asks a system to do two hard things together: find useful evidence for a complex information need, then write a cited answer that is actually supported by that evidence.</p>
      </section>
      <section id="climbmix">
        <p class="section-kicker">2026 Corpus</p>
        <h2>What Is ClimbMix?</h2>
      </section>
      <section id="year-2025">
        <p class="section-kicker">Historical Context</p>
        <h2>What Happened In 2025?</h2>
      </section>
      <section id="techniques">
        <p class="section-kicker">Technique Gallery</p>
        <h2>Approaches People Used In 2025</h2>
      </section>
      <section id="changes">
        <p class="section-kicker">Comparison</p>
        <h2>What Changed From 2025 To 2026?</h2>
      </section>
      <section id="strategy">
        <p class="section-kicker">System Design</p>
        <h2>How I Would Approach 2026</h2>
      </section>
      <section id="evaluation">
        <p class="section-kicker">Evaluation</p>
        <h2>How To Think About Grading</h2>
      </section>
      <section id="appendix">
        <p class="section-kicker">Reference</p>
        <h2>Appendix</h2>
      </section>
    </main>
  </div>
  <script>
    const navLinks = [...document.querySelectorAll('#page-nav a')];
    const sections = navLinks.map(link => document.querySelector(link.getAttribute('href')));
    const progressBar = document.querySelector('#progress-bar');

    function updateProgress() {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const pct = scrollable <= 0 ? 0 : (window.scrollY / scrollable) * 100;
      progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;

      let active = sections[0];
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= 160) active = section;
      }
      navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${active.id}`);
      });
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  </script>
</body>
</html>
```

- [ ] **Step 2: Open the file locally**

Run:

```bash
python3 -m http.server 8765
```

Expected: server starts and `http://localhost:8765/reports/trec-rag-briefing-report.html` displays the shell without console errors. Stop the server after inspection unless continuing with browser testing.

- [ ] **Step 3: Commit the shell**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Add TREC RAG briefing report shell"
```

Expected: commit succeeds with only `reports/trec-rag-briefing-report.html` staged.

---

### Task 3: Add Core 2026 And 2025 Explanatory Content

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Fill the orientation and ClimbMix sections**

Replace the minimal section bodies for `orientation` and `climbmix` with reader-facing content that includes this information:

```html
<div class="grid">
  <article class="card">
    <span class="tag">Retrieval task</span>
    <h3>Return ranked document IDs</h3>
    <p>The retrieval task asks for a standard TREC-style run: one ranked list of document IDs per topic. It measures whether your search stage can surface evidence that helps answer the narrative.</p>
    <pre>topic_id Q0 docid rank score run_id</pre>
  </article>
  <article class="card">
    <span class="tag">RAG task</span>
    <h3>Write a cited answer</h3>
    <p>The RAG task asks for prose, not raw chunks. Each substantive answer sentence should point to the source documents that support it.</p>
  </article>
  <article class="card">
    <span class="tag">Mental model</span>
    <h3>Evidence first, answer second</h3>
    <p>A strong system has to cover the user's facets, choose evidence, and avoid claims that the retrieved documents do not support.</p>
  </article>
</div>
```

Also include these ClimbMix points in ordinary prose:

```text
ClimbMix is a broad curated web/pretraining-style collection rather than a single-domain corpus.
The official unit is a row/document with a ClimbMix document ID.
Local chunking is useful for ranking and answer writing, but final retrieval rows and citations must map back to official document IDs.
```

- [ ] **Step 2: Fill the 2025 context section**

Add a compact historical explanation with these exact factual claims:

```text
TREC RAG 2025 used MS MARCO V2.1 segmented documents.
The 2025 track included Retrieval, Augmented Generation, Retrieval-Augmented Generation, and Relevance Judgment tasks.
The track used long narrative information needs, not only short keyword queries.
The overview paper reports 105 narratives, with official assessed analysis centered on 22 selected narratives.
```

Include a caveat block:

```html
<div class="card">
  <span class="tag">Careful wording</span>
  <h3>Retrieval winner is not the same thing as full RAG winner</h3>
  <p>UTokyo's <code>4method_merge</code> is well documented as a top 2025 retrieval run. Full RAG answer rankings used different metrics, so the report should avoid treating one retrieval result as the whole competition winner.</p>
</div>
```

- [ ] **Step 3: Add the 2025 vs 2026 comparison table**

Add this table to `changes`:

```html
<div class="table-wrap">
  <table>
    <thead>
      <tr><th>Dimension</th><th>TREC RAG 2025</th><th>TREC RAG 2026</th></tr>
    </thead>
    <tbody>
      <tr><td>Corpus</td><td>MS MARCO V2.1 segmented documents</td><td>ClimbMix document rows</td></tr>
      <tr><td>Tasks</td><td>Retrieval, AG, RAG, and RJ</td><td>Retrieval and RAG</td></tr>
      <tr><td>Generation setup</td><td>AG used organizer-provided evidence; RAG allowed end-to-end systems</td><td>RAG systems own retrieval and generation</td></tr>
      <tr><td>Evidence IDs</td><td>MS MARCO segment IDs</td><td>ClimbMix document IDs</td></tr>
      <tr><td>Main lesson</td><td>Long narratives reward coverage across facets</td><td>The same lesson applies, now over a broader web-style corpus</td></tr>
    </tbody>
  </table>
</div>
```

- [ ] **Step 4: Validate factual language manually**

Run:

```bash
rg -n "MS MARCO V2.1|105 narratives|22 selected|4method_merge|ClimbMix|Retrieval and RAG" reports/trec-rag-briefing-report.html
```

Expected: each phrase appears in the relevant section, with caveated wording around rankings.

- [ ] **Step 5: Commit the content sections**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Add TREC RAG briefing context"
```

Expected: commit succeeds.

---

### Task 4: Add Beginner Technique Gallery

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Add technique card markup**

Inside `#techniques`, add a `div` with `id="technique-grid"`:

```html
<p class="lede">The 2025 systems are most useful as a menu of techniques. You do not need to copy one run exactly; you need to understand what each tool is good for.</p>
<div class="grid technique-grid" id="technique-grid"></div>
```

- [ ] **Step 2: Add technique data in JavaScript**

Before `updateProgress()`, add this array:

```javascript
const techniques = [
  {
    name: 'BM25 / lexical retrieval',
    group: 'Retrieval',
    plain: 'Search for documents that share important words with the query.',
    helps: 'It is fast, simple, and often strong when the narrative contains distinctive terms.',
    fails: 'It can miss documents that use different wording for the same concept.',
    example: 'Query: "microplastic exposure in seafood". BM25 will love documents that literally say microplastic, exposure, and seafood.'
  },
  {
    name: 'Query expansion',
    group: 'Retrieval',
    plain: 'Add related words or phrases before searching.',
    helps: 'It catches vocabulary mismatch, such as "heart attack" and "myocardial infarction".',
    fails: 'Bad expansions drift away from the user need and retrieve broad background documents.'
  },
  {
    name: 'Query decomposition',
    group: 'Retrieval',
    plain: 'Split one long narrative into several smaller searches.',
    helps: 'It improves coverage when the topic asks for causes, effects, dates, comparisons, and policy details at once.',
    fails: 'Each sub-query can become too narrow or pull in off-topic documents if the decomposition is sloppy.'
  },
  {
    name: 'Dense retrieval',
    group: 'Retrieval',
    plain: 'Use embeddings to search by semantic similarity rather than exact words.',
    helps: 'It can find paraphrases and conceptually related evidence.',
    fails: 'It may retrieve documents that feel topically similar but do not answer the precise facet.'
  },
  {
    name: 'Hybrid sparse plus dense retrieval',
    group: 'Retrieval',
    plain: 'Run lexical and embedding search, then combine their candidates.',
    helps: 'It balances exact terminology with semantic matching. UTokyo used this kind of combination in its strong 2025 retrieval system.',
    fails: 'It needs deduping and fusion; otherwise one retriever can flood the candidate pool.'
  },
  {
    name: 'SPLADE-style learned sparse retrieval',
    group: 'Retrieval',
    plain: 'Use a model to expand sparse term weights while keeping an inverted-index style search.',
    helps: 'It behaves partly like lexical retrieval and partly like semantic expansion.',
    fails: 'It adds model complexity and still needs careful fusion with other candidate sources.'
  },
  {
    name: 'Reciprocal Rank Fusion',
    group: 'Fusion',
    plain: 'Merge ranked lists by rewarding documents that appear near the top of multiple lists.',
    helps: 'It is simple and robust when combining BM25, dense search, SPLADE, and facet searches.',
    fails: 'It cannot rescue a candidate that no retriever found.'
  },
  {
    name: 'HyDE',
    group: 'Retrieval',
    plain: 'Generate a hypothetical answer, embed that answer, and search with the embedding.',
    helps: 'It can turn an abstract query into answer-shaped language that matches relevant documents.',
    fails: 'If the hypothetical answer hallucinates, it can pull retrieval toward unsupported assumptions.'
  },
  {
    name: 'LLM reranking',
    group: 'Reranking',
    plain: 'Ask a stronger model to reorder retrieved candidates by usefulness for the full narrative.',
    helps: 'It can judge whether a document answers the actual multi-part need, not just whether it matches terms.',
    fails: 'It is slower, costlier, and can over-prefer fluent but weakly grounded snippets.'
  },
  {
    name: 'Evidence selection',
    group: 'Generation',
    plain: 'Choose the few chunks that should actually support the answer.',
    helps: 'It prevents the generator from seeing a messy pile of semi-related search results.',
    fails: 'If selection misses a facet, the final answer will look polished but incomplete.'
  },
  {
    name: 'Citation-first generation',
    group: 'Generation',
    plain: 'Plan claims around source evidence before writing the final prose.',
    helps: 'It keeps the answer grounded and makes unsupported claims easier to catch.',
    fails: 'It can produce stiff prose if the system only stitches citations together without synthesis.'
  },
  {
    name: 'Post-generation checks',
    group: 'Quality',
    plain: 'Review whether each sentence has support and whether the answer covers the important facets.',
    helps: 'It catches uncited claims, missing facets, and citation indices that point to the wrong source.',
    fails: 'A weak checker may rubber-stamp the same mistakes the generator made.'
  }
];
```

- [ ] **Step 3: Render technique cards**

Add this rendering code after the `techniques` array:

```javascript
const techniqueGrid = document.querySelector('#technique-grid');
if (techniqueGrid) {
  techniqueGrid.innerHTML = techniques.map((item, index) => `
    <article class="card technique-card">
      <span class="tag">${item.group}</span>
      <h3>${index + 1}. ${item.name}</h3>
      <p><strong>Plain English:</strong> ${item.plain}</p>
      <details>
        <summary>When to use it</summary>
        <p>${item.helps}</p>
      </details>
      <details>
        <summary>Watch out for</summary>
        <p>${item.fails}</p>
      </details>
      ${item.example ? `<pre>${item.example}</pre>` : ''}
    </article>
  `).join('');
}
```

- [ ] **Step 4: Add CSS for technique cards**

Add:

```css
.technique-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.technique-card {
  display: grid;
  align-content: start;
  gap: 10px;
}

.technique-card details {
  padding: 10px 12px;
  background: #f8fafc;
}

.technique-card p {
  margin-bottom: 0;
}

@media (max-width: 860px) {
  .technique-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 5: Verify card rendering**

Run:

```bash
node -e "const fs=require('fs'); const html=fs.readFileSync('reports/trec-rag-briefing-report.html','utf8'); console.log((html.match(/name: '/g)||[]).length)"
```

Expected: prints `12`.

- [ ] **Step 6: Commit the technique gallery**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Add beginner TREC RAG technique gallery"
```

Expected: commit succeeds.

---

### Task 5: Add Worked Examples For Solving 2026

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Add strategy overview cards**

Inside `#strategy`, add:

```html
<p class="lede">For 2026, I would not rely on one raw BM25 query for every topic. I would keep direct search as the baseline, then add decomposition, fusion, reranking, evidence selection, and citation checks.</p>
<div class="grid">
  <article class="card">
    <span class="tag">Step 1</span>
    <h3>Search broadly</h3>
    <p>Run the title, the full narrative, and several facet queries. Keep enough candidates to avoid losing rare but important evidence.</p>
  </article>
  <article class="card">
    <span class="tag">Step 2</span>
    <h3>Fuse and rerank</h3>
    <p>Use RRF to merge candidate lists, then rerank against the full narrative so documents that cover more facets move up.</p>
  </article>
  <article class="card">
    <span class="tag">Step 3</span>
    <h3>Write from evidence</h3>
    <p>Select chunks, map them back to ClimbMix document IDs, and generate only claims that can be cited.</p>
  </article>
</div>
```

- [ ] **Step 2: Add tab markup for examples**

Add:

```html
<div class="tabs" role="tablist" aria-label="Worked examples">
  <button class="active" data-tab="direct" type="button">Direct BM25</button>
  <button data-tab="facets" type="button">Facet Search</button>
  <button data-tab="noise" type="button">Noisy Decomposition</button>
  <button data-tab="citations" type="button">Cited Answer</button>
</div>
<div class="tab-panel" data-panel="direct">
  <h3>Example 1: Search the narrative directly</h3>
  <p>A direct query is a good first baseline because it is reproducible and reveals obvious matches.</p>
  <pre>query = title + " " + narrative
search(climbmix_bm25, query, top_k=100)</pre>
</div>
<div class="tab-panel" data-panel="facets" hidden>
  <h3>Example 2: Split the need into facets</h3>
  <p>If the narrative asks for causes, impacts, and policy responses, run those as separate searches and fuse the results.</p>
  <pre>facet_queries = [
  "causes of the issue",
  "measured impacts and affected populations",
  "policy responses and mitigation"
]
candidates = rrf(search_each(facet_queries))</pre>
</div>
<div class="tab-panel" data-panel="noise" hidden>
  <h3>Example 3: Decomposition can add noise</h3>
  <p>A sub-query like "policy" by itself is too vague. Keep facet queries anchored to the original topic terms.</p>
  <pre>bad = "policy responses"
better = "policy responses to [specific topic terms from the narrative]"</pre>
</div>
<div class="tab-panel" data-panel="citations" hidden>
  <h3>Example 4: Turn evidence into cited prose</h3>
  <p>Each sentence should be traceable to one or more retrieved documents. Unsupported claims should be removed or searched for again.</p>
  <pre>{
  "answer": [
    {
      "text": "The evidence indicates that the issue has multiple causes rather than one dominant driver.",
      "citations": [0, 2]
    }
  ],
  "references": ["shard_01789_3390", "shard_02004_0007", "shard_04501_2210"]
}</pre>
</div>
```

- [ ] **Step 3: Add tab interaction JavaScript**

Add this script before `updateProgress()`:

```javascript
const tabButtons = [...document.querySelectorAll('[data-tab]')];
const tabPanels = [...document.querySelectorAll('[data-panel]')];

tabButtons.forEach(button => {
  button.addEventListener('click', () => {
    const selected = button.dataset.tab;
    tabButtons.forEach(item => item.classList.toggle('active', item === button));
    tabPanels.forEach(panel => {
      panel.hidden = panel.dataset.panel !== selected;
    });
  });
});
```

- [ ] **Step 4: Verify tabs exist**

Run:

```bash
node -e "const fs=require('fs'); const html=fs.readFileSync('reports/trec-rag-briefing-report.html','utf8'); console.log((html.match(/data-tab=/g)||[]).length, (html.match(/data-panel=/g)||[]).length)"
```

Expected: prints `4 4`.

- [ ] **Step 5: Commit worked examples**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Add worked TREC RAG strategy examples"
```

Expected: commit succeeds.

---

### Task 6: Add Evaluation, Appendix, And Sources

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Fill evaluation section**

Add this content to `#evaluation`:

```html
<div class="grid">
  <article class="card">
    <span class="tag">Retrieval</span>
    <h3>Relevance is facet-aware</h3>
    <p>A document can be more valuable when it helps answer multiple parts of a long narrative. That is why breadth matters as much as top-rank precision.</p>
  </article>
  <article class="card">
    <span class="tag">RAG</span>
    <h3>Coverage plus support</h3>
    <p>A good answer should cover the important pieces of the information need and cite documents that support each substantive claim.</p>
  </article>
  <article class="card">
    <span class="tag">Practical</span>
    <h3>Uncited claims are liabilities</h3>
    <p>If a sentence is only connective prose, it may not need a citation. If it states a factual answer, either cite it or remove it.</p>
  </article>
</div>
```

- [ ] **Step 2: Fill appendix with collapsible references**

Add:

```html
<details open>
  <summary>2026 submission formats</summary>
  <p>Retrieval uses the six-column TREC run format:</p>
  <pre>topic_id Q0 docid rank score run_id</pre>
  <p>RAG uses JSONL with metadata, references, and answer sentences that cite reference indices.</p>
</details>
<details>
  <summary>Development data mental model</summary>
  <p>The local 2026 development data includes RAG25-style topics and qrels, plus ResearchRubrics prompts and rubrics. Treat qrels as judged pools for development, not as exhaustive labels over all of ClimbMix.</p>
</details>
<details>
  <summary>Source caveats</summary>
  <p>System rankings and metrics are tied to specific tasks and metrics. A strong retrieval run should not be described as the best full answer-generation system unless the source says that.</p>
</details>
<details>
  <summary>Sources</summary>
  <ul class="source-list">
    <li><a href="https://trec-rag.github.io/trec25/">Official TREC RAG 2025 page</a></li>
    <li><a href="https://arxiv.org/abs/2603.09891">TREC RAG 2025 overview paper</a></li>
    <li><a href="https://trec.nist.gov/pubs/trec34/appendices/trec2025-rag-retrieval.html">NIST 2025 retrieval appendix</a></li>
    <li><a href="https://trec.nist.gov/pubs/trec34/papers/UTokyo.rag.pdf">UTokyo 2025 TREC RAG paper</a></li>
    <li><a href="https://pages.nist.gov/trec-browser/trec34/rag/proceedings/">TREC 2025 proceedings browser</a></li>
    <li><a href="https://pages.nist.gov/trec-browser/trec34/rag/runs/">TREC 2025 runs browser</a></li>
  </ul>
</details>
```

- [ ] **Step 3: Add local-source note**

Add a paragraph in the appendix:

```html
<p>The 2026 task and development-data descriptions in this report are grounded in the local official guideline files under <code>trec-rag-skills/skills/trec-rag-2026-track-guidelines/references/</code> and the checked-out development-data README files under <code>trec-rag-data/trec-rag-2026/development-data/</code>.</p>
```

- [ ] **Step 4: Verify source links**

Run:

```bash
node -e "const fs=require('fs'); const html=fs.readFileSync('reports/trec-rag-briefing-report.html','utf8'); const links=[...html.matchAll(/href=\"(https?:\/\/[^\"]+)\"/g)].map(m=>m[1]); console.log(links.length); console.log(links.join('\n'))"
```

Expected: prints at least `6` source links, including the 2025 page, overview paper, NIST appendix, UTokyo paper, proceedings browser, and runs browser.

- [ ] **Step 5: Commit evaluation and appendix**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Add TREC RAG evaluation appendix and sources"
```

Expected: commit succeeds.

---

### Task 7: Polish Layout And Accessibility

**Files:**
- Modify: `reports/trec-rag-briefing-report.html`

- [ ] **Step 1: Add utility styling for callouts and examples**

Add:

```css
.callout {
  margin: 18px 0;
  padding: 14px 16px;
  border-left: 4px solid var(--accent);
  border-radius: 6px;
  background: #eef8fa;
}

.split {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.muted {
  color: var(--muted);
}

@media (max-width: 700px) {
  .split {
    grid-template-columns: 1fr;
  }
  .nav {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 2: Add accessibility labels to interactive controls**

Ensure tab buttons have `type="button"` and the tab container has `role="tablist"` with an `aria-label`. Keep native `<details>` for expandable cards because they are keyboard accessible by default.

- [ ] **Step 3: Check for empty headings or sections**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
html = Path('reports/trec-rag-briefing-report.html').read_text()
for needle in ['<section id="orientation">', '<section id="climbmix">', '<section id="year-2025">', '<section id="techniques">', '<section id="strategy">', '<section id="appendix">']:
    print(needle, html.count(needle))
print('empty h2 tags:', html.count('<h2></h2>'))
print('placeholder tokens:', sum(html.lower().count(x) for x in ['todo', 'tbd', 'placeholder']))
PY
```

Expected:

```text
<section id="orientation"> 1
<section id="climbmix"> 1
<section id="year-2025"> 1
<section id="techniques"> 1
<section id="strategy"> 1
<section id="appendix"> 1
empty h2 tags: 0
placeholder tokens: 0
```

- [ ] **Step 4: Commit polish**

Run:

```bash
git add reports/trec-rag-briefing-report.html
git commit -m "Polish TREC RAG briefing layout"
```

Expected: commit succeeds.

---

### Task 8: Final Verification

**Files:**
- Read: `reports/trec-rag-briefing-report.html`
- No required file changes unless verification finds issues

- [ ] **Step 1: Validate expected content counts**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
html = Path('reports/trec-rag-briefing-report.html').read_text()
checks = {
    'sections': html.count('<section id='),
    'technique names': html.count("name: '"),
    'tabs': html.count('data-tab='),
    'panels': html.count('data-panel='),
    'source links': html.count('href="https://'),
    'details': html.count('<details'),
}
for key, value in checks.items():
    print(f'{key}: {value}')
assert checks['sections'] == 8
assert checks['technique names'] == 12
assert checks['tabs'] == 4
assert checks['panels'] == 4
assert checks['source links'] >= 6
assert checks['details'] >= 20
PY
```

Expected: command exits `0` and prints the counts.

- [ ] **Step 2: Start a local server for browser inspection**

Run:

```bash
python3 -m http.server 8765
```

Expected: server starts. Open `http://localhost:8765/reports/trec-rag-briefing-report.html`.

- [ ] **Step 3: Browser-check core interactions**

In the browser:

```text
Scroll the page and confirm the progress bar moves.
Click each nav item and confirm it jumps to the right section.
Open and close multiple technique-card details.
Click all four worked-example tabs and confirm only the selected panel is visible.
Narrow the browser below 860px and confirm cards stack without text overflow.
```

Expected: all interactions work, text remains readable, and no section overlaps another.

- [ ] **Step 4: Stop the server**

Stop the Python server with `Ctrl-C`.

Expected: terminal returns to the shell prompt.

- [ ] **Step 5: Final git status**

Run:

```bash
git status --short
```

Expected: only intended files are modified or untracked. Pre-existing untracked files such as `.env`, `notes.md`, `tmp/`, `trec-rag-data/`, and `trec-rag-skills/` may remain untouched.


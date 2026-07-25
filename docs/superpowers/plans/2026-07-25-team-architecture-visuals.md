# TREC RAG 2025 Team Architecture Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add consistent, readable six-stage architecture visuals for all 15 team dossiers in the existing TREC RAG 2025 interactive writeup.

**Architecture:** Keep the report's existing team data and prose as the source-backed content. Add one standardized SVG renderer and one 15-entry visual data map inside the existing standalone HTML; each team card receives the same six named stages and a short text equivalent. Extend the existing smoke test and perform browser checks at desktop, mobile, and print sizes.

**Tech Stack:** Standalone HTML5, inline SVG, embedded CSS, vanilla JavaScript, Node.js smoke tests, available headless Chrome/Playwright.

## Global Constraints

- Use the same six stage names and left-to-right order for every team: Narrative, Plan, Search, Evidence, Answer, Check.
- Use deterministic inline SVG; do not reuse paper figures or generated raster text in the team visuals.
- Preserve existing team prose, source links, scores, caveats, and the organizer/runtime distinction.
- Render all 15 team visuals from one renderer and one data map.
- Keep stage colors paired with text labels; color is never the only encoding.
- Provide accessible SVG titles/descriptions and a visible text equivalent for every team.
- Keep the visual readable at 390px, printable, and free of horizontal overflow.
- Do not introduce external runtime dependencies or public hosting.

---

### Task 1: Standardized Team Visual Renderer and Coverage Test

**Files:**
- Modify: `reports/trec-rag-2025-writeups/interactive-writeup.html`
- Modify: `reports/trec-rag-2025-writeups/interactive-writeup.test.js`
- Read: `reports/trec-rag-2025-writeups/README.md`
- Read: `reports/trec-rag-2025-writeups/interactive-writeup.html` team dossiers and existing architecture data

**Interfaces:**
- Consumes: the 15 existing `.team-card` elements and their team names.
- Produces: one `team-architecture-visual` SVG/text equivalent per team, rendered by `renderArchitecturePanels()` from a `teamVisuals` map keyed by exact team name.

- [x] **Step 1: Extend the smoke test with failing visual contracts**

Add these assertions to `reports/trec-rag-2025-writeups/interactive-writeup.test.js`:

```js
const visualStages = ["Narrative", "Plan", "Search", "Evidence", "Answer", "Check"];
for (const stage of visualStages) {
  assert(html.includes(`data-visual-stage="${stage}"`), `Missing visual stage: ${stage}`);
}
assert(html.includes("teamVisuals"), "Missing standardized visual data map");
assert(html.includes("team-architecture-visual"), "Missing team visual renderer output");
assert(html.includes("team-architecture-text"), "Missing text equivalent for team visual");
assert(html.includes('role="img"'), "Team SVGs should have an accessible image role");
assert(html.includes("All 15 teams use the same six stages"), "Missing visual legend");

const requiredVisualTeams = [
  "CFDA Lab", "NIT Agartala", "University of Glasgow Terrier", "WING-II",
  "GRILL Lab", "IIUoT", "GenAIus", "Tokyo University of Science", "HLTCOE",
  "MIT Lincoln Laboratory", "IDACCS", "NC State LAS", "UTokyo-HitU", "DUTH",
  "WaterlooClarke",
];
for (const team of requiredVisualTeams) {
  assert(html.includes(`"${team}":`), `Missing visual data for ${team}`);
}
```

- [x] **Step 2: Run the focused test to verify it fails**

Run:

```bash
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
```

Expected: FAIL because the six-stage visual data/renderer signals are not yet present.

- [x] **Step 3: Add the shared visual data map**

Add a `teamVisuals` object next to `architecturePanels`. Each entry must have:

```js
{
  summary: "short plain-English flow sentence",
  distinctive: "one concise distinctive move",
  stages: [
    { stage: "Narrative", label: "...", detail: "..." },
    { stage: "Plan", label: "...", detail: "..." },
    { stage: "Search", label: "...", detail: "..." },
    { stage: "Evidence", label: "...", detail: "..." },
    { stage: "Answer", label: "...", detail: "..." },
    { stage: "Check", label: "...", detail: "..." },
  ],
}
```

Use these exact team-specific stage labels, keeping the fixed stage names:

```text
CFDA Lab: Long narrative | Subqueries | BM25 + embeddings | RRF + rerank | Subanswers → synthesis | Sentence support
NIT Agartala: Long narrative | Direct query | BM25 + DPR | Hybrid + cross-encoder | Falcon cited answer | RAG metrics
University of Glasgow Terrier: Long narrative | Explicit subqueries | Sparse + E5 | MonoT5 rerank | Merge subanswers | Coverage comparison
WING-II: Fixed evidence | No runtime plan | Greedy submodular | Evidence cards | Citation-first claims | Refiner + support
GRILL Lab: Long narrative | Subquestions | BM25 + expand | Gap queries + RRF | Agentic synthesis | Saturation stop
IIUoT: Long narrative | Query rewrite | BM25 | RankZephyr + MMR | Reference-first claims | Support filter
GenAIus: Fixed passages | No retrieval plan | Top-20 input | Atomic nuggets | Nugget-ID claims | RJ signals
Tokyo University of Science: Long narrative | Viewpoints | Sparse + dense | RRF + keystones | Partial / final answer | Coverage vs redundancy
HLTCOE: Retrieved documents | Nugget ideation | BM25 + PLAID-X | Merge + filter | Nugget-grounded report | Citation + coverage
MIT Lincoln Laboratory: Long narrative | Minimal subquestions | SPLADEv3 | Rerank + RRF + SETR | GPT-5 citations | Ablation metrics
IDACCS: Long / multilingual | No decomposition | PLAID-X | mxbai + occams | GPT-4.1 rewrite | Blame attribution
NC State LAS: Long narrative | Agent / subquestions | BM25 + SPLADE | Nugget feedback | Cited agent answer | Vital / sub coverage
UTokyo-HitU: Long narrative | Keyword + HyDE | BM25 + SPLADE + dense | 4-way RRF + LLM | Ragnarok citations | Retrieval + RAG scores
DUTH: Query-passage pairs | No answer plan | BM25 + ColBERT | Qwen / StableLM labels | TREC qrels | Human calibration
WaterlooClarke: Long narrative | Portfolio plan | BM25 + T5 | Claims + support | GARE / nuggetizer | Pairwise select
```

- [x] **Step 4: Add the shared SVG renderer and panel output**

Implement `renderTeamArchitectureVisual(teamName, visual)` beside the existing
architecture rendering helpers. It should:

- return one inline SVG with `viewBox="0 0 1200 246"`, `role="img"`, and unique
  escaped `<title>`/`<desc>` identifiers;
- draw the six stages left-to-right with arrows, fixed stage names, and the
  team-specific labels from `teamVisuals`;
- include the visible legend text: `All 15 teams use the same six stages:
  Narrative → Plan → Search → Evidence → Answer → Check.`;
- include a visible `.team-architecture-text` equivalent that states the flow
  in plain language, plus a concise `.team-architecture-distinctive` note;
- escape team, stage, and label strings before inserting them into HTML/SVG.

Update `renderArchitecturePanels()` to render this visual for every team. Keep
the existing evidence notes and decoder, but do not render paper figures inside
the team architecture panel.

- [x] **Step 5: Add responsive and print styling**

Add styles for `.team-architecture-visual`, `.team-architecture-svg`,
`.team-architecture-text`, `.team-architecture-distinctive`, and
`.team-architecture-legend`. Make the panel span the team-card width on open
cards, keep the SVG readable at 390px without horizontal page overflow, and
provide print/reduced-motion rules. Do not use color as the only stage signal.

- [x] **Step 6: Run focused verification**

Run the focused smoke test, inspect the diff for duplicated or inconsistent
stage names, and use the repository's headless Chrome helper at 1440×1000 and
390×844. Open each of the 15 Details buttons and assert that exactly one
`.team-architecture-visual` appears in the opened card, all six stage data
attributes are present, `scrollWidth === clientWidth`, and the browser reports
no console errors. Also verify the print DOM retains the legend and text
equivalent.

- [ ] **Step 7: Commit the implementation**

Commit only the report and smoke-test changes with a concise user-facing
message after all checks pass.

### Task 2: Browser and Accessibility Review

After Task 1 is complete, review the rendered report at desktop, mobile, and
print sizes. Confirm that the six-stage sequence is visually obvious without
reading the paper, each card uses the same vocabulary, labels remain legible,
keyboard disclosure still works, and the text equivalent is available to screen
readers. Fix any issues found, rerun the full report test suite, and record the
verification evidence in the implementation handoff.

**Verification evidence:** Node smoke tests passed for the interactive writeup,
the promising-architecture report, and the reports index. Chrome CDP checks
opened all 15 team cards and found one visual with all six stage attributes per
card, no horizontal page overflow at 390px, no console warnings/errors, and
print media retained the legend and text equivalent. A desktop and mobile
screenshot review caught and fixed nested SVG label overlap and mobile grid
min-content overflow.

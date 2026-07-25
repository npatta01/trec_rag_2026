# Promising TREC RAG 2025 Architecture HTML Report Design

## Objective

Convert `reports/2025-promising-rag-architecture.md` into an accessible,
self-contained HTML report without removing or weakening any source-backed
content. The Markdown remains the canonical research record. The HTML is a
derived reading experience for desktop, mobile, screen-reader, print, and
private remote viewing.

## Scope

The report will:

- preserve every substantive Markdown section, claim, caveat, score,
  comparison, source link, and verified/inference/unknown distinction;
- embed the generated composite architecture image and make it usable without
  right-clicking;
- add the report to `reports/index.html`;
- provide a sanitized derived copy through the existing tailnet-only rendered
  artifact portal;
- add focused automated checks for structure, safety, links, and responsive
  rendering.

The report will not:

- replace or delete the canonical Markdown;
- introduce external runtime dependencies, analytics, remote fonts, or
  third-party scripts;
- publish through Codex Sites, Tailscale Funnel, a public listener, or any
  other public service;
- change the report's research conclusions or invent a universal 2025 winner.

## Reading Experience

Use a full-content interactive article, visually consistent with the existing
TREC RAG reports but smaller and more focused.

### Page structure

1. A skip link and semantic page landmarks.
2. A sticky top bar with Reports Home and section navigation.
3. A hero containing:
   - the report title;
   - the one-sentence architecture conclusion;
   - a clear notice that the recommendation is a cross-team inference;
   - quick links to the pipeline and team evidence.
4. A prominent architecture figure with descriptive caption, full-size link,
   and in-page lightbox.
5. A plain-English pipeline section explaining narrative, coverage units,
   subqueries, nuggets, claims, citations, and repair.
6. A “who contributed what” comparison that separates:
   - best verified retrieval;
   - best documented full-RAG manual result;
   - clearest inspectable pipeline;
   - clearest claim-controlled generation;
   - clearest nugget-to-claim interface.
7. Full team evidence sections preserving all Markdown content.
8. A recommended composite architecture section.
9. Separate verified, inferred, and unknown callouts.
10. A primary-source section linking to every local PDF used by the Markdown.

### Progressive disclosure

All report content remains present in the HTML document. Longer team evidence
sections may use native `<details>` elements to improve scanning. They must be
keyboard accessible, work without JavaScript, and expand automatically for
printing. Essential conclusions, definitions, and caveats remain visible by
default.

## Visual Design

Reuse the repository's established visual language:

- warm off-white background and white reading surfaces;
- teal for navigation and verified evidence;
- blue for retrieval;
- amber for evidence organization;
- coral for caveats and verification;
- restrained borders, shadows, and corner radii;
- system fonts only.

The layout uses a centered readable measure for prose and wider containers only
for the architecture figure and comparison table. It collapses to a single
column on mobile without horizontal page scrolling.

## Accessibility

The page will include:

- semantic heading order and landmarks;
- a keyboard-visible skip link;
- descriptive link text;
- descriptive image alternative text and a visible caption;
- keyboard-operable navigation, details elements, and lightbox;
- visible focus styles;
- sufficient foreground/background contrast;
- reduced-motion handling;
- no interaction whose meaning depends on color alone;
- tables with captions and scoped headers;
- print styles that expose all content and source URLs;
- a no-JavaScript reading path for all substantive content.

## Data and Source Flow

`reports/2025-promising-rag-architecture.md` remains the source of truth.
The initial HTML is a checked-in standalone artifact under `reports/`. Updates
to research content should be made in Markdown first and then mirrored into the
HTML. Relative PDF links will resolve to
`reports/trec-rag-2025-writeups/pdfs/`, and the architecture image will resolve
to `reports/trec-rag-2025-writeups/figures/`.

The remote viewing copy will contain only:

- the derived HTML;
- the generated architecture PNG;
- any explicitly required local source PDFs only if the HTML links depend on
  them and the existing portal mapping can expose them narrowly.

The preferred remote copy avoids duplicating PDFs and instead keeps source
links inside the repository report when remote file serving cannot be
restricted narrowly.

## Failure Handling

- If an image fails to load, the figure caption and alternative text still
  explain the complete pipeline.
- If JavaScript is disabled, all report prose, details content, citations, and
  source links remain readable; only the optional lightbox is unavailable.
- If the private portal cannot safely expose the source PDFs, the portal copy
  will retain descriptive source citations without exposing a broader
  repository tree.
- Broken local links fail the smoke test.

## Verification

Add a focused Node smoke test that checks:

- required sections and complete source-team coverage;
- semantic landmarks, skip link, figure alternative text, table captions, and
  source links;
- absence of secrets, unfinished placeholder markers, external runtime
  dependencies, and public-hosting directives;
- existence of every repository-relative linked image and PDF.

Run Playwright or the repository's headless Chrome helper at desktop and mobile
viewports to check:

- no horizontal overflow;
- readable navigation and content;
- keyboard focus and details behavior;
- figure lightbox behavior;
- print-mode content visibility where supported.

Before handoff:

- run the smoke test and `git diff --check`;
- inspect desktop and mobile screenshots;
- verify the private HTTPS artifact URL returns the intended report;
- confirm the Tailscale Serve mapping remains tailnet-only;
- scan the rendered copy for secrets, raw datasets, logs, and unrelated files.

## Deliverables

- `reports/2025-promising-rag-architecture.html`
- a focused HTML smoke test beside the report
- an updated `reports/index.html` and index smoke test
- the existing canonical Markdown and architecture PNG
- a sanitized derived copy under
  `/home/npatta01/codex-rendered/plans/`
- an updated `/home/npatta01/codex-rendered/index.html`

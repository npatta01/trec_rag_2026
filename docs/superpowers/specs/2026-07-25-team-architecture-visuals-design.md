# TREC RAG 2025 Team Architecture Visuals Design

## Objective

Make the existing 15-team TREC RAG 2025 interactive writeup understandable
without reading the team papers or long prose blocks. Every team dossier will
receive a comparable, source-backed architecture visual.

## Visual contract

Every team uses the same six stage names and left-to-right order:

1. **Narrative** — what entered the system;
2. **Plan** — decomposition, rewrite, portfolio, or fixed-input boundary;
3. **Search** — the retrieval or agent-search mechanism;
4. **Evidence** — reranking, fusion, nuggets, cards, or selection;
5. **Answer** — the generated output and citation form;
6. **Check** — coverage, support, calibration, or evaluation loop.

Each stage has a short team-specific label of two to four words. The stage
names, colors, arrows, and legend never change. The visual is followed by one
plain-English flow sentence and a short “distinctive move” note. Long evidence,
scores, caveats, and source links remain below in the existing dossier prose.

## Implementation

Use deterministic inline SVG rendered from a single JavaScript data map inside
`reports/trec-rag-2025-writeups/interactive-writeup.html`. Do not reuse paper
figures or generated raster text in the team panels. SVG labels must be
searchable, exact, and escaped before insertion. The same renderer must produce
all 15 diagrams, and every team in the existing team grid must have a visual
record.

The visual data is descriptive, not a new ranking claim. Every team-specific
label must be traceable to the existing dossier text and local paper source.
The visual must not imply that a fixed-retrieval AG system performed runtime
retrieval or that organizer-side nuggets were hidden participant inputs.

## Accessibility and responsive behavior

- Each SVG has a unique accessible title/description and `role="img"`.
- The same flow appears as a visible text equivalent below the SVG.
- Stage labels remain readable at 390px without horizontal scrolling.
- Stage colors are paired with stage names and are not the only encoding.
- The visual remains visible with JavaScript enabled; the existing no-JavaScript
  team prose remains intact.
- Print CSS keeps the six-stage flow and text equivalent together.
- Reduced-motion settings disable decorative transitions.

## Verification

Extend the existing interactive-writeup smoke test to require:

- the shared six stage names and legend;
- all 15 team names in the visual data map;
- 15 rendered visual containers and 15 accessible SVG titles;
- text equivalents containing the standardized stage names;
- no external image/runtime dependency introduced by the visual renderer.

Use the available headless browser to inspect 1440px and 390px screenshots,
check `scrollWidth === clientWidth`, confirm every team visual is present when
its card is opened, and verify the SVG text remains readable in print mode.

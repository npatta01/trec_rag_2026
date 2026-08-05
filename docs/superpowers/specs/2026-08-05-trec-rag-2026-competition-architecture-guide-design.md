# TREC RAG 2026 Competition Architecture Guide Design

## Status

Conversational visual design approved. Quarto authoring feedback is now
incorporated; implementation planning remains behind the written-spec review
gate.

The work starts from merge commit
`7e9d3b9159f5222a761ba58ad515b236dc93a423` on a new clean branch and worktree.
No prior cache, output, environment, report artifact, or brainstorming mockup is
an implementation input.

## Objective

Create a polished, image-led Quarto walkthrough, rendered as standalone HTML,
that helps a first-time reader follow one official narrative through the
supported TREC RAG 2026 competition workflow:

1. bounded retrieval planning;
2. document retrieval and passage ranking;
3. local evidence selection and Nuggetizer-assisted canonicalization;
4. the authenticated selected-evidence handoff;
5. GPT-5.6 Sol answer generation;
6. validation, citation conversion, and organizer submission.

The same guide must also orient coding agents and contributors without turning
the friend-facing explanation into a dense implementation reference.

## Audience

The primary audience is a technically curious friend who has not worked on
information retrieval or RAG systems. The secondary audience is a contributor
or coding agent looking for the supported competition path and its non-negotiable
boundaries.

Every unfamiliar term receives a short first-principles definition. The guide
uses progressive disclosure: the essential story and limits are visible, while
exact identifiers, failure behavior, configuration fields, and source links
live in expandable technical notes.

## Scope

### Included

- The supported two-command 2026 competition path implemented by
  `trec_rag.competition_retrieval` and `trec_rag.competition_rag`.
- The canonical checked-in retrieval configuration
  `configs/rag26_competition_retrieval_v2.yaml`.
- The canonical GPT-5.6 Sol generation configuration
  `configs/rag26_competition_rag_gpt_sol_v2.yaml`.
- The exact boundary represented by `generation_handoff_manifest.json`.
- The organizer-facing retrieval artifacts and RAG submission, including which
  stage creates and consumes each artifact.
- Mechanically verified behavior at commit `7e9d3b9`.

### Excluded

- Legacy baselines, 2025 pipelines, DeepAgent prototypes, organizer Pi tracing,
  RAGDoll evaluation, development-only adapters, and historical experiments.
- Claims that the model quality, retrieval benefit, or claim entailment has been
  established. Existing verification proves mechanics and contracts, not model
  quality.
- Live retrieval, generation, evaluation, or hosted calls.
- Real test-narrative text, raw corpus text, caches, generated answers, provider
  responses, qrels, gold nuggets, or private run artifacts.
- Public deployment or changes to sharing permissions.

## Deliverables

### Main report

Create:

- `reports/2026-competition-architecture.qmd`
- `reports/2026-competition-architecture.html`

The `.qmd` file is the canonical authored source. The `.html` file is a
checked-in generated artifact for friend-facing reading and must never be
edited by hand. The report is a static vertical story rendered by Quarto with
embedded resources and no external runtime dependency. Prefer Quarto Markdown,
figures, captions, cross-references, table of contents, and built-in lightbox
behavior over hand-authored page structure. Small semantic HTML fragments are
allowed only where Quarto has no equivalent, such as a native disclosure with a
required no-JavaScript fallback.

Do not add a repository-wide `_quarto.yml`: no tracked Quarto project exists at
the fixed base, and a global configuration could alter unrelated reports. Keep
all format settings in this report's YAML front matter. A small local stylesheet
may be used for the report's visual system and must be embedded into the
generated HTML. Avoid custom JavaScript unless a verified accessibility need
cannot be met by Quarto's built-in behavior.

### Reusable visual sources

Create these local, individually viewable SVG files:

1. `reports/2026-competition-architecture/01-whole-system.svg`
2. `reports/2026-competition-architecture/02-retrieval-system.svg`
3. `reports/2026-competition-architecture/03-bounded-deepseek-planning.svg`
4. `reports/2026-competition-architecture/04-per-query-candidate-accounting.svg`
5. `reports/2026-competition-architecture/05-evidence-and-nuggetizer.svg`
6. `reports/2026-competition-architecture/06-generation-handoff-contract.svg`
7. `reports/2026-competition-architecture/07-sol-generation.svg`
8. `reports/2026-competition-architecture/08-validation-and-retries.svg`
9. `reports/2026-competition-architecture/09-organizer-output-split.svg`

Also create:

- `reports/2026-competition-architecture/README.md`

The asset README maps every diagram to its purpose, factual source files, and
required text equivalent. Each SVG carries its own theme-aware CSS, accessible
title, description, and visible labels so it remains meaningful when opened
outside the report.

### Verification and discovery

Create:

- `reports/2026-competition-architecture.test.js`

Modify:

- `AGENTS.md`
- `README.md`
- `reports/index.html`
- `reports/index.test.js`

`AGENTS.md` receives a concise Architecture Orientation section rather than a
copy of the report. The root README and report index make the walkthrough easy
for friends and contributors to discover.

## Quarto Authoring and Render Contract

Use the installed Quarto CLI to render the one report directly; do not turn the
repository into a Quarto project. The canonical command, run from the repository
root, is:

```bash
quarto render reports/2026-competition-architecture.qmd --to html
```

The `.qmd` front matter owns the complete rendering contract. It must:

- render HTML beside the source file;
- embed CSS, fonts, and Quarto runtime resources in the HTML;
- provide light and dark color schemes;
- enable a generated table of contents and figure enlargement;
- preserve local direct links to the nine reusable SVG sources;
- produce no network request at viewing time.

Author the narrative, headings, glossary, captions, figure references, source
notes, and most layout in Quarto Markdown. Keep styling in a small local style
source if needed. Use raw HTML sparingly and only for semantics that cannot be
expressed faithfully in Quarto Markdown. Start with no custom JavaScript; add a
local script only if browser verification proves that a required interaction or
accessibility behavior cannot be achieved with Quarto's generated behavior.

The generated HTML is committed so readers do not need Quarto installed. Each
render handoff records the Quarto version used; the initial implementation is
verified with the host's Quarto `1.9.38`. A clean re-render must leave the
checked-in HTML unchanged.

## Reading Experience

The report opens with three short elements:

1. a one-sentence statement of the problem;
2. the fictional running narrative;
3. a compact glossary for narrative, query lane, document, passage, evidence,
   canonical hint, handoff, and citation.

The page then follows the nine diagrams in order. Each section contains:

- one visible takeaway sentence;
- one diagram;
- one short plain-language explanation;
- a visible text equivalent of the diagram's flow;
- a native `<details>` disclosure for exact implementation notes and sources;
- Quarto's figure-enlargement control and a direct link to its SVG.

A compact section navigator and persistent legend help readers retain context.
The navigator becomes an ordinary wrapping link list on narrow screens and in
print.

## Running Example

Use this clearly labeled fictional narrative:

> How can cities reduce extreme heat? Compare tree canopy, cool roofs, and
> cooling centers, including effectiveness, equity, cost, and maintenance.

The example demonstrates why one narrative benefits from multiple focused
queries. Captions may show illustrative query wording, passage summaries, or a
cited-answer outline, but must never imply that those illustrative records came
from ClimbMix or an actual competition run.

The example threads through captions and selected callouts. It does not appear
inside every architecture node, which would overload the diagrams.

## Visual Method

### Editorial character

Use an editorial/scientific visual style: approachable color, ample whitespace,
short labels, and small explanatory callouts. Under that styling, every shape
has a stable technical meaning.

The guide preserves the user's desired funnel methodology as a story:

> expand one narrative into focused searches, gather many candidates, then
> narrow toward selected evidence and one validated answer.

It does not use a literal proportional funnel or Sankey. The real pipeline
changes units from documents to chunks to passages, and chunking can expand the
item count. A proportional narrowing shape would therefore make a false
quantitative claim.

### Shape grammar

- **Rounded rectangle:** an executable process or transformation.
- **Stacked document cards:** input or output artifacts such as narratives,
  documents, passage sets, manifests, and submissions.
- **Pill-shaped model/package node:** a named model or hosted package boundary,
  including DeepSeek, Mixedbread, Nuggetizer, and GPT-5.6 Sol.
- **Manifest document with checksum/seal band:** the authenticated
  `generation_handoff_manifest.json` contract. Do not use a padlock, which could
  imply encryption.
- **Tinted container:** a system or execution boundary such as Retrieval,
  Generation, hosted service, remote Pyserini, local GPU, or local code.
- **Diamond:** an actual pass/fail validation decision only.
- **Solid arrow:** data movement.
- **Dashed arrow:** retry or fallback control flow only.
- **Equal-sized numeric callout:** a ceiling, budget, or attempt limit. Area is
  never proportional to a count.

The whole visual vocabulary appears in one compact legend and is reused
literally across all nine SVGs.

### Color grammar

- Blue: planning and retrieval.
- Teal: passage ranking and evidence selection.
- Gold: authenticated handoff.
- Violet: answer generation.
- Orange: retry or fallback.
- Green: validated output.

Color reinforces labels and shapes but never carries meaning alone. Light and
dark themes use different surfaces, foregrounds, structural strokes, and
shadows while preserving stable stage identity. Grayscale must retain execution
order, boundaries, decisions, and status through labels and line styles.

## Figure Contracts

### 1. Whole-system overview

Show five landmarks:

```text
official narrative
  → Retrieval system
  → authenticated selected-evidence manifest
  → Generation system
  → validated organizer RAG JSONL
```

Retrieval's TREC run and full-text ZIP branch toward the organizer outside the
Generation boundary. No arrow may let Generation read those artifacts.

The figure's job is orientation, not implementation detail. It contains no
model revisions, retry counts, or cache mechanics.

### 2. Retrieval system

Show the supported fixed stage order:

```text
bounded planning
  → Pyserini ClimbMix search
  → chunking and Mixedbread scoring
  → local exact-span evidence selection
  → Nuggetizer-assisted canonicalization
  → sealed publication
```

Use small textual boundary tags for hosted planning, remote Pyserini, local GPU,
local code, and hosted canonicalization. Include small orientation badges for
`≤1,000 documents / focused query` and `≤100 passages / focused query`; the next
figure explains the unit transition.

Label the planning portion **Bounded agentic planning**, not unqualified
**Agentic retrieval**. The latter could be confused with an open-ended tool loop
or the separate `deepagent_retrieval.py` prototype.

### 3. Bounded DeepSeek planning

Start with one narrative document artifact. It creates:

- the guaranteed untouched-original query lane;
- a one-shot, schema-validated DeepSeek plan with at most eight focused
  subnarratives.

Show planning failure as a dashed control-flow branch that retains only the
untouched original lane and produces no downstream evidence or canonical call.
Do not show an iterative search/planning agent loop.

Use **DeepSeek** inside the figure. Put the pinned model identifier and prompt
contract in the expandable note.

### 4. Per-focused-query candidate accounting

Show equal-sized stages with explicit units and ceilings:

```text
≤1,000 ClimbMix documents / focused query
  → variable number of chunks
       (≤3,500 characters, 350-character overlap)
  → Mixedbread scoring
  → ≤100 passages / focused query
```

Label the bridge **unit change and possible expansion**. Label both endpoint
values **configured ceilings, not observed counts**. Do not imply a fixed 10:1
filtering ratio.

Use **Pyserini**, **ClimbMix**, and **Mixedbread** inside the figure. Put the
exact index, model revision, and cache identities in the expandable note.

### 5. Evidence selection and Nuggetizer

Keep factual evidence authority on the main solid-arrow trunk:

```text
exact source spans
  → local scoring within each subnarrative
  → deduplication and diversity clustering
  → ≤40 admitted evidence spans / subnarrative
```

Then show Nuggetizer-assisted canonicalization:

- at most one hosted call per non-empty subnarrative;
- at most 20 advisory canonical claim hints per subnarrative;
- at most 3 supporting documents per claim.

Place this sentence visibly in the figure:

> Selected passages are factual authority; canonical claim hints are advisory.

Show canonical transport or admission failure as a dashed fallback to exact
extractive evidence. Do not imply that Nuggetizer admits evidence, consumes
organizer gold nuggets, or replaces source passages.

### 6. Generation handoff contract

Render one manifest document crossing a strong Retrieval/Generation boundary.

The included half lists:

- exact official narrative;
- selected passages grouped by subnarrative and cluster;
- advisory canonical claim hints;
- the only raw document IDs the model may cite.

The excluded half lists:

- complete documents and document-head windows;
- organizer TREC run;
- full-text ZIP;
- qrels;
- gold nuggets;
- RAGDoll scores.

The boundary communicates authenticated, tamper-evident context and a citation
domain. It does not claim encryption.

### 7. GPT-5.6 Sol generation

Show:

```text
authenticate complete handoff
  → render frozen topic context and prompt
  → GPT-5.6 Sol through OpenRouter
  → parse strict structured output
  → normalize citations
  → validate
  → organizer submission record
```

Use **GPT-5.6 Sol** inside the figure. Put `openai/gpt-5.6-sol`, medium
reasoning, the 12,000-token ceiling, timeout, concurrency, and config path in the
expandable note.

Keep the TREC run, ZIP, qrels, gold nuggets, and RAGDoll scores physically
outside the Generation container rather than drawing them as blocked inputs
after Sol.

### 8. Validation and retries

Separate two retry domains:

1. **Transport:** at most three attempts for one provider call.
2. **Semantic:** at most two model attempts total for one topic.

The main flow is:

```text
raw-document-ID draft
  → normalize and rebuild references
  → validation decision
     ├─ pass → integer organizer citation indexes
     └─ fail → one new semantic model attempt using the same evidence boundary
```

Do not label the semantic loop as deterministic code repair. It is another
bounded model attempt with retry instructions.

### 9. Organizer output split

Show two sibling output families:

**Retrieval creates**

- `r_output_trec_rag_2026.tsv`;
- `retrieval_with_text.jsonl.zip`;
- `retrieval_export_manifest.json`;
- `generation_handoff_manifest.json`.

**Generation consumes only the handoff and creates**

- `rag_output_trec_rag_2026.jsonl`;
- its private generation-state records under the configured output directory.

Show that the retrieval run and ZIP go to organizer retrieval evaluation, while
the RAG JSONL goes to organizer answer evaluation. Generation must have no read
arrow from the run or ZIP.

## Naming and Detail Policy

Show recognizable names inside diagrams:

- DeepSeek;
- Pyserini;
- ClimbMix;
- Mixedbread;
- Nuggetizer;
- GPT-5.6 Sol;
- OpenRouter where it clarifies a hosted boundary.

Put these items in captions or expandable technical notes:

- full model aliases and pinned revisions;
- schema and prompt versions;
- config and source paths;
- cache and resume mechanics;
- concurrency and device placement;
- provenance hashes;
- quality caveats.

This division keeps the main images readable while preserving exactness for
contributors.

## Source Provenance

The report is source-backed by tracked content at commit `7e9d3b9`, primarily:

- `AGENTS.md`;
- `code/trec_rag/README.md`;
- `configs/rag26_competition_retrieval_v2.yaml`;
- `configs/rag26_competition_rag_gpt_sol_v2.yaml`;
- `code/trec_rag/competition_retrieval.py`;
- `code/trec_rag/competition_rag.py`;
- `code/trec_rag/facet_extraction.py`;
- `code/trec_rag/canonical_nuggets.py`;
- `code/trec_rag/generation_handoff.py`;
- `code/trec_rag/topic_records.py`;
- the official track references in the pinned `trec-rag-skills` submodule.

Every technical note links to the relevant repository file or section. Claims
derived from code rather than an explicit prose contract are labeled as such.
No external network source is required at report runtime.

## Agent Documentation

Add a concise `## Architecture Orientation` section to `AGENTS.md` with:

- a link to the HTML walkthrough;
- a nearby source link naming the `.qmd` as the file agents should edit and
  re-render;
- a statement that it covers only the supported competition path;
- the ordered Retrieval → handoff → Generation boundary;
- the following invariants:
  - the untouched narrative remains a retrieval lane;
  - DeepSeek planning is bounded and one-shot;
  - 1,000 documents and 100 passages are per-query ceilings with a unit change;
  - local selected passages, not canonical hints, are factual authority;
  - Generation consumes only the authenticated handoff;
  - Generation never opens the run, ZIP, qrels, gold nuggets, or RAGDoll scores;
  - transport and semantic retry limits are separate.

Keep the existing operational commands and run-safety rules intact.

## Interaction and Progressive Enhancement

- Use Quarto's generated table of contents and ordinary anchor links for
  section navigation.
- Use native `<details>`/`<summary>` for technical notes.
- Use Quarto's built-in figure enlargement rather than a hand-built modal;
  `Escape`, the close control, and focus return must work.
- Provide a direct SVG link beside every enlargement control.
- Do not require hover to access information.
- With JavaScript disabled, all nine images, captions, text equivalents,
  disclosures, and direct SVG links remain usable.
- Respect `prefers-reduced-motion`; no looping or decorative motion is needed.
- Print output removes interactive chrome while retaining every figure,
  caption, text equivalent, and source note.

## Accessibility and Responsive Behavior

Each SVG must contain:

- `role="img"`;
- a unique `<title>` and `<desc>` referenced by `aria-labelledby`;
- visible labels with a minimum readable size;
- a companion visible text equivalent in the report.

The report must:

- support widths down to 320 pixels without page-level horizontal overflow;
- reflow large diagrams into a legible mobile composition rather than scaling
  desktop text below 11 screen pixels;
- preserve keyboard order and visible focus;
- maintain sufficient contrast in light and dark modes;
- retain meaning in grayscale and when color perception differs;
- avoid text embedded as raster pixels;
- make image enlargement usable without right-clicking.

## Failure and Fallback Behavior

This is a static report, so runtime failure handling is deliberately small:

- If JavaScript fails, native reading and disclosure behavior remains intact.
- If an SVG fails to load, its visible text equivalent, caption, and direct
  source link remain present.
- If image enlargement is unavailable, direct SVG navigation remains usable.
- Broken source or asset links fail the smoke tests.
- No report code fetches network content, private artifacts, or API data.

## Verification

### Quarto render verification

Before static or browser checks:

1. record `quarto --version` in the verification evidence;
2. render the canonical `.qmd` with the documented command;
3. confirm the output is
   `reports/2026-competition-architecture.html`;
4. confirm a second clean render produces no diff in the checked-in HTML;
5. confirm no `_quarto.yml` or other repository-wide Quarto project file was
   added.

The `.qmd` is reviewed as the authored source. The `.html` is reviewed as the
rendered reader experience. A correction to prose, structure, or styling must
be made in the source and re-rendered, never patched into generated HTML.

### Static smoke tests

`reports/2026-competition-architecture.test.js` must assert:

- the canonical `.qmd` exists and identifies embedded-resource, light/dark,
  table-of-contents, and figure-enlargement behavior;
- the rendered HTML exists and identifies Quarto as its generator;
- all nine expected SVG paths exist and are referenced in order;
- each SVG has an accessible title and description;
- all nine visible text equivalents exist;
- the stable shape/color legend exists;
- the fictional narrative is marked illustrative;
- `≤1,000 documents / focused query` and
  `≤100 passages / focused query` appear in the retrieval map and quantity
  section;
- documents, chunks, passages, selected evidence, and advisory hints remain
  distinct terms;
- DeepSeek, Pyserini, ClimbMix, Mixedbread, Nuggetizer, and GPT-5.6 Sol appear;
- the Nuggetizer section states that passages are factual authority and hints
  are advisory;
- the handoff's included and excluded contents are present;
- the Sol section has no input edge from the run, ZIP, qrels, gold nuggets, or
  RAGDoll scores;
- the three-transport-attempt and two-semantic-attempt limits are distinct;
- the report contains no external script, stylesheet, font, image, or fetch
  dependency;
- Quarto's enlargement controls, direct SVG links, and native technical
  disclosures exist.

Extend `reports/index.test.js` to require the new report entry. Add focused
checks that `AGENTS.md` and `README.md` link to the report and retain the required
architecture language.

### Browser verification

Use the repository's headless browser support at minimum on:

- 1440×1000 desktop light;
- 1440×1000 desktop dark;
- 390×844 mobile light;
- 390×844 mobile dark;
- print media;
- a grayscale screenshot or equivalent contrast inspection.

For every viewport/theme combination verify:

- no page-level horizontal overflow;
- no clipped or overlapping labels;
- no text below 11 screen pixels;
- no console errors or failed asset loads;
- all section links land correctly;
- native disclosures open and close;
- all nine enlargement controls work;
- keyboard focus enters and leaves the enlarged view correctly;
- direct SVG links open the intended diagram;
- the Retrieval/Generation boundary and forbidden-input relationship remain
  obvious without color.

Run the existing report smoke tests after the new focused tests to catch
regressions:

```bash
node reports/index.test.js
node reports/trec-rag-briefing-report.test.js
node reports/2025-promising-rag-architecture.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
```

## Acceptance Criteria

The work is complete when:

1. a friend can accurately retell the path from narrative to organizer RAG
   JSONL after reading the visible story only;
2. the reader can distinguish documents, chunks, passages, selected evidence,
   and advisory hints;
3. the reader understands why the system uses bounded planning rather than an
   open-ended retrieval agent loop;
4. the 1,000-document and 100-passage values are understood as per-query
   ceilings with a unit change;
5. Nuggetizer is understood as advisory canonicalization after local evidence
   selection;
6. only `generation_handoff_manifest.json` visibly crosses from Retrieval into
   Generation;
7. GPT-5.6 Sol, validation, retry limits, and deterministic citation conversion
   are visible without crowding the overview;
8. every diagram works in light, dark, mobile, desktop, print, keyboard, and
   grayscale contexts;
9. the canonical `.qmd`, rendered report, and nine local SVGs are discoverable
   from `AGENTS.md`, the root README, and the reports index;
10. a clean Quarto re-render leaves the checked-in HTML unchanged;
11. the branch contains no copied cache, output, environment, or private run
    artifact.

## Approved Decisions

- Quarto `.qmd` is the canonical authored source; generated standalone HTML is
  the primary friend-facing artifact.
- This is a standalone Quarto document, not a repository-wide Quarto project.
- Generated HTML is never edited by hand.
- The guide covers only the supported 2026 competition path.
- One whole-system overview is followed by a guided story.
- Retrieval and Generation receive separate detailed system images.
- The guide uses nine focused diagrams.
- The style is editorial/scientific with conventional architecture semantics.
- Funnel methodology remains conceptual, not proportional geometry.
- Recognizable names appear in images; exact identifiers live in expandable
  notes.
- The fictional extreme-heat narrative is the running example.
- Key 1,000/100 ceilings appear as badges and in a dedicated quantity figure.
- The page uses a normal vertical reading flow rather than an animated stepper.
- `AGENTS.md` stays concise and links to the report.
- The root README and reports index also link to the report.
- Each diagram is stored as an individual local, theme-aware SVG.
- Essential explanations remain visible; deep technical detail is expandable.

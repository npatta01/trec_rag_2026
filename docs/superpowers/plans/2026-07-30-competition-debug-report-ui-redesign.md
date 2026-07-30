# Competition Debug Report UI Redesign Implementation Plan

> **Execution:** Use the subagent-driven workflow with a fresh implementer and review gate for each task. Apply TDD for behavior changes and run verification from `/tmp/trec-rag-competition-paths` with the existing `.venv`.

**Goal:** Turn the sealed two-topic post-run report into a readable, self-contained investigation UI that displays one topic at a time, explains each selected document with its strongest stored passage, and renders RAG answers as linked prose.

**Architecture:** Keep artifact loading and validation in `competition_debug_report.py`; add small deterministic presentation projections beside the existing report dataclasses and replace only the HTML renderer. The report remains a single escaped HTML document with inline CSS and progressive inline JavaScript. All visible claims must come from already validated data or the standard RAG config.

**Constraints:** Do not alter retrieval, RAG generation, organizer artifacts, submission formats, or the organizer repository. Do not add network dependencies. Keep the rendered copy private on the already-authorized tailnet-only portal.

---

## Task 1: Project human-readable selected-document evidence and RAG provenance

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Test: `code/tests/test_competition_debug_report.py`

1. Add failing tests that require:
   - a selected document's best stored passage to be the lowest `aggregate_rank`, with decomposition order as the deterministic tie-break;
   - an explicit no-passage state;
   - a plain-language selection reason derived from original/facet membership and the sealed selected lane/rank;
   - provider and model to be retained from the standard RAG config, while run ID/description continue to come from validated output metadata.
2. Run the focused tests and confirm they fail for the missing projection fields/helpers.
3. Extend `RagOutputReport` with validated provider/model fields and populate them in the existing strict RAG loader. Add private, pure projection helpers for selection reason and best-passage choice; never compare logits between subnarratives.
4. Run the focused tests, then the complete debug-report test module.
5. Commit only these data-projection changes with message `Add debug report presentation projections`.

## Task 2: Replace the all-topics page with an accessible single-topic shell

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Test: `code/tests/test_competition_debug_report.py`

1. Add failing renderer tests for:
   - native named `<details>` topics with only the first topic initially open;
   - one self-contained topic switcher with semantic buttons/labels, URL-fragment restoration, left/right keyboard support, and exactly-one-open enforcement in bundled JavaScript;
   - collapsed validation/configuration receipts;
   - no external stylesheet, script, font, or image dependency;
   - source-derived values remaining HTML-escaped and absent from executable JavaScript.
2. Run the focused tests and confirm the current all-topics navigation fails them.
3. Implement the topic shell and responsive visual system in the existing renderer. Use native `<details name="competition-topic">` as the no-JavaScript fallback and progressive JavaScript only for switching/state restoration. Keep 44px controls, visible focus, light/dark schemes, reduced-motion handling, and no page-level horizontal overflow.
4. Run the focused tests and the complete debug-report test module.
5. Commit with message `Add single-topic debug report navigation`.

## Task 3: Render selected documents as evidence cards and RAG output as linked prose

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Test: `code/tests/test_competition_debug_report.py`

1. Add failing renderer tests that require:
   - the selected-document stage to contain ordered cards, selection reason, strongest passage excerpt, subnarrative, aggregate rank, and collapsed technical provenance;
   - the old selected-document table/caption to be absent;
   - each RAG item to render as a numbered paragraph with citation chips linked to stable reference-card anchors;
   - each reference card to contain citation index, DocID, bounded excerpt, and collapsed document detail;
   - a provenance banner showing implementation, run metadata, provider/model, validation state, and word count;
   - the old RAG answer table/caption to be absent.
2. Run the focused tests and confirm they fail against the current table renderer.
3. Replace the two wide tables with card/article renderers. Keep technical trace data available through disclosures and preserve stable, collision-safe anchor namespaces.
4. Run focused tests, the complete debug-report module, and the full repository test suite.
5. Commit with message `Improve debug report evidence presentation`.

## Task 4: Rebuild, visually verify, and refresh the private portal artifact

**Files/artifacts:**
- Source report: `/home/npatta01/data/competitions/trec_rag_2026/outputs/codex-pr24-two-topic-smoke/competition_debug_report.html`
- Derived portal copy: `/home/npatta01/codex-rendered/plans/trec-rag-2026-competition-debug-report.html`
- Portal index: `/home/npatta01/codex-rendered/index.html` (change only if its existing link/description needs correction)

1. Record organizer/checkpoint repository state and sealed artifact hashes before rebuilding.
2. Invoke the repository's post-run debug-report CLI with the existing two-topic retrieval and RAG configs; do not rerun retrieval or generation.
3. Validate the generated HTML is self-contained and contains both topics, the standard RAG provenance (`openrouter`, `openai/gpt-5.6-sol` for this run), document evidence cards, citation targets, and no legacy selected-document/RAG tables.
4. Use headless Chrome at desktop and mobile viewport sizes to verify:
   - only one topic is visible at a time;
   - direct topic fragments and left/right controls switch topics;
   - citation links resolve to reference cards;
   - `document.documentElement.scrollWidth <= document.documentElement.clientWidth`;
   - screenshots show legible cards and answer prose.
5. Run the full test suite again and confirm organizer/checkpoint state and sealed artifacts are unchanged.
6. Copy only the sanitized derived HTML to the existing rendered location, preserving private permissions. Verify the source and portal-copy SHA-256 match, live HTTPS returns the new artifact, and `tailscale serve status` still reports tailnet-only with no Funnel.
7. Commit any final source/test adjustments only; never commit generated private report data.

## Task 3b: Close the remaining stage-presentation and accessibility contracts

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Test: `code/tests/test_competition_debug_report.py`

1. Add failing renderer tests that require subnarratives to use readable cards,
   new documents to remain grouped but progressively disclosed, passage ranking
   tables to remain behind per-subnarrative disclosures, canonical diagnostic
   records to remain behind disclosures, and final retrieval to use compact
   cards with detailed provenance collapsed.
2. Rename the cross-facet passage projection in the UI to “Representative
   stored passage” and display its deterministic selection rule. Do not call it
   strongest or imply that facet-local aggregate ranks form a semantic global
   leaderboard.
3. Give every disclosure summary a 44-pixel minimum target and consolidate the
   bounded-excerpt implementation.
4. State explicitly that implementation authorship is not present in sealed
   run artifacts; keep provider/model/run provenance derived from the standard
   config/output rather than hard-coding a person or PR.
5. Run focused tests, the complete debug-report module, and the full repository
   suite. Commit with message `Complete debug report stage presentation`.

## Completion Evidence

- Focused debug-report tests pass.
- Full repository suite passes.
- Desktop and mobile Chrome checks pass with screenshots retained only as local scratch evidence.
- Two-topic post-run report renders one active topic, explanatory selected-document cards, and linked RAG prose.
- Live private portal artifact matches the generated source by SHA-256.
- Organizer repository, retrieval/RAG outputs, and competition submission artifacts remain unchanged.

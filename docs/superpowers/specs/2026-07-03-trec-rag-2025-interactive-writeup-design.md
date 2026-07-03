# TREC RAG 2025 Interactive Writeup Design

## Goal

Build a single self-contained HTML page that explains the TREC RAG 2025 team approaches to a reader who is new to retrieval-augmented generation.

## Artifact

- Create `docs/trec-rag-2025-writeups/interactive-writeup.html`.
- Create `docs/trec-rag-2025-writeups/interactive-writeup.test.js` as a lightweight smoke test.
- Do not require a dev server, package install, or build step.

## Audience

The reader does not know the field. The page should define the core vocabulary before comparing approaches:

- retrieval
- generation
- chunks or segments
- citations
- reranking
- query decomposition
- nuggets
- relevance judgment
- support checking

## Content Structure

1. Opening explanation: what RAG is and why TREC 2025 made the task harder with long narrative queries.
2. Task setup: retrieval, augmented generation, end-to-end RAG, and relevance judgment.
3. Interactive pipeline: query -> retrieve -> select/organize evidence -> generate answer -> verify citations.
4. Approach patterns: hybrid retrieval, decomposition, reranking, nuggets/evidence cards, agentic loops, citation-first generation, judging/calibration.
5. Cross-team synthesis: compare architecture families, retrieval methods, evidence organization, generation/citation strategies, and official-score tradeoffs.
6. Team explorer: 15 official team writeups with deep dossiers. Each team must include architecture, what they tried, models/tools, results or observations, caveats, reusable lessons, and source PDF links.
7. Takeaways: practical lessons we can use when building a RAG system.

## Interaction

- Filter chips should show/hide team cards by approach tag.
- A search box should filter team cards by text.
- Expand/collapse controls should reveal team details.
- Pipeline step buttons should update an explanatory panel.
- Glossary pills should update a plain-English definition panel.
- The deeper team dossiers should remain collapsible so the page stays approachable for a beginner.

## Visual Direction

The page should feel like a clean research briefing, not an academic PDF. Use a restrained but not one-note palette, clear typography, compact cards, and code-native diagrams. Avoid decorative blobs, oversized hero marketing, nested cards, and filler.

## Source Grounding

Use the official local PDF collection and manifest in `docs/trec-rag-2025-writeups/`. Include links to the local PDFs and official NIST proceedings pages. Note that the participant roster has additional run submitters without official proceedings PDFs.

## Verification

The smoke test must fail before the HTML exists and pass after implementation. It should verify:

- required sections exist
- the 15 team entries are present
- all 15 team entries are marked as deep dossiers
- each dossier includes Architecture, What they tried, Models and tools, Results and observations, Caveats, and What we can reuse
- expected approach tags are represented
- embedded JavaScript contains handlers for filtering, search, glossary, pipeline, and card expansion
- all local PDF links referenced in the team data exist

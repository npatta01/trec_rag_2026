# README Reviewer Poster Design

## Objective

Make the GitHub README explain the completed NP Labs TREC RAG 2026 project to
competition reviewers and newcomers even when they do not open the artifact
hub. Preserve the hub as the primary destination while turning the top of the
README into a concise, visual project poster.

## Reader-facing structure

The README begins with the project identity and a prominent link to the
published artifact hub. It then presents, in this order:

1. A plain-language problem statement describing evidence retrieval and cited
   answer generation for 119 official narratives.
2. The existing whole-system architecture SVG at a readable full-column width.
   The image and its caption link explicitly to the full Architecture Report.
3. A compact verified-results table covering 119 narratives, three Retrieval
   runs, two RAG runs, and five Evalbase-accepted organizer files.
4. Four explicit reviewer links:
   - Architecture Report
   - Retrieval Quality Report
   - RAGDoll Evaluation — Single-pass RAG
   - RAGDoll Evaluation — Multi-stage RAG
5. The existing repository references and developer notes, kept below the
   reviewer-facing material.

## Visual and content constraints

- Reuse `reports/2026-competition-architecture/01-whole-system.svg`; do not
  invent a second architecture or create a decorative illustration.
- Use standard GitHub-flavored Markdown so the README works on desktop, mobile,
  light mode, and dark mode without custom runtime dependencies.
- Give the diagram descriptive alternative text and retain nearby text links so
  the page remains useful without images.
- Keep claims aligned with the submission ledger and final architecture report.
- Do not expose private caches, raw evaluation data, credentials, or unpublished
  artifacts.
- Keep setup and testing instructions available, but visually subordinate to
  the project story.

## Verification

Extend the README contract test to assert the problem statement, architecture
image and report target, verified result counts, and all four explicit report
links. Run the README, hub, reports-index, and architecture smoke tests, plus
Markdown link and whitespace checks. Verify the rendered README from the public
`master` branch after merge.

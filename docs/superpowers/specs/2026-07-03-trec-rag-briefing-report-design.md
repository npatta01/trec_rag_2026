# TREC RAG Briefing Report Design

## Goal

Build an interactive HTML briefing for people who are new to TREC RAG, retrieval, and RAG evaluation. The report should explain what the 2026 competition is about, what changed from 2025, what the ClimbMix collection is, and how a practical system might approach retrieval and cited answer generation.

The tone should be approachable and concrete. A reader should leave with enough context to understand the task files, discuss retrieval strategies, and start designing a baseline or stronger system.

## Audience

The primary reader is technically comfortable but new to information retrieval and TREC-style evaluation. The report should assume basic familiarity with LLMs and search, but should define terms such as BM25, dense retrieval, reranking, RRF, HyDE, sub-narrative, qrels, and citation support.

## Deliverable

Create a single self-contained HTML file:

```text
trec-rag-briefing-report.html
```

The file should run directly in a browser without a build step or server. It should include embedded CSS and JavaScript for navigation, expandable examples, and lightweight interactivity.

## Content Structure

### 1. Orientation

Explain the competition in plain language:

- TREC RAG asks systems to retrieve useful source documents and generate cited answers for complex information needs.
- TREC RAG 2026 has one track with two main tasks: Retrieval and Retrieval-Augmented Generation.
- Retrieval output is a ranked list of document IDs.
- RAG output is a human-readable answer with sentence-level citations.

This section should make clear that the work is not only about making an LLM answer; it is also about finding enough relevant evidence and attributing claims to the right documents.

### 2. What Is ClimbMix?

Explain the 2026 corpus:

- ClimbMix is a broad curated web/pretraining-style text collection, not a narrow domain collection.
- The official unit is a ClimbMix row/document with a ClimbMix document ID.
- A team may chunk documents internally, but submitted citations and retrieval rows need to map back to official ClimbMix document IDs.
- The collection is accessed through the Pyserini/BM25 baseline infrastructure and related official tooling.

Avoid overemphasizing raw data numbers here. Put detailed development-data counts in an appendix.

### 3. What Happened In 2025?

Give historical context:

- TREC RAG 2025 used the MS MARCO V2.1 segmented document collection.
- The 2025 track included Retrieval, Augmented Generation, Retrieval-Augmented Generation, and Relevance Judgment tasks.
- Queries were long narrative information needs.
- Official analysis focused on a subset of assessed narratives, with relevance and answer quality judged around sub-narrative coverage, completeness, and attribution.

This section should frame 2025 as the predecessor that taught the community what worked for long narrative RAG.

### 4. Beginner-Friendly Technique Gallery From 2025

This is a major section. It should teach techniques through simple cards, not just list winning runs.

Each technique card should include:

- What it means.
- Why a team might use it.
- When it helps.
- How it can fail.
- A small example or analogy where useful.

Cards to include:

- BM25 / lexical retrieval.
- Query expansion.
- Query decomposition into facets or sub-questions.
- Dense retrieval with embeddings.
- Hybrid sparse plus dense retrieval.
- SPLADE-style learned sparse retrieval.
- Reciprocal Rank Fusion.
- HyDE, using a hypothetical answer as a search probe.
- LLM reranking.
- Evidence selection from retrieved candidates.
- Citation-first answer generation.
- Post-generation citation and consistency checks.

The section should mention 2025 systems as examples without making the page feel like a leaderboard. UTokyo's strong retrieval run is useful for explaining hybrid retrieval, RRF, HyDE, and LLM reranking. Other runs and approaches can be used as examples where they clarify evidence selection, citation-first generation, or query decomposition.

If a top-scoring RAG run lacks an easily available participant-method writeup, label it carefully as a top-scoring run from the overview tables rather than inventing method details.

### 5. 2025 To 2026: What Changed?

Compare the two years in a compact table:

- 2025 corpus: MS MARCO V2.1 segments.
- 2026 corpus: ClimbMix document rows.
- 2025 tasks: Retrieval, AG, RAG, RJ.
- 2026 tasks: Retrieval and RAG.
- 2025 fixed-evidence AG task existed; 2026 emphasizes participant-owned retrieval for RAG.
- 2025 techniques still matter, but systems must now adapt them to ClimbMix document IDs and larger web-style data.

### 6. How I Would Approach Solving 2026

This is the other major section. Use multiple worked examples.

Recommended pipeline:

1. Parse each topic into title, narrative, and facets.
2. Run a direct BM25 search as a baseline.
3. Generate several facet queries from the narrative.
4. Search each facet query.
5. Merge candidates with RRF.
6. Deduplicate and map any local chunks back to parent ClimbMix document IDs.
7. Rerank candidates against the full narrative.
8. Select evidence chunks or passages for answer planning.
9. Generate short cited answer sentences.
10. Run citation and coverage checks before submission.

Worked examples should show:

- Direct narrative search as a baseline.
- Facet decomposition improving recall.
- A case where decomposition can add noise.
- How multiple retrieved documents become a coherent answer.
- How an uncited claim should be removed, softened, or backed by another citation.

### 7. Evaluation And Submission Mental Model

Explain evaluation at a practical level:

- Retrieval is graded using judged document relevance.
- Relevance is tied to how much of the narrative or sub-narratives a document helps answer.
- RAG is judged on whether the answer covers important information needs and whether claims are supported by citations.
- Since long narratives have multiple facets, high-scoring systems need both breadth and precision.

Keep exact development-data counts and file-specific details in an expandable appendix.

### 8. Appendix

Include collapsible reference sections:

- 2026 submission file formats.
- 2026 development data counts and what each file means.
- Example retrieval run row.
- Example RAG JSONL object shape.
- Source list and caveats.

## Interaction Design

The report should feel like a hybrid slide deck and explainer:

- Sticky section navigation.
- Progress indicator.
- Expandable cards for technique details.
- Toggleable "beginner" and "implementation" notes where useful.
- Worked examples with step-by-step reveal controls.
- Comparison tables for 2025 vs 2026 and technique tradeoffs.

No login, backend, external package, or network access should be required to read the report.

## Visual Style

The visual style should be clear, calm, and technical:

- Dense enough for serious reading.
- Friendly enough for newcomers.
- Use modest color accents to distinguish 2025, 2026, retrieval, generation, and evaluation concepts.
- Avoid marketing-page hero treatment; the first screen should immediately orient the reader to the problem.

## Source Grounding

The report should cite or link to:

- Official TREC RAG 2026 guidelines and development-data documentation from the local `trec-rag-skills` and `trec-rag-data` repositories.
- Official TREC RAG 2025 page.
- TREC RAG 2025 overview paper.
- NIST 2025 retrieval appendix.
- UTokyo 2025 TREC RAG paper for the well-documented top retrieval approach.
- TREC Browser run/proceedings pages for examples of other submitted methods when useful.

Statements about system rankings or exact metrics should be phrased narrowly and tied to the source that supports them.

## Error Handling And Caveats

The report should avoid overstating uncertain details:

- Do not imply that the 2025 top retrieval run was also the top full RAG answer run.
- Do not describe an unpublished or unavailable participant method as if it were known.
- Explain that development qrels are useful for comparison but are not exhaustive corpus-wide truth.
- Explain that local chunking is allowed as an internal strategy, but official IDs still matter for submitted retrieval rows and citations.

## Testing

Before marking the report complete:

- Validate that the HTML file opens locally.
- Check that all interactive controls work.
- Check mobile and desktop layouts.
- Check that text does not overflow cards, buttons, tables, or examples.
- Check that source links are present for factual sections.
- Check that the report contains no placeholder text.

## Out Of Scope

This report will not:

- Implement a retrieval or RAG pipeline.
- Download or index ClimbMix.
- Evaluate submissions.
- Create a full slide-export workflow.
- Depend on a frontend framework or build system.


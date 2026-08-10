# Final Submission Documentation Handoff Design

**Date:** 2026-08-09

**Status:** approved in conversation

## Objective

Leave the completed TREC RAG 2026 repository with one accurate, approachable
entrypoint to the final system, accepted submission files, architecture,
validation workflows, skills, reports, implementation, and official inputs.
The handoff must describe what was actually submitted rather than presenting an
earlier experiment as the supported final path.

## Verified Starting State

The five checked-in organizer files were uploaded and accepted by Evalbase, as
confirmed by the user on 2026-08-09. A fresh combined preflight against the
pinned official 119-topic release passed:

- all three Retrieval TSV files contain 4,246 rows across 119 topics, with
  narrative-specific depths from 1 to 121;
- both RAG JSONL files contain exactly 119 valid reports with no missing,
  extra, or duplicate topics;
- the SHA-256 of every checked-in upload file matches its value in
  `submissions/trec-rag-2026/SUBMISSION_LEDGER.md` and bundle metadata;
- the pinned `trec-rag-data` and `trec-rag-skills` submodule commits still
  match the current official repository heads.

The private source artifact and sealed generation handoff are not present in
this linked worktree. Their additive integrity, citation-domain, and exact-hint
checks must therefore remain described as previously completed and recorded in
the checked-in bundle metadata, not represented as freshly rerun here.

## Final Architecture Story

The architecture report will explain one frozen source system followed by two
submission branches:

```text
Official narrative
  -> bounded one-shot DeepSeek planning
  -> original and subnarrative BM25 lanes over ClimbMix
  -> Mixedbread passage scoring and selected evidence
  -> authenticated facet-deepseek-b40-v3 source artifact
       |-> cache-first candidate core -> three Retrieval TSV runs
       `-> sealed selected-evidence handoff -> two RAG JSONL runs
```

The Retrieval branch admits a robust variable-depth candidate core from the
authenticated source lanes, completes only the targeted narrative ×
subnarrative score matrix, and holds the candidate set fixed while producing
three orderings:

1. narrative plus subnarrative (`r26-narr-facet-v1`);
2. subnarrative evidence breadth (`r26-facet-breadth-v1`);
3. narrative only (`r26-narrative-v1`).

The RAG branch authenticates the selected-evidence handoff and produces two
independent generation strategies:

1. bounded multi-stage Luna planning/auditing/screening plus Sol drafting and
   bounded revision (`rag26-ms1-final`);
2. single-pass Sol over the same sealed evidence (`rag26-ss1`).

The report must state explicitly that the organizer-facing Retrieval TSV is
not an input to either RAG generator. The Retrieval and RAG bundles are sibling
products with shared source provenance. Generation reads only its sealed
selected-evidence handoff and never reads the Retrieval TSV, full-text ZIP,
qrels, gold nuggets, or RAGDoll scores.

## Documentation Surfaces

### Root artifact hub

Create `index.html` as a standalone, dependency-free landing page modeled on
the compact card grid in `npatta01/music-crs-2026`. It will link six concerns:

- **Architecture:** the rendered final-system walkthrough and canonical QMD;
- **Accepted submissions:** the submission ledger and the Retrieval/RAG bundle
  READMEs;
- **Validation and skills:** the repo-local submission validator, official
  track contract, Pyserini API skill, and private debug-report skill;
- **Reports:** the report index, 2026 briefing, and 2025 method library;
- **Code:** the competition README, final source modules, and checked-in
  canonical configs;
- **Official inputs:** the pinned data and skills submodules.

The page will use responsive cards, system fonts, dark/light color-scheme
support, visible keyboard focus, reduced-motion handling, and print-safe
styles. It will not publish or expose private outputs.

### Architecture report

Keep `reports/2026-competition-architecture.qmd` as the canonical authored
source and regenerate `reports/2026-competition-architecture.html` with
Quarto. Reuse the existing local themes, CSS, and image-led explanatory style.
Revise or replace the diagrams under
`reports/2026-competition-architecture/` so every visual matches the final
two-branch architecture. Each visual keeps an informative alt description and
a nearby text equivalent.

The report will contain, in reading order:

1. final system map and the frozen-source branching point;
2. bounded source retrieval and selected-evidence construction;
3. robust candidate admission and variable `k`;
4. targeted cached scoring and the three Retrieval orderings;
5. the sealed RAG handoff boundary;
6. single-pass RAG;
7. multi-stage RAG and its bounded fallback behavior;
8. local validation, final five-file artifact map, and accepted status;
9. source/provenance notes and links to the exact checked-in artifacts.

The report may summarize aggregate, privacy-reviewed values already committed
in bundle metadata, including 119 topics, 4,246 Retrieval rows per run,
variable depth 1–121, 691 evidence groups, provider-call counts, costs, answer
word ranges, reference ranges, and recorded warnings. It must not quote test
narratives, document text, prompts, provider responses, or generated answers.

### Repository navigation and status

Update the following surfaces to use the same canonical links and vocabulary:

- `README.md`: lead with a completed-project status and a short key-artifact
  map before historical/research material;
- `reports/index.html`: prioritize the final architecture and link back to the
  root artifact hub and accepted submissions;
- `submissions/trec-rag-2026/SUBMISSION_LEDGER.md`: change all five statuses to
  Evalbase accepted and record portal IDs/timestamps as `Not recorded` rather
  than inventing values;
- the Retrieval and RAG bundle READMEs: describe the files as submitted and
  accepted while preserving hashes, validation evidence, and portal notes;
- `AGENTS.md`: add a prominent completed-project/key-artifacts section and
  make the final bundle architecture authoritative for future orientation.

The agent instructions must distinguish canonical authored files from
generated HTML, preserve the five accepted uploads, point to the validation
skill and official submodule contracts, retain privacy boundaries, and prevent
experimental plans or configs from being mistaken for unfinished work.

## Link Rules

- Use repository-relative links in Markdown and HTML so local browsing and
  GitHub Pages-style hosting both work.
- Link to bundle directories through their READMEs and to exact organizer files
  through the submission ledger.
- Link generated reports from reader-facing pages and link their canonical
  sources where future maintenance needs them.
- Do not link ignored `outputs/`, caches, private work state, raw provider
  material, or absent local-only paths.
- Preserve the current repository-private handling rules even though the final
  organizer files are tracked.

## Verification

Completion requires:

1. rerun the combined `validate-trec-rag-2026-submissions` preflight for all
   five files and retain the normalized counts/status in the handoff;
2. recompute all five SHA-256 values and compare them with the ledger and
   bundle metadata;
3. render the architecture QMD with the repository-supported Quarto version;
4. run the architecture and report-index Node smoke tests;
5. add and run a root landing-page smoke test covering required links,
   accessibility landmarks, responsive styles, and absence of private paths;
6. check internal links and repository-relative targets for the root page,
   README, report index, architecture report, submission ledger, and bundle
   READMEs;
7. inspect the rendered root page and architecture report at desktop and
   mobile widths with Playwright when available;
8. run `git diff --check` and confirm no submission bytes, secrets, private
   data, or unrelated files changed.

## Out of Scope

- No retrieval, reranking, generation, evaluator, or hosted-model run.
- No changes to any of the five accepted organizer files or their metadata
  hashes.
- No public deployment, GitHub Pages configuration, Tailscale publication, or
  sharing-permission change.
- No reconstruction of missing Evalbase IDs or timestamps.
- No rewrite of historical experiment records, plans, or specifications.

## Completion Criteria

A first-time reader can start at `index.html` and reach the final architecture,
all accepted submissions, their validation path, agent skills, reports, code,
and official inputs without guessing. Every surface describes the same final
two-branch system and the accepted state, and all rendered pages pass the
specified static and responsive checks.

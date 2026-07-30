# Competition Debug Report Design

## Objective

Add a read-only, post-run debugging path that explains a completed competition
retrieval and optional RAG generation run as one self-contained HTML report.
The report must make each pipeline stage inspectable without rerunning
retrieval, reranking, DeepSeek planning, canonicalization, or answer generation.

The change has two deliverables:

1. a repository CLI, `trec_rag.competition_debug_report`; and
2. a generic repository agent skill that invokes the CLI for any compatible
   completed competition run.

Neither deliverable changes organizer submission artifacts or the behavior of
`trec_rag.competition_retrieval` and `trec_rag.competition_rag`.

## Operator Interface

The primary command accepts the same standard configuration files as the two
competition paths:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/example-retrieval.yaml \
  --rag-config configs/example-rag.yaml
```

`--retrieval-config` is required. `--rag-config` is optional so a completed
retrieval-only run can still be explained. The CLI also accepts repeatable
`--topic` selectors and an optional `--output` override. Without `--output`, it
writes `competition_debug_report.html` inside the retrieval run's configured
output directory.

The CLI does not expose a passage-count option. It reads the complete stored
passage ranking and uses progressive disclosure in HTML: the first five ranked
entries are immediately visible and all remaining entries are expandable.

The CLI does not expose a deep-verification mode. Normal report generation must
not scan `canonical/candidates.jsonl`; those ledgers are approximately 17 GB and
32 GB for the two-topic smoke run. Existing competition commands remain the
authority for exhaustive checkpoint validation.

On success, stdout contains one compact JSON receipt with:

- the absolute HTML output path;
- included topic IDs in official order;
- whether a RAG config/output was included;
- source artifact SHA-256 values; and
- the report schema version.

Malformed or mismatched required artifacts produce a clear error and nonzero
exit. An omitted RAG config produces a visible "RAG output not supplied" stage,
not an error.

## Artifact Model

The report builder reads only existing, bounded artifacts. It performs strict
shape, identity, ordering, join, and source-offset validation for the data it
renders. It does not duplicate the exhaustive candidate-ledger validation.

### Narrative and subnarratives

`<topic>/decomposition.json` supplies the original narrative, topic identity,
generated subnarratives, and BM25 queries. Subnarratives retain their stored
order and identifiers.

### New documents

"New document" has one precise report meaning: a document in
`scoring/selection.json`'s union pool whose membership set does not include the
`original` narrative lane. It is new relative to the original reranked eligible
pool, not new to the corpus.

New documents are grouped by their stored first-seen lane. Counts, identities,
lane provenance, ranks, memberships, and text hashes are shown for every row.
An expandable text excerpt is shown only when the document also reached the
selected pool, because the bounded sealed union artifact does not retain text
for discarded rows. The report does not read unsealed provider caches to recover
that discarded text. It must not infer newness from
`selected_from_lane != original`, because a document can be selected through a
facet lane while still belonging to the original lane.

### Selected documents

`scoring/selected_documents.jsonl` supplies the ordered selected document pool.
`scoring/selection.json` supplies memberships and trace information explaining
selection actions. The report shows selection rank, source lane and rank,
original-versus-facet-only status, memberships, selection rationale, and an
expandable excerpt.

### Top passages by subnarrative

`scoring/selected_subnarrative_scores.jsonl` supplies the complete stored
document ranking per subnarrative. Rows are grouped by stored subnarrative order
and ordered by `aggregate_rank`. Exact passage text is reconstructed by slicing
the matching selected document with each stored winning passage's character
offsets. Invalid offsets or source mismatches fail the report build.

Raw cross-encoder values are labeled as logits, never probabilities. The first
five ranked entries are open by default; remaining stored entries are available
inside an expandable section.

### Final selected nuggets

`canonical/canonical-nuggets.jsonl` is the authoritative final nugget source.
It supplies canonical claim text, nugget kind, result state, configured budget,
and supporting evidence. `canonical/subnarrative-selections.jsonl` supplies the
selected evidence clusters and budget snapshot used before canonicalization.

The report joins these artifacts by topic and subnarrative identity. It shows
the final claims, supporting document IDs, selected clusters, exact evidence
snippets, and configured caps. It does not read or display unused rows from the
candidate ledger.

### Final retrieval output

The root `retrieval_provenance.jsonl`, organizer TREC run, full-text ZIP, and
retrieval export manifest supply the final retrieval projection. The report
shows:

- the selected-pool depth and final supported depth;
- how canonical support reduced the selected pool to the organizer run;
- final rank and document identity;
- lane memberships and subnarrative scores;
- supported canonical nuggets; and
- artifact checksums and validation status.

### Final RAG output

When `--rag-config` is supplied, the standard generation config resolves its
canonical query, retrieval-run, document-archive, and output paths. The builder
uses the production competition RAG loaders and submission validator. It shows
the final answer in stored order, reference document IDs, citation-to-reference
mapping, word count, and output checksum.

The retrieval and RAG configs must refer to compatible topic identities and the
same organizer retrieval inputs. A mismatch fails rather than producing a
misleading report.

## HTML Information Architecture

The output is one deterministic, standalone HTML5 file with no external runtime
dependencies. It contains:

1. a run summary with source paths, hashes, validation state, and topic counts;
2. a compact pipeline legend defining every stage and the precise meaning of
   "new document";
3. one topic section per official topic, in official order; and
4. within each topic, ordered sections for narrative, subnarratives, new
   documents, selected documents, passages, nuggets, retrieval output, and RAG
   output.

Large collections use semantic tables, cards, and `<details>` elements rather
than truncating the underlying stored rankings. Document bodies use bounded
excerpts; exact selected passage and nugget evidence text is retained. Topic
navigation and stage anchors make the report usable on desktop and mobile.

All source-controlled strings are HTML-escaped, including narratives, queries,
document text, claims, answers, IDs, and error/status labels. Embedded JSON used
by optional inline interactions is serialized safely so source text cannot end
a script block. The baseline report remains usable with JavaScript disabled.

Accessibility requirements include semantic headings, table captions and scope
attributes, visible keyboard focus, sufficient contrast, reduced-motion
support, horizontally scrollable wide tables, and no color-only status cues.

## Privacy and Output Safety

The report contains raw corpus excerpts and document IDs. It is a private run
artifact, not a sanitized publication artifact. The default output stays under
the ignored retrieval output directory and is not copied to the rendered-plan
portal, Codex Sites, or another shared location.

The builder never reads `.env`, provider request/response caches, API keys, or
authorization headers. It writes only the requested HTML path, using an atomic
same-directory replacement so a failed build does not destroy an existing
report. Existing organizer and checkpoint artifacts remain read-only.

## Module Boundaries

`code/trec_rag/competition_debug_report.py` owns four small layers:

1. strict bounded-artifact loading and cross-artifact validation;
2. typed report-view records that define the stage data;
3. deterministic HTML rendering; and
4. CLI argument handling, atomic output, and JSON receipt emission.

The public Python seam is a builder that accepts resolved retrieval and optional
RAG configs and returns an immutable report receipt. Rendering consumes the
typed view model rather than reading files directly, keeping data contracts and
presentation independently testable.

The implementation reuses production configuration, query, TREC-run, document
archive, and RAG submission validators where their contracts match. It must not
call private routines that trigger exhaustive candidate validation as a side
effect.

## Generic Agent Skill

The tracked repo-local
`.agents/skills/trec-rag-competition-debug-report/SKILL.md` skill provides
generic instructions for any compatible agent. The implementation is delivered
in the same parent-repository PR as the report CLI, with no dependency on an
unpublished organizer-submodule revision. It triggers when a user asks to
inspect, explain, visualize, or debug a completed competition retrieval or RAG
run. The `trec-rag-skills` submodule remains pinned to the official organizer
revision used as source provenance.

The skill:

- locates the repository and standard retrieval/RAG configs;
- distinguishes completed post-run reporting from requests to execute a run;
- invokes the CLI using `uv run --no-sync .venv/bin/python`;
- passes explicit topic selectors only when the user requests a subset;
- never invokes hosted retrieval or model generation;
- reports the absolute local HTML path and the CLI's compact receipt;
- warns that the report contains private corpus text and does not publish it;
- asks for one path only when multiple completed runs are genuinely ambiguous;
  and
- remains independent of the two smoke-topic IDs and their local output paths.

The skill documentation includes example retrieval-only and retrieval-plus-RAG
invocations and the expected JSON receipt fields. It relies on the CLI for
artifact validation and does not reproduce parsing logic in shell commands.

## Verification

Test-first implementation covers:

- strict required-field and topic-identity validation;
- deterministic official topic and stored-rank ordering;
- the exact facet-only definition for new documents;
- selection trace and lane-membership joins;
- exact passage slicing and invalid-offset rejection;
- canonical nugget, cluster, budget, and supporting-document joins;
- organizer output depth and provenance rendering;
- optional RAG state and production submission validation;
- citation-to-reference rendering and word counts;
- hostile HTML/script strings and safe serialization;
- atomic output behavior and stable JSON CLI receipts;
- absence of candidate-ledger reads and external/network calls;
- generic skill trigger, command, privacy, and no-rerun instructions; and
- self-contained, accessible, responsive HTML structure.

After unit and contract tests pass, the CLI runs against the existing two-topic
smoke artifacts. The resulting report is checked for both topic IDs, all stage
headings, stored passage/nugget coverage, final retrieval depths of 71 and 60,
and two validated RAG outputs. When Playwright is available, the HTML is
inspected at desktop and mobile viewports for overflow, focus behavior, and
readability.

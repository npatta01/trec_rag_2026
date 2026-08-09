# Multipage Competition Debug Report Design

## Status

Approved in conversation on 2026-08-09.

This design extends the existing competition debug report. It does not replace
the artifact contracts established by the original report design or its
information-hierarchy follow-up.

## Problem

The completed 119-topic 2026 retrieval run produces a valid standalone debug
report that is 807,713,079 bytes. That is an appropriate amount of private trace
data to retain, but it is too large for one browser document: opening the run
requires parsing and laying out every topic even when the reader wants only the
run summary or one topic.

Splitting the report does not promise to reduce the total stored trace by much.
It changes the loading unit:

- the run summary becomes a small page suitable for routine use;
- opening one topic loads only that topic's raw trace; and
- the browser never needs to hold all 119 topic traces at once.

The summary also needs to remain useful after evaluation scores arrive without
mixing retrieval, nugget-coverage, and answer/citation measures into a false
single leaderboard.

## Goals

1. Add a deterministic static bundle containing one summary and one complete
   raw page per selected topic.
2. Keep the summary free of corpus passages, document identifiers, and other
   raw trace content.
3. Make run health, scale, fallbacks, and future evaluation scores visible at a
   glance.
4. Preserve the complete existing per-topic debugging story and its stage
   anchors.
5. Preserve the existing single-file CLI and Python interfaces for callers that
   still need them.
6. Publish a bundle only after every page, link, privacy check, and hash passes.

## Non-goals

- Reducing or rewriting the underlying sealed retrieval artifacts.
- Changing retrieval, RAG generation, evaluation, or organizer submissions.
- Combining metrics from different evaluation tasks into one score.
- Adding a server-side application, database, client-side data API, or external
  runtime dependency.
- Making raw topic traces public.
- Reading the exhaustive candidate ledgers that the existing report excludes.

## Operator Interface

The existing command and `--output FILE` behavior continue to produce one
standalone HTML file.

Multipage output is explicit through a new mutually exclusive
`--output-dir DIR` option:

```bash
.venv/bin/python -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report
```

An optional validated evaluation bundle can overlay scores:

```bash
.venv/bin/python -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --evaluation-manifest outputs/private-evaluation/evaluation_manifest.json \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report-scored
```

`--rag-config` and repeatable `--topic` selectors keep their existing meanings.
Every report topic must exist in the evaluation scope and retain the
evaluation's official relative order. A report may use an explicit subset of a
larger evaluation scope; extra evaluated topics are ignored. A missing report
topic or conflicting order fails instead of silently showing a mixture of
scored and unscored topics. An absent evaluation manifest is valid and produces
an explicit "Evaluation not supplied" state.

`--output` and `--output-dir` are mutually exclusive. A multipage target is
create-only: the command refuses an existing path. This lets the builder create
and validate a sibling staging directory and atomically rename it into place
without deleting or partially replacing an earlier report.

The repo-local competition-debug-report skill changes its default for large or
multi-topic runs to the bundle interface. It can still request legacy
single-file output when the user explicitly asks for it.

## Output Contract

The bundle has this fixed shape:

```text
competition-debug-report/
├── index.html
├── bundle-manifest.json
└── topics/
    ├── rag2026-0.html
    ├── rag2026-1.html
    └── ...
```

`index.html` is the privacy-safe run summary. Each file in `topics/` contains
the complete raw trace for exactly one selected topic. Topic filenames are
derived only from validated topic identifiers and cannot contain path
separators or traversal components.

Every HTML page is self-contained, with inline CSS and JavaScript and no
network-loaded assets. Repeating the small presentation shell is accepted in
exchange for portable, independently viewable topic pages. The raw trace data,
not the shell, is expected to dominate bundle size.

`bundle-manifest.json` is written last inside staging and contains:

- the bundle schema and renderer version;
- safe source and run identities already validated by the report builder;
- selected topic IDs in official order;
- the relative path, byte size, and SHA-256 of `index.html`;
- the relative path, byte size, and SHA-256 of every topic page;
- total bundle bytes;
- whether RAG output and an evaluation overlay were included; and
- the evaluation manifest SHA-256 when supplied.

The manifest does not copy private text, document identifiers, private source
paths, provider events, credentials, or full environment details.

The CLI success receipt reports the absolute bundle directory and index path,
the ordered topic IDs, bundle-manifest hash, page count, total bytes, RAG and
evaluation inclusion, and report schema version.

## Summary Presentation Model

The renderer uses a dedicated allowlisted `RunSummary` view. It is constructed
from already validated topic views plus an optional validated evaluation
overlay. Raw topic records are never passed to the summary template.

The summary may contain only:

- topic IDs and their official order;
- completion and fallback state;
- final retrieval depth;
- subnarrative and query counts;
- canonical nugget count;
- run-level totals, medians, and ranges derived from those integers;
- explicitly allowlisted score values, names, definitions, availability, and
  aggregate rules from the evaluation manifest;
- short, non-sensitive renderer/model labels needed to interpret scores; and
- relative links to private topic pages.

It excludes narratives, queries, answers, claims, passages, corpus text,
document IDs, citations, source paths, prompts, provider responses, full
digests, secrets, and environment values.

The builder applies two privacy barriers before publication:

1. only the `RunSummary` fields can reach the template; and
2. the rendered summary is scanned with the existing generic private-pattern
   checks and a run-derived denylist of private values.

Any match fails the whole bundle build. Raw topic pages are intentionally not
subject to the summary denylist because they are private trace artifacts; they
retain the existing escaping and safe-serialization checks.

## Summary Information Architecture

The summary answers three questions in order.

### 1. Did the run complete cleanly?

Top cards show:

- completed topics over selected topics;
- fallback count, split by fallback kind when available;
- total submitted documents;
- median and range of final retrieval depth;
- total canonical nuggets; and
- evaluation state.

Nonzero failures or fallbacks use text and iconography as well as color.

### 2. How large and complex was the run?

A compact distribution section shows topic counts and run-level medians/ranges
for depth, subnarratives, queries, and canonical nuggets. It uses semantic
cards and a small accessible table rather than loading a charting library.

### 3. How did it score?

Scores are grouped into separate families:

- retrieval relevance;
- nugget or obligation coverage; and
- answer and citation quality.

Each family shows its authoritative aggregate values, definitions, aggregation
rule, and availability. A metric whose required judgments are absent or
incomplete reads "Unavailable" with the manifest's reason. Missing values are
never rendered as zero. The page does not calculate a composite across
families and does not compare unlike evaluation setups.

The summary finishes with one sortable, filterable row per topic. Stable base
columns are topic ID, health/fallback state, depth, subnarratives, queries,
nuggets, and raw-trace link. Score columns are the union of metrics the
validated overlay actually supplies, grouped under their metric family and
sorted by stable metric name within each family.

The initial order is "needs attention": failures, fallbacks, then official
topic order. Selecting a score column sorts low-to-high for that named metric;
unavailable values remain grouped last. The active sort and metric family are
always visible. There is no implicit or unlabeled "primary score."

Filtering and sorting are progressive enhancements implemented over the
already rendered table. With JavaScript disabled, every row and value remains
readable in official topic order.

## Evaluation Overlay

The first supported score source is the repository's validated
`evaluation_manifest.json` contract used by `friendly_report.py`. The debug
report does not open qrels, gold nuggets, raw RAGDoll tasks, or judge provider
responses directly.

A narrow `EvaluationOverlay` adapter extracts only:

- exact scope and topic order;
- retrieval macro and per-topic metrics plus availability;
- nugget-coverage macro and per-topic metrics plus availability;
- citation/answer macro and per-topic metrics plus availability;
- metric definitions and macro rules; and
- safe evaluation identity required to explain the score source.

The adapter validates the existing evaluation schema before projecting these
fields. It preserves the manifest's numeric precision internally and formats
only at render time. Aggregate values come from the authoritative manifest;
they are not recomputed from rounded table cells.

When the report selects a strict subset of a larger evaluation scope, valid
per-topic cells are retained but the larger-scope macro is shown as unavailable
with a scope-mismatch reason. The adapter neither relabels the full-run macro as
a subset result nor recomputes it from published per-topic cells.

If future official scores use another schema, they require a separate explicit
adapter into the same `EvaluationOverlay` model. The summary renderer does not
gain format-specific branching.

## Raw Topic Pages and Navigation

Each raw page reuses the existing one-topic information hierarchy:

1. Narrative
2. Subnarratives
3. Generated answer, or explicit RAG-not-supplied state
4. Funnel overview
5. New documents
6. Selected documents
7. Top passages
8. Final selected nuggets
9. Final retrieval

No stored row that appears in the legacy topic section is dropped. The only
structural change is the page shell.

A sticky but unobtrusive topic navigation region provides:

- Back to summary;
- Previous topic;
- Next topic; and
- the existing stage-anchor menu.

Previous and next follow the validated official topic order and are absent, not
disabled, at the ends. All links are relative, so the bundle remains portable.
Browser back/forward behavior and fragment navigation require no server.

The page title, top heading, and navigation identify the current topic. Focus
styles, semantic landmarks, table captions, disclosure controls, reduced-motion
support, wide-table scrolling, and mobile behavior retain the existing report
accessibility contract.

## Module Boundaries

`competition_debug_report.py` remains the deep module: callers provide standard
configs, optional topic selectors, an optional evaluation manifest, and an
output destination. The module owns artifact validation, privacy projection,
page rendering, safe publication, and the receipt.

The existing single-file `build_debug_report(...) -> DebugReportReceipt` seam
remains unchanged. A sibling
`build_debug_report_bundle(...) -> DebugReportBundleReceipt` seam owns bundle
publication.

Internally, both paths share:

1. one validated run loader producing the existing typed topic views;
2. one topic-body renderer;
3. separate single-file, summary, and one-topic page shells; and
4. shared escaping, serialization, accessibility, and source validation.

The topic renderer receives one immutable topic view and navigation metadata.
It does not read files. The summary renderer receives only `RunSummary`. The
evaluation adapter receives only the supplied validated evaluation manifest.
These seams prevent summary privacy from depending on template discipline and
prevent score formats from spreading through the raw report renderer.

## Publication and Failure Semantics

The bundle builder:

1. validates retrieval, optional RAG, and optional evaluation inputs once;
2. creates a sibling staging directory with restrictive local permissions;
3. constructs the allowlisted summary model;
4. renders every topic page independently;
5. renders and privacy-scans the summary;
6. validates filenames, internal links, topic coverage, hashes, and sizes;
7. writes `bundle-manifest.json` last;
8. fsyncs the completed files and directory where supported; and
9. atomically renames staging to the previously absent target.

Any error removes only the owned staging directory and leaves source artifacts
and any earlier report untouched. Cleanup resolves and validates the exact
staging path before removal. A target collision, duplicate topic ID, invalid
topic filename, scope mismatch, missing page, broken relative link, privacy
match, or hash disagreement produces a clear error and nonzero exit.

The builder makes no network calls and never starts retrieval, generation, or
evaluation.

## Private Serving

Canonical bundles remain under ignored retrieval output directories. A
sanitized derived copy may be placed under `$HOME/codex-rendered/plans/` only
after the required privacy review and with explicit user authorization for raw
topic traces. The existing private Tailscale Serve portal is the only approved
web delivery path; no Funnel or public listener is added.

Handoff verification must confirm:

- the summary and representative topic URLs return the intended content;
- all topic links stay under the intended bundle path;
- the Serve mapping remains tailnet-only; and
- no unrelated repository paths, secrets, logs, or datasets are exposed.

The summary's privacy-safe design does not make the raw topic pages safe for
public hosting.

## Verification

Implementation follows test-first development.

Unit and contract tests cover:

- backward-compatible legacy single-file behavior;
- mutual exclusion of `--output` and `--output-dir`;
- create-only target and staging cleanup behavior;
- one page per selected topic in official order;
- deterministic topic filenames, HTML bytes, hashes, and receipt fields;
- exact preservation of all legacy topic stages and rows;
- previous, next, summary, and stage links, including first/last topics;
- manifest-last publication and complete hash/size reconciliation;
- summary totals, medians, ranges, fallback counts, and run health;
- evaluation scope and schema rejection;
- dynamic metric families and columns;
- authoritative macro values and visible aggregation definitions;
- unavailable metrics remaining unavailable rather than becoming zero;
- score sorting with unavailable values last and no cross-family composite;
- summary allowlist and run-derived privacy denylist;
- hostile HTML/script strings and safe inline serialization;
- no candidate-ledger reads, network calls, or pipeline reruns; and
- explicit RAG-not-supplied and evaluation-not-supplied states.

An integration fixture builds a multi-topic bundle and validates every file
against `bundle-manifest.json`. A representative real-run build first uses two
topics, then the sealed 119-topic output after the cheap checks pass.

When Playwright is available, checks run at desktop and mobile viewports for:

- summary table filtering and each metric sort direction;
- keyboard navigation and visible focus;
- disclosure and fragment behavior;
- page-level overflow; and
- usable previous/next navigation.

The final 119-topic receipt records summary size, total bundle size, and topic
page size distribution. Completion claims include focused/full test results and
live tailnet-only HTTP verification, or state exactly which verification could
not run and why.

## Expected Outcome

The approximately 808 MB of trace data remains available, but no longer forms
one browser document. Routine use opens a compact run-and-score summary, and a
reader pays the loading cost of a raw trace only for the topic they choose. The
same structure remains useful before evaluation, while scores can be added
later without rebuilding or weakening the retrieval and generation boundaries.

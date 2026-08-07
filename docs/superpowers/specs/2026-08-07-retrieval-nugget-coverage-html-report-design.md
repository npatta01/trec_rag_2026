# Retrieval Nugget Coverage HTML Report Design

**Date:** 2026-08-07  
**Status:** Approved

## Objective

Add a year-neutral, self-contained HTML report for completed retrieval nugget
coverage evaluations. The report must answer, at run and topic level:

1. What did the narrative ask for?
2. What answer obligations did the evaluator expect?
3. Which obligations were full, partial, or unsupported?
4. Which canonical retrieval nuggets supported each judgment, and what remained
   missing?
5. What subnarratives and BM25 queries did retrieval plan?

The report is a private diagnostic derivative. It does not change the evaluator,
make hosted calls, reopen passages, or claim that retrieval subnarratives were
used as coverage evidence.

## Terminology and Interpretation Boundary

- **Narrative** is the exact authenticated information need from the generation
  handoff.
- **Retrieval-plan context** is the subnarrative text and BM25 queries from the
  topic's decomposition checkpoint.
- **Answer obligation** is the planner-derived requirement scored by the
  retrieval nugget coverage evaluator.
- **Canonical retrieval nugget** is the claim-hint text from the authenticated
  generation handoff.
- **Coverage evidence** is only canonical nugget text cited by a coverage
  judgment.

Retrieval-plan context must remain visually and semantically separate from
coverage evidence. The report may help a reader compare the two, but it must not
state or imply that a subnarrative caused, supports, or was evaluated by a
coverage judgment.

## Scope

### Included

- One standalone HTML file containing an overview and deep-linkable detailed
  topic views.
- Run-level aggregation across the selected completed coverage bundles.
- Exact narrative, subnarrative, BM25 query, obligation, gap, and canonical
  nugget text.
- Read-only validation of the authenticated handoff, completed coverage bundle,
  and hash-checked decomposition checkpoint.
- Responsive light and dark themes with a persistent manual override.
- A year-neutral CLI route added to the existing competition debug-report skill.
- Rendering and browser verification against the completed 22-topic 2025
  development evaluation after the generic implementation passes tests.

### Excluded

- New planner or judge calls, retries, searches, reranking, generation, or other
  network access.
- Passage text, document IDs, retrieval scores, provider responses, credentials,
  qrels, organizer gold nuggets, or RAGDoll output.
- Rejudging nugget faithfulness or attributing a low score to retrieval,
  selection, or canonicalization.
- Changes to evaluator schema, prompt, model, scoring, persistence, or CLI
  behavior.
- Public deployment. A privacy-reviewed presentation copy may be exposed only
  through the existing private tailnet portal when explicitly requested.

## Repository and Module Boundaries

The feature is split across two modules so HTML concerns do not deepen the
evaluator's already-large implementation:

- `code/trec_rag/retrieval_nugget_coverage.py`
  - Add a small public, read-only loader for a completed coverage work directory.
  - The loader reuses the evaluator's existing canonical artifact, request
    identity, score, and manifest validators.
  - It returns typed validated data and never publishes or repairs artifacts.
- `code/trec_rag/retrieval_nugget_coverage_report.py`
  - Own report-specific records, decomposition loading, aggregation, HTML/CSS/JS
    rendering, atomic output publication, and CLI parsing.
  - It may import `load_validated_decomposition` from
    `competition_retrieval.py`, but it must not import or load passage/report
    structures from `competition_debug_report.py`.

Tests live in:

- `code/tests/test_retrieval_nugget_coverage.py` for the new completed-bundle
  loader;
- `code/tests/test_retrieval_nugget_coverage_report.py` for report behavior; and
- `code/tests/test_retrieval_nugget_coverage_skill.py` for the executable skill
  route.

Documentation changes stay beside the feature:

- `code/trec_rag/README.md`; and
- `.agents/skills/trec-rag-competition-debug-report/SKILL.md`.

## Inputs and CLI

The CLI is generic and performs no hosted calls:

```bash
.venv/bin/python -m trec_rag.retrieval_nugget_coverage_report \
  --handoff-manifest OUTPUT_DIR/generation_handoff_manifest.json \
  --coverage-root OUTPUT_DIR/retrieval_nugget_coverage_v2 \
  --output OUTPUT_DIR/retrieval_nugget_coverage_report.html
```

Optional repeated `--topic TOPIC_ID` selectors preserve their command-line order.
Without selectors, the renderer discovers topic directories under
`--coverage-root`, retains authenticated handoff order, and includes only
directories containing a complete coverage manifest. At least one topic is
required. Unknown, duplicate, incomplete, unsafe, or contradictory topic state
fails the entire build rather than silently dropping data.

`--output` must end in `.html`, must not resolve through a symbolic link, and is
published atomically. Rebuilding from the same validated inputs produces the
same bytes; the report contains no current timestamp or random identifier.

## Validation and Provenance

### Authenticated handoff

Use `load_generation_handoff` and `select_generation_topics`. Narrative and
ordered claim-hint text come only from this authenticated boundary. The renderer
must not read canonical nugget files directly.

### Completed coverage bundle

The new public evaluator loader takes the authenticated bound input and one work
directory. It must:

1. require canonical `input.json`, `plan.json`, `judgments.json`, `report.json`,
   and manifest-last `manifest.json`;
2. reconstruct the evaluator identity from the manifest while requiring current
   evaluator, planner-prompt, and judge-prompt schema identities;
3. validate input identity, exact planner and judge request digests, frozen plan,
   judgment IDs, supporting nugget IDs, safe provider metadata, report payload,
   artifact hashes, and completed-stage count through the same functions used by
   resume;
4. return an immutable typed record containing the bound input, frozen plan,
   judgments, scored report, identity, and source hashes; and
5. perform no write, repair, backend selection, or hosted call.

### Retrieval-plan context

For each selected topic, load `decomposition/manifest.json` and
`decomposition/result.json` from the handoff manifest's parent output directory.
Require canonical JSON, safe regular files beneath the topic directory, the
expected result filename/byte count/SHA-256 from the checkpoint manifest, and a
valid planner identity object. Then call `load_validated_decomposition` against
the exact authenticated topic ID and narrative. An original-only fallback is
represented explicitly with zero generated subnarratives.

The coverage manifest SHA-256 and decomposition manifest/result SHA-256 values
appear in a collapsed provenance section. The report describes the decomposition
as hash-checked retrieval-plan context, not as part of the authenticated coverage
judgment chain.

## Aggregation

Run-level values are computed locally from the validated topic reports:

- evaluated topic count;
- total canonical nugget count;
- total required and supplemental obligation counts;
- topic-macro required coverage: arithmetic mean of each topic's
  `required_coverage`;
- topic-macro strict-full rate: arithmetic mean of each topic's
  `strict_full_rate`;
- total `full`, `partial`, and `unsupported` label counts; and
- number of topics with perfect required coverage.

The report labels both macro rates explicitly. It never combines required and
supplemental obligations into one unlabeled denominator. Topic rows retain exact
artifact values; percentages are rounded only for display.

## Information Architecture

The single HTML file has two navigation levels.

### Run overview

The landing view contains:

- metric cards for the aggregate values above;
- a concise limitations callout;
- topic search;
- status filters (`all`, `has gaps`, `perfect`, `unsupported`);
- sorting by required coverage, strict-full rate, or topic ID; and
- one compact row per evaluated topic.

Lowest required coverage appears first by default. Selecting a row opens that
topic without expanding every topic in the document.

### Detailed topic view

Exactly one topic detail is visible at a time. It contains, in order:

1. topic ID, required coverage, strict-full rate, and label counts;
2. a compact disclosure containing the full exact narrative;
3. a separately styled **Retrieval plan context — not coverage evidence**
   disclosure containing ordered subnarratives and their nested BM25 queries;
4. ordered facets and answer-obligation rows, each showing requirement, kind,
   label, support test, exact narrative spans, missing elements, and supporting
   nuggets; and
5. a collapsed complete canonical nugget inventory with local aliases and
   mapped/unmapped badges.

The first partial or unsupported obligation is expanded by default; all other
obligations remain compact. A user can expand or collapse any disclosure.
Supporting nuggets show the local `n001`-style alias and exact claim text, not
internal canonical IDs.

## Navigation State

Topic navigation uses a URL fragment such as `#topic=407`. Selecting a topic
updates history with `pushState`; browser back/forward restores the prior view.
An invalid or unavailable topic fragment returns to the overview with a visible
message rather than a blank page. The report works when opened from `file://` and
does not require a web server.

## Theme and Accessibility

All visual values use shared CSS custom properties. No component may hard-code a
light-only surface or text color.

- Default theme: `system`, following `prefers-color-scheme`.
- Manual choices: `Light`, `System`, and `Dark`.
- Persistence: local storage, guarded so blocked storage does not break the
  report.
- Initial paint: an inline pre-render script applies a saved explicit theme
  before visible content to avoid a white flash.
- Print: forced light palette with navigation controls hidden and disclosures
  expanded.
- Status: icon/text labels accompany color; color is never the only signal.
- Interaction: semantic buttons and `<details>/<summary>`, visible focus rings,
  keyboard operation, skip link, and announced empty/filter states.
- Motion: no required animation, and `prefers-reduced-motion` disables optional
  transitions.
- Responsive layout: one reading column on mobile; metric cards and controls
  reflow without horizontal scrolling.

## HTML Safety and Privacy

The renderer emits one self-contained file with no external scripts, fonts,
stylesheets, images, or network requests. All Python-rendered text is HTML
escaped. Embedded JSON additionally escapes `<`, `>`, `&`, U+2028, and U+2029
before entering a script block. The report uses no `innerHTML` with data-derived
content.

Allowed content:

- narrative text;
- retrieval subnarrative and BM25 query text;
- answer obligations, support tests, exact narrative spans, labels, and gaps;
- canonical nugget text and local aliases;
- aggregate metrics, evaluator/model identities, and hashes.

Forbidden content:

- passage or document text;
- document IDs, ranks, scores, or citation IDs;
- provider response bodies or unsafe provider metadata;
- secrets, environment data, qrels, organizer gold nuggets, and RAG output.

The canonical report remains under the ignored private output tree. Any browser
presentation copy must contain only the allowlisted derivative above and remain
on the existing tailnet-only portal.

## Skill Route

Enhance the existing `trec-rag-competition-debug-report` skill rather than
creating another similarly named skill. The route must:

- trigger when a user asks to view, render, browse, summarize, or inspect
  retrieval nugget coverage results;
- distinguish this sanitized coverage report from the raw competition debug
  report;
- show the cache-only renderer command and state that it makes zero hosted
  calls;
- require a privacy review before a presentation copy is served; and
- explain that subnarratives are context, while canonical nugget text is the
  evaluator's evidence representation.

The existing skill test must first fail on the absent executable route, then pass
after the route is added.

## Verification

### Automated

- Completed-bundle loader rejects changed input, plan, judgments, report,
  request identity, manifest identity, artifact hash, unknown nugget ID, missing
  artifact, and symbolic-link input without writing anything.
- Decomposition loader rejects noncanonical JSON, unsafe paths, manifest/result
  disagreement, wrong topic/narrative, malformed planner identity, and malformed
  subnarratives.
- Aggregates use hand-checked literal expectations and distinguish topic-macro
  scores from obligation counts.
- HTML tests assert escaped hostile text, absence of forbidden data, deterministic
  bytes, complete topic/obligation/nugget data, unique IDs, and expected
  navigation/theme/accessibility landmarks.
- CLI tests cover discovery, selectors, duplicates, incomplete topics, output
  extension, atomic publication, and machine-safe errors.
- Skill behavior tests execute the documented route against controlled fixtures
  and verify zero hosted calls.

### Browser

Run headless Chromium on desktop and mobile widths in forced light and forced
dark modes. Verify:

- no console errors or network requests;
- no horizontal overflow;
- overview → topic → back/forward navigation;
- search, filters, sorting, and disclosures;
- keyboard focus visibility;
- manual theme override and system fallback; and
- print stylesheet behavior.

Finally render the real 22-topic 2025 evaluation and reconcile the displayed
aggregate values with the validated JSON artifacts before producing a private
browser handoff.

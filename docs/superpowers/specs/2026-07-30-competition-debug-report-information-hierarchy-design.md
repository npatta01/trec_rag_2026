# Competition Debug Report Information Hierarchy

## Goal

Make each topic tell its story before presenting pipeline diagnostics. A reader
should see the official narrative, the decomposition, and the generated answer
first, then understand how many records survived each retrieval and evidence
stage. Detailed document and scoring records remain available without
dominating the default reading flow.

## Topic Reading Order

Each topic retains its existing one-topic-at-a-time container and stable stage
anchors. Inside that container, stages render in this order:

1. Narrative
2. Subnarratives
3. Generated answer
4. Funnel overview
5. New documents
6. Selected documents
7. Top passages
8. Final selected nuggets
9. Final retrieval

The generated-answer stage keeps the existing `final-rag` anchor for backward
compatibility. When RAG output is absent, its early position shows the existing
explicit “not included” state.

## Story Stages

The narrative shows the official question as the primary content. Its source
seal moves into a collapsed technical disclosure.

Subnarrative cards show only the subnarrative ID and text by default. Literal
BM25 queries and all query hashes move into each card's existing technical
disclosure. This keeps the decomposition readable while retaining exact
retrieval provenance.

The generated answer begins with answer prose and inline citation chips.
Generation provenance, the output receipt, and referenced-document cards move
below the answer in collapsed disclosures. Provider, model, run identity, word
count, validation state, and the sealed-artifact authorship limitation remain
available without preceding the answer.

## Funnel Overview

Add a new `funnel-overview` stage immediately after the generated answer. It is
computed only from the already validated immutable `TopicReport`; it performs
no new artifact reads and does not rerun any pipeline stage.

The overall funnel displays these counts in order:

- facet-only documents discovered: `len(new_documents)`;
- selected documents: `len(selected_documents)`;
- document × subnarrative rankings: `len(passage_rankings)`;
- stored winning passages: the sum of `len(ranking.winning_passages)`;
- selected evidence clusters: `len(evidence_clusters)`;
- final canonical nuggets: `len(canonical_nuggets)`; and
- final retrieval documents: `len(retrieval_output.documents)`.

Facet-only discoveries are visually identified as an input-pool expansion, not
as the previous denominator of selected documents. Likewise, document ×
subnarrative rankings are labeled as pairs rather than unique documents. This
prevents the overview from implying invalid conversion rates between unlike
units.

Below the overall counts, a compact per-subnarrative table shows:

- ranked documents: unique DocIDs among passage rankings for that
  subnarrative;
- stored passages: the sum of stored winning-passage records for those
  rankings;
- evidence clusters: the count belonging to that subnarrative; and
- final nuggets: the count belonging to that subnarrative.

Rows follow sealed decomposition order. Counts are integers derived from the
stored records; logits and facet-local ranks are not aggregated or compared.
For an original-only fallback with no generated subnarratives, the table is
replaced by a clear fallback message while the overall counts remain visible.

## Compact Selected Documents

The selected-document section retains stored selection order but no longer
opens every explanation card by default.

Each selected document becomes a one-line disclosure summary containing:

- selection rank and DocID;
- original/facet status; and
- selected lane and lane rank.

Opening the document reveals the trace-derived selection reason,
representative stored passage, membership coverage, and technical provenance.
The representative-passage caveat and deterministic selection rule remain at
section level.

The first ten compact document disclosures render directly. Documents 11
through the selected-pool depth remain complete inside one “Show remaining”
disclosure. This reduces default page length without truncating or dropping any
record.

## Presentation and Accessibility

The funnel uses a responsive sequence of count cards and a narrow comparison
table. On mobile, count cards wrap into one or two columns and the table remains
inside its own horizontal scroll container without causing page-level
overflow. Count labels name their unit explicitly.

All new disclosures retain native markers and 44-pixel minimum targets.
Keyboard behavior, topic fragments, citation fragments, light/dark color
schemes, source-string escaping, and the dependency-free standalone document
remain unchanged.

## Verification

Test-first contracts cover:

- story-stage order before funnel diagnostics;
- answer prose appearing before provenance and references;
- exact overall funnel counts from a controlled fixture;
- per-subnarrative ranked-document, passage, cluster, and nugget counts in
  decomposition order;
- explicit fallback rendering without subnarratives;
- concise subnarrative defaults with query details preserved;
- ten visible compact selected-document summaries plus a complete remainder;
- expanded selected-document evidence and technical records remaining intact;
- hostile-string escaping, stable anchors, citation navigation, native
  disclosure controls, and mobile overflow safety.

After focused and full tests, regenerate the existing sealed two-topic report,
run real Chrome desktop/mobile checks, and refresh the already-authorized
tailnet-only rendered copy only after verification passes.

## Scope Boundaries

This changes only post-run HTML projection and presentation. It does not alter
retrieval, generation, artifact schemas, organizer outputs, submission files,
or the organizer repository. It does not add public hosting or external
runtime dependencies.

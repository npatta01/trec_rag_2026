# Competition Debug Report UI Redesign

## Goal

Make the post-run competition report useful for understanding a run rather
than presenting every stored record as one long collection of tables. Preserve
the report's sealed-artifact validation, private self-contained delivery, and
complete diagnostic data while showing one topic at a time and foregrounding
human-readable explanations.

## Interaction Model

The report remains one standalone HTML file. Each topic is a native
`<details>` disclosure in one named group, so opening a topic closes the other
topics even without JavaScript. The first topic is open initially.

Bundled inline JavaScript progressively enhances the topic controls into a
tab-like switcher. It keeps exactly one topic open, updates the URL fragment,
supports left/right arrow keys, and restores a directly linked topic on load.
Bundled inline CSS supplies the visual system and responsive layout. No CDN,
font download, image request, or external runtime is permitted. With
JavaScript disabled, the native disclosures remain fully usable.

Run configuration paths, artifact receipts, and low-level validation details
move into a collapsed "Validation details" disclosure above the topics.

## Selected Documents

Replace the wide selected-document table with ordered document cards. Each
card shows:

- selection rank and DocID;
- a plain-language reason derived only from the sealed selection trace;
- original/facet status, selected lane, lane rank, and memberships;
- the best stored passage for that document;
- the passage's subnarrative and within-subnarrative aggregate rank; and
- a collapsed technical-provenance section for hashes, scores, and complete
  membership/rationale details.

"Best stored passage" means the passage with the lowest stored aggregate rank
among all subnarratives for that selected document. Ties follow decomposition
order. The report does not compare logits across subnarratives or invent a
semantic explanation. If no passage exists, the card says so explicitly.

## RAG Output

Replace the answer table with an article-like answer view. Each stored answer
item becomes a numbered paragraph. Citations render as compact inline chips at
the end of the paragraph and link to reference cards below the answer. Each
reference card shows its citation index, DocID, and a bounded excerpt; technical
document details remain collapsed.

A compact provenance banner distinguishes implementation from generation:

- implementation: `trec_rag.competition_rag`;
- run ID and run description;
- configured provider and model; and
- validation state and answer word count.

For the current report, this will make clear that Tanwir authored PR #24's RAG
implementation while the displayed output was generated locally through
OpenRouter using `openai/gpt-5.6-sol`. The generic renderer must derive the
provider/model/run fields from the standard RAG config and output metadata; it
must not hard-code a person's name.

## Other Stages

Narrative and subnarratives use readable text cards. New documents remain
grouped by first-seen lane. Passage rankings and canonical nuggets keep their
complete stored records behind progressive disclosures. Final retrieval stays
compact, with detailed rows collapsed. Diagnostic tables are retained only
where column comparison is materially useful.

## Responsive and Accessible Presentation

- One-column cards on mobile and a wider reading column on desktop.
- No page-level horizontal overflow.
- Minimum 44-pixel interactive targets.
- Visible keyboard focus, semantic headings, labels, and live topic state.
- Topic controls support mouse, touch, Enter/Space, and arrow keys.
- Citation links have descriptive accessible names.
- Dark and light color schemes remain supported.
- Reduced-motion preferences disable nonessential transitions.
- Source-derived strings are escaped before rendering and never interpolated
  into executable JavaScript.

## Verification

Add test-first contracts for:

- exactly one initially visible topic and native no-JavaScript fallback;
- enhanced topic switching, fragment restoration, and keyboard semantics;
- deterministic selected-document reason and best-passage selection;
- card rendering without the old selected-document table;
- paragraph-based RAG rendering with citation chips and reference targets;
- model/provider/run provenance from the standard RAG config;
- hostile-string escaping and absence of external dependencies; and
- responsive markup with no wide RAG/selected-document tables.

Run focused tests, the full suite, the real two-topic post-run CLI, and Chrome
headless desktop/mobile rendering. Verify both topics can be selected, only one
is visible at a time, citation links resolve, the private report remains
self-contained, and organizer/checkpoint hashes remain unchanged. Replace the
authorized tailnet-only rendered copy only after these checks pass.

## Scope Boundaries

This redesign changes only post-run report data projection and presentation.
It does not rerun or alter retrieval, reranking, canonicalization, RAG
generation, organizer outputs, or the competition submission formats. It does
not add external web dependencies or make the private report public.

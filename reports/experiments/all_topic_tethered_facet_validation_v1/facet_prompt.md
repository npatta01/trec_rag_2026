# All-topic tethered-facet rendering contract v1

## Purpose and boundary

Render retrieval facets from one supplied development-topic narrative. This is
an offline, qrels-blind planning task. Do not inspect judgments, retrieved
documents, candidate answers, model output, or web sources. A facet is a search
query for one explicit informational obligation in the narrative; it is not an
answer or a hypothesis about the answer.

The experiment-specific authorized topic order is:

`14, 31, 37, 58, 72, 84, 144, 161, 200, 213, 219, 224, 225, 233, 273, 300, 407, 477, 499, 515, 707, 897`

This authorization does not change the repository's shared protected-topic
constants. Planning must fsync the authorization receipt and a separate,
read-only authorization pre-seal before opening the topic source; the final
seal binds both. Qrels fields are forbidden anywhere in the planning manifest.

## Rendering rules

For each topic:

1. Enumerate each explicit informational obligation once. Do not invent facts,
   causes, outcomes, examples, dates, populations, or narrower subtopics.
2. Produce three to nine facets. Fewer than three is allowed only when the
   narrative itself contains fewer than three explicit obligations.
3. Give each facet a deterministic topic-local obligation ID and a globally
   unique facet ID. Preserve authorized topic order and narrative obligation
   order; never reorder by expected usefulness.
4. Write a concise query containing all three tethers:
   - **subject anchor:** the entity or concept being investigated;
   - **population/domain:** the applicable people, place, field, or setting;
   - **relation:** the requested impact, cause, comparison, definition,
     recommendation, count, or other explicit relation.
5. Reject a naked abstract query such as `impact`, `causes`, `definition`, or
   `responsibility`. A query must remain interpretable without neighboring
   facets.
6. Record the exact output of the frozen `lowercase-alphanumeric-v1` analyzer
   and SHA-256 of the UTF-8 query. Analyzer output, query text, and hash must
   agree exactly.

Historical facet wording may be reused only when it passes every rule above.
Otherwise, rerender the obligation from the narrative.

## Bridge terms

A term absent from the narrative is allowed only when it is a conventional
name needed for one of these purposes:

- `domain_disambiguation`
- `population_binding`
- `relation_paraphrasing`
- `standard_concept_name`

Every bridge term must record:

- `surface`: exact text added to the query;
- `source`: where the conventional wording came from;
- `purpose`: one allowed value above;
- `scope_rationale`: why it stays within the narrative's request;
- `analyzer_output`: exact analyzer terms for the surface;
- `not_candidate_answer_rationale`: why it is not a possible answer.

Reject bridge terms that supply a candidate answer, factual claim, cause,
outcome, example, date, or narrower subtopic. Prefer no bridge term when the
narrative already provides a clear tether.

Every analyzed bridge surface must be nonempty, occur exactly in the query, and
cover every query term that is not supported by the narrative or obligation.
Only ordinary analyzer glue words and conservative morphological variants are
allowed without a bridge record. Anchor, domain, and relation tethers must each
have nonempty analyzer output and be supported by the narrative or obligation.

## Required facet fields

Each facet record contains only:

`topic_id`, `facet_id`, `obligation_id`, `obligation`, `query`,
`anchor_terms`, `domain_terms`, `relation_terms`, `analyzer_terms`,
`bridge_terms`, `manifest_order`, and `query_sha256`.

The validator checks exact scope, uniqueness, deterministic order, query
identity, analyzer identity, all three tethers, complete bridge provenance,
facet-count bounds, and absence of qrels fields before a request plan can be
created.

## Retrieval planning invariants

- Reuse the exact original-narrative cache at depth 1,000 for all 22 topics.
- Supply an approved cache root and authenticate each existing regular cache
  file under it from actual bytes: validate its 64-hex SHA-256, recomputed
  request key, topic/query/variant/depth identity, raw-response provenance, and
  exact 1,000-candidate schema. Reject missing, tampered, or escaping paths.
- Schedule zero original-narrative requests.
- Schedule each accepted facet exactly once at depth 200.
- Freeze the exact facet request count before retrieval.
- Any later retrieval must use the exact-identity cache, save raw responses
  first, and start requests no more frequently than once every 3.0 seconds.
- An immutable failed attempt must not be retried automatically.

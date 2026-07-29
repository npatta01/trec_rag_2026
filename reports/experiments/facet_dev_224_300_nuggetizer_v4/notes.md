# Facet development Topics 224 and 300

## Result

This live run completed planning, BM25 retrieval, local Mixedbread reranking,
extractive evidence, and canonicalization for both topics. It did not complete
the final organizer export at real scale. The tracked record is therefore
evidence for experimental mechanics, retrieval behavior, and a bounded semantic
audit—not a production-readiness claim.

At one fixed total budget, original-only retrieval was stronger at 100
documents, while original-plus-subnarrative round-robin was stronger at 500 and
1,000 documents for both topics under all three qrel judges. The development
qrels are pooled LLM judgments and included original-query runs, not these new
generated lanes, so unjudged documents remain unknown and near-top comparisons
favor the original query.

## Flow and scale

| Topic | Subnarratives | Selected documents | Exact candidates | Candidate bytes | Selected evidence | Claims | Canonical states |
|---:|---:|---:|---:|---:|---:|---:|---|
| 224 | 5 | 100 | 47,090 | 5.30 GB | 200 | 90 | 5/5 complete |
| 300 | 4 | 100 | 70,772 | 8.99 GB | 160 | 75 | 4/4 complete |
| **Total** | **9** | **200** | **117,862** | **14.29 GB** | **360** | **165** | **9/9 complete** |

All 165 claim IDs were unique. All 196 evidence links matched the selected
source text, document ID, and document hash. Every claim used one to three
supporting evidence records, and neither topic used extractive fallback. The
nine canonical calls recorded 18,739 prompt tokens, 4,527 completion tokens,
and USD 0.004341233 total cost. Planning-call cost was not retained.

Both canonical topic checkpoints completed after roughly 26 minutes. Export
then spent more than 48 additional minutes reloading and revalidating the 14.29
GB candidate ledgers and reached approximately 16.6 GB observed RSS. The run was
interrupted before manifest-last publication, leaving no partial root export.
Issue #25 tracks streaming selected-candidate validation and reusable sentence
boundaries.

## Semantic audit method

The audit reviewed all nine subnarrative definitions and scanned all 165 claim
texts. For each subnarrative it inspected the first, middle, and last stored
claim with every cited evidence record: 27 claim/evidence pairs total.
Structural checks covered all 196 evidence links. Organizer nuggets and qrels
were not consulted during this semantic review.

- **Strong support:** the sealed evidence directly supports the claim without
  an unresolved antecedent or unstated topic assumption.
- **Atomic:** the claim expresses one main fact or a tightly coherent group.
- **Useful:** the claim is supported, specific, in scope, and not merely a
  redundant or vacuous restatement.

| Quality check | Result |
|---|---:|
| Strong, self-contained support | 23/27 |
| Reasonably atomic | 25/27 |
| Specific, in-scope, useful, and nonredundant | 16/27 |

Topic 224's challenges lane was strongest. Its policy lane admitted generic or
meta material that did not explain how policy is shaped, and its worker-options
lane sometimes described actions available to supporters rather than migrant
workers.

Topic 300's decomposition covered the narrative, but its umbrella-strategy and
global-measures lanes overlapped. In the Antarctica lane, only 9 of 20 claims
or their evidence explicitly referenced Antarctica, the Southern Ocean,
glaciers, ice sheets, or refugia. The audit also found one direction-changing
rewrite and one malformed thesis/proposal fragment.

Atomicity was not the principal limitation. Subnarrative-specific admission,
context-aware entailment, malformed-fragment filtering, and semantic
deduplication are the next quality work in issue #26.

## Reproduction and provenance

`config.yaml` is the sanitized live configuration. `metrics.json` contains only
aggregate values and no document text. `recall-table.md` exposes every recall
value cited by the findings report. `manifest.yaml` pins the run revision,
qrel hashes, compact local-evaluation hashes, and sealed decomposition/canonical
hashes. Large checkpoints, raw documents, and raw model responses remain
ignored and are not required to inspect the stated aggregate results.

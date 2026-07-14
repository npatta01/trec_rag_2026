# Facet-aware fusion pilot v1: frozen manifest

This directory contains the qrels-blind planning input for the held-out
facet-aware fusion pilot. `manifest.json` freezes four topics in independent
selection-hash order and 24 narrative-derived facet queries:

| Topic | Facets | Narrative obligations |
| --- | ---: | --- |
| 233 | 3 | positive impact, negative impact, depression contribution |
| 273 | 7 | poverty perception, resource distribution, historical events, country economic changes, Morocco location, Cameroon resource, continent capacity |
| 161 | 7 | arguments/views, laws, Rapture beliefs, political ideology, historical change, current options/pill, rights-group priorities |
| 14 | 7 | compensation, inclusion, cultural influence, business, equipment, training, mindset |

## Provenance and firewall

- Selection sorts every eligible topic by
  `SHA256("rag25_facet_aware_fusion_v1" || topic_id)` and takes the first four:
  `233`, `273`, `161`, `14`.
- Protected topics `144`, `213`, `224`, `407`, `515` and prior-pilot topics
  `200`, `225`, `707`, `897` are rejected by the validator.
- Each selected narrative is bound to the exact query text and SHA-256 of its
  existing `__original__` retrieval-cache file. Cache filenames are recorded as
  basenames so the manifest is path-independent.
- Facet terms are taken directly from the selected narratives. No bridge term
  was necessary; every `bridge_terms` array is therefore empty. If bridge terms
  are added in a future experiment, each record must carry an allowed purpose
  and a non-empty rationale.
- The frozen analyzer fingerprint is recorded under
  `hashes.analyzer_fingerprint_sha256`. Canonical hashes also cover topic
  selection, source-topic records, facets, and the full unhashed payload.
- `qrels_opened` is fixed to `false`. This task neither reads nor accepts a
  qrels path.

## Reproduction

Create the manifest once from the original-cache root:

```bash
.venv/bin/python -m trec_rag.facet_aware_fusion_manifest create \
  --cache-root /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote \
  --output "/tmp/facet_aware_fusion_manifest.$$.json"
```

The command uses exclusive creation and refuses to overwrite an existing
manifest. `load_manifest(path, cache_root=...)` accepts only the canonical
sorted, indented JSON encoding and revalidates all frozen content, hashes, and
recorded original-cache sources against the explicit cache root.

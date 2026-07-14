# Deep facet candidate pilot v1

This directory contains the durable, qrels-blind definition and final report for
the four-topic candidate-retrieval pilot. Live retrieval responses, MiniLM score
caches, and intermediate rankings stay under the untracked output directory:

`outputs/rag25_deep_facet_candidates_v1/`

## Frozen boundary

- Topics, in order: `219`, `72`, `300`, `84`
- Original streams: four authenticated cached queries at depth 1,000
- Facet streams: 25 exact source-tethered queries at depth 200
- Protected and previously qrels-exposed topics are rejected
- Retrieval limit: one request start every three seconds; no retries
- Local scorer: the already materialized MiniLM revision named in the design
- Qrels stay closed until candidates, gates, scores, rankings, and hashes are sealed

The source-backed design is in
`docs/superpowers/specs/2026-07-13-deep-facet-candidate-ranking-design.md`.
The executable plan is in
`docs/superpowers/plans/2026-07-13-deep-facet-candidate-ranking.md`.

## Rebuild the manifest

```bash
.venv/bin/python -m trec_rag.deep_facet_candidate_manifest create \
  --cache-root /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote \
  --output reports/experiments/deep_facet_candidate_pilot_v1/manifest.json
```

Creation is intentionally create-only. Remove or rename an obsolete local copy
only after verifying its provenance; the command will not overwrite evidence.

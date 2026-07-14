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

## Result

- Facet retrieval added 177 grade-2-or-higher documents beyond the four
  original top-1,000 pools, with additions on every topic.
- The qrels-blind stream gate lost two relevant documents.
- RRF had the best aggregate nDCG@10 (`0.4326`). DUAL retained much more novel
  evidence at 1,000 (`155/177`) but reduced nDCG@10 to `0.3344`.
- The preregistered decision is **stop; diagnose fusion**. This is not evidence
  against facet decomposition or the accepted candidate union.
- The advisor's fixed post-qrels cascade preserved the exact RRF top 10 and
  retained `155/177` novel relevant documents by rank 1,000, but graded
  Recall@500 was only `0.1870`, below RRF (`0.2101`) and GLOBAL (`0.1979`).
- The cascade therefore stops. Its ranking was sealed before diagnostic metrics,
  and it made zero new retrieval or inference calls. These four exposed topics
  provide mechanism evidence only, not confirmation or production promotion.
- The independently reviewed next experiment is a separately approval-gated,
  protected-head Mixedbread rerank over the frozen disagreement pool—not another
  cutoff or coefficient adjustment on these topics.

Open `report.html` for the rendered, source-backed technical report. Its canonical
input is `artifact.json`; `report_data.sqlite` contains the exact bounded datasets
and executable SQLite queries used by its native cards, charts, and tables.

## Rebuild the manifest

```bash
.venv/bin/python -m trec_rag.deep_facet_candidate_manifest create \
  --cache-root /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote \
  --output reports/experiments/deep_facet_candidate_pilot_v1/manifest.json
```

Creation is intentionally create-only. Remove or rename an obsolete local copy
only after verifying its provenance; the command will not overwrite evidence.

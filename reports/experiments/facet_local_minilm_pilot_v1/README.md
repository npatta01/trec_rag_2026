# Facet-local MiniLM pilot v1

This directory freezes the source boundary for the B-first facet-local MiniLM
pilot. `manifest.json` names the exact 31 retrieval streams inherited from the
pre-qrels R1 population: four original-query streams, five retained prompt-lab
facets, and 22 repaired R1 facets across topics 200, 225, 707, and 897.

The create-only source snapshot is generated outside git at
`outputs/rag25_facet_local_minilm_v1/source_v1/`. Its `candidates.jsonl` contains
3,100 canonical candidate rows with full query and document text, their SHA-256
hashes, source scores, ranks, and document IDs. `source_receipt.json` binds the
snapshot schema, row count, byte count, SHA-256, the exact R1 manifest, the prior
freeze, and both verified source ledgers. The durable manifest repeats those
bindings and records each stream's ledger/cache path and request, response,
source-candidate, and copied-candidate hashes.

Downstream pilot code must read only the candidate snapshot plus its receipt; it
must not reopen either retrieval ledger. The freeze step performs no retrieval,
network call, model inference, or qrels access.

## Technical report

`artifact.json` is the canonical, source-backed Data Analytics report input and
`report.html` is its generated self-contained reader. Both files are published
create-only, so an existing report is never overwritten. The report is built only
from authenticated saved artifacts: the v2 tokenizer-only preflight, completed
local scoring and benchmark receipts, ranking freeze, v3 blinded-review freeze,
the approval-scoped qrels-consumption registry, the one-time evaluation outputs,
and `derived_v2/representative_provenance_v2.json`. The latter is an offline,
self-hashed derivation that binds each saved representative to its qualifying
facet ranks and selected MiniLM window. It does not reopen the qrels projection,
candidate snapshot, retrieval ledgers, ranking rows, private review mappings, or
inference cache.

Build the create-only artifact from the repository root:

```bash
.venv/bin/python -m trec_rag.build_facet_local_minilm_report \
  --manifest reports/experiments/facet_local_minilm_pilot_v1/manifest.json \
  --preflight outputs/rag25_facet_local_minilm_v1/preflight_v2 \
  --scoring outputs/rag25_facet_local_minilm_v1/full_scoring_v1 \
  --freeze outputs/rag25_facet_local_minilm_v1/freeze_v1 \
  --review outputs/rag25_facet_local_minilm_v1/review_v3 \
  --evaluation outputs/rag25_facet_local_minilm_v1/evaluation_v1 \
  --output reports/experiments/facet_local_minilm_pilot_v1/artifact.json \
  --html-output reports/experiments/facet_local_minilm_pilot_v1/report.html \
  --renderer /home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs
```

The builder invokes that pinned portable renderer into a private temporary file,
verifies the package, then atomically publishes the HTML without replacement.
The stable consumption registry lives beside the approval as
`outputs/rag25_facet_local_minilm_v1/approvals/qrels_access_v1.json.consumed.json`;
unlike the legacy output-local receipt, it blocks reuse of that approval with a
different output directory. Existing completed evaluations can adopt the guard
offline with `adopt_qrels_consumption_registry`, and representative v2 can be
rebuilt offline with `derive_representative_provenance_v2`; both helpers are in
`trec_rag.facet_local_minilm_evaluate` and neither reads qrels.
The adoption CLI is `code/tools/adopt_facet_local_minilm_qrels_consumption.py`;
it accepts only the saved approval, saved access receipt, and completed
evaluation directory, with no qrels argument.

The report's decision is `B_filters_but_fusion_blocks`: MiniLM improves blinded
facet relevance and promotes relevant candidates before fusion, but the current
family-balanced RRF retains no novel relevant document versus corrected C0.
This is a four-topic descriptive pilot; Stage A was neither permitted nor run.

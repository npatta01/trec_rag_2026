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

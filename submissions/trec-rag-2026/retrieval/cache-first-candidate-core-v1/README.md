# TREC RAG 2026 cache-first retrieval runs

This private repository bundle contains three organizer-facing Retrieval (`R`)
run files. Each run uses the same variable-depth candidate set for every
narrative and changes only the ordering. All three exact files were uploaded
and accepted by Evalbase.

## Accepted files

These `r_output_trec_rag_2026.tsv` files were submitted separately:

| Suggested priority | Variant | Run ID | File |
|---:|---|---|---|
| 1 | Narrative + subnarrative | `r26-narr-facet-v1` | `combo/r_output_trec_rag_2026.tsv` |
| 2 | Subnarrative evidence breadth | `r26-facet-breadth-v1` | `breadth/r_output_trec_rag_2026.tsv` |
| 3 | Narrative only | `r26-narrative-v1` | `narrative/r_output_trec_rag_2026.tsv` |

The table preserves the submitted priority order. It was a recommendation, not
an organizer rule.

Start with the repository-wide [`submission ledger`](../../SUBMISSION_LEDGER.md).
It links the exact accepted file, recorded SHA-256, and archived Evalbase
response for each run. The [final architecture report](../../../../reports/2026-competition-architecture.html)
places this bundle beside the RAG branch from the shared frozen source.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the end-to-end design, formulas,
runtime-model classification, and system boundaries.

## Shared candidate cutoff

For each authenticated source lane `u`, the existing aggregate document score
is `A_u(d)`, its median is `m_u`, and its median absolute deviation is
`MAD_u`. A document is admitted by a lane when:

```text
MAD_u > 0: A_u(d) >= m_u + 2.5 * 1.4826 * MAD_u
MAD_u = 0: A_u(d) > m_u
```

The final candidate set is the union of the original-narrative admissions and
all authenticated `facet:<subnarrative>:text` admissions. If that union is
empty, the system selects exactly the strongest original-narrative document.
Each facet-text lane is treated as one pooled subnarrative source; planned BM25
query strings are not treated as separate pools because the source artifact
does not contain separate result lanes for them.

## Ranking variants

- **Narrative:** narrative-score percentile, then retrieval rank and document
  ID for deterministic ties.
- **Narrative + subnarrative:** `0.5 * narrative percentile + 0.5 * facet
  score`, where facet score is `0.7 * best subnarrative percentile + 0.3 *
  second-best subnarrative percentile` (or the only percentile when there is
  one).
- **Evidence breadth:** number of supported subnarratives, then number of
  strong supporting passages, then the narrative+subnarrative score. Passage
  overlap is suppressed within each document/subnarrative and at most three
  passages are retained before applying the same robust raw-score threshold.

## Organizer-format validation

All three files:

- contain all 119 official narrative IDs exactly;
- use the required six whitespace-separated fields
  `topic_id Q0 docid rank score run_id`;
- use ClimbMix `shard_*` document IDs;
- restart dense ranks at 1 and keep scores non-increasing per narrative;
- contain no duplicate document within a narrative;
- use narrative-specific depths without padding or fixed truncation.

Each run contains 4,246 rows. Depth ranges from 1 to 121 documents, with 61
distinct values of `k` and no fallback topics. The document set for a given
narrative is identical across the three runs.

See `metadata.json` for the compact provenance and Evalbase answer record, and
`retrieval-baseline-runs-manifest.json` for authenticated per-topic cutoff statistics,
source-pool hashes, matrix hashes, scorer identity, and run-file receipts.

The successful remote publication was produced at source commit
`22b9feef18dddb6111beedb35b8c021516fab552`. Commit
`a2f5a5ffeebc6938dd946b2850876e3fc8f32903` subsequently made floating-point
aggregation independent of Python 3.11/3.12 evaluation details. Commit
`c8caca97060664ad323e89b89375f3a75aae9ee1` shortened the combo run tag to the
20-character Evalbase limit. The submission files and run manifest were
then regenerated from the authenticated final matrices without model calls.

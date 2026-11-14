# Run archive and Luna planner outputs

The run artifacts are too large for git. They live in a Hugging Face bucket,
and this page is the map to them. Nothing here is needed to read the reports or
the paper — it is for reproducing or auditing a run.

## The backup

The [Hugging Face backup](https://huggingface.co/buckets/Npatta01/trec-rag-2026-artifacts/tree/trec_rag_2026/backups/full-2026-11-09-v1)
contains the seven run archives: the Retrieval traces, the exact evidence
handoff for both RAG runs, the saved multi-stage work, and both RAGDoll
evaluations. The [restore guide](https://huggingface.co/buckets/Npatta01/trec-rag-2026-artifacts/resolve/trec_rag_2026/backups/full-2026-11-09-v1/RESTORE.md)
gives the download, checksum, and extraction commands.

## Luna planner outputs

Download [rag-multistage.tar.zst](https://huggingface.co/buckets/Npatta01/trec-rag-2026-artifacts/resolve/trec_rag_2026/backups/full-2026-11-09-v1/rag-multistage.tar.zst).
Inside the archive, topic directories are under
`outputs/rag26-ms1-multistage-final/work/bounded_revision/<topic-directory>/`.

- `receipts/planner.json` — the planner call record, including the raw response
  and the accepted payload when present. There are 119 receipts.
- `blueprint.state.json` — the validated plan used by later stages. There are
  112 saved blueprints.

List the planner receipts, or extract the receipt for topic `rag2026-0`:

```bash
tar --zstd -tf rag-multistage.tar.zst | grep '/receipts/planner\.json$'
```

```bash
tar --zstd -xOf rag-multistage.tar.zst \
  outputs/rag26-ms1-multistage-final/work/bounded_revision/rag2026-0-57748010/receipts/planner.json
```

## Per-topic traces

The [release bundle](https://github.com/npatta01/trec_rag_2026/releases/tag/trec-rag-2026-artifacts-v1)
carries the per-topic trace CSV behind the
[Retrieval Quality Analysis](https://npatta01.github.io/trec_rag_2026/reports/2026-retrieval-nugget-coverage.html):
every subnarrative, query and nugget across all 119 topics.

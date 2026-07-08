# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## Remote Pyserini Helpers

Stable notebook imports come from `remote_pyserini.py`. Implementation is split
by responsibility:

- `repo_env.py` loads `.env` / `.env.local` from the worktree or shared checkout.
- `env_config.py` holds typed environment helpers.
- `remote_config.py` builds `RemotePyseriniConfig`.
- `remote_client.py` sends search requests and normalizes candidate rows.

Inputs:

- `.env.example` includes the public hosted endpoint defaults; copy it to a
  local `.env` and fill in `PYSERINI_API_TOKEN`
- endpoint environment: set `INDEX_URL`; or set `PYSERINI_INDEX` and
  `PYSERINI_BASE_URL`; or use the aliases `DEFAULT_INDEX` and
  `DEFAULT_BASE_URL`
- other environment variables: `PYSERINI_API_TOKEN`, `EXTERNAL_PYSERINI_HITS`,
  and `SAMPLE_QUERIES`
- `.env` / `.env.local` files in the active worktree or shared checkout

Outputs:

- typed `RemotePyseriniConfig`
- authenticated remote Pyserini search requests
- normalized candidate rows with `rank`, `docid`, `score`, `text`, and
  `text_length`

Default query:

- `rag25-topics-dev.tsv` topic `31`, an e-waste / recycling development topic.
  Override it with `SAMPLE_QUERIES`, separated by semicolons.

Run validation:

```bash
PYTHONPATH=code uv run --with pytest pytest code/tests/test_remote_pyserini.py -q
```

## Config-Driven RAG Pipeline

The pipeline is the preferred path for experiments. It keeps query
understanding, retrieval, ranking, evidence selection, generation, and
evaluation as separate stages. V1 ships one runnable configuration:
`configs/rag25_bm25_full_query_v1.yaml`.

Run:

```bash
PYTHONPATH=code uv run --with pyyaml python -m trec_rag.pipeline \
  --config configs/rag25_bm25_full_query_v1.yaml
```

The experiment ID is the run identity. If `experiment.output_dir` is omitted,
stage outputs are written to `outputs/<experiment.id>/`. When running from a
linked worktree, remote retriever cache files are stored under the shared
checkout root instead of the worktree:
`<shared-checkout>/outputs/<experiment.id>/cache/`.

Topic parsing follows `topics.format` in the YAML (`tsv` or `jsonl`), not the
filename suffix. For remote Pyserini runs, a retriever `index` in YAML takes
precedence over `PYSERINI_INDEX`; if `INDEX_URL` is set, it must point at the
same index declared in YAML.

Current V1 stages:

- `original_topic` query understanding: use the original narrative/prompt text.
- `pyserini_remote` retriever: run remote BM25 over ClimbMix. Set
  `cache: true` to read and write request-keyed raw response caches, or
  `cache: false` to always call the remote endpoint.
- `passthrough` ranking: accept exactly one retrieval stream and deduplicate by
  best rank.
- `top_k` evidence selection: choose text-bearing ranked candidates.
- `placeholder` generation: write valid cited RAG JSONL for plumbing checks.
- `dev_projected_qrels` evaluation: compute development diagnostics such as
  `ndcg@10`, `recall@100`, `graded_recall@100`, and
  `ideal_dcg_coverage@100`.

Outputs:

- `r_output_trec_rag_2026.tsv`
- `rag_output_trec_rag_2026.jsonl`
- `retrieval_metrics.json`
- `stage_queries.jsonl`
- `stage_retrieved.jsonl`
- `stage_ranked.jsonl`
- `stage_evidence.jsonl`

Shared cache:

- `<repo-root-or-shared-checkout>/outputs/<experiment.id>/cache/`

Run all Python tests:

```bash
PYTHONPATH=code uv run --with pytest --with pyyaml --with semantic-text-splitter pytest code/tests -q
```

## Chunking Helpers

Stable chunking contracts live in `chunking.py`. The public API is intentionally
small:

- `ChunkingConfig`: backend-neutral size, overlap, and trim settings.
- `TextChunk`: stable output record with `document_id`, `chunk_id`, text, and
  character offsets.
- `TextChunker`: protocol that rerankers and evidence selectors should depend
  on.
- `SemanticTextChunker`: default adapter backed by `semantic-text-splitter`.

Inputs:

- raw document text
- a stable `document_id`, usually the ClimbMix `docid`

Outputs:

- ordered `TextChunk` records with IDs like `shard_1_2:0000`

Run validation:

```bash
PYTHONPATH=code uv run --with pytest --with semantic-text-splitter \
  pytest code/tests/test_chunking.py -q
```

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

## BM25 Retrieval Baseline

The first reusable baseline is topic-text BM25 retrieval over the hosted
ClimbMix Pyserini index. It combines `title` and `narrative` for official-style
JSONL topics, uses the full text from local development TSV topics, and writes
the TREC retrieval runfile format:

```text
topic_id Q0 docid rank score run_id
```

Inputs:

- official-style JSONL topics with `id`, `title`, and `narrative`
- local development TSV topics shaped as `qid<TAB>text`; the full text is used
  for retrieval, while a short derived title is kept for normalized records
- `.env` / `.env.local` settings for `INDEX_URL` or `DEFAULT_BASE_URL` plus
  `DEFAULT_INDEX`, and `PYSERINI_API_TOKEN` when the endpoint requires it

Run:

```bash
PYTHONPATH=code python -m trec_rag.baselines.bm25_retrieval \
  --topics path/to/topics.jsonl \
  --output outputs/baseline/r_output_trec_rag_2026.tsv \
  --cache-dir outputs/baseline/cache \
  --hits 100
```

Outputs:

- `outputs/baseline/r_output_trec_rag_2026.tsv`: retrieval runfile
- `outputs/baseline/cache/<topic_id>.json`: raw hosted Pyserini response plus
  the query text used for that topic

The runner validates the generated runfile before exiting. Validation checks
for six-column TREC rows, numeric ranks and scores, one or more rows for every
input topic, duplicate document IDs within a topic, and contiguous ranks from
`1`.

Run all Python tests:

```bash
PYTHONPATH=code uv run --with pytest pytest code/tests -q
```

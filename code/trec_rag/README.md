# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## `remote_pyserini.py`

Inputs:

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

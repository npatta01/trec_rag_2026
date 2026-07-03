# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## `remote_pyserini.py`

Inputs:

- environment variables: `INDEX_URL`, `PYSERINI_API_TOKEN`,
  `EXTERNAL_PYSERINI_HITS`, `SAMPLE_QUERIES`, optionally `PYSERINI_INDEX` and
  `PYSERINI_BASE_URL`
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

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
evaluation as separate stages. V1 ships a single-query BM25 configuration:
`configs/rag25_bm25_full_query_v1.yaml`. The facet/RRF BM25 configuration is
`configs/rag25_bm25_litellm_facets_rrf_v1.yaml`.

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
- `title` query understanding: use `Topic.title` directly as a short BM25
  query.
- `llm_facets` query understanding: call LiteLLM at
  `LITELLM_BASE_URL` with `LITELLM_MODEL` and parse strict JSON facet queries.
- `pyserini_remote` retriever: run remote BM25 over ClimbMix. Set
  `cache: true` to read and write request-keyed raw response caches, or
  `cache: false` to always call the remote endpoint.
- `passthrough` ranking: accept exactly one retrieval stream and deduplicate by
  best rank.
- `rrf` ranking: fuse title and facet BM25 streams with reciprocal rank fusion,
  deduplicate by `docid`, and preserve per-query provenance.
- `top_k` evidence selection: choose text-bearing ranked candidates.
- `placeholder` generation: write valid cited RAG JSONL for plumbing checks.
- `dev_projected_qrels` evaluation: compute development diagnostics such as
  `ndcg@10` and `recall@100`.

Run the BM25 facets + RRF pipeline after LiteLLM is listening on port 4000:

```bash
PYTHONPATH=code uv run --with pyyaml python -m trec_rag.pipeline \
  --config configs/rag25_bm25_litellm_facets_rrf_v1.yaml
```

### Local LiteLLM/vLLM facet service

On Windows, run vLLM from WSL2 Ubuntu. Official vLLM is Linux-first, so native
Windows is not the supported service path. For the detected RTX 5070 Ti 16 GB
class GPU, start the 4B Qwen model with capped context and concurrency:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen-local \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.72 \
  --enforce-eager
```

If that startup health check fails, use `Qwen/Qwen3-1.7B` with the same flags.
Then start LiteLLM from this repo with disk caching. On this Windows setup,
use the Python module entrypoint so Application Control does not block the
generated `litellm` launcher, and include the `caching` extra:

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run --with "litellm[proxy,caching]" `
  python -m litellm.proxy.proxy_cli `
  --config configs/litellm_qwen_disk_cache.example.yaml `
  --port 4000
```

Manual service checks:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:4000/v1/models
```

LiteLLM disk cache entries are configured under `outputs/litellm-cache`.
Pipeline-side facet JSON caches are stored under
`outputs/<experiment.id>/cache/facets/`.

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
PYTHONPATH=code uv run --with pytest --with pyyaml pytest code/tests -q
```

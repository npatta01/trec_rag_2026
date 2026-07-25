# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## Remote Pyserini Helpers

Stable notebook imports come from `remote_pyserini.py`. Implementation is split
by responsibility:

- `repo_env.py` loads `.env` / `.env.local` from the worktree or shared checkout.
- `env_config.py` holds typed environment helpers.
- `remote_config.py` builds `RemotePyseriniConfig`.

### Hosted API rate and continuation policy

Hosted requests use `requests` plus `requests-ratelimiter`. The default policy is
one request start per origin every three seconds with burst `1`. The per-host
limiter is stored in `cache/retrieval/pyserini_remote/rate-limit.sqlite`, so
cooperating processes and restarts share the budget. Override the interval or
state location with `PYSERINI_MIN_INTERVAL_SECONDS` and
`PYSERINI_LIMITER_STATE_PATH`; burst values other than `1` are rejected.

Each transport entry is one immutable attempt: redirects and automatic retries
are disabled and the timeout is 30 seconds. Before transport entry, an
identity-complete reservation is appended to `external-call-ledger.jsonl`.
Every response is captured under `attempts/`, even when reusable response
caching is disabled. A `429` raises
`RemotePyseriniThrottled` with the parsed `Retry-After` delay but does not sleep
and retry within that attempt. It writes an identity-bound continuation ticket;
set `PYSERINI_CONTINUATION_TICKET` to that value after the recorded not-before
time. A continuation must use the same endpoint, index, exact query text, and hit depth. The
request-keyed cache verifies that identity and reuses successful raw response
files, so only missing requests reach the transport.

Successful response bytes are written before JSON decoding. The exact bytes are
the primary `.json` cache file; the adjacent `.meta.json` records its SHA-256,
request identity, and effective rate policy. Never put bearer tokens in either
artifact or in continuation records.
- `remote_client.py` sends search requests and normalizes candidate rows.

Inputs:

- `.env.example` includes the public hosted ClimbMix search endpoint; copy it
  to a local `.env` and fill in `PYSERINI_API_TOKEN`
- endpoint environment: set `INDEX_URL` to the hosted ClimbMix search endpoint
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
.venv/bin/python -m pytest code/tests/test_remote_pyserini.py -q
```

## Config-Driven RAG Pipeline

The pipeline is the preferred path for experiments. It keeps query
understanding, retrieval, ranking, evidence selection, generation, and
evaluation as separate stages. V1 ships runnable BM25 and BM25-plus-reranker
configurations:

- `configs/rag25_bm25_full_query_v1.yaml`
- `configs/rag25_bm25_mixedbread_rerank_v1.yaml`

Run:

```bash
.venv/bin/python -m trec_rag.pipeline \
  --config configs/rag25_bm25_full_query_v1.yaml
```

Run the BM25 plus coverage-aware reranker config after the score artifacts have
been generated:

```bash
.venv/bin/python -m trec_rag.pipeline \
  --config configs/rag25_bm25_mixedbread_rerank_v1.yaml
```

The experiment ID is the run identity. If `experiment.output_dir` is omitted,
stage outputs are written to `outputs/<experiment.id>/`. Remote retriever cache
files default to a shared request cache under the repo-root cache directory:
`<shared-checkout>/cache/retrieval/pyserini_remote/`. The cache is shared
across linked worktrees and experiment configs when the request fingerprint
matches.

Topic parsing follows `topics.format` in the YAML (`tsv` or `jsonl`), not the
filename suffix. For remote Pyserini runs, `INDEX_URL` must point at the same
index declared in YAML.

Current V1 stages:

- `original_topic` query understanding: use the original narrative/prompt text.
- `pyserini_remote` retriever: run remote BM25 over ClimbMix. Set
  `cache: true` to read and write request-keyed raw response caches, or
  `cache: false` to always call the remote endpoint.
- `passthrough` ranking: accept exactly one retrieval stream and deduplicate by
  best rank.
- `coverage_aware_long_doc_aggregate` ranking: rerank a BM25 candidate stream
  with cached Mixedbread long-document and window scores.
- `top_k` evidence selection: choose text-bearing ranked candidates.
- `placeholder` generation: write valid cited RAG JSONL for plumbing checks.
- `dev_projected_qrels` evaluation: compute development diagnostics such as
  `ndcg@10`, `precision@10`, `recall@100`, `hit_rate@10`,
  `relevant_count@10`, `graded_recall@100`, and `ideal_dcg_coverage@100`.

Outputs:

- `r_output_trec_rag_2026.tsv`
- `rag_output_trec_rag_2026.jsonl`
- `retrieval_metrics.json`
- `stage_queries.jsonl`
- `stage_retrieved.jsonl`
- `stage_ranked.jsonl`
- `stage_evidence.jsonl`
- `run_metadata.json` with actual retrieval cache hits/misses and reranker
  artifact usage

Shared cache:

- `<repo-root-or-shared-checkout>/cache/retrieval/pyserini_remote/`
- `<repo-root-or-shared-checkout>/cache/reranker/artifacts/`
- `<repo-root-or-shared-checkout>/cache/reranker/score_cache/`

The Mixedbread config retrieves the cached BM25 top 1,000 but explicitly
reranks only the top 50 candidates. The untouched BM25 tail is appended after
the reranked head. This keeps the runnable config aligned with the depth at
which the coverage-aware formula was selected while retaining deeper candidate
pool metrics.

## Structured Facet Query Core

`facet_query_planning.py` is a pure, reusable boundary for externally supplied
structured facet plans. Its public typed records are the frozen dataclasses
`TokenRange`, `Anchor`, `Expansion`, `CoverageItem`, `Facet`, `FacetPlan`, and
`FacetPlanningResult`; `NarrativeToken` and `NarrativeTokenTape` describe the
token tape. Invalid plans raise `FacetPlanValidationError` from
`validate_facet_plan()`. `tokenize_narrative()` exposes the deterministic
non-whitespace token tape used to resolve half-open token ranges. The public
rendering entry point is:

```python
render_facet_queries(topic: Topic, plan: FacetPlan) -> FacetPlanningResult
```

Inputs are an existing `Topic` and a fully typed `FacetPlan` whose topic ID and
SHA-256 match the topic narrative. The core renders only the official
`Topic.narrative`: it never reads, copies, or synthesizes `Topic.title`. A
successful result has one `QueryVariant` per validated facet with
`source_type="structured_facet"`. Any `FacetPlanValidationError`, including a
late invalid facet, returns exactly one untouched original query:

```python
QueryVariant(
    topic_id=topic.id,
    variant_name="original",
    query_text=topic.narrative,
    source_type="original_topic",
)
```

`FacetPlanningResult` carries `queries`, `used_fallback`, and `error`. On
success, `queries` is the rendered facet-query tuple, `used_fallback` is
`False`, and `error` is `None`. For an invalid plan, `queries` is exactly the
original query above, `used_fallback` is `True`, and `error` retains the
validation-error message.

Frozen validation limits are: at most 16 coverage items, 8 facets, 4 global
anchors, 2 source ranges per coverage item, 2 coverage items per facet, and 8
content tokens per anchor. Each facet accepts at most 3 expansions; each has
1--3 analyzed words, and expansions may add at most 6 new unique content
tokens per facet. Expansion relations are limited to `alias`, `acronym`,
`technical_term`, `common_variant`, and `neutral_search_term`; their anchor
scope, numeric runs, and query-syntax safety are mechanically checked.

Validate the core and its existing topic/pipeline compatibility boundary with:

```bash
.venv/bin/python -m pytest \
  code/tests/test_facet_query_planning.py \
  code/tests/test_topics.py \
  code/tests/test_pipeline.py \
  -q
```

This core establishes deterministic, auditable rendering safety, not generation
quality or retrieval quality. No automatic facet-plan or answer-aspect generator
is wired into the runtime pipeline. The next promotion gate is an offline,
held-out retrieval experiment comparing externally supplied plans with the
original narrative before considering any generator or runtime integration.

## BM25 Versus Reranker Comparison

`pipeline_comparison.py` runs both configs, validates that they use the same
topics, qrels, relevance threshold, and candidate pool, then evaluates both
ranked lists with one metric suite. It writes aggregate metrics and reusable
per-topic views, including judged coverage so projected-qrels gaps remain
visible.

Inputs:

- baseline and candidate pipeline YAML files
- their shared topic file and projected qrels
- cached BM25 responses and, for the reranker, cached score artifacts

Outputs:

- `metrics.json`: aggregate deltas, topic counts, provenance, and cache usage
- `topic_metrics.csv`: one wide row per topic
- `topic_metric_deltas.csv`: one row per topic and metric

Run the default BM25/Mixedbread comparison:

```bash
.venv/bin/python -m trec_rag.pipeline_comparison
```

Validate the comparison helpers:

```bash
.venv/bin/python -m pytest code/tests/test_pipeline_comparison.py -q
```

## Restoring Shared Cache Artifacts

GitHub Release cache archives are packaged with a top-level `cache/` directory.
Extract the archive from the repository root, not from inside an existing
`cache/` directory:

```bash
cd /path/to/trec_rag_2026
tar --use-compress-program=unzstd \
  -xf trec-rag-cache-rag25-dev-20260709.tar.zst
```

After extraction, these paths should exist at the repo root:

```text
cache/retrieval/pyserini_remote/
cache/reranker/artifacts/rag25_bm25_mixedbread_rerank_v1/
cache/reranker/score_cache/
```

For linked worktrees, extract into the main/shared checkout root. The pipeline
resolves worktree cache paths back to that shared root automatically.

Run all Python tests:

```bash
.venv/bin/python -m pytest -q
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
.venv/bin/python -m pytest code/tests/test_chunking.py -q
```

## Reranker Score Cache

`rerank_score_cache.py` builds the document and window score JSONL files used
by the coverage-aware Mixedbread reranker config. It reads the shared BM25
retrieval cache for `configs/rag25_bm25_mixedbread_rerank_v1.yaml` and writes
generated score artifacts under the shared checkout:
`cache/reranker/artifacts/rag25_bm25_mixedbread_rerank_v1/`.

The expensive model outputs are also saved in a versioned, content-addressed
cache under `<shared-checkout>/cache/reranker/score_cache/schema_v2/`. Cache keys
include the scoring backend and version, exact model revision, score
representation, inference dtype, maximum sequence length, score kind, input
policy, query hash, and scored-text hash. This lets different configs,
experiments, and linked worktrees reuse scores for the same actual reranker
input without mixing incompatible model outputs.

The coverage-aware formula consumes **raw logits**. The cache builder therefore
passes an identity activation to `CrossEncoder.predict`; the model's default
sigmoid output is not interchangeable because saturated probabilities destroy
the score span used by the coverage bonus. Artifact rows carry this metadata,
and the pipeline rejects artifacts that do not match the config.

Inputs:

- the reranker pipeline config
- shared top-1,000 BM25 cache files (the scoring depth is selected at build time)
- `sentence-transformers` with `mixedbread-ai/mxbai-rerank-base-v2`

Outputs:

- `st_crossencoder_longctx_hits50_..._raw_logits_artifact_v2_scores.jsonl`
- `st_chunk_eval_hits50_..._raw_logits_artifact_v2_scores.jsonl`
- optional top-1,000 artifacts with the same schema-v2 row interface

The config pins the model revision, backend version, score representation,
dtype, and input policy. If any of those change, build a new artifact/cache
namespace instead of relabeling or reusing older probability-score files.
Artifact schema v2 also records query/document content hashes, scoring and
chunking policy, and the expected window count. Stale, partial, or conflicting
rows are rejected or repaired from the content cache before ranking.

Run one topic slowly on CPU:

```bash
.venv/bin/python -m trec_rag.rerank_score_cache \
  --topics 14 \
  --device cpu \
  --sleep-between-topics 5
```

Set up the local environment. On AMD ROCm hosts this auto-selects the ROCm
reranker environment; elsewhere it syncs the standard project environment:

```bash
code/tools/setup_env.sh
```

Run on AMD ROCm; PyTorch exposes HIP devices through the `cuda` device name:

```bash
.venv/bin/python-rocm -m trec_rag.rerank_score_cache --device cuda
```

The ROCm setup path downloads the `.python-version` interpreter with `uv`,
installs AMD ROCm PyTorch wheels from the `rocm` uv dependency group, discovers
the host ROCm runtime library path, and writes local helpers under
`.venv/bin/`. The generated files are machine-local; the reusable script can be
run on another compatible AMD ROCm Linux host.

To let uv manage the environment directly:

```bash
uv sync --group rocm
```

The repo's `.python-version` fixes this to Python 3.12.13. The wrapper script
still adds the ROCm `LD_LIBRARY_PATH` helper and runs the probe.

Validate without model loading:

```bash
.venv/bin/python -m trec_rag.rerank_score_cache --dry-run
```

Validate only the ROCm/PyTorch environment:

```bash
.venv/bin/python-rocm -m trec_rag.rocm_probe
```

### Modal compute and local promotion

`code/tools/modal_rerank_score_cache.py` runs the same score builder on an
A100-80GB and persists both its ordinary artifact-v2 JSONL files and its
schema-v2 global content cache in a Modal Volume. A fresh run removes uploaded
reranker caches before scoring, so it cannot silently combine local ROCm scores
with Modal CUDA scores. Interrupted runs can resume inside their run-scoped
Volume directory. The local entrypoints pass the selected app and Volume names
into the remote status writer explicitly, so custom environment-selected names
remain accurate in durable runtime and warm-cache provenance.

Upload a prepared `workspace/` archive containing the repository, data
submodules, and all 22 retrieval-cache files, then start a detached run:

```bash
uvx modal volume create trec-rag-rerank-score-cache-v2
uvx modal volume put trec-rag-rerank-score-cache-v2 \
  workspace.tar.zst inputs/workspace.tar.zst
uvx modal run --detach code/tools/modal_rerank_score_cache.py::main \
  --run-id rag25-hits1000-YYYYMMDD \
  --input-sha256 SHA256_OF_WORKSPACE_ARCHIVE
```

After scoring reports `completed`, prove the Modal cache is warm by rebuilding
fresh artifacts on CPU with model loading disabled. The check requires 22 zero
document-model counts, 22 zero window-model counts, an unchanged content cache,
and order-independent, key-by-key equality with the canonical artifacts. The
JSONL byte order can legitimately differ because cold misses group duplicate
content keys while warm hits follow candidate order.

```bash
uvx modal run --detach code/tools/modal_rerank_score_cache.py::verify \
  --layout run-scoped \
  --run-id rag25-hits1000-YYYYMMDD \
  --verification-id warm-cache-v1
```

Downloaded results cross a separate local trust boundary. Validation derives
all expected query, document, and chunk hashes from the local BM25 cache;
checks the exact config/model context, artifact metadata, cache keys, raw-logit
scores, per-topic coverage, and Modal runtime hashes; and verifies artifact
scores against the downloaded global cache. Run `promote` quiescent, with no
pipeline reader or second promoter. It is process-transactional: it copies and
re-hashes incoming files, archives existing targets, switches each file with
`os.replace`, and rolls back all targets if a later in-process check fails. The
archive is the recovery point for a host crash between individual file swaps.
Promotion rejects a runtime status without a document and window artifact
digest pair; a present scoring contract must contain every required field.

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.rerank_cache_promotion validate \
  --staged-document-artifact STAGING/document_scores.jsonl \
  --staged-window-artifact STAGING/window_scores.jsonl \
  --staged-score-cache-root STAGING/score_cache \
  --runtime-status STAGING/runtime_status.json

PYTHONPATH=code .venv/bin/python -m trec_rag.rerank_cache_promotion promote \
  --staged-document-artifact STAGING/document_scores.jsonl \
  --staged-window-artifact STAGING/window_scores.jsonl \
  --staged-score-cache-root STAGING/score_cache \
  --runtime-status STAGING/runtime_status.json \
  --destination-document-artifact CACHE/document_scores.jsonl \
  --destination-window-artifact CACHE/window_scores.jsonl \
  --destination-score-cache-root cache/reranker/score_cache \
  --archive-root cache/reranker/promotion_archive
```

The Modal and local paths differ, but the artifact rows, cache keys, validation
rules, and pipeline consumer interface are the same.

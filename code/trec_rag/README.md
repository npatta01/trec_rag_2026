# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## Experimental competition retrieval run

`trec_rag.competition_retrieval` is the only supported interface for the
organizer-compatible facet experiment. Run it with the strict checked-in
configuration:

```bash
uv run --no-sync python -m trec_rag.competition_retrieval configs/rag26_competition_retrieval_v1.yaml
```

The command runs all official topics by default. Repeat `--topic ID` for an
explicit set or use `--topic-subset FILE.csv`, whose required `topic_id`
column is restored to official source order. The two selector forms cannot be
combined.

The runner executes five internal stages:

1. **Planning:** DeepSeek produces a bounded structured decomposition while the
   untouched narrative remains the first retrieval lane. Saved BM25 suggestions
   are validated historical metadata, not active lanes.
2. **Retrieval and reranking:** Pyserini retrieves the original narrative plus
   one full-text lane per admitted subnarrative. Mixedbread reranks only the
   configured head of each lane, followed by deterministic round-robin
   selection.
3. **Extractive evidence:** exact source spans are scored within their own
   subnarrative, deduplicated, diversity-clustered, and selected under the
   configured budget. Raw logits are ranking values, not calibrated
   probabilities or a portable cutoff.
4. **Canonicalization:** the Nuggetizer package is wrapped for at most one
   hosted call per non-empty subnarrative. All selected passages are sent once
   for that subnarrative, with `candidate_nugget_id` retained as the upstream
   document ID and join key. Package scoring runs locally, while hosted scoring
   and assignment are deliberately bypassed; exact evidence admission and
   extractive fallback remain local. Generation quality is still experimental,
   and the adapter can be replaced behind the existing canonical-backend seam.
5. **Organizer export:** validated topic checkpoints are exported once in
   official topic order.

The YAML has five operational blocks. `experiment.id` fixes both
`outputs/<experiment.id>/` and the run ID; `topics.path` identifies the
narrative-only source; and `retrieval`, `reranking`, and `nuggets` fix
cache locations, model/index identities, depths, selection policy, evidence
budget, claim limit, and supporting-document limit. Output paths and run tags
are conventions, not configuration knobs.

A completed run publishes these six conventional files beneath the experiment
directory:

- `r_output_trec_rag_2026.tsv`: the variable-depth official evidence run;
- `retrieval_candidate_pool.trec`: a broader diagnostic pool;
- `retrieval_with_text.jsonl.zip`: the mandatory full-text archive;
- `retrieval_provenance.jsonl`: source and selection provenance;
- `resolved_config.yaml`: normalized non-secret settings and topic-source hash;
- `retrieval_export_manifest.json`: the manifest-last export seal.

Per-topic stages are also sealed by hashes through
`<topic-id>/canonical/complete.json`. Resume revalidates checkpoint schemas,
source bytes, configured identities, and the hash chain rather than trusting
file existence. Compatible sealed topics are skipped, request/content-addressed
caches retain their existing reuse rules, and the export is regenerated and
re-read. A tracked-dirty worktree is rejected before runtime dependencies are
constructed; changing `experiment.id` starts a separate checkpoint tree.

Planning transport, parsing, schema, or semantic failure retains exactly the
untouched original-narrative retrieval lane and produces empty downstream
evidence/canonical ledgers without a hosted canonical call. Canonical transport
or admission failure uses deterministic exact extractive evidence instead of an
unsupported model claim. Hosted calls are one-shot, and cached replies are
revalidated before reuse.

Runtime inputs are only official topic IDs and untouched narratives. Topic
titles, organizer subnarratives, organizer nuggets, and qrels are excluded from
planning, retrieval, evidence, and canonicalization; they belong only in
post-seal evaluation. Outputs include source text and generated claims, so keep
`outputs/` ignored and private.

**Model quality is not validated.** The two-topic pilot verifies mechanics,
provenance, fallback, and byte-stable resume behavior, but it does not establish
that generated decompositions improve retrieval or that canonical claims are
entailed. Promotion requires a frozen, held-out topic evaluation measuring
retrieval coverage, evidence quality, claim grounding, redundancy, and failure
rate.

## Competition fixed-retrieval RAG inputs

`trec_rag.competition_rag` strictly loads the organizer-facing inputs for fixed
retrieval answer generation. Its checked-in configuration is
`configs/rag26_competition_rag_gpt_sol_v1.yaml`. By default it selects all 119
canonical topics and joins these three files:

- the headerless `narrative_id<TAB>narrative` organizer topic TSV;
- `outputs/facet-deepseek-b40-v1/r_output_trec_rag_2026.tsv`, a six-field TREC
  run; and
- `outputs/facet-deepseek-b40-v1/retrieval_with_text.jsonl.zip`, whose required
  core is `query.qid`, `candidates[].docid`, and `candidates[].doc`.

The ZIP is a generation sidecar, not a TREC submission. Retrieval submits only
`r_output_trec_rag_2026.tsv`; generation submits only its final JSONL. The
deterministic generation destination is
`outputs/rag26_competition_rag_gpt_sol_v1/rag_output_trec_rag_2026.jsonl`.

Set up the environment once, then run the two paths independently with their
canonical configurations. Generation consumes the three files above after
retrieval has published its TSV and ZIP; it never reads retrieval manifests or
per-topic checkpoints.

```bash
code/tools/setup_env.sh

# Publish the full retrieval TSV and full-text ZIP.
uv run --no-sync python -m trec_rag.competition_retrieval configs/rag26_competition_retrieval_v1.yaml

# Generate the full organizer JSONL from those files.
uv run --no-sync python -m trec_rag.competition_rag --config configs/rag26_competition_rag_gpt_sol_v1.yaml
```

For a two-topic smoke run, keep the checked-in configurations and their full
exports unchanged. Repository-relative paths are resolved from the checkout
containing the config, so put local variants under the ignored
`configs/local/` directory, not `/tmp`:

```bash
mkdir -p configs/local
cp configs/rag26_competition_retrieval_v1.yaml configs/local/rag26_competition_retrieval_two_topic_smoke.yaml
cp configs/rag26_competition_rag_gpt_sol_v1.yaml configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml
```

In `configs/local/rag26_competition_retrieval_two_topic_smoke.yaml`, replace
the complete `experiment` block with this distinct retrieval namespace; leave
all other blocks identical to the checked-in retrieval config:

```yaml
experiment:
  id: facet-deepseek-b40-v1-two-topic-smoke
```

In `configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml`, replace
the complete `experiment` and `inputs` blocks with the following. The generation
output and both retrieval inputs now point to smoke-only directories, while
`inputs.topic_ids` restores the requested IDs to canonical TSV order:

```yaml
experiment:
  id: rag26-competition-rag-gpt-sol-two-topic-smoke
  output_dir: outputs/rag26-competition-rag-gpt-sol-two-topic-smoke
  mode: create

inputs:
  queries: trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv
  run: outputs/facet-deepseek-b40-v1-two-topic-smoke/r_output_trec_rag_2026.tsv
  documents: outputs/facet-deepseek-b40-v1-two-topic-smoke/retrieval_with_text.jsonl.zip
  archive_member: null
  topic_ids: [rag2026-0, rag2026-1]
```

Run the two paths with those local configs. The repeated retrieval selectors
bound the expensive retrieval work; the generation config independently bounds
the downstream join:

```bash
uv run --no-sync python -m trec_rag.competition_retrieval configs/local/rag26_competition_retrieval_two_topic_smoke.yaml --topic rag2026-0 --topic rag2026-1
uv run --no-sync python -m trec_rag.competition_rag --config configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml
```

These commands publish only under
`outputs/facet-deepseek-b40-v1-two-topic-smoke/` and
`outputs/rag26-competition-rag-gpt-sol-two-topic-smoke/`; they never replace
the full canonical exports.

`experiment.mode: create` refuses an existing generation output or work tree.
For an interrupted generation, switch the local config to `resume`; valid
per-topic rows are reused and missing rows are generated. To replace a
generation result, use `overwrite`: it removes only that generation JSONL and
its dedicated `work/` directory before starting again, never the retrieval TSV
or ZIP inputs.

Raw provider responses are retained only when safely available: parsed JSON is
stored as a recursively sanitized structured envelope with no duplicate raw
body. Opaque non-JSON bodies are never persisted, for any HTTP status including
a 2xx semantic failure; their artifacts contain only the status when available,
an omission marker, UTF-8 byte length, and SHA-256. Run the module's targeted
contract suite with the repository environment already set up:

```bash
uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q
```

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

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

## Structured Sparse Query Planning

`query_planner.py` generates an auditable plan before retrieval. The local LLM
identifies exact narrative anchors, a compact set of lexical expansions,
adaptive facets, and a coverage map. It does not write free-form BM25 queries.
The renderer constructs query strings only from literal narrative spans,
resolved anchors, and provenance-tagged expansion terms.

Inputs:

- TSV or JSONL topics accepted by `topics.py`
- an OpenAI-compatible `/v1/chat/completions` endpoint
- a model revision, reasoning effort, seed, and strict JSON response schema

Outputs:

- one validated query plan per topic
- deterministic global and facet query strings
- model, prompt, schema, latency, usage, renderer, and analyzer provenance
- content-addressed caches under the shared checkout's `cache/query_plans/`

The current local smoke-test service uses the existing ROCm vLLM image:

```bash
podman run --rm --name trec-rag-gpt-oss \
  --group-add keep-groups \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --device /dev/kfd \
  --device /dev/dri \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  docker.io/vllm/vllm-openai-rocm:latest \
  --model openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --served-model-name gpt-oss-local \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.30 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --enforce-eager
```

Generate the five diagnostic development plans without running retrieval:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.query_plan_cli \
  --topic-ids 144,213,224,407,515 \
  --model-revision 6cee5e81ee83917806bbde320786a8fb61efebee
```

The CLI fails the run when any first emission is unparsable, violates exact
span/reference constraints, exceeds expansion budgets, or cannot be rendered
deterministically. Semantic coverage and unsupported-entity checks remain a
separate advisor gate before retrieval.

Run planner validation:

```bash
.venv/bin/python -m pytest code/tests/test_query_planner.py -q
```

### Query planner v2

V2 replaces model-copied source strings with exact token ranges, derives
anchor inheritance and retrieval strings in Python, and uses a pinned local
Lucene reference analyzer for budgets. The reference implements the documented
Anserini default chain; it is not presented as a verified fingerprint of the
hosted ClimbMix deployment.

The current delivered boundary is schema `query_plan_v2_1` with prompt
`sparse_query_planner_v6`. V2.1 removes decoder-side `uniqueItems` for vLLM
0.24 compatibility; Python still rejects duplicate references.

Start the local analyzer sidecar:

```bash
code/tools/run_lucene_analyzer.sh
```

The v2 runner is serial and raw-first. It preflights the analyzer and the local
server context, writes exact HTTP bytes before UTF-8/JSON parsing, and commits
immutable per-topic outcomes. By default it refuses non-loopback model,
analyzer, and tokenizer endpoints, so the experiment makes no paid API,
retrieval, or reranker calls.

Before any HTTP request, compile-lint the new synthetic and all five exact
diagnostic schemas inside the pinned container (no inference):

```bash
PYTHONPATH=code .venv/bin/python code/tools/query_plan_v2_schema_preflight.py
```

The resulting `compiler_manifest_xgrammar.json` is accepted only when the
inspected server launch command explicitly pins backend `xgrammar`.

Run the one synthetic transport/schema smoke (not a development topic):

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.query_plan_v2_cli \
  --run-kind synthetic_transport_smoke \
  --run-id query_plan_v2_1_synthetic_smoke_001 \
  --output outputs/query_planner_v2_1_synthetic_smoke_001/plans.jsonl \
  --expected-analyzer-fingerprint-sha256 \
    f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4
```

Only after that artifact receives advisor approval, run the locked diagnostic
set. The CLI accepts exactly these five IDs in this order:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.query_plan_v2_cli \
  --run-kind diagnostic_first_emission \
  --topic-ids 144,213,224,407,515 \
  --run-id query_plan_v2_gpt_oss_20b_diagnostic_001 \
  --output outputs/query_plan_v2_gpt_oss_20b_diagnostic_001/plans.jsonl \
  --expected-analyzer-fingerprint-sha256 \
    f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4
```

Rebuild derived JSONL files and the manifest without contacting any service:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.query_plan_v2_cli \
  --rebuild-only \
  --outcome-dir outputs/query_planner_v2_1_synthetic_smoke_001/outcomes/query_plan_v2_1_synthetic_smoke_001 \
  --output outputs/query_planner_v2_1_synthetic_smoke_001/plans.jsonl
```

Run all planner, analyzer, and ledger tests:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_query_planner_v2_contract.py \
  code/tests/test_query_planner_v2_generation_contract.py \
  code/tests/test_query_plan_v2_cli.py \
  code/tests/test_query_analyzer_contract.py
```

### Deterministic sparse v1

`det_sparse_v1` is the separately versioned continuation after the v2.1 model
arm's admission failure. It makes no model call. It keeps the exact original
query, splits only source-backed request units, copies the first unit as shared
parent context, caps facets at four, derives optional single-word expansion
from frozen BM25 results, and combines streams with weighted reciprocal-rank
fusion. Model anchor kinds and free-form scope references do not exist in this
arm.

The frozen diagnostic configuration is
`configs/det_sparse_v1.yaml`. It names exactly four non-locked difficult-topic
diagnostics, `hits=100`, RRF `k=60`, no model or reranker, and a durable ceiling
of 36 external requests. The four explicit ablations are original (`O`),
original plus facets (`F`), original plus PRF expansion (`E`), and the combined
arm (`FE`). This set is deliberately qrel-conditioned and cannot support a
generalization claim.

Run the synthetic/offline contract tests without contacting retrieval or a
model:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deterministic_sparse.py \
  code/tests/test_det_sparse_config.py \
  code/tests/test_det_sparse_arms.py \
  code/tests/test_rrf.py \
  code/tests/test_det_sparse_ledger.py \
  code/tests/test_det_sparse_budget.py \
  code/tests/test_det_sparse_transport.py \
  code/tests/test_det_sparse_freeze.py \
  code/tests/test_det_sparse_preflight.py \
  code/tests/test_det_sparse_provenance.py \
  code/tests/test_det_sparse_nonregression.py \
  code/tests/test_det_sparse_run.py
```

After the source is committed and the pinned local analyzer is running, the
offline preflight command materializes exact plans and a create-only
pre-retrieval freeze. It reads neither qrels nor the retrieval endpoint:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.det_sparse_preflight \
  --config configs/det_sparse_v1.yaml
```

The retrieval ledger in `det_sparse_ledger.py` reserves every attempt before
the call, writes exact response bytes before parsing, verifies exact identities
and hashes, and stops before request 37. The formal pilot uses a fresh run-local
ledger and canonical exact-query aliases rather than a pre-existing shared
cache. A second experiment-global ticket ledger is stored under the Git common
directory, so linked worktrees cannot reset the 36-call ceiling. The only
admissible network implementation is a no-retry/no-redirect transport bound to
the single frozen HTTPS endpoint.

Passing preflight does not itself authorize retrieval. External execution is
currently fail-closed because the hosted ClimbMix service exposes no immutable
index revision; changing that requires a separately reviewed, versioned gate.
Evaluation helpers replay plans, PRF, arms, and rankings from frozen raw ledger
evidence before their first qrels read, then hash and parse the same qrels byte
snapshot and apply the preregistered
paired promotion gates.

### Deterministic sparse v2

`det_sparse_v2` is a new, isolated offline arm created after exact-shape review
retired v1 without retrieval. It keeps the source-backed splitter and local
Lucene analyzer, but protects the first unit as `f01` and gives every child a
bounded strict prefix of that unit instead of copying the entire parent. Every
facet must have a BM25 term-frequency signature distinct from the original and
from every other facet. This fixes the v1 two-unit collapse mechanically; an
advisor still has to judge whether the resulting queries are meaningful.

The frozen configuration is `configs/det_sparse_v2.yaml`. It screens exactly
13 fresh candidate topics without qrels, baseline scores, retrieval, a model,
or a reranker, then deterministically selects one eligible topic from each of
four structural strata. If a stratum is empty or any selected plan falls back,
the pilot stops without substitution. The known five planner topics and all
four burned v1 topics are excluded.

Run the offline v2 contract tests:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deterministic_sparse_v2.py \
  code/tests/test_det_sparse_v2_config.py \
  code/tests/test_det_sparse_v2_selection.py \
  code/tests/test_det_sparse_v2_provenance.py \
  code/tests/test_det_sparse_v2_preflight.py
```

After committing a clean source tree and starting the pinned loopback analyzer,
create the qrels-blind, create-only preflight:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.det_sparse_v2_preflight \
  --config configs/det_sparse_v2.yaml
```

This command writes candidate screening, selected plans, exact base queries,
coverage paths, request projections, and a pre-retrieval freeze. It reserves
the create-only attempt before screening and replays the sealed semantics with
a fresh local analyzer before reporting success. It makes zero external,
model, and reranker calls and does not open qrels. Retrieval remains hard-blocked
while the collection identity is
`hosted_climbmix_unknown_revision`; passing mechanical and advisor shape review
does not open that gate.

### Deterministic sparse v3

`det_sparse_v3` is the fresh continuation after v2 was archived without
retrieval. V2's leading-prefix anchor was mechanically distinct but lost the
topic referent in two child queries. V3 instead extracts a compact exact span
from anywhere in the first unit, using only jointly supported cross-unit term
recurrence. Conversational source occurrences do not count as anchor evidence,
disjoint recurrent subjects cause abstention, and every child receives the
same exact source-backed anchor unconditionally.

The frozen design is `docs/superpowers/det_sparse_v3_design.md`. It excludes all
known-five, v1, and v2 topics and reserves the nine untouched IDs for a
qrels-blind critical-first selection. A selected critical case must contain a
raw and final child with no anchor-core BM25 term before insertion. The other
three topics come from fixed structural quantile bins. No local model is used;
model-assisted exact-span selection remains a separately versioned challenger
only if this deterministic method abstains or fails.

V3 keeps the same closed external gate: its offline preflight may use only the
digest-pinned loopback Lucene analyzer and must record zero retrieval, model,
and reranker calls plus `qrels_opened=false`. Passing preflight would still not
authorize retrieval.

The executable contract is `configs/det_sparse_v3.yaml`. The preflight reads
only the nine raw-byte-matched candidate rows from the configured TSV, after it
has attested a clean committed source tree and written create-only reservation
and conversational-inventory records. It seals all nine candidate plans, the
four selected plans (when selection succeeds), criticality and coverage
ledgers, canonical query records, request ceilings, and a replayable freeze
under `outputs/rag25_det_sparse_structural4_v3/preflight/`.

After the preregistered synthetic, full-repository, advisor, and live-Lucene
gates pass, the one allowed build command is:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.det_sparse_v3_preflight \
  --config configs/det_sparse_v3.yaml
```

A successful existing freeze can be checked with the same command plus
`--validate-only`. Validation creates a fresh local analyzer client and
recomputes the projection, nine plans, selection, witnesses, canonical bytes,
and exact artifact inventory. Neither mode authorizes an external request.

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

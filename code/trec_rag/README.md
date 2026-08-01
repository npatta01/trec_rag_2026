# TREC RAG Helpers

Reusable Python helpers for repository notebooks and experiments.

## Reusable tracing modules

`trec_rag.tracing` contains the reusable trace model and Phoenix-export
boundaries:

- `trec_rag.tracing.models` defines the immutable `SpanSpec` and `TraceBundle`
  records.
- `trec_rag.tracing.openai_semantics` converts `SpanSpec` payloads into
  normalized messages, captured generic tool schemas, OpenAI request/response
  envelopes, and OpenInference LLM attributes.
- `trec_rag.tracing.phoenix_export` exports a saved `TraceBundle`; normalized
  envelopes populate generic input/output fields while complete strict JSON is
  retained under `pi.native.input_json` and `pi.native.output_json`.
  These attributes use canonical JSON encoding and preserve every parsed value,
  including exact string contents. Native JSONL remains authoritative for
  source bytes such as whitespace and object-key order.

## Temporary organizer Pi reproduction harness

`trec_rag.experiments.organizer_pi` is a temporary experimental reproduction
harness, not a main production path. Its `inputs` module prepares byte-faithful
single-topic inputs, `event_trace` owns `piika_tool_schemas()` and constructs
durable trace bundles from native organizer artifacts, and `cli` builds and
exports those bundles. A partial
fixed-run failure record retains the selected `query_id`; native failures and
validation become error-status spans. Native Pi JSONL events are authoritative;
the bundle and Phoenix view are derived records. Keep native events, normalized
output records, bundles, and receipts together so the derived trace can always
be checked against its source.

The adapter emits only captured messages and known invocation parameters. In
particular, it does not fabricate the Pi system prompt when that prompt was not
captured. Validate semantic, export, and event-normalization behavior together:

For fixed traces with captured native thinking, the trace exposes one `CHAIN`
child per native block while retaining one LLM/cost span. Child timing is a
reconstructed equal partition of the generation interval; the extracted
reasoning is omitted from the LLM presentation to avoid duplication, while the
complete native response value remains canonically encoded under
`pi.native.output_json`.

```bash
.venv/bin/python -m pytest \
  code/tests/tracing/test_openai_semantics.py \
  code/tests/tracing/test_phoenix_export.py \
  code/tests/experiments/organizer_pi/test_event_trace.py \
  code/tests/experiments/organizer_pi/test_cli.py -q
```

Choose ignored local paths first. The CLI rejects bundle and receipt paths that
Git would track:

```bash
TRACE_ROOT=outputs/organizer-pi-phoenix/rag2026-1
git check-ignore "$TRACE_ROOT/probe"
```

Build the Piika agentic bundle from its one-topic TSV, native events, and
normalized per-query run artifact:

```bash
.venv/bin/python -m trec_rag.experiments.organizer_pi.cli build \
  --baseline piika-agentic \
  --topic rag2026-1 \
  --topic-tsv "$TRACE_ROOT/inputs/rag2026-1.tsv" \
  --events "$TRACE_ROOT/piika/merged/raw-events/rag2026-1.jsonl" \
  --record "$TRACE_ROOT/piika/merged/rag2026-1.json" \
  --session rag2026-1-comparison \
  --bundle "$TRACE_ROOT/piika/rag2026-1.trace.json"
```

Build the fixed-retrieval bundle from the same topic plus the filtered 100-row
run, published document ZIP, and unmodified organizer script:

```bash
DOCUMENT_ZIP="$TRACE_ROOT/sources/trec-rag-data/trec-rag-2026/baselines/retrieval/bm25_climbmix_top1000_with_text.jsonl.zip"
ORGANIZER_SCRIPT="$TRACE_ROOT/sources/trec-rag-data/trec-rag-2026/baselines/rag/code/ragnarok_style_ag.py"
ORGANIZER_SCRIPT_SHA256="<pinned lowercase SHA-256 from source provenance>"

.venv/bin/python -m trec_rag.experiments.organizer_pi.cli build \
  --baseline ragnarok-fixed \
  --topic rag2026-1 \
  --topic-tsv "$TRACE_ROOT/inputs/rag2026-1.tsv" \
  --events "$TRACE_ROOT/fixed/raw/rag2026-1.events.jsonl" \
  --record "$TRACE_ROOT/fixed/rows/rag2026-1.json" \
  --ranked-run "$TRACE_ROOT/inputs/rag2026-1.top100.trec" \
  --documents "$DOCUMENT_ZIP" \
  --organizer-script "$ORGANIZER_SCRIPT" \
  --organizer-script-sha256 "$ORGANIZER_SCRIPT_SHA256" \
  --session rag2026-1-comparison \
  --bundle "$TRACE_ROOT/fixed/rag2026-1.trace.json"
```

The fixed builder streams the ZIP, retains only the selected 100 document IDs,
and independently applies the organizer's 1,000-word cap to each document. It
imports `SYSTEM_PROMPT` and `prompt()` from the supplied organizer script only
after it matches the separately pinned SHA-256. Completed runs require a
matching native Pi user message. An explicit failed run with a native failure
event can still produce a partial error-status bundle when failure occurred
before prompt emission; that bundle omits evidence and prompt spans that were
not actually reached.

The captured comparison used only `rag2026-1`. The reusable converter accepts
another topic ID only when all supplied single-topic artifacts agree on that
same ID; this does not broaden the original hosted experiment.

Export is a separate command and requires the optional `observability`
dependency group (`uv sync --group observability`). Put these values in the
repository's ignored `.env` or `.env.local`; never put an API key on the command
line:

```text
PHOENIX_API_KEY=<secret>
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<space>
PHOENIX_PROJECT_NAME=trec-rag-2026-pi-baselines
```

Then export each saved bundle:

```bash
.venv/bin/python -m trec_rag.experiments.organizer_pi.cli export \
  --bundle "$TRACE_ROOT/piika/rag2026-1.trace.json" \
  --receipt "$TRACE_ROOT/piika/phoenix-receipt.json"

.venv/bin/python -m trec_rag.experiments.organizer_pi.cli export \
  --bundle "$TRACE_ROOT/fixed/rag2026-1.trace.json" \
  --receipt "$TRACE_ROOT/fixed/phoenix-receipt.json"
```

The Phoenix project is `trec-rag-2026-pi-baselines`. Export sends the complete
narrative, prompts, search/tool inputs and outputs, selected document text,
assistant messages, and validation record to the configured hosted collector.
This full-content transmission is intentional and must use only data authorized
for that Phoenix project. The CLI scans configured credential values before
export and writes the public project name, trace ID, root span ID, and span count
to the receipt only after every span export and the provider flush succeed. A
failed export leaves the durable bundle intact and does not create a new
receipt.

## Experimental competition retrieval run

`trec_rag.competition_retrieval` is the only supported interface for the
organizer-compatible facet experiment. Run it with the strict checked-in
configuration:

```bash
uv run --no-sync .venv/bin/python-rocm -m trec_rag.competition_retrieval configs/rag26_competition_retrieval_v1.yaml
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

## Evidence bundle boundary

`trec_rag.evidence_bundle` defines the versioned cross-stage contract for the
Evidence Bundle v1 plan.

Inputs:
- neutral retrieval rows with lane metadata, document text, rank, score,
  retriever identity, and optional parent/trace identifiers
- optional in-memory bundle relations for derived selections, evidence spans,
  nuggets, and hashed trace references

Outputs:
- a validated `EvidenceBundle` with frozen lane, document, retrieval-event,
  selection, selection-member, evidence, nugget, and trace-reference records
- deterministic JSON-compatible dictionaries through `to_dict()` /
  `from_dict()`, wrapped in a top-level `schema_version: evidence_bundle_v1`
  marker; v1 decoding requires the complete relation/key shape and rejects
  Boolean values in numeric scalar fields
- deterministic downstream projections through `to_trec_run(selection_id=...)`,
  `to_document_records(selection_id=...)`, and
  `to_fixed_rag_context(selection_id=...)`
- `write_fixed_rag_package(bundles, output_dir, selection_id=...)`, which sorts
  validated per-topic bundles by topic ID and writes one deterministic package:
  `trec_rag_2026_queries.tsv`, an organizer-compatible six-column
  `r_output_trec_rag_2026.tsv`, `retrieval_with_text.jsonl`, deterministic
  `retrieval_with_text.jsonl.zip`, and `fixed_rag_context.jsonl`

Validation:
- every identifier and text hash is checked, including each lane query hash;
  document and query text may contain ordinary tabs and line breaks
- every document/lane membership must have a corresponding retrieval event
- every selection member audits one input document with inclusion state,
  optional output rank, and a rejection reason for excluded documents; the
  member relation must match the exact union implied by its source lanes
- the `natural_union` selection includes every unique document without output
  ranks and is rejected by rank-bearing projections; named ranked selections
  provide contiguous explicit ranks without an implicit top-100 limit
- every evidence span has non-empty lane support and resolves exactly inside its
  parent document text
- every nugget has non-empty known evidence support, and an optional
  `subnarrative_id` must resolve to a lane whose kind is `subnarrative`

Downstream consumption:
- fixed retrieval and fixed-bundle RAG consume deterministic projections from
  the same validated bundle boundary; document and fixed-context records carry
  the topic query text and its hash alongside the selected evidence
- agentic downstream work may consume the same bundle projections as read-only
  context, but any newly retrieved documents or agent-produced evidence must be
  recorded as a new bundle revision instead of being hidden only in trace logs

Compatibility note:
- the bundle is the cross-stage contract; existing retrieval, decomposition,
  evidence, and nugget stage schemas remain stage-local and unchanged in this
  implementation

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
uv run --no-sync .venv/bin/python-rocm -m trec_rag.competition_retrieval configs/rag26_competition_retrieval_v1.yaml

# Generate the full organizer JSONL from those files.
uv run --no-sync .venv/bin/python -m trec_rag.competition_rag --config configs/rag26_competition_rag_gpt_sol_v1.yaml
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
uv run --no-sync .venv/bin/python-rocm -m trec_rag.competition_retrieval configs/local/rag26_competition_retrieval_two_topic_smoke.yaml --topic rag2026-0 --topic rag2026-1
uv run --no-sync .venv/bin/python -m trec_rag.competition_rag --config configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml
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
uv run --no-sync .venv/bin/python -m pytest code/tests/test_competition_rag.py -q
```

## Private post-run competition debug report

`trec_rag.competition_debug_report` explains an already completed competition
run from its standard retrieval config and, optionally, its matching standard
RAG config. A retrieval-only report uses:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v1.yaml
```

Include validated final answers by supplying the RAG config rather than a raw
output path:

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v1.yaml \
  --rag-config configs/rag26_competition_rag_gpt_sol_v1.yaml
```

Repeat `--topic ID` to select exported topics in official order. Use `--output
PATH` only for an existing parent directory inside this repository. Otherwise,
the report is atomically written to
`<retrieval-output>/competition_debug_report.html`. Output overrides must end
in `.html`; symbolic links and existing files not generated by this report
command are rejected so organizer, checkpoint, and publication artifacts stay
read-only.

This command is a read-only, post-run validator and renderer: it makes no
retrieval, API, network, or model calls and does not scan the large candidate
ledgers. It reads only bounded sealed artifacts, validates every RAG output
record with the production submission validator, and emits exactly one compact
JSON receipt on stdout after successful replacement.

**Keep the HTML private.** It contains raw corpus text, generated claims,
answers, and document identifiers. It is not a sanitized publication artifact
and must not be uploaded or shared without an explicit privacy review.

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

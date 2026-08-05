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
PYTHONPATH=code .venv/bin/python-rocm -m trec_rag.competition_retrieval \
  configs/rag26_competition_retrieval_v2.yaml
```

The command runs all official topics by default. Repeat `--topic ID` for an
explicit set or use `--topic-subset FILE.csv`, whose required `topic_id`
column is restored to official source order. The two selector forms cannot be
combined.

The runner executes five internal stages:

1. **Planning:** DeepSeek produces a bounded structured decomposition while the
   untouched narrative remains the first retrieval lane. Saved BM25 suggestions
   are validated historical metadata, not active lanes.
2. **Shared passage retrieval:** for every focused original or subnarrative
   query, Pyserini retrieves up to 1,000 documents. Every non-empty returned
   document is chunked and scored with Mixedbread; the query retains one global
   top-100 passage list. The fixed and agentic adapters call this same search
   boundary and cache identity.
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
5. **Organizer export:** validated, evidence-backed documents are deduplicated
   and ranked at a narrative-specific depth. The 1,000-document and 100-passage
   search ceilings never pad or cap the final organizer run.

The YAML has six operational blocks. `experiment.id` fixes both
`outputs/<experiment.id>/` and the run ID; `topics.path` identifies the
narrative-only source; `execution.topic_workers` controls independent local
topic processes; and `retrieval`, `passage`, and `nuggets` fix cache locations,
model/index identities, the 1,000/100 shared-search policy, evidence budget,
claim limit, and supporting-document limit. Output paths and run tags are
conventions, not configuration knobs.

Each topic worker constructs its own scorer and may place another model copy on
the auto-selected accelerator. Choose `topic_workers` from available device
memory and use `1` on a memory-constrained single GPU. The runner intentionally
does not impose a device-count ceiling because CPU and externally sharded topic
workers use the same topic-first contract.

A completed run publishes these three conventional files beneath the experiment
directory:

- `r_output_trec_rag_2026.tsv`: the variable-depth official evidence run;
- `retrieval_with_text.jsonl.zip`: the mandatory full-text archive;
- `retrieval_export_manifest.json`: the manifest-last export seal.

The official run and full-text archive are deterministic projections of each
topic's validated Evidence Bundle. Candidate-pool, provenance, and resolved
configuration sidecars are not exported; their sealed stage inputs remain
available inside the private per-topic checkpoints when needed for validation.

Per-topic stages are also sealed by hashes through
`<topic-id>/canonical/complete.json`. Resume revalidates checkpoint schemas,
source bytes, configured identities, and the hash chain rather than trusting
file existence. Each topic owns one `records.sqlite3` ledger and one dispatch
receipt. Complete and bounded-incomplete topics are both resumable; each carries
an explicit stopping reason. The projection and dispatch receipt both bind the
SHA-256 of the exact YAML bytes used by that topic, so a projection-only crash
cannot be promoted after even a non-semantic config edit. Compatible sealed topics are skipped,
request/content-addressed caches retain their existing reuse rules, and the
export is regenerated and re-read in official source order. A tracked-dirty
worktree is rejected before runtime dependencies are constructed; changing
`experiment.id` starts a separate checkpoint tree.

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

### Splitting one run across hosts

Topic selection plus per-topic sealing make it practical to run part of a run on
a rented GPU box and finish it locally. Partition the topics with
`--topic-subset`, never overlap the two sets, and copy the remote work back
before the final local run. Sealed topics revalidate and are skipped, the
remaining topics run locally, and the export is regenerated over every selected
topic, so the organizer TSV, full-text ZIP, and handoff never need manual
merging.

The downloadable artifact is **two directories**, not one:

| Path | Required | Why |
| --- | --- | --- |
| `outputs/<experiment.id>/<topic-id>/` | yes | sealed per-topic checkpoints, ledger, and receipts |
| `cache/documents/v1/` | yes | content-addressed document text the export reads for every row |
| `cache/retrieval/pyserini_remote/` | no | avoids re-querying the hosted index if a topic is rerun |
| `cache/canonical/<prompt-version>/` | no | avoids re-paying for hosted canonical calls on a rerun |
| `cache/reranker/` | **no — do not copy** | the score database path is derived from the scoring context, so both hosts write the same filename with different contents and a copy clobbers local scores |

Skipping `cache/documents/v1/` is the quiet failure: the checkpoints validate
and the export then fails on missing document text. Both required directories
are content-addressed or topic-scoped, so they merge by copy.

Both hosts must agree on the things the checkpoints seal, or resume fails
closed:

- **The same config bytes.** The YAML SHA-256 is sealed into every dispatch and
  projection receipt, so `experiment.id` and every other field must be
  byte-identical on both hosts.
- **`passage.device` pinned, not `auto`.** The resolved device string is part of
  the sealed passage-search identity, and the exporting host recomputes it.
  `auto` resolving to `cpu` remotely and `cuda` locally rejects the download.
  The checked-in config pins `cuda`, which both ROCm and NVIDIA report.
- **The same `INDEX_URL`.** It is part of the retriever identity and is required
  even for a resume-and-export pass that issues no queries.
- **A clean tracked worktree containing real git metadata.** The runner rejects
  a dirty tree and reads `HEAD` for the recorded commit, so ship a checkout
  rather than a file tarball.
- **No commit change inside a topic.** All phases of one topic must share a
  commit; different topics may carry different commits.
- **Both pinned model snapshots pre-cached.** The retrieval path loads two, each
  at a pinned revision with `local_files_only=True`, and each must resolve from
  the cache alone before a run starts:

  | Snapshot | Revision | Loaded by |
  | --- | --- | --- |
  | `mixedbread-ai/mxbai-rerank-base-v2` | `3ea9d4dffa7d12a4f366be8e275c349de9fc9865` | passage scoring |
  | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | evidence selection |

  `code/tools/setup_cuda_env.sh` fetches both and re-resolves each with
  `local_files_only=True`. Caching only the reranker is the trap: selection runs
  *after* passage scoring, so the missing MiniLM surfaces as a late failure with
  the expensive stage already paid for.

Set up a rented NVIDIA box with `code/tools/setup_env.sh`, which now detects
CUDA as well as ROCm and syncs the `cuda` dependency group. That group pins the
same `sentence-transformers`, `transformers`, and `numpy` versions the `rocm`
group resolves to, so reranker score-cache entries stay interchangeable and the
recorded backend version stays honest. The two groups are declared conflicting,
so `uv.lock` carries separate CUDA and ROCm resolution forks. Regenerate it on a
ROCm host whenever either group changes, because `repo.radeon.com` must be
reachable while uv resolves both forks.

`code/tools/verify_torch_groups.sh` proves three things without installing
anything:

1. **`uv.lock` is current with `pyproject.toml`.** `uv lock --check` re-resolves
   and compares the result against the committed lock, failing if they differ.
   It never writes the lock and never installs, but it does re-resolve, so it
   needs package-index access.
2. **Each group selects the torch build it should.** `uv export --frozen` reads
   the committed lock alone. The `rocm` group must resolve a `repo.radeon.com`
   wheel. The `cuda` group must resolve *exactly* `torch==2.9.1` — a direct URL,
   a CPU wheel, or any other version fails — and the export must also carry the
   `nvidia-cuda-runtime`, `nvidia-cublas`, and `nvidia-cudnn` packages a
   GPU-enabled wheel depends on, which is what separates the real build from a
   CPU-only one published under the same version.
3. **That `cuda` torch comes from PyPI.** The requirements format renders every
   registry package as `name==version` with no source attached, so
   `torch==2.9.1` reads identically whether PyPI, a mirror, or a private index
   served it. To settle it, the script also exports the same group as PEP 751
   metadata (`uv export --frozen --group cuda --format pylock.toml`), which does
   record a per-package index, and requires the single `torch` entry there to be
   version `2.9.1` with index exactly `https://pypi.org/simple`. Asking uv for
   the group-scoped export is what keeps this honest — reading an unscoped
   multi-fork `uv.lock` block would leave the fork ambiguous.

Every uv call runs with `--no-cache` and a throwaway `--cache-dir` removed on
exit, so the script neither reads nor writes the shared persistent uv cache, and
it never touches a model, retrieval, or reranker cache. Every call also runs
with `--no-python-downloads`, which is what makes "installs nothing" hold on a
host that does not already have the pinned Python 3.12.13: without it, `uv lock`
would fetch and install a managed interpreter just to resolve. With it, such a
host fails loudly instead. Safe to run beside an active pipeline task. When uv
itself fails for an unrelated reason — no network, a rejected credential, an
unknown flag — the script reports uv's own message instead of blaming the
dependency group, and it withholds the "regenerate the lock" hint, which is
printed only when the lock is actually diagnosed as stale.

```bash
./code/tools/verify_torch_groups.sh
OK   uv.lock is current with pyproject.toml
OK   rocm: torch @ https://repo.radeon.com/rocm/.../torch-2.9.1+rocm7.2.1...whl
OK   cuda: torch==2.9.1 from https://pypi.org/simple
```

`code/tests/test_verify_torch_groups.py` and
`code/tests/test_env_setup_contract.py` cover this bootstrap path hermetically:
the first runs the script against a stub `uv` on `PATH`, the second executes the
setup script's prefetch block against a stubbed `huggingface_hub` and drives the
production loaders with stub loaders. Neither reaches the network, resolves
dependencies, or downloads a model.

```bash
.venv/bin/python -m pytest \
  code/tests/test_verify_torch_groups.py \
  code/tests/test_env_setup_contract.py -q
```

These tests do write files — a throwaway SQLite score-cache database and a stub
uv cache directory — but only under pytest's per-test `tmp_path`. Nothing is
written to the shared persistent caches under `cache/` (`cache/reranker/`,
`cache/retrieval/`, `cache/documents/`), to the Hugging Face model cache, or to
the shared uv cache, so the suite is safe to run beside a live pipeline task.

**Model quality is not validated.** The two-topic pilot verifies mechanics,
provenance, fallback, and byte-stable resume behavior, but it does not establish
that generated decompositions improve retrieval or that canonical claims are
entailed. Promotion requires a frozen, held-out topic evaluation measuring
retrieval coverage, evidence quality, claim grounding, redundancy, and failure
rate.

## Topic-record source geometry

`topic_geometry.py` derives immutable validation geometry once per exact UTF-8
document body. Input is a full lowercase content SHA-256 plus the exact source
text. Output is a `DocumentGeometry` containing character-to-byte offsets,
scoring text/source boundaries, exact paragraph and sentence spans, and O(1)
membership/adjacency indexes. `DocumentGeometryIndex` reuses that object for
repeated candidates, rejects a digest mismatch or conflicting source, and
exposes derivation counters for bounded-work tests. It performs no SQL, model,
network, cache publication, or topic lookup; TopicRecords owns those concerns.

Validate the module with:

```bash
.venv/bin/python -m pytest code/tests/test_topic_geometry.py -q
```

The tests cover exact Unicode byte/scoring coordinates, paragraph and sentence
membership/adjacency, one derivation per content hash across multiple documents,
digest conflicts, invalid lookups, and duplicate derived-span rejection.

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
- the competition retrieval checkpoint now seals the retrieval-stage bundle at
  `retrieval/evidence-bundle.json`; it contains the full retained natural
  document union before any downstream context projection
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

Migration note:
- the bundle is the cross-stage contract; decomposition, evidence, and nugget
  records remain stage-local, while retrieval checkpoints now include the
  sealed bundle artifact as the retrieval handoff. The exporter no longer
  publishes the superseded candidate-pool, provenance, or resolved-config
  sidecars.

## Competition selected-evidence RAG input

`trec_rag.competition_rag` accepts one immutable input:
`generation_handoff_manifest.json`. Retrieval builds it from the authenticated
per-topic `TopicRecords` snapshot and includes the exact official narrative,
selected passages grouped by subnarrative and cluster, advisory canonical claim
hints, and the only document IDs generation may cite. It contains no complete
documents, document-head windows, qrels, gold nuggets, or RAGDoll scores.

The checked-in one-shot configs are
`configs/rag26_competition_rag_gpt_sol_v2.yaml` and
`configs/rag26_competition_rag_deepseek_v2.yaml`. They use the same runner,
prompt, validation, retry, and citation-conversion path. Their model settings
are the intended difference. Each config has one input field:

```yaml
inputs:
  handoff_manifest: outputs/facet-deepseek-b40-v3/generation_handoff_manifest.json
```

Generation validates the complete handoff before mutating generation state.
The model must cite raw ClimbMix document IDs from the current topic's sealed
domain; code validates those IDs and deterministically converts them to the
organizer's integer citation indexes. The organizer TREC run and full-text ZIP
remain retrieval/evaluation artifacts, but fixed generation never opens them.

Set up the environment once, run retrieval to completion, confirm the root
manifest-last receipt includes the handoff, and then run generation:

```bash
code/tools/setup_env.sh

PYTHONPATH=code .venv/bin/python-rocm -m trec_rag.competition_retrieval \
  configs/rag26_competition_retrieval_v2.yaml

.venv/bin/python -m trec_rag.competition_rag \
  --config configs/rag26_competition_rag_gpt_sol_v2.yaml
```

For a two-topic smoke, preserve the checked-in full-run configs and copy both
under ignored `configs/local/`. Give each a smoke-only experiment ID/output
directory. Point the local RAG config at the local retrieval handoff and put the
same topic IDs under `experiment.topic_ids`:

```bash
mkdir -p configs/local
cp configs/rag26_competition_retrieval_v2.yaml \
  configs/local/two-topic-passage-v2.yaml
cp configs/rag26_competition_rag_gpt_sol_v2.yaml \
  configs/local/rag26-competition-rag-gpt-sol-two-topic-v2.yaml
```

```yaml
# Retrieval config
experiment:
  id: facet-deepseek-selected-evidence-two-topic-smoke
```

```yaml
# RAG config
experiment:
  id: rag26-competition-rag-gpt-sol-two-topic-smoke
  output_dir: outputs/rag26-competition-rag-gpt-sol-two-topic-smoke
  mode: create
  topic_ids: [rag2026-0, rag2026-1]

inputs:
  handoff_manifest: outputs/facet-deepseek-selected-evidence-two-topic-smoke/generation_handoff_manifest.json
```

Run retrieval with repeated selectors, then start generation only after the
smoke handoff exists:

```bash
PYTHONPATH=code .venv/bin/python-rocm -m trec_rag.competition_retrieval \
  configs/local/two-topic-passage-v2.yaml \
  --topic rag2026-0 --topic rag2026-1

.venv/bin/python -m trec_rag.competition_rag \
  --config configs/local/rag26-competition-rag-gpt-sol-two-topic-v2.yaml
```

Before starting smoke retrieval, verify: two selected topics;
`execution.topic_workers: 2`; up to 1,000 organizer documents per focused
query; Mixedbread scoring over every returned non-empty document; top 100
passages per query; one SQLite ledger per topic; expected organizer, reranker,
and canonical-cache reuse; the smoke-only output roots; and zero hosted
generation calls before the generation command.

Manifest order remains authoritative even if `experiment.topic_ids` is listed
in another order. `experiment.mode: create` refuses existing generation state;
`resume` reuses a topic only when the handoff, selected context, rendered prompt,
prompt contract, semantic-attempt policy, and model settings match; `overwrite`
removes only that generation JSONL and its dedicated `work/` directory. It never
removes or rewrites retrieval artifacts or the handoff.

Each topic gets at most two semantic attempts. Parsed provider responses are
stored only after recursive secret redaction. Opaque non-JSON bodies are stored
as status, byte length, and SHA-256, never verbatim. Run the offline contract
tests with:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_generation_handoff.py \
  code/tests/test_competition_rag.py
```

## Private post-run competition debug report

`trec_rag.competition_debug_report` explains an already completed competition
run from its standard retrieval config and, optionally, its matching standard
RAG config. A retrieval-only report uses:

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml
```

Include validated final answers by supplying the RAG config rather than a raw
output path:

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --rag-config configs/rag26_competition_rag_gpt_sol_v2.yaml
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
one request start per origin every six seconds with burst `1`. The hosted index throttled a run already paced at three seconds on a single serialized connection, so its own limit is stricter than ours was. The per-host
limiter is stored in `cache/retrieval/pyserini_remote/rate-limit.sqlite`, so
cooperating processes and restarts share the budget. Override the interval or
state location with `PYSERINI_MIN_INTERVAL_SECONDS` and
`PYSERINI_LIMITER_STATE_PATH`; burst values other than `1` are rejected.

Only an explicit throttle latches the shared budget. Any other transport
failure is appended to the ledger as a `failed` event and raised to its caller,
because gating every later query behind a ticket bound to one query wedges the
retriever until somebody replays a query they may not know.

A pending ticket blocks all other queries by design, so recovering never
requires remembering what was in flight:

```bash
.venv/bin/python -m trec_rag.continuation            # what is pending, and when it may retry
.venv/bin/python -m trec_rag.continuation --resume   # reissue it; makes one hosted call
.venv/bin/python -m trec_rag.continuation --discard  # drop it without a hosted call
```

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

## Ledger-driven submission selection

`trec_rag.deepagent_submission` decides which documents a retrieval run
submits. The task asks for "all and only" the documents that are relevant and
useful as evidence, ranked by usefulness, with k chosen per narrative.

**Inputs.** An `EvidenceCoverageReport`. **Outputs.** `rank_for_submission`
returns `DocumentUsefulness` rows, most useful first; `submission_rows` renders
TREC run rows; `selection_summary` says what the set leans on.

**Why the ledger and not the reranker score.** A score says a document looks
on-topic; the ledger says it actually supplied evidence an agent cited.
Measured on topic 224, restricted to documents the qrels cover: 38% of ledger
documents were graded "answers the question" against a 13% base rate, and
**none** was graded irrelevant. Documents behind a vital nugget reached 43%.

Ranking weights drafted evidence above vital, vital above need breadth, and
need breadth above raw nugget count: reaching an answer is proof of usefulness,
vital is a prediction of it, and one document can supply many near-identical
claims. Superseded nuggets stop vouching for their documents. Ties break by
document id, so the same ledger always produces the same run file.

**What is deliberately not done.** No cutoff is tuned against development
qrels. On the measured run 90% of the agent's documents were outside the
judgement pool, so a set-based score there measures pooling coverage, not
selection quality. `max_documents` exists as an operator valve, not a default;
unset, the run is exactly the documents that supplied evidence.

**Unvalidated.** The *selection* has a measured lift. The *ordering within* it
does not: the top 25 of a real 215-document ledger contained only 2 judged
documents, which cannot show whether the ranking concentrates relevance.

## Passage-first retrieval selection

`trec_rag.topic_passage_search` is the single retrieval-and-passage policy used
by both fixed facet retrieval and DeepAgent retrieval. For every focused query
it:

1. requests retrieval depth 1,000 and retains every returned source document;
2. stores each document body once in the content-addressed document store;
3. chunks and Mixedbread-scores passages from all returned documents; and
4. returns the global top 100 passages ordered by raw logit, source-document
   rank, and passage ID.

There is no hidden depth-100 scoring cutoff, top-16 agent cutoff, per-document
cap, or second diversity selector. The result retains every returned
source-document record, the scored-passage count, source offsets and hashes,
and an explicit
`complete` or `incomplete` stopping reason. Missing corpus data, organizer
failure, and scoring failure are persisted as honest incomplete results rather
than silently replaced by another retrieval path.

`trec_rag.deepagent_passages` now only groups already ranked `SourcePassage`
rows for handle metadata. It never retrieves, scores, reranks, or truncates.
`code/tests/test_retrieval_path_parity.py` guards the shared policy boundary;
`code/tests/test_topic_passage_search.py` covers ordering, all-document
scoring, retries, cache identity, and incomplete outcomes.

Document bodies are content-addressed and reused across topics. Passage scores
are cached by the exact query, text, model, revision, runtime, dtype, batch, and
chunker identity. A new facet query still has to score its own query/passage
pairs, while repeated or resumed work reuses validated cache entries.

## Experimental Deep Agent retrieval SDK

`trec_rag.deepagent_retrieval` is an experimental agentic coordinator over the
same topic-scoped passage search and records ledger used by fixed retrieval. It
is not selected by the fixed competition CLI; an agentic experiment must
construct it explicitly.

Pass an already constructed `TopicPassageSearch` and a `TopicRecordsBuilder`
for the same topic. The first search uses the untouched narrative without
rewriting it, and every search/handoff/completion update is written to that
topic's ledger:

```python
from trec_rag.deepagent_retrieval import DeepAgentRetriever

agent = DeepAgentRetriever.from_env(passage_search=topic_passage_search)
result = agent.retrieve(topic_records_builder, provided_narrative)
for candidate in result.candidates:
    print(candidate.rank, candidate.docid, candidate.score)
```

The passage search topic ID must equal the records-builder topic ID. The
builder also carries the run ID, so one topic's evidence cannot be committed to
another topic or experiment. The caller owns final `TopicRecordsBuilder`
publication after retrieval.

Topic-file lookup remains deliberately separate and always requires an
explicit path. It is a convenience for experiments, not a second SDK input:

```python
from pathlib import Path

from trec_rag.topics import load_topic_narrative

provided_narrative = load_topic_narrative(
    "224",
    Path("trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"),
)
```

Inputs and configuration:

- `OPENROUTER_API_KEY` is required by `DeepAgentRetriever.from_env`.
  The supplied passage-search adapter owns organizer credentials and validated
  retrieval/reranking caches. `PHOENIX_COLLECTOR_ENDPOINT` is required only
  when exporting Phoenix traces; omit it to disable tracing.
- Optional: `PHOENIX_API_KEY` is required by Phoenix Cloud endpoints, and
  `PHOENIX_PROJECT_NAME` overrides the default
  `trec-rag-deepagent-retrieval` project. `DEEPAGENT_MODEL` overrides the
  default `openrouter:deepseek/deepseek-v4-flash` model.
- The model specification must be `openrouter:<model-id>`. The SDK constructs
  the OpenRouter client with `max_retries=0`; it does not silently retry model,
  Pyserini, or Phoenix calls.

The SDK first runs the untouched narrative through the shared depth-1,000,
top-100-passage search, then the main coordinator delegates bounded research
tasks. Compact document metadata shown to a coordinator is capped separately
from the passage result. Final document candidates are document-ID deduplicated
with deterministic reciprocal-rank fusion (RRF, `k=60`) and return at most
twenty candidates; raw BM25 scores from different queries are never compared.

| Role | Available capabilities | Deliberately unavailable |
| --- | --- | --- |
| Main coordinator | `task`, compact retrieval-state read/update, round completion, `complete_retrieval`, `read_file` for automatic spill | Direct search/snippet tools, recursive general-purpose subagents, filesystem mutation, `write_todos` |
| Researcher | ClimbMix search, bounded snippet extraction, compact state read, `read_file` for automatic spill | `task`, semantic state updates, filesystem mutation, `write_todos` |

There is one restricted `researcher` subagent type and no default
general-purpose subagent. `write_todos` is intentionally not installed: the
need/facet/nugget state and immutable result are the retrieval workflow's
auditable state, not a general task tracker.

### Fixed researcher budget

The following invocation-local limits are fixed defaults. Pass a
`ResearchBudgetConfig` to the constructor or `from_env` only for a deliberate
POC experiment; they are not environment variables.

| Limit | Default |
| --- | ---: |
| Researcher invocations / concurrent researchers | 10 / 3 |
| Combined researcher search + snippet attempts | 100 |
| Tool calls / searches / snippets per researcher | 20 / 8 / 16 |
| Model calls, researcher / main coordinator | 30 / 40 |
| Soft warning / hard admission deadline | 30 min / 60 min |
| Consecutive no-yield calls per researcher | 3 |
| Consecutive no-progress rounds | 2 |

The original narrative is never rewritten for its deterministic first search.
Each researcher receives compact task JSON containing its task ID, round,
depth, motivating need IDs, and gap; it can formulate and refine its own
queries. The JSON envelope must begin `task.description`; optional detailed
research instructions can follow after a newline. Its first model action is
mechanically restricted to
`search_passages`; after that attempt, snippet and compact-state tools become
available. The coordinator merges completed evidence bundles with one semantic
state update per batch, then closes the round. An empty round is refused without
consuming the round, and the next coordinator action is mechanically restricted
to a researcher `task`. Once a round's researchers have all finished, the
coordinator cannot skip or abandon that round: its next turn is restricted to
one `update_retrieval_state` merge and the turn after that to
`complete_research_round`, so a round with completed research is always recorded
before the run can end. That merge delta must also carry `set_need_status` rows
for every need whose evidence changed, so a need never stays `unaddressed` after
its researchers returned. Closing a round out of order is refused with
`ROUND_SEQUENCE_INVALID` rather than raising. A researcher that has three
successive retrieval calls with no novel evidence must return its bundle; two
successive rounds with no accepted coverage progress stop further research.
Rounds have no cap of their own: a round cannot close without a finished
researcher, so researcher invocations already bound them and a separate limit
could only strand researchers the run was allowed to spend. When research ends,
the coordinator gets one final directed turn to record a synthesis: a draft
answer and grounded nugget IDs for every need the evidence supports. It must
then call `complete_retrieval`; this is the only coordinator tool that can make
the coverage state terminal. The transition checks every need with live,
non-superseded evidence and requires at least one selected live
`draft_nugget_id`. A rejected completion returns deterministic `need_ids` for
the open needs, and a draft selection whose nugget was later superseded no
longer counts.

`AgentRetrievalResult` contains the input `narrative`, completed `searches`
(`AgentSearch` records), fused `candidates` (`RankedCandidate` records with
per-search provenance), the agent's `rationale`, `stopping_reason`, immutable
`coverage_report`, immutable `budget_snapshot`, and immutable
`trace_flush_succeeded`. Its `topic_snapshot` is the one holistic ledger view
taken after researcher handoffs and completion state have been persisted.
Inspect the coverage and budget through the Python SDK when deciding whether
the evidence is ready for a downstream draft:

```python
coverage = result.coverage_report
print(coverage.state_hash, coverage.unresolved_need_ids)
print(result.budget_snapshot.as_dict())
for need in coverage.needs:
    print(need.need_id, need.status, need.draft_answer)
```

`trace_flush_succeeded` is `True` when configured tracing flushes successfully
or tracing is disabled and no export is required; it is `False` when export
fails. Export failure does not retry or change retrieval. Budget exhaustion is
an honest grounded partial result: completed searches, snippets, actions, and
coverage gaps are returned, but exhaustion never claims coverage is complete.
When a coverage terminal reason and a budget stop coexist, `stopping_reason`
retains the coverage reason; inspect `budget_snapshot.stop_code` for the
independent budget outcome. Reaching the coordinator's `max_main_models`
ceiling records `MAIN_MODEL_BUDGET_EXHAUSTED`, so a run cut short by that
ceiling reports `budget_exhausted` rather than claiming `agent_completed`. A search that cannot reach the index
reports `RETRIEVAL_UNAVAILABLE` to the agent and makes `stopping_reason`
`retrieval_unavailable`, so an outage is never read as a narrative with no
evidence. Provider, deadline, and retrieval failures preserve all evidence
already admitted and seal the ledger `incomplete`; downstream generation may
still consume that explicit partial result. There is no alternate fallback
retriever. Before completion, every live coverage citation is compared with
the passages actually admitted through durable researcher handoffs. A mismatch
seals `incomplete/evidence_validation_failed`; it can never be reported as
complete.

The agent maintains three distinct stores, each with a different job:

- The invocation-local **need map** records narrative-derived needs, facets,
  remaining gaps, statuses, and any draft answer.
- The mechanical **retrieval ledger** records searches, inspected pages,
  document/focus pagination state, residual signals, and consumed actions.
- The grounded **nugget store** holds only concise claims linked to
  resolved citations into returned snippets. It stores the cited span, not a
  copy of the passage, so a reported quote cannot disagree with its snippet.

Use `view_retrieval_state` to inspect a compact frontier or a bounded state
view and `update_retrieval_state` to add needs, facets, nuggets, evidence, and
coverage judgments. Researcher search and snippet calls atomically record their
own actual arguments; they do not require `choose_next_action` authorization.
The legacy `choose_next_action` callback is retained for compatibility with
older injected toolsets, but it cannot record a terminal stop;
`complete_retrieval` is the only valid terminal transition. The coordinator's
semantic coverage state is invocation-local. Accepted searches, facets,
passages, researcher handoffs, and final completion are also projected into the
persistent per-topic `TopicRecordsBuilder`; that sealed topic ledger is the
durable handoff boundary.

`update_retrieval_state(delta)` is the universal append/update entry point for
the three stores. Its model-facing delta accepts these eight optional lists:

| Delta section | Row input |
| --- | --- |
| `add_needs` | `need_id`, exact `narrative_span`, `question` |
| `add_facets` | `facet_id`, `need_ids`, `dimension`, `value`, `origin`, optional `origin_snippet_id` |
| `add_nuggets` | `nugget_id`, `text`, `need_ids`, `facet_ids`, `evidence` as a list of `{"cite": "S3.2"}`, optional `contradicts` |
| `add_evidence` | `nugget_id`, `cite` |
| `set_facet_status` | `facet_id`, `status`, optional `status_reason` and `supporting_nugget_ids` |
| `set_need_status` | `need_id`, `status`, `remaining_gap`, optional `draft_answer` and `draft_nugget_ids` |
| `supersede_nuggets` | `nugget_id`, `superseded_by` |
| `abandon_documents` | `document_id`, `reason` |

The result reports `accepted_ids`, row-level `rejected` entries,
`state_version`, and `state_hash`. An unknown section is reported as
`UNKNOWN_SECTION` without discarding valid rows in the same delta. A delta with
no accepted rows and no other rejection is reported as `EMPTY_DELTA`, including
an empty recognized list such as `{"add_needs": []}`.

Any rejection also adds a `rejected_summary` count by code. When a citation
fails to resolve (`UNKNOWN_CITATION`, `INVALID_CITATION`, `DUPLICATE_EVIDENCE`),
the result adds `unadmitted_sections` and an `evidence_rejection_notice` stating
that the claim never entered the ledger, that the cited needs remain
unsupported, and that the only recovery is delegating a researcher again.

Agent-facing search and snippet behavior is deliberately narrow:

- Search results visible to the agent contain only document ID, rank, score,
  and text length metadata. No title field is emitted because the endpoint does
  not return titles.
- For any document returned during the current invocation, the agent can ask
  for up to ten relevance-ranked snippets at a time and follow `next_cursor`
  for another page. A page may contain multiple snippets from the same
  document. One bounded snippet may equal a complete short document that fits
  in one chunk; long documents remain relevance-chunked and paginated and are
  never blindly injected whole. Each page reports `page_index`,
  `residual_count`, `residual_top_score`, `returned_min_score`, and
  `pages_estimated`; these scores describe continuity only within that page's
  ranking, never calibrated relevance or comparability across documents or
  focus queries.
- Agents cite evidence by handle and never transcribe it. Each returned
  snippet carries a handle such as `S3` and numbered sentences; evidence is
  `S3` for a whole snippet, `S3.2` for one sentence, or `S3.2-4` for a
  contiguous range. The SDK resolves the handle against stored sentence spans
  and derives the reported `document_id`, `snippet_id`, `page_index`, and
  `quote`, so an ungrounded quote is not a reachable outcome. Handles are
  assigned once per invocation and stay valid across pages and rounds. A need
  becomes `answerable` only with a nonblank
  draft answer and grounded nugget IDs; otherwise it remains unaddressed,
  partial, or conflicted. A nugget's `single_document` or `multi_document`
  support label describes the observed support count only; it does not
  establish source independence.
- The search transport and snippet extractor own their respective caches. The
  model receives no cache keys, paths, bypass switches, or other cache controls.
- Deep Agents may spill oversized tool results or temporary notes into
  invocation-local state scratch. Need-map and ledger state, caches, and
  scratch are separate: state is discarded after the retrieval invocation,
  cache-first search/snippet tools retain their own validated reusable results,
  and scratch is only a temporary overflow for the current invocation.

The reusable `trec_rag.deepagent_snippets` layer accepts a document ID, complete
document text, focus query, and optional cursor. It returns a typed
`SnippetExtractionResult` containing the model-visible `SnippetPage` plus
internal cache status, ranker backend, and page offset used by tracing. Its
default factory uses the repository cache roots; experiments can instead inject
an explicit extractor or ranker without changing the agent-facing tool schema.
The default Mixedbread adapter loads with `local_files_only=True`, so the pinned
`mixedbread-ai/mxbai-rerank-base-v2` revision
`3ea9d4dffa7d12a4f366be8e275c349de9fc9865` must already be present in the
local Hugging Face cache before the first uncached snippet-ranking call.

The reusable layer validates its boundary before returning or caching a page:

- document IDs and focus queries must be nonblank, document text must be a
  string, and cursors must be strings or `None`;
- opaque cursors are schema-checked and bound to the document, focus query,
  document-text hash, ranker/chunker identity, extraction configuration, and
  next page offset;
- page size and chunk maximum must be positive integers, overlap must be
  non-negative and smaller than the chunk maximum, and the duplicate-overlap
  ratio must be a finite value from zero through one;
- chunk IDs/ranges/text must match the source document, ranker outputs must
  cover every known chunk exactly once, and every relevance score must be
  finite;
- cached identities, response and private source-manifest digests and bindings,
  page fields, snippet records, offsets, and continuation cursors are validated
  before reuse.

Malformed or identity-mismatched cache data raises the constant,
non-disclosing `invalid snippet cache entry` integrity error. At the agent tool
boundary, unknown documents, blank queries, invalid cursors, and extraction
failures likewise return fixed errors without internal cache, model, or path
details.

The retrieval presentation bounds are explicit keyword-only constructor and
`from_env` options. They accept positive integers only (booleans are rejected)
and are not environment variables:

```python
retriever = DeepAgentRetriever.from_env(
    hits_per_search=10,
    max_followup_searches=8,
    fused_result_limit=20,
)
result = retriever.retrieve(provided_narrative)
```

`hits_per_search` controls the remote request depth and model candidate view;
trace evidence is additionally subject to an absolute safety ceiling.
`max_followup_searches` controls the per-researcher search cap when no explicit
`ResearchBudgetConfig` is supplied, and `fused_result_limit` controls the
deterministic final RRF depth. Defaults are 10, 8, and 20 respectively. Run
live retrieval or local snippet/reranker work on this ROCm host with
`.venv/bin/python-rocm`, rather than `uv run` or an unconfigured interpreter.

Phoenix tracing is optional. Search spans contain document lengths rather than
document text. Snippet-page spans contain bounded IDs, offsets, relevance
scores, backend/cache metadata, and snippet text only when `trace_content=True`;
with `trace_content=False`, narrative, query, snippet text, and automatically
instrumented LangChain inputs/outputs use a redacted marker. With the default
`trace_content=True`, Phoenix shows automatic agent, model, and tool
inputs/outputs for interactive debugging, including snippet-tool arguments and
results. Purpose-specific manual spans still exclude credentials,
authorization headers, raw provider responses, continuation-ticket values,
cursors, scratch paths, and local cache paths. Keep provider credentials in
ignored local environment files rather than source or notebooks. Tracing
configuration is process-global and idempotent:
identical normalized live setup reuses its provider/exporter, while a
conflicting endpoint, project, credential, injected provider, or content mode
raises a constant non-disclosing configuration error instead of silently
reconfiguring instrumentation.

Validate the isolated SDK and its existing transport boundaries with:

```bash
.venv/bin/python -m pytest \
  code/tests/test_topics.py \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_pipeline.py \
  code/tests/test_remote_pyserini.py -q
```

## Offline post-batch nuggetizer probe

`post_batch_nuggetizer_probe.py` is an offline proof-of-concept for the
centralized canonicalization design. It is not wired into
`DeepAgentRetriever`, `run_pipeline`, or any submission artifact.

Inputs:

- one Phoenix trace for topic `224`, loaded through the configured Phoenix
  client;
- the trace's researcher bundles and accepted ledger, selected with
  `--input-source researcher` or `--input-source ledger`;
- the untouched narrative, snippet observations, and exact citation handles
  reconstructed in memory from the trace.

Outputs:

- a sanitized A/B comparison of provisional claims and canonical nuggets;
- grounding failures, alias mappings, and summary counts printed to stdout;
- no trace content, raw snippets, provider response, or cache files written to
  disk.

Validation and safety bounds:

- the trace reconstruction rejects missing or conflicting identities and
  citation observations before a hosted call;
- exact snippet grounding is checked before and after canonicalization;
- at most one hosted canonicalization call is made, with no retries;
- the existing canonical-nugget and nuggetizer-adapter tests remain the
  reusable validation boundary.

The implementation is [post_batch_nuggetizer_probe.py](post_batch_nuggetizer_probe.py).
Run it only as a deliberate local experiment after loading the Phoenix and
OpenRouter environment from ignored files; it is not a pipeline entry point.

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

## Answer-quality evaluation

`ragdoll_io.py` and `dev_rag_inputs.py` support scoring generated answers with the organizer
harness, tracked as the `ragdoll` submodule. Neither is on the competition path.

### `ragdoll_io.py`

Derives RAGDoll inputs from a published submission JSONL, writing sidecar files rather than
editing the submission. RAGDoll resolves a topic id only from a top-level `qid`, while the
organizer contract requires exactly the `metadata`/`references`/`answer` root keys, so an
unadapted submission joins to nothing and reports empty metrics instead of failing.

```bash
.venv/bin/python -m trec_rag.ragdoll_io \
  --submission outputs/<id>/rag_output_trec_rag_2026.jsonl \
  --topics trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
  --gold-nuggets trec-rag-data/trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl \
  --answers-out <dir>/answers.jsonl --nuggets-out <dir>/nuggets.jsonl \
  --handoff-manifest outputs/<retrieval-id>/generation_handoff_manifest.json \
  --generation-identity outputs/<rag-id>/work/generation_identity.json \
  --support-out <dir>/support_input.jsonl
```

Outputs: an answers file and reshaped gold nuggets for `ragdoll nuggetizer eval`, and
optionally resolved rows for `ragdoll support judge`. Validation: the answers/nuggets join must
be non-empty, every submission narrative must equal the authoritative `--topics` narrative,
every cited docid must resolve to text, and a citation naming a document outside `references`
is rejected because RAGDoll would silently drop it from the metric denominator.

Support evaluation accepts only selected-evidence v2 inputs. `--handoff-manifest` resolves each
citation to all exact selected passages shown for that topic/docid, in authenticated evidence
order. `--generation-identity` then proves that the handoff digest, prompt contract, run ID, and
topic contexts are the ones recorded by this answer-generation run. The adapter does not accept
a retrieval document archive or a word limit, so a full-document head cannot be substituted or
silently truncated. Judging against that wrong view once inflated No Support and inverted the
conclusion; those evaluation artifacts were discarded.

After `ragdoll support judge` finishes, rerun the same adapter command with
`--support-judgments <dir>/support_judgments.jsonl`. It fails closed unless the output contains
exactly one completed `FS`, `PS`, or `NS` judgment for every expected citation and each judgment
still matches the input statement, selected evidence, run, topic, and document.

### `dev_rag_inputs.py`

Builds a TREC run and document JSONL for development topics from the archived Pyserini
responses under `cache/retrieval/pyserini_remote`. Those archives predate the retriever
provenance sidecar, so they cannot be replayed through `trec_rag.pipeline`; this module reads
them as an archive and never writes a sidecar, because synthesizing provenance would defeat
that check. The emitted ordering is therefore raw BM25, not reranked.

```bash
.venv/bin/python -m trec_rag.dev_rag_inputs \
  --topic 58 --topic 213 --cache-dir cache/retrieval/pyserini_remote \
  --run-path <dir>/run.tsv --documents-path <dir>/documents.jsonl \
  --run-id <id> --depth 100 [--passage-words 1000]
```

Outputs: a six-column TREC run and one document row per topic, both shaped for
`trec_rag.competition_rag`. With `--passage-words`, each document is reduced to its most
query-relevant chunks via `trec_rag.chunking.SemanticTextChunker` instead of being left for the
generator to head-truncate. Validation: ranks are dense and scores non-increasing so
`load_trec_run` accepts the output, and duplicate docids are collapsed to their best rank.

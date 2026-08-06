# Handover: `claude/cloud-topics-artifact-download-hy5osu` (PR #39)

Written 2026-08-05 by a Claude Code cloud session, for continuing on the local ROCm host.

Goal of the branch: make it possible to run a **subset of competition retrieval topics on a
rented GPU box**, download the artifact, and finish the run locally.

## State

- Branch `claude/cloud-topics-artifact-download-hy5osu`, pushed, working tree clean. PR #39 OPEN,
  `mergeable_state: clean`. No CI — the repo has no test workflows, only the dynamic Copilot
  reviewer agent. For the exact commit list, read `git log origin/master..` rather than trusting
  a count written here; in order, the work was:
  - CUDA dependency group, `setup_cuda_env.sh`, device pin, docs
  - both offline models prefetched, new experiment namespace (review round 1)
  - `verify_torch_groups.sh` and the hermetic bootstrap contract tests (review round 2)
  - this handover, plus any later follow-ups
- Owner review, three rounds. All review findings are addressed. The local ROCm host regenerated
  `uv.lock`, and the strict non-installing verifier accepts both resolution forks — see below.
- Tests, full suite — **pre-follow-up evidence, measured at `eab10ce`** ("Bind the bootstrap
  contract to the production loaders"), before the uncommitted `verify_torch_groups.sh` /
  `test_verify_torch_groups.py` follow-up landed on top of it: **1870 passed**, plus 15 failures
  and 5 collection errors that are all `No module named 'numpy'` / missing `transformers` metadata
  in the cloud container. Identical before and after the diff. The owner confirmed
  `test_evidence_local_contract.py` passes 4/4 on a host with the ML stack, so those are
  environmental. **The full suite has not been rerun since that revision.** Worth running it once
  on a venv that has both the ML stack and `deepagents`/`langchain`/OpenTelemetry — no single
  environment has covered all of it yet.
  ```bash
  .venv/bin/python -m pytest -q
  ```
- Tests, current targeted result — independently verified on the working tree carrying the
  verifier follow-up: **36 passed** for the verifier/setup contract files and **340 passed** for
  the broader retrieval/RAG/setup/verifier slice, alongside a clean `bash -n` on
  `verify_torch_groups.sh` and a clean `git diff --check`.
  ```bash
  .venv/bin/python -m pytest -q \
    code/tests/test_competition_retrieval_v2.py \
    code/tests/test_competition_topic_dispatch.py \
    code/tests/test_retrieval_export.py \
    code/tests/test_competition_rag.py \
    code/tests/test_env_setup_contract.py \
    code/tests/test_verify_torch_groups.py
  ```

## `uv.lock`: regenerated and verified

`pyproject.toml` gained a `cuda` dependency group and a `[tool.uv] conflicts` declaration. The
cloud session could not regenerate `uv.lock`: `repo.radeon.com` was denied by that environment's
egress policy (403 on CONNECT), and the conflicts declaration forces the ROCm fork to re-resolve.
The local ROCm host completed `uv lock` in 237 ms from its existing uv cache. The committed lock
now carries both `torch==2.9.1` from PyPI with its `nvidia-*` runtime dependencies and the pinned
ROCm Torch/Triton URLs.

Acceptance verification on that regenerated lock:

```bash
./code/tools/verify_torch_groups.sh
OK   uv.lock is current with pyproject.toml
OK   rocm: torch @ https://repo.radeon.com/rocm/.../torch-2.9.1%2Brocm7.2.1...whl
OK   cuda: torch==2.9.1 from https://pypi.org/simple
Both hardware groups select their intended torch source.
```

`verify_torch_groups.sh` installs nothing and writes no shared cache: `uv lock --check` only
compares a fresh resolution against the committed lock, `uv export --frozen` reads the lock alone,
and both run with `--no-cache` and a throwaway `--cache-dir` removed on exit, so the shared
persistent uv cache and every model, retrieval, and reranker cache are untouched. Both also run
with `--no-python-downloads`, which is the part of "installs nothing" that the cache flags do not
cover: this repo pins Python 3.12.13, and on a host lacking it `uv lock` would provision a managed
interpreter just to resolve. With the flag such a host fails loudly instead of silently installing
one. It is safe to run beside an active task. It does need package-index access, because
`uv lock --check` re-resolves.

The CUDA line is checked strictly: exactly `torch==2.9.1`, no direct URL, no CPU wheel, plus
`nvidia-cuda-runtime`/`nvidia-cublas`/`nvidia-cudnn` present in the export as evidence the resolved
wheel is GPU-enabled. An actual CUDA-host sync and CUDA probe remain part of provisioning the
rented NVIDIA host; they were not run against the local ROCm environment.

The source claim needed a second export, because the requirements format cannot carry one: it
renders every registry package as `name==version`, so a mirror or a private index would print the
same `torch==2.9.1` as PyPI. The script therefore also runs
`uv export --frozen --group cuda --format pylock.toml` — PEP 751 metadata, which records the index
per package — and requires the one `torch` entry to be version `2.9.1` with index exactly
`https://pypi.org/simple`. Scoping that export to the group is the point; an unscoped multi-fork
`uv.lock` block would not say which fork the cuda resolution took. The extra call is still
non-installing and still cache-isolated.

One diagnostic nuance worth keeping: the "regenerate the lock" hint prints only when the lock is
actually diagnosed as stale. An unsupported flag, a rejected credential, or a dead network says
nothing about currency, and sending the reader to re-resolve against an index they cannot reach
would be advice that cannot work.

Mechanism note, because the review and the original PR description disagreed: `--locked`
**re-resolves** in order to decide whether the lock is current, and that re-resolution pulls the
ROCm URL requirements even when only `--group cuda` is requested. So against a stale lock the
real sequence is fetch-then-reject-as-stale, not a silent ROCm install. Verified by running it.

The real forked lock is now committed and verified for freshness, exact Torch sources, and NVIDIA
runtime dependencies. Hardware execution still needs the rented NVIDIA host.

## What this branch does

**1. A CUDA install path, which the repo never had.** `torch`/`sentence-transformers`/
`transformers`/`numpy` previously lived only in the `rocm` group behind hardcoded
`repo.radeon.com` wheel URLs; every other host fell through to plain `uv sync`, which installs no
torch at all, and reranking died with "sentence-transformers is required". The new `cuda` group
pins `torch==2.9.1` (PyPI CUDA build), `sentence-transformers==5.6.0`, `transformers==5.13.0`,
`numpy==2.5.1` — the same versions the ROCm group resolves to and the Modal image in
`code/tools/modal_rerank_score_cache.py` pins, so `DEFAULT_BACKEND_VERSION` stays honest and
reranker score-cache entries stay interchangeable across hosts.

**2. `code/tools/setup_cuda_env.sh`.** Syncs that group, prefetches **both** models the
production path loads with `local_files_only=True` — `mxbai-rerank-base-v2` for passage scoring
and `all-MiniLM-L6-v2` for evidence selection — verifies each resolves from cache alone, and
probes CUDA. `setup_env.sh` now dispatches ROCm → NVIDIA → plain. Caching only the reranker is the
trap: selection runs after scoring, so a missing MiniLM fails late with the expensive stage paid.

`code/tests/test_env_setup_contract.py` guards this hermetically — no network, no torch needed,
and no write to any shared persistent cache. It executes the script's prefetch block against a
stubbed `huggingface_hub`, and separately drives the two production loaders with stub loaders to
assert the set they request equals the set the script prefetches. Drift in either direction fails.
Mutation-checked: deleting the MiniLM line from the script fails four of the eight tests.

Precise about "no cache writes", because the two are not the same thing: the suite does create a
throwaway SQLite score-cache database (constructing `MixedbreadPassageScorer` opens one eagerly),
but only under pytest's per-test `tmp_path`. Nothing lands in the shared persistent caches under
`cache/` — `cache/reranker/`, `cache/retrieval/`, `cache/documents/` — nor in the Hugging Face
model cache or the shared uv cache, so running it beside a live pipeline task cannot clobber a
real score database.

`code/tests/test_verify_torch_groups.py` covers `verify_torch_groups.sh` the same way: it runs the
real script with a stub `uv` first on `PATH` and a `TMPDIR` redirected into `tmp_path`, so no
resolution, download, or install happens. It pins the behaviours the review rounds asked for — lock
currency proved non-installingly, `--no-cache` plus a temporary `--cache-dir` plus
`--no-python-downloads` on every uv call, the strict CUDA acceptance (exact `torch==2.9.1`, NVIDIA runtime deps present, CPU URLs and wrong
versions rejected), the PyPI-source proof read from the group-scoped `pylock.toml` export
(including the alternate-registry false positive that the requirement line alone cannot catch),
uv's own error surfaced verbatim when the failure is unrelated to the dependency group, and the
regenerate-lock hint withheld for unsupported-flag, authentication, and network failures. The
missing-`uv` case runs with an empty `PATH`, which the script survives because it locates itself
with shell builtins instead of `dirname` — otherwise coreutils would fail first and swallow the
diagnostic.

**3. `passage.device: cuda`, replacing `auto`.** The resolved device string is part of the sealed
passage-search identity, so `auto` is not portable across hosts (see findings).

**4. New experiment namespace.** The device pin changes the sealed config hash, so the retrieval
config moves to `facet-deepseek-b40-v3` and both RAG configs' handoff paths follow.
`facet-deepseek-b40-v2` checkpoints are preserved and simply unused.

**5. Docs.** "Splitting one run across hosts" in `code/trec_rag/README.md`; `AGENTS.md`
environment bullet updated.

## Key findings worth not relitigating

These came out of reading the pipeline and are the reason the diff looks the way it does.

- **The downloadable artifact is two directories, not one.** `document_store_dir()` resolves to
  `cache/documents/v1`, *outside* `outputs/`, and `retrieval_export.py:2897` reads every row's
  full text from it. Copying only `outputs/<id>/` gives you checkpoints that validate and an
  export that then fails on missing text. This is the quiet failure mode.
- **Do not copy `cache/reranker/`.** The score DB filename is derived from the scoring context
  alone, so both hosts produce the *same* path with different contents and a copy clobbers local
  scores. Unnecessary anyway: scores are already baked into each topic's sealed records.
- **`_choose_device` is in the sealed identity.** `_configured_passage_search_identity`
  (`competition_retrieval.py:1874`) embeds it, and `_validate_scoring_manifest`
  (`retrieval_export.py:2678`) recomputes it on the *validating* host and demands exact dict
  equality. `test_passage_builder_rejects_scorer_on_different_resolved_auto_device` locks this in.
  ROCm reports `cuda`, so ROCm↔NVIDIA agree; a CPU-only box does not.
- **Design tension, deliberately left alone.** The score cache *excludes* device
  (`device_family` stays `"unspecified"` for the competition scorer), so the cache treats CPU and
  GPU scores as interchangeable while the checkpoint treats them as incompatible. Pinning the
  device sidesteps it rather than resolving it.
- **Both hosts must agree on:** byte-identical config YAML (its SHA-256 is sealed into every
  dispatch and projection receipt), `INDEX_URL` (required by `retrievers.py:249` even for a
  resume-and-export pass that issues no queries), a clean tracked worktree with real git metadata
  (`git status --porcelain` and `git rev-parse HEAD` both run), and no commit change *inside* a
  topic (`retrieval_export.py:1519`). Different topics may carry different commits —
  `source_code_commits` is a set.
- **Never overlap topic sets.** Decomposition is a live LLM call, so two hosts running the same
  topic ID produce different sealed decompositions and merging becomes an arbitrary pick.
  Partition with `--topic-subset`.
- **No manual merging of run files.** `export_retrieval_run` regenerates the TREC TSV, full-text
  ZIP, and handoff over whatever topics are selected. Drop the remote topic dirs in, run locally
  over the full set, and sealed topics revalidate and are skipped.
- **The RAG stage is already portable.** `competition_rag` reads only the handoff manifest, whose
  evidence text is inline. No document store, no GPU.

## Suggested next steps

1. Run the full test suite with the ML stack present.
2. Provision an NVIDIA box (the local `dstack` skill is the obvious route) and bootstrap it with
   `code/tools/setup_env.sh`, which now auto-detects CUDA. The box needs `INDEX_URL` byte-identical
   to the local value, `PYSERINI_API_TOKEN`, `OPENROUTER_API_KEY`, egress to
   `api.castorini.uwaterloo.ca` and `openrouter.ai`, and disk for the HF weights plus the document
   store. If the provisioning has a persistent volume, point `HF_HOME` at it — that is what the
   Modal runner does, and it keeps the pinned weights across restarts.
3. Smoke it on two topics per `code/trec_rag/README.md` before committing to a large split run.
   Full runs still need explicit authorization per `AGENTS.md`.

## Deliberately not done

- **No merge tool for competition-context reranker caches.** `rerank_cache_promotion.py` is the
  right shape — logical import, `imports` table, transactional — but it is bound to
  `PipelineConfig` and the rag25 document+window artifacts, not `FacetPilotConfig` /
  `score_kind="passage"`. Avoidable, so the README says not to copy `cache/reranker/` instead.
  Build it only if cross-host cache *reuse* becomes worth it.
- **No live run of any kind.** Nothing in this branch has been exercised against real hardware,
  the hosted index, or a provider. Every claim here is from reading the code and running tests.

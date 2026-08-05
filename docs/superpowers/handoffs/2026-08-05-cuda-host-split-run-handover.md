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
- Owner review, two rounds. Everything is addressed **except the `uv.lock` P1, which is still a
  merge blocker** — see below.
- Tests: **1870 passed**, plus 15 failures and 5 collection errors that are all
  `No module named 'numpy'` / missing `transformers` metadata in the cloud container. Identical
  before and after the diff. The owner confirmed `test_evidence_local_contract.py` passes 4/4 on a
  host with the ML stack, so those are environmental. Worth running the whole suite once on a venv
  that has both the ML stack and `deepagents`/`langchain`/OpenTelemetry — no single environment has
  covered all of it yet.
  ```bash
  .venv/bin/python -m pytest -q
  ```

## ⛔ The one open blocker: `uv.lock`

`pyproject.toml` gained a `cuda` dependency group and a `[tool.uv] conflicts` declaration, but
`uv.lock` still records only the ROCm group. **`code/tools/setup_cuda_env.sh` cannot bootstrap a
CUDA host until the lock is regenerated.** The cloud session could not do it: `repo.radeon.com`
is denied by that environment's egress policy (403 on CONNECT), and the conflicts declaration
forces the ROCm fork to re-resolve, so `uv lock` fails on the pinned ROCm `torch`/`triton` URLs.

On this host, where radeon is reachable:

```bash
uv lock
./code/tools/verify_torch_groups.sh   # the acceptance check; must print OK for both groups
uv sync --group cuda --locked         # must succeed with no repo.radeon.com fetch
uv sync --group rocm                  # confirm the ROCm env still resolves as before
```

`verify_torch_groups.sh` is read-only (`uv export --frozen` resolves from the committed lock and
never installs, downloads, or touches a cache), so it is safe to run beside an active task. Today
it prints `OK rocm: torch @ https://repo.radeon.com/...` and `FAIL cuda: uv.lock does not carry
this group`, exit 1. After regeneration the cuda line should read `OK cuda: torch==2.9.1` and the
script should exit 0. Then commit the lock and the blocker clears.

Mechanism note, because the review and the original PR description disagreed: `--locked`
**re-resolves** in order to decide whether the lock is current, and that re-resolution pulls the
ROCm URL requirements even when only `--group cuda` is requested. So against a stale lock the
real sequence is fetch-then-reject-as-stale, not a silent ROCm install. Verified by running it.

The CUDA pin set was confirmed to resolve on PyPI in isolation (58 packages, CUDA-enabled torch
with the `nvidia-*` runtime deps). The real forked lock is unverified.

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

`code/tests/test_env_setup_contract.py` guards this hermetically — no network, no cache writes, no
torch needed. It executes the script's prefetch block against a stubbed `huggingface_hub`, and
separately drives the two production loaders with stub loaders to assert the set they request
equals the set the script prefetches. Drift in either direction fails. Mutation-checked: deleting
the MiniLM line from the script fails four of the eight tests.

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

1. Regenerate and commit `uv.lock` (above). This unblocks the PR.
2. Run the full test suite with the ML stack present.
3. Provision an NVIDIA box (the local `dstack` skill is the obvious route) and bootstrap it with
   `code/tools/setup_env.sh`, which now auto-detects CUDA. The box needs `INDEX_URL` byte-identical
   to the local value, `PYSERINI_API_TOKEN`, `OPENROUTER_API_KEY`, egress to
   `api.castorini.uwaterloo.ca` and `openrouter.ai`, and disk for the HF weights plus the document
   store. If the provisioning has a persistent volume, point `HF_HOME` at it — that is what the
   Modal runner does, and it keeps the pinned weights across restarts.
4. Smoke it on two topics per `code/trec_rag/README.md` before committing to a large split run.
   Full runs still need explicit authorization per `AGENTS.md`.

## Deliberately not done

- **No merge tool for competition-context reranker caches.** `rerank_cache_promotion.py` is the
  right shape — logical import, `imports` table, transactional — but it is bound to
  `PipelineConfig` and the rag25 document+window artifacts, not `FacetPilotConfig` /
  `score_kind="passage"`. Avoidable, so the README says not to copy `cache/reranker/` instead.
  Build it only if cross-host cache *reuse* becomes worth it.
- **No live run of any kind.** Nothing in this branch has been exercised against real hardware,
  the hosted index, or a provider. Every claim here is from reading the code and running tests.

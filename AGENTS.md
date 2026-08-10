# Shared Agent Instructions

These instructions are the shared contract for Codex, Claude, and any future
coding agent working in this repository.

## Completed Project and Key Artifacts

This TREC RAG 2026 project is complete. All five tracked organizer files were
uploaded and accepted by Evalbase. Do not treat historical plans,
specifications, experiment notes, or unused configs as an active backlog, and
do not start new retrieval, generation, or evaluation work unless the user
explicitly reopens the project.

Start every orientation or handoff from these artifacts:

- [`index.html`](index.html) is the reader-facing project map.
- [`submissions/trec-rag-2026/SUBMISSION_LEDGER.md`](submissions/trec-rag-2026/SUBMISSION_LEDGER.md)
  is the control sheet for exact accepted files, priorities, hashes, and portal
  notes.
- [`reports/2026-competition-architecture.html`](reports/2026-competition-architecture.html)
  explains the final system; edit only its canonical
  [`reports/2026-competition-architecture.qmd`](reports/2026-competition-architecture.qmd)
  source and re-render with Quarto.
- The final [Retrieval bundle](submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md)
  and [RAG bundle](submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md)
  document the submitted variants and provenance.
- [`.agents/skills/validate-trec-rag-2026-submissions/SKILL.md`](.agents/skills/validate-trec-rag-2026-submissions/SKILL.md)
  is the canonical repo-local delivery preflight. It composes the official
  [track contract](trec-rag-skills/skills/trec-rag-2026-track-guidelines/SKILL.md)
  with organizer AutoJudge checks.
- [`code/trec_rag/README.md`](code/trec_rag/README.md) is the implementation and
  reproduction reference; [`reports/index.html`](reports/index.html) collects
  supporting reports.

Final analyses and their boundary:

- The tracked [Retrieval Quality Analysis](reports/2026-retrieval-nugget-coverage.html)
  is the project's separate 119-topic Retrieval coverage report.
- The accepted RAG files are the immutable evaluated inputs for two separate
  119-topic RAGDoll citation-support reports:
  - [RAG Analysis: rag26-ss1](https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html) — **Private / tailnet**;
    input: `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl`.
  - [RAG Analysis: rag26-ms1-final](https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html) — **Private / tailnet**;
    input: `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/multistage/rag_output_trec_rag_2026.jsonl`.
- These reports measure citation support, not an official TREC score. qrel/gold
  metrics are unavailable; generated claims are not gold nuggets. RAGDoll
  results did not change the accepted files or priorities—there was no priority
  influence, and the accepted files and priorities remain unchanged.
- Use [`.agents/skills/trec-rag-competition-debug-report/SKILL.md`](.agents/skills/trec-rag-competition-debug-report/SKILL.md)
  for the accepted-input evaluation contract. Keep it explicit: raw evaluation
  work stays outside git and the rendered portal. Keep work directories, caches,
  raw passages, tasks, events, judgments, assignments, and manifests private.

Treat the five organizer files under `submissions/trec-rag-2026/` as immutable
accepted artifacts. Documentation may link and verify them, but must not
regenerate, reformat, or overwrite them. Evalbase submission IDs and upload
timestamps were not recorded; preserve that unknown explicitly rather than
inventing values.

## Repository Shape

- Keep source-backed reports under `reports/`.
- Keep design notes, implementation plans, and agent process records under
  `docs/superpowers/`.
- Keep reusable implementation code under `code/`.
- Keep temporary files, downloaded scratch data, and local servers out of git.
- Prefer one shared instruction file over tool-specific behavior. If a
  tool-specific file is needed, it should point back here.

## Report Work

- Preserve source provenance. If a claim comes from a paper, overview, local PDF,
  or extracted figure, keep that provenance visible in the report.
- Keep reports approachable for first-time readers: define jargon, separate
  metrics by task, and avoid flattening different evaluation setups into one
  false leaderboard.
- For standalone HTML reports, avoid external runtime dependencies unless the
  project explicitly chooses them.
- When adding visuals, keep local source files and make image interactions usable
  without right-clicking.

## Code Work

- Put reusable scripts, parsers, and data-processing utilities in `code/`.
- Prefer small, inspectable data records between pipeline stages.
- Add README notes beside new code explaining inputs, outputs, and validation.
- Keep generated artifacts separate from reusable code.

## Environment Setup

- Use the repo-pinned Python in `.python-version`; uv will install it when
  needed.
- In linked worktrees, copy the local `.env` from the main/shared checkout when
  it is missing so endpoint tokens and local paths stay available. Keep `.env`
  out of git.
- Use one local virtual environment, `.venv/`, for the active machine.
- Run `code/tools/setup_env.sh` to set up the environment. It auto-detects AMD
  ROCm and syncs the `rocm` dependency group, or NVIDIA and syncs the `cuda`
  group; otherwise it syncs the standard project environment, which has no
  torch. The `rocm` and `cuda` groups are mutually exclusive, so `uv.lock` must
  be regenerated on a host that can reach `repo.radeon.com`.
- After setup, run Python commands through `.venv/bin/python` or
  `.venv/bin/python-rocm` instead of relying on `uv run`, because `uv run` syncs
  only uv's static default groups and does not auto-detect ROCm hardware.
- On AMD ROCm hosts, use `.venv/bin/python-rocm` for commands that need PyTorch
  GPU access. The helper only adds the detected ROCm runtime library path before
  invoking `.venv/bin/python`.
- Keep reusable cache artifacts under the repo-root `cache/` directory in the
  main/shared checkout: retrieval responses under `cache/retrieval/` and
  reranker scores under `cache/reranker/`.
- Shared cache archives should contain a top-level `cache/` directory. Restore
  them by extracting from the repo root so the final paths are
  `./cache/retrieval/...` and `./cache/reranker/...`.
- Keep `.venv/` and generated activation helpers out of git.

## Final Architecture Orientation

Read the friend-facing
[`reports/2026-competition-architecture.html`](reports/2026-competition-architecture.html)
for the image-led walkthrough. Agents should edit
[`reports/2026-competition-architecture.qmd`](reports/2026-competition-architecture.qmd)
and re-render it with Quarto; never patch the generated HTML directly. The guide
covers the final frozen source and its two sibling submission branches:
cache-first Retrieval TSVs and selected-evidence RAG JSONLs.

Keep these boundaries intact:

- The untouched narrative always remains a Retrieval lane.
- DeepSeek planning is bounded and one-shot, not an open-ended search loop.
- Each valid subnarrative contains 1–3 BM25 query lanes, for at most 25 searches per topic including the original lane.
- The 1,000-document and 100-passage values are per-query ceilings with a documents-to-passages unit change.
- Selected passages are factual authority; canonical claim hints are advisory.
- The authenticated `facet-deepseek-b40-v3` source is the common provenance
  point; the final Retrieval and RAG bundles are sibling derivations.
- Retrieval submission replay robustly admits a narrative-specific candidate
  core, completes a cache-first score matrix only over that core, and orders
  the identical per-topic set three ways. Submitted depth ranges from 1 to 121.
- RAG Generation consumes only the authenticated selected-evidence handoff.
  The organizer-facing Retrieval TSV is not Generation input.
- Generation never opens the Retrieval TSV, full-text ZIP, qrels, gold
  nuggets, or RAGDoll scores.
- Single-pass Sol and bounded multi-stage Luna + Sol Generation use the same
  sealed evidence but separate strategy-specific state and organizer files.
- Multi-stage Generation bounds Luna planning, per-group audits, and operation
  screening; Sol drafting and revision remain bounded, with the already
  validated draft as the deterministic fallback.
- Transport, semantic, and stage reservation limits remain separate.

## Historical Run and Reproduction Safety

- No further competition runs are planned. The commands and safety rules below
  are retained only for reproducibility if the user explicitly reopens the
  project; they are not a request or next-action list.

- Treat `trec-rag-skills/skills/trec-rag-2026-track-guidelines/SKILL.md`
  as the canonical task and submission contract. Read its retrieval or RAG
  reference before changing a runner, config, validator, or organizer-facing
  output. Use
  `trec-rag-skills/skills/pyserini-rest-api/SKILL.md` for API mechanics and
  token-safety rules.
- The original source workflow has two ordered commands. Retrieval must finish
  first and publish the sealed selected-evidence handoff before RAG. These
  commands describe the source runner and config shape; they do not regenerate
  the accepted cache-first Retrieval bundle or its multi-stage RAG sibling:

  ```bash
  .venv/bin/python-rocm -m trec_rag.competition_retrieval \
    configs/rag26_competition_retrieval_v2.yaml

  .venv/bin/python -m trec_rag.competition_rag \
    --config configs/rag26_competition_rag_gpt_sol_v2.yaml
  ```

- The checked-in configs select all 119 test narratives by default. Do not use
  them for an exploratory one-, two-, or three-topic run. Follow the two-topic smoke instructions in
  `code/trec_rag/README.md`: copy both configs into ignored `configs/local/`,
  assign unique experiment IDs/output directories, point the RAG input at the
  smoke retrieval handoff, and set RAG `experiment.topic_ids` to the same IDs.
  Bound retrieval with one repeated selector per narrative:

  ```bash
  .venv/bin/python-rocm -m trec_rag.competition_retrieval \
    configs/local/<retrieval-smoke-config>.yaml \
    --topic rag2026-0 --topic rag2026-1

  .venv/bin/python -m trec_rag.competition_rag \
    --config configs/local/<rag-smoke-config>.yaml
  ```

- Before a live run, confirm submodules are at the commits recorded by the
  superproject, the tracked worktree is clean, the selected narrative IDs and
  output namespace are correct, and the required secrets are present without
  printing them. Retrieval needs `INDEX_URL`, `PYSERINI_API_TOKEN`, and
  `OPENROUTER_API_KEY`; fixed-retrieval generation needs
  `OPENROUTER_API_KEY`. Load them from ignored `.env`/`.env.local` files.
- Report the selected topic count, expected cache reuse or misses, hosted/model
  calls, and output directories before starting a cache-miss or full live run.
  Full retrieval performs remote planning, Pyserini retrieval, local GPU
  reranking, and hosted canonicalization; full RAG makes hosted generation
  calls. Do not start either full run without explicit user authorization.
- Retrieval resumes only checkpoints whose schemas, source bytes, identities,
  and hash chain validate. RAG `experiment.mode: create` refuses existing
  generation state; use `resume` after interruption. Use `overwrite` only with
  explicit user authorization because it deletes that RAG config's generation
  JSONL and dedicated `work/` directory.
- Verify retrieval by reading its manifest-last receipt and confirming
  `generation_handoff_manifest.json` plus the organizer-facing TREC run and
  full-text ZIP. Verify RAG by validating `rag_output_trec_rag_2026.jsonl`
  against the handoff's exact topic IDs, narratives, citation domains, word
  limit, and organizer citation rules. Generation must never open the TREC run,
  full-text ZIP, qrels, gold nuggets, or RAGDoll scores.
- Keep `outputs/`, caches, raw corpus text, generated claims, provider responses,
  and debug reports private. Do not commit, publish, serve, or copy them into a
  rendered-artifact directory without explicit authorization and a privacy
  review.

## Verification

- Run targeted tests after changes.
- For rendered UI changes, use Playwright on desktop and mobile viewports when
  available.
- Report any verification that could not be run and why.

## Git Hygiene

- Stage only files that belong to the requested change.
- Do not commit local scratch files, secrets, server logs, or unrelated user
  changes.
- Use concise commits that describe the user-facing change.
- When creating, switching to, or working inside a linked worktree, make sure
  git submodules are initialized and updated with
  `git submodule update --init --recursive`. If `.githooks/` is configured via
  `git config core.hooksPath .githooks`, the tracked hooks handle this after
  checkout and merge.

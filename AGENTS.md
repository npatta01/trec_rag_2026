# Shared Agent Instructions

These instructions are the shared contract for Codex, Claude, and any future
coding agent working in this repository.

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
  ROCm and syncs the `rocm` dependency group; otherwise it syncs the standard
  project environment.
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

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

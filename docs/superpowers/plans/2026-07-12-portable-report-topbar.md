# Portable Report Top-Bar Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress shared interactive chrome only for portable reports and restore official overflow verification.

**Architecture:** Add one derived `showTopBar` condition in the shared React reader and use it at both shell render sites. Rebuild the existing packaged portable reader; do not introduce report-specific HTML or a new runtime.

**Tech Stack:** React/TypeScript, Node test runner, Playwright Core smoke tests, Vite single-file build, packaged Data Analytics HTML verifier.

## Global Constraints

- Modify only the installed Data Analytics plugin source/tests/generated assets needed by its existing build.
- Portable dashboards and every non-portable mode retain their top bar.
- Portable reports retain exactly one visible title through the required first Markdown heading.
- Do not weaken browser verification or change Chromium flags.
- Do not publish, share, or modify marketplace configuration.

---

### Task 1: Hide portable-report chrome and rebuild the reader

**Files:**
- Modify: `/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/src/analytics-app/App.tsx`
- Modify: `/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/tests/native-style-contract.test.mjs`
- Modify: `/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/tests/portable-browser.smoke.mjs`
- Generate: the plugin's existing portable-reader assets through `npm run build:portable-reader` and `npm run normalize:assets`

**Interfaces:**
- Consumes: `environment.mode`, `manifest.surface`, and the existing `AnalyticsTopBar` component.
- Produces: unchanged reader layout except no top bar for portable reports.

- [x] **Step 1: Write failing tests**

Require the source to derive a portable-report-only `showTopBar` boolean and
require browser report checks to assert zero top bars, one visible `h1`, a
vertical scrollbar, and `scrollWidth <= clientWidth`.

- [x] **Step 2: Verify RED**

Run:

```bash
npm test -- --test-name-pattern="portable report chrome"
```

Expected: fail because portable reports still render `AnalyticsTopBar`.

- [x] **Step 3: Implement the minimal condition**

Derive:

```ts
const showTopBar = !(environment.mode === "portable" && manifest?.surface === "report");
```

Wrap both shared `AnalyticsTopBar` render sites with `showTopBar` without
changing the component or dashboard behavior.

- [x] **Step 4: Verify source tests and rebuild**

Run the focused Node tests, then:

```bash
npm run build:portable-reader
npm run normalize:assets
```

Expected: focused tests pass and packaged reader asset parts are regenerated.

- [x] **Step 5: Verify browser behavior and real delivery**

Run `npm run test:portable-browser`, then the official repository report
delivery command. Expected: zero top bars for portable reports, one visible
title, no horizontal overflow at all tested widths, and a successful delivery
receipt with `stages.verification = "passed"`.

## Execution record

The installed remote-cache package did not include `node_modules`, so the
documented Vite build and standalone Playwright smoke could not run without an
unapproved dependency download. A deterministic local patcher instead audited
and regenerated the existing packaged reader asset, removing exactly the
portable-report top-bar call while preserving the dashboard call. The focused
source-contract test passed. The repository's official delivery command then
provided the real browser coverage: validation, packaging, source interaction,
and overflow checks passed at 1440 px and 390 px. This local plugin-cache repair
may be replaced by a future plugin refresh.

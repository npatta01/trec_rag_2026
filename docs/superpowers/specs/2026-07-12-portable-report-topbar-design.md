# Portable Report Top-Bar Repair Design

## Objective

Repair the installed Data Analytics portable reader so a long portable report
renders without global horizontal overflow and follows the existing
content-only export contract.

## Verified defect

Portable reports unconditionally render `.analytics-top-bar`. Its shared CSS
uses `width: 100vw` and full-bleed margins. When a vertical scrollbar is
present, the reader iframe has `100vw = 800px` but
`document.documentElement.clientWidth = 785px`. The top bar therefore extends
7.5px beyond both sides and the official verifier rejects the report. Direct
DOM measurement confirmed that every report content block fits; only the top
bar and its title overflow.

## Chosen behavior

- Suppress `AnalyticsTopBar` only when `environment.mode == "portable"` and
  `manifest.surface == "report"`.
- Keep the top bar unchanged for portable dashboards, inline surfaces, hosted
  surfaces, and all non-portable report modes.
- Keep the report's required first Markdown `#` heading as the single visible
  title.
- Do not patch generated report HTML, weaken the verifier, alter browser flags,
  or add a second renderer.

## Verification

- Add a source-contract test for the portable-report-only condition.
- Strengthen the portable report browser smoke to require:
  - zero `.analytics-top-bar` elements;
  - exactly one visible `h1` in the report reader;
  - `scrollWidth <= clientWidth` at desktop, tablet, and mobile sizes;
  - a document taller than the viewport so the scrollbar case is exercised.
- Rebuild the packaged portable reader assets.
- Run focused plugin tests, the portable browser smoke, and the repository's
  official `deliver_portable_artifact.mjs` command on the real report.

## Scope and persistence

The installed `data-analytics` package is a remote cache, not a Git checkout or
local marketplace plugin. The repair is local and private, changes no
marketplace configuration, and may be replaced by a future plugin refresh.

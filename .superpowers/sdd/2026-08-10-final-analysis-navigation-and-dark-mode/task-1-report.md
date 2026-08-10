# Task 1 report — Repair canonical Quarto inline-code theming

Recorded 2026-08-10. This task changed only the architecture report's
canonical stylesheet, its source/render smoke test, and the Quarto-generated
HTML. No QMD, navigation, accepted artifact, portal, or hosted/provider state
was changed.

## Implementation

- Added the light inline-code tokens `--guide-code-ink: #6f3488`,
  `--guide-code-bg: #f3e8f7`, and `--guide-code-border: #d8b9e3` to `:root`.
- Added the dark inline-code tokens `--guide-code-ink: #f0cfff`,
  `--guide-code-bg: #2c2332`, and `--guide-code-border: #765484` to
  `body.quarto-dark`.
- Added `:not(pre) > code:not(.sourceCode)` styling for the border, background,
  and foreground, with an explicit `body.quarto-dark` rule so Quarto/Bootstrap
  dark-mode styles cannot fall back to Bootstrap's light gray code background.
- Kept the existing `code { overflow-wrap: anywhere; }` rule and did not add
  styling to `pre`, `.sourceCode` blocks, or syntax-token selectors.
- Added source/render contract assertions for all three tokens, the explicit
  dark inline-code selector, and the rendered `--guide-code-bg` token.
- Regenerated `reports/2026-competition-architecture.html` from the canonical
  QMD with Quarto 1.9.38.

## TDD evidence

The new contract test was run before the CSS implementation:

```text
node reports/2026-competition-architecture.test.js
```

It failed as intended with:

```text
Error: report CSS should define --guide-code-ink
```

After the CSS change and canonical rerender:

```text
quarto render reports/2026-competition-architecture.qmd
node reports/2026-competition-architecture.test.js
```

The render completed with `Output created: 2026-competition-architecture.html`
and the smoke test reported:

```text
2026 competition architecture smoke test passed
```

## Browser verification

Local headless Google Chrome was launched against the generated standalone
`file://` HTML. The test clicked the actual `.quarto-color-scheme-toggle`,
selected the inline `facet:<subnarrative>:text` code element, inspected a real
`pre code` block, and applied a 390px viewport through the Chrome DevTools
Protocol.

- Body class after the click: `quarto-dark`.
- Inline foreground: `rgb(240, 207, 255)`.
- Inline background: `rgb(44, 35, 50)`.
- Inline border: `rgb(118, 84, 132)`.
- Computed WCAG contrast: `10.817:1` (required minimum `4.5:1`).
- Source `<pre><code>` and its parent `<pre>` retained the same computed
  background and foreground before and after toggling (`rgba(0, 0, 0, 0)` and
  `rgb(0, 0, 0)` in this report, respectively).
- Desktop viewport: `1425px` client width and `1425px` scroll width; no
  horizontal overflow, and the selected inline code stayed within the viewport.
- Mobile viewport: `375px` client width and `375px` scroll width after Chrome's
  390px window emulation; no horizontal overflow, and the selected inline code
  stayed within the viewport.

## Final checks

```text
git diff --check
```

completed successfully. The owned files are:

- `reports/2026-competition-architecture/report.css`
- `reports/2026-competition-architecture.test.js`
- `reports/2026-competition-architecture.html`
- `.superpowers/sdd/2026-08-10-final-analysis-navigation-and-dark-mode/task-1-report.md`

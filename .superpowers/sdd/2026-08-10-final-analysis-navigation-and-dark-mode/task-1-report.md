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
  transparent backgrounds before and after toggling; the first-pass foreground
  remained black in both modes and was repaired in Fix Round 1 below.
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

## Fix Round 1

The first browser pass showed that the plain fenced formula (`pre > code`
without `.sourceCode`) inherited a black foreground in dark mode. Its
transparent background was correct, but black text was not readable on the
dark page surface. The repair adds only this narrowly scoped declaration:

```css
body.quarto-dark pre > code:not(.sourceCode) {
  color: var(--guide-ink);
}
```

It does not add or change a background on `pre`, `pre > code`, `.sourceCode`,
or syntax-token selectors. The existing inline-code background remains limited
to `:not(pre) > code:not(.sourceCode)` and its explicit dark-mode counterpart.

The smoke test now uses exact, scoped contracts rather than token substrings:

- It extracts `:root` and `body.quarto-dark` blocks and checks each code token
  against its required literal value.
- It checks exact normalized declaration bodies for the light inline selector,
  dark inline selector, and dark plain-fenced-code selector.
- It checks the same canonical values and declaration blocks in the rendered
  HTML, and rejects Bootstrap's `#f8f9fa` dark background regression.
- It scans report CSS rules to reject any background declaration attached to a
  real `pre`, `.sourceCode`, or `code span` selector, while correctly allowing
  the intentional `:not(pre)` and `:not(.sourceCode)` exclusions.
- The exact expected scopes/declarations act as mutation guards: moving a token
  between scopes, deleting a declaration, changing a value, or adding a
  background to the plain fenced-code rule fails the test.

TDD RED evidence for the new contract:

```text
node reports/2026-competition-architecture.test.js
```

failed because the required
`body.quarto-dark pre > code:not(.sourceCode)` rule was absent. After adding
the foreground-only rule, the test next failed because the generated HTML was
stale; rerendering from the QMD resolved that contract failure.

The canonical rerender and strict smoke test then passed:

```text
quarto render reports/2026-competition-architecture.qmd
node reports/2026-competition-architecture.test.js
2026 competition architecture smoke test passed
```

Fresh headless Chrome verification covered light mode, the actual Quarto dark
toggle, and a 390px emulated viewport:

- Light inline code computed as `rgb(111, 52, 136)` on
  `rgb(243, 232, 247)`, with `7.015:1` contrast.
- Dark inline code computed as `rgb(240, 207, 255)` on `rgb(44, 35, 50)`,
  with `10.817:1` contrast.
- Dark plain fenced code computed as `rgb(238, 245, 240)` over its inherited
  effective page background `rgb(15, 21, 18)`, with `16.681:1` contrast.
- Fenced-code and parent-`pre` backgrounds remained exactly
  `rgba(0, 0, 0, 0)` before and after the toggle; only the intended dark
  foreground changed.
- Desktop width was `1425px` client/scroll with no horizontal overflow. The
  390px emulation produced `375px` client/scroll with no overflow or clipping.

Final Fix Round 1 checks:

```text
node reports/2026-competition-architecture.test.js
git diff --check
```

Both completed successfully before the Fix Round 1 commit.

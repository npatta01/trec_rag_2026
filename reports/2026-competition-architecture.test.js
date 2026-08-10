const fs = require("node:fs");
const path = require("node:path");

const reportsRoot = __dirname;
const sourcePath = path.join(reportsRoot, "2026-competition-architecture.qmd");
const htmlPath = path.join(reportsRoot, "2026-competition-architecture.html");
const assetRoot = path.join(reportsRoot, "2026-competition-architecture");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(sourcePath), "canonical Quarto source should exist");
assert(fs.existsSync(htmlPath), "rendered architecture HTML should exist");
assert(
  fs.existsSync(path.join(assetRoot, "report.css")),
  "local report CSS should exist",
);
assert(
  !fs.existsSync(path.join(reportsRoot, "..", "_quarto.yml")),
  "repo-wide Quarto config must not exist",
);

const qmd = fs.readFileSync(sourcePath, "utf8");
const normalizedQmd = qmd.replace(/\s+/g, " ");
const html = fs.readFileSync(htmlPath, "utf8");
const reportCss = fs.readFileSync(path.join(assetRoot, "report.css"), "utf8");

const analysisLinks = [
  ["Retrieval Quality Analysis", "2026-retrieval-nugget-coverage.html"],
  [
    "RAG Analysis: rag26-ss1",
    "https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html",
  ],
  [
    "RAG Analysis: rag26-ms1-final",
    "https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html",
  ],
];

for (const [label, href] of analysisLinks) {
  assert(normalizedQmd.includes(label), `QMD should label ${label}`);
  const escapedHref = href.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const qmdOccurrences = (qmd.match(new RegExp(`\\]\\(${escapedHref}\\)`, "g")) || []).length;
  const htmlOccurrences = (html.match(new RegExp(`href="${escapedHref}"`, "g")) || []).length;
  assert(qmdOccurrences === 1, `QMD should link ${href} exactly once`);
  assert(htmlOccurrences === 1, `rendered report should link ${href} exactly once`);
}
for (const href of analysisLinks.slice(1).map(([, target]) => target)) {
  const hrefIndex = normalizedQmd.indexOf(href);
  const nearby = normalizedQmd.slice(Math.max(0, hrefIndex - 180), hrefIndex + href.length + 240);
  assert(
    nearby.includes("Private / tailnet"),
    `QMD should mark ${href} as Private / tailnet near the link`,
  );
}
assert(
  ![...normalizedQmd.matchAll(/href="([^"]+)"/g)].some(([, href]) => /comparison|side[- ]by[- ]side/i.test(href)),
  "architecture report should not add a comparison or side-by-side report target",
);
assert(
  !/\]\([^)]*(?:comparison|side[- ]by[- ]side)[^)]*\)/i.test(qmd),
  "architecture source should not add a comparison or side-by-side report target",
);
for (const signal of [
  "separate 119-topic RAGDoll citation-support reports",
  "RAGDoll measures citation support, not official TREC correctness",
  "qrel/gold metrics are unavailable",
  "evaluation did not influence accepted priority",
]) {
  assert(normalizedQmd.includes(signal), `QMD should state final-analysis scope: ${signal}`);
}

function normalizeCss(value) {
  return value.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\s+/g, " ").trim();
}

function extractRule(source, selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(`${escaped}\\s*\\{([^{}]*)\\}`, "m").exec(source);
  assert(match, `CSS should define the ${selector} rule`);
  return normalizeCss(match[1]);
}

function assertRule(source, selector, declarations, label) {
  const actual = extractRule(source, selector);
  const expected = normalizeCss(declarations);
  assert(actual === expected, `${label || selector} declarations changed: ${actual}`);
}

function assertScopedToken(source, scope, token, value) {
  const declarations = extractRule(source, scope);
  assert(
    new RegExp(`(?:^|; )${token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}: ${value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:;|$)`).test(declarations),
    `${scope} should set ${token} to ${value}`,
  );
}

assertScopedToken(reportCss, ":root", "--guide-code-ink", "#6f3488");
assertScopedToken(reportCss, ":root", "--guide-code-bg", "#f3e8f7");
assertScopedToken(reportCss, ":root", "--guide-code-border", "#d8b9e3");
assertScopedToken(reportCss, "body.quarto-dark", "--guide-code-ink", "#f0cfff");
assertScopedToken(reportCss, "body.quarto-dark", "--guide-code-bg", "#2c2332");
assertScopedToken(reportCss, "body.quarto-dark", "--guide-code-border", "#765484");

const inlineCodeSelector = ":not(pre) > code:not(.sourceCode)";
const darkInlineCodeSelector = "body.quarto-dark :not(pre) > code:not(.sourceCode)";
const plainDarkCodeSelector = "body.quarto-dark pre > code:not(.sourceCode)";
const inlineCodeDeclarations = `
  border: 1px solid var(--guide-code-border);
  background: var(--guide-code-bg);
  color: var(--guide-code-ink);
`;
const darkInlineCodeDeclarations = `
  border-color: var(--guide-code-border);
  background: var(--guide-code-bg);
  color: var(--guide-code-ink);
`;
const plainDarkCodeDeclarations = "color: var(--guide-ink);";

assertRule(reportCss, inlineCodeSelector, inlineCodeDeclarations, "light inline-code");
assertRule(reportCss, darkInlineCodeSelector, darkInlineCodeDeclarations, "dark inline-code");
assertRule(reportCss, plainDarkCodeSelector, plainDarkCodeDeclarations, "dark plain fenced code");
assert(
  !extractRule(reportCss, plainDarkCodeSelector).includes("background"),
  "dark plain fenced code must not add a background declaration",
);

for (const selector of [inlineCodeSelector, darkInlineCodeSelector]) {
  assert(
    !/(?:^|[\s>])pre(?:[\s>]|$)/.test(selector) &&
      !/(?:^|[\s>])\.sourceCode(?:[\s>]|$)/.test(selector),
    `${selector} must exclude code blocks`,
  );
}

for (const match of reportCss.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
  const selector = normalizeCss(match[1]);
  const declarations = normalizeCss(match[2]);
  if (/\bbackground(?:-color)?\s*:/.test(declarations)) {
    assert(
      !/(?:^|[\s>])pre(?:[\s>]|$)|(?:^|[\s>])\.sourceCode(?:[\s>]|$)|(?:^|[\s>])code\s+span/.test(selector),
      `report CSS must not add code-block or syntax-token backgrounds: ${selector}`,
    );
  }
}

for (const [scope, token, value] of [
  [":root", "--guide-code-ink", "#6f3488"],
  [":root", "--guide-code-bg", "#f3e8f7"],
  [":root", "--guide-code-border", "#d8b9e3"],
  ["body.quarto-dark", "--guide-code-ink", "#f0cfff"],
  ["body.quarto-dark", "--guide-code-bg", "#2c2332"],
  ["body.quarto-dark", "--guide-code-border", "#765484"],
]) {
  assertScopedToken(html, scope, token, value);
}
for (const [selector, declarations] of [
  [inlineCodeSelector, inlineCodeDeclarations],
  [darkInlineCodeSelector, darkInlineCodeDeclarations],
  [plainDarkCodeSelector, plainDarkCodeDeclarations],
]) {
  const canonical = normalizeCss(`${selector} { ${declarations} }`);
  assert(normalizeCss(html).includes(canonical), `rendered HTML should contain ${selector} declarations`);
}
assert(
  !/body\.quarto-dark[^{}]*\{[^{}]*--guide-code-bg:\s*#f8f9fa/.test(reportCss),
  "dark inline-code background must not regress to Bootstrap light gray",
);

assert(
  /body\.quarto-dark\s*\{/.test(reportCss),
  "report CSS should switch guide variables on Quarto's dark-mode body class",
);
assert(
  /@media print[\s\S]*figure\.quarto-float img\s*\{[\s\S]*max-height:/m.test(reportCss),
  "print CSS should constrain tall figures to a printable page",
);
assert(
  /@media print[\s\S]*details::details-content\s*\{[\s\S]*content-visibility:\s*visible/m.test(reportCss),
  "print CSS should expose closed disclosure content",
);

for (const signal of [
  "embed-resources: true",
  "light: 2026-competition-architecture/theme-light.scss",
  "dark: 2026-competition-architecture/theme-dark.scss",
  "toc: true",
  "lightbox: true",
  "2026-competition-architecture/report.css",
]) {
  assert(qmd.includes(signal), `Missing Quarto source contract: ${signal}`);
}

assert(
  /<meta[^>]+name="generator"[^>]+quarto/i.test(html),
  "HTML should identify Quarto as generator",
);
assert(
  !/<(?:script|link|img)[^>]+(?:src|href)="https?:/i.test(html),
  "HTML runtime assets must be local or embedded",
);
assert(
  !html.includes("fonts.googleapis.com") &&
    !html.includes("fonts%2Egoogleapis%2Ecom"),
  "HTML must not retain an embedded remote-font import",
);

const figures = [
  "01-whole-system.svg",
  "02-frozen-source-retrieval.svg",
  "03-bounded-deepseek-planning.svg",
  "04-variable-depth-candidate-core.svg",
  "05-targeted-scoring-and-runs.svg",
  "06-generation-handoff-contract.svg",
  "07-single-pass-rag.svg",
  "08-multistage-rag.svg",
  "09-accepted-submissions.svg",
];

for (const figure of figures) {
  const figurePath = path.join(assetRoot, figure);
  assert(fs.existsSync(figurePath), `Missing architecture figure: ${figure}`);
  const svg = fs.readFileSync(figurePath, "utf8");
  assert(svg.includes('role="img"'), `${figure} should expose image semantics`);
  assert(
    /aria-labelledby="[^"]+ [^"]+"/.test(svg),
    `${figure} should reference title and description`,
  );
  assert(
    /<title id="[^"]+">[^<]+<\/title>/.test(svg),
    `${figure} should have a titled accessible name`,
  );
  assert(
    /<desc id="[^"]+">[^<]+<\/desc>/.test(svg),
    `${figure} should have an accessible description`,
  );
  assert(
    svg.includes("prefers-color-scheme: dark"),
    `${figure} should support dark mode`,
  );
  assert(
    svg.includes('vector-effect="non-scaling-stroke"'),
    `${figure} should retain line weight when enlarged`,
  );
}

const assetReadmePath = path.join(assetRoot, "README.md");
assert(fs.existsSync(assetReadmePath), "architecture asset README should exist");
const assetReadme = fs.readFileSync(assetReadmePath, "utf8");
for (const figure of figures) {
  assert(assetReadme.includes(figure), `Asset README should document ${figure}`);
}
assert(
  assetReadme.includes("conceptual") && /no\s+test narrative/.test(assetReadme),
  "Asset README should explain conceptual geometry and the privacy boundary",
);

const requiredConcepts = [
  "Frozen source, two submission branches",
  "facet-deepseek-b40-v3",
  "DeepSeek V4 Flash",
  "deepseek/deepseek-v4-flash-20260423",
  "climbmix-400b",
  "3500 characters",
  "350-character overlap",
  "mixedbread-ai/mxbai-rerank-base-v2",
  "median absolute deviation",
  "2.5 × 1.4826 × MAD",
  "variable depth",
  "1–121 documents",
  "4,246 rows",
  "r26-narr-facet-v1",
  "r26-facet-breadth-v1",
  "r26-narrative-v1",
  "Selected passages are factual authority",
  "Canonical claim hints are advisory",
  "generation_handoff_manifest.json",
  "rag26-ms1-final",
  "rag26-ss1",
  "openai/gpt-5.6-luna",
  "openai/gpt-5.6-sol",
  "691 evidence groups",
  "12,000-token ceiling",
  "1,024-word ceiling",
  "Accepted by Evalbase",
  "The Retrieval TSV is not Generation input",
];

for (const concept of requiredConcepts) {
  assert(normalizedQmd.includes(concept), `Missing architecture concept in QMD: ${concept}`);
}

assert(
  fs.readFileSync(path.join(assetRoot, "04-variable-depth-candidate-core.svg"), "utf8").includes("1–121 documents"),
  "Candidate-core figure should show the observed variable-depth range",
);
assert(
  fs.readFileSync(path.join(assetRoot, "03-bounded-deepseek-planning.svg"), "utf8").includes("1–3 BM25 queries each"),
  "Planning figure should distinguish subnarratives from their BM25 queries",
);

let previousFigureIndex = -1;
for (const figure of figures) {
  const currentFigureIndex = qmd.indexOf(figure);
  assert(currentFigureIndex > previousFigureIndex, `Figure should appear in story order: ${figure}`);
  previousFigureIndex = currentFigureIndex;
  assert(html.includes(figure), `Rendered report should link directly to ${figure}`);
}

assert(
  (qmd.match(/\*\*Text equivalent:\*\*/g) || []).length === 9,
  "QMD should contain nine visible text equivalents",
);
assert(
  (html.match(/<strong>Text equivalent:<\/strong>/g) || []).length === 9,
  "Rendered report should contain nine visible text equivalents",
);
assert(
  (qmd.match(/<details>/g) || []).length >= 9 &&
    (qmd.match(/<summary>Implementation notes and sources<\/summary>/g) || []).length >= 9,
  "Each figure should have a native implementation disclosure",
);
for (const signal of [
  "Solid arrows",
  "Dashed arrows",
  "documents",
  "chunks",
  "passages",
  "selected evidence",
  "advisory hints",
  "Retrieval TSV",
  "full-text ZIP",
  "qrels",
  "gold nuggets",
  "RAGDoll scores",
  "sibling",
]) {
  assert(qmd.includes(signal), `Missing architecture distinction: ${signal}`);
}

for (const forbidden of [
  "Authorization:",
  "Bearer ",
  "rag2026-",
  "TODO",
  "placeholder",
]) {
  assert(!qmd.includes(forbidden), `QMD should not expose ${forbidden}`);
}
for (const forbidden of ["Authorization:", "Bearer ", "rag2026-"]) {
  assert(!html.includes(forbidden), `Rendered report should not expose ${forbidden}`);
}

const agents = fs.readFileSync(path.join(reportsRoot, "..", "AGENTS.md"), "utf8");
const normalizedAgents = agents.replace(/\s+/g, " ");
const rootReadme = fs.readFileSync(path.join(reportsRoot, "..", "README.md"), "utf8");

for (const target of [
  "reports/2026-competition-architecture.html",
  "reports/2026-competition-architecture.qmd",
]) {
  assert(agents.includes(target), `AGENTS.md should link ${target}`);
  assert(rootReadme.includes(target), `README.md should link ${target}`);
}

for (const signal of [
  "## Final Architecture Orientation",
  "two sibling submission branches",
  "per-query ceilings with a documents-to-passages unit change",
  "1–3 BM25 query lanes",
  "at most 25 searches per topic",
  "Selected passages are factual authority",
  "authenticated selected-evidence handoff",
  "organizer-facing Retrieval TSV is not Generation input",
  "Transport, semantic, and stage reservation limits remain separate",
]) {
  assert(normalizedAgents.includes(signal), `AGENTS.md should retain architecture signal: ${signal}`);
}

console.log("2026 competition architecture smoke test passed");

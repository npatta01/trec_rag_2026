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
const html = fs.readFileSync(htmlPath, "utf8");
const reportCss = fs.readFileSync(path.join(assetRoot, "report.css"), "utf8");

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
  "02-retrieval-system.svg",
  "03-bounded-deepseek-planning.svg",
  "04-per-query-candidate-accounting.svg",
  "05-evidence-and-nuggetizer.svg",
  "06-generation-handoff-contract.svg",
  "07-sol-generation.svg",
  "08-validation-and-retries.svg",
  "09-organizer-output-split.svg",
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
  assetReadme.includes("fictional") && assetReadme.includes("conceptual"),
  "Asset README should explain fictional content and conceptual geometry",
);

const requiredConcepts = [
  "Illustrative narrative — not a TREC test topic",
  "DeepSeek V4 Flash",
  "deepseek/deepseek-v4-flash-20260423",
  "0–8 focused subnarratives",
  "Pyserini",
  "ClimbMix",
  "≤1,000 documents / focused query",
  "3500 characters",
  "350-character overlap",
  "Mixedbread",
  "mixedbread-ai/mxbai-rerank-base-v2",
  "≤100 passages / focused query",
  "Nuggetizer",
  "Selected passages are factual authority",
  "Canonical claim hints are advisory",
  "generation_handoff_manifest.json",
  "openai/gpt-5.6-sol",
  "12,000-token ceiling",
  "≤3 transport attempts",
  "≤2 semantic attempts",
  "1,024-word ceiling",
  "organizer RAG JSONL",
];

for (const concept of requiredConcepts) {
  assert(qmd.includes(concept), `Missing architecture concept in QMD: ${concept}`);
}

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
  "TREC run",
  "full-text ZIP",
  "qrels",
  "gold nuggets",
  "RAGDoll scores",
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
const rootReadme = fs.readFileSync(path.join(reportsRoot, "..", "README.md"), "utf8");

for (const target of [
  "reports/2026-competition-architecture.html",
  "reports/2026-competition-architecture.qmd",
]) {
  assert(agents.includes(target), `AGENTS.md should link ${target}`);
  assert(rootReadme.includes(target), `README.md should link ${target}`);
}

for (const signal of [
  "## Architecture Orientation",
  "Retrieval → handoff → Generation",
  "per-query ceilings with a documents-to-passages unit change",
  "Selected passages are factual authority",
  "authenticated handoff",
  "TREC run, full-text ZIP, qrels, gold nuggets, or RAGDoll scores",
  "Transport and semantic retry limits are separate",
]) {
  assert(agents.includes(signal), `AGENTS.md should retain architecture signal: ${signal}`);
}

console.log("2026 competition architecture smoke test passed");

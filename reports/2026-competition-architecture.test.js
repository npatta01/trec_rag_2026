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

console.log("2026 competition architecture smoke test passed");

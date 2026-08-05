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

console.log("2026 competition architecture smoke test passed");

const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "2025-promising-rag-architecture.html");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(htmlPath), "HTML report should exist");
const html = fs.readFileSync(htmlPath, "utf8");

for (const signal of [
  '<a class="skip-link" href="#main-content">',
  '<main id="main-content">',
  'id="start"',
  'id="pipeline"',
  'id="definitions"',
  'id="team-evidence"',
  'id="recommended-build"',
  'id="confidence"',
  'id="sources"',
  "UTokyo-HitU",
  "NC State LAS",
  "MITLL",
  "CFDA",
  "Tokyo University of Science",
  "WaterlooClarke",
  "GenAIus",
  "0.6934",
  "0.37",
  "0.65",
  "claims / nuggets + source IDs",
  'alt="Seven-stage recommended RAG architecture',
  'scope="col"',
  "@media print",
  "prefers-reduced-motion",
]) {
  assert(html.includes(signal), `Missing report signal: ${signal}`);
}

for (const forbidden of [
  "PYSERINI_API_TOKEN",
  "Authorization",
  "Bearer ",
  "https://cdn.",
  "fonts.googleapis.com",
  "Tailscale Funnel",
  "TODO",
  "placeholder",
]) {
  assert(!html.includes(forbidden), `Report should not expose ${forbidden}`);
}

const localTargets = [...html.matchAll(/(?:href|src)="([^"]+)"/g)]
  .map((match) => match[1])
  .filter((target) => !target.startsWith("#") && !target.startsWith("data:"));

for (const target of localTargets) {
  assert(
    !/^https?:/.test(target) || target.startsWith("https://trec.nist.gov/"),
    `Unexpected external runtime/source target: ${target}`,
  );
  if (!/^https?:/.test(target)) {
    assert(fs.existsSync(path.join(root, target)), `Missing local target: ${target}`);
  }
}

console.log("2025 architecture HTML smoke test passed");

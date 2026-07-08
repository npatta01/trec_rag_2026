const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "ragtime-2025-reproduction-playbook.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "ragtime-2025-reproduction-playbook.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");

const requiredSignals = [
  "TREC RAGTIME 2025 Reproduction Playbook",
  "Executive Summary",
  "Plain-Language Glossary",
  "In Plain English",
  "How To Start",
  "Show implementation details",
  "Show official scores, run ids, and source evidence",
  "Skip to report content",
  'id="main-content"',
  "BAAI/bge-m3",
  "mxbai-rerank-large-v1",
  "check_sentence_attested",
  "Microsoft Autogen-style team structures",
];

for (const signal of requiredSignals) {
  assert(html.includes(signal), `Missing RAGTIME playbook signal: ${signal}`);
}

const candidateCards = (html.match(/class="candidate-card"/g) || []).length;
assert(candidateCards === 8, `Expected 8 candidate cards, found ${candidateCards}`);

const detailDrawers = (html.match(/<details/g) || []).length;
assert(detailDrawers === 16, `Expected 16 collapsed detail drawers, found ${detailDrawers}`);

assert(!html.includes("<details open"), "Detail drawers should be collapsed by default");

const forbiddenSignals = [
  "PYSERINI_API_TOKEN",
  "Authorization",
  "Bearer ",
  "TODO",
  "placeholder",
];

for (const signal of forbiddenSignals) {
  assert(!html.includes(signal), `RAGTIME playbook should not expose ${signal}`);
}

console.log("ragtime reproduction playbook smoke test passed");

const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "index.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "reports/index.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");

const requiredSignals = [
  "TREC RAG Reports",
  "Reports Home",
  "TREC RAG 2026 Briefing",
  "TREC RAG 2025 Writeups",
  "TREC RAGTIME 2025 Reproduction Playbook",
  "reports-index",
  "Start with the briefing",
  "Use the 2025 writeups as the technique library",
  "reports/trec-rag-briefing-report.html",
  "trec-rag-2025-writeups/interactive-writeup.html",
  "reports/ragtime-2025-reproduction-playbook.html",
  "class=\"topbar\"",
  "class=\"hero\"",
  "class=\"shell hero-grid\"",
  "class=\"route-grid\"",
  "class=\"report-card",
  "class=\"label",
  "class=\"button primary\"",
];

for (const signal of requiredSignals) {
  assert(html.includes(signal), `Missing index signal: ${signal}`);
}

const forbiddenSignals = [
  "PYSERINI_API_TOKEN",
  "Authorization",
  "Bearer ",
  "TODO",
  "placeholder",
];

for (const signal of forbiddenSignals) {
  assert(!html.includes(signal), `Index should not expose ${signal}`);
}

const linkedFiles = [
  "trec-rag-briefing-report.html",
  "trec-rag-2025-writeups/interactive-writeup.html",
  "ragtime-2025-reproduction-playbook.html",
];

for (const linkedFile of linkedFiles) {
  assert(fs.existsSync(path.join(root, linkedFile)), `Linked report should exist: ${linkedFile}`);
}

console.log("reports index smoke test passed");

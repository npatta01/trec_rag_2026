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
  "Final 2026 submission architecture",
  "Frozen source → Retrieval submissions + RAG submissions",
  "Accepted submissions",
  "../index.html",
  "../submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "TREC RAG 2026 Briefing",
  "TREC RAG 2026 Competition Architecture",
  "one frozen authenticated source",
  "2026-competition-architecture.html",
  "TREC RAG 2025 Writeups",
  "Promising 2025 RAG Architecture",
  "Narrative to verified answer",
  "2025-promising-rag-architecture.html",
  "All-topic tethered-facet validation",
  "Experiment history",
  "reports-index",
  "Start with the final architecture",
  "Audit the accepted submissions",
  "trec-rag-briefing-report.html",
  "trec-rag-2025-writeups/interactive-writeup.html",
  "experiments/all_topic_tethered_facet_validation_v1/report.html",
  "../experiment.md",
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
  "../index.html",
  "../submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "2026-competition-architecture.html",
  "trec-rag-briefing-report.html",
  "trec-rag-2025-writeups/interactive-writeup.html",
  "2025-promising-rag-architecture.html",
  "experiments/all_topic_tethered_facet_validation_v1/report.html",
  "../experiment.md",
];

for (const linkedFile of linkedFiles) {
  assert(fs.existsSync(path.join(root, linkedFile)), `Linked report should exist: ${linkedFile}`);
}

console.log("reports index smoke test passed");

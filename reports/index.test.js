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
  "2026-ragdoll-rag26-ss1.html",
  "2026-ragdoll-rag26-ms1-final.html",
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

const publicRagReports = [
  ["2026-ragdoll-rag26-ss1.html", "3155 judge tasks", "1335 full support", "1809 partial support", "11 no support"],
  ["2026-ragdoll-rag26-ms1-final.html", "7008 judge tasks", "1651 full support", "5136 partial support", "221 no support"],
];
for (const [filename, ...signals] of publicRagReports) {
  const reportPath = path.join(root, filename);
  const source = fs.readFileSync(reportPath, "utf8");
  assert(fs.statSync(reportPath).size < 10_000_000, `${filename} should stay below 10 MB`);
  for (const signal of signals) {
    assert(source.includes(signal), `${filename} should include ${signal}`);
  }
  for (const forbidden of ["/home/", "tail481212", "Private post-run evaluation", "Bearer "]) {
    assert(!source.includes(forbidden), `${filename} should not expose ${forbidden}`);
  }
}

const analysisLinks = [
  ["Retrieval Quality Analysis", "2026-retrieval-nugget-coverage.html"],
  [
    "RAG Analysis: rag26-ss1",
    "2026-ragdoll-rag26-ss1.html",
  ],
  [
    "RAG Analysis: rag26-ms1-final",
    "2026-ragdoll-rag26-ms1-final.html",
  ],
];

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function assertAnalysisBindings(source, label) {
  for (const [visibleLabel, href] of analysisLinks) {
    const match = new RegExp(
      `<a\\b[^>]*href="${escapeRegExp(href)}"[^>]*>([\\s\\S]*?)</a>`,
    ).exec(source);
    assert(match, `${label} should contain an anchor for ${href}`);
    const anchorText = match[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    assert(
      anchorText === visibleLabel,
      `${label} should bind ${visibleLabel} to ${href}`,
    );
  }
}

function replaceAnchorLabel(source, href, nextLabel) {
  return source.replace(
    new RegExp(`(<a\\b[^>]*href="${escapeRegExp(href)}"[^>]*>)[\\s\\S]*?(</a>)`),
    `$1${nextLabel}$2`,
  );
}

function assertRejectsMutation(mutatedSource, label) {
  let rejected = false;
  try {
    assertAnalysisBindings(mutatedSource, label);
  } catch {
    rejected = true;
  }
  assert(rejected, `${label} should reject swapped analysis labels`);
}

assertAnalysisBindings(html, "reports index");
const swappedAnalysisLabels = replaceAnchorLabel(
  replaceAnchorLabel(html, analysisLinks[1][1], analysisLinks[2][0]),
  analysisLinks[2][1],
  analysisLinks[1][0],
);
assert(swappedAnalysisLabels !== html, "reports index mutation should swap visible labels");
assertRejectsMutation(swappedAnalysisLabels, "reports index");

const reportHrefs = [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
for (const [label, href] of analysisLinks) {
  const occurrences = (html.match(new RegExp(`href="${href.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`, "g")) || []).length;
  assert(occurrences === 1, `reports index should link ${href} exactly once`);
  assert(html.includes(label), `reports index should label ${label}`);
}
for (const href of analysisLinks.map(([, target]) => target)) {
  assert(fs.existsSync(path.join(root, href)), `analysis report should exist: ${href}`);
}
assert(
  !reportHrefs.some((href) => /comparison|side[- ]by[- ]side/i.test(href)),
  "reports index should not add a comparison or side-by-side report target",
);

console.log("reports index smoke test passed");

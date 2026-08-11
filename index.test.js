const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "index.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "root index.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");
const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
const internalHrefs = hrefs.filter(
  (href) => !href.startsWith("#") && !/^[a-z][a-z0-9+.-]*:/i.test(href),
);

assert(internalHrefs.length >= 18, "artifact hub should expose the full project link map");
for (const href of internalHrefs) {
  const target = href.split("#", 1)[0].split("?", 1)[0];
  assert(target && !target.startsWith("/"), `internal link must be repository-relative: ${href}`);
  assert(fs.existsSync(path.resolve(root, target)), `internal link should resolve: ${href}`);
}

assert((html.match(/<article class="link-card /g) || []).length === 7, "artifact hub should render seven cards");
for (const landmark of ["<main", "<header", "<section", "<footer", "aria-label="]) {
  assert(html.includes(landmark), `artifact hub should expose ${landmark}`);
}
for (const behavior of ["focus-visible", "prefers-reduced-motion", "@media (max-width: 620px)", "@media print"]) {
  assert(html.includes(behavior), `artifact hub should preserve ${behavior}`);
}
for (const forbidden of [
  "outputs/",
  "cache/",
  "PYSERINI_API_TOKEN",
  "OPENROUTER_API_KEY",
  "Authorization",
  "Bearer ",
]) {
  assert(!html.includes(forbidden), `artifact hub should not expose ${forbidden}`);
}

const ledgerPath = path.join(root, "submissions/trec-rag-2026/SUBMISSION_LEDGER.md");
const ledger = fs.readFileSync(ledgerPath, "utf8");
for (const runId of [
  "r26-narr-facet-v1",
  "r26-facet-breadth-v1",
  "r26-narrative-v1",
  "rag26-ms1-final",
  "rag26-ss1",
]) {
  const rows = ledger
    .split("\n")
    .filter((line) => line.startsWith("|") && line.includes(`\`${runId}\``));
  assert(rows.length === 2, `ledger should contain artifact and confirmation rows for ${runId}`);
  assert(rows.every((line) => line.includes("Accepted by Evalbase")), `ledger should mark ${runId} accepted`);
  assert(
    rows[1].includes("| Not recorded | Not recorded | Accepted by Evalbase |"),
    `ledger should preserve unknown portal metadata for ${runId}`,
  );
}

const analysisLinks = [
  ["Retrieval Quality Analysis", "reports/2026-retrieval-nugget-coverage.html"],
  [
    "RAG Analysis: rag26-ss1",
    "reports/2026-ragdoll-rag26-ss1.html",
  ],
  [
    "RAG Analysis: rag26-ms1-final",
    "reports/2026-ragdoll-rag26-ms1-final.html",
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

assertAnalysisBindings(html, "artifact hub");
const swappedAnalysisLabels = replaceAnchorLabel(
  replaceAnchorLabel(html, analysisLinks[1][1], analysisLinks[2][0]),
  analysisLinks[2][1],
  analysisLinks[1][0],
);
assert(swappedAnalysisLabels !== html, "artifact hub mutation should swap visible labels");
assertRejectsMutation(swappedAnalysisLabels, "artifact hub");

for (const [label, href] of analysisLinks) {
  const occurrences = (html.match(new RegExp(`href="${href.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`, "g")) || []).length;
  assert(occurrences === 1, `artifact hub should link ${href} exactly once`);
  assert(html.includes(label), `artifact hub should label ${label}`);
}
for (const href of analysisLinks.map(([, target]) => target)) {
  assert(fs.existsSync(path.join(root, href)), `artifact hub target should exist: ${href}`);
}
assert(
  !hrefs.some((href) => /comparison|side[- ]by[- ]side/i.test(href)),
  "artifact hub should not add a comparison or side-by-side report target",
);

console.log("root artifact hub smoke test passed");

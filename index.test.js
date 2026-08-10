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

assert((html.match(/<article class="link-card /g) || []).length === 6, "artifact hub should render six cards");
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

console.log("root artifact hub smoke test passed");

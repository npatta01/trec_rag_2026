const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "2025-promising-rag-architecture.html");
const figureTarget =
  "trec-rag-2025-writeups/figures/2025-promising-composite-architecture.png";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(htmlPath), "HTML report should exist");
const html = fs.readFileSync(htmlPath, "utf8");

for (const signal of [
  '<a class="skip-link" href="#main-content">',
  '<header class="site-header topbar">',
  '<a class="reports-home" href="index.html">Reports Home</a>',
  '<main id="main-content" tabindex="-1">',
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
  `<a class="button full-size-link" href="${figureTarget}">Open full-size architecture figure</a>`,
  "<caption>Cross-team roles in the recommended composite architecture</caption>",
  'scope="col"',
  "Nugget-to-claim interface",
  "<td>GenAIus</td>",
  "query-conditioned atomic nuggets",
  "@media print",
  "details > * { display: block !important; }",
  'a[href]::after { content: " (" attr(href) ")"',
  ".js-only { display: none; }",
  ".js .js-only { display: inline-block; }",
  ".button:hover { background: var(--focus); color: #fff; }",
  'window.addEventListener("beforeprint", expandDisclosuresForPrint);',
  'window.addEventListener("afterprint", restoreDisclosuresAfterPrint);',
  "prefers-reduced-motion",
  "function trapFocus(event)",
  'if (event.key !== "Tab" || lightbox.hidden) return;',
  "if (event.shiftKey && document.activeElement === first)",
  "else if (!event.shiftKey && document.activeElement === last)",
  "setBackgroundInert(open);",
]) {
  assert(html.includes(signal), `Missing report signal: ${signal}`);
}

assert(
  /\.topbar\s*\{[^}]*position:\s*sticky[^}]*top:\s*0/s.test(html),
  "Report top bar should remain sticky at the top of the viewport",
);

const figureLinkPattern = new RegExp(
  `<a\\s+class="button full-size-link"\\s+href="${figureTarget.replaceAll(".", "\\.")}"`,
);
assert(
  figureLinkPattern.test(html),
  "Figure should have an ordinary full-size image link that works without JavaScript",
);

const disclosureTags = [...html.matchAll(/<details(?:\s[^>]*)?>/g)].map(
  (match) => match[0],
);
assert(disclosureTags.length === 6, "Report should contain all six team disclosures");
assert(
  disclosureTags.every((tag) => /\sopen(?:\s|>)/.test(tag)),
  "Team disclosures should default open so content remains exposed without JavaScript",
);

for (const target of [
  "trec-rag-2025-writeups/pdfs/00-overview-rag.pdf",
  "trec-rag-2025-writeups/pdfs/01-cfdalab-rag.pdf",
  "trec-rag-2025-writeups/pdfs/07-genaius-rag.pdf",
  "trec-rag-2025-writeups/pdfs/08-tus-rag.pdf",
  "trec-rag-2025-writeups/pdfs/10-mitll-rag.pdf",
  "trec-rag-2025-writeups/pdfs/12-ncsu-las-rag-ragtime.pdf",
  "trec-rag-2025-writeups/pdfs/13-utokyo-rag.pdf",
  "trec-rag-2025-writeups/pdfs/15-waterlooclarke-dragun-rag.pdf",
]) {
  assert(
    html.includes(`href="${target}"`),
    `Missing expected primary PDF target: ${target}`,
  );
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

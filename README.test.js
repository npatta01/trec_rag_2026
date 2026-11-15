// The README is the front door, not the manual: it has to say what this is,
// show the film, explain the system once, and hand the reader to the project
// site. Everything else lives on the site, so this test caps both the length
// and the number of ways out.
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const source = fs.readFileSync(path.join(root, "README.md"), "utf8");
const site = "https://npatta01.github.io/trec_rag_2026/";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const required = [
  "# NP Labs · TREC RAG 2026",
  "119",
  "1,024 words",
  "Five of our files were accepted by Evalbase",
  // the film, as a still that links to the player on the site
  "reports/2026-trec-rag-approach-still.jpg",
  `[▶ Watch the approach](${site}#approach)`,
  `## [Project site →](${site})`,
  "[Paper (PDF)](output/pdf/2026-trec-rag-paper.pdf)",
  "reports/2026-trec-rag-talk.html",
  "## How the system works",
  // the compact landscape view, not the report's tall whole-system figure
  "reports/2026-competition-architecture/00-system-at-a-glance.svg",
  "2026-competition-architecture.html",
  "## What is in the repo",
  "## Checks",
  "[`AGENTS.md`](AGENTS.md)",
];

for (const value of required) {
  assert(source.includes(value), `README missing: ${value}`);
}

// The film and the system diagram are the two pictures; the film comes first
// because it is the thing somebody can watch without deciding to read.
assert(
  source.indexOf("2026-trec-rag-approach-still.jpg") < source.indexOf("00-system-at-a-glance.svg"),
  "the film should sit above the architecture diagram",
);
// A README figure has to be glanceable. The report's own whole-system diagram
// is 800x1180 and runs ~900px down a README column before anything else shows.
assert(
  !source.includes("01-whole-system.svg"),
  "the README takes the landscape figure; the tall one belongs to the report",
);
assert(
  source.indexOf("Project site") < source.indexOf("## What is in the repo"),
  "the site link must come before the repository detail",
);

const lines = source.split("\n").length;
assert(lines <= 75, `README should stay at or under 75 lines, found ${lines}`);

const links = [...source.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)].map((m) => m[1]);
assert(links.length <= 15, `README should carry at most 15 links, found ${links.length}`);
assert(!source.includes("tar --zstd"), "archive recipes belong in docs/run-archive.md");

for (const target of links) {
  if (/^(https?:|#)/.test(target)) continue;
  assert(fs.existsSync(path.join(root, target)), `README target missing: ${target}`);
}

// A GitHub-hosted attachment does render as a <video> here, but the player is
// muted by force, opens on a black box because GitHub strips poster, and sizes
// itself to whatever metadata it has managed to load. A still that links to the
// player on the project site shows the film, keeps the sound, and works in any
// renderer — so the README embeds one and no <video> tag.
assert(!/<video[\s>]/.test(source), "the README embeds a still that links to the film, not a player");

console.log("README front-door contract passed");

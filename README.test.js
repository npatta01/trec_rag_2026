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
  `[▶ Watch it with sound](${site}#approach)`,
  `## [Project site →](${site})`,
  "[Paper (PDF)](output/pdf/2026-trec-rag-paper.pdf)",
  "reports/2026-trec-rag-talk.html",
  "## How the system works",
  "reports/2026-competition-architecture/01-whole-system.svg",
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
  source.indexOf("<video") < source.indexOf("01-whole-system.svg"),
  "the film should sit above the architecture diagram",
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

// GitHub renders <video> in a README only when the src is one of its own
// uploaded attachments; every other source is stripped and would ship a blank
// gap. Checked against api.github.com/markdown and a live rendered README.
const videos = [...source.matchAll(/<video[^>]*\bsrc="([^"]+)"/g)].map((m) => m[1]);
assert(videos.length === 1, `README should carry exactly one player, found ${videos.length}`);
assert(
  videos[0].startsWith("https://github.com/user-attachments/assets/"),
  `a README <video> src must be a GitHub attachment, found ${videos[0]}`,
);

console.log("README front-door contract passed");

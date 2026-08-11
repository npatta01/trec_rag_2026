const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const source = fs.readFileSync(path.join(root, "README.md"), "utf8");
const hubLabel = "View the project artifact hub";
const hubUrl = "https://npatta01.github.io/trec_rag_2026/";
const required = [
  "# NP Labs · TREC RAG 2026",
  "NP Labs submission for TREC RAG 2026",
  `[${hubLabel}](${hubUrl})`,
  "## The problem",
  "[![The NP Labs TREC RAG 2026 system",
  "reports/2026-competition-architecture/01-whole-system.svg",
  "## Project at a glance",
  "| Official narratives | 119 |",
  "| Retrieval runs | 3 |",
  "| RAG runs | 2 |",
  "| Evalbase-accepted organizer files | 5 |",
  "Architecture Report",
  "Retrieval Quality Report",
  "RAGDoll Evaluation — Single-pass RAG",
  "RAGDoll Evaluation — Multi-stage RAG",
  "submissions/trec-rag-2026/SUBMISSION_LEDGER.md",
  "reports/2026-competition-architecture.qmd",
  ".agents/skills/validate-trec-rag-2026-submissions/SKILL.md",
  "code/trec_rag/README.md",
  "## Developer notes",
];

for (const value of required) {
  if (!source.includes(value)) throw new Error(`README missing: ${value}`);
}
if (source.indexOf(hubLabel) > source.indexOf("SUBMISSION_LEDGER.md")) {
  throw new Error("primary hub link must precede repository detail links");
}
for (const match of source.matchAll(/\[[^\]]+\]\((?!https?:|#)([^)]+)\)/g)) {
  if (!fs.existsSync(path.join(root, match[1]))) {
    throw new Error(`README target missing: ${match[1]}`);
  }
}

console.log("README artifact-first contract passed");

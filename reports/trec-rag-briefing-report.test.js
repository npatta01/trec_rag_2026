const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "trec-rag-briefing-report.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "trec-rag-briefing-report.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");

const requiredSections = [
  'id="start-here"',
  'id="plain-english"',
  'id="climbmix"',
  'id="data-at-a-glance"',
  'id="samples"',
  'id="year-2025"',
  'id="pipeline"',
  'id="implementation-kit"',
  'id="evaluation"',
  'id="failure-diagnosis"',
  'id="appendix"',
];

for (const section of requiredSections) {
  assert(html.includes(section), `Missing required briefing section ${section}`);
}

const requiredStyleSignals = [
  "TREC RAG 2026 Briefing",
  "class=\"topbar\"",
  "class=\"hero\"",
  "class=\"shell hero-grid\"",
  "class=\"route-grid\"",
  "class=\"label",
  "class=\"button primary\"",
  "class=\"pipeline-diagram\"",
  "class=\"stage-button",
  "class=\"definition-panel\"",
  "class=\"leaderboard-note\"",
  "class=\"code-block\"",
];

for (const signal of requiredStyleSignals) {
  assert(html.includes(signal), `Missing 2025-style signal: ${signal}`);
}

const requiredContentSignals = [
  "Start Here: How to Use This Briefing",
  "RAG In Plain English",
  "ClimbMix is the 2026 evidence collection",
  "Collection And Dev Set At A Glance",
  "553,240,576 documents",
  "22 narrative topics",
  "26,341 judgments",
  "26,151 unique ClimbMix docids",
  "1,255 nuggets across 22 topics",
  "30 prompts with 755 rubric criteria",
  "Dev-set document count nuance",
  "the released development data does not define a separate mini corpus",
  "Environment and energy",
  "Health and safety",
  "Society and policy",
  "Technology and media",
  "Example development topics",
  "Environmental and health impacts of e-waste",
  "Nuclear energy pros, cons, safety",
  "Korean War origins, ending, US involvement",
  "de-identified Electronic Health Records",
  "autonomous AI agents that interact with the open internet",
  "plant-based meat startup",
  "Sample Documents And Organizer Answers",
  "RAG25 dev qid 58",
  "shard_05975_34428",
  "What a retrieved ClimbMix document actually looks like",
  "shard_04044_42652",
  "Fission reactions occur when heavy atomic nuclei",
  "GET /v1/climbmix-400b/doc/shard_04044_42652",
  "sample-q58-rag-v1",
  "The 2025 Lesson Library",
  "The 2026 Build Pipeline",
  "Implementation Kit",
  "Failure Diagnosis",
  "Citation support is not coverage",
  "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
  "trec-rag-data/trec-rag-2026/development-data/topics/research-rubrics-topics-dev.tsv",
  "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/README.md",
  "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle",
  "trec-rag-2025-writeups/interactive-writeup.html",
];

for (const signal of requiredContentSignals) {
  assert(html.includes(signal), `Missing briefing content signal: ${signal}`);
}

const requiredInteractions = [
  'data-term="narrative"',
  'data-term="nugget"',
  'data-step="baseline"',
  'data-step="coverage"',
  "function showGlossary",
  "function selectPipelineStep",
  "addEventListener",
];

for (const signal of requiredInteractions) {
  assert(html.includes(signal), `Missing interaction signal: ${signal}`);
}

const forbiddenSignals = [
  'class="sidebar"',
  'class="layout"',
  "PYSERINI_API_TOKEN",
  "Authorization",
  "Bearer ",
  "TODO",
  "placeholder",
];

for (const signal of forbiddenSignals) {
  assert(!html.includes(signal), `Briefing should not include ${signal}`);
}

console.log("trec-rag-briefing-report smoke test passed");

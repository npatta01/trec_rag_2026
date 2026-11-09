const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "2026-trec-rag-talk.html");
const qmdPath = path.join(root, "2026-trec-rag-talk.qmd");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(qmdPath), "the deck source 2026-trec-rag-talk.qmd should exist");
assert(fs.existsSync(htmlPath), "the rendered 2026-trec-rag-talk.html should exist");

const qmd = fs.readFileSync(qmdPath, "utf8");
const html = fs.readFileSync(htmlPath, "utf8");

// Source files the deck is built from.
for (const asset of [
  "2026-trec-rag-talk/theme.scss",
  "2026-trec-rag-talk/01-approach.svg",
  "2026-trec-rag-talk/02-one-lane.svg",
  "2026-trec-rag-talk/03-two-writers.svg",
]) {
  assert(fs.existsSync(path.join(root, asset)), `${asset} should exist`);
}

// Self-contained: nothing may load from the network at display time.
const runtimeRefs = (html.match(/(?:src|href)="https?:\/\/[^"]+"/g) || []).filter(
  (ref) => !ref.includes("npatta01.github.io")
);
assert(runtimeRefs.length === 0,
  `deck must not load external resources, found: ${runtimeRefs.slice(0, 3).join(", ")}`);

// Diagrams carry alt text for screen readers and must not print a caption.
const figAlts = qmd.match(/fig-alt="/g) || [];
assert(figAlts.length === 3, `expected 3 fig-alt diagrams, found ${figAlts.length}`);
assert(!/!\[[^\]]+\]\(/.test(qmd),
  "diagrams must use ![](src){fig-alt=...}, never a bracketed caption");

// The paired structure: every stage has a 'how it works' and an example slide.
for (const stage of ["Planning", "Searching", "Evidence and nuggets", "Two writers"]) {
  assert(qmd.includes(`## ${stage}: how it works`) || qmd.includes(`## ${stage}: how it works`),
    `missing mechanism slide for ${stage}`);
}
for (const stage of ["Planning", "Two writers", "Evidence and nuggets"]) {
  assert(qmd.includes(`## ${stage}: this request`),
    `missing worked-example slide for ${stage}`);
}

// Both worked examples, and the numbers quoted for them.
const requiredFacts = [
  "rag2026-17",
  "rag2026-13",
  "out-of-home placement",
  "0.833", "1.000", "0.667",      // rag2026-17 rubric and field median
  "6 / 4 / 0",                     // rag2026-17 head-to-head
  "8 / 0 / 4",                     // rag2026-13 head-to-head, humans favour multi-stage
  "0.7740", "0.8738",              // Batch 1 all-strict
  "0.5705", "0.6578",              // Batch 2 human nuggets all-strict
  "0.4116", "0.3499",              // retrieval precision vs median
  "0.757", "0.653",                // RAGDoll citation support, favours single-pass
];
for (const fact of requiredFacts) {
  assert(qmd.includes(fact), `deck should quote: ${fact}`);
}

// Prose wraps across lines in the source, so compare on collapsed whitespace.
const flat = qmd.replace(/\s+/g, " ");

// Evidence boundaries that must survive editing.
const requiredBoundaries = [
  "not a claim that it is novel",
  "ceilings, not yields",
  "only one of 42 scored topics",
  "rubric items were not released",
  "114 evaluated runs from 27 teams",
  "should not be read as better grounding",
];
for (const boundary of requiredBoundaries) {
  assert(flat.includes(boundary), `deck should keep the boundary note: ${boundary}`);
}

// Slide count: 1 title + 17 content.
const sections = html.match(/<section[^>]*class="[^"]*slide/g) || [];
assert(sections.length >= 17,
  `expected at least 17 content slides, found ${sections.length}`);

console.log("reports/2026-trec-rag-talk.test.js OK");

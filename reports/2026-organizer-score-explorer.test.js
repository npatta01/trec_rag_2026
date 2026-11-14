const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "2026-organizer-score-explorer.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "reports/2026-organizer-score-explorer.html should exist");
const html = fs.readFileSync(htmlPath, "utf8");

// The page must stand alone: no network at runtime.
assert(!/<script[^>]+\bsrc=/i.test(html), "no external scripts");
assert(!/<link[^>]+stylesheet/i.test(html), "no external stylesheets");
assert(!/https?:\/\/(?!arxiv|dl\.acm)/i.test(html.replace(/<p>Source:[\s\S]*?<\/p>/g, "")),
  "no runtime network references outside prose");

// Framing that keeps the numbers honest.
const requiredSignals = [
  "Organizer Score Explorer",
  "Read each collection on its own",
  "114 evaluated runs from 27 teams",
  "no metric is recomputed here",
  "Bradley–Terry",
];
for (const signal of requiredSignals) {
  assert(html.includes(signal), `explorer should state: ${signal}`);
}

// Embedded data: shape and counts.
const match = html.match(
  /<script type="application\/json" id="data">([\s\S]*?)<\/script>/
);
assert(match, "embedded data block should exist");
const data = JSON.parse(match[1]);

assert(data.rows.length === 530,
  `expected 530 per-narrative rows, got ${data.rows.length}`);
assert(data.collections.length === 6,
  `expected 6 collections, got ${data.collections.length}`);

const perCollection = {};
for (const row of data.rows) {
  perCollection[row.c] = (perCollection[row.c] || 0) + 1;
}
const expected = {
  b1_rubric: 34,   // 17 narratives x 2 RAG runs
  b2_manual: 50,   // 25 x 2
  b2_auto: 50,     // 25 x 2
  doc: 246,        // 82 narratives x 3 retrieval runs
  ngt_man: 75,     // 25 x 3
  ngt_auto: 75,    // 25 x 3
};
for (const [id, count] of Object.entries(expected)) {
  assert(perCollection[id] === count,
    `collection ${id}: expected ${count} rows, got ${perCollection[id]}`);
}

// The organizer's own overall rows must survive into the summary.
for (const id of Object.keys(expected)) {
  assert(data.summary[id], `summary missing collection ${id}`);
}
const b2 = data.summary.b2_manual;
assert(Math.abs(b2["rag26-ms1-final"].values.all_strict - 0.6578) < 5e-4,
  "multi-stage Batch-2 manual all_strict should match the released 0.6578");
assert(Math.abs(b2["rag26-ss1"].values.all_strict - 0.5705) < 5e-4,
  "single-pass Batch-2 manual all_strict should match the released 0.5705");

// Cost measures must never be ranked as quality.
assert(data.costMetrics.includes("effort") && data.costMetrics.includes("num_ret"),
  "effort and num_ret should be declared cost metrics");
for (const metric of data.costMetrics) {
  assert(!data.higherIsBetter.includes(metric),
    `${metric} is a cost measure and must not be scored as better-when-higher`);
}

console.log("reports/2026-organizer-score-explorer.test.js OK");

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
  "2026-trec-rag-talk/04-plan.svg",
  "2026-trec-rag-talk/05-evidence.svg",
  "2026-trec-rag-talk/06-retrieval.svg",
  "2026-trec-rag-talk/07-scene.svg",
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
assert(figAlts.length === 7, `expected 7 fig-alt figures, found ${figAlts.length}`);
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
  "rag2026-18",
  "rag2026-13",
  "lead hazards", "made lead-safe",
  "0.857", "1.000", "0.929", "0.786",   // rag2026-18 nuggets vs field median
  "62.9%", "45.0%",                      // assessor battle win rates
  "0.4116", "0.3499",                    // retrieval precision vs median
  "0.757", "0.653",                      // RAGDoll citation support, favours single-pass
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
  "one of 42 scored topics",
  "rubric items were not released",
  "should not be read as better grounding",
  // The battle counts are per-run against the field, never head to head.
  "not against the other one of",
];
for (const boundary of requiredBoundaries) {
  assert(flat.includes(boundary), `deck should keep the boundary note: ${boundary}`);
}

// Slide count: 1 title + 17 content.
const sections = html.match(/<section[^>]*class="[^"]*slide/g) || [];
assert(sections.length >= 17,
  `expected at least 17 content slides, found ${sections.length}`);

// The organizers define wins/ties/losses as battles against the whole field,
// so the deck must never describe them as a duel between our two runs.
assert(!/head-to-head/i.test(qmd),
  "side-by-side counts are per-run against the field, not head-to-head");

// The worked example was changed; no slide or figure may still cite the old one.
for (const stale of ["rag2026-17", "out-of-home", "maltreatment", "preterm", "NICU"]) {
  assert(!qmd.includes(stale), `deck still references the retired example: ${stale}`);
}
for (const fig of ["01-approach", "02-one-lane", "04-plan", "05-evidence", "06-retrieval", "07-scene"]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  for (const stale of ["maltreatment", "preterm", "NICU"]) {
    assert(!svg.includes(stale), `${fig}.svg still references the retired example: ${stale}`);
  }
}

// The planning example must stay expandable, one entry per subnarrative,
// and must state the per-request query arithmetic rather than a bare total.
const details = (html.match(/<details>/g) || []).length;
assert(details === 8, `expected 8 expandable subnarratives, found ${details}`);
assert(/8 subnarratives . 2 queries = 16 searches/.test(qmd),
  "the planning slide must show the per-request query arithmetic");

// Per-request figures must never read as totals for the whole run.
const funnel = fs.readFileSync(path.join(root, "2026-trec-rag-talk/01-approach.svg"), "utf8");
assert(/ONE REQUEST/.test(funnel) && /per-request ceiling/.test(funnel),
  "the funnel must say its figures are per request, not totals for the run");

console.log("reports/2026-trec-rag-talk.test.js OK");

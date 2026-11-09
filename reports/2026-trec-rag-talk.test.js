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
  "2026-trec-rag-talk/img-request.png",
  "2026-trec-rag-talk/img-writers.png",
]) {
  assert(fs.existsSync(path.join(root, asset)), `${asset} should exist`);
}

// Self-contained: nothing may be FETCHED from the network at display time.
// Anchors are fine and expected — the closing slide links out on purpose.
const fetched = [
  ...(html.match(/\ssrc="https?:\/\/[^"]+"/g) || []),
  ...(html.match(/<link\b[^>]*\shref="https?:\/\/[^"]+"/g) || []),
];
assert(fetched.length === 0,
  `deck must not load external resources, found: ${fetched.slice(0, 3).join(", ")}`);

// Diagrams carry alt text for screen readers and must not print a caption.
const figAlts = qmd.match(/fig-alt="/g) || [];
assert(figAlts.length === 8, `expected 8 fig-alt figures, found ${figAlts.length}`);
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
for (const fig of ["01-approach", "02-one-lane", "04-plan", "05-evidence", "06-retrieval"]) {
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
// The trace records 17 planned queries for this request: 16 plus the
// untouched-narrative lane. The slide must not stop at 16.
assert(/= 16 searches, plus the untouched request = 17 lanes/.test(qmd),
  "the planning slide must account for the untouched-narrative lane");

// The 40 budget is a SELECTED PASSAGE cap. In the handoff schema a "group" is
// a subnarrative (group_id == subnarrative_id), one per subnarrative on all
// 119 topics, so "40 groups per subnarrative" is wrong in two ways.
for (const f of ["01-approach", "02-one-lane", "05-evidence"]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${f}.svg`), "utf8");
  assert(!/groups per subnarrative/.test(svg),
    `${f}.svg must not call the 40-passage budget a group cap`);
}
assert(!/evidence groups/.test(qmd),
  "the deck must not call selected passages 'evidence groups'");

// Near-duplicate detection is a named model at a named threshold.
const evidence = fs.readFileSync(path.join(root, "2026-trec-rag-talk/05-evidence.svg"), "utf8");
for (const mech of ["all-MiniLM-L6-v2", "0.92", "MMR"]) {
  assert(evidence.includes(mech),
    `the evidence figure should name the dedup mechanism: ${mech}`);
}

// Per-topic audit counts are published, so the deck must not claim otherwise;
// only the kept/rejected decisions are private.
assert(!/no per-topic audit log/.test(qmd),
  "the per-topic audit counts are published in the release; don't deny them");
assert(flat.includes("every evidence group — no sampling"),
  "the Luna slide should state the audit breadth the trace records");
assert(flat.includes("its one retry on 2 of 119 requests"),
  "the writers slide should quote the measured single-pass retry count");

// Per-request figures must never read as totals for the whole run.
const funnel = fs.readFileSync(path.join(root, "2026-trec-rag-talk/01-approach.svg"), "utf8");
assert(/ONE REQUEST/.test(funnel) && /configured ceiling/.test(funnel),
  "the funnel must frame its figures as configured ceilings for one request");
// It introduces the approach before the example, so it must carry no
// example-specific counts.
for (const leak of [">16<", ">16,000<", ">1,600<", ">320<"]) {
  assert(!funnel.includes(leak),
    `the approach funnel must not show one request's own numbers: ${leak}`);
}

// ---- the two-writers figure ----------------------------------------------
// It names three actors and both models, and draws the five handoffs that the
// bounded-revision state machine actually records
// (planner, draft, audit_groups, revision, operation_screen).
const writers = fs.readFileSync(path.join(root, "2026-trec-rag-talk/03-two-writers.svg"), "utf8");
for (const fact of [
  "openai/gpt-5.6-luna",        // Luna, the reviewer
  "openai/gpt-5.6-sol",         // Sol, the writer
  "Plans the answer",
  "Writes the draft",
  "Audits every group",
  "Proposes edits",
  "Screens each edit",
  "≤6 missing-detail cards each", // MAX_AUDIT_CARDS_PER_GROUP
  "local applies only what passed",
]) {
  assert(writers.includes(fact), `the two-writers figure should show: ${fact}`);
}
// The state machine has one boolean per stage and no round counter, so the
// figure must not imply the revision repeats.
assert(/one pass, not a repeating loop/.test(writers),
  "the two-writers figure must say the relay is a single pass");
for (const n of ["1", "2", "3", "4", "5"]) {
  assert(writers.includes(`>${n}</text>`), `the relay should number step ${n}`);
}

// Role glyphs must beat Reveal's own `.reveal section svg { display: block }`,
// or they drop onto a line of their own, centred.
const theme = fs.readFileSync(path.join(root, "2026-trec-rag-talk/theme.scss"), "utf8");
assert(/\.reveal section svg\.glyph/.test(theme),
  "the glyph rule must match Reveal's specificity for section svg");
assert(/svg\.glyph\{[^}]*display:inline-block/.test(html.replace(/\s+/g, "")) ||
       /display%3Ainline%2Dblock/.test(html),
  "the compiled deck should render glyphs inline");
const glyphs = (qmd.match(/class="glyph"/g) || []).length;
assert(glyphs === 4, `expected 4 role glyphs in the deck, found ${glyphs}`);

console.log("reports/2026-trec-rag-talk.test.js OK");

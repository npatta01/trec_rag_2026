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
  "2026-trec-rag-talk/00-task.svg",
  "2026-trec-rag-talk/00-architecture.svg",
  "2026-trec-rag-talk/img-request.png",
  "2026-trec-rag-talk/img-writers.png",
  "2026-trec-rag-talk/img-request-forest.png",
  "2026-trec-rag-talk/img-writers-forest.png",
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
assert(figAlts.length === 10, `expected 10 fig-alt figures, found ${figAlts.length}`);
// The palette-variant scenes are decorative duplicates of a described image,
// so they are hidden from screen readers rather than described twice. (The
// role glyphs are aria-hidden too, hence matching on the scene class.)
const variants = (qmd.match(/\{\.scene-forest aria-hidden="true"\}/g) || []).length;
assert(variants === 2, `expected 2 hidden scene variants, found ${variants}`);
const primaries = (qmd.match(/\{\.scene-slate fig-alt=/g) || []).length;
assert(primaries === 2, `expected 2 described scene images, found ${primaries}`);
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
assert(sections.length >= 20,
  `expected at least 20 content slides, found ${sections.length}`);

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
const details = (html.match(/<details\b/g) || []).length;
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

// ---- palettes and inlined figures -----------------------------------------
// A linked <img src="x.svg"> is a separate document, so the page's CSS custom
// properties cannot reach inside it and no palette switch could repaint the
// diagrams. inline-svg.lua inlines them instead.
assert(!/<img[^>]+src="[^"]*\.svg"/i.test(html),
  "figures must be inlined, not linked, or the palette cannot reach them");
const inlined = (html.match(/<svg\b/gi) || []).length;
assert(inlined >= 6, `expected the 6 figures inlined, found ${inlined} <svg> elements`);

// Each figure keeps its own accessible name once inlined.
for (const id of ["ap", "ln", "tw", "pl", "ev", "rt"]) {
  assert(html.includes(`id="${id}-title"`) && html.includes(`id="${id}-desc"`),
    `inlined figure ${id} should keep its <title> and <desc>`);
}

// Six style blocks now share one document, so every figure class is namespaced.
for (const [fig, tok] of [["01-approach", "ap"], ["02-one-lane", "ln"],
                          ["03-two-writers", "tw"], ["04-plan", "pl"],
                          ["05-evidence", "ev"], ["06-retrieval", "rt"]]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  const classes = new Set();
  for (const m of svg.matchAll(/class="([^"]*)"/g)) {
    m[1].split(/\s+/).filter(Boolean).forEach((c) => classes.add(c));
  }
  const bare = [...classes].filter((c) => !c.startsWith(`${tok}-`));
  assert(bare.length === 0,
    `${fig}.svg has un-namespaced classes that would collide once inlined: ${bare.join(", ")}`);
}

// The figures' own colour blocks are dropped on the way in; the page owns them.
assert(!/:root\s*\{\s*color-scheme/.test(html),
  "inlined figures must not carry their own :root palette into the page");
// ...but they keep it on disk, so each file still renders when opened alone.
for (const fig of ["01-approach", "03-two-writers", "05-evidence"]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  assert(/:root\s*\{[^}]*--bg:/.test(svg),
    `${fig}.svg should keep its own :root so it renders standalone`);
}

// Two palettes, and they must actually differ in the dominant accent.
const scss = fs.readFileSync(path.join(root, "2026-trec-rag-talk/theme.scss"), "utf8");
for (const mix of ["slate-light", "slate-dark", "forest-light", "forest-dark"]) {
  assert(scss.includes(`@mixin ${mix}`), `theme should define the ${mix} palette`);
}
const retOf = (mix) => {
  const body = scss.split(`@mixin ${mix} {`)[1].split("}")[0];
  return (body.match(/--ret:\s*(#[0-9a-f]{6})/i) || [])[1];
};
assert(retOf("slate-dark") !== retOf("forest-dark"),
  "the palettes must differ in --ret, the accent used most in the figures");
assert(/:root\[data-palette="forest"\]/.test(scss),
  "Forest must be selectable by attribute");

// The control itself.
assert(/id="palette-toggle"/.test(html), "the palette control should be in the deck");
assert(/trec-talk-palette/.test(html), "the chosen palette should be remembered");

// ---- pinned light/dark -----------------------------------------------------
// The deck is often shown on a machine whose system setting is the opposite of
// what the room needs, so a pinned choice has to beat prefers-color-scheme.
assert(/id="theme-toggle"/.test(html), "the brightness control should be in the deck");
assert(/trec-talk-theme/.test(html), "the brightness choice should be remembered");

// The media query must yield to a pin, or pinning light on a dark machine does
// nothing.
assert(/@media \(prefers-color-scheme: dark\)[\s\S]{0,200}:root:not\(\[data-theme="light"\]\)/.test(scss),
  "the dark media query must step aside for a pinned light theme");

// And the forced-dark rules must come after the media query, since they carry
// the same specificity and order is what decides.
const mediaAt = scss.indexOf("@media (prefers-color-scheme: dark)");
const pinnedAt = scss.indexOf(':root[data-theme="dark"] {');
assert(mediaAt > 0 && pinnedAt > mediaAt,
  "forced-dark rules must be written after the dark media query to win on ties");
assert(/:root\[data-theme="dark"\]\[data-palette="forest"\]/.test(scss),
  "each palette needs its own forced-dark rule");
assert(/:root\[data-theme="dark"\]\s*\{\s*color-scheme: dark/.test(scss),
  "color-scheme should follow a pinned theme, not the system");

// Quarto colours a bare <summary> a slate blue that falls to 2.74:1 on a dark
// panel. These are the clickable questions, so the deck sets its own.
assert(/\.qgrid summary \{[\s\S]*?color: var\(--ink\)/.test(scss),
  "the expandable questions must set their own colour, not inherit Quarto's");

// ---- the task and architecture figures -------------------------------------
const task = fs.readFileSync(path.join(root, "2026-trec-rag-talk/00-task.svg"), "utf8");
for (const fact of ["ClimbMix", "BM25", "119", "1,024", "every statement cited"]) {
  assert(task.includes(fact), `the task figure should name: ${fact}`);
}
const arch = fs.readFileSync(path.join(root, "2026-trec-rag-talk/00-architecture.svg"), "utf8");
for (const fact of ["deepseek-v4-flash", "Pyserini BM25", "mxbai-rerank-base-v2",
                    "all-MiniLM-L6-v2", "gpt-5.6-sol", "gpt-5.6-luna", "local GPU"]) {
  assert(arch.includes(fact), `the architecture figure should name: ${fact}`);
}
// Figure classes stay namespaced, including the two new ones.
for (const [fig, tok] of [["00-task", "tk"], ["00-architecture", "ar"]]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  const bare = [...new Set([].concat(...[...svg.matchAll(/class="([^"]*)"/g)]
    .map((m) => m[1].split(/\s+/).filter(Boolean))))].filter((c) => !c.startsWith(`${tok}-`));
  assert(bare.length === 0, `${fig}.svg has un-namespaced classes: ${bare.join(", ")}`);
}

// The planning queries are shown, not hidden behind a click.
const openDetails = (qmd.match(/<details open>/g) || []).length;
assert(openDetails === 8, `expected 8 pre-expanded subnarratives, found ${openDetails}`);

// The dedup step is shown concretely, and flagged as illustrative wording
// because real corpus passages stay private.
assert(/cosine \*\*0\.94\*\*/.test(qmd), "the evidence slide should show a worked duplicate pair");
assert(flat.includes("my wording, standing in for real passages"),
  "the illustrative sentences must be marked as not being corpus text");

// The nuggetizer's attribution limit, straight from the config.
assert(/each attributed to at most \*\*3\*\*/.test(qmd),
  "the nugget slide should state maximum_supporting_documents_per_claim");

console.log("reports/2026-trec-rag-talk.test.js OK");

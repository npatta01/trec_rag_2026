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
  "2026-trec-rag-talk/02-one-lane.svg",
  "2026-trec-rag-talk/03a-single-pass.svg",
  "2026-trec-rag-talk/03b-multi-stage.svg",
  "2026-trec-rag-talk/07-topic18.svg",
  "2026-trec-rag-talk/08-disagree.svg",
  "2026-trec-rag-talk/09-overall.svg",
  "2026-trec-rag-talk/02b-rerank.svg",
  "2026-trec-rag-talk/11a-retrieval-cutoff.svg",
  "2026-trec-rag-talk/13-cost.svg",
  "2026-trec-rag-talk/15-coverage.svg",
  "2026-trec-rag-talk/16-luna.svg",
  "2026-trec-rag-talk/04-plan.svg",
  "2026-trec-rag-talk/05-evidence.svg",
  "2026-trec-rag-talk/06-retrieval.svg",
  "2026-trec-rag-talk/00-task.svg",
  "2026-trec-rag-talk/00-architecture.svg",
  "2026-trec-rag-talk/img-request.png",
  "2026-trec-rag-talk/img-request-forest.png",
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
assert(figAlts.length === 22, `expected 22 fig-alt figures, found ${figAlts.length}`);
// The palette-variant scenes are decorative duplicates of a described image,
// so they are hidden from screen readers rather than described twice. (The
// role glyphs are aria-hidden too, hence matching on the scene class.)
// Only the request scene is left: the two writers at desks were cut, because
// Sol and Luna are models and drawing them as people at desks obscured the
// stage machine the slide is actually about.
const variants = (qmd.match(/\{\.scene-forest aria-hidden="true"\}/g) || []).length;
assert(variants === 1, `expected 1 hidden scene variant, found ${variants}`);
const primaries = (qmd.match(/\{\.scene-slate fig-alt=/g) || []).length;
assert(primaries === 1, `expected 1 described scene image, found ${primaries}`);
assert(!/img-writers/.test(qmd) && !fs.existsSync(path.join(root, "2026-trec-rag-talk/img-writers.png")),
  "the desk illustration was cut; nothing should still point at it");
assert(!/!\[[^\]]+\]\(/.test(qmd),
  "diagrams must use ![](src){fig-alt=...}, never a bracketed caption");

// The paired structure: the mechanism, then the same stage on the real
// request. Not every stage earns both - reranking has no interesting example
// and the nugget hints lost theirs - but the pairs that exist must stay paired.
for (const stage of ["Planning", "Searching"]) {
  assert(qmd.includes(`## ${stage}: how it works`), `missing mechanism slide for ${stage}`);
}
for (const head of ["## Single-pass: Sol alone", "## Multi-stage: the work changes hands",
                    "## The two places Luna is used", "## Two writers: this request",
                    "## Reranking: a document becomes passages",
                    "## Selecting the evidence"]) {
  assert(qmd.includes(head), `missing slide: ${head}`);
}
for (const stage of ["Planning", "Two writers"]) {
  assert(qmd.includes(`## ${stage}: this request`),
    `missing worked-example slide for ${stage}`);
}

// Both worked examples, and the numbers quoted for them.
// The scored numbers moved out of tables and into charts, so each is checked
// where it now lives. A number that is only in prose is no longer a number the
// room can see.
const requiredFacts = [
  "rag2026-18",
  "rag2026-13",
  "lead hazards", "made lead-safe",
  "22 statements · 504 words", "38 statements · 800 words",
];
const chartedFacts = [
  ["07-topic18", ["0.857", "1.000", "0.929", "0.786", "7 preferred", "3 lost", "6 preferred"]],
  ["08-disagree", ["1.000", "0.857", "5 of 10", "8 of 12"]],
  ["09-overall", ["62.9%", "45.0%", "399", "422", "26", "15"]],
  ["06-retrieval", ["0.4116", "0.3499"]],
];
for (const [fig, facts] of chartedFacts) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  for (const fact of facts) {
    assert(svg.includes(fact), `${fig}.svg should plot: ${fact}`);
  }
}
// 26 + 15 + 1 = 42, drawn as one square per topic rather than asserted.
const grid = fs.readFileSync(path.join(root, "2026-trec-rag-talk/09-overall.svg"), "utf8");
const cells = (grid.match(/<rect x="\d+" y="\d+" width="30" height="30"/g) || []).length;
assert(cells === 42, `the overall figure should draw 42 topic squares, found ${cells}`);
// The citation-support result (RAGDoll favoured single-pass, 0.757 to 0.653)
// came off the deck at the author's request - the close is the architecture
// and nothing else. It is still reachable through the linked reports. What the
// deck must never do is claim the opposite.
assert(!/better grounding|better.grounded|more faithful/i.test(qmd),
  "the deck must not claim multi-stage was better grounded - it was not");
assert(/2026-ragdoll-rag26-ss1/.test(qmd) && /2026-ragdoll-rag26-ms1-final/.test(qmd),
  "the citation-support reports must stay linked, since the number is no longer stated");
// Prose wraps across lines in the source, so compare on collapsed whitespace.
const flat = qmd.replace(/\s+/g, " ");
for (const fact of requiredFacts) {
  assert(flat.includes(fact), `deck should quote: ${fact}`);
}

// Evidence boundaries that must survive editing.
const requiredBoundaries = [
  // The framing lives with the aggregate numbers now, not on the task slide.
  "a claim that it is novel",
  "the organizers' own, from the score files",
  "42 scored topics",
  "rubric items were not released",
  // The citation-support limit came off the deck entirely at the author's
  // request; the check above pins what the deck must not claim instead.
  // The battle counts are per-run against the field, never head to head. The
  // phrasing has been cut twice for reading as slop; the claim it carries has
  // to survive each rewrite, so the guard tracks the claim's current sentence.
  "not our two answers against each other",
  "other teams' answers",
];
for (const boundary of requiredBoundaries) {
  assert(flat.includes(boundary), `deck should keep the boundary note: ${boundary}`);
}

// The task slide states the task. The framing of what this talk does and does
// not claim belongs with the numbers, not in front of them.
assert(!flat.includes("This is about how the approach works"),
  "the task slide should not open with a disclaimer");

// Slide count: 1 title + 19 content. The funnel came out: slide 3 already
// names the pieces and the lane figure already draws these exact ceilings
// against one real query, so it was the same story told a third time.
const sections = html.match(/<section[^>]*class="[^"]*slide/g) || [];
assert(sections.length === 31,
  `expected 31 content slides, found ${sections.length}`);

// Four sections, named the way the work is actually divided. The contents are
// shown once, at the top; the three breaks after it carry a picture instead,
// because the same list four times taught the room nothing new.
const dividers = (html.match(/class="[^"]*\bdivider\b/g) || []).length;
assert(dividers === 6, `expected 6 dividers (contents, 4 sections, appendix), found ${dividers}`);
assert((qmd.match(/class="roadmap"/g) || []).length === 1,
  "the roadmap belongs on the opening contents slide only");
const legs = (qmd.match(/<span class="leg[ "]/g) || []).length;
assert(legs === 5, `the contents slide should list the five sections, found ${legs}`);
assert((qmd.match(/class="leg leg-now"/g) || []).length === 1,
  "the contents slide lights the section it opens");
for (const sec of ["Architecture", "Retrieval", "Answer Generation", "Overall score", "Cost"]) {
  assert(qmd.includes(`</span>${sec}</span>`), `the contents should name: ${sec}`);
}
const shots = (qmd.match(/\{\.secshot\}/g) || []).length;
assert(shots === 4, `each of the four section breaks carries one image, found ${shots}`);
assert(!/01-approach/.test(qmd) && !fs.existsSync(path.join(root, "2026-trec-rag-talk/01-approach.svg")),
  "the funnel figure was cut; nothing should still point at it");

// The organizers define wins/ties/losses as battles against the whole field,
// so the deck must never describe them as a duel between our two runs.
assert(!/head-to-head/i.test(qmd),
  "side-by-side counts are per-run against the field, not head-to-head");

// The worked example was changed; no slide or figure may still cite the old one.
for (const stale of ["rag2026-17", "out-of-home", "maltreatment", "preterm", "NICU"]) {
  assert(!qmd.includes(stale), `deck still references the retired example: ${stale}`);
}
for (const fig of ["02-one-lane", "02b-rerank", "04-plan", "05-evidence", "06-retrieval",
                   "03a-single-pass", "03b-multi-stage", "07-topic18",
                   "08-disagree", "09-overall"]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  for (const stale of ["maltreatment", "preterm", "NICU"]) {
    assert(!svg.includes(stale), `${fig}.svg still references the retired example: ${stale}`);
  }
}

// The planning example must stay expandable, one entry per subnarrative:
// two on the slide, all eight in the appendix.
const details = (html.match(/<details\b/g) || []).length;
assert(details === 10, `expected 2 + 8 expandable subnarratives, found ${details}`);
// The arithmetic moved with the full grid. The trace records 17 planned
// queries for this request: 16 plus the untouched-narrative lane, and the
// appendix must not stop at 16.
assert(/8 subnarratives . 2 queries = 16, plus the untouched request = 17/.test(qmd),
  "the appendix must account for the untouched-narrative lane");

// Planning is told over two slides: the eight angles, then the queries each
// one became. The audience tracks a question by its number and colour across
// the pair, so both slides must list the same eight in the same order.
const angleCards = [...qmd.matchAll(/<div class="qcard">.*?<\/svg><span>(.*?)<\/span><\/div>/gs)]
  .map((m) => m[1].trim());
const queryCards = [...qmd.matchAll(/<summary><svg class="glyph qico".*?<\/svg>(.*?)<\/summary>/gs)]
  .map((m) => m[1].trim());
assert(angleCards.length === 8,
  `expected 8 angle cards, found ${angleCards.length}`);
// Two on the slide, eight in the appendix: the slide's pair must be the first
// two angles, and the appendix must repeat all eight in the same order.
assert(JSON.stringify(queryCards.slice(0, 2)) === JSON.stringify(angleCards.slice(0, 2)),
  `the queries slide must show the first two angles, in order:\n` +
  `  angles:  ${JSON.stringify(angleCards.slice(0, 2))}\n  queries: ${JSON.stringify(queryCards.slice(0, 2))}`);
assert(JSON.stringify(queryCards.slice(2)) === JSON.stringify(angleCards),
  `the appendix must list the same eight angles in the same order:\n` +
  `  angles:  ${JSON.stringify(angleCards)}\n  appendix: ${JSON.stringify(queryCards.slice(2))}`);

// The 40 budget is a SELECTED PASSAGE cap. In the handoff schema a "group" is
// a subnarrative (group_id == subnarrative_id), one per subnarrative on all
// 119 topics, so "40 groups per subnarrative" is wrong in two ways.
for (const f of ["02-one-lane", "05-evidence"]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${f}.svg`), "utf8");
  assert(!/groups per subnarrative/.test(svg),
    `${f}.svg must not call the 40-passage budget a group cap`);
}
assert(!/evidence groups/.test(qmd),
  "the deck must not call selected passages 'evidence groups'");

// Near-duplicate detection is a named model at a named threshold.
const evidence = fs.readFileSync(path.join(root, "2026-trec-rag-talk/05-evidence.svg"), "utf8");
for (const mech of ["all-MiniLM-L6-v2", "0.92"]) {
  assert(evidence.includes(mech),
    `the evidence figure should name the dedup mechanism: ${mech}`);
}
// MMR came off the figure as clutter, but the step is real: it stays named in
// the caption, where it costs one word instead of four lines.
assert(flat.includes("spread by MMR"),
  "the selection step is dedup AND spread; MMR must still be named somewhere");
assert(!/each next pick/.test(evidence),
  "the four-line MMR gloss was cut from the figure");

// Per-topic audit counts are published, so the deck must not claim otherwise;
// only the kept/rejected decisions are private.
assert(!/no per-topic audit log/.test(qmd),
  "the per-topic audit counts are published in the release; don't deny them");
assert(!/\b(she|her|hers)\b/i.test(qmd.replace(/\bwhether\b/gi, "")),
  "Luna is a model; the deck should not give it a gender");
// Audit breadth was a caption line and came off with the rest of the copy.
// The figure still bounds the audit, which is the part that constrains it.
assert(/≤6 cards per group/.test(fs.readFileSync(
  path.join(root, "2026-trec-rag-talk/03b-multi-stage.svg"), "utf8")),
  "the relay figure must still bound the per-group audit");

// Every per-query figure is a configured cap, never a measured yield. That
// used to be an amber warning box on the slide; it is now carried by the ≤ on
// each number, so each one has to actually be there.
const lane = fs.readFileSync(path.join(root, "2026-trec-rag-talk/02-one-lane.svg"), "utf8");
for (const cap of ["≤1,000", "≤100", "≤40", "≤20"]) {
  assert(lane.includes(cap), `the lane figure must mark ${cap} as a ceiling`);
}
assert(!/measured yield|conceptual/.test(lane),
  "the lane figure should not re-explain the ≤ in a footnote");

// ---- the two writers, now a slide each ------------------------------------
// The relay draws the five handoffs the bounded-revision state machine records
// (planner, draft, audit_groups, revision, operation_screen) — one pass, no
// round counter — and single-pass draws the two steps it actually has.
const single = fs.readFileSync(path.join(root, "2026-trec-rag-talk/03a-single-pass.svg"), "utf8");
const writers = fs.readFileSync(path.join(root, "2026-trec-rag-talk/03b-multi-stage.svg"), "utf8");
assert(!fs.existsSync(path.join(root, "2026-trec-rag-talk/03-two-writers.svg")),
  "the combined two-writers figure was split; it should not still be on disk");

for (const fact of [
  "openai/gpt-5.6-sol",         // Sol, the writer
  "at most 1,024 words",        // the organizer word limit, checked locally
  "used on 2 of 119",           // the measured single-pass retry count
  "119 answers",
]) {
  assert(single.includes(fact), `the single-pass figure should show: ${fact}`);
}

for (const fact of [
  "openai/gpt-5.6-luna",        // Luna, the reviewer
  "openai/gpt-5.6-sol",         // Sol, the writer
  "Plans",
  // The labels were a sentence each and overran their boxes, so each step is
  // a verb with its bound underneath. The bounds are the point; keep them.
  "Drafts",
  "Audits",
  "Proposes",
  "Screens",
  "3–8 obligations",
  "≤6 cards per group",         // MAX_AUDIT_CARDS_PER_GROUP
  "at most 3",                  // MAX_SOL_RESERVATIONS
  "all five must pass",
  "applies only",               // local code applies the passed edits, nothing else
]) {
  assert(writers.includes(fact), `the multi-stage figure should show: ${fact}`);
}
assert(/not a repeating loop/.test(writers),
  "the multi-stage figure must say the relay is a single pass");
for (const n of ["1", "2", "3", "4", "5"]) {
  assert(writers.includes(`>${n}</text>`), `the relay should number step ${n}`);
}
// Each one keeps the lane labels that say who did what.
for (const [svg, who] of [[single, ["Sol", "local code"]],
                          [writers, ["Luna", "Sol", "local code"]]]) {
  for (const name of who) {
    assert(svg.includes(`>${name}</text>`), `the figure should label the lane: ${name}`);
  }
}

// Role glyphs must beat Reveal's own `.reveal section svg { display: block }`,
// or they drop onto a line of their own, centred.
const theme = fs.readFileSync(path.join(root, "2026-trec-rag-talk/theme.scss"), "utf8");
assert(/\.reveal section svg\.glyph/.test(theme),
  "the glyph rule must match Reveal's specificity for section svg");
assert(/svg\.glyph\{[^}]*display:inline-block/.test(html.replace(/\s+/g, "")) ||
       /display%3Ainline%2Dblock/.test(html),
  "the compiled deck should render glyphs inline");
const glyphs = (qmd.match(/class="glyph[ "]/g) || []).length;
assert(glyphs >= 15, `expected the deck's symbol set, found ${glyphs} glyphs`);

// ---- the symbol vocabulary -------------------------------------------------
// Sol means sun and Luna means moon, so the marks are the names. They have to
// agree between the figure and the prose, and the old nib/magnifier pair that
// carried no such meaning must be gone from both.
assert(/id="sp-g-sun"/.test(single),
  "the single-pass figure must mark Sol as a sun");
assert(/id="ms-g-sun"/.test(writers) && /id="ms-g-moon"/.test(writers),
  "the multi-stage figure must mark Sol as a sun and Luna as a moon");
// Two suns and one moon: both cards on the two-writers slide. The slide about
// Luna's two jobs is one figure now, which carries its own marks.
assert((qmd.match(/class="glyph g-sun"/g) || []).length === 2 &&
       (qmd.match(/class="glyph g-moon"/g) || []).length === 1,
  "Sol and Luna keep their marks on the slides that name them");
for (const src of [qmd, single, writers]) {
  assert(!/M0 -13 L8 3 L0 12/.test(src) && !/cx="-2" cy="-3" r="8\.5"/.test(src),
    "the old pen-nib and magnifier glyphs should be gone");
}
// Each stage of each pipeline figure carries a symbol, not just a coloured box.
for (const [fig, tok, want] of [["00-architecture", "ar", 5], ["02-one-lane", "ln", 4]]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  const used = (svg.match(new RegExp(`<use href="#${tok}-g-`, "g")) || []).length;
  assert(used >= want, `${fig}.svg should give each stage a symbol, found ${used}`);
}
// Every subnarrative card gets its own symbol, so eight open cards do not read
// as one wall of identical boxes.
const qicons = (qmd.match(/class="glyph qico"/g) || []).length;
assert(qicons === 18,
  `expected 8 angles + 2 on the slide + 8 in the appendix, found ${qicons}`);
// The art is a dark backdrop with the title printed over it, so the words on
// a break must be light in BOTH themes - the theme's own ink would vanish.
assert(/section\.divider-img \{[^}]*background: #0b1220/.test(theme),
  "an image break carries its own dark ground, so the contrast check is honest");
assert(/section\.divider-img > h2 \{[\s\S]*?color: #f2f6f9/.test(theme),
  "a title over the art must be light in both themes");
assert(/nth-child\(3n \+ 1\)/.test(theme),
  "the subnarrative cards should cycle through accents");
assert(/\.nargrid \.qcard:nth-child\(3n \+ 1\)/.test(theme),
  "the angle cards must cycle the same accents as the query cards");

// ---- figure colours must actually apply ------------------------------------
// Each figure's stylesheet sets `text{fill:var(--ink)}`. A presentation
// attribute has specificity zero — below any author rule — so a
// fill="var(--ret)" on a <text> silently loses to it, and every coloured role
// label, numbered chip and knockout label renders as plain ink. One of them
// was ink on its own accent bar at 1.05:1. Inline styles win; attributes do
// not, so no figure may colour text with an attribute again.
const figures = ["00-task", "00-architecture", "02-one-lane", "02b-rerank",
                 "03a-single-pass", "03b-multi-stage", "04-plan", "05-evidence",
                 "06-retrieval", "07-topic18", "08-disagree", "09-overall",
                 "11a-retrieval-cutoff", "13-cost", "15-coverage", "16-luna"];
for (const f of figures) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${f}.svg`), "utf8");
  assert(!/<text\b[^>]*\sfill="var\(/.test(svg),
    `${f}.svg: colour <text> with style="fill:…", not a fill attribute (it loses to text{fill})`);
}

// ---- every palette clears AA on its own tints ------------------------------
const chan = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const relLum = (hex) => 0.2126 * chan(parseInt(hex.slice(1, 3), 16)) +
                        0.7152 * chan(parseInt(hex.slice(3, 5), 16)) +
                        0.0722 * chan(parseInt(hex.slice(5, 7), 16));
const contrast = (a, b) => {
  const [x, y] = [relLum(a), relLum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};
const sheet = fs.readFileSync(path.join(root, "2026-trec-rag-talk/theme.scss"), "utf8");
for (const name of ["slate-light", "slate-dark", "forest-light", "forest-dark"]) {
  const body = sheet.match(new RegExp(`@mixin ${name} \\{([\\s\\S]*?)\\n\\}`));
  assert(body, `palette ${name} should exist`);
  const tok = Object.fromEntries([...body[1].matchAll(/--([a-z-]+):\s*(#[0-9a-f]{6})/gi)]
    .map((m) => [m[1], m[2].toLowerCase()]));
  for (const k of ["ret", "gen", "ok", "adv"]) {
    // These accents are used as text on their own soft tint (role labels,
    // numbered chips) and on the page ground.
    for (const on of [`${k}-soft`, "bg"]) {
      const got = contrast(tok[k], tok[on]);
      assert(got >= 4.5, `${name}: --${k} on --${on} is ${got.toFixed(2)}:1, under AA`);
    }
  }
  assert(contrast(tok.ink, tok.bg) >= 7 && contrast(tok.muted, tok.bg) >= 4.5,
    `${name}: body text must clear AA on the page ground`);
}

// ---- the hint claims are not a slide ---------------------------------------
// "advisory", then "hint", then the n092 trap: three separate readers bounced
// off the same idea, and it never changed what the answer may cite. The rule
// stays on the evidence slide in one line; the dedicated trace is gone.
assert(!/10-nugget-trace/.test(qmd),
  "the nugget trace slide was cut; nothing should still point at it");
assert(flat.includes("nothing else") && flat.includes("these exact passages"),
  "the evidence slide must still say what the writers are given");

// ---- reranking is its own slide --------------------------------------------
// "it is not clear how document goes to passage and score" - the unit change
// was a caption under a figure about something else. It gets the slide now.
// The slide is the title and the figure, nothing else: its eyebrow, its
// caption and the figure's own foot bar all said the same thing three times.
const rerank = fs.readFileSync(path.join(root, "2026-trec-rag-talk/02b-rerank.svg"), "utf8");
for (const fact of ["3,500", "350", "≤100 KEPT", "0.81", "dropped"]) {
  assert(rerank.includes(fact), `the rerank figure should show: ${fact}`);
}
assert(!/The unit changes here/.test(rerank),
  "the figure's foot bar repeated its own column heads");

// The disagreement slide now reports the whole distribution, not one case.
// 42 scored topics = 20 agreeing + 5 disagreeing + 17 with one measure tied,
// computed from the organizer sxs counts and rubric scores.
const dis = fs.readFileSync(path.join(root, "2026-trec-rag-talk/12-disagree-all.svg"), "utf8");
for (const t of ["rag2026-13", "rag2026-59", "rag2026-97", "rag2026-2", "rag2026-109"]) {
  assert(dis.includes(t), `the disagreement figure should list ${t}`);
}
assert(/>20</.test(dis) && /ature>5</.test(dis) === false && dis.includes(">5<") && dis.includes(">17<"),
  "the disagreement figure must show the 20 / 5 / 17 split of the 42");

// The retrieval figure's numbers come straight from the submitted TSVs:
// 4,246 rows, depth 1-121, 61 distinct depths, identical doc set per run.
// It was one slide and unreadable. The cutoff and the three orderings are two
// separate ideas, so they are two figures.
const cutoff = fs.readFileSync(path.join(root, "2026-trec-rag-talk/11a-retrieval-cutoff.svg"), "utf8");
for (const fact of ["1 – 121", "61", "4,246", "1.4826"]) {
  assert(cutoff.includes(fact), `the cutoff figure should state: ${fact}`);
}
// The three orderings are a sentence now, not a second figure.
assert(flat.includes("same set, different order"),
  "the appendix must still state what the three files have in common");
assert(!/11-retrieval-submission/.test(qmd) && !/11b-retrieval-orderings/.test(qmd),
  "the retrieval figures were reworked; nothing should point at the retired ones");
// Retrieval was the weaker of the two submissions, so its mechanics sit after
// the close, where they answer a question instead of spending the room's time.
const appendixAt = qmd.indexOf("## Appendix {.divider}");
assert(appendixAt > 0, "the retrieval detail belongs in an appendix");
assert(qmd.indexOf("## The Retrieval submission") > appendixAt &&
       qmd.indexOf("## How the Retrieval run scored") > appendixAt,
  "both retrieval slides belong after the appendix divider");
// ---- the three-layer problem ----------------------------------------------
// Title, then a kicker, then a caption: three places saying one thing. Every
// kicker that only restated its own title or its own figure came out, so the
// ones left have to earn it - a topic id, a unit, a scope.
const slides = qmd.split(/\n## /).slice(1);
const threeLayer = slides.filter((b) =>
  /\{\.eyebrow\}|\{\.trace-step\}/.test(b) && /\{\.cap\}|\{\.boundary\}/.test(b));
assert(threeLayer.length <= 9,
  `at most 9 slides may carry title + kicker + caption, found ${threeLayer.length}`);
for (const dead of ["Drop the duplicates, spread the rest", "One model call, then a validator",
                    "Seven steps, three pairs of hands", "Across everything NIST scored",
                    "If any of this is worth a second look"]) {
  assert(!qmd.includes(dead), `this kicker only restated its slide: ${dead}`);
}

assert(qmd.indexOf("## Everything here is public") < appendixAt,
  "the talk should end before the appendix starts");

// ---- cost, counted in the only unit that was actually counted -------------
// 119 planner calls + 691 nuggetizer calls (one per subnarrative) = 810 shared.
// Generation: single-pass 119 + 2 retries = 121; multi-stage 4 per request
// (plan, draft, revise, screen) x 119 = 476, + 691 audits = 1,167.
// The subnarrative and query totals are counted from the published coverage
// report, which carries every topic's plan: 3-8 subnarratives, 691 in all.
const cost = fs.readFileSync(path.join(root, "2026-trec-rag-talk/13-cost.svg"), "utf8");
// Per narrative first, because that is the number a listener can hold:
// retrieval 4-9 calls, generation 1 against 7-12. Totals follow.
for (const fact of ["4 – 9", "7 – 12", "119", "691", "810", "121", "1,167",
                    "9.6", "1,787", "per narrative"]) {
  assert(cost.includes(fact), `the cost figure should state: ${fact}`);
}
assert(476 + 691 === 1167 && 119 + 2 === 121,
  "the cost arithmetic on the figure must still add up");
// Tokens were never metered, so the deck must not imply a price.
assert(/THIS COUNTS CALLS, NOT MONEY/.test(cost) && /never metered/.test(cost),
  "the cost slide must say plainly that it counts calls, not money");
assert(!/\$|USD|dollars? spent|per 1M tokens/.test(cost),
  "no figure may put a price on a run that was never metered");

// ---- the hedging budget ----------------------------------------------------
// A warning box on every slide teaches the room to skip them, so only the
// notes that stop a number on their own slide being misread may stay. The
// standalone limits slide read as slop and came out; the deck now closes on
// the architecture, and the one limit that changes a conclusion rides there.
const boxes = (qmd.match(/::: \{\.boundary\}/g) || []).length;
assert(boxes <= 3, `at most 3 boundary boxes, found ${boxes}`);
assert(!/What I would and would not claim/.test(qmd),
  "the standalone limits slide was cut; its one load-bearing limit moved to the close");
assert(/## Where that leaves the architecture/.test(qmd),
  "the talk should close on the architecture");
// It used to open on the same figure too. One of the two had to go, and the
// close is the one the author asked to keep.
assert(!/^## The architecture$/m.test(qmd),
  "the architecture is shown once, at the close, not twice");
assert((qmd.match(/00-architecture\.svg/g) || []).length === 1,
  "the architecture figure should appear on exactly one slide");

// The prompt slide was cut, and so, later, was the one line that carried it.
// The constraint is real but it is the speaker's to say out loud: the slide is
// the title and the figure. Nothing should reintroduce the prompt dump.
assert(!/promptbox|promptlead/.test(qmd), "nothing should still reference the prompt slide");
assert(!/Do not answer the topic/.test(qmd),
  "the planner's instruction was cut as a caption; do not quietly put it back");

// ---- what a plan contains, from narrative_blueprint.py ---------------------
// "I am lost on this myself" - a schema record does not show what happens.
// The slide is one picture: one obligation in plain words, an arrow, and the
// sentence Sol actually wrote from it in the submitted answer.
const luna = fs.readFileSync(path.join(root, "2026-trec-rag-talk/16-luna.svg"), "utf8");
for (const fact of ["3 TO 8 OBLIGATIONS", "legality and reliability", "140 words",
                    "c003 · c011 · c014", "SOL WRITES FROM IT",
                    "registered with", "Class B status"]) {
  assert(luna.includes(fact), `the Luna figure should show: ${fact}`);
}
// Five named checks, each with its own gloss - "what are the five" was the
// question the chip row never answered.
for (const gate of ["support", "atomicity", "materiality", "redundancy",
                    "replacement safety"]) {
  assert(luna.includes(`>${gate}</text>`), `the screen should name the check: ${gate}`);
}
assert((luna.match(/class="lu-gloss"/g) || []).length === 5,
  "every check carries a plain-words gloss, or it is just a word");
assert(/never writes a sentence of the answer/.test(luna),
  "the figure must still say Luna writes no prose");

// Pandoc wraps a run of inline spans in a <p>, which silently collapses any
// grid or flex row built from them onto a single line. This has regressed
// twice, and it renders without error, so the un-wrapping rule is pinned.
assert(/\.record > p,\s*\n\.gatelist > p,\s*\n\.adds > p,\s*\n\.reflist > p,\s*\n\.roadmap > p \{\s*\n\s*display: contents;/.test(theme),
  "the spans-in-a-<p> wrapper must be dropped, or these grids collapse");

// ---- citations ------------------------------------------------------------
// Every model named on a slide has to be findable. They live in an appendix,
// not under the diagrams, so the slides stay readable.
const refsAt = qmd.indexOf("## Appendix · what this was built on");
assert(refsAt > qmd.indexOf("## Everything here is public"),
  "the reference list belongs in the appendix");
for (const ref of ["mxbai-rerank-base-v2", "Pyserini", "all-MiniLM-L6-v2",
                   "deepseek-v4-flash", "AutoNuggetizer", "arena-rank"]) {
  assert(qmd.slice(refsAt).includes(ref), `the appendix should cite: ${ref}`);
}
for (const url of ["huggingface.co/mixedbread-ai/mxbai-rerank-base-v2",
                   "huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                   "doi.org/10.1145/3404835.3463238",
                   "arxiv.org/abs/2411.09607",
                   "github.com/lmarena/arena-rank"]) {
  assert(qmd.includes(url), `the appendix should link: ${url}`);
}

// ---- the type floor ------------------------------------------------------
// "the copy might be small to see". A figure drawn in a 1600-unit viewBox is
// scaled to about 1150px of slide width, so an 18px label lands near 13px on
// a 760-high slide - roughly 1.7% of slide height, which the back row cannot
// read. Every figure's smallest type must clear ~17 slide px once scaled.
const SLIDE_CONTENT_PX = 1150;
const FLOOR = 17;
for (const file of fs.readdirSync(path.join(root, "2026-trec-rag-talk"))) {
  if (!file.endsWith(".svg")) continue;
  const svg = fs.readFileSync(path.join(root, "2026-trec-rag-talk", file), "utf8");
  const vb = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  if (!vb) continue;
  const k = SLIDE_CONTENT_PX / parseFloat(vb[1]);
  const sizes = [...svg.matchAll(/font-size[:=]"?(\d+(?:\.\d+)?)/g)].map((m) => parseFloat(m[1]));
  if (!sizes.length) continue;
  const smallest = Math.min(...sizes);
  assert(smallest * k >= FLOOR - 0.5,
    `${file}: smallest type is ${(smallest * k).toFixed(1)} slide px ` +
    `(${smallest}px in a ${vb[1]}-unit viewBox) — the floor is ${FLOOR}`);
}

// ---- palettes and inlined figures -----------------------------------------
// A linked <img src="x.svg"> is a separate document, so the page's CSS custom
// properties cannot reach inside it and no palette switch could repaint the
// diagrams. inline-svg.lua inlines them instead.
assert(!/<img[^>]+src="[^"]*\.svg"/i.test(html),
  "figures must be inlined, not linked, or the palette cannot reach them");
const inlined = (html.match(/<svg\b/gi) || []).length;
assert(inlined >= 6, `expected the 6 figures inlined, found ${inlined} <svg> elements`);

// Each figure keeps its own accessible name once inlined.
for (const id of ["tk", "ar", "ln", "rr", "sp", "ms", "pl", "ev", "rt", "t18",
                  "dg", "ov", "rc", "co", "cv", "lu", "da"]) {
  assert(html.includes(`id="${id}-title"`) && html.includes(`id="${id}-desc"`),
    `inlined figure ${id} should keep its <title> and <desc>`);
}

// Six style blocks now share one document, so every figure class is namespaced.
for (const [fig, tok] of [["02-one-lane", "ln"], ["03a-single-pass", "sp"],
                          ["03b-multi-stage", "ms"], ["07-topic18", "t18"],
                          ["08-disagree", "dg"], ["09-overall", "ov"],
                          ["02b-rerank", "rr"],
                          ["11a-retrieval-cutoff", "rc"],
                          ["13-cost", "co"],
                          ["15-coverage", "cv"],
                          ["16-luna", "lu"],
                          ["04-plan", "pl"],
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
for (const fig of ["02-one-lane", "03a-single-pass", "03b-multi-stage", "05-evidence"]) {
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
                    "all-MiniLM-L6-v2", "gpt-5.6-sol", "gpt-5.6-luna",
                    "Retrieval run", "Written answers"]) {
  assert(arch.includes(fact), `the architecture figure should name: ${fact}`);
}
// Which steps are hosted and which are local is a detail about my hardware, not
// part of the architecture. It was cut; the figure stays about the five pieces.
for (const cut of ["WHERE EACH STEP RUNS", "local GPU", "no cost per request",
                   "hosted index"]) {
  assert(!arch.includes(cut),
    `the architecture figure should not carry the where-it-runs detail: ${cut}`);
}
// (A single "hosted call" elsewhere is the nuggetizer, which is a step, not a
// statement about my hardware.)
assert(!/own GPU|everything else is a hosted call/.test(qmd),
  "the architecture slide should not restate where each step runs");
// Figure classes stay namespaced, including the two new ones.
for (const [fig, tok] of [["00-task", "tk"], ["00-architecture", "ar"]]) {
  const svg = fs.readFileSync(path.join(root, `2026-trec-rag-talk/${fig}.svg`), "utf8");
  const bare = [...new Set([].concat(...[...svg.matchAll(/class="([^"]*)"/g)]
    .map((m) => m[1].split(/\s+/).filter(Boolean))))].filter((c) => !c.startsWith(`${tok}-`));
  assert(bare.length === 0, `${fig}.svg has un-namespaced classes: ${bare.join(", ")}`);
}

// The planning queries are shown, not hidden behind a click.
// Two lanes on the slide, all eight in the appendix: the full grid was a wall
// mid-talk, but it is the thing people ask to see afterwards.
const openDetails = (qmd.match(/<details open>/g) || []).length;
assert(openDetails === 10, `expected 2 lanes on the slide and 8 in the appendix, found ${openDetails}`);
// Eight lanes on one appendix slide only fit at microscopic type, so they run
// across two slides at the size the rest of the deck uses.
const appendixQueriesAt = qmd.indexOf("## Appendix · query lanes 1–4");
assert(appendixQueriesAt > qmd.indexOf("## Appendix {.divider}"),
  "the full query grid belongs in the appendix");
assert(qmd.includes("## Appendix · query lanes 5–8"),
  "the second half of the lanes needs its own slide");
assert((qmd.slice(appendixQueriesAt).match(/<details open>/g) || []).length === 8,
  "the appendix carries all eight lanes");
assert(!/\.qall/.test(theme), "the shrink-to-fit appendix hack should be gone");

// The dedup step is shown concretely, and flagged as illustrative wording
// because real corpus passages stay private.
assert(/cosine \*\*0\.94\*\*/.test(qmd), "the evidence slide should show a worked duplicate pair");
assert(flat.includes("standing in for the real passages"),
  "the illustrative sentences must be marked as not being corpus text");

// The nuggetizer's cap still rides on the lane figure, which is now the only
// place the hint claims appear at all.
// The stage list was two lines of prose per stage. One short line each now.
assert(/≤20/.test(lane) && /hints for the writer, never cited/.test(lane),
  "the lane figure should still bound and label the nugget claims");
for (const stage of ["Search", "Rerank", "Select", "Nuggets"]) {
  assert(lane.includes(`>${stage}</text>`), `the lane figure should keep the stage: ${stage}`);
}

console.log("reports/2026-trec-rag-talk.test.js OK");

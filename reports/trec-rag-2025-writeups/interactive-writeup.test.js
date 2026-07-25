const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "interactive-writeup.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "interactive-writeup.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");

const requiredSections = [
  'id="start-here"',
  'id="plain-english"',
  'id="track-setup"',
  'id="trec-primer"',
  'id="worked-example"',
  'id="pipeline"',
  'id="approach-map"',
  'id="metric-decoder"',
  'id="leaderboard"',
  'id="cross-team-synthesis"',
  'id="idea-bank"',
  'id="try-steal"',
  'id="implementation-kit"',
  'id="team-explorer"',
  'id="coverage-audit"',
  'id="failure-diagnosis"',
  'id="takeaways"',
  'id="sources"',
];

for (const section of requiredSections) {
  assert(html.includes(section), `Missing required section ${section}`);
}

assert(html.includes('rel="icon"'), "Standalone report should include an inline favicon");
assert(html.includes("data:image/svg+xml"), "Standalone report favicon should not request /favicon.ico");
assert(html.includes("Reports Home"), "2025 report should link back to the common reports index");
assert(html.includes('href="../index.html"'), "2025 report home link should point to reports/index.html");

const teamEntries = [...html.matchAll(/class="team-card"/g)];
assert(teamEntries.length === 15, `Expected 15 team cards, found ${teamEntries.length}`);

const deepDossiers = [...html.matchAll(/data-depth="deep-dossier"/g)];
assert(deepDossiers.length === 15, `Expected 15 deep team dossiers, found ${deepDossiers.length}`);

assert(html.includes(".team-card.open {"), "Open team cards should have explicit wide-card CSS");
assert(html.includes("grid-column: 1 / -1;"), "Open team cards should span the full team grid");
assert(html.includes("function renderArchitecturePanels"), "Team details should render synthetic architecture panels");
assert(html.includes("function enhanceTechnicalText"), "Team details should enhance models, methods, runs, and metrics inline");
assert(html.includes("function openFigureLightbox"), "Official figures should open in an in-page lightbox");
assert(html.includes("termDecoder"), "Team details should include a jargon decoder dictionary");
assert(html.includes("architecture-panel"), "Missing architecture panel styles/markup");
const visualStages = ["Narrative", "Plan", "Search", "Evidence", "Answer", "Check"];
assert(html.includes('data-visual-stage="${escapeHTML(entry.stage)}"'), "Missing visual stage data attributes");
assert(html.includes(`const teamVisualStageNames = ["${visualStages.join('", "')}"];`), "Six-stage vocabulary should be explicit and ordered");
assert(html.includes("teamVisuals"), "Missing standardized visual data map");
assert(html.includes("team-architecture-visual"), "Missing team visual renderer output");
assert(html.includes("team-architecture-text"), "Missing text equivalent for team visual");
assert(html.includes('role="img"'), "Team SVGs should have an accessible image role");
assert(html.includes("All 15 teams use the same six stages"), "Missing visual legend");

const requiredVisualTeams = [
  "CFDA Lab", "NIT Agartala", "University of Glasgow Terrier", "WING-II",
  "GRILL Lab", "IIUoT", "GenAIus", "Tokyo University of Science", "HLTCOE",
  "MIT Lincoln Laboratory", "IDACCS", "NC State LAS", "UTokyo-HitU", "DUTH",
  "WaterlooClarke",
];
for (const team of requiredVisualTeams) {
  assert(html.includes(`"${team}":`), `Missing visual data for ${team}`);
}
assert(html.includes("official-figure"), "Missing official paper architecture figure styling");
assert(html.includes("official-figure-button"), "Official figures should be clickable without right-clicking");
assert(html.includes('id="figureLightbox"'), "Missing figure lightbox markup");
assert(html.includes("figure-lightbox"), "Missing figure lightbox styling");
assert(html.includes(".figure-lightbox[hidden]"), "Hidden lightbox should not render over the report on page load");
assert(html.includes("table-layout: fixed;"), "Compact score tables should use fixed layout so long run names do not push cards wider");
assert(html.includes("overflow-wrap: anywhere;"), "Compact cards and tables should wrap long run names, model names, and metric labels");
assert(html.includes("@media (max-width: 1240px) and (min-width: 961px)"), "Synthesis cards should switch to wider columns on laptop-width screens");
assert(html.includes("Reusable Playbook"), "Missing combined reusable playbook section");
assert(html.includes("What We Can Use First"), "Combined playbook should include the practical takeaway summary");
assert(html.includes("playbook-takeaways"), "What We Can Use summary should live inside the combined playbook");
assert(html.includes("playbook-actions"), "Try/steal checklist should live inside the combined playbook");
assert(!html.includes('<section id="try-steal"'), "Try/steal should no longer be a separate section");
assert(!html.includes('<section id="takeaways"'), "What We Can Use should no longer be a separate bottom section");
assert(html.includes("idea-card"), "Missing reusable playbook idea cards");
assert(html.includes("beginner-guide"), "Reusable playbook should include a beginner reading guide");
assert(html.includes("idea-plain"), "Reusable idea cards should include plain-English explanations");
assert(html.includes("idea-lens"), "Reusable idea cards should include beginner context lenses");
assert(html.includes("Everything to Try / Steal"), "Missing consolidated try/steal checklist inside playbook");
assert(html.includes("steal-card"), "Try/steal section should use scannable checklist cards");
assert(html.includes("steal-list"), "Try/steal section should include concrete checklist items");
assert(html.includes("implementation roadmap"), "Try/steal section should include a staged implementation roadmap");
assert(html.includes("Start Here: How to Read This Report"), "Missing beginner reading path");
assert(html.includes("One Tiny TREC RAG Example"), "Missing worked RAG example");
assert(html.includes("Metric Decoder"), "Missing metric decoder");
assert(html.includes("Leaderboard"), "Missing leaderboard section");
assert(html.includes("Manual Answer Coverage"), "Missing manual answer coverage leaderboard");
assert(html.includes("Retrieval Leaderboard"), "Missing retrieval leaderboard");
assert(html.includes("Automated Coverage Cross-Check"), "Missing automated coverage leaderboard");
assert(html.includes("Citation Support Leaderboard"), "Missing citation support leaderboard");
assert(html.includes("official overview table 7"), "Leaderboard should cite the manual answer source table");
assert(html.includes("official overview table 5"), "Leaderboard should cite the retrieval source table");
assert(html.includes("official overview table 9"), "Leaderboard should cite the automated coverage source table");
assert(html.includes("official overview table 11"), "Leaderboard should cite the support source table");
assert(html.includes("no local writeup"), "Leaderboard should flag official runs without local team dossiers");
assert(html.includes("Official vs Team-Side Scores"), "Missing metric provenance explanation");
assert(html.includes("Implementation Kit"), "Missing implementation kit");
assert(html.includes("Failure Diagnosis"), "Missing failure diagnosis section");
assert(html.includes("How to Interpret TREC"), "Missing TREC interpretation primer");
assert(html.includes("coverage-table"), "Missing source coverage audit table");
assert(html.includes("reuse-list"), "Team reusable lessons should be rendered as multi-point lists");
assert(html.includes("term-decoder"), "Missing term decoder styling");
assert(html.includes("tech-model"), "Missing model highlight styling");
assert(html.includes("tech-metric"), "Missing metric highlight styling");
assert(html.includes("tech-method"), "Missing method highlight styling");
assert(html.includes("tech-run"), "Missing run highlight styling");

const requiredTeams = [
  "CFDA Lab",
  "NIT Agartala",
  "University of Glasgow Terrier",
  "WING-II",
  "GRILL Lab",
  "IIUoT",
  "GenAIus",
  "Tokyo University of Science",
  "HLTCOE",
  "MIT Lincoln Laboratory",
  "IDACCS",
  "NC State LAS",
  "UTokyo-HitU",
  "DUTH",
  "WaterlooClarke",
];

for (const team of requiredTeams) {
  assert(html.includes(team), `Missing team entry for ${team}`);
}

const requiredTags = [
  "hybrid retrieval",
  "query decomposition",
  "reranking",
  "nuggets",
  "evidence selection",
  "agentic retrieval",
  "citation checking",
  "relevance judging",
];

for (const tag of requiredTags) {
  assert(html.includes(`data-tag="${tag}"`) || html.includes(tag), `Missing approach tag ${tag}`);
}

const requiredHooks = [
  "function applyFilters",
  "function selectPipelineStep",
  "function showGlossary",
  "function toggleTeam",
  "addEventListener",
];

for (const hook of requiredHooks) {
  assert(html.includes(hook), `Missing interaction hook ${hook}`);
}

const dossierSections = [
  "Architecture",
  "What they tried",
  "Models and tools",
  "Results and observations",
  "Caveats",
  "What we can reuse",
];

for (const section of dossierSections) {
  const count = (html.match(new RegExp(`<h4>${section}</h4>`, "g")) || []).length;
  assert(count >= 15, `Expected at least 15 '${section}' headings, found ${count}`);
}

const reuseLists = [...html.matchAll(/class="[^"]*\breuse-list\b[^"]*"/g)];
assert(reuseLists.length >= 15, `Expected a multi-point reuse list for each team, found ${reuseLists.length}`);

const reuseItems = [...html.matchAll(/<li><strong>[^<]+:<\/strong>/g)];
assert(reuseItems.length >= 45, `Expected at least 45 concrete reusable lessons, found ${reuseItems.length}`);

const ideaCards = [...html.matchAll(/class="idea-card"/g)];
assert(ideaCards.length >= 12, `Expected at least 12 reusable idea bank cards, found ${ideaCards.length}`);

const ideaPlainBlocks = [...html.matchAll(/class="idea-plain"/g)];
assert(ideaPlainBlocks.length >= 12, `Expected plain-English context for each idea card, found ${ideaPlainBlocks.length}`);

const ideaLensBlocks = [...html.matchAll(/class="idea-lens"/g)];
assert(ideaLensBlocks.length >= 12, `Expected beginner context lenses for each idea card, found ${ideaLensBlocks.length}`);

const coverageRows = [...html.matchAll(/data-audit-team=/g)];
assert(coverageRows.length === 15, `Expected source coverage audit rows for 15 teams, found ${coverageRows.length}`);

const stealCards = [...html.matchAll(/class="steal-card"/g)];
assert(stealCards.length >= 8, `Expected at least 8 try/steal checklist cards, found ${stealCards.length}`);

const stealItems = [...html.matchAll(/data-steal-item=/g)];
assert(stealItems.length >= 35, `Expected at least 35 concrete try/steal checklist items, found ${stealItems.length}`);

const recipeCards = [...html.matchAll(/class="recipe-detail"/g)];
assert(recipeCards.length >= 8, `Expected at least 8 recipe detail cards, found ${recipeCards.length}`);

const recipeFields = [...html.matchAll(/data-recipe-field=/g)];
assert(recipeFields.length >= 48, `Expected recipe cards with input/output/knobs/logs/pass-fail fields, found ${recipeFields.length}`);

const evidenceChips = [...html.matchAll(/class="evidence-chip/g)];
assert(evidenceChips.length >= 12, `Expected evidence-strength chips on reusable ideas and claims, found ${evidenceChips.length}`);

const provenanceChips = [...html.matchAll(/class="provenance-chip/g)];
assert(provenanceChips.length >= 8, `Expected provenance chips for key scores/rankings, found ${provenanceChips.length}`);

const metricRows = [...html.matchAll(/data-metric-row=/g)];
assert(metricRows.length >= 7, `Expected at least 7 metric decoder rows, found ${metricRows.length}`);

const leaderboardTables = [...html.matchAll(/class="leaderboard-table"/g)];
assert(leaderboardTables.length >= 4, `Expected at least 4 leaderboard tables, found ${leaderboardTables.length}`);

const leaderboardCards = [...html.matchAll(/class="leaderboard-card"/g)];
assert(leaderboardCards.length >= 8, `Expected leaderboard summary and board cards, found ${leaderboardCards.length}`);

const requiredLeaderboardSignals = [
  "LAS-agentic-RAG-agent",
  "4method_merge",
  "r_2method_ag_gpt41",
  "bm25-rz7b-2025a",
  "UTokyo-HitU",
  "NC State LAS",
  "IIUoT",
  "Relevance judging footnote",
];

for (const signal of requiredLeaderboardSignals) {
  assert(html.includes(signal), `Missing leaderboard signal: ${signal}`);
}

const exampleSteps = [...html.matchAll(/data-example-step=/g)];
assert(exampleSteps.length >= 6, `Expected at least 6 worked example steps, found ${exampleSteps.length}`);

const schemaBlocks = [...html.matchAll(/data-schema-block=/g)];
assert(schemaBlocks.length >= 6, `Expected at least 6 implementation schema blocks, found ${schemaBlocks.length}`);

const promptBlocks = [...html.matchAll(/data-prompt-block=/g)];
assert(promptBlocks.length >= 6, `Expected at least 6 starter prompt blocks, found ${promptBlocks.length}`);

const experimentRows = [...html.matchAll(/data-experiment-row=/g)];
assert(experimentRows.length >= 8, `Expected at least 8 experiment matrix rows, found ${experimentRows.length}`);

const diagnosisRows = [...html.matchAll(/data-diagnosis-row=/g)];
assert(diagnosisRows.length >= 8, `Expected at least 8 failure diagnosis rows, found ${diagnosisRows.length}`);

const reusePlaybooks = [...html.matchAll(/class="reuse-list reuse-playbook"/g)];
assert(reusePlaybooks.length >= 15, `Expected a rich reuse playbook for each team, found ${reusePlaybooks.length}`);

const reusableMoveItems = [...html.matchAll(/<strong>Reusable move:<\/strong>/g)];
assert(reusableMoveItems.length >= 15, `Expected each team playbook to name a reusable move, found ${reusableMoveItems.length}`);

const plainEnglishItems = [...html.matchAll(/<strong>Plain English:<\/strong>/g)];
assert(plainEnglishItems.length >= 27, `Expected plain-English explanations across idea cards and team playbooks, found ${plainEnglishItems.length}`);

const tryFirstItems = [...html.matchAll(/<strong>Try first:<\/strong>/g)];
assert(tryFirstItems.length >= 27, `Expected try-first guidance across idea cards and team playbooks, found ${tryFirstItems.length}`);

const measureItems = [...html.matchAll(/<strong>Measure:<\/strong>/g)];
assert(measureItems.length >= 15, `Expected each team playbook to include measurement guidance, found ${measureItems.length}`);

const relatedTeamsItems = [...html.matchAll(/<strong>Related teams:<\/strong>/g)];
assert(relatedTeamsItems.length >= 15, `Expected each team playbook to connect related teams, found ${relatedTeamsItems.length}`);

const requiredIdeaBankSignals = [
  "Reusable Playbook",
  "What We Can Use First",
  "How to use this section",
  "Plain English",
  "Why it matters",
  "Try first",
  "Watch for",
  "baseline ladder",
  "coverage ledger",
  "citation-first",
  "portfolio generation",
  "support is not coverage",
  "human-in-the-loop judging",
];

for (const signal of requiredIdeaBankSignals) {
  assert(html.includes(signal), `Missing reusable idea bank signal: ${signal}`);
}

const requiredStealSignals = [
  "Start with a measured baseline",
  "Keep a coverage ledger",
  "Fuse complementary retrievers",
  "Rerank after broad retrieval",
  "Select evidence before generation",
  "Generate from nuggets or evidence cards",
  "Bind citations while writing",
  "Evaluate support and coverage separately",
];

for (const signal of requiredStealSignals) {
  assert(html.includes(signal), `Missing try/steal checklist signal: ${signal}`);
}

const requiredImplementationSignals = [
  "NarrativeNeed",
  "RetrievedCandidate",
  "EvidenceCard",
  "Nugget",
  "ClaimCitation",
  "EvaluationRecord",
  "decomposition prompt",
  "HyDE prompt",
  "citation-first generation prompt",
  "support judging prompt",
  "query drift",
  "context stuffing",
  "compression loss",
  "agent over-searching",
  "Assembled on July 3, 2026",
  "run",
  "qrels",
  "judgments",
  "best idea to steal",
  "strong official result",
  "team-side ablation",
  "design hypothesis",
  "official overview",
  "manual subset",
  "clip2025",
  "IRIT-ISIR-EV",
  "hltcoe-rerank",
  "digsci",
  "RMIT-IR",
  "hltcoe-multiagt",
];

for (const signal of requiredImplementationSignals) {
  assert(html.includes(signal), `Missing implementation/advisor signal: ${signal}`);
}

const internalProcessTerms = [
  "advisor review",
  "Galileo",
  "Parfit",
  "TODO",
  "placeholder",
];

for (const term of internalProcessTerms) {
  assert(!html.includes(term), `Report should not expose internal/process term: ${term}`);
}

const deepSignals = [
  "nDCG@30",
  "strict vital score",
  "sub-narrative coverage",
  "weighted precision",
  "RRF",
  "SPLADE",
  "HyDE Vector Mix",
  "RankZephyr",
  "Crucible",
  "Keystone-Docs",
  "submodular",
  "planner-executor",
];

for (const signal of deepSignals) {
  assert(html.includes(signal), `Missing deep technical signal: ${signal}`);
}

const pdfLinks = [...html.matchAll(/href="(pdfs\/[^"]+\.pdf)"/g)].map((match) => match[1]);
assert(pdfLinks.length >= 15, `Expected at least 15 local PDF links, found ${pdfLinks.length}`);

for (const pdfLink of pdfLinks) {
  const pdfPath = path.join(root, pdfLink);
  assert(fs.existsSync(pdfPath), `Linked PDF does not exist: ${pdfLink}`);
}

const architectureEntries = [...html.matchAll(/keyStages:/g)];
assert(architectureEntries.length >= 15, `Expected architecture definitions for 15 teams, found ${architectureEntries.length}`);
assert(!html.includes('class="paper-visual"'), "Expanded cards should not render full-page PDF screenshots as the primary visual");

const officialFigureEntries = [...html.matchAll(/paperFigure:/g)];
assert(officialFigureEntries.length >= 8, `Expected at least 8 official architecture/workflow figures, found ${officialFigureEntries.length}`);
const embeddedOfficialFigures = [...html.matchAll(/paperFigure:\s*{[\s\S]*?src:\s*"data:image\/jpeg;base64,/g)];
assert(embeddedOfficialFigures.length >= 8, `Expected official architecture figures to be embedded as data URIs, found ${embeddedOfficialFigures.length}`);
assert(!html.includes('src: "figures/'), "Official architecture figures should not depend on relative image files");

const requiredDecoderTerms = [
  "occams",
  "UMBRELA",
  "SETR",
  "Parasol Score",
  "Crucible",
  "HyDE Vector Mix",
  "RankZephyr",
  "AutoNuggetizer",
  "blame/miniblame",
];

for (const term of requiredDecoderTerms) {
  assert(html.includes(`"${term}"`) || html.includes(term), `Missing jargon decoder term: ${term}`);
}

console.log("interactive-writeup smoke test passed");

"use strict";

const state = {
  status: null,
  evidence: null,
  uploadedEvidence: null,
  benchmark: null,
  answer: null,
  activeFacet: "all",
  activeAnswerTab: "answer",
  activeJobId: null,
  pollingTimer: null,
};

const byId = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" && payload.error ? payload.error : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

function showToast(message, isError = false) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.toggle("is-error", isError);
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 4200);
}

function textOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function fixedMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function percentMetric(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "-";
}

function defaultRunId(topicId = "topic") {
  const now = new Date();
  const stamp = [
    now.getUTCFullYear(),
    String(now.getUTCMonth() + 1).padStart(2, "0"),
    String(now.getUTCDate()).padStart(2, "0"),
    "-",
    String(now.getUTCHours()).padStart(2, "0"),
    String(now.getUTCMinutes()).padStart(2, "0"),
    String(now.getUTCSeconds()).padStart(2, "0"),
  ].join("");
  return `rag26-${String(topicId).replace(/[^A-Za-z0-9._-]/g, "-")}-${stamp}`;
}

function setView(name) {
  document.querySelectorAll("[data-view]").forEach((view) => {
    const active = view.dataset.view === name;
    view.hidden = !active;
    view.classList.toggle("is-active", active);
  });
  document.querySelectorAll("[data-view-link]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.viewLink === name);
  });
  const labels = {
    generate: ["Competition output", "Generate answer"],
    benchmark: ["Controlled evidence", "Benchmark"],
    runs: ["Local execution", "Run history"],
  };
  byId("view-eyebrow").textContent = labels[name][0];
  byId("view-title").textContent = labels[name][1];
  window.location.hash = name;
  if (name === "runs") refreshRuns();
}

function renderStatus(status) {
  state.status = status;
  byId("generator-dot").className = `status-dot ${status.generator.ready ? "is-ready" : "is-down"}`;
  byId("auditor-dot").className = `status-dot ${status.auditor.ready ? "is-ready" : "is-down"}`;
  const badge = byId("readiness-badge");
  badge.className = `readiness-badge ${status.ready ? "is-ready" : "is-blocked"}`;
  badge.textContent = status.ready ? "Services ready" : "Service action required";
  byId("generate-button").disabled = !status.ready;
}

async function refreshStatus() {
  const badge = byId("readiness-badge");
  badge.className = "readiness-badge is-checking";
  badge.textContent = "Checking services";
  try {
    renderStatus(await api("/api/status"));
  } catch (error) {
    badge.className = "readiness-badge is-blocked";
    badge.textContent = "Status unavailable";
    showToast(error.message, true);
  }
}

function renderEvidence(summary, label = "Default frozen ledger") {
  state.evidence = summary;
  byId("topic-label").textContent = `Topic ${summary.topic_id}`;
  byId("topic-narrative").textContent = summary.narrative;
  byId("facet-count").textContent = textOrDash(summary.facet_count);
  byId("claim-count").textContent = textOrDash(summary.claim_count);
  byId("source-word-count").textContent = textOrDash(summary.source_word_count);
  const band = summary.candidate_word_band || {};
  byId("word-band").textContent = `${textOrDash(band.minimum)}-${textOrDash(band.maximum)}`;
  byId("evidence-file-name").textContent = label;
  if (!byId("run-id").dataset.edited) {
    byId("run-id").value = defaultRunId(summary.topic_id);
  }
}

function summarizeUploadedEvidence(value) {
  const facets = Array.isArray(value.facets) ? value.facets : [];
  const band = value.word_band || {};
  if (!value.topic_id || !value.narrative || facets.length === 0) {
    throw new Error("The selected file is not a frozen evidence ledger");
  }
  return {
    topic_id: value.topic_id,
    narrative: value.narrative,
    facet_count: facets.length,
    claim_count: value.supported_source_claim_count,
    source_word_count: value.source_claim_word_count,
    candidate_word_band: { minimum: band.minimum, maximum: band.maximum },
  };
}

function setAnswerTab(name) {
  state.activeAnswerTab = name;
  document.querySelectorAll("[data-answer-tab]").forEach((button) => {
    const active = button.dataset.answerTab === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  byId("answer-panel").hidden = name !== "answer";
  byId("validation-panel").hidden = name !== "validation";
  byId("json-panel").hidden = name !== "json";
  byId("facet-filter").disabled = name !== "answer";
}

function sentenceCount(answer) {
  return answer.sections.reduce((total, section) => total + section.sentences.length, 0);
}

function renderSentenceList() {
  const root = byId("sentence-list");
  root.replaceChildren();
  if (!state.answer) return;
  let sentenceNumber = 0;
  state.answer.sections.forEach((section, sectionIndex) => {
    if (state.activeFacet !== "all" && state.activeFacet !== String(sectionIndex)) return;
    const sectionElement = document.createElement("section");
    sectionElement.className = "facet-section";
    const title = document.createElement("h3");
    title.className = "facet-title";
    const titleText = document.createElement("strong");
    titleText.textContent = section.label;
    const count = document.createElement("span");
    count.textContent = `${section.sentences.length} sentences`;
    title.append(titleText, count);
    sectionElement.append(title);
    section.sentences.forEach((sentence) => {
      sentenceNumber += 1;
      const row = document.createElement("article");
      row.className = "sentence-row";
      const number = document.createElement("span");
      number.className = "sentence-number";
      number.textContent = String(sentenceNumber).padStart(2, "0");
      const content = document.createElement("div");
      const text = document.createElement("p");
      text.className = "sentence-text";
      text.textContent = sentence.text;
      const citations = document.createElement("div");
      citations.className = "citation-list";
      sentence.citations.forEach((citation) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "citation-button";
        button.textContent = citation;
        button.title = `Copy ${citation}`;
        button.addEventListener("click", () => copyText(citation, "Citation copied"));
        citations.append(button);
      });
      content.append(text, citations);
      row.append(number, content);
      sectionElement.append(row);
    });
    root.append(sectionElement);
  });
}

function validationItem(title, detail) {
  const item = document.createElement("div");
  item.className = "validation-item";
  const icon = document.createElement("span");
  icon.className = "validation-icon";
  icon.textContent = "\u2713";
  const content = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = title;
  const span = document.createElement("span");
  span.textContent = detail;
  content.append(strong, span);
  item.append(icon, content);
  return item;
}

function renderValidation(summary) {
  const panel = byId("validation-panel");
  const grid = document.createElement("div");
  grid.className = "validation-grid";
  grid.append(
    validationItem("Organizer schema", summary.official_format_valid ? "Valid JSONL entry" : "Validation unavailable"),
    validationItem("Word limit", `${textOrDash(summary.submitted_word_count)} / 1,024 words`),
    validationItem("Sentence citations", `${percentMetric(summary.citation_coverage)} coverage`),
    validationItem(
      "Support gate",
      `${textOrDash(summary.repaired_and_retained_sentence_count ?? 0)} repaired, ${textOrDash(summary.excluded_sentence_count)} excluded`
    ),
    validationItem("Nugget isolation", "No organizer nuggets read"),
    validationItem("Generator requests", "One primary completion, up to one repair")
  );
  panel.replaceChildren(grid);
}

function renderAnswer(answer) {
  state.answer = answer;
  const summary = answer.summary || {};
  byId("answer-words").textContent = textOrDash(summary.submitted_word_count);
  byId("answer-sentences").textContent = textOrDash(summary.submitted_sentence_count ?? sentenceCount(answer));
  byId("answer-references").textContent = textOrDash(summary.reference_count ?? answer.references.length);
  const status = byId("answer-status");
  status.textContent = summary.official_format_valid ? "Valid" : textOrDash(summary.status);
  status.classList.toggle("valid-text", Boolean(summary.official_format_valid));
  byId("answer-source").textContent = answer.source === "generated" ? `Generated run ${summary.run_id}` : "Frozen benchmark preview";

  const select = byId("facet-filter");
  select.replaceChildren(new Option("All facets", "all"));
  answer.sections.forEach((section, index) => select.add(new Option(section.label, String(index))));
  state.activeFacet = "all";
  select.value = "all";
  renderSentenceList();
  renderValidation(summary);
  byId("json-panel").textContent = `${JSON.stringify(answer.official, null, 2)}\n`;
  byId("download-button").disabled = false;
  byId("copy-button").disabled = false;
}

async function copyText(value, message) {
  try {
    await navigator.clipboard.writeText(value);
    showToast(message);
  } catch (_error) {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.className = "clipboard-fallback";
    document.body.append(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
    showToast(message);
  }
}

function downloadAnswer() {
  if (!state.answer) return;
  const jsonl = `${JSON.stringify(state.answer.official)}\n`;
  const filename = `${state.answer.summary.run_id || "trec-rag-answer"}.jsonl`;
  const link = document.createElement("a");
  if (state.answer.download_url) {
    link.href = state.answer.download_url;
  } else {
    link.href = URL.createObjectURL(new Blob([jsonl], { type: "application/jsonl" }));
  }
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  if (!state.answer.download_url) window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function metricCell(label, value) {
  const cell = document.createElement("div");
  cell.className = "metric-cell";
  const progress = document.createElement("progress");
  progress.max = 1;
  progress.value = typeof value === "number" ? value : 0;
  progress.setAttribute("aria-label", `${label}: ${fixedMetric(value)}`);
  const text = document.createElement("span");
  text.textContent = fixedMetric(value);
  cell.append(progress, text);
  return cell;
}

function renderBenchmark(benchmark) {
  state.benchmark = benchmark;
  const chart = byId("benchmark-chart");
  chart.replaceChildren();
  benchmark.models.forEach((model) => {
    const row = document.createElement("div");
    row.className = `benchmark-row ${model.model_key === benchmark.winner ? "is-winner" : ""}`;
    const name = document.createElement("div");
    name.className = "benchmark-row-name";
    const strong = document.createElement("strong");
    strong.textContent = model.display_name;
    const note = document.createElement("span");
    note.textContent = model.model_key === benchmark.winner ? "Selected generator" : "Controlled candidate";
    name.append(strong, note);
    row.append(
      name,
      metricCell("Strict coverage", model.strict_coverage),
      metricCell("Partial coverage", model.partial_credit_coverage),
      metricCell("Vital strict coverage", model.vital_strict_coverage)
    );
    chart.append(row);
  });

  const table = byId("benchmark-table-body");
  table.replaceChildren();
  benchmark.models.forEach((model) => {
    const row = document.createElement("tr");
    if (model.model_key === benchmark.winner) row.className = "winner-row";
    const values = [
      model.display_name,
      fixedMetric(model.strict_coverage),
      fixedMetric(model.partial_credit_coverage),
      fixedMetric(model.vital_strict_coverage),
      textOrDash(model.candidate_words),
      textOrDash(model.submitted_words),
      textOrDash(model.excluded),
      typeof model.cost === "number" ? `$${model.cost.toFixed(4)}` : "-",
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    });
    table.append(row);
  });
}

function jobProgressPercent(progress) {
  if (!progress) return 0;
  if (progress.stage === "complete") return 100;
  if (progress.stage === "failed") return 100;
  if (progress.stage === "preflight") return 8;
  if (progress.stage === "generation") return progress.completed ? 28 : 14;
  if (progress.stage === "support_audit") {
    const portion = progress.total ? progress.completed / progress.total : 0;
    return Math.round(28 + portion * 68);
  }
  if (progress.stage === "repair") return 97;
  return 3;
}

function renderProgress(job) {
  const panel = byId("progress-panel");
  panel.hidden = false;
  const progress = job.progress || {};
  const percent = jobProgressPercent(progress);
  byId("progress-stage").textContent = String(progress.stage || job.status || "queued").replaceAll("_", " ");
  byId("progress-value").textContent = `${percent}%`;
  byId("progress-bar").value = percent;
  byId("progress-bar").textContent = `${percent}%`;
  byId("progress-message").textContent = progress.message || "Processing";
}

async function pollRun(jobId) {
  window.clearTimeout(state.pollingTimer);
  try {
    const job = await api(`/api/runs/${jobId}`);
    renderProgress(job);
    if (job.status === "complete") {
      state.activeJobId = null;
      byId("generate-button").disabled = !state.status?.ready;
      renderAnswer(job.answer);
      setAnswerTab("answer");
      showToast("Organizer answer generated and frozen");
      return;
    }
    if (job.status === "failed") {
      state.activeJobId = null;
      byId("generate-button").disabled = !state.status?.ready;
      showToast(job.error || "Generation failed", true);
      return;
    }
    state.pollingTimer = window.setTimeout(() => pollRun(jobId), 1400);
  } catch (error) {
    state.pollingTimer = window.setTimeout(() => pollRun(jobId), 2400);
    showToast(error.message, true);
  }
}

async function submitGeneration(event) {
  event.preventDefault();
  if (state.activeJobId) return;
  const form = new FormData(event.currentTarget);
  const payload = {
    run_id: form.get("run_id"),
    team_id: form.get("team_id"),
    run_desc: form.get("run_desc"),
  };
  if (state.uploadedEvidence) payload.evidence = state.uploadedEvidence;
  byId("generate-button").disabled = true;
  byId("progress-panel").hidden = false;
  try {
    const job = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    state.activeJobId = job.job_id;
    renderProgress(job);
    pollRun(job.job_id);
  } catch (error) {
    byId("generate-button").disabled = !state.status?.ready;
    showToast(error.message, true);
  }
}

function createRunStatus(status) {
  const badge = document.createElement("span");
  badge.className = `run-status is-${status}`;
  badge.textContent = status;
  return badge;
}

async function openRun(jobId) {
  try {
    const job = await api(`/api/runs/${jobId}`);
    if (job.status === "complete") {
      renderAnswer(job.answer);
      setView("generate");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      showToast(`Run is ${job.status}`);
    }
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderRuns(runs) {
  const root = byId("run-list");
  root.replaceChildren();
  if (!runs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No generation runs in this server session.";
    root.append(empty);
    return;
  }
  runs.forEach((run) => {
    const row = document.createElement("article");
    row.className = "run-row";
    const identity = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = run.job_id;
    const topic = document.createElement("span");
    topic.className = "run-meta";
    topic.textContent = `Topic ${run.topic_id}`;
    identity.append(name, topic);
    const stage = document.createElement("span");
    stage.className = "run-meta";
    stage.textContent = run.progress?.message || "Waiting";
    const status = createRunStatus(run.status);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary-button";
    button.textContent = run.status === "complete" ? "Open" : "View";
    button.addEventListener("click", () => openRun(run.job_id));
    row.append(identity, stage, status, button);
    root.append(row);
  });
}

async function refreshRuns() {
  try {
    const payload = await api("/api/runs");
    renderRuns(payload.runs);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadInitialData() {
  byId("download-button").disabled = true;
  byId("copy-button").disabled = true;
  const results = await Promise.allSettled([
    refreshStatus(),
    api("/api/evidence"),
    api("/api/benchmark"),
    api("/api/preview"),
  ]);
  if (results[1].status === "fulfilled") renderEvidence(results[1].value);
  else showToast(results[1].reason.message, true);
  if (results[2].status === "fulfilled") renderBenchmark(results[2].value);
  else showToast(results[2].reason.message, true);
  if (results[3].status === "fulfilled") renderAnswer(results[3].value);
  else showToast(results[3].reason.message, true);
}

document.querySelectorAll("[data-view-link]").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    setView(button.dataset.viewLink);
  });
});

document.querySelectorAll("[data-answer-tab]").forEach((button) => {
  button.addEventListener("click", () => setAnswerTab(button.dataset.answerTab));
});

byId("facet-filter").addEventListener("change", (event) => {
  state.activeFacet = event.target.value;
  renderSentenceList();
});

byId("run-id").addEventListener("input", (event) => {
  event.target.dataset.edited = "true";
});

byId("evidence-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const value = JSON.parse(await file.text());
    state.uploadedEvidence = value;
    renderEvidence(summarizeUploadedEvidence(value), file.name);
    showToast("Evidence ledger loaded");
  } catch (error) {
    event.target.value = "";
    state.uploadedEvidence = null;
    showToast(error.message, true);
  }
});

byId("generation-form").addEventListener("submit", submitGeneration);
byId("refresh-button").addEventListener("click", refreshStatus);
byId("refresh-runs-button").addEventListener("click", refreshRuns);
byId("download-button").addEventListener("click", downloadAnswer);
byId("copy-button").addEventListener("click", () => {
  if (state.answer) copyText(`${JSON.stringify(state.answer.official)}\n`, "Organizer JSONL copied");
});

const requestedView = window.location.hash.replace("#", "");
setView(["generate", "benchmark", "runs"].includes(requestedView) ? requestedView : "generate");
loadInitialData();

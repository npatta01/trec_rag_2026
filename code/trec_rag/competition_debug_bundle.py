"""Allowlisted summary and evaluation views for the competition debug bundle.

The raw debug-report records intentionally remain outside this module's summary
interface.  ``build_run_summary`` projects only the small set of counts and
links needed by the summary page, while ``load_evaluation_overlay`` adapts the
already validated offline-evaluation manifest into typed, read-only views.
"""

from __future__ import annotations

import contextvars
import ctypes
import errno
import json
import html
from html.parser import HTMLParser
import math
import os
import re
import stat
import statistics
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from trec_rag.competition_debug_report import (
    DebugReportData,
    DebugReportBundleReceipt,
    TopicPageNavigation,
    TopicReport,
    _REPORT_SCHEMA_VERSION,
    render_debug_topic_page,
)
from trec_rag.friendly_report import assert_publishable, build_presentation
from trec_rag.offline_evaluation import EvaluationError, load_manifest
from trec_rag.repo_env import find_repo_root


@dataclass(frozen=True)
class MetricAvailability:
    available: bool
    reason: str | None


@dataclass(frozen=True)
class MetricFamilyOverlay:
    key: str
    label: str
    definitions: Mapping[str, str]
    macro_rule: str
    macro: Mapping[str, float]
    macro_availability: MetricAvailability
    per_topic: Mapping[str, Mapping[str, float]]
    per_topic_availability: Mapping[str, MetricAvailability]


@dataclass(frozen=True)
class EvaluationOverlay:
    manifest_sha256: str
    topic_ids: tuple[str, ...]
    families: tuple[MetricFamilyOverlay, ...]
    source_label: str


@dataclass(frozen=True)
class TopicSummary:
    topic_id: str
    health: str
    fallback_kinds: tuple[str, ...]
    depth: int
    subnarratives: int
    queries: int
    nuggets: int
    href: str
    metrics: Mapping[str, Mapping[str, float]]
    metric_availability: Mapping[str, MetricAvailability]


@dataclass(frozen=True)
class RunSummary:
    topics: tuple[TopicSummary, ...]
    completed_topics: int
    fallback_topics: int
    submitted_documents: int
    canonical_nuggets: int
    distributions: Mapping[str, Mapping[str, int | float]]
    evaluation: EvaluationOverlay | None
    rag_included: bool


_FAMILY_LABELS = (
    ("retrieval", "Retrieval relevance"),
    ("nugget_coverage", "Nugget or obligation coverage"),
    ("citation_support", "Answer and citation quality"),
)
_METRIC_BOOKKEEPING_FIELDS = frozenset({"topic_id", "run_id", "sentences"})


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: int | float) -> str:
    """Format a finite number for people without changing its sort value."""
    return f"{value:.6f}"


def _sort_number(value: int | float) -> str:
    """Return a compact, invariant decimal for the DOM sort attribute."""
    return repr(float(value))


def _unavailable(reason: str | None) -> str:
    return f"Unavailable — {_escape(reason or 'not supplied')}"


def _distribution_text(distribution: Mapping[str, int | float]) -> str:
    return (
        f"minimum {_format_number(distribution['minimum'])}; "
        f"median {_format_number(distribution['median'])}; "
        f"maximum {_format_number(distribution['maximum'])}"
    )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} must be an object")
    return value


def _topic_health(topic: TopicReport) -> tuple[str, tuple[str, ...]]:
    fallbacks = {
        result.state
        for result in topic.canonical_results
        if result.state == "fallback_extractive"
    }
    if topic.original_only_fallback:
        fallbacks.add("original_only")
    if fallbacks:
        return "fallback", tuple(sorted(fallbacks))
    if any(result.state == "empty" for result in topic.canonical_results):
        return "empty", ()
    return "complete", ()


def _distribution(values: Sequence[int]) -> Mapping[str, int | float]:
    if not values:
        raise ValueError("a summary distribution requires at least one value")
    return MappingProxyType(
        {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    )


def build_run_summary(
    data: DebugReportData, evaluation: EvaluationOverlay | None
) -> RunSummary:
    """Project validated topic views into the summary's privacy-safe allowlist."""
    topic_summaries: list[TopicSummary] = []
    families_by_key = (
        {family.key: family for family in evaluation.families}
        if evaluation is not None
        else {}
    )
    if evaluation is not None and set(families_by_key) != {
        key for key, _label in _FAMILY_LABELS
    }:
        raise ValueError("evaluation overlay has an unsupported metric family")
    for topic in data.topics:
        health, fallback_kinds = _topic_health(topic)
        metrics: dict[str, Mapping[str, float]] = {}
        metric_availability: dict[str, MetricAvailability] = {}
        if evaluation is not None:
            for family in evaluation.families:
                if topic.topic_id in family.per_topic:
                    metrics[family.key] = family.per_topic[topic.topic_id]
                else:
                    metrics[family.key] = MappingProxyType({})
                availability = family.per_topic_availability.get(topic.topic_id)
                if availability is not None:
                    metric_availability[family.key] = availability
        topic_summaries.append(
            TopicSummary(
                topic_id=topic.topic_id,
                health=health,
                fallback_kinds=fallback_kinds,
                depth=len(topic.retrieval_output.documents),
                subnarratives=len(topic.subnarratives),
                queries=sum(
                    len(subnarrative.bm25_queries)
                    for subnarrative in topic.subnarratives
                ),
                nuggets=len(topic.canonical_nuggets),
                href=f"topics/{topic.topic_id}.html",
                metrics=MappingProxyType(dict(metrics)),
                metric_availability=MappingProxyType(dict(metric_availability)),
            )
        )

    topics = tuple(topic_summaries)
    depths = [topic.depth for topic in topics]
    subnarratives = [topic.subnarratives for topic in topics]
    queries = [topic.queries for topic in topics]
    nuggets = [topic.nuggets for topic in topics]
    distributions = MappingProxyType(
        {
            "depth": _distribution(depths),
            "subnarratives": _distribution(subnarratives),
            "queries": _distribution(queries),
            "nuggets": _distribution(nuggets),
        }
    )
    return RunSummary(
        topics=topics,
        completed_topics=sum(topic.health != "empty" for topic in topics),
        fallback_topics=sum(topic.health == "fallback" for topic in topics),
        submitted_documents=sum(topic.depth for topic in topics),
        canonical_nuggets=sum(topic.nuggets for topic in topics),
        distributions=distributions,
        evaluation=evaluation,
        rag_included=data.rag_config_path is not None,
    )


def _availability(value: object, *, label: str) -> MetricAvailability:
    raw = _mapping(value, label=label)
    available = raw.get("available")
    if type(available) is not bool:
        raise EvaluationError(f"{label}.available must be a boolean")
    reason = raw.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise EvaluationError(f"{label}.reason must be text or null")
    if not available and (not isinstance(reason, str) or not reason.strip()):
        raise EvaluationError(f"{label}: unavailable metrics require a reason")
    return MetricAvailability(available=available, reason=reason)


def _numeric_mapping(
    value: object,
    *,
    label: str,
    allowed_names: set[str] | None = None,
    ignored_names: frozenset[str] = frozenset(),
) -> Mapping[str, float]:
    raw = _mapping(value, label=label)
    projected: dict[str, float] = {}
    for name, metric in raw.items():
        if not isinstance(name, str) or not name:
            raise EvaluationError(f"{label} has an invalid metric name")
        # Some repository manifest cells carry non-score bookkeeping fields
        # (for example ``topic_id``, ``run_id``, and ``sentences``).  The
        # metric definitions are the explicit allowlist for what may cross
        # this seam; bookkeeping is discarded before it can reach the summary.
        if allowed_names is not None and name not in allowed_names:
            if name in ignored_names:
                continue
            raise EvaluationError(f"{label} has an unexpected metric {name!r}")
        if type(metric) not in (int, float) or not math.isfinite(metric):
            raise EvaluationError(f"{label}.{name} must be a finite number")
        # Keep the source number's value and precision. JSON metrics are ints or
        # floats; the annotation is intentionally narrower than the runtime view.
        projected[name] = metric  # type: ignore[assignment]
    return MappingProxyType({name: projected[name] for name in sorted(projected)})


def _string_mapping(value: object, *, label: str) -> Mapping[str, str]:
    raw = _mapping(value, label=label)
    projected: dict[str, str] = {}
    for name, definition in raw.items():
        if not isinstance(name, str) or not name:
            raise EvaluationError(f"{label} has an invalid metric name")
        if not isinstance(definition, str):
            raise EvaluationError(f"{label}.{name} must be text")
        projected[name] = definition
    return MappingProxyType({name: projected[name] for name in sorted(projected)})


def _project_metric_family(
    manifest: Mapping[str, Any],
    *,
    key: str,
    label: str,
    topic_ids: Sequence[str],
    scope_topic_ids: Sequence[str],
    evaluation_topic_count: int,
) -> MetricFamilyOverlay:
    metrics_root = _mapping(manifest.get("metrics"), label="metrics")
    raw = _mapping(metrics_root.get(key), label=f"metrics.{key}")
    definitions_root = _mapping(
        manifest.get("metric_definitions"), label="metric_definitions"
    )
    raw_definitions = definitions_root.get(key, {})
    definitions = _string_mapping(raw_definitions, label=f"metric_definitions.{key}")
    macro_rule = definitions_root.get("macro")
    if not isinstance(macro_rule, str) or not macro_rule.strip():
        raise EvaluationError("metric_definitions.macro must be nonempty text")

    allowed_metric_names = set(definitions)
    macro = _numeric_mapping(
        raw.get("macro"),
        label=f"metrics.{key}.macro",
        allowed_names=allowed_metric_names,
    )
    macro_availability = _availability(
        raw.get("macro_availability"), label=f"metrics.{key}.macro_availability"
    )
    if macro_availability.available:
        if not definitions:
            raise EvaluationError(
                f"metrics.{key}: available macro has no defined metrics"
            )
        if set(macro) != set(definitions):
            raise EvaluationError(
                f"metrics.{key}.macro is missing or has extra defined metrics"
            )
    raw_per_topic = _mapping(raw.get("per_topic"), label=f"metrics.{key}.per_topic")
    raw_availability = _mapping(
        raw.get("per_topic_availability"),
        label=f"metrics.{key}.per_topic_availability",
    )

    scope_ids = set(scope_topic_ids)
    extra_metric_topics = set(raw_per_topic) - scope_ids
    if extra_metric_topics:
        raise EvaluationError(
            f"metrics.{key}.per_topic contains topics outside the evaluation scope"
        )
    extra_availability_topics = set(raw_availability) - scope_ids
    if extra_availability_topics:
        raise EvaluationError(
            f"metrics.{key}.per_topic_availability contains topics outside the evaluation scope"
        )

    per_topic: dict[str, Mapping[str, float]] = {}
    per_topic_availability: dict[str, MetricAvailability] = {}
    selected_ids = set(topic_ids)
    for topic_id in scope_topic_ids:
        if topic_id not in raw_availability:
            raise EvaluationError(
                f"metrics.{key}.per_topic_availability is missing {topic_id}"
            )
        availability = _availability(
            raw_availability[topic_id],
            label=f"metrics.{key}.per_topic_availability.{topic_id}",
        )
        if topic_id in selected_ids:
            per_topic_availability[topic_id] = availability
        if topic_id in raw_per_topic:
            projected = _numeric_mapping(
                raw_per_topic[topic_id],
                label=f"metrics.{key}.per_topic.{topic_id}",
                allowed_names=allowed_metric_names,
                ignored_names=_METRIC_BOOKKEEPING_FIELDS,
            )
            if availability.available:
                if not definitions:
                    raise EvaluationError(
                        f"metrics.{key}.{topic_id}: available cell has no defined metrics"
                    )
                if set(projected) != set(definitions):
                    raise EvaluationError(
                        f"metrics.{key}.{topic_id} is missing or has extra defined metrics"
                    )
            if topic_id in selected_ids:
                per_topic[topic_id] = projected
        elif availability.available:
            raise EvaluationError(
                f"metrics.{key}.{topic_id}: available cell is missing"
            )

    # A scope mismatch invalidates an otherwise available aggregate.  Preserve
    # a stronger repository-owned unavailability reason (for example the
    # absence of released gold nuggets), because replacing it with a scope
    # warning would hide the actual reason that family cannot be scored.
    if len(topic_ids) != evaluation_topic_count and macro_availability.available:
        macro_availability = MetricAvailability(
            available=False,
            reason=(
                f"{evaluation_topic_count}-topic evaluation scope does not match the "
                f"{len(topic_ids)}-topic report scope"
            ),
        )

    return MetricFamilyOverlay(
        key=key,
        label=label,
        definitions=definitions,
        macro_rule=macro_rule,
        macro=macro,
        macro_availability=macro_availability,
        per_topic=MappingProxyType(dict(per_topic)),
        per_topic_availability=MappingProxyType(dict(per_topic_availability)),
    )


def load_evaluation_overlay(
    path: Path, selected_topic_ids: Sequence[str]
) -> EvaluationOverlay:
    """Validate and project an offline-evaluation manifest into safe metric views."""
    path = Path(path)
    payload = path.read_bytes()
    try:
        snapshot_manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{path}: manifest is not valid JSON") from error
    try:
        loaded_manifest = load_manifest(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{path}: manifest changed during load") from error
    if loaded_manifest != snapshot_manifest:
        raise EvaluationError(f"{path}: manifest changed during load")
    manifest = snapshot_manifest
    try:
        build_presentation(manifest)
    except EvaluationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"{path}: invalid evaluation manifest: {error}") from error

    scope_raw = _mapping(manifest.get("scope"), label="scope")
    scope_values = scope_raw.get("topic_ids")
    if not isinstance(scope_values, Sequence) or isinstance(scope_values, (str, bytes)):
        raise EvaluationError("scope.topic_ids must be a list")
    scope = tuple(scope_values)
    if any(not isinstance(topic_id, str) or not topic_id for topic_id in scope):
        raise EvaluationError("scope.topic_ids must contain nonempty text")
    if len(set(scope)) != len(scope):
        raise EvaluationError("evaluation scope repeats a topic")

    selected = tuple(selected_topic_ids)
    if not selected or any(not isinstance(topic_id, str) or not topic_id for topic_id in selected):
        raise EvaluationError("selected report scope must contain topic IDs")
    if len(set(selected)) != len(selected):
        raise EvaluationError("selected report scope repeats a topic")
    positions = [scope.index(topic_id) for topic_id in selected if topic_id in scope]
    if len(positions) != len(selected):
        raise EvaluationError("evaluation scope is missing a selected report topic")
    if positions != sorted(positions):
        raise EvaluationError("evaluation and report topic order conflict")

    families = tuple(
        _project_metric_family(
            manifest,
            key=key,
            label=label,
            topic_ids=selected,
            scope_topic_ids=scope,
            evaluation_topic_count=len(scope),
        )
        for key, label in _FAMILY_LABELS
    )
    return EvaluationOverlay(
        manifest_sha256=sha256(payload).hexdigest(),
        topic_ids=selected,
        families=families,
        source_label="Validated offline evaluation manifest",
    )


# ---------------------------------------------------------------------------
# Privacy-safe summary HTML
# ---------------------------------------------------------------------------


_SUMMARY_STYLE = """
:root {
  --bg: #eef3f8; --surface: #ffffff; --surface-muted: #f7f9fc;
  --ink: #172033; --muted: #4a5870; --line: #d8e1ec; --blue: #2563eb;
  --blue-soft: #dbeafe; --green: #15803d; --green-soft: #dcfce7;
  --amber: #9a5b00; --amber-soft: #fff4d6; --red: #b42318;
  --red-soft: #fee4e2; --shadow: 0 14px 38px rgb(32 48 75 / 9%);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: var(--bg);
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; background: var(--bg); }
main { width: min(100% - 2rem, 78rem); margin: 0 auto; padding: 2rem 0 4rem; }
a { color: var(--blue); }
.hero, .section { background: var(--surface); border: 1px solid var(--line);
  border-radius: 1.1rem; box-shadow: var(--shadow); margin-top: 1rem; }
.hero { margin-top: 0; padding: clamp(1.4rem, 4vw, 3rem); }
.section { padding: clamp(1.2rem, 3vw, 2rem); }
.eyebrow { color: var(--blue); font-size: .76rem; font-weight: 800;
  letter-spacing: .09em; text-transform: uppercase; }
h1 { margin: .55rem 0 0; font-size: clamp(2rem, 6vw, 3.7rem); line-height: 1;
  letter-spacing: -.045em; }
h2 { margin: 0; font-size: clamp(1.35rem, 3vw, 1.9rem); letter-spacing: -.025em; }
h3 { margin: 0; font-size: 1rem; }
p { line-height: 1.55; }
.lede { max-width: 54rem; margin: 1rem 0 0; color: var(--muted); font-size: 1.05rem; }
.status-line { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.2rem; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: .42rem .7rem;
  background: var(--surface-muted); color: var(--muted); font-size: .84rem; font-weight: 700; }
.pill.ok { color: var(--green); background: var(--green-soft); border-color: #b5e1c3; }
.pill.attention { color: var(--amber); background: var(--amber-soft); border-color: #f0cf8b; }
.section-head { display: flex; justify-content: space-between; align-items: baseline;
  gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.section-note, .caption { color: var(--muted); font-size: .9rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: .75rem; }
.card { padding: 1rem; border: 1px solid var(--line); border-radius: .85rem; background: var(--surface-muted); }
.card-label { color: var(--muted); font-size: .8rem; font-weight: 700; }
.card-value { display: block; margin-top: .35rem; font-size: 1.55rem; font-weight: 800; }
.card-detail { display: block; margin-top: .35rem; color: var(--muted); font-size: .8rem; line-height: 1.4; }
.distribution-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: .75rem; }
.distribution { border: 1px solid var(--line); border-radius: .85rem; padding: 1rem; }
.distribution strong { display: block; margin-bottom: .4rem; }
.distribution span { color: var(--muted); font-variant-numeric: tabular-nums; font-size: .88rem; }
.family + .family { margin-top: 1.4rem; padding-top: 1.4rem; border-top: 1px solid var(--line); }
.family-rule { margin: .4rem 0 .85rem; color: var(--muted); font-size: .9rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: .7rem; }
.metric-card { border: 1px solid var(--line); border-radius: .75rem; padding: .85rem; }
.metric-card h3 { font-size: .95rem; }
.metric-value { display: block; margin-top: .3rem; font-size: 1.2rem; font-variant-numeric: tabular-nums; }
.metric-definition { display: block; margin-top: .35rem; color: var(--muted); font-size: .8rem; line-height: 1.4; }
.unavailable { color: var(--amber); font-weight: 700; }
.toolbar { display: flex; flex-wrap: wrap; align-items: end; gap: .7rem; margin-bottom: 1rem; }
.toolbar label { display: grid; gap: .25rem; color: var(--muted); font-size: .82rem; font-weight: 700; }
.toolbar input, .toolbar select, .toolbar button { min-height: 2.35rem; border: 1px solid #aebdce;
  border-radius: .5rem; background: var(--surface); color: var(--ink); padding: .45rem .65rem; font: inherit; }
.toolbar input { min-width: min(25rem, 100%); }
.toolbar button { cursor: pointer; color: var(--blue); font-weight: 700; }
.toolbar button:hover, .toolbar button:focus-visible { background: var(--blue-soft); }
.sort-status { margin: 0 0 .8rem; color: var(--muted); font-size: .85rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: .75rem; }
table { width: 100%; border-collapse: collapse; min-width: 58rem; }
caption { padding: .8rem 1rem; text-align: left; color: var(--muted); font-size: .86rem; }
th, td { border-top: 1px solid var(--line); padding: .75rem .7rem; text-align: left; vertical-align: top; }
th { background: var(--surface-muted); color: var(--muted); font-size: .78rem; letter-spacing: .02em; }
th button { border: 0; padding: 0; background: transparent; color: inherit; cursor: pointer; font: inherit; font-weight: 800; text-align: left; }
th button:hover, th button:focus-visible { color: var(--blue); text-decoration: underline; }
td { font-size: .88rem; }
td.numeric { font-variant-numeric: tabular-nums; white-space: nowrap; }
td.status-fallback, td.status-empty { color: var(--amber); font-weight: 700; }
td.status-complete { color: var(--green); font-weight: 700; }
.topic-link { font-weight: 750; }
.family-label { display: block; margin-bottom: .2rem; color: var(--muted); font-size: .7rem; font-weight: 600; }
@media (max-width: 40rem) {
  main { width: min(100% - 1rem, 78rem); padding-top: 1rem; }
  .hero, .section { border-radius: .8rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; } }
"""


_SUMMARY_SCRIPT = r"""
(() => {
  const table = document.querySelector("#topic-table");
  if (!table) return;
  const rows = Array.from(document.querySelectorAll("#topic-table tbody tr"));
  const official = new Map(rows.map((row, index) => [row, Number(row.dataset.officialIndex ?? index)]));
  const missing = Number.POSITIVE_INFINITY;
  const headers = Array.from(document.querySelectorAll("#topic-table thead th[data-column]"));
  const search = document.querySelector("#topic-search");
  const health = document.querySelector("#health-filter");
  const reset = document.querySelector("#attention-reset");
  const sortStatus = document.querySelector("#sort-status");

  function numericValue(row, column) {
    const cell = row.cells[column];
    return cell && cell.dataset.sortValue !== undefined
      ? Number(cell.dataset.sortValue)
      : missing;
  }

  function sortRows(column, direction) {
    rows.sort((left, right) => {
      const leftValue = numericValue(left, column);
      const rightValue = numericValue(right, column);
      const delta = leftValue - rightValue;
      if (Number.isFinite(delta) && delta !== 0) return direction * delta;
      if (leftValue !== rightValue) return leftValue === missing ? 1 : -1;
      return official.get(left) - official.get(right);
    });
    rows.forEach((row) => row.parentNode.appendChild(row));
  }

  function needsAttention(row) {
    const state = row.dataset.health;
    if (state === "empty" || state === "failure") return 0;
    if (state === "fallback") return 1;
    return 2;
  }

  function sortNeedsAttention() {
    rows.sort((left, right) => needsAttention(left) - needsAttention(right) || official.get(left) - official.get(right));
    rows.forEach((row) => row.parentNode.appendChild(row));
  }

  function applyFilter() {
    const query = (search?.value || "").trim().toLowerCase();
    const selectedHealth = (health?.value || "all").toLowerCase();
    rows.forEach((row) => {
      const searchable = `${row.dataset.topicId || ""} ${row.dataset.health || ""} ${row.dataset.fallback || ""}`.toLowerCase();
      const matchesQuery = !query || searchable.includes(query);
      const matchesHealth = selectedHealth === "all" || (row.dataset.health || "").toLowerCase() === selectedHealth;
      row.hidden = !(matchesQuery && matchesHealth);
    });
  }

  function markSort(activeHeader, direction) {
    headers.forEach((header) => {
      header.setAttribute("aria-sort", header === activeHeader ? (direction > 0 ? "ascending" : "descending") : "none");
    });
  }

  headers.forEach((header) => {
    const button = header.querySelector("button");
    if (!button) return;
    button.addEventListener("click", () => {
      const direction = button.dataset.direction === "ascending" ? -1 : 1;
      button.dataset.direction = direction > 0 ? "ascending" : "descending";
      sortRows(Number(header.dataset.column), direction);
      markSort(header, direction);
      if (sortStatus) sortStatus.textContent = `Sorted by ${button.dataset.label} ${direction > 0 ? "ascending" : "descending"}.`;
      applyFilter();
    });
  });
  search?.addEventListener("input", applyFilter);
  health?.addEventListener("change", applyFilter);
  reset?.addEventListener("click", () => {
    sortNeedsAttention();
    headers.forEach((header) => {
      const button = header.querySelector("button");
      if (button) button.dataset.direction = "";
      header.setAttribute("aria-sort", "none");
    });
    if (sortStatus) sortStatus.textContent = "Needs attention: failures and fallbacks first, then official topic order.";
    applyFilter();
  });
  sortNeedsAttention();
  applyFilter();
})();
"""


def _render_health_cards(summary: RunSummary) -> str:
    selected = len(summary.topics)
    depth = summary.distributions["depth"]
    fallback_detail = "none"
    kind_counts: dict[str, int] = {}
    for topic in summary.topics:
        for kind in topic.fallback_kinds:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    if kind_counts:
        fallback_detail = ", ".join(
            f"{kind}: {kind_counts[kind]}" for kind in sorted(kind_counts)
        )
    if summary.evaluation is None:
        evaluation_state = '<span class="unavailable">Evaluation not supplied</span>'
    else:
        evaluation_state = _escape(summary.evaluation.source_label)
    return f"""
    <section class="section" aria-labelledby="health-heading">
      <div class="section-head"><div><h2 id="health-heading">Run health</h2>
        <p class="section-note">Validated run counts and completion state.</p></div></div>
      <div class="cards">
        <div class="card"><span class="card-label">Completed topics</span>
          <strong class="card-value">{summary.completed_topics} / {selected}</strong></div>
        <div class="card"><span class="card-label">Fallback topics</span>
          <strong class="card-value">{summary.fallback_topics}</strong>
          <span class="card-detail">Kinds: {_escape(fallback_detail)}</span></div>
        <div class="card"><span class="card-label">Submitted documents</span>
          <strong class="card-value">{summary.submitted_documents}</strong></div>
        <div class="card"><span class="card-label">Final depth</span>
          <strong class="card-value">{_format_number(depth['median'])}</strong>
          <span class="card-detail">Range {_format_number(depth['minimum'])}–{_format_number(depth['maximum'])}</span></div>
        <div class="card"><span class="card-label">Canonical nuggets</span>
          <strong class="card-value">{summary.canonical_nuggets}</strong></div>
        <div class="card"><span class="card-label">Evaluation</span>
          <strong class="card-value">{evaluation_state}</strong></div>
      </div>
    </section>
    """


def _render_distributions(summary: RunSummary) -> str:
    labels = (
        ("depth", "Final retrieval depth"),
        ("subnarratives", "Subnarratives"),
        ("queries", "Generated BM25 queries"),
        ("nuggets", "Canonical nuggets"),
    )
    cards = "\n".join(
        f'<div class="distribution"><strong>{_escape(label)}</strong>'
        f'<span>{_distribution_text(summary.distributions[key])}</span></div>'
        for key, label in labels
    )
    return f"""
    <section class="section" aria-labelledby="distribution-heading">
      <div class="section-head"><div><h2 id="distribution-heading">Run scale</h2>
        <p class="section-note">Minimum, median, and maximum across selected topics.</p></div></div>
      <div class="distribution-grid">{cards}</div>
    </section>
    """


def _render_metric_value(
    value: int | float | None,
    availability: MetricAvailability | None,
    *,
    include_sort: bool = False,
) -> str:
    if availability is not None and not availability.available:
        return f'<span class="unavailable">{_unavailable(availability.reason)}</span>'
    if value is None:
        return '<span class="unavailable">Unavailable — metric not supplied</span>'
    sort = f' data-sort-kind="number" data-sort-value="{_escape(_sort_number(value))}"' if include_sort else ""
    return f'<span class="numeric"{sort}>{_escape(_format_number(value))}</span>'


def _render_score_families(summary: RunSummary) -> str:
    if summary.evaluation is None:
        families = tuple(
            f'<section class="family" aria-labelledby="family-{key}">'
            f'<h2 id="family-{key}">{_escape(label)}</h2>'
            '<p class="family-rule"><span class="unavailable">Evaluation not supplied</span></p>'
            '<p class="caption">Unavailable — evaluation not supplied.</p></section>'
            for key, label in _FAMILY_LABELS
        )
    else:
        rendered: list[str] = []
        for family in summary.evaluation.families:
            metrics = sorted(family.definitions)
            cards: list[str] = []
            for name in metrics:
                value = family.macro.get(name)
                metric_value = value if family.macro_availability.available else None
                cards.append(
                    f'<article class="metric-card" data-metric="{_escape(family.key + ":" + name)}">'
                    f'<h3>{_escape(name)}</h3>'
                    f'<span class="metric-value">{_render_metric_value(metric_value, family.macro_availability)}</span>'
                    f'<span class="metric-definition">{_escape(family.definitions[name])}</span></article>'
                )
            if not metrics:
                cards.append(
                    f'<p class="caption"><span class="unavailable">{_unavailable(family.macro_availability.reason)}</span></p>'
                )
            rendered.append(
                f'<section class="family" aria-labelledby="family-{_escape(family.key)}">'
                f'<h2 id="family-{_escape(family.key)}">{_escape(family.label)}</h2>'
                f'<p class="family-rule">{_escape(family.macro_rule)}</p>'
                f'<div class="metric-grid">{"".join(cards)}</div></section>'
            )
        families = tuple(rendered)
    return f"""
    <section class="section" aria-labelledby="scores-heading">
      <div class="section-head"><div><h2 id="scores-heading">Evaluation score families</h2>
        <p class="section-note">Metric families remain separate; no cross-family composite is calculated.</p></div></div>
      {''.join(families)}
    </section>
    """


def _topic_metric_columns(summary: RunSummary) -> tuple[tuple[str, str, str], ...]:
    if summary.evaluation is None:
        return ()
    return tuple(
        (family.key, name, family.label)
        for family in summary.evaluation.families
        for name in sorted(family.definitions)
    )


def _render_topic_table(summary: RunSummary) -> str:
    columns = _topic_metric_columns(summary)
    headers = [
        '<th scope="col">Topic</th>',
        '<th scope="col">Health</th>',
        '<th scope="col" data-column="2" data-sort-kind="number" aria-sort="none"><button type="button" data-label="final depth">Final depth</button></th>',
        '<th scope="col" data-column="3" data-sort-kind="number" aria-sort="none"><button type="button" data-label="subnarratives">Subnarratives</button></th>',
        '<th scope="col" data-column="4" data-sort-kind="number" aria-sort="none"><button type="button" data-label="generated BM25 queries">Queries</button></th>',
        '<th scope="col" data-column="5" data-sort-kind="number" aria-sort="none"><button type="button" data-label="canonical nuggets">Nuggets</button></th>',
        '<th scope="col">Raw trace</th>',
    ]
    for index, (family_key, name, family_label) in enumerate(columns, start=7):
        headers.append(
            f'<th scope="col" data-column="{index}" data-sort-kind="number" '
            f'data-metric="{_escape(family_key + ":" + name)}" aria-sort="none">'
            f'<span class="family-label">{_escape(family_label)}</span>'
            f'<button type="button" data-label="{_escape(family_label + " " + name)}">{_escape(name)}</button></th>'
        )

    # Keep the source order in static HTML.  The script progressively enhances
    # it into the Needs attention order when JavaScript is available.
    ordered_topics = summary.topics
    rows: list[str] = []
    for topic in ordered_topics:
        official_order = summary.topics.index(topic)
        fallback = ", ".join(topic.fallback_kinds)
        state_label = topic.health if not fallback else f"{topic.health}: {fallback}"
        cells = [
            f'<td><a class="topic-link" href="{_escape(topic.href)}">{_escape(topic.topic_id)}</a></td>',
            f'<td class="status-{_escape(topic.health)}">{_escape(state_label)}</td>',
        ]
        for column, value in enumerate((topic.depth, topic.subnarratives, topic.queries, topic.nuggets), start=2):
            cells.append(
                f'<td class="numeric" data-sort-kind="number" data-sort-value="{_escape(str(value))}">{value}</td>'
            )
        cells.append(f'<td><a href="{_escape(topic.href)}">Open topic</a></td>')
        for family_key, metric_name, _family_label in columns:
            availability = topic.metric_availability.get(family_key)
            value = topic.metrics.get(family_key, {}).get(metric_name)
            if availability is None:
                availability = MetricAvailability(False, "metric not supplied")
            if availability.available and value is not None:
                cells.append(
                    f'<td class="numeric" data-sort-kind="number" data-sort-value="{_escape(_sort_number(value))}" '
                    f'data-metric="{_escape(family_key + ":" + metric_name)}">{_escape(_format_number(value))}</td>'
                )
            else:
                cells.append(
                    f'<td data-metric="{_escape(family_key + ":" + metric_name)}">'
                    f'<span class="unavailable">{_unavailable(availability.reason)}</span></td>'
                )
        rows.append(
            f'<tr data-topic-id="{_escape(topic.topic_id)}" data-health="{_escape(topic.health)}" '
            f'data-fallback="{_escape(fallback)}" data-official-index="{official_order}">'
            f'{"".join(cells)}</tr>'
        )
    return f"""
    <section class="section" aria-labelledby="topics-heading">
      <div class="section-head"><div><h2 id="topics-heading">Topics</h2>
        <p class="section-note">Initial order puts failures and fallbacks first; score families never collapse into one ranking.</p></div></div>
      <div class="toolbar" aria-label="Topic controls">
        <label for="topic-search">Search topics or status
          <input id="topic-search" type="search" aria-label="Search topics or status" placeholder="Topic ID or health state">
        </label>
        <label for="health-filter">Health
          <select id="health-filter" aria-label="Filter by topic health">
            <option value="all">All health states</option><option value="complete">Complete</option>
            <option value="fallback">Fallback</option><option value="empty">Empty</option>
          </select>
        </label>
        <button id="attention-reset" type="button" aria-label="Reset to Needs attention order">Needs attention</button>
      </div>
      <p id="sort-status" class="sort-status" aria-live="polite">Needs attention: failures and fallbacks first, then official topic order.</p>
      <div class="table-wrap"><table id="topic-table"><caption>Per-topic counts and separately reported evaluation metrics. Unavailable values are not zero.</caption>
        <thead><tr>{"".join(headers)}</tr></thead><tbody>{"".join(rows)}</tbody>
      </table></div>
    </section>
    """


def render_bundle_summary(summary: RunSummary, *, denylist: Sequence[str]) -> str:
    """Render and privacy-scan the allowlisted run summary.

    The summary renderer receives only ``RunSummary``; raw topic records are not
    available to the template.  The scan is deliberately performed once, on the
    final page, after all escaped values and static behaviour have been assembled.
    """
    body = "\n".join(
        (
            _render_health_cards(summary),
            _render_distributions(summary),
            _render_score_families(summary),
            _render_topic_table(summary),
        )
    )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run summary</title><style>{_SUMMARY_STYLE}</style></head>
<body><main><header class="hero"><div class="eyebrow">Private competition debug bundle</div>
<h1>Run summary</h1><p class="lede">A privacy-safe overview of retrieval health, run scale, and separately reported evaluation evidence. Detailed topic traces are linked only as private local pages.</p>
<div class="status-line"><span class="pill ok">{len(summary.topics)} selected topics</span>
<span class="pill {'attention' if summary.fallback_topics else 'ok'}">{summary.fallback_topics} fallback topics</span>
<span class="pill">{'RAG output supplied' if summary.rag_included else 'RAG output not supplied'}</span></div></header>
{body}
</main><script>{_SUMMARY_SCRIPT}</script></body></html>'''
    assert_publishable(page, denylist=tuple(denylist))
    return page


def _summary_denylist(data: DebugReportData) -> tuple[str, ...]:
    """Collect run-derived short sensitive values for a summary privacy scan.

    This projection intentionally includes identifiers and provenance digests but
    excludes long narrative, query, answer, and passage text.  Generic privacy
    patterns in ``friendly_report`` cover credential-shaped and provider fields.
    """
    values: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, str) and value.strip():
            values.add(value)

    for path in (data.retrieval_config_path, data.rag_config_path, data.output_dir):
        if path is not None:
            add(str(Path(path).resolve()))
    for source, digest in data.source_sha256s.items():
        add(str(source))
        add(str(digest))
    for topic in data.topics:
        add(topic.narrative_sha256)
        for subnarrative in topic.subnarratives:
            add(subnarrative.semantic_query_sha256)
            for digest in subnarrative.bm25_query_sha256s:
                add(digest)
        for document in topic.retrieval_output.documents:
            add(document.docid)
            for digest in document.source_seals.values():
                add(digest)
            for score in document.subnarrative_scores:
                add(score.get("semantic_query_sha256"))
                add(score.get("text_sha256"))
                for digest in score.get("bm25_query_sha256s", ()):
                    add(digest)
        for document in topic.selected_documents:
            add(document.docid)
            add(document.text_sha256)
        for document in topic.new_documents:
            add(document.docid)
            add(document.text_sha256)
            for provenance in document.lane_provenance:
                add(provenance.text_sha256)
        for ranking in topic.passage_rankings:
            add(ranking.docid)
            add(ranking.document_sha256)
            for passage in ranking.winning_passages:
                add(passage.passage_id)
        for cluster in topic.evidence_clusters:
            add(cluster.cluster_id)
            for evidence in cluster.evidence:
                add(evidence.docid)
                add(evidence.document_sha256)
                add(evidence.text_sha256)
        for nugget in topic.canonical_nuggets:
            add(nugget.canonical_nugget_id)
            for evidence in nugget.evidence:
                add(evidence.docid)
                add(evidence.document_sha256)
                add(evidence.text_sha256)
        if topic.rag_output is not None:
            add(topic.rag_output.run_id)
            add(topic.rag_output.output_sha256)
            for reference in topic.rag_output.references:
                add(reference)
            for item in topic.rag_output.answer_items:
                for docid in item.citation_docids:
                    add(docid)
    return tuple(sorted(values))


# ---------------------------------------------------------------------------
# Create-only bundle publication
# ---------------------------------------------------------------------------


_BUNDLE_SCHEMA_VERSION = "competition_debug_report_bundle_v1"
_SAFE_TOPIC_FILENAME = re.compile(r"[^/\\\x00-\x1f\x7f]+")
_FileIdentity = tuple[int, int]


@dataclass(frozen=True)
class _PageReceipt:
    path: str
    bytes: int
    sha256: str
    topic_id: str | None = None

    def as_json(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }
        if self.topic_id is not None:
            value["topic_id"] = self.topic_id
        return value


def _lstat_identity(path: Path, *, directory: bool = False) -> _FileIdentity:
    try:
        record = path.lstat()
    except FileNotFoundError as error:
        kind = "directory" if directory else "file"
        raise ValueError(f"bundle {kind} disappeared: {path}") from error
    if stat.S_ISLNK(record.st_mode):
        raise ValueError(f"bundle path must not be a symbolic link: {path}")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(record.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"bundle path is not a {kind}: {path}")
    return (record.st_dev, record.st_ino)


def _identity_matches(path: Path, expected: _FileIdentity, *, directory: bool = False) -> bool:
    try:
        return _lstat_identity(path, directory=directory) == expected
    except ValueError:
        return False


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


_FILE_IDENTITY_REGISTRAR: contextvars.ContextVar[
    Callable[[Path, _FileIdentity], None] | None
] = contextvars.ContextVar("bundle_file_identity_registrar", default=None)


def _write_bundle_file(path: Path, payload: bytes) -> _FileIdentity:
    """Exclusively create one private bundle file and durably write its bytes."""
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        record = os.fstat(descriptor)
        identity = (record.st_dev, record.st_ino)
        registrar = _FILE_IDENTITY_REGISTRAR.get()
        if registrar is not None:
            registrar(path, identity)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except Exception:
        if descriptor != -1:
            os.close(descriptor)
        raise
    return identity


def _file_receipt(path: Path, relative_path: str, *, topic_id: str | None = None) -> _PageReceipt:
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return _PageReceipt(
        path=relative_path,
        bytes=size,
        sha256=digest.hexdigest(),
        topic_id=topic_id,
    )


def _hash_matches(path: Path, expected_sha256: str, expected_bytes: int) -> bool:
    try:
        _lstat_identity(path)
        actual = _file_receipt(path, "")
    except (OSError, ValueError):
        return False
    return actual.bytes == expected_bytes and actual.sha256 == expected_sha256


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_fd(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _open_directory_fd_at(parent_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )


def _fsync_directory_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _read_receipt_from_fd(
    descriptor: int,
    relative_path: str,
    *,
    topic_id: str | None = None,
) -> _PageReceipt:
    digest = sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    return _PageReceipt(
        path=relative_path,
        bytes=size,
        sha256=digest.hexdigest(),
        topic_id=topic_id,
    )


def _validate_pinned_payload(
    guardian_fd: int,
    payload_name: str,
    expected_payload_identity: _FileIdentity,
    expected_file_identities: Mapping[str, _FileIdentity],
    expected_receipts: Sequence[_PageReceipt],
    expected_topic_ids: Sequence[str],
) -> None:
    """Validate the complete payload through descriptor-relative operations.

    This is deliberately the last authoritative filesystem check in the
    publication primitive.  Callers may perform an earlier reconciliation for
    link validation, but a wrapper that mutates a page after that check still
    reaches this descriptor-pinned identity/hash/coverage check before the
    no-replace syscall.
    """
    payload_fd = _open_directory_fd_at(guardian_fd, payload_name)
    topics_fd: int | None = None
    try:
        payload_record = os.fstat(payload_fd)
        payload_identity = (payload_record.st_dev, payload_record.st_ino)
        if payload_identity != expected_payload_identity or not stat.S_ISDIR(
            payload_record.st_mode
        ):
            raise ValueError("bundle staging payload identity changed before publication")

        receipt_by_path = {receipt.path: receipt for receipt in expected_receipts}
        if len(receipt_by_path) != len(expected_receipts):
            raise ValueError("bundle publication receipts contain duplicate paths")
        expected_paths = set(expected_file_identities)
        if expected_paths != set(receipt_by_path):
            raise ValueError("bundle publication file coverage does not reconcile")
        topic_receipts = tuple(
            receipt
            for receipt in expected_receipts
            if receipt.path.startswith("topics/")
        )
        if tuple(receipt.topic_id for receipt in topic_receipts) != tuple(expected_topic_ids):
            raise ValueError("bundle publication topic coverage does not reconcile")
        if any(receipt.topic_id is not None for receipt in expected_receipts if not receipt.path.startswith("topics/")):
            raise ValueError("bundle publication receipt coverage is invalid")

        expected_direct = {
            path for path in expected_paths if len(Path(path).parts) == 1
        }
        expected_topics = {
            Path(path).name
            for path in expected_paths
            if Path(path).parts[:1] == ("topics",)
        }
        if set(os.listdir(payload_fd)) != {"topics", *expected_direct}:
            raise ValueError(
                "bundle publication payload identity/coverage does not reconcile"
            )
        topics_fd = _open_directory_fd_at(payload_fd, "topics")
        if set(os.listdir(topics_fd)) != expected_topics:
            raise ValueError("bundle publication topic-file coverage does not reconcile")

        for relative_path, expected_identity in sorted(expected_file_identities.items()):
            parts = Path(relative_path).parts
            if len(parts) == 1:
                directory_fd = payload_fd
                name = parts[0]
            elif len(parts) == 2 and parts[0] == "topics":
                if topics_fd is None:
                    raise ValueError("bundle publication topics directory is unavailable")
                directory_fd = topics_fd
                name = parts[1]
            else:
                raise ValueError("bundle publication contains an unsafe file path")
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(
                    f"bundle file identity changed for {relative_path} before publication"
                ) from error
            try:
                before = os.fstat(descriptor)
                actual_identity = (before.st_dev, before.st_ino)
                if not stat.S_ISREG(before.st_mode) or actual_identity != expected_identity:
                    raise ValueError(
                        f"bundle file identity changed for {relative_path} before publication"
                    )
                after = os.fstat(descriptor)
                if (
                    (after.st_dev, after.st_ino) != expected_identity
                    or not stat.S_ISREG(after.st_mode)
                    or before.st_size != receipt_by_path[relative_path].bytes
                    or after.st_size != receipt_by_path[relative_path].bytes
                ):
                    raise ValueError(
                        f"bundle size or identity reconciliation failed for {relative_path}"
                    )
            finally:
                os.close(descriptor)
    finally:
        if topics_fd is not None:
            os.close(topics_fd)
        os.close(payload_fd)


def _rename_noreplace_fds(
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
    *,
    expected_old_identity: _FileIdentity | None = None,
    expected_file_identities: Mapping[str, _FileIdentity] | None = None,
    expected_receipts: Sequence[_PageReceipt] | None = None,
    expected_topic_ids: Sequence[str] | None = None,
    target_parent_path: Path | None = None,
    expected_target_parent_identity: _FileIdentity | None = None,
    approved_roots: Sequence[Path] = (),
) -> int | None:
    """Atomically rename one descriptor-relative entry without replacing a target.

    For publication, ``target_parent_path`` causes the destination descriptor to
    be opened afresh here.  The caller's setup descriptor is never used as the
    publication destination, so a moved-and-recreated pathname cannot redirect
    the rename.  The fresh descriptor is closed after the syscall; the caller
    opens a separate fresh descriptor for the post-rename parent fsync.
    """
    publication_parent_fd: int | None = None
    try:
        destination_fd = new_dir_fd
        if target_parent_path is not None:
            target_parent = Path(target_parent_path)
            try:
                resolved_parent = target_parent.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(
                    "bundle target parent disappeared before publication"
                ) from error
            if resolved_parent != target_parent:
                raise ValueError("bundle target parent path changed before publication")
            if approved_roots and not any(
                resolved_parent.is_relative_to(Path(root).resolve())
                for root in approved_roots
            ):
                raise ValueError("bundle target parent is outside approved roots")
            publication_parent_fd = _open_directory_fd(resolved_parent)
            destination_fd = publication_parent_fd
            parent_record = os.fstat(publication_parent_fd)
            parent_identity = (parent_record.st_dev, parent_record.st_ino)
            if (
                expected_target_parent_identity is not None
                and parent_identity != expected_target_parent_identity
            ):
                raise ValueError("bundle target parent identity changed before publication")
        if expected_old_identity is not None:
            try:
                source_record = os.stat(
                    old_name, dir_fd=old_dir_fd, follow_symlinks=False
                )
            except OSError as error:
                raise ValueError(
                    "bundle staging payload disappeared before publication"
                ) from error
            if stat.S_ISLNK(source_record.st_mode) or (
                source_record.st_dev,
                source_record.st_ino,
            ) != expected_old_identity:
                raise ValueError("bundle staging payload identity changed before publication")
        if expected_file_identities is not None:
            if (
                expected_old_identity is None
                or expected_receipts is None
                or expected_topic_ids is None
            ):
                raise ValueError("bundle publication reconciliation inputs are incomplete")
            _validate_pinned_payload(
                old_dir_fd,
                old_name,
                expected_old_identity,
                expected_file_identities,
                expected_receipts,
                expected_topic_ids,
            )
        if sys.platform != "linux":
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory rename is unavailable on this platform",
            )
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory rename is unavailable",
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            old_dir_fd,
            os.fsencode(old_name),
            destination_fd,
            os.fsencode(new_name),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), new_name)
        return None
    finally:
        if publication_parent_fd is not None:
            os.close(publication_parent_fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Compatibility wrapper for descriptor-relative no-replace publication."""
    old_dir_fd = _open_directory_fd(source.parent)
    new_dir_fd = _open_directory_fd(destination.parent)
    try:
        publication_parent_fd = _rename_noreplace_fds(
            old_dir_fd,
            source.name,
            new_dir_fd,
            destination.name,
        )
        if publication_parent_fd is not None:
            os.close(publication_parent_fd)
    finally:
        os.close(old_dir_fd)
        os.close(new_dir_fd)


def _rename_cleanup_noreplace(
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
    *,
    expected_old_identity: _FileIdentity,
) -> None:
    """Move one cleanup candidate without depending on the publication seam."""
    try:
        source_record = os.stat(old_name, dir_fd=old_dir_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError("bundle cleanup candidate disappeared") from error
    source_identity = (source_record.st_dev, source_record.st_ino)
    if stat.S_ISLNK(source_record.st_mode) or source_identity != expected_old_identity:
        raise ValueError("bundle cleanup candidate identity changed")
    if sys.platform != "linux":
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace cleanup rename is unavailable on this platform",
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace cleanup rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        old_dir_fd,
        os.fsencode(old_name),
        new_dir_fd,
        os.fsencode(new_name),
        1,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), new_name)


def _safe_topic_filename(topic_id: str) -> str:
    if (
        not isinstance(topic_id, str)
        or topic_id in {".", ".."}
        or not topic_id
        or _SAFE_TOPIC_FILENAME.fullmatch(topic_id) is None
    ):
        raise ValueError(f"topic ID {topic_id!r} is not a safe topic filename")
    return f"{topic_id}.html"


def _resolve_bundle_output(data: DebugReportData, output_dir: Path) -> Path:
    requested = Path(output_dir)
    if requested.exists() or requested.is_symlink():
        if requested.is_symlink():
            raise ValueError("bundle output must be absent and not a symbolic link")
        raise ValueError("bundle output must be absent")
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    parent = requested.parent.resolve()
    if not parent.is_dir():
        raise ValueError("bundle output parent must be an existing directory")
    target = (parent / requested.name).resolve()
    repo_root = find_repo_root(data.retrieval_config_path.parent).resolve()
    retrieval_output = data.output_dir.resolve()
    if not (target.is_relative_to(repo_root) or target.is_relative_to(retrieval_output)):
        raise ValueError(
            "bundle output must remain inside the repository or retrieval output"
        )
    if target.exists() or target.is_symlink():
        raise ValueError("bundle output must be absent and not a symbolic link")
    return target


@dataclass(frozen=True)
class _PageLinkGraph:
    hrefs: tuple[str, ...]
    anchors: frozenset[str]


def _parse_page_links(page: str) -> _PageLinkGraph:
    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.hrefs: list[str] = []
            self.anchors: set[str] = set()

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            attributes = dict(attrs)
            href = attributes.get("href")
            if tag == "a" and href:
                self.hrefs.append(href)
            for key in ("id", "name"):
                value = attributes.get(key)
                if value:
                    self.anchors.add(value)

    parser = LinkParser()
    parser.feed(page)
    return _PageLinkGraph(
        hrefs=tuple(
        href
        for href in parser.hrefs
        if not href.startswith(("http://", "https://", "mailto:"))
        ),
        anchors=frozenset(parser.anchors),
    )


def _read_page_link_graph(
    staging: Path, page_paths: Mapping[Path, Path]
) -> Mapping[Path, _PageLinkGraph]:
    graph: dict[Path, _PageLinkGraph] = {}
    for relative, page in page_paths.items():
        page_text = page.read_text(encoding="utf-8")
        graph[relative] = _parse_page_links(page_text)
        del page_text
    return graph


def _validate_link_graph(
    staging: Path, graph: Mapping[Path, _PageLinkGraph]
) -> None:
    for source_relative, links in graph.items():
        source = staging / source_relative
        for href in links.hrefs:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
                raise ValueError(f"bundle page contains an unsafe link: {href}")
            relative = parsed.path
            destination = source if not relative else (source.parent / relative).resolve()
            if relative and (
                not destination.is_relative_to(staging) or not destination.is_file()
            ):
                raise ValueError(f"bundle page link does not resolve: {href}")
            destination_relative = destination.relative_to(staging)
            destination_links = graph.get(destination_relative)
            if destination_links is None:
                raise ValueError(f"bundle page link does not resolve: {href}")
            fragment = unquote(parsed.fragment)
            if fragment and fragment not in destination_links.anchors:
                raise ValueError(f"bundle page fragment does not resolve: {href}")


def _validate_page_links(staging: Path, page: Path) -> None:
    """Compatibility helper for one page; publication uses the compact graph."""
    relative = page.relative_to(staging)
    graph = _read_page_link_graph(staging, {relative: page})
    _validate_link_graph(staging, graph)


def _validate_staging_links(staging: Path, page_paths: Mapping[Path, Path]) -> None:
    graph = _read_page_link_graph(staging, page_paths)
    _validate_link_graph(staging, graph)


def _validate_staging(
    staging: Path,
    staging_identity: _FileIdentity,
    data: DebugReportData,
    index_receipt: _PageReceipt,
    topic_receipts: Sequence[_PageReceipt],
    file_identities: Mapping[str, _FileIdentity],
) -> None:
    if not _identity_matches(staging, staging_identity, directory=True):
        raise ValueError("bundle staging directory identity changed")
    expected_files = {Path(index_receipt.path), *(Path(item.path) for item in topic_receipts)}
    if (staging / "bundle-manifest.json").exists():
        expected_files.add(Path("bundle-manifest.json"))
    actual_files = {
        path.relative_to(staging)
        for path in staging.rglob("*")
        if not path.is_dir() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("bundle staging contains unlisted or missing files")
    topic_ids = tuple(topic.topic_id for topic in data.topics)
    if topic_ids != tuple(item.topic_id for item in topic_receipts):
        raise ValueError("bundle topic coverage or order does not reconcile")
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("bundle topic coverage repeats a topic")
    if len(set(item.path for item in topic_receipts)) != len(topic_receipts):
        raise ValueError("bundle topic paths are not unique")
    pages = (index_receipt, *topic_receipts)
    page_paths: dict[Path, Path] = {}
    for receipt in pages:
        path = staging / Path(receipt.path)
        page_paths[Path(receipt.path)] = path
        identity = file_identities.get(receipt.path)
        if identity is None or not _identity_matches(path, identity):
            raise ValueError(f"bundle file identity changed for {receipt.path}")
        if not _hash_matches(path, receipt.sha256, receipt.bytes):
            raise ValueError(f"bundle hash reconciliation failed for {receipt.path}")
        if not _identity_matches(path, identity):
            raise ValueError(f"bundle file identity changed for {receipt.path}")
    _validate_staging_links(staging, page_paths)


def _final_reconcile_staging(
    staging: Path,
    staging_identity: _FileIdentity,
    data: DebugReportData,
    index_receipt: _PageReceipt,
    topic_receipts: Sequence[_PageReceipt],
    manifest_receipt: _PageReceipt,
    file_identities: Mapping[str, _FileIdentity],
) -> None:
    """Reconcile compact filesystem receipts after the expensive scan.

    Content hashes are intentionally not reread here.  The preceding link and
    streamed-hash phase produced the receipts; this final pass only confirms
    exact coverage, regular-file type, inode, and byte size immediately before
    descriptor-relative publication.
    """
    if not _identity_matches(staging, staging_identity, directory=True):
        raise ValueError("bundle staging payload identity changed")
    expected_files = {Path(index_receipt.path), *(Path(item.path) for item in topic_receipts), Path("bundle-manifest.json")}
    actual_files = {
        path.relative_to(staging)
        for path in staging.rglob("*")
        if not path.is_dir() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("bundle staging contains unlisted or missing files")
    topic_ids = tuple(topic.topic_id for topic in data.topics)
    if topic_ids != tuple(item.topic_id for item in topic_receipts):
        raise ValueError("bundle topic coverage or order does not reconcile")
    pages = (index_receipt, *topic_receipts)
    for receipt in pages:
        path = staging / Path(receipt.path)
        identity = file_identities.get(receipt.path)
        if identity is None or not _identity_matches(path, identity):
            raise ValueError(f"bundle file identity changed for {receipt.path}")
        record = path.lstat()
        if not stat.S_ISREG(record.st_mode) or record.st_size != receipt.bytes:
            raise ValueError(f"bundle file size/type reconciliation failed for {receipt.path}")
    manifest_identity = file_identities.get("bundle-manifest.json")
    manifest_path = staging / "bundle-manifest.json"
    if manifest_identity is None or not _identity_matches(manifest_path, manifest_identity):
        raise ValueError("bundle manifest file identity changed")
    manifest_record = manifest_path.lstat()
    if (
        not stat.S_ISREG(manifest_record.st_mode)
        or manifest_record.st_size != manifest_receipt.bytes
    ):
        raise ValueError("bundle manifest is not a regular file")


def _freeze_staging_tree(
    payload_fd: int,
    file_identities: Mapping[str, _FileIdentity],
    *,
    freeze_payload: bool,
) -> None:
    """Make all listed files and containing directories read-only by FD."""
    topics_fd = _open_directory_fd_at(payload_fd, "topics")
    try:
        topics_record = os.fstat(topics_fd)
        if not stat.S_ISDIR(topics_record.st_mode):
            raise ValueError("bundle topics directory is not a directory")
        for relative_path, expected_identity in sorted(file_identities.items()):
            parts = Path(relative_path).parts
            if len(parts) == 1:
                directory_fd = payload_fd
                name = parts[0]
            elif len(parts) == 2 and parts[0] == "topics":
                directory_fd = topics_fd
                name = parts[1]
            else:
                raise ValueError("bundle staging contains an unsafe file path")
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(f"bundle file identity changed for {relative_path}") from error
            try:
                record = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(record.st_mode)
                    or (record.st_dev, record.st_ino) != expected_identity
                ):
                    raise ValueError(f"bundle file identity changed for {relative_path}")
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
        os.fchmod(topics_fd, 0o500)
        if freeze_payload:
            os.fchmod(payload_fd, 0o500)
    finally:
        os.close(topics_fd)


@dataclass
class _StagingState:
    guardian_path: Path
    guardian_identity: _FileIdentity
    guardian_fd: int
    payload_path: Path
    payload_identity: _FileIdentity
    payload_fd: int
    topics_path: Path
    topics_identity: _FileIdentity
    file_identities: dict[str, _FileIdentity]
    payload_published: bool = False


def _unique_entry_name(parent_fd: int, prefix: str) -> str:
    for _attempt in range(100):
        name = f"{prefix}{uuid.uuid4().hex}"
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return name
    raise OSError("could not allocate a unique bundle quarantine name")


def _make_quarantine_directory(parent_fd: int) -> tuple[str, int, _FileIdentity]:
    for _attempt in range(100):
        name = _unique_entry_name(parent_fd, ".bundle-quarantine-")
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        descriptor = _open_directory_fd_at(parent_fd, name)
        record = os.fstat(descriptor)
        return name, descriptor, (record.st_dev, record.st_ino)
    raise OSError("could not create a bundle quarantine directory")


def _entry_identity_at(parent_fd: int, name: str) -> _FileIdentity:
    record = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    return (record.st_dev, record.st_ino)


def _unlink_quarantined_entry(
    quarantine_fd: int, name: str, expected_identity: _FileIdentity
) -> None:
    """Delete a quarantined regular file only if its inode is still owned."""
    record = os.stat(name, dir_fd=quarantine_fd, follow_symlinks=False)
    actual = (record.st_dev, record.st_ino)
    if stat.S_ISLNK(record.st_mode) or not stat.S_ISREG(record.st_mode):
        raise ValueError("bundle quarantine entry is no longer a regular file")
    if actual != expected_identity:
        raise ValueError("bundle quarantine file identity changed before deletion")
    os.unlink(name, dir_fd=quarantine_fd)


def _rmdir_quarantined_entry(
    quarantine_fd: int, name: str, expected_identity: _FileIdentity
) -> None:
    """Delete a quarantined empty directory only if its inode is still owned."""
    record = os.stat(name, dir_fd=quarantine_fd, follow_symlinks=False)
    actual = (record.st_dev, record.st_ino)
    if stat.S_ISLNK(record.st_mode) or not stat.S_ISDIR(record.st_mode):
        raise ValueError("bundle quarantine entry is no longer a directory")
    if actual != expected_identity:
        raise ValueError("bundle quarantine directory identity changed before deletion")
    os.rmdir(name, dir_fd=quarantine_fd)


def _restore_quarantined_entry(
    quarantine_fd: int,
    quarantine_name: str,
    original_fd: int,
    original_name: str,
) -> bool:
    """Restore a foreign quarantined entry without replacing an occupant."""
    try:
        identity = _entry_identity_at(quarantine_fd, quarantine_name)
    except OSError:
        return False
    try:
        _rename_cleanup_noreplace(
            quarantine_fd,
            quarantine_name,
            original_fd,
            original_name,
            expected_old_identity=identity,
        )
    except (OSError, ValueError):
        return False
    return True


def _remove_guardian_path(path: Path, identity: _FileIdentity) -> bool:
    """Quarantine and remove only an unchanged empty guardian pathname."""
    try:
        parent_fd = _open_directory_fd(path.parent)
    except OSError:
        return False
    quarantine_name: str | None = None
    try:
        if _entry_identity_at(parent_fd, path.name) != identity:
            return False
        quarantine_name = _unique_entry_name(parent_fd, ".bundle-guardian-")
        _rename_cleanup_noreplace(
            parent_fd,
            path.name,
            parent_fd,
            quarantine_name,
            expected_old_identity=identity,
        )
        try:
            _rmdir_quarantined_entry(parent_fd, quarantine_name, identity)
        except (OSError, ValueError):
            _restore_quarantined_entry(
                parent_fd,
                quarantine_name,
                parent_fd,
                path.name,
            )
            return False
        return True
    except (OSError, ValueError):
        if quarantine_name is not None:
            _restore_quarantined_entry(
                parent_fd,
                quarantine_name,
                parent_fd,
                path.name,
            )
        return False
    finally:
        os.close(parent_fd)


def _cleanup_staging(state: _StagingState | None, target: Path) -> None:
    if state is None:
        return
    guardian = state.guardian_path
    if guardian.parent.resolve() != target.parent.resolve():
        return
    if not guardian.name.startswith(f".{target.name}.") or not guardian.name.endswith(".tmp"):
        return
    if not _identity_matches(guardian, state.guardian_identity, directory=True):
        # The guardian pathname was moved or replaced.  Never follow it; the
        # pinned moved guardian is left for manual recovery.
        return
    if state.payload_published:
        _remove_guardian_path(guardian, state.guardian_identity)
        return
    if not _identity_matches(state.payload_path, state.payload_identity, directory=True):
        return
    if not _identity_matches(state.topics_path, state.topics_identity, directory=True):
        return
    try:
        payload_record = os.fstat(state.payload_fd)
        if (payload_record.st_dev, payload_record.st_ino) != state.payload_identity:
            return
        os.fchmod(state.payload_fd, 0o700)
        topics_fd = _open_directory_fd_at(state.payload_fd, "topics")
        try:
            topics_record = os.fstat(topics_fd)
            if (topics_record.st_dev, topics_record.st_ino) != state.topics_identity:
                return
            os.fchmod(topics_fd, 0o700)
        finally:
            os.close(topics_fd)
    except OSError:
        return
    quarantine_name: str | None = None
    quarantine_identity: _FileIdentity | None = None
    quarantine_fd: int | None = None
    payload_quarantine_name: str | None = None
    payload_quarantine_fd: int | None = None
    try:
        payload_quarantine_name = _unique_entry_name(state.guardian_fd, ".bundle-payload-")
        _rename_cleanup_noreplace(
            state.guardian_fd,
            "payload",
            state.guardian_fd,
            payload_quarantine_name,
            expected_old_identity=state.payload_identity,
        )
        payload_quarantine_fd = _open_directory_fd_at(
            state.guardian_fd, payload_quarantine_name
        )
        quarantine_name, quarantine_fd, quarantine_identity = _make_quarantine_directory(
            payload_quarantine_fd
        )
        topics_fd = _open_directory_fd_at(payload_quarantine_fd, "topics")
        try:
            for index, (relative_path, identity) in enumerate(
                sorted(state.file_identities.items())
            ):
                parts = Path(relative_path).parts
                if len(parts) == 1:
                    source_fd = payload_quarantine_fd
                elif len(parts) == 2 and parts[0] == "topics":
                    source_fd = topics_fd
                else:
                    return
                quarantine_file_name = f"{index:08d}-{parts[-1]}"
                _rename_cleanup_noreplace(
                    source_fd,
                    parts[-1],
                    quarantine_fd,
                    quarantine_file_name,
                    expected_old_identity=identity,
                )
            _rmdir_quarantined_entry(
                payload_quarantine_fd, "topics", state.topics_identity
            )
        finally:
            os.close(topics_fd)
        for index, (_relative_path, identity) in enumerate(
            sorted(state.file_identities.items())
        ):
            quarantine_file_name = f"{index:08d}-{Path(_relative_path).name}"
            try:
                _unlink_quarantined_entry(quarantine_fd, quarantine_file_name, identity)
            except (OSError, ValueError):
                _restore_quarantined_entry(
                    quarantine_fd,
                    quarantine_file_name,
                    payload_quarantine_fd,
                    Path(_relative_path).name,
                )
                return
        _rmdir_quarantined_entry(
            payload_quarantine_fd,
            quarantine_name,
            quarantine_identity,
        )
        _rmdir_quarantined_entry(
            state.guardian_fd,
            payload_quarantine_name,
            state.payload_identity,
        )
    except (OSError, ValueError):
        return
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        if payload_quarantine_fd is not None:
            os.close(payload_quarantine_fd)
    _remove_guardian_path(guardian, state.guardian_identity)


def _render_and_write_topic(
    topic: TopicReport,
    *,
    navigation: TopicPageNavigation,
    path: Path,
    relative_path: str,
    file_identities: dict[str, _FileIdentity],
) -> tuple[_PageReceipt, _FileIdentity]:
    rendered = render_debug_topic_page(topic, navigation=navigation)
    payload = rendered.encode("utf-8")
    del rendered
    written_identity = _write_bundle_file(path, payload)
    del payload
    if written_identity is None:
        written_identity = _lstat_identity(path)
    file_identities[relative_path] = written_identity
    return (
        _file_receipt(path, relative_path, topic_id=topic.topic_id),
        written_identity,
    )


def _validate_manifest_file(
    staging: Path,
    receipt: _PageReceipt,
    file_identities: Mapping[str, _FileIdentity],
) -> None:
    path = staging / Path(receipt.path)
    identity = file_identities.get(receipt.path)
    if identity is None or not _identity_matches(path, identity):
        raise ValueError("bundle manifest file identity changed")
    if not _hash_matches(path, receipt.sha256, receipt.bytes):
        raise ValueError("bundle manifest hash reconciliation failed")
    if not _identity_matches(path, identity):
        raise ValueError("bundle manifest file identity changed")


def _validate_published_receipts(
    target: Path,
    payload_identity: _FileIdentity,
    receipts: Sequence[_PageReceipt],
    file_identities: Mapping[str, _FileIdentity],
    *,
    topics_identity: _FileIdentity | None = None,
) -> None:
    receipt_by_path = {receipt.path: receipt for receipt in receipts}
    if len(receipt_by_path) != len(receipts):
        raise ValueError("published bundle receipts contain duplicate paths")
    if set(file_identities) != set(receipt_by_path):
        raise ValueError("published bundle file coverage does not reconcile")

    expected_files = {Path(relative_path) for relative_path in file_identities}
    if any(
        (not path.is_relative_to(Path(".")))
        or any(part in ("", ".", "..") for part in path.parts)
        or not (
            (len(path.parts) == 1 and path.name in {"index.html", "bundle-manifest.json"})
            or (len(path.parts) == 2 and path.parts[0] == "topics")
        )
        for path in expected_files
    ):
        raise ValueError("published bundle contains an unsafe file path")
    expected_topic_names = {
        path.name for path in expected_files if path.parts[:1] == ("topics",)
    }
    expected_direct_names = {
        path.name for path in expected_files if len(path.parts) == 1
    }
    if expected_direct_names != {"index.html", "bundle-manifest.json"}:
        raise ValueError("published bundle root file coverage does not reconcile")
    expected_root_names = {"index.html", "topics", "bundle-manifest.json"}

    def validate_coverage() -> None:
        if not _identity_matches(target, payload_identity, directory=True):
            raise ValueError("published bundle target identity changed before receipt")
        actual_root_names = {entry.name for entry in target.iterdir()}
        if actual_root_names != expected_root_names:
            raise ValueError("published bundle root coverage does not reconcile")
        topics_path = target / "topics"
        actual_topics_identity = _lstat_identity(topics_path, directory=True)
        if topics_identity is not None and actual_topics_identity != topics_identity:
            raise ValueError("published bundle topics directory identity changed")
        actual_topic_names = {entry.name for entry in topics_path.iterdir()}
        if actual_topic_names != expected_topic_names:
            raise ValueError("published bundle topic coverage does not reconcile")
        for name in ("index.html", "bundle-manifest.json"):
            _lstat_identity(target / name)
        for name in actual_topic_names:
            _lstat_identity(topics_path / name)

    def validate_file_records() -> None:
        for relative_path in sorted(file_identities):
            receipt = receipt_by_path[relative_path]
            path = target / Path(relative_path)
            expected_identity = file_identities[relative_path]
            try:
                record = path.lstat()
            except OSError as error:
                raise ValueError(
                    f"published bundle receipt path changed for {relative_path}"
                ) from error
            if (
                stat.S_ISLNK(record.st_mode)
                or not stat.S_ISREG(record.st_mode)
                or (record.st_dev, record.st_ino) != expected_identity
                or record.st_size != receipt.bytes
            ):
                raise ValueError(
                    f"published bundle receipt identity or size changed for {relative_path}"
                )

    # This sequential pass checks each receipt, then the fast second sweep
    # catches a replacement of an early entry made during the first pass.
    validate_coverage()
    validate_file_records()
    validate_coverage()
    validate_file_records()


def build_bundle_from_data(
    data: DebugReportData,
    *,
    output_dir: Path,
    evaluation_manifest_path: Path | None = None,
) -> DebugReportBundleReceipt:
    """Publish one validated run as a deterministic create-only HTML bundle."""
    target = _resolve_bundle_output(data, Path(output_dir))
    approved_roots = (
        find_repo_root(data.retrieval_config_path.parent).resolve(),
        data.output_dir.resolve(),
    )
    if not data.topics:
        raise ValueError("bundle requires at least one topic")
    topic_filenames = tuple(_safe_topic_filename(topic.topic_id) for topic in data.topics)
    if len(set(topic_filenames)) != len(topic_filenames):
        raise ValueError("bundle topic filenames are not unique")
    evaluation = (
        load_evaluation_overlay(Path(evaluation_manifest_path), tuple(topic.topic_id for topic in data.topics))
        if evaluation_manifest_path is not None
        else None
    )
    summary = build_run_summary(data, evaluation)
    publication_parent_fd: int | None = None
    target_parent_identity = _lstat_identity(target.parent, directory=True)
    state: _StagingState | None = None
    guardian: Path | None = None
    guardian_identity: _FileIdentity | None = None
    guardian_fd: int | None = None
    payload: Path | None = None
    payload_identity: _FileIdentity | None = None
    payload_fd: int | None = None
    ownership_token: object | None = None
    try:
        guardian = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
        )
        guardian_identity = _lstat_identity(guardian, directory=True)
        guardian.chmod(0o700)
        guardian_fd = _open_directory_fd(guardian)
        payload = guardian / "payload"
        payload.mkdir(mode=0o700)
        payload_identity = _lstat_identity(payload, directory=True)
        payload_fd = _open_directory_fd(payload)
        topics_dir = payload / "topics"
        topics_dir.mkdir(mode=0o700)
        topics_identity = _lstat_identity(topics_dir, directory=True)
        state = _StagingState(
            guardian_path=guardian,
            guardian_identity=guardian_identity,
            guardian_fd=guardian_fd,
            payload_path=payload,
            payload_identity=payload_identity,
            payload_fd=payload_fd,
            topics_path=topics_dir,
            topics_identity=topics_identity,
            file_identities={},
        )

        def register_owned_file(path: Path, identity: _FileIdentity) -> None:
            relative = path.relative_to(payload).as_posix()
            state.file_identities[relative] = identity

        ownership_token = _FILE_IDENTITY_REGISTRAR.set(register_owned_file)

        index_body = render_bundle_summary(summary, denylist=_summary_denylist(data)).encode("utf-8")
        index_path = payload / "index.html"
        index_written_identity = _write_bundle_file(index_path, index_body)
        del index_body
        if index_written_identity is None:
            index_written_identity = _lstat_identity(index_path)
        state.file_identities["index.html"] = index_written_identity
        index_receipt = _file_receipt(index_path, "index.html")

        topic_receipts: list[_PageReceipt] = []
        total_topics = len(data.topics)
        for position, (topic, filename) in enumerate(zip(data.topics, topic_filenames, strict=True)):
            navigation = TopicPageNavigation(
                summary_href="../index.html",
                position=position + 1,
                total=total_topics,
                previous_href=None if position == 0 else f"{topic_filenames[position - 1]}",
                next_href=None if position + 1 == total_topics else f"{topic_filenames[position + 1]}",
            )
            topic_path = topics_dir / filename
            topic_receipt, _topic_identity = _render_and_write_topic(
                topic,
                navigation=navigation,
                path=topic_path,
                relative_path=f"topics/{filename}",
                file_identities=state.file_identities,
            )
            topic_receipts.append(topic_receipt)

        _validate_staging(
            payload,
            payload_identity,
            data,
            index_receipt,
            topic_receipts,
            state.file_identities,
        )
        _freeze_staging_tree(
            state.payload_fd,
            state.file_identities,
            freeze_payload=False,
        )
        pages = (index_receipt, *topic_receipts)
        manifest: dict[str, Any] = {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "report_schema_version": _REPORT_SCHEMA_VERSION,
            "topic_ids": [topic.topic_id for topic in data.topics],
            "index": index_receipt.as_json(),
            "topics": [receipt.as_json() for receipt in topic_receipts],
            "sources": dict(sorted(data.source_sha256s.items())),
            "rag_included": data.rag_config_path is not None,
            "evaluation": (
                {
                    "included": True,
                    "schema_version": "trec_rag_offline_evaluation_bundle_v1",
                    "sha256": evaluation.manifest_sha256,
                }
                if evaluation is not None
                else {"included": False, "schema_version": None, "sha256": None}
            ),
            "page_count": len(pages),
            "total_bytes": 0,
        }
        manifest["total_bytes"] = sum(page.bytes for page in pages)
        while True:
            body = _canonical_json_bytes(manifest)
            total = sum(page.bytes for page in pages) + len(body)
            if manifest["total_bytes"] == total:
                break
            manifest["total_bytes"] = total
        manifest_path = payload / "bundle-manifest.json"
        manifest_bytes = len(body)
        manifest_sha256 = sha256(body).hexdigest()
        manifest_written_identity = _write_bundle_file(manifest_path, body)
        if manifest_written_identity is None:
            manifest_written_identity = _lstat_identity(manifest_path)
        state.file_identities["bundle-manifest.json"] = manifest_written_identity
        manifest_receipt = _file_receipt(manifest_path, "bundle-manifest.json")
        del body
        if (
            manifest_receipt.bytes != manifest_bytes
            or manifest_receipt.sha256 != manifest_sha256
        ):
            raise ValueError("bundle manifest hash reconciliation failed")
        _validate_manifest_file(payload, manifest_receipt, state.file_identities)
        _freeze_staging_tree(
            state.payload_fd,
            state.file_identities,
            # Linux checks write permission on the directory being moved when
            # renaming a directory.  The root is therefore frozen through its
            # still-pinned descriptor immediately after the atomic move; files
            # and the topics directory are already frozen for the scan.
            freeze_payload=False,
        )
        _fsync_directory(topics_dir)
        _fsync_directory(payload)
        _fsync_directory_fd(guardian_fd)
        _final_reconcile_staging(
            payload,
            payload_identity,
            data,
            index_receipt,
            topic_receipts,
            manifest_receipt,
            state.file_identities,
        )
        if target.exists() or target.is_symlink():
            raise ValueError("bundle output must remain absent before publication")
        if not _identity_matches(payload, payload_identity, directory=True):
            raise ValueError("bundle staging payload identity changed")
        publication_parent_fd = _rename_noreplace_fds(
            guardian_fd,
            "payload",
            -1,
            target.name,
            expected_old_identity=payload_identity,
            expected_file_identities=state.file_identities,
            expected_receipts=(*pages, manifest_receipt),
            expected_topic_ids=tuple(topic.topic_id for topic in data.topics),
            target_parent_path=target.parent,
            expected_target_parent_identity=target_parent_identity,
            approved_roots=approved_roots,
        )
        state.payload_published = True
        # Keep the published root read-only.  The descriptor still points at
        # the renamed inode, so this cannot be redirected by a path swap.
        os.fchmod(state.payload_fd, 0o500)
        _remove_guardian_path(guardian, guardian_identity)
        try:
            if publication_parent_fd is None:
                publication_parent_fd = _open_directory_fd(target.parent)
                parent_record = os.fstat(publication_parent_fd)
                if (
                    (parent_record.st_dev, parent_record.st_ino)
                    != target_parent_identity
                ):
                    raise ValueError("bundle target parent identity changed after publication")
            _fsync_directory_fd(publication_parent_fd)
        except Exception as error:
            raise OSError(
                "bundle published but target parent durability fsync failed"
            ) from error
        finally:
            if isinstance(publication_parent_fd, int):
                os.close(publication_parent_fd)
                publication_parent_fd = None
        _validate_published_receipts(
            target,
            payload_identity,
            (*pages, manifest_receipt),
            state.file_identities,
            topics_identity=state.topics_identity,
        )
        all_bytes = sum(page.bytes for page in pages) + manifest_receipt.bytes
        return DebugReportBundleReceipt(
            schema_version=_BUNDLE_SCHEMA_VERSION,
            report_schema_version=_REPORT_SCHEMA_VERSION,
            output_dir=target,
            index_path=target / "index.html",
            manifest_path=target / "bundle-manifest.json",
            topic_ids=tuple(topic.topic_id for topic in data.topics),
            bundle_manifest_sha256=manifest_receipt.sha256,
            page_count=len(pages),
            total_bytes=all_bytes,
            rag_included=data.rag_config_path is not None,
            evaluation_included=evaluation is not None,
        )
    finally:
        try:
            _cleanup_staging(state, target)
        finally:
            if ownership_token is not None:
                _FILE_IDENTITY_REGISTRAR.reset(ownership_token)
            if state is not None:
                os.close(state.payload_fd)
                os.close(state.guardian_fd)
            else:
                if payload_fd is not None:
                    os.close(payload_fd)
                if guardian_fd is not None:
                    os.close(guardian_fd)
                if (
                    guardian is not None
                    and guardian_identity is not None
                    and payload is not None
                    and payload_identity is not None
                    and _identity_matches(guardian, guardian_identity, directory=True)
                    and _identity_matches(payload, payload_identity, directory=True)
                ):
                    try:
                        payload.rmdir()
                    except OSError:
                        pass
                    _remove_guardian_path(guardian, guardian_identity)
            if isinstance(publication_parent_fd, int):
                os.close(publication_parent_fd)


__all__ = [
    "EvaluationOverlay",
    "MetricAvailability",
    "MetricFamilyOverlay",
    "RunSummary",
    "TopicSummary",
    "build_run_summary",
    "build_bundle_from_data",
    "load_evaluation_overlay",
    "render_bundle_summary",
]

"""Allowlisted summary and evaluation views for the competition debug bundle.

The raw debug-report records intentionally remain outside this module's summary
interface.  ``build_run_summary`` projects only the small set of counts and
links needed by the summary page, while ``load_evaluation_overlay`` adapts the
already validated offline-evaluation manifest into typed, read-only views.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trec_rag.competition_debug_report import DebugReportData, TopicReport
from trec_rag.friendly_report import build_presentation
from trec_rag.offline_evaluation import EvaluationError, load_manifest


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
    value: object, *, label: str, allowed_names: set[str] | None = None
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
            continue
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
    raw_per_topic = _mapping(raw.get("per_topic"), label=f"metrics.{key}.per_topic")
    raw_availability = _mapping(
        raw.get("per_topic_availability"),
        label=f"metrics.{key}.per_topic_availability",
    )

    per_topic: dict[str, Mapping[str, float]] = {}
    per_topic_availability: dict[str, MetricAvailability] = {}
    for topic_id in topic_ids:
        if topic_id not in raw_availability:
            raise EvaluationError(
                f"metrics.{key}.per_topic_availability is missing {topic_id}"
            )
        per_topic_availability[topic_id] = _availability(
            raw_availability[topic_id],
            label=f"metrics.{key}.per_topic_availability.{topic_id}",
        )
        if topic_id in raw_per_topic:
            per_topic[topic_id] = _numeric_mapping(
                raw_per_topic[topic_id],
                label=f"metrics.{key}.per_topic.{topic_id}",
                allowed_names=allowed_metric_names,
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
    manifest = load_manifest(path)
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


__all__ = [
    "EvaluationOverlay",
    "MetricAvailability",
    "MetricFamilyOverlay",
    "RunSummary",
    "TopicSummary",
    "build_run_summary",
    "load_evaluation_overlay",
]

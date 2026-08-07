"""Typed, year-neutral inputs for retrieval nugget coverage reports.

This module loads only the sealed handoff, validated decomposition checkpoints,
and completed coverage bundles needed by a report renderer.  It deliberately
does not inspect retrieval, passage, canonicalization, or provider artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from types import MappingProxyType

from trec_rag.competition_retrieval import load_validated_decomposition
from trec_rag.generation_handoff import (
    GenerationTopic,
    load_generation_handoff,
    select_generation_topics,
)
from trec_rag.retrieval_nugget_coverage import (
    CompletedCoverageEvaluation,
    load_completed_coverage_evaluation,
)
from trec_rag.topics import Topic


@dataclass(frozen=True, slots=True)
class RetrievalSubnarrativeContext:
    subnarrative_id: str
    text: str
    bm25_queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalPlanContext:
    used_fallback: bool
    subnarratives: tuple[RetrievalSubnarrativeContext, ...]
    planner_identity: Mapping[str, object]
    manifest_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.planner_identity, Mapping):
            raise TypeError("planner_identity must be a mapping")
        object.__setattr__(
            self,
            "planner_identity",
            MappingProxyType(dict(self.planner_identity)),
        )


@dataclass(frozen=True, slots=True)
class CoverageReportTopic:
    evaluation: CompletedCoverageEvaluation
    retrieval_plan: RetrievalPlanContext


@dataclass(frozen=True, slots=True)
class CoverageRunSummary:
    topic_count: int
    nugget_count: int
    required_obligation_count: int
    supplemental_obligation_count: int
    topic_macro_required_coverage: float
    topic_macro_strict_full_rate: float
    label_counts: Mapping[str, int]
    perfect_required_topic_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label_counts, Mapping):
            raise TypeError("label_counts must be a mapping")
        object.__setattr__(
            self,
            "label_counts",
            MappingProxyType(dict(self.label_counts)),
        )


@dataclass(frozen=True, slots=True)
class CoverageReportData:
    topics: tuple[CoverageReportTopic, ...]
    summary: CoverageRunSummary


_MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024
_DECOMPOSITION_MANIFEST_SCHEMA = "facet-decomposition-manifest-v1"
_DECOMPOSITION_MANIFEST_KEYS = frozenset(
    {"schema_version", "planner", "result_file", "result_bytes", "result_sha256"}
)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value}")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_canonical_json(path: Path) -> tuple[object, bytes]:
    """Read one small, regular, canonical JSON checkpoint file."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"checkpoint path is a symbolic link: {path}")
    try:
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {path}")
        if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError(f"checkpoint file is too large: {path}")
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint file is unreadable: {path}") from exc
    if len(source) > _MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint file is too large: {path}")
    try:
        payload = json.loads(
            source.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"checkpoint JSON is invalid: {path}") from exc
    try:
        canonical = _canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint JSON cannot be canonicalized: {path}") from exc
    if source != canonical:
        raise ValueError(f"checkpoint JSON is not canonical: {path}")
    return payload, source


def _safe_identity_value(value: object, *, depth: int = 0) -> object:
    """Return JSON identity data only when it is safe immutable data."""
    if depth > 32:
        raise ValueError("planner identity is too deeply nested")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("planner identity contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_safe_identity_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("planner identity keys must be text")
        return {
            key: _safe_identity_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    raise ValueError("planner identity contains unsupported data")


def _safe_child_directory(root: Path, name: str) -> Path:
    """Resolve one non-symlink directory beneath ``root``."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("topic path must be one safe directory name")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"retrieval output root is not a directory: {root}")
    path = root / name
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"topic directory is missing or unsafe: {path}")
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved.parent != root_resolved:
        raise ValueError(f"topic directory escapes retrieval output root: {path}")
    return path


def _load_retrieval_plan_context(
    *,
    handoff_manifest_path: Path,
    topic: GenerationTopic,
) -> RetrievalPlanContext:
    """Load and hash-check one topic's saved decomposition checkpoint."""
    if not hasattr(topic, "topic_id") or not hasattr(topic, "narrative"):
        raise TypeError("topic must expose topic_id and narrative")
    handoff_path = Path(handoff_manifest_path)
    if handoff_path.is_symlink() or not handoff_path.is_file():
        raise ValueError("handoff manifest path is missing or unsafe")
    output_root = handoff_path.parent
    topic_root = _safe_child_directory(output_root, topic.topic_id)
    decomposition_root = topic_root / "decomposition"
    if decomposition_root.is_symlink() or not decomposition_root.is_dir():
        raise ValueError("decomposition directory is missing or unsafe")
    manifest_path = decomposition_root / "manifest.json"
    result_path = decomposition_root / "result.json"
    manifest, manifest_bytes = _read_canonical_json(manifest_path)
    result, result_bytes = _read_canonical_json(result_path)
    if not isinstance(manifest, dict) or set(manifest) != _DECOMPOSITION_MANIFEST_KEYS:
        raise ValueError("decomposition checkpoint manifest fields are invalid")
    if manifest.get("schema_version") != _DECOMPOSITION_MANIFEST_SCHEMA:
        raise ValueError("decomposition checkpoint manifest schema is invalid")
    planner = manifest.get("planner")
    if not isinstance(planner, Mapping) or not planner:
        raise ValueError("decomposition checkpoint planner identity is invalid")
    try:
        planner_identity = _safe_identity_value(planner)
    except ValueError as exc:
        raise ValueError("decomposition checkpoint planner identity is invalid") from exc
    if not isinstance(planner_identity, dict) or not planner_identity:
        raise ValueError("decomposition checkpoint planner identity is invalid")
    if manifest.get("result_file") != result_path.name:
        raise ValueError("decomposition checkpoint result filename changed")
    if type(manifest.get("result_bytes")) is not int or manifest["result_bytes"] != len(result_bytes):
        raise ValueError("decomposition checkpoint result byte count changed")
    result_sha256 = manifest.get("result_sha256")
    if (
        not isinstance(result_sha256, str)
        or len(result_sha256) != 64
        or any(character not in "0123456789abcdef" for character in result_sha256)
        or result_sha256 != sha256(result_bytes).hexdigest()
    ):
        raise ValueError("decomposition checkpoint result hash changed")
    if not isinstance(result, dict):
        raise ValueError("decomposition result must be a JSON object")

    validated = load_validated_decomposition(
        Topic(topic.topic_id, "", topic.narrative),
        result_path,
    )
    subnarratives = tuple(
        RetrievalSubnarrativeContext(
            subnarrative.subnarrative_id,
            subnarrative.text,
            tuple(subnarrative.bm25_queries),
        )
        for subnarrative in validated.result.subnarratives
    )
    return RetrievalPlanContext(
        used_fallback=validated.result.used_fallback,
        subnarratives=subnarratives,
        planner_identity=planner_identity,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        result_sha256=sha256(result_bytes).hexdigest(),
    )


_COVERAGE_ARTIFACT_NAMES = frozenset(
    {"input.json", "plan.json", "judgments.json", "report.json", "manifest.json"}
)


def _discover_coverage_topics(
    coverage_root: Path,
    known_topic_ids: set[str],
) -> None:
    """Reject contradictory topic directories before any bundle is loaded."""
    root = Path(coverage_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("coverage root is missing or unsafe")
    for child in root.iterdir():
        if child.is_symlink():
            raise ValueError(f"coverage topic path is a symbolic link: {child}")
        if not child.is_dir():
            continue
        if child.name not in known_topic_ids:
            manifest = child / "manifest.json"
            if manifest.is_symlink() or manifest.exists():
                raise ValueError(f"unknown coverage topic directory: {child.name}")
            continue
        manifest = child / "manifest.json"
        if not manifest.exists() and any(
            (child / name).exists() or (child / name).is_symlink()
            for name in _COVERAGE_ARTIFACT_NAMES - {"manifest.json"}
        ):
            raise ValueError(f"incomplete coverage bundle for topic {child.name}")


def load_coverage_report_data(
    *,
    handoff_manifest_path: Path,
    coverage_root: Path,
    topic_ids: Sequence[str] = (),
) -> CoverageReportData:
    """Authenticate, discover, and load all selected coverage report inputs."""
    handoff_path = Path(handoff_manifest_path)
    handoff = load_generation_handoff(handoff_path)
    if topic_ids:
        selected_manifest_order = select_generation_topics(handoff, topic_ids)
        selected_by_id = {topic.topic_id: topic for topic in selected_manifest_order}
        selected_topics = tuple(selected_by_id[topic_id] for topic_id in topic_ids)
    else:
        selected_topics = select_generation_topics(handoff, None)
    if not selected_topics:
        raise ValueError("no topics were selected for coverage report")

    known_topic_ids = {topic.topic_id for topic in handoff.topics}
    _discover_coverage_topics(Path(coverage_root), known_topic_ids)
    root = Path(coverage_root)
    report_topics: list[CoverageReportTopic] = []
    for topic in selected_topics:
        topic_root = _safe_child_directory(root, topic.topic_id)
        manifest_path = topic_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"no complete coverage bundle for topic {topic.topic_id}")
        evaluation = load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id=topic.topic_id,
            work_dir=topic_root,
        )
        retrieval_plan = _load_retrieval_plan_context(
            handoff_manifest_path=handoff_path,
            topic=topic,
        )
        report_topics.append(
            CoverageReportTopic(
                evaluation=evaluation,
                retrieval_plan=retrieval_plan,
            )
        )
    frozen_topics = tuple(report_topics)
    return CoverageReportData(
        topics=frozen_topics,
        summary=summarize_coverage_topics(frozen_topics),
    )


def summarize_coverage_topics(
    topics: Sequence[CoverageReportTopic],
) -> CoverageRunSummary:
    """Aggregate completed topic evaluations into renderer-ready run metrics."""
    if isinstance(topics, (str, bytes)) or not isinstance(topics, Sequence):
        raise TypeError("topics must be a sequence")
    if not topics:
        raise ValueError("cannot summarize an empty topic sequence")

    label_counts: dict[str, int] = {
        "full": 0,
        "partial": 0,
        "unsupported": 0,
    }
    nugget_count = 0
    required_obligation_count = 0
    supplemental_obligation_count = 0
    required_coverages: list[float] = []
    strict_full_rates: list[float] = []
    perfect_required_topic_count = 0

    for topic in topics:
        evaluation = topic.evaluation
        plan = evaluation.plan
        report = evaluation.report
        nugget_count += len(evaluation.bound_input.nuggets)
        for obligation in plan.obligations:
            if obligation.kind == "required_explicit":
                required_obligation_count += 1
            elif obligation.kind == "supplemental_inferred":
                supplemental_obligation_count += 1
            else:
                raise ValueError(f"unknown coverage obligation kind: {obligation.kind!r}")
        for label in label_counts:
            value = report.label_counts.get(label, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid label count for {label!r}")
            label_counts[label] += value
        required_coverages.append(report.required_coverage)
        strict_full_rates.append(report.strict_full_rate)
        if report.required_coverage == 1.0:
            perfect_required_topic_count += 1

    return CoverageRunSummary(
        topic_count=len(topics),
        nugget_count=nugget_count,
        required_obligation_count=required_obligation_count,
        supplemental_obligation_count=supplemental_obligation_count,
        topic_macro_required_coverage=math.fsum(required_coverages) / len(topics),
        topic_macro_strict_full_rate=math.fsum(strict_full_rates) / len(topics),
        label_counts=label_counts,
        perfect_required_topic_count=perfect_required_topic_count,
    )


def _html_text(value: object) -> str:
    """Escape one value for a text node or quoted HTML attribute."""
    return escape(str(value), quote=True)


def _html_id(topic_id: object, index: int) -> str:
    """Build a deterministic, safe, topic-prefixed identifier."""
    text = str(topic_id)
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip("-") or "topic"
    digest = sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"topic-{index + 1}-{slug[:48]}-{digest}"


def _display_percentage(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _status_label(label: str) -> str:
    return {
        "full": "Full",
        "partial": "Partial",
        "unsupported": "Unsupported",
    }.get(label, label.replace("_", " ").title())


def _status_markup(label: str) -> str:
    icon = {"full": "✓", "partial": "◐", "unsupported": "×"}.get(label, "•")
    status_class = re.sub(r"[^a-z0-9-]", "-", str(label).lower())
    return (
        f'<span class="status status-{_html_text(status_class)}">'
        f'<span class="status-icon" aria-hidden="true">{icon}</span>'
        f'<span>{_html_text(_status_label(str(label)))}</span></span>'
    )


def _identity_rows(identity: Mapping[str, object]) -> str:
    rows: list[str] = []
    for key in sorted(identity, key=lambda item: str(item)):
        value = identity[key]
        if isinstance(value, Mapping):
            value_text = ", ".join(
                f"{_html_text(nested_key)}: {_html_text(nested_value)}"
                for nested_key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
            )
        elif isinstance(value, (list, tuple)):
            value_text = ", ".join(_html_text(item) for item in value)
        else:
            value_text = _html_text(value)
        rows.append(
            f'<div class="provenance-row"><dt>{_html_text(key)}</dt><dd>{value_text}</dd></div>'
        )
    return "".join(rows)


def _render_topic_detail(topic: CoverageReportTopic, *, index: int) -> str:
    evaluation = topic.evaluation
    bound_input = evaluation.bound_input
    plan = evaluation.plan
    report = evaluation.report
    topic_id = str(bound_input.topic_id)
    topic_key = _html_id(topic_id, index)
    narrative_id = f"{topic_key}-narrative"
    plan_context_id = f"{topic_key}-retrieval-plan"
    nugget_by_id = {
        nugget.nugget_id: (f"n{nugget_index:03d}", nugget.text)
        for nugget_index, nugget in enumerate(bound_input.nuggets, start=1)
    }
    judgment_by_id = {judgment.obligation_id: judgment for judgment in report.judgments}
    obligations_markup: list[str] = []
    first_gap_open = False
    for facet_index, facet in enumerate(plan.facets, start=1):
        facet_rows: list[str] = []
        for obligation_index, obligation in enumerate(facet.obligations, start=1):
            judgment = judgment_by_id.get(obligation.obligation_id)
            label = "unsupported" if judgment is None else judgment.label
            is_open = not first_gap_open and label in {"partial", "unsupported"}
            if is_open:
                first_gap_open = True
            summary_text = f"Obligation {facet_index}.{obligation_index}: {obligation.requirement}"
            spans = "".join(f"<li>{_html_text(span)}</li>" for span in obligation.narrative_spans)
            missing = "" if judgment is None else judgment.missing_elements
            missing_markup = _html_text(missing) if missing else "None"
            supporting_ids = () if judgment is None else judgment.supporting_nugget_ids
            support_rows: list[str] = []
            for nugget_id in supporting_ids:
                if nugget_id not in nugget_by_id:
                    continue
                alias, nugget_text = nugget_by_id[nugget_id]
                support_rows.append(
                    f'<li><code>{_html_text(alias)}</code><span>{_html_text(nugget_text)}</span></li>'
                )
            support_markup = "".join(support_rows) or "<li>None</li>"
            open_attribute = " open" if is_open else ""
            facet_rows.append(
                f'<details class="obligation"{open_attribute}>'
                f"<summary>{_html_text(summary_text)}</summary>"
                f'<dl class="obligation-grid">'
                f'<div><dt>Requirement</dt><dd>{_html_text(obligation.requirement)}</dd></div>'
                f'<div><dt>Kind</dt><dd><code>{_html_text(obligation.kind)}</code></dd></div>'
                f'<div><dt>Judgment</dt><dd>{_status_markup(label)}</dd></div>'
                f'<div><dt>Support test</dt><dd>{_html_text(obligation.support_test)}</dd></div>'
                f'<div><dt>Exact narrative spans</dt><dd><ul class="compact-list">{spans or "<li>None</li>"}</ul></dd></div>'
                f'<div><dt>Missing elements</dt><dd>{missing_markup}</dd></div>'
                f'<div class="supporting-nuggets"><dt>Supporting canonical nuggets</dt>'
                f'<dd><ul class="nugget-list">{support_markup}</ul></dd></div>'
                f"</dl></details>"
            )
        obligations_markup.append(
            f'<section class="facet" aria-labelledby="{topic_key}-facet-{facet_index}">'
            f'<h4 id="{topic_key}-facet-{facet_index}">{_html_text(facet.title)}</h4>'
            f'<p class="facet-id">Facet {_html_text(facet.facet_id)}</p>'
            f'{"".join(facet_rows)}</section>'
        )

    inventory_rows: list[str] = []
    uncited_ids = set(report.uncited_nugget_ids)
    uncited_aliases = set(report.uncited_nugget_aliases)
    for nugget_index, nugget in enumerate(bound_input.nuggets, start=1):
        alias = f"n{nugget_index:03d}"
        mapped = nugget.nugget_id not in uncited_ids and alias not in uncited_aliases
        badge = "Mapped" if mapped else "Unmapped"
        inventory_rows.append(
            f'<li><code>{alias}</code><span class="badge">{badge}</span>'
            f'<span class="nugget-text">{_html_text(nugget.text)}</span></li>'
        )
    identity = getattr(topic.retrieval_plan, "planner_identity", {})
    plan_rows: list[str] = []
    if topic.retrieval_plan.used_fallback:
        plan_rows.append("<p>Original narrative lane used; no generated subnarratives were saved.</p>")
    else:
        for sub_index, subnarrative in enumerate(topic.retrieval_plan.subnarratives, start=1):
            queries = "".join(
                f"<li>{_html_text(query)}</li>" for query in subnarrative.bm25_queries
            )
            plan_rows.append(
                f'<section class="subnarrative"><h3>Subnarrative {sub_index}: '
                f'{_html_text(subnarrative.text)}</h3>'
                f'<p class="subnarrative-id">{_html_text(subnarrative.subnarrative_id)}</p>'
                f'<h4>BM25 queries</h4><ol class="compact-list">{queries or "<li>None</li>"}</ol></section>'
            )
    artifact_rows = "".join(
        f'<div class="provenance-row"><dt>{_html_text(name)}</dt><dd><code>{_html_text(value)}</code></dd></div>'
        for name, value in sorted(getattr(evaluation, "artifact_hashes", {}).items(), key=lambda item: str(item[0]))
    )
    evaluator_identity = getattr(evaluation, "identity", report.identity)
    evaluator_rows = "".join(
        f'<div class="provenance-row"><dt>{_html_text(name)}</dt><dd>{_html_text(value)}</dd></div>'
        for name, value in (
            ("schema version", evaluator_identity.schema_version),
            ("planner prompt", evaluator_identity.planner_prompt_version),
            ("judge prompt", evaluator_identity.judge_prompt_version),
            ("planner model", evaluator_identity.planner_model),
            ("judge model", evaluator_identity.judge_model),
        )
    )
    unmapped = "".join(
        f"<li>{_html_text(span)}</li>" for span in report.unmapped_narrative_spans
    )
    return (
        f'<article class="topic-detail" id="{topic_key}-detail" data-topic-id="{_html_text(topic_id)}" hidden>'
        f'<header class="detail-header"><p class="eyebrow">Topic detail</p>'
        f'<h2 id="{topic_key}-title">{_html_text(topic_id)}</h2>'
        f'<dl class="detail-metrics">'
        f'<div><dt>Required coverage</dt><dd>{_display_percentage(report.required_coverage)}</dd></div>'
        f'<div><dt>Strict-full rate</dt><dd>{_display_percentage(report.strict_full_rate)}</dd></div>'
        f'<div><dt>Full</dt><dd>{_html_text(report.label_counts.get("full", 0))}</dd></div>'
        f'<div><dt>Partial</dt><dd>{_html_text(report.label_counts.get("partial", 0))}</dd></div>'
        f'<div><dt>Unsupported</dt><dd>{_html_text(report.label_counts.get("unsupported", 0))}</dd></div>'
        f'</dl></header>'
        f'<details class="narrative-disclosure" id="{narrative_id}"><summary>Exact authenticated narrative</summary>'
        f'<p class="narrative">{_html_text(bound_input.narrative)}</p></details>'
        f'<details class="retrieval-plan" id="{plan_context_id}"><summary>Retrieval plan context — not coverage evidence</summary>'
        f'<p class="notice">Subnarratives and BM25 queries are context only. Canonical nugget text below is the evaluator evidence representation.</p>'
        f'{"".join(plan_rows) or "<p>No generated subnarratives.</p>"}'
        f'<dl class="provenance">{_identity_rows(identity)}</dl></details>'
        f'<section class="obligations" aria-labelledby="{topic_key}-obligations-title"><h3 id="{topic_key}-obligations-title">Answer obligations</h3>'
        f'{"".join(obligations_markup) or "<p>No obligations.</p>"}'
        f'<section class="unmapped"><h4>Unmapped narrative spans</h4><ul class="compact-list">{unmapped or "<li>None</li>"}</ul></section></section>'
        f'<details class="nugget-inventory"><summary>Complete canonical nugget inventory</summary>'
        f'<p>Local aliases are used here so the report does not expose internal source identifiers.</p>'
        f'<ol class="nugget-list inventory-list">{"".join(inventory_rows) or "<li>None</li>"}</ol></details>'
        f'<details class="provenance-disclosure"><summary>Provenance hashes and evaluator identity</summary>'
        f'<dl class="provenance">'
        f'<div class="provenance-row"><dt>Coverage manifest SHA-256</dt><dd><code>{_html_text(evaluation.manifest_sha256)}</code></dd></div>'
        f'<div class="provenance-row"><dt>Coverage plan SHA-256</dt><dd><code>{_html_text(report.plan_sha256)}</code></dd></div>'
        f'<div class="provenance-row"><dt>Authenticated input SHA-256</dt><dd><code>{_html_text(getattr(bound_input, "input_sha256", ""))}</code></dd></div>'
        f'<div class="provenance-row"><dt>Authenticated narrative SHA-256</dt><dd><code>{_html_text(bound_input.narrative_sha256)}</code></dd></div>'
        f'<div class="provenance-row"><dt>Authenticated handoff manifest SHA-256</dt><dd><code>{_html_text(bound_input.manifest_sha256)}</code></dd></div>'
        f'<div class="provenance-row"><dt>Authenticated nugget text SHA-256 values</dt><dd><code>{_html_text(", ".join(bound_input.nugget_text_sha256s))}</code></dd></div>'
        f'<div class="provenance-row"><dt>Decomposition manifest SHA-256</dt><dd><code>{_html_text(topic.retrieval_plan.manifest_sha256)}</code></dd></div>'
        f'<div class="provenance-row"><dt>Decomposition result SHA-256</dt><dd><code>{_html_text(topic.retrieval_plan.result_sha256)}</code></dd></div>'
        f'{artifact_rows}{evaluator_rows}</dl></details></article>'
    )


def render_coverage_report_html(data: CoverageReportData) -> bytes:
    """Render deterministic, standalone UTF-8 HTML for validated report data."""
    if not isinstance(data, CoverageReportData):
        raise TypeError("data must be CoverageReportData")
    indexed_topics = sorted(
        enumerate(data.topics),
        key=lambda item: (
            float(item[1].evaluation.report.required_coverage),
            str(item[1].evaluation.bound_input.topic_id),
        ),
    )
    topic_rows: list[str] = []
    detail_rows: list[str] = []
    for index, topic in indexed_topics:
        evaluation = topic.evaluation
        bound_input = evaluation.bound_input
        report = evaluation.report
        topic_id = str(bound_input.topic_id)
        topic_key = _html_id(topic_id, index)
        label_counts = report.label_counts
        has_gaps = bool(label_counts.get("partial", 0) or label_counts.get("unsupported", 0))
        perfect = float(report.required_coverage) == 1.0
        unsupported = bool(label_counts.get("unsupported", 0))
        topic_text = " ".join(
            [
                topic_id,
                str(bound_input.narrative),
                *[str(nugget.text) for nugget in bound_input.nuggets],
            ]
        )
        topic_rows.append(
            f'<li class="topic-row" data-topic-id="{_html_text(topic_id)}" '
            f'data-required-coverage="{float(report.required_coverage):.12g}" '
            f'data-strict-full-rate="{float(report.strict_full_rate):.12g}" '
            f'data-has-gaps="{"true" if has_gaps else "false"}" '
            f'data-perfect="{"true" if perfect else "false"}" '
            f'data-unsupported="{"true" if unsupported else "false"}" '
            f'data-search="{_html_text(topic_text.lower())}">'
            f'<button type="button" class="topic-button" data-topic-id="{_html_text(topic_id)}" '
            f'aria-controls="{topic_key}-detail" aria-pressed="false">'
            f'<span class="topic-button-title">{_html_text(topic_id)}</span>'
            f'<span class="topic-button-metrics"><span>Required {_display_percentage(report.required_coverage)}</span>'
            f'<span>Strict full {_display_percentage(report.strict_full_rate)}</span></span>'
            f'<span class="topic-button-status">{_status_markup("unsupported" if unsupported else ("partial" if has_gaps else "full"))}</span>'
            f'</button></li>'
        )
        detail_rows.append(_render_topic_detail(topic, index=index))

    summary = data.summary
    limitations = (
        "Coverage evidence is limited to canonical nugget text cited by each judgment.",
        "Retrieval-plan subnarratives and BM25 queries are context, not coverage evidence.",
        "Canonical nuggets are claim hints; this report does not rejudge their factual faithfulness.",
        "Required coverage is a facet-macro average over required obligations.",
        "Strict-full rate is an obligation-micro share over required obligations.",
        "Label counts include required and supplemental obligations.",
        "This derivative excludes passages, document IDs, scores, provider responses, and secrets.",
    )
    limitation_markup = "".join(f"<li>{_html_text(item)}</li>" for item in limitations)
    label_counts = summary.label_counts
    html = f'''<!doctype html>
<html lang="en" data-theme="system">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieval nugget coverage report</title>
<script>
(function () {{
  try {{
    var saved = window.localStorage.getItem("retrieval-nugget-coverage-theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  }} catch (error) {{}}
}}());
</script>
<style>
:root {{
  --color-bg: #f4f7fb;
  --color-surface: #ffffff;
  --color-surface-muted: #eaf0f7;
  --color-surface-accent: #e5f3f0;
  --color-text: #17212b;
  --color-text-muted: #536273;
  --color-border: #c7d2df;
  --color-accent: #176b67;
  --color-accent-strong: #0e4f4b;
  --color-focus: #b54708;
  --color-full: #176b67;
  --color-partial: #a15c00;
  --color-unsupported: #a83246;
  --color-shadow: rgba(23, 33, 43, 0.12);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --color-bg: #111820;
    --color-surface: #1b2732;
    --color-surface-muted: #243440;
    --color-surface-accent: #173b3a;
    --color-text: #eff5f8;
    --color-text-muted: #b8c8d4;
    --color-border: #405360;
    --color-accent: #70d1c5;
    --color-accent-strong: #9be7dc;
    --color-focus: #f4b183;
    --color-full: #70d1c5;
    --color-partial: #f1b45e;
    --color-unsupported: #ff8ea0;
    --color-shadow: rgba(0, 0, 0, 0.34);
  }}
}}
[data-theme="light"] {{
  --color-bg: #f4f7fb;
  --color-surface: #ffffff;
  --color-surface-muted: #eaf0f7;
  --color-surface-accent: #e5f3f0;
  --color-text: #17212b;
  --color-text-muted: #536273;
  --color-border: #c7d2df;
  --color-accent: #176b67;
  --color-accent-strong: #0e4f4b;
  --color-focus: #b54708;
  --color-full: #176b67;
  --color-partial: #a15c00;
  --color-unsupported: #a83246;
  --color-shadow: rgba(23, 33, 43, 0.12);
}}
[data-theme="dark"] {{
  --color-bg: #111820;
  --color-surface: #1b2732;
  --color-surface-muted: #243440;
  --color-surface-accent: #173b3a;
  --color-text: #eff5f8;
  --color-text-muted: #b8c8d4;
  --color-border: #405360;
  --color-accent: #70d1c5;
  --color-accent-strong: #9be7dc;
  --color-focus: #f4b183;
  --color-full: #70d1c5;
  --color-partial: #f1b45e;
  --color-unsupported: #ff8ea0;
  --color-shadow: rgba(0, 0, 0, 0.34);
}}
* {{ box-sizing: border-box; }}
html {{ background: var(--color-bg); color: var(--color-text); scroll-behavior: smooth; }}
body {{ margin: 0; background: var(--color-bg); color: var(--color-text); font: 16px/1.55 system-ui, sans-serif; }}
a {{ color: var(--color-accent-strong); }}
button, input, select {{ font: inherit; }}
button, select, input {{ color: var(--color-text); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 0.45rem; }}
button {{ cursor: pointer; }}
button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible, a:focus-visible {{ outline: 3px solid var(--color-focus); outline-offset: 3px; }}
.skip-link {{ position: absolute; left: 1rem; top: 0.5rem; transform: translateY(-150%); padding: 0.5rem 0.75rem; background: var(--color-surface); color: var(--color-text); z-index: 3; }}
.skip-link:focus {{ transform: translateY(0); }}
.page-shell {{ width: 100%; max-width: 72rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
.eyebrow {{ color: var(--color-accent-strong); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
h1, h2, h3, h4, h5 {{ line-height: 1.2; }}
h1 {{ max-width: 20ch; font-size: clamp(2rem, 5vw, 4rem); margin: 0.2rem 0 1rem; }}
h2 {{ margin-top: 0; }}
.lede {{ max-width: 62ch; color: var(--color-text-muted); font-size: 1.1rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr)); gap: 0.75rem; margin: 2rem 0; }}
.metric-card, .callout, .topic-row, .topic-detail, details {{ background: var(--color-surface); border: 1px solid var(--color-border); box-shadow: 0 0.3rem 1rem var(--color-shadow); }}
.metric-card {{ border-radius: 0.65rem; padding: 1rem; }}
.metric-card dt, .detail-metrics dt, .obligation-grid dt, .provenance dt {{ color: var(--color-text-muted); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
.metric-card dd, .detail-metrics dd {{ margin: 0.15rem 0 0; font-size: 1.45rem; font-weight: 750; }}
.callout {{ border-left: 0.35rem solid var(--color-accent); border-radius: 0.5rem; padding: 1rem 1.2rem; margin: 1rem 0 2rem; }}
.callout h2 {{ font-size: 1.1rem; margin: 0; }}
.controls {{ display: grid; grid-template-columns: minmax(12rem, 2fr) repeat(2, minmax(10rem, 1fr)); gap: 0.75rem; align-items: end; margin: 1.5rem 0 0.75rem; }}
.control {{ display: grid; gap: 0.3rem; }}
.control label {{ color: var(--color-text-muted); font-size: 0.85rem; font-weight: 700; }}
.control input, .control select {{ min-height: 2.7rem; padding: 0.5rem 0.65rem; width: 100%; }}
.theme-switcher {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 1rem 0; }}
.theme-switcher button[aria-pressed="true"] {{ border-color: var(--color-accent); box-shadow: inset 0 -0.2rem 0 var(--color-accent); }}
.topic-list {{ display: grid; gap: 0.65rem; list-style: none; padding: 0; margin: 0; }}
.topic-row {{ border-radius: 0.55rem; overflow: hidden; }}
.topic-button {{ display: grid; grid-template-columns: minmax(9rem, 1.5fr) repeat(2, minmax(8rem, 1fr)); align-items: center; gap: 0.75rem; width: 100%; border: 0; border-radius: 0; padding: 0.9rem 1rem; text-align: left; background: transparent; }}
.topic-button:hover {{ background: var(--color-surface-muted); }}
.topic-button-title {{ font-weight: 750; overflow-wrap: anywhere; }}
.topic-button-metrics {{ display: grid; color: var(--color-text-muted); font-size: 0.9rem; }}
.topic-button-status {{ justify-self: end; }}
.status {{ display: inline-flex; align-items: center; gap: 0.35rem; font-weight: 700; white-space: nowrap; }}
.status-icon {{ display: inline-grid; place-items: center; width: 1.25rem; height: 1.25rem; border: 2px solid currentColor; border-radius: 50%; line-height: 1; }}
.status-full {{ color: var(--color-full); }}
.status-partial {{ color: var(--color-partial); }}
.status-unsupported {{ color: var(--color-unsupported); }}
.filter-status {{ min-height: 1.6rem; color: var(--color-text-muted); }}
.detail-view[hidden], .topic-detail[hidden] {{ display: none; }}
.back-link {{ margin: 1rem 0; padding: 0.5rem 0.75rem; }}
.detail-header {{ border-bottom: 1px solid var(--color-border); margin-bottom: 1rem; }}
.detail-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr)); gap: 0.5rem; margin: 1rem 0; }}
.detail-metrics > div {{ padding: 0.65rem; background: var(--color-surface-muted); border-radius: 0.4rem; }}
details {{ border-radius: 0.55rem; margin: 0.8rem 0; overflow: hidden; box-shadow: none; }}
details > summary {{ cursor: pointer; padding: 0.85rem 1rem; font-weight: 750; }}
details > summary:hover {{ background: var(--color-surface-muted); }}
details > :not(summary) {{ padding-left: 1rem; padding-right: 1rem; }}
.narrative-disclosure, .retrieval-plan, .nugget-inventory, .provenance-disclosure {{ border-left: 0.35rem solid var(--color-accent); }}
.narrative {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
.notice {{ color: var(--color-text-muted); }}
.subnarrative {{ border-top: 1px solid var(--color-border); padding: 0.5rem 0 0.75rem; }}
.subnarrative h3 {{ margin-bottom: 0.2rem; }}
.subnarrative-id, .facet-id {{ color: var(--color-text-muted); font-size: 0.85rem; }}
.compact-list {{ margin-top: 0.3rem; padding-left: 1.25rem; }}
.obligations > h3 {{ margin-top: 2rem; }}
.facet {{ margin: 1.4rem 0; }}
.facet h4 {{ margin-bottom: 0; }}
.obligation {{ background: var(--color-surface-muted); }}
.obligation-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr)); gap: 0.8rem; margin: 0; padding-bottom: 1rem; }}
.obligation-grid > div {{ min-width: 0; }}
.obligation-grid dd {{ margin: 0.2rem 0 0; overflow-wrap: anywhere; }}
.obligation-grid .supporting-nuggets {{ grid-column: 1 / -1; }}
.nugget-list {{ list-style: none; padding: 0; margin: 0.3rem 0 0; }}
.nugget-list li {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: baseline; padding: 0.35rem 0; border-bottom: 1px solid var(--color-border); }}
.nugget-text {{ flex: 1 1 20rem; overflow-wrap: anywhere; }}
code {{ color: var(--color-accent-strong); overflow-wrap: anywhere; }}
.badge {{ border: 1px solid var(--color-border); border-radius: 999px; color: var(--color-text-muted); font-size: 0.75rem; padding: 0.1rem 0.4rem; }}
.unmapped {{ margin: 1rem 0; padding: 0.75rem 1rem; background: var(--color-surface-accent); border-radius: 0.4rem; }}
.provenance {{ margin: 0.5rem 0 1rem; }}
.provenance-row {{ display: grid; grid-template-columns: minmax(10rem, 0.7fr) minmax(0, 1.6fr); gap: 0.75rem; padding: 0.35rem 0; border-bottom: 1px solid var(--color-border); }}
.provenance dd {{ margin: 0; overflow-wrap: anywhere; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }}
@media (max-width: 46rem) {{
  .page-shell {{ padding-left: 0.8rem; padding-right: 0.8rem; }}
  .controls {{ grid-template-columns: 1fr; }}
  .topic-button {{ grid-template-columns: 1fr; }}
  .topic-button-status {{ justify-self: start; }}
  .provenance-row {{ grid-template-columns: 1fr; gap: 0.1rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0.01ms !important; }}
}}
@media print {{
  :root {{
    --color-bg: #ffffff; --color-surface: #ffffff; --color-surface-muted: #f0f0f0; --color-surface-accent: #f0f0f0;
    --color-text: #000000; --color-text-muted: #333333; --color-border: #999999; --color-accent: #000000; --color-accent-strong: #000000;
    --color-focus: #000000; --color-full: #000000; --color-partial: #000000; --color-unsupported: #000000; --color-shadow: transparent;
  }}
  .skip-link, .controls, .theme-switcher, .back-link {{ display: none !important; }}
  #overview-view:not([hidden]), .detail-view:not([hidden]) {{ display: block !important; }}
  details:not([open]) > :not(summary) {{ display: block !important; }}
  details {{ break-inside: avoid; box-shadow: none; }}
}}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
<div class="page-shell">
<header>
<p class="eyebrow">Private diagnostic derivative</p>
<h1>Retrieval nugget coverage</h1>
<p class="lede">A read-only view of authenticated narratives, evaluator obligations, canonical nugget support, and hash-checked retrieval-plan context.</p>
<div class="theme-switcher" aria-label="Theme selection">
<span>Theme:</span><button type="button" data-theme-choice="light" aria-pressed="false">Light</button>
<button type="button" data-theme-choice="system" aria-pressed="true">System</button>
<button type="button" data-theme-choice="dark" aria-pressed="false">Dark</button>
</div>
</header>
<main id="main-content">
<section id="overview-view" aria-labelledby="overview-title">
<h2 id="overview-title">Run overview</h2>
<dl class="cards">
<div class="metric-card"><dt>Evaluated topics</dt><dd>{_html_text(summary.topic_count)}</dd></div>
<div class="metric-card"><dt>Canonical nuggets</dt><dd>{_html_text(summary.nugget_count)}</dd></div>
<div class="metric-card"><dt>Required obligations</dt><dd>{_html_text(summary.required_obligation_count)}</dd></div>
<div class="metric-card"><dt>Supplemental obligations</dt><dd>{_html_text(summary.supplemental_obligation_count)}</dd></div>
<div class="metric-card"><dt>Topic-macro required coverage</dt><dd>{_display_percentage(summary.topic_macro_required_coverage)}</dd></div>
<div class="metric-card"><dt>Topic-macro strict-full rate</dt><dd>{_display_percentage(summary.topic_macro_strict_full_rate)}</dd></div>
<div class="metric-card"><dt>Full / partial / unsupported</dt><dd>{_html_text(label_counts.get("full", 0))} / {_html_text(label_counts.get("partial", 0))} / {_html_text(label_counts.get("unsupported", 0))}</dd></div>
<div class="metric-card"><dt>Perfect required coverage</dt><dd>{_html_text(summary.perfect_required_topic_count)}</dd></div>
</dl>
<aside class="callout" aria-labelledby="limitations-title"><h2 id="limitations-title">Limitations and interpretation boundary</h2><ul>{limitation_markup}</ul></aside>
<div class="controls" role="search" aria-label="Filter coverage topics">
<div class="control"><label for="topic-search">Search topics</label><input id="topic-search" type="search" autocomplete="off" placeholder="Topic ID or narrative"></div>
<div class="control"><label for="status-filter">Status</label><select id="status-filter"><option value="all">All topics</option><option value="has-gaps">Has gaps</option><option value="perfect">Perfect required coverage</option><option value="unsupported">Has unsupported obligations</option></select></div>
<div class="control"><label for="sort-topics">Sort</label><select id="sort-topics"><option value="required">Required coverage</option><option value="strict">Strict-full rate</option><option value="topic">Topic ID</option></select></div>
</div>
<p id="filter-status" class="filter-status" role="status" aria-live="polite"></p>
<ol id="topic-list" class="topic-list">{"".join(topic_rows)}</ol>
</section>
<section id="detail-view" class="detail-view" aria-labelledby="detail-view-title" hidden>
<button type="button" class="back-link" id="back-to-overview">← Back to overview</button>
<h2 id="detail-view-title" class="sr-only">Topic detail</h2>
{"".join(detail_rows)}
</section>
<p id="live-region" class="sr-only" role="status" aria-live="polite"></p>
</main>
</div>
<script>
(function () {{
  "use strict";
  var root = document.documentElement;
  var overview = document.getElementById("overview-view");
  var detailView = document.getElementById("detail-view");
  var topicList = document.getElementById("topic-list");
  var topicRows = Array.prototype.slice.call(document.querySelectorAll(".topic-row"));
  var topicButtons = Array.prototype.slice.call(document.querySelectorAll(".topic-button"));
  var topicDetails = Array.prototype.slice.call(document.querySelectorAll(".topic-detail"));
  var search = document.getElementById("topic-search");
  var statusFilter = document.getElementById("status-filter");
  var sortTopics = document.getElementById("sort-topics");
  var filterStatus = document.getElementById("filter-status");
  var liveRegion = document.getElementById("live-region");
  var themeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-theme-choice]"));
  var backButton = document.getElementById("back-to-overview");
  var returnFocusButton = null;

  function setTheme(choice, persist) {{
    if (choice === "light" || choice === "dark") root.setAttribute("data-theme", choice);
    else root.setAttribute("data-theme", "system");
    themeButtons.forEach(function (button) {{ button.setAttribute("aria-pressed", String(button.getAttribute("data-theme-choice") === choice)); }});
    if (persist) {{ try {{ if (choice === "system") window.localStorage.removeItem("retrieval-nugget-coverage-theme"); else window.localStorage.setItem("retrieval-nugget-coverage-theme", choice); }} catch (error) {{}} }}
  }}
  themeButtons.forEach(function (button) {{ button.addEventListener("click", function () {{ setTheme(button.getAttribute("data-theme-choice"), true); }}); }});
  try {{ var savedTheme = window.localStorage.getItem("retrieval-nugget-coverage-theme"); setTheme(savedTheme === "light" || savedTheme === "dark" ? savedTheme : "system", false); }} catch (error) {{ setTheme("system", false); }}

  function detailFor(topicId) {{ return topicDetails.find(function (detail) {{ return detail.getAttribute("data-topic-id") === topicId; }}); }}
  function showOverview(message, restoreFocus) {{
    overview.hidden = false;
    detailView.hidden = true;
    topicDetails.forEach(function (detail) {{ detail.hidden = true; }});
    topicButtons.forEach(function (button) {{ button.setAttribute("aria-pressed", "false"); }});
    if (message) liveRegion.textContent = message;
    var focusTarget = restoreFocus ? returnFocusButton : null;
    returnFocusButton = null;
    if (focusTarget && document.contains(focusTarget) && !focusTarget.hidden) focusTarget.focus();
  }}
  function showTopic(topicId, push) {{
    var selected = detailFor(topicId);
    if (!selected) {{ showOverview("That topic was not found; showing the overview."); return; }}
    returnFocusButton = topicButtons.find(function (button) {{ return button.getAttribute("data-topic-id") === topicId; }}) || null;
    overview.hidden = true;
    detailView.hidden = false;
    topicDetails.forEach(function (detail) {{ detail.hidden = detail !== selected; }});
    topicButtons.forEach(function (button) {{ button.setAttribute("aria-pressed", String(button.getAttribute("data-topic-id") === topicId)); }});
    if (push) history.pushState(null, "", "#topic=" + encodeURIComponent(topicId));
    liveRegion.textContent = "Showing topic " + topicId + ".";
    backButton.focus();
    selected.scrollIntoView({{ block: "start" }});
  }}
  function restoreLocation() {{
    if (!location.hash) {{ showOverview("", true); return; }}
    if (location.hash.indexOf("#topic=") !== 0) {{ showOverview("That link was not a valid topic; showing the overview.", true); return; }}
    var encoded = location.hash.slice(7);
    var topicId;
    try {{ topicId = decodeURIComponent(encoded); }} catch (error) {{ topicId = ""; }}
    if (!topicId || !detailFor(topicId)) showOverview("That topic was not found; showing the overview.", true);
    else showTopic(topicId, false);
  }}
  topicButtons.forEach(function (button) {{ button.addEventListener("click", function () {{ showTopic(button.getAttribute("data-topic-id"), true); }}); }});
  backButton.addEventListener("click", function () {{ history.pushState(null, "", location.pathname + location.search); showOverview("", true); }});
  window.addEventListener("popstate", restoreLocation);

  function reorderTopics() {{
    var mode = sortTopics.value;
    topicRows.slice().sort(function (left, right) {{
      var a = mode === "topic" ? left.getAttribute("data-topic-id") : Number(left.getAttribute(mode === "strict" ? "data-strict-full-rate" : "data-required-coverage"));
      var b = mode === "topic" ? right.getAttribute("data-topic-id") : Number(right.getAttribute(mode === "strict" ? "data-strict-full-rate" : "data-required-coverage"));
      if (a < b) return -1; if (a > b) return 1; return left.getAttribute("data-topic-id").localeCompare(right.getAttribute("data-topic-id"));
    }}).forEach(function (row) {{ topicList.appendChild(row); }});
  }}
  function applyFilters() {{
    var query = search.value.trim().toLowerCase();
    var mode = statusFilter.value;
    var visible = 0;
    topicRows.forEach(function (row) {{
      var statusMatch = mode === "all" || (mode === "has-gaps" && row.getAttribute("data-has-gaps") === "true") || (mode === "perfect" && row.getAttribute("data-perfect") === "true") || (mode === "unsupported" && row.getAttribute("data-unsupported") === "true");
      var queryMatch = !query || row.getAttribute("data-search").indexOf(query) !== -1;
      row.hidden = !(statusMatch && queryMatch);
      if (!row.hidden) visible += 1;
    }});
    filterStatus.textContent = visible + " of " + topicRows.length + " topics shown.";
  }}
  search.addEventListener("input", applyFilters);
  statusFilter.addEventListener("change", applyFilters);
  sortTopics.addEventListener("change", function () {{ reorderTopics(); applyFilters(); }});
  reorderTopics();
  applyFilters();
  restoreLocation();
}}());
</script>
</body>
</html>
'''
    return html.encode("utf-8")


class _CliError(Exception):
    """An expected command-line failure rendered as structured stderr JSON."""


class _ParserExit(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _SafeArgumentParser(argparse.ArgumentParser):
    """Argparse parser that never raises a user-facing ``SystemExit`` error."""

    def error(self, message: str) -> None:
        raise _CliError(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, sys.stdout if status == 0 else sys.stderr)
        raise _ParserExit(status)


def _coverage_report_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Render a retrieval nugget coverage report without hosted calls."
    )
    parser.add_argument("--handoff-manifest", required=True, type=Path)
    parser.add_argument("--coverage-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", action="append", default=[], dest="topic_ids")
    return parser


def _safe_cli_error(stage: str) -> dict[str, object]:
    reason = {
        "config": "report configuration is invalid",
        "load": "coverage report inputs are invalid",
        "render": "coverage report rendering failed",
        "publish": "coverage report publication failed",
    }.get(stage, "coverage report failed")
    return {
        "status": "error",
        "error": {
            "type": "retrieval_nugget_coverage_report_error",
            "stage": stage,
            "reason": reason,
        },
    }


def _validate_report_output_path(path: Path) -> Path:
    """Validate a report destination before opening or replacing anything."""
    path = Path(path)
    if path.suffix != ".html":
        raise ValueError("output path must have a .html suffix")
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("output parent directory is missing or not a directory")

    # Check every lexical parent component before any resolve operation.  This
    # keeps a symlinked directory from redirecting a temporary file or replace.
    current = parent
    while True:
        if current.is_symlink():
            raise ValueError(f"output parent is a symbolic link: {current}")
        if not current.is_dir():
            raise ValueError(f"output parent is not a directory: {current}")
        if current == current.parent:
            break
        current = current.parent

    if path.is_symlink():
        raise ValueError(f"output path is a symbolic link: {path}")
    if path.exists():
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise ValueError(f"output path is unreadable: {path}") from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"output path is not a regular file: {path}")
        if mode & 0o222 == 0 or not os.access(path, os.W_OK):
            raise OSError(f"output path is not writable: {path}")

    try:
        parent_mode = parent.stat().st_mode
    except OSError as exc:
        raise ValueError(f"output parent is unreadable: {parent}") from exc
    if parent_mode & 0o222 == 0 or not os.access(parent, os.W_OK | os.X_OK):
        raise OSError(f"output parent is not writable: {parent}")
    return path


def _fsync_parent_directory(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_coverage_report(path: Path, body: bytes) -> Path:
    """Atomically publish deterministic report bytes to a validated HTML path."""
    if not isinstance(body, bytes):
        raise TypeError("report body must be bytes")
    path = _validate_report_output_path(Path(path))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_parent_directory(path.parent)
        return path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Render and atomically publish one zero-hosted-call coverage report."""
    parser = _coverage_report_parser()
    stage = "config"
    try:
        arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
        topic_ids = tuple(arguments.topic_ids)
        if any(not isinstance(topic_id, str) or not topic_id.strip() for topic_id in topic_ids):
            raise ValueError("topic selector cannot be empty")
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError("duplicate topic selector")
        output_path = _validate_report_output_path(arguments.output)
        stage = "load"
        data = load_coverage_report_data(
            handoff_manifest_path=arguments.handoff_manifest,
            coverage_root=arguments.coverage_root,
            topic_ids=topic_ids,
        )
        stage = "render"
        body = render_coverage_report_html(data)
        stage = "publish"
        output = publish_coverage_report(output_path, body)
        receipt = {
            "status": "ok",
            "selected_topic_count": len(data.topics),
            "output": str(output),
            "output_sha256": sha256(body).hexdigest(),
            "hosted_calls": 0,
        }
        print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
        return 0
    except _ParserExit as exc:
        return exc.status
    except Exception:
        print(
            json.dumps(_safe_cli_error(stage), separators=(",", ":"), sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CoverageReportData",
    "CoverageReportTopic",
    "CoverageRunSummary",
    "RetrievalPlanContext",
    "RetrievalSubnarrativeContext",
    "load_coverage_report_data",
    "main",
    "publish_coverage_report",
    "render_coverage_report_html",
    "summarize_coverage_topics",
]

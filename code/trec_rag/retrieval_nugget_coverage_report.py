"""Typed, year-neutral inputs for retrieval nugget coverage reports.

This module loads only the sealed handoff, validated decomposition checkpoints,
and completed coverage bundles needed by a report renderer.  It deliberately
does not inspect retrieval, passage, canonicalization, or provider artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
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


__all__ = [
    "CoverageReportData",
    "CoverageReportTopic",
    "CoverageRunSummary",
    "RetrievalPlanContext",
    "RetrievalSubnarrativeContext",
    "load_coverage_report_data",
    "summarize_coverage_topics",
]

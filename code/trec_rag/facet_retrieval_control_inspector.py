"""Qrels-free inspection for the frozen facet retrieval-control streams."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    ControlManifest,
    _validate_manifest,
)
from .pipeline_models import RetrievedCandidate


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CONTENT_QUALITY_PATTERNS = (
    "essay",
    "homework",
    "term paper",
    "write a custom",
    "writing service",
    "assignment help",
)


@dataclass(frozen=True)
class InspectionStream:
    """The diagnostic groups copied from one exact R1 stream record."""

    topic_id: str
    stream_id: str
    anchor_groups: tuple[tuple[str, ...], ...]
    intent_groups: tuple[tuple[str, ...], ...]
    forbidden_drift_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class InspectionResult:
    """Top-five decision signals plus descriptive top-ten diagnostics."""

    topic_id: str
    stream_id: str
    inspected_top5: int
    inspected_top10: int
    anchor_top5_count: int
    anchor_top10_count: int
    anchor_intent_cohit_top5_count: int
    anchor_intent_cohit_top10_count: int
    domain_drift_top5_count: int
    domain_drift_top10_count: int
    content_quality_top5_count: int
    content_quality_top10_count: int
    coherence_failed: bool
    domain_drift_warning: bool
    content_quality_warning: bool
    rejected: bool
    decision: str
    top_docids: tuple[str, ...]
    top_snippets: tuple[str, ...]


def _reject_protected_topic(topic_id: str) -> None:
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")


def _parse_groups(value: object, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"R1 inspection field {field} must be a non-empty list")
    groups: list[tuple[str, ...]] = []
    for raw_group in value:
        if (
            not isinstance(raw_group, list)
            or not raw_group
            or not all(isinstance(term, str) and term for term in raw_group)
        ):
            raise ValueError(f"R1 inspection field {field} contains an invalid group")
        groups.append(tuple(raw_group))
    return tuple(groups)


def load_inspection_streams(
    r1_path: Path,
    manifest: ControlManifest,
) -> dict[tuple[str, str], InspectionStream]:
    """Load only registered diagnostic groups from the byte-pinned R1 source."""

    for stream in manifest.streams:
        _reject_protected_topic(stream.topic_id)
    _validate_manifest(manifest)
    source = Path(r1_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest.r1_manifest_sha256:
        raise ValueError("R1 source manifest SHA-256 does not match the control manifest")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("R1 source manifest is not valid JSON") from exc
    raw_streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(raw_streams, list):
        raise ValueError("R1 source manifest streams must be a list")
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for raw in raw_streams:
        if not isinstance(raw, dict):
            raise ValueError("R1 source manifest contains a non-object stream")
        topic_id = raw.get("topic_id")
        stream_id = raw.get("stream_id")
        if isinstance(topic_id, str) and isinstance(stream_id, str):
            indexed[(topic_id, stream_id)] = raw

    result: dict[tuple[str, str], InspectionStream] = {}
    for control_stream in manifest.streams:
        boundary = (control_stream.topic_id, control_stream.stream_id)
        raw = indexed.get(boundary)
        if raw is None:
            raise ValueError(f"R1 source manifest is missing stream {boundary!r}")
        result[boundary] = InspectionStream(
            topic_id=control_stream.topic_id,
            stream_id=control_stream.stream_id,
            anchor_groups=_parse_groups(raw.get("anchor_groups"), "anchor_groups"),
            intent_groups=_parse_groups(raw.get("intent_groups"), "intent_groups"),
            forbidden_drift_groups=_parse_groups(
                raw.get("forbidden_drift_groups"), "forbidden_drift_groups"
            ),
        )
    return result


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def _matches_term(tokens: tuple[str, ...], term: str) -> bool:
    return any(token.startswith(term) for token in tokens)


def _matches_groups(
    tokens: tuple[str, ...], groups: tuple[tuple[str, ...], ...]
) -> bool:
    return all(any(_matches_term(tokens, term) for term in group) for group in groups)


def _content_quality_warning(text: str) -> bool:
    normalized = " ".join(_TOKEN_RE.findall(text.lower()))
    return any(pattern in normalized for pattern in _CONTENT_QUALITY_PATTERNS)


def inspect_stream(
    stream: InspectionStream,
    candidates: Sequence[RetrievedCandidate],
) -> InspectionResult:
    """Inspect an arm deterministically; top-ten signals never change the decision."""

    _reject_protected_topic(stream.topic_id)
    foreign_topics = sorted({row.topic_id for row in candidates} - {stream.topic_id})
    if foreign_topics:
        raise ValueError(f"inspection candidates contain foreign topics: {foreign_topics!r}")
    ordered = sorted(
        candidates,
        key=lambda row: (row.rank, row.docid, -row.score, row.variant_name, row.text),
    )[:10]
    diagnostic_rows: list[tuple[bool, bool, bool, bool]] = []
    for candidate in ordered:
        tokens = _tokens(candidate.text)
        anchor = _matches_groups(tokens, stream.anchor_groups)
        intent = _matches_groups(tokens, stream.intent_groups)
        drift = any(
            any(_matches_term(tokens, term) for term in group)
            for group in stream.forbidden_drift_groups
        )
        diagnostic_rows.append(
            (anchor, anchor and intent, drift, _content_quality_warning(candidate.text))
        )

    top5 = diagnostic_rows[:5]
    anchor_top5 = sum(row[0] for row in top5)
    cohit_top5 = sum(row[1] for row in top5)
    drift_top5 = sum(row[2] for row in top5)
    content_top5 = sum(row[3] for row in top5)
    coherence_failed = len(top5) < 5 or anchor_top5 < 3 or cohit_top5 < 3
    domain_warning = drift_top5 >= 2
    content_warning = content_top5 >= 2
    rejected = coherence_failed or (domain_warning and content_warning)
    if coherence_failed:
        decision = "reject_coherence"
    elif domain_warning and content_warning:
        decision = "reject_independent_warnings"
    elif domain_warning or content_warning:
        decision = "keep_with_warning"
    else:
        decision = "keep"

    return InspectionResult(
        topic_id=stream.topic_id,
        stream_id=stream.stream_id,
        inspected_top5=len(top5),
        inspected_top10=len(diagnostic_rows),
        anchor_top5_count=anchor_top5,
        anchor_top10_count=sum(row[0] for row in diagnostic_rows),
        anchor_intent_cohit_top5_count=cohit_top5,
        anchor_intent_cohit_top10_count=sum(row[1] for row in diagnostic_rows),
        domain_drift_top5_count=drift_top5,
        domain_drift_top10_count=sum(row[2] for row in diagnostic_rows),
        content_quality_top5_count=content_top5,
        content_quality_top10_count=sum(row[3] for row in diagnostic_rows),
        coherence_failed=coherence_failed,
        domain_drift_warning=domain_warning,
        content_quality_warning=content_warning,
        rejected=rejected,
        decision=decision,
        top_docids=tuple(row.docid for row in ordered),
        top_snippets=tuple(row.text[:400] for row in ordered),
    )

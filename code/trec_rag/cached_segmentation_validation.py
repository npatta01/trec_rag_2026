"""Authenticated before/after validation for cached sentence re-segmentation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import tempfile
from typing import Any, NoReturn

from trec_rag.document_store import DocumentStore
from trec_rag.evidence_store import _iter_candidate_requests
from trec_rag.facet_evidence import _byte_offsets, _source_spans
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
)
from trec_rag.retrieval_nugget_coverage import (
    CompletedCoverageEvaluation,
    CoverageModelBackend,
    CoverageRunConfig,
    load_completed_coverage_evaluation,
    run_coverage_evaluation,
    seed_coverage_plan_from_completed_baseline,
)
from trec_rag.topic_records import TopicRecords


STRUCTURAL_COMPARISON_SCHEMA = "cached-segmentation-structural-comparison-v1"
STRUCTURAL_MANIFEST_SCHEMA = "cached-segmentation-structural-manifest-v1"
SEMANTIC_COMPARISON_SCHEMA = "cached-segmentation-semantic-comparison-v1"
SEMANTIC_MANIFEST_SCHEMA = "cached-segmentation-semantic-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LABEL_VALUES = {"unsupported": 0.0, "partial": 0.5, "full": 1.0}


@dataclass(frozen=True)
class TextShapeMetrics:
    count: int
    median_characters: float
    short_count: int
    short_fraction: float
    fragment_count: int
    fragment_fraction: float


@dataclass(frozen=True)
class StructuralTopicMetrics:
    topic_id: str
    narrative_sha256: str
    candidate_request_count: int
    source_document_count: int
    source_document_sha256s: tuple[str, ...]
    old_line_units: TextShapeMetrics
    fixed_segmentation_units: TextShapeMetrics
    baseline_candidate_units: TextShapeMetrics
    candidate_units: TextShapeMetrics
    baseline_selected_evidence: TextShapeMetrics
    candidate_selected_evidence: TextShapeMetrics
    baseline_representatives: TextShapeMetrics
    candidate_representatives: TextShapeMetrics
    baseline_claim_hints: TextShapeMetrics
    candidate_claim_hints: TextShapeMetrics
    baseline_candidate_kind_counts: tuple[tuple[str, int], ...]
    candidate_kind_counts: tuple[tuple[str, int], ...]
    selected_cluster_count: int
    upstream_identity_sha256: str
    gates_passed: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class StructuralComparison:
    schema_version: str
    baseline_handoff_sha256: str
    candidate_handoff_sha256: str
    topic_ids: tuple[str, ...]
    topics: tuple[StructuralTopicMetrics, ...]
    aggregate_old_line_units: TextShapeMetrics
    aggregate_fixed_segmentation_units: TextShapeMetrics
    aggregate_baseline_candidate_units: TextShapeMetrics
    aggregate_candidate_units: TextShapeMetrics
    aggregate_baseline_selected_evidence: TextShapeMetrics
    aggregate_candidate_selected_evidence: TextShapeMetrics
    aggregate_baseline_representatives: TextShapeMetrics
    aggregate_candidate_representatives: TextShapeMetrics
    aggregate_baseline_claim_hints: TextShapeMetrics
    aggregate_candidate_claim_hints: TextShapeMetrics
    improvement_observed: bool
    gates_passed: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class SemanticObligationComparison:
    obligation_id: str
    kind: str
    baseline_label: str
    candidate_label: str
    label_delta: float


@dataclass(frozen=True)
class SemanticTopicComparison:
    topic_id: str
    narrative_sha256: str
    plan_sha256: str
    baseline_required_coverage: float
    candidate_required_coverage: float
    required_coverage_delta: float
    baseline_strict_full_rate: float
    candidate_strict_full_rate: float
    strict_full_rate_delta: float
    baseline_label_counts: tuple[tuple[str, int], ...]
    candidate_label_counts: tuple[tuple[str, int], ...]
    obligations: tuple[SemanticObligationComparison, ...]
    improved_obligation_ids: tuple[str, ...]
    regressed_obligation_ids: tuple[str, ...]
    baseline_artifact_sha256s: tuple[tuple[str, str], ...]
    candidate_artifact_sha256s: tuple[tuple[str, str], ...]
    candidate_judge_calls: int


@dataclass(frozen=True)
class SemanticComparison:
    schema_version: str
    diagnostic_scope: str
    baseline_handoff_sha256: str
    candidate_handoff_sha256: str
    topic_ids: tuple[str, ...]
    topics: tuple[SemanticTopicComparison, ...]
    baseline_topic_macro_required_coverage: float
    candidate_topic_macro_required_coverage: float
    topic_macro_required_coverage_delta: float
    baseline_topic_macro_strict_full_rate: float
    candidate_topic_macro_strict_full_rate: float
    topic_macro_strict_full_rate_delta: float
    improved_obligation_ids: tuple[str, ...]
    regressed_obligation_ids: tuple[str, ...]
    planner_calls: int
    candidate_judge_calls: int
    expected_candidate_judge_calls: int
    gates_passed: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class _SelectedDocument:
    docid: str
    rank: int
    text_sha256: str
    text: str


@dataclass(frozen=True)
class _LoadedCandidateTopic:
    texts: tuple[str, ...]
    candidate_ids: frozenset[str]
    kind_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _CandidateRequestPopulation:
    sources: tuple[str, ...]
    source_sha256s: tuple[str, ...]
    request_count: int
    sealed_bytes: bytes


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _strict_json(body: bytes, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-standard constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _measure_texts(texts: Sequence[str]) -> TextShapeMetrics:
    normalized = tuple(text.strip() for text in texts if text.strip())
    lengths = tuple(len(text) for text in normalized)
    short_count = sum(length < 40 for length in lengths)
    fragment_count = sum(
        len(text) < 40 or text.rstrip()[-1:] not in ".?!"
        for text in normalized
    )
    count = len(normalized)
    return TextShapeMetrics(
        count=count,
        median_characters=(float(statistics.median(lengths)) if lengths else 0.0),
        short_count=short_count,
        short_fraction=(short_count / count if count else 0.0),
        fragment_count=fragment_count,
        fragment_fraction=(fragment_count / count if count else 0.0),
    )


def _artifact_bytes(
    output_root: Path,
    topic_id: str,
    *,
    manifest_relative: str,
    artifact_relative: str,
) -> bytes:
    topic_root = Path(output_root) / topic_id
    manifest_path = topic_root / manifest_relative
    manifest = _strict_json(manifest_path.read_bytes(), str(manifest_path))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{manifest_path} must contain an object")
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list):
        raise ValueError(f"{manifest_path} artifact receipts changed")
    matches = [
        row
        for row in receipts
        if isinstance(row, Mapping)
        and row.get("relative_path") == artifact_relative
    ]
    if len(matches) != 1:
        raise ValueError(f"{artifact_relative} receipt is missing or duplicated")
    receipt = matches[0]
    body = (topic_root / artifact_relative).read_bytes()
    if (
        type(receipt.get("bytes")) is not int
        or receipt["bytes"] != len(body)
        or not isinstance(receipt.get("sha256"), str)
        or _SHA256.fullmatch(receipt["sha256"]) is None
        or receipt["sha256"] != _digest(body)
    ):
        raise ValueError(f"artifact {artifact_relative} changed after sealing")
    return body


def _decomposition_bytes(output_root: Path, topic_id: str) -> bytes:
    topic_root = Path(output_root) / topic_id
    result_path = topic_root / "decomposition" / "result.json"
    manifest_path = result_path.with_name("manifest.json")
    manifest = _strict_json(manifest_path.read_bytes(), str(manifest_path))
    body = result_path.read_bytes()
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("result_file") != result_path.name
        or manifest.get("result_bytes") != len(body)
        or manifest.get("result_sha256") != _digest(body)
    ):
        raise ValueError("decomposition result changed after sealing")
    return body


def _upstream_identity(output_root: Path, topic_id: str) -> tuple[str, bytes]:
    decomposition = _decomposition_bytes(output_root, topic_id)
    retrieval = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="retrieval/complete.json",
        artifact_relative="retrieval/audit.json",
    )
    selected = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="scoring/complete.json",
        artifact_relative="scoring/selected_documents.jsonl",
    )
    identity = {
        "decomposition_sha256": _digest(decomposition),
        "retrieval_audit_sha256": _digest(retrieval),
        "selected_documents_sha256": _digest(selected),
    }
    return _digest(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ), selected


def _selected_documents(
    body: bytes,
    *,
    topic_id: str,
    store: DocumentStore,
) -> tuple[_SelectedDocument, ...]:
    rows: list[_SelectedDocument] = []
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        value = _strict_json(raw_line, f"selected_documents:{line_number}")
        if not isinstance(value, Mapping):
            raise ValueError("selected_documents rows must be objects")
        docid = value.get("docid")
        rank = value.get("selection_rank")
        text = value.get("text")
        digest = value.get("text_sha256")
        if (
            value.get("topic_id") != topic_id
            or not isinstance(docid, str)
            or not docid
            or type(rank) is not int
            or rank <= 0
            or not isinstance(text, str)
            or not isinstance(digest, str)
            or digest != _digest(text.encode("utf-8"))
            or store.read_text(digest) != text
        ):
            raise ValueError("selected_documents row changed or is invalid")
        rows.append(_SelectedDocument(docid, rank, digest, text))
    rows.sort(key=lambda row: (row.rank, row.docid))
    if (
        [row.rank for row in rows] != list(range(1, len(rows) + 1))
        or len({row.docid for row in rows}) != len(rows)
    ):
        raise ValueError("selected_documents ranks or docids changed")
    return tuple(rows)


def _topic_by_id(handoff: GenerationHandoff, topic_id: str) -> GenerationTopic:
    matches = tuple(topic for topic in handoff.topics if topic.topic_id == topic_id)
    if len(matches) != 1:
        raise ValueError(f"handoff topic {topic_id!r} is absent or duplicated")
    return matches[0]


def _validate_handoff_sources(topic: GenerationTopic, store: DocumentStore) -> None:
    for evidence in topic.evidence:
        source = store.read_text(evidence.document_sha256)
        span = evidence.source_span
        if (
            source[span.start_char : span.end_char] != evidence.text
            or len(source[: span.start_char].encode("utf-8")) != span.start_byte
            or len(source[: span.end_char].encode("utf-8")) != span.end_byte
        ):
            raise ValueError("handoff evidence provenance changed")


def _representative_texts(topic: GenerationTopic) -> tuple[str, ...]:
    evidence = {row.evidence_id: row.text for row in topic.evidence}
    return tuple(
        evidence[cluster.representative_evidence_id]
        for group in topic.groups
        for cluster in group.selected_clusters
    )


def _segmentation_texts(sources: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        span.text
        for source in sources
        for span in _source_spans(source, _byte_offsets(source))
    )


def _load_candidate_request_population(
    output_root: Path,
    document_store_root: Path,
    topic_id: str,
) -> _CandidateRequestPopulation:
    request_relative = "canonical/handoff/candidate-requests.jsonl"
    manifest_relative = "canonical/handoff/handoff-manifest.json"
    request_body = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="canonical/complete.json",
        artifact_relative=request_relative,
    )
    manifest_body = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="canonical/complete.json",
        artifact_relative=manifest_relative,
    )
    manifest = _strict_json(manifest_body, f"candidate handoff:{topic_id}")
    request_path = Path(output_root) / topic_id / request_relative
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("topic_id") != topic_id
        or manifest.get("requests_file") != request_path.name
        or manifest.get("requests_sha256") != _digest(request_body)
        or not isinstance(manifest.get("request_schema_version"), str)
    ):
        raise ValueError("candidate request handoff identity changed")
    store = DocumentStore(Path(document_store_root))
    requests = tuple(
        _iter_candidate_requests(
            request_path,
            document_store=store,
            expected_schema=manifest["request_schema_version"],
        )
    )
    if any(request.topic_id != topic_id for request in requests):
        raise ValueError("candidate request topic identity changed")
    source_sha256s = tuple(sorted({request.document_sha256 for request in requests}))
    document_count = len({request.document_id for request in requests})
    if document_count != manifest.get("document_count"):
        raise ValueError("candidate request document count changed")
    return _CandidateRequestPopulation(
        sources=tuple(request.source for request in requests),
        source_sha256s=source_sha256s,
        request_count=len(requests),
        sealed_bytes=request_body,
    )


def _load_sealed_candidate_topic(
    output_root: Path,
    document_store_root: Path,
    topic_id: str,
    *,
    require_current_splitter: bool,
) -> _LoadedCandidateTopic:
    """Read a byte-sealed candidate population, including legacy splitters."""
    database_relative = "records.sqlite3"
    records_manifest_relative = "canonical/records-manifest.json"
    database_body = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="canonical/complete.json",
        artifact_relative=database_relative,
    )
    records_manifest_body = _artifact_bytes(
        output_root,
        topic_id,
        manifest_relative="canonical/complete.json",
        artifact_relative=records_manifest_relative,
    )
    records_manifest = _strict_json(
        records_manifest_body, f"records manifest:{topic_id}"
    )
    database_path = Path(output_root) / topic_id / database_relative
    if (
        not isinstance(records_manifest, Mapping)
        or records_manifest.get("topic_id") != topic_id
        or records_manifest.get("records_file") != database_path.name
        or records_manifest.get("database_bytes") != len(database_body)
        or records_manifest.get("database_sha256") != _digest(database_body)
        or not isinstance(records_manifest.get("document_sha256s"), list)
        or not isinstance(records_manifest.get("row_counts"), Mapping)
    ):
        raise ValueError("records manifest identity changed")
    store = DocumentStore(Path(document_store_root))
    uri = database_path.resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as database:
        database.execute("PRAGMA query_only=ON")
        if database.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("candidate SQLite integrity check failed")
        rows = database.execute(
            "SELECT c.candidate_nugget_id, c.candidate_kind, c.start_char, "
            "c.end_char, c.text_sha256, c.sentence_splitter_version, "
            "d.content_sha256 FROM candidate AS c "
            "JOIN document_binding AS d ON d.document_pk=c.document_pk "
            "WHERE c.topic_id=? ORDER BY c.subnarrative_id, c.candidate_nugget_id",
            (topic_id,),
        ).fetchall()
        database_document_sha256s = tuple(
            row[0]
            for row in database.execute(
                "SELECT DISTINCT content_sha256 FROM document_binding "
                "WHERE topic_id=? ORDER BY content_sha256",
                (topic_id,),
            ).fetchall()
        )
    if len(rows) != records_manifest["row_counts"].get("candidate"):
        raise ValueError("sealed candidate row count changed")
    if database_document_sha256s != tuple(records_manifest["document_sha256s"]):
        raise ValueError("sealed candidate document closure changed")
    texts: list[str] = []
    candidate_ids: set[str] = set()
    kinds: Counter[str] = Counter()
    for candidate_id, kind, start, end, text_sha256, splitter, document_sha256 in rows:
        source = store.read_text(document_sha256)
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_ids
            or not isinstance(kind, str)
            or not kind
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(source)
            or not isinstance(text_sha256, str)
            or text_sha256 != _digest(source[start:end].encode("utf-8"))
            or not isinstance(splitter, str)
            or not splitter
        ):
            raise ValueError("sealed candidate source row changed")
        candidate_ids.add(candidate_id)
        kinds[kind] += 1
        texts.append(source[start:end])
    loaded = _LoadedCandidateTopic(
        texts=tuple(texts),
        candidate_ids=frozenset(candidate_ids),
        kind_counts=tuple(sorted(kinds.items())),
    )
    if require_current_splitter:
        current = load_structural_topic(output_root, document_store_root, topic_id)
        if current != loaded:
            raise ValueError("current candidate validation disagrees with sealed population")
    return loaded


def load_structural_topic(
    output_root: Path,
    document_store_root: Path,
    topic_id: str,
) -> _LoadedCandidateTopic:
    """Load and source-validate every fixed candidate for one topic."""
    topic_root = Path(output_root) / topic_id
    store = DocumentStore(Path(document_store_root))
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic_id,
        store,
    ) as records:
        records.validate_all_sources()
        candidates = tuple(records.load_candidates().values())
    kinds = Counter(candidate.candidate_kind for candidate in candidates)
    return _LoadedCandidateTopic(
        texts=tuple(candidate.text for candidate in candidates),
        candidate_ids=frozenset(
            candidate.candidate_nugget_id for candidate in candidates
        ),
        kind_counts=tuple(sorted(kinds.items())),
    )


def compare_structural_runs(
    *,
    baseline_output_root: Path,
    candidate_output_root: Path,
    document_store_root: Path,
    baseline_handoff_path: Path,
    candidate_handoff_path: Path,
    topic_ids: Sequence[str],
) -> StructuralComparison:
    ordered_topic_ids = tuple(topic_ids)
    if (
        not ordered_topic_ids
        or any(not isinstance(topic_id, str) or not topic_id for topic_id in ordered_topic_ids)
        or len(set(ordered_topic_ids)) != len(ordered_topic_ids)
    ):
        raise ValueError("topic_ids must be unique non-empty strings")
    baseline_handoff = load_generation_handoff(Path(baseline_handoff_path))
    candidate_handoff = load_generation_handoff(Path(candidate_handoff_path))
    if tuple(topic.topic_id for topic in baseline_handoff.topics) != ordered_topic_ids:
        raise ValueError("baseline handoff topic order changed")
    if tuple(topic.topic_id for topic in candidate_handoff.topics) != ordered_topic_ids:
        raise ValueError("candidate handoff topic order changed")

    store = DocumentStore(Path(document_store_root))
    topic_metrics: list[StructuralTopicMetrics] = []
    all_old_lines: list[str] = []
    all_fixed_segments: list[str] = []
    all_baseline_candidate_texts: list[str] = []
    all_candidate_texts: list[str] = []
    all_baseline_evidence: list[str] = []
    all_candidate_evidence: list[str] = []
    all_baseline_representatives: list[str] = []
    all_candidate_representatives: list[str] = []
    all_baseline_claims: list[str] = []
    all_candidate_claims: list[str] = []
    for topic_id in ordered_topic_ids:
        baseline_topic = _topic_by_id(baseline_handoff, topic_id)
        candidate_topic = _topic_by_id(candidate_handoff, topic_id)
        if (
            baseline_topic.narrative != candidate_topic.narrative
            or baseline_topic.narrative_sha256 != candidate_topic.narrative_sha256
            or baseline_topic.source_receipts.official_topics_sha256
            != candidate_topic.source_receipts.official_topics_sha256
        ):
            raise ValueError("candidate narrative or official topic identity changed")
        _validate_handoff_sources(baseline_topic, store)
        _validate_handoff_sources(candidate_topic, store)

        baseline_identity, baseline_selected_bytes = _upstream_identity(
            Path(baseline_output_root), topic_id
        )
        candidate_identity, candidate_selected_bytes = _upstream_identity(
            Path(candidate_output_root), topic_id
        )
        if baseline_identity != candidate_identity:
            raise ValueError("authenticated upstream identity changed")
        if baseline_selected_bytes != candidate_selected_bytes:
            raise ValueError("selected_documents bytes changed")
        _selected_documents(
            baseline_selected_bytes,
            topic_id=topic_id,
            store=store,
        )
        baseline_requests = _load_candidate_request_population(
            Path(baseline_output_root),
            Path(document_store_root),
            topic_id,
        )
        candidate_requests = _load_candidate_request_population(
            Path(candidate_output_root),
            Path(document_store_root),
            topic_id,
        )
        if baseline_requests.sealed_bytes != candidate_requests.sealed_bytes:
            raise ValueError("candidate request population changed")
        baseline_candidates = _load_sealed_candidate_topic(
            Path(baseline_output_root),
            Path(document_store_root),
            topic_id,
            require_current_splitter=False,
        )
        loaded_candidates = _load_sealed_candidate_topic(
            Path(candidate_output_root),
            Path(document_store_root),
            topic_id,
            require_current_splitter=True,
        )
        if any(
            evidence.evidence_id not in baseline_candidates.candidate_ids
            for evidence in baseline_topic.evidence
        ):
            raise ValueError("baseline handoff references an unvalidated candidate")
        if any(
            evidence.evidence_id not in loaded_candidates.candidate_ids
            for evidence in candidate_topic.evidence
        ):
            raise ValueError("candidate handoff references an unvalidated candidate")

        old_lines = tuple(
            line.strip()
            for source in baseline_requests.sources
            for line in source.splitlines()
            if line.strip()
        )
        fixed_segments = _segmentation_texts(baseline_requests.sources)
        baseline_evidence_texts = tuple(row.text for row in baseline_topic.evidence)
        candidate_evidence_texts = tuple(row.text for row in candidate_topic.evidence)
        baseline_representative_texts = _representative_texts(baseline_topic)
        candidate_representative_texts = _representative_texts(candidate_topic)
        baseline_claim_texts = tuple(row.text for row in baseline_topic.claim_hints)
        candidate_claim_texts = tuple(row.text for row in candidate_topic.claim_hints)
        baseline_evidence = _measure_texts(
            baseline_evidence_texts
        )
        candidate_evidence = _measure_texts(
            candidate_evidence_texts
        )
        baseline_representatives = _measure_texts(
            baseline_representative_texts
        )
        candidate_representatives = _measure_texts(
            candidate_representative_texts
        )
        old_line_metrics = _measure_texts(old_lines)
        fixed_segment_metrics = _measure_texts(fixed_segments)
        baseline_candidate_metrics = _measure_texts(baseline_candidates.texts)
        candidate_metrics = _measure_texts(loaded_candidates.texts)
        failures: list[str] = []
        if candidate_metrics.short_fraction > baseline_candidate_metrics.short_fraction:
            failures.append("candidate short fraction worsened")
        if candidate_metrics.fragment_fraction > baseline_candidate_metrics.fragment_fraction:
            failures.append("candidate fragment fraction worsened")
        if candidate_evidence.short_fraction > baseline_evidence.short_fraction:
            failures.append("selected evidence short fraction worsened")
        if candidate_evidence.fragment_fraction > baseline_evidence.fragment_fraction:
            failures.append("selected evidence fragment fraction worsened")
        if candidate_representatives.fragment_fraction > baseline_representatives.fragment_fraction:
            failures.append("representative fragment fraction worsened")
        candidate_claims = _measure_texts(candidate_claim_texts)
        baseline_claims = _measure_texts(baseline_claim_texts)
        if candidate_claims.fragment_fraction > baseline_claims.fragment_fraction:
            failures.append("claim-hint fragment fraction worsened")
        if topic_id == "407":
            if fixed_segment_metrics.median_characters <= 11:
                failures.append("topic 407 segmentation median did not exceed 11")
            if fixed_segment_metrics.short_fraction >= 0.604:
                failures.append("topic 407 segmentation short fraction did not beat 0.604")

        topic_metrics.append(
            StructuralTopicMetrics(
                topic_id=topic_id,
                narrative_sha256=baseline_topic.narrative_sha256,
                candidate_request_count=baseline_requests.request_count,
                source_document_count=len(baseline_requests.source_sha256s),
                source_document_sha256s=baseline_requests.source_sha256s,
                old_line_units=old_line_metrics,
                fixed_segmentation_units=fixed_segment_metrics,
                baseline_candidate_units=baseline_candidate_metrics,
                candidate_units=candidate_metrics,
                baseline_selected_evidence=baseline_evidence,
                candidate_selected_evidence=candidate_evidence,
                baseline_representatives=baseline_representatives,
                candidate_representatives=candidate_representatives,
                baseline_claim_hints=baseline_claims,
                candidate_claim_hints=candidate_claims,
                baseline_candidate_kind_counts=baseline_candidates.kind_counts,
                candidate_kind_counts=loaded_candidates.kind_counts,
                selected_cluster_count=sum(
                    len(group.selected_clusters) for group in candidate_topic.groups
                ),
                upstream_identity_sha256=baseline_identity,
                gates_passed=not failures,
                gate_failures=tuple(failures),
            )
        )

        all_old_lines.extend(old_lines)
        all_fixed_segments.extend(fixed_segments)
        all_baseline_candidate_texts.extend(baseline_candidates.texts)
        all_candidate_texts.extend(loaded_candidates.texts)
        all_baseline_evidence.extend(baseline_evidence_texts)
        all_candidate_evidence.extend(candidate_evidence_texts)
        all_baseline_representatives.extend(baseline_representative_texts)
        all_candidate_representatives.extend(candidate_representative_texts)
        all_baseline_claims.extend(baseline_claim_texts)
        all_candidate_claims.extend(candidate_claim_texts)

    aggregate_old_lines = _measure_texts(all_old_lines)
    aggregate_fixed_segments = _measure_texts(all_fixed_segments)
    aggregate_baseline_candidates = _measure_texts(all_baseline_candidate_texts)
    aggregate_candidates = _measure_texts(all_candidate_texts)
    aggregate_baseline_evidence = _measure_texts(all_baseline_evidence)
    aggregate_candidate_evidence = _measure_texts(all_candidate_evidence)
    aggregate_baseline_representatives = _measure_texts(all_baseline_representatives)
    aggregate_candidate_representatives = _measure_texts(all_candidate_representatives)
    aggregate_baseline_claims = _measure_texts(all_baseline_claims)
    aggregate_candidate_claims = _measure_texts(all_candidate_claims)
    aggregate_failures: list[str] = []
    if aggregate_fixed_segments.median_characters <= aggregate_old_lines.median_characters:
        aggregate_failures.append("aggregate segmentation median did not improve")
    if aggregate_fixed_segments.short_fraction > aggregate_old_lines.short_fraction:
        aggregate_failures.append("aggregate segmentation short fraction worsened")
    if aggregate_candidates.short_fraction > aggregate_baseline_candidates.short_fraction:
        aggregate_failures.append("aggregate candidate short fraction worsened")
    if aggregate_candidates.fragment_fraction > aggregate_baseline_candidates.fragment_fraction:
        aggregate_failures.append("aggregate candidate fragment fraction worsened")
    if (
        aggregate_candidate_evidence.short_fraction
        > aggregate_baseline_evidence.short_fraction
    ):
        aggregate_failures.append("aggregate selected evidence short fraction worsened")
    if (
        aggregate_candidate_evidence.fragment_fraction
        > aggregate_baseline_evidence.fragment_fraction
    ):
        aggregate_failures.append("aggregate selected evidence fragment fraction worsened")
    if (
        aggregate_candidate_representatives.fragment_fraction
        > aggregate_baseline_representatives.fragment_fraction
    ):
        aggregate_failures.append("aggregate representative fragment fraction worsened")
    if (
        aggregate_candidate_claims.fragment_fraction
        > aggregate_baseline_claims.fragment_fraction
    ):
        aggregate_failures.append("aggregate claim-hint fragment fraction worsened")
    improvement_observed = any(
        (
            aggregate_candidates.short_fraction
            < aggregate_baseline_candidates.short_fraction,
            aggregate_candidates.fragment_fraction
            < aggregate_baseline_candidates.fragment_fraction,
            aggregate_candidate_evidence.short_fraction
            < aggregate_baseline_evidence.short_fraction,
            aggregate_candidate_evidence.fragment_fraction
            < aggregate_baseline_evidence.fragment_fraction,
            aggregate_candidate_representatives.fragment_fraction
            < aggregate_baseline_representatives.fragment_fraction,
            aggregate_candidate_claims.fragment_fraction
            < aggregate_baseline_claims.fragment_fraction,
        )
    )
    if not improvement_observed:
        aggregate_failures.append("no fixed-output readability improvement observed")
    topic_failures = tuple(
        f"{topic.topic_id}: {failure}"
        for topic in topic_metrics
        for failure in topic.gate_failures
    )
    gate_failures = (*topic_failures, *aggregate_failures)

    return StructuralComparison(
        schema_version=STRUCTURAL_COMPARISON_SCHEMA,
        baseline_handoff_sha256=_digest(Path(baseline_handoff_path).read_bytes()),
        candidate_handoff_sha256=_digest(Path(candidate_handoff_path).read_bytes()),
        topic_ids=ordered_topic_ids,
        topics=tuple(topic_metrics),
        aggregate_old_line_units=aggregate_old_lines,
        aggregate_fixed_segmentation_units=aggregate_fixed_segments,
        aggregate_baseline_candidate_units=aggregate_baseline_candidates,
        aggregate_candidate_units=aggregate_candidates,
        aggregate_baseline_selected_evidence=aggregate_baseline_evidence,
        aggregate_candidate_selected_evidence=aggregate_candidate_evidence,
        aggregate_baseline_representatives=aggregate_baseline_representatives,
        aggregate_candidate_representatives=aggregate_candidate_representatives,
        aggregate_baseline_claim_hints=aggregate_baseline_claims,
        aggregate_candidate_claim_hints=aggregate_candidate_claims,
        improvement_observed=improvement_observed,
        gates_passed=not gate_failures,
        gate_failures=gate_failures,
    )


class _ForbiddenPlannerBackend:
    def complete(self, request: object) -> NoReturn:
        raise ValueError("semantic comparison forbids planner calls")


def _semantic_artifact_hashes(
    completed: CompletedCoverageEvaluation,
) -> tuple[tuple[str, str], ...]:
    hashes = dict(completed.artifact_hashes)
    hashes["manifest.json"] = completed.manifest_sha256
    return tuple(sorted(hashes.items()))


def _semantic_label_counts(labels: Sequence[str]) -> tuple[tuple[str, int], ...]:
    counts = Counter(labels)
    return tuple((label, counts[label]) for label in ("full", "partial", "unsupported"))


def _semantic_gate_failures(
    topics: Sequence[SemanticTopicComparison],
    *,
    baseline_required: float,
    candidate_required: float,
    baseline_strict: float,
    candidate_strict: float,
) -> tuple[str, ...]:
    """Reject local losses even when unrelated gains preserve macro means."""
    failures: list[str] = []
    for topic in topics:
        if topic.required_coverage_delta < 0:
            failures.append(f"{topic.topic_id}: required coverage regressed")
        if topic.strict_full_rate_delta < 0:
            failures.append(f"{topic.topic_id}: strict-full rate regressed")
        failures.extend(
            f"{topic.topic_id}: obligation {obligation.obligation_id} regressed"
            for obligation in topic.obligations
            if obligation.label_delta < 0
        )
    if candidate_required < baseline_required:
        failures.append("topic-macro required coverage regressed")
    if candidate_strict < baseline_strict:
        failures.append("topic-macro strict-full rate regressed")
    return tuple(failures)


def compare_semantic_runs(
    *,
    baseline_handoff_path: Path,
    baseline_coverage_root: Path,
    candidate_handoff_path: Path,
    candidate_coverage_root: Path,
    topic_ids: Sequence[str],
    judge: CoverageModelBackend | None = None,
) -> SemanticComparison:
    """Judge candidate nuggets against authenticated baseline obligation plans."""

    ordered_topic_ids = tuple(topic_ids)
    if (
        not ordered_topic_ids
        or any(
            not isinstance(topic_id, str) or not topic_id
            for topic_id in ordered_topic_ids
        )
        or len(set(ordered_topic_ids)) != len(ordered_topic_ids)
    ):
        raise ValueError("topic_ids must be unique non-empty strings")
    baseline_handoff = load_generation_handoff(Path(baseline_handoff_path))
    candidate_handoff = load_generation_handoff(Path(candidate_handoff_path))
    if tuple(topic.topic_id for topic in baseline_handoff.topics) != ordered_topic_ids:
        raise ValueError("baseline handoff topic order changed")
    if tuple(topic.topic_id for topic in candidate_handoff.topics) != ordered_topic_ids:
        raise ValueError("candidate handoff topic order changed")

    topics: list[SemanticTopicComparison] = []
    expected_judge_calls = 0
    actual_judge_calls = 0
    for topic_id in ordered_topic_ids:
        baseline_topic = _topic_by_id(baseline_handoff, topic_id)
        candidate_topic = _topic_by_id(candidate_handoff, topic_id)
        if (
            baseline_topic.narrative != candidate_topic.narrative
            or baseline_topic.narrative_sha256 != candidate_topic.narrative_sha256
        ):
            raise ValueError(f"candidate narrative changed for topic {topic_id}")

        baseline_work_dir = Path(baseline_coverage_root) / topic_id
        candidate_work_dir = Path(candidate_coverage_root) / topic_id
        baseline = load_completed_coverage_evaluation(
            handoff_manifest_path=Path(baseline_handoff_path),
            topic_id=topic_id,
            work_dir=baseline_work_dir,
        )
        post_plan_names = ("judgments.json", "report.json", "manifest.json")
        if not any((candidate_work_dir / name).exists() for name in post_plan_names):
            seed_coverage_plan_from_completed_baseline(
                baseline_handoff_manifest_path=Path(baseline_handoff_path),
                baseline_work_dir=baseline_work_dir,
                candidate_handoff_manifest_path=Path(candidate_handoff_path),
                candidate_work_dir=candidate_work_dir,
                topic_id=topic_id,
            )
        baseline_plan_path = baseline_work_dir / "plan.json"
        candidate_plan_path = candidate_work_dir / "plan.json"
        if (
            not candidate_plan_path.is_file()
            or candidate_plan_path.is_symlink()
            or candidate_plan_path.read_bytes() != baseline_plan_path.read_bytes()
        ):
            raise ValueError("candidate frozen plan changed")

        judge_was_missing = not (candidate_work_dir / "judgments.json").is_file()
        expected_judge_calls += int(judge_was_missing)
        receipt = run_coverage_evaluation(
            CoverageRunConfig(
                handoff_manifest_path=Path(candidate_handoff_path),
                topic_id=topic_id,
                work_dir=candidate_work_dir,
                planner_model=baseline.identity.planner_model,
                judge_model=baseline.identity.judge_model,
                mode="resume",
                allow_hosted_calls=True,
            ),
            planner=_ForbiddenPlannerBackend(),
            judge=judge,
        )
        if "planner" not in receipt.reused_stages:
            raise ValueError("candidate did not reuse the frozen baseline plan")
        if receipt.hosted_calls != int(judge_was_missing):
            raise ValueError("candidate hosted-call count changed")
        actual_judge_calls += receipt.hosted_calls

        candidate = load_completed_coverage_evaluation(
            handoff_manifest_path=Path(candidate_handoff_path),
            topic_id=topic_id,
            work_dir=candidate_work_dir,
        )
        if baseline.identity != candidate.identity:
            raise ValueError("candidate evaluator model or prompt identity changed")
        if (
            baseline.plan.plan_sha256 != candidate.plan.plan_sha256
            or baseline.plan.canonical_bytes != candidate.plan.canonical_bytes
            or (baseline_work_dir / "plan.json").read_bytes()
            != (candidate_work_dir / "plan.json").read_bytes()
        ):
            raise ValueError("candidate frozen plan changed")

        baseline_judgments = {
            row.obligation_id: row for row in baseline.judgments
        }
        candidate_judgments = {
            row.obligation_id: row for row in candidate.judgments
        }
        obligation_rows: list[SemanticObligationComparison] = []
        improved: list[str] = []
        regressed: list[str] = []
        for obligation in baseline.plan.obligations:
            old = baseline_judgments[obligation.obligation_id]
            fixed = candidate_judgments[obligation.obligation_id]
            delta = _LABEL_VALUES[fixed.label] - _LABEL_VALUES[old.label]
            obligation_rows.append(
                SemanticObligationComparison(
                    obligation_id=obligation.obligation_id,
                    kind=obligation.kind,
                    baseline_label=old.label,
                    candidate_label=fixed.label,
                    label_delta=delta,
                )
            )
            if delta > 0:
                improved.append(obligation.obligation_id)
            elif delta < 0:
                regressed.append(obligation.obligation_id)

        topics.append(
            SemanticTopicComparison(
                topic_id=topic_id,
                narrative_sha256=baseline.bound_input.narrative_sha256,
                plan_sha256=baseline.plan.plan_sha256,
                baseline_required_coverage=baseline.report.required_coverage,
                candidate_required_coverage=candidate.report.required_coverage,
                required_coverage_delta=(
                    candidate.report.required_coverage
                    - baseline.report.required_coverage
                ),
                baseline_strict_full_rate=baseline.report.strict_full_rate,
                candidate_strict_full_rate=candidate.report.strict_full_rate,
                strict_full_rate_delta=(
                    candidate.report.strict_full_rate
                    - baseline.report.strict_full_rate
                ),
                baseline_label_counts=_semantic_label_counts(
                    tuple(row.label for row in baseline.judgments)
                ),
                candidate_label_counts=_semantic_label_counts(
                    tuple(row.label for row in candidate.judgments)
                ),
                obligations=tuple(obligation_rows),
                improved_obligation_ids=tuple(improved),
                regressed_obligation_ids=tuple(regressed),
                baseline_artifact_sha256s=_semantic_artifact_hashes(baseline),
                candidate_artifact_sha256s=_semantic_artifact_hashes(candidate),
                candidate_judge_calls=1,
            )
        )

    if actual_judge_calls != expected_judge_calls:
        raise ValueError("candidate judge-call total changed")
    baseline_required = statistics.fmean(
        topic.baseline_required_coverage for topic in topics
    )
    candidate_required = statistics.fmean(
        topic.candidate_required_coverage for topic in topics
    )
    baseline_strict = statistics.fmean(
        topic.baseline_strict_full_rate for topic in topics
    )
    candidate_strict = statistics.fmean(
        topic.candidate_strict_full_rate for topic in topics
    )
    failures = _semantic_gate_failures(
        topics,
        baseline_required=baseline_required,
        candidate_required=candidate_required,
        baseline_strict=baseline_strict,
        candidate_strict=candidate_strict,
    )

    return SemanticComparison(
        schema_version=SEMANTIC_COMPARISON_SCHEMA,
        diagnostic_scope=(
            "Planner-derived paired nugget coverage diagnostic; not organizer ground truth"
        ),
        baseline_handoff_sha256=_digest(Path(baseline_handoff_path).read_bytes()),
        candidate_handoff_sha256=_digest(Path(candidate_handoff_path).read_bytes()),
        topic_ids=ordered_topic_ids,
        topics=tuple(topics),
        baseline_topic_macro_required_coverage=baseline_required,
        candidate_topic_macro_required_coverage=candidate_required,
        topic_macro_required_coverage_delta=candidate_required - baseline_required,
        baseline_topic_macro_strict_full_rate=baseline_strict,
        candidate_topic_macro_strict_full_rate=candidate_strict,
        topic_macro_strict_full_rate_delta=candidate_strict - baseline_strict,
        improved_obligation_ids=tuple(
            f"{topic.topic_id}:{obligation_id}"
            for topic in topics
            for obligation_id in topic.improved_obligation_ids
        ),
        regressed_obligation_ids=tuple(
            f"{topic.topic_id}:{obligation_id}"
            for topic in topics
            for obligation_id in topic.regressed_obligation_ids
        ),
        planner_calls=0,
        candidate_judge_calls=len(topics),
        expected_candidate_judge_calls=len(topics),
        gates_passed=not failures,
        gate_failures=failures,
    )


def _comparison_bytes(
    comparison: StructuralComparison | SemanticComparison,
) -> bytes:
    return (
        json.dumps(
            asdict(comparison),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_create_only(path: Path, body: bytes) -> None:
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"conflicting validation artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_structural(args: argparse.Namespace) -> int:
    comparison = compare_structural_runs(
        baseline_output_root=args.baseline_output_root,
        candidate_output_root=args.candidate_output_root,
        document_store_root=args.document_store_root,
        baseline_handoff_path=args.baseline_handoff,
        candidate_handoff_path=args.candidate_handoff,
        topic_ids=tuple(args.topic),
    )
    output_dir = Path(args.output_dir)
    comparison_path = output_dir / "structural-comparison.json"
    comparison_body = _comparison_bytes(comparison)
    _publish_create_only(comparison_path, comparison_body)
    manifest = {
        "schema_version": STRUCTURAL_MANIFEST_SCHEMA,
        "comparison_file": comparison_path.name,
        "comparison_bytes": len(comparison_body),
        "comparison_sha256": _digest(comparison_body),
        "topic_ids": list(comparison.topic_ids),
        "gates_passed": comparison.gates_passed,
    }
    _publish_create_only(
        output_dir / "structural-comparison-manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n",
    )
    return 0 if comparison.gates_passed else 2


def _run_semantic(args: argparse.Namespace) -> int:
    comparison = compare_semantic_runs(
        baseline_handoff_path=args.baseline_handoff,
        baseline_coverage_root=args.baseline_coverage_root,
        candidate_handoff_path=args.candidate_handoff,
        candidate_coverage_root=args.candidate_coverage_root,
        topic_ids=tuple(args.topic),
    )
    output_dir = Path(args.output_dir)
    comparison_path = output_dir / "semantic-comparison.json"
    comparison_body = _comparison_bytes(comparison)
    _publish_create_only(comparison_path, comparison_body)
    manifest = {
        "schema_version": SEMANTIC_MANIFEST_SCHEMA,
        "comparison_file": comparison_path.name,
        "comparison_bytes": len(comparison_body),
        "comparison_sha256": _digest(comparison_body),
        "topic_ids": list(comparison.topic_ids),
        "planner_calls": comparison.planner_calls,
        "candidate_judge_calls": comparison.candidate_judge_calls,
        "gates_passed": comparison.gates_passed,
    }
    _publish_create_only(
        output_dir / "semantic-comparison-manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n",
    )
    return 0 if comparison.gates_passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    structural = subparsers.add_parser("structural")
    structural.add_argument("--baseline-output-root", type=Path, required=True)
    structural.add_argument("--candidate-output-root", type=Path, required=True)
    structural.add_argument("--document-store-root", type=Path, required=True)
    structural.add_argument("--baseline-handoff", type=Path, required=True)
    structural.add_argument("--candidate-handoff", type=Path, required=True)
    structural.add_argument("--output-dir", type=Path, required=True)
    structural.add_argument("--topic", action="append", required=True)
    structural.set_defaults(handler=_run_structural)
    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("--baseline-handoff", type=Path, required=True)
    semantic.add_argument("--baseline-coverage-root", type=Path, required=True)
    semantic.add_argument("--candidate-handoff", type=Path, required=True)
    semantic.add_argument("--candidate-coverage-root", type=Path, required=True)
    semantic.add_argument("--output-dir", type=Path, required=True)
    semantic.add_argument("--topic", action="append", required=True)
    semantic.set_defaults(handler=_run_semantic)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

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
import statistics
import tempfile
from typing import Any

from trec_rag.document_store import DocumentStore
from trec_rag.facet_evidence import _byte_offsets, _source_spans
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
)
from trec_rag.topic_records import TopicRecords


STRUCTURAL_COMPARISON_SCHEMA = "cached-segmentation-structural-comparison-v1"
STRUCTURAL_MANIFEST_SCHEMA = "cached-segmentation-structural-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
    selected_document_count: int
    source_document_sha256s: tuple[str, ...]
    old_line_units: TextShapeMetrics
    fixed_segmentation_units: TextShapeMetrics
    candidate_units: TextShapeMetrics
    baseline_selected_evidence: TextShapeMetrics
    candidate_selected_evidence: TextShapeMetrics
    baseline_representatives: TextShapeMetrics
    candidate_representatives: TextShapeMetrics
    baseline_claim_hints: TextShapeMetrics
    candidate_claim_hints: TextShapeMetrics
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
    aggregate_candidate_units: TextShapeMetrics
    aggregate_baseline_selected_evidence: TextShapeMetrics
    aggregate_candidate_selected_evidence: TextShapeMetrics
    aggregate_baseline_representatives: TextShapeMetrics
    aggregate_candidate_representatives: TextShapeMetrics
    aggregate_baseline_claim_hints: TextShapeMetrics
    aggregate_candidate_claim_hints: TextShapeMetrics
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


def _segmentation_texts(documents: Sequence[_SelectedDocument]) -> tuple[str, ...]:
    return tuple(
        span.text
        for document in documents
        for span in _source_spans(document.text, _byte_offsets(document.text))
    )


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
        documents = _selected_documents(
            baseline_selected_bytes,
            topic_id=topic_id,
            store=store,
        )
        loaded_candidates = load_structural_topic(
            Path(candidate_output_root),
            Path(document_store_root),
            topic_id,
        )
        if any(
            evidence.evidence_id not in loaded_candidates.candidate_ids
            for evidence in candidate_topic.evidence
        ):
            raise ValueError("candidate handoff references an unvalidated candidate")

        old_lines = tuple(
            line.strip()
            for document in documents
            for line in document.text.splitlines()
            if line.strip()
        )
        fixed_segments = _segmentation_texts(documents)
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
        failures: list[str] = []
        if fixed_segment_metrics.median_characters <= old_line_metrics.median_characters:
            failures.append("fixed segmentation median did not improve")
        if fixed_segment_metrics.short_fraction > old_line_metrics.short_fraction:
            failures.append("fixed segmentation short fraction worsened")
        if topic_id == "407":
            if fixed_segment_metrics.median_characters <= 11:
                failures.append("topic 407 median did not exceed 11")
            if fixed_segment_metrics.short_fraction >= 0.604:
                failures.append("topic 407 short fraction did not beat 0.604")

        topic_metrics.append(
            StructuralTopicMetrics(
                topic_id=topic_id,
                narrative_sha256=baseline_topic.narrative_sha256,
                selected_document_count=len(documents),
                source_document_sha256s=tuple(row.text_sha256 for row in documents),
                old_line_units=old_line_metrics,
                fixed_segmentation_units=fixed_segment_metrics,
                candidate_units=_measure_texts(loaded_candidates.texts),
                baseline_selected_evidence=baseline_evidence,
                candidate_selected_evidence=candidate_evidence,
                baseline_representatives=baseline_representatives,
                candidate_representatives=candidate_representatives,
                baseline_claim_hints=_measure_texts(
                    baseline_claim_texts
                ),
                candidate_claim_hints=_measure_texts(
                    candidate_claim_texts
                ),
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
        all_candidate_texts.extend(loaded_candidates.texts)
        all_baseline_evidence.extend(baseline_evidence_texts)
        all_candidate_evidence.extend(candidate_evidence_texts)
        all_baseline_representatives.extend(baseline_representative_texts)
        all_candidate_representatives.extend(candidate_representative_texts)
        all_baseline_claims.extend(baseline_claim_texts)
        all_candidate_claims.extend(candidate_claim_texts)

    aggregate_old_lines = _measure_texts(all_old_lines)
    aggregate_fixed_segments = _measure_texts(all_fixed_segments)
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
    if aggregate_candidates.short_fraction > aggregate_old_lines.short_fraction:
        aggregate_failures.append("aggregate candidate short fraction exceeded old lines")
    if (
        aggregate_candidate_evidence.short_fraction
        > aggregate_baseline_evidence.short_fraction
    ):
        aggregate_failures.append("aggregate selected evidence short fraction worsened")
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
        aggregate_candidate_units=aggregate_candidates,
        aggregate_baseline_selected_evidence=aggregate_baseline_evidence,
        aggregate_candidate_selected_evidence=aggregate_candidate_evidence,
        aggregate_baseline_representatives=aggregate_baseline_representatives,
        aggregate_candidate_representatives=aggregate_candidate_representatives,
        aggregate_baseline_claim_hints=aggregate_baseline_claims,
        aggregate_candidate_claim_hints=aggregate_candidate_claims,
        gates_passed=not gate_failures,
        gate_failures=gate_failures,
    )


def _comparison_bytes(comparison: StructuralComparison) -> bytes:
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
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

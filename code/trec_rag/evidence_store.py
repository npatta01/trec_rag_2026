"""Typed, strict artifact stages for extractive facet evidence.

It projects a sealed retrieval checkpoint, strictly decodes and seals evidence
JSONL, manages temporary SQLite spill, and invokes only pinned local evidence
models. Hosted-model and retrieval clients are deliberately outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from trec_rag.evidence_local import (
    LocalMiniLMSimilarity,
    MixedbreadSentencePairScorer,
    minilm_similarity_identity,
    mixedbread_sentence_scorer_identity,
)
from trec_rag.facet_evidence import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
    SCORING_NORMALIZATION_VERSION,
    SENTENCE_SPLITTER_VERSION,
    BudgetSnapshot,
    CandidateSubnarrative,
    EvidenceMember,
    ExtractiveCandidate,
    ExtractiveCandidateRequest,
    PassageProvenance,
    ScoredPassage,
    SentenceEvidence,
    SemanticCluster,
    SelectionCandidate,
    SelectionPolicy,
    SubnarrativeContext,
    SubnarrativeSelection,
    SourceSpan,
    extract_document_candidates,
    project_source_spans,
    select_subnarrative_candidates,
    validate_candidate_request,
    validate_extractive_candidate_source,
    _SourceValidationCache,
    _source_validation_cache,
    _canonical_identity,
)
from trec_rag.topics import Topic

if TYPE_CHECKING:
    from trec_rag.competition_retrieval import ValidatedDecomposition


REQUEST_SCHEMA_VERSION = "extractive_candidate_request_v1"
CONTEXT_SCHEMA_VERSION = "subnarrative_selection_context_v1"
HANDOFF_SCHEMA_VERSION = "facet_canonical_handoff_v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "extractive_candidate_manifest_v1"
SELECTION_MANIFEST_SCHEMA_VERSION = "subnarrative_selection_manifest_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TOPIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SCORING_ARTIFACTS = (
    "scoring/lane_scores.jsonl",
    "scoring/selected_documents.jsonl",
    "scoring/selection.json",
    "scoring/selected_subnarrative_scores.jsonl",
)
_SCORING_MANIFEST_FIELDS = frozenset({
    "schema_version", "phase", "topic_id", "narrative_sha256",
    "decomposition_source_sha256", "code_commit", "retriever",
    "selection_schema_version",
    "retrieval_manifest_sha256", "scorer", "rerank_depth", "selection_k",
    "selection_policy", "score_policy", "selected_set_sha256", "artifacts",
})
_SELECTED_FIELDS = frozenset({
    "topic_id", "docid", "selection_rank", "selected_from_lane",
    "selected_from_lane_rank", "text_sha256", "text",
})
_SCORE_FIELDS = frozenset({
    "topic_id", "lane_name", "semantic_query_sha256", "docid", "bm25_rank",
    "bm25_score", "aggregate_rank", "aggregate_score", "long_document_raw_logit",
    "weighted_passage_raw_logit", "within_document_span_support", "winning_passages",
    "score_representation", "text_sha256", "selection_rank", "subnarrative_id",
    "bm25_queries", "bm25_query_sha256s", "downstream_only",
})
_PASSAGE_FIELDS = frozenset({
    "chunk_index", "start_char", "end_char", "raw_logit", "weighted_rank",
})


@dataclass(frozen=True)
class HandoffArtifacts:
    requests_path: Path
    contexts_path: Path
    manifest_path: Path
    scoring_manifest_sha256: str
    resumed: bool


@dataclass(frozen=True)
class _FileReceipt:
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class _SelectedDocument:
    docid: str
    selection_rank: int
    selected_from_lane: str
    selected_from_lane_rank: int
    text_sha256: str


def materialize_candidate_inputs(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    pilot_root: Path,
    output_dir: Path,
    code_commit: str,
    official_topics_sha256: str,
) -> HandoffArtifacts:
    """Create exact request/context ledgers from one sealed scoring checkpoint."""
    _validate_identity(topic, decomposition, code_commit, official_topics_sha256)
    topic_root = Path(pilot_root) / topic.id
    score_manifest_path = topic_root / "scoring" / "complete.json"
    score_manifest_bytes = score_manifest_path.read_bytes()
    score_manifest_receipt = _FileReceipt(
        len(score_manifest_bytes), _digest(score_manifest_bytes)
    )
    score_manifest = _strict_object(score_manifest_bytes, "scoring manifest")
    receipts = _validate_scoring_manifest(
        score_manifest,
        topic,
        decomposition,
        topic_root,
        code_commit,
    )

    selected_path = topic_root / "scoring" / "selected_documents.jsonl"
    score_path = topic_root / "scoring" / "selected_subnarrative_scores.jsonl"
    selected_receipt = receipts[selected_path.relative_to(topic_root).as_posix()]
    score_receipt = receipts[score_path.relative_to(topic_root).as_posix()]
    sealed_documents = _scan_selected_documents(
        selected_path, topic, selected_receipt
    )
    selected_set_sha256 = _digest(
        json.dumps(
            [row.docid for row in sealed_documents], separators=(",", ":")
        ).encode("utf-8")
    )
    if score_manifest["selected_set_sha256"] != selected_set_sha256:
        raise ValueError("scoring manifest selected document set changed")

    output_dir = Path(output_dir)
    requests_path = output_dir / "candidate-requests.jsonl"
    contexts_path = output_dir / "selection-contexts.jsonl"
    manifest_path = output_dir / "handoff-manifest.json"
    sealed = manifest_path.exists()
    if sealed and (not requests_path.is_file() or not contexts_path.is_file()):
        raise ValueError("existing handoff artifacts are incomplete")

    if decomposition.result.used_fallback:
        if score_receipt.byte_count:
            raise ValueError("original-only fallback must not contain subnarrative scores")
        _require_receipt(score_path, score_receipt, "subnarrative score artifact")
        _require_receipt(selected_path, selected_receipt, "selected document artifact")
        scores: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
        context_rows: tuple[dict[str, object], ...] = ()
        document_count = 0
    else:
        if decomposition.result.plan is None or not decomposition.result.subnarratives:
            raise ValueError("admitted decomposition has no generated subnarratives")
        scores = _subnarrative_scores(
            score_path,
            score_receipt,
            topic,
            decomposition,
            sealed_documents,
        )
        context_rows = tuple(
            _context_row(topic, subnarrative)
            for subnarrative in decomposition.result.subnarratives
        )
        document_count = len(sealed_documents)

    request_temp: Path | None = None
    context_temp: Path | None = None
    try:
        if decomposition.result.used_fallback:
            request_temp, request_receipt = _write_jsonl_temp(requests_path, ())
            passage_count = 0
        else:
            request_temp, request_receipt, passage_count = _write_requests_temp(
                requests_path,
                selected_path,
                selected_receipt,
                topic,
                decomposition,
                sealed_documents,
                scores,
            )
        context_temp, context_receipt = _write_jsonl_temp(contexts_path, context_rows)
        _require_receipt(score_manifest_path, score_manifest_receipt, "scoring manifest")
        manifest = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "context_schema_version": CONTEXT_SCHEMA_VERSION,
            "topic_id": topic.id,
            "official_topics_sha256": official_topics_sha256,
            "narrative_sha256": decomposition.narrative_sha256,
            "decomposition_source_sha256": decomposition.source_sha256,
            "scoring_manifest_sha256": score_manifest_receipt.sha256,
            "selected_documents_sha256": selected_receipt.sha256,
            "subnarrative_scores_sha256": score_receipt.sha256,
            "code_commit": code_commit,
            "input_roles": [
                "official_topic_narrative",
                "validated_generated_decomposition",
                "sealed_scoring_checkpoint",
            ],
            "used_original_only_fallback": decomposition.result.used_fallback,
            "requests_file": requests_path.name,
            "contexts_file": contexts_path.name,
            "requests_sha256": request_receipt.sha256,
            "contexts_sha256": context_receipt.sha256,
            "document_count": document_count,
            "subnarrative_count": len(context_rows),
            "passage_count": passage_count,
            "retrieval_network_calls": 0,
            "hosted_llm_calls": 0,
        }
        manifest_bytes = _canonical_json(manifest) + b"\n"
        if sealed:
            if (
                _file_receipt(requests_path) != request_receipt
                or _file_receipt(contexts_path) != context_receipt
                or manifest_path.read_bytes() != manifest_bytes
            ):
                raise ValueError("existing handoff artifacts conflict with sealed inputs")
            return HandoffArtifacts(
                requests_path,
                contexts_path,
                manifest_path,
                score_manifest_receipt.sha256,
                True,
            )

        os.replace(request_temp, requests_path)
        request_temp = None
        os.replace(context_temp, contexts_path)
        context_temp = None
        _atomic_write(manifest_path, manifest_bytes)
        return HandoffArtifacts(
            requests_path,
            contexts_path,
            manifest_path,
            score_manifest_receipt.sha256,
            False,
        )
    finally:
        for temporary in (request_temp, context_temp):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _validate_identity(
    topic: Topic,
    decomposition: Any,
    code_commit: str,
    official_topics_sha256: str,
) -> None:
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    if not isinstance(topic.id, str) or not _SAFE_TOPIC_ID.fullmatch(topic.id):
        raise ValueError("topic ID must be a safe path component")
    if not _COMMIT.fullmatch(code_commit):
        raise ValueError("code commit must be a full lowercase SHA-1")
    if not _SHA256.fullmatch(official_topics_sha256):
        raise ValueError("official topics source must have a lowercase SHA-256")
    if (
        getattr(decomposition, "topic_id", None) != topic.id
        or getattr(decomposition, "narrative_sha256", None) != _digest(topic.narrative.encode())
        or not _SHA256.fullmatch(str(getattr(decomposition, "source_sha256", "")))
        or not hasattr(decomposition, "result")
    ):
        raise ValueError("decomposition differs from the exact official topic narrative")


def _validate_scoring_manifest(
    manifest: Mapping[str, Any],
    topic: Topic,
    decomposition: Any,
    topic_root: Path,
    code_commit: str,
) -> dict[str, _FileReceipt]:
    _require_fields(manifest, _SCORING_MANIFEST_FIELDS, "scoring manifest")
    expected = {
        "schema_version": "facet_pilot_v2",
        "selection_schema_version": "facet_pilot_selection_v2",
        "phase": "score",
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": code_commit,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"scoring manifest {key} identity changed")
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list) or len(artifact_rows) != len(_SCORING_ARTIFACTS):
        raise ValueError("scoring manifest artifact receipts are invalid")
    if [row.get("relative_path") for row in artifact_rows if isinstance(row, dict)] != list(_SCORING_ARTIFACTS):
        raise ValueError("scoring manifest artifact order changed")
    result: dict[str, _FileReceipt] = {}
    for row in artifact_rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise ValueError("scoring manifest artifact receipt is invalid")
        relative = row["relative_path"]
        if (
            not isinstance(relative, str)
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
        ):
            raise ValueError("scoring manifest artifact path is invalid")
        receipt = _file_receipt(topic_root / relative)
        if row["bytes"] != receipt.byte_count or row["sha256"] != receipt.sha256:
            raise ValueError("scoring checkpoint artifact hash changed")
        result[relative] = receipt
    return result


def _scan_selected_documents(
    path: Path,
    topic: Topic,
    expected_receipt: _FileReceipt,
) -> tuple[_SelectedDocument, ...]:
    seen: set[str] = set()
    result: list[_SelectedDocument] = []
    stream = _JsonlStream(path)
    for index, row in enumerate(stream, start=1):
        identity, _ = _selected_document(row, index, topic, seen)
        result.append(identity)
    _require_matching_receipt(
        stream.receipt, expected_receipt, "selected document artifact"
    )
    return tuple(result)


def _selected_document(
    row: Mapping[str, Any],
    index: int,
    topic: Topic,
    seen: set[str],
) -> tuple[_SelectedDocument, str]:
    _require_fields(row, _SELECTED_FIELDS, f"selected document {index}")
    docid = row["docid"]
    text = row["text"]
    if (
        row["topic_id"] != topic.id
        or not isinstance(docid, str)
        or not docid
        or docid in seen
        or not isinstance(text, str)
        or not text.strip()
        or row["text_sha256"] != _digest(text.encode())
        or row["selection_rank"] != index
        or isinstance(row["selection_rank"], bool)
        or not isinstance(row["selected_from_lane"], str)
        or not row["selected_from_lane"]
        or isinstance(row["selected_from_lane_rank"], bool)
        or not isinstance(row["selected_from_lane_rank"], int)
        or row["selected_from_lane_rank"] <= 0
    ):
        raise ValueError("selected document identity is invalid")
    seen.add(docid)
    return (
        _SelectedDocument(
            docid=docid,
            selection_rank=row["selection_rank"],
            selected_from_lane=row["selected_from_lane"],
            selected_from_lane_rank=row["selected_from_lane_rank"],
            text_sha256=row["text_sha256"],
        ),
        text,
    )


def _subnarrative_scores(
    path: Path,
    expected_receipt: _FileReceipt,
    topic: Topic,
    decomposition: Any,
    documents: tuple[_SelectedDocument, ...],
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    document_by_id = {row.docid: row for row in documents}
    subnarrative_by_id = {
        row.subnarrative_id: row for row in decomposition.result.subnarratives
    }
    result: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    stream = _JsonlStream(path)
    for index, row in enumerate(stream, start=1):
        _require_fields(row, _SCORE_FIELDS, f"subnarrative score {index}")
        doc = document_by_id.get(row["docid"])
        subnarrative = subnarrative_by_id.get(row["subnarrative_id"])
        key = (row["docid"], row["subnarrative_id"])
        if key in result:
            raise ValueError("duplicate document/subnarrative score row")
        if (
            doc is None
            or subnarrative is None
            or row["topic_id"] != topic.id
            or row["selection_rank"] != doc.selection_rank
            or row["text_sha256"] != doc.text_sha256
            or row["lane_name"] != f"subnarrative:{subnarrative.subnarrative_id}"
            or row["semantic_query_sha256"] != subnarrative.semantic_query_sha256
            or row["bm25_queries"] != list(subnarrative.bm25_queries)
            or row["bm25_query_sha256s"] != list(subnarrative.bm25_query_sha256s)
            or row["downstream_only"] is not True
            or row["score_representation"] != "raw_logits"
        ):
            raise ValueError("subnarrative score identity is invalid")
        for name in (
            "bm25_score", "aggregate_score", "long_document_raw_logit",
            "weighted_passage_raw_logit",
        ):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("subnarrative score values must be finite")
        for name in ("bm25_rank", "aggregate_rank", "within_document_span_support"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "within_document_span_support" else 1):
                raise ValueError("subnarrative score ranks are invalid")
        result[key] = _validate_passage_records(row["winning_passages"])
    _require_matching_receipt(
        stream.receipt, expected_receipt, "subnarrative score artifact"
    )
    expected_count = len(documents) * len(decomposition.result.subnarratives)
    if len(result) != expected_count or any(
        (doc.docid, subnarrative.subnarrative_id) not in result
        for doc in documents
        for subnarrative in decomposition.result.subnarratives
    ):
        raise ValueError("subnarrative scores do not form the required document-by-subnarrative set")
    return result


def _validate_passage_records(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("subnarrative score must contain winning passages")
    chunk_indexes: set[int] = set()
    ranks: list[int] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("winning passage must be an object")
        _require_fields(row, _PASSAGE_FIELDS, "winning passage")
        chunk_index = row["chunk_index"]
        start, end, rank, score = (
            row["start_char"], row["end_char"], row["weighted_rank"], row["raw_logit"]
        )
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
            or chunk_index in chunk_indexes
            or any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end, rank))
            or start < 0
            or end <= start
            or rank <= 0
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("winning passage values are invalid")
        chunk_indexes.add(chunk_index)
        ranks.append(rank)
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("winning passage ranks must be contiguous")
    return tuple(dict(row) for row in value)


def _write_requests_temp(
    output_path: Path,
    selected_path: Path,
    selected_receipt: _FileReceipt,
    topic: Topic,
    decomposition: Any,
    documents: tuple[_SelectedDocument, ...],
    scores: Mapping[tuple[str, str], tuple[dict[str, Any], ...]],
) -> tuple[Path, _FileReceipt, int]:
    subnarratives = tuple(
        CandidateSubnarrative(row.subnarrative_id, row.text)
        for row in decomposition.result.subnarratives
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    digest = sha256()
    byte_count = 0
    passage_count = 0
    seen: set[str] = set()
    stream = _JsonlStream(selected_path)
    document_index = 0
    try:
        with os.fdopen(descriptor, "wb") as sink:
            for document_index, row in enumerate(stream, start=1):
                identity, source = _selected_document(row, document_index, topic, seen)
                if document_index > len(documents) or identity != documents[document_index - 1]:
                    raise ValueError("selected document changed between validation passes")
                source_passages: list[tuple[Any, dict[str, Any]]] = []
                for subnarrative in decomposition.result.subnarratives:
                    for passage in scores[(identity.docid, subnarrative.subnarrative_id)]:
                        source_passages.append((subnarrative, passage))
                scoring_text, projected = project_source_spans(
                    source,
                    tuple(
                        (passage["start_char"], passage["end_char"])
                        for _, passage in source_passages
                    ),
                )
                scoring_sha256 = _digest(scoring_text.encode())
                passages: list[ScoredPassage] = []
                for (subnarrative, passage), (start, end, chunk) in zip(
                    source_passages, projected, strict=True
                ):
                    passages.append(ScoredPassage(
                        passage_id=(
                            f"{topic.id}:{subnarrative.subnarrative_id}:"
                            f"{identity.docid}:{passage['chunk_index']:04d}"
                        ),
                        lane_id=f"subnarrative:{subnarrative.subnarrative_id}",
                        query_id=subnarrative.semantic_query_sha256,
                        scoring_start_char=start,
                        scoring_end_char=end,
                        scoring_text_sha256=scoring_sha256,
                        chunk_text_sha256=_digest(chunk.encode()),
                        cross_encoder_score=float(passage["raw_logit"]),
                        cross_encoder_rank=passage["weighted_rank"],
                    ))
                request = ExtractiveCandidateRequest(
                    topic_id=topic.id,
                    document_id=identity.docid,
                    source=source,
                    document_sha256=identity.text_sha256,
                    scoring_text_sha256=scoring_sha256,
                    subnarratives=subnarratives,
                    passages=tuple(passages),
                )
                validate_candidate_request(request)
                body = _canonical_json(_request_row(request)) + b"\n"
                sink.write(body)
                digest.update(body)
                byte_count += len(body)
                passage_count += len(passages)
            if document_index != len(documents):
                raise ValueError("selected document changed between validation passes")
            _require_matching_receipt(
                stream.receipt, selected_receipt, "selected document artifact"
            )
            sink.flush()
            os.fsync(sink.fileno())
        return temporary, _FileReceipt(byte_count, digest.hexdigest()), passage_count
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

def _request_row(request: ExtractiveCandidateRequest) -> dict[str, object]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "topic_id": request.topic_id,
        "document_id": request.document_id,
        "source": request.source,
        "document_sha256": request.document_sha256,
        "scoring_text_sha256": request.scoring_text_sha256,
        "subnarratives": [
            {"subnarrative_id": row.subnarrative_id, "text": row.text}
            for row in request.subnarratives
        ],
        "passages": [
            {
                "passage_id": row.passage_id,
                "lane_id": row.lane_id,
                "query_id": row.query_id,
                "scoring_start_char": row.scoring_start_char,
                "scoring_end_char": row.scoring_end_char,
                "scoring_text_sha256": row.scoring_text_sha256,
                "chunk_text_sha256": row.chunk_text_sha256,
                "cross_encoder_score": row.cross_encoder_score,
                "cross_encoder_rank": row.cross_encoder_rank,
            }
            for row in request.passages
        ],
    }


def _context_row(topic: Topic, subnarrative: Any) -> dict[str, object]:
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "topic_id": topic.id,
        "official_narrative": topic.narrative,
        "official_narrative_sha256": _digest(topic.narrative.encode()),
        "subnarrative_id": subnarrative.subnarrative_id,
        "subnarrative_text": subnarrative.text,
        "subnarrative_sha256": subnarrative.semantic_query_sha256,
    }


class _JsonlStream:
    """One-use streaming strict-JSONL reader with a final byte receipt."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.receipt: _FileReceipt | None = None
        self._iterated = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._iterated:
            raise RuntimeError("JSONL stream can only be consumed once")
        self._iterated = True
        digest = sha256()
        byte_count = 0
        with self.path.open("rb") as source:
            for line_number, raw in enumerate(source, start=1):
                digest.update(raw)
                byte_count += len(raw)
                if not raw.strip():
                    raise ValueError(f"{self.path}:{line_number}: blank JSONL row")
                yield _strict_object(raw, f"{self.path}:{line_number}")
        self.receipt = _FileReceipt(byte_count, digest.hexdigest())


def _strict_object(source: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value


def _require_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the sealed contract")


def _write_jsonl_temp(
    output_path: Path,
    rows: tuple[Mapping[str, object], ...],
) -> tuple[Path, _FileReceipt]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    digest = sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as sink:
            for row in rows:
                body = _canonical_json(row) + b"\n"
                sink.write(body)
                digest.update(body)
                byte_count += len(body)
            sink.flush()
            os.fsync(sink.fileno())
        return temporary, _FileReceipt(byte_count, digest.hexdigest())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _file_receipt(path: Path) -> _FileReceipt:
    digest = sha256()
    byte_count = 0
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return _FileReceipt(byte_count, digest.hexdigest())


def _require_receipt(path: Path, expected: _FileReceipt, label: str) -> None:
    _require_matching_receipt(_file_receipt(path), expected, label)


def _require_matching_receipt(
    actual: _FileReceipt | None,
    expected: _FileReceipt,
    label: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{label} hash changed")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _jsonable(candidate: ExtractiveCandidate) -> dict[str, object]:
    return asdict(candidate)


def write_candidate_jsonl(path: str | Path, candidates: Iterable[ExtractiveCandidate]) -> str:
    """Atomically write canonical candidate JSONL and return its byte SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spool_descriptor, spool_path = tempfile.mkstemp(prefix=f".{target.name}.sort.", dir=target.parent)
    os.close(spool_descriptor)
    temporary: str | None = None
    try:
        with sqlite3.connect(spool_path) as spool:
            spool.execute("PRAGMA journal_mode=OFF")
            spool.execute("PRAGMA synchronous=OFF")
            spool.execute("PRAGMA temp_store=FILE")
            spool.execute("PRAGMA cache_size=-2048")
            spool.execute(
                "CREATE TABLE candidates ("
                "topic_id TEXT NOT NULL, docid TEXT NOT NULL, "
                "subnarrative_id TEXT NOT NULL, candidate_id TEXT NOT NULL UNIQUE, "
                "body BLOB NOT NULL, "
                "PRIMARY KEY (topic_id, docid, subnarrative_id, candidate_id)) "
                "WITHOUT ROWID"
            )
            for candidate in candidates:
                line = json.dumps(
                    _jsonable(candidate),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                try:
                    spool.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                        (
                            candidate.topic_id,
                            candidate.docid,
                            candidate.subnarrative_id,
                            candidate.candidate_nugget_id,
                            line,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("candidate IDs must be unique before writing") from exc
            spool.commit()
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            digest = sha256()
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    for (line,) in spool.execute(
                        "SELECT body FROM candidates ORDER BY "
                        "topic_id, docid, subnarrative_id, candidate_id"
                    ):
                        handle.write(line)
                        digest.update(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                temporary = None
            except BaseException:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        try:
            os.unlink(spool_path)
        except FileNotFoundError:
            pass
    return digest.hexdigest()


_CANDIDATE_REQUEST_FIELDS = frozenset({
    "schema_version", "topic_id", "document_id", "source", "document_sha256",
    "scoring_text_sha256", "subnarratives", "passages",
})
_CANDIDATE_SUBNARRATIVE_FIELDS = frozenset({"subnarrative_id", "text"})
_CANDIDATE_PASSAGE_FIELDS = frozenset({
    "passage_id", "lane_id", "query_id", "scoring_start_char", "scoring_end_char",
    "scoring_text_sha256", "chunk_text_sha256", "cross_encoder_score", "cross_encoder_rank",
})


@dataclass(frozen=True)
class _CandidateRequestAudit:
    source_sha256: str
    document_count: int
    unique_subnarrative_count: int


@dataclass(frozen=True)
class CandidateArtifacts:
    candidates_path: Path
    manifest_path: Path


def generate_candidate_artifacts(
    paths: HandoffArtifacts,
    *,
    score_cache_root: Path,
    device: str,
    scorer: object | None = None,
) -> CandidateArtifacts:
    """Generate and seal candidate JSONL from one typed handoff."""
    if not isinstance(paths, HandoffArtifacts):
        raise TypeError("paths must be HandoffArtifacts")
    input_path = Path(paths.requests_path)
    canonical_root = input_path.parent.parent
    output_path = canonical_root / "candidates.jsonl"
    manifest_path = canonical_root / "candidate-manifest.json"
    resolved_paths = (
        input_path.resolve(), output_path.resolve(), manifest_path.resolve()
    )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("candidate input and output paths must name different files")
    audit = _audit_candidate_requests(input_path)
    if scorer is None and audit.document_count:
        scorer = MixedbreadSentencePairScorer(
            score_cache_root=Path(score_cache_root),
            device=device,
        )
    identity = (
        mixedbread_sentence_scorer_identity()
        if scorer is None
        else _candidate_scorer_identity(scorer)
    )
    run_identity = {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "source_sha256": audit.source_sha256,
        "scorer": identity,
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": SCORING_NORMALIZATION_VERSION,
    }
    _reject_conflicting_candidate_outputs(output_path, manifest_path, run_identity)
    candidate_count = 0
    second_pass_sha256 = sha256()

    def candidate_stream():
        nonlocal candidate_count
        for request in _iter_candidate_requests(
            input_path, digest=second_pass_sha256
        ):
            if scorer is None:
                raise ValueError(
                    "request JSONL changed while candidates were being generated"
                )
            for candidate in extract_document_candidates(request, scorer):
                candidate_count += 1
                yield candidate
        if second_pass_sha256.hexdigest() != audit.source_sha256:
            raise ValueError(
                "request JSONL changed while candidates were being generated"
            )

    candidates_sha256 = write_candidate_jsonl(output_path, candidate_stream())
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        **run_identity,
        "source_file": input_path.name,
        "candidate_file": output_path.name,
        "input_sha256": run_identity["source_sha256"],
        "candidates_sha256": candidates_sha256,
        "output_sha256": candidates_sha256,
        "document_count": audit.document_count,
        "unique_document_count": audit.document_count,
        "unique_subnarrative_count": audit.unique_subnarrative_count,
        "failure_count": 0,
        "candidate_count": candidate_count,
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }
    _atomic_candidate_json_write(manifest_path, manifest)
    return CandidateArtifacts(output_path, manifest_path)


def _iter_candidate_requests(path: Path, *, digest: Any | None = None) -> Iterator[ExtractiveCandidateRequest]:
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if digest is not None:
                digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL rows are not allowed")
            value = _strict_candidate_json_object(raw_line, path, line_number)
            _require_candidate_fields(value, _CANDIDATE_REQUEST_FIELDS, f"{path}:{line_number}: request")
            if value["schema_version"] != REQUEST_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported request schema version")
            subnarratives = value["subnarratives"]
            passages = value["passages"]
            if not isinstance(subnarratives, list) or not isinstance(passages, list):
                raise ValueError(f"{path}:{line_number}: subnarratives and passages must be arrays")
            try:
                request = ExtractiveCandidateRequest(
                    topic_id=value["topic_id"],
                    document_id=value["document_id"],
                    source=value["source"],
                    document_sha256=value["document_sha256"],
                    scoring_text_sha256=value["scoring_text_sha256"],
                    subnarratives=tuple(_candidate_subnarrative(row, path, line_number) for row in subnarratives),
                    passages=tuple(_candidate_passage(row, path, line_number) for row in passages),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid request: {exc}") from exc
            _validate_candidate_source_hashes(request, path, line_number)
            yield request


def _audit_candidate_requests(path: Path) -> _CandidateRequestAudit:
    descriptor, audit_path = tempfile.mkstemp(prefix=".extractive-candidate-request-audit.")
    os.close(descriptor)
    digest = sha256()
    document_count = 0
    try:
        with sqlite3.connect(audit_path) as audit:
            audit.execute("PRAGMA journal_mode=OFF")
            audit.execute("PRAGMA synchronous=OFF")
            audit.execute("PRAGMA temp_store=FILE")
            audit.execute("PRAGMA cache_size=-2048")
            audit.execute("CREATE TABLE documents (document_id TEXT PRIMARY KEY)")
            audit.execute(
                "CREATE TABLE subnarratives ("
                "topic_id TEXT NOT NULL, subnarrative_id TEXT NOT NULL, "
                "text_sha256 TEXT NOT NULL, "
                "PRIMARY KEY (topic_id, subnarrative_id, text_sha256))"
            )
            for line_number, request in enumerate(_iter_candidate_requests(path, digest=digest), start=1):
                try:
                    audit.execute("INSERT INTO documents VALUES (?)", (request.document_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate document_id {request.document_id!r}"
                    ) from exc
                audit.executemany(
                    "INSERT OR IGNORE INTO subnarratives VALUES (?, ?, ?)",
                    (
                        (request.topic_id, subnarrative.subnarrative_id, subnarrative.text_sha256)
                        for subnarrative in request.subnarratives
                    ),
                )
                document_count += 1
            unique_subnarrative_count = audit.execute(
                "SELECT COUNT(*) FROM subnarratives"
            ).fetchone()[0]
    finally:
        try:
            os.unlink(audit_path)
        except FileNotFoundError:
            pass
    return _CandidateRequestAudit(
        source_sha256=digest.hexdigest(),
        document_count=document_count,
        unique_subnarrative_count=unique_subnarrative_count,
    )


def _strict_candidate_json_object(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw_line.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: JSONL rows must be objects")
    return value


def _require_candidate_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label} has " + "; ".join(details) + " fields")


def _candidate_subnarrative(value: Any, path: Path, line_number: int) -> CandidateSubnarrative:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: subnarratives must contain objects")
    _require_candidate_fields(value, _CANDIDATE_SUBNARRATIVE_FIELDS, f"{path}:{line_number}: subnarrative")
    return CandidateSubnarrative(**value)


def _candidate_passage(value: Any, path: Path, line_number: int) -> ScoredPassage:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: passages must contain objects")
    _require_candidate_fields(value, _CANDIDATE_PASSAGE_FIELDS, f"{path}:{line_number}: passage")
    return ScoredPassage(**value)


def _validate_candidate_source_hashes(request: ExtractiveCandidateRequest, path: Path, line_number: int) -> None:
    document_sha256 = sha256(request.source.encode("utf-8")).hexdigest()
    scoring_sha256 = sha256(" ".join(request.source.split()).encode("utf-8")).hexdigest()
    if request.document_sha256 != document_sha256:
        raise ValueError(f"{path}:{line_number}: conflicting source document_sha256")
    if request.scoring_text_sha256 != scoring_sha256:
        raise ValueError(f"{path}:{line_number}: conflicting source scoring_text_sha256")


def _candidate_scorer_identity(scorer: Any) -> dict[str, object]:
    identity = getattr(scorer, "identity", None)
    if not isinstance(identity, dict):
        raise TypeError("scorer must expose an identity dictionary")
    required = {
        "model", "model_revision", "backend_version", "score_representation",
        "inference_dtype", "score_kind", "sentence_max_length",
    }
    if set(identity) != required:
        raise ValueError("scorer identity fields differ from the local sentence scorer contract")
    return dict(identity)


def _reject_conflicting_candidate_outputs(
    output: Path,
    manifest: Path,
    identity: Mapping[str, object],
) -> None:
    if not manifest.exists():
        return
    if not output.is_file() or not manifest.is_file():
        raise ValueError("a sealed candidate manifest requires its output file")
    try:
        existing = _strict_candidate_json_object(manifest.read_bytes(), manifest, 1)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if existing.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION or any(
        existing.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("existing output manifest identity differs from this run")


def _atomic_candidate_json_write(path: Path, value: Mapping[str, object]) -> None:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

_CONTEXT_FIELDS = frozenset({
    "schema_version", "topic_id", "official_narrative", "official_narrative_sha256",
    "subnarrative_id", "subnarrative_text", "subnarrative_sha256",
})
_CANDIDATE_FIELDS = frozenset({
    "schema_version", "topic_id", "docid", "subnarrative_id", "candidate_nugget_id",
    "nugget_type", "candidate_kind", "text", "evidence_sentences", "matched_paragraph",
    "context_before", "context_after", "passages", "sentence_cross_encoder_score",
    "rank_within_document_subnarrative", "document_sha256", "scoring_text_sha256",
    "subnarrative_sha256", "sentence_splitter_version",
})
_CANDIDATE_MANIFEST_FIELDS = frozenset({
    "schema_version", "request_schema_version", "candidate_schema_version", "source_sha256",
    "scorer", "sentence_splitter_version", "scoring_normalization_version", "source_file",
    "candidate_file", "input_sha256", "candidates_sha256", "output_sha256", "document_count",
    "unique_document_count", "unique_subnarrative_count", "failure_count", "candidate_count",
    "retrieval_network_calls", "hosted_llm_calls",
})
_SCORER_IDENTITY_FIELDS = frozenset({
    "model", "model_revision", "backend_version", "score_representation",
    "inference_dtype", "score_kind", "sentence_max_length",
})
_SELECTION_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def decode_extractive_candidate(
    source: bytes | str,
    *,
    document_text: str,
    subnarrative_text: str,
    source_cache: _SourceValidationCache | None = None,
) -> ExtractiveCandidate:
    """Strictly decode and source-validate one sealed candidate row."""
    from trec_rag.facet_extraction import _decode_json

    if isinstance(source, str):
        source = source.encode("utf-8")
    if not isinstance(source, bytes):
        raise TypeError("candidate source must be bytes or text")
    root = _decode_json(source, "extractive candidate")
    _exact_fields(root, set(_CANDIDATE_FIELDS), "extractive candidate")

    def mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
        row = _selection_mapping(value, label)
        _exact_fields(row, fields, label)
        return row

    span_fields = {
        "text", "start_char", "end_char", "start_byte", "end_byte", "text_sha256",
    }

    def span(value: object, label: str) -> SourceSpan:
        return SourceSpan(**mapping(value, span_fields, label))

    evidence_raw = root["evidence_sentences"]
    passages_raw = root["passages"]
    if not isinstance(evidence_raw, list) or not isinstance(passages_raw, list):
        raise ValueError("extractive candidate evidence and passages must be arrays")
    evidence = tuple(
        SentenceEvidence(
            **mapping(
                value,
                span_fields | {"cross_encoder_score"},
                "candidate evidence sentence",
            )
        )
        for value in evidence_raw
    )
    passage_fields = {
        "passage_id", "lane_id", "query_id", "scoring_start_char",
        "scoring_end_char", "source_start_char", "source_end_char",
        "source_start_byte", "source_end_byte", "source_text",
        "source_text_sha256", "scoring_text_sha256", "chunk_text_sha256",
        "normalization_version", "cross_encoder_score", "cross_encoder_rank",
    }
    passages = tuple(
        PassageProvenance(**mapping(value, passage_fields, "candidate passage"))
        for value in passages_raw
    )
    before_raw, after_raw = root["context_before"], root["context_after"]
    candidate = ExtractiveCandidate(
        schema_version=root["schema_version"],
        topic_id=root["topic_id"],
        docid=root["docid"],
        subnarrative_id=root["subnarrative_id"],
        candidate_nugget_id=root["candidate_nugget_id"],
        nugget_type=root["nugget_type"],
        candidate_kind=root["candidate_kind"],
        text=root["text"],
        evidence_sentences=evidence,
        matched_paragraph=span(root["matched_paragraph"], "candidate paragraph"),
        context_before=(
            None if before_raw is None else span(before_raw, "candidate context_before")
        ),
        context_after=(
            None if after_raw is None else span(after_raw, "candidate context_after")
        ),
        passages=passages,
        sentence_cross_encoder_score=root["sentence_cross_encoder_score"],
        rank_within_document_subnarrative=root[
            "rank_within_document_subnarrative"
        ],
        document_sha256=root["document_sha256"],
        scoring_text_sha256=root["scoring_text_sha256"],
        subnarrative_sha256=root["subnarrative_sha256"],
        sentence_splitter_version=root["sentence_splitter_version"],
    )
    validate_extractive_candidate_source(
        candidate,
        source=document_text,
        subnarrative_text=subnarrative_text,
        _source_cache=source_cache,
    )
    return candidate


def load_validated_candidate_artifacts(
    candidates_path: Path,
    manifest_path: Path,
    *,
    documents: Mapping[str, str],
    subnarratives: Mapping[str, str],
    required_candidate_keys: frozenset[tuple[str, str]] | None = None,
) -> Mapping[tuple[str, str], ExtractiveCandidate]:
    """Stream a sealed ledger and source-validate the candidates a caller needs."""
    candidates_path, manifest_path = Path(candidates_path), Path(manifest_path)
    manifest = _strict_selection_json_object(
        manifest_path.read_bytes(), manifest_path, 1
    )
    _require_selection_fields(
        manifest, _CANDIDATE_MANIFEST_FIELDS, "candidate manifest"
    )
    _validate_candidate_manifest_identity(manifest, candidates_path)
    if required_candidate_keys is not None and (
        not isinstance(required_candidate_keys, frozenset)
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
            for key in required_candidate_keys
        )
    ):
        raise TypeError("required_candidate_keys must be a frozenset of string pairs")
    digest = sha256()
    candidate_count = 0
    seen_keys: set[tuple[str, str]] = set()
    result: dict[tuple[str, str], ExtractiveCandidate] = {}
    source_caches: dict[str, _SourceValidationCache] = {}
    with candidates_path.open("rb") as candidate_file:
        for line_number, encoded_line in enumerate(candidate_file, start=1):
            digest.update(encoded_line)
            candidate_count += 1
            if not encoded_line.endswith(b"\n"):
                raise ValueError("candidate JSONL must end with LF")
            raw = encoded_line[:-1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            if not raw:
                raise ValueError("candidate JSONL contains a blank row")
            value = _strict_selection_json_object(raw, candidates_path, line_number)
            subnarrative_id = value.get("subnarrative_id")
            candidate_nugget_id = value.get("candidate_nugget_id")
            if (
                not isinstance(subnarrative_id, str)
                or not subnarrative_id
                or not isinstance(candidate_nugget_id, str)
                or not candidate_nugget_id
            ):
                raise ValueError("candidate artifact has an invalid candidate identity")
            key = (subnarrative_id, candidate_nugget_id)
            if key in seen_keys:
                raise ValueError("candidate artifact contains a duplicate candidate ID")
            seen_keys.add(key)
            docid = value.get("docid")
            document_text = documents.get(docid) if isinstance(docid, str) else None
            subnarrative_text = subnarratives.get(subnarrative_id)
            if document_text is None:
                raise ValueError(
                    "canonical evidence document is absent from selected documents"
                )
            if subnarrative_text is None:
                raise ValueError("candidate subnarrative is absent from canonical plan")
            source_cache = source_caches.get(docid)
            if source_cache is None:
                source_cache = _source_validation_cache(document_text)
                source_caches[docid] = source_cache
            candidate = decode_extractive_candidate(
                raw,
                document_text=document_text,
                subnarrative_text=subnarrative_text,
                source_cache=source_cache,
            )
            if required_candidate_keys is None or key in required_candidate_keys:
                result[key] = candidate
    _reconcile_candidate_manifest(
        manifest,
        _ScannedCandidates(candidate_count, digest.hexdigest()),
    )
    return MappingProxyType(result)


@dataclass(frozen=True)
class SelectionArtifacts:
    selections_path: Path
    manifest_path: Path


def select_evidence_artifacts(
    paths: CandidateArtifacts,
    contexts: Path,
    *,
    device: str,
    similarity: object | None = None,
    policy: SelectionPolicy | None = None,
) -> SelectionArtifacts:
    """Select and seal clustered evidence from typed candidate artifacts."""
    if not isinstance(paths, CandidateArtifacts):
        raise TypeError("paths must be CandidateArtifacts")
    candidates_path = Path(paths.candidates_path)
    candidate_manifest_path = Path(paths.manifest_path)
    contexts_path = Path(contexts)
    canonical_root = candidates_path.parent
    output_path = canonical_root / "subnarrative-selections.jsonl"
    manifest_path = canonical_root / "selection-manifest.json"
    resolved_paths = (
        candidates_path.resolve(), candidate_manifest_path.resolve(),
        contexts_path.resolve(), output_path.resolve(), manifest_path.resolve(),
    )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("all selection input and output paths must name different files")
    policy = SelectionPolicy() if policy is None else policy
    if not isinstance(policy, SelectionPolicy):
        raise TypeError("policy must be SelectionPolicy")
    embedding_batch_size = 256
    contexts_bytes = contexts_path.read_bytes()
    loaded_contexts = _load_contexts(contexts_bytes, contexts_path)
    candidate_manifest_bytes = candidate_manifest_path.read_bytes()
    candidate_manifest = _strict_selection_json_object(
        candidate_manifest_bytes, candidate_manifest_path, 1
    )
    _require_selection_fields(
        candidate_manifest, _CANDIDATE_MANIFEST_FIELDS, "candidate manifest"
    )
    _validate_candidate_manifest_identity(candidate_manifest, candidates_path)
    locked_similarity: _LockedSimilarity | None = None
    if similarity is None and loaded_contexts:
        similarity = LocalMiniLMSimilarity(
            device=device, batch_size=embedding_batch_size
        )
    if similarity is None:
        similarity_identity = minilm_similarity_identity(
            device=device, batch_size=embedding_batch_size
        )
    else:
        locked_similarity = _LockedSimilarity(similarity)
        similarity_identity = dict(locked_similarity.identity)
    loaded_candidate_projection_count = 0
    with _CandidateSpill(loaded_contexts, directory=None) as spill:
        scanned = spill.scan(candidates_path)
        _reconcile_candidate_manifest(candidate_manifest, scanned)
        selections_list: list[SubnarrativeSelection] = []
        for context in loaded_contexts:
            if locked_similarity is None:
                raise RuntimeError("similarity provider was not initialized")
            pool = spill.load_precluster_pool(context, policy.precluster_limit)
            loaded_candidate_projection_count += len(pool.candidates)
            selected = select_subnarrative_candidates(
                context, pool.candidates, locked_similarity, policy
            )
            selections_list.append(replace(
                selected,
                candidate_count=pool.candidate_count,
                exact_group_count=pool.exact_group_count,
            ))
        selections = tuple(selections_list)
    candidates_sha256 = scanned.candidates_sha256
    output_bytes = b"".join(
        _selection_canonical_json(_selection_json(selection)) + b"\n"
        for selection in selections
    )
    if any(
        _identity_dict(selection.similarity_identity) != similarity_identity
        for selection in selections
    ):
        raise ValueError("all selections must share one immutable embedding identity")
    manifest = {
        "schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_manifest_schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidates_file": candidates_path.name,
        "candidate_manifest_file": candidate_manifest_path.name,
        "contexts_file": contexts_path.name,
        "selection_file": output_path.name,
        "candidates_sha256": candidates_sha256,
        "candidate_manifest_sha256": sha256(candidate_manifest_bytes).hexdigest(),
        "contexts_sha256": sha256(contexts_bytes).hexdigest(),
        "selections_sha256": sha256(output_bytes).hexdigest(),
        "output_sha256": sha256(output_bytes).hexdigest(),
        "candidate_rows_scanned": scanned.candidate_count,
        "candidate_projection_count": sum(
            selection.candidate_count for selection in selections
        ),
        "loaded_candidate_projection_count": loaded_candidate_projection_count,
        "context_count": len(loaded_contexts),
        "selection_count": len(selections),
        "exact_group_count": sum(
            selection.exact_group_count for selection in selections
        ),
        "semantic_cluster_count": sum(
            selection.semantic_cluster_count for selection in selections
        ),
        "selected_cluster_count": sum(
            len(selection.clusters) for selection in selections
        ),
        "policy": _policy_json(policy),
        "embedding_identity": similarity_identity,
        "candidate_scorer_identity": dict(candidate_manifest["scorer"]),
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }
    manifest_bytes = _selection_canonical_json(manifest) + b"\n"
    if _validate_existing_selection(
        output_path, manifest_path, output_bytes, manifest_bytes
    ):
        return SelectionArtifacts(output_path, manifest_path)
    _atomic_selection_write(output_path, output_bytes)
    _atomic_selection_write(manifest_path, manifest_bytes)
    return SelectionArtifacts(output_path, manifest_path)


def _strict_selection_json_object(raw: bytes, path: Path, line_number: int) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: JSON must be an object")
    return value


def _require_selection_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label} has " + "; ".join(details) + " fields")


def _load_contexts(source: bytes, path: Path) -> tuple[SubnarrativeContext, ...]:
    contexts: dict[tuple[str, str], SubnarrativeContext] = {}
    narratives: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL rows are not allowed")
        value = _strict_selection_json_object(raw_line, path, line_number)
        _require_selection_fields(value, _CONTEXT_FIELDS, f"{path}:{line_number}: context")
        if value["schema_version"] != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"{path}:{line_number}: unsupported context schema version")
        context = SubnarrativeContext(
            topic_id=value["topic_id"],
            official_narrative=value["official_narrative"],
            subnarrative_id=value["subnarrative_id"],
            subnarrative_text=value["subnarrative_text"],
        )
        if value["official_narrative_sha256"] != context.official_narrative_sha256:
            raise ValueError(f"{path}:{line_number}: official narrative hash mismatch")
        if value["subnarrative_sha256"] != context.subnarrative_sha256:
            raise ValueError(f"{path}:{line_number}: subnarrative hash mismatch")
        key = (context.topic_id, context.subnarrative_id)
        if key in contexts:
            raise ValueError(f"{path}:{line_number}: duplicate topic/subnarrative context")
        previous_narrative = narratives.setdefault(
            context.topic_id, (context.official_narrative, context.official_narrative_sha256)
        )
        if previous_narrative != (context.official_narrative, context.official_narrative_sha256):
            raise ValueError(f"{path}:{line_number}: conflicting official narrative for topic")
        contexts[key] = context
    return tuple(contexts[key] for key in sorted(contexts))


def decode_subnarrative_selection(source: bytes | str) -> SubnarrativeSelection:
    """Strictly decode the canonical selection JSON into validated records."""
    from trec_rag.facet_extraction import _decode_json

    if isinstance(source, str):
        source = source.encode("utf-8")
    if not isinstance(source, bytes):
        raise TypeError("selection source must be bytes or text")
    root = _decode_json(source, "subnarrative selection")
    _exact_fields(root, {
        "schema_version", "topic_id", "official_narrative", "official_narrative_sha256",
        "subnarrative_id", "subnarrative_text", "subnarrative_sha256", "policy",
        "embedding_identity", "candidate_count", "exact_group_count", "precluster_count",
        "semantic_cluster_count", "clusters", "snapshots",
    }, "subnarrative selection")
    context = SubnarrativeContext(
        topic_id=root["topic_id"],
        official_narrative=root["official_narrative"],
        subnarrative_id=root["subnarrative_id"],
        subnarrative_text=root["subnarrative_text"],
    )
    if root["official_narrative_sha256"] != context.official_narrative_sha256:
        raise ValueError("subnarrative selection official narrative hash mismatch")
    if root["subnarrative_sha256"] != context.subnarrative_sha256:
        raise ValueError("subnarrative selection generated subnarrative hash mismatch")
    policy_raw = _selection_mapping(root["policy"], "selection policy")
    _exact_fields(policy_raw, {
        "budgets", "precluster_limit", "semantic_threshold", "mmr_lambda",
    }, "selection policy")
    budgets = policy_raw["budgets"]
    if not isinstance(budgets, list):
        raise ValueError("selection policy budgets must be an array")
    policy = SelectionPolicy(
        budgets=tuple(budgets),
        precluster_limit=policy_raw["precluster_limit"],
        semantic_threshold=policy_raw["semantic_threshold"],
        mmr_lambda=policy_raw["mmr_lambda"],
    )
    identity = _canonical_identity(root["embedding_identity"])
    clusters_raw = root["clusters"]
    if not isinstance(clusters_raw, list):
        raise ValueError("subnarrative selection clusters must be an array")
    clusters = tuple(_decode_cluster(row) for row in clusters_raw)
    snapshots_raw = root["snapshots"]
    if not isinstance(snapshots_raw, list):
        raise ValueError("subnarrative selection snapshots must be an array")
    snapshots: list[BudgetSnapshot] = []
    for value in snapshots_raw:
        row = _selection_mapping(value, "selection snapshot")
        _exact_fields(row, {"budget", "cluster_ids", "exhausted"}, "selection snapshot")
        cluster_ids = row["cluster_ids"]
        if not isinstance(cluster_ids, list):
            raise ValueError("selection snapshot cluster_ids must be an array")
        snapshots.append(BudgetSnapshot(
            budget=row["budget"],
            cluster_ids=tuple(cluster_ids),
            exhausted=row["exhausted"],
        ))
    return SubnarrativeSelection(
        schema_version=root["schema_version"],
        context=context,
        policy=policy,
        similarity_identity=identity,
        candidate_count=root["candidate_count"],
        exact_group_count=root["exact_group_count"],
        precluster_count=root["precluster_count"],
        semantic_cluster_count=root["semantic_cluster_count"],
        clusters=clusters,
        snapshots=tuple(snapshots),
    )


def _decode_cluster(value: object) -> SemanticCluster:
    row = _selection_mapping(value, "semantic cluster")
    _exact_fields(row, {
        "cluster_id", "representative_candidate_nugget_id", "representative_text",
        "representative_raw_logit", "members", "supports", "support_document_count",
    }, "semantic cluster")
    members_raw, supports_raw = row["members"], row["supports"]
    if not isinstance(members_raw, list) or not isinstance(supports_raw, list):
        raise ValueError("semantic cluster members and supports must be arrays")
    return SemanticCluster(
        cluster_id=row["cluster_id"],
        representative_candidate_nugget_id=row["representative_candidate_nugget_id"],
        representative_text=row["representative_text"],
        representative_raw_logit=row["representative_raw_logit"],
        members=tuple(_decode_member(member) for member in members_raw),
        supports=tuple(_decode_member(member) for member in supports_raw),
        support_document_count=row["support_document_count"],
    )


def _decode_member(value: object) -> EvidenceMember:
    row = _selection_mapping(value, "evidence member")
    _exact_fields(row, {
        "candidate_nugget_id", "candidate_kind", "text", "docid", "document_sha256",
        "raw_logit",
    }, "evidence member")
    return EvidenceMember(
        candidate_nugget_id=row["candidate_nugget_id"],
        candidate_kind=row["candidate_kind"],
        text=row["text"],
        docid=row["docid"],
        document_sha256=row["document_sha256"],
        raw_logit=row["raw_logit"],
    )


def _selection_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has unexpected or missing fields")


def _validate_candidate_manifest_identity(manifest: Mapping[str, Any], candidates: Path) -> None:
    expected = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "sentence_splitter_version": "exact_rules_v1",
        "scoring_normalization_version": "trec_rag_whitespace_v1",
        "candidate_file": candidates.name,
        "failure_count": 0,
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }
    for key, expected_value in expected.items():
        if manifest[key] != expected_value:
            raise ValueError(f"candidate manifest {key} identity mismatch")
    scorer = manifest["scorer"]
    if not isinstance(scorer, dict):
        raise ValueError("candidate manifest scorer identity mismatch")
    try:
        _require_selection_fields(scorer, _SCORER_IDENTITY_FIELDS, "candidate manifest scorer identity")
    except ValueError as exc:
        raise ValueError(f"candidate manifest scorer identity mismatch: {exc}") from exc
    for key in _SCORER_IDENTITY_FIELDS - {"sentence_max_length"}:
        if not isinstance(scorer[key], str) or not scorer[key]:
            raise ValueError("candidate manifest scorer identity values must be non-empty strings")
    if scorer["score_representation"] != "raw_logits" or scorer["score_kind"] != "extractive_sentence_v1":
        raise ValueError("candidate manifest scorer identity mismatch")
    if (
        isinstance(scorer["sentence_max_length"], bool)
        or not isinstance(scorer["sentence_max_length"], int)
        or scorer["sentence_max_length"] <= 0
    ):
        raise ValueError("candidate manifest scorer identity sentence_max_length must be positive")
    for key in ("candidate_count", "document_count", "unique_document_count", "unique_subnarrative_count"):
        if isinstance(manifest[key], bool) or not isinstance(manifest[key], int) or manifest[key] < 0:
            raise ValueError(f"candidate manifest {key} must be a non-negative integer")
    for key in ("source_sha256", "input_sha256", "candidates_sha256", "output_sha256"):
        if not isinstance(manifest[key], str) or not _SELECTION_SHA256.fullmatch(manifest[key]):
            raise ValueError(f"candidate manifest {key} must be a lowercase SHA-256 digest")
    if manifest["source_sha256"] != manifest["input_sha256"]:
        raise ValueError("candidate manifest source/input SHA-256 identity mismatch")


@dataclass(frozen=True)
class _ScannedCandidates:
    candidate_count: int
    candidates_sha256: str


@dataclass(frozen=True)
class _PreclusterPool:
    candidates: tuple[SelectionCandidate, ...]
    candidate_count: int
    exact_group_count: int


class _LockedSimilarity:
    """Take one identity snapshot while delegating cosine work to the provider."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.identity = MappingProxyType(_provider_identity(delegate))

    def cosine_matrix(self, texts: Sequence[str]) -> Any:
        return self._delegate.cosine_matrix(texts)


class _CandidateSpill:
    """A deleted local SQLite index for exact, bounded precluster projection."""

    def __init__(self, contexts: Sequence[SubnarrativeContext], *, directory: Path | None) -> None:
        self._contexts = {(row.topic_id, row.subnarrative_id): row for row in contexts}
        self._directory = directory
        self._path: Path | None = None
        self._database: sqlite3.Connection | None = None

    def __enter__(self) -> _CandidateSpill:
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".candidate-selection-", suffix=".sqlite3",
            dir=self._directory,
        )
        os.close(descriptor)
        self._path = Path(raw_path)
        try:
            self._database = sqlite3.connect(self._path)
            self._database.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=FILE;
                CREATE TABLE seen_candidates (
                    candidate_nugget_id TEXT PRIMARY KEY
                ) WITHOUT ROWID;
                CREATE TABLE document_identities (
                    docid TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    document_sha256 TEXT NOT NULL,
                    PRIMARY KEY (docid, topic_id, document_sha256)
                ) WITHOUT ROWID;
                CREATE TABLE subnarrative_identities (
                    topic_id TEXT NOT NULL,
                    subnarrative_id TEXT NOT NULL,
                    subnarrative_sha256 TEXT NOT NULL,
                    PRIMARY KEY (topic_id, subnarrative_id, subnarrative_sha256)
                ) WITHOUT ROWID;
                CREATE TABLE selected_candidates (
                    topic_id TEXT NOT NULL,
                    subnarrative_id TEXT NOT NULL,
                    candidate_kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    candidate_nugget_id TEXT PRIMARY KEY,
                    docid TEXT NOT NULL,
                    document_sha256 TEXT NOT NULL,
                    raw_logit REAL NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX selected_candidate_groups ON selected_candidates (
                    topic_id, subnarrative_id, candidate_kind, text
                );
                CREATE TABLE exact_groups (
                    topic_id TEXT NOT NULL,
                    subnarrative_id TEXT NOT NULL,
                    candidate_kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    representative_candidate_nugget_id TEXT NOT NULL,
                    representative_raw_logit REAL NOT NULL,
                    PRIMARY KEY (topic_id, subnarrative_id, candidate_kind, text)
                ) WITHOUT ROWID;
                CREATE TABLE chosen_groups (
                    topic_id TEXT NOT NULL,
                    subnarrative_id TEXT NOT NULL,
                    candidate_kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    group_rank INTEGER NOT NULL,
                    PRIMARY KEY (topic_id, subnarrative_id, candidate_kind, text)
                ) WITHOUT ROWID;
            """)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._database is not None:
            self._database.close()
            self._database = None
        if self._path is not None:
            for path in (self._path, Path(str(self._path) + "-wal"), Path(str(self._path) + "-shm")):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self._path = None

    @property
    def database(self) -> sqlite3.Connection:
        if self._database is None:
            raise RuntimeError("candidate spill is not open")
        return self._database

    def scan(self, path: Path) -> _ScannedCandidates:
        digest = sha256()
        database = self.database
        with path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                digest.update(raw_line)
                if not raw_line.endswith(b"\n") or not raw_line.strip():
                    raise ValueError(
                        f"{path}:{line_number}: candidate JSONL rows must be non-blank and newline-terminated"
                    )
                value = _strict_selection_json_object(raw_line, path, line_number)
                _require_selection_fields(value, _CANDIDATE_FIELDS, f"{path}:{line_number}: candidate")
                candidate = self._validated_candidate(value, path, line_number)
                subnarrative_sha256 = value["subnarrative_sha256"]
                if not isinstance(subnarrative_sha256, str) or not _SELECTION_SHA256.fullmatch(subnarrative_sha256):
                    raise ValueError(f"{path}:{line_number}: invalid candidate subnarrative hash")
                try:
                    database.execute(
                        "INSERT INTO seen_candidates VALUES (?)",
                        (candidate.candidate_nugget_id,),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate candidate ID {candidate.candidate_nugget_id!r}"
                    ) from exc
                database.execute(
                    "INSERT OR IGNORE INTO document_identities VALUES (?, ?, ?)",
                    (candidate.docid, candidate.topic_id, candidate.document_sha256),
                )
                database.execute(
                    "INSERT OR IGNORE INTO subnarrative_identities VALUES (?, ?, ?)",
                    (candidate.topic_id, candidate.subnarrative_id, subnarrative_sha256),
                )
                context = self._contexts.get((candidate.topic_id, candidate.subnarrative_id))
                if context is not None:
                    if subnarrative_sha256 != context.subnarrative_sha256:
                        raise ValueError(f"{path}:{line_number}: candidate subnarrative hash mismatch")
                    self._insert_selected(candidate)
                if line_number % 10_000 == 0:
                    database.commit()
        database.commit()
        candidate_count = self._scalar("SELECT COUNT(*) FROM seen_candidates")
        document_count = self._scalar("SELECT COUNT(DISTINCT docid) FROM document_identities")
        if self._scalar("SELECT COUNT(*) FROM document_identities") != document_count:
            raise ValueError("candidate rows contain conflicting document identities")
        subnarrative_count = self._scalar("SELECT COUNT(*) FROM subnarrative_identities")
        distinct_subnarratives = self._scalar(
            "SELECT COUNT(*) FROM (SELECT DISTINCT topic_id, subnarrative_id FROM subnarrative_identities)"
        )
        if subnarrative_count != distinct_subnarratives:
            raise ValueError("candidate rows contain conflicting subnarrative identities")
        return _ScannedCandidates(candidate_count, digest.hexdigest())

    def _validated_candidate(self, value: Mapping[str, Any], path: Path, line_number: int) -> SelectionCandidate:
        if value["schema_version"] != CANDIDATE_SCHEMA_VERSION or value["nugget_type"] != "extractive":
            raise ValueError(f"{path}:{line_number}: candidate schema identity mismatch")
        if value["sentence_splitter_version"] != "exact_rules_v1":
            raise ValueError(f"{path}:{line_number}: sentence splitter identity mismatch")
        try:
            return SelectionCandidate(
                topic_id=value["topic_id"],
                subnarrative_id=value["subnarrative_id"],
                candidate_nugget_id=value["candidate_nugget_id"],
                candidate_kind=value["candidate_kind"],
                text=value["text"],
                docid=value["docid"],
                document_sha256=value["document_sha256"],
                raw_logit=value["sentence_cross_encoder_score"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid candidate projection: {exc}") from exc

    def _insert_selected(self, candidate: SelectionCandidate) -> None:
        values = (
            candidate.topic_id, candidate.subnarrative_id, candidate.candidate_kind,
            candidate.text, candidate.candidate_nugget_id, candidate.docid,
            candidate.document_sha256, candidate.raw_logit,
        )
        self.database.execute("INSERT INTO selected_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
        self.database.execute("""
            INSERT INTO exact_groups VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, subnarrative_id, candidate_kind, text) DO UPDATE SET
                representative_candidate_nugget_id=excluded.representative_candidate_nugget_id,
                representative_raw_logit=excluded.representative_raw_logit
            WHERE excluded.representative_raw_logit > exact_groups.representative_raw_logit
               OR (
                   excluded.representative_raw_logit = exact_groups.representative_raw_logit
                   AND excluded.representative_candidate_nugget_id
                       < exact_groups.representative_candidate_nugget_id
               )
        """, (
            candidate.topic_id, candidate.subnarrative_id, candidate.candidate_kind,
            candidate.text, candidate.candidate_nugget_id, candidate.raw_logit,
        ))

    def load_precluster_pool(self, context: SubnarrativeContext, limit: int) -> _PreclusterPool:
        scope = (context.topic_id, context.subnarrative_id)
        candidate_count = self._scalar(
            "SELECT COUNT(*) FROM selected_candidates WHERE topic_id=? AND subnarrative_id=?",
            scope,
        )
        exact_group_count = self._scalar(
            "SELECT COUNT(*) FROM exact_groups WHERE topic_id=? AND subnarrative_id=?",
            scope,
        )
        groups = self.database.execute("""
            SELECT candidate_kind, text
            FROM exact_groups
            WHERE topic_id=? AND subnarrative_id=?
            ORDER BY representative_raw_logit DESC, representative_candidate_nugget_id ASC
            LIMIT ?
        """, (*scope, limit)).fetchall()
        self.database.execute("DELETE FROM chosen_groups")
        self.database.executemany(
            "INSERT INTO chosen_groups VALUES (?, ?, ?, ?, ?)",
            ((*scope, kind, text, rank) for rank, (kind, text) in enumerate(groups, start=1)),
        )
        rows = self.database.execute("""
            SELECT c.topic_id, c.subnarrative_id, c.candidate_nugget_id,
                   c.candidate_kind, c.text, c.docid, c.document_sha256, c.raw_logit
            FROM selected_candidates AS c
            JOIN chosen_groups AS g
              ON c.topic_id=g.topic_id
             AND c.subnarrative_id=g.subnarrative_id
             AND c.candidate_kind=g.candidate_kind
             AND c.text=g.text
            ORDER BY g.group_rank ASC, c.raw_logit DESC, c.candidate_nugget_id ASC
        """).fetchall()
        candidates = tuple(SelectionCandidate(*row) for row in rows)
        return _PreclusterPool(candidates, candidate_count, exact_group_count)

    def _scalar(self, query: str, parameters: Sequence[object] = ()) -> int:
        row = self.database.execute(query, tuple(parameters)).fetchone()
        if row is None or len(row) != 1 or not isinstance(row[0], int):
            raise RuntimeError("candidate spill count query failed")
        return row[0]


def _reconcile_candidate_manifest(
    manifest: Mapping[str, Any],
    scanned: _ScannedCandidates,
) -> None:
    if manifest["candidates_sha256"] != scanned.candidates_sha256 or manifest["output_sha256"] != scanned.candidates_sha256:
        raise ValueError("candidate manifest candidates_sha256 does not match candidate bytes")
    if manifest["candidate_count"] != scanned.candidate_count:
        raise ValueError("candidate manifest candidate_count does not match scanned candidate rows")


def _member_json(member: Any) -> dict[str, object]:
    return {
        "candidate_nugget_id": member.candidate_nugget_id,
        "candidate_kind": member.candidate_kind,
        "text": member.text,
        "docid": member.docid,
        "document_sha256": member.document_sha256,
        "raw_logit": member.raw_logit,
    }


def _policy_json(policy: SelectionPolicy) -> dict[str, object]:
    return {
        "budgets": list(policy.budgets),
        "precluster_limit": policy.precluster_limit,
        "semantic_threshold": policy.semantic_threshold,
        "mmr_lambda": policy.mmr_lambda,
    }


def _identity_dict(identity: Sequence[tuple[str, object]]) -> dict[str, object]:
    return {key: value for key, value in identity}


def _provider_identity(similarity: Any) -> dict[str, object]:
    value = getattr(similarity, "identity", None)
    if not isinstance(value, Mapping):
        raise ValueError("similarity provider must expose an identity mapping")
    return dict(value)


def _selection_json(selection: SubnarrativeSelection) -> dict[str, object]:
    context = selection.context
    return {
        "schema_version": selection.schema_version,
        "topic_id": context.topic_id,
        "official_narrative": context.official_narrative,
        "official_narrative_sha256": context.official_narrative_sha256,
        "subnarrative_id": context.subnarrative_id,
        "subnarrative_text": context.subnarrative_text,
        "subnarrative_sha256": context.subnarrative_sha256,
        "policy": _policy_json(selection.policy),
        "embedding_identity": _identity_dict(selection.similarity_identity),
        "candidate_count": selection.candidate_count,
        "exact_group_count": selection.exact_group_count,
        "precluster_count": selection.precluster_count,
        "semantic_cluster_count": selection.semantic_cluster_count,
        "clusters": [{
            "cluster_id": cluster.cluster_id,
            "representative_candidate_nugget_id": cluster.representative_candidate_nugget_id,
            "representative_text": cluster.representative_text,
            "representative_raw_logit": cluster.representative_raw_logit,
            "members": [_member_json(member) for member in cluster.members],
            "supports": [_member_json(member) for member in cluster.supports],
            "support_document_count": cluster.support_document_count,
        } for cluster in selection.clusters],
        "snapshots": [{
            "budget": snapshot.budget,
            "cluster_ids": list(snapshot.cluster_ids),
            "exhausted": snapshot.exhausted,
        } for snapshot in selection.snapshots],
    }


def _selection_canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _validate_existing_selection(
    output: Path,
    manifest: Path,
    expected_output: bytes,
    expected_manifest: bytes,
) -> bool:
    if not manifest.exists():
        return False
    if not output.is_file() or not manifest.is_file():
        raise ValueError("existing selection run conflicts with requested run")
    try:
        if output.read_bytes() != expected_output or manifest.read_bytes() != expected_manifest:
            raise ValueError
    except (OSError, ValueError) as exc:
        raise ValueError("existing selection run conflicts with requested run") from exc
    return True


def _atomic_selection_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

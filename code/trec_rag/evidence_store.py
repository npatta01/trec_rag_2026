"""Typed, strict artifact stages for extractive facet evidence.

It projects a sealed retrieval checkpoint, strictly decodes and seals evidence
JSONL, and invokes only pinned local evidence models. Hosted-model and
retrieval clients are deliberately outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
from trec_rag.document_store import DocumentStore, DocumentStoreIntegrityError
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
    _canonical_identity,
)
from trec_rag.topic_records import (
    CANDIDATE_STAGE,
    FacetRecord,
    TOPIC_RECORDS_SCHEMA_VERSION,
    PublishedTopicRecords,
    TopicRecords,
    TopicRecordsBuilder,
    ValidatedTopicRecords,
)
from trec_rag.topic_passage_search import PassageSearchResult, SourceDocument, SourcePassage
from trec_rag.topics import Topic

if TYPE_CHECKING:
    from trec_rag.competition_retrieval import ValidatedDecomposition


REQUEST_SCHEMA_VERSION = "extractive_candidate_request_v1"
PASSAGE_REQUEST_SCHEMA_VERSION = "extractive_candidate_request_v2"
_SUPPORTED_REQUEST_SCHEMA_VERSIONS = frozenset({
    REQUEST_SCHEMA_VERSION,
    PASSAGE_REQUEST_SCHEMA_VERSION,
})
CONTEXT_SCHEMA_VERSION = "subnarrative_selection_context_v1"
HANDOFF_SCHEMA_VERSION = "facet_canonical_handoff_v1"
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
    "selection_policy", "selection_scope", "score_policy", "selected_set_sha256",
    "passage_search",
    "artifacts",
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
    passage_results: Sequence[PassageSearchResult] = (),
    document_store_root: Path | None = None,
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
        if passage_results:
            scores = {}
        else:
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
        document_count = (
            len({
                passage.docid
                for result in passage_results
                if result.query.primary_subnarrative_id != "original"
                for passage in result.passages
            })
            if passage_results
            else len(sealed_documents)
        )

    request_schema_version = (
        PASSAGE_REQUEST_SCHEMA_VERSION if passage_results else REQUEST_SCHEMA_VERSION
    )
    request_temp: Path | None = None
    context_temp: Path | None = None
    try:
        if decomposition.result.used_fallback:
            request_temp, request_receipt = _write_jsonl_temp(requests_path, ())
            passage_count = 0
        elif passage_results:
            if document_store_root is None:
                raise ValueError(
                    "document_store_root is required for passage-first handoff"
                )
            request_temp, request_receipt, passage_count = _write_requests_from_passages(
                requests_path,
                topic,
                decomposition,
                passage_results,
                document_store_root=Path(document_store_root),
            )
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
            "request_schema_version": request_schema_version,
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
    if manifest.get("selection_scope") != (
        "internal_fixed_path_projection_not_final_submission"
    ):
        raise ValueError("scoring manifest selection scope is not internal-only")
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


def _write_requests_from_passages(
    output_path: Path,
    topic: Topic,
    decomposition: Any,
    passage_results: Sequence[PassageSearchResult],
    *,
    document_store_root: Path,
) -> tuple[Path, _FileReceipt, int]:
    """Build one facet request per sealed semantic passage group.

    The fixed v2 handoff population is the shared top-100 passage set.  The
    round-robin selected-document artifact is retained as retrieval metadata,
    but never filters these semantic evidence requests.
    """
    subnarratives = {
        row.subnarrative_id: CandidateSubnarrative(row.subnarrative_id, row.text)
        for row in decomposition.result.subnarratives
    }
    documents: dict[str, SourceDocument] = {}
    groups: dict[tuple[str, str], list[tuple[SourcePassage, str]]] = {}
    for result in passage_results:
        subnarrative_id = result.query.primary_subnarrative_id
        if subnarrative_id == "original":
            continue
        if subnarrative_id not in subnarratives:
            raise ValueError("semantic passage result is not in the decomposition")
        by_docid = {document.docid: document for document in result.documents}
        for passage in result.passages:
            document = by_docid.get(passage.docid)
            if document is None or passage.content_sha256 != document.content_sha256:
                raise ValueError("sealed passage source document identity changed")
            previous = documents.setdefault(passage.docid, document)
            if previous.content_sha256 != document.content_sha256:
                raise ValueError("sealed passage source document conflicts across lanes")
            groups.setdefault((subnarrative_id, passage.docid), []).append(
                (passage, result.query.query_id)
            )
    if not groups:
        temporary, receipt = _write_jsonl_temp(output_path, ())
        return temporary, receipt, 0

    store = DocumentStore(document_store_root)
    source_cache: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    passage_count = 0
    for (subnarrative_id, docid), passages in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        document = documents[docid]
        source = source_cache.get(document.content_sha256)
        if source is None:
            source = store.read_text(document.content_sha256)
            if not source.strip() or _digest(source.encode()) != document.content_sha256:
                raise ValueError("topic CAS document does not match source document")
            source_cache[document.content_sha256] = source
        for passage, _ in passages:
            if (
                source[passage.start_char : passage.end_char] != passage.text
                or _digest(passage.text.encode()) != passage.text_sha256
                or len(source[: passage.start_char].encode()) != passage.start_byte
                or len(source[: passage.end_char].encode()) != passage.end_byte
            ):
                raise ValueError("sealed passage offsets or text do not match topic CAS")
        scoring_text, projected = project_source_spans(
            source,
            tuple(
                (passage.start_char, passage.end_char)
                for passage, _ in passages
            ),
        )
        scoring_sha256 = _digest(scoring_text.encode())
        scored_passages = tuple(
            ScoredPassage(
                passage_id=passage.passage_id,
                lane_id=f"subnarrative:{subnarrative_id}",
                query_id=query_id,
                scoring_start_char=start,
                scoring_end_char=end,
                scoring_text_sha256=scoring_sha256,
                chunk_text_sha256=_digest(chunk.encode()),
                cross_encoder_score=passage.raw_logit,
                cross_encoder_rank=passage.rank,
            )
            for (passage, query_id), (start, end, chunk) in zip(
                passages,
                projected,
                strict=True,
            )
        )
        request = ExtractiveCandidateRequest(
            topic_id=topic.id,
            document_id=docid,
            source=source,
            document_sha256=document.content_sha256,
            scoring_text_sha256=scoring_sha256,
            subnarratives=(subnarratives[subnarrative_id],),
            passages=scored_passages,
        )
        validate_candidate_request(request)
        rows.append(_passage_request_row(request, passages))
        passage_count += len(scored_passages)

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
        return temporary, _FileReceipt(byte_count, digest.hexdigest()), passage_count
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


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
                source_passages = [
                    (subnarrative, passage)
                    for subnarrative in subnarratives
                    for passage in scores[(identity.docid, subnarrative.subnarrative_id)]
                ]
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
                        query_id=subnarrative.text_sha256,
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


def _passage_request_row(
    request: ExtractiveCandidateRequest,
    source_passages: Sequence[tuple[SourcePassage, str]],
) -> dict[str, object]:
    """Encode a CAS-backed request without serializing document text."""
    if len(request.subnarratives) != 1:
        raise ValueError("passage-first requests must contain exactly one facet")
    if len(source_passages) != len(request.passages):
        raise ValueError("passage-first request geometry changed during projection")
    facet = request.subnarratives[0]
    return {
        "schema_version": PASSAGE_REQUEST_SCHEMA_VERSION,
        "topic_id": request.topic_id,
        "document_id": request.document_id,
        "content_sha256": request.document_sha256,
        "scoring_text_sha256": request.scoring_text_sha256,
        "facet": {
            "subnarrative_id": facet.subnarrative_id,
            "text": facet.text,
        },
        "passages": [
            {
                "passage_id": scored.passage_id,
                "lane_id": scored.lane_id,
                "query_id": query_id,
                "source_start_char": source.start_char,
                "source_end_char": source.end_char,
                "source_start_byte": source.start_byte,
                "source_end_byte": source.end_byte,
                "source_text_sha256": source.text_sha256,
                "scoring_start_char": scored.scoring_start_char,
                "scoring_end_char": scored.scoring_end_char,
                "scoring_text_sha256": scored.scoring_text_sha256,
                "chunk_text_sha256": scored.chunk_text_sha256,
                "cross_encoder_score": scored.cross_encoder_score,
                "cross_encoder_rank": scored.cross_encoder_rank,
            }
            for (source, query_id), scored in zip(
                source_passages,
                request.passages,
                strict=True,
            )
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
_PASSAGE_REQUEST_FIELDS = frozenset({
    "schema_version", "topic_id", "document_id", "content_sha256",
    "scoring_text_sha256", "facet", "passages",
})
_CANDIDATE_SUBNARRATIVE_FIELDS = frozenset({"subnarrative_id", "text"})
_CANDIDATE_PASSAGE_FIELDS = frozenset({
    "passage_id", "lane_id", "query_id", "scoring_start_char", "scoring_end_char",
    "scoring_text_sha256", "chunk_text_sha256", "cross_encoder_score", "cross_encoder_rank",
})
_PASSAGE_REQUEST_PASSAGE_FIELDS = frozenset({
    "passage_id", "lane_id", "query_id", "source_start_char", "source_end_char",
    "source_start_byte", "source_end_byte", "source_text_sha256",
    "scoring_start_char", "scoring_end_char", "scoring_text_sha256",
    "chunk_text_sha256", "cross_encoder_score", "cross_encoder_rank",
})


@dataclass(frozen=True)
class _CandidateRequestAudit:
    source_sha256: str
    document_count: int
    unique_subnarrative_count: int


@dataclass(frozen=True)
class CandidateArtifacts:
    records_path: Path
    manifest_path: Path
    document_store_root: Path
    # Process-local validation state; it must never enter serialized artifacts.
    validation_session: ValidatedTopicRecords | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def _fixed_retrieval_completion(
    passage_results: Sequence[PassageSearchResult],
    *,
    admitted_document_count: int,
) -> tuple[str, str]:
    """Return the explicit topic completion pair sealed by fixed retrieval."""
    if type(admitted_document_count) is not int or admitted_document_count < 0:
        raise ValueError("admitted_document_count must be a non-negative integer")
    results = tuple(passage_results)
    if any(not isinstance(result, PassageSearchResult) for result in results):
        raise TypeError("passage_results must contain PassageSearchResult values")
    incomplete_reasons: set[str] = set()
    for result in results:
        if result.status == "incomplete":
            if result.stopping_reason is None:
                raise ValueError("incomplete passage search is missing a stopping reason")
            incomplete_reasons.add(result.stopping_reason)
    for reason in ("scoring_failed", "retrieval_unavailable", "no_evidence"):
        if reason in incomplete_reasons:
            return "incomplete", reason
    if results or admitted_document_count > 0:
        return "complete", "coverage_sufficient"
    return "incomplete", "no_evidence"


def generate_candidate_artifacts(
    paths: HandoffArtifacts,
    *,
    run_id: str,
    score_cache_root: Path,
    device: str,
    document_store_root: Path,
    scorer: object | None = None,
    facets: Sequence[FacetRecord] = (),
    passage_results: Sequence[PassageSearchResult] = (),
) -> CandidateArtifacts:
    """Generate and seal one topic records database from a typed handoff."""
    if not isinstance(paths, HandoffArtifacts):
        raise TypeError("paths must be HandoffArtifacts")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    input_path = Path(paths.requests_path)
    canonical_root = input_path.parent.parent
    topic_root = canonical_root.parent
    records_path = topic_root / "records.sqlite3"
    manifest_path = canonical_root / "records-manifest.json"
    handoff_manifest_path = Path(paths.manifest_path)
    resolved_paths = (
        input_path.resolve(), records_path.resolve(), manifest_path.resolve()
    )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("candidate input and output paths must name different files")
    handoff_manifest = _strict_candidate_json_object(
        handoff_manifest_path.read_bytes(), handoff_manifest_path, 1
    )
    topic_id = handoff_manifest.get("topic_id")
    if not isinstance(topic_id, str) or not _SAFE_TOPIC_ID.fullmatch(topic_id):
        raise ValueError("handoff manifest topic_id is invalid")
    request_schema_version = handoff_manifest.get("request_schema_version")
    if (
        not isinstance(request_schema_version, str)
        or request_schema_version not in _SUPPORTED_REQUEST_SCHEMA_VERSIONS
    ):
        raise ValueError("handoff manifest request schema is unsupported")
    if handoff_manifest.get("requests_file") != input_path.name:
        raise ValueError("handoff manifest requests_file does not match input basename")
    document_store_root = Path(document_store_root)
    document_store = DocumentStore(document_store_root)
    audit = _audit_candidate_requests(
        input_path,
        document_store=document_store,
        expected_schema=request_schema_version,
    )
    requests_sha256 = handoff_manifest.get("requests_sha256")
    if (
        not isinstance(requests_sha256, str)
        or not _SHA256.fullmatch(requests_sha256)
        or requests_sha256 != audit.source_sha256
    ):
        raise ValueError("handoff manifest request hash does not match audited requests")
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
        "request_schema_version": request_schema_version,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "source_file": input_path.name,
        "source_sha256": audit.source_sha256,
        "scorer": identity,
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": SCORING_NORMALIZATION_VERSION,
    }
    second_pass_sha256 = sha256()
    builder = TopicRecordsBuilder(
        topic_root,
        topic_id,
        document_store,
        run_id=run_id,
    )
    published: PublishedTopicRecords | None = None
    try:
        builder.add_facets(facets)
        for passage_result in passage_results:
            builder.add_passage_search(passage_result)
        for request in _iter_candidate_requests(
            input_path,
            document_store=document_store,
            expected_schema=request_schema_version,
            digest=second_pass_sha256,
        ):
            if scorer is None:
                raise ValueError(
                    "request JSONL changed while candidates were being generated"
                )
            builder.bind_document(
                request.document_id,
                request.source,
                expected_sha256=request.document_sha256,
            )
            for candidate in extract_document_candidates(request, scorer):
                builder.add_candidate(candidate)
        if second_pass_sha256.hexdigest() != audit.source_sha256:
            raise ValueError(
                "request JSONL changed while candidates were being generated"
            )
        completion = _fixed_retrieval_completion(
            passage_results,
            admitted_document_count=audit.document_count,
        )
        builder.set_completion(*completion)
        published = builder.publish(run_identity)
    finally:
        if published is None:
            builder._cleanup()
    return CandidateArtifacts(
        records_path,
        manifest_path,
        document_store_root,
        validation_session=published.validation_session,
    )


def _iter_candidate_requests(
    path: Path,
    *,
    document_store: DocumentStore | None = None,
    expected_schema: str | None = None,
    digest: Any | None = None,
) -> Iterator[ExtractiveCandidateRequest]:
    if (
        expected_schema is not None
        and expected_schema not in _SUPPORTED_REQUEST_SCHEMA_VERSIONS
    ):
        raise ValueError("expected request schema is unsupported")
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if digest is not None:
                digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL rows are not allowed")
            value = _strict_candidate_json_object(raw_line, path, line_number)
            schema_version = value.get("schema_version")
            if (
                not isinstance(schema_version, str)
                or schema_version not in _SUPPORTED_REQUEST_SCHEMA_VERSIONS
            ):
                raise ValueError(f"{path}:{line_number}: unsupported request schema version")
            if expected_schema is not None and schema_version != expected_schema:
                raise ValueError(
                    f"{path}:{line_number}: request schema differs from handoff manifest"
                )
            try:
                if schema_version == REQUEST_SCHEMA_VERSION:
                    request = _decode_legacy_candidate_request(
                        value,
                        path,
                        line_number,
                    )
                else:
                    if document_store is None:
                        raise ValueError(
                            "passage-first requests require a DocumentStore"
                        )
                    request = _decode_passage_candidate_request(
                        value,
                        path,
                        line_number,
                        document_store,
                    )
            except (DocumentStoreIntegrityError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid request: {exc}") from exc
            yield request


def _decode_legacy_candidate_request(
    value: Mapping[str, Any],
    path: Path,
    line_number: int,
) -> ExtractiveCandidateRequest:
    _require_candidate_fields(
        value,
        _CANDIDATE_REQUEST_FIELDS,
        f"{path}:{line_number}: request",
    )
    subnarratives = value["subnarratives"]
    passages = value["passages"]
    if not isinstance(subnarratives, list) or not isinstance(passages, list):
        raise ValueError("subnarratives and passages must be arrays")
    request = ExtractiveCandidateRequest(
        topic_id=value["topic_id"],
        document_id=value["document_id"],
        source=value["source"],
        document_sha256=value["document_sha256"],
        scoring_text_sha256=value["scoring_text_sha256"],
        subnarratives=tuple(
            _candidate_subnarrative(row, path, line_number)
            for row in subnarratives
        ),
        passages=tuple(
            _candidate_passage(row, path, line_number) for row in passages
        ),
    )
    _validate_candidate_source_hashes(request, path, line_number)
    validate_candidate_request(request)
    return request


def _decode_passage_candidate_request(
    value: Mapping[str, Any],
    path: Path,
    line_number: int,
    document_store: DocumentStore,
) -> ExtractiveCandidateRequest:
    _require_candidate_fields(
        value,
        _PASSAGE_REQUEST_FIELDS,
        f"{path}:{line_number}: passage-first request",
    )
    facet = _candidate_subnarrative(value["facet"], path, line_number)
    raw_passages = value["passages"]
    if not isinstance(raw_passages, list) or not raw_passages:
        raise ValueError("passage-first passages must be a non-empty array")
    content_sha256 = value["content_sha256"]
    source = document_store.read_text(content_sha256)
    if not source.strip() or _digest(source.encode("utf-8")) != content_sha256:
        raise ValueError("CAS source is empty or differs from content_sha256")

    source_spans: list[tuple[int, int]] = []
    for raw in raw_passages:
        if not isinstance(raw, dict):
            raise ValueError("passage-first passages must contain objects")
        _require_candidate_fields(
            raw,
            _PASSAGE_REQUEST_PASSAGE_FIELDS,
            f"{path}:{line_number}: passage-first passage",
        )
        start = raw["source_start_char"]
        end = raw["source_end_char"]
        start_byte = raw["source_start_byte"]
        end_byte = raw["source_end_byte"]
        if any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in (start, end, start_byte, end_byte)
        ):
            raise ValueError("source passage offsets must be integers")
        if start < 0 or end <= start or end > len(source):
            raise ValueError("source passage character offsets are out of range")
        if (
            start_byte != len(source[:start].encode("utf-8"))
            or end_byte != len(source[:end].encode("utf-8"))
            or end_byte <= start_byte
        ):
            raise ValueError("source passage byte offsets differ from CAS text")
        source_slice = source[start:end]
        if _digest(source_slice.encode("utf-8")) != raw["source_text_sha256"]:
            raise ValueError("source passage digest differs from CAS text")
        source_spans.append((start, end))

    scoring_text, projected = project_source_spans(source, tuple(source_spans))
    scoring_sha256 = _digest(scoring_text.encode("utf-8"))
    if value["scoring_text_sha256"] != scoring_sha256:
        raise ValueError("request scoring digest differs from CAS text")
    passages: list[ScoredPassage] = []
    passage_ids: set[str] = set()
    passage_ranks: set[int] = set()
    for raw, (scoring_start, scoring_end, chunk) in zip(
        raw_passages,
        projected,
        strict=True,
    ):
        if any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in (
                raw["scoring_start_char"],
                raw["scoring_end_char"],
            )
        ):
            raise ValueError("passage scoring offsets must be integers")
        if (
            raw["scoring_start_char"] != scoring_start
            or raw["scoring_end_char"] != scoring_end
            or raw["scoring_text_sha256"] != scoring_sha256
            or raw["chunk_text_sha256"] != _digest(chunk.encode("utf-8"))
        ):
            raise ValueError("passage scoring geometry differs from CAS projection")
        passage = ScoredPassage(
            passage_id=raw["passage_id"],
            lane_id=raw["lane_id"],
            query_id=raw["query_id"],
            scoring_start_char=scoring_start,
            scoring_end_char=scoring_end,
            scoring_text_sha256=scoring_sha256,
            chunk_text_sha256=raw["chunk_text_sha256"],
            cross_encoder_score=raw["cross_encoder_score"],
            cross_encoder_rank=raw["cross_encoder_rank"],
        )
        if passage.lane_id != f"subnarrative:{facet.subnarrative_id}":
            raise ValueError("passage lane differs from request facet")
        if passage.passage_id in passage_ids or passage.cross_encoder_rank in passage_ranks:
            raise ValueError("passage IDs and ranks must be unique within a request")
        passage_ids.add(passage.passage_id)
        passage_ranks.add(passage.cross_encoder_rank)
        passages.append(passage)

    request = ExtractiveCandidateRequest(
        topic_id=value["topic_id"],
        document_id=value["document_id"],
        source=source,
        document_sha256=content_sha256,
        scoring_text_sha256=scoring_sha256,
        subnarratives=(facet,),
        passages=tuple(passages),
    )
    _validate_candidate_source_hashes(request, path, line_number)
    validate_candidate_request(request)
    return request


def _audit_candidate_requests(
    path: Path,
    *,
    document_store: DocumentStore,
    expected_schema: str,
) -> _CandidateRequestAudit:
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
            audit.execute(
                "CREATE TABLE request_groups ("
                "document_id TEXT NOT NULL, subnarrative_id TEXT NOT NULL, "
                "PRIMARY KEY (document_id, subnarrative_id))"
            )
            audit.execute(
                "CREATE TABLE subnarratives ("
                "topic_id TEXT NOT NULL, subnarrative_id TEXT NOT NULL, "
                "text_sha256 TEXT NOT NULL, "
                "PRIMARY KEY (topic_id, subnarrative_id, text_sha256))"
            )
            for line_number, request in enumerate(
                _iter_candidate_requests(
                    path,
                    document_store=document_store,
                    expected_schema=expected_schema,
                    digest=digest,
                ),
                start=1,
            ):
                try:
                    audit.executemany(
                        "INSERT INTO request_groups VALUES (?, ?)",
                        (
                            (request.document_id, subnarrative.subnarrative_id)
                            for subnarrative in request.subnarratives
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate document/subnarrative request"
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
        "inference_dtype", "score_kind", "sentence_max_length", "input_policy",
    }
    if set(identity) != required:
        raise ValueError("scorer identity fields differ from the local sentence scorer contract")
    return dict(identity)


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
_CANDIDATE_STAGE_IDENTITY_FIELDS = frozenset({
    "request_schema_version", "candidate_schema_version", "source_file",
    "source_sha256", "scorer", "sentence_splitter_version",
    "scoring_normalization_version",
})
_SCORER_IDENTITY_FIELDS = frozenset({
    "model", "model_revision", "backend_version", "score_representation",
    "inference_dtype", "score_kind", "sentence_max_length", "input_policy",
})


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


def _canonical_candidate_artifact_paths(
    paths: CandidateArtifacts,
) -> tuple[Path, Path, Path]:
    records_path = Path(paths.records_path)
    manifest_path = Path(paths.manifest_path)
    document_store_root = Path(paths.document_store_root)
    if (
        records_path.name != "records.sqlite3"
        or manifest_path.name != "records-manifest.json"
        or manifest_path.parent.name != "canonical"
        or records_path.parent.resolve() != manifest_path.parent.parent.resolve()
    ):
        raise ValueError("canonical candidate artifact paths are invalid")
    return records_path, manifest_path, document_store_root


def _validate_candidate_stage_identity(
    records: TopicRecords,
) -> dict[str, object]:
    # TopicRecords.open validated this exact manifest snapshot against the DB seal.
    identity_json = records._manifest.get("identity_json")
    try:
        identity = json.loads(identity_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate stage identity JSON is invalid") from exc
    if not isinstance(identity, dict):
        raise ValueError("candidate stage identity must be an object")
    _require_selection_fields(
        identity,
        _CANDIDATE_STAGE_IDENTITY_FIELDS,
        "candidate stage identity",
    )
    expected = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": SCORING_NORMALIZATION_VERSION,
    }
    if identity["request_schema_version"] not in _SUPPORTED_REQUEST_SCHEMA_VERSIONS:
        raise ValueError("candidate stage identity request_schema_version mismatch")
    for key, expected_value in expected.items():
        if identity[key] != expected_value:
            raise ValueError(f"candidate stage identity {key} mismatch")
    source_file = identity["source_file"]
    if (
        not isinstance(source_file, str)
        or not source_file
        or source_file in {".", ".."}
        or "/" in source_file
        or "\\" in source_file
        or "\x00" in source_file
    ):
        raise ValueError("candidate stage identity source_file must be a basename")
    source_sha256 = identity["source_sha256"]
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError(
            "candidate stage identity source_sha256 must be a lowercase SHA-256 digest"
        )
    scorer = identity["scorer"]
    if not isinstance(scorer, dict):
        raise ValueError("candidate stage identity scorer must be an object")
    _require_selection_fields(
        scorer,
        _SCORER_IDENTITY_FIELDS,
        "candidate stage identity scorer",
    )
    for key in _SCORER_IDENTITY_FIELDS - {"sentence_max_length"}:
        if not isinstance(scorer[key], str) or not scorer[key]:
            raise ValueError(
                "candidate stage identity scorer values must be non-empty strings"
            )
    if (
        scorer["score_representation"] != "raw_logits"
        or scorer["score_kind"] != "extractive_sentence_v1"
        or scorer["input_policy"] != SCORING_NORMALIZATION_VERSION
    ):
        raise ValueError("candidate stage identity scorer contract mismatch")
    sentence_max_length = scorer["sentence_max_length"]
    if (
        isinstance(sentence_max_length, bool)
        or not isinstance(sentence_max_length, int)
        or sentence_max_length <= 0
    ):
        raise ValueError(
            "candidate stage identity scorer sentence_max_length must be positive"
        )
    return identity


def load_validated_candidate_artifacts(
    paths: CandidateArtifacts,
    *,
    expected_topic_id: str,
    required_candidate_keys: frozenset[tuple[str, str]] | None = None,
) -> Mapping[tuple[str, str], ExtractiveCandidate]:
    """Open sealed topic records and reconstruct the candidates a caller needs."""
    if not isinstance(paths, CandidateArtifacts):
        raise TypeError("paths must be CandidateArtifacts")
    records_path, manifest_path, document_store_root = (
        _canonical_candidate_artifact_paths(paths)
    )
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
    with TopicRecords.open(
        records_path,
        manifest_path,
        expected_topic_id,
        DocumentStore(document_store_root),
        validation_session=paths.validation_session,
    ) as records:
        _validate_candidate_stage_identity(records)
        return MappingProxyType(dict(records.load_candidates(required_candidate_keys)))


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
    records_path, records_manifest_path, document_store_root = (
        _canonical_candidate_artifact_paths(paths)
    )
    contexts_path = Path(contexts)
    canonical_root = records_manifest_path.parent
    output_path = canonical_root / "subnarrative-selections.jsonl"
    manifest_path = canonical_root / "selection-manifest.json"
    resolved_paths = (
        records_path.resolve(), records_manifest_path.resolve(),
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
    records_manifest_bytes = records_manifest_path.read_bytes()
    records_manifest = _strict_selection_json_object(
        records_manifest_bytes, records_manifest_path, 1
    )
    expected_topic_id = (
        loaded_contexts[0].topic_id
        if loaded_contexts
        else records_manifest.get("topic_id")
    )
    if not isinstance(expected_topic_id, str) or not expected_topic_id:
        raise ValueError("records manifest topic_id is invalid")
    loaded_candidate_projection_count = 0
    records_receipt = None
    with TopicRecords.open(
        records_path,
        records_manifest_path,
        expected_topic_id,
        DocumentStore(document_store_root),
        validation_session=paths.validation_session,
    ) as records:
        candidate_stage_identity = _validate_candidate_stage_identity(records)
        scorer_identity = candidate_stage_identity["scorer"]
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
        selections_list: list[SubnarrativeSelection] = []
        for context in loaded_contexts:
            if locked_similarity is None:
                raise RuntimeError("similarity provider was not initialized")
            pool = records.selection_pool(context, policy.precluster_limit)
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
        records_receipt = records.receipt
    if records_receipt is None:  # pragma: no cover - context manager invariant
        raise RuntimeError("topic records receipt was not resolved")
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
        "records_schema_version": TOPIC_RECORDS_SCHEMA_VERSION,
        "records_stage": CANDIDATE_STAGE,
        "records_file": "records.sqlite3",
        "records_manifest_file": "records-manifest.json",
        "contexts_file": contexts_path.name,
        "selection_file": output_path.name,
        "records_database_sha256": records_receipt.database_sha256,
        "candidate_semantic_sha256": records_receipt.semantic_sha256,
        "contexts_sha256": sha256(contexts_bytes).hexdigest(),
        "selections_sha256": sha256(output_bytes).hexdigest(),
        "output_sha256": sha256(output_bytes).hexdigest(),
        "candidate_rows_scanned": records_receipt.row_counts["candidate"],
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
        "candidate_scorer_identity": dict(scorer_identity),
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


class _LockedSimilarity:
    """Take one identity snapshot while delegating cosine work to the provider."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.identity = MappingProxyType(_provider_identity(delegate))

    def cosine_matrix(self, texts: Sequence[str]) -> Any:
        return self._delegate.cosine_matrix(texts)


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

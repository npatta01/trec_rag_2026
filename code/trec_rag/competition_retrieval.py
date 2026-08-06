"""The supported public interface for official facet retrieval exports."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass, replace
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any

import requests

from trec_rag.canonical_nuggets import PROMPT_VERSION, run_canonical_stage
from trec_rag.competition_cache_bundle import assert_no_incomplete_cache_bundle_merge
from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.document_store import DocumentStore
from trec_rag.evidence_store import (
    generate_candidate_artifacts,
    materialize_candidate_inputs,
    select_evidence_artifacts,
)
from trec_rag.evidence_local import LocalMiniLMSimilarity, MixedbreadSentencePairScorer
from trec_rag.evidence_bundle import BundleLane, EvidenceBundle
from trec_rag.facet_evidence import SelectionPolicy
from trec_rag.facet_extraction import (
    FacetPlanningResult,
    GeneratedQueryPlan,
    OPENROUTER_DEEPSEEK_MODEL,
    OpenRouterDeepSeekFacetBackend,
    PROMPT_VERSION as FACET_PROMPT_VERSION,
    SCHEMA_VERSION as FACET_SCHEMA_VERSION,
    extract_facets,
    plan_facet_queries,
)
from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.facet_retrieval import (
    LONG_DOCUMENT_WEIGHT,
    RELATIVE_SPAN_DELTA,
    SPAN_SUPPORT_CAP,
    SPAN_SUPPORT_WEIGHT,
    STRONGEST_PASSAGE_WEIGHT,
    TOP_WINDOW_WEIGHTS,
    FacetRetrievalResult,
    LaneRanking,
    LaneDocumentScore,
    PassageScore,
    RetrievalAuditCandidate,
    SELECTION_DEPTH,
    _union_pool,
    build_pyserini_retriever,
    build_retrieval_lanes,
    round_robin_select,
    run_facet_retrieval,
)
from trec_rag.mixedbread_passage_scorer import (
    BACKEND as PASSAGE_SCORER_BACKEND,
    BACKEND_VERSION as PASSAGE_SCORER_BACKEND_VERSION,
    INFERENCE_DTYPE as PASSAGE_SCORER_INFERENCE_DTYPE,
    INPUT_POLICY as PASSAGE_SCORER_INPUT_POLICY,
    MAX_LENGTH as PASSAGE_SCORER_MAX_LENGTH,
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
    SCORE_REPRESENTATION as PASSAGE_SCORER_SCORE_REPRESENTATION,
    MixedbreadPassageScorer,
)
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.planning_seed import (
    PLANNING_SEED_MANIFEST_FILENAME,
    PLANNING_SEED_MANIFEST_SCHEMA,
    PLANNING_SEED_MODE,
    PLANNING_SEED_RECEIPT_FILENAME,
    PLANNING_SEED_RECEIPT_SCHEMA,
)
from trec_rag.repo_env import load_repo_env, repo_cache_root
from trec_rag.remote_pyserini import RemotePyseriniThrottled
from trec_rag.rerank_score_cache import _choose_device
from trec_rag.retrieval_export import (
    RetrievalExportReceipt,
    TopicProjectionReceipt,
    build_topic_projection,
    export_retrieval_run,
    read_retrieval_export_receipt,
    read_topic_projection_receipt,
    validate_retrieval_topic_checkpoints,
)
from trec_rag.topics import Topic, load_narrative_topics
from trec_rag.topic_records import FacetRecord, TopicRecords
from trec_rag.topic_passage_search import (
    FocusedQuery,
    OrganizerRequestFailed,
    OrganizerRequestTerminal,
    PassageSearchPolicy,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
    TopicPassageSearch,
)
from trec_rag.topic_dispatch import (
    TopicJob,
    TopicJobReceipt,
    dispatch_topics,
    read_topic_receipt,
)


SCHEMA = "facet_pilot_v2"
SELECTION_SCHEMA = "facet_pilot_selection_v2"
# Compatibility-only fixed-path projection size; this is not organizer-final R.
INTERNAL_FIXED_SELECTION_K = SELECTION_DEPTH
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TOPIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DECOMPOSITION_MANIFEST_SCHEMA = "facet-decomposition-manifest-v1"
_DECOMPOSITION_PRODUCER_SCHEMA = "decomposition-producer-identity-v1"
CACHE_OPERATION_STAGE_NAMES = (
    "planning",
    "retrieval",
    "passage_scores",
    "sentence_scores",
    "similarity",
    "canonicalization",
)
_CACHE_OPERATION_COUNTER_NAMES = (
    "cache_hits",
    "cache_misses",
    "network_calls",
    "provider_calls",
    "model_batches",
)
_CACHE_OPERATION_PHASE_NAMES = ("planning", "retrieval", "scoring", "canonical")
_CACHE_OPERATION_RECEIPT_SCHEMA = "cache-operation-receipt-v1"
_CACHE_OPERATION_MANIFEST_SCHEMA = "cache-operation-manifest-v1"


@dataclass(frozen=True)
class ValidatedDecomposition:
    topic_id: str
    narrative_sha256: str
    source_sha256: str
    result: FacetPlanningResult


@dataclass(frozen=True)
class ValidatedRetrievalAuditLane:
    """One strictly decoded retrieval lane bound to its canonical plan query."""

    lane: Any
    requested_depth: int
    returned_count: int
    candidates: tuple[RetrievalAuditCandidate, ...]
    passage_result: PassageSearchResult | None = None


@dataclass(frozen=True)
class TopicPhaseOutcome:
    topic_id: str
    phase: str
    manifest_path: Path
    resumed: bool


@dataclass(frozen=True)
class _CanonicalPhaseResult:
    outcome: TopicPhaseOutcome
    provisional_manifest_bytes: bytes
    validation_session: Any | None = None


@dataclass(frozen=True)
class _TopicTaskOutcome:
    """The path-free result exchanged by one serial topic task."""

    topic_id: str
    resumed: bool
    projection_receipt: TopicProjectionReceipt


@dataclass(frozen=True)
class ExternalAdapters:
    """Optional seams for the three hosted or remote official-run boundaries."""

    planning_backend: Any | None = None
    retriever: Any | None = None
    canonical_backend_factory: Callable[[], Any] | None = None


@dataclass(frozen=True)
class RunReceipt:
    """The stable, public result of an official run."""

    experiment_id: str
    selected_topic_ids: tuple[str, ...]
    resumed_topic_ids: tuple[str, ...]
    retrieval_export: RetrievalExportReceipt


@dataclass(frozen=True)
class _CacheOperationReceipt:
    topic_id: str
    path: Path
    sha256: str
    projection_manifest_sha256: str
    stages: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class _RuntimeDependencies:
    """Private local seams retained for production construction and tests."""

    code_commit: str
    document_scorer: Any | None
    candidate_scorer: Any | None
    similarity: Any | None
    cache_ignore_checker: Callable[[Path], bool] | None
    planning_backend: Any | None = None
    retriever: Any | None = None
    canonical_backend_factory: Callable[[], Any] | None = None


class _TopicPassageSearchAdapter:
    """Public fixed-path adapter over the shared search and topic CAS."""

    def __init__(
        self,
        search: TopicPassageSearch,
        document_store: DocumentStore,
        passage_identity: Mapping[str, object],
    ) -> None:
        self._search = search
        self._document_store = document_store
        self.identity = dict(passage_identity)

    @property
    def topic_id(self) -> str:
        return self._search.topic_id

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        return self._search.search(query)

    def read_text(self, content_sha256: str) -> str:
        return self._document_store.read_text(content_sha256)


class _OfflineStagingDocumentStore:
    """Validate source CAS objects and copy them only into ephemeral output state."""

    def __init__(self, *, source_root: Path, stage_root: Path) -> None:
        self._source = DocumentStore(Path(source_root))
        self._stage = DocumentStore(Path(stage_root))

    def admit_text(self, text: str, *, expected_sha256: str | None = None) -> Any:
        if expected_sha256 is None:
            expected_sha256 = _hash(text.encode("utf-8"))
        source_receipt = self._source.verify(expected_sha256)
        if self._source.read_text(expected_sha256) != text:
            raise ValueError("shared document cache text differs from retrieval result")
        staged_receipt = self._stage.admit_text(
            text,
            expected_sha256=expected_sha256,
        )
        if staged_receipt != source_receipt:
            raise ValueError("staged document receipt differs from shared cache")
        return staged_receipt

    def read_text(self, digest: str) -> str:
        return self._source.read_text(digest)


class _DepthBoundExternalRetriever:
    """Adapt the public one-argument retriever seam to shared passage search."""

    def __init__(self, retriever: Any, depth: int) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("retrieval depth must be a positive integer")
        retrieve = getattr(retriever, "retrieve", None)
        if not callable(retrieve):
            raise TypeError("external retriever must expose callable retrieve(query)")
        identity = _retriever_identity(retriever, retrieval_depth=depth)
        try:
            canonical_identity = json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("external retriever identity must be canonical JSON") from exc
        if json.loads(canonical_identity) != identity:
            raise ValueError("external retriever identity must be immutable JSON data")
        self._retriever = retriever
        self._depth = depth
        self._canonical_identity = canonical_identity
        self.identity = MappingProxyType(dict(identity))
        self._continuation_required = False

    def _validate_current_identity(self) -> None:
        current = _retriever_identity(
            self._retriever,
            retrieval_depth=self._depth,
        )
        try:
            canonical = json.dumps(
                current,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("external retriever identity must be canonical JSON") from exc
        if canonical != self._canonical_identity:
            raise ValueError("external retriever identity changed after construction")

    def retrieve(
        self, query: QueryVariant, *, depth: int
    ) -> Sequence[RetrievedCandidate]:
        if depth != self._depth:
            raise ValueError("shared passage search requested an unbound retrieval depth")
        self._validate_current_identity()
        if self._continuation_required:
            raise OrganizerRequestTerminal(
                "Pyserini continuation is required before another transport attempt"
            )
        try:
            return tuple(self._retriever.retrieve(query))
        except OrganizerRequestFailed:
            raise
        except RemotePyseriniThrottled as exc:
            self._continuation_required = True
            raise OrganizerRequestTerminal(
                "Pyserini throttle is terminal until an explicit continuation is supplied"
            ) from exc
        except requests.RequestException as exc:
            raise OrganizerRequestFailed(
                "Pyserini transport failure is retryable by shared passage search"
            ) from exc


class _SealedPassageSearchAdapter:
    """Pure score-phase adapter that only returns decoded checkpoint rows."""

    def __init__(self, results: Sequence[PassageSearchResult], document_store: DocumentStore):
        self._results = {result.query.query_id: result for result in results}
        self._document_store = document_store

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        result = self._results.get(query.query_id)
        if result is None:
            raise ValueError("sealed passage checkpoint is missing a fixed lane")
        if result.query != query:
            raise ValueError("sealed passage checkpoint query identity changed")
        return result

    def read_text(self, content_sha256: str) -> str:
        return self._document_store.read_text(content_sha256)


def load_validated_decomposition(
    topic: Topic,
    path: Path,
) -> ValidatedDecomposition:
    """Revalidate a saved live plan against the exact official narrative."""
    source = Path(path).read_bytes()
    if not source or len(source) > 2 * 1024 * 1024:
        raise ValueError("saved decomposition size is invalid")
    root = _loads(source, "saved decomposition")
    required = {
        "schema_version",
        "topic",
        "narrative_sha256",
        "used_fallback",
        "error",
        "queries",
        "plan",
        "subnarratives",
    }
    if not isinstance(root, dict) or set(root) != required:
        raise ValueError("saved decomposition fields are invalid")
    if root["schema_version"] != SCHEMA:
        raise ValueError("saved decomposition schema version is invalid")
    digest = _hash(topic.narrative.encode())
    saved_topic = root["topic"]
    if (
        not isinstance(saved_topic, dict)
        or set(saved_topic) != {"id", "narrative"}
        or saved_topic.get("id") != topic.id
        or saved_topic.get("narrative") != topic.narrative
        or root["narrative_sha256"] != digest
    ):
        raise ValueError("saved decomposition differs from the exact official narrative")
    original = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    if root["used_fallback"] is True:
        error = root["error"]
        if (
            not isinstance(error, str)
            or not error.strip()
            or root["plan"] is not None
            or root["subnarratives"] != []
            or root["queries"] != [jsonable(original)]
        ):
            raise ValueError(
                "saved fallback must be the exact original-narrative fallback"
            )
        fallback = FacetPlanningResult((original,), True, error, None, ())
        return ValidatedDecomposition(topic.id, digest, _hash(source), fallback)
    if (
        root["used_fallback"] is not False
        or root["error"] is not None
        or root["plan"] is None
    ):
        raise ValueError("retrieval requires a valid non-fallback decomposition")
    try:
        rendered = plan_facet_queries(topic, root["plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("saved decomposition failed typed validation") from exc
    if rendered.used_fallback or rendered.plan is None:
        raise ValueError("saved decomposition failed deterministic plan validation")
    if root["queries"] != jsonable(rendered.queries):
        raise ValueError("saved rendered queries differ from deterministic rendering")
    if (
        root["subnarratives"] != jsonable(rendered.subnarratives)
        or root["plan"] != _plan_payload(rendered.plan)
    ):
        raise ValueError("saved plan records differ from deterministic rendering")
    return ValidatedDecomposition(topic.id, digest, _hash(source), rendered)


def decode_retrieval_decomposition(
    topic: Topic,
    source: bytes,
    *,
    expected_source_sha256: str,
) -> ValidatedDecomposition:
    """Strictly decode the receipted retrieval projection of a saved plan."""
    if not isinstance(source, bytes) or not source or len(source) > 2 * 1024 * 1024:
        raise ValueError("receipted decomposition size is invalid")
    root = _loads(source, "receipted decomposition")
    fields = {
        "schema_version", "topic_id", "narrative", "narrative_sha256",
        "source_sha256", "queries", "plan", "subnarratives",
    }
    narrative_sha256 = _hash(topic.narrative.encode("utf-8"))
    if (
        not isinstance(root, dict)
        or set(root) != fields
        or root.get("schema_version") != SCHEMA
        or root.get("topic_id") != topic.id
        or root.get("narrative") != topic.narrative
        or root.get("narrative_sha256") != narrative_sha256
        or root.get("source_sha256") != expected_source_sha256
        or not isinstance(expected_source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256)
    ):
        raise ValueError("receipted decomposition identity changed")
    original = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    if root["plan"] is None:
        if root["queries"] != [jsonable(original)] or root["subnarratives"] != []:
            raise ValueError("receipted original-only decomposition changed")
        result = FacetPlanningResult(
            (original,), True, "sealed original-only fallback", None, ()
        )
    else:
        result = plan_facet_queries(topic, root["plan"])
        if result.used_fallback or result.plan is None:
            raise ValueError("receipted decomposition plan is invalid")
        if (
            root["queries"] != jsonable(result.queries)
            or root["subnarratives"] != jsonable(result.subnarratives)
            or root["plan"] != _plan_payload(result.plan)
        ):
            raise ValueError("receipted decomposition differs from canonical rendering")
    return ValidatedDecomposition(
        topic.id, narrative_sha256, expected_source_sha256, result
    )


def decode_retrieval_audit(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    source: bytes,
    *,
    requested_depth: int,
) -> tuple[ValidatedRetrievalAuditLane, ...]:
    """Strictly decode retrieval audit rows against canonical plan lanes."""
    if isinstance(requested_depth, bool) or not isinstance(requested_depth, int) or requested_depth <= 0:
        raise ValueError("retrieval audit requested depth is invalid")
    root = _loads(source, "retrieval audit")
    expected_fields = {
        "schema_version", "topic_id", "narrative_sha256",
        "decomposition_source_sha256", "requested_depth", "lanes",
        "passage_search_results",
    }
    expected_lanes = build_retrieval_lanes(
        topic, decomposition.result.queries, decomposition.result.subnarratives
    )
    if (
        not isinstance(root, dict)
        or set(root) != expected_fields
        or root.get("schema_version") != SCHEMA
        or root.get("topic_id") != topic.id
        or root.get("narrative_sha256") != decomposition.narrative_sha256
        or root.get("decomposition_source_sha256") != decomposition.source_sha256
        or type(root.get("requested_depth")) is not int
        or root["requested_depth"] != requested_depth
        or not isinstance(root.get("lanes"), list)
        or len(root["lanes"]) != len(expected_lanes)
        or not isinstance(root.get("passage_search_results"), list)
        or len(root["passage_search_results"]) != len(expected_lanes)
    ):
        raise ValueError("retrieval audit identity or lane set changed")
    decoded: list[ValidatedRetrievalAuditLane] = []
    lane_fields = {
        "lane_name", "subnarrative_id", "bm25_query_sha256",
        "semantic_query_sha256", "returned_count", "retained_count", "candidates",
    }
    candidate_fields = {"docid", "bm25_rank", "bm25_score", "text_sha256"}
    for raw, lane, passage_raw in zip(
        root["lanes"], expected_lanes, root["passage_search_results"], strict=True
    ):
        if (
            not isinstance(raw, dict)
            or set(raw) != lane_fields
            or raw.get("lane_name") != lane.retrieval_query.variant_name
            or raw.get("subnarrative_id") != lane.subnarrative_id
            or raw.get("bm25_query_sha256") != lane.bm25_query_sha256
            or raw.get("semantic_query_sha256") != lane.semantic_query_sha256
            or type(raw.get("returned_count")) is not int
            or raw["returned_count"] < 0
            or type(raw.get("retained_count")) is not int
            or raw["retained_count"] < 0
            or raw["retained_count"] != min(raw["returned_count"], requested_depth)
            or not isinstance(raw.get("candidates"), list)
            or len(raw["candidates"]) != raw["retained_count"]
        ):
            raise ValueError("retrieval audit lane identity or counts changed")
        candidates: list[RetrievalAuditCandidate] = []
        seen: set[str] = set()
        for rank, value in enumerate(raw["candidates"], start=1):
            if not isinstance(value, dict) or set(value) != candidate_fields:
                raise ValueError("retrieval audit candidate schema changed")
            docid = value.get("docid")
            score = value.get("bm25_score")
            text_sha256 = value.get("text_sha256")
            if (
                not isinstance(docid, str) or not docid or docid in seen
                or type(value.get("bm25_rank")) is not int
                or value["bm25_rank"] != rank
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not isinstance(text_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", text_sha256)
            ):
                raise ValueError(
                    "duplicate or invalid retrieval audit topic-document candidate"
                )
            seen.add(docid)
            candidates.append(
                RetrievalAuditCandidate(docid, rank, score, text_sha256)
            )
        passage_result = _decode_passage_result(passage_raw)
        if (
            passage_result.query.query_id != lane.retrieval_query.variant_name
            or passage_result.query.text != lane.scoring_query.query_text
            or passage_result.query.primary_subnarrative_id
            != (lane.subnarrative_id or "original")
            or passage_result.requested_documents != requested_depth
            or passage_result.returned_documents != raw["returned_count"]
            or len(passage_result.documents) != len(candidates)
            or any(
                (
                    document.docid,
                    document.source_rank,
                    document.source_score,
                    document.content_sha256,
                )
                != (
                    candidate.docid,
                    candidate.bm25_rank,
                    candidate.bm25_score,
                    candidate.text_sha256,
                )
                for document, candidate in zip(
                    passage_result.documents, candidates, strict=True
                )
            )
        ):
            raise ValueError("passage search documents differ from retrieval audit")
        decoded.append(
            ValidatedRetrievalAuditLane(
                lane,
                requested_depth,
                raw["returned_count"],
                tuple(candidates),
                passage_result,
            )
        )
    return tuple(decoded)


def _passage_result_json(result: PassageSearchResult) -> dict[str, object]:
    """Serialize shared passage rows without changing their source identities."""
    return {
        "query": jsonable(result.query),
        "status": result.status,
        "stopping_reason": result.stopping_reason,
        "requested_documents": result.requested_documents,
        "returned_documents": result.returned_documents,
        "scored_documents": result.scored_documents,
        "scored_passages": result.scored_passages,
        "documents": [jsonable(document) for document in result.documents],
        "passages": [
            {
                "passage_id": passage.passage_id,
                "docid": passage.docid,
                "content_sha256": passage.content_sha256,
                "source_rank": passage.source_rank,
                "source_score": passage.source_score,
                "start_char": passage.start_char,
                "end_char": passage.end_char,
                "start_byte": passage.start_byte,
                "end_byte": passage.end_byte,
                "text_sha256": passage.text_sha256,
                "text": passage.text,
                "raw_logit": passage.raw_logit,
                "rank": passage.rank,
                "score_cache_key": passage.score_cache_key,
                "scoring_text_sha256": passage.scoring_text_sha256,
                "chunker_identity": dict(passage.chunker_identity),
            }
            for passage in result.passages
        ],
        "attempt_count": result.attempt_count,
        "source_exhausted": result.source_exhausted,
    }


def _decode_passage_result(value: object) -> PassageSearchResult:
    if not isinstance(value, dict):
        raise ValueError("passage search result must be an object")
    expected = {
        "query", "status", "stopping_reason", "requested_documents",
        "returned_documents", "scored_documents", "scored_passages",
        "documents", "passages", "attempt_count", "source_exhausted",
    }
    if set(value) != expected or not isinstance(value.get("query"), dict):
        raise ValueError("passage search result fields changed")
    query = value["query"]
    if set(query) != {
        "query_id", "text", "primary_subnarrative_id", "supporting_subnarrative_ids"
    }:
        raise ValueError("passage search query fields changed")
    documents_raw = value["documents"]
    passages_raw = value["passages"]
    if not isinstance(documents_raw, list) or not isinstance(passages_raw, list):
        raise ValueError("passage search rows must be lists")
    documents = tuple(
        SourceDocument(
            row["docid"], row["content_sha256"], row["source_rank"],
            row["source_score"], row["best_passage_id"],
            row["best_passage_raw_logit"],
        )
        for row in documents_raw
        if isinstance(row, dict)
        and set(row) == {
            "docid", "content_sha256", "source_rank", "source_score",
            "best_passage_id", "best_passage_raw_logit",
        }
    )
    if len(documents) != len(documents_raw):
        raise ValueError("passage search document fields changed")
    passages: list[SourcePassage] = []
    passage_fields = {
        "passage_id", "docid", "content_sha256", "source_rank", "source_score",
        "start_char", "end_char", "start_byte", "end_byte", "text_sha256", "text",
        "raw_logit", "rank", "score_cache_key", "scoring_text_sha256", "chunker_identity",
    }
    for row in passages_raw:
        if not isinstance(row, dict) or set(row) != passage_fields:
            raise ValueError("passage search passage fields changed")
        passages.append(
            SourcePassage(
                row["passage_id"], row["docid"], row["content_sha256"],
                row["source_rank"], row["source_score"], row["start_char"],
                row["end_char"], row["start_byte"], row["end_byte"],
                row["text_sha256"], row["text"], row["raw_logit"], row["rank"],
                row["score_cache_key"], row["scoring_text_sha256"],
                row["chunker_identity"],
            )
        )
    return PassageSearchResult(
        FocusedQuery(
            query["query_id"], query["text"], query["primary_subnarrative_id"],
            tuple(query["supporting_subnarrative_ids"]),
        ),
        value["status"], value["stopping_reason"], value["requested_documents"],
        value["returned_documents"], value["scored_documents"], value["scored_passages"],
        documents, tuple(passages), value["attempt_count"], value["source_exhausted"],
    )


def validate_scoring_selection(
    source: bytes,
    *,
    topic_id: str,
    selected_documents: Sequence[Mapping[str, object]],
    lane_score_rows: Sequence[Mapping[str, object]],
    audit_lanes: Sequence[ValidatedRetrievalAuditLane],
    rerank_depth: int,
    selected_set_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Replay producer selection and require its exact sealed serialization."""
    value = _loads(source, "selection checkpoint")
    if not isinstance(value, dict):
        raise ValueError("selection checkpoint must be an object")
    requested_count = value.get("requested_count")
    if isinstance(requested_count, bool) or not isinstance(requested_count, int) or requested_count <= 0:
        raise ValueError("selection requested count must be a positive integer")
    if isinstance(rerank_depth, bool) or not isinstance(rerank_depth, int) or rerank_depth <= 0:
        raise ValueError("selection rerank depth is invalid")
    scores_by_lane: dict[str, list[Mapping[str, object]]] = {}
    for row in lane_score_rows:
        lane_name = row.get("lane_name")
        if not isinstance(lane_name, str):
            raise ValueError("selection lane score identity is invalid")
        scores_by_lane.setdefault(lane_name, []).append(row)
    rankings: list[LaneRanking] = []
    for audited in audit_lanes:
        lane_name = audited.lane.retrieval_query.variant_name
        score_rows = scores_by_lane.pop(lane_name, [])
        passage_result = audited.passage_result
        if passage_result is None or len(passage_result.documents) != len(audited.candidates):
            raise ValueError("selection passage binding differs from retrieval audit")
        eligible = tuple(
            candidate
            for candidate, document in zip(
                audited.candidates, passage_result.documents, strict=True
            )
            if document.best_passage_id is not None
        )[:rerank_depth]
        if len(score_rows) != len(eligible):
            raise ValueError("selection lane scores differ from retrieval audit")
        audit_by_doc = {row.docid: row for row in eligible}
        source_by_doc = {
            document.docid: document for document in passage_result.documents
        }
        typed_scores: list[LaneDocumentScore] = []
        for expected_rank, row in enumerate(score_rows, start=1):
            audit = audit_by_doc.get(row.get("docid"))
            source_document = source_by_doc.get(row.get("docid"))
            expected_passages = [
                {
                    "chunk_index": index,
                    "start_char": passage.start_char,
                    "end_char": passage.end_char,
                    "raw_logit": passage.raw_logit,
                    "weighted_rank": index,
                }
                for index, passage in enumerate(
                    (
                        passage
                        for passage in passage_result.passages
                        if passage.docid == row.get("docid")
                    ),
                    start=1,
                )
            ]
            if (
                type(row.get("aggregate_rank")) is not int
                or row["aggregate_rank"] != expected_rank
                or audit is None
                or source_document is None
                or row.get("bm25_rank") != audit.bm25_rank
                or row.get("bm25_score") != audit.bm25_score
                or row.get("text_sha256") != audit.text_sha256
                or row.get("bm25_query_sha256") != audited.lane.bm25_query_sha256
                or row.get("semantic_query_sha256") != audited.lane.semantic_query_sha256
                or row.get("aggregate_score")
                != source_document.best_passage_raw_logit
                or row.get("long_document_raw_logit")
                != source_document.best_passage_raw_logit
                or row.get("weighted_passage_raw_logit")
                != source_document.best_passage_raw_logit
                or row.get("within_document_span_support") != 0
                or row.get("winning_passages") != expected_passages
            ):
                raise ValueError(
                    "selection lane score differs from retrieval audit or original narrative query"
                )
            passages = tuple(
                PassageScore(
                    passage["chunk_index"], passage["start_char"],
                    passage["end_char"], passage["raw_logit"],
                    passage["weighted_rank"],
                )
                for passage in row["winning_passages"]
            )
            typed_scores.append(
                LaneDocumentScore(
                    topic_id=topic_id,
                    lane_name=lane_name,
                    bm25_query=audited.lane.retrieval_query.query_text,
                    bm25_query_sha256=audited.lane.bm25_query_sha256,
                    semantic_query=audited.lane.scoring_query.query_text,
                    semantic_query_sha256=audited.lane.semantic_query_sha256,
                    docid=audit.docid,
                    text=audit.text_sha256,
                    bm25_rank=audit.bm25_rank,
                    bm25_score=audit.bm25_score,
                    aggregate_rank=row["aggregate_rank"],
                    aggregate_score=row["aggregate_score"],
                    long_document_raw_logit=row["long_document_raw_logit"],
                    weighted_passage_raw_logit=row["weighted_passage_raw_logit"],
                    within_document_span_support=row["within_document_span_support"],
                    winning_passages=passages,
                    score_representation=row["score_representation"],
                )
            )
        rankings.append(
            LaneRanking(
                lane=audited.lane,
                retrieval_returned_count=audited.returned_count,
                retrieval_retained_count=len(audited.candidates),
                retrieval_audit_candidates=audited.candidates,
                documents=tuple(typed_scores),
                retrieval_requested_depth=audited.requested_depth,
                rerank_depth=passage_result.scored_documents,
            )
        )
    if scores_by_lane:
        raise ValueError("selection lane scores include an unknown lane")
    replayed = round_robin_select(rankings, limit=requested_count)
    if len(replayed.documents) != len(selected_documents):
        raise ValueError("selection replay differs from selected documents")
    for replayed_row, selected in zip(
        replayed.documents, selected_documents, strict=True
    ):
        if (
            replayed_row.docid != selected.get("docid")
            or replayed_row.selection_rank
            != selected.get("selection_rank", selected.get("rank"))
            or replayed_row.selected_from_lane != selected.get("selected_from_lane")
            or replayed_row.selected_from_lane_rank
            != selected.get("selected_from_lane_rank")
            or replayed_row.text != selected.get("text_sha256")
        ):
            raise ValueError("selection replay differs from selected document provenance")
    result = FacetRetrievalResult(
        topic_id=topic_id,
        lanes=tuple(rankings),
        selection=replayed,
        original_only_control=rankings[0].documents,
        union_pool=_union_pool(rankings),
        rerank_depth=rerank_depth,
        selection_k=requested_count,
    )
    expected = _selection(result, selected_set_sha256)
    if json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) != json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise ValueError(
            "selection checkpoint lane score or trace differs from deterministic replay"
        )
    return tuple(expected["memberships"])


def _retrieve_topic(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    output_dir: Path,
    cache_dir: Path,
    code_commit: str,
    corpus_epoch: str | None,
    retriever: Any | None = None,
    retrieval_depth: int,
    passage_search: Any | None = None,
    passage_identity: Mapping[str, object],
    cache_only: bool = False,
) -> TopicPhaseOutcome:
    """Run and seal one shared passage search per fixed lane."""
    if (
        isinstance(retrieval_depth, bool)
        or not isinstance(retrieval_depth, int)
        or retrieval_depth <= 0
    ):
        raise ValueError("retrieval_depth must be a positive integer")
    _inputs(topic, decomposition, code_commit)
    retriever = retriever or build_pyserini_retriever(
        Path(cache_dir),
        hits=retrieval_depth,
        corpus_epoch=corpus_epoch,
        cache_only=cache_only,
    )
    identity = _retriever_identity(retriever, retrieval_depth=retrieval_depth)
    root = Path(output_dir) / topic.id
    manifest = root / "retrieval" / "complete.json"
    expected = _expected(
        "retrieve",
        topic,
        decomposition,
        code_commit,
        identity,
        passage_identity,
    )
    artifacts = (
        "decomposition.json",
        "retrieval/audit.json",
        "retrieval/evidence-bundle.json",
    )
    if _resume(manifest, root, expected, artifacts):
        return TopicPhaseOutcome(topic.id, "retrieve", manifest, True)

    if passage_search is None:
        raise TypeError("v2 retrieval requires a shared passage_search adapter")
    result = run_facet_retrieval(
        topic,
        decomposition.result.queries,
        subnarratives=decomposition.result.subnarratives,
        passage_search=passage_search,
        retrieval_depth=retrieval_depth,
    )
    lanes = tuple(lane.lane for lane in result.lanes)
    audit_lanes: list[dict[str, object]] = []
    passage_results: list[PassageSearchResult] = []
    for lane_result in result.lanes:
        lane = lane_result.lane
        passage_result = lane_result.passage_result
        if passage_result is None:
            raise ValueError("fixed retrieval lane is missing its passage result")
        _validate_passage_result_identity(passage_result, passage_identity)
        passage_results.append(passage_result)
        candidates = [
            {
                "docid": document.docid,
                "bm25_rank": document.source_rank,
                "bm25_score": document.source_score,
                "text_sha256": document.content_sha256,
            }
            for document in passage_result.documents
        ]
        audit_lanes.append(
            _audit_lane(lane, passage_result.returned_documents, candidates)
        )
    decomposition_record = {
        "schema_version": SCHEMA,
        "topic_id": topic.id,
        "narrative": topic.narrative,
        "narrative_sha256": decomposition.narrative_sha256,
        "source_sha256": decomposition.source_sha256,
        "queries": jsonable(decomposition.result.queries),
        "plan": _plan_payload(decomposition.result.plan),
        "subnarratives": jsonable(decomposition.result.subnarratives),
    }
    audit = _audit(
        topic,
        decomposition,
        audit_lanes,
        retrieval_depth=retrieval_depth,
        passage_search_results=passage_results,
    )
    evidence_bundle = {
        "schema_version": "facet_passage_evidence_v2",
        "topic_id": topic.id,
        "lanes": [
            {
                "lane_id": lane_result.lane.retrieval_query.variant_name,
                "lane_kind": (
                    "narrative"
                    if lane_result.lane.subnarrative_id is None
                    else "subnarrative"
                ),
                "query_text": lane_result.lane.scoring_query.query_text,
                "query_text_sha256": lane_result.lane.semantic_query_sha256,
                "documents": [
                    jsonable(document)
                    for document in lane_result.passage_result.documents
                ] if lane_result.passage_result is not None else [],
                "passages": (
                    _passage_result_json(lane_result.passage_result)["passages"]
                    if lane_result.passage_result is not None
                    else []
                ),
            }
            for lane_result in result.lanes
        ],
    }
    _write_json(root / artifacts[0], decomposition_record)
    _write_json(root / artifacts[1], audit)
    _write_json(root / artifacts[2], evidence_bundle)
    _complete(manifest, root, expected, artifacts)
    return TopicPhaseOutcome(topic.id, "retrieve", manifest, False)


def _score_topic(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    output_dir: Path,
    cache_dir: Path,
    score_cache_root: Path,
    code_commit: str,
    corpus_epoch: str | None,
    retriever: Any | None,
    scorer: Any | None,
    device: str,
    retrieval_depth: int,
    rerank_depth: int,
    selection_k: int,
    document_store_root: Path | None = None,
    expected_retriever_identity: Mapping[str, object],
    expected_passage_identity: Mapping[str, object],
) -> TopicPhaseOutcome:
    """Validate and project the sealed passage-first retrieval checkpoint."""
    _inputs(topic, decomposition, code_commit)
    root = Path(output_dir) / topic.id
    retrieve_manifest = root / "retrieval" / "complete.json"
    _resume(
        retrieve_manifest,
        root,
        _expected(
            "retrieve",
            topic,
            decomposition,
            code_commit,
            dict(expected_retriever_identity),
            expected_passage_identity,
        ),
        (
            "decomposition.json",
            "retrieval/audit.json",
            "retrieval/evidence-bundle.json",
        ),
        required=True,
    )
    expected = _expected(
        "score",
        topic,
        decomposition,
        code_commit,
        dict(expected_retriever_identity),
        expected_passage_identity,
    ) | {
        "selection_schema_version": SELECTION_SCHEMA,
        "retrieval_manifest_sha256": _hash(retrieve_manifest.read_bytes()),
        "scorer": {"source": "sealed_topic_passage_search"},
        "rerank_depth": rerank_depth,
        "selection_k": selection_k,
        "selection_scope": "internal_fixed_path_projection_not_final_submission",
        "selection_policy": "round_robin_lane_order_no_fusion",
        "score_policy": {
            "document_order": "best_passage_raw_logit_source_rank_docid",
            "passage_scores": "sealed_topic_passage_search",
        },
    }
    artifacts = (
        "scoring/lane_scores.jsonl",
        "scoring/selected_documents.jsonl",
        "scoring/selection.json",
        "scoring/selected_subnarrative_scores.jsonl",
    )
    manifest = root / "scoring" / "complete.json"
    if _resume(manifest, root, expected, artifacts):
        return TopicPhaseOutcome(topic.id, "score", manifest, True)

    audit_bytes = (root / "retrieval/audit.json").read_bytes()
    audited_lanes = decode_retrieval_audit(
        topic,
        decomposition,
        audit_bytes,
        requested_depth=retrieval_depth,
    )
    passage_results = tuple(
        audited.passage_result
        for audited in audited_lanes
        if audited.passage_result is not None
    )
    if len(passage_results) != len(audited_lanes):
        raise ValueError("retrieval audit is missing a passage result")
    for passage_result in passage_results:
        _validate_passage_result_identity(passage_result, expected_passage_identity)
    store = DocumentStore(
        Path(document_store_root)
        if document_store_root is not None
        else document_store_dir(Path(output_dir).parent.parent)
    )
    result = run_facet_retrieval(
        topic,
        decomposition.result.queries,
        subnarratives=decomposition.result.subnarratives,
        passage_search=_SealedPassageSearchAdapter(passage_results, store),
        retrieval_depth=retrieval_depth,
        selection_k=selection_k,
    )
    if _loads(audit_bytes, "retrieval audit") != _result_audit(topic, decomposition, result):
        raise ValueError("scoring retrieval differs from the completed retrieval audit")
    lane_rows = [_score_row(row) for lane in result.lanes for row in lane.documents]
    selected = [
        {
            "topic_id": row.topic_id,
            "docid": row.docid,
            "selection_rank": row.selection_rank,
            "selected_from_lane": row.selected_from_lane,
            "selected_from_lane_rank": row.selected_from_lane_rank,
            "text_sha256": _hash(row.text.encode()),
            "text": row.text,
        }
        for row in result.selection.documents
    ]
    selected_hash = _hash(
        json.dumps(
            [row["docid"] for row in selected],
            separators=(",", ":"),
        ).encode()
    )
    _write_jsonl(root / artifacts[0], lane_rows)
    _write_jsonl(root / artifacts[1], selected)
    _write_json(root / artifacts[2], _selection(result, selected_hash))
    _write_jsonl(
        root / artifacts[3],
        _project_subnarrative_scores(result, decomposition.result.subnarratives),
    )
    _complete(
        manifest,
        root,
        expected | {"selected_set_sha256": selected_hash},
        artifacts,
    )
    return TopicPhaseOutcome(topic.id, "score", manifest, False)


def _run_topic(
    topic: Topic,
    config: FacetPilotConfig,
    identity: str,
    dependencies: _RuntimeDependencies,
    *,
    config_sha256: str,
    expected_retriever_identity: Mapping[str, object],
    offline_cache_only: bool = False,
    offline_source_document_store_root: Path | None = None,
    offline_stage_document_store_root: Path | None = None,
) -> _TopicTaskOutcome:
    """Run one configured topic through the private checkpointed workflow."""
    if not isinstance(config, FacetPilotConfig):
        raise TypeError("config must be a FacetPilotConfig")
    if not isinstance(dependencies, _RuntimeDependencies):
        raise TypeError("dependencies must be _RuntimeDependencies")
    if not isinstance(offline_cache_only, bool):
        raise TypeError("offline_cache_only must be Boolean")
    if offline_cache_only and (
        offline_source_document_store_root is None
        or offline_stage_document_store_root is None
    ):
        raise ValueError("offline cache replay requires source and stage document roots")
    active_document_store_root = (
        Path(offline_stage_document_store_root)
        if offline_cache_only
        else document_store_dir(config.root_dir)
    )
    runtime_topic = _runtime_topic(topic)
    before_accounting = _runtime_cache_accounting(dependencies)
    planning_cache_stats: dict[str, int] = {}
    canonical_cache_stats: dict[str, int] = {}
    decomposition, decomposition_resumed = _decompose_topic(
        runtime_topic,
        config.output_dir,
        dependencies.planning_backend,
        planning_cache_root=repo_cache_root(config.root_dir),
        cache_only=offline_cache_only,
        cache_stats=planning_cache_stats,
    )
    decomposition_producer_sha256 = _decomposition_producer_sha256(
        runtime_topic,
        config.output_dir,
        dependencies.planning_backend,
    )
    passage_identity = _configured_passage_search_identity(
        retrieval_depth=config.retrieval.documents_per_query,
        passages_per_query=config.passage.passages_per_query,
        chunk_max_characters=config.passage.chunk_max_characters,
        chunk_overlap_characters=config.passage.chunk_overlap_characters,
        model=config.passage.model,
        device=config.passage.device,
    )
    passage_search = _build_topic_passage_search(
        runtime_topic,
        retriever=dependencies.retriever,
        scorer=dependencies.document_scorer,
        document_store_root=active_document_store_root,
        retrieval_cache_dir=config.retrieval.cache_dir,
        retrieval_index=config.retrieval.index,
        corpus_epoch=config.retrieval.corpus_epoch,
        score_cache_root=config.passage.score_cache_dir,
        device=config.passage.device,
        retrieval_depth=config.retrieval.documents_per_query,
        passages_per_query=config.passage.passages_per_query,
        chunk_max_characters=config.passage.chunk_max_characters,
        chunk_overlap_characters=config.passage.chunk_overlap_characters,
        cache_only=offline_cache_only,
        offline_source_document_store_root=offline_source_document_store_root,
    )
    if passage_search.identity != passage_identity:
        raise ValueError("constructed passage search identity differs from v2 config")
    passage_identity = passage_search.identity
    retrieval_outcome = _retrieve_topic(
        runtime_topic,
        decomposition,
        output_dir=config.output_dir,
        cache_dir=config.retrieval.cache_dir,
        code_commit=dependencies.code_commit,
        corpus_epoch=config.retrieval.corpus_epoch,
        retriever=dependencies.retriever,
        retrieval_depth=config.retrieval.documents_per_query,
        passage_search=passage_search,
        passage_identity=passage_identity,
        cache_only=offline_cache_only,
    )
    scoring_outcome = _score_topic(
        runtime_topic,
        decomposition,
        output_dir=config.output_dir,
        cache_dir=config.retrieval.cache_dir,
        score_cache_root=config.passage.score_cache_dir,
        code_commit=dependencies.code_commit,
        corpus_epoch=config.retrieval.corpus_epoch,
        retriever=None,
        scorer=None,
        device=config.passage.device,
        retrieval_depth=config.retrieval.documents_per_query,
        rerank_depth=config.retrieval.documents_per_query,
        selection_k=INTERNAL_FIXED_SELECTION_K,
        document_store_root=active_document_store_root,
        expected_retriever_identity=expected_retriever_identity,
        expected_passage_identity=passage_identity,
    )
    canonical = _canonical_topic(
        runtime_topic,
        decomposition,
        config=config,
        official_topics_sha256=identity,
        dependencies=dependencies,
        offline_cache_only=offline_cache_only,
        cache_stats=canonical_cache_stats,
        document_store_root=active_document_store_root,
    )
    topic_root = config.output_dir / runtime_topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        runtime_topic.id,
        DocumentStore(active_document_store_root),
        validation_session=canonical.validation_session,
    ) as records:
        projection_receipt = build_topic_projection(
            config,
            runtime_topic,
            records,
            expected_retriever_identity=expected_retriever_identity,
            decomposition_producer_sha256=decomposition_producer_sha256,
            config_sha256=config_sha256,
            canonical_manifest_bytes=canonical.provisional_manifest_bytes,
        )
    after_accounting = _runtime_cache_accounting(dependencies)
    stages = _cache_accounting_delta(before_accounting, after_accounting)
    stages["planning"].update(
        cache_hits=planning_cache_stats.get("cache_hits", 0),
        cache_misses=planning_cache_stats.get("cache_misses", 0),
        provider_calls=planning_cache_stats.get("provider_calls", 0),
    )
    stages["canonicalization"].update(
        cache_hits=canonical_cache_stats.get("cache_hits", 0),
        cache_misses=canonical_cache_stats.get("cache_misses", 0),
        provider_calls=canonical_cache_stats.get("provider_calls", 0),
    )
    _publish_topic_cache_operation_receipt(
        config=config,
        topic=runtime_topic,
        config_sha256=config_sha256,
        projection_manifest_sha256=projection_receipt.manifest_sha256,
        mode="offline-cache-only" if offline_cache_only else "online",
        phases={
            "planning": {"resumed": decomposition_resumed},
            "retrieval": {"resumed": retrieval_outcome.resumed},
            "scoring": {"resumed": scoring_outcome.resumed},
            "canonical": {"resumed": canonical.outcome.resumed},
        },
        stages=stages,
    )
    return _TopicTaskOutcome(runtime_topic.id, False, projection_receipt)


def _build_topic_passage_search(
    topic: Topic,
    *,
    retriever: Any | None,
    scorer: Any | None,
    document_store_root: Path,
    retrieval_cache_dir: Path,
    retrieval_index: str,
    corpus_epoch: str | None,
    score_cache_root: Path,
    device: str,
    retrieval_depth: int,
    passages_per_query: int,
    chunk_max_characters: int,
    chunk_overlap_characters: int,
    cache_only: bool = False,
    offline_source_document_store_root: Path | None = None,
) -> _TopicPassageSearchAdapter:
    if retriever is None:
        retriever = build_pyserini_retriever(
            retrieval_cache_dir,
            index=retrieval_index,
            hits=retrieval_depth,
            corpus_epoch=corpus_epoch,
            cache_only=cache_only,
        )
    if scorer is None:
        scorer = MixedbreadPassageScorer(
            score_cache_root=score_cache_root,
            device=device,
            read_only=cache_only,
        )
    scorer_identity = getattr(scorer, "identity", None)
    if not isinstance(scorer_identity, Mapping):
        raise TypeError("passage scorer must expose an identity mapping")
    actual_scorer_identity = dict(scorer_identity)
    if cache_only:
        if offline_source_document_store_root is None:
            raise ValueError("cache-only passage search requires source document cache")
        store: Any = _OfflineStagingDocumentStore(
            source_root=offline_source_document_store_root,
            stage_root=document_store_root,
        )
    else:
        store = DocumentStore(document_store_root)
    passage_identity = _configured_passage_search_identity(
        retrieval_depth=retrieval_depth,
        passages_per_query=passages_per_query,
        chunk_max_characters=chunk_max_characters,
        chunk_overlap_characters=chunk_overlap_characters,
        model=MIXEDBREAD_MODEL,
        device=device,
    )
    expected_scorer_identity = passage_identity["scorer"]
    if actual_scorer_identity != expected_scorer_identity:
        raise ValueError("passage scorer identity differs from v2 config")
    passage_identity = {
        **passage_identity,
        "scorer": actual_scorer_identity,
    }
    search = TopicPassageSearch(
        topic.id,
        store,
        _DepthBoundExternalRetriever(retriever, retrieval_depth),
        SemanticTextChunker(
            ChunkingConfig(
                max_characters=chunk_max_characters,
                overlap_characters=chunk_overlap_characters,
            )
        ),
        scorer,
        PassageSearchPolicy(
            retrieval_depth=retrieval_depth,
            passage_limit=passages_per_query,
        ),
    )
    return _TopicPassageSearchAdapter(search, store, passage_identity)


def _decompose_topic(
    topic: Topic,
    output_dir: Path,
    backend: Any,
    *,
    planning_cache_root: Path | None = None,
    cache_only: bool = False,
    cache_stats: dict[str, int] | None = None,
) -> tuple[ValidatedDecomposition, bool]:
    path = output_dir / topic.id / "decomposition" / "result.json"
    manifest_path = path.with_name("manifest.json")
    seed_manifest_path = output_dir / topic.id / PLANNING_SEED_MANIFEST_FILENAME
    seed_receipt_path = output_dir / PLANNING_SEED_RECEIPT_FILENAME
    if path.is_file():
        producer_sha256 = _decomposition_producer_sha256(
            topic,
            output_dir,
            backend,
        )
        result = load_validated_decomposition(topic, path)
        if cache_only:
            cached = extract_facets(
                topic,
                planning_cache_root=planning_cache_root,
                cache_only=True,
                cache_stats=cache_stats,
            )
            if cached != result.result:
                raise ValueError("planning cache differs from decomposition checkpoint")
        if _decomposition_producer_sha256(topic, output_dir, backend) != producer_sha256:
            raise ValueError("decomposition producer changed during validation")
        return result, True
    if (
        manifest_path.exists()
        or seed_manifest_path.exists()
        or _planning_seed_receipt_claims_topic(seed_receipt_path, topic.id)
    ):
        raise ValueError("decomposition producer exists without its result")
    planner_identity = _planner_identity(backend)
    if planning_cache_root is None:
        if backend is None:
            backend = OpenRouterDeepSeekFacetBackend()
        result = extract_facets(topic, backend)
    else:
        result = extract_facets(
            topic,
            backend,
            planning_cache_root=planning_cache_root,
            backend_factory=(OpenRouterDeepSeekFacetBackend if backend is None else None),
            cache_only=cache_only,
            cache_stats=cache_stats,
        )
    record = {
        "schema_version": SCHEMA,
        "topic": {"id": topic.id, "narrative": topic.narrative},
        "narrative_sha256": _hash(topic.narrative.encode()),
        "used_fallback": result.used_fallback,
        "error": result.error,
        "queries": jsonable(result.queries),
        "plan": _plan_payload(result.plan),
        "subnarratives": jsonable(result.subnarratives),
    }
    result_bytes = _json_bytes(record)
    _write_bytes(path, result_bytes)
    _write_json(
        manifest_path,
        {
            "schema_version": _DECOMPOSITION_MANIFEST_SCHEMA,
            "planner": planner_identity,
            "result_file": path.name,
            "result_bytes": len(result_bytes),
            "result_sha256": _hash(result_bytes),
        },
    )
    producer_sha256 = _decomposition_producer_sha256(topic, output_dir, backend)
    loaded = load_validated_decomposition(topic, path)
    if _decomposition_producer_sha256(topic, output_dir, backend) != producer_sha256:
        raise ValueError("decomposition producer changed during validation")
    return loaded, False


def _planning_seed_receipt_claims_topic(receipt_path: Path, topic_id: str) -> bool:
    """Return whether a valid aggregate seed receipt inventories ``topic_id``."""
    if not receipt_path.exists():
        return False
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise ValueError("planning seed aggregate receipt is unreadable") from exc
    receipt = _loads(receipt_bytes, "planning seed aggregate receipt")
    if (
        not isinstance(receipt, dict)
        or _json_bytes(receipt, pretty=False) != receipt_bytes
        or receipt.get("schema_version") != PLANNING_SEED_RECEIPT_SCHEMA
        or receipt.get("mode") != PLANNING_SEED_MODE
        or receipt.get("planning_backend_invocations") != 0
        or receipt.get("hosted_planning_calls") != 0
        or receipt.get("poison_backend_result") != "passed"
    ):
        raise ValueError("planning seed aggregate receipt is invalid")
    content_sha256 = receipt.get("receipt_content_sha256")
    receipt_content = dict(receipt)
    receipt_content.pop("receipt_content_sha256", None)
    if content_sha256 != _hash(_json_bytes(receipt_content, pretty=False)):
        raise ValueError("planning seed aggregate receipt digest is invalid")
    topic_ids = receipt.get("topic_ids")
    manifest_digests = receipt.get("manifest_digests")
    if (
        not isinstance(topic_ids, list)
        or not isinstance(manifest_digests, list)
        or any(not isinstance(item, str) or not item for item in topic_ids)
        or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in manifest_digests
        )
        or receipt.get("topic_manifest_digests") != manifest_digests
        or receipt.get("topic_count") != len(topic_ids)
        or len(manifest_digests) != len(topic_ids)
        or len(set(topic_ids)) != len(topic_ids)
    ):
        raise ValueError("planning seed aggregate topic inventory is invalid")
    return topic_id in topic_ids


def _decomposition_producer_sha256(
    topic: Topic,
    output_dir: Path,
    backend: Any | None,
) -> str:
    """Validate and identify the sole producer authorized for one decomposition."""
    result_path = output_dir / topic.id / "decomposition" / "result.json"
    live_manifest_path = result_path.with_name("manifest.json")
    seed_manifest_path = output_dir / topic.id / PLANNING_SEED_MANIFEST_FILENAME
    seed_receipt_path = output_dir / PLANNING_SEED_RECEIPT_FILENAME
    if not result_path.is_file():
        raise ValueError("decomposition result is missing")
    live_present = live_manifest_path.exists()
    seed_present = seed_manifest_path.exists()
    seed_claimed = _planning_seed_receipt_claims_topic(seed_receipt_path, topic.id)
    if seed_claimed and not seed_present:
        raise ValueError("planning seed producer receipt is incomplete")
    if live_present and seed_present:
        raise ValueError("decomposition has multiple producer receipts")
    if live_present:
        manifest_bytes, result_bytes = _validate_decomposition_manifest(
            result_path,
            live_manifest_path,
            planner_identity=_planner_identity(backend),
        )
        identity = {
            "schema_version": _DECOMPOSITION_PRODUCER_SCHEMA,
            "mode": "live-planner",
            "manifest_sha256": _hash(manifest_bytes),
            "result_sha256": _hash(result_bytes),
        }
        return _hash(_json_bytes(identity, pretty=False))
    if seed_present:
        return _seeded_decomposition_producer_sha256(
            topic,
            output_dir,
            result_path,
            seed_manifest_path,
            seed_receipt_path,
        )
    raise ValueError("decomposition checkpoint producer receipt is missing")


def _seeded_decomposition_producer_sha256(
    topic: Topic,
    output_dir: Path,
    result_path: Path,
    manifest_path: Path,
    receipt_path: Path,
) -> str:
    """Bind one seeded result to its topic manifest and aggregate receipt."""
    try:
        manifest_bytes = manifest_path.read_bytes()
        receipt_bytes = receipt_path.read_bytes()
        result_bytes = result_path.read_bytes()
    except OSError as exc:
        raise ValueError("planning seed producer receipt is incomplete") from exc
    manifest = _loads(manifest_bytes, "planning seed topic manifest")
    receipt = _loads(receipt_bytes, "planning seed aggregate receipt")
    if (
        not isinstance(manifest, dict)
        or _json_bytes(manifest, pretty=False) != manifest_bytes
        or manifest.get("schema_version") != PLANNING_SEED_MANIFEST_SCHEMA
        or manifest.get("mode") != PLANNING_SEED_MODE
        or manifest.get("topic_id") != topic.id
        or manifest.get("narrative_sha256") != _hash(topic.narrative.encode())
        or manifest.get("destination_relative_path")
        != f"{topic.id}/decomposition/result.json"
        or manifest.get("destination_bytes") != len(result_bytes)
        or manifest.get("destination_byte_length") != len(result_bytes)
        or manifest.get("source_bytes") != len(result_bytes)
        or manifest.get("source_byte_length") != len(result_bytes)
        or manifest.get("destination_sha256") != _hash(result_bytes)
        or manifest.get("source_sha256") != _hash(result_bytes)
        or manifest.get("bytes_equal") is not True
    ):
        raise ValueError("planning seed topic manifest is invalid")
    if (
        not isinstance(receipt, dict)
        or _json_bytes(receipt, pretty=False) != receipt_bytes
        or receipt.get("schema_version") != PLANNING_SEED_RECEIPT_SCHEMA
        or receipt.get("mode") != PLANNING_SEED_MODE
        or receipt.get("planning_backend_invocations") != 0
        or receipt.get("hosted_planning_calls") != 0
        or receipt.get("poison_backend_result") != "passed"
    ):
        raise ValueError("planning seed aggregate receipt is invalid")
    content_sha256 = receipt.get("receipt_content_sha256")
    receipt_content = dict(receipt)
    receipt_content.pop("receipt_content_sha256", None)
    if content_sha256 != _hash(_json_bytes(receipt_content, pretty=False)):
        raise ValueError("planning seed aggregate receipt digest is invalid")
    topic_ids = receipt.get("topic_ids")
    manifest_digests = receipt.get("manifest_digests")
    topic_manifest_digests = receipt.get("topic_manifest_digests")
    if (
        not isinstance(topic_ids, list)
        or not isinstance(manifest_digests, list)
        or any(not isinstance(item, str) or not item for item in topic_ids)
        or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in manifest_digests
        )
        or topic_manifest_digests != manifest_digests
        or receipt.get("topic_count") != len(topic_ids)
        or len(manifest_digests) != len(topic_ids)
        or len(set(topic_ids)) != len(topic_ids)
        or topic.id not in topic_ids
    ):
        raise ValueError("planning seed aggregate topic inventory is invalid")
    topic_index = topic_ids.index(topic.id)
    if manifest_digests[topic_index] != _hash(manifest_bytes):
        raise ValueError("planning seed topic manifest is not in its aggregate receipt")
    import trec_rag.facet_retrieval_lanes as lane_projector

    projector_path = getattr(lane_projector, "__file__", None)
    if not isinstance(projector_path, str):
        raise ValueError("planning seed lane projector source is unavailable")
    current_projector_sha256 = _hash(Path(projector_path).read_bytes())
    if (
        receipt.get("validator_module_sha256")
        != manifest.get("validator_module_sha256")
        or receipt.get("validator_module_sha256") != _hash(Path(__file__).read_bytes())
        or receipt.get("validator_identity") != manifest.get("validator_identity")
        or receipt.get("lane_projector_module_sha256")
        != manifest.get("lane_projector_module_sha256")
        or receipt.get("lane_projector_module_sha256")
        != current_projector_sha256
        or receipt.get("lane_projector_identity")
        != manifest.get("lane_projector_identity")
        or receipt.get("lane_projector_version")
        != manifest.get("lane_projector_version")
        or receipt.get("lane_projector_version")
        != lane_projector.FACET_RETRIEVAL_LANE_PROJECTOR_VERSION
    ):
        raise ValueError("planning seed validator or projector identity changed")
    identity = {
        "schema_version": _DECOMPOSITION_PRODUCER_SCHEMA,
        "mode": PLANNING_SEED_MODE,
        "result_sha256": _hash(result_bytes),
        "topic_manifest_sha256": _hash(manifest_bytes),
        "aggregate_receipt_sha256": _hash(receipt_bytes),
    }
    return _hash(_json_bytes(identity, pretty=False))


def _planner_identity(backend: Any | None) -> dict[str, object]:
    if backend is None or isinstance(backend, OpenRouterDeepSeekFacetBackend):
        identity: object = {
            "backend": (
                "trec_rag.facet_extraction.OpenRouterDeepSeekFacetBackend"
            ),
            "model": OPENROUTER_DEEPSEEK_MODEL,
            "prompt_version": FACET_PROMPT_VERSION,
            "response_schema": FACET_SCHEMA_VERSION,
        }
    else:
        identity = getattr(backend, "identity", None)
    if not isinstance(identity, Mapping) or not identity:
        raise TypeError("planning backend must expose a non-empty identity mapping")
    value = dict(identity)
    try:
        restored = _loads(_json_bytes(value, pretty=False), "planner identity")
    except (TypeError, ValueError) as exc:
        raise ValueError("planner identity must be immutable JSON data") from exc
    if restored != value:
        raise ValueError("planner identity must be immutable JSON data")
    return value


def _validate_decomposition_manifest(
    result_path: Path,
    manifest_path: Path,
    *,
    planner_identity: Mapping[str, object],
) -> tuple[bytes, bytes]:
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("decomposition checkpoint manifest is missing") from exc
    manifest = _loads(manifest_bytes, "decomposition checkpoint manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "planner",
            "result_file",
            "result_bytes",
            "result_sha256",
        }
        or manifest.get("schema_version") != _DECOMPOSITION_MANIFEST_SCHEMA
        or _json_bytes(manifest) != manifest_bytes
    ):
        raise ValueError("decomposition checkpoint manifest is invalid")
    if manifest.get("planner") != dict(planner_identity):
        raise ValueError("decomposition checkpoint planner identity changed")
    result_bytes = result_path.read_bytes()
    if (
        manifest.get("result_file") != result_path.name
        or type(manifest.get("result_bytes")) is not int
        or manifest["result_bytes"] != len(result_bytes)
        or not isinstance(manifest.get("result_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest["result_sha256"])
        or manifest["result_sha256"] != _hash(result_bytes)
    ):
        raise ValueError("decomposition checkpoint result changed")
    return manifest_bytes, result_bytes


def _canonical_topic(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    config: FacetPilotConfig,
    official_topics_sha256: str,
    dependencies: _RuntimeDependencies,
    offline_cache_only: bool = False,
    cache_stats: dict[str, int] | None = None,
    document_store_root: Path | None = None,
) -> _CanonicalPhaseResult:
    """Run the local evidence tail and canonical stage for one topic."""
    root = config.output_dir / topic.id
    active_document_store_root = (
        document_store_dir(config.root_dir)
        if document_store_root is None
        else Path(document_store_root)
    )
    handoff_root = root / "canonical" / "handoff"
    passage_results = tuple(
        audited.passage_result
        for audited in decode_retrieval_audit(
            topic,
            decomposition,
            (root / "retrieval" / "audit.json").read_bytes(),
            requested_depth=config.retrieval.documents_per_query,
        )
        if audited.passage_result is not None
    )
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=config.output_dir,
        output_dir=handoff_root,
        code_commit=dependencies.code_commit,
        official_topics_sha256=official_topics_sha256,
        passage_results=passage_results,
        document_store_root=active_document_store_root,
    )
    canonical_root = root / "canonical"
    selected_budget = config.nuggets.evidence_budget_per_subnarrative
    selection_policy = SelectionPolicy(budgets=(selected_budget,))
    manifest = canonical_root / "complete.json"
    artifacts = (
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "records.sqlite3",
        "canonical/records-manifest.json",
        "canonical/subnarrative-selections.jsonl",
        "canonical/selection-manifest.json",
        "canonical/canonical-nuggets.jsonl",
        "canonical/canonical-nugget-manifest.json",
    )
    expected = {
        "schema_version": SCHEMA,
        "phase": "canonical",
        "topic_id": topic.id,
        "official_topics_sha256": official_topics_sha256,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "scoring_manifest_sha256": handoff.scoring_manifest_sha256,
        "handoff_manifest_sha256": _hash(handoff.manifest_path.read_bytes()),
        "code_commit": dependencies.code_commit,
        "input_roles": [
            "official_topic_narrative",
            "validated_generated_decomposition",
            "sealed_scoring_checkpoint",
        ],
        "selection_policy": {
            "budgets": [selected_budget],
            "precluster_limit": selection_policy.precluster_limit,
            "semantic_threshold": selection_policy.semantic_threshold,
            "mmr_lambda": selection_policy.mmr_lambda,
        },
        "selected_budget": selected_budget,
        "canonical_claim_cap": config.nuggets.maximum_claims_per_subnarrative,
        "canonical_supporting_document_cap": (
            config.nuggets.maximum_supporting_documents_per_claim
        ),
    }
    if _resume(manifest, root, expected, artifacts) and not offline_cache_only:
        return _CanonicalPhaseResult(
            TopicPhaseOutcome(topic.id, "canonical", manifest, True),
            manifest.read_bytes(),
        )

    candidate_artifacts = generate_candidate_artifacts(
        handoff,
        run_id=config.run_id,
        score_cache_root=config.passage.score_cache_dir,
        device=config.passage.device,
        document_store_root=active_document_store_root,
        scorer=dependencies.candidate_scorer,
        facets=(
            FacetRecord("original", topic.narrative, "initial"),
            *tuple(
                FacetRecord(facet.subnarrative_id, facet.text, "initial")
                for facet in decomposition.result.subnarratives
            ),
        ),
        passage_results=passage_results,
    )
    selection_artifacts = select_evidence_artifacts(
        candidate_artifacts,
        handoff.contexts_path,
        device=config.passage.device,
        similarity=dependencies.similarity,
        policy=selection_policy,
    )
    run_canonical_stage(
        selections_path=selection_artifacts.selections_path,
        selection_manifest_path=selection_artifacts.manifest_path,
        selected_budget=selected_budget,
        max_canonical_claims=config.nuggets.maximum_claims_per_subnarrative,
        max_supporting_documents_per_claim=(
            config.nuggets.maximum_supporting_documents_per_claim
        ),
        output_path=canonical_root / "canonical-nuggets.jsonl",
        manifest_path=canonical_root / "canonical-nugget-manifest.json",
        cache_dir=canonical_response_cache_dir(config.root_dir),
        backend_factory=dependencies.canonical_backend_factory,
        cache_ignore_checker=dependencies.cache_ignore_checker,
        cache_only=offline_cache_only,
        cache_stats=cache_stats,
    )
    provisional_manifest_bytes = _complete_manifest_bytes(
        root, expected, artifacts, pretty=False
    )
    return _CanonicalPhaseResult(
        TopicPhaseOutcome(topic.id, "canonical", manifest, False),
        provisional_manifest_bytes,
        candidate_artifacts.validation_session,
    )


def _topics_sha256(topics: Sequence[Topic]) -> str:
    return _hash(
        json.dumps(
            [
                {"id": topic.id, "narrative": topic.narrative}
                for topic in topics
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def canonical_response_cache_dir(root_dir: Path) -> Path:
    """Return the shared, prompt-versioned response cache for canonical calls."""
    return repo_cache_root(root_dir) / "canonical" / PROMPT_VERSION


def document_store_dir(root_dir: Path) -> Path:
    """Return the shared versioned content-addressed document-store root."""
    return repo_cache_root(root_dir) / "documents" / "v1"


def _tracked_worktree_is_dirty(repo: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=Path(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _runtime_topic(topic: Topic) -> Topic:
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    if (
        not isinstance(topic.id, str)
        or not topic.id
        or not isinstance(topic.narrative, str)
        or not topic.narrative.strip()
    ):
        raise ValueError("topic ID and narrative must be non-empty text")
    if not _SAFE_TOPIC_ID.fullmatch(topic.id):
        raise ValueError("topic ID must be a safe path component")
    return Topic(topic.id, "", topic.narrative)


def _plan_payload(plan: GeneratedQueryPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": plan.topic_id,
        "subnarratives": [
            {
                "subnarrative": row.text,
                "bm25_queries": list(row.bm25_queries),
            }
            for row in plan.subnarratives
        ],
    }


def _candidates(
    topic: Topic,
    query: QueryVariant,
    value: object,
) -> list[RetrievedCandidate]:
    if not isinstance(value, list) or any(
        not isinstance(row, RetrievedCandidate) for row in value
    ):
        raise ValueError("retriever returned invalid candidates")
    for row in value:
        if (
            row.topic_id != topic.id
            or row.variant_name != query.variant_name
            or row.query_text != query.query_text
            or not row.docid
            or not row.text.strip()
            or isinstance(row.rank, bool)
            or not isinstance(row.rank, int)
            or isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(row.score)
        ):
            raise ValueError("retrieved candidate identity, rank, or score is invalid")
    rows = sorted(value, key=lambda row: row.rank)
    if [row.rank for row in rows] != list(range(1, len(rows) + 1)) or len(
        {row.docid for row in rows}
    ) != len(rows):
        raise ValueError("retrieval ranks and document IDs must be unique and contiguous")
    return rows


def _retriever_identity(
    retriever: Any,
    *,
    retrieval_depth: int,
) -> dict[str, object]:
    exposed_identity = getattr(retriever, "identity", None)
    if isinstance(exposed_identity, Mapping):
        identity = dict(exposed_identity)
    else:
        config = retriever.config
        if config.type == "pyserini_remote":
            raise ValueError(
                "Pyserini retriever must expose an immutable checkpoint identity"
            )
        remote = retriever.client.config
        identity = {
            "name": config.name,
            "type": config.type,
            "index": config.index,
            "index_url": remote.index_url,
            "hits": config.hits,
        }
    if identity.get("type") == "pyserini_remote":
        required = {
            "name",
            "type",
            "index",
            "index_url",
            "hits",
            "corpus_epoch",
            "retrieval_cache_schema",
            "parser_version",
            "extractor_version",
            "field_path",
            "scoring_normalizer_version",
        }
        missing = sorted(required - set(identity))
        if missing:
            raise ValueError(
                "Pyserini checkpoint identity is incomplete: " + ", ".join(missing)
            )
    if identity.get("hits") != retrieval_depth:
        raise ValueError("retriever must bind the configured retrieval depth")
    return identity


def _configured_passage_search_identity(
    *,
    retrieval_depth: int,
    passages_per_query: int,
    chunk_max_characters: int,
    chunk_overlap_characters: int,
    model: str,
    device: str,
    batch_size: int = 8,
    implementation_version: int = 1,
) -> dict[str, object]:
    """Return the config-bound shared-search identity used by both phases."""
    return {
        "schema_version": "topic-passage-search-v1",
        "scorer": {
            "backend": PASSAGE_SCORER_BACKEND,
            "backend_version": PASSAGE_SCORER_BACKEND_VERSION,
            "model": model,
            "model_revision": MIXEDBREAD_REVISION,
            "score_representation": PASSAGE_SCORER_SCORE_REPRESENTATION,
            "inference_dtype": PASSAGE_SCORER_INFERENCE_DTYPE,
            "max_length": PASSAGE_SCORER_MAX_LENGTH,
            "batch_size": batch_size,
            "input_policy": PASSAGE_SCORER_INPUT_POLICY,
            "device": _choose_device(device),
            "implementation_version": implementation_version,
        },
        "chunker": {
            "backend": "trec_rag.chunking.SemanticTextChunker",
            "max_characters": chunk_max_characters,
            "overlap_characters": chunk_overlap_characters,
            "trim": True,
        },
        "policy": {
            "retrieval_depth": retrieval_depth,
            "passage_limit": passages_per_query,
            "max_attempts": 3,
        },
    }


def _validate_passage_result_identity(
    result: PassageSearchResult,
    expected: Mapping[str, object],
) -> None:
    expected_chunker = expected.get("chunker")
    if not isinstance(expected_chunker, Mapping):
        raise ValueError("passage search chunker identity is invalid")
    if any(dict(passage.chunker_identity) != dict(expected_chunker) for passage in result.passages):
        raise ValueError("passage search chunker identity changed")


def _expected(
    phase: str,
    topic: Topic,
    decomposition: ValidatedDecomposition,
    commit: str,
    retriever: dict[str, object],
    passage_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": commit,
        "retriever": retriever,
        "passage_search": dict(passage_identity),
    }


def _inputs(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    commit: str,
) -> None:
    _runtime_topic(topic)
    if (
        decomposition.topic_id != topic.id
        or decomposition.narrative_sha256 != _hash(topic.narrative.encode())
    ):
        raise ValueError("decomposition differs from the exact official narrative")
    if not _COMMIT.fullmatch(commit):
        raise ValueError("code commit must be a full lowercase SHA-1")


def _audit_lane(
    lane: Any,
    returned: int,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "lane_name": lane.retrieval_query.variant_name,
        "subnarrative_id": lane.subnarrative_id,
        "bm25_query_sha256": lane.bm25_query_sha256,
        "semantic_query_sha256": lane.semantic_query_sha256,
        "returned_count": returned,
        "retained_count": len(candidates),
        "candidates": candidates,
    }


def _audit(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    lanes: list[dict[str, object]],
    *,
    retrieval_depth: int,
    passage_search_results: Sequence[PassageSearchResult],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "requested_depth": retrieval_depth,
        "lanes": lanes,
        "passage_search_results": [
            _passage_result_json(result) for result in passage_search_results
        ],
    }


def _result_audit(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    result: FacetRetrievalResult,
) -> dict[str, object]:
    lanes = []
    for lane in result.lanes:
        candidates = [
            {
                "docid": row.docid,
                "bm25_rank": row.bm25_rank,
                "bm25_score": row.bm25_score,
                "text_sha256": row.text_sha256,
            }
            for row in lane.retrieval_audit_candidates
        ]
        lanes.append(_audit_lane(lane.lane, lane.retrieval_returned_count, candidates))
    passage_results = tuple(
        lane.passage_result
        for lane in result.lanes
        if lane.passage_result is not None
    )
    if len(passage_results) != len(result.lanes):
        raise ValueError("fixed retrieval result is missing a sealed passage result")
    return _audit(
        topic,
        decomposition,
        lanes,
        retrieval_depth=result.lanes[0].retrieval_requested_depth,
        passage_search_results=passage_results,
    )


def _score_row(score: LaneDocumentScore) -> dict[str, object]:
    row = jsonable(score)
    row["text_sha256"] = _hash(row.pop("text").encode())
    row.pop("bm25_query")
    row.pop("semantic_query")
    return row


def _project_subnarrative_scores(
    result: FacetRetrievalResult,
    subnarratives: Sequence[Any],
) -> list[dict[str, object]]:
    """Persist semantic lanes as sealed passage rows, without a cross matrix."""
    expected_ids = {facet.subnarrative_id for facet in subnarratives}
    rows: list[dict[str, object]] = []
    for lane in result.lanes:
        subnarrative_id = lane.lane.subnarrative_id
        if subnarrative_id is None or subnarrative_id not in expected_ids:
            continue
        if lane.passage_result is None:
            raise ValueError("semantic lane is missing its sealed passage result")
        rows.append(
            {
                "schema_version": "facet_passage_lane_v2",
                "subnarrative_id": subnarrative_id,
                "passage_search": _passage_result_json(lane.passage_result),
            }
        )
    return rows


def _selection(
    result: FacetRetrievalResult,
    selected_hash: str,
) -> dict[str, object]:
    selection = result.selection
    return {
        "schema_version": SELECTION_SCHEMA,
        "topic_id": result.topic_id,
        "requested_count": selection.requested_count,
        "selected_count": len(selection.documents),
        "complete": selection.complete,
        "all_lanes_exhausted": selection.all_lanes_exhausted,
        "selected_set_sha256": selected_hash,
        "selected_order": [row.docid for row in selection.documents],
        "lane_statuses": jsonable(selection.lane_statuses),
        "trace": jsonable(selection.trace),
        "memberships": [
            {
                "docid": row.docid,
                "lanes": [
                    {
                        "lane_name": score.lane_name,
                        "aggregate_rank": score.aggregate_rank,
                        "aggregate_score": score.aggregate_score,
                        "bm25_rank": score.bm25_rank,
                        "bm25_score": score.bm25_score,
                    }
                    for score in row.lane_scores
                ],
            }
            for row in selection.documents
        ],
        "original_only_control": [
            {
                "docid": row.docid,
                "aggregate_rank": row.aggregate_rank,
                "aggregate_score": row.aggregate_score,
            }
            for row in result.original_only_control
        ],
        "union_pool": [
            {
                "docid": row.docid,
                "first_seen_lane": row.first_seen_lane,
                "memberships": [score.lane_name for score in row.lane_scores],
            }
            for row in result.union_pool
        ],
        "lanes": [
            {
                "lane_name": row.lane.retrieval_query.variant_name,
                "retrieval_audit_sha256": row.retrieval_audit_sha256,
                "selection_eligible_count": row.selection_eligible_count,
                "audit_only_count": row.audit_only_count,
            }
            for row in result.lanes
        ],
    }


def _resume(
    manifest: Path,
    root: Path,
    expected: dict[str, object],
    artifacts: tuple[str, ...],
    *,
    required: bool = False,
) -> bool:
    if not manifest.exists():
        if required:
            raise ValueError("required phase checkpoint is missing")
        return False
    value = _loads(manifest.read_bytes(), "checkpoint manifest")
    fields = set(expected) | {"artifacts"}
    if expected["phase"] == "score":
        fields.add("selected_set_sha256")
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or any(value.get(key) != item for key, item in expected.items())
    ):
        raise ValueError("checkpoint input identity changed")
    receipts = value.get("artifacts")
    if (
        not isinstance(receipts, list)
        or [row.get("relative_path") for row in receipts if isinstance(row, dict)]
        != list(artifacts)
    ):
        raise ValueError("checkpoint artifacts changed")
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise ValueError("checkpoint artifact receipt changed")
        byte_count, digest = _file_receipt(root / receipt["relative_path"])
        if receipt.get("bytes") != byte_count or receipt.get("sha256") != digest:
            raise ValueError("checkpoint artifact hash changed")
    return True


def _complete(
    manifest: Path,
    root: Path,
    values: dict[str, object],
    artifacts: tuple[str, ...],
) -> None:
    _write_bytes(manifest, _complete_manifest_bytes(root, values, artifacts))


def _complete_atomic(
    manifest: Path,
    root: Path,
    values: dict[str, object],
    artifacts: tuple[str, ...],
) -> None:
    _write_bytes(manifest, _complete_manifest_bytes(root, values, artifacts))


def _complete_manifest_bytes(
    root: Path,
    values: dict[str, object],
    artifacts: tuple[str, ...],
    *,
    pretty: bool = True,
) -> bytes:
    receipts = []
    for relative in artifacts:
        byte_count, digest = _file_receipt(root / relative)
        receipts.append(
            {"relative_path": relative, "bytes": byte_count, "sha256": digest}
        )
    return _json_bytes(values | {"artifacts": receipts}, pretty=pretty)


def _json_bytes(value: object, *, pretty: bool = True) -> bytes:
    options: dict[str, object] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _json_bytes(value))


def _write_bytes(path: Path, body: bytes) -> None:
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _loads(value: bytes, label: str) -> object:
    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        return json.loads(
            value,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _hash(value: bytes) -> str:
    return sha256(value).hexdigest()


def _file_receipt(path: Path) -> tuple[int, str]:
    digest = sha256()
    byte_count = 0
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return byte_count, digest.hexdigest()


def _validated_cache_operation_stages(
    stages: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    if not isinstance(stages, Mapping) or set(stages) != set(CACHE_OPERATION_STAGE_NAMES):
        raise ValueError("cache operation stages changed")
    validated: dict[str, dict[str, int]] = {}
    for stage_name in CACHE_OPERATION_STAGE_NAMES:
        counters = stages[stage_name]
        if not isinstance(counters, Mapping) or set(counters) != set(
            _CACHE_OPERATION_COUNTER_NAMES
        ):
            raise ValueError(f"cache operation {stage_name} counters changed")
        row: dict[str, int] = {}
        for counter_name in _CACHE_OPERATION_COUNTER_NAMES:
            value = counters[counter_name]
            if type(value) is not int or value < 0:
                raise ValueError("cache operation counters must be non-negative integers")
            row[counter_name] = value
        validated[stage_name] = row
    return validated


def _validated_cache_operation_phases(
    phases: Mapping[str, Mapping[str, bool]],
) -> dict[str, dict[str, bool]]:
    if not isinstance(phases, Mapping) or set(phases) != set(
        _CACHE_OPERATION_PHASE_NAMES
    ):
        raise ValueError("cache operation phases changed")
    validated: dict[str, dict[str, bool]] = {}
    for phase_name in _CACHE_OPERATION_PHASE_NAMES:
        phase = phases[phase_name]
        if (
            not isinstance(phase, Mapping)
            or set(phase) != {"resumed"}
            or not isinstance(phase["resumed"], bool)
        ):
            raise ValueError("cache operation phase outcome is invalid")
        validated[phase_name] = {"resumed": phase["resumed"]}
    return validated


def _operation_payload_with_digest(payload: Mapping[str, object]) -> dict[str, object]:
    content = dict(payload)
    content["receipt_content_sha256"] = _hash(_json_bytes(content, pretty=False))
    return content


def _publish_create_only_json(path: Path, payload: Mapping[str, object]) -> bytes:
    body = _json_bytes(dict(payload), pretty=False)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"conflicting immutable operation receipt: {path}")
        return body
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ValueError(f"conflicting immutable operation receipt: {path}")
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return body


def _publish_topic_cache_operation_receipt(
    *,
    config: FacetPilotConfig,
    topic: Topic,
    config_sha256: str,
    projection_manifest_sha256: str,
    mode: str,
    phases: Mapping[str, Mapping[str, bool]],
    stages: Mapping[str, Mapping[str, int]],
) -> _CacheOperationReceipt:
    if mode not in {"online", "offline-cache-only"}:
        raise ValueError("cache operation mode is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise ValueError("cache operation config_sha256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", projection_manifest_sha256):
        raise ValueError("cache operation projection digest is invalid")
    validated_stages = _validated_cache_operation_stages(stages)
    validated_phases = _validated_cache_operation_phases(phases)
    if mode == "offline-cache-only" and any(
        counters[counter] != 0
        for counters in validated_stages.values()
        for counter in (
            "cache_misses",
            "network_calls",
            "provider_calls",
            "model_batches",
        )
    ):
        raise ValueError(
            "offline cache operation requires zero misses, calls, and model batches"
        )
    payload = _operation_payload_with_digest(
        {
            "schema_version": _CACHE_OPERATION_RECEIPT_SCHEMA,
            "mode": mode,
            "run_id": config.run_id,
            "topic_id": topic.id,
            "config_sha256": config_sha256,
            "projection_manifest_sha256": projection_manifest_sha256,
            "phases": validated_phases,
            "stages": validated_stages,
        }
    )
    path = config.output_dir / topic.id / "cache-operation-receipt.json"
    body = _publish_create_only_json(path, payload)
    return _CacheOperationReceipt(
        topic_id=topic.id,
        path=path,
        sha256=_hash(body),
        projection_manifest_sha256=projection_manifest_sha256,
        stages=MappingProxyType(
            {
                name: MappingProxyType(dict(counters))
                for name, counters in validated_stages.items()
            }
        ),
    )


def _read_topic_cache_operation_receipt(
    *,
    config: FacetPilotConfig,
    topic: Topic,
    config_sha256: str,
    projection_manifest_sha256: str,
    mode: str,
) -> _CacheOperationReceipt:
    path = config.output_dir / topic.id / "cache-operation-receipt.json"
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ValueError("cache operation topic receipt is missing") from exc
    value = _loads(body, "cache operation topic receipt")
    if not isinstance(value, dict) or _json_bytes(value, pretty=False) != body:
        raise ValueError("cache operation topic receipt is not canonical")
    expected_fields = {
        "schema_version",
        "mode",
        "run_id",
        "topic_id",
        "config_sha256",
        "projection_manifest_sha256",
        "phases",
        "stages",
        "receipt_content_sha256",
    }
    content = dict(value)
    digest = content.pop("receipt_content_sha256", None)
    if (
        set(value) != expected_fields
        or digest != _hash(_json_bytes(content, pretty=False))
        or value.get("schema_version") != _CACHE_OPERATION_RECEIPT_SCHEMA
        or value.get("mode") != mode
        or value.get("run_id") != config.run_id
        or value.get("topic_id") != topic.id
        or value.get("config_sha256") != config_sha256
        or value.get("projection_manifest_sha256") != projection_manifest_sha256
    ):
        raise ValueError("cache operation topic receipt identity changed")
    phases = _validated_cache_operation_phases(value.get("phases", {}))
    stages = _validated_cache_operation_stages(value.get("stages", {}))
    if mode == "offline-cache-only" and any(
        counters[counter] != 0
        for counters in stages.values()
        for counter in (
            "cache_misses",
            "network_calls",
            "provider_calls",
            "model_batches",
        )
    ):
        raise ValueError(
            "offline cache operation requires zero misses, calls, and model batches"
        )
    del phases
    return _CacheOperationReceipt(
        topic_id=topic.id,
        path=path,
        sha256=_hash(body),
        projection_manifest_sha256=projection_manifest_sha256,
        stages=MappingProxyType(
            {
                name: MappingProxyType(dict(counters))
                for name, counters in stages.items()
            }
        ),
    )


def _publish_cache_operation_manifest(
    *,
    config: FacetPilotConfig,
    topics: Sequence[Topic],
    config_sha256: str,
    mode: str,
    receipts: Sequence[_CacheOperationReceipt],
    export_manifest_path: Path,
) -> Path:
    ordered_topics = tuple(topics)
    ordered_receipts = tuple(receipts)
    if len(ordered_topics) != len(ordered_receipts) or any(
        topic.id != receipt.topic_id
        for topic, receipt in zip(ordered_topics, ordered_receipts, strict=True)
    ):
        raise ValueError("cache operation topic receipt order changed")
    totals = {
        stage: {counter: 0 for counter in _CACHE_OPERATION_COUNTER_NAMES}
        for stage in CACHE_OPERATION_STAGE_NAMES
    }
    receipt_digests: list[str] = []
    projection_digests: list[str] = []
    for topic, receipt in zip(ordered_topics, ordered_receipts, strict=True):
        body = receipt.path.read_bytes()
        if _hash(body) != receipt.sha256:
            raise ValueError("cache operation topic receipt changed")
        value = _loads(body, "cache operation topic receipt")
        if not isinstance(value, dict) or _json_bytes(value, pretty=False) != body:
            raise ValueError("cache operation topic receipt is not canonical")
        content = dict(value)
        content_digest = content.pop("receipt_content_sha256", None)
        if (
            content_digest != _hash(_json_bytes(content, pretty=False))
            or value.get("schema_version") != _CACHE_OPERATION_RECEIPT_SCHEMA
            or value.get("mode") != mode
            or value.get("run_id") != config.run_id
            or value.get("topic_id") != topic.id
            or value.get("config_sha256") != config_sha256
            or value.get("projection_manifest_sha256")
            != receipt.projection_manifest_sha256
        ):
            raise ValueError("cache operation topic receipt identity changed")
        stages = _validated_cache_operation_stages(value.get("stages", {}))
        for stage_name, counters in stages.items():
            for counter_name, counter_value in counters.items():
                totals[stage_name][counter_name] += counter_value
        receipt_digests.append(receipt.sha256)
        projection_digests.append(receipt.projection_manifest_sha256)
    export_path = Path(export_manifest_path)
    export_bytes = export_path.read_bytes()
    payload = _operation_payload_with_digest(
        {
            "schema_version": _CACHE_OPERATION_MANIFEST_SCHEMA,
            "mode": mode,
            "run_id": config.run_id,
            "config_sha256": config_sha256,
            "topic_ids": [topic.id for topic in ordered_topics],
            "topic_receipt_sha256s": receipt_digests,
            "projection_manifest_sha256s": projection_digests,
            "retrieval_export_manifest": export_path.name,
            "retrieval_export_manifest_sha256": _hash(export_bytes),
            "totals": totals,
        }
    )
    path = config.output_dir / "cache-operation-manifest.json"
    _publish_create_only_json(path, payload)
    return path


def _nonnegative_counter(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _accounting_values(value: object) -> dict[str, int]:
    if isinstance(value, Mapping):
        getter = value.get
    else:
        def getter(name: str, default: object = 0) -> object:
            return getattr(value, name, default)

    return {
        "cache_hits": _nonnegative_counter(getter("cache_hits", getter("hits", 0))),
        "cache_misses": _nonnegative_counter(
            getter("cache_misses", getter("misses", 0))
        ),
        "model_batches": _nonnegative_counter(getter("model_batches", 0)),
    }


def _required_local_accounting(
    provider: object,
    *,
    attribute: str,
    label: str,
) -> dict[str, int]:
    try:
        value = getattr(provider, attribute)
    except Exception as exc:
        raise TypeError(f"{label} must expose exact cache accounting") from exc
    names = ("cache_hits", "cache_misses", "model_batches")
    if isinstance(value, Mapping):
        if any(name not in value for name in names):
            raise TypeError(f"{label} must expose exact cache accounting")
        raw = tuple(value[name] for name in names)
    else:
        if any(not hasattr(value, name) for name in names):
            raise TypeError(f"{label} must expose exact cache accounting")
        raw = tuple(getattr(value, name) for name in names)
    if any(type(item) is not int or item < 0 for item in raw):
        raise TypeError(f"{label} must expose exact cache accounting")
    return dict(zip(names, raw, strict=True))


def _runtime_cache_accounting(
    dependencies: _RuntimeDependencies,
) -> dict[str, dict[str, int]]:
    stages = {
        name: {counter: 0 for counter in _CACHE_OPERATION_COUNTER_NAMES}
        for name in CACHE_OPERATION_STAGE_NAMES
    }
    retriever = dependencies.retriever
    if retriever is not None:
        cache_summary = getattr(retriever, "cache_summary", None)
        transport_calls = getattr(retriever, "transport_calls", None)
        if (
            not callable(cache_summary)
            or type(transport_calls) is not int
            or transport_calls < 0
        ):
            raise TypeError("retriever must expose exact cache accounting")
        summary = cache_summary()
        if (
            not isinstance(summary, Mapping)
            or not {"hits", "misses"}.issubset(summary)
            or any(
                type(summary[name]) is not int or summary[name] < 0
                for name in ("hits", "misses")
            )
        ):
            raise TypeError("retriever must expose exact cache accounting")
        retrieval_values = _accounting_values(summary)
        stages["retrieval"].update(retrieval_values)
        stages["retrieval"]["network_calls"] = transport_calls
    if dependencies.document_scorer is not None:
        stages["passage_scores"].update(
            _required_local_accounting(
                dependencies.document_scorer,
                attribute="stats",
                label="passage scorer",
            )
        )
    if dependencies.candidate_scorer is not None:
        stages["sentence_scores"].update(
            _required_local_accounting(
                dependencies.candidate_scorer,
                attribute="accounting",
                label="sentence scorer",
            )
        )
    if dependencies.similarity is not None:
        stages["similarity"].update(
            _required_local_accounting(
                dependencies.similarity,
                attribute="accounting",
                label="similarity provider",
            )
        )
    return stages


def _cache_accounting_delta(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for stage_name in CACHE_OPERATION_STAGE_NAMES:
        result[stage_name] = {}
        for counter_name in _CACHE_OPERATION_COUNTER_NAMES:
            delta = after[stage_name][counter_name] - before[stage_name][counter_name]
            if delta < 0:
                raise ValueError("cache operation accounting moved backwards")
            result[stage_name][counter_name] = delta
    return result


def _production_dependencies(*, load_environment: bool = True) -> _RuntimeDependencies:
    repo = Path(__file__).resolve().parents[2]
    if load_environment:
        load_repo_env(repo)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _RuntimeDependencies(
        code_commit=code_commit,
        document_scorer=None,
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
    )


def _topic_completion_for_dispatch(
    config: FacetPilotConfig,
    topic: Topic,
) -> tuple[str, str]:
    """Read the sealed topic completion pair used by the dispatch receipt."""
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        if records.run_id != config.run_id:
            raise ValueError("topic records run identity differs from its config")
        snapshot = records.topic_snapshot()
    if snapshot.status is None and snapshot.stopping_reason is None:
        raise ValueError("topic completion state is not sealed")
    if snapshot.status is None or snapshot.stopping_reason is None:
        raise ValueError("topic completion state is only partially sealed")
    return snapshot.status, snapshot.stopping_reason


def _validated_existing_topic_job_receipt(
    config: FacetPilotConfig,
    topic: Topic,
    *,
    expected_config_sha256: str,
    expected_retriever_identity: Mapping[str, object],
    planning_backend: Any | None,
    operation_mode: str = "online",
    allow_missing_operation_receipt: bool = False,
) -> TopicJobReceipt | None:
    """Recover a projection-only crash into the run-bound dispatch protocol."""
    try:
        projection = read_topic_projection_receipt(config, topic)
    except ValueError as exc:
        if (
            str(exc) == "expanded canonical checkpoint is missing"
            and not (
                config.output_dir / topic.id / "canonical" / "complete.json"
            ).is_file()
        ):
            return None
        raise
    source_seals = dict(projection.source_seals)
    if source_seals.get("config_sha256") != expected_config_sha256:
        raise ValueError("projection config identity changed")
    producer_sha256 = _decomposition_producer_sha256(
        topic,
        config.output_dir,
        planning_backend,
    )
    if source_seals.get("decomposition_producer_sha256") != producer_sha256:
        raise ValueError("projection decomposition producer identity changed")
    validated = validate_retrieval_topic_checkpoints(
        config,
        (topic,),
        expected_retriever_identity=expected_retriever_identity,
        expected_decomposition_producer_sha256={topic.id: producer_sha256},
    )
    if len(validated) != 1 or validated[0] != projection:
        raise ValueError("topic projection validation returned a different receipt")
    status, stopping_reason = _topic_completion_for_dispatch(config, topic)
    operation_path = config.output_dir / topic.id / "cache-operation-receipt.json"
    if not operation_path.exists() and allow_missing_operation_receipt:
        return None
    _read_topic_cache_operation_receipt(
        config=config,
        topic=topic,
        config_sha256=expected_config_sha256,
        projection_manifest_sha256=projection.manifest_sha256,
        mode=operation_mode,
    )
    return TopicJobReceipt(
        topic_id=topic.id,
        projection_manifest_sha256=projection.manifest_sha256,
        status=status,
        stopping_reason=stopping_reason,
    )


def _run_production_topic_job(job: TopicJob) -> TopicJobReceipt:
    """Construct all live dependencies inside one topic worker process."""
    if not isinstance(job, TopicJob):
        raise TypeError("job must be a TopicJob")
    config = load_facet_pilot_config(
        job.config_path,
        source_bytes=job.config_bytes,
    )
    if config.run_id != job.run_id:
        raise ValueError("topic job run identity differs from its config")
    expected_root = (config.output_dir / job.topic_id).resolve()
    if expected_root != job.topic_root:
        raise ValueError("topic job root differs from its config")
    official_topics = tuple(
        _runtime_topic(topic) for topic in load_narrative_topics(config.topics_path)
    )
    matches = tuple(topic for topic in official_topics if topic.id == job.topic_id)
    if len(matches) != 1:
        raise ValueError("topic job identity is absent or duplicated in official topics")
    topic = matches[0]
    cache_root = repo_cache_root(config.root_dir)
    assert_no_incomplete_cache_bundle_merge(cache_root)
    dependencies = (
        _production_dependencies(load_environment=False)
        if job.offline_cache_only
        else _production_dependencies()
    )
    retriever = build_pyserini_retriever(
        config.retrieval.cache_dir,
        index=config.retrieval.index,
        hits=config.retrieval.documents_per_query,
        corpus_epoch=config.retrieval.corpus_epoch,
        cache_only=job.offline_cache_only,
    )
    expected_retriever_identity = _retriever_identity(
        retriever,
        retrieval_depth=config.retrieval.documents_per_query,
    )
    if not job.offline_cache_only:
        recovered = _validated_existing_topic_job_receipt(
            config,
            topic,
            expected_config_sha256=job.config_sha256,
            expected_retriever_identity=expected_retriever_identity,
            planning_backend=None,
            operation_mode="online",
            allow_missing_operation_receipt=True,
        )
        if recovered is not None:
            return recovered
    passage_scorer = MixedbreadPassageScorer(
        score_cache_root=config.passage.score_cache_dir,
        device=config.passage.device,
        read_only=job.offline_cache_only,
    )
    candidate_scorer = MixedbreadSentencePairScorer(
        score_cache_root=config.passage.score_cache_dir,
        device=config.passage.device,
        read_only=job.offline_cache_only,
    )
    similarity = LocalMiniLMSimilarity(
        device=config.passage.device,
        cache_root=cache_root,
        cache_only=job.offline_cache_only,
    )
    dependencies = replace(
        dependencies,
        retriever=retriever,
        document_scorer=passage_scorer,
        candidate_scorer=candidate_scorer,
        similarity=similarity,
    )
    if job.offline_cache_only:
        outcome = _run_offline_topic_staged(
            job,
            topic,
            config,
            _topics_sha256(official_topics),
            dependencies,
            config_sha256=job.config_sha256,
            expected_retriever_identity=expected_retriever_identity,
        )
    else:
        outcome = _run_topic(
            topic,
            config,
            _topics_sha256(official_topics),
            dependencies,
            config_sha256=job.config_sha256,
            expected_retriever_identity=expected_retriever_identity,
        )
    if outcome.topic_id != topic.id or outcome.projection_receipt.topic_id != topic.id:
        raise ValueError("topic worker produced a wrong topic")
    status, stopping_reason = _topic_completion_for_dispatch(config, topic)
    return TopicJobReceipt(
        topic_id=topic.id,
        projection_manifest_sha256=outcome.projection_receipt.manifest_sha256,
        status=status,
        stopping_reason=stopping_reason,
    )


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing any destination entry."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("offline topic publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _publish_directory_create_only(source: Path, destination: Path) -> None:
    _rename_directory_noreplace(Path(source), Path(destination))
    descriptor = os.open(
        Path(destination).parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_offline_topic_staged(
    job: TopicJob,
    topic: Topic,
    config: FacetPilotConfig,
    identity: str,
    dependencies: _RuntimeDependencies,
    *,
    config_sha256: str,
    expected_retriever_identity: Mapping[str, object],
) -> _TopicTaskOutcome:
    """Build an offline replay privately and publish its topic tree only on success."""
    if job.topic_root.exists():
        recovered = _validated_existing_topic_job_receipt(
            config,
            topic,
            expected_config_sha256=config_sha256,
            expected_retriever_identity=expected_retriever_identity,
            planning_backend=dependencies.planning_backend,
            operation_mode="offline-cache-only",
            allow_missing_operation_receipt=False,
        )
        if recovered is None:
            raise ValueError("offline published topic did not validate")
        projection = read_topic_projection_receipt(config, topic)
        if projection.manifest_sha256 != recovered.projection_manifest_sha256:
            raise ValueError("offline projection differs from its validated receipt")
        return _TopicTaskOutcome(topic.id, True, projection)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".offline-cache-{topic.id}-", dir=config.root_dir)
    )
    try:
        source_cache_root = repo_cache_root(config.root_dir)
        stage_cache_root = stage_root / "cache"
        stage_cache_root.mkdir()
        stage_document_store_root = stage_cache_root / "documents" / "v1"
        stage_document_store_root.mkdir(parents=True)
        for name in (
            "canonical",
            "planning-cache-v1",
            "retrieval",
            "reranker",
            "similarity-cache-v1",
        ):
            source = source_cache_root / name
            if source.exists():
                (stage_cache_root / name).symlink_to(
                    source,
                    target_is_directory=True,
                )
        staged_config = replace(config, root_dir=stage_root)
        outcome = _run_topic(
            topic,
            staged_config,
            identity,
            dependencies,
            offline_cache_only=True,
            offline_source_document_store_root=(
                source_cache_root / "documents" / "v1"
            ),
            offline_stage_document_store_root=stage_document_store_root,
            config_sha256=config_sha256,
            expected_retriever_identity=expected_retriever_identity,
        )
        staged_topic_root = staged_config.output_dir / topic.id
        if not staged_topic_root.is_dir():
            raise RuntimeError("offline cache replay did not produce its topic tree")
        job.topic_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            _publish_directory_create_only(staged_topic_root, job.topic_root)
        except FileExistsError as exc:
            raise ValueError(
                "offline cache replay topic output appeared during publication"
            ) from exc
        return outcome
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def run_official(
    config: str | Path,
    *,
    topic_ids: Sequence[str] | None = None,
    topic_subset: Path | None = None,
    external: ExternalAdapters | None = None,
    offline_cache_only: bool = False,
) -> RunReceipt:
    """Run selected official topics and export their organizer-compatible receipt."""
    return _run_official(
        config,
        topic_ids=topic_ids,
        topic_subset=topic_subset,
        external=external,
        offline_cache_only=offline_cache_only,
        dependency_factory=_production_dependencies,
    )


def _run_official(
    config: str | Path,
    *,
    topic_ids: Sequence[str] | None,
    topic_subset: Path | None,
    external: ExternalAdapters | None,
    offline_cache_only: bool = False,
    dependency_factory: Callable[[], _RuntimeDependencies],
) -> RunReceipt:
    """Private implementation that delays runtime construction until preflight passes."""
    config_path = Path(config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = _hash(config_bytes)
    loaded_config = load_facet_pilot_config(config_path)
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("retrieval config changed while it was being loaded")
    if not isinstance(offline_cache_only, bool):
        raise TypeError("offline_cache_only must be Boolean")
    cache_root = repo_cache_root(loaded_config.root_dir)
    assert_no_incomplete_cache_bundle_merge(cache_root)
    if topic_ids is None:
        requested_ids: tuple[str, ...] = ()
    else:
        if isinstance(topic_ids, str):
            raise TypeError("topic_ids must be a sequence of topic IDs, not a string")
        requested_ids = tuple(topic_ids)
        if not requested_ids:
            raise ValueError("topic_ids must not be empty; omit it to select all topics")
    if external is None:
        adapters = ExternalAdapters()
    elif isinstance(external, ExternalAdapters):
        adapters = external
    else:
        raise TypeError("external must be an ExternalAdapters instance")

    selected_topics = tuple(
        _runtime_topic(topic)
        for topic in select_configured_topics(
            loaded_config,
            topic_ids=requested_ids,
            subset_csv=topic_subset,
        )
    )
    official_topics = tuple(
        _runtime_topic(topic)
        for topic in load_narrative_topics(loaded_config.topics_path)
    )
    official_topics_sha256 = _topics_sha256(official_topics)

    repo = Path(__file__).resolve().parents[2]
    if _tracked_worktree_is_dirty(repo):
        raise RuntimeError("config-driven runs reject worktrees with tracked changes")

    jobs = tuple(
        TopicJob(
            topic_id=topic.id,
            run_id=loaded_config.run_id,
            config_path=config_path,
            config_bytes=config_bytes,
            config_sha256=config_sha256,
            topic_root=(loaded_config.output_dir / topic.id).resolve(),
            offline_cache_only=offline_cache_only,
        )
        for topic in selected_topics
    )
    production_dispatch = (
        dependency_factory is _production_dependencies
        and adapters.planning_backend is None
        and adapters.retriever is None
        and adapters.canonical_backend_factory is None
    )
    if offline_cache_only and dependency_factory is _production_dependencies:
        dependencies = _production_dependencies(load_environment=False)
    else:
        dependencies = dependency_factory()
    identity_retriever = adapters.retriever
    if selected_topics and identity_retriever is None:
        identity_retriever = build_pyserini_retriever(
            loaded_config.retrieval.cache_dir,
            index=loaded_config.retrieval.index,
            hits=loaded_config.retrieval.documents_per_query,
            corpus_epoch=loaded_config.retrieval.corpus_epoch,
            cache_only=offline_cache_only,
        )
    if identity_retriever is None:
        raise ValueError("selected topics require a configured retriever identity")

    expected_retriever_identity = _retriever_identity(
        identity_retriever,
        retrieval_depth=loaded_config.retrieval.documents_per_query,
    )

    resume_planning_backend = None if production_dispatch else adapters.planning_backend
    resumed: list[str] = []
    for job, topic in zip(jobs, selected_topics, strict=True):
        sealed = read_topic_receipt(job, missing_ok=True)
        if sealed is None:
            continue
        recovered = _validated_existing_topic_job_receipt(
            loaded_config,
            topic,
            expected_config_sha256=job.config_sha256,
            expected_retriever_identity=expected_retriever_identity,
            planning_backend=resume_planning_backend,
            operation_mode=(
                "offline-cache-only" if job.offline_cache_only else "online"
            ),
            allow_missing_operation_receipt=False,
        )
        if recovered != sealed:
            raise ValueError("topic dispatch receipt differs from validated projection")
        resumed.append(topic.id)
    resumed_topic_ids = tuple(resumed)
    pending_topic_ids = {
        topic.id for topic in selected_topics if topic.id not in set(resumed_topic_ids)
    }

    if not production_dispatch:
        inline_scorer = dependencies.document_scorer
        if pending_topic_ids and inline_scorer is None:
            inline_scorer = MixedbreadPassageScorer(
                score_cache_root=loaded_config.passage.score_cache_dir,
                device=loaded_config.passage.device,
                read_only=offline_cache_only,
            )
        inline_candidate_scorer = dependencies.candidate_scorer
        if pending_topic_ids and inline_candidate_scorer is None:
            inline_candidate_scorer = MixedbreadSentencePairScorer(
                score_cache_root=loaded_config.passage.score_cache_dir,
                device=loaded_config.passage.device,
                read_only=offline_cache_only,
            )
        inline_similarity = dependencies.similarity
        if pending_topic_ids and inline_similarity is None:
            inline_similarity = LocalMiniLMSimilarity(
                device=loaded_config.passage.device,
                cache_root=cache_root,
                cache_only=offline_cache_only,
            )
        dependencies = replace(
            dependencies,
            planning_backend=adapters.planning_backend,
            retriever=identity_retriever,
            document_scorer=inline_scorer,
            candidate_scorer=inline_candidate_scorer,
            similarity=inline_similarity,
            canonical_backend_factory=adapters.canonical_backend_factory,
        )

    selected_by_id = {topic.id: topic for topic in selected_topics}

    if production_dispatch:
        topic_worker = _run_production_topic_job
        topic_workers = loaded_config.execution.topic_workers
    else:
        def run_injected_topic_job(job: TopicJob) -> TopicJobReceipt:
            topic = selected_by_id.get(job.topic_id)
            if topic is None:
                raise ValueError("injected topic worker received an unknown topic")
            if not job.offline_cache_only:
                recovered = _validated_existing_topic_job_receipt(
                    loaded_config,
                    topic,
                    expected_config_sha256=job.config_sha256,
                    expected_retriever_identity=expected_retriever_identity,
                    planning_backend=dependencies.planning_backend,
                    operation_mode="online",
                    allow_missing_operation_receipt=True,
                )
                if recovered is not None:
                    return recovered
            if job.offline_cache_only:
                outcome = _run_offline_topic_staged(
                    job,
                    topic,
                    loaded_config,
                    official_topics_sha256,
                    dependencies,
                    config_sha256=job.config_sha256,
                    expected_retriever_identity=expected_retriever_identity,
                )
            else:
                outcome = _run_topic(
                    topic,
                    loaded_config,
                    official_topics_sha256,
                    dependencies,
                    config_sha256=job.config_sha256,
                    expected_retriever_identity=expected_retriever_identity,
                )
            if (
                outcome.topic_id != topic.id
                or outcome.projection_receipt.topic_id != topic.id
            ):
                raise ValueError("topic task projection receipt contains a wrong topic")
            status, stopping_reason = _topic_completion_for_dispatch(
                loaded_config,
                topic,
            )
            return TopicJobReceipt(
                topic_id=topic.id,
                projection_manifest_sha256=(
                    outcome.projection_receipt.manifest_sha256
                ),
                status=status,
                stopping_reason=stopping_reason,
            )

        topic_worker = run_injected_topic_job
        # Injected adapters can contain live clients and model objects. Keep
        # that explicit testing seam inline rather than attempting to pickle it.
        topic_workers = 1

    dispatched_receipts = dispatch_topics(
        jobs,
        topic_worker,
        max_workers=topic_workers,
    )
    final_planning_backend = None if production_dispatch else dependencies.planning_backend
    decomposition_producer_sha256s = {
        topic.id: _decomposition_producer_sha256(
            topic,
            loaded_config.output_dir,
            final_planning_backend,
        )
        for topic in selected_topics
    }
    ordered_receipts = validate_retrieval_topic_checkpoints(
        loaded_config,
        selected_topics,
        expected_retriever_identity=expected_retriever_identity,
        expected_decomposition_producer_sha256=decomposition_producer_sha256s,
    )
    if len(dispatched_receipts) != len(ordered_receipts):
        raise RuntimeError("all selected topic dispatch receipts are required before export")
    for topic, dispatch_receipt, projection_receipt in zip(
        selected_topics,
        dispatched_receipts,
        ordered_receipts,
        strict=True,
    ):
        if (
            dispatch_receipt.topic_id != topic.id
            or projection_receipt.topic_id != topic.id
            or projection_receipt.manifest_sha256
            != dispatch_receipt.projection_manifest_sha256
        ):
            raise ValueError("topic dispatch receipt differs from its projection seal")
    export_retrieval_run(
        loaded_config,
        selected_topics,
        ordered_receipts,
        code_commit=dependencies.code_commit,
    )
    retrieval_export = read_retrieval_export_receipt(loaded_config, selected_topics)
    operation_mode = "offline-cache-only" if offline_cache_only else "online"
    operation_receipts = tuple(
        _read_topic_cache_operation_receipt(
            config=loaded_config,
            topic=topic,
            config_sha256=config_sha256,
            projection_manifest_sha256=projection_receipt.manifest_sha256,
            mode=operation_mode,
        )
        for topic, projection_receipt in zip(
            selected_topics,
            ordered_receipts,
            strict=True,
        )
    )
    assert_no_incomplete_cache_bundle_merge(cache_root)
    _publish_cache_operation_manifest(
        config=loaded_config,
        topics=selected_topics,
        config_sha256=config_sha256,
        mode=operation_mode,
        receipts=operation_receipts,
        export_manifest_path=retrieval_export.manifest,
    )
    return RunReceipt(
        experiment_id=loaded_config.experiment.id,
        selected_topic_ids=tuple(topic.id for topic in selected_topics),
        resumed_topic_ids=resumed_topic_ids,
        retrieval_export=retrieval_export,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the official facet retrieval export.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--offline-cache-only", action="store_true")
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--topic", action="append", dest="topic_ids")
    selectors.add_argument("--topic-subset", type=Path)
    args = parser.parse_args(argv)
    receipt = run_official(
        args.config,
        topic_ids=None if args.topic_ids is None else tuple(args.topic_ids),
        topic_subset=args.topic_subset,
        offline_cache_only=args.offline_cache_only,
    )
    print(f"output={receipt.retrieval_export.manifest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

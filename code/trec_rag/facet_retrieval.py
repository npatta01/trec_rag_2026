"""Coverage-first retrieval and Mixedbread scoring for the facet pilot.

This module deliberately stops before evidence extraction and evaluation.  It
uses only a topic's official narrative and validated query variants; organizer
material and qrels are not inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Protocol

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.facet_extraction import (
    GeneratedQueryPlan,
    Subnarrative,
    render_facet_queries,
)
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RankedCandidate, RetrievedCandidate
from trec_rag.ranking import _weighted, coverage_aware_long_doc_rank
from trec_rag.rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SCORE_REPRESENTATION,
    GlobalScoreCache,
    ScoreCacheContext,
    _append_jsonl,
    _choose_device,
    _load_cross_encoder,
    _read_document_scores,
    _read_window_scores,
    _score_document_rows,
    _score_window_rows,
    _validate_model_dtype,
)
from trec_rag.retrievers import PyseriniRemoteRetriever
from trec_rag.topics import Topic


RETRIEVAL_DEPTH = 1_000
RERANK_DEPTH = 100
SELECTION_DEPTH = 100
MIXEDBREAD_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
MIXEDBREAD_REVISION = DEFAULT_MODEL_REVISION
BACKEND_VERSION = DEFAULT_BACKEND_VERSION
SCORE_REPRESENTATION = DEFAULT_SCORE_REPRESENTATION
INFERENCE_DTYPE = DEFAULT_INFERENCE_DTYPE
INPUT_POLICY = "trec_rag_raw_v2"
DOCUMENT_MAX_LENGTH = 32_768
DOCUMENT_PAIR_BUFFER_TOKENS = 512
WINDOW_MAX_LENGTH = 1_024
CHUNK_MAX_CHARACTERS = 3_500
CHUNK_OVERLAP_CHARACTERS = 350
TOP_WINDOW_WEIGHTS = (0.55, 0.25, 0.13, 0.07)
LONG_DOCUMENT_WEIGHT = 0.5
STRONGEST_PASSAGE_WEIGHT = 0.5
SPAN_SUPPORT_WEIGHT = 0.25
RELATIVE_SPAN_DELTA = 1.0
SPAN_SUPPORT_CAP = 6
MIN_NEW_SPAN_CHARACTERS = 800
_BACKEND = "sentence-transformers-cross-encoder"
_DOCUMENT_SCORE_KIND = "doc_max_32768_buf512"


__all__ = [
    "BACKEND_VERSION",
    "CHUNK_MAX_CHARACTERS",
    "CHUNK_OVERLAP_CHARACTERS",
    "CoverageSelection",
    "DOCUMENT_MAX_LENGTH",
    "DOCUMENT_PAIR_BUFFER_TOKENS",
    "FacetRetrievalResult",
    "FacetRetrievalLane",
    "INFERENCE_DTYPE",
    "LaneDocumentScore",
    "LaneRetriever",
    "LaneRanking",
    "LaneScorer",
    "LaneSelectionStatus",
    "MIXEDBREAD_MODEL",
    "MIXEDBREAD_REVISION",
    "MixedbreadCoverageScorer",
    "PassageScore",
    "RERANK_DEPTH",
    "RETRIEVAL_DEPTH",
    "RetrievalAuditCandidate",
    "SCORE_REPRESENTATION",
    "SELECTION_DEPTH",
    "SelectedDocument",
    "SelectionTraceEvent",
    "SubnarrativeDocumentScore",
    "UnionPoolDocument",
    "WINDOW_MAX_LENGTH",
    "build_pyserini_retriever",
    "build_retrieval_lanes",
    "round_robin_select",
    "run_facet_retrieval",
    "score_selected_documents",
]


class LaneRetriever(Protocol):
    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]: ...


class LaneScorer(Protocol):
    def score_lane(
        self,
        topic: Topic,
        lane: "FacetRetrievalLane",
        candidates: list[RetrievedCandidate],
    ) -> tuple["LaneDocumentScore", ...]: ...


@dataclass(frozen=True)
class FacetRetrievalLane:
    """One BM25 retrieval identity paired with one semantic scoring identity."""

    retrieval_query: QueryVariant
    scoring_query: QueryVariant
    subnarrative_id: str | None
    bm25_query_sha256: str = field(init=False)
    semantic_query_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.retrieval_query.topic_id != self.scoring_query.topic_id
            or self.retrieval_query.variant_name != self.scoring_query.variant_name
        ):
            raise ValueError("retrieval and scoring query lane identities must match")
        object.__setattr__(
            self,
            "bm25_query_sha256",
            sha256(self.retrieval_query.query_text.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(
            self,
            "semantic_query_sha256",
            sha256(self.scoring_query.query_text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class PassageScore:
    chunk_index: int
    start_char: int
    end_char: int
    raw_logit: float
    weighted_rank: int


@dataclass(frozen=True)
class LaneDocumentScore:
    """One document's inspectable score inside exactly one query lane."""

    topic_id: str
    lane_name: str
    bm25_query: str
    bm25_query_sha256: str
    semantic_query: str
    semantic_query_sha256: str
    docid: str
    text: str
    bm25_rank: int
    bm25_score: float
    aggregate_rank: int
    aggregate_score: float
    long_document_raw_logit: float
    weighted_passage_raw_logit: float
    within_document_span_support: int
    winning_passages: tuple[PassageScore, ...]
    score_representation: str = SCORE_REPRESENTATION


@dataclass(frozen=True)
class RetrievalAuditCandidate:
    """Text-free retrieval receipt retained for every accepted BM25 rank."""

    docid: str
    bm25_rank: int
    bm25_score: float
    text_sha256: str


@dataclass(frozen=True)
class LaneRanking:
    lane: FacetRetrievalLane
    retrieval_returned_count: int
    retrieval_retained_count: int
    retrieval_audit_candidates: tuple[RetrievalAuditCandidate, ...]
    documents: tuple[LaneDocumentScore, ...]
    retrieval_requested_depth: int = RETRIEVAL_DEPTH
    rerank_depth: int = RERANK_DEPTH
    retrieval_audit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        audit_bytes = json.dumps(
            {
                "topic_id": self.lane.retrieval_query.topic_id,
                "lane_name": self.lane.retrieval_query.variant_name,
                "bm25_query_sha256": self.lane.bm25_query_sha256,
                "semantic_query_sha256": self.lane.semantic_query_sha256,
                "retrieval_requested_depth": self.retrieval_requested_depth,
                "retrieval_retained_count": self.retrieval_retained_count,
                "candidates": [
                    {
                        "docid": row.docid,
                        "bm25_rank": row.bm25_rank,
                        "bm25_score": row.bm25_score,
                        "text_sha256": row.text_sha256,
                    }
                    for row in self.retrieval_audit_candidates
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "retrieval_audit_sha256", sha256(audit_bytes).hexdigest())

    @property
    def retrieval_exhausted(self) -> bool:
        return self.retrieval_returned_count < self.retrieval_requested_depth

    @property
    def selection_eligible_count(self) -> int:
        return len(self.documents)

    @property
    def audit_only_candidates(self) -> tuple[RetrievalAuditCandidate, ...]:
        return tuple(
            row for row in self.retrieval_audit_candidates if row.bm25_rank > self.rerank_depth
        )

    @property
    def audit_only_count(self) -> int:
        return len(self.audit_only_candidates)


@dataclass(frozen=True)
class LaneSelectionStatus:
    lane_name: str
    available_count: int
    consumed_count: int
    selected_count: int
    duplicate_skips: int
    exhausted: bool


@dataclass(frozen=True)
class SelectionTraceEvent:
    slot: int
    lane_name: str
    docid: str | None
    lane_rank: int | None
    action: str
    lane_exhausted: bool


@dataclass(frozen=True)
class SelectedDocument:
    topic_id: str
    docid: str
    text: str
    selection_rank: int
    selected_from_lane: str
    selected_from_lane_rank: int
    lane_scores: tuple[LaneDocumentScore, ...]


@dataclass(frozen=True)
class CoverageSelection:
    requested_count: int
    documents: tuple[SelectedDocument, ...]
    lane_statuses: tuple[LaneSelectionStatus, ...]
    trace: tuple[SelectionTraceEvent, ...]
    all_lanes_exhausted: bool

    @property
    def complete(self) -> bool:
        return len(self.documents) == self.requested_count


@dataclass(frozen=True)
class UnionPoolDocument:
    topic_id: str
    docid: str
    text: str
    first_seen_lane: str
    lane_scores: tuple[LaneDocumentScore, ...]


@dataclass(frozen=True)
class FacetRetrievalResult:
    topic_id: str
    lanes: tuple[LaneRanking, ...]
    selection: CoverageSelection
    original_only_control: tuple[LaneDocumentScore, ...]
    union_pool: tuple[UnionPoolDocument, ...]
    rerank_depth: int
    selection_k: int


@dataclass(frozen=True)
class SubnarrativeDocumentScore:
    """Downstream-only semantic score; it never changes document selection."""

    topic_id: str
    docid: str
    selection_rank: int
    subnarrative_id: str
    semantic_query: str
    semantic_query_sha256: str
    bm25_queries: tuple[str, ...]
    bm25_query_sha256s: tuple[str, ...]
    score: LaneDocumentScore
    downstream_only: bool = True


def build_retrieval_lanes(
    topic: Topic,
    queries: Sequence[QueryVariant],
    subnarratives: Sequence[Subnarrative],
) -> tuple[FacetRetrievalLane, ...]:
    """Bind the original and one full-text lane per generated subnarrative."""
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    rows = tuple(queries)
    for query in rows:
        if not isinstance(query, QueryVariant) or query.topic_id != topic.id:
            raise ValueError("all retrieval queries must belong to the topic")
    originals = [
        query
        for query in rows
        if query.variant_name == "original" and query.source_type == "original_topic"
    ]
    if len(originals) != 1 or originals[0].query_text != topic.narrative:
        raise ValueError("retrieval requires one exact official-narrative original lane")
    allowed = {"original_topic", "generated_subnarrative_bm25"}
    if any(query.source_type not in allowed for query in rows):
        raise ValueError("unsupported query source type in facet retrieval plan")
    semantic_rows = tuple(subnarratives)
    if any(
        not isinstance(row, Subnarrative) or row.topic_id != topic.id
        for row in semantic_rows
    ):
        raise ValueError("all subnarratives must belong to the topic")
    if len({row.subnarrative_id for row in semantic_rows}) != len(semantic_rows):
        raise ValueError("subnarrative IDs must be unique")
    if semantic_rows:
        admitted = render_facet_queries(
            topic,
            GeneratedQueryPlan(topic.id, semantic_rows),
        )
        if (
            admitted.used_fallback
            or admitted.subnarratives != semantic_rows
            or admitted.queries != rows
        ):
            raise ValueError("retrieval requires the complete ordered canonical query plan")
    elif rows != (
        QueryVariant(topic.id, "original", topic.narrative, "original_topic"),
    ):
        raise ValueError("retrieval requires the complete ordered canonical query plan")

    bound = [
        FacetRetrievalLane(
            retrieval_query=originals[0],
            scoring_query=QueryVariant(
                topic.id,
                "original",
                topic.narrative,
                "semantic_original",
            ),
            subnarrative_id=None,
        )
    ]
    for subnarrative in semantic_rows:
        lane_name = f"facet:{subnarrative.subnarrative_id}:text"
        bound.append(
            FacetRetrievalLane(
                retrieval_query=QueryVariant(
                    topic.id,
                    lane_name,
                    subnarrative.text,
                    "generated_subnarrative",
                ),
                scoring_query=QueryVariant(
                    topic.id,
                    lane_name,
                    subnarrative.text,
                    "generated_subnarrative",
                ),
                subnarrative_id=subnarrative.subnarrative_id,
            )
        )
    return tuple(bound)


def build_pyserini_retriever(
    cache_dir: Path,
    *,
    index: str = "climbmix-400b",
    hits: int = RETRIEVAL_DEPTH,
    client: Any | None = None,
    continuation_ticket: str | None = None,
) -> PyseriniRemoteRetriever:
    """Build the real identity-bound ClimbMix BM25 adapter."""
    if not isinstance(index, str) or not index.strip():
        raise ValueError("retrieval index must be non-empty text")
    _validate_depth(hits, "retrieval hits")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original", "generated_subnarrative"),
        hits=hits,
        index=index.strip(),
        cache=True,
    )
    return PyseriniRemoteRetriever(
        config,
        cache_dir=Path(cache_dir),
        client=client,
        continuation_ticket=continuation_ticket,
    )


def round_robin_select(
    lanes: Sequence[LaneRanking],
    *,
    limit: int = SELECTION_DEPTH,
) -> CoverageSelection:
    """Select in frozen lane order, advancing past duplicate document IDs."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("selection limit must be a positive integer")
    ordered = tuple(lanes)
    if not ordered:
        raise ValueError("selection requires at least one lane")
    names = [lane.lane.retrieval_query.variant_name for lane in ordered]
    if len(set(names)) != len(names):
        raise ValueError("selection lane names must be unique")
    memberships = _lane_memberships(ordered)
    positions = [0 for _lane in ordered]
    selected_per_lane = [0 for _lane in ordered]
    duplicate_skips = [0 for _lane in ordered]
    exhaustion_logged = [False for _lane in ordered]
    selected_ids: set[str] = set()
    selected: list[SelectedDocument] = []
    trace: list[SelectionTraceEvent] = []

    while len(selected) < limit:
        round_progress = False
        for lane_index, lane in enumerate(ordered):
            if len(selected) >= limit:
                break
            slot = len(selected) + 1
            supplied = False
            while positions[lane_index] < len(lane.documents):
                row = lane.documents[positions[lane_index]]
                positions[lane_index] += 1
                exhausted_now = positions[lane_index] == len(lane.documents)
                if row.docid in selected_ids:
                    duplicate_skips[lane_index] += 1
                    trace.append(
                        SelectionTraceEvent(
                            slot,
                            lane.lane.retrieval_query.variant_name,
                            row.docid,
                            row.aggregate_rank,
                            "duplicate_skip",
                            exhausted_now,
                        )
                    )
                    continue
                selected_ids.add(row.docid)
                lane_scores = memberships[row.docid]
                selected.append(
                    SelectedDocument(
                        topic_id=row.topic_id,
                        docid=row.docid,
                        text=_one_document_text(lane_scores),
                        selection_rank=slot,
                        selected_from_lane=lane.lane.retrieval_query.variant_name,
                        selected_from_lane_rank=row.aggregate_rank,
                        lane_scores=lane_scores,
                    )
                )
                selected_per_lane[lane_index] += 1
                trace.append(
                    SelectionTraceEvent(
                        slot,
                        lane.lane.retrieval_query.variant_name,
                        row.docid,
                        row.aggregate_rank,
                        "selected",
                        exhausted_now,
                    )
                )
                supplied = True
                round_progress = True
                break
            if not supplied and positions[lane_index] >= len(lane.documents):
                if not exhaustion_logged[lane_index]:
                    trace.append(
                        SelectionTraceEvent(
                            slot,
                            lane.lane.retrieval_query.variant_name,
                            None,
                            None,
                            "exhausted",
                            True,
                        )
                    )
                    exhaustion_logged[lane_index] = True
        if not round_progress:
            break

    statuses = tuple(
        LaneSelectionStatus(
            lane_name=lane.lane.retrieval_query.variant_name,
            available_count=len(lane.documents),
            consumed_count=positions[index],
            selected_count=selected_per_lane[index],
            duplicate_skips=duplicate_skips[index],
            exhausted=positions[index] >= len(lane.documents),
        )
        for index, lane in enumerate(ordered)
    )
    return CoverageSelection(
        requested_count=limit,
        documents=tuple(selected),
        lane_statuses=statuses,
        trace=tuple(trace),
        all_lanes_exhausted=all(status.exhausted for status in statuses),
    )


def run_facet_retrieval(
    topic: Topic,
    queries: Sequence[QueryVariant],
    retriever: LaneRetriever,
    scorer: LaneScorer,
    *,
    subnarratives: Sequence[Subnarrative],
    retrieval_depth: int = RETRIEVAL_DEPTH,
    rerank_depth: int = RERANK_DEPTH,
    selection_k: int = SELECTION_DEPTH,
) -> FacetRetrievalResult:
    """Run configured-depth retrieval, top-N scoring, and coverage-first selection."""
    _validate_depth(retrieval_depth, "retrieval_depth")
    _validate_depth(rerank_depth, "rerank_depth", maximum=retrieval_depth)
    _validate_depth(selection_k, "selection_k")
    lane_queries = build_retrieval_lanes(topic, queries, subnarratives)
    retrieved_lanes: list[
        tuple[FacetRetrievalLane, int, list[RetrievedCandidate]]
    ] = []
    source_hash_by_docid: dict[str, str] = {}
    for lane in lane_queries:
        query = lane.retrieval_query
        raw_candidates = retriever.retrieve(query)
        returned_count = len(raw_candidates)
        retained = _canonical_candidates(
            topic,
            query,
            raw_candidates,
            require_contiguous_ranks=True,
        )[:retrieval_depth]
        for candidate in retained:
            text_hash = sha256(candidate.text.encode("utf-8")).hexdigest()
            previous = source_hash_by_docid.setdefault(candidate.docid, text_hash)
            if previous != text_hash:
                raise ValueError(
                    "document source text identity conflicts across retrieval lanes"
                )
        retrieved_lanes.append((lane, returned_count, retained))

    lane_results: list[LaneRanking] = []
    for lane, returned_count, retained in retrieved_lanes:
        eligible = retained[:rerank_depth]
        semantic_eligible = [
            replace(row, query_text=lane.scoring_query.query_text)
            for row in eligible
        ]
        scored = tuple(scorer.score_lane(topic, lane, semantic_eligible))
        scored = _validated_lane_scores(lane, semantic_eligible, scored)
        audit = tuple(
            RetrievalAuditCandidate(
                docid=row.docid,
                bm25_rank=row.rank,
                bm25_score=row.score,
                text_sha256=sha256(row.text.encode("utf-8")).hexdigest(),
            )
            for row in retained
        )
        lane_results.append(
            LaneRanking(
                lane=lane,
                retrieval_returned_count=returned_count,
                retrieval_retained_count=len(retained),
                retrieval_audit_candidates=audit,
                documents=scored,
                retrieval_requested_depth=retrieval_depth,
                rerank_depth=rerank_depth,
            )
        )
    lanes = tuple(lane_results)
    selection = round_robin_select(lanes, limit=selection_k)
    return FacetRetrievalResult(
        topic_id=topic.id,
        lanes=lanes,
        selection=selection,
        original_only_control=lanes[0].documents,
        union_pool=_union_pool(lanes),
        rerank_depth=rerank_depth,
        selection_k=selection_k,
    )


def score_selected_documents(
    topic: Topic,
    selection: CoverageSelection,
    subnarratives: Sequence[Subnarrative],
    scorer: LaneScorer,
) -> tuple[SubnarrativeDocumentScore, ...]:
    """Cross-score selected documents downstream; never feed scores into selection."""
    _validate_selection(topic, selection)
    facets = tuple(subnarratives)
    if any(row.topic_id != topic.id for row in facets):
        raise ValueError("subnarratives must belong to the selected topic")
    if len({row.subnarrative_id for row in facets}) != len(facets):
        raise ValueError("subnarrative IDs must be unique")
    by_facet: dict[str, dict[str, LaneDocumentScore]] = {}
    for facet in facets:
        lane_name = f"subnarrative:{facet.subnarrative_id}"
        lane = FacetRetrievalLane(
            retrieval_query=QueryVariant(
                topic.id,
                lane_name,
                facet.text,
                "selected_subnarrative_pool",
            ),
            scoring_query=QueryVariant(
                topic.id,
                lane_name,
                facet.text,
                "semantic_subnarrative",
            ),
            subnarrative_id=facet.subnarrative_id,
        )
        candidates = [
            RetrievedCandidate(
                topic.id,
                lane_name,
                "selected_document_pool",
                facet.text,
                row.docid,
                row.selection_rank,
                0.0,
                row.text,
            )
            for row in selection.documents
        ]
        scores = _validated_lane_scores(
            lane,
            candidates,
            tuple(scorer.score_lane(topic, lane, candidates)),
        )
        by_facet[facet.subnarrative_id] = {row.docid: row for row in scores}
    return tuple(
        SubnarrativeDocumentScore(
            topic_id=topic.id,
            docid=document.docid,
            selection_rank=document.selection_rank,
            subnarrative_id=facet.subnarrative_id,
            semantic_query=facet.text,
            semantic_query_sha256=facet.semantic_query_sha256,
            bm25_queries=facet.bm25_queries,
            bm25_query_sha256s=facet.bm25_query_sha256s,
            score=by_facet[facet.subnarrative_id][document.docid],
        )
        for document in selection.documents
        for facet in facets
    )


class _LazyPinnedModel:
    def __init__(
        self,
        *,
        max_length: int,
        device: str,
        loader: Callable[..., Any],
    ) -> None:
        self._max_length = max_length
        self._device = device
        self._loader = loader
        self._model: Any | None = None

    def predict(self, pairs: Any, **kwargs: Any) -> Any:
        if self._model is None:
            self._model = self._loader(
                MIXEDBREAD_MODEL,
                revision=MIXEDBREAD_REVISION,
                max_length=self._max_length,
                device=_choose_device(self._device),
            )
            _validate_model_dtype(self._model, INFERENCE_DTYPE)
        return self._model.predict(pairs, **kwargs)


class MixedbreadCoverageScorer:
    """Real pinned Mixedbread adapter backed by content-addressed raw logits."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        score_cache_root: Path,
        device: str = "auto",
        model_loader: Callable[..., Any] = _load_cross_encoder,
        document_batch_size: int = 1,
        window_batch_size: int = 8,
    ) -> None:
        _validate_depth(document_batch_size, "document_batch_size")
        _validate_depth(window_batch_size, "window_batch_size")
        self.artifact_dir = Path(artifact_dir)
        self.document_score_path = self.artifact_dir / "document_scores.jsonl"
        self.window_score_path = self.artifact_dir / "window_scores.jsonl"
        self.document_batch_size = document_batch_size
        self.window_batch_size = window_batch_size
        self.chunker = SemanticTextChunker(
            ChunkingConfig(
                max_characters=CHUNK_MAX_CHARACTERS,
                overlap_characters=CHUNK_OVERLAP_CHARACTERS,
            )
        )
        context = {
            "backend": _BACKEND,
            "model": MIXEDBREAD_MODEL,
            "model_revision": MIXEDBREAD_REVISION,
            "backend_version": BACKEND_VERSION,
            "score_representation": SCORE_REPRESENTATION,
            "inference_dtype": INFERENCE_DTYPE,
            "input_policy": INPUT_POLICY,
        }
        self.document_cache = GlobalScoreCache(
            Path(score_cache_root),
            ScoreCacheContext(
                **context,
                max_length=DOCUMENT_MAX_LENGTH - DOCUMENT_PAIR_BUFFER_TOKENS,
                score_kind=_DOCUMENT_SCORE_KIND,
                requested_max_length=DOCUMENT_MAX_LENGTH,
                pair_buffer_tokens=DOCUMENT_PAIR_BUFFER_TOKENS,
            ),
        )
        self.window_cache = GlobalScoreCache(
            Path(score_cache_root),
            ScoreCacheContext(
                **context,
                max_length=WINDOW_MAX_LENGTH,
                score_kind="window",
                requested_max_length=WINDOW_MAX_LENGTH,
                chunk_max_characters=CHUNK_MAX_CHARACTERS,
                chunk_overlap_characters=CHUNK_OVERLAP_CHARACTERS,
            ),
        )
        self._document_model = _LazyPinnedModel(
            max_length=DOCUMENT_MAX_LENGTH - DOCUMENT_PAIR_BUFFER_TOKENS,
            device=device,
            loader=model_loader,
        )
        self._window_model = _LazyPinnedModel(
            max_length=WINDOW_MAX_LENGTH,
            device=device,
            loader=model_loader,
        )

    @property
    def identity(self) -> dict[str, object]:
        return {
            "model": MIXEDBREAD_MODEL,
            "model_revision": MIXEDBREAD_REVISION,
            "backend_version": BACKEND_VERSION,
            "score_representation": SCORE_REPRESENTATION,
            "inference_dtype": INFERENCE_DTYPE,
            "document_max_length": DOCUMENT_MAX_LENGTH,
            "document_pair_buffer_tokens": DOCUMENT_PAIR_BUFFER_TOKENS,
            "window_max_length": WINDOW_MAX_LENGTH,
            "chunk_max_characters": CHUNK_MAX_CHARACTERS,
            "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
        }

    def score_lane(
        self,
        topic: Topic,
        lane: FacetRetrievalLane,
        candidates: list[RetrievedCandidate],
    ) -> tuple[LaneDocumentScore, ...]:
        if not candidates:
            return ()
        candidates = _canonical_candidates(topic, lane.scoring_query, candidates)
        existing_documents = _read_document_scores(
            self.document_score_path,
            context=self.document_cache.context,
        )
        document_rows = _score_document_rows(
            model=self._document_model,
            topic=topic,
            candidates=candidates,
            existing_scores=existing_documents,
            batch_size=self.document_batch_size,
            score_cache=self.document_cache,
            score_kind=_DOCUMENT_SCORE_KIND,
        )
        _append_jsonl(self.document_score_path, document_rows)
        existing_windows = _read_window_scores(
            self.window_score_path,
            context=self.window_cache.context,
        )
        window_rows = _score_window_rows(
            model=self._window_model,
            topic=topic,
            candidates=candidates,
            existing_scores=existing_windows,
            batch_size=self.window_batch_size,
            chunker=self.chunker,
            score_cache=self.window_cache,
        )
        _append_jsonl(self.window_score_path, window_rows)
        ranked = coverage_aware_long_doc_rank(
            candidates,
            document_score_path=self.document_score_path,
            window_score_path=self.window_score_path,
            expected_document_score_metadata=self.document_cache.context.artifact_metadata,
            expected_window_score_metadata=self.window_cache.context.artifact_metadata,
            long_document_weight=LONG_DOCUMENT_WEIGHT,
            strongest_passage_weight=STRONGEST_PASSAGE_WEIGHT,
            coverage_bonus_weight=SPAN_SUPPORT_WEIGHT,
            relative_span_delta=RELATIVE_SPAN_DELTA,
            support_cap=SPAN_SUPPORT_CAP,
            min_new_chars=MIN_NEW_SPAN_CHARACTERS,
            top_window_weights=TOP_WINDOW_WEIGHTS,
        )
        all_windows = _read_window_scores(
            self.window_score_path,
            context=self.window_cache.context,
        )
        return tuple(
            self._to_lane_score(lane, candidates, row, all_windows)
            for row in ranked
        )

    @staticmethod
    def _to_lane_score(
        lane: FacetRetrievalLane,
        candidates: Sequence[RetrievedCandidate],
        ranked: RankedCandidate,
        all_windows: dict[tuple[str, str, int], list[dict[str, Any]]],
    ) -> LaneDocumentScore:
        source = min(
            (row for row in candidates if row.docid == ranked.docid),
            key=lambda row: (row.rank, -row.score),
        )
        aggregate = next(
            row
            for row in reversed(ranked.provenance)
            if row.get("ranker") == "coverage_aware_long_doc_aggregate"
        )
        query_hash = lane.semantic_query_sha256
        document_hash = sha256(source.text.encode("utf-8")).hexdigest()
        windows = [
            row
            for (topic_id, docid, _chunk_index), rows in all_windows.items()
            if topic_id == lane.retrieval_query.topic_id and docid == source.docid
            for row in rows
            if row.get("query_sha256") == query_hash
            and row.get("document_text_sha256") == document_hash
        ]
        windows.sort(key=lambda row: (-float(row["score"]), int(row["chunk_index"])))
        winning = tuple(
            PassageScore(
                chunk_index=int(row["chunk_index"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                raw_logit=float(row["score"]),
                weighted_rank=index,
            )
            for index, row in enumerate(windows[: len(TOP_WINDOW_WEIGHTS)], start=1)
        )
        return LaneDocumentScore(
            topic_id=ranked.topic_id,
            lane_name=lane.retrieval_query.variant_name,
            bm25_query=lane.retrieval_query.query_text,
            bm25_query_sha256=lane.bm25_query_sha256,
            semantic_query=lane.scoring_query.query_text,
            semantic_query_sha256=lane.semantic_query_sha256,
            docid=ranked.docid,
            text=ranked.text,
            bm25_rank=source.rank,
            bm25_score=source.score,
            aggregate_rank=ranked.rank,
            aggregate_score=ranked.score,
            long_document_raw_logit=float(aggregate["long_document_relevance"]),
            weighted_passage_raw_logit=float(aggregate["strongest_passage_relevance"]),
            within_document_span_support=int(aggregate["bounded_coverage_support"]),
            winning_passages=winning,
        )


def _validate_depth(value: int, label: str, *, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")


def _validate_selection(topic: Topic, selection: CoverageSelection) -> None:
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    if not isinstance(selection, CoverageSelection):
        raise TypeError("selection must be a CoverageSelection")
    requested = selection.requested_count
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("selection requested count must be a positive integer")

    documents = selection.documents
    if (
        not isinstance(documents, tuple)
        or any(not isinstance(row, SelectedDocument) for row in documents)
        or len(documents) > requested
    ):
        raise ValueError("selection document count is invalid")
    selection_ranks = [row.selection_rank for row in documents]
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
        for rank in selection_ranks
    ):
        raise ValueError("selected document ranks must be positive integers")
    if selection_ranks != list(range(1, len(documents) + 1)):
        raise ValueError("selected document ranks must be unique and contiguous")
    docids = [row.docid for row in documents]
    if (
        any(not isinstance(docid, str) or not docid for docid in docids)
        or len(set(docids)) != len(docids)
    ):
        raise ValueError("selected document IDs must be nonempty and unique")

    selected_count_by_lane: dict[str, int] = {}
    for document in documents:
        if document.topic_id != topic.id:
            raise ValueError("selected document belongs to a different topic")
        if not isinstance(document.text, str) or not document.text.strip():
            raise ValueError("selected document must carry nonempty source text")
        if (
            isinstance(document.selected_from_lane_rank, bool)
            or not isinstance(document.selected_from_lane_rank, int)
            or document.selected_from_lane_rank <= 0
        ):
            raise ValueError("selected document lane rank must be a positive integer")
        lane_scores = document.lane_scores
        if (
            not isinstance(lane_scores, tuple)
            or not lane_scores
            or any(not isinstance(score, LaneDocumentScore) for score in lane_scores)
        ):
            raise ValueError("selected document must retain its lane scores")
        lane_names = [score.lane_name for score in lane_scores]
        if len(set(lane_names)) != len(lane_names):
            raise ValueError("selected document lane scores must have unique lanes")
        if any(
            score.topic_id != topic.id
            or score.docid != document.docid
            or score.text != document.text
            for score in lane_scores
        ):
            raise ValueError("selected document source identity differs from its lane scores")
        selected_lane_scores = [
            score
            for score in lane_scores
            if score.lane_name == document.selected_from_lane
        ]
        if (
            len(selected_lane_scores) != 1
            or selected_lane_scores[0].aggregate_rank
            != document.selected_from_lane_rank
        ):
            raise ValueError("selected document origin differs from its lane score")
        selected_count_by_lane[document.selected_from_lane] = (
            selected_count_by_lane.get(document.selected_from_lane, 0) + 1
        )

    statuses = selection.lane_statuses
    if (
        not isinstance(statuses, tuple)
        or not statuses
        or any(not isinstance(status, LaneSelectionStatus) for status in statuses)
    ):
        raise ValueError("selection must retain lane statuses")
    status_names = [status.lane_name for status in statuses]
    if (
        any(not isinstance(name, str) or not name for name in status_names)
        or len(set(status_names)) != len(status_names)
    ):
        raise ValueError("selection lane status names must be nonempty and unique")
    for status in statuses:
        counters = (
            status.available_count,
            status.consumed_count,
            status.selected_count,
            status.duplicate_skips,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            raise ValueError("selection lane counters must be nonnegative integers")
        if (
            status.consumed_count > status.available_count
            or status.selected_count + status.duplicate_skips
            != status.consumed_count
            or status.selected_count
            != selected_count_by_lane.get(status.lane_name, 0)
            or not isinstance(status.exhausted, bool)
            or status.exhausted
            != (status.consumed_count == status.available_count)
        ):
            raise ValueError("selection lane status is inconsistent")
    if set(selected_count_by_lane) - set(status_names):
        raise ValueError("selected document refers to an unknown selection lane")
    if not isinstance(selection.all_lanes_exhausted, bool) or (
        selection.all_lanes_exhausted
        != all(status.exhausted for status in statuses)
    ):
        raise ValueError("selection exhaustion state is inconsistent")
    if len(documents) < requested and not selection.all_lanes_exhausted:
        raise ValueError("incomplete selection must have exhausted every lane")


def _canonical_candidates(
    topic: Topic,
    query: QueryVariant,
    candidates: Sequence[RetrievedCandidate],
    *,
    require_contiguous_ranks: bool = False,
) -> list[RetrievedCandidate]:
    rows = list(candidates)
    for row in rows:
        if (
            not isinstance(row, RetrievedCandidate)
            or row.topic_id != topic.id
            or row.variant_name != query.variant_name
            or row.query_text != query.query_text
        ):
            raise ValueError("retrieved candidate identity differs from its query lane")
        if (
            isinstance(row.rank, bool)
            or not isinstance(row.rank, int)
            or row.rank <= 0
        ):
            raise ValueError("retrieved candidate rank must be a positive integer")
        if (
            isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(row.score)
        ):
            raise ValueError("retrieved candidate score must be a finite number")
    ordered = sorted(rows, key=lambda row: (row.rank, -row.score, row.docid))
    ranks = [row.rank for row in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError("retrieved candidate ranks must be unique")
    docids = [row.docid for row in ordered]
    if len(set(docids)) != len(docids):
        raise ValueError("retrieval lane contains a duplicate document ID")
    expected_ranks = list(range(1, len(ordered) + 1))
    if require_contiguous_ranks and ranks != expected_ranks:
        raise ValueError("retrieved candidate ranks must be unique and contiguous")
    return ordered


def _validated_lane_scores(
    lane: FacetRetrievalLane,
    candidates: Sequence[RetrievedCandidate],
    scores: Sequence[LaneDocumentScore],
) -> tuple[LaneDocumentScore, ...]:
    score_rows = tuple(scores)
    if any(
        not isinstance(row, LaneDocumentScore)
        or isinstance(row.aggregate_rank, bool)
        or not isinstance(row.aggregate_rank, int)
        or row.aggregate_rank <= 0
        for row in score_rows
    ):
        raise ValueError("lane aggregate ranks must be positive integers")
    rows = tuple(sorted(score_rows, key=lambda row: row.aggregate_rank))
    expected_docids = {row.docid for row in candidates}
    expected_by_doc = {
        docid: min(
            (candidate for candidate in candidates if candidate.docid == docid),
            key=lambda candidate: (candidate.rank, -candidate.score),
        )
        for docid in expected_docids
    }
    observed_docids = {row.docid for row in rows}
    if observed_docids != expected_docids or len(observed_docids) != len(rows):
        raise ValueError("lane scorer must return every eligible document exactly once")
    if [row.aggregate_rank for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("lane aggregate ranks must be unique and contiguous")
    if any(
        row.topic_id != lane.retrieval_query.topic_id
        or row.lane_name != lane.retrieval_query.variant_name
        or row.bm25_query != lane.retrieval_query.query_text
        or row.bm25_query_sha256 != lane.bm25_query_sha256
        or row.semantic_query != lane.scoring_query.query_text
        or row.semantic_query_sha256 != lane.semantic_query_sha256
        or row.score_representation != SCORE_REPRESENTATION
        or not math.isfinite(row.aggregate_score)
        for row in rows
    ):
        raise ValueError("lane scorer output identity or score is invalid")
    if any(
        row.text != expected_by_doc[row.docid].text
        or row.bm25_rank != expected_by_doc[row.docid].rank
        or row.bm25_score != expected_by_doc[row.docid].score
        for row in rows
    ):
        raise ValueError("lane scorer changed retrieval provenance")
    for row in rows:
        _validate_lane_score_components(row)
    return rows


def _validate_lane_score_components(row: LaneDocumentScore) -> None:
    scalar_scores = (
        row.aggregate_score,
        row.long_document_raw_logit,
        row.weighted_passage_raw_logit,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in scalar_scores
    ):
        raise ValueError("lane score components must be finite numbers")
    support = row.within_document_span_support
    if (
        isinstance(support, bool)
        or not isinstance(support, int)
        or not 0 <= support <= SPAN_SUPPORT_CAP
    ):
        raise ValueError("lane score span support must be an integer within bounds")

    passages = row.winning_passages
    if (
        not isinstance(passages, tuple)
        or not passages
        or len(passages) > len(TOP_WINDOW_WEIGHTS)
        or any(not isinstance(passage, PassageScore) for passage in passages)
    ):
        raise ValueError("lane score must contain a bounded passage tuple")
    chunk_indices: set[int] = set()
    for passage in passages:
        integer_fields = (
            passage.chunk_index,
            passage.start_char,
            passage.end_char,
            passage.weighted_rank,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields
        ):
            raise ValueError("passage indices, offsets, and ranks must be integers")
        if (
            passage.chunk_index < 0
            or passage.chunk_index in chunk_indices
            or passage.start_char < 0
            or passage.end_char <= passage.start_char
            or passage.end_char > len(row.text)
        ):
            raise ValueError("passage identity or offsets are invalid")
        if (
            isinstance(passage.raw_logit, bool)
            or not isinstance(passage.raw_logit, (int, float))
            or not math.isfinite(passage.raw_logit)
        ):
            raise ValueError("passage score must be finite")
        chunk_indices.add(passage.chunk_index)

    if [passage.weighted_rank for passage in passages] != list(
        range(1, len(passages) + 1)
    ):
        raise ValueError("passage weighted ranks must be unique and contiguous")
    if list(passages) != sorted(
        passages,
        key=lambda passage: (-passage.raw_logit, passage.chunk_index),
    ):
        raise ValueError("passages must be ordered by score and chunk index")

    expected_weighted = _weighted(
        [float(passage.raw_logit) for passage in passages],
        TOP_WINDOW_WEIGHTS,
    )
    if not math.isclose(
        row.weighted_passage_raw_logit,
        expected_weighted,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("weighted passage score formula does not match its components")
    expected_aggregate = (
        LONG_DOCUMENT_WEIGHT * row.long_document_raw_logit
        + STRONGEST_PASSAGE_WEIGHT * row.weighted_passage_raw_logit
        + SPAN_SUPPORT_WEIGHT * support
    )
    if not math.isclose(
        row.aggregate_score,
        expected_aggregate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("aggregate score formula does not match its components")


def _lane_memberships(
    lanes: Sequence[LaneRanking],
) -> dict[str, tuple[LaneDocumentScore, ...]]:
    order = {
        lane.lane.retrieval_query.variant_name: index
        for index, lane in enumerate(lanes)
    }
    memberships: dict[str, list[LaneDocumentScore]] = {}
    for lane in lanes:
        for row in lane.documents:
            memberships.setdefault(row.docid, []).append(row)
    return {
        docid: tuple(sorted(rows, key=lambda row: order[row.lane_name]))
        for docid, rows in memberships.items()
    }


def _one_document_text(scores: Sequence[LaneDocumentScore]) -> str:
    texts = {row.text for row in scores}
    if len(texts) != 1:
        raise ValueError("one document has conflicting source text across lanes")
    return texts.pop()


def _union_pool(lanes: Sequence[LaneRanking]) -> tuple[UnionPoolDocument, ...]:
    memberships = _lane_memberships(lanes)
    seen: set[str] = set()
    rows: list[UnionPoolDocument] = []
    for lane in lanes:
        for score in lane.documents:
            if score.docid in seen:
                continue
            seen.add(score.docid)
            lane_scores = memberships[score.docid]
            rows.append(
                UnionPoolDocument(
                    topic_id=score.topic_id,
                    docid=score.docid,
                    text=_one_document_text(lane_scores),
                    first_seen_lane=lane.lane.retrieval_query.variant_name,
                    lane_scores=lane_scores,
                )
            )
    return tuple(rows)

"""The supported public interface for official facet retrieval exports."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from trec_rag.canonical_nuggets import PROMPT_VERSION, run_canonical_stage
from trec_rag.evidence_store import (
    generate_candidate_artifacts,
    materialize_candidate_inputs,
    select_evidence_artifacts,
)
from trec_rag.facet_evidence import SelectionPolicy
from trec_rag.facet_extraction import (
    FacetPlanningResult,
    GeneratedQueryPlan,
    OpenRouterDeepSeekFacetBackend,
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
    MixedbreadCoverageScorer,
    PassageScore,
    RetrievalAuditCandidate,
    _union_pool,
    build_pyserini_retriever,
    build_retrieval_lanes,
    round_robin_select,
    run_facet_retrieval,
    score_selected_documents,
)
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.repo_env import load_repo_env, repo_cache_root
from trec_rag.retrieval_export import (
    RetrievalExportReceipt,
    export_retrieval_run,
    read_retrieval_export_receipt,
    validate_retrieval_topic_checkpoints,
)
from trec_rag.topics import Topic, load_narrative_topics


SCHEMA = "facet_pilot_v2"
SELECTION_SCHEMA = "facet_pilot_selection_v2"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TOPIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


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


@dataclass(frozen=True)
class TopicPhaseOutcome:
    topic_id: str
    phase: str
    manifest_path: Path
    resumed: bool


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
    ):
        raise ValueError("retrieval audit identity or lane set changed")
    decoded: list[ValidatedRetrievalAuditLane] = []
    lane_fields = {
        "lane_name", "subnarrative_id", "bm25_query_sha256",
        "semantic_query_sha256", "returned_count", "retained_count", "candidates",
    }
    candidate_fields = {"docid", "bm25_rank", "bm25_score", "text_sha256"}
    for raw, lane in zip(root["lanes"], expected_lanes, strict=True):
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
        decoded.append(
            ValidatedRetrievalAuditLane(
                lane, requested_depth, raw["returned_count"], tuple(candidates)
            )
        )
    return tuple(decoded)


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
        eligible = audited.candidates[:rerank_depth]
        if len(score_rows) != len(eligible):
            raise ValueError("selection lane scores differ from retrieval audit")
        audit_by_doc = {row.docid: row for row in eligible}
        typed_scores: list[LaneDocumentScore] = []
        for expected_rank, row in enumerate(score_rows, start=1):
            audit = audit_by_doc.get(row.get("docid"))
            if (
                type(row.get("aggregate_rank")) is not int
                or row["aggregate_rank"] != expected_rank
                or audit is None
                or row.get("bm25_rank") != audit.bm25_rank
                or row.get("bm25_score") != audit.bm25_score
                or row.get("text_sha256") != audit.text_sha256
                or row.get("bm25_query_sha256") != audited.lane.bm25_query_sha256
                or row.get("semantic_query_sha256") != audited.lane.semantic_query_sha256
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
                rerank_depth=rerank_depth,
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
    retriever: Any | None = None,
    retrieval_depth: int,
) -> TopicPhaseOutcome:
    """Fill the shared retrieval cache and save a text-free retrieval audit."""
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
    )
    identity = _retriever_identity(retriever, retrieval_depth=retrieval_depth)
    root = Path(output_dir) / topic.id
    manifest = root / "retrieval" / "complete.json"
    expected = _expected("retrieve", topic, decomposition, code_commit, identity)
    artifacts = ("decomposition.json", "retrieval/audit.json")
    if _resume(manifest, root, expected, artifacts):
        return TopicPhaseOutcome(topic.id, "retrieve", manifest, True)

    lanes = build_retrieval_lanes(
        topic,
        decomposition.result.queries,
        decomposition.result.subnarratives,
    )
    hashes: dict[str, str] = {}
    audit_lanes = []
    for lane in lanes:
        query = lane.retrieval_query
        rows = _candidates(topic, query, retriever.retrieve(query))
        retained = rows[:retrieval_depth]
        candidates = []
        for row in retained:
            text_hash = _hash(row.text.encode())
            if hashes.setdefault(row.docid, text_hash) != text_hash:
                raise ValueError("document source text conflicts across retrieval lanes")
            candidates.append(
                {
                    "docid": row.docid,
                    "bm25_rank": row.rank,
                    "bm25_score": row.score,
                    "text_sha256": text_hash,
                }
            )
        audit_lanes.append(_audit_lane(lane, len(rows), candidates))
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
    )
    _write_json(root / artifacts[0], decomposition_record)
    _write_json(root / artifacts[1], audit)
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
    retriever: Any | None,
    scorer: Any | None,
    device: str,
    retrieval_depth: int,
    rerank_depth: int,
    selection_k: int,
) -> TopicPhaseOutcome:
    """Rerank and checkpoint selected documents plus subnarrative scores."""
    _inputs(topic, decomposition, code_commit)
    retriever = retriever or build_pyserini_retriever(
        Path(cache_dir),
        hits=retrieval_depth,
    )
    retriever_identity = _retriever_identity(
        retriever,
        retrieval_depth=retrieval_depth,
    )
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
            retriever_identity,
        ),
        ("decomposition.json", "retrieval/audit.json"),
        required=True,
    )
    scorer = scorer or MixedbreadCoverageScorer(
        artifact_dir=Path(output_dir) / "scorer-ledger",
        score_cache_root=Path(score_cache_root),
        device=device,
    )
    scorer_identity = dict(scorer.identity)
    expected = _expected(
        "score",
        topic,
        decomposition,
        code_commit,
        retriever_identity,
    ) | {
        "selection_schema_version": SELECTION_SCHEMA,
        "retrieval_manifest_sha256": _hash(retrieve_manifest.read_bytes()),
        "scorer": scorer_identity,
        "rerank_depth": rerank_depth,
        "selection_k": selection_k,
        "selection_policy": "round_robin_lane_order_no_fusion",
        "score_policy": {
            "long_document_weight": LONG_DOCUMENT_WEIGHT,
            "strongest_passage_weight": STRONGEST_PASSAGE_WEIGHT,
            "span_support_weight": SPAN_SUPPORT_WEIGHT,
            "span_support_cap": SPAN_SUPPORT_CAP,
            "relative_span_delta": RELATIVE_SPAN_DELTA,
            "top_window_weights": list(TOP_WINDOW_WEIGHTS),
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

    result = run_facet_retrieval(
        topic,
        decomposition.result.queries,
        retriever,
        scorer,
        subnarratives=decomposition.result.subnarratives,
        retrieval_depth=retrieval_depth,
        rerank_depth=rerank_depth,
        selection_k=selection_k,
    )
    if _loads(
        (root / "retrieval/audit.json").read_bytes(),
        "retrieval audit",
    ) != _result_audit(topic, decomposition, result):
        raise ValueError("scoring retrieval differs from the completed retrieval audit")
    cross = score_selected_documents(
        topic,
        result.selection,
        decomposition.result.subnarratives,
        scorer,
    )
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
    _write_jsonl(root / artifacts[3], [_cross_row(row) for row in cross])
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
) -> TopicPhaseOutcome:
    """Run one configured topic through the private checkpointed workflow."""
    if not isinstance(config, FacetPilotConfig):
        raise TypeError("config must be a FacetPilotConfig")
    if not isinstance(dependencies, _RuntimeDependencies):
        raise TypeError("dependencies must be _RuntimeDependencies")
    runtime_topic = _runtime_topic(topic)
    decomposition, _ = _decompose_topic(
        runtime_topic,
        config.output_dir,
        dependencies.planning_backend,
    )
    _retrieve_topic(
        runtime_topic,
        decomposition,
        output_dir=config.output_dir,
        cache_dir=config.retrieval.cache_dir,
        code_commit=dependencies.code_commit,
        retriever=dependencies.retriever,
        retrieval_depth=config.retrieval.candidate_depth_per_query,
    )
    _score_topic(
        runtime_topic,
        decomposition,
        output_dir=config.output_dir,
        cache_dir=config.retrieval.cache_dir,
        score_cache_root=config.reranking.score_cache_dir,
        code_commit=dependencies.code_commit,
        retriever=dependencies.retriever,
        scorer=dependencies.document_scorer,
        device=config.reranking.device,
        retrieval_depth=config.retrieval.candidate_depth_per_query,
        rerank_depth=config.reranking.rerank_depth_per_query,
        selection_k=config.reranking.candidate_pool_depth,
    )
    return _canonical_topic(
        runtime_topic,
        decomposition,
        config=config,
        official_topics_sha256=identity,
        dependencies=dependencies,
    )


def _decompose_topic(
    topic: Topic,
    output_dir: Path,
    backend: Any,
) -> tuple[ValidatedDecomposition, bool]:
    path = output_dir / topic.id / "decomposition" / "result.json"
    if path.is_file():
        return load_validated_decomposition(topic, path), True
    if backend is None:
        backend = OpenRouterDeepSeekFacetBackend()
    result = extract_facets(topic, backend)
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
    _write_json(path, record)
    return load_validated_decomposition(topic, path), False


def _canonical_topic(
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    config: FacetPilotConfig,
    official_topics_sha256: str,
    dependencies: _RuntimeDependencies,
) -> TopicPhaseOutcome:
    """Run the local evidence tail and canonical stage for one topic."""
    root = config.output_dir / topic.id
    handoff_root = root / "canonical" / "handoff"
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=config.output_dir,
        output_dir=handoff_root,
        code_commit=dependencies.code_commit,
        official_topics_sha256=official_topics_sha256,
    )
    canonical_root = root / "canonical"
    selected_budget = config.nuggets.evidence_budget_per_subnarrative
    selection_policy = SelectionPolicy(budgets=(selected_budget,))
    manifest = canonical_root / "complete.json"
    artifacts = (
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "canonical/candidates.jsonl",
        "canonical/candidate-manifest.json",
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
    if _resume(manifest, root, expected, artifacts):
        return TopicPhaseOutcome(topic.id, "canonical", manifest, True)

    candidate_artifacts = generate_candidate_artifacts(
        handoff,
        score_cache_root=config.reranking.score_cache_dir,
        device=config.reranking.device,
        scorer=dependencies.candidate_scorer,
    )
    selection_artifacts = select_evidence_artifacts(
        candidate_artifacts,
        handoff.contexts_path,
        device=config.reranking.device,
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
    )
    _complete_atomic(manifest, root, expected, artifacts)
    return TopicPhaseOutcome(topic.id, "canonical", manifest, False)


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
    if isinstance(getattr(retriever, "identity", None), Mapping):
        identity = dict(retriever.identity)
    else:
        config, remote = retriever.config, retriever.client.config
        identity = {
            "name": config.name,
            "type": config.type,
            "index": config.index,
            "index_url": remote.index_url,
            "hits": config.hits,
        }
    if identity.get("hits") != retrieval_depth:
        raise ValueError("retriever must bind the configured retrieval depth")
    return identity


def _expected(
    phase: str,
    topic: Topic,
    decomposition: ValidatedDecomposition,
    commit: str,
    retriever: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": commit,
        "retriever": retriever,
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
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "requested_depth": retrieval_depth,
        "lanes": lanes,
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
    return _audit(
        topic,
        decomposition,
        lanes,
        retrieval_depth=result.lanes[0].retrieval_requested_depth,
    )


def _score_row(score: LaneDocumentScore) -> dict[str, object]:
    row = jsonable(score)
    row["text_sha256"] = _hash(row.pop("text").encode())
    row.pop("bm25_query")
    row.pop("semantic_query")
    return row


def _cross_row(value: Any) -> dict[str, object]:
    row = _score_row(value.score)
    row.pop("bm25_query_sha256")
    row.update(
        {
            "selection_rank": value.selection_rank,
            "subnarrative_id": value.subnarrative_id,
            "bm25_queries": list(value.bm25_queries),
            "bm25_query_sha256s": list(value.bm25_query_sha256s),
            "downstream_only": True,
        }
    )
    return row


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
    receipts = []
    for relative in artifacts:
        byte_count, digest = _file_receipt(root / relative)
        receipts.append(
            {"relative_path": relative, "bytes": byte_count, "sha256": digest}
        )
    _write_json(manifest, values | {"artifacts": receipts})


def _complete_atomic(
    manifest: Path,
    root: Path,
    values: dict[str, object],
    artifacts: tuple[str, ...],
) -> None:
    receipts = []
    for relative in artifacts:
        byte_count, digest = _file_receipt(root / relative)
        receipts.append(
            {"relative_path": relative, "bytes": byte_count, "sha256": digest}
        )
    _write_json(manifest, values | {"artifacts": receipts})


def _write_json(path: Path, value: object) -> None:
    body = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
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


def _production_dependencies() -> _RuntimeDependencies:
    repo = Path(__file__).resolve().parents[2]
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


def run_official(
    config: str | Path,
    *,
    topic_ids: Sequence[str] | None = None,
    topic_subset: Path | None = None,
    external: ExternalAdapters | None = None,
) -> RunReceipt:
    """Run selected official topics and export their organizer-compatible receipt."""
    return _run_official(
        config,
        topic_ids=topic_ids,
        topic_subset=topic_subset,
        external=external,
        dependency_factory=_production_dependencies,
    )


def _run_official(
    config: str | Path,
    *,
    topic_ids: Sequence[str] | None,
    topic_subset: Path | None,
    external: ExternalAdapters | None,
    dependency_factory: Callable[[], _RuntimeDependencies],
) -> RunReceipt:
    """Private implementation that delays runtime construction until preflight passes."""
    config_path = Path(config)
    loaded_config = load_facet_pilot_config(config_path)
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

    pending_topics = tuple(
        topic
        for topic in selected_topics
        if not (loaded_config.output_dir / topic.id / "canonical" / "complete.json").is_file()
    )
    resumed_topics = tuple(topic for topic in selected_topics if topic not in pending_topics)
    validate_retrieval_topic_checkpoints(loaded_config, resumed_topics)
    resumed_topic_ids = [topic.id for topic in resumed_topics]

    dependencies = dependency_factory()
    shared_retriever = adapters.retriever
    shared_scorer = dependencies.document_scorer
    if pending_topics and shared_retriever is None:
        shared_retriever = build_pyserini_retriever(
            loaded_config.retrieval.cache_dir,
            index=loaded_config.retrieval.index,
            hits=loaded_config.retrieval.candidate_depth_per_query,
        )
    if pending_topics and shared_scorer is None:
        shared_scorer = MixedbreadCoverageScorer(
            artifact_dir=loaded_config.output_dir / "scorer-ledger",
            score_cache_root=loaded_config.reranking.score_cache_dir,
            device=loaded_config.reranking.device,
        )
    dependencies = replace(
        dependencies,
        planning_backend=adapters.planning_backend,
        retriever=shared_retriever,
        document_scorer=shared_scorer,
        canonical_backend_factory=adapters.canonical_backend_factory,
    )

    for topic in pending_topics:
        outcome = _run_topic(
            topic,
            loaded_config,
            official_topics_sha256,
            dependencies,
        )
        if outcome.resumed and topic.id not in resumed_topic_ids:
            resumed_topic_ids.append(topic.id)

    if any(
        not (loaded_config.output_dir / topic.id / "canonical" / "complete.json").is_file()
        for topic in selected_topics
    ):
        raise RuntimeError("all selected canonical checkpoints must exist before export")
    export_retrieval_run(
        loaded_config,
        selected_topics,
        code_commit=dependencies.code_commit,
    )
    retrieval_export = read_retrieval_export_receipt(loaded_config, selected_topics)
    return RunReceipt(
        experiment_id=loaded_config.experiment.id,
        selected_topic_ids=tuple(topic.id for topic in selected_topics),
        resumed_topic_ids=tuple(resumed_topic_ids),
        retrieval_export=retrieval_export,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the official facet retrieval export.")
    parser.add_argument("config", type=Path)
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--topic", action="append", dest="topic_ids")
    selectors.add_argument("--topic-subset", type=Path)
    args = parser.parse_args(argv)
    receipt = run_official(
        args.config,
        topic_ids=None if args.topic_ids is None else tuple(args.topic_ids),
        topic_subset=args.topic_subset,
    )
    print(f"output={receipt.retrieval_export.manifest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

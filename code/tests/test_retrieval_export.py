from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import os
import pickle
from pathlib import Path
import shutil
import sqlite3
import stat
import threading
from types import SimpleNamespace
from typing import Sequence
import zipfile

import pytest

import trec_rag.retrieval_export as retrieval_export_module
import trec_rag.topic_records as topic_records_module
from trec_rag.canonical_nuggets import run_canonical_stage
from trec_rag.document_store import DocumentStore
from trec_rag.evidence_bundle import EvidenceBundle
from trec_rag.facet_pilot_config import (
    ExperimentSettings,
    FacetPilotConfig,
    NuggetSettings,
    PassageSettings,
    RetrievalSettings,
)
from trec_rag.evidence_store import CandidateArtifacts, select_evidence_artifacts
from trec_rag.facet_evidence import (
    SENTENCE_SPLITTER_VERSION,
    CandidateSubnarrative,
    ExtractiveCandidate,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SelectionPolicy,
    extract_document_candidates,
)
from trec_rag.facet_extraction import BackendReply, plan_facet_queries
from trec_rag.facet_retrieval import LaneDocumentScore, PassageScore
from trec_rag.generation_handoff import GenerationHandoff, load_generation_handoff
from trec_rag.mixedbread_passage_scorer import ScoredPassage as MixedbreadScoredPassage
from trec_rag.competition_retrieval import (
    INTERNAL_FIXED_SELECTION_K,
    _build_topic_passage_search,
    _configured_passage_search_identity,
    document_store_dir,
    _plan_payload,
    _retrieve_topic,
    _score_topic,
    load_validated_decomposition,
)
from trec_rag.pipeline_models import RetrievedCandidate, jsonable
from trec_rag.retrieval_export import (
    RetrievalExportReceipt,
    TopicProjectionReceipt,
    build_topic_projection as _build_topic_projection,
    export_retrieval_run,
    read_retrieval_export_receipt,
    read_topic_projection_receipt,
    validate_retrieval_topic_checkpoints,
)
from trec_rag.topic_records import (
    TopicRecords,
    TopicRecordsBuilder,
    TopicRecordsIntegrityError,
)
from trec_rag.topics import Topic


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _receipt(topic_root: Path, relative_path: str) -> dict[str, object]:
    body = (topic_root / relative_path).read_bytes()
    return {
        "relative_path": relative_path,
        "bytes": len(body),
        "sha256": sha256(body).hexdigest(),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _root_artifact_bodies(receipt: RetrievalExportReceipt) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in (
            receipt.official_run,
            receipt.with_text_archive,
            receipt.generation_handoff,
        )
    }


_export_retrieval_run = export_retrieval_run
_FIXTURE_DECOMPOSITION_PRODUCER_SHA256 = sha256(
    b"fixture-decomposition-producer-v1"
).hexdigest()


def export_retrieval_run(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
    *,
    code_commit: str,
) -> RetrievalExportReceipt:
    """Adapt historical fixtures to the receipt-only, manifest-last API."""
    receipts = []
    for topic in topics:
        topic_root = config.output_dir / topic.id
        projection_path = topic_root / "canonical" / "retrieval-projection.json"
        projection_manifest_path = (
            topic_root / "canonical" / "retrieval-projection-manifest.json"
        )
        if projection_path.is_file() and projection_manifest_path.is_file():
            receipts.append(read_topic_projection_receipt(config, topic))
            continue
        try:
            with TopicRecords.open(
                topic_root / "records.sqlite3",
                topic_root / "canonical" / "records-manifest.json",
                topic.id,
                DocumentStore(document_store_dir(config.root_dir)),
            ) as records:
                complete = topic_root / "canonical" / "complete.json"
                provisional = complete.read_bytes()
                complete.unlink()
                try:
                    receipts.append(
                        build_topic_projection(
                            config,
                            topic,
                            records,
                            canonical_manifest_bytes=provisional,
                        )
                    )
                except BaseException:
                    if not complete.exists():
                        complete.write_bytes(provisional)
                    raise
        except topic_records_module.TopicRecordsIntegrityError as exc:
            raise ValueError(str(exc)) from exc
    return _export_retrieval_run(
        config,
        topics,
        tuple(receipts),
        code_commit=code_commit,
    )


def _normal_plan(topic: Topic):
    result = plan_facet_queries(
        topic,
        {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": [
                {
                    "subnarrative": f"Coverage for subnarrative {index}.",
                    "bm25_queries": [f"subnarrative-{index} lexical"],
                }
                for index in (1, 2)
            ],
        },
    )
    assert result.plan is not None and not result.used_fallback
    return result


class _ConstantSentenceScorer:
    def score_pairs(self, pairs):
        return [1.0] * len(pairs)


class _DistinctSimilarity:
    identity = {"model": "fake-minilm"}

    def cosine_matrix(self, texts):
        return [
            [1.0 if left == right else 0.0 for right in range(len(texts))]
            for left in range(len(texts))
        ]


class _CanonicalFixtureBackend:
    def complete(self, request):
        return BackendReply(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "claim": f"Fixture canonical claim {index}.",
                            "evidence_aliases": [evidence.alias],
                            "importance": "vital",
                        }
                        for index, evidence in enumerate(request.evidence, start=1)
                    ]
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            response_body=b'{"fixture":"canonical"}',
            status=200,
            metadata={
                "requested_model": "deepseek/deepseek-v4-flash-20260423",
                "response_model": "deepseek/deepseek-v4-flash-20260423",
                "provider": "fixture",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


class _FixtureRetriever:
    identity = {
        "name": "climbmix_bm25",
        "type": "pyserini_remote",
        "index": "climbmix-400b",
        "index_url": "https://retrieval.example.invalid/search",
        "hits": 1000,
        "corpus_epoch": "test-corpus-epoch",
        "retrieval_cache_schema": "organizer-retrieval-cache-v2",
        "parser_version": "organizer-response-v2",
        "extractor_version": "organizer-exact-doc-string-v1",
        "field_path": ["doc"],
        "scoring_normalizer_version": "whitespace-score-v1",
    }

    def __init__(self, topic: Topic, docids: Sequence[str]) -> None:
        self.topic = topic
        self.docids = tuple(docids)

    def retrieve(self, query):
        if query.variant_name != "original":
            return []
        return [
            RetrievedCandidate(
                self.topic.id,
                query.variant_name,
                "climbmix_bm25",
                query.query_text,
                docid,
                rank,
                float(100 - rank),
                f"Full text for {docid}.",
            )
            for rank, docid in enumerate(self.docids, start=1)
        ]


class _FixturePassageScorer:
    def __init__(self, identity: dict[str, object]) -> None:
        self.identity = dict(identity)

    @staticmethod
    def cache_key(query_text: str, passage_text: str) -> str:
        return sha256(f"{query_text}\0{passage_text}".encode()).hexdigest()

    @staticmethod
    def rank(query_text, chunks):
        del query_text
        return tuple(
            MixedbreadScoredPassage(chunk, float(len(chunks) - index))
            for index, chunk in enumerate(chunks)
        )


def build_topic_projection(
    config: FacetPilotConfig,
    topic: Topic,
    records: TopicRecords,
    *,
    expected_retriever_identity: dict[str, object] | None = None,
    decomposition_producer_sha256: str = _FIXTURE_DECOMPOSITION_PRODUCER_SHA256,
    canonical_manifest_bytes: bytes | None = None,
) -> TopicProjectionReceipt:
    """Bind historical fixtures to the fresh manifest-last publication path."""
    expected = (
        dict(_FixtureRetriever.identity)
        if expected_retriever_identity is None
        else expected_retriever_identity
    )
    complete = config.output_dir / topic.id / "canonical" / "complete.json"
    projection_files = (
        complete.parent / "retrieval-projection.json",
        complete.parent / "retrieval-projection-manifest.json",
        complete.parent / "generation-projection.json",
        complete.parent / "generation-projection-manifest.json",
    )
    provisional = canonical_manifest_bytes
    removed_fixture_complete = False
    if (
        provisional is None
        and complete.exists()
        and not any(path.exists() for path in projection_files)
    ):
        provisional = complete.read_bytes()
        complete.unlink()
        removed_fixture_complete = True
    try:
        return _build_topic_projection(
            config,
            topic,
            records,
            expected_retriever_identity=expected,
            decomposition_producer_sha256=decomposition_producer_sha256,
            config_sha256=sha256(
                f"fixture-config:{config.run_id}".encode("utf-8")
            ).hexdigest(),
            canonical_manifest_bytes=provisional,
        )
    except BaseException:
        if removed_fixture_complete and not complete.exists():
            assert provisional is not None
            complete.write_bytes(provisional)
        raise


class _FixtureDocumentScorer:
    identity = {
        "model": "mixedbread-ai/mxbai-rerank-base-v2",
        "model_revision": "test-revision",
        "backend_version": "test-backend",
        "score_representation": "raw_logits",
    }

    def score_lane(self, topic, lane, candidates):
        return tuple(
            LaneDocumentScore(
                topic_id=topic.id,
                lane_name=lane.retrieval_query.variant_name,
                bm25_query=lane.retrieval_query.query_text,
                bm25_query_sha256=lane.bm25_query_sha256,
                semantic_query=lane.scoring_query.query_text,
                semantic_query_sha256=lane.semantic_query_sha256,
                docid=row.docid,
                text=row.text,
                bm25_rank=row.rank,
                bm25_score=row.score,
                aggregate_rank=rank,
                aggregate_score=4.0,
                long_document_raw_logit=4.0,
                weighted_passage_raw_logit=3.0,
                within_document_span_support=2,
                winning_passages=(PassageScore(0, 0, len(row.text), 3.0, 1),),
            )
            for rank, row in enumerate(candidates, start=1)
        )


def _candidate_for(
    topic: Topic,
    docid: str,
    subnarrative_id: str,
    subnarrative_text: str,
) -> ExtractiveCandidate:
    source = f"Full text for {docid}."
    source_sha256 = sha256(source.encode("utf-8")).hexdigest()
    request = ExtractiveCandidateRequest(
        topic_id=topic.id,
        document_id=docid,
        source=source,
        document_sha256=source_sha256,
        scoring_text_sha256=source_sha256,
        subnarratives=(CandidateSubnarrative(subnarrative_id, subnarrative_text),),
        passages=(
            ScoredPassage(
                passage_id=f"{topic.id}:{subnarrative_id}:{docid}:0000",
                lane_id=f"subnarrative:{subnarrative_id}",
                query_id=subnarrative_id,
                scoring_start_char=0,
                scoring_end_char=len(source),
                scoring_text_sha256=source_sha256,
                chunk_text_sha256=source_sha256,
                cross_encoder_score=3.0,
                cross_encoder_rank=1,
            ),
        ),
    )
    candidates = extract_document_candidates(request, _ConstantSentenceScorer())
    assert len(candidates) == 1
    return candidates[0]


def _write_evidence_artifacts(
    canonical: Path,
    topic: Topic,
    plan,
    supported: Sequence[str],
    document_store_root: Path,
    *,
    run_id: str,
    retrieval_status: str = "complete",
    retrieval_stopping_reason: str = "coverage_sufficient",
) -> list[ExtractiveCandidate]:
    policy = SelectionPolicy(budgets=(40,), precluster_limit=400)
    fixture_docids = tuple(supported) or ("unselected-doc",)
    candidates = [
        _candidate_for(
            topic,
            docid,
            subnarrative.subnarrative_id,
            subnarrative.text,
        )
        for docid in fixture_docids
        for subnarrative in plan.subnarratives
    ]
    request_bytes = b""
    (canonical / "handoff" / "candidate-requests.jsonl").parent.mkdir(
        parents=True, exist_ok=True
    )
    (canonical / "handoff" / "candidate-requests.jsonl").write_bytes(request_bytes)
    scorer_identity = {
        "model": "fake-extractive-scorer",
        "model_revision": "test-revision",
        "backend_version": "test-backend",
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "score_kind": "extractive_sentence_v1",
        "sentence_max_length": 512,
        "input_policy": "trec_rag_whitespace_v1",
    }
    identity = {
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "source_file": "candidate-requests.jsonl",
        "source_sha256": sha256(request_bytes).hexdigest(),
        "scorer": scorer_identity,
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": "trec_rag_whitespace_v1",
    }
    topic_root = canonical.parent
    builder = TopicRecordsBuilder(
        topic_root,
        topic.id,
        DocumentStore(document_store_root),
        run_id=run_id,
    )
    for docid in fixture_docids:
        builder.bind_document(docid, f"Full text for {docid}.")
    for candidate in candidates:
        builder.add_candidate(candidate)
    builder.set_completion(retrieval_status, retrieval_stopping_reason)
    builder.publish(identity)

    context_rows = [
        {
            "schema_version": "subnarrative_selection_context_v1",
            "topic_id": topic.id,
            "official_narrative": topic.narrative,
            "official_narrative_sha256": sha256(topic.narrative.encode()).hexdigest(),
            "subnarrative_id": subnarrative.subnarrative_id,
            "subnarrative_text": subnarrative.text,
            "subnarrative_sha256": subnarrative.semantic_query_sha256,
        }
        for subnarrative in plan.subnarratives
    ]
    context_bytes = b"".join(_canonical_json(row) for row in context_rows)
    contexts_path = canonical / "handoff" / "selection-contexts.jsonl"
    contexts_path.write_bytes(context_bytes)
    selected_artifacts = select_evidence_artifacts(
        CandidateArtifacts(
            topic_root / "records.sqlite3",
            canonical / "records-manifest.json",
            document_store_root,
        ),
        contexts_path,
        device="cpu",
        similarity=_DistinctSimilarity(),
        policy=policy,
    )
    if not supported:
        selection_rows = _read_jsonl(selected_artifacts.selections_path)
        for row in selection_rows:
            row["clusters"] = []
            row["semantic_cluster_count"] = 0
            row["snapshots"] = [
                {**snapshot, "cluster_ids": [], "exhausted": True}
                for snapshot in row["snapshots"]
            ]
        selection_bytes = b"".join(_canonical_json(row) for row in selection_rows)
        selected_artifacts.selections_path.write_bytes(selection_bytes)
        selection_manifest = json.loads(selected_artifacts.manifest_path.read_bytes())
        selection_manifest.update(
            {
                "selections_sha256": sha256(selection_bytes).hexdigest(),
                "output_sha256": sha256(selection_bytes).hexdigest(),
                "semantic_cluster_count": 0,
                "selected_cluster_count": 0,
            }
        )
        selected_artifacts.manifest_path.write_bytes(_canonical_json(selection_manifest))
    return candidates


def _config_and_topics(tmp_path: Path) -> tuple[FacetPilotConfig, tuple[Topic, ...]]:
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text(
        "rag2026-0\tExplain a demonstrated topic.\n", encoding="utf-8"
    )
    topics = (Topic("rag2026-0", "", "Explain a demonstrated topic."),)
    config = FacetPilotConfig(
        root_dir=tmp_path,
        experiment=ExperimentSettings("demo"),
        topics_path=topics_path,
        retrieval=RetrievalSettings(
            index="climbmix-400b",
            cache_dir=tmp_path / "cache" / "retrieval",
            query_sources=("original", "subnarrative"),
            documents_per_query=1000,
        ),
        passage=PassageSettings(
            model="mixedbread-ai/mxbai-rerank-base-v2",
            score_cache_dir=tmp_path / "cache" / "reranker",
            device="auto",
            passages_per_query=100,
            chunk_max_characters=3500,
            chunk_overlap_characters=350,
        ),
        nuggets=NuggetSettings(
            evidence_budget_per_subnarrative=40,
            maximum_claims_per_subnarrative=20,
            maximum_supporting_documents_per_claim=3,
        ),
    )
    return config, topics


def _write_sealed_topic(
    output_dir: Path,
    topic: Topic,
    *,
    selected: Sequence[str],
    supported: Sequence[str],
    source_commit: str,
    official_topics: Sequence[Topic] | None = None,
    retrieval_status: str = "complete",
    retrieval_stopping_reason: str = "coverage_sufficient",
) -> None:
    topic_root = output_dir / topic.id
    scoring = topic_root / "scoring"
    canonical = topic_root / "canonical"
    canonical.mkdir(parents=True)

    planning = _normal_plan(topic)
    plan = planning.plan
    assert plan is not None
    decomposition_path = topic_root / "decomposition" / "result.json"
    decomposition_path.parent.mkdir(parents=True, exist_ok=True)
    decomposition_path.write_bytes(
        _canonical_json(
            {
                "schema_version": "facet_pilot_v2",
                "topic": {"id": topic.id, "narrative": topic.narrative},
                "narrative_sha256": sha256(topic.narrative.encode()).hexdigest(),
                "used_fallback": False,
                "error": None,
                "queries": jsonable(planning.queries),
                "plan": _plan_payload(plan),
                "subnarratives": jsonable(planning.subnarratives),
            }
        )
    )
    decomposition = load_validated_decomposition(topic, decomposition_path)
    retriever = _FixtureRetriever(topic, selected)
    passage_identity = _configured_passage_search_identity(
        retrieval_depth=1000,
        passages_per_query=100,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        model="mixedbread-ai/mxbai-rerank-base-v2",
        device="auto",
    )
    passage_search = _build_topic_passage_search(
        topic,
        retriever=retriever,
        scorer=_FixturePassageScorer(passage_identity["scorer"]),
        document_store_root=document_store_dir(output_dir.parent.parent),
        retrieval_cache_dir=output_dir / "cache",
        retrieval_index="climbmix-400b",
        corpus_epoch="test-corpus-epoch",
        score_cache_root=output_dir / "score-cache",
        device="auto",
        retrieval_depth=1000,
        passages_per_query=100,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )
    _retrieve_topic(
        topic,
        decomposition,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        code_commit=source_commit,
        corpus_epoch="test-corpus-epoch",
        retriever=retriever,
        retrieval_depth=1000,
        passage_search=passage_search,
        passage_identity=passage_identity,
    )
    _score_topic(
        topic,
        decomposition,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        score_cache_root=output_dir / "score-cache",
        code_commit=source_commit,
        corpus_epoch="test-corpus-epoch",
        retriever=retriever,
        scorer=None,
        device="auto",
        retrieval_depth=1000,
        rerank_depth=1000,
        selection_k=INTERNAL_FIXED_SELECTION_K,
        expected_retriever_identity=retriever.identity,
        expected_passage_identity=passage_identity,
    )
    scoring_manifest_path = scoring / "complete.json"

    _write_evidence_artifacts(
        canonical,
        topic,
        plan,
        supported,
        document_store_dir(output_dir.parent.parent),
        run_id=output_dir.name,
        retrieval_status=retrieval_status,
        retrieval_stopping_reason=retrieval_stopping_reason,
    )
    run_canonical_stage(
        selections_path=canonical / "subnarrative-selections.jsonl",
        selection_manifest_path=canonical / "selection-manifest.json",
        selected_budget=40,
        output_path=canonical / "canonical-nuggets.jsonl",
        manifest_path=canonical / "canonical-nugget-manifest.json",
        cache_dir=canonical / "response-cache",
        max_canonical_claims=20,
        max_supporting_documents_per_claim=3,
        backend_factory=_CanonicalFixtureBackend,
        cache_ignore_checker=lambda _path: True,
    )
    canonical_paths = (
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
    for relative in canonical_paths:
        path = topic_root / relative
        if path.exists() or relative == "canonical/canonical-nugget-manifest.json":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json({"fixture": relative}))
    official_topics_sha256 = sha256(
        json.dumps(
            [
                {"id": official.id, "narrative": official.narrative}
                for official in (
                    (topic,) if official_topics is None else official_topics
                )
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    canonical_manifest = {
        "schema_version": "facet_pilot_v2",
        "phase": "canonical",
        "topic_id": topic.id,
        "official_topics_sha256": official_topics_sha256,
        "narrative_sha256": sha256(topic.narrative.encode()).hexdigest(),
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": source_commit,
        "scoring_manifest_sha256": sha256(
            scoring_manifest_path.read_bytes()
        ).hexdigest(),
        "handoff_manifest_sha256": sha256(
            (topic_root / "canonical/handoff/handoff-manifest.json").read_bytes()
        ).hexdigest(),
        "input_roles": [
            "official_topic_narrative",
            "validated_generated_decomposition",
            "sealed_scoring_checkpoint",
        ],
        "selection_policy": {
            "budgets": [40],
            "precluster_limit": 400,
            "semantic_threshold": 0.92,
            "mmr_lambda": 0.7,
        },
        "selected_budget": 40,
        "canonical_claim_cap": 20,
        "canonical_supporting_document_cap": 3,
        "artifacts": [_receipt(topic_root, path) for path in canonical_paths],
    }
    (canonical / "complete.json").write_bytes(_canonical_json(canonical_manifest))


def _write_canonical_nugget_manifest(
    topic_root: Path,
    *,
    request_sha256s: list[str] | None = None,
) -> None:
    canonical = topic_root / "canonical"
    nuggets = (canonical / "canonical-nuggets.jsonl").read_bytes()
    rows = _read_jsonl(canonical / "canonical-nuggets.jsonl")
    states: dict[str, int] = {}
    for row in rows:
        state = str(row["state"])
        states[state] = states.get(state, 0) + 1
    manifest = {
        "schema_version": "canonical_nugget_manifest_v2",
        "result_schema_version": "canonical_nugget_result_v2",
        "canonical_response_schema_version": "canonical_nuggets_v2",
        "selection_schema_version": "subnarrative_selection_v1",
        "selection_manifest_schema_version": "subnarrative_selection_manifest_v1",
        "selection_file": "subnarrative-selections.jsonl",
        "selection_manifest_file": "selection-manifest.json",
        "canonical_nugget_file": "canonical-nuggets.jsonl",
        "selections_sha256": sha256(
            (canonical / "subnarrative-selections.jsonl").read_bytes()
        ).hexdigest(),
        "selection_manifest_sha256": sha256(
            (canonical / "selection-manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_nuggets_sha256": sha256(nuggets).hexdigest(),
        "output_sha256": sha256(nuggets).hexdigest(),
        "selected_budget": rows[0]["selected_budget"] if rows else 40,
        "selection_count": len(rows),
        "result_count": len(rows),
        "state_counts": dict(sorted(states.items())),
        "max_canonical_claims": 20,
        "max_supporting_documents_per_claim": 3,
        "request_sha256s": (
            [str(row["request_sha256"]) for row in rows]
            if request_sha256s is None
            else request_sha256s
        ),
        "model": "deepseek/deepseek-v4-flash-20260423",
        "prompt_version": "canonical_nuggetizer_v5",
        "hosted_llm_calls": 0,
        "validated_cache_hits": 0,
        "raw_cache_writes": 0,
        "validated_cache_writes": 0,
    }
    (canonical / "canonical-nugget-manifest.json").write_bytes(
        _canonical_json(manifest)
    )


def _rewrite_manifest(path: Path, **changes: object) -> None:
    manifest = json.loads(path.read_bytes())
    manifest.update(changes)
    path.write_bytes(_canonical_json(manifest))


def _resign_scoring_manifest(topic_root: Path) -> None:
    scoring_manifest_path = topic_root / "scoring" / "complete.json"
    canonical_manifest_path = topic_root / "canonical" / "complete.json"
    canonical = json.loads(canonical_manifest_path.read_bytes())
    canonical["scoring_manifest_sha256"] = sha256(
        scoring_manifest_path.read_bytes()
    ).hexdigest()
    canonical_manifest_path.write_bytes(_canonical_json(canonical))


def _resign_scoring_artifact(topic_root: Path, relative_path: str) -> None:
    scoring_manifest_path = topic_root / "scoring" / "complete.json"
    scoring = json.loads(scoring_manifest_path.read_bytes())
    receipt = next(
        row for row in scoring["artifacts"] if row["relative_path"] == relative_path
    )
    receipt.update(_receipt(topic_root, relative_path))
    scoring_manifest_path.write_bytes(_canonical_json(scoring))
    _resign_scoring_manifest(topic_root)


def _resign_retrieval_manifest_into_scoring(topic_root: Path) -> None:
    retrieval_manifest = topic_root / "retrieval" / "complete.json"
    scoring_manifest = topic_root / "scoring" / "complete.json"
    scoring = json.loads(scoring_manifest.read_bytes())
    scoring["retrieval_manifest_sha256"] = sha256(
        retrieval_manifest.read_bytes()
    ).hexdigest()
    scoring_manifest.write_bytes(_canonical_json(scoring))
    _resign_scoring_manifest(topic_root)


def _resign_retrieval_artifact(topic_root: Path, relative_path: str) -> None:
    retrieval_manifest = topic_root / "retrieval" / "complete.json"
    retrieval = json.loads(retrieval_manifest.read_bytes())
    receipt = next(
        row for row in retrieval["artifacts"] if row["relative_path"] == relative_path
    )
    receipt.update(_receipt(topic_root, relative_path))
    retrieval_manifest.write_bytes(_canonical_json(retrieval))
    _resign_retrieval_manifest_into_scoring(topic_root)


def _resign_canonical_artifact(topic_root: Path, relative_path: str) -> None:
    canonical_manifest = topic_root / "canonical" / "complete.json"
    canonical = json.loads(canonical_manifest.read_bytes())
    receipt = next(
        row for row in canonical["artifacts"] if row["relative_path"] == relative_path
    )
    receipt.update(_receipt(topic_root, relative_path))
    canonical_manifest.write_bytes(_canonical_json(canonical))


def _coherently_reseal_forged_source_span(topic_root: Path) -> None:
    records_path = topic_root / "records.sqlite3"
    connection = sqlite3.connect(records_path)
    try:
        changed = connection.execute(
            "UPDATE candidate_span SET start_byte=1 "
            "WHERE role='evidence' AND ordinal=0"
        ).rowcount
        assert changed > 0
        semantic_sha256 = topic_records_module._semantic_sha256(connection)
        row_counts = topic_records_module._row_counts(connection)
        sealed = connection.execute(
            "UPDATE stage_seal SET semantic_sha256=?, row_counts_json=? "
            "WHERE stage=?",
            (
                semantic_sha256,
                topic_records_module._canonical_json(row_counts),
                topic_records_module.CANDIDATE_STAGE,
            ),
        ).rowcount
        assert sealed == 1
        connection.commit()
    finally:
        connection.close()

    database_sha256 = sha256(records_path.read_bytes()).hexdigest()
    records_manifest_path = topic_root / "canonical" / "records-manifest.json"
    _rewrite_manifest(
        records_manifest_path,
        database_sha256=database_sha256,
        database_bytes=records_path.stat().st_size,
        semantic_sha256=semantic_sha256,
    )
    selection_manifest_path = topic_root / "canonical" / "selection-manifest.json"
    _rewrite_manifest(
        selection_manifest_path,
        records_database_sha256=database_sha256,
        candidate_semantic_sha256=semantic_sha256,
    )
    nugget_manifest_path = (
        topic_root / "canonical" / "canonical-nugget-manifest.json"
    )
    _rewrite_manifest(
        nugget_manifest_path,
        selection_manifest_sha256=sha256(
            selection_manifest_path.read_bytes()
        ).hexdigest(),
    )
    for relative_path in (
        "records.sqlite3",
        "canonical/records-manifest.json",
        "canonical/selection-manifest.json",
        "canonical/canonical-nugget-manifest.json",
    ):
        _resign_canonical_artifact(topic_root, relative_path)


def _resign_decomposition_digest_chain(topic_root: Path, digest: str) -> None:
    decomposition_path = topic_root / "decomposition.json"
    decomposition = json.loads(decomposition_path.read_bytes())
    decomposition["source_sha256"] = digest
    decomposition_path.write_bytes(_canonical_json(decomposition))

    audit_path = topic_root / "retrieval" / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    audit["decomposition_source_sha256"] = digest
    audit_path.write_bytes(_canonical_json(audit))

    retrieval_path = topic_root / "retrieval" / "complete.json"
    retrieval = json.loads(retrieval_path.read_bytes())
    retrieval["decomposition_source_sha256"] = digest
    for relative in ("decomposition.json", "retrieval/audit.json"):
        receipt = next(
            row for row in retrieval["artifacts"] if row["relative_path"] == relative
        )
        receipt.update(_receipt(topic_root, relative))
    retrieval_path.write_bytes(_canonical_json(retrieval))

    scoring_path = topic_root / "scoring" / "complete.json"
    scoring = json.loads(scoring_path.read_bytes())
    scoring["decomposition_source_sha256"] = digest
    scoring["retrieval_manifest_sha256"] = sha256(
        retrieval_path.read_bytes()
    ).hexdigest()
    scoring_path.write_bytes(_canonical_json(scoring))

    canonical_path = topic_root / "canonical" / "complete.json"
    canonical = json.loads(canonical_path.read_bytes())
    canonical["decomposition_source_sha256"] = digest
    canonical["scoring_manifest_sha256"] = sha256(
        scoring_path.read_bytes()
    ).hexdigest()
    canonical_path.write_bytes(_canonical_json(canonical))


def _write_fallback_decomposition_source(topic_root: Path, topic: Topic) -> None:
    path = topic_root / "decomposition" / "result.json"
    path.write_bytes(
        _canonical_json(
            {
                "schema_version": "facet_pilot_v2",
                "topic": {"id": topic.id, "narrative": topic.narrative},
                "narrative_sha256": sha256(topic.narrative.encode()).hexdigest(),
                "used_fallback": True,
                "error": "fixture rejected plan",
                "queries": [
                    {
                        "topic_id": topic.id,
                        "variant_name": "original",
                        "query_text": topic.narrative,
                        "source_type": "original_topic",
                    }
                ],
                "plan": None,
                "subnarratives": [],
            }
        )
    )
    _resign_decomposition_digest_chain(
        topic_root, sha256(path.read_bytes()).hexdigest()
    )


def test_export_rejects_resigned_planner_source_digest_chain(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    _resign_decomposition_digest_chain(config.output_dir / topic.id, "9" * 64)

    with pytest.raises(ValueError, match="decomposition|planner"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


@pytest.mark.parametrize("mutation", ("id", "kind", "claim"))
def test_export_rejects_resigned_canonical_result_semantic_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    path = topic_root / "canonical" / "canonical-nuggets.jsonl"
    rows = _read_jsonl(path)
    nugget = rows[0]["nuggets"][0]
    if mutation == "id":
        nugget["canonical_nugget_id"] = "canonical-forged"
    elif mutation == "kind":
        nugget["nugget_kind"] = "extractive_fallback"
    else:
        nugget["claim_text"] = "Altered but re-signed claim."
    path.write_bytes(b"".join(_canonical_json(row) for row in rows))
    _write_canonical_nugget_manifest(topic_root)
    _resign_canonical_artifact(topic_root, "canonical/canonical-nuggets.jsonl")
    _resign_canonical_artifact(
        topic_root, "canonical/canonical-nugget-manifest.json"
    )

    with pytest.raises(ValueError, match="canonical nugget"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


def test_export_rejects_resigned_incomplete_retrieval_audit(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    audit_path = topic_root / "retrieval" / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    audit["lanes"] = []
    audit_path.write_bytes(_canonical_json(audit))
    _resign_retrieval_artifact(topic_root, "retrieval/audit.json")

    with pytest.raises(ValueError, match="retrieval audit"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


@pytest.mark.parametrize(
    "mutation",
    ("semantic-hash", "bm25-hash", "missing-subnarrative"),
)
def test_export_rejects_resigned_plan_or_subnarrative_completeness_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    if mutation in {"semantic-hash", "bm25-hash"}:
        decomposition_path = topic_root / "decomposition.json"
        decomposition = json.loads(decomposition_path.read_bytes())
        field = (
            "semantic_query_sha256"
            if mutation == "semantic-hash"
            else "bm25_query_sha256s"
        )
        decomposition["subnarratives"][0][field] = (
            "0" * 64 if mutation == "semantic-hash" else ["0" * 64]
        )
        decomposition_path.write_bytes(_canonical_json(decomposition))
        _resign_retrieval_artifact(topic_root, "decomposition.json")
    else:
        scores_path = topic_root / "scoring" / "selected_subnarrative_scores.jsonl"
        scores = [
            row
            for row in _read_jsonl(scores_path)
            if row["subnarrative_id"] == "subnarrative-1"
        ]
        scores_path.write_bytes(b"".join(_canonical_json(row) for row in scores))
        _resign_scoring_artifact(
            topic_root, "scoring/selected_subnarrative_scores.jsonl"
        )
        nuggets_path = topic_root / "canonical" / "canonical-nuggets.jsonl"
        nuggets = [
            row
            for row in _read_jsonl(nuggets_path)
            if row["subnarrative_id"] == "subnarrative-1"
        ]
        nuggets_path.write_bytes(b"".join(_canonical_json(row) for row in nuggets))
        _write_canonical_nugget_manifest(topic_root)
        _resign_canonical_artifact(
            topic_root, "canonical/canonical-nuggets.jsonl"
        )
        _resign_canonical_artifact(
            topic_root, "canonical/canonical-nugget-manifest.json"
        )

    with pytest.raises(ValueError, match="decomposition|subnarrative|canonical"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


@pytest.mark.parametrize(
    "mutation",
    ("boolean-count", "incomplete-flag", "lane-status", "trace"),
)
def test_export_rejects_resigned_selection_semantic_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    selection_path = topic_root / "scoring" / "selection.json"
    selection = json.loads(selection_path.read_bytes())
    if mutation == "boolean-count":
        selection["selected_count"] = True
    elif mutation == "incomplete-flag":
        selection["complete"] = True
    elif mutation == "lane-status":
        selection["lane_statuses"][0]["selected_count"] = 0
    else:
        selection["trace"] = []
    selection_path.write_bytes(_canonical_json(selection))
    _resign_scoring_artifact(topic_root, "scoring/selection.json")

    with pytest.raises(ValueError, match="selection"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


@pytest.mark.parametrize("mutation", ("forged-alias", "forged-source-span"))
def test_export_rejects_resigned_canonical_evidence_absent_from_sealed_selection(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    if mutation == "forged-alias":
        nuggets_path = topic_root / "canonical" / "canonical-nuggets.jsonl"
        nuggets = _read_jsonl(nuggets_path)
        nuggets[0]["nuggets"][0]["evidence"][0]["candidate_nugget_id"] = "forged"
        nuggets_path.write_bytes(b"".join(_canonical_json(row) for row in nuggets))
        _write_canonical_nugget_manifest(topic_root)
        changed = (
            "canonical/canonical-nuggets.jsonl",
            "canonical/canonical-nugget-manifest.json",
        )
        error_pattern = "canonical evidence"
    else:
        _coherently_reseal_forged_source_span(topic_root)
        changed = ()
        error_pattern = "candidate evidence source span is inconsistent"
    for relative in changed:
        _resign_canonical_artifact(topic_root, relative)

    with pytest.raises(ValueError, match=error_pattern):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


def test_export_rejects_legacy_sidecars_without_deleting_them(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-b", "doc-a", "doc-c"),
        supported=("doc-a", "doc-c"),
        source_commit="a" * 40,
    )
    legacy_names = (
        "retrieval_candidate_pool.trec",
        "retrieval_provenance.jsonl",
        "resolved_config.yaml",
    )
    for name in legacy_names:
        (config.output_dir / name).write_text("legacy\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="legacy retrieval export artifacts are incompatible",
    ):
        export_retrieval_run(config, topics, code_commit="a" * 40)

    for name in legacy_names:
        assert (config.output_dir / name).read_bytes() == b"legacy\n"
    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_build_projection_rejects_base_only_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    canonical_root = config.output_dir / topic.id / "canonical"
    complete = canonical_root / "complete.json"
    original_complete = complete.read_bytes()

    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        canonical_root / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        with pytest.raises(
            ValueError,
            match="legacy canonical checkpoint is incompatible",
        ):
            _build_topic_projection(
                config,
                topic,
                records,
                expected_retriever_identity=dict(_FixtureRetriever.identity),
                decomposition_producer_sha256=(
                    _FIXTURE_DECOMPOSITION_PRODUCER_SHA256
                ),
                config_sha256=sha256(
                    f"fixture-config:{config.run_id}".encode("utf-8")
                ).hexdigest(),
            )

    assert complete.read_bytes() == original_complete
    assert not (canonical_root / "retrieval-projection.json").exists()
    assert not (canonical_root / "retrieval-projection-manifest.json").exists()
    assert not (canonical_root / "generation-projection.json").exists()
    assert not (canonical_root / "generation-projection-manifest.json").exists()


def test_export_uses_only_versioned_topic_records_artifacts(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    receipt = export_retrieval_run(config, topics, code_commit="a" * 40)

    assert receipt.official_run.read_text() == "rag2026-0 Q0 doc-a 1 1 demo\n"
    assert (config.output_dir / topics[0].id / "records.sqlite3").is_file()
    assert (
        config.output_dir / topics[0].id / "canonical" / "records-manifest.json"
    ).is_file()
    assert not (
        config.output_dir / topics[0].id / "canonical" / "candidates.jsonl"
    ).exists()
    assert not (
        config.output_dir / topics[0].id / "canonical" / "candidate-manifest.json"
    ).exists()


def test_export_writes_deterministic_bundle_projections_and_manifest(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topics = (
        replace(
            topics[0],
            narrative='Explain a "demonstrated" topic using C:\\data.',
        ),
    )
    config.topics_path.write_text(
        f"{topics[0].id}\t{topics[0].narrative}\n",
        encoding="utf-8",
    )
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="b" * 40,
    )

    first = export_retrieval_run(config, topics, code_commit="b" * 40)
    first_zip = first.with_text_archive.read_bytes()
    second = export_retrieval_run(config, topics, code_commit="b" * 40)

    assert second.with_text_archive.read_bytes() == first_zip
    with zipfile.ZipFile(second.with_text_archive) as archive:
        assert archive.namelist() == ["retrieval_with_text.jsonl"]
        member_bytes = archive.read("retrieval_with_text.jsonl")
        row = json.loads(member_bytes)
    canonical_nuggets = row["canonical_nuggets"]["doc-a"]
    assert {item["importance"] for item in canonical_nuggets} == {"vital"}
    assert {item["subnarrative_id"] for item in canonical_nuggets} == {
        "subnarrative-1",
        "subnarrative-2",
    }
    one_byte_mutation = member_bytes[:-1] + b" "
    assert len(one_byte_mutation) == len(member_bytes)
    assert one_byte_mutation != member_bytes
    assert member_bytes != one_byte_mutation
    assert row["query"] == {
        "qid": "rag2026-0",
        "selection_id": "official",
        "text": topics[0].narrative,
        "text_sha256": sha256(topics[0].narrative.encode("utf-8")).hexdigest(),
    }
    assert row["candidates"][0]["doc"] == "Full text for doc-a."
    assert row["candidates"][0]["text_sha256"] == sha256(
        b"Full text for doc-a."
    ).hexdigest()
    manifest = json.loads(second.manifest.read_bytes())
    assert manifest["schema_version"] == "retrieval_export_manifest_v6"
    assert manifest["export_code_commit"] == "b" * 40
    assert manifest["source_code_commits"] == ["b" * 40]
    assert manifest["score_semantics"] == "ordinal_selection_order"
    assert manifest["artifacts"]["r_output_trec_rag_2026.tsv"]["sha256"]
    assert set(manifest["artifacts"]) == {
        "generation_handoff_manifest.json",
        "r_output_trec_rag_2026.tsv",
        "retrieval_with_text.jsonl.zip",
    }
    handoff = load_generation_handoff(second.generation_handoff)
    assert isinstance(handoff, GenerationHandoff)
    assert handoff.producer.retrieval_run_id == config.run_id
    assert handoff.topics[0].citation_docids == ("doc-a",)
    assert handoff.topics[0].evidence[0].text == "Full text for doc-a."
    assert not (config.output_dir / "retrieval_candidate_pool.trec").exists()
    assert not (config.output_dir / "retrieval_provenance.jsonl").exists()
    assert not (config.output_dir / "resolved_config.yaml").exists()
    assert (
        str(document_store_dir(config.root_dir)).encode()
        not in first.official_run.read_bytes()
    )
    assert b"records.sqlite3" not in first_zip
    assert str(document_store_dir(config.root_dir)).encode() not in first_zip


def test_export_publishes_handoff_before_outer_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    real_atomic_write = retrieval_export_module._atomic_write
    publication_order: list[str] = []

    def record_write(path: Path, body: bytes) -> None:
        publication_order.append(path.name)
        real_atomic_write(path, body)

    monkeypatch.setattr(retrieval_export_module, "_atomic_write", record_write)

    export_retrieval_run(config, topics, code_commit="b" * 40)

    assert publication_order[-4:] == [
        "r_output_trec_rag_2026.tsv",
        "retrieval_with_text.jsonl.zip",
        "generation_handoff_manifest.json",
        "retrieval_export_manifest.json",
    ]


@pytest.mark.parametrize(
    "stopping_reason",
    ["budget_exhausted", "evidence_validation_failed"],
)
def test_incomplete_topic_projection_is_sealed_and_exportable_for_generation(
    tmp_path: Path,
    stopping_reason: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
        retrieval_status="incomplete",
        retrieval_stopping_reason=stopping_reason,
    )
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)

    assert receipt.retrieval_status == "incomplete"
    assert receipt.retrieval_stopping_reason == stopping_reason
    assert receipt.topic_snapshot_sha256 == receipt.records_receipt.semantic_sha256
    projection = json.loads(
        (
            config.output_dir
            / topic.id
            / "canonical"
            / "retrieval-projection.json"
        ).read_bytes()
    )
    assert projection["retrieval_status"] == "incomplete"
    assert projection["retrieval_stopping_reason"] == stopping_reason
    assert projection["topic_snapshot_sha256"] == receipt.topic_snapshot_sha256

    exported = _export_retrieval_run(
        config,
        topics,
        (receipt,),
        code_commit="b" * 40,
    )
    with zipfile.ZipFile(exported.with_text_archive) as archive:
        handoff = json.loads(archive.read("retrieval_with_text.jsonl"))
    assert handoff["retrieval_status"] == "incomplete"
    assert handoff["retrieval_stopping_reason"] == stopping_reason
    generation_topic = load_generation_handoff(
        exported.generation_handoff
    ).topics[0]
    assert generation_topic.topic_id == topic.id
    assert generation_topic.citation_docids == ("doc-a",)


def test_global_manifest_records_topic_workers_search_identity_and_ordered_statuses(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a", "doc-b"),
        supported=("doc-a", "doc-b"),
        source_commit="a" * 40,
    )
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)

    exported = _export_retrieval_run(
        config,
        topics,
        (receipt,),
        code_commit="b" * 40,
    )
    manifest = json.loads(exported.manifest.read_bytes())

    assert manifest["execution"] == {"topic_workers": 1}
    assert manifest["passage_search"]["policy"] == {
        "max_attempts": 3,
        "passage_limit": 100,
        "retrieval_depth": 1000,
    }
    assert manifest["passage_search"]["scorer"]["model"] == (
        "mixedbread-ai/mxbai-rerank-base-v2"
    )
    assert manifest["topic_statuses"] == {
        topic.id: {
            "status": "complete",
            "stopping_reason": "coverage_sufficient",
        }
    }
    assert manifest["topic_receipts"] == [
        {
            "topic_id": topic.id,
            "projection_manifest_sha256": receipt.manifest_sha256,
        }
    ]
    assert manifest["topic_depths"][topic.id]["official"] == 2


def test_global_manifest_is_not_published_when_any_topic_receipt_is_missing(
    tmp_path: Path,
) -> None:
    config, _ = _config_and_topics(tmp_path)
    topics = (
        Topic("topic-a", "", "Explain topic A."),
        Topic("topic-b", "", "Explain topic B."),
    )
    config.topics_path.write_text(
        "".join(f"{topic.id}\t{topic.narrative}\n" for topic in topics),
        encoding="utf-8",
    )
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
        official_topics=topics,
    )
    with TopicRecords.open(
        config.output_dir / topics[0].id / "records.sqlite3",
        config.output_dir / topics[0].id / "canonical" / "records-manifest.json",
        topics[0].id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topics[0], records)

    with pytest.raises(ValueError, match="coverage differs|coverage is incomplete"):
        _export_retrieval_run(
            config,
            topics,
            (receipt,),
            code_commit="b" * 40,
        )
    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-topic-db",
        "missing-cas",
        "corrupt-cas",
        "modified-db",
        "changed-semantic-seal",
        "stale-wal-shm",
        "malformed-selection-metadata",
    ),
)
def test_export_fails_closed_on_topic_records_integrity_failures(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    records_path = topic_root / "records.sqlite3"
    records_manifest_path = topic_root / "canonical" / "records-manifest.json"
    store_root = document_store_dir(config.root_dir)

    if mutation == "wrong-topic-db":
        wrong_topic = Topic("wrong-topic", "", topic.narrative)
        wrong_root = tmp_path / "wrong-topic-records"
        wrong_candidate = _candidate_for(
            wrong_topic,
            "doc-a",
            "subnarrative-1",
            "Coverage for subnarrative 1.",
        )
        wrong_builder = TopicRecordsBuilder(
            wrong_root,
            wrong_topic.id,
            DocumentStore(store_root),
            run_id="wrong-test-run",
        )
        wrong_builder.bind_document("doc-a", "Full text for doc-a.")
        wrong_builder.add_candidate(wrong_candidate)
        wrong_builder.publish({
            "request_schema_version": "extractive_candidate_request_v1",
            "candidate_schema_version": "extractive_candidate_nugget_v1",
            "source_file": "candidate-requests.jsonl",
            "source_sha256": sha256(b"").hexdigest(),
            "scorer": {
                "model": "fake-extractive-scorer",
                "model_revision": "test-revision",
                "backend_version": "test-backend",
                "score_representation": "raw_logits",
                "inference_dtype": "float32",
                "score_kind": "extractive_sentence_v1",
                "sentence_max_length": 512,
                "input_policy": "trec_rag_whitespace_v1",
            },
            "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
            "scoring_normalization_version": "trec_rag_whitespace_v1",
        })
        shutil.copy2(wrong_root / "records.sqlite3", records_path)
        shutil.copy2(
            wrong_root / "canonical" / "records-manifest.json",
            records_manifest_path,
        )
        selection_manifest_path = (
            topic_root / "canonical" / "selection-manifest.json"
        )
        records_manifest = json.loads(records_manifest_path.read_bytes())
        _rewrite_manifest(
            selection_manifest_path,
            records_database_sha256=sha256(records_path.read_bytes()).hexdigest(),
            candidate_semantic_sha256=records_manifest["semantic_sha256"],
        )
        nugget_manifest_path = (
            topic_root / "canonical" / "canonical-nugget-manifest.json"
        )
        _rewrite_manifest(
            nugget_manifest_path,
            selection_manifest_sha256=sha256(
                selection_manifest_path.read_bytes()
            ).hexdigest(),
        )
        for relative in (
            "records.sqlite3",
            "canonical/records-manifest.json",
            "canonical/selection-manifest.json",
            "canonical/canonical-nugget-manifest.json",
        ):
            _resign_canonical_artifact(topic_root, relative)
    elif mutation in {"missing-cas", "corrupt-cas"}:
        records_manifest = json.loads(records_manifest_path.read_bytes())
        digest = records_manifest["document_sha256s"][0]
        object_path = store_root / "sha256" / digest[:2] / f"{digest}.utf8"
        if mutation == "missing-cas":
            object_path.unlink()
        else:
            object_path.write_bytes(b"corrupt")
    elif mutation == "modified-db":
        records_path.write_bytes(records_path.read_bytes() + b"tampered")
    elif mutation == "changed-semantic-seal":
        _rewrite_manifest(records_manifest_path, semantic_sha256="0" * 64)
        selection_manifest_path = (
            topic_root / "canonical" / "selection-manifest.json"
        )
        _rewrite_manifest(selection_manifest_path, candidate_semantic_sha256="0" * 64)
        nugget_manifest_path = (
            topic_root / "canonical" / "canonical-nugget-manifest.json"
        )
        _rewrite_manifest(
            nugget_manifest_path,
            selection_manifest_sha256=sha256(
                selection_manifest_path.read_bytes()
            ).hexdigest(),
        )
        for relative in (
            "canonical/records-manifest.json",
            "canonical/selection-manifest.json",
            "canonical/canonical-nugget-manifest.json",
        ):
            _resign_canonical_artifact(topic_root, relative)
    elif mutation == "stale-wal-shm":
        (records_path.parent / "records.sqlite3-wal").write_bytes(b"stale")
        (records_path.parent / "records.sqlite3-shm").write_bytes(b"stale")
    else:
        selection_manifest_path = (
            topic_root / "canonical" / "selection-manifest.json"
        )
        _rewrite_manifest(selection_manifest_path, records_file="candidates.jsonl")
        nugget_manifest_path = (
            topic_root / "canonical" / "canonical-nugget-manifest.json"
        )
        _rewrite_manifest(
            nugget_manifest_path,
            selection_manifest_sha256=sha256(
                selection_manifest_path.read_bytes()
            ).hexdigest(),
        )
        for relative in (
            "canonical/selection-manifest.json",
            "canonical/canonical-nugget-manifest.json",
        ):
            _resign_canonical_artifact(topic_root, relative)

    with pytest.raises(
        ValueError,
        match=(
            "topic|document|source|database|semantic|WAL/SHM|"
            "selection manifest|checkpoint artifact"
        ),
    ):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_export_rejects_tampered_checkpoint_without_manifest(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    selected = (
        config.output_dir / topics[0].id / "scoring" / "selected_documents.jsonl"
    )
    selected.write_bytes(selected.read_bytes() + b" ")

    with pytest.raises(ValueError, match="checkpoint artifact hash"):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_export_rejects_sealed_duplicate_topic_document_pair(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    selected_path = topic_root / "scoring" / "selected_documents.jsonl"
    duplicate = json.loads(selected_path.read_bytes())
    duplicate["selection_rank"] = 2
    selected_path.write_bytes(selected_path.read_bytes() + _canonical_json(duplicate))
    _resign_scoring_artifact(topic_root, "scoring/selected_documents.jsonl")

    with pytest.raises(ValueError, match="duplicate.*topic-document"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_rejects_canonical_evidence_absent_from_retrieval_bundle(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-x",),
        source_commit="c" * 40,
    )

    with pytest.raises(
        ValueError,
        match="evidence document.*absent from retrieval evidence bundle",
    ):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_topic_projection_includes_canonical_evidence_beyond_internal_selection(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    retrieved_docids = tuple(f"doc-{index:03d}" for index in range(1, 102))
    supported_docids = ("doc-101", "doc-050")
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=retrieved_docids,
        supported=supported_docids,
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topic.id
    internal_selected = _read_jsonl(
        topic_root / "scoring" / "selected_documents.jsonl"
    )
    assert len(internal_selected) == 100
    assert "doc-101" not in {row["docid"] for row in internal_selected}

    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)

    projection = json.loads(
        (topic_root / "canonical" / "retrieval-projection.json").read_bytes()
    )
    projected_docids = [row["docid"] for row in projection["candidates"]]
    assert projected_docids == ["doc-050", "doc-101"]
    assert receipt.official_document_count == 2


def test_supported_document_order_keeps_source_rank_bound_to_best_score_lane() -> None:
    audit = (
        SimpleNamespace(
            passage_result=SimpleNamespace(
                documents=(
                    SimpleNamespace(
                        docid="doc-a",
                        best_passage_raw_logit=10.0,
                        source_rank=100,
                    ),
                    SimpleNamespace(
                        docid="doc-b",
                        best_passage_raw_logit=10.0,
                        source_rank=50,
                    ),
                )
            )
        ),
        SimpleNamespace(
            passage_result=SimpleNamespace(
                documents=(
                    SimpleNamespace(
                        docid="doc-a",
                        best_passage_raw_logit=1.0,
                        source_rank=1,
                    ),
                )
            )
        ),
    )

    assert retrieval_export_module._order_supported_docids(
        {"doc-a", "doc-b"}, audit
    ) == ("doc-b", "doc-a")


def test_export_rejects_existing_manifest_when_recorded_artifact_changed(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    receipt = export_retrieval_run(config, topics, code_commit="c" * 40)
    receipt.official_run.write_bytes(receipt.official_run.read_bytes() + b" ")

    with pytest.raises(ValueError, match="existing export artifact hash"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_rejects_sealed_checkpoint_schema_drift(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    scoring_manifest = topic_root / "scoring" / "complete.json"
    _rewrite_manifest(scoring_manifest, schema_version="facet_pilot_future")
    _rewrite_manifest(
        topic_root / "canonical" / "complete.json",
        scoring_manifest_sha256=sha256(scoring_manifest.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="checkpoint schema"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_rejects_mixed_checkpoint_code_identities(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    canonical_manifest = (
        config.output_dir / topics[0].id / "canonical" / "complete.json"
    )
    _rewrite_manifest(canonical_manifest, code_commit="e" * 40)

    with pytest.raises(ValueError, match="checkpoint code identities"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_accepts_compatible_source_commit_that_differs_from_export_commit(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="d" * 40,
    )

    receipt = export_retrieval_run(config, topics, code_commit="c" * 40)

    manifest = json.loads(receipt.manifest.read_bytes())
    assert manifest["export_code_commit"] == "c" * 40
    assert manifest["source_code_commits"] == ["d" * 40]
    assert manifest["source_seals"][topics[0].id]["source_code_commit"] == "d" * 40


def test_export_rejects_selected_topic_without_supported_document(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=(),
        source_commit="c" * 40,
    )

    with pytest.raises(ValueError, match="no supported document"):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_export_still_requires_cross_scores_for_nonfallback_plan(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    scores = topic_root / "scoring" / "selected_subnarrative_scores.jsonl"
    scores.write_bytes(b"")
    _resign_scoring_artifact(
        topic_root,
        "scoring/selected_subnarrative_scores.jsonl",
    )

    with pytest.raises(ValueError, match="score matrix is empty"):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


@pytest.mark.parametrize(
    ("relative_path", "message"),
    (
        ("scoring/selected_documents.jsonl", "selected document checkpoint is empty"),
        ("scoring/lane_scores.jsonl", "lane score artifact is empty"),
    ),
)
def test_scored_projection_rejects_empty_internal_scoring_artifacts(
    tmp_path: Path,
    relative_path: str,
    message: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    (topic_root / relative_path).write_bytes(b"")
    _resign_scoring_artifact(topic_root, relative_path)

    with pytest.raises(ValueError, match=message):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_scored_projection_rejects_missing_bound_winning_passage(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    lane_scores_path = topic_root / "scoring" / "lane_scores.jsonl"
    lane_scores = _read_jsonl(lane_scores_path)
    assert lane_scores[0]["winning_passages"]
    lane_scores[0]["winning_passages"] = []
    lane_scores_path.write_bytes(
        b"".join(_canonical_json(row) for row in lane_scores)
    )
    _resign_scoring_artifact(topic_root, "scoring/lane_scores.jsonl")

    with pytest.raises(ValueError, match="lane score differs from retrieval audit"):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


@pytest.mark.parametrize(
    "hash_field",
    ("bm25_query_sha256", "semantic_query_sha256"),
)
def test_export_rejects_fallback_score_from_nonoriginal_query(
    tmp_path: Path,
    hash_field: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=(),
        source_commit="c" * 40,
    )
    topic_root = config.output_dir / topic.id
    _write_fallback_decomposition_source(topic_root, topic)
    decomposition_path = topic_root / "decomposition.json"
    decomposition = json.loads(decomposition_path.read_bytes())
    decomposition.update(
        {
            "queries": [
                {
                    "topic_id": topic.id,
                    "variant_name": "original",
                    "query_text": topic.narrative,
                    "source_type": "original_topic",
                }
            ],
            "plan": None,
            "subnarratives": [],
        }
    )
    decomposition_path.write_bytes(_canonical_json(decomposition))
    _resign_retrieval_artifact(topic_root, "decomposition.json")
    audit_path = topic_root / "retrieval" / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    audit["lanes"] = audit["lanes"][:1]
    audit_path.write_bytes(_canonical_json(audit))
    _resign_retrieval_artifact(topic_root, "retrieval/audit.json")
    score_path = topic_root / "scoring" / "selected_subnarrative_scores.jsonl"
    score_path.write_bytes(b"")
    _resign_scoring_artifact(
        topic_root,
        "scoring/selected_subnarrative_scores.jsonl",
    )
    nugget_path = topic_root / "canonical" / "canonical-nuggets.jsonl"
    nugget_path.write_bytes(b"")
    _write_canonical_nugget_manifest(topic_root)
    _resign_canonical_artifact(topic_root, "canonical/canonical-nuggets.jsonl")
    _resign_canonical_artifact(
        topic_root,
        "canonical/canonical-nugget-manifest.json",
    )
    lane_score_path = topic_root / "scoring" / "lane_scores.jsonl"
    lane_scores = _read_jsonl(lane_score_path)
    lane_scores[0][hash_field] = "a" * 64
    lane_score_path.write_bytes(
        b"".join(_canonical_json(row) for row in lane_scores)
    )
    _resign_scoring_artifact(topic_root, "scoring/lane_scores.jsonl")
    selection_path = topic_root / "scoring" / "selection.json"
    selection = json.loads(selection_path.read_bytes())
    selection["lane_statuses"] = selection["lane_statuses"][:1]
    selection["trace"] = [
        row for row in selection["trace"] if row["lane_name"] == "original"
    ]
    selection["lanes"] = selection["lanes"][:1]
    selection_path.write_bytes(_canonical_json(selection))
    _resign_scoring_artifact(topic_root, "scoring/selection.json")

    with pytest.raises(ValueError, match="retrieval audit identity"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_rejects_old_export_manifest_contract(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    receipt = export_retrieval_run(config, topics, code_commit="b" * 40)
    _rewrite_manifest(receipt.manifest, schema_version="retrieval_export_manifest_v1")

    with pytest.raises(ValueError, match="unsupported existing export manifest schema"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


def test_export_replaces_valid_manifest_across_exporter_revisions(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    export_retrieval_run(config, topics, code_commit="b" * 40)

    receipt = export_retrieval_run(config, topics, code_commit="c" * 40)

    manifest = json.loads(receipt.manifest.read_bytes())
    assert manifest["export_code_commit"] == "c" * 40
    assert manifest["source_code_commits"] == ["a" * 40]


def test_export_replaces_valid_export_for_changed_topic_selection(
    tmp_path: Path,
) -> None:
    config, original_topics = _config_and_topics(tmp_path)
    topics = (
        original_topics[0],
        Topic("rag2026-1", "", "Explain a second demonstrated topic."),
    )
    config.topics_path.write_text(
        "".join(f"{topic.id}\t{topic.narrative}\n" for topic in topics),
        encoding="utf-8",
    )
    for topic, docid in zip(topics, ("doc-a", "doc-b"), strict=True):
        _write_sealed_topic(
            config.output_dir,
            topic,
            selected=(docid,),
            supported=(docid,),
            source_commit="a" * 40,
            official_topics=topics,
        )

    first = export_retrieval_run(config, topics[:1], code_commit="b" * 40)
    first_artifacts = _root_artifact_bodies(first)

    replaced = export_retrieval_run(config, topics, code_commit="b" * 40)

    manifest = json.loads(replaced.manifest.read_bytes())
    assert manifest["selected_topic_ids"] == [topic.id for topic in topics]
    assert set(manifest["topic_depths"]) == {topic.id for topic in topics}
    assert set(manifest["source_seals"]) == {topic.id for topic in topics}
    assert all(
        body != first_artifacts[name]
        for name, body in _root_artifact_bodies(replaced).items()
    )
    assert read_retrieval_export_receipt(config, topics) == replaced
    with pytest.raises(ValueError, match="existing export manifest identity changed"):
        read_retrieval_export_receipt(config, topics[:1])


def test_export_selection_replacement_still_rejects_run_id_mismatch_before_write(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    receipt = export_retrieval_run(config, topics, code_commit="b" * 40)
    original_artifacts = _root_artifact_bodies(receipt)
    _rewrite_manifest(receipt.manifest, run_id="different-run")

    with pytest.raises(ValueError, match="existing export manifest identity changed"):
        export_retrieval_run(config, topics, code_commit="b" * 40)

    assert json.loads(receipt.manifest.read_bytes())["run_id"] == "different-run"
    assert _root_artifact_bodies(receipt) == original_artifacts


def test_export_rejects_tampered_existing_topic_order_before_replacement(
    tmp_path: Path,
) -> None:
    config, original_topics = _config_and_topics(tmp_path)
    topics = (
        original_topics[0],
        Topic("rag2026-1", "", "Explain a second demonstrated topic."),
    )
    config.topics_path.write_text(
        "".join(f"{topic.id}\t{topic.narrative}\n" for topic in topics),
        encoding="utf-8",
    )
    for topic, docid in zip(topics, ("doc-a", "doc-b"), strict=True):
        _write_sealed_topic(
            config.output_dir,
            topic,
            selected=(docid,),
            supported=(docid,),
            source_commit="a" * 40,
            official_topics=topics,
        )
    receipt = export_retrieval_run(config, topics, code_commit="b" * 40)
    original_artifacts = _root_artifact_bodies(receipt)
    manifest = json.loads(receipt.manifest.read_bytes())
    manifest["selected_topic_ids"].reverse()
    receipt.manifest.write_bytes(_canonical_json(manifest))
    tampered_manifest = receipt.manifest.read_bytes()

    with pytest.raises(ValueError, match="existing export manifest identity changed"):
        export_retrieval_run(config, topics[:1], code_commit="b" * 40)

    assert receipt.manifest.read_bytes() == tampered_manifest
    assert _root_artifact_bodies(receipt) == original_artifacts


def test_export_failure_mid_replacement_unseals_and_next_rerun_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    first = export_retrieval_run(config, topics, code_commit="b" * 40)
    real_atomic_write = retrieval_export._atomic_write
    writes = 0

    def fail_second_artifact(path: Path, body: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated export publication failure")
        real_atomic_write(path, body)

    monkeypatch.setattr(retrieval_export, "_atomic_write", fail_second_artifact)
    with pytest.raises(OSError, match="simulated export publication failure"):
        export_retrieval_run(config, topics, code_commit="c" * 40)
    assert not first.manifest.exists()

    monkeypatch.setattr(retrieval_export, "_atomic_write", real_atomic_write)
    recovered = export_retrieval_run(config, topics, code_commit="c" * 40)
    assert recovered.manifest.is_file()
    assert json.loads(recovered.manifest.read_bytes())["export_code_commit"] == "c" * 40


@pytest.mark.parametrize(
    "changed_config",
    (
        lambda config: replace(
            config,
            retrieval=replace(config.retrieval, index="changed-index"),
        ),
        lambda config: replace(
            config,
            retrieval=replace(config.retrieval, documents_per_query=999),
        ),
        lambda config: replace(
            config,
            passage=replace(config.passage, model="changed-model"),
        ),
        lambda config: replace(
            config,
            passage=replace(config.passage, passages_per_query=99),
        ),
        lambda config: replace(
            config,
            passage=replace(config.passage, chunk_max_characters=99),
        ),
        lambda config: replace(
            config,
            passage=replace(config.passage, chunk_overlap_characters=99),
        ),
        lambda config: replace(
            config,
            nuggets=replace(config.nuggets, evidence_budget_per_subnarrative=39),
        ),
        lambda config: replace(
            config,
            nuggets=replace(config.nuggets, maximum_claims_per_subnarrative=19),
        ),
        lambda config: replace(
            config,
            nuggets=replace(
                config.nuggets,
                maximum_supporting_documents_per_claim=2,
            ),
        ),
    ),
    ids=(
        "retrieval-index",
        "retrieval-depth",
        "scorer-model",
        "rerank-depth",
        "selection-depth",
        "selection-policy",
        "evidence-budget",
        "claim-cap",
        "support-document-cap",
    ),
)
def test_export_rejects_checkpoint_from_changed_config(
    tmp_path: Path,
    changed_config,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )

    with pytest.raises(ValueError, match="checkpoint.*configuration"):
        export_retrieval_run(
            changed_config(config),
            topics,
            code_commit="b" * 40,
        )


def test_export_rejects_checkpoint_with_stale_narrative_identity(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    scoring_path = topic_root / "scoring" / "complete.json"
    canonical_path = topic_root / "canonical" / "complete.json"
    _rewrite_manifest(scoring_path, narrative_sha256="0" * 64)
    _rewrite_manifest(
        canonical_path,
        narrative_sha256="0" * 64,
        scoring_manifest_sha256=sha256(scoring_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="narrative"):
        export_retrieval_run(config, topics, code_commit="b" * 40)


@pytest.mark.parametrize(
    "missing",
    (
        "name",
        "index",
        "index_url",
        "hits",
        "corpus_epoch",
        "retrieval_cache_schema",
        "parser_version",
        "extractor_version",
        "field_path",
        "scoring_normalizer_version",
    ),
)
def test_export_requires_complete_pyserini_retriever_identity(
    tmp_path: Path,
    missing: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    scoring_path = topic_root / "scoring" / "complete.json"
    scoring = json.loads(scoring_path.read_bytes())
    scoring["retriever"].pop(missing)

    with pytest.raises(ValueError, match="retriever.*identity"):
        retrieval_export_module._validate_scoring_manifest(
            scoring,
            config,
            topics[0],
        )


def test_export_compares_current_complete_pyserini_retriever_identity(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    scoring = json.loads(
        (config.output_dir / topics[0].id / "scoring" / "complete.json").read_bytes()
    )
    expected = dict(_FixtureRetriever.identity)
    scoring["retriever"]["parser_version"] = "changed-parser"

    with pytest.raises(ValueError, match="retriever.*identity"):
        retrieval_export_module._validate_scoring_manifest(
            scoring,
            config,
            topics[0],
            expected_retriever_identity=expected,
        )


def test_export_retrieval_source_chain_requires_complete_and_current_identity(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    stale = {
        "name": "climbmix_bm25",
        "type": "pyserini_remote",
        "index": "climbmix-400b",
        "index_url": "https://retrieval.example.invalid/search",
        "hits": 1000,
    }
    retrieval_path = topic_root / "retrieval" / "complete.json"
    retrieval = json.loads(retrieval_path.read_bytes())
    retrieval["retriever"] = stale
    retrieval_path.write_bytes(_canonical_json(retrieval))
    _resign_retrieval_manifest_into_scoring(topic_root)
    scoring_path = topic_root / "scoring" / "complete.json"
    scoring = json.loads(scoring_path.read_bytes())
    scoring["retriever"] = stale
    scoring_path.write_bytes(_canonical_json(scoring))

    with pytest.raises(ValueError, match="retriever.*identity"):
        retrieval_export_module._validate_retrieval_source_chain(
            topic_root,
            scoring,
            config,
            topics[0],
            "a" * 40,
            expected_retriever_identity=dict(_FixtureRetriever.identity),
        )


def test_export_rejects_disagreed_decomposition_and_official_source_seals(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    canonical_path = config.output_dir / topics[0].id / "canonical" / "complete.json"
    _rewrite_manifest(canonical_path, decomposition_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="configuration"):
        export_retrieval_run(config, topics, code_commit="b" * 40)

    _rewrite_manifest(
        canonical_path,
        decomposition_source_sha256="d" * 64,
        official_topics_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="configuration"):
        export_retrieval_run(config, topics, code_commit="b" * 40)


@pytest.mark.parametrize(
    "mutation",
    (
        "retrieval-manifest-bytes",
        "retrieval-source-commit",
        "retrieval-index",
        "decomposition-source",
        "handoff-manifest-bytes",
        "score-policy",
    ),
)
def test_export_validates_actual_sealed_source_chain(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    retrieval_manifest = topic_root / "retrieval" / "complete.json"
    if mutation == "retrieval-manifest-bytes":
        retrieval_manifest.write_bytes(retrieval_manifest.read_bytes() + b" ")
    elif mutation in {"retrieval-source-commit", "retrieval-index"}:
        retrieval = json.loads(retrieval_manifest.read_bytes())
        if mutation == "retrieval-source-commit":
            retrieval["code_commit"] = "b" * 40
        else:
            retrieval["retriever"]["index"] = "other-index"
        retrieval_manifest.write_bytes(_canonical_json(retrieval))
        _resign_retrieval_manifest_into_scoring(topic_root)
    elif mutation == "decomposition-source":
        decomposition = topic_root / "decomposition.json"
        record = json.loads(decomposition.read_bytes())
        record["source_sha256"] = "9" * 64
        decomposition.write_bytes(_canonical_json(record))
        _resign_retrieval_artifact(topic_root, "decomposition.json")
    elif mutation == "handoff-manifest-bytes":
        handoff = topic_root / "canonical" / "handoff" / "handoff-manifest.json"
        handoff.write_bytes(_canonical_json({"changed": True}))
        _resign_canonical_artifact(
            topic_root, "canonical/handoff/handoff-manifest.json"
        )
    else:
        scoring_manifest = topic_root / "scoring" / "complete.json"
        scoring = json.loads(scoring_manifest.read_bytes())
        scoring["score_policy"] = {"future_policy": True}
        scoring_manifest.write_bytes(_canonical_json(scoring))
        _resign_scoring_manifest(topic_root)

    with pytest.raises(ValueError):
        export_retrieval_run(config, topics, code_commit="c" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_export_rejects_unrecognized_provenance_fields(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    selected_path = topic_root / "scoring" / "selected_documents.jsonl"
    row = json.loads(selected_path.read_bytes())
    row["unreviewed_passthrough"] = "must not escape"
    selected_path.write_bytes(_canonical_json(row))
    _resign_scoring_artifact(topic_root, "scoring/selected_documents.jsonl")

    with pytest.raises(ValueError, match="selected document schema"):
        export_retrieval_run(config, topics, code_commit="b" * 40)


def test_export_cross_checks_membership_bm25_against_lane_scores(tmp_path: Path) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    selection_path = topic_root / "scoring" / "selection.json"
    selection = json.loads(selection_path.read_bytes())
    selection["memberships"][0]["lanes"][0]["bm25_score"] = -123.0
    selection_path.write_bytes(_canonical_json(selection))
    _resign_scoring_artifact(topic_root, "scoring/selection.json")

    with pytest.raises(ValueError, match="lane score"):
        export_retrieval_run(config, topics, code_commit="b" * 40)


@pytest.mark.parametrize("mutation", ("omitted", "extra"))
def test_export_requires_exact_membership_lane_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    lane_scores_path = topic_root / "scoring" / "lane_scores.jsonl"
    selection_path = topic_root / "scoring" / "selection.json"
    if mutation == "omitted":
        lane = _read_jsonl(lane_scores_path)[0]
        lane["lane_name"] = "subnarrative:subnarrative-1"
        lane["bm25_query_sha256"] = sha256(b"alternate").hexdigest()
        lane["semantic_query_sha256"] = sha256(b"alternate").hexdigest()
        lane_scores_path.write_bytes(
            lane_scores_path.read_bytes() + _canonical_json(lane)
        )
        _resign_scoring_artifact(topic_root, "scoring/lane_scores.jsonl")
    else:
        selection = json.loads(selection_path.read_bytes())
        extra = dict(selection["memberships"][0]["lanes"][0])
        extra["lane_name"] = "unsealed-extra"
        selection["memberships"][0]["lanes"].append(extra)
        selection_path.write_bytes(_canonical_json(selection))
        _resign_scoring_artifact(topic_root, "scoring/selection.json")

    with pytest.raises(ValueError, match="membership.*lane|lane score"):
        export_retrieval_run(config, topics, code_commit="b" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate-subnarrative",
        "missing-subnarrative",
        "wrong-budget",
        "wrong-state",
        "wrong-request",
        "request-count",
    ),
)
def test_export_validates_canonical_result_completeness(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    _write_sealed_topic(
        config.output_dir,
        topics[0],
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topics[0].id
    nuggets_path = topic_root / "canonical" / "canonical-nuggets.jsonl"
    rows = _read_jsonl(nuggets_path)
    sealed_requests = [str(row["request_sha256"]) for row in rows]
    if mutation == "duplicate-subnarrative":
        rows[1]["subnarrative_id"] = rows[0]["subnarrative_id"]
    elif mutation == "missing-subnarrative":
        rows.pop()
    elif mutation == "wrong-budget":
        for row in rows:
            row["selected_budget"] = 39
    elif mutation == "wrong-state":
        rows[0]["state"] = "future-state"
    elif mutation == "wrong-request":
        rows[0]["request_sha256"] = "a" * 64

    nuggets_path.write_bytes(b"".join(_canonical_json(row) for row in rows))
    request_sha256s = sealed_requests if mutation == "wrong-request" else None
    _write_canonical_nugget_manifest(
        topic_root,
        request_sha256s=(
            sealed_requests[:1]
            if mutation == "request-count"
            else request_sha256s
        ),
    )
    _resign_canonical_artifact(topic_root, "canonical/canonical-nuggets.jsonl")
    _resign_canonical_artifact(
        topic_root, "canonical/canonical-nugget-manifest.json"
    )

    with pytest.raises(ValueError, match="canonical nugget"):
        export_retrieval_run(config, topics, code_commit="b" * 40)

    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_retrieval_stage_emits_validated_bundles_for_two_topics(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    second = Topic("rag2026-1", "", "Explain the second demonstrated topic.")

    for topic in (*topics, second):
        _write_sealed_topic(
            config.output_dir,
            topic,
            selected=("doc-a",),
            supported=("doc-a",),
            source_commit="a" * 40,
        )

    bundles = [
        json.loads(
            (
                config.output_dir
                / topic.id
                / "retrieval"
                / "evidence-bundle.json"
            ).read_bytes()
        )
        for topic in (*topics, second)
    ]

    assert [bundle["topic_id"] for bundle in bundles] == [
        "rag2026-0",
        "rag2026-1",
    ]
    assert all(
        bundle["schema_version"] == "facet_passage_evidence_v2"
        for bundle in bundles
    )
    assert all(
        sum(len(lane["documents"]) for lane in bundle["lanes"]) == 1
        for bundle in bundles
    )


def test_task55_projection_receipt_api_is_exposed() -> None:
    import trec_rag.retrieval_export as retrieval_export

    assert callable(getattr(retrieval_export, "build_topic_projection", None))
    assert hasattr(retrieval_export, "TopicProjectionReceipt")


def test_task55_export_api_requires_projection_receipts() -> None:
    import inspect
    import trec_rag.retrieval_export as retrieval_export

    parameters = inspect.signature(retrieval_export.export_retrieval_run).parameters
    assert "projection_receipts" in parameters
    assert parameters["projection_receipts"].default is inspect.Parameter.empty


def test_provisional_projection_builds_before_canonical_complete_exists(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    complete_path = config.output_dir / topic.id / "canonical" / "complete.json"
    provisional_manifest_bytes = complete_path.read_bytes()
    complete_path.unlink()

    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(
            config,
            topic,
            records,
            canonical_manifest_bytes=provisional_manifest_bytes,
        )

    assert receipt.topic_id == topic.id
    assert complete_path.is_file()
    assert set(
        item["relative_path"]
        for item in json.loads(complete_path.read_bytes())["artifacts"]
    ) == {
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "records.sqlite3",
        "canonical/records-manifest.json",
        "canonical/subnarrative-selections.jsonl",
        "canonical/selection-manifest.json",
        "canonical/canonical-nuggets.jsonl",
        "canonical/canonical-nugget-manifest.json",
        "canonical/retrieval-projection.json",
        "canonical/retrieval-projection-manifest.json",
        "canonical/generation-projection.json",
        "canonical/generation-projection-manifest.json",
    }


def test_projection_publication_orders_completion_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    complete_path = config.output_dir / topic.id / "canonical" / "complete.json"
    provisional_manifest_bytes = complete_path.read_bytes()
    complete_path.unlink()
    events: list[str] = []
    publish = retrieval_export._publish_create_only

    def record_publish(path: Path, body: bytes) -> None:
        events.append(path.name)
        publish(path, body)

    monkeypatch.setattr(retrieval_export, "_publish_create_only", record_publish)
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        build_topic_projection(
            config,
            topic,
            records,
            canonical_manifest_bytes=provisional_manifest_bytes,
        )

    assert events == [
        "retrieval-projection.json",
        "generation-projection.json",
        "retrieval-projection-manifest.json",
        "generation-projection-manifest.json",
        "complete.json",
    ]


@pytest.mark.parametrize("failure_boundary", (1, 2, 3, 4))
def test_projection_boundary_failure_leaves_new_completion_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: int,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    complete_path = config.output_dir / topic.id / "canonical" / "complete.json"
    provisional_manifest_bytes = complete_path.read_bytes()
    complete_path.unlink()
    publish = retrieval_export._publish_create_only
    call_count = 0

    def fail_before_boundary(path: Path, body: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == failure_boundary:
            raise RuntimeError("injected projection publication failure")
        publish(path, body)

    monkeypatch.setattr(
        retrieval_export, "_publish_create_only", fail_before_boundary
    )
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        with pytest.raises(RuntimeError, match="injected"):
            build_topic_projection(
                config,
                topic,
                records,
                canonical_manifest_bytes=provisional_manifest_bytes,
            )

    assert not complete_path.exists()


def test_projection_receipt_reader_roundtrips_and_checkpoint_validation_preserves_order(
    tmp_path: Path,
) -> None:
    config, _ = _config_and_topics(tmp_path)
    topics = (
        Topic("topic-2", "", "Explain topic two."),
        Topic("topic-1", "", "Explain topic one."),
    )
    config.topics_path.write_text(
        "".join(f"{topic.id}\t{topic.narrative}\n" for topic in topics),
        encoding="utf-8",
    )
    receipts: list[TopicProjectionReceipt] = []
    for topic, docid in zip(topics, ("doc-2", "doc-1"), strict=True):
        _write_sealed_topic(
            config.output_dir,
            topic,
            selected=(docid,),
            supported=(docid,),
            source_commit="a" * 40,
            official_topics=topics,
        )
        with TopicRecords.open(
            config.output_dir / topic.id / "records.sqlite3",
            config.output_dir / topic.id / "canonical" / "records-manifest.json",
            topic.id,
            DocumentStore(document_store_dir(config.root_dir)),
        ) as records:
            receipts.append(build_topic_projection(config, topic, records))

    assert read_topic_projection_receipt(config, topics[0]) == receipts[0]
    assert validate_retrieval_topic_checkpoints(
        config,
        tuple(reversed(topics)),
        expected_retriever_identity=dict(_FixtureRetriever.identity),
        expected_decomposition_producer_sha256={
            topic.id: _FIXTURE_DECOMPOSITION_PRODUCER_SHA256 for topic in topics
        },
        max_workers=2,
    ) == (receipts[1], receipts[0])


def test_projection_receipt_reader_rejects_topic_records_from_another_run(
    tmp_path: Path,
) -> None:
    old_config, topic, _receipt, _projection_path, _manifest_path = (
        _projection_fixture(tmp_path)
    )
    new_config = replace(
        old_config,
        experiment=ExperimentSettings("different-run"),
    )
    shutil.copytree(
        old_config.output_dir / topic.id,
        new_config.output_dir / topic.id,
    )

    with pytest.raises(ValueError, match="topic records manifest run identity changed"):
        read_topic_projection_receipt(new_config, topic)


def test_base_only_checkpoint_is_not_considered_fully_resumed(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )

    with pytest.raises(ValueError, match="legacy canonical checkpoint is incompatible"):
        validate_retrieval_topic_checkpoints(
            config,
            (topic,),
            expected_retriever_identity=dict(_FixtureRetriever.identity),
            expected_decomposition_producer_sha256={
                topic.id: _FIXTURE_DECOMPOSITION_PRODUCER_SHA256
            },
        )


def test_fresh_projection_publication_is_idempotent_and_byte_stable(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    complete_path = config.output_dir / topic.id / "canonical" / "complete.json"
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        first = build_topic_projection(config, topic, records)
        expanded_bytes = complete_path.read_bytes()
        second = build_topic_projection(config, topic, records)

    assert second == first
    assert complete_path.read_bytes() == expanded_bytes
    artifacts = json.loads(expanded_bytes)["artifacts"]
    assert len(artifacts) == 13
    assert {item["relative_path"] for item in artifacts} == {
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "records.sqlite3",
        "canonical/records-manifest.json",
        "canonical/subnarrative-selections.jsonl",
        "canonical/selection-manifest.json",
        "canonical/canonical-nuggets.jsonl",
        "canonical/canonical-nugget-manifest.json",
        "canonical/retrieval-projection.json",
        "canonical/retrieval-projection-manifest.json",
        "canonical/generation-projection.json",
        "canonical/generation-projection-manifest.json",
    }
    for item in artifacts:
        body = (complete_path.parent.parent / item["relative_path"]).read_bytes()
        assert item == {
            "relative_path": item["relative_path"],
            "bytes": len(body),
            "sha256": sha256(body).hexdigest(),
        }


def test_projection_publication_requires_current_retriever_identity_before_publish(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    canonical_root = config.output_dir / topic.id / "canonical"
    complete_path = canonical_root / "complete.json"
    old_complete_bytes = complete_path.read_bytes()
    changed_identity = dict(_FixtureRetriever.identity)
    changed_identity["corpus_epoch"] = "different-current-epoch"

    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        canonical_root / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        with pytest.raises(TypeError, match="expected_retriever_identity"):
            _build_topic_projection(
                config,
                topic,
                records,
                decomposition_producer_sha256=(
                    _FIXTURE_DECOMPOSITION_PRODUCER_SHA256
                ),
            )
        with pytest.raises(TypeError, match="expected_retriever_identity"):
            _build_topic_projection(
                config,
                topic,
                records,
                expected_retriever_identity=None,  # type: ignore[arg-type]
                decomposition_producer_sha256=(
                    _FIXTURE_DECOMPOSITION_PRODUCER_SHA256
                ),
            )
        with pytest.raises(ValueError, match="retriever identity changed"):
            build_topic_projection(
                config,
                topic,
                records,
                expected_retriever_identity=changed_identity,
            )

    assert complete_path.read_bytes() == old_complete_bytes
    assert not (canonical_root / "retrieval-projection.json").exists()
    assert not (canonical_root / "retrieval-projection-manifest.json").exists()


def test_checkpoint_validation_rejects_null_expected_retriever_identity_before_read(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    canonical_root = config.output_dir / topic.id / "canonical"
    complete_path = canonical_root / "complete.json"
    old_complete_bytes = complete_path.read_bytes()

    with pytest.raises(TypeError, match="expected_retriever_identity"):
        validate_retrieval_topic_checkpoints(
            config,
            (topic,),
            expected_retriever_identity=None,  # type: ignore[arg-type]
            expected_decomposition_producer_sha256={
                topic.id: _FIXTURE_DECOMPOSITION_PRODUCER_SHA256
            },
        )

    assert complete_path.read_bytes() == old_complete_bytes
    assert not (canonical_root / "retrieval-projection.json").exists()
    assert not (canonical_root / "retrieval-projection-manifest.json").exists()


@pytest.mark.parametrize("mutation", ("missing-projection", "missing-manifest", "base-only"))
def test_projection_receipt_reader_rejects_partial_or_tampered_checkpoint(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topic, _receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    complete_path = projection_path.parent / "complete.json"
    if mutation == "missing-projection":
        projection_path.unlink()
    elif mutation == "missing-manifest":
        manifest_path.unlink()
    else:
        complete = json.loads(complete_path.read_bytes())
        complete["artifacts"] = [
            item
            for item in complete["artifacts"]
            if not item["relative_path"].startswith("canonical/retrieval-projection")
        ]
        complete_path.write_bytes(_canonical_json(complete))

    with pytest.raises(ValueError):
        read_topic_projection_receipt(config, topic)


def test_build_projection_trusts_supplied_open_records_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        def fail_validation(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("projection must not rerun deep source validation")

        def fail_open(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("projection must not reopen TopicRecords")

        monkeypatch.setattr(records, "validate_all_sources", fail_validation)
        monkeypatch.setattr(TopicRecords, "open", fail_open)
        receipt = build_topic_projection(config, topic, records)

    assert receipt.topic_id == topic.id
    assert receipt.official_document_count == 1


def test_build_projection_does_not_read_records_source_paths_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    records_path = topic_root / "records.sqlite3"
    records_manifest_path = topic_root / "canonical" / "records-manifest.json"
    with TopicRecords.open(
        records_path,
        records_manifest_path,
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        real_open = Path.open
        real_read_bytes = Path.read_bytes
        real_resolve = Path.resolve
        real_stat = Path.stat
        real_streamed_file_identity = retrieval_export._streamed_file_identity

        def reject_source_path(path: Path) -> None:
            if path in (records_path, records_manifest_path):
                raise AssertionError(f"projection touched TopicRecords source path: {path}")

        def guarded_open(path: Path, *args: object, **kwargs: object):
            reject_source_path(path)
            return real_open(path, *args, **kwargs)

        def guarded_read_bytes(path: Path) -> bytes:
            reject_source_path(path)
            return real_read_bytes(path)

        def guarded_resolve(path: Path, *args: object, **kwargs: object) -> Path:
            reject_source_path(path)
            return real_resolve(path, *args, **kwargs)

        def guarded_stat(path: Path, *args: object, **kwargs: object):
            reject_source_path(path)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", guarded_open)
        monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
        monkeypatch.setattr(Path, "resolve", guarded_resolve)
        monkeypatch.setattr(Path, "stat", guarded_stat)
        monkeypatch.setattr(
            retrieval_export,
            "_streamed_file_identity",
            lambda path: (_ for _ in ()).throw(
                AssertionError(f"projection hashed a source path: {path}")
            )
            if path in (records_path, records_manifest_path)
            else real_streamed_file_identity(path),
        )

        receipt = build_topic_projection(config, topic, records)

    assert receipt.records_receipt.topic_id == topic.id


def test_build_projection_uses_pinned_records_receipt_after_manifest_replacement(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    records_manifest_path = topic_root / "canonical" / "records-manifest.json"
    replacement = tmp_path / "replacement-records-manifest.json"
    replacement.write_bytes(b"{}\n")
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        records_manifest_path,
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        authoritative = TopicRecords.assert_current(records)
        os.replace(replacement, records_manifest_path)
        projection_receipt = build_topic_projection(config, topic, records)

    assert projection_receipt.records_receipt.schema_version == authoritative.schema_version
    assert projection_receipt.records_receipt.manifest_sha256 == authoritative.manifest_sha256
    assert projection_receipt.records_receipt.manifest_bytes == authoritative.manifest_bytes


def test_build_projection_rejects_forged_copied_records_handle(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        forged = object.__new__(TopicRecords)
        forged.__dict__.update(vars(records))

        with pytest.raises(TopicRecordsIntegrityError, match="registered|handle"):
            build_topic_projection(config, topic, forged)


def test_build_projection_loads_candidates_with_nonempty_finite_required_key_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        captured: list[object] = []
        real_load_candidates = records.load_candidates

        def capture_required_keys(required_keys: object):
            captured.append(required_keys)
            return real_load_candidates(required_keys)  # type: ignore[arg-type]

        monkeypatch.setattr(records, "load_candidates", capture_required_keys)
        build_topic_projection(config, topic, records)

    assert len(captured) == 1
    required_keys = captured[0]
    assert isinstance(required_keys, (set, frozenset))
    assert required_keys
    assert all(
        isinstance(key, tuple)
        and len(key) == 2
        and all(isinstance(part, str) and part for part in key)
        for key in required_keys
    )


def test_projection_records_receipt_copies_authoritative_receipt_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        actual = TopicRecords.assert_current(records)
        authoritative = replace(actual, schema_version="authoritative-topic-records-v2")

        def return_authoritative(_records: object):
            return authoritative

        monkeypatch.setattr(
            TopicRecords,
            "assert_current",
            staticmethod(return_authoritative),
        )
        receipt = build_topic_projection(config, topic, records)

    assert receipt.records_receipt.schema_version == authoritative.schema_version
    assert receipt.records_receipt.topic_id == authoritative.topic_id
    assert receipt.records_receipt.database_sha256 == authoritative.database_sha256
    assert receipt.records_receipt.database_bytes == authoritative.database_bytes
    assert receipt.records_receipt.manifest_sha256 == authoritative.manifest_sha256
    assert receipt.records_receipt.manifest_bytes == authoritative.manifest_bytes
    assert receipt.records_receipt.semantic_sha256 == authoritative.semantic_sha256
    assert receipt.records_receipt.document_sha256s == authoritative.document_sha256s
    assert dict(receipt.records_receipt.row_counts) == dict(authoritative.row_counts)


def test_receipt_only_export_does_not_reopen_or_rehash_topic_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        projection_receipt = build_topic_projection(config, topic, records)

    def fail_reopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("global export must not reopen TopicRecords")

    def fail_hash(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise AssertionError("global export must not rehash TopicRecords or source files")

    monkeypatch.setattr(TopicRecords, "open", fail_reopen)
    monkeypatch.setattr(retrieval_export, "_streamed_file_identity", fail_hash)
    result = _export_retrieval_run(
        config,
        topics,
        (projection_receipt,),
        code_commit="b" * 40,
    )

    assert result.official_run.read_bytes() == b"rag2026-0 Q0 doc-a 1 1 demo\n"


def test_projection_receipt_is_frozen_path_free_and_strictly_serializable(
    tmp_path: Path,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)

    with pytest.raises(FrozenInstanceError):
        receipt.topic_id = "changed"  # type: ignore[misc]
    restored = retrieval_export.TopicProjectionReceipt.from_dict(receipt.to_dict())
    assert restored == receipt
    assert dict(receipt.source_seals)["config_sha256"] == sha256(
        f"fixture-config:{config.run_id}".encode("utf-8")
    ).hexdigest()
    json.dumps(receipt.to_dict())
    pickle.loads(pickle.dumps(receipt))

    def assert_path_free(value: object) -> None:
        assert not isinstance(value, (Path, Topic, EvidenceBundle, TopicRecords))
        if hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                assert_path_free(getattr(value, field_name))
        elif isinstance(value, tuple):
            for item in value:
                assert_path_free(item)

    assert_path_free(receipt)
    projection = topic_root / "canonical" / "retrieval-projection.json"
    projection_rows = projection.read_bytes().splitlines()
    assert projection.read_bytes().endswith(b"\n")
    assert len(projection_rows) == 1
    manifest = json.loads(
        (topic_root / "canonical" / "retrieval-projection-manifest.json").read_bytes()
    )
    assert set(manifest) == retrieval_export._PROJECTION_MANIFEST_FIELDS


def test_projection_receipt_rejects_a_disagreed_topic_snapshot_source_seal(
    tmp_path: Path,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, _projection_path, _manifest_path = _projection_fixture(
        tmp_path
    )
    del config, topic
    payload = receipt.to_dict()
    payload["source_seals"]["topic_snapshot_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="snapshot seal"):
        retrieval_export.TopicProjectionReceipt.from_dict(payload)


def test_global_export_restores_supplied_official_topic_order(
    tmp_path: Path,
) -> None:
    config, _ = _config_and_topics(tmp_path)
    topics = (
        Topic("topic-2", "", "Explain topic two."),
        Topic("topic-1", "", "Explain topic one."),
    )
    config.topics_path.write_text(
        "".join(f"{topic.id}\t{topic.narrative}\n" for topic in topics),
        encoding="utf-8",
    )
    receipts = []
    for topic, docid in zip(topics, ("doc-2", "doc-1"), strict=True):
        _write_sealed_topic(
            config.output_dir,
            topic,
            selected=(docid,),
            supported=(docid,),
            source_commit="a" * 40,
            official_topics=topics,
        )
        topic_root = config.output_dir / topic.id
        with TopicRecords.open(
            topic_root / "records.sqlite3",
            topic_root / "canonical" / "records-manifest.json",
            topic.id,
            DocumentStore(document_store_dir(config.root_dir)),
        ) as records:
            receipts.append(build_topic_projection(config, topic, records))

    result = _export_retrieval_run(
        config,
        topics,
        tuple(reversed(receipts)),
        code_commit="b" * 40,
    )
    assert result.official_run.read_bytes() == (
        b"topic-2 Q0 doc-2 1 1 demo\n"
        b"topic-1 Q0 doc-1 1 1 demo\n"
    )
    with zipfile.ZipFile(result.with_text_archive) as archive:
        rows = archive.read("retrieval_with_text.jsonl").splitlines()
    assert [json.loads(row)["query"]["qid"] for row in rows] == [
        "topic-2",
        "topic-1",
    ]


@pytest.mark.parametrize("failure", ("missing", "duplicate", "wrong", "stale"))
def test_global_export_rejects_projection_receipt_coverage_before_publication(
    tmp_path: Path,
    failure: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)
    if failure == "missing":
        receipts = ()
    elif failure == "duplicate":
        receipts = (receipt, receipt)
    elif failure == "wrong":
        receipts = (replace(receipt, topic_id="wrong-topic"),)
    else:
        receipts = (replace(receipt, projection_sha256="0" * 64),)

    with pytest.raises((TypeError, ValueError)):
        _export_retrieval_run(config, topics, receipts, code_commit="b" * 40)
    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


@pytest.mark.parametrize("mutation", ("projection", "manifest", "source-seal"))
def test_global_export_rejects_altered_projection_receipts_before_publication(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)
    if mutation == "projection":
        projection_path = topic_root / "canonical" / "retrieval-projection.json"
        projection_path.write_bytes(projection_path.read_bytes()[:-1] + b" ")
    else:
        manifest_path = (
            topic_root / "canonical" / "retrieval-projection-manifest.json"
        )
        manifest = json.loads(manifest_path.read_bytes())
        if mutation == "source-seal":
            manifest["source_seals"]["scoring_manifest_sha256"] = "0" * 64
        else:
            manifest["official_document_count"] = 2
        manifest_path.write_bytes(_canonical_json(manifest))

    with pytest.raises(ValueError):
        _export_retrieval_run(config, topics, (receipt,), code_commit="b" * 40)
    assert not (config.output_dir / "retrieval_export_manifest.json").exists()


def test_topic_projection_publication_is_create_only_and_rejects_partial_state(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    projection_path = topic_root / "canonical" / "retrieval-projection.json"
    manifest_path = topic_root / "canonical" / "retrieval-projection-manifest.json"
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        first = build_topic_projection(config, topic, records)
        projection_bytes = projection_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        second = build_topic_projection(config, topic, records)
    assert second == first
    assert projection_path.read_bytes() == projection_bytes
    assert manifest_path.read_bytes() == manifest_bytes

    projection_path.unlink()
    with pytest.raises(ValueError, match="partial|conflicting|incomplete"):
        with TopicRecords.open(
            topic_root / "records.sqlite3",
            topic_root / "canonical" / "records-manifest.json",
            topic.id,
            DocumentStore(document_store_dir(config.root_dir)),
        ) as records:
            build_topic_projection(config, topic, records)


def _projection_fixture(
    tmp_path: Path,
) -> tuple[FacetPilotConfig, Topic, TopicProjectionReceipt, Path, Path]:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)
    return (
        config,
        topic,
        receipt,
        topic_root / "canonical" / "retrieval-projection.json",
        topic_root / "canonical" / "retrieval-projection-manifest.json",
    )


def _generation_projection_paths(projection_path: Path) -> tuple[Path, Path]:
    return (
        projection_path.with_name("generation-projection.json"),
        projection_path.with_name("generation-projection-manifest.json"),
    )


def _reseal_projection_receipt(
    receipt: TopicProjectionReceipt,
    manifest_path: Path,
    projection_path: Path,
    projection_bytes: bytes,
) -> TopicProjectionReceipt:
    projection_path.write_bytes(projection_bytes)
    manifest = json.loads(manifest_path.read_bytes())
    manifest.update(
        {
            "projection_sha256": sha256(projection_bytes).hexdigest(),
            "projection_bytes": len(projection_bytes),
        }
    )
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return replace(
        receipt,
        projection_sha256=sha256(projection_bytes).hexdigest(),
        projection_bytes=len(projection_bytes),
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        manifest_bytes=len(manifest_bytes),
    )


def test_export_does_not_reopen_projection_path_after_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        receipt = build_topic_projection(config, topic, records)
    projection_path = topic_root / "canonical" / "retrieval-projection.json"
    original_read_bytes = Path.read_bytes
    replacement_reads = 0

    def read_bytes(path: Path) -> bytes:
        nonlocal replacement_reads
        body = original_read_bytes(path)
        if path == projection_path:
            replacement_reads += 1
            if replacement_reads == 1:
                replacement = json.loads(body)
                candidate = replacement["candidates"][0]
                candidate["docid"] = "doc-b"
                candidate["doc"] = "Replacement text."
                candidate["text_sha256"] = sha256(
                    b"Replacement text."
                ).hexdigest()
                path.write_bytes(_canonical_json(replacement))
        return body

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = _export_retrieval_run(
        config,
        topics,
        (receipt,),
        code_commit="b" * 40,
    )

    assert replacement_reads == 1
    assert result.official_run.read_bytes() == b"rag2026-0 Q0 doc-a 1 1 demo\n"


@pytest.mark.parametrize(
    "mutation",
    ("noncanonical", "duplicate", "missing-lf", "extra-lf", "trailing-whitespace"),
)
def test_export_rejects_coherent_noncanonical_projection_jsonl(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    first = _export_retrieval_run(
        config,
        (topic,),
        (receipt,),
        code_commit="b" * 40,
    )
    original = projection_path.read_bytes()
    if mutation == "noncanonical":
        projection_bytes = (
            json.dumps(json.loads(original), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
    elif mutation == "duplicate":
        row = json.loads(original)
        query = json.dumps(row["query"], ensure_ascii=False, separators=(",", ":"))
        candidates = json.dumps(
            row["candidates"], ensure_ascii=False, separators=(",", ":")
        )
        projection_bytes = (
            '{"query":'
            + query
            + ',"query":'
            + query
            + ',"candidates":'
            + candidates
            + "}\n"
        ).encode("utf-8")
    elif mutation == "missing-lf":
        projection_bytes = original[:-1]
    elif mutation == "extra-lf":
        projection_bytes = original + b"\n"
    else:
        projection_bytes = original[:-1] + b" \n"
    mutated_receipt = _reseal_projection_receipt(
        receipt,
        manifest_path,
        projection_path,
        projection_bytes,
    )
    before = _root_artifact_bodies(first)
    before_manifest = first.manifest.read_bytes()

    with pytest.raises(ValueError):
        _export_retrieval_run(
            config,
            (topic,),
            (mutated_receipt,),
            code_commit="c" * 40,
        )

    assert _root_artifact_bodies(first) == before
    assert first.manifest.read_bytes() == before_manifest


def test_export_rejects_coherent_noncanonical_projection_manifest(
    tmp_path: Path,
) -> None:
    config, topic, receipt, _projection_path, manifest_path = _projection_fixture(tmp_path)
    first = _export_retrieval_run(
        config,
        (topic,),
        (receipt,),
        code_commit="b" * 40,
    )
    manifest_bytes = manifest_path.read_bytes()
    noncanonical_manifest = (
        json.dumps(json.loads(manifest_bytes), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(noncanonical_manifest)
    mutated_receipt = replace(
        receipt,
        manifest_sha256=sha256(noncanonical_manifest).hexdigest(),
        manifest_bytes=len(noncanonical_manifest),
    )
    before = _root_artifact_bodies(first)
    before_manifest = first.manifest.read_bytes()

    with pytest.raises(ValueError):
        _export_retrieval_run(
            config,
            (topic,),
            (mutated_receipt,),
            code_commit="c" * 40,
        )

    assert _root_artifact_bodies(first) == before
    assert first.manifest.read_bytes() == before_manifest


def test_topic_projection_retries_matching_projection_only_state(
    tmp_path: Path,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    generation_path, generation_manifest_path = _generation_projection_paths(
        projection_path
    )
    generation_bytes = generation_path.read_bytes()
    generation_manifest_bytes = generation_manifest_path.read_bytes()
    manifest_path.unlink()

    retrieval_export._publish_topic_projection(
        config,
        topic,
        projection_bytes,
        manifest_bytes,
        generation_bytes,
        generation_manifest_bytes,
        receipt,
    )

    assert projection_path.read_bytes() == projection_bytes
    assert manifest_path.read_bytes() == manifest_bytes


def test_identical_topic_projection_calls_can_race_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    generation_path, generation_manifest_path = _generation_projection_paths(
        projection_path
    )
    generation_bytes = generation_path.read_bytes()
    generation_manifest_bytes = generation_manifest_path.read_bytes()
    projection_path.unlink()
    manifest_path.unlink()
    generation_path.unlink()
    generation_manifest_path.unlink()
    barrier = threading.Barrier(2)
    original_publish = retrieval_export._publish_create_only

    def synchronized_publish(path: Path, body: bytes) -> None:
        barrier.wait()
        original_publish(path, body)

    monkeypatch.setattr(
        retrieval_export,
        "_publish_create_only",
        synchronized_publish,
    )
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            retrieval_export._publish_topic_projection(
                config,
                topic,
                projection_bytes,
                manifest_bytes,
                generation_bytes,
                generation_manifest_bytes,
                receipt,
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            failures.append(exc)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert projection_path.read_bytes() == projection_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert generation_path.read_bytes() == generation_bytes
    assert generation_manifest_path.read_bytes() == generation_manifest_bytes


@pytest.mark.parametrize("conflict", ("projection", "manifest", "projection-only"))
def test_topic_projection_publication_rejects_any_byte_conflict(
    tmp_path: Path,
    conflict: str,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    generation_path, generation_manifest_path = _generation_projection_paths(
        projection_path
    )
    generation_bytes = generation_path.read_bytes()
    generation_manifest_bytes = generation_manifest_path.read_bytes()
    if conflict == "projection":
        projection_path.write_bytes(projection_bytes + b" ")
    elif conflict == "manifest":
        manifest_path.write_bytes(manifest_bytes + b" ")
    else:
        manifest_path.unlink()
        projection_path.write_bytes(projection_bytes + b" ")

    with pytest.raises(ValueError):
        retrieval_export._publish_topic_projection(
            config,
            topic,
            projection_bytes,
            manifest_bytes,
            generation_bytes,
            generation_manifest_bytes,
            receipt,
        )


def test_topic_projection_rejects_manifest_without_projection(
    tmp_path: Path,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    generation_path, generation_manifest_path = _generation_projection_paths(
        projection_path
    )
    generation_bytes = generation_path.read_bytes()
    generation_manifest_bytes = generation_manifest_path.read_bytes()
    projection_path.unlink()

    with pytest.raises(ValueError):
        retrieval_export._publish_topic_projection(
            config,
            topic,
            projection_bytes,
            manifest_bytes,
            generation_bytes,
            generation_manifest_bytes,
            receipt,
        )


def test_topic_projection_fsyncs_canonical_directory_at_each_publish_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.retrieval_export as retrieval_export

    config, topic, receipt, projection_path, manifest_path = _projection_fixture(tmp_path)
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    generation_path, generation_manifest_path = _generation_projection_paths(
        projection_path
    )
    generation_bytes = generation_path.read_bytes()
    generation_manifest_bytes = generation_manifest_path.read_bytes()
    projection_path.unlink()
    manifest_path.unlink()
    generation_path.unlink()
    generation_manifest_path.unlink()
    canonical = projection_path.parent
    directory_sync_states: list[tuple[bool, bool, bool, bool]] = []
    original_fsync = os.fsync

    def fsync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            directory_sync_states.append(
                (
                    projection_path.exists(),
                    manifest_path.exists(),
                    generation_path.exists(),
                    generation_manifest_path.exists(),
                )
            )
        original_fsync(file_descriptor)

    monkeypatch.setattr(retrieval_export.os, "fsync", fsync)
    retrieval_export._publish_topic_projection(
        config,
        topic,
        projection_bytes,
        manifest_bytes,
        generation_bytes,
        generation_manifest_bytes,
        receipt,
    )

    assert directory_sync_states == [
        (True, False, False, False),
        (True, False, True, False),
        (True, True, True, False),
        (True, True, True, True),
    ]
    assert canonical.is_dir()


def test_generation_projection_contains_only_selected_evidence_contract(
    tmp_path: Path,
) -> None:
    _config, _topic, _receipt, projection_path, _manifest_path = (
        _projection_fixture(tmp_path)
    )
    generation_path = projection_path.parent / "generation-projection.json"
    payload = json.loads(generation_path.read_bytes())

    assert set(payload) == {
        "topic_id",
        "narrative",
        "narrative_sha256",
        "groups",
        "evidence",
        "claim_hints",
        "citation_docids",
        "source_receipts",
        "context_sha256",
    }
    assert set(payload["source_receipts"]) == {
        "official_topics",
        "retrieval_topic",
        "selected_evidence",
        "canonical_claim_hints",
    }
    assert all(
        set(evidence) == {
            "evidence_id",
            "group_id",
            "cluster_id",
            "cluster_ordinal",
            "support_ordinal",
            "candidate_kind",
            "docid",
            "document_rank",
            "text",
            "text_sha256",
            "document_sha256",
            "source_span",
        }
        and set(evidence["source_span"])
        == {"start_char", "end_char", "start_byte", "end_byte"}
        for evidence in payload["evidence"]
    )
    assert not (
        {"documents", "full_documents", "document_text", "window", "gold", "qrels"}
        & set(payload)
    )

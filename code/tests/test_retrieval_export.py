from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence
import zipfile

import pytest

from trec_rag.canonical_nuggets import run_canonical_stage
from trec_rag.facet_pilot_config import (
    ExperimentSettings,
    FacetPilotConfig,
    NuggetSettings,
    RerankingSettings,
    RetrievalSettings,
)
from trec_rag.evidence_store import CandidateArtifacts, select_evidence_artifacts
from trec_rag.facet_evidence import (
    CandidateSubnarrative,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SelectionPolicy,
    extract_document_candidates,
)
from trec_rag.facet_extraction import BackendReply, plan_facet_queries
from trec_rag.facet_retrieval import LaneDocumentScore, PassageScore
from trec_rag.competition_retrieval import (
    _plan_payload,
    _retrieve_topic,
    _score_topic,
    load_validated_decomposition,
)
from trec_rag.pipeline_models import RetrievedCandidate, jsonable
from trec_rag.retrieval_export import (
    RetrievalExportReceipt,
    export_retrieval_run,
    read_retrieval_export_receipt,
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
            receipt.candidate_pool_run,
            receipt.with_text_archive,
            receipt.provenance,
            receipt.resolved_config,
        )
    }


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
) -> dict[str, object]:
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
                passage_id=f"passage-{docid}",
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
    return jsonable(candidates[0])


def _write_evidence_artifacts(
    canonical: Path,
    topic: Topic,
    plan,
    supported: Sequence[str],
) -> list[dict[str, object]]:
    policy = SelectionPolicy(budgets=(40,), precluster_limit=400)
    candidates = [
        _candidate_for(
            topic,
            docid,
            plan.subnarratives[0].subnarrative_id,
            plan.subnarratives[0].text,
        )
        for docid in supported
    ]
    candidate_bytes = b"".join(_canonical_json(row) for row in candidates)
    (canonical / "candidates.jsonl").write_bytes(candidate_bytes)
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
    }
    candidate_manifest = {
        "schema_version": "extractive_candidate_manifest_v1",
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "source_sha256": sha256(request_bytes).hexdigest(),
        "scorer": scorer_identity,
        "sentence_splitter_version": "exact_rules_v1",
        "scoring_normalization_version": "trec_rag_whitespace_v1",
        "source_file": "candidate-requests.jsonl",
        "candidate_file": "candidates.jsonl",
        "input_sha256": sha256(request_bytes).hexdigest(),
        "candidates_sha256": sha256(candidate_bytes).hexdigest(),
        "output_sha256": sha256(candidate_bytes).hexdigest(),
        "document_count": len(candidates),
        "unique_document_count": len(candidates),
        "unique_subnarrative_count": 1 if candidates else 0,
        "failure_count": 0,
        "candidate_count": len(candidates),
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }
    candidate_manifest_bytes = _canonical_json(candidate_manifest)
    (canonical / "candidate-manifest.json").write_bytes(candidate_manifest_bytes)

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
            canonical / "candidates.jsonl", canonical / "candidate-manifest.json"
        ),
        contexts_path,
        device="cpu",
        similarity=_DistinctSimilarity(),
        policy=policy,
    )
    cluster_by_candidate = {
        member["candidate_nugget_id"]: cluster["cluster_id"]
        for selection in _read_jsonl(selected_artifacts.selections_path)
        for cluster in selection["clusters"]
        for member in cluster["supports"]
    }
    for candidate in candidates:
        candidate["cluster_id"] = cluster_by_candidate[candidate["candidate_nugget_id"]]
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
            candidate_depth_per_query=1000,
        ),
        reranking=RerankingSettings(
            model="mixedbread-ai/mxbai-rerank-base-v2",
            score_cache_dir=tmp_path / "cache" / "reranker",
            device="auto",
            rerank_depth_per_query=100,
            candidate_pool_depth=100,
            selection_policy="round_robin_subnarrative_coverage",
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
    _retrieve_topic(
        topic,
        decomposition,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        code_commit=source_commit,
        retriever=retriever,
        retrieval_depth=1000,
    )
    _score_topic(
        topic,
        decomposition,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        score_cache_root=output_dir / "score-cache",
        code_commit=source_commit,
        retriever=retriever,
        scorer=_FixtureDocumentScorer(),
        device="cpu",
        retrieval_depth=1000,
        rerank_depth=100,
        selection_k=100,
    )
    scoring_manifest_path = scoring / "complete.json"

    _write_evidence_artifacts(canonical, topic, plan, supported)
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
        "canonical/candidates.jsonl",
        "canonical/candidate-manifest.json",
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
        "result_schema_version": "canonical_nugget_result_v1",
        "canonical_response_schema_version": "canonical_nuggets_v1",
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
        "prompt_version": "canonical_nuggetizer_v4",
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
    else:
        candidate_path = topic_root / "canonical" / "candidates.jsonl"
        candidates = _read_jsonl(candidate_path)
        candidates[0]["evidence_sentences"][0]["start_byte"] = 1
        candidate_path.write_bytes(
            b"".join(_canonical_json(row) for row in candidates)
        )
        candidate_manifest_path = topic_root / "canonical" / "candidate-manifest.json"
        _rewrite_manifest(
            candidate_manifest_path,
            candidates_sha256=sha256(candidate_path.read_bytes()).hexdigest(),
            output_sha256=sha256(candidate_path.read_bytes()).hexdigest(),
        )
        selection_manifest_path = topic_root / "canonical" / "selection-manifest.json"
        _rewrite_manifest(
            selection_manifest_path,
            candidates_sha256=sha256(candidate_path.read_bytes()).hexdigest(),
            candidate_manifest_sha256=sha256(
                candidate_manifest_path.read_bytes()
            ).hexdigest(),
        )
        canonical_nugget_manifest = (
            topic_root / "canonical" / "canonical-nugget-manifest.json"
        )
        _rewrite_manifest(
            canonical_nugget_manifest,
            selection_manifest_sha256=sha256(
                selection_manifest_path.read_bytes()
            ).hexdigest(),
        )
        changed = (
            "canonical/candidates.jsonl",
            "canonical/candidate-manifest.json",
            "canonical/selection-manifest.json",
            "canonical/canonical-nugget-manifest.json",
        )
    for relative in changed:
        _resign_canonical_artifact(topic_root, relative)

    with pytest.raises(ValueError, match="canonical evidence|candidate"):
        export_retrieval_run(config, (topic,), code_commit="b" * 40)


def test_export_writes_exact_variable_depth_official_and_candidate_runs(
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

    receipt = export_retrieval_run(config, topics, code_commit="a" * 40)

    assert receipt.official_run.read_bytes() == (
        b"rag2026-0 Q0 doc-a 1 2 demo\n"
        b"rag2026-0 Q0 doc-c 2 1 demo\n"
    )
    assert receipt.candidate_pool_run.read_bytes() == (
        b"rag2026-0 Q0 doc-b 1 3 demo-candidate-pool\n"
        b"rag2026-0 Q0 doc-a 2 2 demo-candidate-pool\n"
        b"rag2026-0 Q0 doc-c 3 1 demo-candidate-pool\n"
    )


def test_export_streams_the_sealed_candidate_ledger(
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
    candidate_path = (
        config.output_dir / topics[0].id / "canonical" / "candidates.jsonl"
    ).resolve()
    original_read_bytes = Path.read_bytes

    def reject_whole_candidate_read(path: Path) -> bytes:
        if path.resolve() == candidate_path:
            raise AssertionError("sealed candidate ledgers must be streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_whole_candidate_read)

    receipt = export_retrieval_run(config, topics, code_commit="a" * 40)

    assert receipt.official_run.read_text() == "rag2026-0 Q0 doc-a 1 1 demo\n"


def test_export_writes_deterministic_full_text_provenance_and_manifest(
    tmp_path: Path,
) -> None:
    config, topics = _config_and_topics(tmp_path)
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
        row = json.loads(archive.read("retrieval_with_text.jsonl"))
    assert row["query"] == {"qid": "rag2026-0", "text": topics[0].narrative}
    assert row["candidates"][0]["doc"] == "Full text for doc-a."
    assert row["candidates"][0]["stage"] == "canonical_supported"
    provenance = _read_jsonl(second.provenance)
    assert "stage" not in provenance[0]
    assert provenance[0]["selected_from_lane"] == "original"
    assert provenance[0]["selected_from_lane_rank"] == 1
    assert provenance[0]["memberships"] == [
        {
            "lane_name": "original",
            "aggregate_rank": 1,
            "aggregate_score": 4.0,
            "bm25_rank": 1,
            "bm25_score": 99.0,
        }
    ]
    assert "bm25_score" not in provenance[0]["subnarrative_scores"][0]
    assert "bm25_rank" not in provenance[0]["subnarrative_scores"][0]
    assert provenance[0]["nuggets"][0]["canonical_nugget_id"]
    manifest = json.loads(second.manifest.read_bytes())
    assert manifest["schema_version"] == "retrieval_export_manifest_v2"
    assert manifest["export_code_commit"] == "b" * 40
    assert manifest["source_code_commits"] == ["b" * 40]
    assert manifest["score_semantics"] == "ordinal_selection_order"
    assert manifest["artifacts"]["r_output_trec_rag_2026.tsv"]["sha256"]


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


def test_export_rejects_canonical_evidence_absent_from_selected_documents(
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

    with pytest.raises(ValueError, match="evidence document.*absent"):
        export_retrieval_run(config, topics, code_commit="c" * 40)


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

    with pytest.raises(ValueError, match="original narrative query"):
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
            retrieval=replace(config.retrieval, candidate_depth_per_query=999),
        ),
        lambda config: replace(
            config,
            reranking=replace(config.reranking, model="changed-model"),
        ),
        lambda config: replace(
            config,
            reranking=replace(config.reranking, rerank_depth_per_query=99),
        ),
        lambda config: replace(
            config,
            reranking=replace(config.reranking, candidate_pool_depth=99),
        ),
        lambda config: replace(
            config,
            reranking=replace(config.reranking, selection_policy="changed-policy"),
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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.facet_extraction import (
    BackendReply,
    FacetPlanningResult,
    GeneratedQueryPlan,
    Subnarrative,
    plan_facet_queries,
)
from trec_rag.evidence_local import mixedbread_sentence_scorer_identity
from trec_rag.evidence_store import (
    generate_candidate_artifacts,
    materialize_candidate_inputs,
)
from trec_rag.mixedbread_passage_scorer import (
    MIXEDBREAD_MODEL,
    MixedbreadPassageScorer,
    ScoredPassage as MixedbreadScoredPassage,
)
from trec_rag.rerank_score_cache import _choose_device
from trec_rag.document_store import (
    DocumentStore,
    DocumentStoreIntegrityError,
    ReadOnlyDocumentStore,
)
from trec_rag.deepagent_budget import ResearchTaskContext
from trec_rag.deepagent_research import ResearchTaskEnvelope, bind_research_task
from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.remote_pyserini import RemotePyseriniThrottled
from trec_rag.topics import Topic
from trec_rag.topic_passage_search import (
    FocusedQuery,
    PassageScoringFailed,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
)
from trec_rag.topic_records import FacetRecord, TopicRecords, TopicRecordsBuilder
from trec_rag.competition_retrieval import (
    CACHE_OPERATION_STAGE_NAMES,
    ValidatedDecomposition,
    ValidatedRetrievalAuditLane,
    _RuntimeDependencies,
    _configured_passage_search_identity,
    _decode_passage_result,
    _build_topic_passage_search,
    _passage_result_json,
    _publish_topic_cache_operation_receipt,
    _selection,
    _plan_payload,
    _topics_sha256,
    _retrieve_topic,
    _run_topic,
    _score_topic,
    load_validated_decomposition,
    validate_scoring_selection,
)
from trec_rag.facet_retrieval import (
    FacetRetrievalResult,
    LaneRanking,
    RetrievalAuditCandidate,
    build_retrieval_lanes,
    round_robin_select,
)


ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"


def _zero_operation_stages() -> dict[str, dict[str, int]]:
    return {
        stage: {
            "cache_hits": 0,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for stage in CACHE_OPERATION_STAGE_NAMES
    }


@pytest.mark.parametrize("stage", ["planning", "retrieval", "passage_scores"])
@pytest.mark.parametrize(
    "counter",
    ["cache_misses", "network_calls", "provider_calls", "model_batches"],
)
def test_cached_upstream_receipt_rejects_each_forbidden_work_counter(
    tmp_path: Path,
    stage: str,
    counter: str,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(base, root_dir=tmp_path)
    topic = Topic("topic-a", "", "narrative")
    stages = _zero_operation_stages()
    stages[stage][counter] = 1

    with pytest.raises(ValueError, match="cached upstream operation"):
        _publish_topic_cache_operation_receipt(
            config=config,
            topic=topic,
            config_sha256="a" * 64,
            projection_manifest_sha256="b" * 64,
            mode="cached-upstream-rescore",
            phases={
                phase: {"resumed": False}
                for phase in ("planning", "retrieval", "scoring", "canonical")
            },
            stages=stages,
        )


def test_cached_upstream_receipt_allows_downstream_rescoring_work(
    tmp_path: Path,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(base, root_dir=tmp_path)
    topic = Topic("topic-a", "", "narrative")
    stages = _zero_operation_stages()
    for stage in ("sentence_scores", "similarity", "canonicalization"):
        stages[stage]["cache_misses"] = 1
        stages[stage]["model_batches"] = 1

    receipt = _publish_topic_cache_operation_receipt(
        config=config,
        topic=topic,
        config_sha256="a" * 64,
        projection_manifest_sha256="b" * 64,
        mode="cached-upstream-rescore",
        phases={
            phase: {"resumed": False}
            for phase in ("planning", "retrieval", "scoring", "canonical")
        },
        stages=stages,
    )

    assert receipt.path.name == (
        "cache-operation-receipt.cached-upstream-rescore.json"
    )


def test_v2_config_pins_shared_passage_defaults() -> None:
    config = load_facet_pilot_config(V2_CONFIG)

    assert config.retrieval.documents_per_query == 1000
    assert config.passage.passages_per_query == 100
    assert config.passage.chunk_max_characters == 3500
    assert config.passage.chunk_overlap_characters == 350
    assert config.passage.model == MIXEDBREAD_MODEL


def test_v1_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "schema_version: facet_pilot_config_v1\n"
        "experiment:\n  id: legacy\n"
        "topics:\n  path: topics.tsv\n"
        "retrieval:\n"
        "  index: climbmix-400b\n"
        "  cache_dir: cache/retrieval\n"
        "  query_sources: [original, subnarrative]\n"
        "  candidate_depth_per_query: 1000\n"
        "reranking:\n"
        "  model: mixedbread-ai/mxbai-rerank-base-v2\n"
        "  score_cache_dir: cache/reranker\n"
        "  device: cpu\n"
        "  rerank_depth_per_query: 100\n"
        "  candidate_pool_depth: 100\n"
        "  selection_policy: round_robin_subnarrative_coverage\n"
        "nuggets:\n"
        "  evidence_budget_per_subnarrative: 40\n"
        "  maximum_claims_per_subnarrative: 20\n"
        "  maximum_supporting_documents_per_claim: 3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="facet_pilot_config_v2"):
        load_facet_pilot_config(path)


def test_default_passage_builder_constructs_adapters_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INDEX_URL", "https://pyserini.invalid/search")
    adapter = _build_topic_passage_search(
        Topic("topic-builder", "title", "find evidence"),
        retriever=None,
        scorer=None,
        document_store_root=tmp_path / "objects",
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1000,
        passages_per_query=100,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )
    assert adapter.identity["policy"] == {
        "retrieval_depth": 1000,
        "passage_limit": 100,
        "max_attempts": 3,
    }


def test_configured_passage_identity_exactly_matches_shared_scorer(
    tmp_path: Path,
) -> None:
    scorer = MixedbreadPassageScorer(
        tmp_path / "score-cache",
        device="cpu",
        batch_size=8,
    )
    identity = _configured_passage_search_identity(
        retrieval_depth=1000,
        passages_per_query=100,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        model=MIXEDBREAD_MODEL,
        device="cpu",
    )

    assert identity["scorer"] == scorer.identity


def test_configured_passage_identity_resolves_auto_device() -> None:
    identity = _configured_passage_search_identity(
        retrieval_depth=1000,
        passages_per_query=100,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        model=MIXEDBREAD_MODEL,
        device="auto",
    )

    assert identity["scorer"]["device"] == _choose_device("auto")
    assert identity["scorer"]["device"] != "auto"


@pytest.mark.parametrize(
    ("field", "value"),
    (("batch_size", 9), ("implementation_version", 2)),
)
def test_passage_builder_rejects_any_shared_scorer_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    actual = MixedbreadPassageScorer(
        tmp_path / "actual-score-cache",
        device="cpu",
    )

    class MismatchedScorer:
        identity = {**actual.identity, field: value}

    with pytest.raises(ValueError, match="identity differs"):
        _build_topic_passage_search(
            Topic("topic-mismatch", "title", "find evidence"),
            retriever=None,
            scorer=MismatchedScorer(),
            document_store_root=tmp_path / "objects",
            retrieval_cache_dir=tmp_path / "retrieval-cache",
            retrieval_index="climbmix-test",
            corpus_epoch="test-epoch",
            score_cache_root=tmp_path / "configured-score-cache",
            device="cpu",
            retrieval_depth=1000,
            passages_per_query=100,
            chunk_max_characters=3500,
            chunk_overlap_characters=350,
        )


def test_passage_builder_rejects_scorer_on_different_resolved_auto_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = MixedbreadPassageScorer(
        tmp_path / "cpu-score-cache",
        device="cpu",
    )
    monkeypatch.setattr(
        "trec_rag.competition_retrieval._choose_device",
        lambda requested: "cuda" if requested == "auto" else requested,
    )

    with pytest.raises(ValueError, match="identity differs"):
        _build_topic_passage_search(
            Topic("topic-device-mismatch", "title", "find evidence"),
            retriever=None,
            scorer=scorer,
            document_store_root=tmp_path / "objects",
            retrieval_cache_dir=tmp_path / "retrieval-cache",
            retrieval_index="climbmix-test",
            corpus_epoch="test-epoch",
            score_cache_root=tmp_path / "configured-score-cache",
            device="auto",
            retrieval_depth=1000,
            passages_per_query=100,
            chunk_max_characters=3500,
            chunk_overlap_characters=350,
        )


def _offline_shared_scorer(tmp_path: Path):
    identity = MixedbreadPassageScorer(
        tmp_path / "identity-cache",
        device="cpu",
    ).identity

    class OfflineSharedScorer:
        def __init__(self) -> None:
            self.identity = identity
            self._stats = {"cache_hits": 0, "cache_misses": 0, "model_batches": 0}

        @property
        def stats(self):
            return dict(self._stats)

        @staticmethod
        def cache_key(query_text: str, passage_text: str) -> str:
            return sha256(f"{query_text}\0{passage_text}".encode()).hexdigest()

        def rank(self, query_text, chunks):
            assert query_text
            self._stats["cache_misses"] += len(chunks)
            self._stats["model_batches"] += int(bool(chunks))
            return tuple(
                MixedbreadScoredPassage(chunk, float(len(chunks) - index))
                for index, chunk in enumerate(chunks)
            )

    return OfflineSharedScorer()


def test_cached_upstream_missing_document_fails_before_passage_scoring(
    tmp_path: Path,
) -> None:
    topic = Topic("topic-cached", "title", "find cached evidence")

    class CachedRetriever:
        identity = {"name": "cached", "type": "test", "hits": 1}

        @staticmethod
        def retrieve(query: QueryVariant):
            return (
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "cached",
                    query.query_text,
                    "doc-missing",
                    1,
                    2.0,
                    "This exact source was not restored into the document cache.",
                ),
            )

    scorer = _offline_shared_scorer(tmp_path)

    def reject_scoring(*_args, **_kwargs):
        pytest.fail("passage scoring must not run after a document-cache miss")

    scorer.rank = reject_scoring
    store_root = tmp_path / "missing-objects"
    search = _build_topic_passage_search(
        topic,
        retriever=CachedRetriever(),
        scorer=scorer,
        document_store_root=store_root,
        document_store=ReadOnlyDocumentStore(store_root),
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        cache_only=True,
    )

    with pytest.raises(DocumentStoreIntegrityError, match="unable to read"):
        search.search(FocusedQuery("original", topic.narrative, "original"))

    assert not store_root.exists()


def test_passage_builder_accepts_external_one_argument_retriever(
    tmp_path: Path,
) -> None:
    class RecordingRetriever:
        identity = {"name": "recording", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.calls: list[QueryVariant] = []

        def retrieve(self, query: QueryVariant):
            self.calls.append(query)
            return (
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "recording",
                    query.query_text,
                    "doc-1",
                    1,
                    2.0,
                    "one offline source passage.",
                ),
            )

    retriever = RecordingRetriever()
    search = _build_topic_passage_search(
        Topic("topic-external", "title", "find evidence"),
        retriever=retriever,
        scorer=_offline_shared_scorer(tmp_path),
        document_store_root=tmp_path / "objects",
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )

    result = search.search(FocusedQuery("original", "find evidence", "original"))

    assert result.status == "complete"
    assert len(result.passages) == 1
    assert len(retriever.calls) == 1


def test_built_passage_adapter_drives_agentic_retrieval_for_its_topic(
    tmp_path: Path,
) -> None:
    topic = Topic("topic-agentic-adapter", "title", "find grounded evidence")

    class OfflineRetriever:
        identity = {"name": "offline", "type": "test", "hits": 1}

        @staticmethod
        def retrieve(query: QueryVariant):
            return (
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "offline",
                    query.query_text,
                    "doc-agentic",
                    1,
                    2.0,
                    "Grounded evidence for the agentic adapter.",
                ),
            )

    object_root = tmp_path / "objects"
    passage_search = _build_topic_passage_search(
        topic,
        retriever=OfflineRetriever(),
        scorer=_offline_shared_scorer(tmp_path),
        document_store_root=object_root,
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )
    records = TopicRecordsBuilder(
        tmp_path / "topic-records",
        topic.id,
        DocumentStore(object_root),
        run_id="run-agentic-adapter",
    )
    tracing = SimpleNamespace(
        agent_span=lambda _narrative: nullcontext(None),
        retriever_span=lambda _query: nullcontext(None),
        snippet_span=lambda _document, _query: nullcontext(None),
        force_flush=lambda: True,
    )
    captured: dict[str, object] = {}

    def agent_factory(_model, toolset):
        def invoke(_payload):
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_needs": [
                            {
                                "need_id": "need-worker",
                                "narrative_span": topic.narrative,
                                "question": "What evidence grounds the worker search?",
                            }
                        ]
                    }
                )
            )
            assert update["accepted_ids"] == ["need-worker"]

            def search_from_worker():
                context = ResearchTaskContext(
                    "researcher-worker", 1, "focused", ("need-worker",)
                )
                envelope = ResearchTaskEnvelope(
                    research_task_id="researcher-worker",
                    round_index=1,
                    depth="focused",
                    motivating_ids=["need-worker"],
                    goal="Find grounded worker evidence.",
                )
                assert toolset.budget.reserve_task(context).ok
                try:
                    with bind_research_task(envelope):
                        return json.loads(
                            toolset.search_passages(
                                "worker-thread grounded evidence",
                                ["need-worker"],
                                "The worker need has no evidence.",
                            )
                        )
                finally:
                    toolset.budget.finish_task(context)

            with ThreadPoolExecutor(max_workers=1) as executor:
                captured["followup"] = executor.submit(search_from_worker).result()
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return SimpleNamespace(invoke=invoke)

    result = DeepAgentRetriever(
        passage_search=passage_search,
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
    ).retrieve(records, topic.narrative)

    assert passage_search.topic_id == topic.id
    assert captured["followup"]["ok"] is True
    assert len(result.searches) == 2
    assert result.searches[0].query == topic.narrative
    assert [candidate.docid for candidate in result.searches[0].candidates] == [
        "doc-agentic"
    ]
    assert result.searches[1].query == "worker-thread grounded evidence"
    assert [query.query_id for query in result.topic_snapshot.queries] == [
        "followup-35bece226111ec70",
        "original",
    ]


def test_long_throttle_does_not_replay_transport_during_active_continuation(
    tmp_path: Path,
) -> None:
    class ThrottledRetriever:
        identity = {"name": "throttled", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query: QueryVariant):
            self.calls += 1
            raise RemotePyseriniThrottled(600)

    retriever = ThrottledRetriever()
    search = _build_topic_passage_search(
        Topic("topic-throttled", "title", "find evidence"),
        retriever=retriever,
        scorer=_offline_shared_scorer(tmp_path),
        document_store_root=tmp_path / "objects",
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )

    result = search.search(FocusedQuery("original", "find evidence", "original"))

    assert result.status == "incomplete"
    assert result.stopping_reason == "retrieval_unavailable"
    assert result.attempt_count == 1
    assert retriever.calls == 1


def test_terminal_throttle_latches_across_lanes_and_seals_retrieval(
    tmp_path: Path,
) -> None:
    topic = Topic("topic-throttled-lanes", "title", "find evidence")
    planning = plan_facet_queries(
        topic,
        {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": [
                {
                    "subnarrative": "one evidence facet",
                    "bm25_queries": ["evidence"],
                }
            ],
        },
    )
    assert planning.plan is not None
    decomposition = ValidatedDecomposition(
        topic.id,
        sha256(topic.narrative.encode()).hexdigest(),
        "d" * 64,
        planning,
    )

    class ThrottledThenPoisonRetriever:
        identity = {"name": "throttled", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query: QueryVariant):
            self.calls += 1
            if self.calls == 1:
                raise RemotePyseriniThrottled(600)
            raise RuntimeError("terminal continuation transport was replayed")

    retriever = ThrottledThenPoisonRetriever()
    passage_identity = _configured_passage_search_identity(
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        model=MIXEDBREAD_MODEL,
        device="cpu",
    )
    passage_search = _build_topic_passage_search(
        topic,
        retriever=retriever,
        scorer=_offline_shared_scorer(tmp_path),
        document_store_root=tmp_path / "objects",
        retrieval_cache_dir=tmp_path / "retrieval-cache",
        retrieval_index="climbmix-test",
        corpus_epoch="test-epoch",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
    )

    outcome = _retrieve_topic(
        topic,
        decomposition,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "retrieval-cache",
        code_commit="a" * 40,
        corpus_epoch="test-epoch",
        retriever=retriever,
        retrieval_depth=1,
        passage_search=passage_search,
        passage_identity=passage_identity,
    )

    audit = json.loads(
        (tmp_path / "output" / topic.id / "retrieval" / "audit.json").read_bytes()
    )
    results = tuple(
        _decode_passage_result(row) for row in audit["passage_search_results"]
    )
    assert retriever.calls == 1
    assert outcome.manifest_path.is_file()
    assert len(results) == 2
    assert all(result.status == "incomplete" for result in results)
    assert all(result.stopping_reason == "retrieval_unavailable" for result in results)
    assert all(result.attempt_count == 1 for result in results)


def test_run_topic_reaches_sealed_score_checkpoint_with_external_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = Topic("topic-offline-run", "title", "find offline evidence")
    query = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    decomposition = ValidatedDecomposition(
        topic.id,
        sha256(topic.narrative.encode()).hexdigest(),
        "d" * 64,
        FacetPlanningResult((query,), True, "fallback", None, ()),
    )
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="offline-v2"),
        retrieval=replace(
            base.retrieval,
            cache_dir=tmp_path / "retrieval-cache",
            documents_per_query=1,
        ),
        passage=replace(
            base.passage,
            score_cache_dir=tmp_path / "passage-score-cache",
            device="cpu",
            passages_per_query=1,
        ),
    )

    class RecordingRetriever:
        identity = {"name": "recording", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.calls: list[QueryVariant] = []
            self.transport_calls = 0

        def cache_summary(self):
            return {"hits": 0, "misses": self.transport_calls}

        def retrieve(self, request: QueryVariant):
            self.calls.append(request)
            self.transport_calls += 1
            return (
                RetrievedCandidate(
                    request.topic_id,
                    request.variant_name,
                    "recording",
                    request.query_text,
                    "doc-offline",
                    1,
                    3.0,
                    "offline evidence sentence.",
                ),
            )

    retriever = RecordingRetriever()
    dependencies = _RuntimeDependencies(
        code_commit="a" * 40,
        document_scorer=_offline_shared_scorer(tmp_path),
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
        retriever=retriever,
    )
    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decompose_topic",
        lambda *args, **kwargs: (decomposition, False),
    )
    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decomposition_producer_sha256",
        lambda *_args, **_kwargs: "e" * 64,
    )

    class ReachedCanonicalBoundary(RuntimeError):
        pass

    def stop_after_scoring(*args, **kwargs):
        score_manifest = (
            config.output_dir / topic.id / "scoring" / "complete.json"
        )
        assert score_manifest.is_file()
        manifest = json.loads(score_manifest.read_text())
        assert manifest["phase"] == "score"
        assert (
            manifest["passage_search"]["scorer"]
            == dependencies.document_scorer.identity
        )
        raise ReachedCanonicalBoundary

    monkeypatch.setattr(
        "trec_rag.competition_retrieval._canonical_topic",
        stop_after_scoring,
    )

    with pytest.raises(ReachedCanonicalBoundary):
        _run_topic(
            topic,
            config,
                "c" * 64,
                dependencies,
                config_sha256="b" * 64,
                expected_retriever_identity=retriever.identity,
        )

    assert len(retriever.calls) == 1


def test_passage_result_roundtrip_preserves_source_ids_and_rejects_tampering() -> None:
    text = "source-bound passage"
    digest = __import__("hashlib").sha256(text.encode()).hexdigest()
    passage_id = "p-doc-1-0000"
    result = PassageSearchResult(
        FocusedQuery("original", "find source", "original"),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (SourceDocument("doc-1", digest, 1, 2.0, passage_id, 3.0),),
        (
            SourcePassage(
                passage_id,
                "doc-1",
                digest,
                1,
                2.0,
                0,
                len(text),
                0,
                len(text.encode()),
                digest,
                text,
                3.0,
                1,
                "cache-key",
                digest,
                {"backend": "test", "implementation": "v1"},
            ),
        ),
        1,
        False,
    )

    payload = _passage_result_json(result)
    assert _decode_passage_result(payload) == result

    tampered = {**payload, "passages": [dict(payload["passages"][0], docid="other-doc")]}
    with pytest.raises(ValueError):
        _decode_passage_result(tampered)


def test_incomplete_scoring_replays_empty_selection_from_bound_best_passages() -> None:
    topic = Topic("topic-incomplete-selection", "title", "find source")
    query = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    lane = build_retrieval_lanes(topic, (query,), ())[0]
    passage_result = PassageSearchResult(
        FocusedQuery("original", topic.narrative, "original"),
        "incomplete",
        "scoring_failed",
        2,
        2,
        0,
        0,
        (
            SourceDocument("doc-1", "a" * 64, 1, 2.0, None, None),
            SourceDocument("doc-2", "b" * 64, 2, 1.0, None, None),
        ),
        (),
        1,
        False,
    )
    audit_candidates = tuple(
        RetrievalAuditCandidate(
            document.docid,
            document.source_rank,
            document.source_score,
            document.content_sha256,
        )
        for document in passage_result.documents
    )
    audited = ValidatedRetrievalAuditLane(
        lane, 2, 2, audit_candidates, passage_result
    )
    ranking = LaneRanking(
        lane,
        2,
        2,
        audit_candidates,
        (),
        retrieval_requested_depth=2,
        rerank_depth=0,
        passage_result=passage_result,
    )
    selection = round_robin_select((ranking,), limit=1)
    result = FacetRetrievalResult(
        topic.id,
        (ranking,),
        selection,
        (),
        (),
        2,
        1,
    )
    selected_set_sha256 = sha256(b"[]").hexdigest()
    source = json.dumps(
        _selection(result, selected_set_sha256),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    memberships = validate_scoring_selection(
        source,
        topic_id=topic.id,
        selected_documents=(),
        lane_score_rows=(),
        audit_lanes=(audited,),
        rerank_depth=2,
        selected_set_sha256=selected_set_sha256,
    )

    assert memberships == ()

    with pytest.raises(ValueError, match="selection lane scores"):
        validate_scoring_selection(
            source,
            topic_id=topic.id,
            selected_documents=(),
            lane_score_rows=({"lane_name": "original"},),
            audit_lanes=(audited,),
            rerank_depth=2,
            selected_set_sha256=selected_set_sha256,
        )


def test_fresh_v2_topic_reaches_real_sealed_projection_without_network_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = Topic("topic-offline-projection", "title", "find offline evidence")
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text(f"{topic.id}\t{topic.narrative}\n", encoding="utf-8")
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        topics_path=topics_path,
        experiment=replace(base.experiment, id="offline-projection"),
        retrieval=replace(
            base.retrieval,
            cache_dir=tmp_path / "retrieval-cache",
            documents_per_query=1,
        ),
        passage=replace(
            base.passage,
            score_cache_dir=tmp_path / "passage-score-cache",
            device="cpu",
            passages_per_query=1,
        ),
    )
    planning = plan_facet_queries(
        topic,
        {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": [
                {
                    "subnarrative": "offline facet evidence",
                    "bm25_queries": ["offline"],
                }
            ],
        },
    )
    assert planning.plan is not None

    class RecordingRetriever:
        identity = {"name": "recording", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.transport_calls = 0

        def cache_summary(self):
            return {
                "hits": 0,
                "misses": self.transport_calls,
                "bypasses": 0,
                "writes": self.transport_calls,
            }

        def retrieve(self, request: QueryVariant):
            self.transport_calls += 1
            return (
                RetrievedCandidate(
                    request.topic_id,
                    request.variant_name,
                    "recording",
                    request.query_text,
                    "doc-offline",
                    1,
                    3.0,
                    "Offline evidence sentence.",
                ),
            )

    class CandidateScorer:
        identity = mixedbread_sentence_scorer_identity()

        def __init__(self) -> None:
            self._cache_misses = 0
            self._model_batches = 0

        @property
        def accounting(self):
            return SimpleNamespace(
                cache_hits=0,
                cache_misses=self._cache_misses,
                model_batches=self._model_batches,
            )

        def score_pairs(self, pairs):
            self._cache_misses += len(pairs)
            self._model_batches += int(bool(pairs))
            return tuple(1.0 for _ in pairs)

    class Similarity:
        identity = {"model": "offline-similarity"}

        def __init__(self) -> None:
            self._cache_misses = 0
            self._model_batches = 0

        @property
        def accounting(self):
            return SimpleNamespace(
                cache_hits=0,
                cache_misses=self._cache_misses,
                model_batches=self._model_batches,
            )

        def cosine_matrix(self, texts):
            self._cache_misses += 1
            self._model_batches += int(bool(texts))
            return tuple(
                tuple(1.0 if left == right else 0.0 for right in texts)
                for left in texts
            )

    class CanonicalBackend:
        def complete(self, request):
            claims = [
                {
                    "claim": f"Offline canonical claim {index}.",
                    "evidence_aliases": [evidence.alias],
                    "importance": "vital",
                }
                for index, evidence in enumerate(request.evidence, start=1)
            ]
            return BackendReply(
                content=json.dumps({"claims": claims}, separators=(",", ":")).encode(),
                response_body=b'{"offline":true}',
                status=200,
                metadata={
                    "requested_model": "deepseek/deepseek-v4-flash-20260423",
                    "response_model": "deepseek/deepseek-v4-flash-20260423",
                    "provider": "offline",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

    retriever = RecordingRetriever()
    dependencies = _RuntimeDependencies(
        code_commit="a" * 40,
        document_scorer=_offline_shared_scorer(tmp_path),
        candidate_scorer=CandidateScorer(),
        similarity=Similarity(),
        cache_ignore_checker=lambda _path: True,
        retriever=retriever,
        canonical_backend_factory=CanonicalBackend,
    )
    def fake_decompose(topic_arg, output_dir, _backend, **_kwargs):
        path = Path(output_dir) / topic_arg.id / "decomposition" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {
                    "schema_version": "facet_pilot_v2",
                    "topic": {
                        "id": topic_arg.id,
                        "narrative": topic_arg.narrative,
                    },
                    "narrative_sha256": sha256(
                        topic_arg.narrative.encode()
                    ).hexdigest(),
                    "used_fallback": planning.used_fallback,
                    "error": planning.error,
                    "queries": jsonable(planning.queries),
                    "plan": _plan_payload(planning.plan),
                    "subnarratives": jsonable(planning.subnarratives),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        return load_validated_decomposition(topic_arg, path), False

    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decompose_topic", fake_decompose
    )
    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decomposition_producer_sha256",
        lambda *_args, **_kwargs: "e" * 64,
    )

    receipt = _run_topic(
        topic,
        config,
        _topics_sha256((topic,)),
        dependencies,
        config_sha256="b" * 64,
        expected_retriever_identity=retriever.identity,
    ).projection_receipt

    assert receipt.topic_id == topic.id
    assert receipt.projection_sha256
    assert (
        config.output_dir / topic.id / "canonical" / "retrieval-projection.json"
    ).is_file()
    assert (
        config.output_dir / topic.id / "canonical" / "retrieval-projection-manifest.json"
    ).is_file()
    operation_path = config.output_dir / topic.id / "cache-operation-receipt.json"
    operation = json.loads(operation_path.read_bytes())
    assert operation["mode"] == "online"
    assert operation["config_sha256"] == "b" * 64
    assert operation["projection_manifest_sha256"] == receipt.manifest_sha256
    assert operation["phases"] == {
        "planning": {"resumed": False},
        "retrieval": {"resumed": False},
        "scoring": {"resumed": False},
        "canonical": {"resumed": False},
    }
    assert operation["stages"]["retrieval"] == {
        "cache_hits": 0,
        "cache_misses": 2,
        "network_calls": 2,
        "provider_calls": 0,
        "model_batches": 0,
    }
    assert operation["stages"]["canonicalization"]["provider_calls"] == 1
    content = dict(operation)
    digest = content.pop("receipt_content_sha256")
    assert digest == sha256(
        (json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def test_fresh_v2_all_unscored_topic_refuses_an_empty_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = Topic("topic-unscored-projection", "title", "find unavailable evidence")
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text(f"{topic.id}\t{topic.narrative}\n", encoding="utf-8")
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        topics_path=topics_path,
        experiment=replace(base.experiment, id="unscored-projection"),
        retrieval=replace(
            base.retrieval,
            cache_dir=tmp_path / "retrieval-cache",
            documents_per_query=1,
        ),
        passage=replace(
            base.passage,
            score_cache_dir=tmp_path / "passage-score-cache",
            device="cpu",
            passages_per_query=1,
        ),
    )
    planning = plan_facet_queries(
        topic,
        {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": [
                {
                    "subnarrative": "unavailable facet evidence",
                    "bm25_queries": ["unavailable"],
                }
            ],
        },
    )
    assert planning.plan is not None

    class RetrievedDocumentPerLane:
        identity = {"name": "recording", "type": "test", "hits": 1}

        def __init__(self) -> None:
            self.calls = 0

        @property
        def transport_calls(self):
            return self.calls

        def cache_summary(self):
            return {"hits": 0, "misses": self.calls}

        def retrieve(self, request: QueryVariant):
            self.calls += 1
            return (
                RetrievedCandidate(
                    request.topic_id,
                    request.variant_name,
                    "recording",
                    request.query_text,
                    f"doc-unscored-{self.calls}",
                    1,
                    3.0,
                    f"Retrieved but unscored evidence {self.calls}.",
                ),
            )

    class AlwaysFailingPassageScorer:
        identity = _offline_shared_scorer(tmp_path).identity
        stats = {"cache_hits": 0, "cache_misses": 0, "model_batches": 0}

        @staticmethod
        def cache_key(query_text: str, passage_text: str) -> str:
            return sha256(f"{query_text}\0{passage_text}".encode()).hexdigest()

        @staticmethod
        def rank(query_text, chunks):
            assert query_text and chunks
            raise PassageScoringFailed("offline scoring failure")

    class CandidateScorer:
        identity = mixedbread_sentence_scorer_identity()
        accounting = SimpleNamespace(
            cache_hits=0,
            cache_misses=0,
            model_batches=0,
        )

        def score_pairs(self, pairs):
            return tuple(1.0 for _ in pairs)

    class Similarity:
        identity = {"model": "offline-similarity"}
        accounting = SimpleNamespace(
            cache_hits=0,
            cache_misses=0,
            model_batches=0,
        )

        def cosine_matrix(self, texts):
            return tuple(
                tuple(1.0 if left == right else 0.0 for right in texts)
                for left in texts
            )

    class CanonicalBackend:
        def complete(self, request):
            return BackendReply(
                content=b'{"claims":[]}',
                response_body=b'{"offline":true}',
                status=200,
                metadata={
                    "requested_model": "deepseek/deepseek-v4-flash-20260423",
                    "response_model": "deepseek/deepseek-v4-flash-20260423",
                    "provider": "offline",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

    def fake_decompose(topic_arg, output_dir, _backend, **_kwargs):
        path = Path(output_dir) / topic_arg.id / "decomposition" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {
                    "schema_version": "facet_pilot_v2",
                    "topic": {
                        "id": topic_arg.id,
                        "narrative": topic_arg.narrative,
                    },
                    "narrative_sha256": sha256(
                        topic_arg.narrative.encode()
                    ).hexdigest(),
                    "used_fallback": planning.used_fallback,
                    "error": planning.error,
                    "queries": jsonable(planning.queries),
                    "plan": _plan_payload(planning.plan),
                    "subnarratives": jsonable(planning.subnarratives),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        return load_validated_decomposition(topic_arg, path), False

    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decompose_topic", fake_decompose
    )
    monkeypatch.setattr(
        "trec_rag.competition_retrieval._decomposition_producer_sha256",
        lambda *_args, **_kwargs: "e" * 64,
    )
    retriever = RetrievedDocumentPerLane()
    with pytest.raises(ValueError, match="no supported document"):
        _run_topic(
            topic,
            config,
            _topics_sha256((topic,)),
            _RuntimeDependencies(
                code_commit="a" * 40,
                document_scorer=AlwaysFailingPassageScorer(),
                candidate_scorer=CandidateScorer(),
                similarity=Similarity(),
                cache_ignore_checker=lambda _path: True,
                retriever=retriever,
                canonical_backend_factory=CanonicalBackend,
            ),
            config_sha256="b" * 64,
            expected_retriever_identity=retriever.identity,
        )

    topic_root = config.output_dir / topic.id
    assert retriever.calls == 2
    assert (topic_root / "scoring" / "lane_scores.jsonl").read_bytes() == b""
    assert (topic_root / "scoring" / "selected_documents.jsonl").read_bytes() == b""
    assert not (topic_root / "canonical" / "retrieval-projection.json").exists()
    assert not (topic_root / "canonical" / "generation-projection.json").exists()


def test_score_phase_projects_sealed_passages_without_retriever_or_scorer(tmp_path: Path) -> None:
    topic = Topic("topic-1", "title", "find source")
    query = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    decomposition = ValidatedDecomposition(
        topic.id,
        sha256(topic.narrative.encode()).hexdigest(),
        "d" * 64,
        FacetPlanningResult((query,), True, "fallback", None, ()),
    )
    text = "source document body"
    digest = sha256(text.encode()).hexdigest()
    passage_identity = _configured_passage_search_identity(
        retrieval_depth=1,
        passages_per_query=1,
        chunk_max_characters=3500,
        chunk_overlap_characters=350,
        model=MIXEDBREAD_MODEL,
        device="cpu",
    )
    result = PassageSearchResult(
        FocusedQuery("original", topic.narrative, "original"),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (SourceDocument("doc-1", digest, 1, 4.0, "p-doc-1-0", 2.5),),
        (
            SourcePassage(
                "p-doc-1-0",
                "doc-1",
                digest,
                1,
                4.0,
                0,
                len(text),
                0,
                len(text.encode()),
                digest,
                text,
                2.5,
                1,
                "cache-key",
                digest,
                passage_identity["chunker"],
            ),
        ),
        1,
        False,
    )

    class FixedPassageSearch:
        def search(self, focused_query: FocusedQuery) -> PassageSearchResult:
            assert focused_query == result.query
            return result

        def read_text(self, content_sha256: str) -> str:
            assert content_sha256 == digest
            return text

    class Poison:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"score phase touched poison dependency: {name}")

    store_root = tmp_path / "objects"
    DocumentStore(store_root).admit_text(text, expected_sha256=digest)
    retriever_identity = {"name": "fixed", "type": "test", "hits": 1}
    _retrieve_topic(
        topic,
        decomposition,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "retrieval-cache",
        code_commit="a" * 40,
        corpus_epoch="test-epoch",
        retriever=type("IdentityOnlyRetriever", (), {"identity": retriever_identity})(),
        retrieval_depth=1,
        passage_search=FixedPassageSearch(),
        passage_identity=passage_identity,
    )

    bundle = json.loads(
        (tmp_path / "output" / topic.id / "retrieval" / "evidence-bundle.json").read_text()
    )
    document_row = bundle["lanes"][0]["documents"][0]
    assert "text" not in document_row
    assert bundle["lanes"][0]["passages"][0]["text"] == text

    _score_topic(
        topic,
        decomposition,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "retrieval-cache",
        score_cache_root=tmp_path / "score-cache",
        code_commit="a" * 40,
        corpus_epoch="test-epoch",
        retriever=Poison(),
        scorer=Poison(),
        device="cpu",
        retrieval_depth=1,
        rerank_depth=1,
        selection_k=1,
        document_store_root=store_root,
        expected_retriever_identity=retriever_identity,
        expected_passage_identity=passage_identity,
    )


def _passage_handoff_fixture(tmp_path: Path):
    topic = Topic("topic-handoff", "title", "find facet evidence")
    facets = (
        Subnarrative(topic.id, "facet-a", "first facet", ("first",)),
        Subnarrative(topic.id, "facet-b", "second facet", ("second",)),
    )
    original = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    queries = (
        original,
        QueryVariant(topic.id, "subnarrative:facet-a", "first facet", "semantic_subnarrative"),
        QueryVariant(topic.id, "subnarrative:facet-b", "second facet", "semantic_subnarrative"),
    )
    decomposition = ValidatedDecomposition(
        topic.id,
        sha256(topic.narrative.encode()).hexdigest(),
        "d" * 64,
        FacetPlanningResult(
            queries,
            False,
            None,
            GeneratedQueryPlan(topic.id, facets),
            facets,
        ),
    )
    scoring = tmp_path / "pilot" / topic.id / "scoring"
    scoring.mkdir(parents=True)
    selected_rows = []
    for rank in (1, 2):
        selected_text = f"selected body {rank}."
        selected_rows.append({
            "topic_id": topic.id,
            "docid": f"selected-{rank}",
            "selection_rank": rank,
            "selected_from_lane": "original",
            "selected_from_lane_rank": rank,
            "text_sha256": sha256(selected_text.encode()).hexdigest(),
            "text": selected_text,
        })
    (scoring / "selected_documents.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected_rows)
    )
    for name in ("lane_scores.jsonl", "selection.json", "selected_subnarrative_scores.jsonl"):
        (scoring / name).write_text("{}\n" if name == "selection.json" else "")
    artifacts = []
    for relative in (
        "scoring/lane_scores.jsonl",
        "scoring/selected_documents.jsonl",
        "scoring/selection.json",
        "scoring/selected_subnarrative_scores.jsonl",
    ):
        body = (tmp_path / "pilot" / topic.id / relative).read_bytes()
        artifacts.append({
            "relative_path": relative,
            "bytes": len(body),
            "sha256": sha256(body).hexdigest(),
        })
    selected_hash = sha256(
        json.dumps([row["docid"] for row in selected_rows], separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "facet_pilot_v2",
        "selection_schema_version": "facet_pilot_selection_v2",
        "phase": "score",
        "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": "a" * 40,
        "retriever": {"name": "fake", "type": "test", "hits": 1},
        "scorer": {"model": MIXEDBREAD_MODEL},
        "rerank_depth": 1,
        "selection_k": 1,
        "selection_policy": "round_robin_lane_order_no_fusion",
        "selection_scope": "internal_fixed_path_projection_not_final_submission",
        "score_policy": {},
        "passage_search": {},
        "retrieval_manifest_sha256": "b" * 64,
        "selected_set_sha256": selected_hash,
        "artifacts": artifacts,
    }
    (scoring / "complete.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    source = "facet passage source."
    source_digest = sha256(source.encode()).hexdigest()
    store_root = tmp_path / "objects"
    DocumentStore(store_root).admit_text(source, expected_sha256=source_digest)
    passage_results = []
    for query_id, facet_id, source_rank, source_score, logit, passage_id in (
        ("subnarrative:facet-a", "facet-a", 9, 1.0, 4.0, "p-facet-a"),
        ("subnarrative:facet-b", "facet-b", 2, 8.0, 3.0, "p-facet-b"),
    ):
        passage_results.append(PassageSearchResult(
            FocusedQuery(query_id, facet_id.replace("-", " "), facet_id),
            "complete",
            None,
            1,
            1,
            1,
            1,
            (SourceDocument("facet-doc", source_digest, source_rank, source_score, passage_id, logit),),
            (SourcePassage(
                passage_id, "facet-doc", source_digest, source_rank, source_score,
                0, len(source), 0, len(source.encode()), source_digest, source,
                logit, 1, f"cache-{facet_id}", source_digest,
                {"backend": "test"},
            ),),
            1,
            False,
        ))
    return topic, decomposition, passage_results, store_root


def test_canonical_handoff_uses_all_semantic_passages_not_selected_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topic, decomposition, passage_results, store_root = _passage_handoff_fixture(tmp_path)
    monkeypatch.setattr(
        "trec_rag.evidence_store._subnarrative_scores",
        lambda *args, **kwargs: pytest.fail("v2 handoff must not build a score matrix"),
    )
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=tmp_path / "pilot",
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
        passage_results=passage_results,
        document_store_root=store_root,
    )
    request_bytes = handoff.requests_path.read_bytes()
    assert b"facet passage source." not in request_bytes
    rows = [json.loads(line) for line in request_bytes.splitlines()]
    assert all("source" not in row for row in rows)
    assert {row["schema_version"] for row in rows} == {
        "extractive_candidate_request_v2"
    }
    assert {row["document_id"] for row in rows} == {"facet-doc"}
    assert {row["passages"][0]["passage_id"] for row in rows} == {
        "p-facet-a", "p-facet-b"
    }
    handoff_manifest = json.loads(handoff.manifest_path.read_text())
    assert handoff_manifest["document_count"] == 1
    assert (
        handoff_manifest["request_schema_version"]
        == "extractive_candidate_request_v2"
    )

    class FixedCandidateScorer:
        identity = mixedbread_sentence_scorer_identity()

        def score_pairs(self, pairs):
            return tuple(1.0 for _ in pairs)

    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=store_root,
        scorer=FixedCandidateScorer(),
        facets=(
            FacetRecord("original", topic.narrative, "initial"),
            FacetRecord("facet-a", "first facet", "initial"),
            FacetRecord("facet-b", "second facet", "initial"),
        ),
        passage_results=passage_results,
    )
    assert artifacts.records_path.is_file()
    with TopicRecords.open(
        artifacts.records_path,
        artifacts.manifest_path,
        topic.id,
        DocumentStore(store_root),
        validation_session=artifacts.validation_session,
    ) as records:
        candidates = records.load_candidates()
        snapshot = records.topic_snapshot()
    assert {candidate.subnarrative_id for candidate in candidates.values()} == {
        "facet-a",
        "facet-b",
    }
    assert (snapshot.status, snapshot.stopping_reason) == (
        "complete",
        "coverage_sufficient",
    )


def test_compact_passage_request_rejects_geometry_tamper(tmp_path: Path) -> None:
    topic, decomposition, passage_results, store_root = _passage_handoff_fixture(
        tmp_path
    )
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=tmp_path / "pilot",
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
        passage_results=passage_results,
        document_store_root=store_root,
    )
    rows = [json.loads(line) for line in handoff.requests_path.read_text().splitlines()]
    rows[0]["passages"][0]["scoring_end_char"] -= 1
    handoff.requests_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )

    with pytest.raises(ValueError, match="scoring geometry"):
        generate_candidate_artifacts(
            handoff,
            run_id="test-run",
            score_cache_root=tmp_path / "score-cache",
            device="cpu",
            document_store_root=store_root,
            passage_results=passage_results,
        )


def test_canonical_handoff_allows_zero_evidence_incomplete_lane(tmp_path: Path) -> None:
    topic, decomposition, _, store_root = _passage_handoff_fixture(tmp_path)
    incomplete = PassageSearchResult(
        FocusedQuery("subnarrative:facet-a", "facet a", "facet-a"),
        "incomplete",
        "no_evidence",
        1,
        0,
        0,
        0,
        (),
        (),
        1,
        True,
    )
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=tmp_path / "pilot",
        output_dir=tmp_path / topic.id / "canonical" / "handoff-zero",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
        passage_results=(incomplete,),
        document_store_root=store_root,
    )
    assert handoff.requests_path.read_text() == ""
    assert json.loads(handoff.manifest_path.read_text())["document_count"] == 0

    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=store_root,
        facets=(FacetRecord("facet-a", "facet a", "initial"),),
        passage_results=(incomplete,),
    )
    with TopicRecords.open(
        artifacts.records_path,
        artifacts.manifest_path,
        topic.id,
        DocumentStore(store_root),
        validation_session=artifacts.validation_session,
    ) as records:
        snapshot = records.topic_snapshot()
    assert (snapshot.status, snapshot.stopping_reason) == (
        "incomplete",
        "no_evidence",
    )


def test_fixed_retrieval_completion_preserves_scoring_failure_across_lanes(
    tmp_path: Path,
) -> None:
    topic, decomposition, _, store_root = _passage_handoff_fixture(tmp_path)
    no_evidence = PassageSearchResult(
        FocusedQuery("subnarrative:facet-a", "facet a", "facet-a"),
        "incomplete",
        "no_evidence",
        1,
        0,
        0,
        0,
        (),
        (),
        1,
        True,
    )
    scoring_failed = PassageSearchResult(
        FocusedQuery("subnarrative:facet-b", "facet b", "facet-b"),
        "incomplete",
        "scoring_failed",
        1,
        0,
        0,
        0,
        (),
        (),
        1,
        True,
    )
    passage_results = (no_evidence, scoring_failed)
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=tmp_path / "pilot",
        output_dir=tmp_path / topic.id / "canonical" / "handoff-failure-priority",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
        passage_results=passage_results,
        document_store_root=store_root,
    )

    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=store_root,
        facets=(
            FacetRecord("facet-a", "facet a", "initial"),
            FacetRecord("facet-b", "facet b", "initial"),
        ),
        passage_results=passage_results,
    )

    with TopicRecords.open(
        artifacts.records_path,
        artifacts.manifest_path,
        topic.id,
        DocumentStore(store_root),
        validation_session=artifacts.validation_session,
    ) as records:
        snapshot = records.topic_snapshot()
    assert (snapshot.status, snapshot.stopping_reason) == (
        "incomplete",
        "scoring_failed",
    )

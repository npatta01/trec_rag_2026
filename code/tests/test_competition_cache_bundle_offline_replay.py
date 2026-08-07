from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from trec_rag import competition_cache_bundle as bundle_module
from trec_rag import competition_retrieval
from trec_rag.competition_cache_bundle import (
    CacheBundleIntegrityError,
    merge_bundles,
    pack_bundle,
    verify_bundle,
)
from trec_rag.evidence_local import (
    LocalMiniLMSimilarity,
    MixedbreadSentencePairScorer,
)
from trec_rag.facet_extraction import (
    BackendReply,
    FacetResponse,
    OpenRouterDeepSeekFacetBackend,
)
from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.remote_client import RemoteSearchResponse
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.topic_dispatch import (
    TopicJob,
    TopicJobReceipt,
    publish_topic_receipt,
)


_ENDPOINT = "https://pyserini.test/v1/climbmix-400b/search"
_CODE_COMMIT = "a" * 40
_TOPIC_ID = "rag2026-0"
_NARRATIVE = "Compare how housing policy changes affect tenants."
_OTHER_TOPIC_ID = "rag2026-1"
_OTHER_NARRATIVE = "Explain how zoning reform changes housing supply."


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_config(
    root: Path,
    *,
    experiment_id: str = "online-cache-fixture",
    include_other_topic: bool = False,
) -> Path:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("fixture boundary\n", encoding="utf-8")
    topics = f"{_TOPIC_ID}\t{_NARRATIVE}\n"
    if include_other_topic:
        topics += f"{_OTHER_TOPIC_ID}\t{_OTHER_NARRATIVE}\n"
    (root / "topics.tsv").write_text(topics, encoding="utf-8")
    path = root / "config.yaml"
    path.write_text(
        f"""\
schema_version: facet_pilot_config_v2
experiment:
  id: {experiment_id}
topics:
  path: topics.tsv
retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  query_sources: [original, subnarrative]
  documents_per_query: 1000
  corpus_epoch: fixture-epoch
passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350
nuggets:
  evidence_budget_per_subnarrative: 2
  maximum_claims_per_subnarrative: 1
  maximum_supporting_documents_per_claim: 1
execution:
  topic_workers: 1
""",
        encoding="utf-8",
    )
    return path


def _plan() -> dict[str, object]:
    return {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": _TOPIC_ID,
        "subnarratives": [
            {
                "subnarrative": "Tenant effects of housing policy changes",
                "bm25_queries": ["housing policy tenant effects"],
            }
        ],
    }


def _planning_envelope() -> bytes:
    return json.dumps(
        {
            "id": "planning-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek/deepseek-v4-flash-20260423",
            "provider": "fixture-provider",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(_plan()),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _PlanningTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def send(self, _request: object) -> FacetResponse:
        self.calls += 1
        if self.fail:
            return FacetResponse(503, b'{"error":"fixture"}')
        return FacetResponse(200, _planning_envelope())


class _RetrievalClient:
    config = RemotePyseriniConfig(_ENDPOINT, None, 1000, ())

    def __init__(self, *, forbid: bool = False) -> None:
        self.forbid = forbid
        self.calls = 0

    def search_raw(self, query_text: str, *, raw_sink=None) -> RemoteSearchResponse:
        if self.forbid:
            raise AssertionError("offline replay reached retrieval transport")
        self.calls += 1
        suffix = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:12]
        raw = json.dumps(
            {
                "api": "v1",
                "index": "climbmix-400b",
                "query": {"text": query_text},
                "candidates": [
                    {
                        "doc": (
                            f"Evidence for {query_text}. "
                            "A second supported sentence adds detail. "
                            "A third sentence provides useful context."
                        ),
                        "docid": f"doc-{suffix}",
                        "rank": 1,
                        "score": 7.5,
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        if raw_sink is not None:
            raw_sink(raw)
        return RemoteSearchResponse(
            raw=raw,
            payload=json.loads(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


class _FakeCrossEncoder:
    @staticmethod
    def parameters():
        return (SimpleNamespace(dtype="torch.bfloat16"),)

    @staticmethod
    def predict(pairs, **_kwargs):
        return [float(len(pairs) - index) for index, _pair in enumerate(pairs)]


def _cross_encoder_loader(*_args, **_kwargs) -> _FakeCrossEncoder:
    return _FakeCrossEncoder()


def _forbidden_model_loader(*_args, **_kwargs):
    raise AssertionError("offline replay attempted to load a model")


class _FakeEmbeddingModel:
    @staticmethod
    def encode(texts, **_kwargs):
        count = len(texts)
        return np.eye(count, dtype=np.float64)


def _embedding_loader(*_args, **_kwargs) -> _FakeEmbeddingModel:
    return _FakeEmbeddingModel()


class _CanonicalBackend:
    calls = 0
    fail = False

    @property
    def transport_invocation_count(self) -> int:
        return type(self).calls

    def complete(self, request) -> BackendReply:
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("canonical fixture failure")
        alias = request.evidence[0].alias
        return BackendReply(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "claim": "Housing policy changes can affect tenants.",
                            "evidence_aliases": [alias],
                            "importance": "vital",
                        }
                    ]
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            response_body=b'{"fixture":true}',
            status=200,
            metadata={
                "requested_model": "deepseek/deepseek-v4-flash-20260423",
                "response_model": "deepseek/deepseek-v4-flash-20260423",
                "provider": "fixture-provider",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


def _prepare_online_topic(
    root: Path,
    *,
    planning_failure: bool = False,
    canonical_failure: bool = False,
    include_other_topic: bool = False,
) -> tuple[Path, FacetPilotConfig, object, dict[str, object], str]:
    config_path = _write_config(root, include_other_topic=include_other_topic)
    config_bytes = config_path.read_bytes()
    config = load_facet_pilot_config(config_path, source_bytes=config_bytes)
    topic = select_configured_topics(config, topic_ids=(_TOPIC_ID,))[0]
    planning = OpenRouterDeepSeekFacetBackend(
        environ={"OPENROUTER_API_KEY": "fixture-only"},
        transport=_PlanningTransport(fail=planning_failure),
    )
    retrieval_client = _RetrievalClient()
    retriever = competition_retrieval.build_pyserini_retriever(
        config.retrieval.cache_dir,
        index=config.retrieval.index,
        hits=config.retrieval.documents_per_query,
        corpus_epoch=config.retrieval.corpus_epoch,
        client=retrieval_client,
    )
    expected_retriever_identity = competition_retrieval._retriever_identity(
        retriever,
        retrieval_depth=config.retrieval.documents_per_query,
    )
    passage_scorer = MixedbreadPassageScorer(
        config.passage.score_cache_dir,
        device="cpu",
        model_loader=_cross_encoder_loader,
    )
    candidate_scorer = MixedbreadSentencePairScorer(
        score_cache_root=config.passage.score_cache_dir,
        device="cpu",
        model_loader=_cross_encoder_loader,
    )
    similarity = LocalMiniLMSimilarity(
        device="cpu",
        loader=_embedding_loader,
        cache_root=root / "cache",
    )
    _CanonicalBackend.calls = 0
    _CanonicalBackend.fail = canonical_failure
    dependencies = competition_retrieval._RuntimeDependencies(
        code_commit=_CODE_COMMIT,
        document_scorer=passage_scorer,
        candidate_scorer=candidate_scorer,
        similarity=similarity,
        cache_ignore_checker=lambda _path: True,
        planning_backend=planning,
        retriever=retriever,
        canonical_backend_factory=_CanonicalBackend,
    )
    official_topics_sha256 = competition_retrieval._topics_sha256(
        select_configured_topics(config)
    )
    try:
        outcome = competition_retrieval._run_topic(
            topic,
            config,
            official_topics_sha256,
            dependencies,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            expected_retriever_identity=expected_retriever_identity,
        )
    finally:
        passage_scorer.score_cache.close()
        candidate_scorer.score_cache.close()
    status, stopping_reason = competition_retrieval._topic_completion_for_dispatch(
        config,
        topic,
    )
    job = TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path.resolve(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        topic_root=(config.output_dir / topic.id).resolve(),
    )
    publish_topic_receipt(
        job,
        TopicJobReceipt(
            topic_id=topic.id,
            projection_manifest_sha256=(outcome.projection_receipt.manifest_sha256),
            status=status,
            stopping_reason=stopping_reason,
        ),
    )
    return (
        config_path,
        config,
        topic,
        expected_retriever_identity,
        official_topics_sha256,
    )


def _run_merged_offline_replay(
    *,
    root: Path,
    source_config: FacetPilotConfig,
    topic: object,
    expected_retriever_identity: dict[str, object],
    official_topics_sha256: str,
    cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    offline_root = root / "offline-replay"
    offline_root.mkdir()
    (offline_root / "AGENTS.md").write_text("fixture boundary\n", encoding="utf-8")
    topics_path = offline_root / "topics.tsv"
    topics_path.write_text(
        f"{_TOPIC_ID}\t{_NARRATIVE}\n",
        encoding="utf-8",
    )
    offline_config = replace(
        source_config,
        root_dir=offline_root,
        experiment=replace(source_config.experiment, id="offline-cache-replay"),
        topics_path=topics_path,
        retrieval=replace(
            source_config.retrieval,
            cache_dir=cache_root / "retrieval" / "pyserini_remote",
        ),
        passage=replace(
            source_config.passage,
            score_cache_dir=cache_root / "reranker",
        ),
    )
    config_path = offline_root / "config.yaml"
    config_bytes = _write_config(
        offline_root / "config-source",
        experiment_id="offline-cache-replay",
    ).read_bytes()
    config_path.write_bytes(config_bytes)
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", str(cache_root))
    retriever = competition_retrieval.build_pyserini_retriever(
        offline_config.retrieval.cache_dir,
        index=offline_config.retrieval.index,
        hits=offline_config.retrieval.documents_per_query,
        corpus_epoch=offline_config.retrieval.corpus_epoch,
        client=_RetrievalClient(forbid=True),
        cache_only=True,
    )
    passage_scorer = MixedbreadPassageScorer(
        offline_config.passage.score_cache_dir,
        device="cpu",
        model_loader=_forbidden_model_loader,
        read_only=True,
    )
    candidate_scorer = MixedbreadSentencePairScorer(
        score_cache_root=offline_config.passage.score_cache_dir,
        device="cpu",
        model_loader=_forbidden_model_loader,
        read_only=True,
    )
    similarity = LocalMiniLMSimilarity(
        device="cpu",
        loader=_forbidden_model_loader,
        cache_root=cache_root,
        cache_only=True,
    )
    dependencies = competition_retrieval._RuntimeDependencies(
        code_commit=_CODE_COMMIT,
        document_scorer=passage_scorer,
        candidate_scorer=candidate_scorer,
        similarity=similarity,
        cache_ignore_checker=None,
        planning_backend=None,
        retriever=retriever,
        canonical_backend_factory=None,
    )
    job = TopicJob(
        topic_id=_TOPIC_ID,
        run_id=offline_config.run_id,
        config_path=config_path.resolve(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        topic_root=(offline_config.output_dir / _TOPIC_ID).resolve(),
        offline_cache_only=True,
    )
    try:
        outcome = competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            offline_config,
            official_topics_sha256,
            dependencies,
            config_sha256=job.config_sha256,
            expected_retriever_identity=expected_retriever_identity,
        )
    finally:
        passage_scorer.score_cache.close()
        candidate_scorer.score_cache.close()
    assert outcome.topic_id == _TOPIC_ID
    return job.topic_root


def test_online_bundle_merge_supports_a_fresh_zero_work_offline_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config_path,
        config,
        topic,
        expected_retriever_identity,
        official_topics_sha256,
    ) = _prepare_online_topic(tmp_path / "online")
    bundle = (tmp_path / "bundle").resolve()

    packed = pack_bundle(config_path, _TOPIC_ID, bundle)
    verified = verify_bundle(bundle)
    assert verified.archive_sha256 == packed.archive_sha256
    merged_cache = (tmp_path / "merged-cache").resolve()
    merge_bundles(
        cache_root=merged_cache,
        outputs_root=(tmp_path / "merged-outputs").resolve(),
        bundle_dirs=(bundle,),
    )
    topic_root = _run_merged_offline_replay(
        root=tmp_path,
        source_config=config,
        topic=topic,
        expected_retriever_identity=expected_retriever_identity,
        official_topics_sha256=official_topics_sha256,
        cache_root=merged_cache,
        monkeypatch=monkeypatch,
    )

    operation = json.loads(
        (topic_root / "cache-operation-receipt.json").read_text(encoding="utf-8")
    )
    assert operation["mode"] == "offline-cache-only"
    for counters in operation["stages"].values():
        assert counters["cache_misses"] == 0
        assert counters["network_calls"] == 0
        assert counters["provider_calls"] == 0
        assert counters["model_batches"] == 0


def test_pack_validation_uses_the_production_staged_offline_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _config, _topic, _identity, _topics_sha = _prepare_online_topic(
        tmp_path / "online"
    )
    original = competition_retrieval._run_offline_topic_staged
    calls: list[tuple[TopicJob, Path]] = []

    def traced(job, topic, config, identity, dependencies, **kwargs):
        calls.append((job, config.root_dir))
        return original(
            job,
            topic,
            config,
            identity,
            dependencies,
            **kwargs,
        )

    monkeypatch.setattr(
        competition_retrieval,
        "_run_offline_topic_staged",
        traced,
    )

    pack_bundle(config_path, _TOPIC_ID, (tmp_path / "bundle").resolve())

    assert len(calls) == 1
    job, replay_root = calls[0]
    assert job.offline_cache_only is True
    assert (
        job.topic_root == replay_root / "outputs" / "online-cache-fixture" / _TOPIC_ID
    )


def test_pack_self_verification_preserves_full_official_topics_identity(
    tmp_path: Path,
) -> None:
    config_path, _config, _topic, _identity, _topics_sha = _prepare_online_topic(
        tmp_path / "online",
        include_other_topic=True,
    )
    bundle = (tmp_path / "bundle").resolve()

    packed = pack_bundle(config_path, _TOPIC_ID, bundle)

    verified = verify_bundle(bundle)
    assert verified.archive_sha256 == packed.archive_sha256
    topics_member = next(
        member for member in verified.members if member.kind == "topics"
    )
    assert topics_member.path == "source-config/official-topics"
    assert topics_member.sha256 == hashlib.sha256(
        (tmp_path / "online/topics.tsv").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("control", ("\t", "\n", "\r"))
def test_offline_replay_topic_rejects_tsv_control_characters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    config_path = _write_config(tmp_path / "repo")
    config = load_facet_pilot_config(
        config_path,
        source_bytes=config_path.read_bytes(),
    )
    result_path = config.output_dir / _TOPIC_ID / "decomposition" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(
        _canonical_json(
            {
                "topic": {
                    "id": _TOPIC_ID,
                    "narrative": f"unsafe{control}narrative",
                }
            }
        )
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_validated_decomposition",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(CacheBundleIntegrityError, match="TSV control"):
        bundle_module._offline_replay_topic(config, _TOPIC_ID)


@pytest.mark.parametrize(
    "namespace",
    (
        "planning-cache-v1",
        "retrieval",
        "reranker",
        "similarity-cache-v1",
        "canonical",
    ),
)
def test_pack_rejects_each_missing_required_cache_stage(
    tmp_path: Path,
    namespace: str,
) -> None:
    config_path, _config, _topic, _identity, _topics_sha = _prepare_online_topic(
        tmp_path / "online"
    )
    cache_namespace = tmp_path / "online" / "cache" / namespace
    assert cache_namespace.exists(), namespace
    shutil.rmtree(cache_namespace)
    destination = (tmp_path / "bundle").resolve()

    with pytest.raises(CacheBundleIntegrityError):
        pack_bundle(config_path, _TOPIC_ID, destination)

    assert tuple(destination.iterdir()) == ()


@pytest.mark.parametrize("failure", ("planning", "canonical"))
def test_pack_rejects_online_fallback_without_a_replayable_validated_cache(
    tmp_path: Path,
    failure: str,
) -> None:
    online_root = tmp_path / "online"
    if failure == "planning":
        with pytest.raises(ValueError, match="has no supported document"):
            _prepare_online_topic(online_root, planning_failure=True)
        config_path = online_root / "config.yaml"
        assert not (online_root / "cache" / "planning-cache-v1").exists()
    else:
        config_path, _config, _topic, _identity, _topics_sha = _prepare_online_topic(
            online_root,
            canonical_failure=True,
        )
    destination = (tmp_path / "bundle").resolve()

    with pytest.raises(CacheBundleIntegrityError):
        pack_bundle(config_path, _TOPIC_ID, destination)

    assert tuple(destination.iterdir()) == ()

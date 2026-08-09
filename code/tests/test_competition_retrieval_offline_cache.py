from __future__ import annotations

from dataclasses import replace
import errno
from hashlib import sha256
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from trec_rag import competition_retrieval
from trec_rag.document_store import DocumentStore, DocumentStoreIntegrityError
from trec_rag.facet_extraction import planning_cache_identity
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.planning_cache import PlanningCache
from trec_rag.retrieval_export import build_topic_projection
from trec_rag.topics import Topic
from trec_rag.topic_dispatch import TopicJob, TopicJobReceipt
from trec_rag.topic_records import TopicRecords

from test_retrieval_export import (
    _FixtureRetriever,
    _config_and_topics,
    _write_sealed_topic,
)


ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"


def _config(tmp_path: Path):
    base = load_facet_pilot_config(V2_CONFIG)
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text("topic-a\tcache replay narrative\n", encoding="utf-8")
    return replace(
        base,
        root_dir=tmp_path,
        topics_path=topics_path,
        experiment=replace(base.experiment, id="offline-replay"),
        retrieval=replace(base.retrieval, cache_dir=tmp_path / "cache" / "retrieval"),
        passage=replace(
            base.passage,
            score_cache_dir=tmp_path / "cache" / "reranker",
            device="cpu",
        ),
    )


def _job(
    config,
    *,
    execution_policy: str = "offline-cache-only",
) -> TopicJob:
    config_bytes = V2_CONFIG.read_bytes()
    return TopicJob(
        topic_id="topic-a",
        run_id=config.run_id,
        config_path=V2_CONFIG.resolve(),
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=(config.output_dir / "topic-a").resolve(),
        execution_policy=execution_policy,
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    if not root.exists():
        return ()
    rows: list[tuple[str, str, bytes | None]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            rows.append((relative, "symlink", str(path.readlink()).encode()))
        elif path.is_dir():
            rows.append((relative, "directory", None))
        else:
            rows.append((relative, "file", path.read_bytes()))
    return tuple(rows)


def _fake_projection(topic: Topic, config) -> SimpleNamespace:
    body = b"offline projection"
    path = (
        config.output_dir
        / topic.id
        / "canonical"
        / "retrieval-projection-manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return SimpleNamespace(
        topic_id=topic.id,
        resumed=False,
        projection_receipt=SimpleNamespace(
            topic_id=topic.id,
            manifest_sha256=sha256(body).hexdigest(),
        ),
    )


def _zero_operation_stages() -> dict[str, dict[str, int]]:
    return {
        stage: {
            "cache_hits": 0,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for stage in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
    }


def test_cli_forwards_offline_cache_only(monkeypatch, capsys) -> None:
    """Catches parsing the flag but launching the normal online runner."""
    captured: dict[str, object] = {}

    def fake_run(config, **kwargs):
        captured.update(config=config, **kwargs)
        return SimpleNamespace(
            retrieval_export=SimpleNamespace(manifest=Path("/tmp/out/complete.json"))
        )

    monkeypatch.setattr(competition_retrieval, "run_official", fake_run)

    assert competition_retrieval.main([str(V2_CONFIG), "--offline-cache-only"]) == 0

    assert captured["offline_cache_only"] is True
    assert capsys.readouterr().out == "output=/tmp/out\n"


def test_cli_forwards_cached_upstream_rescore(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(config, **kwargs):
        captured.update(config=config, **kwargs)
        return SimpleNamespace(
            retrieval_export=SimpleNamespace(manifest=Path("/tmp/out/complete.json"))
        )

    monkeypatch.setattr(competition_retrieval, "run_official", fake_run)

    assert (
        competition_retrieval.main([str(V2_CONFIG), "--cached-upstream-rescore"])
        == 0
    )

    assert captured["cached_upstream_rescore"] is True
    assert capsys.readouterr().out == "output=/tmp/out\n"


def test_cli_rejects_both_cache_execution_modes() -> None:
    with pytest.raises(SystemExit):
        competition_retrieval.main(
            [
                str(V2_CONFIG),
                "--offline-cache-only",
                "--cached-upstream-rescore",
            ]
        )


def test_runner_refuses_incomplete_merge_before_dependency_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a partially merged cache reaching client or model factories."""
    config = _config(tmp_path)
    monkeypatch.setattr(
        competition_retrieval, "load_facet_pilot_config", lambda *_a, **_k: config
    )

    def reject_merge(_cache_root):
        raise RuntimeError("incomplete cache bundle merge")

    monkeypatch.setattr(
        competition_retrieval,
        "assert_no_incomplete_cache_bundle_merge",
        reject_merge,
    )

    with pytest.raises(RuntimeError, match="incomplete cache bundle merge"):
        competition_retrieval._run_official(
            V2_CONFIG,
            topic_ids=("topic-a",),
            topic_subset=None,
            external=None,
            offline_cache_only=True,
            dependency_factory=lambda: pytest.fail(
                "merge refusal must precede dependency construction"
            ),
        )


@pytest.mark.parametrize(
    ("execution_policy", "sentence_read_only", "similarity_cache_only"),
    (
        ("offline-cache-only", True, True),
        ("cached-upstream-rescore", False, False),
    ),
)
def test_cache_policy_production_worker_constructs_expected_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_policy: str,
    sentence_read_only: bool,
    similarity_cache_only: bool,
) -> None:
    """Catches a cache policy silently constructing a forbidden adapter."""
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config, execution_policy=execution_policy)
    calls: dict[str, object] = {"guards": 0}
    base_dependencies = competition_retrieval._RuntimeDependencies(
        code_commit="c" * 40,
        document_scorer=None,
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
    )

    monkeypatch.setattr(
        competition_retrieval,
        "load_facet_pilot_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_narrative_topics",
        lambda _path: (topic,),
    )
    def production_dependencies(**kwargs):
        calls["dependency_kwargs"] = kwargs
        return base_dependencies

    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        production_dependencies,
    )

    def guard(_cache_root):
        calls["guards"] = int(calls["guards"]) + 1

    monkeypatch.setattr(
        competition_retrieval,
        "assert_no_incomplete_cache_bundle_merge",
        guard,
    )

    retriever = SimpleNamespace(
        identity={"name": "fake", "type": "test", "hits": 1000},
        cache_summary=lambda: {"hits": 0, "misses": 0},
        transport_calls=0,
    )

    def build_retriever(*_args, **kwargs):
        calls["retriever"] = kwargs
        return retriever

    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        build_retriever,
    )

    passage = SimpleNamespace(
        identity={"passage": "identity"},
        stats={},
        score_cache=SimpleNamespace(close=lambda: None),
    )
    sentence = SimpleNamespace(
        identity={"sentence": "identity"},
        accounting=SimpleNamespace(),
        score_cache=SimpleNamespace(close=lambda: None),
    )
    similarity = SimpleNamespace(
        identity={"similarity": "identity"}, accounting=SimpleNamespace()
    )

    def passage_factory(*_args, **kwargs):
        calls["passage"] = kwargs
        return passage

    def sentence_factory(*_args, **kwargs):
        calls["sentence"] = kwargs
        return sentence

    def similarity_factory(*_args, **kwargs):
        calls["similarity"] = kwargs
        return similarity

    monkeypatch.setattr(
        competition_retrieval, "MixedbreadPassageScorer", passage_factory
    )
    monkeypatch.setattr(
        competition_retrieval,
        "MixedbreadSentencePairScorer",
        sentence_factory,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "LocalMiniLMSimilarity",
        similarity_factory,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_retriever_identity",
        lambda *_args, **_kwargs: retriever.identity,
    )

    def fake_run_topic(topic_arg, staged_config, _identity, dependencies, **kwargs):
        assert topic_arg == topic
        assert (staged_config.root_dir != config.root_dir) is (
            execution_policy == "offline-cache-only"
        )
        assert dependencies.retriever is retriever
        assert dependencies.document_scorer is passage
        assert dependencies.candidate_scorer is sentence
        assert dependencies.similarity is similarity
        assert kwargs["execution_policy"] == execution_policy
        body = b"offline projection"
        path = (
            staged_config.output_dir
            / topic.id
            / "canonical"
            / "retrieval-projection-manifest.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return SimpleNamespace(
            topic_id=topic.id,
            projection_receipt=SimpleNamespace(
                topic_id=topic.id,
                manifest_sha256=sha256(body).hexdigest(),
            ),
        )

    monkeypatch.setattr(competition_retrieval, "_run_topic", fake_run_topic)
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda *_args: ("complete", "coverage_sufficient"),
    )

    receipt = competition_retrieval._run_production_topic_job(job)

    assert receipt.topic_id == topic.id
    assert calls["guards"] == 1
    assert calls["retriever"]["cache_only"] is True
    assert calls["passage"]["read_only"] is True
    assert calls["sentence"]["read_only"] is sentence_read_only
    assert calls["similarity"]["cache_only"] is similarity_cache_only
    assert calls["dependency_kwargs"] == (
        {"load_environment": False}
        if execution_policy == "offline-cache-only"
        else {}
    )
    assert calls["similarity"]["cache_root"] == tmp_path / "cache"
    assert (
        job.topic_root / "canonical" / "retrieval-projection-manifest.json"
    ).is_file()


def test_offline_decomposition_requires_the_planning_cache_without_a_backend(
    tmp_path: Path,
) -> None:
    """Catches a resumed/fresh offline decomposition reaching hosted planning."""
    topic = Topic("topic-a", "", "cache replay narrative")
    payload = {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": topic.id,
        "subnarratives": [
            {"subnarrative": "cached facet", "bm25_queries": ["cached query"]}
        ],
    }
    PlanningCache(tmp_path / "cache").store(planning_cache_identity(topic), payload)
    stats: dict[str, int] = {}

    decomposition, resumed = competition_retrieval._decompose_topic(
        topic,
        tmp_path / "outputs",
        None,
        planning_cache_root=tmp_path / "cache",
        cache_only=True,
        cache_stats=stats,
    )

    assert resumed is False
    assert decomposition.result.plan is not None
    assert decomposition.result.plan.subnarratives[0].text == "cached facet"
    assert stats == {
        "cache_hits": 1,
        "cache_misses": 0,
        "backend_calls": 0,
        "provider_calls": 0,
    }


def test_offline_topic_and_root_receipts_authenticate_zero_work_and_order(
    tmp_path: Path,
) -> None:
    """Catches unauthenticated counters, wrong topic order, or non-zero offline work."""
    config = _config(tmp_path)
    topics = (
        Topic("topic-a", "", "first"),
        Topic("topic-b", "", "second"),
    )
    config_sha256 = "a" * 64
    stages_a = {
        name: {
            "cache_hits": 1,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for name in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
    }
    stages_b = {
        name: {
            "cache_hits": 2,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for name in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
    }
    receipts = tuple(
        competition_retrieval._publish_topic_cache_operation_receipt(
            config=config,
            topic=topic,
            config_sha256=config_sha256,
            projection_manifest_sha256=("b" if topic.id == "topic-a" else "c") * 64,
            mode="offline-cache-only",
            phases={
                "planning": {"resumed": False},
                "retrieval": {"resumed": False},
                "scoring": {"resumed": False},
                "canonical": {"resumed": False},
            },
            stages=stages,
        )
        for topic, stages in zip(topics, (stages_a, stages_b), strict=True)
    )
    export_manifest = config.output_dir / "generation_handoff_manifest.json"
    export_manifest.parent.mkdir(parents=True, exist_ok=True)
    export_manifest.write_bytes(b"sealed export\n")

    manifest_path = competition_retrieval._publish_cache_operation_manifest(
        config=config,
        topics=topics,
        config_sha256=config_sha256,
        mode="offline-cache-only",
        receipts=receipts,
        export_manifest_path=export_manifest,
    )

    manifest = competition_retrieval._loads(
        manifest_path.read_bytes(), "cache operation manifest"
    )
    assert manifest["topic_ids"] == ["topic-a", "topic-b"]
    assert manifest["topic_receipt_sha256s"] == [row.sha256 for row in receipts]
    assert manifest["totals"] == {
        name: {
            "cache_hits": 3,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for name in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
    }
    content = dict(manifest)
    digest = content.pop("receipt_content_sha256")
    assert (
        digest
        == sha256(competition_retrieval._json_bytes(content, pretty=False)).hexdigest()
    )

    bad_stages = {name: dict(values) for name, values in stages_a.items()}
    bad_stages["retrieval"]["network_calls"] = 1
    with pytest.raises(ValueError, match="zero misses, calls, and model batches"):
        competition_retrieval._publish_topic_cache_operation_receipt(
            config=config,
            topic=Topic("topic-c", "", "third"),
            config_sha256=config_sha256,
            projection_manifest_sha256="d" * 64,
            mode="offline-cache-only",
            phases={
                "planning": {"resumed": False},
                "retrieval": {"resumed": False},
                "scoring": {"resumed": False},
                "canonical": {"resumed": False},
            },
            stages=bad_stages,
        )
    assert not (config.output_dir / "topic-c" / "cache-operation-receipt.json").exists()


def test_offline_document_admission_writes_only_to_the_private_stage(
    tmp_path: Path,
) -> None:
    """Catches cache replay creating temp/CAS files in the merged shared cache."""
    source_root = tmp_path / "shared-documents"
    text = "cached source document"
    receipt = DocumentStore(source_root).admit_text(text)
    before = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    stage_root = tmp_path / "stage-documents"
    store = competition_retrieval._OfflineStagingDocumentStore(
        source_root=source_root,
        stage_root=stage_root,
    )

    admitted = store.admit_text(text, expected_sha256=receipt.content_sha256)

    after = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    assert admitted == receipt
    assert after == before
    assert store.read_text(receipt.content_sha256) == text
    assert DocumentStore(stage_root).read_text(receipt.content_sha256) == text


@pytest.mark.parametrize("cache_hit", [True, False], ids=("hit", "miss"))
def test_offline_stage_uses_versioned_source_cas_and_bypasses_cache_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_hit: bool,
) -> None:
    """Catches /documents vs /documents/v1 and env-rerouted staging writes."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    config = _config(checkout)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
    shared_cache = tmp_path / "merged-cache"
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", str(tmp_path / "poison-cache"))
    source_store = DocumentStore(shared_cache / "documents" / "v1")
    source_text = "cached source document"
    source_receipt = source_store.admit_text(source_text)
    before = _tree_snapshot(shared_cache)

    def fake_run_topic(topic_arg, staged_config, _identity, _dependencies, **kwargs):
        assert kwargs["offline_source_document_store_root"] == (
            shared_cache / "documents" / "v1"
        )
        stage_document_root = kwargs["offline_stage_document_store_root"]
        assert (
            stage_document_root == staged_config.root_dir / "cache" / "documents" / "v1"
        )
        assert not stage_document_root.is_relative_to(shared_cache)
        assert kwargs["runtime_cache_root"] == staged_config.root_dir / "cache"
        store = competition_retrieval._OfflineStagingDocumentStore(
            source_root=kwargs["offline_source_document_store_root"],
            stage_root=stage_document_root,
        )
        if cache_hit:
            store.admit_text(source_text, expected_sha256=source_receipt.content_sha256)
            return _fake_projection(topic_arg, staged_config)
        store.admit_text("uncached document")
        raise AssertionError("an uncached document was admitted")

    monkeypatch.setattr(competition_retrieval, "_run_topic", fake_run_topic)
    dependencies = competition_retrieval._RuntimeDependencies(
        code_commit="c" * 40,
        document_scorer=None,
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
    )

    if cache_hit:
        outcome = competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            config,
            "d" * 64,
            dependencies,
            config_sha256=job.config_sha256,
            expected_retriever_identity={"type": "test"},
            source_cache_root=shared_cache,
        )
        assert outcome.topic_id == topic.id
        assert job.topic_root.is_dir()
    else:
        with pytest.raises(DocumentStoreIntegrityError, match="document"):
            competition_retrieval._run_offline_topic_staged(
                job,
                topic,
                config,
                "d" * 64,
                dependencies,
                config_sha256=job.config_sha256,
                expected_retriever_identity={"type": "test"},
                source_cache_root=shared_cache,
            )
        assert not job.topic_root.exists()

    assert _tree_snapshot(shared_cache) == before


def test_offline_stage_rejects_a_shared_cache_that_contains_its_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches an absolute cache override turning private staging into cache state."""
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        lambda topic_arg, staged_config, *_args, **_kwargs: _fake_projection(
            topic_arg,
            staged_config,
        ),
    )

    with pytest.raises(ValueError, match="cache.*output.*overlap"):
        competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            config,
            "d" * 64,
            competition_retrieval._RuntimeDependencies(
                code_commit="c" * 40,
                document_scorer=None,
                candidate_scorer=None,
                similarity=None,
                cache_ignore_checker=None,
            ),
            config_sha256=job.config_sha256,
            expected_retriever_identity={"type": "test"},
        )

    assert not job.topic_root.exists()
    assert not any(
        path.name.startswith(".offline-cache-") for path in tmp_path.iterdir()
    )


def test_offline_stage_recovers_a_fully_validated_published_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a crash after create-only topic publication but before dispatch."""
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
    job.topic_root.mkdir(parents=True)
    projection = SimpleNamespace(topic_id=topic.id, manifest_sha256="e" * 64)
    validated = TopicJobReceipt(
        topic_id=topic.id,
        projection_manifest_sha256=projection.manifest_sha256,
        status="complete",
        stopping_reason="coverage_sufficient",
    )
    captured: dict[str, object] = {}

    def validate(*_args, **kwargs):
        captured.update(kwargs)
        return validated

    monkeypatch.setattr(
        competition_retrieval,
        "_validated_existing_topic_job_receipt",
        validate,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        lambda *_args: projection,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        lambda *_args, **_kwargs: pytest.fail("validated topic was rerun"),
    )

    outcome = competition_retrieval._run_offline_topic_staged(
        job,
        topic,
        config,
        "d" * 64,
        competition_retrieval._RuntimeDependencies(
            code_commit="c" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        ),
        config_sha256=job.config_sha256,
        expected_retriever_identity={"type": "test"},
    )

    assert outcome.resumed is True
    assert outcome.projection_receipt is projection
    assert captured["operation_mode"] == "offline-cache-only"
    assert captured["allow_missing_operation_receipt"] is False


def test_offline_stage_reopens_a_real_sealed_topic_tree(tmp_path: Path) -> None:
    """Exercises projection, TopicRecords, source-chain, and operation validation."""
    config, topics = _config_and_topics(tmp_path)
    topic = topics[0]
    _write_sealed_topic(
        config.output_dir,
        topic,
        selected=("doc-a",),
        supported=("doc-a",),
        source_commit="a" * 40,
    )
    decomposition_path = config.output_dir / topic.id / "decomposition" / "result.json"
    decomposition_bytes = decomposition_path.read_bytes()
    (decomposition_path.parent / "manifest.json").write_bytes(
        competition_retrieval._json_bytes(
            {
                "schema_version": "facet-decomposition-manifest-v1",
                "planner": competition_retrieval._planner_identity(None),
                "result_file": "result.json",
                "result_bytes": len(decomposition_bytes),
                "result_sha256": sha256(decomposition_bytes).hexdigest(),
            }
        )
    )
    producer_sha256 = competition_retrieval._decomposition_producer_sha256(
        topic,
        config.output_dir,
        None,
    )
    config_bytes = b"real sealed offline recovery fixture\n"
    config_sha256 = sha256(config_bytes).hexdigest()
    canonical_complete = config.output_dir / topic.id / "canonical" / "complete.json"
    provisional_manifest = canonical_complete.read_bytes()
    canonical_complete.unlink()
    with TopicRecords.open(
        config.output_dir / topic.id / "records.sqlite3",
        config.output_dir / topic.id / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(competition_retrieval.document_store_dir(config.root_dir)),
    ) as records:
        projection = build_topic_projection(
            config,
            topic,
            records,
            expected_retriever_identity=dict(_FixtureRetriever.identity),
            decomposition_producer_sha256=producer_sha256,
            config_sha256=config_sha256,
            canonical_manifest_bytes=provisional_manifest,
        )
    competition_retrieval._publish_topic_cache_operation_receipt(
        config=config,
        topic=topic,
        config_sha256=config_sha256,
        projection_manifest_sha256=projection.manifest_sha256,
        mode="offline-cache-only",
        phases={
            phase: {"resumed": False}
            for phase in competition_retrieval._CACHE_OPERATION_PHASE_NAMES
        },
        stages=_zero_operation_stages(),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(config_bytes)
    job = TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path.resolve(),
        config_bytes=config_bytes,
        config_sha256=config_sha256,
        topic_root=(config.output_dir / topic.id).resolve(),
        execution_policy="offline-cache-only",
    )

    outcome = competition_retrieval._run_offline_topic_staged(
        job,
        topic,
        config,
        competition_retrieval._topics_sha256(topics),
        competition_retrieval._RuntimeDependencies(
            code_commit="a" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        ),
        config_sha256=config_sha256,
        expected_retriever_identity=dict(_FixtureRetriever.identity),
    )

    assert outcome.resumed is True
    assert outcome.projection_receipt == projection


def test_offline_stage_refuses_a_partial_existing_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
    job.topic_root.mkdir(parents=True)
    monkeypatch.setattr(
        competition_retrieval,
        "_validated_existing_topic_job_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("corrupt topic")),
    )

    with pytest.raises(ValueError, match="corrupt topic"):
        competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            config,
            "d" * 64,
            competition_retrieval._RuntimeDependencies(
                code_commit="c" * 40,
                document_scorer=None,
                candidate_scorer=None,
                similarity=None,
                cache_ignore_checker=None,
            ),
            config_sha256=job.config_sha256,
            expected_retriever_identity={"type": "test"},
        )


def test_offline_publication_preserves_a_concurrent_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches os.rename replacing a destination directory created by a racer."""
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)

    def fake_run_topic(topic_arg, staged_config, *_args, **_kwargs):
        outcome = _fake_projection(topic_arg, staged_config)
        job.topic_root.mkdir(parents=True)
        return outcome

    monkeypatch.setattr(competition_retrieval, "_run_topic", fake_run_topic)

    with pytest.raises(ValueError, match="appeared during publication"):
        competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            config,
            "d" * 64,
            competition_retrieval._RuntimeDependencies(
                code_commit="c" * 40,
                document_scorer=None,
                candidate_scorer=None,
                similarity=None,
                cache_ignore_checker=None,
            ),
            config_sha256=job.config_sha256,
            expected_retriever_identity={"type": "test"},
        )

    assert job.topic_root.is_dir()
    assert tuple(job.topic_root.iterdir()) == ()


def test_offline_create_only_publication_fsyncs_tree_and_both_rename_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "private-stage"
    source = source_parent / "topic-a"
    nested = source / "canonical"
    nested.mkdir(parents=True)
    root_file = source / "records.sqlite3"
    nested_file = nested / "complete.json"
    root_file.write_bytes(b"sealed records")
    nested_file.write_bytes(b"sealed manifest")
    destination = tmp_path / "published" / "topic-a"
    destination.parent.mkdir()
    fsynced: list[tuple[Path, bool]] = []

    def record_fsync(descriptor: int) -> None:
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        fsynced.append((path, stat.S_ISDIR(os.fstat(descriptor).st_mode)))

    monkeypatch.setattr(os, "fsync", record_fsync)

    competition_retrieval._publish_directory_create_only(
        source,
        destination,
        managed_root=tmp_path,
    )

    assert not source.exists()
    assert (destination / "records.sqlite3").read_bytes() == b"sealed records"
    assert (destination / "canonical" / "complete.json").read_bytes() == (
        b"sealed manifest"
    )
    synced_paths = [path for path, _is_directory in fsynced]
    for required in (root_file, nested_file, nested, source):
        assert required in synced_paths
    assert synced_paths.index(nested_file) < synced_paths.index(nested)
    assert synced_paths.index(nested) < synced_paths.index(source)
    assert synced_paths[-2:] == [destination.parent, source_parent]
    assert dict(fsynced)[root_file] is False
    assert dict(fsynced)[nested_file] is False
    assert dict(fsynced)[nested] is True
    assert dict(fsynced)[source] is True
    assert all(
        path == tmp_path or path.is_relative_to(tmp_path) for path in synced_paths
    )


def test_offline_create_only_publication_durably_creates_destination_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "private-stage"
    source = source_parent / "topic-a"
    source.mkdir(parents=True)
    (source / "sealed.txt").write_text("sealed", encoding="utf-8")
    published = tmp_path / "published"
    experiment = published / "experiment"
    destination = experiment / "topic-a"
    fsynced: list[Path] = []

    def record_fsync(descriptor: int) -> None:
        fsynced.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))

    monkeypatch.setattr(os, "fsync", record_fsync)

    try:
        competition_retrieval._publish_directory_create_only(
            source,
            destination,
            managed_root=tmp_path,
        )
    except FileNotFoundError as exc:
        pytest.fail(f"publication did not create destination ancestors: {exc}")

    assert (destination / "sealed.txt").read_text(encoding="utf-8") == "sealed"
    assert tmp_path in fsynced
    assert published in fsynced
    assert experiment in fsynced


def test_offline_create_only_publication_accepts_a_concurrent_parent_creator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "private-stage" / "topic-a"
    source.mkdir(parents=True)
    (source / "sealed.txt").write_text("sealed", encoding="utf-8")
    published = tmp_path / "published"
    destination = published / "topic-a"
    original_mkdir = Path.mkdir
    raced = False

    def race_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal raced
        if path == published and not raced:
            raced = True
            original_mkdir(path, *args, **kwargs)
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race_mkdir)

    competition_retrieval._publish_directory_create_only(
        source,
        destination,
        managed_root=tmp_path,
    )

    assert raced is True
    assert (destination / "sealed.txt").read_text(encoding="utf-8") == "sealed"


def test_durable_mkdirs_fsyncs_a_parent_when_child_appears_before_lstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = tmp_path / "published"
    original_lstat = Path.lstat
    observed: list[Path] = []
    raced = False

    def race_lstat(path: Path) -> os.stat_result:
        nonlocal raced
        if path == published and not raced:
            raced = True
            published.mkdir()
        return original_lstat(path)

    def record_fsync(descriptor: int) -> None:
        observed.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))

    monkeypatch.setattr(Path, "lstat", race_lstat)
    monkeypatch.setattr(os, "fsync", record_fsync)

    competition_retrieval._durable_mkdirs(published, managed_root=tmp_path)

    assert raced is True
    assert tmp_path in observed
    assert tmp_path.parent not in observed


def test_offline_publication_exdev_leaves_no_topic_or_private_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        lambda topic_arg, staged_config, *_args, **_kwargs: _fake_projection(
            topic_arg,
            staged_config,
        ),
    )

    def reject_cross_device(_source: Path, _destination: Path) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(
        competition_retrieval,
        "_rename_directory_noreplace",
        reject_cross_device,
    )

    with pytest.raises(OSError) as caught:
        competition_retrieval._run_offline_topic_staged(
            job,
            topic,
            config,
            "d" * 64,
            competition_retrieval._RuntimeDependencies(
                code_commit="c" * 40,
                document_scorer=None,
                candidate_scorer=None,
                similarity=None,
                cache_ignore_checker=None,
            ),
            config_sha256=job.config_sha256,
            expected_retriever_identity={"type": "test"},
        )

    assert caught.value.errno == errno.EXDEV
    assert not job.topic_root.exists()
    assert not any(
        path.name.startswith(".offline-cache-") for path in tmp_path.iterdir()
    )


def test_offline_publication_without_renameat2_preserves_the_private_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "private-stage" / "topic-a"
    source.mkdir(parents=True)
    (source / "sealed.txt").write_text("sealed", encoding="utf-8")
    destination = tmp_path / "published" / "topic-a"
    destination.parent.mkdir()
    monkeypatch.setattr(
        competition_retrieval.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="requires Linux renameat2"):
        competition_retrieval._publish_directory_create_only(
            source,
            destination,
            managed_root=tmp_path,
        )

    assert (source / "sealed.txt").read_text(encoding="utf-8") == "sealed"
    assert not destination.exists()


def test_operation_accounting_rejects_a_dependency_without_exact_counters() -> None:
    """Catches an uninstrumented adapter being reported as machine-verified zero work."""
    dependencies = competition_retrieval._RuntimeDependencies(
        code_commit="c" * 40,
        document_scorer=None,
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
        retriever=SimpleNamespace(identity={"type": "uninstrumented"}),
    )

    with pytest.raises(TypeError, match="retriever must expose exact cache accounting"):
        competition_retrieval._runtime_cache_accounting(dependencies)

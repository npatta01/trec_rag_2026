from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag import competition_retrieval
from trec_rag.document_store import DocumentStore, DocumentStoreIntegrityError
from trec_rag.facet_extraction import planning_cache_identity
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.planning_cache import PlanningCache
from trec_rag.topics import Topic
from trec_rag.topic_dispatch import TopicJob, TopicJobReceipt


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


def _job(config, *, offline_cache_only: bool = True) -> TopicJob:
    config_bytes = V2_CONFIG.read_bytes()
    return TopicJob(
        topic_id="topic-a",
        run_id=config.run_id,
        config_path=V2_CONFIG.resolve(),
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=(config.output_dir / "topic-a").resolve(),
        offline_cache_only=offline_cache_only,
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


def test_runner_refuses_incomplete_merge_before_dependency_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a partially merged cache reaching client or model factories."""
    config = _config(tmp_path)
    monkeypatch.setattr(competition_retrieval, "load_facet_pilot_config", lambda *_a, **_k: config)

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


def test_offline_production_worker_constructs_only_read_only_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches any offline worker boundary silently constructing online adapters."""
    config = _config(tmp_path)
    topic = Topic("topic-a", "", "cache replay narrative")
    job = _job(config)
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
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda **_kwargs: base_dependencies,
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

    passage = SimpleNamespace(identity={"passage": "identity"}, stats={})
    sentence = SimpleNamespace(identity={"sentence": "identity"}, accounting=SimpleNamespace())
    similarity = SimpleNamespace(identity={"similarity": "identity"}, accounting=SimpleNamespace())

    def passage_factory(*_args, **kwargs):
        calls["passage"] = kwargs
        return passage

    def sentence_factory(*_args, **kwargs):
        calls["sentence"] = kwargs
        return sentence

    def similarity_factory(*_args, **kwargs):
        calls["similarity"] = kwargs
        return similarity

    monkeypatch.setattr(competition_retrieval, "MixedbreadPassageScorer", passage_factory)
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
        assert staged_config.root_dir != config.root_dir
        assert dependencies.retriever is retriever
        assert dependencies.document_scorer is passage
        assert dependencies.candidate_scorer is sentence
        assert dependencies.similarity is similarity
        assert kwargs["offline_cache_only"] is True
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
    assert calls["sentence"]["read_only"] is True
    assert calls["similarity"]["cache_only"] is True
    assert calls["similarity"]["cache_root"] == tmp_path / "cache"
    assert (job.topic_root / "canonical" / "retrieval-projection-manifest.json").is_file()


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
    assert digest == sha256(
        competition_retrieval._json_bytes(content, pretty=False)
    ).hexdigest()

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
    assert not (
        config.output_dir / "topic-c" / "cache-operation-receipt.json"
    ).exists()


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
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", str(shared_cache))
    source_store = DocumentStore(shared_cache / "documents" / "v1")
    source_text = "cached source document"
    source_receipt = source_store.admit_text(source_text)
    before = _tree_snapshot(shared_cache)

    def fake_run_topic(topic_arg, staged_config, _identity, _dependencies, **kwargs):
        assert kwargs["offline_source_document_store_root"] == (
            shared_cache / "documents" / "v1"
        )
        stage_document_root = kwargs["offline_stage_document_store_root"]
        assert stage_document_root == staged_config.root_dir / "cache" / "documents" / "v1"
        assert not stage_document_root.is_relative_to(shared_cache)
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
            )
        assert not job.topic_root.exists()

    assert _tree_snapshot(shared_cache) == before


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


def test_offline_create_only_publication_fsyncs_the_destination_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sealed.txt").write_text("sealed", encoding="utf-8")
    destination = tmp_path / "published" / "topic-a"
    destination.parent.mkdir()
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", fsynced.append)

    competition_retrieval._publish_directory_create_only(source, destination)

    assert not source.exists()
    assert (destination / "sealed.txt").read_text(encoding="utf-8") == "sealed"
    assert len(fsynced) == 1


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

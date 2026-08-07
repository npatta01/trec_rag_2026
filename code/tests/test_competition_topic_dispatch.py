from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag import competition_retrieval
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.topics import Topic
from trec_rag.topic_dispatch import (
    TopicJob,
    TopicJobReceipt,
    dispatch_topics,
    read_topic_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"

_PROCESS_CONFIG = None
_PROCESS_TOPIC = None
_PROCESS_CONFIG_BYTES = b""
_PROCESS_MARKER_ROOT: Path | None = None


def _process_load_config(path: Path, *, source_bytes: bytes | None = None):
    assert source_bytes == _PROCESS_CONFIG_BYTES
    assert Path(path).is_absolute()
    assert _PROCESS_CONFIG is not None
    return _PROCESS_CONFIG


def _process_topics(_path: Path):
    assert _PROCESS_TOPIC is not None
    return (_PROCESS_TOPIC,)


def _write_process_marker(name: str) -> None:
    assert _PROCESS_MARKER_ROOT is not None
    _PROCESS_MARKER_ROOT.mkdir(parents=True, exist_ok=True)
    (_PROCESS_MARKER_ROOT / name).write_text(str(os.getpid()), encoding="utf-8")


def _process_dependencies():
    _write_process_marker("dependencies.pid")
    return competition_retrieval._RuntimeDependencies(
        code_commit="c" * 40,
        document_scorer=None,
        candidate_scorer=None,
        similarity=None,
        cache_ignore_checker=None,
    )


def _process_retriever(*_args, **_kwargs):
    _write_process_marker("retriever.pid")
    return SimpleNamespace(identity={"name": "fake", "type": "test", "hits": 1000})


def _process_run_topic(topic, config, _identity, _dependencies, **_kwargs):
    _write_process_marker("topic-pipeline.pid")
    body = b"offline process projection"
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
        projection_receipt=SimpleNamespace(
            topic_id=topic.id,
            manifest_sha256=sha256(body).hexdigest(),
        ),
    )


def _process_completion(_config, _topic):
    _write_process_marker("completion.pid")
    return "complete", "coverage_sufficient"


def test_checked_in_config_enables_two_topic_workers() -> None:
    config = load_facet_pilot_config(V2_CONFIG)

    assert config.execution.topic_workers == 2
    assert config.resolved_payload(())["execution"] == {"topic_workers": 2}


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_config_rejects_invalid_topic_worker_count(tmp_path: Path, value: object) -> None:
    text = V2_CONFIG.read_text(encoding="utf-8")
    rendered = f'"{value}"' if isinstance(value, str) else str(value).lower()
    text = text.replace(
        "execution:\n  topic_workers: 2",
        f"execution:\n  topic_workers: {rendered}",
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="execution.topic_workers"):
        load_facet_pilot_config(path)


def test_official_runner_dispatches_pending_topics_in_source_order_without_parent_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="dispatch-test"),
        topics_path=tmp_path / "topics.tsv",
    )
    topics = (
        Topic("topic-b", "", "second in lexical order, first in source order"),
        Topic("topic-a", "", "first in lexical order, second in source order"),
    )
    # Simulate a crash after every projection was published but before either
    # run-bound topic-job receipt was created. Projection presence alone must
    # not bypass dispatch receipt recovery.
    published: set[str] = {topic.id for topic in topics}
    projection_receipts = {
        topic.id: SimpleNamespace(
            topic_id=topic.id,
            manifest_sha256=("a" if topic.id == "topic-b" else "b") * 64,
        )
        for topic in topics
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(competition_retrieval, "load_facet_pilot_config", lambda _: config)
    monkeypatch.setattr(
        competition_retrieval,
        "select_configured_topics",
        lambda *_args, **_kwargs: topics,
    )
    monkeypatch.setattr(competition_retrieval, "load_narrative_topics", lambda _: topics)
    monkeypatch.setattr(competition_retrieval, "_tracked_worktree_is_dirty", lambda _: False)

    def read_projection(_config, topic):
        if topic.id not in published:
            raise ValueError("expanded canonical checkpoint is missing")
        return projection_receipts[topic.id]

    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        read_projection,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "validate_retrieval_topic_checkpoints",
        lambda _config, selected, **_kwargs: tuple(
            projection_receipts[topic.id] for topic in selected
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_decomposition_producer_sha256",
        lambda *_args, **_kwargs: "d" * 64,
    )
    fake_retriever = SimpleNamespace(
        identity={"name": "fake", "type": "test", "hits": 1000}
    )
    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        lambda *_args, **_kwargs: fake_retriever,
    )

    class PoisonScorer:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("parent constructed a Mixedbread scorer")

    monkeypatch.setattr(competition_retrieval, "MixedbreadPassageScorer", PoisonScorer)

    def fake_dependencies():
        return competition_retrieval._RuntimeDependencies(
            code_commit="c" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        )

    monkeypatch.setattr(competition_retrieval, "_production_dependencies", fake_dependencies)

    def fake_dispatch(jobs, worker, *, max_workers):
        captured["jobs"] = tuple(jobs)
        captured["worker"] = worker
        captured["max_workers"] = max_workers
        published.update(job.topic_id for job in jobs)
        return tuple(
            TopicJobReceipt(
                topic_id=job.topic_id,
                projection_manifest_sha256=projection_receipts[job.topic_id].manifest_sha256,
                status="complete",
                stopping_reason="coverage_sufficient",
            )
            for job in jobs
        )

    monkeypatch.setattr(competition_retrieval, "dispatch_topics", fake_dispatch)
    monkeypatch.setattr(
        competition_retrieval,
        "export_retrieval_run",
        lambda _config, selected, receipts, **_kwargs: captured.update(
            selected=tuple(selected), receipts=tuple(receipts)
        ),
    )
    export_receipt = SimpleNamespace(manifest=tmp_path / "complete.json")
    monkeypatch.setattr(
        competition_retrieval,
        "read_retrieval_export_receipt",
        lambda *_args, **_kwargs: export_receipt,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_read_topic_cache_operation_receipt",
        lambda **kwargs: SimpleNamespace(
            topic_id=kwargs["topic"].id,
            projection_manifest_sha256=kwargs["projection_manifest_sha256"],
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_publish_cache_operation_manifest",
        lambda **kwargs: captured.update(operation_manifest=kwargs),
    )

    result = competition_retrieval._run_official(
        V2_CONFIG,
        topic_ids=None,
        topic_subset=None,
        external=None,
        dependency_factory=competition_retrieval._production_dependencies,
    )

    jobs = captured["jobs"]
    assert [job.topic_id for job in jobs] == ["topic-b", "topic-a"]
    assert all(job.run_id == "dispatch-test" for job in jobs)
    assert all(job.config_bytes == V2_CONFIG.read_bytes() for job in jobs)
    assert captured["worker"] is competition_retrieval._run_production_topic_job
    assert captured["max_workers"] == 2
    assert [receipt.topic_id for receipt in captured["receipts"]] == [
        "topic-b",
        "topic-a",
    ]
    assert result.selected_topic_ids == ("topic-b", "topic-a")
    assert result.resumed_topic_ids == ()
    assert result.retrieval_export is export_receipt
    assert captured["operation_manifest"]["mode"] == "online"


def test_dispatch_receipt_rejects_an_unsealed_topic_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(base, root_dir=tmp_path)
    topic = Topic("topic-a", "", "narrative")

    class Records:
        run_id = config.run_id

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def topic_snapshot(self):
            return SimpleNamespace(status=None, stopping_reason=None)

    monkeypatch.setattr(
        competition_retrieval,
        "TopicRecords",
        SimpleNamespace(open=lambda *_args, **_kwargs: Records()),
    )
    monkeypatch.setattr(competition_retrieval, "DocumentStore", lambda _root: object())

    with pytest.raises(ValueError, match="completion state is not sealed"):
        competition_retrieval._topic_completion_for_dispatch(config, topic)


def test_dispatch_receipt_rejects_a_records_database_from_another_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="run-new"),
    )
    topic = Topic("topic-a", "", "narrative")

    class Records:
        run_id = "run-old"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def topic_snapshot(self):
            return SimpleNamespace(
                status="complete",
                stopping_reason="coverage_sufficient",
            )

    monkeypatch.setattr(
        competition_retrieval,
        "TopicRecords",
        SimpleNamespace(open=lambda *_args, **_kwargs: Records()),
    )
    monkeypatch.setattr(competition_retrieval, "DocumentStore", lambda _root: object())

    with pytest.raises(ValueError, match="run identity"):
        competition_retrieval._topic_completion_for_dispatch(config, topic)


def test_real_production_worker_crosses_process_boundary_with_pinned_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global _PROCESS_CONFIG, _PROCESS_TOPIC, _PROCESS_CONFIG_BYTES, _PROCESS_MARKER_ROOT

    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="process-test"),
        topics_path=tmp_path / "topics.tsv",
    )
    topic = Topic("topic-a", "", "narrative")
    config_path = (tmp_path / "config.yaml").resolve()
    pinned_bytes = b"pinned immutable config bytes\n"
    # The path deliberately disagrees. The worker must consume the bytes in
    # TopicJob instead of reloading this mutable path.
    config_path.write_bytes(b"changed same-path config\n")
    marker_root = tmp_path / "process-markers"
    _PROCESS_CONFIG = config
    _PROCESS_TOPIC = topic
    _PROCESS_CONFIG_BYTES = pinned_bytes
    _PROCESS_MARKER_ROOT = marker_root

    monkeypatch.setattr(
        competition_retrieval,
        "load_facet_pilot_config",
        _process_load_config,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_narrative_topics",
        _process_topics,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        _process_dependencies,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        _process_retriever,
    )
    monkeypatch.setattr(competition_retrieval, "_run_topic", _process_run_topic)
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        _process_completion,
    )

    job = TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path,
        config_bytes=pinned_bytes,
        config_sha256=sha256(pinned_bytes).hexdigest(),
        topic_root=(config.output_dir / topic.id).resolve(),
    )
    receipt = dispatch_topics(
        (job,),
        competition_retrieval._run_production_topic_job,
        max_workers=2,
    )[0]

    assert receipt.topic_id == topic.id
    child_pids = {
        int((marker_root / name).read_text(encoding="utf-8"))
        for name in (
            "dependencies.pid",
            "retriever.pid",
            "topic-pipeline.pid",
            "completion.pid",
        )
    }
    assert len(child_pids) == 1
    assert child_pids != {os.getpid()}


@pytest.mark.parametrize("topic_raises", [False, True], ids=("success", "topic-error"))
def test_production_worker_closes_both_score_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topic_raises: bool,
) -> None:
    """Catches worker exit while either scorer still owns a SQLite connection."""
    class TrackingCache:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    caches: list[TrackingCache] = []

    class TrackingScorer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.score_cache = TrackingCache()
            caches.append(self.score_cache)

    job = _patch_score_cache_lifecycle_worker(
        tmp_path,
        monkeypatch,
        passage_scorer_factory=TrackingScorer,
        candidate_scorer_factory=TrackingScorer,
    )
    projection = SimpleNamespace(topic_id=job.topic_id, manifest_sha256="a" * 64)

    def run_topic(*_args, **_kwargs):
        if topic_raises:
            raise RuntimeError("topic run failed")
        return SimpleNamespace(topic_id=job.topic_id, projection_receipt=projection)

    monkeypatch.setattr(competition_retrieval, "_run_topic", run_topic)

    if topic_raises:
        with pytest.raises(RuntimeError, match="topic run failed"):
            competition_retrieval._run_production_topic_job(job)
    else:
        receipt = competition_retrieval._run_production_topic_job(job)
        assert receipt.projection_manifest_sha256 == projection.manifest_sha256

    assert len(caches) == 2
    assert [cache.close_calls for cache in caches] == [1, 1]


def _patch_score_cache_lifecycle_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passage_scorer_factory,
    candidate_scorer_factory,
) -> TopicJob:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="score-cache-failure-lifecycle"),
        topics_path=tmp_path / "topics.tsv",
    )
    topic = Topic("topic-a", "", "narrative")
    config_bytes = b"pinned score cache failure lifecycle config\n"
    config_path = (tmp_path / "config.yaml").resolve()
    config_path.write_bytes(config_bytes)

    monkeypatch.setattr(
        competition_retrieval,
        "load_facet_pilot_config",
        lambda _path, *, source_bytes=None: config,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_narrative_topics",
        lambda _path: (topic,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda **_kwargs: competition_retrieval._RuntimeDependencies(
            code_commit="c" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        lambda *_args, **_kwargs: SimpleNamespace(
            identity={"name": "fake", "type": "test", "hits": 1000}
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_validated_existing_topic_job_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "MixedbreadPassageScorer",
        passage_scorer_factory,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "MixedbreadSentencePairScorer",
        candidate_scorer_factory,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "LocalMiniLMSimilarity",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        lambda *_args, **_kwargs: SimpleNamespace(
            topic_id=topic.id,
            projection_receipt=SimpleNamespace(
                topic_id=topic.id,
                manifest_sha256="a" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda *_args: ("complete", "coverage_sufficient"),
    )
    return TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path,
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=(config.output_dir / topic.id).resolve(),
    )


@pytest.mark.parametrize("failing_cache", ["passage", "sentence"])
def test_production_worker_attempts_both_score_cache_closes_when_one_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_cache: str,
) -> None:
    """Catches cleanup aborting after the first score-cache close failure."""
    close_calls = {"passage": 0, "sentence": 0}
    close_order: list[str] = []

    class TrackingCache:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            close_calls[self.name] += 1
            close_order.append(self.name)
            if self.name == failing_cache:
                raise RuntimeError(f"{self.name} score cache close failed")

    def scorer_factory(name: str):
        return lambda *_args, **_kwargs: SimpleNamespace(
            score_cache=TrackingCache(name)
        )

    job = _patch_score_cache_lifecycle_worker(
        tmp_path,
        monkeypatch,
        passage_scorer_factory=scorer_factory("passage"),
        candidate_scorer_factory=scorer_factory("sentence"),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"^{failing_cache} score cache close failed$",
    ):
        competition_retrieval._run_production_topic_job(job)

    assert close_order == ["sentence", "passage"]
    assert close_calls == {"passage": 1, "sentence": 1}


def test_production_worker_closes_first_score_cache_when_second_scorer_init_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches delayed cleanup registration leaking the first scorer's cache."""

    class TrackingCache:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    passage_cache = TrackingCache()
    passage_init_calls = 0
    sentence_init_calls = 0

    def passage_scorer(*_args, **_kwargs):
        nonlocal passage_init_calls
        passage_init_calls += 1
        return SimpleNamespace(score_cache=passage_cache)

    def failing_sentence_scorer(*_args, **_kwargs):
        nonlocal sentence_init_calls
        sentence_init_calls += 1
        raise RuntimeError("sentence scorer construction failed")

    job = _patch_score_cache_lifecycle_worker(
        tmp_path,
        monkeypatch,
        passage_scorer_factory=passage_scorer,
        candidate_scorer_factory=failing_sentence_scorer,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^sentence scorer construction failed$",
    ):
        competition_retrieval._run_production_topic_job(job)

    assert passage_init_calls == 1
    assert sentence_init_calls == 1
    assert passage_cache.close_calls == 1


@pytest.mark.parametrize("operation_exists", [True, False], ids=("sealed", "missing"))
def test_production_worker_recovers_projection_only_after_operation_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_exists: bool,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="projection-recovery"),
        topics_path=tmp_path / "topics.tsv",
    )
    topic = Topic("topic-a", "", "narrative")
    config_bytes = b"pinned recovery config\n"
    config_path = (tmp_path / "config.yaml").resolve()
    config_path.write_bytes(b"changed path contents\n")
    topic_root = (config.output_dir / topic.id).resolve()
    projection_body = b"sealed projection manifest"
    projection_path = (
        topic_root / "canonical" / "retrieval-projection-manifest.json"
    )
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.write_bytes(projection_body)
    producer_sha256 = "d" * 64
    projection = SimpleNamespace(
        topic_id=topic.id,
        manifest_sha256=sha256(projection_body).hexdigest(),
        source_seals=(
            ("config_sha256", sha256(config_bytes).hexdigest()),
            ("decomposition_producer_sha256", producer_sha256),
        ),
    )
    if operation_exists:
        competition_retrieval._publish_topic_cache_operation_receipt(
            config=config,
            topic=topic,
            config_sha256=sha256(config_bytes).hexdigest(),
            projection_manifest_sha256=projection.manifest_sha256,
            mode="online",
            phases={
                phase: {"resumed": True}
                for phase in ("planning", "retrieval", "scoring", "canonical")
            },
            stages={
                stage: {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "network_calls": 0,
                    "provider_calls": 0,
                    "model_batches": 0,
                }
                for stage in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
            },
        )

    monkeypatch.setattr(
        competition_retrieval,
        "load_facet_pilot_config",
        lambda _path, *, source_bytes=None: (
            config
            if source_bytes == config_bytes
            else pytest.fail("worker reloaded mutable config path")
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_narrative_topics",
        lambda _path: (topic,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: competition_retrieval._RuntimeDependencies(
            code_commit="c" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        lambda *_args, **_kwargs: SimpleNamespace(
            identity={"name": "fake", "type": "test", "hits": 1000}
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        lambda _config, _topic: projection,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "validate_retrieval_topic_checkpoints",
        lambda _config, _topics, **_kwargs: (projection,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_decomposition_producer_sha256",
        lambda *_args, **_kwargs: producer_sha256,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda _config, _topic: ("complete", "coverage_sufficient"),
    )
    runs: list[str] = []

    def rerun_topic(*_args, **_kwargs):
        runs.append(topic.id)
        if not operation_exists:
            competition_retrieval._publish_topic_cache_operation_receipt(
                config=config,
                topic=topic,
                config_sha256=sha256(config_bytes).hexdigest(),
                projection_manifest_sha256=projection.manifest_sha256,
                mode="online",
                phases={
                    phase: {"resumed": True}
                    for phase in ("planning", "retrieval", "scoring", "canonical")
                },
                stages={
                    stage: {
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "network_calls": 0,
                        "provider_calls": 0,
                        "model_batches": 0,
                    }
                    for stage in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
                },
            )
        return SimpleNamespace(topic_id=topic.id, projection_receipt=projection)

    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        rerun_topic,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "MixedbreadPassageScorer",
        lambda *_args, **_kwargs: SimpleNamespace(
            score_cache=SimpleNamespace(close=lambda: None)
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "MixedbreadSentencePairScorer",
        lambda *_args, **_kwargs: SimpleNamespace(
            score_cache=SimpleNamespace(close=lambda: None)
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "LocalMiniLMSimilarity",
        lambda *_args, **_kwargs: object(),
    )
    job = TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path,
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=topic_root,
    )

    receipt = dispatch_topics(
        (job,),
        competition_retrieval._run_production_topic_job,
        max_workers=1,
    )[0]

    assert receipt.projection_manifest_sha256 == projection.manifest_sha256
    assert read_topic_receipt(job) == receipt
    assert runs == ([] if operation_exists else [topic.id])


def test_parent_resume_refuses_dispatch_when_operation_receipt_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the parent accepting a job seal that root aggregation cannot verify."""
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(base, root_dir=tmp_path)
    topic = Topic("topic-a", "", "narrative")
    config_sha256 = "a" * 64
    producer_sha256 = "b" * 64
    projection = SimpleNamespace(
        topic_id=topic.id,
        manifest_sha256="c" * 64,
        source_seals=(
            ("config_sha256", config_sha256),
            ("decomposition_producer_sha256", producer_sha256),
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        lambda *_args: projection,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_decomposition_producer_sha256",
        lambda *_args, **_kwargs: producer_sha256,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "validate_retrieval_topic_checkpoints",
        lambda *_args, **_kwargs: (projection,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda *_args: ("complete", "coverage_sufficient"),
    )

    with pytest.raises(ValueError, match="cache operation topic receipt is missing"):
        competition_retrieval._validated_existing_topic_job_receipt(
            config,
            topic,
            expected_config_sha256=config_sha256,
            expected_retriever_identity={"type": "test"},
            planning_backend=None,
            operation_mode="online",
            allow_missing_operation_receipt=False,
        )

    assert (
        competition_retrieval._validated_existing_topic_job_receipt(
            config,
            topic,
            expected_config_sha256=config_sha256,
            expected_retriever_identity={"type": "test"},
            planning_backend=None,
            operation_mode="online",
            allow_missing_operation_receipt=True,
        )
        is None
    )


def test_production_worker_refuses_projection_recovery_from_other_config_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_facet_pilot_config(V2_CONFIG)
    config = replace(
        base,
        root_dir=tmp_path,
        experiment=replace(base.experiment, id="projection-config-mismatch"),
        topics_path=tmp_path / "topics.tsv",
    )
    topic = Topic("topic-a", "", "narrative")
    old_config_sha256 = sha256(b"exact config A\n").hexdigest()
    config_bytes = b"exact config B\n"
    config_path = (tmp_path / "config.yaml").resolve()
    config_path.write_bytes(config_bytes)
    projection = SimpleNamespace(
        topic_id=topic.id,
        manifest_sha256=sha256(b"sealed projection A").hexdigest(),
        source_seals=(("config_sha256", old_config_sha256),),
    )

    monkeypatch.setattr(
        competition_retrieval,
        "load_facet_pilot_config",
        lambda _path, *, source_bytes=None: config,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "load_narrative_topics",
        lambda _path: (topic,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: competition_retrieval._RuntimeDependencies(
            code_commit="c" * 40,
            document_scorer=None,
            candidate_scorer=None,
            similarity=None,
            cache_ignore_checker=None,
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "build_pyserini_retriever",
        lambda *_args, **_kwargs: SimpleNamespace(
            identity={"name": "fake", "type": "test", "hits": 1000}
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        lambda _config, _topic: projection,
    )
    monkeypatch.setattr(
        competition_retrieval,
        "validate_retrieval_topic_checkpoints",
        lambda _config, _topics, **_kwargs: (projection,),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda _config, _topic: ("complete", "coverage_sufficient"),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_run_topic",
        lambda *_args, **_kwargs: pytest.fail("mismatched projection was rerun"),
    )
    job = TopicJob(
        topic_id=topic.id,
        run_id=config.run_id,
        config_path=config_path,
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=(config.output_dir / topic.id).resolve(),
    )

    with pytest.raises(ValueError, match="projection config identity changed"):
        competition_retrieval._run_production_topic_job(job)

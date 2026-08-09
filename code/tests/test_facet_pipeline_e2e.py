from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.canonical_nuggets as canonical_nuggets
import trec_rag.facet_pilot_config as facet_pilot_config
import trec_rag.nuggetizer_adapter as nuggetizer_adapter
import trec_rag.competition_retrieval as competition_retrieval
from trec_rag.facet_extraction import BackendReply, planning_cache_identity
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.facet_retrieval import LaneDocumentScore, PassageScore
from trec_rag.competition_retrieval import (
    ExternalAdapters,
    RunReceipt,
    _configured_passage_search_identity,
    run_official,
)
from trec_rag.mixedbread_passage_scorer import ScoredPassage as MixedbreadScoredPassage
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.retrieval_export import RetrievalExportReceipt
from trec_rag.topic_dispatch import TopicDispatchError
from trec_rag.topics import Topic


def _write_config(
    root: Path,
    *,
    experiment_id: str = "official-interface-test",
    topics: tuple[str, ...] = ("topic-2", "topic-1"),
    extra: str = "",
) -> Path:
    (root / "topics.tsv").write_text(
        "".join(
            f"{topic_id}\tOfficial narrative for {topic_id}.\n" for topic_id in topics
        ),
        encoding="utf-8",
    )
    config = root / "official.yaml"
    config.write_text(
        f"""\
schema_version: facet_pilot_config_v2
experiment:
  id: {experiment_id}
topics:
  path: topics.tsv
retrieval:
  index: climbmix-test
  cache_dir: cache/retrieval
  query_sources: [original, subnarrative]
  documents_per_query: 1000
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
{extra}
""",
        encoding="utf-8",
    )
    return config


def test_config_uses_conventional_identity_and_binds_topic_source(
    tmp_path: Path,
) -> None:
    config = load_facet_pilot_config(
        _write_config(tmp_path, experiment_id="facet-b40-v1")
    )
    selected = facet_pilot_config.select_configured_topics(
        config,
        topic_ids=("topic-1",),
    )

    assert config.run_id == "facet-b40-v1"
    assert config.output_dir == tmp_path / "outputs" / "facet-b40-v1"
    assert config.resolved_payload(selected)["topics"] == {
        "path": "topics.tsv",
        "sha256": hashlib.sha256(
            b"topic-2\tOfficial narrative for topic-2.\n"
            b"topic-1\tOfficial narrative for topic-1.\n"
        ).hexdigest(),
        "selected_topic_ids": ["topic-1"],
    }


def test_cli_passes_repeated_topics_to_run_official_in_argument_order(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Catches CLI selector reordering or loss before the retrieval run starts."""
    calls: list[
        tuple[Path, tuple[str, ...] | None, Path | None, bool, bool]
    ] = []
    config_path = tmp_path / "competition.yaml"

    def run(
        config,
        *,
        topic_ids,
        topic_subset,
        offline_cache_only,
        cached_upstream_rescore,
    ):
        calls.append(
            (
                config,
                topic_ids,
                topic_subset,
                offline_cache_only,
                cached_upstream_rescore,
            )
        )
        return SimpleNamespace(
            retrieval_export=SimpleNamespace(
                manifest=tmp_path / "outputs" / "retrieval_export_manifest.json"
            )
        )

    monkeypatch.setattr(competition_retrieval, "run_official", run)

    assert (
        competition_retrieval.main(
            [str(config_path), "--topic", "rag2026-1", "--topic", "rag2026-0"]
        )
        == 0
    )

    assert calls == [
        (config_path, ("rag2026-1", "rag2026-0"), None, False, False)
    ]
    assert capsys.readouterr().out == f"output={tmp_path / 'outputs'}\n"


def test_config_routes_relative_reusable_caches_to_shared_checkout(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    worktree = tmp_path / "worktree"
    git_dir = shared / ".git" / "worktrees" / "facet-pilot"
    git_dir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (worktree / "cache" / "retrieval").mkdir(parents=True)
    (worktree / "cache" / "reranker").mkdir(parents=True)

    config = load_facet_pilot_config(_write_config(worktree))

    assert config.retrieval.cache_dir == shared / "cache" / "retrieval"
    assert config.passage.score_cache_dir == shared / "cache" / "reranker"
    assert competition_retrieval.document_store_dir(config.root_dir) == (
        shared / "cache" / "documents" / "v1"
    )


def test_config_rejects_reusable_cache_outside_shared_cache(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "cache_dir: cache/retrieval",
            f"cache_dir: {tmp_path / 'outputs' / 'response-cache'}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="beneath the repository cache"):
        load_facet_pilot_config(config_path)


@pytest.mark.parametrize(
    ("selector", "expected"),
    (
        ("all", ("topic-2", "topic-1")),
        ("ids", ("topic-2", "topic-1")),
        ("csv", ("topic-1",)),
    ),
)
def test_config_selects_all_ids_or_csv_in_official_order(
    tmp_path: Path,
    selector: str,
    expected: tuple[str, ...],
) -> None:
    config = load_facet_pilot_config(_write_config(tmp_path))
    subset = tmp_path / "subset.csv"
    subset.write_text("topic_id\ntopic-1\n", encoding="utf-8")
    kwargs: dict[str, object] = {}
    if selector == "ids":
        kwargs["topic_ids"] = ("topic-1", "topic-2")
    elif selector == "csv":
        kwargs["subset_csv"] = subset

    selected = facet_pilot_config.select_configured_topics(config, **kwargs)

    assert tuple(topic.id for topic in selected) == expected


def test_config_normalizes_programmatic_topic_id_whitespace(tmp_path: Path) -> None:
    config = load_facet_pilot_config(_write_config(tmp_path))

    selected = facet_pilot_config.select_configured_topics(
        config,
        topic_ids=(" topic-1 ", "\ttopic-2\n"),
    )

    assert tuple(topic.id for topic in selected) == ("topic-2", "topic-1")


def test_config_rejects_duplicate_unknown_and_mixed_topic_selectors(
    tmp_path: Path,
) -> None:
    config = load_facet_pilot_config(_write_config(tmp_path))
    subset = tmp_path / "subset.csv"
    subset.write_text("topic_id\ntopic-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        facet_pilot_config.select_configured_topics(
            config,
            topic_ids=("topic-1", "topic-1"),
        )
    with pytest.raises(ValueError, match="unknown"):
        facet_pilot_config.select_configured_topics(config, topic_ids=("missing",))
    with pytest.raises(ValueError, match="mutually exclusive"):
        facet_pilot_config.select_configured_topics(
            config,
            topic_ids=("topic-1",),
            subset_csv=subset,
        )


@pytest.mark.parametrize("level", ("top-level", "nested"))
def test_config_rejects_duplicate_yaml_keys_recursively(
    tmp_path: Path,
    level: str,
) -> None:
    config_path = _write_config(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    if level == "top-level":
        source = source.replace(
            "schema_version: facet_pilot_config_v2\n",
            "schema_version: facet_pilot_config_v2\n"
            "schema_version: facet_pilot_config_v2\n",
            1,
        )
    else:
        source = source.replace(
            "experiment:\n  id: official-interface-test\n",
            "experiment:\n  id: first\n  id: second\n",
            1,
        )
    config_path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_facet_pilot_config(config_path)


def test_run_official_preserves_source_order_and_returns_only_export_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches reordered topics, dropped configured limits, and renamed export files."""
    config_path = _write_config(tmp_path)
    planning_backend = object()
    retriever_identity = {
        "name": "fixture-retriever",
        "type": "fixture",
        "index": "climbmix-test",
        "index_url": "https://retrieval.invalid/search",
        "hits": 1000,
    }
    retriever = SimpleNamespace(identity=retriever_identity)

    def canonical_factory() -> object:
        return object()

    local = competition_retrieval._RuntimeDependencies(
        code_commit="a" * 40,
        document_scorer=object(),
        candidate_scorer=object(),
        similarity=object(),
        cache_ignore_checker=lambda _path: True,
    )
    calls: list[tuple[object, object, object, object, object]] = []
    export_events: list[str] = []
    projection_receipts: dict[str, object] = {}

    monkeypatch.setattr(
        competition_retrieval, "_production_dependencies", lambda: local
    )
    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )

    def missing_projection(_config, _topic):
        raise ValueError("expanded canonical checkpoint is missing")

    monkeypatch.setattr(
        competition_retrieval,
        "read_topic_projection_receipt",
        missing_projection,
        raising=False,
    )

    def run_topic(
        topic,
        config,
        identity,
        dependencies,
        *,
        config_sha256,
        expected_retriever_identity,
        execution_policy,
    ):
        assert execution_policy == "online"
        assert len(config_sha256) == 64
        calls.append(
            (topic, config, identity, dependencies, expected_retriever_identity)
        )
        manifest_bytes = b"{}\n"
        manifest_path = (
            config.output_dir
            / topic.id
            / "canonical"
            / "retrieval-projection-manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes)
        projection_receipt = SimpleNamespace(
            topic_id=topic.id,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
        projection_receipts[topic.id] = projection_receipt
        competition_retrieval._publish_topic_cache_operation_receipt(
            config=config,
            topic=topic,
            config_sha256=config_sha256,
            projection_manifest_sha256=projection_receipt.manifest_sha256,
            mode="online",
            phases={
                name: {"resumed": False}
                for name in ("planning", "retrieval", "scoring", "canonical")
            },
            stages={
                stage: {
                    counter: 0
                    for counter in competition_retrieval._CACHE_OPERATION_COUNTER_NAMES
                }
                for stage in competition_retrieval.CACHE_OPERATION_STAGE_NAMES
            },
        )
        return competition_retrieval._TopicTaskOutcome(
            topic.id,
            False,
            projection_receipt,
        )

    def export(config, topics, projection_receipts, *, code_commit):
        assert [topic.id for topic in topics] == ["topic-2", "topic-1"]
        assert [receipt.topic_id for receipt in projection_receipts] == [
            "topic-2",
            "topic-1",
        ]
        assert code_commit == "a" * 40
        export_events.append("written")
        output = config.output_dir
        return RetrievalExportReceipt(
            official_run=output / "r_output_trec_rag_2026.tsv",
            with_text_archive=output / "retrieval_with_text.jsonl.zip",
            generation_handoff=output / "generation_handoff_manifest.json",
            manifest=output / "retrieval_export_manifest.json",
        )

    validated_export = RetrievalExportReceipt(
        official_run=tmp_path / "validated" / "r_output_trec_rag_2026.tsv",
        with_text_archive=tmp_path / "validated" / "retrieval_with_text.jsonl.zip",
        generation_handoff=tmp_path / "validated" / "generation_handoff_manifest.json",
        manifest=tmp_path / "validated" / "retrieval_export_manifest.json",
    )

    def read_export(config, topics):
        assert export_events == ["written"]
        assert [topic.id for topic in topics] == ["topic-2", "topic-1"]
        assert config.experiment.id == "official-interface-test"
        validated_export.manifest.parent.mkdir(parents=True, exist_ok=True)
        validated_export.manifest.write_bytes(b"{}\n")
        export_events.append("validated")
        return validated_export

    monkeypatch.setattr(competition_retrieval, "_run_topic", run_topic)
    monkeypatch.setattr(
        competition_retrieval,
        "_topic_completion_for_dispatch",
        lambda _config, _topic: ("complete", "coverage_sufficient"),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "validate_retrieval_topic_checkpoints",
        lambda _config, topics, **_kwargs: tuple(
            projection_receipts[topic.id] for topic in topics
        ),
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_decomposition_producer_sha256",
        lambda topic, _output_dir, _backend: hashlib.sha256(
            f"fixture-producer:{topic.id}".encode()
        ).hexdigest(),
    )
    monkeypatch.setattr(competition_retrieval, "export_retrieval_run", export)
    monkeypatch.setattr(
        competition_retrieval, "read_retrieval_export_receipt", read_export
    )

    receipt = run_official(
        config_path,
        topic_ids=("topic-1", "topic-2"),
        external=ExternalAdapters(
            planning_backend=planning_backend,
            retriever=retriever,
            canonical_backend_factory=canonical_factory,
        ),
    )

    assert isinstance(receipt, RunReceipt)
    assert receipt.experiment_id == "official-interface-test"
    assert receipt.selected_topic_ids == ("topic-2", "topic-1")
    assert receipt.resumed_topic_ids == ()
    assert receipt.retrieval_export.official_run.name == "r_output_trec_rag_2026.tsv"
    assert (
        receipt.retrieval_export.with_text_archive.name
        == "retrieval_with_text.jsonl.zip"
    )
    assert receipt.retrieval_export.manifest.name == "retrieval_export_manifest.json"
    assert receipt.retrieval_export is validated_export
    assert export_events == ["written", "validated"]
    assert [
        topic.id
        for topic, _config, _identity, _dependencies, _retriever_identity in calls
    ] == [
        "topic-2",
        "topic-1",
    ]
    for _topic, config, identity, dependencies, expected_identity in calls:
        assert len(identity) == 64
        assert expected_identity == retriever_identity
        assert dependencies.planning_backend is planning_backend
        assert dependencies.retriever is retriever
        assert dependencies.document_scorer is local.document_scorer
        assert dependencies.candidate_scorer is local.candidate_scorer
        assert dependencies.similarity is local.similarity
        assert dependencies.canonical_backend_factory is canonical_factory
        assert config.retrieval.documents_per_query == 1000
        assert config.passage.passages_per_query == 100
        assert config.passage.chunk_max_characters == 3500
        assert config.passage.chunk_overlap_characters == 350
        assert config.nuggets.evidence_budget_per_subnarrative == 2
        assert config.nuggets.maximum_claims_per_subnarrative == 1
        assert config.nuggets.maximum_supporting_documents_per_claim == 1


def test_run_official_rejects_corrupt_unsealed_checkpoint_before_topic_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A partial topic is never mistaken for a run-bound dispatch receipt."""
    config_path = _write_config(tmp_path)
    corrupt = tmp_path / "outputs" / "official-interface-test" / "topic-2" / "canonical"
    corrupt.mkdir(parents=True)
    (corrupt / "complete.json").write_text("{}", encoding="utf-8")
    dependency_calls: list[bool] = []
    local = competition_retrieval._RuntimeDependencies(
        code_commit="a" * 40,
        document_scorer=object(),
        candidate_scorer=object(),
        similarity=object(),
        cache_ignore_checker=lambda _path: True,
    )

    def dependencies():
        dependency_calls.append(True)
        return local

    monkeypatch.setattr(competition_retrieval, "_production_dependencies", dependencies)
    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )

    with pytest.raises(TopicDispatchError, match="canonical checkpoint"):
        run_official(
            config_path,
            topic_ids=("topic-2",),
            external=ExternalAdapters(
                planning_backend=object(),
                retriever=SimpleNamespace(
                    identity={
                        "name": "fixture-retriever",
                        "type": "fixture",
                        "index": "climbmix-test",
                        "index_url": "https://retrieval.invalid/search",
                        "hits": 1000,
                    }
                ),
                canonical_backend_factory=lambda: object(),
            ),
        )
    assert dependency_calls == [True]


def test_run_official_rejects_dirty_tree_before_corrupt_resumed_seal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches resume validation running before the required dirty-tree preflight."""
    config_path = _write_config(tmp_path)
    corrupt = tmp_path / "outputs" / "official-interface-test" / "topic-2" / "canonical"
    corrupt.mkdir(parents=True)
    (corrupt / "complete.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: True
    )

    def dependencies():
        raise AssertionError("dirty trees must fail before runtime dependencies")

    monkeypatch.setattr(competition_retrieval, "_production_dependencies", dependencies)

    with pytest.raises(RuntimeError, match="tracked changes"):
        run_official(config_path, topic_ids=("topic-2",))


def _write_pipeline_config(root: Path) -> Path:
    (root / "topics.tsv").write_text(
        "housing-1\tHousing tenants compare rent increases and zoning changes.\n",
        encoding="utf-8",
    )
    path = root / "pipeline.yaml"
    path.write_text(
        """\
schema_version: facet_pilot_config_v2
experiment:
  id: public-e2e
topics:
  path: topics.tsv
retrieval:
  index: climbmix-test
  cache_dir: cache/retrieval
  query_sources: [original, subnarrative]
  documents_per_query: 1000
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
""",
        encoding="utf-8",
    )
    return path


def _plan(topic_id: str) -> dict[str, object]:
    return {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": topic_id,
        "subnarratives": [
            {
                "subnarrative": "How rent increases affect housing tenants",
                "bm25_queries": ["rent increases housing tenant impacts"],
            },
            {
                "subnarrative": "How zoning changes affect housing tenants",
                "bm25_queries": ["zoning changes housing tenant impacts"],
            },
        ],
    }


class _PlanningBackend:
    def __init__(self, *, reject: bool = False) -> None:
        self.identity = {
            "backend": "fixture-planner",
            "model": "fixture-v1",
            "prompt_version": "fixture-v1",
        }
        self.reject = reject
        self.calls: list[Topic] = []

    @property
    def transport_invocation_count(self) -> int:
        return len(self.calls)

    @staticmethod
    def planning_request_identity(topic: Topic):
        return planning_cache_identity(topic)

    def extract(self, topic: Topic) -> object:
        self.calls.append(topic)
        if self.reject:
            raise RuntimeError()
        return _plan(topic.id)


def test_decomposition_checkpoint_rejects_changed_planner_identity(
    tmp_path: Path,
) -> None:
    topic = Topic("housing-1", "", "Explain housing impacts.")
    first_backend = _PlanningBackend()

    _result, resumed = competition_retrieval._decompose_topic(
        topic,
        tmp_path,
        first_backend,
    )
    assert resumed is False
    assert len(first_backend.calls) == 1

    changed_backend = _PlanningBackend()
    changed_backend.identity = {
        **first_backend.identity,
        "prompt_version": "fixture-v2",
    }
    with pytest.raises(ValueError, match="planner identity changed"):
        competition_retrieval._decompose_topic(
            topic,
            tmp_path,
            changed_backend,
        )

    assert changed_backend.calls == []


class _Retriever:
    identity = {
        "name": "fake-climbmix",
        "index": "climbmix-test",
        "index_url": "https://retrieval.invalid/search",
        "hits": 1000,
    }

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.transport_calls = 0

    def cache_summary(self) -> dict[str, int]:
        return {
            "hits": 0,
            "misses": self.transport_calls,
            "bypasses": 0,
            "writes": self.transport_calls,
        }

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        self.transport_calls += 1
        self.calls.append(query.variant_name)
        lane = query.variant_name.replace(":", "-")
        return [
            RetrievedCandidate(
                topic_id=query.topic_id,
                variant_name=query.variant_name,
                retriever_name="fake-climbmix",
                query_text=query.query_text,
                docid=f"{lane}-d{rank}",
                rank=rank,
                score=float(10 - rank),
                text=f"Full document text for {lane} rank {rank}.",
            )
            for rank in range(1, 4)
        ]


class _DocumentScorer:
    def __init__(self) -> None:
        self.identity = dict(
            _configured_passage_search_identity(
                retrieval_depth=1000,
                passages_per_query=100,
                chunk_max_characters=3500,
                chunk_overlap_characters=350,
                model="mixedbread-ai/mxbai-rerank-base-v2",
                device="cpu",
            )["scorer"]
        )
        self._cache_misses = 0
        self._model_batches = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": 0,
            "cache_misses": self._cache_misses,
            "model_batches": self._model_batches,
        }

    @staticmethod
    def cache_key(query_text: str, passage_text: str) -> str:
        return hashlib.sha256(f"{query_text}\0{passage_text}".encode()).hexdigest()

    def rank(
        self,
        query_text: str,
        chunks: object,
    ) -> tuple[MixedbreadScoredPassage, ...]:
        del query_text
        rows = tuple(chunks)
        self._cache_misses += len(rows)
        self._model_batches += int(bool(rows))
        return tuple(
            MixedbreadScoredPassage(chunk, float(len(rows) - index))
            for index, chunk in enumerate(rows)
        )

    def score_lane(
        self, _topic: Topic, lane: object, candidates: object
    ) -> tuple[LaneDocumentScore, ...]:
        return tuple(
            LaneDocumentScore(
                topic_id=row.topic_id,
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


class _CandidateScorer:
    identity = {
        "model": "fake-local-mixedbread",
        "model_revision": "test-pin",
        "backend_version": "test",
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "score_kind": "extractive_sentence_v1",
        "sentence_max_length": 512,
        "input_policy": "trec_rag_whitespace_v1",
    }

    def __init__(self) -> None:
        self.calls: list[object] = []
        self._cache_misses = 0
        self._model_batches = 0

    @property
    def accounting(self) -> dict[str, int]:
        return {
            "cache_hits": 0,
            "cache_misses": self._cache_misses,
            "model_batches": self._model_batches,
        }

    def score_pairs(self, pairs: object) -> tuple[float, ...]:
        rows = tuple(pairs)
        self.calls.append(rows)
        self._cache_misses += len(rows)
        self._model_batches += int(bool(rows))
        return tuple(float(len(rows) - index) for index, _row in enumerate(rows))


class _Similarity:
    identity = {"model": "fake-minilm", "revision": "test", "normalized": True}

    def __init__(self) -> None:
        self.calls: list[object] = []
        self._cache_misses = 0
        self._model_batches = 0

    @property
    def accounting(self) -> dict[str, int]:
        return {
            "cache_hits": 0,
            "cache_misses": self._cache_misses,
            "model_batches": self._model_batches,
        }

    def cosine_matrix(self, texts: object) -> tuple[tuple[float, ...], ...]:
        rows = tuple(texts)
        self.calls.append(rows)
        self._cache_misses += len(rows)
        self._model_batches += int(bool(rows))
        return tuple(
            tuple(1.0 if left == right else 0.0 for right in rows) for left in rows
        )


class _CanonicalBackend:
    def __init__(self) -> None:
        self.requests: list[object] = []

    @property
    def transport_invocation_count(self) -> int:
        return len(self.requests)

    def complete(self, request: object) -> BackendReply:
        self.requests.append(request)
        content = json.dumps(
            {
                "claims": [
                    {
                        "claim": f"Canonical claim for {request.subnarrative_id}.",
                        "evidence_aliases": ["e001"],
                        "importance": "vital",
                    }
                ]
            }
        ).encode()
        return BackendReply(
            content,
            b'{"safe":"raw response"}',
            200,
            {
                "requested_model": "deepseek/deepseek-v4-flash-20260423",
                "response_model": "deepseek/deepseek-v4-flash-20260423",
                "provider": "fake-provider",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )


def _local_dependencies(
    candidate_scorer: _CandidateScorer,
    similarity: _Similarity,
) -> competition_retrieval._RuntimeDependencies:
    return competition_retrieval._RuntimeDependencies(
        code_commit="f" * 40,
        document_scorer=_DocumentScorer(),
        candidate_scorer=candidate_scorer,
        similarity=similarity,
        cache_ignore_checker=lambda _path: True,
    )


def test_public_run_executes_one_topic_without_network_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches an official runner that bypasses stages or changes sealed bytes on resume."""
    config = _write_pipeline_config(tmp_path)
    planning = _PlanningBackend()
    retriever = _Retriever()
    candidate_scorer = _CandidateScorer()
    similarity = _Similarity()
    canonical = _CanonicalBackend()
    wrapper_factories: list[bool] = []

    def wrapper_factory() -> _CanonicalBackend:
        wrapper_factories.append(True)
        return canonical

    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )
    monkeypatch.setattr(
        canonical_nuggets,
        "OpenRouterCanonicalNuggetBackend",
        lambda: (_ for _ in ()).throw(
            AssertionError("the native canonical backend must not be the default")
        ),
    )
    monkeypatch.setattr(
        nuggetizer_adapter,
        "NuggetizerCanonicalNuggetBackend",
        wrapper_factory,
    )
    adapters = ExternalAdapters(
        planning_backend=planning,
        retriever=retriever,
    )

    real_build_topic_projection = getattr(
        competition_retrieval, "build_topic_projection", None
    )
    assert real_build_topic_projection is not None
    build_events: list[str] = []

    def build_topic_projection(*args, **kwargs):
        complete = (
            tmp_path
            / "outputs"
            / "public-e2e"
            / "housing-1"
            / "canonical"
            / "complete.json"
        )
        assert not complete.exists()
        build_events.append("before")
        receipt = real_build_topic_projection(*args, **kwargs)
        assert complete.is_file()
        assert (complete.parent / "retrieval-projection.json").is_file()
        assert (complete.parent / "retrieval-projection-manifest.json").is_file()
        expanded = json.loads(complete.read_bytes())
        assert {row["relative_path"] for row in expanded["artifacts"]} == {
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
        build_events.append("after")
        return receipt

    monkeypatch.setattr(
        competition_retrieval,
        "build_topic_projection",
        build_topic_projection,
    )

    first = run_official(config, external=adapters)
    output = tmp_path / "outputs" / "public-e2e"
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    second = run_official(config, external=adapters)
    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    changed_planner = _PlanningBackend()
    changed_planner.identity = {
        **planning.identity,
        "prompt_version": "fixture-v2",
    }
    with pytest.raises(ValueError, match="planner identity changed"):
        run_official(
            config,
            external=ExternalAdapters(
                planning_backend=changed_planner,
                retriever=retriever,
            ),
        )

    assert first.selected_topic_ids == ("housing-1",)
    assert second.resumed_topic_ids == ("housing-1",)
    assert build_events == ["before", "after"]
    assert before == after
    assert len(planning.calls) == 1
    assert changed_planner.calls == []
    assert wrapper_factories == [True]
    assert len(canonical.requests) == 2
    canonical_rows = [
        json.loads(line)
        for line in (output / "housing-1" / "canonical" / "canonical-nuggets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["subnarrative_id"] for row in canonical_rows} == {
        "subnarrative-1",
        "subnarrative-2",
    }
    for row in canonical_rows:
        assert row["state"] == "complete"
        assert len(row["nuggets"]) == 1
        evidence = row["nuggets"][0]["evidence"]
        assert len(evidence) == 1
        assert set(evidence[0]) == {
            "candidate_nugget_id",
            "candidate_kind",
            "text",
            "text_sha256",
            "docid",
            "document_sha256",
            "cluster_id",
        }
        assert evidence[0]["candidate_nugget_id"]
        assert evidence[0]["docid"]
    assert (output / "housing-1" / "canonical" / "complete.json").is_file()
    assert (
        tmp_path / "cache" / "canonical" / canonical_nuggets.PROMPT_VERSION
    ).is_dir()
    assert not (output / "housing-1" / "canonical" / "response-cache").exists()
    assert first.retrieval_export.official_run.read_text().startswith("housing-1 Q0 ")


def test_rejected_plan_exports_original_only_without_downstream_hosted_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan failure cannot fabricate selected evidence for generation."""

    config = _write_pipeline_config(tmp_path)
    planning = _PlanningBackend(reject=True)
    retriever = _Retriever()
    candidate_scorer = _CandidateScorer()
    similarity = _Similarity()
    canonical_factories: list[bool] = []
    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )

    with pytest.raises(TopicDispatchError, match="no supported document"):
        run_official(
            config,
            external=ExternalAdapters(
                planning_backend=planning,
                retriever=retriever,
                canonical_backend_factory=lambda: canonical_factories.append(True),
            ),
        )

    canonical_root = tmp_path / "outputs" / "public-e2e" / "housing-1" / "canonical"
    decomposition = json.loads(
        (
            tmp_path
            / "outputs"
            / "public-e2e"
            / "housing-1"
            / "decomposition"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert decomposition["error"] == "RuntimeError"
    output = canonical_root.parent.parent
    assert not (output / "retrieval_export_manifest.json").exists()
    assert not (output / "generation_handoff_manifest.json").exists()
    assert retriever.calls == ["original"]
    assert candidate_scorer.calls == []
    assert similarity.calls == []
    assert canonical_factories == []
    assert (canonical_root / "canonical-nuggets.jsonl").read_bytes() == b""


def test_rejected_plan_with_empty_original_retrieval_reports_no_supported_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted original fallback is empty, not downstream scoring."""

    class _EmptyRetriever(_Retriever):
        def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
            self.transport_calls += 1
            self.calls.append(query.variant_name)
            return []

    config = _write_pipeline_config(tmp_path)
    planning = _PlanningBackend(reject=True)
    retriever = _EmptyRetriever()
    candidate_scorer = _CandidateScorer()
    similarity = _Similarity()
    canonical_factories: list[bool] = []
    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )

    with pytest.raises(TopicDispatchError, match="no supported document"):
        run_official(
            config,
            external=ExternalAdapters(
                planning_backend=planning,
                retriever=retriever,
                canonical_backend_factory=lambda: canonical_factories.append(True),
            ),
        )

    output = tmp_path / "outputs" / "public-e2e"
    canonical_root = output / "housing-1" / "canonical"
    assert retriever.calls == ["original"]
    assert candidate_scorer.calls == []
    assert similarity.calls == []
    assert canonical_factories == []
    assert (canonical_root / "canonical-nuggets.jsonl").read_bytes() == b""
    assert not (output / "retrieval_export_manifest.json").exists()
    assert not (output / "generation_handoff_manifest.json").exists()


def test_public_run_rejects_base_only_legacy_checkpoint_without_hosted_or_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_pipeline_config(tmp_path)
    planning = _PlanningBackend()
    retriever = _Retriever()
    candidate_scorer = _CandidateScorer()
    similarity = _Similarity()
    canonical = _CanonicalBackend()

    monkeypatch.setattr(
        competition_retrieval, "_tracked_worktree_is_dirty", lambda _repo: False
    )
    monkeypatch.setattr(
        competition_retrieval,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )
    run_official(
        config,
        external=ExternalAdapters(
            planning_backend=planning,
            retriever=retriever,
            canonical_backend_factory=lambda: canonical,
        ),
    )
    planning_calls = len(planning.calls)
    retrieval_calls = len(retriever.calls)
    canonical_calls = len(canonical.requests)

    canonical_root = tmp_path / "outputs" / "public-e2e" / "housing-1" / "canonical"
    complete = canonical_root / "complete.json"
    legacy = json.loads(complete.read_bytes())
    legacy["artifacts"] = [
        row
        for row in legacy["artifacts"]
        if not row["relative_path"].startswith(
            (
                "canonical/retrieval-projection",
                "canonical/generation-projection",
            )
        )
    ]
    complete.write_bytes(
        (
            json.dumps(
                legacy,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    (canonical_root / "retrieval-projection.json").unlink()
    (canonical_root / "retrieval-projection-manifest.json").unlink()
    (canonical_root / "generation-projection.json").unlink()
    (canonical_root / "generation-projection-manifest.json").unlink()
    (
        tmp_path / "outputs" / "public-e2e" / "housing-1" / "topic-job-receipt.json"
    ).unlink()

    class _FailingPlanning:
        def extract(self, _topic: Topic) -> object:
            raise AssertionError("legacy checkpoint rejection must not plan")

    class _FailingRetriever:
        identity = dict(_Retriever.identity)

        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            raise AssertionError("legacy checkpoint rejection must not retrieve")

    def failing_canonical_factory() -> object:
        raise AssertionError(
            "legacy checkpoint rejection must not call the canonical backend"
        )

    base_only_complete = complete.read_bytes()
    with pytest.raises(
        TopicDispatchError,
        match="legacy canonical checkpoint is incompatible",
    ):
        run_official(
            config,
            external=ExternalAdapters(
                planning_backend=_FailingPlanning(),
                retriever=_FailingRetriever(),
                canonical_backend_factory=failing_canonical_factory,
            ),
        )

    assert len(planning.calls) == planning_calls
    assert len(retriever.calls) == retrieval_calls
    assert len(canonical.requests) == canonical_calls
    assert complete.read_bytes() == base_only_complete
    assert not (canonical_root / "retrieval-projection.json").exists()
    assert not (canonical_root / "retrieval-projection-manifest.json").exists()
    assert not (canonical_root / "generation-projection.json").exists()
    assert not (canonical_root / "generation-projection-manifest.json").exists()

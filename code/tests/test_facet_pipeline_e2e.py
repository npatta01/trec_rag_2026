from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import trec_rag.facet_pilot_config as facet_pilot_config
import trec_rag.official_run as official_run
from trec_rag.facet_extraction import BackendReply
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.facet_retrieval import LaneDocumentScore, PassageScore
from trec_rag.official_run import ExternalAdapters, RunReceipt, run_official
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.retrieval_export import RetrievalExportReceipt
from trec_rag.topics import Topic


def _write_config(
    root: Path,
    *,
    experiment_id: str = "official-interface-test",
    topics: tuple[str, ...] = ("topic-2", "topic-1"),
    extra: str = "",
) -> Path:
    (root / "topics.tsv").write_text(
        "".join(f"{topic_id}\tOfficial narrative for {topic_id}.\n" for topic_id in topics),
        encoding="utf-8",
    )
    config = root / "official.yaml"
    config.write_text(
        f"""\
schema_version: facet_pilot_config_v1
experiment:
  id: {experiment_id}
topics:
  path: topics.tsv
retrieval:
  index: climbmix-test
  cache_dir: cache/retrieval
  query_sources: [original, subnarrative]
  candidate_depth_per_query: 7
reranking:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  rerank_depth_per_query: 5
  candidate_pool_depth: 3
  selection_policy: round_robin_subnarrative_coverage
nuggets:
  evidence_budget_per_subnarrative: 2
  maximum_claims_per_subnarrative: 1
  maximum_supporting_documents_per_claim: 1
{extra}
""",
        encoding="utf-8",
    )
    return config


def test_config_uses_conventional_identity_and_binds_topic_source(tmp_path: Path) -> None:
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
            "schema_version: facet_pilot_config_v1\n",
            "schema_version: facet_pilot_config_v1\n"
            "schema_version: facet_pilot_config_v1\n",
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
    retriever = object()
    canonical_factory = lambda: object()
    local = official_run._RuntimeDependencies(
        code_commit="a" * 40,
        document_scorer=object(),
        candidate_scorer=object(),
        similarity=object(),
        cache_ignore_checker=lambda _path: True,
    )
    calls: list[tuple[object, object, object, object]] = []
    export_events: list[str] = []

    monkeypatch.setattr(official_run, "_production_dependencies", lambda: local)
    monkeypatch.setattr(official_run, "_tracked_worktree_is_dirty", lambda _repo: False)

    def run_topic(topic, config, identity, dependencies):
        calls.append((topic, config, identity, dependencies))
        (config.output_dir / topic.id / "canonical").mkdir(
            parents=True,
            exist_ok=True,
        )
        (config.output_dir / topic.id / "canonical" / "complete.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return official_run.TopicPhaseOutcome(
            topic.id,
            "canonical",
            config.output_dir / topic.id / "canonical" / "complete.json",
            False,
        )

    def export(config, topics, *, code_commit):
        assert [topic.id for topic in topics] == ["topic-2", "topic-1"]
        assert code_commit == "a" * 40
        export_events.append("written")
        output = config.output_dir
        return RetrievalExportReceipt(
            official_run=output / "r_output_trec_rag_2026.tsv",
            candidate_pool_run=output / "retrieval_candidate_pool.trec",
            with_text_archive=output / "retrieval_with_text.jsonl.zip",
            provenance=output / "retrieval_provenance.jsonl",
            resolved_config=output / "resolved_config.yaml",
            manifest=output / "retrieval_export_manifest.json",
        )

    validated_export = RetrievalExportReceipt(
        official_run=tmp_path / "validated" / "r_output_trec_rag_2026.tsv",
        candidate_pool_run=tmp_path / "validated" / "retrieval_candidate_pool.trec",
        with_text_archive=tmp_path / "validated" / "retrieval_with_text.jsonl.zip",
        provenance=tmp_path / "validated" / "retrieval_provenance.jsonl",
        resolved_config=tmp_path / "validated" / "resolved_config.yaml",
        manifest=tmp_path / "validated" / "retrieval_export_manifest.json",
    )

    def read_export(config, topics):
        assert export_events == ["written"]
        assert [topic.id for topic in topics] == ["topic-2", "topic-1"]
        assert config.experiment.id == "official-interface-test"
        export_events.append("validated")
        return validated_export

    monkeypatch.setattr(official_run, "_run_topic", run_topic)
    monkeypatch.setattr(official_run, "export_retrieval_run", export)
    monkeypatch.setattr(official_run, "read_retrieval_export_receipt", read_export)

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
    assert receipt.retrieval_export.candidate_pool_run.name == "retrieval_candidate_pool.trec"
    assert receipt.retrieval_export.with_text_archive.name == "retrieval_with_text.jsonl.zip"
    assert receipt.retrieval_export.provenance.name == "retrieval_provenance.jsonl"
    assert receipt.retrieval_export.resolved_config.name == "resolved_config.yaml"
    assert receipt.retrieval_export.manifest.name == "retrieval_export_manifest.json"
    assert receipt.retrieval_export is validated_export
    assert export_events == ["written", "validated"]
    assert [topic.id for topic, _config, _identity, _dependencies in calls] == [
        "topic-2",
        "topic-1",
    ]
    for _topic, config, identity, dependencies in calls:
        assert len(identity) == 64
        assert dependencies.planning_backend is planning_backend
        assert dependencies.retriever is retriever
        assert dependencies.document_scorer is local.document_scorer
        assert dependencies.candidate_scorer is local.candidate_scorer
        assert dependencies.similarity is local.similarity
        assert dependencies.canonical_backend_factory is canonical_factory
        assert config.retrieval.candidate_depth_per_query == 7
        assert config.reranking.rerank_depth_per_query == 5
        assert config.reranking.candidate_pool_depth == 3
        assert config.nuggets.evidence_budget_per_subnarrative == 2
        assert config.nuggets.maximum_claims_per_subnarrative == 1
        assert config.nuggets.maximum_supporting_documents_per_claim == 1


def test_run_official_rejects_corrupt_resumed_seal_before_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches existence-only resume checks that construct dependencies too early."""
    config_path = _write_config(tmp_path)
    corrupt = tmp_path / "outputs" / "official-interface-test" / "topic-2" / "canonical"
    corrupt.mkdir(parents=True)
    (corrupt / "complete.json").write_text("{}", encoding="utf-8")

    def dependencies():
        raise AssertionError("corrupt seals must fail before dependency construction")

    monkeypatch.setattr(official_run, "_production_dependencies", dependencies)
    monkeypatch.setattr(official_run, "_tracked_worktree_is_dirty", lambda _repo: False)

    with pytest.raises(ValueError):
        run_official(config_path, topic_ids=("topic-2",))


def test_run_official_rejects_dirty_tree_before_corrupt_resumed_seal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches resume validation running before the required dirty-tree preflight."""
    config_path = _write_config(tmp_path)
    corrupt = tmp_path / "outputs" / "official-interface-test" / "topic-2" / "canonical"
    corrupt.mkdir(parents=True)
    (corrupt / "complete.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(official_run, "_tracked_worktree_is_dirty", lambda _repo: True)

    def dependencies():
        raise AssertionError("dirty trees must fail before runtime dependencies")

    monkeypatch.setattr(official_run, "_production_dependencies", dependencies)

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
schema_version: facet_pilot_config_v1
experiment:
  id: public-e2e
topics:
  path: topics.tsv
retrieval:
  index: climbmix-test
  cache_dir: cache/retrieval
  query_sources: [original, subnarrative]
  candidate_depth_per_query: 3
reranking:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  rerank_depth_per_query: 3
  candidate_pool_depth: 2
  selection_policy: round_robin_subnarrative_coverage
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
        self.reject = reject
        self.calls: list[Topic] = []

    def extract(self, topic: Topic) -> object:
        self.calls.append(topic)
        if self.reject:
            raise ValueError("fake model rejected the plan")
        return _plan(topic.id)


class _Retriever:
    identity = {
        "name": "fake-climbmix",
        "index": "climbmix-test",
        "index_url": "https://retrieval.invalid/search",
        "hits": 3,
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
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
    identity = {
        "model": "mixedbread-ai/mxbai-rerank-base-v2",
        "model_revision": "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        "backend_version": "test",
        "score_representation": "raw_logits",
    }

    def score_lane(self, _topic: Topic, lane: object, candidates: object) -> tuple[LaneDocumentScore, ...]:
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
                winning_passages=(
                    PassageScore(0, 0, len(row.text), 3.0, 1),
                ),
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
    }

    def __init__(self) -> None:
        self.calls: list[object] = []

    def score_pairs(self, pairs: object) -> tuple[float, ...]:
        rows = tuple(pairs)
        self.calls.append(rows)
        return tuple(float(len(rows) - index) for index, _row in enumerate(rows))


class _Similarity:
    identity = {"model": "fake-minilm", "revision": "test", "normalized": True}

    def __init__(self) -> None:
        self.calls: list[object] = []

    def cosine_matrix(self, texts: object) -> tuple[tuple[float, ...], ...]:
        rows = tuple(texts)
        self.calls.append(rows)
        return tuple(
            tuple(1.0 if left == right else 0.0 for right in rows)
            for left in rows
        )


class _CanonicalBackend:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def complete(self, request: object) -> BackendReply:
        self.requests.append(request)
        content = json.dumps(
            {
                "claims": [
                    {
                        "claim": f"Canonical claim for {request.subnarrative_id}.",
                        "evidence_aliases": ["e001"],
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
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )


def _local_dependencies(
    candidate_scorer: _CandidateScorer,
    similarity: _Similarity,
) -> official_run._RuntimeDependencies:
    return official_run._RuntimeDependencies(
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
    monkeypatch.setattr(official_run, "_tracked_worktree_is_dirty", lambda _repo: False)
    monkeypatch.setattr(
        official_run,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )
    adapters = ExternalAdapters(
        planning_backend=planning,
        retriever=retriever,
        canonical_backend_factory=lambda: canonical,
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

    assert first.selected_topic_ids == ("housing-1",)
    assert second.resumed_topic_ids == ("housing-1",)
    assert before == after
    assert len(planning.calls) == 1
    assert len(canonical.requests) == 2
    assert (output / "housing-1" / "canonical" / "complete.json").is_file()
    assert first.retrieval_export.official_run.read_text().startswith("housing-1 Q0 ")


def test_rejected_plan_exports_original_only_without_downstream_hosted_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches fallback exports that invent canonical support or skip publication."""
    import zipfile

    config = _write_pipeline_config(tmp_path)
    planning = _PlanningBackend(reject=True)
    retriever = _Retriever()
    candidate_scorer = _CandidateScorer()
    similarity = _Similarity()
    canonical_factories: list[bool] = []
    monkeypatch.setattr(official_run, "_tracked_worktree_is_dirty", lambda _repo: False)
    monkeypatch.setattr(
        official_run,
        "_production_dependencies",
        lambda: _local_dependencies(candidate_scorer, similarity),
    )

    receipt = run_official(
        config,
        external=ExternalAdapters(
            planning_backend=planning,
            retriever=retriever,
            canonical_backend_factory=lambda: canonical_factories.append(True),
        ),
    )

    canonical_root = tmp_path / "outputs" / "public-e2e" / "housing-1" / "canonical"
    exported = receipt.retrieval_export
    assert receipt.selected_topic_ids == ("housing-1",)
    assert all(
        path.is_file()
        for path in (
            exported.official_run,
            exported.candidate_pool_run,
            exported.with_text_archive,
            exported.provenance,
            exported.resolved_config,
            exported.manifest,
        )
    )
    assert exported.official_run.read_bytes() == (
        b"housing-1 Q0 original-d1 1 2 public-e2e\n"
        b"housing-1 Q0 original-d2 2 1 public-e2e\n"
    )
    assert exported.candidate_pool_run.read_bytes() == (
        b"housing-1 Q0 original-d1 1 2 public-e2e-candidate-pool\n"
        b"housing-1 Q0 original-d2 2 1 public-e2e-candidate-pool\n"
    )
    with zipfile.ZipFile(exported.with_text_archive) as archive:
        assert archive.namelist() == ["retrieval_with_text.jsonl"]
        with_text = json.loads(archive.read("retrieval_with_text.jsonl"))
    assert [row["docid"] for row in with_text["candidates"]] == [
        "original-d1",
        "original-d2",
    ]
    assert {row["stage"] for row in with_text["candidates"]} == {
        "original_only_fallback"
    }
    provenance = [
        json.loads(line) for line in exported.provenance.read_bytes().splitlines()
    ]
    assert [row["docid"] for row in provenance] == ["original-d1", "original-d2"]
    assert {row["stage"] for row in provenance} == {"original_only_fallback"}
    assert all(row["subnarrative_scores"] == [] for row in provenance)
    assert all(row["nuggets"] == [] for row in provenance)
    assert all(
        [lane["lane_name"] for lane in row["memberships"]] == ["original"]
        for row in provenance
    )
    manifest = json.loads(exported.manifest.read_bytes())
    assert manifest["official_row_count"] == 2
    assert manifest["candidate_pool_row_count"] == 2
    assert manifest["topic_depths"] == {
        "housing-1": {"official": 2, "candidate_pool": 2}
    }
    assert retriever.calls == ["original", "original"]
    assert candidate_scorer.calls == []
    assert similarity.calls == []
    assert canonical_factories == []
    assert (canonical_root / "canonical-nuggets.jsonl").read_bytes() == b""

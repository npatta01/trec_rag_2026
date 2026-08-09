from __future__ import annotations

from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from trec_rag.agentic_generation_export import (
    AgenticFullTextCandidate,
    AgenticRetrievalRow,
    AgenticTopicProjection,
    serialize_agentic_retrieval_topic,
)
from trec_rag.agentic_retrieval_export import (
    EXPORT_MANIFEST_FILENAME,
    GENERATION_HANDOFF_FILENAME,
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
)
from trec_rag.agentic_run_state import SubmoduleRevision, load_run_plan
import trec_rag.competition_agentic_retrieval as agentic_retrieval
from trec_rag.competition_agentic_retrieval import (
    AgenticRepositoryBinding,
    AgenticRunnerError,
    TopicExecutionResult,
    TopicOperationalError,
    _build_production_topic_executor,
    _main,
    _probe_repository,
    _run_agentic_retrieval,
)
from trec_rag.generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationTopic,
    SelectedCluster,
    TopicSourceReceipts,
)
from trec_rag.topic_records import TopicRecordsReceipt
from trec_rag.topics import Topic


REVISION = "1" * 40
SUBMODULES = (
    SubmoduleRevision("ragdoll", "2" * 40),
    SubmoduleRevision("trec-rag-skills", "3" * 40),
)
SECRETS = {
    "INDEX_URL": "https://index.invalid/v1/climbmix-400b/search",
    "PYSERINI_API_TOKEN": "pyserini-secret-value",
    "OPENROUTER_API_KEY": "openrouter-secret-value",
}
ROOT_ARTIFACTS = (
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
    GENERATION_HANDOFF_FILENAME,
    EXPORT_MANIFEST_FILENAME,
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: object) -> bytes:
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


def _workspace(
    tmp_path: Path,
    *,
    topic_count: int = 2,
    run_id: str = "agentic-cli-test",
) -> tuple[Path, tuple[Topic, ...]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("# test root\n", encoding="utf-8")
    official_topics = tuple(
        Topic(
            f"rag2026-{index}",
            "",
            f"Narrative {index} asks for grounded café evidence.",
        )
        for index in range(119)
    )
    topics = official_topics[:topic_count]
    topics_path = (
        tmp_path
        / "trec-rag-data"
        / "trec-rag-2026"
        / "test-data"
        / "trec_rag_2026_queries.tsv"
    )
    topics_path.parent.mkdir(parents=True)
    topics_path.write_text(
        "".join(
            f"{topic.id}\t{topic.narrative}\n" for topic in official_topics
        ),
        encoding="utf-8",
    )
    config = tmp_path / "agentic.yaml"
    config.write_text(
        f"""\
schema_version: agentic_retrieval_config_v1
retrieval_mode: agentic
experiment:
  id: {run_id}
topics:
  path: trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv
execution:
  topic_workers: 1
caches:
  document_store_dir: cache/documents/v1
  model_cache_dir: cache/models/huggingface
retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  documents_per_query: 1000
  hits_per_search: 10
  corpus_epoch: climbmix-400b-operator-archive-facet-2025-v2-sha256-2ccad901eb908ac14747bafd2069f7c8aa96ecaaa8cc30457ff47c73ce3bf81f
passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  revision: 3ea9d4dffa7d12a4f366be8e275c349de9fc9865
  score_cache_dir: cache/reranker
  device: auto
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350
snippets:
  result_cache_dir: cache/reranker/deepagent_snippets
  snippets_per_page: 10
models:
  coordinator_and_researcher: openrouter:deepseek/deepseek-v4-flash
agent:
  fused_result_limit: 20
budget:
  max_researcher_invocations: 20
  max_concurrent: 3
  max_retrieval_calls: 100
  max_tools_per_researcher: 20
  max_searches_per_researcher: 8
  max_snippets_per_researcher: 16
  max_passage_searches_per_researcher: 8
  max_models_per_researcher: 30
  max_main_models: 80
  synthesis_reserve_turns: 2
  no_yield_calls: 3
  no_progress_rounds: 2
""",
        encoding="utf-8",
    )
    return config, topics


def _binding(
    *,
    revision: str = REVISION,
    submodules: tuple[SubmoduleRevision, ...] = SUBMODULES,
) -> AgenticRepositoryBinding:
    return AgenticRepositoryBinding(revision, submodules)


def _generation_topic(
    *,
    topic: Topic,
    docid: str,
    document: str,
    official_topics_sha256: str,
    retrieval_topic_sha256: str,
) -> GenerationTopic:
    group_id = f"group-{topic.id}"
    cluster_id = f"cluster-{topic.id}"
    evidence_id = f"evidence-{topic.id}"
    return GenerationTopic(
        topic_id=topic.id,
        narrative=topic.narrative,
        groups=(
            EvidenceGroup(
                group_id=group_id,
                kind="generated_subnarrative",
                text=f"Grounded need for {topic.id}",
                selected_clusters=(
                    SelectedCluster(
                        cluster_id=cluster_id,
                        ordinal=1,
                        representative_evidence_id=evidence_id,
                        evidence_ids=(evidence_id,),
                    ),
                ),
            ),
        ),
        evidence=(
            EvidencePassage(
                evidence_id=evidence_id,
                group_id=group_id,
                cluster_id=cluster_id,
                cluster_ordinal=1,
                support_ordinal=1,
                candidate_kind="agentic_grounded_nugget",
                docid=docid,
                document_rank=1,
                text=document,
                document_sha256=_digest(document),
                source_span=EvidenceSourceSpan(
                    start_char=0,
                    end_char=len(document),
                    start_byte=0,
                    end_byte=len(document.encode("utf-8")),
                ),
            ),
        ),
        claim_hints=(
            ClaimHint(
                claim_id=f"claim-{topic.id}",
                group_id=group_id,
                kind="agentic_grounded_nugget",
                text=f"Grounded claim for {topic.id}.",
                evidence_ids=(evidence_id,),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256=official_topics_sha256,
            retrieval_topic_sha256=retrieval_topic_sha256,
        ),
    )


def _projection(
    topic: Topic, *, official_topics_sha256: str
) -> AgenticTopicProjection:
    docid = f"doc-{topic.id}"
    document = f"Document for {topic.id}: grounded café evidence."
    rows = (AgenticRetrievalRow(topic.id, docid, 1, 1),)
    candidates = (
        AgenticFullTextCandidate(
            docid=docid,
            text=document,
            rank=1,
            score=1,
            document_sha256=_digest(document),
        ),
    )
    preliminary = AgenticTopicProjection(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=rows,
        full_text_candidates=candidates,
        generation_topic=_generation_topic(
            topic=topic,
            docid=docid,
            document=document,
            official_topics_sha256=official_topics_sha256,
            retrieval_topic_sha256="0" * 64,
        ),
        retrieval_topic_sha256="0" * 64,
    )
    retrieval_digest = sha256(
        serialize_agentic_retrieval_topic(preliminary)
    ).hexdigest()
    return AgenticTopicProjection(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=rows,
        full_text_candidates=candidates,
        generation_topic=_generation_topic(
            topic=topic,
            docid=docid,
            document=document,
            official_topics_sha256=official_topics_sha256,
            retrieval_topic_sha256=retrieval_digest,
        ),
        retrieval_topic_sha256=retrieval_digest,
    )


def _success(request) -> TopicExecutionResult:
    projection = _projection(
        request.topic,
        official_topics_sha256=request.official_topics_sha256,
    )
    document_sha256s = tuple(
        sorted(
            candidate.document_sha256
            for candidate in projection.full_text_candidates
        )
    )
    return TopicExecutionResult(
        status="complete",
        stopping_reason="coverage_sufficient",
        synthesis_outcome="coordinator_selected",
        projection=projection,
        records_receipt=TopicRecordsReceipt(
            database_sha256="b" * 64,
            database_bytes=4096,
            semantic_sha256="c" * 64,
            topic_id=request.topic.id,
            run_id=request.plan.run_id,
            document_sha256s=document_sha256s,
            row_counts={
                "candidate": 0,
                "candidate_passage_link": 0,
                "candidate_span": 0,
                "document_binding": len(document_sha256s),
                "passage": len(document_sha256s),
                "query_facet": 0,
                "query_identity": 0,
                "query_passage": 0,
                "researcher_evidence": 0,
                "researcher_facet_update": 0,
                "researcher_handoff": 0,
                "retrieval_candidate": 0,
                "stage_seal": 1,
                "subnarrative_identity": 0,
                "topic_completion": 1,
                "topic_identity": 1,
            },
            schema_version="topic-records-v4",
            manifest_sha256="f" * 64,
            manifest_bytes=512,
        ),
    )


def _failure(reason: str = "zero_grounded_nuggets") -> TopicExecutionResult:
    return TopicExecutionResult(
        status="incomplete",
        stopping_reason=reason,
        synthesis_outcome=None,
        projection=None,
        records_receipt=None,
    )


def _run(
    config: Path,
    *,
    factory,
    resume: bool = False,
    topic_ids: tuple[str, ...] | None = None,
    binding: AgenticRepositoryBinding | None = None,
    repository_probe=None,
    environment_loader=lambda _root: None,
    environ=SECRETS,
):
    return _run_agentic_retrieval(
        config,
        resume=resume,
        topic_ids=topic_ids,
        environment_loader=environment_loader,
        environ=environ,
        repository_probe=(
            repository_probe
            if repository_probe is not None
            else lambda _root: binding or _binding()
        ),
        topic_executor_factory=factory,
    )


def test_repository_binding_rejects_non_sha1_source_revision() -> None:
    with pytest.raises(ValueError, match="source_revision"):
        AgenticRepositoryBinding("1" * 64, ())


@pytest.mark.parametrize(
    ("head", "submodule_status", "match"),
    (
        ("1" * 64, "", "HEAD"),
        (REVISION, f" {'2' * 64} ragdoll\n", "submodule"),
    ),
)
def test_repository_probe_rejects_non_sha1_git_object_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    submodule_status: str,
    match: str,
) -> None:
    def fake_git(_root: Path, arguments: tuple[str, ...]) -> str:
        if arguments[0] == "status":
            return ""
        if arguments[0] == "rev-parse":
            return f"{head}\n"
        if arguments[0] == "submodule":
            return submodule_status
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "trec_rag.competition_agentic_retrieval._run_git", fake_git
    )

    with pytest.raises(AgenticRunnerError, match=match):
        _probe_repository(tmp_path)


@pytest.mark.parametrize("missing_name", tuple(SECRETS))
def test_preflight_rejects_missing_secret_names_before_dependency_construction(
    tmp_path: Path,
    missing_name: str,
) -> None:
    config, _topics = _workspace(tmp_path)
    built = 0

    def factory(_config):
        nonlocal built
        built += 1
        pytest.fail("live dependencies must not be constructed")

    environ = {key: value for key, value in SECRETS.items() if key != missing_name}
    with pytest.raises(AgenticRunnerError, match=missing_name) as caught:
        _run(config, factory=factory, environ=environ)

    assert built == 0
    assert not (tmp_path / "outputs").exists()
    assert all(value not in str(caught.value) for value in SECRETS.values())


def test_preflight_rejects_dirty_or_changing_inputs_before_live_construction(
    tmp_path: Path,
) -> None:
    for case in ("dirty", "config_changed"):
        case_root = tmp_path / case
        config, _topics = _workspace(case_root, run_id=f"run-{case}")
        built = 0

        def factory(_config):
            nonlocal built
            built += 1
            pytest.fail("live dependencies must not be constructed")

        def probe(_root: Path) -> AgenticRepositoryBinding:
            if case == "dirty":
                raise AgenticRunnerError(
                    "dirty_worktree", "tracked worktree changes are not allowed"
                )
            config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return _binding()

        with pytest.raises(AgenticRunnerError, match="tracked|changed"):
            _run(config, factory=factory, repository_probe=probe)

        assert built == 0
        assert not (case_root / "outputs").exists()


def test_create_writes_plan_before_executor_and_zero_grounded_waits_for_manual_resume(
    tmp_path: Path,
) -> None:
    config, topics = _workspace(tmp_path, topic_count=1)
    calls: list[tuple[str, int]] = []

    def factory(loaded_config):
        assert os.environ["HF_HOME"] == str(loaded_config.caches.model_cache_dir)

        def execute(request):
            stored = load_run_plan(loaded_config.output_dir / "work")
            assert stored.plan_sha256 == request.plan.plan_sha256
            calls.append((request.topic.id, request.attempt.attempt_number))
            return _failure()

        return execute

    receipt = _run(config, factory=factory, topic_ids=(topics[0].id,))

    assert receipt.executed_topic_ids == (topics[0].id,)
    assert receipt.unresolved_topic_ids == (topics[0].id,)
    assert calls == [(topics[0].id, 1)]
    assert receipt.export is None
    assert receipt.resume_command == (
        ".venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval "
        f"{config} --resume --topic {topics[0].id}"
    )
    assert not any((tmp_path / "outputs" / "agentic-cli-test" / name).exists() for name in ROOT_ARTIFACTS)
    diagnostic = tmp_path / "outputs" / "agentic-cli-test" / "work" / "topics" / topics[0].id / "attempts" / "000001" / "failure.json"
    assert json.loads(diagnostic.read_text(encoding="utf-8")) == {
        "attempt_number": 1,
        "cache_reuse_available": True,
        "reason": "zero_grounded_nuggets",
        "schema_version": "agentic_topic_failure_v1",
        "topic_id": topics[0].id,
    }


def test_initialize_only_freezes_selected_cohort_without_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, topics = _workspace(tmp_path, topic_count=2, run_id="init-only")
    binding = _binding()

    monkeypatch.setattr(agentic_retrieval, "_probe_repository", lambda _root: binding)
    monkeypatch.setattr(
        agentic_retrieval,
        "_load_environment",
        lambda _root: pytest.fail("initialization must not load runtime secrets"),
    )
    monkeypatch.setattr(
        agentic_retrieval,
        "_check_secret_presence",
        lambda _env: pytest.fail("initialization must not require secrets"),
    )
    monkeypatch.setattr(
        agentic_retrieval,
        "_build_production_topic_executor",
        lambda _config: pytest.fail("initialization must not construct providers"),
    )

    receipt = agentic_retrieval.initialize_agentic_run(
        config, topic_ids=(topics[1].id, topics[0].id)
    )

    plan = load_run_plan(tmp_path / "outputs" / "init-only" / "work")
    assert receipt.plan == plan
    assert plan.planned_topic_ids == (topics[0].id, topics[1].id)
    assert receipt.plan_path.read_bytes() == agentic_retrieval.serialize_run_plan(plan)
    assert receipt.receipt_path.is_file()
    receipt_body = receipt.receipt_path.read_bytes()
    receipt_payload = json.loads(receipt_body)
    assert receipt_body == (
        json.dumps(receipt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    assert receipt_payload["plan_sha256"] == plan.plan_sha256
    assert not (tmp_path / "outputs" / "init-only" / "retrieval_export_manifest.json").exists()


def test_initialize_only_is_identical_only_and_rejects_changed_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, topics = _workspace(tmp_path, topic_count=2, run_id="init-idempotent")
    monkeypatch.setattr(agentic_retrieval, "_probe_repository", lambda _root: _binding())

    first = agentic_retrieval.initialize_agentic_run(config, topic_ids=(topics[0].id,))
    second = agentic_retrieval.initialize_agentic_run(config, topic_ids=(topics[0].id,))
    assert second.plan == first.plan
    assert second.plan_path.read_bytes() == first.plan_path.read_bytes()

    with pytest.raises(agentic_retrieval.AgenticRunnerError, match="cohort"):
        agentic_retrieval.initialize_agentic_run(config, topic_ids=(topics[1].id,))


def test_initialize_only_rejects_changed_config_under_same_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, topics = _workspace(tmp_path, topic_count=2, run_id="init-config-drift")
    monkeypatch.setattr(agentic_retrieval, "_probe_repository", lambda _root: _binding())

    agentic_retrieval.initialize_agentic_run(config, topic_ids=(topics[0].id,))
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(agentic_retrieval.AgenticRunnerError, match="config"):
        agentic_retrieval.initialize_agentic_run(config, topic_ids=(topics[0].id,))


def test_initialize_only_cli_prints_machine_readable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, topics = _workspace(tmp_path, topic_count=2, run_id="init-cli")
    monkeypatch.setattr(agentic_retrieval, "_probe_repository", lambda _root: _binding())
    assert _main(
        [str(config), "--initialize-only", "--topic", topics[0].id],
        runner=agentic_retrieval.initialize_agentic_run,
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "initialized"
    assert output["planned_topic_ids"] == [topics[0].id]


def test_create_refuses_any_existing_run_namespace_without_building_dependencies(
    tmp_path: Path,
) -> None:
    config, _topics = _workspace(tmp_path)
    namespace = tmp_path / "outputs" / "agentic-cli-test"
    namespace.mkdir(parents=True)
    (namespace / "foreign.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(AgenticRunnerError, match="already exists"):
        _run(
            config,
            factory=lambda _config: pytest.fail(
                "existing namespace must fail before dependencies"
            ),
        )

    assert (namespace / "foreign.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_resume_all_skips_sealed_topic_and_restarts_failure_in_new_attempt(
    tmp_path: Path,
) -> None:
    config, topics = _workspace(tmp_path)
    first_calls: list[str] = []

    def first_factory(_config):
        def execute(request):
            first_calls.append(request.topic.id)
            return _success(request) if request.topic.id == topics[0].id else _failure()

        return execute

    first = _run(
        config,
        factory=first_factory,
        topic_ids=tuple(topic.id for topic in topics),
    )
    assert first_calls == [topic.id for topic in topics]
    assert first.unresolved_topic_ids == (topics[1].id,)

    repair_calls: list[tuple[str, int]] = []

    def repair_factory(_config):
        def execute(request):
            repair_calls.append((request.topic.id, request.attempt.attempt_number))
            return _success(request)

        return execute

    repaired = _run(config, factory=repair_factory, resume=True)

    assert repaired.skipped_topic_ids == (topics[0].id,)
    assert repaired.executed_topic_ids == (topics[1].id,)
    assert repair_calls == [(topics[1].id, 2)]
    assert repaired.unresolved_topic_ids == ()
    assert repaired.export is not None
    assert repaired.export.topic_ids == tuple(topic.id for topic in topics)
    assert repaired.export.manifest.is_file()


def test_targeted_twentieth_topic_repair_publishes_the_immutable_full_cohort(
    tmp_path: Path,
) -> None:
    config, topics = _workspace(tmp_path, topic_count=20)
    failed = topics[-1].id

    first = _run(
        config,
        factory=lambda _config: lambda request: (
            _failure("provider_failure")
            if request.topic.id == failed
            else _success(request)
        ),
        topic_ids=tuple(topic.id for topic in topics),
    )
    assert first.unresolved_topic_ids == (failed,)
    assert first.export is None

    repaired_ids: list[str] = []

    def repair_factory(_config):
        def execute(request):
            repaired_ids.append(request.topic.id)
            return _success(request)

        return execute

    repaired = _run(
        config,
        factory=repair_factory,
        resume=True,
        topic_ids=(failed,),
    )

    assert repaired_ids == [failed]
    assert repaired.executed_topic_ids == (failed,)
    assert repaired.skipped_topic_ids == tuple(topic.id for topic in topics[:-1])
    assert repaired.export is not None
    assert repaired.export.topic_ids == tuple(topic.id for topic in topics)
    run_topic_ids = tuple(
        line.split()[0]
        for line in repaired.export.run_file.read_text(encoding="utf-8").splitlines()
    )
    assert run_topic_ids == tuple(topic.id for topic in topics)


def test_targeted_resume_rejects_selector_outside_original_plan_before_factory(
    tmp_path: Path,
) -> None:
    config, topics = _workspace(tmp_path, topic_count=2)
    _run(
        config,
        factory=lambda _config: lambda request: _failure(),
        topic_ids=(topics[0].id,),
    )

    with pytest.raises(AgenticRunnerError, match="outside the run plan"):
        _run(
            config,
            resume=True,
            topic_ids=(topics[1].id,),
            factory=lambda _config: pytest.fail(
                "invalid selector must fail before dependencies"
            ),
        )


@pytest.mark.parametrize("drift", ("config", "source", "submodule"))
def test_resume_refuses_config_source_and_submodule_drift_before_factory(
    tmp_path: Path,
    drift: str,
) -> None:
    config, _topics = _workspace(tmp_path)
    _run(
        config,
        factory=lambda _config: lambda request: _failure(),
        topic_ids=tuple(topic.id for topic in _topics),
    )

    binding = _binding()
    if drift == "config":
        config.write_text(config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    elif drift == "source":
        binding = _binding(revision="4" * 40)
    else:
        binding = _binding(
            submodules=(
                SubmoduleRevision("ragdoll", "5" * 40),
                SUBMODULES[1],
            )
        )

    with pytest.raises(AgenticRunnerError, match="config bytes|source revision|submodule"):
        _run(
            config,
            resume=True,
            binding=binding,
            factory=lambda _config: pytest.fail(
                "drift must fail before dependencies"
            ),
        )


def test_operational_failure_diagnostic_and_cli_output_never_leak_provider_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, topics = _workspace(tmp_path, topic_count=1)
    provider_text = "provider response contains opaque-sensitive-provider-marker"

    def factory(_config):
        def execute(_request):
            try:
                raise RuntimeError(provider_text)
            except RuntimeError as exc:
                raise TopicOperationalError("provider_failure") from exc

        return execute

    runner = partial(
        _run_agentic_retrieval,
        environment_loader=lambda _root: None,
        environ=SECRETS,
        repository_probe=lambda _root: _binding(),
        topic_executor_factory=factory,
    )
    exit_code = _main([str(config), "--topic", topics[0].id], runner=runner)
    captured = capsys.readouterr()

    assert exit_code != 0
    assert captured.out == ""
    assert provider_text not in captured.err
    assert all(secret not in captured.err for secret in SECRETS.values())
    payload = json.loads(captured.err)
    assert payload["status"] == "incomplete"
    assert payload["unresolved_topic_ids"] == [topics[0].id]
    assert payload["resume_command"].endswith(
        f"{config} --resume --topic {topics[0].id}"
    )
    diagnostic = tmp_path / "outputs" / "agentic-cli-test" / "work" / "topics" / topics[0].id / "attempts" / "000001" / "failure.json"
    assert provider_text not in diagnostic.read_text(encoding="utf-8")


def test_fake_provider_one_topic_run_uses_production_wiring_and_prints_json_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.competition_retrieval as competition_retrieval
    import trec_rag.deepagent_retrieval as deepagent_retrieval
    import trec_rag.deepagent_snippets as deepagent_snippets
    import trec_rag.facet_retrieval as facet_retrieval
    import trec_rag.mixedbread_passage_scorer as mixedbread_passage_scorer
    from trec_rag.deepagent_budget import BudgetSnapshot
    from trec_rag.deepagent_evidence import (
        EvidenceCoverageReport,
        EvidenceReference,
        NeedReport,
        NuggetReport,
    )
    from trec_rag.document_store import DocumentStore
    from trec_rag.pipeline_models import RankedCandidate, RetrievedCandidate
    from trec_rag.topic_passage_search import (
        FocusedQuery,
        PassageSearchResult,
        SourceDocument,
        SourcePassage,
    )
    from trec_rag.topic_records import (
        FacetRecord,
        ResearcherEvidence,
        ResearcherHandoff,
        TopicRecordsBuilder,
    )

    config, topics = _workspace(tmp_path, topic_count=1)
    topic = topics[0]
    document = "Production-shaped offline evidence for the café narrative."
    document_sha256 = _digest(document)
    passage_id = "passage-" + "a" * 64
    passage = SourcePassage(
        passage_id=passage_id,
        docid="doc-production-shaped",
        content_sha256=document_sha256,
        source_rank=1,
        source_score=1.0,
        start_char=0,
        end_char=len(document),
        start_byte=0,
        end_byte=len(document.encode("utf-8")),
        text_sha256=document_sha256,
        text=document,
        raw_logit=3.0,
        rank=1,
        score_cache_key="offline-score-cache-key",
        scoring_text_sha256=document_sha256,
        chunker_identity={"backend": "offline-production-shape"},
    )
    source_document = SourceDocument(
        docid=passage.docid,
        content_sha256=document_sha256,
        source_rank=1,
        source_score=1.0,
        best_passage_id=passage_id,
        best_passage_raw_logit=passage.raw_logit,
    )
    search_result = PassageSearchResult(
        query=FocusedQuery("offline-query", topic.narrative, "facet-1"),
        status="complete",
        stopping_reason=None,
        requested_documents=1,
        returned_documents=1,
        scored_documents=1,
        scored_passages=1,
        documents=(source_document,),
        passages=(passage,),
        attempt_count=1,
        source_exhausted=False,
    )
    DocumentStore(tmp_path / "cache" / "documents" / "v1").admit_text(document)

    constructed: dict[str, object] = {}
    pyserini = object()
    scorer = object()
    passage_search = object()

    def build_pyserini(cache_dir: Path, **kwargs: object) -> object:
        constructed["retrieval_cache_dir"] = cache_dir
        constructed["pyserini_kwargs"] = kwargs
        return pyserini

    class FakePassageScorer:
        def __new__(cls, **kwargs: object) -> object:
            constructed["score_cache"] = kwargs
            return scorer

    class FakeSnippetRanker:
        def __init__(self, **kwargs: object) -> None:
            constructed["snippet_ranker"] = kwargs

    class FakeSnippetExtractor:
        def __init__(self, **kwargs: object) -> None:
            constructed["snippet_extractor"] = kwargs

    def build_passage_search(selected_topic: Topic, **kwargs: object) -> object:
        assert selected_topic == topic
        assert kwargs["retriever"] is pyserini
        assert kwargs["scorer"] is scorer
        constructed["passage_search"] = kwargs
        return passage_search

    class FakeProviderRetriever:
        @classmethod
        def from_env(cls, **kwargs: object) -> "FakeProviderRetriever":
            assert kwargs["passage_search"] is passage_search
            constructed["deepagent"] = kwargs
            return cls()

        def retrieve(
            self, records: TopicRecordsBuilder, narrative: str
        ) -> deepagent_retrieval.AgentRetrievalResult:
            assert isinstance(records, TopicRecordsBuilder)
            assert narrative == topic.narrative
            constructed["records_builder"] = records
            records.add_facets((FacetRecord("facet-1", "Offline facet", "initial"),))
            records.add_passage_search(search_result)
            records.add_researcher_handoff(
                ResearcherHandoff(
                    records.run_id,
                    "researcher-1",
                    1,
                    (ResearcherEvidence("facet-1", passage_id, "relevant"),),
                    (),
                )
            )
            records.set_completion("complete", "coverage_sufficient")
            snapshot = records.topic_snapshot()
            report = EvidenceCoverageReport(
                needs=(
                    NeedReport(
                        need_id="need-1",
                        narrative_span=topic.narrative,
                        question="What is the grounded evidence?",
                        status="answerable",
                        remaining_gap="",
                        facet_ids=("facet-1",),
                        nugget_ids=("nugget-1",),
                        draft_answer="Grounded offline answer.",
                        draft_nugget_ids=("nugget-1",),
                    ),
                ),
                facets=(),
                nuggets=(
                    NuggetReport(
                        nugget_id="nugget-1",
                        text="The offline evidence grounds the answer.",
                        need_ids=("need-1",),
                        facet_ids=("facet-1",),
                        evidence=(
                            EvidenceReference(
                                document_id=source_document.docid,
                                snippet_id=passage_id,
                                page_index=0,
                                quote="Model-facing quote is not exported.",
                            ),
                        ),
                        contradicts=(),
                        support="single_document",
                        superseded_by=None,
                        importance="vital",
                        support_ratio=1.0,
                    ),
                ),
                actions=(),
                searches=(),
                documents=(),
                unresolved_need_ids=(),
                search_count=1,
                inspected_page_count=1,
                state_version=1,
                state_hash="e" * 64,
                terminal_reason="coverage_sufficient",
            )
            candidate = RetrievedCandidate(
                topic_id=topic.id,
                variant_name="original",
                retriever_name="offline",
                query_text=topic.narrative,
                docid=source_document.docid,
                rank=1,
                score=1.0,
                text=document,
            )
            ranked = RankedCandidate(
                topic_id=topic.id,
                docid=source_document.docid,
                rank=1,
                score=1.0,
                text=document,
                provenance=[],
            )
            search = deepagent_retrieval.AgentSearch(
                query=topic.narrative,
                kind="original",
                candidates=(candidate,),
                cache_status="hit",
                passages=(passage,),
            )
            return deepagent_retrieval.AgentRetrievalResult(
                narrative=topic.narrative,
                searches=(search,),
                candidates=(ranked,),
                rationale="offline fake provider",
                stopping_reason="coverage_sufficient",
                synthesis_outcome="coordinator_selected",
                coverage_report=report,
                budget_snapshot=BudgetSnapshot(
                    elapsed_seconds=0.0,
                    remaining_researchers=19,
                    remaining_retrieval_calls=99,
                    active_researchers=0,
                    completed_researchers=1,
                    completed_rounds=1,
                    soft_deadline_reached=False,
                    hard_deadline_reached=False,
                    stop_code=None,
                ),
                trace_flush_succeeded=True,
                topic_snapshot=snapshot,
            )

    monkeypatch.setattr(
        facet_retrieval, "build_pyserini_retriever", build_pyserini
    )
    monkeypatch.setattr(
        mixedbread_passage_scorer, "MixedbreadPassageScorer", FakePassageScorer
    )
    monkeypatch.setattr(
        deepagent_snippets, "LocalMixedbreadSnippetRanker", FakeSnippetRanker
    )
    monkeypatch.setattr(
        deepagent_snippets, "RelevantSnippetExtractor", FakeSnippetExtractor
    )
    monkeypatch.setattr(
        competition_retrieval, "_build_topic_passage_search", build_passage_search
    )
    monkeypatch.setattr(
        deepagent_retrieval, "DeepAgentRetriever", FakeProviderRetriever
    )
    runner = partial(
        _run_agentic_retrieval,
        environment_loader=lambda _root: None,
        environ=SECRETS,
        repository_probe=lambda _root: _binding(),
        topic_executor_factory=_build_production_topic_executor,
    )

    exit_code = _main([str(config), "--topic", topics[0].id], runner=runner)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"
    assert payload["planned_topic_ids"] == [topics[0].id]
    assert payload["executed_topic_ids"] == [topics[0].id]
    assert Path(payload["export_manifest"]).name == EXPORT_MANIFEST_FILENAME
    assert constructed["retrieval_cache_dir"] == tmp_path / "cache" / "retrieval" / "pyserini_remote"
    assert constructed["score_cache"] == {
        "score_cache_root": tmp_path / "cache" / "reranker",
        "device": "auto",
    }
    assert isinstance(constructed["records_builder"], TopicRecordsBuilder)
    records_path = (
        tmp_path
        / "outputs"
        / "agentic-cli-test"
        / "work"
        / "topics"
        / topic.id
        / "attempts"
        / "000001"
        / "topic_records"
    )
    assert (records_path / "records.sqlite3").is_file()
    assert (records_path / "canonical" / "records-manifest.json").is_file()
    assert all(
        (tmp_path / "outputs" / "agentic-cli-test" / name).is_file()
        for name in ROOT_ARTIFACTS
    )

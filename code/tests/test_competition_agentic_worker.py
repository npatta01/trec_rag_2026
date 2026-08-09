from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from trec_rag.agentic_run_state import load_run_plan
from trec_rag.competition_agentic_retrieval import (
    AgenticRunPlanReceipt,
    AgenticRunnerError,
    initialize_agentic_run,
)
import trec_rag.competition_agentic_retrieval as retrieval

from test_competition_agentic_retrieval import (
    SECRETS,
    _binding,
    _failure,
    _success,
    _workspace,
)


def _initialize(tmp_path: Path):
    config, topics = _workspace(tmp_path, topic_count=3, run_id="worker-test")
    original_probe = retrieval._probe_repository
    retrieval._probe_repository = lambda _root: _binding()
    try:
        receipt = initialize_agentic_run(config, topic_ids=tuple(topic.id for topic in topics))
    finally:
        retrieval._probe_repository = original_probe
    assert isinstance(receipt, AgenticRunPlanReceipt)
    return config, topics, receipt


def _run(
    config: Path,
    plan_body: bytes,
    assigned: tuple[str, ...],
    factory,
):
    worker = importlib.import_module("trec_rag.competition_agentic_worker")
    return worker._run_agentic_worker(
        config,
        plan_body=plan_body,
        assigned_topic_ids=assigned,
        environment_loader=lambda _root: None,
        environ=SECRETS,
        repository_probe=lambda _root: _binding(),
        topic_executor_factory=factory,
    )


def test_worker_executes_only_assigned_topics_and_never_exports(
    tmp_path: Path,
) -> None:
    config, topics, receipt = _initialize(tmp_path)
    calls: list[str] = []
    sibling_manifest = (
        receipt.output_dir
        / "work"
        / "topics"
        / topics[0].id
        / "topic_projection_manifest.json"
    )
    sibling_manifest.parent.mkdir(parents=True)
    sibling_manifest.write_text("not a valid sibling seal\n", encoding="utf-8")

    def factory(_config):
        def execute(request):
            calls.append(request.topic.id)
            return _success(request)

        return execute

    result = _run(config, receipt.plan_path.read_bytes(), (topics[1].id,), factory)

    assert calls == [topics[1].id]
    assert result.assigned_topic_ids == (topics[1].id,)
    assert [outcome.topic_id for outcome in result.outcomes] == [topics[1].id]
    assert result.outcomes[0].status == "complete"
    assert not (tmp_path / "outputs" / "worker-test" / "retrieval_export_manifest.json").exists()
    assert not (receipt.output_dir / "work" / "topics" / topics[0].id / "attempts").exists()
    assert not (receipt.output_dir / "work" / "topics" / topics[2].id / "attempts").exists()


@pytest.mark.parametrize(
    "assigned",
    (("rag2026-99",), ("rag2026-0", "rag2026-0")),
)
def test_worker_rejects_foreign_or_duplicate_assignments(
    tmp_path: Path, assigned: tuple[str, ...]
) -> None:
    config, _topics, receipt = _initialize(tmp_path)

    with pytest.raises(AgenticRunnerError, match="assigned topic"):
        _run(config, receipt.plan_path.read_bytes(), assigned, lambda _config: pytest.fail())


def test_worker_rejects_config_or_source_drift_before_provider_construction(
    tmp_path: Path,
) -> None:
    config, topics, receipt = _initialize(tmp_path)
    original = config.read_text(encoding="utf-8")
    config.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(AgenticRunnerError, match="config"):
        _run(config, receipt.plan_path.read_bytes(), (topics[0].id,), lambda _config: pytest.fail())

    config.write_text(original, encoding="utf-8")
    source = tmp_path / "trec-rag-data" / "trec-rag-2026" / "test-data" / "trec_rag_2026_queries.tsv"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Narrative 0 asks", "Changed narrative 0 asks", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(AgenticRunnerError, match="topic source|narrative"):
        _run(config, receipt.plan_path.read_bytes(), (topics[0].id,), lambda _config: pytest.fail())


def test_worker_preserves_failed_attempts_and_skips_existing_seals(
    tmp_path: Path,
) -> None:
    config, topics, receipt = _initialize(tmp_path)
    first = _run(
        config,
        receipt.plan_path.read_bytes(),
        (topics[0].id,),
        lambda _config: lambda request: _failure("provider_failure"),
    )
    assert first.outcomes[0].status == "incomplete"

    second = _run(
        config,
        receipt.plan_path.read_bytes(),
        (topics[0].id,),
        lambda _config: lambda request: _success(request),
    )
    assert second.outcomes[0].status == "complete"
    attempts = receipt.output_dir / "work" / "topics" / topics[0].id / "attempts"
    assert (attempts / "000001" / "failure.json").is_file()
    assert (attempts / "000002").is_dir()


def test_worker_rejects_plan_bytes_that_conflict_with_installed_plan(
    tmp_path: Path,
) -> None:
    config, topics, receipt = _initialize(tmp_path)
    other_root = tmp_path / "other"
    other_config, _other_topics = _workspace(other_root, topic_count=3, run_id="other-worker")
    original_probe = retrieval._probe_repository
    retrieval._probe_repository = lambda _root: _binding()
    try:
        other = initialize_agentic_run(other_config, topic_ids=(topics[0].id,))
    finally:
        retrieval._probe_repository = original_probe

    with pytest.raises(AgenticRunnerError, match="plan|publication"):
        _run(config, other.plan_path.read_bytes(), (topics[0].id,), lambda _config: pytest.fail())

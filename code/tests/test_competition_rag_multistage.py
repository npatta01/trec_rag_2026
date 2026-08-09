from __future__ import annotations

import asyncio
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import trec_rag.competition_rag_multistage as competition_rag_multistage
from trec_rag.competition_rag import RagGenerationConfig
from trec_rag.competition_rag_multistage import (
    _multistage_identity,
    run_multistage_generation,
)
from trec_rag.generation_handoff import (
    SOURCE_CONTRACT,
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    SelectedCluster,
    TopicSourceReceipts,
    write_generation_handoff,
)
from trec_rag.narrative_blueprint_trial import (
    TRIAL_CONTRACT_VERSION,
    _bounded_identity,
    _bounded_private_root,
)


def _topic(index: int) -> GenerationTopic:
    topic_id = f"rag2026-{index}"
    group_id = f"{topic_id}-group"
    evidence_id = f"{topic_id}-evidence"
    cluster_id = f"{topic_id}-cluster"
    docid = f"DOC-{index}"
    text = f"Selected evidence for topic {index}."
    return GenerationTopic(
        topic_id=topic_id,
        narrative=f"Explain topic {index}.",
        groups=(
            EvidenceGroup(
                group_id=group_id,
                kind="generated_subnarrative",
                text=f"Topic {index} background",
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
                candidate_kind="extractive",
                docid=docid,
                document_rank=1,
                text=text,
                document_sha256=sha256(docid.encode()).hexdigest(),
                source_span=EvidenceSourceSpan(
                    start_char=0,
                    end_char=len(text),
                    start_byte=0,
                    end_byte=len(text.encode()),
                ),
            ),
        ),
        claim_hints=(
            ClaimHint(
                claim_id=f"{topic_id}-claim",
                group_id=group_id,
                kind="canonical",
                text=text,
                evidence_ids=(evidence_id,),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="1" * 64,
            retrieval_topic_sha256=sha256(topic_id.encode()).hexdigest(),
        ),
    )


def _handoff() -> GenerationHandoff:
    return GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="fixture-retrieval",
            producer_revision="fixture-revision",
        ),
        topics=(_topic(0), _topic(1)),
    )


def _config(tmp_path: Path) -> RagGenerationConfig:
    output_dir = tmp_path / "outputs/multistage"
    return RagGenerationConfig(
        schema_version="competition_rag_config_v2",
        handoff_manifest_path=tmp_path / "handoff.json",
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        team_id="castorini",
        run_id="multistage-test",
        run_desc="Frozen multi-stage test run.",
        topic_ids=None,
        concurrency=2,
        resume=False,
        overwrite=False,
        provider="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        structured_output="strict_schema",
        temperature=None,
        max_tokens=12000,
        timeout_seconds=900.0,
        transport_max_attempts=3,
    )


def _write_completed_topic(
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    operation_screen_crash_fallback: bool = False,
) -> Path:
    root = _bounded_private_root(config, topic)
    final_path = root / "evaluation/final/submission.jsonl"
    final_path.parent.mkdir(parents=True)
    record = {
        "metadata": {
            "team_id": config.team_id,
            "narrative_id": topic.topic_id,
            "narrative": topic.narrative,
            "run_id": f"{config.run_id}-final",
            "run_desc": config.run_desc,
        },
        "references": [topic.citation_docids[0]],
        "answer": [{"text": topic.claim_hints[0].text, "citations": [0]}],
    }
    final_bytes = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
    final_path.write_bytes(final_bytes)
    state = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "identity": _bounded_identity(config, handoff, topic),
        "stages": {"final": True},
        "operation_screen": {
            "crash_fallback": operation_screen_crash_fallback,
        },
        "stage_hashes": {
            "evaluation/final/submission.jsonl": sha256(final_bytes).hexdigest(),
        },
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return root


def test_create_publishes_all_validated_topics_in_handoff_order(tmp_path: Path) -> None:
    handoff = _handoff()
    config = _config(tmp_path)
    invoked: list[tuple[str, str]] = []

    async def fake_topic_runner(
        runner_config: RagGenerationConfig,
        runner_handoff: GenerationHandoff,
        topic: GenerationTopic,
        *,
        api_key: str,
        state_mode: str,
    ) -> Path:
        assert api_key == "fixture-key"
        invoked.append((topic.topic_id, state_mode))
        return _write_completed_topic(runner_config, runner_handoff, topic)

    asyncio.run(
        run_multistage_generation(
            config,
            handoff,
            api_key="fixture-key",
            topic_runner=fake_topic_runner,
        )
    )

    rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert invoked == [("rag2026-0", "create"), ("rag2026-1", "create")]
    assert [row["metadata"]["narrative_id"] for row in rows] == [
        "rag2026-0",
        "rag2026-1",
    ]
    assert all(
        row["metadata"]["run_id"] == f"{config.run_id}-final" for row in rows
    )
    identity = json.loads(
        (config.work_dir / "multistage_generation_identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert identity["handoff_manifest_sha256"] == handoff.manifest_sha256
    assert (config.work_dir / ".multistage-generation.lock").is_file()
    assert not config.output_path.with_name(f".{config.output_path.name}.lock").exists()


def test_resume_reuses_started_topic_and_creates_unstarted_topic(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    create_config = _config(tmp_path)
    resume_config = replace(create_config, resume=True)
    topics = handoff.topics
    create_config.work_dir.mkdir(parents=True)
    (create_config.work_dir / "multistage_generation_identity.json").write_text(
        json.dumps(_multistage_identity(create_config, handoff, topics)),
        encoding="utf-8",
    )
    existing_root = _write_completed_topic(create_config, handoff, topics[0])
    invoked: list[tuple[str, str]] = []

    async def fake_topic_runner(
        runner_config: RagGenerationConfig,
        runner_handoff: GenerationHandoff,
        topic: GenerationTopic,
        *,
        api_key: str,
        state_mode: str,
    ) -> Path:
        invoked.append((topic.topic_id, state_mode))
        if topic == topics[0]:
            assert state_mode == "resume"
            return existing_root
        assert state_mode == "create"
        return _write_completed_topic(runner_config, runner_handoff, topic)

    asyncio.run(
        run_multistage_generation(
            resume_config,
            handoff,
            api_key="fixture-key",
            topic_runner=fake_topic_runner,
        )
    )

    assert invoked == [("rag2026-0", "resume"), ("rag2026-1", "create")]
    assert len(resume_config.output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_crash_consumed_screen_fallback_publishes_with_durable_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff = _handoff()
    config = _config(tmp_path)

    async def fake_topic_runner(
        runner_config: RagGenerationConfig,
        runner_handoff: GenerationHandoff,
        topic: GenerationTopic,
        **_kwargs: object,
    ) -> Path:
        return _write_completed_topic(
            runner_config,
            runner_handoff,
            topic,
            operation_screen_crash_fallback=topic == handoff.topics[0],
        )

    asyncio.run(
        run_multistage_generation(
            config,
            handoff,
            api_key="fixture-key",
            topic_runner=fake_topic_runner,
        )
    )

    assert config.output_path.is_file()
    report = json.loads(
        (config.work_dir / "failures.json").read_text(encoding="utf-8")
    )
    assert report["failure_count"] == 0
    assert report["warning_count"] == 1
    assert report["warnings"] == [
        {
            "topic_id": "rag2026-0",
            "kind": "operation_screen_crash_fallback",
            "message": (
                "operation-screen call was crash-consumed; published validated "
                "draft fallback"
            ),
        }
    ]
    assert "topic rag2026-0: warning: operation-screen call was crash-consumed" in (
        capsys.readouterr().err
    )


def test_resume_refuses_changed_run_identity_before_dispatch(tmp_path: Path) -> None:
    handoff = _handoff()
    create_config = _config(tmp_path)
    create_config.work_dir.mkdir(parents=True)
    (create_config.work_dir / "multistage_generation_identity.json").write_text(
        json.dumps(_multistage_identity(create_config, handoff, handoff.topics)),
        encoding="utf-8",
    )
    changed_config = replace(
        create_config,
        resume=True,
        model="openai/different-model",
    )
    invoked: list[str] = []

    async def fake_topic_runner(*args: object, **kwargs: object) -> Path:
        invoked.append("called")
        raise AssertionError("topic runner must not be called")

    with pytest.raises(ValueError, match="identity differs"):
        asyncio.run(
            run_multistage_generation(
                changed_config,
                handoff,
                api_key="fixture-key",
                topic_runner=fake_topic_runner,
            )
        )

    assert invoked == []
    assert not changed_config.output_path.exists()


def test_missing_final_never_publishes_and_reports_each_failed_topic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff = _handoff()
    config = _config(tmp_path)
    completed: list[str] = []

    async def fake_topic_runner(
        runner_config: RagGenerationConfig,
        runner_handoff: GenerationHandoff,
        topic: GenerationTopic,
        *,
        api_key: str,
        state_mode: str,
    ) -> Path:
        if topic == handoff.topics[0]:
            raise RuntimeError("provider detail that must not escape")
        root = _write_completed_topic(runner_config, runner_handoff, topic)
        completed.append(topic.topic_id)
        return root

    with pytest.raises(RuntimeError, match="1 of 2 topics") as exc_info:
        asyncio.run(
            run_multistage_generation(
                config,
                handoff,
                api_key="fixture-key",
                topic_runner=fake_topic_runner,
            )
        )

    assert "provider detail" not in str(exc_info.value)
    assert completed == ["rag2026-1"]
    assert not config.output_path.exists()
    assert (
        "topic rag2026-0: RuntimeError: provider detail that must not escape"
        in capsys.readouterr().err
    )
    failure_report = json.loads(
        (config.work_dir / "failures.json").read_text(encoding="utf-8")
    )
    assert failure_report == {
        "failure_count": 1,
        "topic_count": 2,
        "warning_count": 0,
        "failures": [
            {
                "topic_id": "rag2026-0",
                "exception_type": "RuntimeError",
                "message": "provider detail that must not escape",
            }
        ],
        "warnings": [],
    }


def test_incomplete_topic_state_never_publishes(tmp_path: Path) -> None:
    handoff = _handoff()
    config = _config(tmp_path)

    async def fake_topic_runner(
        runner_config: RagGenerationConfig,
        runner_handoff: GenerationHandoff,
        topic: GenerationTopic,
        *,
        api_key: str,
        state_mode: str,
    ) -> Path:
        if topic == handoff.topics[1]:
            return _write_completed_topic(runner_config, runner_handoff, topic)
        root = _bounded_private_root(runner_config, topic)
        root.mkdir(parents=True)
        (root / "state.json").write_text(
            json.dumps(
                {
                    "trial_contract_version": TRIAL_CONTRACT_VERSION,
                    "identity": _bounded_identity(
                        runner_config, runner_handoff, topic
                    ),
                    "stages": {"final": False},
                    "stage_hashes": {},
                }
            ),
            encoding="utf-8",
        )
        return root

    with pytest.raises(RuntimeError, match="1 of 2 topics"):
        asyncio.run(
            run_multistage_generation(
                config,
                handoff,
                api_key="fixture-key",
                topic_runner=fake_topic_runner,
            )
        )

    assert not config.output_path.exists()
    failure_report = json.loads(
        (config.work_dir / "failures.json").read_text(encoding="utf-8")
    )
    assert failure_report["failures"] == [
        {
            "topic_id": "rag2026-0",
            "exception_type": "ValueError",
            "message": "multi-stage topic is incomplete: rag2026-0",
        }
    ]


def test_dry_run_reports_exact_budget_without_writes_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff = GenerationHandoff(
        producer=_handoff().producer,
        topics=(_topic(0), _topic(1), _topic(2)),
    )
    handoff_path = tmp_path / "outputs/retrieval/generation_handoff_manifest.json"
    write_generation_handoff(handoff_path, handoff)
    config_path = tmp_path / "multistage.yaml"
    config_path.write_text(
        """schema_version: competition_rag_config_v2
experiment:
  id: multistage-dry-run
  output_dir: outputs/multistage-dry-run
  mode: create
  topic_ids: [rag2026-0, rag2026-1, rag2026-2]
submission:
  team_id: castorini
  run_desc: Multi-stage dry-run fixture.
inputs:
  handoff_manifest: outputs/retrieval/generation_handoff_manifest.json
generation:
  type: openrouter
  api_base: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: openai/gpt-5.6-sol
  reasoning_effort: medium
  structured_output: strict_schema
  temperature: null
  max_tokens: 12000
  timeout_seconds: 900
  transport_max_attempts: 3
  concurrency: 2
""",
        encoding="utf-8",
    )
    provider_calls: list[str] = []

    async def fail_if_called(*args: object, **kwargs: object) -> None:
        provider_calls.append("called")

    monkeypatch.setattr(
        competition_rag_multistage,
        "run_multistage_generation",
        fail_if_called,
    )

    competition_rag_multistage.main(
        ["--config", str(config_path), "--dry-run"]
    )

    output = capsys.readouterr().out
    assert "topics=3,groups=3" in output
    assert "sol_routine:6,sol_max:9,luna_min:6,luna_max:9,provider:0" in output
    assert "concurrency=2" in output
    assert f"handoff={handoff_path}" in output
    assert f"output={tmp_path / 'outputs/multistage-dry-run/rag_output_trec_rag_2026.jsonl'}" in output
    assert f"work={tmp_path / 'outputs/multistage-dry-run/work'}" in output
    assert provider_calls == []
    assert not (tmp_path / "outputs/multistage-dry-run").exists()

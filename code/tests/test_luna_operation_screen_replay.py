"""Focused tests for the authenticated one-call operation-screen replay."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from trec_rag import luna_operation_screen_replay
from trec_rag.bounded_splice import splice_response_schema
from trec_rag.competition_rag import RagGenerationConfig
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
from trec_rag.luna_operation_screen_replay import (
    _may_start_operation_screen_call,
    finalize_operation_screen,
    load_operation_screen_source,
    render_operation_screen_prompt,
    run_luna_operation_screen_replay,
)
from trec_rag.narrative_blueprint import (
    project_blueprint,
    render_blueprint_writer_context,
    render_planner_prompt,
    serialize_blueprint_state,
    validate_blueprint,
)
from trec_rag.narrative_blueprint_trial import (
    TRIAL_CONTRACT_VERSION,
    _bounded_revision_prompt,
    _digest_json,
    _digest_text,
    LUNA_MODEL,
)


def _topic() -> GenerationTopic:
    evidence = (
        EvidencePassage(
            evidence_id="source-a",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="extractive",
            docid="DOC-A",
            document_rank=1,
            text="Evidence supporting the original background statement.",
            document_sha256="a" * 64,
            source_span=EvidenceSourceSpan(
                0,
                len("Evidence supporting the original background statement."),
                0,
                len("Evidence supporting the original background statement."),
            ),
        ),
        EvidencePassage(
            evidence_id="source-b",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=2,
            candidate_kind="extractive",
            docid="DOC-B",
            document_rank=2,
            text="Evidence fully supporting one additional concrete safeguard.",
            document_sha256="b" * 64,
            source_span=EvidenceSourceSpan(
                0,
                len("Evidence fully supporting one additional concrete safeguard."),
                0,
                len("Evidence fully supporting one additional concrete safeguard."),
            ),
        ),
    )
    return GenerationTopic(
        topic_id="233",
        narrative="Explain the background, compare the impacts, and recommend safeguards.",
        groups=(
            EvidenceGroup(
                group_id="group-1",
                kind="generated_subnarrative",
                text="Background, impacts, and safeguards",
                selected_clusters=(
                    SelectedCluster(
                        cluster_id="cluster-1",
                        ordinal=1,
                        representative_evidence_id="source-a",
                        evidence_ids=("source-a", "source-b"),
                    ),
                ),
            ),
        ),
        evidence=evidence,
        claim_hints=(
            ClaimHint("claim-a", "group-1", "canonical", "Background claim", ("source-a",)),
            ClaimHint("claim-b", "group-1", "canonical", "Impact claim", ("source-a",)),
            ClaimHint("claim-c", "group-1", "canonical", "Safeguard claim", ("source-b",)),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="1" * 64,
            retrieval_topic_sha256="2" * 64,
        ),
    )


def _blueprint(topic: GenerationTopic):
    payload = {
        "obligations": [
            {
                "label": "Background",
                "narrative_spans": ["background"],
                "priority": "must",
                "answer_mode": "explain",
                "target_words": 300,
                "selected_claim_aliases": ["c001"],
            },
            {
                "label": "Impacts",
                "narrative_spans": ["compare the impacts"],
                "priority": "should",
                "answer_mode": "compare",
                "target_words": 275,
                "selected_claim_aliases": ["c002"],
            },
            {
                "label": "Safeguards",
                "narrative_spans": ["recommend safeguards"],
                "priority": "should",
                "answer_mode": "recommend",
                "target_words": 275,
                "selected_claim_aliases": ["c003"],
            },
        ]
    }
    blueprint = validate_blueprint(topic, payload)
    return blueprint, project_blueprint(topic, blueprint)


def _draft(topic: GenerationTopic) -> dict[str, object]:
    return {
        "metadata": {
            "team_id": "castorini",
            "narrative_id": topic.topic_id,
            "narrative": topic.narrative,
            "run_id": "source-draft",
            "run_desc": "source draft",
        },
        "references": ["DOC-A"],
        "answer": [{"text": "The draft explains the background.", "citations": [0]}],
    }


def _audit_cards() -> list[dict[str, object]]:
    return [
        {
            "group_alias": "g001",
            "missing_detail": "One concrete safeguard is missing.",
            "evidence_aliases": ["e002"],
            "importance": "must",
            "omission_type": "named_detail",
            "rationale": "The complete narrative explicitly requests safeguards.",
            "replacement_answer_index": None,
            "card_id": "a001",
        }
    ]


def _revision_payload() -> dict[str, object]:
    return {
        "decision": "edit",
        "operations": [
            {
                "start_index": 1,
                "delete_count": 0,
                "new_object": {
                    "text": "The evidence also supports one concrete safeguard.",
                    "citations": ["DOC-B"],
                },
                "audit_card_ids": ["a001"],
            }
        ],
    }


def _write_json(path, value) -> str:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    return sha256(data.encode()).hexdigest()


def _write_source_root(tmp_path, topic: GenerationTopic, *, handoff_sha256: str = "h" * 64):
    root = tmp_path / "source"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    blueprint, projection = _blueprint(topic)
    planner_hash = sha256(render_planner_prompt(topic).encode()).hexdigest()
    writer_context = render_blueprint_writer_context(topic, blueprint, projection)
    blueprint_state = serialize_blueprint_state(
        topic,
        blueprint,
        projection,
        planner_prompt_sha256=planner_hash,
        writer_context_sha256=sha256(writer_context.encode()).hexdigest(),
    )
    draft = _draft(topic)
    cards = _audit_cards()
    stage_hashes = {
        "blueprint.state.json": _write_json(root / "blueprint.state.json", blueprint_state),
        "draft.record.json": _write_json(root / "draft.record.json", draft),
        "audit.merged.json": _write_json(root / "audit.merged.json", cards),
    }
    revision_prompt = _bounded_revision_prompt(
        topic,
        blueprint,
        projection,
        draft=draft,
        audit_cards=tuple(cards),
    )
    receipt = {
        "stage": "revision",
        "model": "openai/gpt-5.6-sol",
        "reasoning_effort": "medium",
        "prompt_sha256": _digest_text(revision_prompt),
        "schema_sha256": _digest_json(splice_response_schema()),
        "reservation_ordinal": 2,
        "outcome": "semantic_success",
        "transport_outcome": "response",
        "accepted_payload": _revision_payload(),
    }
    _write_json(receipts / "revision.json", receipt)
    call = {
        key: receipt[key]
        for key in (
            "stage",
            "model",
            "reasoning_effort",
            "prompt_sha256",
            "schema_sha256",
            "reservation_ordinal",
            "outcome",
            "transport_outcome",
        )
    }
    state = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "identity": {
            "handoff_manifest_sha256": handoff_sha256,
            "topic_context_sha256": topic.context_sha256,
            "sol_model": receipt["model"],
            "sol_reasoning_effort": receipt["reasoning_effort"],
        },
        "stages": {
            "planner": True,
            "draft": True,
            "audit_merge": True,
            "revision": True,
            "final": True,
        },
        "stage_hashes": stage_hashes,
        "calls": [call],
        "sol_reservations": [
            {
                "ordinal": 2,
                "role": "revision",
                "stage": "revision",
                "status": "semantic_returned",
                "receipt": "receipts/revision.json",
            }
        ],
    }
    _write_json(root / "state.json", state)
    return root


def _handoff(topic: GenerationTopic) -> GenerationHandoff:
    return GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="fixture-retrieval",
            producer_revision="deadbeef",
        ),
        topics=(topic,),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        team_id="castorini",
        run_desc="operation screen replay",
        run_id="screen-run",
    )


def test_source_loader_authenticates_revision_and_prompt_exposes_decision_context(tmp_path) -> None:
    topic = _topic()
    root = _write_source_root(tmp_path, topic)
    handoff = SimpleNamespace(manifest_sha256="h" * 64)

    source = load_operation_screen_source(
        root,
        config=_config(),
        handoff=handoff,
        topic=topic,
    )
    prompt = render_operation_screen_prompt(topic, source)

    assert source.draft["metadata"]["run_id"] == "screen-run-draft"
    assert len(source.operations) == 1
    assert set(source.source_hashes) == {
        "state.json",
        "blueprint.state.json",
        "draft.record.json",
        "audit.merged.json",
        "receipts/revision.json",
    }
    assert topic.narrative in prompt
    assert topic.evidence[1].text in prompt
    assert "DRAFT ANSWER OBJECTS" in prompt
    assert "op001" in prompt and "a001" in prompt
    assert "fully_supported" in prompt
    assert "replacement_safe" in prompt
    assert "coherent candidate subset" in prompt
    assert "another retained operation" in prompt
    assert "decisions only" in prompt
    assert "never rewrite" in prompt


def test_source_loader_rejects_registered_or_revision_receipt_tampering(tmp_path) -> None:
    topic = _topic()
    handoff = SimpleNamespace(manifest_sha256="h" * 64)
    root = _write_source_root(tmp_path, topic)
    (root / "audit.merged.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage hash mismatch"):
        load_operation_screen_source(root, config=_config(), handoff=handoff, topic=topic)

    root = _write_source_root(tmp_path / "receipt-case", topic)
    receipt_path = root / "receipts" / "revision.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prompt_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="revision receipt prompt or schema hash"):
        load_operation_screen_source(root, config=_config(), handoff=handoff, topic=topic)


def test_finalization_applies_only_accepted_operations_and_falls_back_atomically(tmp_path) -> None:
    topic = _topic()
    source = load_operation_screen_source(
        _write_source_root(tmp_path, topic),
        config=_config(),
        handoff=SimpleNamespace(manifest_sha256="h" * 64),
        topic=topic,
    )
    accepted_payload = {
        "decisions": [
            {
                "operation_id": "op001",
                "fully_supported": True,
                "atomic": True,
                "material": True,
                "nonredundant": True,
                "replacement_safe": True,
            }
        ]
    }

    final, result, used_fallback, error = finalize_operation_screen(
        topic,
        source.draft,
        source.operations,
        accepted_payload,
        config=_config(),
        run_id="screen-run-final",
    )
    assert used_fallback is False and error is None
    assert result is not None and len(result.accepted_operations) == 1
    assert final["answer"][-1]["text"] == "The evidence also supports one concrete safeguard."

    fallback, result, used_fallback, error = finalize_operation_screen(
        topic,
        source.draft,
        source.operations,
        {"decisions": []},
        config=_config(),
        run_id="screen-run-final",
    )
    assert result is None and used_fallback is True
    assert error is not None and "decide every operation" in error
    assert fallback["answer"] == source.draft["answer"]


def test_resume_never_repeats_an_ambiguous_or_completed_semantic_call() -> None:
    assert _may_start_operation_screen_call(
        state_mode="create",
        reservation_status="not_started",
        receipt_outcome=None,
    ) is True
    assert _may_start_operation_screen_call(
        state_mode="resume",
        reservation_status="terminal_transport_failure",
        receipt_outcome="terminal_transport_failure",
    ) is True
    assert _may_start_operation_screen_call(
        state_mode="resume",
        reservation_status="semantic_success",
        receipt_outcome="semantic_success",
    ) is False
    with pytest.raises(RuntimeError, match="ambiguous pending"):
        _may_start_operation_screen_call(
            state_mode="resume",
            reservation_status="pending",
            receipt_outcome=None,
        )


def test_runner_writes_one_call_manifest_and_screened_final(tmp_path, monkeypatch) -> None:
    topic = _topic()
    handoff = _handoff(topic)
    handoff_path = tmp_path / "handoff" / "generation_handoff_manifest.json"
    write_generation_handoff(handoff_path, handoff)
    source_root = _write_source_root(
        tmp_path,
        topic,
        handoff_sha256=handoff.manifest_sha256,
    )
    output_dir = tmp_path / "output"
    config = RagGenerationConfig(
        schema_version="competition_rag_config_v2",
        handoff_manifest_path=handoff_path,
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        team_id="castorini",
        run_id="operation-screen-run",
        run_desc="operation screen fixture",
        topic_ids=(topic.topic_id,),
        concurrency=1,
        resume=False,
        overwrite=False,
        provider="openrouter",
        api_base="https://openrouter.invalid/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model=LUNA_MODEL,
        reasoning_effort="medium",
        structured_output="strict_schema",
        temperature=None,
        max_tokens=2000,
        timeout_seconds=30.0,
        transport_max_attempts=1,
    )
    screen_payload = {
        "decisions": [
            {
                "operation_id": "op001",
                "fully_supported": True,
                "atomic": True,
                "material": True,
                "nonredundant": True,
                "replacement_safe": True,
            }
        ]
    }

    async def fake_provider_call(_generator, **kwargs):
        receipt = {
            "stage": kwargs["stage"],
            "model": kwargs["model"],
            "reasoning_effort": kwargs["reasoning_effort"],
            "prompt_sha256": _digest_text(kwargs["user_prompt"]),
            "system_prompt_sha256": _digest_text(kwargs["system_prompt"]),
            "schema_sha256": _digest_json(kwargs["response_schema"]),
            "reservation_ordinal": kwargs["reservation_ordinal"],
            "outcome": "semantic_success",
            "transport_outcome": "response",
            "provider_cost": 0.001,
            "usage": {"cost": 0.001},
            "accepted_payload": screen_payload,
            "raw_response": {"usage": {"cost": 0.001}},
            "latency_seconds": 0.1,
            "error": None,
        }
        _write_json(kwargs["root"] / "receipts" / "operation-screen.json", receipt)
        call = {
            key: receipt[key]
            for key in (
                "stage",
                "model",
                "reasoning_effort",
                "prompt_sha256",
                "schema_sha256",
                "reservation_ordinal",
                "outcome",
                "transport_outcome",
                "provider_cost",
                "usage",
                "latency_seconds",
                "error",
            )
        }
        return screen_payload, call

    monkeypatch.setattr(
        luna_operation_screen_replay,
        "_bounded_provider_call",
        fake_provider_call,
    )
    root = asyncio.run(
        run_luna_operation_screen_replay(
            config,
            handoff,
            topic,
            source_root=source_root,
            api_key="fixture-key",
            state_mode="create",
        )
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    final = json.loads(
        (root / "evaluation" / "final" / "submission.jsonl").read_text(encoding="utf-8")
    )
    assert manifest["luna_calls"] == 1
    assert manifest["sol_calls"] == 0
    assert manifest["accepted_operations"] == 1
    assert manifest["rejected_operations"] == 0
    assert manifest["used_draft_fallback"] is False
    assert final["answer"][-1]["text"] == "The evidence also supports one concrete safeguard."

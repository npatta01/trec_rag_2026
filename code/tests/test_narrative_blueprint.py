from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

import trec_rag.narrative_blueprint_trial as narrative_blueprint_trial
from trec_rag.competition_rag import RagGenerationConfig, SemanticCompletionError
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
)
from trec_rag.narrative_blueprint import (
    BlueprintValidationError,
    load_blueprint_state,
    planner_response_schema,
    project_blueprint,
    render_blueprint_writer_context,
    render_planner_prompt,
    serialize_blueprint_state,
    validate_blueprint,
)
from trec_rag.narrative_blueprint_trial import (
    TRIAL_CONTRACT_VERSION,
    _bounded_provider_call,
    _bounded_recover_pending,
    _bounded_manifest,
    _bounded_record_draft_failure,
    _bounded_rebind_candidate,
    _bounded_recovered_payload,
    _audit_card_docids,
    _digest_json,
    _digest_text,
    _bounded_revision_prompt,
    audit_response_schema,
    merge_audit_cards,
)
from trec_rag.bounded_splice import splice_response_schema


def _topic_fixture() -> GenerationTopic:
    evidence = (
        EvidencePassage(
            evidence_id="g1-linked",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="extractive",
            docid="DOCID_SENTINEL_ALPHA",
            document_rank=1,
            text="AUTHORITY_SENTINEL_ALPHA",
            document_sha256="a" * 64,
            source_span=EvidenceSourceSpan(0, 24, 0, 24),
        ),
        EvidencePassage(
            evidence_id="g1-unlinked",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=2,
            candidate_kind="extractive",
            docid="DOCID_SENTINEL_BETA",
            document_rank=2,
            text="AUTHORITY_SENTINEL_BETA",
            document_sha256="b" * 64,
            source_span=EvidenceSourceSpan(30, 53, 30, 53),
        ),
        EvidencePassage(
            evidence_id="g2-linked",
            group_id="group-2",
            cluster_id="cluster-2",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="extractive",
            docid="DOCID_SENTINEL_GAMMA",
            document_rank=3,
            text="AUTHORITY_SENTINEL_GAMMA",
            document_sha256="c" * 64,
            source_span=EvidenceSourceSpan(60, 84, 60, 84),
        ),
    )
    return GenerationTopic(
        topic_id="rag2026-test",
        narrative=(
            "Explain the policy background and compare its social impacts. "
            "Recommend safeguards for implementation."
        ),
        groups=(
            EvidenceGroup(
                group_id="group-1",
                kind="generated_subnarrative",
                text="GROUP_SENTINEL_BACKGROUND",
                selected_clusters=(
                    SelectedCluster(
                        cluster_id="cluster-1",
                        ordinal=1,
                        representative_evidence_id="g1-linked",
                        evidence_ids=("g1-linked", "g1-unlinked"),
                    ),
                ),
            ),
            EvidenceGroup(
                group_id="group-2",
                kind="generated_subnarrative",
                text="GROUP_SENTINEL_IMPACTS",
                selected_clusters=(
                    SelectedCluster(
                        cluster_id="cluster-2",
                        ordinal=1,
                        representative_evidence_id="g2-linked",
                        evidence_ids=("g2-linked",),
                    ),
                ),
            ),
        ),
        evidence=evidence,
        claim_hints=(
            ClaimHint(
                claim_id="native-claim-alpha",
                group_id="group-1",
                kind="canonical",
                text="CLAIM_SENTINEL_BACKGROUND",
                evidence_ids=("g1-linked",),
            ),
            ClaimHint(
                claim_id="native-claim-beta",
                group_id="group-2",
                kind="canonical",
                text="CLAIM_SENTINEL_IMPACTS",
                evidence_ids=("g2-linked",),
            ),
            ClaimHint(
                claim_id="native-claim-gamma",
                group_id="group-1",
                kind="canonical",
                text="CLAIM_SENTINEL_SAFEGUARDS",
                evidence_ids=("g1-linked",),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="1" * 64,
            retrieval_topic_sha256="2" * 64,
        ),
    )


def _valid_payload() -> dict[str, object]:
    return {
        "obligations": [
            {
                "label": "Explain policy background",
                "narrative_spans": ["policy background"],
                "priority": "must",
                "answer_mode": "explain",
                "target_words": 300,
                "selected_claim_aliases": ["c001"],
            },
            {
                "label": "Compare social impacts",
                "narrative_spans": ["compare its social impacts"],
                "priority": "should",
                "answer_mode": "compare",
                "target_words": 275,
                "selected_claim_aliases": ["c002"],
            },
            {
                "label": "Recommend safeguards",
                "narrative_spans": ["Recommend safeguards"],
                "priority": "could",
                "answer_mode": "recommend",
                "target_words": 275,
                "selected_claim_aliases": ["c003"],
            },
        ]
    }


def test_planner_prompt_is_compact_and_contains_no_authoritative_evidence() -> None:
    topic = _topic_fixture()

    prompt = render_planner_prompt(topic)

    assert topic.narrative in prompt
    assert "g001" in prompt and "g002" in prompt
    assert "c001" in prompt and "c003" in prompt
    assert topic.groups[0].text in prompt
    assert topic.claim_hints[0].text in prompt
    for evidence in topic.evidence:
        assert evidence.text not in prompt
        assert evidence.evidence_id not in prompt
        assert evidence.docid not in prompt
    for claim in topic.claim_hints:
        assert claim.claim_id not in prompt


def test_planner_schema_is_strict_and_has_exact_contract() -> None:
    schema = planner_response_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["obligations"]
    obligations = schema["properties"]["obligations"]
    assert obligations["minItems"] == 3
    assert obligations["maxItems"] == 8
    assert obligations["items"]["additionalProperties"] is False
    assert obligations["items"]["required"] == [
        "label",
        "narrative_spans",
        "priority",
        "answer_mode",
        "target_words",
        "selected_claim_aliases",
    ]


def test_audit_schema_requires_every_declared_card_property() -> None:
    schema = audit_response_schema()

    card = schema["properties"]["cards"]["items"]
    assert card["required"] == [
        "group_alias",
        "missing_detail",
        "evidence_aliases",
        "importance",
        "omission_type",
        "rationale",
        "replacement_answer_index",
    ]


def _audit_card(*, group_alias: str, detail: str, importance: str, omission_type: str) -> dict[str, object]:
    return {
        "group_alias": group_alias,
        "missing_detail": detail,
        "evidence_aliases": ["e001" if group_alias == "g001" else "e003"],
        "importance": importance,
        "omission_type": omission_type,
        "rationale": "The detail is supported by the selected evidence.",
        "replacement_answer_index": 0,
    }


def test_merged_audit_cards_get_stable_ids_without_changing_rank() -> None:
    topic = _topic_fixture()
    cards_by_group = {
        "group-1": (
            _audit_card(
                group_alias="g001",
                detail="lower-ranked detail",
                importance="should",
                omission_type="missing",
            ),
            _audit_card(
                group_alias="g001",
                detail="highest-ranked detail",
                importance="must",
                omission_type="missing quantity/example",
            ),
        ),
        "group-2": (
            _audit_card(
                group_alias="g002",
                detail="third-ranked detail",
                importance="could",
                omission_type="too generic",
            ),
        ),
    }

    merged = merge_audit_cards(topic, cards_by_group)

    assert [card["missing_detail"] for card in merged] == [
        "highest-ranked detail",
        "lower-ranked detail",
        "third-ranked detail",
    ]
    assert [card["card_id"] for card in merged] == ["a001", "a002", "a003"]
    assert merged == merge_audit_cards(topic, cards_by_group)


def test_splice_revision_prompt_indexes_draft_and_uses_audit_card_ids() -> None:
    topic = _topic_fixture()
    blueprint = validate_blueprint(topic, _valid_payload())
    projection = project_blueprint(topic, blueprint)
    cards = merge_audit_cards(
        topic,
        {
            "group-1": (_audit_card(
                group_alias="g001",
                detail="missing detail",
                importance="must",
                omission_type="missing",
            ),),
            "group-2": (),
        },
    )
    draft = {
        "references": ["DOCID_SENTINEL_ALPHA"],
        "answer": [
            {"text": "Draft first object.", "citations": [0]},
            {"text": "Draft second object.", "citations": [0]},
        ],
    }

    prompt = _bounded_revision_prompt(
        topic,
        blueprint,
        projection,
        draft=draft,
        audit_cards=cards,
    )

    assert "DRAFT ANSWER OBJECTS" in prompt
    assert "[0]" in prompt and "[1]" in prompt
    assert "a001" in prompt
    assert "operations" in prompt
    assert "one self-contained sentence stating one atomic claim" in prompt
    assert "single strongest citation" in prompt
    assert "linked to that operation's named audit cards" in prompt
    assert "Return a complete replacement organizer JSON object" not in prompt


def test_audit_card_docids_route_only_each_cards_linked_evidence() -> None:
    topic = _topic_fixture()
    cards = (
        {
            "card_id": "a001",
            "evidence_aliases": ("e001",),
        },
        {
            "card_id": "a002",
            "evidence_aliases": ("e001", "e002"),
        },
    )

    assert _audit_card_docids(topic, cards) == {
        "a001": ("DOCID_SENTINEL_ALPHA",),
        "a002": ("DOCID_SENTINEL_ALPHA", "DOCID_SENTINEL_BETA"),
    }


@pytest.mark.parametrize("evidence_aliases", [(), ("e999",)])
def test_audit_card_docids_reject_invalid_evidence_aliases(evidence_aliases) -> None:
    with pytest.raises(ValueError, match="evidence_aliases"):
        _audit_card_docids(
            _topic_fixture(),
            ({"card_id": "a001", "evidence_aliases": evidence_aliases},),
        )


def test_splice_trial_uses_a_new_contract_version() -> None:
    assert TRIAL_CONTRACT_VERSION != "bounded_narrative_revision_trial_v1"


def test_recovered_revision_payload_requires_current_prompt_and_schema_hashes() -> None:
    payload = {"decision": "keep_draft", "operations": []}
    state = {
        "recovered_payloads": {"revision": payload},
        "calls": [
            {
                "stage": "revision",
                "prompt_sha256": _digest_text("old splice prompt"),
                "schema_sha256": _digest_json(splice_response_schema()),
            }
        ],
    }

    with pytest.raises(ValueError, match="revision recovery hash mismatch"):
        _bounded_recovered_payload(
            state,
            "revision",
            expected_prompt_sha256=_digest_text("current splice prompt"),
            expected_schema_sha256=_digest_json(splice_response_schema()),
        )

    assert state["recovered_payloads"]["revision"] == payload


def test_invalid_draft_failure_seals_without_repair_audit_or_revision(tmp_path) -> None:
    root = tmp_path / "bounded"
    root.mkdir()
    topic = _topic_fixture()
    state = {
        "identity": {
            "handoff_manifest_sha256": "h" * 64,
            "topic_context_sha256": "t" * 64,
        },
        "stages": {
            "planner": True,
            "draft": False,
            "audit_groups": [],
            "audit_merge": False,
            "revision": False,
            "final": False,
        },
        "luna_reservations": [{"ordinal": 1, "stage": "planner"}],
        "sol_reservations": [{"ordinal": 1, "role": "draft"}],
        "calls": [],
    }

    _bounded_record_draft_failure(
        root,
        state,
        topic,
        reason="draft candidate failed deterministic validation",
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert state["failure"] == "draft candidate failed deterministic validation"
    assert manifest["final"] is None
    assert state["stages"]["audit_groups"] == []
    assert state["stages"]["audit_merge"] is False
    assert state["stages"]["revision"] is False
    assert state["stages"]["final"] is False
    assert [item["role"] for item in state["sol_reservations"]] == ["draft"]


def test_retryable_http_status_is_transport_not_semantic_failure(tmp_path) -> None:
    root = tmp_path / "bounded"
    (root / "receipts").mkdir(parents=True)

    class TransientHttpFailure:
        def complete_json(self, **_kwargs):
            raise SemanticCompletionError(
                "OpenRouter HTTP 503; no repair call was made",
                {"http_status": 503},
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        payload, call = asyncio.run(
            _bounded_provider_call(
                TransientHttpFailure(),
                root=root,
                api_key="fixture-key",
                topic_id="rag2026-test",
                stage="draft",
                model="openai/gpt-5.6-sol",
                reasoning_effort="medium",
                system_prompt="system",
                user_prompt="user",
                response_schema={"type": "object"},
                executor=executor,
                reservation_ordinal=1,
            )
        )

    assert payload is None
    assert call["outcome"] == "terminal_transport_failure"
    assert call["transport_outcome"] == "failure"


def test_pending_reservation_ignores_a_stale_receipt_ordinal(tmp_path) -> None:
    root = tmp_path / "bounded"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "draft.json").write_text(
        json.dumps(
            {
                "stage": "draft",
                "reservation_ordinal": 1,
                "outcome": "terminal_transport_failure",
                "transport_outcome": "failure",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "luna_reservations": [],
        "sol_reservations": [
            {
                "ordinal": 2,
                "role": "draft",
                "stage": "draft",
                "status": "pending",
                "receipt": "receipts/draft.json",
            }
        ],
        "recovered_payloads": {},
        "calls": [],
    }

    _bounded_recover_pending(root, state)

    assert state["sol_reservations"][0]["status"] == "crash_consumed"
    assert state["calls"] == []


def test_recovery_rejects_semantic_success_without_an_accepted_payload(
    tmp_path,
) -> None:
    root = tmp_path / "bounded"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "draft.json").write_text(
        json.dumps(
            {
                "stage": "draft",
                "reservation_ordinal": 1,
                "outcome": "semantic_success",
                "transport_outcome": "response",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "luna_reservations": [],
        "sol_reservations": [
            {
                "ordinal": 1,
                "role": "draft",
                "stage": "draft",
                "status": "pending",
                "receipt": "receipts/draft.json",
            }
        ],
        "recovered_payloads": {},
        "calls": [],
    }

    _bounded_recover_pending(root, state)

    assert state["sol_reservations"][0]["status"] == "semantic_error"
    assert state["recovered_payloads"] == {}


def test_manifest_tolerates_a_recovered_call_without_model(tmp_path) -> None:
    root = tmp_path / "bounded"
    root.mkdir()
    topic = _topic_fixture()
    state = {
        "identity": {
            "handoff_manifest_sha256": "h" * 64,
            "topic_context_sha256": "t" * 64,
        },
        "stages": {},
        "sol_reservations": [],
        "calls": [{"stage": "planner", "transport_outcome": "response"}],
    }

    manifest = _bounded_manifest(root, state, topic=topic, draft=None, final=None)

    assert manifest["luna_call_count"] == 0


def test_invalid_initial_draft_uses_repair_then_skips_audit_and_revision(
    tmp_path,
    monkeypatch,
) -> None:
    topic = _topic_fixture()
    handoff = GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="fixture-retrieval",
            producer_revision="fixture-revision",
        ),
        topics=(topic,),
    )
    output_dir = tmp_path / "outputs/bounded-draft-repair"
    config = RagGenerationConfig(
        schema_version="competition_rag_config_v2",
        handoff_manifest_path=tmp_path / "handoff.json",
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        team_id="castorini",
        run_id="bounded-draft-repair",
        run_desc="Bounded draft repair fixture.",
        topic_ids=(topic.topic_id,),
        concurrency=1,
        resume=False,
        overwrite=False,
        provider="openrouter",
        api_base="https://openrouter.invalid/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        structured_output="strict_schema",
        temperature=None,
        max_tokens=12000,
        timeout_seconds=30.0,
        transport_max_attempts=1,
    )
    invalid_draft = {
        "references": [topic.citation_docids[0]],
        "answer": [
            {
                "text": " ".join(["word"] * 1025),
                "citations": [topic.citation_docids[0]],
            }
        ],
    }
    repaired_draft = {
        "references": [topic.citation_docids[0]],
        "answer": [
            {
                "text": "The repaired answer is locally valid.",
                "citations": [topic.citation_docids[0]],
            }
        ],
    }
    payloads = {
        "planner": _valid_payload(),
        "draft": invalid_draft,
        "repair": repaired_draft,
    }
    invoked: list[str] = []
    created_generators = []
    generator_class = narrative_blueprint_trial.OpenRouterJsonGenerator

    def tracking_generator(**kwargs):
        generator = generator_class(**kwargs)
        created_generators.append(generator)
        return generator

    async def fake_provider_call(_generator, **kwargs):
        stage = kwargs["stage"]
        invoked.append(stage)
        payload = payloads[stage]
        return payload, {
            "stage": stage,
            "model": kwargs["model"],
            "reasoning_effort": kwargs["reasoning_effort"],
            "prompt_sha256": _digest_text(kwargs["user_prompt"]),
            "schema_sha256": _digest_json(kwargs["response_schema"]),
            "reservation_ordinal": kwargs["reservation_ordinal"],
            "outcome": "semantic_success",
            "latency_seconds": 0.01,
            "provider_cost": 0.0,
            "usage": {},
            "transport_outcome": "response",
            "error": None,
        }

    monkeypatch.setattr(
        narrative_blueprint_trial,
        "_bounded_provider_call",
        fake_provider_call,
    )
    monkeypatch.setattr(
        narrative_blueprint_trial,
        "OpenRouterJsonGenerator",
        tracking_generator,
    )

    root = asyncio.run(
        narrative_blueprint_trial._run_bounded_revision(
            config,
            handoff,
            topic,
            api_key="fixture-key",
            state_mode="create",
        )
    )

    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    final = json.loads(
        (root / "evaluation/final/submission.jsonl").read_text(encoding="utf-8")
    )
    assert invoked == ["planner", "draft", "repair"]
    assert [item["role"] for item in state["sol_reservations"]] == [
        "draft",
        "repair",
    ]
    assert state["stages"]["audit_groups"] == []
    assert state["stages"]["revision"] is False
    assert state["stages"]["final"] is True
    assert final["answer"][0]["text"] == "The repaired answer is locally valid."
    assert len(created_generators) == 2
    assert all(generator._http_client.is_closed for generator in created_generators)


def test_bounded_rebind_prunes_references_orphaned_by_splice_replacement() -> None:
    topic = _topic_fixture()
    candidate = {
        "metadata": {
            "team_id": "castorini",
            "narrative_id": topic.topic_id,
            "narrative": topic.narrative,
            "run_id": "draft-run",
            "run_desc": "bounded trial",
        },
        "references": ["DOCID_SENTINEL_ALPHA", "DOCID_SENTINEL_BETA"],
        "answer": [
            {
                "text": "Only the second reference remains cited after replacement.",
                "citations": [1],
            }
        ],
    }

    rebound = _bounded_rebind_candidate(
        candidate,
        topic=topic,
        config=SimpleNamespace(team_id="castorini", run_desc="bounded trial"),
        run_id="final-run",
    )

    assert rebound["references"] == ["DOCID_SENTINEL_BETA"]
    assert rebound["answer"] == [
        {
            "text": "Only the second reference remains cited after replacement.",
            "citations": [0],
        }
    ]
    assert rebound["metadata"]["run_id"] == "final-run"
    assert candidate["references"] == ["DOCID_SENTINEL_ALPHA", "DOCID_SENTINEL_BETA"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(obligations=p["obligations"][:2]), "3-8"),
        (lambda p: [o.update(priority="should") for o in p["obligations"]], "must"),
        (lambda p: p["obligations"][0].update(answer_mode="invent"), "answer_mode"),
        (lambda p: p["obligations"][0].update(narrative_spans=["not present"]), "narrative"),
        (lambda p: p["obligations"][0].update(selected_claim_aliases=["c999"]), "claim alias"),
        (lambda p: p["obligations"][0].update(selected_claim_aliases=[]), "claim"),
        (lambda p: p["obligations"][0].update(target_words=1), "850-950"),
    ],
)
def test_blueprint_validation_fails_closed(mutate, message) -> None:
    payload = deepcopy(_valid_payload())
    mutate(payload)
    with pytest.raises(BlueprintValidationError, match=message):
        validate_blueprint(_topic_fixture(), payload)


def test_blueprint_accepts_disjoint_spans_after_nfkc_casefold_and_whitespace() -> None:
    payload = _valid_payload()
    payload["obligations"][0]["narrative_spans"] = [
        "POLICY   BACKGROUND",
        "sOcIaL impacts",
    ]

    blueprint = validate_blueprint(_topic_fixture(), payload)

    assert blueprint.obligations[0].narrative_spans == (
        "POLICY   BACKGROUND",
        "sOcIaL impacts",
    )


def test_blueprint_rejects_duplicate_normalized_span_sets_and_aliases() -> None:
    payload = _valid_payload()
    payload["obligations"][1]["narrative_spans"] = ["POLICY BACKGROUND"]
    with pytest.raises(BlueprintValidationError, match="duplicate.*span"):
        validate_blueprint(_topic_fixture(), payload)

    payload = _valid_payload()
    payload["obligations"][0]["selected_claim_aliases"] = ["c001", "c001"]
    with pytest.raises(BlueprintValidationError, match="duplicate.*claim"):
        validate_blueprint(_topic_fixture(), payload)


def test_must_widens_to_full_selected_groups_but_should_stays_claim_linked() -> None:
    topic = _topic_fixture()
    blueprint = validate_blueprint(topic, _valid_payload())

    projection = project_blueprint(topic, blueprint)

    assert projection.obligations[0].evidence_ids == (
        "g1-linked",
        "g1-unlinked",
    )
    assert projection.obligations[1].evidence_ids == ("g2-linked",)
    assert projection.evidence_ids == (
        "g1-linked",
        "g1-unlinked",
        "g2-linked",
    )
    assert projection.citation_docids == (
        "DOCID_SENTINEL_ALPHA",
        "DOCID_SENTINEL_BETA",
        "DOCID_SENTINEL_GAMMA",
    )


def test_writer_context_maps_obligations_and_renders_each_passage_once() -> None:
    topic = _topic_fixture()
    blueprint = validate_blueprint(topic, _valid_payload())
    projection = project_blueprint(topic, blueprint)

    context = render_blueprint_writer_context(topic, blueprint, projection)

    assert "NARRATIVE BLUEPRINT" in context
    assert "g1-linked" not in context
    assert "DOCID_SENTINEL_ALPHA" in context
    assert context.count("AUTHORITY_SENTINEL_ALPHA") == 1
    assert context.count("AUTHORITY_SENTINEL_BETA") == 1
    assert context.count("AUTHORITY_SENTINEL_GAMMA") == 1
    assert "OBLIGATION 1" in context
    assert "e001" in context and "e003" in context
    assert "native-claim-alpha" not in context
    assert "CLAIM_SENTINEL_BACKGROUND" in context


def test_blueprint_state_round_trip_is_authenticated_and_canonical() -> None:
    topic = _topic_fixture()
    blueprint = validate_blueprint(topic, _valid_payload())
    projection = project_blueprint(topic, blueprint)
    planner_hash = sha256(render_planner_prompt(topic).encode()).hexdigest()
    writer_context = render_blueprint_writer_context(topic, blueprint, projection)
    writer_hash = sha256(writer_context.encode()).hexdigest()
    state = serialize_blueprint_state(
        topic,
        blueprint,
        projection,
        planner_prompt_sha256=planner_hash,
        writer_context_sha256=writer_hash,
    )

    assert load_blueprint_state(topic, deepcopy(state), planner_prompt_sha256=planner_hash) == (
        blueprint,
        projection,
    )
    assert state["state_sha256"] == sha256(
        json.dumps(
            {key: value for key, value in state.items() if key != "state_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    for key in ("topic_context_sha256", "planner_prompt_sha256", "writer_context_sha256"):
        tampered = deepcopy(state)
        tampered[key] = "f" * 64
        with pytest.raises(BlueprintValidationError, match="state"):
            load_blueprint_state(topic, tampered, planner_prompt_sha256=planner_hash)

    tampered = deepcopy(state)
    tampered["blueprint"]["obligations"][0]["label"] = "changed"
    with pytest.raises(BlueprintValidationError, match="state"):
        load_blueprint_state(topic, tampered, planner_prompt_sha256=planner_hash)

    tampered = deepcopy(state)
    tampered["projection"]["evidence_ids"] = ["g2-linked"]
    with pytest.raises(BlueprintValidationError, match="state"):
        load_blueprint_state(topic, tampered, planner_prompt_sha256=planner_hash)

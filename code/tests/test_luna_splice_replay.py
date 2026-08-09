from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from trec_rag.bounded_splice import SpliceValidationError
from trec_rag.generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationTopic,
    SelectedCluster,
    TopicSourceReceipts,
)
from trec_rag.luna_splice_replay import (
    _may_start_luna_call,
    finalize_luna_splice_payload,
    assemble_luna_splice_candidate,
    load_replay_source,
    luna_splice_response_schema,
    render_luna_splice_prompt,
)
from trec_rag.narrative_blueprint import (
    project_blueprint,
    render_blueprint_writer_context,
    render_planner_prompt,
    serialize_blueprint_state,
    validate_blueprint,
)
from trec_rag.narrative_blueprint_trial import TRIAL_CONTRACT_VERSION


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
            source_span=EvidenceSourceSpan(0, 54, 0, 54),
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
            text="Evidence supporting one additional concrete safeguard.",
            document_sha256="b" * 64,
            source_span=EvidenceSourceSpan(0, 54, 0, 54),
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


def test_whole_answer_prompt_exposes_direct_evidence_and_requires_meaningful_edits() -> None:
    topic = _topic()
    blueprint, projection = _blueprint(topic)

    prompt = render_luna_splice_prompt(
        topic,
        blueprint,
        projection,
        draft=_draft(topic),
    )

    assert topic.narrative in prompt
    assert topic.evidence[1].text in prompt
    assert "DRAFT ANSWER OBJECTS" in prompt and "[0]" in prompt
    assert "at most three" in prompt
    assert "Prefer insertion" in prompt
    assert "clearly more useful than everything removed" in prompt
    assert "audit_card_ids` must contain selected-evidence aliases" in prompt
    assert "MERGED AUDIT CARDS" not in prompt
    assert "never return a full answer" in prompt


def test_luna_splice_schema_and_local_validation_cap_operations_at_three() -> None:
    topic = _topic()
    draft = _draft(topic)
    operation = {
        "start_index": 1,
        "delete_count": 0,
        "new_object": {
            "text": "The evidence also supports one concrete safeguard.",
            "citations": ["DOC-B"],
        },
        "audit_card_ids": ["e002"],
    }

    schema = luna_splice_response_schema()
    assert schema["properties"]["operations"]["maxItems"] == 3

    assembled = assemble_luna_splice_candidate(
        topic,
        draft,
        {"decision": "edit", "operations": [operation]},
    )
    assert assembled["references"] == ["DOC-A", "DOC-B"]
    assert assembled["answer"][-1] == {
        "text": "The evidence also supports one concrete safeguard.",
        "citations": [1],
    }

    too_many = {"decision": "edit", "operations": [deepcopy(operation) for _ in range(4)]}
    with pytest.raises(SpliceValidationError, match="at most three operations"):
        assemble_luna_splice_candidate(topic, draft, too_many)


def _write_source_root(tmp_path, topic: GenerationTopic):
    root = tmp_path / "source"
    root.mkdir()
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
    files = {
        "blueprint.state.json": blueprint_state,
        "draft.record.json": _draft(topic),
    }
    hashes = {}
    for name, value in files.items():
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (root / name).write_text(data, encoding="utf-8")
        hashes[name] = sha256(data.encode()).hexdigest()
    state = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "identity": {
            "handoff_manifest_sha256": "h" * 64,
            "topic_context_sha256": topic.context_sha256,
        },
        "stages": {"planner": True, "draft": True},
        "stage_hashes": hashes,
    }
    (root / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_replay_source_loader_authenticates_handoff_topic_and_registered_files(tmp_path) -> None:
    topic = _topic()
    root = _write_source_root(tmp_path, topic)
    config = SimpleNamespace(team_id="castorini", run_desc="replay", run_id="replay-run")
    handoff = SimpleNamespace(manifest_sha256="h" * 64)

    blueprint, projection, draft, source_hashes = load_replay_source(
        root,
        config=config,
        handoff=handoff,
        topic=topic,
    )

    assert blueprint == _blueprint(topic)[0]
    assert projection == _blueprint(topic)[1]
    assert draft["metadata"]["run_id"] == "replay-run-draft"
    assert set(source_hashes) == {"state.json", "blueprint.state.json", "draft.record.json"}

    bad_handoff = SimpleNamespace(manifest_sha256="x" * 64)
    with pytest.raises(ValueError, match="handoff digest"):
        load_replay_source(root, config=config, handoff=bad_handoff, topic=topic)

    (root / "draft.record.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage hash mismatch"):
        load_replay_source(root, config=config, handoff=handoff, topic=topic)


def test_invalid_luna_payload_falls_back_to_validated_draft() -> None:
    topic = _topic()
    config = SimpleNamespace(team_id="castorini", run_desc="replay")
    invalid = {
        "decision": "edit",
        "operations": [
            {
                "start_index": 1,
                "delete_count": 0,
                "new_object": {
                    "text": "This citation is not linked to the named evidence alias.",
                    "citations": ["DOC-B"],
                },
                "audit_card_ids": ["e001"],
            }
        ],
    }

    final, used_fallback, error = finalize_luna_splice_payload(
        topic,
        _draft(topic),
        invalid,
        config=config,
        run_id="replay-run-final",
    )

    assert used_fallback is True
    assert "linked audit-card evidence" in error
    assert final["answer"] == _draft(topic)["answer"]
    assert final["metadata"]["run_id"] == "replay-run-final"


def test_resume_never_repeats_an_ambiguous_pending_semantic_call() -> None:
    with pytest.raises(RuntimeError, match="ambiguous pending Luna call"):
        _may_start_luna_call(
            state_mode="resume",
            reservation_status="pending",
            receipt_outcome=None,
        )

    assert _may_start_luna_call(
        state_mode="resume",
        reservation_status="terminal_transport_failure",
        receipt_outcome="terminal_transport_failure",
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.topic213_controlled_generator_benchmark import (
    GENERATION_SYSTEM_PROMPT,
    SingleCandidateJsonClient,
    _make_judge_client,
    _resolve_config_path,
    build_candidate_generation,
    build_frozen_evidence,
    build_generation_payload,
    choose_winner,
    filter_supported_generation,
    semantic_request_sha256,
    validate_generation_response,
    validate_manifest_hashes,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer


LABELS = [f"Facet {number}" for number in range(1, 11)]


def test_judge_client_uses_compact_windows_safe_checkpoint_names(tmp_path: Path):
    client = _make_judge_client(
        output_dir=tmp_path / "deepseek_v4_flash",
        config={"api_base": "http://localhost:4000/v1", "api_key": "none"},
        purpose="nugget_evaluation",
    )

    assert client.checkpoint_dir.name == "nugget_cache"
    assert client.checkpoint_dir.is_dir()


def test_config_path_is_normalized_before_manifest_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_config_path(Path("config.yaml")) == config_path.resolve()


def _source_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    sections: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for number, label in enumerate(LABELS, 1):
        claim_id = f"A{number:03d}"
        passages = [
            {
                "passage_id": f"P{number:03d}-{passage_number}",
                "document_id": f"shard_{number:05d}_{passage_number}",
                "text": (
                    f"Supported historical fact number {number} contains detail "
                    f"{passage_number} and shared context."
                ),
            }
            for passage_number in range(1, 5)
        ]
        sections.append(
            {
                "sub_narrative": label,
                "claims": [
                    {
                        "claim_id": claim_id,
                        "text": f"Supported historical fact number {number} contains shared context.",
                        "supporting_passages": passages,
                        "document_ids": [row["document_id"] for row in passages],
                    },
                    {
                        "claim_id": f"U{number:03d}",
                        "text": f"Unsupported claim {number}.",
                        "supporting_passages": passages,
                        "document_ids": [passages[0]["document_id"]],
                    },
                ],
            }
        )
        audits.extend(
            [
                {"claim_id": claim_id, "status": "supported"},
                {"claim_id": f"U{number:03d}", "status": "unsupported"},
            ]
        )
    return (
        {
            "experiment_id": "source-experiment",
            "topic_id": "213",
            "narrative": "A shared narrative.",
            "input_accounting": {},
            "source_metadata": {},
            "sections": sections,
        },
        audits,
    )


def _frozen_fixture() -> dict[str, object]:
    source, audits = _source_fixture()
    return build_frozen_evidence(
        source_generation=source,
        support_rows=audits,
        sentence_quotas=[1] * 10,
        facet_word_targets=[10] * 10,
        expected_claim_count=10,
        expected_source_word_count=80,
        minimum_words=70,
        maximum_words=120,
        tokenizer=SpacySentenceTokenizer(),
    )


def _valid_response() -> dict[str, object]:
    return {
        "sentences": [
            {
                "claim_id": f"A{number:03d}",
                "text": (
                    f"Supported historical fact number {number} contains shared context "
                    "and remains fully grounded."
                ),
            }
            for number in range(1, 11)
        ]
    }


def test_frozen_evidence_excludes_rejected_claims_and_caps_citations():
    frozen = _frozen_fixture()

    assert frozen["supported_source_claim_count"] == 10
    assert frozen["organizer_nuggets_available"] is False
    assert [facet["sentence_quota"] for facet in frozen["facets"]] == [1] * 10
    serialized = json.dumps(frozen)
    assert "Unsupported claim" not in serialized
    for facet in frozen["facets"]:
        for claim in facet["source_claims"]:
            assert len(claim["selected_citation_passages"]) == 1


def test_generation_payload_hides_passages_and_preserves_all_claim_slots():
    payload = build_generation_payload(_frozen_fixture())
    serialized = json.dumps(payload)

    assert "supporting_passages" not in serialized
    assert "selected_citation_passages" not in serialized
    assert payload["exact_total_sentence_count"] == 10
    assert [row["claim_id"] for facet in payload["facets"] for row in facet["claims"]] == [
        f"A{number:03d}" for number in range(1, 11)
    ]


def test_generation_validator_enforces_claim_order_sentence_shape_and_word_band():
    frozen = _frozen_fixture()
    normalized = validate_generation_response(
        _valid_response(), frozen=frozen, tokenizer=SpacySentenceTokenizer()
    )

    assert len(normalized) == 10
    assert normalized[0]["claim_id"] == "A001"

    missing = _valid_response()
    missing["sentences"] = missing["sentences"][:-1]
    with pytest.raises(ValueError, match="exactly 10"):
        validate_generation_response(
            missing, frozen=frozen, tokenizer=SpacySentenceTokenizer()
        )

    reordered = _valid_response()
    reordered["sentences"][0], reordered["sentences"][1] = (
        reordered["sentences"][1],
        reordered["sentences"][0],
    )
    with pytest.raises(ValueError, match="claim order"):
        validate_generation_response(
            reordered, frozen=frozen, tokenizer=SpacySentenceTokenizer()
        )


def test_semantic_request_hash_excludes_generator_identity():
    frozen = _frozen_fixture()
    payload = build_generation_payload(frozen)

    left = semantic_request_sha256(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=3500,
        temperature=0.0,
    )
    right = semantic_request_sha256(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=3500,
        temperature=0.0,
    )

    assert left == right


def test_model_transport_overrides_do_not_change_semantic_hash(tmp_path: Path, monkeypatch):
    captured: list[dict[str, object]] = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "id": "generation",
                "model": "response-model",
                "provider": "provider",
                "choices": [{"message": {"content": '{"sentences": []}'}}],
                "usage": {"total_tokens": 1},
            }

    def fake_post(*args, **kwargs):
        captured.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(
        "trec_rag.topic213_controlled_generator_benchmark.requests.post", fake_post
    )
    payload = {"same": "payload"}
    clients = [
        SingleCandidateJsonClient(
            api_base="http://local/v1",
            model="qwen-local",
            api_key="none",
            checkpoint_path=tmp_path / "qwen.json",
            call_log_path=tmp_path / "qwen.jsonl",
            timeout_seconds=30,
            transport_max_attempts=3,
        ),
        SingleCandidateJsonClient(
            api_base="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-v4-flash",
            api_key="secret",
            checkpoint_path=tmp_path / "deepseek.json",
            call_log_path=tmp_path / "deepseek.jsonl",
            timeout_seconds=30,
            transport_max_attempts=3,
            request_overrides={"reasoning": {"enabled": False}},
        ),
    ]

    receipts = [
        client.complete_once(
            system_prompt="Same prompt.",
            payload=payload,
            max_tokens=100,
            temperature=0.0,
        ).receipt
        for client in clients
    ]

    assert receipts[0]["semantic_request_sha256"] == receipts[1]["semantic_request_sha256"]
    assert "reasoning" not in captured[0]
    assert captured[1]["reasoning"] == {"enabled": False}


def test_single_candidate_client_does_not_retry_invalid_semantic_output(
    tmp_path: Path, monkeypatch
):
    calls = 0

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "not json"}}]}

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(
        "trec_rag.topic213_controlled_generator_benchmark.requests.post", fake_post
    )
    client = SingleCandidateJsonClient(
        api_base="http://local/v1",
        model="model",
        api_key="none",
        checkpoint_path=tmp_path / "checkpoint.json",
        call_log_path=tmp_path / "calls.jsonl",
        timeout_seconds=30,
        transport_max_attempts=3,
    )

    with pytest.raises(ValueError, match="no semantic retry"):
        client.complete_once(
            system_prompt="Return JSON.",
            payload={"value": 1},
            max_tokens=10,
            temperature=0.0,
        )

    assert calls == 1


def test_candidate_uses_frozen_citations_regardless_of_generated_wording():
    frozen = _frozen_fixture()
    first = validate_generation_response(
        _valid_response(), frozen=frozen, tokenizer=SpacySentenceTokenizer()
    )
    second = [dict(row) for row in first]
    second[0]["text"] = (
        "Shared context remains grounded in supported historical fact number one and its evidence."
    )
    source, _ = _source_fixture()

    candidate_a = build_candidate_generation(
        source_generation=source,
        frozen=frozen,
        normalized_sentences=first,
        model_key="model-a",
        model_identity="model/a",
        receipt={"semantic_request_sha256": "same"},
        experiment_id="experiment",
        run_id="run-a",
    )
    candidate_b = build_candidate_generation(
        source_generation=source,
        frozen=frozen,
        normalized_sentences=second,
        model_key="model-b",
        model_identity="model/b",
        receipt={"semantic_request_sha256": "same"},
        experiment_id="experiment",
        run_id="run-b",
    )

    assert candidate_a["sections"][0]["claims"][0]["document_ids"] == (
        candidate_b["sections"][0]["claims"][0]["document_ids"]
    )


def test_support_filter_removes_rejected_sentence_and_prunes_ledger():
    frozen = _frozen_fixture()
    source, _ = _source_fixture()
    normalized = validate_generation_response(
        _valid_response(), frozen=frozen, tokenizer=SpacySentenceTokenizer()
    )
    candidate = build_candidate_generation(
        source_generation=source,
        frozen=frozen,
        normalized_sentences=normalized,
        model_key="model-a",
        model_identity="model/a",
        receipt={"semantic_request_sha256": "same"},
        experiment_id="experiment",
        run_id="run-a",
    )
    audits = [
        {
            "claim_id": f"S{number:03d}",
            "status": "unsupported" if number == 1 else "supported",
            "document_ids": [f"shard_{number:05d}_1"],
            "claim_text": normalized[number - 1]["text"],
        }
        for number in range(1, 11)
    ]

    final, kept, excluded = filter_supported_generation(candidate, audits)

    assert len(kept) == 9
    assert len(excluded) == 1
    assert len(final["evidence_ledger"]) == 9


def test_winner_order_is_strict_then_vital_then_partial_then_support_then_cost():
    rows = [
        {
            "model_key": "partial-leader",
            "strict_coverage": 0.40,
            "vital_strict_coverage": 0.60,
            "partial_credit_coverage": 0.70,
            "unsupported_submitted": 0,
            "cost": 0.0,
        },
        {
            "model_key": "strict-leader",
            "strict_coverage": 0.42,
            "vital_strict_coverage": 0.50,
            "partial_credit_coverage": 0.50,
            "unsupported_submitted": 0,
            "cost": 1.0,
        },
    ]

    assert choose_winner(rows)["model_key"] == "strict-leader"


def test_manifest_hash_validation_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("answer", encoding="utf-8")
    import hashlib

    manifest = {
        "artifacts": [
            {"path": "artifact.json", "sha256": hashlib.sha256(b"answer").hexdigest()}
        ]
    }
    assert validate_manifest_hashes(manifest, artifact_root=tmp_path) == 1
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest_hashes(manifest, artifact_root=tmp_path)

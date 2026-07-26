from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from trec_rag.topic213_deepseek_v4_flash_oneshot import (
    EXACT_MODEL,
    OpenRouterOneShotJsonClient,
    filter_supported_sentences,
    load_supported_claim_input,
    validate_one_shot_response,
    validate_sentence_level_submission,
    verify_manifest_hashes,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer


def _source():
    return {
        "narrative": "A narrative",
        "sections": [
            {
                "sub_narrative": "Facet A",
                "claims": [
                    {
                        "claim_id": "A1",
                        "text": "Fact one.",
                        "supporting_passages": [
                            {"document_id": "shard_001_1", "text": "Fact one."}
                        ],
                    },
                    {"claim_id": "A2", "text": "Rejected fact."},
                ],
            },
            {
                "sub_narrative": "Facet B",
                "claims": [
                    {
                        "claim_id": "B1",
                        "text": "Fact two.",
                        "supporting_passages": [
                            {"document_id": "shard_002_2", "text": "Fact two."}
                        ],
                    }
                ],
            },
        ],
    }


def _supported_input():
    return load_supported_claim_input(
        _source(),
        [
            {"claim_id": "A1", "status": "supported"},
            {"claim_id": "A2", "status": "unsupported"},
            {"claim_id": "B1", "status": "supported"},
        ],
        expected_count=2,
    )


def _response():
    return {
        "sections": [
            {
                "sub_narrative": "Facet A",
                "answer": [{"text": "Fact one.", "source_claim_ids": ["A1"]}],
            },
            {
                "sub_narrative": "Facet B",
                "answer": [{"text": "Fact two.", "source_claim_ids": ["B1"]}],
            },
        ]
    }


def test_supported_input_excludes_every_non_supported_claim():
    payload, source_by_id, labels = _supported_input()
    assert labels == ["Facet A", "Facet B"]
    assert list(source_by_id) == ["A1", "B1"]
    assert [row["claim_id"] for section in payload["sections"] for row in section["source_claims"]] == [
        "A1",
        "B1",
    ]


def test_one_shot_validator_enforces_spacy_sentence_and_same_facet_sources():
    _, source_by_id, labels = _supported_input()
    tokenizer = SpacySentenceTokenizer()
    assert len(
        validate_one_shot_response(
            _response(),
            labels=labels,
            source_by_id=source_by_id,
            maximum_sentences_per_section=3,
            target_maximum_words=20,
            tokenizer=tokenizer,
        )
    ) == 2

    invalid = _response()
    invalid["sections"][0]["answer"][0]["text"] = "Fact one. Another fact."
    with pytest.raises(ValueError, match="exactly one SpaCy sentence"):
        validate_one_shot_response(
            invalid,
            labels=labels,
            source_by_id=source_by_id,
            maximum_sentences_per_section=3,
            target_maximum_words=20,
            tokenizer=tokenizer,
        )

    invalid = _response()
    invalid["sections"][0]["answer"][0]["source_claim_ids"] = ["B1"]
    with pytest.raises(ValueError, match="cross-section"):
        validate_one_shot_response(
            invalid,
            labels=labels,
            source_by_id=source_by_id,
            maximum_sentences_per_section=3,
            target_maximum_words=20,
            tokenizer=tokenizer,
        )


class _Response:
    status_code = 200

    def __init__(self, content: str):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "generation-id",
            "model": EXACT_MODEL,
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _client(tmp_path: Path, attempts: int = 3):
    return OpenRouterOneShotJsonClient(
        api_base="https://openrouter.ai/api/v1",
        model=EXACT_MODEL,
        api_key="test-key",
        checkpoint_path=tmp_path / "checkpoint.json",
        timeout_seconds=10,
        max_transport_attempts=attempts,
    )


def test_openrouter_one_shot_uses_exact_model_and_one_request(monkeypatch, tmp_path):
    bodies = []

    def post(*args, **kwargs):
        bodies.append(kwargs["json"])
        return _Response('{"sections": []}')

    monkeypatch.setattr(requests, "post", post)
    completion = _client(tmp_path).complete_once(
        system_prompt="system", payload={"sections": []}, max_tokens=100, temperature=0
    )
    assert len(bodies) == 1
    assert bodies[0]["model"] == EXACT_MODEL
    assert bodies[0]["temperature"] == 0
    assert bodies[0]["reasoning"] == {"enabled": False}
    assert completion.receipt["semantic_generation_request_count"] == 1
    assert completion.receipt["transport_attempt_count"] == 1


def test_openrouter_retries_only_an_identical_transport_request(monkeypatch, tmp_path):
    bodies = []

    def post(*args, **kwargs):
        bodies.append(json.dumps(kwargs["json"], sort_keys=True))
        if len(bodies) == 1:
            raise requests.ConnectionError("temporary")
        return _Response('{"sections": []}')

    monkeypatch.setattr(requests, "post", post)
    completion = _client(tmp_path).complete_once(
        system_prompt="system", payload={"sections": []}, max_tokens=100, temperature=0
    )
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert completion.receipt["transport_attempt_count"] == 2


def test_openrouter_does_not_retry_malformed_generation(monkeypatch, tmp_path):
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response("not json")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError, match="JSON object"):
        _client(tmp_path).complete_once(
            system_prompt="system", payload={}, max_tokens=100, temperature=0
        )
    assert calls == 1


def test_filter_excludes_non_supported_sentence_without_rewrite():
    candidate = {
        "sections": [
            {
                "claims": [
                    {"claim_id": "S1", "evidence_claim_ids": ["A1"], "text": "Keep."},
                    {"claim_id": "S2", "evidence_claim_ids": ["B1"], "text": "Drop."},
                ]
            }
        ],
        "evidence_ledger": [{"claim_id": "A1"}, {"claim_id": "B1"}],
    }
    kept, excluded = filter_supported_sentences(
        candidate,
        [
            {"claim_id": "S1", "status": "supported"},
            {"claim_id": "S2", "status": "partially_supported"},
        ],
    )
    assert [row["claim_id"] for row in kept] == ["S1"]
    assert [row["claim_id"] for row in excluded] == ["S2"]
    assert [row["text"] for row in candidate["sections"][0]["claims"]] == ["Keep."]
    assert [row["claim_id"] for row in candidate["evidence_ledger"]] == ["A1"]


def test_submission_requires_one_sentence_and_first_use_reference_order():
    entry = {
        "metadata": {
            "team_id": "team",
            "narrative_id": "213",
            "narrative": "Narrative",
            "run_id": "run",
            "run_desc": "desc",
        },
        "references": ["shard_001_1"],
        "answer": [{"text": "One sentence.", "citations": ["shard_001_1"]}],
    }
    assert validate_sentence_level_submission(entry, tokenizer=SpacySentenceTokenizer()) == 2
    entry["answer"][0]["text"] = "One sentence. Two sentences."
    with pytest.raises(ValueError, match="SpaCy"):
        validate_sentence_level_submission(entry, tokenizer=SpacySentenceTokenizer())


def test_manifest_hash_verifier_detects_tampering(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original", encoding="utf-8")
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.yaml").write_text(
        f"artifacts:\n  artifact.txt:\n    bytes: 8\n    sha256: {digest}\n",
        encoding="utf-8",
    )
    assert verify_manifest_hashes(tmp_path) == 1
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        verify_manifest_hashes(tmp_path)

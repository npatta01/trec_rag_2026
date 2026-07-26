from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.topic213_deepseek_facet_hierarchical import (
    build_candidate_generation,
    build_generation_input,
    extract_candidate_facts,
    load_supported_source_claims,
    SingleShotOpenRouterJsonClient,
    summarize_deepseek_calls,
    synthesize_from_extractions,
    validate_manifest_hashes,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer


LABELS = [f"Facet {number}" for number in range(1, 11)]


class RecordingClient:
    model = "deepseek/deepseek-v4-flash"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _source_generation():
    sections = []
    audit = []
    for number, label in enumerate(LABELS, 1):
        claim_id = f"A{number:03d}"
        unsupported_id = f"U{number:03d}"
        passage = {
            "passage_id": f"P{number:03d}",
            "document_id": f"shard_{number:05d}_1",
            "text": f"Evidence for supported fact {number}.",
        }
        sections.append(
            {
                "sub_narrative": label,
                "claims": [
                    {
                        "claim_id": claim_id,
                        "text": f"Supported fact {number}.",
                        "supporting_passages": [passage],
                        "document_ids": [passage["document_id"]],
                        "evidence_claim_ids": [f"E{number:03d}"],
                    },
                    {
                        "claim_id": unsupported_id,
                        "text": f"Unsupported fact {number}.",
                        "supporting_passages": [passage],
                        "document_ids": [passage["document_id"]],
                    },
                ],
            }
        )
        audit.extend(
            [
                {"claim_id": claim_id, "status": "supported"},
                {"claim_id": unsupported_id, "status": "unsupported"},
            ]
        )
    generation = {
        "experiment_id": "source",
        "topic_id": "213",
        "narrative": "Narrative",
        "input_accounting": {},
        "source_metadata": {},
        "sections": sections,
    }
    return generation, audit


def test_supported_filter_excludes_every_unsupported_claim():
    generation, audit = _source_generation()
    labels, claims_by_label, source_by_id = load_supported_source_claims(
        generation, audit, expected_claim_count=10
    )
    generation_input = build_generation_input(labels, claims_by_label)

    assert labels == LABELS
    assert set(source_by_id) == {f"A{number:03d}" for number in range(1, 11)}
    serialized = json.dumps(generation_input)
    assert "Unsupported fact" not in serialized
    assert generation_input["organizer_nuggets_available"] is False


def test_extraction_makes_exactly_one_call_per_facet():
    generation, audit = _source_generation()
    labels, claims_by_label, _ = load_supported_source_claims(
        generation, audit, expected_claim_count=10
    )
    responses = [
        {
            "candidate_facts": [
                {"text": f"Supported fact {number}.", "source_claim_ids": [f"A{number:03d}"]}
            ]
        }
        for number in range(1, 11)
    ]
    client = RecordingClient(responses)
    extracted = extract_candidate_facts(
        client=client,
        labels=labels,
        claims_by_label=claims_by_label,
        tokenizer=SpacySentenceTokenizer(),
        max_tokens=500,
        temperature=0.0,
        maximum_facts_per_facet=3,
    )

    assert len(client.calls) == 10
    assert [row["candidate_facts"][0]["fact_id"] for row in extracted] == [
        f"F{number:02d}-001" for number in range(1, 11)
    ]
    assert all(call["temperature"] == 0.0 for call in client.calls)


def test_synthesis_is_one_call_over_stage_a_outputs_only():
    extracted = [
        {
            "facet_number": number,
            "sub_narrative": label,
            "input_source_claim_ids": [f"A{number:03d}"],
            "candidate_facts": [
                {
                    "fact_id": f"F{number:02d}-001",
                    "text": f"Supported fact {number}.",
                    "source_claim_ids": [f"A{number:03d}"],
                }
            ],
        }
        for number, label in enumerate(LABELS, 1)
    ]
    response = {
        "sections": [
            {
                "sub_narrative": label,
                "answer": [
                    {
                        "text": f"Supported fact {number}.",
                        "extracted_fact_ids": [f"F{number:02d}-001"],
                    }
                ],
            }
            for number, label in enumerate(LABELS, 1)
        ]
    }
    client = RecordingClient([response])
    result = synthesize_from_extractions(
        client=client,
        extracted_sections=extracted,
        tokenizer=SpacySentenceTokenizer(),
        max_tokens=1000,
        temperature=0.0,
        maximum_sentences_per_section=2,
        target_maximum_words=500,
    )

    assert len(client.calls) == 1
    payload = client.calls[0]["payload"]
    assert "stage_a_outputs" in payload
    assert "source_claims" not in json.dumps(payload)
    assert "supporting_passages" not in json.dumps(payload)
    assert len(result) == 10


def test_candidate_preserves_fact_claim_passage_lineage_and_caps_citations():
    generation, audit = _source_generation()
    _, _, source_by_id = load_supported_source_claims(
        generation, audit, expected_claim_count=10
    )
    extracted = [
        {
            "sub_narrative": LABELS[0],
            "candidate_facts": [
                {
                    "fact_id": "F01-001",
                    "text": "Supported fact 1.",
                    "source_claim_ids": ["A001"],
                }
            ],
        }
    ]
    synthesized = [
        {
            "sub_narrative": LABELS[0],
            "answer": [
                {"text": "Supported fact 1.", "extracted_fact_ids": ["F01-001"]}
            ],
        }
    ]
    candidate = build_candidate_generation(
        source_generation=generation,
        extracted_sections=extracted,
        synthesized_sections=synthesized,
        source_by_id=source_by_id,
        experiment={"id": "exp", "run_id": "run"},
        generation_config={"model": "deepseek/deepseek-v4-flash", "temperature": 0.0},
    )
    claim = candidate["sections"][0]["claims"][0]

    assert claim["extracted_fact_ids"] == ["F01-001"]
    assert claim["evidence_claim_ids"] == ["A001"]
    assert claim["document_ids"] == ["shard_00001_1"]
    assert len(claim["document_ids"]) <= 3


def test_call_summary_requires_ten_extractions_and_one_synthesis():
    class FakeClient:
        model = "deepseek/deepseek-v4-flash"
        network_request_count = 11
        records = [
            {"stage": f"deepseek_extract_facet_{number:02d}", "usage": {"total_tokens": 10}}
            for number in range(1, 11)
        ] + [{"stage": "deepseek_synthesize_all_facets", "usage": {"total_tokens": 20}}]

    summary = summarize_deepseek_calls(FakeClient())
    assert summary["logical_calls"] == 11
    assert summary["extraction_calls"] == 10
    assert summary["synthesis_calls"] == 1
    assert summary["usage_totals"]["total_tokens"] == 120


def test_manifest_hash_validation_detects_tampering(tmp_path: Path):
    repo_file = tmp_path / "source.json"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifact = artifact_dir / "answer.json"
    repo_file.write_text("source", encoding="utf-8")
    artifact.write_text("answer", encoding="utf-8")

    import hashlib

    manifest = {
        "source_inputs": [
            {"path": "source.json", "sha256": hashlib.sha256(b"source").hexdigest()}
        ],
        "artifacts": [
            {"path": "answer.json", "sha256": hashlib.sha256(b"answer").hexdigest()}
        ],
    }
    assert validate_manifest_hashes(
        manifest, artifact_root=artifact_dir, repo_root=tmp_path
    ) == 2
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest_hashes(manifest, artifact_root=artifact_dir, repo_root=tmp_path)


def test_openrouter_checkpoint_uses_windows_safe_hash_filename(tmp_path: Path, monkeypatch):
    captured = {}

    class Response:
        headers = {"X-Generation-Id": "gen-test"}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "gen-test",
                "model": "deepseek/deepseek-v4-flash",
                "provider": "Baidu",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"ok": true}'}}
                ],
                "usage": {"total_tokens": 3},
            }

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(
        "trec_rag.topic213_deepseek_facet_hierarchical.requests.post", fake_post
    )
    checkpoint_dir = tmp_path / "a_very_long_experiment_directory_name" / "checkpoints"
    client = SingleShotOpenRouterJsonClient(
        api_base="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
        api_key="secret",
        checkpoint_dir=checkpoint_dir,
        call_log_path=tmp_path / "calls.jsonl",
        timeout_seconds=30,
        provider_order=["Baidu"],
    )
    assert client.complete_json(
        stage="deepseek_synthesize_all_facets",
        system_prompt="Return JSON.",
        payload={"value": 1},
        max_tokens=20,
        temperature=0.0,
    ) == {"ok": True}

    checkpoint_names = [path.name for path in checkpoint_dir.glob("*.json")]
    assert len(checkpoint_names) == 1
    assert len(checkpoint_names[0]) == 69
    assert captured["reasoning"] == {"enabled": False}
    assert captured["provider"]["order"] == ["Baidu"]

from __future__ import annotations

import json

import pytest
import yaml

from trec_rag.topic213_gpt_sol_oneshot import (
    StrictOneShotJsonClient,
    build_official_entry,
    filter_supported_sentences,
    load_supported_source_claims,
    validate_oneshot_answer,
    validate_organizer_submission,
    verify_manifest_hashes,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer


LABEL_A = '"What triggered the Korean War?"'
LABEL_B = '"How did the Korean War conclude?"'


def _source():
    generation = {
        "experiment_id": "source",
        "topic_id": "213",
        "narrative": "A narrative",
        "input_accounting": {},
        "source_metadata": {},
        "sections": [
            {
                "sub_narrative": LABEL_A,
                "claims": [
                    {
                        "claim_id": "A001",
                        "text": "North Korea invaded South Korea.",
                        "supporting_passages": [
                            {
                                "passage_id": "P1",
                                "document_id": "shard_00001_1",
                                "text": "North Korea invaded South Korea in 1950.",
                            }
                        ],
                    },
                    {
                        "claim_id": "A999",
                        "text": "An unsupported source claim.",
                        "supporting_passages": [
                            {
                                "passage_id": "PX",
                                "document_id": "shard_00999_9",
                                "text": "Unrelated text.",
                            }
                        ],
                    },
                ],
            },
            {
                "sub_narrative": LABEL_B,
                "claims": [
                    {
                        "claim_id": "A002",
                        "text": "An armistice stopped the fighting.",
                        "supporting_passages": [
                            {
                                "passage_id": "P2",
                                "document_id": "shard_00002_2",
                                "text": "The armistice stopped active fighting.",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    audits = [
        {"claim_id": "A001", "status": "supported"},
        {"claim_id": "A999", "status": "unsupported"},
        {"claim_id": "A002", "status": "supported"},
    ]
    return generation, audits


@pytest.fixture(scope="module")
def tokenizer():
    return SpacySentenceTokenizer()


def test_supported_ledger_excludes_non_supported_claims():
    generation, audits = _source()
    labels, source_by_id = load_supported_source_claims(generation, audits)
    assert labels == [LABEL_A, LABEL_B]
    assert list(source_by_id) == ["A001", "A002"]


def test_validate_one_shot_enforces_sections_sentences_and_evidence(tokenizer):
    generation, audits = _source()
    labels, source_by_id = load_supported_source_claims(generation, audits)
    answer = validate_oneshot_answer(
        {
            "answer": [
                {
                    "sub_narrative": LABEL_A,
                    "text": "North Korea invaded South Korea.",
                    "citations": ["shard_00001_1"],
                    "source_claim_ids": ["A001"],
                },
                {
                    "sub_narrative": LABEL_B,
                    "text": "An armistice stopped the fighting.",
                    "citations": ["shard_00002_2"],
                    "source_claim_ids": ["A002"],
                },
            ]
        },
        labels=labels,
        source_by_id=source_by_id,
        maximum_sentences_per_section=2,
        target_maximum_words=100,
        tokenizer=tokenizer,
    )
    assert len(answer) == 2


def test_validate_one_shot_repairs_inherited_apostrophe_mojibake(tokenizer):
    generation, audits = _source()
    labels, source_by_id = load_supported_source_claims(generation, audits)
    answer = validate_oneshot_answer(
        {
            "answer": [
                {
                    "sub_narrative": LABEL_A,
                    "text": "North Koreaâ€™s army invaded South Korea.",
                    "citations": ["shard_00001_1"],
                    "source_claim_ids": ["A001"],
                },
                {
                    "sub_narrative": LABEL_B,
                    "text": "An armistice stopped the fighting.",
                    "citations": ["shard_00002_2"],
                    "source_claim_ids": ["A002"],
                },
            ]
        },
        labels=labels,
        source_by_id=source_by_id,
        maximum_sentences_per_section=2,
        target_maximum_words=100,
        tokenizer=tokenizer,
    )
    assert answer[0]["text"] == "North Korea's army invaded South Korea."


@pytest.mark.parametrize(
    "change,error",
    [
        ({"text": "First sentence. Second sentence."}, "exactly one SpaCy sentence"),
        ({"citations": ["shard_00002_2"]}, "outside its source claims"),
        ({"source_claim_ids": ["A002"]}, "cross-section source claim"),
    ],
)
def test_validate_one_shot_rejects_malformed_semantics(tokenizer, change, error):
    generation, audits = _source()
    labels, source_by_id = load_supported_source_claims(generation, audits)
    first = {
        "sub_narrative": LABEL_A,
        "text": "North Korea invaded South Korea.",
        "citations": ["shard_00001_1"],
        "source_claim_ids": ["A001"],
    }
    first.update(change)
    with pytest.raises(ValueError, match=error):
        validate_oneshot_answer(
            {
                "answer": [
                    first,
                    {
                        "sub_narrative": LABEL_B,
                        "text": "An armistice stopped the fighting.",
                        "citations": ["shard_00002_2"],
                        "source_claim_ids": ["A002"],
                    },
                ]
            },
            labels=labels,
            source_by_id=source_by_id,
            maximum_sentences_per_section=2,
            target_maximum_words=100,
            tokenizer=tokenizer,
        )


def test_filter_excludes_rejected_sentence_without_repair(tokenizer):
    candidate = {
        "sections": [
            {
                "sub_narrative": LABEL_A,
                "claims": [
                    {
                        "claim_id": "S001",
                        "text": "North Korea invaded South Korea.",
                        "document_ids": ["shard_00001_1"],
                        "evidence_claim_ids": ["A001"],
                    },
                    {
                        "claim_id": "S002",
                        "text": "A rejected sentence.",
                        "document_ids": ["shard_00001_1"],
                        "evidence_claim_ids": ["A001"],
                    },
                ],
            }
        ],
        "evidence_ledger": [{"claim_id": "A001"}],
    }
    final, kept, excluded = filter_supported_sentences(
        candidate,
        [
            {"claim_id": "S001", "status": "supported"},
            {"claim_id": "S002", "status": "unsupported"},
        ],
    )
    assert [row["claim_id"] for row in final["sections"][0]["claims"]] == ["S001"]
    assert [row["claim_id"] for row in kept] == ["S001"]
    assert [row["claim_id"] for row in excluded] == ["S002"]
    assert final["support_filter"]["rewrite_or_repair_attempted"] is False


def test_official_entry_has_unique_references_and_spacy_sentences(tokenizer):
    generation = {
        "topic_id": "213",
        "narrative": "A narrative",
        "run_id": "run",
        "model_identity": "openai/gpt-5.6-sol",
        "context_policy": {"source_experiment_id": "source"},
        "support_filter": {"judge_model": "qwen-local"},
        "sections": [
            {
                "claims": [
                    {
                        "text": "North Korea invaded South Korea.",
                        "document_ids": ["shard_00001_1"],
                    },
                    {
                        "text": "The war ended in an armistice.",
                        "document_ids": ["shard_00001_1", "shard_00002_2"],
                    },
                ]
            }
        ],
    }
    entry = build_official_entry(generation=generation, team_id="team", run_desc="desc")
    assert entry["references"] == ["shard_00001_1", "shard_00002_2"]
    assert validate_organizer_submission(entry, maximum_words=1024, tokenizer=tokenizer) == 11


def test_strict_client_does_not_retry_malformed_semantic_response(monkeypatch, tmp_path):
    class Response:
        status_code = 200
        text = '{"choices": [{"message": {"content": "not-json"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return json.loads(self.text)

    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr("trec_rag.topic213_gpt_sol_oneshot.requests.post", fake_post)
    client = StrictOneShotJsonClient(
        api_base="https://example.test/v1",
        model="openai/gpt-5.6-sol",
        api_key="secret",
        checkpoint_dir=tmp_path / "checkpoints",
        call_log_path=tmp_path / "calls.jsonl",
        timeout_seconds=10,
        transport_max_attempts=3,
    )
    with pytest.raises(ValueError, match="no repair call"):
        client.complete_json_once(
            stage="once",
            system_prompt="system",
            payload={"input": "value"},
            response_schema={"type": "object"},
            max_tokens=10,
            temperature=0.0,
        )
    assert len(calls) == 1


def test_manifest_verifier_checks_every_binding(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("fixed", encoding="utf-8")
    import hashlib

    manifest = {
        "files": {
            "artifact.txt": {
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        }
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert verify_manifest_hashes(path, repo_root=tmp_path) == 1

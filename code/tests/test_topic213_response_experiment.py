from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.topic213_response_experiment import (
    PassageCorpus,
    PassageUnit,
    batch_passages,
    compute_evaluation_metrics,
    evaluate_nuggets,
    generate_response,
    load_nuggets,
    load_passage_corpus,
    render_generated_response,
    _normalize_sub_narrative,
    _map_batch,
    _batch_claims,
    validate_release_accounting,
)


LABEL_A = '"What triggered the Korean War?"'
LABEL_B = '"How did the Korean War conclude?"'


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_and_batch_passages_uses_every_document_and_passage_once(tmp_path):
    path = tmp_path / "passages.jsonl"
    _write_jsonl(
        path,
        [
            {
                "document_id": "doc-a",
                "evidence": [
                    {"sub_narrative": LABEL_A, "passages": ["Alpha.", "Beta."]},
                    {"sub_narrative": LABEL_B, "passages": ["Gamma."]},
                ],
            },
            {
                "document_id": "doc-b",
                "evidence": [
                    {"sub_narrative": LABEL_A, "passages": ["Delta."]},
                ],
            },
        ],
    )

    corpus = load_passage_corpus(path)
    batches = batch_passages(corpus.passages, max_input_chars=11, max_passages=2)

    assert corpus.document_ids == ("doc-a", "doc-b")
    assert len(corpus.passages) == 4
    assert [unit.passage_id for batch in batches for unit in batch] == [
        "P000001",
        "P000002",
        "P000003",
        "P000004",
    ]
    assert {unit.document_id for batch in batches for unit in batch} == {"doc-a", "doc-b"}


def test_load_passage_corpus_rejects_duplicate_document_ids(tmp_path):
    path = tmp_path / "passages.jsonl"
    row = {
        "document_id": "doc-a",
        "evidence": [{"sub_narrative": LABEL_A, "passages": ["Alpha."]}],
    }
    _write_jsonl(path, [row, row])

    with pytest.raises(ValueError, match="duplicate document_id"):
        load_passage_corpus(path)


def test_validate_release_accounting_preserves_two_failed_calls():
    corpus = PassageCorpus(
        document_ids=("doc-a", "doc-b"),
        sub_narratives=(LABEL_A,),
        passages=(
            PassageUnit("P000001", "doc-a", LABEL_A, "Alpha."),
            PassageUnit("P000002", "doc-b", LABEL_A, "Beta."),
        ),
        evidence_assignment_count=2,
    )
    manifest = {
        "population_counts": {
            "passage_documents_eligible": 4,
            "passage_model_calls_attempted": 4,
            "passage_valid_responses": 2,
            "passage_failed_responses": 2,
            "passage_output_documents": 2,
            "passage_evidence_assignments": 2,
            "verbatim_passages": 2,
        },
        "passage_extraction": {
            "failed_document_ids": ["doc-failed-a", "doc-failed-b"],
        },
    }

    accounting = validate_release_accounting(corpus, manifest)

    assert accounting["documents_processed"] == 2
    assert accounting["documents_unavailable"] == 2
    assert accounting["failed_document_ids"] == ["doc-failed-a", "doc-failed-b"]


def test_load_nuggets_requires_exact_topic_and_expected_count(tmp_path):
    path = tmp_path / "nuggets.jsonl"
    nuggets = [
        {
            "text": f"Fact {index}",
            "mapped_sub_narrative": LABEL_A,
            "importance": "vital" if index < 27 else "okay",
            "source": "post-edit",
        }
        for index in range(50)
    ]
    _write_jsonl(path, [{"qid": "213", "nuggets": nuggets}])

    loaded = load_nuggets(path, topic_id="213", expected_count=50)

    assert len(loaded) == 50
    assert loaded[0]["nugget_id"] == "213-N001"
    assert sum(row["importance"] == "vital" for row in loaded) == 27


def test_render_generated_response_cites_every_claim():
    generation = {
        "topic_id": "213",
        "narrative": "Korean War narrative",
        "sections": [
            {
                "sub_narrative": LABEL_A,
                "claims": [
                    {
                        "claim_id": "A001",
                        "text": "North Korea invaded South Korea.",
                        "document_ids": ["doc-a", "doc-b"],
                    }
                ],
            }
        ],
    }

    markdown = render_generated_response(generation)

    assert "## What triggered the Korean War?" in markdown
    assert "North Korea invaded South Korea. [doc-a; doc-b]" in markdown


def test_normalize_sub_narrative_restores_exact_released_label():
    assert _normalize_sub_narrative("What triggered the Korean War?", [LABEL_A, LABEL_B]) == LABEL_A
    typo_label = "New: What motivated China involvement in te Korean War?"
    assert _normalize_sub_narrative(
        "New: What motivated China involvement in the Korean War?", [typo_label]
    ) == typo_label


def test_map_batch_keeps_only_citations_from_the_supplied_batch():
    class FakeClient:
        model = "fake"

        def complete_json(self, **_kwargs):
            return {
                "claims": [
                    {
                        "text": "Supported in this batch.",
                        "sub_narrative": LABEL_A,
                        "passage_ids": ["P000001", "P999999"],
                    },
                    {
                        "text": "No in-batch support.",
                        "sub_narrative": LABEL_A,
                        "passage_ids": ["P999999"],
                    },
                ]
            }

    claims = _map_batch(
        FakeClient(),
        batch=[PassageUnit("P000001", "doc-a", LABEL_A, "Evidence.")],
        narrative="Narrative",
        allowed_labels=[LABEL_A],
        max_claims=8,
        max_tokens=100,
        temperature=0.0,
        validation_attempts=1,
    )

    assert len(claims) == 1
    assert claims[0]["passage_ids"] == ["P000001"]
    assert claims[0]["discarded_out_of_batch_passage_ids"] == ["P999999"]


def test_reduction_batching_counts_only_the_payload_not_attached_passage_text():
    claims = [
        {
            "claim_id": f"M{index}",
            "text": "Short claim",
            "document_ids": [f"doc-{index}"],
            "supporting_passages": [{"text": "x" * 10_000}],
        }
        for index in range(3)
    ]

    batches = _batch_claims(claims, max_input_chars=100, max_claims=20)

    assert [len(batch) for batch in batches] == [3]


def test_compute_evaluation_metrics_separates_vital_and_okay():
    comparison = [
        {"importance": "vital", "status": "supported", "mapped_sub_narrative": LABEL_A},
        {
            "importance": "vital",
            "status": "partially_supported",
            "mapped_sub_narrative": LABEL_A,
        },
        {"importance": "okay", "status": "missing", "mapped_sub_narrative": LABEL_B},
        {"importance": "okay", "status": "contradicted", "mapped_sub_narrative": LABEL_B},
    ]
    support_audit = [
        {"status": "supported", "document_ids": ["doc-a"]},
        {"status": "unsupported", "document_ids": ["doc-b"]},
    ]

    metrics = compute_evaluation_metrics(
        comparison,
        support_audit,
        response_text="One two three four.",
    )

    assert metrics["nuggets"]["all"]["strict_coverage"] == pytest.approx(0.25)
    assert metrics["nuggets"]["vital"]["strict_coverage"] == pytest.approx(0.5)
    assert metrics["nuggets"]["vital"]["partial_credit_coverage"] == pytest.approx(0.75)
    assert metrics["answer_claims"]["citation_coverage"] == 1.0
    assert metrics["answer_claims"]["unsupported_claim_count"] == 1
    assert metrics["response"]["word_count"] == 4


def test_generation_consumes_full_corpus_before_nugget_evaluation():
    corpus = PassageCorpus(
        document_ids=("doc-a", "doc-b"),
        sub_narratives=(LABEL_A, LABEL_B),
        passages=(
            PassageUnit("P000001", "doc-a", LABEL_A, "Alpha."),
            PassageUnit("P000002", "doc-a", LABEL_B, "Beta."),
            PassageUnit("P000003", "doc-b", LABEL_A, "Gamma."),
            PassageUnit("P000004", "doc-b", LABEL_B, "Delta."),
        ),
        evidence_assignment_count=4,
    )

    class FakeClient:
        model = "fake-model"

        def __init__(self):
            self.calls = []

        def complete_json(self, *, stage, system_prompt, payload, max_tokens, temperature):
            self.calls.append((stage, json.dumps(payload, sort_keys=True)))
            if stage.startswith("map_evidence"):
                return {
                    "claims": [
                        {
                            "text": f"Claim from {row['passage_id']}",
                            "sub_narrative": row["sub_narrative"],
                            "passage_ids": [row["passage_id"]],
                        }
                        for row in payload["passages"]
                    ]
                }
            if stage.startswith("reduce_evidence"):
                return {
                    "claims": [
                        {
                            "text": f"Consolidated {payload['sub_narrative']}",
                            "source_claim_ids": [row["claim_id"] for row in payload["claims"]],
                        }
                    ]
                }
            if stage.startswith("generate_section"):
                return {
                    "claims": [
                        {
                            "text": f"Answer for {payload['sub_narrative']}",
                            "evidence_claim_ids": [
                                payload["evidence_claims"][0]["evidence_claim_id"]
                            ],
                        }
                    ]
                }
            if stage.startswith("audit_support"):
                return {
                    "assessments": [
                        {"claim_id": row["claim_id"], "status": "supported", "notes": ""}
                        for row in payload["claims"]
                    ]
                }
            if stage.startswith("evaluate_nuggets"):
                return {
                    "assessments": [
                        {
                            "nugget_id": row["nugget_id"],
                            "status": "supported",
                            "matched_claim_ids": [payload["answer_claims"][0]["claim_id"]],
                            "evidence_claim_ids": [
                                payload["release_evidence_claims"][0]["evidence_claim_id"]
                            ],
                            "notes": "",
                        }
                        for row in payload["nuggets"]
                    ]
                }
            raise AssertionError(stage)

    client = FakeClient()
    generation, support_audit = generate_response(
        topic_id="213",
        narrative="Korean War narrative",
        corpus=corpus,
        release_accounting={"documents_processed": 2, "passages_processed": 4},
        client=client,
        generation_config={
            "max_input_chars": 100,
            "max_passages_per_batch": 10,
            "map_max_claims": 10,
        },
        experiment_id="experiment",
        run_id="run",
        source_metadata={},
    )

    generation_prompts = [payload for stage, payload in client.calls if stage != "evaluate_nuggets"]
    assert all("organizer secret nugget" not in payload for payload in generation_prompts)
    assert {
        passage_id
        for claim in generation["evidence_ledger"]
        for passage_id in claim["passage_ids"]
    } == {"P000001", "P000002", "P000003", "P000004"}
    assert len(support_audit) == 2

    comparison = evaluate_nuggets(
        nuggets=[
            {
                "nugget_id": "213-N001",
                "text": "organizer secret nugget",
                "mapped_sub_narrative": LABEL_A,
                "importance": "vital",
                "source": "post-edit",
            }
        ],
        generation=generation,
        client=client,
        evaluation_config={},
    )

    assert len(comparison) == 1
    assert comparison[0]["status"] == "supported"
    assert "organizer secret nugget" in client.calls[-1][1]

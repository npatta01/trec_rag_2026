from __future__ import annotations

import pytest

from trec_rag.topic213_ragnarok_experiment import (
    SpacySentenceTokenizer,
    generate_ragnarok_top20_response,
    postprocess_ragnarok_response,
)
from trec_rag.topic213_response_experiment import PassageCorpus, PassageUnit


LABEL_A = '"What triggered the Korean War?"'
LABEL_B = '"How did the Korean War conclude?"'


@pytest.fixture(scope="module")
def tokenizer() -> SpacySentenceTokenizer:
    return SpacySentenceTokenizer()


def test_spacy_postprocessor_maps_inline_citations_to_zero_based_indexes(tokenizer):
    raw = f"""## {LABEL_A}
North Korea invaded South Korea [1]. The United Nations responded [1, 2].

## {LABEL_B}
An armistice stopped the fighting [2].
"""

    answer, sections = postprocess_ragnarok_response(
        raw,
        labels=[LABEL_A, LABEL_B],
        reference_count=2,
        tokenizer=tokenizer,
    )

    assert [row["text"] for row in answer] == [
        "North Korea invaded South Korea.",
        "The United Nations responded.",
        "An armistice stopped the fighting.",
    ]
    assert [row["citations"] for row in answer] == [[0], [0, 1], [1]]
    assert [row["sub_narrative"] for row in sections] == [LABEL_A, LABEL_B]


@pytest.mark.parametrize(
    "raw,error",
    [
        (
            f"## {LABEL_A}\nAn uncited factual sentence.\n## {LABEL_B}\nEnded [1].",
            "uncited sentence",
        ),
        (
            f"## {LABEL_A}\nStarted [3].\n## {LABEL_B}\nEnded [1].",
            "outside the top-2 references",
        ),
        (
            f"## {LABEL_B}\nEnded [1].\n## {LABEL_A}\nStarted [1].",
            "supplied order",
        ),
    ],
)
def test_spacy_postprocessor_rejects_invalid_ragnarok_output(tokenizer, raw, error):
    with pytest.raises(ValueError, match=error):
        postprocess_ragnarok_response(
            raw,
            labels=[LABEL_A, LABEL_B],
            reference_count=2,
            tokenizer=tokenizer,
        )


def test_generation_preserves_top20_reference_contract_and_document_ids():
    corpus = PassageCorpus(
        document_ids=("D1", "D2"),
        sub_narratives=(LABEL_A, LABEL_B),
        passages=(
            PassageUnit("P000001", "D1", LABEL_A, "North Korea invaded South Korea."),
            PassageUnit("P000002", "D2", LABEL_B, "An armistice stopped the fighting."),
        ),
        evidence_assignment_count=2,
    )

    class FakeClient:
        model = "fake-model"

        def complete_text(self, **_kwargs):
            return (
                f"## {LABEL_A}\nNorth Korea invaded South Korea [1].\n"
                f"## {LABEL_B}\nAn armistice stopped the fighting [2]."
            )

        def complete_json(self, *, stage, payload, **_kwargs):
            assert stage.startswith("audit_support")
            claim = payload["claims"][0]
            return {
                "assessments": [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "supported",
                        "notes": "",
                    }
                ]
            }

    generation, audit, artifacts = generate_ragnarok_top20_response(
        client=FakeClient(),
        topic_id="213",
        narrative="Korean War",
        corpus=corpus,
        selected_passage_ids=["P000001", "P000002"],
        release_accounting={"documents_processed": 2},
        retrieval_record={},
        generation_config={"evidence_depth": 2},
        experiment_id="experiment",
        run_id="run",
        source_metadata={},
    )

    record = artifacts["ragnarok_record"]
    assert record["references"] == ["P000001", "P000002"]
    assert [row["citations"] for row in record["answer"]] == [[0], [1]]
    assert [
        claim["document_ids"]
        for section in generation["sections"]
        for claim in section["claims"]
    ] == [["D1"], ["D2"]]
    assert len(audit) == 2
    assert generation["context_policy"]["nuggets_available_during_generation"] is False

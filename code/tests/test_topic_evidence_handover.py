"""Contracts for the deterministic Topic 213 evidence handover inputs."""

from __future__ import annotations

from itertools import pairwise
import json
from pathlib import Path

import pytest

from trec_rag.topic_evidence_handover import (
    CANONICAL_SUB_NARRATIVES,
    EligibleDocument,
    TopicEvidenceInputs,
    load_topic213_inputs,
    missing_document_ids,
    render_handover_markdown,
    score_sub_narrative_pairs,
    validate_reviewed_handover,
)


@pytest.fixture
def fixture_paths(tmp_path: Path) -> dict[str, object]:
    """Small real-format inputs; canonical population validation is disabled."""

    topic_tsv = tmp_path / "topics.tsv"
    topic_tsv.write_text(
        "212\tAnother topic\n213\tThe authoritative Topic 213 narrative\n",
        encoding="utf-8",
    )
    nuggets_jsonl = tmp_path / "nuggets.jsonl"
    nuggets_jsonl.write_text(
        json.dumps(
            {
                "qid": "213",
                "nuggets": [
                    {"mapped_sub_narrative": f"SN{number}"}
                    for number in range(1, 11)
                ]
                + [{"mapped_sub_narrative": "SN1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "213 0 doc-1 2\n213 0 doc-2 3\n213 0 doc-3 4\n"
        "213 0 excluded-1 1\n212 0 excluded-2 4\n",
        encoding="utf-8",
    )
    accepted_union = tmp_path / "accepted_union.jsonl"
    accepted_union.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"topic_id": "213", "document_id": "doc-2", "text": "second text"},
                {"topic_id": "213", "document_id": "doc-1", "text": "first text"},
                {"topic_id": "212", "document_id": "excluded-2", "text": "other topic"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    supplemental_documents = tmp_path / "supplemental.jsonl"
    supplemental_documents.write_text(
        json.dumps({"document_id": "doc-3", "text": "third text"}) + "\n",
        encoding="utf-8",
    )
    return {
        "topic_tsv": topic_tsv,
        "nuggets_jsonl": nuggets_jsonl,
        "qrels": qrels,
        "accepted_union": accepted_union,
        "supplemental_documents": supplemental_documents,
        "canonical": False,
    }


@pytest.fixture
def sample_handover() -> dict[str, object]:
    return {
        "topic_id": "213",
        "narrative": "The authoritative Topic 213 narrative",
        "sub_narratives": [
            {
                "sub_narrative": sub_narrative,
                "documents": [
                    {
                        "document_id": f"doc-{(number - 1) * 5 + rank}",
                        "topic_qrel_grade": 2,
                        "support_score": 2,
                        "claims": [f"Supported claim {number}-{rank}"],
                    }
                    for rank in range(1, 6)
                ],
            }
            for number, sub_narrative in enumerate(CANONICAL_SUB_NARRATIVES, start=1)
        ],
    }


@pytest.fixture
def sample_inputs() -> TopicEvidenceInputs:
    return TopicEvidenceInputs(
        topic_id="213",
        narrative="The authoritative Topic 213 narrative",
        sub_narratives=CANONICAL_SUB_NARRATIVES,
        documents=tuple(
            EligibleDocument(
                document_id=f"doc-{index}",
                text=f"source text {index}",
                topic_qrel_grade=2,
            )
            for index in range(1, 51)
        ),
    )


def deterministic_scorer(_sub_narrative: str, chunk_texts: list[str]) -> list[float]:
    """A local score function whose expected order comes from fixture IDs."""

    return [float(int(chunk_text.rsplit(" ", 1)[-1])) for chunk_text in chunk_texts]


def test_missing_document_ids_returns_sorted_qrel_documents_without_text(sample_inputs):
    """Catches a fetch plan that omits or nondeterministically orders absent qrel IDs."""

    assert missing_document_ids(
        eligible_docids={"doc-a", "doc-b", "doc-c"},
        available_docids={"doc-b"},
    ) == ["doc-a", "doc-c"]


def test_shortlist_scores_every_pair_and_orders_descending(sample_inputs):
    """Catches a shortlist that skips pairs or ranks lower model scores first."""

    result = score_sub_narrative_pairs(
        sample_inputs,
        deterministic_scorer,
        shortlist_depth=2,
    )

    assert result["pair_count"] == len(sample_inputs.sub_narratives) * len(sample_inputs.documents)
    assert all(
        row["model_score"] >= next_row["model_score"]
        for rows in result["shortlists"].values()
        for row, next_row in pairwise(rows)
    )


def test_shortlist_keeps_qrel_grade_separate_from_model_score(sample_inputs):
    """Catches a review shortlist that overwrites organizer grades with model logits."""

    row = score_sub_narrative_pairs(sample_inputs, deterministic_scorer)["shortlists"][
        sample_inputs.sub_narratives[0]
    ][0]

    assert isinstance(row["topic_qrel_grade"], int)
    assert isinstance(row["model_score"], float)


def test_load_topic213_inputs_preserves_population_and_subnarratives(fixture_paths):
    """Catches a loader that includes ineligible docs or alters nugget labels."""

    loaded = load_topic213_inputs(**fixture_paths)

    assert loaded.topic_id == "213"
    assert loaded.narrative == "The authoritative Topic 213 narrative"
    assert loaded.sub_narratives == tuple(f"SN{number}" for number in range(1, 11))
    assert [(document.document_id, document.topic_qrel_grade) for document in loaded.documents] == [
        ("doc-1", 2),
        ("doc-2", 3),
        ("doc-3", 4),
    ]


def test_loader_rejects_missing_text_for_eligible_document(fixture_paths):
    """Catches a loader that silently scores an eligible document without text."""

    accepted_union = fixture_paths["accepted_union"]
    assert isinstance(accepted_union, Path)
    accepted_union.write_text(
        json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "first text"})
        + "\n"
        + json.dumps({"topic_id": "213", "document_id": "doc-2", "text": ""})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing text"):
        load_topic213_inputs(**fixture_paths)


def test_loader_rejects_duplicate_accepted_union_document_id(fixture_paths):
    """Catches ambiguous joins caused by duplicate accepted-union identities."""

    accepted_union = fixture_paths["accepted_union"]
    assert isinstance(accepted_union, Path)
    accepted_union.write_text(
        "\n".join(
            (
                json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "first text"}),
                json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "duplicate text"}),
                json.dumps({"topic_id": "213", "document_id": "doc-2", "text": "second text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate document ID"):
        load_topic213_inputs(**fixture_paths)


def test_loader_merges_supplemental_document_text_before_eligibility_validation(fixture_paths):
    """Catches a loader that rejects qrel documents absent from the accepted union."""

    loaded = load_topic213_inputs(**fixture_paths)

    assert loaded.documents[-1].document_id == "doc-3"
    assert loaded.documents[-1].text == "third text"


def test_loader_allows_matching_accepted_union_and_supplemental_text(fixture_paths):
    """Catches a merge that rejects a harmless duplicate source record."""

    supplemental_documents = fixture_paths["supplemental_documents"]
    assert isinstance(supplemental_documents, Path)
    supplemental_documents.write_text(
        "\n".join(
            (
                json.dumps({"document_id": "doc-1", "text": "first text"}),
                json.dumps({"document_id": "doc-3", "text": "third text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_topic213_inputs(**fixture_paths)

    assert [document.document_id for document in loaded.documents] == ["doc-1", "doc-2", "doc-3"]


def test_loader_rejects_conflicting_accepted_union_and_supplemental_text(fixture_paths):
    """Catches a merge that silently selects one source's conflicting text."""

    supplemental_documents = fixture_paths["supplemental_documents"]
    assert isinstance(supplemental_documents, Path)
    supplemental_documents.write_text(
        "\n".join(
            (
                json.dumps({"document_id": "doc-1", "text": "conflicting text"}),
                json.dumps({"document_id": "doc-3", "text": "third text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting text"):
        load_topic213_inputs(**fixture_paths)


def test_loader_requires_canonical_population_of_173(fixture_paths):
    """Catches accidental use of a partial population in canonical artifact mode."""

    fixture_paths["canonical"] = True

    with pytest.raises(ValueError, match="173"):
        load_topic213_inputs(**fixture_paths)


def test_canonical_loader_requires_the_released_sub_narrative_tuple(fixture_paths):
    """Catches a canonical load that accepts ten altered or reordered labels."""

    qrels = fixture_paths["qrels"]
    accepted_union = fixture_paths["accepted_union"]
    nuggets_jsonl = fixture_paths["nuggets_jsonl"]
    assert isinstance(qrels, Path)
    assert isinstance(accepted_union, Path)
    assert isinstance(nuggets_jsonl, Path)
    qrels.write_text(
        "".join(f"213 0 doc-{index} 2\n" for index in range(1, 174)),
        encoding="utf-8",
    )
    accepted_union.write_text(
        "".join(
            json.dumps({"topic_id": "213", "document_id": f"doc-{index}", "text": f"text {index}"})
            + "\n"
            for index in range(1, 174)
        ),
        encoding="utf-8",
    )
    nuggets_jsonl.write_text(
        json.dumps(
            {
                "qid": "213",
                "nuggets": [
                    {"mapped_sub_narrative": label}
                    for label in (*CANONICAL_SUB_NARRATIVES[1:], CANONICAL_SUB_NARRATIVES[0])
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture_paths["supplemental_documents"] = None
    fixture_paths["canonical"] = True

    with pytest.raises(ValueError, match="exact released mapped_sub_narrative"):
        load_topic213_inputs(**fixture_paths)


def test_handover_requires_five_supported_unique_documents(sample_handover, sample_inputs):
    """Catches a reviewed record with a merely related, unusable document."""

    sample_handover["sub_narratives"][0]["documents"][0]["support_score"] = 1

    with pytest.raises(ValueError, match="support_score"):
        validate_reviewed_handover(
            sample_handover,
            inputs=sample_inputs,
        )


def test_handover_rejects_duplicate_or_uneligible_documents(sample_handover, sample_inputs):
    """Catches five-item lists that are not five eligible, distinct documents."""

    sample_handover["sub_narratives"][0]["documents"][4]["document_id"] = "doc-1"

    with pytest.raises(ValueError, match="unique"):
        validate_reviewed_handover(
            sample_handover,
            inputs=sample_inputs,
        )


def test_handover_requires_the_exact_canonical_sub_narrative_tuple(sample_handover, sample_inputs):
    """Catches a final artifact that swaps the released coverage-area order."""

    rows = sample_handover["sub_narratives"]
    assert isinstance(rows, list)
    rows[0]["sub_narrative"], rows[1]["sub_narrative"] = (
        rows[1]["sub_narrative"],
        rows[0]["sub_narrative"],
    )

    with pytest.raises(ValueError, match="exact released mapped_sub_narrative"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_handover_requires_qrel_grade_to_match_input_provenance(sample_handover, sample_inputs):
    """Catches a reviewer record that alters the organizer qrel grade."""

    rows = sample_handover["sub_narratives"]
    assert isinstance(rows, list)
    rows[0]["documents"][0]["topic_qrel_grade"] = 3

    with pytest.raises(ValueError, match="does not match input provenance"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_handover_accepts_explicit_qrel_grade_mapping(sample_handover):
    """Catches a validator that only works with the loader's concrete type."""

    validate_reviewed_handover(
        sample_handover,
        qrel_grades={f"doc-{index}": 2 for index in range(1, 51)},
    )


def test_handover_rejects_unallowlisted_content_and_absolute_paths(sample_handover, sample_inputs):
    """Catches artifacts that add raw content fields or absolute machine paths."""

    sample_handover["sub_narratives"][0]["documents"][0]["content"] = "raw document body"

    with pytest.raises(ValueError, match="allowlisted"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)

    del sample_handover["sub_narratives"][0]["documents"][0]["content"]
    sample_handover["sub_narratives"][0]["documents"][0]["claims"] = ["evidence=/tmp/raw.txt"]

    with pytest.raises(ValueError, match="absolute filesystem path"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


@pytest.mark.parametrize(
    ("target", "unsafe_value", "error"),
    (
        ("narrative", "PYSERINI_API_TOKEN=private-token", "sensitive credential value"),
        ("claims", "Authorization: Bearer private-token", "sensitive credential value"),
        ("claims", r"evidence=C:\private\raw.txt", "absolute filesystem path"),
        ("review_rationale", "evidence=~/private/raw.txt", "absolute filesystem path"),
    ),
)
def test_handover_rejects_sensitive_values_and_embedded_paths(
    sample_handover,
    sample_inputs,
    target,
    unsafe_value,
    error,
):
    """Catches secrets and filesystem paths hidden in permitted string fields."""

    document = sample_handover["sub_narratives"][0]["documents"][0]
    if target == "narrative":
        sample_handover["narrative"] = unsafe_value
    elif target == "claims":
        document["claims"] = [unsafe_value]
    else:
        document["review_rationale"] = unsafe_value

    with pytest.raises(ValueError, match=error):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_render_handover_markdown_exposes_review_fields_without_document_text(sample_handover):
    """Catches a renderer that leaks source text instead of the reviewed claims."""

    sample_handover["sub_narratives"][0]["documents"][0]["text"] = "raw ClimbMix source text"

    markdown = render_handover_markdown(sample_handover)

    assert CANONICAL_SUB_NARRATIVES[0] in markdown
    assert "doc-1" in markdown
    assert "topic qrel grade: 2" in markdown
    assert "support score: 2" in markdown
    assert "Supported claim 1-1" in markdown
    assert "raw ClimbMix source text" not in markdown

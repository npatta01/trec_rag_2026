from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.cached_segmentation_validation import compare_structural_runs
from trec_rag.document_store import DocumentStore
from trec_rag.facet_evidence import (
    CandidateSubnarrative,
    ExtractiveCandidateRequest,
    ScoredPassage,
    _scoring_text_and_boundaries,
    extract_document_candidates,
)
from trec_rag.generation_handoff import (
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
from trec_rag.topic_records import TopicRecordsBuilder


TOPIC_ID = "407"
NARRATIVE = "Explain how housing costs changed."
SOURCE = (
    "Housing Costs\n\n"
    "Rents soared because demand increased.\n"
    "Prices rose 17%.\n"
    "from Dubai and\n"
    "continued mid sentence.\n"
)


def _digest(value: str | bytes) -> str:
    body = value.encode("utf-8") if isinstance(value, str) else value
    return sha256(body).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


class _Scorer:
    @staticmethod
    def score_pairs(pairs):
        return tuple(float(len(pair.sentence_text)) for pair in pairs)


def _fixed_candidates():
    scoring_text, _ = _scoring_text_and_boundaries(SOURCE)
    request = ExtractiveCandidateRequest(
        topic_id=TOPIC_ID,
        document_id="doc-1",
        source=SOURCE,
        document_sha256=_digest(SOURCE),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("housing", "Housing cost evidence"),),
        passages=(
            ScoredPassage(
                "passage-1",
                "subnarrative:housing",
                "query-1",
                0,
                len(scoring_text),
                _digest(scoring_text),
                _digest(scoring_text),
                4.0,
                1,
            ),
        ),
    )
    return extract_document_candidates(request, _Scorer())


def _write_records(output_root: Path, store: DocumentStore):
    candidates = _fixed_candidates()
    builder = TopicRecordsBuilder(
        output_root / TOPIC_ID,
        TOPIC_ID,
        store,
        run_id="fixed-run",
    )
    builder.bind_document("doc-1", SOURCE, expected_sha256=_digest(SOURCE))
    for candidate in candidates:
        builder.add_candidate(candidate)
    builder.publish({"fixture": "cached-segmentation-validation"})
    return candidates


def _span(text: str) -> EvidenceSourceSpan:
    start_char = SOURCE.index(text)
    end_char = start_char + len(text)
    start_byte = len(SOURCE[:start_char].encode("utf-8"))
    end_byte = len(SOURCE[:end_char].encode("utf-8"))
    return EvidenceSourceSpan(start_char, end_char, start_byte, end_byte)


def _handoff(*, run_id: str, evidence_id: str, evidence_text: str) -> GenerationHandoff:
    evidence = EvidencePassage(
        evidence_id=evidence_id,
        group_id="housing",
        cluster_id="cluster-1",
        cluster_ordinal=1,
        support_ordinal=1,
        candidate_kind="exact_sentence",
        docid="doc-1",
        document_rank=1,
        text=evidence_text,
        document_sha256=_digest(SOURCE),
        source_span=_span(evidence_text),
    )
    group = EvidenceGroup(
        "housing",
        "generated_subnarrative",
        "Housing cost evidence",
        (SelectedCluster("cluster-1", 1, evidence_id, (evidence_id,)),),
    )
    claim = ClaimHint(
        "claim-1",
        "housing",
        "canonical",
        evidence_text,
        (evidence_id,),
    )
    topic = GenerationTopic(
        TOPIC_ID,
        NARRATIVE,
        (group,),
        (evidence,),
        (claim,),
        TopicSourceReceipts("a" * 64, "b" * 64),
    )
    return GenerationHandoff(
        HandoffProducer("topic_records_v4", run_id, "c" * 40),
        (topic,),
    )


def _write_sealed_upstream(output_root: Path) -> None:
    topic_root = output_root / TOPIC_ID
    decomposition = _canonical({"topic_id": TOPIC_ID, "narrative": NARRATIVE})
    decomposition_path = topic_root / "decomposition" / "result.json"
    decomposition_path.parent.mkdir(parents=True)
    decomposition_path.write_bytes(decomposition)
    (decomposition_path.parent / "manifest.json").write_bytes(
        _canonical(
            {
                "result_file": decomposition_path.name,
                "result_bytes": len(decomposition),
                "result_sha256": _digest(decomposition),
            }
        )
    )

    retrieval = _canonical({"topic_id": TOPIC_ID, "documents": ["doc-1"]})
    retrieval_path = topic_root / "retrieval" / "audit.json"
    retrieval_path.parent.mkdir(parents=True)
    retrieval_path.write_bytes(retrieval)
    (retrieval_path.parent / "complete.json").write_bytes(
        _canonical(
            {
                "artifacts": [
                    {
                        "relative_path": "retrieval/audit.json",
                        "bytes": len(retrieval),
                        "sha256": _digest(retrieval),
                    }
                ]
            }
        )
    )

    selected = _canonical(
        {
            "topic_id": TOPIC_ID,
            "docid": "doc-1",
            "selection_rank": 1,
            "text_sha256": _digest(SOURCE),
            "text": SOURCE,
        }
    )
    selected_path = topic_root / "scoring" / "selected_documents.jsonl"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_bytes(selected)
    (selected_path.parent / "complete.json").write_bytes(
        _canonical(
            {
                "artifacts": [
                    {
                        "relative_path": "scoring/selected_documents.jsonl",
                        "bytes": len(selected),
                        "sha256": _digest(selected),
                    }
                ]
            }
        )
    )


def _fixture(tmp_path: Path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    store_root = tmp_path / "objects"
    store = DocumentStore(store_root)
    store.admit_text(SOURCE)
    candidates = _write_records(candidate_root, store)
    _write_sealed_upstream(baseline_root)
    _write_sealed_upstream(candidate_root)
    old_text = "from Dubai and"
    fixed = max(candidates, key=lambda candidate: (len(candidate.text), candidate.text))
    baseline_handoff = baseline_root / "generation_handoff_manifest.json"
    candidate_handoff = candidate_root / "generation_handoff_manifest.json"
    write_generation_handoff(
        baseline_handoff,
        _handoff(run_id="baseline-run", evidence_id="old-fragment", evidence_text=old_text),
    )
    write_generation_handoff(
        candidate_handoff,
        _handoff(
            run_id="fixed-run",
            evidence_id=fixed.candidate_nugget_id,
            evidence_text=fixed.text,
        ),
    )
    return baseline_root, candidate_root, store_root, baseline_handoff, candidate_handoff


def test_structural_comparison_authenticates_same_sources_and_improves_fragments(
    tmp_path: Path,
) -> None:
    baseline_root, candidate_root, store_root, old_handoff, fixed_handoff = _fixture(
        tmp_path
    )

    comparison = compare_structural_runs(
        baseline_output_root=baseline_root,
        candidate_output_root=candidate_root,
        document_store_root=store_root,
        baseline_handoff_path=old_handoff,
        candidate_handoff_path=fixed_handoff,
        topic_ids=(TOPIC_ID,),
    )

    topic = comparison.topics[0]
    assert topic.topic_id == TOPIC_ID
    assert topic.selected_document_count == 1
    assert topic.old_line_units.count == 5
    assert topic.fixed_segmentation_units.median_characters > (
        topic.old_line_units.median_characters
    )
    assert topic.baseline_representatives.fragment_count == 1
    assert topic.candidate_representatives.fragment_count == 0
    assert topic.candidate_units.count == len(_fixed_candidates())
    assert comparison.aggregate_candidate_representatives.fragment_fraction == 0.0
    assert comparison.gates_passed is True


def test_structural_comparison_rejects_changed_authenticated_upstream(
    tmp_path: Path,
) -> None:
    baseline_root, candidate_root, store_root, old_handoff, fixed_handoff = _fixture(
        tmp_path
    )
    selected_path = candidate_root / TOPIC_ID / "scoring" / "selected_documents.jsonl"
    selected_path.write_bytes(selected_path.read_bytes().replace(b"Prices", b"Values"))

    with pytest.raises(ValueError, match="selected_documents.*changed|artifact.*changed"):
        compare_structural_runs(
            baseline_output_root=baseline_root,
            candidate_output_root=candidate_root,
            document_store_root=store_root,
            baseline_handoff_path=old_handoff,
            candidate_handoff_path=fixed_handoff,
            topic_ids=(TOPIC_ID,),
        )


def test_structural_comparison_reports_aggregate_representative_regression(
    tmp_path: Path,
) -> None:
    baseline_root, candidate_root, store_root, old_handoff, fixed_handoff = _fixture(
        tmp_path
    )
    candidates = _fixed_candidates()
    good = max(candidates, key=lambda candidate: (len(candidate.text), candidate.text))
    bad = min(candidates, key=lambda candidate: (len(candidate.text), candidate.text))
    old_handoff.unlink()
    fixed_handoff.unlink()
    write_generation_handoff(
        old_handoff,
        _handoff(
            run_id="baseline-run",
            evidence_id="baseline-good",
            evidence_text=good.text,
        ),
    )
    write_generation_handoff(
        fixed_handoff,
        _handoff(
            run_id="fixed-run",
            evidence_id=bad.candidate_nugget_id,
            evidence_text=bad.text,
        ),
    )

    comparison = compare_structural_runs(
        baseline_output_root=baseline_root,
        candidate_output_root=candidate_root,
        document_store_root=store_root,
        baseline_handoff_path=old_handoff,
        candidate_handoff_path=fixed_handoff,
        topic_ids=(TOPIC_ID,),
    )

    assert comparison.gates_passed is False
    assert "aggregate representative fragment fraction worsened" in (
        comparison.gate_failures
    )

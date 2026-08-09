from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.cached_segmentation_validation import (
    compare_semantic_runs,
    compare_structural_runs,
)
from trec_rag.document_store import DocumentStore
from trec_rag.facet_extraction import BackendReply
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
from trec_rag.retrieval_nugget_coverage import (
    CoverageModelRequest,
    CoverageRunConfig,
    run_coverage_evaluation,
    validate_and_freeze_plan,
)


TOPIC_ID = "407"
NARRATIVE = "Explain how housing costs changed."
SOURCE = (
    "Housing Costs\n\n"
    "Rents soared because demand increased.\n"
    "Prices rose 17%.\n"
    "from Dubai and\n"
    "continued mid sentence.\n"
)
SEMANTIC_PLAN = {
    "schema_version": "retrieval_nugget_plan_v1",
    "facets": [
        {
            "title": "Housing costs",
            "obligations": [
                {
                    "requirement": "Explain how housing costs changed.",
                    "support_test": "The direction of the change is present.",
                    "kind": "required_explicit",
                    "narrative_spans": ["how housing costs changed"],
                }
            ],
        }
    ],
    "unmapped_narrative_spans": [],
}


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


class _CoverageBackend:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.requests: list[CoverageModelRequest] = []

    def complete(self, request: CoverageModelRequest) -> BackendReply:
        self.requests.append(request)
        content = json.dumps(self.payloads.pop(0), separators=(",", ":")).encode()
        return BackendReply(
            content=content,
            response_body=b'{"provider":"fixture"}',
            status=200,
            metadata={"requested_model": request.model},
        )


def _semantic_judgment(label: str) -> dict[str, object]:
    return {
        "schema_version": "retrieval_nugget_judgment_v1",
        "judgments": [
            {
                "obligation_id": "f001-o001",
                "label": label,
                "supporting_nugget_aliases": (
                    [] if label == "unsupported" else ["n001"]
                ),
                "missing_elements": "" if label == "full" else "Missing detail.",
            }
        ],
    }


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


def _write_baseline_coverage(
    handoff_path: Path,
    coverage_root: Path,
    *,
    label: str = "partial",
) -> None:
    planner = _CoverageBackend([SEMANTIC_PLAN])
    judge = _CoverageBackend([_semantic_judgment(label)])
    run_coverage_evaluation(
        CoverageRunConfig(
            handoff_manifest_path=handoff_path,
            topic_id=TOPIC_ID,
            work_dir=coverage_root / TOPIC_ID,
            allow_hosted_calls=True,
        ),
        planner=planner,
        judge=judge,
    )
    assert [request.stage for request in planner.requests] == ["planner"]
    assert [request.stage for request in judge.requests] == ["judge"]


def test_semantic_comparison_reuses_plan_and_reports_candidate_improvement(
    tmp_path: Path,
) -> None:
    _, _, _, baseline_handoff, candidate_handoff = _fixture(tmp_path)
    baseline_coverage = tmp_path / "baseline-coverage"
    candidate_coverage = tmp_path / "candidate-coverage"
    _write_baseline_coverage(baseline_handoff, baseline_coverage)
    fixed_judge = _CoverageBackend([_semantic_judgment("full")])

    comparison = compare_semantic_runs(
        baseline_handoff_path=baseline_handoff,
        baseline_coverage_root=baseline_coverage,
        candidate_handoff_path=candidate_handoff,
        candidate_coverage_root=candidate_coverage,
        topic_ids=(TOPIC_ID,),
        judge=fixed_judge,
    )

    topic = comparison.topics[0]
    expected_plan = validate_and_freeze_plan(NARRATIVE, SEMANTIC_PLAN)
    assert topic.plan_sha256 == expected_plan.plan_sha256
    assert topic.baseline_required_coverage == 0.5
    assert topic.candidate_required_coverage == 1.0
    assert topic.required_coverage_delta == 0.5
    assert topic.strict_full_rate_delta == 1.0
    assert topic.improved_obligation_ids == ("f001-o001",)
    assert topic.regressed_obligation_ids == ()
    assert topic.baseline_artifact_sha256s
    assert topic.candidate_artifact_sha256s
    assert comparison.planner_calls == 0
    assert comparison.candidate_judge_calls == 1
    assert comparison.gates_passed is True
    assert [request.stage for request in fixed_judge.requests] == ["judge"]
    assert (candidate_coverage / TOPIC_ID / "plan.json").read_bytes() == (
        baseline_coverage / TOPIC_ID / "plan.json"
    ).read_bytes()
    resumed_judge = _CoverageBackend([])
    resumed = compare_semantic_runs(
        baseline_handoff_path=baseline_handoff,
        baseline_coverage_root=baseline_coverage,
        candidate_handoff_path=candidate_handoff,
        candidate_coverage_root=candidate_coverage,
        topic_ids=(TOPIC_ID,),
        judge=resumed_judge,
    )
    assert resumed == comparison
    assert resumed_judge.requests == []


def test_semantic_comparison_lists_regression_and_fails_macro_gate(
    tmp_path: Path,
) -> None:
    _, _, _, baseline_handoff, candidate_handoff = _fixture(tmp_path)
    baseline_coverage = tmp_path / "baseline-coverage"
    _write_baseline_coverage(baseline_handoff, baseline_coverage, label="full")

    comparison = compare_semantic_runs(
        baseline_handoff_path=baseline_handoff,
        baseline_coverage_root=baseline_coverage,
        candidate_handoff_path=candidate_handoff,
        candidate_coverage_root=tmp_path / "candidate-coverage",
        topic_ids=(TOPIC_ID,),
        judge=_CoverageBackend([_semantic_judgment("partial")]),
    )

    assert comparison.topics[0].regressed_obligation_ids == ("f001-o001",)
    assert comparison.gates_passed is False
    assert comparison.gate_failures == (
        "topic-macro required coverage regressed",
        "topic-macro strict-full rate regressed",
    )


def test_semantic_comparison_rejects_missing_complete_baseline(
    tmp_path: Path,
) -> None:
    _, _, _, baseline_handoff, candidate_handoff = _fixture(tmp_path)

    with pytest.raises(ValueError, match="completed coverage bundle is missing"):
        compare_semantic_runs(
            baseline_handoff_path=baseline_handoff,
            baseline_coverage_root=tmp_path / "missing-baseline",
            candidate_handoff_path=candidate_handoff,
            candidate_coverage_root=tmp_path / "candidate-coverage",
            topic_ids=(TOPIC_ID,),
            judge=_CoverageBackend([]),
        )

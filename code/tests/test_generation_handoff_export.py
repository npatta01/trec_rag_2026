from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.canonical_nuggets import (
    OPENROUTER_DEEPSEEK_MODEL,
    _result_json,
    build_canonical_nugget_request,
    canonicalize_subnarrative,
)
from trec_rag.document_store import DocumentStore
from trec_rag.evidence_store import (
    CandidateArtifacts,
    load_validated_candidate_artifacts,
)
from trec_rag.facet_evidence import (
    BudgetSnapshot,
    CandidateSubnarrative,
    EvidenceMember,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SelectionPolicy,
    SemanticCluster,
    SubnarrativeContext,
    SubnarrativeSelection,
    _scoring_text_and_boundaries,
    extract_document_candidates,
)
from trec_rag.facet_extraction import BackendReply
from trec_rag.generation_handoff import (
    GenerationHandoff,
    HandoffProducer,
    SOURCE_CONTRACT,
    serialize_generation_handoff,
)
from trec_rag.generation_handoff_export import (
    ValidatedGenerationSnapshot,
    prepare_generation_handoff_artifact,
    project_generation_topic,
)
from trec_rag.topic_records import TopicRecordsBuilder
from trec_rag.topics import Topic


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class _IncreasingScorer:
    def score_pairs(self, pairs):
        return tuple(1.0 + index / 10 for index, _pair in enumerate(pairs))


class _CanonicalBackend:
    def complete(self, request) -> BackendReply:
        alpha = next(
            evidence
            for evidence in request.evidence
            if evidence.text == "Alpha café evidence."
        )
        content = json.dumps(
            {
                "claims": [
                    {
                        "claim": "Alpha is the selected safety finding.",
                        "evidence_aliases": [alpha.alias],
                        "importance": "vital",
                    }
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return BackendReply(
            content=content,
            response_body=b'{"fixture":"canonical"}',
            status=200,
            metadata={
                "requested_model": OPENROUTER_DEEPSEEK_MODEL,
                "response_model": OPENROUTER_DEEPSEEK_MODEL,
                "provider": "fixture",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


def _candidates(topic_id: str, docid: str, source: str, subnarrative: str):
    scoring_text, _boundaries = _scoring_text_and_boundaries(source)
    request = ExtractiveCandidateRequest(
        topic_id=topic_id,
        document_id=docid,
        source=source,
        document_sha256=_digest(source),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("sub-1", subnarrative),),
        passages=(
            ScoredPassage(
                f"{docid}-passage",
                "original",
                topic_id,
                0,
                len(scoring_text),
                _digest(scoring_text),
                _digest(scoring_text),
                2.0,
                1,
            ),
        ),
    )
    return extract_document_candidates(request, _IncreasingScorer())


def _member(candidate) -> EvidenceMember:
    return EvidenceMember(
        candidate_nugget_id=candidate.candidate_nugget_id,
        candidate_kind=candidate.candidate_kind,
        text=candidate.text,
        docid=candidate.docid,
        document_sha256=candidate.document_sha256,
        raw_logit=candidate.sentence_cross_encoder_score,
    )


def _fixture(tmp_path: Path, *, original_narrative_fallback: bool = False):
    topic = Topic(
        id="rag2026-58",
        title="unused",
        narrative="Explain the documented safety findings.",
    )
    subnarrative = (
        topic.narrative
        if original_narrative_fallback
        else "Documented safety findings"
    )
    first_source = "Préface.\n\nAlpha café evidence. Gamma evidence repeats document A."
    second_source = "Intro.\n\nBeta Ω evidence."
    first_candidates = _candidates(topic.id, "doc-a", first_source, subnarrative)
    second_candidates = _candidates(topic.id, "doc-b", second_source, subnarrative)
    alpha = next(row for row in first_candidates if row.text == "Alpha café evidence.")
    gamma = next(
        row for row in first_candidates if row.text == "Gamma evidence repeats document A."
    )
    beta = next(row for row in second_candidates if row.text == "Beta Ω evidence.")

    store = DocumentStore(tmp_path / "objects")
    topic_root = tmp_path / topic.id
    builder = TopicRecordsBuilder(topic_root, topic.id, store, run_id="fixture-run")
    builder.bind_document("doc-a", first_source)
    builder.bind_document("doc-b", second_source)
    for candidate in (*first_candidates, *second_candidates):
        builder.add_candidate(candidate)
    receipt = builder.publish(
        {
            "request_schema_version": "extractive_candidate_request_v1",
            "candidate_schema_version": "extractive_candidate_nugget_v1",
            "source_file": "candidate-requests.jsonl",
            "source_sha256": "f" * 64,
            "scorer": {
                "model": "fixture-scorer",
                "model_revision": "fixture-v1",
                "backend_version": "fixture-v1",
                "score_representation": "raw_logits",
                "inference_dtype": "float32",
                "score_kind": "extractive_sentence_v1",
                "sentence_max_length": 512,
                "input_policy": "trec_rag_whitespace_v1",
            },
            "sentence_splitter_version": "exact_rules_v1",
            "scoring_normalization_version": "trec_rag_whitespace_v1",
        }
    )
    context = SubnarrativeContext(
        topic.id,
        topic.narrative,
        "sub-1",
        subnarrative,
    )
    beta_member, alpha_member, gamma_member = map(_member, (beta, alpha, gamma))
    clusters = (
        SemanticCluster(
            cluster_id="cluster-1",
            representative_candidate_nugget_id=beta.candidate_nugget_id,
            representative_text=beta.text,
            representative_raw_logit=beta.sentence_cross_encoder_score,
            members=(beta_member, alpha_member),
            supports=(beta_member, alpha_member),
            support_document_count=2,
        ),
        SemanticCluster(
            cluster_id="cluster-2",
            representative_candidate_nugget_id=gamma.candidate_nugget_id,
            representative_text=gamma.text,
            representative_raw_logit=gamma.sentence_cross_encoder_score,
            members=(gamma_member,),
            supports=(gamma_member,),
            support_document_count=1,
        ),
    )
    policy = SelectionPolicy(budgets=(2,))
    selection = SubnarrativeSelection(
        schema_version="subnarrative_selection_v1",
        context=context,
        policy=policy,
        similarity_identity=(("model", "fixture"),),
        candidate_count=3,
        exact_group_count=3,
        precluster_count=3,
        semantic_cluster_count=2,
        clusters=clusters,
        snapshots=(
            BudgetSnapshot(
                budget=2,
                cluster_ids=("cluster-1", "cluster-2"),
                exhausted=False,
            ),
        ),
    )
    required_candidate_keys = frozenset(
        (selection.context.subnarrative_id, member.candidate_nugget_id)
        for cluster in clusters
        for member in cluster.supports
    )
    candidates = load_validated_candidate_artifacts(
        CandidateArtifacts(
            topic_root / "records.sqlite3",
            topic_root / "canonical" / "records-manifest.json",
            tmp_path / "objects",
        ),
        expected_topic_id=topic.id,
        required_candidate_keys=required_candidate_keys,
    )
    request = build_canonical_nugget_request(selection, 2)
    result = _result_json(canonicalize_subnarrative(request, _CanonicalBackend()))
    return topic, candidates, receipt, selection, result, (first_source, second_source)


def _snapshot(
    topic: Topic,
    candidates,
    retrieval_topic_sha256: str,
    selection: SubnarrativeSelection,
    canonical_results,
    *,
    original_narrative_fallback: bool = False,
) -> ValidatedGenerationSnapshot:
    return ValidatedGenerationSnapshot(
        topic=topic,
        official_topics_sha256="a" * 64,
        retrieval_topic_sha256=retrieval_topic_sha256,
        selections=(selection,),
        candidates=candidates,
        canonical_results=tuple(canonical_results),
        selected_budget=2,
        max_canonical_claims=20,
        max_supporting_documents_per_claim=3,
        document_ranks={"doc-a": 1, "doc-b": 2},
        original_narrative_fallback=original_narrative_fallback,
    )


def test_projector_reconstructs_only_snapshot_evidence_and_keeps_unused_support(
    tmp_path: Path,
) -> None:
    topic, candidates, receipt, selection, result, sources = _fixture(tmp_path)
    first_source, second_source = sources

    projected = project_generation_topic(
        _snapshot(
            topic,
            candidates,
            receipt.semantic_sha256,
            selection,
            (result,),
        )
    )

    assert projected.source_receipts.retrieval_topic_sha256 == receipt.semantic_sha256
    assert projected.citation_docids == ("doc-a", "doc-b")
    assert [row.docid for row in projected.evidence] == ["doc-b", "doc-a", "doc-a"]
    assert [row.text for row in projected.evidence] == [
        "Beta Ω evidence.",
        "Alpha café evidence.",
        "Gamma evidence repeats document A.",
    ]
    assert projected.groups[0].selected_clusters[0].evidence_ids == tuple(
        row.evidence_id for row in projected.evidence[:2]
    )
    assert projected.claim_hints[0].evidence_ids == (
        projected.evidence[1].evidence_id,
    )
    assert projected.evidence[0].evidence_id not in projected.claim_hints[0].evidence_ids
    assert projected.evidence[2].evidence_id not in projected.claim_hints[0].evidence_ids

    sources_by_doc = {"doc-a": first_source, "doc-b": second_source}
    for evidence in projected.evidence:
        source = sources_by_doc[evidence.docid]
        span = evidence.source_span
        assert span.start_char > 0
        assert span.start_byte > 0
        if evidence.docid == "doc-a":
            assert span.start_byte > span.start_char
        assert source[span.start_char : span.end_char] == evidence.text
        assert (
            source.encode("utf-8")[span.start_byte : span.end_byte].decode("utf-8")
            == evidence.text
        )

    handoff = GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="facet-deepseek-selected-evidence-v1",
            producer_revision="fixture-revision",
        ),
        topics=(projected,),
    )
    projected_again = project_generation_topic(
        _snapshot(
            topic,
            candidates,
            receipt.semantic_sha256,
            selection,
            (result,),
        )
    )
    second = GenerationHandoff(producer=handoff.producer, topics=(projected_again,))
    assert serialize_generation_handoff(handoff) == serialize_generation_handoff(second)


def test_projector_marks_a_single_explicit_official_narrative_fallback(
    tmp_path: Path,
) -> None:
    topic, candidates, receipt, selection, _result, _sources = _fixture(
        tmp_path,
        original_narrative_fallback=True,
    )

    projected = project_generation_topic(
        _snapshot(
            topic,
            candidates,
            receipt.semantic_sha256,
            selection,
            (),
            original_narrative_fallback=True,
        )
    )

    assert len(projected.groups) == 1
    assert projected.groups[0].kind == "official_narrative_fallback"
    assert projected.groups[0].text == topic.narrative
    assert projected.claim_hints == ()


def test_projector_keeps_selected_evidence_without_canonical_advice(
    tmp_path: Path,
) -> None:
    topic, candidates, receipt, selection, _result, _sources = _fixture(tmp_path)

    projected = project_generation_topic(
        _snapshot(
            topic,
            candidates,
            receipt.semantic_sha256,
            selection,
            (),
        )
    )

    assert projected.claim_hints == ()
    assert projected.citation_docids == ("doc-a", "doc-b")
    assert len(projected.evidence) == 3


def test_projector_rejects_unknown_or_duplicate_canonical_advice(
    tmp_path: Path,
) -> None:
    topic, candidates, receipt, selection, result, _sources = _fixture(tmp_path)
    unknown = {**result, "subnarrative_id": "not-selected"}

    with pytest.raises(ValueError, match="unknown selected group"):
        project_generation_topic(
            _snapshot(
                topic,
                candidates,
                receipt.semantic_sha256,
                selection,
                (unknown,),
            )
        )
    with pytest.raises(ValueError, match="duplicate canonical result"):
        project_generation_topic(
            _snapshot(
                topic,
                candidates,
                receipt.semantic_sha256,
                selection,
                (result, result),
            )
        )


def test_prepare_handoff_artifact_is_deterministic_and_does_not_publish_early(
    tmp_path: Path,
) -> None:
    topic, candidates, receipt, selection, result, _sources = _fixture(tmp_path)
    projected = project_generation_topic(
        _snapshot(
            topic,
            candidates,
            receipt.semantic_sha256,
            selection,
            (result,),
        )
    )
    output_dir = tmp_path / "export"

    first = prepare_generation_handoff_artifact(
        output_dir=output_dir,
        retrieval_run_id="facet-deepseek-selected-evidence-v1",
        producer_revision="b" * 40,
        topics=(projected,),
    )
    second = prepare_generation_handoff_artifact(
        output_dir=output_dir,
        retrieval_run_id="facet-deepseek-selected-evidence-v1",
        producer_revision="b" * 40,
        topics=(projected,),
    )

    assert first.path == output_dir / "generation_handoff_manifest.json"
    assert not first.path.exists()
    assert first.body == second.body == serialize_generation_handoff(first.handoff)
    assert first.sha256 == second.sha256 == sha256(first.body).hexdigest()
    assert first.handoff.topics == (projected,)
    assert first.handoff.producer.to_payload() == {
        "source_contract": SOURCE_CONTRACT,
        "retrieval_run_id": "facet-deepseek-selected-evidence-v1",
        "producer_revision": "b" * 40,
    }

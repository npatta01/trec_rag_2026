from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.facet_evidence import (
    CandidateSubnarrative,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SelectionCandidate,
    SelectionPolicy,
    SubnarrativeContext,
    _byte_offsets,
    _scoring_text_and_boundaries,
    _word_before,
    extract_document_candidates,
    select_subnarrative_candidates,
)
from trec_rag.evidence_store import (
    CandidateArtifacts,
    generate_candidate_artifacts,
    load_validated_candidate_artifacts,
    materialize_candidate_inputs,
    select_evidence_artifacts,
    write_candidate_jsonl,
)
from trec_rag.facet_extraction import FacetPlanningResult, plan_facet_queries
from trec_rag.competition_retrieval import ValidatedDecomposition
from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


def _digest(value: str | bytes) -> str:
    body = value.encode("utf-8") if isinstance(value, str) else value
    return sha256(body).hexdigest()


def _scoring_text(source: str) -> str:
    return " ".join(source.split())


class _LiteralScorer:
    identity = {
        "model": "literal-local-scorer",
        "model_revision": "test-v1",
        "backend_version": "test-v1",
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "score_kind": "extractive_sentence_v1",
        "sentence_max_length": 512,
    }

    def score_pairs(self, pairs):
        values = {
            "Dr. Ada measured  3.5 meters.": 1.25,
            "This result confirms safety.": 3.5,
            "Ωmega evidence proves durability!": -0.5,
            "Another detail ends.": 0.25,
        }
        return tuple(values[pair.sentence_text] for pair in pairs)


class _FixedScorer:
    identity = _LiteralScorer.identity

    def __init__(self, values):
        self.values = values

    def score_pairs(self, pairs):
        return self.values(pairs) if callable(self.values) else self.values


def _request() -> ExtractiveCandidateRequest:
    source = "Evidence sentence. This depends on it."
    scoring_text = _scoring_text(source)
    return ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-validation",
        source=source,
        document_sha256=_digest(source),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(
            ScoredPassage(
                passage_id="p1",
                lane_id="original",
                query_id="topic-224",
                scoring_start_char=0,
                scoring_end_char=len(scoring_text),
                scoring_text_sha256=_digest(scoring_text),
                chunk_text_sha256=_digest(scoring_text),
                cross_encoder_score=1.0,
                cross_encoder_rank=1,
            ),
        ),
    )


def test_word_before_scans_backward_without_copying_the_paragraph_prefix() -> None:
    """Regress the quadratic prefix copy in sentence-boundary validation."""

    class NoSliceText(str):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("sentence splitting must not copy a text prefix")
            return super().__getitem__(key)

    text = NoSliceText("A long paragraph ends with e.g.")

    assert _word_before(text, len(text) - 1) == "e.g"


def test_source_coordinate_cache_reuses_document_offsets() -> None:
    class CountedText(str):
        iterations = 0

        def __iter__(self):
            type(self).iterations += 1
            if type(self).iterations > 1:
                raise AssertionError("document coordinates must be reused")
            return super().__iter__()

    source = CountedText("Repeated source text.")

    assert _byte_offsets(source) == _byte_offsets(source)


def test_bounded_document_projection_cache_retains_only_current_document() -> None:
    """Document projection caches release superseded document text."""
    _scoring_text_and_boundaries.cache_clear()
    _byte_offsets.cache_clear()
    try:
        sources = (
            "First\t document.",
            "Second\n document.",
            "Café  Ωmega.",
        )

        for source in sources:
            _scoring_text_and_boundaries(source)
            _byte_offsets(source)

        assert _scoring_text_and_boundaries.cache_info().currsize == 1
        assert _byte_offsets.cache_info().currsize == 1
        assert _scoring_text_and_boundaries("Café  Ωmega.") == (
            "Café Ωmega.",
            (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12),
        )
        assert _byte_offsets("Café  Ωmega.") == (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14)
    finally:
        _scoring_text_and_boundaries.cache_clear()
        _byte_offsets.cache_clear()


def test_exact_extraction_keeps_character_byte_and_adjacent_evidence() -> None:
    source = (
        "Heading: Café findings\n\n"
        "Dr. Ada measured  3.5 meters. This result confirms safety.\n\n"
        "Ωmega evidence proves durability! Another detail ends.\n"
    )
    scoring_text = _scoring_text(source)
    first_start = scoring_text.index("Dr. Ada")
    first_end = scoring_text.index(" Ωmega")
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-unicode",
        source=source,
        document_sha256=_digest(source),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(ScoredPassage(
            "p1", "original", "topic-224", first_start, first_end,
            _digest(scoring_text), _digest(scoring_text[first_start:first_end]), 7.2, 1,
        ),),
    )

    candidates = extract_document_candidates(request, _LiteralScorer())

    measured = next(row for row in candidates if row.text == "Dr. Ada measured  3.5 meters.")
    sentence = measured.evidence_sentences[0]
    assert source[sentence.start_char:sentence.end_char] == measured.text
    assert source.encode()[sentence.start_byte:sentence.end_byte].decode() == measured.text
    assert measured.context_before is not None
    assert measured.context_before.text == "Heading: Café findings"
    assert measured.context_after is not None
    assert measured.context_after.text == "Ωmega evidence proves durability! Another detail ends."
    assert [row.text for row in candidates if row.candidate_kind == "exact_sentence_pair"] == [
        "Dr. Ada measured  3.5 meters. This result confirms safety."
    ]


def test_candidate_bytes_are_deterministic_and_source_hashes_fail_closed(tmp_path: Path) -> None:
    request = _request()
    passage = request.passages[0]
    scoring_text = _scoring_text(request.source)
    second = ScoredPassage(
        "p2", "subnarrative", "sub-safety", scoring_text.index("This"), len(scoring_text),
        _digest(scoring_text), _digest(scoring_text[scoring_text.index("This"):]), 0.5, 2,
    )
    scorer = _FixedScorer(lambda pairs: tuple(float(i + 1) for i, _ in enumerate(pairs)))
    first = extract_document_candidates(replace(request, passages=(passage, second, second)), scorer)
    reordered = extract_document_candidates(replace(request, passages=(second, passage, second)), scorer)
    first_path, second_path = tmp_path / "first.jsonl", tmp_path / "second.jsonl"

    first_hash = write_candidate_jsonl(first_path, first)
    second_hash = write_candidate_jsonl(second_path, tuple(reversed(reordered)))

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_hash == second_hash == _digest(first_path.read_bytes())
    with pytest.raises(ValueError, match="document_sha256"):
        extract_document_candidates(replace(request, document_sha256="0" * 64), scorer)


def test_candidate_artifact_matches_fixed_golden_bytes(tmp_path: Path) -> None:
    source = "Evidence sentence."
    digest = "2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-a",
        source=source,
        document_sha256=digest,
        scoring_text_sha256=digest,
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(ScoredPassage(
            "p1", "original", "topic-224", 0, 18, digest, digest, 1.0, 1,
        ),),
    )
    output = tmp_path / "candidates.jsonl"

    write_candidate_jsonl(
        output,
        extract_document_candidates(request, _FixedScorer((1.25,))),
    )

    assert output.read_bytes() == (
        b'{"candidate_kind":"exact_sentence","candidate_nugget_id":"ecn1_d493c7728fc4a51bd71774e2274b50d1541ae595ea92adcadde3eaf7aa1317b1","context_after":null,"context_before":null,"docid":"doc-a","document_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","evidence_sentences":[{"cross_encoder_score":1.25,"end_byte":18,"end_char":18,"start_byte":0,"start_char":0,"text":"Evidence sentence.","text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"}],"matched_paragraph":{"end_byte":18,"end_char":18,"start_byte":0,"start_char":0,"text":"Evidence sentence.","text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"},"nugget_type":"extractive","passages":[{"chunk_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","cross_encoder_rank":1,"cross_encoder_score":1.0,"lane_id":"original","normalization_version":"trec_rag_whitespace_v1","passage_id":"p1","query_id":"topic-224","scoring_end_char":18,"scoring_start_char":0,"scoring_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","source_end_byte":18,"source_end_char":18,"source_start_byte":0,"source_start_char":0,"source_text":"Evidence sentence.","source_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"}],"rank_within_document_subnarrative":1,"schema_version":"extractive_candidate_nugget_v1","scoring_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","sentence_cross_encoder_score":1.25,"sentence_splitter_version":"exact_rules_v1","subnarrative_id":"safety","subnarrative_sha256":"2e194dbac187930e8b8016c241be054aaeb7e4344f8676edbefa0ddadb065db0","text":"Evidence sentence.","topic_id":"224"}\n'
    )


def test_nonfinite_sentence_scores_fail_closed() -> None:
    with pytest.raises(ValueError, match="sentence score must be finite"):
        extract_document_candidates(
            _request(),
            _FixedScorer(lambda pairs: (float("nan"),) * len(pairs)),
        )


class _Similarity:
    identity = {"model": "literal-test-matrix", "revision": "test-v1", "normalized": True}

    def __init__(self, pairs: dict[frozenset[str], float]) -> None:
        self.pairs = pairs

    def cosine_matrix(self, texts):
        return tuple(tuple(
            1.0 if left == right else self.pairs.get(frozenset((left, right)), 0.0)
            for right in texts
        ) for left in texts)


def _selection_candidate(identifier: str, text: str, score: float, document: str) -> SelectionCandidate:
    return SelectionCandidate(
        topic_id="224",
        subnarrative_id="sub-1",
        candidate_nugget_id=identifier,
        candidate_kind="exact_sentence",
        text=text,
        docid=f"doc-{document}",
        document_sha256=document * 64,
        raw_logit=score,
    )


def test_selection_deduplicates_clusters_diversifies_and_honors_budgets() -> None:
    context = SubnarrativeContext("224", "Explain coastal safety.", "sub-1", "Coastal safety")
    opened = "Acme opened the coastal center."
    inaugurated = "Acme inaugurated the coastal center."
    shelters = "Shelters received emergency supplies."
    candidates = (
        _selection_candidate("n1", opened, 10.0, "a"),
        _selection_candidate("n2", opened, 9.0, "b"),
        _selection_candidate("n3", inaugurated, 8.0, "c"),
        _selection_candidate("n4", shelters, 7.0, "d"),
    )
    similarity = _Similarity({frozenset((opened, inaugurated)): 0.96})

    selection = select_subnarrative_candidates(
        context,
        candidates,
        similarity,
        SelectionPolicy(budgets=(1, 2), semantic_threshold=0.92, mmr_lambda=0.7),
    )

    assert selection.candidate_count == 4
    assert selection.exact_group_count == 3
    assert selection.semantic_cluster_count == 2
    assert [len(cluster.members) for cluster in selection.clusters] == [3, 1]
    assert selection.snapshots[0].cluster_ids == (selection.clusters[0].cluster_id,)
    assert selection.snapshots[1].cluster_ids == tuple(cluster.cluster_id for cluster in selection.clusters)


def test_selection_artifact_matches_fixed_golden_digest(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    text = "Exact coastal evidence."
    text_digest = _digest(text)
    subnarrative = "Documented coastal safety measures"
    subnarrative_digest = _digest(subnarrative)
    candidate = {
        "schema_version": "extractive_candidate_nugget_v1",
        "topic_id": "224",
        "docid": "doc-a",
        "subnarrative_id": "subnarrative-1",
        "candidate_nugget_id": "n1",
        "nugget_type": "extractive",
        "candidate_kind": "exact_sentence",
        "text": text,
        "evidence_sentences": [{
            "text": text, "start_char": 0, "end_char": 23,
            "start_byte": 0, "end_byte": 23, "text_sha256": text_digest,
            "cross_encoder_score": 1.5,
        }],
        "matched_paragraph": {
            "text": text, "start_char": 0, "end_char": 23,
            "start_byte": 0, "end_byte": 23, "text_sha256": text_digest,
        },
        "context_before": None,
        "context_after": None,
        "passages": [{
            "passage_id": "p1", "lane_id": "original", "query_id": "topic-224",
            "scoring_start_char": 0, "scoring_end_char": 23,
            "source_start_char": 0, "source_end_char": 23,
            "source_start_byte": 0, "source_end_byte": 23,
            "source_text": text, "source_text_sha256": text_digest,
            "scoring_text_sha256": text_digest, "chunk_text_sha256": text_digest,
            "normalization_version": "trec_rag_whitespace_v1",
            "cross_encoder_score": 2.0, "cross_encoder_rank": 1,
        }],
        "sentence_cross_encoder_score": 1.5,
        "rank_within_document_subnarrative": 1,
        "document_sha256": text_digest,
        "scoring_text_sha256": text_digest,
        "subnarrative_sha256": subnarrative_digest,
        "sentence_splitter_version": "exact_rules_v1",
    }
    candidates_path = canonical_root / "candidates.jsonl"
    candidate_bytes = (
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    candidates_path.write_bytes(candidate_bytes)
    candidate_manifest_path = canonical_root / "candidate-manifest.json"
    candidate_manifest_path.write_text(json.dumps({
        "schema_version": "extractive_candidate_manifest_v1",
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "source_sha256": "1" * 64,
        "scorer": {
            "model": "fake-local-mixedbread", "model_revision": "pin",
            "backend_version": "test", "score_representation": "raw_logits",
            "inference_dtype": "float32", "score_kind": "extractive_sentence_v1",
            "sentence_max_length": 512,
        },
        "sentence_splitter_version": "exact_rules_v1",
        "scoring_normalization_version": "trec_rag_whitespace_v1",
        "source_file": "requests.jsonl",
        "candidate_file": "candidates.jsonl",
        "input_sha256": "1" * 64,
        "candidates_sha256": _digest(candidate_bytes),
        "output_sha256": _digest(candidate_bytes),
        "document_count": 1,
        "unique_document_count": 1,
        "unique_subnarrative_count": 1,
        "failure_count": 0,
        "candidate_count": 1,
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }, sort_keys=True, separators=(",", ":")) + "\n")
    narrative = "Explain documented coastal safety measures."
    contexts_path = canonical_root / "contexts.jsonl"
    contexts_path.write_text(json.dumps({
        "schema_version": "subnarrative_selection_context_v1",
        "topic_id": "224",
        "official_narrative": narrative,
        "official_narrative_sha256": _digest(narrative),
        "subnarrative_id": "subnarrative-1",
        "subnarrative_text": subnarrative,
        "subnarrative_sha256": subnarrative_digest,
    }, sort_keys=True, separators=(",", ":")) + "\n")

    artifacts = select_evidence_artifacts(
        CandidateArtifacts(candidates_path, candidate_manifest_path),
        contexts_path,
        device="cpu",
        similarity=_Similarity({}),
    )

    assert _digest(artifacts.selections_path.read_bytes()) == (
        "02a0b34a1a0d78a0d465ea00c012ad1fc4fb0ef77549fbab9519994818ffb99f"
    )


def test_nonfinite_similarity_fails_closed() -> None:
    context = SubnarrativeContext("224", "Explain coastal safety.", "sub-1", "Coastal safety")
    candidates = (
        _selection_candidate("n1", "First evidence.", 2.0, "a"),
        _selection_candidate("n2", "Second evidence.", 1.0, "b"),
    )
    with pytest.raises(ValueError, match="cosine matrix value must be finite"):
        select_subnarrative_candidates(
            context,
            candidates,
            _Similarity({frozenset(("First evidence.", "Second evidence.")): float("inf")}),
            SelectionPolicy(budgets=(2,)),
        )


def _topic() -> Topic:
    return Topic(
        id="housing-1",
        title="must never be downstream input",
        narrative="Explain how rent increases affect housing tenants.",
    )


def _decomposition(topic: Topic, *, fallback: bool = False) -> ValidatedDecomposition:
    if fallback:
        result = FacetPlanningResult(
            queries=(QueryVariant(topic.id, "original", topic.narrative, "original_topic"),),
            used_fallback=True,
            error="invalid generated plan",
        )
    else:
        result = plan_facet_queries(topic, {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": [{
                "subnarrative": "How rent increases affect housing tenants",
                "bm25_queries": ["rent increases housing tenant impacts"],
            }],
        })
    return ValidatedDecomposition(topic.id, _digest(topic.narrative), "d" * 64, result)


def _write_scoring_checkpoint(
    root: Path,
    topic: Topic,
    decomposition: ValidatedDecomposition,
    *,
    fallback: bool,
) -> None:
    scoring = root / topic.id / "scoring"
    scoring.mkdir(parents=True)
    source = "Lead.\n\n  Evidence\t sentence about cafés and rents.  \nTail."
    selected = {
        "topic_id": topic.id, "docid": "doc-1", "selection_rank": 1,
        "selected_from_lane": "original", "selected_from_lane_rank": 1,
        "text_sha256": _digest(source), "text": source,
    }
    (scoring / "selected_documents.jsonl").write_text(
        json.dumps(selected, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (scoring / "lane_scores.jsonl").write_text("")
    (scoring / "selection.json").write_text("{}\n")
    if fallback:
        cross_body = ""
    else:
        sub = decomposition.result.subnarratives[0]
        start = source.index("Evidence") - 2
        end = source.index(".", start) + 3
        cross = {
            "topic_id": topic.id, "docid": "doc-1", "selection_rank": 1,
            "subnarrative_id": sub.subnarrative_id,
            "lane_name": f"subnarrative:{sub.subnarrative_id}",
            "semantic_query_sha256": sub.semantic_query_sha256,
            "bm25_queries": list(sub.bm25_queries),
            "bm25_query_sha256s": list(sub.bm25_query_sha256s),
            "bm25_rank": 1, "bm25_score": 4.5, "aggregate_rank": 1,
            "aggregate_score": 7.25, "long_document_raw_logit": 6.0,
            "weighted_passage_raw_logit": 7.25, "within_document_span_support": 1,
            "score_representation": "raw_logits", "text_sha256": _digest(source),
            "winning_passages": [{
                "chunk_index": 0, "start_char": start, "end_char": end,
                "raw_logit": 7.25, "weighted_rank": 1,
            }],
            "downstream_only": True,
        }
        cross_body = json.dumps(cross, sort_keys=True, separators=(",", ":")) + "\n"
    (scoring / "selected_subnarrative_scores.jsonl").write_text(cross_body)
    relative_paths = (
        "scoring/lane_scores.jsonl", "scoring/selected_documents.jsonl",
        "scoring/selection.json", "scoring/selected_subnarrative_scores.jsonl",
    )
    topic_root = root / topic.id
    artifacts = []
    for relative in relative_paths:
        body = (topic_root / relative).read_bytes()
        artifacts.append({"relative_path": relative, "bytes": len(body), "sha256": _digest(body)})
    manifest = {
        "schema_version": "facet_pilot_v2", "selection_schema_version": "facet_pilot_selection_v2",
        "phase": "score", "topic_id": topic.id,
        "narrative_sha256": decomposition.narrative_sha256,
        "decomposition_source_sha256": decomposition.source_sha256,
        "code_commit": "a" * 40, "retriever": {"name": "fake"},
        "scorer": {"model": "fake"}, "rerank_depth": 100, "selection_k": 100,
        "selection_policy": "round_robin_lane_order_no_fusion", "score_policy": {},
        "retrieval_manifest_sha256": "b" * 64, "selected_set_sha256": _digest('["doc-1"]'),
        "artifacts": artifacts,
    }
    (scoring / "complete.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def test_resigned_semantic_tamper_is_rejected(tmp_path: Path) -> None:
    topic = _topic()
    decomposition = _decomposition(topic)
    pilot_root = tmp_path / "pilot"
    _write_scoring_checkpoint(pilot_root, topic, decomposition, fallback=False)
    score_path = pilot_root / topic.id / "scoring" / "selected_subnarrative_scores.jsonl"
    score_path.write_bytes(b"")
    manifest_path = pilot_root / topic.id / "scoring" / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = next(row for row in manifest["artifacts"] if row["relative_path"].endswith("selected_subnarrative_scores.jsonl"))
    receipt.update({"bytes": 0, "sha256": _digest(b"")})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="document-by-subnarrative"):
        materialize_candidate_inputs(
            topic, decomposition, pilot_root=pilot_root, output_dir=tmp_path / "downstream",
            code_commit="a" * 40, official_topics_sha256="c" * 64,
        )


def test_original_only_pipeline_is_empty_and_never_constructs_local_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _topic()
    decomposition = _decomposition(topic, fallback=True)
    pilot_root = tmp_path / "pilot"
    _write_scoring_checkpoint(pilot_root, topic, decomposition, fallback=True)
    handoff = materialize_candidate_inputs(
        topic, decomposition, pilot_root=pilot_root, output_dir=tmp_path / "canonical" / "handoff",
        code_commit="a" * 40, official_topics_sha256="c" * 64,
    )
    monkeypatch.setattr(
        "trec_rag.evidence_store.MixedbreadSentencePairScorer",
        lambda **kwargs: pytest.fail("empty input must not construct the sentence scorer"),
    )
    monkeypatch.setattr(
        "trec_rag.evidence_store.LocalMiniLMSimilarity",
        lambda **kwargs: pytest.fail("empty input must not construct the similarity model"),
    )

    candidates = generate_candidate_artifacts(
        handoff, score_cache_root=tmp_path / "score-cache", device="cpu"
    )
    selections = select_evidence_artifacts(candidates, handoff.contexts_path, device="cpu")

    assert handoff.requests_path.read_bytes() == b""
    assert handoff.contexts_path.read_bytes() == b""
    assert candidates.candidates_path.read_bytes() == b""
    assert selections.selections_path.read_bytes() == b""
    candidate_manifest = json.loads(candidates.manifest_path.read_bytes())
    selection_manifest = json.loads(selections.manifest_path.read_bytes())
    assert candidate_manifest["retrieval_network_calls"] == 0
    assert candidate_manifest["hosted_llm_calls"] == 0
    assert selection_manifest["retrieval_network_calls"] == 0
    assert selection_manifest["hosted_llm_calls"] == 0


def test_candidate_validation_streams_when_only_selected_candidates_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _topic()
    decomposition = _decomposition(topic, fallback=True)
    pilot_root = tmp_path / "pilot"
    _write_scoring_checkpoint(pilot_root, topic, decomposition, fallback=True)
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=pilot_root,
        output_dir=tmp_path / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
    )
    artifacts = generate_candidate_artifacts(
        handoff, score_cache_root=tmp_path / "score-cache", device="cpu"
    )
    original_read_bytes = Path.read_bytes

    def reject_candidate_buffering(path: Path) -> bytes:
        if path == artifacts.candidates_path:
            raise AssertionError("candidate validation must stream the JSONL ledger")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_candidate_buffering)

    assert load_validated_candidate_artifacts(
        artifacts.candidates_path,
        artifacts.manifest_path,
        documents={},
        subnarratives={},
        required_candidate_keys=frozenset(),
    ) == {}


def test_candidate_validation_rejects_semantic_tamper_in_an_unselected_row(
    tmp_path: Path,
) -> None:
    topic = _topic()
    decomposition = _decomposition(topic)
    pilot_root = tmp_path / "pilot"
    _write_scoring_checkpoint(pilot_root, topic, decomposition, fallback=False)
    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=pilot_root,
        output_dir=tmp_path / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
    )
    artifacts = generate_candidate_artifacts(
        handoff,
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        scorer=_FixedScorer(lambda pairs: tuple(1.0 for _ in pairs)),
    )
    rows = artifacts.candidates_path.read_bytes().splitlines()
    tampered = json.loads(rows[0])
    tampered["document_sha256"] = "0" * 64
    rows[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
    candidate_bytes = b"\n".join(rows) + b"\n"
    artifacts.candidates_path.write_bytes(candidate_bytes)
    manifest = json.loads(artifacts.manifest_path.read_bytes())
    manifest["candidates_sha256"] = _digest(candidate_bytes)
    manifest["output_sha256"] = _digest(candidate_bytes)
    artifacts.manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    request = json.loads(handoff.requests_path.read_bytes().splitlines()[0])

    with pytest.raises(ValueError, match="document_sha256"):
        load_validated_candidate_artifacts(
            artifacts.candidates_path,
            artifacts.manifest_path,
            documents={request["document_id"]: request["source"]},
            subnarratives={
                row["subnarrative_id"]: row["text"]
                for row in request["subnarratives"]
            },
            required_candidate_keys=frozenset(),
        )

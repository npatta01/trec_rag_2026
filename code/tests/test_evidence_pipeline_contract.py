from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

import trec_rag.evidence_store as evidence_store_module
from trec_rag.document_store import DocumentStore
from trec_rag.facet_evidence import (
    SENTENCE_SPLITTER_VERSION,
    CandidateSubnarrative,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SelectionCandidate,
    SelectionPolicy,
    SubnarrativeContext,
    _byte_offsets,
    _scoring_text_and_boundaries,
    extract_document_candidates,
    select_subnarrative_candidates,
)
from trec_rag.evidence_store import (
    CandidateArtifacts,
    HandoffArtifacts,
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
from trec_rag.topic_records import (
    FacetRecord,
    TopicRecordsBuilder,
    TopicRecordsIntegrityError,
)
from trec_rag.topic_passage_search import (
    FocusedQuery,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
)


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
        "input_policy": "trec_rag_whitespace_v1",
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


def _candidate_stage_scorer_identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "model": "literal-local-scorer",
        "model_revision": "test-v1",
        "backend_version": "test-v1",
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "score_kind": "extractive_sentence_v1",
        "sentence_max_length": 512,
        "input_policy": "trec_rag_whitespace_v1",
    }
    identity.update(overrides)
    return identity


def _candidate_stage_identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "source_file": "candidate-requests.jsonl",
        "source_sha256": "a" * 64,
        "scorer": _candidate_stage_scorer_identity(),
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": "trec_rag_whitespace_v1",
    }
    identity.update(overrides)
    return identity


def _without_field(value: dict[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    del result[field]
    return result


def test_candidate_artifacts_session_is_worker_local_and_not_part_of_value_identity(
    tmp_path: Path,
) -> None:
    paths = CandidateArtifacts(
        tmp_path / "records.sqlite3",
        tmp_path / "canonical" / "records-manifest.json",
        tmp_path / "objects",
    )
    with_session = replace(paths, validation_session=object())

    assert with_session == paths
    assert "validation_session" not in repr(with_session)


def test_generate_candidate_artifacts_returns_exact_published_validation_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "topic-224" / "canonical" / "handoff"
    handoff_root.mkdir(parents=True)
    requests_path = handoff_root / "candidate-requests.jsonl"
    contexts_path = handoff_root / "selection-contexts.jsonl"
    manifest_path = handoff_root / "handoff-manifest.json"
    requests_path.write_bytes(b"")
    contexts_path.write_bytes(b"")
    manifest_path.write_bytes(
        b'{"request_schema_version":"extractive_candidate_request_v1",'
        b'"topic_id":"224","requests_file":"candidate-requests.jsonl",'
        b'"requests_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4'
        b'649b934ca495991b7852b855"}\n'
    )
    handoff = HandoffArtifacts(
        requests_path,
        contexts_path,
        manifest_path,
        "a" * 64,
        False,
    )
    validation_session = object()
    published = SimpleNamespace(validation_session=validation_session)

    class BuilderSpy:
        def __init__(self, destination, topic_id, document_store, *, run_id):
            assert run_id == "test-run"
            self.published_identity = None

        def publish(self, identity):
            self.published_identity = identity
            return published

        def add_facets(self, facets):
            assert facets == ()

        def add_passage_search(self, result):
            raise AssertionError("empty fixture must not add passage search rows")

        def set_completion(self, status, stopping_reason):
            assert (status, stopping_reason) == ("incomplete", "no_evidence")

        def _cleanup(self):
            raise AssertionError("successful publication must not be cleaned up")

    monkeypatch.setattr(evidence_store_module, "TopicRecordsBuilder", BuilderSpy)

    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=tmp_path / "objects",
    )

    assert artifacts.validation_session is validation_session


class _ValidationSessionRecordsSpy:
    def __init__(self) -> None:
        self._manifest = {
            "identity_json": json.dumps(
                _candidate_stage_identity(), sort_keys=True, separators=(",", ":")
            )
        }
        self.receipt = SimpleNamespace(
            database_sha256="a" * 64,
            semantic_sha256="b" * 64,
            row_counts={"candidate": 0},
        )
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exited = True

    def load_candidates(self, required_keys):
        return {}


def _candidate_artifacts_for_open_spy(
    tmp_path: Path,
    validation_session: object,
) -> tuple[CandidateArtifacts, Path]:
    canonical_root = tmp_path / "topic-224" / "canonical"
    canonical_root.mkdir(parents=True)
    records_path = canonical_root.parent / "records.sqlite3"
    manifest_path = canonical_root / "records-manifest.json"
    manifest_path.write_bytes(b'{"topic_id":"224"}\n')
    contexts_path = canonical_root / "contexts.jsonl"
    contexts_path.write_bytes(b"")
    return (
        CandidateArtifacts(
            records_path,
            manifest_path,
            tmp_path / "objects",
            validation_session,
        ),
        contexts_path,
    )


def test_load_validated_candidate_artifacts_passes_validation_session_to_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_session = object()
    artifacts, _ = _candidate_artifacts_for_open_spy(tmp_path, validation_session)
    opened: list[object] = []
    records = _ValidationSessionRecordsSpy()

    class TopicRecordsSpy:
        @staticmethod
        def open(*args, **kwargs):
            opened.append(kwargs["validation_session"])
            return records

    monkeypatch.setattr(evidence_store_module, "TopicRecords", TopicRecordsSpy)

    assert load_validated_candidate_artifacts(
        artifacts,
        expected_topic_id="224",
        required_candidate_keys=frozenset(),
    ) == {}
    assert opened == [validation_session]
    assert records.exited


def test_select_evidence_artifacts_passes_validation_session_to_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_session = object()
    artifacts, contexts_path = _candidate_artifacts_for_open_spy(
        tmp_path, validation_session
    )
    opened: list[object] = []
    records = _ValidationSessionRecordsSpy()

    class TopicRecordsSpy:
        @staticmethod
        def open(*args, **kwargs):
            opened.append(kwargs["validation_session"])
            return records

    monkeypatch.setattr(evidence_store_module, "TopicRecords", TopicRecordsSpy)

    select_evidence_artifacts(artifacts, contexts_path, device="cpu")

    assert opened == [validation_session]
    assert records.exited


def _publish_candidate_stage_fixture(
    tmp_path: Path,
    identity: dict[str, object],
) -> tuple[CandidateArtifacts, Path]:
    topic_root = tmp_path / "topic-224"
    document_store_root = tmp_path / "objects"
    builder = TopicRecordsBuilder(
        topic_root,
        "224",
        DocumentStore(document_store_root),
        run_id="test-run",
    )
    builder.publish(identity)
    contexts_path = topic_root / "canonical" / "contexts.jsonl"
    contexts_path.write_bytes(b"")
    return (
        CandidateArtifacts(
            topic_root / "records.sqlite3",
            topic_root / "canonical" / "records-manifest.json",
            document_store_root,
        ),
        contexts_path,
    )


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


def test_a_pair_never_spans_a_sentence_no_passage_covers() -> None:
    """A pair quotes from its first span's start to its last span's end.

    Sentences with no covering passage used to be filtered out before pairing,
    which left non-adjacent sentences adjacent in the list. The pair built
    across that gap quoted a middle sentence no retrieved passage supports.
    """
    source = (
        "Alpha reported a clear improvement. "
        "Bravo measured nothing unusual at all. "
        "This result confirms the safety profile."
    )
    digest = _digest(source)
    scoring_text = _scoring_text(source)
    third_start = scoring_text.index("This")
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-a",
        source=source,
        document_sha256=digest,
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        # One passage over the first sentence, one over the third. Nothing
        # covers the middle sentence.
        passages=(
            ScoredPassage(
                "p1", "original", "topic-224", 0, scoring_text.index("Bravo") - 1,
                _digest(scoring_text), _digest(scoring_text[: scoring_text.index("Bravo") - 1]), 1.0, 1,
            ),
            ScoredPassage(
                "p3", "original", "topic-224", third_start, len(scoring_text),
                _digest(scoring_text), _digest(scoring_text[third_start:]), 0.9, 2,
            ),
        ),
    )

    candidates = extract_document_candidates(
        request, _FixedScorer(lambda pairs: tuple(1.0 for _ in pairs))
    )

    # "This result..." carries a dependency cue, so before the fix it paired
    # with "Alpha reported..." across the uncovered middle sentence.
    assert [(row.candidate_kind, row.text) for row in candidates] == [
        ("exact_sentence", "Alpha reported a clear improvement."),
        ("exact_sentence", "This result confirms the safety profile."),
    ]
    assert not any("Bravo" in row.text for row in candidates)


def test_a_heading_reaches_selection_joined_to_the_sentence_it_introduces() -> None:
    """A heading is not evidence alone, and must not be silently discarded.

    It has to survive both pair validators too: the builder and the validators
    once disagreed about which pairs were admissible, and a pair the builder
    emitted but a validator rejected aborted the entire candidate stage.
    """
    source = "Grocery Costs:\n\nGroceries are reasonably priced here. Imports cost more."
    digest = _digest(source)
    scoring_text = _scoring_text(source)
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-a",
        source=source,
        document_sha256=digest,
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Grocery evidence"),),
        passages=(ScoredPassage(
            "p1", "original", "topic-224", 0, len(scoring_text),
            _digest(scoring_text), _digest(scoring_text), 1.0, 1,
        ),),
    )

    candidates = extract_document_candidates(
        request, _FixedScorer(lambda pairs: tuple(1.0 for _ in pairs))
    )

    # Never admitted alone...
    assert "Grocery Costs:" not in [row.text for row in candidates]
    # ...but reaches selection as part of a full thought.
    assert [row.text for row in candidates if row.candidate_kind == "exact_sentence_pair"] == [
        "Grocery Costs:\n\nGroceries are reasonably priced here."
    ]


def test_a_dependency_cue_still_pairs_when_both_sentences_are_covered() -> None:
    """Positive control for the pairing guard above: real pairs still form."""
    source = "Alpha reported a clear improvement. This result confirms the safety profile."
    digest = _digest(source)
    scoring_text = _scoring_text(source)
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-a",
        source=source,
        document_sha256=digest,
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(ScoredPassage(
            "p1", "original", "topic-224", 0, len(scoring_text),
            _digest(scoring_text), _digest(scoring_text), 1.0, 1,
        ),),
    )

    candidates = extract_document_candidates(
        request, _FixedScorer(lambda pairs: tuple(1.0 for _ in pairs))
    )

    assert [row.text for row in candidates if row.candidate_kind == "exact_sentence_pair"] == [
        "Alpha reported a clear improvement. This result confirms the safety profile."
    ]


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
        b'{"candidate_kind":"exact_sentence","candidate_nugget_id":"ecn1_d493c7728fc4a51bd71774e2274b50d1541ae595ea92adcadde3eaf7aa1317b1","context_after":null,"context_before":null,"docid":"doc-a","document_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","evidence_sentences":[{"cross_encoder_score":1.25,"end_byte":18,"end_char":18,"start_byte":0,"start_char":0,"text":"Evidence sentence.","text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"}],"matched_paragraph":{"end_byte":18,"end_char":18,"start_byte":0,"start_char":0,"text":"Evidence sentence.","text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"},"nugget_type":"extractive","passages":[{"chunk_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","cross_encoder_rank":1,"cross_encoder_score":1.0,"lane_id":"original","normalization_version":"trec_rag_whitespace_v1","passage_id":"p1","query_id":"topic-224","scoring_end_char":18,"scoring_start_char":0,"scoring_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","source_end_byte":18,"source_end_char":18,"source_start_byte":0,"source_start_char":0,"source_text":"Evidence sentence.","source_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720"}],"rank_within_document_subnarrative":1,"schema_version":"extractive_candidate_nugget_v1","scoring_text_sha256":"2f9d0ed51668da4cb7bac73526265ba58f58db77ed2e4eba87128c4ef5c7c720","sentence_cross_encoder_score":1.25,"sentence_splitter_version":"spacy_en_core_web_sm_3.8.0_v1","subnarrative_id":"safety","subnarrative_sha256":"2e194dbac187930e8b8016c241be054aaeb7e4344f8676edbefa0ddadb065db0","text":"Evidence sentence.","topic_id":"224"}\n'
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
    text = "Exact coastal evidence."
    text_digest = _digest(text)
    subnarrative = "Documented coastal safety measures"
    subnarrative_digest = _digest(subnarrative)
    topic_root = tmp_path / "topic-224"
    canonical_root = topic_root / "canonical"
    canonical_root.mkdir(parents=True)
    request = ExtractiveCandidateRequest(
        topic_id="224",
        document_id="doc-a",
        source=text,
        document_sha256=text_digest,
        scoring_text_sha256=text_digest,
        subnarratives=(CandidateSubnarrative("subnarrative-1", subnarrative),),
        passages=(ScoredPassage(
            "p1", "original", "topic-224", 0, len(text), text_digest,
            text_digest, 2.0, 1,
        ),),
    )
    candidate = extract_document_candidates(request, _FixedScorer((1.5,)))[0]
    document_store_root = tmp_path / "objects"
    builder = TopicRecordsBuilder(
        topic_root, "224", DocumentStore(document_store_root), run_id="test-run"
    )
    builder.bind_document("doc-a", text, expected_sha256=text_digest)
    builder.add_candidate(candidate)
    builder.publish({
        "request_schema_version": "extractive_candidate_request_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "source_file": "candidate-requests.jsonl",
        "source_sha256": text_digest,
        "scorer": dict(_LiteralScorer.identity),
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
        "scoring_normalization_version": "trec_rag_whitespace_v1",
    })
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
        CandidateArtifacts(
            topic_root / "records.sqlite3",
            canonical_root / "records-manifest.json",
            document_store_root,
        ),
        contexts_path,
        device="cpu",
        similarity=_Similarity({}),
    )

    selection_manifest = json.loads(artifacts.manifest_path.read_bytes())
    records_manifest = json.loads(
        (topic_root / "canonical" / "records-manifest.json").read_bytes()
    )
    assert selection_manifest["records_file"] == "records.sqlite3"
    assert selection_manifest["records_manifest_file"] == "records-manifest.json"
    assert selection_manifest["records_schema_version"] == "topic-records-v4"
    assert selection_manifest["records_stage"] == "canonical-candidates-v1"
    assert selection_manifest["records_database_sha256"] == _digest(
        (topic_root / "records.sqlite3").read_bytes()
    )
    assert selection_manifest["candidate_semantic_sha256"] == records_manifest[
        "semantic_sha256"
    ]
    assert not {
        "candidates_file",
        "candidate_manifest_file",
        "candidates_sha256",
        "candidate_manifest_sha256",
    } & selection_manifest.keys()
    assert _digest(artifacts.selections_path.read_bytes()) == (
        "9beb32dade0e1c3baa439056401e7bc7569453ef72d8ea8a56f2724429414f2e"
    )


@pytest.mark.parametrize("consumer", ("selection", "loading"))
@pytest.mark.parametrize(
    "identity",
    (
        pytest.param(
            _candidate_stage_identity(request_schema_version="request-v0"),
            id="request-schema",
        ),
        pytest.param(
            _candidate_stage_identity(candidate_schema_version="candidate-v0"),
            id="candidate-schema",
        ),
        pytest.param(
            _candidate_stage_identity(sentence_splitter_version="splitter-v0"),
            id="sentence-splitter",
        ),
        pytest.param(
            _candidate_stage_identity(
                scoring_normalization_version="normalization-v0"
            ),
            id="scoring-normalization",
        ),
        pytest.param(
            _candidate_stage_identity(
                source_file="handoff/candidate-requests.jsonl"
            ),
            id="source-basename",
        ),
        pytest.param(
            _candidate_stage_identity(source_sha256="A" * 64),
            id="source-digest",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_candidate_stage_scorer_identity(
                    score_representation="probabilities"
                )
            ),
            id="scorer-representation",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_candidate_stage_scorer_identity(
                    score_kind="extractive_sentence_v0"
                )
            ),
            id="scorer-kind",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_candidate_stage_scorer_identity(model="")
            ),
            id="scorer-empty-string",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_candidate_stage_scorer_identity(sentence_max_length=0)
            ),
            id="scorer-max-length",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_without_field(
                    _candidate_stage_scorer_identity(), "model_revision"
                )
            ),
            id="scorer-missing-field",
        ),
        pytest.param(
            _candidate_stage_identity(
                scorer=_candidate_stage_scorer_identity(extra="unexpected")
            ),
            id="scorer-extra-field",
        ),
        pytest.param(
            _without_field(_candidate_stage_identity(), "source_file"),
            id="identity-missing-field",
        ),
        pytest.param(
            _candidate_stage_identity(extra="unexpected"),
            id="identity-extra-field",
        ),
    ),
)
def test_candidate_stage_identity_fails_closed_for_both_consumers(
    tmp_path: Path,
    identity: dict[str, object],
    consumer: str,
) -> None:
    artifacts, contexts_path = _publish_candidate_stage_fixture(tmp_path, identity)

    with pytest.raises(ValueError, match="candidate stage identity"):
        if consumer == "selection":
            select_evidence_artifacts(artifacts, contexts_path, device="cpu")
        else:
            load_validated_candidate_artifacts(
                artifacts,
                expected_topic_id="224",
                required_candidate_keys=frozenset(),
            )

    assert not (artifacts.manifest_path.parent / "subnarrative-selections.jsonl").exists()
    assert not (artifacts.manifest_path.parent / "selection-manifest.json").exists()


@pytest.mark.parametrize("layout", ("renamed", "different-topic-roots"))
def test_selection_rejects_copied_noncanonical_candidate_artifact_paths(
    tmp_path: Path,
    layout: str,
) -> None:
    artifacts, contexts_path = _publish_candidate_stage_fixture(
        tmp_path, _candidate_stage_identity()
    )
    if layout == "renamed":
        copied_records = artifacts.records_path.with_name("records-copy.sqlite3")
        copied_manifest = artifacts.manifest_path.with_name(
            "records-manifest-copy.json"
        )
    else:
        copied_records = tmp_path / "records-copy" / "records.sqlite3"
        copied_manifest = (
            tmp_path
            / "manifest-copy"
            / "canonical"
            / "records-manifest.json"
        )
    copied_records.parent.mkdir(parents=True, exist_ok=True)
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    copied_records.write_bytes(artifacts.records_path.read_bytes())
    copied_manifest.write_bytes(artifacts.manifest_path.read_bytes())
    copied = CandidateArtifacts(
        copied_records,
        copied_manifest,
        artifacts.document_store_root,
    )

    with pytest.raises(ValueError, match="canonical candidate artifact paths"):
        select_evidence_artifacts(copied, contexts_path, device="cpu")

    assert not (copied_manifest.parent / "subnarrative-selections.jsonl").exists()
    assert not (copied_manifest.parent / "selection-manifest.json").exists()


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
        "selection_policy": "round_robin_lane_order_no_fusion",
        "selection_scope": "internal_fixed_path_projection_not_final_submission",
        "score_policy": {},
        "passage_search": {},
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
        topic,
        decomposition,
        pilot_root=pilot_root,
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
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
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=tmp_path / "objects",
    )
    selections = select_evidence_artifacts(candidates, handoff.contexts_path, device="cpu")

    assert handoff.requests_path.read_bytes() == b""
    assert handoff.contexts_path.read_bytes() == b""
    assert candidates.records_path.is_file()
    assert candidates.manifest_path.is_file()
    assert candidates.document_store_root == tmp_path / "objects"
    assert candidates.records_path == tmp_path / topic.id / "records.sqlite3"
    assert candidates.manifest_path == (
        tmp_path / topic.id / "canonical" / "records-manifest.json"
    )
    assert not (candidates.records_path.parent / "canonical" / "candidates.jsonl").exists()
    assert not (candidates.records_path.parent / "canonical" / "candidate-manifest.json").exists()
    assert selections.selections_path.read_bytes() == b""
    records_manifest = json.loads(candidates.manifest_path.read_bytes())
    selection_manifest = json.loads(selections.manifest_path.read_bytes())
    assert records_manifest["row_counts"]["candidate"] == 0
    assert selection_manifest["records_file"] == "records.sqlite3"
    assert selection_manifest["records_manifest_file"] == "records-manifest.json"
    assert selection_manifest["records_schema_version"] == "topic-records-v4"
    assert selection_manifest["records_stage"] == "canonical-candidates-v1"
    assert selection_manifest["retrieval_network_calls"] == 0
    assert selection_manifest["hosted_llm_calls"] == 0


def test_official_passage_path_preserves_query_provenance_into_candidate_links(
    tmp_path: Path,
) -> None:
    topic = _topic()
    decomposition = _decomposition(topic)
    pilot_root = tmp_path / "pilot"
    _write_scoring_checkpoint(pilot_root, topic, decomposition, fallback=False)

    source = "Lead.\n\n  Evidence\t sentence about cafés and rents.  \nTail."
    document_sha256 = _digest(source)
    passage_start = source.index("Evidence")
    passage_end = source.index(".", passage_start) + 1
    passage_text = source[passage_start:passage_end]
    passage_id = "official-passage"
    query_id = "official-query-1"
    subnarrative = decomposition.result.subnarratives[0]
    passage_result = PassageSearchResult(
        FocusedQuery(query_id, subnarrative.text, subnarrative.subnarrative_id),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (SourceDocument("doc-1", document_sha256, 1, 4.5, passage_id, 7.25),),
        (SourcePassage(
            passage_id,
            "doc-1",
            document_sha256,
            1,
            4.5,
            passage_start,
            passage_end,
            len(source[:passage_start].encode()),
            len(source[:passage_end].encode()),
            _digest(passage_text),
            passage_text,
            7.25,
            1,
            "cache-official",
            _digest(passage_text),
            {"backend": "test", "implementation": "official"},
        ),),
        1,
        False,
    )
    document_store_root = tmp_path / "objects"
    DocumentStore(document_store_root).admit_text(
        source,
        expected_sha256=document_sha256,
    )

    handoff = materialize_candidate_inputs(
        topic,
        decomposition,
        pilot_root=pilot_root,
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
        passage_results=(passage_result,),
        document_store_root=document_store_root,
    )
    request = json.loads(handoff.requests_path.read_bytes())
    assert request["passages"][0]["query_id"] == query_id

    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        document_store_root=document_store_root,
        scorer=_FixedScorer(lambda pairs: tuple(1.0 for _ in pairs)),
        facets=(FacetRecord(subnarrative.subnarrative_id, subnarrative.text, "initial"),),
        passage_results=(passage_result,),
    )

    connection = sqlite3.connect(artifacts.records_path)
    try:
        assert tuple(connection.execute(
            "SELECT query_id FROM query_identity ORDER BY query_id"
        )) == ((query_id,),)
        assert tuple(connection.execute(
            "SELECT l.query_id, qp.raw_logit, qp.passage_rank "
            "FROM candidate_passage_link AS l "
            "JOIN query_passage AS qp "
            "ON qp.query_id=l.query_id AND qp.passage_pk=l.passage_pk"
        )) == ((query_id, 7.25, 1),)
    finally:
        connection.close()


def test_candidate_validation_opens_records_and_validates_all_sources(
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
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
    )
    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        scorer=_FixedScorer(lambda pairs: tuple(1.0 for _ in pairs)),
        document_store_root=tmp_path / "objects",
    )
    records_manifest = json.loads(artifacts.manifest_path.read_bytes())
    stage_identity = json.loads(records_manifest["identity_json"])
    assert stage_identity["source_sha256"] == _digest(
        handoff.requests_path.read_bytes()
    )
    assert stage_identity["source_file"] == handoff.requests_path.name
    assert stage_identity["scorer"] == _LiteralScorer.identity
    assert stage_identity["sentence_splitter_version"] == SENTENCE_SPLITTER_VERSION
    assert stage_identity["scoring_normalization_version"] == (
        "trec_rag_whitespace_v1"
    )
    assert load_validated_candidate_artifacts(
        artifacts,
        expected_topic_id=topic.id,
        required_candidate_keys=frozenset(),
    ) == {}


def test_candidate_generation_rejects_request_hash_tamper_before_publication(
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
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
    )
    rows = [
        json.loads(line)
        for line in handoff.requests_path.read_bytes().splitlines()
    ]
    rows[0]["passages"][0]["cross_encoder_score"] = 99.0
    handoff.requests_path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            for row in rows
        )
    )

    with pytest.raises(ValueError, match="request.*hash|sealed.*request"):
        generate_candidate_artifacts(
            handoff,
            run_id="test-run",
            score_cache_root=tmp_path / "score-cache",
            device="cpu",
            scorer=_FixedScorer(lambda pairs: tuple(1.0 for _ in pairs)),
            document_store_root=tmp_path / "objects",
        )

    assert not (tmp_path / topic.id / "records.sqlite3").exists()


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
        output_dir=tmp_path / topic.id / "canonical" / "handoff",
        code_commit="a" * 40,
        official_topics_sha256="c" * 64,
    )
    artifacts = generate_candidate_artifacts(
        handoff,
        run_id="test-run",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        scorer=_FixedScorer(lambda pairs: tuple(1.0 for _ in pairs)),
        document_store_root=tmp_path / "objects",
    )
    manifest = json.loads(artifacts.manifest_path.read_bytes())
    manifest["semantic_sha256"] = "0" * 64
    artifacts.manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(TopicRecordsIntegrityError, match="manifest bytes"):
        load_validated_candidate_artifacts(
            artifacts,
            expected_topic_id=topic.id,
            required_candidate_keys=frozenset(),
        )

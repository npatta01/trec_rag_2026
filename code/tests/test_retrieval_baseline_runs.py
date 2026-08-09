from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk
from trec_rag.document_store import DocumentStore
from trec_rag.mixedbread_passage_scorer import ScoredPassage
from trec_rag.retrieval_candidate_core import CandidateCore, CandidateLaneStat
from trec_rag.retrieval_baseline_runs import (
    BreadthPassage,
    DocumentScore,
    PassageScore,
    breadth_counts,
    build_rankings,
    cutoff_decision,
    export_runs,
    load_topic_input,
    main,
    midrank_percentiles,
    read_topic_matrix,
    rank_topic_matrix,
    score_topic,
    score_topics,
    select_eligible_documents,
    suppress_overlaps,
    topic_sort_key,
    weighted_passage_score,
    write_topic_matrix,
)


def test_midrank_percentiles_give_equal_values_equal_percentiles() -> None:
    assert midrank_percentiles({"low": 1.0, "tie-a": 2.0, "tie-b": 2.0, "high": 4.0}) == {
        "low": 0.0,
        "tie-a": 0.5,
        "tie-b": 0.5,
        "high": 1.0,
    }


def test_midrank_percentiles_map_a_single_value_to_one() -> None:
    assert midrank_percentiles({"only": -3.25}) == {"only": 1.0}


def test_overlap_coefficient_of_exactly_one_half_suppresses_lower_score() -> None:
    passages = (
        PassageScore(start_char=0, end_char=100, raw_score=9.0),
        PassageScore(start_char=50, end_char=150, raw_score=8.0),
        PassageScore(start_char=150, end_char=200, raw_score=7.0),
    )

    assert suppress_overlaps(passages) == (passages[0], passages[2])


def test_weighted_passage_score_renormalizes_the_available_top_four() -> None:
    passages = (
        PassageScore(start_char=0, end_char=10, raw_score=4.0),
        PassageScore(start_char=20, end_char=30, raw_score=2.0),
    )

    assert weighted_passage_score(passages) == pytest.approx(3.375)


def test_weighted_passage_score_is_python_version_independent() -> None:
    passages = (
        PassageScore(start_char=0, end_char=10, raw_score=4.9375),
        PassageScore(start_char=20, end_char=30, raw_score=4.625),
        PassageScore(start_char=40, end_char=50, raw_score=4.5625),
        PassageScore(start_char=60, end_char=70, raw_score=3.4375),
    )

    assert weighted_passage_score(passages).hex() == "0x1.2d28f5c28f5c2p+2"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_passage_score_rejects_nonfinite_model_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PassageScore(start_char=0, end_char=10, raw_score=value)


def _document(
    docid: str,
    rank: int,
    narrative: float,
    raw_narrative: float | None = None,
    raw_subnarratives: dict[str, float] | None = None,
    source_ranks: dict[str, int] | None = None,
    **subnarratives: float,
) -> DocumentScore:
    return DocumentScore(
        docid=docid,
        best_retrieval_rank=rank,
        narrative_raw_score=narrative if raw_narrative is None else raw_narrative,
        subnarrative_raw_scores=(
            subnarratives if raw_subnarratives is None else raw_subnarratives
        ),
        narrative_percentile=narrative,
        subnarrative_percentiles=subnarratives,
        subnarrative_source_ranks=(
            {name: rank for name in subnarratives}
            if source_ranks is None
            else source_ranks
        ),
    )


def test_positive_mad_cutoff_admits_only_values_at_or_above_robust_threshold() -> None:
    documents = (
        _document("d0", 1, 0.0, s=0.0),
        _document("d1", 2, 0.1, s=0.1),
        _document("d2", 3, 0.2, s=0.2),
        _document("d3", 4, 0.3, s=0.3),
        _document("d4", 5, 1.0, s=1.0),
    )

    assert select_eligible_documents(documents) == ("d4",)


def test_zero_mad_cutoff_uses_strictly_above_median() -> None:
    documents = (
        _document("a", 1, 0.5, s=0.5),
        _document("b", 2, 0.5, s=0.5),
        _document("c", 3, 0.5, s=0.5),
        _document("strong", 4, 0.9, s=0.9),
    )

    assert select_eligible_documents(documents) == ("strong",)


def test_cutoff_uses_per_unit_raw_scores_and_unions_the_admitted_documents() -> None:
    documents = (
        _document(
            "narrative-outlier",
            1,
            0.5,
            raw_narrative=10.0,
            raw_subnarratives={"s": 0.0},
            s=0.5,
        ),
        _document(
            "facet-outlier",
            2,
            0.5,
            raw_narrative=0.0,
            raw_subnarratives={"s": 10.0},
            s=0.5,
        ),
        _document("weak-a", 3, 0.5, raw_narrative=0.0, raw_subnarratives={"s": 0.0}, s=0.5),
        _document("weak-b", 4, 0.5, raw_narrative=0.0, raw_subnarratives={"s": 0.0}, s=0.5),
        _document("weak-c", 5, 0.5, raw_narrative=0.0, raw_subnarratives={"s": 0.0}, s=0.5),
    )

    decision = cutoff_decision(documents)

    assert decision.eligible_docids == (
        "facet-outlier",
        "narrative-outlier",
    )
    assert decision.pre_fallback_count == 2
    assert decision.fallback_used is False
    assert {row.unit_id: row.admitted_count for row in decision.units} == {
        "__narrative__": 1,
        "s": 1,
    }
    assert decision.admission_multiplicity_histogram == {"1": 2}


def test_empty_cutoff_falls_back_by_narrative_then_rank_then_bytewise_docid() -> None:
    documents = (
        _document("z", 1, 0.4, s=0.5),
        _document("b", 2, 0.5, s=0.5),
        _document("a", 2, 0.5, s=0.5),
    )

    assert select_eligible_documents(documents) == ("a",)


def test_one_subnarrative_facet_score_uses_its_full_weight() -> None:
    document = _document("d", 1, 0.2, only=0.8)

    assert document.facet_score == pytest.approx(0.8)
    assert document.combo_score == pytest.approx(0.5)


def test_breadth_uses_robust_per_subnarrative_passage_threshold() -> None:
    documents = tuple(
        _document(docid, rank, 0.0, source_ranks={}, s=0.0)
        for rank, docid in enumerate(("weak-0", "weak-1", "weak-2", "weak-3", "strong"), start=1)
    )
    hits = tuple(
        BreadthPassage(
            subnarrative_id="s",
            docid=docid,
            passage=PassageScore(start_char=0, end_char=10, raw_score=score),
        )
        for docid, score in zip(
            ("weak-0", "weak-1", "weak-2", "weak-3", "strong"),
            (0.0, 1.0, 2.0, 3.0, 20.0),
        )
    )

    counts = breadth_counts(documents, hits)

    assert counts["strong"] == (1, 1)
    assert all(counts[f"weak-{index}"] == (0, 0) for index in range(4))


def test_breadth_retains_at_most_three_overlap_suppressed_passages_per_document() -> None:
    documents = (_document("target", 1, 1.0, source_ranks={}, s=1.0),) + tuple(
        _document(f"weak-{index}", index + 2, 0.0, source_ranks={}, s=0.0)
        for index in range(4)
    )
    hits = tuple(
        BreadthPassage(
            subnarrative_id="s",
            docid="target",
            passage=PassageScore(
                start_char=index * 20,
                end_char=index * 20 + 10,
                raw_score=100.0 - index,
            ),
        )
        for index in range(4)
    ) + tuple(
        BreadthPassage("s", f"weak-{index}", PassageScore(0, 10, 0.0))
        for index in range(4)
    )

    counts = breadth_counts(documents, hits)

    assert counts["target"] == (1, 3)
    assert all(counts[f"weak-{index}"] == (0, 0) for index in range(4))


def test_all_rankings_share_the_cutoff_set_without_padding() -> None:
    documents = (
        _document("narrative", 2, 1.0, s1=0.8, s2=0.7),
        _document("facet", 1, 0.8, s1=1.0, s2=0.9),
        _document("weak-a", 3, 0.2, s1=0.2, s2=0.2),
        _document("weak-b", 4, 0.2, s1=0.2, s2=0.2),
        _document("weak-c", 5, 0.2, s1=0.2, s2=0.2),
    )
    hits = (
        BreadthPassage("s1", "facet", PassageScore(0, 10, 9.0)),
        BreadthPassage("s2", "facet", PassageScore(20, 30, 8.0)),
        BreadthPassage("s1", "narrative", PassageScore(0, 10, 7.0)),
    )

    rankings = build_rankings(
        documents,
        hits,
        eligible_docids=("facet", "narrative"),
    )

    assert rankings.eligible_docids == ("facet", "narrative")
    assert set(rankings.narrative) == set(rankings.eligible_docids)
    assert set(rankings.combo) == set(rankings.eligible_docids)
    assert set(rankings.breadth) == set(rankings.eligible_docids)
    assert rankings.narrative == ("narrative", "facet")
    assert rankings.combo == ("facet", "narrative")
    assert rankings.breadth == ("facet", "narrative")


def test_variable_cutoff_may_admit_more_than_one_thousand_documents() -> None:
    documents = tuple(
        _document(f"strong-{index:04d}", index + 1, 0.9, s=0.9)
        for index in range(1_001)
    ) + tuple(
        _document(f"weak-{index:04d}", index + 1_002, 0.1, s=0.1)
        for index in range(1_002)
    )

    eligible = select_eligible_documents(documents)

    assert len(eligible) == 1_001


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file_receipt(topic_dir: Path, relative_path: str) -> dict[str, object]:
    body = (topic_dir / relative_path).read_bytes()
    return {
        "bytes": len(body),
        "relative_path": relative_path,
        "sha256": sha256(body).hexdigest(),
    }


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    topic_dir = source / "rag2026-2"
    _write_json(
        source / "retrieval_export_manifest.json",
        {
            "export_code_commit": "a" * 40,
            "run_id": "fixture-run",
            "selected_topic_ids": ["rag2026-2"],
        },
    )
    document_root = tmp_path / "documents"
    store = DocumentStore(document_root)
    first = store.admit_text("first document text")
    second = store.admit_text("second document text")
    _write_json(
        topic_dir / "decomposition/result.json",
        {
            "error": None,
            "topic": {"id": "rag2026-2", "narrative": "official narrative"},
            "subnarratives": [
                {
                    "topic_id": "rag2026-2",
                    "subnarrative_id": "s1",
                    "text": "pooled subnarrative",
                    "bm25_queries": ["variant one", "variant two"],
                }
            ],
        },
    )
    decomposition_body = (topic_dir / "decomposition/result.json").read_bytes()
    _write_json(
        topic_dir / "decomposition/manifest.json",
        {
            "planner": {"model": "fixture"},
            "result_bytes": len(decomposition_body),
            "result_file": "result.json",
            "result_sha256": sha256(decomposition_body).hexdigest(),
            "schema_version": "facet-decomposition-manifest-v1",
        },
    )
    _write_json(topic_dir / "decomposition.json", {"fixture": True})
    _write_json(
        topic_dir / "retrieval/audit.json",
        {
            "topic_id": "rag2026-2",
            "lanes": [
                {
                    "lane_name": "original",
                    "subnarrative_id": None,
                    "candidates": [
                        {
                            "docid": "d1",
                            "bm25_rank": 5,
                            "bm25_score": 1.0,
                            "text_sha256": first.content_sha256,
                        },
                        {
                            "docid": "d2",
                            "bm25_rank": 1,
                            "bm25_score": 2.0,
                            "text_sha256": second.content_sha256,
                        },
                    ],
                },
                {
                    "lane_name": "facet:s1:text",
                    "subnarrative_id": "s1",
                    "candidates": [
                        {
                            "docid": "d1",
                            "bm25_rank": 2,
                            "bm25_score": 3.0,
                            "text_sha256": first.content_sha256,
                        }
                    ],
                },
            ],
        },
    )
    _write_json(topic_dir / "retrieval/evidence-bundle.json", {"fixture": True})
    _write_json(
        topic_dir / "retrieval/complete.json",
        {
            "artifacts": [
                _file_receipt(topic_dir, "decomposition.json"),
                _file_receipt(topic_dir, "retrieval/audit.json"),
                _file_receipt(topic_dir, "retrieval/evidence-bundle.json"),
            ],
            "decomposition_source_sha256": sha256(decomposition_body).hexdigest(),
            "narrative_sha256": sha256(b"official narrative").hexdigest(),
            "phase": "retrieve",
            "schema_version": "facet_pilot_v2",
            "topic_id": "rag2026-2",
        },
    )
    _write_json(
        topic_dir / "scoring/selection.json",
        {
            "topic_id": "rag2026-2",
            "union_pool": [
                {"docid": "d2", "memberships": ["original"]},
                {"docid": "d1", "memberships": ["original", "facet:s1:text"]},
            ],
        },
    )
    lane_score_path = topic_dir / "scoring/lane_scores.jsonl"
    lane_score_path.parent.mkdir(parents=True, exist_ok=True)
    lane_score_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in (
                {
                    "topic_id": "rag2026-2",
                    "lane_name": "original",
                    "docid": "d1",
                    "aggregate_score": 5.0,
                },
                {
                    "topic_id": "rag2026-2",
                    "lane_name": "original",
                    "docid": "d2",
                    "aggregate_score": 4.0,
                },
                {
                    "topic_id": "rag2026-2",
                    "lane_name": "facet:s1:text",
                    "docid": "d1",
                    "aggregate_score": 6.0,
                },
            )
        ),
        encoding="utf-8",
    )
    for relative_path in (
        "scoring/selected_documents.jsonl",
        "scoring/selected_subnarrative_scores.jsonl",
    ):
        path = topic_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    retrieval_manifest_body = (topic_dir / "retrieval/complete.json").read_bytes()
    _write_json(
        topic_dir / "scoring/complete.json",
        {
            "artifacts": [
                _file_receipt(topic_dir, "scoring/lane_scores.jsonl"),
                _file_receipt(topic_dir, "scoring/selected_documents.jsonl"),
                _file_receipt(topic_dir, "scoring/selection.json"),
                _file_receipt(
                    topic_dir, "scoring/selected_subnarrative_scores.jsonl"
                ),
            ],
            "decomposition_source_sha256": sha256(decomposition_body).hexdigest(),
            "narrative_sha256": sha256(b"official narrative").hexdigest(),
            "phase": "score",
            "retrieval_manifest_sha256": sha256(
                retrieval_manifest_body
            ).hexdigest(),
            "schema_version": "facet_pilot_v2",
            "topic_id": "rag2026-2",
        },
    )
    _write_json(topic_dir / "canonical/retrieval-projection.json", {"fixture": True})
    projection_body = (
        topic_dir / "canonical/retrieval-projection.json"
    ).read_bytes()
    scoring_manifest_body = (topic_dir / "scoring/complete.json").read_bytes()
    projection_manifest = {
        "phase": "retrieval_projection",
        "projection_bytes": len(projection_body),
        "projection_filename": "retrieval-projection.json",
        "projection_sha256": sha256(projection_body).hexdigest(),
        "retrieval_status": "complete",
        "retrieval_stopping_reason": "coverage_sufficient",
        "schema_version": "retrieval_projection_manifest_v4",
        "source_seals": {
            "config_sha256": "0" * 64,
            "decomposition_source_sha256": sha256(decomposition_body).hexdigest(),
            "narrative_sha256": sha256(b"official narrative").hexdigest(),
            "retrieval_manifest_sha256": sha256(retrieval_manifest_body).hexdigest(),
            "scoring_manifest_sha256": sha256(scoring_manifest_body).hexdigest(),
        },
        "topic_id": "rag2026-2",
    }
    _write_json(
        topic_dir / "canonical/retrieval-projection-manifest.json",
        projection_manifest,
    )
    projection_manifest_body = (
        topic_dir / "canonical/retrieval-projection-manifest.json"
    ).read_bytes()
    receipt = {
        "config_sha256": "0" * 64,
        "mode": "online",
        "projection_manifest_sha256": sha256(projection_manifest_body).hexdigest(),
        "run_id": "source",
        "schema_version": "topic-job-receipt-v3",
        "status": "complete",
        "stopping_reason": "coverage_sufficient",
        "topic_id": "rag2026-2",
    }
    (topic_dir / "topic-job-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return source, document_root


def test_load_topic_input_pools_query_variants_and_uses_best_source_rank(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)

    topic = load_topic_input(source, "rag2026-2", document_root)

    assert topic.topic_id == "rag2026-2"
    assert topic.narrative == "official narrative"
    assert len(topic.subnarratives) == 1
    assert topic.subnarratives[0].text == "pooled subnarrative"
    assert topic.subnarratives[0].retrieval_query_texts == (
        "variant one",
        "variant two",
    )
    assert [(row.docid, row.best_retrieval_rank) for row in topic.documents] == [
        ("d1", 2),
        ("d2", 1),
    ]
    assert [row.text for row in topic.documents] == [
        "first document text",
        "second document text",
    ]
    assert topic.documents[0].subnarrative_source_ranks == {"s1": 2}
    assert topic.documents[1].subnarrative_source_ranks == {}
    assert topic.source_sha256s["scoring/lane_scores.jsonl"] == sha256(
        (source / "rag2026-2/scoring/lane_scores.jsonl").read_bytes()
    ).hexdigest()


def test_load_topic_input_rejects_union_not_bound_in_retrieval_audit(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    selection_path = source / "rag2026-2/scoring/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["union_pool"].append({"docid": "ghost", "memberships": []})
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="digest|union pool"):
        load_topic_input(source, "rag2026-2", document_root)


def test_load_topic_input_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    audit_path = source / "rag2026-2/retrieval/audit.json"
    audit_path.write_text('{"topic_id":"rag2026-2","topic_id":"other"}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON|duplicate|digest"):
        load_topic_input(source, "rag2026-2", document_root)


@pytest.mark.parametrize("topic_id", [".", "..", "rag2026-x", "rag2026-2/other"])
def test_load_topic_input_requires_official_topic_id(topic_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="topic_id"):
        load_topic_input(tmp_path, topic_id, tmp_path)


def test_load_topic_input_rejects_zero_subnarratives(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    decomposition_path = source / "rag2026-2/decomposition/result.json"
    decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
    decomposition["subnarratives"] = []
    _write_json(decomposition_path, decomposition)

    with pytest.raises(ValueError, match="source seal|digest|subnarrative"):
        load_topic_input(source, "rag2026-2", document_root)


def test_load_topic_input_verifies_document_content_digest(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    audit = json.loads(
        (source / "rag2026-2/retrieval/audit.json").read_text(encoding="utf-8")
    )
    digest = audit["lanes"][0]["candidates"][0]["text_sha256"]
    object_path = document_root / "sha256" / digest[:2] / f"{digest}.utf8"
    object_path.write_text("tampered text", encoding="utf-8")

    with pytest.raises(Exception, match="digest"):
        load_topic_input(source, "rag2026-2", document_root)


def test_load_topic_input_can_materialize_only_an_authenticated_candidate_subset(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    audit = json.loads(
        (source / "rag2026-2/retrieval/audit.json").read_text(encoding="utf-8")
    )
    second_digest = audit["lanes"][0]["candidates"][1]["text_sha256"]
    (
        document_root
        / "sha256"
        / second_digest[:2]
        / f"{second_digest}.utf8"
    ).unlink()

    topic = load_topic_input(
        source,
        "rag2026-2",
        document_root,
        selected_docids=("d1",),
    )

    assert tuple(row.docid for row in topic.documents) == ("d1",)
    with pytest.raises(ValueError, match="selected docids"):
        load_topic_input(
            source,
            "rag2026-2",
            document_root,
            selected_docids=("ghost",),
        )


def test_topic_sort_key_orders_numeric_suffixes_naturally() -> None:
    assert sorted(["rag2026-10", "rag2026-2", "rag2026-1"], key=topic_sort_key) == [
        "rag2026-1",
        "rag2026-2",
        "rag2026-10",
    ]


class _FakePassageScorer:
    def __init__(self, *, cached: bool) -> None:
        self.cached = cached
        self._stats = {"cache_hits": 0, "cache_misses": 0, "model_batches": 0}

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "sentence-transformers-cross-encoder",
            "backend_version": "5.6.0",
            "model": "mixedbread-ai/mxbai-rerank-base-v2",
            "model_revision": "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
            "score_representation": "raw_logits",
            "inference_dtype": "bfloat16",
            "max_length": 1024,
            "batch_size": 32,
            "device": "cuda",
            "input_policy": "topic_passage_query_text_v1",
            "implementation_version": 1,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def rank(
        self,
        query_text: str,
        chunks: tuple[TextChunk, ...],
    ) -> tuple[ScoredPassage, ...]:
        count = len(chunks)
        if self.cached:
            self._stats["cache_hits"] += count
        else:
            self._stats["cache_misses"] += count
            self._stats["model_batches"] += (count + 31) // 32
        return tuple(
            ScoredPassage(
                chunk=chunk,
                relevance_score=float(len(query_text)) + chunk.start_char / 100.0,
            )
            for chunk in chunks
        )


def _candidate_core(
    topic: object,
    candidate_docids: tuple[str, ...] | None = None,
) -> CandidateCore:
    documents = tuple(topic.documents)  # type: ignore[attr-defined]
    selected = candidate_docids or tuple(row.docid for row in documents)
    selected = tuple(sorted(selected, key=lambda value: value.encode("utf-8")))
    admitted_digest = sha256(
        b"".join(docid.encode("utf-8") + b"\n" for docid in selected)
    ).hexdigest()
    empty_digest = sha256(b"").hexdigest()
    lanes = (
        CandidateLaneStat(
            lane_name="original",
            median=0.0,
            mad=0.0,
            threshold=0.0,
            comparison="strictly_greater_than",
            observed_count=len(documents),
            admitted_count=len(selected),
            admitted_docids_sha256=admitted_digest,
        ),
    ) + tuple(
        CandidateLaneStat(
            lane_name=f"facet:{row.subnarrative_id}:text",
            median=0.0,
            mad=0.0,
            threshold=0.0,
            comparison="strictly_greater_than",
            observed_count=max(1, len(documents)),
            admitted_count=0,
            admitted_docids_sha256=empty_digest,
        )
        for row in topic.subnarratives  # type: ignore[attr-defined]
    )
    return CandidateCore(
        topic_id=topic.topic_id,  # type: ignore[attr-defined]
        lane_scores_sha256=topic.source_sha256s[  # type: ignore[attr-defined]
            "scoring/lane_scores.jsonl"
        ],
        candidate_docids=selected,
        pre_fallback_count=len(selected),
        fallback_used=False,
        admission_multiplicity_histogram={"1": len(selected)},
        lanes=lanes,
    )


def test_score_topic_builds_candidate_only_narrative_and_subnarrative_matrix(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    core = _candidate_core(topic, ("d1",))
    expected_chunks = len(
        chunker.split_text(topic.documents[0].text, document_id="d1")
    )
    scorer = _FakePassageScorer(cached=False)

    matrix = score_topic(topic, candidate_core=core, scorer=scorer, chunker=chunker)

    assert [row.unit_id for row in matrix.units] == ["__narrative__", "s1"]
    assert matrix.candidate_core == core
    assert len(matrix.passages) == expected_chunks * 2
    assert matrix.cache_stats == {
        "cache_hits": 0,
        "cache_misses": expected_chunks * 2,
        "model_batches": ((expected_chunks + 31) // 32) * 2,
    }
    assert {row.docid for row in matrix.passages} == {"d1"}


def test_topic_matrix_round_trip_is_hashed_and_contains_no_document_text(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    matrix = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=chunker,
    )
    output_dir = tmp_path / "matrix"

    manifest_path = write_topic_matrix(matrix, output_dir)
    restored = read_topic_matrix(output_dir)

    assert restored == matrix
    assert manifest_path.name == "topic-matrix-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header = json.loads(
        (output_dir / "topic-matrix.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert header["schema_version"] == "retrieval-baseline-topic-matrix-v3"
    assert header["candidate_core"]["candidate_docids"] == ["d1", "d2"]
    assert manifest["schema_version"] == "retrieval-baseline-topic-matrix-manifest-v3"
    assert len(manifest["candidate_core_sha256"]) == 64
    assert manifest["topic_id"] == "rag2026-2"
    assert manifest["passage_pair_count"] == len(matrix.passages)
    assert manifest["document_semantic_pair_count"] == (
        len(matrix.documents) * len(matrix.units)
    )
    artifact_text = (output_dir / "topic-matrix.jsonl").read_text(encoding="utf-8")
    assert "first document text" not in artifact_text
    assert "second document text" not in artifact_text


def test_score_topics_reuses_one_scorer_and_preserves_requested_topic_order(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    first = load_topic_input(source, "rag2026-2", document_root)
    second = replace(first, topic_id="rag2026-10")
    first_core = _candidate_core(first, ("d1",))
    second_core = replace(first_core, topic_id="rag2026-10")
    scorer = _FakePassageScorer(cached=False)

    matrices = score_topics(
        ((second, second_core), (first, first_core)),
        scorer=scorer,
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )

    assert tuple(matrix.topic_id for matrix in matrices) == (
        "rag2026-10",
        "rag2026-2",
    )
    assert all(tuple(row.docid for row in matrix.documents) == ("d1",) for matrix in matrices)
    assert matrices[0].cache_stats == matrices[1].cache_stats
    assert scorer.stats["model_batches"] == sum(
        matrix.cache_stats["model_batches"] for matrix in matrices
    )


def test_score_topic_rejects_candidate_identity_before_scoring(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    scorer = _FakePassageScorer(cached=False)

    with pytest.raises(ValueError, match="topic identity"):
        score_topic(
            topic,
            candidate_core=replace(_candidate_core(topic), topic_id="rag2026-3"),
            scorer=scorer,
            chunker=SemanticTextChunker(
                ChunkingConfig(max_characters=3_500, overlap_characters=350)
            ),
        )

    assert scorer.stats == {"cache_hits": 0, "cache_misses": 0, "model_batches": 0}


def test_cache_only_replay_records_zero_model_batches_and_identical_scores(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    core = _candidate_core(topic)
    first = score_topic(
        topic,
        candidate_core=core,
        scorer=_FakePassageScorer(cached=False),
        chunker=chunker,
    )

    replay = score_topic(
        topic,
        candidate_core=core,
        scorer=_FakePassageScorer(cached=True),
        chunker=chunker,
    )

    first_root = tmp_path / "first"
    replay_root = tmp_path / "replay"
    write_topic_matrix(first, first_root)
    write_topic_matrix(replay, replay_root)

    assert replay.passages == first.passages
    assert (first_root / "topic-matrix.jsonl").read_bytes() == (
        replay_root / "topic-matrix.jsonl"
    ).read_bytes()
    assert (first_root / "topic-matrix-manifest.json").read_bytes() == (
        replay_root / "topic-matrix-manifest.json"
    ).read_bytes()
    assert replay.cache_stats["cache_misses"] == 0
    assert replay.cache_stats["model_batches"] == 0
    assert replay.cache_stats["cache_hits"] == len(replay.passages)


def test_rank_topic_matrix_keeps_every_candidate_and_uses_cross_scored_breadth(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )

    ranked = rank_topic_matrix(matrix)

    assert len(ranked.document_scores) == 2
    assert ranked.rankings.eligible_docids == ("d1", "d2")
    assert ranked.candidate_core == matrix.candidate_core
    assert {hit.docid for hit in ranked.breadth_passages} == {"d1", "d2"}
    assert set(ranked.rankings.narrative) == set(ranked.rankings.eligible_docids)
    assert set(ranked.rankings.combo) == set(ranked.rankings.eligible_docids)
    assert set(ranked.rankings.breadth) == set(ranked.rankings.eligible_docids)


def test_export_runs_writes_three_deterministic_naturally_ordered_trec_files(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    base = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )
    matrices = (
        replace(
            base,
            topic_id="rag2026-10",
            candidate_core=replace(base.candidate_core, topic_id="rag2026-10"),
        ),
        base,
    )

    first_manifest = export_runs(matrices, tmp_path / "first")
    second_manifest = export_runs(matrices, tmp_path / "second")

    first = json.loads(first_manifest.read_text(encoding="utf-8"))
    second = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert first == second
    assert set(first["run_files"]) == {"narrative", "combo", "breadth"}
    for name, relative_path in first["run_files"].items():
        assert relative_path.endswith("/r_output_trec_rag_2026.tsv")
        first_body = (tmp_path / "first" / relative_path).read_bytes()
        second_body = (tmp_path / "second" / second["run_files"][name]).read_bytes()
        assert first_body == second_body
        lines = first_body.decode("utf-8").splitlines()
        assert lines[0].startswith("rag2026-2 Q0 ")
        assert any(line.startswith("rag2026-10 Q0 ") for line in lines)
    assert [row["topic_id"] for row in first["topics"]] == [
        "rag2026-2",
        "rag2026-10",
    ]
    assert first["source_export_manifest_sha256"] == sha256(
        (source / "retrieval_export_manifest.json").read_bytes()
    ).hexdigest()
    assert all(
        row["narrative_docids_sha256"] == row["combo_docids_sha256"]
        == row["breadth_docids_sha256"]
        for row in first["topics"]
    )
    for row in first["topics"]:
        assert row["document_semantic_pair_count"] == 4
        assert row["candidate_count"] == 2
        assert row["k"] == row["candidate_count"]
        assert row["subnarrative_source_pools"]["s1"]["count"] == 1
        assert len(row["subnarrative_source_pools"]["s1"]["docids_sha256"]) == 64
        assert sum(
            row["admission_multiplicity_histogram"].values()
        ) == row["pre_fallback_count"]
        assert row["candidate_lanes"][0]["lane_name"] == "original"
        assert row["source_sha256s"] == base.source_sha256s
    for name, receipt in first["run_file_receipts"].items():
        body = (tmp_path / "first" / first["run_files"][name]).read_bytes()
        assert receipt["sha256"] == sha256(body).hexdigest()
        assert receipt["bytes"] == len(body)


def test_export_runs_rejects_mismatched_root_export_manifests(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    base = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )
    conflicting = replace(
        base,
        topic_id="rag2026-10",
        candidate_core=replace(base.candidate_core, topic_id="rag2026-10"),
        source_sha256s={
            **base.source_sha256s,
            "retrieval_export_manifest.json": "f" * 64,
        },
    )

    with pytest.raises(ValueError, match="retrieval export manifest"):
        export_runs((base, conflicting), tmp_path / "runs")


def test_matrix_rejects_unpinned_identity_and_incomplete_chunk_coverage(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )

    with pytest.raises(ValueError, match="scorer identity"):
        write_topic_matrix(
            replace(matrix, scorer_identity={**matrix.scorer_identity, "model": "wrong"}),
            tmp_path / "wrong-identity",
        )
    with pytest.raises(ValueError, match="coverage"):
        write_topic_matrix(
            replace(matrix, passages=matrix.passages[:-1]),
            tmp_path / "missing-passage",
        )
    with pytest.raises(ValueError, match="candidate docids|candidate-core"):
        write_topic_matrix(
            replace(
                matrix,
                candidate_core=_candidate_core(topic, ("d1",)),
            ),
            tmp_path / "wrong-candidate-set",
        )
    with pytest.raises(ValueError, match="lane-score source hash"):
        write_topic_matrix(
            replace(
                matrix,
                candidate_core=replace(
                    matrix.candidate_core,
                    lane_scores_sha256="f" * 64,
                ),
            ),
            tmp_path / "wrong-candidate-source",
        )


def test_matrix_publication_is_create_only(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )
    output = tmp_path / "matrix"
    write_topic_matrix(matrix, output)
    changed_passage = replace(
        matrix.passages[0], raw_score=matrix.passages[0].raw_score + 1.0
    )

    with pytest.raises(ValueError, match="conflicting immutable"):
        write_topic_matrix(
            replace(matrix, passages=(changed_passage, *matrix.passages[1:])),
            output,
        )


def test_rank_and_verify_cli_use_existing_topic_matrices(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        candidate_core=_candidate_core(topic),
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=3_500, overlap_characters=350)
        ),
    )
    matrix_root = tmp_path / "matrices"
    write_topic_matrix(matrix, matrix_root / "rag2026-2")
    run_root = tmp_path / "runs"

    assert main(
        [
            "rank",
            "--matrix-dir",
            str(matrix_root),
            "--output-dir",
            str(run_root),
        ]
    ) == 0
    rank_receipt = json.loads(capsys.readouterr().out)
    assert rank_receipt["topic_count"] == 1
    assert main(
        [
            "verify",
            "--matrix-dir",
            str(matrix_root),
            "--output-dir",
            str(run_root),
        ]
    ) == 0
    verify_receipt = json.loads(capsys.readouterr().out)
    assert verify_receipt == {"status": "verified", "topic_count": 1}

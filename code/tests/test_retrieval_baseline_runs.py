from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk
from trec_rag.document_store import DocumentStore
from trec_rag.mixedbread_passage_scorer import ScoredPassage
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


def test_breadth_takes_three_per_document_before_global_top_one_hundred() -> None:
    documents = (_document("target", 1, 1.0, s=1.0),) + tuple(
        _document(f"other-{index:03d}", index + 2, 0.0, s=0.0)
        for index in range(97)
    )
    hits = tuple(
        BreadthPassage(
            subnarrative_id="s",
            docid="target",
            passage=PassageScore(
                start_char=index * 20,
                end_char=index * 20 + 10,
                raw_score=200.0 - index,
            ),
        )
        for index in range(4)
    ) + tuple(
        BreadthPassage(
            subnarrative_id="s",
            docid=f"other-{index:03d}",
            passage=PassageScore(
                start_char=0,
                end_char=10,
                raw_score=100.0 - index,
            ),
        )
        for index in range(97)
    )

    counts = breadth_counts(documents, hits)

    assert counts["target"] == (1, 3)
    assert counts["other-096"] == (1, 1)


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

    rankings = build_rankings(documents, hits)

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


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    topic_dir = source / "rag2026-2"
    document_root = tmp_path / "documents"
    store = DocumentStore(document_root)
    first = store.admit_text("first document text")
    second = store.admit_text("second document text")
    _write_json(
        topic_dir / "topic-job-receipt.json",
        {"status": "complete", "topic_id": "rag2026-2"},
    )
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


def test_load_topic_input_rejects_union_not_bound_in_retrieval_audit(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    selection_path = source / "rag2026-2/scoring/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["union_pool"].append({"docid": "ghost", "memberships": []})
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="union pool"):
        load_topic_input(source, "rag2026-2", document_root)


def test_load_topic_input_rejects_zero_subnarratives(tmp_path: Path) -> None:
    source, document_root = _source_fixture(tmp_path)
    decomposition_path = source / "rag2026-2/decomposition/result.json"
    decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
    decomposition["subnarratives"] = []
    _write_json(decomposition_path, decomposition)

    with pytest.raises(ValueError, match="subnarrative"):
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
        return {"model": "deterministic-fake", "batch_size": 8}

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
            self._stats["model_batches"] += (count + 7) // 8
        return tuple(
            ScoredPassage(
                chunk=chunk,
                relevance_score=float(len(query_text)) + chunk.start_char / 100.0,
            )
            for chunk in chunks
        )


def test_score_topic_builds_complete_narrative_and_subnarrative_matrix(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=12, overlap_characters=3)
    )
    expected_chunks = sum(
        len(chunker.split_text(document.text, document_id=document.docid))
        for document in topic.documents
    )
    scorer = _FakePassageScorer(cached=False)

    matrix = score_topic(topic, scorer=scorer, chunker=chunker)

    assert [row.unit_id for row in matrix.units] == ["__narrative__", "s1"]
    assert len(matrix.passages) == expected_chunks * 2
    assert matrix.cache_stats == {
        "cache_hits": 0,
        "cache_misses": expected_chunks * 2,
        "model_batches": ((expected_chunks + 7) // 8) * 2,
    }
    assert {row.docid for row in matrix.passages} == {"d1", "d2"}


def test_topic_matrix_round_trip_is_hashed_and_contains_no_document_text(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=12, overlap_characters=3)
    )
    matrix = score_topic(topic, scorer=_FakePassageScorer(cached=False), chunker=chunker)
    output_dir = tmp_path / "matrix"

    manifest_path = write_topic_matrix(matrix, output_dir)
    restored = read_topic_matrix(output_dir)

    assert restored == matrix
    assert manifest_path.name == "topic-matrix-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["topic_id"] == "rag2026-2"
    assert manifest["passage_pair_count"] == len(matrix.passages)
    assert manifest["document_semantic_pair_count"] == (
        len(matrix.documents) * len(matrix.units)
    )
    artifact_text = (output_dir / "topic-matrix.jsonl").read_text(encoding="utf-8")
    assert "first document text" not in artifact_text
    assert "second document text" not in artifact_text


def test_cache_only_replay_records_zero_model_batches_and_identical_scores(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=12, overlap_characters=3)
    )
    first = score_topic(topic, scorer=_FakePassageScorer(cached=False), chunker=chunker)

    replay = score_topic(topic, scorer=_FakePassageScorer(cached=True), chunker=chunker)

    assert replay.passages == first.passages
    assert replay.cache_stats["cache_misses"] == 0
    assert replay.cache_stats["model_batches"] == 0
    assert replay.cache_stats["cache_hits"] == len(replay.passages)


def test_rank_topic_matrix_uses_complete_scores_but_pooled_sources_for_breadth(
    tmp_path: Path,
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=12, overlap_characters=3)
        ),
    )

    ranked = rank_topic_matrix(matrix)

    assert len(ranked.document_scores) == 2
    assert ranked.rankings.eligible_docids
    assert {hit.docid for hit in ranked.breadth_passages} == {"d1"}
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
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=12, overlap_characters=3)
        ),
    )
    matrices = (replace(base, topic_id="rag2026-10"), base)

    first_manifest = export_runs(matrices, tmp_path / "first")
    second_manifest = export_runs(matrices, tmp_path / "second")

    first = json.loads(first_manifest.read_text(encoding="utf-8"))
    second = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert first == second
    assert set(first["run_files"]) == {"narrative", "combo", "breadth"}
    for name, relative_path in first["run_files"].items():
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
    assert all(
        row["narrative_docids_sha256"] == row["combo_docids_sha256"]
        == row["breadth_docids_sha256"]
        for row in first["topics"]
    )
    for row in first["topics"]:
        assert row["document_semantic_pair_count"] == 4
        assert row["subnarrative_source_pools"]["s1"]["count"] == 1
        assert len(row["subnarrative_source_pools"]["s1"]["docids_sha256"]) == 64
        assert sum(
            row["admission_multiplicity_histogram"].values()
        ) == row["pre_fallback_count"]


def test_rank_and_verify_cli_use_existing_topic_matrices(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, document_root = _source_fixture(tmp_path)
    topic = load_topic_input(source, "rag2026-2", document_root)
    matrix = score_topic(
        topic,
        scorer=_FakePassageScorer(cached=False),
        chunker=SemanticTextChunker(
            ChunkingConfig(max_characters=12, overlap_characters=3)
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

from __future__ import annotations

import math

import pytest

from trec_rag.topic213_response_experiment import PassageCorpus, PassageUnit
from trec_rag.topic213_utokyo_experiment import (
    generate_top20_response,
    normalize_answer_label,
    prepare_query,
    rerank_fused_candidates,
    retrieve_four_streams,
)
from trec_rag.utokyo_retrieval import (
    RankedPassage,
    bm25_rank,
    hyde_vector_mix,
    reciprocal_rank_fusion,
    sliding_window_rerank,
)


LABEL_A = '"What triggered the Korean War?"'
LABEL_B = '"How did the Korean War conclude?"'


def _corpus() -> PassageCorpus:
    return PassageCorpus(
        document_ids=("doc-a", "doc-b", "doc-c"),
        sub_narratives=(LABEL_A, LABEL_B),
        passages=(
            PassageUnit("P000001", "doc-a", LABEL_A, "North Korea invaded South Korea."),
            PassageUnit("P000002", "doc-b", LABEL_B, "An armistice ended the fighting."),
            PassageUnit("P000003", "doc-c", LABEL_A, "Unrelated background text."),
        ),
        evidence_assignment_count=3,
    )


def test_hyde_vector_mix_normalizes_inputs_and_output():
    mixed = hyde_vector_mix([3.0, 0.0], [0.0, 4.0], alpha=0.7)

    expected_norm = math.sqrt(0.3**2 + 0.7**2)
    assert mixed == pytest.approx([0.3 / expected_norm, 0.7 / expected_norm])
    assert math.sqrt(sum(value**2 for value in mixed)) == pytest.approx(1.0)


def test_rrf_keeps_stream_provenance_and_uses_k_60():
    streams = {
        "lexical": [
            RankedPassage("p1", 1, 9.0),
            RankedPassage("p2", 2, 8.0),
        ],
        "dense": [
            RankedPassage("p2", 1, 0.9),
            RankedPassage("p3", 2, 0.8),
        ],
    }

    fused, provenance = reciprocal_rank_fusion(streams, k=60, top_k=3)

    assert [row.passage_id for row in fused] == ["p2", "p1", "p3"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert provenance["p2"]["stream_ranks"] == {"lexical": 2, "dense": 1}


def test_sliding_window_moves_from_lower_to_higher_ranks_for_each_pass():
    calls: list[tuple[int, int, list[str]]] = []

    def rerank(window, pass_index, window_index):
        calls.append((pass_index, window_index, list(window)))
        return list(reversed(window))

    result, audit = sliding_window_rerank(
        [f"p{index:02d}" for index in range(20)],
        rerank_window=rerank,
        window_size=10,
        stride=5,
        num_passes=2,
    )

    assert len(result) == 20
    assert len(audit) == 6
    assert [(row["start"], row["end"]) for row in audit[:3]] == [
        (10, 20),
        (5, 15),
        (0, 10),
    ]
    assert calls[0][0:2] == (1, 1)
    assert calls[3][0:2] == (2, 1)


def test_bm25_keyword_query_finds_vocabulary_expansion_match():
    pytest.importorskip("rank_bm25")
    corpus = _corpus()

    ranked = bm25_rank(corpus.passages, query="war ceasefire armistice", top_k=3)

    assert ranked[0].passage_id == "P000002"


def test_query_preparation_generates_hyde_and_keywords_without_evidence():
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def complete_json(self, *, stage, payload, **_kwargs):
            self.payloads.append((stage, payload))
            if stage == "utokyo_hyde":
                return {"hypothetical_answer": "A natural hypothetical Korean War answer."}
            if stage == "utokyo_keywords":
                return {"keywords": ["Korean conflict", "armistice"]}
            raise AssertionError(stage)

    client = FakeClient()
    result = prepare_query(client=client, query="Korean War", config={})

    assert result["keywords"] == ["Korean conflict", "armistice"]
    assert result["bm25_expanded_query"].startswith("Korean War Korean conflict")
    assert all("passages" not in payload for _, payload in client.payloads)


def test_answer_label_normalization_restores_released_prefix_and_typo():
    released = [
        "New: How does the Korean War affect UN",
        "New: What motivated China involvement in te Korean War?",
    ]

    assert normalize_answer_label("How does the Korean War affect UN?", released) == released[0]
    assert (
        normalize_answer_label(
            "What motivated China involvement in the Korean War?", released
        )
        == released[1]
    )


def test_four_stream_retrieval_scores_every_passage(monkeypatch, tmp_path):
    corpus = _corpus()
    seen: list[tuple[str, int]] = []

    def fake_bm25(passages, *, query, top_k):
        seen.append(("bm25", len(passages)))
        return [
            RankedPassage(row.passage_id, rank, float(len(passages) - rank))
            for rank, row in enumerate(passages, 1)
        ][:top_k]

    def fake_splade(passages, **_kwargs):
        seen.append(("splade", len(passages)))
        return [1.0, 3.0, 2.0]

    def fake_dense(passages, *, model_name, **_kwargs):
        seen.append((model_name, len(passages)))
        return [3.0, 2.0, 1.0]

    monkeypatch.setattr("trec_rag.topic213_utokyo_experiment.bm25_rank", fake_bm25)
    monkeypatch.setattr("trec_rag.topic213_utokyo_experiment.splade_scores", fake_splade)
    monkeypatch.setattr("trec_rag.topic213_utokyo_experiment.dense_hyde_scores", fake_dense)
    config = {
        "stream_depth": 3,
        "hyde_alpha": 0.7,
        "splade": {"model": "splade"},
        "dense": [
            {"name": "bge_hyde", "model": "bge"},
            {"name": "qwen_hyde", "model": "qwen"},
        ],
    }

    streams, runtime = retrieve_four_streams(
        corpus=corpus,
        query_preparation={
            "original_query": "query",
            "hypothetical_answer": "hypothesis",
            "bm25_expanded_query": "query keyword",
        },
        config=config,
        cache_dir=tmp_path,
    )

    assert set(streams) == {"bm25_keywords", "splade", "bge_hyde", "qwen_hyde"}
    assert runtime["passages_scored_per_stream"] == 3
    assert seen == [("bm25", 3), ("splade", 3), ("bge", 3), ("qwen", 3)]


def test_local_reranker_appends_a_small_omission_and_audits_it():
    corpus = _corpus()
    lookup = {passage.passage_id: passage for passage in corpus.passages}

    class FakeClient:
        def complete_json(self, *, payload, **_kwargs):
            return {"ranked_passage_ids": payload["current_order_best_to_worst"][:-1]}

    candidates = [
        RankedPassage(passage.passage_id, rank, float(4 - rank))
        for rank, passage in enumerate(corpus.passages, 1)
    ]
    order, audit = rerank_fused_candidates(
        client=FakeClient(),
        query="Korean War",
        candidates=candidates,
        passage_lookup=lookup,
        config={
            "candidate_depth": 3,
            "window_size": 3,
            "stride": 1,
            "num_passes": 1,
            "max_missing_ids": 1,
        },
    )

    assert order == ["P000001", "P000002", "P000003"]
    assert audit[0]["normalization"]["omitted_passage_ids"] == ["P000003"]


def test_local_reranker_canonicalizes_numeric_passage_id_formatting():
    corpus = _corpus()
    lookup = {passage.passage_id: passage for passage in corpus.passages}

    class FakeClient:
        def complete_json(self, *, payload, **_kwargs):
            ids = list(payload["current_order_best_to_worst"])
            ids[0] = "P0000001"
            return {"ranked_passage_ids": ids}

    candidates = [
        RankedPassage(passage.passage_id, rank, float(4 - rank))
        for rank, passage in enumerate(corpus.passages, 1)
    ]
    order, audit = rerank_fused_candidates(
        client=FakeClient(),
        query="Korean War",
        candidates=candidates,
        passage_lookup=lookup,
        config={
            "candidate_depth": 3,
            "window_size": 3,
            "stride": 1,
            "num_passes": 1,
        },
    )

    assert order[0] == "P000001"
    assert audit[0]["normalization"]["corrected_passage_ids"] == {
        "P0000001": "P000001"
    }


def test_top20_generation_preserves_all_sections_and_selected_citations_only():
    corpus = _corpus()

    class FakeClient:
        model = "fake-model"

        def complete_json(self, *, stage, payload, **_kwargs):
            if stage == "utokyo_generate_top20":
                return {
                    "sections": [
                        {
                            "sub_narrative": LABEL_A,
                            "claims": [
                                {"text": "North Korea invaded.", "passage_ids": ["P000001"]}
                            ],
                        },
                        {
                            "sub_narrative": LABEL_B,
                            "claims": [
                                {"text": "An armistice ended fighting.", "passage_ids": ["P000002"]}
                            ],
                        },
                    ]
                }
            if stage.startswith("audit_support"):
                claim = payload["claims"][0]
                return {
                    "assessments": [
                        {"claim_id": claim["claim_id"], "status": "supported", "notes": ""}
                    ]
                }
            raise AssertionError(stage)

    generation, audit = generate_top20_response(
        client=FakeClient(),
        topic_id="213",
        narrative="Korean War",
        corpus=corpus,
        selected_passage_ids=["P000001", "P000002"],
        release_accounting={"documents_processed": 3, "passages_processed": 3},
        retrieval_record={},
        generation_config={},
        experiment_id="experiment",
        run_id="run",
        source_metadata={},
    )

    assert [section["sub_narrative"] for section in generation["sections"]] == [
        LABEL_A,
        LABEL_B,
    ]
    assert generation["context_policy"]["nuggets_available_during_generation"] is False
    assert {
        passage_id
        for section in generation["sections"]
        for claim in section["claims"]
        for passage_id in claim["passage_ids"]
    } == {"P000001", "P000002"}
    assert len(audit) == 2

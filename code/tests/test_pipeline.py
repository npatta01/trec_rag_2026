import json

import pytest

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.evidence import select_top_k_evidence
from trec_rag.generation import generate_placeholder_rag
from trec_rag.pipeline import run_pipeline
from trec_rag.pipeline_config import load_pipeline_config
from trec_rag.pipeline_models import (
    QueryVariant,
    RankedCandidate,
    RetrievedCandidate,
)
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import passthrough_rank
from trec_rag.retrievers import cache_path, normalize_retrieved_candidates
from trec_rag.topics import Topic


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_config(path, body):
    path.write_text(body, encoding="utf-8")
    return path


def test_config_loads_defaults_and_rejects_submission_run_id(tmp_path):
    config_path = write_config(
        tmp_path / "config.yaml",
        """
experiment:
  id: rag25_bm25_full_query_v1
submission:
  team_id: local-baseline
topics:
  path: topics.tsv
  format: tsv
query_understanding:
  variants:
    - name: original
      type: original_topic
retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [original]
    hits: 100
    index: climbmix-400b
ranking:
  type: passthrough
  dedupe:
    by: docid
    keep: best_rank
    preserve_provenance: true
evidence:
  type: top_k
  k: 5
  require_text: true
  allow_fewer: true
generation:
  type: placeholder
evaluation:
  kind: dev_projected_qrels
  qrels: qrels.txt
  metrics: [ndcg@10, recall@100]
  relevance_threshold: 2
""",
    )

    config = load_pipeline_config(config_path)

    assert config.experiment.id == "rag25_bm25_full_query_v1"
    assert config.output_dir == tmp_path / "outputs" / "rag25_bm25_full_query_v1"
    assert config.run_id == "rag25_bm25_full_query_v1"

    bad_path = write_config(
        tmp_path / "bad.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline, run_id: old-style}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="submission.run_id.*experiment.id"):
        load_pipeline_config(bad_path)


def test_config_rejects_duplicate_names_and_unknown_query_variant(tmp_path):
    duplicate_path = write_config(
        tmp_path / "duplicate.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding:
  variants:
    - {name: original, type: original_topic}
    - {name: original, type: original_topic}
retrievers:
  - {name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )
    unknown_path = write_config(
        tmp_path / "unknown.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding:
  variants:
    - {name: original, type: original_topic}
retrievers:
  - {name: climbmix_bm25, type: pyserini_remote, query_variants: [missing], hits: 10}
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="duplicate query variant"):
        load_pipeline_config(duplicate_path)
    with pytest.raises(ValueError, match="unknown query variant.*missing"):
        load_pipeline_config(unknown_path)


def test_config_rejects_unsupported_passthrough_dedupe_policy(tmp_path):
    config_path = write_config(
        tmp_path / "bad_dedupe.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking:
  type: passthrough
  dedupe: {by: url, keep: average_score, preserve_provenance: true}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="dedupe.*docid.*best_rank"):
        load_pipeline_config(config_path)


def test_config_resolves_input_paths_from_shared_checkout_for_linked_worktrees(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "worktree"
    git_dir = shared / ".git" / "worktrees" / "wt"
    config_dir = worktree / "configs"
    (shared / "trec-rag-data").mkdir(parents=True)
    worktree.mkdir()
    config_dir.mkdir()
    git_dir.mkdir(parents=True)
    (worktree / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (shared / "trec-rag-data" / "topics.tsv").write_text("31\tPrompt\n", encoding="utf-8")
    (shared / "trec-rag-data" / "qrels.txt").write_text("31 0 doc-a 4\n", encoding="utf-8")
    config_path = write_config(
        config_dir / "config.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: trec-rag-data/topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
evaluation:
  qrels: trec-rag-data/qrels.txt
  metrics: [recall@10]
""",
    )

    config = load_pipeline_config(config_path)

    assert config.topics.path == shared / "trec-rag-data" / "topics.tsv"
    assert config.evaluation.qrels == shared / "trec-rag-data" / "qrels.txt"
    assert config.output_dir == worktree / "outputs" / "demo"


def test_original_topic_query_uses_narrative_without_title_concat():
    topic = Topic(
        id="31",
        title="Derived or official title",
        narrative="Full narrative or prompt text.",
    )

    variants = build_query_variants(
        topic,
        variant_configs=[{"name": "original", "type": "original_topic"}],
    )

    assert variants == [
        QueryVariant(
            topic_id="31",
            variant_name="original",
            query_text="Full narrative or prompt text.",
            source_type="original_topic",
        )
    ]


def test_retrieval_normalization_and_cache_path_include_topic_variant_and_retriever():
    query = QueryVariant(
        topic_id="31",
        variant_name="original",
        query_text="e-waste narrative",
        source_type="original_topic",
    )
    response = {
        "candidates": [
            {"rank": 1, "docid": "doc-a", "score": 9.0, "doc": {"contents": "A text"}},
            {"rank": 2, "docid": "doc-b", "score": 8.0, "doc": {"contents": "B text"}},
        ]
    }

    candidates = normalize_retrieved_candidates(
        response,
        query=query,
        retriever_name="climbmix_bm25",
    )

    assert cache_path("31", "original", "climbmix_bm25").name == (
        "31__original__climbmix_bm25.json"
    )
    assert candidates == [
        RetrievedCandidate(
            topic_id="31",
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text="e-waste narrative",
            docid="doc-a",
            rank=1,
            score=9.0,
            text="A text",
        ),
        RetrievedCandidate(
            topic_id="31",
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text="e-waste narrative",
            docid="doc-b",
            rank=2,
            score=8.0,
            text="B text",
        ),
    ]


def test_passthrough_ranking_dedupes_docids_and_preserves_provenance():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-b", 2, 8.0, "B"),
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 3, 7.0, "A later"),
    ]

    ranked = passthrough_rank(candidates)

    assert [(row.docid, row.rank, row.score) for row in ranked] == [
        ("doc-a", 1, 9.0),
        ("doc-b", 2, 8.0),
    ]
    assert len(ranked[0].provenance) == 2
    assert {item["source_rank"] for item in ranked[0].provenance} == {1, 3}


def test_passthrough_ranking_restarts_final_ranks_per_topic():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("32", "original", "bm25", "query", "doc-b", 1, 8.0, "B"),
    ]

    ranked = passthrough_rank(candidates)

    assert [(row.topic_id, row.docid, row.rank) for row in ranked] == [
        ("31", "doc-a", 1),
        ("32", "doc-b", 1),
    ]


def test_passthrough_rejects_multiple_retrieval_streams():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("31", "rewrite", "bm25", "query", "doc-b", 1, 8.0, "B"),
    ]

    with pytest.raises(ValueError, match="passthrough.*one retrieval stream"):
        passthrough_rank(candidates)


def test_evidence_allows_fewer_text_bearing_candidates_and_placeholder_rag_is_cited():
    topic = Topic(id="31", title="E-waste", narrative="Explain e-waste impacts.")
    ranked = [
        RankedCandidate(
            topic_id="31",
            docid="doc-a",
            rank=1,
            score=9.0,
            text="A useful evidence sentence.",
            provenance=[],
        ),
        RankedCandidate(
            topic_id="31",
            docid="doc-b",
            rank=2,
            score=8.0,
            text="",
            provenance=[],
        ),
    ]

    evidence = select_top_k_evidence(ranked, k=2, require_text=True, allow_fewer=True)
    record = generate_placeholder_rag(
        topic,
        evidence,
        team_id="local-baseline",
        run_id="rag25_bm25_full_query_v1",
    )

    assert [item.docid for item in evidence] == ["doc-a"]
    assert record["references"] == ["doc-a"]
    assert record["answer"][0]["citations"] == [0]
    assert "placeholder" in record["answer"][0]["text"].lower()


def test_qrels_metrics_handle_graded_labels_and_unjudged_docs(tmp_path):
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text(
        "31 0 doc-a 4\n"
        "31 0 doc-b 2\n"
        "31 0 doc-c 1\n"
        "32 0 doc-z 0\n",
        encoding="utf-8",
    )
    ranked = [
        RankedCandidate("31", "doc-unjudged", 1, 10.0, "", []),
        RankedCandidate("31", "doc-b", 2, 9.0, "", []),
        RankedCandidate("31", "doc-a", 3, 8.0, "", []),
        RankedCandidate("32", "doc-z", 1, 5.0, "", []),
    ]

    qrels = parse_qrels(qrels_path)
    metrics = evaluate_ranked(
        ranked,
        qrels,
        metric_names=["ndcg@3", "recall@3"],
        relevance_threshold=2,
    )

    assert metrics["per_topic"]["31"]["recall@3"] == 1.0
    assert 0.0 < metrics["per_topic"]["31"]["ndcg@3"] < 1.0
    assert metrics["per_topic"]["32"]["recall@3"] == 0.0
    assert metrics["metrics"]["recall@3"] == 0.5


def test_run_pipeline_writes_stage_outputs_with_fake_retriever(tmp_path):
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text("31\tExplain e-waste impacts.\n", encoding="utf-8")
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("31 0 doc-a 4\n", encoding="utf-8")
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment:
  id: rag25_fake_v1
submission:
  team_id: local-baseline
topics:
  path: {topics_path}
  format: tsv
query_understanding:
  variants:
    - name: original
      type: original_topic
retrievers:
  - name: fake
    type: fake
    query_variants: [original]
ranking:
  type: passthrough
evidence:
  type: top_k
  k: 2
  require_text: true
  allow_fewer: true
generation:
  type: placeholder
evaluation:
  kind: dev_projected_qrels
  qrels: {qrels_path}
  metrics: [ndcg@10, recall@100]
  relevance_threshold: 2
""",
    )

    class FakeRetriever:
        def retrieve(self, query):
            return [
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    "doc-a",
                    1,
                    9.0,
                    "Evidence text.",
                )
            ]

    result = run_pipeline(
        config_path,
        retriever_factories={"fake": lambda _config, _cache_dir: FakeRetriever()},
    )

    assert result.output_dir == tmp_path / "outputs" / "rag25_fake_v1"
    assert (result.output_dir / "r_output_trec_rag_2026.tsv").read_text(
        encoding="utf-8"
    ).splitlines() == ["31 Q0 doc-a 1 9.0 rag25_fake_v1"]
    assert read_jsonl(result.output_dir / "stage_queries.jsonl")[0]["query_text"] == (
        "Explain e-waste impacts."
    )
    assert read_jsonl(result.output_dir / "rag_output_trec_rag_2026.jsonl")[0][
        "references"
    ] == ["doc-a"]
    assert json.loads((result.output_dir / "retrieval_metrics.json").read_text())["metrics"][
        "recall@100"
    ] == 1.0

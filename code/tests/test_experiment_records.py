import csv

from trec_rag.experiment_records import write_experiment_indexes


def test_write_experiment_indexes_from_per_run_folders(tmp_path):
    experiments_dir = tmp_path / "reports" / "experiments"
    record_dir = experiments_dir / "rag25_bm25_full_dev_hits1000_v1"
    record_dir.mkdir(parents=True)
    (record_dir / "manifest.yaml").write_text(
        """
experiment:
  id: rag25_bm25_full_dev_hits1000_v1
  runtime_id: cache_demo_two_topics_hits1000
  run_date: 2026-07-04
  split: dev
config:
  retriever: pyserini_remote
  index: climbmix-400b
  query_source: original_topic_narrative
  hits: 1000
  ranking: passthrough
data:
  topics: trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv
  topic_count: 22
  qrels: trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/example.qrels
local_artifacts:
  output_dir: tmp/hits1000-full-dev/output
  cache_dir: /tmp/cache
counts:
  runfile_rows: 22000
  retrieved_rows: 22000
  rag_rows: 22
metrics:
  ndcg_at_10: 0.4139611685
  recall_at_100: 0.1080879457
""",
        encoding="utf-8",
    )
    (record_dir / "topic_scores.csv").write_text(
        "experiment_id,runtime_id,run_date,split,hits,topic_id,ndcg_at_10,recall_at_100\n"
        "rag25_bm25_full_dev_hits1000_v1,cache_demo_two_topics_hits1000,2026-07-04,dev,1000,overall,0.4139611685,0.1080879457\n"
        "rag25_bm25_full_dev_hits1000_v1,cache_demo_two_topics_hits1000,2026-07-04,dev,1000,14,0.3553332721,0.1223021583\n",
        encoding="utf-8",
    )

    write_experiment_indexes(experiments_dir)

    run_rows = list(csv.DictReader((experiments_dir / "runs.csv").open()))
    score_rows = list(csv.DictReader((experiments_dir / "topic_scores.csv").open()))

    assert b"\r\n" not in (experiments_dir / "runs.csv").read_bytes()
    assert b"\r\n" not in (experiments_dir / "topic_scores.csv").read_bytes()
    assert run_rows == [
        {
            "experiment_id": "rag25_bm25_full_dev_hits1000_v1",
            "runtime_id": "cache_demo_two_topics_hits1000",
            "run_date": "2026-07-04",
            "split": "dev",
            "topics": "rag25-topics-dev.tsv",
            "topic_count": "22",
            "retriever": "pyserini_remote",
            "index": "climbmix-400b",
            "query_source": "original_topic_narrative",
            "hits": "1000",
            "ranking": "passthrough",
            "evaluation_qrels": "example.qrels",
            "record_dir": "rag25_bm25_full_dev_hits1000_v1",
            "output_dir": "tmp/hits1000-full-dev/output",
            "cache_dir": "/tmp/cache",
            "runfile_rows": "22000",
            "retrieved_rows": "22000",
            "rag_rows": "22",
            "ndcg_at_10": "0.4139611685",
            "recall_at_100": "0.1080879457",
            "notes": "",
        }
    ]
    assert [row["topic_id"] for row in score_rows] == ["overall", "14"]

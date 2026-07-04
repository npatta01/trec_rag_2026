import json

import pytest

from trec_rag.baselines.bm25_retrieval import (
    RetrievalRow,
    build_query,
    candidates_to_run_rows,
    run_bm25_retrieval,
    validate_retrieval_run,
)
from trec_rag.topics import Topic, derive_title, load_topics, write_topics_jsonl


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_topics_reads_official_jsonl_and_preserves_numeric_zero_id(tmp_path):
    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(
        json.dumps(
            {
                "id": 0,
                "title": "Industrial Revolution",
                "narrative": " Explain causes and consequences. ",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_topics(topics_path) == [
        Topic(
            id="0",
            title="Industrial Revolution",
            narrative="Explain causes and consequences.",
        )
    ]


def test_load_topics_reads_dev_tsv_and_derives_short_title(tmp_path):
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text(
        "31\t  Explain e-waste risks, recycling innovations, and local sustainability benefits.  \n",
        encoding="utf-8",
    )

    assert load_topics(topics_path, title_words=6) == [
        Topic(
            id="31",
            title="Explain e-waste risks recycling innovations and",
            narrative="Explain e-waste risks, recycling innovations, and local sustainability benefits.",
        )
    ]
    assert derive_title("  Climate adaptation: heat, floods, and health.  ", max_words=5) == (
        "Climate adaptation heat floods and"
    )


def test_load_topics_rejects_missing_or_blank_required_fields(tmp_path):
    topics_path = tmp_path / "bad.jsonl"
    topics_path.write_text(
        json.dumps({"id": "9", "title": "   ", "narrative": "usable narrative"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1.*title"):
        load_topics(topics_path)


def test_write_topics_jsonl_round_trips_loaded_topics(tmp_path):
    output_path = tmp_path / "converted.jsonl"
    topics = [Topic(id="31", title="E-waste impacts", narrative="Explain e-waste impacts.")]

    write_topics_jsonl(topics, output_path)

    assert read_jsonl(output_path) == [
        {"id": "31", "title": "E-waste impacts", "narrative": "Explain e-waste impacts."}
    ]
    assert load_topics(output_path) == topics


def test_candidates_to_run_rows_sorts_by_returned_rank_and_limits_depth():
    topic = Topic(id="31", title="E-waste impacts", narrative="Explain e-waste impacts.")
    candidates = [
        {"rank": 2, "docid": "doc-b", "score": 7.5, "text": "second"},
        {"rank": 1, "docid": "doc-a", "score": 9.0, "text": "first"},
        {"rank": 3, "docid": "doc-c", "score": 5.0, "text": "third"},
    ]

    rows = candidates_to_run_rows(topic, candidates, run_id="bm25_test", depth=2)

    assert rows == [
        RetrievalRow(topic_id="31", docid="doc-a", rank=1, score=9.0, run_id="bm25_test"),
        RetrievalRow(topic_id="31", docid="doc-b", rank=2, score=7.5, run_id="bm25_test"),
    ]


def test_candidates_to_run_rows_rejects_missing_docids():
    topic = Topic(id="31", title="E-waste impacts", narrative="Explain e-waste impacts.")

    with pytest.raises(ValueError, match="topic 31.*missing docid"):
        candidates_to_run_rows(
            topic,
            [{"rank": 1, "score": 9.0, "text": "no docid"}],
            run_id="bm25_test",
            depth=10,
        )


def test_build_query_uses_original_topic_narrative_without_title_concat():
    official_topic = Topic(
        id="99",
        title="Athlete compensation",
        narrative="Explain inclusion, cultural influence, and the business side of sports.",
    )
    derived_tsv_topic = Topic(
        id="14",
        title="I'm interested in sports societal impact",
        narrative=(
            "I'm interested in sports' societal impact, particularly concerning athlete "
            "compensation and inclusion."
        ),
    )

    assert build_query(official_topic) == (
        "Explain inclusion, cultural influence, and the business side of sports."
    )
    assert build_query(derived_tsv_topic) == (
        "I'm interested in sports' societal impact, particularly concerning athlete "
        "compensation and inclusion."
    )


def test_run_bm25_retrieval_uses_full_topic_text_writes_runfile_and_caches_raw_responses(tmp_path):
    class FakeClient:
        def __init__(self):
            self.queries = []

        def search(self, query):
            self.queries.append(query)
            return {
                "candidates": [
                    {"rank": 2, "docid": "doc-b", "score": 7.5, "doc": {"contents": "second"}},
                    {"rank": 1, "docid": "doc-a", "score": 9.0, "doc": {"contents": "first"}},
                ]
            }

    topics = [
        Topic(
            id="31",
            title="E-waste impacts",
            narrative="Use the narrative as part of the BM25 query.",
        )
    ]
    output_path = tmp_path / "r_output_trec_rag_2026.tsv"
    cache_dir = tmp_path / "cache"
    fake_client = FakeClient()

    rows = run_bm25_retrieval(
        topics,
        client=fake_client,
        output_path=output_path,
        cache_dir=cache_dir,
        run_id="pyserini_climbmix_bm25_top100",
        depth=100,
    )

    assert fake_client.queries == ["Use the narrative as part of the BM25 query."]
    assert rows == [
        RetrievalRow(
            topic_id="31",
            docid="doc-a",
            rank=1,
            score=9.0,
            run_id="pyserini_climbmix_bm25_top100",
        ),
        RetrievalRow(
            topic_id="31",
            docid="doc-b",
            rank=2,
            score=7.5,
            run_id="pyserini_climbmix_bm25_top100",
        ),
    ]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "31 Q0 doc-a 1 9.0 pyserini_climbmix_bm25_top100",
        "31 Q0 doc-b 2 7.5 pyserini_climbmix_bm25_top100",
    ]
    assert json.loads((cache_dir / "31.json").read_text(encoding="utf-8")) == {
        "query": "Use the narrative as part of the BM25 query.",
        "response": {
            "candidates": [
                {"rank": 2, "docid": "doc-b", "score": 7.5, "doc": {"contents": "second"}},
                {"rank": 1, "docid": "doc-a", "score": 9.0, "doc": {"contents": "first"}},
            ]
        },
    }


def test_validate_retrieval_run_accepts_valid_trec_rows(tmp_path):
    runfile = tmp_path / "run.tsv"
    runfile.write_text(
        "31 Q0 doc-a 1 9.0 bm25\n"
        "31 Q0 doc-b 2 7.5 bm25\n"
        "32 Q0 doc-c 1 8.0 bm25\n",
        encoding="utf-8",
    )

    validate_retrieval_run(runfile, expected_topic_ids=["31", "32"])


def test_validate_retrieval_run_rejects_missing_topics_duplicate_docids_and_bad_ranks(tmp_path):
    runfile = tmp_path / "run.tsv"
    runfile.write_text(
        "31 Q0 doc-a 1 9.0 bm25\n"
        "31 Q0 doc-a 2 8.0 bm25\n"
        "31 Q0 doc-b 4 7.0 bm25\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        validate_retrieval_run(runfile, expected_topic_ids=["31", "32"])

    message = str(exc_info.value)
    assert "missing topics: 32" in message
    assert "duplicate docid doc-a for topic 31" in message
    assert "topic 31 ranks must be contiguous from 1" in message


def test_validate_retrieval_run_rejects_malformed_rows(tmp_path):
    runfile = tmp_path / "run.tsv"
    runfile.write_text("31 Q0 doc-a 1 9.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1.*six columns"):
        validate_retrieval_run(runfile, expected_topic_ids=["31"])

import json

import pytest

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


def test_load_topics_honors_explicit_format_independent_of_suffix(tmp_path):
    topics_path = tmp_path / "topics.data"
    topics_path.write_text("31\tExplain e-waste impacts.\n", encoding="utf-8")

    assert load_topics(topics_path, topic_format="tsv") == [
        Topic(
            id="31",
            title="Explain e-waste impacts",
            narrative="Explain e-waste impacts.",
        )
    ]


def test_load_topics_rejects_unknown_explicit_format(tmp_path):
    topics_path = tmp_path / "topics.data"
    topics_path.write_text("31\tExplain e-waste impacts.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported topic format"):
        load_topics(topics_path, topic_format="csv")


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

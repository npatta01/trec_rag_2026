from __future__ import annotations

from hashlib import sha256

import pytest

from trec_rag.organizer_pi_inputs import (
    OrganizerTopic,
    select_topic,
    sha256_file,
    write_topic_run,
    write_topic_tsv,
)


def test_select_topic_preserves_exact_narrative(tmp_path):
    source = tmp_path / "queries.tsv"
    source.write_text("rag2026-0\tfirst\nrag2026-1\tline one\tline two\n", encoding="utf-8")

    topic = select_topic(source, "rag2026-1")

    assert topic == OrganizerTopic("rag2026-1", "line one\tline two")


@pytest.mark.parametrize(
    ("body", "topic_id"),
    [
        ("rag2026-0\tfirst\n", "rag2026-1"),
        ("rag2026-1\tfirst\nrag2026-1\tsecond\n", "rag2026-1"),
        ("rag2026-1\t   \n", "rag2026-1"),
        ("rag2026-1\n", "rag2026-1"),
    ],
    ids=["missing", "duplicate", "empty-narrative", "missing-narrative-column"],
)
def test_select_topic_rejects_invalid_topic_matches(tmp_path, body, topic_id):
    source = tmp_path / "queries.tsv"
    source.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="expected exactly one non-empty topic"):
        select_topic(source, topic_id)


def test_write_topic_tsv_writes_one_row_with_untouched_narrative(tmp_path):
    output = write_topic_tsv(
        OrganizerTopic("rag2026-1", "line one\tline two"), tmp_path / "one.tsv"
    )

    assert output == tmp_path / "one.tsv"
    assert output.read_bytes() == b"rag2026-1\tline one\tline two\n"


def test_write_topic_run_keeps_rank_order_and_exactly_expected_depth(tmp_path):
    source = tmp_path / "run.trec"
    source.write_text(
        "rag2026-1 Q0 d2 2 8.0 tag\n"
        "rag2026-0 Q0 x 1 9.0 tag\n"
        "rag2026-1 Q0 d1 1 9.0 tag\n",
        encoding="utf-8",
    )

    output = write_topic_run(source, "rag2026-1", tmp_path / "one.trec", expected_depth=2)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "rag2026-1 Q0 d1 1 9.0 tag",
        "rag2026-1 Q0 d2 2 8.0 tag",
    ]


def test_write_topic_run_preserves_matching_row_bytes_when_sorting(tmp_path):
    source = tmp_path / "run.trec"
    source.write_bytes(
        b"rag2026-1\tQ0\td2\t2\t8.0\ttag\r\n"
        b"rag2026-1  Q0  d1  1  9.0  tag\r\n"
    )

    output = write_topic_run(source, "rag2026-1", tmp_path / "one.trec", expected_depth=2)

    assert output.read_bytes() == (
        b"rag2026-1  Q0  d1  1  9.0  tag\r\n"
        b"rag2026-1\tQ0\td2\t2\t8.0\ttag\r\n"
    )


@pytest.mark.parametrize(
    ("body", "expected_depth"),
    [
        ("rag2026-1 Q0 d1 1 9 tag extra\n", 1),
        ("rag2026-1 Q0 d1 0 9 tag\n", 1),
        ("rag2026-1 Q0 d1 -1 9 tag\n", 1),
        ("rag2026-1 Q0 d1 1 9 tag\nrag2026-1 Q0 d2 1 8 tag\n", 2),
        ("rag2026-1 Q0 d1 1 9 tag\nrag2026-1 Q0 d2 3 8 tag\n", 2),
        ("rag2026-1 Q0 d1 1 9 tag\nrag2026-1 Q0 d1 2 8 tag\n", 2),
        ("rag2026-1 Q0 d1 1 9 tag\n", 2),
    ],
    ids=[
        "malformed-columns",
        "zero-rank",
        "negative-rank",
        "duplicate-rank",
        "non-contiguous-ranks",
        "duplicate-docid",
        "wrong-depth",
    ],
)
def test_write_topic_run_rejects_invalid_matching_rows_without_replacing_output(
    tmp_path, body, expected_depth
):
    source = tmp_path / "run.trec"
    output = tmp_path / "one.trec"
    source.write_text(body, encoding="utf-8")
    output.write_bytes(b"existing output\n")

    with pytest.raises(ValueError):
        write_topic_run(source, "rag2026-1", output, expected_depth=expected_depth)

    assert output.read_bytes() == b"existing output\n"


def test_sha256_file_hashes_exact_file_bytes(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"line one\r\nline two\x00")

    assert sha256_file(source) == sha256(b"line one\r\nline two\x00").hexdigest()

"""Tests for the RAGDoll adapters and the archived development-topic input builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import trec_rag.competition_rag as competition_rag
import trec_rag.dev_rag_inputs as dev_rag_inputs
import trec_rag.ragdoll_io as ragdoll_io

NARRATIVE = "I want to understand nuclear energy tradeoffs."


def _submission_record(qid: str = "58") -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": "local-baseline",
            "narrative_id": qid,
            "narrative": NARRATIVE,
            "run_id": "dev-spike",
            "run_desc": "development spike",
        },
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {"text": "Nuclear power emits little carbon.", "citations": [0]},
            {"text": "Waste storage remains unresolved.", "citations": [1]},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows), encoding="utf-8"
    )
    return path


def _gold_record(qid: str = "58", count: int = 3) -> dict[str, Any]:
    return {
        "qid": qid,
        "nuggets": [
            {
                "text": f"gold nugget {index}",
                # Released values really do carry literal embedded quotes.
                "mapped_sub_narrative": '"Key advantages of nuclear energy"',
                "importance": "vital" if index % 2 == 0 else "okay",
                "source": "post-edit",
            }
            for index in range(count)
        ],
    }


def test_answer_rows_expose_qid_that_ragdoll_can_resolve(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])

    rows = ragdoll_io.answer_rows(submission)

    assert len(rows) == 1
    payload = rows[0].as_dict()
    # RAGDoll reads qid/topic_id/query_id only; metadata.narrative_id is invisible to it.
    assert payload["qid"] == "58"
    assert payload["topic_id"] == "58"
    assert payload["answer_text"] == (
        "Nuclear power emits little carbon. Waste storage remains unresolved."
    )
    assert payload["response_length"] == 9


def test_derived_answers_never_mutate_the_submission_record(tmp_path: Path) -> None:
    record = _submission_record()
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    before = submission.read_bytes()

    ragdoll_io.answer_rows(submission)

    assert submission.read_bytes() == before
    # The organizer contract allows exactly these three root keys.
    reloaded = json.loads(before.decode("utf-8").splitlines()[0])
    assert set(reloaded) == {"metadata", "references", "answer"}
    competition_rag.validate_submission_record(
        reloaded,
        topic_id="58",
        narrative=NARRATIVE,
        allowed_docids=["climbmix-a", "climbmix-b"],
        team_id="local-baseline",
        run_id="dev-spike",
        run_desc="development spike",
    )


def test_gold_nuggets_are_reshaped_without_losing_importance(tmp_path: Path) -> None:
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(count=4)])

    rows = ragdoll_io.gold_nugget_rows(nuggets, narratives={"58": NARRATIVE})

    assert len(rows) == 1
    reshaped = rows[0]["nuggets"]
    assert len(reshaped) == 4
    assert [nugget["importance"] for nugget in reshaped] == ["vital", "okay", "vital", "okay"]
    # mapped_sub_narrative and source are intentionally dropped.
    assert all(set(nugget) == {"text", "importance"} for nugget in reshaped)
    assert rows[0]["query"] == NARRATIVE


def test_assert_join_rejects_a_silently_empty_join(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record(qid="58")])
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(qid="999")])

    answers = ragdoll_io.answer_rows(submission)
    gold = ragdoll_io.gold_nugget_rows(nuggets, narratives={"999": NARRATIVE})

    with pytest.raises(ValueError, match="share no topic ids"):
        ragdoll_io.assert_join(answers, gold)


def test_missing_gold_nuggets_for_a_requested_topic_is_an_error(tmp_path: Path) -> None:
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(qid="58")])

    with pytest.raises(ValueError, match="no gold nuggets for 213"):
        ragdoll_io.gold_nugget_rows(
            nuggets, narratives={"58": NARRATIVE, "213": NARRATIVE}, topic_ids=["58", "213"]
        )


def _documents_file(path: Path, docids: list[str]) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "query": {"qid": "58"},
                "candidates": [
                    {"docid": docid, "doc": f"text supporting {docid}"} for docid in docids
                ],
            }
        ],
    )


def test_support_rows_carry_segments_for_every_cited_docid(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    documents = _documents_file(tmp_path / "documents.jsonl", ["climbmix-a", "climbmix-b"])

    rows = ragdoll_io.resolved_support_rows(submission, documents)

    assert len(rows) == 1
    row = rows[0]
    # ragdoll support skips any citation whose docid is absent from segments.
    assert set(row["segments"]) == {"climbmix-a", "climbmix-b"}
    assert row["topic_id"] == "58"
    assert row["run_id"] == "dev-spike"
    assert row["answer"] == _submission_record()["answer"]


def test_support_rows_reject_a_reference_without_document_text(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    documents = _documents_file(tmp_path / "documents.jsonl", ["climbmix-a"])

    with pytest.raises(ValueError, match="climbmix-b"):
        ragdoll_io.resolved_support_rows(submission, documents)


def test_support_rows_reject_a_citation_outside_references(tmp_path: Path) -> None:
    """A foreign docid citation would be silently dropped from RAGDoll's denominator."""
    record = _submission_record()
    record["answer"][0]["citations"] = ["climbmix-elsewhere"]
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    documents = _documents_file(tmp_path / "documents.jsonl", ["climbmix-a", "climbmix-b"])

    with pytest.raises(ValueError, match="not in references"):
        ragdoll_io.resolved_support_rows(submission, documents)


def test_select_passages_counts_the_joiner_against_the_budget() -> None:
    document = (
        ("alpha beta gamma " * 60)
        + ("nuclear fission uranium reactor " * 30)
        + ("delta epsilon " * 80)
    )
    for budget in (50, 240, 300):
        out = dev_rag_inputs.select_passages(
            document, "nuclear fission uranium", budget_words=budget
        )
        assert len(out.split()) <= budget, budget


def test_select_passages_honours_a_budget_smaller_than_one_window() -> None:
    document = ("filler words here " * 100) + ("nuclear fission uranium " * 20)
    out = dev_rag_inputs.select_passages(document, "nuclear fission", budget_words=40)
    # Previously every window was rejected and the document opening was returned instead.
    assert "fission" in out
    assert len(out.split()) <= 40


def test_select_passages_validates_window_parameters() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        dev_rag_inputs.select_passages("a b c", "a", budget_words=10, window_words=0)
    with pytest.raises(ValueError, match="must be positive"):
        dev_rag_inputs.select_passages("a b c", "a", budget_words=10, stride_words=0)


def _archive(topic_id: str, count: int) -> dict[str, Any]:
    return {
        "topic_id": topic_id,
        "hits": count,
        "response": {
            "query": {"text": NARRATIVE},
            "candidates": [
                {
                    "docid": f"shard_{index:05d}",
                    "rank": index + 1,
                    "score": 50.0 - index,
                    "doc": f"document text {index}",
                }
                for index in range(count)
            ],
        },
    }


def test_archive_builds_a_run_competition_rag_accepts(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "58__original__climbmix_bm25__abc123.json").write_text(
        json.dumps(_archive("58", 5)), encoding="utf-8"
    )
    run_path = tmp_path / "run.tsv"
    documents_path = tmp_path / "documents.jsonl"

    dev_rag_inputs.build(
        topic_ids=["58"],
        cache_dir=cache_dir,
        run_path=run_path,
        documents_path=documents_path,
        run_id="dev-spike",
        depth=3,
    )

    selected = competition_rag.load_trec_run(run_path, {"58"}, 3)
    assert selected["58"] == ["shard_00000", "shard_00001", "shard_00002"]
    documents = competition_rag.load_documents(
        documents_path, None, set(selected["58"]), 1000
    )
    assert documents["shard_00000"] == "document text 0"


def test_duplicate_docids_are_collapsed_and_ranks_stay_dense(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    archive = _archive("58", 3)
    # A repeated docid at a worse rank must not produce a rank gap.
    archive["response"]["candidates"].append(
        {"docid": "shard_00000", "rank": 4, "score": 10.0, "doc": "duplicate"}
    )
    (cache_dir / "58__original__climbmix_bm25__abc123.json").write_text(
        json.dumps(archive), encoding="utf-8"
    )
    run_path = tmp_path / "run.tsv"

    dev_rag_inputs.build(
        topic_ids=["58"],
        cache_dir=cache_dir,
        run_path=run_path,
        documents_path=tmp_path / "documents.jsonl",
        run_id="dev-spike",
        depth=10,
    )

    ranks = [int(line.split()[3]) for line in run_path.read_text().splitlines()]
    assert ranks == [1, 2, 3]


def test_select_passages_prefers_relevant_windows_over_the_document_opening() -> None:
    boilerplate = "Menu Home About Contact Subscribe Newsletter " * 40
    evidence = "Nuclear fission splits uranium nuclei and releases heat for steam. " * 6
    filler = "unrelated commentary about gardening " * 80
    document = boilerplate + evidence + filler

    selected = dev_rag_inputs.select_passages(
        document, "nuclear fission uranium", budget_words=150
    )

    assert len(selected.split()) <= 150
    assert "fission" in selected and "uranium" in selected
    # Head truncation would have returned only the navigation block.
    assert "fission" not in " ".join(document.split()[:150])


def test_select_passages_returns_short_documents_untouched() -> None:
    document = "A short document about nuclear power."
    assert (
        dev_rag_inputs.select_passages(document, "nuclear", budget_words=100) == document
    )


def test_select_passages_falls_back_when_no_query_term_matches() -> None:
    document = " ".join(f"word{i}" for i in range(400))
    out = dev_rag_inputs.select_passages(document, "nuclear fission", budget_words=50)
    # No window matches, so the opening is used rather than returning nothing.
    assert out.split() == document.split()[:50]


def test_document_rows_apply_the_passage_budget(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    long_text = "Nuclear fission releases heat. " + ("padding words here " * 500)
    archive = {
        "topic_id": "58",
        "response": {
            "query": {"text": "nuclear fission"},
            "candidates": [
                {"docid": "shard_1", "rank": 1, "score": 9.0, "doc": long_text}
            ],
        },
    }
    (cache_dir / "58__original__climbmix_bm25__abc.json").write_text(
        json.dumps(archive), encoding="utf-8"
    )
    documents_path = tmp_path / "documents.jsonl"

    dev_rag_inputs.build(
        topic_ids=["58"],
        cache_dir=cache_dir,
        run_path=tmp_path / "run.tsv",
        documents_path=documents_path,
        run_id="dev-spike",
        depth=1,
        passage_words=60,
    )

    row = json.loads(documents_path.read_text().splitlines()[0])
    assert len(row["candidates"][0]["doc"].split()) <= 60


def test_missing_archive_names_the_topic(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(ValueError, match="213: no archived retrieval response"):
        dev_rag_inputs.archive_path(cache_dir, "213")

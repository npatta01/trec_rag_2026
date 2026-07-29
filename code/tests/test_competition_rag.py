from __future__ import annotations

import json
import os
import stat
import threading
import time
import zipfile
import asyncio
import copy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import trec_rag.competition_rag as competition_rag
from trec_rag.competition_rag import (
    OpenRouterJsonGenerator,
    RagGenerationConfig,
    arguments,
    build_submission_record,
    load_documents,
    load_queries,
    load_rag_generation_config,
    load_trec_run,
    parse_generated_json,
    run_generation,
    select_queries,
    validate_submission_record,
)


def _config_text(*, inputs_extra: str = "", root_extra: str = "") -> str:
    return f"""schema_version: competition_rag_config_v1
experiment:
  id: rag26_competition_rag_gpt_sol_v1
  output_dir: outputs/rag26_competition_rag_gpt_sol_v1
  mode: create
submission:
  team_id: castorini
  run_desc: Fixed retrieval with GPT-5.6 Sol answer generation.
inputs:
  queries: official/topics.tsv
  run: outputs/facet-deepseek-b40-v1/r_output_trec_rag_2026.tsv
  documents: outputs/facet-deepseek-b40-v1/retrieval_with_text.jsonl.zip
  archive_member: retrieval_with_text.jsonl
{inputs_extra}retrieval:
  top_k: 100
  max_document_words: 1000
generation:
  type: openrouter
  api_base: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: openai/gpt-5.6-sol
  reasoning_effort: medium
  temperature: null
  max_tokens: 6000
  timeout_seconds: 900
  transport_max_attempts: 3
  concurrency: 4
{root_extra}"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "competition-rag.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_strict_generation_config_and_resolves_inputs_from_checkout(
    tmp_path: Path,
) -> None:
    (tmp_path / "official").mkdir()
    (tmp_path / "official/topics.tsv").write_text("rag2026-0\tQuestion\n", encoding="utf-8")
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))

    assert config.schema_version == "competition_rag_config_v1"
    assert config.queries_path == tmp_path / "official/topics.tsv"
    assert config.run_path == tmp_path / "outputs/facet-deepseek-b40-v1/r_output_trec_rag_2026.tsv"
    assert config.documents_path == tmp_path / "outputs/facet-deepseek-b40-v1/retrieval_with_text.jsonl.zip"
    assert config.topic_ids is None
    assert config.output_path == (
        tmp_path / "outputs/rag26_competition_rag_gpt_sol_v1/rag_output_trec_rag_2026.jsonl"
    )
    assert config.run_id == "rag26_competition_rag_gpt_sol_v1"


@pytest.mark.parametrize(
    "output_dir",
    [
        "/tmp/competition-rag-output",
        "../competition-rag-output",
        "outputs/../competition-rag-output",
        "outputs",
    ],
)
def test_config_rejects_output_directories_outside_a_dedicated_outputs_child(
    tmp_path: Path, output_dir: str
) -> None:
    content = _config_text().replace(
        "  output_dir: outputs/rag26_competition_rag_gpt_sol_v1\n",
        f"  output_dir: {output_dir}\n",
    )

    with pytest.raises(ValueError, match="experiment.output_dir"):
        load_rag_generation_config(_write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_config_text().replace("schema_version: competition_rag_config_v1\n", ""), "schema_version"),
        (_config_text().replace("competition_rag_config_v1", "other_schema"), "schema_version"),
        (_config_text(root_extra="unexpected: true\n"), "unknown"),
        (_config_text().replace("submission:\n  team_id: castorini\n  run_desc: Fixed retrieval with GPT-5.6 Sol answer generation.\n", ""), "missing"),
        (_config_text().replace("  concurrency: 4\n", "  concurrency: 4\n  concurrency: 2\n"), "duplicate YAML key"),
    ],
)
def test_config_rejects_missing_unknown_wrong_schema_and_duplicate_keys(
    tmp_path: Path, content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_rag_generation_config(_write_config(tmp_path, content))


def test_topic_selection_is_unique_known_and_in_canonical_tsv_order(tmp_path: Path) -> None:
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text(
        "rag2026-2\tSecond official question\n"
        "rag2026-0\tFirst official question\n"
        "rag2026-1\tThird official question\n",
        encoding="utf-8",
    )
    queries = load_queries(topics_path)

    assert select_queries(queries, ["rag2026-1", "rag2026-2"]) == [
        ("rag2026-2", "Second official question"),
        ("rag2026-1", "Third official question"),
    ]
    assert select_queries(queries, None) == queries

    for topic_ids, message in [
        ([], "at least one"),
        (["rag2026-1", "rag2026-1"], "duplicate"),
        (["not-an-official-topic"], "unknown"),
    ]:
        with pytest.raises(ValueError, match=message):
            select_queries(queries, topic_ids)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("qid\tnarrative\nrag2026-0\tQuestion\n", "header"),
        ("rag2026-0\tQuestion\tUnexpected third field\n", "exactly two"),
        ("rag2026-0\n", "exactly two"),
    ],
)
def test_topics_are_exactly_two_headerless_tsv_fields(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_queries(path)


def test_topic_narrative_preserves_exact_tsv_field_content(tmp_path: Path) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text("rag2026-0\t  Exact narrative field content  \n", encoding="utf-8")

    assert load_queries(path) == [
        ("rag2026-0", "  Exact narrative field content  ")
    ]


def test_topic_ids_reject_noncanonical_surrounding_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text(" rag2026-0\tExact narrative\n", encoding="utf-8")

    with pytest.raises(ValueError, match="topic id"):
        load_queries(path)


def test_trec_run_requires_organizer_six_field_ranked_rows(tmp_path: Path) -> None:
    path = tmp_path / "run.tsv"
    path.write_text(
        "rag2026-0 Q0 climbmix-a 1 9.25 facet-deepseek-b40-v1\n"
        "rag2026-0 Q0 climbmix-b 2 8.75 facet-deepseek-b40-v1\n"
        "rag2026-1 Q0 climbmix-c 1 7.00 facet-deepseek-b40-v1\n",
        encoding="utf-8",
    )

    assert load_trec_run(path, {"rag2026-0", "rag2026-1"}, top_k=1) == {
        "rag2026-0": ["climbmix-a"],
        "rag2026-1": ["climbmix-c"],
    }


def test_trec_run_accepts_increasing_noncontiguous_variable_depth(tmp_path: Path) -> None:
    path = tmp_path / "run.tsv"
    path.write_text(
        "rag2026-0 Q0 climbmix-a 1 9.25 facet-deepseek-b40-v1\n"
        "rag2026-0 Q0 climbmix-b 3 8.75 facet-deepseek-b40-v1\n",
        encoding="utf-8",
    )

    assert load_trec_run(path, {"rag2026-0"}, top_k=None) == {
        "rag2026-0": ["climbmix-a", "climbmix-b"]
    }


def test_trec_run_rejects_rows_out_of_raw_rank_order(tmp_path: Path) -> None:
    path = tmp_path / "run.tsv"
    path.write_text(
        "rag2026-0 Q0 climbmix-b 2 8.75 facet-deepseek-b40-v1\n"
        "rag2026-0 Q0 climbmix-a 1 9.25 facet-deepseek-b40-v1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="order"):
        load_trec_run(path, {"rag2026-0"}, top_k=None)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("rag2026-0 Q0 doc 1 1.0 run extra\n", "six"),
        ("rag2026-0 Q1 doc 1 1.0 run\n", "Q0"),
        ("rag2026-0 Q0 doc 0 1.0 run\n", "positive"),
        ("rag2026-0 Q0 doc-a 2 2.0 run\n", "start at 1"),
        ("rag2026-0 Q0 doc-a 1 2.0 run\nrag2026-0 Q0 doc-b 1 1.0 run\n", "duplicate rank"),
        ("rag2026-0 Q0 doc-a 1 2.0 run\nrag2026-0 Q0 doc-a 2 1.0 run\n", "duplicate docid"),
        ("rag2026-0 Q0 doc-a 1 score run\n", "score"),
        ("rag2026-0 Q0 doc-a 1 1.0 run\nrag2026-0 Q0 doc-b 2 2.0 run\n", "non-increasing"),
        ("rag2026-0 Q0 doc-a 1 2.0 one\nrag2026-1 Q0 doc-b 1 1.0 two\n", "run tag"),
        ("rag2026-0 Q0 doc-a 1 2.0 \n", "six"),
    ],
)
def test_trec_run_rejects_non_organizer_rows(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "run.tsv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_trec_run(path, {"rag2026-0", "rag2026-1"}, top_k=None)


def test_loads_organizer_document_zip_with_local_extension_fields(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            json.dumps(
                {
                    "query": {"qid": "rag2026-0", "text": "Official question", "local_note": "kept"},
                    "candidates": [
                        {"docid": "climbmix-a", "doc": "First  document text", "stage": "canonical_supported"},
                        {"docid": "climbmix-b", "doc": "Second document text", "score": 8.75},
                    ],
                    "receipt_sha256": "local extension",
                }
            )
            + "\n",
        )

    assert load_documents(
        archive_path,
        archive_member=None,
        wanted_docids={"climbmix-a", "climbmix-b"},
        max_words=2,
    ) == {"climbmix-a": "First document", "climbmix-b": "Second document"}


def test_loads_qid_only_organizer_query_core(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            json.dumps(
                {
                    "query": {"qid": "rag2026-0"},
                    "candidates": [{"docid": "climbmix-a", "doc": "Document text."}],
                }
            )
            + "\n",
        )

    assert load_documents(
        archive_path,
        archive_member=None,
        wanted_docids={"climbmix-a"},
        max_words=100,
    ) == {"climbmix-a": "Document text."}


def test_document_zip_rejects_missing_or_wrong_archive_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    row = json.dumps(
        {
            "query": {"qid": "rag2026-0"},
            "candidates": [{"docid": "climbmix-a", "doc": "Document text."}],
        }
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("first.jsonl", row + "\n")
        archive.writestr("second.jsonl", row + "\n")

    with pytest.raises(ValueError, match="archive_member"):
        load_documents(archive_path, None, {"climbmix-a"}, 100)
    with pytest.raises(ValueError, match="archive_member"):
        load_documents(archive_path, "missing.jsonl", {"climbmix-a"}, 100)


def test_document_zip_rejects_malformed_json_with_line_context(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("retrieval_with_text.jsonl", "{not-json}\n")

    with pytest.raises(ValueError, match=r"retrieval_with_text.*:1: invalid JSON"):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


def test_document_zip_normalizes_corrupt_archive_failure(tmp_path: Path) -> None:
    archive_path = tmp_path / "corrupt-documents.zip"
    archive_path.write_bytes(b"this is not a ZIP archive")

    with pytest.raises(
        ValueError,
        match=r"corrupt-documents\.zip: invalid ZIP document archive",
    ):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


def test_document_zip_normalizes_non_utf8_member_failure(tmp_path: Path) -> None:
    archive_path = tmp_path / "non-utf8-documents.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("retrieval_with_text.jsonl", b"\xff\xfe\x80")

    with pytest.raises(
        ValueError,
        match=r"non-utf8-documents\.zip: selected document member is not valid UTF-8",
    ):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


@pytest.mark.parametrize(
    "row",
    [
        {"candidates": [{"docid": "climbmix-a", "doc": "Document text."}]},
        {"query": {}, "candidates": [{"docid": "climbmix-a", "doc": "Document text."}]},
        {"query": {"qid": "rag2026-0"}},
        {"query": {"qid": "rag2026-0"}, "candidates": [{"doc": "Document text."}]},
        {"query": {"qid": "rag2026-0"}, "candidates": [{"docid": "climbmix-a"}]},
    ],
)
def test_document_zip_rejects_missing_required_organizer_core_fields(
    tmp_path: Path, row: dict[str, object]
) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("retrieval_with_text.jsonl", json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="organizer|candidate"):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


def test_document_zip_rejects_conflicting_duplicate_document_text(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    rows = [
        {
            "query": {"qid": "rag2026-0"},
            "candidates": [{"docid": "climbmix-a", "doc": "First text."}],
        },
        {
            "query": {"qid": "rag2026-1"},
            "candidates": [{"docid": "climbmix-a", "doc": "Conflicting text."}],
        },
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )

    with pytest.raises(ValueError, match="conflicting duplicate document climbmix-a"):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


def test_document_zip_rejects_missing_wanted_docids(tmp_path: Path) -> None:
    archive_path = tmp_path / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            json.dumps(
                {
                    "query": {"qid": "rag2026-0"},
                    "candidates": [{"docid": "climbmix-other", "doc": "Other text."}],
                }
            )
            + "\n",
        )

    with pytest.raises(ValueError, match="missing 1 ranked documents.*climbmix-a"):
        load_documents(archive_path, None, {"climbmix-a"}, 100)


def test_checked_in_competition_config_selects_every_official_topic() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_rag_generation_config(
        repo_root / "configs/rag26_competition_rag_gpt_sol_v1.yaml"
    )
    queries = load_queries(config.queries_path)

    assert config.topic_ids is None
    assert config.run_path == repo_root / "outputs/facet-deepseek-b40-v1/r_output_trec_rag_2026.tsv"
    assert config.documents_path == repo_root / "outputs/facet-deepseek-b40-v1/retrieval_with_text.jsonl.zip"
    assert len(select_queries(queries, config.topic_ids)) == 119


def _generated_record() -> dict[str, Any]:
    return {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {"text": "The first source supports the finding.", "citations": [0]},
            {"text": "The second source adds a limitation.", "citations": [1]},
        ],
    }


def _submission_record() -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": "castorini",
            "narrative_id": "rag2026-0",
            "narrative": "Official question",
            "run_id": "rag26_competition_rag_gpt_sol_v1",
            "run_desc": "Fixed retrieval with GPT-5.6 Sol answer generation.",
        },
        **_generated_record(),
    }


def _validate_submission(record: dict[str, Any]) -> None:
    validate_submission_record(
        record,
        topic_id="rag2026-0",
        narrative="Official question",
        allowed_docids=["climbmix-a", "climbmix-b", "climbmix-c"],
        team_id="castorini",
        run_id="rag26_competition_rag_gpt_sol_v1",
        run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
    )


def test_validates_organizer_shaped_submission_record() -> None:
    _validate_submission(_submission_record())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update({"extra": True}), "root object or metadata"),
        (lambda row: row["metadata"].update({"model": "gpt"}), "root object or metadata"),
        (lambda row: row["references"].append("climbmix-a"), "duplicated or outside"),
        (lambda row: row["references"].append("not-selected"), "duplicated or outside"),
        (lambda row: row["answer"][0].update({"citations": [True]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [0, 0]}), "unique citations"),
        (lambda row: row["answer"][0].update({"citations": [2]}), "citation"),
        (lambda row: row["answer"][0].update({"heading": "Finding"}), "invalid fields"),
        (lambda row: row.update({"answer": row["answer"][:1]}), "uncited references"),
    ],
)
def test_rejects_submission_records_outside_the_organizer_contract(
    mutate: Any, message: str
) -> None:
    record = copy.deepcopy(_submission_record())
    mutate(record)

    with pytest.raises(ValueError, match=message):
        _validate_submission(record)


def test_rejects_answer_over_1024_whitespace_words() -> None:
    record = _submission_record()
    record["references"] = ["climbmix-a"]
    record["answer"] = [{"text": "word " * 1025, "citations": [0]}]

    with pytest.raises(ValueError, match="1,024 words"):
        _validate_submission(record)


@pytest.mark.parametrize("extra_key", ["metadata", "model_note", "unexpected"])
def test_generated_root_must_contain_exactly_references_and_answer(
    extra_key: str,
) -> None:
    generated = _generated_record()
    generated[extra_key] = {"ignored": "must be rejected before metadata injection"}

    with pytest.raises(ValueError, match="generated root"):
        build_submission_record(
            generated,
            topic_id="rag2026-0",
            narrative="Official question",
            team_id="castorini",
            run_id="rag26_competition_rag_gpt_sol_v1",
            run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
        )


def test_build_submission_record_injects_exact_official_metadata() -> None:
    record = build_submission_record(
        _generated_record(),
        topic_id="rag2026-0",
        narrative="Official question",
        team_id="castorini",
        run_id="rag26_competition_rag_gpt_sol_v1",
        run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
    )

    assert list(record) == ["metadata", "references", "answer"]
    assert list(record["metadata"]) == [
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    ]


def test_parses_only_one_plain_or_fenced_json_object() -> None:
    assert parse_generated_json('{"references": [], "answer": []}') == {
        "references": [],
        "answer": [],
    }
    assert parse_generated_json('```json\n{"references": [], "answer": []}\n```') == {
        "references": [],
        "answer": [],
    }
    with pytest.raises(ValueError, match="one JSON object"):
        parse_generated_json('before {"answer": []} after')


class FakeGenerator:
    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append(
            {
                "topic_id": topic_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
            }
        )
        return copy.deepcopy(self.outputs[topic_id]), {"id": f"fake-{topic_id}"}


def _pipeline_config(tmp_path: Path) -> RagGenerationConfig:
    (tmp_path / "official").mkdir()
    (tmp_path / "official/topics.tsv").write_text(
        "rag2026-2\tQuestion two\nrag2026-1\tQuestion one\n",
        encoding="utf-8",
    )
    inputs = tmp_path / "outputs/facet-deepseek-b40-v1"
    inputs.mkdir(parents=True)
    (inputs / "r_output_trec_rag_2026.tsv").write_text(
        "rag2026-1 Q0 climbmix-a 1 9.0 bm25\n"
        "rag2026-1 Q0 climbmix-b 2 8.0 bm25\n"
        "rag2026-2 Q0 climbmix-c 1 7.0 bm25\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(inputs / "retrieval_with_text.jsonl.zip", "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            json.dumps(
                {
                    "query": {"qid": "rag2026-1"},
                    "candidates": [
                        {"docid": "climbmix-a", "doc": "Evidence A."},
                        {"docid": "climbmix-b", "doc": "Evidence B."},
                    ],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "query": {"qid": "rag2026-2"},
                    "candidates": [{"docid": "climbmix-c", "doc": "Evidence C."}],
                }
            )
            + "\n",
        )
    return load_rag_generation_config(_write_config(tmp_path, _config_text()))


def _topic_output(docids: list[str], text: str) -> dict[str, Any]:
    return {
        "references": docids,
        "answer": [{"text": text, "citations": list(range(len(docids)))}],
    }


def test_runs_generation_in_official_query_order_and_atomically_consolidates(
    tmp_path: Path,
) -> None:
    config = _pipeline_config(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a", "climbmix-b"], "A and B support it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    rows = [json.loads(line) for line in config.output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["narrative_id"] for row in rows] == ["rag2026-2", "rag2026-1"]
    assert rows[0]["answer"][0]["citations"] == [0]
    assert {call["topic_id"] for call in generator.calls} == {"rag2026-1", "rag2026-2"}
    topic_one_prompt = next(call["user_prompt"] for call in generator.calls if call["topic_id"] == "rag2026-1")
    assert "[1] docid: climbmix-a" in topic_one_prompt
    assert "[2] docid: climbmix-b" in topic_one_prompt


def test_resume_reuses_valid_rows_without_model_calls(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    first = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )
    asyncio.run(run_generation(config, first))

    resumed = FakeGenerator({})
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert resumed.calls == []
    assert config.output_path.exists()


def test_generation_limits_parallel_model_calls_to_configured_concurrency(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), concurrency=1)

    class CountingGenerator(FakeGenerator):
        def __init__(self) -> None:
            super().__init__(
                {
                    "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
                    "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
                }
            )
            self._lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0

        def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            with self._lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.02)
                return super().complete_json(**kwargs)
            finally:
                with self._lock:
                    self.active -= 1

    generator = CountingGenerator()
    asyncio.run(run_generation(config, generator))

    assert generator.maximum_active == 1


def test_overwrite_clears_only_generation_state_before_failure_then_resume_regenerates(
    tmp_path: Path,
) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Old one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Old two."),
        }
    )
    asyncio.run(run_generation(config, initial))
    retrieval_bytes = config.run_path.read_bytes()
    documents_bytes = config.documents_path.read_bytes()

    with pytest.raises(RuntimeError, match="2 topic"):
        asyncio.run(run_generation(replace(config, overwrite=True), FakeGenerator({})))

    assert not config.output_path.exists()
    assert retrieval_bytes == config.run_path.read_bytes()
    assert documents_bytes == config.documents_path.read_bytes()

    resumed = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Fresh one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Fresh two."),
        }
    )
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert {call["topic_id"] for call in resumed.calls} == {"rag2026-1", "rag2026-2"}
    assert "Old one." not in config.output_path.read_text(encoding="utf-8")
    assert "Fresh one." in config.output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "overlap",
    [
        "queries_equal_output",
        "run_inside_work",
        "documents_contains_work",
        "documents_contains_output",
    ],
)
def test_overwrite_rejects_input_and_deletion_target_overlap_before_deleting(
    tmp_path: Path, overlap: str
) -> None:
    config = _pipeline_config(tmp_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text("old output\n", encoding="utf-8")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    sentinel = config.work_dir / "old-row.json"
    sentinel.write_text("old row\n", encoding="utf-8")
    original_queries = config.queries_path.read_bytes()
    original_run = config.run_path.read_bytes()

    if overlap == "queries_equal_output":
        config = replace(config, output_path=config.queries_path, overwrite=True)
    elif overlap == "run_inside_work":
        nested_run = config.work_dir / "run.trec"
        nested_run.write_bytes(original_run)
        config = replace(config, run_path=nested_run, overwrite=True)
    elif overlap == "documents_contains_work":
        container = tmp_path / "input-container"
        container.mkdir()
        nested_work = container / "work"
        nested_work.mkdir()
        (nested_work / "old-row.json").write_text("old row\n", encoding="utf-8")
        config = replace(config, documents_path=container, work_dir=nested_work, overwrite=True)
        sentinel = nested_work / "old-row.json"
    else:
        container = tmp_path / "input-container"
        container.mkdir()
        nested_output = container / "submission.jsonl"
        nested_output.write_text("old output\n", encoding="utf-8")
        config = replace(config, documents_path=container, output_path=nested_output, overwrite=True)

    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(run_generation(config, FakeGenerator({})))

    assert sentinel.read_text(encoding="utf-8") == "old row\n"
    assert config.queries_path.read_bytes() == original_queries
    if overlap != "run_inside_work":
        assert config.run_path.read_bytes() == original_run


def test_overwrite_clears_old_state_before_fallible_input_loading(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Old one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Old two."),
        }
    )
    asyncio.run(run_generation(config, initial))
    valid_run = config.run_path.read_text(encoding="utf-8")
    config.run_path.write_text("not a six-field TREC row\n", encoding="utf-8")

    with pytest.raises(ValueError, match="six TREC fields"):
        asyncio.run(run_generation(replace(config, overwrite=True), FakeGenerator({})))

    assert not config.output_path.exists()
    assert not config.work_dir.exists()
    config.run_path.write_text(valid_run, encoding="utf-8")
    resumed = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Fresh one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Fresh two."),
        }
    )
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert {call["topic_id"] for call in resumed.calls} == {"rag2026-1", "rag2026-2"}
    assert "Old one." not in config.output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("artifact", ["raw/response.json", "errors/topic.txt", "scratch.bin"])
def test_create_rejects_any_existing_work_artifact(tmp_path: Path, artifact: str) -> None:
    config = _pipeline_config(tmp_path)
    path = config.work_dir / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts exist"):
        asyncio.run(run_generation(config, FakeGenerator({})))

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_final_publication_does_not_reuse_predictable_temp_name(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    old_shared_temp = config.output_path.with_name(config.output_path.name + ".tmp")
    old_shared_temp.write_text("unrelated sentinel\n", encoding="utf-8")
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert old_shared_temp.read_text(encoding="utf-8") == "unrelated sentinel\n"


def test_atomic_publication_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pipeline_config(tmp_path)
    synced_kinds: list[str] = []
    real_fsync = os.fsync

    def observing_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        synced_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(competition_rag.os, "fsync", observing_fsync)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert "file" in synced_kinds
    assert "directory" in synced_kinds


def test_second_process_cannot_mutate_same_generation_artifacts(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    entered = threading.Event()
    release = threading.Event()
    first_errors: list[BaseException] = []

    class BlockingGenerator(FakeGenerator):
        def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timed out")
            return super().complete_json(**kwargs)

    first = BlockingGenerator(
        {"rag2026-2": _topic_output(["climbmix-c"], "First process.")}
    )

    def run_first() -> None:
        try:
            asyncio.run(run_generation(config, first))
        except BaseException as exc:
            first_errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            asyncio.run(
                run_generation(
                    config,
                    FakeGenerator(
                        {"rag2026-2": _topic_output(["climbmix-c"], "Second process.")}
                    ),
                )
            )
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_errors == []


def test_invalid_model_output_keeps_final_submission_absent(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["not-retrieved"], "Unsupported."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    assert not config.output_path.exists()
    assert list((config.work_dir / "errors").glob("*.txt"))


def test_persisted_raw_responses_and_errors_redact_configured_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pipeline_config(tmp_path)
    api_key = "reflected-secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class ReflectingGenerator:
        def complete_json(
            self, *, topic_id: str, **kwargs: Any
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del kwargs
            if topic_id == "rag2026-1":
                return (
                    _topic_output(["climbmix-a"], "A supports it."),
                    {
                        "authorization": f"Bearer {api_key}",
                        "nested": [f"prefix-{api_key}-suffix"],
                    },
                )
            raise ValueError(f"provider reflected {api_key} in an error")

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, ReflectingGenerator()))

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in config.work_dir.rglob("*")
        if path.is_file()
    )
    assert api_key not in persisted
    assert "[REDACTED]" in persisted


def test_openrouter_redacts_its_key_from_persisted_success_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "client-only-secret-token"
    monkeypatch.delenv(config.api_key_env, raising=False)
    response = FakeHttpResponse(
        200,
        {
            "id": f"provider-reflected-{api_key}",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            _topic_output(["climbmix-c"], "C supports it.")
                        )
                    }
                }
            ],
        },
    )
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    asyncio.run(run_generation(config, generator))

    raw_path = next((config.work_dir / "raw").glob("*.json"))
    persisted = raw_path.read_text(encoding="utf-8")
    assert api_key not in persisted
    assert "provider-reflected-[REDACTED]" in persisted


class FakeHttpResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _openrouter_generator() -> OpenRouterJsonGenerator:
    return OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key="secret",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=3,
    )


def _provider_success() -> FakeHttpResponse:
    return FakeHttpResponse(
        200,
        {
            "id": "response-1",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(_topic_output(["climbmix-a"], "Supported."))
                    }
                }
            ],
        },
    )


def test_openrouter_request_uses_strict_schema_and_medium_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        captured.update({"url": url, **kwargs})
        return _provider_success()

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    generated, raw = _openrouter_generator().complete_json(
        topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
    )

    assert captured["url"] == "https://openrouter.example/v1/chat/completions"
    assert captured["json"]["reasoning"] == {"effort": "medium", "exclude": True}
    assert captured["json"]["provider"] == {"require_parameters": True}
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert generated["answer"][0]["citations"] == [0]
    assert raw["id"] == "response-1"


def test_malformed_semantic_completion_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return FakeHttpResponse(200, {"choices": [{"message": {"content": "not JSON"}}]})

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    with pytest.raises(ValueError, match="no repair call") as error:
        _openrouter_generator().complete_json(
            topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
        )

    assert calls == 1
    assert error.value.raw_response["choices"][0]["message"]["content"] == "not JSON"


def test_transient_retries_repeat_the_identical_request(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    responses = [FakeHttpResponse(500, {"error": "temporary"}), _provider_success()]

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        del url
        bodies.append(copy.deepcopy(kwargs["json"]))
        return responses.pop(0)

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)
    _openrouter_generator().complete_json(
        topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
    )

    assert len(bodies) == 2
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("status_code", [400, 429, 503])
def test_http_failures_persist_sanitized_status_body_and_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "reflected-secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)
    payload = {"error": {"message": f"rejected bearer {api_key}"}}

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        del url, kwargs
        return FakeHttpResponse(status_code, payload)

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    assert raw["http_status"] == status_code
    assert raw["envelope"] == {"error": {"message": "rejected bearer [REDACTED]"}}
    assert api_key not in failed_raw.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "escaped_key",
    [
        r"\u0073ecret-token",
        "".join(f"\\u{ord(character):04X}" for character in "secret-token"),
    ],
)
def test_http_json_failure_omits_body_with_decodable_escaped_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escaped_key: str
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)
    response = FakeHttpResponse(
        400,
        {"error": {"message": f"rejected bearer {api_key}"}},
    )
    response.text = '{"error":{"message":"rejected bearer ' + escaped_key + '"}}'
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    assert raw == {
        "http_status": 400,
        "envelope": {"error": {"message": "rejected bearer [REDACTED]"}},
    }
    assert api_key not in json.dumps(raw)


def test_http_non_json_failure_omits_body_and_keeps_non_reversible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(400, None)
    body_text = r"gateway rejected \u0073ecret-token"
    response.text = body_text
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    body_bytes = body_text.encode("utf-8")
    assert raw == {
        "http_status": 400,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }


def test_http_non_json_failure_omits_plain_body_and_keeps_non_reversible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(400, None)
    body_text = f"gateway rejected bearer {api_key}"
    response.text = body_text
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    persisted = failed_raw.read_text(encoding="utf-8")
    body_bytes = body_text.encode("utf-8")
    assert json.loads(persisted) == {
        "http_status": 400,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }
    assert api_key not in persisted
    assert body_text not in persisted


def test_http_non_json_success_never_persists_nested_decodable_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(200, None)
    body_text = r"gateway reflected \u005cu0073ecret-token"
    response.text = body_text
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    persisted = failed_raw.read_text(encoding="utf-8")
    body_bytes = body_text.encode("utf-8")
    assert json.loads(persisted) == {
        "http_status": 200,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }
    assert api_key not in persisted
    assert body_text not in persisted


@pytest.mark.parametrize(
    "body_text",
    [
        r"gateway reflected \u005cu0073ecret-token",
        "gateway reflected secret%2Dtoken",
        "gateway reflected secret&#45;token",
        "gateway reflected secret=2Dtoken",
        pytest.param(
            "gateway reflected secret&#38#45token",
            id="nested-semicolonless-html-entity",
        ),
    ],
)
def test_http_non_json_failure_never_persists_encoded_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body_text: str
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(503, None)
    response.text = body_text
    monkeypatch.setattr(competition_rag.requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    persisted = failed_raw.read_text(encoding="utf-8")
    raw = json.loads(persisted)
    body_bytes = body_text.encode("utf-8")
    assert raw == {
        "http_status": 503,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }
    assert api_key not in persisted
    assert body_text not in persisted


def test_cli_validates_topic_selection_before_env_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("not-an-official-topic",))
    config_path = tmp_path / "competition-rag.yaml"
    events: list[str] = []

    class GuardedEnvironment(dict[str, str]):
        def get(self, key: str, default: str = "") -> str:
            events.append(f"credential:{key}")
            return super().get(key, default)

    monkeypatch.setattr(competition_rag, "arguments", lambda: config_path)
    monkeypatch.setattr(competition_rag, "load_rag_generation_config", lambda _: config)
    monkeypatch.setattr(competition_rag, "load_repo_env", lambda _: events.append("env"))
    monkeypatch.setattr(competition_rag.os, "environ", GuardedEnvironment())

    with pytest.raises(SystemExit, match="unknown topic ID"):
        competition_rag.main()

    assert events == []


def test_cli_accepts_only_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "competition.yaml"

    assert arguments(["--config", str(config_path)]) == config_path
    with pytest.raises(SystemExit):
        arguments(["--config", str(config_path), "--model", "other"])

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
        (" qid\tnarrative\nrag2026-0\tQuestion\n", "header"),
        ("rag2026-0\n", "expected qid<TAB>narrative"),
    ],
)
def test_topics_reject_headers_and_missing_narratives(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_queries(path)


def test_canonical_topic_parser_preserves_extra_tabs_and_unicode_line_separators(
    tmp_path: Path,
) -> None:
    path = tmp_path / "topics.tsv"
    narrative = "Question with an extra\ttab and a Unicode line separator\u2028inside."
    path.write_text(f"rag2026-0\t{narrative}\n", encoding="utf-8")

    assert load_queries(path) == [("rag2026-0", narrative)]


def test_topic_narrative_preserves_exact_tsv_field_content(tmp_path: Path) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text("rag2026-0\t  Exact narrative field content  \n", encoding="utf-8")

    assert load_queries(path) == [
        ("rag2026-0", "  Exact narrative field content  ")
    ]


def test_topic_ids_preserve_organizer_topic_text(tmp_path: Path) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text(" rag2026-0\tExact narrative\n", encoding="utf-8")

    assert load_queries(path) == [(" rag2026-0", "Exact narrative")]


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


def test_plain_document_input_normalizes_non_utf8_failure(tmp_path: Path) -> None:
    documents_path = tmp_path / "non-utf8-documents.jsonl"
    documents_path.write_bytes(b"\xff\n")

    with pytest.raises(
        ValueError,
        match=r"non-utf8-documents\.jsonl: document input is not valid UTF-8",
    ):
        load_documents(documents_path, None, {"climbmix-a"}, 100)


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


def test_provider_schema_uses_supported_strict_cardinality_constraints() -> None:
    schema = competition_rag.output_schema()
    references = schema["properties"]["references"]
    answer = schema["properties"]["answer"]
    citations = answer["items"]["properties"]["citations"]

    assert references["minItems"] == 1
    assert answer["minItems"] == 1
    assert citations["minItems"] == 1
    assert citations["maxItems"] == 3
    assert "uniqueItems" not in json.dumps(schema)


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


def test_validates_all_organizer_allowed_rag_citation_forms_and_metadata() -> None:
    record = _submission_record()
    record["metadata"]["participant_note"] = "extra metadata is organizer-valid"
    record["references"] = ["climbmix-a", "climbmix-b", "climbmix-c"]
    record["answer"] = [
        {"text": "This heading is intentionally uncited.", "citations": []},
        {"text": "This claim uses a reference index.", "citations": [0]},
        {"text": "This claim uses a direct document ID.", "citations": ["climbmix-b"]},
    ]

    _validate_submission(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update({"extra": True}), "root object or metadata"),
        (lambda row: row["references"].append("climbmix-a"), "duplicated or outside"),
        (lambda row: row["references"].append("not-selected"), "duplicated or outside"),
        (lambda row: row["answer"][0].update({"citations": [True]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [2]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": ["not-selected"]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [0, 1, 0, 1]}), "0-3"),
        (lambda row: row["answer"][0].update({"heading": "Finding"}), "invalid fields"),
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


def test_prompt_labels_documents_by_docid_without_numeric_pseudo_citations() -> None:
    prompt = competition_rag.render_prompt(
        "What does the evidence say?",
        ["climbmix-a", "climbmix-b"],
        {"climbmix-a": "Evidence A.", "climbmix-b": "Evidence B."},
    )

    assert "Reference document docid: climbmix-a\nEvidence A." in prompt
    assert "Reference document docid: climbmix-b\nEvidence B." in prompt
    assert "[1] docid:" not in prompt
    assert "[2] docid:" not in prompt


def test_prompt_profiles_differ_and_default_is_unchanged() -> None:
    documents = {"climbmix-a": "Evidence A."}

    default = competition_rag.render_prompt("Narrative?", ["climbmix-a"], documents)
    atomic = competition_rag.render_prompt(
        "Narrative?", ["climbmix-a"], documents, "atomic_claims"
    )

    assert default == competition_rag.render_prompt(
        "Narrative?", ["climbmix-a"], documents, "default"
    )
    assert "one self-contained sentence" in atomic
    assert "one self-contained sentence" not in default
    # Both profiles keep the organizer-facing constraints (templates are line-wrapped).
    for prompt in (default, atomic):
        flat = " ".join(prompt.split())
        assert "one to three unique zero-based citation indexes" in flat
        assert "strongest to weakest support" in flat
        assert "Reference document docid: climbmix-a" in prompt


def test_full_budget_profile_keeps_atomic_wording_and_adds_budget_guidance() -> None:
    documents = {"climbmix-a": "Evidence A."}

    atomic = competition_rag.render_prompt(
        "Narrative?", ["climbmix-a"], documents, "atomic_claims"
    )
    full = competition_rag.render_prompt(
        "Narrative?", ["climbmix-a"], documents, "atomic_claims_full_budget"
    )

    for prompt in (atomic, full):
        flat = " ".join(prompt.split())
        assert "one self-contained sentence" in flat
        assert "strongest to weakest support" in flat
    assert "Aim for roughly 900" in " ".join(full.split())
    assert "Aim for roughly 900" not in " ".join(atomic.split())
    # The budget guidance must not license padding.
    assert "Do not pad" in full


def test_normalizer_rebuilds_references_from_citations_without_touching_text() -> None:
    record = {
        "metadata": {"narrative_id": "58"},
        # b and d are never cited; a and c are, out of order.
        "references": ["doc-a", "doc-b", "doc-c", "doc-d"],
        "answer": [
            {"text": "Second reference first.", "citations": [2]},
            {"text": "Then the first one.", "citations": [0, 2]},
        ],
    }

    out = competition_rag.normalize_generated_record(record)

    # References are exactly the cited docids, ordered by first use.
    assert out["references"] == ["doc-c", "doc-a"]
    # Every reference is now cited, which is what the strict profile demands.
    used = {c for item in out["answer"] for c in item["citations"]}
    assert used == set(range(len(out["references"])))
    # Citations still point at the same documents they did before.
    assert [out["references"][c] for c in out["answer"][0]["citations"]] == ["doc-c"]
    assert [out["references"][c] for c in out["answer"][1]["citations"]] == ["doc-a", "doc-c"]
    # Answer text is untouched.
    assert [i["text"] for i in out["answer"]] == [i["text"] for i in record["answer"]]


def test_normalized_record_satisfies_the_strict_profile(tmp_path: Path) -> None:
    generated = {
        "references": ["climbmix-a", "climbmix-b", "climbmix-c"],
        "answer": [{"text": "A grounded claim.", "citations": [1]}],
    }
    record = competition_rag.normalize_generated_record(
        competition_rag.build_submission_record(
            generated,
            topic_id="58",
            narrative="Official question",
            team_id="local-baseline",
            run_id="dev-spike",
            run_desc="development spike",
        )
    )

    # Would have raised "uncited references" without normalization.
    competition_rag._validate_generated_submission_record(
        record,
        topic_id="58",
        narrative="Official question",
        allowed_docids=["climbmix-a", "climbmix-b", "climbmix-c"],
        team_id="local-baseline",
        run_id="dev-spike",
        run_desc="development spike",
    )
    assert record["references"] == ["climbmix-b"]


def test_focused_profile_drops_the_cite_every_reference_rule() -> None:
    focused = " ".join(
        competition_rag.render_prompt(
            "Narrative?", ["climbmix-a"], {"climbmix-a": "Evidence."}, "focused_citations"
        ).split()
    )
    default = " ".join(
        competition_rag.render_prompt(
            "Narrative?", ["climbmix-a"], {"climbmix-a": "Evidence."}, "default"
        ).split()
    )

    assert "cite every reference" in default
    assert "cite every reference" not in focused
    assert "single document that best supports" in focused


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            {"references": ["a"], "answer": ["oops"]},
            "not an object",
            id="answer-item-is-a-string",
        ),
        pytest.param(
            {"references": ["a"], "answer": [{"citations": [0]}]},
            "text",
            id="missing-text",
        ),
        pytest.param(
            {"references": ["a"], "answer": [{"text": "x", "citations": 0}]},
            "citations list",
            id="citations-not-a-list",
        ),
        pytest.param(
            {"references": "a", "answer": [{"text": "x", "citations": [0]}]},
            "references must be a nonempty list",
            id="references-not-a-list",
        ),
    ],
)
def test_post_processors_reject_malformed_shapes_cleanly(
    record: dict[str, Any], expected: str
) -> None:
    """These run before validation, so a malformed model response must not raise TypeError.

    Relaxing structured output to json_object produced exactly these shapes from a live model.
    """
    with pytest.raises(ValueError, match=expected):
        competition_rag.normalize_generated_record(
            competition_rag.trim_to_word_limit(record)
        )


def test_normalizer_collapses_a_duplicate_citation() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b"],
        "answer": [{"text": "Claim.", "citations": [1, 1]}],
    }
    out = competition_rag.normalize_generated_record(record)
    # Would otherwise survive as [0, 0] and fail the unique-citation rule.
    assert out["answer"][0]["citations"] == [0]
    assert out["references"] == ["doc-b"]


def test_normalizer_canonicalizes_a_docid_listed_twice() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b", "doc-a"],
        "answer": [{"text": "Claim.", "citations": [0, 2]}],
    }
    out = competition_rag.normalize_generated_record(record)
    # Both positions name the same document, so one citation of one reference remains.
    assert out["references"] == ["doc-a"]
    assert out["answer"][0]["citations"] == [0]


def test_normalizer_reports_an_invalid_citation_rather_than_dropping_it() -> None:
    """A fabricated citation index must fail loudly, not be silently pruned.

    Two of the generators tested for this track emitted document ids outside the retrieval
    pool, so this is a live failure mode rather than a hypothetical one.
    """
    record = {
        "metadata": {"narrative_id": "58"},
        "references": ["doc-a", "doc-b"],
        "answer": [
            {"text": "Grounded claim.", "citations": [0]},
            {"text": "Fabricated pointer.", "citations": [9]},
        ],
    }

    with pytest.raises(ValueError, match="answer\\[1\\] has an invalid citation"):
        competition_rag.normalize_generated_record(record)


def test_normalizer_rejects_a_partially_invalid_citation_list() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b"],
        "answer": [{"text": "Claim.", "citations": [0, 9]}],
    }

    with pytest.raises(ValueError, match="invalid citation"):
        competition_rag.normalize_generated_record(record)


def test_word_cap_trim_drops_trailing_objects_and_frees_their_references() -> None:
    record = {
        "metadata": {"narrative_id": "58"},
        "references": ["doc-a", "doc-b"],
        "answer": [
            {"text": "word " * 900, "citations": [0]},
            {"text": "word " * 200, "citations": [1]},
        ],
    }

    trimmed = competition_rag.trim_to_word_limit(record)
    assert len(trimmed["answer"]) == 1
    assert sum(len(i["text"].split()) for i in trimmed["answer"]) <= 1024

    # The reference only the dropped object cited is normalized away.
    out = competition_rag.normalize_generated_record(trimmed)
    assert out["references"] == ["doc-a"]


def test_word_cap_trim_is_a_no_op_when_already_within_budget() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a"],
        "answer": [{"text": "Short claim.", "citations": [0]}],
    }
    assert competition_rag.trim_to_word_limit(record) is record


def test_tail_contract_profile_restates_the_contract_after_the_question() -> None:
    """With 100 documents inserted the contract sits ~120k tokens from the generation point."""
    prompt = competition_rag.render_prompt(
        "What about nuclear?", ["d1"], {"d1": "evidence"}, "focused_citations_tail"
    )

    assert prompt.index("Question: What about nuclear?") < prompt.index("Restating the output")
    # The leading contract is preserved, not moved.
    assert prompt.index("one self-contained sentence") < prompt.index("Reference document")
    flat = " ".join(prompt.split())
    assert flat.count("never more than 1,024") == 1


def test_contract_in_system_profile_moves_the_contract_to_the_system_channel() -> None:
    system = competition_rag.system_prompt_for("contract_in_system")
    user = competition_rag.render_prompt("Q?", ["d1"], {"d1": "t"}, "contract_in_system")

    assert "one to three unique zero-based" in system
    assert "one to three unique zero-based" not in user
    # Other profiles keep the original short system message.
    assert competition_rag.system_prompt_for("focused_citations") == competition_rag.SYSTEM_PROMPT
    assert competition_rag.system_prompt_for("default") == competition_rag.SYSTEM_PROMPT


def test_system_prompt_for_rejects_an_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unsupported prompt profile"):
        competition_rag.system_prompt_for("nonexistent")


def test_unknown_prompt_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported prompt profile"):
        competition_rag.render_prompt("Narrative?", [], {}, "nonexistent")


def test_configuration_mode_guidance_uses_config_values_not_cli_flags(tmp_path: Path) -> None:
    existing = _pipeline_config(tmp_path)
    existing.output_path.parent.mkdir(parents=True, exist_ok=True)
    existing.output_path.write_text("old output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment\\.mode: resume") as create_error:
        asyncio.run(run_generation(existing, FakeGenerator({})))
    assert "experiment.mode: overwrite" in str(create_error.value)
    assert "--resume" not in str(create_error.value)
    assert "--overwrite" not in str(create_error.value)

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = _pipeline_config(failed_root)
    with pytest.raises(RuntimeError, match="experiment\\.mode: resume") as failure_error:
        asyncio.run(run_generation(failed, FakeGenerator({})))
    assert "--resume" not in str(failure_error.value)


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
    assert "Reference document docid: climbmix-a" in topic_one_prompt
    assert "Reference document docid: climbmix-b" in topic_one_prompt


def test_generation_consumes_the_retrieval_export_file_contract(
    tmp_path: Path,
) -> None:
    """Catch generation drifting from the public TSV-and-ZIP handoff."""
    (tmp_path / "official").mkdir()
    (tmp_path / "official/topics.tsv").write_text(
        "rag2026-9\tUnselected official question\n"
        "rag2026-3\tSecond selected official question\n"
        "rag2026-4\tFirst selected official question\n",
        encoding="utf-8",
    )
    retrieval = tmp_path / "outputs/facet-deepseek-b40-v1"
    retrieval.mkdir(parents=True)
    (retrieval / "r_output_trec_rag_2026.tsv").write_text(
        "rag2026-3 Q0 climbmix-3a 1 2 facet-deepseek-b40-v1\n"
        "rag2026-3 Q0 climbmix-3b 2 1 facet-deepseek-b40-v1\n"
        "rag2026-4 Q0 climbmix-4a 1 1 facet-deepseek-b40-v1\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(retrieval / "retrieval_with_text.jsonl.zip", "w") as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            "".join(
                json.dumps(row) + "\n"
                for row in [
                    {
                        "query": {
                            "qid": "rag2026-3",
                            "text": "Second selected official question",
                        },
                        "candidates": [
                            {
                                "docid": "climbmix-3a",
                                "rank": 1,
                                "score": 2,
                                "doc": "First selected document.",
                                "index": "climbmix-400b",
                                "stage": "canonical_supported",
                            },
                            {
                                "docid": "climbmix-3b",
                                "rank": 2,
                                "score": 1,
                                "doc": "Second selected document.",
                                "index": "climbmix-400b",
                                "stage": "canonical_supported",
                            },
                        ],
                    },
                    {
                        "query": {
                            "qid": "rag2026-4",
                            "text": "First selected official question",
                        },
                        "candidates": [
                            {
                                "docid": "climbmix-4a",
                                "rank": 1,
                                "score": 1,
                                "doc": "Third selected document.",
                                "index": "climbmix-400b",
                                "stage": "canonical_supported",
                            }
                        ],
                    },
                ]
            ),
        )
    config = load_rag_generation_config(
        _write_config(
            tmp_path,
            _config_text(
                inputs_extra="  topic_ids: [rag2026-4, rag2026-3]\n",
            ),
        )
    )
    generator = FakeGenerator(
        {
            "rag2026-3": _topic_output(
                ["climbmix-3a", "climbmix-3b"], "Both selected documents support it."
            ),
            "rag2026-4": _topic_output(
                ["climbmix-4a"], "The selected document supports it."
            ),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert config.output_path.name == "rag_output_trec_rag_2026.jsonl"
    rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["narrative_id"] for row in rows] == [
        "rag2026-3",
        "rag2026-4",
    ]
    assert [list(row) for row in rows] == [
        ["metadata", "references", "answer"],
        ["metadata", "references", "answer"],
    ]
    assert [list(row["metadata"]) for row in rows] == [
        ["team_id", "narrative_id", "narrative", "run_id", "run_desc"],
        ["team_id", "narrative_id", "narrative", "run_id", "run_desc"],
    ]
    assert [row["references"] for row in rows] == [
        ["climbmix-3a", "climbmix-3b"],
        ["climbmix-4a"],
    ]
    assert [row["answer"][0]["citations"] for row in rows] == [[0, 1], [0]]


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


def test_resume_regenerates_row_with_non_strict_metadata(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )
    asyncio.run(run_generation(config, initial))
    row_path = next(
        path
        for path in (config.work_dir / "rows").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["metadata"]["narrative_id"]
        == "rag2026-1"
    )
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["metadata"]["participant_note"] = "organizer-valid but not generated-profile"
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    resumed = FakeGenerator(
        {"rag2026-1": _topic_output(["climbmix-a"], "Regenerated strict row.")}
    )

    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert [call["topic_id"] for call in resumed.calls] == ["rag2026-1"]
    output_rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    regenerated = next(
        item for item in output_rows if item["metadata"]["narrative_id"] == "rag2026-1"
    )
    assert set(regenerated["metadata"]) == {
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    }


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


@pytest.mark.parametrize(
    ("references", "citations"),
    [
        pytest.param(["climbmix-a"], [], id="empty-citations"),
        pytest.param(["climbmix-a"], ["climbmix-a"], id="direct-docid-citation"),
    ],
)
def test_generation_rejects_organizer_valid_rows_outside_strict_profile(
    tmp_path: Path,
    references: list[str],
    citations: list[int | str],
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": references,
        "answer": [{"text": "Generated answer.", "citations": citations}],
    }

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    assert not config.output_path.exists()


def test_generation_now_prunes_an_uncited_reference_instead_of_failing(tmp_path: Path) -> None:
    """Uncited references used to fail the strict profile; they are normalized away."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [{"text": "Only the second source is cited.", "citations": [1]}],
    }

    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    rows = [json.loads(line) for line in config.output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["references"] == ["climbmix-b"]
    assert rows[0]["answer"][0]["citations"] == [0]
    assert rows[0]["answer"][0]["text"] == "Only the second source is cited."


def test_generation_normalizes_duplicate_citations_end_to_end(tmp_path: Path) -> None:
    """Cover the full generate-to-publish path, not just the normalizer in isolation."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [{"text": "One document, cited twice.", "citations": [1, 1]}],
    }

    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    rows = [json.loads(line) for line in config.output_path.read_text().splitlines()]
    assert rows[0]["references"] == ["climbmix-b"]
    assert rows[0]["answer"][0]["citations"] == [0]


def test_truncated_completion_is_rejected_rather_than_shortened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grammar-constrained decoding can close the JSON validly while the answer is cut short."""
    generator = competition_rag.OpenRouterJsonGenerator(
        api_base="https://openrouter.test",
        api_key="test-key",
        model="m",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=10,
        timeout_seconds=5.0,
        transport_max_attempts=1,
    )
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": json.dumps(
                        {
                            "references": ["climbmix-a"],
                            "answer": [{"text": "Cut short.", "citations": [0]}],
                        }
                    )
                },
            }
        ]
    }
    monkeypatch.setattr(
        competition_rag.requests, "post", lambda *a, **k: FakeHttpResponse(200, body)
    )
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)

    with pytest.raises(competition_rag.SemanticCompletionError, match="truncated"):
        generator.complete_json(
            topic_id="rag2026-1",
            system_prompt="s",
            user_prompt="u",
            response_schema=competition_rag.output_schema(),
        )


@pytest.mark.parametrize(
    "finish_reason",
    [
        pytest.param("error", id="provider-error-with-partial-content"),
        pytest.param("content_filter", id="content-filter"),
        pytest.param("tool_calls", id="tool-calls"),
        pytest.param(None, id="absent"),
        pytest.param("unrecognised", id="unknown"),
    ],
)
def test_only_a_stop_finish_reason_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None
) -> None:
    """OpenRouter returns HTTP 200 with finish_reason 'error' and partial content."""
    generator = competition_rag.OpenRouterJsonGenerator(
        api_base="https://openrouter.test",
        api_key="test-key",
        model="m",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=100,
        timeout_seconds=5.0,
        transport_max_attempts=1,
    )
    choice: dict[str, Any] = {
        "message": {
            "content": json.dumps(
                {
                    "references": ["climbmix-a"],
                    "answer": [{"text": "Looks complete.", "citations": [0]}],
                }
            )
        }
    }
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    monkeypatch.setattr(
        competition_rag.requests,
        "post",
        lambda *a, **k: FakeHttpResponse(200, {"choices": [choice]}),
    )
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)

    with pytest.raises(competition_rag.SemanticCompletionError, match="rather than 'stop'"):
        generator.complete_json(
            topic_id="rag2026-1",
            system_prompt="s",
            user_prompt="u",
            response_schema=competition_rag.output_schema(),
        )


def test_resume_refuses_rows_generated_under_different_settings(tmp_path: Path) -> None:
    """Resume reuses saved rows; shape revalidation cannot detect a changed model or prompt."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": [0]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    changed = replace(config, resume=True, prompt_profile="focused_citations_tail")
    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(run_generation(changed, FakeGenerator({"rag2026-1": generated})))


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"max_tokens": 7000}, id="max-tokens"),
        pytest.param({"temperature": 0.2}, id="temperature"),
        pytest.param({"api_base": "https://elsewhere.test"}, id="api-base"),
        pytest.param({"model": "other/model"}, id="model"),
    ],
)
def test_resume_refuses_any_request_affecting_change(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """Every setting that reaches the request must invalidate a resume."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": [0]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(
            run_generation(
                replace(config, resume=True, **change), FakeGenerator({"rag2026-1": generated})
            )
        )


def test_generation_identity_covers_every_request_and_input_selector(tmp_path: Path) -> None:
    """archive_member fails earlier during input loading, so assert it directly."""
    config = _pipeline_config(tmp_path)
    base = competition_rag._generation_identity(config)

    for field, value in [
        ("archive_member", "other.jsonl"),
        ("temperature", 0.2),
        ("max_tokens", 7000),
        ("api_base", "https://elsewhere.test"),
        ("structured_output", "json_object"),
        ("prompt_profile", "focused_citations_tail"),
        ("top_k", 50),
        ("max_document_words", 500),
    ]:
        changed = competition_rag._generation_identity(replace(config, **{field: value}))
        assert changed != base, f"{field} does not invalidate the identity"


def test_resume_refuses_rows_that_predate_settings_tracking(tmp_path: Path) -> None:
    """A workdir from before identity tracking cannot be shown to match; adopt nothing."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": [0]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))
    (config.work_dir / "generation_identity.json").unlink()

    with pytest.raises(ValueError, match="predate settings tracking"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_refuses_an_older_identity_version(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": [0]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))
    path = config.work_dir / "generation_identity.json"
    recorded = json.loads(path.read_text())
    recorded["identity_version"] = 1
    path.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(ValueError, match="older revision"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_accepts_rows_generated_under_the_same_settings(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": [0]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    resumed = FakeGenerator({})
    asyncio.run(run_generation(replace(config, resume=True), resumed))
    assert resumed.calls == [], "resume should reuse the saved row without regenerating"


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
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            _topic_output(["climbmix-c"], "C supports it.")
                        )
                    },
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


@pytest.mark.parametrize(
    ("api_key", "reflected_key"),
    [
        ("secret-token", "secret%2Dtoken"),
        ("secret-token", "secret%252Dtoken"),
        pytest.param(
            "secret-token",
            r"secret%5Cu002Dtoken",
            id="percent-then-unicode-escape",
        ),
        pytest.param(
            "secret-token",
            r"secret\u00252Dtoken",
            id="unicode-escape-then-percent",
        ),
        pytest.param(
            "secret%2Dtoken",
            "secret%252Dtoken",
            id="configured-secret-contains-percent-escape",
        ),
    ],
)
def test_percent_encoded_api_key_in_parsed_envelope_never_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    reflected_key: str,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    response = FakeHttpResponse(
        200,
        {
            "id": f"provider-reflected-{reflected_key}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            _topic_output(["climbmix-c"], "C supports it.")
                        )
                    },
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

    raw = json.loads(next((config.work_dir / "raw").glob("*.json")).read_text(encoding="utf-8"))
    assert raw["id"] == "[REDACTED]"
    persisted = json.dumps(raw)
    assert api_key not in persisted
    assert reflected_key not in persisted


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
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(_topic_output(["climbmix-a"], "Supported."))
                    },
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
        return FakeHttpResponse(200, {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]})

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

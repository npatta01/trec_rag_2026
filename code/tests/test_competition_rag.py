from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from trec_rag.competition_rag import (
    load_documents,
    load_queries,
    load_rag_generation_config,
    load_trec_run,
    select_queries,
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

from __future__ import annotations

import asyncio
import copy
import json
import zipfile
from dataclasses import replace
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
    validate_submission_record,
)


def _valid_record() -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": "castorini",
            "narrative_id": "rag2026-6",
            "narrative": "Compare the available evidence.",
            "run_id": "gpt-sol-bm25",
            "run_desc": "BM25 retrieval with GPT Sol answer generation.",
        },
        "references": ["shard_1", "shard_2"],
        "answer": [
            {"text": "The first supported finding is important.", "citations": [0]},
            {"text": "The second source adds a limitation.", "citations": [0, 1]},
        ],
    }


def _validate(record: dict[str, Any]) -> None:
    validate_submission_record(
        record,
        topic_id="rag2026-6",
        narrative="Compare the available evidence.",
        allowed_docids=["shard_1", "shard_2", "shard_3"],
        team_id="castorini",
        run_id="gpt-sol-bm25",
        run_desc="BM25 retrieval with GPT Sol answer generation.",
    )


def test_validates_organizer_shaped_record() -> None:
    _validate(_valid_record())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update({"extra": True}), "root object or metadata"),
        (
            lambda row: row["metadata"].update({"model": "gpt-sol"}),
            "root object or metadata",
        ),
        (lambda row: row["references"].append("shard_1"), "duplicated or outside"),
        (lambda row: row["references"].append("shard_9"), "duplicated or outside"),
        (lambda row: row["answer"][0].update({"citations": ["shard_1"]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [0, 0]}), "unique citations"),
        (lambda row: row["answer"][0].update({"citations": [2]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [True]}), "citation"),
        (lambda row: row["answer"][0].update({"heading": "Finding"}), "invalid fields"),
        (lambda row: row.update({"answer": row["answer"][:1]}), "uncited references"),
    ],
)
def test_rejects_records_outside_official_contract(mutate: Any, message: str) -> None:
    record = copy.deepcopy(_valid_record())
    mutate(record)
    with pytest.raises(ValueError, match=message):
        _validate(record)


def test_rejects_answer_over_1024_whitespace_words() -> None:
    record = _valid_record()
    record["references"] = ["shard_1"]
    record["answer"] = [{"text": "word " * 1025, "citations": [0]}]

    with pytest.raises(ValueError, match="1,024 words"):
        _validate(record)


def test_build_submission_record_injects_only_official_metadata() -> None:
    generated = {
        "metadata": {"model_wrote": "this must be ignored"},
        "references": ["shard_1"],
        "answer": [{"text": "A supported answer.", "citations": [0]}],
    }

    record = build_submission_record(
        generated,
        topic_id="rag2026-6",
        narrative="Compare the available evidence.",
        team_id="castorini",
        run_id="gpt-sol-bm25",
        run_desc="BM25 retrieval with GPT Sol answer generation.",
    )

    assert list(record) == ["metadata", "references", "answer"]
    assert list(record["metadata"]) == [
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    ]
    assert "model_wrote" not in record["metadata"]


def test_parses_plain_or_fenced_generated_json() -> None:
    assert parse_generated_json('{"references": [], "answer": []}') == {
        "references": [],
        "answer": [],
    }
    assert parse_generated_json('```json\n{"references": [], "answer": []}\n```') == {
        "references": [],
        "answer": [],
    }
    with pytest.raises(ValueError, match="one JSON object"):
        parse_generated_json("before {\"answer\": []} after")


def test_loads_queries_ranked_run_and_document_zip(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.tsv"
    query_path.write_text(
        "qid\tnarrative\nrag2026-2\tSecond topic\nrag2026-1\tFirst\twith tab\n",
        encoding="utf-8",
    )
    run_path = tmp_path / "run.txt"
    run_path.write_text(
        "rag2026-1 Q0 shard_2 2 4.0 bm25\n"
        "rag2026-1 Q0 shard_1 1 5.0 bm25\n"
        "rag2026-2 Q0 shard_3 1 7.0 bm25\n",
        encoding="utf-8",
    )
    archive_path = tmp_path / "documents.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "documents.jsonl",
            json.dumps(
                {
                    "query": {"qid": "rag2026-1"},
                    "candidates": [
                        {"docid": "shard_1", "doc": "one two three four"},
                        {"docid": "shard_2", "text": "alpha beta gamma"},
                    ],
                }
            )
            + "\n"
            + json.dumps({"docid": "shard_3", "contents": "red green blue"})
            + "\n",
        )

    queries = load_queries(query_path)
    ranked = load_trec_run(run_path, {qid for qid, _ in queries}, top_k=1)
    documents = load_documents(
        archive_path,
        archive_member=None,
        wanted_docids={docid for rows in ranked.values() for docid in rows},
        max_words=3,
    )

    assert queries == [
        ("rag2026-2", "Second topic"),
        ("rag2026-1", "First\twith tab"),
    ]
    assert ranked == {"rag2026-1": ["shard_1"], "rag2026-2": ["shard_3"]}
    assert documents == {
        "shard_1": "one two three",
        "shard_3": "red green blue",
    }


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


def _pipeline_inputs(tmp_path: Path) -> RagGenerationConfig:
    query_path = tmp_path / "queries.tsv"
    query_path.write_text(
        "rag2026-2\tWhat is supported for topic two?\n"
        "rag2026-1\tWhat is supported for topic one?\n",
        encoding="utf-8",
    )
    run_path = tmp_path / "run.txt"
    run_path.write_text(
        "rag2026-1 Q0 shard_a 1 9.0 bm25\n"
        "rag2026-1 Q0 shard_b 2 8.0 bm25\n"
        "rag2026-2 Q0 shard_c 1 7.0 bm25\n",
        encoding="utf-8",
    )
    document_path = tmp_path / "documents.jsonl"
    document_path.write_text(
        json.dumps({"docid": "shard_a", "text": "Evidence A."})
        + "\n"
        + json.dumps({"docid": "shard_b", "text": "Evidence B."})
        + "\n"
        + json.dumps({"docid": "shard_c", "text": "Evidence C."})
        + "\n",
        encoding="utf-8",
    )
    return RagGenerationConfig(
        queries_path=query_path,
        run_path=run_path,
        documents_path=document_path,
        output_path=tmp_path / "submission.jsonl",
        work_dir=tmp_path / "work",
        team_id="castorini",
        run_id="gpt-sol-bm25",
        run_desc="BM25 retrieval with GPT Sol answer generation.",
        concurrency=2,
    )


def test_runs_fixed_retrieval_generation_in_official_query_order(tmp_path: Path) -> None:
    config = _pipeline_inputs(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": {
                "references": ["shard_a", "shard_b"],
                "answer": [
                    {"text": "A and B support the answer.", "citations": [0, 1]}
                ],
            },
            "rag2026-2": {
                "references": ["shard_c"],
                "answer": [{"text": "C supports the answer.", "citations": [0]}],
            },
        }
    )

    asyncio.run(run_generation(config, generator))

    rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["narrative_id"] for row in rows] == [
        "rag2026-2",
        "rag2026-1",
    ]
    assert rows[0]["answer"][0]["citations"] == [0]
    assert all(type(citation) is int for row in rows for item in row["answer"] for citation in item["citations"])
    assert {call["topic_id"] for call in generator.calls} == {"rag2026-1", "rag2026-2"}
    topic_one_prompt = next(
        call["user_prompt"]
        for call in generator.calls
        if call["topic_id"] == "rag2026-1"
    )
    assert "[1] docid: shard_a" in topic_one_prompt
    assert "[2] docid: shard_b" in topic_one_prompt


def test_resume_reuses_valid_topic_rows_without_model_calls(tmp_path: Path) -> None:
    config = _pipeline_inputs(tmp_path)
    first = FakeGenerator(
        {
            "rag2026-1": {
                "references": ["shard_a"],
                "answer": [{"text": "A supports this.", "citations": [0]}],
            },
            "rag2026-2": {
                "references": ["shard_c"],
                "answer": [{"text": "C supports this.", "citations": [0]}],
            },
        }
    )
    asyncio.run(run_generation(config, first))

    second = FakeGenerator({})
    asyncio.run(run_generation(replace(config, resume=True), second))

    assert second.calls == []
    assert config.output_path.exists()


def test_invalid_model_output_does_not_write_consolidated_submission(
    tmp_path: Path,
) -> None:
    config = _pipeline_inputs(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": {
                "references": ["not_retrieved"],
                "answer": [{"text": "Unsupported.", "citations": [0]}],
            },
            "rag2026-2": {
                "references": ["shard_c"],
                "answer": [{"text": "C supports this.", "citations": [0]}],
            },
        }
    )

    with pytest.raises(RuntimeError, match="1 topic.*failed"):
        asyncio.run(run_generation(config, generator))

    assert not config.output_path.exists()
    assert list((config.work_dir / "errors").glob("*.txt"))


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


def test_openrouter_request_uses_strict_schema_and_medium_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        captured.update({"url": url, **kwargs})
        return FakeHttpResponse(
            200,
            {
                "id": "response-1",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "references": ["shard_1"],
                                    "answer": [
                                        {"text": "Supported.", "citations": [0]}
                                    ],
                                }
                            )
                        }
                    }
                ],
            },
        )

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    generated, raw = _openrouter_generator().complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert captured["url"] == "https://openrouter.example/v1/chat/completions"
    assert captured["json"]["model"] == "openai/gpt-5.6-sol"
    assert captured["json"]["reasoning"] == {"effort": "medium", "exclude": True}
    assert captured["json"]["provider"] == {"require_parameters": True}
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert "temperature" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert generated["answer"][0]["citations"] == [0]
    assert raw["id"] == "response-1"


def test_malformed_semantic_completion_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return FakeHttpResponse(
            200,
            {"choices": [{"message": {"content": "not JSON"}}]},
        )

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    with pytest.raises(ValueError, match="no repair call") as error:
        _openrouter_generator().complete_json(
            topic_id="rag2026-1",
            system_prompt="system",
            user_prompt="user",
            response_schema={"type": "object"},
        )

    assert calls == 1
    assert error.value.raw_response["choices"][0]["message"]["content"] == "not JSON"


def test_transient_retries_repeat_the_identical_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []
    responses = [
        FakeHttpResponse(500, {"error": "temporary"}),
        FakeHttpResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "references": ["shard_1"],
                                    "answer": [
                                        {"text": "Supported.", "citations": [0]}
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        ),
    ]

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        del url
        bodies.append(copy.deepcopy(kwargs["json"]))
        return responses.pop(0)

    monkeypatch.setattr(competition_rag.requests, "post", fake_post)
    monkeypatch.setattr(competition_rag.time, "sleep", lambda _: None)
    _openrouter_generator().complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert len(bodies) == 2
    assert bodies[0] == bodies[1]


def _write_generation_config(tmp_path: Path, extra_generation: str = "") -> Path:
    config_path = tmp_path / "competition.yaml"
    config_path.write_text(
        """experiment:
  id: rag26_gpt_sol_bm25
  output_dir: generated/rag26_gpt_sol_bm25
  mode: resume

submission:
  team_id: castorini
  run_desc: Full fixed BM25 retrieval with GPT Sol answer generation.

inputs:
  queries: inputs/topics.tsv
  run: inputs/bm25.trec
  documents: inputs/documents.zip
  archive_member: documents.jsonl

retrieval:
  top_k: 100
  max_document_words: 750

generation:
  type: openrouter
  api_base: https://openrouter.example/v1
  api_key_env: OPENROUTER_API_KEY
  model: openai/gpt-5.6-sol
  reasoning_effort: high
  temperature: 0.0
  max_tokens: 7000
  timeout_seconds: 120
  transport_max_attempts: 2
  concurrency: 3
"""
        + extra_generation,
        encoding="utf-8",
    )
    return config_path


def test_loads_strict_yaml_generation_config(tmp_path: Path) -> None:
    config_path = _write_generation_config(tmp_path)

    config = load_rag_generation_config(config_path)

    assert config.queries_path == tmp_path / "inputs/topics.tsv"
    assert config.run_path == tmp_path / "inputs/bm25.trec"
    assert config.documents_path == tmp_path / "inputs/documents.zip"
    assert config.archive_member == "documents.jsonl"
    assert config.output_path == (
        tmp_path / "generated/rag26_gpt_sol_bm25/rag_output_trec_rag_2026.jsonl"
    )
    assert config.work_dir == tmp_path / "generated/rag26_gpt_sol_bm25/work"
    assert config.run_id == "rag26_gpt_sol_bm25"
    assert config.team_id == "castorini"
    assert config.top_k == 100
    assert config.max_document_words == 750
    assert config.model == "openai/gpt-5.6-sol"
    assert config.reasoning_effort == "high"
    assert config.max_tokens == 7000
    assert config.timeout_seconds == 120
    assert config.transport_max_attempts == 2
    assert config.concurrency == 3
    assert config.resume is True
    assert config.overwrite is False


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = _write_generation_config(tmp_path, "  surprise: true\n")

    with pytest.raises(ValueError, match="unknown generation field.*surprise"):
        load_rag_generation_config(config_path)


def test_cli_accepts_only_a_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "competition.yaml"

    assert arguments(["--config", str(config_path)]) == config_path
    with pytest.raises(SystemExit):
        arguments(["--config", str(config_path), "--model", "another-model"])


def test_checked_in_example_config_stays_loadable() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    config = load_rag_generation_config(
        repo_root / "configs/rag26_competition_gpt_sol_bm25.example.yaml"
    )

    assert config.run_id == "rag26_competition_gpt_sol_bm25_v1"
    assert config.output_path.name == "rag_output_trec_rag_2026.jsonl"
    assert config.top_k is None
    assert config.temperature is None
    assert config.resume is False
    assert config.overwrite is False

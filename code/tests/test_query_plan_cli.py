import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from trec_rag import query_plan_cli
from trec_rag.query_planner import (
    HttpJsonResponse,
    QueryPlanGenerationError,
    QueryPlanValidationError,
    RenderedQuery,
)
from trec_rag.topics import Topic


@dataclass(frozen=True)
class _FakeProvenance:
    elapsed_seconds: float = 0.01


class _FakePlan:
    def __init__(self, topic_id: str) -> None:
        self.topic_id = topic_id

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "facet_count": 1,
            "global_expansion": {"terms": []},
        }


@dataclass(frozen=True)
class _FakeResult:
    plan: _FakePlan
    provenance: _FakeProvenance
    cache_hit: bool
    cache_path: Path | None


@dataclass
class _MixedRun:
    status: int
    output: Path
    failure_output: Path
    manifest_path: Path
    outcome_dir: Path
    args: argparse.Namespace
    generate_calls: list[str]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_mixed_parallel_batch(monkeypatch, tmp_path: Path):
    topic_ids = (
        "cached",
        "invalid-response",
        "timeout",
        "invalid-cache",
        "unexpected",
        "render-failure",
        "fresh",
    )
    topics = [
        Topic(id=topic_id, title=topic_id, narrative=f"Narrative for {topic_id}")
        for topic_id in topic_ids
    ]
    outcomes: dict[str, _FakeResult | Exception] = {
        "cached": _FakeResult(
            plan=_FakePlan("cached"),
            provenance=_FakeProvenance(),
            cache_hit=True,
            cache_path=tmp_path / "cache" / "cached.json",
        ),
        "invalid-response": QueryPlanGenerationError(
            "query planner returned an invalid plan",
            response={"id": "response-with-invalid-plan"},
            content='{"facet_count": "not-an-integer"}',
        ),
        "timeout": TimeoutError("planner request timed out"),
        "invalid-cache": QueryPlanValidationError("cached plan failed validation"),
        "unexpected": RuntimeError("unanticipated per-topic failure"),
        "render-failure": _FakeResult(
            plan=_FakePlan("render-failure"),
            provenance=_FakeProvenance(),
            cache_hit=False,
            cache_path=tmp_path / "cache" / "render-failure.json",
        ),
        "fresh": _FakeResult(
            plan=_FakePlan("fresh"),
            provenance=_FakeProvenance(),
            cache_hit=False,
            cache_path=tmp_path / "cache" / "fresh.json",
        ),
    }
    generate_calls: list[str] = []

    class FakeGenerator:
        def __init__(self, **kwargs) -> None:
            self.model = kwargs["model"]
            self.model_revision = kwargs["model_revision"]
            self.base_url = kwargs["base_url"]
            self.reasoning_effort = kwargs["reasoning_effort"]
            self.max_tokens = kwargs["max_tokens"]
            self.temperature = kwargs["temperature"]
            self.seed = kwargs["seed"]
            self.max_facets = kwargs["max_facets"]

        def request_payload(self, topic: Topic) -> dict[str, object]:
            return {
                "model": self.model,
                "messages": [{"role": "developer", "content": "test prompt"}],
                "response_format": {
                    "json_schema": {"schema": {"topic_id": topic.id}}
                },
            }

        def generate(
            self,
            topic: Topic,
            *,
            cache: bool = True,
            response_hook=None,
        ) -> _FakeResult:
            generate_calls.append(topic.id)
            outcome = outcomes[topic.id]
            if response_hook is not None and (
                (
                    not isinstance(outcome, Exception)
                    and not outcome.cache_hit
                )
                or isinstance(outcome, QueryPlanGenerationError)
            ):
                response = (
                    HttpJsonResponse(
                        outcome.response,
                        raw_body=b'\xff{"wire":"invalid utf8"}',
                        http_status=200,
                        response_headers={"Content-Type": "application/json"},
                    )
                    if isinstance(outcome, QueryPlanGenerationError)
                    else {"id": f"response-{topic.id}", "choices": []}
                )
                response_hook(response, 0.005)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def fake_render(topic: Topic, plan: _FakePlan) -> list[RenderedQuery]:
        if topic.id == "render-failure":
            raise RuntimeError("rendering failed for one topic")
        return [
            RenderedQuery(
                variant_name="parent",
                source_type="llm_parent",
                query_text=topic.narrative,
                components=(topic.narrative,),
                unique_content_tokens=(topic.id,),
            )
        ]

    monkeypatch.setattr(query_plan_cli, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(query_plan_cli, "load_repo_env", lambda _: None)
    monkeypatch.setattr(query_plan_cli, "load_topics", lambda *args, **kwargs: topics)
    monkeypatch.setattr(query_plan_cli, "QueryPlanGenerator", FakeGenerator)
    monkeypatch.setattr(query_plan_cli, "render_query_plan", fake_render)

    output = tmp_path / "results" / "plans.jsonl"
    failure_output = tmp_path / "results" / "failures.jsonl"
    outcome_dir = tmp_path / "results" / "outcomes" / "mixed-run"
    args = argparse.Namespace(
        topics=tmp_path / "topics.tsv",
        topic_format="tsv",
        topic_ids=topic_ids,
        output=output,
        failure_output=failure_output,
        cache_dir=tmp_path / "cache",
        base_url="http://planner.test/v1",
        model="planner-test-model",
        model_revision="test-revision",
        api_key=None,
        reasoning_effort="low",
        max_tokens=256,
        temperature=0.0,
        seed=7,
        max_facets=4,
        timeout=1.0,
        workers=4,
        no_cache=False,
        run_id="mixed-run",
        attempt_kind="first_emission",
        outcome_dir=outcome_dir,
        rebuild_only=False,
    )

    status = query_plan_cli.generate_plans(args)
    manifest_path = output.with_name("plans.manifest.json")
    return _MixedRun(
        status=status,
        output=output,
        failure_output=failure_output,
        manifest_path=manifest_path,
        outcome_dir=outcome_dir,
        args=args,
        generate_calls=generate_calls,
    )


def test_parallel_mixed_batch_writes_success_failure_and_manifest_ledgers(
    monkeypatch, tmp_path: Path
):
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)

    assert run.status == 1
    assert run.output.is_file()
    assert run.failure_output.is_file()
    assert run.manifest_path.is_file()

    records = _read_jsonl(run.output)
    assert [record["topic"]["id"] for record in records] == ["cached", "fresh"]
    assert [record["cache_hit"] for record in records] == [True, False]
    assert records[0]["cache_path"].endswith("cache/cached.json")
    assert records[1]["cache_path"].endswith("cache/fresh.json")

    failures = _read_jsonl(run.failure_output)
    assert [failure["topic"]["id"] for failure in failures] == [
        "invalid-response",
        "timeout",
        "invalid-cache",
        "unexpected",
        "render-failure",
    ]

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["successful"] == 2
    assert manifest["failed"] == 5
    assert manifest["workers"] == 4
    assert manifest["missing"] == 0
    assert manifest["output"] == str(run.output)
    assert manifest["failure_output"] == str(run.failure_output)

    outcome_files = sorted(run.outcome_dir.glob("[0-9][0-9]_*.json"))
    assert [path.name for path in outcome_files] == [
        "01_cached.json",
        "02_invalid-response.json",
        "03_timeout.json",
        "04_invalid-cache.json",
        "05_unexpected.json",
        "06_render-failure.json",
        "07_fresh.json",
    ]
    outcomes = [json.loads(path.read_text(encoding="utf-8")) for path in outcome_files]
    assert [outcome["topic_id"] for outcome in outcomes] == list(run.args.topic_ids)
    assert [outcome["status"] for outcome in outcomes] == [
        "success",
        "plan_validation_error",
        "timeout",
        "plan_validation_error",
        "unexpected_exception",
        "unexpected_exception",
        "success",
    ]
    assert not list(tmp_path.rglob("*.tmp"))


def test_failure_ledger_classifies_failures_without_fabricating_model_payloads(
    monkeypatch, tmp_path: Path
):
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)
    failures = {
        failure["topic"]["id"]: failure
        for failure in _read_jsonl(run.failure_output)
    }

    invalid_response = failures["invalid-response"]
    assert invalid_response["failure_kind"] == "plan_validation"
    assert invalid_response["error_type"] == "QueryPlanGenerationError"
    assert invalid_response["response"] == {"id": "response-with-invalid-plan"}
    assert invalid_response["content"] == '{"facet_count": "not-an-integer"}'

    invalid_cache = failures["invalid-cache"]
    assert invalid_cache["failure_kind"] == "plan_validation"
    assert invalid_cache["error_type"] == "QueryPlanValidationError"
    assert "response" not in invalid_cache
    assert "content" not in invalid_cache

    transport = failures["timeout"]
    assert transport["failure_kind"] == "transport"
    assert transport["error_type"] == "TimeoutError"
    assert "response" not in transport
    assert "content" not in transport

    unexpected = failures["unexpected"]
    assert unexpected["failure_kind"] == "unexpected"
    assert unexpected["error_type"] == "RuntimeError"
    assert "response" not in unexpected
    assert "content" not in unexpected

    render_failure = failures["render-failure"]
    assert render_failure["failure_kind"] == "unexpected"
    assert render_failure["error_type"] == "RuntimeError"
    assert "response" not in render_failure
    assert "content" not in render_failure


def test_raw_response_is_committed_before_invalid_plan_outcome(
    monkeypatch, tmp_path: Path
):
    original_atomic_write_json = query_plan_cli._atomic_write_json
    write_order: list[Path] = []

    def recording_atomic_write_json(path: Path, payload) -> None:
        if payload.get("topic_id") == "invalid-response" and "failure" in payload:
            raw_path = Path(payload["raw_response_path"])
            assert raw_path.is_file()
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            assert raw_record["topic_id"] == "invalid-response"
            assert raw_record["response"] == {"id": "response-with-invalid-plan"}
            assert base64.b64decode(raw_record["raw_body_base64"]) == (
                b'\xff{"wire":"invalid utf8"}'
            )
            assert raw_record["raw_body_sha256"]
        original_atomic_write_json(path, payload)
        write_order.append(path)

    monkeypatch.setattr(
        query_plan_cli, "_atomic_write_json", recording_atomic_write_json
    )
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)

    raw_path = run.outcome_dir / "raw_responses" / "02_invalid-response.json"
    outcome_path = run.outcome_dir / "02_invalid-response.json"
    assert raw_path in write_order
    assert outcome_path in write_order
    assert write_order.index(raw_path) < write_order.index(outcome_path)
    assert not list(run.outcome_dir.rglob("*.tmp"))


def test_rebuild_only_reconstructs_summaries_without_generating_again(
    monkeypatch, tmp_path: Path
):
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)
    expected_records = _read_jsonl(run.output)
    expected_failures = _read_jsonl(run.failure_output)
    outcome_snapshots = {
        path.name: path.read_bytes()
        for path in run.outcome_dir.glob("[0-9][0-9]_*.json")
    }
    generate_call_count = len(run.generate_calls)

    run.output.write_text("stale plan summary\n", encoding="utf-8")
    run.failure_output.write_text("stale failure summary\n", encoding="utf-8")
    run.manifest_path.write_text("stale manifest\n", encoding="utf-8")
    run.args.rebuild_only = True

    status = query_plan_cli.generate_plans(run.args)

    assert status == 1
    assert len(run.generate_calls) == generate_call_count
    assert _read_jsonl(run.output) == expected_records
    assert _read_jsonl(run.failure_output) == expected_failures
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["successful"] == 2
    assert manifest["failed"] == 5
    assert manifest["missing"] == 0
    assert {
        path.name: path.read_bytes()
        for path in run.outcome_dir.glob("[0-9][0-9]_*.json")
    } == outcome_snapshots
    assert not list(run.outcome_dir.rglob("*.tmp"))


def test_rebuild_only_uses_original_run_identity_order_and_frozen_settings(
    monkeypatch, tmp_path: Path
):
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)
    expected_records = _read_jsonl(run.output)
    expected_failures = _read_jsonl(run.failure_output)
    generate_call_count = len(run.generate_calls)

    run.args.rebuild_only = True
    run.args.topic_ids = ("cached",)
    run.args.run_id = "misleading-rebuild-id"
    run.args.attempt_kind = "targeted_revision"
    run.args.model = "different-current-model"
    run.args.workers = 1
    run.args.timeout = 99.0
    monkeypatch.setattr(query_plan_cli, "SCHEMA_VERSION", "future_schema")
    monkeypatch.setattr(query_plan_cli, "PROMPT_VERSION", "future_prompt")
    monkeypatch.setattr(query_plan_cli, "RENDERER_VERSION", "future_renderer")
    monkeypatch.setattr(query_plan_cli, "ANALYZER_VERSION", "future_analyzer")

    status = query_plan_cli.generate_plans(run.args)

    assert status == 1
    assert len(run.generate_calls) == generate_call_count
    assert _read_jsonl(run.output) == expected_records
    assert _read_jsonl(run.failure_output) == expected_failures
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "mixed-run"
    assert manifest["attempt_kind"] == "first_emission"
    assert manifest["request_order"] == list(
        (
            "cached",
            "invalid-response",
            "timeout",
            "invalid-cache",
            "unexpected",
            "render-failure",
            "fresh",
        )
    )
    assert manifest["model"] == "planner-test-model"
    assert manifest["workers"] == 4
    assert manifest["timeout_seconds"] == 1.0
    assert manifest["schema_version"] == "query_plan_v1"
    assert manifest["prompt_version"] != "future_prompt"
    assert manifest["renderer_version"] != "future_renderer"
    assert manifest["analyzer_version"] != "future_analyzer"


def test_rebuild_rejects_nested_record_identity_tampering(monkeypatch, tmp_path: Path):
    run = _run_mixed_parallel_batch(monkeypatch, tmp_path)
    outcome_path = run.outcome_dir / "07_fresh.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["record"]["run_id"] = "different-run"
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    run.args.rebuild_only = True

    with pytest.raises(ValueError, match="record identity mismatch.*run_id"):
        query_plan_cli.generate_plans(run.args)

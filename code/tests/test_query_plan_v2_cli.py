import argparse
import hashlib
import json
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag import query_plan_v2_cli
from trec_rag.query_analyzer import AnalyzedQuery, AnalyzerFingerprint
from trec_rag.query_planner import (
    HttpJsonResponse,
    PlanV2Failure,
    PlanV2Outcome,
    RenderedQuery,
    tokenize_narrative,
)
from trec_rag.topics import Topic


_FINGERPRINT = AnalyzerFingerprint(
    contract_version="lucene_default_english_v1",
    implementation="test_lucene_reference",
    lucene_version="10.4.0",
    analyzer_class="org.apache.lucene.analysis.en.EnglishAnalyzer",
    tokenizer="StandardTokenizer",
    filters=("EnglishPossessiveFilter", "LowerCaseFilter", "StopFilter", "PorterStemFilter"),
    stopword_sha256="a" * 64,
    unicode_version=None,
    index_id="test-index",
)


@dataclass(frozen=True)
class _FakeProvenance:
    analyzer_fingerprint: dict[str, object]
    provider: str
    base_url: str
    model: str
    model_revision: str
    schema_version: str
    prompt_version: str
    renderer_version: str
    tokenizer_version: str
    schema_sha256: str
    prompt_sha256: str
    request_sha256: str
    instruction_role: str
    reasoning_effort: str
    enable_thinking: bool
    max_tokens: int
    temperature: float
    top_p: None = None
    top_k: None = None
    presence_penalty: None = None
    seed: int = 0
    response_model: str = "gpt-oss-local"
    elapsed_seconds: float = 0.01


class _FakePlan:
    analyzer_fingerprint = _FINGERPRINT
    facets = (object(),)

    def __init__(self, topic_id: str) -> None:
        self.topic_id = topic_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": query_plan_v2_cli.V2_SCHEMA_VERSION,
            "topic_id": self.topic_id,
            "analyzer_fingerprint": self.analyzer_fingerprint.to_dict(),
            "facet_count": 1,
            "facets": [{"facet_id": "f1"}],
        }


@dataclass(frozen=True)
class _FakeResult:
    outcome: PlanV2Outcome
    provenance: _FakeProvenance
    token_tape: object


def _fake_request(topic) -> dict[str, object]:
    return {
        "model": "gpt-oss-local",
        "messages": [{"role": "developer", "content": "test prompt"}],
        "response_format": {"json_schema": {"schema": {"topic_id": topic.id}}},
    }


def _result(topic, *, fallback_status: str | None = None) -> _FakeResult:
    rendered = RenderedQuery(
        variant_name="original" if fallback_status else "facet_1",
        source_type="original" if fallback_status else "llm_facet",
        query_text=topic.narrative,
        components=(topic.narrative,),
        unique_content_tokens=("synthet", "transport"),
    )
    failure = (
        PlanV2Failure(
            status=fallback_status,
            error_type="QueryPlanGenerationError",
            error="model output was not valid JSON",
            hard_failure=False,
            expansion_audit=(),
        )
        if fallback_status
        else None
    )
    outcome = PlanV2Outcome(
        used_fallback=fallback_status is not None,
        plan=None if fallback_status else _FakePlan(topic.id),
        failure=failure,
        rendered_queries=(rendered,),
    )
    request = _fake_request(topic)
    schema = request["response_format"]["json_schema"]["schema"]
    return _FakeResult(
        outcome=outcome,
        provenance=_FakeProvenance(
            analyzer_fingerprint=_FINGERPRINT.to_dict(),
            provider="openai_compatible_chat_completions",
            base_url="http://127.0.0.1:8000/v1",
            model="gpt-oss-local",
            model_revision=query_plan_v2_cli.GPT_OSS_20B_REVISION,
            schema_version=query_plan_v2_cli.V2_SCHEMA_VERSION,
            prompt_version=query_plan_v2_cli.V2_PROMPT_VERSION,
            renderer_version=query_plan_v2_cli.V2_RENDERER_VERSION,
            tokenizer_version=query_plan_v2_cli.TOKENIZER_VERSION,
            schema_sha256=query_plan_v2_cli._sha256_request_json(schema),
            prompt_sha256=hashlib.sha256(b"test prompt").hexdigest(),
            request_sha256=query_plan_v2_cli._sha256_request_json(request),
            instruction_role="developer",
            reasoning_effort="low",
            enable_thinking=False,
            max_tokens=256,
            temperature=0.0,
        ),
        token_tape=tokenize_narrative(topic.narrative),
    )


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "run_kind": "synthetic_transport_smoke",
        "topics": tmp_path / "topics.tsv",
        "topic_format": "tsv",
        "topic_ids": None,
        "output": tmp_path / "results" / "plans.jsonl",
        "failure_output": tmp_path / "results" / "failures.jsonl",
        "outcome_dir": tmp_path / "results" / "outcomes" / "test-run",
        "run_id": "test-run",
        "rebuild_only": False,
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "gpt-oss-local",
        "model_revision": query_plan_v2_cli.GPT_OSS_20B_REVISION,
        "api_key": None,
        "allow_remote_model": False,
        "instruction_role": "developer",
        "reasoning_effort": "low",
        "enable_thinking": False,
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "presence_penalty": None,
        "seed": 0,
        "timeout": 2.0,
        "analyzer_url": "http://127.0.0.1:18081",
        "analyzer_timeout": 1.0,
        "expected_analyzer_fingerprint_sha256": query_plan_v2_cli._sha256_json(
            _FINGERPRINT.to_dict()
        ),
        "schema_preflight_manifest": tmp_path / "schema-preflight.json",
        "server_tokenizer_url": "http://127.0.0.1:8000/tokenize",
        "tokenizer_timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _install_fakes(
    monkeypatch,
    tmp_path: Path,
    args: argparse.Namespace,
    *,
    fallback_status: str | None = None,
    post_raw_exception: Exception | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {"events": [], "generate_calls": 0}

    class FakeAnalyzer:
        def __init__(
            self,
            base_url: str,
            *,
            timeout: float,
            expected_fingerprint: AnalyzerFingerprint | None = None,
        ) -> None:
            state["events"].append("analyzer_init")
            self.base_url = base_url
            self.timeout = timeout
            assert expected_fingerprint in {None, _FINGERPRINT}

        @property
        def fingerprint(self) -> AnalyzerFingerprint:
            state["events"].append("analyzer_fingerprint")
            return _FINGERPRINT

        def analyze(self, text: str) -> AnalyzedQuery:
            state["events"].append("analyzer_preflight")
            return AnalyzedQuery(
                tokens=("synthet", "transport"),
                unique_tokens=("synthet", "transport"),
                fingerprint=_FINGERPRINT,
            )

    class FakeGenerator:
        def __init__(self, **kwargs) -> None:
            state["events"].append("generator_init")
            self.base_url = kwargs["base_url"]
            self.model = kwargs["model"]
            self.model_revision = kwargs["model_revision"]
            self.timeout = kwargs["timeout"]
            self.max_tokens = kwargs["max_tokens"]
            self.temperature = kwargs["temperature"]
            self.top_p = kwargs["top_p"]
            self.top_k = kwargs["top_k"]
            self.presence_penalty = kwargs["presence_penalty"]
            self.reasoning_effort = kwargs["reasoning_effort"]
            self.enable_thinking = kwargs["enable_thinking"]
            self.instruction_role = kwargs["instruction_role"]
            self.seed = kwargs["seed"]

        def request_payload(self, topic) -> dict[str, object]:
            return _fake_request(topic)

        def generate(self, topic, *, response_hook=None) -> _FakeResult:
            state["generate_calls"] += 1
            state["events"].append("generator_generate")
            response = HttpJsonResponse(
                {
                    "id": "response-test",
                    "model": self.model,
                    "usage": {"completion_tokens": 7},
                },
                raw_body=b'{"wire":"exact response bytes"}',
                http_status=200,
                response_headers={"Content-Type": "application/json"},
            )
            assert response_hook is not None
            response_hook(response, 0.005)
            raw_files = list((args.outcome_dir / "raw_responses").glob("*.json"))
            outcome_files = list(args.outcome_dir.glob("[0-9][0-9]_*.json"))
            state["raw_committed_before_result"] = len(raw_files) == 1
            state["outcome_absent_before_result"] = not outcome_files
            if post_raw_exception is not None:
                raise post_raw_exception
            state["events"].append("model_result_parse")
            return _result(topic, fallback_status=fallback_status)

    monkeypatch.setattr(query_plan_v2_cli, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(query_plan_v2_cli, "load_repo_env", lambda _: None)
    monkeypatch.setattr(
        query_plan_v2_cli, "RemoteLuceneQueryAnalyzer", FakeAnalyzer
    )
    monkeypatch.setattr(query_plan_v2_cli, "QueryPlanGeneratorV2", FakeGenerator)
    monkeypatch.setattr(
        query_plan_v2_cli,
        "_verify_schema_preflight",
        lambda path, *, topics: {
            "path": str(path),
            "sha256": "b" * 64,
            "manifest": {"topics": [topic.id for topic in topics]},
        },
    )
    monkeypatch.setattr(
        query_plan_v2_cli,
        "_verify_live_runtime",
        lambda schema_preflight, **kwargs: {
            "verified_at": "test",
            "container_name": "fake-local-container",
            "schema_preflight_sha256": schema_preflight["sha256"],
            "endpoint": kwargs["base_url"],
        },
    )
    monkeypatch.setattr(
        query_plan_v2_cli,
        "_verify_source_revision",
        lambda _repo_root: {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test-branch",
            "tracked_worktree_clean": True,
        },
    )
    monkeypatch.setattr(
        query_plan_v2_cli,
        "_context_preflight",
        lambda **kwargs: {
            "topic_id": kwargs["topic"].id,
            "tokenizer_url": kwargs["tokenizer_url"],
            "prompt_tokens": 100,
            "max_output_tokens": kwargs["generator"].max_tokens,
            "requested_total_tokens": 100 + kwargs["generator"].max_tokens,
            "max_model_len": 8192,
            "remaining_context_tokens": 8192
            - 100
            - kwargs["generator"].max_tokens,
        },
    )
    return state


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_valid_result_preflights_analyzer_and_commits_raw_bytes_first(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path)
    state = _install_fakes(monkeypatch, tmp_path, args)

    status = query_plan_v2_cli.generate_plans(args)

    assert status == 0
    assert state["generate_calls"] == 1
    events = state["events"]
    assert events.index("analyzer_preflight") < events.index("generator_init")
    assert events.index("generator_init") < events.index("generator_generate")
    assert state["raw_committed_before_result"] is True
    assert state["outcome_absent_before_result"] is True
    records = _read_jsonl(args.output)
    assert len(records) == 1
    assert records[0]["topic"]["id"] == query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.id
    assert records[0]["plan"]["facet_count"] == 1
    assert _read_jsonl(args.failure_output) == []

    raw_path = next((args.outcome_dir / "raw_responses").glob("*.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    exact = b'{"wire":"exact response bytes"}'
    assert raw["capture_level"] == "exact_http_body_before_decode_or_json_parse"
    assert raw["raw_body_sha256"] == hashlib.sha256(exact).hexdigest()


def test_returned_original_only_fallback_is_classified_as_failure(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args, fallback_status="invalid_json")

    status = query_plan_v2_cli.generate_plans(args)

    assert status == 1
    assert _read_jsonl(args.output) == []
    failures = _read_jsonl(args.failure_output)
    assert len(failures) == 1
    assert failures[0]["status"] == "invalid_json"
    assert failures[0]["failure"]["status"] == "invalid_json"
    assert failures[0]["fallback_rendered_queries"] == [
        {
            "components": [query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.narrative],
            "query_text": query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.narrative,
            "renderer_budget_exception": False,
            "source_type": "original",
            "unique_content_tokens": ["synthet", "transport"],
            "variant_name": "original",
        }
    ]
    manifest = json.loads(
        args.output.with_name("plans.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status_counts"] == {"invalid_json": 1}


def test_create_only_writes_and_run_artifact_collisions_are_refused(
    monkeypatch, tmp_path: Path
):
    create_only = tmp_path / "create-only.json"
    query_plan_v2_cli._write_json(create_only, {"value": "first"}, replace=False)
    with pytest.raises(FileExistsError):
        query_plan_v2_cli._write_json(
            create_only, {"value": "replacement"}, replace=False
        )
    assert json.loads(create_only.read_text(encoding="utf-8")) == {"value": "first"}

    args = _args(tmp_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("existing evidence\n", encoding="utf-8")

    class NoService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("artifact collision must be checked before services")

    monkeypatch.setattr(query_plan_v2_cli, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(query_plan_v2_cli, "load_repo_env", lambda _: None)
    monkeypatch.setattr(query_plan_v2_cli, "RemoteLuceneQueryAnalyzer", NoService)
    monkeypatch.setattr(query_plan_v2_cli, "QueryPlanGeneratorV2", NoService)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        query_plan_v2_cli.generate_plans(args)
    assert args.output.read_text(encoding="utf-8") == "existing evidence\n"


def test_rebuild_uses_only_committed_ledgers_and_makes_zero_service_calls(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args)
    assert query_plan_v2_cli.generate_plans(args) == 0
    expected = _read_jsonl(args.output)
    args.output.write_text("stale summary\n", encoding="utf-8")
    args.failure_output.write_text("stale summary\n", encoding="utf-8")

    class NoService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("rebuild must not instantiate a service client")

    monkeypatch.setattr(query_plan_v2_cli, "RemoteLuceneQueryAnalyzer", NoService)
    monkeypatch.setattr(query_plan_v2_cli, "QueryPlanGeneratorV2", NoService)
    args.rebuild_only = True

    assert query_plan_v2_cli.generate_plans(args) == 0
    assert _read_jsonl(args.output) == expected


def test_rebuild_rejects_tampered_raw_response_sha(monkeypatch, tmp_path: Path):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args)
    assert query_plan_v2_cli.generate_plans(args) == 0
    raw_path = next((args.outcome_dir / "raw_responses").glob("*.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["raw_body_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    args.rebuild_only = True

    with pytest.raises(ValueError, match="raw response hash mismatch"):
        query_plan_v2_cli.generate_plans(args)


def test_rebuild_reports_a_missing_terminal_outcome(monkeypatch, tmp_path: Path):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args)
    assert query_plan_v2_cli.generate_plans(args) == 0
    next(args.outcome_dir.glob("[0-9][0-9]_*.json")).unlink()
    next((args.outcome_dir / "raw_responses").glob("*.json")).unlink()
    args.rebuild_only = True

    assert query_plan_v2_cli.generate_plans(args) == 1
    manifest = json.loads(
        args.output.with_name("plans.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["successful"] == 0
    assert manifest["failed"] == 0
    assert manifest["missing"] == 1
    assert manifest["missing_topics"] == [query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.id]


def test_rebuild_reports_valid_raw_without_outcome_as_raw_only_crash_evidence(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args)
    assert query_plan_v2_cli.generate_plans(args) == 0
    next(args.outcome_dir.glob("[0-9][0-9]_*.json")).unlink()
    assert len(list((args.outcome_dir / "raw_responses").glob("*.json"))) == 1
    args.rebuild_only = True

    assert query_plan_v2_cli.generate_plans(args) == 1
    manifest = json.loads(
        args.output.with_name("plans.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["successful"] == 0
    assert manifest["failed"] == 0
    assert manifest["missing_topics"] == [query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.id]
    assert manifest["raw_only_topics"] == [query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC.id]
    assert _read_jsonl(args.failure_output) == []


@pytest.mark.parametrize(
    ("fallback_status", "tamper_kind", "error_match"),
    [
        (None, "success_identity", "successful record identity mismatch"),
        (None, "success_provenance", "generation provenance mismatch"),
        ("invalid_json", "failure_identity", "failure record identity/status mismatch"),
        ("invalid_json", "failure_provenance", "generation provenance mismatch"),
    ],
)
def test_rebuild_rejects_tampered_nested_identity_or_provenance_hash(
    monkeypatch,
    tmp_path: Path,
    fallback_status: str | None,
    tamper_kind: str,
    error_match: str,
):
    args = _args(tmp_path)
    _install_fakes(monkeypatch, tmp_path, args, fallback_status=fallback_status)
    expected_status = 1 if fallback_status else 0
    assert query_plan_v2_cli.generate_plans(args) == expected_status
    outcome_path = next(args.outcome_dir.glob("[0-9][0-9]_*.json"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if tamper_kind == "success_identity":
        outcome["record"]["topic"]["id"] = "different-topic"
    elif tamper_kind == "success_provenance":
        outcome["record"]["provenance"]["request_sha256"] = "0" * 64
    elif tamper_kind == "failure_identity":
        outcome["failure"]["topic_id"] = "different-topic"
    else:
        outcome["failure"]["provenance"]["prompt_sha256"] = "0" * 64
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    args.rebuild_only = True

    with pytest.raises(ValueError, match=error_match):
        query_plan_v2_cli.generate_plans(args)


def test_post_raw_analyzer_urlerror_aborts_before_second_topic_model_call(
    monkeypatch, tmp_path: Path
):
    topic_ids = ("144", "213")
    topics = [
        Topic(id=topic_id, title=f"Topic {topic_id}", narrative=f"Narrative {topic_id}")
        for topic_id in topic_ids
    ]
    monkeypatch.setattr(query_plan_v2_cli, "DIAGNOSTIC_TOPIC_IDS", topic_ids)
    args = _args(
        tmp_path,
        run_kind="diagnostic_first_emission",
        topic_ids=topic_ids,
    )
    state = _install_fakes(
        monkeypatch,
        tmp_path,
        args,
        post_raw_exception=urllib.error.URLError("analyzer sidecar unavailable"),
    )
    monkeypatch.setattr(query_plan_v2_cli, "load_topics", lambda *args, **kwargs: topics)

    assert query_plan_v2_cli.generate_plans(args) == 1
    assert state["generate_calls"] == 1
    outcomes = list(args.outcome_dir.glob("[0-9][0-9]_*.json"))
    raw_responses = list((args.outcome_dir / "raw_responses").glob("*.json"))
    assert len(outcomes) == 1
    assert len(raw_responses) == 1
    outcome = json.loads(outcomes[0].read_text(encoding="utf-8"))
    assert outcome["topic_id"] == "144"
    assert outcome["status"] == "analyzer_integrity_error"
    assert outcome["failure"]["failure_kind"] == "analyzer_integrity"
    manifest = json.loads(
        args.output.with_name("plans.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["failed"] == 1
    assert manifest["missing"] == 1
    assert manifest["missing_topics"] == ["213"]
    assert manifest["status_counts"] == {"analyzer_integrity_error": 1}


def test_synthetic_smoke_refuses_development_topic_ids_before_services(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path, topic_ids=("144",))

    class NoService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("invalid synthetic selection must not call services")

    monkeypatch.setattr(query_plan_v2_cli, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(query_plan_v2_cli, "load_repo_env", lambda _: None)
    monkeypatch.setattr(query_plan_v2_cli, "RemoteLuceneQueryAnalyzer", NoService)
    monkeypatch.setattr(query_plan_v2_cli, "QueryPlanGeneratorV2", NoService)

    with pytest.raises(ValueError, match="synthetic smoke does not accept"):
        query_plan_v2_cli.generate_plans(args)


def _schema_preflight_fixture(topic: Topic) -> dict[str, object]:
    tape = tokenize_narrative(topic.narrative)
    schema = query_plan_v2_cli.query_plan_v2_json_schema(
        topic_id=topic.id,
        token_count=tape.token_count,
    )
    return {
        "manifest_version": "query_plan_v2_schema_preflight_v1",
        "schema_version": query_plan_v2_cli.V2_SCHEMA_VERSION,
        "prompt_version": query_plan_v2_cli.V2_PROMPT_VERSION,
        "required_server_backend": "xgrammar",
        "all_passed": True,
        "compiler": {
            "vllm": "0.24.0",
            "xgrammar": "0.2.3",
            "llguidance": "1.7.6",
            "xgrammar_strict": "pass",
            "llguidance_check": "pass",
        },
        "container": {
            "container_name": "trec-rag-gpt-oss",
            "image_id": "image-id-test",
            "image_digest": query_plan_v2_cli.PINNED_VLLM_IMAGE_DIGEST,
            "launch_command": [
                "openai/gpt-oss-20b",
                "--revision",
                query_plan_v2_cli.GPT_OSS_20B_REVISION,
                "--served-model-name",
                "gpt-oss-local",
                "--max-model-len",
                "8192",
                "--port",
                "8000",
                "--structured-outputs-config",
                '{"backend":"xgrammar"}',
            ],
        },
        "schemas": [
            {
                "topic_id": topic.id,
                "narrative_sha256": tape.narrative_sha256,
                "token_count": tape.token_count,
                "schema_sha256": query_plan_v2_cli._sha256_request_json(schema),
                "unsupported_feature_count": 0,
                "xgrammar_strict": "pass",
            }
        ],
    }


def test_schema_preflight_manifest_binds_exact_schema_and_xgrammar_backend(tmp_path):
    topic = query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC
    path = tmp_path / "compiler-manifest.json"
    path.write_text(json.dumps(_schema_preflight_fixture(topic)), encoding="utf-8")

    verified = query_plan_v2_cli._verify_schema_preflight(path, topics=[topic])

    assert verified["path"] == str(path.resolve())
    assert verified["manifest"]["required_server_backend"] == "xgrammar"


def test_schema_preflight_manifest_rejects_schema_hash_or_backend_drift(tmp_path):
    topic = query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC
    manifest = _schema_preflight_fixture(topic)
    path = tmp_path / "compiler-manifest.json"
    manifest["schemas"][0]["schema_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence mismatch"):
        query_plan_v2_cli._verify_schema_preflight(path, topics=[topic])

    manifest = _schema_preflight_fixture(topic)
    manifest["container"]["launch_command"] = ["--structured-outputs-config.backend", "auto"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="XGrammar backend"):
        query_plan_v2_cli._verify_schema_preflight(path, topics=[topic])


def test_live_runtime_attestation_binds_container_port_version_and_model(monkeypatch):
    topic = query_plan_v2_cli.SYNTHETIC_SMOKE_TOPIC
    manifest = _schema_preflight_fixture(topic)
    command = manifest["container"]["launch_command"]
    inspected = [
        {
            "State": {"Running": True},
            "Image": "image-id-test",
            "Config": {"Cmd": command},
            "NetworkSettings": {
                "Ports": {
                    "8000/tcp": [
                        {"HostIp": "127.0.0.1", "HostPort": "8000"}
                    ]
                }
            },
        }
    ]
    image = [{"Digest": query_plan_v2_cli.PINNED_VLLM_IMAGE_DIGEST}]

    def fake_run(command, **_kwargs):
        payload = image if command[1:3] == ["image", "inspect"] else inspected
        return SimpleNamespace(stdout=json.dumps(payload))

    def fake_local_json(url, **_kwargs):
        if url.endswith("/version"):
            return {"version": "0.24.0"}
        return {
            "data": [
                {
                    "id": "gpt-oss-local",
                    "root": "openai/gpt-oss-20b",
                    "max_model_len": 8192,
                }
            ]
        }

    monkeypatch.setattr(query_plan_v2_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(query_plan_v2_cli, "_read_local_json", fake_local_json)
    preflight = {"manifest": manifest, "sha256": "a" * 64}

    verified = query_plan_v2_cli._verify_live_runtime(
        preflight,
        base_url="http://127.0.0.1:8000/v1",
        tokenizer_url="http://127.0.0.1:8000/tokenize",
        expected_model="gpt-oss-local",
        timeout=1.0,
    )

    assert verified["container_name"] == "trec-rag-gpt-oss"
    assert verified["model_record"]["root"] == "openai/gpt-oss-20b"

    with pytest.raises(ValueError, match="not bound"):
        query_plan_v2_cli._verify_live_runtime(
            preflight,
            base_url="http://127.0.0.1:9000/v1",
            tokenizer_url="http://127.0.0.1:9000/tokenize",
            expected_model="gpt-oss-local",
            timeout=1.0,
        )


def test_source_revision_attestation_requires_clean_committed_tree(monkeypatch, tmp_path):
    values = {
        ("status", "--porcelain", "--untracked-files=no"): "",
        ("rev-parse", "HEAD"): "1" * 40 + "\n",
        ("rev-parse", "HEAD^{tree}"): "2" * 40 + "\n",
        ("branch", "--show-current"): "codex/test\n",
    }

    def fake_run(command, **_kwargs):
        return SimpleNamespace(stdout=values[tuple(command[1:])])

    monkeypatch.setattr(query_plan_v2_cli.subprocess, "run", fake_run)
    verified = query_plan_v2_cli._verify_source_revision(tmp_path)
    assert verified == {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "branch": "codex/test",
        "tracked_worktree_clean": True,
    }

    values[("status", "--porcelain", "--untracked-files=no")] = " M code/file.py\n"
    with pytest.raises(ValueError, match="not clean"):
        query_plan_v2_cli._verify_source_revision(tmp_path)

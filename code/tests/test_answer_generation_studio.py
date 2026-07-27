from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import requests

from trec_rag.answer_generation_studio import (
    GenerationRunResult,
    run_best_answer_generation,
    validate_frozen_evidence_for_generation,
    validate_run_id,
)
from trec_rag.answer_generation_server import (
    StudioApplication,
    build_evidence_summary,
    create_server,
)
from trec_rag.topic213_controlled_generator_benchmark import (
    GENERATION_SYSTEM_PROMPT,
    GenerationCompletion,
    semantic_request_sha256,
)


def _frozen_fixture() -> dict[str, object]:
    facets: list[dict[str, object]] = []
    position = 0
    for facet_number in range(1, 3):
        claims: list[dict[str, object]] = []
        for claim_number in range(1, 3):
            position += 1
            document_id = f"shard_{facet_number:05d}_{claim_number}"
            text = f"Grounded fact {position} contains stable evidence for testing."
            claims.append(
                {
                    "position": position,
                    "claim_id": f"A{position:03d}",
                    "text": text,
                    "sub_narrative": f"Facet {facet_number}",
                    "document_ids": [document_id],
                    "selected_citation_passages": [
                        {
                            "passage_id": f"P{position:03d}",
                            "document_id": document_id,
                            "text": text,
                        }
                    ],
                    "available_supporting_passage_count": 1,
                }
            )
        facets.append(
            {
                "facet_number": facet_number,
                "sub_narrative": f"Facet {facet_number}",
                "sentence_quota": 2,
                "word_target": 16,
                "source_claims": claims,
            }
        )
    return {
        "schema_version": "topic213-controlled-generator-benchmark-v1",
        "prompt_version": "claim-preserving-42-slot-generation-v1",
        "topic_id": "demo-1",
        "narrative": "Explain the grounded demo facts.",
        "source_experiment_id": "demo-source",
        "organizer_nuggets_available": False,
        "supported_source_claim_count": 4,
        "source_claim_word_count": 32,
        "exact_total_sentence_count": 4,
        "word_band": {"minimum": 28, "maximum": 40},
        "facet_word_target_total": 32,
        "input_accounting": {},
        "source_metadata": {},
        "facets": facets,
    }


class FakeGenerator:
    model = "openai/gpt-5.6-sol"

    def complete_once(self, *, system_prompt, payload, max_tokens, temperature):
        if "facets" in payload:
            sentences = [
                {"claim_id": claim["claim_id"], "text": claim["text"]}
                for facet in payload["facets"]
                for claim in facet["claims"]
            ]
        else:
            sentences = [
                {"claim_id": claim["claim_id"], "text": claim["source_claim"]}
                for claim in payload["sentences"]
            ]
        parsed = {"sentences": sentences}
        semantic_hash = semantic_request_sha256(
            system_prompt=system_prompt,
            payload=payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return GenerationCompletion(
            parsed=parsed,
            raw_content=json.dumps(parsed),
            receipt={
                "semantic_request_sha256": semantic_hash,
                "request_sha256": "fake-request",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01},
                "elapsed_seconds": 0.1,
                "network_requests": 1,
            },
        )


class FakeJudge:
    model = "qwen-local"

    def __init__(self):
        self.seen_claim_ids: set[str] = set()

    def complete_json(self, *, stage, system_prompt, payload, max_tokens, temperature):
        claim = payload["claims"][0]
        reject = claim["claim_id"] == "S002" and claim["claim_id"] not in self.seen_claim_ids
        self.seen_claim_ids.add(claim["claim_id"])
        return {
            "assessments": [
                {
                    "claim_id": claim["claim_id"],
                    "status": "unsupported" if reject else "supported",
                    "notes": "fixture decision",
                }
            ]
        }


def _config() -> dict[str, object]:
    return {
        "experiment_id": "answer-generation-studio-test",
        "generation": {
            "key": "gpt_5_6_sol",
            "model": "openai/gpt-5.6-sol",
            "model_identity": "openai/gpt-5.6-sol",
            "max_tokens": 3500,
            "temperature": 0.0,
        },
        "audit": {
            "model_identity": "Qwen/Qwen3-4B-Instruct-2507",
            "max_tokens": 350,
            "temperature": 0.0,
            "validation_attempts": 1,
        },
    }


def test_frozen_evidence_validation_is_nugget_blind_and_exact():
    validated = validate_frozen_evidence_for_generation(_frozen_fixture())

    assert validated["exact_total_sentence_count"] == 4
    assert validated["supported_source_claim_count"] == 4

    contaminated = _frozen_fixture()
    contaminated["organizer_nuggets_available"] = True
    with pytest.raises(ValueError, match="nugget-blind"):
        validate_frozen_evidence_for_generation(contaminated)


def test_run_id_is_path_safe():
    assert validate_run_id("rag26-topic-213-v1") == "rag26-topic-213-v1"
    with pytest.raises(ValueError, match="run ID"):
        validate_run_id("../escape")


def test_best_answer_generation_writes_valid_frozen_bundle(tmp_path: Path):
    events: list[dict[str, object]] = []
    run_dir = tmp_path / "run"

    result = run_best_answer_generation(
        frozen=_frozen_fixture(),
        output_dir=run_dir,
        run_id="rag26-demo-v1",
        team_id="demo-team",
        run_desc="Controlled claim-preserving generation.",
        config=_config(),
        generator_client=FakeGenerator(),
        judge_client=FakeJudge(),
        progress=events.append,
    )

    assert isinstance(result, GenerationRunResult)
    assert result.summary["candidate_sentence_count"] == 4
    assert result.summary["submitted_sentence_count"] == 4
    assert result.summary["excluded_sentence_count"] == 0
    assert result.summary["initially_rejected_sentence_count"] == 1
    assert result.summary["repaired_and_retained_sentence_count"] == 1
    assert result.summary["citation_coverage"] == 1.0
    assert result.summary["organizer_nuggets_read"] is False
    assert result.summary["official_format_valid"] is True
    assert events[0]["stage"] == "preflight"
    assert events[-1]["stage"] == "complete"

    submission_lines = (run_dir / "rag_output_trec_rag_2026.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(submission_lines) == 1
    submission = json.loads(submission_lines[0])
    assert len(submission["answer"]) == 4
    assert all(len(item["citations"]) == 1 for item in submission["answer"])
    assert (run_dir / "repair_receipt.json").is_file()

    freeze = json.loads((run_dir / "generation_freeze.json").read_text(encoding="utf-8"))
    assert freeze["organizer_nuggets_read"] is False
    assert len(freeze["artifacts"]) >= 10
    assert (run_dir / "manifest.yaml").is_file()


def test_generation_rejects_existing_nonempty_run_directory(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_best_answer_generation(
            frozen=_frozen_fixture(),
            output_dir=run_dir,
            run_id="rag26-demo-v1",
            team_id="demo-team",
            run_desc="Controlled claim-preserving generation.",
            config=_config(),
            generator_client=FakeGenerator(),
            judge_client=FakeJudge(),
        )


def test_evidence_summary_exposes_facets_without_passage_text():
    summary = build_evidence_summary(_frozen_fixture())

    assert summary["topic_id"] == "demo-1"
    assert summary["facet_count"] == 2
    assert summary["claim_count"] == 4
    assert summary["candidate_word_band"] == {"minimum": 28, "maximum": 40}
    assert "passage" not in json.dumps(summary).casefold()


@pytest.mark.parametrize("benchmark_layout", ["comparison", "coverage_repair"])
def test_http_app_runs_background_job_and_serves_submission(
    tmp_path: Path, benchmark_layout: str
):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_frozen_fixture()), encoding="utf-8")
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    if benchmark_layout == "comparison":
        (benchmark_dir / "comparison_metrics.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_key": "gpt_5_6_sol",
                            "display_name": "GPT-5.6 Sol",
                            "strict_coverage": 0.46,
                            "partial_credit_coverage": 0.52,
                            "vital_strict_coverage": 0.444,
                        }
                    ],
                    "winner": {"model_key": "gpt_5_6_sol"},
                }
            ),
            encoding="utf-8",
        )
        preview_dir = benchmark_dir / "gpt_5_6_sol"
        preview_dir.mkdir()
    else:
        (benchmark_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model": {
                        "key": "gpt_5_6_sol",
                        "display_name": "GPT-5.6 Sol",
                        "identity": "openai/gpt-5.6-sol",
                    },
                    "nuggets": {
                        "all": {
                            "strict_coverage": 0.54,
                            "partial_credit_coverage": 0.63,
                        },
                        "vital": {"strict_coverage": 0.63},
                    },
                    "official_submission": {
                        "candidate_word_count": 940,
                        "word_count": 941,
                        "sentence_count": 48,
                        "excluded_after_repair": 0,
                    },
                    "answer_claims": {
                        "citation_coverage": 1.0,
                        "unsupported_claim_count": 0,
                    },
                    "generation": {"cost": 0.1279},
                    "generation_freeze_sha256": "fixture-freeze",
                }
            ),
            encoding="utf-8",
        )
        preview_dir = benchmark_dir
    preview_entry = {
        "metadata": {"narrative_id": "demo-1", "run_id": "preview"},
        "references": ["shard_00001_1"],
        "answer": [{"text": "Preview sentence.", "citations": ["shard_00001_1"]}],
    }
    (preview_dir / "rag_output_trec_rag_2026.jsonl").write_text(
        json.dumps(preview_entry) + "\n", encoding="utf-8"
    )
    (preview_dir / "response_generation.json").write_text(
        json.dumps(
            {
                "topic_id": "demo-1",
                "narrative": "Demo narrative",
                "sections": [
                    {
                        "sub_narrative": "Facet 1",
                        "claims": [
                            {
                                "text": "Preview sentence.",
                                "document_ids": ["shard_00001_1"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if benchmark_layout == "comparison":
        (preview_dir / "metrics.json").write_text(
            json.dumps(
                {"official_submission": {"word_count": 2, "sentence_count": 1}}
            ),
            encoding="utf-8",
        )
    (tmp_path / "report.html").write_text(
        "<!doctype html><title>Decision report</title>", encoding="utf-8"
    )

    config = {
        **_config(),
        "app": {
            "default_evidence": str(evidence_path.relative_to(tmp_path)),
            "run_root": "outputs/studio",
            "benchmark_report_dir": str(benchmark_dir.relative_to(tmp_path)),
            "report_path": "report.html",
        },
        "submission": {
            "team_id": "demo-team",
            "run_desc": "Controlled claim-preserving generation.",
        },
    }

    def fake_runner(**kwargs):
        kwargs["progress"](
            {"stage": "generation", "message": "Generated", "completed": 1, "total": 1}
        )
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        official = {
            "metadata": {"narrative_id": "demo-1", "run_id": kwargs["run_id"]},
            "references": ["shard_00001_1"],
            "answer": [{"text": "Generated sentence.", "citations": ["shard_00001_1"]}],
        }
        (output_dir / "rag_output_trec_rag_2026.jsonl").write_text(
            json.dumps(official) + "\n", encoding="utf-8"
        )
        summary = {
            "status": "valid",
            "topic_id": "demo-1",
            "run_id": kwargs["run_id"],
            "submitted_sentence_count": 1,
            "submitted_word_count": 2,
            "reference_count": 1,
            "official_format_valid": True,
        }
        final = {
            "topic_id": "demo-1",
            "narrative": "Demo narrative",
            "sections": [
                {
                    "sub_narrative": "Facet 1",
                    "claims": [
                        {
                            "text": "Generated sentence.",
                            "document_ids": ["shard_00001_1"],
                        }
                    ],
                }
            ],
        }
        return GenerationRunResult(output_dir, summary, official, final)

    app = StudioApplication(
        repo_root=tmp_path,
        config=config,
        runner=fake_runner,
        judge_health_probe=lambda _url: True,
        generator_key_present=lambda _name: True,
    )
    server = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        app_page = requests.get(f"{base}/", timeout=5)
        assert app_page.status_code == 200
        assert "Answer Generation Studio" in app_page.text
        report_page = requests.get(f"{base}/report", timeout=5)
        assert report_page.status_code == 200
        assert "Decision report" in report_page.text
        assert requests.get(f"{base}/api/status", timeout=5).json()["ready"] is True
        assert requests.get(f"{base}/api/evidence", timeout=5).json()["claim_count"] == 4
        assert requests.get(f"{base}/api/benchmark", timeout=5).json()["winner"] == "gpt_5_6_sol"
        assert requests.get(f"{base}/api/preview", timeout=5).json()["source"] == "benchmark"
        response = requests.post(
            f"{base}/api/runs",
            json={"run_id": "rag26-ui-test"},
            timeout=5,
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        deadline = time.time() + 5
        job = {}
        while time.time() < deadline:
            job = requests.get(f"{base}/api/runs/{job_id}", timeout=5).json()
            if job["status"] == "complete":
                break
            time.sleep(0.02)
        assert job["status"] == "complete"
        submission = requests.get(
            f"{base}/api/runs/{job_id}/submission", timeout=5
        )
        assert submission.status_code == 200
        assert submission.headers["content-type"].startswith("application/jsonl")
        assert "Generated sentence" in submission.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

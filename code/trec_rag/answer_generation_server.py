"""Local HTTP server for the competition answer-generation studio."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from trec_rag.answer_generation_studio import (
    GenerationRunResult,
    run_best_answer_generation,
    validate_frozen_evidence_for_generation,
    validate_run_id,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


WEB_ROOT = Path(__file__).with_name("web") / "answer_generation_studio"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
Runner = Callable[..., GenerationRunResult]


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"configured path escapes repository root: {value}") from exc
    return resolved


def build_evidence_summary(value: Mapping[str, object]) -> dict[str, object]:
    frozen = validate_frozen_evidence_for_generation(value)
    return {
        "topic_id": frozen["topic_id"],
        "narrative": frozen["narrative"],
        "source_experiment_id": frozen["source_experiment_id"],
        "facet_count": len(frozen["facets"]),
        "claim_count": frozen["supported_source_claim_count"],
        "source_word_count": frozen["source_claim_word_count"],
        "candidate_word_band": frozen["word_band"],
        "organizer_maximum_words": 1024,
        "nugget_blind": True,
        "facets": [
            {
                "number": facet["facet_number"],
                "label": facet["sub_narrative"],
                "claim_count": facet["sentence_quota"],
                "word_target": facet["word_target"],
            }
            for facet in frozen["facets"]
        ],
    }


def _answer_payload(
    *,
    source: str,
    summary: Mapping[str, object],
    official: Mapping[str, object],
    generation: Mapping[str, object],
    download_url: str | None,
) -> dict[str, object]:
    return {
        "source": source,
        "summary": dict(summary),
        "metadata": dict(official.get("metadata", {})),
        "references": list(official.get("references", [])),
        "sections": [
            {
                "label": section["sub_narrative"],
                "sentences": [
                    {
                        "text": claim["text"],
                        "citations": list(claim["document_ids"]),
                    }
                    for claim in section["claims"]
                ],
            }
            for section in generation["sections"]
        ],
        "official": dict(official),
        "download_url": download_url,
    }


class StudioApplication:
    """Thread-safe application state with a bounded local filesystem surface."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config: Mapping[str, object],
        runner: Runner = run_best_answer_generation,
        judge_health_probe: Callable[[str], bool] | None = None,
        generator_key_present: Callable[[str], bool] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = dict(config)
        self.app_config = dict(config["app"])
        self.runner = runner
        self._judge_health_probe = judge_health_probe or self._default_judge_health_probe
        self._generator_key_present = generator_key_present or (
            lambda name: bool(os.environ.get(name, "").strip())
        )
        self.default_evidence_path = _resolve_inside(
            self.repo_root, str(self.app_config["default_evidence"])
        )
        self.run_root = _resolve_inside(self.repo_root, str(self.app_config["run_root"]))
        self.benchmark_dir = _resolve_inside(
            self.repo_root, str(self.app_config["benchmark_report_dir"])
        )
        self.report_path = _resolve_inside(
            self.repo_root, str(self.app_config["report_path"])
        )
        self.default_evidence = validate_frozen_evidence_for_generation(
            json.loads(self.default_evidence_path.read_text(encoding="utf-8"))
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _default_judge_health_probe(url: str) -> bool:
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            payload = response.json()
            return isinstance(payload.get("data"), list) and bool(payload["data"])
        except (requests.RequestException, ValueError, TypeError):
            return False

    def status(self) -> dict[str, object]:
        generation = self.config["generation"]
        audit = self.config["audit"]
        key_name = str(generation.get("api_key_env", "OPENROUTER_API_KEY"))
        generator_ready = self._generator_key_present(key_name)
        audit_base = os.environ.get(
            str(audit.get("api_base_env", "LITELLM_BASE_URL")),
            str(audit.get("api_base", "http://localhost:4000/v1")),
        ).rstrip("/")
        judge_ready = self._judge_health_probe(f"{audit_base}/models")
        return {
            "ready": generator_ready and judge_ready,
            "generator": {
                "ready": generator_ready,
                "label": "GPT-5.6 Sol",
                "model": generation["model_identity"],
                "route": "OpenRouter",
            },
            "auditor": {
                "ready": judge_ready,
                "label": "Local Qwen support gate",
                "model": audit.get("model_identity", "qwen-local"),
                "route": "LiteLLM / vLLM",
            },
            "contract": {
                "maximum_words": 1024,
                "maximum_citations_per_sentence": 3,
                "nugget_blind": True,
                "primary_semantic_generation_requests": 1,
                "maximum_semantic_repair_requests": 1,
            },
        }

    def evidence(self) -> dict[str, object]:
        return build_evidence_summary(self.default_evidence)

    def benchmark(self) -> dict[str, object]:
        metrics_path = self.benchmark_dir / "comparison_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows = [
                {
                    key: row.get(key)
                    for key in [
                        "model_key",
                        "display_name",
                        "strict_coverage",
                        "partial_credit_coverage",
                        "vital_strict_coverage",
                        "candidate_words",
                        "submitted_words",
                        "submitted_claims",
                        "excluded",
                        "citation_coverage",
                        "unsupported_submitted",
                        "cost",
                    ]
                }
                for row in metrics.get("models", [])
            ]
            winner_key = metrics.get("winner", {}).get("model_key")
            baselines = metrics.get("baselines", {})
            semantic_hash = metrics.get("semantic_request_sha256")
            freeze_hash = metrics.get("aggregate_generation_freeze_sha256")
        else:
            metrics = json.loads(
                (self.benchmark_dir / "metrics.json").read_text(encoding="utf-8")
            )
            model = metrics["model"]
            nuggets = metrics["nuggets"]
            official = metrics["official_submission"]
            answer_claims = metrics["answer_claims"]
            model_key = str(model["key"])
            rows = [
                {
                    "model_key": model_key,
                    "display_name": model["display_name"],
                    "strict_coverage": nuggets["all"]["strict_coverage"],
                    "partial_credit_coverage": nuggets["all"][
                        "partial_credit_coverage"
                    ],
                    "vital_strict_coverage": nuggets["vital"]["strict_coverage"],
                    "candidate_words": official["candidate_word_count"],
                    "submitted_words": official["word_count"],
                    "submitted_claims": official["sentence_count"],
                    "excluded": official["excluded_after_repair"],
                    "citation_coverage": answer_claims["citation_coverage"],
                    "unsupported_submitted": answer_claims[
                        "unsupported_claim_count"
                    ],
                    "cost": metrics["generation"]["cost"],
                }
            ]
            winner_key = model_key
            baselines = {}
            semantic_hash = None
            freeze_hash = metrics.get("generation_freeze_sha256")

        def baseline_summary(name: str) -> dict[str, object] | None:
            value = baselines.get(name)
            if not isinstance(value, Mapping):
                return None
            all_nuggets = value.get("nuggets", {}).get("all", {})
            vital = value.get("nuggets", {}).get("vital", {})
            response = value.get("response", {})
            return {
                "strict_coverage": all_nuggets.get("strict_coverage"),
                "partial_credit_coverage": all_nuggets.get("partial_credit_coverage"),
                "vital_strict_coverage": vital.get("strict_coverage"),
                "word_count": response.get("word_count"),
            }

        return {
            "winner": winner_key,
            "models": rows,
            "baselines": {
                "original_qwen": baseline_summary("original_qwen"),
                "organizer_compliant_qwen": baseline_summary(
                    "organizer_compliant_qwen"
                ),
            },
            "semantic_request_sha256": semantic_hash,
            "aggregate_generation_freeze_sha256": freeze_hash,
            "report_url": "/report",
        }

    def preview(self) -> dict[str, object]:
        comparison = self.benchmark()
        winner_key = str(comparison["winner"])
        model_dir = self.benchmark_dir / winner_key
        if not model_dir.is_dir():
            model_dir = self.benchmark_dir
        official_lines = [
            line
            for line in (model_dir / "rag_output_trec_rag_2026.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        official = json.loads(official_lines[0])
        generation = json.loads(
            (model_dir / "response_generation.json").read_text(encoding="utf-8")
        )
        winner_row = next(
            row
            for row in comparison["models"]
            if row["model_key"] == winner_key
        )
        model_identity = "openai/gpt-5.6-sol"
        run_metrics_path = model_dir / "metrics.json"
        if run_metrics_path.is_file():
            run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
            model_identity = str(
                run_metrics.get("model", {}).get("identity", model_identity)
            )
        summary = {
            "status": "valid",
            "topic_id": generation["topic_id"],
            "run_id": official.get("metadata", {}).get("run_id"),
            "generator": model_identity,
            "candidate_word_count": winner_row.get("candidate_words"),
            "submitted_word_count": winner_row.get("submitted_words"),
            "submitted_sentence_count": winner_row.get("submitted_claims"),
            "excluded_sentence_count": winner_row.get("excluded"),
            "reference_count": len(official.get("references", [])),
            "citation_coverage": winner_row.get("citation_coverage"),
            "official_format_valid": True,
        }
        return _answer_payload(
            source="benchmark",
            summary=summary,
            official=official,
            generation=generation,
            download_url=None,
        )

    def _update_job(self, job_id: str, values: Mapping[str, object]) -> None:
        with self._lock:
            self._jobs[job_id].update(values)
            self._jobs[job_id]["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    def start_run(self, payload: Mapping[str, object]) -> dict[str, object]:
        evidence_value = payload.get("evidence", self.default_evidence)
        if not isinstance(evidence_value, Mapping):
            raise ValueError("evidence must be a frozen-ledger JSON object")
        evidence = validate_frozen_evidence_for_generation(evidence_value)
        suggested_id = datetime.now(timezone.utc).strftime("rag26-%Y%m%d-%H%M%S")
        run_id = validate_run_id(str(payload.get("run_id", suggested_id)))
        team_id = str(payload.get("team_id", self.config["submission"]["team_id"])).strip()
        run_desc = str(payload.get("run_desc", self.config["submission"]["run_desc"])).strip()
        if not team_id or len(team_id) > 80:
            raise ValueError("team ID must contain 1-80 characters")
        if not run_desc or len(run_desc) > 500:
            raise ValueError("run description must contain 1-500 characters")
        output_dir = (self.run_root / run_id).resolve()
        output_dir.relative_to(self.run_root)
        with self._lock:
            if run_id in self._jobs:
                raise ValueError(f"run ID already exists in this session: {run_id}")
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(f"run ID already has output artifacts: {run_id}")
            now = datetime.now(timezone.utc).isoformat()
            self._jobs[run_id] = {
                "job_id": run_id,
                "status": "queued",
                "topic_id": evidence["topic_id"],
                "created_at_utc": now,
                "updated_at_utc": now,
                "progress": {
                    "stage": "queued",
                    "message": "Waiting to start",
                    "completed": 0,
                    "total": 1,
                },
            }

        def worker() -> None:
            self._update_job(run_id, {"status": "running"})

            def progress(event: dict[str, object]) -> None:
                self._update_job(run_id, {"progress": dict(event)})

            try:
                result = self.runner(
                    frozen=evidence,
                    output_dir=output_dir,
                    run_id=run_id,
                    team_id=team_id,
                    run_desc=run_desc,
                    config=self.config,
                    progress=progress,
                )
                answer = _answer_payload(
                    source="generated",
                    summary=result.summary,
                    official=result.official_entry,
                    generation=result.final_generation,
                    download_url=f"/api/runs/{run_id}/submission",
                )
                self._update_job(
                    run_id,
                    {
                        "status": "complete",
                        "progress": {
                            "stage": "complete",
                            "message": "Organizer submission frozen",
                            "completed": 1,
                            "total": 1,
                        },
                        "summary": result.summary,
                        "answer": answer,
                        "_output_dir": result.output_dir,
                    },
                )
            except Exception as exc:  # surfaced to the local operator through job state
                self._update_job(
                    run_id,
                    {
                        "status": "failed",
                        "error": str(exc),
                        "progress": {
                            "stage": "failed",
                            "message": "Generation failed",
                            "completed": 1,
                            "total": 1,
                        },
                    },
                )

        threading.Thread(target=worker, name=f"answer-generation-{run_id}", daemon=True).start()
        return self.job(run_id)

    def job(self, job_id: str) -> dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return {
                key: value
                for key, value in self._jobs[job_id].items()
                if not key.startswith("_")
            }

    def jobs(self) -> list[dict[str, object]]:
        with self._lock:
            ids = list(reversed(self._jobs))
        return [self.job(job_id) for job_id in ids]

    def submission_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("status") != "complete":
                raise KeyError(job_id)
            output_dir = Path(job["_output_dir"])
        path = output_dir / "rag_output_trec_rag_2026.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


class _Handler(BaseHTTPRequestHandler):
    server_version = "TrecRagAnswerStudio/1.0"

    @property
    def app(self) -> StudioApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _security_headers(self, *, inline_styles: bool = False) -> None:
        style_policy = "'self' 'unsafe-inline'" if inline_styles else "'self'"
        self.send_header(
            "Content-Security-Policy",
            f"default-src 'self'; script-src 'self'; style-src {style_policy}; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        disposition: str | None = None,
        inline_styles: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self._security_headers(inline_styles=inline_styles)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value: object, *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body must contain 1 byte to 2 MiB")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/status":
                self._send_json(self.app.status())
            elif path == "/api/evidence":
                self._send_json(self.app.evidence())
            elif path == "/api/benchmark":
                self._send_json(self.app.benchmark())
            elif path == "/api/preview":
                self._send_json(self.app.preview())
            elif path == "/api/runs":
                self._send_json({"runs": self.app.jobs()})
            elif path.startswith("/api/runs/") and path.endswith("/submission"):
                job_id = path.removeprefix("/api/runs/").removesuffix("/submission").strip("/")
                submission = self.app.submission_path(job_id)
                self._send_bytes(
                    submission.read_bytes(),
                    content_type="application/jsonl; charset=utf-8",
                    disposition=f'attachment; filename="{job_id}.jsonl"',
                )
            elif path.startswith("/api/runs/"):
                job_id = path.removeprefix("/api/runs/").strip("/")
                self._send_json(self.app.job(job_id))
            elif path == "/report":
                self._send_bytes(
                    self.app.report_path.read_bytes(),
                    content_type="text/html; charset=utf-8",
                    inline_styles=True,
                )
            elif path in {
                "/comparison_metrics.json",
                "/manifest.yaml",
                "/frozen_evidence_ledger.json",
                "/gpt_5_6_sol/rag_output_trec_rag_2026.jsonl",
            }:
                artifact = self.app.benchmark_dir / path.lstrip("/")
                content_types = {
                    ".json": "application/json; charset=utf-8",
                    ".yaml": "application/yaml; charset=utf-8",
                    ".jsonl": "application/jsonl; charset=utf-8",
                }
                self._send_bytes(
                    artifact.read_bytes(),
                    content_type=content_types.get(artifact.suffix, "application/octet-stream"),
                )
            else:
                static_files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/index.html": ("index.html", "text/html; charset=utf-8"),
                    "/app.css": ("app.css", "text/css; charset=utf-8"),
                    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                    "/mark.svg": ("mark.svg", "image/svg+xml"),
                }
                if path not in static_files:
                    self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
                    return
                filename, content_type = static_files[path]
                self._send_bytes(
                    (WEB_ROOT / filename).read_bytes(), content_type=content_type
                )
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "run not found")
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "artifact not found")
        except (ValueError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/runs":
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            job = self.app.start_run(self._read_json())
            self._send_json(job, status=HTTPStatus.ACCEPTED)
        except (ValueError, TypeError, KeyError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: StudioApplication) -> None:
        super().__init__(address, _Handler)
        self.app = app


def create_server(
    app: StudioApplication, *, host: str = "127.0.0.1", port: int = 8765
) -> StudioHTTPServer:
    return StudioHTTPServer((host, port), app)


def load_application(
    config_path: Path, *, runner: Runner = run_best_answer_generation
) -> StudioApplication:
    config_path = config_path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("studio config must be a mapping")
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    return StudioApplication(repo_root=repo_root, config=config, runner=runner)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag26_answer_generation_studio_gpt56_sol_v1.yaml"),
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    app = load_application(args.config)
    host = args.host or str(app.app_config.get("host", "127.0.0.1"))
    port = args.port or int(app.app_config.get("port", 8765))
    server = create_server(app, host=host, port=port)
    print(f"Answer Generation Studio: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

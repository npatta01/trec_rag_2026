"""Command-line entry point for generating auditable sparse query plans."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from trec_rag.query_planner import (
    ANALYZER_VERSION,
    PROMPT_VERSION,
    RENDERER_VERSION,
    SCHEMA_VERSION,
    QueryPlanGenerationError,
    QueryPlanGenerator,
    QueryPlanValidationError,
    render_query_plan,
)
from trec_rag.repo_env import find_repo_root, load_repo_env, repo_cache_root
from trec_rag.topics import Topic, load_topics


DEFAULT_TOPIC_IDS = ("144", "213", "224", "407", "515")
ATTEMPT_KINDS = (
    "first_emission",
    "replacement_after_harness_incident",
    "targeted_revision",
)
OUTCOME_STATUSES = frozenset(
    {
        "success",
        "invalid_json",
        "plan_validation_error",
        "render_validation_error",
        "http_error",
        "timeout",
        "unexpected_exception",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    _atomic_write_text(path, ("\n".join(lines) + "\n") if lines else "")


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not safe:
        raise ValueError("run and topic identifiers must contain a safe character")
    return safe


def _outcome_path(outcome_dir: Path, ordinal: int, topic: Topic) -> Path:
    return outcome_dir / f"{ordinal:02d}_{_safe_component(topic.id)}.json"


def _raw_response_path(outcome_dir: Path, ordinal: int, topic: Topic) -> Path:
    return outcome_dir / "raw_responses" / f"{ordinal:02d}_{_safe_component(topic.id)}.json"


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _frozen_request_config(
    generator: QueryPlanGenerator, topic: Topic
) -> dict[str, object]:
    request = generator.request_payload(topic)
    response_format = request.get("response_format")
    schema: object = None
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, Mapping):
            schema = json_schema.get("schema")
    messages = request.get("messages")
    prompt = None
    if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
        prompt = messages[0].get("content")
    return {
        "provider": "openai_compatible_chat_completions",
        "base_url": generator.base_url,
        "model": generator.model,
        "model_revision": generator.model_revision,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "renderer_version": RENDERER_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "schema_sha256": _sha256_json(schema),
        "prompt_sha256": hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
        "request_sha256": _sha256_json(request),
        "reasoning_effort": generator.reasoning_effort,
        "max_tokens": generator.max_tokens,
        "temperature": generator.temperature,
        "seed": generator.seed,
        "max_facets": generator.max_facets,
    }


def _exception_chain(exc: BaseException) -> list[BaseException]:
    result: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        current = current.__cause__ or current.__context__
    return result


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, QueryPlanGenerationError):
        return exc.status
    if isinstance(exc, QueryPlanValidationError):
        return "plan_validation_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    chain = _exception_chain(exc)
    if any(isinstance(item, TimeoutError) for item in chain):
        return "timeout"
    if any(isinstance(item, (urllib.error.HTTPError, urllib.error.URLError)) for item in chain):
        return "http_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    return "unexpected_exception"


def _failure_kind(status: str) -> str:
    if status in {"timeout", "http_error"}:
        return "transport"
    if status in {
        "invalid_json",
        "plan_validation_error",
        "render_validation_error",
    }:
        return "plan_validation"
    return "unexpected"


def _response_summary(response: Mapping[str, object] | None) -> dict[str, object]:
    if response is None:
        return {}
    usage = response.get("usage")
    return {
        "http_status": getattr(response, "http_status", None),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def _parse_topic_ids(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("topic IDs must not be empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("topic IDs must be unique")
    return result


def _select_topics(topics: list[Topic], topic_ids: tuple[str, ...]) -> list[Topic]:
    by_id = {topic.id: topic for topic in topics}
    missing = [topic_id for topic_id in topic_ids if topic_id not in by_id]
    if missing:
        raise ValueError("unknown topic ID(s): " + ", ".join(missing))
    return [by_id[topic_id] for topic_id in topic_ids]


def _topics_from_run_metadata(metadata: Mapping[str, object]) -> list[Topic]:
    raw_topics = metadata.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("run metadata lacks its ordered topic records")
    topics: list[Topic] = []
    for index, raw_topic in enumerate(raw_topics):
        if not isinstance(raw_topic, Mapping):
            raise ValueError(f"run metadata topic {index} is not an object")
        try:
            topics.append(
                Topic(
                    id=str(raw_topic["id"]),
                    title=str(raw_topic["title"]),
                    narrative=str(raw_topic["narrative"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"run metadata topic {index} lacks {exc.args[0]}"
            ) from exc
    expected_order = [topic.id for topic in topics]
    if metadata.get("request_order") != expected_order:
        raise ValueError("run metadata topics do not match request_order")
    return topics


def _rebuild_summary(
    *,
    args: argparse.Namespace,
    topics: list[Topic],
    outcome_dir: Path,
    run_metadata: Mapping[str, object],
) -> dict[str, object]:
    run_id = str(run_metadata["run_id"])
    attempt_kind = str(run_metadata["attempt_kind"])
    run_started_at = str(run_metadata["run_started_at"])
    frozen_settings_raw = run_metadata.get("frozen_model_settings")
    if not isinstance(frozen_settings_raw, Mapping):
        raise ValueError("run metadata lacks frozen_model_settings")
    frozen_settings = dict(frozen_settings_raw)
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing_topics: list[str] = []
    statuses: dict[str, int] = {}
    for ordinal, topic in enumerate(topics, start=1):
        path = _outcome_path(outcome_dir, ordinal, topic)
        if not path.exists():
            missing_topics.append(topic.id)
            continue
        outcome = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(outcome, dict):
            raise ValueError(f"outcome is not a JSON object: {path}")
        expected_identity = {
            "run_id": run_id,
            "attempt_kind": attempt_kind,
            "request_ordinal": ordinal,
            "topic_id": topic.id,
        }
        mismatches = [
            key
            for key, expected in expected_identity.items()
            if outcome.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                f"outcome identity mismatch for {path}: {', '.join(mismatches)}"
            )
        status = str(outcome.get("status") or "unexpected_exception")
        if status not in OUTCOME_STATUSES:
            raise ValueError(f"unsupported outcome status in {path}: {status}")
        statuses[status] = statuses.get(status, 0) + 1
        if status == "success":
            record = outcome.get("record")
            if not isinstance(record, dict):
                raise ValueError(f"successful outcome lacks a record: {path}")
            record_topic = record.get("topic")
            nested_identity = {
                "run_id": record.get("run_id"),
                "attempt_kind": record.get("attempt_kind"),
                "request_ordinal": record.get("request_ordinal"),
                "topic_id": (
                    record_topic.get("id")
                    if isinstance(record_topic, Mapping)
                    else None
                ),
            }
            nested_mismatches = [
                key
                for key, expected in expected_identity.items()
                if nested_identity.get(key) != expected
            ]
            if nested_mismatches:
                raise ValueError(
                    f"record identity mismatch for {path}: "
                    + ", ".join(nested_mismatches)
                )
            records.append(record)
        else:
            failure = outcome.get("failure")
            if not isinstance(failure, dict):
                raise ValueError(f"failed outcome lacks a failure record: {path}")
            nested_mismatches = [
                key
                for key, expected in expected_identity.items()
                if failure.get(key) != expected
            ]
            if nested_mismatches:
                raise ValueError(
                    f"failure identity mismatch for {path}: "
                    + ", ".join(nested_mismatches)
                )
            failures.append(failure)

    failure_path = args.failure_output or args.output.with_name(
        f"{args.output.stem}.failures.jsonl"
    )
    manifest_path = args.output.with_name(f"{args.output.stem}.manifest.json")
    _atomic_write_jsonl(args.output, records)
    _atomic_write_jsonl(failure_path, failures)
    manifest: dict[str, object] = {
        "run_id": run_id,
        "attempt_kind": attempt_kind,
        "run_started_at": run_started_at,
        "summary_written_at": _utc_now(),
        "schema_version": frozen_settings.get("schema_version"),
        "prompt_version": frozen_settings.get("prompt_version"),
        "renderer_version": frozen_settings.get("renderer_version"),
        "analyzer_version": frozen_settings.get("analyzer_version"),
        "topics": [topic.id for topic in topics],
        "request_order": [topic.id for topic in topics],
        "successful": len(records),
        "failed": len(failures),
        "missing": len(missing_topics),
        "missing_topics": missing_topics,
        "status_counts": statuses,
        "model": frozen_settings.get("model"),
        "model_revision": frozen_settings.get("model_revision"),
        "base_url": frozen_settings.get("base_url"),
        "reasoning_effort": frozen_settings.get("reasoning_effort"),
        "max_tokens": frozen_settings.get("max_tokens"),
        "temperature": frozen_settings.get("temperature"),
        "seed": frozen_settings.get("seed"),
        "max_facets": frozen_settings.get("max_facets"),
        "workers": run_metadata.get("concurrency"),
        "timeout_seconds": run_metadata.get("timeout_seconds"),
        "cache_dir": run_metadata.get("cache_dir"),
        "outcome_dir": str(outcome_dir),
        "output": str(args.output),
        "failure_output": str(failure_path),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def generate_plans(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(Path.cwd())
    load_repo_env(repo_root)
    rebuild_only = bool(getattr(args, "rebuild_only", False))
    if rebuild_only:
        requested_outcome_dir = getattr(args, "outcome_dir", None)
        requested_run_id = getattr(args, "run_id", None)
        if requested_outcome_dir is None and requested_run_id is None:
            raise ValueError("--rebuild-only requires --outcome-dir or --run-id")
        outcome_dir = requested_outcome_dir or (
            args.output.parent / "outcomes" / _safe_component(requested_run_id)
        )
        run_metadata_path = outcome_dir / "_run.json"
        if not run_metadata_path.exists():
            raise FileNotFoundError(f"run metadata does not exist: {run_metadata_path}")
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"run metadata is not a JSON object: {run_metadata_path}")
        topics = _topics_from_run_metadata(metadata)
        manifest = _rebuild_summary(
            args=args,
            topics=topics,
            outcome_dir=outcome_dir,
            run_metadata=metadata,
        )
        return 1 if manifest["failed"] or manifest["missing"] else 0

    topics = _select_topics(
        load_topics(args.topics, topic_format=args.topic_format),
        args.topic_ids,
    )
    cache_dir = args.cache_dir or repo_cache_root(repo_root) / "query_plans"
    generator = QueryPlanGenerator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        model_revision=args.model_revision,
        cache_dir=cache_dir,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        max_facets=args.max_facets,
    )
    run_id = getattr(args, "run_id", None) or (
        "query_plan_" + _safe_component(_utc_now())
    )
    attempt_kind = getattr(args, "attempt_kind", "first_emission")
    if attempt_kind not in ATTEMPT_KINDS:
        raise ValueError(f"unsupported attempt kind: {attempt_kind}")
    requested_outcome_dir = getattr(args, "outcome_dir", None)
    outcome_dir = requested_outcome_dir or (
        args.output.parent / "outcomes" / _safe_component(run_id)
    )
    run_started_at = _utc_now()
    run_metadata_path = outcome_dir / "_run.json"

    collision_paths = [
        run_metadata_path,
        args.output,
        args.failure_output
        or args.output.with_name(f"{args.output.stem}.failures.jsonl"),
        args.output.with_name(f"{args.output.stem}.manifest.json"),
    ]
    collision_paths.extend(
        _outcome_path(outcome_dir, ordinal, topic)
        for ordinal, topic in enumerate(topics, start=1)
    )
    collision_paths.extend(
        _raw_response_path(outcome_dir, ordinal, topic)
        for ordinal, topic in enumerate(topics, start=1)
    )
    existing = [str(path) for path in collision_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing run artifact: " + ", ".join(existing)
        )

    run_metadata: dict[str, object] = {
        "run_id": run_id,
        "attempt_kind": attempt_kind,
        "run_started_at": run_started_at,
        "topics": [asdict(topic) for topic in topics],
        "request_order": [topic.id for topic in topics],
        "concurrency": args.workers,
        "timeout_seconds": args.timeout,
        "cache_enabled": not args.no_cache,
        "cache_dir": str(cache_dir),
        "output": str(args.output),
        "failure_output": str(
            args.failure_output
            or args.output.with_name(f"{args.output.stem}.failures.jsonl")
        ),
        "outcome_dir": str(outcome_dir),
        "frozen_model_settings": {
            "base_url": generator.base_url,
            "model": generator.model,
            "model_revision": generator.model_revision,
            "reasoning_effort": generator.reasoning_effort,
            "max_tokens": generator.max_tokens,
            "temperature": generator.temperature,
            "seed": generator.seed,
            "max_facets": generator.max_facets,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "renderer_version": RENDERER_VERSION,
            "analyzer_version": ANALYZER_VERSION,
        },
    }
    _atomic_write_json(run_metadata_path, run_metadata)

    def persist_fallback(ordinal: int, topic: Topic, exc: Exception) -> None:
        path = _outcome_path(outcome_dir, ordinal, topic)
        if path.exists():
            return
        now = _utc_now()
        status = _classify_exception(exc)
        failure = {
            "run_id": run_id,
            "attempt_kind": attempt_kind,
            "request_ordinal": ordinal,
            "topic_id": topic.id,
            "topic": asdict(topic),
            "status": status,
            "failure_kind": _failure_kind(status),
            "request_started_at": now,
            "finished_at": now,
            "elapsed_seconds": 0.0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_write_json(
            path,
            {
                "run_id": run_id,
                "attempt_kind": attempt_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "status": status,
                "failure": failure,
            },
        )

    def generate_one(ordinal: int, topic: Topic) -> None:
        started_at = _utc_now()
        started = time.perf_counter()
        outcome_path = _outcome_path(outcome_dir, ordinal, topic)
        raw_path = _raw_response_path(outcome_dir, ordinal, topic)
        frozen_config = _frozen_request_config(generator, topic)
        raw_response_written = False
        response_meta: dict[str, object] = {}

        def preserve_response(response: Mapping[str, object], elapsed: float) -> None:
            nonlocal raw_response_written, response_meta
            response_meta = _response_summary(response)
            raw_body = getattr(response, "raw_body", None)
            response_headers = getattr(response, "response_headers", None)
            raw_record: dict[str, object] = {
                "run_id": run_id,
                "attempt_kind": attempt_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "request_started_at": started_at,
                "response_received_at": _utc_now(),
                "request_elapsed_seconds": elapsed,
                "capture_level": (
                    "exact_http_body_and_parsed_json"
                    if isinstance(raw_body, bytes)
                    else "parsed_json_envelope"
                ),
                "http_status": getattr(response, "http_status", None),
                "response_headers": (
                    dict(response_headers)
                    if isinstance(response_headers, Mapping)
                    else {}
                ),
                "response_headers_capture_level": "normalized_mapping_duplicates_collapsed",
                "raw_body_utf8": (
                    raw_body.decode("utf-8", errors="replace")
                    if isinstance(raw_body, bytes)
                    else None
                ),
                "raw_body_base64": (
                    base64.b64encode(raw_body).decode("ascii")
                    if isinstance(raw_body, bytes)
                    else None
                ),
                "raw_body_sha256": (
                    hashlib.sha256(raw_body).hexdigest()
                    if isinstance(raw_body, bytes)
                    else None
                ),
                "frozen_request_config": frozen_config,
                "response": dict(response),
            }
            _atomic_write_json(raw_path, raw_record)
            raw_response_written = True

        try:
            result = generator.generate(
                topic,
                cache=not args.no_cache,
                response_hook=preserve_response,
            )
            rendered = render_query_plan(topic, result.plan)
            finished_at = _utc_now()
            elapsed_seconds = time.perf_counter() - started
            record: dict[str, object] = {
                "run_id": run_id,
                "attempt_kind": attempt_kind,
                "request_ordinal": ordinal,
                "topic": asdict(topic),
                "plan": result.plan.to_dict(),
                "rendered_queries": [asdict(row) for row in rendered],
                "provenance": asdict(result.provenance),
                "cache_hit": result.cache_hit,
                "cache_path": str(result.cache_path) if result.cache_path else None,
                "outcome_path": str(outcome_path.resolve()),
                "raw_response_path": (
                    str(raw_path.resolve()) if raw_response_written else None
                ),
                "renderer_version": RENDERER_VERSION,
                "analyzer_version": ANALYZER_VERSION,
            }
            outcome: dict[str, object] = {
                "run_id": run_id,
                "attempt_kind": attempt_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "status": "success",
                "request_started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed_seconds,
                "concurrency": args.workers,
                "timeout_seconds": args.timeout,
                "frozen_request_config": frozen_config,
                "raw_response_path": (
                    str(raw_path.resolve()) if raw_response_written else None
                ),
                "cache_path": str(result.cache_path) if result.cache_path else None,
                "response": response_meta,
                "record": record,
            }
            plan = record["plan"]
            provenance = record["provenance"]
            assert isinstance(plan, dict) and isinstance(provenance, dict)
            success_message = (
                f"{topic.id}: {plan['facet_count']} facets, "
                f"{len(plan['global_expansion']['terms'])} global terms, "
                f"{float(provenance['elapsed_seconds']):.2f}s, "
                f"cache_hit={record['cache_hit']}"
            )
            _atomic_write_json(outcome_path, outcome)
            try:
                print(success_message)
            except Exception:
                pass
        except Exception as exc:
            if outcome_path.exists():
                return
            finished_at = _utc_now()
            elapsed_seconds = time.perf_counter() - started
            status = _classify_exception(exc)
            response = exc.response if isinstance(exc, QueryPlanGenerationError) else None
            if response and not response_meta:
                response_meta = _response_summary(response)
            failure: dict[str, object] = {
                "run_id": run_id,
                "attempt_kind": attempt_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "topic": asdict(topic),
                "status": status,
                "failure_kind": _failure_kind(status),
                "request_started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed_seconds,
                "concurrency": args.workers,
                "timeout_seconds": args.timeout,
                "frozen_request_config": frozen_config,
                "raw_response_path": (
                    str(raw_path.resolve()) if raw_response_written else None
                ),
                "cache_path": None,
                "response_metadata": response_meta,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, QueryPlanGenerationError):
                if exc.response:
                    failure["response"] = exc.response
                if exc.content is not None:
                    failure["content"] = exc.content
            _atomic_write_json(
                outcome_path,
                {
                    "run_id": run_id,
                    "attempt_kind": attempt_kind,
                    "request_ordinal": ordinal,
                    "topic_id": topic.id,
                    "status": status,
                    "request_started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": elapsed_seconds,
                    "concurrency": args.workers,
                    "timeout_seconds": args.timeout,
                    "frozen_request_config": frozen_config,
                    "raw_response_path": (
                        str(raw_path.resolve()) if raw_response_written else None
                    ),
                    "response": response_meta,
                    "failure": failure,
                },
            )
            try:
                print(f"{topic.id}: FAILED [{status}]: {exc}")
            except Exception:
                pass

    try:
        if args.workers == 1:
            for ordinal, topic in enumerate(topics, start=1):
                try:
                    generate_one(ordinal, topic)
                except Exception as exc:
                    persist_fallback(ordinal, topic, exc)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures = {
                    executor.submit(generate_one, ordinal, topic): (ordinal, topic)
                    for ordinal, topic in enumerate(topics, start=1)
                }
                for future in concurrent.futures.as_completed(futures):
                    ordinal, topic = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        persist_fallback(ordinal, topic, exc)
    finally:
        manifest = _rebuild_summary(
            args=args,
            topics=topics,
            outcome_dir=outcome_dir,
            run_metadata=run_metadata,
        )

    print(f"Wrote {manifest['successful']} plan(s) to {args.output}")
    if manifest["failed"]:
        print(
            f"Wrote {manifest['failed']} failure record(s) to "
            f"{manifest['failure_output']}"
        )
    if manifest["missing"]:
        print(f"Missing {manifest['missing']} per-topic outcome(s)")
    return 1 if manifest["failed"] or manifest["missing"] else 0


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = find_repo_root(Path.cwd())
    parser = argparse.ArgumentParser(
        description="Generate schema-constrained query plans without running retrieval."
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=(
            repo_root
            / "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
        ),
    )
    parser.add_argument("--topic-format", choices=("tsv", "jsonl"), default="tsv")
    parser.add_argument(
        "--topic-ids",
        type=_parse_topic_ids,
        default=DEFAULT_TOPIC_IDS,
        help="Comma-separated topic IDs in output order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs/query_planner_gpt_oss_smoke_v1/plans.jsonl",
    )
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument(
        "--run-id",
        help="Stable identifier used for the immutable per-topic outcome directory.",
    )
    parser.add_argument(
        "--attempt-kind",
        choices=ATTEMPT_KINDS,
        default="first_emission",
    )
    parser.add_argument("--outcome-dir", type=Path)
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Rebuild JSONL summaries from an existing per-topic outcome directory.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gpt-oss-local")
    parser.add_argument("--model-revision")
    parser.add_argument("--api-key")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-facets", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    return generate_plans(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

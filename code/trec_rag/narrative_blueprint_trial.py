"""Thin, throwaway one-topic experiment for narrative-blueprint generation.

Question: does a compact narrative blueprint improve answer coverage for one
fixed-evidence topic under the 1,024-word cap without degrading citations?

This prototype intentionally has no resume, overwrite, orchestration, or test
surface. It exists to make one planner call and at most two writer calls for a
single topic using the existing sealed handoff and generation helpers.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from trec_rag.competition_rag import (
    SEMANTIC_RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
    SemanticCompletionError,
    OpenRouterJsonGenerator,
    _atomic_write_text,
    _redact,
    _safe_topic_name,
    _validate_artifact_paths,
    _validate_exact_hint_citations,
    _validate_generated_submission_record,
    build_submission_record,
    load_rag_generation_config,
    normalize_generated_record,
    output_schema,
    trim_to_word_limit,
)
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
    select_generation_topics,
)
from trec_rag.narrative_blueprint import (
    BLUEPRINT_CONTRACT_VERSION,
    load_blueprint_state,
    planner_response_schema,
    project_blueprint,
    render_blueprint_writer_context,
    render_planner_prompt,
    serialize_blueprint_state,
    validate_blueprint,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


PLANNER_SYSTEM_PROMPT = """You are a narrative blueprint planner. Plan coverage of the complete
official narrative from the narrative and advisory group/claim hints supplied by the user.
Generated groups are retrieval structure, not separate answer requirements. Claim hints are
advisory. Do not invent evidence, document identifiers, or claim identifiers; use only the
provided local aliases. Return only the requested JSON object."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _digest_json(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _write_json(path: Path, value: object, *, api_key: str) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _redact(value, (api_key,)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _topic_for_cli(config: Any, handoff: GenerationHandoff, topic_id: str) -> GenerationTopic:
    if config.topic_ids is not None and config.topic_ids != (topic_id,):
        raise ValueError(
            "prototype config must select exactly the CLI topic and no other topic"
        )
    topics = select_generation_topics(handoff, (topic_id,))
    if len(topics) != 1:
        raise ValueError("prototype requires exactly one topic")
    return topics[0]


def _print_dry_run(
    config: Any, topic: GenerationTopic, *, writer_attempts: int
) -> None:
    planner_prompt = render_planner_prompt(topic)
    evidence_chars = sum(len(row.text) for row in topic.evidence)
    print(f"topic={topic.topic_id}")
    print(
        "counts="
        f"groups:{len(topic.groups)},claims:{len(topic.claim_hints)},"
        f"evidence_passages:{len(topic.evidence)},citation_docids:{len(topic.citation_docids)}"
    )
    print(
        "sizes="
        f"narrative_words:{len(topic.narrative.split())},"
        f"planner_prompt_chars:{len(planner_prompt)},evidence_chars:{evidence_chars}"
    )
    print(
        f"budget=planner:1,writer:{writer_attempts},"
        f"total_semantic_calls:{1 + writer_attempts}"
    )
    del config


def _private_root(config: Any, topic: GenerationTopic) -> Path:
    return config.resolved_work_dir / "blueprint_prototype" / _safe_topic_name(topic.topic_id)


def _refuse_existing_state(config: Any) -> None:
    if config.resume or config.overwrite:
        raise ValueError("prototype refuses resume/overwrite; use a new experiment config")
    if config.output_path.exists():
        raise ValueError(
            "prototype output already exists; use a new experiment config (no overwrite)"
        )
    if config.resolved_work_dir.exists():
        if not config.resolved_work_dir.is_dir() or any(config.resolved_work_dir.iterdir()):
            raise ValueError(
                "prototype work state already exists; use a new experiment config (no resume)"
            )


async def _complete(
    generator: OpenRouterJsonGenerator,
    *,
    topic_id: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    executor: ThreadPoolExecutor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = executor.submit(
        partial(
            generator.complete_json,
            topic_id=topic_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
        )
    )
    while not completion.done():
        await asyncio.sleep(0.01)
    return completion.result()


async def _run_live(
    config: Any,
    topic: GenerationTopic,
    generator: OpenRouterJsonGenerator,
    *,
    api_key: str,
    writer_attempts: int,
) -> dict[str, Any]:
    _validate_artifact_paths(config)
    _refuse_existing_state(config)
    private_root = _private_root(config, topic)
    private_root.mkdir(parents=True, exist_ok=False)
    planner_prompt = render_planner_prompt(topic)
    planner_prompt_sha256 = _digest_text(planner_prompt)
    planner_schema = planner_response_schema()
    planner_raw: dict[str, Any] | None = None
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blueprint-prototype") as executor:
            planner_payload, planner_raw = await _complete(
                generator,
                topic_id=topic.topic_id,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=planner_prompt,
                response_schema=planner_schema,
                executor=executor,
            )
        _write_json(
            private_root / "planner.receipt.json",
            {
                "stage": "planner",
                "contract_version": BLUEPRINT_CONTRACT_VERSION,
                "prompt_sha256": planner_prompt_sha256,
                "schema_sha256": _digest_json(planner_schema),
                "raw_response": planner_raw,
            },
            api_key=api_key,
        )
        blueprint = validate_blueprint(topic, planner_payload)
        projection = project_blueprint(topic, blueprint)
        writer_context = render_blueprint_writer_context(topic, blueprint, projection)
        writer_context_sha256 = _digest_text(writer_context)
        state = serialize_blueprint_state(
            topic,
            blueprint,
            projection,
            planner_prompt_sha256=planner_prompt_sha256,
            writer_context_sha256=writer_context_sha256,
        )
        _write_json(private_root / "blueprint.state.json", state, api_key=api_key)
        # Load the exact bytes we just wrote so the prototype exercises the authenticated seam.
        load_blueprint_state(
            topic,
            json.loads((private_root / "blueprint.state.json").read_text(encoding="utf-8")),
            planner_prompt_sha256=planner_prompt_sha256,
        )
    except SemanticCompletionError as exc:
        _write_json(
            private_root / "planner.receipt.json",
            {
                "stage": "planner",
                "prompt_sha256": planner_prompt_sha256,
                "schema_sha256": _digest_json(planner_schema),
                "error": f"{type(exc).__name__}: {exc}",
                "raw_response": exc.raw_response,
            },
            api_key=api_key,
        )
        raise
    except Exception as exc:
        if planner_raw is None:
            _write_json(
                private_root / "planner.receipt.json",
                {
                    "stage": "planner",
                    "prompt_sha256": planner_prompt_sha256,
                    "schema_sha256": _digest_json(planner_schema),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                api_key=api_key,
            )
        raise

    last_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blueprint-prototype") as executor:
        for attempt in range(1, writer_attempts + 1):
            writer_raw: dict[str, Any] | None = None
            try:
                user_prompt = writer_context
                if attempt == 2:
                    user_prompt += SEMANTIC_RETRY_INSTRUCTION
                generated, writer_raw = await _complete(
                    generator,
                    topic_id=topic.topic_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_schema=output_schema(),
                    executor=executor,
                )
                _write_json(
                    private_root / f"writer.attempt-{attempt}.receipt.json",
                    {
                        "stage": "writer",
                        "attempt": attempt,
                        "prompt_sha256": _digest_text(user_prompt),
                        "schema_sha256": _digest_json(output_schema()),
                        "raw_response": writer_raw,
                    },
                    api_key=api_key,
                )
                record = normalize_generated_record(
                    trim_to_word_limit(
                        build_submission_record(
                            generated,
                            topic_id=topic.topic_id,
                            narrative=topic.narrative,
                            team_id=config.team_id,
                            run_id=config.run_id,
                            run_desc=config.run_desc,
                        )
                    ),
                    allowed_docids=list(projection.citation_docids),
                )
                _validate_generated_submission_record(
                    record,
                    topic_id=topic.topic_id,
                    narrative=topic.narrative,
                    allowed_docids=list(projection.citation_docids),
                    team_id=config.team_id,
                    run_id=config.run_id,
                    run_desc=config.run_desc,
                )
                _validate_exact_hint_citations(record, topic=topic)
                _atomic_write_text(
                    config.output_path,
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
                )
                return record
            except SemanticCompletionError as exc:
                last_error = exc
                _write_json(
                    private_root / f"writer.attempt-{attempt}.receipt.json",
                    {
                        "stage": "writer",
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": exc.raw_response,
                    },
                    api_key=api_key,
                )
                if not exc.retryable:
                    break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                if writer_raw is None:
                    _write_json(
                        private_root / f"writer.attempt-{attempt}.receipt.json",
                        {
                            "stage": "writer",
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        api_key=api_key,
                    )
            except Exception as exc:
                last_error = exc
                _write_json(
                    private_root / f"writer.attempt-{attempt}.receipt.json",
                    {
                        "stage": "writer",
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    api_key=api_key,
                )
                break
    if last_error is None:
        raise RuntimeError("writer semantic attempt budget must be positive")
    raise RuntimeError(
        f"writer failed after at most {writer_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Prototype RAG YAML config.")
    parser.add_argument("--topic", required=True, help="Exactly one handoff topic ID.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print topic/count/size/budget statistics without provider calls.",
    )
    parser.add_argument(
        "--writer-attempts",
        type=int,
        choices=(1, 2),
        default=2,
        help="Maximum writer semantic attempts for this isolated invocation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    config = load_rag_generation_config(args.config)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topic = _topic_for_cli(config, handoff, args.topic)
    if args.dry_run:
        _print_dry_run(config, topic, writer_attempts=args.writer_attempts)
        return

    repo_root = find_repo_root(args.config.resolve().parent)
    load_repo_env(repo_root)
    api_key = os.environ.get(config.api_key_env, "")
    generator = OpenRouterJsonGenerator(
        api_base=config.api_base,
        api_key=api_key,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        structured_output=config.structured_output,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        transport_max_attempts=config.transport_max_attempts,
    )
    asyncio.run(
        _run_live(
            config,
            topic,
            generator,
            api_key=api_key,
            writer_attempts=args.writer_attempts,
        )
    )
    print(f"completed topic={topic.topic_id} output={config.output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc

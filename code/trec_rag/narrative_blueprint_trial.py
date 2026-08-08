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
    BlueprintProjection,
    NarrativeBlueprint,
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


NO_PLANNER_CONTROL_INSTRUCTION = """

No-planner control: treat every distinct request in the official narrative as a checklist. Aim
for 850–950 words only when the selected evidence supports that coverage; never pad or repeat.
Cover the core answer plus distinct high-specificity facts, including quantities, named
mechanisms, actors, interventions, contrasts, and caveats. Preserve every supported member of
an evidence-backed list rather than replacing it with a generic summary. Keep each answer
object self-contained and cite only the strongest supporting selected docids."""


NO_PLANNER_OUTPUT_CONTRACT = """

Output contract: return exactly one JSON object with exactly `references` and `answer`; do not
return Markdown. `references` must contain only raw ClimbMix docid strings from the evidence
catalog, and only when an answer object cites them. Each `answer` item must contain one
self-contained sentence stating one claim and one to three unique raw docid strings in
`citations`, ordered strongest support first. Never use numeric citation indexes, never cite an
unsupported passage, and keep the complete answer at or below 1,024 whitespace-separated words."""


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


def _render_no_planner_control_prompt(topic: GenerationTopic) -> str:
    """Render one deterministic, deduplicated full-evidence control prompt."""

    group_aliases = {
        group.group_id: f"g{index:03d}"
        for index, group in enumerate(topic.groups, start=1)
    }
    claim_aliases = {
        claim.claim_id: f"c{index:03d}"
        for index, claim in enumerate(topic.claim_hints, start=1)
    }
    lines = [
        "NO-PLANNER FULL-EVIDENCE CONTROL",
        "",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        "RETRIEVAL GROUPS AND ADVISORY CLAIM HINTS (DISPLAY ALIASES ONLY):",
    ]
    for group in topic.groups:
        lines.append(f"[{group_aliases[group.group_id]}] GROUP: {group.text}")
        for claim in topic.claim_hints:
            if claim.group_id == group.group_id:
                lines.append(f"  [{claim_aliases[claim.claim_id]}] CLAIM HINT: {claim.text}")
    lines.extend(["", "SELECTED EVIDENCE CATALOG (EVERY PASSAGE IS AUTHORITATIVE):"])
    for index, evidence in enumerate(topic.evidence, start=1):
        lines.extend(
            [
                f"[e{index:03d}] DOCID: {evidence.docid}",
                evidence.text,
            ]
        )
    lines.extend([NO_PLANNER_CONTROL_INSTRUCTION, NO_PLANNER_OUTPUT_CONTRACT])
    return "\n".join(lines)


def _render_hybrid_writer_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
) -> str:
    """Render one hybrid writer prompt from an authenticated planner state."""

    claim_aliases = {
        claim.claim_id: f"c{index:03d}"
        for index, claim in enumerate(topic.claim_hints, start=1)
    }
    evidence_alias_by_id = {
        evidence.evidence_id: f"e{index:03d}"
        for index, evidence in enumerate(topic.evidence, start=1)
    }
    selected_claim_ids = {
        claim_id
        for obligation in projection.obligations
        for claim_id in obligation.claim_ids
    }
    lines = [
        "NARRATIVE BLUEPRINT HYBRID WRITER",
        "Write a grounded answer to the complete official narrative using the authenticated",
        "blueprint as the core coverage checklist and every handoff passage as factual authority.",
        "Selected claim hints are advisory. Facts must be verified against the evidence catalog.",
        "Unselected claim hints are optional specificity candidates, never requirements or authority.",
        "Obligation-selected evidence aliases are focus cues only; do not omit other handoff passages.",
        "Preserve supported named lists, numbers, quantities, mechanisms, actors, interventions,",
        "contrasts, and caveats instead of replacing them with generic summaries.",
        "Use the full citation domain below; the planner cannot prune it. Cite only exact docids",
        "supported by the evidence catalog.",
        "",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        "BLUEPRINT OBLIGATIONS (CORE):",
    ]
    for index, (blueprint_obligation, projection_obligation) in enumerate(
        zip(blueprint.obligations, projection.obligations), start=1
    ):
        selected_aliases = ", ".join(
            claim_aliases[claim_id] for claim_id in projection_obligation.claim_ids
        )
        evidence_aliases = ", ".join(
            evidence_alias_by_id[evidence_id]
            for evidence_id in projection_obligation.evidence_ids
        )
        lines.extend(
            [
                f"OBLIGATION {index} [{projection_obligation.priority}; "
                f"{projection_obligation.answer_mode}; ~{projection_obligation.target_words} words] "
                f"{blueprint_obligation.label}",
                "  Narrative anchors: "
                + " | ".join(blueprint_obligation.narrative_spans),
                "  Selected claim aliases (core): " + selected_aliases,
                "  Selected evidence aliases: " + evidence_aliases,
            ]
        )
    lines.extend(["", "ADVISORY CLAIM HINTS (EVERY HINT EXACTLY ONCE):"])
    for claim in topic.claim_hints:
        alias = claim_aliases[claim.claim_id]
        status = (
            "SELECTED CORE CLAIM"
            if claim.claim_id in selected_claim_ids
            else "UNSELECTED OPTIONAL SPECIFICITY CANDIDATE"
        )
        lines.append(f"[{alias}] {status}: {claim.text}")
    lines.extend(
        [
            "",
            "FULL EVIDENCE CATALOG (EVERY PASSAGE EXACTLY ONCE; AUTHORITATIVE):",
        ]
    )
    for evidence in topic.evidence:
        lines.extend(
            [
                f"[{evidence_alias_by_id[evidence.evidence_id]}] DOCID: {evidence.docid}",
                evidence.text,
            ]
        )
    lines.extend(
        [
            "",
            "FULL CITATION DOMAIN (PLANNER CANNOT PRUNE):",
            ", ".join(topic.citation_docids),
            "",
            "Return exactly one organizer JSON object with exactly `references` and `answer`;",
            "do not return Markdown. `references` must contain only raw docid strings from",
            "the full citation domain, and only when an answer object cites them. Each answer",
            "item must contain one self-contained sentence stating one claim and one to three",
            "unique raw docid strings in `citations`, ordered strongest support first. Never",
            "use numeric citation indexes. Keep the complete answer at or below 1,024 words.",
        ]
    )
    return "\n".join(lines)


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
    config: Any,
    topic: GenerationTopic,
    *,
    no_planner_control: bool,
    planner_only: bool,
    hybrid_plan: tuple[NarrativeBlueprint, BlueprintProjection] | None = None,
    writer_attempts: int,
) -> None:
    if hybrid_plan is not None:
        prompt = _render_hybrid_writer_prompt(topic, *hybrid_plan)
    elif no_planner_control:
        prompt = _render_no_planner_control_prompt(topic)
    else:
        prompt = render_planner_prompt(topic)
    evidence_chars = sum(len(row.text) for row in topic.evidence)
    print(f"topic={topic.topic_id}")
    print(
        "counts="
        f"groups:{len(topic.groups)},claims:{len(topic.claim_hints)},"
        f"evidence_passages:{len(topic.evidence)},citation_docids:{len(topic.citation_docids)}"
    )
    prompt_kind = "writer" if no_planner_control or hybrid_plan is not None else "planner"
    print(
        "sizes="
        f"narrative_words:{len(topic.narrative.split())},"
        f"{prompt_kind}_prompt_chars:{len(prompt)},"
        f"evidence_chars:{evidence_chars}"
    )
    planner_calls = 0 if no_planner_control or hybrid_plan is not None else 1
    effective_writer_attempts = 0 if planner_only else writer_attempts
    total_calls = planner_calls + effective_writer_attempts
    total_label = "total" if planner_only else "total_semantic_calls"
    print(
        "budget="
        f"planner:{planner_calls},writer:{effective_writer_attempts},"
        f"{total_label}:{total_calls}"
    )
    del config


def _private_root(
    config: Any, topic: GenerationTopic, *, mode: str = "blueprint_prototype"
) -> Path:
    return config.resolved_work_dir / mode / _safe_topic_name(topic.topic_id)


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


async def _run_planner_only(
    config: Any,
    topic: GenerationTopic,
    generator: OpenRouterJsonGenerator,
    *,
    api_key: str,
) -> Path:
    _validate_artifact_paths(config)
    _refuse_existing_state(config)
    private_root = _private_root(config, topic, mode="planner_only")
    private_root.mkdir(parents=True, exist_ok=False)
    planner_prompt = render_planner_prompt(topic)
    planner_prompt_sha256 = _digest_text(planner_prompt)
    planner_schema = planner_response_schema()
    planner_raw: dict[str, Any] | None = None
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blueprint-planner-only") as executor:
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
                "mode": "planner_only",
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
        state = serialize_blueprint_state(
            topic,
            blueprint,
            projection,
            planner_prompt_sha256=planner_prompt_sha256,
            writer_context_sha256=_digest_text(writer_context),
        )
        state_path = private_root / "blueprint.state.json"
        _write_json(state_path, state, api_key=api_key)
        load_blueprint_state(
            topic,
            json.loads(state_path.read_text(encoding="utf-8")),
            planner_prompt_sha256=planner_prompt_sha256,
        )
        return private_root
    except SemanticCompletionError as exc:
        _write_json(
            private_root / "planner.receipt.json",
            {
                "stage": "planner",
                "mode": "planner_only",
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
                    "mode": "planner_only",
                    "prompt_sha256": planner_prompt_sha256,
                    "schema_sha256": _digest_json(planner_schema),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                api_key=api_key,
            )
        raise


def _load_hybrid_plan(
    topic: GenerationTopic, path: Path
) -> tuple[NarrativeBlueprint, BlueprintProjection]:
    """Load and authenticate a planner-only state for a held-out writer run."""

    plan_path = path.expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError(f"hybrid planner state is not a file: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read hybrid planner state: {plan_path}") from exc
    planner_prompt_sha256 = _digest_text(render_planner_prompt(topic))
    try:
        return load_blueprint_state(
            topic,
            payload,
            planner_prompt_sha256=planner_prompt_sha256,
        )
    except Exception as exc:
        raise ValueError(f"invalid hybrid planner state: {plan_path}: {exc}") from exc


async def _run_hybrid(
    config: Any,
    topic: GenerationTopic,
    generator: OpenRouterJsonGenerator,
    *,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    api_key: str,
) -> dict[str, Any]:
    _validate_artifact_paths(config)
    _refuse_existing_state(config)
    private_root = _private_root(config, topic, mode="hybrid")
    private_root.mkdir(parents=True, exist_ok=False)
    user_prompt = _render_hybrid_writer_prompt(topic, blueprint, projection)
    writer_raw: dict[str, Any] | None = None
    receipt_path = private_root / "writer.receipt.json"
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blueprint-hybrid") as executor:
            generated, writer_raw = await _complete(
                generator,
                topic_id=topic.topic_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=output_schema(),
                executor=executor,
            )
        _write_json(
            receipt_path,
            {
                "stage": "writer",
                "mode": "hybrid",
                "attempt": 1,
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
            allowed_docids=list(topic.citation_docids),
        )
        _validate_generated_submission_record(
            record,
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            allowed_docids=list(topic.citation_docids),
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
        _write_json(
            receipt_path,
            {
                "stage": "writer",
                "mode": "hybrid",
                "attempt": 1,
                "prompt_sha256": _digest_text(user_prompt),
                "schema_sha256": _digest_json(output_schema()),
                "error": f"{type(exc).__name__}: {exc}",
                "raw_response": exc.raw_response,
            },
            api_key=api_key,
        )
        raise
    except Exception as exc:
        if not receipt_path.exists():
            _write_json(
                receipt_path,
                {
                    "stage": "writer",
                    "mode": "hybrid",
                    "attempt": 1,
                    "prompt_sha256": _digest_text(user_prompt),
                    "schema_sha256": _digest_json(output_schema()),
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response": writer_raw,
                },
                api_key=api_key,
            )
        raise


async def _run_no_planner_control(
    config: Any,
    topic: GenerationTopic,
    generator: OpenRouterJsonGenerator,
    *,
    api_key: str,
) -> dict[str, Any]:
    _validate_artifact_paths(config)
    _refuse_existing_state(config)
    private_root = _private_root(config, topic, mode="no_planner_control")
    private_root.mkdir(parents=True, exist_ok=False)
    user_prompt = _render_no_planner_control_prompt(topic)
    writer_raw: dict[str, Any] | None = None
    receipt_path = private_root / "writer.receipt.json"
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blueprint-control") as executor:
            generated, writer_raw = await _complete(
                generator,
                topic_id=topic.topic_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=output_schema(),
                executor=executor,
            )
        _write_json(
            receipt_path,
            {
                "stage": "writer",
                "control": "no_planner",
                "attempt": 1,
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
            allowed_docids=list(topic.citation_docids),
        )
        _validate_generated_submission_record(
            record,
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            allowed_docids=list(topic.citation_docids),
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
        _write_json(
            receipt_path,
            {
                "stage": "writer",
                "control": "no_planner",
                "attempt": 1,
                "prompt_sha256": _digest_text(user_prompt),
                "schema_sha256": _digest_json(output_schema()),
                "error": f"{type(exc).__name__}: {exc}",
                "raw_response": exc.raw_response,
            },
            api_key=api_key,
        )
        raise
    except Exception as exc:
        if not receipt_path.exists():
            _write_json(
                receipt_path,
                {
                    "stage": "writer",
                    "control": "no_planner",
                    "attempt": 1,
                    "prompt_sha256": _digest_text(user_prompt),
                    "schema_sha256": _digest_json(output_schema()),
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response": writer_raw,
                },
                api_key=api_key,
            )
        raise


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Prototype RAG YAML config.")
    parser.add_argument("--topic", required=True, help="Exactly one handoff topic ID.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print topic/count/size/budget statistics without provider calls.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-planner-control",
        action="store_true",
        help="Run the approved one-writer, no-planner control.",
    )
    mode_group.add_argument(
        "--planner-only",
        action="store_true",
        help="Run exactly one planner call, persist its state, and skip the writer.",
    )
    mode_group.add_argument(
        "--hybrid-plan",
        type=Path,
        help="Load a validated planner-only blueprint.state.json and run one writer call.",
    )
    parser.add_argument(
        "--writer-attempts",
        type=int,
        choices=(1, 2),
        default=None,
        help="Maximum writer semantic attempts for this isolated invocation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    config = load_rag_generation_config(args.config)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topic = _topic_for_cli(config, handoff, args.topic)
    if args.no_planner_control:
        if args.writer_attempts not in {None, 1}:
            raise SystemExit("--no-planner-control requires --writer-attempts 1")
        writer_attempts = 1
    elif args.planner_only:
        writer_attempts = 0
    elif args.hybrid_plan is not None:
        if args.writer_attempts not in {None, 1}:
            raise SystemExit("--hybrid-plan requires --writer-attempts 1")
        writer_attempts = 1
    else:
        writer_attempts = args.writer_attempts if args.writer_attempts is not None else 2
    hybrid_plan = (
        _load_hybrid_plan(topic, args.hybrid_plan)
        if args.hybrid_plan is not None
        else None
    )
    if args.dry_run:
        _print_dry_run(
            config,
            topic,
            no_planner_control=args.no_planner_control,
            planner_only=args.planner_only,
            hybrid_plan=hybrid_plan,
            writer_attempts=writer_attempts,
        )
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
    if args.planner_only:
        work_path = asyncio.run(
            _run_planner_only(config, topic, generator, api_key=api_key)
        )
        print(f"completed planner-only topic={topic.topic_id} work={work_path}")
        return
    if args.hybrid_plan is not None:
        if hybrid_plan is None:
            raise RuntimeError("hybrid planner state was not loaded")
        asyncio.run(
            _run_hybrid(
                config,
                topic,
                generator,
                blueprint=hybrid_plan[0],
                projection=hybrid_plan[1],
                api_key=api_key,
            )
        )
        print(f"completed hybrid topic={topic.topic_id} output={config.output_path}")
        return
    if args.no_planner_control:
        asyncio.run(_run_no_planner_control(config, topic, generator, api_key=api_key))
    else:
        asyncio.run(
            _run_live(
                config,
                topic,
                generator,
                api_key=api_key,
                writer_attempts=writer_attempts,
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

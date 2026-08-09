"""Throwaway one-call Luna replay over an authenticated bounded-revision draft."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from trec_rag.bounded_splice import (
    SpliceValidationError,
    apply_splice_operations,
    splice_response_schema,
    validate_splice_payload,
)
from trec_rag.competition_rag import (
    OpenRouterJsonGenerator,
    _atomic_write_text,
    _safe_topic_name,
    _validate_artifact_paths,
    load_rag_generation_config,
)
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
)
from trec_rag.narrative_blueprint import (
    BlueprintProjection,
    NarrativeBlueprint,
    load_blueprint_state,
    render_planner_prompt,
)
from trec_rag.narrative_blueprint_trial import (
    LUNA_MODEL,
    LUNA_REASONING_EFFORT,
    _bounded_draft_objects_prompt,
    _bounded_provider_call,
    _bounded_read_state,
    _bounded_rebind_candidate,
    _bounded_revalidate_files,
    _bounded_write_candidate,
    _bounded_write_identity,
    _digest_json,
    _digest_text,
    _render_hybrid_common_context,
    _topic_for_cli,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


LUNA_SPLICE_REPLAY_CONTRACT_VERSION = "luna_whole_answer_splice_replay_v1"
MAX_REPLAY_OPERATIONS = 3
LUNA_SPLICE_SYSTEM_PROMPT = (
    "You are a whole-answer selected-evidence omission editor. Use only the supplied "
    "authenticated passages, preserve useful draft content, and return only the requested JSON."
)


def luna_splice_response_schema() -> dict[str, object]:
    """Return the existing splice schema with the replay's smaller edit budget."""

    schema = deepcopy(splice_response_schema())
    schema["properties"]["operations"]["maxItems"] = MAX_REPLAY_OPERATIONS
    return schema


def _evidence_alias_docids(topic: GenerationTopic) -> dict[str, tuple[str, ...]]:
    """Bind each prompt evidence alias to exactly one authenticated document."""

    return {
        f"e{index:03d}": (row.docid,)
        for index, row in enumerate(topic.evidence, start=1)
    }


def render_luna_splice_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    *,
    draft: dict[str, Any],
) -> str:
    """Render one whole-answer, direct-evidence splice request."""

    context = _render_hybrid_common_context(topic, blueprint, projection)
    lines = [
        "WHOLE-ANSWER LUNA SPLICE REPLAY",
        "Review the complete validated draft against the untouched official narrative and all",
        "authenticated selected evidence. Return exactly one JSON object with `decision` and",
        "`operations`; never return a full answer.",
        "Use `keep_draft` with an empty operations array unless a small edit would materially",
        "improve the complete narrative. For `edit`, return at most three operations against the",
        "immutable original draft indexes.",
        "Prefer insertion when supported word headroom exists. Use replacement only when the new",
        "sentence is clearly more useful than everything removed; preserve every distinct caveat,",
        "tradeoff, causal qualification, and narrative-relevant detail in the removed range.",
        "Do not add a detail merely because it is interesting. Screen every proposed operation for",
        "full-narrative importance, nonredundancy, local coherence, and complete citation support",
        "before returning it. If no operation clears that bar, return `keep_draft`.",
        "Each new object must be one self-contained sentence stating one atomic claim. Prefer one",
        "strongest citation and use at most two unique raw docids. The operation's",
        "`audit_card_ids` must contain selected-evidence aliases such as `e001`, not audit cards;",
        "every cited docid must be the document bound to one of those named evidence aliases.",
        "Do not reuse one selected-evidence alias across multiple operations.",
        "Use `delete_count: 0` to insert before an index and `delete_count: 1`, `2`, or `3` to",
        "replace that original contiguous range. Pure deletion is not supported.",
        *_bounded_draft_objects_prompt(draft),
    ]
    return context + "\n\n" + "\n".join(lines)


def assemble_luna_splice_candidate(
    topic: GenerationTopic,
    draft: dict[str, Any],
    payload: object,
) -> dict[str, Any]:
    """Validate and atomically apply one direct-evidence Luna splice payload."""

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("operations"), list)
        and len(payload["operations"]) > MAX_REPLAY_OPERATIONS
    ):
        raise SpliceValidationError("at most three operations are allowed")
    operations = validate_splice_payload(
        draft,
        payload,
        tuple(topic.citation_docids),
        _evidence_alias_docids(topic),
    )
    if operations is None:
        return deepcopy(draft)
    return apply_splice_operations(draft, operations)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rebind_replay_candidate(
    record: dict[str, Any],
    *,
    topic: GenerationTopic,
    config: Any,
    run_id: str,
) -> dict[str, Any]:
    rebound = deepcopy(record)
    metadata = rebound.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("replay candidate has invalid metadata")
    metadata["team_id"] = config.team_id
    metadata["run_desc"] = config.run_desc
    metadata["run_id"] = run_id
    return _bounded_rebind_candidate(
        rebound,
        topic=topic,
        config=config,
        run_id=run_id,
    )


def load_replay_source(
    source_root: Path,
    *,
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
) -> tuple[NarrativeBlueprint, BlueprintProjection, dict[str, Any], dict[str, str]]:
    """Authenticate and load one completed bounded-revision planner/draft pair."""

    source_root = source_root.resolve()
    state = _bounded_read_state(source_root)
    _bounded_revalidate_files(source_root, state)
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("replay source has no authenticated identity")
    if identity.get("handoff_manifest_sha256") != handoff.manifest_sha256:
        raise ValueError("replay source handoff digest differs from the selected handoff")
    if identity.get("topic_context_sha256") != topic.context_sha256:
        raise ValueError("replay source topic-context digest differs from the selected topic")
    stages = state.get("stages")
    if not isinstance(stages, dict) or not stages.get("planner") or not stages.get("draft"):
        raise ValueError("replay source planner and draft must both be complete")

    blueprint_path = source_root / "blueprint.state.json"
    draft_path = source_root / "draft.record.json"
    try:
        blueprint_payload = json.loads(blueprint_path.read_text(encoding="utf-8"))
        draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read replay source planner or draft") from exc
    blueprint, projection = load_blueprint_state(
        topic,
        blueprint_payload,
        planner_prompt_sha256=sha256(render_planner_prompt(topic).encode()).hexdigest(),
    )
    draft = _rebind_replay_candidate(
        draft_payload,
        topic=topic,
        config=config,
        run_id=f"{config.run_id}-draft",
    )
    source_hashes = {
        "state.json": _file_sha256(source_root / "state.json"),
        "blueprint.state.json": _file_sha256(blueprint_path),
        "draft.record.json": _file_sha256(draft_path),
    }
    return blueprint, projection, draft, source_hashes


def finalize_luna_splice_payload(
    topic: GenerationTopic,
    draft: dict[str, Any],
    payload: object,
    *,
    config: Any,
    run_id: str,
) -> tuple[dict[str, Any], bool, str | None]:
    """Return a validated replay final, falling back atomically on invalid operations."""

    try:
        assembled = assemble_luna_splice_candidate(topic, draft, payload)
        final = _rebind_replay_candidate(
            assembled,
            topic=topic,
            config=config,
            run_id=run_id,
        )
        return final, False, None
    except (SpliceValidationError, ValueError, RuntimeError, TypeError) as exc:
        fallback = _rebind_replay_candidate(
            draft,
            topic=topic,
            config=config,
            run_id=run_id,
        )
        return fallback, True, f"{type(exc).__name__}: {exc}"


def _replay_root(config: Any, topic: GenerationTopic) -> Path:
    return config.resolved_work_dir / "luna_splice_replay" / _safe_topic_name(topic.topic_id)


def _write_state(root: Path, state: dict[str, Any]) -> None:
    _atomic_write_text(
        root / "state.json",
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _replay_identity(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    source_hashes: dict[str, str],
    prompt: str,
    schema: dict[str, object],
) -> dict[str, Any]:
    return {
        "contract_version": LUNA_SPLICE_REPLAY_CONTRACT_VERSION,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "topic_id": topic.topic_id,
        "topic_context_sha256": topic.context_sha256,
        "run_id": config.run_id,
        "team_id": config.team_id,
        "run_desc": config.run_desc,
        "provider": config.provider,
        "api_base": config.api_base,
        "api_key_env": config.api_key_env,
        "model": LUNA_MODEL,
        "reasoning_effort": LUNA_REASONING_EFFORT,
        "structured_output": config.structured_output,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "transport_max_attempts": config.transport_max_attempts,
        "source_hashes": source_hashes,
        "prompt_sha256": _digest_text(prompt),
        "system_prompt_sha256": _digest_text(LUNA_SPLICE_SYSTEM_PROMPT),
        "schema_sha256": _digest_json(schema),
    }


def _read_receipt(root: Path) -> dict[str, Any] | None:
    path = root / "receipts" / "whole-answer-splice.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read Luna replay receipt") from exc
    if not isinstance(payload, dict):
        raise ValueError("Luna replay receipt must be an object")
    return payload


def _receipt_payload(
    receipt: dict[str, Any] | None,
    *,
    prompt_sha256: str,
    schema_sha256: str,
) -> tuple[str | None, dict[str, Any] | None]:
    if receipt is None:
        return None, None
    if (
        receipt.get("prompt_sha256") != prompt_sha256
        or receipt.get("schema_sha256") != schema_sha256
    ):
        raise ValueError("Luna replay receipt prompt or schema hash differs")
    outcome = receipt.get("outcome")
    payload = receipt.get("accepted_payload")
    if outcome == "semantic_success" and not isinstance(payload, dict):
        raise ValueError("Luna replay receipt has no accepted semantic payload")
    return str(outcome), payload if isinstance(payload, dict) else None


def _may_start_luna_call(
    *,
    state_mode: str,
    reservation_status: str,
    receipt_outcome: str | None,
) -> bool:
    """Decide whether a call is safe without repeating an ambiguous semantic request."""

    if receipt_outcome in {"semantic_success", "semantic_error", "ambiguous_failure"}:
        return False
    if (
        state_mode == "resume"
        and reservation_status == "pending"
        and receipt_outcome is None
    ):
        raise RuntimeError("ambiguous pending Luna call cannot be repeated")
    return receipt_outcome in {None, "terminal_transport_failure"}


def _operation_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    operations = payload.get("operations") if isinstance(payload, dict) else None
    if not isinstance(operations, list):
        return {"decision": None, "count": 0, "insertions": 0, "replacements": 0}
    insertions = sum(
        isinstance(item, dict) and item.get("delete_count") == 0 for item in operations
    )
    return {
        "decision": payload.get("decision"),
        "count": len(operations),
        "insertions": insertions,
        "replacements": len(operations) - insertions,
    }


def _candidate_summary(record: dict[str, Any]) -> dict[str, int]:
    answer = record.get("answer")
    objects = answer if isinstance(answer, list) else []
    return {
        "words": sum(
            len(item.get("text", "").split()) for item in objects if isinstance(item, dict)
        ),
        "answer_objects": len(objects),
        "references": len(record.get("references", [])),
    }


async def run_luna_splice_replay(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    source_root: Path,
    api_key: str,
    state_mode: str,
) -> Path:
    """Run or resume one authenticated whole-answer Luna splice replay."""

    _validate_artifact_paths(config)
    blueprint, projection, draft, source_hashes = load_replay_source(
        source_root,
        config=config,
        handoff=handoff,
        topic=topic,
    )
    prompt = render_luna_splice_prompt(
        topic,
        blueprint,
        projection,
        draft=draft,
    )
    schema = luna_splice_response_schema()
    identity = _replay_identity(
        config,
        handoff,
        topic,
        source_hashes=source_hashes,
        prompt=prompt,
        schema=schema,
    )
    root = _replay_root(config, topic)
    if state_mode == "create":
        if root.exists():
            raise ValueError(f"Luna replay state already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        (root / "receipts").mkdir()
        state: dict[str, Any] = {
            "contract_version": LUNA_SPLICE_REPLAY_CONTRACT_VERSION,
            "identity": identity,
            "reservation": {"stage": "whole-answer-splice", "status": "not_started"},
            "call": None,
            "final": False,
            "used_draft_fallback": False,
            "validation_error": None,
        }
        _write_state(root, state)
    elif state_mode == "resume":
        try:
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Luna replay state under {root}") from exc
        if not isinstance(state, dict) or state.get("contract_version") != LUNA_SPLICE_REPLAY_CONTRACT_VERSION:
            raise ValueError("Luna replay state has an unsupported contract")
        if state.get("identity") != identity:
            raise ValueError("Luna replay resume identity differs from create identity")
        if state.get("final") and (root / "manifest.json").is_file():
            return root
    else:
        raise ValueError("state_mode must be create or resume")

    draft_path = root / "draft.record.json"
    _atomic_write_text(
        draft_path,
        json.dumps(draft, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    draft_output = _bounded_write_candidate(root, "draft", draft)
    draft_identity = _bounded_write_identity(
        root,
        config=config,
        handoff=handoff,
        topic=topic,
        arm="draft",
    )

    receipt = _read_receipt(root)
    receipt_outcome, payload = _receipt_payload(
        receipt,
        prompt_sha256=identity["prompt_sha256"],
        schema_sha256=identity["schema_sha256"],
    )
    if _may_start_luna_call(
        state_mode=state_mode,
        reservation_status=str(state["reservation"].get("status", "")),
        receipt_outcome=receipt_outcome,
    ):
        state["reservation"]["status"] = "pending"
        _write_state(root, state)
        luna = OpenRouterJsonGenerator(
            api_base=config.api_base,
            api_key=api_key,
            model=LUNA_MODEL,
            reasoning_effort=LUNA_REASONING_EFFORT,
            structured_output=config.structured_output,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            transport_max_attempts=config.transport_max_attempts,
        )
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="luna-splice-replay") as executor:
            payload, call = await _bounded_provider_call(
                luna,
                root=root,
                api_key=api_key,
                topic_id=topic.topic_id,
                stage="whole-answer-splice",
                model=LUNA_MODEL,
                reasoning_effort=LUNA_REASONING_EFFORT,
                system_prompt=LUNA_SPLICE_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=schema,
                executor=executor,
                reservation_ordinal=1,
            )
        state["call"] = call
        state["reservation"]["status"] = call["outcome"]
        _write_state(root, state)
        if call["outcome"] == "terminal_transport_failure":
            raise RuntimeError("Luna replay transport failed; resume may retry it")
    elif receipt is not None:
        state["call"] = {
            key: receipt.get(key)
            for key in (
                "stage",
                "model",
                "reasoning_effort",
                "prompt_sha256",
                "schema_sha256",
                "reservation_ordinal",
                "outcome",
                "latency_seconds",
                "provider_cost",
                "usage",
                "transport_outcome",
                "error",
            )
        }
        state["reservation"]["status"] = receipt_outcome

    if payload is None:
        final = _rebind_replay_candidate(
            draft,
            topic=topic,
            config=config,
            run_id=f"{config.run_id}-final",
        )
        used_fallback = True
        validation_error = "Luna replay returned no accepted semantic payload"
    else:
        final, used_fallback, validation_error = finalize_luna_splice_payload(
            topic,
            draft,
            payload,
            config=config,
            run_id=f"{config.run_id}-final",
        )

    final_output = _bounded_write_candidate(root, "final", final)
    final_identity = _bounded_write_identity(
        root,
        config=config,
        handoff=handoff,
        topic=topic,
        arm="final",
    )
    state["final"] = True
    state["used_draft_fallback"] = used_fallback
    state["validation_error"] = validation_error
    state["artifacts"] = {
        "draft_record": _file_sha256(draft_path),
        "draft_submission": _file_sha256(draft_output),
        "draft_identity": _file_sha256(draft_identity),
        "final_submission": _file_sha256(final_output),
        "final_identity": _file_sha256(final_identity),
    }
    _write_state(root, state)
    call = state.get("call") if isinstance(state.get("call"), dict) else {}
    manifest = {
        "contract_version": LUNA_SPLICE_REPLAY_CONTRACT_VERSION,
        "topic_id": topic.topic_id,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "topic_context_sha256": topic.context_sha256,
        "source_hashes": source_hashes,
        "luna_calls": 1 if call else 0,
        "sol_calls": 0,
        "provider_cost": float(call.get("provider_cost", 0.0) or 0.0),
        "transport_outcome": call.get("transport_outcome"),
        "semantic_outcome": call.get("outcome"),
        "operations": _operation_summary(payload),
        "used_draft_fallback": used_fallback,
        "validation_error": validation_error,
        "draft": _candidate_summary(draft),
        "final": _candidate_summary(final),
        "final_equals_draft": final.get("answer") == draft.get("answer"),
    }
    _atomic_write_text(
        root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return root


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("create", "resume"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    config = load_rag_generation_config(args.config)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topic = _topic_for_cli(config, handoff, args.topic)
    blueprint, projection, draft, source_hashes = load_replay_source(
        args.source_root,
        config=config,
        handoff=handoff,
        topic=topic,
    )
    if args.dry_run:
        prompt = render_luna_splice_prompt(
            topic,
            blueprint,
            projection,
            draft=draft,
        )
        print(f"topic={topic.topic_id}")
        print(f"source={args.source_root.resolve()}")
        print(
            "counts="
            f"groups:{len(topic.groups)},claims:{len(topic.claim_hints)},"
            f"evidence_passages:{len(topic.evidence)},citation_docids:{len(topic.citation_docids)}"
        )
        print(f"sizes=prompt_chars:{len(prompt)},source_files:{len(source_hashes)}")
        print("calls=luna_whole_answer_splice:1,sol:0,provider:0")
        return
    if args.state_mode is None:
        raise SystemExit("live Luna replay requires --state-mode create|resume")
    repo_root = find_repo_root(args.config.resolve().parent)
    load_repo_env(repo_root)
    api_key = os.environ.get(config.api_key_env, "")
    root = asyncio.run(
        run_luna_splice_replay(
            config,
            handoff,
            topic,
            source_root=args.source_root,
            api_key=api_key,
            state_mode=args.state_mode,
        )
    )
    print(f"completed Luna splice replay topic={topic.topic_id} work={root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc

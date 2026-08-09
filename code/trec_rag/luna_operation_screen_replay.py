"""Authenticated one-call Luna screening over frozen Sol splice operations."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from trec_rag.bounded_splice import (
    SpliceOperation,
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
from trec_rag.luna_splice_replay import (
    _candidate_summary,
    _file_sha256,
    _rebind_replay_candidate,
)
from trec_rag.narrative_blueprint import (
    BlueprintProjection,
    NarrativeBlueprint,
    load_blueprint_state,
    render_planner_prompt,
)
from trec_rag.narrative_blueprint_trial import (
    _audit_card_docids,
    _bounded_draft_objects_prompt,
    _bounded_provider_call,
    _bounded_read_state,
    _bounded_revalidate_files,
    _bounded_revision_prompt,
    _bounded_write_candidate,
    _bounded_write_identity,
    _digest_json,
    _digest_text,
    _render_hybrid_common_context,
    _topic_for_cli,
    LUNA_MODEL,
    LUNA_REASONING_EFFORT,
)
from trec_rag.operation_screen import (
    OperationScreenResult,
    OperationScreenValidationError,
    operation_ids,
    operation_screen_response_schema,
    validate_operation_screen_payload,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


OPERATION_SCREEN_REPLAY_CONTRACT_VERSION = "luna_operation_screen_replay_v1"
LUNA_OPERATION_SCREEN_SYSTEM_PROMPT = (
    "You are a strict selected-evidence operation judge. Judge every proposed edit, never "
    "rewrite answer content, and return only the requested JSON."
)


@dataclass(frozen=True)
class OperationScreenSource:
    blueprint: NarrativeBlueprint
    projection: BlueprintProjection
    draft: dict[str, Any]
    audit_cards: tuple[dict[str, Any], ...]
    operations: tuple[SpliceOperation, ...]
    source_hashes: dict[str, str]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_audit_cards(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read merged audit cards") from exc
    if not isinstance(value, list) or any(not isinstance(card, dict) for card in value):
        raise ValueError("merged audit cards must be an array of objects")
    return tuple(dict(card) for card in value)


def _revision_reservation(state: dict[str, Any]) -> dict[str, Any]:
    matches = [
        reservation
        for reservation in state.get("sol_reservations", [])
        if isinstance(reservation, dict)
        and reservation.get("role") == "revision"
        and reservation.get("stage") == "revision"
    ]
    if len(matches) != 1:
        raise ValueError("source must contain exactly one revision reservation")
    reservation = matches[0]
    if (
        reservation.get("status") != "semantic_returned"
        or reservation.get("receipt") != "receipts/revision.json"
    ):
        raise ValueError("source revision reservation is not completed")
    return reservation


def _validate_revision_receipt(
    state: dict[str, Any],
    receipt: dict[str, Any],
    *,
    prompt_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    if (
        receipt.get("prompt_sha256") != prompt_sha256
        or receipt.get("schema_sha256") != schema_sha256
    ):
        raise ValueError("revision receipt prompt or schema hash differs")
    if receipt.get("stage") != "revision" or receipt.get("outcome") != "semantic_success":
        raise ValueError("revision receipt is not a semantic-success revision")
    payload = receipt.get("accepted_payload")
    if not isinstance(payload, dict):
        raise ValueError("revision receipt has no accepted splice payload")

    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("source state has no identity")
    if (
        receipt.get("model") != identity.get("sol_model")
        or receipt.get("reasoning_effort") != identity.get("sol_reasoning_effort")
    ):
        raise ValueError("revision receipt model identity differs")
    reservation = _revision_reservation(state)
    if receipt.get("reservation_ordinal") != reservation.get("ordinal"):
        raise ValueError("revision receipt reservation differs")

    matching_calls = [
        call
        for call in state.get("calls", [])
        if isinstance(call, dict)
        and call.get("stage") == "revision"
        and call.get("reservation_ordinal") == reservation.get("ordinal")
    ]
    if len(matching_calls) != 1:
        raise ValueError("source state revision call is missing or ambiguous")
    call = matching_calls[0]
    for key in (
        "model",
        "reasoning_effort",
        "prompt_sha256",
        "schema_sha256",
        "outcome",
        "transport_outcome",
    ):
        if call.get(key) != receipt.get(key):
            raise ValueError(f"source state revision call differs for {key}")
    return payload


def load_operation_screen_source(
    source_root: Path,
    *,
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
) -> OperationScreenSource:
    """Authenticate and load one frozen draft, audit, and Sol operation set."""

    root = source_root.resolve()
    state = _bounded_read_state(root)
    _bounded_revalidate_files(root, state)
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("operation-screen source has no authenticated identity")
    if identity.get("handoff_manifest_sha256") != handoff.manifest_sha256:
        raise ValueError("operation-screen source handoff digest differs")
    if identity.get("topic_context_sha256") != topic.context_sha256:
        raise ValueError("operation-screen source topic-context digest differs")
    stages = state.get("stages")
    required_stages = ("planner", "draft", "audit_merge", "revision", "final")
    if not isinstance(stages, dict) or any(not stages.get(stage) for stage in required_stages):
        raise ValueError("operation-screen source is not a completed bounded revision")

    blueprint_path = root / "blueprint.state.json"
    draft_path = root / "draft.record.json"
    audit_path = root / "audit.merged.json"
    receipt_path = root / "receipts" / "revision.json"
    blueprint_payload = _load_json_object(blueprint_path, label="source blueprint state")
    draft_original = _load_json_object(draft_path, label="source draft")
    audit_cards = _load_audit_cards(audit_path)
    receipt = _load_json_object(receipt_path, label="source revision receipt")
    blueprint, projection = load_blueprint_state(
        topic,
        blueprint_payload,
        planner_prompt_sha256=sha256(render_planner_prompt(topic).encode()).hexdigest(),
    )
    revision_prompt = _bounded_revision_prompt(
        topic,
        blueprint,
        projection,
        draft=draft_original,
        audit_cards=audit_cards,
    )
    revision_payload = _validate_revision_receipt(
        state,
        receipt,
        prompt_sha256=_digest_text(revision_prompt),
        schema_sha256=_digest_json(splice_response_schema()),
    )
    operations = validate_splice_payload(
        draft_original,
        revision_payload,
        tuple(topic.citation_docids),
        _audit_card_docids(topic, audit_cards),
    )
    if operations is None:
        raise ValueError("source revision contains no operations to screen")
    draft = _rebind_replay_candidate(
        draft_original,
        topic=topic,
        config=config,
        run_id=f"{config.run_id}-draft",
    )
    return OperationScreenSource(
        blueprint=blueprint,
        projection=projection,
        draft=draft,
        audit_cards=audit_cards,
        operations=operations,
        source_hashes={
            "state.json": _file_sha256(root / "state.json"),
            "blueprint.state.json": _file_sha256(blueprint_path),
            "draft.record.json": _file_sha256(draft_path),
            "audit.merged.json": _file_sha256(audit_path),
            "receipts/revision.json": _file_sha256(receipt_path),
        },
    )


def _operation_payload(operation_id: str, operation: SpliceOperation) -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        "start_index": operation.start_index,
        "delete_count": operation.delete_count,
        "new_object": {
            "text": operation.text,
            "citations": list(operation.citations),
        },
        "audit_card_ids": list(operation.audit_card_ids),
    }


def render_operation_screen_prompt(
    topic: GenerationTopic,
    source: OperationScreenSource,
) -> str:
    """Render one whole-answer, decisions-only screen over frozen operations."""

    context = _render_hybrid_common_context(topic, source.blueprint, source.projection)
    cards_by_id = {str(card.get("card_id")): card for card in source.audit_cards}
    selected_card_ids = tuple(
        dict.fromkeys(
            card_id
            for operation in source.operations
            for card_id in operation.audit_card_ids
        )
    )
    lines = [
        "WHOLE-ANSWER OPERATION SCREEN",
        "Judge the frozen Sol operations against the complete official narrative, surviving draft,",
        "named audit cards, and authenticated selected evidence. Return decisions only; never rewrite",
        "answer prose, citations, indexes, ranges, or audit-card IDs.",
        "For each operation return every required boolean gate:",
        "- `fully_supported`: every cited document's selected passages fully support the complete",
        "  new sentence without outside knowledge; collective or partial support is false.",
        "- `atomic`: the new sentence is one coherent claim at the answer-object citation unit.",
        "- `material`: the edit materially improves the answer to the complete narrative.",
        "- `nonredundant`: surviving draft prose does not already communicate the same point.",
        "- `replacement_safe`: true for an insertion; for a replacement, true only when the new",
        "  object is more useful than everything removed and loses no distinct caveat, causal",
        "  qualification, tradeoff, or narrative-relevant detail.",
        "Local code accepts an operation only when all five gates are true. Do not optimize the",
        "number accepted and do not infer an expected decision from the topic or operation ID.",
        *_bounded_draft_objects_prompt(source.draft),
        "FROZEN VALIDATED SOL OPERATIONS (stable IDs; immutable):",
    ]
    for operation_id, operation in zip(
        operation_ids(source.operations), source.operations, strict=True
    ):
        lines.append(
            f"[{operation_id}] "
            + json.dumps(
                _operation_payload(operation_id, operation),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    lines.append("NAMED MERGED AUDIT CARDS (advisory, not factual authority):")
    for card_id in selected_card_ids:
        card = cards_by_id.get(card_id)
        if card is None:
            raise ValueError(f"operation names missing audit card: {card_id}")
        lines.append(f"[{card_id}] {json.dumps(card, ensure_ascii=False, sort_keys=True)}")
    lines.append("Return exactly the strict JSON response; decisions only and never rewrite.")
    return context + "\n\n" + "\n".join(lines)


def finalize_operation_screen(
    topic: GenerationTopic,
    draft: dict[str, Any],
    operations: tuple[SpliceOperation, ...],
    payload: object,
    *,
    config: Any,
    run_id: str,
) -> tuple[dict[str, Any], OperationScreenResult | None, bool, str | None]:
    """Apply only accepted operations, or return the validated draft atomically."""

    try:
        result = validate_operation_screen_payload(payload, operations)
        assembled = (
            apply_splice_operations(draft, result.accepted_operations)
            if result.accepted_operations
            else draft
        )
        final = _rebind_replay_candidate(
            assembled,
            topic=topic,
            config=config,
            run_id=run_id,
        )
        return final, result, False, None
    except (
        OperationScreenValidationError,
        SpliceValidationError,
        ValueError,
        RuntimeError,
        TypeError,
    ) as exc:
        fallback = _rebind_replay_candidate(
            draft,
            topic=topic,
            config=config,
            run_id=run_id,
        )
        return fallback, None, True, f"{type(exc).__name__}: {exc}"


def _may_start_operation_screen_call(
    *,
    state_mode: str,
    reservation_status: str,
    receipt_outcome: str | None,
) -> bool:
    """Refuse any possible duplicate semantic screen request."""

    if receipt_outcome in {"semantic_success", "semantic_error", "ambiguous_failure"}:
        return False
    if state_mode == "resume" and reservation_status == "pending" and receipt_outcome is None:
        raise RuntimeError("ambiguous pending operation-screen call cannot be repeated")
    return receipt_outcome in {None, "terminal_transport_failure"}


def _replay_root(config: Any, topic: GenerationTopic) -> Path:
    return config.resolved_work_dir / "luna_operation_screen_replay" / _safe_topic_name(
        topic.topic_id
    )


def _write_state(root: Path, state: dict[str, Any]) -> None:
    _atomic_write_text(
        root / "state.json",
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _screen_identity(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    source_hashes: dict[str, str],
    prompt: str,
    schema: dict[str, object],
) -> dict[str, Any]:
    return {
        "contract_version": OPERATION_SCREEN_REPLAY_CONTRACT_VERSION,
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
        "system_prompt_sha256": _digest_text(LUNA_OPERATION_SCREEN_SYSTEM_PROMPT),
        "schema_sha256": _digest_json(schema),
    }


def _read_screen_receipt(root: Path) -> dict[str, Any] | None:
    path = root / "receipts" / "operation-screen.json"
    if not path.is_file():
        return None
    return _load_json_object(path, label="operation-screen receipt")


def _screen_receipt_payload(
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
        raise ValueError("operation-screen receipt prompt or schema hash differs")
    outcome = receipt.get("outcome")
    payload = receipt.get("accepted_payload")
    if outcome == "semantic_success" and not isinstance(payload, dict):
        raise ValueError("operation-screen receipt has no accepted semantic payload")
    return str(outcome), payload if isinstance(payload, dict) else None


async def run_luna_operation_screen_replay(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    source_root: Path,
    api_key: str,
    state_mode: str,
) -> Path:
    """Run or resume one authenticated decisions-only operation screen."""

    _validate_artifact_paths(config)
    if config.model != LUNA_MODEL or config.reasoning_effort != LUNA_REASONING_EFFORT:
        raise ValueError(
            "operation-screen replay config must use Luna with medium reasoning"
        )
    source = load_operation_screen_source(
        source_root,
        config=config,
        handoff=handoff,
        topic=topic,
    )
    prompt = render_operation_screen_prompt(topic, source)
    schema = operation_screen_response_schema(len(source.operations))
    identity = _screen_identity(
        config,
        handoff,
        topic,
        source_hashes=source.source_hashes,
        prompt=prompt,
        schema=schema,
    )
    root = _replay_root(config, topic)
    if state_mode == "create":
        if root.exists():
            raise ValueError(f"operation-screen replay state already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        (root / "receipts").mkdir()
        state: dict[str, Any] = {
            "contract_version": OPERATION_SCREEN_REPLAY_CONTRACT_VERSION,
            "identity": identity,
            "reservation": {"stage": "operation-screen", "status": "not_started"},
            "call": None,
            "final": False,
            "used_draft_fallback": False,
            "validation_error": None,
        }
        _write_state(root, state)
    elif state_mode == "resume":
        state = _load_json_object(root / "state.json", label="operation-screen replay state")
        if state.get("contract_version") != OPERATION_SCREEN_REPLAY_CONTRACT_VERSION:
            raise ValueError("operation-screen replay state has an unsupported contract")
        if state.get("identity") != identity:
            raise ValueError("operation-screen replay resume identity differs from create identity")
        if state.get("final") and (root / "manifest.json").is_file():
            return root
    else:
        raise ValueError("state_mode must be create or resume")

    draft_path = root / "draft.record.json"
    _atomic_write_text(
        draft_path,
        json.dumps(source.draft, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    draft_output = _bounded_write_candidate(root, "draft", source.draft)
    draft_identity = _bounded_write_identity(
        root,
        config=config,
        handoff=handoff,
        topic=topic,
        arm="draft",
    )

    receipt = _read_screen_receipt(root)
    receipt_outcome, payload = _screen_receipt_payload(
        receipt,
        prompt_sha256=identity["prompt_sha256"],
        schema_sha256=identity["schema_sha256"],
    )
    if _may_start_operation_screen_call(
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
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="luna-operation-screen",
        ) as executor:
            payload, call = await _bounded_provider_call(
                luna,
                root=root,
                api_key=api_key,
                topic_id=topic.topic_id,
                stage="operation-screen",
                model=LUNA_MODEL,
                reasoning_effort=LUNA_REASONING_EFFORT,
                system_prompt=LUNA_OPERATION_SCREEN_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=schema,
                executor=executor,
                reservation_ordinal=1,
            )
        state["call"] = call
        state["reservation"]["status"] = call["outcome"]
        _write_state(root, state)
        if call["outcome"] == "terminal_transport_failure":
            raise RuntimeError("operation-screen transport failed; resume may retry it")
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
            source.draft,
            topic=topic,
            config=config,
            run_id=f"{config.run_id}-final",
        )
        result = None
        used_fallback = True
        validation_error = "operation screen returned no accepted semantic payload"
    else:
        final, result, used_fallback, validation_error = finalize_operation_screen(
            topic,
            source.draft,
            source.operations,
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

    decisions = result.decisions if result is not None else ()
    rejection_reasons = Counter(
        reason
        for decision in decisions
        for reason in decision.rejection_reasons
    )
    accepted_count = len(result.accepted_operations) if result is not None else 0
    call = state.get("call") if isinstance(state.get("call"), dict) else {}
    manifest = {
        "contract_version": OPERATION_SCREEN_REPLAY_CONTRACT_VERSION,
        "topic_id": topic.topic_id,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "topic_context_sha256": topic.context_sha256,
        "source_hashes": source.source_hashes,
        "screened_operations": len(source.operations),
        "screen_payload_valid": result is not None,
        "accepted_operations": accepted_count,
        "rejected_operations": len(source.operations) - accepted_count if result else 0,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "luna_calls": 1 if call else 0,
        "sol_calls": 0,
        "provider_cost": float(call.get("provider_cost", 0.0) or 0.0),
        "transport_outcome": call.get("transport_outcome"),
        "semantic_outcome": call.get("outcome"),
        "used_draft_fallback": used_fallback,
        "validation_error": validation_error,
        "draft": _candidate_summary(source.draft),
        "final": _candidate_summary(final),
        "final_equals_draft": final.get("answer") == source.draft.get("answer"),
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
    source = load_operation_screen_source(
        args.source_root,
        config=config,
        handoff=handoff,
        topic=topic,
    )
    if args.dry_run:
        prompt = render_operation_screen_prompt(topic, source)
        print(f"topic={topic.topic_id}")
        print(f"source={args.source_root.resolve()}")
        print(
            "counts="
            f"groups:{len(topic.groups)},claims:{len(topic.claim_hints)},"
            f"evidence_passages:{len(topic.evidence)},citation_docids:{len(topic.citation_docids)},"
            f"operations:{len(source.operations)}"
        )
        print(f"sizes=prompt_chars:{len(prompt)},source_files:{len(source.source_hashes)}")
        print("calls=luna_operation_screen:1,sol:0,provider:0")
        return
    if args.state_mode is None:
        raise SystemExit("live operation-screen replay requires --state-mode create|resume")
    repo_root = find_repo_root(args.config.resolve().parent)
    load_repo_env(repo_root)
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key.strip():
        raise ValueError(f"missing required API key: {config.api_key_env}")
    root = asyncio.run(
        run_luna_operation_screen_replay(
            config,
            handoff,
            topic,
            source_root=args.source_root,
            api_key=api_key,
            state_mode=args.state_mode,
        )
    )
    print(f"completed Luna operation-screen replay topic={topic.topic_id} work={root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc

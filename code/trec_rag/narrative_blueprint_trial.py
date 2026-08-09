"""Thin, throwaway one-topic experiment for narrative-blueprint generation.

Question: does a compact narrative blueprint improve answer coverage for one
fixed-evidence topic under the 1,024-word cap without degrading citations?

Legacy modes intentionally have no resume, overwrite, orchestration, or test
surface. The opt-in bounded-revision mode adds durable one-topic state and a
strict call ledger while continuing to use the existing sealed handoff helpers.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import time
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
    _generation_identity,
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
from trec_rag.bounded_splice import (
    SpliceOperation,
    SpliceValidationError,
    apply_splice_operations,
    splice_response_schema,
    validate_repaired_splice_payload,
    validate_splice_payload,
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
from trec_rag.operation_screen import (
    OperationScreenResult,
    OperationScreenValidationError,
    operation_ids,
    operation_screen_response_schema,
    validate_operation_screen_payload,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


PLANNER_SYSTEM_PROMPT = """You are a narrative blueprint planner. Plan coverage of the complete
official narrative from the narrative and advisory group/claim hints supplied by the user.
Generated groups are retrieval structure, not separate answer requirements. Claim hints are
advisory. Do not invent evidence, document identifiers, or claim identifiers; use only the
provided local aliases. Return only the requested JSON object."""


TRIAL_CONTRACT_VERSION = "bounded_narrative_revision_trial_v5_operation_screen"
SPLICE_PROMPT_CONTRACT_VERSION = "bounded_splice_revision_prompt_v2_citation_bound"
LUNA_MODEL = "openai/gpt-5.6-luna"
LUNA_REASONING_EFFORT = "medium"
MAX_SOL_RESERVATIONS = 3
MAX_AUDIT_CARDS_PER_GROUP = 6
MAX_MERGED_AUDIT_CARDS = 24

LUNA_OPERATION_SCREEN_SYSTEM_PROMPT = (
    "You are a strict selected-evidence operation judge. Judge every proposed edit, never "
    "rewrite answer content, and return only the requested JSON."
)

_AUDIT_IMPORTANCE = {"must": 3, "should": 2, "could": 1}
_AUDIT_OMISSION_TYPES = (
    "missing",
    "too generic",
    "incomplete enumeration",
    "missing quantity/example",
    "unbalanced tradeoff",
    "redundant-space replacement",
)
_AUDIT_SPECIFICITY = {
    "missing quantity/example": 6,
    "incomplete enumeration": 5,
    "unbalanced tradeoff": 4,
    "missing": 3,
    "too generic": 2,
    "redundant-space replacement": 1,
}


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


def _render_hybrid_common_context(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
) -> str:
    """Render authenticated narrative, plan, hints, evidence, and citation context."""

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
        ]
    )
    return "\n".join(lines)


def _render_hybrid_writer_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
) -> str:
    """Render one hybrid writer prompt from an authenticated planner state."""

    common = _render_hybrid_common_context(topic, blueprint, projection)
    return common + "\n" + "\n".join(
        [
            "Return exactly one organizer JSON object with exactly `references` and `answer`;",
            "do not return Markdown. `references` must contain only raw docid strings from",
            "the full citation domain, and only when an answer object cites them. Each answer",
            "item must contain one self-contained sentence stating one claim and one to three",
            "unique raw docid strings in `citations`, ordered strongest support first. Never",
            "use numeric citation indexes. Keep the complete answer at or below 1,024 words.",
        ]
    )


def _audit_aliases(
    topic: GenerationTopic,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return deterministic display aliases and the evidence-to-group mapping."""

    group_aliases = {
        group.group_id: f"g{index:03d}"
        for index, group in enumerate(topic.groups, start=1)
    }
    evidence_aliases = {
        evidence.evidence_id: f"e{index:03d}"
        for index, evidence in enumerate(topic.evidence, start=1)
    }
    evidence_groups = {evidence.evidence_id: evidence.group_id for evidence in topic.evidence}
    return group_aliases, evidence_aliases, evidence_groups


def audit_response_schema() -> dict[str, object]:
    """Return the strict structured-output schema for one group omission audit."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["cards"],
        "properties": {
            "cards": {
                "type": "array",
                "maxItems": MAX_AUDIT_CARDS_PER_GROUP,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "group_alias",
                        "missing_detail",
                        "evidence_aliases",
                        "importance",
                        "omission_type",
                        "rationale",
                        "replacement_answer_index",
                    ],
                    "properties": {
                        "group_alias": {"type": "string"},
                        "missing_detail": {"type": "string"},
                        "evidence_aliases": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["must", "should", "could"],
                        },
                        "omission_type": {
                            "type": "string",
                            "enum": list(_AUDIT_OMISSION_TYPES),
                        },
                        "rationale": {"type": "string"},
                        "replacement_answer_index": {
                            "anyOf": [
                                {"type": "integer", "minimum": 0},
                                {"type": "null"},
                            ]
                        },
                    },
                },
            }
        },
    }


def render_group_audit_prompt(
    topic: GenerationTopic,
    *,
    group_id: str,
    draft: dict[str, Any],
) -> str:
    """Render one evidence-only post-draft audit for an authenticated group."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    if not isinstance(group_id, str) or group_id not in {group.group_id for group in topic.groups}:
        raise ValueError(f"unknown audit group: {group_id!r}")
    if not isinstance(draft, dict):
        raise TypeError("draft must be an object")

    group_aliases, evidence_aliases, _evidence_groups = _audit_aliases(topic)
    group = next(group for group in topic.groups if group.group_id == group_id)
    claims = [claim for claim in topic.claim_hints if claim.group_id == group_id]
    evidence = [row for row in topic.evidence if row.group_id == group_id]
    lines = [
        "BOUNDED NARRATIVE OMISSION AUDIT",
        "Audit only the supplied draft against the untouched official narrative and this",
        "authenticated group's selected evidence. Do not request retrieval or invent facts.",
        "Return at most six evidence-backed cards. Empty output is valid.",
        "Use only the local aliases shown below; never emit raw document or claim IDs.",
        "A card must identify a concise missing detail, cite one or more passages that fully",
        "support it, and explain why it matters. Do not reward interesting but unsupported facts.",
        "",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        f"GROUP [{group_aliases[group.group_id]}]: {group.text}",
        "",
        "ADVISORY CLAIM HINTS (ALL LINKED HINTS):",
    ]
    if claims:
        for claim in claims:
            claim_index = next(
                index for index, candidate in enumerate(topic.claim_hints, start=1)
                if candidate.claim_id == claim.claim_id
            )
            lines.append(f"[c{claim_index:03d}] {claim.text}")
    else:
        lines.append("(none)")
    lines.extend(["", "SELECTED EVIDENCE (ALL GROUP PASSAGES):"])
    for row in evidence:
        lines.extend(
            [
                f"[{evidence_aliases[row.evidence_id]}] DOCID: {row.docid}",
                row.text,
            ]
        )
    lines.extend(
        [
            "",
            "VALIDATED DRAFT (PRIVATE CANDIDATE):",
            json.dumps(draft, ensure_ascii=False, sort_keys=True),
            "",
            "Return exactly the audit response JSON object described by the schema.",
            "Allowed omission_type values: " + ", ".join(_AUDIT_OMISSION_TYPES) + ".",
            "replacement_answer_index is optional and is zero-based when present.",
        ]
    )
    return "\n".join(lines)


def _normalized_audit_detail(value: str) -> str:
    return " ".join(value.split()).casefold().rstrip(".?!")


def validate_group_audit(
    topic: GenerationTopic,
    *,
    group_id: str,
    payload: object,
) -> tuple[dict[str, Any], ...]:
    """Validate one strict audit payload and resolve all aliases locally."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    group_aliases, evidence_aliases, evidence_groups = _audit_aliases(topic)
    if group_id not in group_aliases:
        raise ValueError(f"unknown audit group: {group_id!r}")
    if not isinstance(payload, dict) or set(payload) != {"cards"}:
        raise ValueError("audit response must contain exactly cards")
    cards = payload["cards"]
    if not isinstance(cards, list) or len(cards) > MAX_AUDIT_CARDS_PER_GROUP:
        raise ValueError("audit cards must be an array of at most six items")
    expected = {
        "group_alias",
        "missing_detail",
        "evidence_aliases",
        "importance",
        "omission_type",
        "rationale",
    }
    output: list[dict[str, Any]] = []
    for index, item in enumerate(cards):
        if not isinstance(item, dict):
            raise ValueError(f"audit cards[{index}] must be an object")
        if set(item) - (expected | {"replacement_answer_index"}):
            raise ValueError(f"audit cards[{index}] has unexpected fields")
        if set(item) != expected and set(item) != expected | {"replacement_answer_index"}:
            raise ValueError(f"audit cards[{index}] is missing required fields")
        alias = item.get("group_alias")
        if alias != group_aliases[group_id]:
            raise ValueError(f"audit cards[{index}] has invalid group_alias {alias!r}")
        detail = item.get("missing_detail")
        rationale = item.get("rationale")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"audit cards[{index}] missing_detail must be nonempty text")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"audit cards[{index}] rationale must be nonempty text")
        evidence_alias_list = item.get("evidence_aliases")
        if (
            not isinstance(evidence_alias_list, list)
            or not evidence_alias_list
            or any(alias not in evidence_aliases.values() for alias in evidence_alias_list)
            or len(set(evidence_alias_list)) != len(evidence_alias_list)
        ):
            raise ValueError(f"audit cards[{index}] has invalid evidence_aliases")
        evidence_ids = tuple(
            evidence_id
            for evidence_id, alias_value in evidence_aliases.items()
            if alias_value in evidence_alias_list
        )
        if any(evidence_groups[evidence_id] != group_id for evidence_id in evidence_ids):
            raise ValueError(f"audit cards[{index}] cites evidence outside its group")
        importance = item.get("importance")
        if importance not in _AUDIT_IMPORTANCE:
            raise ValueError(f"audit cards[{index}] has invalid importance")
        omission_type = item.get("omission_type")
        if omission_type not in _AUDIT_SPECIFICITY:
            raise ValueError(f"audit cards[{index}] has invalid omission_type")
        replacement_index = item.get("replacement_answer_index")
        if replacement_index is not None and (
            type(replacement_index) is not int or replacement_index < 0
        ):
            raise ValueError(f"audit cards[{index}] has invalid replacement_answer_index")
        output.append(
            {
                "group_alias": alias,
                "missing_detail": detail.strip(),
                "evidence_aliases": tuple(evidence_alias_list),
                "importance": importance,
                "omission_type": omission_type,
                "rationale": rationale.strip(),
                "replacement_answer_index": replacement_index,
            }
        )
    return tuple(output)


def merge_audit_cards(
    topic: GenerationTopic,
    cards_by_group: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[dict[str, Any], ...]:
    """Deduplicate, rank, and cap validated audit cards deterministically."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    group_order = {group.group_id: index for index, group in enumerate(topic.groups)}
    evidence_order = {
        evidence.evidence_id: index for index, evidence in enumerate(topic.evidence)
    }
    group_aliases, evidence_aliases, _evidence_groups = _audit_aliases(topic)
    del evidence_aliases
    if any(group_id not in group_order for group_id in cards_by_group):
        raise ValueError("audit cards contain an unknown group")
    retained: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    seen_details: set[str] = set()
    ordinal = 0
    for group in topic.groups:
        for card in cards_by_group.get(group.group_id, ()):
            if not isinstance(card, dict):
                raise ValueError("audit cards must be objects")
            if card.get("group_alias") != group_aliases[group.group_id]:
                raise ValueError("audit card group alias does not match its group")
            detail_key = _normalized_audit_detail(str(card.get("missing_detail", "")))
            if not detail_key or detail_key in seen_details:
                ordinal += 1
                continue
            seen_details.add(detail_key)
            aliases = card.get("evidence_aliases")
            evidence_ids = [
                evidence_id
                for evidence_id, alias in _audit_aliases(topic)[1].items()
                if alias in aliases
            ]
            first_evidence = min(
                (evidence_order[evidence_id] for evidence_id in evidence_ids),
                default=len(evidence_order),
            )
            key = (
                -_AUDIT_IMPORTANCE.get(str(card.get("importance")), 0),
                -_AUDIT_SPECIFICITY.get(str(card.get("omission_type")), 0),
                group_order[group.group_id],
                first_evidence,
                ordinal,
            )
            retained.append((key, dict(card)))
            ordinal += 1
    retained.sort(key=lambda item: item[0])
    return tuple(
        {
            **card,
            "card_id": f"a{index:03d}",
        }
        for index, (_key, card) in enumerate(retained[:MAX_MERGED_AUDIT_CARDS], start=1)
    )


def _audit_card_docids(
    topic: GenerationTopic,
    audit_cards: tuple[dict[str, Any], ...],
) -> dict[str, tuple[str, ...]]:
    """Bind each merged audit card to only its authenticated evidence docids."""

    _group_aliases, evidence_aliases, _evidence_groups = _audit_aliases(topic)
    docid_by_alias = {
        evidence_aliases[evidence.evidence_id]: evidence.docid
        for evidence in topic.evidence
    }
    routed: dict[str, tuple[str, ...]] = {}
    for index, card in enumerate(audit_cards):
        card_id = card.get("card_id")
        if not isinstance(card_id, str) or not card_id or card_id in routed:
            raise ValueError(f"audit cards[{index}] has invalid card_id")
        aliases = card.get("evidence_aliases")
        if (
            not isinstance(aliases, (list, tuple))
            or not aliases
            or any(not isinstance(alias, str) or alias not in docid_by_alias for alias in aliases)
        ):
            raise ValueError(f"audit cards[{index}] has invalid evidence_aliases")
        routed[card_id] = tuple(dict.fromkeys(docid_by_alias[alias] for alias in aliases))
    return routed


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


def _bounded_private_root(config: Any, topic: GenerationTopic) -> Path:
    return config.resolved_work_dir / "bounded_revision" / _safe_topic_name(topic.topic_id)


def _bounded_identity(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
) -> dict[str, Any]:
    return {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "handoff_schema_version": handoff.schema_version,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "topic_id": topic.topic_id,
        "topic_context_sha256": topic.context_sha256,
        "config_run_id": config.run_id,
        "team_id": config.team_id,
        "run_desc": config.run_desc,
        "provider": config.provider,
        "api_base": config.api_base,
        "api_key_env": config.api_key_env,
        "sol_model": config.model,
        "sol_reasoning_effort": config.reasoning_effort,
        "structured_output": config.structured_output,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "transport_max_attempts": config.transport_max_attempts,
        "luna_model": LUNA_MODEL,
        "luna_reasoning_effort": LUNA_REASONING_EFFORT,
        "planner_prompt_sha256": _digest_text(render_planner_prompt(topic)),
        "writer_system_prompt_sha256": _digest_text(SYSTEM_PROMPT),
        "audit_system_prompt_sha256": _digest_text(
            "You are an evidence-only omission auditor. Return only the requested JSON."
        ),
        "audit_schema_sha256": _digest_json(audit_response_schema()),
        "writer_schema_sha256": _digest_json(output_schema()),
        "splice_schema_sha256": _digest_json(splice_response_schema()),
        "splice_prompt_contract_version": SPLICE_PROMPT_CONTRACT_VERSION,
        "splice_prompt_contract_sha256": _digest_text(SPLICE_PROMPT_CONTRACT_VERSION),
        "operation_screen_system_prompt_sha256": _digest_text(
            LUNA_OPERATION_SCREEN_SYSTEM_PROMPT
        ),
    }


def _bounded_write_state(root: Path, state: dict[str, Any]) -> None:
    _atomic_write_text(
        root / "state.json",
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _bounded_read_state(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bounded-revision state under {root}") from exc
    if not isinstance(payload, dict) or payload.get("trial_contract_version") != TRIAL_CONTRACT_VERSION:
        raise ValueError("bounded-revision state has an unsupported trial contract")
    return payload


def _bounded_register_file(root: Path, state: dict[str, Any], path: Path) -> None:
    relative = str(path.relative_to(root))
    state.setdefault("stage_hashes", {})[relative] = _digest_bytes(path.read_bytes())


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _bounded_revalidate_files(root: Path, state: dict[str, Any]) -> None:
    for relative, expected in state.get("stage_hashes", {}).items():
        path = root / relative
        if not path.is_file() or _digest_bytes(path.read_bytes()) != expected:
            raise ValueError(f"bounded-revision stage hash mismatch: {relative}")


def _bounded_recover_pending(root: Path, state: dict[str, Any]) -> None:
    changed = False
    state.setdefault("recovered_payloads", {})
    for reservation in [
        *state.get("luna_reservations", []),
        *state.get("sol_reservations", []),
    ]:
        receipt_path = root / str(reservation.get("receipt", ""))
        receipt: dict[str, Any] = {}
        if receipt_path.is_file():
            try:
                loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    receipt = loaded
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        stage = str(reservation.get("stage") or reservation.get("role") or "")
        if (
            receipt.get("outcome") == "semantic_success"
            and isinstance(receipt.get("accepted_payload"), dict)
        ):
            state["recovered_payloads"].setdefault(stage, receipt["accepted_payload"])
        if reservation.get("status") != "pending":
            continue
        outcome = receipt.get("outcome")
        if outcome == "terminal_transport_failure":
            reservation["status"] = "terminal_transport_failure"
        elif outcome == "semantic_success" and isinstance(
            receipt.get("accepted_payload"), dict
        ):
            reservation["status"] = "semantic_returned"
            state["recovered_payloads"][stage] = receipt["accepted_payload"]
        elif outcome == "semantic_error":
            reservation["status"] = "semantic_error"
        elif outcome == "semantic_success":
            reservation["status"] = "semantic_error"
            reservation["status"] = "semantic_returned"
        else:
            reservation["status"] = "crash_consumed"
        if receipt:
            ordinal = reservation.get("ordinal")
            if not any(
                call.get("stage") == stage
                and call.get("reservation_ordinal") == ordinal
                for call in state.get("calls", [])
            ):
                state.setdefault("calls", []).append(
                    {
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
                        if key in receipt
                    }
                )
        changed = True
    if changed:
        _bounded_write_state(root, state)


def _bounded_semantic_reservations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        reservation
        for reservation in state.get("sol_reservations", [])
        if reservation.get("status") != "terminal_transport_failure"
    ]


def _bounded_reserve_sol(
    root: Path,
    state: dict[str, Any],
    *,
    role: str,
    receipt_name: str,
    has_validation_errors: bool = False,
) -> int:
    if role not in {"draft", "revision", "repair"}:
        raise ValueError(f"unsupported Sol reservation role: {role}")
    reservations = _bounded_semantic_reservations(state)
    if len(reservations) >= MAX_SOL_RESERVATIONS:
        raise ValueError("bounded-revision Sol reservation ceiling is exhausted")
    if role == "repair" and not has_validation_errors:
        raise ValueError("repair reservation requires deterministic validation errors")
    if role != "repair" and len([item for item in reservations if item["role"] != "repair"]) >= 2:
        raise ValueError("bounded-revision routine Sol reservation ceiling is exhausted")
    if any(
        item["role"] == role
        and item.get("status") != "terminal_transport_failure"
        for item in state.get("sol_reservations", [])
    ):
        raise ValueError(f"bounded-revision role already reserved: {role}")
    ordinal = max(
        (int(item.get("ordinal", 0)) for item in state.get("sol_reservations", [])),
        default=0,
    ) + 1
    state.setdefault("sol_reservations", []).append(
        {
            "ordinal": ordinal,
            "role": role,
            "stage": receipt_name,
            "status": "pending",
            "receipt": f"receipts/{receipt_name}.json",
        }
    )
    _bounded_write_state(root, state)
    return ordinal


def _bounded_reserve_luna(
    root: Path,
    state: dict[str, Any],
    *,
    stage: str,
) -> int:
    prior = [item for item in state.get("luna_reservations", []) if item.get("stage") == stage]
    if any(item.get("status") in {"crash_consumed", "semantic_returned"} for item in prior):
        raise ValueError(f"bounded-revision Luna stage is already consumed: {stage}")
    ordinal = max(
        (int(item.get("ordinal", 0)) for item in state.get("luna_reservations", [])),
        default=0,
    ) + 1
    state.setdefault("luna_reservations", []).append(
        {
            "ordinal": ordinal,
            "stage": stage,
            "status": "pending",
            "receipt": f"receipts/{stage}.json",
        }
    )
    _bounded_write_state(root, state)
    return ordinal


def _bounded_finish_luna(
    root: Path,
    state: dict[str, Any],
    *,
    ordinal: int,
    status: str,
) -> None:
    for reservation in state.get("luna_reservations", []):
        if reservation.get("ordinal") == ordinal:
            reservation["status"] = status
            break
    else:
        raise ValueError(f"unknown Luna reservation ordinal: {ordinal}")
    _bounded_write_state(root, state)


def _bounded_recovered_payload(
    state: dict[str, Any],
    stage: str,
    *,
    expected_prompt_sha256: str | None = None,
    expected_schema_sha256: str | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    payloads = state.setdefault("recovered_payloads", {})
    if stage not in payloads:
        return False, None
    if expected_prompt_sha256 is not None or expected_schema_sha256 is not None:
        calls = [call for call in state.get("calls", []) if call.get("stage") == stage]
        recorded = calls[-1] if calls else {}
        if (
            expected_prompt_sha256 is not None
            and recorded.get("prompt_sha256") != expected_prompt_sha256
        ) or (
            expected_schema_sha256 is not None
            and recorded.get("schema_sha256") != expected_schema_sha256
        ):
            raise ValueError(f"{stage} recovery hash mismatch")
    payload = payloads.pop(stage)
    if not isinstance(payload, dict):
        raise ValueError(f"recovered {stage} payload is not an object")
    return True, payload


def _bounded_blocked_stage(
    state: dict[str, Any], *, stage: str, luna: bool
) -> bool:
    reservations = state.get("luna_reservations" if luna else "sol_reservations", [])
    return any(
        item.get("stage", item.get("role")) == stage
        and item.get("status")
        in {
            "crash_consumed",
            "ambiguous_failure",
            "semantic_error",
            "semantic_returned",
        }
        for item in reservations
    )


def _bounded_stage_status(
    state: dict[str, Any], *, stage: str, luna: bool
) -> str | None:
    reservations = state.get("luna_reservations" if luna else "sol_reservations", [])
    statuses = [
        str(item.get("status"))
        for item in reservations
        if item.get("stage", item.get("role")) == stage
    ]
    return statuses[-1] if statuses else None


def _bounded_finish_sol(
    root: Path,
    state: dict[str, Any],
    *,
    ordinal: int,
    status: str,
) -> None:
    for reservation in state.get("sol_reservations", []):
        if reservation.get("ordinal") == ordinal:
            reservation["status"] = status
            break
    else:
        raise ValueError(f"unknown Sol reservation ordinal: {ordinal}")
    _bounded_write_state(root, state)


def _bounded_usage(raw: object) -> tuple[dict[str, Any], float]:
    if not isinstance(raw, dict):
        return {}, 0.0
    usage = raw.get("usage")
    usage_value = dict(usage) if isinstance(usage, dict) else {}
    cost_value = usage_value.get("cost", raw.get("cost", 0.0))
    try:
        cost = float(cost_value) if cost_value is not None else 0.0
    except (TypeError, ValueError):
        cost = 0.0
    return usage_value, cost


async def _bounded_provider_call(
    generator: OpenRouterJsonGenerator,
    *,
    root: Path,
    api_key: str,
    topic_id: str,
    stage: str,
    model: str,
    reasoning_effort: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    executor: ThreadPoolExecutor,
    reservation_ordinal: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.monotonic()
    raw_response: object | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None
    outcome = "semantic_success"
    transport_outcome = "response"
    try:
        payload, raw_response = await _complete(
            generator,
            topic_id=topic_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            executor=executor,
        )
    except SemanticCompletionError as exc:
        raw_response = exc.raw_response
        error = f"{type(exc).__name__}: {exc}"
        outcome = "semantic_error"
    except RuntimeError as exc:
        error = f"{type(exc).__name__}: {exc}"
        if (
            isinstance(generator, OpenRouterJsonGenerator)
            and str(exc) == "OpenRouter generation transport failed"
        ):
            outcome = "terminal_transport_failure"
            transport_outcome = "failure"
        else:
            outcome = "ambiguous_failure"
            transport_outcome = "unknown"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outcome = "ambiguous_failure"
        transport_outcome = "unknown"
    usage, cost = _bounded_usage(raw_response)
    receipt = {
        "stage": stage,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_sha256": _digest_text(user_prompt),
        "system_prompt_sha256": _digest_text(system_prompt),
        "schema_sha256": _digest_json(response_schema),
        "reservation_ordinal": reservation_ordinal,
        "latency_seconds": round(time.monotonic() - started, 6),
        "usage": usage,
        "provider_cost": cost,
        "outcome": outcome,
        "transport_outcome": transport_outcome,
        "error": error,
        "accepted_payload": payload if outcome == "semantic_success" else None,
        "raw_response": raw_response,
    }
    _write_json(root / "receipts" / f"{stage}.json", receipt, api_key=api_key)
    return payload, {
        "stage": stage,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_sha256": receipt["prompt_sha256"],
        "schema_sha256": receipt["schema_sha256"],
        "reservation_ordinal": reservation_ordinal,
        "outcome": outcome,
        "latency_seconds": receipt["latency_seconds"],
        "provider_cost": cost,
        "usage": usage,
        "transport_outcome": transport_outcome,
        "error": error,
    }


def _bounded_candidate(
    generated: dict[str, Any],
    *,
    topic: GenerationTopic,
    config: Any,
    run_id: str,
) -> dict[str, Any]:
    record = normalize_generated_record(
        build_submission_record(
            generated,
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            team_id=config.team_id,
            run_id=run_id,
            run_desc=config.run_desc,
        ),
        allowed_docids=list(topic.citation_docids),
    )
    _validate_generated_submission_record(
        record,
        topic_id=topic.topic_id,
        narrative=topic.narrative,
        allowed_docids=list(topic.citation_docids),
        team_id=config.team_id,
        run_id=run_id,
        run_desc=config.run_desc,
    )
    _validate_exact_hint_citations(record, topic=topic)
    return record


def _bounded_rebind_candidate(
    record: dict[str, Any],
    *,
    topic: GenerationTopic,
    config: Any,
    run_id: str,
) -> dict[str, Any]:
    rebound = json.loads(json.dumps(record, ensure_ascii=False))
    rebound["metadata"]["run_id"] = run_id
    rebound = normalize_generated_record(rebound)
    _validate_generated_submission_record(
        rebound,
        topic_id=topic.topic_id,
        narrative=topic.narrative,
        allowed_docids=list(topic.citation_docids),
        team_id=config.team_id,
        run_id=run_id,
        run_desc=config.run_desc,
    )
    _validate_exact_hint_citations(rebound, topic=topic)
    return rebound


def _bounded_finalize_operation_screen(
    topic: GenerationTopic,
    draft: dict[str, Any],
    operations: tuple[SpliceOperation, ...],
    payload: object,
    *,
    config: Any,
    run_id: str,
) -> tuple[dict[str, Any], OperationScreenResult | None, bool, str | None]:
    """Apply the accepted subset, falling back atomically to the validated draft."""

    try:
        result = validate_operation_screen_payload(payload, operations)
        assembled = (
            apply_splice_operations(draft, result.accepted_operations)
            if result.accepted_operations
            else draft
        )
        final = _bounded_rebind_candidate(
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
        fallback = _bounded_rebind_candidate(
            draft,
            topic=topic,
            config=config,
            run_id=run_id,
        )
        return fallback, None, True, f"{type(exc).__name__}: {exc}"


def _bounded_write_candidate(root: Path, arm: str, record: dict[str, Any]) -> Path:
    path = root / "evaluation" / arm / "submission.jsonl"
    _atomic_write_text(path, json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _bounded_write_identity(
    root: Path,
    *,
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    arm: str,
) -> Path:
    run_id = f"{config.run_id}-{arm}"
    identity = _generation_identity(replace(config, run_id=run_id), handoff, (topic,))
    path = root / "evaluation" / arm / "generation_identity.json"
    _atomic_write_text(path, json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def _bounded_candidate_summary(root: Path, arm: str, record: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer", [])
    words = sum(len(item.get("text", "").split()) for item in answer if isinstance(item, dict))
    path = root / "evaluation" / arm / "submission.jsonl"
    return {
        "words": words,
        "answer_objects": len(answer) if isinstance(answer, list) else 0,
        "output_sha256": _digest_bytes(path.read_bytes()),
    }


def _bounded_draft_objects_prompt(draft: dict[str, Any]) -> list[str]:
    """Render the validated draft with immutable zero-based answer indexes."""

    lines = [
        "VALIDATED DRAFT REFERENCES (existing numeric positions are immutable):",
        json.dumps(draft["references"], ensure_ascii=False, sort_keys=True),
        "DRAFT ANSWER OBJECTS (immutable; every index refers to this original array):",
    ]
    lines.extend(
        f"[{index}] {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
        for index, item in enumerate(draft["answer"])
    )
    return lines


def _bounded_revision_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    *,
    draft: dict[str, Any],
    audit_cards: tuple[dict[str, Any], ...],
) -> str:
    context = _render_hybrid_common_context(topic, blueprint, projection)
    lines = [
        "BOUNDED POST-DRAFT SPLICE REVISION",
        "Return exactly one JSON object with `decision` and `operations`; never return a full answer.",
        "Use `keep_draft` with an empty operations array when no bounded edit would improve the",
        "complete narrative. For `edit`, describe atomic operations against the immutable original",
        "draft indexes above. `delete_count: 0` inserts before an index (including append at the",
        "original answer length); `delete_count: 1`, `2`, or `3` replaces that contiguous range",
        "with one new object. Pure deletion is not supported and every operation needs one or more",
        "unique merged audit-card IDs.",
        "Hard splice budgets: at most 6 operations, 4 insertions, 8 touched original objects,",
        "and 1,024 whitespace-separated assembled answer words. Audit cards are advisory candidates,",
        "not requirements. Prefer merge or replacement over unnecessary insertion, preserve caveats",
        "and balance, retain causal qualifications and planner `must` obligations, and do not create",
        "a citation-by-citation inventory. Every new object must be one self-contained sentence stating one atomic claim.",
        "Prefer a single strongest citation; use a second only when it",
        "independently supports the complete object. Use at most two unique raw docids, and every",
        "citation must come from evidence linked to that operation's named audit cards as well as",
        "the full authenticated citation domain above.",
        *_bounded_draft_objects_prompt(draft),
        "MERGED AUDIT CARDS (ADVISORY; stable IDs are authenticated allowlist values):",
    ]
    lines.extend(
        f"[{card['card_id']}] {json.dumps(card, ensure_ascii=False, sort_keys=True)}"
        for card in audit_cards
    )
    return context + "\n\n" + "\n".join(lines)


def _bounded_operation_payload(
    operation_id: str,
    operation: SpliceOperation,
) -> dict[str, Any]:
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


def _bounded_operation_screen_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    *,
    draft: dict[str, Any],
    audit_cards: tuple[dict[str, Any], ...],
    operations: tuple[SpliceOperation, ...],
) -> str:
    """Render one whole-answer, decisions-only screen over frozen operations."""

    context = _render_hybrid_common_context(topic, blueprint, projection)
    cards_by_id = {str(card.get("card_id")): card for card in audit_cards}
    selected_card_ids = tuple(
        dict.fromkeys(
            card_id
            for operation in operations
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
        "Choose one coherent candidate subset among the otherwise supported, atomic, and material",
        "operations. Judge `nonredundant` and `replacement_safe` in the draft that would result",
        "from applying that subset, not against the unchanged draft one operation at a time. For",
        "example, another retained operation may restore a distinct detail removed by a replacement;",
        "the replacement may then be safe, and the restoring insertion is not redundant.",
        "Local code accepts an operation only when all five gates are true. Do not optimize the",
        "number accepted and do not infer an expected decision from the topic or operation ID.",
        *_bounded_draft_objects_prompt(draft),
        "FROZEN VALIDATED SOL OPERATIONS (stable IDs; immutable):",
    ]
    for operation_id, operation in zip(
        operation_ids(operations), operations, strict=True
    ):
        lines.append(
            f"[{operation_id}] "
            + json.dumps(
                _bounded_operation_payload(operation_id, operation),
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


def _bounded_splice_repair_prompt(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    *,
    draft: dict[str, Any],
    audit_cards: tuple[dict[str, Any], ...],
    initial_payload: dict[str, Any],
    validation_errors: tuple[str, ...],
) -> str:
    """Render a splice-only repair prompt authorized by the initial response."""

    context = _render_hybrid_common_context(topic, blueprint, projection)
    lines = [
        "DETERMINISTIC SPLICE VALIDATION REPAIR",
        "Return exactly one response matching the splice schema: a `keep_draft` wrapper with no",
        "operations, or an `edit` wrapper containing a corrected subset/reordering of the initial",
        "operations. You may correct only wrapper fields, indexes, ranges, budgets, or audit-card",
        "references. Every repaired `new_object` must exactly equal a `new_object` from the initial",
        "splice response; do not introduce or rewrite answer prose or citations.",
        "Retain an operation only when its object is one self-contained sentence stating one atomic",
        "claim, has at most two citations (prefer the single strongest), and every citation is linked",
        "to that operation's named audit cards. Otherwise omit it or return `keep_draft`.",
        *_bounded_draft_objects_prompt(draft),
        "MERGED AUDIT CARDS (ADVISORY; stable IDs are authenticated allowlist values):",
    ]
    lines.extend(
        f"[{card['card_id']}] {json.dumps(card, ensure_ascii=False, sort_keys=True)}"
        for card in audit_cards
    )
    lines.extend(
        [
            "INITIAL SPLICE RESPONSE:",
            json.dumps(initial_payload, ensure_ascii=False, sort_keys=True),
            "EXACT LOCAL VALIDATOR ERRORS:",
            *validation_errors,
        ]
    )
    return context + "\n\n" + "\n".join(lines)


def _bounded_manifest(
    root: Path,
    state: dict[str, Any],
    *,
    topic: GenerationTopic,
    draft: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> dict[str, Any]:
    reservations = state.get("sol_reservations", [])
    by_role = {role: 0 for role in ("draft", "revision", "repair")}
    for reservation in reservations:
        by_role[reservation["role"]] += 1
    transports: dict[str, int] = {}
    total_cost = 0.0
    for call in state.get("calls", []):
        outcome = str(call.get("transport_outcome", "unknown"))
        transports[outcome] = transports.get(outcome, 0) + 1
        total_cost += float(call.get("provider_cost", 0.0) or 0.0)
    manifest: dict[str, Any] = {
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "topic_id": topic.topic_id,
        "handoff_manifest_sha256": state["identity"]["handoff_manifest_sha256"],
        "topic_context_sha256": state["identity"]["topic_context_sha256"],
        "stage_completion": state.get("stages", {}),
        "luna_call_count": sum(1 for call in state.get("calls", []) if call["model"] == LUNA_MODEL),
        "sol_reservations_by_role": by_role,
        "sol_reservation_count": len(reservations),
        "http_transport_outcomes": transports,
        "total_provider_reported_cost": round(total_cost, 12),
        "draft": _bounded_candidate_summary(root, "draft", draft) if draft is not None else None,
        "final": _bounded_candidate_summary(root, "final", final) if final is not None else None,
        "operation_screen": state.get("operation_screen"),
        "failure": state.get("failure"),
    }
    return manifest


def _bounded_record_draft_failure(
    root: Path,
    state: dict[str, Any],
    topic: GenerationTopic,
    *,
    reason: str,
) -> None:
    """Persist a sanitized draft failure without creating downstream stages."""

    state["failure"] = reason
    _bounded_write_state(root, state)
    manifest = _bounded_manifest(root, state, topic=topic, draft=None, final=None)
    _atomic_write_text(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")


async def _run_bounded_revision(
    config: Any,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
    *,
    api_key: str,
    state_mode: str,
) -> Path:
    _validate_artifact_paths(config)
    root = _bounded_private_root(config, topic)
    if state_mode == "create":
        if root.exists():
            raise ValueError(f"bounded-revision state already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        (root / "receipts").mkdir()
        state: dict[str, Any] = {
            "trial_contract_version": TRIAL_CONTRACT_VERSION,
            "identity": _bounded_identity(config, handoff, topic),
            "stages": {
                "planner": False,
                "draft": False,
                "audit_groups": [],
                "audit_merge": False,
                "revision": False,
                "operation_screen": False,
                "final": False,
            },
            "stage_hashes": {},
            "luna_reservations": [],
            "sol_reservations": [],
            "recovered_payloads": {},
            "calls": [],
            "failure": None,
        }
        _bounded_write_state(root, state)
    elif state_mode == "resume":
        if not root.is_dir():
            raise ValueError(f"bounded-revision state does not exist: {root}")
        state = _bounded_read_state(root)
        if state.get("identity") != _bounded_identity(config, handoff, topic):
            raise ValueError("bounded-revision resume identity differs from create identity")
        _bounded_revalidate_files(root, state)
        _bounded_recover_pending(root, state)
        if state.get("stages", {}).get("final") and (root / "manifest.json").is_file():
            return root
    else:
        raise ValueError("state_mode must be create or resume")

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
    sol = OpenRouterJsonGenerator(
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

    blueprint: NarrativeBlueprint
    projection: BlueprintProjection
    planner_state_path = root / "blueprint.state.json"
    planner_prompt = render_planner_prompt(topic)
    planner_schema = planner_response_schema()
    if state["stages"].get("planner"):
        loaded = load_blueprint_state(
            topic,
            json.loads(planner_state_path.read_text(encoding="utf-8")),
            planner_prompt_sha256=_digest_text(planner_prompt),
        )
        blueprint, projection = loaded
    else:
        recovered, planner_payload = _bounded_recovered_payload(state, "planner")
        if recovered:
            planner_call = None
        else:
            if _bounded_blocked_stage(state, stage="planner", luna=True):
                raise RuntimeError("planner reservation was consumed without a reusable payload")
            planner_ordinal = _bounded_reserve_luna(root, state, stage="planner")
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bounded-luna") as executor:
                planner_payload, planner_call = await _bounded_provider_call(
                    luna,
                    root=root,
                    api_key=api_key,
                    topic_id=topic.topic_id,
                    stage="planner",
                    model=LUNA_MODEL,
                    reasoning_effort=LUNA_REASONING_EFFORT,
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    user_prompt=planner_prompt,
                    response_schema=planner_schema,
                    executor=executor,
                    reservation_ordinal=planner_ordinal,
                )
            state["calls"].append(planner_call)
            state["luna_call_count"] = state.get("luna_call_count", 0) + 1
            _bounded_finish_luna(
                root,
                state,
                ordinal=planner_ordinal,
                status=(
                    "semantic_returned"
                    if planner_payload is not None
                    else planner_call["outcome"]
                ),
            )
        if planner_payload is None:
            state["failure"] = "planner did not return an accepted semantic response"
            _bounded_write_state(root, state)
            _atomic_write_text(
                root / "manifest.json",
                json.dumps(_bounded_manifest(root, state, topic=topic, draft=None, final=None), indent=2) + "\n",
            )
            raise RuntimeError(state["failure"])
        blueprint = validate_blueprint(topic, planner_payload)
        projection = project_blueprint(topic, blueprint)
        writer_context = render_blueprint_writer_context(topic, blueprint, projection)
        planner_state = serialize_blueprint_state(
            topic,
            blueprint,
            projection,
            planner_prompt_sha256=_digest_text(planner_prompt),
            writer_context_sha256=_digest_text(writer_context),
        )
        _write_json(planner_state_path, planner_state, api_key=api_key)
        _bounded_register_file(root, state, planner_state_path)
        state["stages"]["planner"] = True
        _bounded_write_state(root, state)

    draft: dict[str, Any] | None = None
    draft_path = root / "draft.record.json"
    draft_errors: tuple[str, ...] = ()
    if state["stages"].get("draft") and draft_path.is_file():
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    else:
        recovered, generated = _bounded_recovered_payload(state, "draft")
        if not recovered:
            if _bounded_blocked_stage(state, stage="draft", luna=False):
                raise RuntimeError("draft reservation was consumed without a reusable payload")
            draft_ordinal = _bounded_reserve_sol(root, state, role="draft", receipt_name="draft")
            draft_prompt = _render_hybrid_writer_prompt(topic, blueprint, projection)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bounded-sol") as executor:
                generated, draft_call = await _bounded_provider_call(
                    sol,
                    root=root,
                    api_key=api_key,
                    topic_id=topic.topic_id,
                    stage="draft",
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=draft_prompt,
                    response_schema=output_schema(),
                    executor=executor,
                    reservation_ordinal=draft_ordinal,
                )
            state["calls"].append(draft_call)
            _bounded_finish_sol(
                root,
                state,
                ordinal=draft_ordinal,
                status=(
                    "semantic_returned" if generated is not None else draft_call["outcome"]
                ),
            )
        if generated is not None:
            try:
                draft = _bounded_candidate(
                    generated, topic=topic, config=config, run_id=f"{config.run_id}-draft"
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                draft_errors = (f"{type(exc).__name__}: {exc}",)
        if draft is not None:
            _atomic_write_text(
                draft_path,
                json.dumps(draft, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            _bounded_register_file(root, state, draft_path)
            draft_output = _bounded_write_candidate(root, "draft", draft)
            identity_path = _bounded_write_identity(
                root, config=config, handoff=handoff, topic=topic, arm="draft"
            )
            _bounded_register_file(root, state, draft_output)
            _bounded_register_file(root, state, identity_path)
            state["stages"]["draft"] = True
        _bounded_write_state(root, state)

    if draft is None:
        failure = (
            "draft candidate failed deterministic validation"
            if draft_errors
            else "draft did not return an accepted semantic response"
        )
        _bounded_record_draft_failure(root, state, topic, reason=failure)
        raise RuntimeError(failure)

    cards_by_group: dict[str, tuple[dict[str, Any], ...]] = {}
    for group in topic.groups:
        if group.group_id in state["stages"].get("audit_groups", []):
            cards_path = root / "audit.cards.json"
            all_cards = json.loads(cards_path.read_text(encoding="utf-8"))
            cards_by_group[group.group_id] = tuple(all_cards.get(group.group_id, ()))
            continue
        audit_stage = f"audit-{_audit_aliases(topic)[0][group.group_id]}"
        recovered, audit_payload = _bounded_recovered_payload(state, audit_stage)
        if not recovered:
            if _bounded_blocked_stage(state, stage=audit_stage, luna=True):
                raise RuntimeError(
                    f"{audit_stage} reservation was consumed without a reusable payload"
                )
            audit_ordinal = _bounded_reserve_luna(root, state, stage=audit_stage)
            audit_prompt = render_group_audit_prompt(
                topic, group_id=group.group_id, draft=draft
            )
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bounded-luna") as executor:
                audit_payload, audit_call = await _bounded_provider_call(
                    luna,
                    root=root,
                    api_key=api_key,
                    topic_id=topic.topic_id,
                    stage=audit_stage,
                    model=LUNA_MODEL,
                    reasoning_effort=LUNA_REASONING_EFFORT,
                    system_prompt="You are an evidence-only omission auditor. Return only the requested JSON.",
                    user_prompt=audit_prompt,
                    response_schema=audit_response_schema(),
                    executor=executor,
                    reservation_ordinal=audit_ordinal,
                )
            state["calls"].append(audit_call)
            _bounded_finish_luna(
                root,
                state,
                ordinal=audit_ordinal,
                status=(
                    "semantic_returned" if audit_payload is not None else audit_call["outcome"]
                ),
            )
            if audit_call["outcome"] == "terminal_transport_failure":
                raise RuntimeError(f"{audit_stage} transport failed; resume may retry it")
            if audit_call["outcome"] == "semantic_error":
                raise RuntimeError(
                    f"{audit_stage} semantic response failed; resume is fail-closed"
                )
        try:
            cards_by_group[group.group_id] = (
                validate_group_audit(topic, group_id=group.group_id, payload=audit_payload)
                if audit_payload is not None
                else ()
            )
        except ValueError:
            cards_by_group[group.group_id] = ()
        state["stages"].setdefault("audit_groups", []).append(group.group_id)
        _atomic_write_text(
            root / "audit.cards.json",
            json.dumps(
                {key: list(value) for key, value in cards_by_group.items()},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _bounded_register_file(root, state, root / "audit.cards.json")
        _bounded_write_state(root, state)

    audit_cards = merge_audit_cards(topic, cards_by_group)
    audit_card_docids = _audit_card_docids(topic, audit_cards)
    merged_path = root / "audit.merged.json"
    _atomic_write_text(merged_path, json.dumps(list(audit_cards), indent=2) + "\n")
    _bounded_register_file(root, state, merged_path)
    state["stages"]["audit_merge"] = True
    _bounded_write_state(root, state)

    final: dict[str, Any] | None = None
    operations_to_screen: tuple[SpliceOperation, ...] | None = None
    revision_errors: tuple[str, ...] = ()
    initial_splice_payload: dict[str, Any] | None = None
    revision_prompt = _bounded_revision_prompt(
        topic, blueprint, projection, draft=draft, audit_cards=audit_cards
    )
    revision_schema = splice_response_schema()
    revision_prompt_sha256 = _digest_text(revision_prompt)
    revision_schema_sha256 = _digest_json(revision_schema)
    recovered, splice_payload = _bounded_recovered_payload(
        state,
        "revision",
        expected_prompt_sha256=revision_prompt_sha256,
        expected_schema_sha256=revision_schema_sha256,
    )
    if not recovered:
        revision_status = _bounded_stage_status(state, stage="revision", luna=False)
        if revision_status in {"crash_consumed", "ambiguous_failure", "semantic_error"}:
            # A provider/schema failure leaves no authenticated operation set to repair.
            splice_payload = None
        elif revision_status == "semantic_returned":
            splice_payload = None
        else:
            revision_ordinal = _bounded_reserve_sol(
                root, state, role="revision", receipt_name="revision"
            )
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bounded-sol") as executor:
                splice_payload, revision_call = await _bounded_provider_call(
                    sol,
                    root=root,
                    api_key=api_key,
                    topic_id=topic.topic_id,
                    stage="revision",
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=revision_prompt,
                    response_schema=revision_schema,
                    executor=executor,
                    reservation_ordinal=revision_ordinal,
                )
            state["calls"].append(revision_call)
            _bounded_finish_sol(
                root,
                state,
                ordinal=revision_ordinal,
                status=(
                    "semantic_returned"
                    if splice_payload is not None
                    else revision_call["outcome"]
                ),
            )
    if splice_payload is not None:
        initial_splice_payload = splice_payload
        try:
            operations = validate_splice_payload(
                draft,
                splice_payload,
                tuple(topic.citation_docids),
                audit_card_docids,
            )
            if operations is None:
                final = _bounded_rebind_candidate(
                    draft,
                    topic=topic,
                    config=config,
                    run_id=f"{config.run_id}-final",
                )
            else:
                assembled = apply_splice_operations(draft, operations)
                final = _bounded_rebind_candidate(
                    assembled,
                    topic=topic,
                    config=config,
                    run_id=f"{config.run_id}-final",
                )
                operations_to_screen = operations
        except (SpliceValidationError, ValueError, RuntimeError, TypeError) as exc:
            revision_errors = (f"{type(exc).__name__}: {exc}",)
    if revision_errors and initial_splice_payload is not None:
        recovered, repaired = _bounded_recovered_payload(state, "repair")
        if not recovered:
            if _bounded_blocked_stage(state, stage="repair", luna=False):
                # A failed repair has no further safe fallback beyond the validated draft.
                repaired = None
            else:
                repair_ordinal = _bounded_reserve_sol(
                    root,
                    state,
                    role="repair",
                    receipt_name="repair",
                    has_validation_errors=True,
                )
                repair_prompt = _bounded_splice_repair_prompt(
                    topic,
                    blueprint,
                    projection,
                    draft=draft,
                    audit_cards=audit_cards,
                    initial_payload=initial_splice_payload,
                    validation_errors=revision_errors,
                )
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bounded-sol") as executor:
                    repaired, repair_call = await _bounded_provider_call(
                        sol,
                        root=root,
                        api_key=api_key,
                        topic_id=topic.topic_id,
                        stage="repair",
                        model=config.model,
                        reasoning_effort=config.reasoning_effort,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=repair_prompt,
                        response_schema=revision_schema,
                        executor=executor,
                        reservation_ordinal=repair_ordinal,
                    )
                state["calls"].append(repair_call)
                _bounded_finish_sol(
                    root,
                    state,
                    ordinal=repair_ordinal,
                    status=(
                        "semantic_returned" if repaired is not None else repair_call["outcome"]
                    ),
                )
        if repaired is not None:
            try:
                operations = validate_repaired_splice_payload(
                    draft,
                    repaired,
                    initial_splice_payload,
                    tuple(topic.citation_docids),
                    audit_card_docids,
                )
                if operations is None:
                    final = _bounded_rebind_candidate(
                        draft,
                        topic=topic,
                        config=config,
                        run_id=f"{config.run_id}-final",
                    )
                else:
                    assembled = apply_splice_operations(draft, operations)
                    final = _bounded_rebind_candidate(
                        assembled,
                        topic=topic,
                        config=config,
                        run_id=f"{config.run_id}-final",
                    )
                    operations_to_screen = operations
            except (SpliceValidationError, ValueError, RuntimeError, TypeError):
                final = None
        if final is None:
            final = _bounded_rebind_candidate(
                draft,
                topic=topic,
                config=config,
                run_id=f"{config.run_id}-final",
            )
    elif final is None:
        final = _bounded_rebind_candidate(
            draft,
            topic=topic,
            config=config,
            run_id=f"{config.run_id}-final",
        )

    if operations_to_screen:
        screen_prompt = _bounded_operation_screen_prompt(
            topic,
            blueprint,
            projection,
            draft=draft,
            audit_cards=audit_cards,
            operations=operations_to_screen,
        )
        screen_schema = operation_screen_response_schema(len(operations_to_screen))
        screen_prompt_sha256 = _digest_text(screen_prompt)
        screen_schema_sha256 = _digest_json(screen_schema)
        recovered, screen_payload = _bounded_recovered_payload(
            state,
            "operation-screen",
            expected_prompt_sha256=screen_prompt_sha256,
            expected_schema_sha256=screen_schema_sha256,
        )
        screen_call: dict[str, Any] | None = None
        if not recovered:
            if _bounded_blocked_stage(state, stage="operation-screen", luna=True):
                screen_payload = None
            else:
                screen_ordinal = _bounded_reserve_luna(
                    root,
                    state,
                    stage="operation-screen",
                )
                with ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="bounded-luna",
                ) as executor:
                    screen_payload, screen_call = await _bounded_provider_call(
                        luna,
                        root=root,
                        api_key=api_key,
                        topic_id=topic.topic_id,
                        stage="operation-screen",
                        model=LUNA_MODEL,
                        reasoning_effort=LUNA_REASONING_EFFORT,
                        system_prompt=LUNA_OPERATION_SCREEN_SYSTEM_PROMPT,
                        user_prompt=screen_prompt,
                        response_schema=screen_schema,
                        executor=executor,
                        reservation_ordinal=screen_ordinal,
                    )
                state["calls"].append(screen_call)
                _bounded_finish_luna(
                    root,
                    state,
                    ordinal=screen_ordinal,
                    status=(
                        "semantic_returned"
                        if screen_payload is not None
                        else screen_call["outcome"]
                    ),
                )
                if screen_call["outcome"] == "terminal_transport_failure":
                    raise RuntimeError(
                        "operation-screen transport failed; resume may retry it"
                    )
        final, screen_result, used_fallback, _ = _bounded_finalize_operation_screen(
            topic,
            draft,
            operations_to_screen,
            screen_payload,
            config=config,
            run_id=f"{config.run_id}-final",
        )
        state["operation_screen"] = {
            "candidate_count": len(operations_to_screen),
            "accepted_count": (
                len(screen_result.accepted_operations)
                if screen_result is not None
                else 0
            ),
            "used_draft_fallback": used_fallback,
        }
    else:
        state["operation_screen"] = {
            "candidate_count": 0,
            "accepted_count": 0,
            "used_draft_fallback": False,
        }
    state["stages"]["operation_screen"] = True

    final_output = _bounded_write_candidate(root, "final", final)
    final_identity = _bounded_write_identity(
        root, config=config, handoff=handoff, topic=topic, arm="final"
    )
    _bounded_register_file(root, state, final_output)
    _bounded_register_file(root, state, final_identity)
    state["stages"]["revision"] = True
    state["stages"]["final"] = True
    _bounded_write_state(root, state)
    manifest = _bounded_manifest(root, state, topic=topic, draft=draft, final=final)
    _atomic_write_text(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return root


def _print_bounded_dry_run(config: Any, topic: GenerationTopic) -> None:
    print(f"topic={topic.topic_id}")
    print(
        "counts="
        f"groups:{len(topic.groups)},claims:{len(topic.claim_hints)},"
        f"evidence_passages:{len(topic.evidence)},citation_docids:{len(topic.citation_docids)}"
    )
    print(
        "calls="
        f"planner:1,audit:{len(topic.groups)},operation_screen_at_most:1,"
        f"luna_total_at_most:{2 + len(topic.groups)},provider:0"
    )
    print("sol_reservations=draft:1,revision:1,repair_only:1,total:3,max:3")
    print(
        "state="
        f"{_bounded_private_root(config, topic)}; no provider calls; private outputs only"
    )


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
        "--bounded-revision",
        action="store_true",
        help="Run the bounded post-draft omission-audit revision trial.",
    )
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
    parser.add_argument(
        "--state-mode",
        choices=("create", "resume"),
        default=None,
        help="Bounded-revision durable state mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    config = load_rag_generation_config(args.config)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topic = _topic_for_cli(config, handoff, args.topic)
    if args.bounded_revision:
        if args.state_mode is None:
            raise SystemExit("--bounded-revision requires --state-mode create|resume")
        if args.writer_attempts is not None:
            raise SystemExit("--bounded-revision does not accept --writer-attempts")
        if args.dry_run:
            _print_bounded_dry_run(config, topic)
            return
        repo_root = find_repo_root(args.config.resolve().parent)
        load_repo_env(repo_root)
        api_key = os.environ.get(config.api_key_env, "")
        root = asyncio.run(
            _run_bounded_revision(
                config,
                handoff,
                topic,
                api_key=api_key,
                state_mode=args.state_mode,
            )
        )
        print(f"completed bounded-revision topic={topic.topic_id} work={root}")
        return
    if args.state_mode is not None:
        raise SystemExit("--state-mode requires --bounded-revision")
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

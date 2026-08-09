"""Throwaway one-call Luna replay over an authenticated bounded-revision draft."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from trec_rag.bounded_splice import (
    SpliceValidationError,
    apply_splice_operations,
    splice_response_schema,
    validate_splice_payload,
)
from trec_rag.generation_handoff import GenerationTopic
from trec_rag.narrative_blueprint import BlueprintProjection, NarrativeBlueprint
from trec_rag.narrative_blueprint_trial import (
    _bounded_draft_objects_prompt,
    _render_hybrid_common_context,
)


LUNA_SPLICE_REPLAY_CONTRACT_VERSION = "luna_whole_answer_splice_replay_v1"
MAX_REPLAY_OPERATIONS = 3


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

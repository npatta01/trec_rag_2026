"""Validated, authenticated planning records for experimental RAG generation.

The planner sees only the official narrative, generated group text, and
advisory claim hints.  The writer receives a projection whose evidence and
citation domain are derived from the authenticated :class:`GenerationTopic`;
provider output never supplies an evidence or document identifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
import unicodedata

from .generation_handoff import GenerationTopic


BLUEPRINT_CONTRACT_VERSION = "narrative_blueprint_v1"
ANSWER_MODES = frozenset(
    {"describe", "explain", "compare", "evaluate", "recommend", "enumerate"}
)
PRIORITIES = frozenset({"must", "should", "could"})
MIN_ALLOCATED_WORDS = 850
MAX_ALLOCATED_WORDS = 950

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BlueprintValidationError(ValueError):
    """A provider blueprint or persisted blueprint state is invalid."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BlueprintValidationError(f"state is not canonical JSON: {exc}") from exc


def _digest(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _normalized_text(value: str) -> str:
    """Normalize text for narrative-span matching, without changing stored text."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BlueprintValidationError(f"{label} must be an object")
    return value


def _require_exact_mapping(
    value: object, expected: set[str], label: str
) -> Mapping[str, object]:
    row = _require_mapping(value, label)
    fields = set(row)
    if fields != expected:
        missing = sorted(expected - fields)
        unknown = sorted(fields - expected)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise BlueprintValidationError(f"{label} fields are invalid ({'; '.join(detail)})")
    return row


def _require_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlueprintValidationError(f"{label} must be non-empty text")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BlueprintValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _claim_alias(index: int) -> str:
    return f"c{index:03d}"


def _group_alias(index: int) -> str:
    return f"g{index:03d}"


def _evidence_alias(index: int) -> str:
    return f"e{index:03d}"


@dataclass(frozen=True)
class BlueprintObligation:
    """One provider-planned obligation, retaining only advisory input labels."""

    label: str
    narrative_spans: tuple[str, ...]
    priority: str
    answer_mode: str
    target_words: int
    selected_claim_aliases: tuple[str, ...]
    claim_ids: tuple[str, ...] = field(init=False)
    group_ids: tuple[str, ...] = field(init=False)

    def _with_claims(
        self, claim_ids: tuple[str, ...], group_ids: tuple[str, ...]
    ) -> BlueprintObligation:
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "group_ids", group_ids)
        return self

    def to_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "narrative_spans": list(self.narrative_spans),
            "priority": self.priority,
            "answer_mode": self.answer_mode,
            "target_words": self.target_words,
            "selected_claim_aliases": list(self.selected_claim_aliases),
        }


@dataclass(frozen=True)
class NarrativeBlueprint:
    """Validated planner result; all claim identity is resolved locally."""

    obligations: tuple[BlueprintObligation, ...]
    contract_version: str = field(init=False, default=BLUEPRINT_CONTRACT_VERSION)

    def to_payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "obligations": [obligation.to_payload() for obligation in self.obligations],
        }


@dataclass(frozen=True)
class BlueprintProjectionObligation:
    """A blueprint obligation with evidence aliases derived from the handoff."""

    label: str
    narrative_spans: tuple[str, ...]
    priority: str
    answer_mode: str
    target_words: int
    selected_claim_aliases: tuple[str, ...]
    claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_aliases: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "narrative_spans": list(self.narrative_spans),
            "priority": self.priority,
            "answer_mode": self.answer_mode,
            "target_words": self.target_words,
            "selected_claim_aliases": list(self.selected_claim_aliases),
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "evidence_aliases": list(self.evidence_aliases),
        }


@dataclass(frozen=True)
class BlueprintProjection:
    """Deterministic selected-evidence projection for the writer."""

    obligations: tuple[BlueprintProjectionObligation, ...]
    evidence_ids: tuple[str, ...]
    evidence_aliases: tuple[str, ...]
    citation_docids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "obligations": [obligation.to_payload() for obligation in self.obligations],
            "evidence_ids": list(self.evidence_ids),
            "evidence_aliases": list(self.evidence_aliases),
            "citation_docids": list(self.citation_docids),
        }


def planner_response_schema() -> dict[str, object]:
    """Return the strict JSON schema accepted from the one-shot planner."""

    obligation_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "minLength": 1},
            "narrative_spans": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 1},
            },
            "priority": {"type": "string", "enum": ["must", "should", "could"]},
            "answer_mode": {
                "type": "string",
                "enum": [
                    "describe",
                    "explain",
                    "compare",
                    "evaluate",
                    "recommend",
                    "enumerate",
                ],
            },
            "target_words": {"type": "integer", "minimum": 1},
            "selected_claim_aliases": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "pattern": r"^c[0-9]{3}$"},
            },
        },
        "required": [
            "label",
            "narrative_spans",
            "priority",
            "answer_mode",
            "target_words",
            "selected_claim_aliases",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "obligations": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": obligation_schema,
            }
        },
        "required": ["obligations"],
    }


def render_planner_prompt(topic: GenerationTopic) -> str:
    """Render compact narrative-first planner input without authority records."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    lines = [
        "NARRATIVE BLUEPRINT PLANNER",
        "Derive 3–8 answer obligations from the complete official narrative.",
        "Generated groups are retrieval structure, not independent output requirements.",
        "Claim hints are advisory routing aids; selected passages remain factual authority.",
        "Select complementary, nonredundant claim aliases that form a coverage checklist.",
        "Favor specific quantities, named mechanisms, actors, interventions, contrasts, and",
        "uncertainties over several aliases that merely restate one broad point.",
        "Use only the aliases below when selecting claims. Do not invent aliases.",
        "",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        "RETRIEVAL GROUPS AND ADVISORY CLAIM HINTS:",
    ]
    claim_aliases = {
        id(claim): _claim_alias(index)
        for index, claim in enumerate(topic.claim_hints, start=1)
    }
    group_aliases = {
        id(group): _group_alias(index) for index, group in enumerate(topic.groups, start=1)
    }
    for group in topic.groups:
        lines.append(f"{group_aliases[id(group)]}: {group.text}")
        for claim in topic.claim_hints:
            if claim.group_id == group.group_id:
                lines.append(f"  {claim_aliases[id(claim)]}: {claim.text}")
    lines.extend(
        [
            "",
            "Return JSON matching the supplied schema. The narrative_spans field is an",
            "authentication boundary: copy every span character-for-character from one",
            "contiguous substring of the OFFICIAL NARRATIVE. Never paraphrase, insert an",
            "ellipsis, or copy a concept that appears only in a group or claim hint.",
            "Give every obligation a distinct normalized anchor set. When splitting one",
            "compound narrative phrase, use the smallest distinctive subphrase for each part",
            "instead of repeating the complete phrase across obligations.",
            "Before returning, verify every span against the official narrative and verify",
            "that target_words across all obligations totals between 850 and 950 inclusive.",
        ]
    )
    return "\n".join(lines)


def _topic_aliases(topic: GenerationTopic) -> tuple[dict[str, str], dict[str, str]]:
    group_aliases = {
        group.group_id: _group_alias(index)
        for index, group in enumerate(topic.groups, start=1)
    }
    claim_aliases = {
        claim.claim_id: _claim_alias(index)
        for index, claim in enumerate(topic.claim_hints, start=1)
    }
    return group_aliases, claim_aliases


def validate_blueprint(topic: GenerationTopic, payload: object) -> NarrativeBlueprint:
    """Validate and locally authenticate one provider planner response."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    root = _require_exact_mapping(payload, {"obligations"}, "blueprint")
    obligations_raw = root["obligations"]
    if not isinstance(obligations_raw, list):
        raise BlueprintValidationError("obligations must be an array")
    if not 3 <= len(obligations_raw) <= 8:
        raise BlueprintValidationError("obligations must contain 3-8 items")

    _group_aliases, claim_aliases_by_id = _topic_aliases(topic)
    claim_by_alias = {
        alias: topic.claim_hints[index]
        for index, alias in enumerate(claim_aliases_by_id.values())
    }
    narrative_normalized = _normalized_text(topic.narrative)
    seen_span_sets: set[frozenset[str]] = set()
    parsed: list[BlueprintObligation] = []
    has_must = False

    for index, raw_obligation in enumerate(obligations_raw, start=1):
        label = f"obligation {index}"
        row = _require_exact_mapping(
            raw_obligation,
            {
                "label",
                "narrative_spans",
                "priority",
                "answer_mode",
                "target_words",
                "selected_claim_aliases",
            },
            label,
        )
        text_label = _require_nonempty_text(row["label"], f"{label} label")
        priority = row["priority"]
        if not isinstance(priority, str) or priority not in PRIORITIES:
            raise BlueprintValidationError(f"{label} priority must be must, should, or could")
        has_must = has_must or priority == "must"
        answer_mode = row["answer_mode"]
        if not isinstance(answer_mode, str) or answer_mode not in ANSWER_MODES:
            raise BlueprintValidationError(f"{label} answer_mode is invalid")

        spans_raw = row["narrative_spans"]
        if not isinstance(spans_raw, list) or not 1 <= len(spans_raw) <= 4:
            raise BlueprintValidationError(f"{label} narrative spans must contain 1-4 items")
        spans: list[str] = []
        normalized_spans: list[str] = []
        for span_index, span_raw in enumerate(spans_raw, start=1):
            span = _require_nonempty_text(span_raw, f"{label} narrative span {span_index}")
            normalized = _normalized_text(span)
            if not normalized:
                raise BlueprintValidationError(f"{label} narrative span is empty")
            if normalized not in narrative_normalized:
                raise BlueprintValidationError(
                    f"{label} narrative span is not present in narrative"
                )
            if normalized in normalized_spans:
                raise BlueprintValidationError(f"{label} has duplicate narrative span")
            spans.append(span)
            normalized_spans.append(normalized)
        span_set = frozenset(normalized_spans)
        if span_set in seen_span_sets:
            raise BlueprintValidationError("duplicate normalized narrative span set")
        seen_span_sets.add(span_set)

        target_words = row["target_words"]
        if (
            isinstance(target_words, bool)
            or not isinstance(target_words, int)
            or target_words <= 0
        ):
            raise BlueprintValidationError(f"{label} target_words must be a positive integer")

        aliases_raw = row["selected_claim_aliases"]
        if not isinstance(aliases_raw, list) or not aliases_raw:
            raise BlueprintValidationError(
                f"{label} must select at least one claim alias"
            )
        aliases: list[str] = []
        claim_ids: list[str] = []
        group_ids: list[str] = []
        for alias_raw in aliases_raw:
            if not isinstance(alias_raw, str) or alias_raw not in claim_by_alias:
                raise BlueprintValidationError(
                    f"{label} references unknown claim alias: {alias_raw!r}"
                )
            if alias_raw in aliases:
                raise BlueprintValidationError(f"{label} has duplicate claim alias")
            aliases.append(alias_raw)
            claim = claim_by_alias[alias_raw]
            claim_ids.append(claim.claim_id)
            if claim.group_id not in group_ids:
                group_ids.append(claim.group_id)

        parsed.append(
            BlueprintObligation(
                label=text_label,
                narrative_spans=tuple(spans),
                priority=priority,
                answer_mode=answer_mode,
                target_words=target_words,
                selected_claim_aliases=tuple(aliases),
            )._with_claims(tuple(claim_ids), tuple(group_ids))
        )

    if not has_must:
        raise BlueprintValidationError("blueprint must contain at least one must obligation")
    total_words = sum(obligation.target_words for obligation in parsed)
    if not MIN_ALLOCATED_WORDS <= total_words <= MAX_ALLOCATED_WORDS:
        raise BlueprintValidationError(
            f"target_words total must be within {MIN_ALLOCATED_WORDS}-{MAX_ALLOCATED_WORDS}"
        )
    return NarrativeBlueprint(obligations=tuple(parsed))


def project_blueprint(
    topic: GenerationTopic, blueprint: NarrativeBlueprint
) -> BlueprintProjection:
    """Resolve claims and derive deterministic per-obligation evidence sets."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    if not isinstance(blueprint, NarrativeBlueprint):
        raise TypeError("blueprint must be NarrativeBlueprint")
    _group_aliases, claim_aliases_by_id = _topic_aliases(topic)
    claim_by_alias = {
        alias: topic.claim_hints[index]
        for index, alias in enumerate(claim_aliases_by_id.values())
    }
    evidence_by_id = {evidence.evidence_id: evidence for evidence in topic.evidence}
    projected_by_obligation: list[tuple[str, ...]] = []
    projection_obligations: list[BlueprintProjectionObligation] = []

    for blueprint_obligation in blueprint.obligations:
        claims = [
            claim_by_alias[alias]
            for alias in blueprint_obligation.selected_claim_aliases
        ]
        linked_ids = {
            evidence_id
            for claim in claims
            for evidence_id in claim.evidence_ids
        }
        represented_groups = {claim.group_id for claim in claims}
        if blueprint_obligation.priority == "must":
            selected_ids = {
                evidence.evidence_id
                for evidence in topic.evidence
                if evidence.group_id in represented_groups
            }
        else:
            selected_ids = linked_ids
        evidence_ids = tuple(
            evidence.evidence_id
            for evidence in topic.evidence
            if evidence.evidence_id in selected_ids
        )
        projected_by_obligation.append(evidence_ids)
        projection_obligations.append(
            BlueprintProjectionObligation(
                label=blueprint_obligation.label,
                narrative_spans=blueprint_obligation.narrative_spans,
                priority=blueprint_obligation.priority,
                answer_mode=blueprint_obligation.answer_mode,
                target_words=blueprint_obligation.target_words,
                selected_claim_aliases=blueprint_obligation.selected_claim_aliases,
                claim_ids=tuple(claim.claim_id for claim in claims),
                evidence_ids=evidence_ids,
                evidence_aliases=(),
            )
        )

    global_ids = tuple(
        evidence.evidence_id
        for evidence in topic.evidence
        if any(evidence.evidence_id in selected for selected in projected_by_obligation)
    )
    alias_by_id = {
        evidence_id: _evidence_alias(index)
        for index, evidence_id in enumerate(global_ids, start=1)
    }
    projection_obligations = [
        BlueprintProjectionObligation(
            label=obligation.label,
            narrative_spans=obligation.narrative_spans,
            priority=obligation.priority,
            answer_mode=obligation.answer_mode,
            target_words=obligation.target_words,
            selected_claim_aliases=obligation.selected_claim_aliases,
            claim_ids=obligation.claim_ids,
            evidence_ids=obligation.evidence_ids,
            evidence_aliases=tuple(alias_by_id[evidence_id] for evidence_id in obligation.evidence_ids),
        )
        for obligation in projection_obligations
    ]
    citation_docids = tuple(
        docid
        for docid in topic.citation_docids
        if any(evidence_by_id[evidence_id].docid == docid for evidence_id in global_ids)
    )
    return BlueprintProjection(
        obligations=tuple(projection_obligations),
        evidence_ids=global_ids,
        evidence_aliases=tuple(alias_by_id[evidence_id] for evidence_id in global_ids),
        citation_docids=citation_docids,
    )


def render_blueprint_writer_context(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
) -> str:
    """Render writer input, including each projected passage exactly once."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    if not isinstance(blueprint, NarrativeBlueprint):
        raise TypeError("blueprint must be NarrativeBlueprint")
    if not isinstance(projection, BlueprintProjection):
        raise TypeError("projection must be BlueprintProjection")
    expected_projection = project_blueprint(topic, blueprint)
    if projection != expected_projection:
        raise BlueprintValidationError("writer projection does not match blueprint")
    claims_by_id = {claim.claim_id: claim for claim in topic.claim_hints}
    evidence_by_id = {evidence.evidence_id: evidence for evidence in topic.evidence}
    evidence_alias_by_id = dict(zip(projection.evidence_ids, projection.evidence_aliases))
    lines = [
        "NARRATIVE BLUEPRINT",
        "Write a grounded answer to the complete official narrative.",
        "Cover must obligations first, then should obligations, and use could obligations only when room remains.",
        "The target word allocations are planning capacity, not a padding quota; never repeat or pad.",
        "Within each obligation, treat its advisory claims as a coverage checklist after verifying them against the passages.",
        "Prefer distinct, high-specificity facts—including quantities, named mechanisms, actors, interventions, contrasts, and uncertainty—over broad repetition.",
        "Do not collapse a supported list of distinct causes, effects, or actions into only a generic category label.",
        "Use only the frozen selected retrieval evidence below and cite its exact docids.",
        "FROZEN SELECTED RETRIEVAL EVIDENCE",
        "",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        "OBLIGATIONS:",
    ]
    for index, (blueprint_obligation, projected_obligation) in enumerate(
        zip(blueprint.obligations, projection.obligations), start=1
    ):
        lines.append(
            f"OBLIGATION {index} [{projected_obligation.priority}; "
            f"{projected_obligation.answer_mode}; ~{projected_obligation.target_words} words] "
            f"{projected_obligation.label}"
        )
        lines.append("  Narrative anchors: " + " | ".join(blueprint_obligation.narrative_spans))
        claim_lines: list[str] = []
        for alias, claim_id in zip(
            projected_obligation.selected_claim_aliases,
            projected_obligation.claim_ids,
        ):
            claim = claims_by_id[claim_id]
            claim_lines.append(f"{alias}: {claim.text}")
        lines.append("  Advisory claims: " + " | ".join(claim_lines))
        lines.append(
            "  Evidence aliases: " + ", ".join(projected_obligation.evidence_aliases)
        )
    lines.extend(["", "SELECTED EVIDENCE CATALOG:"])
    for evidence_id in projection.evidence_ids:
        evidence = evidence_by_id[evidence_id]
        lines.extend(
            [
                f"[{evidence_alias_by_id[evidence_id]}] DOCID: {evidence.docid}",
                evidence.text,
            ]
        )
    lines.extend(
        [
            "",
            "Return the existing organizer JSON object with sentence-level answer entries.",
            "Each answer entry may cite zero to three exact docids from this catalog; keep the full answer at or below 1,024 words.",
        ]
    )
    return "\n".join(lines)


def _serialize_projection(projection: BlueprintProjection) -> dict[str, object]:
    return projection.to_payload()


def serialize_blueprint_state(
    topic: GenerationTopic,
    blueprint: NarrativeBlueprint,
    projection: BlueprintProjection,
    *,
    planner_prompt_sha256: str,
    writer_context_sha256: str,
) -> dict[str, object]:
    """Create a canonical, self-authenticating state record for resume."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    if not isinstance(blueprint, NarrativeBlueprint):
        raise TypeError("blueprint must be NarrativeBlueprint")
    if not isinstance(projection, BlueprintProjection):
        raise TypeError("projection must be BlueprintProjection")
    _require_sha256(planner_prompt_sha256, "planner_prompt_sha256")
    _require_sha256(writer_context_sha256, "writer_context_sha256")
    expected_projection = project_blueprint(topic, blueprint)
    if projection != expected_projection:
        raise BlueprintValidationError("state projection does not match blueprint")
    state_without_hash: dict[str, object] = {
        "contract_version": BLUEPRINT_CONTRACT_VERSION,
        "topic_id": topic.topic_id,
        "topic_context_sha256": topic.context_sha256,
        "planner_prompt_sha256": planner_prompt_sha256,
        "writer_context_sha256": writer_context_sha256,
        "blueprint": blueprint.to_payload(),
        "projection": _serialize_projection(projection),
    }
    return {
        **state_without_hash,
        "state_sha256": _digest(state_without_hash),
    }


def _parse_blueprint_state_blueprint(topic: GenerationTopic, value: object) -> NarrativeBlueprint:
    row = _require_exact_mapping(value, {"contract_version", "obligations"}, "blueprint state blueprint")
    if row["contract_version"] != BLUEPRINT_CONTRACT_VERSION:
        raise BlueprintValidationError("state blueprint contract version differs")
    payload = {"obligations": row["obligations"]}
    return validate_blueprint(topic, payload)


def load_blueprint_state(
    topic: GenerationTopic,
    payload: object,
    *,
    planner_prompt_sha256: str,
) -> tuple[NarrativeBlueprint, BlueprintProjection]:
    """Authenticate and load a persisted blueprint/projection pair."""

    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    _require_sha256(planner_prompt_sha256, "planner_prompt_sha256")
    state = _require_exact_mapping(
        payload,
        {
            "contract_version",
            "topic_id",
            "topic_context_sha256",
            "planner_prompt_sha256",
            "writer_context_sha256",
            "blueprint",
            "projection",
            "state_sha256",
        },
        "blueprint state",
    )
    state_without_hash = {
        key: value for key, value in state.items() if key != "state_sha256"
    }
    try:
        actual_state_hash = _digest(state_without_hash)
    except BlueprintValidationError as exc:
        raise BlueprintValidationError(f"state authentication failed: {exc}") from exc
    if state["state_sha256"] != actual_state_hash:
        raise BlueprintValidationError("state authentication hash differs")
    if state["contract_version"] != BLUEPRINT_CONTRACT_VERSION:
        raise BlueprintValidationError("state contract version differs")
    if state["topic_id"] != topic.topic_id:
        raise BlueprintValidationError("state topic ID differs")
    if state["topic_context_sha256"] != topic.context_sha256:
        raise BlueprintValidationError("state topic context differs")
    if state["planner_prompt_sha256"] != planner_prompt_sha256:
        raise BlueprintValidationError("state planner prompt hash differs")
    writer_hash = _require_sha256(state["writer_context_sha256"], "writer_context_sha256")

    blueprint = _parse_blueprint_state_blueprint(topic, state["blueprint"])
    expected_projection = project_blueprint(topic, blueprint)
    projection_row = _require_exact_mapping(
        state["projection"],
        {"obligations", "evidence_ids", "evidence_aliases", "citation_docids"},
        "blueprint state projection",
    )
    expected_projection_payload = expected_projection.to_payload()
    if dict(projection_row) != expected_projection_payload:
        raise BlueprintValidationError("state projection mapping differs")
    actual_writer_hash = _text_digest(
        render_blueprint_writer_context(topic, blueprint, expected_projection)
    )
    if writer_hash != actual_writer_hash:
        raise BlueprintValidationError("state writer context hash differs")
    return blueprint, expected_projection


__all__ = [
    "ANSWER_MODES",
    "BLUEPRINT_CONTRACT_VERSION",
    "BlueprintObligation",
    "BlueprintProjection",
    "BlueprintProjectionObligation",
    "BlueprintValidationError",
    "MAX_ALLOCATED_WORDS",
    "MIN_ALLOCATED_WORDS",
    "NarrativeBlueprint",
    "PRIORITIES",
    "load_blueprint_state",
    "planner_response_schema",
    "project_blueprint",
    "render_blueprint_writer_context",
    "render_planner_prompt",
    "serialize_blueprint_state",
    "validate_blueprint",
]

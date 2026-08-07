"""Pure validation and scoring domain for retrieval nugget coverage.

The planner and judge payloads are untrusted model output.  This module keeps
their admission rules and the deterministic score independent of any transport
or persistence code so callers can test the evaluator without hosted calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal, Protocol
import unicodedata

from trec_rag.facet_extraction import BackendReply


PLAN_SCHEMA_VERSION = "retrieval_nugget_plan_v1"
JUDGMENT_SCHEMA_VERSION = "retrieval_nugget_judgment_v1"
MAX_FACETS = 12
MAX_OBLIGATIONS_PER_FACET = 8
MAX_OBLIGATIONS = 40
MAX_JUDGE_REQUEST_BYTES = 1_000_000
MAX_COMPLETION_TOKENS = 4096
PLANNER_PROMPT_VERSION = "retrieval_nugget_planner_v1"
JUDGE_PROMPT_VERSION = "retrieval_nugget_judge_v1"
PLANNER_SYSTEM_PROMPT = (
    "Create a complete, minimal obligation plan from the supplied narrative. "
    "Return only the requested JSON object."
)
JUDGE_SYSTEM_PROMPT = (
    "Judge each frozen obligation against the supplied retrieval nuggets. "
    "Use only the supplied text and return only the requested JSON object."
)
MAX_TEXT_CHARACTERS = 300
_PLAN_ROOT_KEYS = frozenset({"schema_version", "facets", "unmapped_narrative_spans"})
_FACET_KEYS = frozenset({"title", "obligations"})
_OBLIGATION_KEYS = frozenset(
    {"requirement", "support_test", "kind", "narrative_spans"}
)
_JUDGMENT_ROOT_KEYS = frozenset({"schema_version", "judgments"})
_JUDGMENT_KEYS = frozenset(
    {"obligation_id", "label", "supporting_nugget_aliases", "missing_elements"}
)
_KINDS = frozenset({"required_explicit", "supplemental_inferred"})
_LABELS = frozenset({"full", "partial", "unsupported"})
_LABEL_VALUE = {"full": 1.0, "partial": 0.5, "unsupported": 0.0}


class NuggetCoverageError(ValueError):
    """A fail-closed validation or scoring error with a stable stage label."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class CoverageNugget:
    """One canonical retrieval claim supplied to the coverage judge."""

    nugget_id: str
    text: str


@dataclass(frozen=True)
class EvaluatorIdentity:
    schema_version: str
    planner_prompt_version: str
    judge_prompt_version: str
    planner_model: str
    judge_model: str


@dataclass(frozen=True)
class CoverageModelRequest:
    """Backend-neutral structured request for one planner or judge call."""

    stage: Literal["planner", "judge"]
    model: str
    messages: tuple[Mapping[str, str], ...]
    response_schema_name: str
    response_schema: Mapping[str, object]
    max_tokens: int


class CoverageModelBackend(Protocol):
    """Injected backend for exactly one structured model completion."""

    def complete(self, request: CoverageModelRequest) -> BackendReply: ...


@dataclass(frozen=True)
class EvaluationCallMetadata:
    """Secret-free metadata returned for the planner and judge calls."""

    planner_metadata: Mapping[str, object]
    judge_metadata: Mapping[str, object]
    planner_status: int
    judge_status: int
    planner_request_sha256: str
    judge_request_sha256: str

    @property
    def planner(self) -> Mapping[str, object]:
        return self.planner_metadata

    @property
    def judge(self) -> Mapping[str, object]:
        return self.judge_metadata


@dataclass(frozen=True)
class CoverageObligation:
    obligation_id: str
    requirement: str
    support_test: str
    kind: str
    narrative_spans: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.obligation_id


@dataclass(frozen=True)
class CoverageFacet:
    facet_id: str
    title: str
    obligations: tuple[CoverageObligation, ...]

    @property
    def id(self) -> str:
        return self.facet_id


@dataclass(frozen=True)
class FrozenPlan:
    """Validated planner output with position-derived IDs and content hash."""

    narrative: str
    schema_version: str
    facets: tuple[CoverageFacet, ...]
    unmapped_narrative_spans: tuple[str, ...]
    canonical_bytes: bytes
    plan_sha256: str

    @property
    def obligations(self) -> tuple[CoverageObligation, ...]:
        return tuple(obligation for facet in self.facets for obligation in facet.obligations)

    @property
    def canonical_json(self) -> bytes:
        return self.canonical_bytes

    @property
    def sha256(self) -> str:
        return self.plan_sha256

    @property
    def required_obligations(self) -> tuple[CoverageObligation, ...]:
        return tuple(
            obligation
            for obligation in self.obligations
            if obligation.kind == "required_explicit"
        )

    @property
    def required_obligation_count(self) -> int:
        return sum(obligation.kind == "required_explicit" for obligation in self.obligations)


@dataclass(frozen=True)
class CoverageJudgment:
    """Validated judgment with aliases resolved to canonical nugget IDs."""

    obligation_id: str
    label: str
    supporting_nugget_ids: tuple[str, ...]
    missing_elements: str

    @property
    def resolved_nugget_ids(self) -> tuple[str, ...]:
        return self.supporting_nugget_ids


@dataclass(frozen=True)
class CoverageReport:
    """Deterministic score and diagnostics for one frozen obligation plan."""

    plan_sha256: str
    identity: EvaluatorIdentity
    required_coverage: float
    strict_full_rate: float
    supplemental_coverage: float | None
    label_counts: Mapping[str, int]
    facet_scores: Mapping[str, float]
    judgments: tuple[CoverageJudgment, ...]
    unmapped_narrative_spans: tuple[str, ...]
    uncited_nugget_ids: tuple[str, ...]
    uncited_nugget_aliases: tuple[str, ...]

    @property
    def obligation_judgments(self) -> tuple[CoverageJudgment, ...]:
        """Compatibility/readability alias for the report's ordered judgments."""
        return self.judgments

    @property
    def required_facet_scores(self) -> Mapping[str, float]:
        return self.facet_scores


EvaluationResult = tuple[
    FrozenPlan,
    tuple[CoverageJudgment, ...],
    CoverageReport,
    EvaluationCallMetadata,
]


def render_planner_request(
    narrative: str,
    model: str | EvaluatorIdentity,
    *,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> CoverageModelRequest:
    """Render the narrative-only planner request deterministically."""

    narrative = _text(narrative, "narrative", None, stage="planner")
    if isinstance(model, EvaluatorIdentity):
        model = model.planner_model
    model = _text(model, "planner model", None, stage="planner")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        _error("planner", "max_tokens must be a positive integer")
    user_payload = {"narrative": narrative}
    return CoverageModelRequest(
        stage="planner",
        model=model,
        messages=_messages(PLANNER_SYSTEM_PROMPT, user_payload, stage="planner"),
        response_schema_name=PLAN_SCHEMA_VERSION,
        response_schema=_planner_response_schema(),
        max_tokens=max_tokens,
    )


def render_judge_request(
    narrative: str,
    plan: FrozenPlan,
    nuggets: Sequence[CoverageNugget],
    model: str | EvaluatorIdentity,
    *,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> CoverageModelRequest:
    """Render the frozen-plan and ordered local-alias judge request."""

    narrative = _text(narrative, "narrative", None, stage="judge")
    if not isinstance(plan, FrozenPlan):
        _error("judge", "plan must be a FrozenPlan")
    if plan.narrative != narrative:
        _error("judge", "plan narrative does not match the judge narrative")
    nugget_rows = _validate_nuggets(nuggets, stage="judge")
    if isinstance(model, EvaluatorIdentity):
        model = model.judge_model
    model = _text(model, "judge model", None, stage="judge")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        _error("judge", "max_tokens must be a positive integer")
    try:
        frozen_plan = json.loads(plan.canonical_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - invariant guard
        _error("judge", f"frozen plan is not canonical JSON: {exc}")
    user_payload = {
        "narrative": narrative,
        "plan": frozen_plan,
        "nuggets": [
            f"n{index:03d}: {nugget.text}"
            for index, nugget in enumerate(nugget_rows, start=1)
        ],
    }
    return CoverageModelRequest(
        stage="judge",
        model=model,
        messages=_messages(JUDGE_SYSTEM_PROMPT, user_payload, stage="judge"),
        response_schema_name=JUDGMENT_SCHEMA_VERSION,
        response_schema=_judge_response_schema(plan, len(nugget_rows)),
        max_tokens=max_tokens,
    )


def evaluate_nugget_coverage(
    *,
    narrative: str,
    nuggets: Sequence[CoverageNugget],
    planner: CoverageModelBackend,
    judge: CoverageModelBackend,
    identity: EvaluatorIdentity,
) -> EvaluationResult:
    """Run one planner call followed by one judge call and score the result."""

    narrative = _text(narrative, "narrative", None, stage="input")
    nugget_rows = _validate_nuggets(nuggets, stage="input")
    if not isinstance(identity, EvaluatorIdentity):
        _error("input", "identity must be an EvaluatorIdentity")
    if not callable(getattr(planner, "complete", None)):
        _error("input", "planner must provide complete(request)")
    if not callable(getattr(judge, "complete", None)):
        _error("input", "judge must provide complete(request)")

    planner_request = render_planner_request(narrative, identity)
    planner_body = _serialized_provider_request(planner_request, stage="planner")
    planner_reply = _complete(planner, planner_request, stage="planner")
    planner_payload = _decode_model_payload(planner_reply, stage="planner")
    frozen_plan = validate_and_freeze_plan(narrative, planner_payload)

    judge_request = render_judge_request(narrative, frozen_plan, nugget_rows, identity)
    judge_body = _serialized_provider_request(judge_request, stage="judge")
    if len(judge_body) > MAX_JUDGE_REQUEST_BYTES:
        _error(
            "judge",
            f"judge request exceeds the {MAX_JUDGE_REQUEST_BYTES:,}-byte limit",
        )
    judge_reply = _complete(judge, judge_request, stage="judge")
    judge_payload = _decode_model_payload(judge_reply, stage="judge")
    judgments = validate_judgments(frozen_plan, nugget_rows, judge_payload)
    report = score_coverage(frozen_plan, nugget_rows, judgments, identity)
    metadata = EvaluationCallMetadata(
        planner_metadata=_safe_metadata(planner_reply, stage="planner"),
        judge_metadata=_safe_metadata(judge_reply, stage="judge"),
        planner_status=planner_reply.status,
        judge_status=judge_reply.status,
        planner_request_sha256=sha256(planner_body).hexdigest(),
        judge_request_sha256=sha256(judge_body).hexdigest(),
    )
    return frozen_plan, judgments, report, metadata


def validate_and_freeze_plan(narrative: str, payload: object) -> FrozenPlan:
    """Validate one planner payload and return a deterministic immutable plan."""

    narrative = _text(narrative, "narrative", None, stage="planner")
    root = _mapping(payload, "planner payload", stage="planner")
    _require_keys(root, _PLAN_ROOT_KEYS, "planner payload", stage="planner")
    if root["schema_version"] != PLAN_SCHEMA_VERSION:
        _error("planner", "schema_version must be retrieval_nugget_plan_v1")

    facets_payload = _list(root["facets"], "facets", 1, MAX_FACETS, stage="planner")
    unmapped_payload = _list(
        root["unmapped_narrative_spans"],
        "unmapped_narrative_spans",
        0,
        None,
        stage="planner",
    )

    parsed_facets: list[tuple[str, tuple[dict[str, object], ...]]] = []
    total_obligations = 0
    required_count = 0
    for facet_index, facet_value in enumerate(facets_payload, start=1):
        facet = _mapping(facet_value, f"facet {facet_index}", stage="planner")
        _require_keys(facet, _FACET_KEYS, f"facet {facet_index}", stage="planner")
        title = _text(facet["title"], f"facet {facet_index} title", MAX_TEXT_CHARACTERS, stage="planner")
        obligations_payload = _list(
            facet["obligations"],
            f"facet {facet_index} obligations",
            1,
            MAX_OBLIGATIONS_PER_FACET,
            stage="planner",
        )
        parsed_obligations: list[dict[str, object]] = []
        for obligation_index, obligation_value in enumerate(obligations_payload, start=1):
            obligation = _mapping(
                obligation_value,
                f"facet {facet_index} obligation {obligation_index}",
                stage="planner",
            )
            _require_keys(
                obligation,
                _OBLIGATION_KEYS,
                f"facet {facet_index} obligation {obligation_index}",
                stage="planner",
            )
            requirement = _text(
                obligation["requirement"],
                f"facet {facet_index} obligation {obligation_index} requirement",
                MAX_TEXT_CHARACTERS,
                stage="planner",
            )
            support_test = _text(
                obligation["support_test"],
                f"facet {facet_index} obligation {obligation_index} support_test",
                MAX_TEXT_CHARACTERS,
                stage="planner",
            )
            kind = obligation["kind"]
            if not isinstance(kind, str) or kind not in _KINDS:
                _error("planner", f"facet {facet_index} obligation {obligation_index} kind is invalid")
            spans_payload = _list(
                obligation["narrative_spans"],
                f"facet {facet_index} obligation {obligation_index} narrative_spans",
                0,
                None,
                stage="planner",
            )
            spans: list[str] = []
            for span_index, span_value in enumerate(spans_payload, start=1):
                span = _text(
                    span_value,
                    f"facet {facet_index} obligation {obligation_index} narrative span {span_index}",
                    None,
                    stage="planner",
                )
                if span not in narrative:
                    _error("planner", f"narrative span {span_index} is not an exact narrative substring")
                spans.append(span)
            if kind == "required_explicit":
                if not spans:
                    _error(
                        "planner",
                        f"facet {facet_index} obligation {obligation_index} required obligation needs a narrative span",
                    )
                required_count += 1
            elif spans:
                _error(
                    "planner",
                    f"facet {facet_index} obligation {obligation_index} supplemental obligation must have empty narrative spans",
                )
            parsed_obligations.append(
                {
                    "requirement": requirement,
                    "support_test": support_test,
                    "kind": kind,
                    "narrative_spans": tuple(spans),
                }
            )
        total_obligations += len(parsed_obligations)
        if total_obligations > MAX_OBLIGATIONS:
            _error("planner", "plan cannot contain more than 40 obligations")
        parsed_facets.append((title, tuple(parsed_obligations)))

    unmapped_spans: list[str] = []
    for index, value in enumerate(unmapped_payload, start=1):
        span = _text(value, f"unmapped narrative span {index}", None, stage="planner")
        if span not in narrative:
            _error("planner", f"unmapped narrative span {index} is not an exact narrative substring")
        unmapped_spans.append(span)
    if required_count == 0:
        _error("planner", "plan must contain at least one required obligation")

    facets: list[CoverageFacet] = []
    canonical_facets: list[dict[str, object]] = []
    for facet_index, (title, parsed_obligations) in enumerate(parsed_facets, start=1):
        facet_id = f"f{facet_index:03d}"
        obligations: list[CoverageObligation] = []
        canonical_obligations: list[dict[str, object]] = []
        for obligation_index, parsed in enumerate(parsed_obligations, start=1):
            obligation_id = f"{facet_id}-o{obligation_index:03d}"
            spans = parsed["narrative_spans"]
            assert isinstance(spans, tuple)
            obligation = CoverageObligation(
                obligation_id=obligation_id,
                requirement=parsed["requirement"],
                support_test=parsed["support_test"],
                kind=parsed["kind"],
                narrative_spans=spans,
            )
            obligations.append(obligation)
            canonical_obligations.append(
                {
                    "obligation_id": obligation_id,
                    "requirement": obligation.requirement,
                    "support_test": obligation.support_test,
                    "kind": obligation.kind,
                    "narrative_spans": list(obligation.narrative_spans),
                }
            )
        facet = CoverageFacet(facet_id=facet_id, title=title, obligations=tuple(obligations))
        facets.append(facet)
        canonical_facets.append(
            {"facet_id": facet_id, "title": title, "obligations": canonical_obligations}
        )

    canonical_value = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "facets": canonical_facets,
        "unmapped_narrative_spans": unmapped_spans,
    }
    canonical_bytes = _canonical_json(canonical_value, stage="planner")
    return FrozenPlan(
        narrative=narrative,
        schema_version=PLAN_SCHEMA_VERSION,
        facets=tuple(facets),
        unmapped_narrative_spans=tuple(unmapped_spans),
        canonical_bytes=canonical_bytes,
        plan_sha256=sha256(canonical_bytes).hexdigest(),
    )


def validate_judgments(
    plan: FrozenPlan,
    nuggets: Sequence[CoverageNugget],
    payload: object,
) -> tuple[CoverageJudgment, ...]:
    """Validate judge output and resolve compact aliases to canonical IDs."""

    if not isinstance(plan, FrozenPlan):
        _error("judge", "plan must be a FrozenPlan")
    nugget_rows = _validate_nuggets(nuggets, stage="judge")
    root = _mapping(payload, "judge payload", stage="judge")
    _require_keys(root, _JUDGMENT_ROOT_KEYS, "judge payload", stage="judge")
    if root["schema_version"] != JUDGMENT_SCHEMA_VERSION:
        _error("judge", "schema_version must be retrieval_nugget_judgment_v1")
    rows = _list(root["judgments"], "judgments", len(plan.obligations), len(plan.obligations), stage="judge")

    obligation_by_id = {obligation.obligation_id: obligation for obligation in plan.obligations}
    aliases = {f"n{index:03d}": nugget.nugget_id for index, nugget in enumerate(nugget_rows, start=1)}
    seen_obligation_ids: set[str] = set()
    parsed: dict[str, CoverageJudgment] = {}
    for index, row_value in enumerate(rows, start=1):
        row = _mapping(row_value, f"judgment {index}", stage="judge")
        _require_keys(row, _JUDGMENT_KEYS, f"judgment {index}", stage="judge")
        obligation_id = row["obligation_id"]
        if not isinstance(obligation_id, str) or obligation_id not in obligation_by_id:
            _error("judge", f"judgment {index} has unknown obligation ID")
        if obligation_id in seen_obligation_ids:
            _error("judge", f"judgment {index} has a duplicate obligation ID")
        seen_obligation_ids.add(obligation_id)
        label = row["label"]
        if not isinstance(label, str) or label not in _LABELS:
            _error("judge", f"judgment {index} label is invalid")
        aliases_payload = _list(
            row["supporting_nugget_aliases"],
            f"judgment {index} supporting_nugget_aliases",
            0,
            len(nugget_rows),
            stage="judge",
        )
        supporting_ids: list[str] = []
        seen_aliases: set[str] = set()
        for alias in aliases_payload:
            if not isinstance(alias, str) or alias not in aliases:
                _error("judge", f"judgment {index} has an unknown supporting nugget alias")
            if alias in seen_aliases:
                _error("judge", f"judgment {index} has a duplicate supporting nugget alias")
            seen_aliases.add(alias)
            supporting_ids.append(aliases[alias])
        missing_elements = row["missing_elements"]
        if label == "unsupported" and supporting_ids:
            _error("judge", f"unsupported judgment {index} must have empty supporting aliases")
        if label in {"full", "partial"} and not supporting_ids:
            _error("judge", f"{label} judgment {index} requires supporting aliases")
        if label == "full":
            if missing_elements != "":
                _error("judge", f"full judgment {index} must have empty missing_elements")
        else:
            _text(
                missing_elements,
                f"judgment {index} missing_elements",
                MAX_TEXT_CHARACTERS,
                stage="judge",
            )
        if not isinstance(missing_elements, str):
            _error("judge", f"judgment {index} missing_elements must be text")
        parsed[obligation_id] = CoverageJudgment(
            obligation_id=obligation_id,
            label=label,
            supporting_nugget_ids=tuple(supporting_ids),
            missing_elements=missing_elements,
        )
    if seen_obligation_ids != set(obligation_by_id):
        _error("judge", "judgments must contain exactly one row per obligation")
    return tuple(parsed[obligation.obligation_id] for obligation in plan.obligations)


def score_coverage(
    plan: FrozenPlan,
    nuggets: Sequence[CoverageNugget],
    judgments: Sequence[CoverageJudgment],
    identity: EvaluatorIdentity,
) -> CoverageReport:
    """Compute full-precision required and supplemental coverage metrics."""

    if not isinstance(plan, FrozenPlan):
        _error("scoring", "plan must be a FrozenPlan")
    if not isinstance(identity, EvaluatorIdentity):
        _error("scoring", "identity must be an EvaluatorIdentity")
    nugget_rows = _validate_nuggets(nuggets, stage="scoring")
    if not isinstance(judgments, Sequence) or isinstance(judgments, (str, bytes)):
        _error("scoring", "judgments must be a sequence")
    expected_obligations = plan.obligations
    expected_ids = {obligation.obligation_id for obligation in expected_obligations}
    if len(judgments) != len(expected_obligations):
        _error("scoring", "judgments must contain exactly one row per obligation")
    by_id: dict[str, CoverageJudgment] = {}
    known_nugget_ids = {nugget.nugget_id for nugget in nugget_rows}
    for judgment in judgments:
        if not isinstance(judgment, CoverageJudgment):
            _error("scoring", "judgments must contain CoverageJudgment records")
        if judgment.obligation_id not in expected_ids:
            _error("scoring", "judgments contain an unknown obligation ID")
        if judgment.obligation_id in by_id:
            _error("scoring", "judgments contain a duplicate obligation ID")
        if not isinstance(judgment.label, str) or judgment.label not in _LABEL_VALUE:
            _error("scoring", "judgments contain an invalid label")
        if not isinstance(judgment.supporting_nugget_ids, tuple):
            _error("scoring", "judgment supporting nugget IDs must be canonical")
        if not all(isinstance(nugget_id, str) for nugget_id in judgment.supporting_nugget_ids):
            _error("scoring", "judgment supporting nugget IDs must be text")
        if len(set(judgment.supporting_nugget_ids)) != len(judgment.supporting_nugget_ids):
            _error("scoring", "judgment supporting nugget IDs must be unique")
        if any(nugget_id not in known_nugget_ids for nugget_id in judgment.supporting_nugget_ids):
            _error("scoring", "judgments contain an unknown nugget ID")
        if judgment.label == "unsupported" and judgment.supporting_nugget_ids:
            _error("scoring", "unsupported judgments must have empty supporting nugget IDs")
        if judgment.label in {"full", "partial"} and not judgment.supporting_nugget_ids:
            _error("scoring", "supported judgments require supporting nugget IDs")
        if judgment.label == "full" and judgment.missing_elements != "":
            _error("scoring", "full judgments must have empty missing_elements")
        if judgment.label != "full":
            _text(judgment.missing_elements, "judgment missing_elements", MAX_TEXT_CHARACTERS, stage="scoring")
        by_id[judgment.obligation_id] = judgment
    if set(by_id) != expected_ids:
        _error("scoring", "judgments must contain exactly one row per obligation")

    ordered_judgments = tuple(by_id[obligation.obligation_id] for obligation in expected_obligations)
    required_facet_scores: dict[str, float] = {}
    required_values: list[float] = []
    supplemental_values: list[float] = []
    label_counts = {label: 0 for label in ("full", "partial", "unsupported")}
    for judgment in ordered_judgments:
        label_counts[judgment.label] += 1
    for facet in plan.facets:
        required_rows = [
            by_id[obligation.obligation_id]
            for obligation in facet.obligations
            if obligation.kind == "required_explicit"
        ]
        if required_rows:
            facet_score = sum(_LABEL_VALUE[row.label] for row in required_rows) / len(required_rows)
            required_facet_scores[facet.facet_id] = facet_score
            required_values.append(facet_score)
        supplemental_values.extend(
            _LABEL_VALUE[by_id[obligation.obligation_id].label]
            for obligation in facet.obligations
            if obligation.kind == "supplemental_inferred"
        )

    required_coverage = sum(required_values) / len(required_values)
    required_count = sum(
        obligation.kind == "required_explicit" for obligation in expected_obligations
    )
    strict_full_rate = sum(
        by_id[obligation.obligation_id].label == "full"
        for obligation in expected_obligations
        if obligation.kind == "required_explicit"
    ) / required_count
    supplemental_coverage = (
        None if not supplemental_values else sum(supplemental_values) / len(supplemental_values)
    )
    cited_ids = {
        nugget_id
        for judgment in ordered_judgments
        for nugget_id in judgment.supporting_nugget_ids
    }
    uncited_ids = tuple(nugget.nugget_id for nugget in nugget_rows if nugget.nugget_id not in cited_ids)
    uncited_aliases = tuple(
        f"n{index:03d}"
        for index, nugget in enumerate(nugget_rows, start=1)
        if nugget.nugget_id in set(uncited_ids)
    )
    return CoverageReport(
        plan_sha256=plan.plan_sha256,
        identity=identity,
        required_coverage=required_coverage,
        strict_full_rate=float(strict_full_rate),
        supplemental_coverage=supplemental_coverage,
        label_counts=MappingProxyType(label_counts),
        facet_scores=MappingProxyType(required_facet_scores),
        judgments=ordered_judgments,
        unmapped_narrative_spans=plan.unmapped_narrative_spans,
        uncited_nugget_ids=uncited_ids,
        uncited_nugget_aliases=uncited_aliases,
    )


def _messages(
    system_prompt: str,
    user_payload: Mapping[str, object],
    *,
    stage: str,
) -> tuple[Mapping[str, str], ...]:
    try:
        user_content = json.dumps(
            user_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - typed inputs guard this
        _error(stage, f"request JSON serialization failed: {exc}")
    return (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    )


def _planner_response_schema() -> dict[str, object]:
    obligation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirement", "support_test", "kind", "narrative_spans"],
        "properties": {
            "requirement": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARACTERS},
            "support_test": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARACTERS},
            "kind": {
                "type": "string",
                "enum": ["required_explicit", "supplemental_inferred"],
            },
            "narrative_spans": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    facet = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "obligations"],
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARACTERS},
            "obligations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OBLIGATIONS_PER_FACET,
                "items": obligation,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "facets", "unmapped_narrative_spans"],
        "properties": {
            "schema_version": {"type": "string", "const": PLAN_SCHEMA_VERSION},
            "facets": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_FACETS,
                "items": facet,
            },
            "unmapped_narrative_spans": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }


def _judge_response_schema(plan: FrozenPlan, nugget_count: int) -> dict[str, object]:
    obligation_ids = [obligation.obligation_id for obligation in plan.obligations]
    aliases = [f"n{index:03d}" for index in range(1, nugget_count + 1)]
    judgment = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "obligation_id",
            "label",
            "supporting_nugget_aliases",
            "missing_elements",
        ],
        "properties": {
            "obligation_id": {"type": "string", "enum": obligation_ids},
            "label": {
                "type": "string",
                "enum": ["full", "partial", "unsupported"],
            },
            "supporting_nugget_aliases": {
                "type": "array",
                "minItems": 0,
                "maxItems": nugget_count,
                "items": {"type": "string", "enum": aliases},
            },
            "missing_elements": {
                "type": "string",
                "maxLength": MAX_TEXT_CHARACTERS,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "judgments"],
        "properties": {
            "schema_version": {"type": "string", "const": JUDGMENT_SCHEMA_VERSION},
            "judgments": {
                "type": "array",
                "minItems": len(obligation_ids),
                "maxItems": len(obligation_ids),
                "items": judgment,
            },
        },
    }


def _serialized_provider_request(
    request: CoverageModelRequest,
    *,
    stage: str,
) -> bytes:
    if not isinstance(request, CoverageModelRequest):
        _error(stage, "request must be a CoverageModelRequest")
    body = {
        "model": request.model,
        "messages": [dict(message) for message in request.messages],
        "max_tokens": request.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_schema_name,
                "strict": True,
                "schema": request.response_schema,
            },
        },
    }
    return _canonical_json(body, stage=stage)


def _complete(
    backend: CoverageModelBackend,
    request: CoverageModelRequest,
    *,
    stage: str,
) -> BackendReply:
    try:
        reply = backend.complete(request)
    except NuggetCoverageError:
        raise
    except Exception as exc:
        _error(stage, f"backend completion failed: {type(exc).__name__}")
    if not isinstance(reply, BackendReply):
        _error(stage, "backend must return BackendReply")
    return reply


def _decode_model_payload(reply: BackendReply, *, stage: str) -> Mapping[str, object]:
    if not isinstance(reply.content, bytes):
        _error(stage, "backend reply content must be UTF-8 JSON bytes")
    try:
        decoded = reply.content.decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _error(stage, f"backend reply content is not valid JSON: {exc}")
    return _mapping(payload, "model payload", stage=stage)


def _safe_metadata(reply: BackendReply, *, stage: str) -> Mapping[str, object]:
    if not isinstance(reply.metadata, Mapping) or not all(
        isinstance(key, str) for key in reply.metadata
    ):
        _error(stage, "backend reply metadata must be a string-keyed object")
    return MappingProxyType(dict(reply.metadata))


def _error(stage: str, reason: str) -> None:
    raise NuggetCoverageError(stage, reason)


def _mapping(value: object, name: str, *, stage: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _error(stage, f"{name} must be an object")
    return value


def _require_keys(
    value: Mapping[str, object], expected: frozenset[str], name: str, *, stage: str
) -> None:
    if set(value) != expected:
        _error(stage, f"{name} has unexpected or missing fields")


def _list(
    value: object,
    name: str,
    minimum: int,
    maximum: int | None,
    *,
    stage: str,
) -> list[object]:
    if not isinstance(value, list) or len(value) < minimum or (
        maximum is not None and len(value) > maximum
    ):
        bound = f"{minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        _error(stage, f"{name} must contain between {bound} items")
    return value


def _text(
    value: object,
    name: str,
    maximum: int | None,
    *,
    stage: str,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _error(stage, f"{name} must be non-empty text without surrounding whitespace")
    if maximum is not None and len(value) > maximum:
        _error(stage, f"{name} exceeds the {maximum}-character limit")
    if any(unicodedata.category(character) == "Cc" for character in value):
        _error(stage, f"{name} contains control characters")
    return value


def _canonical_json(value: object, *, stage: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _error(stage, f"canonical JSON serialization failed: {exc}")


def _validate_nuggets(
    nuggets: Sequence[CoverageNugget], *, stage: str
) -> tuple[CoverageNugget, ...]:
    if not isinstance(nuggets, Sequence) or isinstance(nuggets, (str, bytes)):
        _error(stage, "nuggets must be a sequence")
    rows: list[CoverageNugget] = []
    seen_ids: set[str] = set()
    for index, nugget in enumerate(nuggets, start=1):
        if not isinstance(nugget, CoverageNugget):
            _error(stage, f"nugget {index} must be a CoverageNugget")
        nugget_id = _text(nugget.nugget_id, f"nugget {index} ID", None, stage=stage)
        text = _text(nugget.text, f"nugget {index} text", None, stage=stage)
        if nugget_id in seen_ids:
            _error(stage, f"nugget {index} has a duplicate nugget ID")
        seen_ids.add(nugget_id)
        rows.append(CoverageNugget(nugget_id=nugget_id, text=text))
    if not rows:
        _error(stage, "at least one retrieval nugget is required")
    return tuple(rows)


__all__ = [
    "BackendReply",
    "CoverageFacet",
    "CoverageJudgment",
    "CoverageModelBackend",
    "CoverageModelRequest",
    "CoverageNugget",
    "CoverageObligation",
    "CoverageReport",
    "EvaluationCallMetadata",
    "EvaluationResult",
    "EvaluatorIdentity",
    "FrozenPlan",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "MAX_COMPLETION_TOKENS",
    "MAX_JUDGE_REQUEST_BYTES",
    "NuggetCoverageError",
    "PLANNER_PROMPT_VERSION",
    "PLANNER_SYSTEM_PROMPT",
    "evaluate_nugget_coverage",
    "render_judge_request",
    "render_planner_request",
    "score_coverage",
    "validate_and_freeze_plan",
    "validate_judgments",
]

"""Pure validation and scoring domain for retrieval nugget coverage.

The planner and judge payloads are untrusted model output.  This module keeps
their admission rules and the deterministic score independent of any transport
or persistence code so callers can test the evaluator without hosted calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
from types import MappingProxyType
from typing import Literal, NoReturn, Protocol
import unicodedata
from urllib.error import URLError

from trec_rag.facet_extraction import (
    BackendReply,
    FacetRequest,
    FacetResponse,
    MAX_RESPONSE_BYTES,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    _UrllibFacetTransport,
    _decode_json,
    _openrouter_completion,
)
from trec_rag.generation_handoff import (
    HandoffIntegrityError,
    load_generation_handoff,
    select_generation_topics,
)


PLAN_SCHEMA_VERSION = "retrieval_nugget_plan_v1"
JUDGMENT_SCHEMA_VERSION = "retrieval_nugget_judgment_v1"
MAX_FACETS = 12
MAX_OBLIGATIONS_PER_FACET = 8
MAX_OBLIGATIONS = 40
MAX_JUDGE_REQUEST_BYTES = 1_000_000
MAX_COMPLETION_TOKENS = 8192
MAX_NARRATIVE_SPANS_PER_OBLIGATION = 8
MAX_UNMAPPED_NARRATIVE_SPANS = 40
MAX_NARRATIVE_SPAN_CHARACTERS = 1000
PLANNER_PROMPT_VERSION = "retrieval_nugget_planner_v4"
JUDGE_PROMPT_VERSION = "retrieval_nugget_judge_v2"
EVALUATOR_SCHEMA_VERSION = "retrieval_nugget_coverage_v2"
INPUT_ARTIFACT_SCHEMA_VERSION = "retrieval_nugget_coverage_input_v1"
PLAN_ARTIFACT_SCHEMA_VERSION = "retrieval_nugget_coverage_plan_v1"
JUDGMENTS_ARTIFACT_SCHEMA_VERSION = "retrieval_nugget_coverage_judgments_v1"
REPORT_ARTIFACT_SCHEMA_VERSION = "retrieval_nugget_coverage_report_v1"
MANIFEST_ARTIFACT_SCHEMA_VERSION = "retrieval_nugget_coverage_manifest_v1"
DEFAULT_PLANNER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-sol"
PLANNER_SYSTEM_PROMPT = (
    "Create a complete, minimal obligation plan from the supplied narrative. "
    "Return only the requested JSON object. Every explicit request in the "
    "narrative must be represented. For each required_explicit obligation, "
    "narrative_spans must contain one or more exact, non-empty substrings copied "
    "verbatim from the narrative. For each supplemental_inferred obligation, "
    "narrative_spans must be an empty list. Every unmapped_narrative_spans entry "
    "must also be an exact narrative substring. Keep the total number of "
    "obligations across all facets at 40 or fewer. Include at least one "
    "required_explicit obligation; a plan with none is invalid and unscoreable. "
    "Spans may preserve exact surrounding whitespace and newline, carriage-return, "
    "or tab characters from the narrative, but they must contain at least one "
    "non-whitespace character. Use at most 8 spans per obligation, at most 40 "
    "unmapped spans, and at most 1000 characters per span."
)
JUDGE_SYSTEM_PROMPT = (
    "Judge each frozen obligation against the supplied retrieval nuggets. Use "
    "only the supplied text and return only the requested JSON object. Return "
    "exactly one judgment for every obligation ID, with no duplicates or "
    "missing rows. supporting_nugget_aliases may contain only the supplied "
    "aliases and must not repeat an alias. A full or partial label requires at "
    "least one supporting alias; an unsupported label requires an empty alias "
    "list. missing_elements must be empty for full and non-empty for partial or "
    "unsupported. Labels are exactly full, partial, or unsupported."
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


@dataclass(frozen=True)
class BoundCoverageInput:
    """The only handoff projection admitted to the coverage evaluator."""

    manifest_sha256: str
    topic_id: str
    narrative: str
    narrative_sha256: str
    nuggets: tuple[CoverageNugget, ...]
    nugget_text_sha256s: tuple[str, ...]
    input_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise NuggetCoverageError("input", "handoff manifest identity is invalid")
        if not isinstance(self.topic_id, str) or not self.topic_id:
            raise NuggetCoverageError("input", "topic identity is invalid")
        _sealed_text(self.narrative, "narrative", stage="input")
        if not isinstance(self.narrative_sha256, str) or len(self.narrative_sha256) != 64:
            raise NuggetCoverageError("input", "narrative identity is invalid")
        if self.narrative_sha256 != sha256(self.narrative.encode("utf-8")).hexdigest():
            raise NuggetCoverageError("input", "narrative identity does not match sealed text")
        rows = _validate_nuggets(self.nuggets, stage="input")
        if tuple(rows) != self.nuggets:
            object.__setattr__(self, "nuggets", rows)
        expected_nugget_hashes = tuple(
            sha256(row.text.encode("utf-8")).hexdigest() for row in rows
        )
        if self.nugget_text_sha256s != expected_nugget_hashes:
            raise NuggetCoverageError("input", "nugget text identities are invalid")
        if self.input_sha256:
            if self.input_sha256 != sha256(_canonical_json(_input_payload(self), stage="input")).hexdigest():
                raise NuggetCoverageError("input", "input identity does not match its fields")
        else:
            object.__setattr__(
                self,
                "input_sha256",
                sha256(_canonical_json(_input_payload(self), stage="input")).hexdigest(),
            )

    @property
    def handoff_manifest_sha256(self) -> str:
        return self.manifest_sha256

    @property
    def topic_narrative_sha256(self) -> str:
        return self.narrative_sha256

    @property
    def nugget_text_hashes(self) -> tuple[str, ...]:
        return self.nugget_text_sha256s

    @property
    def ordered_nugget_text_sha256s(self) -> tuple[str, ...]:
        return self.nugget_text_sha256s


@dataclass(frozen=True)
class CoverageRunConfig:
    """One-topic, cache-first coverage run configuration."""

    handoff_manifest_path: Path
    topic_id: str
    work_dir: Path | None = None
    planner_model: str = DEFAULT_PLANNER_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    mode: Literal["create", "resume"] = "create"
    allow_hosted_calls: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "handoff_manifest_path", Path(self.handoff_manifest_path))
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", Path(self.work_dir))
        if self.mode not in {"create", "resume"}:
            raise NuggetCoverageError("config", "mode must be create or resume")
        _validate_topic_id(self.topic_id, stage="config")
        _text(self.planner_model, "planner model", None, stage="config")
        _text(self.judge_model, "judge model", None, stage="config")

    @property
    def resolved_work_dir(self) -> Path:
        if self.work_dir is not None:
            return self.work_dir
        return self.handoff_manifest_path.parent / "retrieval_nugget_coverage" / self.topic_id


@dataclass(frozen=True)
class CoverageRunReceipt:
    """Secret-free receipt returned by a persisted coverage run."""

    status: str
    topic_id: str
    work_dir: Path
    artifact_hashes: Mapping[str, str]
    hosted_calls: int
    reused_stages: tuple[str, ...]
    obligation_count: int
    nugget_count: int
    required_coverage: float
    strict_full_rate: float
    supplemental_coverage: float | None

    @property
    def input_sha256(self) -> str | None:
        return self.artifact_hashes.get("input.json")

    @property
    def plan_sha256(self) -> str | None:
        return self.artifact_hashes.get("plan.json")

    @property
    def judgments_sha256(self) -> str | None:
        return self.artifact_hashes.get("judgments.json")

    @property
    def report_sha256(self) -> str | None:
        return self.artifact_hashes.get("report.json")

    @property
    def manifest_sha256(self) -> str | None:
        return self.artifact_hashes.get("manifest.json")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "status": self.status,
            "topic_id": self.topic_id,
            "work_dir": str(self.work_dir),
            "artifact_hashes": dict(self.artifact_hashes),
            "hosted_calls": self.hosted_calls,
            "reused_stages": list(self.reused_stages),
            "obligation_count": self.obligation_count,
            "nugget_count": self.nugget_count,
            "required_coverage": round(self.required_coverage, 4),
            "strict_full_rate": round(self.strict_full_rate, 4),
            "supplemental_coverage": (
                None if self.supplemental_coverage is None else round(self.supplemental_coverage, 4)
            ),
        }


@dataclass(frozen=True, slots=True)
class CompletedCoverageEvaluation:
    """Read-only, fully validated coverage state for report consumers."""

    bound_input: BoundCoverageInput
    identity: EvaluatorIdentity
    plan: FrozenPlan
    judgments: tuple[CoverageJudgment, ...]
    report: CoverageReport
    artifact_hashes: Mapping[str, str]
    manifest_sha256: str


class OpenRouterCoverageBackend:
    """OpenRouter adapter for one strict structured planner or judge call."""

    redirects_allowed = False
    semantic_retries = 0

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        api_key: str | None = None,
        transport: object | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        transport_max_attempts: int = 3,
    ) -> None:
        key = api_key
        if key is None:
            key = (os.environ if environ is None else environ).get("OPENROUTER_API_KEY")
        if not isinstance(key, str) or not key:
            raise ValueError("OPENROUTER_API_KEY must be set to non-empty text")
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ValueError("OpenRouter credentials require an HTTPS endpoint")
        if (
            isinstance(transport_max_attempts, bool)
            or not isinstance(transport_max_attempts, int)
            or transport_max_attempts < 1
        ):
            raise ValueError("transport_max_attempts must be a positive integer")
        self._api_key = key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._transport = _UrllibFacetTransport() if transport is None else transport
        self._transport_max_attempts = transport_max_attempts
        self._transport_invocation_count = 0

    @property
    def transport_invocation_count(self) -> int:
        return self._transport_invocation_count

    def complete(self, request: CoverageModelRequest) -> BackendReply:
        if not isinstance(request, CoverageModelRequest):
            raise TypeError("request must be a CoverageModelRequest")
        body = _provider_request_bytes(request)
        sent = FacetRequest(
            url=self._endpoint,
            body=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        for attempt in range(1, self._transport_max_attempts + 1):
            try:
                send = getattr(self._transport, "send", None)
                if not callable(send):
                    raise TypeError("transport must provide send(request)")
                self._transport_invocation_count += 1
                response = send(sent)
            except Exception as exc:
                if _is_transient_transport_error(exc) and attempt < self._transport_max_attempts:
                    continue
                raise RuntimeError("OpenRouter transport failed") from None
            if not isinstance(response, FacetResponse):
                raise RuntimeError("OpenRouter transport returned an invalid response")
            if response.status == 429 or response.status >= 500:
                if attempt < self._transport_max_attempts:
                    continue
                raise RuntimeError(f"OpenRouter returned HTTP {response.status}")
            if not 200 <= response.status < 300:
                raise RuntimeError(f"OpenRouter returned HTTP {response.status}")
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise RuntimeError("OpenRouter response exceeded byte cap")
            try:
                envelope = _decode_json(response.body, "OpenRouter response")
                if _contains_credential(envelope, self._api_key):
                    raise RuntimeError("OpenRouter response reflected a credential")
                completion = _openrouter_completion(envelope)
                content = completion.content.encode("utf-8")
                decoded_content = _decode_json(content, "OpenRouter completion")
                if _contains_credential(decoded_content, self._api_key):
                    raise RuntimeError("OpenRouter completion reflected a credential")
            except RuntimeError:
                raise
            except Exception:
                # Semantic response failures are deliberately never retried.
                raise RuntimeError("OpenRouter response failed validation") from None
            metadata = {
                "requested_model": request.model,
                "response_model": completion.response_model,
                "provider": completion.provider,
                "finish_reason": completion.finish_reason,
                "usage": dict(completion.usage),
            }
            return BackendReply(
                content=content,
                response_body=response.body,
                status=response.status,
                metadata=metadata,
                response_bodies=(response.body,),
                metadata_entries=(metadata,),
            )
        raise AssertionError("unreachable OpenRouter retry state")


def render_planner_request(
    narrative: str,
    model: str | EvaluatorIdentity,
    *,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> CoverageModelRequest:
    """Render the narrative-only planner request deterministically."""

    narrative = _sealed_text(narrative, "narrative", stage="planner")
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

    narrative = _sealed_text(narrative, "narrative", stage="judge")
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

    narrative = _sealed_text(narrative, "narrative", stage="input")
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


def coverage_input_from_handoff(path: Path, topic_id: str) -> BoundCoverageInput:
    """Authenticate one handoff and project exactly one topic into coverage input."""

    try:
        handoff = load_generation_handoff(Path(path))
        topics = select_generation_topics(handoff, [topic_id])
    except HandoffIntegrityError:
        raise NuggetCoverageError("handoff", "sealed generation handoff was rejected") from None
    except (OSError, TypeError, ValueError):
        raise NuggetCoverageError("handoff", "sealed generation handoff was rejected") from None
    if len(topics) != 1:
        raise NuggetCoverageError("input", "exactly one topic is required")
    topic = topics[0]
    if not topic.narrative:
        raise NuggetCoverageError("input", "topic narrative is empty")
    if not topic.claim_hints:
        raise NuggetCoverageError("input", "topic has no canonical retrieval nugget")
    # This intentionally reads only claim_id and text from each authenticated ClaimHint.
    nuggets = tuple(
        CoverageNugget(nugget_id=hint.claim_id, text=hint.text)
        for hint in topic.claim_hints
    )
    return BoundCoverageInput(
        manifest_sha256=handoff.manifest_sha256,
        topic_id=topic.topic_id,
        narrative=topic.narrative,
        narrative_sha256=topic.narrative_sha256,
        nuggets=nuggets,
        nugget_text_sha256s=tuple(
            sha256(nugget.text.encode("utf-8")).hexdigest() for nugget in nuggets
        ),
    )


def run_coverage_evaluation(
    config: CoverageRunConfig,
    planner: CoverageModelBackend | None = None,
    judge: CoverageModelBackend | None = None,
) -> CoverageRunReceipt:
    """Run or resume one private, hash-bound coverage evaluation."""

    if not isinstance(config, CoverageRunConfig):
        raise TypeError("config must be a CoverageRunConfig")
    bound = coverage_input_from_handoff(config.handoff_manifest_path, config.topic_id)
    identity = EvaluatorIdentity(
        schema_version=EVALUATOR_SCHEMA_VERSION,
        planner_prompt_version=PLANNER_PROMPT_VERSION,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        planner_model=config.planner_model,
        judge_model=config.judge_model,
    )
    work_dir = config.resolved_work_dir
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        raise NuggetCoverageError("persistence", "work directory is not a directory")
    if config.mode == "create":
        if _work_has_artifacts(work_dir):
            raise NuggetCoverageError("persistence", "create refuses a non-empty work directory")
        if not config.allow_hosted_calls:
            raise NuggetCoverageError(
                "cache", "cache-only is missing planner and judge stage(s)"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        _publish_once(work_dir / "input.json", _input_payload(bound))
        existing_manifest = None
        reused_stages: list[str] = []
    else:
        if not work_dir.is_dir():
            raise NuggetCoverageError("cache", "cache-only is missing planner and judge stages")
        _validate_artifact_order(work_dir)
        input_path = work_dir / "input.json"
        if not input_path.exists():
            raise NuggetCoverageError("cache", "cache-only is missing planner and judge stages")
        _validate_input_artifact(input_path, bound)
        existing_manifest = _load_artifact(work_dir / "manifest.json") if (work_dir / "manifest.json").exists() else None
        reused_stages = []

    plan: FrozenPlan | None = None
    judgments: tuple[CoverageJudgment, ...] | None = None
    planner_request_sha256: str | None = None
    judge_request_sha256: str | None = None
    planner_metadata: Mapping[str, object] = {}
    judge_metadata: Mapping[str, object] = {}
    hosted_calls = 0

    plan_path = work_dir / "plan.json"
    judgments_path = work_dir / "judgments.json"
    report_path = work_dir / "report.json"

    if existing_manifest is not None and not all(
        path.exists() for path in (plan_path, judgments_path, report_path)
    ):
        raise NuggetCoverageError(
            "persistence", "manifest exists without every preceding artifact"
        )

    if config.mode == "resume" and plan_path.exists():
        plan, planner_request_sha256, planner_metadata = _load_plan_artifact(
            plan_path, bound.narrative, identity
        )
        reused_stages.append("planner")
    if config.mode == "resume" and judgments_path.exists():
        if plan is None:
            raise NuggetCoverageError("persistence", "judgments artifact exists without a valid plan")
        judgments, judge_request_sha256, judge_metadata = _load_judgments_artifact(
            judgments_path, plan, bound.nuggets, identity
        )
        reused_stages.append("judge")

    if existing_manifest is not None:
        if plan is None or judgments is None:
            raise NuggetCoverageError(
                "persistence", "manifest exists without valid planner and judge stages"
            )
        existing_hashes = {
            name: _sha256_file(work_dir / name)
            for name in ("input.json", "plan.json", "judgments.json", "report.json")
        }
        _validate_manifest(
            existing_manifest,
            bound,
            identity,
            existing_hashes,
            planner_metadata,
            judge_metadata,
            _completed_stage_count(plan, judgments),
        )

    backend: CoverageModelBackend | None = None

    def get_backend() -> CoverageModelBackend:
        nonlocal backend
        if backend is None:
            if not config.allow_hosted_calls:
                missing = []
                if plan is None:
                    missing.append("planner")
                if judgments is None:
                    missing.append("judge")
                raise NuggetCoverageError(
                    "cache",
                    "cache-only is missing " + " and ".join(missing) + " stage(s)",
                )
            backend = OpenRouterCoverageBackend()
        return backend

    if plan is None:
        if not config.allow_hosted_calls and planner is not None:
            raise NuggetCoverageError("cache", "cache-only is missing planner and judge stage(s)")
        selected = planner if planner is not None else get_backend()
        request = render_planner_request(bound.narrative, identity)
        reply = _complete(selected, request, stage="planner")
        payload = _decode_model_payload(reply, stage="planner")
        plan = validate_and_freeze_plan(bound.narrative, payload)
        planner_request_sha256 = sha256(_serialized_provider_request(request, stage="planner")).hexdigest()
        planner_metadata = _safe_metadata(reply, stage="planner")
        _publish_once(
            plan_path,
            _plan_payload(plan, identity, planner_request_sha256, planner_metadata),
        )
        hosted_calls += 1
    if judgments is None:
        if not config.allow_hosted_calls and judge is not None:
            raise NuggetCoverageError("cache", "cache-only is missing judge stage(s)")
        selected = judge if judge is not None else get_backend()
        request = render_judge_request(bound.narrative, plan, bound.nuggets, identity)
        request_bytes = _serialized_provider_request(request, stage="judge")
        if len(request_bytes) > MAX_JUDGE_REQUEST_BYTES:
            raise NuggetCoverageError("judge", f"judge request exceeds the {MAX_JUDGE_REQUEST_BYTES:,}-byte limit")
        reply = _complete(selected, request, stage="judge")
        payload = _decode_model_payload(reply, stage="judge")
        judgments = validate_judgments(plan, bound.nuggets, payload)
        judge_request_sha256 = sha256(request_bytes).hexdigest()
        judge_metadata = _safe_metadata(reply, stage="judge")
        _publish_once(
            judgments_path,
            _judgments_payload(plan, judgments, identity, judge_request_sha256, judge_metadata),
        )
        hosted_calls += 1

    if plan is None or judgments is None:
        raise NuggetCoverageError("persistence", "coverage stages are incomplete")
    report = score_coverage(plan, bound.nuggets, judgments, identity)
    report_payload = _report_payload(report, plan)
    if report_path.exists():
        existing_report = _load_artifact(report_path)
        if existing_report != report_payload:
            raise NuggetCoverageError("persistence", "report artifact does not match validated stages")
    else:
        _publish_once(report_path, report_payload)

    artifact_hashes = {
        name: _sha256_file(work_dir / name)
        for name in ("input.json", "plan.json", "judgments.json", "report.json")
    }
    manifest_payload = _manifest_payload(
        bound,
        identity,
        artifact_hashes,
        planner_metadata,
        judge_metadata,
        _completed_stage_count(plan, judgments),
    )
    manifest_path = work_dir / "manifest.json"
    if existing_manifest is not None:
        _validate_manifest(
            existing_manifest,
            bound,
            identity,
            artifact_hashes,
            planner_metadata,
            judge_metadata,
            _completed_stage_count(plan, judgments),
        )
    else:
        _publish_once(manifest_path, manifest_payload)
    artifact_hashes["manifest.json"] = _sha256_file(manifest_path)
    return CoverageRunReceipt(
        status="complete",
        topic_id=bound.topic_id,
        work_dir=work_dir,
        artifact_hashes=MappingProxyType(dict(artifact_hashes)),
        hosted_calls=hosted_calls,
        reused_stages=tuple(reused_stages),
        obligation_count=len(plan.obligations),
        nugget_count=len(bound.nuggets),
        required_coverage=report.required_coverage,
        strict_full_rate=report.strict_full_rate,
        supplemental_coverage=report.supplemental_coverage,
    )


def load_completed_coverage_evaluation(
    *,
    handoff_manifest_path: Path,
    topic_id: str,
    work_dir: Path,
) -> CompletedCoverageEvaluation:
    """Validate and load a complete coverage bundle without writing or calling a backend."""

    bound = coverage_input_from_handoff(Path(handoff_manifest_path), topic_id)
    work_root = Path(work_dir)
    if work_root.is_symlink() or (work_root.exists() and not work_root.is_dir()):
        raise NuggetCoverageError("persistence", "work directory is not a directory")
    if not work_root.is_dir():
        raise NuggetCoverageError("persistence", "completed coverage bundle is missing")

    required_names = ("input.json", "plan.json", "judgments.json", "report.json", "manifest.json")
    required_paths = {name: work_root / name for name in required_names}
    for name, path in required_paths.items():
        if path.is_symlink():
            raise NuggetCoverageError("persistence", f"{name} is a symbolic link")
        if not path.is_file():
            raise NuggetCoverageError("persistence", f"completed coverage bundle is missing {name}")

    manifest_payload = _load_artifact(required_paths["manifest.json"])
    identity = _decode_manifest_identity(manifest_payload)
    _validate_input_artifact(required_paths["input.json"], bound)
    plan, _planner_request_sha256, planner_metadata = _load_plan_artifact(
        required_paths["plan.json"], bound.narrative, identity
    )
    judgments, _judge_request_sha256, judge_metadata = _load_judgments_artifact(
        required_paths["judgments.json"], plan, bound.nuggets, identity
    )
    report = score_coverage(plan, bound.nuggets, judgments, identity)
    expected_report = _report_payload(report, plan)
    if _load_artifact(required_paths["report.json"]) != expected_report:
        raise NuggetCoverageError("persistence", "report artifact does not match validated stages")

    artifact_hashes = {
        name: _sha256_file(required_paths[name])
        for name in ("input.json", "plan.json", "judgments.json", "report.json")
    }
    _validate_manifest(
        manifest_payload,
        bound,
        identity,
        artifact_hashes,
        planner_metadata,
        judge_metadata,
        2,
    )
    return CompletedCoverageEvaluation(
        bound_input=bound,
        identity=identity,
        plan=plan,
        judgments=judgments,
        report=report,
        artifact_hashes=MappingProxyType(dict(artifact_hashes)),
        manifest_sha256=_sha256_file(required_paths["manifest.json"]),
    )


def _input_payload(bound: BoundCoverageInput) -> dict[str, object]:
    return {
        "schema_version": INPUT_ARTIFACT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_sha256": bound.manifest_sha256,
        "topic_id": bound.topic_id,
        "narrative_sha256": bound.narrative_sha256,
        "nuggets": [
            {
                "alias": f"n{index:03d}",
                "claim_id": nugget.nugget_id,
                "text_sha256": bound.nugget_text_sha256s[index - 1],
            }
            for index, nugget in enumerate(bound.nuggets, start=1)
        ],
    }


def _identity_payload(identity: EvaluatorIdentity) -> dict[str, str]:
    return {
        "schema_version": identity.schema_version,
        "planner_prompt_version": identity.planner_prompt_version,
        "judge_prompt_version": identity.judge_prompt_version,
        "planner_model": identity.planner_model,
        "judge_model": identity.judge_model,
    }


def _plan_payload(
    plan: FrozenPlan,
    identity: EvaluatorIdentity,
    request_sha256: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": PLAN_ARTIFACT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "identity": _identity_payload(identity),
        "request_sha256": request_sha256,
        "plan_sha256": plan.plan_sha256,
        "plan": json.loads(plan.canonical_bytes),
        "provider_metadata": _safe_provider_metadata(metadata),
    }


def _judgments_payload(
    plan: FrozenPlan,
    judgments: Sequence[CoverageJudgment],
    identity: EvaluatorIdentity,
    request_sha256: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": JUDGMENTS_ARTIFACT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "identity": _identity_payload(identity),
        "plan_sha256": plan.plan_sha256,
        "request_sha256": request_sha256,
        "judgments": [
            {
                "obligation_id": row.obligation_id,
                "label": row.label,
                "supporting_nugget_ids": list(row.supporting_nugget_ids),
                "missing_elements": row.missing_elements,
            }
            for row in judgments
        ],
        "provider_metadata": _safe_provider_metadata(metadata),
    }


def _report_payload(report: CoverageReport, plan: FrozenPlan) -> dict[str, object]:
    return {
        "schema_version": REPORT_ARTIFACT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "plan_sha256": report.plan_sha256,
        "identity": _identity_payload(report.identity),
        "required_coverage": report.required_coverage,
        "strict_full_rate": report.strict_full_rate,
        "supplemental_coverage": report.supplemental_coverage,
        "label_counts": dict(report.label_counts),
        "facet_scores": dict(report.facet_scores),
        "judgments": [
            {
                "obligation_id": judgment.obligation_id,
                "label": judgment.label,
                "supporting_nugget_ids": list(judgment.supporting_nugget_ids),
                "missing_elements": judgment.missing_elements,
            }
            for judgment in report.judgments
        ],
        "obligations": [
            {
                "obligation_id": obligation.obligation_id,
                "requirement": obligation.requirement,
                "support_test": obligation.support_test,
                "kind": obligation.kind,
                "narrative_spans": list(obligation.narrative_spans),
            }
            for obligation in plan.obligations
        ],
        "unmapped_narrative_spans": list(report.unmapped_narrative_spans),
        "uncited_nugget_ids": list(report.uncited_nugget_ids),
        "uncited_nugget_aliases": list(report.uncited_nugget_aliases),
        "core_assumption": (
            "Canonical retrieval nuggets are assumed to faithfully represent the selected passages "
            "from which they were derived; this evaluator does not reopen passages."
        ),
        "limits": [
            "The score measures coverage of a planner-derived plan, not ground truth.",
            "It never opens selected passages and cannot detect a nugget that misstates its source.",
            "A low score cannot separate retrieval, selection, and canonicalization failures.",
            "Scores are not comparable across evaluator schemas, prompts, or model identities.",
        ],
    }


def _manifest_payload(
    bound: BoundCoverageInput,
    identity: EvaluatorIdentity,
    artifact_hashes: Mapping[str, str],
    planner_metadata: Mapping[str, object],
    judge_metadata: Mapping[str, object],
    completed_stages: int,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_ARTIFACT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "identity": _identity_payload(identity),
        "topic_id": bound.topic_id,
        "manifest_sha256": bound.manifest_sha256,
        "narrative_sha256": bound.narrative_sha256,
        "nugget_text_sha256s": list(bound.nugget_text_sha256s),
        "artifact_hashes": dict(artifact_hashes),
        "completed_stages": completed_stages,
        "provider_metadata": {
            "planner": _safe_provider_metadata(planner_metadata),
            "judge": _safe_provider_metadata(judge_metadata),
        },
    }


def _completed_stage_count(
    plan: FrozenPlan | None, judgments: Sequence[CoverageJudgment] | None
) -> int:
    """Return deterministic completed-stage provenance for the sealed manifest."""

    return int(plan is not None) + int(judgments is not None)


def _safe_provider_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    safe: dict[str, object] = {}
    allowed = {"requested_model", "response_model", "provider", "finish_reason", "usage"}
    for key in allowed:
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "usage" and isinstance(value, Mapping):
            safe[key] = {
                usage_key: usage_value
                for usage_key, usage_value in value.items()
                if usage_key in {
                    "prompt_tokens", "completion_tokens", "total_tokens", "cost", "is_byok"
                }
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe


def _validate_request_digest(
    value: object, expected: str, *, stage: Literal["planner", "judge"]
) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NuggetCoverageError("persistence", f"{stage} request identity is invalid")
    if value != expected:
        raise NuggetCoverageError("persistence", f"{stage} request identity does not match")


def _publish_once(path: Path, payload: Mapping[str, object]) -> str:
    """Publish canonical bytes once, refusing contradictory state."""

    path = Path(path)
    body = _canonical_json(payload, stage="persistence")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise NuggetCoverageError("persistence", "artifact destination is a symbolic link")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError:
            raise NuggetCoverageError("persistence", "artifact destination cannot be read") from None
        if existing != body:
            raise NuggetCoverageError("persistence", "artifact destination contains different bytes")
        return sha256(body).hexdigest()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != body:
                raise NuggetCoverageError("persistence", "artifact destination contains different bytes") from None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return sha256(body).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_artifact(path: Path) -> Mapping[str, object]:
    try:
        source = Path(path).read_bytes()
        payload = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise NuggetCoverageError("persistence", "artifact is not valid canonical JSON") from None
    if not isinstance(payload, Mapping) or _canonical_json(payload, stage="persistence") != source:
        raise NuggetCoverageError("persistence", "artifact bytes are not canonical")
    return payload


def _decode_manifest_identity(payload: Mapping[str, object]) -> EvaluatorIdentity:
    """Decode the exact, current evaluator identity sealed in a manifest."""

    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or not all(isinstance(key, str) for key in identity):
        raise NuggetCoverageError("persistence", "manifest identity is invalid")
    expected_keys = {
        "schema_version",
        "planner_prompt_version",
        "judge_prompt_version",
        "planner_model",
        "judge_model",
    }
    if set(identity) != expected_keys:
        raise NuggetCoverageError("persistence", "manifest identity has unexpected or missing fields")
    if (
        identity.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or identity.get("planner_prompt_version") != PLANNER_PROMPT_VERSION
        or identity.get("judge_prompt_version") != JUDGE_PROMPT_VERSION
    ):
        raise NuggetCoverageError("persistence", "manifest identity does not match evaluator contract")
    try:
        planner_model = _text(identity.get("planner_model"), "planner model", None, stage="persistence")
        judge_model = _text(identity.get("judge_model"), "judge model", None, stage="persistence")
    except NuggetCoverageError:
        raise NuggetCoverageError("persistence", "manifest identity contains an invalid model") from None
    return EvaluatorIdentity(
        schema_version=EVALUATOR_SCHEMA_VERSION,
        planner_prompt_version=PLANNER_PROMPT_VERSION,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        planner_model=planner_model,
        judge_model=judge_model,
    )


def _sha256_file(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise NuggetCoverageError("persistence", "artifact is unreadable") from None


def _validate_input_artifact(path: Path, bound: BoundCoverageInput) -> None:
    payload = _load_artifact(path)
    if payload != _input_payload(bound):
        raise NuggetCoverageError(
            "persistence",
            "input artifact is stale for the current evaluator contract or authenticated handoff projection",
        )


def _load_plan_artifact(
    path: Path, narrative: str, identity: EvaluatorIdentity
) -> tuple[FrozenPlan, str, Mapping[str, object]]:
    payload = _load_artifact(path)
    if (
        payload.get("schema_version") != PLAN_ARTIFACT_SCHEMA_VERSION
        or payload.get("evaluator_schema_version") != EVALUATOR_SCHEMA_VERSION
        or payload.get("identity") != _identity_payload(identity)
    ):
        raise NuggetCoverageError("persistence", "planner artifact identity does not match")
    plan_payload = payload.get("plan")
    if not isinstance(plan_payload, Mapping):
        raise NuggetCoverageError("persistence", "planner artifact plan is invalid")
    try:
        raw_plan_payload = {
            "schema_version": plan_payload["schema_version"],
            "facets": [
                {
                    "title": facet["title"],
                    "obligations": [
                        {
                            key: obligation[key]
                            for key in ("requirement", "support_test", "kind", "narrative_spans")
                        }
                        for obligation in facet["obligations"]
                    ],
                }
                for facet in plan_payload["facets"]
            ],
            "unmapped_narrative_spans": plan_payload["unmapped_narrative_spans"],
        }
    except (KeyError, TypeError):
        raise NuggetCoverageError("persistence", "planner artifact plan is invalid") from None
    try:
        plan = validate_and_freeze_plan(narrative, raw_plan_payload)
    except NuggetCoverageError:
        raise NuggetCoverageError("persistence", "planner artifact plan is invalid") from None
    if json.loads(plan.canonical_bytes) != dict(plan_payload):
        raise NuggetCoverageError("persistence", "planner artifact plan is not canonical")
    if payload.get("plan_sha256") != plan.plan_sha256:
        raise NuggetCoverageError("persistence", "planner artifact hash does not match")
    request_sha256 = payload.get("request_sha256")
    expected_request_sha256 = sha256(
        _serialized_provider_request(
            render_planner_request(narrative, identity), stage="planner"
        )
    ).hexdigest()
    _validate_request_digest(request_sha256, expected_request_sha256, stage="planner")
    metadata = payload.get("provider_metadata")
    return plan, request_sha256, _safe_provider_metadata(metadata if isinstance(metadata, Mapping) else {})


def _load_judgments_artifact(
    path: Path,
    plan: FrozenPlan,
    nuggets: Sequence[CoverageNugget],
    identity: EvaluatorIdentity,
) -> tuple[tuple[CoverageJudgment, ...], str, Mapping[str, object]]:
    payload = _load_artifact(path)
    if (
        payload.get("schema_version") != JUDGMENTS_ARTIFACT_SCHEMA_VERSION
        or payload.get("evaluator_schema_version") != EVALUATOR_SCHEMA_VERSION
        or payload.get("identity") != _identity_payload(identity)
        or payload.get("plan_sha256") != plan.plan_sha256
    ):
        raise NuggetCoverageError("persistence", "judge artifact identity does not match")
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        raise NuggetCoverageError("persistence", "judge artifact judgments are invalid")
    nugget_aliases = {nugget.nugget_id: f"n{index:03d}" for index, nugget in enumerate(nuggets, start=1)}
    known_nugget_ids = set(nugget_aliases)
    for row in rows:
        if not isinstance(row, Mapping):
            raise NuggetCoverageError("persistence", "judge artifact judgments are invalid")
        persisted_ids = row.get("supporting_nugget_ids")
        if not isinstance(persisted_ids, list) or any(
            not isinstance(nugget_id, str) or nugget_id not in known_nugget_ids
            for nugget_id in persisted_ids
        ):
            raise NuggetCoverageError(
                "persistence", "judge artifact contains an unknown nugget ID"
            )
    canonical_rows = {
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "judgments": [
            {
                "obligation_id": row.get("obligation_id") if isinstance(row, Mapping) else None,
                "label": row.get("label") if isinstance(row, Mapping) else None,
                "supporting_nugget_aliases": [
                    nugget_aliases[nugget_id]
                    for nugget_id in (row.get("supporting_nugget_ids", []) if isinstance(row, Mapping) else [])
                    if nugget_id in nugget_aliases
                ],
                "missing_elements": row.get("missing_elements") if isinstance(row, Mapping) else None,
            }
            for row in rows
        ],
    }
    # Validate the persisted canonical IDs through the same semantic validator.
    try:
        parsed = validate_judgments(plan, nuggets, canonical_rows)
    except NuggetCoverageError:
        raise NuggetCoverageError(
            "persistence", "judge artifact judgments are invalid"
        ) from None
    expected_rows = [
        {
            "obligation_id": row.obligation_id,
            "label": row.label,
            "supporting_nugget_ids": list(row.supporting_nugget_ids),
            "missing_elements": row.missing_elements,
        }
        for row in parsed
    ]
    if rows != expected_rows:
        raise NuggetCoverageError("persistence", "judge artifact judgments are not canonical")
    request_sha256 = payload.get("request_sha256")
    try:
        expected_request_sha256 = sha256(
            _serialized_provider_request(
                render_judge_request(plan.narrative, plan, nuggets, identity),
                stage="judge",
            )
        ).hexdigest()
    except NuggetCoverageError:
        raise NuggetCoverageError("persistence", "judge request identity is invalid") from None
    _validate_request_digest(request_sha256, expected_request_sha256, stage="judge")
    metadata = payload.get("provider_metadata")
    return parsed, request_sha256, _safe_provider_metadata(metadata if isinstance(metadata, Mapping) else {})


def _validate_manifest(
    payload: Mapping[str, object],
    bound: BoundCoverageInput,
    identity: EvaluatorIdentity,
    artifact_hashes: Mapping[str, str],
    planner_metadata: Mapping[str, object],
    judge_metadata: Mapping[str, object],
    completed_stages: int,
) -> None:
    expected = _manifest_payload(
        bound,
        identity,
        artifact_hashes,
        planner_metadata,
        judge_metadata,
        completed_stages,
    )
    if dict(payload) != expected:
        raise NuggetCoverageError("persistence", "manifest identity or artifact hash does not match")


def _work_has_artifacts(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def _validate_artifact_order(path: Path) -> None:
    """Reject impossible persisted stage order before loading or backend calls."""

    artifact_paths = [
        path / name
        for name in ("input.json", "plan.json", "judgments.json", "report.json", "manifest.json")
    ]
    missing_predecessor = False
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            if missing_predecessor:
                raise NuggetCoverageError(
                    "persistence", "persisted artifacts are out of stage order"
                )
        else:
            missing_predecessor = True


def _provider_request_bytes(request: CoverageModelRequest) -> bytes:
    return _canonical_json(_request_payload(request), stage=request.stage)


def _request_payload(request: CoverageModelRequest) -> dict[str, object]:
    """Build the deterministic, privacy-constrained OpenRouter request body."""

    return {
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
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False},
        "seed": 0,
        "stream": False,
    }


def _contains_credential(value: object, credential: str) -> bool:
    if isinstance(value, str):
        return credential in value
    if isinstance(value, Mapping):
        return any(_contains_credential(key, credential) or _contains_credential(item, credential) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_credential(item, credential) for item in value)
    return False


def _is_transient_transport_error(error: BaseException) -> bool:
    """Classify only known transient network causes for bounded retries."""

    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, (TimeoutError, ConnectionError)):
        return True
    return isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN


def validate_and_freeze_plan(narrative: str, payload: object) -> FrozenPlan:
    """Validate one planner payload and return a deterministic immutable plan."""

    narrative = _sealed_text(narrative, "narrative", stage="planner")
    root = _mapping(payload, "planner payload", stage="planner")
    _require_keys(root, _PLAN_ROOT_KEYS, "planner payload", stage="planner")
    if root["schema_version"] != PLAN_SCHEMA_VERSION:
        _error("planner", "schema_version must be retrieval_nugget_plan_v1")

    facets_payload = _list(root["facets"], "facets", 1, MAX_FACETS, stage="planner")
    unmapped_payload = _list(
        root["unmapped_narrative_spans"],
        "unmapped_narrative_spans",
        0,
        MAX_UNMAPPED_NARRATIVE_SPANS,
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
                MAX_NARRATIVE_SPANS_PER_OBLIGATION,
                stage="planner",
            )
            spans: list[str] = []
            for span_index, span_value in enumerate(spans_payload, start=1):
                span = _sealed_text(
                    span_value,
                    f"facet {facet_index} obligation {obligation_index} narrative span {span_index}",
                    stage="planner",
                )
                if len(span) > MAX_NARRATIVE_SPAN_CHARACTERS:
                    _error(
                        "planner",
                        f"facet {facet_index} obligation {obligation_index} narrative span {span_index} exceeds the {MAX_NARRATIVE_SPAN_CHARACTERS}-character limit",
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
        span = _sealed_text(value, f"unmapped narrative span {index}", stage="planner")
        if len(span) > MAX_NARRATIVE_SPAN_CHARACTERS:
            _error(
                "planner",
                f"unmapped narrative span {index} exceeds the {MAX_NARRATIVE_SPAN_CHARACTERS}-character limit",
            )
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

    if not required_values:
        _error("scoring", "plan must contain at least one required obligation")
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
    uncited_id_set = set(uncited_ids)
    uncited_aliases = tuple(
        f"n{index:03d}"
        for index, nugget in enumerate(nugget_rows, start=1)
        if nugget.nugget_id in uncited_id_set
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
        "description": (
            "A required_explicit obligation needs exact narrative substrings; a "
            "supplemental_inferred obligation must have no narrative spans. Exact "
            "surrounding whitespace and newline, carriage-return, or tab characters "
            "may be preserved, but every span must contain at least one "
            "non-whitespace character."
        ),
        "additionalProperties": False,
        "required": ["requirement", "support_test", "kind", "narrative_spans"],
        "properties": {
            "requirement": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARACTERS},
            "support_test": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARACTERS},
            "kind": {
                "type": "string",
                "enum": ["required_explicit", "supplemental_inferred"],
                "description": (
                    "required_explicit has one or more exact narrative substrings; "
                    "supplemental_inferred has an empty narrative_spans list."
                ),
            },
            "narrative_spans": {
                "type": "array",
                "maxItems": MAX_NARRATIVE_SPANS_PER_OBLIGATION,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_NARRATIVE_SPAN_CHARACTERS,
                },
                "description": "For required_explicit, copy exact nonblank narrative substrings (including exact surrounding whitespace or permitted line controls) with at most 8 spans and at most 1000 characters per span; for supplemental_inferred, use an empty list.",
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
        "description": (
            "Every span must be an exact narrative substring, and the total "
            "number of obligations across all facets must be at most 40. The "
            "plan must contain at least one required_explicit obligation. Each "
            "span is at most 1000 characters and unmapped_narrative_spans has at "
            "most 40 entries."
        ),
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
                "maxItems": MAX_UNMAPPED_NARRATIVE_SPANS,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_NARRATIVE_SPAN_CHARACTERS,
                },
                "description": "Each entry must be an exact nonblank substring of the supplied narrative, may preserve exact surrounding whitespace or permitted line controls, is at most 1000 characters, and there are at most 40 entries.",
            },
        },
    }


def _judge_response_schema(plan: FrozenPlan, nugget_count: int) -> dict[str, object]:
    obligation_ids = [obligation.obligation_id for obligation in plan.obligations]
    aliases = [f"n{index:03d}" for index in range(1, nugget_count + 1)]
    judgment = {
        "type": "object",
        "description": (
            "Return exactly one row per obligation. Full and partial require a "
            "supporting alias; unsupported requires an empty alias list."
        ),
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
                "description": "full, partial, and unsupported are the only labels; unsupported has no supporting aliases.",
            },
            "supporting_nugget_aliases": {
                "type": "array",
                "minItems": 0,
                "maxItems": nugget_count,
                "items": {"type": "string", "enum": aliases},
                "description": "Use only supplied aliases, without duplicates; full and partial require at least one.",
            },
            "missing_elements": {
                "type": "string",
                "maxLength": MAX_TEXT_CHARACTERS,
                "description": "Use an empty string for full and non-empty text for partial or unsupported.",
            },
        },
    }
    return {
        "type": "object",
        "description": "Return exactly one judgment for every frozen obligation ID, with no missing or duplicate rows.",
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
    return _canonical_json(_request_payload(request), stage=stage)


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
    return MappingProxyType(_safe_provider_metadata(reply.metadata))


def _error(stage: str, reason: str) -> NoReturn:
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


def _sealed_text(value: object, name: str, *, stage: str) -> str:
    """Validate authenticated source text without changing its sealed bytes."""

    if not isinstance(value, str) or not value or not value.strip():
        _error(stage, f"{name} must be non-empty text")
    for character in value:
        category = unicodedata.category(character)
        if category == "Cc" and character not in "\n\r\t":
            _error(stage, f"{name} contains unsafe control characters")
    return value


def _validate_topic_id(value: object, *, stage: str) -> str:
    topic_id = _text(value, "topic", None, stage=stage)
    if (
        topic_id in {".", ".."}
        or "/" in topic_id
        or "\\" in topic_id
        or Path(topic_id).is_absolute()
    ):
        _error(stage, "topic is unsafe for a work-directory path")
    return topic_id


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
        text = _sealed_text(nugget.text, f"nugget {index} text", stage=stage)
        if nugget_id in seen_ids:
            _error(stage, f"nugget {index} has a duplicate nugget ID")
        seen_ids.add(nugget_id)
        rows.append(CoverageNugget(nugget_id=nugget_id, text=text))
    if not rows:
        _error(stage, "at least one retrieval nugget is required")
    return tuple(rows)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise NuggetCoverageError("config", "invalid command-line arguments")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Evaluate one topic's canonical retrieval nugget coverage")
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--mode", choices=("create", "resume"), default="create")
    parser.add_argument("--allow-hosted-calls", action="store_true")
    return parser


def _safe_cli_error(error: Exception) -> dict[str, object]:
    if isinstance(error, NuggetCoverageError):
        stage = error.stage
        if stage == "cache":
            if "planner and judge" in error.reason:
                reason = "missing planner and judge stages"
            elif "planner" in error.reason:
                reason = "missing planner stage"
            elif "judge" in error.reason:
                reason = "missing judge stage"
            else:
                reason = "required cache stage is missing"
        else:
            reason = {
                "handoff": "sealed handoff rejected",
                "input": "coverage input rejected",
                "planner": "planner stage failed",
                "judge": "judge stage failed",
                "persistence": "private artifact state is invalid",
                "config": "run configuration is invalid",
            }.get(stage, "coverage evaluation failed")
    else:
        stage = "run"
        reason = "coverage evaluation failed"
    return {"error": {"type": "retrieval_nugget_coverage_error", "stage": stage, "reason": reason}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    try:
        arguments = parser.parse_args(argv)
        config = CoverageRunConfig(
            handoff_manifest_path=arguments.handoff_manifest,
            topic_id=arguments.topic,
            work_dir=arguments.work_dir,
            planner_model=arguments.planner_model,
            judge_model=arguments.judge_model,
            mode=arguments.mode,
            allow_hosted_calls=arguments.allow_hosted_calls,
        )
        receipt = run_coverage_evaluation(config)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps(_safe_cli_error(exc), sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(receipt.to_payload(), sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "BackendReply",
    "CoverageFacet",
    "CoverageJudgment",
    "CoverageModelBackend",
    "CoverageModelRequest",
    "CoverageNugget",
    "CoverageObligation",
    "CoverageReport",
    "CoverageRunConfig",
    "CoverageRunReceipt",
    "CompletedCoverageEvaluation",
    "BoundCoverageInput",
    "EvaluationCallMetadata",
    "EvaluationResult",
    "EvaluatorIdentity",
    "FrozenPlan",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "MAX_COMPLETION_TOKENS",
    "MAX_JUDGE_REQUEST_BYTES",
    "MAX_NARRATIVE_SPANS_PER_OBLIGATION",
    "MAX_NARRATIVE_SPAN_CHARACTERS",
    "MAX_UNMAPPED_NARRATIVE_SPANS",
    "NuggetCoverageError",
    "OpenRouterCoverageBackend",
    "PLANNER_PROMPT_VERSION",
    "PLANNER_SYSTEM_PROMPT",
    "evaluate_nugget_coverage",
    "coverage_input_from_handoff",
    "load_completed_coverage_evaluation",
    "run_coverage_evaluation",
    "main",
    "render_judge_request",
    "render_planner_request",
    "score_coverage",
    "validate_and_freeze_plan",
    "validate_judgments",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import socket
import ssl
from urllib.error import URLError

import pytest

from trec_rag.facet_extraction import FacetResponse
from trec_rag.generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    SelectedCluster,
    TopicSourceReceipts,
    write_generation_handoff,
)
import trec_rag.retrieval_nugget_coverage as coverage_module
from trec_rag.retrieval_nugget_coverage import (
    BackendReply,
    CoverageNugget,
    CoverageModelRequest,
    CoverageRunConfig,
    CoverageRunReceipt,
    EvaluatorIdentity,
    FrozenPlan,
    NuggetCoverageError,
    OpenRouterCoverageBackend,
    BoundCoverageInput,
    coverage_input_from_handoff,
    evaluate_nugget_coverage,
    load_completed_coverage_evaluation,
    main,
    render_judge_request,
    render_planner_request,
    seed_coverage_plan_from_completed_baseline,
    run_coverage_evaluation,
    score_coverage,
    validate_and_freeze_plan,
    validate_judgments,
)


NARRATIVE = (
    "Explain the projected cost and assumptions, and add useful historical context."
)

VALID_PLAN = {
    "schema_version": "retrieval_nugget_plan_v1",
    "facets": [
        {
            "title": "Costs and assumptions",
            "obligations": [
                {
                    "requirement": "Describe the projected cost and its assumptions.",
                    "support_test": "A figure and the assumptions used to derive it are present.",
                    "kind": "required_explicit",
                    "narrative_spans": ["cost and assumptions"],
                },
                {
                    "requirement": "Add useful historical context.",
                    "support_test": "A relevant historical comparison is present.",
                    "kind": "supplemental_inferred",
                    "narrative_spans": [],
                },
            ],
        }
    ],
    "unmapped_narrative_spans": [],
}

VALID_JUDGMENTS = {
    "schema_version": "retrieval_nugget_judgment_v1",
    "judgments": [
        {
            "obligation_id": "f001-o001",
            "label": "partial",
            "supporting_nugget_aliases": ["n001"],
            "missing_elements": "The assumptions are missing.",
        },
        {
            "obligation_id": "f001-o002",
            "label": "full",
            "supporting_nugget_aliases": ["n002"],
            "missing_elements": "",
        },
    ],
}


def _nuggets() -> tuple[CoverageNugget, ...]:
    return (
        CoverageNugget("claim-cost", "The projected cost is $10 million."),
        CoverageNugget("claim-history", "Historically, the comparable cost was lower."),
        CoverageNugget("claim-unused", "An uncited related detail."),
    )


def _identity() -> EvaluatorIdentity:
    return EvaluatorIdentity(
        schema_version="test-schema",
        planner_prompt_version="planner-v1",
        judge_prompt_version="judge-v1",
        planner_model="planner-model",
        judge_model="judge-model",
    )


def test_current_default_identity_uses_sol_and_v2_schema(tmp_path: Path) -> None:
    config = CoverageRunConfig(tmp_path / "handoff.json", "topic-defaults")

    assert coverage_module.EVALUATOR_SCHEMA_VERSION == "retrieval_nugget_coverage_v2"
    assert config.planner_model == "openai/gpt-5.6-sol"
    assert config.judge_model == "openai/gpt-5.6-sol"


def test_plan_validation_assigns_local_ids_and_freezes_deterministically() -> None:
    first = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    second = validate_and_freeze_plan(NARRATIVE, json.loads(json.dumps(VALID_PLAN)))

    assert isinstance(first, FrozenPlan)
    assert first == second
    assert first.facets[0].facet_id == "f001"
    assert first.facets[0].obligations[0].obligation_id == "f001-o001"
    assert first.facets[0].obligations[1].obligation_id == "f001-o002"
    assert first.facets[0].obligations[0].kind == "required_explicit"
    assert first.facets[0].obligations[0].narrative_spans == ("cost and assumptions",)
    assert first.facets[0].obligations[1].narrative_spans == ()
    assert first.required_obligation_count == 1
    assert first.canonical_bytes == second.canonical_bytes
    assert first.plan_sha256 == hashlib.sha256(first.canonical_bytes).hexdigest()
    assert json.loads(first.canonical_bytes) == {
        "facets": [
            {
                "facet_id": "f001",
                "obligations": [
                    {
                        "kind": "required_explicit",
                        "narrative_spans": ["cost and assumptions"],
                        "obligation_id": "f001-o001",
                        "requirement": "Describe the projected cost and its assumptions.",
                        "support_test": "A figure and the assumptions used to derive it are present.",
                    },
                    {
                        "kind": "supplemental_inferred",
                        "narrative_spans": [],
                        "obligation_id": "f001-o002",
                        "requirement": "Add useful historical context.",
                        "support_test": "A relevant historical comparison is present.",
                    },
                ],
                "title": "Costs and assumptions",
            }
        ],
        "schema_version": "retrieval_nugget_plan_v1",
        "unmapped_narrative_spans": [],
    }


@pytest.mark.parametrize(
    ("narrative", "span"),
    [
        ("Explain the projected cost\nand assumptions now.", "cost\nand assumptions"),
        ("Explain the projected  cost and assumptions  now.", "  cost and assumptions  "),
    ],
)
def test_plan_freeze_accepts_exact_multiline_and_padded_narrative_spans(
    narrative: str, span: str
) -> None:
    payload = json.loads(json.dumps(VALID_PLAN))
    payload["facets"][0]["obligations"][0]["narrative_spans"] = [span]

    plan = validate_and_freeze_plan(narrative, payload)

    assert plan.facets[0].obligations[0].narrative_spans == (span,)


def test_plan_validation_rejects_more_than_eight_narrative_spans_per_obligation() -> None:
    spans = [f"span-{index}" for index in range(9)]
    payload = json.loads(json.dumps(VALID_PLAN))
    payload["facets"][0]["obligations"][0]["narrative_spans"] = spans

    with pytest.raises(NuggetCoverageError, match="narrative_spans"):
        validate_and_freeze_plan(" ".join(spans), payload)


def test_plan_validation_rejects_more_than_forty_unmapped_narrative_spans() -> None:
    unmapped = [f"unmapped-{index}" for index in range(41)]
    payload = json.loads(json.dumps(VALID_PLAN))
    payload["unmapped_narrative_spans"] = unmapped

    with pytest.raises(NuggetCoverageError, match="unmapped"):
        validate_and_freeze_plan("cost and assumptions " + " ".join(unmapped), payload)


def test_plan_validation_rejects_narrative_span_over_one_thousand_characters() -> None:
    oversized = "x" * 1001
    payload = json.loads(json.dumps(VALID_PLAN))
    payload["facets"][0]["obligations"][0]["narrative_spans"] = [oversized]

    with pytest.raises(NuggetCoverageError, match="narrative span"):
        validate_and_freeze_plan(oversized, payload)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda plan: {**plan, "extra": True},
        lambda plan: {**plan, "facets": []},
        lambda plan: {**plan, "facets": [{"title": "", "obligations": plan["facets"][0]["obligations"]}]},
        lambda plan: {
            **plan,
            "facets": [
                {
                    **plan["facets"][0],
                    "obligations": [
                        {**plan["facets"][0]["obligations"][0], "unexpected": 1},
                        plan["facets"][0]["obligations"][1],
                    ],
                }
            ],
        },
    ],
)
def test_plan_validation_rejects_exact_key_and_bound_violations(mutator) -> None:
    with pytest.raises(NuggetCoverageError) as caught:
        validate_and_freeze_plan(NARRATIVE, mutator(VALID_PLAN))
    assert caught.value.stage == "planner"


def test_plan_validation_rejects_non_substring_and_supplemental_spans() -> None:
    required = json.loads(json.dumps(VALID_PLAN))
    required["facets"][0]["obligations"][0]["narrative_spans"] = ["not in narrative"]
    with pytest.raises(NuggetCoverageError, match="narrative span"):
        validate_and_freeze_plan(NARRATIVE, required)

    supplemental = json.loads(json.dumps(VALID_PLAN))
    supplemental["facets"][0]["obligations"][1]["narrative_spans"] = ["context"]
    with pytest.raises(NuggetCoverageError, match="supplemental"):
        validate_and_freeze_plan(NARRATIVE, supplemental)


def test_plan_validation_enforces_obligation_cap_and_required_obligation() -> None:
    too_many = {
        "schema_version": "retrieval_nugget_plan_v1",
        "facets": [
            {
                "title": f"Facet {index}",
                "obligations": [
                    {
                        "requirement": f"Requirement {index}-{item}",
                        "support_test": f"Support test {index}-{item}",
                        "kind": "supplemental_inferred",
                        "narrative_spans": [],
                    }
                    for item in range(1, 9)
                ],
            }
            for index in range(1, 7)
        ],
        "unmapped_narrative_spans": [],
    }
    with pytest.raises(NuggetCoverageError, match="40"):
        validate_and_freeze_plan(NARRATIVE, too_many)

    no_required = json.loads(json.dumps(VALID_PLAN))
    for obligation in no_required["facets"][0]["obligations"]:
        obligation["kind"] = "supplemental_inferred"
        obligation["narrative_spans"] = []
    with pytest.raises(NuggetCoverageError, match="required"):
        validate_and_freeze_plan(NARRATIVE, no_required)


def test_judgment_validation_resolves_aliases_and_enforces_semantics() -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    judgments = validate_judgments(plan, _nuggets(), VALID_JUDGMENTS)

    assert judgments[0].obligation_id == "f001-o001"
    assert judgments[0].supporting_nugget_ids == ("claim-cost",)
    assert judgments[1].supporting_nugget_ids == ("claim-history",)

    unsupported = json.loads(json.dumps(VALID_JUDGMENTS))
    unsupported["judgments"][0]["label"] = "unsupported"
    unsupported["judgments"][0]["supporting_nugget_aliases"] = ["n001"]
    with pytest.raises(NuggetCoverageError, match="unsupported"):
        validate_judgments(plan, _nuggets(), unsupported)

    full_with_missing = json.loads(json.dumps(VALID_JUDGMENTS))
    full_with_missing["judgments"][0]["label"] = "full"
    with pytest.raises(NuggetCoverageError, match="missing_elements"):
        validate_judgments(plan, _nuggets(), full_with_missing)

    unknown_alias = json.loads(json.dumps(VALID_JUDGMENTS))
    unknown_alias["judgments"][0]["supporting_nugget_aliases"] = ["n999"]
    with pytest.raises(NuggetCoverageError, match="alias"):
        validate_judgments(plan, _nuggets(), unknown_alias)


def test_judgment_validation_requires_exactly_one_row_per_obligation() -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    duplicate = json.loads(json.dumps(VALID_JUDGMENTS))
    duplicate["judgments"][1]["obligation_id"] = "f001-o001"
    with pytest.raises(NuggetCoverageError, match="duplicate"):
        validate_judgments(plan, _nuggets(), duplicate)

    missing = json.loads(json.dumps(VALID_JUDGMENTS))
    missing["judgments"].pop()
    with pytest.raises(NuggetCoverageError, match="between"):
        validate_judgments(plan, _nuggets(), missing)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: {**payload, "extra": True},
        lambda payload: {
            **payload,
            "judgments": [
                {**payload["judgments"][0], "extra": True},
                payload["judgments"][1],
            ],
        },
        lambda payload: {
            **payload,
            "judgments": [
                {key: value for key, value in payload["judgments"][0].items() if key != "label"},
                payload["judgments"][1],
            ],
        },
    ],
)
def test_judgment_validation_rejects_exact_key_violations(mutator) -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)

    with pytest.raises(NuggetCoverageError, match="unexpected or missing"):
        validate_judgments(plan, _nuggets(), mutator(json.loads(json.dumps(VALID_JUDGMENTS))))


def test_judgment_validation_rejects_unknown_obligation_id() -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    payload = json.loads(json.dumps(VALID_JUDGMENTS))
    payload["judgments"][0]["obligation_id"] = "f999-o001"

    with pytest.raises(NuggetCoverageError, match="unknown obligation ID"):
        validate_judgments(plan, _nuggets(), payload)


def test_judgment_validation_rejects_duplicate_supporting_aliases() -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    payload = json.loads(json.dumps(VALID_JUDGMENTS))
    payload["judgments"][0]["supporting_nugget_aliases"] = ["n001", "n001"]

    with pytest.raises(NuggetCoverageError, match="duplicate supporting nugget alias"):
        validate_judgments(plan, _nuggets(), payload)


def test_plan_validation_rejects_more_than_twelve_facets() -> None:
    facet = json.loads(json.dumps(VALID_PLAN["facets"][0]))
    payload = {
        "schema_version": "retrieval_nugget_plan_v1",
        "facets": [facet for _ in range(13)],
        "unmapped_narrative_spans": [],
    }

    with pytest.raises(NuggetCoverageError, match="between 1 and 12"):
        validate_and_freeze_plan(NARRATIVE, payload)


def test_plan_validation_rejects_more_than_eight_obligations_per_facet() -> None:
    required = json.loads(json.dumps(VALID_PLAN["facets"][0]["obligations"][0]))
    supplemental = {
        "requirement": "Add another historical detail.",
        "support_test": "Another historical comparison is present.",
        "kind": "supplemental_inferred",
        "narrative_spans": [],
    }
    payload = {
        "schema_version": "retrieval_nugget_plan_v1",
        "facets": [
            {
                "title": "Costs and assumptions",
                "obligations": [required, *[dict(supplemental) for _ in range(8)]],
            }
        ],
        "unmapped_narrative_spans": [],
    }

    with pytest.raises(NuggetCoverageError, match="between 1 and 8"):
        validate_and_freeze_plan(NARRATIVE, payload)


def test_score_coverage_macro_averages_required_facets_and_reports_diagnostics() -> None:
    plan_payload = json.loads(json.dumps(VALID_PLAN))
    plan_payload["facets"].append(
        {
            "title": "Safety",
            "obligations": [
                {
                    "requirement": "Explain safety.",
                    "support_test": "A safety explanation is present.",
                    "kind": "required_explicit",
                    "narrative_spans": ["Explain"],
                }
            ],
        }
    )
    plan = validate_and_freeze_plan(NARRATIVE, plan_payload)
    judgment_payload = json.loads(json.dumps(VALID_JUDGMENTS))
    judgment_payload["judgments"].append(
        {
            "obligation_id": "f002-o001",
            "label": "unsupported",
            "supporting_nugget_aliases": [],
            "missing_elements": "Safety is absent.",
        }
    )
    judgments = validate_judgments(plan, _nuggets(), judgment_payload)
    report = score_coverage(plan, _nuggets(), judgments, _identity())

    # Facet one is 0.5 (partial); facet two is 0.0. Macro average is 0.25.
    assert report.required_coverage == pytest.approx(0.25)
    assert report.strict_full_rate == pytest.approx(0.0)
    assert report.supplemental_coverage == pytest.approx(1.0)
    assert report.label_counts == {"full": 1, "partial": 1, "unsupported": 1}
    assert report.facet_scores["f001"] == pytest.approx(0.5)
    assert report.facet_scores["f002"] == pytest.approx(0.0)
    assert report.uncited_nugget_ids == ("claim-unused",)
    assert report.uncited_nugget_aliases == ("n003",)


def test_score_coverage_reports_null_when_no_supplemental_obligations() -> None:
    plan_payload = json.loads(json.dumps(VALID_PLAN))
    plan_payload["facets"][0]["obligations"].pop()
    plan = validate_and_freeze_plan(NARRATIVE, plan_payload)
    judgments_payload = json.loads(json.dumps(VALID_JUDGMENTS))
    judgments_payload["judgments"].pop()
    judgments_payload["judgments"][0]["label"] = "full"
    judgments_payload["judgments"][0]["supporting_nugget_aliases"] = ["n001"]
    judgments_payload["judgments"][0]["missing_elements"] = ""
    judgments = validate_judgments(plan, _nuggets(), judgments_payload)

    report = score_coverage(plan, _nuggets(), judgments, _identity())

    assert report.supplemental_coverage is None


def test_score_coverage_rejects_a_hand_built_plan_with_no_required_obligations() -> None:
    valid = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    supplemental = replace(
        valid.facets[0].obligations[0],
        kind="supplemental_inferred",
        narrative_spans=(),
    )
    hand_built = replace(
        valid,
        facets=(replace(valid.facets[0], obligations=(supplemental,)),),
    )
    judgments = validate_judgments(hand_built, _nuggets(), _judge_payload(hand_built))

    with pytest.raises(NuggetCoverageError, match="required") as caught:
        score_coverage(hand_built, _nuggets(), judgments, _identity())

    assert caught.value.stage == "scoring"


class RecordingBackend:
    def __init__(self, replies: list[BackendReply]) -> None:
        self.replies = list(replies)
        self.requests: list[CoverageModelRequest] = []

    def complete(self, request: CoverageModelRequest) -> BackendReply:
        self.requests.append(request)
        return self.replies.pop(0)


def _reply(payload: object, *, metadata: dict[str, object] | None = None) -> BackendReply:
    content = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return BackendReply(
        content=content,
        response_body=b'{"provider":"raw"}',
        status=200,
        metadata={} if metadata is None else metadata,
    )


def _judge_payload(plan: FrozenPlan) -> dict[str, object]:
    return {
        "schema_version": "retrieval_nugget_judgment_v1",
        "judgments": [
            {
                "obligation_id": obligation.obligation_id,
                "label": "full",
                "supporting_nugget_aliases": ["n001"],
                "missing_elements": "",
            }
            for obligation in plan.obligations
        ],
    }


def test_planner_request_isolated_and_schema_is_strict() -> None:
    request = render_planner_request(NARRATIVE, _identity().planner_model)

    assert request.stage == "planner"
    assert request.model == "planner-model"
    request_text = json.dumps(request.messages, ensure_ascii=False)
    assert NARRATIVE in request_text
    assert "claim-cost" not in request_text
    assert "The projected cost is $10 million." not in request_text
    assert all(term not in request_text.casefold() for term in ("passages", "docids", "ranks", "importance"))

    schema = request.response_schema
    assert set(schema["required"]) == {
        "schema_version",
        "facets",
        "unmapped_narrative_spans",
    }
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False
    facet_schema = schema["properties"]["facets"]["items"]
    assert set(facet_schema["required"]) == {"title", "obligations"}
    obligation_schema = facet_schema["properties"]["obligations"]["items"]
    assert set(obligation_schema["required"]) == {
        "requirement",
        "support_test",
        "kind",
        "narrative_spans",
    }
    assert obligation_schema["properties"]["kind"]["enum"] == [
        "required_explicit",
        "supplemental_inferred",
    ]


def test_planner_and_judge_contracts_describe_all_local_invariants() -> None:
    planner_request = render_planner_request(NARRATIVE, _identity().planner_model)
    planner_text = json.dumps(planner_request.messages, ensure_ascii=False).casefold()
    planner_schema = planner_request.response_schema
    planner_obligation = planner_schema["properties"]["facets"]["items"]["properties"]["obligations"]["items"]

    assert "exact" in planner_text and "substring" in planner_text
    assert "required_explicit" in planner_text and "supplemental_inferred" in planner_text
    assert "40" in planner_text
    assert "empty" in planner_text
    assert "exact" in str(planner_schema).casefold()
    assert "40" in str(planner_schema)
    assert "supplemental" in str(planner_obligation).casefold()
    assert "at least one required" in planner_text
    assert "at least one required" in str(planner_schema).casefold()
    assert "8" in planner_text and "40" in planner_text and "1000" in planner_text
    assert planner_obligation["properties"]["narrative_spans"]["maxItems"] == 8
    assert planner_obligation["properties"]["narrative_spans"]["items"]["maxLength"] == 1000
    assert planner_schema["properties"]["unmapped_narrative_spans"]["maxItems"] == 40
    assert planner_schema["properties"]["unmapped_narrative_spans"]["items"]["maxLength"] == 1000

    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    judge_request = render_judge_request(NARRATIVE, plan, _nuggets(), _identity().judge_model)
    judge_text = json.dumps(judge_request.messages, ensure_ascii=False).casefold()
    judge_schema = judge_request.response_schema

    for term in ("full", "partial", "unsupported", "supporting_nugget_aliases", "missing_elements"):
        assert term in judge_text
        assert term in str(judge_schema).casefold()
    assert "empty" in judge_text
    assert "exactly one" in judge_text


def test_judge_request_contains_only_aliases_and_frozen_plan() -> None:
    plan = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    request = render_judge_request(NARRATIVE, plan, _nuggets(), _identity().judge_model)

    assert request.stage == "judge"
    request_text = json.dumps(request.messages, ensure_ascii=False)
    assert NARRATIVE in request_text
    assert "f001-o001" in request_text
    assert "f001-o002" in request_text
    assert "n001: The projected cost is $10 million." in request_text
    assert "n002: Historically, the comparable cost was lower." in request_text
    assert "n003: An uncited related detail." in request_text
    assert "claim-cost" not in request_text
    assert all(term not in request_text.casefold() for term in ("passages", "docids", "ranks", "importance"))

    schema = request.response_schema
    assert set(schema["required"]) == {"schema_version", "judgments"}
    judgment_schema = schema["properties"]["judgments"]["items"]
    assert set(judgment_schema["required"]) == {
        "obligation_id",
        "label",
        "supporting_nugget_aliases",
        "missing_elements",
    }
    assert judgment_schema["properties"]["obligation_id"]["enum"] == ["f001-o001", "f001-o002"]
    assert judgment_schema["properties"]["label"]["enum"] == ["full", "partial", "unsupported"]
    assert judgment_schema["properties"]["supporting_nugget_aliases"]["items"]["enum"] == [
        "n001",
        "n002",
        "n003",
    ]


def test_evaluation_makes_exactly_two_calls_and_returns_safe_metadata() -> None:
    plan_payload = VALID_PLAN
    frozen = validate_and_freeze_plan(NARRATIVE, plan_payload)
    planner = RecordingBackend([_reply(plan_payload, metadata={"provider": "planner"})])
    judge = RecordingBackend([_reply(_judge_payload(frozen), metadata={"provider": "judge"})])

    result = evaluate_nugget_coverage(
        narrative=NARRATIVE,
        nuggets=_nuggets(),
        planner=planner,
        judge=judge,
        identity=_identity(),
    )

    assert len(planner.requests) == 1
    assert len(judge.requests) == 1
    assert result[0].plan_sha256 == frozen.plan_sha256
    assert result[1][0].label == "full"
    assert result[2].required_coverage == pytest.approx(1.0)
    assert result[3].planner_metadata == {"provider": "planner"}
    assert result[3].judge_metadata == {"provider": "judge"}
    assert not hasattr(result[3], "planner_response_body")


def test_evaluation_metadata_uses_the_persisted_provider_allowlist() -> None:
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    metadata = {
        "provider": "planner",
        "requested_model": "planner-model",
        "secret": "do-not-persist",
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "secret": "do-not-persist",
        },
    }
    planner = RecordingBackend([_reply(VALID_PLAN, metadata=metadata)])
    judge = RecordingBackend([_reply(_judge_payload(frozen), metadata=metadata)])

    result = evaluate_nugget_coverage(
        narrative=NARRATIVE,
        nuggets=_nuggets(),
        planner=planner,
        judge=judge,
        identity=_identity(),
    )

    expected = {
        "provider": "planner",
        "requested_model": "planner-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    assert result[3].planner_metadata == expected
    assert result[3].judge_metadata == expected


def test_malformed_planner_prevents_judge_without_repair_call() -> None:
    planner = RecordingBackend([_reply({"not": "a plan"}), _reply(VALID_PLAN)])
    judge = RecordingBackend([])

    with pytest.raises(NuggetCoverageError) as caught:
        evaluate_nugget_coverage(
            narrative=NARRATIVE,
            nuggets=_nuggets(),
            planner=planner,
            judge=judge,
            identity=_identity(),
        )

    assert caught.value.stage == "planner"
    assert len(planner.requests) == 1
    assert judge.requests == []


def test_malformed_judge_does_not_trigger_semantic_repair() -> None:
    planner = RecordingBackend([_reply(VALID_PLAN)])
    judge = RecordingBackend([_reply({"not": "judgments"}), _reply({})])

    with pytest.raises(NuggetCoverageError) as caught:
        evaluate_nugget_coverage(
            narrative=NARRATIVE,
            nuggets=_nuggets(),
            planner=planner,
            judge=judge,
            identity=_identity(),
        )

    assert caught.value.stage == "judge"
    assert len(planner.requests) == 1
    assert len(judge.requests) == 1


def test_oversized_judge_request_prevents_judge_call() -> None:
    huge_nuggets = (CoverageNugget("huge", "x" * 1_000_001),)
    planner = RecordingBackend([_reply(VALID_PLAN)])
    judge = RecordingBackend([])

    with pytest.raises(NuggetCoverageError, match="1,000,000") as caught:
        evaluate_nugget_coverage(
            narrative=NARRATIVE,
            nuggets=huge_nuggets,
            planner=planner,
            judge=judge,
            identity=_identity(),
        )

    assert caught.value.stage == "judge"
    assert len(planner.requests) == 1
    assert judge.requests == []


def test_empty_nuggets_are_rejected_before_either_backend_call() -> None:
    planner = RecordingBackend([])
    judge = RecordingBackend([])

    with pytest.raises(NuggetCoverageError, match="at least one"):
        evaluate_nugget_coverage(
            narrative=NARRATIVE,
            nuggets=(),
            planner=planner,
            judge=judge,
            identity=_identity(),
        )

    assert planner.requests == []
    assert judge.requests == []


def _coverage_handoff(
    *,
    empty_claim_hints: bool = False,
    narrative: str = NARRATIVE,
    claim_texts: tuple[str, str] | None = None,
) -> GenerationHandoff:
    evidence_text = "A selected passage supporting the first canonical claim."
    evidence = EvidencePassage(
        evidence_id="ev-1",
        group_id="group-1",
        cluster_id="cluster-1",
        cluster_ordinal=1,
        support_ordinal=1,
        candidate_kind="passage",
        docid="DOC-1",
        document_rank=1,
        text=evidence_text,
        document_sha256="a" * 64,
        source_span=EvidenceSourceSpan(
            start_char=0,
            end_char=len(evidence_text),
            start_byte=0,
            end_byte=len(evidence_text.encode("utf-8")),
        ),
    )
    group = EvidenceGroup(
        group_id="group-1",
        kind="generated_subnarrative",
        text="A retrieval subnarrative.",
        selected_clusters=(
            SelectedCluster(
                cluster_id="cluster-1",
                ordinal=1,
                representative_evidence_id="ev-1",
                evidence_ids=("ev-1",),
            ),
        ),
    )
    if claim_texts is None:
        claim_texts = (
            "The first canonical retrieval claim.",
            "The second canonical retrieval claim.",
        )
    claims = () if empty_claim_hints else (
        ClaimHint(
            claim_id="claim-z",
            group_id="group-1",
            kind="canonical",
            text=claim_texts[0],
            evidence_ids=("ev-1",),
        ),
        ClaimHint(
            claim_id="claim-a",
            group_id="group-1",
            kind="canonical",
            text=claim_texts[1],
            evidence_ids=("ev-1",),
        ),
    )
    topic = GenerationTopic(
        topic_id="topic-coverage",
        narrative=narrative,
        groups=(group,),
        evidence=(evidence,),
        claim_hints=claims,
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="b" * 64,
            retrieval_topic_sha256="c" * 64,
        ),
    )
    return GenerationHandoff(
        producer=HandoffProducer(
            source_contract="topic_records_v4",
            retrieval_run_id="retrieval-run",
            producer_revision="revision-1",
        ),
        topics=(topic,),
    )


def _write_coverage_handoff(tmp_path: Path, *, empty_claim_hints: bool = False) -> Path:
    path = tmp_path / "generation_handoff_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_generation_handoff(path, _coverage_handoff(empty_claim_hints=empty_claim_hints))
    return path


def test_coverage_input_from_handoff_authenticates_and_assigns_ordered_aliases(tmp_path: Path) -> None:
    path = _write_coverage_handoff(tmp_path)

    bound = coverage_input_from_handoff(path, "topic-coverage")

    assert isinstance(bound, BoundCoverageInput)
    assert bound.topic_id == "topic-coverage"
    assert bound.manifest_sha256 == _coverage_handoff().manifest_sha256
    assert bound.narrative_sha256 == hashlib.sha256(NARRATIVE.encode()).hexdigest()
    assert [nugget.nugget_id for nugget in bound.nuggets] == ["claim-z", "claim-a"]
    assert [hashlib.sha256(nugget.text.encode()).hexdigest() for nugget in bound.nuggets] == list(bound.nugget_text_sha256s)


def test_bound_input_rejects_a_narrative_hash_mismatch() -> None:
    source = _coverage_handoff()
    topic = source.topics[0]
    nuggets = tuple(
        CoverageNugget(claim.claim_id, claim.text) for claim in topic.claim_hints
    )

    with pytest.raises(NuggetCoverageError, match="narrative identity"):
        BoundCoverageInput(
            manifest_sha256=source.manifest_sha256,
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            narrative_sha256="0" * 64,
            nuggets=nuggets,
            nugget_text_sha256s=tuple(
                hashlib.sha256(nugget.text.encode()).hexdigest() for nugget in nuggets
            ),
        )


def test_coverage_input_rejects_unknown_topic_and_empty_claim_hints(tmp_path: Path) -> None:
    path = _write_coverage_handoff(tmp_path)
    with pytest.raises(ValueError):
        coverage_input_from_handoff(path, "unknown-topic")

    empty = _write_coverage_handoff(tmp_path / "empty", empty_claim_hints=True)
    with pytest.raises(NuggetCoverageError, match="canonical retrieval nugget"):
        coverage_input_from_handoff(empty, "topic-coverage")


def test_coverage_input_rejects_altered_manifest_bytes(tmp_path: Path) -> None:
    path = _write_coverage_handoff(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"topic-coverage", b"topic-corrupt", 1))

    with pytest.raises(ValueError):
        coverage_input_from_handoff(path, "topic-coverage")


def test_coverage_input_preserves_authenticated_surrounding_whitespace_and_multiline_text(
    tmp_path: Path,
) -> None:
    narrative = "\n  " + NARRATIVE + "\n"
    claim_texts = ("\n first claim\n", "  second claim\n")
    path = tmp_path / "whitespace" / "generation_handoff_manifest.json"
    write_generation_handoff(
        path,
        _coverage_handoff(narrative=narrative, claim_texts=claim_texts),
    )

    bound = coverage_input_from_handoff(path, "topic-coverage")

    assert bound.narrative == narrative
    assert tuple(nugget.text for nugget in bound.nuggets) == claim_texts
    assert bound.narrative_sha256 == hashlib.sha256(narrative.encode()).hexdigest()
    assert bound.nugget_text_sha256s == tuple(
        hashlib.sha256(text.encode()).hexdigest() for text in claim_texts
    )


@pytest.mark.parametrize("field", ["narrative", "claim"])
def test_coverage_input_rejects_control_text_from_authenticated_handoff(
    tmp_path: Path, field: str
) -> None:
    narrative = NARRATIVE + "\x00" if field == "narrative" else NARRATIVE
    claim_texts = ("claim\x00", "second claim") if field == "claim" else None
    path = tmp_path / field / "generation_handoff_manifest.json"
    write_generation_handoff(
        path,
        _coverage_handoff(narrative=narrative, claim_texts=claim_texts),
    )

    with pytest.raises(NuggetCoverageError, match="control"):
        coverage_input_from_handoff(path, "topic-coverage")


@pytest.mark.parametrize("field", ["narrative", "claim"])
def test_coverage_input_rejects_whitespace_only_sealed_text(
    tmp_path: Path, field: str
) -> None:
    narrative = " \n\t" if field == "narrative" else NARRATIVE
    claim_texts = (" \n\t", "second claim") if field == "claim" else None
    path = tmp_path / field / "generation_handoff_manifest.json"
    write_generation_handoff(
        path,
        _coverage_handoff(narrative=narrative, claim_texts=claim_texts),
    )

    with pytest.raises(NuggetCoverageError, match="non-empty"):
        coverage_input_from_handoff(path, "topic-coverage")


def test_run_accepts_whitespace_preserving_source_text_from_a_sealed_handoff(
    tmp_path: Path,
) -> None:
    handoff_path = tmp_path / "generation_handoff_manifest.json"
    write_generation_handoff(
        handoff_path,
        _coverage_handoff(
            narrative="\n  " + NARRATIVE + "\n",
            claim_texts=("\n first claim\n", "  second claim\n"),
        ),
    )
    work_dir = tmp_path / "whitespace-run"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)

    receipt = run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    assert receipt.required_coverage == pytest.approx(1.0)


def test_run_preserves_authenticated_unicode_zwj_source_text_end_to_end(
    tmp_path: Path,
) -> None:
    family = "👨\u200d👩\u200d👧\u200d👦"
    narrative = "\n  Explain the projected cost and assumptions " + family + ".\n"
    claim_texts = ("\n Family evidence " + family + ".\n", "  second claim\n")
    handoff_path = tmp_path / "generation_handoff_manifest.json"
    write_generation_handoff(
        handoff_path,
        _coverage_handoff(narrative=narrative, claim_texts=claim_texts),
    )
    work_dir = tmp_path / "unicode-run"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    planner = RecordingBackend([_reply(VALID_PLAN)])
    judge = RecordingBackend([_reply(_judge_payload(frozen))])

    receipt = run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir), planner=planner, judge=judge
    )

    assert receipt.required_coverage == pytest.approx(1.0)
    planner_payload = json.loads(planner.requests[0].messages[1]["content"])
    judge_payload = json.loads(judge.requests[0].messages[1]["content"])
    assert planner_payload["narrative"] == narrative
    assert judge_payload["narrative"] == narrative
    assert judge_payload["nuggets"][0] == "n001: " + claim_texts[0]


def _coverage_config(
    handoff_path: Path,
    work_dir: Path | None,
    *,
    mode: str = "create",
    allow_hosted_calls: bool = True,
) -> CoverageRunConfig:
    return CoverageRunConfig(
        handoff_manifest_path=handoff_path,
        topic_id="topic-coverage",
        work_dir=work_dir,
        planner_model="planner-model",
        judge_model="judge-model",
        mode=mode,
        allow_hosted_calls=allow_hosted_calls,
    )


def test_create_persists_private_canonical_artifacts_and_safe_receipt(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "coverage-work"
    planner = RecordingBackend([_reply(VALID_PLAN, metadata={"provider": "planner"})])
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    judge = RecordingBackend([_reply(_judge_payload(frozen), metadata={"provider": "judge"})])

    receipt = run_coverage_evaluation(_coverage_config(handoff_path, work_dir), planner=planner, judge=judge)

    assert isinstance(receipt, CoverageRunReceipt)
    assert set(path.name for path in work_dir.iterdir()) == {
        "input.json", "plan.json", "judgments.json", "report.json", "manifest.json"
    }
    input_payload = json.loads((work_dir / "input.json").read_text())
    assert "narrative" not in input_payload
    assert all("text" not in nugget for nugget in input_payload["nuggets"])
    assert NARRATIVE not in json.dumps(input_payload)
    assert receipt.topic_id == "topic-coverage"
    assert receipt.hosted_calls == 2
    assert receipt.required_coverage == pytest.approx(1.0)


def _snapshot_tree(path: Path) -> dict[str, bytes | None]:
    return {
        str(entry.relative_to(path)): (entry.read_bytes() if entry.is_file() else None)
        for entry in sorted(path.iterdir())
    }


def test_load_completed_coverage_evaluation_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "completed-loader"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    judge_payload = _judge_payload(frozen)
    judge_payload["judgments"][0]["label"] = "partial"
    judge_payload["judgments"][0]["missing_elements"] = "The assumptions are missing."
    planner = RecordingBackend([_reply(VALID_PLAN)])
    judge = RecordingBackend([_reply(judge_payload)])
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=planner,
        judge=judge,
    )
    before = _snapshot_tree(work_dir)

    def fail_if_called(*_args, **_kwargs) -> BackendReply:
        raise AssertionError("loader must not call a fixture backend")

    monkeypatch.setattr(planner, "complete", fail_if_called)
    monkeypatch.setattr(judge, "complete", fail_if_called)

    class ExplodingBackend:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("loader must not construct a backend")

    monkeypatch.setattr(coverage_module, "OpenRouterCoverageBackend", ExplodingBackend)

    loaded = load_completed_coverage_evaluation(
        handoff_manifest_path=handoff_path,
        topic_id="topic-coverage",
        work_dir=work_dir,
    )

    assert loaded.bound_input.topic_id == "topic-coverage"
    assert loaded.plan.obligations == frozen.obligations
    expected_judgments = validate_judgments(
        frozen, coverage_input_from_handoff(handoff_path, "topic-coverage").nuggets,
        judge_payload,
    )
    assert loaded.judgments == expected_judgments
    assert loaded.report.required_coverage == 0.5
    assert loaded.artifact_hashes.keys() == {
        "input.json", "plan.json", "judgments.json", "report.json"
    }
    assert len(loaded.manifest_sha256) == 64
    assert before == _snapshot_tree(work_dir)


def _write_canonical_payload(path: Path, payload: object) -> None:
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _completed_bundle(tmp_path: Path) -> tuple[Path, Path, FrozenPlan]:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "completed-corruption"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    return handoff_path, work_dir, frozen


@pytest.mark.parametrize("artifact", ["input.json", "plan.json", "judgments.json", "report.json"])
def test_load_completed_coverage_evaluation_rejects_changed_artifact_without_writing(tmp_path: Path, artifact: str) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    path = work_dir / artifact
    payload = json.loads(path.read_text())
    if artifact == "input.json":
        payload["topic_id"] = "topic-corrupt"
    elif artifact == "plan.json":
        payload["plan_sha256"] = "0" * 64
    elif artifact == "judgments.json":
        payload["request_sha256"] = "0" * 64
    else:
        payload["required_coverage"] = 0.0
    _write_canonical_payload(path, payload)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


@pytest.mark.parametrize("artifact", ["plan.json", "judgments.json"])
def test_load_completed_coverage_evaluation_rejects_changed_provider_request_digest_without_writing(
    tmp_path: Path, artifact: str
) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    path = work_dir / artifact
    payload = json.loads(path.read_text())
    payload["request_sha256"] = "f" * 64
    _write_canonical_payload(path, payload)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="request identity"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "stale-schema"),
        ("planner_prompt_version", "stale-planner"),
        ("judge_prompt_version", "stale-judge"),
        ("planner_model", ""),
        ("judge_model", "unsafe\x00model"),
    ],
)
def test_load_completed_coverage_evaluation_rejects_stale_or_unsafe_manifest_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"][field] = value
    _write_canonical_payload(manifest_path, manifest)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="manifest identity"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


@pytest.mark.parametrize("identity_change", ["missing", "extra"])
def test_load_completed_coverage_evaluation_rejects_manifest_identity_key_changes_without_writing(
    tmp_path: Path, identity_change: str
) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if identity_change == "missing":
        del manifest["identity"]["planner_model"]
    else:
        manifest["identity"]["unexpected"] = "value"
    _write_canonical_payload(manifest_path, manifest)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="manifest identity"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


@pytest.mark.parametrize("manifest_field", ["artifact_hashes", "completed_stages"])
def test_load_completed_coverage_evaluation_rejects_changed_manifest_binding_without_writing(
    tmp_path: Path, manifest_field: str
) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest_field == "artifact_hashes":
        manifest["artifact_hashes"]["report.json"] = "0" * 64
    else:
        manifest["completed_stages"] = 1
    _write_canonical_payload(manifest_path, manifest)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="manifest"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


def test_load_completed_coverage_evaluation_rejects_unknown_supporting_nugget_id_without_writing(tmp_path: Path) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    path = work_dir / "judgments.json"
    payload = json.loads(path.read_text())
    payload["judgments"][0]["supporting_nugget_ids"] = ["unknown-nugget"]
    _write_canonical_payload(path, payload)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="unknown nugget"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


@pytest.mark.parametrize("missing_name", ["input.json", "plan.json", "judgments.json", "report.json", "manifest.json"])
def test_load_completed_coverage_evaluation_rejects_missing_required_artifact_without_writing(tmp_path: Path, missing_name: str) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    (work_dir / missing_name).unlink()
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="missing"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


def test_load_completed_coverage_evaluation_rejects_symlinked_work_directory(tmp_path: Path) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    symlink = tmp_path / "coverage-link"
    symlink.symlink_to(work_dir, target_is_directory=True)

    with pytest.raises(NuggetCoverageError, match="work directory"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=symlink,
        )


def test_load_completed_coverage_evaluation_rejects_symlinked_required_artifact_without_writing(tmp_path: Path) -> None:
    handoff_path, work_dir, _frozen = _completed_bundle(tmp_path)
    report_path = work_dir / "report.json"
    target = tmp_path / "report-target.json"
    target.write_bytes(report_path.read_bytes())
    report_path.unlink()
    report_path.symlink_to(target)
    before = _snapshot_tree(work_dir)

    with pytest.raises(NuggetCoverageError, match="symbolic link"):
        load_completed_coverage_evaluation(
            handoff_manifest_path=handoff_path,
            topic_id="topic-coverage",
            work_dir=work_dir,
        )

    assert before == _snapshot_tree(work_dir)


def test_report_artifact_carries_v1_assumption_and_honest_limits(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "report-contract"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)

    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    report = json.loads((work_dir / "report.json").read_text())
    assert "canonical retrieval nuggets are assumed to faithfully represent" in report["core_assumption"].casefold()
    assert "does not reopen passages" in report["core_assumption"].casefold()
    limits = " ".join(report["limits"]).casefold()
    assert "planner-derived" in limits and "not ground truth" in limits
    assert "cannot detect a nugget that misstates its source" in limits
    assert "retrieval, selection, and canonicalization" in limits
    assert "not comparable" in limits and "model identities" in limits


def test_create_refuses_nonempty_work_dir_and_cache_only_names_missing_stages(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "nonempty"
    work_dir.mkdir()
    (work_dir / "unrelated.txt").write_text("private")

    with pytest.raises(NuggetCoverageError, match="non-empty"):
        run_coverage_evaluation(_coverage_config(handoff_path, work_dir))

    missing = tmp_path / "missing"
    with pytest.raises(NuggetCoverageError, match="planner and judge") as caught:
        cache_planner = RecordingBackend([_reply(VALID_PLAN)])
        cache_judge = RecordingBackend([])
        run_coverage_evaluation(
            _coverage_config(handoff_path, missing, allow_hosted_calls=False),
            planner=cache_planner,
            judge=cache_judge,
        )
    assert caught.value.stage == "cache"
    assert cache_planner.requests == []
    assert cache_judge.requests == []


def test_cache_only_create_writes_nothing_then_authorized_create_succeeds(
    tmp_path: Path,
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "cache-first"
    cache_planner = RecordingBackend([_reply(VALID_PLAN)])
    cache_judge = RecordingBackend([])

    with pytest.raises(NuggetCoverageError, match="planner and judge"):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, allow_hosted_calls=False),
            planner=cache_planner,
            judge=cache_judge,
        )

    assert not work_dir.exists()
    assert cache_planner.requests == []
    assert cache_judge.requests == []

    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    receipt = run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir, allow_hosted_calls=True),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    assert receipt.status == "complete"
    assert (work_dir / "manifest.json").exists()


def test_resume_reuses_valid_stages_without_backend_calls_and_reproduces_hashes(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "resume"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    first = run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    resumed_planner = RecordingBackend([])
    resumed_judge = RecordingBackend([])
    second = run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir, mode="resume"),
        planner=resumed_planner,
        judge=resumed_judge,
    )

    assert resumed_planner.requests == []
    assert resumed_judge.requests == []
    assert first.artifact_hashes == second.artifact_hashes
    assert second.reused_stages == ("planner", "judge")


def test_resume_rejects_changed_model_or_artifact_bytes(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "resume"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    changed = _coverage_config(handoff_path, work_dir, mode="resume")
    object.__setattr__(changed, "planner_model", "changed-model")
    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(changed, planner=RecordingBackend([]), judge=RecordingBackend([]))

    report = work_dir / "report.json"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]), judge=RecordingBackend([]),
        )


class FakeOpenRouterTransport:
    def __init__(self, responses: list[FacetResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _openrouter_response(content: object, *, finish_reason: str = "stop") -> FacetResponse:
    envelope = {
        "id": "completion-1", "object": "chat.completion", "created": 1,
        "model": "provider-model", "provider": "provider-name",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(content)},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return FacetResponse(status=200, body=json.dumps(envelope).encode())


def test_openrouter_backend_uses_strict_schema_and_redacts_credentials() -> None:
    transport = FakeOpenRouterTransport([_openrouter_response(VALID_PLAN)])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "fake-secret"}, transport=transport
    )
    request = render_planner_request(NARRATIVE, "planner-model")

    reply = backend.complete(request)

    sent = transport.requests[0]
    assert sent.url == "https://openrouter.ai/api/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer fake-secret"
    body = json.loads(sent.body)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == request.response_schema_name
    assert body["provider"] == {"require_parameters": True, "data_collection": "deny"}
    assert body["reasoning"] == {"enabled": False}
    assert "temperature" not in body
    assert body["seed"] == 0
    assert body["stream"] is False
    assert body["max_tokens"] == 8192
    assert reply.metadata["provider"] == "provider-name"
    assert b"fake-secret" not in reply.response_body


def test_openrouter_backend_retries_only_transient_transport_and_not_semantics() -> None:
    transport = FakeOpenRouterTransport([TimeoutError("temporary"), _openrouter_response(VALID_PLAN)])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=2
    )
    backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert len(transport.requests) == 2

    semantic = FakeOpenRouterTransport([
        _openrouter_response(VALID_PLAN, finish_reason="error"), _openrouter_response(VALID_PLAN)
    ])
    semantic_backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=semantic, transport_max_attempts=3
    )
    with pytest.raises(Exception):
        semantic_backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert len(semantic.requests) == 1


def test_default_work_dir_is_under_handoff_parent(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(_coverage_config(handoff_path, None, allow_hosted_calls=False))
    assert (tmp_path / "retrieval_nugget_coverage" / "topic-coverage").parent == tmp_path / "retrieval_nugget_coverage"


def test_cli_cache_only_emits_one_safe_json_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)

    code = main([
        "--handoff-manifest", str(handoff_path),
        "--topic", "topic-coverage",
        "--work-dir", str(tmp_path / "cli-work"),
    ])

    captured = capsys.readouterr()
    assert code != 0
    assert len(captured.out.splitlines()) == 1
    assert NARRATIVE not in captured.out
    assert "canonical retrieval claim" not in captured.out
    assert "OPENROUTER_API_KEY" not in captured.out
    payload = json.loads(captured.out)
    assert payload["error"]["stage"] == "cache"
    assert payload["error"]["reason"] == "missing planner and judge stages"


def test_cli_cache_errors_name_exact_missing_stage_on_partial_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "partial-cli"

    class FailingJudge:
        def complete(self, request: CoverageModelRequest) -> BackendReply:
            raise RuntimeError("simulated interruption")

    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir),
            planner=RecordingBackend([_reply(VALID_PLAN)]),
            judge=FailingJudge(),
        )

    code = main([
        "--handoff-manifest", str(handoff_path),
        "--topic", "topic-coverage",
        "--work-dir", str(work_dir),
        "--planner-model", "planner-model",
        "--judge-model", "judge-model",
        "--mode", "resume",
    ])
    captured = capsys.readouterr()

    assert code != 0
    payload = json.loads(captured.out)
    assert payload["error"]["stage"] == "cache"
    assert payload["error"]["reason"] == "missing judge stage"
    assert "planner" not in payload["error"]["reason"]


def test_manifest_rejects_unknown_fields_and_changed_provider_or_call_history(
    tmp_path: Path,
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "manifest"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN, metadata={"provider": "planner"})]),
        judge=RecordingBackend([_reply(_judge_payload(frozen), metadata={"provider": "judge"})]),
    )

    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_stages"] = 99
    manifest["provider_metadata"]["planner"]["provider"] = "tampered-provider"
    manifest["unexpected"] = True
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]),
            judge=RecordingBackend([]),
        )


def test_interrupted_then_resumed_stages_seal_deterministic_call_count(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "partial"
    class FailingJudge:
        def complete(self, request: CoverageModelRequest) -> BackendReply:
            raise RuntimeError("simulated interruption")

    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir),
            planner=RecordingBackend([_reply(VALID_PLAN)]),
            judge=FailingJudge(),
        )

    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir, mode="resume"),
        planner=RecordingBackend([]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    manifest = json.loads((work_dir / "manifest.json").read_text())
    assert manifest["completed_stages"] == 2
    assert "hosted_calls" not in manifest


@pytest.mark.parametrize("request_digest", ["not-a-digest", "0" * 64])
def test_interrupted_resume_rejects_tampered_or_malformed_planner_request_digest(
    tmp_path: Path, request_digest: str
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "planner-request-digest"

    class FailingJudge:
        def complete(self, request: CoverageModelRequest) -> BackendReply:
            raise RuntimeError("simulated interruption")

    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir),
            planner=RecordingBackend([_reply(VALID_PLAN)]),
            judge=FailingJudge(),
        )
    plan_path = work_dir / "plan.json"
    plan_payload = json.loads(plan_path.read_text())
    plan_payload["request_sha256"] = request_digest
    plan_path.write_bytes(json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode())

    resumed_planner = RecordingBackend([])
    resumed_judge = RecordingBackend([])
    with pytest.raises(NuggetCoverageError) as caught:
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=resumed_planner,
            judge=resumed_judge,
        )

    assert caught.value.stage == "persistence"
    assert resumed_planner.requests == []
    assert resumed_judge.requests == []
    assert not (work_dir / "manifest.json").exists()


def test_seed_completed_baseline_reuses_exact_plan_and_calls_only_candidate_judge(
    tmp_path: Path,
) -> None:
    baseline_handoff = _write_coverage_handoff(tmp_path / "baseline")
    baseline_work = tmp_path / "baseline-work"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(baseline_handoff, baseline_work),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    candidate_handoff = tmp_path / "candidate" / "generation_handoff_manifest.json"
    write_generation_handoff(
        candidate_handoff,
        _coverage_handoff(
            claim_texts=(
                "A stronger fixed canonical retrieval claim.",
                "A second stronger fixed canonical retrieval claim.",
            )
        ),
    )
    candidate_work = tmp_path / "candidate-work"

    seed_coverage_plan_from_completed_baseline(
        baseline_handoff_manifest_path=baseline_handoff,
        baseline_work_dir=baseline_work,
        candidate_handoff_manifest_path=candidate_handoff,
        candidate_work_dir=candidate_work,
        topic_id="topic-coverage",
    )

    planner = RecordingBackend([])
    judge = RecordingBackend([_reply(_judge_payload(frozen))])
    receipt = run_coverage_evaluation(
        _coverage_config(candidate_handoff, candidate_work, mode="resume"),
        planner=planner,
        judge=judge,
    )

    assert (candidate_work / "plan.json").read_bytes() == (
        baseline_work / "plan.json"
    ).read_bytes()
    assert planner.requests == []
    assert len(judge.requests) == 1
    assert receipt.hosted_calls == 1
    assert receipt.reused_stages == ("planner",)


def test_seed_completed_baseline_rejects_candidate_narrative_drift(
    tmp_path: Path,
) -> None:
    baseline_handoff = _write_coverage_handoff(tmp_path / "baseline")
    baseline_work = tmp_path / "baseline-work"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(baseline_handoff, baseline_work),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    candidate_handoff = tmp_path / "candidate" / "generation_handoff_manifest.json"
    write_generation_handoff(
        candidate_handoff,
        _coverage_handoff(narrative=NARRATIVE + " Changed."),
    )

    with pytest.raises(NuggetCoverageError, match="narrative"):
        seed_coverage_plan_from_completed_baseline(
            baseline_handoff_manifest_path=baseline_handoff,
            baseline_work_dir=baseline_work,
            candidate_handoff_manifest_path=candidate_handoff,
            candidate_work_dir=tmp_path / "candidate-work",
            topic_id="topic-coverage",
        )


@pytest.mark.parametrize("request_digest", ["not-a-digest", "0" * 64])
def test_interrupted_resume_rejects_tampered_or_malformed_judge_request_digest(
    tmp_path: Path, request_digest: str
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "judge-request-digest"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    (work_dir / "manifest.json").unlink()
    judgments_path = work_dir / "judgments.json"
    judgments_payload = json.loads(judgments_path.read_text())
    judgments_payload["request_sha256"] = request_digest
    judgments_path.write_bytes(
        json.dumps(judgments_payload, sort_keys=True, separators=(",", ":")).encode()
    )

    resumed_planner = RecordingBackend([])
    resumed_judge = RecordingBackend([])
    with pytest.raises(NuggetCoverageError) as caught:
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=resumed_planner,
            judge=resumed_judge,
        )

    assert caught.value.stage == "persistence"
    assert resumed_planner.requests == []
    assert resumed_judge.requests == []
    assert not (work_dir / "manifest.json").exists()


def test_resume_classifies_unknown_persisted_nugget_ids_as_persistence_errors(
    tmp_path: Path,
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "unknown-nugget"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    judgments_path = work_dir / "judgments.json"
    payload = json.loads(judgments_path.read_text())
    payload["judgments"][0]["supporting_nugget_ids"] = ["unknown-nugget-id"]
    judgments_path.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )

    with pytest.raises(NuggetCoverageError) as caught:
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]),
            judge=RecordingBackend([]),
        )

    assert caught.value.stage == "persistence"


def test_resume_rejects_orphan_report_before_any_backend_call(tmp_path: Path) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "orphan-report"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )
    (work_dir / "plan.json").unlink()
    (work_dir / "judgments.json").unlink()
    (work_dir / "manifest.json").unlink()

    planner = RecordingBackend([])
    judge = RecordingBackend([])
    with pytest.raises(NuggetCoverageError) as caught:
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=planner,
            judge=judge,
        )

    assert caught.value.stage == "persistence"
    assert planner.requests == []
    assert judge.requests == []


def test_openrouter_does_not_retry_permanent_transport_errors() -> None:
    transport = FakeOpenRouterTransport([ValueError("permanent configuration failure")])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=3
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert len(transport.requests) == 1


def test_cli_argument_errors_are_safe_json_and_do_not_echo_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--mode", "fake-secret"])

    captured = capsys.readouterr()
    assert code != 0
    assert len(captured.out.splitlines()) == 1
    assert "fake-secret" not in captured.out
    payload = json.loads(captured.out)
    assert payload["error"]["stage"] == "config"


@pytest.mark.parametrize("topic_id", ["../escape", "nested/topic", "/absolute", ".."])
def test_topic_id_rejects_path_unsafe_work_directory_names(
    tmp_path: Path, topic_id: str
) -> None:
    with pytest.raises(NuggetCoverageError, match="topic"):
        CoverageRunConfig(
            handoff_manifest_path=tmp_path / "handoff.json",
            topic_id=topic_id,
        )


def test_empty_narrative_is_rejected_before_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = _coverage_handoff()
    object.__setattr__(handoff.topics[0], "narrative", "")
    monkeypatch.setattr(coverage_module, "load_generation_handoff", lambda _path: handoff)

    with pytest.raises(NuggetCoverageError, match="narrative is empty"):
        coverage_input_from_handoff(tmp_path / "ignored.json", "topic-coverage")


@pytest.mark.parametrize("change", ["narrative", "nugget_order"])
def test_resume_rejects_changed_authenticated_handoff_projection(
    tmp_path: Path, change: str
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path / "original")
    work_dir = tmp_path / "projection"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    original = _coverage_handoff()
    topic = original.topics[0]
    if change == "narrative":
        changed_topic = replace(topic, narrative=NARRATIVE + " A changed sentence.")
    else:
        changed_topic = replace(topic, claim_hints=tuple(reversed(topic.claim_hints)))
    changed_handoff = replace(original, topics=(changed_topic,))
    changed_path = tmp_path / f"changed-{change}" / "generation_handoff_manifest.json"
    write_generation_handoff(changed_path, changed_handoff)

    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(changed_path, work_dir, mode="resume"),
            planner=RecordingBackend([]),
            judge=RecordingBackend([]),
        )


def test_resume_reports_stale_input_for_changed_evaluator_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "stale-input"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    monkeypatch.setattr(coverage_module, "EVALUATOR_SCHEMA_VERSION", "test-schema-v3")
    with pytest.raises(NuggetCoverageError) as caught:
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]),
            judge=RecordingBackend([]),
        )

    assert caught.value.reason == (
        "input artifact is stale for the current evaluator contract or "
        "authenticated handoff projection"
    )
    assert NARRATIVE not in str(caught.value)


def test_resume_rejects_changed_prompt_or_schema_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "identity"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    monkeypatch.setattr(coverage_module, "PLANNER_PROMPT_VERSION", "changed-prompt")
    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]), judge=RecordingBackend([]),
        )


def test_openrouter_rejects_reflected_credential_without_leaking_it() -> None:
    secret = "actual-fake-credential"
    transport = FakeOpenRouterTransport([
        _openrouter_response({"schema_version": "x", "reflected": secret})
    ])
    backend = OpenRouterCoverageBackend(environ={"OPENROUTER_API_KEY": secret}, transport=transport)

    with pytest.raises(RuntimeError) as caught:
        backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert secret not in str(caught.value)
    assert len(transport.requests) == 1


def test_manifest_is_published_last_and_cli_create_then_resume_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "cli"
    published: list[str] = []
    original_publish = coverage_module._publish_once

    def recording_publish(path: Path, payload: dict[str, object]) -> str:
        published.append(Path(path).name)
        return original_publish(path, payload)

    monkeypatch.setattr(coverage_module, "_publish_once", recording_publish)
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)

    class FakeHostedBackend:
        def complete(self, request: CoverageModelRequest) -> BackendReply:
            if request.stage == "planner":
                return _reply(VALID_PLAN)
            return _reply(_judge_payload(frozen))

    monkeypatch.setattr(coverage_module, "OpenRouterCoverageBackend", FakeHostedBackend)
    create_code = main([
        "--handoff-manifest", str(handoff_path), "--topic", "topic-coverage",
        "--work-dir", str(work_dir), "--allow-hosted-calls",
    ])
    create_output = capsys.readouterr().out
    resume_code = main([
        "--handoff-manifest", str(handoff_path), "--topic", "topic-coverage",
        "--work-dir", str(work_dir), "--mode", "resume",
    ])
    resume_output = capsys.readouterr().out

    assert create_code == 0 and resume_code == 0
    assert json.loads(create_output)["status"] == "complete"
    assert json.loads(resume_output)["status"] == "complete"
    assert published == ["input.json", "plan.json", "judgments.json", "report.json", "manifest.json"]


def test_cli_redacts_obligation_span_and_credential_text_on_safe_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    code = main([
        "--handoff-manifest", str(handoff_path), "--topic", "topic-coverage",
        "--work-dir", str(tmp_path / "safe"), "--planner-model", "actual-fake-credential",
    ])
    output = capsys.readouterr().out

    assert code != 0
    for sensitive in (
        NARRATIVE,
        "Describe the projected cost and its assumptions.",
        "cost and assumptions",
        "actual-fake-credential",
    ):
        assert sensitive not in output


def test_resume_rejects_changed_evaluator_schema_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    work_dir = tmp_path / "schema-identity"
    frozen = validate_and_freeze_plan(NARRATIVE, VALID_PLAN)
    run_coverage_evaluation(
        _coverage_config(handoff_path, work_dir),
        planner=RecordingBackend([_reply(VALID_PLAN)]),
        judge=RecordingBackend([_reply(_judge_payload(frozen))]),
    )

    monkeypatch.setattr(coverage_module, "EVALUATOR_SCHEMA_VERSION", "changed-schema")
    with pytest.raises(NuggetCoverageError):
        run_coverage_evaluation(
            _coverage_config(handoff_path, work_dir, mode="resume"),
            planner=RecordingBackend([]), judge=RecordingBackend([]),
        )


@pytest.mark.parametrize("reason", [TimeoutError("temporary"), ConnectionError("temporary")])
def test_openrouter_retries_transient_url_errors(reason: Exception) -> None:
    transport = FakeOpenRouterTransport([
        URLError(reason),
        _openrouter_response(VALID_PLAN),
    ])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=2
    )

    backend.complete(render_planner_request(NARRATIVE, "planner-model"))

    assert len(transport.requests) == 2


@pytest.mark.parametrize(
    "reason",
    [ValueError("permanent configuration failure"), ssl.SSLError("certificate verify failed")],
)
def test_openrouter_does_not_retry_permanent_or_certificate_url_errors(reason: Exception) -> None:
    transport = FakeOpenRouterTransport([URLError(reason)])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=3
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert len(transport.requests) == 1


def test_openrouter_retries_temporary_dns_url_errors() -> None:
    transport = FakeOpenRouterTransport([
        URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS failure")),
        _openrouter_response(VALID_PLAN),
    ])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=2
    )

    backend.complete(render_planner_request(NARRATIVE, "planner-model"))

    assert len(transport.requests) == 2


@pytest.mark.parametrize("dns_errno", [socket.EAI_NONAME, socket.EAI_FAIL])
def test_openrouter_does_not_retry_permanent_dns_url_errors(dns_errno: int) -> None:
    transport = FakeOpenRouterTransport([
        URLError(socket.gaierror(dns_errno, "permanent DNS failure")),
    ])
    backend = OpenRouterCoverageBackend(
        environ={"OPENROUTER_API_KEY": "key"}, transport=transport, transport_max_attempts=3
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        backend.complete(render_planner_request(NARRATIVE, "planner-model"))
    assert len(transport.requests) == 1


def test_cli_redacts_runtime_obligation_span_and_credential_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff_path = _write_coverage_handoff(tmp_path)
    sensitive_plan = json.loads(json.dumps(VALID_PLAN))
    sensitive_plan["facets"][0]["obligations"][0]["requirement"] = "obligation-secret"
    sensitive_plan["facets"][0]["obligations"][0]["support_test"] = "span-secret"
    credential = "actual-fake-credential"

    class FailingAfterPlanBackend:
        def complete(self, request: CoverageModelRequest) -> BackendReply:
            if request.stage == "planner":
                return _reply(sensitive_plan)
            raise RuntimeError(f"{credential}: obligation-secret cost and assumptions")

    monkeypatch.setattr(coverage_module, "OpenRouterCoverageBackend", FailingAfterPlanBackend)
    code = main([
        "--handoff-manifest", str(handoff_path), "--topic", "topic-coverage",
        "--work-dir", str(tmp_path / "runtime-safe"), "--allow-hosted-calls",
    ])
    output = capsys.readouterr().out

    assert code != 0
    for sensitive in (
        credential,
        "obligation-secret",
        "span-secret",
        "cost and assumptions",
    ):
        assert sensitive not in output

from __future__ import annotations

import hashlib
import json

import pytest

from trec_rag.retrieval_nugget_coverage import (
    BackendReply,
    CoverageNugget,
    CoverageModelRequest,
    EvaluatorIdentity,
    FrozenPlan,
    NuggetCoverageError,
    evaluate_nugget_coverage,
    render_judge_request,
    render_planner_request,
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
        schema_version="retrieval_nugget_coverage_v1",
        planner_prompt_version="planner-v1",
        judge_prompt_version="judge-v1",
        planner_model="planner-model",
        judge_model="judge-model",
    )


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

from __future__ import annotations

import hashlib
import json

import pytest

from trec_rag.retrieval_nugget_coverage import (
    CoverageNugget,
    EvaluatorIdentity,
    FrozenPlan,
    NuggetCoverageError,
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

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import trec_rag.adaptive_evidence_discovery as discovery_module

from trec_rag.adaptive_evidence_discovery import (
    _attach_contract_folds,
    _build_derived_query,
    _build_discovery_messages,
    _build_phrase_controls,
    _build_reservoirs,
    _extract_repeated_phrases,
    _finalize_discovery_records,
    _freeze_o1,
    _merge_nuggets,
    _qualify_opposite_support,
    _validate_model_response,
    _validate_nugget_atomicity,
    _validate_proposal,
    finalize_discovery_records,
    record_discovery_unavailable,
    run_discovery_preflight,
    run_discovery_validation,
    run_proposal_pass,
    verify_discovery_terminal,
)
from trec_rag.adaptive_evidence_local_model import (
    DISCOVERY_SCHEMA,
    LocalJsonModel,
)


def _o0() -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "obligation_id": "219-positive",
            "kind": "o0",
            "text": "Positive effects of technology on society and daily life.",
            "anchor_terms": ["technology"],
            "relation_terms": ["positive", "society", "daily life"],
        }
    ]


def _passages(count: int) -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "variant": "219-positive",
            "document_id": f"doc-{index // 2:02d}",
            "fold": (index // 2) % 2,
            "score": 100.0 - index,
            "window_id": f"window-{index:02d}",
            "window_text": f"technology produces positive social effect {index}",
        }
        for index in range(count * 2)
    ]


def _parent() -> dict[str, object]:
    return {
        "topic_id": "219",
        "obligation_id": "219-positive",
        "kind": "o0",
        "text": "Positive effects of technology on society and daily life.",
        "anchor_terms": ["technology"],
        "relation_terms": ["positive", "society", "daily life"],
        "population_terms": ["society", "daily life"],
        "domain_terms": ["technology"],
    }


def _proposal(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "proposal_id": "219-positive:f0:0",
        "topic_id": "219",
        "parent_id": "219-positive",
        "kind": "o1",
        "label": "technology accessibility benefits",
        "scope_rationale": "A category of positive effects within the parent scope.",
        "document_id": "source-doc",
        "fold": 0,
        "support_span": "technology improves accessibility",
        "subject": "technology",
        "population": "society",
        "domain": "technology",
        "relation": "positive effects",
    }
    value.update(changes)
    return value


def _support() -> list[dict[str, object]]:
    return [
        {
            "document_id": "other-doc",
            "fold": 1,
            "qualified": True,
            "support_span": "technology improves accessibility",
        }
    ]


def _nugget(text: str) -> dict[str, object]:
    return {
        "topic_id": "219",
        "parent_id": "219-positive",
        "subject": "technology",
        "relation": "improves",
        "object": text,
        "support_span": text,
        "document_id": text,
        "fold": 0,
    }


def _atomic_nugget(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "topic_id": "219",
        "parent_id": "219-positive",
        "subject": "technology",
        "relation": "improves",
        "object": "information access",
        "support_span": "Technology improves information access.",
        "support_document_id": "doc-1",
        "document_id": "doc-1",
        "fold": 0,
    }
    value.update(changes)
    return value


def test_reservoir_has_ten_distinct_docs_per_parent_fold() -> None:
    rows = _build_reservoirs(_o0(), _passages(30), limit=10)
    assert len(rows[("219-positive", 0)]) == 10
    assert len({row["document_id"] for row in rows[("219-positive", 0)]}) == 10


def test_authenticated_contract_folds_are_joined_to_base_score_rows() -> None:
    rows = [
        {
            "topic_id": "219",
            "document_id": "doc-1",
            "variant": "219-positive",
            "score": 1.0,
        }
    ]
    documents = [{"topic_id": "219", "document_id": "doc-1", "fold": 1}]
    joined = _attach_contract_folds(rows, documents)
    assert joined[0]["fold"] == 1
    assert "fold" not in rows[0]


def test_answer_fact_cannot_become_o1() -> None:
    proposal = _proposal(label="COVID-19 caused a 42 percent increase", kind="o1")
    decision = _validate_proposal(
        proposal,
        _parent(),
        opposite_fold_support=_support(),
    )
    assert decision["accepted"] is False
    assert "candidate_answer" in decision["reasons"]


@pytest.mark.parametrize(
    "label",
    [
        "technology improved access for 42 percent of users",
        "technology saved $5 billion",
        "technology reduced delays in January 2025",
        "technology caused better health outcomes",
        "the program improves student performance",
        "the system produced 12 megawatts",
        "technology is harmful",
        "the policy began on Monday",
        "three million users adopted technology",
    ],
)
def test_fact_like_or_concrete_answer_labels_cannot_become_o1(label: str) -> None:
    decision = _validate_proposal(
        _proposal(label=label),
        _parent(),
        opposite_fold_support=_support(),
    )

    assert decision["accepted"] is False
    assert "candidate_answer" in decision["reasons"]


@pytest.mark.parametrize(
    "label",
    [
        "technology boosted access for dozens of users",
        "technology brought access",
        "technology enables access",
        "technology improving access",
        "technology responded with benefits",
        "it provides access",
    ],
)
def test_o1_positive_grammar_rejects_assertions_outside_verb_denylist(
    label: str,
) -> None:
    decision = _validate_proposal(
        _proposal(label=label),
        _parent(),
        opposite_fold_support=_support(),
    )

    assert decision["accepted"] is False
    assert "candidate_answer" in decision["reasons"]


@pytest.mark.parametrize(
    "label",
    [
        "information accessibility benefits",
        "social connection benefits",
        "educational access",
    ],
)
def test_short_abstract_in_scope_category_labels_remain_valid(label: str) -> None:
    decision = _validate_proposal(
        _proposal(label=label),
        _parent(),
        opposite_fold_support=_support(),
    )

    assert decision == {"accepted": True, "reasons": []}


@pytest.mark.parametrize(
    ("parent", "proposal"),
    [
        (
            {
                "topic_id": "219",
                "obligation_id": "219-positive",
                "text": "Technology accessibility benefits for society.",
                "anchor_terms": ["technology"],
                "relation_terms": ["accessibility", "benefits"],
                "population_terms": ["society"],
                "domain_terms": ["technology"],
            },
            {
                "parent_id": "219-positive",
                "label": "digital accessibility benefits",
                "subject": "technology",
                "population": "society",
                "domain": "technology",
                "relation": "accessibility benefits",
            },
        ),
        (
            {
                "topic_id": "300",
                "obligation_id": "300-safety",
                "text": "Vaccine safety risks for public health.",
                "anchor_terms": ["vaccine"],
                "relation_terms": ["safety", "risks"],
                "population_terms": ["public health"],
                "domain_terms": ["vaccine"],
            },
            {
                "topic_id": "300",
                "parent_id": "300-safety",
                "label": "vaccine safety risks",
                "subject": "vaccine",
                "population": "public health",
                "domain": "vaccine",
                "relation": "safety risks",
            },
        ),
        (
            {
                "topic_id": "84",
                "obligation_id": "84-allocation",
                "text": "Device allocation mechanisms for limited supply.",
                "anchor_terms": ["device"],
                "relation_terms": ["allocation", "mechanisms"],
                "population_terms": ["limited supply"],
                "domain_terms": ["device"],
            },
            {
                "topic_id": "84",
                "parent_id": "84-allocation",
                "label": "device allocation mechanisms",
                "subject": "device",
                "population": "limited supply",
                "domain": "device",
                "relation": "allocation mechanisms",
            },
        ),
    ],
)
def test_positive_o1_abstract_categories_are_valid_across_parents(
    parent: dict[str, object],
    proposal: dict[str, object],
) -> None:
    decision = _validate_proposal(
        _proposal(**proposal),
        parent,
        opposite_fold_support=_support(),
    )

    assert decision == {"accepted": True, "reasons": []}


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"population": "hospital patients"}, "scope"),
        ({"population": "children"}, "scope"),
        ({"domain": "clinical medicine"}, "scope"),
        ({"domain": "financial markets"}, "scope"),
        ({"domain": ""}, "schema"),
    ],
)
def test_o1_population_and_domain_must_stay_inside_frozen_parent_scope(
    changes: dict[str, object],
    expected_reason: str,
) -> None:
    decision = _validate_proposal(
        _proposal(**changes),
        _parent(),
        opposite_fold_support=_support(),
    )

    assert decision["accepted"] is False
    assert expected_reason in decision["reasons"]


def test_o1_requires_distinct_opposite_fold_document() -> None:
    decision = _validate_proposal(
        _proposal(),
        _parent(),
        opposite_fold_support=[],
    )
    assert decision["accepted"] is False
    assert "cross_fold_support" in decision["reasons"]


def test_nugget_merging_preserves_singletons_and_merges_jaccard_080() -> None:
    merged = _merge_nuggets(
        [
            _nugget("a b c d e f g h i"),
            _nugget("a b c d e f g h j"),
            _nugget("rare x"),
        ]
    )
    assert len(merged) == 2
    assert any(row["singleton"] is True for row in merged)


def test_local_json_model_fails_terminal_only_before_runtime_access() -> None:
    class ForbiddenRuntime:
        @property
        def torch(self) -> object:
            raise AssertionError("model runtime was accessed")

    with pytest.raises(RuntimeError, match="^discovery v1 is terminal-only$"):
        LocalJsonModel(runtime=ForbiddenRuntime())


def test_local_json_generate_fails_terminal_only_before_prompt_access() -> None:
    class ForbiddenMessages:
        def __iter__(self) -> object:
            raise AssertionError("prompt messages were accessed")

    adapter = LocalJsonModel.__new__(LocalJsonModel)
    with pytest.raises(RuntimeError, match="^discovery v1 is terminal-only$"):
        adapter.generate(ForbiddenMessages(), DISCOVERY_SCHEMA)  # type: ignore[arg-type]


def test_local_json_receipt_fails_terminal_only_before_runtime_access() -> None:
    adapter = LocalJsonModel.__new__(LocalJsonModel)
    with pytest.raises(RuntimeError, match="^discovery v1 is terminal-only$"):
        adapter.execution_receipt()


def test_model_response_rejects_spans_not_in_supplied_passage() -> None:
    passages = [
        {
            "document_id": "doc-1",
            "fold": 0,
            "passage_text": "Technology can improve access to information.",
        }
    ]
    response = {
        "status": "supported",
        "o1": [
            {
                "label": "information access benefits",
                "scope_rationale": "A positive societal effect of technology.",
                "subject": "technology",
                "population": "society",
                "relation": "positive effects",
                "support_document_id": "doc-1",
                "support_span": "outside knowledge",
            }
        ],
        "n1": [],
    }

    accepted, rejected = _validate_model_response(response, passages)

    assert accepted == {"o1": [], "n1": []}
    assert rejected[0]["reason"] == "support_span"


def test_n1_atomicity_accepts_one_supported_subject_relation_object_fact() -> None:
    decision = _validate_nugget_atomicity(_atomic_nugget())

    assert decision == {"accepted": True, "reasons": []}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"subject": "technology and education"}, "subject_coordination"),
        ({"relation": "improves and expands"}, "relation_coordination"),
        ({"object": "information access and health care"}, "object_coordination"),
        (
            {"object": "information access, health care, education"},
            "object_coordination",
        ),
        (
            {
                "support_span": (
                    "Technology improves information access. It also lowers costs."
                )
            },
            "support_multiple_sentences",
        ),
        (
            {
                "object": "information access",
                "support_span": (
                    "Technology improves information access while lowering costs."
                ),
            },
            "support_multiple_clauses",
        ),
        ({"support_span": "Technology is widely used."}, "unsupported_sro"),
    ],
)
def test_n1_atomicity_rejects_compound_or_unsupported_claims(
    changes: dict[str, object],
    reason: str,
) -> None:
    decision = _validate_nugget_atomicity(_atomic_nugget(**changes))

    assert decision["accepted"] is False
    assert reason in decision["reasons"]


@pytest.mark.parametrize(
    "support_span",
    [
        "Technology improves access. it lowers costs.",
        "Technology improves access! it lowers costs.",
        "Technology improves access? it lowers costs.",
    ],
)
def test_n1_atomicity_rejects_internal_terminator_before_lowercase_sentence(
    support_span: str,
) -> None:
    decision = _validate_nugget_atomicity(
        _atomic_nugget(object="access", support_span=support_span)
    )

    assert decision["accepted"] is False
    assert "support_multiple_sentences" in decision["reasons"]


@pytest.mark.parametrize(
    "support_span",
    [
        "Technology improves information access: costs decline.",
        "Technology improves information access — costs decline.",
        "Technology improves information access – costs decline.",
        "Technology improves information access -- costs decline.",
        "Technology improves information access\ncosts decline.",
        "Technology improves information access when networks expand.",
        "Technology improves information access if networks expand.",
        "Technology improves information access after networks expand.",
        "Technology improves information access before networks expand.",
    ],
)
def test_n1_atomicity_rejects_delimiters_and_subordinate_clauses(
    support_span: str,
) -> None:
    decision = _validate_nugget_atomicity(_atomic_nugget(support_span=support_span))

    assert decision["accepted"] is False
    assert "support_multiple_clauses" in decision["reasons"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"subject": "technology: education"}, "subject_coordination"),
        ({"relation": "improves when available"}, "relation_coordination"),
        (
            {"object": "information access — lower costs"},
            "object_coordination",
        ),
        ({"object": "information access\nlower costs"}, "object_coordination"),
    ],
)
def test_n1_atomicity_applies_nonatomic_checks_to_each_sro_field(
    changes: dict[str, object],
    reason: str,
) -> None:
    decision = _validate_nugget_atomicity(_atomic_nugget(**changes))

    assert decision["accepted"] is False
    assert reason in decision["reasons"]


@pytest.mark.parametrize(
    "support_span",
    [
        "Technology improves information access",
        "Technology improves information access.",
        "Technology improves information access!",
        "Technology improves information access?",
    ],
)
def test_n1_atomicity_allows_at_most_one_trailing_terminator(
    support_span: str,
) -> None:
    decision = _validate_nugget_atomicity(_atomic_nugget(support_span=support_span))

    assert decision == {"accepted": True, "reasons": []}


def test_n1_atomicity_allows_single_hyphen_inside_atomic_terms() -> None:
    decision = _validate_nugget_atomicity(
        _atomic_nugget(
            object="cost-effective access",
            support_span="Technology improves cost-effective access.",
        )
    )

    assert decision == {"accepted": True, "reasons": []}


def test_model_response_rejects_non_atomic_n1_before_acceptance() -> None:
    passages = [
        {
            "document_id": "doc-1",
            "fold": 0,
            "passage_text": "Technology improves access and reduces costs.",
        }
    ]
    response = {
        "status": "supported",
        "o1": [],
        "n1": [
            {
                "subject": "technology",
                "relation": "improves and reduces",
                "object": "access and costs",
                "support_document_id": "doc-1",
                "support_span": "Technology improves access and reduces costs.",
            }
        ],
    }

    accepted, rejected = _validate_model_response(response, passages)

    assert accepted["n1"] == []
    assert rejected[0]["reason"] == "n1_atomicity"
    assert "relation_coordination" in rejected[0]["atomicity_reasons"]


def test_repeated_phrase_control_requires_two_documents_across_folds() -> None:
    rows = _extract_repeated_phrases(
        [
            {
                "document_id": "a",
                "fold": 0,
                "passage_text": "Technology improves daily information access.",
            },
            {
                "document_id": "b",
                "fold": 1,
                "passage_text": "Technology improves daily information access for families.",
            },
            {
                "document_id": "c",
                "fold": 0,
                "passage_text": "Technology improves daily information access.",
            },
        ]
    )
    by_phrase = {row["phrase"]: row for row in rows}
    assert by_phrase["technology improves"]["folds"] == [0, 1]
    assert by_phrase["technology improves"]["document_ids"] == ["a", "b", "c"]
    assert "families" not in by_phrase


def test_repeated_phrase_control_rejects_copied_content_with_distinct_ids() -> None:
    rows = _extract_repeated_phrases(
        [
            {
                "document_id": "original",
                "document_sha256": "a" * 64,
                "fold": 0,
                "passage_text": "Technology improves daily information access.",
            },
            {
                "document_id": "copied-under-new-id",
                "document_sha256": "b" * 64,
                "fold": 1,
                "passage_text": "Technology improves daily information access.",
            },
        ]
    )

    assert "technology improves" not in {row["phrase"] for row in rows}


def _phrase_control_parent() -> dict[str, object]:
    return {
        "topic_id": "219",
        "obligation_id": "219-accessibility",
        "text": "Technology accessibility benefits for society.",
        "anchor_terms": ["technology"],
        "relation_terms": ["accessibility", "benefits", "society"],
        "population_terms": ["society"],
        "domain_terms": ["technology"],
        "wrong_domain_patterns": ["stock price"],
    }


def test_fresh_phrase_control_preflight_has_an_accepted_control() -> None:
    parent = _phrase_control_parent()
    rows = _build_phrase_controls(
        [parent],
        [
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-a",
                "fold": 0,
                "passage_text": (
                    "Technology accessibility benefits improve society."
                ),
            },
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-b",
                "fold": 1,
                "passage_text": (
                    "Research describes technology accessibility benefits broadly."
                ),
            },
        ],
    )

    by_phrase = {row["phrase"]: row for row in rows}
    control = by_phrase["technology accessibility benefits"]
    assert control["content_distinct"] is True
    assert control["scope_preserved"] is True
    assert control["abstract_category"] is True
    assert control["accepted_control"] is True
    assert sum(row["accepted_control"] is True for row in rows) > 0


def test_phrase_control_rejects_copied_content_before_scope_acceptance() -> None:
    parent = _phrase_control_parent()
    copied = "Technology accessibility benefits improve society."
    rows = _build_phrase_controls(
        [parent],
        [
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-a",
                "fold": 0,
                "passage_text": copied,
            },
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-b",
                "fold": 1,
                "passage_text": copied,
            },
        ],
    )

    assert rows == []


@pytest.mark.parametrize(
    "phrase",
    [
        "medical treatment risks",
        "technology stock price risks",
    ],
)
def test_phrase_control_rejects_wrong_scope_or_wrong_domain(phrase: str) -> None:
    parent = _phrase_control_parent()
    rows = _build_phrase_controls(
        [parent],
        [
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-a",
                "fold": 0,
                "passage_text": f"{phrase} appear in one source.",
            },
            {
                "parent_id": "219-accessibility",
                "document_id": "doc-b",
                "fold": 1,
                "passage_text": f"Researchers discuss {phrase} broadly.",
            },
        ],
    )

    control = {row["phrase"]: row for row in rows}[phrase]
    assert control["scope_preserved"] is False
    assert control["accepted_control"] is False


def test_freeze_o1_is_lexicographic_and_caps_parent_and_topic() -> None:
    rows: list[dict[str, object]] = []
    for index in range(6):
        rows.append(
            {
                "accepted": True,
                "topic_id": "219",
                "parent_id": "219-positive" if index < 2 else f"219-p{index}",
                "label": "z label" if index == 0 else f"label {index}",
                "validating_document_count": 2,
                "independent_stream_count": 1,
                "source_diversity": 1,
                "parent_local_rank": index,
            }
        )

    accepted, rejected = _freeze_o1(rows)

    assert len(accepted) == 4
    assert len({row["parent_id"] for row in accepted}) == 4
    assert all("freeze_limit" in row["reasons"] for row in rejected)


def test_freeze_o1_uses_normalized_label_before_quality_metrics() -> None:
    rows = [
        {
            "accepted": True,
            "topic_id": "219",
            "parent_id": "219-positive",
            "proposal_id": "zulu",
            "label": "Zulu category",
            "validating_document_count": 99,
            "independent_stream_count": 99,
            "source_diversity": 99,
            "parent_local_rank": 0,
        },
        {
            "accepted": True,
            "topic_id": "219",
            "parent_id": "219-positive",
            "proposal_id": "alpha",
            "label": "  ALPHA---category ",
            "validating_document_count": 1,
            "independent_stream_count": 1,
            "source_diversity": 1,
            "parent_local_rank": 50,
        },
    ]

    accepted, rejected = _freeze_o1(rows)

    assert [row["proposal_id"] for row in accepted] == ["alpha"]
    assert [row["proposal_id"] for row in rejected] == ["zulu"]


def test_freeze_o1_applies_topic_cap_after_exact_lexical_order() -> None:
    rows = [
        {
            "accepted": True,
            "topic_id": "219",
            "parent_id": f"219-parent-{index}",
            "proposal_id": label.casefold(),
            "label": label,
            "validating_document_count": 10 - index,
            "independent_stream_count": 2,
            "source_diversity": 2,
            "parent_local_rank": index,
        }
        for index, label in enumerate(
            [
                "Zulu category",
                "bravo category",
                "Alpha category",
                "delta category",
                "charlie category",
            ]
        )
    ]

    accepted, rejected = _freeze_o1(rows)

    assert [row["label"] for row in accepted] == [
        "Alpha category",
        "bravo category",
        "charlie category",
        "delta category",
    ]
    assert [row["label"] for row in rejected] == ["Zulu category"]


def test_derived_query_contains_full_narrative_parent_and_heading() -> None:
    broad = {"text": "full narrative"}
    parent = {"text": "complete parent O0"}
    query = _build_derived_query(broad, parent, "information access")
    assert query == (
        "full narrative\n\nExplicit obligation:\ncomplete parent O0"
        "\n\nCorpus-derived sub-obligation:\ninformation access"
    )


def test_discovery_prompt_freezes_scope_spans_and_no_outside_knowledge() -> None:
    passages = [
        {
            "document_id": "doc-1",
            "fold": 0,
            "passage_text": "Technology improves information access.",
        }
    ]
    messages = _build_discovery_messages(_parent(), passages)
    content = "\n".join(message["content"] for message in messages)
    assert "preserve the parent subject, population, domain, and relation" in content
    assert "abstract O1 categories separately from specific N1 facts" in content
    assert "exact substrings from the supplied passages" in content
    assert "never use outside knowledge" in content
    assert "return unsupported rather than inventing evidence" in content
    assert "Technology improves information access." in content


def test_discovery_prompt_embeds_exact_authoritative_frozen_schema() -> None:
    messages = _build_discovery_messages(
        _parent(),
        [
            {
                "document_id": "doc-1",
                "fold": 0,
                "passage_text": "Technology improves information access.",
            }
        ],
    )
    payload = json.loads(messages[1]["content"])
    assert payload["response_json_schema"] == DISCOVERY_SCHEMA
    assert "response_json_schema is authoritative" in messages[0]["content"]
    assert "exactly one JSON object" in messages[0]["content"]


def test_deterministic_support_uses_exact_opposite_fold_anchor_relation_sentence() -> None:
    proposal = _proposal(
        label="information access benefits",
        source_fold=0,
        source_document_sha256="source-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "same-fold",
            "document_sha256": "same-fold-sha",
            "fold": 0,
            "score": 9.0,
            "window_id": "same",
            "window_text": "Technology has positive effects on society.",
        },
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "opposite",
            "document_sha256": "opposite-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "opposite-window",
            "window_text": (
                "Unrelated preface. Technology improves information access, "
                "a positive effect for society. Another sentence."
            ),
        },
    ]

    support = _qualify_opposite_support(proposal, _parent(), scores)

    assert len(support) == 1
    assert support[0]["document_id"] == "opposite"
    assert support[0]["fold"] == 1
    assert support[0]["qualified"] is True
    assert support[0]["support_span"] == (
        "Technology improves information access, a positive effect for society."
    )
    assert support[0]["support_span"] in scores[1]["window_text"]


def test_deterministic_support_rejects_duplicate_or_incoherent_passage() -> None:
    proposal = _proposal(
        source_fold=0,
        source_document_sha256="duplicate-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "duplicate-copy",
            "document_sha256": "duplicate-sha",
            "fold": 1,
            "score": 9.0,
            "window_id": "duplicate",
            "window_text": "Technology has positive effects on society.",
        },
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "incoherent",
            "document_sha256": "different-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "incoherent",
            "window_text": "A product review discusses technology stock prices.",
        },
    ]

    assert _qualify_opposite_support(proposal, _parent(), scores) == []


class _ForbiddenPath:
    def __fspath__(self) -> str:
        raise AssertionError("filesystem path was accessed")


@pytest.mark.parametrize(
    "operation",
    [
        lambda path: run_discovery_preflight(
            contract_dir=path,
            scores_dir=path,
            output_dir=path,
        ),
        lambda path: run_proposal_pass(path),
        lambda path: record_discovery_unavailable(path),
        lambda path: run_discovery_validation(
            output_dir=path,
            opposite_fold_scores=path,
        ),
    ],
    ids=["preflight", "proposal", "terminal-record", "validation"],
)
def test_public_mutation_boundaries_fail_terminal_only_before_filesystem_access(
    operation: object,
) -> None:
    with pytest.raises(RuntimeError, match="^discovery v1 is terminal-only$"):
        operation(_ForbiddenPath())  # type: ignore[operator]


def test_public_finalization_fails_terminal_only_before_input_access() -> None:
    class ForbiddenSequence:
        def __iter__(self) -> object:
            raise AssertionError("finalization input was accessed")

    forbidden = ForbiddenSequence()
    with pytest.raises(RuntimeError, match="^discovery v1 is terminal-only$"):
        finalize_discovery_records(
            proposals=forbidden,  # type: ignore[arg-type]
            nuggets=forbidden,  # type: ignore[arg-type]
            parents=forbidden,  # type: ignore[arg-type]
            broad_by_topic={},
            score_rows=forbidden,  # type: ignore[arg-type]
        )


def test_discovery_cli_exposes_only_terminal_verify_and_inspect(capsys: object) -> None:
    with pytest.raises(SystemExit) as excinfo:
        discovery_module.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "{verify,inspect}" in output
    for disabled in ("preflight", "propose", "validate", "finalize", "retry"):
        assert disabled not in output


def test_discovery_v1_public_api_is_read_only() -> None:
    assert discovery_module.__all__ == (
        "inspect_discovery_terminal",
        "verify_discovery_terminal",
        "main",
    )


@pytest.mark.parametrize(
    "command",
    ["preflight", "propose", "validate", "finalize", "retry", "record"],
)
def test_discovery_cli_mutation_commands_are_unreachable(
    command: str,
    capsys: object,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        discovery_module.main([command])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_discovery_cli_routes_both_read_only_terminal_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: object,
) -> None:
    calls: list[Path] = []

    def fake_verify(path: Path) -> dict[str, object]:
        calls.append(path)
        return {
            "status": "discovery_unavailable",
            "completed_proposal_pass_count": 0,
            "total_qwen_proposal_generation_call_count": 2,
        }

    monkeypatch.setattr(discovery_module, "verify_discovery_terminal", fake_verify)

    assert discovery_module.main(["verify", "--output", str(tmp_path)]) == 0
    verified_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "status=discovery_unavailable" in verified_output
    assert discovery_module.main(["inspect", "--output", str(tmp_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspected["status"] == "discovery_unavailable"
    assert calls == [tmp_path, tmp_path]


def _canonical_terminal_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery"
    )


def _copy_canonical_terminal(tmp_path: Path) -> Path:
    root = tmp_path / "discovery"
    shutil.copytree(_canonical_terminal_root(), root)
    return root


def _write_pretty_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_terminal_cross_hashes(root: Path, preflight: dict[str, object]) -> None:
    _write_pretty_json(root / "preflight.json", preflight)
    integration_path = root / "integration_preflight_failure.json"
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    integration["preflight_sha256"] = _file_sha256(root / "preflight.json")
    _write_pretty_json(integration_path, integration)

    integration_sha256 = _file_sha256(integration_path)
    marker_path = root / "corrected_pass_started.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["integration_preflight_failure_sha256"] = integration_sha256
    _write_pretty_json(marker_path, marker)

    failure_path = root / "corrected_pass_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["corrected_pass_started_sha256"] = _file_sha256(marker_path)
    failure["integration_preflight_failure_sha256"] = integration_sha256
    _write_pretty_json(failure_path, failure)

    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bindings"]["preflight_sha256"] = _file_sha256(root / "preflight.json")
    receipt["bindings"]["integration_preflight_failure_sha256"] = integration_sha256
    receipt["artifacts"]["corrected_pass_started.json"]["sha256"] = (
        _file_sha256(marker_path)
    )
    receipt["artifacts"]["corrected_pass_failure.json"]["sha256"] = (
        _file_sha256(failure_path)
    )
    _write_pretty_json(receipt_path, receipt)

    verification_path = root / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["terminal_receipt_sha256"] = _file_sha256(receipt_path)
    _write_pretty_json(verification_path, verification)


@pytest.mark.parametrize(
    "required_name",
    [
        "broad.jsonl",
        "parents.jsonl",
        "reservoirs.jsonl",
        "repeated_phrases.jsonl",
        "verification.json",
    ],
)
def test_terminal_verifier_requires_every_canonical_artifact(
    tmp_path: Path,
    required_name: str,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    (root / required_name).unlink()

    with pytest.raises(ValueError, match="canonical inventory"):
        verify_discovery_terminal(root)


@pytest.mark.parametrize(
    "tampered_name",
    [
        "broad.jsonl",
        "parents.jsonl",
        "reservoirs.jsonl",
        "repeated_phrases.jsonl",
        "verification.json",
    ],
)
def test_terminal_verifier_authenticates_every_canonical_artifact(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    with (root / tampered_name).open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(ValueError, match="canonical artifact"):
        verify_discovery_terminal(root)


def test_terminal_verifier_checks_preflight_artifact_metadata(
    tmp_path: Path,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    preflight["artifacts"]["broad.jsonl"]["rows"] = 5
    _rewrite_terminal_cross_hashes(root, preflight)

    with pytest.raises(ValueError, match="preflight artifact metadata"):
        verify_discovery_terminal(root)


def test_terminal_verifier_rejects_protected_topic_with_recomputed_bindings(
    tmp_path: Path,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    preflight["topic_ids"] = ["144", "72", "300", "84"]
    _rewrite_terminal_cross_hashes(root, preflight)

    with pytest.raises(ValueError, match="protected topic 144"):
        verify_discovery_terminal(root)


def test_terminal_verifier_requires_exact_canonical_topic_scope(
    tmp_path: Path,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    preflight["topic_ids"] = ["999", "72", "300", "84"]
    _rewrite_terminal_cross_hashes(root, preflight)

    with pytest.raises(ValueError, match="canonical pilot topic scope"):
        verify_discovery_terminal(root)


@pytest.mark.parametrize(
    "missing_name",
    [
        "preflight.json",
        "integration_preflight_failure.json",
        "corrected_pass_started.json",
        "corrected_pass_failure.json",
    ],
)
def test_terminal_verifier_requires_complete_failure_history(
    tmp_path: Path,
    missing_name: str,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    (root / missing_name).unlink()

    with pytest.raises(ValueError, match="unreadable"):
        verify_discovery_terminal(root)


@pytest.mark.parametrize(
    "tampered_name",
    [
        "preflight.json",
        "integration_preflight_failure.json",
        "corrected_pass_started.json",
        "corrected_pass_failure.json",
    ],
)
def test_terminal_verifier_rejects_tampered_failure_history(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    (root / tampered_name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        verify_discovery_terminal(root)


def test_terminal_verifier_rejects_mismatched_receipt_artifact_hash(
    tmp_path: Path,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    receipt["bindings"]["preflight_sha256"] = "0" * 64
    (root / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact hash differs"):
        verify_discovery_terminal(root)


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "rejected_o1.jsonl",
        "prompt_receipts.jsonl",
        "partial-proposals",
        "../scoring/provisional-o1",
        "../scoring/o1",
    ],
)
def test_terminal_verifier_forbids_every_partial_or_downstream_artifact(
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    root = _copy_canonical_terminal(tmp_path)
    forbidden = root / forbidden_name
    if forbidden.suffix:
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_text("\n", encoding="utf-8")
    else:
        forbidden.mkdir(parents=True)

    with pytest.raises(ValueError, match="unexpectedly contains"):
        verify_discovery_terminal(root)


def test_checked_in_canonical_terminal_history_remains_verifiable() -> None:
    receipt = verify_discovery_terminal(_canonical_terminal_root())

    assert receipt["status"] == "discovery_unavailable"
    assert receipt["total_qwen_proposal_generation_call_count"] == 2
    assert receipt["completed_proposal_pass_count"] == 0
    assert receipt["proposed_o1_count"] == 0
    assert receipt["proposed_n1_count"] == 0
    assert receipt["validation_model_call_count"] == 0
    assert receipt["qrels_opened"] is False
    assert receipt["network_call_count"] == 0
    assert receipt["retrieval_call_count"] == 0
    assert receipt["hosted_inference_call_count"] == 0
    assert receipt["paid_call_count"] == 0
    assert receipt["external_cost_usd"] == 0.0


def test_finalize_records_uses_only_deterministic_opposite_fold_support() -> None:
    parent = _parent()
    proposal = _proposal(
        label="information access benefits",
        source_fold=0,
        source_document_id="source-doc",
        source_document_sha256="source-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "opposite",
            "document_sha256": "opposite-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "opposite-window",
            "window_text": (
                "Technology improves information access, a positive effect for society."
            ),
        }
    ]
    result = _finalize_discovery_records(
        proposals=[proposal],
        nuggets=[_nugget("technology improves information access")],
        parents=[parent],
        broad_by_topic={"219": {"text": "full narrative"}},
        score_rows=scores,
    )

    assert len(result["accepted_o1"]) == 1
    accepted = result["accepted_o1"][0]
    assert accepted["validation_method"] == "deterministic_opposite_fold_minilm"
    assert accepted["opposite_fold_support"][0]["document_id"] == "opposite"
    assert accepted["query"].endswith(
        "Corpus-derived sub-obligation:\ninformation access benefits"
    )
    assert len(result["accepted_n1"]) == 1


def test_finalize_records_emits_unsupported_without_corroboration() -> None:
    result = _finalize_discovery_records(
        proposals=[_proposal(source_fold=0)],
        nuggets=[],
        parents=[_parent()],
        broad_by_topic={"219": {"text": "full narrative"}},
        score_rows=[],
    )

    assert result["accepted_o1"] == []
    assert result["rejected_o1"][0]["status"] == "unsupported"
    assert "cross_fold_support" in result["rejected_o1"][0]["reasons"]

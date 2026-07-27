from __future__ import annotations

from trec_rag.topic213_coverage_repair_experiment import (
    _plan_facet_candidates,
    apply_audited_repairs,
    build_frozen_evidence_from_claims,
    select_covering_passages,
    select_diverse_candidates,
)


def _candidate(claim_id: str, text: str, passage_text: str) -> dict[str, object]:
    return {
        "candidate_id": claim_id,
        "text": text,
        "sub_narrative": "Facet",
        "supporting_passages": [
            {
                "passage_id": f"P-{claim_id}",
                "document_id": f"shard_{claim_id}",
                "text": passage_text,
            }
        ],
    }


def test_diverse_selection_rejects_paraphrases_and_keeps_distinct_concepts():
    candidates = [
        _candidate("A", "North Korea invaded South Korea in June 1950.", "North Korea invaded South Korea in June 1950."),
        _candidate("B", "The June 1950 war began when North Korea invaded South Korea.", "The war began when North Korea invaded South Korea in June 1950."),
        _candidate("C", "NSC-68 called for a large military buildup to contain Soviet expansion.", "NSC-68 called for a rapid military buildup to contain Soviet expansion."),
        _candidate("D", "The war became an issue in the 1952 presidential election.", "The Korean conflict became a central issue in the 1952 presidential election."),
    ]

    selected, reserves = select_diverse_candidates(candidates, quota=3, reserve_count=1)

    selected_ids = {row["candidate_id"] for row in selected}
    assert {"C", "D"} <= selected_ids
    assert len(selected_ids & {"A", "B"}) == 1
    assert len(reserves) == 1


def test_covering_passage_selection_uses_multiple_documents_for_composite_fact():
    passages = [
        {
            "passage_id": "P1",
            "document_id": "shard_1",
            "text": "MacArthur advanced United Nations forces beyond the 38th parallel.",
        },
        {
            "passage_id": "P2",
            "document_id": "shard_2",
            "text": "Chinese forces intervened after the advance approached the Yalu River.",
        },
        {
            "passage_id": "P3",
            "document_id": "shard_3",
            "text": "The Korean War began in June 1950.",
        },
    ]

    selected = select_covering_passages(
        "MacArthur's advance beyond the 38th parallel toward the Yalu River prompted Chinese intervention.",
        passages,
        maximum_citations=3,
    )

    assert {row["passage_id"] for row in selected} == {"P1", "P2"}
    assert len(selected) == 2


def test_frozen_evidence_uses_dynamic_quotas_and_is_nugget_blind():
    labels = ["Facet 1", "Facet 2"]
    claims = {
        label: [
            {
                **_candidate(f"{facet}-{number}", f"Supported fact {facet}-{number} is documented.", f"Supported fact {facet}-{number} is documented."),
                "claim_id": f"C{facet}{number}",
            }
            for number in range(1, quota + 1)
        ]
        for facet, (label, quota) in enumerate(zip(labels, [2, 3], strict=True), 1)
    }

    frozen = build_frozen_evidence_from_claims(
        topic_id="213",
        narrative="Explain the topic.",
        source_experiment_id="full-passages",
        labels=labels,
        claims_by_label=claims,
        sentence_quotas={"Facet 1": 2, "Facet 2": 3},
        facet_word_targets={"Facet 1": 40, "Facet 2": 60},
        minimum_words=90,
        maximum_words=110,
        source_metadata={"passages": "full set"},
        input_accounting={"passages_processed": 1478},
    )

    assert frozen["organizer_nuggets_available"] is False
    assert frozen["exact_total_sentence_count"] == 5
    assert [facet["sentence_quota"] for facet in frozen["facets"]] == [2, 3]
    assert all(
        1 <= len(claim["selected_citation_passages"]) <= 3
        for facet in frozen["facets"]
        for claim in facet["source_claims"]
    )


def test_audited_repair_replaces_rejected_sentence_without_dropping_slot():
    candidate = {
        "context_policy": {},
        "evidence_ledger": [
            {
                "claim_id": "A001",
                "sub_narrative": "Facet",
                "text": "MacArthur advanced beyond the 38th parallel.",
                "document_ids": ["shard_1"],
                "supporting_passages": [],
            }
        ],
        "sections": [
            {
                "sub_narrative": "Facet",
                "claims": [
                    {
                        "claim_id": "S001",
                        "text": "MacArthur recklessly advanced and inevitably caused Chinese intervention.",
                        "evidence_claim_ids": ["A001"],
                        "document_ids": ["shard_1"],
                        "supporting_passages": [],
                    }
                ],
            }
        ],
    }
    initial_audits = [{"claim_id": "S001", "status": "partially_supported"}]
    repairs = {
        "S001": "MacArthur advanced United Nations forces beyond the 38th parallel."
    }
    repair_audits = [{"claim_id": "S001", "status": "supported"}]

    final, kept, excluded = apply_audited_repairs(
        candidate,
        initial_audits=initial_audits,
        repair_text_by_claim_id=repairs,
        repair_audits=repair_audits,
    )

    claim = final["sections"][0]["claims"][0]
    assert claim["text"] == repairs["S001"]
    assert claim["evidence_claim_ids"] == ["A001"]
    assert len(kept) == 1
    assert excluded == []
    assert final["support_filter"] == {
        "candidate_sentence_count": 1,
        "submitted_sentence_count": 1,
        "excluded_sentence_count": 0,
        "repair_attempted_count": 1,
        "repair_retained_count": 1,
    }


def test_facet_planner_accepts_top_quota_and_appends_deterministic_reserves():
    class Planner:
        def complete_json(self, **_kwargs):
            return {"ranked_candidate_ids": ["C", "A"]}

    candidates = [
        {"candidate_id": "A", "text": "First fact."},
        {"candidate_id": "B", "text": "Second fact."},
        {"candidate_id": "C", "text": "Third fact."},
    ]

    ranked = _plan_facet_candidates(
        Planner(),
        label="Facet",
        candidates=candidates,
        quota=2,
        audit_config={"validation_attempts": 1},
    )

    assert [row["candidate_id"] for row in ranked] == ["C", "A", "B"]

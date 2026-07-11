from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

import trec_rag.det_sparse_v3_selection as selection_module
from trec_rag.det_sparse_v3_config import CANDIDATE_TOPIC_IDS, SELECTION_SEED
from trec_rag.det_sparse_v3_selection import (
    screen_and_select_structural_topics_v3,
    selection_digest,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery
from trec_rag.topics import Topic


FINGERPRINT = AnalyzerFingerprint(
    contract_version="selection-test",
    implementation="selection-test",
    lucene_version="test",
    analyzer_class="test",
    tokenizer="test",
    filters=(),
    stopword_sha256="0" * 64,
    unicode_version="test",
    index_id="test",
)


class Analyzer:
    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        tokens = tuple(text.lower().split())
        return AnalyzedQuery(tokens, tuple(dict.fromkeys(tokens)), FINGERPRINT)


@dataclass(frozen=True)
class Row:
    label: str


@dataclass(frozen=True)
class Anchor:
    query_text: str
    core_terms: tuple[str, ...]
    evidence_sha256: str
    core_sha256: str


@dataclass(frozen=True)
class Failure:
    code: str
    message: str


class FakePlan:
    def __init__(
        self,
        *,
        topic_id: str,
        narrative: str,
        eligible: bool,
        raw_anchorless: bool,
        final_anchorless: bool,
        unit_count: int = 2,
        facet_count: int = 2,
        core_count: int = 2,
    ) -> None:
        self.topic_id = topic_id
        self.status = "ok" if eligible else "fallback"
        self.narrative_sha256 = hashlib.sha256(narrative.encode("utf-8")).hexdigest()
        self.token_tape_sha256 = hashlib.sha256(
            f"tape:{topic_id}".encode("utf-8")
        ).hexdigest()
        self.analyzer_token_sha256 = hashlib.sha256(
            json.dumps(
                narrative.lower().split(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.analyzer_fingerprint_sha256 = hashlib.sha256(
            json.dumps(
                FINGERPRINT.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.original_query_text = narrative
        self.lexical_units = tuple(range(unit_count))
        self.facets = tuple(range(facet_count)) if eligible else ()
        self.merge_audit = tuple(range(max(0, unit_count - facet_count)))
        self.anchor = (
            Anchor(
                query_text="alpha beta",
                core_terms=tuple(f"core{index}" for index in range(core_count)),
                evidence_sha256=hashlib.sha256(
                    f"evidence:{topic_id}".encode("utf-8")
                ).hexdigest(),
                core_sha256=hashlib.sha256(
                    f"core:{topic_id}".encode("utf-8")
                ).hexdigest(),
            )
            if eligible
            else None
        )
        if eligible:
            self.raw_criticality = (
                Row("anchorless" if raw_anchorless and index == 0 else "complete")
                for index in range(unit_count - 1)
            )
            self.raw_criticality = tuple(self.raw_criticality)
            self.final_criticality = tuple(
                Row("anchorless" if final_anchorless and index == 0 else "complete")
                for index in range(facet_count - 1)
            )
            self.failure = None
        else:
            self.raw_criticality = ()
            self.final_criticality = ()
            self.failure = Failure("synthetic_ineligible", "synthetic ineligible plan")

    def to_dict(self):
        return {
            "topic_id": self.topic_id,
            "status": self.status,
            "narrative_sha256": self.narrative_sha256,
            "unit_count": len(self.lexical_units),
            "facet_count": len(self.facets),
            "raw_labels": [row.label for row in self.raw_criticality],
            "final_labels": [row.label for row in self.final_criticality],
            "failure": (
                None
                if self.failure is None
                else {"code": self.failure.code, "message": self.failure.message}
            ),
        }


def _topics():
    return tuple(
        Topic(
            id=topic_id,
            title=f"topic {topic_id}",
            narrative="alpha beta gamma delta epsilon",
        )
        for topic_id in CANDIDATE_TOPIC_IDS
    )


def _install_plans(
    monkeypatch,
    *,
    eligible_count: int = 9,
    raw_anchorless_ids: set[str] | None = None,
    final_anchorless_ids: set[str] | None = None,
):
    eligible_ids = set(CANDIDATE_TOPIC_IDS[:eligible_count])
    raw_ids = raw_anchorless_ids if raw_anchorless_ids is not None else {"14"}
    final_ids = final_anchorless_ids if final_anchorless_ids is not None else set(raw_ids)

    def build(topic_id, narrative, query_analyzer):
        del query_analyzer
        return FakePlan(
            topic_id=topic_id,
            narrative=narrative,
            eligible=topic_id in eligible_ids,
            raw_anchorless=topic_id in raw_ids,
            final_anchorless=topic_id in final_ids,
        )

    monkeypatch.setattr(selection_module, "_build_plan", build)


def _digest_for(topic: Topic) -> str:
    narrative_sha256 = hashlib.sha256(topic.narrative.encode("utf-8")).hexdigest()
    return selection_digest(
        topic_id=topic.id,
        narrative_sha256=narrative_sha256,
        seed=SELECTION_SEED,
    )


def test_v3_selection_is_critical_first_then_exact_half_open_thirds(monkeypatch):
    topics = _topics()
    _install_plans(monkeypatch)

    outcome = screen_and_select_structural_topics_v3(
        topics,
        query_analyzer=Analyzer(),
    )

    selection = outcome.selection
    assert selection.status == "ok"
    assert selection.critical_pool_topic_ids == ("14",)
    assert selection.critical_topic_id == "14"
    assert selection.remaining_ordered_topic_ids == CANDIDATE_TOPIC_IDS[1:]
    assert [(item.start, item.end) for item in selection.quantile_bins] == [
        (0, 2),
        (2, 5),
        (5, 8),
    ]
    expected_members = (
        CANDIDATE_TOPIC_IDS[1:3],
        CANDIDATE_TOPIC_IDS[3:6],
        CANDIDATE_TOPIC_IDS[6:9],
    )
    assert tuple(item.topic_ids for item in selection.quantile_bins) == expected_members
    expected_winners = tuple(
        min(
            (topic for topic in topics if topic.id in members),
            key=lambda topic: (_digest_for(topic), int(topic.id)),
        ).id
        for members in expected_members
    )
    assert selection.selected_topic_ids == ("14", *expected_winners)
    assert selection.provisional_selected_topic_ids == selection.selected_topic_ids

    screens = {row.topic_id: row for row in selection.screens}
    assert all(len(row.plan_semantic_sha256) == 64 for row in screens.values())
    assert all(len(row.selection_digest) == 64 for row in screens.values())
    assert screens["14"].raw_criticality.anchorless == 1
    assert screens["14"].final_criticality.anchorless == 1
    assert screens["14"].selection_role == "critical"
    for index, winner in enumerate(expected_winners):
        assert screens[winner].selection_role == f"bin{index}"
        assert screens[winner].quantile_bin == index


@pytest.mark.parametrize(
    ("eligible_count", "expected_slices"),
    [
        (4, ((0, 1), (1, 2), (2, 3))),
        (5, ((0, 1), (1, 2), (2, 4))),
        (6, ((0, 1), (1, 3), (3, 5))),
        (9, ((0, 2), (2, 5), (5, 8))),
    ],
)
def test_v3_quantile_slices_use_approved_floor_boundaries(
    monkeypatch,
    eligible_count,
    expected_slices,
):
    _install_plans(monkeypatch, eligible_count=eligible_count)

    outcome = screen_and_select_structural_topics_v3(
        _topics(),
        query_analyzer=Analyzer(),
    )

    assert outcome.selection.status == "ok"
    assert tuple(
        (item.start, item.end) for item in outcome.selection.quantile_bins
    ) == expected_slices


def test_v3_critical_digest_and_bin_digest_ties_break_only_by_numeric_id(
    monkeypatch,
):
    _install_plans(
        monkeypatch,
        raw_anchorless_ids={"14", "31"},
        final_anchorless_ids={"14", "31"},
    )
    monkeypatch.setattr(selection_module, "selection_digest", lambda **kwargs: "0" * 64)

    outcome = screen_and_select_structural_topics_v3(
        _topics(),
        query_analyzer=Analyzer(),
    )

    assert outcome.selection.status == "ok"
    assert outcome.selection.critical_topic_id == "14"
    assert outcome.selection.selected_topic_ids == ("14", "31", "72", "273")


def test_v3_selected_merge_masked_critical_stops_without_replacement(monkeypatch):
    topics = _topics()
    possible = {"14", "31"}
    chosen = min(
        (topic for topic in topics if topic.id in possible),
        key=lambda topic: (_digest_for(topic), int(topic.id)),
    ).id
    survivor = next(topic_id for topic_id in possible if topic_id != chosen)
    _install_plans(
        monkeypatch,
        raw_anchorless_ids=possible,
        final_anchorless_ids={survivor},
    )

    outcome = screen_and_select_structural_topics_v3(
        topics,
        query_analyzer=Analyzer(),
    )

    selection = outcome.selection
    assert selection.status == "failure"
    assert selection.failure is not None
    assert selection.failure.code == "selected_critical_merge_masked"
    assert selection.critical_topic_id == chosen
    assert selection.selected_topic_ids == ()
    assert len(selection.provisional_selected_topic_ids) == 4
    assert survivor not in selection.provisional_selected_topic_ids[:1]
    screens = {row.topic_id: row for row in selection.screens}
    assert screens[chosen].selection_role == "critical"
    assert screens[chosen].raw_criticality.anchorless == 1
    assert screens[chosen].final_criticality.anchorless == 0
    assert screens[survivor].final_criticality.anchorless == 1


def test_v3_selection_stops_for_insufficient_eligibility_or_no_critical_pool(
    monkeypatch,
):
    _install_plans(monkeypatch, eligible_count=3)
    too_few = screen_and_select_structural_topics_v3(
        _topics(),
        query_analyzer=Analyzer(),
    )
    assert too_few.selection.status == "failure"
    assert too_few.selection.failure is not None
    assert too_few.selection.failure.code == "fewer_than_four_eligible_topics"

    _install_plans(
        monkeypatch,
        eligible_count=9,
        raw_anchorless_ids=set(),
        final_anchorless_ids=set(),
    )
    no_critical = screen_and_select_structural_topics_v3(
        _topics(),
        query_analyzer=Analyzer(),
    )
    assert no_critical.selection.status == "failure"
    assert no_critical.selection.failure is not None
    assert no_critical.selection.failure.code == "no_anchor_critical_topic"
    assert no_critical.selection.selected_topic_ids == ()


def test_v3_selection_rejects_candidate_order_and_seed_drift():
    topics = _topics()
    with pytest.raises(ValueError, match="candidate universe/order"):
        screen_and_select_structural_topics_v3(
            topics,
            query_analyzer=Analyzer(),
            candidate_topic_ids=tuple(reversed(CANDIDATE_TOPIC_IDS)),
        )
    with pytest.raises(ValueError, match="selection seed"):
        screen_and_select_structural_topics_v3(
            topics,
            query_analyzer=Analyzer(),
            seed="changed",
        )
    with pytest.raises(ValueError, match="frozen ID order"):
        screen_and_select_structural_topics_v3(
            tuple(reversed(topics)),
            query_analyzer=Analyzer(),
        )


def test_selection_digest_uses_exact_utf8_nul_encoding():
    narrative_sha256 = "a" * 64
    expected = hashlib.sha256(
        SELECTION_SEED.encode("utf-8")
        + b"\0"
        + b"14"
        + b"\0"
        + narrative_sha256.encode("ascii")
    ).hexdigest()
    assert selection_digest(
        topic_id="14",
        narrative_sha256=narrative_sha256,
        seed=SELECTION_SEED,
    ) == expected
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        selection_digest(
            topic_id="14",
            narrative_sha256="A" * 64,
            seed=SELECTION_SEED,
        )

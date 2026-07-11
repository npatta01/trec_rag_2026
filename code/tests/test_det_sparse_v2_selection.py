from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pytest

import trec_rag.det_sparse_v2_selection as selection_module
from trec_rag.det_sparse_v2_config import CANDIDATE_TOPIC_IDS, SELECTION_SEED
from trec_rag.det_sparse_v2_selection import screen_and_select_structural_topics
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique
from trec_rag.topics import Topic


FINGERPRINT = AnalyzerFingerprint(
    contract_version="selection-test",
    implementation="selection-test",
    lucene_version="test",
    analyzer_class="test",
    tokenizer="letters",
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
        tokens = tuple(re.findall(r"[^\W\d_]+", text.lower()))
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


@dataclass(frozen=True)
class FakePlan:
    topic_id: str
    unit_count: int
    status: str = "ok"
    failure: object | None = None

    @property
    def lexical_units(self):
        return tuple(range(self.unit_count))

    def to_dict(self):
        return {
            "topic_id": self.topic_id,
            "unit_count": self.unit_count,
            "status": self.status,
            "failure": self.failure,
        }


def _topics():
    return tuple(
        Topic(
            id=topic_id,
            title=f"topic {topic_id}",
            narrative=" ".join(
                f"term{letter}"
                for letter in "abcdefghijklm"[: 5 + index % 8]
            ),
        )
        for index, topic_id in enumerate(CANDIDATE_TOPIC_IDS)
    )


def _digest(topic):
    narrative_sha = hashlib.sha256(topic.narrative.encode("utf-8")).hexdigest()
    return hashlib.sha256(
        SELECTION_SEED.encode("utf-8")
        + b"\0"
        + topic.id.encode("utf-8")
        + b"\0"
        + narrative_sha.encode("ascii")
    ).hexdigest()


def test_qrels_blind_selection_reproduces_one_hash_winner_per_stratum(monkeypatch):
    topics = _topics()
    unit_counts = {
        **{topic_id: 2 for topic_id in CANDIDATE_TOPIC_IDS[:6]},
        **{topic_id: 3 for topic_id in CANDIDATE_TOPIC_IDS[6:9]},
        **{topic_id: 4 for topic_id in CANDIDATE_TOPIC_IDS[9:]},
    }
    monkeypatch.setattr(
        selection_module,
        "build_deterministic_sparse_v2_plan",
        lambda topic_id, narrative, query_analyzer: FakePlan(
            topic_id,
            unit_counts[topic_id],
        ),
    )

    outcome = screen_and_select_structural_topics(
        topics,
        query_analyzer=Analyzer(),
    )

    assert outcome.selection.status == "ok"
    screens = {row.topic_id: row for row in outcome.selection.screens}
    assert all(len(row.plan_semantic_sha256) == 64 for row in screens.values())
    two = sorted(
        topics[:6],
        key=lambda topic: (
            screens[topic.id].original_unique_term_count,
            int(topic.id),
        ),
    )
    split = (len(two) + 1) // 2
    expected = (
        min(two[:split], key=_digest).id,
        min(two[split:], key=_digest).id,
        min(topics[6:9], key=_digest).id,
        min(topics[9:], key=_digest).id,
    )
    assert outcome.selection.selected_topic_ids == expected
    assert [screens[topic_id].stratum for topic_id in expected] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert all(screens[topic_id].selection_digest == _digest(
        next(topic for topic in topics if topic.id == topic_id)
    ) for topic_id in expected)


def test_empty_structural_stratum_stops_without_manual_replacement(monkeypatch):
    topics = _topics()
    monkeypatch.setattr(
        selection_module,
        "build_deterministic_sparse_v2_plan",
        lambda topic_id, narrative, query_analyzer: FakePlan(
            topic_id,
            2 if int(topic_id) % 2 else 3,
        ),
    )

    outcome = screen_and_select_structural_topics(
        topics,
        query_analyzer=Analyzer(),
    )

    assert outcome.selection.status == "failure"
    assert outcome.selection.selected_topic_ids == ()
    assert outcome.selection.failure is not None
    assert outcome.selection.failure.code == "empty_stratum_D"


def test_selection_rejects_candidate_order_or_seed_drift():
    topics = _topics()
    with pytest.raises(ValueError, match="candidate universe"):
        screen_and_select_structural_topics(
            topics,
            query_analyzer=Analyzer(),
            candidate_topic_ids=tuple(reversed(CANDIDATE_TOPIC_IDS)),
        )
    with pytest.raises(ValueError, match="selection seed"):
        screen_and_select_structural_topics(
            topics,
            query_analyzer=Analyzer(),
            seed="changed",
        )

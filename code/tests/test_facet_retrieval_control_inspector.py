import json
from pathlib import Path

import pytest

from trec_rag.facet_retrieval_control_inspector import (
    InspectionStream,
    inspect_stream,
    load_inspection_streams,
)
from trec_rag.facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    build_control_manifest,
)
from trec_rag.pipeline_models import RetrievedCandidate


REPO_ROOT = Path(__file__).resolve().parents[2]
R1_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)


def _candidate(rank: int, text: str, *, topic_id: str = "707") -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id=topic_id,
        variant_name="facet_control_v1:W0:f02",
        retriever_name="synthetic",
        query_text="sorbitol human health adverse effects safety",
        docid=f"doc-{rank:02d}",
        rank=rank,
        score=float(101 - rank),
        text=text,
    )


def _stream(topic_id: str = "707") -> InspectionStream:
    return InspectionStream(
        topic_id=topic_id,
        stream_id="f02",
        anchor_groups=(("sorbitol",),),
        intent_groups=(("health", "risk", "effect", "safeti"),),
        forbidden_drift_groups=(("dog", "canin", "pet"),),
    )


def test_inspector_reports_top_ten_without_changing_top_five_decision():
    candidates = [
        _candidate(rank, "Sorbitol human health effects study")
        for rank in range(1, 6)
    ] + [
        _candidate(rank, "Sorbitol dog health effects study")
        for rank in range(6, 11)
    ]

    result = inspect_stream(_stream(), list(reversed(candidates)))

    assert result.domain_drift_top5_count == 0
    assert result.domain_drift_top10_count == 5
    assert result.anchor_top10_count == 10
    assert result.anchor_intent_cohit_top10_count == 10
    assert result.content_quality_top10_count == 0
    assert result.decision == "keep"
    assert result.top_docids == tuple(f"doc-{rank:02d}" for rank in range(1, 11))


def test_inspector_keeps_top_five_rejection_rules():
    candidates = [
        _candidate(rank, "Sorbitol dog health effects essay homework")
        for rank in range(1, 4)
    ] + [
        _candidate(rank, "Sorbitol human health effects study")
        for rank in range(4, 11)
    ]

    result = inspect_stream(_stream(), candidates)

    assert result.domain_drift_warning is True
    assert result.content_quality_warning is True
    assert result.rejected is True
    assert result.decision == "reject_independent_warnings"


def test_inspection_specs_come_from_tracked_r1_source():
    streams = load_inspection_streams(R1_PATH, build_control_manifest())

    assert tuple(streams) == (
        ("200", "f07a"),
        ("225", "f02"),
        ("225", "f04"),
        ("707", "f02"),
    )
    assert streams[("225", "f04")].anchor_groups == (("aggress",),)
    assert streams[("707", "f02")].forbidden_drift_groups == (
        ("dog", "canin", "pet"),
    )


def test_inspection_source_tampering_is_rejected(tmp_path):
    payload = json.loads(R1_PATH.read_text(encoding="utf-8"))
    payload["streams"][0]["anchor_groups"] = [["tampered"]]
    path = tmp_path / "r1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="R1 source manifest SHA-256"):
        load_inspection_streams(path, build_control_manifest())


@pytest.mark.parametrize("topic_id", PROTECTED_TOPIC_IDS)
def test_inspector_rejects_protected_topics(topic_id):
    with pytest.raises(ValueError, match=f"protected topic {topic_id}"):
        inspect_stream(_stream(topic_id), [])

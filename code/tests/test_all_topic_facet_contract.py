from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.adaptive_evidence_contract import PROTECTED_TOPIC_IDS
from trec_rag.all_topic_facet_contract import (
    ALL_TOPIC_IDS,
    FACET_DEPTH,
    ORIGINAL_DEPTH,
    REQUEST_INTERVAL_SECONDS,
    build_request_plan,
    freeze_planning,
    validate_authorized_scope,
    validate_facet_manifest,
    verify_planning,
)


def _facet(
    *,
    topic_id: str = "14",
    facet_id: str = "14-compensation",
    query: str = "sports athletes compensation societal impact",
    anchor_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    relation_terms: list[str] | None = None,
    bridge_terms: list[dict[str, object]] | None = None,
    order: int = 0,
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "facet_id": facet_id,
        "obligation_id": f"{topic_id}:o1",
        "obligation": "How athlete compensation affects the societal role of sports.",
        "query": query,
        "anchor_terms": ["sports", "athletes"] if anchor_terms is None else anchor_terms,
        "domain_terms": ["societal"] if domain_terms is None else domain_terms,
        "relation_terms": ["compensation", "impact"] if relation_terms is None else relation_terms,
        "analyzer_terms": query.lower().split(),
        "bridge_terms": [] if bridge_terms is None else bridge_terms,
        "manifest_order": order,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    }


def _manifest(facet: dict[str, object]) -> dict[str, object]:
    topic_id = str(facet["topic_id"])
    return {
        "schema_version": "all-topic-tethered-facet-manifest-v1",
        "experiment_id": "all_topic_tethered_facet_validation_v1",
        "topic_ids": [topic_id],
        "analyzer": {"contract": "lowercase-whitespace-v1"},
        "topics": [
            {
                "topic_id": topic_id,
                "narrative": "A narrative.",
                "narrative_sha256": hashlib.sha256(b"A narrative.").hexdigest(),
                "explicit_obligation_count": 1,
                "manifest_order": 0,
            }
        ],
        "facets": [facet],
    }


def test_scope_is_exactly_the_authorized_22() -> None:
    assert validate_authorized_scope(ALL_TOPIC_IDS) == ALL_TOPIC_IDS
    with pytest.raises(ValueError, match="outside all-topic authorization"):
        validate_authorized_scope((*ALL_TOPIC_IDS, "999"))


def test_shared_protected_constants_are_not_weakened() -> None:
    assert {"144", "213", "224", "407", "515"} <= PROTECTED_TOPIC_IDS


def test_facet_requires_tethered_subject_domain_and_relation() -> None:
    facet = _facet(query="impact", anchor_terms=[], domain_terms=[], relation_terms=[])
    with pytest.raises(ValueError, match="tether"):
        validate_facet_manifest(_manifest(facet), require_all_topics=False)


def test_manifest_rejects_duplicate_obligations_order_hash_and_qrels_fields() -> None:
    payload = _manifest(_facet())
    duplicate = copy.deepcopy(payload)
    second = copy.deepcopy(duplicate["facets"][0])
    second["facet_id"] = "14-second"
    second["obligation_id"] = "14:o2"
    second["manifest_order"] = 1
    duplicate["facets"].append(second)
    duplicate["topics"][0]["explicit_obligation_count"] = 2
    with pytest.raises(ValueError, match="obligation"):
        validate_facet_manifest(duplicate, require_all_topics=False)

    changed_hash = copy.deepcopy(payload)
    changed_hash["facets"][0]["query_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="query hash"):
        validate_facet_manifest(changed_hash, require_all_topics=False)

    qrels = copy.deepcopy(payload)
    qrels["qrels_path"] = "forbidden"
    with pytest.raises(ValueError, match="qrels"):
        validate_facet_manifest(qrels, require_all_topics=False)


def test_bridge_terms_require_complete_allowed_provenance() -> None:
    bridge = {
        "surface": "professional",
        "source": "standard domain vocabulary",
        "purpose": "candidate_answer",
        "scope_rationale": "Binds athletes to the professional sports domain.",
        "analyzer_output": ["professional"],
        "not_candidate_answer_rationale": "A domain label, not an effect.",
    }
    with pytest.raises(ValueError, match="bridge purpose"):
        validate_facet_manifest(
            _manifest(_facet(query="sports athletes compensation societal impact professional", bridge_terms=[bridge])),
            require_all_topics=False,
        )


def test_request_plan_reuses_originals_and_only_schedules_facets() -> None:
    payload = _manifest(_facet())
    cache = {
        "14": {
            "path": "/cache/14.json",
            "sha256": "a" * 64,
            "query": "A narrative.",
            "query_sha256": hashlib.sha256(b"A narrative.").hexdigest(),
            "hits": ORIGINAL_DEPTH,
            "candidate_count": ORIGINAL_DEPTH,
        }
    }
    plan = build_request_plan(payload, cache, require_all_topics=False)

    assert plan["original_cache_hit_count"] == 1
    assert plan["original_request_count"] == 0
    assert plan["facet_request_count"] == 1
    assert plan["request_interval_seconds"] == REQUEST_INTERVAL_SECONDS
    assert plan["originals"][0]["depth"] == ORIGINAL_DEPTH
    assert plan["facet_requests"][0]["depth"] == FACET_DEPTH


def test_freeze_seals_authorization_before_source_loader_and_verifies(tmp_path: Path) -> None:
    output = tmp_path / "planning"
    events: list[str] = []

    def source_loader() -> dict[str, object]:
        assert (output / "authorization.json").is_file()
        events.append("source")
        payload = _manifest(_facet())
        return payload

    cache = {
        "14": {
            "path": "/cache/14.json",
            "sha256": "b" * 64,
            "query": "A narrative.",
            "query_sha256": hashlib.sha256(b"A narrative.").hexdigest(),
            "hits": 1000,
            "candidate_count": 1000,
        }
    }
    freeze_planning(
        output,
        topic_ids=("14",),
        source_loader=source_loader,
        original_cache=cache,
        require_all_topics=False,
    )

    assert events == ["source"]
    assert {path.name for path in output.iterdir()} == {
        "authorization.json",
        "manifest.json",
        "request_plan.json",
        "SEALED.json",
    }
    assert verify_planning(output, require_all_topics=False)["verified"] is True

    manifest = json.loads((output / "manifest.json").read_text())
    manifest["facets"][0]["query"] += " changed"
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        verify_planning(output, require_all_topics=False)

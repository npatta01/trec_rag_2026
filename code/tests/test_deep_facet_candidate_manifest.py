from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.deep_facet_candidate_manifest import (
    EXCLUDED_TOPIC_IDS,
    TOPIC_IDS,
    assert_mutation_allowed,
    build_manifest,
    load_manifest,
    main,
    validate_manifest,
)


EXPECTED_QUERIES = {
    "219": (
        "I'm interested in understanding technology's societal impact, exploring "
        "its positive and negative effects on daily life, government, and business "
        "sectors like telehealth. Could you also explain the role of technical "
        "societies and why rationing devices might be needed with technological "
        "advancements?"
    ),
    "72": (
        "I want to understand why deforestation is such a major problem. Specifically, "
        "how does it impact the environment, climate, animals, and humans? Could you "
        "also explain its main causes, effects on rainforests like the Amazon, and what "
        "actions can prevent it?"
    ),
    "300": (
        "I'm interested in learning about effective strategies to prevent and reduce "
        "global warming and climate change, including specific actions that can help "
        "regions like Antarctica. I’d also like to know what global measures can be "
        "taken and how the economic costs of addressing global warming compare to just "
        "dealing with its impacts."
    ),
    "84": (
        "I'm looking for detailed information about vaccines, including their safety, "
        "the causes of public hesitancy—especially around COVID-19—and the different "
        "types of human vaccines and their recommended schedules. I'm also interested "
        "in how vaccination affects global health, historical public health challenges, "
        "and recommendations for animal vaccinations."
    ),
}


def _candidate(rank: int) -> dict[str, object]:
    return {
        "docid": f"doc-{rank}",
        "rank": rank,
        "score": float(1001 - rank),
        "doc": {"text": f"Document text {rank}"},
    }


def _write_cache(cache_root: Path, topic_id: str, query: str, *, hits: int = 1000) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{topic_id}__original__climbmix_bm25__fixture.json"
    payload = {
        "topic_id": topic_id,
        "query": query,
        "variant_name": "original",
        "retriever_name": "climbmix_bm25",
        "hits": hits,
        "response": {
            "api": "fixture",
            "index": "climbmix-400b",
            "query": {"text": query},
            "candidates": [_candidate(rank) for rank in range(1, hits + 1)],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "retrieval"
    for topic_id, query in EXPECTED_QUERIES.items():
        _write_cache(root, topic_id, query)
    return root


def test_exact_boundary_and_original_cache_contract(cache_root: Path) -> None:
    value = build_manifest(cache_root)

    assert value["topic_ids"] == ["219", "72", "300", "84"]
    assert len(value["facets"]) == 25
    assert {
        topic_id: sum(facet["topic_id"] == topic_id for facet in value["facets"])
        for topic_id in TOPIC_IDS
    } == {"219": 7, "72": 7, "300": 4, "84": 7}
    assert all(topic["original_hits"] == 1000 for topic in value["topics"])
    assert all(topic["original_candidate_count"] == 1000 for topic in value["topics"])
    assert all(topic["common_query"] for topic in value["topics"])
    assert set(value["topic_ids"]).isdisjoint(EXCLUDED_TOPIC_IDS)


def test_facets_freeze_gate_terms_and_analyzer_output(cache_root: Path) -> None:
    value = build_manifest(cache_root)

    for order, facet in enumerate(value["facets"]):
        assert facet["manifest_order"] == order
        assert facet["anchor_terms"]
        assert facet["relation_terms"]
        assert isinstance(facet["wrong_domain_patterns"], list)
        assert facet["analyzer_terms"]
        assert facet["query_sha256"] == hashlib.sha256(
            facet["query"].encode("utf-8")
        ).hexdigest()

    biodiversity = next(f for f in value["facets"] if f["facet_id"] == "72-animals")
    assert biodiversity["bridge_terms"] == [
        {
            "term": "biodiversity",
            "purpose": "standard_concept_name",
            "rationale": (
                "Standard concept name for the explicitly requested animal and "
                "environmental effects; it is not a candidate answer."
            ),
        }
    ]


def test_build_rejects_short_duplicate_or_textless_original_cache(cache_root: Path) -> None:
    _write_cache(cache_root, "219", EXPECTED_QUERIES["219"], hits=999)
    with pytest.raises(ValueError, match="hits=1000"):
        build_manifest(cache_root)

    _write_cache(cache_root, "219", EXPECTED_QUERIES["219"])
    path = next(cache_root.glob("219__original__*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response"]["candidates"][1]["docid"] = "doc-1"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="1,000 unique"):
        build_manifest(cache_root)

    _write_cache(cache_root, "219", EXPECTED_QUERIES["219"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response"]["candidates"][0]["doc"] = {"text": ""}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="text-bearing"):
        build_manifest(cache_root)


def test_validation_rejects_topic_or_hash_drift(cache_root: Path) -> None:
    value = build_manifest(cache_root)

    protected = copy.deepcopy(value)
    protected["topic_ids"][0] = "144"
    with pytest.raises(ValueError, match="excluded topic"):
        validate_manifest(protected, cache_root=cache_root)

    query_drift = copy.deepcopy(value)
    query_drift["facets"][0]["query"] += " answer"
    with pytest.raises(ValueError, match="frozen facets"):
        validate_manifest(query_drift, cache_root=cache_root)


def test_sentinel_refuses_mutation(tmp_path: Path) -> None:
    assert_mutation_allowed(tmp_path)
    (tmp_path / "QRELS_ACCESSED").write_text("sealed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="qrels already accessed"):
        assert_mutation_allowed(tmp_path)


def test_create_and_load_round_trip(cache_root: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "pilot" / "manifest.json"
    assert main(["create", "--cache-root", str(cache_root), "--output", str(output)]) == 0
    assert output.read_bytes().endswith(b"\n")
    loaded = load_manifest(output, cache_root=cache_root)
    assert loaded == build_manifest(cache_root)
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "facet_count": 25,
        "original_rows": 4000,
        "qrels_opened": False,
        "topic_count": 4,
    }

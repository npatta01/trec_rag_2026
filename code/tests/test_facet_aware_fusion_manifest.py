from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.facet_aware_fusion_manifest import (
    ALLOWED_BRIDGE_PURPOSES,
    ANALYZER_FINGERPRINT_SHA256,
    ELIGIBLE_TOPIC_IDS,
    EXPERIMENT_ID,
    PRIOR_PILOT_TOPIC_IDS,
    PROTECTED_TOPIC_IDS,
    TOPIC_IDS,
    build_manifest,
    load_manifest,
    main,
    validate_manifest,
)


EXPECTED_QUERIES = {
    "233": (
        "I'm interested in learning how social media affects mental health, "
        "particularly among teenagers. I want to know about both its positive "
        "and negative impacts, and understand why it might contribute to depression."
    ),
    "273": (
        "I want to understand why Africa, despite its rich resources, is often seen "
        "as underdeveloped or poor. I'm particularly interested in how factors like "
        "resource distribution, historical events, and economic changes in specific "
        "countries contribute to this perception. Additionally, I'd like basic facts "
        "such as Morocco's location, Cameroon's most important resource, and how many "
        "continents could fit inside Africa."
    ),
    "161": (
        "I want to understand the main arguments surrounding abortion and why people "
        "hold such different views on it. I'm also curious how laws, beliefs like the "
        "Rapture or political ideologies, and historical changes have shaped abortion "
        "rights. Finally, I'd like to learn about current options, such as the abortion "
        "pill, and what women's rights groups are prioritizing today."
    ),
    "14": (
        "I'm interested in sports' societal impact, particularly concerning athlete "
        "compensation, inclusion, cultural influence, and the business side. I also "
        "want to understand how evolving equipment, training, and mindset shape both "
        "athletes and the popularity of different sports."
    ),
}

EXPECTED_FACETS = [
    ("233-positive-impact", "teenager social media positive mental health impacts"),
    ("233-negative-impact", "teenager social media negative mental health impacts"),
    ("233-depression-contribution", "why social media contributes to teenager depression"),
    ("273-poverty-perception", "why resource-rich Africa is seen as underdeveloped or poor"),
    ("273-resource-distribution", "Africa resource distribution contribution to underdevelopment perception"),
    ("273-historical-events", "Africa historical events contribution to underdevelopment perception"),
    ("273-economic-changes", "Africa economic changes in specific countries contribution to underdevelopment perception"),
    ("273-morocco-location", "Morocco location in Africa"),
    ("273-cameroon-resource", "Cameroon most important resource"),
    ("273-continent-capacity", "how many continents could fit inside Africa"),
    ("161-arguments-and-views", "abortion main arguments why people hold different views"),
    ("161-laws-and-rights", "how laws shaped abortion rights"),
    ("161-rapture-beliefs", "how beliefs like the Rapture shaped abortion rights"),
    ("161-political-ideologies", "how political ideologies shaped abortion rights"),
    ("161-historical-changes", "how historical changes shaped abortion rights"),
    ("161-current-options", "current abortion options abortion pill"),
    ("161-rights-group-priorities", "women's rights groups priorities today abortion"),
    ("14-athlete-compensation", "sports societal impact athlete compensation"),
    ("14-inclusion", "sports societal impact inclusion"),
    ("14-cultural-influence", "sports societal impact cultural influence"),
    ("14-business", "sports societal impact business side"),
    ("14-equipment", "evolving sports equipment shapes athletes and sport popularity"),
    ("14-training", "evolving sports training shapes athletes and sport popularity"),
    ("14-mindset", "evolving sports mindset shapes athletes and sport popularity"),
]


def _write_cache(cache_root: Path, topic_id: str, query: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{topic_id}__original__climbmix_bm25__fixture.json"
    path.write_text(
        json.dumps(
            {
                "topic_id": topic_id,
                "query": query,
                "variant_name": "original",
                "retriever_name": "climbmix_bm25",
                "hits": 100,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _canonical_sha256(value: object) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refresh_content_hashes(payload: dict[str, object]) -> None:
    payload["hashes"]["topics_sha256"] = _canonical_sha256(payload["topics"])
    unhashed = {key: value for key, value in payload.items() if key != "hashes"}
    payload["hashes"]["freeze_sha256"] = _canonical_sha256(unhashed)


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "retrieval"
    for topic_id, query in EXPECTED_QUERIES.items():
        _write_cache(root, topic_id, query)
    return root


def test_manifest_has_exact_frozen_boundary_and_24_facets(cache_root: Path) -> None:
    payload = build_manifest(cache_root)

    assert payload["topic_ids"] == ["233", "273", "161", "14"]
    assert len(payload["facets"]) == 24
    assert {row["topic_id"] for row in payload["facets"]}.isdisjoint(PROTECTED_TOPIC_IDS)
    assert {row["topic_id"] for row in payload["facets"]}.isdisjoint(PRIOR_PILOT_TOPIC_IDS)
    assert [row["manifest_order"] for row in payload["facets"]] == list(range(24))
    assert [(row["facet_id"], row["query"]) for row in payload["facets"]] == EXPECTED_FACETS
    assert {
        topic_id: sum(row["topic_id"] == topic_id for row in payload["facets"])
        for topic_id in TOPIC_IDS
    } == {"233": 3, "273": 7, "161": 7, "14": 7}


def test_every_facet_is_tethered_and_bridge_terms_are_audited(cache_root: Path) -> None:
    payload = build_manifest(cache_root)
    expected_keys = {
        "topic_id",
        "facet_id",
        "query",
        "obligation",
        "anchor_terms",
        "relation_terms",
        "wrong_domain_patterns",
        "bridge_terms",
        "manifest_order",
    }

    for facet in payload["facets"]:
        assert set(facet) == expected_keys
        assert facet["anchor_terms"] and facet["relation_terms"]
        for bridge in facet["bridge_terms"]:
            assert set(bridge) == {"term", "purpose", "rationale"}
            assert bridge["purpose"] in ALLOWED_BRIDGE_PURPOSES
            assert bridge["rationale"]


def test_selection_hashes_recompute_to_exact_eligible_order(cache_root: Path) -> None:
    payload = build_manifest(cache_root)
    expected = sorted(
        ELIGIBLE_TOPIC_IDS,
        key=lambda topic_id: hashlib.sha256(
            f"{EXPERIMENT_ID}{topic_id}".encode("utf-8")
        ).hexdigest(),
    )

    assert [row["topic_id"] for row in payload["topic_selection"]] == expected
    assert payload["topic_ids"] == expected[:4]
    assert payload["hashes"]["analyzer_fingerprint_sha256"] == ANALYZER_FINGERPRINT_SHA256
    for order, row in enumerate(payload["topic_selection"]):
        assert row == {
            "topic_id": expected[order],
            "selection_sha256": hashlib.sha256(
                f"{EXPERIMENT_ID}{expected[order]}".encode("utf-8")
            ).hexdigest(),
            "selection_order": order,
            "selected": order < 4,
        }


def test_source_topics_bind_exact_queries_and_existing_cache_files(cache_root: Path) -> None:
    payload = build_manifest(cache_root)

    assert [row["topic_id"] for row in payload["topics"]] == list(TOPIC_IDS)
    assert [row["query"] for row in payload["topics"]] == [
        EXPECTED_QUERIES[topic_id] for topic_id in TOPIC_IDS
    ]
    for order, row in enumerate(payload["topics"]):
        source = cache_root / row["original_cache_filename"]
        assert source.is_file()
        assert row["manifest_order"] == order
        assert row["original_cache_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_build_rejects_missing_duplicate_or_mismatched_original_cache(tmp_path: Path) -> None:
    root = tmp_path / "retrieval"
    for topic_id, query in EXPECTED_QUERIES.items():
        if topic_id != "233":
            _write_cache(root, topic_id, query)

    with pytest.raises(ValueError, match="exactly one original cache file.*233"):
        build_manifest(root)

    first = _write_cache(root, "233", EXPECTED_QUERIES["233"])
    duplicate = root / "233__original__climbmix_bm25__duplicate.json"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="exactly one original cache file.*233"):
        build_manifest(root)

    duplicate.unlink()
    _write_cache(root, "233", "wrong source query")
    with pytest.raises(ValueError, match="query mismatch.*233"):
        build_manifest(root)


def test_validate_rejects_unsupported_keys_reordering_and_topic_firewall(cache_root: Path) -> None:
    payload = build_manifest(cache_root)

    unsupported = copy.deepcopy(payload)
    unsupported["qrels_path"] = "forbidden"
    with pytest.raises(ValueError, match="unsupported manifest keys"):
        validate_manifest(unsupported, cache_root=cache_root)

    unsupported_facet = copy.deepcopy(payload)
    unsupported_facet["facets"][0]["answer_hint"] = "forbidden"
    with pytest.raises(ValueError, match="unsupported facet keys"):
        validate_manifest(unsupported_facet, cache_root=cache_root)

    reordered = copy.deepcopy(payload)
    reordered["facets"][0], reordered["facets"][1] = (
        reordered["facets"][1],
        reordered["facets"][0],
    )
    with pytest.raises(ValueError, match="facet records.*order"):
        validate_manifest(reordered, cache_root=cache_root)

    protected = copy.deepcopy(payload)
    protected["topic_ids"][0] = "144"
    with pytest.raises(ValueError, match="protected topic"):
        validate_manifest(protected, cache_root=cache_root)

    prior = copy.deepcopy(payload)
    prior["topic_ids"][0] = "200"
    with pytest.raises(ValueError, match="prior-pilot topic"):
        validate_manifest(prior, cache_root=cache_root)

    protected_facet = copy.deepcopy(payload)
    protected_facet["facets"][0]["topic_id"] = "144"
    with pytest.raises(ValueError, match="protected topic"):
        validate_manifest(protected_facet, cache_root=cache_root)

    prior_facet = copy.deepcopy(payload)
    prior_facet["facets"][0]["topic_id"] = "200"
    with pytest.raises(ValueError, match="prior-pilot topic"):
        validate_manifest(prior_facet, cache_root=cache_root)


def test_validate_rejects_changed_queries_and_hashes(cache_root: Path) -> None:
    payload = build_manifest(cache_root)

    changed_query = copy.deepcopy(payload)
    changed_query["topics"][0]["query"] += " changed"
    with pytest.raises(ValueError, match="exact source query"):
        validate_manifest(changed_query, cache_root=cache_root)

    changed_hash = copy.deepcopy(payload)
    changed_hash["hashes"]["facets_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hashes"):
        validate_manifest(changed_hash, cache_root=cache_root)

    qrels_opened = copy.deepcopy(payload)
    qrels_opened["qrels_opened"] = True
    with pytest.raises(ValueError, match="qrels_opened"):
        validate_manifest(qrels_opened, cache_root=cache_root)


def test_validate_rejects_rebound_nonexistent_original_cache(cache_root: Path) -> None:
    payload = build_manifest(cache_root)
    payload["topics"][0]["original_cache_filename"] = (
        "233__original__climbmix_bm25__rebound.json"
    )
    payload["topics"][0]["original_cache_sha256"] = "a" * 64
    _refresh_content_hashes(payload)

    with pytest.raises(ValueError, match="original cache file.*topic 233.*does not exist"):
        validate_manifest(payload, cache_root=cache_root)


def test_load_requires_canonical_json_and_cli_is_create_only(
    cache_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "manifest.json"

    assert main(["create", "--cache-root", str(cache_root), "--output", str(output)]) == 0
    loaded = load_manifest(output, cache_root=cache_root)
    assert loaded["topic_ids"] == list(TOPIC_IDS)
    assert output.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(FileExistsError):
        main(["create", "--cache-root", str(cache_root), "--output", str(output)])

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_manifest(noncanonical, cache_root=cache_root)

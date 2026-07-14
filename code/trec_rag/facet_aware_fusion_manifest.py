"""Frozen held-out manifest for the facet-aware fusion pilot.

This module only plans the experiment.  It reads the four selected original
retrieval-cache records to bind their exact query text; it never reads qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "rag25_facet_aware_fusion_manifest_v1"
EXPERIMENT_ID = "rag25_facet_aware_fusion_v1"
ELIGIBLE_TOPIC_IDS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "161",
    "219",
    "233",
    "273",
    "300",
    "477",
    "499",
)
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PRIOR_PILOT_TOPIC_IDS = frozenset({"200", "225", "707", "897"})
TOPIC_IDS = ("233", "273", "161", "14")
ALLOWED_BRIDGE_PURPOSES = frozenset(
    {
        "domain_disambiguation",
        "population_binding",
        "relation_paraphrasing",
        "standard_concept_name",
    }
)
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)

SOURCE_QUERIES = {
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

_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "selection_salt",
    "eligible_topic_ids",
    "protected_topic_ids",
    "prior_pilot_topic_ids",
    "topic_ids",
    "topic_selection",
    "topics",
    "facets",
    "hashes",
    "qrels_opened",
}
_TOPIC_KEYS = {
    "topic_id",
    "query",
    "selection_sha256",
    "manifest_order",
    "original_cache_filename",
    "original_cache_sha256",
}
_FACET_KEYS = {
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
_BRIDGE_KEYS = {"term", "purpose", "rationale"}
_HASH_KEYS = {
    "analyzer_fingerprint_sha256",
    "topic_selection_sha256",
    "topics_sha256",
    "facets_sha256",
    "freeze_sha256",
}


def _facet(
    topic_id: str,
    facet_id: str,
    query: str,
    obligation: str,
    anchor_terms: list[str],
    relation_terms: list[str],
    wrong_domain_patterns: list[str],
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "facet_id": facet_id,
        "query": query,
        "obligation": obligation,
        "anchor_terms": anchor_terms,
        "relation_terms": relation_terms,
        "wrong_domain_patterns": wrong_domain_patterns,
        "bridge_terms": [],
    }


def _facet_records() -> list[dict[str, object]]:
    records = [
        _facet("233", "233-positive-impact", "teenager social media positive mental health impacts", "Positive impacts of social media on teenagers' mental health.", ["social media", "teenager"], ["positive", "mental health"], ["social media marketing", "brand engagement"]),
        _facet("233", "233-negative-impact", "teenager social media negative mental health impacts", "Negative impacts of social media on teenagers' mental health.", ["social media", "teenager"], ["negative", "mental health"], ["social media marketing", "brand engagement"]),
        _facet("233", "233-depression-contribution", "why social media contributes to teenager depression", "Why social media might contribute to depression among teenagers.", ["social media", "teenager"], ["contribute", "depression"], ["social media marketing", "brand engagement"]),
        _facet("273", "273-poverty-perception", "why resource-rich Africa is seen as underdeveloped or poor", "Why resource-rich Africa is often perceived as underdeveloped or poor.", ["Africa", "resources"], ["underdeveloped", "poor"], ["African violet", "Africa song"]),
        _facet("273", "273-resource-distribution", "Africa resource distribution contribution to underdevelopment perception", "How resource distribution contributes to perceptions of African underdevelopment.", ["Africa", "resource distribution"], ["contribution", "underdevelopment"], ["African violet", "Africa song"]),
        _facet("273", "273-historical-events", "Africa historical events contribution to underdevelopment perception", "How historical events contribute to perceptions of African underdevelopment.", ["Africa", "historical events"], ["contribution", "underdevelopment"], ["African violet", "Africa song"]),
        _facet("273", "273-economic-changes", "Africa economic changes in specific countries contribution to underdevelopment perception", "How economic changes in specific African countries contribute to the perception.", ["Africa", "economic changes"], ["specific countries", "perception"], ["African violet", "Africa song"]),
        _facet("273", "273-morocco-location", "Morocco location in Africa", "Morocco's location.", ["Morocco"], ["location", "Africa"], ["Morocco Indiana", "Morocco football"]),
        _facet("273", "273-cameroon-resource", "Cameroon most important resource", "Cameroon's most important resource.", ["Cameroon"], ["most important", "resource"], ["Cameroon football", "Cameroon sheep"]),
        _facet("273", "273-continent-capacity", "how many continents could fit inside Africa", "How many continents could fit inside Africa.", ["Africa", "continents"], ["how many", "fit inside"], ["Africa song", "African violet"]),
        _facet("161", "161-arguments-and-views", "abortion main arguments why people hold different views", "The main abortion arguments and why people hold different views.", ["abortion"], ["arguments", "different views"], ["abort operation", "process abortion"]),
        _facet("161", "161-laws-and-rights", "how laws shaped abortion rights", "How laws have shaped abortion rights.", ["abortion rights"], ["laws", "shaped"], ["abort operation", "process abortion"]),
        _facet("161", "161-rapture-beliefs", "how beliefs like the Rapture shaped abortion rights", "How beliefs such as the Rapture have shaped abortion rights.", ["abortion rights", "Rapture"], ["beliefs", "shaped"], ["Rapture video game", "Rapture song"]),
        _facet("161", "161-political-ideologies", "how political ideologies shaped abortion rights", "How political ideologies have shaped abortion rights.", ["abortion rights"], ["political ideologies", "shaped"], ["abort operation", "process abortion"]),
        _facet("161", "161-historical-changes", "how historical changes shaped abortion rights", "How historical changes have shaped abortion rights.", ["abortion rights"], ["historical changes", "shaped"], ["abort operation", "process abortion"]),
        _facet("161", "161-current-options", "current abortion options abortion pill", "Current abortion options, including the abortion pill.", ["abortion"], ["current options", "abortion pill"], ["abort operation", "process abortion"]),
        _facet("161", "161-rights-group-priorities", "women's rights groups priorities today abortion", "What women's rights groups prioritize today regarding abortion.", ["abortion", "women's rights groups"], ["priorities", "today"], ["abort operation", "process abortion"]),
        _facet("14", "14-athlete-compensation", "sports societal impact athlete compensation", "The societal impact of athlete compensation.", ["sports", "athlete"], ["compensation", "societal impact"], ["sports betting odds", "fantasy sports"]),
        _facet("14", "14-inclusion", "sports societal impact inclusion", "The societal impact of inclusion in sports.", ["sports"], ["inclusion", "societal impact"], ["sports betting odds", "fantasy sports"]),
        _facet("14", "14-cultural-influence", "sports societal impact cultural influence", "Sports' societal and cultural influence.", ["sports"], ["cultural influence", "societal impact"], ["sports betting odds", "fantasy sports"]),
        _facet("14", "14-business", "sports societal impact business side", "The societal impact of the business side of sports.", ["sports"], ["business side", "societal impact"], ["sports betting odds", "fantasy sports"]),
        _facet("14", "14-equipment", "evolving sports equipment shapes athletes and sport popularity", "How evolving equipment shapes athletes and sport popularity.", ["sports", "equipment"], ["shapes", "athletes", "popularity"], ["sports betting odds", "video game equipment"]),
        _facet("14", "14-training", "evolving sports training shapes athletes and sport popularity", "How evolving training shapes athletes and sport popularity.", ["sports", "training"], ["shapes", "athletes", "popularity"], ["sports betting odds", "sports game tutorial"]),
        _facet("14", "14-mindset", "evolving sports mindset shapes athletes and sport popularity", "How evolving mindset shapes athletes and sport popularity.", ["sports", "mindset"], ["shapes", "athletes", "popularity"], ["sports betting odds", "fantasy sports"]),
    ]
    for order, record in enumerate(records):
        record["manifest_order"] = order
    return records


def _selection_hash(topic_id: str) -> str:
    return hashlib.sha256(f"{EXPERIMENT_ID}{topic_id}".encode("utf-8")).hexdigest()


def _topic_selection() -> list[dict[str, object]]:
    ordered = sorted(ELIGIBLE_TOPIC_IDS, key=_selection_hash)
    return [
        {
            "topic_id": topic_id,
            "selection_sha256": _selection_hash(topic_id),
            "selection_order": order,
            "selected": order < len(TOPIC_IDS),
        }
        for order, topic_id in enumerate(ordered)
    ]


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _expected_hashes(payload: Mapping[str, object]) -> dict[str, str]:
    unhashed = {key: value for key, value in payload.items() if key != "hashes"}
    return {
        "analyzer_fingerprint_sha256": ANALYZER_FINGERPRINT_SHA256,
        "topic_selection_sha256": _sha256(payload["topic_selection"]),
        "topics_sha256": _sha256(payload["topics"]),
        "facets_sha256": _sha256(payload["facets"]),
        "freeze_sha256": _sha256(unhashed),
    }


def _read_original_cache(cache_root: Path, topic_id: str) -> dict[str, object]:
    matches = sorted(cache_root.glob(f"{topic_id}__original__*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one original cache file for topic {topic_id}; "
            f"found {len(matches)}"
        )
    path = matches[0]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid original cache file for topic {topic_id}: {path}") from exc
    if not isinstance(record, dict) or str(record.get("topic_id")) != topic_id:
        raise ValueError(f"topic mismatch in original cache file for topic {topic_id}")
    if record.get("query") != SOURCE_QUERIES[topic_id]:
        raise ValueError(f"query mismatch in original cache file for topic {topic_id}")
    return {
        "topic_id": topic_id,
        "query": SOURCE_QUERIES[topic_id],
        "selection_sha256": _selection_hash(topic_id),
        "manifest_order": TOPIC_IDS.index(topic_id),
        "original_cache_filename": path.name,
        "original_cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_manifest(cache_root: Path) -> dict[str, object]:
    """Build and validate the frozen manifest from selected original caches."""

    cache_root = Path(cache_root)
    topics = [_read_original_cache(cache_root, topic_id) for topic_id in TOPIC_IDS]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_salt": EXPERIMENT_ID,
        "eligible_topic_ids": list(ELIGIBLE_TOPIC_IDS),
        "protected_topic_ids": sorted(PROTECTED_TOPIC_IDS, key=int),
        "prior_pilot_topic_ids": sorted(PRIOR_PILOT_TOPIC_IDS, key=int),
        "topic_ids": list(TOPIC_IDS),
        "topic_selection": _topic_selection(),
        "topics": topics,
        "facets": _facet_records(),
        "qrels_opened": False,
    }
    payload["hashes"] = _expected_hashes(payload)
    validate_manifest(payload)
    return payload


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def validate_manifest(payload: Mapping[str, object]) -> None:
    """Reject any manifest that differs from the frozen, audited contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    keys = set(payload)
    if keys != _TOP_LEVEL_KEYS:
        raise ValueError(f"unsupported manifest keys: {sorted(keys ^ _TOP_LEVEL_KEYS)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected manifest schema_version")
    if payload["experiment_id"] != EXPERIMENT_ID or payload["selection_salt"] != EXPERIMENT_ID:
        raise ValueError("unexpected experiment identity or selection salt")
    if payload["qrels_opened"] is not False:
        raise ValueError("qrels_opened must remain false during manifest creation")

    topic_ids = [str(value) for value in _require_sequence(payload["topic_ids"], "topic_ids")]
    if set(topic_ids) & PROTECTED_TOPIC_IDS:
        raise ValueError("manifest contains a protected topic")
    if set(topic_ids) & PRIOR_PILOT_TOPIC_IDS:
        raise ValueError("manifest contains a prior-pilot topic")
    if topic_ids != list(TOPIC_IDS):
        raise ValueError("topic_ids do not match the frozen hash-selected order")
    if payload["eligible_topic_ids"] != list(ELIGIBLE_TOPIC_IDS):
        raise ValueError("eligible_topic_ids do not match the frozen boundary")
    if payload["protected_topic_ids"] != sorted(PROTECTED_TOPIC_IDS, key=int):
        raise ValueError("protected_topic_ids do not match the firewall")
    if payload["prior_pilot_topic_ids"] != sorted(PRIOR_PILOT_TOPIC_IDS, key=int):
        raise ValueError("prior_pilot_topic_ids do not match the firewall")

    selection = _require_sequence(payload["topic_selection"], "topic_selection")
    if selection != _topic_selection():
        raise ValueError("topic_selection does not match recomputed SHA-256 order")

    topics = _require_sequence(payload["topics"], "topics")
    if len(topics) != len(TOPIC_IDS):
        raise ValueError("topics must contain exactly four records")
    for order, row in enumerate(topics):
        if not isinstance(row, Mapping) or set(row) != _TOPIC_KEYS:
            raise ValueError("unsupported topic keys")
        topic_id = str(row["topic_id"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError("topic records contain a protected topic")
        if topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError("topic records contain a prior-pilot topic")
        if topic_id != TOPIC_IDS[order] or row["manifest_order"] != order:
            raise ValueError("topic records are missing or reordered")
        if row["query"] != SOURCE_QUERIES[topic_id]:
            raise ValueError(f"topic {topic_id} does not contain the exact source query")
        if row["selection_sha256"] != _selection_hash(topic_id):
            raise ValueError(f"topic {topic_id} has an invalid selection hash")
        filename = row["original_cache_filename"]
        digest = row["original_cache_sha256"]
        if not isinstance(filename, str) or not filename.startswith(f"{topic_id}__original__"):
            raise ValueError(f"topic {topic_id} has an invalid original cache filename")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"topic {topic_id} has an invalid original cache hash")

    facets = _require_sequence(payload["facets"], "facets")
    for row in facets:
        if not isinstance(row, Mapping) or set(row) != _FACET_KEYS:
            raise ValueError("unsupported facet keys")
        for bridge in _require_sequence(row["bridge_terms"], "bridge_terms"):
            if not isinstance(bridge, Mapping) or set(bridge) != _BRIDGE_KEYS:
                raise ValueError("unsupported bridge-term keys")
            if bridge["purpose"] not in ALLOWED_BRIDGE_PURPOSES or not bridge["rationale"]:
                raise ValueError("bridge terms require an allowed purpose and rationale")
    facet_topics = {str(row["topic_id"]) for row in facets}
    if facet_topics & PROTECTED_TOPIC_IDS:
        raise ValueError("facet records contain a protected topic")
    if facet_topics & PRIOR_PILOT_TOPIC_IDS:
        raise ValueError("facet records contain a prior-pilot topic")
    if facets != _facet_records():
        raise ValueError("facet records differ from the frozen content or order")

    hashes = payload["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != _HASH_KEYS:
        raise ValueError("hashes contain unsupported or missing keys")
    if dict(hashes) != _expected_hashes(payload):
        raise ValueError("manifest hashes do not match canonical content")


def load_manifest(path: Path) -> dict[str, object]:
    """Load a manifest only when its bytes and content are canonical."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if raw != _canonical_bytes(payload, pretty=True):
        raise ValueError("manifest is not canonical JSON")
    validate_manifest(payload)
    return payload


def _create(cache_root: Path, output: Path) -> dict[str, object]:
    payload = build_manifest(cache_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_canonical_bytes(payload, pretty=True))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create the frozen manifest once")
    create.add_argument("--cache-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "create":  # pragma: no cover - argparse enforces this.
        parser.error("unsupported command")
    payload = _create(args.cache_root, args.output)
    print(
        json.dumps(
            {
                "topic_count": len(payload["topic_ids"]),
                "facet_count": len(payload["facets"]),
                "qrels_opened": payload["qrels_opened"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

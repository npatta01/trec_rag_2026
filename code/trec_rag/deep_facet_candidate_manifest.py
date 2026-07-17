"""Frozen query and source boundary for the deep-facet candidate pilot.

This module is deliberately qrels-blind.  It authenticates the four existing
original-query caches and freezes the exact 25 facet requests used downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "rag25_deep_facet_candidate_manifest_v1"
EXPERIMENT_ID = "rag25_deep_facet_candidates_v1"
TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
QRELS_EXPOSED_TOPIC_IDS = frozenset(
    {"200", "225", "707", "897", "233", "273", "161", "14"}
)
EXCLUDED_TOPIC_IDS = PROTECTED_TOPIC_IDS | QRELS_EXPOSED_TOPIC_IDS
ELIGIBLE_TOPIC_IDS = ("31", "37", "58", "72", "84", "219", "300", "477", "499")
ORIGINAL_HITS = 1000
FACET_HITS = 200
ANALYZER_VERSION = "deep_facet_ascii_stem_stop_v1"

SOURCE_QUERIES = {
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

COMMON_QUERIES = {
    "219": (
        "technology societal impacts daily life government business telehealth "
        "technical societies device rationing"
    ),
    "72": (
        "deforestation causes effects environment climate animals humans Amazon "
        "rainforest prevention"
    ),
    "300": (
        "global warming climate change prevention mitigation Antarctica global "
        "measures economic costs impacts"
    ),
    "84": (
        "vaccine safety hesitancy COVID human vaccine types schedules global health "
        "history animal vaccination"
    ),
}

_STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has "
    "have he her hers him his how i if in into is it its like may me might more most "
    "my of on or our ours she should so than that the their theirs them then there "
    "these they this those to too us want was we were what when where which who why "
    "will with would you your".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Small frozen English stemmer used only for audit/Jaccard token sets."""

    if len(token) <= 3 or token.isdigit():
        return token
    replacements = (
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("biliti", "ble"),
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ness", ""),
        ("ement", ""),
        ("ments", ""),
        ("ment", ""),
        ("ingly", ""),
        ("edly", ""),
        ("ing", ""),
        ("ies", "i"),
        ("ied", "i"),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    )
    for suffix, replacement in replacements:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def analyze_terms(text: str) -> list[str]:
    """Return stable unique lowercased, stopped, stemmed analyzer terms."""

    seen: set[str] = set()
    result: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        token = _stem(raw)
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _facet(
    topic_id: str,
    slug: str,
    query: str,
    obligation: str,
    anchors: Sequence[str],
    relations: Sequence[str],
    wrong_domains: Sequence[str] = (),
    *,
    bridge_terms: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "facet_id": f"{topic_id}-{slug}",
        "query": query,
        "obligation": obligation,
        "anchor_terms": list(anchors),
        "relation_terms": list(relations),
        "wrong_domain_patterns": list(wrong_domains),
        "bridge_terms": [dict(item) for item in bridge_terms],
        "analyzer_terms": analyze_terms(query),
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
    }


def _facet_records() -> list[dict[str, object]]:
    biodiversity = {
        "term": "biodiversity",
        "purpose": "standard_concept_name",
        "rationale": (
            "Standard concept name for the explicitly requested animal and "
            "environmental effects; it is not a candidate answer."
        ),
    }
    rows = [
        _facet("219", "positive", "technology positive effects on society and daily life", "Positive effects of technology on society and daily life.", ["technology"], ["positive", "society", "daily life"], ["technology stock", "product review"]),
        _facet("219", "negative", "technology negative effects on society and daily life", "Negative effects of technology on society and daily life.", ["technology"], ["negative", "society", "daily life"], ["technology stock", "product review"]),
        _facet("219", "government", "technology societal impact on government", "Technology's societal impact on government.", ["technology", "government"], ["impact", "society"], ["company governance", "government job"]),
        _facet("219", "business", "technology societal impact on business", "Technology's societal impact on business.", ["technology", "business"], ["impact", "society"], ["business technology product", "stock price"]),
        _facet("219", "telehealth", "technology impact on telehealth and health care delivery", "Technology's impact on telehealth and health-care delivery.", ["technology", "telehealth"], ["impact", "health care", "delivery"], ["delivery driver", "food delivery"]),
        _facet("219", "societies", "role of technical societies in technology", "Role of technical societies in technology.", ["technical societies", "technology"], ["role"], ["secret society", "social club"]),
        _facet("219", "rationing", "why rationing devices may be needed with technological advancements", "Why technological advances may require device rationing.", ["device", "technology"], ["rationing", "needed", "advancement"], ["food rationing", "ration recipe"]),
        _facet("72", "environment", "deforestation environmental impacts", "Environmental impacts of deforestation.", ["deforestation"], ["environment", "impact"], ["forest product", "tree cutting tool"]),
        _facet("72", "climate", "deforestation climate impacts", "Climate impacts of deforestation.", ["deforestation"], ["climate", "impact"], ["weather forecast", "forest product"]),
        _facet("72", "animals", "deforestation impacts on animals and biodiversity", "Effects of deforestation on animals.", ["deforestation"], ["animal", "biodiversity", "impact"], ["animal game", "pet care"], bridge_terms=[biodiversity]),
        _facet("72", "humans", "deforestation impacts on humans", "Effects of deforestation on people.", ["deforestation"], ["human", "people", "impact"], ["forest product", "human anatomy"]),
        _facet("72", "causes", "main causes of deforestation", "Main causes of deforestation.", ["deforestation"], ["cause"], ["forest product", "tree disease"]),
        _facet("72", "amazon", "deforestation effects on the Amazon rainforest", "Effects of deforestation on the Amazon rainforest.", ["deforestation", "Amazon rainforest"], ["effect"], ["Amazon company", "Amazon product"]),
        _facet("72", "prevention", "actions to prevent deforestation", "Actions that prevent deforestation.", ["deforestation"], ["prevent", "action"], ["forest product", "tree disease prevention"]),
        _facet("300", "strategies", "effective specific strategies to prevent and reduce global warming and climate change", "Specific strategies to prevent or reduce global warming and climate change.", ["global warming", "climate change"], ["prevent", "reduce", "strategy"], ["warming recipe", "climate control product"]),
        _facet("300", "antarctica", "climate change actions for Antarctica", "Climate-change actions relevant to Antarctica.", ["climate change", "Antarctica"], ["action"], ["Antarctica tourism", "weather forecast"]),
        _facet("300", "global", "international and government measures to prevent and reduce climate change", "International and government measures against climate change.", ["climate change"], ["international", "government", "measure", "prevent", "reduce"], ["climate control product", "weather forecast"]),
        _facet("300", "economics", "economic cost of addressing global warming compared with its impacts", "Cost of addressing global warming versus bearing its impacts.", ["global warming"], ["economic cost", "addressing", "impact", "compared"], ["warming product", "heating cost"]),
        _facet("84", "safety", "human vaccine safety", "Safety of human vaccines.", ["vaccine", "human"], ["safety"], ["animal vaccine", "computer vaccine"]),
        _facet("84", "hesitancy", "causes of public vaccine hesitancy especially COVID-19", "Causes of public vaccine hesitancy, especially for COVID-19.", ["vaccine", "COVID-19"], ["hesitancy", "cause", "public"], ["animal vaccine", "computer vaccine"]),
        _facet("84", "types", "types of human vaccines", "Types of vaccines used in humans.", ["vaccine", "human"], ["type"], ["animal vaccine", "computer vaccine"]),
        _facet("84", "schedules", "recommended human vaccination schedules", "Recommended vaccination schedules for humans.", ["vaccination", "human"], ["recommended", "schedule"], ["animal vaccination", "computer schedule"]),
        _facet("84", "global-health", "vaccination impact on global health", "Impact of vaccination on global health.", ["vaccination", "global health"], ["impact"], ["animal vaccination", "travel schedule"]),
        _facet("84", "history", "historical public health challenges involving vaccination", "Historical public-health challenges involving vaccination.", ["vaccination", "public health"], ["historical", "challenge"], ["animal vaccination", "computer history"]),
        _facet("84", "animals", "animal vaccination recommendations", "Recommendations for animal vaccination.", ["animal", "vaccination"], ["recommendation"], ["human vaccination", "computer vaccine"]),
    ]
    for order, row in enumerate(rows):
        row["manifest_order"] = order
    return rows


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _selection_hash(topic_id: str) -> str:
    return hashlib.sha256(f"{EXPERIMENT_ID}{topic_id}".encode("utf-8")).hexdigest()


def _topic_selection() -> list[dict[str, object]]:
    ordered = sorted(ELIGIBLE_TOPIC_IDS, key=_selection_hash)
    return [
        {
            "topic_id": topic_id,
            "selection_sha256": _selection_hash(topic_id),
            "selection_order": order,
            "selected": order < 4,
        }
        for order, topic_id in enumerate(ordered)
    ]


def _document_text(candidate: Mapping[str, object]) -> str:
    value = candidate.get("doc")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "contents", "body"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _read_original_cache(cache_root: Path, topic_id: str) -> dict[str, object]:
    matches = sorted(Path(cache_root).glob(f"{topic_id}__original__*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one original cache file for topic {topic_id}; found {len(matches)}"
        )
    path = matches[0]
    raw = path.read_bytes()
    try:
        source = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid original cache JSON for topic {topic_id}") from exc
    if not isinstance(source, Mapping) or str(source.get("topic_id")) != topic_id:
        raise ValueError(f"topic mismatch in original cache file for topic {topic_id}")
    if source.get("query") != SOURCE_QUERIES[topic_id]:
        raise ValueError(f"query mismatch in original cache file for topic {topic_id}")
    if source.get("hits") != ORIGINAL_HITS:
        raise ValueError(f"original cache for topic {topic_id} must freeze hits=1000")
    response = source.get("response")
    candidates = response.get("candidates") if isinstance(response, Mapping) else None
    if not isinstance(candidates, list) or len(candidates) != ORIGINAL_HITS:
        raise ValueError(f"original cache for topic {topic_id} must contain exactly 1,000 candidates")
    docids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("docid"), str):
            raise ValueError(f"original cache for topic {topic_id} has an invalid candidate")
        docids.append(str(candidate["docid"]))
        if not _document_text(candidate):
            raise ValueError(f"original cache for topic {topic_id} must be text-bearing")
    if len(set(docids)) != ORIGINAL_HITS:
        raise ValueError(f"original cache for topic {topic_id} must contain 1,000 unique document IDs")
    response_query = response.get("query") if isinstance(response, Mapping) else None
    if isinstance(response_query, Mapping):
        response_query = response_query.get("text")
    if response_query != SOURCE_QUERIES[topic_id]:
        raise ValueError(f"response query mismatch in original cache file for topic {topic_id}")
    return {
        "topic_id": topic_id,
        "query": SOURCE_QUERIES[topic_id],
        "common_query": COMMON_QUERIES[topic_id],
        "manifest_order": TOPIC_IDS.index(topic_id),
        "selection_sha256": _selection_hash(topic_id),
        "original_hits": ORIGINAL_HITS,
        "original_candidate_count": ORIGINAL_HITS,
        "original_cache_filename": path.name,
        "original_cache_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _expected_hashes(payload: Mapping[str, object]) -> dict[str, str]:
    unhashed = {key: value for key, value in payload.items() if key != "hashes"}
    return {
        "topics_sha256": _sha256(payload["topics"]),
        "facets_sha256": _sha256(payload["facets"]),
        "freeze_sha256": _sha256(unhashed),
    }


def build_manifest(cache_root: Path) -> dict[str, object]:
    topics = [_read_original_cache(Path(cache_root), topic_id) for topic_id in TOPIC_IDS]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "topic_ids": list(TOPIC_IDS),
        "eligible_topic_ids": list(ELIGIBLE_TOPIC_IDS),
        "excluded_topic_ids": sorted(EXCLUDED_TOPIC_IDS, key=int),
        "topic_selection": _topic_selection(),
        "topics": topics,
        "facets": _facet_records(),
        "analyzer": {
            "version": ANALYZER_VERSION,
            "rules": "lowercase ascii-alphanumeric; stop list; frozen suffix stemmer",
            "stopwords_sha256": hashlib.sha256("\n".join(sorted(_STOPWORDS)).encode()).hexdigest(),
        },
        "retrieval": {
            "endpoint": "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search",
            "index": "climbmix-400b",
            "original_hits": ORIGINAL_HITS,
            "facet_hits": FACET_HITS,
            "max_external_requests": 25,
            "minimum_interval_seconds": 3,
            "retry_count": 0,
        },
        "qrels_opened": False,
    }
    payload["hashes"] = _expected_hashes(payload)
    validate_manifest(payload, cache_root=Path(cache_root))
    return payload


def validate_manifest(payload: Mapping[str, object], *, cache_root: Path) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected manifest identity")
    topic_ids = payload.get("topic_ids")
    if not isinstance(topic_ids, list):
        raise ValueError("topic_ids must be an array")
    if set(map(str, topic_ids)) & EXCLUDED_TOPIC_IDS:
        raise ValueError("manifest contains an excluded topic")
    if topic_ids != list(TOPIC_IDS):
        raise ValueError("topic_ids do not match the frozen order")
    if payload.get("qrels_opened") is not False:
        raise ValueError("manifest cannot record qrels access")
    if payload.get("topic_selection") != _topic_selection():
        raise ValueError("topic selection does not match the frozen hash order")
    if payload.get("facets") != _facet_records():
        raise ValueError("manifest differs from the frozen facets")
    topics = payload.get("topics")
    if not isinstance(topics, list) or len(topics) != 4:
        raise ValueError("manifest must contain four topic records")
    expected_topics = [_read_original_cache(Path(cache_root), topic_id) for topic_id in TOPIC_IDS]
    if topics != expected_topics:
        raise ValueError("manifest topic sources differ from authenticated caches")
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping) or dict(hashes) != _expected_hashes(payload):
        raise ValueError("manifest hashes do not match canonical content")


def assert_mutation_allowed(output_root: Path) -> None:
    if (Path(output_root) / "QRELS_ACCESSED").exists():
        raise ValueError("qrels already accessed; upstream mutation is forbidden")


def load_manifest(path: Path, *, cache_root: Path) -> dict[str, object]:
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {path}") from exc
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, pretty=True):
        raise ValueError("manifest is not canonical JSON")
    validate_manifest(payload, cache_root=Path(cache_root))
    return payload


def _create(cache_root: Path, output: Path) -> dict[str, object]:
    assert_mutation_allowed(output.parent)
    payload = build_manifest(cache_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_canonical_bytes(payload, pretty=True))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--cache-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = _create(args.cache_root, args.output)
    print(
        json.dumps(
            {
                "topic_count": len(payload["topics"]),
                "facet_count": len(payload["facets"]),
                "original_rows": sum(int(row["original_candidate_count"]) for row in payload["topics"]),
                "qrels_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

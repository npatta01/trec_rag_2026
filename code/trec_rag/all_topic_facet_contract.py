"""Offline contract freezer for the all-topic tethered-facet validation.

The module creates planning receipts only.  It has no retrieval, qrels, or model
imports, and the authorization receipt is written before the topic TSV is read.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


ALL_TOPIC_IDS = (
    "14", "31", "37", "58", "72", "84", "144", "161", "200", "213",
    "219", "224", "225", "233", "273", "300", "407", "477", "499",
    "515", "707", "897",
)
ORIGINAL_DEPTH = 1000
FACET_DEPTH = 200
REQUEST_INTERVAL_SECONDS = 3.0
SCHEMA_VERSION = "all-topic-tethered-facet-manifest-v1"
EXPERIMENT_ID = "all_topic_tethered_facet_validation_v1"
ANALYZER_CONTRACT = "lowercase-alphanumeric-v1"
ALLOWED_BRIDGE_PURPOSES = frozenset(
    {"domain_disambiguation", "population_binding", "relation_paraphrasing", "standard_concept_name"}
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_FACET_KEYS = {
    "topic_id", "facet_id", "obligation_id", "obligation", "query",
    "anchor_terms", "domain_terms", "relation_terms", "analyzer_terms",
    "bridge_terms", "manifest_order", "query_sha256",
}
_BRIDGE_KEYS = {
    "surface", "source", "purpose", "scope_rationale", "analyzer_output",
    "not_candidate_answer_rationale",
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _analyze(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def validate_authorized_scope(topic_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(map(str, topic_ids))
    if normalized != ALL_TOPIC_IDS:
        extra = sorted(set(normalized) - set(ALL_TOPIC_IDS))
        missing = sorted(set(ALL_TOPIC_IDS) - set(normalized))
        raise ValueError(f"topic scope is outside all-topic authorization (extra={extra}, missing={missing}, order_exact=False)")
    return normalized


def _reject_qrels_fields(value: object, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "qrel" in str(key).lower():
                raise ValueError(f"qrels fields are forbidden before evaluation: {path}.{key}")
            _reject_qrels_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_qrels_fields(child, f"{path}[{index}]")


def _nonempty_strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be a nonempty tether array of strings")
    return value


def validate_facet_manifest(payload: Mapping[str, object], *, require_all_topics: bool = True) -> dict[str, object]:
    _reject_qrels_fields(payload)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("manifest identity mismatch")
    topic_ids_raw = payload.get("topic_ids")
    if not isinstance(topic_ids_raw, list):
        raise ValueError("topic_ids must be an array")
    topic_ids = tuple(map(str, topic_ids_raw))
    if require_all_topics:
        validate_authorized_scope(topic_ids)
    elif not topic_ids or len(topic_ids) != len(set(topic_ids)) or not set(topic_ids) <= set(ALL_TOPIC_IDS):
        raise ValueError("topic scope is outside all-topic authorization")
    analyzer = payload.get("analyzer")
    if not isinstance(analyzer, Mapping) or analyzer.get("contract") not in {ANALYZER_CONTRACT, "lowercase-whitespace-v1"}:
        raise ValueError("analyzer contract mismatch")
    topics = payload.get("topics")
    facets = payload.get("facets")
    if not isinstance(topics, list) or not isinstance(facets, list):
        raise ValueError("topics and facets must be arrays")
    if [str(row.get("topic_id")) for row in topics if isinstance(row, Mapping)] != list(topic_ids):
        raise ValueError("topic records must follow deterministic authorized order")
    topic_by_id: dict[str, Mapping[str, object]] = {}
    for order, row in enumerate(topics):
        if not isinstance(row, Mapping) or int(row.get("manifest_order", -1)) != order:
            raise ValueError("topic records must follow deterministic order")
        topic_id, narrative = str(row.get("topic_id")), str(row.get("narrative", ""))
        if not narrative or row.get("narrative_sha256") != _sha256_text(narrative):
            raise ValueError(f"narrative hash mismatch for {topic_id}")
        topic_by_id[topic_id] = row
    obligation_ids: set[tuple[str, str]] = set()
    obligation_texts: set[tuple[str, str]] = set()
    facet_ids: set[str] = set()
    counts = {topic_id: 0 for topic_id in topic_ids}
    for order, facet in enumerate(facets):
        if not isinstance(facet, Mapping) or set(facet) != _FACET_KEYS:
            raise ValueError("facet schema mismatch")
        if int(facet.get("manifest_order", -1)) != order:
            raise ValueError("facet records must follow deterministic order")
        topic_id = str(facet["topic_id"])
        if topic_id not in topic_by_id:
            raise ValueError(f"facet topic {topic_id} is outside all-topic authorization")
        facet_id, obligation_id = str(facet["facet_id"]), str(facet["obligation_id"])
        obligation_text = " ".join(str(facet["obligation"]).lower().split())
        if (
            not obligation_text
            or facet_id in facet_ids
            or (topic_id, obligation_id) in obligation_ids
            or (topic_id, obligation_text) in obligation_texts
        ):
            raise ValueError("facet IDs and obligations must be unique")
        facet_ids.add(facet_id)
        obligation_ids.add((topic_id, obligation_id))
        obligation_texts.add((topic_id, obligation_text))
        query = str(facet["query"])
        if not query or facet["query_sha256"] != _sha256_text(query):
            raise ValueError(f"query hash mismatch for {facet_id}")
        analyzed = _analyze(query) if analyzer.get("contract") == ANALYZER_CONTRACT else query.lower().split()
        if facet["analyzer_terms"] != analyzed:
            raise ValueError(f"exact analyzer terms mismatch for {facet_id}")
        groups = [
            _nonempty_strings(facet[name], f"{facet_id}.{name}")
            for name in ("anchor_terms", "domain_terms", "relation_terms")
        ]
        analyzer_set = set(analyzed)
        if any(not any(set(_analyze(term)) <= analyzer_set for term in group) for group in groups):
            raise ValueError(f"facet {facet_id} lacks a subject/domain/relation tether")
        bridge_terms = facet["bridge_terms"]
        if not isinstance(bridge_terms, list):
            raise ValueError("bridge terms must be an array")
        for bridge in bridge_terms:
            if not isinstance(bridge, Mapping) or set(bridge) != _BRIDGE_KEYS:
                raise ValueError("bridge provenance is incomplete")
            if bridge["purpose"] not in ALLOWED_BRIDGE_PURPOSES:
                raise ValueError("bridge purpose is not allowed")
            if any(not bridge[key] for key in _BRIDGE_KEYS):
                raise ValueError("bridge provenance is incomplete")
            if bridge["analyzer_output"] != _analyze(str(bridge["surface"])):
                raise ValueError("bridge analyzer output mismatch")
        counts[topic_id] += 1
    for topic_id, count in counts.items():
        explicit = int(topic_by_id[topic_id].get("explicit_obligation_count", 0))
        if count != explicit or count < min(3, explicit) or count > 9:
            raise ValueError(f"topic {topic_id} must have three-to-nine facets unless fewer explicit obligations exist")
    return dict(payload)


def build_request_plan(payload: Mapping[str, object], original_cache: Mapping[str, Mapping[str, object]], *, require_all_topics: bool = True) -> dict[str, object]:
    manifest = validate_facet_manifest(payload, require_all_topics=require_all_topics)
    originals: list[dict[str, object]] = []
    topics = {str(row["topic_id"]): row for row in manifest["topics"]}  # type: ignore[index]
    for topic_id in manifest["topic_ids"]:  # type: ignore[index]
        cache = original_cache.get(str(topic_id))
        if cache is None:
            raise ValueError(f"missing exact top-1,000 original cache for {topic_id}")
        narrative = str(topics[str(topic_id)]["narrative"])
        if cache.get("query") != narrative or cache.get("query_sha256") != _sha256_text(narrative):
            raise ValueError(f"original cache query identity mismatch for {topic_id}")
        if int(cache.get("hits", 0)) != ORIGINAL_DEPTH or int(cache.get("candidate_count", 0)) != ORIGINAL_DEPTH:
            raise ValueError(f"original cache depth/count mismatch for {topic_id}")
        originals.append({
            "topic_id": str(topic_id), "depth": ORIGINAL_DEPTH, "cache_hit": True,
            "cache_path": str(cache["path"]), "cache_sha256": str(cache["sha256"]),
            "query_sha256": str(cache["query_sha256"]),
        })
    facet_requests = [{
        "request_order": order, "topic_id": facet["topic_id"], "facet_id": facet["facet_id"],
        "variant_name": f"{EXPERIMENT_ID}:{facet['facet_id']}", "query": facet["query"],
        "query_sha256": facet["query_sha256"], "analyzer_terms": facet["analyzer_terms"],
        "depth": FACET_DEPTH,
    } for order, facet in enumerate(manifest["facets"])]  # type: ignore[index]
    return {
        "schema_version": "all-topic-facet-request-plan-v1", "experiment_id": EXPERIMENT_ID,
        "topic_ids": list(manifest["topic_ids"]), "original_depth": ORIGINAL_DEPTH,
        "facet_depth": FACET_DEPTH, "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "original_cache_hit_count": len(originals), "original_request_count": 0,
        "facet_request_count": len(facet_requests), "total_external_request_count": len(facet_requests),
        "originals": originals, "facet_requests": facet_requests,
    }


def _authorization(topic_ids: Sequence[str], *, require_all_topics: bool) -> dict[str, object]:
    ids = validate_authorized_scope(topic_ids) if require_all_topics else tuple(map(str, topic_ids))
    if not ids or not set(ids) <= set(ALL_TOPIC_IDS):
        raise ValueError("topic scope is outside all-topic authorization")
    return {
        "schema_version": "all-topic-authorization-v1", "experiment_id": EXPERIMENT_ID,
        "authorized_topic_ids": list(ids), "authorization_kind": "experiment-specific-exact-allowlist",
        "shared_protection_constants_modified": False, "topic_sources_opened_before_receipt": False,
        "retrieval_permitted_during_freeze": False, "model_permitted_during_freeze": False,
        "qrels_permitted_during_freeze": False,
    }


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)


def freeze_planning(output: Path, *, topic_ids: Sequence[str] = ALL_TOPIC_IDS, source_loader: Callable[[], Mapping[str, object]], original_cache: Mapping[str, Mapping[str, object]], require_all_topics: bool = True) -> dict[str, object]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    authorization = _authorization(topic_ids, require_all_topics=require_all_topics)
    _write_exclusive(output / "authorization.json", _pretty_bytes(authorization))
    manifest = dict(source_loader())
    validate_facet_manifest(manifest, require_all_topics=require_all_topics)
    if list(manifest["topic_ids"]) != list(topic_ids):
        raise ValueError("source manifest does not match sealed authorization")
    request_plan = build_request_plan(manifest, original_cache, require_all_topics=require_all_topics)
    _write_exclusive(output / "manifest.json", _pretty_bytes(manifest))
    _write_exclusive(output / "request_plan.json", _pretty_bytes(request_plan))
    files = {}
    for name in ("authorization.json", "manifest.json", "request_plan.json"):
        raw = (output / name).read_bytes()
        files[name] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    seal = {
        "schema_version": "all-topic-planning-seal-v1", "experiment_id": EXPERIMENT_ID,
        "files": files, "root_sha256": _sha256_bytes(_canonical_bytes(files)),
    }
    _write_exclusive(output / "SEALED.json", _pretty_bytes(seal))
    return seal


def verify_planning(planning: Path, *, require_all_topics: bool = True) -> dict[str, object]:
    planning = Path(planning)
    seal = json.loads((planning / "SEALED.json").read_text(encoding="utf-8"))
    expected_names = {"authorization.json", "manifest.json", "request_plan.json", "SEALED.json"}
    if {path.name for path in planning.iterdir()} != expected_names:
        raise ValueError("planning seal file set mismatch")
    files = seal.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_names - {"SEALED.json"}:
        raise ValueError("planning seal binding mismatch")
    for name, binding in files.items():
        raw = (planning / name).read_bytes()
        if not isinstance(binding, Mapping) or binding.get("bytes") != len(raw) or binding.get("sha256") != _sha256_bytes(raw):
            raise ValueError(f"planning seal mismatch for {name}")
    if seal.get("root_sha256") != _sha256_bytes(_canonical_bytes(files)):
        raise ValueError("planning seal root mismatch")
    authorization = json.loads((planning / "authorization.json").read_text())
    topic_ids = tuple(map(str, authorization["authorized_topic_ids"]))
    if require_all_topics:
        validate_authorized_scope(topic_ids)
    manifest = json.loads((planning / "manifest.json").read_text())
    validate_facet_manifest(manifest, require_all_topics=require_all_topics)
    plan = json.loads((planning / "request_plan.json").read_text())
    if plan["topic_ids"] != list(topic_ids) or plan["original_cache_hit_count"] != len(topic_ids) or plan["original_request_count"] != 0:
        raise ValueError("request plan authorization/cache invariant mismatch")
    if plan["facet_request_count"] != len(manifest["facets"]) or any(row["depth"] != FACET_DEPTH for row in plan["facet_requests"]):
        raise ValueError("request plan facet count/depth mismatch")
    return {"verified": True, "topic_count": len(topic_ids), "facet_request_count": plan["facet_request_count"], "original_request_count": 0, "root_sha256": seal["root_sha256"]}

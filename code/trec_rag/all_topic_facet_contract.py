"""Qrels-blind planning contract for all-topic tethered-facet validation.

Only the exact ordered 22-topic experiment may use the public artifact APIs.
Small synthetic tests use private pure validators, which cannot freeze or verify
experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
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
ORIGINAL_RETRIEVER_NAME = "climbmix_bm25"
ORIGINAL_RETRIEVER_TYPE = "pyserini_remote"
ORIGINAL_INDEX = "climbmix-400b"
ORIGINAL_INDEX_URL = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
ORIGINAL_RESPONSE_API = "v1"
SCHEMA_VERSION = "all-topic-tethered-facet-manifest-v1"
EXPERIMENT_ID = "all_topic_tethered_facet_validation_v1"
ANALYZER_CONTRACT = "lowercase-alphanumeric-v1"
AUTHORIZATION_SCHEMA_VERSION = "all-topic-authorization-v1"
AUTHORIZATION_PRESEAL_SCHEMA_VERSION = "all-topic-authorization-preseal-v1"
REQUEST_PLAN_SCHEMA_VERSION = "all-topic-facet-request-plan-v1"
PLANNING_SEAL_SCHEMA_VERSION = "all-topic-planning-seal-v1"
ALLOWED_BRIDGE_PURPOSES = frozenset(
    {
        "domain_disambiguation",
        "population_binding",
        "relation_paraphrasing",
        "standard_concept_name",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ORDINARY_QUERY_TERMS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
        "for", "from", "how", "in", "into", "is", "of", "on", "or", "that",
        "the", "their", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "with",
    }
)
_FACET_KEYS = {
    "topic_id", "facet_id", "obligation_id", "obligation", "query",
    "anchor_terms", "domain_terms", "relation_terms", "analyzer_terms",
    "bridge_terms", "manifest_order", "query_sha256",
}
_MANIFEST_KEYS = {
    "schema_version", "experiment_id", "topic_ids", "analyzer", "topics",
    "facets",
}
_ANALYZER_KEYS = {"contract"}
_TOPIC_KEYS = {
    "topic_id", "narrative", "narrative_sha256",
    "explicit_obligation_count", "manifest_order",
}
_BRIDGE_KEYS = {
    "surface", "source", "purpose", "scope_rationale", "analyzer_output",
    "not_candidate_answer_rationale",
}
_AUTHORIZATION_FIREWALL = {
    "shared_protection_constants_modified": False,
    "topic_sources_opened_before_receipt": False,
    "retrieval_permitted_during_freeze": False,
    "model_permitted_during_freeze": False,
    "qrels_permitted_during_freeze": False,
}
_AUTHORIZATION_KEYS = {
    "schema_version",
    "experiment_id",
    "authorized_topic_ids",
    "authorization_kind",
    *_AUTHORIZATION_FIREWALL,
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _analyze(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _stem(token: str) -> str:
    """Conservative morphology used only to recognize source/query variants."""

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        return base[:-1] if len(base) > 2 and base[-1] == base[-2] else base
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _supported_by_source(token: str, source_tokens: set[str]) -> bool:
    return token in source_tokens or any(_stem(token) == _stem(item) for item in source_tokens)


def validate_authorized_scope(topic_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(map(str, topic_ids))
    if normalized != ALL_TOPIC_IDS:
        extra = sorted(set(normalized) - set(ALL_TOPIC_IDS))
        missing = sorted(set(ALL_TOPIC_IDS) - set(normalized))
        raise ValueError(
            "topic scope is outside all-topic authorization "
            f"(extra={extra}, missing={missing}, order_exact=False)"
        )
    return normalized


def _reject_qrels_fields(value: object, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "qrel" in str(key).lower():
                raise ValueError(
                    f"qrels fields are forbidden before evaluation: {path}.{key}"
                )
            _reject_qrels_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_qrels_fields(child, f"{path}[{index}]")


def _tether_tokens(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty tether array of strings")
    analyzed: list[str] = []
    for item in value:
        terms = _analyze(item)
        if not terms:
            raise ValueError(f"{label} contains an empty analyzed tether")
        analyzed.extend(terms)
    return analyzed


def _validate_bridge_terms(
    value: object,
    *,
    query_terms: list[str],
) -> set[str]:
    if not isinstance(value, list):
        raise ValueError("bridge terms must be an array")
    covered: set[str] = set()
    query_text = " ".join(query_terms)
    for bridge in value:
        if not isinstance(bridge, Mapping) or set(bridge) != _BRIDGE_KEYS:
            raise ValueError("bridge provenance is incomplete")
        for key in _BRIDGE_KEYS - {"analyzer_output"}:
            if not isinstance(bridge.get(key), str):
                raise ValueError("bridge provenance values must be strings")
        if bridge.get("purpose") not in ALLOWED_BRIDGE_PURPOSES:
            raise ValueError("bridge purpose is not allowed")
        if any(not bridge.get(key) for key in _BRIDGE_KEYS - {"analyzer_output"}):
            raise ValueError("bridge provenance is incomplete")
        surface_terms = _analyze(str(bridge.get("surface", "")))
        analyzer_output = bridge.get("analyzer_output")
        if (
            not isinstance(analyzer_output, list)
            or any(not isinstance(term, str) for term in analyzer_output)
            or not surface_terms
            or analyzer_output != surface_terms
        ):
            raise ValueError("bridge analyzer output must be exact and nonempty")
        surface_text = " ".join(surface_terms)
        if not re.search(rf"(?:^| ){re.escape(surface_text)}(?: |$)", query_text):
            raise ValueError(f"bridge surface {bridge['surface']!r} is absent from query")
        covered.update(surface_terms)
    return covered


def _validate_facet_manifest(
    payload: Mapping[str, object], authorized_topic_ids: Sequence[str]
) -> dict[str, object]:
    """Pure subset validator for tests; it cannot freeze or verify artifacts."""

    authorized = tuple(map(str, authorized_topic_ids))
    if not authorized or len(set(authorized)) != len(authorized):
        raise ValueError("private authorized topic scope must be unique and nonempty")
    _reject_qrels_fields(payload)
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("manifest top-level schema mismatch")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("manifest identity mismatch")
    topic_ids_raw = payload.get("topic_ids")
    if not isinstance(topic_ids_raw, list):
        raise ValueError("topic_ids must be an array")
    if any(not isinstance(topic_id, str) for topic_id in topic_ids_raw):
        raise ValueError("manifest topic IDs must be exact strings")
    topic_ids = tuple(topic_ids_raw)
    if topic_ids != authorized:
        raise ValueError("manifest topic_ids do not follow authorized order")
    analyzer = payload.get("analyzer")
    if (
        not isinstance(analyzer, Mapping)
        or set(analyzer) != _ANALYZER_KEYS
        or not isinstance(analyzer.get("contract"), str)
        or analyzer.get("contract") != ANALYZER_CONTRACT
    ):
        raise ValueError("analyzer contract or schema mismatch")
    topics = payload.get("topics")
    facets = payload.get("facets")
    if not isinstance(topics, list) or not isinstance(facets, list):
        raise ValueError("topics and facets must be arrays")
    if any(not isinstance(row, Mapping) or set(row) != _TOPIC_KEYS for row in topics):
        raise ValueError("topic record schema mismatch")
    if any(not isinstance(row.get("topic_id"), str) for row in topics):
        raise ValueError("topic record IDs must be exact strings")
    if [row.get("topic_id") for row in topics] != list(authorized):
        raise ValueError("topic records must follow authorized order")

    topic_by_id: dict[str, Mapping[str, object]] = {}
    for order, row in enumerate(topics):
        assert isinstance(row, Mapping)
        manifest_order = row.get("manifest_order")
        if (
            isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order != order
        ):
            raise ValueError("topic manifest order must be an exact integer")
        topic_id_value = row.get("topic_id")
        if not isinstance(topic_id_value, str):
            raise ValueError("topic record ID must be an exact string")
        topic_id = topic_id_value
        narrative = row.get("narrative")
        if (
            not isinstance(narrative, str)
            or not narrative
            or row.get("narrative_sha256") != _sha256_text(narrative)
        ):
            raise ValueError(f"narrative hash mismatch for {topic_id}")
        explicit = row.get("explicit_obligation_count")
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError(f"explicit obligation count is invalid for {topic_id}")
        topic_by_id[topic_id] = row

    facets_by_topic: dict[str, list[Mapping[str, object]]] = {
        topic_id: [] for topic_id in authorized
    }
    observed_topic_order: list[str] = []
    for facet in facets:
        if not isinstance(facet, Mapping) or set(facet) != _FACET_KEYS:
            raise ValueError("facet schema mismatch")
        topic_id_value = facet.get("topic_id")
        if not isinstance(topic_id_value, str):
            raise ValueError("facet topic ID must be an exact string")
        topic_id = topic_id_value
        if topic_id not in facets_by_topic:
            raise ValueError(f"facet topic {topic_id} is outside all-topic authorization")
        if not observed_topic_order or observed_topic_order[-1] != topic_id:
            observed_topic_order.append(topic_id)
        facets_by_topic[topic_id].append(facet)
    expected_nonempty_order = [topic_id for topic_id in authorized if facets_by_topic[topic_id]]
    if observed_topic_order != expected_nonempty_order:
        raise ValueError("facet records must follow authorized topic order")

    obligation_ids: set[tuple[str, str]] = set()
    obligation_texts: set[tuple[str, str]] = set()
    facet_ids: set[str] = set()
    for topic_id in authorized:
        topic = topic_by_id[topic_id]
        topic_facets = facets_by_topic[topic_id]
        explicit = int(topic["explicit_obligation_count"])
        count = len(topic_facets)
        if count != explicit or count < min(3, explicit) or count > 9:
            raise ValueError(
                f"topic {topic_id} must have three-to-nine facets unless fewer explicit obligations exist"
            )
        narrative = str(topic["narrative"])
        for local_order, facet in enumerate(topic_facets):
            facet_id_value = facet["facet_id"]
            obligation_id_value = facet["obligation_id"]
            if not isinstance(facet_id_value, str) or not isinstance(
                obligation_id_value, str
            ):
                raise ValueError("facet and obligation IDs must be exact strings")
            facet_id = facet_id_value
            obligation_id = obligation_id_value
            manifest_order = facet.get("manifest_order")
            if (
                isinstance(manifest_order, bool)
                or not isinstance(manifest_order, int)
                or manifest_order != local_order
            ):
                raise ValueError("facet topic-local manifest order must be an exact integer")
            if obligation_id != f"{topic_id}:o{local_order + 1}":
                raise ValueError("facets must follow deterministic topic-local obligation order")
            obligation = facet.get("obligation")
            if not isinstance(obligation, str) or not obligation.strip():
                raise ValueError("facet obligation must be nonempty")
            obligation_text = " ".join(obligation.lower().split())
            if (
                facet_id in facet_ids
                or (topic_id, obligation_id) in obligation_ids
                or (topic_id, obligation_text) in obligation_texts
            ):
                raise ValueError("facet IDs and obligations must be unique")
            facet_ids.add(facet_id)
            obligation_ids.add((topic_id, obligation_id))
            obligation_texts.add((topic_id, obligation_text))

            query = facet.get("query")
            if (
                not isinstance(query, str)
                or not query
                or facet.get("query_sha256") != _sha256_text(query)
            ):
                raise ValueError(f"query hash mismatch for {facet_id}")
            analyzed = _analyze(query)
            if not analyzed or facet.get("analyzer_terms") != analyzed:
                raise ValueError(f"exact analyzer terms mismatch for {facet_id}")
            query_set = set(analyzed)
            source_set = set(_analyze(f"{narrative} {obligation}"))
            for name in ("anchor_terms", "domain_terms", "relation_terms"):
                tether_terms = _tether_tokens(facet[name], f"{facet_id}.{name}")
                if any(not _supported_by_source(term, query_set) for term in tether_terms) or any(
                    not _supported_by_source(term, source_set) for term in tether_terms
                ):
                    raise ValueError(
                        f"facet {facet_id} has a {name} tether absent from its query or source"
                    )
            bridge_coverage = _validate_bridge_terms(
                facet["bridge_terms"], query_terms=analyzed
            )
            unsupported = sorted(
                term
                for term in query_set
                if not _supported_by_source(term, source_set)
                and term not in _ORDINARY_QUERY_TERMS
                and term not in bridge_coverage
            )
            if unsupported:
                raise ValueError(
                    f"unsupported query terms require bridge provenance: {', '.join(unsupported)}"
                )
    return dict(payload)


def validate_facet_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact ordered 22-topic experiment manifest."""

    validate_authorized_scope(tuple(map(str, payload.get("topic_ids", ()))))
    return _validate_facet_manifest(payload, ALL_TOPIC_IDS)


def _safe_cache_file(path_value: object, approved_cache_root: Path) -> Path:
    try:
        root = Path(approved_cache_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("approved cache root does not exist") from exc
    if not root.is_dir():
        raise ValueError("approved cache root must be a directory")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("cache path must be nonempty text")
    supplied = Path(path_value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("cache path is missing or escapes approved cache root") from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ValueError("cache path is unreadable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("cache path must identify an existing regular file")
    return resolved


def _request_cache_key(payload: Mapping[str, object]) -> str:
    identity = {
        "retriever_name": payload.get("retriever_name"),
        "retriever_type": payload.get("retriever_type"),
        "index": payload.get("index"),
        "index_url": payload.get("index_url"),
        "hits": payload.get("hits"),
        "query_text": payload.get("query"),
    }
    return _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:16]


def _authenticate_original_cache(
    topic_id: str,
    narrative: str,
    binding: Mapping[str, object],
    approved_cache_root: Path,
) -> dict[str, object]:
    declared_sha = binding.get("sha256")
    if not isinstance(declared_sha, str) or _SHA256_RE.fullmatch(declared_sha) is None:
        raise ValueError(f"cache sha256 is invalid for {topic_id}")
    path = _safe_cache_file(binding.get("path"), approved_cache_root)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cache file is unreadable for {topic_id}") from exc
    actual_sha = _sha256_bytes(raw)
    if actual_sha != declared_sha:
        raise ValueError(f"cache content hash mismatch for {topic_id}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache payload is invalid for {topic_id}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"cache payload must be an object for {topic_id}")
    if (
        payload.get("topic_id") != topic_id
        or payload.get("variant_name") != "original"
        or payload.get("query") != narrative
        or payload.get("hits") != ORIGINAL_DEPTH
        or payload.get("cache_key") != _request_cache_key(payload)
    ):
        raise ValueError(f"cache request identity mismatch for {topic_id}")
    approved_identity = {
        "retriever_name": ORIGINAL_RETRIEVER_NAME,
        "retriever_type": ORIGINAL_RETRIEVER_TYPE,
        "index": ORIGINAL_INDEX,
        "index_url": ORIGINAL_INDEX_URL,
    }
    if any(payload.get(key) != expected for key, expected in approved_identity.items()):
        raise ValueError(f"approved cache identity mismatch for {topic_id}")
    response = payload.get("response")
    if not isinstance(response, Mapping):
        raise ValueError(f"cache raw response provenance is missing for {topic_id}")
    response_query = response.get("query")
    candidates = response.get("candidates")
    if (
        response.get("api") != ORIGINAL_RESPONSE_API
        or response.get("index") != payload.get("index")
        or not isinstance(response_query, Mapping)
        or response_query.get("text") != narrative
        or not isinstance(candidates, list)
        or len(candidates) != ORIGINAL_DEPTH
    ):
        raise ValueError(f"cache raw response provenance/count mismatch for {topic_id}")
    seen: set[str] = set()
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"cache candidate schema mismatch for {topic_id}")
        docid = candidate.get("docid")
        if (
            not isinstance(docid, str)
            or not docid
            or docid in seen
            or candidate.get("rank") != expected_rank
            or not isinstance(candidate.get("doc"), str)
        ):
            raise ValueError(f"cache candidates are invalid for {topic_id}")
        seen.add(docid)
    request_identity = {
        "cache_key": payload["cache_key"],
        "topic_id": topic_id,
        "variant_name": "original",
        "retriever_name": payload["retriever_name"],
        "retriever_type": payload["retriever_type"],
        "index": payload["index"],
        "index_url": payload["index_url"],
        "hits": ORIGINAL_DEPTH,
        "query_sha256": _sha256_text(narrative),
    }
    return {
        "topic_id": topic_id,
        "depth": ORIGINAL_DEPTH,
        "candidate_count": len(candidates),
        "cache_hit": True,
        "cache_path": str(path),
        "cache_sha256": actual_sha,
        "request_identity": request_identity,
        "raw_response_provenance": {
            "api": response["api"],
            "index": response["index"],
            "query_sha256": _sha256_text(str(response_query["text"])),
            "cache_content_sha256": actual_sha,
        },
    }


def _build_request_plan(
    payload: Mapping[str, object],
    original_cache: Mapping[str, Mapping[str, object]],
    approved_cache_root: Path,
    authorized_topic_ids: Sequence[str],
) -> dict[str, object]:
    """Pure subset request-plan builder; it cannot freeze artifacts."""

    manifest = _validate_facet_manifest(payload, authorized_topic_ids)
    topics = {str(row["topic_id"]): row for row in manifest["topics"]}  # type: ignore[index]
    originals: list[dict[str, object]] = []
    for topic_id in authorized_topic_ids:
        binding = original_cache.get(str(topic_id))
        if not isinstance(binding, Mapping):
            raise ValueError(f"missing exact top-1,000 original cache for {topic_id}")
        originals.append(
            _authenticate_original_cache(
                str(topic_id),
                str(topics[str(topic_id)]["narrative"]),
                binding,
                Path(approved_cache_root),
            )
        )
    facet_requests = [
        {
            "request_order": order,
            "topic_id": facet["topic_id"],
            "facet_id": facet["facet_id"],
            "variant_name": f"{EXPERIMENT_ID}:{facet['facet_id']}",
            "query": facet["query"],
            "query_sha256": facet["query_sha256"],
            "analyzer_terms": facet["analyzer_terms"],
            "depth": FACET_DEPTH,
        }
        for order, facet in enumerate(manifest["facets"])  # type: ignore[index]
    ]
    return {
        "schema_version": REQUEST_PLAN_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "topic_ids": list(authorized_topic_ids),
        "original_depth": ORIGINAL_DEPTH,
        "facet_depth": FACET_DEPTH,
        "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "original_cache_hit_count": len(originals),
        "original_request_count": 0,
        "facet_request_count": len(facet_requests),
        "total_external_request_count": len(facet_requests),
        "originals": originals,
        "facet_requests": facet_requests,
    }


def build_request_plan(
    payload: Mapping[str, object],
    original_cache: Mapping[str, Mapping[str, object]],
    *,
    approved_cache_root: Path,
) -> dict[str, object]:
    """Authenticate caches and build the exact ordered 22-topic request plan."""

    validate_facet_manifest(payload)
    return _build_request_plan(
        payload, original_cache, Path(approved_cache_root), ALL_TOPIC_IDS
    )


def discover_original_caches(
    approved_cache_root: Path,
    narratives: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Return one authenticated original-cache binding per authorized topic.

    Discovery is deliberately local and byte-authenticated.  It neither follows
    directory entries that are symlinks nor accepts a merely plausible filename
    as proof of identity.
    """

    validate_authorized_scope(tuple(map(str, narratives)))
    root = Path(approved_cache_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("approved cache root must be a directory")
    bindings: dict[str, dict[str, str]] = {}
    for topic_id in ALL_TOPIC_IDS:
        narrative = narratives.get(topic_id)
        if not isinstance(narrative, str) or not narrative:
            raise ValueError(f"missing narrative for authorized topic {topic_id}")
        matches: list[dict[str, str]] = []
        pattern = f"{topic_id}__original__{ORIGINAL_RETRIEVER_NAME}__*.json"
        for path in sorted(root.glob(pattern)):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"cache discovery found unsafe path for {topic_id}")
            raw = path.read_bytes()
            binding = {"path": str(path), "sha256": _sha256_bytes(raw)}
            try:
                _authenticate_original_cache(topic_id, narrative, binding, root)
            except ValueError:
                continue
            matches.append(binding)
        if not matches:
            raise ValueError(f"missing exact top-1,000 original cache for {topic_id}")
        if len(matches) != 1:
            raise ValueError(f"duplicate authenticated original caches for {topic_id}")
        bindings[topic_id] = matches[0]
    return bindings


def _authorization() -> dict[str, object]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "authorized_topic_ids": list(ALL_TOPIC_IDS),
        "authorization_kind": "experiment-specific-exact-allowlist",
        **_AUTHORIZATION_FIREWALL,
    }


def _validate_authorization(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _AUTHORIZATION_KEYS:
        raise ValueError("authorization semantics have an invalid schema")
    if (
        value.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("authorization_kind")
        != "experiment-specific-exact-allowlist"
    ):
        raise ValueError("authorization semantics identity mismatch")
    raw_ids = value.get("authorized_topic_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("authorization semantics topic IDs are invalid")
    validate_authorized_scope(tuple(map(str, raw_ids)))
    if any(value.get(key) is not expected for key, expected in _AUTHORIZATION_FIREWALL.items()):
        raise ValueError("authorization semantics firewall flags mismatch")
    return dict(value)


def _authorization_preseal(authorization_raw: bytes) -> dict[str, object]:
    authorization = _validate_authorization(json.loads(authorization_raw))
    return {
        "schema_version": AUTHORIZATION_PRESEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "authorized_topic_ids": list(ALL_TOPIC_IDS),
        "firewall_flags": dict(_AUTHORIZATION_FIREWALL),
        "authorization_schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization": {
            "bytes": len(authorization_raw),
            "sha256": _sha256_bytes(authorization_raw),
        },
        "authorization_semantics_sha256": _sha256_bytes(
            _canonical_bytes(authorization)
        ),
    }


def _validate_authorization_preseal(value: object, authorization_raw: bytes) -> None:
    expected = _authorization_preseal(authorization_raw)
    if value != expected:
        raise ValueError("authorization pre-seal binding mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_fsync(path: Path, raw: bytes, *, read_only: bool = False) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if read_only:
        path.chmod(0o400)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def freeze_planning(
    output: Path,
    *,
    source_loader: Callable[[], Mapping[str, object]],
    original_cache: Mapping[str, Mapping[str, object]],
    approved_cache_root: Path,
) -> dict[str, object]:
    """Freeze planning only after an immutable exact-22 authorization pre-seal."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    authorization_raw = _pretty_bytes(_authorization())
    preseal_raw = _pretty_bytes(_authorization_preseal(authorization_raw))
    authorization_path = output / "authorization.json"
    preseal_path = output / "AUTHORIZATION_SEALED.json"
    _write_exclusive_fsync(authorization_path, authorization_raw, read_only=True)
    _write_exclusive_fsync(preseal_path, preseal_raw, read_only=True)

    manifest = dict(source_loader())
    if (
        authorization_path.read_bytes() != authorization_raw
        or preseal_path.read_bytes() != preseal_raw
        or authorization_path.stat().st_mode & 0o222
        or preseal_path.stat().st_mode & 0o222
    ):
        raise ValueError("authorization pre-seal changed during source loading")
    validate_facet_manifest(manifest)
    request_plan = build_request_plan(
        manifest,
        original_cache,
        approved_cache_root=Path(approved_cache_root),
    )
    _write_exclusive_fsync(output / "manifest.json", _pretty_bytes(manifest))
    _write_exclusive_fsync(
        output / "request_plan.json", _pretty_bytes(request_plan)
    )
    files: dict[str, dict[str, object]] = {}
    for name in (
        "authorization.json",
        "AUTHORIZATION_SEALED.json",
        "manifest.json",
        "request_plan.json",
    ):
        raw = (output / name).read_bytes()
        files[name] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    seal = {
        "schema_version": PLANNING_SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "authorization_preseal_sha256": files["AUTHORIZATION_SEALED.json"]["sha256"],
        "files": files,
        "root_sha256": _sha256_bytes(_canonical_bytes(files)),
    }
    _write_exclusive_fsync(output / "SEALED.json", _pretty_bytes(seal))
    return seal


def verify_planning(
    planning: Path, *, approved_cache_root: Path
) -> dict[str, object]:
    """Verify authorization semantics, pre-seal, caches, and exact-22 plan."""

    planning = Path(planning)
    expected_names = {
        "authorization.json",
        "AUTHORIZATION_SEALED.json",
        "manifest.json",
        "request_plan.json",
        "SEALED.json",
    }
    try:
        observed_names = {path.name for path in planning.iterdir()}
    except OSError as exc:
        raise ValueError("planning directory is unreadable") from exc
    if observed_names != expected_names:
        raise ValueError("planning seal file set mismatch")
    seal = _load_json(planning / "SEALED.json", "planning seal")
    if (
        seal.get("schema_version") != PLANNING_SEAL_SCHEMA_VERSION
        or seal.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("planning seal identity mismatch")
    files = seal.get("files")
    bound_names = expected_names - {"SEALED.json"}
    if not isinstance(files, Mapping) or set(files) != bound_names:
        raise ValueError("planning seal binding mismatch")
    for name, binding in files.items():
        raw = (planning / name).read_bytes()
        if (
            not isinstance(binding, Mapping)
            or binding.get("bytes") != len(raw)
            or binding.get("sha256") != _sha256_bytes(raw)
        ):
            raise ValueError(f"planning seal mismatch for {name}")
    if seal.get("root_sha256") != _sha256_bytes(_canonical_bytes(files)):
        raise ValueError("planning seal root mismatch")
    if seal.get("authorization_preseal_sha256") != files[
        "AUTHORIZATION_SEALED.json"
    ]["sha256"]:
        raise ValueError("planning seal authorization pre-seal hash mismatch")

    authorization_raw = (planning / "authorization.json").read_bytes()
    _validate_authorization(json.loads(authorization_raw))
    preseal = _load_json(
        planning / "AUTHORIZATION_SEALED.json", "authorization pre-seal"
    )
    _validate_authorization_preseal(preseal, authorization_raw)
    if (
        (planning / "authorization.json").stat().st_mode & 0o222
        or (planning / "AUTHORIZATION_SEALED.json").stat().st_mode & 0o222
    ):
        raise ValueError("authorization semantics are not immutable")

    manifest = _load_json(planning / "manifest.json", "facet manifest")
    validate_facet_manifest(manifest)
    plan = _load_json(planning / "request_plan.json", "request plan")
    originals = plan.get("originals")
    if not isinstance(originals, list) or len(originals) != len(ALL_TOPIC_IDS):
        raise ValueError("request plan cache bindings are invalid")
    cache_bindings: dict[str, Mapping[str, object]] = {}
    for row in originals:
        if not isinstance(row, Mapping):
            raise ValueError("request plan cache binding must be an object")
        cache_bindings[str(row.get("topic_id"))] = {
            "path": row.get("cache_path"),
            "sha256": row.get("cache_sha256"),
        }
    expected_plan = _build_request_plan(
        manifest,
        cache_bindings,
        Path(approved_cache_root),
        ALL_TOPIC_IDS,
    )
    if plan != expected_plan:
        raise ValueError("request plan authorization/cache invariant mismatch")
    return {
        "verified": True,
        "topic_count": len(ALL_TOPIC_IDS),
        "facet_request_count": plan["facet_request_count"],
        "original_request_count": 0,
        "root_sha256": seal["root_sha256"],
    }


def _manifest_narratives(payload: Mapping[str, object]) -> dict[str, str]:
    topics = payload.get("topics")
    if not isinstance(topics, list):
        raise ValueError("manifest topics must be an array")
    result: dict[str, str] = {}
    for row in topics:
        if not isinstance(row, Mapping):
            raise ValueError("manifest topic record must be an object")
        topic_id = str(row.get("topic_id"))
        narrative = row.get("narrative")
        if not isinstance(narrative, str):
            raise ValueError(f"manifest narrative is invalid for {topic_id}")
        result[topic_id] = narrative
    validate_authorized_scope(tuple(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze or verify planning using only local manifest/cache bytes."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--manifest", type=Path, required=True)
    freeze_parser.add_argument("--cache-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--cache-root", type=Path, required=True)
    verify_parser.add_argument("--planning", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "freeze":
        manifest = _load_json(args.manifest, "facet manifest")
        validate_facet_manifest(manifest)
        bindings = discover_original_caches(
            args.cache_root, _manifest_narratives(manifest)
        )
        freeze_planning(
            args.output,
            source_loader=lambda: manifest,
            original_cache=bindings,
            approved_cache_root=args.cache_root,
        )
    else:
        verify_planning(args.planning, approved_cache_root=args.cache_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())

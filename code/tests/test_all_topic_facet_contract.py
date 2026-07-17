from __future__ import annotations

import builtins
import copy
import hashlib
import http.client
import inspect
import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest

from trec_rag.adaptive_evidence_contract import PROTECTED_TOPIC_IDS
from trec_rag.all_topic_facet_contract import (
    ALL_TOPIC_IDS,
    ANALYZER_CONTRACT,
    FACET_DEPTH,
    ORIGINAL_DEPTH,
    REQUEST_INTERVAL_SECONDS,
    _build_request_plan,
    _validate_facet_manifest,
    build_request_plan,
    discover_original_caches,
    freeze_planning,
    main,
    validate_authorized_scope,
    validate_facet_manifest,
    verify_planning,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MANIFEST = (
    REPO_ROOT
    / "reports/experiments/all_topic_tethered_facet_validation_v1/facet_manifest.json"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _analyze(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _facet(
    *,
    topic_id: str = "14",
    facet_id: str | None = None,
    obligation_id: str | None = None,
    obligation: str = "How athlete compensation affects the societal role of sports.",
    query: str = "sports athletes compensation societal impact",
    anchor_terms: list[str] | None = None,
    domain_terms: list[str] | None = None,
    relation_terms: list[str] | None = None,
    bridge_terms: list[dict[str, object]] | None = None,
    order: int = 0,
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "facet_id": facet_id or f"{topic_id}-compensation",
        "obligation_id": obligation_id or f"{topic_id}:o{order + 1}",
        "obligation": obligation,
        "query": query,
        "anchor_terms": ["sports", "athlete"] if anchor_terms is None else anchor_terms,
        "domain_terms": ["societal"] if domain_terms is None else domain_terms,
        "relation_terms": ["compensation", "impact"] if relation_terms is None else relation_terms,
        "analyzer_terms": _analyze(query),
        "bridge_terms": [] if bridge_terms is None else bridge_terms,
        "manifest_order": order,
        "query_sha256": _sha256(query.encode()),
    }


def _manifest(topic_ids: tuple[str, ...] = ("14",)) -> dict[str, object]:
    topics: list[dict[str, object]] = []
    facets: list[dict[str, object]] = []
    for topic_order, topic_id in enumerate(topic_ids):
        narrative = (
            "Sports athletes and athlete compensation have a societal impact."
        )
        topics.append(
            {
                "topic_id": topic_id,
                "narrative": narrative,
                "narrative_sha256": _sha256(narrative.encode()),
                "explicit_obligation_count": 1,
                "manifest_order": topic_order,
            }
        )
        facets.append(_facet(topic_id=topic_id))
    return {
        "schema_version": "all-topic-tethered-facet-manifest-v1",
        "experiment_id": "all_topic_tethered_facet_validation_v1",
        "topic_ids": list(topic_ids),
        "analyzer": {"contract": ANALYZER_CONTRACT},
        "topics": topics,
        "facets": facets,
    }


def _write_cache(
    root: Path,
    *,
    topic_id: str,
    narrative: str,
    mutate: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "retriever_name": "climbmix_bm25",
        "retriever_type": "pyserini_remote",
        "index": "climbmix-400b",
        "index_url": "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search",
        "hits": ORIGINAL_DEPTH,
        "query_text": narrative,
    }
    cache_key = _sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )[:16]
    payload: dict[str, object] = {
        "cache_key": cache_key,
        "hits": ORIGINAL_DEPTH,
        "index": identity["index"],
        "index_url": identity["index_url"],
        "query": narrative,
        "response": {
            "api": "v1",
            "index": identity["index"],
            "query": {"text": narrative},
            "candidates": [
                {"doc": f"document {rank}", "docid": f"{topic_id}-{rank}", "rank": rank, "score": 1.0}
                for rank in range(1, ORIGINAL_DEPTH + 1)
            ],
        },
        "retriever_name": identity["retriever_name"],
        "retriever_type": identity["retriever_type"],
        "topic_id": topic_id,
        "variant_name": "original",
    }
    if mutate:
        payload.update(mutate)
    path = root / f"{topic_id}__original__climbmix_bm25__{cache_key}.json"
    raw = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": _sha256(raw)}


def _caches(root: Path, manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = manifest["topics"]
    assert isinstance(rows, list)
    return {
        str(row["topic_id"]): _write_cache(
            root, topic_id=str(row["topic_id"]), narrative=str(row["narrative"])
        )
        for row in rows
    }


def _write_exact_cache_files(
    root: Path, topic_ids: tuple[str, ...]
) -> dict[str, str]:
    narratives = {topic_id: f"Narrative for topic {topic_id}." for topic_id in topic_ids}
    for topic_id, narrative in narratives.items():
        _write_cache(root, topic_id=topic_id, narrative=narrative)
    return narratives


def _forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("network access is forbidden during planning")


def test_production_manifest_is_exact_and_valid() -> None:
    payload = json.loads(PRODUCTION_MANIFEST.read_text())
    validated = validate_facet_manifest(payload)
    assert tuple(validated["topic_ids"]) == ALL_TOPIC_IDS
    assert 3 * len(ALL_TOPIC_IDS) <= len(validated["facets"]) <= 9 * len(ALL_TOPIC_IDS)


def test_reviewed_production_facet_splits_and_queries_are_frozen() -> None:
    payload = json.loads(PRODUCTION_MANIFEST.read_text())
    by_topic: dict[str, list[dict[str, object]]] = {
        topic_id: [] for topic_id in ALL_TOPIC_IDS
    }
    for facet in payload["facets"]:
        by_topic[str(facet["topic_id"])].append(facet)

    assert [row["facet_id"] for row in by_topic["58"]] == [
        "58-pros", "58-cons", "58-safety", "58-accident-risks", "58-uses",
        "58-climate", "58-fusion", "58-bison-energy", "58-peninsula-clean-energy",
    ]
    assert [row["facet_id"] for row in by_topic["144"]][2:7] == [
        "144-safety", "144-trust", "144-services", "144-regulation",
        "144-economic-development",
    ]
    assert [row["facet_id"] for row in by_topic["200"]] == [
        "200-definition", "200-why", "200-how", "200-responsibility",
        "200-historical-impact", "200-societal-impact", "200-conclusion",
        "200-lasting-effects", "200-historical-comparison",
    ]
    assert [row["facet_id"] for row in by_topic["213"]] == [
        "213-origins", "213-ending", "213-us-involvement", "213-us-politics",
        "213-errors", "213-presidents",
    ]
    assert [row["facet_id"] for row in by_topic["219"]] == [
        "219-daily-life-positive", "219-daily-life-negative",
        "219-government-positive", "219-government-negative",
        "219-business-positive", "219-business-negative", "219-societies",
        "219-rationing",
    ]
    queries = {row["facet_id"]: row["query"] for row in payload["facets"]}
    assert queries["161-current-options"] == (
        "current abortion options including abortion pill"
    )
    assert queries["477-definition"] == "concept of race how defined"
    assert queries["499-western-countries"] == (
        "euthanasia Western countries perspectives influence debates"
    )


def test_cache_discovery_binds_each_topic_once(tmp_path: Path) -> None:
    narratives = _write_exact_cache_files(tmp_path, ALL_TOPIC_IDS)
    bindings = discover_original_caches(tmp_path, narratives)
    assert tuple(bindings) == ALL_TOPIC_IDS
    assert len({row["path"] for row in bindings.values()}) == len(ALL_TOPIC_IDS)


def test_cache_discovery_rejects_duplicate_binding(tmp_path: Path) -> None:
    narratives = _write_exact_cache_files(tmp_path, ALL_TOPIC_IDS)
    original = next(tmp_path.glob("14__original__*.json"))
    duplicate = tmp_path / original.name.replace(".json", "__duplicate.json")
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        discover_original_caches(tmp_path, narratives)


def test_cli_freeze_and_verify_never_open_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    narratives = _write_exact_cache_files(tmp_path, ALL_TOPIC_IDS)
    payload = json.loads(PRODUCTION_MANIFEST.read_text())
    for topic in payload["topics"]:
        narratives[str(topic["topic_id"])] = str(topic["narrative"])
    # Rebuild with the exact production narratives expected by authentication.
    for path in tmp_path.glob("*.json"):
        path.unlink()
    for topic_id, narrative in narratives.items():
        _write_cache(tmp_path, topic_id=topic_id, narrative=narrative)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    planning = tmp_path / "planning"
    assert main([
        "freeze", "--manifest", str(PRODUCTION_MANIFEST),
        "--cache-root", str(tmp_path), "--output", str(planning),
    ]) == 0
    assert main([
        "verify", "--cache-root", str(tmp_path), "--planning", str(planning),
    ]) == 0


def test_cli_firewall_blocks_network_qrels_and_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(PRODUCTION_MANIFEST.read_text())
    narratives = {
        str(topic["topic_id"]): str(topic["narrative"])
        for topic in payload["topics"]
    }
    for topic_id, narrative in narratives.items():
        _write_cache(tmp_path, topic_id=topic_id, narrative=narrative)

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", _forbidden)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", _forbidden)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", _forbidden)
    real_open = builtins.open
    real_os_open = os.open
    real_path_open = Path.open
    real_import = builtins.__import__

    def guarded_open(file: object, *args: object, **kwargs: object):
        if "qrel" in str(file).lower():
            raise AssertionError("qrels access is forbidden during planning")
        return real_open(file, *args, **kwargs)

    def guarded_path_open(path: Path, *args: object, **kwargs: object):
        if "qrel" in str(path).lower():
            raise AssertionError("qrels access is forbidden during planning")
        return real_path_open(path, *args, **kwargs)

    def guarded_os_open(path: object, *args: object, **kwargs: object):
        if "qrel" in str(path).lower():
            raise AssertionError("qrels access is forbidden during planning")
        return real_os_open(path, *args, **kwargs)

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name.split(".", 1)[0] in {"torch", "transformers", "sentence_transformers"}:
            raise AssertionError("model loading is forbidden during planning")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    planning = tmp_path / "planning"
    assert main([
        "freeze", "--manifest", str(PRODUCTION_MANIFEST),
        "--cache-root", str(tmp_path), "--output", str(planning),
    ]) == 0
    assert main([
        "verify", "--cache-root", str(tmp_path), "--planning", str(planning),
    ]) == 0


def test_cache_discovery_rejects_missing_topic_and_escaping_symlink(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    narratives = _write_exact_cache_files(approved, ALL_TOPIC_IDS)
    missing = next(approved.glob("14__original__*.json"))
    missing.unlink()
    with pytest.raises(ValueError, match="missing.*14"):
        discover_original_caches(approved, narratives)

    _write_cache(approved, topic_id="14", narrative=narratives["14"])
    path = next(approved.glob("14__original__*.json"))
    outside = tmp_path / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe path.*14"):
        discover_original_caches(approved, narratives)


def test_manifest_rejects_coerced_ids_boolean_orders_and_extra_schema() -> None:
    for mutation in ("topic-list-id", "topic-record-id", "facet-topic-id"):
        payload = _manifest()
        if mutation == "topic-list-id":
            payload["topic_ids"][0] = 14
        elif mutation == "topic-record-id":
            payload["topics"][0]["topic_id"] = 14
        else:
            payload["facets"][0]["topic_id"] = 14
        with pytest.raises(ValueError, match="topic.*string"):
            _validate_facet_manifest(payload, ("14",))

    for collection, field in (
        ("topics", "manifest_order"),
        ("facets", "manifest_order"),
    ):
        payload = _manifest()
        payload[collection][0][field] = False
        with pytest.raises(ValueError, match="order.*integer"):
            _validate_facet_manifest(payload, ("14",))

    payload = _manifest()
    payload["topics"][0]["explicit_obligation_count"] = True
    with pytest.raises(ValueError, match="obligation count"):
        _validate_facet_manifest(payload, ("14",))

    for target in ("top", "analyzer", "topic"):
        payload = _manifest()
        if target == "top":
            payload["extra"] = "forbidden"
        elif target == "analyzer":
            payload["analyzer"]["extra"] = "forbidden"
        else:
            payload["topics"][0]["extra"] = "forbidden"
        with pytest.raises(ValueError, match="schema"):
            _validate_facet_manifest(payload, ("14",))


def test_scope_is_exactly_the_authorized_22() -> None:
    assert validate_authorized_scope(ALL_TOPIC_IDS) == ALL_TOPIC_IDS
    with pytest.raises(ValueError, match="outside all-topic authorization"):
        validate_authorized_scope((*ALL_TOPIC_IDS, "999"))


def test_public_artifact_apis_have_no_scope_escape_hatch() -> None:
    for function in (
        validate_facet_manifest,
        build_request_plan,
        freeze_planning,
        verify_planning,
    ):
        assert "require_all_topics" not in inspect.signature(function).parameters
    with pytest.raises(ValueError, match="outside all-topic authorization"):
        validate_facet_manifest(_manifest())


def test_shared_protected_constants_are_not_weakened() -> None:
    assert {"144", "213", "224", "407", "515"} <= PROTECTED_TOPIC_IDS


def test_only_frozen_alphanumeric_analyzer_is_accepted() -> None:
    payload = _manifest()
    payload["analyzer"] = {"contract": "lowercase-whitespace-v1"}
    with pytest.raises(ValueError, match="analyzer contract"):
        _validate_facet_manifest(payload, ("14",))


@pytest.mark.parametrize("field", ["anchor_terms", "domain_terms", "relation_terms"])
def test_tethers_require_nonempty_analyzed_text_in_query_and_source(field: str) -> None:
    payload = _manifest()
    facet = payload["facets"][0]
    facet[field] = ["!!!"]
    with pytest.raises(ValueError, match="tether"):
        _validate_facet_manifest(payload, ("14",))

    payload = _manifest()
    facet = payload["facets"][0]
    facet[field] = ["unsupported"]
    with pytest.raises(ValueError, match="tether"):
        _validate_facet_manifest(payload, ("14",))


def test_query_rejects_unsupported_terms_without_bridge_provenance() -> None:
    payload = _manifest()
    facet = payload["facets"][0]
    facet["query"] += " banana"
    facet["analyzer_terms"] = _analyze(str(facet["query"]))
    facet["query_sha256"] = _sha256(str(facet["query"]).encode())
    with pytest.raises(ValueError, match="unsupported query terms.*banana"):
        _validate_facet_manifest(payload, ("14",))


def test_bridge_surface_occurs_in_query_and_covers_unsupported_terms() -> None:
    payload = _manifest()
    facet = payload["facets"][0]
    facet["bridge_terms"] = [
        {
            "surface": "professional",
            "source": "standard domain vocabulary",
            "purpose": "domain_disambiguation",
            "scope_rationale": "Binds athletes to the professional sports domain.",
            "analyzer_output": ["professional"],
            "not_candidate_answer_rationale": "A domain label, not an effect.",
        }
    ]
    with pytest.raises(ValueError, match="bridge surface.*absent"):
        _validate_facet_manifest(payload, ("14",))

    facet["query"] += " professional"
    facet["analyzer_terms"] = _analyze(str(facet["query"]))
    facet["query_sha256"] = _sha256(str(facet["query"]).encode())
    _validate_facet_manifest(payload, ("14",))


def test_bridge_analyzer_output_must_be_exact_and_nonempty() -> None:
    payload = _manifest()
    facet = payload["facets"][0]
    facet["query"] += " professional"
    facet["analyzer_terms"] = _analyze(str(facet["query"]))
    facet["query_sha256"] = _sha256(str(facet["query"]).encode())
    facet["bridge_terms"] = [
        {
            "surface": "!!!",
            "source": "standard domain vocabulary",
            "purpose": "domain_disambiguation",
            "scope_rationale": "Domain binding.",
            "analyzer_output": [],
            "not_candidate_answer_rationale": "Not an answer.",
        }
    ]
    with pytest.raises(ValueError, match="bridge analyzer output"):
        _validate_facet_manifest(payload, ("14",))


def test_manifest_enforces_authorized_topic_and_topic_local_facet_order() -> None:
    payload = _manifest(("14", "31"))
    payload["topic_ids"] = ["31", "14"]
    with pytest.raises(ValueError, match="authorized order"):
        _validate_facet_manifest(payload, ("14", "31"))

    payload = _manifest(("14", "31"))
    payload["facets"][1]["manifest_order"] = 1
    with pytest.raises(ValueError, match="topic-local"):
        _validate_facet_manifest(payload, ("14", "31"))

    payload = _manifest(("14",))
    second = _facet(
        topic_id="14",
        facet_id="14-inclusion",
        obligation_id="14:o3",
        obligation="How athlete inclusion affects the societal role of sports.",
        query="sports athlete inclusion societal impact",
        relation_terms=["inclusion", "impact"],
        order=1,
    )
    payload["facets"].append(second)
    payload["topics"][0]["explicit_obligation_count"] = 2
    with pytest.raises(ValueError, match="obligation order"):
        _validate_facet_manifest(payload, ("14",))


def test_manifest_rejects_duplicate_obligations_hash_and_qrels_fields() -> None:
    payload = _manifest()
    duplicate = copy.deepcopy(payload)
    second = copy.deepcopy(duplicate["facets"][0])
    second["facet_id"] = "14-second"
    second["obligation_id"] = "14:o2"
    second["manifest_order"] = 1
    duplicate["facets"].append(second)
    duplicate["topics"][0]["explicit_obligation_count"] = 2
    with pytest.raises(ValueError, match="obligation"):
        _validate_facet_manifest(duplicate, ("14",))

    changed_hash = copy.deepcopy(payload)
    changed_hash["facets"][0]["query_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="query hash"):
        _validate_facet_manifest(changed_hash, ("14",))

    qrels = copy.deepcopy(payload)
    qrels["qrels_path"] = "forbidden"
    with pytest.raises(ValueError, match="qrels"):
        _validate_facet_manifest(qrels, ("14",))


def test_cache_is_authenticated_from_real_bytes_and_bound_in_plan(tmp_path: Path) -> None:
    payload = _manifest()
    cache = _caches(tmp_path, payload)
    plan = _build_request_plan(payload, cache, tmp_path, ("14",))

    original = plan["originals"][0]
    assert original["cache_sha256"] == cache["14"]["sha256"]
    assert original["request_identity"]["hits"] == ORIGINAL_DEPTH
    assert original["candidate_count"] == ORIGINAL_DEPTH
    assert original["raw_response_provenance"]["api"] == "v1"
    assert plan["original_request_count"] == 0
    assert plan["facet_requests"][0]["depth"] == FACET_DEPTH
    assert plan["request_interval_seconds"] == REQUEST_INTERVAL_SECONDS


@pytest.mark.parametrize("failure", ["missing", "tampered", "bad_sha", "escape"])
def test_cache_authentication_fails_closed(tmp_path: Path, failure: str) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    payload = _manifest()
    cache = _caches(approved, payload)
    path = Path(str(cache["14"]["path"]))
    if failure == "missing":
        path.unlink()
    elif failure == "tampered":
        path.write_bytes(path.read_bytes() + b"\n")
    elif failure == "bad_sha":
        cache["14"]["sha256"] = "not-a-sha256"
    else:
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        cache["14"]["sha256"] = _sha256(outside.read_bytes())
    with pytest.raises(ValueError, match="cache"):
        _build_request_plan(payload, cache, approved, ("14",))


def test_cache_identity_and_raw_provenance_come_from_payload(tmp_path: Path) -> None:
    payload = _manifest()
    narrative = str(payload["topics"][0]["narrative"])
    for mutation in (
        {"hits": 999},
        {"topic_id": "31"},
        {"variant_name": "facet"},
        {"query": narrative + " changed"},
    ):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        root.mkdir()
        cache = {
            "14": _write_cache(root, topic_id="14", narrative=narrative, mutate=mutation)
        }
        with pytest.raises(ValueError, match="cache"):
            _build_request_plan(payload, cache, root, ("14",))


def test_cache_rejects_self_consistent_but_unapproved_backend(tmp_path: Path) -> None:
    payload = _manifest()
    narrative = str(payload["topics"][0]["narrative"])
    binding = _write_cache(
        tmp_path, topic_id="14", narrative=narrative
    )
    path = Path(str(binding["path"]))
    cache_payload = json.loads(path.read_text())
    cache_payload.update(
        {
            "retriever_name": "unapproved",
            "index": "unapproved",
            "index_url": "https://example.invalid/search",
        }
    )
    cache_payload["response"]["index"] = "unapproved"
    request_identity = {
        "retriever_name": cache_payload["retriever_name"],
        "retriever_type": cache_payload["retriever_type"],
        "index": cache_payload["index"],
        "index_url": cache_payload["index_url"],
        "hits": cache_payload["hits"],
        "query_text": cache_payload["query"],
    }
    cache_payload["cache_key"] = _sha256(
        json.dumps(request_identity, sort_keys=True, separators=(",", ":")).encode()
    )[:16]
    raw = json.dumps(cache_payload, sort_keys=True).encode()
    path.write_bytes(raw)
    binding["sha256"] = _sha256(raw)

    with pytest.raises(ValueError, match="approved cache identity"):
        _build_request_plan(payload, {"14": binding}, tmp_path, ("14",))


def test_cache_rejects_unapproved_raw_response_api(tmp_path: Path) -> None:
    payload = _manifest()
    narrative = str(payload["topics"][0]["narrative"])
    binding = _write_cache(tmp_path, topic_id="14", narrative=narrative)
    path = Path(str(binding["path"]))
    cache_payload = json.loads(path.read_text())
    cache_payload["response"]["api"] = "v2"
    raw = json.dumps(cache_payload, sort_keys=True).encode()
    path.write_bytes(raw)
    binding["sha256"] = _sha256(raw)

    with pytest.raises(ValueError, match="raw response provenance"):
        _build_request_plan(payload, {"14": binding}, tmp_path, ("14",))


def test_public_request_plan_requires_exact_22(tmp_path: Path) -> None:
    payload = _manifest(ALL_TOPIC_IDS)
    cache = _caches(tmp_path, payload)
    plan = build_request_plan(payload, cache, approved_cache_root=tmp_path)
    assert plan["topic_ids"] == list(ALL_TOPIC_IDS)
    assert plan["original_cache_hit_count"] == 22


def test_authorization_preseal_is_immutable_before_source_loading(tmp_path: Path) -> None:
    payload = _manifest(ALL_TOPIC_IDS)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache = _caches(cache_root, payload)
    output = tmp_path / "planning"

    def source_loader() -> dict[str, object]:
        authorization = output / "authorization.json"
        preseal = output / "AUTHORIZATION_SEALED.json"
        assert authorization.is_file() and preseal.is_file()
        assert authorization.stat().st_mode & 0o222 == 0
        assert preseal.stat().st_mode & 0o222 == 0
        return payload

    freeze_planning(
        output,
        source_loader=source_loader,
        original_cache=cache,
        approved_cache_root=cache_root,
    )
    assert {path.name for path in output.iterdir()} == {
        "authorization.json",
        "AUTHORIZATION_SEALED.json",
        "manifest.json",
        "request_plan.json",
        "SEALED.json",
    }
    assert verify_planning(output, approved_cache_root=cache_root)["verified"] is True


def test_source_loader_authorization_mutation_aborts_freeze(tmp_path: Path) -> None:
    payload = _manifest(ALL_TOPIC_IDS)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache = _caches(cache_root, payload)
    output = tmp_path / "planning"

    def source_loader() -> dict[str, object]:
        path = output / "authorization.json"
        path.chmod(0o600)
        value = json.loads(path.read_text())
        value["qrels_permitted_during_freeze"] = True
        path.write_text(json.dumps(value))
        return payload

    with pytest.raises(ValueError, match="authorization pre-seal"):
        freeze_planning(
            output,
            source_loader=source_loader,
            original_cache=cache,
            approved_cache_root=cache_root,
        )


def test_verification_rejects_mutated_authorization_semantics_even_if_resealed(
    tmp_path: Path,
) -> None:
    payload = _manifest(ALL_TOPIC_IDS)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache = _caches(cache_root, payload)
    output = tmp_path / "planning"
    freeze_planning(
        output,
        source_loader=lambda: payload,
        original_cache=cache,
        approved_cache_root=cache_root,
    )

    authorization_path = output / "authorization.json"
    authorization_path.chmod(0o600)
    authorization = json.loads(authorization_path.read_text())
    authorization["retrieval_permitted_during_freeze"] = True
    authorization_path.write_text(json.dumps(authorization, sort_keys=True, indent=2) + "\n")
    authorization_path.chmod(0o400)

    preseal_path = output / "AUTHORIZATION_SEALED.json"
    preseal_path.chmod(0o600)
    preseal = json.loads(preseal_path.read_text())
    raw = authorization_path.read_bytes()
    preseal["authorization"] = {"bytes": len(raw), "sha256": _sha256(raw)}
    preseal_path.write_text(json.dumps(preseal, sort_keys=True, indent=2) + "\n")
    preseal_path.chmod(0o400)

    seal_path = output / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    for name in ("authorization.json", "AUTHORIZATION_SEALED.json"):
        raw = (output / name).read_bytes()
        seal["files"][name] = {"bytes": len(raw), "sha256": _sha256(raw)}
    seal["authorization_preseal_sha256"] = seal["files"][
        "AUTHORIZATION_SEALED.json"
    ]["sha256"]
    canonical = json.dumps(
        seal["files"], sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    seal["root_sha256"] = _sha256(canonical)
    seal_path.write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="authorization semantics"):
        verify_planning(output, approved_cache_root=cache_root)

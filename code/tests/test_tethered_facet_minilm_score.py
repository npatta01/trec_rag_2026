from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import trec_rag.tethered_facet_minilm_score as module
from trec_rag.facet_local_minilm_preflight import (
    ALLOW_PATTERNS,
    APPROVAL_SCHEMA_VERSION,
    APPROVAL_SCOPE,
    approval_allow_patterns_sha256,
    materialize_model,
)
from trec_rag.tethered_facet_minilm_score import (
    aggregate_document_scores,
    build_argument_parser,
    build_parser,
    build_tethered_candidates,
    create_preflight,
    enforce_preflight_ceiling,
    render_tethered_query,
    run_local_scoring,
)


def manifest() -> dict[str, object]:
    return {
        "topic_ids": ["219", "72", "300", "84"],
        "topics": [
            {"topic_id": "219", "query": "Full narrative 219"},
            {"topic_id": "72", "query": "Full narrative 72"},
            {"topic_id": "300", "query": "Full narrative 300"},
            {"topic_id": "84", "query": "Full narrative 84"},
        ],
        "facets": [
            {
                "topic_id": "219",
                "facet_id": "219-positive",
                "manifest_order": 0,
                "query": "one facet",
            },
            {
                "topic_id": "84",
                "facet_id": "84-safety",
                "manifest_order": 18,
                "query": "safety facet",
            },
        ],
    }


def phase1_rows() -> list[dict[str, object]]:
    return [
        {
            "topic_id": topic_id,
            "facet_id": facet_id,
            "manifest_order": order,
            "document_id": document_id,
            "docid": document_id,
            "rank": rank,
            "query": facet_query,
            "text": f"text {document_id}",
        }
        for topic_id, facet_id, order, facet_query, document_id, rank in (
            ("219", "219-positive", 0, "one facet", "d2", 2),
            ("84", "84-safety", 18, "safety facet", "d4", 2),
            ("219", "219-positive", 0, "one facet", "d1", 1),
            ("84", "84-safety", 18, "safety facet", "d3", 1),
        )
    ]


def accepted_gates() -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "facet_id": "219-positive",
            "manifest_order": 0,
            "status": "accepted",
        },
        {
            "topic_id": "84",
            "facet_id": "84-safety",
            "manifest_order": 18,
            "status": "accepted",
        },
        {
            "topic_id": "72",
            "facet_id": "72-rejected",
            "manifest_order": 7,
            "status": "rejected",
        },
    ]


def production_manifest() -> dict[str, object]:
    topics = [dict(row) for row in manifest()["topics"]]
    facets: list[dict[str, object]] = []
    order = 0
    for topic_id in module.TOPIC_IDS:
        for number in range(6):
            facets.append(
                {
                    "topic_id": topic_id,
                    "facet_id": f"{topic_id}-facet-{number}",
                    "manifest_order": order,
                    "query": f"facet query {topic_id} {number}",
                }
            )
            order += 1
    facets.append(
        {
            "topic_id": "84",
            "facet_id": "84-rejected",
            "manifest_order": order,
            "query": "rejected facet query",
        }
    )
    payload: dict[str, object] = {
        "schema_version": "rag25_deep_facet_candidate_manifest_v1",
        "experiment_id": "rag25_deep_facet_candidates_v1",
        "topic_ids": list(module.TOPIC_IDS),
        "topics": topics,
        "facets": facets,
        "qrels_opened": False,
    }
    unhashed = dict(payload)
    payload["hashes"] = {
        "topics_sha256": _sha256(_compact(topics) + b"\n"),
        "facets_sha256": _sha256(_compact(facets) + b"\n"),
        "freeze_sha256": _sha256(_compact(unhashed) + b"\n"),
    }
    return payload


def production_gates() -> list[dict[str, object]]:
    return [
        {
            "topic_id": facet["topic_id"],
            "facet_id": facet["facet_id"],
            "manifest_order": facet["manifest_order"],
            "accepted": facet["facet_id"] != "84-rejected",
            "status": (
                "rejected" if facet["facet_id"] == "84-rejected" else "accepted"
            ),
        }
        for facet in production_manifest()["facets"]
    ]


def production_phase1_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for facet in production_manifest()["facets"]:
        for rank in range(1, 201):
            document_id = f"{facet['facet_id']}-d{rank}"
            query = str(facet["query"])
            text = f"text {document_id}"
            rows.append(
                {
                    "topic_id": facet["topic_id"],
                    "facet_id": facet["facet_id"],
                    "manifest_order": facet["manifest_order"],
                    "document_id": document_id,
                    "docid": document_id,
                    "rank": rank,
                    "query": query,
                    "query_sha256": _sha256(query.encode()),
                    "text": text,
                    "text_sha256": _sha256(text.encode()),
                }
            )
    return rows


ACCEPTED_UNION_COUNTS = {"219": 2182, "72": 2127, "300": 1712, "84": 2093}


def production_accepted_union_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_topic: dict[str, list[str]] = {topic_id: [] for topic_id in module.TOPIC_IDS}
    for candidate in production_phase1_rows():
        if candidate["facet_id"] != "84-rejected":
            by_topic[str(candidate["topic_id"])].append(str(candidate["document_id"]))
    for topic_id in module.TOPIC_IDS:
        docids = by_topic[topic_id]
        for number in range(ACCEPTED_UNION_COUNTS[topic_id] - len(docids)):
            docids.append(f"{topic_id}-union-extra-{number}")
        for order, document_id in enumerate(docids, start=1):
            rows.append(
                {
                    "schema_version": "deep-facet-candidate-union-v1",
                    "union": "accepted",
                    "topic_id": topic_id,
                    "document_id": document_id,
                    "union_order": order,
                }
            )
    return rows


def _compact(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class WordTokenizer:
    def encode(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        return text.split()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(token_ids)

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


class FakeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scores: dict[str, float] = {}
        self.aliases: dict[tuple[str, str], str] = {}

    def cache_key(self, *, query_text: str, text: str) -> str:
        return self.aliases.get(
            (query_text, text), _sha256(_compact([query_text, text]))
        )

    def get(self, *, query_text: str, text: str):
        return self.scores.get(self.cache_key(query_text=query_text, text=text))

    def add_many(self, rows):
        for query_text, text, score in rows:
            self.scores[self.cache_key(query_text=query_text, text=text)] = score


class FakeRunner:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.forwarded_keys: list[str] = []

    def score(self, rows):
        self.forwarded_keys.extend(row["cache_key"] for row in rows)
        return [self.scores[row["cache_key"]] for row in rows]


def scoring_fixture(tmp_path: Path) -> tuple[Path, FakeCache]:
    output = output_path(tmp_path)
    output.mkdir(parents=True)
    cache = FakeCache(tmp_path / "score-cache.jsonl")
    rows = []
    candidates = []
    coverage = {}
    accepted_facets_by_topic = {"219": 7, "72": 7, "300": 4, "84": 6}
    accepted_facets = [
        (topic_id, facet_number)
        for topic_id in module.TOPIC_IDS
        for facet_number in range(accepted_facets_by_topic[topic_id])
    ]
    number = 0
    for topic_id, facet_number in accepted_facets:
        facet_id = f"{topic_id}-facet-{facet_number}"
        for rank in range(1, 201):
            number += 1
            key_number = 1 if number % 2 else 2
            query, text = f"q{key_number}", f"t{key_number}"
            cache_key = f"k{key_number}"
            cache.aliases[(query, text)] = cache_key
            rows.append(
                {
                    "topic_id": topic_id,
                    "facet_id": facet_id,
                    "document_id": f"d{number}",
                    "rank": rank,
                    "query": query,
                    "query_sha256": _sha256(query.encode()),
                    "document_sha256": _sha256(text.encode()),
                    "window_id": f"w{number}",
                    "window_text": text,
                    "window_sha256": _sha256(text.encode()),
                    "cache_key": cache_key,
                    "document_start_token": 0,
                    "document_end_token": 256,
                    "cache_hit": key_number == 1,
                }
            )
            coverage[f"{facet_id}:d{number}"] = [1.0]
            candidates.append(
                {
                    "schema_version": module.CANDIDATE_SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "facet_id": facet_id,
                    "document_id": f"d{number}",
                    "rank": rank,
                    "query": query,
                    "query_sha256": _sha256(query.encode()),
                    "text": text,
                    "text_sha256": _sha256(text.encode()),
                }
            )
    window_bytes = b"".join(_compact(row) + b"\n" for row in rows)
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidates)
    (output / "windows.jsonl").write_bytes(window_bytes)
    (output / "candidates.jsonl").write_bytes(candidate_bytes)
    preflight = {
        "schema_version": module.PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model_constructed": False,
        "model": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
        "model_materialization_receipt": str(tmp_path / "materialization.json"),
        "model_materialization_receipt_sha256": "a" * 64,
        "candidates_file": "candidates.jsonl",
        "windows_file": "windows.jsonl",
        "windows_sha256": _sha256(window_bytes),
        "candidates_sha256": _sha256(candidate_bytes),
        "device_probe": working_rocm_probe(),
        "runtime_evidence": {
            "policy": "prior_compatible_rocm_rate_plus_fixed_seconds_v1",
            "prior_scoring_receipt_sha256": "b" * 64,
            "prior_unique_forward_pair_count": 100,
            "prior_elapsed_seconds": 1.0,
            "policy_pairs_per_second": 100.0,
            "prior_pairs_per_second": 100.0,
            "prior_observed_pairs_per_second": 100.0,
            "fixed_seconds": module.REFERENCE_FIXED_SECONDS,
            "projected_unique_cache_miss_count": 1,
            "projected_inference_seconds": 30.01,
            "runtime_ceiling_seconds": module.RUNTIME_CEILING_SECONDS,
            "prior_peak_device_memory_bytes": 1,
            "prior_peak_host_memory_bytes": 1,
            "tokenizer_planning_elapsed_seconds": 0.1,
        },
        "tokenizer": {"class": "FakeTokenizer", "local_files_only": True},
        "score_cache": {
            "context": module.score_cache_context().artifact_metadata,
            "binding": {
                "state": "absent",
                "path": str(cache.path.resolve()),
                "bytes": 0,
                "sha256": None,
            },
        },
        "sources": {
            name: {"path": str(tmp_path / name), "sha256": "c" * 64}
            for name in (
                "manifest",
                "phase1_scoring_receipt",
                "phase1_preflight_receipt",
                "phase1_candidates",
                "phase1_windows",
                "phase1_scores",
                "gate_summary",
                "gates",
                "model_materialization_receipt",
            )
        },
        "summary": {
            "query_document_pair_count": module.PAIR_COUNT,
            "window_count": module.PAIR_COUNT,
            "unique_pair_count": 2,
            "unique_cache_miss_count": 1,
            "accepted_facet_count": module.ACCEPTED_FACET_COUNT,
            "cache_hit_window_count": module.PAIR_COUNT // 2,
            "cache_miss_window_count": module.PAIR_COUNT // 2,
            "topic_pair_counts": {
                topic: accepted_facets_by_topic[topic] * 200
                for topic in module.TOPIC_IDS
            },
            "facet_pair_counts": {
                f"{topic}-facet-{number}": 200
                for topic in module.TOPIC_IDS
                for number in range(accepted_facets_by_topic[topic])
            },
            "topic_window_counts": {
                topic: accepted_facets_by_topic[topic] * 200
                for topic in module.TOPIC_IDS
            },
            "facet_window_counts": {
                f"{topic}-facet-{number}": 200
                for topic in module.TOPIC_IDS
                for number in range(accepted_facets_by_topic[topic])
            },
            "document_window_coverage": coverage,
        },
        "ceilings": {
            "exact_query_document_pair_count": module.PAIR_COUNT,
            "maximum_window_count": module.WINDOW_CEILING,
            "maximum_runtime_seconds": module.RUNTIME_CEILING_SECONDS,
        },
    }
    preflight["sources"]["accepted_union"] = {
        "path": str(tmp_path / "accepted_union.jsonl"),
        "bytes": 1,
        "sha256": "d" * 64,
        "rows": module.ACCEPTED_UNION_COUNT,
        "topic_counts": module.ACCEPTED_UNION_COUNTS,
    }
    preflight["sources"]["model_materialization_receipt"] = {
        "path": preflight["model_materialization_receipt"],
        "sha256": preflight["model_materialization_receipt_sha256"],
    }
    preflight["sources"]["phase1_scoring_receipt"]["sha256"] = (
        preflight["runtime_evidence"]["prior_scoring_receipt_sha256"]
    )
    manifest_facets = [
        {
            "topic_id": topic_id,
            "facet_id": f"{topic_id}-facet-{facet_number}",
            "manifest_order": order,
            "query": f"facet query {topic_id} {facet_number}",
        }
        for order, (topic_id, facet_number) in enumerate(accepted_facets)
    ]
    manifest_facets.append(
        {
            "topic_id": "84",
            "facet_id": "84-rejected",
            "manifest_order": len(manifest_facets),
            "query": "rejected facet query",
        }
    )
    manifest_payload = {
        "schema_version": "rag25_deep_facet_candidate_manifest_v1",
        "experiment_id": "rag25_deep_facet_candidates_v1",
        "topic_ids": list(module.TOPIC_IDS),
        "topics": [
            {"topic_id": topic_id, "query": f"Full narrative {topic_id}"}
            for topic_id in module.TOPIC_IDS
        ],
        "facets": manifest_facets,
        "qrels_opened": False,
    }
    unhashed_manifest = dict(manifest_payload)
    manifest_payload["hashes"] = {
        "topics_sha256": _sha256(_compact(manifest_payload["topics"]) + b"\n"),
        "facets_sha256": _sha256(_compact(manifest_facets) + b"\n"),
        "freeze_sha256": _sha256(_compact(unhashed_manifest) + b"\n"),
    }
    manifest_source = _pretty(manifest_payload)
    manifest_path = tmp_path / "manifest"
    manifest_path.write_bytes(manifest_source)
    gates_source = _pretty(
        {
            "schema_version": "deep-facet-candidate-gate-v1",
            "gates": [
                {
                    "topic_id": facet["topic_id"],
                    "facet_id": facet["facet_id"],
                    "manifest_order": facet["manifest_order"],
                    "status": (
                        "rejected"
                        if facet["facet_id"] == "84-rejected"
                        else "accepted"
                    ),
                }
                for facet in manifest_facets
            ],
        }
    )
    gates_path = tmp_path / "gates"
    gates_path.write_bytes(gates_source)
    preflight["sources"]["manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_source),
    }
    preflight["sources"]["gates"] = {
        "path": str(gates_path),
        "sha256": _sha256(gates_source),
    }
    (output / "preflight.json").write_bytes(_pretty(preflight))
    cache.scores["k1"] = 1.0
    return output / "preflight.json", cache


def test_verify_derives_uneven_topic_pairs_from_authenticated_facets(tmp_path):
    preflight_path, _cache = scoring_fixture(tmp_path)

    verified = module.verify_preflight(preflight_path)

    assert verified["summary"]["topic_pair_counts"] == {
        "219": 1400,
        "72": 1400,
        "300": 800,
        "84": 1200,
    }


def working_rocm_probe() -> dict[str, object]:
    return {
        "available": True,
        "execution_backend": "rocm",
        "device": "cuda",
        "device_count": 1,
        "device_name": "test AMD GPU",
        "hip_version": "test-rocm",
        "probe_allocation_bytes": 4,
        "peak_device_memory_bytes": 4,
    }


def output_path(tmp_path: Path) -> Path:
    return tmp_path / "post_qrels_tethered_facet_minilm_v1" / "scoring"


def restamp_accepted_union(sources: dict[str, Path]) -> None:
    union_path = sources["gate"] / "u_accepted.jsonl"
    summary_path = sources["gate"] / "summary.json"
    source = union_path.read_bytes()
    summary = json.loads(summary_path.read_bytes())
    summary["artifacts"]["u_accepted.jsonl"] = {
        "bytes": len(source),
        "sha256": _sha256(source),
    }
    summary_path.write_bytes(_pretty(summary))


def fixture_sources(tmp_path: Path) -> dict[str, Path]:
    manifest_path = tmp_path / "manifest.json"
    phase1 = tmp_path / "phase1_v1"
    gate = tmp_path / "gate_v1"
    phase1.mkdir()
    gate.mkdir()

    manifest_bytes = _pretty(production_manifest())
    manifest_path.write_bytes(manifest_bytes)
    candidate_bytes = b"".join(
        _compact(row) + b"\n" for row in production_phase1_rows()
    )
    (phase1 / "candidates.jsonl").write_bytes(candidate_bytes)
    approval = tmp_path / "approval.json"
    approval.write_bytes(
        _pretty(
            {
                "schema_version": APPROVAL_SCHEMA_VERSION,
                "approval_scope": APPROVAL_SCOPE,
                "approved_by": "unit test",
                "model_id": module.MODEL_ID,
                "revision": module.MODEL_REVISION,
                "allow_patterns": list(ALLOW_PATTERNS),
                "allow_patterns_sha256": approval_allow_patterns_sha256(),
                "acknowledged_network_download": True,
                "acknowledged_safe_files_only": True,
                "acknowledged_no_model_or_tokenizer_construction": True,
                "acknowledged_no_inference_qrels_retrieval_or_paid_calls": True,
            }
        )
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for number, filename in enumerate(ALLOW_PATTERNS, start=1):
        (snapshot / filename).write_text(f"safe file {number}\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    materialize_model(
        model_id=module.MODEL_ID,
        revision=module.MODEL_REVISION,
        approval_path=approval,
        output_dir=model_dir,
        snapshot_download_fn=lambda **_kwargs: str(snapshot),
    )
    model_receipt = model_dir / "materialization.json"
    model_receipt_sha256 = _sha256(model_receipt.read_bytes())
    phase1_preflight_bytes = _pretty(
        {
            "schema_version": "deep-facet-candidate-minilm-preflight-v1",
            "status": "tokenizer_only_preflight_complete",
            "phase": "phase1_facet_local",
            "qrels_opened": False,
            "network_access_supported": False,
            "hosted_inference_supported": False,
            "model_constructed": False,
            "candidates_sha256": _sha256(candidate_bytes),
            "model_materialization_receipt": str(model_receipt),
            "model_materialization_receipt_sha256": model_receipt_sha256,
            "model": module.MODEL_ID,
            "model_revision": module.MODEL_REVISION,
            "pairs_per_second": 100.0,
            "fixed_seconds": 30.0,
            "projected_runtime_seconds": 40.0,
            "runtime_ceiling_seconds": 600.0,
            "summary": {
                "document_count": 5000,
                "window_count": 5000,
                "unique_pair_count": 5000,
                "unique_uncached_pair_count": 1000,
                "cache_hit_window_count": 4000,
            },
        }
    )
    (phase1 / "preflight.json").write_bytes(phase1_preflight_bytes)
    phase1_windows = b"".join(
        _compact({"window": number}) + b"\n" for number in range(5000)
    )
    phase1_scores = b"".join(
        _compact({"score": number}) + b"\n" for number in range(5000)
    )
    phase1_preflight = json.loads(phase1_preflight_bytes)
    phase1_preflight["windows_sha256"] = _sha256(phase1_windows)
    phase1_preflight_bytes = _pretty(phase1_preflight)
    (phase1 / "preflight.json").write_bytes(phase1_preflight_bytes)
    (phase1 / "windows.jsonl").write_bytes(phase1_windows)
    (phase1 / "scores.jsonl").write_bytes(phase1_scores)
    (phase1 / "scoring_receipt.json").write_bytes(
        _pretty(
            {
                "schema_version": "deep-facet-candidate-minilm-receipt-v1",
                "status": "complete",
                "phase": "phase1_facet_local",
                "qrels_opened": False,
                "network_access_supported": False,
                "hosted_inference_supported": False,
                "preflight_sha256": _sha256(phase1_preflight_bytes),
                "windows_sha256": _sha256(phase1_windows),
                "scores_sha256": _sha256(phase1_scores),
                "model": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
                "planned_window_count": 5000,
                "completed_window_count": 5000,
                "unique_forward_pair_count": 1000,
                "cache_reuse_pair_count": 4000,
                "elapsed_seconds": 10.0,
                "peak_device_memory_bytes": 123456,
                "peak_host_memory_bytes": 654321,
                "device": "cuda",
                "execution_backend": "rocm",
            }
        )
    )
    gates_bytes = _pretty(
        {
            "schema_version": "deep-facet-candidate-gate-v1",
            "gates": production_gates(),
        }
    )
    (gate / "gates.json").write_bytes(gates_bytes)
    accepted_union_bytes = b"".join(
        _compact(row) + b"\n" for row in production_accepted_union_rows()
    )
    (gate / "u_accepted.jsonl").write_bytes(accepted_union_bytes)
    (gate / "summary.json").write_bytes(
        _pretty(
            {
                "schema_version": "deep-facet-candidate-gate-v1",
                "status": "complete",
                "qrels_opened": False,
                "facet_count": 25,
                "accepted_facet_count": 24,
                "rejected_facet_count": 1,
                "topic_counts": {
                    topic_id: {
                        "accepted_facets": 6,
                        "accepted_union": ACCEPTED_UNION_COUNTS[topic_id],
                        "raw_union": ACCEPTED_UNION_COUNTS[topic_id],
                    }
                    for topic_id in module.TOPIC_IDS
                },
                "artifacts": {
                    "gates.json": {
                        "bytes": len(gates_bytes),
                        "sha256": _sha256(gates_bytes),
                    },
                    "u_accepted.jsonl": {
                        "bytes": len(accepted_union_bytes),
                        "sha256": _sha256(accepted_union_bytes),
                    },
                },
            }
        )
    )
    return {
        "manifest": manifest_path,
        "phase1": phase1,
        "gate": gate,
        "model_receipt": model_receipt,
    }


def test_render_tethered_query_is_exact_and_rejects_blank_parts():
    assert render_tethered_query("Full narrative", "one facet") == (
        "Full narrative\n\nFocus: one facet"
    )
    with pytest.raises(ValueError, match="narrative"):
        render_tethered_query(" ", "one facet")
    with pytest.raises(ValueError, match="facet query"):
        render_tethered_query("Full narrative", " ")


def test_candidates_cover_only_accepted_facets_and_preserve_identity():
    rows = build_tethered_candidates(manifest(), phase1_rows(), accepted_gates())
    assert len(rows) == 4
    assert {
        (row["topic_id"], row["facet_id"], row["document_id"]) for row in rows
    } == {
        ("219", "219-positive", "d1"),
        ("219", "219-positive", "d2"),
        ("84", "84-safety", "d3"),
        ("84", "84-safety", "d4"),
    }
    assert all(
        row["query"].endswith(f"Focus: {row['facet_query']}") for row in rows
    )
    assert [(row["facet_id"], row["document_id"]) for row in rows] == [
        ("219-positive", "d1"),
        ("219-positive", "d2"),
        ("84-safety", "d3"),
        ("84-safety", "d4"),
    ]


def test_preflight_is_tokenizer_only_and_freezes_exact_cache_state(tmp_path):
    cache = FakeCache(tmp_path / "cache.jsonl")
    output = output_path(tmp_path)
    result = create_preflight(
        fixture_sources(tmp_path),
        output,
        tokenizer=WordTokenizer(),
        cache=cache,
        device_probe=working_rocm_probe,
    )
    assert result["status"] == "tokenizer_only_preflight_complete"
    assert result["model_constructed"] is False
    assert result["network_access_supported"] is False
    assert result["summary"]["query_document_pair_count"] == 4800
    assert result["summary"]["accepted_facet_count"] == 24
    assert result["summary"]["cache_hit_window_count"] == 0
    assert result["summary"]["cache_miss_window_count"] == 4800
    assert result["runtime_evidence"]["prior_pairs_per_second"] == 100.0
    assert result["runtime_evidence"]["fixed_seconds"] == 30.0
    assert result["runtime_evidence"]["projected_inference_seconds"] == 78.0
    assert result["device_probe"] == working_rocm_probe()
    assert result["runtime_evidence"]["prior_peak_device_memory_bytes"] == 123456
    assert result["runtime_evidence"]["prior_peak_host_memory_bytes"] == 654321
    assert (output / "candidates.jsonl").exists()
    assert (output / "windows.jsonl").exists()
    assert (output / "preflight.json").read_bytes() == _pretty(result)


@pytest.mark.parametrize(
    "pairs,windows,seconds",
    [(4799, 20, 1), (4800, 25001, 1), (4800, 20, 601)],
)
def test_preflight_stops_on_exact_population_or_resource_drift(
    pairs, windows, seconds
):
    with pytest.raises(ValueError, match="preflight ceiling"):
        enforce_preflight_ceiling(pairs, windows, seconds)


def test_preflight_cli_has_only_local_authenticated_inputs():
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "preflight",
            "--manifest",
            "manifest.json",
            "--phase1",
            "phase1_v1",
            "--gate",
            "gate_v1",
            "--output",
            "post_qrels_tethered_facet_minilm_v1/scoring",
        ]
    )
    assert args.command == "preflight"
    forbidden = {"qrels", "retrieval", "network", "model_download", "hosted_inference"}
    assert forbidden.isdisjoint(vars(args))


def test_candidate_join_rejects_protected_topics_before_indexing():
    bad_rows = phase1_rows()
    bad_rows[0] = {**bad_rows[0], "topic_id": "144"}
    with pytest.raises(ValueError, match="protected topic 144"):
        build_tethered_candidates(manifest(), bad_rows, accepted_gates())


def test_preflight_authenticates_phase1_receipt_hashes_before_candidates(tmp_path):
    sources = fixture_sources(tmp_path)
    (sources["phase1"] / "preflight.json").write_bytes(_pretty({"tampered": True}))
    with pytest.raises(ValueError, match="phase-1 preflight receipt hash"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_candidate_join_rejects_stale_query_and_text_hashes():
    stale_query = phase1_rows()
    stale_query[0] = {**stale_query[0], "query_sha256": "0" * 64}
    with pytest.raises(ValueError, match="query hash"):
        build_tethered_candidates(manifest(), stale_query, accepted_gates())

    stale_text = phase1_rows()
    stale_text[0] = {**stale_text[0], "text_sha256": "0" * 64}
    with pytest.raises(ValueError, match="text hash"):
        build_tethered_candidates(manifest(), stale_text, accepted_gates())


def test_default_loader_uses_explicit_tokenizer_only_backend(tmp_path, monkeypatch):
    sources = fixture_sources(tmp_path)
    captured: dict[str, object] = {}

    def load_tokenizer(path, *, auto_tokenizer_cls=None):
        captured["path"] = path
        captured["auto_tokenizer_cls"] = auto_tokenizer_cls
        return WordTokenizer()

    monkeypatch.setattr(module, "load_verified_tokenizer", load_tokenizer)
    create_preflight(
        sources,
        output_path(tmp_path),
        cache=FakeCache(tmp_path / "cache.jsonl"),
        device_probe=working_rocm_probe,
    )

    assert captured["auto_tokenizer_cls"] is module.TokenizerOnlyAuto


def test_preflight_invokes_frozen_ceiling_gate(tmp_path, monkeypatch):
    calls: list[tuple[int, int, float]] = []
    original = module.enforce_preflight_ceiling

    def enforce(pairs, windows, seconds):
        calls.append((pairs, windows, seconds))
        original(pairs, windows, seconds)

    monkeypatch.setattr(module, "enforce_preflight_ceiling", enforce)
    create_preflight(
        fixture_sources(tmp_path),
        output_path(tmp_path),
        tokenizer=WordTokenizer(),
        cache=FakeCache(tmp_path / "cache.jsonl"),
        device_probe=working_rocm_probe,
    )

    assert calls == [(4800, 4800, 78.0)]


def test_preflight_rejects_slow_prior_rocm_projection(tmp_path):
    sources = fixture_sources(tmp_path)
    receipt_path = sources["phase1"] / "scoring_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    preflight_path = sources["phase1"] / "preflight.json"
    preflight = json.loads(preflight_path.read_bytes())
    preflight["pairs_per_second"] = 1.0
    preflight["projected_runtime_seconds"] = 1030.0
    preflight_path.write_bytes(_pretty(preflight))
    receipt["preflight_sha256"] = _sha256(preflight_path.read_bytes())
    receipt_path.write_bytes(_pretty(receipt))

    with pytest.raises(ValueError, match="preflight ceiling"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_uses_slower_authenticated_observed_rate(tmp_path):
    sources = fixture_sources(tmp_path)
    preflight_path = sources["phase1"] / "preflight.json"
    preflight = json.loads(preflight_path.read_bytes())
    preflight["pairs_per_second"] = 300.0
    preflight["projected_runtime_seconds"] = 30.0 + 1000 / 300.0
    preflight_path.write_bytes(_pretty(preflight))
    scoring_path = sources["phase1"] / "scoring_receipt.json"
    scoring = json.loads(scoring_path.read_bytes())
    scoring["preflight_sha256"] = _sha256(preflight_path.read_bytes())
    scoring_path.write_bytes(_pretty(scoring))

    result = create_preflight(
        sources,
        output_path(tmp_path),
        tokenizer=WordTokenizer(),
        cache=FakeCache(tmp_path / "cache.jsonl"),
        device_probe=working_rocm_probe,
    )

    assert result["runtime_evidence"]["policy_pairs_per_second"] == 300.0
    assert result["runtime_evidence"]["prior_observed_pairs_per_second"] == 100.0
    assert result["runtime_evidence"]["prior_pairs_per_second"] == 100.0
    assert result["runtime_evidence"]["projected_inference_seconds"] == 78.0


def test_preflight_requires_working_rocm_without_constructing_model(tmp_path):
    with pytest.raises(ValueError, match="working ROCm device"):
        create_preflight(
            fixture_sources(tmp_path),
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=lambda: {
                **working_rocm_probe(),
                "available": False,
                "device_count": 0,
            },
        )
    assert not output_path(tmp_path).exists()


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("manifest", "hashes"),
        ("phase1_preflight", "candidates_sha256"),
        ("phase1_preflight", "pairs_per_second"),
        ("phase1_preflight", "fixed_seconds"),
        ("phase1_preflight", "runtime_ceiling_seconds"),
        ("phase1_scoring", "preflight_sha256"),
        ("gate_summary", "accepted_facet_count"),
        ("gate_summary", "topic_counts"),
    ],
)
def test_preflight_rejects_missing_controlling_bindings_before_candidates(
    tmp_path, artifact, field
):
    sources = fixture_sources(tmp_path)
    paths = {
        "manifest": sources["manifest"],
        "phase1_preflight": sources["phase1"] / "preflight.json",
        "phase1_scoring": sources["phase1"] / "scoring_receipt.json",
        "gate_summary": sources["gate"] / "summary.json",
    }
    path = paths[artifact]
    payload = json.loads(path.read_bytes())
    payload.pop(field)
    path.write_bytes(_pretty(payload))
    if artifact == "phase1_preflight":
        scoring_path = sources["phase1"] / "scoring_receipt.json"
        scoring = json.loads(scoring_path.read_bytes())
        scoring["preflight_sha256"] = _sha256(path.read_bytes())
        scoring_path.write_bytes(_pretty(scoring))

    with pytest.raises(ValueError, match="required"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_rejects_noncanonical_publication_path(tmp_path):
    with pytest.raises(ValueError, match="post_qrels_tethered_facet_minilm_v1/scoring"):
        create_preflight(
            fixture_sources(tmp_path),
            tmp_path / "scoring",
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_rejects_accepted_union_hash_mismatch(tmp_path):
    sources = fixture_sources(tmp_path)
    with (sources["gate"] / "u_accepted.jsonl").open("ab") as sink:
        sink.write(b'{}\n')
    with pytest.raises(ValueError, match="accepted union.*hash"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_rejects_mismatched_accepted_union_topic_count(tmp_path):
    sources = fixture_sources(tmp_path)
    summary_path = sources["gate"] / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary["topic_counts"]["219"]["accepted_union"] = 2181
    summary_path.write_bytes(_pretty(summary))

    with pytest.raises(ValueError, match="accepted union.*counts"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_rejects_facet_candidate_absent_from_accepted_union(tmp_path):
    sources = fixture_sources(tmp_path)
    union_path = sources["gate"] / "u_accepted.jsonl"
    rows = union_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["document_id"] = "219-replacement-not-a-facet-candidate"
    rows[0] = _compact(first).decode()
    union_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    restamp_accepted_union(sources)

    with pytest.raises(ValueError, match="absent from accepted union"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_preflight_rejects_protected_accepted_union_topic_before_indexing(tmp_path):
    sources = fixture_sources(tmp_path)
    union_path = sources["gate"] / "u_accepted.jsonl"
    rows = union_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["topic_id"] = "144"
    rows[0] = _compact(first).decode()
    union_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    restamp_accepted_union(sources)

    with pytest.raises(ValueError, match="protected topic 144"):
        create_preflight(
            sources,
            output_path(tmp_path),
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
            device_probe=working_rocm_probe,
        )


def test_rocm_probe_uses_disk_backed_temp_without_model_construction(monkeypatch):
    calls: list[dict[str, object]] = []

    class Completed:
        stdout = json.dumps(working_rocm_probe())

    def run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", run)

    assert module.probe_working_rocm_device() == working_rocm_probe()
    assert calls[0]["kwargs"]["env"]["TMPDIR"] == "/var/tmp"
    assert "torch.zeros" in calls[0]["args"][0][2]
    assert "AutoModel" not in calls[0]["args"][0][2]


def test_scoring_only_forwards_exact_cache_misses(tmp_path, monkeypatch):
    preflight_path, cache = scoring_fixture(tmp_path)
    runner = FakeRunner(scores={"k2": 2.0})
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )

    receipt = run_local_scoring(preflight_path, runner=runner)

    assert runner.forwarded_keys == ["k2"]
    assert receipt["unique_forward_pair_count"] == 1
    assert receipt["cache_hit_count"] == 1


def test_document_aggregation_matches_existing_top4_contract():
    rows = aggregate_document_scores(
        [
            {
                "topic_id": "219",
                "facet_id": "219-positive",
                "document_id": "d1",
                "query_sha256": "1" * 64,
                "document_sha256": "2" * 64,
                "window_id": "overlap",
                "window_sha256": "3" * 64,
                "document_start_token": 0,
                "document_end_token": 256,
                "score": 8.0,
                "model": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
            },
            {
                "topic_id": "219",
                "facet_id": "219-positive",
                "document_id": "d1",
                "query_sha256": "1" * 64,
                "document_sha256": "2" * 64,
                "window_id": "discarded-overlap",
                "window_sha256": "4" * 64,
                "document_start_token": 64,
                "document_end_token": 320,
                "score": 7.0,
                "model": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
            },
            {
                "topic_id": "219",
                "facet_id": "219-positive",
                "document_id": "d1",
                "query_sha256": "1" * 64,
                "document_sha256": "2" * 64,
                "window_id": "distinct",
                "window_sha256": "5" * 64,
                "document_start_token": 256,
                "document_end_token": 512,
                "score": 5.92,
                "model": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
            },
        ]
    )

    assert rows == [
        {
            "topic_id": "219",
            "facet_id": "219-positive",
            "document_id": "d1",
            "score": pytest.approx(7.35),
            "selected_window_count": 2,
            "query_sha256": "1" * 64,
            "text_sha256": "2" * 64,
            "model": module.MODEL_ID,
            "model_revision": module.MODEL_REVISION,
            "window_hashes": ["3" * 64, "5" * 64],
        }
    ]


def test_score_cli_has_no_download_network_retrieval_or_qrels_argument():
    parser = build_parser()
    parsers = [parser]
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            parsers.extend(choices.values())
    help_text = "\n".join(candidate.format_help() for candidate in parsers)
    for forbidden in ("qrels", "endpoint", "download", "retrieval"):
        assert forbidden not in help_text.lower()
    assert parser.parse_args(["score", "--preflight", "preflight.json"]).command == "score"
    assert parser.parse_args(["verify", "--preflight", "preflight.json"]).command == "verify"


def test_protected_topic_fails_before_cache_or_model(monkeypatch):
    class ExplodingDependency:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dependency must not be constructed")

    monkeypatch.setattr(module, "GlobalScoreCache", ExplodingDependency)
    preflight = {
        "schema_version": module.PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
        "topic_ids": ["144"],
        "summary": {},
    }
    with pytest.raises(ValueError, match="protected topic 144"):
        module.verify_preflight(preflight)


def test_verify_recomputes_document_aggregation_after_receipt_restamp(
    tmp_path, monkeypatch
):
    preflight_path, cache = scoring_fixture(tmp_path)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )
    run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))
    assert module.verify_scoring(preflight_path)["status"] == "complete"
    document_path = preflight_path.parent / "document_scores.jsonl"
    lines = document_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["score"] = 99.0
    lines[0] = _compact(first).decode()
    document_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt_path = preflight_path.parent / "scoring_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["document_scores_sha256"] = _sha256(document_path.read_bytes())
    receipt_path.write_bytes(_pretty(receipt))

    with pytest.raises(ValueError, match="document|aggregation"):
        module.verify_scoring(preflight_path)


def test_scoring_finalization_cannot_cross_hard_ceiling(tmp_path, monkeypatch):
    preflight_path, cache = scoring_fixture(tmp_path)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )
    ticks = iter((0.0, 1.0, 2.0, 601.0))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(RuntimeError, match="600-second ceiling"):
        run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))
    assert not (preflight_path.parent / "scoring_receipt.json").exists()


def test_preflight_rejects_nonlocal_artifact_filenames(tmp_path):
    preflight_path, _cache = scoring_fixture(tmp_path)
    preflight = json.loads(preflight_path.read_bytes())
    preflight["windows_file"] = "../windows.jsonl"

    with pytest.raises(ValueError, match="artifact filenames"):
        module.verify_preflight(preflight)


def test_verify_binds_full_score_payload_to_preflight_windows(tmp_path, monkeypatch):
    preflight_path, cache = scoring_fixture(tmp_path)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )
    run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))
    output = preflight_path.parent
    windows = [json.loads(line) for line in (output / "windows.jsonl").read_bytes().splitlines()]
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_bytes().splitlines()]
    windows[0]["document_end_token"] += 1
    scores[0]["document_end_token"] += 1
    (output / "windows.jsonl").write_bytes(b"".join(_compact(row) + b"\n" for row in windows))
    score_bytes = b"".join(_compact(row) + b"\n" for row in scores)
    (output / "scores.jsonl").write_bytes(score_bytes)
    documents = [
        {"schema_version": module.DOCUMENT_SCORE_SCHEMA_VERSION, **row}
        for row in module.aggregate_document_scores(scores)
    ]
    document_bytes = b"".join(_compact(row) + b"\n" for row in documents)
    (output / "document_scores.jsonl").write_bytes(document_bytes)
    receipt_path = output / "scoring_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["scores_sha256"] = _sha256(score_bytes)
    receipt["document_scores_sha256"] = _sha256(document_bytes)
    receipt_path.write_bytes(_pretty(receipt))

    with pytest.raises(ValueError, match="planned windows|planned window"):
        module.verify_scoring(preflight_path)


def test_scoring_requires_candidate_and_window_identity_sets_to_match(
    tmp_path, monkeypatch
):
    preflight_path, cache = scoring_fixture(tmp_path)
    output = preflight_path.parent
    candidates = [
        json.loads(line) for line in (output / "candidates.jsonl").read_bytes().splitlines()
    ]
    candidates[-1]["document_id"] = "different-document"
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidates)
    (output / "candidates.jsonl").write_bytes(candidate_bytes)
    preflight = json.loads(preflight_path.read_bytes())
    preflight["candidates_sha256"] = _sha256(candidate_bytes)
    preflight_path.write_bytes(_pretty(preflight))
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )

    with pytest.raises(ValueError, match="candidate.*window identities"):
        run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("retrieval_path_supported", True),
        ("retrieval_path_supported", None),
        ("network_access_supported", True),
        ("hosted_inference_supported", True),
        ("model_constructed", True),
        ("model_constructed", None),
    ],
)
def test_verify_preflight_rejects_restamped_safety_flag_tamper(
    tmp_path, field, replacement
):
    preflight_path, _cache = scoring_fixture(tmp_path)
    preflight = json.loads(preflight_path.read_bytes())
    if replacement is None:
        preflight.pop(field)
    else:
        preflight[field] = replacement

    with pytest.raises(ValueError, match="preflight|safety"):
        module.verify_preflight(preflight)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("tokenizer", None),
        ("score_cache", "remove"),
        ("sources", {}),
        ("runtime_evidence", "remove"),
        ("device_probe", None),
    ],
)
def test_verify_preflight_rejects_missing_critical_evidence_mapping(
    tmp_path, field, replacement
):
    preflight_path, _cache = scoring_fixture(tmp_path)
    preflight = json.loads(preflight_path.read_bytes())
    if replacement == "remove":
        preflight.pop(field)
    else:
        preflight[field] = replacement

    with pytest.raises(ValueError, match="evidence|preflight|tokenizer|cache|source|runtime|ROCm"):
        module.verify_preflight(preflight)


def test_scoring_rejects_cache_binding_drift_before_model(tmp_path, monkeypatch):
    preflight_path, cache = scoring_fixture(tmp_path)
    cache.path.write_text("changed after preflight\n", encoding="utf-8")
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )

    with pytest.raises(ValueError, match="score-cache binding"):
        run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))


def test_verify_preflight_binds_runtime_to_prior_scoring_source(tmp_path):
    preflight_path, _cache = scoring_fixture(tmp_path)
    preflight = json.loads(preflight_path.read_bytes())
    preflight["runtime_evidence"]["prior_scoring_receipt_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="runtime.*source|prior scoring"):
        module.verify_preflight(preflight)


def test_verify_scoring_rejects_restamped_candidate_identity_tamper(
    tmp_path, monkeypatch
):
    preflight_path, cache = scoring_fixture(tmp_path)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )
    run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))
    output = preflight_path.parent
    candidate_path = output / "candidates.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_bytes().splitlines()]
    candidates[0]["document_id"] = "restamped-different-document"
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidates)
    candidate_path.write_bytes(candidate_bytes)

    preflight = json.loads(preflight_path.read_bytes())
    preflight["candidates_sha256"] = _sha256(candidate_bytes)
    preflight_path.write_bytes(_pretty(preflight))
    preflight_sha = _sha256(preflight_path.read_bytes())

    reservation_path = output / "scoring_reservation.json"
    reservation = json.loads(reservation_path.read_bytes())
    reservation["preflight_sha256"] = preflight_sha
    reservation_path.write_bytes(_pretty(reservation))

    receipt_path = output / "scoring_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["preflight_sha256"] = preflight_sha
    receipt["scoring_reservation_sha256"] = _sha256(reservation_path.read_bytes())
    receipt_path.write_bytes(_pretty(receipt))

    with pytest.raises(ValueError, match="candidate.*(window identities|lineage)"):
        module.verify_scoring(preflight_path)


def test_verify_scoring_rejects_restamped_candidate_query_semantics(
    tmp_path, monkeypatch
):
    preflight_path, cache = scoring_fixture(tmp_path)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: type("Receipt", (), {"sha256": "a" * 64})(),
    )
    run_local_scoring(preflight_path, runner=FakeRunner(scores={"k2": 2.0}))
    output = preflight_path.parent
    candidate_path = output / "candidates.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_bytes().splitlines()]
    candidates[0]["query"] = "restamped query"
    candidates[0]["query_sha256"] = _sha256(b"restamped query")
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidates)
    candidate_path.write_bytes(candidate_bytes)
    preflight = json.loads(preflight_path.read_bytes())
    preflight["candidates_sha256"] = _sha256(candidate_bytes)
    preflight_path.write_bytes(_pretty(preflight))
    preflight_sha = _sha256(preflight_path.read_bytes())
    reservation_path = output / "scoring_reservation.json"
    reservation = json.loads(reservation_path.read_bytes())
    reservation["preflight_sha256"] = preflight_sha
    reservation_path.write_bytes(_pretty(reservation))
    receipt_path = output / "scoring_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["preflight_sha256"] = preflight_sha
    receipt["scoring_reservation_sha256"] = _sha256(reservation_path.read_bytes())
    receipt_path.write_bytes(_pretty(receipt))

    with pytest.raises(ValueError, match="candidate.*lineage|query"):
        module.verify_scoring(preflight_path)


def test_verify_preflight_rejects_inconsistent_coverage_cardinality(tmp_path):
    preflight_path, _cache = scoring_fixture(tmp_path)
    preflight = json.loads(preflight_path.read_bytes())
    first_key = next(iter(preflight["summary"]["document_window_coverage"]))
    preflight["summary"]["document_window_coverage"][first_key].append(0.5)

    with pytest.raises(ValueError, match="coverage|population"):
        module.verify_preflight(preflight)

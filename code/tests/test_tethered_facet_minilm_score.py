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
    build_argument_parser,
    build_tethered_candidates,
    create_preflight,
    enforce_preflight_ceiling,
    render_tethered_query,
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

    def cache_key(self, *, query_text: str, text: str) -> str:
        return _sha256(_compact([query_text, text]))

    def get(self, *, query_text: str, text: str):
        return self.scores.get(self.cache_key(query_text=query_text, text=text))


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
    (gate / "summary.json").write_bytes(
        _pretty(
            {
                "schema_version": "deep-facet-candidate-gate-v1",
                "status": "complete",
                "qrels_opened": False,
                "facet_count": 25,
                "accepted_facet_count": 24,
                "rejected_facet_count": 1,
                "artifacts": {
                    "gates.json": {
                        "bytes": len(gates_bytes),
                        "sha256": _sha256(gates_bytes),
                    }
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

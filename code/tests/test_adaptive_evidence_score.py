from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag import adaptive_evidence_score as score_module
from trec_rag.adaptive_evidence_score import (
    build_score_candidates,
    build_score_preflight,
    load_score_contract,
    persist_score_preflight,
    run_score_preflight,
    score_window_rows,
)


class _Tokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        del add_special_tokens, truncation
        return [len(token) for token in text.split()]

    def decode(
        self,
        tokens: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        if tokens == [6]:
            return "cached"
        return " ".join("x" * token for token in tokens)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is True
        return 3


def _contract() -> dict[str, object]:
    narrative = "full narrative"
    facet_query = (
        "full narrative\n\nExplicit obligation:\nPositive effects on society."
    )
    return {
        "obligations": [
            {
                "topic_id": "219",
                "obligation_id": "219:broad",
                "kind": "broad",
                "query": narrative,
            },
            {
                "topic_id": "219",
                "obligation_id": "219-positive",
                "kind": "o0",
                "source_facet_id": "219-positive",
                "query": facet_query,
            },
        ],
        "documents": [
            {
                "topic_id": "219",
                "document_id": "original-only",
                "union_order": 1,
                "text": "cached",
                "text_sha256": hashlib.sha256(b"cached").hexdigest(),
                "fold": 0,
                "provenance": [{"family": "original", "rank": 1}],
            },
            {
                "topic_id": "219",
                "document_id": "facet-doc",
                "union_order": 2,
                "text": "uncached passage",
                "text_sha256": hashlib.sha256(b"uncached passage").hexdigest(),
                "fold": 1,
                "provenance": [
                    {
                        "family": "facet",
                        "facet_id": "219-positive",
                        "rank": 1,
                    }
                ],
            },
        ],
    }


def test_score_populations_keep_broad_universal_and_o0_parent_local() -> None:
    rows = build_score_candidates(_contract())
    identities = {(row["document_id"], row["obligation_id"]) for row in rows}
    assert ("original-only", "219:broad") in identities
    assert ("facet-doc", "219:broad") in identities
    assert ("facet-doc", "219-positive") in identities
    assert ("original-only", "219-positive") not in identities
    assert all(row["query"].startswith("full narrative") for row in rows)


def test_o1_scores_only_its_parent_population() -> None:
    derived = [
        {
            "topic_id": "219",
            "obligation_id": "219-positive:o1:access",
            "kind": "o1",
            "parent_id": "219-positive",
            "text": "accessibility effects",
            "query": (
                "full narrative\n\nExplicit obligation:\nPositive effects on society."
                "\n\nCorpus-derived sub-obligation:\naccessibility effects"
            ),
        }
    ]
    rows = build_score_candidates(_contract(), derived=derived)
    o1 = [row for row in rows if row["obligation_id"].endswith(":access")]
    assert {row["document_id"] for row in o1} == {"facet-doc"}


def test_preflight_records_exact_hits_misses_and_never_reads_qrels() -> None:
    preflight = build_score_preflight(
        build_score_candidates(_contract()),
        _Tokenizer(),
        cache_lookup=lambda query, text: 0.25 if text == "cached" else None,
    )
    assert preflight["qrels_opened"] is False
    assert preflight["summary"] == {
        "document_count": 2,
        "candidate_count": 3,
        "window_count": 3,
        "unique_pair_count": 3,
        "cache_hit_count": 1,
        "cache_miss_count": 2,
        "cache_hit_window_count": 1,
        "cache_miss_window_count": 2,
    }
    windows = preflight["windows"]
    assert sum(row["cache_hit"] for row in windows) == 1


def test_preflight_rejects_protected_before_tokenizer_or_cache_access() -> None:
    touched: list[str] = []

    class ForbiddenTokenizer:
        def encode(self, *args: object, **kwargs: object) -> list[int]:
            touched.append("tokenizer")
            return []

    candidate = {
        **build_score_candidates(_contract())[0],
        "topic_id": "144",
    }
    with pytest.raises(ValueError, match="protected topic 144"):
        build_score_preflight(
            [candidate],
            ForbiddenTokenizer(),
            cache_lookup=lambda query, text: touched.append("cache"),  # type: ignore[arg-type,return-value]
        )

    assert touched == []


def _write_scoring_preflight(
    path: Path,
    cache_root: Path,
    *,
    topic_id: str = "219",
) -> SimpleNamespace:
    cache = score_module.GlobalScoreCache(
        cache_root,
        score_module.score_cache_context(),
    )
    windows = [
        {
            "topic_id": topic_id,
            "variant": f"{topic_id}-facet",
            "window_id": "w1",
            "query": "facet query",
            "window_text": "same missing passage",
            "cache_hit": False,
            "cache_key": cache.cache_key(
                query_text="facet query", text="same missing passage"
            ),
        },
        {
            "topic_id": topic_id,
            "variant": f"{topic_id}-facet",
            "window_id": "w2",
            "query": "facet query",
            "window_text": "same missing passage",
            "cache_hit": False,
            "cache_key": cache.cache_key(
                query_text="facet query", text="same missing passage"
            ),
        },
        {
            "topic_id": topic_id,
            "variant": f"{topic_id}:broad",
            "window_id": "w3",
            "query": "broad query",
            "window_text": "cached passage",
            "cache_hit": True,
            "cache_key": cache.cache_key(
                query_text="broad query", text="cached passage"
            ),
        },
    ]
    for window in windows:
        window["schema_version"] = "facet-local-minilm-window-plan-row-v1"
    window_bytes = _jsonl_bytes(windows)
    candidate_bytes = b""
    materialization_path = path / "materialization.json"
    materialization_path.parent.mkdir(parents=True)
    materialization_path.write_text("receipt", encoding="utf-8")
    cache.add_many([("broad query", "cached passage", 0.25)])
    receipt = {
        "schema_version": "adaptive-evidence-score-preflight-v1",
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "network": False,
        "external_cost_usd": 0.0,
        "inference_count": 0,
        "inference_authorized": False,
        "model_constructed": False,
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "summary": {
            "document_count": 2,
            "candidate_count": 2,
            "window_count": 3,
            "unique_pair_count": 2,
            "cache_hit_count": 1,
            "cache_miss_count": 1,
            "cache_hit_window_count": 1,
            "cache_miss_window_count": 2,
        },
        "artifacts": {
            "candidates.jsonl": {
                "rows": 0,
                "bytes": 0,
                "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            },
            "windows.jsonl": {
                "rows": len(windows),
                "bytes": len(window_bytes),
                "sha256": hashlib.sha256(window_bytes).hexdigest(),
            },
        },
        "coverage_matrix": {
            "topics": [{"topic_id": topic_id, "window_count": len(windows)}],
            "obligations": [
                {
                    "topic_id": topic_id,
                    "obligation_id": f"{topic_id}-facet",
                    "window_count": 2,
                },
                {
                    "topic_id": topic_id,
                    "obligation_id": f"{topic_id}:broad",
                    "window_count": 1,
                },
            ],
            "documents": [],
        },
        "bindings": {
            "model_materialization": {
                "receipt_path": str(materialization_path),
                "receipt_sha256": "a" * 64,
                "snapshot_sha256": "b" * 64,
            },
            "score_cache": {
                "root": str(cache_root),
                "path": str(cache.path),
                "context": score_module.score_cache_context().artifact_metadata,
            },
        },
    }
    path.mkdir(exist_ok=True)
    (path / "candidates.jsonl").write_bytes(candidate_bytes)
    (path / "windows.jsonl").write_bytes(window_bytes)
    (path / "preflight.json").write_bytes(_compact_bytes(receipt))
    return SimpleNamespace(
        receipt_path=materialization_path,
        source=b"receipt",
        sha256="a" * 64,
        payload={
            "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
            "snapshot_sha256": "b" * 64,
        },
        snapshot=path / "snapshot",
    )


def test_resume_accepts_only_hash_matching_complete_shards(tmp_path: Path) -> None:
    shard = tmp_path / "219__broad.jsonl"
    shard.write_text('{"window_id":"w1","score":0.5}\n', encoding="utf-8")
    receipt = {
        "rows": 1,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }

    assert score_module.verify_completed_shard(shard, receipt) == 1

    shard.write_text('{"window_id":"w1","score":0.6}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="shard hash"):
        score_module.verify_completed_shard(shard, receipt)


class _InstrumentedPredictor:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.forward_pair_count = 0
        self.forward_call_count = 0

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        self.forward_pair_count += len(pairs)
        self.forward_call_count += 1
        return [0.75] * len(pairs)

    def execution_receipt(self) -> dict[str, object]:
        return {
            "device": "cuda",
            "execution_backend": "rocm",
            "device_name": "fake AMD",
            "torch_version": "fake torch",
            "torch_hip_version": "fake hip",
            "peak_device_memory_bytes": 1234,
            "peak_host_memory_bytes": 4567,
            "forward_pair_count": self.forward_pair_count,
            "forward_call_count": self.forward_call_count,
            "inference_elapsed_seconds": 0.5,
        }


def test_local_scoring_writes_complete_shards_and_resumes_without_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    verified = _write_scoring_preflight(preflight, cache_root)
    output = tmp_path / "base"
    predictor = _InstrumentedPredictor()

    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: verified,
    )
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: predictor,
        raising=False,
    )

    receipt = score_module.run_local_scoring(preflight, output, cache_root)

    assert receipt["status"] == "complete"
    assert receipt["completed_window_count"] == 3
    assert receipt["unique_forward_pair_count"] == 1
    assert len(receipt["shards"]) == 2
    assert predictor.calls == [[("facet query", "same missing passage")]]
    assert receipt["device_name"] == "fake AMD"

    (output / "receipt.json").unlink()
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: pytest.fail("complete shards must not reload the model"),
    )
    resumed = score_module.run_local_scoring(preflight, output, cache_root)

    assert resumed["resumed_shard_count"] == 2
    assert resumed["device_name"] == "fake AMD"
    assert resumed["torch_hip_version"] == "fake hip"
    assert resumed["peak_device_memory_bytes"] == 1234
    assert score_module.verify_local_scoring(output)["completed_window_count"] == 3
    assert score_module.main(["verify", "--output", str(output)]) == 0
    receipt_path = output / "receipt.json"
    receipt_source = receipt_path.read_bytes()
    unsafe_receipt = json.loads(receipt_source)
    unsafe_receipt["model_loading"]["local_files_only"] = False
    receipt_path.write_bytes(_compact_bytes(unsafe_receipt))
    with pytest.raises(ValueError, match="binding"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    wrong_snapshot_receipt = json.loads(receipt_source)
    wrong_snapshot_receipt["model_snapshot_sha256"] = "f" * 64
    receipt_path.write_bytes(_compact_bytes(wrong_snapshot_receipt))
    with pytest.raises(ValueError, match="materialization"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    wrong_execution_receipt = json.loads(receipt_source)
    wrong_execution_receipt["torch_hip_version"] = "tampered"
    receipt_path.write_bytes(_compact_bytes(wrong_execution_receipt))
    with pytest.raises(ValueError, match="execution"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    legacy_receipt = json.loads(receipt_source)
    legacy_receipt.pop("model_constructed")
    legacy_receipt["device"] = "cpu"
    legacy_receipt["execution_backend"] = "network"
    legacy_sidecar_sources: dict[Path, bytes] = {}
    legacy_shards = []
    for shard in legacy_receipt["shards"]:
        shard_path = output / str(shard["path"])
        sidecar_path = shard_path.with_suffix(".receipt.json")
        legacy_sidecar_sources[sidecar_path] = sidecar_path.read_bytes()
        legacy_shard = dict(shard)
        legacy_shard.pop("execution")
        sidecar_path.write_bytes(_compact_bytes(legacy_shard))
        legacy_shards.append(legacy_shard)
    legacy_receipt["shards"] = legacy_shards
    receipt_path.write_bytes(_compact_bytes(legacy_receipt))
    try:
        with pytest.raises(ValueError, match="execution"):
            score_module.verify_local_scoring(output)
    finally:
        for sidecar_path, source in legacy_sidecar_sources.items():
            sidecar_path.write_bytes(source)
        receipt_path.write_bytes(receipt_source)
    wrong_throughput_receipt = json.loads(receipt_source)
    wrong_throughput_receipt["forward_pairs_per_second"] = 999.0
    receipt_path.write_bytes(_compact_bytes(wrong_throughput_receipt))
    with pytest.raises(ValueError, match="throughput"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    wrong_count_receipt = json.loads(receipt_source)
    wrong_count_receipt["unique_forward_pair_count"] = 99
    receipt_path.write_bytes(_compact_bytes(wrong_count_receipt))
    with pytest.raises(ValueError, match="count"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    wrong_planned_receipt = json.loads(receipt_source)
    wrong_planned_receipt["planned_candidate_count"] += 1
    receipt_path.write_bytes(_compact_bytes(wrong_planned_receipt))
    with pytest.raises(ValueError, match="planned"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    wrong_restored_receipt = json.loads(receipt_source)
    wrong_restored_receipt["restored_cache_pair_count"] = 99
    receipt_path.write_bytes(_compact_bytes(wrong_restored_receipt))
    with pytest.raises(ValueError, match="restored"):
        score_module.verify_local_scoring(output)
    receipt_path.write_bytes(receipt_source)
    shard_path = output / str(resumed["shards"][0]["path"])
    shard_source = shard_path.read_bytes()
    sidecar_path = shard_path.with_suffix(".receipt.json")
    sidecar_source = sidecar_path.read_bytes()
    score_rows = [json.loads(line) for line in shard_source.splitlines()]
    score_rows[0]["device"] = "cpu"
    changed_shard = _jsonl_bytes(score_rows)
    shard_path.write_bytes(changed_shard)
    changed_sidecar = json.loads(sidecar_source)
    changed_sidecar.update(
        {
            "bytes": len(changed_shard),
            "sha256": hashlib.sha256(changed_shard).hexdigest(),
        }
    )
    sidecar_path.write_bytes(_compact_bytes(changed_sidecar))
    changed_receipt = json.loads(receipt_source)
    changed_receipt["shards"] = [
        changed_sidecar if row["path"] == changed_sidecar["path"] else row
        for row in changed_receipt["shards"]
    ]
    receipt_path.write_bytes(_compact_bytes(changed_receipt))
    with pytest.raises(ValueError, match="device"):
        score_module.verify_local_scoring(output)
    shard_path.write_bytes(shard_source)
    sidecar_path.write_bytes(sidecar_source)
    receipt_path.write_bytes(receipt_source)
    unexpected = output / "shards" / "unexpected.jsonl"
    unexpected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected shard"):
        score_module.verify_local_scoring(output)
    unexpected.unlink()
    shard_path.write_bytes(shard_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="shard hash"):
        score_module.run_local_scoring(preflight, output, cache_root)


def test_partial_resume_restores_shared_pair_from_completed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    verified = _write_scoring_preflight(preflight, cache_root)
    window_path = preflight / "windows.jsonl"
    windows = [json.loads(line) for line in window_path.read_text().splitlines()]
    for row in windows:
        row.update(
            {
                "query": "one shared query",
                "window_text": "one shared passage",
                "cache_hit": False,
            }
        )
    cache = score_module.GlobalScoreCache(cache_root, score_module.score_cache_context())
    shared_key = cache.cache_key(
        query_text="one shared query",
        text="one shared passage",
    )
    for row in windows:
        row["cache_key"] = shared_key
    window_bytes = _jsonl_bytes(windows)
    window_path.write_bytes(window_bytes)
    preflight_path = preflight / "preflight.json"
    preflight_receipt = json.loads(preflight_path.read_bytes())
    preflight_receipt["artifacts"]["windows.jsonl"].update(
        {
            "bytes": len(window_bytes),
            "sha256": hashlib.sha256(window_bytes).hexdigest(),
        }
    )
    preflight_receipt["summary"].update(
        {
            "unique_pair_count": 1,
            "cache_hit_count": 0,
            "cache_miss_count": 1,
            "cache_hit_window_count": 0,
            "cache_miss_window_count": 3,
        }
    )
    preflight_path.write_bytes(_compact_bytes(preflight_receipt))
    first_predictor = _InstrumentedPredictor()
    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: verified,
    )
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: first_predictor,
        raising=False,
    )
    output = tmp_path / "base"
    first_receipt = score_module.run_local_scoring(preflight, output, cache_root)
    assert first_receipt["unique_forward_pair_count"] == 1

    (output / "receipt.json").unlink()
    broad = next(
        row for row in first_receipt["shards"] if row["obligation_id"].endswith(":broad")
    )
    broad_path = output / str(broad["path"])
    broad_path.unlink()
    broad_path.with_suffix(".receipt.json").unlink()
    cache.path.unlink()
    resumed_predictor = _InstrumentedPredictor()
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: resumed_predictor,
    )
    conflicting_cache = score_module.GlobalScoreCache(
        cache_root, score_module.score_cache_context()
    )
    conflicting_cache.add_many(
        [("one shared query", "one shared passage", -0.5)]
    )
    with pytest.raises(ValueError, match="conflicting score"):
        score_module.run_local_scoring(preflight, output, cache_root)
    conflicting_cache.path.unlink()

    resumed = score_module.run_local_scoring(preflight, output, cache_root)

    assert resumed_predictor.calls == []
    assert resumed["unique_forward_pair_count"] == 1
    assert resumed["restored_cache_pair_count"] == 1


def test_local_scoring_rejects_protected_receipt_before_window_cache_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    _write_scoring_preflight(preflight, cache_root, topic_id="144")
    (preflight / "windows.jsonl").unlink()
    touched: list[str] = []
    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: touched.append("model"),
    )

    with pytest.raises(ValueError, match="protected topic 144"):
        score_module.run_local_scoring(preflight, tmp_path / "base", cache_root)

    assert touched == []


def test_local_scoring_rejects_frozen_cache_key_outside_pinned_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    verified = _write_scoring_preflight(preflight, cache_root)
    windows = [
        json.loads(line)
        for line in (preflight / "windows.jsonl").read_text().splitlines()
    ]
    for row in windows[:2]:
        row["cache_key"] = "f" * 64
    window_bytes = _jsonl_bytes(windows)
    (preflight / "windows.jsonl").write_bytes(window_bytes)
    receipt_path = preflight / "preflight.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["artifacts"]["windows.jsonl"].update(
        {
            "bytes": len(window_bytes),
            "sha256": hashlib.sha256(window_bytes).hexdigest(),
        }
    )
    receipt_path.write_bytes(_compact_bytes(receipt))
    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: verified,
    )
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: pytest.fail("invalid cache key must fail first"),
        raising=False,
    )

    with pytest.raises(ValueError, match="cache key"):
        score_module.run_local_scoring(preflight, tmp_path / "base", cache_root)


def test_local_scoring_records_explicit_cache_only_execution_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    verified = _write_scoring_preflight(preflight, cache_root)
    window_path = preflight / "windows.jsonl"
    windows = [json.loads(line) for line in window_path.read_text().splitlines()]
    for row in windows:
        row["cache_hit"] = False
    window_bytes = _jsonl_bytes(windows)
    window_path.write_bytes(window_bytes)
    preflight_path = preflight / "preflight.json"
    preflight_receipt = json.loads(preflight_path.read_bytes())
    preflight_receipt["artifacts"]["windows.jsonl"].update(
        {
            "bytes": len(window_bytes),
            "sha256": hashlib.sha256(window_bytes).hexdigest(),
        }
    )
    preflight_receipt["summary"].update(
        {
            "cache_hit_count": 0,
            "cache_miss_count": 2,
            "cache_hit_window_count": 0,
            "cache_miss_window_count": 3,
        }
    )
    preflight_path.write_bytes(_compact_bytes(preflight_receipt))
    cache = score_module.GlobalScoreCache(cache_root, score_module.score_cache_context())
    cache.add_many([("facet query", "same missing passage", 0.75)])
    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: verified,
    )
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: pytest.fail("all-cache run must not load the model"),
        raising=False,
    )

    receipt = score_module.run_local_scoring(preflight, tmp_path / "base", cache_root)

    assert receipt["unique_forward_pair_count"] == 0
    assert receipt["model_constructed"] is False
    assert receipt["device"] == "cache"
    assert receipt["execution_backend"] == "global_score_cache"
    assert score_module.verify_local_scoring(tmp_path / "base")["status"] == "complete"


def test_verifier_requires_execution_for_every_forward_bearing_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = tmp_path / "preflight"
    cache_root = tmp_path / "cache"
    verified = _write_scoring_preflight(preflight, cache_root)
    window_path = preflight / "windows.jsonl"
    windows = [json.loads(line) for line in window_path.read_text().splitlines()]
    for row in windows:
        row["cache_hit"] = False
    window_bytes = _jsonl_bytes(windows)
    window_path.write_bytes(window_bytes)
    preflight_path = preflight / "preflight.json"
    preflight_receipt = json.loads(preflight_path.read_bytes())
    preflight_receipt["artifacts"]["windows.jsonl"].update(
        {
            "bytes": len(window_bytes),
            "sha256": hashlib.sha256(window_bytes).hexdigest(),
        }
    )
    preflight_receipt["summary"].update(
        {
            "cache_hit_count": 0,
            "cache_miss_count": 2,
            "cache_hit_window_count": 0,
            "cache_miss_window_count": 3,
        }
    )
    preflight_path.write_bytes(_compact_bytes(preflight_receipt))
    cache = score_module.GlobalScoreCache(cache_root, score_module.score_cache_context())
    cache.path.unlink()
    predictor = _InstrumentedPredictor()
    monkeypatch.setattr(
        score_module,
        "load_verified_materialization",
        lambda path: verified,
    )
    monkeypatch.setattr(
        score_module,
        "_load_rocm_predictor",
        lambda materialization: predictor,
        raising=False,
    )
    output = tmp_path / "base"
    receipt = score_module.run_local_scoring(preflight, output, cache_root)
    forward_shards = [
        row for row in receipt["shards"] if row["forward_pair_count"] > 0
    ]
    assert len(forward_shards) == 2
    changed_shard = dict(forward_shards[-1])
    changed_shard.pop("execution")
    sidecar = output / str(changed_shard["path"])
    sidecar = sidecar.with_suffix(".receipt.json")
    sidecar.write_bytes(_compact_bytes(changed_shard))
    changed_receipt = dict(receipt)
    changed_receipt["shards"] = [
        changed_shard if row["path"] == changed_shard["path"] else row
        for row in receipt["shards"]
    ]
    (output / "receipt.json").write_bytes(_compact_bytes(changed_receipt))

    with pytest.raises(ValueError, match="execution"):
        score_module.verify_local_scoring(output)


class _FakeTensor:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> "_FakeTensor":
        self.devices.append(device)
        return self


class _FakeLogits:
    shape = (2, 1)

    def detach(self) -> "_FakeLogits":
        return self

    def float(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def tolist(self) -> list[list[float]]:
        return [[0.5], [-0.25]]


class _FakeTokenizerAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str], dict[str, object]]] = []

    def __call__(
        self,
        queries: list[str],
        texts: list[str],
        **kwargs: object,
    ) -> dict[str, _FakeTensor]:
        self.calls.append((queries, texts, kwargs))
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}


class _FakeModelAdapter:
    def __init__(self) -> None:
        self.float_calls = 0
        self.eval_calls = 0
        self.devices: list[str] = []

    def float(self) -> "_FakeModelAdapter":
        self.float_calls += 1
        return self

    def eval(self) -> "_FakeModelAdapter":
        self.eval_calls += 1
        return self

    def to(self, device: str) -> "_FakeModelAdapter":
        self.devices.append(device)
        return self

    def __call__(self, **inputs: object) -> SimpleNamespace:
        assert inputs
        return SimpleNamespace(logits=_FakeLogits())


class _FakeLoader:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def from_pretrained(self, path: Path, **kwargs: object) -> object:
        self.calls.append((path, kwargs))
        return self.value


class _FakeTorchAdapter:
    float32 = "float32"
    __version__ = "2.9.1+rocm"

    def __init__(self, *, available: bool = True) -> None:
        self.version = SimpleNamespace(hip="7.2.1")
        self.cuda = SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: 1 if available else 0,
            get_device_name=lambda index: "AMD Radeon 8060S",
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1234,
            synchronize=lambda: None,
        )

    @contextmanager
    def inference_mode(self):
        yield


def test_rocm_predictor_loads_only_authenticated_float32_snapshot() -> None:
    tokenizer = _FakeTokenizerAdapter()
    model = _FakeModelAdapter()
    tokenizer_loader = _FakeLoader(tokenizer)
    model_loader = _FakeLoader(model)
    clock_values = iter([10.0, 11.0])
    runtime = SimpleNamespace(
        torch=_FakeTorchAdapter(),
        auto_tokenizer_cls=tokenizer_loader,
        auto_model_cls=model_loader,
        clock=lambda: next(clock_values),
        host_memory_bytes=lambda: 4567,
    )
    verified = SimpleNamespace(snapshot=Path("/authenticated/snapshot"))

    predictor = score_module._load_rocm_predictor(verified, runtime=runtime)
    scores = predictor([("q1", "t1"), ("q2", "t2")])

    assert scores == [0.5, -0.25]
    assert tokenizer_loader.calls == [
        (
            verified.snapshot,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]
    assert model_loader.calls == [
        (
            verified.snapshot,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "torch_dtype": runtime.torch.float32,
            },
        )
    ]
    assert model.float_calls == model.eval_calls == 1
    assert model.devices == ["cuda"]
    assert tokenizer.calls[0][2] == {
        "padding": True,
        "truncation": False,
        "max_length": 512,
        "return_tensors": "pt",
    }
    assert predictor.forward_pair_count == 2
    assert predictor.forward_call_count == 1
    assert predictor.execution_receipt()["execution_backend"] == "rocm"
    assert predictor.execution_receipt()["device_name"] == "AMD Radeon 8060S"
    assert predictor.execution_receipt()["peak_device_memory_bytes"] == 1234


def test_rocm_predictor_probes_gpu_before_model_snapshot_access() -> None:
    tokenizer_loader = _FakeLoader(_FakeTokenizerAdapter())
    model_loader = _FakeLoader(_FakeModelAdapter())
    runtime = SimpleNamespace(
        torch=_FakeTorchAdapter(available=False),
        auto_tokenizer_cls=tokenizer_loader,
        auto_model_cls=model_loader,
        clock=lambda: 0.0,
        host_memory_bytes=lambda: 0,
    )

    with pytest.raises(RuntimeError, match="ROCm"):
        score_module._load_rocm_predictor(
            SimpleNamespace(snapshot=Path("/must-not-open")),
            runtime=runtime,
        )

    assert tokenizer_loader.calls == []
    assert model_loader.calls == []


def test_cache_lookup_runs_once_per_unique_query_window_pair() -> None:
    first = next(
        row
        for row in build_score_candidates(_contract())
        if row["document_id"] == "original-only"
    )
    candidates = [first, {**first, "document_id": "duplicate-text", "rank": 9}]
    calls: list[tuple[str, str]] = []

    def lookup(query: str, text: str) -> None:
        calls.append((query, text))
        return None

    preflight = build_score_preflight(candidates, _Tokenizer(), lookup)

    assert calls == [("full narrative", "cached")]
    assert preflight["summary"]["candidate_count"] == 2
    assert preflight["summary"]["window_count"] == 2
    assert preflight["summary"]["unique_pair_count"] == 1


def _compact_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_compact_bytes(row) for row in rows)


def _write_contract(path: Path, *, topic_ids: list[str] | None = None) -> None:
    topic_ids = topic_ids or ["219", "72", "300", "84"]
    obligations = [
        {
            "topic_id": topic_id,
            "obligation_id": f"{topic_id}:broad",
            "kind": "broad",
            "query": f"narrative {topic_id}",
        }
        for topic_id in topic_ids
    ]
    documents = [
        {
            "topic_id": topic_id,
            "document_id": f"d-{topic_id}",
            "union_order": 1,
            "text": f"text {topic_id}",
            "text_sha256": hashlib.sha256(f"text {topic_id}".encode()).hexdigest(),
            "fold": 0,
            "provenance": [{"family": "original", "rank": 1}],
        }
        for topic_id in topic_ids
    ]
    folds = [
        {
            "topic_id": row["topic_id"],
            "document_id": row["document_id"],
            "union_order": row["union_order"],
            "fold": row["fold"],
        }
        for row in documents
    ]
    manifest = {
        "schema_version": "adaptive-evidence-contract-v1",
        "topic_ids": topic_ids,
        "protected_topic_ids": ["144", "213", "224", "407", "515"],
        "document_count": len(documents),
        "broad_obligation_count": len(obligations),
        "o0_obligation_count": 0,
        "qrels_opened": False,
    }
    content = {
        "manifest.json": _compact_bytes(manifest),
        "obligations.jsonl": _jsonl_bytes(obligations),
        "documents.jsonl": _jsonl_bytes(documents),
        "folds.jsonl": _jsonl_bytes(folds),
    }
    summary = {
        "schema_version": "adaptive-evidence-contract-v1",
        "status": "complete",
        "topic_ids": topic_ids,
        "document_count": len(documents),
        "broad_obligation_count": len(obligations),
        "o0_obligation_count": 0,
        "protected_topic_count": 0,
        "qrels_opened": False,
        "artifact_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in content.items()
        },
    }
    path.mkdir()
    for name, value in content.items():
        (path / name).write_bytes(value)
    (path / "summary.json").write_bytes(_compact_bytes(summary))


def test_contract_loader_authenticates_all_task1_artifacts(tmp_path: Path) -> None:
    contract_dir = tmp_path / "contract"
    _write_contract(contract_dir)

    contract = load_score_contract(
        contract_dir,
        expected_document_count=4,
        expected_o0_count=0,
    )

    assert len(contract["documents"]) == 4
    assert len(contract["obligations"]) == 4
    (contract_dir / "folds.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="folds.jsonl hash"):
        load_score_contract(
            contract_dir,
            expected_document_count=4,
            expected_o0_count=0,
        )


def test_contract_loader_rejects_protected_manifest_before_row_files(
    tmp_path: Path,
) -> None:
    contract_dir = tmp_path / "contract"
    _write_contract(contract_dir, topic_ids=["144"])
    (contract_dir / "documents.jsonl").unlink()

    with pytest.raises(ValueError, match="protected topic 144"):
        load_score_contract(
            contract_dir,
            expected_document_count=1,
            expected_o0_count=0,
            expected_broad_count=1,
        )


def _replace_contract_jsonl(
    contract_dir: Path,
    name: str,
    rows: list[dict[str, object]],
) -> None:
    content = _jsonl_bytes(rows)
    (contract_dir / name).write_bytes(content)
    summary_path = contract_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_sha256"][name] = hashlib.sha256(content).hexdigest()
    summary_path.write_bytes(_compact_bytes(summary))


def test_contract_loader_requires_exact_unique_document_population(
    tmp_path: Path,
) -> None:
    contract_dir = tmp_path / "contract"
    _write_contract(contract_dir)
    rows = [
        json.loads(line)
        for line in (contract_dir / "documents.jsonl").read_text().splitlines()
    ]
    rows[-1] = {**rows[0], "union_order": 2}
    _replace_contract_jsonl(contract_dir, "documents.jsonl", rows)

    with pytest.raises(ValueError, match="exactly 4 unique topic-document"):
        load_score_contract(
            contract_dir,
            expected_document_count=4,
            expected_o0_count=0,
        )


def test_contract_loader_requires_one_broad_obligation_per_pilot_topic(
    tmp_path: Path,
) -> None:
    contract_dir = tmp_path / "contract"
    _write_contract(contract_dir)
    rows = [
        json.loads(line)
        for line in (contract_dir / "obligations.jsonl").read_text().splitlines()
    ]
    rows[-1] = {
        **rows[0],
        "obligation_id": "219:duplicate-broad",
    }
    _replace_contract_jsonl(contract_dir, "obligations.jsonl", rows)

    with pytest.raises(ValueError, match="exactly one broad obligation per pilot topic"):
        load_score_contract(
            contract_dir,
            expected_document_count=4,
            expected_o0_count=0,
        )


def test_candidate_builder_rejects_duplicate_document_identity() -> None:
    contract = _contract()
    duplicate = {**contract["documents"][0], "union_order": 3}  # type: ignore[index]
    contract["documents"].append(duplicate)  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="unique topic-document"):
        build_score_candidates(contract)


def test_o1_rejects_missing_parent_explicitly() -> None:
    with pytest.raises(ValueError, match="O1 parent 219-missing does not exist"):
        build_score_candidates(
            _contract(),
            derived=[
                {
                    "topic_id": "219",
                    "obligation_id": "219-missing:o1:x",
                    "kind": "o1",
                    "parent_id": "219-missing",
                    "query": "full narrative\n\nderived",
                }
            ],
        )


def test_o1_rejects_non_o0_parent() -> None:
    with pytest.raises(ValueError, match="O1 parent 219:broad must be kind o0"):
        build_score_candidates(
            _contract(),
            derived=[
                {
                    "topic_id": "219",
                    "obligation_id": "219:broad:o1:x",
                    "kind": "o1",
                    "parent_id": "219:broad",
                    "query": "full narrative\n\nderived",
                }
            ],
        )


def test_o1_rejects_cross_topic_parent() -> None:
    with pytest.raises(ValueError, match="O1 parent 219-positive must share topic 72"):
        build_score_candidates(
            _contract(),
            derived=[
                {
                    "topic_id": "72",
                    "obligation_id": "72:o1:x",
                    "kind": "o1",
                    "parent_id": "219-positive",
                    "query": "narrative 72\n\nderived",
                }
            ],
        )


@pytest.mark.parametrize("source_facet_id", [None, "219-wrong-facet"])
def test_o1_rejects_invalid_parent_source_facet(
    source_facet_id: str | None,
) -> None:
    contract = _contract()
    parent = contract["obligations"][1]  # type: ignore[index]
    if source_facet_id is None:
        parent.pop("source_facet_id")
    else:
        parent["source_facet_id"] = source_facet_id

    with pytest.raises(ValueError, match="O1 parent 219-positive source facet is invalid"):
        build_score_candidates(
            contract,
            derived=[
                {
                    "topic_id": "219",
                    "obligation_id": "219-positive:o1:x",
                    "kind": "o1",
                    "parent_id": "219-positive",
                    "query": "full narrative\n\nderived",
                }
            ],
        )


def test_persist_preflight_is_create_only_and_hashes_artifacts(
    tmp_path: Path,
) -> None:
    candidates = build_score_candidates(_contract())
    plan = build_score_preflight(
        candidates,
        _Tokenizer(),
        cache_lookup=lambda query, text: None,
    )
    output = tmp_path / "preflight"

    receipt = persist_score_preflight(
        output,
        candidates=candidates,
        preflight=plan,
        bindings={"model_materialization_receipt_sha256": "a" * 64},
    )

    assert receipt["qrels_opened"] is False
    assert receipt["network"] is False
    assert receipt["external_cost_usd"] == 0.0
    for name in ("candidates.jsonl", "windows.jsonl"):
        assert receipt["artifacts"][name]["sha256"] == hashlib.sha256(  # type: ignore[index]
            (output / name).read_bytes()
        ).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        persist_score_preflight(
            output,
            candidates=candidates,
            preflight=plan,
            bindings={},
        )


def test_run_preflight_rejects_existing_output_before_any_input_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        run_score_preflight(
            contract_dir=tmp_path / "missing-contract",
            output_dir=output,
        )


def _window(window_id: str, query: str, text: str) -> dict[str, object]:
    return {
        "topic_id": "219",
        "window_id": window_id,
        "query": query,
        "window_text": text,
    }


def test_score_window_rows_predicts_each_miss_once_and_restores_all_rows() -> None:
    rows = [_window("w1", "q", "same"), _window("w2", "q", "same")]
    cache: dict[tuple[str, str], float] = {}
    predicted: list[list[tuple[str, str]]] = []
    added: list[tuple[str, str, float]] = []

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        predicted.append(pairs)
        return [0.75 for _ in pairs]

    def cache_add(pairs: object) -> None:
        for query, text, score in pairs:  # type: ignore[union-attr]
            added.append((query, text, score))
            cache[(query, text)] = score

    scores = score_window_rows(
        rows,
        predict=predict,
        cache_get=lambda query, text: cache.get((query, text)),
        cache_add=cache_add,
    )

    assert predicted == [[("q", "same")]]
    assert added == [("q", "same", 0.75)]
    assert [row["window_id"] for row in scores] == ["w1", "w2"]
    assert {row["score"] for row in scores} == {0.75}
    assert {row["model_revision"] for row in scores} == {
        "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    }


def test_score_window_rows_uses_complete_cache_without_prediction() -> None:
    def forbidden_predict(pairs: list[tuple[str, str]]) -> list[float]:
        raise AssertionError(f"predictor must not run for cached pairs: {pairs}")

    scores = score_window_rows(
        [_window("cached", "q", "text")],
        predict=forbidden_predict,
        cache_get=lambda query, text: 0.5,
        cache_add=lambda pairs: None,
    )

    assert scores[0]["score"] == 0.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_score_window_rows_rejects_nonfinite_scores(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        score_window_rows(
            [_window("bad", "q", "text")],
            predict=lambda pairs: [value],
            cache_get=lambda query, text: None,
            cache_add=lambda pairs: None,
        )


def test_score_window_rows_rejects_predictor_count_and_missing_cache_coverage() -> None:
    row = _window("missing", "q", "text")
    with pytest.raises(ValueError, match="predictor score count"):
        score_window_rows(
            [row],
            predict=lambda pairs: [],
            cache_get=lambda query, text: None,
            cache_add=lambda pairs: None,
        )
    with pytest.raises(ValueError, match="cache does not cover"):
        score_window_rows(
            [row],
            predict=lambda pairs: [0.25],
            cache_get=lambda query, text: None,
            cache_add=lambda pairs: None,
        )


@pytest.mark.parametrize("topic_id", ["144", None])
def test_score_window_rows_rejects_forbidden_topic_before_cache_or_prediction(
    topic_id: str | None,
) -> None:
    touched: list[str] = []
    row = _window("forbidden", "q", "text")
    if topic_id is None:
        row.pop("topic_id")
        match = "window topic_id must be nonempty"
    else:
        row["topic_id"] = topic_id
        match = "protected topic 144"

    with pytest.raises(ValueError, match=match):
        score_window_rows(
            [row],
            predict=lambda pairs: touched.append("predict"),  # type: ignore[arg-type,return-value]
            cache_get=lambda query, text: touched.append("cache_get"),  # type: ignore[return-value]
            cache_add=lambda pairs: touched.append("cache_add"),
        )

    assert touched == []

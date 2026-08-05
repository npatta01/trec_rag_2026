from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.tethered_facet_minilm_score as module


class _ScoreCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[tuple[str, str], float] = {}
        self.lookup_calls: list[tuple[object, ...]] = []
        self.score_many_calls: list[tuple[tuple[object, ...], int]] = []

    def cache_key(self, *, query_text: str, text: str) -> str:
        return "key" if (query_text, text) == ("query", "window") else "unexpected"

    def lookup_many(self, pairs: object) -> list[float | None]:
        materialized = tuple(pairs)  # type: ignore[arg-type]
        self.lookup_calls.append(materialized)
        return [self.values.get((pair[0], pair[1])) for pair in materialized]  # type: ignore[index]

    def score_many(self, pairs: object, compute_batch, batch_size: int) -> list[float]:
        materialized = tuple(pairs)  # type: ignore[arg-type]
        self.score_many_calls.append((materialized, batch_size))
        values = list(compute_batch(materialized))
        for pair, value in zip(materialized, values, strict=True):  # type: ignore[arg-type]
            self.values[(pair[0], pair[1])] = float(value)  # type: ignore[index]
        return values


class _Runner:
    peak_device_memory_bytes = 7

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def score(self, rows: list[dict[str, object]]) -> list[float]:
        self.calls.append(rows)
        return [0.25] * len(rows)


def test_local_scoring_uses_score_many_and_bounded_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "scoring"
    output.mkdir()
    preflight_path = output / "preflight.json"
    preflight_path.write_text("{}\n", encoding="utf-8")
    candidate = {
        "schema_version": module.CANDIDATE_SCHEMA_VERSION,
        "topic_id": "219",
        "facet_id": "f1",
        "document_id": "d1",
    }
    window = {
        "topic_id": "219",
        "facet_id": "f1",
        "document_id": "d1",
        "query": "query",
        "window_text": "window",
        "cache_key": "key",
        "cache_hit": False,
    }
    (output / "candidates.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    (output / "windows.jsonl").write_text(
        json.dumps(window) + "\n", encoding="utf-8"
    )
    window_bytes = (output / "windows.jsonl").read_bytes()
    candidate_bytes = (output / "candidates.jsonl").read_bytes()
    cache = _ScoreCache(tmp_path / "score-cache.sqlite3")
    binding = {
        "state": "absent",
        "path": str(cache.path),
        "bytes": 0,
        "sha256": None,
    }
    preflight = {
        "score_cache": {"binding": binding},
        "windows_sha256": module._sha256_bytes(window_bytes),
        "candidates_sha256": module._sha256_bytes(candidate_bytes),
        "summary": {"window_count": 1, "unique_cache_miss_count": 1},
        "model_materialization_receipt": "unused",
        "model_materialization_receipt_sha256": "receipt-sha",
    }
    runner = _Runner()
    monkeypatch.setattr(module, "PAIR_COUNT", 1)
    monkeypatch.setattr(module, "verify_preflight", lambda _path: preflight)
    monkeypatch.setattr(module, "GlobalScoreCache", lambda *_args: cache)
    monkeypatch.setattr(module, "_cache_binding", lambda _cache: binding)
    monkeypatch.setattr(
        module,
        "load_verified_materialization",
        lambda _path: SimpleNamespace(sha256="receipt-sha"),
    )
    monkeypatch.setattr(
        module,
        "aggregate_document_scores",
        lambda _rows: [{"topic_id": "219", "facet_id": "f1", "document_id": "d1"}],
    )

    receipt = module.run_local_scoring(preflight_path, tmp_path, runner=runner)

    assert receipt["status"] == "complete"
    assert cache.score_many_calls == [((("query", "window"),), module.BATCH_SIZE)]
    assert len(cache.lookup_calls) == 2
    assert runner.calls == [[window]]

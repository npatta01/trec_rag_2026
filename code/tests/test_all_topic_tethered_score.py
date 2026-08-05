from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.all_topic_tethered_score as module
from trec_rag.all_topic_tethered_score import (
    build_score_plan,
    percentile_features,
    render_tethered_query,
    run_scores,
)
from trec_rag.rerank_score_cache import GlobalScoreCache


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _TokenizerOnly:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        self.encoded.append(text)
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


class _EmptyCache:
    scores: dict[str, float] = {}

    def get(self, *, query_text: str, text: str):
        return None


def _manifest() -> dict[str, object]:
    narrative = "full narrative"
    query = "subject relation facet"
    return {
        "topic_ids": ["14"],
        "topics": [
            {
                "topic_id": "14",
                "manifest_order": 0,
                "narrative": narrative,
                "narrative_sha256": _sha(narrative),
            }
        ],
        "facets": [
            {
                "topic_id": "14",
                "facet_id": "f1",
                "manifest_order": 0,
                "query": query,
                "query_sha256": _sha(query),
                "analyzer_terms": ["subject", "relation", "facet"],
            }
        ],
    }


def _union() -> list[dict[str, object]]:
    text = "document passage"
    return [
        {
            "topic_id": "14",
            "document_id": "d1",
            "text": text,
            "stream_provenance": [
                {
                    "stream_id": "original",
                    "stream_rank": 1,
                    "text_sha256": _sha(text),
                },
                {
                    "stream_id": "f1",
                    "stream_rank": 2,
                    "text_sha256": _sha(text),
                },
            ],
        }
    ]


def _score_rows() -> list[dict[str, object]]:
    return [
        {"topic_id": "14", "query_id": "n", "document_id": "a", "score": 9.0},
        {"topic_id": "14", "query_id": "n", "document_id": "b", "score": 1.0},
        {"topic_id": "14", "query_id": "f1", "document_id": "a", "score": 2.0},
        {"topic_id": "14", "query_id": "f1", "document_id": "b", "score": 2.0},
    ]


def _percentiles(rows: list[dict[str, object]], query_id: str) -> list[float]:
    return [
        float(row["percentile"])
        for row in rows
        if row["query_id"] == query_id
    ]


def test_tethered_query_contains_narrative_and_one_facet() -> None:
    query = render_tethered_query("full narrative", "subject relation facet")
    assert "full narrative" in query and "subject relation facet" in query


def test_raw_scores_never_normalize_across_queries() -> None:
    rows = percentile_features(_score_rows())
    assert {(r["topic_id"], r["query_id"]) for r in rows} == {
        ("14", "n"),
        ("14", "f1"),
    }
    assert _percentiles(rows, "n") == [1.0, 0.0]
    assert _percentiles(rows, "f1") == [0.5, 0.5]


def test_preflight_has_no_model_load() -> None:
    plan = build_score_plan(
        _union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["external_calls"] == {
        "model_load": 0,
        "model_inference": 0,
        "network": 0,
        "hosted": 0,
        "paid": 0,
        "qrels": 0,
    }
    assert plan["pair_count"] == 3
    assert plan["window_count"] == 3
    assert plan["cache_hit_count"] == 0
    assert plan["cache_miss_count"] == 3


def test_preflight_reuses_one_facet_pair_even_when_original_also_retrieved_it() -> None:
    plan = build_score_plan(
        _union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert [row["query_id"] for row in plan["pairs"]] == ["n", "g", "f1"]


def test_percentiles_reject_duplicate_query_document_identity() -> None:
    rows = _score_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        percentile_features(rows)


def test_explicitly_authorized_historical_topic_uses_same_window_policy() -> None:
    manifest = _manifest()
    manifest["topic_ids"] = ["144"]
    manifest["topics"][0]["topic_id"] = "144"  # type: ignore[index]
    manifest["facets"][0]["topic_id"] = "144"  # type: ignore[index]
    union = _union()
    union[0]["topic_id"] = "144"
    plan = build_score_plan(
        union, manifest, backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["pair_count"] == 3
    assert {row["topic_id"] for row in plan["windows"]} == {"144"}


def test_topic_outside_experiment_allowlist_is_rejected() -> None:
    manifest = _manifest()
    manifest["topic_ids"] = ["999"]
    manifest["topics"][0]["topic_id"] = "999"  # type: ignore[index]
    manifest["facets"][0]["topic_id"] = "999"  # type: ignore[index]
    union = _union()
    union[0]["topic_id"] = "999"
    with pytest.raises(ValueError, match="allowlist"):
        build_score_plan(
            union, manifest, backend=_TokenizerOnly(), cache=_EmptyCache()
        )


def test_authenticated_union_accepts_normalized_duplicate_text_provenance() -> None:
    union = _union()
    union[0]["stream_provenance"][1]["text_sha256"] = _sha("document  passage")  # type: ignore[index]
    plan = build_score_plan(
        union, _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["pair_count"] == 3


def test_repeated_query_identities_tokenize_the_document_only_once() -> None:
    backend = _TokenizerOnly()
    build_score_plan(_union(), _manifest(), backend=backend, cache=_EmptyCache())
    assert backend.encoded.count("document passage") == 1


class _InterruptingRunner:
    peak_device_memory_bytes = 123

    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def score(self, rows):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("synthetic interruption")
        return [float(int(str(row["cache_key"])[0], 16)) for row in rows]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def test_run_is_resumable_without_mutating_shared_cache(tmp_path, monkeypatch) -> None:
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    cache_root = tmp_path / "cache"
    cache = GlobalScoreCache(cache_root, module.score_cache_context())
    plan = build_score_plan(
        _union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    _write_jsonl(
        scoring / "pairs.jsonl",
        [
            {
                "schema_version": module.PAIR_SCHEMA_VERSION,
                **{key: value for key, value in row.items() if key != "text"},
            }
            for row in plan["pairs"]
        ],
    )
    _write_jsonl(scoring / "windows.jsonl", plan["windows"])
    preflight = {
        "score_cache": {"binding": module._cache_binding(cache)},
        "unique_cache_miss_count": plan["unique_cache_miss_count"],
        "unique_pair_count": plan["unique_pair_count"],
        "window_count": plan["window_count"],
        "pair_count": plan["pair_count"],
        "cache_hit_count": plan["cache_hit_count"],
        "cache_miss_count": plan["cache_miss_count"],
        "artifacts": {},
        "model_materialization_receipt": "unused-with-injected-runner",
    }
    _write_json(scoring / "preflight.json", preflight)
    monkeypatch.setattr(
        module,
        "verify_preflight",
        lambda *args, **kwargs: {
            "root_sha256": "a" * 64,
            "window_count": plan["window_count"],
        },
    )
    monkeypatch.setattr(module, "BATCH_SIZE", 1)
    monkeypatch.setattr(module, "EXPECTED_PAIR_COUNT", 3)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_scores(
            scoring,
            cache_root=cache_root,
            runner=_InterruptingRunner(fail_call=2),
        )
    assert cache.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
    assert (scoring / "score_ledger.jsonl").is_file()
    receipt = run_scores(
        scoring, cache_root=cache_root, runner=_InterruptingRunner()
    )
    assert receipt["status"] == "complete"
    assert cache.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
    assert module.verify_scores(scoring)["verified"] is True


def _refresh_terminal_seal(scoring: Path) -> None:
    receipt = json.loads((scoring / "scoring_receipt.json").read_text())
    receipt["artifacts"] = {
        name: module._file_binding(scoring / name)
        for name in ("scores.jsonl", "document_scores.jsonl", "features.jsonl")
    }
    _write_json(scoring / "scoring_receipt.json", receipt)
    files = {
        name: module._file_binding(scoring / name)
        for name in (
            "scores.jsonl", "document_scores.jsonl", "features.jsonl",
            "scoring_receipt.json", "score_ledger.jsonl",
        )
    }
    seal = {
        "schema_version": module.SCORING_SEAL_SCHEMA_VERSION,
        "experiment_id": module.EXPERIMENT_ID,
        "score_plan_root_sha256": "a" * 64,
        "files": files,
        "root_sha256": module._sha256_bytes(module._compact_bytes(files)),
    }
    _write_json(scoring / "SCORING_SEALED.json", seal)


@pytest.mark.parametrize("artifact", ["document_scores.jsonl", "features.jsonl"])
def test_verify_recomputes_aggregates_and_features(
    tmp_path, monkeypatch, artifact
) -> None:
    # Reuse the complete synthetic run setup through the prior test helper body.
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    cache_root = tmp_path / "cache"
    cache = GlobalScoreCache(cache_root, module.score_cache_context())
    plan = build_score_plan(_union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache())
    _write_jsonl(scoring / "pairs.jsonl", [
        {"schema_version": module.PAIR_SCHEMA_VERSION, **{k: v for k, v in row.items() if k != "text"}}
        for row in plan["pairs"]
    ])
    _write_jsonl(scoring / "windows.jsonl", plan["windows"])
    _write_json(scoring / "preflight.json", {
        "score_cache": {"binding": module._cache_binding(cache)},
        "unique_cache_miss_count": plan["unique_cache_miss_count"],
        "unique_pair_count": plan["unique_pair_count"],
        "window_count": plan["window_count"], "pair_count": plan["pair_count"],
        "cache_hit_count": 0, "cache_miss_count": plan["window_count"],
        "model_materialization_receipt": "unused",
    })
    monkeypatch.setattr(module, "verify_preflight", lambda *a, **k: {
        "root_sha256": "a" * 64, "window_count": plan["window_count"]
    })
    monkeypatch.setattr(module, "EXPECTED_PAIR_COUNT", 3)
    run_scores(scoring, cache_root=cache_root, runner=_InterruptingRunner())
    rows = list(module._iter_jsonl(scoring / artifact, artifact))
    field = "score" if artifact.startswith("document") else "percentile"
    rows[0][field] = float(rows[0][field]) + 0.125
    _write_jsonl(scoring / artifact, rows)
    _refresh_terminal_seal(scoring)
    with pytest.raises(ValueError, match="recomputation"):
        module.verify_scores(scoring)


def test_semantic_replay_rejects_derived_query_and_window_drift(tmp_path) -> None:
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    manifest_path = tmp_path / "manifest.json"
    union_path = tmp_path / "accepted_union.jsonl"
    _write_json(manifest_path, _manifest())
    _write_jsonl(union_path, _union())
    backend = _TokenizerOnly()
    cache = _EmptyCache()
    plan = build_score_plan(_union(), _manifest(), backend=backend, cache=cache)
    pair_rows = [
        {"schema_version": module.PAIR_SCHEMA_VERSION, **{k: v for k, v in row.items() if k != "text"}}
        for row in plan["pairs"]
    ]
    _write_jsonl(scoring / "pairs.jsonl", pair_rows)
    _write_jsonl(scoring / "windows.jsonl", plan["windows"])
    payload = {
        "pair_count": 3, "window_count": 3, "cache_hit_count": 0,
        "cache_miss_count": 3, "unique_pair_count": 3,
        "unique_cache_miss_count": 3,
        "sources": {
            "manifest": {"path": str(manifest_path), "sha256": module._file_binding(manifest_path)["sha256"]},
            "accepted_union": {"path": str(union_path), **module._file_binding(union_path)},
        },
    }
    identities, windows = module._replay_plan_derivation(
        scoring, payload, tokenizer=_TokenizerOnly(), cache=cache,
        authenticate_upstream=False,
    )
    assert len(identities) == 3 and len(windows) == 3
    pair_rows[2]["query"] = "forged tethered query"
    _write_jsonl(scoring / "pairs.jsonl", pair_rows)
    with pytest.raises(ValueError, match="semantic derivation"):
        module._replay_plan_derivation(
            scoring, payload, tokenizer=_TokenizerOnly(), cache=cache,
            authenticate_upstream=False,
        )
    _write_jsonl(scoring / "pairs.jsonl", [
        {"schema_version": module.PAIR_SCHEMA_VERSION, **{k: v for k, v in row.items() if k != "text"}}
        for row in plan["pairs"]
    ])
    window_rows = list(plan["windows"])
    window_rows[0] = {**window_rows[0], "document_end_token": 0}
    _write_jsonl(scoring / "windows.jsonl", window_rows)
    with pytest.raises(ValueError, match="semantic derivation"):
        module._replay_plan_derivation(
            scoring, payload, tokenizer=_TokenizerOnly(), cache=cache,
            authenticate_upstream=False,
        )


def test_terminal_publication_recovers_after_interruption(tmp_path, monkeypatch) -> None:
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    cache_root = tmp_path / "cache"
    cache = GlobalScoreCache(cache_root, module.score_cache_context())
    plan = build_score_plan(_union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache())
    _write_jsonl(scoring / "pairs.jsonl", [
        {"schema_version": module.PAIR_SCHEMA_VERSION, **{k: v for k, v in row.items() if k != "text"}}
        for row in plan["pairs"]
    ])
    _write_jsonl(scoring / "windows.jsonl", plan["windows"])
    _write_json(scoring / "preflight.json", {
        "score_cache": {"binding": module._cache_binding(cache)},
        "unique_cache_miss_count": 3, "unique_pair_count": 3,
        "window_count": 3, "pair_count": 3,
        "cache_hit_count": 0, "cache_miss_count": 3,
        "model_materialization_receipt": "unused",
    })
    monkeypatch.setattr(module, "verify_preflight", lambda *a, **k: {
        "root_sha256": "a" * 64, "window_count": 3
    })
    monkeypatch.setattr(module, "EXPECTED_PAIR_COUNT", 3)
    def interrupt(name):
        if name == "scores.jsonl":
            raise RuntimeError("publication interruption")
    with pytest.raises(RuntimeError, match="publication interruption"):
        run_scores(scoring, cache_root=cache_root, runner=_InterruptingRunner(), publish_hook=interrupt)
    assert (scoring / "scores.jsonl").exists()
    assert not (scoring / "SCORING_SEALED.json").exists()
    receipt = run_scores(scoring, cache_root=cache_root, runner=_InterruptingRunner())
    assert receipt["status"] == "complete"
    assert module.verify_scores(scoring)["verified"] is True


def test_create_and_semantically_verify_preflight_end_to_end(
    tmp_path, monkeypatch
) -> None:
    planning = tmp_path / "planning"
    retrieval = tmp_path / "retrieval"
    scoring = tmp_path / "scoring"
    cache_root = tmp_path / "cache"
    planning.mkdir()
    retrieval.mkdir()
    manifest_path = planning / "manifest.json"
    union_path = retrieval / "accepted_union.jsonl"
    _write_json(manifest_path, _manifest())
    _write_jsonl(union_path, _union())
    reference = tmp_path / "reference.json"
    _write_json(reference, {
        "unique_forward_pair_count": 10, "elapsed_seconds": 2.0,
        "peak_device_memory_bytes": 100, "peak_host_memory_bytes": 200,
    })
    receipt_path = tmp_path / "materialization.json"
    receipt_path.write_text("{}\n")
    materialization = SimpleNamespace(
        receipt_path=receipt_path.resolve(), sha256=_sha(receipt_path.read_text())
    )
    monkeypatch.setattr(module, "ALL_TOPIC_IDS", ("14",))
    monkeypatch.setattr(module, "EXPECTED_UNION_ROWS", 1)
    monkeypatch.setattr(module, "EXPECTED_FACET_ASSOCIATIONS", 1)
    monkeypatch.setattr(module, "EXPECTED_PAIR_COUNT", 3)
    monkeypatch.setattr(module, "EXPECTED_FACET_COUNT", 1)
    monkeypatch.setattr(module, "PLANNING_ROOT_SHA256", "1" * 64)
    monkeypatch.setattr(module, "RETRIEVAL_ROOT_SHA256", "2" * 64)
    monkeypatch.setattr(module, "MANIFEST_SHA256", module._file_binding(manifest_path)["sha256"])
    monkeypatch.setattr(module, "ACCEPTED_UNION_SHA256", module._file_binding(union_path)["sha256"])
    monkeypatch.setattr(module, "verify_planning", lambda *a, **k: {
        "root_sha256": "1" * 64
    })
    monkeypatch.setattr(module, "verify_retrieval", lambda *a, **k: {
        "root_sha256": "2" * 64, "accepted_union_rows": 1, "topic_count": 1
    })
    monkeypatch.setattr(module, "load_verified_materialization", lambda *a, **k: materialization)
    monkeypatch.setattr(module, "load_verified_tokenizer", lambda *a, **k: _TokenizerOnly())
    payload = module.create_preflight(
        retrieval, scoring, planning_dir=planning, model_receipt=receipt_path,
        cache_root=cache_root, reference_receipt=reference,
    )
    seal = json.loads((scoring / "SCORE_PLAN_SEALED.json").read_text())
    monkeypatch.setattr(module, "APPROVED_SCORE_PLAN_ROOT_SHA256", seal["root_sha256"])
    cache = GlobalScoreCache(cache_root, module.score_cache_context())
    verified = module.verify_preflight(
        scoring, expected_root_sha256=seal["root_sha256"],
        semantic_replay=True, authenticate_upstream=False,
        tokenizer=_TokenizerOnly(), cache=cache,
    )
    assert verified["pair_count"] == payload["pair_count"] == 3
    assert verified["window_count"] == payload["window_count"] == 3

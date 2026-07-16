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


def fixture_sources(tmp_path: Path) -> dict[str, Path]:
    manifest_path = tmp_path / "manifest.json"
    phase1 = tmp_path / "phase1_v1"
    gate = tmp_path / "gate_v1"
    phase1.mkdir()
    gate.mkdir()

    manifest_bytes = _pretty(manifest())
    manifest_path.write_bytes(manifest_bytes)
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in phase1_rows())
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
            "status": "tokenizer_only_preflight_complete",
            "qrels_opened": False,
            "candidates_sha256": _sha256(candidate_bytes),
            "model_materialization_receipt": str(model_receipt),
            "model_materialization_receipt_sha256": model_receipt_sha256,
        }
    )
    (phase1 / "preflight.json").write_bytes(phase1_preflight_bytes)
    phase1_windows = b'{"window":1}\n'
    phase1_scores = b'{"score":1}\n'
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
                "candidate_rows": 4,
                "candidates_sha256": _sha256(candidate_bytes),
                "manifest_sha256": _sha256(manifest_bytes),
                "preflight_sha256": _sha256(phase1_preflight_bytes),
                "windows_sha256": _sha256(phase1_windows),
                "scores_sha256": _sha256(phase1_scores),
                "model_materialization_receipt": str(model_receipt),
                "model_materialization_receipt_sha256": model_receipt_sha256,
                "model": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
            }
        )
    )
    gates_bytes = _pretty({"gates": accepted_gates()})
    (gate / "gates.json").write_bytes(gates_bytes)
    (gate / "summary.json").write_bytes(
        _pretty(
            {
                "schema_version": "deep-facet-candidate-gate-v1",
                "status": "complete",
                "qrels_opened": False,
                "accepted_facet_count": 2,
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
    result = create_preflight(
        fixture_sources(tmp_path),
        tmp_path / "out",
        tokenizer=WordTokenizer(),
        cache=cache,
    )
    assert result["status"] == "tokenizer_only_preflight_complete"
    assert result["model_constructed"] is False
    assert result["network_access_supported"] is False
    assert result["summary"]["query_document_pair_count"] == 4
    assert result["summary"]["cache_hit_window_count"] == 0
    assert result["summary"]["cache_miss_window_count"] == 4
    assert (tmp_path / "out" / "candidates.jsonl").exists()
    assert (tmp_path / "out" / "windows.jsonl").exists()
    assert (tmp_path / "out" / "preflight.json").read_bytes() == _pretty(result)


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
            "out",
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
            tmp_path / "out",
            tokenizer=WordTokenizer(),
            cache=FakeCache(tmp_path / "cache.jsonl"),
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
        tmp_path / "out",
        cache=FakeCache(tmp_path / "cache.jsonl"),
    )

    assert captured["auto_tokenizer_cls"] is module.TokenizerOnlyAuto

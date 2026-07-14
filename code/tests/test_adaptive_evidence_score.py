from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.adaptive_evidence_score import (
    build_score_candidates,
    build_score_preflight,
    load_score_contract,
    persist_score_preflight,
    run_score_preflight,
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

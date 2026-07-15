from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_contract as contract_module
from trec_rag.adaptive_obligation_v2_contract import (
    PILOT_TOPIC_IDS,
    _publish_v2_contract,
    _verify_fixture_v2_contract,
    build_v2_contract,
    document_fold,
    reject_protected_before_access,
    split_exact_units,
    verify_v2_contract,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compact(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _documents_for(topic_id: str) -> list[dict[str, object]]:
    by_fold: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    candidate = 0
    while any(len(rows) < 10 for rows in by_fold.values()):
        document_id = f"{topic_id}-document-{candidate:03d}"
        fold = document_fold(topic_id, document_id)
        if len(by_fold[fold]) < 10:
            text = f"Document {document_id}. Café evidence! Final fragment"
            by_fold[fold].append(
                {
                    "schema_version": "deep-facet-candidate-union-v1",
                    "topic_id": topic_id,
                    "document_id": document_id,
                    "fold": fold,
                    "union_order": candidate + 1,
                    "text": text,
                    "text_sha256": _sha(text),
                    "provenance": [],
                }
            )
        candidate += 1
    return [*by_fold[0], *by_fold[1]]


def _authenticated_fixture(*, mode: str = "fixture_only") -> dict[str, object]:
    documents = [
        row
        for topic_id in PILOT_TOPIC_IDS
        for row in _documents_for(topic_id)
    ]
    obligations: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for topic_index, topic_id in enumerate(PILOT_TOPIC_IDS):
        obligations.append(
            {
                "topic_id": topic_id,
                "obligation_id": f"{topic_id}:broad",
                "kind": "broad",
                "manifest_order": -1,
                "text": f"Broad {topic_id}",
                "query": f"Broad query {topic_id}",
            }
        )
        local_documents = [
            row for row in documents if row["topic_id"] == topic_id
        ]
        for local_parent in range(6):
            order = topic_index * 6 + local_parent
            parent_id = f"{topic_id}-parent-{local_parent}"
            obligations.append(
                {
                    "topic_id": topic_id,
                    "obligation_id": parent_id,
                    "source_facet_id": parent_id,
                    "kind": "o0",
                    "parent_id": None,
                    "manifest_order": order,
                    "text": f"Obligation {order} for {topic_id}.",
                    "query": f"Narrative {topic_id}\n\nExplicit obligation:\nObligation {order}.",
                }
            )
            for document_index, document in enumerate(local_documents):
                document_id = str(document["document_id"])
                window_text = (
                    f"Evidence for {parent_id} and {document_id}.\n"
                    "- First exact item\n"
                    "2) Café conclusion! short final"
                )
                query = f"Frozen query for {parent_id}"
                window_id = f"window-{parent_id}-{document_index:02d}"
                score_rows.append(
                    {
                        "topic_id": topic_id,
                        "variant": parent_id,
                        "document_id": document_id,
                        "document_sha256": document["text_sha256"],
                        "window_id": window_id,
                        "window_text": window_text,
                        "window_sha256": _sha(window_text),
                        "query": query,
                        "query_sha256": _sha(query),
                        "document_start_token": document_index,
                        "document_end_token": document_index + 20,
                        "score": float(100 - document_index),
                    }
                )
            # A weaker later window proves within-document reduction.
            first = local_documents[0]
            weak_text = f"Weaker evidence for {parent_id}."
            weak_query = f"Frozen query for {parent_id}"
            score_rows.append(
                {
                    "topic_id": topic_id,
                    "variant": parent_id,
                    "document_id": first["document_id"],
                    "document_sha256": first["text_sha256"],
                    "window_id": f"weak-{parent_id}",
                    "window_text": weak_text,
                    "window_sha256": _sha(weak_text),
                    "query": weak_query,
                    "query_sha256": _sha(weak_query),
                    "document_start_token": 500,
                    "document_end_token": 510,
                    "score": -1000.0,
                }
            )

    bindings: dict[str, object] = {
        "mode": mode,
        "contract_summary_sha256": "c" * 64,
        "base_score_receipt_sha256": "d" * 64,
        "document_count": len(documents),
        "broad_obligation_count": 4,
        "o0_obligation_count": 24,
        "window_count": len(score_rows),
        "pair_count": len(
            {(str(row["query"]), str(row["window_text"])) for row in score_rows}
        ),
        "shard_count": 28,
    }
    if mode == "authenticated_paths":
        bindings.update(
            {
                "contract_dir": "/authenticated/contract",
                "base_scores_dir": "/authenticated/scores",
            }
        )
    return {
        "topic_ids": list(PILOT_TOPIC_IDS),
        "obligations": obligations,
        "documents": documents,
        "score_rows": score_rows,
        "source_bindings": bindings,
    }


def _artifact(path: str, content: bytes, rows: int) -> dict[str, object]:
    return {
        "path": path,
        "rows": rows,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reseal(output: Path, name: str, rows: list[dict[str, object]]) -> None:
    content = b"".join(_compact(row) for row in rows)
    (output / name).write_bytes(content)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"][name] = _artifact(name, content, len(rows))
    manifest_content = _pretty(manifest)
    manifest_path.write_bytes(manifest_content)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["artifacts"][name] = _artifact(name, content, len(rows))
    receipt["artifacts"]["manifest.json"] = _artifact(
        "manifest.json", manifest_content, 1
    )
    receipt_path.write_bytes(_pretty(receipt))


@pytest.mark.parametrize("topic_id", ["144", "213", "224", "407", "515"])
def test_protected_topic_fails_before_any_loader(topic_id: str) -> None:
    touched: list[str] = []
    with pytest.raises(ValueError, match="protected"):
        reject_protected_before_access(
            [topic_id],
            source_loader=lambda: touched.append("source"),
        )
    assert touched == []


def test_unit_ids_bind_exact_source_span() -> None:
    source = "Technology helps access. It can also exclude people."
    units = split_exact_units(
        topic_id="219",
        parent_id="219-positive",
        fold=0,
        document_id="d1",
        window_id="w1",
        text=source,
    )
    assert [row["text"] for row in units] == [
        "Technology helps access.",
        "It can also exclude people.",
    ]
    assert all(source[row["start"] : row["end"]] == row["text"] for row in units)
    assert len({row["unit_id"] for row in units}) == 2


def test_units_cover_punctuation_lists_unicode_and_short_final_exactly() -> None:
    source = "  What?!  Café works… yes.\n - Première item\n2) 第二 item!\nshort final  "
    units = split_exact_units(
        topic_id="219",
        parent_id="219-positive",
        fold=1,
        document_id="d1",
        window_id="w1",
        text=source,
    )
    assert [row["text"] for row in units] == [
        "What?!",
        "Café works… yes.",
        "- Première item",
        "2) 第二 item!",
        "short final",
    ]
    for row in units:
        start, end = int(row["start"]), int(row["end"])
        assert 0 <= start < end <= len(source)
        assert source[start:end] == row["text"]
        assert row["text_sha256"] == _sha(str(row["text"]))


@pytest.mark.parametrize("source", ["", "   ", "\n\t\n"])
def test_units_skip_empty_whitespace_only_source(source: str) -> None:
    assert split_exact_units(
        topic_id="219",
        parent_id="219-positive",
        fold=0,
        document_id="d1",
        window_id="w1",
        text=source,
    ) == []


def test_contract_has_exact_parent_fold_reservoirs_and_no_discovery_source() -> None:
    contract = build_v2_contract(_authenticated_fixture())
    assert len(contract["parents"]) == 24
    assert len(contract["reservoirs"]) == 48
    assert all(row["document_count"] == 10 for row in contract["reservoirs"])
    assert all(len(set(row["document_ids"])) == 10 for row in contract["reservoirs"])
    assert "discovery" not in contract["source_bindings"]
    assert "discovery" not in json.dumps(contract["source_bindings"])


def test_build_is_deterministic_under_every_input_reordering() -> None:
    source = _authenticated_fixture()
    reversed_source = copy.deepcopy(source)
    reversed_source["topic_ids"] = list(reversed(reversed_source["topic_ids"]))
    for name in ("obligations", "documents", "score_rows"):
        reversed_source[name] = list(reversed(reversed_source[name]))
    assert build_v2_contract(source) == build_v2_contract(reversed_source)


def test_parent_local_score_offsets_do_not_change_reservoirs() -> None:
    source = _authenticated_fixture()
    shifted = copy.deepcopy(source)
    for row in shifted["score_rows"]:
        parent_index = int(str(row["variant"]).rsplit("-", 1)[-1])
        row["score"] = float(row["score"]) + parent_index * 1_000_000.0
    def identities(value: dict[str, object]) -> list[tuple[object, ...]]:
        return [
            (
                row["topic_id"],
                row["parent_id"],
                row["fold"],
                row["document_ids"],
                row["window_ids"],
            )
            for row in value["reservoirs"]
        ]

    assert identities(build_v2_contract(source)) == identities(build_v2_contract(shifted))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_contract_rejects_nonfinite_scores(value: float) -> None:
    source = _authenticated_fixture()
    source["score_rows"][0]["score"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        build_v2_contract(source)


def test_contract_rejects_duplicate_documents() -> None:
    source = _authenticated_fixture()
    source["documents"].append(copy.deepcopy(source["documents"][0]))
    source["source_bindings"]["document_count"] += 1
    with pytest.raises(ValueError, match="duplicate"):
        build_v2_contract(source)


def test_contract_rejects_insufficient_parent_fold_reservoir() -> None:
    source = _authenticated_fixture()
    target_parent = "219-parent-0"
    fold_zero_ids = {
        row["document_id"]
        for row in source["documents"]
        if row["topic_id"] == "219" and row["fold"] == 0
    }
    source["score_rows"] = [
        row
        for row in source["score_rows"]
        if not (
            row["variant"] == target_parent
            and row["document_id"] in fold_zero_ids
        )
    ]
    source["source_bindings"]["window_count"] = len(source["score_rows"])
    source["source_bindings"]["pair_count"] = len(source["score_rows"])
    with pytest.raises(ValueError, match="fewer than ten"):
        build_v2_contract(source)


@pytest.mark.parametrize("fold", [-1, 2, "0"])
def test_contract_rejects_wrong_document_folds(fold: object) -> None:
    source = _authenticated_fixture()
    source["documents"][0]["fold"] = fold
    with pytest.raises(ValueError, match="fold"):
        build_v2_contract(source)


def test_contract_rejects_wrong_parent_count() -> None:
    source = _authenticated_fixture()
    source["obligations"] = [
        row
        for row in source["obligations"]
        if row.get("obligation_id") != "219-parent-0"
    ]
    with pytest.raises(ValueError, match="24"):
        build_v2_contract(source)


def test_contract_requires_one_broad_obligation_per_pilot_topic() -> None:
    source = _authenticated_fixture()
    broad = next(
        row
        for row in source["obligations"]
        if row["obligation_id"] == "84:broad"
    )
    broad["topic_id"] = "219"
    with pytest.raises(ValueError, match="broad"):
        build_v2_contract(source)


@pytest.mark.parametrize("mode", ["protected", "wrong_count"])
def test_authenticated_preflight_fails_before_large_source_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    contract = tmp_path / "source-contract"
    scores = tmp_path / "source-scores"
    contract.mkdir()
    scores.mkdir()
    topic_ids = list(PILOT_TOPIC_IDS)
    document_count = 8_114
    if mode == "protected":
        topic_ids[0] = "144"
    else:
        document_count -= 1
    (contract / "manifest.json").write_bytes(
        _compact(
            {
                "topic_ids": topic_ids,
                "document_count": document_count,
                "broad_obligation_count": 4,
                "o0_obligation_count": 24,
                "qrels_opened": False,
            }
        )
    )
    (contract / "summary.json").write_bytes(
        _compact(
            {
                "schema_version": "adaptive-evidence-contract-v1",
                "topic_ids": list(PILOT_TOPIC_IDS),
                "document_count": 8_114,
                "broad_obligation_count": 4,
                "o0_obligation_count": 24,
                "protected_topic_count": 0,
                "qrels_opened": False,
            }
        )
    )
    (scores / "receipt.json").write_bytes(_compact({"shards": []}))
    touched: list[str] = []
    monkeypatch.setattr(
        contract_module,
        "load_score_contract",
        lambda path: touched.append("documents"),
    )
    monkeypatch.setattr(
        contract_module,
        "verify_local_scoring",
        lambda path: touched.append("scores"),
    )
    with pytest.raises(ValueError, match="protected|canonical counts"):
        contract_module.load_authenticated_v2_sources(contract, scores)
    assert touched == []


def _preflight_receipt_shards(mode: str) -> list[dict[str, object]]:
    topics = [PILOT_TOPIC_IDS[index % 4] for index in range(28)]
    if mode == "missing_topic":
        topics = ["219"] * 28
    obligations = [f"{topic}-queue-{index}" for index, topic in enumerate(topics)]
    if mode == "duplicate_queue":
        obligations[1] = obligations[0]
        topics[1] = topics[0]
    base, remainder = divmod(98_053, 28)
    return [
        {
            "topic_id": topic,
            "obligation_id": obligations[index],
            "path": f"shards/shard-{index}.jsonl",
            "rows": base + (index < remainder),
            "bytes": 1,
            "sha256": "a" * 64,
        }
        for index, topic in enumerate(topics)
    ]


@pytest.mark.parametrize("mode", ["missing_topic", "duplicate_queue"])
def test_score_receipt_preflight_requires_exact_topics_and_unique_queues_before_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    contract = tmp_path / "source-contract"
    scores = tmp_path / "source-scores"
    contract.mkdir()
    scores.mkdir()
    (contract / "manifest.json").write_bytes(
        _compact(
            {
                "schema_version": "adaptive-evidence-contract-v1",
                "status": "complete",
                "topic_ids": list(PILOT_TOPIC_IDS),
                "document_count": 8_114,
                "broad_obligation_count": 4,
                "o0_obligation_count": 24,
                "qrels_opened": False,
            }
        )
    )
    (contract / "summary.json").write_bytes(
        _compact(
            {
                "schema_version": "adaptive-evidence-contract-v1",
                "status": "complete",
                "topic_ids": list(PILOT_TOPIC_IDS),
                "document_count": 8_114,
                "broad_obligation_count": 4,
                "o0_obligation_count": 24,
                "protected_topic_count": 0,
                "qrels_opened": False,
            }
        )
    )
    shards = _preflight_receipt_shards(mode)
    (scores / "receipt.json").write_bytes(
        _compact(
            {
                "completed_window_count": 98_053,
                "unique_pair_count": 96_911,
                "shard_count": 28,
                "shards": shards,
                "qrels_opened": False,
                "network_call_count": 0,
                "retrieval_call_count": 0,
                "hosted_inference_call_count": 0,
                "paid_call_count": 0,
                "external_cost_usd": 0.0,
            }
        )
    )
    touched: list[str] = []

    def forbidden(_path: Path) -> object:
        touched.append("large")
        raise AssertionError("large callback touched")

    monkeypatch.setattr(contract_module, "load_score_contract", forbidden)
    monkeypatch.setattr(contract_module, "verify_local_scoring", forbidden)
    with pytest.raises(ValueError, match="topic coverage|duplicate.*queue"):
        contract_module.load_authenticated_v2_sources(contract, scores)
    assert touched == []


def _write_snapshot_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, list[dict[str, object]]]]:
    monkeypatch.setattr(contract_module, "EXPECTED_DOCUMENT_COUNT", 4)
    monkeypatch.setattr(contract_module, "EXPECTED_O0_COUNT", 0)
    monkeypatch.setattr(contract_module, "EXPECTED_WINDOW_COUNT", 4)
    monkeypatch.setattr(contract_module, "EXPECTED_PAIR_COUNT", 4)
    monkeypatch.setattr(contract_module, "EXPECTED_SHARD_COUNT", 4)
    contract = tmp_path / "source-contract"
    scores = tmp_path / "source-scores"
    shards_root = scores / "shards"
    contract.mkdir()
    shards_root.mkdir(parents=True)
    manifest = {
        "schema_version": "adaptive-evidence-contract-v1",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "document_count": 4,
        "broad_obligation_count": 4,
        "o0_obligation_count": 0,
        "qrels_opened": False,
    }
    obligations = [
        {
            "topic_id": topic_id,
            "obligation_id": f"{topic_id}:broad",
            "kind": "broad",
            "manifest_order": -1,
            "text": f"Broad {topic_id}",
            "query": f"Query {topic_id}",
        }
        for topic_id in PILOT_TOPIC_IDS
    ]
    documents: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for topic_id in PILOT_TOPIC_IDS:
        document_id = f"document-{topic_id}"
        text = f"Original document {topic_id}."
        documents.append(
            {
                "topic_id": topic_id,
                "document_id": document_id,
                "fold": document_fold(topic_id, document_id),
                "union_order": 1,
                "text": text,
                "text_sha256": _sha(text),
            }
        )
        window_text = f"Original window {topic_id}."
        query = f"Query {topic_id}"
        score_rows.append(
            {
                "topic_id": topic_id,
                "variant": f"{topic_id}:broad",
                "document_id": document_id,
                "document_sha256": _sha(text),
                "window_id": f"window-{topic_id}",
                "window_text": window_text,
                "window_sha256": _sha(window_text),
                "query": query,
                "query_sha256": _sha(query),
                "document_start_token": 0,
                "document_end_token": 4,
                "score": 1.0,
            }
        )
    artifact_rows = {
        "manifest.json": (manifest, 1),
        "obligations.jsonl": (obligations, len(obligations)),
        "documents.jsonl": (documents, len(documents)),
        "folds.jsonl": ([], 0),
    }
    artifact_hashes: dict[str, str] = {}
    for name, (rows, _count) in artifact_rows.items():
        content = _compact(rows) if name == "manifest.json" else b"".join(
            _compact(row) for row in rows
        )
        (contract / name).write_bytes(content)
        artifact_hashes[name] = hashlib.sha256(content).hexdigest()
    (contract / "summary.json").write_bytes(
        _compact(
            {
                "schema_version": "adaptive-evidence-contract-v1",
                "status": "complete",
                "topic_ids": list(PILOT_TOPIC_IDS),
                "document_count": 4,
                "broad_obligation_count": 4,
                "o0_obligation_count": 0,
                "protected_topic_count": 0,
                "qrels_opened": False,
                "artifact_sha256": artifact_hashes,
            }
        )
    )
    shard_bindings: list[dict[str, object]] = []
    for index, row in enumerate(score_rows):
        name = f"shard-{index}.jsonl"
        content = _compact(row)
        (shards_root / name).write_bytes(content)
        shard_bindings.append(
            {
                "topic_id": row["topic_id"],
                "obligation_id": row["variant"],
                "path": f"shards/{name}",
                "rows": 1,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    receipt = {
        "completed_window_count": 4,
        "unique_pair_count": 4,
        "shard_count": 4,
        "shards": shard_bindings,
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    (scores / "receipt.json").write_bytes(_compact(receipt))
    return contract, scores, {
        "obligations": obligations,
        "documents": documents,
        "score_rows": score_rows,
        "receipt": [receipt],
    }


def test_contract_document_swap_after_deep_verifier_is_not_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, scores, expected = _write_snapshot_sources(tmp_path, monkeypatch)

    def swap_contract(_root: Path) -> dict[str, object]:
        forged = [dict(row) for row in expected["documents"]]
        forged[0]["text"] = "Forged replacement."
        forged[0]["text_sha256"] = _sha("Forged replacement.")
        forged_bytes = b"".join(_compact(row) for row in forged)
        (contract / "documents.jsonl").write_bytes(forged_bytes)
        summary = json.loads((contract / "summary.json").read_bytes())
        summary["artifact_sha256"]["documents.jsonl"] = hashlib.sha256(
            forged_bytes
        ).hexdigest()
        (contract / "summary.json").write_bytes(_compact(summary))
        return {
            "schema_version": "adaptive-evidence-contract-v1",
            "obligations": expected["obligations"],
            "documents": forged,
        }

    monkeypatch.setattr(contract_module, "load_score_contract", swap_contract)
    monkeypatch.setattr(
        contract_module,
        "verify_local_scoring",
        lambda _root: expected["receipt"][0],
    )
    with pytest.raises(
        ValueError, match="changed after snapshot|deep-validated.*snapshot"
    ):
        contract_module.load_authenticated_v2_sources(contract, scores)


def test_score_shard_swap_after_deep_verifier_is_not_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, scores, expected = _write_snapshot_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contract_module,
        "load_score_contract",
        lambda _root: {
            "schema_version": "adaptive-evidence-contract-v1",
            "obligations": expected["obligations"],
            "documents": expected["documents"],
        },
    )

    def swap_scores(_root: Path) -> dict[str, object]:
        forged = dict(expected["score_rows"][0])
        forged["window_text"] = "Forged replacement window."
        forged["window_sha256"] = _sha(str(forged["window_text"]))
        forged_bytes = _compact(forged)
        (scores / "shards" / "shard-0.jsonl").write_bytes(forged_bytes)
        receipt = copy.deepcopy(expected["receipt"][0])
        receipt["shards"][0]["bytes"] = len(forged_bytes)
        receipt["shards"][0]["sha256"] = hashlib.sha256(forged_bytes).hexdigest()
        (scores / "receipt.json").write_bytes(_compact(receipt))
        return receipt

    monkeypatch.setattr(contract_module, "verify_local_scoring", swap_scores)
    with pytest.raises(
        ValueError, match="changed after snapshot|deep-validated.*snapshot"
    ):
        contract_module.load_authenticated_v2_sources(contract, scores)


def _return_while_original_directory_is_restored(
    root: Path, value: object
) -> object:
    original = root.with_name(f"{root.name}-original")
    root.rename(original)
    root.mkdir()
    try:
        return value
    finally:
        root.rmdir()
        original.rename(root)


def test_contract_validator_b_cannot_authorize_restored_snapshot_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, scores, expected = _write_snapshot_sources(tmp_path, monkeypatch)
    forged_documents = copy.deepcopy(expected["documents"])
    forged_documents[0]["text"] = "Deep-validated B document."
    forged_documents[0]["text_sha256"] = _sha("Deep-validated B document.")

    def validate_b_then_restore_a(root: Path) -> object:
        assert root == contract
        return _return_while_original_directory_is_restored(
            root,
            {
                "schema_version": "adaptive-evidence-contract-v1",
                "obligations": expected["obligations"],
                "documents": forged_documents,
            },
        )

    monkeypatch.setattr(
        contract_module, "load_score_contract", validate_b_then_restore_a
    )
    monkeypatch.setattr(
        contract_module,
        "verify_local_scoring",
        lambda _root: expected["receipt"][0],
    )
    with pytest.raises(ValueError, match="deep-validated contract.*snapshot"):
        contract_module.load_authenticated_v2_sources(contract, scores)


def test_score_validator_b_cannot_authorize_restored_snapshot_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, scores, expected = _write_snapshot_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contract_module,
        "load_score_contract",
        lambda _root: {
            "schema_version": "adaptive-evidence-contract-v1",
            "obligations": expected["obligations"],
            "documents": expected["documents"],
        },
    )
    forged_receipt = copy.deepcopy(expected["receipt"][0])
    forged_receipt["shards"][0]["sha256"] = "f" * 64

    def validate_b_then_restore_a(root: Path) -> object:
        assert root == scores
        return _return_while_original_directory_is_restored(root, forged_receipt)

    monkeypatch.setattr(
        contract_module, "verify_local_scoring", validate_b_then_restore_a
    )
    with pytest.raises(ValueError, match="deep-validated score receipt.*snapshot"):
        contract_module.load_authenticated_v2_sources(contract, scores)


@pytest.mark.parametrize(
    "field",
    ["restored_cache_pair_count", "resumed_shard_count"],
)
@pytest.mark.parametrize("value", [1, True, "0", 0.0, None])
def test_captured_legacy_resume_field_must_be_exact_integer_zero_before_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    contract, scores, expected = _write_snapshot_sources(tmp_path, monkeypatch)
    captured_receipt = expected["receipt"][0]
    captured_receipt[field] = value
    (scores / "receipt.json").write_bytes(_compact(captured_receipt))
    monkeypatch.setattr(
        contract_module,
        "load_score_contract",
        lambda _root: {
            "schema_version": "adaptive-evidence-contract-v1",
            "obligations": expected["obligations"],
            "documents": expected["documents"],
        },
    )
    deep_receipt = copy.deepcopy(captured_receipt)
    deep_receipt.pop(field)

    def validate_b_then_restore_a(root: Path) -> object:
        assert root == scores
        return _return_while_original_directory_is_restored(root, deep_receipt)

    monkeypatch.setattr(
        contract_module, "verify_local_scoring", validate_b_then_restore_a
    )
    with pytest.raises(ValueError, match="legacy.*integer zero"):
        contract_module.load_authenticated_v2_sources(contract, scores)


@pytest.mark.parametrize(
    "kind",
    ["declared_count", "pair_count", "document_hash", "window_hash", "query_hash"],
)
def test_contract_rejects_source_count_and_hash_tampering(kind: str) -> None:
    source = _authenticated_fixture()
    if kind == "declared_count":
        source["source_bindings"]["window_count"] += 1
    elif kind == "pair_count":
        source["source_bindings"]["pair_count"] -= 1
    elif kind == "document_hash":
        source["documents"][0]["text_sha256"] = "0" * 64
    elif kind == "window_hash":
        source["score_rows"][0]["window_sha256"] = "0" * 64
    else:
        source["score_rows"][0]["query_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="count|hash"):
        build_v2_contract(source)


def test_selected_units_validate_against_immutable_window_text() -> None:
    contract = build_v2_contract(_authenticated_fixture())
    windows = {
        (row["parent_id"], row["fold"], document["document_id"], document["window_id"]): document["window_text"]
        for row in contract["reservoirs"]
        for document in row["documents"]
    }
    for unit in contract["units"]:
        key = (
            unit["parent_id"], unit["fold"], unit["document_id"], unit["window_id"]
        )
        text = windows[key]
        assert text[unit["start"] : unit["end"]] == unit["text"]


def test_publish_verify_and_create_only_retry(tmp_path: Path) -> None:
    source = _authenticated_fixture()
    output = tmp_path / "contract"
    receipt = _publish_v2_contract(build_v2_contract(source), output)
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json", "parents.jsonl", "reservoirs.jsonl", "units.jsonl", "receipt.json"
    }
    assert receipt["parent_count"] == 24
    assert receipt["reservoir_count"] == 48
    assert _verify_fixture_v2_contract(output, source=source) == receipt
    with pytest.raises(FileExistsError, match="create-only"):
        _publish_v2_contract(build_v2_contract(source), output)


@pytest.mark.parametrize("mode", ["unexpected", "partial", "child_symlink", "root_symlink"])
def test_verifier_rejects_unsafe_or_partial_inventory(tmp_path: Path, mode: str) -> None:
    source = _authenticated_fixture()
    output = tmp_path / "contract"
    _publish_v2_contract(build_v2_contract(source), output)
    target = output
    if mode == "unexpected":
        (output / "extra.json").write_text("{}\n", encoding="utf-8")
    elif mode == "partial":
        (output / "units.jsonl").unlink()
    elif mode == "child_symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_text("{}\n", encoding="utf-8")
        (output / "units.jsonl").unlink()
        (output / "units.jsonl").symlink_to(outside)
    else:
        target = tmp_path / "linked"
        target.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="inventory"):
        _verify_fixture_v2_contract(target, source=source)


def test_atomic_publication_failure_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _authenticated_fixture()
    output = tmp_path / "contract"
    original = contract_module._write_fsynced
    calls = 0

    def fail_third(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated write failure")
        original(path, content)

    monkeypatch.setattr(contract_module, "_write_fsynced", fail_third)
    with pytest.raises(OSError, match="simulated"):
        _publish_v2_contract(build_v2_contract(source), output)
    assert not output.exists()
    assert list(tmp_path.glob(".contract.staging-*")) == []


def test_verifier_reconstructs_and_rejects_fully_resealed_unit_tamper(
    tmp_path: Path,
) -> None:
    source = _authenticated_fixture()
    output = tmp_path / "contract"
    _publish_v2_contract(build_v2_contract(source), output)
    units = [json.loads(line) for line in (output / "units.jsonl").read_text().splitlines()]
    units[0]["text"] = "forged but resealed"
    units[0]["text_sha256"] = _sha("forged but resealed")
    _reseal(output, "units.jsonl", units)
    with pytest.raises(ValueError, match="reconstruction"):
        _verify_fixture_v2_contract(output, source=source)


def test_public_verifier_reconstructs_only_from_authenticated_bound_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _authenticated_fixture(mode="authenticated_paths")
    output = tmp_path / "contract"
    _publish_v2_contract(build_v2_contract(source), output)
    calls: list[tuple[Path, Path]] = []

    def load(contract_dir: Path, scores_dir: Path) -> dict[str, object]:
        calls.append((contract_dir, scores_dir))
        return source

    monkeypatch.setattr(contract_module, "load_authenticated_v2_sources", load)
    assert verify_v2_contract(output)["status"] == "complete"
    assert calls == [(Path("/authenticated/contract"), Path("/authenticated/scores"))]


def test_output_receipt_has_zero_task_safety_counters(tmp_path: Path) -> None:
    output = tmp_path / "contract"
    receipt = _publish_v2_contract(
        build_v2_contract(_authenticated_fixture()), output
    )
    assert receipt["qrels_opened"] is False
    for name in (
        "network_call_count", "retrieval_call_count", "hosted_inference_call_count",
        "paid_call_count", "model_load_count", "tokenizer_load_count", "inference_count",
    ):
        assert receipt[name] == 0
    assert receipt["external_cost_usd"] == 0.0

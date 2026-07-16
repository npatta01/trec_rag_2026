from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.tethered_facet_soft_coverage_evaluate import (
    evaluate_frozen_proxy,
    evaluate_proxy,
    normalized_recall_auc,
)


TOPICS = ("219", "72", "300", "84")
ARMS = ("RRF", "NARRATIVE", "FIXED-O0", "TETHERED-DUAL", "TETHERED-DUAL-NR", "RRF100-TETHERED-DUAL")


def _rankings() -> dict[str, dict[str, list[dict[str, object]]]]:
    base = {
        "219": ["a", "b", "c", "d"],
        "72": ["e", "f", "g", "h"],
        "300": ["i", "j", "k", "l"],
        "84": ["m", "n", "o", "p"],
    }
    result: dict[str, dict[str, list[dict[str, object]]]] = {}
    for topic, documents in base.items():
        result[topic] = {}
        for ordinal, arm in enumerate(ARMS):
            order = documents[ordinal % len(documents) :] + documents[: ordinal % len(documents)]
            result[topic][arm] = [
                {
                    "document_id": document,
                    "coverage_facet": f"{topic}-facet" if arm.startswith("TETHERED") and rank == 1 else None,
                    "coverage_bonus": 0.1 if arm.startswith("TETHERED") and rank == 1 else 0.0,
                }
                for rank, document in enumerate(order, start=1)
            ]
    return result


def _qrels() -> dict[str, dict[str, int]]:
    return {
        "219": {"a": 3, "b": 1, "d": 2, "outside": 4},
        "72": {"e": 2, "g": 3},
        "300": {"j": 2},
        "84": {"m": 1, "o": 4},
    }


def _provenance() -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        topic: {
            document: ([{"family": "facet", "facet_id": f"{topic}-facet"}] if index % 2 else [{"family": "original"}])
            for index, document in enumerate(documents)
        }
        for topic, documents in {
            "219": ["a", "b", "c", "d"], "72": ["e", "f", "g", "h"],
            "300": ["i", "j", "k", "l"], "84": ["m", "n", "o", "p"],
        }.items()
    }


def test_normalized_recall_auc_uses_the_complete_topic_depth() -> None:
    assert normalized_recall_auc(["r", "x", "r2"], {"r", "r2"}) == pytest.approx((0.5 + 0.5 + 1.0) / 3.0)


def test_full_depth_recall_is_identical_for_complete_permutations() -> None:
    result = evaluate_proxy(_rankings(), _qrels(), _provenance(), depths=(1, 2, 4))
    full = {arm: values["aggregate"]["binary_recall_full"] for arm, values in result["arms"].items()}
    assert len(set(full.values())) == 1


def test_proxy_reports_requested_metrics_and_does_not_claim_nugget_coverage() -> None:
    result = evaluate_proxy(_rankings(), _qrels(), _provenance(), depths=(1, 2, 4))
    topic = result["arms"]["TETHERED-DUAL"]["per_topic"]["219"]
    assert topic["binary_recall@1"] == pytest.approx(1 / 3)
    assert topic["graded_recall@1"] == pytest.approx(3 / 25)
    assert topic["relevant_count@1"] == 1
    assert topic["judged_rate@1"] == 1
    assert topic["facet_only_relevant_retained@4"] == 1
    assert topic["ndcg@1"] is not None
    assert "binary_recall@1_delta_vs_RRF" in result["per_topic_deltas"]["219"]["TETHERED-DUAL"]
    assert result["coverage_proxy"]["is_true_nugget_coverage"] is False
    assert "does not prove" in result["coverage_proxy"]["caveat"]
    assert result["coverage_proxy"]["qrels_positive_facet_attribution"]["TETHERED-DUAL"]["219"]["219-facet"] == 1


def test_grade_below_two_is_not_relevant_and_auc_uses_all_qrels_relevant() -> None:
    result = evaluate_proxy(_rankings(), _qrels(), _provenance(), depths=(1, 4))
    topic = result["arms"]["RRF"]["per_topic"]["219"]
    assert topic["total_relevant"] == 3
    assert topic["binary_recall_full"] == pytest.approx(2 / 3)
    assert topic["recall_auc"] == pytest.approx((1 / 3 + 1 / 3 + 1 / 3 + 2 / 3) / 4)


def test_aggregate_recall_retention_and_judged_rate_use_pooled_denominators() -> None:
    rankings, provenance = _rankings(), _provenance()
    for topic, keep in (("300", {"i", "j"}), ("84", {"m", "n"})):
        for arm in ARMS:
            rankings[topic][arm] = [row for row in rankings[topic][arm] if row["document_id"] in keep]
        provenance[topic] = {document: value for document, value in provenance[topic].items() if document in keep}
    result = evaluate_proxy(rankings, _qrels(), provenance, depths=(1, 4))
    aggregate = result["arms"]["RRF"]["aggregate"]
    assert aggregate["binary_recall_full"] == pytest.approx(5 / 7)
    assert aggregate["graded_recall_full"] == pytest.approx(23 / 53)
    assert aggregate["judged_rate_full"] == pytest.approx(7 / 12)
    assert aggregate["binary_recall_full_aggregation"] == {
        "method": "pooled_micro", "numerator": 5, "denominator": 7, "value": pytest.approx(5 / 7)
    }
    tethered = result["arms"]["TETHERED-DUAL"]["aggregate"]
    assert tethered["facet_only_relevant_retention@1"] == pytest.approx(1 / 2)
    assert tethered["facet_only_relevant_retention@1_aggregation"]["method"] == "pooled_micro"
    assert aggregate["recall_auc_aggregation"]["method"] == "macro_topic_mean"
    assert aggregate["ndcg@10_aggregation"]["method"] == "macro_topic_mean"


def test_qrels_positive_facet_attribution_requires_positive_coverage_gain() -> None:
    rankings = _rankings()
    rankings["219"]["TETHERED-DUAL"][0]["coverage_bonus"] = 0.0
    result = evaluate_proxy(rankings, _qrels(), _provenance(), depths=(1, 4))
    assert result["coverage_proxy"]["qrels_positive_facet_attribution"]["TETHERED-DUAL"]["219"] == {}


def test_evaluate_proxy_rejects_protected_or_incomplete_scope() -> None:
    rankings = _rankings()
    rankings["144"] = rankings.pop("219")
    with pytest.raises(ValueError, match="protected topic 144"):
        evaluate_proxy(rankings, _qrels(), _provenance(), depths=(1, 4))
    rankings = _rankings()
    rankings["219"]["RRF"].pop()
    with pytest.raises(ValueError, match="complete permutation"):
        evaluate_proxy(rankings, _qrels(), _provenance(), depths=(1, 4))


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def test_frozen_evaluation_verifies_freeze_before_opening_qrels_and_writes_create_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, qrels_path, union_path, output = tmp_path / "freeze", tmp_path / "qrels.jsonl", tmp_path / "union.jsonl", tmp_path / "evaluation"
    freeze.mkdir()
    ranking_rows = [{"topic_id": topic, "arm": arm, "rank": rank, **row} for topic, arms in _rankings().items() for arm, rows in arms.items() for rank, row in enumerate(rows, 1)]
    _jsonl(freeze / "rankings.jsonl", ranking_rows)
    ranking_bytes = (freeze / "rankings.jsonl").read_bytes()
    (freeze / "SEALED.json").write_text(json.dumps({
        "root_sha256": "a" * 64,
        "files": {"rankings.jsonl": {"bytes": len(ranking_bytes), "sha256": hashlib.sha256(ranking_bytes).hexdigest()}},
    }), encoding="utf-8")
    _jsonl(qrels_path, [{"topic_id": topic, "document_id": document, "grade": grade} for topic, rows in _qrels().items() for document, grade in rows.items()])
    _jsonl(union_path, [{"topic_id": topic, "document_id": document, "union_order": rank, "provenance": provenance} for topic, rows in _provenance().items() for rank, (document, provenance) in enumerate(rows.items(), 1)])
    qrels_hash = hashlib.sha256(qrels_path.read_bytes()).hexdigest()
    events: list[str] = []

    def verify(path: Path) -> dict[str, object]:
        events.append("verified")
        assert path == freeze
        return {"topic_ids": list(TOPICS), "arms": list(ARMS), "topic_summary": {topic: {"complete_permutations": {arm: True for arm in ARMS}} for topic in TOPICS}}

    original = Path.read_bytes
    ranking_reads = 0

    def read_bytes(path: Path) -> bytes:
        nonlocal ranking_reads
        if path == freeze / "rankings.jsonl":
            ranking_reads += 1
            if ranking_reads > 1:
                return b'{"mutated":true}\n'
        if path == qrels_path:
            assert events == ["verified"]
            events.append("qrels-opened")
        return original(path)

    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate.verify_soft_freeze", verify)
    summary = evaluate_frozen_proxy(freeze, qrels_path, union_path, output, expected_qrels_sha256=qrels_hash, depths=(1, 2, 4), reader=read_bytes)
    assert summary["status"] == "complete"
    assert events == ["verified", "qrels-opened"]
    assert ranking_reads == 1
    assert set(path.name for path in output.iterdir()) == {"metrics.json", "diagnostics.json", "summary.json", "input_bindings.json"}
    bindings = json.loads((output / "input_bindings.json").read_text(encoding="utf-8"))
    assert bindings["freeze"]["seal_root_sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        evaluate_frozen_proxy(freeze, qrels_path, union_path, output, expected_qrels_sha256=qrels_hash, depths=(1, 2, 4))


def test_frozen_evaluation_authenticates_projection_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, qrels_path, union_path = tmp_path / "freeze", tmp_path / "qrels.jsonl", tmp_path / "union.jsonl"
    freeze.mkdir(); qrels_path.write_bytes(b"not json\n"); union_path.write_bytes(b"")
    (freeze / "rankings.jsonl").write_bytes(b"")
    (freeze / "SEALED.json").write_text(json.dumps({
        "root_sha256": "a" * 64,
        "files": {"rankings.jsonl": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}},
    }), encoding="utf-8")
    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate.verify_soft_freeze", lambda _path: {})
    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate._load_verified_rankings", lambda *_args: (_rankings(), {"all": "verified"}))
    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate._load_union", lambda _content: _provenance())
    with pytest.raises(ValueError, match="qrels projection SHA-256"):
        evaluate_frozen_proxy(freeze, qrels_path, union_path, tmp_path / "out", expected_qrels_sha256="0" * 64, depths=(1,))


def test_frozen_evaluation_rejects_post_verify_ranking_mutation_before_qrels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, qrels_path, union_path = tmp_path / "freeze", tmp_path / "qrels.jsonl", tmp_path / "union.jsonl"
    freeze.mkdir(); qrels_path.write_bytes(b"must not open"); union_path.write_bytes(b"")
    rows = [{"topic_id": topic, "arm": arm, "rank": rank, **row} for topic, arms in _rankings().items() for arm, arm_rows in arms.items() for rank, row in enumerate(arm_rows, 1)]
    original = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    mutated_rows = list(rows)
    first, second = mutated_rows[0].copy(), mutated_rows[1].copy()
    first["document_id"], second["document_id"] = second["document_id"], first["document_id"]
    mutated_rows[0], mutated_rows[1] = first, second
    mutated = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in mutated_rows)
    (freeze / "rankings.jsonl").write_bytes(original)
    (freeze / "SEALED.json").write_text(json.dumps({
        "root_sha256": "b" * 64,
        "files": {"rankings.jsonl": {"bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}},
    }), encoding="utf-8")
    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate.verify_soft_freeze", lambda _path: {
        "topic_summary": {topic: {"complete_permutations": {arm: True for arm in ARMS}} for topic in TOPICS}
    })

    def reader(path: Path) -> bytes:
        if path == freeze / "rankings.jsonl":
            return mutated
        if path == qrels_path:
            raise AssertionError("qrels opened before ranking buffer authentication")
        return path.read_bytes()

    with pytest.raises(ValueError, match="ranking buffer differs from sealed artifact"):
        evaluate_frozen_proxy(freeze, qrels_path, union_path, tmp_path / "out", reader=reader)


def test_frozen_evaluation_rejects_coordinated_post_verify_seal_and_ranking_replacement_before_qrels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, qrels_path, union_path = tmp_path / "freeze", tmp_path / "qrels.jsonl", tmp_path / "union.jsonl"
    freeze.mkdir(); qrels_path.write_bytes(b"must not open"); union_path.write_bytes(b"")
    rows = [{"topic_id": topic, "arm": arm, "rank": rank, **row} for topic, arms in _rankings().items() for arm, arm_rows in arms.items() for rank, row in enumerate(arm_rows, 1)]
    original = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    mutated_rows = list(rows)
    first, second = mutated_rows[0].copy(), mutated_rows[1].copy()
    first["document_id"], second["document_id"] = second["document_id"], first["document_id"]
    mutated_rows[0], mutated_rows[1] = first, second
    mutated = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in mutated_rows)

    def seal_bytes(root: str, ranking_bytes: bytes) -> bytes:
        return json.dumps({
            "root_sha256": root,
            "files": {"rankings.jsonl": {"bytes": len(ranking_bytes), "sha256": hashlib.sha256(ranking_bytes).hexdigest()}},
        }, sort_keys=True).encode()

    original_seal = seal_bytes("a" * 64, original)
    mutated_seal = seal_bytes("b" * 64, mutated)
    (freeze / "rankings.jsonl").write_bytes(original)
    (freeze / "SEALED.json").write_bytes(original_seal)
    verified = False

    def verify(_path: Path) -> dict[str, object]:
        nonlocal verified
        verified = True
        return {"topic_summary": {topic: {"complete_permutations": {arm: True for arm in ARMS}} for topic in TOPICS}}

    def reader(path: Path) -> bytes:
        if path == freeze / "SEALED.json":
            return mutated_seal if verified else original_seal
        if path == freeze / "rankings.jsonl":
            return mutated if verified else original
        if path == qrels_path:
            raise AssertionError("qrels opened after coordinated freeze replacement")
        return path.read_bytes()

    monkeypatch.setattr("trec_rag.tethered_facet_soft_coverage_evaluate.verify_soft_freeze", verify)
    with pytest.raises(ValueError, match="seal changed during authenticated snapshot"):
        evaluate_frozen_proxy(freeze, qrels_path, union_path, tmp_path / "out", reader=reader)

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import trec_rag.tethered_facet_evaluate as module
from trec_rag.tethered_facet_evaluate import (
    TOPIC_IDS,
    decide,
    derive_novel_set,
    evaluate,
    evaluate_arm,
    load_projection,
    validate_protected_head,
    build_representatives,
)


class ExplodingRead:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("projection bytes were read before freeze verification")


def test_evaluation_accepts_projection_only_after_verified_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "_task3_snapshot",
        lambda _: (_ for _ in ()).throw(ValueError("bad seal")),
    )

    with pytest.raises(ValueError, match="bad seal"):
        evaluate(tmp_path / "freeze", tmp_path / "projection.jsonl", tmp_path / "out")


def test_top100_identity_is_a_hard_invariant() -> None:
    with pytest.raises(ValueError, match="top 100"):
        validate_protected_head(rrf=list(range(100)), arm=list(range(99)) + [999])


def _projection_bytes(
    topic_ids: tuple[str, ...] = TOPIC_IDS, *, grade: int = 2
) -> bytes:
    return b"".join(
        json.dumps(
            {"topic_id": topic, "document_id": f"{topic}-relevant", "grade": grade},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for topic in topic_ids
    )


def test_projection_requires_exact_contiguous_topics_and_rejects_protected(
    tmp_path: Path,
) -> None:
    projection = tmp_path / "projection.jsonl"
    projection.write_bytes(_projection_bytes())
    assert list(load_projection(projection)) == list(TOPIC_IDS)

    projection.write_bytes(_projection_bytes(tuple(reversed(TOPIC_IDS))))
    with pytest.raises(ValueError, match="exact topics in order"):
        load_projection(projection)

    projection.write_bytes(_projection_bytes((TOPIC_IDS[0], "144", *TOPIC_IDS[1:])))
    with pytest.raises(ValueError, match="protected topic"):
        load_projection(projection)

    projection.write_bytes(
        b'{"document_id":"d","grade":true,"topic_id":"219"}\n'
        + _projection_bytes(TOPIC_IDS[1:])
    )
    with pytest.raises(ValueError, match="line 1 is invalid"):
        load_projection(projection)


def test_projection_accepts_grade_four_and_rejects_out_of_range_grades(
    tmp_path: Path,
) -> None:
    projection = tmp_path / "projection.jsonl"
    projection.write_bytes(_projection_bytes(grade=4))
    assert all(
        set(topic_grades.values()) == {4}
        for topic_grades in load_projection(projection).values()
    )

    for grade in (-1, 5):
        projection.write_bytes(_projection_bytes(grade=grade))
        with pytest.raises(ValueError, match="line 1 is invalid"):
            load_projection(projection)


def test_derive_novel_set_uses_relevant_facet_candidates_absent_original_1000() -> None:
    qrels = {
        topic: {f"{topic}-novel": 2, f"{topic}-not-relevant": 1, f"{topic}-old": 3}
        for topic in TOPIC_IDS
    }
    facets = {
        topic: {f"{topic}-novel", f"{topic}-not-relevant", f"{topic}-old"}
        for topic in TOPIC_IDS
    }
    original = {topic: [f"{topic}-old"] for topic in TOPIC_IDS}

    result = derive_novel_set(qrels, facets, original, expected_total=4)

    assert result == {topic: {f"{topic}-novel"} for topic in TOPIC_IDS}
    with pytest.raises(ValueError, match="177"):
        derive_novel_set(qrels, facets, original)


def test_evaluate_arm_reports_metrics_novel_retention_and_basket_facet_yield() -> None:
    rankings = {}
    qrels = {}
    novel = {}
    rows = {}
    for topic in TOPIC_IDS:
        rankings[topic] = [f"{topic}-r", f"{topic}-n", f"{topic}-u"]
        qrels[topic] = {f"{topic}-r": 2, f"{topic}-n": 3}
        novel[topic] = {f"{topic}-n"}
        rows[topic] = [
            {"document_id": f"{topic}-r", "rank": 1, "source": "rrf_basket", "generating_facet": None},
            {"document_id": f"{topic}-n", "rank": 2, "source": "facet_basket", "generating_facet": f"{topic}-facet"},
            {"document_id": f"{topic}-u", "rank": 3, "source": "facet_basket", "generating_facet": f"{topic}-facet"},
        ]

    result = evaluate_arm(rankings, qrels, novel, ranking_rows=rows, depths=(2, 3))

    assert result["aggregate"]["recall@2"] == pytest.approx(1.0)
    assert result["aggregate"]["graded_recall@2"] == pytest.approx(1.0)
    assert result["aggregate"]["judged_rate@3"] == pytest.approx(2 / 3)
    assert result["aggregate"]["novel_retained@2"] == 4
    assert result["aggregate"]["novel_retention@2"] == pytest.approx(1.0)
    assert result["basket_contributions"]["facet_basket"]["relevant_count"] == 4
    assert result["facet_yield"]["219-facet"] == {
        "selected_count": 2,
        "relevant_count": 1,
        "relevant_yield": 0.5,
    }


def passing_evidence() -> dict[str, object]:
    metric = lambda r500, g500, r1000, g1000, j500=0.5, j1000=0.5: {
        "recall@500": r500,
        "graded_recall@500": g500,
        "recall@1000": r1000,
        "graded_recall@1000": g1000,
        "judged_rate@500": j500,
        "judged_rate@1000": j1000,
    }
    return {
        "top100_identity": True,
        "aggregate": {
            "RRF": metric(0.40, 0.41, 0.60, 0.61),
            "FACET-2B": metric(0.45, 0.46, 0.65, 0.66),
            "TETHERED-2B": {
                **metric(0.46, 0.46, 0.66, 0.67),
                "novel_retained@500": 89,
                "novel_retained@1000": 142,
            },
        },
        "per_topic": {
            topic: {
                "RRF": metric(0.40, 0.41, 0.60, 0.61),
                "FACET-2B": metric(0.45, 0.46, 0.65, 0.66),
                "TETHERED-2B": metric(0.44, 0.45, 0.66, 0.67),
            }
            for topic in TOPIC_IDS
        },
        "documented_basket_shortage": False,
    }


def break_guard(evidence: dict[str, object], guard: str) -> dict[str, object]:
    if guard == "top100_identity":
        evidence["top100_identity"] = False
    elif guard == "recall500":
        evidence["aggregate"]["TETHERED-2B"]["recall@500"] = 0.39
    elif guard == "novel500":
        evidence["aggregate"]["TETHERED-2B"]["novel_retained@500"] = 88
    elif guard == "recall1000":
        evidence["aggregate"]["TETHERED-2B"]["graded_recall@1000"] = 0.60
    elif guard == "novel1000":
        evidence["aggregate"]["TETHERED-2B"]["novel_retained@1000"] = 141
    elif guard == "per_topic_loss":
        evidence["per_topic"][TOPIC_IDS[0]]["TETHERED-2B"]["recall@500"] = 0.42
    elif guard == "judged_coverage":
        evidence["aggregate"]["TETHERED-2B"]["judged_rate@500"] = 0.44
    return evidence


def test_decision_pass_requires_every_frozen_guard() -> None:
    result = decide(passing_evidence())
    assert result["label"] == "mechanical_pass"
    assert all(result["guards"].values())


@pytest.mark.parametrize(
    "guard",
    [
        "top100_identity",
        "recall500",
        "novel500",
        "recall1000",
        "novel1000",
        "per_topic_loss",
        "judged_coverage",
    ],
)
def test_each_failed_guard_prevents_pass(guard: str) -> None:
    assert decide(break_guard(passing_evidence(), guard))["label"] != "mechanical_pass"


def test_only_coverage_or_basket_shortage_alone_is_inconclusive() -> None:
    coverage = decide(break_guard(passing_evidence(), "judged_coverage"))
    assert coverage["label"] == "inconclusive"

    shortage = passing_evidence()
    shortage["documented_basket_shortage"] = True
    assert decide(shortage)["label"] == "inconclusive"

    both = break_guard(passing_evidence(), "recall500")
    both["documented_basket_shortage"] = True
    assert decide(both)["label"] == "mechanical_fail"


def test_recall500_requires_strict_improvement_over_facet_on_one_metric() -> None:
    evidence = passing_evidence()
    evidence["aggregate"]["TETHERED-2B"]["recall@500"] = 0.45
    evidence["aggregate"]["TETHERED-2B"]["graded_recall@500"] = 0.46
    assert decide(evidence)["guards"]["recall500"] is False


def test_decision_thresholds_include_exact_point_zero_two_and_point_zero_five() -> None:
    evidence = passing_evidence()
    evidence["per_topic"][TOPIC_IDS[0]]["TETHERED-2B"]["recall@500"] = 0.43
    evidence["aggregate"]["TETHERED-2B"]["judged_rate@500"] = 0.45
    result = decide(evidence)
    assert result["guards"]["per_topic_loss"] is True
    assert result["guards"]["judged_coverage"] is True


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_evaluation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    head = {topic: [f"{topic}-head-{index:03d}" for index in range(100)] for topic in TOPIC_IDS}
    topic_inputs = {}
    ranking_rows = []
    for topic in TOPIC_IDS:
        novel = f"{topic}-novel"
        topic_inputs[topic] = {
            "topic_id": topic,
            "accepted_union": [*head[topic], novel],
            "rrf": head[topic],
            "dual": [*head[topic], novel],
            "facets": [{"facet_id": f"{topic}-facet", "scores": {novel: 1.0}}],
        }
        for arm in ("FACET-2B", "TETHERED-2B"):
            for rank, document_id in enumerate([*head[topic], novel], 1):
                ranking_rows.append(
                    {
                        "topic_id": topic,
                        "arm": arm,
                        "rank": rank,
                        "document_id": document_id,
                        "source": "protected_head" if rank <= 100 else "facet_basket",
                        "generating_facet": None if rank <= 100 else f"{topic}-facet",
                        "shortage": 0,
                    }
                )
    _write_json(
        freeze / "input_bindings.json",
        {"topic_inputs": {arm: topic_inputs for arm in ("FACET-2B", "TETHERED-2B")}},
    )
    (freeze / "rankings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ranking_rows),
        encoding="utf-8",
    )
    _write_json(freeze / "summary.json", {
        "topic_summary": {
            topic: {
                "facet_shortage_counts": {
                    arm: {f"{topic}-facet": 0}
                    for arm in ("FACET-2B", "TETHERED-2B")
                }
            }
            for topic in TOPIC_IDS
        }
    })
    _write_json(freeze / "SEALED.json", {
        "root_sha256": "a" * 64,
        "files": {
            name: {
                "bytes": len((freeze / name).read_bytes()),
                "sha256": hashlib.sha256((freeze / name).read_bytes()).hexdigest(),
            }
            for name in ("input_bindings.json", "rankings.jsonl", "summary.json")
        },
    })
    synthetic_snapshot = module.Task3Snapshot(
        freeze_dir=freeze,
        seal={"root_sha256": "a" * 64},
        seal_bytes=(freeze / "SEALED.json").read_bytes(),
        bindings=json.loads((freeze / "input_bindings.json").read_text()),
        summary=json.loads((freeze / "summary.json").read_text()),
        artifact_buffers={
            name: (freeze / name).read_bytes()
            for name in ("SEALED.json", "input_bindings.json", "rankings.jsonl", "summary.json")
        },
        producer_buffers={},
        producer_hashes={},
    )
    monkeypatch.setattr(module, "_task3_snapshot", lambda _path: synthetic_snapshot)

    prior_root = tmp_path / "prior"
    prior = prior_root / "evaluation_v1"
    prior.mkdir(parents=True)
    prior_freeze = prior_root / "freeze_v1"
    prior_freeze.mkdir()
    prior_seal_bytes = (json.dumps({"root_sha256": "b" * 64}, sort_keys=True) + "\n").encode()
    (prior_freeze / "SEALED.json").write_bytes(prior_seal_bytes)
    prior_rows = []
    projection = prior / "qrels_projection.jsonl"
    projection_bytes = b"".join(
        json.dumps(
            {"topic_id": topic, "document_id": f"{topic}-novel", "grade": 2},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for topic in TOPIC_IDS
    )
    projection.write_bytes(projection_bytes)
    for topic in TOPIC_IDS:
        for rank, document_id in enumerate([*head[topic], f"{topic}-novel"], 1):
            prior_rows.append(
                {
                    "topic_id": topic,
                    "arm": "RRF",
                    "rank": rank,
                    "document_id": document_id,
                    "original_rank": rank if rank <= 100 else None,
                    "best_facet_rank": 10**9 if rank <= 100 else 1,
                    "facet_percentiles": {} if rank <= 100 else {f"{topic}-facet": 1.0},
                }
            )
    (prior_freeze / "rankings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prior_rows),
        encoding="utf-8",
    )
    _write_json(
        prior / "qrels_access_receipt.json",
        {
            "schema_version": "deep-facet-candidate-evaluation-v1",
            "status": "qrels_access_boundary_crossed",
            "qrels_opened": True,
            "upstream_mutation_forbidden": True,
            "topic_ids": list(TOPIC_IDS),
            "qrels_projection_rows": len(TOPIC_IDS),
            "qrels_projection_sha256": hashlib.sha256(projection_bytes).hexdigest(),
            "seal_sha256": hashlib.sha256(prior_seal_bytes).hexdigest(),
            "seal_root_sha256": "b" * 64,
            "qrels_source_name": "injected_test_loader",
            "evaluator_code_sha256": "c" * 64,
        },
    )
    qrels = {topic: {f"{topic}-novel": 2} for topic in TOPIC_IDS}
    novel = {topic: {f"{topic}-novel"} for topic in TOPIC_IDS}
    rrf = {topic: [*head[topic], f"{topic}-novel"] for topic in TOPIC_IDS}
    recomputed = evaluate_arm(rrf, qrels, novel, depths=(100, 500, 1000))
    _write_json(
        prior / "metrics.json",
        {
            "schema_version": "deep-facet-candidate-evaluation-v1",
            "topic_ids": list(TOPIC_IDS),
            "novel_relevant_count": 4,
            "aggregate": {"RRF": recomputed["aggregate"]},
            "per_topic": {
                topic: {"RRF": recomputed["per_topic"][topic]}
                for topic in TOPIC_IDS
            },
            "discovery": {
                topic: {"novel_relevant_ids": [f"{topic}-novel"]}
                for topic in TOPIC_IDS
            },
        },
    )
    metrics_bytes = (prior / "metrics.json").read_bytes()
    decision_bytes = b"{}\n"
    (prior / "decision.json").write_bytes(decision_bytes)
    _write_json(
        prior / "summary.json",
        {
            "schema_version": "deep-facet-candidate-evaluation-v1",
            "status": "complete",
            "qrels_opened": True,
            "topic_ids": list(TOPIC_IDS),
            "novel_relevant_count": 4,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        },
    )
    monkeypatch.setattr(
        module,
        "HISTORICAL_PRIOR_EVALUATION_IDENTITY",
        {
            "schema_version": "deep-facet-candidate-evaluation-v1",
            "topic_ids": list(TOPIC_IDS),
            "qrels_projection_rows": 4,
            "files": {
                name: hashlib.sha256((prior / name).read_bytes()).hexdigest()
                for name in (
                    "qrels_access_receipt.json",
                    "qrels_projection.jsonl",
                    "metrics.json",
                    "decision.json",
                    "summary.json",
                )
            },
        },
        raising=False,
    )
    return freeze, prior


def test_evaluate_authenticates_projection_and_writes_create_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "NOVEL_RELEVANT_TOTAL", 4)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(
        module,
        "verify_prior_seal",
        lambda _path: {"root_sha256": "b" * 64},
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "build_representatives",
        lambda _freeze, _qrels, **_kwargs: [{"movement": "promoted"}, {"movement": "demoted"}],
    )
    monkeypatch.setattr(module, "_extended_diagnostics", lambda *_args: {})
    output = tmp_path / "evaluation"

    summary = evaluate(freeze, prior, output)

    assert summary["status"] == "complete"
    assert summary["post_qrels_diagnostic"] is True
    assert summary["production_validation"] is False
    assert {path.name for path in output.iterdir()} == {
        "metrics.json",
        "diagnostics.json",
        "decision.json",
        "summary.json",
        "input_bindings.json",
    }
    bindings = json.loads((output / "input_bindings.json").read_text())
    assert bindings["qrels_projection"]["sha256"] == hashlib.sha256(
        (prior / "qrels_projection.jsonl").read_bytes()
    ).hexdigest()
    assert bindings["prior_metrics"]["path"] == str((prior / "metrics.json").resolve())
    assert bindings["task3_root_sha256"] == "a" * 64
    assert bindings["prior_summary"]["path"] == str((prior / "summary.json").resolve())
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert set(diagnostics["per_topic_deltas"]) == set(TOPIC_IDS)
    assert "facet_yield" in diagnostics
    assert {row["movement"] for row in diagnostics["representatives"]} == {"promoted", "demoted"}

    with pytest.raises(FileExistsError, match="create-only"):
        evaluate(freeze, prior, output)


def test_evaluate_rejects_projection_hash_mismatch_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    projection = prior / "qrels_projection.jsonl"
    receipt = json.loads((prior / "qrels_access_receipt.json").read_text())
    receipt["qrels_projection_sha256"] = "0" * 64
    _write_json(prior / "qrels_access_receipt.json", receipt)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"root_sha256": "a" * 64})
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64}, raising=False)

    with pytest.raises(ValueError, match="projection SHA-256"):
        evaluate(freeze, prior, tmp_path / "out")


def test_evaluate_rejects_prior_metrics_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    summary = json.loads((prior / "summary.json").read_text())
    summary["metrics_sha256"] = "0" * 64
    _write_json(prior / "summary.json", summary)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(
        module,
        "verify_prior_seal",
        lambda _path: {"root_sha256": "b" * 64},
    )

    with pytest.raises(ValueError, match="prior metrics SHA-256"):
        evaluate(freeze, prior, tmp_path / "out")


def test_prior_seal_is_verified_before_projection_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    projection = prior / "qrels_projection.jsonl"
    real_read = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path == projection:
            raise AssertionError("projection read before prior seal verification")
        return real_read(path)

    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(
        module,
        "verify_prior_seal",
        lambda _path: (_ for _ in ()).throw(ValueError("bad prior seal")),
        raising=False,
    )
    monkeypatch.setattr(Path, "read_bytes", guarded_read)

    with pytest.raises(ValueError, match="bad prior seal"):
        evaluate(freeze, prior, tmp_path / "out")


def test_restamped_projection_receipt_cannot_replace_prior_seal_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    projection = prior / "qrels_projection.jsonl"
    changed = projection.read_bytes().replace(b'"grade":2', b'"grade":3')
    projection.write_bytes(changed)
    receipt_path = prior / "qrels_access_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["qrels_projection_sha256"] = hashlib.sha256(changed).hexdigest()
    receipt["seal_sha256"] = "0" * 64
    receipt["seal_root_sha256"] = "1" * 64
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(module, "NOVEL_RELEVANT_TOTAL", 4)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64}, raising=False)

    with pytest.raises(ValueError, match="prior seal"):
        evaluate(freeze, prior, tmp_path / "out")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "wrong-schema"),
        ("status", "complete"),
        ("qrels_opened", False),
        ("upstream_mutation_forbidden", False),
    ],
)
def test_receipt_requires_exact_evaluation_boundary_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    receipt_path = prior / "qrels_access_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64}, raising=False)

    with pytest.raises(ValueError, match="receipt contract"):
        evaluate(freeze, prior, tmp_path / "out")


def test_restamped_prior_metrics_must_equal_recomputed_verified_rrf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    metrics_path = prior / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["aggregate"]["RRF"]["recall@500"] = 0.0
    _write_json(metrics_path, metrics)
    summary_path = prior / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    _write_json(summary_path, summary)
    monkeypatch.setattr(module, "NOVEL_RELEVANT_TOTAL", 4)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64}, raising=False)

    with pytest.raises(ValueError, match="recomputed RRF"):
        evaluate(freeze, prior, tmp_path / "out")


def test_prior_evaluation_directory_is_a_closed_artifact_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    (prior / "qrels_projection-copy.jsonl").write_bytes(b"lookalike\n")
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})

    with pytest.raises(ValueError, match="missing or extra files"):
        evaluate(freeze, prior, tmp_path / "out")


def test_fully_restamped_consistent_evaluation_fails_historical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, prior = _synthetic_evaluation_inputs(tmp_path, monkeypatch)
    projection_path = prior / "qrels_projection.jsonl"
    projection = projection_path.read_bytes().replace(b'"grade":2', b'"grade":3')
    projection_path.write_bytes(projection)
    receipt_path = prior / "qrels_access_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["qrels_projection_sha256"] = hashlib.sha256(projection).hexdigest()
    _write_json(receipt_path, receipt)
    metrics_path = prior / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["restamped_but_numerically_consistent"] = True
    _write_json(metrics_path, metrics)
    decision_path = prior / "decision.json"
    _write_json(decision_path, {"restamped_but_numerically_consistent": True})
    summary_path = prior / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    summary["decision_sha256"] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    summary["restamped_but_numerically_consistent"] = True
    _write_json(summary_path, summary)
    monkeypatch.setattr(module, "NOVEL_RELEVANT_TOTAL", 4)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    monkeypatch.setattr(
        module, "verify_prior_seal", lambda _path: {"root_sha256": "b" * 64}
    )

    with pytest.raises(ValueError, match="historical identity"):
        evaluate(freeze, prior, tmp_path / "out")


def test_production_historical_identity_is_exact_and_committed() -> None:
    identity = module.HISTORICAL_PRIOR_EVALUATION_IDENTITY
    assert identity["schema_version"] == "deep-facet-candidate-evaluation-v1"
    assert identity["topic_ids"] == list(TOPIC_IDS)
    assert identity["qrels_projection_rows"] == 4633
    assert identity["files"] == {
        "qrels_access_receipt.json": "eaa4cdbe63f9a5ef8d93c8ddd16f8dc2f361867986310d8dfab78e3a005c1cca",
        "qrels_projection.jsonl": "03fc4bd18be36b7ea2d446975fec9fe17ac6698dcf068918c6bb228e9aab5e87",
        "metrics.json": "9a5ca726c9e99bab634997ccf0cf632ea096f696f17631a477c505f004a19196",
        "decision.json": "f7ebefa9bc9564e6f264f519fc46734f4583a05e7883a76c29af60dbb2f0ee3c",
        "summary.json": "4a1e0bdfc6fa0221431397bc3114ac0d0359d2c375288dd5e2df032d0c6dd61f",
    }


def test_cli_and_evaluate_api_have_no_original_qrels_path() -> None:
    assert "qrels" not in inspect.signature(evaluate).parameters
    parser = module._parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--prior-evaluation" in options
    assert "--projection" not in options
    assert "--qrels" not in options


def test_task4_builds_bounded_authenticated_promoted_and_demoted_representatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(deep, tethered)
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(
        facet_topics=facet_topics,
        tethered_topics=tethered_topics,
        input_paths=paths,
        output=freeze,
    )
    qrels = {
        topic: {f"{topic}-d{index:03d}": (2 if index % 2 else 0) for index in range(650)}
        for topic in TOPIC_IDS
    }

    rows = build_representatives(freeze, qrels, max_per_class=2)

    assert len(rows) == 4
    assert {row["movement"] for row in rows} == {"promoted", "demoted"}
    required = {
        "topic_id", "facet_id", "movement", "document_id", "narrative",
        "facet_query", "selected_passage", "facet_only_percentile",
        "tethered_percentile", "qrels_grade", "facet_only_final_rank",
        "tethered_final_rank", "prior_bm25_rank", "passage_provenance",
        "ranking_provenance",
    }
    assert all(set(row) == required for row in rows)
    tethered_candidates = {
        (row["topic_id"], row["facet_id"], row["document_id"]): row
        for row in map(json.loads, (tethered / "candidates.jsonl").read_text().splitlines())
    }
    assert all(
        tethered_candidates[(row["topic_id"], row["facet_id"], row["document_id"])]["query"]
        == row["narrative"] + "\n\nFocus: " + row["facet_query"]
        for row in rows
    )
    assert all(row["passage_provenance"]["window_sha256"] for row in rows)
    assert all(row["ranking_provenance"]["task3_rankings_sha256"] for row in rows)
    assert all(0 < row["facet_only_percentile"] <= 1 for row in rows)
    assert all(0 < row["tethered_percentile"] <= 1 for row in rows)


def test_representatives_project_authenticated_rejected_historical_facet_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    gates_path = deep / "gate_v1" / "gates.json"
    gates = json.loads(gates_path.read_text())
    gates["gates"].append(
        {
            "topic_id": "84",
            "facet_id": "84-rejected",
            "manifest_order": 24,
            "status": "rejected",
        }
    )
    gates_path.write_text(json.dumps(gates), encoding="utf-8")
    candidate_path = deep / "phase1_v1" / "candidates.jsonl"
    score_path = deep / "phase1_v1" / "scores.jsonl"
    with candidate_path.open("a", encoding="utf-8") as candidates, score_path.open(
        "a", encoding="utf-8"
    ) as scores:
        for rank in (1, 2):
            document_id = f"84-rejected-d{rank}"
            query = "rejected historical facet"
            text = f"rejected historical text {rank}"
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            candidates.write(
                json.dumps(
                    {
                        "topic_id": "84",
                        "facet_id": "84-rejected",
                        "manifest_order": 24,
                        "document_id": document_id,
                        "docid": document_id,
                        "query": query,
                        "query_sha256": query_hash,
                        "text": text,
                        "text_sha256": text_hash,
                        "rank": rank,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            scores.write(
                json.dumps(
                    {
                        "schema_version": "deep-facet-candidate-minilm-score-v1",
                        "topic_id": "84",
                        "facet_id": "84-rejected",
                        "document_id": document_id,
                        "window_id": f"rejected-{rank}",
                        "window_text": text,
                        "window_sha256": text_hash,
                        "document_sha256": text_hash,
                        "document_start_token": 0,
                        "document_end_token": 4,
                        "query_sha256": query_hash,
                        "score": float(rank),
                        "model": "synthetic/minilm",
                        "model_revision": "fixture-revision",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    monkeypatch.setattr(
        rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep)
    )
    monkeypatch.setattr(
        rank_module, "verify_scoring", lambda _path: {"status": "complete"}
    )
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(
        deep, tethered
    )
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(
        facet_topics=facet_topics,
        tethered_topics=tethered_topics,
        input_paths=paths,
        output=freeze,
    )
    qrels = {
        topic: {
            f"{topic}-d{index:03d}": (2 if index % 2 else 0)
            for index in range(650)
        }
        for topic in TOPIC_IDS
    }

    rows = build_representatives(freeze, qrels, max_per_class=1)

    assert len(rows) == 2
    assert all(row["facet_id"] != "84-rejected" for row in rows)


@pytest.mark.parametrize(
    "attack",
    ["rogue_historical_extra", "rogue_tethered_extra", "missing_tethered"],
)
def test_representative_candidate_projection_requires_frozen_semantic_coverage(
    attack: str,
) -> None:
    gates = {
        "gates": [
            {
                "topic_id": topic,
                "facet_id": f"{topic}-facet",
                "manifest_order": order,
                "status": "accepted",
            }
            for order, topic in enumerate(TOPIC_IDS)
        ]
    }
    expected = {
        (topic, f"{topic}-facet", f"{topic}-d1") for topic in TOPIC_IDS
    }
    facet_rows = [
        {"topic_id": topic, "facet_id": facet, "document_id": document}
        for topic, facet, document in sorted(expected)
    ]
    tethered_rows = list(facet_rows)
    if attack == "rogue_historical_extra":
        facet_rows.append(
            {
                "topic_id": "219",
                "facet_id": "219-facet",
                "document_id": "219-rogue",
            }
        )
    elif attack == "rogue_tethered_extra":
        tethered_rows.append(
            {
                "topic_id": "219",
                "facet_id": "219-facet",
                "document_id": "219-rogue",
            }
        )
    else:
        tethered_rows.pop()
    contents = {
        "accepted_gates": json.dumps(gates).encode(),
        "facet_candidates": b"".join(
            json.dumps(row).encode() + b"\n" for row in facet_rows
        ),
        "tethered_candidates": b"".join(
            json.dumps(row).encode() + b"\n" for row in tethered_rows
        ),
    }

    with pytest.raises(ValueError, match="candidate coverage drifted"):
        module._representative_candidate_maps(contents, expected)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("unknown_historical", "no authenticated gate"),
        ("tethered_rejected", "outside authenticated accepted"),
        ("duplicate_rejected", "duplicated"),
        ("bad_status", "identity or status drifted"),
    ],
)
def test_representative_candidate_projection_rejects_unprojectable_raw_rows(
    attack: str, message: str
) -> None:
    gates = {
        "gates": [
            {
                "topic_id": topic,
                "facet_id": f"{topic}-facet",
                "manifest_order": order,
                "status": "accepted",
            }
            for order, topic in enumerate(TOPIC_IDS)
        ]
        + [
            {
                "topic_id": "84",
                "facet_id": "84-rejected",
                "manifest_order": 24,
                "status": "rejected",
            }
        ]
    }
    expected = {
        (topic, f"{topic}-facet", f"{topic}-d1") for topic in TOPIC_IDS
    }
    facet_rows = [
        {"topic_id": topic, "facet_id": facet, "document_id": document}
        for topic, facet, document in sorted(expected)
    ]
    tethered_rows = list(facet_rows)
    rejected = {
        "topic_id": "84",
        "facet_id": "84-rejected",
        "document_id": "84-rejected-d1",
    }
    if attack == "unknown_historical":
        facet_rows.append(
            {"topic_id": "84", "facet_id": "84-unknown", "document_id": "d"}
        )
    elif attack == "tethered_rejected":
        tethered_rows.append(rejected)
    elif attack == "duplicate_rejected":
        facet_rows.extend((rejected, dict(rejected)))
    else:
        gates["gates"][-1]["status"] = "skipped"
    contents = {
        "accepted_gates": json.dumps(gates).encode(),
        "facet_candidates": b"".join(
            json.dumps(row).encode() + b"\n" for row in facet_rows
        ),
        "tethered_candidates": b"".join(
            json.dumps(row).encode() + b"\n" for row in tethered_rows
        ),
    }

    with pytest.raises(ValueError, match=message):
        module._representative_candidate_maps(contents, expected)


@pytest.mark.parametrize("bad_qrels", [
    {**{topic: {} for topic in TOPIC_IDS}, "144": {}},
    {topic: {} for topic in TOPIC_IDS[:-1]},
])
def test_representatives_reject_protected_extra_or_missing_qrels_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_qrels: dict[str, dict[str, int]]
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(deep, tethered)
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(facet_topics=facet_topics, tethered_topics=tethered_topics, input_paths=paths, output=freeze)

    with pytest.raises(ValueError, match="qrels.*exact protected pilot topics"):
        build_representatives(freeze, bad_qrels)


def test_representatives_consume_each_authenticated_source_buffer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(deep, tethered)
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(facet_topics=facet_topics, tethered_topics=tethered_topics, input_paths=paths, output=freeze)
    watched = (deep / "phase1_v1" / "candidates.jsonl").resolve()
    original_read = Path.read_bytes
    reads = 0

    def guarded_read(path: Path) -> bytes:
        nonlocal reads
        if path.resolve() == watched:
            reads += 1
            if reads > 1:
                raise AssertionError("authenticated source was reread after its hash check")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    qrels = {topic: {} for topic in TOPIC_IDS}

    rows = build_representatives(freeze, qrels, max_per_class=1)

    assert rows
    assert reads == 1


@pytest.mark.parametrize("attack", ["window_hash", "model", "span"])
def test_representatives_reject_nested_window_model_and_span_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    scores_path = tethered / "scores.jsonl"
    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    target = next(row for row in rows if row["document_id"] == "219-d500")
    if attack == "window_hash":
        target["window_sha256"] = "0" * 64
    elif attack == "model":
        target["model"] = "tampered/model"
    else:
        target["document_end_token"] = target["document_start_token"]
    scores_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(deep, tethered)
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(facet_topics=facet_topics, tethered_topics=tethered_topics, input_paths=paths, output=freeze)

    with pytest.raises(ValueError, match="window|model|span|passage"):
        build_representatives(freeze, {topic: {} for topic in TOPIC_IDS}, max_per_class=1)


def test_diagnostics_reject_scoring_pair_accounting_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "summary.json").write_text(json.dumps({
        "topic_summary": {
            topic: {
                "duplicate_skip_totals": {
                    arm: {f"{topic}-facet": 0}
                    for arm in ("FACET-2B", "TETHERED-2B")
                },
                "facet_shortage_counts": {
                    arm: {f"{topic}-facet": 0}
                    for arm in ("FACET-2B", "TETHERED-2B")
                },
            }
            for topic in TOPIC_IDS
        }
    }))
    preflight = json.dumps({
        "summary": {"query_document_pair_count": 1000, "unique_pair_count": 1000},
        "runtime_evidence": {"projected_inference_seconds": 1.0},
    }).encode()
    receipt = json.dumps({
        "cache_hit_count": 400,
        "unique_forward_pair_count": 599,
    }).encode()
    accepted_gates = json.dumps(
        {
            "gates": [
                {
                    "topic_id": topic,
                    "facet_id": f"{topic}-facet",
                    "manifest_order": order,
                    "status": "accepted",
                }
                for order, topic in enumerate(TOPIC_IDS)
            ]
        }
    ).encode()
    contents = {
        "accepted_gates": accepted_gates,
        "facet_candidates": b"",
        "facet_window_scores": b"",
        "tethered_candidates": b"",
        "tethered_window_scores": b"",
        "tethered_document_scores": b"",
        "tethered_preflight": preflight,
        "tethered_scoring_receipt": receipt,
    }
    summary = json.loads((freeze / "summary.json").read_text())
    snapshot = module.Task3Snapshot(
        freeze_dir=freeze,
        seal={},
        seal_bytes=b"",
        bindings={
            "topic_inputs": {
                arm: {topic: {"facets": []} for topic in TOPIC_IDS}
                for arm in ("FACET-2B", "TETHERED-2B")
            }
        },
        summary=summary,
        artifact_buffers={},
        producer_buffers=contents,
        producer_hashes={name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
    )
    monkeypatch.setattr(
        module, "_task3_snapshot", lambda _path: snapshot
    )
    monkeypatch.setattr(
        module,
        "_representative_sources",
        lambda *_args, **_kwargs: (contents, {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}),
    )
    arms = {
        arm: {"facet_yield": {}} for arm in ("FACET-2B", "TETHERED-2B")
    }
    ranking_rows = {
        topic: {arm: [] for arm in ("FACET-2B", "TETHERED-2B")}
        for topic in TOPIC_IDS
    }

    with pytest.raises(ValueError, match="unique scoring pair accounting"):
        module._extended_diagnostics(
            freeze, {topic: {} for topic in TOPIC_IDS}, arms, ranking_rows
        )


def test_task3_snapshot_rejects_invalid_seal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_tethered_facet_two_basket import _loader_sources, _seal_for_loader
    import trec_rag.tethered_facet_two_basket as rank_module

    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(rank_module, "verify_prior_seal", lambda _path: _seal_for_loader(deep))
    monkeypatch.setattr(rank_module, "verify_scoring", lambda _path: {"status": "complete"})
    facet_topics, tethered_topics, paths = rank_module.load_frozen_inputs(deep, tethered)
    freeze = tmp_path / "task3"
    rank_module.freeze_rankings(
        facet_topics=facet_topics, tethered_topics=tethered_topics,
        input_paths=paths, output=freeze,
    )
    seal_path = freeze / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    seal["root_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal))

    with pytest.raises(ValueError, match="snapshot seal root"):
        module._task3_snapshot(freeze)

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
)


class ExplodingRead:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("projection bytes were read before freeze verification")


def test_evaluation_accepts_projection_only_after_verified_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "verify_freeze",
        lambda _: (_ for _ in ()).throw(ValueError("bad seal")),
    )
    monkeypatch.setattr(Path, "read_bytes", ExplodingRead())

    with pytest.raises(ValueError, match="bad seal"):
        evaluate(tmp_path / "freeze", tmp_path / "projection.jsonl", tmp_path / "out")


def test_top100_identity_is_a_hard_invariant() -> None:
    with pytest.raises(ValueError, match="top 100"):
        validate_protected_head(rrf=list(range(100)), arm=list(range(99)) + [999])


def _projection_bytes(topic_ids: tuple[str, ...] = TOPIC_IDS) -> bytes:
    return b"".join(
        json.dumps(
            {"topic_id": topic, "document_id": f"{topic}-relevant", "grade": 2},
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


def _synthetic_evaluation_inputs(tmp_path: Path) -> tuple[Path, Path]:
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
    _write_json(freeze / "SEALED.json", {"root_sha256": "a" * 64})

    prior = tmp_path / "prior"
    prior.mkdir()
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
    _write_json(
        prior / "qrels_access_receipt.json",
        {
            "topic_ids": list(TOPIC_IDS),
            "qrels_projection_rows": len(TOPIC_IDS),
            "qrels_projection_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        },
    )
    _write_json(
        prior / "metrics.json",
        {
            "topic_ids": list(TOPIC_IDS),
            "discovery": {
                topic: {"novel_relevant_ids": [f"{topic}-novel"]}
                for topic in TOPIC_IDS
            },
        },
    )
    metrics_bytes = (prior / "metrics.json").read_bytes()
    _write_json(
        prior / "summary.json",
        {"metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest()},
    )
    return freeze, projection


def test_evaluate_authenticates_projection_and_writes_create_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, projection = _synthetic_evaluation_inputs(tmp_path)
    monkeypatch.setattr(module, "NOVEL_RELEVANT_TOTAL", 4)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})
    output = tmp_path / "evaluation"

    summary = evaluate(freeze, projection, output)

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
        projection.read_bytes()
    ).hexdigest()
    assert bindings["prior_metrics"]["path"] == str((projection.parent / "metrics.json").resolve())
    assert bindings["task3_root_sha256"] == "a" * 64
    assert bindings["prior_summary"]["path"] == str((projection.parent / "summary.json").resolve())
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert set(diagnostics["per_topic_deltas"]) == set(TOPIC_IDS)
    assert "facet_yield" in diagnostics

    with pytest.raises(FileExistsError, match="create-only"):
        evaluate(freeze, projection, output)


def test_evaluate_rejects_projection_hash_mismatch_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, projection = _synthetic_evaluation_inputs(tmp_path)
    receipt = json.loads((projection.parent / "qrels_access_receipt.json").read_text())
    receipt["qrels_projection_sha256"] = "0" * 64
    _write_json(projection.parent / "qrels_access_receipt.json", receipt)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"root_sha256": "a" * 64})

    with pytest.raises(ValueError, match="projection SHA-256"):
        evaluate(freeze, projection, tmp_path / "out")


def test_evaluate_rejects_prior_metrics_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, projection = _synthetic_evaluation_inputs(tmp_path)
    summary = json.loads((projection.parent / "summary.json").read_text())
    summary["metrics_sha256"] = "0" * 64
    _write_json(projection.parent / "summary.json", summary)
    monkeypatch.setattr(module, "verify_freeze", lambda _path: {"status": "verified"})

    with pytest.raises(ValueError, match="prior metrics SHA-256"):
        evaluate(freeze, projection, tmp_path / "out")


def test_cli_and_evaluate_api_have_no_original_qrels_path() -> None:
    assert "qrels" not in inspect.signature(evaluate).parameters
    parser = module._parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--projection" in options
    assert "--qrels" not in options

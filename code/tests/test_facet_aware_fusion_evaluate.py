from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.facet_aware_fusion_evaluate as module
from trec_rag.facet_aware_fusion_rank import create_ranking_freeze
from trec_rag.facet_aware_fusion_evaluate import (
    ARM_NAMES,
    TOPIC_IDS,
    build_evaluation,
    decide,
    evaluate,
    evaluate_ranking,
    load_verified_freeze,
    main,
    qrels_consumption_identity,
    qrels_consumption_registry_path,
    read_qrels,
    validate_qrels_authorization,
    validate_self_hash,
)


@pytest.fixture(autouse=True)
def _isolated_consumption_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = tmp_path / "git-common" / "trec-rag-qrels-consumptions"
    monkeypatch.setattr(
        module,
        "trusted_qrels_consumption_dir",
        lambda: root / "rag25_facet_aware_fusion_v1",
    )


def _ranking(*leading: str, prefix: str) -> list[str]:
    result = list(dict.fromkeys(leading))
    result.extend(
        f"{prefix}-filler-{index:03d}"
        for index in range(1, 101)
        if f"{prefix}-filler-{index:03d}" not in result
    )
    return result[:100]


def _rankings() -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {arm: {} for arm in ARM_NAMES}
    for topic_id in TOPIC_IDS:
        shared = f"{topic_id}-shared"
        novel = f"{topic_id}-novel"
        for arm in ARM_NAMES:
            leading = (shared, novel) if arm == "CXQ" else (shared,)
            result[arm][topic_id] = _ranking(*leading, prefix=f"{topic_id}-{arm}")
    return result


def _qrels() -> dict[str, dict[str, int]]:
    return {
        topic_id: {
            f"{topic_id}-shared": 3,
            f"{topic_id}-novel": 2,
            f"{topic_id}-nonrelevant": 0,
        }
        for topic_id in TOPIC_IDS
    }


def _prefusion_candidates() -> dict[str, list[str]]:
    return {topic_id: [f"{topic_id}-novel"] for topic_id in TOPIC_IDS}


def _prefusion_provenance_fixture():
    rows: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    pools: dict[str, str] = {}
    for topic_index, topic_id in enumerate(TOPIC_IDS):
        originals = [
            {
                "provenance_stage": "original_rank",
                "topic_id": topic_id,
                "family": "original",
                "document_id": f"{topic_id}-original-{rank:03d}",
                "rank": rank,
            }
            for rank in range(1, 101)
        ]
        facets = [
            {
                "provenance_stage": "facet_minilm_rank",
                "topic_id": topic_id,
                "family": "facet",
                "facet_id": f"{topic_id}-facet-01",
                "manifest_order": topic_index,
                "document_id": f"{topic_id}-facet-{rank:03d}",
                "rank": rank,
                "accepted": True,
            }
            for rank in range(1, 51)
        ]
        rows.extend((*originals, *facets))
        gates.append(
            {
                "topic_id": topic_id,
                "facet_id": f"{topic_id}-facet-01",
                "manifest_order": topic_index,
                "accepted": True,
            }
        )
        pool = sorted(
            {
                *(row["document_id"] for row in originals),
                *(row["document_id"] for row in facets),
            }
        )
        pools[topic_id] = hashlib.sha256(module._canonical_json_bytes(pool)).hexdigest()
    return rows, gates, pools


def _decision_fixture() -> dict[str, object]:
    per_topic = {
        topic_id: {"ndcg@10": -0.02, "novel_relevant_retained": 1}
        for topic_id in TOPIC_IDS
    }
    return {
        "systems": {
            "RRF": {"aggregate": {"graded_recall@100": 0.50, "ndcg@10": 0.70}},
            "XQ": {"aggregate": {"graded_recall@100": 0.54, "ndcg@10": 0.70}},
            "CXQ": {"aggregate": {"graded_recall@100": 0.56, "ndcg@10": 0.69}},
        },
        "comparisons": {
            "XQ": {
                "aggregate_deltas": {"graded_recall@100": 0.04, "ndcg@10": 0.0},
                "per_topic": {
                    topic_id: {"metric_deltas": {"ndcg@10": 0.0}}
                    for topic_id in TOPIC_IDS
                },
                "novel_relevant_retained": 3,
                "novel_retention_fraction": 0.75,
                "topics_with_positive_novel_retention": 3,
            },
            "CXQ": {
                "aggregate_deltas": {"graded_recall@100": 0.06, "ndcg@10": -0.01},
                "per_topic": {
                    topic_id: {
                        "metric_deltas": {"ndcg@10": row["ndcg@10"]},
                        "novel_relevant_retained": row["novel_relevant_retained"],
                    }
                    for topic_id, row in per_topic.items()
                },
                "novel_relevant_retained": 4,
                "novel_retention_fraction": 1.0,
                "topics_with_positive_novel_retention": 4,
            },
        },
    }


def _write_authorization(
    tmp_path: Path,
    *,
    ranking_freeze_sha256: str = "c" * 64,
    projection_source: bytes | None = None,
) -> tuple[Path, Path, Path]:
    projection = tmp_path / "projection.qrels"
    if projection_source is None:
        projection_source = b"".join(
            f"{topic_id} 0 {topic_id}-shared 3\n{topic_id} 0 {topic_id}-novel 2\n".encode()
            for topic_id in TOPIC_IDS
        )
    projection.write_bytes(projection_source)
    manifest = tmp_path / "qrels_manifest.json"
    manifest_payload = {
        "schema_version": "pilot-qrels-projection-v1",
        "status": "authorized_projection",
        "topic_ids": list(TOPIC_IDS),
        "projection_path": projection.name,
        "projection_sha256": hashlib.sha256(projection_source).hexdigest(),
    }
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    approval = tmp_path / "qrels_approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-access-approval-v1",
                "status": "approved",
                "topic_ids": list(TOPIC_IDS),
                "qrels_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "qrels_projection_sha256": hashlib.sha256(projection_source).hexdigest(),
                "ranking_freeze_sha256": ranking_freeze_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return projection, manifest, approval


def test_qrels_cannot_open_before_complete_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qrels_path = tmp_path / "do-not-open.qrels"
    qrels_path.write_text("233 0 forbidden 4\n", encoding="utf-8")
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path == qrels_path:
            raise AssertionError("qrels opened before freeze verification")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)

    with pytest.raises(ValueError, match="freeze"):
        evaluate(tmp_path / "partial-freeze", qrels_path, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_loader_consumes_authenticated_task3_rankings_and_top20_provenance(
    tmp_path: Path,
) -> None:
    frozen_rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for topic_index, topic_id in enumerate(TOPIC_IDS):
        originals = [
            {
                "topic_id": topic_id,
                "family": "original",
                "document_id": f"{topic_id}-original-{rank:03d}",
                "rank": rank,
            }
            for rank in range(1, 101)
        ]
        facets = [
            {
                "topic_id": topic_id,
                "family": "facet",
                "facet_id": f"{topic_id}-facet-01",
                "manifest_order": topic_index,
                "document_id": f"{topic_id}-facet-{rank:03d}",
                "rank": rank,
                "accepted": True,
            }
            for rank in range(1, 51)
        ]
        frozen_rows.extend((*originals, *facets))
        provenance.extend(
            {**row, "provenance_stage": "original_rank"} for row in originals
        )
        provenance.extend(
            {**row, "provenance_stage": "facet_minilm_rank"} for row in facets
        )
        gates.append(
            {
                "topic_id": topic_id,
                "facet_id": f"{topic_id}-facet-01",
                "manifest_order": topic_index,
                "accepted": True,
            }
        )
    freeze_dir = tmp_path / "freeze"
    create_ranking_freeze(
        freeze_dir,
        frozen_rows=frozen_rows,
        gates=gates,
        score_rows=[],
        candidate_provenance=provenance,
        input_hashes={"manifest_sha256": "0" * 64},
        topic_order=TOPIC_IDS,
    )

    freeze, rankings, prefusion, freeze_sha256 = load_verified_freeze(freeze_dir)

    assert freeze["topic_ids"] == list(TOPIC_IDS)
    assert tuple(rankings) == ARM_NAMES
    assert all(len(rankings[arm][topic_id]) == 100 for arm in ARM_NAMES for topic_id in TOPIC_IDS)
    assert prefusion["233"] == [
        f"233-facet-{rank:03d}" for rank in range(1, 21)
    ]
    assert len(freeze_sha256) == 64


def test_prefusion_provenance_requires_complete_ranked_facets_and_gate_agreement() -> None:
    rows, gates, pools = _prefusion_provenance_fixture()

    prefusion = module.validate_prefusion_provenance(rows, gates, pools)

    assert prefusion["233"] == [
        f"233-facet-{rank:03d}" for rank in range(1, 21)
    ]

    missing_rank = [
        row
        for row in rows
        if not (
            row["provenance_stage"] == "facet_minilm_rank"
            and row["topic_id"] == "233"
            and row["rank"] == 1
        )
    ]
    with pytest.raises(ValueError, match="rank|pool"):
        module.validate_prefusion_provenance(missing_rank, gates, pools)

    wrong_family = [dict(row) for row in rows]
    next(
        row
        for row in wrong_family
        if row["provenance_stage"] == "facet_minilm_rank"
    )["family"] = "original"
    with pytest.raises(ValueError, match="family"):
        module.validate_prefusion_provenance(wrong_family, gates, pools)

    disagreeing_gates = [dict(row) for row in gates]
    disagreeing_gates[0]["accepted"] = False
    with pytest.raises(ValueError, match="gate"):
        module.validate_prefusion_provenance(rows, disagreeing_gates, pools)


def test_ranking_metrics_use_graded_gain_and_fixed_judged_denominators() -> None:
    ranking = _ranking("a", "unjudged", "b", prefix="metric")
    qrels = {"a": 4, "b": 2, "not-retrieved": 1, "metric-filler-001": 0}

    metrics = evaluate_ranking(ranking, qrels)

    ideal_dcg = (2**4 - 1) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / 2
    actual_dcg = (2**4 - 1) + (2**2 - 1) / 2
    assert metrics["graded_recall@100"] == pytest.approx(6 / 7)
    assert metrics["ndcg@10"] == pytest.approx(actual_dcg / ideal_dcg)
    assert metrics["relevant@10"] == 2
    assert metrics["judged_rate@10"] == pytest.approx(3 / 10)
    assert metrics["judged_rate@100"] == pytest.approx(3 / 100)


def test_build_evaluation_reports_rrf_deltas_gains_and_novel_retention() -> None:
    result = build_evaluation(_rankings(), _qrels(), _prefusion_candidates())

    cxq = result["comparisons"]["CXQ"]
    assert cxq["aggregate_deltas"]["graded_recall@100"] == pytest.approx(0.4)
    assert cxq["novel_relevant_retained"] == 4
    assert cxq["pre_fusion_novel_relevant"] == 4
    assert cxq["novel_retention_fraction"] == 1.0
    assert cxq["topics_with_positive_novel_retention"] == 4
    assert cxq["per_topic"]["233"]["relevant_gained_vs_rrf"] == ["233-novel"]
    assert cxq["per_topic"]["233"]["relevant_lost_vs_rrf"] == []
    assert set(cxq["per_topic"]["233"]["metric_deltas"]) == {
        "graded_recall@100",
        "ndcg@10",
        "relevant@10",
        "judged_rate@10",
        "judged_rate@100",
    }


def test_build_evaluation_rejects_reordered_or_forbidden_topic_boundaries() -> None:
    rankings = _rankings()
    rankings["RRF"] = dict(reversed(list(rankings["RRF"].items())))
    with pytest.raises(ValueError, match="topic order"):
        build_evaluation(rankings, _qrels(), _prefusion_candidates())

    rankings = _rankings()
    rankings["RRF"]["200"] = rankings["RRF"].pop("233")
    with pytest.raises(ValueError, match="prior-pilot|exact topics"):
        build_evaluation(rankings, _qrels(), _prefusion_candidates())


def test_promotion_rule_is_mechanical() -> None:
    decision = decide(_decision_fixture())

    assert decision["promoted_arm"] == "CXQ"
    assert decision["checks"]["novel_retention_fraction"] is True
    assert all(decision["checks"].values())


def test_xq_is_promoted_only_when_it_is_cxqs_sole_failed_check() -> None:
    fixture = _decision_fixture()
    fixture["systems"]["XQ"]["aggregate"]["graded_recall@100"] = 0.60
    fixture["comparisons"]["XQ"]["aggregate_deltas"]["graded_recall@100"] = 0.10
    fixture["comparisons"]["XQ"]["novel_relevant_retained"] = 5

    decision = decide(fixture)

    assert decision["promoted_arm"] == "XQ"
    assert decision["failed_checks"] == ["not_worse_than_xq_on_both_primary_metrics"]


def test_cxq_must_be_noninferior_to_xq_on_each_primary_metric() -> None:
    fixture = _decision_fixture()
    fixture["systems"]["XQ"]["aggregate"]["graded_recall@100"] = 0.57
    fixture["comparisons"]["XQ"]["aggregate_deltas"]["graded_recall@100"] = 0.07

    decision = decide(fixture)

    assert decision["checks"]["not_worse_than_xq_on_both_primary_metrics"] is False
    assert decision["promoted_arm"] == "XQ"


def test_rrf_remains_when_any_other_cxq_guardrail_fails() -> None:
    fixture = _decision_fixture()
    fixture["comparisons"]["CXQ"]["per_topic"]["233"]["metric_deltas"][
        "ndcg@10"
    ] = -0.11

    decision = decide(fixture)

    assert decision["promoted_arm"] == "RRF"
    assert decision["checks"]["per_topic_ndcg_guardrail"] is False


def test_ndcg_guardrail_boundaries_tolerate_binary_float_subtraction() -> None:
    fixture = _decision_fixture()
    fixture["comparisons"]["CXQ"]["aggregate_deltas"]["ndcg@10"] = 0.69 - 0.70
    fixture["comparisons"]["CXQ"]["per_topic"]["233"]["metric_deltas"][
        "ndcg@10"
    ] = 0.60 - 0.70

    decision = decide(fixture)

    assert decision["checks"]["aggregate_ndcg_guardrail"] is True
    assert decision["checks"]["per_topic_ndcg_guardrail"] is True


@pytest.mark.parametrize(
    ("topic_order", "message"),
    [
        (("273", "233", "161", "14"), "topic order"),
        (("233", "273", "161"), "exactly"),
        (("233", "273", "161", "14", "31"), "outside"),
        (("233", "273", "161", "200"), "prior-pilot"),
        (("233", "273", "161", "144"), "protected"),
    ],
)
def test_qrels_projection_requires_exact_contiguous_topic_order(
    tmp_path: Path, topic_order: tuple[str, ...], message: str
) -> None:
    path = tmp_path / "projection.qrels"
    path.write_text(
        "".join(f"{topic_id} 0 {topic_id}-doc 2\n" for topic_id in topic_order),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        read_qrels(path)


def test_consumption_identity_is_path_independent_for_copy_and_symlink(
    tmp_path: Path,
) -> None:
    _projection, _manifest, approval = _write_authorization(tmp_path)
    copied = tmp_path / "copied-approval.json"
    copied.write_bytes(approval.read_bytes())
    linked = tmp_path / "linked-approval.json"
    linked.symlink_to(approval)

    identity = qrels_consumption_identity(approval)

    assert qrels_consumption_identity(copied) == identity
    assert qrels_consumption_identity(linked) == identity
    assert qrels_consumption_registry_path(copied) == qrels_consumption_registry_path(
        approval
    )
    assert qrels_consumption_registry_path(linked) == qrels_consumption_registry_path(
        approval
    )


def test_consumption_identity_cannot_be_bypassed_by_approval_reserialization(
    tmp_path: Path,
) -> None:
    _projection, _manifest, approval = _write_authorization(tmp_path)
    payload = json.loads(approval.read_text(encoding="utf-8"))
    payload["ignored_nonce"] = 1
    reserialized = tmp_path / "reserialized-approval.json"
    reserialized.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    original_identity = qrels_consumption_identity(approval)
    changed_identity = qrels_consumption_identity(reserialized)

    assert changed_identity["qrels_approval_sha256"] != original_identity[
        "qrels_approval_sha256"
    ]
    assert changed_identity["identity_sha256"] == original_identity["identity_sha256"]
    assert qrels_consumption_registry_path(reserialized) == qrels_consumption_registry_path(
        approval
    )


def test_qrels_manifest_cannot_authorize_projection_symlink_outside_directory(
    tmp_path: Path,
) -> None:
    authorized = tmp_path / "authorized"
    authorized.mkdir()
    outside = tmp_path / "outside.qrels"
    outside.write_bytes(
        b"".join(f"{topic_id} 0 {topic_id}-doc 2\n".encode() for topic_id in TOPIC_IDS)
    )
    projection = authorized / "projection.qrels"
    projection.symlink_to(outside)
    manifest = authorized / "qrels_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-projection-v1",
                "status": "authorized_projection",
                "topic_ids": list(TOPIC_IDS),
                "projection_path": projection.name,
                "projection_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    approval = authorized / "qrels_approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-access-approval-v1",
                "status": "approved",
                "topic_ids": list(TOPIC_IDS),
                "qrels_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "qrels_projection_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                "ranking_freeze_sha256": "c" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remain inside"):
        validate_qrels_authorization(
            projection,
            manifest,
            approval,
            ranking_freeze_sha256="c" * 64,
        )


def test_evaluate_consumes_registry_and_receipt_before_single_projection_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projection, manifest, approval = _write_authorization(tmp_path)
    output = tmp_path / "evaluation"
    rankings = _rankings()
    monkeypatch.setattr(
        module,
        "load_verified_freeze",
        lambda _path: (
            {
                "topic_ids": list(TOPIC_IDS),
                "arms": list(ARM_NAMES),
                "depth": 100,
                "complete": True,
                "qrels_opened": False,
            },
            rankings,
            _prefusion_candidates(),
            "c" * 64,
        ),
    )
    original_read = module.os.read
    projection_reads: list[Path] = []

    def guarded(descriptor: int, count: int) -> bytes:
        assert (output / "qrels_access_receipt.json").exists()
        assert qrels_consumption_registry_path(approval).exists()
        chunk = original_read(descriptor, count)
        if chunk:
            projection_reads.append(projection)
        return chunk

    monkeypatch.setattr(module.os, "read", guarded)

    result = evaluate(
        tmp_path / "freeze",
        projection,
        output,
        qrels_manifest=manifest,
        qrels_approval=approval,
    )

    assert projection_reads == [projection]
    assert set(path.name for path in output.iterdir()) == {
        "metrics.json",
        "gains_losses.json",
        "decision.json",
        "qrels_access_receipt.json",
    }
    assert result["decision"]["promoted_arm"] == "CXQ"
    for name in ("metrics.json", "gains_losses.json", "decision.json"):
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert validate_self_hash(payload) is True
        assert payload["bindings"]["ranking_freeze_sha256"] == "c" * 64

    monkeypatch.setattr(
        module,
        "read_qrels",
        lambda *_args, **_kwargs: pytest.fail("qrels projection reopened"),
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate(
            tmp_path / "freeze",
            projection,
            tmp_path / "evaluation-two",
            qrels_manifest=manifest,
            qrels_approval=approval,
        )


def test_projection_path_swap_after_authorization_cannot_read_outside_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projection, manifest, approval = _write_authorization(tmp_path)
    outside = tmp_path / "outside.qrels"
    outside.write_text("144 0 protected 4\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_verified_freeze",
        lambda _path: (
            {
                "topic_ids": list(TOPIC_IDS),
                "arms": list(ARM_NAMES),
                "depth": 100,
                "complete": True,
                "qrels_opened": False,
            },
            _rankings(),
            _prefusion_candidates(),
            "c" * 64,
        ),
    )
    real_receipt = module._create_receipt

    def swap_after_receipt(*args, **kwargs):
        receipt = real_receipt(*args, **kwargs)
        projection.unlink()
        projection.symlink_to(outside)
        return receipt

    monkeypatch.setattr(module, "_create_receipt", swap_after_receipt)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == projection and path.is_symlink():
            raise AssertionError("outside qrels bytes were read after path swap")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    with pytest.raises(ValueError, match="projection"):
        evaluate(
            tmp_path / "freeze",
            projection,
            tmp_path / "evaluation",
            qrels_manifest=manifest,
            qrels_approval=approval,
        )

    assert qrels_consumption_registry_path(approval).exists()


def test_projection_parent_swap_after_authorization_is_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authorized = tmp_path / "authorized"
    authorized.mkdir()
    projection, manifest, local_approval = _write_authorization(authorized)
    approval = tmp_path / "approval.json"
    approval.write_bytes(local_approval.read_bytes())
    local_approval.unlink()
    monkeypatch.setattr(
        module,
        "load_verified_freeze",
        lambda _path: (
            {
                "topic_ids": list(TOPIC_IDS),
                "arms": list(ARM_NAMES),
                "depth": 100,
                "complete": True,
                "qrels_opened": False,
            },
            _rankings(),
            _prefusion_candidates(),
            "c" * 64,
        ),
    )
    real_receipt = module._create_receipt

    def swap_parent_after_receipt(*args, **kwargs):
        receipt = real_receipt(*args, **kwargs)
        authorized.rename(tmp_path / "authorized-original")
        authorized.mkdir()
        projection.write_text("144 0 protected 4\n", encoding="utf-8")
        return receipt

    monkeypatch.setattr(module, "_create_receipt", swap_parent_after_receipt)
    original_read = module.os.read

    def guarded_read(descriptor: int, count: int) -> bytes:
        target = Path(f"/proc/self/fd/{descriptor}").resolve()
        if target == projection.resolve():
            raise AssertionError("replacement-directory qrels bytes were read")
        return original_read(descriptor, count)

    monkeypatch.setattr(module.os, "read", guarded_read)

    with pytest.raises(ValueError, match="directory.*changed"):
        evaluate(
            tmp_path / "freeze",
            projection,
            tmp_path / "evaluation",
            qrels_manifest=manifest,
            qrels_approval=approval,
        )


def test_evaluate_infers_established_authorized_input_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    experiment = tmp_path / "outputs/rag25_facet_aware_fusion_v1"
    projection_dir = experiment / "authorized_inputs/pilot_qrels_projection_v1"
    projection_dir.mkdir(parents=True)
    projection = projection_dir / "pilot_topics_233_273_161_14.qrels"
    projection.write_bytes(
        b"".join(
            f"{topic_id} 0 {topic_id}-shared 3\n{topic_id} 0 {topic_id}-novel 2\n".encode()
            for topic_id in TOPIC_IDS
        )
    )
    manifest = projection_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-projection-v1",
                "status": "authorized_projection",
                "topic_ids": list(TOPIC_IDS),
                "projection_path": projection.name,
                "projection_sha256": hashlib.sha256(projection.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    approval = experiment / "approvals/qrels_access_v1.json"
    approval.parent.mkdir()
    approval.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-access-approval-v1",
                "status": "approved",
                "topic_ids": list(TOPIC_IDS),
                "qrels_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "qrels_projection_sha256": hashlib.sha256(projection.read_bytes()).hexdigest(),
                "ranking_freeze_sha256": "c" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "load_verified_freeze",
        lambda _path: (
            {
                "topic_ids": list(TOPIC_IDS),
                "arms": list(ARM_NAMES),
                "depth": 100,
                "complete": True,
                "qrels_opened": False,
            },
            _rankings(),
            _prefusion_candidates(),
            "c" * 64,
        ),
    )

    result = evaluate(tmp_path / "freeze", projection, tmp_path / "evaluation")

    assert result["decision"]["promoted_arm"] == "CXQ"


def test_cli_evaluate_forwards_explicit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *args, **kwargs: observed.update({"args": args, **kwargs})
        or {"decision": {"promoted_arm": "RRF"}},
    )

    assert main(
        [
            "evaluate",
            "--freeze",
            str(tmp_path / "freeze"),
            "--qrels",
            str(tmp_path / "projection.qrels"),
            "--qrels-manifest",
            str(tmp_path / "manifest.json"),
            "--qrels-approval",
            str(tmp_path / "approval.json"),
            "--output",
            str(tmp_path / "evaluation"),
        ]
    ) == 0
    assert observed["args"] == (
        tmp_path / "freeze",
        tmp_path / "projection.qrels",
        tmp_path / "evaluation",
    )
    assert observed["qrels_manifest"] == tmp_path / "manifest.json"
    assert observed["qrels_approval"] == tmp_path / "approval.json"


def test_cli_defaults_match_established_authorized_input_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *args, **kwargs: observed.update({"args": args, **kwargs})
        or {"decision": {"promoted_arm": "RRF"}},
    )

    assert main(
        [
            "evaluate",
            "--freeze",
            str(tmp_path / "freeze"),
            "--output",
            str(tmp_path / "evaluation"),
        ]
    ) == 0
    experiment = Path("outputs/rag25_facet_aware_fusion_v1")
    projection_dir = experiment / "authorized_inputs/pilot_qrels_projection_v1"
    assert observed["args"] == (
        tmp_path / "freeze",
        projection_dir / "pilot_topics_233_273_161_14.qrels",
        tmp_path / "evaluation",
    )
    assert observed["qrels_manifest"] == projection_dir / "manifest.json"
    assert observed["qrels_approval"] == experiment / "approvals/qrels_access_v1.json"

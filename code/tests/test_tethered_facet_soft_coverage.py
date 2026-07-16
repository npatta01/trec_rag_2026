from __future__ import annotations

import json
from pathlib import Path

import pytest

import trec_rag.tethered_facet_soft_coverage as module

from trec_rag.tethered_facet_soft_coverage import (
    _orders_from_rows,
    _replace_facet_scores,
    build_soft_permutations,
    freeze_soft_rankings,
    load_authenticated_inputs,
    load_topic_rows,
    verify_soft_freeze,
)


def _topic_input(*, reversed_rows: bool = False) -> dict[str, object]:
    docids = [f"d{index:03d}" for index in range(120)]
    rows = list(reversed(docids)) if reversed_rows else docids
    return {
        "topic_id": "219",
        "docids": rows,
        "texts": {document_id: f"document {document_id}" for document_id in rows},
        "original_rank": {
            document_id: index + 1 for index, document_id in enumerate(docids)
        },
        "facets": [
            {
                "facet_id": "positive",
                "manifest_order": 0,
                "scores": {
                    document_id: float(120 - int(document_id[1:]))
                    for document_id in rows
                },
                "bm25_rank": {
                    document_id: index + 1
                    for index, document_id in enumerate(docids)
                },
            },
            {
                "facet_id": "negative",
                "manifest_order": 1,
                "scores": {
                    document_id: float(int(document_id[1:]))
                    for document_id in rows
                },
                "bm25_rank": {
                    document_id: 120 - index
                    for index, document_id in enumerate(docids)
                },
            },
        ],
        "common_scores": {
            document_id: float(120 - index)
            for index, document_id in enumerate(docids)
        },
        "narrative_scores": {
            document_id: float(index) for index, document_id in enumerate(docids)
        },
    }


def _controls() -> dict[str, list[str]]:
    docids = [f"d{index:03d}" for index in range(120)]
    return {
        "RRF": docids,
        "NARRATIVE": list(reversed(docids)),
        "FIXED-O0": docids[::2] + docids[1::2],
    }


def test_soft_rankings_are_complete_and_protected_head_is_not_a_cutoff() -> None:
    rankings, _audit = build_soft_permutations(_topic_input(), _controls())
    expected = set(_topic_input()["docids"])
    for arm in ("TETHERED-DUAL", "TETHERED-DUAL-NR", "RRF100-TETHERED-DUAL"):
        assert set(rankings[arm]) == expected
        assert len(rankings[arm]) == len(expected)
    assert rankings["RRF100-TETHERED-DUAL"][:100] == _controls()["RRF"][:100]


def test_tethered_scores_replace_only_facet_local_features() -> None:
    _rankings, audit = build_soft_permutations(_topic_input(), _controls())
    assert audit["parameters"]["dual"] == {
        "G": 0.35,
        "N": 0.15,
        "R": 0.15,
        "L": 0.25,
        "B": 0.10,
        "D": -0.15,
    }
    assert audit["parameters"]["facet_score_source"] == "narrative_tethered"


def test_protected_head_arm_emits_no_reordered_dual_coverage_attribution() -> None:
    _rankings, audit = build_soft_permutations(_topic_input(), _controls())
    assert audit["RRF100-TETHERED-DUAL"] == {}


def test_protected_topic_rejects_before_reader_runs() -> None:
    opened = False

    def reader(_path: Path) -> list[dict[str, object]]:
        nonlocal opened
        opened = True
        return []

    with pytest.raises(ValueError, match="protected topic 144"):
        load_topic_rows(["144"], reader=reader)
    assert opened is False


@pytest.mark.parametrize("topic_id", ["144", "999"])
def test_build_rejects_topic_before_feature_access(
    monkeypatch: pytest.MonkeyPatch, topic_id: str
) -> None:
    accessed = False

    def forbidden(_topic_input):
        nonlocal accessed
        accessed = True
        raise AssertionError("feature access must not run")

    monkeypatch.setattr(module, "_build_with_audit", forbidden)
    with pytest.raises(ValueError, match="protected topic|unexpected topic"):
        build_soft_permutations({**_topic_input(), "topic_id": topic_id}, _controls())
    assert accessed is False


def test_soft_output_is_deterministic_under_input_reordering() -> None:
    forward, _ = build_soft_permutations(_topic_input(), _controls())
    reverse, _ = build_soft_permutations(
        _topic_input(reversed_rows=True), _controls()
    )
    assert reverse == forward


def test_freeze_is_create_only_complete_and_sealed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"topic_id":"219"}\n', encoding="utf-8")
    topic_inputs = {
        topic_id: {**_topic_input(), "topic_id": topic_id}
        for topic_id in ("219", "72", "300", "84")
    }
    controls = {topic_id: _controls() for topic_id in topic_inputs}
    output = tmp_path / "freeze"

    summary = freeze_soft_rankings(
        topic_inputs=topic_inputs,
        controls=controls,
        input_paths={"fixture": source},
        output=output,
    )

    assert summary["ranking_row_count"] == 4 * 6 * 120
    assert summary["protected_topic_count"] == 0
    assert all(summary[counter] == 0 for counter in (
        "network_call_count",
        "retrieval_call_count",
        "model_load_count",
        "inference_count",
        "hosted_inference_call_count",
        "paid_call_count",
    ))
    assert set(path.name for path in output.iterdir()) == {
        "parameters.json",
        "input_bindings.json",
        "rankings.jsonl",
        "summary.json",
        "SEALED.json",
    }
    assert verify_soft_freeze(output)["ranking_row_count"] == 4 * 6 * 120
    binding = json.loads((output / "input_bindings.json").read_text())
    assert binding["inputs"]["fixture"]["path"] == "source.jsonl"
    assert binding["inputs"]["fixture"]["rows"] == 1
    with pytest.raises(FileExistsError, match="create-only"):
        freeze_soft_rankings(
            topic_inputs=topic_inputs,
            controls=controls,
            input_paths={"fixture": source},
            output=output,
        )


def test_verify_rejects_mutated_ranking(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    topic_inputs = {
        topic_id: {**_topic_input(), "topic_id": topic_id}
        for topic_id in ("219", "72", "300", "84")
    }
    output = tmp_path / "freeze"
    freeze_soft_rankings(
        topic_inputs=topic_inputs,
        controls={topic_id: _controls() for topic_id in topic_inputs},
        input_paths={"fixture": source},
        output=output,
    )
    with (output / "rankings.jsonl").open("ab") as sink:
        sink.write(b"{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_soft_freeze(output)


def _fixture_freeze(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    topic_inputs = {
        topic_id: {**_topic_input(), "topic_id": topic_id}
        for topic_id in ("219", "72", "300", "84")
    }
    output = tmp_path / "freeze"
    freeze_soft_rankings(
        topic_inputs=topic_inputs,
        controls={topic_id: _controls() for topic_id in topic_inputs},
        input_paths={"fixture": source},
        output=output,
    )
    return output


def _write_rows(output: Path, rows: list[dict[str, object]]) -> None:
    content = module._jsonl_bytes(rows)
    (output / "rankings.jsonl").write_bytes(content)
    summary = json.loads((output / "summary.json").read_text())
    summary["ranking_row_count"] = len(rows)
    summary["artifacts"]["rankings.jsonl"] = {
        "bytes": len(content),
        "sha256": module._sha256(content),
    }
    (output / "summary.json").write_bytes(module._pretty_bytes(summary))


def _reseal(output: Path) -> None:
    files = {
        name: (output / name).read_bytes()
        for name in ("parameters.json", "input_bindings.json", "rankings.jsonl", "summary.json")
    }
    material = {
        "schema_version": module.SEAL_SCHEMA_VERSION,
        "status": "sealed_before_evaluation",
        "qrels_opened": False,
        "files": {
            name: {"bytes": len(content), "sha256": module._sha256(content)}
            for name, content in sorted(files.items())
        },
    }
    seal = {**material, "root_sha256": module._sha256(module._canonical_bytes(material))}
    (output / "SEALED.json").write_bytes(module._pretty_bytes(seal))


def test_verify_rejects_a_sealed_empty_arm_population(tmp_path: Path) -> None:
    output = _fixture_freeze(tmp_path)
    rows = module._read_jsonl(output / "rankings.jsonl")
    rows = [
        row for row in rows
        if not (row["topic_id"] == "219" and row["arm"] == "RRF")
    ]
    _write_rows(output, rows)
    summary = json.loads((output / "summary.json").read_text())
    summary["topic_summary"]["219"]["ranking_counts"]["RRF"] = 0
    summary["topic_summary"]["219"]["complete_permutations"]["RRF"] = False
    (output / "summary.json").write_bytes(module._pretty_bytes(summary))
    _reseal(output)
    with pytest.raises(ValueError, match="nonempty complete permutation"):
        verify_soft_freeze(output)


def test_verify_reconciles_topic_summary_counts_and_flags(tmp_path: Path) -> None:
    output = _fixture_freeze(tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    summary["topic_summary"]["219"]["ranking_counts"]["RRF"] = 119
    summary["topic_summary"]["219"]["complete_permutations"]["RRF"] = False
    (output / "summary.json").write_bytes(module._pretty_bytes(summary))
    _reseal(output)
    with pytest.raises(ValueError, match="topic summary"):
        verify_soft_freeze(output)


def test_verify_enforces_exact_rrf100_protected_head(tmp_path: Path) -> None:
    output = _fixture_freeze(tmp_path)
    rows = module._read_jsonl(output / "rankings.jsonl")
    protected = [
        row for row in rows
        if row["topic_id"] == "219" and row["arm"] == "RRF100-TETHERED-DUAL"
    ]
    protected[0]["document_id"], protected[1]["document_id"] = (
        protected[1]["document_id"], protected[0]["document_id"]
    )
    _write_rows(output, rows)
    _reseal(output)
    with pytest.raises(ValueError, match="protected head differs"):
        verify_soft_freeze(output)


def test_adapter_replaces_only_facet_score_maps() -> None:
    topic = _topic_input()
    replacement_rows = [
        {
            "topic_id": "219",
            "facet_id": facet["facet_id"],
            "document_id": document_id,
            "score": -float(int(document_id[1:])),
        }
        for facet in topic["facets"]
        for document_id in facet["scores"]
    ]

    replaced = _replace_facet_scores({"219": topic}, replacement_rows)["219"]

    assert replaced["common_scores"] == topic["common_scores"]
    assert replaced["narrative_scores"] == topic["narrative_scores"]
    assert replaced["texts"] == topic["texts"]
    assert replaced["original_rank"] == topic["original_rank"]
    for before, after in zip(topic["facets"], replaced["facets"], strict=True):
        assert after["bm25_rank"] == before["bm25_rank"]
        assert after["manifest_order"] == before["manifest_order"]
        assert after["scores"] != before["scores"]


def test_single_arm_control_file_uses_authenticated_filename_identity() -> None:
    rows = [
        {"topic_id": topic_id, "topic_rank": 1, "document_id": f"{topic_id}-d"}
        for topic_id in ("219", "72", "300", "84")
    ]
    orders = _orders_from_rows(
        rows, ["NARRATIVE"], rank_field="topic_rank", source_arm="NARRATIVE"
    )
    assert orders["219"]["NARRATIVE"] == ["219-d"]


def test_real_loader_inventory_directly_binds_gate_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path / "deep"
    tethered = tmp_path / "tethered"
    baselines = tmp_path / "baselines"
    required = [
        deep / "freeze_v1" / "SEALED.json",
        deep / "freeze_v1" / "rankings.jsonl",
        deep / "gate_v1" / "u_accepted.jsonl",
        deep / "gate_v1" / "streams.jsonl",
        deep / "gate_v1" / "summary.json",
        deep / "phase2_v1" / "scores.jsonl",
        deep / "phase2_v1" / "scoring_receipt.json",
        tethered / "document_scores.jsonl",
        tethered / "scoring_receipt.json",
        baselines / "receipt.json",
        baselines / "narrative.jsonl",
        baselines / "fixed_o0.jsonl",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tethered / "preflight.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {})
    monkeypatch.setattr(module, "_verify_inputs", lambda _gate, _phase2: None)
    monkeypatch.setattr(module, "verify_scoring", lambda _path: {})
    monkeypatch.setattr(module, "verify_baseline_rankings", lambda _path: {})
    monkeypatch.setattr(
        module,
        "_load_topic_inputs",
        lambda _manifest, _gate, _phase2: {
            topic_id: {**_topic_input(), "topic_id": topic_id}
            for topic_id in ("219", "72", "300", "84")
        },
    )

    def rows(_topics, paths, **_kwargs):
        name = paths[0].name
        if name == "document_scores.jsonl":
            return [
                {
                    "topic_id": topic_id,
                    "facet_id": facet["facet_id"],
                    "document_id": document_id,
                    "score": float(index),
                }
                for topic_id in ("219", "72", "300", "84")
                for facet in _topic_input()["facets"]
                for index, document_id in enumerate(facet["scores"])
            ]
        arm = {"rankings.jsonl": "RRF", "narrative.jsonl": "NARRATIVE", "fixed_o0.jsonl": "FIXED-O0"}[name]
        rank_field = "rank" if arm == "RRF" else "topic_rank"
        return [
            {"topic_id": topic_id, "arm": arm, rank_field: rank, "document_id": document_id}
            for topic_id in ("219", "72", "300", "84")
            for rank, document_id in enumerate(_controls()[arm], start=1)
        ]

    monkeypatch.setattr(module, "load_topic_rows", rows)
    _inputs, _controls_by_topic, source_paths = load_authenticated_inputs(
        deep, tethered, baselines
    )
    assert source_paths["accepted_streams"] == deep / "gate_v1" / "streams.jsonl"

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.facet_retrieval_control_experiment import (
    build_topic_alternatives,
    index_control_streams,
)
from trec_rag.facet_retrieval_control_freeze import (
    FREEZE_SCHEMA_VERSION,
    _parser,
    create_control_freeze,
    main,
    verify_prior_freeze,
    verify_sha256,
)
from trec_rag.facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    build_control_manifest,
)
from trec_rag.pipeline_models import RetrievedCandidate


REPO_ROOT = Path(__file__).resolve().parents[2]
R1_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)


def _row(topic_id: str, variant: str, rank: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id=topic_id,
        variant_name=variant,
        retriever_name="synthetic",
        query_text=variant,
        docid=f"{topic_id}-doc-{rank:03d}",
        rank=rank,
        score=float(101 - rank),
        text=f"{topic_id} evidence {rank}",
    )


def _candidate_inputs():
    source = json.loads(R1_PATH.read_text(encoding="utf-8"))
    r1_arm = []
    for topic_id in ("200", "225", "707", "897"):
        r1_arm.extend(
            _row(topic_id, "prompt_lab_v1:original", rank)
            for rank in range(1, 101)
        )
    for stream in source["streams"]:
        variant = f"sparse_relevance_v1:R1:{stream['stream_id']}"
        r1_arm.extend(
            _row(stream["topic_id"], variant, rank) for rank in range(1, 101)
        )

    control_rows = []
    for stream in build_control_manifest().streams:
        for arm_id in ("W0", "W1", "W2"):
            variant = f"facet_control_v1:{arm_id}:{stream.stream_id}"
            control_rows.extend(
                _row(stream.topic_id, variant, rank) for rank in range(1, 101)
            )
    return r1_arm, control_rows


def _provenance_variants(rows):
    return {
        item["variant_name"]
        for row in rows
        for item in row.provenance
    }


def test_topic_alternative_matrix_is_complete():
    r1_arm, control_rows = _candidate_inputs()

    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())

    assert len(matrix) == 25
    assert sum(key.startswith("R2:200:") for key in matrix) == 4
    assert sum(key.startswith("R2:225:") for key in matrix) == 16
    assert sum(key.startswith("R2:707:") for key in matrix) == 4
    assert [key for key in matrix if key.startswith("R2:897:")] == ["R2:897:B0"]
    assert all(len(rows) == 100 for rows in matrix.values())


def test_only_registered_streams_are_replaced_and_facet_count_is_preserved():
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())

    variants = _provenance_variants(matrix["R2:225:B0-W1"])
    assert "sparse_relevance_v1:R1:f02" in variants
    assert "sparse_relevance_v1:R1:f04" not in variants
    assert "facet_control_v1:W1:f04" in variants
    assert "sparse_relevance_v1:R1:f01" in variants
    assert "sparse_relevance_v1:R1:f03" in variants
    assert "sparse_relevance_v1:R1:f05" in variants
    assert len(variants - {"prompt_lab_v1:original"}) == 5


def test_family_weights_are_half_original_and_half_across_facets():
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())

    weights = {
        item["variant_name"]: item["rrf_weight"]
        for item in matrix["R2:200:W2"][0].provenance
    }
    assert weights["prompt_lab_v1:original"] == 0.5
    assert len(weights) == 10
    assert set(weights.values()) == {0.5, 0.5 / 9}
    assert sum(weight for variant, weight in weights.items() if "original" not in variant) == pytest.approx(0.5)


def test_shuffled_inputs_produce_identical_rankings():
    r1_arm, control_rows = _candidate_inputs()
    expected = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    random.Random(17).shuffle(r1_arm)
    random.Random(23).shuffle(control_rows)

    assert build_topic_alternatives(
        r1_arm, control_rows, build_control_manifest()
    ) == expected


def test_control_index_contains_exact_four_arms_for_four_streams():
    r1_arm, control_rows = _candidate_inputs()
    indexed = index_control_streams(
        r1_arm, control_rows, build_control_manifest()
    )

    assert len(indexed) == 16
    assert set(arm for _topic, _stream, arm in indexed) == {"B0", "W0", "W1", "W2"}
    assert all(len(rows) == 100 for rows in indexed.values())


@pytest.mark.parametrize("topic_id", PROTECTED_TOPIC_IDS)
def test_protected_ids_fail_before_fusion_access(topic_id, monkeypatch):
    r1_arm, control_rows = _candidate_inputs()
    r1_arm[0] = replace(r1_arm[0], topic_id=topic_id)
    fusion_calls = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_experiment.reciprocal_rank_fusion",
        lambda *args, **kwargs: fusion_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=f"protected topic {topic_id}"):
        build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    assert fusion_calls == []


def test_freeze_is_create_only_and_binds_all_ranking_hashes(tmp_path):
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    output = tmp_path / "freeze"

    freeze = create_control_freeze(
        output,
        matrix,
        inspections={"synthetic": {"decision": "keep"}},
        bindings={
            "manifest_sha256": "a" * 64,
            "prior_freeze_sha256": "b" * 64,
            "request_sha256": {"request": "c" * 64},
            "response_sha256": {"request": "d" * 64},
            "candidate_sha256": {"request": "e" * 64},
        },
    )

    assert freeze["schema_version"] == FREEZE_SCHEMA_VERSION
    assert freeze["status"] == "frozen_before_qrels"
    assert len(freeze["rankings"]) == 25
    assert all(len(value["sha256"]) == 64 for value in freeze["rankings"].values())
    assert len(freeze["inspection_sha256"]) == 64
    assert len(freeze["fusion_sha256"]) == 64
    assert (output / "freeze.json").is_file()
    assert len(list((output / "rankings").glob("*.jsonl"))) == 25

    with pytest.raises(FileExistsError):
        create_control_freeze(
            output,
            matrix,
            inspections={},
            bindings=freeze["bindings"],
        )


def test_freezer_cli_has_no_qrels_boundary():
    parser = _parser()

    assert all("qrel" not in option for action in parser._actions for option in action.option_strings)


def test_input_hash_mismatch_fails_before_output_creation(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "freeze"

    with pytest.raises(ValueError, match="SHA-256"):
        verify_sha256(input_path, "0" * 64)
    assert not output.exists()


def test_prior_freeze_verifies_self_and_ranking_hashes(tmp_path):
    prior = tmp_path / "prior"
    rankings = prior / "rankings"
    rankings.mkdir(parents=True)
    ranking_path = rankings / "R1.jsonl"
    ranking_path.write_text('{"rank": 1}\n', encoding="utf-8")
    import hashlib

    ranking_sha = hashlib.sha256(
        (json.dumps([{"rank": 1}], separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()
    payload = {
        "schema_version": "sparse-relevance-ranking-freeze-v1",
        "status": "frozen_before_qrels",
        "rankings": {
            "R1:family_rrf": {
                "path": "rankings/R1.jsonl",
                "rows": 1,
                "sha256": ranking_sha,
            }
        },
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    payload["freeze_sha256"] = hashlib.sha256(compact.encode()).hexdigest()
    freeze_path = prior / "freeze.json"
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")

    assert verify_prior_freeze(freeze_path) == hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()

    ranking_path.write_text('{"rank": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prior ranking SHA-256"):
        verify_prior_freeze(freeze_path)


def test_cli_rejects_protected_candidate_before_any_ledger_access(tmp_path, monkeypatch):
    manifest_path = (
        REPO_ROOT
        / "reports"
        / "experiments"
        / "facet_retrieval_control_pilot_v1"
        / "manifest.json"
    )
    candidate_path = tmp_path / "r1.jsonl"
    protected = _row(PROTECTED_TOPIC_IDS[0], "prompt_lab_v1:original", 1)
    candidate_path.write_text(json.dumps(protected.__dict__) + "\n", encoding="utf-8")
    import hashlib

    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    ledger_access = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.verify_prior_freeze",
        lambda _path: "b" * 64,
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._ledger_from_existing",
        lambda *args: ledger_access.append(args),
    )

    with pytest.raises(ValueError, match=f"protected topic {PROTECTED_TOPIC_IDS[0]}"):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--r1-source-manifest",
                str(R1_PATH),
                "--prior-freeze",
                str(tmp_path / "prior.json"),
                "--prior-ledger",
                str(tmp_path / "prior-ledger"),
                "--r1-candidates",
                str(candidate_path),
                "--r1-candidates-sha256",
                candidate_sha,
                "--control-ledger",
                str(tmp_path / "control-ledger"),
                "--shared-cache",
                str(tmp_path / "cache"),
                "--endpoint",
                "https://example.test/search",
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert ledger_access == []
    assert not (tmp_path / "output").exists()

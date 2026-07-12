import json
import random
import errno
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    build_topic_alternatives,
    index_control_streams,
)
from trec_rag.facet_retrieval_control_freeze import (
    FREEZE_SCHEMA_VERSION,
    PRIOR_FREEZE_FILE_SHA256,
    _parser,
    _publish_directory_noreplace,
    build_expected_r1_requests,
    create_control_freeze,
    main,
    validate_verified_r1_arm,
    verify_prior_freeze,
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
    kept_base = {
        "225": ("f06", "f07"),
        "707": ("f01", "f03"),
        "897": ("f01",),
    }
    for topic_id, stream_ids in kept_base.items():
        for stream_id in stream_ids:
            r1_arm.extend(
                _row(topic_id, f"prompt_lab_v1:facet:{stream_id}", rank)
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

    assert tuple(matrix) == EXPECTED_ALTERNATIVE_NAMES
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
    assert "prompt_lab_v1:facet:f06" in variants
    assert "prompt_lab_v1:facet:f07" in variants
    assert len(variants - {"prompt_lab_v1:original"}) == 7


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


@pytest.mark.parametrize("mutation", ("renamed", "missing", "swapped"))
def test_freeze_requires_the_complete_exact_alternative_name_and_topic_matrix(
    tmp_path, mutation
):
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    mutated = dict(matrix)
    if mutation == "renamed":
        mutated["R2:200:NOT-AN-ARM"] = mutated.pop("R2:200:W0")
    elif mutation == "missing":
        mutated.pop("R2:225:W2-W2")
    else:
        mutated["R2:200:W0"], mutated["R2:707:W0"] = (
            mutated["R2:707:W0"],
            mutated["R2:200:W0"],
        )

    with pytest.raises(ValueError, match="exact 25|target-topic"):
        create_control_freeze(
            tmp_path / mutation,
            mutated,
            inspections={},
            bindings={
                "manifest_sha256": "a" * 64,
                "prior_freeze_sha256": "b" * 64,
                "request_sha256": {"request": "c" * 64},
                "response_sha256": {"request": "d" * 64},
                "candidate_sha256": {"request": "e" * 64},
            },
        )
    assert not (tmp_path / mutation).exists()


def test_unserializable_inspection_leaves_final_output_absent_and_retryable(tmp_path):
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    output = tmp_path / "freeze"
    bindings = {
        "manifest_sha256": "a" * 64,
        "prior_freeze_sha256": "b" * 64,
        "request_sha256": {"request": "c" * 64},
        "response_sha256": {"request": "d" * 64},
        "candidate_sha256": {"request": "e" * 64},
    }

    with pytest.raises(TypeError):
        create_control_freeze(
            output,
            matrix,
            inspections={"bad": object()},
            bindings=bindings,
        )
    assert not output.exists()

    freeze = create_control_freeze(
        output,
        matrix,
        inspections={"good": True},
        bindings=bindings,
    )
    assert freeze["status"] == "frozen_before_qrels"


def test_atomic_publish_never_replaces_concurrently_created_destination(
    tmp_path, monkeypatch
):
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    output = tmp_path / "freeze"
    bindings = {
        "manifest_sha256": "a" * 64,
        "prior_freeze_sha256": "b" * 64,
        "request_sha256": {"request": "c" * 64},
        "response_sha256": {"request": "d" * 64},
        "candidate_sha256": {"request": "e" * 64},
    }
    raced = {}

    def race_before_publish(stage, destination):
        destination.mkdir(mode=0o711)
        raced["inode"] = destination.stat().st_ino
        raced["mode"] = destination.stat().st_mode
        _publish_directory_noreplace(stage, destination)

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._publish_directory_noreplace",
        race_before_publish,
    )

    with pytest.raises(FileExistsError):
        create_control_freeze(
            output,
            matrix,
            inspections={"good": True},
            bindings=bindings,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert output.stat().st_ino == raced["inode"]
    assert output.stat().st_mode == raced["mode"]
    assert list(tmp_path.glob(".freeze.staging-*")) == []

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._publish_directory_noreplace",
        _publish_directory_noreplace,
    )
    retry = tmp_path / "retry"
    freeze = create_control_freeze(
        retry,
        matrix,
        inspections={"good": True},
        bindings=bindings,
    )
    assert freeze["status"] == "frozen_before_qrels"
    assert (retry / "freeze.json").is_file()


def test_atomic_no_replace_helper_publishes_normally(tmp_path):
    stage = tmp_path / "stage"
    output = tmp_path / "published"
    stage.mkdir()
    (stage / "marker").write_text("staged", encoding="utf-8")

    _publish_directory_noreplace(stage, output)

    assert not stage.exists()
    assert (output / "marker").read_text(encoding="utf-8") == "staged"


def test_atomic_no_replace_helper_fails_closed_when_renameat2_is_unavailable(
    tmp_path, monkeypatch
):
    stage = tmp_path / "stage"
    output = tmp_path / "published"
    stage.mkdir()
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.ctypes.CDLL",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(OSError) as error:
        _publish_directory_noreplace(stage, output)

    assert error.value.errno == errno.ENOSYS
    assert stage.is_dir()
    assert not output.exists()


def test_freezer_cli_has_no_qrels_boundary():
    parser = _parser()

    options = {
        option for action in parser._actions for option in action.option_strings
    }
    assert not {"--qrels", "--r1-candidates", "--r1-candidates-sha256"} & options
    assert {
        "--base-run",
        "--base-cache",
        "--r1-run",
        "--r1-cache",
        "--control-run",
        "--control-cache",
    } <= options


def test_cli_rejects_protected_manifest_before_freeze_ledger_cache_or_fusion(
    tmp_path, monkeypatch
):
    manifest = build_control_manifest()
    protected_manifest = replace(
        manifest,
        streams=(
            replace(manifest.streams[0], topic_id=PROTECTED_TOPIC_IDS[0]),
            *manifest.streams[1:],
        ),
    )
    accesses = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.load_control_manifest",
        lambda _path: protected_manifest,
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.verify_prior_freeze",
        lambda _path: accesses.append("freeze"),
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._ledger_from_existing",
        lambda *args: accesses.append("ledger"),
    )

    with pytest.raises(ValueError, match=f"protected topic {PROTECTED_TOPIC_IDS[0]}"):
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--prior-freeze",
                str(tmp_path / "prior"),
                "--base-run",
                str(tmp_path / "base-run"),
                "--base-cache",
                str(tmp_path / "base-cache"),
                "--r1-run",
                str(tmp_path / "r1-run"),
                "--r1-cache",
                str(tmp_path / "r1-cache"),
                "--control-run",
                str(tmp_path / "control-run"),
                "--control-cache",
                str(tmp_path / "control-cache"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert accesses == []
    assert not (tmp_path / "output").exists()


def test_cli_keeps_base_r1_and_control_runs_bound_to_their_own_caches(
    tmp_path, monkeypatch
):
    manifest_path = (
        REPO_ROOT
        / "reports"
        / "experiments"
        / "facet_retrieval_control_pilot_v1"
        / "manifest.json"
    )
    captured = {}
    empty_lineage = {
        "ledger_sha256": {"base": "1" * 64, "r1": "2" * 64},
        "r1_arm_sha256": "3" * 64,
        "request_sha256": {"base": "4" * 64},
        "response_sha256": {"base": "5" * 64},
        "candidate_sha256": {"base": "6" * 64},
    }

    def fake_r1(_manifest, **kwargs):
        captured["r1"] = kwargs
        return [], empty_lineage

    def fake_control(_manifest, run, cache):
        captured["control"] = (run, cache)
        return [], {"control": "7" * 64}, {"control": "8" * 64}, {"control": "9" * 64}

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.verify_prior_freeze",
        lambda _path: "a" * 64,
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.load_verified_r1_arm", fake_r1
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._load_control_candidates",
        fake_control,
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze._ledger_tree_sha256",
        lambda path: "b" * 64,
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.freeze_control_experiment",
        lambda **kwargs: {"status": "frozen_before_qrels"},
    )

    paths = {name: tmp_path / name for name in (
        "base-run", "base-cache", "r1-run", "r1-cache", "control-run", "control-cache"
    )}
    assert main(
        [
            "--manifest", str(manifest_path),
            "--prior-freeze", str(tmp_path / "prior"),
            "--base-run", str(paths["base-run"]),
            "--base-cache", str(paths["base-cache"]),
            "--r1-run", str(paths["r1-run"]),
            "--r1-cache", str(paths["r1-cache"]),
            "--control-run", str(paths["control-run"]),
            "--control-cache", str(paths["control-cache"]),
            "--output", str(tmp_path / "output"),
        ]
    ) == 0
    assert captured["r1"] == {
        "base_run": paths["base-run"],
        "base_cache": paths["base-cache"],
        "r1_run": paths["r1-run"],
        "r1_cache": paths["r1-cache"],
    }
    assert captured["control"] == (paths["control-run"], paths["control-cache"])


def test_minimal_self_rehashed_prior_freeze_substitution_is_rejected(
    tmp_path, monkeypatch
):
    assert PRIOR_FREEZE_FILE_SHA256 == (
        "4a78b44ede4b979a3cb3ec96348088e4e08626e2cc4c92b464c6e36097a71389"
    )
    prior = tmp_path / "prior"
    rankings = prior / "rankings"
    rankings.mkdir(parents=True)
    ranking_path = rankings / "R1.jsonl"
    ranking_path.write_text('{"rank": 1}\n', encoding="utf-8")
    import hashlib

    ranking_sha = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
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
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_freeze.PRIOR_FREEZE_FILE_SHA256",
        hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="exact immutable prior freeze contract"):
        verify_prior_freeze(freeze_path)


def test_exact_r1_arm_request_lineage_and_stream_counts():
    base_requests, r1_requests = build_expected_r1_requests(
        build_control_manifest(), R1_PATH
    )
    assert len(base_requests) == 9
    assert len(r1_requests) == 22
    assert {
        (request.identity.topic_id, request.identity.variant_name)
        for request in base_requests
    } == {
        ("200", "prompt_lab_v1:original"),
        ("225", "prompt_lab_v1:original"),
        ("225", "prompt_lab_v1:facet:f06"),
        ("225", "prompt_lab_v1:facet:f07"),
        ("707", "prompt_lab_v1:original"),
        ("707", "prompt_lab_v1:facet:f01"),
        ("707", "prompt_lab_v1:facet:f03"),
        ("897", "prompt_lab_v1:original"),
        ("897", "prompt_lab_v1:facet:f01"),
    }
    assert all(request.identity.request_key for request in (*base_requests, *r1_requests))


@pytest.mark.parametrize("mutation", ("missing", "extra", "substituted", "protected"))
def test_r1_arm_rejects_any_lineage_mutation(mutation):
    base_requests, r1_requests = build_expected_r1_requests(
        build_control_manifest(), R1_PATH
    )
    rows = [
        RetrievedCandidate(
            topic_id=request.identity.topic_id,
            variant_name=request.identity.variant_name,
            retriever_name=request.identity.retriever_version,
            query_text=request.query_text,
            docid=f"{request.identity.request_key}-{rank:03d}",
            rank=rank,
            score=float(101 - rank),
            text="evidence",
        )
        for request in (*base_requests, *r1_requests)
        for rank in range(1, 101)
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(rows[0])
    elif mutation == "substituted":
        rows[0] = replace(rows[0], query_text="substituted")
    else:
        rows[0] = replace(rows[0], topic_id=PROTECTED_TOPIC_IDS[0])

    with pytest.raises(ValueError, match="R1 arm|protected topic"):
        validate_verified_r1_arm(rows, base_requests, r1_requests)

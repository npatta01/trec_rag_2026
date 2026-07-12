import json
import random
import errno
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    StreamArmEvaluation,
    build_topic_alternatives,
    evaluate_stream_arm,
    index_control_streams,
    retrieval_repair_decision,
    select_stream_arm,
    selected_ranking_references,
)
from trec_rag.facet_retrieval_control_evaluate import (
    _parser as _evaluation_parser,
    evaluate_control_freeze,
    main as evaluation_main,
    publish_control_evaluation,
    verify_control_freeze,
)
from trec_rag.facet_retrieval_control_freeze import (
    CandidateSnapshot,
    FREEZE_SCHEMA_VERSION,
    FREEZE_SCHEMA_VERSION_V2,
    FREEZE_WRITER_SCHEMA_VERSION,
    PRIOR_FREEZE_FILE_SHA256,
    SUPPORTED_FREEZE_SCHEMA_VERSIONS,
    _parser,
    _publish_directory_noreplace,
    build_candidate_snapshot,
    build_expected_r1_requests,
    create_control_freeze as _production_create_control_freeze,
    main,
    validate_verified_r1_arm,
    validate_candidate_snapshot,
    validate_control_freeze_schema,
    verify_control_freeze_candidate_snapshot,
    verify_prior_freeze,
)
from trec_rag.facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    build_control_manifest,
)
from trec_rag.facet_retrieval_control_run import build_control_requests
from trec_rag.pipeline_models import RetrievedCandidate, jsonable


REPO_ROOT = Path(__file__).resolve().parents[2]
R1_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)


def _row(
    topic_id: str,
    variant: str,
    rank: int,
    *,
    query_text: str | None = None,
    retriever_name: str = "synthetic",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id=topic_id,
        variant_name=variant,
        retriever_name=retriever_name,
        query_text=variant if query_text is None else query_text,
        docid=f"{topic_id}-doc-{rank:03d}",
        rank=rank,
        score=float(101 - rank),
        text=f"{topic_id} evidence {rank}",
    )


def _candidate_inputs():
    manifest = build_control_manifest()
    base_requests, r1_requests = build_expected_r1_requests(manifest, R1_PATH)
    r1_arm = []
    for request in (*base_requests, *r1_requests):
        r1_arm.extend(
            _row(
                request.identity.topic_id,
                request.identity.variant_name,
                rank,
                query_text=request.query_text,
                retriever_name=request.identity.retriever_version,
            )
            for rank in range(1, 101)
        )

    control_rows = []
    for request in build_control_requests(
        manifest,
        endpoint="http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search",
    ):
        control_rows.extend(
            _row(
                request.identity.topic_id,
                request.identity.variant_name,
                rank,
                query_text=request.query_text,
                retriever_name=request.identity.retriever_version,
            )
            for rank in range(1, 101)
        )
    return r1_arm, control_rows


def _source_lineage(r1_arm, control_rows):
    import hashlib

    manifest = build_control_manifest()
    base_requests, r1_requests = build_expected_r1_requests(manifest, R1_PATH)
    control_requests = build_control_requests(
        manifest,
        endpoint="http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search",
    )
    requests = {
        (request.identity.topic_id, request.identity.variant_name): request
        for request in (*base_requests, *r1_requests, *control_requests)
    }
    result = {}
    for row in (*r1_arm, *control_rows):
        key = (row.topic_id, row.variant_name)
        if key in result:
            continue
        request = requests[key]
        assert request.query_text == row.query_text
        assert request.identity.retriever_version == row.retriever_name
        request_sha256 = request.identity.request_key
        result[key] = {
            "request_sha256": request_sha256,
            "response_sha256": hashlib.sha256(
                f"response|{request_sha256}".encode()
            ).hexdigest(),
            "candidate_sha256": hashlib.sha256(
                f"candidates|{request_sha256}".encode()
            ).hexdigest(),
        }
    return result


def _candidate_snapshot_fixture():
    r1_arm, control_rows = _candidate_inputs()
    lineage = _source_lineage(r1_arm, control_rows)
    snapshot = build_candidate_snapshot(
        r1_arm,
        control_rows,
        build_control_manifest(),
        lineage,
    )
    return r1_arm, control_rows, lineage, snapshot


def _snapshot_bindings(lineage):
    return {
        "manifest_sha256": "a" * 64,
        "prior_freeze_sha256": "b" * 64,
        "ledger_sha256": {
            "base": "1" * 64,
            "r1": "2" * 64,
            "control": "3" * 64,
        },
        "r1_arm_sha256": "4" * 64,
        "request_sha256": {
            value["request_sha256"]: value["request_sha256"]
            for value in lineage.values()
        },
        "response_sha256": {
            value["request_sha256"]: value["response_sha256"]
            for value in lineage.values()
        },
        "candidate_sha256": {
            value["request_sha256"]: value["candidate_sha256"]
            for value in lineage.values()
        },
    }


def create_control_freeze(output, rankings, *, inspections, bindings=None):
    _r1, _control, lineage, snapshot = _candidate_snapshot_fixture()
    source_bindings = _snapshot_bindings(lineage)
    if bindings is not None:
        source_bindings["manifest_sha256"] = bindings["manifest_sha256"]
        source_bindings["prior_freeze_sha256"] = bindings["prior_freeze_sha256"]
        for optional in ("ledger_sha256", "r1_arm_sha256"):
            if optional in bindings:
                source_bindings[optional] = bindings[optional]
    return _production_create_control_freeze(
        output,
        rankings,
        inspections=inspections,
        bindings=source_bindings,
        candidate_snapshot=snapshot,
    )


def _complete_inspections(r1_arm, control_rows):
    manifest = build_control_manifest()
    indexed = index_control_streams(r1_arm, control_rows, manifest)
    return {
        f"{stream.topic_id}/{stream.stream_id}/{arm_id}": {
            "topic_id": stream.topic_id,
            "stream_id": stream.stream_id,
            "inspected_top5": 5,
            "inspected_top10": 10,
            "anchor_top5_count": 5,
            "anchor_top10_count": 10,
            "anchor_intent_cohit_top5_count": 5,
            "anchor_intent_cohit_top10_count": 10,
            "domain_drift_top5_count": 0,
            "domain_drift_top10_count": 0,
            "content_quality_top5_count": 0,
            "content_quality_top10_count": 0,
            "coherence_failed": False,
            "domain_drift_warning": False,
            "content_quality_warning": False,
            "rejected": False,
            "decision": "keep",
            "top_docids": [
                row.docid
                for row in indexed[(stream.topic_id, stream.stream_id, arm_id)][:10]
            ],
            "top_snippets": ["synthetic evidence"] * 10,
        }
        for stream in manifest.streams
        for arm_id in ("B0", "W0", "W1", "W2")
    }


def _create_evaluation_freeze(tmp_path):
    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    output = tmp_path / "freeze"
    create_control_freeze(
        output,
        matrix,
        inspections=_complete_inspections(r1_arm, control_rows),
        bindings={
            "manifest_sha256": "a" * 64,
            "prior_freeze_sha256": "b" * 64,
            "request_sha256": {"unused": "c" * 64},
            "response_sha256": {"unused": "d" * 64},
            "candidate_sha256": {"unused": "e" * 64},
            "ledger_sha256": {
                "base": "1" * 64,
                "r1": "2" * 64,
                "control": "3" * 64,
            },
            "r1_arm_sha256": "4" * 64,
        },
    )
    return output, matrix


def _rewrite_control_root(freeze_dir, payload):
    import hashlib

    from trec_rag.facet_retrieval_control_freeze import _canonical_json

    without_self = dict(payload)
    without_self.pop("freeze_sha256", None)
    payload["freeze_sha256"] = hashlib.sha256(
        _canonical_json(without_self)
    ).hexdigest()
    (freeze_dir / "freeze.json").write_bytes(_canonical_json(payload))


def _rewrite_inspections(freeze_dir, inspections):
    import hashlib

    from trec_rag.facet_retrieval_control_freeze import _canonical_json

    content = _canonical_json(inspections)
    (freeze_dir / "inspection.json").write_bytes(content)
    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["inspection_sha256"] = hashlib.sha256(content).hexdigest()
    _rewrite_control_root(freeze_dir, payload)


def _convert_to_valid_v1(freeze_dir):
    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["schema_version"] = FREEZE_SCHEMA_VERSION
    for field in (
        "candidate_streams_sha256",
        "candidates_sha256",
        "candidate_stream_rows_sha256",
    ):
        payload["bindings"].pop(field)
    _rewrite_control_root(freeze_dir, payload)
    (freeze_dir / "candidate_streams.json").unlink()
    (freeze_dir / "candidates.jsonl").unlink()
    return payload


def _write_synthetic_prior_freeze(tmp_path, matrix, monkeypatch):
    import hashlib

    import trec_rag.facet_retrieval_control_freeze as freeze_module

    references = {
        "200": "R2:200:B0",
        "225": "R2:225:B0-B0",
        "707": "R2:707:B0",
        "897": "R2:897:B0",
    }
    rows = [
        row
        for topic_id in ("200", "225", "707", "897")
        for row in matrix[references[topic_id]]
    ]
    raw_rows = [jsonable(row) for row in rows]
    semantic = hashlib.sha256(
        (json.dumps(raw_rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()
    prior = tmp_path / "prior"
    rankings_dir = prior / "rankings"
    rankings_dir.mkdir(parents=True)
    ranking_records = {}
    ranking_names = (
        f"{arm}:{fusion}"
        for arm in ("ALL", "F0", "O", "R0", "R1")
        for fusion in ("family_rrf", "interleave", "uniform_rrf")
    )
    content = b"".join(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode() + b"\n"
        for raw in raw_rows
    )
    for key in ranking_names:
        relative = f"rankings/{key.replace(':', '__')}.jsonl"
        (prior / relative).write_bytes(content)
        ranking_records[key] = {"path": relative, "rows": 400, "sha256": semantic}
    fusion = {"synthetic": "family-balanced"}
    fusion_sha256 = hashlib.sha256(
        (json.dumps(fusion, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()
    manifest_sha256 = {"R0": "5" * 64, "R1": "6" * 64}
    payload = {
        "schema_version": "sparse-relevance-ranking-freeze-v1",
        "status": "frozen_before_qrels",
        "topic_ids": ["200", "225", "707", "897"],
        "manifest_sha256": manifest_sha256,
        "fusion_definitions": fusion,
        "fusion_definitions_sha256": fusion_sha256,
        "rankings": ranking_records,
    }
    compact = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    payload["freeze_sha256"] = hashlib.sha256(compact).hexdigest()
    freeze_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    (prior / "freeze.json").write_bytes(freeze_bytes)
    monkeypatch.setattr(
        freeze_module,
        "PRIOR_FREEZE_FILE_SHA256",
        hashlib.sha256(freeze_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        freeze_module, "_PRIOR_FREEZE_INTERNAL_SHA256", payload["freeze_sha256"]
    )
    monkeypatch.setattr(freeze_module, "_PRIOR_FUSION_SHA256", fusion_sha256)
    monkeypatch.setattr(freeze_module, "_PRIOR_MANIFEST_SHA256", manifest_sha256)
    return prior, references


def _provenance_variants(rows):
    return {
        item["variant_name"]
        for row in rows
        for item in row.provenance
    }


def _arm(
    arm_id,
    *,
    gain=0,
    recall=0.0,
    ndcg=0.0,
    drift=0,
    content=0,
    coherence_failed=False,
):
    return StreamArmEvaluation(
        arm_id=arm_id,
        overlap_with_original_top100=0,
        relevant_at_10=0,
        relevant_at_100=0,
        graded_recall_at_100=recall,
        ndcg_at_10=ndcg,
        unique_relevant_contribution=0,
        unique_graded_gain=gain,
        domain_drift_top10_count=drift,
        content_quality_top10_count=content,
        coherence_failed=coherence_failed,
    )


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


def test_candidate_snapshot_is_exactly_twenty_streams_and_two_thousand_rows():
    _r1, _control, _lineage, snapshot = _candidate_snapshot_fixture()

    manifest, rows = validate_candidate_snapshot(snapshot)

    assert snapshot.schema_version == "facet-control-candidate-snapshot-v1"
    assert manifest["stream_count"] == 20
    assert manifest["row_count"] == 2000
    assert len(manifest["streams"]) == 20
    assert len(rows) == 2000
    assert set(rows[0]) == {
        "schema_version",
        "topic_id",
        "stream_id",
        "arm_id",
        "query_sha256",
        "rank",
        "docid",
    }
    assert all("text" not in row and "score" not in row for row in rows)
    assert rows == sorted(
        rows,
        key=lambda row: (
            row["topic_id"],
            row["stream_id"],
            row["arm_id"],
            row["rank"],
        ),
    )
    assert {(entry["role"], entry["source_kind"]) for entry in manifest["streams"]} == {
        ("original", "prior_base"),
        ("facet", "prior_r1"),
        ("facet", "control"),
    }


def test_candidate_snapshot_is_deterministic_under_source_reordering():
    r1_arm, control_rows = _candidate_inputs()
    lineage = _source_lineage(r1_arm, control_rows)
    expected = build_candidate_snapshot(
        r1_arm, control_rows, build_control_manifest(), lineage
    )
    random.Random(31).shuffle(r1_arm)
    random.Random(37).shuffle(control_rows)

    actual = build_candidate_snapshot(
        r1_arm, control_rows, build_control_manifest(), lineage
    )

    assert actual == expected


@pytest.mark.parametrize("mutation", ("omission", "corruption", "wrong_query"))
def test_candidate_snapshot_validation_rejects_corrupt_bytes(mutation):
    _r1, _control, _lineage, snapshot = _candidate_snapshot_fixture()
    lines = snapshot.candidate_bytes.splitlines(keepends=True)
    if mutation == "omission":
        candidate_bytes = b"".join(lines[:-1])
    elif mutation == "corruption":
        candidate_bytes = snapshot.candidate_bytes + b"not-json\n"
    else:
        row = json.loads(lines[0])
        row["query_sha256"] = "0" * 64
        lines[0] = (
            json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        )
        candidate_bytes = b"".join(lines)

    with pytest.raises(ValueError, match="candidate snapshot"):
        validate_candidate_snapshot(replace(snapshot, candidate_bytes=candidate_bytes))


def test_candidate_snapshot_validation_rejects_manifest_stream_omission():
    _r1, _control, _lineage, snapshot = _candidate_snapshot_fixture()
    manifest = json.loads(snapshot.manifest_bytes)
    manifest["streams"].pop()
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()

    with pytest.raises(ValueError, match="candidate snapshot"):
        validate_candidate_snapshot(replace(snapshot, manifest_bytes=manifest_bytes))


@pytest.mark.parametrize("mutation", ("query", "role", "source_kind", "lineage"))
def test_candidate_snapshot_validation_rejects_fully_rehashed_identity_tampering(
    mutation,
):
    import hashlib

    _r1, _control, _lineage, snapshot = _candidate_snapshot_fixture()
    manifest = json.loads(snapshot.manifest_bytes)
    rows = [json.loads(line) for line in snapshot.candidate_bytes.splitlines()]
    target = manifest["streams"][0]
    target_key = (target["topic_id"], target["stream_id"], target["arm_id"])
    if mutation == "query":
        target["query_sha256"] = "0" * 64
        for row in rows:
            if (row["topic_id"], row["stream_id"], row["arm_id"]) == target_key:
                row["query_sha256"] = "0" * 64
    elif mutation == "role":
        target["role"] = "facet" if target["role"] == "original" else "original"
    elif mutation == "source_kind":
        target["source_kind"] = "control"
    else:
        target["request_sha256"] = manifest["streams"][1]["request_sha256"]

    candidate_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
    )
    stream_rows = [
        row
        for row in rows
        if (row["topic_id"], row["stream_id"], row["arm_id"]) == target_key
    ]
    target["stream_rows_sha256"] = hashlib.sha256(
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
            + b"\n"
            for row in stream_rows
        )
    ).hexdigest()
    manifest["candidate_file_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()

    with pytest.raises(ValueError, match="candidate snapshot"):
        validate_candidate_snapshot(
            replace(
                snapshot,
                manifest_bytes=manifest_bytes,
                candidate_bytes=candidate_bytes,
            )
        )


def test_control_freeze_schema_dispatch_preserves_v1_and_writes_v2():
    assert FREEZE_SCHEMA_VERSION == "facet-control-ranking-freeze-v1"
    assert FREEZE_SCHEMA_VERSION_V2 == "facet-control-ranking-freeze-v2"
    assert FREEZE_WRITER_SCHEMA_VERSION == FREEZE_SCHEMA_VERSION_V2
    assert SUPPORTED_FREEZE_SCHEMA_VERSIONS == {
        FREEZE_SCHEMA_VERSION,
        FREEZE_SCHEMA_VERSION_V2,
    }
    assert validate_control_freeze_schema(FREEZE_SCHEMA_VERSION) == FREEZE_SCHEMA_VERSION
    assert (
        validate_control_freeze_schema(FREEZE_SCHEMA_VERSION_V2)
        == FREEZE_SCHEMA_VERSION_V2
    )
    for invalid in ("facet-control-ranking-freeze-v3", None, []):
        with pytest.raises(ValueError, match="unsupported control freeze schema"):
            validate_control_freeze_schema(invalid)


def test_candidate_snapshot_reader_accepts_legacy_v1_without_snapshot(tmp_path):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    _convert_to_valid_v1(freeze_dir)

    assert verify_control_freeze_candidate_snapshot(freeze_dir) is None


def test_candidate_snapshot_reader_rejects_minimal_self_hashed_v1(tmp_path):
    freeze_dir = tmp_path / "minimal-v1"
    freeze_dir.mkdir()
    payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_qrels",
    }
    _rewrite_control_root(freeze_dir, payload)

    with pytest.raises(ValueError, match="control freeze"):
        verify_control_freeze_candidate_snapshot(freeze_dir)


@pytest.mark.parametrize(
    "mutation",
    ("unknown", "missing", "status", "binding", "ranking_index"),
)
def test_candidate_snapshot_reader_rejects_rehashed_v1_root_mutations(
    tmp_path, mutation
):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    payload = _convert_to_valid_v1(freeze_dir)
    if mutation == "unknown":
        payload["unknown"] = True
    elif mutation == "missing":
        payload.pop("fusion_sha256")
    elif mutation == "status":
        payload["status"] = "draft"
    elif mutation == "binding":
        payload["bindings"].pop("manifest_sha256")
    else:
        payload["rankings"].pop(next(iter(payload["rankings"])))
    _rewrite_control_root(freeze_dir, payload)

    with pytest.raises(ValueError, match="control freeze|bindings|rankings"):
        verify_control_freeze_candidate_snapshot(freeze_dir)


@pytest.mark.parametrize("artifact", ("inspection.json", "fusion.json", "ranking"))
def test_candidate_snapshot_reader_rejects_v1_artifact_tampering(
    tmp_path, artifact
):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    _convert_to_valid_v1(freeze_dir)
    if artifact == "ranking":
        path = next((freeze_dir / "rankings").glob("*.jsonl"))
    else:
        path = freeze_dir / artifact
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="inspection|fusion|ranking"):
        verify_control_freeze_candidate_snapshot(freeze_dir)


@pytest.mark.parametrize("depth", (5, 10))
def test_candidate_snapshot_reader_rejects_v1_cohit_above_anchor(tmp_path, depth):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    _convert_to_valid_v1(freeze_dir)
    inspections = json.loads((freeze_dir / "inspection.json").read_bytes())
    record = inspections["200/f07a/B0"]
    if depth == 5:
        record["anchor_top5_count"] = 3
        record["anchor_intent_cohit_top5_count"] = 5
    else:
        record["anchor_top5_count"] = 3
        record["anchor_intent_cohit_top5_count"] = 3
        record["anchor_top10_count"] = 3
        record["anchor_intent_cohit_top10_count"] = 5
    _rewrite_inspections(freeze_dir, inspections)

    with pytest.raises(ValueError, match="inspection"):
        verify_control_freeze_candidate_snapshot(freeze_dir)


def test_candidate_snapshot_reader_rejects_query_tamper_with_recomputed_root_hash(
    tmp_path,
):
    import hashlib

    r1_arm, control_rows = _candidate_inputs()
    matrix = build_topic_alternatives(
        r1_arm, control_rows, build_control_manifest()
    )
    freeze_dir = tmp_path / "tampered-v2"
    create_control_freeze(freeze_dir, matrix, inspections={})
    manifest = json.loads((freeze_dir / "candidate_streams.json").read_bytes())
    rows = [
        json.loads(line)
        for line in (freeze_dir / "candidates.jsonl").read_bytes().splitlines()
    ]
    target = manifest["streams"][0]
    target_key = (target["topic_id"], target["stream_id"], target["arm_id"])
    target["query_sha256"] = "0" * 64
    for row in rows:
        if (row["topic_id"], row["stream_id"], row["arm_id"]) == target_key:
            row["query_sha256"] = "0" * 64
    candidate_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
    )
    stream_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
        if (row["topic_id"], row["stream_id"], row["arm_id"]) == target_key
    )
    target["stream_rows_sha256"] = hashlib.sha256(stream_bytes).hexdigest()
    manifest["candidate_file_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (freeze_dir / "candidate_streams.json").write_bytes(manifest_bytes)
    (freeze_dir / "candidates.jsonl").write_bytes(candidate_bytes)

    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["bindings"]["candidate_streams_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    payload["bindings"]["candidates_sha256"] = hashlib.sha256(
        candidate_bytes
    ).hexdigest()
    payload["bindings"]["candidate_stream_rows_sha256"][
        "/".join(target_key)
    ] = target["stream_rows_sha256"]
    payload.pop("freeze_sha256")
    payload["freeze_sha256"] = hashlib.sha256(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    (freeze_dir / "freeze.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate snapshot"):
        verify_control_freeze_candidate_snapshot(freeze_dir)


@pytest.mark.parametrize(
    "mutation",
    (
        "protected",
        "wrong_source",
        "wrong_lineage",
        "wrong_query",
        "wrong_rank",
        "duplicate",
    ),
)
def test_candidate_snapshot_builder_rejects_invalid_in_memory_sources(mutation):
    r1_arm, control_rows = _candidate_inputs()
    lineage = _source_lineage(r1_arm, control_rows)
    if mutation == "protected":
        r1_arm[0] = replace(r1_arm[0], topic_id=PROTECTED_TOPIC_IDS[0])
    elif mutation == "wrong_source":
        lineage.pop(("200", "prompt_lab_v1:original"))
    elif mutation == "wrong_lineage":
        lineage[("200", "prompt_lab_v1:original")]["request_sha256"] = lineage[
            ("225", "prompt_lab_v1:original")
        ]["request_sha256"]
    elif mutation == "wrong_query":
        r1_arm[0] = replace(r1_arm[0], query_text="substituted")
    elif mutation == "wrong_rank":
        control_rows[0] = replace(control_rows[0], rank=101)
    else:
        control_rows[1] = replace(control_rows[1], docid=control_rows[0].docid)

    with pytest.raises(ValueError, match="candidate snapshot|protected topic|source lineage"):
        build_candidate_snapshot(
            r1_arm,
            control_rows,
            build_control_manifest(),
            lineage,
        )


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

    assert freeze["schema_version"] == FREEZE_WRITER_SCHEMA_VERSION
    assert FREEZE_WRITER_SCHEMA_VERSION == "facet-control-ranking-freeze-v2"
    assert freeze["status"] == "frozen_before_qrels"
    assert len(freeze["rankings"]) == 25
    assert all(len(value["sha256"]) == 64 for value in freeze["rankings"].values())
    assert len(freeze["inspection_sha256"]) == 64
    assert len(freeze["fusion_sha256"]) == 64
    assert (output / "freeze.json").is_file()
    assert (output / "candidate_streams.json").is_file()
    assert len((output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 2000
    assert freeze["bindings"]["candidate_streams_sha256"]
    assert freeze["bindings"]["candidates_sha256"]
    assert len(freeze["bindings"]["candidate_stream_rows_sha256"]) == 20
    assert len(list((output / "rankings").glob("*.jsonl"))) == 25
    assert isinstance(verify_control_freeze_candidate_snapshot(output), CandidateSnapshot)

    with pytest.raises(FileExistsError):
        create_control_freeze(
            output,
            matrix,
            inspections={},
            bindings=freeze["bindings"],
        )


def test_freeze_rejects_snapshot_source_hash_substitution_before_publication(tmp_path):
    r1_arm, control_rows, lineage, snapshot = _candidate_snapshot_fixture()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    bindings = _snapshot_bindings(lineage)
    first_key = next(iter(bindings["candidate_sha256"]))
    bindings["candidate_sha256"][first_key] = "0" * 64

    with pytest.raises(ValueError, match="candidate snapshot source hashes"):
        _production_create_control_freeze(
            tmp_path / "freeze",
            matrix,
            inspections={},
            bindings=bindings,
            candidate_snapshot=snapshot,
        )
    assert not (tmp_path / "freeze").exists()


@pytest.mark.parametrize(
    "missing",
    (
        "manifest_sha256",
        "prior_freeze_sha256",
        "request_sha256",
        "response_sha256",
        "candidate_sha256",
        "ledger_sha256",
        "r1_arm_sha256",
    ),
)
def test_v2_writer_requires_every_official_source_binding_before_staging(
    tmp_path, missing
):
    r1_arm, control_rows, lineage, snapshot = _candidate_snapshot_fixture()
    matrix = build_topic_alternatives(r1_arm, control_rows, build_control_manifest())
    bindings = _snapshot_bindings(lineage)
    bindings.pop(missing)
    output = tmp_path / f"missing-{missing}"

    with pytest.raises(ValueError, match="bindings"):
        _production_create_control_freeze(
            output,
            matrix,
            inspections={},
            bindings=bindings,
            candidate_snapshot=snapshot,
        )
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.staging-*")) == []


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
        "trec_rag.facet_retrieval_control_freeze._source_lineage_for_requests",
        lambda *args: {},
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


def test_selection_prioritizes_unique_graded_gain_and_tie_breaks():
    arms = {
        "B0": _arm("B0", gain=3, recall=0.10, ndcg=0.30, drift=1, content=1),
        "W0": _arm("W0", gain=5, recall=0.09, ndcg=0.28, drift=1),
        "W1": _arm("W1", gain=5, recall=0.11, ndcg=0.25, drift=1),
        "W2": _arm("W2", gain=5, recall=0.11, ndcg=0.25, drift=1),
    }

    assert select_stream_arm(arms).arm_id == "W1"


def test_selection_excludes_coherence_failure_and_joint_noise_increase_only():
    arms = {
        "B0": _arm("B0", gain=1, drift=1, content=1),
        "W0": _arm("W0", gain=9, drift=1, content=1, coherence_failed=True),
        "W1": _arm("W1", gain=8, drift=2, content=2),
        "W2": _arm("W2", gain=7, drift=2, content=1),
    }

    assert select_stream_arm(arms).arm_id == "W2"


@pytest.mark.parametrize("expected", ("B0", "W0", "W1", "W2"))
def test_exact_ties_prefer_b0_then_w0_then_w1_then_w2(expected):
    preference = ("B0", "W0", "W1", "W2")
    arms = {
        arm_id: _arm(
            arm_id,
            coherence_failed=preference.index(arm_id) < preference.index(expected),
        )
        for arm_id in preference
    }

    assert select_stream_arm(arms).arm_id == expected


def test_selection_rejects_an_incomplete_arm_set():
    with pytest.raises(ValueError, match="exactly B0, W0, W1, and W2"):
        select_stream_arm({"B0": _arm("B0"), "W0": _arm("W0")})


def test_stream_metrics_define_unique_gain_and_zero_qrels_denominators():
    original = [
        _row("200", "prompt_lab_v1:original", rank)
        for rank in range(1, 101)
    ]
    facet = [
        replace(
            _row("200", "facet_control_v1:W1:f07a", rank),
            docid=(f"unique-{rank}" if rank <= 2 else original[rank - 1].docid),
        )
        for rank in range(1, 101)
    ]
    inspection = {
        "coherence_failed": False,
        "domain_drift_top10_count": 1,
        "content_quality_top10_count": 2,
    }

    metrics = evaluate_stream_arm(
        "W1",
        facet,
        original,
        {"unique-1": 4, "unique-2": 2, original[2].docid: 3},
        inspection,
    )
    assert metrics.unique_relevant_contribution == 2
    assert metrics.unique_graded_gain == 6
    assert metrics.relevant_at_10 == 3
    assert metrics.overlap_with_original_top100 == 98

    zero = evaluate_stream_arm("W1", facet, original, {}, inspection)
    assert zero.graded_recall_at_100 == 0.0
    assert zero.ndcg_at_10 == 0.0


def test_selected_ranking_references_are_only_pre_qrels_alternatives():
    selected = {
        "200/f07a": "W1",
        "225/f02": "B0",
        "225/f04": "W2",
        "707/f02": "W0",
    }

    references = selected_ranking_references(selected)

    assert references == {
        "200": "R2:200:W1",
        "225": "R2:225:B0-W2",
        "707": "R2:707:W0",
        "897": "R2:897:B0",
    }
    assert set(references.values()) <= set(EXPECTED_ALTERNATIVE_NAMES)


@pytest.mark.parametrize(
    ("graded_recall", "ndcg", "deltas", "selected_noise", "expected"),
    (
        (0.51, 0.48, (0.0, -0.10, 0.01, 0.02), 4, "retrieval_repair_success"),
        (0.50, 0.50, (0.0, 0.0, 0.0, 0.0), 4, "retrieval_repair_failed"),
        (0.51, 0.479, (0.0, 0.0, 0.0, 0.0), 4, "retrieval_repair_failed"),
        (0.51, 0.50, (0.0, -0.101, 0.0, 0.0), 4, "retrieval_repair_failed"),
        (0.51, 0.50, (0.0, 0.0, 0.0, 0.0), 5, "retrieval_repair_failed"),
    ),
)
def test_retrieval_repair_decision_is_mechanical(
    graded_recall, ndcg, deltas, selected_noise, expected
):
    assert retrieval_repair_decision(
        r2_graded_recall=graded_recall,
        r1_graded_recall=0.50,
        r2_ndcg=ndcg,
        r1_ndcg=0.50,
        per_topic_ndcg_deltas=deltas,
        selected_noise=selected_noise,
        b0_noise=4,
    ) == expected


def test_corrupt_freeze_fails_before_qrels_open(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "freeze.json").write_text('{"status": "corrupt"}', encoding="utf-8")
    qrels = tmp_path / "qrels.txt"
    opened = []
    original_open = Path.open

    def fail_if_qrels(path, *args, **kwargs):
        if path == qrels:
            opened.append(path)
            raise AssertionError("qrels opened before freeze verification")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_if_qrels)
    with pytest.raises(ValueError, match="freeze"):
        evaluate_control_freeze(freeze, qrels)
    assert opened == []


def test_final_output_collision_fails_before_qrels_open(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    output.mkdir()
    qrels = tmp_path / "qrels.txt"
    opened = []
    original_open = Path.open

    def fail_if_qrels(path, *args, **kwargs):
        if path == qrels:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_if_qrels)
    with pytest.raises(FileExistsError, match="create-only"):
        evaluate_control_freeze(tmp_path / "missing-freeze", qrels, output_dir=output)
    assert opened == []


def test_control_freeze_verifier_rejects_tampered_ranking(tmp_path):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    ranking = next((freeze_dir / "rankings").glob("*.jsonl"))
    ranking.write_bytes(ranking.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="ranking.*SHA-256|freeze"):
        verify_control_freeze(freeze_dir)


def test_publish_evaluation_is_create_only_and_cleans_up_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    payloads = {
        "stream_evaluation.json": {"value": 1},
        "selection.json": {"value": 2},
        "evaluation.json": {"value": 3},
        "decision.json": {"value": 4},
    }
    original = Path.write_bytes
    calls = []

    def fail_second(path, content):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("synthetic write failure")
        return original(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        publish_control_evaluation(output, payloads)
    assert not output.exists()
    assert list(tmp_path.glob(".evaluation.staging-*")) == []

    monkeypatch.setattr(Path, "write_bytes", original)
    publish_control_evaluation(output, payloads)
    assert sorted(path.name for path in output.iterdir()) == sorted(payloads)
    with pytest.raises(FileExistsError):
        publish_control_evaluation(output, payloads)


def test_evaluation_cli_has_exact_task7_arguments():
    parser = _evaluation_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help" and option != "-h"
    }
    assert options == {"--freeze-dir", "--prior-freeze", "--qrels", "--output"}


def test_valid_v1_freeze_requires_candidate_snapshot(tmp_path):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["schema_version"] = FREEZE_SCHEMA_VERSION
    for field in (
        "candidate_streams_sha256",
        "candidates_sha256",
        "candidate_stream_rows_sha256",
    ):
        payload["bindings"].pop(field)
    _rewrite_control_root(freeze_dir, payload)

    with pytest.raises(ValueError, match="candidate snapshot required"):
        verify_control_freeze(freeze_dir)


@pytest.mark.parametrize("mutation", ("truncated", "inconsistent"))
def test_inspection_schema_and_derived_decision_verify_before_qrels(
    tmp_path, mutation
):
    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    inspections = json.loads((freeze_dir / "inspection.json").read_bytes())
    record = inspections["200/f07a/B0"]
    if mutation == "truncated":
        record.pop("top_snippets")
    else:
        record["coherence_failed"] = True
    _rewrite_inspections(freeze_dir, inspections)

    with pytest.raises(ValueError, match="inspection"):
        evaluate_control_freeze(
            freeze_dir,
            tmp_path / "must-not-open.qrels",
            prior_freeze_path=tmp_path / "unused-prior",
        )


def test_rehashed_candidate_snapshot_corruption_fails_before_qrels(tmp_path):
    import hashlib

    freeze_dir, _matrix = _create_evaluation_freeze(tmp_path)
    rows = [
        json.loads(line)
        for line in (freeze_dir / "candidates.jsonl").read_bytes().splitlines()
        if line
    ]
    rows[0]["docid"] = rows[1]["docid"]
    candidate_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
    )
    (freeze_dir / "candidates.jsonl").write_bytes(candidate_bytes)
    manifest = json.loads((freeze_dir / "candidate_streams.json").read_bytes())
    manifest["candidate_file_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (freeze_dir / "candidate_streams.json").write_bytes(manifest_bytes)
    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["bindings"]["candidate_streams_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    payload["bindings"]["candidates_sha256"] = hashlib.sha256(
        candidate_bytes
    ).hexdigest()
    _rewrite_control_root(freeze_dir, payload)

    with pytest.raises(ValueError, match="candidate snapshot"):
        evaluate_control_freeze(
            freeze_dir,
            tmp_path / "must-not-open.qrels",
            prior_freeze_path=tmp_path / "unused-prior",
        )


def test_broken_symlink_output_collisions_fail_before_qrels(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    qrels = tmp_path / "qrels"
    opens = []
    original_open = Path.open

    def spy(path, *args, **kwargs):
        if path == qrels:
            opens.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    with pytest.raises(FileExistsError, match="create-only"):
        evaluate_control_freeze(
            tmp_path / "missing-freeze",
            qrels,
            prior_freeze_path=tmp_path / "missing-prior",
            output_dir=output,
        )
    with pytest.raises(FileExistsError, match="create-only"):
        publish_control_evaluation(
            output,
            {
                "stream_evaluation.json": {},
                "selection.json": {},
                "evaluation.json": {},
                "decision.json": {},
            },
        )
    assert opens == []


def test_task7_cli_uses_verified_immutable_snapshots_end_to_end(tmp_path, monkeypatch):
    import hashlib

    import trec_rag.facet_retrieval_control_freeze as freeze_module
    from trec_rag.facet_retrieval_control_freeze import _canonical_json

    freeze_dir, matrix = _create_evaluation_freeze(tmp_path)
    prior, baseline_references = _write_synthetic_prior_freeze(
        tmp_path, matrix, monkeypatch
    )
    payload = json.loads((freeze_dir / "freeze.json").read_bytes())
    payload["bindings"]["prior_freeze_sha256"] = (
        freeze_module.PRIOR_FREEZE_FILE_SHA256
    )
    _rewrite_control_root(freeze_dir, payload)
    qrels = tmp_path / "synthetic.qrels"
    qrels.write_text(
        "\n".join(
            [
                *(
                    f"{topic_id} 0 {topic_id}-doc-001 4"
                    for topic_id in ("200", "225", "707", "897")
                ),
                f"{PROTECTED_TOPIC_IDS[0]} 0 protected-doc 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"
    original_open = Path.open
    mutated = []

    def mutate_after_verification(path, *args, **kwargs):
        if path == qrels and not mutated:
            mutated.append(True)
            control_ranking = freeze_dir / "rankings" / "R2__200__B0.jsonl"
            control_ranking.write_bytes(b"mutated after verification\n")
            prior_ranking = prior / "rankings" / "R1__family_rrf.jsonl"
            prior_ranking.write_bytes(b"mutated after verification\n")
            (freeze_dir / "inspection.json").write_bytes(b"{}\n")
            (freeze_dir / "candidates.jsonl").write_bytes(b"{}\n")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate_after_verification)
    assert evaluation_main(
        [
            "--freeze-dir",
            str(freeze_dir),
            "--prior-freeze",
            str(prior),
            "--qrels",
            str(qrels),
            "--output",
            str(output),
        ]
    ) == 0

    result = {
        name: json.loads((output / name).read_text(encoding="utf-8"))
        for name in (
            "stream_evaluation.json",
            "selection.json",
            "evaluation.json",
            "decision.json",
        )
    }
    assert mutated == [True]
    assert result["selection.json"]["selected_rankings"] == baseline_references
    assert result["selection.json"]["frozen_alternative_count"] == 25
    assert result["stream_evaluation.json"]["qrels_policy"][
        "protected_qrels_topics_skipped"
    ] == [PROTECTED_TOPIC_IDS[0]]
    arm = result["stream_evaluation.json"]["streams"]["200/f07a"]["arms"]["B0"]
    assert set(arm["inspection"]) == {
        "topic_id",
        "stream_id",
        "inspected_top5",
        "inspected_top10",
        "anchor_top5_count",
        "anchor_top10_count",
        "anchor_intent_cohit_top5_count",
        "anchor_intent_cohit_top10_count",
        "domain_drift_top5_count",
        "domain_drift_top10_count",
        "content_quality_top5_count",
        "content_quality_top10_count",
        "coherence_failed",
        "domain_drift_warning",
        "content_quality_warning",
        "rejected",
        "decision",
        "top_docids",
        "top_snippets",
    }
    assert arm["metrics"]["unique_graded_gain"] == 0
    for name, saved in result.items():
        expected_hash = saved.pop("artifact_sha256")
        assert hashlib.sha256(_canonical_json(saved)).hexdigest() == expected_hash

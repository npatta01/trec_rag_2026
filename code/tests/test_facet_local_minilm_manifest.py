import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import trec_rag.facet_local_minilm_manifest as module
from trec_rag.facet_local_minilm_manifest import (
    KEPT_BASE_FACETS,
    PRIOR_FREEZE_SHA256,
    R1_MANIFEST_SHA256,
    build_facet_local_manifest,
    load_facet_local_manifest,
    load_facet_local_source_snapshot,
    write_facet_local_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
R1_MANIFEST = Path(
    "reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json"
)
PRIOR_FREEZE = Path("outputs/rag25_sparse_relevance_paired_v1/freeze_v1")
BASE_RUN = Path(
    "outputs/rag25_det_sparse_prompt_lab_v1/rate_limited_continuation_v1"
)
BASE_CACHE = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/"
    "rag25_det_sparse_prompt_lab_base_http_restart_v1"
)
R1_RUN = Path("outputs/rag25_sparse_relevance_paired_v1/run_v2/R1/ledger")
R1_CACHE = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/"
    "rag25_sparse_relevance_paired_v1"
)

EXPECTED_KEPT_BASE_FACETS = {
    ("225", "prompt_lab_v1:facet:f06"),
    ("225", "prompt_lab_v1:facet:f07"),
    ("707", "prompt_lab_v1:facet:f01"),
    ("707", "prompt_lab_v1:facet:f03"),
    ("897", "prompt_lab_v1:facet:f01"),
}
EXPECTED_REPAIRED_FACETS = {
    ("200", "sparse_relevance_v1:R1:f01"),
    ("200", "sparse_relevance_v1:R1:f02"),
    ("200", "sparse_relevance_v1:R1:f03"),
    ("200", "sparse_relevance_v1:R1:f04"),
    ("200", "sparse_relevance_v1:R1:f05a"),
    ("200", "sparse_relevance_v1:R1:f05b"),
    ("200", "sparse_relevance_v1:R1:f06"),
    ("200", "sparse_relevance_v1:R1:f07a"),
    ("200", "sparse_relevance_v1:R1:f07b"),
    ("225", "sparse_relevance_v1:R1:f01"),
    ("225", "sparse_relevance_v1:R1:f02"),
    ("225", "sparse_relevance_v1:R1:f03"),
    ("225", "sparse_relevance_v1:R1:f04"),
    ("225", "sparse_relevance_v1:R1:f05"),
    ("707", "sparse_relevance_v1:R1:f02"),
    ("897", "sparse_relevance_v1:R1:f02a"),
    ("897", "sparse_relevance_v1:R1:f02b"),
    ("897", "sparse_relevance_v1:R1:f03"),
    ("897", "sparse_relevance_v1:R1:f04a"),
    ("897", "sparse_relevance_v1:R1:f04b"),
    ("897", "sparse_relevance_v1:R1:f05"),
    ("897", "sparse_relevance_v1:R1:f06"),
}


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _jsonl_bytes(rows):
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


@pytest.fixture(scope="module")
def frozen_inputs(tmp_path_factory):
    return {
        "r1_manifest_path": R1_MANIFEST,
        "prior_freeze_path": PRIOR_FREEZE,
        "base_run": BASE_RUN,
        "base_cache": BASE_CACHE,
        "r1_run": R1_RUN,
        "r1_cache": R1_CACHE,
        "source_output": tmp_path_factory.mktemp("facet-local-source") / "source_v1",
    }


@pytest.fixture(scope="module")
def frozen_source(frozen_inputs):
    manifest = build_facet_local_manifest(**frozen_inputs)
    rows, receipt = load_facet_local_source_snapshot(
        frozen_inputs["source_output"], manifest
    )
    return manifest, rows, receipt


def test_manifest_has_exact_r1_population(frozen_source):
    manifest, _rows, _receipt = frozen_source

    assert manifest.topic_ids == ("200", "225", "707", "897")
    assert len(manifest.streams) == 31
    assert sum(stream.expected_rows for stream in manifest.streams) == 3100
    assert manifest.stream_counts == {"200": 10, "225": 8, "707": 4, "897": 9}
    assert sum(stream.family == "original" for stream in manifest.streams) == 4
    assert sum(stream.family == "facet" for stream in manifest.streams) == 27


def test_manifest_has_exact_retained_and_repaired_facets(frozen_source):
    manifest, _rows, _receipt = frozen_source
    retained = {
        (stream.topic_id, stream.variant)
        for stream in manifest.streams
        if stream.source_kind == "base" and stream.family == "facet"
    }
    repaired = {
        (stream.topic_id, stream.variant)
        for stream in manifest.streams
        if stream.source_kind == "r1"
    }

    assert KEPT_BASE_FACETS == frozenset(EXPECTED_KEPT_BASE_FACETS)
    assert retained == EXPECTED_KEPT_BASE_FACETS
    assert repaired == EXPECTED_REPAIRED_FACETS


def test_protected_topic_rejected_before_source_loader(monkeypatch):
    monkeypatch.setattr(module, "PILOT_TOPIC_IDS", ("144",))

    with pytest.raises(ValueError, match="protected topic 144"):
        build_facet_local_manifest(source_loader=lambda: pytest.fail("source read"))


def test_candidate_snapshot_has_exact_rows_text_and_hashes(frozen_source):
    manifest, rows, receipt = frozen_source
    grouped = {}
    for row in rows:
        key = (row["topic_id"], row["variant"])
        grouped.setdefault(key, []).append(row)
        assert row["query_sha256"] == _sha256(row["query"].encode("utf-8"))
        assert row["text_sha256"] == _sha256(row["text"].encode("utf-8"))
        assert isinstance(row["source_score"], (int, float))
        assert not isinstance(row["source_score"], bool)
        assert math.isfinite(row["source_score"])

    assert len(rows) == 3100
    assert len(grouped) == 31
    for stream in manifest.streams:
        stream_rows = grouped[(stream.topic_id, stream.variant)]
        assert [row["rank"] for row in stream_rows] == list(range(1, 101))
        assert len({row["document_id"] for row in stream_rows}) == 100
        assert {row["query"] for row in stream_rows} == {stream.query}
        assert stream.query_sha256 == _sha256(stream.query.encode("utf-8"))
        assert stream.candidates_sha256 == _sha256(_jsonl_bytes(stream_rows))

    assert Path(receipt["candidate_file"]).name == "candidates.jsonl"


def test_candidate_file_and_receipt_are_byte_bound(frozen_inputs, frozen_source):
    manifest, rows, receipt = frozen_source
    candidate_path = frozen_inputs["source_output"] / "candidates.jsonl"
    receipt_path = frozen_inputs["source_output"] / "source_receipt.json"
    candidate_bytes = candidate_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()

    assert candidate_bytes == _jsonl_bytes(rows)
    assert receipt["candidate_schema_version"] == manifest.candidate_schema_version
    assert receipt["candidate_rows"] == manifest.candidate_rows == 3100
    assert receipt["candidate_bytes"] == manifest.candidate_bytes == len(candidate_bytes)
    assert receipt["candidates_sha256"] == manifest.candidates_sha256 == _sha256(
        candidate_bytes
    )
    assert manifest.source_receipt_sha256 == _sha256(receipt_bytes)
    assert manifest.r1_manifest_sha256 == R1_MANIFEST_SHA256
    assert manifest.prior_freeze_sha256 == PRIOR_FREEZE_SHA256
    assert R1_MANIFEST_SHA256 == _sha256((REPO_ROOT / R1_MANIFEST).read_bytes())
    assert PRIOR_FREEZE_SHA256 == _sha256(
        (REPO_ROOT / PRIOR_FREEZE / "freeze.json").read_bytes()
    )


def test_serialization_is_deterministic_and_round_trips(tmp_path, frozen_source):
    manifest, _rows, _receipt = frozen_source
    first = manifest.to_json_bytes()
    second = manifest.to_json_bytes()
    path = tmp_path / "manifest.json"
    path.write_bytes(first)

    assert first == second
    assert first == (
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert load_facet_local_manifest(path) == manifest

    with pytest.raises(FrozenInstanceError):
        manifest.streams[0].query = "changed"


def test_source_and_manifest_outputs_are_create_only(
    tmp_path, frozen_inputs, frozen_source
):
    manifest, _rows, _receipt = frozen_source

    with pytest.raises(FileExistsError, match="create-only"):
        build_facet_local_manifest(
            **frozen_inputs,
            source_loader=lambda: pytest.fail("existing output reopened source"),
        )

    manifest_path = tmp_path / "manifest.json"
    write_facet_local_manifest(manifest_path, manifest)
    with pytest.raises(FileExistsError):
        write_facet_local_manifest(manifest_path, manifest)


def test_downstream_loader_accepts_snapshot_and_receipt_only(
    monkeypatch, frozen_inputs, frozen_source
):
    manifest, rows, receipt = frozen_source
    monkeypatch.setattr(
        module,
        "RetrievalLedger",
        lambda *args, **kwargs: pytest.fail("downstream reopened a source ledger"),
    )

    assert load_facet_local_source_snapshot(
        frozen_inputs["source_output"], manifest
    ) == (rows, receipt)
    with pytest.raises(ValueError, match="snapshot and receipt only"):
        load_facet_local_source_snapshot(BASE_RUN, manifest)

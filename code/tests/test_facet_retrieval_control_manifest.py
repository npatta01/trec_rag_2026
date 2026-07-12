import hashlib
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trec_rag.facet_retrieval_control_manifest import (
    R1_MANIFEST_SHA256,
    ControlManifestError,
    build_control_manifest,
    load_control_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
R1_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "facet_retrieval_control_pilot_v1"
    / "manifest.json"
)

EXPECTED = {
    ("200", "f07a"): (
        "Holocaust enduring effects European Jews",
        "Holocaust Holocaust enduring effects European European Jews Jews",
    ),
    ("225", "f02"): (
        "violent video games exposure desensitization violence research",
        "violent video games violent video games exposure desensitization violence research",
    ),
    ("225", "f04"): (
        "aggressive behavior children risk factors psychology",
        "aggressive aggressive behavior children children risk factors psychology",
    ),
    ("707", "f02"): (
        "sorbitol human health adverse effects safety",
        "sorbitol sorbitol human human health adverse effects safety",
    ),
}

EXPECTED_ARMS = [
    ("B0", 0.9, 0.4, False),
    ("W0", 0.9, 0.4, True),
    ("W1", 0.4, 0.4, True),
    ("W2", 0.4, 0.0, True),
]


def _analyze(text):
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _write_json(tmp_path, value, filename="manifest.json"):
    path = tmp_path / filename
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manifest_is_exact_and_bounded():
    manifest = build_control_manifest(R1_PATH)

    assert {
        (stream.topic_id, stream.stream_id): (
            stream.baseline_query,
            stream.reweighted_query,
        )
        for stream in manifest.streams
    } == EXPECTED
    assert len(manifest.streams) == 4
    assert sum(arm.external for stream in manifest.streams for arm in stream.arms) == 12
    assert manifest.max_external_requests == 12
    assert manifest.hits == 100
    assert manifest.min_interval_seconds == 10.0
    assert all(
        set(_analyze(stream.reweighted_query))
        <= set(_analyze(stream.baseline_query))
        for stream in manifest.streams
    )


def test_every_stream_has_the_exact_ordered_arms():
    manifest = build_control_manifest(R1_PATH)

    assert all(
        [
            (arm.arm_id, arm.k1, arm.b, arm.external)
            for arm in stream.arms
        ]
        == EXPECTED_ARMS
        for stream in manifest.streams
    )


def test_manifest_dataclasses_are_immutable():
    manifest = build_control_manifest(R1_PATH)

    with pytest.raises(FrozenInstanceError):
        manifest.hits = 10
    with pytest.raises(FrozenInstanceError):
        manifest.streams[0].topic_id = "144"
    with pytest.raises(FrozenInstanceError):
        manifest.streams[0].arms[0].k1 = 2.0


def test_generated_manifest_round_trips_exactly():
    loaded = load_control_manifest(MANIFEST_PATH)

    assert loaded == build_control_manifest(R1_PATH)
    assert MANIFEST_PATH.read_text(encoding="utf-8") == (
        json.dumps(loaded.to_dict(), ensure_ascii=False, indent=2) + "\n"
    )
    assert hashlib.sha256(R1_PATH.read_bytes()).hexdigest() == R1_MANIFEST_SHA256


def test_duplicate_stream_is_rejected(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["streams"].append(payload["streams"][0])

    with pytest.raises(ControlManifestError, match="duplicate.*stream"):
        load_control_manifest(_write_json(tmp_path, payload))


def test_protected_topic_is_rejected(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["streams"][0]["topic_id"] = "144"

    with pytest.raises(ControlManifestError, match="protected topic 144"):
        load_control_manifest(_write_json(tmp_path, payload))


def test_build_rejects_tampered_r1_baseline_namespace(tmp_path):
    payload = json.loads(R1_PATH.read_text(encoding="utf-8"))
    selected = next(
        stream
        for stream in payload["streams"]
        if (stream["topic_id"], stream["stream_id"]) == ("225", "f02")
    )
    selected["query"] = "violent video games changed baseline"

    with pytest.raises(ControlManifestError, match="exact R1 baseline namespace"):
        build_control_manifest(_write_json(tmp_path, payload, "r1_manifest.json"))


def test_unknown_root_key_is_rejected(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = "root"

    with pytest.raises(ControlManifestError, match="unknown manifest key.*unexpected"):
        load_control_manifest(_write_json(tmp_path, payload))


def test_unknown_stream_key_is_rejected(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["streams"][0]["unexpected"] = "stream"

    with pytest.raises(ControlManifestError, match="unknown stream key.*unexpected"):
        load_control_manifest(_write_json(tmp_path, payload))


def test_unknown_arm_key_is_rejected(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["streams"][0]["arms"][0]["unexpected"] = "arm"

    with pytest.raises(ControlManifestError, match="unknown arm key.*unexpected"):
        load_control_manifest(_write_json(tmp_path, payload))

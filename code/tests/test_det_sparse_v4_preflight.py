from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as contract
from trec_rag import det_sparse_v4_preflight as preflight


def test_offline_preflight_report_is_non_inference_and_binds_artifacts():
    report = preflight.build_offline_preflight_report()

    assert report["schema_version"] == "semantic_anchor_offline_preflight_report_v1"
    assert report["status"] == "offline_preflight_pass"
    assert report["inference_authorized"] is False
    assert report["external_cost_authorized"] is False
    assert report["cost_counters"] == preflight.ZERO_COST_COUNTERS
    assert report["denied_topic_ids"] == list(contract.DENIED_TOPIC_IDS)
    assert report["runner_visible_artifacts"] == list(contract.runner_visible_artifacts())
    assert report["scorer_only_artifacts"] == list(contract.scorer_only_artifacts())
    assert report["artifact_count"] == len(report["artifact_sha256"])
    assert report["artifact_sha256"] == contract.validate_artifact_bundle()

    preflight.validate_offline_preflight_report(report)


def test_offline_preflight_report_rejects_inference_or_cost_authorization():
    report = preflight.build_offline_preflight_report()

    report["inference_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize inference"):
        preflight.validate_offline_preflight_report(report)

    report = preflight.build_offline_preflight_report()
    report["cost_counters"] = dict(preflight.ZERO_COST_COUNTERS, model_calls=1)
    with pytest.raises(ValueError, match="cost counters"):
        preflight.validate_offline_preflight_report(report)


def test_offline_preflight_fails_on_denied_imports(tmp_path: Path):
    bad_source = tmp_path / "bad_v4_runner.py"
    bad_source.write_text(
        "from trec_rag.topics import load_topics\n"
        "from trec_rag.remote_pyserini import SearchClient\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="denied import audit failed"):
        preflight.build_offline_preflight_report(source_paths=[bad_source])


def test_offline_preflight_fails_on_denied_path_fragments(tmp_path: Path):
    bad_source = tmp_path / "bad_paths.py"
    bad_source.write_text(
        "ARTIFACT = 'cache/retrieval/topic-run.jsonl'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="denied path-fragment audit failed"):
        preflight.build_offline_preflight_report(source_paths=[bad_source])


def test_offline_preflight_fails_on_missing_required_artifact(tmp_path: Path):
    copied = tmp_path / "artifacts"
    shutil.copytree(contract.ARTIFACT_DIR, copied)
    (copied / "semantic_anchor_case_registry_v1.json").unlink()

    with pytest.raises(ValueError, match="artifact missing"):
        preflight.build_offline_preflight_report(copied)


def test_offline_preflight_cli_writes_create_only_json(tmp_path: Path):
    output = tmp_path / "preflight.json"

    assert preflight.main(["--output", output.as_posix(), "--pretty"]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    preflight.validate_offline_preflight_report(report)

    with pytest.raises(FileExistsError):
        preflight.main(["--output", output.as_posix()])

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

import trec_rag.det_sparse_v2_provenance as provenance_module
from trec_rag.det_sparse_v2_provenance import (
    MODULE_BINDINGS,
    SOURCE_FILES,
    current_runtime_provenance,
    current_source_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_FILES = {
    "code/trec_rag/deterministic_sparse_v2.py",
    "code/trec_rag/det_sparse_v2_config.py",
    "code/trec_rag/det_sparse_v2_selection.py",
    "code/trec_rag/det_sparse_v2_preflight.py",
    "code/trec_rag/det_sparse_v2_provenance.py",
    "configs/det_sparse_v2.yaml",
}
SHARED_FILES = {
    "code/trec_rag/deterministic_sparse.py",
    "code/trec_rag/det_sparse_config.py",
    "code/trec_rag/det_sparse_freeze.py",
    "code/trec_rag/evaluation.py",
    "code/trec_rag/query_analyzer.py",
    "code/trec_rag/pipeline_models.py",
    "code/trec_rag/query_planner.py",
    "code/trec_rag/query_schema_compat.py",
    "code/trec_rag/topics.py",
    "code/trec_rag/repo_env.py",
    "code/tools/lucene_analyzer/AnalyzerServer.java",
    "code/tools/run_lucene_analyzer.sh",
}


def _fake_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout

    def run(arguments, **_kwargs):
        suffix = tuple(arguments[1:])
        if suffix == ("status", "--porcelain"):
            return Completed("")
        if suffix == ("rev-parse", "HEAD"):
            return Completed("a" * 40 + "\n")
        if suffix == ("rev-parse", "HEAD^{tree}"):
            return Completed("b" * 40 + "\n")
        raise AssertionError(f"unexpected git command: {arguments!r}")

    monkeypatch.setattr(provenance_module.subprocess, "run", run)


def _fake_lucene_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_dir = tmp_path / "cache" / "lucene-analyzer" / "10.4.0"
    payloads = {
        "lib/lucene-analysis-common-10.4.0.jar": b"analysis-common",
        "lib/lucene-core-10.4.0.jar": b"lucene-core",
        "classes/AnalyzerServer$1.class": b"inner-class",
        "classes/AnalyzerServer.class": b"main-class",
    }
    bindings = {}
    for relative, payload in payloads.items():
        path = cache_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        bindings[relative] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(provenance_module, "LUCENE_RUNTIME_ARTIFACT_SHA256", bindings)
    monkeypatch.setattr(provenance_module, "_lucene_cache_dir", lambda _root: cache_dir)
    container = {
        "Name": provenance_module.LUCENE_CONTAINER_NAME,
        "Image": provenance_module.LUCENE_IMAGE_ID,
        "ImageDigest": provenance_module.LUCENE_IMAGE_DIGEST,
        "Config": {
            "Image": provenance_module.LUCENE_IMAGE_REFERENCE,
            "Cmd": ["java", "-cp", "/build/classes:/build/lib/*", "AnalyzerServer"],
            "Env": [
                "ANALYZER_PORT=18081",
                "ANALYZER_INDEX_ID=hosted_climbmix_unknown_revision",
            ],
        },
        "State": {"Running": True},
        "Mounts": [
            {
                "Source": str(cache_dir),
                "Destination": "/build",
                "RW": False,
            }
        ],
        "NetworkSettings": {
            "Ports": {"18081/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18081"}]}
        },
    }
    monkeypatch.setattr(provenance_module, "_inspect_lucene_container", lambda: container)
    return cache_dir, container


def test_v2_boundary_is_fully_hashed_and_import_bound(monkeypatch):
    _fake_clean_git(monkeypatch)

    provenance = current_source_provenance(REPO_ROOT)

    assert V2_FILES | SHARED_FILES <= set(SOURCE_FILES)
    assert set(SOURCE_FILES) == set(provenance["source_files_sha256"])
    assert provenance["required_module_bindings"] == dict(MODULE_BINDINGS)
    for relative in (V2_FILES | SHARED_FILES) - {
        "configs/det_sparse_v2.yaml",
        "code/tools/lucene_analyzer/AnalyzerServer.java",
        "code/tools/run_lucene_analyzer.sh",
    }:
        module_matches = [
            module_name
            for module_name, bound_path in MODULE_BINDINGS.items()
            if bound_path == relative
        ]
        assert module_matches, f"{relative} is not import-location bound"


def test_required_bindings_do_not_depend_on_import_order(monkeypatch):
    _fake_clean_git(monkeypatch)
    module_name = "trec_rag.det_sparse_v2_selection"

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    before_import = current_source_provenance(REPO_ROOT)

    loaded_later = ModuleType(module_name)
    loaded_later.__file__ = str(REPO_ROOT / MODULE_BINDINGS[module_name])
    monkeypatch.setitem(sys.modules, module_name, loaded_later)
    after_import = current_source_provenance(REPO_ROOT)

    assert before_import == after_import
    assert module_name in before_import["required_module_bindings"]


def test_relocated_loaded_v2_module_is_rejected(monkeypatch):
    _fake_clean_git(monkeypatch)
    monkeypatch.setattr(
        provenance_module,
        "__file__",
        "/tmp/substituted_det_sparse_v2_provenance.py",
    )

    with pytest.raises(ValueError, match="comes from"):
        current_source_provenance(REPO_ROOT)


def test_dirty_git_tree_is_rejected_before_attestation(monkeypatch):
    class Completed:
        stdout = " M code/trec_rag/deterministic_sparse_v2.py\n"

    monkeypatch.setattr(
        provenance_module.subprocess,
        "run",
        lambda *_args, **_kwargs: Completed(),
    )

    with pytest.raises(ValueError, match="clean source tree"):
        current_source_provenance(REPO_ROOT)


def test_runtime_provenance_is_stable_across_module_import_order(
    monkeypatch,
    tmp_path,
):
    _fake_lucene_runtime(monkeypatch, tmp_path)
    module_name = "trec_rag.det_sparse_v2_selection"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    before_import = current_runtime_provenance(REPO_ROOT)

    loaded_later = ModuleType(module_name)
    loaded_later.__file__ = str(REPO_ROOT / MODULE_BINDINGS[module_name])
    monkeypatch.setitem(sys.modules, module_name, loaded_later)

    assert current_runtime_provenance(REPO_ROOT) == before_import
    assert set(before_import["environment_file_sha256"]) == {
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    }
    assert before_import["lucene_analyzer_runtime"]["image_digest"] == (
        provenance_module.LUCENE_IMAGE_DIGEST
    )


def test_runtime_provenance_rejects_tampered_lucene_artifact(monkeypatch, tmp_path):
    cache_dir, _container = _fake_lucene_runtime(monkeypatch, tmp_path)
    current_runtime_provenance(REPO_ROOT)
    (cache_dir / "lib/lucene-core-10.4.0.jar").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        current_runtime_provenance(REPO_ROOT)


def test_runtime_provenance_rejects_mutable_image_reference(monkeypatch, tmp_path):
    _cache_dir, container = _fake_lucene_runtime(monkeypatch, tmp_path)
    container["Config"]["Image"] = "docker.io/library/eclipse-temurin:21-jdk"

    with pytest.raises(ValueError, match="immutable image reference"):
        current_runtime_provenance(REPO_ROOT)

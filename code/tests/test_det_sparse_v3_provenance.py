from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

import trec_rag.det_sparse_v2_provenance as v2_provenance
import trec_rag.det_sparse_v3_provenance as provenance_module
from trec_rag.det_sparse_v3_provenance import (
    FROZEN_DESIGN_PATH,
    FROZEN_DESIGN_SHA256,
    LUCENE_ATTESTATION_MODULE,
    LUCENE_ATTESTATION_SOURCE_PATH,
    LUCENE_ATTESTATION_SOURCE_SHA256,
    MODULE_BINDINGS,
    SOURCE_FILES,
    current_runtime_provenance,
    current_source_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_FILES = {
    "code/trec_rag/deterministic_sparse_v3.py",
    "code/trec_rag/det_sparse_v3_config.py",
    "code/trec_rag/det_sparse_v3_selection.py",
    "code/trec_rag/det_sparse_v3_preflight.py",
    "code/trec_rag/det_sparse_v3_provenance.py",
    "configs/det_sparse_v3.yaml",
    FROZEN_DESIGN_PATH,
}
SHARED_FILES = {
    "code/trec_rag/__init__.py",
    "code/trec_rag/deterministic_sparse.py",
    "code/trec_rag/det_sparse_config.py",
    "code/trec_rag/det_sparse_freeze.py",
    LUCENE_ATTESTATION_SOURCE_PATH,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        raise AssertionError(f"unexpected git arguments: {arguments!r}")

    monkeypatch.setattr(provenance_module, "_git", fake_git)


def _synthetic_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Materialize only inert source bytes; never copy data or experiment files."""

    root = tmp_path / "synthetic-repo"
    pinned_files = {
        FROZEN_DESIGN_PATH,
        LUCENE_ATTESTATION_SOURCE_PATH,
    }
    for relative in SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            (REPO_ROOT / relative).read_bytes()
            if relative in pinned_files
            else f"synthetic source: {relative}\n".encode("utf-8")
        )
        path.write_bytes(payload)
    for relative in v2_provenance.ENVIRONMENT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic environment: {relative}\n".encode("utf-8"))

    # Source provenance validates every bound module that happens to be loaded.
    # Point only those in-memory test modules at their inert synthetic files.
    for module_name, relative in MODULE_BINDINGS.items():
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(
                module,
                "__file__",
                str(root / relative),
                raising=False,
            )
    return root


def _fake_lucene_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    cache_dir = tmp_path / "cache" / "lucene-analyzer" / "10.4.0"
    payloads = {
        "lib/lucene-analysis-common-10.4.0.jar": b"analysis-common",
        "lib/lucene-core-10.4.0.jar": b"lucene-core",
        "classes/AnalyzerServer$1.class": b"inner-class",
        "classes/AnalyzerServer.class": b"main-class",
    }
    expected_hashes: dict[str, str] = {}
    for relative, payload in payloads.items():
        path = cache_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected_hashes[relative] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        v2_provenance,
        "LUCENE_RUNTIME_ARTIFACT_SHA256",
        expected_hashes,
    )
    monkeypatch.setattr(v2_provenance, "_lucene_cache_dir", lambda _root: cache_dir)
    container: dict[str, object] = {
        "Name": v2_provenance.LUCENE_CONTAINER_NAME,
        "Image": v2_provenance.LUCENE_IMAGE_ID,
        "ImageDigest": v2_provenance.LUCENE_IMAGE_DIGEST,
        "Config": {
            "Image": v2_provenance.LUCENE_IMAGE_REFERENCE,
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
            "Ports": {
                "18081/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18081"}]
            }
        },
    }
    monkeypatch.setattr(
        v2_provenance,
        "_inspect_lucene_container",
        lambda: container,
    )
    return cache_dir, container


def test_frozen_contract_and_lucene_reuse_hashes_match_reviewed_bytes():
    assert _sha256(REPO_ROOT / FROZEN_DESIGN_PATH) == FROZEN_DESIGN_SHA256
    assert (
        _sha256(REPO_ROOT / LUCENE_ATTESTATION_SOURCE_PATH)
        == LUCENE_ATTESTATION_SOURCE_SHA256
    )


def test_v3_source_boundary_is_exact_complete_and_import_bound(
    tmp_path,
    monkeypatch,
):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)

    provenance = current_source_provenance(root)

    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    assert set(SOURCE_FILES) == V3_FILES | SHARED_FILES
    assert set(provenance["source_files_sha256"]) == set(SOURCE_FILES)
    assert provenance["source_files_sha256"][FROZEN_DESIGN_PATH] == (
        FROZEN_DESIGN_SHA256
    )
    assert provenance["source_files_sha256"][LUCENE_ATTESTATION_SOURCE_PATH] == (
        LUCENE_ATTESTATION_SOURCE_SHA256
    )
    assert provenance["required_module_bindings"] == dict(MODULE_BINDINGS)
    python_sources = {
        relative for relative in SOURCE_FILES if relative.endswith(".py")
    }
    assert set(MODULE_BINDINGS.values()) == python_sources
    assert len(MODULE_BINDINGS) == len(set(MODULE_BINDINGS.values()))
    assert provenance["lucene_attestation_reuse"] == {
        "module": LUCENE_ATTESTATION_MODULE,
        "source_path": LUCENE_ATTESTATION_SOURCE_PATH,
        "source_sha256": LUCENE_ATTESTATION_SOURCE_SHA256,
        "source_schema_version": v2_provenance.SOURCE_PROVENANCE_SCHEMA_VERSION,
        "runtime_schema_version": v2_provenance.RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "runtime_version": v2_provenance.LUCENE_RUNTIME_VERSION,
        "image_digest": v2_provenance.LUCENE_IMAGE_DIGEST,
    }


def test_every_local_import_in_the_boundary_is_explicitly_bound():
    for relative in SOURCE_FILES:
        if not relative.endswith(".py"):
            continue
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        # Only import-time dependencies belong to this closure.  The shared
        # freezer retains legacy execution helpers with function-local imports,
        # but v3 calls only create/verify and must not bind that dormant v1 arm.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "trec_rag" or node.module.startswith("trec_rag."):
                    imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name == "trec_rag"
                    or alias.name.startswith("trec_rag.")
                )
        assert imported_modules <= set(MODULE_BINDINGS), (
            relative,
            sorted(imported_modules - set(MODULE_BINDINGS)),
        )


def test_v3_never_binds_legacy_experiment_artifacts():
    forbidden_paths = {
        "configs/det_sparse_v1.yaml",
        "configs/det_sparse_v2.yaml",
        "code/trec_rag/deterministic_sparse_v2.py",
        "code/trec_rag/det_sparse_v2_config.py",
        "code/trec_rag/det_sparse_v2_selection.py",
        "code/trec_rag/det_sparse_v2_preflight.py",
    }
    forbidden_prefixes = (
        "cache/",
        "outputs/",
        "reports/experiments/",
    )

    assert not (set(SOURCE_FILES) & forbidden_paths)
    assert not any(relative.startswith(forbidden_prefixes) for relative in SOURCE_FILES)
    assert {
        relative for relative in SOURCE_FILES if "det_sparse_v2" in relative
    } == {LUCENE_ATTESTATION_SOURCE_PATH}


def test_v3_bindings_are_import_order_independent(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    module_name = "trec_rag.det_sparse_v3_selection"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    before = current_source_provenance(root)
    loaded = ModuleType(module_name)
    loaded.__file__ = str(root / MODULE_BINDINGS[module_name])
    monkeypatch.setitem(sys.modules, module_name, loaded)

    assert current_source_provenance(root) == before


@pytest.mark.parametrize(
    "module_name",
    (
        "trec_rag",
        "trec_rag.det_sparse_v3_preflight",
        "trec_rag.det_sparse_v3_selection",
        LUCENE_ATTESTATION_MODULE,
    ),
)
def test_relocated_loaded_module_is_rejected(
    module_name,
    tmp_path,
    monkeypatch,
):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    relocated = ModuleType(module_name)
    relocated.__file__ = f"/tmp/relocated/{module_name}.py"
    monkeypatch.setitem(sys.modules, module_name, relocated)

    with pytest.raises(ValueError, match="comes from"):
        current_source_provenance(root)


def test_loaded_module_without_source_path_is_rejected(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    module_name = "trec_rag.det_sparse_v3_selection"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(ValueError, match="lacks a source path"):
        current_source_provenance(root)


def test_dirty_tree_is_rejected_before_any_hashing(tmp_path, monkeypatch):
    def dirty_git(_root: Path, *arguments: str) -> str:
        assert arguments == ("status", "--porcelain")
        return "?? code/trec_rag/det_sparse_v3_preflight.py"

    monkeypatch.setattr(provenance_module, "_git", dirty_git)
    monkeypatch.setattr(
        provenance_module,
        "_sha256",
        lambda _path: pytest.fail("dirty source must be rejected before hashing"),
    )

    with pytest.raises(ValueError, match="clean source tree"):
        current_source_provenance(tmp_path)


def test_missing_v3_preflight_is_rejected(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    (root / "code/trec_rag/det_sparse_v3_preflight.py").unlink()

    with pytest.raises(ValueError, match="required source file is missing"):
        current_source_provenance(root)


def test_aliased_source_file_is_rejected(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    target = root / "outside.py"
    target.write_text("substituted\n", encoding="utf-8")
    source = root / "code/trec_rag/det_sparse_v3_config.py"
    source.unlink()
    source.symlink_to(target)

    with pytest.raises(ValueError, match="missing or aliased"):
        current_source_provenance(root)


def test_changed_frozen_design_is_rejected(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    (root / FROZEN_DESIGN_PATH).write_bytes(b"changed frozen contract")

    with pytest.raises(ValueError, match="frozen v3 design hash mismatch"):
        current_source_provenance(root)


@pytest.mark.parametrize("entrypoint", ("source", "runtime"))
def test_changed_lucene_reuse_source_is_rejected(
    entrypoint,
    tmp_path,
    monkeypatch,
):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _fake_clean_git(monkeypatch)
    (root / LUCENE_ATTESTATION_SOURCE_PATH).write_bytes(b"substituted helper")
    call = (
        current_source_provenance
        if entrypoint == "source"
        else current_runtime_provenance
    )

    with pytest.raises(
        ValueError,
        match="Lucene attestation reuse source hash mismatch",
    ):
        call(root)


def test_runtime_reuses_exact_digest_pinned_lucene_attestation(
    tmp_path,
    monkeypatch,
):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _cache_dir, _container = _fake_lucene_runtime(monkeypatch, tmp_path)

    runtime = current_runtime_provenance(root)

    assert runtime["schema_version"] == "det_sparse_v3_runtime_provenance_v1"
    assert runtime["lucene_attestation_reuse"]["source_sha256"] == (
        LUCENE_ATTESTATION_SOURCE_SHA256
    )
    assert runtime["lucene_attestation_reuse"]["image_digest"] == (
        v2_provenance.LUCENE_IMAGE_DIGEST
    )
    assert runtime["lucene_analyzer_runtime"]["image_digest"] == (
        v2_provenance.LUCENE_IMAGE_DIGEST
    )
    assert set(runtime["environment_file_sha256"]) == {
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    }


def test_runtime_rejects_relocated_lucene_attestation_module(
    tmp_path,
    monkeypatch,
):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    relocated = ModuleType(LUCENE_ATTESTATION_MODULE)
    relocated.__file__ = "/tmp/substituted_det_sparse_v2_provenance.py"
    monkeypatch.setitem(sys.modules, LUCENE_ATTESTATION_MODULE, relocated)

    with pytest.raises(ValueError, match="comes from"):
        current_runtime_provenance(root)


def test_runtime_rejects_tampered_lucene_artifact(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    cache_dir, _container = _fake_lucene_runtime(monkeypatch, tmp_path)
    current_runtime_provenance(root)
    (cache_dir / "lib/lucene-core-10.4.0.jar").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        current_runtime_provenance(root)


def test_runtime_rejects_mutable_lucene_image_reference(tmp_path, monkeypatch):
    root = _synthetic_source_tree(tmp_path, monkeypatch)
    _cache_dir, container = _fake_lucene_runtime(monkeypatch, tmp_path)
    config = container["Config"]
    assert isinstance(config, dict)
    config["Image"] = "docker.io/library/eclipse-temurin:21-jdk"

    with pytest.raises(ValueError, match="immutable image reference"):
        current_runtime_provenance(root)

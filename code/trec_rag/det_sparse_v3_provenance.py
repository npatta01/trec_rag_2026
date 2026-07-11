"""Clean-source and exact-runtime provenance for ``det_sparse_v3``.

V3 reuses the already byte-attested local Lucene runtime implementation but
defines a new, explicit source/import boundary.  No v1/v2 experiment artifact
is read or accepted by this module.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import trec_rag.det_sparse_v2_provenance as _v2_provenance
from trec_rag.det_sparse_v2_provenance import (
    ENVIRONMENT_FILES,
    RUNTIME_PACKAGES,
    _git,
    _required_unaliased_file,
    _sha256,
    _validated_git_oid,
)


SOURCE_PROVENANCE_SCHEMA_VERSION = "det_sparse_v3_source_provenance_v1"
RUNTIME_PROVENANCE_SCHEMA_VERSION = "det_sparse_v3_runtime_provenance_v1"
FROZEN_DESIGN_PATH = "docs/superpowers/det_sparse_v3_design.md"
FROZEN_DESIGN_SHA256 = (
    "79e21723f2f8346cb0c0249921f7f6e2eab136f35d739a6c6827638edd100a1d"
)
LUCENE_ATTESTATION_MODULE = "trec_rag.det_sparse_v2_provenance"
LUCENE_ATTESTATION_SOURCE_PATH = "code/trec_rag/det_sparse_v2_provenance.py"
LUCENE_ATTESTATION_SOURCE_SHA256 = (
    "7e73f335d6b012ca44486d54c2a3924555ac880b5908660eefd45c542e2bd8f6"
)

# Keep the complete boundary explicit and independent of import order.  The v2
# provenance implementation is included because v3 reuses only its immutable
# Lucene runtime attestation helpers, not any v2 plan or artifact.
SOURCE_FILES = (
    "code/trec_rag/__init__.py",
    "code/trec_rag/deterministic_sparse_v3.py",
    "code/trec_rag/det_sparse_v3_config.py",
    "code/trec_rag/det_sparse_v3_selection.py",
    "code/trec_rag/det_sparse_v3_preflight.py",
    "code/trec_rag/det_sparse_v3_provenance.py",
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
    "configs/det_sparse_v3.yaml",
    FROZEN_DESIGN_PATH,
)

_MODULE_BINDING_ITEMS = (
    ("trec_rag", "code/trec_rag/__init__.py"),
    ("trec_rag.deterministic_sparse_v3", "code/trec_rag/deterministic_sparse_v3.py"),
    ("trec_rag.det_sparse_v3_config", "code/trec_rag/det_sparse_v3_config.py"),
    (
        "trec_rag.det_sparse_v3_selection",
        "code/trec_rag/det_sparse_v3_selection.py",
    ),
    (
        "trec_rag.det_sparse_v3_preflight",
        "code/trec_rag/det_sparse_v3_preflight.py",
    ),
    (
        "trec_rag.det_sparse_v3_provenance",
        "code/trec_rag/det_sparse_v3_provenance.py",
    ),
    ("trec_rag.deterministic_sparse", "code/trec_rag/deterministic_sparse.py"),
    ("trec_rag.det_sparse_config", "code/trec_rag/det_sparse_config.py"),
    ("trec_rag.det_sparse_freeze", "code/trec_rag/det_sparse_freeze.py"),
    (
        LUCENE_ATTESTATION_MODULE,
        LUCENE_ATTESTATION_SOURCE_PATH,
    ),
    ("trec_rag.evaluation", "code/trec_rag/evaluation.py"),
    ("trec_rag.query_analyzer", "code/trec_rag/query_analyzer.py"),
    ("trec_rag.pipeline_models", "code/trec_rag/pipeline_models.py"),
    ("trec_rag.query_planner", "code/trec_rag/query_planner.py"),
    ("trec_rag.query_schema_compat", "code/trec_rag/query_schema_compat.py"),
    ("trec_rag.topics", "code/trec_rag/topics.py"),
    ("trec_rag.repo_env", "code/trec_rag/repo_env.py"),
)

MODULE_BINDINGS: Mapping[str, str] = MappingProxyType(dict(_MODULE_BINDING_ITEMS))


def _validate_loaded_module_binding(
    root: Path,
    module_name: str,
    relative: str,
    *,
    required: bool,
) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        if required:
            raise ValueError(f"required provenance module is not loaded: {module_name}")
        return
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValueError(f"loaded module lacks a source path: {module_name}")
    actual = Path(module_file).resolve()
    expected = (root / relative).resolve()
    if actual != expected:
        raise ValueError(
            f"loaded module {module_name} comes from {actual}, expected {expected}"
        )


def _validate_loaded_module_bindings(root: Path) -> None:
    for module_name, relative in _MODULE_BINDING_ITEMS:
        _validate_loaded_module_binding(
            root,
            module_name,
            relative,
            required=False,
        )


def _require_pinned_hash(
    hashes: Mapping[str, str],
    relative: str,
    expected: str,
    *,
    label: str,
) -> str:
    observed = hashes.get(relative)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {relative}")
    return observed


def _lucene_attestation_reuse_identity(
    root: Path,
    *,
    observed_sha256: str | None = None,
) -> dict[str, str]:
    if observed_sha256 is None:
        observed_sha256 = _sha256(
            _required_unaliased_file(root, LUCENE_ATTESTATION_SOURCE_PATH)
        )
    if observed_sha256 != LUCENE_ATTESTATION_SOURCE_SHA256:
        raise ValueError(
            "Lucene attestation reuse source hash mismatch: "
            f"{LUCENE_ATTESTATION_SOURCE_PATH}"
        )
    _validate_loaded_module_binding(
        root,
        LUCENE_ATTESTATION_MODULE,
        LUCENE_ATTESTATION_SOURCE_PATH,
        required=True,
    )
    return {
        "module": LUCENE_ATTESTATION_MODULE,
        "source_path": LUCENE_ATTESTATION_SOURCE_PATH,
        "source_sha256": observed_sha256,
        "source_schema_version": _v2_provenance.SOURCE_PROVENANCE_SCHEMA_VERSION,
        "runtime_schema_version": _v2_provenance.RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "runtime_version": _v2_provenance.LUCENE_RUNTIME_VERSION,
        "image_digest": _v2_provenance.LUCENE_IMAGE_DIGEST,
    }


def current_source_provenance(root: Path) -> dict[str, object]:
    """Bind a clean Git tree, exact v3 source bytes, and import locations."""

    root = root.resolve()
    if _git(root, "status", "--porcelain"):
        raise ValueError("det_sparse_v3 preflight requires a clean source tree")
    hashes = {
        relative: _sha256(_required_unaliased_file(root, relative))
        for relative in SOURCE_FILES
    }
    _require_pinned_hash(
        hashes,
        FROZEN_DESIGN_PATH,
        FROZEN_DESIGN_SHA256,
        label="frozen v3 design",
    )
    lucene_reuse = _lucene_attestation_reuse_identity(
        root,
        observed_sha256=_require_pinned_hash(
            hashes,
            LUCENE_ATTESTATION_SOURCE_PATH,
            LUCENE_ATTESTATION_SOURCE_SHA256,
            label="Lucene attestation reuse source",
        ),
    )
    _validate_loaded_module_bindings(root)
    return {
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "commit": _validated_git_oid(_git(root, "rev-parse", "HEAD"), label="commit"),
        "tree": _validated_git_oid(
            _git(root, "rev-parse", "HEAD^{tree}"), label="tree"
        ),
        "source_tree_clean": True,
        "python_import_root": str((root / "code").resolve()),
        "source_files_sha256": hashes,
        "required_module_bindings": dict(_MODULE_BINDING_ITEMS),
        "lucene_attestation_reuse": lucene_reuse,
    }


def current_runtime_provenance(root: Path) -> dict[str, object]:
    """Bind Python plus the exact digest-pinned Lucene runtime."""

    root = root.resolve()
    lucene_reuse = _lucene_attestation_reuse_identity(root)
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    environment_hashes = {
        relative: _sha256(_required_unaliased_file(root, relative))
        for relative in ENVIRONMENT_FILES
    }
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": packages,
        "environment_file_sha256": environment_hashes,
        "lucene_attestation_reuse": lucene_reuse,
        "lucene_analyzer_runtime": _v2_provenance._lucene_runtime_provenance(root),
    }


current_v3_source_provenance = current_source_provenance
current_v3_runtime_provenance = current_runtime_provenance

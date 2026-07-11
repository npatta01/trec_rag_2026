"""Source and runtime provenance for the deterministic sparse v2 pilot.

The v2 planner is deliberately isolated from the cost-bearing v1 execution
path, but its preflight artifacts still need an exact, reproducible identity.
This module therefore keeps the complete source boundary in fixed constants
instead of deriving it from whichever modules happen to have been imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from trec_rag.repo_env import repo_cache_root


SOURCE_PROVENANCE_SCHEMA_VERSION = "det_sparse_v2_source_provenance_v1"
RUNTIME_PROVENANCE_SCHEMA_VERSION = "det_sparse_v2_runtime_provenance_v1"

# Keep this tuple explicit.  In particular, do not replace it with a scan of
# imported modules: doing so would make the attestation depend on import order.
SOURCE_FILES = (
    "code/trec_rag/deterministic_sparse_v2.py",
    "code/trec_rag/det_sparse_v2_config.py",
    "code/trec_rag/det_sparse_v2_selection.py",
    "code/trec_rag/det_sparse_v2_preflight.py",
    "code/trec_rag/det_sparse_v2_provenance.py",
    # Shared implementation boundaries used by the v2 path.
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
    "configs/det_sparse_v2.yaml",
)

_MODULE_BINDING_ITEMS = (
    (
        "trec_rag.deterministic_sparse_v2",
        "code/trec_rag/deterministic_sparse_v2.py",
    ),
    (
        "trec_rag.det_sparse_v2_config",
        "code/trec_rag/det_sparse_v2_config.py",
    ),
    (
        "trec_rag.det_sparse_v2_selection",
        "code/trec_rag/det_sparse_v2_selection.py",
    ),
    (
        "trec_rag.det_sparse_v2_preflight",
        "code/trec_rag/det_sparse_v2_preflight.py",
    ),
    (
        "trec_rag.det_sparse_v2_provenance",
        "code/trec_rag/det_sparse_v2_provenance.py",
    ),
    ("trec_rag.deterministic_sparse", "code/trec_rag/deterministic_sparse.py"),
    ("trec_rag.det_sparse_config", "code/trec_rag/det_sparse_config.py"),
    ("trec_rag.det_sparse_freeze", "code/trec_rag/det_sparse_freeze.py"),
    ("trec_rag.evaluation", "code/trec_rag/evaluation.py"),
    ("trec_rag.query_analyzer", "code/trec_rag/query_analyzer.py"),
    ("trec_rag.pipeline_models", "code/trec_rag/pipeline_models.py"),
    ("trec_rag.query_planner", "code/trec_rag/query_planner.py"),
    ("trec_rag.query_schema_compat", "code/trec_rag/query_schema_compat.py"),
    ("trec_rag.topics", "code/trec_rag/topics.py"),
    ("trec_rag.repo_env", "code/trec_rag/repo_env.py"),
)

# A read-only public mapping prevents accidental in-process mutation of the
# required binding set before asking for an attestation.
MODULE_BINDINGS: Mapping[str, str] = MappingProxyType(dict(_MODULE_BINDING_ITEMS))

ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock", ".python-version")
RUNTIME_PACKAGES = ("PyYAML", "semantic-text-splitter")

LUCENE_RUNTIME_VERSION = "local_lucene_reference_runtime_v1"
LUCENE_CONTAINER_NAME = "trec-rag-lucene-analyzer"
LUCENE_IMAGE_DIGEST = (
    "sha256:1eeacc8c295ed4805f6ffead2417b1936aad296b02ea9e56b457230befc9e98d"
)
LUCENE_IMAGE_ID = (
    "bb77d4704c9688b850e05d48d0f3624b14cc48cf0d27df07a99de0b1b40be990"
)
LUCENE_IMAGE_REFERENCE = (
    "docker.io/library/eclipse-temurin@" + LUCENE_IMAGE_DIGEST
)
LUCENE_RUNTIME_ARTIFACT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "lib/lucene-analysis-common-10.4.0.jar": (
            "8e768c9b2a3870f1fc2655181516699e719a56b9aaf8664226a11ae7d90cb4e9"
        ),
        "lib/lucene-core-10.4.0.jar": (
            "8f894d211a8123938ccb9ff6827d136747e0eb6b1782ada6ac9086aa911b52e2"
        ),
        "classes/AnalyzerServer$1.class": (
            "5cae0218cee91a48b095fb3524b909dda260a42065a70990a6c72eb0ace2bdc1"
        ),
        "classes/AnalyzerServer.class": (
            "17fa5fa403cceced39a0858e17ec8229ec818888bb2a88b3bdf7fe931a14927b"
        ),
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_unaliased_file(root: Path, relative: str) -> Path:
    """Resolve a required regular file while rejecting symlink substitution."""

    candidate = root / relative
    lexical = Path(os.path.abspath(candidate))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"required source file is missing: {relative}") from exc
    if resolved != lexical or not resolved.is_file():
        raise ValueError(f"required source file is missing or aliased: {relative}")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validated_git_oid(value: str, *, label: str) -> str:
    normalized = value.lower()
    if (
        len(normalized) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"Git returned an invalid {label} object ID")
    return normalized


def _validate_loaded_module_bindings(root: Path) -> None:
    """Reject a loaded bound module whose source is outside this checkout."""

    for module_name, relative in _MODULE_BINDING_ITEMS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ValueError(f"loaded module lacks a source path: {module_name}")
        actual = Path(module_file).resolve()
        expected = (root / relative).resolve()
        if actual != expected:
            raise ValueError(
                f"loaded module {module_name} comes from {actual}, expected {expected}"
            )


def _lucene_cache_dir(root: Path) -> Path:
    return repo_cache_root(root) / "lucene-analyzer" / "10.4.0"


def _required_runtime_file(cache_dir: Path, relative: str) -> Path:
    candidate = cache_dir / relative
    lexical = Path(os.path.abspath(candidate))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"required Lucene runtime artifact is missing: {relative}") from exc
    if resolved != lexical or not resolved.is_file():
        raise ValueError(
            f"required Lucene runtime artifact is missing or aliased: {relative}"
        )
    return resolved


def _inspect_lucene_container() -> dict[str, object]:
    completed = subprocess.run(
        ("podman", "inspect", LUCENE_CONTAINER_NAME),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Lucene container inspection is not valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("Lucene container inspection must contain exactly one object")
    return raw[0]


def _lucene_runtime_provenance(root: Path) -> dict[str, object]:
    """Bind the exact running Java image, JARs, and compiled server bytes."""

    cache_dir = _lucene_cache_dir(root).resolve()
    artifact_hashes: dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}
    for relative, expected_sha256 in LUCENE_RUNTIME_ARTIFACT_SHA256.items():
        path = _required_runtime_file(cache_dir, relative)
        observed_sha256 = _sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"Lucene runtime artifact hash mismatch: {relative}")
        artifact_hashes[relative] = observed_sha256
        artifact_sizes[relative] = path.stat().st_size

    container = _inspect_lucene_container()
    if container.get("Name") != LUCENE_CONTAINER_NAME:
        raise ValueError("Lucene analyzer container name differs from the frozen name")
    if container.get("Image") != LUCENE_IMAGE_ID:
        raise ValueError("Lucene analyzer image ID differs from the frozen image")
    if container.get("ImageDigest") != LUCENE_IMAGE_DIGEST:
        raise ValueError("Lucene analyzer image digest differs from the frozen digest")
    config = container.get("Config")
    if not isinstance(config, dict):
        raise ValueError("Lucene analyzer container config is missing")
    if config.get("Image") != LUCENE_IMAGE_REFERENCE:
        raise ValueError("Lucene analyzer did not launch from the immutable image reference")
    if config.get("Cmd") != [
        "java",
        "-cp",
        "/build/classes:/build/lib/*",
        "AnalyzerServer",
    ]:
        raise ValueError("Lucene analyzer launch command differs from the frozen command")
    environment = config.get("Env")
    if not isinstance(environment, list) or not {
        "ANALYZER_PORT=18081",
        "ANALYZER_INDEX_ID=hosted_climbmix_unknown_revision",
    }.issubset(environment):
        raise ValueError("Lucene analyzer environment differs from the frozen contract")
    state = container.get("State")
    if not isinstance(state, dict) or state.get("Running") is not True:
        raise ValueError("Lucene analyzer container is not running")

    mounts = container.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("Lucene analyzer container mounts are missing")
    build_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == "/build"
    ]
    if (
        len(build_mounts) != 1
        or Path(str(build_mounts[0].get("Source"))).resolve() != cache_dir
        or build_mounts[0].get("RW") is not False
    ):
        raise ValueError("Lucene analyzer /build mount is not the frozen read-only cache")
    network = container.get("NetworkSettings")
    ports = network.get("Ports") if isinstance(network, dict) else None
    if ports != {"18081/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18081"}]}:
        raise ValueError("Lucene analyzer port is not the exact loopback binding")

    return {
        "runtime_version": LUCENE_RUNTIME_VERSION,
        "container_name": LUCENE_CONTAINER_NAME,
        "image_reference": LUCENE_IMAGE_REFERENCE,
        "image_digest": LUCENE_IMAGE_DIGEST,
        "image_id": LUCENE_IMAGE_ID,
        "cache_dir": str(cache_dir),
        "artifact_sha256": artifact_hashes,
        "artifact_size": artifact_sizes,
        "launch_command": config["Cmd"],
        "loopback_port": "127.0.0.1:18081",
        "build_mount_read_only": True,
    }


def current_source_provenance(root: Path) -> dict[str, object]:
    """Bind a clean Git tree, exact v2 source bytes, and import locations."""

    root = root.resolve()
    if _git(root, "status", "--porcelain"):
        raise ValueError("det_sparse_v2 preflight requires a clean source tree")

    hashes = {
        relative: _sha256(_required_unaliased_file(root, relative))
        for relative in SOURCE_FILES
    }
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
        # Always return the complete mapping, including modules not yet loaded.
        "required_module_bindings": dict(_MODULE_BINDING_ITEMS),
    }


def current_runtime_provenance(root: Path) -> dict[str, object]:
    """Bind the interpreter, dependency versions, and pinned environment files."""

    root = root.resolve()
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
        "lucene_analyzer_runtime": _lucene_runtime_provenance(root),
    }


# Explicit aliases make call sites self-documenting while retaining the
# familiar v1-style API names above.
current_v2_source_provenance = current_source_provenance
current_v2_runtime_provenance = current_runtime_provenance

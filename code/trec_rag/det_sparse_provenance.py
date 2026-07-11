"""Source/import provenance for the cost-bearing deterministic sparse path."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SOURCE_FILES = (
    "code/trec_rag/det_sparse_run.py",
    "code/trec_rag/det_sparse_budget.py",
    "code/trec_rag/det_sparse_transport.py",
    "code/trec_rag/det_sparse_provenance.py",
    "code/trec_rag/det_sparse_freeze.py",
    "code/trec_rag/det_sparse_preflight.py",
    "code/trec_rag/det_sparse_ledger.py",
    "code/trec_rag/det_sparse_arms.py",
    "code/trec_rag/det_sparse_config.py",
    "code/trec_rag/deterministic_sparse.py",
    "code/trec_rag/ranking.py",
    "code/trec_rag/query_analyzer.py",
    "code/trec_rag/pipeline_models.py",
    "code/trec_rag/query_planner.py",
    "code/trec_rag/evaluation.py",
    "code/trec_rag/topics.py",
    "code/trec_rag/repo_env.py",
    "code/tools/lucene_analyzer/AnalyzerServer.java",
    "code/tools/run_lucene_analyzer.sh",
    "configs/det_sparse_v1.yaml",
)

MODULE_BINDINGS = {
    "trec_rag.det_sparse_run": "code/trec_rag/det_sparse_run.py",
    "trec_rag.det_sparse_budget": "code/trec_rag/det_sparse_budget.py",
    "trec_rag.det_sparse_transport": "code/trec_rag/det_sparse_transport.py",
    "trec_rag.det_sparse_provenance": "code/trec_rag/det_sparse_provenance.py",
    "trec_rag.det_sparse_freeze": "code/trec_rag/det_sparse_freeze.py",
    "trec_rag.det_sparse_preflight": "code/trec_rag/det_sparse_preflight.py",
    "trec_rag.det_sparse_ledger": "code/trec_rag/det_sparse_ledger.py",
    "trec_rag.det_sparse_arms": "code/trec_rag/det_sparse_arms.py",
    "trec_rag.det_sparse_config": "code/trec_rag/det_sparse_config.py",
    "trec_rag.deterministic_sparse": "code/trec_rag/deterministic_sparse.py",
    "trec_rag.ranking": "code/trec_rag/ranking.py",
    "trec_rag.query_analyzer": "code/trec_rag/query_analyzer.py",
    "trec_rag.pipeline_models": "code/trec_rag/pipeline_models.py",
    "trec_rag.query_planner": "code/trec_rag/query_planner.py",
    "trec_rag.evaluation": "code/trec_rag/evaluation.py",
    "trec_rag.topics": "code/trec_rag/topics.py",
    "trec_rag.repo_env": "code/trec_rag/repo_env.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_provenance(root: Path) -> dict[str, object]:
    """Bind clean Git identity, exact source bytes, and loaded module locations."""

    root = root.resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    if git("status", "--porcelain"):
        raise ValueError("formal execution requires a clean source tree")
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = (root / relative).resolve()
        if path != (root / relative).absolute() or not path.is_file():
            raise ValueError(f"required source file is missing or aliased: {relative}")
        hashes[relative] = _sha256(path)
    for module_name, relative in MODULE_BINDINGS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError(f"loaded module lacks a source path: {module_name}")
        actual = Path(module_file).resolve()
        expected = (root / relative).resolve()
        if actual != expected:
            raise ValueError(
                f"loaded module {module_name} comes from {actual}, expected {expected}"
            )
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "source_tree_clean": True,
        "python_import_root": str((root / "code").resolve()),
        "source_files_sha256": hashes,
        "required_module_bindings": dict(MODULE_BINDINGS),
    }


def current_runtime_provenance(root: Path) -> dict[str, object]:
    packages = {}
    for package in ("PyYAML", "pytest", "semantic-text-splitter"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    files = {}
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        path = root / name
        if path.is_file():
            files[name] = _sha256(path)
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "environment_file_sha256": files,
    }

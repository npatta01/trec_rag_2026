from __future__ import annotations

from pathlib import Path

import pytest

import trec_rag.det_sparse_budget as budget_module
import trec_rag.det_sparse_transport as transport_module
from trec_rag.det_sparse_provenance import (
    MODULE_BINDINGS,
    SOURCE_FILES,
    current_source_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_FILES = {
    "code/trec_rag/det_sparse_budget.py",
    "code/trec_rag/det_sparse_transport.py",
    "code/trec_rag/det_sparse_provenance.py",
}


def _fake_clean_git(monkeypatch):
    class Completed:
        def __init__(self, stdout):
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

    monkeypatch.setattr("trec_rag.det_sparse_provenance.subprocess.run", run)


def test_boundary_modules_are_explicitly_hashed_and_import_bound(monkeypatch):
    _fake_clean_git(monkeypatch)

    provenance = current_source_provenance(REPO_ROOT)

    assert BOUNDARY_FILES.issubset(SOURCE_FILES)
    assert BOUNDARY_FILES.issubset(provenance["source_files_sha256"])
    assert MODULE_BINDINGS["trec_rag.det_sparse_budget"] in BOUNDARY_FILES
    assert MODULE_BINDINGS["trec_rag.det_sparse_transport"] in BOUNDARY_FILES
    assert MODULE_BINDINGS["trec_rag.det_sparse_provenance"] in BOUNDARY_FILES


@pytest.mark.parametrize("module", [budget_module, transport_module])
def test_relocated_boundary_module_is_rejected(monkeypatch, module):
    _fake_clean_git(monkeypatch)
    monkeypatch.setattr(module, "__file__", "/tmp/substituted_boundary.py")

    with pytest.raises(ValueError, match="comes from"):
        current_source_provenance(REPO_ROOT)

"""Markers for tests that cannot run, or are known to fail, off Linux.

The pipeline was built and run on Linux and parts of it require Linux by
design, so a macOS checkout cannot execute every test.  Each marker here names
one specific reason and carries the narrowest condition that detects it, so a
Linux host still runs everything and a macOS host reports a skip that says what
is missing instead of a failure that looks like a regression.

`code/trec_rag/README.md` explains each reason in prose.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


@cache
def _bash_major_version() -> int:
    """Return bash's major version, or 0 when it cannot be determined."""
    try:
        completed = subprocess.run(
            ["bash", "-c", "echo ${BASH_VERSINFO[0]}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if completed.returncode != 0:
        return 0
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return 0


@cache
def _reranker_dependencies_installed() -> bool:
    try:
        version("transformers")
    except PackageNotFoundError:
        return False
    return True


@cache
def _temporary_directory_is_symlinked() -> bool:
    """True when the temporary root reaches its real path through a symlink.

    macOS resolves ``/var`` to ``/private/var``, so a path built from
    :func:`tempfile.gettempdir` and the same path after ``resolve()`` disagree.
    """
    root = Path(tempfile.gettempdir())
    return root.resolve() != root


REQUIRES_LINUX_PUBLICATION = pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "publishing a bundle or an offline topic needs Linux: "
        "competition_retrieval uses renameat2 and competition_debug_bundle "
        "refuses a non-Linux host, both so a directory can be published "
        "without replacing whatever is already at the destination. The macOS "
        "equivalent is renameatx_np with RENAME_EXCL, which nothing here uses "
        "yet."
    ),
)

REQUIRES_BASH_4 = pytest.mark.skipif(
    _bash_major_version() < 4,
    reason=(
        "the shard wrapper scripts use `declare -A`, which needs bash 4 or "
        "newer; macOS ships bash 3.2. Install a newer bash (`brew install "
        "bash`) and put it ahead of /bin on PATH to run these."
    ),
)

REQUIRES_RERANKER_DEPENDENCIES = pytest.mark.skipif(
    not _reranker_dependencies_installed(),
    reason=(
        "`transformers` is pinned to `sys_platform == 'linux' and "
        "platform_machine == 'x86_64'` in pyproject.toml, so local reranking "
        "is only installable on a Linux x86_64 host."
    ),
)

REQUIRES_PROCFS = pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason=(
        "this test names the file behind a descriptor by reading "
        "/proc/self/fd, which only Linux provides. The macOS equivalent is "
        "fcntl F_GETPATH."
    ),
)

REQUIRES_FORK_START_METHOD = pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason=(
        "this test relies on a child process inheriting state, which needs "
        "the `fork` start method; macOS defaults to `spawn` and segfaults "
        "when forking a process that already holds threads."
    ),
)

# Not a platform limit: the production code compares a resolved path against an
# unresolved root, so any host whose temporary directory sits behind a symlink
# fails. Kept strict so the marker turns into a failure the day it is fixed.
XFAIL_SYMLINKED_TEMPORARY_DIRECTORY = pytest.mark.xfail(
    _temporary_directory_is_symlinked(),
    strict=True,
    reason=(
        "suspected portability bug, not a platform limit: the bundle and "
        "segmentation validators reject a path component that is a symbolic "
        "link and compare realpath output against an unresolved root, so they "
        "fail wherever the temporary directory is symlinked (macOS puts it "
        "under /var, a link to /private/var)."
    ),
)

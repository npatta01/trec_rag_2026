"""Guards on the platform markers themselves.

A marker that skips a test is a test that stops reporting, so the two ways
these could quietly go wrong are worth pinning: a condition that is true
everywhere would hide the test on Linux too, and a non-strict xfail would let
the suspected portability bug behind
`XFAIL_SYMLINKED_TEMPORARY_DIRECTORY` turn into expected behaviour the day
somebody fixes it.
"""

from __future__ import annotations

import pathlib
import sys

import platform_support
from platform_support import (
    REQUIRES_BASH_4,
    REQUIRES_FORK_START_METHOD,
    REQUIRES_LINUX_PUBLICATION,
    REQUIRES_PROCFS,
    REQUIRES_RERANKER_DEPENDENCIES,
    XFAIL_SYMLINKED_TEMPORARY_DIRECTORY,
)

ALL_MARKERS = (
    REQUIRES_BASH_4,
    REQUIRES_FORK_START_METHOD,
    REQUIRES_LINUX_PUBLICATION,
    REQUIRES_PROCFS,
    REQUIRES_RERANKER_DEPENDENCIES,
    XFAIL_SYMLINKED_TEMPORARY_DIRECTORY,
)


def test_every_marker_carries_a_host_dependent_condition() -> None:
    """No marker may be unconditional; each must read something off the host."""
    for marker in ALL_MARKERS:
        condition = marker.args[0]
        assert isinstance(condition, bool), (
            f"{marker} must resolve its condition to a bool at import, so a "
            "Linux host evaluates it to False rather than skipping blindly"
        )


def test_every_marker_explains_itself() -> None:
    for marker in ALL_MARKERS:
        reason = marker.kwargs["reason"]
        assert len(reason) >= 60, f"{marker} needs a reason that says what is missing"


def test_the_suspected_bug_stays_strict() -> None:
    """A fix must turn this xfail into a failure, not into silence."""
    assert XFAIL_SYMLINKED_TEMPORARY_DIRECTORY.kwargs["strict"] is True


def test_conditions_hold_on_a_linux_host() -> None:
    """Every condition must be false on Linux, or it hides tests there too.

    The markers freeze their conditions at import, so this re-derives each one
    the way a Linux host would see it: procfs present, `fork` available, a real
    temporary directory, and `transformers` installed from the Linux x86_64
    pin. Only the bash version is a property of the local install rather than
    of the platform.
    """
    linux_view = {
        "procfs": not pathlib.Path("/proc/self/fd").is_dir(),
        "fork": platform_support.multiprocessing.get_start_method() != "fork",
        "symlinked tmp": platform_support._temporary_directory_is_symlinked(),
    }
    if sys.platform == "linux":
        assert not any(linux_view.values()), (
            f"a Linux host must not satisfy any skip condition: {linux_view}"
        )
    else:
        # Off Linux the conditions are expected to fire; assert they are the
        # reason the markers exist rather than silently inverted.
        assert REQUIRES_PROCFS.args[0] == linux_view["procfs"]
        assert REQUIRES_FORK_START_METHOD.args[0] == linux_view["fork"]
        assert (
            XFAIL_SYMLINKED_TEMPORARY_DIRECTORY.args[0] == linux_view["symlinked tmp"]
        )

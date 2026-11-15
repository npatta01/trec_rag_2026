"""Guards on the platform markers themselves.

A marker that skips a test is a test that stops reporting, so the way these
could quietly go wrong is a condition that is true everywhere: it would hide
the test on Linux too. Each condition must therefore be read off the host and
must be false on Linux.
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
)

ALL_MARKERS = (
    REQUIRES_BASH_4,
    REQUIRES_FORK_START_METHOD,
    REQUIRES_LINUX_PUBLICATION,
    REQUIRES_PROCFS,
    REQUIRES_RERANKER_DEPENDENCIES,
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


def test_conditions_hold_on_a_linux_host() -> None:
    """Every platform condition must be false on Linux.

    Only the bash version is a property of the local install rather than of the
    platform, so it is excluded: a Linux host with bash 3 would legitimately
    skip those.
    """
    platform_conditions = {
        "publication": sys.platform != "linux",
        "procfs": not pathlib.Path("/proc/self/fd").is_dir(),
        "fork": platform_support.multiprocessing.get_start_method() != "fork",
    }
    if sys.platform == "linux":
        assert not any(platform_conditions.values()), (
            f"a Linux host must not satisfy any platform condition: "
            f"{platform_conditions}"
        )
    else:
        # Off Linux they are expected to fire; assert the markers read the same
        # facts rather than having been inverted or pinned to a constant.
        assert REQUIRES_LINUX_PUBLICATION.args[0] == platform_conditions["publication"]
        assert REQUIRES_PROCFS.args[0] == platform_conditions["procfs"]
        assert REQUIRES_FORK_START_METHOD.args[0] == platform_conditions["fork"]

"""Contract tests for the hardware bootstrap scripts.

These are hermetic: the prefetch test executes the setup script's embedded
Python against a stubbed ``huggingface_hub``, so it never reaches the network,
never writes to a model cache, and never needs torch installed.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
import re
import sys
import types

import pytest

from trec_rag.evidence_local import (
    MINILM_MODEL,
    MINILM_REVISION,
    LocalMiniLMSimilarity,
)
from trec_rag.mixedbread_passage_scorer import (
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
    MixedbreadPassageScorer,
)


ROOT = Path(__file__).resolve().parents[2]
SETUP_ENV = ROOT / "code" / "tools" / "setup_env.sh"
SETUP_CUDA = ROOT / "code" / "tools" / "setup_cuda_env.sh"
VERIFY_TORCH_GROUPS = ROOT / "code" / "tools" / "verify_torch_groups.sh"

# Every snapshot the competition retrieval path loads with local_files_only=True.
# Mixedbread scores passages; MiniLM is constructed later by LocalMiniLMSimilarity
# for evidence selection, so a host missing it fails only after the expensive
# scoring stage.
REQUIRED_OFFLINE_SNAPSHOTS = frozenset(
    {
        (MIXEDBREAD_MODEL, MIXEDBREAD_REVISION),
        (MINILM_MODEL, MINILM_REVISION),
    }
)


def _embedded_python_blocks(script: Path) -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY\n", script.read_text(encoding="utf-8"), re.DOTALL)


def _run_prefetch_block(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Execute the setup script's prefetch block with a recording stub."""
    calls: list[dict[str, object]] = []

    def snapshot_download(model: str, *, revision: str, local_files_only: bool = False) -> str:
        calls.append(
            {
                "model": model,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        if local_files_only and not any(
            call["model"] == model
            and call["revision"] == revision
            and not call["local_files_only"]
            for call in calls[:-1]
        ):
            raise OSError(f"{model} is not in the local cache")
        return f"/stub-cache/{model}@{revision}"

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    blocks = _embedded_python_blocks(SETUP_CUDA)
    prefetch = [block for block in blocks if "snapshot_download" in block]
    assert len(prefetch) == 1, "expected exactly one prefetch block in setup_cuda_env.sh"
    exec(compile(prefetch[0], str(SETUP_CUDA), "exec"), {"__name__": "__main__"})
    return calls


def test_cuda_setup_prefetches_every_required_offline_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _run_prefetch_block(monkeypatch)

    downloaded = {
        (call["model"], call["revision"]) for call in calls if not call["local_files_only"]
    }
    assert downloaded == set(REQUIRED_OFFLINE_SNAPSHOTS)


def test_cuda_setup_verifies_each_snapshot_through_the_offline_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh cache must not pass setup while still failing later in selection."""
    calls = _run_prefetch_block(monkeypatch)

    verified = {
        (call["model"], call["revision"]) for call in calls if call["local_files_only"]
    }
    assert verified == set(REQUIRED_OFFLINE_SNAPSHOTS)


def test_prefetch_block_fails_when_a_snapshot_is_missing_from_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline verification is real: it raises when the download is skipped."""
    calls: list[str] = []

    def snapshot_download(model: str, *, revision: str, local_files_only: bool = False) -> str:
        calls.append(model)
        if local_files_only and model == MINILM_MODEL:
            raise OSError(f"{model} is not in the local cache")
        return f"/stub-cache/{model}@{revision}"

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    prefetch = [
        block for block in _embedded_python_blocks(SETUP_CUDA) if "snapshot_download" in block
    ]
    with pytest.raises(OSError, match=MINILM_MODEL):
        exec(compile(prefetch[0], str(SETUP_CUDA), "exec"), {"__name__": "__main__"})


def _selection_path_request() -> tuple[str, str, bool]:
    """Record the snapshot the evidence-selection path asks for."""
    calls: list[tuple[str, dict[str, object]]] = []

    def loader(model: str, **kwargs: object) -> object:
        calls.append((model, kwargs))
        return object()

    LocalMiniLMSimilarity(device="cpu", loader=loader)._load()
    model, kwargs = calls[0]
    return model, str(kwargs["revision"]), bool(kwargs["local_files_only"])


def _passage_scoring_request(tmp_path: Path) -> tuple[str, str, bool]:
    """Record the snapshot the passage-scoring path asks for."""
    calls: list[dict[str, object]] = []

    def loader(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    scorer = MixedbreadPassageScorer(
        tmp_path / "score-cache",
        device="cpu",
        model_loader=loader,
    )
    # The loader records its arguments before dtype validation rejects the stub.
    with suppress(Exception):
        scorer._get_model()
    kwargs = calls[0]
    return (
        str(kwargs["model_name"]),
        str(kwargs["revision"]),
        bool(kwargs["local_files_only"]),
    )


def test_selection_path_loads_minilm_offline_from_a_prefetched_snapshot() -> None:
    """The MiniLM that selection loads must be one the bootstrap already cached."""
    model, revision, local_files_only = _selection_path_request()

    assert (model, revision) in REQUIRED_OFFLINE_SNAPSHOTS
    assert local_files_only is True


def test_bootstrap_prefetches_exactly_what_the_production_loaders_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the drift gap in both directions, not just against a static list."""
    selection_model, selection_revision, _ = _selection_path_request()
    scoring_model, scoring_revision, _ = _passage_scoring_request(tmp_path)
    requested = {
        (selection_model, selection_revision),
        (scoring_model, scoring_revision),
    }

    calls = _run_prefetch_block(monkeypatch)
    prefetched = {
        (call["model"], call["revision"]) for call in calls if not call["local_files_only"]
    }

    assert requested == prefetched


def test_setup_env_dispatches_to_both_hardware_scripts() -> None:
    text = SETUP_ENV.read_text(encoding="utf-8")

    assert "setup_rocm_ryzen_env.sh" in text
    assert "setup_cuda_env.sh" in text
    # ROCm is checked first: a ROCm host must not be captured by the CUDA branch.
    assert text.index("has_amd_rocm_device; then") < text.index("has_nvidia_cuda_device; then")


def test_hardware_setup_scripts_are_executable() -> None:
    for script in (SETUP_ENV, SETUP_CUDA, VERIFY_TORCH_GROUPS):
        assert script.stat().st_mode & 0o111, f"{script} must be executable"


def test_cuda_group_pins_the_versions_the_rocm_group_resolves_to() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for pin in (
        "sentence-transformers==5.6.0",
        "transformers==5.13.0",
        "numpy==2.5.1",
        "torch==2.9.1 ",
    ):
        assert pin in pyproject, f"cuda group must pin {pin}"
    # One resolution cannot hold both torch builds.
    assert 'conflicts = [[{ group = "rocm" }, { group = "cuda" }]]' in pyproject

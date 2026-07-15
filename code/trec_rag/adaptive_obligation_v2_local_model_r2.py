"""Separately approval-gated local JSON runtime for R2 proposal jobs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .adaptive_obligation_v2_contract import canonical_sha256
from .adaptive_obligation_v2_local_model import load_pinned_local_runtime
from .adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    PRIMARY_JOB_COUNT,
    PRIMARY_MAX_NEW_TOKENS,
    RETRY_MAX_NEW_TOKENS,
    _open_directory_no_symlinks,
)
from .adaptive_obligation_v2_propose_r2 import R2_PROPOSAL_SCHEMA


R2_APPROVAL_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-approval-r2"
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not any(char not in _HEX for char in value)
    )


def _verify_existing_parent_without_symlinks(path: Path) -> None:
    destination = Path(path)
    try:
        if str(destination.resolve(strict=False)) != str(destination):
            raise OSError("ledger path resolves through a symlink")
        descriptor = _open_directory_no_symlinks(destination.parent)
    except (OSError, RuntimeError) as exc:
        raise PermissionError(
            "R2 proposal requires an absolute safe ledger destination"
        ) from exc
    else:
        os.close(descriptor)
    if destination.exists() and (
        destination.is_symlink() or not destination.is_dir()
    ):
        raise PermissionError(
            "R2 proposal requires an absolute safe ledger destination"
        )


def verify_r2_inference_approval(
    approval: object,
    preflight: object,
    *,
    ledger_dir: Path,
) -> dict[str, object]:
    """Require an exact R2 approval bound to its frozen ledger destination."""

    destination = Path(ledger_dir)
    if (
        not isinstance(preflight, Mapping)
        or not destination.is_absolute()
        or preflight.get("ledger_dir") != str(destination)
    ):
        raise PermissionError(
            "R2 proposal requires an absolute safe ledger destination"
        )
    _verify_existing_parent_without_symlinks(destination)
    if (
        not _is_sha256(preflight.get("receipt_sha256"))
        or preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
        or type(preflight.get("primary_call_count")) is not int
        or preflight.get("primary_call_count") != PRIMARY_JOB_COUNT
        or type(preflight.get("retry_call_ceiling")) is not int
        or preflight.get("retry_call_ceiling") != PRIMARY_JOB_COUNT
        or not isinstance(preflight.get("model_snapshot"), Mapping)
    ):
        raise PermissionError("R2 proposal approval required")
    required = {
        "schema_version": R2_APPROVAL_SCHEMA_VERSION,
        "stage": "proposal_r2",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": PRIMARY_JOB_COUNT,
        "retry_call_ceiling": PRIMARY_JOB_COUNT,
        "ledger_dir": str(destination),
        "approved": True,
    }
    if (
        not isinstance(approval, Mapping)
        or dict(approval) != required
        or type(approval.get("primary_call_count")) is not int
        or type(approval.get("retry_call_ceiling")) is not int
        or approval.get("approved") is not True
    ):
        raise PermissionError("R2 proposal approval required")
    return dict(approval)


class R2LocalJsonModel:
    """Replay exact frozen R2 prompt counts through the pinned local runtime."""

    def __init__(
        self,
        *,
        approval: object,
        preflight: Mapping[str, object],
        ledger_dir: Path,
    ) -> None:
        verify_r2_inference_approval(
            approval,
            preflight,
            ledger_dir=ledger_dir,
        )
        jobs = preflight.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != PRIMARY_JOB_COUNT:
            raise ValueError("R2 requires exactly 48 frozen proposal jobs")
        prompt_counts: dict[str, int] = {}
        for job in jobs:
            if not isinstance(job, Mapping):
                raise ValueError("R2 requires unique frozen message hashes and counts")
            messages = job.get("messages")
            prompt_token_count = job.get("prompt_token_count")
            if not isinstance(messages, list):
                raise ValueError("R2 requires unique frozen message hashes and counts")
            messages_sha256 = canonical_sha256(messages)
            if (
                job.get("messages_sha256") != messages_sha256
                or messages_sha256 in prompt_counts
                or type(prompt_token_count) is not int
                or prompt_token_count <= 0
            ):
                raise ValueError("R2 requires unique frozen message hashes and counts")
            prompt_counts[messages_sha256] = prompt_token_count
        manifest = preflight.get("model_snapshot")
        if not isinstance(manifest, Mapping):
            raise PermissionError("R2 proposal approval required")
        self._prompt_counts = prompt_counts
        self._runtime = load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            model_manifest=manifest,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
    ) -> tuple[bytes, int]:
        if canonical_sha256(schema) != canonical_sha256(R2_PROPOSAL_SCHEMA):
            raise ValueError("R2 proposal schema differs")
        prompt_token_count = self._prompt_counts.get(canonical_sha256(messages))
        if prompt_token_count is None:
            raise ValueError("R2 proposal messages differ from frozen jobs")
        if type(max_new_tokens) is not int or max_new_tokens not in (
            PRIMARY_MAX_NEW_TOKENS,
            RETRY_MAX_NEW_TOKENS,
        ):
            raise ValueError("R2 proposal output ceiling differs")
        return self._runtime.generate_json_bytes(  # type: ignore[attr-defined,no-any-return]
            messages,
            schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            expected_prompt_tokens=prompt_token_count,
        )

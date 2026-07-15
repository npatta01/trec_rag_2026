"""Freeze the authenticated, tokenizer-only adaptive-obligation R2 preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS
from .adaptive_obligation_v2_contract import (
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    canonical_sha256,
)
from . import adaptive_obligation_v2_propose as r1


R2_PREFLIGHT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-preflight-r2"
R2_JOB_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-job-r2"
R2_PROMPT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-prompt-r2"
R2_PROMPT_REVISION = "tail-contract-r2"
R2_PROPOSAL_RECEIPT_SCHEMA_VERSION = (
    "adaptive-obligation-v2-proposal-receipt-r2"
)

R2_PROPOSAL_SCHEMA: dict[str, object] = deepcopy(r1.PROPOSAL_SCHEMA)
R2_SYSTEM_INSTRUCTIONS = r1._PROPOSAL_INSTRUCTIONS + """
Before emitting JSON, check the final output_contract in the user payload.
Apply its exact label and scope_rationale limits. If you cannot satisfy that
contract from the supplied evidence, return a schema-valid UNSUPPORTED object."""
R2_OUTPUT_CONTRACT: dict[str, object] = {
    "return": "exactly one JSON object and no prose",
    "label": {"max_unicode_characters": 80, "max_words": 10},
    "scope_rationale": {
        "exact_sentences": 1,
        "max_unicode_characters": 160,
        "max_words": 25,
    },
    "support_unit_ids_for_supported": {
        "minimum_items": 1,
        "maximum_items": 2,
        "source": "supplied evidence_units only",
    },
    "self_check": "silently verify every limit before emitting JSON",
    "on_failure": "return a schema-valid UNSUPPORTED object",
}

_OUTPUT_NAMES = frozenset({"jobs.jsonl", "schema.json", "prompt.json", "receipt.json"})
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "prompt_revision",
        "status",
        "topic_ids",
        "job_count",
        "primary_call_count",
        "retry_call_ceiling",
        "worst_case_call_ceiling",
        "primary_max_new_tokens",
        "retry_max_new_tokens",
        "prompt_token_counts",
        "tokenizer_load_count",
        "model_load_count",
        "inference_count",
        "model",
        "model_revision",
        "model_snapshot",
        "tokenizer_files",
        "tokenizer_contract",
        "tokenizer_identity_sha256",
        "tokenizer_loading",
        "model_construction_allowed",
        "generation_allowed",
        "prompt_sha256",
        "schema_sha256",
        "contract_receipt",
        "code_sha256",
        "ledger_dir",
        "proposal_dir",
        "artifacts",
        "expected_runtime",
        "planned_storage",
        "qrels_opened",
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
        "external_cost_usd",
    }
)

# Stable R1 primitives are deliberately named in this module so tests can replace
# only the production I/O boundary without changing any R1 behavior.
_load_verified_contract = r1._load_verified_contract
_snapshot_inventory = r1._snapshot_inventory
_tokenizer_file_inventory = r1._tokenizer_file_inventory
_tokenizer_contract = r1._tokenizer_contract


def _ordered_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _json_exact(observed: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/number equality coercions."""

    return r1._compact(observed) == r1._compact(expected)


def _static_receipt_metadata(
    *, ledger_dir: Path, proposal_dir: Path
) -> dict[str, object]:
    return {
        "schema_version": R2_PREFLIGHT_SCHEMA_VERSION,
        "prompt_revision": R2_PROMPT_REVISION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": r1.PRIMARY_JOB_COUNT,
        "primary_call_count": r1.PRIMARY_JOB_COUNT,
        "retry_call_ceiling": r1.PRIMARY_JOB_COUNT,
        "worst_case_call_ceiling": r1.PRIMARY_JOB_COUNT * 2,
        "primary_max_new_tokens": r1.PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": r1.RETRY_MAX_NEW_TOKENS,
        "tokenizer_load_count": 1,
        "model_load_count": 0,
        "inference_count": 0,
        "model": r1.MODEL_ID,
        "model_revision": r1.MODEL_REVISION,
        "tokenizer_loading": {
            "backend": "tokenizers",
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_required": False,
            "transformers_required": False,
        },
        "model_construction_allowed": False,
        "generation_allowed": False,
        "ledger_dir": str(ledger_dir),
        "proposal_dir": str(proposal_dir),
        "expected_runtime": {
            "phase": "inference_free_preflight_r2",
            "proposal_calls_executed": 0,
            "validation_calls_executed": 0,
        },
        "planned_storage": {
            "artifact_names": [
                "jobs.jsonl",
                "schema.json",
                "prompt.json",
                "receipt.json",
            ],
            "job_rows": r1.PRIMARY_JOB_COUNT,
            "ledger_dir": str(ledger_dir),
            "proposal_dir": str(proposal_dir),
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }


def _prompt_contract() -> dict[str, object]:
    return {
        "schema_version": R2_PROMPT_SCHEMA_VERSION,
        "prompt_revision": R2_PROMPT_REVISION,
        "instructions": R2_SYSTEM_INSTRUCTIONS,
        "output_contract": R2_OUTPUT_CONTRACT,
        "response_schema_sha256": canonical_sha256(R2_PROPOSAL_SCHEMA),
        "decoding": {
            "do_sample": False,
            "temperature": 0,
            "seed": 0,
            "primary_max_new_tokens": r1.PRIMARY_MAX_NEW_TOKENS,
            "retry_max_new_tokens": r1.RETRY_MAX_NEW_TOKENS,
        },
        "retry_policy": {
            "maximum_retries_per_job": 1,
            "only_if_primary_completion_reaches_ceiling_and_json_is_truncated": True,
            "semantic_or_schema_retry_allowed": False,
            "recursive_repair_allowed": False,
        },
    }


def render_r2_proposal_messages(
    parent: Mapping[str, object],
    reservoir: Mapping[str, object],
    units: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Render one R2 prompt with the compact output contract at the user tail."""

    text = str(parent["text"])
    query = str(parent["query"])
    suffix = f"\n\nExplicit obligation:\n{text}"
    if not query.endswith(suffix):
        raise ValueError("parent query does not preserve the complete O0 suffix")
    payload = {
        "narrative": query[: -len(suffix)],
        "parent_o0": dict(parent),
        "source_fold": reservoir["fold"],
        "reservoir_id": reservoir["reservoir_id"],
        "evidence_units": [dict(unit) for unit in units],
        "response_json_schema": R2_PROPOSAL_SCHEMA,
        "output_contract": R2_OUTPUT_CONTRACT,
    }
    return [
        {"role": "system", "content": R2_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": _ordered_compact(payload)},
    ]


def _validate_contract_rows(contract: object) -> tuple[
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
]:
    r1._reject_protected_contract(contract)
    if not isinstance(contract, Mapping):
        raise ValueError("R2 proposal contract must be an object")
    parents = contract.get("parents")
    reservoirs = contract.get("reservoirs")
    units = contract.get("units")
    if (
        not isinstance(parents, list)
        or not isinstance(reservoirs, list)
        or not isinstance(units, list)
    ):
        raise ValueError("R2 proposal contract rows are missing")
    if any(not isinstance(row, Mapping) for row in [*parents, *reservoirs, *units]):
        raise ValueError("R2 proposal contract rows must be objects")
    receipt = contract.get("receipt")
    if not isinstance(receipt, Mapping) or (
        receipt.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("parent_count") != 24
        or receipt.get("reservoir_count") != r1.PRIMARY_JOB_COUNT
        or receipt.get("qrels_opened") is not False
        or any(receipt.get(name) != 0 for name in r1._ZERO_COUNTERS)
        or receipt.get("tokenizer_load_count") != 0
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("R2 proposal source contract receipt differs")
    if len(parents) != 24 or len(reservoirs) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("R2 proposal contract must contain 24 parents and 48 reservoirs")
    return parents, reservoirs, units


def build_r2_proposal_jobs(contract: object) -> list[dict[str, object]]:
    """Rebuild all 48 R2 jobs directly from authenticated contract rows."""

    parents, reservoirs, units = _validate_contract_rows(contract)
    parent_by_id = {str(row["parent_id"]): row for row in parents}
    unit_by_id = {str(row["unit_id"]): row for row in units}
    if len(parent_by_id) != len(parents) or len(unit_by_id) != len(units):
        raise ValueError("R2 proposal contract contains duplicate parent or unit identities")
    parent_order = [str(row["parent_id"]) for row in parents]
    reservoir_by_pair: dict[tuple[str, int], Mapping[str, object]] = {}
    for reservoir in reservoirs:
        fold = reservoir.get("fold")
        if isinstance(fold, bool) or fold not in (0, 1):
            raise ValueError("R2 proposal reservoir fold must be exactly 0 or 1")
        key = (str(reservoir.get("parent_id")), int(fold))
        if key in reservoir_by_pair:
            raise ValueError("R2 proposal contract contains a duplicate reservoir")
        reservoir_by_pair[key] = reservoir
    expected_pairs = {
        (parent_id, fold) for parent_id in parent_order for fold in (0, 1)
    }
    if set(reservoir_by_pair) != expected_pairs:
        raise ValueError("R2 proposal reservoirs do not cover every parent/fold")

    jobs: list[dict[str, object]] = []
    for parent_id in parent_order:
        parent = parent_by_id[parent_id]
        for fold in (0, 1):
            reservoir = reservoir_by_pair[(parent_id, fold)]
            documents = reservoir.get("documents")
            if (
                reservoir.get("topic_id") != parent.get("topic_id")
                or reservoir.get("document_count") != 10
                or not isinstance(documents, list)
                or len(documents) != 10
            ):
                raise ValueError("R2 proposal reservoir identity or document count differs")
            ordered_units: list[Mapping[str, object]] = []
            seen_unit_ids: set[str] = set()
            for document in documents:
                if not isinstance(document, Mapping) or not isinstance(
                    document.get("unit_ids"), list
                ):
                    raise ValueError("R2 proposal reservoir unit inventory is invalid")
                for raw_unit_id in document["unit_ids"]:
                    unit_id = str(raw_unit_id)
                    if unit_id in seen_unit_ids or unit_id not in unit_by_id:
                        raise ValueError("R2 proposal unit identity is duplicate or unknown")
                    unit = unit_by_id[unit_id]
                    if (
                        unit.get("topic_id") != parent.get("topic_id")
                        or unit.get("parent_id") != parent_id
                        or unit.get("fold") != fold
                        or unit.get("document_id") != document.get("document_id")
                        or unit.get("window_id") != document.get("window_id")
                    ):
                        raise ValueError(
                            "R2 proposal unit escapes its parent/fold reservoir"
                        )
                    seen_unit_ids.add(unit_id)
                    ordered_units.append(unit)
            if not ordered_units:
                raise ValueError("R2 proposal reservoir must contain evidence units")
            messages = render_r2_proposal_messages(parent, reservoir, ordered_units)
            identity = {
                "topic_id": reservoir["topic_id"],
                "parent_id": parent_id,
                "fold": fold,
                "reservoir_id": reservoir["reservoir_id"],
                "messages_sha256": canonical_sha256(messages),
                "schema_sha256": canonical_sha256(R2_PROPOSAL_SCHEMA),
            }
            jobs.append(
                {
                    "schema_version": R2_JOB_SCHEMA_VERSION,
                    "job_id": canonical_sha256(identity),
                    **identity,
                    "parent_manifest_order": parent["manifest_order"],
                    "input_unit_count": len(ordered_units),
                    "input_unit_ids": [unit["unit_id"] for unit in ordered_units],
                    "messages": messages,
                    "primary_max_new_tokens": r1.PRIMARY_MAX_NEW_TOKENS,
                    "retry_max_new_tokens": r1.RETRY_MAX_NEW_TOKENS,
                    "retry_policy": _prompt_contract()["retry_policy"],
                }
            )
    if len(jobs) != r1.PRIMARY_JOB_COUNT or len(
        {job["job_id"] for job in jobs}
    ) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("R2 proposal jobs must contain exactly 48 unique jobs")
    return jobs


def _require_absent_safe_destination(path: Path, *, label: str) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError(f"{label} must be an absolute safe destination")
    if r1._path_present(destination):
        raise FileExistsError(f"create-only destination exists: {destination}")
    try:
        resolved = destination.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"{label} must be an absolute safe destination") from exc
    if str(resolved) != str(destination):
        raise ValueError(f"{label} must be an absolute safe destination")
    try:
        descriptor = r1._open_directory_no_symlinks(destination.parent)
    except OSError as exc:
        raise ValueError(f"{label} must be an absolute safe destination") from exc
    else:
        os.close(descriptor)
    return destination


def _load_pinned_tokenizer_after_auth(*, snapshot_dir: Path = r1.MODEL_SNAPSHOT) -> object:
    """Load only the pinned tokenizer after all source and destination checks."""

    return r1._load_local_tokenizer(snapshot_dir)


def _code_contract() -> dict[str, str]:
    return {
        "adaptive_obligation_v2_contract.py": r1._sha256_file(
            Path(__file__).with_name("adaptive_obligation_v2_contract.py")
        ),
        "adaptive_obligation_v2_propose.py": r1._sha256_file(
            Path(__file__).with_name("adaptive_obligation_v2_propose.py")
        ),
        "adaptive_obligation_v2_propose_r2.py": r1._sha256_file(Path(__file__)),
    }


class _R2PreflightMaterial(dict[str, object]):
    """Receipt-shaped private material that carries bytes only until publication."""

    def __init__(
        self,
        receipt: Mapping[str, object],
        *,
        contents: Mapping[str, bytes],
        output_dir: Path,
    ) -> None:
        super().__init__(receipt)
        self.contents = dict(contents)
        self.output_dir = Path(output_dir)


def _build_authenticated_r2_preflight(
    contract: object,
    *,
    tokenizer: object,
    model_snapshot: Mapping[str, object],
    output_dir: Path,
    ledger_dir: Path,
    proposal_dir: Path,
    tokenizer_contract: Mapping[str, object] | None = None,
    contract_dir: Path | None = None,
    contract_receipt_sha256: str | None = None,
) -> _R2PreflightMaterial:
    """Unit-test seam for already authenticated sources and an injected tokenizer."""

    output = _require_absent_safe_destination(output_dir, label="R2 preflight")
    ledger = _require_absent_safe_destination(ledger_dir, label="R2 ledger")
    proposals = _require_absent_safe_destination(proposal_dir, label="R2 proposal")
    jobs = build_r2_proposal_jobs(contract)
    snapshot = dict(model_snapshot)
    tokenizer_files = _tokenizer_file_inventory(snapshot)
    frozen_tokenizer = dict(
        tokenizer_contract if tokenizer_contract is not None else _tokenizer_contract()
    )

    counted: list[dict[str, object]] = []
    for job in jobs:
        token_ids = tokenizer.apply_chat_template(
            job["messages"], tokenize=True, add_generation_prompt=True
        )
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or not token_ids
        ):
            raise ValueError("tokenizer returned an invalid R2 prompt token sequence")
        counted.append({**job, "prompt_token_count": len(token_ids)})

    source_path: str | None = None
    if contract_dir is not None:
        source_root = Path(contract_dir)
        if not source_root.is_absolute() or str(source_root.resolve()) != str(source_root):
            raise ValueError("authenticated R2 contract path must be absolute and exact")
        source_path = str(source_root)
    if source_path is not None and (
        not isinstance(contract_receipt_sha256, str)
        or len(contract_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in contract_receipt_sha256)
    ):
        raise ValueError("authenticated R2 contract receipt hash is invalid")

    jobs_bytes = b"".join(r1._compact_bytes(job) for job in counted)
    schema_bytes = r1._pretty_bytes(R2_PROPOSAL_SCHEMA)
    prompt = _prompt_contract()
    prompt_bytes = r1._pretty_bytes(prompt)
    counts = [int(job["prompt_token_count"]) for job in counted]
    contract_receipt: Mapping[str, object] = {}
    if isinstance(contract, Mapping) and isinstance(contract.get("receipt"), Mapping):
        contract_receipt = contract["receipt"]  # type: ignore[assignment]
    receipt: dict[str, object] = {
        **_static_receipt_metadata(
            ledger_dir=ledger,
            proposal_dir=proposals,
        ),
        "prompt_token_counts": {
            "count": len(counts),
            "minimum": min(counts),
            "maximum": max(counts),
            "total": sum(counts),
            "by_job": [
                {
                    "job_id": job["job_id"],
                    "prompt_token_count": job["prompt_token_count"],
                }
                for job in counted
            ],
        },
        "model_snapshot": snapshot,
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": frozen_tokenizer,
        "tokenizer_identity_sha256": canonical_sha256(frozen_tokenizer),
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(R2_PROPOSAL_SCHEMA),
        "contract_receipt": {
            "path": source_path,
            "sha256": contract_receipt_sha256,
            "schema_version": contract_receipt.get("schema_version"),
            "status": contract_receipt.get("status"),
        },
        "code_sha256": _code_contract(),
        "artifacts": {
            "jobs.jsonl": {
                "path": "jobs.jsonl",
                "bytes": len(jobs_bytes),
                "rows": len(counted),
                "sha256": r1._sha256(jobs_bytes),
            },
            "schema.json": {
                "path": "schema.json",
                "bytes": len(schema_bytes),
                "rows": 1,
                "sha256": r1._sha256(schema_bytes),
            },
            "prompt.json": {
                "path": "prompt.json",
                "bytes": len(prompt_bytes),
                "rows": 1,
                "sha256": r1._sha256(prompt_bytes),
            },
        },
    }
    contents = {
        "jobs.jsonl": jobs_bytes,
        "schema.json": schema_bytes,
        "prompt.json": prompt_bytes,
        "receipt.json": r1._pretty_bytes(receipt),
    }
    return _R2PreflightMaterial(
        receipt,
        contents=contents,
        output_dir=output,
    )


def _require_output_inventory(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2 preflight root must be a regular directory")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != _OUTPUT_NAMES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("R2 preflight inventory is partial, unexpected, or unsafe")


def publish_r2_proposal_preflight(
    *,
    contract_dir: Path,
    output_dir: Path,
    ledger_dir: Path,
    proposal_dir: Path,
) -> dict[str, object]:
    """Path-only production alias for the authenticated R2 builder/publisher."""

    return build_r2_proposal_preflight(
        contract_dir=contract_dir,
        output_dir=output_dir,
        ledger_dir=ledger_dir,
        proposal_dir=proposal_dir,
    )


def _validate_receipt_metadata(
    receipt: Mapping[str, object],
) -> dict[str, Path]:
    if receipt.get("schema_version") != R2_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("R2 preflight schema version differs")
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("R2 preflight metadata keys differ")
    ledger_value = receipt.get("ledger_dir")
    proposal_value = receipt.get("proposal_dir")
    if not isinstance(ledger_value, str) or not isinstance(proposal_value, str):
        raise ValueError("R2 preflight metadata destinations differ")
    destinations = {
        "ledger_dir": Path(ledger_value),
        "proposal_dir": Path(proposal_value),
    }
    expected = _static_receipt_metadata(
        ledger_dir=destinations["ledger_dir"],
        proposal_dir=destinations["proposal_dir"],
    )
    observed = {name: receipt[name] for name in expected}
    if not _json_exact(observed, expected):
        raise ValueError("R2 preflight metadata values differ")
    return destinations


def verify_r2_proposal_preflight(output_dir: Path) -> dict[str, object]:
    """Rebuild R2 jobs and recount each prompt with the pinned local tokenizer."""

    root = Path(output_dir)
    _require_output_inventory(root)
    receipt = r1._read_json(root / "receipt.json", "R2 proposal preflight receipt")
    frozen_destinations = _validate_receipt_metadata(receipt)

    for name, destination in frozen_destinations.items():
        _require_absent_safe_destination(
            destination, label=f"R2 {name}"
        )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "jobs.jsonl",
        "schema.json",
        "prompt.json",
    }:
        raise ValueError("R2 preflight artifact bindings differ")
    jobs, jobs_bytes = r1._read_jobs(root / "jobs.jsonl")
    schema_bytes = (root / "schema.json").read_bytes()
    prompt_bytes = (root / "prompt.json").read_bytes()
    expected_artifacts = {
        "jobs.jsonl": {
            "path": "jobs.jsonl",
            "bytes": len(jobs_bytes),
            "rows": len(jobs),
            "sha256": r1._sha256(jobs_bytes),
        },
        "schema.json": {
            "path": "schema.json",
            "bytes": len(schema_bytes),
            "rows": 1,
            "sha256": r1._sha256(schema_bytes),
        },
        "prompt.json": {
            "path": "prompt.json",
            "bytes": len(prompt_bytes),
            "rows": 1,
            "sha256": r1._sha256(prompt_bytes),
        },
    }
    if not _json_exact(artifacts, expected_artifacts):
        raise ValueError("R2 preflight artifact bindings differ")
    schema = r1._read_json(root / "schema.json", "R2 proposal schema")
    prompt = r1._read_json(root / "prompt.json", "R2 proposal prompt")
    if (
        not _json_exact(schema, R2_PROPOSAL_SCHEMA)
        or not _json_exact(prompt, _prompt_contract())
        or prompt.get("schema_version") != R2_PROMPT_SCHEMA_VERSION
        or receipt.get("schema_sha256") != canonical_sha256(schema)
        or receipt.get("prompt_sha256") != canonical_sha256(prompt)
    ):
        raise ValueError("R2 proposal schema or prompt contract differs")
    if len(jobs) != r1.PRIMARY_JOB_COUNT or len(
        {job.get("job_id") for job in jobs}
    ) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("R2 jobs must contain exactly 48 unique jobs")

    pairs: set[tuple[str, int]] = set()
    for job in jobs:
        fold = job.get("fold")
        messages = job.get("messages")
        prompt_count = job.get("prompt_token_count")
        if (
            job.get("schema_version") != R2_JOB_SCHEMA_VERSION
            or isinstance(fold, bool)
            or fold not in (0, 1)
            or not isinstance(messages, list)
            or job.get("messages_sha256") != canonical_sha256(messages)
            or job.get("schema_sha256") != canonical_sha256(R2_PROPOSAL_SCHEMA)
            or job.get("primary_max_new_tokens") != r1.PRIMARY_MAX_NEW_TOKENS
            or job.get("retry_max_new_tokens") != r1.RETRY_MAX_NEW_TOKENS
            or type(prompt_count) is not int
            or prompt_count < 1
        ):
            raise ValueError("R2 jobs contain an invalid frozen proposal job")
        identity = {
            "topic_id": job.get("topic_id"),
            "parent_id": job.get("parent_id"),
            "fold": fold,
            "reservoir_id": job.get("reservoir_id"),
            "messages_sha256": job.get("messages_sha256"),
            "schema_sha256": job.get("schema_sha256"),
        }
        if job.get("job_id") != canonical_sha256(identity):
            raise ValueError("R2 job identity differs")
        pairs.add((str(job.get("parent_id")), int(fold)))
    if len(pairs) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("R2 parent/fold identities differ")

    source_binding = receipt.get("contract_receipt")
    if not isinstance(source_binding, Mapping) or set(source_binding) != {
        "path",
        "sha256",
        "schema_version",
        "status",
    }:
        raise ValueError("R2 contract receipt binding is missing")
    source_path = source_binding.get("path")
    source_sha256 = source_binding.get("sha256")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("R2 contract path is missing")
    contract_root = Path(source_path)
    if not contract_root.is_absolute() or str(contract_root.resolve()) != source_path:
        raise ValueError("R2 contract path must be absolute and exact")
    expected_source_binding = {
        "path": source_path,
        "sha256": source_sha256,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "complete",
    }
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
        or source_sha256 != r1._sha256_file(contract_root / "receipt.json")
        or not _json_exact(source_binding, expected_source_binding)
    ):
        raise ValueError("R2 contract receipt hash or identity differs")
    contract = _load_verified_contract(contract_root)
    expected_jobs = build_r2_proposal_jobs(contract)
    for observed, expected in zip(jobs, expected_jobs, strict=True):
        without_count = {
            key: value
            for key, value in observed.items()
            if key != "prompt_token_count"
        }
        if not _json_exact(without_count, expected):
            raise ValueError("R2 jobs differ from authenticated contract reconstruction")

    observed_snapshot = _snapshot_inventory()
    if not _json_exact(
        receipt.get("model_snapshot"), observed_snapshot
    ) or not _json_exact(
        receipt.get("tokenizer_files"), _tokenizer_file_inventory(observed_snapshot)
    ):
        raise ValueError("R2 model snapshot or tokenizer inventory differs")
    observed_tokenizer_contract = _tokenizer_contract()
    if (
        not _json_exact(receipt.get("tokenizer_contract"), observed_tokenizer_contract)
        or receipt.get("tokenizer_identity_sha256")
        != canonical_sha256(observed_tokenizer_contract)
    ):
        raise ValueError("R2 pinned tokenizer identity differs")
    if not _json_exact(receipt.get("code_sha256"), _code_contract()):
        raise ValueError("R2 frozen builder code hashes differ")

    prompt_counts = receipt.get("prompt_token_counts")
    counts = [int(job["prompt_token_count"]) for job in jobs]
    expected_prompt_counts = {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "total": sum(counts),
        "by_job": [
            {
                "job_id": job["job_id"],
                "prompt_token_count": job["prompt_token_count"],
            }
            for job in jobs
        ],
    }
    if not isinstance(prompt_counts, Mapping) or not _json_exact(
        prompt_counts, expected_prompt_counts
    ):
        raise ValueError("R2 prompt token count receipt differs")
    tokenizer = _load_pinned_tokenizer_after_auth(snapshot_dir=r1.MODEL_SNAPSHOT)
    recomputed: list[int] = []
    for job in jobs:
        token_ids = tokenizer.apply_chat_template(
            job["messages"], tokenize=True, add_generation_prompt=True
        )
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or not token_ids
        ):
            raise ValueError("pinned tokenizer returned an invalid R2 prompt sequence")
        recomputed.append(len(token_ids))
    if recomputed != counts:
        raise ValueError("R2 prompt counts differ from the pinned tokenizer")
    return receipt


def _capture_r2_inference_approval(path: Path) -> object:
    """Capture exact canonical R2 approval bytes before any other access."""

    try:
        source = r1._capture_regular_file_no_symlinks(Path(path))
        approval = json.loads(source)
        canonical = r1._pretty_bytes(approval) if isinstance(approval, dict) else None
    except (
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise PermissionError("R2 proposal approval required") from exc
    required_names = {
        "schema_version",
        "stage",
        "preflight_sha256",
        "model",
        "model_revision",
        "primary_call_count",
        "retry_call_ceiling",
        "ledger_dir",
        "approved",
    }
    preflight_sha256 = (
        approval.get("preflight_sha256") if isinstance(approval, Mapping) else None
    )
    ledger_value = approval.get("ledger_dir") if isinstance(approval, Mapping) else None
    if (
        not isinstance(approval, dict)
        or source != canonical
        or set(approval) != required_names
        or approval.get("schema_version")
        != "adaptive-obligation-v2-proposal-approval-r2"
        or approval.get("stage") != "proposal_r2"
        or approval.get("model") != r1.MODEL_ID
        or approval.get("model_revision") != r1.MODEL_REVISION
        or type(approval.get("primary_call_count")) is not int
        or approval.get("primary_call_count") != r1.PRIMARY_JOB_COUNT
        or type(approval.get("retry_call_ceiling")) is not int
        or approval.get("retry_call_ceiling") != r1.PRIMARY_JOB_COUNT
        or approval.get("approved") is not True
        or not isinstance(preflight_sha256, str)
        or len(preflight_sha256) != 64
        or any(char not in "0123456789abcdef" for char in preflight_sha256)
        or not isinstance(ledger_value, str)
        or not Path(ledger_value).is_absolute()
    ):
        raise PermissionError("R2 proposal approval required")
    return r1._CapturedApproval(
        value=approval,
        source=source,
        sha256=r1._sha256(source),
    )


def _capture_and_verify_r2_preflight(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> object:
    """Capture once and reconstruct all R2 jobs in an isolated verifier copy."""

    return r1._capture_and_verify_preflight(
        Path(path),
        expected_receipt_sha256=expected_receipt_sha256,
        verifier=verify_r2_proposal_preflight,
    )


def _validate_static_r2_jobs(
    jobs: list[dict[str, object]], receipt: Mapping[str, object]
) -> None:
    job_ids = [job.get("job_id") for job in jobs]
    if (
        len(jobs) != r1.PRIMARY_JOB_COUNT
        or any(
            not isinstance(job_id, str)
            or len(job_id) != 64
            or any(char not in "0123456789abcdef" for char in job_id)
            for job_id in job_ids
        )
        or len(set(job_ids)) != r1.PRIMARY_JOB_COUNT
    ):
        raise ValueError("captured R2 proposal jobs must be exactly 48 unique rows")
    pairs: set[tuple[str, int]] = set()
    for job in jobs:
        fold = job.get("fold")
        messages = job.get("messages")
        unit_ids = job.get("input_unit_ids")
        prompt_token_count = job.get("prompt_token_count")
        if (
            job.get("schema_version") != R2_JOB_SCHEMA_VERSION
            or isinstance(fold, bool)
            or fold not in (0, 1)
            or not isinstance(messages, list)
            or job.get("messages_sha256") != canonical_sha256(messages)
            or job.get("schema_sha256") != canonical_sha256(R2_PROPOSAL_SCHEMA)
            or job.get("primary_max_new_tokens") != r1.PRIMARY_MAX_NEW_TOKENS
            or job.get("retry_max_new_tokens") != r1.RETRY_MAX_NEW_TOKENS
            or not _json_exact(
                job.get("retry_policy"), _prompt_contract()["retry_policy"]
            )
            or type(prompt_token_count) is not int
            or prompt_token_count < 1
            or not isinstance(unit_ids, list)
            or job.get("input_unit_count") != len(unit_ids)
            or any(
                not isinstance(unit_id, str)
                or len(unit_id) != 64
                or any(char not in "0123456789abcdef" for char in unit_id)
                for unit_id in unit_ids
            )
            or len(set(unit_ids)) != len(unit_ids)
        ):
            raise ValueError("captured R2 proposal job binding differs")
        identity = {
            "topic_id": job.get("topic_id"),
            "parent_id": job.get("parent_id"),
            "fold": fold,
            "reservoir_id": job.get("reservoir_id"),
            "messages_sha256": job.get("messages_sha256"),
            "schema_sha256": job.get("schema_sha256"),
        }
        if job.get("job_id") != canonical_sha256(identity):
            raise ValueError("captured R2 proposal job identity differs")
        pairs.add((str(job.get("parent_id")), int(fold)))
    if len(pairs) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("captured R2 proposal parent/fold inventory differs")
    counts = [int(job["prompt_token_count"]) for job in jobs]
    expected_counts = {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "total": sum(counts),
        "by_job": [
            {
                "job_id": job["job_id"],
                "prompt_token_count": job["prompt_token_count"],
            }
            for job in jobs
        ],
    }
    if not _json_exact(receipt.get("prompt_token_counts"), expected_counts):
        raise ValueError("captured R2 prompt token metadata differs")


def _capture_static_r2_preflight(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> object:
    """Authenticate approved R2 bytes without requiring frozen outputs absent."""

    contents = r1._capture_preflight_files(Path(path))
    receipt_source = contents["receipt.json"]
    receipt_sha256 = r1._sha256(receipt_source)
    if receipt_sha256 != expected_receipt_sha256:
        raise ValueError("R2 proposal preflight receipt hash differs")
    try:
        receipt = json.loads(receipt_source)
        schema = json.loads(contents["schema.json"])
        prompt = json.loads(contents["prompt.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("captured R2 proposal preflight JSON is invalid") from exc
    if (
        not isinstance(receipt, dict)
        or receipt_source != r1._pretty_bytes(receipt)
        or not isinstance(schema, dict)
        or contents["schema.json"] != r1._pretty_bytes(schema)
        or not isinstance(prompt, dict)
        or contents["prompt.json"] != r1._pretty_bytes(prompt)
        or not _json_exact(schema, R2_PROPOSAL_SCHEMA)
        or not _json_exact(prompt, _prompt_contract())
    ):
        raise ValueError("captured R2 proposal preflight is not canonical")
    destinations = _validate_receipt_metadata(receipt)
    for destination in destinations.values():
        try:
            exact = str(destination.resolve(strict=False)) == str(destination)
        except (OSError, RuntimeError):
            exact = False
        if not destination.is_absolute() or not exact:
            raise ValueError("captured R2 proposal destinations differ")
    if (
        receipt.get("schema_sha256") != canonical_sha256(schema)
        or receipt.get("prompt_sha256") != canonical_sha256(prompt)
        or not isinstance(receipt.get("model_snapshot"), Mapping)
    ):
        raise ValueError("captured R2 proposal preflight identity differs")
    jobs, jobs_source = r1._read_jobs_bytes(contents["jobs.jsonl"])
    artifacts = receipt.get("artifacts")
    expected_artifacts = {
        "jobs.jsonl": {
            "path": "jobs.jsonl",
            "bytes": len(jobs_source),
            "rows": len(jobs),
            "sha256": r1._sha256(jobs_source),
        },
        "schema.json": {
            "path": "schema.json",
            "bytes": len(contents["schema.json"]),
            "rows": 1,
            "sha256": r1._sha256(contents["schema.json"]),
        },
        "prompt.json": {
            "path": "prompt.json",
            "bytes": len(contents["prompt.json"]),
            "rows": 1,
            "sha256": r1._sha256(contents["prompt.json"]),
        },
    }
    if not _json_exact(artifacts, expected_artifacts):
        raise ValueError("captured R2 proposal artifact bindings differ")
    _validate_static_r2_jobs(jobs, receipt)
    source_binding = receipt.get("contract_receipt")
    source_path = (
        source_binding.get("path") if isinstance(source_binding, Mapping) else None
    )
    try:
        source_path_exact = (
            isinstance(source_path, str)
            and bool(source_path)
            and Path(source_path).is_absolute()
            and str(Path(source_path).resolve(strict=False)) == source_path
        )
    except (OSError, RuntimeError):
        source_path_exact = False
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "sha256", "schema_version", "status"}
        or not source_path_exact
        or not isinstance(source_binding.get("sha256"), str)
        or len(str(source_binding["sha256"])) != 64
        or any(
            char not in "0123456789abcdef"
            for char in str(source_binding["sha256"])
        )
        or source_binding.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or source_binding.get("status") != "complete"
    ):
        raise ValueError("captured R2 source contract binding differs")
    return r1._CapturedPreflight(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        jobs=jobs,
        contents=contents,
    )


def _capture_and_replay_authenticated_r2_source_chain(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> object:
    """Authenticate captured R2 bytes and every frozen upstream dependency.

    Unlike the inference-time verifier, this replay deliberately permits the
    frozen ledger and proposal leaves to exist. It therefore starts from the
    create-state-independent byte capture and then re-derives the complete
    source chain without trusting receipt-only bindings.
    """

    captured = _capture_static_r2_preflight(
        Path(path),
        expected_receipt_sha256=expected_receipt_sha256,
    )
    receipt = captured.receipt
    jobs = captured.jobs

    source_binding = receipt.get("contract_receipt")
    source_path = (
        source_binding.get("path") if isinstance(source_binding, Mapping) else None
    )
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("authenticated R2 contract path is missing")
    contract_root = Path(source_path)
    try:
        exact_source_path = (
            contract_root.is_absolute()
            and str(contract_root.resolve(strict=False)) == source_path
        )
    except (OSError, RuntimeError):
        exact_source_path = False
    if not exact_source_path:
        raise ValueError("authenticated R2 contract path differs")

    try:
        contract_receipt_source = r1._capture_regular_file_no_symlinks(
            contract_root / "receipt.json"
        )
        contract_receipt = json.loads(contract_receipt_source)
    except (
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("authenticated R2 contract receipt is unavailable") from exc
    expected_source_binding = {
        "path": source_path,
        "sha256": r1._sha256(contract_receipt_source),
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "complete",
    }
    if (
        not isinstance(contract_receipt, dict)
        or contract_receipt_source != r1._pretty_bytes(contract_receipt)
        or contract_receipt.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract_receipt.get("status") != "complete"
        or not _json_exact(source_binding, expected_source_binding)
    ):
        raise ValueError("authenticated R2 contract receipt identity differs")

    contract = _load_verified_contract(contract_root)
    if (
        not isinstance(contract, Mapping)
        or not _json_exact(contract.get("receipt"), contract_receipt)
    ):
        raise ValueError("authenticated R2 contract content differs from its receipt")
    expected_jobs = build_r2_proposal_jobs(contract)
    if len(expected_jobs) != len(jobs):
        raise ValueError("authenticated R2 job reconstruction count differs")
    for observed, expected in zip(jobs, expected_jobs, strict=True):
        without_count = {
            key: value
            for key, value in observed.items()
            if key != "prompt_token_count"
        }
        if not _json_exact(without_count, expected):
            raise ValueError(
                "authenticated R2 jobs differ from contract reconstruction"
            )

    observed_snapshot = _snapshot_inventory()
    if (
        not _json_exact(receipt.get("model_snapshot"), observed_snapshot)
        or not _json_exact(
            receipt.get("tokenizer_files"),
            _tokenizer_file_inventory(observed_snapshot),
        )
    ):
        raise ValueError("authenticated R2 model or tokenizer files differ")
    observed_tokenizer_contract = _tokenizer_contract()
    if (
        not _json_exact(
            receipt.get("tokenizer_contract"), observed_tokenizer_contract
        )
        or receipt.get("tokenizer_identity_sha256")
        != canonical_sha256(observed_tokenizer_contract)
    ):
        raise ValueError("authenticated R2 tokenizer identity differs")
    if not _json_exact(receipt.get("code_sha256"), _code_contract()):
        raise ValueError("authenticated R2 frozen code identities differ")

    counts = [int(job["prompt_token_count"]) for job in jobs]
    prompt_counts = receipt.get("prompt_token_counts")
    expected_prompt_counts = {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "total": sum(counts),
        "by_job": [
            {
                "job_id": job["job_id"],
                "prompt_token_count": job["prompt_token_count"],
            }
            for job in jobs
        ],
    }
    if not isinstance(prompt_counts, Mapping) or not _json_exact(
        prompt_counts, expected_prompt_counts
    ):
        raise ValueError("authenticated R2 prompt token metadata differs")
    tokenizer = _load_pinned_tokenizer_after_auth(snapshot_dir=r1.MODEL_SNAPSHOT)
    recomputed: list[int] = []
    for job in jobs:
        token_ids = tokenizer.apply_chat_template(
            job["messages"], tokenize=True, add_generation_prompt=True
        )
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or not token_ids
        ):
            raise ValueError(
                "pinned tokenizer returned an invalid authenticated R2 sequence"
            )
        recomputed.append(len(token_ids))
    if recomputed != counts:
        raise ValueError("authenticated R2 prompt token recount differs")
    return captured


def _load_r2_model_after_approval(
    *,
    approval: object,
    preflight: Mapping[str, object],
    ledger_dir: Path,
) -> object:
    from .adaptive_obligation_v2_local_model_r2 import R2LocalJsonModel

    return R2LocalJsonModel(
        approval=approval,
        preflight=preflight,
        ledger_dir=ledger_dir,
    )


def run_r2_job_with_retry(
    job: Mapping[str, object],
    *,
    generate: Callable[[int], object],
    ledger: object,
) -> dict[str, object]:
    """Run the unchanged proposal classifier under the frozen R2 schema."""

    if not _json_exact(R2_PROPOSAL_SCHEMA, r1.PROPOSAL_SCHEMA):
        raise RuntimeError("R2 accepted proposal schema differs from R1")
    try:
        return r1.run_job_with_retry(
            job,
            generate=generate,
            ledger=ledger,
        )
    except ValueError as exc:
        if "parse:" in str(exc):
            raise ValueError(f"R2 proposal JSON {exc}") from exc
        raise


def _execute_authenticated_r2_jobs(
    preflight: Mapping[str, object],
    *,
    ledger: object,
    model: object,
) -> dict[str, object]:
    jobs = preflight.get("jobs")
    job_ids = (
        [job.get("job_id") for job in jobs if isinstance(job, Mapping)]
        if isinstance(jobs, list)
        else []
    )
    if (
        not isinstance(jobs, list)
        or len(jobs) != r1.PRIMARY_JOB_COUNT
        or len(job_ids) != r1.PRIMARY_JOB_COUNT
        or any(not isinstance(job_id, str) for job_id in job_ids)
        or len(set(job_ids)) != r1.PRIMARY_JOB_COUNT
    ):
        raise ValueError("R2 proposal execution requires exactly 48 frozen jobs")
    generate_method = getattr(model, "generate", None)
    if not callable(generate_method):
        raise TypeError("R2 proposal model must expose generate")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("R2 proposal execution job differs")

        def generate(
            ceiling: int,
            *,
            frozen_job: Mapping[str, object] = job,
        ) -> object:
            return generate_method(
                frozen_job["messages"],
                R2_PROPOSAL_SCHEMA,
                max_new_tokens=ceiling,
            )

        run_r2_job_with_retry(job, generate=generate, ledger=ledger)
    completion = ledger.seal_completion()
    sealed = ledger.read_sealed_results()
    results = sealed.get("results")
    if not isinstance(results, list) or len(results) != r1.PRIMARY_JOB_COUNT:
        raise ValueError("R2 proposal sealed result inventory differs")
    return {
        "status": "complete",
        "job_count": len(results),
        "results": results,
        "event_count": len(ledger.read_events()),
        "completion": completion,
    }


def execute_r2_proposals(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
) -> dict[str, object]:
    """Execute R2 only after its exact path-bound approval authenticates."""

    captured_approval = _capture_r2_inference_approval(Path(approval_path))
    captured_preflight = _capture_and_verify_r2_preflight(
        Path(preflight_dir),
        expected_receipt_sha256=str(
            captured_approval.value["preflight_sha256"]
        ),
    )
    preflight = {
        **captured_preflight.receipt,
        "receipt_sha256": captured_preflight.receipt_sha256,
        "jobs": captured_preflight.jobs,
    }
    from .adaptive_obligation_v2_local_model_r2 import (
        verify_r2_inference_approval,
    )

    destination = Path(ledger_dir)
    verify_r2_inference_approval(
        captured_approval.value,
        preflight,
        ledger_dir=destination,
    )
    anchor = r1._build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=captured_approval.sha256,
    )
    ledger = r1.AppendOnlyAttemptLedger(
        destination,
        expected_anchor=anchor,
        create_only=True,
    )
    model = _load_r2_model_after_approval(
        approval=captured_approval.value,
        preflight=preflight,
        ledger_dir=destination,
    )
    return _execute_authenticated_r2_jobs(
        preflight,
        ledger=ledger,
        model=model,
    )


def _r2_proposal_inventory_material(
    captured_preflight: object,
    *,
    approval_sha256: str,
    sealed: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, bytes]]:
    rows, receipt, contents = r1._proposal_inventory_material(
        captured_preflight,
        approval_sha256=approval_sha256,
        sealed=sealed,
    )
    receipt = {
        **receipt,
        "schema_version": R2_PROPOSAL_RECEIPT_SCHEMA_VERSION,
    }
    contents = {
        "proposals.jsonl": contents["proposals.jsonl"],
        "receipt.json": r1._pretty_bytes(receipt),
    }
    return rows, receipt, contents


def _require_frozen_proposal_destination(
    output_dir: Path,
    receipt: Mapping[str, object],
) -> Path:
    destination = Path(output_dir)
    frozen = receipt.get("proposal_dir")
    try:
        exact = str(destination.resolve(strict=False)) == str(destination)
    except (OSError, RuntimeError):
        exact = False
    if (
        not isinstance(frozen, str)
        or not destination.is_absolute()
        or str(destination) != frozen
        or not exact
    ):
        raise ValueError("R2 finalization requires the frozen proposal destination")
    try:
        descriptor = r1._open_directory_no_symlinks(destination.parent)
    except OSError as exc:
        raise ValueError(
            "R2 finalization requires the frozen proposal destination"
        ) from exc
    else:
        os.close(descriptor)
    return destination


def finalize_r2_proposal_inventory(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish R2 proposals only from the full sealed approved ledger."""

    captured_approval = _capture_r2_inference_approval(Path(approval_path))
    captured_preflight = _capture_and_replay_authenticated_r2_source_chain(
        Path(preflight_dir),
        expected_receipt_sha256=str(
            captured_approval.value["preflight_sha256"]
        ),
    )
    preflight = {
        **captured_preflight.receipt,
        "receipt_sha256": captured_preflight.receipt_sha256,
        "jobs": captured_preflight.jobs,
    }
    from .adaptive_obligation_v2_local_model_r2 import (
        verify_r2_inference_approval,
    )

    ledger_destination = Path(ledger_dir)
    verify_r2_inference_approval(
        captured_approval.value,
        preflight,
        ledger_dir=ledger_destination,
    )
    proposal_destination = _require_frozen_proposal_destination(
        Path(output_dir), captured_preflight.receipt
    )
    anchor = r1._build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=captured_approval.sha256,
    )
    ledger = r1.AppendOnlyAttemptLedger(
        ledger_destination,
        expected_anchor=anchor,
        create_only=False,
    )
    sealed = ledger.read_sealed_results()
    _rows, receipt, contents = _r2_proposal_inventory_material(
        captured_preflight,
        approval_sha256=captured_approval.sha256,
        sealed=sealed,
    )
    r1._publish_proposal_inventory(proposal_destination, contents)
    return receipt


def load_authenticated_r2_proposal_inventory(
    *,
    output_dir: Path,
    preflight_dir: Path,
    ledger_dir: Path,
) -> dict[str, object]:
    """Replay the complete R2 chain before returning durable proposals."""

    contents = r1._capture_proposal_inventory_files(Path(output_dir))
    try:
        receipt = json.loads(contents["receipt.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("R2 proposal inventory receipt is invalid JSON") from exc
    approval_sha256 = (
        receipt.get("approval_sha256") if isinstance(receipt, Mapping) else None
    )
    preflight_sha256 = (
        receipt.get("proposal_preflight_receipt_sha256")
        if isinstance(receipt, Mapping)
        else None
    )
    if (
        not isinstance(receipt, dict)
        or contents["receipt.json"] != r1._pretty_bytes(receipt)
        or receipt.get("schema_version")
        != R2_PROPOSAL_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or not isinstance(approval_sha256, str)
        or len(approval_sha256) != 64
        or any(char not in "0123456789abcdef" for char in approval_sha256)
        or not isinstance(preflight_sha256, str)
        or len(preflight_sha256) != 64
        or any(char not in "0123456789abcdef" for char in preflight_sha256)
    ):
        raise ValueError("R2 proposal inventory receipt differs")
    captured_preflight = _capture_and_replay_authenticated_r2_source_chain(
        Path(preflight_dir),
        expected_receipt_sha256=preflight_sha256,
    )
    _require_frozen_proposal_destination(
        Path(output_dir), captured_preflight.receipt
    )
    if captured_preflight.receipt.get("ledger_dir") != str(Path(ledger_dir)):
        raise ValueError("R2 proposal inventory frozen ledger destination differs")
    anchor = r1._build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=approval_sha256,
    )
    ledger = r1.AppendOnlyAttemptLedger(
        Path(ledger_dir),
        expected_anchor=anchor,
        create_only=False,
    )
    sealed = ledger.read_sealed_results()
    expected_rows, expected_receipt, expected_contents = (
        _r2_proposal_inventory_material(
            captured_preflight,
            approval_sha256=approval_sha256,
            sealed=sealed,
        )
    )
    observed_rows = r1._read_proposal_rows(contents["proposals.jsonl"])
    if (
        contents != expected_contents
        or receipt != expected_receipt
        or observed_rows != expected_rows
    ):
        raise ValueError("R2 proposal inventory differs from sealed ledger replay")
    return {
        "proposals": observed_rows,
        "receipt": receipt,
        "receipt_sha256": r1._sha256(contents["receipt.json"]),
        "proposal_preflight": dict(captured_preflight.receipt),
    }


def build_r2_proposal_preflight(
    *,
    contract_dir: Path,
    output_dir: Path,
    ledger_dir: Path,
    proposal_dir: Path,
) -> dict[str, object]:
    """Authenticate production sources, load only the tokenizer, and publish R2."""

    supplied_contract_root = Path(contract_dir)
    contract = _load_verified_contract(supplied_contract_root)
    r1._reject_protected_contract(contract)
    snapshot = _snapshot_inventory()
    output = _require_absent_safe_destination(output_dir, label="R2 preflight")
    ledger = _require_absent_safe_destination(ledger_dir, label="R2 ledger")
    proposals = _require_absent_safe_destination(proposal_dir, label="R2 proposal")
    frozen_tokenizer_contract = _tokenizer_contract()
    contract_root = supplied_contract_root.resolve()
    source_receipt_sha256 = r1._sha256_file(contract_root / "receipt.json")
    tokenizer = _load_pinned_tokenizer_after_auth(snapshot_dir=r1.MODEL_SNAPSHOT)
    material = _build_authenticated_r2_preflight(
        contract,
        tokenizer=tokenizer,
        model_snapshot=snapshot,
        tokenizer_contract=frozen_tokenizer_contract,
        contract_dir=contract_root,
        contract_receipt_sha256=source_receipt_sha256,
        output_dir=output,
        ledger_dir=ledger,
        proposal_dir=proposals,
    )
    publication_capability = object()

    def publish_authenticated_material(*, capability: object) -> dict[str, object]:
        if capability is not publication_capability:  # pragma: no cover - closure only
            raise PermissionError(
                "R2 publication requires production-authenticated material"
            )
        destination = _require_absent_safe_destination(
            output, label="R2 preflight"
        )
        for name, frozen in (("ledger_dir", ledger), ("proposal_dir", proposals)):
            _require_absent_safe_destination(frozen, label=f"R2 {name}")
        if set(material.contents) != _OUTPUT_NAMES or material.contents[
            "receipt.json"
        ] != r1._pretty_bytes(dict(material)):
            raise ValueError("R2 publication material differs from its receipt")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        published = False
        try:
            for artifact_name in (
                "jobs.jsonl",
                "schema.json",
                "prompt.json",
                "receipt.json",
            ):
                r1._write_fsynced(
                    staging / artifact_name,
                    material.contents[artifact_name],
                )
            _require_output_inventory(staging)
            r1._fsync_directory(staging)
            verified = verify_r2_proposal_preflight(staging)
            if not _json_exact(verified, dict(material)):
                raise ValueError("staged R2 verification receipt differs")
            r1._rename_noreplace(staging, destination)
            published = True
            r1._fsync_directory(destination.parent)
            return verified
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging)

    return publish_authenticated_material(capability=publication_capability)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    build = actions.add_parser("build-preflight")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ledger-destination", type=Path, required=True)
    build.add_argument("--proposal-destination", type=Path, required=True)
    verify = actions.add_parser("verify-preflight")
    verify.add_argument("--output", type=Path, required=True)
    execute = actions.add_parser("execute")
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--approval", type=Path, required=True)
    execute.add_argument("--ledger", type=Path, required=True)
    finalize = actions.add_parser("finalize")
    finalize.add_argument("--preflight", type=Path, required=True)
    finalize.add_argument("--approval", type=Path, required=True)
    finalize.add_argument("--ledger", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-preflight":
        receipt = build_r2_proposal_preflight(
            contract_dir=args.contract,
            output_dir=args.output.absolute(),
            ledger_dir=args.ledger_destination,
            proposal_dir=args.proposal_destination,
        )
    elif args.command == "verify-preflight":
        receipt = verify_r2_proposal_preflight(args.output)
    elif args.command == "execute":
        result = execute_r2_proposals(
            preflight_dir=args.preflight,
            approval_path=args.approval,
            ledger_dir=args.ledger,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "jobs": result["job_count"],
                    "events": result["event_count"],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    else:
        receipt = finalize_r2_proposal_inventory(
            preflight_dir=args.preflight,
            approval_path=args.approval,
            ledger_dir=args.ledger,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "proposals": receipt["proposal_count"],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    token_counts = receipt["prompt_token_counts"]
    print(
        json.dumps(
            {
                "status": (
                    "verified" if args.command == "verify-preflight" else "complete"
                ),
                "jobs": receipt["job_count"],
                "primary_calls": receipt["primary_call_count"],
                "retry_ceiling": receipt["retry_call_ceiling"],
                "worst_case_calls": receipt["worst_case_call_ceiling"],
                "prompt_tokens": {
                    "minimum": token_counts["minimum"],
                    "maximum": token_counts["maximum"],
                    "total": token_counts["total"],
                },
                "tokenizer_loads": receipt["tokenizer_load_count"],
                "model_loads": receipt["model_load_count"],
                "inference_calls": receipt["inference_count"],
                "network_calls": receipt["network_call_count"],
                "retrieval_calls": receipt["retrieval_call_count"],
                "qrels_opened": receipt["qrels_opened"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

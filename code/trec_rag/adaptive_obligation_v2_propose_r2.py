"""Freeze the authenticated, tokenizer-only adaptive-obligation R2 preflight."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
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
        publication_authorized: bool,
    ) -> None:
        super().__init__(receipt)
        self.contents = dict(contents)
        self.output_dir = Path(output_dir)
        self.publication_authorized = publication_authorized


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
    _publication_authorized: bool = False,
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
        "schema_version": R2_PREFLIGHT_SCHEMA_VERSION,
        "prompt_revision": R2_PROMPT_REVISION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": len(counted),
        "primary_call_count": r1.PRIMARY_JOB_COUNT,
        "retry_call_ceiling": r1.PRIMARY_JOB_COUNT,
        "worst_case_call_ceiling": r1.PRIMARY_JOB_COUNT * 2,
        "primary_max_new_tokens": r1.PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": r1.RETRY_MAX_NEW_TOKENS,
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
        "tokenizer_load_count": 1,
        "model_load_count": 0,
        "inference_count": 0,
        "model": r1.MODEL_ID,
        "model_revision": r1.MODEL_REVISION,
        "model_snapshot": snapshot,
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": frozen_tokenizer,
        "tokenizer_identity_sha256": canonical_sha256(frozen_tokenizer),
        "tokenizer_loading": {
            "backend": "tokenizers",
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_required": False,
            "transformers_required": False,
        },
        "model_construction_allowed": False,
        "generation_allowed": False,
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(R2_PROPOSAL_SCHEMA),
        "contract_receipt": {
            "path": source_path,
            "sha256": contract_receipt_sha256,
            "schema_version": contract_receipt.get("schema_version"),
            "status": contract_receipt.get("status"),
        },
        "code_sha256": _code_contract(),
        "ledger_dir": str(ledger),
        "proposal_dir": str(proposals),
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
            "ledger_dir": str(ledger),
            "proposal_dir": str(proposals),
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
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
        publication_authorized=_publication_authorized,
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
    preflight: Mapping[str, object], *, output_dir: Path
) -> dict[str, object]:
    """Verify a sibling staging tree before atomically publishing it no-replace."""

    if not isinstance(preflight, _R2PreflightMaterial) or not (
        preflight.publication_authorized
    ):
        raise PermissionError("R2 publication requires production-authenticated material")
    destination = _require_absent_safe_destination(
        output_dir, label="R2 preflight"
    )
    if destination != preflight.output_dir:
        raise ValueError("R2 publication output differs from the authenticated destination")
    for name in ("ledger_dir", "proposal_dir"):
        value = preflight.get(name)
        if not isinstance(value, str):
            raise ValueError("R2 frozen destination is missing")
        _require_absent_safe_destination(Path(value), label=f"R2 {name}")
    if set(preflight.contents) != _OUTPUT_NAMES or preflight.contents[
        "receipt.json"
    ] != r1._pretty_bytes(dict(preflight)):
        raise ValueError("R2 publication material differs from its receipt")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        for name in ("jobs.jsonl", "schema.json", "prompt.json", "receipt.json"):
            r1._write_fsynced(staging / name, preflight.contents[name])
        _require_output_inventory(staging)
        r1._fsync_directory(staging)
        verified = verify_r2_proposal_preflight(staging)
        if verified != dict(preflight):
            raise ValueError("staged R2 verification receipt differs")
        r1._rename_noreplace(staging, destination)
        published = True
        r1._fsync_directory(destination.parent)
        return verified
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _validate_receipt_header(receipt: Mapping[str, object]) -> None:
    if receipt.get("schema_version") != R2_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("R2 preflight schema version differs")
    if (
        receipt.get("prompt_revision") != R2_PROMPT_REVISION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("job_count") != r1.PRIMARY_JOB_COUNT
        or receipt.get("primary_call_count") != r1.PRIMARY_JOB_COUNT
        or receipt.get("retry_call_ceiling") != r1.PRIMARY_JOB_COUNT
        or receipt.get("worst_case_call_ceiling") != r1.PRIMARY_JOB_COUNT * 2
        or receipt.get("primary_max_new_tokens") != r1.PRIMARY_MAX_NEW_TOKENS
        or receipt.get("retry_max_new_tokens") != r1.RETRY_MAX_NEW_TOKENS
        or receipt.get("tokenizer_load_count") != 1
        or any(receipt.get(name) != 0 for name in r1._ZERO_COUNTERS)
        or receipt.get("qrels_opened") is not False
        or receipt.get("external_cost_usd") != 0.0
        or receipt.get("model") != r1.MODEL_ID
        or receipt.get("model_revision") != r1.MODEL_REVISION
        or receipt.get("model_construction_allowed") is not False
        or receipt.get("generation_allowed") is not False
    ):
        raise ValueError("R2 preflight counts, model, or safety receipt differs")


def verify_r2_proposal_preflight(output_dir: Path) -> dict[str, object]:
    """Rebuild R2 jobs and recount each prompt with the pinned local tokenizer."""

    root = Path(output_dir)
    _require_output_inventory(root)
    receipt = r1._read_json(root / "receipt.json", "R2 proposal preflight receipt")
    _validate_receipt_header(receipt)

    frozen_destinations: dict[str, Path] = {}
    for name in ("ledger_dir", "proposal_dir"):
        value = receipt.get(name)
        if not isinstance(value, str):
            raise ValueError("R2 frozen destination is missing")
        frozen_destinations[name] = _require_absent_safe_destination(
            Path(value), label=f"R2 {name}"
        )
    planned = receipt.get("planned_storage")
    if not isinstance(planned, Mapping) or any(
        planned.get(name) != str(path) for name, path in frozen_destinations.items()
    ):
        raise ValueError("R2 planned destinations differ")

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
    r1._artifact_matches("jobs.jsonl", jobs_bytes, len(jobs), artifacts)
    r1._artifact_matches("schema.json", schema_bytes, 1, artifacts)
    r1._artifact_matches("prompt.json", prompt_bytes, 1, artifacts)
    schema = r1._read_json(root / "schema.json", "R2 proposal schema")
    prompt = r1._read_json(root / "prompt.json", "R2 proposal prompt")
    if (
        schema != R2_PROPOSAL_SCHEMA
        or prompt != _prompt_contract()
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
    if not isinstance(source_binding, Mapping):
        raise ValueError("R2 contract receipt binding is missing")
    source_path = source_binding.get("path")
    source_sha256 = source_binding.get("sha256")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("R2 contract path is missing")
    contract_root = Path(source_path)
    if not contract_root.is_absolute() or str(contract_root.resolve()) != source_path:
        raise ValueError("R2 contract path must be absolute and exact")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
        or source_sha256 != r1._sha256_file(contract_root / "receipt.json")
        or source_binding.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or source_binding.get("status") != "complete"
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
        if without_count != expected:
            raise ValueError("R2 jobs differ from authenticated contract reconstruction")

    observed_snapshot = _snapshot_inventory()
    if receipt.get("model_snapshot") != observed_snapshot or receipt.get(
        "tokenizer_files"
    ) != _tokenizer_file_inventory(observed_snapshot):
        raise ValueError("R2 model snapshot or tokenizer inventory differs")
    observed_tokenizer_contract = _tokenizer_contract()
    if (
        receipt.get("tokenizer_contract") != observed_tokenizer_contract
        or receipt.get("tokenizer_identity_sha256")
        != canonical_sha256(observed_tokenizer_contract)
    ):
        raise ValueError("R2 pinned tokenizer identity differs")
    if receipt.get("code_sha256") != _code_contract():
        raise ValueError("R2 frozen builder code hashes differ")

    prompt_counts = receipt.get("prompt_token_counts")
    counts = [int(job["prompt_token_count"]) for job in jobs]
    if (
        not isinstance(prompt_counts, Mapping)
        or prompt_counts.get("count") != len(counts)
        or prompt_counts.get("minimum") != min(counts)
        or prompt_counts.get("maximum") != max(counts)
        or prompt_counts.get("total") != sum(counts)
        or prompt_counts.get("by_job")
        != [
            {
                "job_id": job["job_id"],
                "prompt_token_count": job["prompt_token_count"],
            }
            for job in jobs
        ]
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
    tokenizer = _load_pinned_tokenizer_after_auth(snapshot_dir=r1.MODEL_SNAPSHOT)
    contract_root = supplied_contract_root.resolve()
    source_receipt_sha256 = r1._sha256_file(contract_root / "receipt.json")
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
        _publication_authorized=True,
    )
    return publish_r2_proposal_preflight(material, output_dir=output)

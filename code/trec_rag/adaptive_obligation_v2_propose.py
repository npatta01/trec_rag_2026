"""Freeze inference-free adaptive-obligation v2 proposal jobs."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_obligation_v2_ledger import (
    AppendOnlyAttemptLedger,
    AttemptSpec,
    classify_completion,
)
from .adaptive_obligation_v2_contract import (
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    canonical_sha256,
    sha256_text,
    verify_v2_contract,
)


SCHEMA_VERSION = "adaptive-obligation-v2-proposal-preflight-v1"
JOB_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-job-v1"
PROPOSAL_INVENTORY_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-inventory-v1"
PROPOSAL_RECEIPT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-receipt-v1"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PRIMARY_MAX_NEW_TOKENS = 256
RETRY_MAX_NEW_TOKENS = 512
PRIMARY_JOB_COUNT = 48
TASK2_BUILDER_CODE_SHA256 = (
    "f6c7a9db448d391a2ed7f1ad3eceba0bc8805e825ed4110ac6a7112985032988"
)
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    / MODEL_REVISION
)

_OUTPUT_NAMES = frozenset({"jobs.jsonl", "schema.json", "prompt.json", "receipt.json"})
_PROPOSAL_INVENTORY_OUTPUT_NAMES = frozenset({"proposals.jsonl", "receipt.json"})
_TOKENIZER_FILE_NAMES = frozenset(
    {"config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
)
_ZERO_COUNTERS = (
    "network_call_count",
    "retrieval_call_count",
    "hosted_inference_call_count",
    "paid_call_count",
    "model_load_count",
    "inference_count",
)

PROPOSAL_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reason_code", "o1"],
    "properties": {
        "status": {"enum": ["SUPPORTED", "UNSUPPORTED"]},
        "reason_code": {
            "enum": [
                "SUPPORTED",
                "NO_ABSTRACT_CHILD",
                "ONLY_ANSWER_FACTS",
                "INSUFFICIENT_SCOPE",
            ]
        },
        "o1": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "scope_rationale", "support_unit_ids"],
                    "properties": {
                        "label": {"type": "string", "minLength": 3, "maxLength": 120},
                        "scope_rationale": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 240,
                        },
                        "support_unit_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                    },
                },
            ]
        },
    },
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "SUPPORTED"}}},
            "then": {
                "properties": {
                    "reason_code": {"const": "SUPPORTED"},
                    "o1": {"type": "object"},
                }
            },
            "else": {
                "properties": {
                    "reason_code": {
                        "enum": [
                            "NO_ABSTRACT_CHILD",
                            "ONLY_ANSWER_FACTS",
                            "INSUFFICIENT_SCOPE",
                        ]
                    },
                    "o1": {"type": "null"},
                }
            },
        }
    ],
}

_PROPOSAL_INSTRUCTIONS = """Return exactly one JSON object matching the supplied schema.
Propose at most one abstract child information need of the complete O0 parent, or return UNSUPPORTED.
Preserve the parent subject, population, domain, and requested relation. Use only the supplied exact evidence units; never use outside knowledge.
Do not turn dates, numbers, named examples, causes, mechanisms, outcomes, or other answer facts into an O1 label.
A supported proposal must cite one or two supplied support_unit_ids. Do not copy passages or add prose outside JSON."""


def _compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact_bytes(value: object) -> bytes:
    return (_compact(value) + "\n").encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validated_contract_binding(
    contract: object,
    contract_dir: Path | None,
    contract_receipt_sha256: str | None,
) -> tuple[Path, str]:
    if contract_dir is None:
        raise ValueError("proposal contract path must be non-null and exact")
    root = Path(contract_dir)
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise ValueError("proposal contract path must be absolute and exact")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("proposal contract path must be an existing regular directory")
    receipt_path = root / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("proposal contract receipt path must be a regular file")
    observed_sha256 = _sha256_file(receipt_path)
    if (
        not isinstance(contract_receipt_sha256, str)
        or len(contract_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in contract_receipt_sha256)
        or contract_receipt_sha256 != observed_sha256
    ):
        raise ValueError("proposal contract receipt hash differs")
    if not isinstance(contract, Mapping) or not isinstance(
        contract.get("receipt"), Mapping
    ):
        raise ValueError("proposal contract receipt is missing")
    if _read_json(receipt_path, "proposal contract receipt") != dict(
        contract["receipt"]  # type: ignore[arg-type]
    ):
        raise ValueError("proposal contract receipt bytes differ from source data")
    return root.resolve(), observed_sha256


def _reject_protected_contract(contract: object) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("proposal contract must be an object")
    topic_ids: list[str] = []
    receipt = contract.get("receipt")
    if isinstance(receipt, Mapping):
        raw_topics = receipt.get("topic_ids")
        if isinstance(raw_topics, list):
            topic_ids.extend(str(value) for value in raw_topics)
    for name in ("parents", "reservoirs", "units"):
        rows = contract.get(name)
        if isinstance(rows, list):
            topic_ids.extend(
                str(row.get("topic_id")) for row in rows if isinstance(row, Mapping)
            )
    for topic_id in topic_ids:
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _snapshot_inventory(snapshot_dir: Path = MODEL_SNAPSHOT) -> dict[str, object]:
    """Bind every local snapshot file to the SHA-256 of its actual bytes."""

    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir() or root.name != MODEL_REVISION:
        raise RuntimeError(f"pinned local model snapshot is missing or unsafe: {root}")
    files: list[dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file():
            raise RuntimeError(f"model snapshot contains a non-file entry: {path.name}")
        target = path.resolve(strict=True)
        observed = target.stat()
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"model snapshot target is not regular: {path.name}")
        blob_id = target.name
        files.append(
            {
                "name": path.name,
                "bytes": observed.st_size,
                "blob_id": blob_id,
                "content_sha256": _sha256_file(target),
            }
        )
    names = {str(row["name"]) for row in files}
    weight_names = {name for name in names if name.endswith(".safetensors")}
    if (
        not _TOKENIZER_FILE_NAMES <= names
        or "model.safetensors.index.json" not in names
        or len(weight_names) != 3
    ):
        raise RuntimeError("pinned local Qwen tokenizer/model snapshot is incomplete")
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _tokenizer_file_inventory(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    files = snapshot.get("files")
    if not isinstance(files, list):
        raise ValueError("model snapshot file inventory is missing")
    selected = [
        dict(row)
        for row in files
        if isinstance(row, Mapping) and row.get("name") in _TOKENIZER_FILE_NAMES
    ]
    if {row.get("name") for row in selected} != _TOKENIZER_FILE_NAMES:
        raise ValueError("tokenizer file inventory is incomplete")
    return selected


def _tokenizer_contract(snapshot_dir: Path = MODEL_SNAPSHOT) -> dict[str, object]:
    root = Path(snapshot_dir)
    tokenizer_path = root / "tokenizer.json"
    config_path = root / "tokenizer_config.json"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned tokenizer config is unreadable") from exc
    chat_template = config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("pinned tokenizer chat template is missing")
    return {
        "loader": "tokenizers.Tokenizer.from_file",
        "tokenizer_json_sha256": _sha256_file(tokenizer_path),
        "tokenizer_config_sha256": _sha256(config_bytes),
        "chat_template": chat_template,
        "chat_template_sha256": _sha256(chat_template.encode("utf-8")),
        "torch_imported": False,
        "transformers_imported": False,
    }


def _prompt_contract() -> dict[str, object]:
    return {
        "schema_version": "adaptive-obligation-v2-proposal-prompt-v1",
        "instructions": _PROPOSAL_INSTRUCTIONS,
        "response_schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "decoding": {
            "do_sample": False,
            "temperature": 0,
            "seed": 0,
            "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
            "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
        },
        "retry_policy": {
            "maximum_retries_per_job": 1,
            "only_if_primary_completion_reaches_ceiling_and_json_is_truncated": True,
            "semantic_or_schema_retry_allowed": False,
            "recursive_repair_allowed": False,
        },
    }


def render_proposal_messages(
    parent: Mapping[str, object],
    reservoir: Mapping[str, object],
    units: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Render one parent/fold proposal prompt without invoking a tokenizer or model."""

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
        "evidence_units": [dict(row) for row in units],
        "response_json_schema": PROPOSAL_SCHEMA,
    }
    return [
        {"role": "system", "content": _PROPOSAL_INSTRUCTIONS},
        {"role": "user", "content": _compact(payload)},
    ]


def build_proposal_jobs(contract: object) -> list[dict[str, object]]:
    """Build exactly one deterministic job for each O0 parent/fold reservoir."""

    _reject_protected_contract(contract)
    if not isinstance(contract, Mapping):
        raise ValueError("proposal contract must be an object")
    parents = contract.get("parents")
    reservoirs = contract.get("reservoirs")
    units = contract.get("units")
    if not isinstance(parents, list) or not isinstance(reservoirs, list) or not isinstance(units, list):
        raise ValueError("proposal contract rows are missing")
    receipt = contract.get("receipt")
    if isinstance(receipt, Mapping) and (
        receipt.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("parent_count") != 24
        or receipt.get("reservoir_count") != PRIMARY_JOB_COUNT
        or receipt.get("qrels_opened") is not False
        or any(receipt.get(name) != 0 for name in _ZERO_COUNTERS)
        or receipt.get("tokenizer_load_count") != 0
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("proposal source contract receipt differs")
    if len(parents) != 24 or len(reservoirs) != PRIMARY_JOB_COUNT:
        raise ValueError("proposal contract must contain 24 parents and 48 reservoirs")
    if any(not isinstance(row, Mapping) for row in [*parents, *reservoirs, *units]):
        raise ValueError("proposal contract rows must be objects")
    parent_by_id = {str(row["parent_id"]): row for row in parents}
    unit_by_id = {str(row["unit_id"]): row for row in units}
    if len(parent_by_id) != len(parents) or len(unit_by_id) != len(units):
        raise ValueError("proposal contract contains duplicate parent or unit identities")
    parent_order = [str(row["parent_id"]) for row in parents]
    reservoir_by_pair: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in reservoirs:
        fold = row.get("fold")
        if isinstance(fold, bool) or fold not in (0, 1):
            raise ValueError("proposal reservoir fold must be exactly 0 or 1")
        key = (str(row.get("parent_id")), int(fold))
        if key in reservoir_by_pair:
            raise ValueError("proposal contract contains a duplicate parent/fold reservoir")
        reservoir_by_pair[key] = row
    expected_pairs = {(parent_id, fold) for parent_id in parent_order for fold in (0, 1)}
    if set(reservoir_by_pair) != expected_pairs:
        raise ValueError("proposal reservoirs do not cover every parent/fold exactly once")
    jobs: list[dict[str, object]] = []
    for parent_id in parent_order:
        parent = parent_by_id[parent_id]
        for fold in (0, 1):
            reservoir = reservoir_by_pair[(parent_id, fold)]
            if (
                reservoir.get("topic_id") != parent.get("topic_id")
                or reservoir.get("document_count") != 10
                or not isinstance(reservoir.get("documents"), list)
                or len(reservoir["documents"]) != 10
            ):
                raise ValueError("proposal reservoir identity or document count differs")
            ordered_units: list[Mapping[str, object]] = []
            seen_unit_ids: set[str] = set()
            for document in reservoir["documents"]:
                if not isinstance(document, Mapping) or not isinstance(
                    document.get("unit_ids"), list
                ):
                    raise ValueError("proposal reservoir document unit inventory is invalid")
                for raw_unit_id in document["unit_ids"]:
                    unit_id = str(raw_unit_id)
                    if unit_id in seen_unit_ids or unit_id not in unit_by_id:
                        raise ValueError("proposal unit identity is duplicate or unknown")
                    unit = unit_by_id[unit_id]
                    if (
                        unit.get("topic_id") != parent.get("topic_id")
                        or unit.get("parent_id") != parent_id
                        or unit.get("fold") != fold
                        or unit.get("document_id") != document.get("document_id")
                        or unit.get("window_id") != document.get("window_id")
                    ):
                        raise ValueError("proposal unit identity escapes its parent/fold reservoir")
                    seen_unit_ids.add(unit_id)
                    ordered_units.append(unit)
            if not ordered_units:
                raise ValueError("proposal reservoir must contain exact evidence units")
            messages = render_proposal_messages(parent, reservoir, ordered_units)
            identity = {
                "topic_id": reservoir["topic_id"],
                "parent_id": parent_id,
                "fold": fold,
                "reservoir_id": reservoir["reservoir_id"],
                "messages_sha256": canonical_sha256(messages),
                "schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
            }
            jobs.append(
                {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": canonical_sha256(identity),
                    **identity,
                    "parent_manifest_order": parent["manifest_order"],
                    "input_unit_count": len(ordered_units),
                    "input_unit_ids": [row["unit_id"] for row in ordered_units],
                    "messages": messages,
                    "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
                    "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
                    "retry_policy": _prompt_contract()["retry_policy"],
                }
            )
    if len(jobs) != PRIMARY_JOB_COUNT or len({row["job_id"] for row in jobs}) != PRIMARY_JOB_COUNT:
        raise ValueError("proposal jobs must contain exactly 48 unique parent/fold jobs")
    return jobs


def build_proposal_preflight(
    contract: object,
    *,
    tokenizer: object,
    model_factory: Callable[[], object] | None,
    output_dir: Path,
    contract_dir: Path | None = None,
    contract_receipt_sha256: str | None = None,
    model_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze tokenizer-counted jobs while deliberately ignoring model_factory."""

    _reject_protected_contract(contract)
    del model_factory
    jobs = build_proposal_jobs(contract)
    bound_contract_dir, source_receipt_hash = _validated_contract_binding(
        contract, contract_dir, contract_receipt_sha256
    )
    destination = Path(output_dir)
    if _path_present(destination):
        raise FileExistsError(f"create-only proposal preflight exists: {destination}")
    verified_contract = _load_verified_contract(bound_contract_dir)
    if not isinstance(contract, Mapping) or dict(contract) != verified_contract:
        raise ValueError("proposal data differs from verified contract rows")
    snapshot = dict(model_snapshot) if model_snapshot is not None else _snapshot_inventory()
    tokenizer_files = _tokenizer_file_inventory(snapshot)
    tokenizer_contract = _tokenizer_contract()
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
            raise ValueError("tokenizer returned an invalid proposal prompt token sequence")
        counted.append({**job, "prompt_token_count": len(token_ids)})
    jobs_bytes = b"".join(_compact_bytes(row) for row in counted)
    schema_bytes = _pretty_bytes(PROPOSAL_SCHEMA)
    prompt = _prompt_contract()
    prompt_bytes = _pretty_bytes(prompt)
    contract_receipt: Mapping[str, object] = {}
    if isinstance(contract, Mapping) and isinstance(contract.get("receipt"), Mapping):
        contract_receipt = contract["receipt"]  # type: ignore[assignment]
    code_sha256 = {
        "adaptive_obligation_v2_contract.py": _sha256_file(
            Path(__file__).with_name("adaptive_obligation_v2_contract.py")
        ),
        "adaptive_obligation_v2_propose.py": TASK2_BUILDER_CODE_SHA256,
    }
    token_counts = [int(row["prompt_token_count"]) for row in counted]
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": len(counted),
        "primary_call_count": PRIMARY_JOB_COUNT,
        "retry_call_ceiling": PRIMARY_JOB_COUNT,
        "worst_case_call_ceiling": PRIMARY_JOB_COUNT * 2,
        "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
        "prompt_token_counts": {
            "count": len(token_counts),
            "minimum": min(token_counts),
            "maximum": max(token_counts),
            "total": sum(token_counts),
            "by_job": [
                {"job_id": row["job_id"], "prompt_token_count": row["prompt_token_count"]}
                for row in counted
            ],
        },
        "tokenizer_load_count": 1,
        "model_load_count": 0,
        "inference_count": 0,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": snapshot,
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": tokenizer_contract,
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
        "schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "contract_receipt": {
            "path": str(bound_contract_dir),
            "sha256": source_receipt_hash,
            "schema_version": contract_receipt.get("schema_version"),
            "status": contract_receipt.get("status"),
        },
        "code_sha256": code_sha256,
        "artifacts": {
            "jobs.jsonl": {
                "path": "jobs.jsonl",
                "bytes": len(jobs_bytes),
                "rows": len(counted),
                "sha256": _sha256(jobs_bytes),
            },
            "schema.json": {
                "path": "schema.json",
                "bytes": len(schema_bytes),
                "rows": 1,
                "sha256": _sha256(schema_bytes),
            },
            "prompt.json": {
                "path": "prompt.json",
                "bytes": len(prompt_bytes),
                "rows": 1,
                "sha256": _sha256(prompt_bytes),
            },
        },
        "expected_runtime": {
            "phase": "inference_free_preflight",
            "proposal_calls_executed": 0,
            "validation_calls_executed": 0,
        },
        "planned_storage": {
            "artifact_names": ["jobs.jsonl", "schema.json", "prompt.json", "receipt.json"],
            "job_rows": PRIMARY_JOB_COUNT,
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    _publish_preflight(
        destination,
        {
            "jobs.jsonl": jobs_bytes,
            "schema.json": schema_bytes,
            "prompt.json": prompt_bytes,
            "receipt.json": _pretty_bytes(receipt),
        },
    )
    return receipt


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"create-only proposal preflight exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _require_output_inventory(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("proposal preflight root must be a regular directory")
    entries = list(root.iterdir())
    if {path.name for path in entries} != _OUTPUT_NAMES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("proposal preflight inventory is partial, unexpected, or unsafe")


def _publish_preflight(destination: Path, contents: Mapping[str, bytes]) -> None:
    if _path_present(destination):
        raise FileExistsError(f"create-only proposal preflight exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("proposal preflight parent must not be a symlink")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        for name in ("jobs.jsonl", "schema.json", "prompt.json", "receipt.json"):
            _write_fsynced(staging / name, contents[name])
        _require_output_inventory(staging)
        if any((staging / name).read_bytes() != content for name, content in contents.items()):
            raise ValueError("staged proposal preflight bytes differ")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        source = path.read_bytes()
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict) or source != _pretty_bytes(value):
        raise ValueError(f"{label} must be one canonical JSON object")
    return value


def _read_jobs(path: Path) -> tuple[list[dict[str, object]], bytes]:
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError("jobs.jsonl is unreadable") from exc
    return _read_jobs_bytes(source)


def _read_jobs_bytes(source: bytes) -> tuple[list[dict[str, object]], bytes]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"jobs.jsonl:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict) or line != _compact_bytes(row).rstrip(b"\n"):
            raise ValueError(f"jobs.jsonl:{line_number} is not canonical")
        rows.append(row)
    return rows, source


def _artifact_matches(
    name: str, content: bytes, rows: int, bindings: Mapping[str, object]
) -> None:
    binding = bindings.get(name)
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != name
        or binding.get("bytes") != len(content)
        or binding.get("rows") != rows
        or binding.get("sha256") != _sha256(content)
    ):
        raise ValueError(f"proposal preflight {name} binding differs")


def verify_proposal_preflight(output_dir: Path) -> dict[str, object]:
    """Authenticate sources/jobs, then reload only the pinned tokenizer to recount."""

    root = Path(output_dir)
    _require_output_inventory(root)
    receipt = _read_json(root / "receipt.json", "proposal preflight receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("job_count") != PRIMARY_JOB_COUNT
        or receipt.get("primary_call_count") != PRIMARY_JOB_COUNT
        or receipt.get("retry_call_ceiling") != PRIMARY_JOB_COUNT
        or receipt.get("worst_case_call_ceiling") != PRIMARY_JOB_COUNT * 2
        or receipt.get("primary_max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
        or receipt.get("retry_max_new_tokens") != RETRY_MAX_NEW_TOKENS
        or receipt.get("tokenizer_load_count") != 1
        or any(receipt.get(name) != 0 for name in _ZERO_COUNTERS)
        or receipt.get("qrels_opened") is not False
        or receipt.get("external_cost_usd") != 0.0
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("model_construction_allowed") is not False
        or receipt.get("generation_allowed") is not False
    ):
        raise ValueError("proposal preflight counts, model, or safety receipt differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "jobs.jsonl",
        "schema.json",
        "prompt.json",
    }:
        raise ValueError("proposal preflight artifact bindings differ")
    jobs, jobs_bytes = _read_jobs(root / "jobs.jsonl")
    schema_bytes = (root / "schema.json").read_bytes()
    prompt_bytes = (root / "prompt.json").read_bytes()
    _artifact_matches("jobs.jsonl", jobs_bytes, len(jobs), artifacts)
    _artifact_matches("schema.json", schema_bytes, 1, artifacts)
    _artifact_matches("prompt.json", prompt_bytes, 1, artifacts)
    schema = _read_json(root / "schema.json", "proposal schema")
    prompt = _read_json(root / "prompt.json", "proposal prompt")
    if (
        schema != PROPOSAL_SCHEMA
        or prompt != _prompt_contract()
        or receipt.get("schema_sha256") != canonical_sha256(schema)
        or receipt.get("prompt_sha256") != canonical_sha256(prompt)
    ):
        raise ValueError("proposal schema or prompt contract differs")
    if len(jobs) != PRIMARY_JOB_COUNT or len({row.get("job_id") for row in jobs}) != PRIMARY_JOB_COUNT:
        raise ValueError("jobs.jsonl must contain exactly 48 unique jobs")
    pairs: set[tuple[str, int]] = set()
    for row in jobs:
        fold = row.get("fold")
        messages = row.get("messages")
        if (
            row.get("schema_version") != JOB_SCHEMA_VERSION
            or fold not in (0, 1)
            or isinstance(fold, bool)
            or not isinstance(messages, list)
            or row.get("messages_sha256") != canonical_sha256(messages)
            or row.get("schema_sha256") != canonical_sha256(PROPOSAL_SCHEMA)
            or row.get("primary_max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
            or row.get("retry_max_new_tokens") != RETRY_MAX_NEW_TOKENS
            or not isinstance(row.get("prompt_token_count"), int)
            or isinstance(row.get("prompt_token_count"), bool)
            or int(row["prompt_token_count"]) < 1
        ):
            raise ValueError("jobs.jsonl contains an invalid frozen proposal job")
        identity = {
            "topic_id": row.get("topic_id"),
            "parent_id": row.get("parent_id"),
            "fold": fold,
            "reservoir_id": row.get("reservoir_id"),
            "messages_sha256": row.get("messages_sha256"),
            "schema_sha256": row.get("schema_sha256"),
        }
        if row.get("job_id") != canonical_sha256(identity):
            raise ValueError("jobs.jsonl job identity differs")
        pairs.add((str(row.get("parent_id")), int(fold)))
    if len(pairs) != PRIMARY_JOB_COUNT:
        raise ValueError("jobs.jsonl parent/fold identities differ")
    prompt_counts = receipt.get("prompt_token_counts")
    counts = [int(row["prompt_token_count"]) for row in jobs]
    if (
        not isinstance(prompt_counts, Mapping)
        or prompt_counts.get("count") != len(counts)
        or prompt_counts.get("minimum") != min(counts)
        or prompt_counts.get("maximum") != max(counts)
        or prompt_counts.get("total") != sum(counts)
        or prompt_counts.get("by_job")
        != [
            {"job_id": row["job_id"], "prompt_token_count": row["prompt_token_count"]}
            for row in jobs
        ]
    ):
        raise ValueError("proposal prompt token counts differ")
    expected_code = {
        "adaptive_obligation_v2_contract.py": _sha256_file(
            Path(__file__).with_name("adaptive_obligation_v2_contract.py")
        ),
        "adaptive_obligation_v2_propose.py": TASK2_BUILDER_CODE_SHA256,
    }
    if receipt.get("code_sha256") != expected_code:
        raise ValueError("proposal frozen builder code hash differs")
    source_binding = receipt.get("contract_receipt")
    if not isinstance(source_binding, Mapping):
        raise ValueError("proposal contract receipt binding is missing")
    source_path = source_binding.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("proposal contract receipt path must be non-null and exact")
    contract_root = Path(source_path)
    if not contract_root.is_absolute() or str(contract_root.resolve()) != source_path:
        raise ValueError("proposal contract receipt path must be absolute and exact")
    receipt_path = contract_root / "receipt.json"
    expected_receipt_sha256 = source_binding.get("sha256")
    try:
        observed_receipt_sha256 = _sha256_file(receipt_path)
    except OSError as exc:
        raise ValueError("proposal source contract receipt hash is unavailable") from exc
    if (
        not isinstance(expected_receipt_sha256, str)
        or len(expected_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_receipt_sha256)
        or expected_receipt_sha256 != observed_receipt_sha256
        or source_binding.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or source_binding.get("status") != "complete"
    ):
        raise ValueError("proposal source contract receipt hash or identity differs")
    contract = _load_verified_contract(contract_root)
    expected_jobs = build_proposal_jobs(contract)
    for observed, expected in zip(jobs, expected_jobs, strict=True):
        without_count = {
            key: value for key, value in observed.items() if key != "prompt_token_count"
        }
        if without_count != expected:
            raise ValueError(
                "jobs.jsonl differs from the authenticated contract reconstruction"
            )
    observed_snapshot = _snapshot_inventory()
    if receipt.get("model_snapshot") != observed_snapshot or receipt.get(
        "tokenizer_files"
    ) != _tokenizer_file_inventory(observed_snapshot):
        raise ValueError("proposal local model snapshot or tokenizer files differ")
    if receipt.get("tokenizer_contract") != _tokenizer_contract():
        raise ValueError("proposal pinned tokenizer or chat template binding differs")
    tokenizer = _load_local_tokenizer()
    recomputed_counts = [
        len(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
        )
        for row in jobs
    ]
    if recomputed_counts != counts:
        raise ValueError(
            "proposal prompt token counts differ from recomputed pinned tokenizer counts"
        )
    return receipt


def _load_verified_contract(contract_dir: Path) -> dict[str, object]:
    root = Path(contract_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("proposal source contract root must be a regular directory")
    receipt = _read_json(root / "receipt.json", "proposal source contract receipt")
    raw_topics = receipt.get("topic_ids")
    if not isinstance(raw_topics, list):
        raise ValueError("proposal source contract topic_ids are missing")
    for topic_id in map(str, raw_topics):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    verified = verify_v2_contract(root)
    if receipt != verified:
        raise ValueError("proposal source contract verifier receipt differs")

    def load_jsonl(name: str) -> list[dict[str, object]]:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"proposal source {name} must be a regular file")
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"proposal source {name}:{line_number} is invalid") from exc
            if not isinstance(row, dict) or line != _compact_bytes(row).rstrip(b"\n"):
                raise ValueError(f"proposal source {name}:{line_number} is not canonical")
            rows.append(row)
        return rows

    return {
        "parents": load_jsonl("parents.jsonl"),
        "reservoirs": load_jsonl("reservoirs.jsonl"),
        "units": load_jsonl("units.jsonl"),
        "receipt": receipt,
    }


def _load_local_tokenizer(snapshot_dir: Path = MODEL_SNAPSHOT) -> object:
    """Construct only the pinned tokenizer, never a model class."""

    before = set(sys.modules)
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    imported = set(sys.modules) - before
    if any(
        name == "torch"
        or name.startswith("torch.")
        or name == "transformers"
        or name.startswith("transformers.")
        for name in imported
    ):
        raise RuntimeError("tokenizer-only loader imported a forbidden model runtime")

    root = Path(snapshot_dir)
    config = json.loads((root / "tokenizer_config.json").read_bytes())
    chat_template = config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("pinned tokenizer chat template is missing")
    backend = Tokenizer.from_file(str(root / "tokenizer.json"))
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    ).from_string(chat_template)

    class LocalChatTokenizer:
        def apply_chat_template(
            self,
            messages: object,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            if not tokenize or not isinstance(messages, Sequence) or isinstance(
                messages, (str, bytes)
            ):
                raise ValueError("proposal preflight requires tokenized chat messages")
            rendered = template.render(
                messages=[dict(row) for row in messages],
                tools=None,
                add_generation_prompt=add_generation_prompt,
            )
            return backend.encode(rendered, add_special_tokens=False).ids

    return LocalChatTokenizer()


@dataclass(frozen=True)
class _CapturedApproval:
    value: dict[str, object]
    source: bytes
    sha256: str


@dataclass(frozen=True)
class _CapturedPreflight:
    receipt: dict[str, object]
    receipt_sha256: str
    jobs: list[dict[str, object]]
    contents: dict[str, bytes]


def _open_directory_no_symlinks(path: Path) -> int:
    raw = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/" if raw.is_absolute() else ".", flags)
    try:
        for component in raw.parts:
            if component in (raw.anchor, "", "."):
                continue
            if component == "..":
                raise OSError("parent traversal is forbidden")
            child = os.open(component, flags | nofollow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise OSError("path is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_stable_regular_at(
    directory_fd: int,
    name: str,
    *,
    require_single_link: bool,
) -> bytes:
    if not name or name in (".", "..") or "/" in name:
        raise OSError("unsafe file name")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            require_single_link and before.st_nlink != 1
        ):
            raise OSError("path is not a single-link regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        source = b"".join(chunks)
        if identity_before != identity_after or len(source) != after.st_size:
            raise OSError("file changed while being captured")
        return source
    finally:
        os.close(descriptor)


def _capture_regular_file_no_symlinks(path: Path) -> bytes:
    source_path = Path(path)
    if not source_path.name:
        raise OSError("file path is missing")
    directory_fd = _open_directory_no_symlinks(source_path.parent)
    try:
        return _read_stable_regular_at(
            directory_fd,
            source_path.name,
            require_single_link=True,
        )
    finally:
        os.close(directory_fd)


def _capture_inference_approval(path: Path) -> _CapturedApproval:
    """Capture and authenticate approval bytes through a no-symlink descriptor walk."""

    try:
        source = _capture_regular_file_no_symlinks(Path(path))
        approval = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("proposal inference approval required") from exc
    required_names = {
        "schema_version",
        "stage",
        "preflight_sha256",
        "model",
        "model_revision",
        "primary_call_count",
        "retry_call_ceiling",
        "approved",
    }
    if (
        not isinstance(approval, dict)
        or source != _pretty_bytes(approval)
        or not required_names <= set(approval)
        or approval.get("schema_version")
        != "adaptive-obligation-v2-proposal-approval-v1"
        or approval.get("stage") != "proposal"
        or approval.get("model") != MODEL_ID
        or approval.get("model_revision") != MODEL_REVISION
        or type(approval.get("primary_call_count")) is not int
        or approval.get("primary_call_count") != PRIMARY_JOB_COUNT
        or type(approval.get("retry_call_ceiling")) is not int
        or approval.get("retry_call_ceiling") != PRIMARY_JOB_COUNT
        or approval.get("approved") is not True
        or not isinstance(approval.get("preflight_sha256"), str)
        or len(str(approval["preflight_sha256"])) != 64
        or any(
            char not in "0123456789abcdef"
            for char in str(approval["preflight_sha256"])
        )
    ):
        raise PermissionError("proposal inference approval required")
    return _CapturedApproval(
        value=approval,
        source=source,
        sha256=_sha256(source),
    )


def _load_inference_approval(path: Path) -> dict[str, object]:
    return _capture_inference_approval(path).value


def _capture_preflight_files(path: Path) -> dict[str, bytes]:
    expected = {"jobs.jsonl", "schema.json", "prompt.json", "receipt.json"}
    try:
        directory_fd = _open_directory_no_symlinks(Path(path))
        try:
            before = os.fstat(directory_fd)
            names_before = set(os.listdir(directory_fd))
            if names_before != expected:
                raise OSError("preflight inventory differs")
            contents = {
                name: _read_stable_regular_at(
                    directory_fd,
                    name,
                    require_single_link=True,
                )
                for name in sorted(expected)
            }
            names_after = set(os.listdir(directory_fd))
            after = os.fstat(directory_fd)
            if (
                names_after != names_before
                or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns
            ):
                raise OSError("preflight changed while being captured")
            return contents
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError("proposal preflight is missing or unsafe") from exc


def _capture_and_verify_preflight(
    path: Path,
    *,
    expected_receipt_sha256: str,
    verifier: Callable[[Path], dict[str, object]] = verify_proposal_preflight,
) -> _CapturedPreflight:
    """Capture once, verify an isolated copy, and retain only captured jobs."""

    contents = _capture_preflight_files(path)
    receipt_source = contents["receipt.json"]
    receipt_sha256 = _sha256(receipt_source)
    if receipt_sha256 != expected_receipt_sha256:
        raise PermissionError("proposal inference approval required")
    try:
        receipt = json.loads(receipt_source)
        schema = json.loads(contents["schema.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("captured proposal preflight JSON is invalid") from exc
    if (
        not isinstance(receipt, dict)
        or receipt_source != _pretty_bytes(receipt)
        or not isinstance(schema, dict)
        or contents["schema.json"] != _pretty_bytes(schema)
        or schema != PROPOSAL_SCHEMA
    ):
        raise ValueError("captured proposal preflight is not canonical")
    jobs, jobs_source = _read_jobs_bytes(contents["jobs.jsonl"])
    if (
        jobs_source != contents["jobs.jsonl"]
        or len(jobs) != PRIMARY_JOB_COUNT
        or len({row.get("job_id") for row in jobs}) != PRIMARY_JOB_COUNT
    ):
        raise ValueError("captured proposal jobs must be exactly 48 unique rows")
    with tempfile.TemporaryDirectory(prefix="adaptive-v2-preflight-") as temporary:
        private_root = Path(temporary) / "snapshot"
        private_root.mkdir(mode=0o700)
        try:
            for name in ("jobs.jsonl", "schema.json", "prompt.json", "receipt.json"):
                _write_fsynced(private_root / name, contents[name])
                os.chmod(private_root / name, 0o400)
            _fsync_directory(private_root)
            os.chmod(private_root, 0o500)
            verified = verifier(private_root)
        finally:
            os.chmod(private_root, 0o700)
        if verified != receipt:
            raise ValueError("isolated proposal preflight verifier receipt differs")
    return _CapturedPreflight(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        jobs=jobs,
        contents=contents,
    )


def _completion_parts(generated: object, ceiling: int) -> tuple[bytes, int]:
    if (
        isinstance(generated, tuple)
        and len(generated) == 2
        and isinstance(generated[0], bytes)
        and type(generated[1]) is int
    ):
        return generated[0], generated[1]
    raise TypeError("model generation must return raw bytes and an exact token count")


def _proposal_request(job: Mapping[str, object], ceiling: int) -> dict[str, object]:
    return {
        "stage": "proposal",
        "job_id": job["job_id"],
        "messages_sha256": canonical_sha256(job["messages"]),
        "schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "max_new_tokens": ceiling,
    }


def _build_run_anchor(
    jobs: Sequence[Mapping[str, object]],
    *,
    preflight_sha256: str,
    approval_sha256: str,
) -> dict[str, object]:
    if len(jobs) != PRIMARY_JOB_COUNT or len(
        {row.get("job_id") for row in jobs}
    ) != PRIMARY_JOB_COUNT:
        raise ValueError("proposal execution requires exactly 48 captured jobs")
    anchored_jobs: list[dict[str, object]] = []
    for job in jobs:
        job_id = job.get("job_id")
        unit_ids = job.get("input_unit_ids")
        if (
            not isinstance(job_id, str)
            or len(job_id) != 64
            or any(char not in "0123456789abcdef" for char in job_id)
            or not isinstance(unit_ids, list)
        ):
            raise ValueError("captured proposal job inventory differs")
        attempts = [
            {
                "attempt_ordinal": ordinal,
                "request_sha256": canonical_sha256(_proposal_request(job, ceiling)),
                "max_new_tokens": ceiling,
            }
            for ordinal, ceiling in enumerate(
                (PRIMARY_MAX_NEW_TOKENS, RETRY_MAX_NEW_TOKENS), start=1
            )
        ]
        anchored_jobs.append(
            {
                "job_id": job_id,
                "input_unit_ids": list(unit_ids),
                "attempts": attempts,
            }
        )
    return {
        "schema_version": "adaptive-obligation-v2-run-anchor-v1",
        "stage": "proposal",
        "preflight_sha256": preflight_sha256,
        "approval_sha256": approval_sha256,
        "job_count": len(anchored_jobs),
        "job_inventory_sha256": canonical_sha256(anchored_jobs),
        "jobs": anchored_jobs,
        "schema": PROPOSAL_SCHEMA,
    }


def _job_source_unit_bindings(
    job: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    messages = job.get("messages")
    input_unit_ids = job.get("input_unit_ids")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[1], Mapping)
        or not isinstance(messages[1].get("content"), str)
        or not isinstance(input_unit_ids, list)
    ):
        raise ValueError("proposal inventory frozen job messages differ")
    try:
        payload = json.loads(str(messages[1]["content"]))
    except json.JSONDecodeError as exc:
        raise ValueError("proposal inventory frozen job payload is invalid") from exc
    evidence_units = payload.get("evidence_units") if isinstance(payload, dict) else None
    if not isinstance(evidence_units, list) or any(
        not isinstance(unit, Mapping) for unit in evidence_units
    ):
        raise ValueError("proposal inventory source units differ")
    bindings: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    identity_names = (
        "topic_id",
        "parent_id",
        "fold",
        "document_id",
        "window_id",
        "start",
        "end",
        "text",
    )
    for source in evidence_units:
        unit = dict(source)
        if any(name not in unit for name in identity_names):
            raise ValueError("proposal inventory source unit identity differs")
        text = unit["text"]
        identity = {name: unit[name] for name in identity_names}
        unit_id = unit.get("unit_id")
        if (
            not isinstance(text, str)
            or not isinstance(unit_id, str)
            or unit_id != canonical_sha256(identity)
            or unit.get("text_sha256") != sha256_text(text)
            or unit.get("topic_id") != job.get("topic_id")
            or unit.get("parent_id") != job.get("parent_id")
            or unit.get("fold") != job.get("fold")
            or unit_id in by_id
        ):
            raise ValueError("proposal inventory source unit identity or hash differs")
        source_identity = {
            "topic_id": unit["topic_id"],
            "parent_id": unit["parent_id"],
            "fold": unit["fold"],
            "document_id": unit["document_id"],
            "window_id": unit["window_id"],
            "start": unit["start"],
            "end": unit["end"],
            "text_sha256": unit["text_sha256"],
        }
        binding = {
            "unit_id": unit_id,
            "document_id": unit["document_id"],
            "window_id": unit["window_id"],
            "text_sha256": unit["text_sha256"],
            "source_identity_sha256": canonical_sha256(source_identity),
        }
        bindings.append(binding)
        by_id[unit_id] = binding
    if [binding["unit_id"] for binding in bindings] != input_unit_ids:
        raise ValueError("proposal inventory source unit order differs")
    return bindings, by_id


def _proposal_inventory_record(
    job: Mapping[str, object],
    sealed_result: Mapping[str, object],
    *,
    source_contract_receipt_sha256: str,
) -> dict[str, object]:
    if sealed_result.get("job_id") != job.get("job_id"):
        raise ValueError("proposal inventory sealed job identity differs")
    value = sealed_result.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("proposal inventory sealed value differs")
    all_units, unit_by_id = _job_source_unit_bindings(job)
    status = value.get("status")
    reason_code = value.get("reason_code")
    if status == "SUPPORTED":
        o1 = value.get("o1")
        if not isinstance(o1, Mapping):
            raise ValueError("proposal inventory supported value differs")
        label = o1.get("label")
        rationale = o1.get("scope_rationale")
        support_unit_ids = o1.get("support_unit_ids")
        if (
            not isinstance(label, str)
            or not isinstance(rationale, str)
            or not isinstance(support_unit_ids, list)
            or any(unit_id not in unit_by_id for unit_id in support_unit_ids)
        ):
            raise ValueError("proposal inventory supported fields differ")
        support_units = [unit_by_id[str(unit_id)] for unit_id in support_unit_ids]
        rationale_sha256: str | None = sha256_text(rationale)
    elif status == "UNSUPPORTED":
        label = None
        support_unit_ids = []
        support_units = []
        rationale_sha256 = None
    else:
        raise ValueError("proposal inventory status differs")
    identity = {
        "job_id": job.get("job_id"),
        "topic_id": job.get("topic_id"),
        "parent_id": job.get("parent_id"),
        "proposal_fold": job.get("fold"),
        "parent_manifest_order": job.get("parent_manifest_order"),
        "source_reservoir_id": job.get("reservoir_id"),
        "status": status,
        "reason_code": reason_code,
        "label": label,
        "scope_rationale_sha256": rationale_sha256,
        "support_unit_ids": support_unit_ids,
        "support_units": support_units,
        "source_unit_inventory_sha256": canonical_sha256(all_units),
        "source_contract_receipt_sha256": source_contract_receipt_sha256,
        "job_messages_sha256": job.get("messages_sha256"),
        "proposal_schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "attempt_ordinal": sealed_result.get("attempt_ordinal"),
        "output_token_count": sealed_result.get("output_token_count"),
        "raw_completion_bytes": sealed_result.get("raw_bytes"),
        "raw_completion_sha256": sealed_result.get("raw_sha256"),
    }
    if (
        not isinstance(identity["topic_id"], str)
        or identity["topic_id"] not in PILOT_TOPIC_IDS
        or not isinstance(identity["parent_id"], str)
        or identity["proposal_fold"] not in (0, 1)
        or isinstance(identity["proposal_fold"], bool)
        or not isinstance(identity["parent_manifest_order"], int)
        or isinstance(identity["parent_manifest_order"], bool)
        or not isinstance(identity["source_reservoir_id"], str)
        or identity["job_messages_sha256"] != canonical_sha256(job.get("messages"))
    ):
        raise ValueError("proposal inventory frozen job metadata differs")
    return {
        "schema_version": PROPOSAL_INVENTORY_SCHEMA_VERSION,
        "proposal_id": canonical_sha256(identity),
        **identity,
    }


def _proposal_inventory_material(
    captured_preflight: _CapturedPreflight,
    *,
    approval_sha256: str,
    sealed: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, bytes]]:
    jobs = captured_preflight.jobs
    expected_anchor = _build_run_anchor(
        jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=approval_sha256,
    )
    if sealed.get("anchor") != expected_anchor:
        raise ValueError("proposal inventory differs from the sealed ledger anchor")
    results = sealed.get("results")
    completion = sealed.get("completion")
    if (
        not isinstance(results, list)
        or len(results) != len(jobs)
        or not isinstance(completion, Mapping)
        or completion.get("completed_job_count") != len(jobs)
        or sealed.get("anchor_sha256") != completion.get("anchor_sha256")
    ):
        raise ValueError("proposal inventory sealed completion differs")
    source_contract = captured_preflight.receipt.get("contract_receipt")
    if not isinstance(source_contract, Mapping) or not isinstance(
        source_contract.get("sha256"), str
    ):
        raise ValueError("proposal inventory source contract binding differs")
    jobs_source = captured_preflight.contents.get("jobs.jsonl")
    jobs_binding = captured_preflight.receipt.get("artifacts")
    jobs_artifact = (
        jobs_binding.get("jobs.jsonl") if isinstance(jobs_binding, Mapping) else None
    )
    if (
        not isinstance(jobs_source, bytes)
        or not isinstance(jobs_artifact, Mapping)
        or jobs_artifact.get("sha256") != _sha256(jobs_source)
        or jobs_artifact.get("bytes") != len(jobs_source)
        or jobs_artifact.get("rows") != len(jobs)
    ):
        raise ValueError("proposal inventory preflight job artifact differs")
    proposal_rows = [
        _proposal_inventory_record(
            job,
            result,
            source_contract_receipt_sha256=str(source_contract["sha256"]),
        )
        for job, result in zip(jobs, results, strict=True)
    ]
    if len({row["proposal_id"] for row in proposal_rows}) != len(proposal_rows):
        raise ValueError("proposal inventory contains duplicate proposal identities")
    proposals_source = b"".join(_compact_bytes(row) for row in proposal_rows)
    supported_count = sum(row["status"] == "SUPPORTED" for row in proposal_rows)
    receipt: dict[str, object] = {
        "schema_version": PROPOSAL_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": len(jobs),
        "proposal_count": len(proposal_rows),
        "supported_count": supported_count,
        "unsupported_count": len(proposal_rows) - supported_count,
        "proposals_sha256": _sha256(proposals_source),
        "proposal_schema_sha256": canonical_sha256(PROPOSAL_SCHEMA),
        "proposal_preflight_receipt_sha256": captured_preflight.receipt_sha256,
        "proposal_preflight_jobs_sha256": _sha256(jobs_source),
        "source_contract_receipt_sha256": source_contract["sha256"],
        "approval_sha256": approval_sha256,
        "run_anchor_sha256": sealed.get("anchor_sha256"),
        "completion_sha256": sealed.get("completion_sha256"),
        "completion": dict(completion),
        "primary_call_count": len(proposal_rows),
        "retry_call_count": sum(
            row.get("attempt_ordinal") == 2 for row in results if isinstance(row, Mapping)
        ),
        "artifacts": {
            "proposals.jsonl": {
                "path": "proposals.jsonl",
                "rows": len(proposal_rows),
                "bytes": len(proposals_source),
                "sha256": _sha256(proposals_source),
            }
        },
    }
    contents = {
        "proposals.jsonl": proposals_source,
        "receipt.json": _pretty_bytes(receipt),
    }
    return proposal_rows, receipt, contents


def _require_proposal_inventory(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("proposal inventory root must be a regular directory")
    entries = list(root.iterdir())
    if {path.name for path in entries} != _PROPOSAL_INVENTORY_OUTPUT_NAMES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("proposal inventory is partial, unexpected, or unsafe")


def _publish_proposal_inventory(
    destination: Path, contents: Mapping[str, bytes]
) -> None:
    if _path_present(destination):
        raise FileExistsError(f"create-only proposal inventory exists: {destination}")
    try:
        parent_descriptor = _open_directory_no_symlinks(destination.parent)
    except OSError as exc:
        raise ValueError("proposal inventory output parent is missing or unsafe") from exc
    else:
        os.close(parent_descriptor)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        for name in ("proposals.jsonl", "receipt.json"):
            _write_fsynced(staging / name, contents[name])
        _require_proposal_inventory(staging)
        if any((staging / name).read_bytes() != source for name, source in contents.items()):
            raise ValueError("staged proposal inventory bytes differ")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _capture_proposal_inventory_files(path: Path) -> dict[str, bytes]:
    try:
        directory_fd = _open_directory_no_symlinks(Path(path))
        try:
            before = os.fstat(directory_fd)
            names_before = set(os.listdir(directory_fd))
            if names_before != _PROPOSAL_INVENTORY_OUTPUT_NAMES:
                raise OSError("proposal inventory differs")
            contents = {
                name: _read_stable_regular_at(
                    directory_fd, name, require_single_link=True
                )
                for name in sorted(_PROPOSAL_INVENTORY_OUTPUT_NAMES)
            }
            names_after = set(os.listdir(directory_fd))
            after = os.fstat(directory_fd)
            if (
                names_after != names_before
                or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns
            ):
                raise OSError("proposal inventory changed while being captured")
            return contents
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError("proposal inventory is missing or unsafe") from exc


def _read_proposal_rows(source: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"proposals.jsonl:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict) or line != _compact_bytes(row).rstrip(b"\n"):
            raise ValueError(f"proposals.jsonl:{line_number} is not canonical")
        rows.append(row)
    return rows


def finalize_proposal_inventory(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish canonical proposals only by replaying a sealed approved run."""

    captured_approval = _capture_inference_approval(Path(approval_path))
    captured_preflight = _capture_and_verify_preflight(
        Path(preflight_dir),
        expected_receipt_sha256=str(captured_approval.value["preflight_sha256"]),
    )
    preflight = {
        **captured_preflight.receipt,
        "receipt_sha256": captured_preflight.receipt_sha256,
    }
    from .adaptive_obligation_v2_local_model import verify_inference_approval

    verify_inference_approval(captured_approval.value, preflight)
    expected_anchor = _build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=captured_approval.sha256,
    )
    ledger = AppendOnlyAttemptLedger(
        Path(ledger_dir), expected_anchor=expected_anchor, create_only=False
    )
    sealed = ledger.read_sealed_results()
    _rows, receipt, contents = _proposal_inventory_material(
        captured_preflight,
        approval_sha256=captured_approval.sha256,
        sealed=sealed,
    )
    _publish_proposal_inventory(Path(output_dir), contents)
    return receipt


def load_authenticated_proposal_inventory(
    *,
    output_dir: Path,
    preflight_dir: Path,
    ledger_dir: Path,
) -> dict[str, object]:
    """Capture and replay-verify the durable proposal inventory for Task 4."""

    contents = _capture_proposal_inventory_files(Path(output_dir))
    try:
        receipt = json.loads(contents["receipt.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("proposal inventory receipt is invalid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or contents["receipt.json"] != _pretty_bytes(receipt)
        or receipt.get("schema_version") != PROPOSAL_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or not isinstance(receipt.get("proposal_preflight_receipt_sha256"), str)
        or not isinstance(receipt.get("approval_sha256"), str)
    ):
        raise ValueError("proposal inventory receipt differs")
    captured_preflight = _capture_and_verify_preflight(
        Path(preflight_dir),
        expected_receipt_sha256=str(receipt["proposal_preflight_receipt_sha256"]),
    )
    expected_anchor = _build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=str(receipt["approval_sha256"]),
    )
    ledger = AppendOnlyAttemptLedger(
        Path(ledger_dir), expected_anchor=expected_anchor, create_only=False
    )
    sealed = ledger.read_sealed_results()
    expected_rows, expected_receipt, expected_contents = _proposal_inventory_material(
        captured_preflight,
        approval_sha256=str(receipt["approval_sha256"]),
        sealed=sealed,
    )
    observed_rows = _read_proposal_rows(contents["proposals.jsonl"])
    if (
        contents != expected_contents
        or receipt != expected_receipt
        or observed_rows != expected_rows
    ):
        raise ValueError("proposal inventory differs from the sealed ledger replay")
    return {
        "proposals": observed_rows,
        "receipt": receipt,
        "receipt_sha256": _sha256(contents["receipt.json"]),
    }


def run_job_with_retry(
    job: Mapping[str, object],
    *,
    generate: Callable[[int], object],
    ledger: AppendOnlyAttemptLedger | None = None,
) -> dict[str, object]:
    """Run one job, retrying only incomplete JSON at the exact primary ceiling."""

    if (
        job.get("primary_max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
        or job.get("retry_max_new_tokens") != RETRY_MAX_NEW_TOKENS
        or not isinstance(job.get("job_id"), str)
        or len(str(job["job_id"])) != 64
        or not isinstance(job.get("messages"), list)
        or not isinstance(job.get("input_unit_ids"), list)
    ):
        raise ValueError("proposal execution job differs from the frozen contract")
    ceilings = (PRIMARY_MAX_NEW_TOKENS, RETRY_MAX_NEW_TOKENS)
    for attempt_ordinal, ceiling in enumerate(ceilings, start=1):
        if ledger is None:
            raw, output_tokens = _completion_parts(generate(ceiling), ceiling)
            classified = classify_completion(
                raw,
                schema=PROPOSAL_SCHEMA,
                output_token_count=output_tokens,
                max_new_tokens=ceiling,
                allowed_support_unit_ids=[str(value) for value in job["input_unit_ids"]],
            )
        else:
            request = _proposal_request(job, ceiling)
            spec = AttemptSpec(
                stage="proposal",
                job_id=str(job["job_id"]),
                attempt_ordinal=attempt_ordinal,
                request_sha256=canonical_sha256(request),
                max_new_tokens=ceiling,
            )

            def invoke(_messages: object, _schema: object, value: int) -> object:
                if value != ceiling:
                    raise ValueError("proposal ledger output ceiling differs")
                return generate(value)

            classified = ledger.run_attempt(
                spec,
                invoke,
                messages=job["messages"],
                schema=PROPOSAL_SCHEMA,
                allowed_support_unit_ids=[str(value) for value in job["input_unit_ids"]],
            )
        classification = classified.get("classification")
        if classification == "valid":
            value = classified.get("value")
            if not isinstance(value, dict):
                raise ValueError("proposal schema result is not an object")
            return value
        if classification == "truncated_at_ceiling" and attempt_ordinal == 1:
            continue
        error = classified.get("error", classification)
        if classification == "truncated_at_ceiling":
            raise ValueError(f"proposal parse truncated at retry ceiling: {error}")
        raise ValueError(f"proposal {error}")
    raise RuntimeError("proposal retry policy exhausted unexpectedly")


def execute_proposals(
    *,
    preflight_dir: Path,
    approval_path: Path,
    output_dir: Path,
    model_factory: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Execute frozen jobs only after a separate create-only approval authenticates."""

    captured_approval = _capture_inference_approval(Path(approval_path))
    approval = captured_approval.value
    captured_preflight = _capture_and_verify_preflight(
        Path(preflight_dir),
        expected_receipt_sha256=str(approval["preflight_sha256"]),
    )
    preflight = {
        **captured_preflight.receipt,
        "receipt_sha256": captured_preflight.receipt_sha256,
    }
    from .adaptive_obligation_v2_local_model import (
        V2LocalJsonModel,
        verify_inference_approval,
    )

    verify_inference_approval(approval, preflight)
    jobs = captured_preflight.jobs
    anchor = _build_run_anchor(
        jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=captured_approval.sha256,
    )
    ledger = AppendOnlyAttemptLedger(
        Path(output_dir),
        expected_anchor=anchor,
        create_only=True,
    )
    if model_factory is None:
        model = V2LocalJsonModel(approval=approval, preflight=preflight)
    else:
        model = model_factory()
    generate_method = getattr(model, "generate", None)
    if not callable(generate_method):
        raise TypeError("proposal model factory must return a generate-capable model")
    results: list[dict[str, object]] = []
    for job in jobs:

        def generate(ceiling: int, *, frozen_job: Mapping[str, object] = job) -> object:
            return generate_method(
                frozen_job["messages"],
                PROPOSAL_SCHEMA,
                max_new_tokens=ceiling,
            )

        results.append(run_job_with_retry(job, generate=generate, ledger=ledger))
    completion = ledger.seal_completion()
    return {
        "status": "complete",
        "job_count": len(results),
        "results": results,
        "event_count": len(ledger.read_events()),
        "completion": completion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-preflight", help="freeze 48 proposal jobs")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-preflight", help="verify frozen proposal jobs")
    verify.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute", help="execute separately approved jobs")
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--approval", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser(
        "finalize", help="materialize canonical proposals from a sealed ledger"
    )
    finalize.add_argument("--preflight", type=Path, required=True)
    finalize.add_argument("--approval", type=Path, required=True)
    finalize.add_argument("--ledger", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-preflight":
        contract = _load_verified_contract(args.contract)
        if _path_present(args.output):
            raise FileExistsError(f"create-only proposal preflight exists: {args.output}")
        snapshot = _snapshot_inventory()
        tokenizer = _load_local_tokenizer()
        receipt = build_proposal_preflight(
            contract,
            tokenizer=tokenizer,
            model_factory=None,
            output_dir=args.output,
            contract_dir=args.contract,
            contract_receipt_sha256=_sha256_file(args.contract / "receipt.json"),
            model_snapshot=snapshot,
        )
    elif args.command == "verify-preflight":
        receipt = verify_proposal_preflight(args.output)
    elif args.command == "execute":
        result = execute_proposals(
            preflight_dir=args.preflight,
            approval_path=args.approval,
            output_dir=args.output,
        )
        print(_compact(result))
        return 0
    else:
        receipt = finalize_proposal_inventory(
            preflight_dir=args.preflight,
            approval_path=args.approval,
            ledger_dir=args.ledger,
            output_dir=args.output,
        )
        print(
            _compact(
                {
                    "status": "complete",
                    "proposals": receipt["proposal_count"],
                    "supported": receipt["supported_count"],
                }
            )
        )
        return 0
    print(
        _compact(
            {
                "status": "verified" if args.command == "verify-preflight" else "complete",
                "jobs": receipt["job_count"],
                "primary_calls": receipt["primary_call_count"],
                "retry_ceiling": receipt["retry_call_ceiling"],
                "worst_case_calls": receipt["worst_case_call_ceiling"],
                "model_loads": receipt["model_load_count"],
                "inference_calls": receipt["inference_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

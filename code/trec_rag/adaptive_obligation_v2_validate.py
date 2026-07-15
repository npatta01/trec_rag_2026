"""Build inference-free opposite-fold validation jobs for adaptive obligation v2."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_obligation_v2_contract import canonical_sha256, sha256_text
from .adaptive_obligation_v2_ledger import AppendOnlyAttemptLedger, AttemptSpec
from .adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    PROPOSAL_RECEIPT_SCHEMA_VERSION,
    _capture_static_preflight_for_task4,
    _compact_bytes,
    _capture_regular_file_no_symlinks,
    _fsync_directory,
    _load_verified_contract,
    _open_directory_no_symlinks,
    _pretty_bytes,
    _read_stable_regular_at,
    _sha256_file,
    _write_fsynced,
    load_authenticated_proposal_inventory,
)


VALIDATION_DECISIONS = (
    "SUPPORTED",
    "NO_EVIDENCE",
    "OUT_OF_SCOPE",
    "ANSWER_FACT",
    "DUPLICATE_O0",
    "WRONG_DOMAIN",
)

SCHEMA_VERSION = "adaptive-obligation-v2-validation-preflight-v1"
JOB_SCHEMA_VERSION = "adaptive-obligation-v2-validation-job-v1"
PRIMARY_MAX_NEW_TOKENS = 256
RETRY_MAX_NEW_TOKENS = 512
VALIDATOR_ROLE = "opposite_fold_semantic_validator"
_CONTRACT_OUTPUT_NAMES = frozenset(
    {
        "manifest.json",
        "parents.jsonl",
        "reservoirs.jsonl",
        "units.jsonl",
        "receipt.json",
    }
)

VALIDATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "support_unit_ids"],
    "properties": {
        "decision": {"enum": list(VALIDATION_DECISIONS)},
        "support_unit_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
            "uniqueItems": True,
        },
    },
}

_VALIDATION_INSTRUCTIONS = (
    "Decide whether the proposed abstract child obligation is independently "
    "supported by the supplied opposite-fold evidence. Return exactly one JSON "
    "object matching the response schema. SUPPORTED must cite one or two supplied "
    "unit IDs; every other decision must cite none."
)


def normalize_label(value: object) -> str:
    """Normalize a label for deterministic copy and duplicate checks."""

    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _validation_prompt_contract() -> dict[str, object]:
    return {
        "stage": "validation",
        "validator_role": VALIDATOR_ROLE,
        "instructions": _VALIDATION_INSTRUCTIONS,
        "job_schema_version": JOB_SCHEMA_VERSION,
        "schema_sha256": canonical_sha256(VALIDATION_SCHEMA),
    }


def _validator_model_metadata(
    proposal_preflight_dir: Path,
    *,
    expected_receipt_sha256: str,
    authenticated_preflight: object = None,
) -> dict[str, object]:
    if authenticated_preflight is None:
        captured = _capture_static_preflight_for_task4(
            Path(proposal_preflight_dir),
            expected_receipt_sha256=expected_receipt_sha256,
        )
        receipt = captured.receipt
        receipt_sha256 = captured.receipt_sha256
    elif isinstance(authenticated_preflight, Mapping):
        receipt = authenticated_preflight
        receipt_sha256 = expected_receipt_sha256
    else:
        raise ValueError("validation validator model metadata differs")
    snapshot = receipt.get("model_snapshot")
    tokenizer_files = receipt.get("tokenizer_files")
    tokenizer_contract = receipt.get("tokenizer_contract")
    if (
        receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or not isinstance(snapshot, Mapping)
        or snapshot.get("model") != MODEL_ID
        or snapshot.get("revision") != MODEL_REVISION
        or not isinstance(snapshot.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot["manifest_sha256"]))
        or not isinstance(tokenizer_files, list)
        or any(not isinstance(row, Mapping) for row in tokenizer_files)
        or not isinstance(tokenizer_contract, Mapping)
    ):
        raise ValueError("validation validator model metadata differs")
    tokenizer_identity = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": dict(tokenizer_contract),
    }
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": dict(snapshot),
        "model_snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "tokenizer_files": [dict(row) for row in tokenizer_files],
        "tokenizer_contract": dict(tokenizer_contract),
        "tokenizer_identity_sha256": canonical_sha256(tokenizer_identity),
        "proposal_preflight_receipt_sha256": receipt_sha256,
    }


def _count_validation_prompts(
    jobs: Sequence[Mapping[str, object]], tokenizer: object
) -> tuple[list[dict[str, object]], dict[str, object]]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("validation preflight tokenizer cannot count chat prompts")
    counted: list[dict[str, object]] = []
    counts: list[int] = []
    for job in jobs:
        token_ids = apply_chat_template(
            job["messages"], tokenize=True, add_generation_prompt=True
        )
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or not token_ids
        ):
            raise ValueError("validation tokenizer returned an invalid token sequence")
        count = len(token_ids)
        counts.append(count)
        counted.append({**dict(job), "prompt_token_count": count})
    summary: dict[str, object] = {
        "count": len(counts),
        "minimum": min(counts) if counts else None,
        "maximum": max(counts) if counts else None,
        "total": sum(counts),
        "by_job": [
            {
                "job_id": job["job_id"],
                "prompt_token_count": job["prompt_token_count"],
            }
            for job in counted
        ],
    }
    return counted, summary


def _compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rows(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"validation contract {name} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"validation contract {name} must contain objects")
    return list(value)  # type: ignore[return-value]


def _capture_contract_files(path: Path) -> dict[str, bytes]:
    try:
        directory_fd = _open_directory_no_symlinks(Path(path))
        try:
            before = os.fstat(directory_fd)
            names_before = set(os.listdir(directory_fd))
            if names_before != _CONTRACT_OUTPUT_NAMES:
                raise OSError("contract inventory differs")
            contents = {
                name: _read_stable_regular_at(
                    directory_fd, name, require_single_link=True
                )
                for name in sorted(_CONTRACT_OUTPUT_NAMES)
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
                raise OSError("contract changed while being captured")
            return contents
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError("validation source contract is missing or unsafe") from exc


def _capture_and_verify_contract_snapshot(
    path: Path, *, expected_receipt_sha256: str
) -> tuple[dict[str, object], str]:
    """Verify an isolated immutable copy and return only its captured rows."""

    contents = _capture_contract_files(Path(path))
    receipt_source = contents["receipt.json"]
    receipt_sha256 = hashlib.sha256(receipt_source).hexdigest()
    if receipt_sha256 != expected_receipt_sha256:
        raise ValueError("validation source contract receipt hash differs")
    try:
        receipt = json.loads(receipt_source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation source contract receipt is invalid") from exc
    if not isinstance(receipt, dict) or receipt_source != _pretty_bytes(receipt):
        raise ValueError("validation source contract receipt is not canonical")
    raw_topics = receipt.get("topic_ids")
    if not isinstance(raw_topics, list):
        raise ValueError("validation source contract topics are missing")
    for topic_id in map(str, raw_topics):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    with tempfile.TemporaryDirectory(prefix="adaptive-v2-contract-") as temporary:
        private_root = Path(temporary) / "snapshot"
        private_root.mkdir(mode=0o700)
        try:
            for name in sorted(_CONTRACT_OUTPUT_NAMES):
                _write_fsynced(private_root / name, contents[name])
                os.chmod(private_root / name, 0o400)
            _fsync_directory(private_root)
            os.chmod(private_root, 0o500)
            contract = _load_verified_contract(private_root)
        finally:
            os.chmod(private_root, 0o700)
    if contract.get("receipt") != receipt:
        raise ValueError("validation verified contract receipt differs")
    expected_rows = {
        "parents": contents["parents.jsonl"],
        "reservoirs": contents["reservoirs.jsonl"],
        "units": contents["units.jsonl"],
    }
    for name, source in expected_rows.items():
        rows = contract.get(name)
        if (
            not isinstance(rows, list)
            or source != b"".join(_compact_bytes(row) for row in rows)
        ):
            raise ValueError("validation contract rows differ from captured bytes")
    return contract, receipt_sha256


def _valid_unit(unit: Mapping[str, object]) -> None:
    text = unit.get("text")
    if not isinstance(text, str) or unit.get("text_sha256") != sha256_text(text):
        raise ValueError("validation source unit hash differs")
    fold = unit.get("fold")
    if isinstance(fold, bool) or fold not in (0, 1):
        raise ValueError("validation source unit identity differs")
    required = (
        "topic_id",
        "parent_id",
        "fold",
        "document_id",
        "window_id",
        "start",
        "end",
        "text",
    )
    if any(name not in unit for name in required):
        raise ValueError("validation source unit identity differs")
    identity = {name: unit[name] for name in required}
    if unit.get("unit_id") != canonical_sha256(identity):
        raise ValueError("validation source unit identity differs")


def _contract_indexes(
    contract: object,
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, Mapping[str, object]],
    dict[tuple[str, int], Mapping[str, object]],
]:
    if not isinstance(contract, Mapping):
        raise ValueError("validation contract must be an object")
    parents = _rows(contract.get("parents"), "parents")
    units = _rows(contract.get("units"), "units")
    reservoirs = _rows(contract.get("reservoirs"), "reservoirs")
    parent_by_id = {str(row.get("parent_id")): row for row in parents}
    unit_by_id = {str(row.get("unit_id")): row for row in units}
    if len(parent_by_id) != len(parents) or len(unit_by_id) != len(units):
        raise ValueError("validation contract contains duplicate source identities")
    reservoir_by_pair: dict[tuple[str, int], Mapping[str, object]] = {}
    for reservoir in reservoirs:
        fold = reservoir.get("fold")
        if isinstance(fold, bool) or fold not in (0, 1):
            raise ValueError("validation reservoir fold differs")
        key = (str(reservoir.get("parent_id")), int(fold))
        if key in reservoir_by_pair:
            raise ValueError("validation contract contains duplicate reservoirs")
        reservoir_by_pair[key] = reservoir
    return parent_by_id, unit_by_id, reservoir_by_pair


def validate_proposal_record(
    proposal: Mapping[str, object], contract: object
) -> dict[str, object]:
    """Fail closed unless one supported proposal binds to exact proposing units."""

    if not isinstance(proposal, Mapping):
        raise ValueError("validation proposal must be an object")
    topic_id = str(proposal.get("topic_id"))
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    if topic_id not in PILOT_TOPIC_IDS:
        raise ValueError("validation proposal topic is unknown")
    if (
        proposal.get("status") != "SUPPORTED"
        or proposal.get("reason_code") != "SUPPORTED"
    ):
        raise ValueError("validation proposal status differs")
    proposal_fold = proposal.get("proposal_fold", proposal.get("fold"))
    if isinstance(proposal_fold, bool) or proposal_fold not in (0, 1):
        raise ValueError("validation proposal fold differs")
    parent_id = proposal.get("parent_id")
    proposal_id = proposal.get("proposal_id")
    label = proposal.get("label")
    support_ids = proposal.get("support_unit_ids")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("validation proposal parent identity differs")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("validation proposal identity differs")
    if not isinstance(label, str) or not 3 <= len(label) <= 120:
        raise ValueError("validation proposal label differs")
    if (
        not isinstance(support_ids, list)
        or not 1 <= len(support_ids) <= 2
        or any(not isinstance(value, str) for value in support_ids)
        or len(set(support_ids)) != len(support_ids)
    ):
        raise ValueError("validation proposal support inventory differs")

    parent_by_id, unit_by_id, reservoir_by_pair = _contract_indexes(contract)
    parent = parent_by_id.get(parent_id)
    if parent is None:
        raise ValueError("validation proposal parent is unknown")
    if parent.get("topic_id") != topic_id:
        raise ValueError("validation proposal source identity differs")
    parent_text = parent.get("text")
    if (
        not isinstance(parent_text, str)
        or parent.get("text_sha256") != sha256_text(parent_text)
    ):
        raise ValueError("validation parent source hash differs")
    query = parent.get("query")
    if not isinstance(query, str) or parent.get("query_sha256") != sha256_text(query):
        raise ValueError("validation parent source hash differs")
    manifest_order = parent.get("manifest_order")
    if (
        isinstance(manifest_order, bool)
        or not isinstance(manifest_order, int)
        or proposal.get("parent_manifest_order", manifest_order) != manifest_order
    ):
        raise ValueError("validation proposal parent identity differs")

    supporting_units: list[Mapping[str, object]] = []
    for unit_id in support_ids:
        unit = unit_by_id.get(unit_id)
        if unit is None:
            raise ValueError("validation proposal cites an unknown unit")
        _valid_unit(unit)
        if (
            unit.get("topic_id") != topic_id
            or unit.get("parent_id") != parent_id
            or unit.get("fold") != proposal_fold
        ):
            raise ValueError("validation proposal source identity differs")
        supporting_units.append(unit)

    source_reservoir = reservoir_by_pair.get((parent_id, int(proposal_fold)))
    if source_reservoir is None:
        raise ValueError("validation proposal source-fold reservoir is missing")
    source_units = _opposite_units(
        parent=parent,
        validation_fold=int(proposal_fold),
        unit_by_id=unit_by_id,
        reservoir=source_reservoir,
    )
    source_unit_ids = {str(unit["unit_id"]) for unit in source_units}
    if not set(support_ids) <= source_unit_ids:
        raise ValueError("validation proposal unit escapes its source-fold reservoir")

    normalized = normalize_label(label)
    if not normalized:
        raise ValueError("validation proposal label is empty")
    if any(normalized == normalize_label(unit["text"]) for unit in supporting_units):
        raise ValueError("validation proposal label copied a support passage")
    same_topic_o0 = (
        row
        for row in parent_by_id.values()
        if row.get("topic_id") == topic_id and isinstance(row.get("text"), str)
    )
    if any(normalized == normalize_label(row["text"]) for row in same_topic_o0):
        raise ValueError("validation proposal label is a duplicate O0")
    return {
        "proposal_id": proposal_id,
        "topic_id": topic_id,
        "parent_id": parent_id,
        "parent_manifest_order": manifest_order,
        "proposal_fold": int(proposal_fold),
        "label": label,
        "normalized_label": normalized,
        "proposal_document_ids": list(
            dict.fromkeys(str(unit["document_id"]) for unit in supporting_units)
        ),
        "proposal_support_unit_ids": list(support_ids),
    }


def render_validation_messages(
    parent: Mapping[str, object],
    proposal: Mapping[str, object],
    units: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Render only narrative, complete O0, label, and opposite-fold units."""

    text = parent.get("text")
    query = parent.get("query")
    if not isinstance(text, str) or not isinstance(query, str):
        raise ValueError("validation parent narrative or O0 differs")
    suffix = f"\n\nExplicit obligation:\n{text}"
    if not query.endswith(suffix):
        raise ValueError("validation parent query does not preserve complete O0")
    payload = {
        "narrative": query[: -len(suffix)],
        "parent_o0": {"parent_id": parent["parent_id"], "text": text},
        "proposed_o1_label": proposal["label"],
        "opposite_fold_units": [dict(unit) for unit in units],
        "response_json_schema": VALIDATION_SCHEMA,
    }
    return [
        {"role": "system", "content": _VALIDATION_INSTRUCTIONS},
        {"role": "user", "content": _compact(payload)},
    ]


def _opposite_units(
    *,
    parent: Mapping[str, object],
    validation_fold: int,
    unit_by_id: Mapping[str, Mapping[str, object]],
    reservoir: Mapping[str, object],
) -> list[Mapping[str, object]]:
    documents = reservoir.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("validation opposite-fold reservoir is empty")
    ordered: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping) or not isinstance(
            document.get("unit_ids"), list
        ):
            raise ValueError("validation reservoir document inventory differs")
        for unit_id in document["unit_ids"]:
            if not isinstance(unit_id, str) or unit_id in seen:
                raise ValueError("validation reservoir unit identity is duplicate")
            unit = unit_by_id.get(unit_id)
            if unit is None:
                raise ValueError("validation reservoir contains an unknown unit")
            _valid_unit(unit)
            if (
                unit.get("topic_id") != parent.get("topic_id")
                or unit.get("parent_id") != parent.get("parent_id")
                or unit.get("fold") != validation_fold
                or unit.get("document_id") != document.get("document_id")
                or unit.get("window_id") != document.get("window_id")
            ):
                raise ValueError("validation opposite-fold source identity differs")
            seen.add(unit_id)
            ordered.append(unit)
    return ordered


def build_validation_jobs(
    proposals: Sequence[Mapping[str, object]], contract: object
) -> list[dict[str, object]]:
    """Build one compact opposite-fold job per supported, valid proposal."""

    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ValueError("validation proposals must be an array")
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise ValueError("validation proposals must contain objects")
        topic_id = str(proposal.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    parent_by_id, unit_by_id, reservoir_by_pair = _contract_indexes(contract)
    jobs: list[dict[str, object]] = []
    seen_proposal_ids: set[str] = set()
    seen_parent_folds: set[tuple[str, int]] = set()
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise ValueError("validation proposals must contain objects")
        if proposal.get("status") == "UNSUPPORTED":
            continue
        valid = validate_proposal_record(proposal, contract)
        proposal_id = str(valid["proposal_id"])
        if proposal_id in seen_proposal_ids:
            raise ValueError("validation proposal identity is duplicate")
        seen_proposal_ids.add(proposal_id)
        parent_id = str(valid["parent_id"])
        parent_fold = (parent_id, int(valid["proposal_fold"]))
        if parent_fold in seen_parent_folds:
            raise ValueError("validation proposal parent/fold identity is duplicate")
        seen_parent_folds.add(parent_fold)
        parent = parent_by_id[parent_id]
        validation_fold = 1 - int(valid["proposal_fold"])
        reservoir = reservoir_by_pair.get((parent_id, validation_fold))
        if reservoir is None:
            raise ValueError("validation opposite-fold reservoir is missing")
        if reservoir.get("topic_id") != valid["topic_id"]:
            raise ValueError("validation opposite-fold reservoir identity differs")
        units = _opposite_units(
            parent=parent,
            validation_fold=validation_fold,
            unit_by_id=unit_by_id,
            reservoir=reservoir,
        )
        messages = render_validation_messages(parent, valid, units)
        identity = {
            "proposal_id": proposal_id,
            "topic_id": valid["topic_id"],
            "parent_id": parent_id,
            "proposal_fold": valid["proposal_fold"],
            "validation_fold": validation_fold,
            "input_unit_ids": [unit["unit_id"] for unit in units],
            "messages_sha256": canonical_sha256(messages),
            "schema_sha256": canonical_sha256(VALIDATION_SCHEMA),
        }
        jobs.append(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": canonical_sha256(identity),
                **identity,
                "parent_manifest_order": valid["parent_manifest_order"],
                "label": valid["label"],
                "proposal_document_ids": valid["proposal_document_ids"],
                "units": [dict(unit) for unit in units],
                "messages": messages,
                "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
                "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
            }
        )
    if len(jobs) > 48:
        raise ValueError("validation job count exceeds the 48-proposal ceiling")
    return jobs


def _rejected_semantic(
    proposal: Mapping[str, object], decision: object, *reasons: str
) -> dict[str, object]:
    return {
        "proposal_id": proposal.get("proposal_id"),
        "topic_id": proposal.get("topic_id"),
        "parent_id": proposal.get("parent_id"),
        "parent_manifest_order": proposal.get("parent_manifest_order"),
        "label": proposal.get("label"),
        "decision": decision,
        "accepted": False,
        "reasons": list(reasons),
    }


def _proposal_documents(proposal: Mapping[str, object]) -> list[str]:
    values = proposal.get("proposal_document_ids")
    if values is None:
        values = proposal.get("support_document_ids")
    if values is None and isinstance(proposal.get("document_id"), str):
        values = [proposal["document_id"]]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        return []
    return list(dict.fromkeys(values))


def validate_semantic_decision(
    proposal: Mapping[str, object],
    decision: object,
    *,
    units: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind a finite validator decision to exact opposite-fold source units."""

    if not isinstance(proposal, Mapping):
        raise ValueError("semantic validation proposal must be an object")
    topic_id = str(proposal.get("topic_id"))
    if topic_id in PROTECTED_TOPIC_IDS:
        return _rejected_semantic(proposal, None, "protected_topic")
    if topic_id not in PILOT_TOPIC_IDS:
        return _rejected_semantic(proposal, None, "unknown_topic")
    if not isinstance(decision, Mapping) or set(decision) != {
        "decision",
        "support_unit_ids",
    }:
        return _rejected_semantic(proposal, None, "invalid_decision_schema")
    code = decision.get("decision")
    support_ids = decision.get("support_unit_ids")
    if code not in VALIDATION_DECISIONS:
        return _rejected_semantic(proposal, code, "invalid_decision")
    if not isinstance(support_ids, list) or any(
        not isinstance(value, str) for value in support_ids
    ):
        return _rejected_semantic(proposal, code, "invalid_support_inventory")
    if code != "SUPPORTED":
        if support_ids:
            return _rejected_semantic(
                proposal, code, "invalid_support_inventory"
            )
        return _rejected_semantic(proposal, code, str(code).casefold())
    if not 1 <= len(support_ids) <= 2 or len(set(support_ids)) != len(support_ids):
        return _rejected_semantic(proposal, code, "invalid_support_inventory")
    proposal_fold = proposal.get("proposal_fold", proposal.get("fold"))
    if isinstance(proposal_fold, bool) or proposal_fold not in (0, 1):
        return _rejected_semantic(proposal, code, "source_identity")
    proposal_documents = _proposal_documents(proposal)
    if not proposal_documents:
        return _rejected_semantic(proposal, code, "source_identity")

    validation_units: list[Mapping[str, object]] = []
    reasons: list[str] = []
    for unit_id in support_ids:
        unit = units.get(unit_id)
        if unit is None:
            reasons.append("unknown_unit")
            continue
        try:
            _valid_unit(unit)
        except ValueError:
            reasons.append("source_identity")
            continue
        if unit.get("fold") == proposal_fold:
            reasons.append("opposite_fold")
        if (
            unit.get("topic_id") != topic_id
            or unit.get("parent_id") != proposal.get("parent_id")
        ):
            reasons.append("source_identity")
        validation_units.append(unit)
    validation_documents = [
        str(unit["document_id"]) for unit in validation_units
    ]
    if set(proposal_documents) & set(validation_documents):
        reasons.append("distinct_document")
    if reasons:
        return _rejected_semantic(proposal, code, *dict.fromkeys(reasons))
    support_documents = list(
        dict.fromkeys([*proposal_documents, *validation_documents])
    )
    return {
        "proposal_id": proposal.get("proposal_id"),
        "topic_id": topic_id,
        "parent_id": proposal.get("parent_id"),
        "parent_manifest_order": proposal.get("parent_manifest_order"),
        "proposal_fold": int(proposal_fold),
        "validation_fold": 1 - int(proposal_fold),
        "label": proposal.get("label"),
        "decision": code,
        "proposal_document_ids": proposal_documents,
        "validation_support_unit_ids": list(support_ids),
        "validation_document_ids": validation_documents,
        "support_document_ids": support_documents,
        "accepted": True,
        "reasons": [],
    }


def _document_count(row: Mapping[str, object]) -> int:
    values = row.get("support_document_ids")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("validated O1 support documents differ")
    return len(set(values))


def acceptance_key(row: Mapping[str, object]) -> tuple[int, int, str, str]:
    """Order one-per-parent winners for the four-per-topic cap."""

    manifest_order = row.get("parent_manifest_order")
    if isinstance(manifest_order, bool) or not isinstance(manifest_order, int):
        raise ValueError("validated O1 parent manifest order differs")
    proposal_id = row.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("validated O1 proposal identity differs")
    return (
        -_document_count(row),
        manifest_order,
        normalize_label(row.get("label")),
        proposal_id,
    )


def _parent_choice_key(row: Mapping[str, object]) -> tuple[int, int, str, str]:
    normalized = normalize_label(row.get("label"))
    return (
        -_document_count(row),
        len(normalized),
        normalized,
        str(row.get("proposal_id")),
    )


def accept_validated_o1(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Deterministically accept at most one O1 per parent and four per topic."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("validated O1 rows must be an array")
    eligible: list[Mapping[str, object]] = []
    seen_proposal_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("validated O1 rows must contain objects")
        if row.get("accepted") is not True or row.get("decision") != "SUPPORTED":
            continue
        proposal_id = row.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("validated O1 proposal identity differs")
        if proposal_id in seen_proposal_ids:
            raise ValueError("validated O1 contains a duplicate proposal ID")
        seen_proposal_ids.add(proposal_id)
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in PILOT_TOPIC_IDS:
            raise ValueError("validated O1 topic is unknown")
        if not isinstance(row.get("parent_id"), str) or not normalize_label(
            row.get("label")
        ):
            raise ValueError("validated O1 identity or label differs")
        acceptance_key(row)
        eligible.append(row)

    one_per_parent: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in eligible:
        key = (str(row["topic_id"]), str(row["parent_id"]))
        current = one_per_parent.get(key)
        if current is None or _parent_choice_key(row) < _parent_choice_key(current):
            one_per_parent[key] = row

    accepted: list[dict[str, object]] = []
    for topic_id in PILOT_TOPIC_IDS:
        topic_rows = [
            row
            for (row_topic, _parent_id), row in one_per_parent.items()
            if row_topic == topic_id
        ]
        accepted.extend(dict(row) for row in sorted(topic_rows, key=acceptance_key)[:4])
    return accepted


def _load_pinned_validation_tokenizer(
    model_metadata: Mapping[str, object],
) -> object:
    snapshot = model_metadata.get("model_snapshot")
    tokenizer_files = model_metadata.get("tokenizer_files")
    tokenizer_contract = model_metadata.get("tokenizer_contract")
    tokenizer_identity = {
        "model": model_metadata.get("model"),
        "model_revision": model_metadata.get("model_revision"),
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": tokenizer_contract,
    }
    if (
        model_metadata.get("model") != MODEL_ID
        or model_metadata.get("model_revision") != MODEL_REVISION
        or not isinstance(snapshot, Mapping)
        or snapshot.get("model") != MODEL_ID
        or snapshot.get("revision") != MODEL_REVISION
        or snapshot.get("manifest_sha256")
        != model_metadata.get("model_snapshot_manifest_sha256")
        or not isinstance(tokenizer_files, list)
        or any(not isinstance(row, Mapping) for row in tokenizer_files)
        or not isinstance(tokenizer_contract, Mapping)
        or canonical_sha256(tokenizer_identity)
        != model_metadata.get("tokenizer_identity_sha256")
    ):
        raise ValueError("validation tokenizer metadata differs before load")
    from .adaptive_obligation_v2_local_model import load_pinned_local_tokenizer

    return load_pinned_local_tokenizer(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        model_manifest=snapshot,
    )


def _build_validation_preflight(
    contract_dir: object,
    *,
    proposal_inventory_dir: object,
    proposal_preflight_dir: object,
    proposal_ledger_dir: object,
    tokenizer: object | None,
) -> dict[str, object]:
    """Private authenticated builder with an optional post-auth test tokenizer."""

    authenticated = load_authenticated_proposal_inventory(
        output_dir=proposal_inventory_dir,  # type: ignore[arg-type]
        preflight_dir=proposal_preflight_dir,  # type: ignore[arg-type]
        ledger_dir=proposal_ledger_dir,  # type: ignore[arg-type]
    )
    proposals = authenticated.get("proposals")
    proposal_receipt = authenticated.get("receipt")
    proposal_receipt_sha256 = authenticated.get("receipt_sha256")
    if (
        not isinstance(proposals, list)
        or any(not isinstance(row, Mapping) for row in proposals)
        or not isinstance(proposal_receipt, Mapping)
        or not isinstance(proposal_receipt_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", proposal_receipt_sha256)
        or proposal_receipt.get("schema_version")
        != PROPOSAL_RECEIPT_SCHEMA_VERSION
        or proposal_receipt.get("status") != "complete"
    ):
        raise ValueError("validation requires an authenticated proposal receipt")
    completion = proposal_receipt.get("completion")
    if (
        proposal_receipt.get("proposal_count") != len(proposals)
        or proposal_receipt.get("job_count") != len(proposals)
        or not isinstance(completion, Mapping)
        or completion.get("completed_job_count") != len(proposals)
        or proposal_receipt.get("run_anchor_sha256")
        != completion.get("anchor_sha256")
    ):
        raise ValueError("validation proposal receipt binding differs")

    for proposal in proposals:
        topic_id = str(proposal.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    source_contract_receipt_sha256 = proposal_receipt.get(
        "source_contract_receipt_sha256"
    )
    if (
        not isinstance(source_contract_receipt_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_contract_receipt_sha256)
    ):
        raise ValueError("validation proposal source contract binding differs")
    try:
        contract_path = Path(contract_dir)
    except TypeError as exc:
        raise ValueError(
            "validation source contract path must be a filesystem path"
        ) from exc
    contract, observed_contract_receipt_sha256 = (
        _capture_and_verify_contract_snapshot(
            contract_path,
            expected_receipt_sha256=source_contract_receipt_sha256,
        )
    )
    if observed_contract_receipt_sha256 != source_contract_receipt_sha256:
        raise ValueError("validation source contract receipt hash differs")

    model_metadata = _validator_model_metadata(
        Path(proposal_preflight_dir),
        expected_receipt_sha256=str(
            proposal_receipt["proposal_preflight_receipt_sha256"]
        ),
        authenticated_preflight=authenticated.get("proposal_preflight"),
    )
    jobs = build_validation_jobs(proposals, contract)
    if jobs:
        if tokenizer is None:
            tokenizer = _load_pinned_validation_tokenizer(model_metadata)
        jobs, prompt_token_counts = _count_validation_prompts(jobs, tokenizer)
    else:
        prompt_token_counts = {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "total": 0,
            "by_job": [],
        }
    job_count = len(jobs)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stage": "validation",
        "validator_role": VALIDATOR_ROLE,
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": job_count,
        "primary_call_count": job_count,
        "retry_call_ceiling": job_count,
        "worst_case_call_ceiling": job_count * 2,
        "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
        "schema_sha256": canonical_sha256(VALIDATION_SCHEMA),
        "prompt_sha256": canonical_sha256(_validation_prompt_contract()),
        "code_sha256": {
            "adaptive_obligation_v2_validate.py": _sha256_file(Path(__file__)),
            "adaptive_obligation_v2_local_model.py": _sha256_file(
                Path(__file__).with_name("adaptive_obligation_v2_local_model.py")
            ),
        },
        "jobs_sha256": canonical_sha256(jobs),
        "jobs": jobs,
        "prompt_token_counts": prompt_token_counts,
        **model_metadata,
        "source_contract_receipt_sha256": observed_contract_receipt_sha256,
        "proposal_receipt": {
            "schema_version": proposal_receipt["schema_version"],
            "status": proposal_receipt["status"],
            "sha256": proposal_receipt_sha256,
            "proposal_count": proposal_receipt["proposal_count"],
            "proposals_sha256": proposal_receipt["proposals_sha256"],
            "proposal_preflight_receipt_sha256": proposal_receipt[
                "proposal_preflight_receipt_sha256"
            ],
            "source_contract_receipt_sha256": source_contract_receipt_sha256,
            "run_anchor_sha256": proposal_receipt["run_anchor_sha256"],
            "completion_sha256": proposal_receipt["completion_sha256"],
        },
        "expected_runtime": {
            "phase": "inference_free_preflight",
            "proposal_calls_executed": 0,
            "validation_calls_executed": 0,
        },
        "model_construction_allowed": False,
        "generation_allowed": False,
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "model_load_count": 0,
        "tokenizer_load_count": 1 if jobs else 0,
        "inference_count": 0,
        "external_cost_usd": 0.0,
    }


def _build_validation_preflight_with_tokenizer(
    contract_dir: object,
    *,
    proposal_inventory_dir: object,
    proposal_preflight_dir: object,
    proposal_ledger_dir: object,
    tokenizer: object,
) -> dict[str, object]:
    """Private post-auth seam for tests that must not load the real tokenizer."""

    return _build_validation_preflight(
        contract_dir,
        proposal_inventory_dir=proposal_inventory_dir,
        proposal_preflight_dir=proposal_preflight_dir,
        proposal_ledger_dir=proposal_ledger_dir,
        tokenizer=tokenizer,
    )


def build_validation_preflight(
    contract_dir: object,
    *,
    proposal_inventory_dir: object,
    proposal_preflight_dir: object,
    proposal_ledger_dir: object,
) -> dict[str, object]:
    """Authenticate metadata, then load one pinned tokenizer and plan V/V/2V."""

    return _build_validation_preflight(
        contract_dir,
        proposal_inventory_dir=proposal_inventory_dir,
        proposal_preflight_dir=proposal_preflight_dir,
        proposal_ledger_dir=proposal_ledger_dir,
        tokenizer=None,
    )


def _validation_request(
    job: Mapping[str, object], ceiling: int
) -> dict[str, object]:
    return {
        "messages": job["messages"],
        "schema": VALIDATION_SCHEMA,
        "max_new_tokens": ceiling,
    }


def _validation_anchor_provenance(
    preflight: Mapping[str, object],
) -> dict[str, object]:
    code_sha256 = preflight.get("code_sha256")
    provenance = {
        "validator_role": preflight.get("validator_role"),
        "model": preflight.get("model"),
        "model_revision": preflight.get("model_revision"),
        "model_snapshot_manifest_sha256": preflight.get(
            "model_snapshot_manifest_sha256"
        ),
        "tokenizer_identity_sha256": preflight.get(
            "tokenizer_identity_sha256"
        ),
        "prompt_sha256": preflight.get("prompt_sha256"),
        "code_identity_sha256": (
            canonical_sha256(code_sha256)
            if isinstance(code_sha256, Mapping)
            else None
        ),
        "jobs_sha256": preflight.get("jobs_sha256"),
        "schema_sha256": preflight.get("schema_sha256"),
    }
    if (
        provenance["validator_role"] != VALIDATOR_ROLE
        or provenance["model"] != MODEL_ID
        or provenance["model_revision"] != MODEL_REVISION
        or any(
            not isinstance(provenance[name], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance[name]))
            for name in (
                "model_snapshot_manifest_sha256",
                "tokenizer_identity_sha256",
                "prompt_sha256",
                "code_identity_sha256",
                "jobs_sha256",
                "schema_sha256",
            )
        )
        or provenance["schema_sha256"] != canonical_sha256(VALIDATION_SCHEMA)
    ):
        raise ValueError("validation run provenance differs")
    return provenance


def _build_validation_run_anchor(
    jobs: Sequence[Mapping[str, object]],
    *,
    preflight_sha256: str,
    approval_sha256: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Bind the real append-only ledger to exact validation jobs and schema."""

    if (
        not re.fullmatch(r"[0-9a-f]{64}", preflight_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", approval_sha256)
    ):
        raise ValueError("validation run approval binding differs")
    anchored_jobs: list[dict[str, object]] = []
    seen: set[str] = set()
    for job in jobs:
        job_id = job.get("job_id")
        unit_ids = job.get("input_unit_ids")
        if (
            not isinstance(job_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", job_id)
            or job_id in seen
            or not isinstance(unit_ids, list)
            or len(set(unit_ids)) != len(unit_ids)
            or any(
                not isinstance(unit_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", unit_id)
                for unit_id in unit_ids
            )
        ):
            raise ValueError("validation run job inventory differs")
        seen.add(job_id)
        attempts = [
            {
                "attempt_ordinal": ordinal,
                "request_sha256": canonical_sha256(
                    _validation_request(job, ceiling)
                ),
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
        "stage": "validation",
        "preflight_sha256": preflight_sha256,
        "approval_sha256": approval_sha256,
        "job_count": len(anchored_jobs),
        "job_inventory_sha256": canonical_sha256(anchored_jobs),
        "jobs": anchored_jobs,
        "schema": VALIDATION_SCHEMA,
        "provenance": dict(provenance),
    }


def _validate_decision_value(
    value: object, allowed_support_unit_ids: Sequence[str]
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "decision",
        "support_unit_ids",
    }:
        raise ValueError("validation schema_error")
    decision = value.get("decision")
    support_ids = value.get("support_unit_ids")
    if decision not in VALIDATION_DECISIONS or not isinstance(support_ids, list):
        raise ValueError("validation schema_error")
    if any(not isinstance(unit_id, str) for unit_id in support_ids) or len(
        set(support_ids)
    ) != len(support_ids):
        raise ValueError("validation schema_error")
    if decision == "SUPPORTED":
        if not 1 <= len(support_ids) <= 2:
            raise ValueError("validation schema_error")
        if not set(support_ids) <= set(allowed_support_unit_ids):
            raise ValueError("validation semantic_error: support unit escapes job")
    elif support_ids:
        raise ValueError("validation schema_error")
    return value


def run_validation_job_with_retry(
    job: Mapping[str, object],
    *,
    ledger: object | None,
    model: object | None,
) -> dict[str, object]:
    """Run only through injected test doubles; never construct a runtime or ledger."""

    if ledger is None or model is None:
        raise ValueError("validation runner requires an injected ledger and model")
    run_attempt = getattr(ledger, "run_attempt", None)
    generate_method = getattr(model, "generate", None)
    if not callable(run_attempt) or not callable(generate_method):
        raise TypeError("validation runner injected ledger and model differ")
    if (
        job.get("primary_max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
        or job.get("retry_max_new_tokens") != RETRY_MAX_NEW_TOKENS
        or not isinstance(job.get("job_id"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(job["job_id"]))
        or not isinstance(job.get("messages"), list)
        or not isinstance(job.get("input_unit_ids"), list)
        or not isinstance(job.get("units"), list)
    ):
        raise ValueError("validation execution job differs from frozen contract")
    input_unit_ids = [str(value) for value in job["input_unit_ids"]]  # type: ignore[union-attr]
    unit_rows = job["units"]
    if any(not isinstance(row, Mapping) for row in unit_rows):  # type: ignore[union-attr]
        raise ValueError("validation execution units differ")
    unit_by_id = {str(row["unit_id"]): row for row in unit_rows}  # type: ignore[union-attr]
    if list(unit_by_id) != input_unit_ids:
        raise ValueError("validation execution unit inventory differs")

    for attempt_ordinal, ceiling in enumerate(
        (PRIMARY_MAX_NEW_TOKENS, RETRY_MAX_NEW_TOKENS), start=1
    ):
        request = _validation_request(job, ceiling)
        spec = AttemptSpec(
            stage="validation",
            job_id=str(job["job_id"]),
            attempt_ordinal=attempt_ordinal,
            request_sha256=canonical_sha256(request),
            max_new_tokens=ceiling,
        )

        def invoke(_messages: object, _schema: object, value: int) -> object:
            if value != ceiling:
                raise ValueError("validation ledger output ceiling differs")
            return generate_method(
                job["messages"], VALIDATION_SCHEMA, max_new_tokens=value
            )

        classified = run_attempt(
            spec,
            invoke,
            messages=job["messages"],
            schema=VALIDATION_SCHEMA,
            allowed_support_unit_ids=input_unit_ids,
        )
        if not isinstance(classified, Mapping):
            raise ValueError("validation ledger classification differs")
        classification = classified.get("classification")
        output_token_count = classified.get("output_token_count")
        if (
            isinstance(output_token_count, bool)
            or not isinstance(output_token_count, int)
            or not 0 <= output_token_count <= ceiling
        ):
            raise ValueError("validation ledger token count differs")
        if classification == "valid":
            decision = _validate_decision_value(
                classified.get("value"), input_unit_ids
            )
            return validate_semantic_decision(job, decision, units=unit_by_id)
        if classification == "truncated_at_ceiling" and attempt_ordinal == 1:
            if output_token_count != ceiling:
                raise ValueError("validation ledger token count did not reach ceiling")
            continue
        error = classified.get("error", classification)
        if classification == "truncated_at_ceiling":
            raise ValueError(f"validation truncated at retry ceiling: {error}")
        raise ValueError(f"validation {error}")
    raise RuntimeError("validation retry policy exhausted unexpectedly")


VALIDATION_INVENTORY_SCHEMA_VERSION = (
    "adaptive-obligation-v2-validation-inventory-v1"
)
VALIDATION_RECEIPT_SCHEMA_VERSION = "adaptive-obligation-v2-validation-receipt-v1"
VALIDATION_APPROVAL_SCHEMA_VERSION = "adaptive-obligation-v2-validation-approval-v1"
_VALIDATION_INVENTORY_NAMES = frozenset(
    {"validated.jsonl", "accepted.jsonl", "receipt.json"}
)
_VALIDATION_PREFLIGHT_NAMES = frozenset({"receipt.json"})


def _verify_validation_preflight_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("validation preflight receipt must be an object")
    expected_names = {
        "schema_version",
        "status",
        "stage",
        "validator_role",
        "topic_ids",
        "job_count",
        "primary_call_count",
        "retry_call_ceiling",
        "worst_case_call_ceiling",
        "primary_max_new_tokens",
        "retry_max_new_tokens",
        "schema_sha256",
        "prompt_sha256",
        "code_sha256",
        "jobs_sha256",
        "jobs",
        "prompt_token_counts",
        "model",
        "model_revision",
        "model_snapshot",
        "model_snapshot_manifest_sha256",
        "tokenizer_files",
        "tokenizer_contract",
        "tokenizer_identity_sha256",
        "proposal_preflight_receipt_sha256",
        "source_contract_receipt_sha256",
        "proposal_receipt",
        "expected_runtime",
        "model_construction_allowed",
        "generation_allowed",
        "qrels_opened",
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
        "model_load_count",
        "tokenizer_load_count",
        "inference_count",
        "external_cost_usd",
    }
    jobs = value.get("jobs")
    if (
        set(value) != expected_names
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("stage") != "validation"
        or value.get("validator_role") != VALIDATOR_ROLE
        or value.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or not isinstance(jobs, list)
        or any(not isinstance(job, Mapping) for job in jobs)
        or type(value.get("job_count")) is not int
        or value.get("job_count") != len(jobs)
        or value.get("primary_call_count") != len(jobs)
        or value.get("retry_call_ceiling") != len(jobs)
        or value.get("worst_case_call_ceiling") != len(jobs) * 2
        or value.get("primary_max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
        or value.get("retry_max_new_tokens") != RETRY_MAX_NEW_TOKENS
        or value.get("schema_sha256") != canonical_sha256(VALIDATION_SCHEMA)
        or value.get("prompt_sha256")
        != canonical_sha256(_validation_prompt_contract())
        or value.get("jobs_sha256") != canonical_sha256(jobs)
        or not isinstance(value.get("source_contract_receipt_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(value["source_contract_receipt_sha256"])
        )
        or value.get("model_construction_allowed") is not False
        or value.get("generation_allowed") is not False
        or value.get("qrels_opened") is not False
        or value.get("external_cost_usd") != 0.0
        or any(
            value.get(name) != 0
            for name in (
                "network_call_count",
                "retrieval_call_count",
                    "hosted_inference_call_count",
                    "paid_call_count",
                    "model_load_count",
                    "inference_count",
                )
            )
        or value.get("tokenizer_load_count") != (1 if jobs else 0)
    ):
        raise ValueError("validation preflight receipt differs")
    snapshot = value.get("model_snapshot")
    tokenizer_files = value.get("tokenizer_files")
    tokenizer_contract = value.get("tokenizer_contract")
    tokenizer_identity = {
        "model": value.get("model"),
        "model_revision": value.get("model_revision"),
        "tokenizer_files": tokenizer_files,
        "tokenizer_contract": tokenizer_contract,
    }
    code_sha256 = value.get("code_sha256")
    expected_code = {
        "adaptive_obligation_v2_validate.py": _sha256_file(Path(__file__)),
        "adaptive_obligation_v2_local_model.py": _sha256_file(
            Path(__file__).with_name("adaptive_obligation_v2_local_model.py")
        ),
    }
    if (
        value.get("model") != MODEL_ID
        or value.get("model_revision") != MODEL_REVISION
        or not isinstance(snapshot, Mapping)
        or snapshot.get("model") != MODEL_ID
        or snapshot.get("revision") != MODEL_REVISION
        or snapshot.get("manifest_sha256")
        != value.get("model_snapshot_manifest_sha256")
        or not isinstance(value.get("model_snapshot_manifest_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(value["model_snapshot_manifest_sha256"])
        )
        or not isinstance(tokenizer_files, list)
        or any(not isinstance(row, Mapping) for row in tokenizer_files)
        or not isinstance(tokenizer_contract, Mapping)
        or value.get("tokenizer_identity_sha256")
        != canonical_sha256(tokenizer_identity)
        or code_sha256 != expected_code
    ):
        raise ValueError("validation validator model provenance differs")
    prompt_counts = value.get("prompt_token_counts")
    observed_counts = [job.get("prompt_token_count") for job in jobs]
    if any(type(count) is not int or count < 1 for count in observed_counts):
        raise ValueError("validation prompt token counts differ")
    integer_counts = [int(count) for count in observed_counts]
    if (
        not isinstance(prompt_counts, Mapping)
        or prompt_counts.get("count") != len(integer_counts)
        or prompt_counts.get("minimum")
        != (min(integer_counts) if integer_counts else None)
        or prompt_counts.get("maximum")
        != (max(integer_counts) if integer_counts else None)
        or prompt_counts.get("total") != sum(integer_counts)
        or prompt_counts.get("by_job")
        != [
            {
                "job_id": job.get("job_id"),
                "prompt_token_count": job.get("prompt_token_count"),
            }
            for job in jobs
        ]
    ):
        raise ValueError("validation prompt token counts differ")
    proposal_receipt = value.get("proposal_receipt")
    if (
        not isinstance(proposal_receipt, Mapping)
        or proposal_receipt.get("schema_version")
        != PROPOSAL_RECEIPT_SCHEMA_VERSION
        or proposal_receipt.get("status") != "complete"
        or proposal_receipt.get("source_contract_receipt_sha256")
        != value["source_contract_receipt_sha256"]
        or proposal_receipt.get("proposal_preflight_receipt_sha256")
        != value.get("proposal_preflight_receipt_sha256")
        or any(
            not isinstance(proposal_receipt.get(name), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(proposal_receipt[name]))
            for name in (
                "sha256",
                "proposals_sha256",
                "proposal_preflight_receipt_sha256",
                "source_contract_receipt_sha256",
                "run_anchor_sha256",
                "completion_sha256",
            )
        )
    ):
        raise ValueError("validation proposal receipt binding differs")
    _build_validation_run_anchor(
        jobs,
        preflight_sha256="a" * 64,
        approval_sha256="b" * 64,
        provenance=_validation_anchor_provenance(value),
    )
    return value


def _publish_validation_directory(
    destination: Path, contents: Mapping[str, bytes]
) -> None:
    destination = Path(destination)
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("validation output path is unsafe")
    try:
        parent_fd = _open_directory_no_symlinks(destination.parent)
    except OSError as exc:
        raise ValueError("validation output parent is missing or unsafe") from exc
    staging_name = f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    published = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        for name, source in contents.items():
            _write_exclusive_at(staging_fd, name, source)
        os.fsync(staging_fd)
        _rename_noreplace_at(parent_fd, staging_name, destination.name)
        published = True
        os.fsync(parent_fd)
    finally:
        if staging_fd is not None:
            if not published:
                for name in contents:
                    try:
                        os.unlink(name, dir_fd=staging_fd)
                    except FileNotFoundError:
                        pass
            os.close(staging_fd)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _write_exclusive_at(directory_fd: int, name: str, source: bytes) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("validation output file name differs")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(source)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short validation output write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _rename_noreplace_at(
    parent_fd: int, staging_name: str, destination_name: str
) -> None:
    if any(
        not name or name in {".", ".."} or "/" in name
        for name in (staging_name, destination_name)
    ):
        raise ValueError("validation publish name differs")
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
    result = renameat2(
        parent_fd,
        os.fsencode(staging_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            f"create-only validation output exists: {destination_name}"
        )
    raise OSError(error, os.strerror(error), destination_name)


def publish_validation_preflight(
    preflight: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    """Freeze one exact Task 4 preflight before its approved ledger run."""

    verified = _verify_validation_preflight_payload(dict(preflight))
    _publish_validation_directory(
        Path(output_dir), {"receipt.json": _pretty_bytes(verified)}
    )
    return verified


def _capture_exact_directory(path: Path, names: frozenset[str], label: str) -> dict[str, bytes]:
    try:
        descriptor = _open_directory_no_symlinks(Path(path))
        try:
            before = os.fstat(descriptor)
            names_before = set(os.listdir(descriptor))
            if names_before != names:
                raise OSError(f"{label} inventory differs")
            contents = {
                name: _read_stable_regular_at(
                    descriptor, name, require_single_link=True
                )
                for name in sorted(names)
            }
            after = os.fstat(descriptor)
            if set(os.listdir(descriptor)) != names_before or (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise OSError(f"{label} changed while captured")
            return contents
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc


def _capture_validation_preflight(path: Path) -> tuple[dict[str, object], bytes]:
    contents = _capture_exact_directory(
        path, _VALIDATION_PREFLIGHT_NAMES, "validation preflight"
    )
    source = contents["receipt.json"]
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation preflight receipt is invalid") from exc
    if not isinstance(value, dict) or source != _pretty_bytes(value):
        raise ValueError("validation preflight receipt is not canonical")
    return _verify_validation_preflight_payload(value), source


def _capture_validation_approval(
    path: Path,
) -> tuple[dict[str, object], bytes, str]:
    try:
        source = _capture_regular_file_no_symlinks(Path(path))
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("validation approval required") from exc
    expected_fields = {
        "schema_version",
        "stage",
        "validator_role",
        "preflight_sha256",
        "model",
        "model_revision",
        "model_snapshot_manifest_sha256",
        "tokenizer_identity_sha256",
        "job_count",
        "primary_call_count",
        "retry_call_ceiling",
        "worst_case_call_ceiling",
        "ledger_dir",
        "approved",
    }
    if (
        not isinstance(value, dict)
        or source != _pretty_bytes(value)
        or set(value) != expected_fields
        or value.get("schema_version") != VALIDATION_APPROVAL_SCHEMA_VERSION
        or value.get("stage") != "validation"
        or value.get("validator_role") != VALIDATOR_ROLE
        or value.get("model") != MODEL_ID
        or value.get("model_revision") != MODEL_REVISION
        or not isinstance(value.get("preflight_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(value["preflight_sha256"]))
        or any(
            not isinstance(value.get(name), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(value[name]))
            for name in (
                "model_snapshot_manifest_sha256",
                "tokenizer_identity_sha256",
            )
        )
        or any(
            type(value.get(name)) is not int
            for name in (
                "job_count",
                "primary_call_count",
                "retry_call_ceiling",
                "worst_case_call_ceiling",
            )
        )
        or value.get("job_count") < 0
        or value.get("primary_call_count") != value.get("job_count")
        or value.get("retry_call_ceiling") != value.get("job_count")
        or value.get("worst_case_call_ceiling") != value.get("job_count") * 2
        or not isinstance(value.get("ledger_dir"), str)
        or not Path(str(value["ledger_dir"])).is_absolute()
        or str(Path(str(value["ledger_dir"])).resolve()) != value["ledger_dir"]
        or value.get("approved") is not True
    ):
        raise PermissionError("validation approval required")
    return value, source, hashlib.sha256(source).hexdigest()


def verify_validation_approval(
    approval: object,
    preflight: Mapping[str, object],
    *,
    ledger_dir: Path,
) -> dict[str, object]:
    """Bind one exact validation approval to its preflight and ledger path."""

    if not isinstance(approval, Mapping):
        raise PermissionError("validation approval required")
    required = {
        "schema_version": VALIDATION_APPROVAL_SCHEMA_VERSION,
        "stage": "validation",
        "validator_role": VALIDATOR_ROLE,
        "preflight_sha256": preflight.get("receipt_sha256"),
        "model": preflight.get("model"),
        "model_revision": preflight.get("model_revision"),
        "model_snapshot_manifest_sha256": preflight.get(
            "model_snapshot_manifest_sha256"
        ),
        "tokenizer_identity_sha256": preflight.get(
            "tokenizer_identity_sha256"
        ),
        "job_count": preflight.get("job_count"),
        "primary_call_count": preflight.get("primary_call_count"),
        "retry_call_ceiling": preflight.get("retry_call_ceiling"),
        "worst_case_call_ceiling": preflight.get("worst_case_call_ceiling"),
        "ledger_dir": str(Path(ledger_dir).resolve()),
        "approved": True,
    }
    if set(approval) != set(required) or any(
        approval.get(name) != expected for name, expected in required.items()
    ):
        raise PermissionError("validation approval required")
    return dict(approval)


class _CapturedPromptCountTokenizer:
    """Replay approved prompt counts without loading tokenizer code or files."""

    def __init__(self, preflight: Mapping[str, object]) -> None:
        jobs = preflight.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("validation preflight jobs are missing")
        self._count_by_messages_sha256: dict[str, int] = {}
        for job in jobs:
            if not isinstance(job, Mapping):
                raise ValueError("validation preflight job differs")
            messages = job.get("messages")
            count = job.get("prompt_token_count")
            if not isinstance(messages, list) or type(count) is not int or count < 1:
                raise ValueError("validation preflight prompt token count differs")
            digest = canonical_sha256(messages)
            if digest in self._count_by_messages_sha256:
                raise ValueError("validation preflight prompt identity is duplicate")
            self._count_by_messages_sha256[digest] = count

    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if (
            not isinstance(messages, list)
            or tokenize is not True
            or add_generation_prompt is not True
        ):
            raise ValueError("validation captured prompt count request differs")
        count = self._count_by_messages_sha256.get(canonical_sha256(messages))
        if count is None:
            raise ValueError("validation captured prompt identity differs")
        return list(range(count))


def _authenticate_validation_after_approval(
    *,
    approval: Mapping[str, object],
    preflight_dir: Path,
    ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    source_contract_dir: Path,
) -> dict[str, object]:
    preflight, source = _capture_validation_preflight(Path(preflight_dir))
    preflight_sha256 = hashlib.sha256(source).hexdigest()
    authenticated = {**preflight, "receipt_sha256": preflight_sha256}
    verify_validation_approval(
        approval, authenticated, ledger_dir=Path(ledger_dir)
    )
    rebuilt = _build_validation_preflight_with_tokenizer(
        Path(source_contract_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        tokenizer=_CapturedPromptCountTokenizer(preflight),
    )
    if source != _pretty_bytes(rebuilt) or preflight != rebuilt:
        raise ValueError(
            "validation preflight differs from authenticated upstream replay"
        )
    verify_validation_approval(
        approval, authenticated, ledger_dir=Path(ledger_dir)
    )
    return authenticated


def _load_validation_model(
    *, approval: Mapping[str, object], preflight: Mapping[str, object]
) -> object:
    from .adaptive_obligation_v2_local_model import V2ValidationLocalJsonModel

    return V2ValidationLocalJsonModel(approval=approval, preflight=preflight)


def _execute_authenticated_validation_jobs(
    preflight: Mapping[str, object],
    *,
    ledger: AppendOnlyAttemptLedger,
    model: object,
) -> list[dict[str, object]]:
    jobs = preflight.get("jobs")
    if not isinstance(jobs, list) or any(
        not isinstance(job, Mapping) for job in jobs
    ):
        raise ValueError("validation execution jobs differ")
    return [
        run_validation_job_with_retry(job, ledger=ledger, model=model)
        for job in jobs
    ]


def execute_validation(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    source_contract_dir: Path,
) -> dict[str, object]:
    """Execute exact validation jobs through an approved create-only ledger."""

    approval, _approval_source, approval_sha256 = _capture_validation_approval(
        Path(approval_path)
    )
    preflight = _authenticate_validation_after_approval(
        approval=approval,
        preflight_dir=Path(preflight_dir),
        ledger_dir=Path(ledger_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        source_contract_dir=Path(source_contract_dir),
    )
    preflight_sha256 = str(preflight["receipt_sha256"])
    jobs = preflight.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("validation execution jobs differ")
    anchor = _build_validation_run_anchor(
        jobs,
        preflight_sha256=preflight_sha256,
        approval_sha256=approval_sha256,
        provenance=_validation_anchor_provenance(preflight),
    )
    ledger = AppendOnlyAttemptLedger(
        Path(ledger_dir), expected_anchor=anchor, create_only=True
    )
    if not jobs:
        completion = ledger.seal_completion()
        return {
            "status": "complete",
            "job_count": 0,
            "results": [],
            "event_count": 0,
            "completion": completion,
        }
    model = _load_validation_model(approval=approval, preflight=preflight)
    results = _execute_authenticated_validation_jobs(
        preflight, ledger=ledger, model=model
    )
    completion = ledger.seal_completion()
    return {
        "status": "complete",
        "job_count": len(results),
        "results": results,
        "event_count": len(ledger.read_events()),
        "completion": completion,
    }


def _authenticated_validation_run(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    source_contract_dir: Path,
) -> tuple[
    dict[str, object],
    bytes,
    str,
    AppendOnlyAttemptLedger,
]:
    approval, _approval_source, approval_sha256 = _capture_validation_approval(
        Path(approval_path)
    )
    preflight = _authenticate_validation_after_approval(
        approval=approval,
        preflight_dir=Path(preflight_dir),
        ledger_dir=Path(ledger_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        source_contract_dir=Path(source_contract_dir),
    )
    source = _pretty_bytes(
        {name: value for name, value in preflight.items() if name != "receipt_sha256"}
    )
    preflight_sha256 = str(preflight["receipt_sha256"])
    ledger, observed_approval_sha256 = _ledger_from_validation_preflight(
        preflight,
        preflight_sha256,
        Path(ledger_dir),
        approval_sha256=approval_sha256,
    )
    if observed_approval_sha256 != approval_sha256:
        raise ValueError("validation ledger approval binding differs")
    return preflight, source, approval_sha256, ledger


def _read_canonical_rows(source: bytes, name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name}:{line_number} is invalid") from exc
        if not isinstance(row, dict) or line + b"\n" != _compact_bytes(row):
            raise ValueError(f"{name}:{line_number} is not canonical")
        rows.append(row)
    return rows


def _validation_inventory_material(
    preflight: Mapping[str, object],
    *,
    preflight_sha256: str,
    approval_sha256: str,
    sealed: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, bytes]]:
    jobs = preflight["jobs"]
    assert isinstance(jobs, list)
    expected_anchor = _build_validation_run_anchor(
        jobs,
        preflight_sha256=preflight_sha256,
        approval_sha256=approval_sha256,
        provenance=_validation_anchor_provenance(preflight),
    )
    results = sealed.get("results")
    completion = sealed.get("completion")
    if (
        sealed.get("anchor") != expected_anchor
        or not isinstance(results, list)
        or len(results) != len(jobs)
        or not isinstance(completion, Mapping)
        or completion.get("completed_job_count") != len(jobs)
        or sealed.get("anchor_sha256") != completion.get("anchor_sha256")
    ):
        raise ValueError("validation inventory differs from sealed ledger")
    validated: list[dict[str, object]] = []
    for job, result in zip(jobs, results, strict=True):
        if not isinstance(job, Mapping) or not isinstance(result, Mapping):
            raise ValueError("validation inventory sealed result differs")
        if result.get("job_id") != job.get("job_id"):
            raise ValueError("validation inventory job identity differs")
        units = job.get("units")
        if not isinstance(units, list) or any(not isinstance(row, Mapping) for row in units):
            raise ValueError("validation inventory unit identity differs")
        unit_by_id = {str(row["unit_id"]): row for row in units}
        value = _validate_decision_value(
            result.get("value"), [str(value) for value in job["input_unit_ids"]]
        )
        row = validate_semantic_decision(job, value, units=unit_by_id)
        row["schema_version"] = VALIDATION_INVENTORY_SCHEMA_VERSION
        row["validation_job_id"] = job["job_id"]
        row["validation_result_sha256"] = canonical_sha256(dict(result))
        validated.append(row)
    accepted = accept_validated_o1(validated)
    validated_source = b"".join(_compact_bytes(row) for row in validated)
    accepted_source = b"".join(_compact_bytes(row) for row in accepted)
    proposal_receipt = preflight["proposal_receipt"]
    assert isinstance(proposal_receipt, Mapping)
    receipt: dict[str, object] = {
        "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "validation_count": len(validated),
        "accepted_count": len(accepted),
        "validation_preflight_sha256": preflight_sha256,
        "validation_jobs_sha256": preflight["jobs_sha256"],
        "source_contract_receipt_sha256": preflight[
            "source_contract_receipt_sha256"
        ],
        "proposal_receipt_sha256": proposal_receipt["sha256"],
        "proposal_inventory_sha256": proposal_receipt["proposals_sha256"],
        "proposal_preflight_receipt_sha256": proposal_receipt[
            "proposal_preflight_receipt_sha256"
        ],
        "approval_sha256": approval_sha256,
        "run_anchor_sha256": sealed["anchor_sha256"],
        "completion_sha256": sealed["completion_sha256"],
        "completion": dict(completion),
        "artifacts": {
            "validated.jsonl": {
                "rows": len(validated),
                "bytes": len(validated_source),
                "sha256": hashlib.sha256(validated_source).hexdigest(),
            },
            "accepted.jsonl": {
                "rows": len(accepted),
                "bytes": len(accepted_source),
                "sha256": hashlib.sha256(accepted_source).hexdigest(),
            },
        },
    }
    contents = {
        "validated.jsonl": validated_source,
        "accepted.jsonl": accepted_source,
        "receipt.json": _pretty_bytes(receipt),
    }
    return validated, accepted, receipt, contents


def _ledger_from_validation_preflight(
    preflight: Mapping[str, object], preflight_sha256: str, ledger_dir: Path,
    *, approval_sha256: str | None = None,
) -> tuple[AppendOnlyAttemptLedger, str]:
    try:
        anchor_source = _capture_regular_file_no_symlinks(Path(ledger_dir) / "anchor.json")
        anchor = json.loads(anchor_source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation ledger anchor is missing or unsafe") from exc
    observed_approval = anchor.get("approval_sha256") if isinstance(anchor, Mapping) else None
    if (
        not isinstance(observed_approval, str)
        or not re.fullmatch(r"[0-9a-f]{64}", observed_approval)
        or (approval_sha256 is not None and observed_approval != approval_sha256)
    ):
        raise ValueError("validation ledger approval binding differs")
    expected = _build_validation_run_anchor(
        preflight["jobs"],  # type: ignore[arg-type]
        preflight_sha256=preflight_sha256,
        approval_sha256=observed_approval,
        provenance=_validation_anchor_provenance(preflight),
    )
    if anchor != expected:
        raise ValueError("validation ledger anchor differs")
    return (
        AppendOnlyAttemptLedger(
            Path(ledger_dir), expected_anchor=expected, create_only=False
        ),
        observed_approval,
    )


def finalize_validation_inventory(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    output_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    source_contract_dir: Path,
) -> dict[str, object]:
    """Publish accepted O1 rows only by replaying a sealed validation ledger."""

    preflight, source, approval_sha256, ledger = _authenticated_validation_run(
        preflight_dir=Path(preflight_dir),
        approval_path=Path(approval_path),
        ledger_dir=Path(ledger_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        source_contract_dir=Path(source_contract_dir),
    )
    preflight_sha256 = hashlib.sha256(source).hexdigest()
    _validated, _accepted, receipt, contents = _validation_inventory_material(
        preflight,
        preflight_sha256=preflight_sha256,
        approval_sha256=approval_sha256,
        sealed=ledger.read_sealed_results(),
    )
    _publish_validation_directory(Path(output_dir), contents)
    return receipt


def load_authenticated_validation_inventory(
    *,
    output_dir: Path,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    source_contract_dir: Path,
) -> dict[str, object]:
    """Authenticate accepted O1 rows against the exact sealed Task 4 replay."""

    contents = _capture_exact_directory(
        Path(output_dir), _VALIDATION_INVENTORY_NAMES, "validation inventory"
    )
    try:
        receipt = json.loads(contents["receipt.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation inventory receipt is invalid") from exc
    if (
        not isinstance(receipt, dict)
        or contents["receipt.json"] != _pretty_bytes(receipt)
        or receipt.get("schema_version") != VALIDATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or not isinstance(receipt.get("approval_sha256"), str)
    ):
        raise ValueError("validation inventory receipt differs")
    preflight, source, approval_sha256, ledger = _authenticated_validation_run(
        preflight_dir=Path(preflight_dir),
        approval_path=Path(approval_path),
        ledger_dir=Path(ledger_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        source_contract_dir=Path(source_contract_dir),
    )
    preflight_sha256 = hashlib.sha256(source).hexdigest()
    if receipt.get("validation_preflight_sha256") != preflight_sha256:
        raise ValueError("validation inventory preflight binding differs")
    if receipt.get("approval_sha256") != approval_sha256:
        raise ValueError("validation inventory approval binding differs")
    validated, accepted, expected_receipt, expected_contents = (
        _validation_inventory_material(
            preflight,
            preflight_sha256=preflight_sha256,
            approval_sha256=approval_sha256,
            sealed=ledger.read_sealed_results(),
        )
    )
    if contents != expected_contents or receipt != expected_receipt:
        raise ValueError("validation inventory differs from the sealed ledger replay")
    if _read_canonical_rows(contents["validated.jsonl"], "validated.jsonl") != validated:
        raise ValueError("validation inventory differs from the sealed ledger replay")
    if _read_canonical_rows(contents["accepted.jsonl"], "accepted.jsonl") != accepted:
        raise ValueError("validation inventory differs from the sealed ledger replay")
    return {
        "validated": validated,
        "accepted": accepted,
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(contents["receipt.json"]).hexdigest(),
    }

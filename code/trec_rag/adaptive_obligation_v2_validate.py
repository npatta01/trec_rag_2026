"""Build inference-free opposite-fold validation jobs for adaptive obligation v2."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_obligation_v2_contract import canonical_sha256, sha256_text
from .adaptive_obligation_v2_ledger import AttemptSpec
from .adaptive_obligation_v2_propose import (
    PROPOSAL_RECEIPT_SCHEMA_VERSION,
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


def build_validation_preflight(
    contract: object,
    *,
    proposal_inventory_dir: object,
    proposal_preflight_dir: object,
    proposal_ledger_dir: object,
) -> dict[str, object]:
    """Plan exact V/V/2V calls from a replay-authenticated proposal inventory."""

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

    jobs = build_validation_jobs(proposals, contract)
    job_count = len(jobs)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "job_count": job_count,
        "primary_call_count": job_count,
        "retry_call_ceiling": job_count,
        "worst_case_call_ceiling": job_count * 2,
        "primary_max_new_tokens": PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
        "schema_sha256": canonical_sha256(VALIDATION_SCHEMA),
        "jobs_sha256": canonical_sha256(jobs),
        "jobs": jobs,
        "proposal_receipt": {
            "schema_version": proposal_receipt["schema_version"],
            "status": proposal_receipt["status"],
            "sha256": proposal_receipt_sha256,
            "proposal_count": proposal_receipt["proposal_count"],
            "proposals_sha256": proposal_receipt["proposals_sha256"],
            "proposal_preflight_receipt_sha256": proposal_receipt[
                "proposal_preflight_receipt_sha256"
            ],
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
        "tokenizer_load_count": 0,
        "inference_count": 0,
        "external_cost_usd": 0.0,
    }


def _validation_request(
    job: Mapping[str, object], ceiling: int
) -> dict[str, object]:
    return {
        "messages": job["messages"],
        "schema": VALIDATION_SCHEMA,
        "max_new_tokens": ceiling,
    }


def _build_validation_run_anchor(
    jobs: Sequence[Mapping[str, object]],
    *,
    preflight_sha256: str,
    approval_sha256: str,
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

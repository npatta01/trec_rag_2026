"""Scorer-only entrypoint for deterministic sparse v4 synthetic qualification.

This module is intentionally offline.  It consumes a sealed synthetic response
summary after dispatch/replay has already proven all 24 raw responses are
committed, then opens scorer-only gold and emits a deterministic scorer review.
It does not perform model inference, retrieval, reranking, topic access, or
relevance-judgment access.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from trec_rag import det_sparse_v4_contract as contract


SEALED_SCORER_INPUT_SCHEMA_VERSION = "semantic_anchor_sealed_scorer_input_v1"
SCORER_REVIEW_SCHEMA_VERSION = "semantic_anchor_scorer_review_v1"
SCORER_REVIEW_STATUS = "scorer_review_pass"
GOLD_LABELS_ARTIFACT = "semantic_anchor_gold_labels_v1.json"
SCORER_INPUT_TERMINAL_STATE = "raw_sealed_pending_scorer"


def build_scorer_review(
    sealed_responses_path: Path,
    *,
    artifact_dir: Path = contract.ARTIFACT_DIR,
) -> dict[str, object]:
    """Build a scorer review from sealed responses, opening gold only after seal checks."""

    case_order = _case_order(artifact_dir)
    sealed_path = sealed_responses_path.resolve()
    sealed_input = _require_mapping(
        contract.load_json_no_duplicates(sealed_path),
        "sealed scorer input",
    )
    responses_by_case = validate_sealed_scorer_input(
        sealed_input,
        case_order=case_order,
    )

    artifact_hashes = contract.validate_artifact_bundle(artifact_dir)
    gold_path = artifact_dir / GOLD_LABELS_ARTIFACT
    gold = _require_mapping(
        contract.load_json_no_duplicates(gold_path),
        "gold labels",
    )
    scorer_receipt = contract.build_scorer_receipt(
        responses_by_case,
        gold=gold,
        case_order=case_order,
    )
    return {
        "schema_version": SCORER_REVIEW_SCHEMA_VERSION,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": SCORER_REVIEW_STATUS,
        "sealed_responses_path": str(sealed_path),
        "sealed_responses_sha256": contract.sha256_file(sealed_path),
        "gold_artifact": GOLD_LABELS_ARTIFACT,
        "gold_sha256": artifact_hashes[GOLD_LABELS_ARTIFACT],
        "terminal_state": _require_mapping(
            sealed_input.get("terminal_receipt"), "terminal receipt"
        )["terminal_state"],
        "sealed_response_count": len(case_order),
        "gold_opened_after_seal": True,
        "inference_authorized": False,
        "dispatch_authorized": False,
        "external_cost_authorized": False,
        "scorer_receipt": scorer_receipt,
    }


def validate_scorer_review(
    review: Mapping[str, object],
    *,
    case_order: Sequence[str] | None = None,
) -> None:
    expected_case_order = tuple(case_order) if case_order is not None else _case_order()
    _require_exact_keys(
        review,
        "scorer review",
        {
            "schema_version",
            "experiment_id",
            "status",
            "sealed_responses_path",
            "sealed_responses_sha256",
            "gold_artifact",
            "gold_sha256",
            "terminal_state",
            "sealed_response_count",
            "gold_opened_after_seal",
            "inference_authorized",
            "dispatch_authorized",
            "external_cost_authorized",
            "scorer_receipt",
        },
    )
    if review.get("schema_version") != SCORER_REVIEW_SCHEMA_VERSION:
        raise ValueError("scorer review schema_version mismatch")
    if review.get("experiment_id") != contract.EXPERIMENT_ID:
        raise ValueError("scorer review experiment_id mismatch")
    if review.get("status") != SCORER_REVIEW_STATUS:
        raise ValueError("scorer review status mismatch")
    if not isinstance(review.get("sealed_responses_path"), str) or not review.get(
        "sealed_responses_path"
    ):
        raise ValueError("scorer review sealed_responses_path must be nonempty string")
    _validate_sha256(review.get("sealed_responses_sha256"), "sealed_responses_sha256")
    if review.get("gold_artifact") != GOLD_LABELS_ARTIFACT:
        raise ValueError("scorer review gold artifact mismatch")
    _validate_sha256(review.get("gold_sha256"), "gold_sha256")
    if review.get("terminal_state") != SCORER_INPUT_TERMINAL_STATE:
        raise ValueError("scorer review terminal_state mismatch")
    if review.get("sealed_response_count") != len(expected_case_order):
        raise ValueError("scorer review sealed_response_count mismatch")
    if review.get("gold_opened_after_seal") is not True:
        raise ValueError("scorer review must open gold after seal")
    if review.get("inference_authorized") is not False:
        raise ValueError("scorer review must not authorize inference")
    if review.get("dispatch_authorized") is not False:
        raise ValueError("scorer review must not authorize dispatch")
    if review.get("external_cost_authorized") is not False:
        raise ValueError("scorer review must not authorize external cost")
    receipt = _require_mapping(review.get("scorer_receipt"), "scorer receipt")
    contract.validate_scorer_receipt(receipt, case_order=expected_case_order)


def validate_sealed_scorer_input(
    value: Mapping[str, object],
    *,
    case_order: Sequence[str],
) -> dict[str, Mapping[str, object]]:
    _require_exact_keys(
        value,
        "sealed scorer input",
        {
            "schema_version",
            "run_manifest",
            "terminal_receipt",
            "reservations",
            "dispatches",
            "raw_responses",
            "case_receipts",
            "responses",
        },
    )
    if value.get("schema_version") != SEALED_SCORER_INPUT_SCHEMA_VERSION:
        raise ValueError("sealed scorer input schema_version mismatch")
    run_manifest = _require_mapping(value.get("run_manifest"), "run manifest")
    terminal = _require_mapping(value.get("terminal_receipt"), "terminal receipt")
    reservations = _mapping_list(value.get("reservations"), "reservations")
    dispatches = _mapping_list(value.get("dispatches"), "dispatches")
    raw_records = _mapping_list(value.get("raw_responses"), "raw responses")
    case_receipts = _mapping_list(value.get("case_receipts"), "case receipts")
    contract.validate_ledger_prefix(
        case_order=case_order,
        reservations=reservations,
        dispatches=dispatches,
        raw_responses=raw_records,
        transport_failures=[],
        case_receipts=case_receipts,
        terminal_receipt=terminal,
        run_manifest=run_manifest,
    )
    contract.validate_terminal_receipt(terminal)
    if terminal.get("terminal_state") != SCORER_INPUT_TERMINAL_STATE:
        raise ValueError("sealed scorer input requires raw_sealed_pending_scorer")
    if terminal.get("attempted_calls") != len(case_order):
        raise ValueError("sealed scorer input attempted_calls mismatch")
    if terminal.get("completed_calls") != len(case_order):
        raise ValueError("sealed scorer input completed_calls mismatch")
    if terminal.get("raw_committed_calls") != len(case_order):
        raise ValueError("sealed scorer input raw_committed_calls mismatch")
    if terminal.get("gold_opened") is not False:
        raise ValueError("sealed scorer input must not open gold before scorer")

    raw_by_case = {str(record["case_id"]): record for record in raw_records}
    raw_responses = value.get("responses")
    if not isinstance(raw_responses, list) or len(raw_responses) != len(case_order):
        raise ValueError("sealed scorer input responses must contain all cases")
    responses_by_case: dict[str, Mapping[str, object]] = {}
    observed_order: list[str] = []
    for row in raw_responses:
        record = _require_mapping(row, "sealed scorer response")
        _require_exact_keys(
            record,
            "sealed scorer response",
            {"case_id", "raw_response_sha256", "raw_response_body"},
        )
        case_id = _require_string(record.get("case_id"), "case_id")
        observed_order.append(case_id)
        raw_response_sha256 = _validate_sha256(
            record.get("raw_response_sha256"), "raw_response_sha256"
        )
        raw_record = raw_by_case.get(case_id)
        if raw_record is None:
            raise ValueError("sealed scorer response lacks raw response record")
        if raw_response_sha256 != raw_record.get("body_sha256"):
            raise ValueError("sealed scorer response hash differs from raw record")
        raw_body = _require_string(record.get("raw_response_body"), "raw_response_body")
        if contract.sha256_bytes(raw_body.encode("utf-8")) != raw_response_sha256:
            raise ValueError("sealed scorer response body hash mismatch")
        try:
            payload = _loads_object_no_duplicates(raw_body, "raw response body")
            response = contract.extract_model_response_from_chat_completion(
                payload,
                case_id=case_id,
                u1_token_count=7,
            )
        except ValueError:
            response = {
                "schema_version": contract.MODEL_RESPONSE_SCHEMA_VERSION,
                "case_id": case_id,
                "decision": "select",
                "start_token": True,
                "end_token": 0,
            }
        responses_by_case[case_id] = response
    if tuple(observed_order) != tuple(case_order):
        raise ValueError("sealed scorer input response case order mismatch")
    if len(responses_by_case) != len(case_order):
        raise ValueError("sealed scorer input duplicate response case_id")
    return responses_by_case


def _mapping_list(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(_require_mapping(item, name) for item in value)


def _loads_object_no_duplicates(text: str, name: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be JSON object")
    return value


def _case_order(artifact_dir: Path = contract.ARTIFACT_DIR) -> tuple[str, ...]:
    case_order_record = _require_mapping(
        contract.load_json_no_duplicates(artifact_dir / "semantic_anchor_case_order_v1.json"),
        "case order",
    )
    raw_case_order = case_order_record.get("case_order")
    if not isinstance(raw_case_order, list):
        raise ValueError("case_order must be list")
    case_order = tuple(_require_string(value, "case_order entry") for value in raw_case_order)
    expected = tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))
    if case_order != expected:
        raise ValueError("fixed scorer case order drifted")
    return case_order


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], name: str, expected_keys: set[str]
) -> None:
    keys = set(value)
    if keys != expected_keys:
        raise ValueError(
            f"{name} keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}"
        )


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase sha256")
    return value


def canonical_review_bytes(review: Mapping[str, object], *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            review,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return f"{text}\n".encode("utf-8")


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score sealed deterministic sparse v4 synthetic responses."
    )
    parser.add_argument("sealed_responses", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=contract.ARTIFACT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    review = build_scorer_review(
        args.sealed_responses,
        artifact_dir=args.artifact_dir,
    )
    validate_scorer_review(review)
    payload = canonical_review_bytes(review, pretty=args.pretty)
    if args.output:
        _create_only(args.output, payload)
    else:
        print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

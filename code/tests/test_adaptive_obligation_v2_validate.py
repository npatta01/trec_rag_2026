from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_validate as validate_module
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256, sha256_text
from trec_rag.adaptive_obligation_v2_ledger import AppendOnlyAttemptLedger
from trec_rag.adaptive_obligation_v2_validate import (
    PROPOSAL_RECEIPT_SCHEMA_VERSION,
    VALIDATION_SCHEMA,
    VALIDATION_DECISIONS,
    accept_validated_o1,
    build_validation_preflight,
    build_validation_jobs,
    run_validation_job_with_retry,
    validate_semantic_decision,
    validate_proposal_record,
)


def _unit(*, fold: int, document_id: str, text: str) -> dict[str, object]:
    identity = {
        "topic_id": "219",
        "parent_id": "parent-1",
        "fold": fold,
        "document_id": document_id,
        "window_id": f"window-{document_id}",
        "start": 0,
        "end": len(text),
        "text": text,
    }
    return {
        "unit_id": canonical_sha256(identity),
        **identity,
        "text_sha256": sha256_text(text),
    }


def _contract_fixture() -> dict[str, object]:
    parent_text = "Risks of automated hiring decisions."
    narrative = "Explain how algorithms affect access to employment."
    parent = {
        "topic_id": "219",
        "parent_id": "parent-1",
        "manifest_order": 3,
        "text": parent_text,
        "text_sha256": sha256_text(parent_text),
        "query": f"{narrative}\n\nExplicit obligation:\n{parent_text}",
        "query_sha256": sha256_text(
            f"{narrative}\n\nExplicit obligation:\n{parent_text}"
        ),
    }
    units = [
        _unit(
            fold=0,
            document_id="proposal-document",
            text="Audits can reveal discriminatory screening patterns.",
        ),
        _unit(
            fold=1,
            document_id="validation-document-a",
            text="Independent evaluations measure disparate selection rates.",
        ),
        _unit(
            fold=1,
            document_id="validation-document-b",
            text="Bias testing compares outcomes across demographic groups.",
        ),
    ]
    reservoirs = []
    for fold in (0, 1):
        local = [row for row in units if row["fold"] == fold]
        documents = [
            {
                "document_id": row["document_id"],
                "document_sha256": "d" * 64,
                "window_id": row["window_id"],
                "window_text": row["text"],
                "window_sha256": sha256_text(str(row["text"])),
                "unit_ids": [row["unit_id"]],
            }
            for row in local
        ]
        reservoirs.append(
            {
                "reservoir_id": f"reservoir-{fold}",
                "topic_id": "219",
                "parent_id": "parent-1",
                "fold": fold,
                "document_count": len(documents),
                "documents": documents,
            }
        )
    return {"parents": [parent], "reservoirs": reservoirs, "units": units}


def _supported_proposal(**changes: object) -> dict[str, object]:
    contract = _contract_fixture()
    source = contract["units"][0]  # type: ignore[index]
    proposal = {
        "proposal_id": "proposal-1",
        "status": "SUPPORTED",
        "reason_code": "SUPPORTED",
        "topic_id": "219",
        "parent_id": "parent-1",
        "proposal_fold": 0,
        "parent_manifest_order": 3,
        "label": "demographic outcome bias audits",
        "scope_rationale": "SECRET PROPOSER RATIONALE",
        "support_unit_ids": [source["unit_id"]],
    }
    proposal.update(changes)
    return proposal


def test_decisions_are_finite() -> None:
    assert set(VALIDATION_DECISIONS) == {
        "SUPPORTED",
        "NO_EVIDENCE",
        "OUT_OF_SCOPE",
        "ANSWER_FACT",
        "DUPLICATE_O0",
        "WRONG_DOMAIN",
    }


def test_validation_uses_only_opposite_fold_units() -> None:
    contract = _contract_fixture()
    proposal = _supported_proposal()
    jobs = build_validation_jobs([proposal], contract)
    assert len(jobs) == 1
    assert jobs[0]["validation_fold"] == 1
    assert jobs[0]["proposal_fold"] == 0
    assert all(unit["fold"] == 1 for unit in jobs[0]["units"])
    rendered = json.dumps(jobs[0], sort_keys=True)
    assert "SECRET PROPOSER RATIONALE" not in rendered
    assert "Audits can reveal discriminatory screening patterns" not in rendered
    assert str(proposal["support_unit_ids"][0]) not in rendered  # type: ignore[index]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"support_unit_ids": ["unknown"]}, "unknown"),
        ({"topic_id": "144"}, "protected"),
        (
            {"label": "Audits can reveal discriminatory screening patterns."},
            "copied",
        ),
        ({"label": "RISKS of automated-hiring decisions"}, "duplicate"),
    ],
)
def test_invalid_proposal_records_fail_before_validation_jobs(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_proposal_record(_supported_proposal(**change), _contract_fixture())


def test_source_identity_and_hash_mismatch_are_rejected() -> None:
    wrong_fold = copy.deepcopy(_contract_fixture())
    wrong_fold["units"][0]["fold"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="identity"):
        validate_proposal_record(_supported_proposal(), wrong_fold)

    wrong_hash = copy.deepcopy(_contract_fixture())
    wrong_hash["units"][0]["text_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="hash"):
        validate_proposal_record(_supported_proposal(), wrong_hash)

    wrong_query_hash = copy.deepcopy(_contract_fixture())
    wrong_query_hash["parents"][0]["query_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="hash"):
        validate_proposal_record(_supported_proposal(), wrong_query_hash)


def test_proposing_unit_must_belong_to_its_source_fold_reservoir() -> None:
    contract = _contract_fixture()
    rogue = _unit(
        fold=0,
        document_id="rogue-document",
        text="Separate evidence that was never selected for the reservoir.",
    )
    contract["units"].append(rogue)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="reservoir"):
        validate_proposal_record(
            _supported_proposal(support_unit_ids=[rogue["unit_id"]]), contract
        )


def test_duplicate_parent_fold_proposals_are_rejected() -> None:
    with pytest.raises(ValueError, match="parent/fold"):
        build_validation_jobs(
            [_supported_proposal(), _supported_proposal(proposal_id="proposal-2")],
            _contract_fixture(),
        )


def test_protected_unsupported_proposal_rejects_before_contract_access() -> None:
    proposal = _supported_proposal(
        topic_id="144",
        status="UNSUPPORTED",
        reason_code="NO_ABSTRACT_CHILD",
        support_unit_ids=[],
    )
    with pytest.raises(ValueError, match="protected"):
        build_validation_jobs([proposal], object())


def test_preflight_rejects_protected_unsupported_before_contract_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _supported_proposal(
        topic_id="144",
        status="UNSUPPORTED",
        reason_code="NO_ABSTRACT_CHILD",
        support_unit_ids=[],
    )
    proposals = [proposal]
    authenticated = _authenticated_proposal_inventory(proposals)
    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: authenticated,
    )
    with pytest.raises(ValueError, match="protected"):
        build_validation_preflight(
            contract_dir=object(),
            proposal_inventory_dir=object(),
            proposal_preflight_dir=object(),
            proposal_ledger_dir=object(),
        )


def _decision(
    unit_id: str, *, decision: str = "SUPPORTED"
) -> dict[str, object]:
    return {"decision": decision, "support_unit_ids": [unit_id]}


def test_semantic_decision_requires_known_opposite_fold_distinct_document() -> None:
    contract = _contract_fixture()
    source = contract["units"][0]  # type: ignore[index]
    opposite = contract["units"][1]  # type: ignore[index]
    proposal = {
        **_supported_proposal(),
        "proposal_document_ids": [source["document_id"]],
    }
    units = {str(row["unit_id"]): row for row in contract["units"]}  # type: ignore[index]

    accepted = validate_semantic_decision(
        proposal, _decision(str(opposite["unit_id"])), units=units
    )
    assert accepted["accepted"] is True
    assert accepted["support_document_ids"] == [
        "proposal-document",
        "validation-document-a",
    ]

    same_document = copy.deepcopy(opposite)
    same_document["document_id"] = "proposal-document"
    same_document["window_id"] = "window-proposal-document"
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
    same_document["unit_id"] = canonical_sha256(
        {name: same_document[name] for name in identity_names}
    )
    rejected = validate_semantic_decision(
        proposal,
        _decision(str(same_document["unit_id"])),
        units={str(same_document["unit_id"]): same_document},
    )
    assert rejected["accepted"] is False
    assert "distinct_document" in rejected["reasons"]


def test_two_proposing_units_from_one_document_are_deduplicated() -> None:
    contract = _contract_fixture()
    opposite = contract["units"][1]  # type: ignore[index]
    proposal = {
        **_supported_proposal(),
        "proposal_document_ids": ["proposal-document", "proposal-document"],
    }
    result = validate_semantic_decision(
        proposal,
        _decision(str(opposite["unit_id"])),
        units={str(opposite["unit_id"]): opposite},
    )
    assert result["accepted"] is True
    assert result["proposal_document_ids"] == ["proposal-document"]


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        ({"decision": "MAYBE", "support_unit_ids": []}, "invalid_decision"),
        ({"decision": "SUPPORTED", "support_unit_ids": ["unknown"]}, "unknown_unit"),
        ({"decision": "WRONG_DOMAIN", "support_unit_ids": []}, "wrong_domain"),
    ],
)
def test_semantic_decision_rejects_invalid_code_unknown_unit_and_wrong_domain(
    decision: dict[str, object], reason: str
) -> None:
    proposal = {
        **_supported_proposal(),
        "proposal_document_ids": ["proposal-document"],
    }
    result = validate_semantic_decision(proposal, decision, units={})
    assert result["accepted"] is False
    assert reason in result["reasons"]


def test_semantic_decision_rejects_source_fold_and_source_identity() -> None:
    contract = _contract_fixture()
    proposal = {
        **_supported_proposal(),
        "proposal_document_ids": ["proposal-document"],
    }
    source_fold = contract["units"][0]  # type: ignore[index]
    result = validate_semantic_decision(
        proposal,
        _decision(str(source_fold["unit_id"])),
        units={str(source_fold["unit_id"]): source_fold},
    )
    assert "opposite_fold" in result["reasons"]

    mismatch = copy.deepcopy(contract["units"][1])  # type: ignore[index]
    mismatch["parent_id"] = "other-parent"
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
    mismatch["unit_id"] = canonical_sha256(
        {name: mismatch[name] for name in identity_names}
    )
    result = validate_semantic_decision(
        proposal,
        _decision(str(mismatch["unit_id"])),
        units={str(mismatch["unit_id"]): mismatch},
    )
    assert "source_identity" in result["reasons"]


def _validated_row(
    proposal_id: str,
    parent_id: str,
    order: int,
    label: str,
    documents: list[str],
) -> dict[str, object]:
    return {
        "accepted": True,
        "decision": "SUPPORTED",
        "proposal_id": proposal_id,
        "topic_id": "219",
        "parent_id": parent_id,
        "parent_manifest_order": order,
        "label": label,
        "support_document_ids": documents,
    }


def test_acceptance_caps_one_parent_and_four_topic_deterministically() -> None:
    rows = [
        _validated_row("p-long", "parent-a", 0, "longer label wins never", ["a", "b"]),
        _validated_row("p-short", "parent-a", 0, "short label", ["c", "d"]),
        _validated_row("p-b", "parent-b", 1, "beta", ["e", "f"]),
        _validated_row("p-c", "parent-c", 2, "charlie", ["g", "h"]),
        _validated_row("p-d", "parent-d", 3, "delta", ["i", "j"]),
        _validated_row("p-e", "parent-e", 4, "echo", ["k"]),
    ]
    accepted = accept_validated_o1(rows)
    reversed_accepted = accept_validated_o1(list(reversed(rows)))
    assert [row["proposal_id"] for row in accepted] == [
        "p-short",
        "p-b",
        "p-c",
        "p-d",
    ]
    assert accepted == reversed_accepted
    assert len({row["parent_id"] for row in accepted}) == 4


def test_acceptance_rejects_duplicate_proposal_ids_before_capping() -> None:
    rows = [
        _validated_row("duplicate", "parent-a", 0, "alpha", ["a", "b"]),
        _validated_row("duplicate", "parent-b", 1, "beta", ["c", "d"]),
    ]
    with pytest.raises(ValueError, match="duplicate proposal"):
        accept_validated_o1(rows)
    with pytest.raises(ValueError, match="duplicate proposal"):
        accept_validated_o1(list(reversed(rows)))


def _authenticated_proposal_inventory(
    proposals: list[dict[str, object]],
    *,
    source_contract_receipt_sha256: str = "f" * 64,
) -> dict[str, object]:
    receipt = {
        "schema_version": PROPOSAL_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "job_count": len(proposals),
        "proposal_count": len(proposals),
        "proposals_sha256": "a" * 64,
        "proposal_preflight_receipt_sha256": "b" * 64,
        "source_contract_receipt_sha256": source_contract_receipt_sha256,
        "run_anchor_sha256": "c" * 64,
        "completion_sha256": "d" * 64,
        "completion": {
            "completed_job_count": len(proposals),
            "anchor_sha256": "c" * 64,
        },
    }
    return {
        "proposals": proposals,
        "receipt": receipt,
        "receipt_sha256": "e" * 64,
    }


def test_validation_preflight_rejects_an_altered_opposite_fold_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals = [_supported_proposal()]
    original_receipt_bytes = (
        json.dumps(
            {"schema_version": "contract-v1", "status": "complete"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    authenticated = _authenticated_proposal_inventory(
        proposals,
        source_contract_receipt_sha256=hashlib.sha256(
            original_receipt_bytes
        ).hexdigest(),
    )
    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: authenticated,
    )
    altered = copy.deepcopy(_contract_fixture())
    altered["reservoirs"][1]["documents"] = altered["reservoirs"][1][  # type: ignore[index]
        "documents"
    ][1:]
    contract_root = tmp_path / "altered-contract"
    contract_root.mkdir()
    altered_receipt = {
        "schema_version": "contract-v1",
        "status": "complete",
        "topic_ids": ["219"],
        "alteration": "opposite-fold reservoir changed",
    }
    (contract_root / "receipt.json").write_bytes(
        (
            json.dumps(altered_receipt, indent=2, sort_keys=True) + "\n"
        ).encode()
    )
    (contract_root / "manifest.json").write_bytes(b"{}\n")
    for name in ("parents", "reservoirs", "units"):
        (contract_root / f"{name}.jsonl").write_bytes(
            b"".join(
                (
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                ).encode()
                for row in altered[name]  # type: ignore[index]
            )
        )

    with pytest.raises(ValueError, match="source contract receipt"):
        build_validation_preflight(
            contract_dir=contract_root,
            proposal_inventory_dir=object(),
            proposal_preflight_dir=object(),
            proposal_ledger_dir=object(),
        )


def test_validation_preflight_rejects_a_caller_supplied_contract_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = _authenticated_proposal_inventory([_supported_proposal()])
    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: authenticated,
    )
    with pytest.raises(ValueError, match="contract path"):
        build_validation_preflight(
            contract_dir=_contract_fixture(),
            proposal_inventory_dir=object(),
            proposal_preflight_dir=object(),
            proposal_ledger_dir=object(),
        )


def test_validation_preflight_requires_authenticated_complete_proposals_and_v_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposals = [_supported_proposal()]
    authenticated = _authenticated_proposal_inventory(proposals)
    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: authenticated,
    )
    contract = _contract_fixture()
    monkeypatch.setattr(
        validate_module,
        "_capture_and_verify_contract_snapshot",
        lambda _path, *, expected_receipt_sha256: (
            contract,
            expected_receipt_sha256,
        ),
    )
    preflight = build_validation_preflight(
        contract_dir=Path("unused-contract"),
        proposal_inventory_dir=object(),
        proposal_preflight_dir=object(),
        proposal_ledger_dir=object(),
    )
    assert preflight["job_count"] == 1
    assert preflight["primary_call_count"] == 1
    assert preflight["retry_call_ceiling"] == 1
    assert preflight["worst_case_call_ceiling"] == 2
    assert preflight["jobs_sha256"] == canonical_sha256(preflight["jobs"])
    for counter in (
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
        "model_load_count",
        "tokenizer_load_count",
        "inference_count",
    ):
        assert preflight[counter] == 0
    assert preflight["qrels_opened"] is False

    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: {**authenticated, "receipt_sha256": "bad"},
    )
    with pytest.raises(ValueError, match="authenticated"):
        build_validation_preflight(
            contract_dir=object(),
            proposal_inventory_dir=object(),
            proposal_preflight_dir=object(),
            proposal_ledger_dir=object(),
        )
    incomplete = {
        **authenticated,
        "receipt": {**authenticated["receipt"], "status": "incomplete"},
    }
    monkeypatch.setattr(
        validate_module,
        "load_authenticated_proposal_inventory",
        lambda **_kwargs: incomplete,
    )
    with pytest.raises(ValueError, match="authenticated"):
        build_validation_preflight(
            contract_dir=object(),
            proposal_inventory_dir=object(),
            proposal_preflight_dir=object(),
            proposal_ledger_dir=object(),
        )


class _FakeValidationModel:
    def __init__(self, completions: list[tuple[bytes, int]]) -> None:
        self.completions = list(completions)
        self.ceilings: list[int] = []

    def generate(
        self,
        messages: object,
        schema: object,
        *,
        max_new_tokens: int,
    ) -> tuple[bytes, int]:
        assert isinstance(messages, list)
        assert schema == VALIDATION_SCHEMA
        self.ceilings.append(max_new_tokens)
        return self.completions.pop(0)


class _FakeValidationLedger:
    def __init__(self, classifications: list[str]) -> None:
        self.classifications = list(classifications)
        self.specs: list[object] = []

    def run_attempt(
        self,
        spec: object,
        generate: object,
        *,
        messages: object,
        schema: object,
        allowed_support_unit_ids: object,
    ) -> dict[str, object]:
        self.specs.append(spec)
        raw, token_count = generate(  # type: ignore[operator]
            messages, schema, spec.max_new_tokens  # type: ignore[union-attr]
        )
        classification = self.classifications.pop(0)
        if classification == "valid":
            return {
                "classification": "valid",
                "value": json.loads(raw),
                "output_token_count": token_count,
            }
        return {
            "classification": classification,
            "error": classification,
            "output_token_count": token_count,
        }


def test_guarded_runner_uses_only_injected_ledger_and_model_and_retries_truncation() -> None:
    job = build_validation_jobs([_supported_proposal()], _contract_fixture())[0]
    opposite_id = str(job["input_unit_ids"][0])  # type: ignore[index]
    primary = (b'{"decision":', 256)
    retry_value = {
        "decision": "SUPPORTED",
        "support_unit_ids": [opposite_id],
    }
    retry = (json.dumps(retry_value).encode(), 20)
    model = _FakeValidationModel([primary, retry])
    ledger = _FakeValidationLedger(["truncated_at_ceiling", "valid"])

    result = run_validation_job_with_retry(job, ledger=ledger, model=model)
    assert result["accepted"] is True
    assert model.ceilings == [256, 512]
    assert [spec.stage for spec in ledger.specs] == ["validation", "validation"]
    assert [spec.attempt_ordinal for spec in ledger.specs] == [1, 2]

    with pytest.raises(ValueError, match="injected ledger and model"):
        run_validation_job_with_retry(job, ledger=None, model=model)


def test_guarded_runner_never_retries_schema_or_semantic_errors() -> None:
    job = build_validation_jobs([_supported_proposal()], _contract_fixture())[0]
    model = _FakeValidationModel([(b"{}", 1)])
    ledger = _FakeValidationLedger(["schema_error"])
    with pytest.raises(ValueError, match="schema_error"):
        run_validation_job_with_retry(job, ledger=ledger, model=model)
    assert model.ceilings == [256]


def test_guarded_runner_rejects_truncation_below_the_exact_ceiling() -> None:
    job = build_validation_jobs([_supported_proposal()], _contract_fixture())[0]
    model = _FakeValidationModel([(b'{"decision":', 12)])
    ledger = _FakeValidationLedger(["truncated_at_ceiling"])
    with pytest.raises(ValueError, match="token count"):
        run_validation_job_with_retry(job, ledger=ledger, model=model)
    assert model.ceilings == [256]


def test_task3_real_ledger_runs_validation_retry_and_sealed_reopen(
    tmp_path: Path,
) -> None:
    job = build_validation_jobs([_supported_proposal()], _contract_fixture())[0]
    anchor = validate_module._build_validation_run_anchor(
        [job], preflight_sha256="a" * 64, approval_sha256="b" * 64
    )
    root = tmp_path / "task3-ledger"
    ledger = AppendOnlyAttemptLedger(
        root,
        expected_anchor=anchor,
        create_only=True,
    )
    opposite_id = str(job["input_unit_ids"][0])  # type: ignore[index]
    completion = json.dumps(
        {"decision": "SUPPORTED", "support_unit_ids": [opposite_id]}
    ).encode()
    model = _FakeValidationModel([(b'{"decision":', 256), (completion, 20)])

    result = run_validation_job_with_retry(job, ledger=ledger, model=model)
    assert result["accepted"] is True
    assert model.ceilings == [256, 512]
    assert [event["classification"] for event in ledger.read_events()[1::2]] == [
        "truncated_at_ceiling",
        "valid",
    ]
    completion_receipt = ledger.seal_completion()
    assert completion_receipt["completed_job_count"] == 1
    reopened = AppendOnlyAttemptLedger(
        root,
        expected_anchor=anchor,
        create_only=False,
    )
    assert len(reopened.read_events()) == 4

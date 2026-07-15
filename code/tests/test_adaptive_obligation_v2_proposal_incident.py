from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_proposal_incident as incident_module
from trec_rag.adaptive_obligation_v2_proposal_incident import (
    R1_RAW_SHA256,
    _prove_rationale_length_is_sole_defect,
    build_r1_incident_receipt,
    publish_r1_incident,
    verify_r1_incident,
)
from trec_rag.adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    PRIMARY_JOB_COUNT,
    PROPOSAL_SCHEMA,
    _build_run_anchor,
)


R1_JOB_ID = "87df2b737285e7243ba3726dc621bf207164cb614bf548f7b3bad2f9d661ba5b"
R1_SUPPORT_IDS = [
    "6915a66af95e39bff8f3c0d878ac528fb92ec9dd7eb4999965af9b7ff86a5fc6",
    "57416cfe817b17adac39c57d4fabaed5e2210ae88a6a50210980a1af629c51a7",
]
R1_RAW = (
    b'{"status":"SUPPORTED","reason_code":"SUPPORTED","o1":{"label":"Positive '
    b'effects of technology on daily life and healthcare","scope_rationale":"The '
    b'evidence units highlight how technology improves healthcare access and daily '
    b'life through tools like telehealth, apps that enhance brain function, and '
    b'medical diagnostics, directly supporting the parent query about positive '
    b'societal impacts.","support_unit_ids":["6915a66af95e39bff8f3c0d878ac528'
    b'fb92ec9dd7eb4999965af9b7ff86a5fc6","57416cfe817b17adac39c57d4fabaed5e'
    b'2210ae88a6a50210980a1af629c51a7"]}}'
)


def _compact_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


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


def _sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _event(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "event_sha256": _sha256(_compact_bytes(payload))}


def _fixture_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for index in range(PRIMARY_JOB_COUNT):
        jobs.append(
            {
                "job_id": R1_JOB_ID if index == 0 else _sha256(f"job-{index}".encode()),
                "input_unit_ids": (
                    R1_SUPPORT_IDS
                    if index == 0
                    else [_sha256(f"unit-{index}".encode())]
                ),
                "messages": [
                    {"role": "system", "content": "fixture"},
                    {"role": "user", "content": f"job {index}"},
                ],
            }
        )
    return jobs


@dataclass
class R1FailureFixture:
    preflight_dir: Path
    approval_path: Path
    ledger_dir: Path

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "preflight_dir": self.preflight_dir,
            "approval_path": self.approval_path,
            "ledger_dir": self.ledger_dir,
        }

    def snapshot_ledger(self) -> dict[str, bytes | None]:
        return {
            str(path.relative_to(self.ledger_dir)): (
                path.read_bytes() if path.is_file() else None
            )
            for path in sorted(self.ledger_dir.rglob("*"))
        }

    def _rewrite_events(self, rows: list[dict[str, object]]) -> None:
        previous = "0" * 64
        rewritten: list[dict[str, object]] = []
        for sequence, row in enumerate(rows, start=1):
            payload = {
                **{key: value for key, value in row.items() if key != "event_sha256"},
                "sequence": sequence,
                "previous_event_sha256": previous,
            }
            current = _event(payload)
            rewritten.append(current)
            previous = str(current["event_sha256"])
        (self.ledger_dir / "events.jsonl").write_bytes(
            b"".join(_compact_bytes(row) for row in rewritten)
        )
        (self.ledger_dir / "head.json").write_bytes(
            _compact_bytes(
                {
                    "schema_version": "adaptive-obligation-v2-attempt-ledger-v1",
                    "event_count": len(rewritten),
                    "head_sha256": previous,
                }
            )
        )

    def mutate(self, mutation: str) -> None:
        if mutation == "approval":
            approval = json.loads(self.approval_path.read_bytes())
            approval["audit_note"] = "changed after the R1 attempt"
            self.approval_path.write_bytes(_pretty_bytes(approval))
            return
        if mutation == "anchor":
            path = self.ledger_dir / "anchor.json"
            anchor = json.loads(path.read_bytes())
            anchor["approval_sha256"] = "0" * 64
            path.write_bytes(_compact_bytes(anchor))
            return
        rows = [
            json.loads(line)
            for line in (self.ledger_dir / "events.jsonl").read_bytes().splitlines()
        ]
        if mutation == "event":
            rows[1]["classification"] = "valid"
            self._rewrite_events(rows)
            return
        if mutation == "raw":
            raw_path = next((self.ledger_dir / "raw").iterdir())
            changed = R1_RAW[:-1] + b" "
            raw_path.write_bytes(changed)
            rows[1]["raw_bytes"] = len(changed)
            rows[1]["raw_sha256"] = _sha256(changed)
            self._rewrite_events(rows)
            return
        raise AssertionError(f"unknown mutation: {mutation}")

    def mutate_ledger_inventory(self, mutation: str) -> None:
        if mutation == "completion":
            (self.ledger_dir / "completion.json").write_bytes(b"{}\n")
        elif mutation == "extra_root":
            (self.ledger_dir / "unexpected.txt").write_bytes(b"unexpected")
        elif mutation == "extra_raw":
            (self.ledger_dir / "raw" / "unexpected.completion").write_bytes(b"extra")
        elif mutation == "lock":
            (self.ledger_dir / "mutation.lock").write_bytes(b"locked")
        else:
            raise AssertionError(f"unknown inventory mutation: {mutation}")

    def rewrite_as_retry_attempt(self) -> None:
        anchor = json.loads((self.ledger_dir / "anchor.json").read_bytes())
        retry = anchor["jobs"][0]["attempts"][1]
        assert isinstance(retry, dict)
        rows = [
            json.loads(line)
            for line in (self.ledger_dir / "events.jsonl").read_bytes().splitlines()
        ]
        for row in rows:
            row.update(retry)
        retry_raw_name = f"{R1_JOB_ID}.2.completion"
        rows[1]["raw_path"] = retry_raw_name
        raw_path = next((self.ledger_dir / "raw").iterdir())
        raw_path.rename(raw_path.with_name(retry_raw_name))
        self._rewrite_events(rows)


@pytest.fixture
def r1_failure_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> R1FailureFixture:
    assert len(R1_RAW) == 546
    assert _sha256(R1_RAW) == R1_RAW_SHA256
    jobs = _fixture_jobs()
    preflight_dir = tmp_path / "proposal_preflight"
    preflight_dir.mkdir()
    receipt = {"schema_version": "fixture-proposal-preflight", "job_count": len(jobs)}
    receipt_source = _pretty_bytes(receipt)
    (preflight_dir / "receipt.json").write_bytes(receipt_source)
    (preflight_dir / "jobs.jsonl").write_bytes(
        b"".join(_compact_bytes(job) for job in jobs)
    )
    (preflight_dir / "schema.json").write_bytes(_pretty_bytes(PROPOSAL_SCHEMA))
    (preflight_dir / "prompt.json").write_bytes(_pretty_bytes({"fixture": True}))

    approval_path = tmp_path / "proposal_approval.json"
    approval = {
        "approved": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preflight_sha256": _sha256(receipt_source),
        "primary_call_count": PRIMARY_JOB_COUNT,
        "retry_call_ceiling": PRIMARY_JOB_COUNT,
        "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
        "stage": "proposal",
    }
    approval_source = _pretty_bytes(approval)
    approval_path.write_bytes(approval_source)

    anchor = _build_run_anchor(
        jobs,
        preflight_sha256=_sha256(receipt_source),
        approval_sha256=_sha256(approval_source),
    )
    ledger_dir = tmp_path / "proposal_ledger"
    raw_dir = ledger_dir / "raw"
    raw_dir.mkdir(parents=True)
    (ledger_dir / "anchor.json").write_bytes(_compact_bytes(anchor))
    (ledger_dir / "mutation.lock").write_bytes(b"")
    raw_name = f"{R1_JOB_ID}.1.completion"
    (raw_dir / raw_name).write_bytes(R1_RAW)
    attempt = anchor["jobs"][0]["attempts"][0]  # type: ignore[index]
    started = _event(
        {
            "schema_version": "adaptive-obligation-v2-attempt-ledger-v1",
            "sequence": 1,
            "previous_event_sha256": "0" * 64,
            "state": "started",
            "stage": "proposal",
            "job_id": R1_JOB_ID,
            **attempt,
        }
    )
    terminal = _event(
        {
            "schema_version": "adaptive-obligation-v2-attempt-ledger-v1",
            "sequence": 2,
            "previous_event_sha256": started["event_sha256"],
            "state": "terminal",
            "stage": "proposal",
            "job_id": R1_JOB_ID,
            **attempt,
            "classification": "schema_error",
            "output_token_count": 186,
            "raw_path": raw_name,
            "raw_bytes": len(R1_RAW),
            "raw_sha256": _sha256(R1_RAW),
        }
    )
    (ledger_dir / "events.jsonl").write_bytes(
        _compact_bytes(started) + _compact_bytes(terminal)
    )
    (ledger_dir / "head.json").write_bytes(
        _compact_bytes(
            {
                "schema_version": "adaptive-obligation-v2-attempt-ledger-v1",
                "event_count": 2,
                "head_sha256": terminal["event_sha256"],
            }
        )
    )

    def verify_fixture_preflight(path: Path) -> dict[str, object]:
        return json.loads((path / "receipt.json").read_bytes())

    monkeypatch.setattr(
        incident_module, "verify_proposal_preflight", verify_fixture_preflight
    )
    return R1FailureFixture(preflight_dir, approval_path, ledger_dir)


def test_incident_replays_exact_terminal_r1_failure(
    r1_failure_fixture: R1FailureFixture,
) -> None:
    receipt = build_r1_incident_receipt(**r1_failure_fixture.paths)

    assert receipt["schema_version"] == "adaptive-obligation-v2-proposal-incident-r1"
    assert receipt["status"] == "aborted"
    assert receipt["attempted_job_count"] == 1
    assert receipt["terminal_schema_error_count"] == 1
    assert receipt["uncalled_job_count"] == 47
    assert receipt["reason_code"] == "scope_rationale_length_exceeded"
    assert receipt["observed_rationale_characters"] == 245
    assert receipt["accepted_rationale_maximum"] == 240
    assert receipt["ledger"]["event_count"] == 2  # type: ignore[index]
    assert receipt["raw_completion"] == {
        "path": f"raw/{R1_JOB_ID}.1.completion",
        "bytes": 546,
        "sha256": R1_RAW_SHA256,
    }
    assert receipt["network_call_count"] == 0
    assert receipt["retrieval_call_count"] == 0
    assert receipt["hosted_inference_call_count"] == 0
    assert receipt["paid_call_count"] == 0
    assert receipt["qrels_opened"] is False


@pytest.mark.parametrize("mutation", ["approval", "anchor", "event", "raw"])
def test_incident_rejects_changed_r1_bytes(
    r1_failure_fixture: R1FailureFixture, mutation: str
) -> None:
    r1_failure_fixture.mutate(mutation)
    with pytest.raises(ValueError, match="R1 incident"):
        build_r1_incident_receipt(**r1_failure_fixture.paths)


def test_incident_never_appends_or_seals_r1_ledger(
    r1_failure_fixture: R1FailureFixture,
) -> None:
    before = r1_failure_fixture.snapshot_ledger()
    build_r1_incident_receipt(**r1_failure_fixture.paths)
    assert r1_failure_fixture.snapshot_ledger() == before


@pytest.mark.parametrize("mutation", ["completion", "extra_root", "extra_raw", "lock"])
def test_incident_rejects_noncanonical_failed_ledger_inventory(
    r1_failure_fixture: R1FailureFixture, mutation: str
) -> None:
    r1_failure_fixture.mutate_ledger_inventory(mutation)
    with pytest.raises(ValueError, match="R1 failed ledger"):
        build_r1_incident_receipt(**r1_failure_fixture.paths)


def test_incident_rejects_rewritten_retry_attempt(
    r1_failure_fixture: R1FailureFixture,
) -> None:
    r1_failure_fixture.rewrite_as_retry_attempt()

    with pytest.raises(ValueError, match="R1 failed ledger"):
        build_r1_incident_receipt(**r1_failure_fixture.paths)


def _compound_invalid_245_character_result() -> tuple[bytes, list[str]]:
    allowed_ids = ["a" * 64]
    value = {
        "status": "SUPPORTED",
        "reason_code": "SUPPORTED",
        "o1": {
            "label": "x",
            "scope_rationale": "r" * 245,
            "support_unit_ids": allowed_ids,
        },
    }
    return _compact_bytes(value).rstrip(b"\n"), allowed_ids


def test_sole_defect_check_rejects_compound_invalid_245_character_result() -> None:
    raw, allowed_ids = _compound_invalid_245_character_result()
    with pytest.raises(ValueError, match="sole schema defect"):
        _prove_rationale_length_is_sole_defect(raw, allowed_ids)


def test_publish_incident_writes_only_one_canonical_receipt(
    r1_failure_fixture: R1FailureFixture, tmp_path: Path
) -> None:
    output_dir = tmp_path / "proposal_run_r1_incident"
    receipt = publish_r1_incident(
        **r1_failure_fixture.paths,
        output_dir=output_dir,
    )

    assert {path.name for path in output_dir.iterdir()} == {"receipt.json"}
    assert (output_dir / "receipt.json").read_bytes() == _pretty_bytes(receipt)
    assert verify_r1_incident(
        **r1_failure_fixture.paths,
        output_dir=output_dir,
    ) == receipt


def test_publish_incident_is_create_only(
    r1_failure_fixture: R1FailureFixture, tmp_path: Path
) -> None:
    output_dir = tmp_path / "proposal_run_r1_incident"
    publish_r1_incident(**r1_failure_fixture.paths, output_dir=output_dir)
    original = (output_dir / "receipt.json").read_bytes()

    with pytest.raises(FileExistsError, match="create-only R1 incident"):
        publish_r1_incident(**r1_failure_fixture.paths, output_dir=output_dir)

    assert (output_dir / "receipt.json").read_bytes() == original


def test_verify_incident_rejects_changed_receipt(
    r1_failure_fixture: R1FailureFixture, tmp_path: Path
) -> None:
    output_dir = tmp_path / "proposal_run_r1_incident"
    receipt = publish_r1_incident(**r1_failure_fixture.paths, output_dir=output_dir)
    receipt["status"] = "complete"
    (output_dir / "receipt.json").write_bytes(_pretty_bytes(receipt))

    with pytest.raises(ValueError, match="R1 incident receipt differs"):
        verify_r1_incident(**r1_failure_fixture.paths, output_dir=output_dir)


def test_incident_cli_exposes_only_build_and_verify_actions() -> None:
    actions = set(
        incident_module._parser()._subparsers._group_actions[0].choices
    )
    assert actions == {"build", "verify"}

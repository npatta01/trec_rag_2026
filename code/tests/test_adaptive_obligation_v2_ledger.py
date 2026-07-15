from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_ledger as ledger_module
from trec_rag.adaptive_obligation_v2_ledger import (
    AppendOnlyAttemptLedger,
    AttemptSpec,
    classify_completion,
)
from trec_rag.adaptive_obligation_v2_propose import PROPOSAL_SCHEMA
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
from trec_rag.adaptive_obligation_v2_validate import VALIDATION_SCHEMA


def _anchor() -> dict[str, object]:
    jobs = [
        {
            "job_id": "a" * 64,
            "input_unit_ids": [],
            "attempts": [
                {
                    "attempt_ordinal": 1,
                    "request_sha256": "b" * 64,
                    "max_new_tokens": 256,
                },
                {
                    "attempt_ordinal": 2,
                    "request_sha256": "c" * 64,
                    "max_new_tokens": 512,
                },
            ],
        }
    ]
    return {
        "schema_version": "adaptive-obligation-v2-run-anchor-v1",
        "stage": "proposal",
        "preflight_sha256": "d" * 64,
        "approval_sha256": "e" * 64,
        "job_count": 1,
        "job_inventory_sha256": canonical_sha256(jobs),
        "jobs": jobs,
        "schema": PROPOSAL_SCHEMA,
    }


def _new_ledger(root: Path) -> AppendOnlyAttemptLedger:
    root.parent.mkdir(parents=True, exist_ok=True)
    return AppendOnlyAttemptLedger(
        root,
        expected_anchor=_anchor(),
        create_only=True,
    )


def _reopen_ledger(root: Path) -> AppendOnlyAttemptLedger:
    return AppendOnlyAttemptLedger(
        root,
        expected_anchor=_anchor(),
        create_only=False,
    )


def _attempt() -> AttemptSpec:
    return AttemptSpec(
        stage="proposal",
        job_id="a" * 64,
        attempt_ordinal=1,
        request_sha256="b" * 64,
        max_new_tokens=256,
    )


def _valid_unsupported_bytes() -> bytes:
    return b'{"status":"UNSUPPORTED","reason_code":"NO_ABSTRACT_CHILD","o1":null}'


def test_attempt_start_precedes_model_call(tmp_path: Path) -> None:
    observed: list[dict[str, object]] = []
    ledger = _new_ledger(tmp_path / "ledger")

    def generate(_messages: object, _schema: object, _max_new_tokens: int) -> tuple[bytes, int]:
        observed.extend(ledger.read_events())
        return _valid_unsupported_bytes(), 9

    ledger.run_attempt(
        _attempt(),
        generate,
        schema=PROPOSAL_SCHEMA,
        allowed_support_unit_ids=[],
    )
    assert observed[-1]["state"] == "started"


def test_classify_completion_requires_exact_ceiling_for_truncation() -> None:
    at_ceiling = classify_completion(
        b'{"status":',
        schema=PROPOSAL_SCHEMA,
        output_token_count=256,
        max_new_tokens=256,
    )
    under_ceiling = classify_completion(
        b'{"status":',
        schema=PROPOSAL_SCHEMA,
        output_token_count=255,
        max_new_tokens=256,
    )
    assert at_ceiling["classification"] == "truncated_at_ceiling"
    assert under_ceiling["classification"] == "parse_error"


def test_reopening_rejects_a_crashed_started_attempt(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    ledger = _new_ledger(root)

    def crash(_messages: object, _schema: object, _max_new_tokens: int) -> bytes:
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        ledger.run_attempt(_attempt(), crash)
    with pytest.raises(ValueError, match="incomplete"):
        _reopen_ledger(root)


def test_raw_bytes_are_fsynced_before_classification_and_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path / "ledger")
    real_classify = ledger_module.classify_completion
    observed: list[str] = []

    def classify(raw: bytes, **kwargs: object) -> dict[str, object]:
        if observed:
            return real_classify(raw, **kwargs)  # type: ignore[arg-type]
        raw_path = ledger.raw_dir / f"{'a' * 64}.1.completion"
        assert raw_path.read_bytes() == raw
        assert [row["state"] for row in ledger.read_events()] == ["started"]
        observed.append("raw-before-parse")
        return real_classify(raw, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ledger_module, "classify_completion", classify)
    result = ledger.run_attempt(
        _attempt(),
        lambda _messages, _schema, _ceiling: (_valid_unsupported_bytes(), 9),
        schema=PROPOSAL_SCHEMA,
    )
    assert result["classification"] == "valid"
    assert observed == ["raw-before-parse"]
    assert [row["state"] for row in ledger.read_events()] == ["started", "terminal"]


def test_raw_survives_a_crash_before_parsing_and_reopen_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ledger"
    ledger = _new_ledger(root)
    monkeypatch.setattr(
        ledger_module,
        "classify_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parse crash")),
    )
    with pytest.raises(RuntimeError, match="parse crash"):
        ledger.run_attempt(
            _attempt(),
            lambda _messages, _schema, _ceiling: (_valid_unsupported_bytes(), 9),
            schema=PROPOSAL_SCHEMA,
        )
    assert (root / "raw" / f"{'a' * 64}.1.completion").read_bytes() == (
        _valid_unsupported_bytes()
    )
    with pytest.raises(ValueError, match="incomplete"):
        _reopen_ledger(root)


def _completed_ledger(tmp_path: Path) -> tuple[Path, AppendOnlyAttemptLedger]:
    root = tmp_path / "ledger"
    ledger = _new_ledger(root)
    ledger.run_attempt(
        _attempt(),
        lambda _messages, _schema, _ceiling: (_valid_unsupported_bytes(), 9),
        schema=PROPOSAL_SCHEMA,
    )
    return root, ledger


def test_reopen_of_nonempty_ledger_requires_completion_seal(tmp_path: Path) -> None:
    root, ledger = _completed_ledger(tmp_path)
    with pytest.raises(ValueError, match="completion seal"):
        _reopen_ledger(root)
    ledger.seal_completion()
    assert _reopen_ledger(root).read_events()[-1]["classification"] == "valid"


def test_reopen_rejects_deleted_or_reordered_events(tmp_path: Path) -> None:
    deleted_root, _ledger = _completed_ledger(tmp_path / "deleted")
    lines = (deleted_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (deleted_root / "events.jsonl").write_bytes(b"".join(lines[:1]))
    with pytest.raises(ValueError, match="deletion|head"):
        _reopen_ledger(deleted_root)

    reordered_root, _ledger = _completed_ledger(tmp_path / "reordered")
    lines = (reordered_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (reordered_root / "events.jsonl").write_bytes(b"".join(lines[::-1]))
    with pytest.raises(ValueError, match="hash chain"):
        _reopen_ledger(reordered_root)


def test_reopen_rejects_raw_tampering_and_extra_artifacts(tmp_path: Path) -> None:
    tampered_root, _ledger = _completed_ledger(tmp_path / "tampered")
    raw = tampered_root / "raw" / f"{'a' * 64}.1.completion"
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw completion"):
        _reopen_ledger(tampered_root)

    extra_root, _ledger = _completed_ledger(tmp_path / "extra")
    (extra_root / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="extra"):
        _reopen_ledger(extra_root)


def test_duplicate_attempt_ordinal_is_never_replayed(tmp_path: Path) -> None:
    _root, ledger = _completed_ledger(tmp_path)
    with pytest.raises(ValueError, match="duplicate ordinals"):
        ledger.run_attempt(
            _attempt(),
            lambda _messages, _schema, _ceiling: pytest.fail("must not generate"),
            schema=PROPOSAL_SCHEMA,
        )


def test_attempt_ordinal_and_ceiling_are_coupled() -> None:
    with pytest.raises(ValueError, match="ordinal.*ceiling"):
        AttemptSpec(
            stage="proposal",
            job_id="a" * 64,
            attempt_ordinal=1,
            request_sha256="b" * 64,
            max_new_tokens=512,
        )


def test_attempt_spec_accepts_only_proposal_or_validation_stage() -> None:
    validation = AttemptSpec(
        stage="validation",
        job_id="a" * 64,
        attempt_ordinal=1,
        request_sha256="b" * 64,
        max_new_tokens=256,
    )
    assert validation.stage == "validation"
    with pytest.raises(ValueError, match="stage"):
        AttemptSpec(
            stage="retrieval",
            job_id="a" * 64,
            attempt_ordinal=1,
            request_sha256="b" * 64,
            max_new_tokens=256,
        )


def test_classifier_validates_validation_schema_and_cited_units() -> None:
    unit_id = "f" * 64
    raw = json.dumps(
        {"decision": "SUPPORTED", "support_unit_ids": [unit_id]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    valid = classify_completion(
        raw,
        schema=VALIDATION_SCHEMA,
        output_token_count=12,
        max_new_tokens=256,
        allowed_support_unit_ids=[unit_id],
    )
    escaped = classify_completion(
        raw,
        schema=VALIDATION_SCHEMA,
        output_token_count=12,
        max_new_tokens=256,
        allowed_support_unit_ids=[],
    )
    invalid_code = classify_completion(
        b'{"decision":"MAYBE","support_unit_ids":[]}',
        schema=VALIDATION_SCHEMA,
        output_token_count=8,
        max_new_tokens=256,
        allowed_support_unit_ids=[],
    )
    assert valid["classification"] == "valid"
    assert escaped["classification"] == "semantic_error"
    assert invalid_code["classification"] == "schema_error"


def test_anchor_stage_is_bound_to_its_exact_schema(tmp_path: Path) -> None:
    proposal_schema_under_validation_stage = {**_anchor(), "stage": "validation"}
    with pytest.raises(ValueError, match="anchor differs"):
        AppendOnlyAttemptLedger(
            tmp_path / "wrong-stage-schema",
            expected_anchor=proposal_schema_under_validation_stage,
            create_only=True,
        )


def test_retry_ordinal_requires_a_truncated_primary(tmp_path: Path) -> None:
    ledger = _new_ledger(tmp_path / "ledger")
    retry = AttemptSpec(
        stage="proposal",
        job_id="a" * 64,
        attempt_ordinal=2,
        request_sha256="c" * 64,
        max_new_tokens=512,
    )
    with pytest.raises(ValueError, match="retry.*primary"):
        ledger.run_attempt(
            retry,
            lambda _messages, _schema, _ceiling: pytest.fail("must not generate"),
            schema=PROPOSAL_SCHEMA,
        )


@pytest.mark.parametrize(
    ("raw", "tokens", "classification"),
    [
        (b'{"status":', 256, "truncated_at_ceiling"),
        (b'{"status":', 255, "parse_error"),
        (b'{} {}', 256, "parse_error"),
        (b'{"status":"BAD"}', 256, "schema_error"),
    ],
)
def test_completion_classification_is_exact(
    raw: bytes, tokens: int, classification: str
) -> None:
    result = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=tokens,
        max_new_tokens=256,
    )
    assert result["classification"] == classification


def test_hostile_schema_value_is_classified_instead_of_crashing() -> None:
    raw = (
        b'{"status":"SUPPORTED","reason_code":"SUPPORTED","o1":'
        b'{"label":"abc","scope_rationale":"abc","support_unit_ids":[[]]}}'
    )
    result = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=20,
        max_new_tokens=256,
    )
    assert result["classification"] == "schema_error"


def test_classifier_rejects_a_substituted_schema() -> None:
    result = classify_completion(
        _valid_unsupported_bytes(),
        schema={},
        output_token_count=9,
        max_new_tokens=256,
    )
    assert result["classification"] == "schema_error"
    assert "supplied proposal schema" in str(result["error"])


@pytest.mark.parametrize(
    "suffix",
    [
        b"t",
        b"tr",
        b"tru",
        b"f",
        b"fa",
        b"fal",
        b"fals",
        b"n",
        b"nu",
        b"nul",
        b"1e",
        b"1e+",
        b"1e-",
    ],
)
def test_incomplete_json_recognizes_literal_and_exponent_prefixes(suffix: bytes) -> None:
    result = classify_completion(
        b'{"status":' + suffix,
        schema=PROPOSAL_SCHEMA,
        output_token_count=256,
        max_new_tokens=256,
    )
    assert result["classification"] == "truncated_at_ceiling"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"status":"SUP',
        b'{"status":"escaped\\',
        b'{"status":"escaped\\u12',
        b'{"status":true',
        b'{"status":true,',
        b'{"status":true,"nested":[',
        b'{"status":true,"nested":[1,{"value":null',
        b'{"status":-1.',
        b'{"status":"caf\xc3',
    ],
)
def test_incomplete_json_accepts_only_structurally_valid_whole_prefixes(
    raw: bytes,
) -> None:
    result = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=256,
        max_new_tokens=256,
    )
    assert result["classification"] == "truncated_at_ceiling"


@pytest.mark.parametrize(
    "raw",
    [
        b'{} {"status":tru',
        b'{"status" tru',
        b'{"bad":"\\q","status":tru',
        b'{"bad":01,"status":tru',
        b'{"bad":1e+,"status":tru',
        b'{"bad":true,}',
        b'{"bad":--1',
        b'{"bad":1.e',
        b'{"bad":"line\nbreak',
        b'{"bad":truth',
        b'{"bad":"caf\xc3(',
    ],
)
def test_incomplete_json_rejects_any_earlier_structural_error(raw: bytes) -> None:
    result = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=256,
        max_new_tokens=256,
    )
    assert result["classification"] == "parse_error"


def test_incomplete_json_rejects_invalid_complete_literal_syntax() -> None:
    result = classify_completion(
        b'{"status":truth}',
        schema=PROPOSAL_SCHEMA,
        output_token_count=256,
        max_new_tokens=256,
    )
    assert result["classification"] == "parse_error"


def _reseal_events(root: Path, mutate: object) -> None:
    rows = [json.loads(line) for line in (root / "events.jsonl").read_bytes().splitlines()]
    mutate(rows)
    previous = "0" * 64
    sealed: list[dict[str, object]] = []
    for sequence, row in enumerate(rows, start=1):
        row = {
            **{key: value for key, value in row.items() if key != "event_sha256"},
            "sequence": sequence,
            "previous_event_sha256": previous,
        }
        payload = ledger_module._canonical_bytes(row)
        row["event_sha256"] = ledger_module._sha256(payload)
        previous = str(row["event_sha256"])
        sealed.append(row)
    (root / "events.jsonl").write_bytes(
        b"".join(ledger_module._canonical_bytes(row) for row in sealed)
    )
    (root / "head.json").write_bytes(
        ledger_module._canonical_bytes(
            {
                "schema_version": ledger_module.LEDGER_SCHEMA_VERSION,
                "event_count": len(sealed),
                "head_sha256": previous,
            }
        )
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[1].update({"unexpected": True}),
        lambda rows: rows[1].update({"request_sha256": "f" * 64}),
        lambda rows: rows[1].update({"output_token_count": True}),
        lambda rows: rows[1].update({"classification": "schema_error"}),
    ],
)
def test_reopen_recomputes_exact_terminal_claims(
    tmp_path: Path, mutate: object
) -> None:
    root, _ledger = _completed_ledger(tmp_path)
    _reseal_events(root, mutate)
    with pytest.raises(ValueError, match="event|spec|token|classification"):
        _reopen_ledger(root)


def test_reopen_requires_expected_anchor_and_rejects_wrong_anchor(tmp_path: Path) -> None:
    root, _ledger = _completed_ledger(tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        AppendOnlyAttemptLedger(root, create_only=False)
    wrong = {**_anchor(), "approval_sha256": "0" * 64}
    with pytest.raises(ValueError, match="anchor"):
        AppendOnlyAttemptLedger(root, expected_anchor=wrong, create_only=False)


def test_interprocess_lock_covers_model_call_and_attempt_mutation(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    first = _new_ledger(root)
    second = _reopen_ledger(root)
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def slow(_messages: object, _schema: object, _ceiling: int) -> tuple[bytes, int]:
        entered.set()
        assert release.wait(timeout=5)
        return _valid_unsupported_bytes(), 9

    def run_first() -> None:
        try:
            first.run_attempt(_attempt(), slow, schema=PROPOSAL_SCHEMA)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=5)
    touched: list[str] = []
    with pytest.raises(RuntimeError, match="locked"):
        second.run_attempt(
            _attempt(),
            lambda *_args: touched.append("model"),
            schema=PROPOSAL_SCHEMA,
        )
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    assert touched == []


def test_completed_run_binds_final_head_against_resealed_deletion(tmp_path: Path) -> None:
    root, ledger = _completed_ledger(tmp_path)
    ledger.seal_completion()
    _reseal_events(root, lambda rows: rows.clear())
    (root / "raw" / f"{'a' * 64}.1.completion").unlink()
    with pytest.raises(ValueError, match="completion|final head"):
        _reopen_ledger(root)

from __future__ import annotations

from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_ledger as ledger_module
from trec_rag.adaptive_obligation_v2_ledger import (
    AppendOnlyAttemptLedger,
    AttemptSpec,
    classify_completion,
)
from trec_rag.adaptive_obligation_v2_propose import PROPOSAL_SCHEMA


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
    ledger = AppendOnlyAttemptLedger(tmp_path / "ledger")

    def generate(_messages: object, _schema: object, _max_new_tokens: int) -> bytes:
        observed.extend(ledger.read_events())
        return _valid_unsupported_bytes()

    ledger.run_attempt(_attempt(), generate)
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
    ledger = AppendOnlyAttemptLedger(root)

    def crash(_messages: object, _schema: object, _max_new_tokens: int) -> bytes:
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        ledger.run_attempt(_attempt(), crash)
    with pytest.raises(ValueError, match="incomplete"):
        AppendOnlyAttemptLedger(root)


def test_raw_bytes_are_fsynced_before_classification_and_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = AppendOnlyAttemptLedger(tmp_path / "ledger")
    real_classify = ledger_module.classify_completion
    observed: list[str] = []

    def classify(raw: bytes, **kwargs: object) -> dict[str, object]:
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
    ledger = AppendOnlyAttemptLedger(root)
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
        AppendOnlyAttemptLedger(root)


def _completed_ledger(tmp_path: Path) -> tuple[Path, AppendOnlyAttemptLedger]:
    root = tmp_path / "ledger"
    ledger = AppendOnlyAttemptLedger(root)
    ledger.run_attempt(
        _attempt(),
        lambda _messages, _schema, _ceiling: (_valid_unsupported_bytes(), 9),
        schema=PROPOSAL_SCHEMA,
    )
    return root, ledger


def test_reopen_rejects_deleted_or_reordered_events(tmp_path: Path) -> None:
    deleted_root, _ledger = _completed_ledger(tmp_path / "deleted")
    lines = (deleted_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (deleted_root / "events.jsonl").write_bytes(b"".join(lines[:1]))
    with pytest.raises(ValueError, match="deletion|head"):
        AppendOnlyAttemptLedger(deleted_root)

    reordered_root, _ledger = _completed_ledger(tmp_path / "reordered")
    lines = (reordered_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (reordered_root / "events.jsonl").write_bytes(b"".join(lines[::-1]))
    with pytest.raises(ValueError, match="hash chain"):
        AppendOnlyAttemptLedger(reordered_root)


def test_reopen_rejects_raw_tampering_and_extra_artifacts(tmp_path: Path) -> None:
    tampered_root, _ledger = _completed_ledger(tmp_path / "tampered")
    raw = tampered_root / "raw" / f"{'a' * 64}.1.completion"
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw completion"):
        AppendOnlyAttemptLedger(tampered_root)

    extra_root, _ledger = _completed_ledger(tmp_path / "extra")
    (extra_root / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="extra"):
        AppendOnlyAttemptLedger(extra_root)


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


def test_retry_ordinal_requires_a_truncated_primary(tmp_path: Path) -> None:
    ledger = AppendOnlyAttemptLedger(tmp_path / "ledger")
    retry = AttemptSpec(
        stage="proposal",
        job_id="a" * 64,
        attempt_ordinal=2,
        request_sha256="b" * 64,
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
    raw = b'{"status":"SUPPORTED","reason_code":"SUPPORTED","o1":{"label":"abc","scope_rationale":"abc","support_unit_ids":[[]]}}'
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

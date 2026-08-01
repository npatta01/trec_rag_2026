from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
from trec_rag import continuation


def write_ticket(cache_dir: Path, **overrides: object) -> dict[str, object]:
    ticket = {
        "ticket": "abc123",
        "not_before_unix": time.time() - 5,
        "retry_after_seconds": 30,
        "failed_attempt_id": "attempt-1",
        "cache_key": "0123456789abcdef",
        "query": "migrant worker protections",
        "index": "climbmix-400b",
        "index_url": "http://example.invalid/v1/climbmix-400b/search",
        "hits": 10,
        **overrides,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / continuation.TICKET_NAME).write_text(
        json.dumps(ticket), encoding="utf-8"
    )
    return ticket


def test_describe_reports_a_clear_retriever(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    report = continuation.describe(tmp_path)

    assert report == {"blocked": False, "state": "clear"}


def test_describe_names_the_blocked_query_without_being_told_it(
    tmp_path: Path,
) -> None:
    """An interrupted run does not need to remember what was in flight."""
    write_ticket(tmp_path)

    report = continuation.describe(tmp_path)

    assert report["blocked"] is True
    assert report["state"] == "pending"
    assert report["query"] == "migrant worker protections"
    assert report["ticket"] == "abc123"
    assert report["eligible_now"] is True
    assert report["seconds_until_eligible"] == 0.0


def test_describe_reports_time_left_before_a_retry_is_allowed(
    tmp_path: Path,
) -> None:
    write_ticket(tmp_path, not_before_unix=time.time() + 120)

    report = continuation.describe(tmp_path)

    assert report["eligible_now"] is False
    assert 0 < float(report["seconds_until_eligible"]) <= 120


def test_discard_unblocks_without_making_a_hosted_call(tmp_path: Path) -> None:
    write_ticket(tmp_path)

    report = continuation.discard(tmp_path)

    assert report["discarded"] == [continuation.TICKET_NAME]
    assert report["state"] == "clear"
    assert not (tmp_path / continuation.TICKET_NAME).exists()


def test_discard_also_clears_a_half_finished_continuation(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / continuation.IN_PROGRESS_NAME).write_text("{}", encoding="utf-8")

    report = continuation.discard(tmp_path)

    assert report["discarded"] == [continuation.IN_PROGRESS_NAME]
    assert report["state"] == "clear"


def test_resume_refuses_before_the_not_before_time(tmp_path: Path) -> None:
    write_ticket(tmp_path, not_before_unix=time.time() + 600)

    with pytest.raises(SystemExit, match="not-before"):
        continuation.resume(tmp_path)

    assert (tmp_path / continuation.TICKET_NAME).exists(), "the ticket survives"


def test_resume_refuses_when_nothing_is_pending(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no pending continuation"):
        continuation.resume(tmp_path)

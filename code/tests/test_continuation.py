from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event, Thread
import time

from filelock import FileLock
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
    ticket_name = (
        continuation.TOPIC_TICKET_NAME
        if cache_dir.parent.name == continuation.TOPIC_STATE_DIRNAME
        else continuation.TICKET_NAME
    )
    (cache_dir / ticket_name).write_text(
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
    (tmp_path / continuation.IN_PROGRESS_NAME).write_text(
        json.dumps(
            {
                "lease_state": "awaiting_continuation",
                "lease_expires_unix": 0,
            }
        ),
        encoding="utf-8",
    )

    report = continuation.discard(tmp_path)

    assert report["discarded"] == [continuation.IN_PROGRESS_NAME]
    assert report["state"] == "clear"


@pytest.mark.parametrize("lease_state", ["active_recovery", "active_transport"])
def test_discard_refuses_another_process_unexpired_active_lease(
    tmp_path: Path,
    lease_state: str,
) -> None:
    state = tmp_path / continuation.TOPIC_STATE_DIRNAME / "topic-a"
    write_ticket(state, topic_id="topic-a")
    (state / continuation.TOPIC_IN_PROGRESS_NAME).write_text(
        json.dumps(
            {
                "owner": f"{os.getpid() + 1}-other-process",
                "attempt_id": "attempt-a",
                "lease_state": lease_state,
                "lease_expires_unix": time.time() + 300,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="active .* lease"):
        continuation.discard(tmp_path, topic_id="topic-a")

    assert (state / continuation.TOPIC_TICKET_NAME).exists()
    assert (state / continuation.TOPIC_IN_PROGRESS_NAME).exists()


@pytest.mark.parametrize("lease_state", ["active_recovery", "active_transport"])
def test_discard_refuses_current_process_unexpired_active_lease(
    tmp_path: Path,
    lease_state: str,
) -> None:
    state = tmp_path / continuation.TOPIC_STATE_DIRNAME / "topic-a"
    write_ticket(state, topic_id="topic-a")
    (state / continuation.TOPIC_IN_PROGRESS_NAME).write_text(
        json.dumps(
            {
                "owner": f"{os.getpid()}-this-process",
                "attempt_id": "attempt-a",
                "lease_state": lease_state,
                "lease_expires_unix": time.time() + 300,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="active .* lease"):
        continuation.discard(tmp_path, topic_id="topic-a")

    assert (state / continuation.TOPIC_TICKET_NAME).exists()
    assert (state / continuation.TOPIC_IN_PROGRESS_NAME).exists()


def test_discard_fails_closed_for_unknown_lease_state(tmp_path: Path) -> None:
    state = tmp_path / continuation.TOPIC_STATE_DIRNAME / "topic-a"
    write_ticket(state, topic_id="topic-a")
    (state / continuation.TOPIC_IN_PROGRESS_NAME).write_text(
        json.dumps(
            {
                "owner": "worker",
                "attempt_id": "attempt-a",
                "lease_state": "future_state",
                "lease_expires_unix": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unknown"):
        continuation.discard(tmp_path, topic_id="topic-a")

    assert (state / continuation.TOPIC_TICKET_NAME).exists()
    assert (state / continuation.TOPIC_IN_PROGRESS_NAME).exists()


def test_discard_fails_closed_for_unknown_completion_marker(tmp_path: Path) -> None:
    state = tmp_path / continuation.TOPIC_STATE_DIRNAME / "topic-a"
    state.mkdir(parents=True)
    (state / continuation.TOPIC_COMPLETION_NAME).write_text(
        json.dumps(
            {
                "schema_version": "future-completion-v2",
                "state": "future_state",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unknown completion marker"):
        continuation.discard(tmp_path, topic_id="topic-a")


def test_discard_waits_for_topic_state_lock_before_removing_state(
    tmp_path: Path,
) -> None:
    state = tmp_path / continuation.TOPIC_STATE_DIRNAME / "topic-a"
    write_ticket(state, topic_id="topic-a")
    lock = FileLock(str(state / "state.lock"))
    acquired = Event()
    release = Event()
    finished = Event()
    result: dict[str, object] = {}

    def hold_lock() -> None:
        with lock:
            acquired.set()
            release.wait(timeout=5)

    def discard() -> None:
        result["report"] = continuation.discard(tmp_path, topic_id="topic-a")
        finished.set()

    holder = Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(timeout=5)
    worker = Thread(target=discard)
    worker.start()
    assert not finished.wait(timeout=0.2)

    release.set()
    holder.join(timeout=5)
    worker.join(timeout=5)

    assert not holder.is_alive()
    assert not worker.is_alive()
    assert result["report"] == {
        "discarded": [continuation.TOPIC_TICKET_NAME],
        "state": "clear",
        "topic_id": "topic-a",
    }


def test_resume_refuses_before_the_not_before_time(tmp_path: Path) -> None:
    write_ticket(tmp_path, not_before_unix=time.time() + 600)

    with pytest.raises(SystemExit, match="not-before"):
        continuation.resume(tmp_path)

    assert (tmp_path / continuation.TICKET_NAME).exists(), "the ticket survives"


def test_resume_refuses_when_nothing_is_pending(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no pending continuation"):
        continuation.resume(tmp_path)


def test_describe_reports_all_pending_topics_instead_of_collapsing_them(
    tmp_path: Path,
) -> None:
    for topic_id in ("topic-a", "topic-b"):
        state = tmp_path / continuation.TOPIC_STATE_DIRNAME / topic_id
        write_ticket(state, topic_id=topic_id)

    report = continuation.describe(tmp_path)

    assert report["state"] == "multiple_pending"
    assert report["topics"] == ["topic-a", "topic-b"]


@pytest.mark.parametrize("mutation", ["resume", "discard"])
def test_mutations_require_topic_id_when_multiple_topics_are_pending(
    tmp_path: Path, mutation: str
) -> None:
    for topic_id in ("topic-a", "topic-b"):
        state = tmp_path / continuation.TOPIC_STATE_DIRNAME / topic_id
        write_ticket(state, topic_id=topic_id)

    with pytest.raises(SystemExit, match="topic id"):
        getattr(continuation, mutation)(tmp_path)


def test_resume_preserves_ticket_corpus_epoch_exactly(tmp_path: Path, monkeypatch) -> None:
    ticket = write_ticket(tmp_path, corpus_epoch="operator-epoch-17")
    observed = {}

    class Retriever:
        def __init__(self, *args, **kwargs):
            observed["epoch"] = kwargs["corpus_epoch"]

        def retrieve(self, _query):
            return []

    monkeypatch.setattr(continuation, "PyseriniRemoteRetriever", Retriever)
    monkeypatch.setattr(continuation, "request_cache_key", lambda *args, **kwargs: ticket["cache_key"])

    continuation.resume(tmp_path)

    assert observed["epoch"] == "operator-epoch-17"

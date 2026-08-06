from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import pytest

from trec_rag.topic_dispatch import (
    TopicDispatchError,
    TopicDispatchIntegrityError,
    TopicJob,
    TopicJobReceipt,
    dispatch_topics,
    publish_topic_receipt,
    read_topic_receipt,
)


def _job(
    root: Path,
    topic_id: str,
    *,
    run_id: str = "run-a",
    offline_cache_only: bool = False,
) -> TopicJob:
    config_bytes = b"schema_version: test\n"
    return TopicJob(
        topic_id=topic_id,
        run_id=run_id,
        config_path=(root / "config.yaml").resolve(),
        config_bytes=config_bytes,
        config_sha256=sha256(config_bytes).hexdigest(),
        topic_root=(root / run_id / topic_id).resolve(),
        offline_cache_only=offline_cache_only,
    )


def _projection_path(job: TopicJob) -> Path:
    return job.topic_root / "canonical" / "retrieval-projection-manifest.json"


def _write_projection(job: TopicJob, payload: bytes) -> str:
    path = _projection_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _receipt(
    job: TopicJob,
    payload: bytes,
    *,
    status: str = "complete",
    stopping_reason: str = "coverage_sufficient",
) -> TopicJobReceipt:
    return TopicJobReceipt(
        topic_id=job.topic_id,
        projection_manifest_sha256=_write_projection(job, payload),
        status=status,
        stopping_reason=stopping_reason,
    )


def _overlap_worker(job: TopicJob) -> TopicJobReceipt:
    job.topic_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    (job.topic_root / "started").write_text(str(started), encoding="utf-8")
    sibling = "topic-a" if job.topic_id == "topic-b" else "topic-b"
    sibling_marker = job.topic_root.parent / sibling / "started"
    deadline = time.monotonic() + 5.0
    while not sibling_marker.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("other topic did not start concurrently")
        time.sleep(0.01)
    time.sleep(0.1)
    ended = time.monotonic()
    (job.topic_root / "interval.json").write_text(
        json.dumps({"start": started, "end": ended}),
        encoding="utf-8",
    )
    return _receipt(job, f"projection:{job.topic_id}".encode("utf-8"))


def _sometimes_failing_worker(job: TopicJob) -> TopicJobReceipt:
    if job.topic_id == "topic-fails":
        raise RuntimeError("deliberate worker failure")
    time.sleep(0.05)
    return _receipt(job, f"projection:{job.topic_id}".encode("utf-8"))


def _interval(job: TopicJob) -> tuple[float, float]:
    value = json.loads((job.topic_root / "interval.json").read_text(encoding="utf-8"))
    return float(value["start"]), float(value["end"])


def test_dispatch_runs_two_disjoint_topics_concurrently_and_returns_source_order(
    tmp_path: Path,
) -> None:
    jobs = (_job(tmp_path, "topic-b"), _job(tmp_path, "topic-a"))

    receipts = dispatch_topics(jobs, _overlap_worker, max_workers=2)

    assert [row.topic_id for row in receipts] == ["topic-b", "topic-a"]
    b_start, b_end = _interval(jobs[0])
    a_start, a_end = _interval(jobs[1])
    assert max(a_start, b_start) < min(a_end, b_end)


def test_dispatch_skips_sealed_topic_before_worker_construction(tmp_path: Path) -> None:
    job = _job(tmp_path, "topic-sealed")
    expected = _receipt(job, b"sealed")
    publish_topic_receipt(job, expected)

    def poison_worker(_: TopicJob) -> TopicJobReceipt:
        raise AssertionError("worker constructed for resumed topic")

    assert dispatch_topics((job,), poison_worker, max_workers=2) == (expected,)


def test_offline_dispatch_does_not_reuse_an_online_mode_receipt(tmp_path: Path) -> None:
    """Catches an online seal suppressing the required offline cache validation."""
    online = _job(tmp_path, "topic-a")
    online_receipt = _receipt(online, b"projection")
    publish_topic_receipt(online, online_receipt)
    offline = _job(tmp_path, "topic-a", offline_cache_only=True)
    calls: list[str] = []

    def offline_worker(job: TopicJob) -> TopicJobReceipt:
        calls.append(job.topic_id)
        return TopicJobReceipt(
            topic_id=job.topic_id,
            projection_manifest_sha256=online_receipt.projection_manifest_sha256,
            status="complete",
            stopping_reason="coverage_sufficient",
        )

    assert dispatch_topics((offline,), offline_worker, max_workers=1) == (
        online_receipt,
    )
    assert calls == ["topic-a"]
    assert read_topic_receipt(online) == online_receipt
    assert read_topic_receipt(offline) == online_receipt


def test_topic_job_rejects_config_bytes_that_do_not_match_the_pinned_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="config_sha256"):
        TopicJob(
            topic_id="topic-a",
            run_id="run-a",
            config_path=(tmp_path / "config.yaml").resolve(),
            config_bytes=b"changed",
            config_sha256="0" * 64,
            topic_root=(tmp_path / "run-a" / "topic-a").resolve(),
        )


def test_sealed_receipt_is_bound_to_the_exact_config_bytes(tmp_path: Path) -> None:
    job = _job(tmp_path, "topic-a")
    publish_topic_receipt(job, _receipt(job, b"projection"))
    changed_bytes = b"schema_version: changed\n"
    changed_job = TopicJob(
        topic_id=job.topic_id,
        run_id=job.run_id,
        config_path=job.config_path,
        config_bytes=changed_bytes,
        config_sha256=sha256(changed_bytes).hexdigest(),
        topic_root=job.topic_root,
    )

    with pytest.raises(TopicDispatchIntegrityError, match="config identity"):
        read_topic_receipt(changed_job)


def test_dispatch_skips_sealed_incomplete_topic(tmp_path: Path) -> None:
    job = _job(tmp_path, "topic-incomplete")
    expected = _receipt(
        job,
        b"partial",
        status="incomplete",
        stopping_reason="budget_exhausted",
    )
    publish_topic_receipt(job, expected)

    assert dispatch_topics(
        (job,),
        lambda _: pytest.fail("sealed incomplete topic was rerun"),
        max_workers=1,
    ) == (expected,)


def test_dispatch_accepts_sealed_evidence_validation_failure(tmp_path: Path) -> None:
    job = _job(tmp_path, "topic-evidence-validation-failed")
    expected = _receipt(
        job,
        b"partial-evidence",
        status="incomplete",
        stopping_reason="evidence_validation_failed",
    )
    publish_topic_receipt(job, expected)

    assert dispatch_topics(
        (job,),
        lambda _: pytest.fail("sealed incomplete topic was rerun"),
        max_workers=1,
    ) == (expected,)


def test_new_run_identity_uses_a_new_topic_namespace(tmp_path: Path) -> None:
    old_job = _job(tmp_path, "topic-a", run_id="run-old")
    old_receipt = _receipt(old_job, b"old")
    publish_topic_receipt(old_job, old_receipt)
    new_job = _job(tmp_path, "topic-a", run_id="run-new")

    new_receipt = dispatch_topics(
        (new_job,), _sometimes_failing_worker, max_workers=1
    )[0]

    assert new_receipt.projection_manifest_sha256 != old_receipt.projection_manifest_sha256
    assert read_topic_receipt(old_job) == old_receipt


def test_duplicate_byte_identical_publication_is_idempotent(tmp_path: Path) -> None:
    job = _job(tmp_path, "topic-a")
    receipt = _receipt(job, b"same")

    assert publish_topic_receipt(job, receipt) == receipt
    assert publish_topic_receipt(job, receipt) == receipt
    assert read_topic_receipt(job) == receipt


def test_conflicting_publication_for_same_run_and_topic_fails_closed(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, "topic-a")
    publish_topic_receipt(job, _receipt(job, b"first"))
    conflicting = _receipt(job, b"second")

    with pytest.raises(TopicDispatchIntegrityError, match="conflicting topic receipt"):
        publish_topic_receipt(job, conflicting)


def test_worker_failure_preserves_other_completed_topic_receipt(tmp_path: Path) -> None:
    good = _job(tmp_path, "topic-good")
    bad = _job(tmp_path, "topic-fails")

    with pytest.raises(TopicDispatchError, match="topic-fails"):
        dispatch_topics((good, bad), _sometimes_failing_worker, max_workers=2)

    assert read_topic_receipt(good).topic_id == "topic-good"
    assert read_topic_receipt(bad, missing_ok=True) is None


@pytest.mark.parametrize("max_workers", [0, -1, True])
def test_dispatch_rejects_invalid_worker_count(tmp_path: Path, max_workers: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        dispatch_topics((), _sometimes_failing_worker, max_workers=max_workers)  # type: ignore[arg-type]

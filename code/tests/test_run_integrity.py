from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from trec_rag.run_integrity import (
    ExclusiveFileLock,
    ThreadSafeJsonStatusWriter,
    validate_or_repair_final_jsonl,
)


def test_thread_safe_status_writer_never_leaves_partial_json(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    writer = ThreadSafeJsonStatusWriter(path)
    threads = [
        threading.Thread(target=writer.write, args=({"state": f"heartbeat-{index}"},))
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert json.loads(path.read_text(encoding="utf-8"))["state"].startswith(
        "heartbeat-"
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_joined_heartbeat_cannot_overwrite_final_status(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    writer = ThreadSafeJsonStatusWriter(path)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(0.001):
            writer.write({"state": "running"})

    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.start()
    writer.write({"state": "running"})
    stop.set()
    heartbeat_thread.join()
    writer.write({"state": "completed"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "completed"}


def test_exclusive_file_lock_rejects_second_writer(tmp_path: Path) -> None:
    path = tmp_path / ".writer.lock"
    first = ExclusiveFileLock(path, {"owner": "first"})
    second = ExclusiveFileLock(path, {"owner": "second"})
    first.acquire()

    with pytest.raises(RuntimeError, match="active writer marker"):
        second.acquire()

    first.release()
    second.acquire()
    second.release()


def test_resume_repairs_only_torn_final_fragment(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b'{"score": 1.0}\n{"score": 2')

    result = validate_or_repair_final_jsonl(path, archive_dir=tmp_path / "archive")

    assert result["repaired"] is True
    assert path.read_bytes() == b'{"score": 1.0}\n'
    archive_path = Path(str(result["fragment_archive"]))
    assert archive_path.read_bytes() == b'{"score": 2'


def test_resume_rejects_invalid_earlier_jsonl_row(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    original = b'{"score": 1.0}\nnot-json\n{"score": 2'
    path.write_bytes(original)

    with pytest.raises(ValueError, match="invalid earlier JSONL row"):
        validate_or_repair_final_jsonl(path, archive_dir=tmp_path / "archive")

    assert path.read_bytes() == original
    assert not (tmp_path / "archive").exists()

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import fcntl
from threading import Event, Thread

from trec_rag.agentic_retrieval_collector import (
    COLLECTOR_JOURNAL_FILENAME,
    WAVE_RECEIPTS_FILENAME,
    IncrementalCollector,
)


@dataclass(frozen=True)
class FakePlan:
    planned_topic_ids: tuple[str, ...]
    plan_sha256: str = "p" * 64


class FakeHF:
    def __init__(self, listing: list[dict[str, object]]) -> None:
        self.listing = listing
        self.download_calls: list[tuple[str, Path]] = []
        self.payloads: dict[str, tuple[bytes, bytes]] = {}

    def list(self, prefix: str) -> list[dict[str, object]]:
        return list(self.listing)

    def download(self, prefix: str, destination: Path) -> None:
        self.download_calls.append((prefix, destination))
        topic_id = prefix.rstrip("/").split("/")[-1]
        archive, marker = self.payloads[topic_id]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bundle.tar.zst").write_bytes(archive)
        (destination / "bundle-complete.json").write_bytes(marker)


def _marker(topic_id: str, archive: bytes, plan_sha256: str = "p" * 64) -> bytes:
    return json.dumps(
        {
            "archive_bytes": len(archive),
            "archive_sha256": sha256(archive).hexdigest(),
            "bundle_schema": "agentic_retrieval_topic_bundle_v1",
            "run_plan_sha256": plan_sha256,
            "topic_id": topic_id,
            "topic_seal_sha256": "s" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"


def _remote_listing(*topic_ids: str, include_marker: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topic_id in topic_ids:
        rows.append({"path": f"run/{topic_id}/bundle.tar.zst", "type": "file"})
        if include_marker:
            rows.append({"path": f"run/{topic_id}/bundle-complete.json", "type": "file"})
    return rows


def _collector(
    tmp_path: Path,
    hf: FakeHF,
    *,
    plan: FakePlan | None = None,
    verify=None,
    importer=None,
    export=None,
    expected_topic_ids=None,
) -> IncrementalCollector:
    selected = plan or FakePlan(("rag2026-14", "rag2026-37"))
    return IncrementalCollector(
        bucket_prefix="run",
        plan=selected,
        staging_root=tmp_path / "staging",
        destination_run_dir=tmp_path / "run-state",
        transport=hf,
        verify_bundle=verify,
        import_bundle=importer,
        export_fn=export,
        expected_topic_ids=expected_topic_ids,
    )


def _successful_bundle_fakes(hf: FakeHF, plan: FakePlan):
    imported: list[str] = []

    def verify(archive: Path, marker: Path, expected_plan_sha256: str):
        payload = json.loads(marker.read_text())
        assert payload["run_plan_sha256"] == expected_plan_sha256
        assert payload["archive_sha256"] == sha256(archive.read_bytes()).hexdigest()
        return {"topic_id": payload["topic_id"], "seal_sha256": payload["topic_seal_sha256"]}

    def importer(archive: Path, marker: Path, destination: Path):
        payload = json.loads(marker.read_text())
        topic_id = payload["topic_id"]
        topic_dir = destination / "topics" / topic_id
        topic_dir.mkdir(parents=True, exist_ok=True)
        (topic_dir / "topic_projection_manifest.json").write_text(
            json.dumps({"topic_id": topic_id, "seal_sha256": payload["topic_seal_sha256"]})
        )
        imported.append(topic_id)
        return {"topic_id": topic_id}

    return verify, importer, imported


def test_exact_topic_components_do_not_collide(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-144"))
    collector = _collector(tmp_path, hf, plan=plan)

    receipt = collector.run_once()

    assert receipt.missing == ("rag2026-14",)
    assert receipt.imported == ()
    assert receipt.failed == ("rag2026-144",)
    assert hf.download_calls[0][0] == "run/rag2026-144"


def test_marker_last_makes_bundle_eligible_and_verifies_offline(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, imported = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    receipt = collector.run_once()

    assert receipt.imported == ("rag2026-14",)
    assert imported == ["rag2026-14"]
    assert len(hf.download_calls) == 1


def test_successful_download_is_preserved_after_import(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    collector.run_once()

    downloaded = hf.download_calls[0][1]
    assert (downloaded / "bundle.tar.zst").read_bytes() == archive
    assert (downloaded / "bundle-complete.json").is_file()


def test_archive_without_completion_marker_remains_missing(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14", include_marker=False))
    collector = _collector(tmp_path, hf, plan=plan)

    receipt = collector.run_once()

    assert receipt.missing == ("rag2026-14",)
    assert not hf.download_calls


def test_identical_topic_is_present_on_restart_without_duplicate_import(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, imported = _successful_bundle_fakes(hf, plan)
    first = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)
    first.run_once()

    second = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)
    receipt = second.run_once()

    assert receipt.present == ("rag2026-14",)
    assert receipt.imported == ()
    assert imported == ["rag2026-14"]
    assert len(hf.download_calls) == 1


def test_rejects_and_quarantines_malformed_or_foreign_bundle(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14", "rag2026-144"))
    valid = b"bundle-14"
    foreign = b"foreign"
    hf.payloads.update(
        {
            "rag2026-14": (valid, _marker("rag2026-14", valid)),
            "rag2026-144": (foreign, b"not-json\n"),
        }
    )
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    receipt = collector.run_once()

    assert receipt.imported == ("rag2026-14",)
    assert receipt.rejected == ("rag2026-144",)
    assert list((tmp_path / "staging" / "quarantine").iterdir())


def test_one_topic_failure_does_not_abort_other_topics(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14", "rag2026-37"))
    hf = FakeHF(_remote_listing("rag2026-14", "rag2026-37"))
    for topic_id in plan.planned_topic_ids:
        archive = topic_id.encode()
        hf.payloads[topic_id] = (archive, _marker(topic_id, archive))

    def verify(archive: Path, marker: Path, expected_plan_sha256: str):
        topic_id = json.loads(marker.read_text())["topic_id"]
        if topic_id == "rag2026-14":
            raise ValueError("bad topic")
        return {"topic_id": topic_id}

    verify_ok, importer, imported = _successful_bundle_fakes(hf, plan)
    def verify_mixed(archive: Path, marker: Path, expected_plan_sha256: str):
        if json.loads(marker.read_text())["topic_id"] == "rag2026-14":
            return verify(archive, marker, expected_plan_sha256)
        return verify_ok(archive, marker, expected_plan_sha256)

    collector = _collector(tmp_path, hf, plan=plan, verify=verify_mixed, importer=importer)
    receipt = collector.run_once()

    assert receipt.rejected == ("rag2026-14",)
    assert receipt.imported == ("rag2026-37",)
    assert imported == ["rag2026-37"]


def test_wave_receipts_are_append_only_and_report_all_categories(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14", "rag2026-37", "rag2026-55"))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    first = collector.run_once()
    second = collector.run_once()

    assert first.imported == ("rag2026-14",)
    assert first.missing == ("rag2026-37", "rag2026-55")
    assert second.present == ("rag2026-14",)
    lines = (tmp_path / "staging" / WAVE_RECEIPTS_FILENAME).read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["wave"] for line in lines] == [1, 2]
    assert (tmp_path / "staging" / COLLECTOR_JOURNAL_FILENAME).exists()


def test_wave_number_allocation_is_serialized_across_two_collectors(tmp_path: Path, monkeypatch) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    first = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)
    second = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)
    barrier = Event()
    original_first = first._next_wave
    original_second = second._next_wave

    def delayed_first():
        barrier.wait(2)
        return original_first()

    def delayed_second():
        barrier.set()
        return original_second()

    monkeypatch.setattr(first, "_next_wave", delayed_first)
    monkeypatch.setattr(second, "_next_wave", delayed_second)
    threads = [Thread(target=collector.run_once) for collector in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    rows = [json.loads(line) for line in (tmp_path / "staging" / WAVE_RECEIPTS_FILENAME).read_text().splitlines()]

    assert sorted(row["wave"] for row in rows) == [1, 2]


def test_receipt_records_plan_archive_marker_and_seal_digests(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    marker = _marker("rag2026-14", archive)
    hf.payloads["rag2026-14"] = (archive, marker)
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    collector.run_once()
    row = json.loads((tmp_path / "staging" / WAVE_RECEIPTS_FILENAME).read_text())
    digests = row["topic_digests"]["rag2026-14"]

    assert digests == {
        "plan_sha256": plan.plan_sha256,
        "archive_sha256": sha256(archive).hexdigest(),
        "marker_sha256": sha256(marker).hexdigest(),
        "topic_seal_sha256": "s" * 64,
    }


def test_import_requires_an_installed_topic_seal(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, _, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=lambda *_: None)

    receipt = collector.run_once()

    assert receipt.failed == ("rag2026-14",)
    assert receipt.imported == ()


def test_traversal_listing_never_constructs_a_remote_download_prefix(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(
        [
            {"path": "run/../rag2026-14/bundle.tar.zst", "type": "file"},
            {"path": "run/../rag2026-14/bundle-complete.json", "type": "file"},
        ]
    )
    collector = _collector(tmp_path, hf, plan=plan)

    receipt = collector.run_once()

    assert not hf.download_calls
    assert receipt.rejected == ("..",)
    assert receipt.missing == ("rag2026-14",)


def test_stale_export_receipt_does_not_suppress_complete_export(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14", "rag2026-37"))
    hf = FakeHF(_remote_listing(*plan.planned_topic_ids))
    for topic_id in plan.planned_topic_ids:
        archive = topic_id.encode()
        hf.payloads[topic_id] = (archive, _marker(topic_id, archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    exports: list[object] = []
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer, export=exports.append)
    (tmp_path / "staging" / "collector-export-receipt.json").write_text("partial\n")

    collector.run_once()

    assert len(exports) == 1


def test_verification_happens_before_shared_import_export_lock(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    lock_path = tmp_path / "run-state" / ".agentic-retrieval-export.lock"

    def verify(archive_path: Path, marker_path: Path, expected_plan_sha256: str):
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return {"topic_id": "rag2026-14", "topic_seal_sha256": "s" * 64}

    _, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    collector.run_once()


def test_incomplete_cohort_does_not_call_export(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14", "rag2026-37"))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    exports: list[object] = []
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer, export=exports.append)

    receipt = collector.run_once()

    assert not receipt.complete
    assert exports == []


def test_no_export_until_complete_cohort_and_export_runs_once(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14", "rag2026-37"))
    hf = FakeHF(_remote_listing(*plan.planned_topic_ids))
    for topic_id in plan.planned_topic_ids:
        archive = topic_id.encode()
        hf.payloads[topic_id] = (archive, _marker(topic_id, archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    exports: list[object] = []
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer, export=exports.append)

    collector.run_once()

    assert len(exports) == 1
    assert tuple(exports[0]["topic_ids"]) == plan.planned_topic_ids


def test_import_and_export_share_a_lock(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    entered = Event()
    release = Event()

    def export(_status):
        entered.set()
        assert release.wait(2)

    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer, export=export)
    second = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer, export=export)
    thread = Thread(target=collector.run_once)
    second_done = Event()

    def run_second() -> None:
        second.run_once()
        second_done.set()

    second_thread = Thread(target=run_second)
    thread.start()
    assert entered.wait(2)
    second_thread.start()
    assert not second_done.wait(0.1)
    release.set()
    thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not thread.is_alive()
    assert second_done.is_set()


def test_watch_stops_when_cohort_is_complete(tmp_path: Path) -> None:
    plan = FakePlan(("rag2026-14",))
    hf = FakeHF(_remote_listing("rag2026-14"))
    archive = b"bundle-14"
    hf.payloads["rag2026-14"] = (archive, _marker("rag2026-14", archive))
    verify, importer, _ = _successful_bundle_fakes(hf, plan)
    collector = _collector(tmp_path, hf, plan=plan, verify=verify, importer=importer)

    receipts = list(collector.watch(max_cycles=3))

    assert len(receipts) == 1
    assert receipts[0].imported == ("rag2026-14",)

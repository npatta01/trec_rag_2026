from __future__ import annotations

import json
from hashlib import sha256
import io
from pathlib import Path
from typing import Any
import zipfile

import pytest

from trec_rag.agentic_retrieval_collector import (
    COLLECTOR_JOURNAL_FILENAME,
    EXPORT_RECEIPT_FILENAME,
    WAVE_RECEIPTS_FILENAME,
    IncrementalCollector,
)
from trec_rag.agentic_retrieval_export import (
    EXPORT_MANIFEST_FILENAME,
    GENERATION_HANDOFF_FILENAME,
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
    load_agentic_retrieval_export,
)
from trec_rag.agentic_retrieval_shard_bundle import (
    TOPIC_SEAL_MEMBER,
    import_topic_bundle,
    pack_topic,
)
from trec_rag.agentic_run_state import install_run_plan, load_run_plan
import trec_rag.agentic_retrieval_shard_bundle as shard_bundle
import trec_rag.agentic_retrieval_export as retrieval_export
import trec_rag.competition_agentic_worker as worker
from trec_rag.competition_agentic_retrieval import initialize_agentic_run
from trec_rag.generation_handoff import load_generation_handoff

from test_competition_agentic_retrieval import (
    SECRETS,
    _binding,
    _success,
    _workspace,
)


class OfflineHF:
    """A marker-last, local-only stand-in for the private HF bucket."""

    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, bytes | None]] = {}
        self.downloads: list[str] = []

    def publish_archive(self, topic_id: str, body: bytes) -> None:
        self.payloads.setdefault(topic_id, {})["archive"] = body

    def publish_marker(self, topic_id: str, body: bytes) -> None:
        self.payloads.setdefault(topic_id, {})["marker"] = body

    def list(self, prefix: str) -> list[dict[str, object]]:
        del prefix
        rows: list[dict[str, object]] = []
        for topic_id in sorted(self.payloads):
            payload = self.payloads[topic_id]
            if payload.get("archive") is not None:
                rows.append({"path": f"run/{topic_id}/bundle.tar.zst", "type": "file"})
            if payload.get("marker") is not None:
                rows.append({"path": f"run/{topic_id}/bundle-complete.json", "type": "file"})
        return rows

    def download(self, prefix: str, destination: Path) -> None:
        topic_id = prefix.rstrip("/").split("/")[-1]
        payload = self.payloads[topic_id]
        archive = payload.get("archive")
        marker = payload.get("marker")
        if archive is None or marker is None:
            raise RuntimeError("offline bucket object is not complete")
        self.downloads.append(topic_id)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bundle.tar.zst").write_bytes(archive)
        (destination / "bundle-complete.json").write_bytes(marker)


def _run_worker(config: Path, plan_body: bytes, topic_id: str):
    return worker._run_agentic_worker(
        config,
        plan_body=plan_body,
        assigned_topic_ids=(topic_id,),
        environment_loader=lambda _root: None,
        environ=SECRETS,
        repository_probe=lambda _root: _binding(),
        topic_executor_factory=lambda _config: lambda request: _success(request),
    )


def _prepare_two_topic_run(tmp_path: Path):
    config, topics = _workspace(
        tmp_path,
        topic_count=2,
        run_id="agentic-distributed-round-trip",
    )
    initialized = initialize_agentic_run(
        config,
        topic_ids=tuple(topic.id for topic in topics),
        repository_probe=lambda _root: _binding(),
    )
    plan = load_run_plan(initialized.plan_path.parent)
    plan_body = initialized.plan_path.read_bytes()
    workers = tuple(_run_worker(config, plan_body, topic.id) for topic in topics)
    assert all(result.complete for result in workers)
    assert [result.assigned_topic_ids for result in workers] == [
        (topics[0].id,),
        (topics[1].id,),
    ]

    bundles: dict[str, tuple[Path, Path]] = {}
    remote_root = tmp_path / "remote"
    for topic in topics:
        topic_root = remote_root / topic.id
        archive = topic_root / "bundle.tar.zst"
        marker = topic_root / "bundle-complete.json"
        pack_topic(initialized.output_dir / "work", topic.id, archive, marker)
        bundles[topic.id] = (archive, marker)
    return topics, plan, plan_body, bundles


def _collector(
    *,
    plan: Any,
    collector_state: Path,
    destination: Path,
    aggregate: Path,
    transport: OfflineHF,
) -> IncrementalCollector:
    return IncrementalCollector(
        bucket_prefix="run",
        plan=plan,
        staging_root=collector_state,
        destination_run_dir=destination,
        transport=transport,
        export_output_dir=aggregate,
        producer_revision=plan.source_revision,
        poll_interval=0,
    )


def test_offline_two_topic_round_trip_is_incremental_idempotent_and_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        topics,
        plan,
        plan_body,
        bundles,
    ) = _prepare_two_topic_run(tmp_path)

    remote = OfflineHF()
    for topic in topics:
        archive, _marker = bundles[topic.id]
        # The archive is uploaded first for both workers.  Only topic one gets
        # its completion marker in wave one.
        remote.publish_archive(topic.id, archive.read_bytes())
    remote.publish_marker(topics[0].id, bundles[topics[0].id][1].read_bytes())
    assert topics[1].id not in {
        row["path"].split("/")[1]
        for row in remote.list("run")
        if row["path"].endswith("bundle-complete.json")
    }

    destination = tmp_path / "coordinator-staging"
    install_run_plan(work_dir=destination, body=plan_body)
    collector_state = tmp_path / "collector-state"
    aggregate = tmp_path / "aggregate"
    first_collector = _collector(
        plan=plan,
        collector_state=collector_state,
        destination=destination,
        aggregate=aggregate,
        transport=remote,
    )
    first = first_collector.run_once()

    assert first.wave == 1
    assert first.imported == (topics[0].id,)
    assert first.missing == (topics[1].id,)
    assert not first.complete
    assert not first.exported
    assert (destination / "topics" / topics[0].id / TOPIC_SEAL_MEMBER).is_file()
    assert not (destination / "topics" / topics[1].id / TOPIC_SEAL_MEMBER).exists()
    assert not (aggregate / EXPORT_MANIFEST_FILENAME).exists()

    # Wave two makes the second worker's marker visible.  A fresh collector
    # instance represents a restart between waves.
    remote.publish_marker(topics[1].id, bundles[topics[1].id][1].read_bytes())
    publication_checks: list[str] = []

    def during_payload_publication(name: str) -> None:
        publication_checks.append(name)
        assert not (aggregate / EXPORT_MANIFEST_FILENAME).exists()

    monkeypatch.setattr(
        retrieval_export,
        "_PUBLICATION_TEST_HOOK",
        during_payload_publication,
    )
    second = _collector(
        plan=plan,
        collector_state=collector_state,
        destination=destination,
        aggregate=aggregate,
        transport=remote,
    ).run_once()

    assert second.wave == 2
    assert second.present == (topics[0].id,)
    assert second.imported == (topics[1].id,)
    assert second.missing == ()
    assert second.complete
    assert second.exported
    assert publication_checks == [
        RETRIEVAL_RUN_FILENAME,
        RETRIEVAL_WITH_TEXT_FILENAME,
        GENERATION_HANDOFF_FILENAME,
    ]
    assert (aggregate / EXPORT_MANIFEST_FILENAME).is_file()

    # A second restart sees both authenticated seals and the manifest-last
    # export receipt, so it performs no duplicate download/import/export.
    third = _collector(
        plan=plan,
        collector_state=collector_state,
        destination=destination,
        aggregate=aggregate,
        transport=remote,
    ).run_once()
    assert third.wave == 3
    assert third.present == plan.planned_topic_ids
    assert third.imported == ()
    assert third.complete
    assert third.exported
    assert remote.downloads == [topics[0].id, topics[1].id]

    export = load_agentic_retrieval_export(
        output_dir=aggregate,
        work_dir=destination,
        plan=plan,
    )
    assert export.topic_ids == plan.planned_topic_ids
    run_lines = (aggregate / RETRIEVAL_RUN_FILENAME).read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in run_lines] == [topic.id for topic in topics]
    assert [line.split()[2] for line in run_lines] == [
        f"doc-{topic.id}" for topic in topics
    ]

    handoff = load_generation_handoff(aggregate / GENERATION_HANDOFF_FILENAME)
    assert [topic.topic_id for topic in handoff.topics] == list(plan.planned_topic_ids)
    trec_docids = {
        topic_id: {line.split()[2] for line in run_lines if line.split()[0] == topic_id}
        for topic_id in plan.planned_topic_ids
    }
    with_text = (aggregate / RETRIEVAL_WITH_TEXT_FILENAME).read_bytes()
    # Use the real published archive; its deterministic member is the only
    # full-text source admitted to generation.
    with zipfile.ZipFile(io.BytesIO(with_text)) as archive:
        rows = [
            json.loads(line)
            for line in archive.read("retrieval_with_text.jsonl").decode("utf-8").splitlines()
        ]
    assert [row["query"]["qid"] for row in rows] == list(plan.planned_topic_ids)
    for topic, row in zip(handoff.topics, rows, strict=True):
        candidate_docids = {candidate["docid"] for candidate in row["candidates"]}
        assert candidate_docids == trec_docids[topic.topic_id]
        assert set(topic.citation_docids) <= candidate_docids
        assert {passage.docid for passage in topic.evidence} <= candidate_docids

    manifest_body = (aggregate / EXPORT_MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_body)
    assert manifest["planned_topic_ids"] == list(plan.planned_topic_ids)
    assert [row["topic_id"] for row in manifest["topics"]] == list(plan.planned_topic_ids)
    assert [row["topic_seal_sha256"] for row in manifest["topics"]] == [
        json.loads(
            (destination / "topics" / topic.id / TOPIC_SEAL_MEMBER).read_text()
        )["seal_sha256"]
        for topic in topics
    ]
    artifacts = {row["name"]: row for row in manifest["artifacts"]}
    for name, artifact in artifacts.items():
        body = (aggregate / name).read_bytes()
        assert artifact["bytes"] == len(body)
        assert artifact["sha256"] == sha256(body).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert manifest["manifest_sha256"] == sha256(canonical).hexdigest()
    assert manifest_body.endswith(b"\n")

    wave_rows = [
        json.loads(line)
        for line in (collector_state / WAVE_RECEIPTS_FILENAME).read_text().splitlines()
    ]
    assert [row["wave"] for row in wave_rows] == [1, 2, 3]
    assert wave_rows[0]["missing"] == [topics[1].id]
    assert wave_rows[1]["imported"] == [topics[1].id]
    assert wave_rows[2]["imported"] == []
    export_receipt = json.loads(
        (collector_state / EXPORT_RECEIPT_FILENAME)
        .read_text()
        .splitlines()[0]
    )
    assert export_receipt["topic_ids"] == list(plan.planned_topic_ids)
    journal_rows = [
        json.loads(line)
        for line in (collector_state / COLLECTOR_JOURNAL_FILENAME).read_text().splitlines()
    ]
    assert [(row["wave"], row["topic_id"], row["status"]) for row in journal_rows] == [
        (1, topics[0].id, "imported"),
        (2, topics[1].id, "imported"),
    ]


def test_offline_bundle_import_recovers_prepare_and_post_install_interruptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics, _plan, plan_body, bundles = _prepare_two_topic_run(tmp_path)
    archive, marker = bundles[topics[0].id]

    prepare_destination = tmp_path / "prepare-destination"
    install_run_plan(work_dir=prepare_destination, body=plan_body)
    original_publish = shard_bundle._publish_identical
    interrupted_prepare = False

    def fail_during_prepare(path: Path, body: bytes, label: str) -> None:
        nonlocal interrupted_prepare
        if label == "prepare journal" and not interrupted_prepare:
            interrupted_prepare = True
            raise RuntimeError("simulated interruption during import prepare")
        original_publish(path, body, label)

    monkeypatch.setattr(shard_bundle, "_publish_identical", fail_during_prepare)
    with pytest.raises(RuntimeError, match="import prepare"):
        import_topic_bundle(archive, marker, prepare_destination)
    assert not (prepare_destination / "topics" / topics[0].id / TOPIC_SEAL_MEMBER).exists()
    monkeypatch.setattr(shard_bundle, "_publish_identical", original_publish)
    recovered_prepare = import_topic_bundle(archive, marker, prepare_destination)
    assert recovered_prepare.topic_id == topics[0].id
    prepare_journal = next(
        (prepare_destination / ".agentic-bundle-imports").glob("*.json")
    )
    assert json.loads(prepare_journal.read_text())["state"] == "complete"

    install_destination = tmp_path / "post-install-destination"
    install_run_plan(work_dir=install_destination, body=plan_body)
    original_write_journal = shard_bundle._write_journal_state
    interrupted_complete = False

    def fail_after_install(path: Path, body: bytes) -> None:
        nonlocal interrupted_complete
        if json.loads(body)["state"] == "complete" and not interrupted_complete:
            interrupted_complete = True
            raise RuntimeError("simulated interruption after install")
        original_write_journal(path, body)

    monkeypatch.setattr(shard_bundle, "_write_journal_state", fail_after_install)
    with pytest.raises(RuntimeError, match="after install"):
        import_topic_bundle(archive, marker, install_destination)
    journal = next((install_destination / ".agentic-bundle-imports").glob("*.json"))
    assert json.loads(journal.read_text())["state"] == "prepare"
    installed_topic = install_destination / "topics" / topics[0].id
    before_retry = {
        path.name: path.read_bytes()
        for path in installed_topic.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(shard_bundle, "_write_journal_state", original_write_journal)
    recovered_install = import_topic_bundle(archive, marker, install_destination)
    assert recovered_install == recovered_prepare
    assert json.loads(journal.read_text())["state"] == "complete"
    assert {
        path.name: path.read_bytes()
        for path in installed_topic.iterdir()
        if path.is_file()
    } == before_retry
    assert import_topic_bundle(archive, marker, install_destination) == recovered_install

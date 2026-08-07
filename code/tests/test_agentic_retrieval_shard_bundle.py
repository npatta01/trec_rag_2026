from __future__ import annotations

import json
import io
import hashlib
from pathlib import Path
import shutil
import tarfile

import pytest
import zstandard

from test_agentic_run_state import TOPICS, _create, _projection, _seal


def _sealed_run(tmp_path: Path) -> tuple[Path, object]:
    run_dir = tmp_path / "run"
    plan = _create(run_dir)
    _seal(run_dir, plan, _projection(tmp_path, TOPICS[0]))
    return run_dir, plan


def test_pack_and_verify_publish_a_deterministic_authenticated_topic_bundle(
    tmp_path: Path,
) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import (
        pack_topic,
        verify_topic_bundle,
    )

    run_dir, plan = _sealed_run(tmp_path)
    archive_one = tmp_path / "one.tar.zst"
    marker_one = tmp_path / "one.complete.json"
    archive_two = tmp_path / "two.tar.zst"
    marker_two = tmp_path / "two.complete.json"

    first = pack_topic(run_dir, TOPICS[0].id, archive_one, marker_one)
    second = pack_topic(run_dir, TOPICS[0].id, archive_two, marker_two)

    assert archive_one.read_bytes() == archive_two.read_bytes()
    assert marker_one.read_bytes() == marker_two.read_bytes()
    assert first == second
    verified = verify_topic_bundle(
        archive_one, marker_one, expected_run_plan_sha256=plan.plan_sha256
    )
    assert verified.topic_id == TOPICS[0].id
    assert verified.run_plan_sha256 == plan.plan_sha256
    marker = json.loads(marker_one.read_text())
    assert marker["archive_sha256"]
    assert marker["archive_size"] == archive_one.stat().st_size
    assert marker["run_plan_sha256"] == plan.plan_sha256
    assert marker["topic_id"] == TOPICS[0].id
    assert marker["topic_seal_sha256"] == verified.topic_seal_sha256


def test_import_is_create_or_identical_only_and_status_summarizes_topic(
    tmp_path: Path,
) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import (
        import_topic_bundle,
        pack_topic,
        summarize_staging,
    )

    source, plan = _sealed_run(tmp_path)
    archive = tmp_path / "topic.tar.zst"
    marker = tmp_path / "topic.complete.json"
    pack_topic(source, TOPICS[0].id, archive, marker)

    staging = tmp_path / "staging"
    imported = import_topic_bundle(archive, marker, staging)
    assert imported.topic_id == TOPICS[0].id
    assert (staging / "run_plan.json").exists() is False
    assert (staging / "topics" / TOPICS[0].id / "topic_projection_manifest.json").exists()
    assert summarize_staging(staging) == {
        "topic_ids": [TOPICS[0].id],
        "completed_topic_ids": [TOPICS[0].id],
        "missing_topic_ids": [],
    }
    assert import_topic_bundle(archive, marker, staging) == imported

    conflict = tmp_path / "different.tar.zst"
    conflict.write_bytes(archive.read_bytes() + b"conflict")
    with pytest.raises(Exception, match="digest|marker|archive"):
        import_topic_bundle(conflict, marker, staging)


def _archive_members(archive_path: Path) -> list[tarfile.TarInfo]:
    from trec_rag.agentic_retrieval_shard_bundle import _decompress_archive

    with tarfile.open(fileobj=io.BytesIO(_decompress_archive(archive_path)), mode="r:") as archive:
        return archive.getmembers()


def _archive_payload(archive_path: Path, name: str) -> bytes:
    from trec_rag.agentic_retrieval_shard_bundle import _decompress_archive

    with tarfile.open(fileobj=io.BytesIO(_decompress_archive(archive_path)), mode="r:") as archive:
        info = archive.getmember(name)
        stream = archive.extractfile(info)
        assert stream is not None
        return stream.read()


def _write_hostile_archive(path: Path, entries: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for info, value in entries:
            if info.type == tarfile.REGTYPE:
                info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    path.write_bytes(zstandard.ZstdCompressor(write_checksum=True).compress(body.getvalue()))


def _marker_for_archive(valid_marker: Path, archive: Path, target: Path) -> None:
    marker = json.loads(valid_marker.read_text())
    marker["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    marker["archive_size"] = archive.stat().st_size
    target.write_text(json.dumps(marker, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def test_bundle_contains_only_projection_closure_and_manifest_is_canonical(
    tmp_path: Path,
) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import BUNDLE_MANIFEST_NAME, TOPIC_MEMBER_NAMES, pack_topic

    source, _ = _sealed_run(tmp_path)
    topic_dir = source / "topics" / TOPICS[0].id
    (topic_dir / "attempts" / "000001" / "raw-response.json").write_text("secret")
    (topic_dir / "records.sqlite3").write_bytes(b"ledger")
    (topic_dir / "cache.sqlite-wal").write_bytes(b"wal")
    archive = tmp_path / "bundle.tar.zst"
    marker = tmp_path / "bundle.complete.json"
    pack_topic(source, TOPICS[0].id, archive, marker)

    infos = _archive_members(archive)
    assert [info.name for info in infos] == [BUNDLE_MANIFEST_NAME, *sorted(TOPIC_MEMBER_NAMES)]
    manifest = _archive_payload(archive, BUNDLE_MANIFEST_NAME)
    assert manifest.endswith(b"\n")
    assert json.loads(manifest.decode()) == json.loads(manifest.decode())
    assert {info.name for info in infos[1:]} == set(TOPIC_MEMBER_NAMES)


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "prefix", "symlink", "hardlink", "device", "truncated", "trailing", "digest"])
def test_verifier_rejects_hostile_archive_streams(tmp_path: Path, kind: str) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import AgenticShardBundleError, pack_topic, verify_topic_bundle

    source, plan = _sealed_run(tmp_path)
    valid_archive = tmp_path / "valid.tar.zst"
    valid_marker = tmp_path / "valid.complete.json"
    pack_topic(source, TOPICS[0].id, valid_archive, valid_marker)
    manifest = _archive_payload(valid_archive, "bundle-manifest.json")
    retrieval = _archive_payload(valid_archive, "retrieval_topic.json")
    generation = _archive_payload(valid_archive, "generation_topic.json")
    receipt = _archive_payload(valid_archive, "topic_records_receipt.json")
    seal = _archive_payload(valid_archive, "topic_projection_manifest.json")

    def info(name: str, size: int | None = None) -> tarfile.TarInfo:
        value = tarfile.TarInfo(name)
        value.mode = 0o600
        value.mtime = value.uid = value.gid = 0
        value.size = len(retrieval) if size is None else size
        return value

    entries = [(info("bundle-manifest.json", len(manifest)), manifest)]
    if kind == "traversal":
        entries.append((info("../retrieval_topic.json"), retrieval))
    elif kind == "duplicate":
        entries.extend([(info("retrieval_topic.json"), retrieval), (info("retrieval_topic.json"), retrieval), (info("generation_topic.json"), generation), (info("topic_records_receipt.json"), receipt)])
    elif kind == "prefix":
        entries.append((info("retrieval_topic.json/child"), retrieval))
    elif kind in {"symlink", "hardlink", "device"}:
        bad = info("retrieval_topic.json")
        bad.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "device": tarfile.CHRTYPE}[kind]
        bad.linkname = "target"
        bad.size = 0
        entries.append((bad, b""))
    elif kind == "digest":
        entries.append((info("generation_topic.json", len(generation)), generation[:-1] + b"x"))
    else:
        entries.extend([(info("generation_topic.json", len(generation)), generation), (info("retrieval_topic.json", len(retrieval)), retrieval), (info("topic_records_receipt.json", len(receipt)), receipt), (info("topic_projection_manifest.json", len(seal)), seal)])
    hostile_archive = tmp_path / f"{kind}.tar.zst"
    _write_hostile_archive(hostile_archive, entries)
    if kind == "truncated":
        hostile_archive.write_bytes(hostile_archive.read_bytes()[:-5])
    elif kind == "trailing":
        hostile_archive.write_bytes(hostile_archive.read_bytes() + b"junk")
    hostile_marker = tmp_path / f"{kind}.complete.json"
    _marker_for_archive(valid_marker, hostile_archive, hostile_marker)
    with pytest.raises(AgenticShardBundleError):
        verify_topic_bundle(hostile_archive, hostile_marker, plan.plan_sha256)


def test_import_verifies_before_creating_lock_or_mutating_destination(tmp_path: Path) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import AgenticShardBundleError, pack_topic, import_topic_bundle

    source, _ = _sealed_run(tmp_path)
    archive = tmp_path / "valid.tar.zst"
    marker = tmp_path / "valid.complete.json"
    pack_topic(source, TOPICS[0].id, archive, marker)
    archive.write_bytes(archive.read_bytes() + b"bad")
    destination = tmp_path / "not-created"
    with pytest.raises(AgenticShardBundleError):
        import_topic_bundle(archive, marker, destination)
    assert not destination.exists()


def test_import_round_trips_against_authenticated_destination_plan(tmp_path: Path) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import import_topic_bundle, pack_topic

    source, plan = _sealed_run(tmp_path)
    archive = tmp_path / "valid.tar.zst"
    marker = tmp_path / "valid.complete.json"
    pack_topic(source, TOPICS[0].id, archive, marker)
    destination = tmp_path / "staging"
    destination.mkdir()
    shutil.copy2(source / "run_plan.json", destination / "run_plan.json")
    imported = import_topic_bundle(archive, marker, destination)
    assert imported.run_plan_sha256 == plan.plan_sha256
    journal = next((destination / ".agentic-bundle-imports").glob("*.json"))
    assert json.loads(journal.read_text())["state"] == "complete"


def test_foreign_run_plan_and_conflicting_marker_are_rejected(tmp_path: Path) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import (
        AgenticShardBundleIntegrityError,
        pack_topic,
        verify_topic_bundle,
    )

    source, plan = _sealed_run(tmp_path / "first")
    _foreign_source, _foreign_plan = _sealed_run(tmp_path / "foreign")
    archive = tmp_path / "bundle.tar.zst"
    marker = tmp_path / "bundle.complete.json"
    pack_topic(source, TOPICS[0].id, archive, marker)
    with pytest.raises(AgenticShardBundleIntegrityError, match="another run plan"):
        verify_topic_bundle(archive, marker, "f" * 64)

    marker.write_bytes(marker.read_bytes().replace(plan.plan_sha256.encode(), b"f" * 64))
    with pytest.raises(AgenticShardBundleIntegrityError):
        verify_topic_bundle(archive, marker, plan.plan_sha256)
    # The immutable archive remains unchanged after marker failure.
    assert archive.is_file()


def test_pack_is_identical_only_for_existing_archive_and_marker(tmp_path: Path) -> None:
    from trec_rag.agentic_retrieval_shard_bundle import AgenticShardBundleConflictError, pack_topic

    source, _ = _sealed_run(tmp_path)
    archive = tmp_path / "bundle.tar.zst"
    marker = tmp_path / "bundle.complete.json"
    first = pack_topic(source, TOPICS[0].id, archive, marker)
    marker.write_bytes(marker.read_bytes().replace(b"agentic-retrieval-topic-bundle-v1", b"foreign-bundle-schema-v1"))
    with pytest.raises(AgenticShardBundleConflictError):
        pack_topic(source, TOPICS[0].id, archive, marker)
    assert first.archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()

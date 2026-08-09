"""Contract tests for the privacy-safe multipage debug-report bundle models."""

from __future__ import annotations

import json
from dataclasses import fields, replace
import gc
from hashlib import sha256
from html.parser import HTMLParser
import os
from pathlib import Path
import socket
import stat
import weakref

import pytest

from offline_evaluation_fixture import TopicSpec, build_run
import trec_rag.competition_debug_bundle as bundle_module
import trec_rag.competition_debug_report as debug_report
from trec_rag.competition_debug_report import load_debug_report_data
from trec_rag.friendly_report import ReportPrivacyError
from trec_rag.offline_evaluation import (
    EvaluationError,
    JudgeOutcome,
    JudgeSettings,
    build_evaluation_bundle,
)


REPOSITORY_ROOT = Path(__file__).parents[2]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "a" and dict(attrs).get("href"):
            self.hrefs.append(str(dict(attrs)["href"]))


def _local_hrefs(page: str) -> tuple[str, ...]:
    parser = _HrefParser()
    parser.feed(page)
    return tuple(
        href
        for href in parser.hrefs
        if not href.startswith(("http://", "https://", "mailto:", "#"))
    )


def test_build_bundle_writes_summary_topics_and_manifest_with_matching_receipt(
    tmp_path: Path,
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "debug-bundle"

    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        output_dir=target,
    )

    assert receipt.output_dir == target.resolve()
    assert receipt.index_path == target.resolve() / "index.html"
    assert receipt.topic_ids == ("alpha-topic", "beta-topic")
    assert receipt.page_count == 3
    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "competition_debug_report_bundle_v1"
    assert [item["topic_id"] for item in manifest["topics"]] == list(receipt.topic_ids)
    assert manifest["index"]["sha256"] == _file_sha256(target / "index.html")
    assert all(
        item["sha256"] == _file_sha256(target / item["path"])
        for item in manifest["topics"]
    )
    assert receipt.bundle_manifest_sha256 == _file_sha256(receipt.manifest_path)
    assert receipt.total_bytes == sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o500
    assert stat.S_IMODE((target / "topics").stat().st_mode) == 0o500
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400
        for path in target.rglob("*")
        if path.is_file()
    )


def test_bundle_is_deterministic_and_loads_run_data_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    first = fixture.retrieval_output / "bundle-a"
    second = fixture.retrieval_output / "bundle-b"
    real_load = debug_report.load_debug_report_data
    calls = 0

    def observed_load(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(debug_report, "load_debug_report_data", observed_load)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=first)
    assert calls == 1
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=second)
    assert calls == 2
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.parametrize("existing_kind", ("file", "directory", "symlink"))
def test_bundle_refuses_every_existing_target(
    tmp_path: Path, existing_kind: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "existing-bundle"
    if existing_kind == "file":
        target.write_text("owned by somebody else", encoding="utf-8")
    elif existing_kind == "directory":
        target.mkdir()
        (target / "owned.txt").write_text("owned by somebody else", encoding="utf-8")
    else:
        destination = fixture.retrieval_output / "symlink-destination"
        destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="bundle output.*(?:absent|symbolic link)"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert target.exists() or target.is_symlink()


@pytest.mark.parametrize("topic_id", ("bad/topic", "..", "bad\x00topic"))
def test_bundle_rejects_unsafe_topic_filenames(tmp_path: Path, topic_id: str) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    unsafe = replace(data, topics=(replace(data.topics[0], topic_id=topic_id),))
    target = fixture.retrieval_output / "unsafe-topic-bundle"

    with pytest.raises(ValueError, match="safe topic filename"):
        bundle_module.build_bundle_from_data(unsafe, output_dir=target, evaluation_manifest_path=None)

    assert not target.exists()


def test_bundle_rejects_an_external_output_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    fixture = build_run(run_root, (TopicSpec("alpha-topic", "alpha"),))
    external_parent = tmp_path / "external"
    external_parent.mkdir()

    with pytest.raises(ValueError, match="inside the repository or retrieval output"):
        debug_report.build_debug_report_bundle(
            fixture.retrieval_config,
            output_dir=external_parent / "bundle",
        )


@pytest.mark.parametrize("failed_name", ("index.html", "bundle-manifest.json"))
def test_bundle_write_failure_removes_only_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "failed-bundle"
    sentinel = fixture.retrieval_output / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    real_write = bundle_module._write_bundle_file

    def fail_named(path: Path, payload: bytes) -> None:
        if path.name == failed_name:
            raise OSError(f"forced {failed_name} failure")
        real_write(path, payload)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", fail_named)
    with pytest.raises(OSError, match="forced"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert list(fixture.retrieval_output.glob(".failed-bundle.*.tmp")) == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_bundle_manifest_is_written_last_and_all_links_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "ordered-bundle"
    writes: list[str] = []
    real_write = bundle_module._write_bundle_file

    def observed(path: Path, payload: bytes) -> None:
        writes.append(path.name)
        real_write(path, payload)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", observed)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert writes[-1] == "bundle-manifest.json"
    for page in target.rglob("*.html"):
        for href in _local_hrefs(page.read_text(encoding="utf-8")):
            assert (page.parent / href.split("#", 1)[0]).resolve().is_file()


@pytest.mark.parametrize(
    "failure_point",
    ("topic-render", "summary-render", "directory-fsync", "rename", "hash"),
)
def test_bundle_failure_points_never_publish_a_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "partial-bundle"
    if failure_point == "topic-render":
        monkeypatch.setattr(
            bundle_module,
            "render_debug_topic_page",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("topic-render")),
        )
    elif failure_point == "summary-render":
        monkeypatch.setattr(
            bundle_module,
            "render_bundle_summary",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("summary-render")),
        )
    elif failure_point == "directory-fsync":
        monkeypatch.setattr(
            bundle_module,
            "_fsync_directory",
            lambda *_args: (_ for _ in ()).throw(OSError("directory-fsync")),
        )
    elif failure_point == "rename":
        monkeypatch.setattr(
            bundle_module,
            "_rename_noreplace_fds",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rename")),
        )
    else:
        monkeypatch.setattr(bundle_module, "_hash_matches", lambda *_args: False)

    with pytest.raises((OSError, RuntimeError, ValueError), match=failure_point):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert list(fixture.retrieval_output.glob(".partial-bundle.*.tmp")) == []


def test_bundle_build_does_not_modify_source_artifacts(tmp_path: Path) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    before = {
        path.relative_to(fixture.retrieval_output): _file_sha256(path)
        for path in fixture.retrieval_output.rglob("*")
        if path.is_file()
    }

    debug_report.build_debug_report_bundle(
        fixture.retrieval_config,
        output_dir=fixture.retrieval_output / "immutable-source-bundle",
    )

    after = {
        relative: _file_sha256(fixture.retrieval_output / relative)
        for relative in before
    }
    assert after == before


def test_bundle_makes_no_network_call_or_candidate_ledger_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "offline-bundle"
    real_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.name == "candidates.jsonl":
            raise AssertionError("candidate ledger was opened")
        return real_open(path, *args, **kwargs)

    def forbid_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(socket, "socket", forbid_socket)

    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)
    assert (target / "bundle-manifest.json").is_file()


def test_bundle_cleanup_refuses_a_substituted_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "staging-substitution"
    real_write = bundle_module._write_bundle_file
    foreign_paths: list[Path] = []
    moved_paths: list[Path] = []

    def substitute_after_index(path: Path, payload: bytes) -> None:
        real_write(path, payload)
        if path.name == "index.html":
            payload_dir = path.parent.parent
            guardian = payload_dir.parent
            moved = guardian.with_name(guardian.name + ".moved")
            guardian.rename(moved)
            guardian.mkdir(mode=0o700)
            (guardian / "foreign-sentinel").write_text("foreign", encoding="utf-8")
            foreign_paths.append(guardian)
            moved_paths.append(moved)
            raise OSError("forced staging substitution")

    monkeypatch.setattr(bundle_module, "_write_bundle_file", substitute_after_index)
    with pytest.raises(OSError, match="forced staging substitution"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert len(foreign_paths) == len(moved_paths) == 1
    assert foreign_paths[0].is_dir()
    assert (foreign_paths[0] / "foreign-sentinel").read_text(encoding="utf-8") == "foreign"
    assert moved_paths[0].is_dir()
    assert not target.exists()


def test_bundle_cleanup_does_not_delete_substituted_foreign_guardian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "cleanup-window"
    real_write = bundle_module._write_bundle_file
    real_remove = bundle_module._remove_guardian_path
    foreign_paths: list[Path] = []
    moved_paths: list[Path] = []

    def fail_topic(path: Path, payload: bytes) -> None:
        if path.name == "bundle-manifest.json":
            raise OSError("forced cleanup")
        real_write(path, payload)

    def substitute_before_remove(path: Path, identity: object) -> None:
        moved = path.with_name(path.name + ".moved")
        path.rename(moved)
        path.mkdir(mode=0o700)
        foreign = path / "foreign-nonempty"
        foreign.write_text("preserve", encoding="utf-8")
        moved_paths.append(moved)
        foreign_paths.append(foreign)
        real_remove(path, identity)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", fail_topic)
    monkeypatch.setattr(bundle_module, "_remove_guardian_path", substitute_before_remove)
    with pytest.raises(OSError, match="forced cleanup"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert moved_paths and moved_paths[0].is_dir()
    assert foreign_paths and foreign_paths[0].read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("replacement", ("symlink", "file"))
def test_bundle_rejects_a_topic_replaced_after_initial_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / f"topic-{replacement}-substitution"
    foreign = fixture.retrieval_output / f"foreign-{replacement}.html"
    foreign.write_text("foreign topic bytes", encoding="utf-8")
    real_validate = bundle_module._validate_staging
    swapped = False

    def replace_topic_before_final_reconciliation(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        real_validate(*args, **kwargs)
        if not swapped:
            swapped = True
            staging = args[0]
            topic_path = staging / "topics" / "alpha-topic.html"
            topic_path.unlink()
            if replacement == "symlink":
                topic_path.symlink_to(foreign)
            else:
                topic_path.write_text("foreign topic bytes", encoding="utf-8")

    monkeypatch.setattr(bundle_module, "_validate_staging", replace_topic_before_final_reconciliation)
    with pytest.raises((ValueError, PermissionError)):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert foreign.read_text(encoding="utf-8") == "foreign topic bytes"


def test_bundle_rename_seam_publishes_pinned_payload_after_guardian_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "guardian-rename-substitution"
    real_rename = bundle_module._rename_noreplace_fds
    moved_paths: list[Path] = []
    foreign_paths: list[Path] = []

    def substitute_guardian(
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        **kwargs: object,
    ) -> None:
        guardian = Path(os.readlink(f"/proc/self/fd/{old_dir_fd}"))
        moved = guardian.with_name(guardian.name + ".moved")
        guardian.rename(moved)
        guardian.mkdir(mode=0o700)
        foreign = guardian / "foreign.txt"
        foreign.write_text("must survive", encoding="utf-8")
        moved_paths.append(moved)
        foreign_paths.append(foreign)
        real_rename(old_dir_fd, old_name, new_dir_fd, new_name, **kwargs)

    monkeypatch.setattr(bundle_module, "_rename_noreplace_fds", substitute_guardian)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert (target / "topics" / "alpha-topic.html").is_file()
    assert "must survive" not in (target / "index.html").read_text(encoding="utf-8")
    assert moved_paths and moved_paths[0].is_dir()
    assert foreign_paths and foreign_paths[0].read_text(encoding="utf-8") == "must survive"


def test_bundle_rename_seam_refuses_a_replaced_payload_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "payload-rename-substitution"
    real_rename = bundle_module._rename_noreplace_fds
    moved_payloads: list[Path] = []
    foreign_payloads: list[Path] = []

    def substitute_payload(
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        **kwargs: object,
    ) -> None:
        guardian = Path(os.readlink(f"/proc/self/fd/{old_dir_fd}"))
        moved = guardian / "payload.moved"
        os.rename(old_name, moved.name, src_dir_fd=old_dir_fd, dst_dir_fd=old_dir_fd)
        foreign = guardian / old_name
        foreign.mkdir(mode=0o700)
        (foreign / "foreign.txt").write_text("must survive", encoding="utf-8")
        moved_payloads.append(moved)
        foreign_payloads.append(foreign)
        real_rename(old_dir_fd, old_name, new_dir_fd, new_name, **kwargs)

    monkeypatch.setattr(bundle_module, "_rename_noreplace_fds", substitute_payload)
    with pytest.raises(ValueError, match="payload identity"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert moved_payloads and moved_payloads[0].is_dir()
    assert foreign_payloads and (foreign_payloads[0] / "foreign.txt").read_text(encoding="utf-8") == "must survive"


def test_bundle_rename_primitive_rechecks_page_after_outer_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "rename-page-substitution"
    foreign = fixture.retrieval_output / "foreign-page.html"
    foreign.write_text("foreign page bytes", encoding="utf-8")
    real_rename = bundle_module._rename_noreplace_fds
    swapped = False

    def substitute_page(
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            payload = Path(os.readlink(f"/proc/self/fd/{old_dir_fd}")) / old_name
            page = payload / "topics" / "alpha-topic.html"
            page.unlink()
            page.symlink_to(foreign)
        real_rename(old_dir_fd, old_name, new_dir_fd, new_name, **kwargs)

    monkeypatch.setattr(bundle_module, "_rename_noreplace_fds", substitute_page)
    with pytest.raises((ValueError, PermissionError)):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert foreign.read_text(encoding="utf-8") == "foreign page bytes"


def test_bundle_rename_refuses_a_recreated_target_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    parent = fixture.retrieval_output / "nested-publish-parent"
    parent.mkdir()
    target = parent / "debug-bundle"
    outside = tmp_path / "moved-publish-parent"
    real_rename = bundle_module._rename_noreplace_fds
    moved = False

    def move_parent_before_publication(
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal moved
        if not moved:
            moved = True
            parent.rename(outside)
            parent.mkdir()
        real_rename(old_dir_fd, old_name, new_dir_fd, new_name, **kwargs)

    monkeypatch.setattr(
        bundle_module, "_rename_noreplace_fds", move_parent_before_publication
    )
    with pytest.raises(ValueError, match="target parent"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert outside.is_dir()
    assert list(parent.iterdir()) == []


def test_bundle_cleanup_quarantine_rejects_foreign_file_at_delete_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "cleanup-file-delete-seam"
    real_write = bundle_module._write_bundle_file
    real_delete = bundle_module._unlink_quarantined_entry
    foreign_paths: list[Path] = []
    substituted = False

    def fail_manifest(path: Path, payload: bytes) -> None:
        if path.name == "bundle-manifest.json":
            raise OSError("forced cleanup file seam")
        real_write(path, payload)

    def substitute_file(quarantine_fd: int, name: str, identity: object) -> None:
        nonlocal substituted
        if not substituted:
            substituted = True
            quarantine = Path(os.readlink(f"/proc/self/fd/{quarantine_fd}"))
            owned = quarantine / name
            owned.unlink()
            owned.write_text("foreign cleanup file", encoding="utf-8")
            foreign_paths.append(owned)
        real_delete(quarantine_fd, name, identity)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", fail_manifest)
    monkeypatch.setattr(bundle_module, "_unlink_quarantined_entry", substitute_file)
    with pytest.raises(OSError, match="forced cleanup file seam"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert foreign_paths
    restored = foreign_paths[0].parent.parent / "index.html"
    assert (
        (foreign_paths[0].is_file()
        and foreign_paths[0].read_text(encoding="utf-8") == "foreign cleanup file")
        or (restored.is_file() and restored.read_text(encoding="utf-8") == "foreign cleanup file")
    )
    assert not target.exists()


def test_bundle_cleanup_quarantine_rejects_foreign_directory_at_delete_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "cleanup-directory-delete-seam"
    real_write = bundle_module._write_bundle_file
    real_rmdir = bundle_module._rmdir_quarantined_entry
    foreign_paths: list[Path] = []
    substituted = False

    def fail_manifest(path: Path, payload: bytes) -> None:
        if path.name == "bundle-manifest.json":
            raise OSError("forced cleanup directory seam")
        real_write(path, payload)

    def substitute_directory(quarantine_fd: int, name: str, identity: object) -> None:
        nonlocal substituted
        if not substituted:
            substituted = True
            quarantine = Path(os.readlink(f"/proc/self/fd/{quarantine_fd}"))
            owned = quarantine / name
            moved = quarantine / f"{name}.owned"
            owned.rename(moved)
            owned.mkdir()
            foreign_paths.append(owned)
        real_rmdir(quarantine_fd, name, identity)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", fail_manifest)
    monkeypatch.setattr(
        bundle_module, "_rmdir_quarantined_entry", substitute_directory
    )
    with pytest.raises(OSError, match="forced cleanup directory seam"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert foreign_paths and foreign_paths[0].is_dir()
    assert not target.exists()


@pytest.mark.parametrize(
    "failed_path",
    ("index.html", "topics/alpha-topic.html", "bundle-manifest.json"),
)
def test_bundle_receipt_failure_still_cleans_registered_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_path: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / f"receipt-failure-{failed_path.replace('/', '-') }"
    real_receipt = bundle_module._file_receipt

    def fail_receipt(path: Path, relative_path: str, **kwargs: object):  # type: ignore[no-untyped-def]
        if relative_path == failed_path:
            raise OSError(f"forced receipt read {failed_path}")
        return real_receipt(path, relative_path, **kwargs)

    monkeypatch.setattr(bundle_module, "_file_receipt", fail_receipt)
    with pytest.raises(OSError, match="forced receipt read"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert list(fixture.retrieval_output.glob(f".{target.name}.*.tmp")) == []


def test_bundle_freezes_pages_before_late_pinned_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "frozen-late-validation"
    real_validate = bundle_module._validate_pinned_payload
    mutation_errors: list[type[BaseException]] = []

    def mutate_early_page_before_pinned_sweep(*args: object, **kwargs: object) -> None:
        guardian_fd = args[0]
        payload_name = args[1]
        payload = Path(os.readlink(f"/proc/self/fd/{guardian_fd}")) / str(payload_name)
        early = payload / "topics" / "alpha-topic.html"
        try:
            early.write_bytes(b"invalid late-scan mutation")
        except BaseException as error:  # pragma: no cover - assertion below records it
            mutation_errors.append(type(error))
        else:
            raise AssertionError("late validation mutated a frozen page")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        bundle_module, "_validate_pinned_payload", mutate_early_page_before_pinned_sweep
    )
    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config, output_dir=target
    )

    assert receipt.output_dir == target.resolve()
    assert mutation_errors == [PermissionError]


def test_bundle_freezes_pages_before_success_receipt_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "frozen-receipt-verification"
    real_validate = bundle_module._validate_published_receipts
    mutation_errors: list[type[BaseException]] = []

    def mutate_early_page_before_receipt(*args: object, **kwargs: object) -> None:
        published = Path(args[0])
        early = published / "topics" / "alpha-topic.html"
        try:
            early.write_bytes(b"invalid receipt mutation")
        except BaseException as error:  # pragma: no cover - assertion below records it
            mutation_errors.append(type(error))
        else:
            raise AssertionError("success receipt verification mutated a frozen page")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        bundle_module, "_validate_published_receipts", mutate_early_page_before_receipt
    )
    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config, output_dir=target
    )

    assert receipt.output_dir == target.resolve()
    assert mutation_errors == [PermissionError]


def test_bundle_freezes_root_against_unlisted_file_during_receipt_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "frozen-root-coverage"
    real_validate = bundle_module._validate_published_receipts
    mutation_errors: list[type[BaseException]] = []

    def add_unlisted_file(*args: object, **kwargs: object) -> None:
        published = Path(args[0])
        try:
            (published / "unlisted.txt").write_text("foreign", encoding="utf-8")
        except BaseException as error:  # pragma: no cover - assertion below records it
            mutation_errors.append(type(error))
        else:
            raise AssertionError("receipt verification mutated the frozen bundle root")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        bundle_module, "_validate_published_receipts", add_unlisted_file
    )
    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config, output_dir=target
    )

    assert receipt.output_dir == target.resolve()
    assert mutation_errors == [PermissionError]
    assert set(path.name for path in target.iterdir()) == {
        "index.html",
        "topics",
        "bundle-manifest.json",
    }


def test_bundle_freezes_topics_directory_during_receipt_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "frozen-topics-coverage"
    real_validate = bundle_module._validate_published_receipts
    mutation_errors: list[type[BaseException]] = []

    def replace_topics_directory(*args: object, **kwargs: object) -> None:
        published = Path(args[0])
        topics = published / "topics"
        try:
            topics.rename(published / "topics-replaced")
            topics.mkdir()
        except BaseException as error:  # pragma: no cover - assertion below records it
            mutation_errors.append(type(error))
        else:
            raise AssertionError("receipt verification replaced the frozen topics directory")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        bundle_module, "_validate_published_receipts", replace_topics_directory
    )
    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config, output_dir=target
    )

    assert receipt.output_dir == target.resolve()
    assert mutation_errors == [PermissionError]
    assert (target / "topics" / "alpha-topic.html").is_file()


def test_bundle_closes_its_page_fds_before_validation_and_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "closed-page-fds"
    real_validate = bundle_module._validate_staging
    observed_page_fds: list[str] = []

    def observe_closed_page_fds(*args: object, **kwargs: object) -> None:
        staging = Path(args[0])
        page_paths = {staging / "index.html", staging / "topics" / "alpha-topic.html"}
        for entry in Path("/proc/self/fd").iterdir():
            try:
                opened = Path(os.readlink(entry))
            except OSError:
                continue
            if opened in page_paths:
                observed_page_fds.append(str(opened))
        real_validate(*args, **kwargs)

    monkeypatch.setattr(bundle_module, "_validate_staging", observe_closed_page_fds)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert observed_page_fds == []


def test_bundle_receipt_verification_reuses_prevalidated_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "receipt-reuses-hashes"
    real_hash_matches = bundle_module._hash_matches
    target_hash_calls: list[Path] = []

    def observed_hash(path: Path, expected_sha256: str, expected_bytes: int) -> bool:
        if target in path.parents:
            target_hash_calls.append(path)
        return real_hash_matches(path, expected_sha256, expected_bytes)

    monkeypatch.setattr(bundle_module, "_hash_matches", observed_hash)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert target_hash_calls == []


@pytest.mark.parametrize("substitute", (False, True))
def test_bundle_partial_write_failure_keeps_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, substitute: bool
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / f"partial-write-{'foreign' if substitute else 'owned'}"
    real_fsync = bundle_module.os.fsync
    foreign: list[Path] = []
    triggered = False

    def fail_write(descriptor: int) -> None:
        nonlocal triggered
        if not triggered:
            triggered = True
            written = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            if substitute:
                written.unlink()
                written.write_text("foreign replacement", encoding="utf-8")
                foreign.append(written)
            raise OSError("forced partial write failure")
        real_fsync(descriptor)

    monkeypatch.setattr(bundle_module.os, "fsync", fail_write)
    with pytest.raises(OSError, match="forced partial write failure"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    if substitute:
        assert foreign
        guardian = foreign[0].parents[1]
        assert any(
            path.read_text(encoding="utf-8") == "foreign replacement"
            for path in guardian.rglob("index.html")
        )
    else:
        assert list(fixture.retrieval_output.glob(f".{target.name}.*.tmp")) == []


def test_bundle_publication_parent_fd_closes_on_libc_configuration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "parent-fd-libc-failure"
    real_cdll = bundle_module.ctypes.CDLL

    def fail_cdll(*args: object, **kwargs: object):
        raise OSError("forced libc configuration failure")

    def output_fds() -> set[str]:
        values: set[str] = set()
        for entry in Path("/proc/self/fd").iterdir():
            try:
                value = os.readlink(entry)
            except OSError:
                continue
            if value == str(fixture.retrieval_output):
                values.add(value)
        return values

    monkeypatch.setattr(bundle_module.ctypes, "CDLL", fail_cdll)
    before = output_fds()
    with pytest.raises(OSError, match="forced libc configuration failure"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)
    assert output_fds() == before
    monkeypatch.setattr(bundle_module.ctypes, "CDLL", real_cdll)


def test_bundle_rejects_broken_relative_cross_page_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "broken-cross-page-fragment"
    real_summary = bundle_module.render_bundle_summary

    def broken_summary(*args: object, **kwargs: object) -> str:
        return real_summary(*args, **kwargs).replace(
            "</main>", '<a href="topics/alpha-topic.html#missing-anchor">broken</a></main>', 1
        )

    monkeypatch.setattr(bundle_module, "render_bundle_summary", broken_summary)
    with pytest.raises(ValueError, match="fragment"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()


def test_bundle_link_validation_parses_each_page_once_per_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "single-scan"
    real_feed = bundle_module.HTMLParser.feed
    real_summary = bundle_module.render_bundle_summary
    real_topic = bundle_module.render_debug_topic_page
    feed_count = 0
    anchor_links = "".join(
        f'<a href="topics/alpha-topic.html#anchor-{index}">link</a>'
        for index in range(10)
    )
    anchors = "".join(f'<a id="anchor-{index}"></a>' for index in range(10))

    def summary_with_many_fragments(*args: object, **kwargs: object) -> str:
        return real_summary(*args, **kwargs).replace("</main>", anchor_links + "</main>", 1)

    def topic_with_anchors(*args: object, **kwargs: object) -> str:
        return real_topic(*args, **kwargs).replace("</main>", anchors + "</main>", 1)

    def observed_feed(parser: HTMLParser, page: str) -> None:
        nonlocal feed_count
        feed_count += 1
        real_feed(parser, page)

    monkeypatch.setattr(bundle_module.HTMLParser, "feed", observed_feed)
    monkeypatch.setattr(bundle_module, "render_bundle_summary", summary_with_many_fragments)
    monkeypatch.setattr(bundle_module, "render_debug_topic_page", topic_with_anchors)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert feed_count == 3


def test_bundle_rejects_a_late_empty_target_without_overwriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "late-target"
    real_rename = bundle_module._rename_noreplace_fds
    injected = False

    def inject_target(
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            target.mkdir()
        real_rename(old_dir_fd, old_name, new_dir_fd, new_name, **kwargs)

    monkeypatch.setattr(bundle_module, "_rename_noreplace_fds", inject_target)
    with pytest.raises(FileExistsError):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert target.is_dir()
    assert list(target.iterdir()) == []
    assert list(fixture.retrieval_output.glob(".late-target.*.tmp")) == []


def test_bundle_keeps_complete_target_when_post_rename_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "durability-failure"
    real_fsync = bundle_module._fsync_directory_fd

    def fail_after_rename(descriptor: int) -> None:
        if target.exists():
            raise OSError("forced post-rename parent fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(bundle_module, "_fsync_directory_fd", fail_after_rename)
    with pytest.raises(OSError, match="durability|post-rename"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert target.is_dir()
    assert (target / "bundle-manifest.json").is_file()
    with pytest.raises(ValueError, match="bundle output must be absent"):
        debug_report.build_debug_report_bundle(
            fixture.retrieval_config, output_dir=target
        )


def test_bundle_reports_parent_reopen_failure_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "parent-reopen-failure"
    real_open = bundle_module._open_directory_fd

    def fail_parent_reopen(path: Path) -> int:
        if path == target.parent and target.exists():
            raise OSError("forced parent reopen")
        return real_open(path)

    monkeypatch.setattr(bundle_module, "_open_directory_fd", fail_parent_reopen)
    with pytest.raises(
        OSError, match="bundle published but target parent durability fsync failed"
    ):
        debug_report.build_debug_report_bundle(
            fixture.retrieval_config, output_dir=target
        )

    assert (target / "bundle-manifest.json").is_file()


def test_bundle_does_not_retain_topic_render_buffer_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "buffer-lifetime"
    real_render = bundle_module.render_debug_topic_page
    references: list[weakref.ReferenceType[str]] = []

    class RenderedPage(str):
        pass

    def observed_render(*args: object, **kwargs: object) -> str:
        rendered = RenderedPage(real_render(*args, **kwargs))
        references.append(weakref.ref(rendered))
        return rendered

    real_validate = bundle_module._validate_staging
    alive_at_validation: list[bool] = []

    def observed_validation(*args: object, **kwargs: object) -> None:
        gc.collect()
        alive_at_validation.extend(reference() is not None for reference in references)
        real_validate(*args, **kwargs)

    monkeypatch.setattr(bundle_module, "render_debug_topic_page", observed_render)
    monkeypatch.setattr(bundle_module, "_validate_staging", observed_validation)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert alive_at_validation and not any(alive_at_validation)


def test_bundle_rejects_a_broken_same_document_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "broken-fragment"
    real_render = bundle_module.render_debug_topic_page

    def broken_render(*args: object, **kwargs: object) -> str:
        return real_render(*args, **kwargs).replace(
            "</main>", '<a href="#missing-anchor">broken</a></main>', 1
        )

    monkeypatch.setattr(bundle_module, "render_debug_topic_page", broken_render)
    with pytest.raises(ValueError, match="fragment"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()


def test_summary_projection_projects_only_safe_counts_and_attention_state(
    tmp_path: Path,
) -> None:
    # Keep the first production import in the test body so the initial RED run
    # reports a genuine test failure rather than a collection error.
    from trec_rag.competition_debug_bundle import build_run_summary

    fixture = build_run(
        tmp_path,
        (
            TopicSpec("alpha-topic", "private alpha narrative", candidate_documents=3),
            TopicSpec("beta-topic", "private beta narrative", candidate_documents=7),
        ),
    )
    data = load_debug_report_data(fixture.retrieval_config, rag_config_path=fixture.rag_config)
    fallback_result = replace(data.topics[1].canonical_results[0], state="fallback_extractive")
    fallback_topic = replace(
        data.topics[1],
        original_only_fallback=True,
        canonical_results=(fallback_result,),
    )

    summary = build_run_summary(replace(data, topics=(data.topics[0], fallback_topic)), None)

    assert summary.completed_topics == 2
    assert summary.fallback_topics == 1
    assert summary.submitted_documents == 2
    assert [topic.health for topic in summary.topics] == ["complete", "fallback"]
    assert summary.distributions["depth"] == {"minimum": 1, "median": 1, "maximum": 1}
    assert "narrative" not in {field.name for field in fields(summary.topics[0])}
    assert "docid" not in {field.name for field in fields(summary.topics[0])}


def test_summary_projection_counts_generated_queries_nuggets_and_safe_topic_links(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)

    summary = build_run_summary(data, None)

    assert [topic.topic_id for topic in summary.topics] == ["alpha-topic", "beta-topic"]
    assert [topic.href for topic in summary.topics] == [
        "topics/alpha-topic.html",
        "topics/beta-topic.html",
    ]
    assert [topic.queries for topic in summary.topics] == [
        sum(len(item.bm25_queries) for item in topic.subnarratives)
        for topic in data.topics
    ]
    assert [topic.nuggets for topic in summary.topics] == [
        len(topic.canonical_nuggets) for topic in data.topics
    ]


def _evaluation_fixture(tmp_path: Path):
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    return fixture, build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        work_dir=tmp_path / "evaluation",
        repository_root=REPOSITORY_ROOT,
        cache_root=tmp_path / "judge-cache",
        qrels_path=fixture.qrels(),
        judge=lambda _task: JudgeOutcome(status="completed", support_label="FS"),
        judge_settings=JudgeSettings(
            provider="fixture",
            model="fixture-model",
            thinking="disabled",
            temperature=0.0,
            system_prompt="fixture prompt",
            agent_binary="fixture-agent",
        ),
        created_utc="2026-08-09T00:00:00+00:00",
    )


def test_evaluation_overlay_preserves_families_scope_and_unavailability(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    _fixture, bundle = _evaluation_fixture(tmp_path)
    overlay = load_evaluation_overlay(bundle.manifest_path, ("beta-topic",))

    assert overlay.topic_ids == ("beta-topic",)
    assert [family.key for family in overlay.families] == [
        "retrieval",
        "nugget_coverage",
        "citation_support",
    ]
    retrieval = overlay.families[0]
    assert retrieval.label == "Retrieval relevance"
    assert "ndcg@10" in retrieval.definitions
    assert retrieval.macro_rule.startswith("Unweighted mean")
    assert "ndcg@10" in retrieval.per_topic["beta-topic"]
    assert "run_id" not in retrieval.per_topic["beta-topic"]
    assert retrieval.macro_availability.available is False
    assert "2-topic evaluation scope" in retrieval.macro_availability.reason
    nuggets = overlay.families[1]
    assert nuggets.macro == {}
    assert nuggets.macro_availability.available is False
    assert "no released gold-nugget file" in nuggets.macro_availability.reason


def _mutated_manifest(tmp_path: Path, mutate) -> Path:
    _fixture, bundle = _evaluation_fixture(tmp_path)
    payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "mutated-evaluation-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "case,mutate,selected",
    (
        (
            "missing selected topic",
            lambda _payload: None,
            ("missing-topic",),
        ),
        (
            "conflicting relative order",
            lambda _payload: None,
            ("beta-topic", "alpha-topic"),
        ),
        (
            "non-finite metric",
            lambda payload: payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"].update(
                {"ndcg@10": float("nan")}
            ),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "duplicate topic IDs",
            lambda payload: payload["scope"]["topic_ids"].append("alpha-topic"),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "unavailable state without reason",
            lambda payload: payload["metrics"]["nugget_coverage"]["macro_availability"].update(
                {"available": False, "reason": ""}
            ),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "unknown evaluation schema",
            lambda payload: payload.update({"schema_version": "unknown-evaluation-schema"}),
            ("alpha-topic", "beta-topic"),
        ),
    ),
)
def test_evaluation_overlay_rejects_invalid_manifest_contract(
    tmp_path: Path, case: str, mutate, selected: tuple[str, ...]
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(tmp_path, mutate)
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, selected)


def test_evaluation_overlay_rejects_missing_available_selected_metric_row(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(
        tmp_path,
        lambda payload: payload["metrics"]["retrieval"]["per_topic"].pop(
            "beta-topic"
        ),
    )
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, ("beta-topic",))


def test_evaluation_overlay_rejects_incomplete_available_macro_metrics(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(
        tmp_path,
        lambda payload: payload["metrics"]["retrieval"]["macro"].pop("ndcg@10"),
    )
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, ("alpha-topic", "beta-topic"))


def test_evaluation_overlay_rejects_manifest_replacement_between_snapshot_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_rag.competition_debug_bundle as debug_bundle

    _fixture, bundle = _evaluation_fixture(tmp_path)
    original_load_manifest = debug_bundle.load_manifest

    def replace_before_load(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")
        return original_load_manifest(path)

    monkeypatch.setattr(debug_bundle, "load_manifest", replace_before_load)
    try:
        overlay = debug_bundle.load_evaluation_overlay(
            bundle.manifest_path, ("alpha-topic", "beta-topic")
        )
    except EvaluationError:
        return
    assert overlay.manifest_sha256 == sha256(bundle.manifest_path.read_bytes()).hexdigest()


def test_evaluation_overlay_rejects_aba_manifest_replacement_at_load_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_rag.competition_debug_bundle as debug_bundle

    _fixture, bundle = _evaluation_fixture(tmp_path)
    snapshot = bundle.manifest_path.read_bytes()
    snapshot_manifest = json.loads(snapshot)
    original_load_manifest = debug_bundle.load_manifest

    def replace_and_restore(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = original_load_manifest(path)
        path.write_bytes(snapshot)
        return loaded

    monkeypatch.setattr(debug_bundle, "load_manifest", replace_and_restore)
    try:
        overlay = debug_bundle.load_evaluation_overlay(
            bundle.manifest_path, ("alpha-topic", "beta-topic")
        )
    except EvaluationError:
        return
    retrieval = overlay.families[0]
    assert retrieval.per_topic["alpha-topic"]["ndcg@10"] == snapshot_manifest[
        "metrics"
    ]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"]
    assert overlay.manifest_sha256 == sha256(snapshot).hexdigest()


def _evaluated_two_topic_data(tmp_path: Path):
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    fixture = build_run(
        tmp_path,
        (
            TopicSpec("alpha-topic", "private alpha narrative </script><script>"),
            TopicSpec("beta-topic", "private beta narrative"),
        ),
    )
    evaluation = build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        work_dir=tmp_path / "evaluation",
        repository_root=REPOSITORY_ROOT,
        cache_root=tmp_path / "judge-cache",
        qrels_path=fixture.qrels(),
        judge=lambda _task: JudgeOutcome(status="completed", support_label="FS"),
        judge_settings=JudgeSettings(
            provider="fixture",
            model="fixture-model",
            thinking="disabled",
            temperature=0.0,
            system_prompt="fixture prompt",
            agent_binary="fixture-agent",
        ),
        created_utc="2026-08-09T00:00:00+00:00",
    )
    data = load_debug_report_data(
        fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
    )
    overlay = load_evaluation_overlay(evaluation.manifest_path, fixture.topic_ids)
    return fixture, data, overlay


def test_summary_html_shows_health_scores_and_unavailability_without_private_text(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture, data, overlay = _evaluated_two_topic_data(tmp_path)
    summary = build_run_summary(data, overlay)
    private_docids = tuple(fixture.docids.values())

    page = render_bundle_summary(summary, denylist=private_docids)

    assert page.startswith("<!doctype html>")
    assert "Run summary" in page
    assert "Retrieval relevance" in page
    assert "Nugget or obligation coverage" in page
    assert "Answer and citation quality" in page
    assert "Unavailable" in page
    assert "no released gold-nugget file" in page
    assert "private alpha narrative" not in page
    assert "</script><script>" not in page
    assert "<script>private alpha narrative" not in page
    assert all(docid not in page for docid in private_docids)
    assert 'href="topics/alpha-topic.html"' in page
    assert 'data-sort-kind="number"' in page
    assert 'type="search"' in page


def test_summary_html_keeps_metric_families_definitions_and_macro_rules(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    _fixture, data, overlay = _evaluated_two_topic_data(tmp_path)
    page = render_bundle_summary(build_run_summary(data, overlay), denylist=())

    assert "Unweighted mean over the topic cells" in page
    assert "Normalized discounted cumulative gain" in page
    assert page.index("Retrieval relevance") < page.index("Nugget or obligation coverage")
    assert page.index("Nugget or obligation coverage") < page.index("Answer and citation quality")
    retrieval_names = sorted(overlay.families[0].definitions)
    assert [page.index(f'data-metric="retrieval:{name}"') for name in retrieval_names] == sorted(
        page.index(f'data-metric="retrieval:{name}"') for name in retrieval_names
    )


def test_summary_without_evaluation_is_explicit_and_keeps_official_row_order(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    page = render_bundle_summary(build_run_summary(data, None), denylist=())

    assert "Evaluation not supplied" in page
    assert page.index('data-topic-id="alpha-topic"') < page.index('data-topic-id="beta-topic"')
    assert "0.000000" not in page


def test_summary_privacy_scan_rejects_a_run_derived_collision(tmp_path: Path) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)

    with pytest.raises(ReportPrivacyError, match="private input value"):
        render_bundle_summary(
            build_run_summary(data, None),
            denylist=("alpha-topic",),
        )


def test_summary_denylist_real_fixture_can_render_without_invalid_document_digest_access(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import (
        _summary_denylist,
        build_run_summary,
        render_bundle_summary,
    )

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    denylist = _summary_denylist(data)

    page = render_bundle_summary(build_run_summary(data, None), denylist=denylist)

    assert page.startswith("<!doctype html>")
    assert denylist


def test_summary_denylist_collects_all_query_and_retrieval_digests(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import _summary_denylist

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    topic = data.topics[0]
    subnarrative = replace(
        topic.subnarratives[0],
        semantic_query_sha256="1" * 64,
        bm25_query_sha256s=("2" * 64,),
    )
    lane_provenance = replace(
        topic.new_documents[0].lane_provenance[0], text_sha256="3" * 64
    )
    new_document = replace(
        topic.new_documents[0], lane_provenance=(lane_provenance,)
    )
    retrieval_score = dict(topic.retrieval_output.documents[0].subnarrative_scores[0])
    retrieval_score.update(
        {
            "semantic_query_sha256": "4" * 64,
            "bm25_query_sha256s": ["5" * 64],
            "text_sha256": "6" * 64,
        }
    )
    retrieval_document = replace(
        topic.retrieval_output.documents[0],
        subnarrative_scores=(retrieval_score,),
    )
    updated_topic = replace(
        topic,
        subnarratives=(subnarrative,),
        new_documents=(new_document, *topic.new_documents[1:]),
        retrieval_output=replace(
            topic.retrieval_output, documents=(retrieval_document,)
        ),
    )
    updated_data = replace(data, topics=(updated_topic,))

    denylist = set(_summary_denylist(updated_data))

    assert {str(index) * 64 for index in range(1, 7)} <= denylist


def test_summary_static_rows_remain_official_order_before_attention_enhancement(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    fallback_result = replace(data.topics[1].canonical_results[0], state="fallback_extractive")
    fallback_topic = replace(
        data.topics[1],
        original_only_fallback=True,
        canonical_results=(fallback_result,),
    )
    summary = build_run_summary(replace(data, topics=(data.topics[0], fallback_topic)), None)

    page = render_bundle_summary(summary, denylist=())

    assert page.index('data-topic-id="alpha-topic"') < page.index('data-topic-id="beta-topic"')


def test_summary_needs_attention_reset_clears_metric_sort_direction() -> None:
    from trec_rag.competition_debug_bundle import _SUMMARY_SCRIPT

    reset_start = _SUMMARY_SCRIPT.index('reset?.addEventListener("click"')
    reset_end = _SUMMARY_SCRIPT.index("  });\n  sortNeedsAttention();", reset_start)
    reset_block = _SUMMARY_SCRIPT[reset_start:reset_end]

    assert 'button.dataset.direction = ""' in reset_block
    assert 'header.setAttribute("aria-sort", "none")' in reset_block


def test_summary_health_cards_report_each_fallback_kind_count(tmp_path: Path) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    fallback_result = replace(data.topics[0].canonical_results[0], state="fallback_extractive")
    fallback_alpha = replace(
        data.topics[0],
        canonical_results=(fallback_result,),
    )
    fallback_beta = replace(data.topics[1], original_only_fallback=True)
    summary = build_run_summary(replace(data, topics=(fallback_alpha, fallback_beta)), None)

    page = render_bundle_summary(summary, denylist=())

    assert "fallback_extractive: 1" in page
    assert "original_only: 1" in page

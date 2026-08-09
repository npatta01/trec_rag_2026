from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from trec_rag.retrieval_baseline_input_bundle import (
    InputBundleError,
    seal_input_directory,
    verify_input_directory,
)


def _staging_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "input"
    source = root / "source/facet-deepseek-b40-v3/rag2026-0"
    source.mkdir(parents=True)
    (source / "fixture.json").write_text("{}\n", encoding="utf-8")
    content = b"private text"
    digest = sha256(content).hexdigest()
    document = root / f"documents/v1/sha256/{digest[:2]}"
    document.mkdir(parents=True)
    (document / f"{digest}.utf8").write_bytes(content)
    scores = root / "portable-scores"
    scores.mkdir()
    (scores / f"{'b' * 64}.jsonl").write_text("{}\n", encoding="utf-8")
    return root


def test_seal_and_verify_input_directory_are_deterministic(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)

    first = seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats={"hits": 7, "misses": 11},
    )
    first_body = first.read_bytes()
    second = seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats={"hits": 7, "misses": 11},
    )

    assert second.read_bytes() == first_body
    verified = verify_input_directory(root, expected_topics=("rag2026-0",))
    assert verified["topic_ids"] == ["rag2026-0"]
    assert verified["cache_stats"] == {"hits": 7, "misses": 11}
    assert verified["member_count"] == 3


def test_verify_input_directory_rejects_tampered_member(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)
    seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats={"hits": 0, "misses": 1},
    )
    member = root / "source/facet-deepseek-b40-v3/rag2026-0/fixture.json"
    member.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(InputBundleError, match="digest|size"):
        verify_input_directory(root, expected_topics=("rag2026-0",))


def test_verify_input_directory_rejects_extra_and_duplicate_manifest_keys(
    tmp_path: Path,
) -> None:
    root = _staging_fixture(tmp_path)
    manifest = seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats={"hits": 0, "misses": 1},
    )
    (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(InputBundleError, match="undeclared|invalid"):
        verify_input_directory(root, expected_topics=("rag2026-0",))

    (root / "unexpected.txt").unlink()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(
        '{"schema_version":"first","schema_version":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputBundleError, match="duplicate|invalid"):
        verify_input_directory(root, expected_topics=("rag2026-0",))


def test_verify_input_directory_rejects_symlink_root_and_manifest(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)
    manifest = seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats={"hits": 0, "misses": 1},
    )
    linked_root = tmp_path / "linked-input"
    linked_root.symlink_to(root, target_is_directory=True)

    with pytest.raises(InputBundleError, match="root"):
        verify_input_directory(linked_root, expected_topics=("rag2026-0",))

    manifest_body = manifest.read_bytes()
    manifest.unlink()
    target = tmp_path / "manifest-target.json"
    target.write_bytes(manifest_body)
    manifest.symlink_to(target)
    with pytest.raises(InputBundleError, match="manifest"):
        verify_input_directory(root, expected_topics=("rag2026-0",))

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import tarfile

import pytest

from trec_rag.retrieval_candidate_core import (
    CandidateCore,
    CandidateLaneStat,
    candidate_core_to_dict,
)
from trec_rag.retrieval_baseline_input_bundle import (
    InputBundleError,
    build_input_directory,
    create_input_archive,
    export_portable_score_cache,
    extract_input_archive,
    import_portable_score_file,
    seal_input_directory,
    verify_input_directory,
)
from trec_rag.document_store import DocumentStore
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_baseline_runs import SemanticUnit, SourceDocument, TopicInput


def _staging_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "input"
    source = root / "source/facet-deepseek-b40-v3/rag2026-0"
    source.mkdir(parents=True)
    (source / "fixture.json").write_text("{}\n", encoding="utf-8")
    (source.parent / "retrieval_export_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "facet-deepseek-b40-v3",
                "export_code_commit": "a" * 40,
                "selected_topic_ids": ["rag2026-0"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    content = b"private text"
    digest = sha256(content).hexdigest()
    document = root / f"documents/v1/sha256/{digest[:2]}"
    document.mkdir(parents=True)
    (document / f"{digest}.utf8").write_bytes(content)
    scores = root / "portable-scores"
    scores.mkdir()
    (scores / f"{'b' * 64}.jsonl").write_text("{}\n", encoding="utf-8")
    core = CandidateCore(
        topic_id="rag2026-0",
        lane_scores_sha256="c" * 64,
        candidate_docids=("doc",),
        pre_fallback_count=0,
        fallback_used=True,
        admission_multiplicity_histogram={},
        lanes=(
            CandidateLaneStat(
                lane_name="original",
                median=0.0,
                mad=0.0,
                threshold=0.0,
                comparison="strictly_greater_than",
                observed_count=1,
                admitted_count=0,
                admitted_docids_sha256=sha256(b"").hexdigest(),
            ),
        ),
    )
    candidate_dir = root / "candidate-cores"
    candidate_dir.mkdir()
    (candidate_dir / "rag2026-0.json").write_text(
        json.dumps(
            candidate_core_to_dict(core),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _cache_stats(hits: int, misses: int) -> dict[str, int]:
    return {
        "candidate_documents": 1,
        "document_semantic_pairs": 1,
        "hits": hits,
        "misses": misses,
        "passage_pairs": hits + misses,
        "portable_rows": hits,
        "unique_cache_keys": hits + misses,
    }


def _topic_stats(root: Path, hits: int, misses: int) -> dict[str, dict[str, object]]:
    core_path = root / "candidate-cores/rag2026-0.json"
    return {
        "rag2026-0": {
            "cache_hits": hits,
            "cache_misses": misses,
            "candidate_core_sha256": sha256(core_path.read_bytes()).hexdigest(),
            "candidate_count": 1,
            "chunk_count": hits + misses,
            "document_semantic_pair_count": 1,
            "first_seen_cache_key_count": hits + misses,
            "passage_pair_count": hits + misses,
            "reused_prior_topic_key_count": 0,
            "semantic_unit_count": 1,
            "unique_cache_key_count": hits + misses,
        }
    }


def _seal(root: Path, *, hits: int, misses: int) -> Path:
    return seal_input_directory(
        root,
        topic_ids=("rag2026-0",),
        source_run_id="facet-deepseek-b40-v3",
        cache_stats=_cache_stats(hits, misses),
        canary_topic_ids=(),
        topic_stats=_topic_stats(root, hits, misses),
    )


def test_seal_and_verify_input_directory_are_deterministic(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)

    first = _seal(root, hits=7, misses=11)
    first_body = first.read_bytes()
    second = _seal(root, hits=7, misses=11)

    assert second.read_bytes() == first_body
    verified = verify_input_directory(root, expected_topics=("rag2026-0",))
    assert verified["topic_ids"] == ["rag2026-0"]
    assert verified["schema_version"] == "retrieval-baseline-private-input-v2"
    assert verified["cache_stats"] == _cache_stats(7, 11)
    assert verified["canary_topic_ids"] == []
    assert verified["topic_stats"] == _topic_stats(root, 7, 11)
    assert len(verified["source_export_manifest_sha256"]) == 64
    assert verified["source_export_code_commit"] == "a" * 40
    assert verified["member_count"] == 5


def test_verify_input_directory_rejects_tampered_member(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)
    _seal(root, hits=0, misses=1)
    member = root / "source/facet-deepseek-b40-v3/rag2026-0/fixture.json"
    member.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(InputBundleError, match="digest|size"):
        verify_input_directory(root, expected_topics=("rag2026-0",))


def test_verify_input_directory_rejects_extra_and_duplicate_manifest_keys(
    tmp_path: Path,
) -> None:
    root = _staging_fixture(tmp_path)
    manifest = _seal(root, hits=0, misses=1)
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
    manifest = _seal(root, hits=0, misses=1)
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


def test_export_and_import_complete_portable_score_cache(tmp_path: Path) -> None:
    source_root = tmp_path / "source-cache"
    source = MixedbreadPassageScorer(
        score_cache_root=source_root,
        device="cpu",
        batch_size=32,
    )
    try:
        source.score_cache.add_many(
            [("query-a", "passage-a", 1.25), ("query-b", "passage-b", -0.5)]
        )
    finally:
        source.score_cache.close()
    artifact = tmp_path / "complete.jsonl"

    exported = export_portable_score_cache(source_root, artifact)
    imported = import_portable_score_file(artifact, tmp_path / "target-cache")

    assert exported["row_count"] == 2
    assert exported["sha256"] == sha256(artifact.read_bytes()).hexdigest()
    assert imported["source_row_count"] == 2
    assert imported["inserted_count"] == 2
    target = MixedbreadPassageScorer(
        score_cache_root=tmp_path / "target-cache",
        device="cpu",
        batch_size=32,
        read_only=True,
    )
    try:
        assert target.score_cache.require_many(
            [("query-a", "passage-a"), ("query-b", "passage-b")]
        ) == [1.25, -0.5]
    finally:
        target.score_cache.close()


def test_extract_input_archive_verifies_inner_manifest(tmp_path: Path) -> None:
    root = _staging_fixture(tmp_path)
    manifest = _seal(root, hits=0, misses=1)
    archive = tmp_path / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(root).as_posix())

    receipt = extract_input_archive(
        archive,
        tmp_path / "extracted",
        expected_manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        expected_topics=("rag2026-0",),
    )

    assert receipt["member_count"] == 5
    assert receipt["topic_ids"] == ["rag2026-0"]
    assert verify_input_directory(tmp_path / "extracted")["member_count"] == 5


def test_create_input_archive_is_byte_deterministic_with_normalized_metadata(
    tmp_path: Path,
) -> None:
    root = _staging_fixture(tmp_path)
    manifest = _seal(root, hits=0, misses=1)

    first = create_input_archive(root, tmp_path / "first.tar.gz")
    second = create_input_archive(root, tmp_path / "second.tar.gz")

    assert first["archive_sha256"] == second["archive_sha256"]
    assert (tmp_path / "first.tar.gz").read_bytes() == (
        tmp_path / "second.tar.gz"
    ).read_bytes()
    assert first["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "first.tar.gz", "r:gz") as stream:
        members = stream.getmembers()
    assert [row.name for row in members] == sorted(row.name for row in members)
    assert all(
        row.uid == 0
        and row.gid == 0
        and row.uname == ""
        and row.gname == ""
        and row.mtime == 0
        and row.mode == 0o600
        and row.isfile()
        for row in members
    )


def test_build_input_directory_derives_core_and_copies_only_candidate_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "facet-deepseek-b40-v3"
    topic_root = source / "rag2026-0"
    for relative in (
        "topic-job-receipt.json",
        "decomposition.json",
        "decomposition/manifest.json",
        "decomposition/result.json",
        "retrieval/audit.json",
        "retrieval/complete.json",
        "retrieval/evidence-bundle.json",
        "scoring/complete.json",
        "scoring/selected_documents.jsonl",
        "scoring/selected_subnarrative_scores.jsonl",
        "scoring/selection.json",
        "canonical/retrieval-projection-manifest.json",
        "canonical/retrieval-projection.json",
    ):
        path = topic_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    lane_path = topic_root / "scoring/lane_scores.jsonl"
    lane_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in (
                {
                    "topic_id": "rag2026-0",
                    "lane_name": "original",
                    "docid": "d1",
                    "aggregate_score": 0.0,
                },
                {
                    "topic_id": "rag2026-0",
                    "lane_name": "original",
                    "docid": "d2",
                    "aggregate_score": 10.0,
                },
                {
                    "topic_id": "rag2026-0",
                    "lane_name": "facet:s1:text",
                    "docid": "d1",
                    "aggregate_score": 1.0,
                },
                {
                    "topic_id": "rag2026-0",
                    "lane_name": "facet:s1:text",
                    "docid": "d2",
                    "aggregate_score": 1.0,
                },
            )
        ),
        encoding="utf-8",
    )
    (source / "retrieval_export_manifest.json").write_text(
        json.dumps(
            {
                "run_id": source.name,
                "export_code_commit": "d" * 40,
                "selected_topic_ids": ["rag2026-0"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    document_root = tmp_path / "documents"
    store = DocumentStore(document_root)
    first = store.admit_text("first weak document")
    second = store.admit_text("second selected document")
    topic = TopicInput(
        topic_id="rag2026-0",
        narrative="narrative",
        subnarratives=(SemanticUnit("s1", "facet", ("unused",)),),
        documents=(
            SourceDocument("d1", first.content_sha256, 2, {"s1": 2}, "first weak document"),
            SourceDocument("d2", second.content_sha256, 1, {}, "second selected document"),
        ),
        source_sha256s={"scoring/lane_scores.jsonl": sha256(lane_path.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(
        "trec_rag.retrieval_baseline_input_bundle.load_topic_input",
        lambda *_args: topic,
    )
    cache_root = tmp_path / "cache"
    cache = MixedbreadPassageScorer(cache_root, device="cpu", batch_size=32)
    try:
        cache.score_cache.add_many(
            [(topic.narrative, "second selected document", 3.0)]
        )
    finally:
        cache.score_cache.close()

    receipt = build_input_directory(
        source_dir=source,
        document_store_root=document_root,
        score_cache_root=cache_root,
        topic_ids=("rag2026-0",),
        output_dir=tmp_path / "built",
        require_official_topic_set=False,
    )

    assert receipt["topic_ids"] == ["rag2026-0"]
    assert receipt["cache_stats"] == {
        "candidate_documents": 1,
        "document_semantic_pairs": 2,
        "hits": 1,
        "misses": 1,
        "passage_pairs": 2,
        "portable_rows": 1,
        "unique_cache_keys": 2,
    }
    built = tmp_path / "built"
    core = json.loads((built / "candidate-cores/rag2026-0.json").read_text())
    assert core["candidate_docids"] == ["d2"]
    copied_documents = list((built / "documents/v1/sha256").rglob("*.utf8"))
    assert [path.read_text(encoding="utf-8") for path in copied_documents] == [
        "second selected document"
    ]
    verified = verify_input_directory(built, expected_topics=("rag2026-0",))
    assert verified["topic_stats"]["rag2026-0"]["candidate_count"] == 1


def test_production_bundle_requires_the_exact_119_topic_set(tmp_path: Path) -> None:
    with pytest.raises(InputBundleError, match="119-topic"):
        build_input_directory(
            source_dir=tmp_path / "source",
            document_store_root=tmp_path / "documents",
            score_cache_root=tmp_path / "cache",
            topic_ids=("rag2026-0",),
            output_dir=tmp_path / "output",
            require_official_topic_set=True,
        )


def test_extract_input_archive_rejects_links_and_wrong_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.tar.gz"
    target = tmp_path / "target.txt"
    target.write_text("outside", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(target, arcname="source/file.txt")
        link = tarfile.TarInfo("documents/link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(target)
        stream.addfile(link)

    with pytest.raises(InputBundleError, match="regular files"):
        extract_input_archive(
            archive,
            tmp_path / "bad-extract",
            expected_manifest_sha256="a" * 64,
            expected_topics=("rag2026-0",),
        )

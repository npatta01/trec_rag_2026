from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
import zstandard

import trec_rag.competition_cache_bundle as bundle_module
from trec_rag.document_store import DocumentStore
from trec_rag.planning_cache import PlanningCache, build_planning_cache_identity
from trec_rag.rerank_score_cache import GlobalScoreCache, ScoreCacheContext
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    TransportIdentity,
)
from trec_rag.similarity_cache import (
    SimilarityCache,
    build_similarity_cache_identity,
)

from trec_rag.competition_cache_bundle import (
    CacheBundleConflictError,
    CacheBundleIntegrityError,
    MERGE_STATE_DIRECTORY,
    assert_no_incomplete_cache_bundle_merge,
    merge_bundles,
    pack_bundle,
    verify_bundle,
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    topics = root / "topics.tsv"
    topics.write_text("rag2026-0\tA tiny narrative.\n", encoding="utf-8")
    config = root / "config.yaml"
    config.write_text(
        """\
schema_version: facet_pilot_config_v2
experiment:
  id: shard-fixture
topics:
  path: topics.tsv
retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  query_sources: [original, subnarrative]
  documents_per_query: 1000
  corpus_epoch: fixture-epoch
passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350
nuggets:
  evidence_budget_per_subnarrative: 40
  maximum_claims_per_subnarrative: 20
  maximum_supporting_documents_per_claim: 3
""",
        encoding="utf-8",
    )

    topic_root = root / "outputs" / "shard-fixture" / "rag2026-0"
    artifact = topic_root / "retrieval" / "audit.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(_canonical_json({"topic_id": "rag2026-0"}))
    body = artifact.read_bytes()
    (artifact.parent / "complete.json").write_bytes(
        _canonical_json(
            {
                "artifacts": [
                    {
                        "bytes": len(body),
                        "relative_path": "retrieval/audit.json",
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                ],
                "phase": "retrieve",
                "topic_id": "rag2026-0",
            }
        )
    )

    planning = root / "cache" / "planning" / "validated" / "plan.json"
    planning.parent.mkdir(parents=True)
    planning.write_bytes(_canonical_json({"validated": True}))
    return config


def _write_manual_bundle(
    root: Path,
    *,
    declarations: list[dict[str, object]],
    entries: list[tuple[tarfile.TarInfo, bytes]],
    trailing: bytes = b"",
) -> Path:
    root.mkdir()
    manifest = _canonical_json(
        {
            "experiment_id": "fixture",
            "members": declarations,
            "schema_version": "trec-rag-cache-bundle-v1",
            "source_config_sha256": "a" * 64,
            "topic_id": "rag2026-0",
        }
    )
    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT
    ) as archive:
        manifest_info = _tar_entry("bundle-manifest.json", manifest)
        archive.addfile(manifest_info, io.BytesIO(manifest))
        for info, body in entries:
            archive.addfile(info, io.BytesIO(body) if info.isfile() else None)
    compressed = zstandard.ZstdCompressor(
        level=3, write_checksum=True, write_content_size=False
    ).compress(tar_buffer.getvalue() + trailing)
    (root / "bundle.tar.zst").write_bytes(compressed)
    (root / "bundle-complete.json").write_bytes(
        _canonical_json(
            {
                "archive": "bundle.tar.zst",
                "archive_sha256": hashlib.sha256(compressed).hexdigest(),
                "archive_size": len(compressed),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "member_count": len(declarations),
                "schema_version": "trec-rag-cache-bundle-v1",
                "topic_id": "rag2026-0",
            }
        )
    )
    return root


def _tar_entry(name: str, body: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(body)
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _declaration(path: str, body: bytes, *, kind: str = "immutable") -> dict[str, object]:
    return {
        "kind": kind,
        "mode": 0o600,
        "path": path,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
    }


def test_pack_is_byte_reproducible_and_verify_is_manifest_driven(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_receipt = pack_bundle(config, "rag2026-0", first.resolve())
    second_receipt = pack_bundle(config, "rag2026-0", second.resolve())

    assert (first / "bundle.tar.zst").read_bytes() == (
        second / "bundle.tar.zst"
    ).read_bytes()
    assert (first / "bundle-complete.json").read_bytes() == (
        second / "bundle-complete.json"
    ).read_bytes()
    assert first_receipt == second_receipt
    verified = verify_bundle(first)
    assert verified.topic_id == "rag2026-0"
    assert [member.path for member in verified.members] == sorted(
        member.path for member in verified.members
    )
    assert "cache/planning/validated/plan.json" in {
        member.path for member in verified.members
    }
    assert "outputs/shard-fixture/rag2026-0/retrieval/complete.json" in {
        member.path for member in verified.members
    }


def test_pack_requires_complete_document_cas_closure(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    missing_digest = hashlib.sha256(b"missing document").hexdigest()
    derivation = (
        config.parent
        / "cache/retrieval/pyserini_remote/v2/aa/request/derived/derivation"
    )
    derivation.mkdir(parents=True)
    (derivation / "derivation-manifest.json").write_bytes(
        _canonical_json({"document_closure": [missing_digest]})
    )

    with pytest.raises(CacheBundleIntegrityError, match="document.*closure"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_validates_retrieval_derivation_and_exact_document_closure(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    query = "fixture query"
    identity = TransportIdentity.from_query(
        query_text=query,
        index_id="climbmix-400b",
        endpoint_identity="https://pyserini.example/search",
        corpus_epoch="fixture-epoch",
        hits=1000,
    )
    normalizer = OrganizerTextNormalizer()
    derivation = DerivationIdentity.from_normalizer(normalizer)
    retrieval = RetrievalCache(
        config.parent / "cache/retrieval/pyserini_remote",
        DocumentStore(config.parent / "cache/documents/v1"),
        normalizer,
    )
    body = "Exact organizer document body."
    raw = _canonical_json(
        {
            "api": "v1",
            "candidates": [
                {"doc": body, "docid": "doc-1", "rank": 1, "score": 3.5}
            ],
            "index": "climbmix-400b",
            "query": {"text": query},
        }
    )[:-1]
    retrieval.commit(identity, derivation, query, raw)

    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    paths = {member.path for member in verify_bundle(bundle).members}
    document_digest = hashlib.sha256(body.encode()).hexdigest()

    assert (
        f"cache/documents/v1/sha256/{document_digest[:2]}/{document_digest}.utf8"
        in paths
    )
    assert any(path.endswith("/transport-manifest.json") for path in paths)
    assert any(path.endswith("/derivation-manifest.json") for path in paths)


def test_pack_requires_validated_planning_entries_for_selected_topic(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    identity = build_planning_cache_identity(
        request_body=b'{"messages":[]}',
        endpoint="https://openrouter.example/api",
        model="deepseek/fixture",
        prompt_version="fixture-v1",
        schema_version="fixture-schema-v1",
        topic_id="rag2026-0",
        narrative="A tiny narrative.",
    )
    entry = PlanningCache(config.parent / "cache").store(
        identity, {"facets": ["one"]}
    )
    first = (tmp_path / "first").resolve()

    pack_bundle(config, "rag2026-0", first)

    assert f"cache/{entry.relative_to(config.parent / 'cache').as_posix()}" in {
        member.path for member in verify_bundle(first).members
    }
    entry.write_bytes(entry.read_bytes().removesuffix(b"\n") + b" \n")
    with pytest.raises(CacheBundleIntegrityError, match="planning cache"):
        pack_bundle(config, "rag2026-0", (tmp_path / "tampered").resolve())


def test_verify_rejects_trailing_decompressed_payload(tmp_path: Path) -> None:
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[],
        entries=[],
        trailing=b"undeclared trailing payload",
    )

    with pytest.raises(CacheBundleIntegrityError, match="decompressed|trailing"):
        verify_bundle(bundle)


def test_verify_rejects_undeclared_bundle_directory_file(tmp_path: Path) -> None:
    bundle = _write_manual_bundle(
        tmp_path / "bundle", declarations=[], entries=[]
    )
    (bundle / "provider-response.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CacheBundleIntegrityError, match="undeclared"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape",
        "/absolute",
        "C:/windows-drive",
        "cache/../../escape",
        "cache\\..\\escape",
    ),
)
def test_verify_rejects_traversal_absolute_and_drive_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    body = b"unsafe"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(unsafe_name, body)],
        entries=[(_tar_entry(unsafe_name, body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="unsafe"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "member_type",
    (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE),
)
def test_verify_rejects_links_and_device_members(
    tmp_path: Path, member_type: bytes
) -> None:
    info = _tar_entry("cache/unsafe", b"")
    info.type = member_type
    info.linkname = "cache/target"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration("cache/unsafe", b"")],
        entries=[(info, b"")],
    )

    with pytest.raises(CacheBundleIntegrityError, match="regular file"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "paths",
    (
        ("cache/same", "cache/same"),
        ("cache/Entry.json", "cache/entry.json"),
        ("cache/caf\N{LATIN SMALL LETTER E WITH ACUTE}.json", "cache/cafe\N{COMBINING ACUTE ACCENT}.json"),
    ),
)
def test_verify_rejects_duplicate_case_and_canonical_collisions(
    tmp_path: Path, paths: tuple[str, str]
) -> None:
    first, second = paths
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(first, b"a"), _declaration(second, b"b")],
        entries=[(_tar_entry(first, b"a"), b"a"), (_tar_entry(second, b"b"), b"b")],
    )

    with pytest.raises(CacheBundleIntegrityError, match="duplicate|collision"):
        verify_bundle(bundle)


def test_verify_rejects_undeclared_archive_member(tmp_path: Path) -> None:
    body = b"undeclared"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[],
        entries=[(_tar_entry("cache/extra", body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="undeclared"):
        verify_bundle(bundle)


@pytest.mark.parametrize("tamper", ("digest", "size"))
def test_verify_rejects_declared_digest_and_size_mismatch(
    tmp_path: Path, tamper: str
) -> None:
    body = b"authenticated"
    declaration = _declaration("cache/entry", body)
    if tamper == "digest":
        declaration["sha256"] = "0" * 64
    else:
        declaration["size"] = len(body) + 1
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[declaration],
        entries=[(_tar_entry("cache/entry", body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="digest|size"):
        verify_bundle(bundle)


def test_verify_enforces_member_and_decompression_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"two bytes"
    bundle = _write_manual_bundle(
        tmp_path / "member",
        declarations=[_declaration("cache/entry", body)],
        entries=[(_tar_entry("cache/entry", body), body)],
    )
    monkeypatch.setattr(bundle_module, "MAX_MEMBER_BYTES", 1)
    with pytest.raises(CacheBundleIntegrityError, match="size limit"):
        verify_bundle(bundle)

    empty = _write_manual_bundle(
        tmp_path / "decompressed", declarations=[], entries=[]
    )
    monkeypatch.setattr(bundle_module, "MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(bundle_module, "MAX_DECOMPRESSED_BYTES", 1024)
    with pytest.raises(CacheBundleIntegrityError, match="decompressed-size"):
        verify_bundle(empty)


def test_merge_installs_verified_files_and_is_idempotent(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()

    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="strict",
    )
    second = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="strict",
    )

    assert second == first
    assert (cache_root / "planning/validated/plan.json").read_bytes() == (
        config.parent / "cache/planning/validated/plan.json"
    ).read_bytes()
    assert (
        outputs_root
        / "shard-fixture/rag2026-0/retrieval/complete.json"
    ).is_file()
    state = cache_root / MERGE_STATE_DIRECTORY / first.merge_id
    assert (state / "prepare.json").is_file()
    assert (state / "conflicts.json").is_file()
    assert first.completion_path == state / "complete.json"
    assert first.completion_path.is_file()


def test_merge_preflights_strict_conflicts_before_destination_mutation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    conflicting = cache_root / "planning/validated/plan.json"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"destination authority")

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="strict",
        )

    assert conflicting.read_bytes() == b"destination authority"
    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_merge_retry_converges_from_durable_prepare_journal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    installed_events = 0

    def interrupt_after_first_install(phase: str) -> None:
        nonlocal installed_events
        if phase.startswith("installed:"):
            installed_events += 1
            if installed_events == 1:
                raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=interrupt_after_first_install,
        )
    with pytest.raises(CacheBundleIntegrityError, match="incomplete"):
        assert_no_incomplete_cache_bundle_merge(cache_root)

    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )

    assert receipt.completion_path.is_file()
    assert receipt.operation_count == receipt.installed_count + receipt.identical_count
    assert_no_incomplete_cache_bundle_merge(cache_root)


def test_pack_exports_portable_scores_and_merge_imports_transactionally(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source_scores = GlobalScoreCache(config.parent / "cache/reranker", context)
    source_scores.seed_many(
        [("query", "passage", 1.25)],
        source_path="fixture",
        source_sha256="1" * 64,
    )
    source_scores.close()
    bundle = (tmp_path / "bundle").resolve()

    pack_bundle(config, "rag2026-0", bundle)
    verified = verify_bundle(bundle)

    score_members = [member for member in verified.members if member.kind == "score"]
    assert [member.path for member in score_members] == [
        f"portable-scores/{context.context_sha256}.jsonl"
    ]
    assert not any(member.path.endswith(".sqlite3") for member in verified.members)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()

    def interrupt_after_score_import(phase: str) -> None:
        if phase.startswith("score-imported:"):
            raise RuntimeError("simulated post-transaction interruption")

    with pytest.raises(RuntimeError, match="post-transaction"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=interrupt_after_score_import,
        )
    with pytest.raises(CacheBundleIntegrityError, match="incomplete"):
        assert_no_incomplete_cache_bundle_merge(cache_root)
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert imported.get(query_text="query", text="passage") == 1.25
    finally:
        imported.close()
    assert receipt.score_import_count == 1
    assert_no_incomplete_cache_bundle_merge(cache_root)


def test_score_conflicts_are_strict_by_default_and_keep_existing_is_audited(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)

    cache_root = (tmp_path / "destination-cache").resolve()
    destination = GlobalScoreCache(cache_root / "reranker", context)
    destination.seed_many(
        [("query", "passage", 2.5)],
        source_path="destination",
        source_sha256="2" * 64,
    )
    destination.close()
    outputs_root = (tmp_path / "destination-outputs").resolve()
    with pytest.raises(CacheBundleConflictError, match="numerical score"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )
    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()

    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )
    retained = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert retained.get(query_text="query", text="passage") == 2.5
    finally:
        retained.close()
    audit = json.loads(
        (receipt.completion_path.parent / "conflicts.json").read_text(encoding="utf-8")
    )
    assert audit["conflicts"] == [
        {
            "cache_key": audit["conflicts"][0]["cache_key"],
            "context_sha256": context.context_sha256,
            "existing_score_hex": (2.5).hex(),
            "kind": "score",
            "query_sha256": hashlib.sha256(b"query").hexdigest(),
            "resolution": "kept-existing",
            "scope": "destination",
            "source_score_hex": (1.25).hex(),
            "text_sha256": hashlib.sha256(b"passage").hexdigest(),
        }
    ]


def test_keep_existing_is_limited_to_similarity_and_audits_numerical_conflict(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    SimilarityCache(config.parent / "cache").store(
        identity, ((1.0, 0.25), (0.25, 1.0))
    )
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    SimilarityCache(cache_root).store(
        identity, ((1.0, 0.5), (0.5, 1.0))
    )

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=(tmp_path / "strict-outputs").resolve(),
            bundle_dirs=(bundle,),
        )
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=(tmp_path / "kept-outputs").resolve(),
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )

    assert SimilarityCache(cache_root).load(identity) == (
        (1.0, 0.5),
        (0.5, 1.0),
    )
    audit = json.loads(
        (receipt.completion_path.parent / "conflicts.json").read_text(encoding="utf-8")
    )
    assert [(row["kind"], row["resolution"]) for row in audit["conflicts"]] == [
        ("similarity", "kept-existing")
    ]


def test_keep_existing_never_overrides_immutable_conflicts(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    conflicting = cache_root / "planning/validated/plan.json"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"immutable destination")

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=(tmp_path / "merged-outputs").resolve(),
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

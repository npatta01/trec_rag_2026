from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import zstandard

import trec_rag.competition_cache_bundle as bundle_module
from trec_rag.document_store import DocumentStore
from trec_rag.facet_pilot_config import (
    load_facet_pilot_config,
    select_configured_topics,
)
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
from trec_rag.topic_dispatch import (
    TopicJob,
    TopicJobReceipt,
    publish_topic_receipt,
)

from trec_rag.competition_cache_bundle import (
    CacheBundleConflictError,
    CacheBundleIntegrityError,
    MERGE_STATE_DIRECTORY,
    assert_no_incomplete_cache_bundle_merge,
    incomplete_merge_journals,
    merge_bundles,
    pack_bundle,
    verify_bundle,
)


_REAL_PRODUCTION_CHECKPOINT_VALIDATOR = (
    bundle_module._validate_production_topic_checkpoint
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


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@pytest.fixture(autouse=True)
def _use_synthetic_checkpoint_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep tiny bundle fixtures while production uses the deep validator."""
    monkeypatch.setattr(
        bundle_module,
        "_validate_production_topic_checkpoint",
        lambda *_args, **_kwargs: None,
    )


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
    artifact_sets = {
        "retrieval": (
            "decomposition.json",
            "retrieval/audit.json",
            "retrieval/evidence-bundle.json",
        ),
        "scoring": (
            "scoring/lane_scores.jsonl",
            "scoring/selected_documents.jsonl",
            "scoring/selection.json",
            "scoring/selected_subnarrative_scores.jsonl",
        ),
        "canonical": (
            "canonical/handoff/candidate-requests.jsonl",
            "canonical/handoff/selection-contexts.jsonl",
            "canonical/handoff/handoff-manifest.json",
            "records.sqlite3",
            "canonical/records-manifest.json",
            "canonical/subnarrative-selections.jsonl",
            "canonical/selection-manifest.json",
            "canonical/canonical-nuggets.jsonl",
            "canonical/canonical-nugget-manifest.json",
            "canonical/retrieval-projection.json",
            "canonical/retrieval-projection-manifest.json",
            "canonical/generation-projection.json",
            "canonical/generation-projection-manifest.json",
        ),
    }
    for relative in {path for paths in artifact_sets.values() for path in paths}:
        path = topic_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "records.sqlite3":
            path.write_bytes(b"authenticated topic-records fixture")
        else:
            path.write_bytes(
                _canonical_json({"fixture": relative, "topic_id": "rag2026-0"})
            )
    decomposition_result = topic_root / "decomposition/result.json"
    decomposition_result.parent.mkdir(parents=True)
    decomposition_result.write_bytes(
        _canonical_json({"fixture": "decomposition source", "topic_id": "rag2026-0"})
    )
    decomposition_body = decomposition_result.read_bytes()
    (decomposition_result.parent / "manifest.json").write_bytes(
        _pretty_json(
            {
                "planner": {"fixture": True},
                "result_bytes": len(decomposition_body),
                "result_file": "result.json",
                "result_sha256": hashlib.sha256(decomposition_body).hexdigest(),
                "schema_version": "facet-decomposition-manifest-v1",
            }
        )
    )
    for phase, relatives in artifact_sets.items():
        manifest = topic_root / phase / "complete.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(
            _pretty_json(
                {
                    "artifacts": [
                        {
                            "bytes": (topic_root / relative).stat().st_size,
                            "relative_path": relative,
                            "sha256": hashlib.sha256(
                                (topic_root / relative).read_bytes()
                            ).hexdigest(),
                        }
                        for relative in relatives
                    ],
                    **(
                        {
                            "decomposition_source_sha256": hashlib.sha256(
                                decomposition_body
                            ).hexdigest(),
                            "retriever": {"fixture": True},
                        }
                        if phase == "retrieval"
                        else {}
                    ),
                    "phase": "retrieve" if phase == "retrieval" else phase,
                    "topic_id": "rag2026-0",
                }
            )
        )
    config_bytes = config.read_bytes()
    job = TopicJob(
        topic_id="rag2026-0",
        run_id="shard-fixture",
        config_path=config.resolve(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        topic_root=topic_root.resolve(),
    )
    projection = topic_root / "canonical/retrieval-projection-manifest.json"
    publish_topic_receipt(
        job,
        TopicJobReceipt(
            topic_id="rag2026-0",
            projection_manifest_sha256=hashlib.sha256(
                projection.read_bytes()
            ).hexdigest(),
            status="complete",
            stopping_reason="coverage_sufficient",
        ),
    )

    identity = build_planning_cache_identity(
        request_body=b'{"messages":[]}',
        endpoint="https://openrouter.example/api",
        model="deepseek/fixture",
        prompt_version="fixture-v1",
        schema_version="fixture-schema-v1",
        topic_id="rag2026-0",
        narrative="A tiny narrative.",
    )
    PlanningCache(root / "cache").store(identity, {"validated": True})
    return config


def _planning_entry(root: Path) -> Path:
    return next((root / "planning-cache-v1").glob("*/*.json"))


def _write_manual_bundle(
    root: Path,
    *,
    declarations: list[dict[str, object]],
    entries: list[tuple[tarfile.TarInfo, bytes]],
    trailing: bytes = b"",
    source_config_sha256: str = "a" * 64,
) -> Path:
    root.mkdir()
    manifest = _canonical_json(
        {
            "experiment_id": "fixture",
            "members": declarations,
            "schema_version": "trec-rag-cache-bundle-v1",
            "source_config_sha256": source_config_sha256,
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


def _declaration(
    path: str, body: bytes, *, kind: str = "immutable"
) -> dict[str, object]:
    return {
        "kind": kind,
        "mode": 0o600,
        "path": path,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
    }


def _manual_config_bytes() -> bytes:
    return b"""\
schema_version: facet_pilot_config_v2
experiment:
  id: fixture
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
"""


def test_pack_is_byte_reproducible_and_verify_is_manifest_driven(
    tmp_path: Path,
) -> None:
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
    assert any(
        member.path.startswith("cache/planning-cache-v1/")
        for member in verified.members
    )
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
            "candidates": [{"doc": body, "docid": "doc-1", "rank": 1, "score": 3.5}],
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
    entry = _planning_entry(config.parent / "cache")
    first = (tmp_path / "first").resolve()

    pack_bundle(config, "rag2026-0", first)

    assert f"cache/{entry.relative_to(config.parent / 'cache').as_posix()}" in {
        member.path for member in verify_bundle(first).members
    }
    entry.write_bytes(entry.read_bytes().removesuffix(b"\n") + b" \n")
    with pytest.raises(CacheBundleIntegrityError, match="planning cache"):
        pack_bundle(config, "rag2026-0", (tmp_path / "tampered").resolve())


def test_pack_rejects_fake_empty_checkpoint_seal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    seal = config.parent / "outputs/shard-fixture/rag2026-0/retrieval/complete.json"
    seal.write_bytes(_canonical_json({}))

    with pytest.raises(
        CacheBundleIntegrityError, match="checkpoint|dispatch|projection"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_requires_production_topic_checkpoint_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path / "repo")

    def reject(*_args: object, **_kwargs: object) -> None:
        raise CacheBundleIntegrityError("production checkpoint rejected")

    monkeypatch.setattr(bundle_module, "_validate_production_topic_checkpoint", reject)

    with pytest.raises(CacheBundleIntegrityError, match="production checkpoint"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_production_checkpoint_validator_cross_binds_dispatch_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path / "repo")
    config_bytes = config_path.read_bytes()
    config = load_facet_pilot_config(config_path, source_bytes=config_bytes)
    topic = select_configured_topics(config, topic_ids=("rag2026-0",))[0]
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    producer_sha256 = "d" * 64
    projection_sha256 = hashlib.sha256(
        (
            config.output_dir
            / topic.id
            / "canonical/retrieval-projection-manifest.json"
        ).read_bytes()
    ).hexdigest()
    projection = SimpleNamespace(
        source_seals=(
            ("config_sha256", config_sha256),
            ("decomposition_producer_sha256", producer_sha256),
        ),
        manifest_sha256=projection_sha256,
        retrieval_status="complete",
        retrieval_stopping_reason="coverage_sufficient",
    )
    dispatch = SimpleNamespace(
        topic_id=topic.id,
        projection_manifest_sha256=projection_sha256,
        status="complete",
        stopping_reason="coverage_sufficient",
    )
    captured: dict[str, object] = {}

    import trec_rag.competition_retrieval as retrieval_runner
    import trec_rag.retrieval_export as retrieval_export
    import trec_rag.topic_dispatch as topic_dispatch

    monkeypatch.setattr(
        retrieval_runner,
        "_decomposition_producer_sha256",
        lambda *_args, **_kwargs: producer_sha256,
    )
    monkeypatch.setattr(
        retrieval_export,
        "read_topic_projection_receipt",
        lambda *_args, **_kwargs: projection,
    )

    def validate(*_args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return (projection,)

    monkeypatch.setattr(
        retrieval_export, "validate_retrieval_topic_checkpoints", validate
    )
    monkeypatch.setattr(
        topic_dispatch, "read_topic_receipt", lambda *_args, **_kwargs: dispatch
    )

    _REAL_PRODUCTION_CHECKPOINT_VALIDATOR(
        config,
        topic,
        config_path=config_path,
        config_bytes=config_bytes,
    )

    assert captured["expected_retriever_identity"] == {"fixture": True}
    assert captured["expected_decomposition_producer_sha256"] == {
        topic.id: producer_sha256
    }


def test_pack_rejects_lone_projection_manifest_as_a_topic_seal(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    topic_root = config.parent / "outputs/shard-fixture/rag2026-0"
    for path in sorted(topic_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    projection = topic_root / "canonical/retrieval-projection-manifest.json"
    projection.parent.mkdir(parents=True, exist_ok=True)
    projection.write_bytes(_canonical_json({"topic_id": "rag2026-0"}))

    with pytest.raises(
        CacheBundleIntegrityError, match="checkpoint|dispatch|projection"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_omits_unreceipted_topic_file_from_authenticated_closure(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    topic_root = config.parent / "outputs/shard-fixture/rag2026-0"
    extra = topic_root / "operator-note.json"
    extra.write_bytes(_canonical_json({"not": "checkpointed"}))
    bundle = (tmp_path / "bundle").resolve()

    pack_bundle(config, "rag2026-0", bundle)

    assert not any(
        member.path.endswith("operator-note.json")
        for member in verify_bundle(bundle).members
    )


def test_pack_rejects_stale_checkpoint_artifact_receipt(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    artifact = config.parent / "outputs/shard-fixture/rag2026-0/retrieval/audit.json"
    artifact.write_bytes(_canonical_json({"topic_id": "rag2026-X"}))

    with pytest.raises(CacheBundleIntegrityError, match="digest|hash|receipt|size"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


@pytest.mark.parametrize(
    "relative",
    ("qrels/poison.json", "qrels.json", "provider-responses/raw.json"),
)
def test_pack_rejects_sensitive_unreceipted_topic_extra(
    tmp_path: Path,
    relative: str,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    poison = config.parent / "outputs/shard-fixture/rag2026-0" / relative
    poison.parent.mkdir(exist_ok=True)
    poison.write_bytes(_canonical_json({"private": True}))

    with pytest.raises(
        CacheBundleIntegrityError, match="forbidden|sensitive|prohibited"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


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
    bundle = _write_manual_bundle(tmp_path / "bundle", declarations=[], entries=[])
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
    (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ),
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
        (
            "cache/caf\N{LATIN SMALL LETTER E WITH ACUTE}.json",
            "cache/cafe\N{COMBINING ACUTE ACCENT}.json",
        ),
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


def test_verify_rejects_nonadjacent_ancestor_collision(tmp_path: Path) -> None:
    paths = ("a", "a-foo", "a/b")
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(path, path.encode()) for path in paths],
        entries=[(_tar_entry(path, path.encode()), path.encode()) for path in paths],
    )

    with pytest.raises(CacheBundleIntegrityError, match="ancestor collision"):
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

    empty = _write_manual_bundle(tmp_path / "decompressed", declarations=[], entries=[])
    monkeypatch.setattr(bundle_module, "MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(bundle_module, "MAX_DECOMPRESSED_BYTES", 1024)
    with pytest.raises(CacheBundleIntegrityError, match="decompressed-size"):
        verify_bundle(empty)


def test_verify_rejects_prohibited_cache_member_even_when_self_declared(
    tmp_path: Path,
) -> None:
    config = _manual_config_bytes()
    poison = _canonical_json({"qrels": "private"})
    declarations = [
        _declaration("cache/qrels/poison.json", poison),
        _declaration("source-config/config.yaml", config, kind="config"),
    ]
    entries = [
        (_tar_entry("cache/qrels/poison.json", poison), poison),
        (_tar_entry("source-config/config.yaml", config), config),
    ]
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=declarations,
        entries=entries,
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="forbidden|prohibited"):
        verify_bundle(bundle)


def test_verify_requires_exactly_one_authenticated_source_config(
    tmp_path: Path,
) -> None:
    missing = _write_manual_bundle(tmp_path / "missing", declarations=[], entries=[])
    with pytest.raises(CacheBundleIntegrityError, match="exactly one.*config"):
        verify_bundle(missing)

    config = _manual_config_bytes()
    duplicate_path = "source-config/duplicate.yaml"
    declarations = [
        _declaration("source-config/config.yaml", config, kind="config"),
        _declaration(duplicate_path, config, kind="config"),
    ]
    duplicate = _write_manual_bundle(
        tmp_path / "duplicate",
        declarations=declarations,
        entries=[
            (_tar_entry("source-config/config.yaml", config), config),
            (_tar_entry(duplicate_path, config), config),
        ],
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )
    with pytest.raises(CacheBundleIntegrityError, match="exactly one.*config"):
        verify_bundle(duplicate)


def test_verify_binds_source_config_digest_and_experiment(tmp_path: Path) -> None:
    config = _manual_config_bytes()
    declaration = _declaration("source-config/config.yaml", config, kind="config")
    entry = (_tar_entry("source-config/config.yaml", config), config)
    wrong_digest = _write_manual_bundle(
        tmp_path / "wrong-digest",
        declarations=[declaration],
        entries=[entry],
        source_config_sha256="0" * 64,
    )
    with pytest.raises(CacheBundleIntegrityError, match="config digest"):
        verify_bundle(wrong_digest)

    wrong_experiment_bytes = config.replace(b"id: fixture", b"id: other")
    wrong_experiment = _write_manual_bundle(
        tmp_path / "wrong-experiment",
        declarations=[
            _declaration(
                "source-config/config.yaml",
                wrong_experiment_bytes,
                kind="config",
            )
        ],
        entries=[
            (
                _tar_entry("source-config/config.yaml", wrong_experiment_bytes),
                wrong_experiment_bytes,
            )
        ],
        source_config_sha256=hashlib.sha256(wrong_experiment_bytes).hexdigest(),
    )
    with pytest.raises(CacheBundleIntegrityError, match="experiment"):
        verify_bundle(wrong_experiment)


@pytest.mark.parametrize(
    ("path", "kind"),
    (
        ("cache/planning-cache-v1/aa/" + "a" * 64 + ".json", "config"),
        ("source-config/config.yaml", "immutable"),
        ("cache/planning-cache-v1/aa/" + "a" * 64 + ".json", "checkpoint"),
        ("outputs/fixture/rag2026-0/retrieval/audit.json", "similarity"),
    ),
)
def test_verify_rejects_wrong_kind_path_combinations(
    tmp_path: Path,
    path: str,
    kind: str,
) -> None:
    config = _manual_config_bytes()
    body = config if path == "source-config/config.yaml" else _canonical_json({})
    declarations = [_declaration(path, body, kind=kind)]
    entries = [(_tar_entry(path, body), body)]
    if path != "source-config/config.yaml":
        declarations.append(
            _declaration("source-config/config.yaml", config, kind="config")
        )
        entries.append((_tar_entry("source-config/config.yaml", config), config))
    paired = sorted(
        zip(declarations, entries, strict=True), key=lambda pair: pair[0]["path"]
    )
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[pair[0] for pair in paired],
        entries=[pair[1] for pair in paired],
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="kind|config|path"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "kind", "body"),
    (
        (
            "cache/planning-cache-v1/aa/" + "a" * 64 + ".json",
            "immutable",
            b'{"identity":{}}\n',
        ),
        (
            "cache/similarity-cache-v1/aa/" + "a" * 64 + ".json",
            "similarity",
            b'{"identity":{},"matrix_hex":[["nan"]]}\n',
        ),
    ),
)
def test_verify_rejects_malformed_cache_identity(
    tmp_path: Path,
    path: str,
    kind: str,
    body: bytes,
) -> None:
    config = _manual_config_bytes()
    declarations = [
        _declaration(path, body, kind=kind),
        _declaration("source-config/config.yaml", config, kind="config"),
    ]
    entries = [
        (_tar_entry(path, body), body),
        (_tar_entry("source-config/config.yaml", config), config),
    ]
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=declarations,
        entries=entries,
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="planning|similarity|identity"):
        verify_bundle(bundle)


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
    source_planning = _planning_entry(config.parent / "cache")
    assert _planning_entry(cache_root).read_bytes() == source_planning.read_bytes()
    assert (outputs_root / "shard-fixture/rag2026-0/retrieval/complete.json").is_file()
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
    source_planning = _planning_entry(config.parent / "cache")
    conflicting = cache_root / source_planning.relative_to(config.parent / "cache")
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
    assert not any(
        member.path.startswith("cache/") and member.path.endswith(".sqlite3")
        for member in verified.members
    )
    assert any(member.path.endswith("/records.sqlite3") for member in verified.members)
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


def test_completed_merge_reimports_safely_missing_portable_score_database(
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
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    database = imported.path
    imported.close()
    database.unlink()

    second = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )

    restored = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert restored.get(query_text="query", text="passage") == 1.25
    finally:
        restored.close()
    assert second == first


def test_completed_merge_fails_closed_on_corrupt_portable_score_database(
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
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    database = imported.path
    imported.close()
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(CacheBundleIntegrityError, match="score|database|corrupt"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_impossible_canonical_receipt_counts(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    forged = json.loads(receipt.completion_path.read_bytes())
    forged["installed_count"] = 999_999
    receipt.completion_path.write_bytes(_canonical_json(forged))
    prepare_path = receipt.completion_path.parent / "prepare.json"

    assert incomplete_merge_journals(cache_root) == (prepare_path,)
    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|count"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_its_flagged_current_journal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    forged = json.loads(receipt.completion_path.read_bytes())
    forged["prepare_sha256"] = "0" * 64
    receipt.completion_path.write_bytes(_canonical_json(forged))
    prepare_path = receipt.completion_path.parent / "prepare.json"

    assert incomplete_merge_journals(cache_root) == (prepare_path,)
    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|journal"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_nonregular_receipt_before_recovery_mutation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    outside = tmp_path / "forged-complete.json"
    outside.write_bytes(receipt.completion_path.read_bytes())
    receipt.completion_path.unlink()
    receipt.completion_path.symlink_to(outside)
    publication_events: list[str] = []

    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|regular"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=publication_events.append,
        )

    assert publication_events == []


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
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    SimilarityCache(cache_root).store(identity, ((1.0, 0.5), (0.5, 1.0)))

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


def test_completed_keep_existing_merge_rejects_changed_similarity_target(
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
    source_cache = SimilarityCache(config.parent / "cache")
    source_cache.store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    destination_cache = SimilarityCache(cache_root)
    destination = destination_cache.store(identity, ((1.0, 0.5), (0.5, 1.0)))
    outputs_root = (tmp_path / "destination-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )
    conflict_path = receipt.completion_path.parent / "conflicts.json"
    original_conflict_bytes = conflict_path.read_bytes()
    replacement = SimilarityCache(tmp_path / "replacement-cache").store(
        identity,
        ((1.0, 0.75), (0.75, 1.0)),
    )
    destination.write_bytes(replacement.read_bytes())

    with pytest.raises(
        CacheBundleIntegrityError, match="completion|conflict|destination"
    ):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert conflict_path.read_bytes() == original_conflict_bytes


def test_keep_existing_rejects_malformed_destination_similarity_before_mutation(
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
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    destination = SimilarityCache(cache_root).entry_path(identity)
    destination.parent.mkdir(parents=True)
    malformed = json.loads(
        SimilarityCache(config.parent / "cache").entry_path(identity).read_bytes()
    )
    malformed["matrix_hex"][0][0] = "nan"
    malformed["matrix_sha256"] = hashlib.sha256(
        json.dumps(
            malformed["matrix_hex"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    destination.write_bytes(_canonical_json(malformed))
    outputs_root = (tmp_path / "destination-outputs").resolve()

    with pytest.raises(CacheBundleIntegrityError, match="similarity"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_keep_existing_rejects_destination_similarity_with_wrong_identity(
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
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    wrong_identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("second", "first"),
    )
    wrong = SimilarityCache(tmp_path / "wrong-cache").store(
        wrong_identity, ((1.0, 0.5), (0.5, 1.0))
    )
    destination = SimilarityCache(cache_root).entry_path(identity)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(wrong.read_bytes())
    outputs_root = (tmp_path / "destination-outputs").resolve()

    with pytest.raises(CacheBundleIntegrityError, match="similarity"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_keep_existing_never_overrides_immutable_conflicts(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    source_planning = _planning_entry(config.parent / "cache")
    conflicting = cache_root / source_planning.relative_to(config.parent / "cache")
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"immutable destination")

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=(tmp_path / "merged-outputs").resolve(),
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

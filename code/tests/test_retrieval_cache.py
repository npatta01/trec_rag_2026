from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trec_rag.document_store import DocumentStore
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    RetrievalCacheConflictError,
    RetrievalCacheIntegrityError,
    RetrievalCacheMiss,
    TransportIdentity,
)


def transport(
    query_text: str = "what is the answer?",
    *,
    corpus_epoch: str = "climbmix-2026-08-01",
    hits: int = 2,
) -> TransportIdentity:
    return TransportIdentity.from_query(
        query_text=query_text,
        index_id="climbmix-400b",
        endpoint_identity="https://pyserini.example/v1/climbmix-400b/search",
        corpus_epoch=corpus_epoch,
        hits=hits,
    )


def response(*, first_text: str = " First\n  exact body. ", first_score: float = 12.5) -> bytes:
    return json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": "what is the answer?"},
            "candidates": [
                {
                    "rank": 1,
                    "docid": "doc-a",
                    "score": first_score,
                    "doc": first_text,
                },
                {
                    "rank": 2,
                    "docid": "doc-b",
                    "score": 8,
                    "doc": "Second\tbody",
                },
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def make_cache(tmp_path: Path, *, normalizer: OrganizerTextNormalizer | None = None) -> RetrievalCache:
    return RetrievalCache(
        tmp_path / "retrieval" / "pyserini_remote",
        DocumentStore(tmp_path / "documents"),
        normalizer or OrganizerTextNormalizer(),
    )


def _hard_crash_with_sealed_attempt_and_transport_staging(
    root: str,
    crash_phase: str,
) -> None:
    cache = make_cache(Path(root))

    def crash_after_staging_receipts(phase: str) -> None:
        if phase == crash_phase:
            os._exit(73)

    cache.publication_hook = crash_after_staging_receipts
    cache.commit(
        transport(),
        DerivationIdentity.from_normalizer(cache.normalizer),
        "what is the answer?",
        response(),
    )


def test_transport_and_derivation_keys_split_remote_and_local_identity() -> None:
    first = transport()
    changed_query = transport("a different query")
    changed_epoch = transport(corpus_epoch="climbmix-2026-08-02")
    changed_hits = transport(hits=10)
    derivation_v1 = DerivationIdentity.from_normalizer(OrganizerTextNormalizer(version="v1"))
    derivation_v2 = DerivationIdentity.from_normalizer(OrganizerTextNormalizer(version="v2"))

    assert len(first.request_key) == 64
    assert first.request_key != changed_query.request_key
    assert first.request_key != changed_epoch.request_key
    assert first.request_key != changed_hits.request_key
    assert derivation_v1.derivation_key != derivation_v2.derivation_key
    assert first.request_key == transport().request_key
    assert first.canonical_dict()["query_sha256"] == hashlib.sha256(
        b"what is the answer?"
    ).hexdigest()


def test_commit_retains_exact_raw_and_ordered_no_text_refs_in_cas(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    raw = response()

    committed = cache.commit(transport(), DerivationIdentity.from_normalizer(cache.normalizer), "what is the answer?", raw)

    assert [hit.docid for hit in committed.hits] == ["doc-a", "doc-b"]
    assert committed.hits[0].content_sha256 == hashlib.sha256(
        " First\n  exact body. ".encode("utf-8")
    ).hexdigest()
    entry = cache.v2_root / transport().request_key[:2] / transport().request_key
    assert gzip.decompress((entry / "raw.body.gz").read_bytes()) == raw
    hit_bytes = (entry / "derived" / committed.derivation_identity.derivation_key / "hits.json").read_bytes()
    assert b"exact body" not in hit_bytes
    assert b'"text"' not in hit_bytes
    assert cache.document_store.read_text(committed.hits[0].content_sha256) == " First\n  exact body. "

    transport_manifest = json.loads((entry / "transport-manifest.json").read_text())
    derivation_manifest = json.loads(
        (entry / "derived" / committed.derivation_identity.derivation_key / "derivation-manifest.json").read_text()
    )
    assert set(transport_manifest) == {
        "schema_version",
        "request_key",
        "transport_identity",
        "query_sha256",
        "raw_sha256",
        "raw_length",
        "gzip_sha256",
        "gzip_length",
        "raw_filename",
    }
    assert transport_manifest["request_key"] == transport().request_key
    assert transport_manifest["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert transport_manifest["raw_length"] == len(raw)
    assert set(derivation_manifest) == {
        "schema_version",
        "derivation_key",
        "derivation_identity",
        "parent_request_key",
        "parent_raw_sha256",
        "hits_filename",
        "hits_sha256",
        "hits_length",
        "hit_count",
        "ordered_semantic_hit_digest",
        "document_closure",
    }


def test_live_commit_accepts_sanitized_full_production_shape(tmp_path: Path) -> None:
    raw = response(first_text="  Café\n\t\u202fexact  ")
    payload = json.loads(raw)

    assert set(payload) == {"api", "index", "query", "candidates"}
    assert payload["query"] == {"text": "what is the answer?"}
    assert all(
        set(candidate) == {"doc", "docid", "rank", "score"}
        and isinstance(candidate["doc"], str)
        for candidate in payload["candidates"]
    )

    cache = make_cache(tmp_path)
    committed = cache.commit(
        transport(),
        DerivationIdentity.from_normalizer(cache.normalizer),
        "what is the answer?",
        raw,
    )

    assert cache.document_store.read_text(committed.hits[0].content_sha256) == (
        "  Café\n\t\u202fexact  "
    )


def test_live_commit_rejects_response_query_mismatch(tmp_path: Path) -> None:
    payload = json.loads(response())
    payload["query"] = {"text": "a different query"}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    cache = make_cache(tmp_path)

    with pytest.raises(RetrievalCacheIntegrityError, match="query"):
        cache.commit(
            transport(),
            DerivationIdentity.from_normalizer(cache.normalizer),
            "what is the answer?",
            raw,
        )


def test_lookup_rederives_normalizer_bump_without_network(tmp_path: Path) -> None:
    old_cache = make_cache(tmp_path, normalizer=OrganizerTextNormalizer(version="v1"))
    raw = response()
    old_cache.commit(transport(), DerivationIdentity.from_normalizer(old_cache.normalizer), "what is the answer?", raw)
    entry = old_cache.v2_root / transport().request_key[:2] / transport().request_key
    transport_manifest_before = (entry / "transport-manifest.json").read_bytes()
    raw_gzip_before = (entry / "raw.body.gz").read_bytes()

    new_cache = make_cache(tmp_path, normalizer=OrganizerTextNormalizer(version="v2"))
    hit = new_cache.lookup(
        transport(),
        DerivationIdentity.from_normalizer(new_cache.normalizer),
        "what is the answer?",
    )

    assert hit is not None
    assert hit.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert [row.docid for row in hit.hits] == ["doc-a", "doc-b"]
    derived = list(
        (new_cache.v2_root / transport().request_key[:2] / transport().request_key / "derived").iterdir()
    )
    assert len(derived) == 2
    assert (entry / "transport-manifest.json").read_bytes() == transport_manifest_before
    assert (entry / "raw.body.gz").read_bytes() == raw_gzip_before


def _derivation_with_change(
    cache: RetrievalCache,
    field: str,
) -> DerivationIdentity:
    values = DerivationIdentity.from_normalizer(cache.normalizer).canonical_dict()
    if field == "field_path":
        values[field] = ["doc", "alternate"]
    else:
        values[field] = f"changed-{values[field]}"
    return DerivationIdentity(**values)


@pytest.mark.parametrize(
    "field",
    (
        "parser_version",
        "extractor_version",
        "field_path",
        "scoring_normalizer_version",
    ),
)
def test_commit_rejects_unbound_derivation_identity_before_publication(
    tmp_path: Path,
    field: str,
) -> None:
    cache = make_cache(tmp_path)

    with pytest.raises(RetrievalCacheIntegrityError, match="derivation identity"):
        cache.commit(
            transport(),
            _derivation_with_change(cache, field),
            "what is the answer?",
            response(),
        )

    assert not cache.v2_root.exists()


def test_lookup_rejects_unbound_derivation_identity_before_republication(
    tmp_path: Path,
) -> None:
    cache = make_cache(tmp_path)
    cache.commit(
        transport(),
        DerivationIdentity.from_normalizer(cache.normalizer),
        "what is the answer?",
        response(),
    )

    with pytest.raises(RetrievalCacheIntegrityError, match="derivation identity"):
        cache.lookup(
            transport(),
            _derivation_with_change(cache, "field_path"),
            "what is the answer?",
        )


def test_legacy_promotion_rejects_unbound_derivation_identity_before_publication(
    tmp_path: Path,
) -> None:
    cache = make_cache(tmp_path)
    legacy_key = "03ad3a6e3c534274"
    raw = response()
    legacy = tmp_path / f"topic-31__original__climbmix_bm25__{legacy_key}.json"
    legacy.write_bytes(raw)
    sidecar = legacy.with_suffix(".meta.json")
    sidecar.write_bytes(
        json.dumps(
            {
                "cache_key": legacy_key,
                "hits": 2,
                "index": "climbmix-400b",
                "index_url": transport().endpoint_identity,
                "query": "what is the answer?",
                "rate_policy": {
                    "burst": 1,
                    "min_interval_seconds": 1.0,
                    "per_host": True,
                },
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "retriever_name": "climbmix_bm25",
                "retriever_type": "pyserini_remote",
                "topic_id": "topic-31",
                "variant_name": "original",
            },
            sort_keys=True,
        ).encode()
    )

    with pytest.raises(RetrievalCacheIntegrityError, match="derivation identity"):
        cache.promote_legacy(
            legacy,
            transport(),
            _derivation_with_change(cache, "parser_version"),
            "what is the answer?",
            sidecar_path=sidecar,
            operator_attestation={
                "operator": "test",
                "corpus_epoch": transport().corpus_epoch,
            },
        )

    assert not (cache.v2_root / "legacy-promotions").exists()


def test_normalizer_rejects_object_field_contract_without_fallback(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": "what is the answer?"},
            "candidates": [
                {
                    "rank": 1,
                    "docid": "doc-a",
                    "score": 1.0,
                    "doc": {"contents": "legacy body", "text": "new exact body"},
                }
            ]
        },
        separators=(",", ":"),
    ).encode()
    with pytest.raises(RetrievalCacheIntegrityError, match="doc must be exact plain text"):
        make_cache(tmp_path).commit(
            transport(hits=1),
            DerivationIdentity.from_normalizer(OrganizerTextNormalizer()),
            "what is the answer?",
            raw,
        )


def test_offline_miss_fails_before_any_external_boundary(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)

    with pytest.raises(RetrievalCacheMiss, match="offline cache miss"):
        cache.lookup(
            transport(),
            DerivationIdentity.from_normalizer(cache.normalizer),
            "what is the answer?",
            offline=True,
        )


@pytest.mark.parametrize(
    "broken",
    [
        {"rank": 1, "docid": "doc-a", "doc": {"text": "body"}},
        {"rank": 1, "score": 1.0, "doc": {"text": "body"}},
        {"rank": 1, "docid": "doc-a", "score": 1.0, "doc": {}},
        {"rank": 1, "docid": "doc-a", "score": float("nan"), "doc": {"text": "body"}},
    ],
)
def test_commit_rejects_missing_or_nonfinite_organizer_fields(
    tmp_path: Path, broken: dict[str, object]
) -> None:
    cache = make_cache(tmp_path)
    raw = json.dumps({"candidates": [broken]}, allow_nan=True).encode()

    with pytest.raises(RetrievalCacheIntegrityError):
        cache.commit(transport(hits=1), DerivationIdentity.from_normalizer(cache.normalizer), "what is the answer?", raw)


def test_same_topic_docid_with_different_bodies_fails_closed(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": "what is the answer?"},
            "candidates": [
                {"rank": 1, "docid": "same", "score": 1.0, "doc": "one"},
                {"rank": 2, "docid": "same", "score": 0.5, "doc": "two"},
            ]
        }
    ).encode()

    with pytest.raises(RetrievalCacheIntegrityError, match="docid.*bodi"):
        cache.commit(transport(hits=2), DerivationIdentity.from_normalizer(cache.normalizer), "what is the answer?", raw)


def test_identical_concurrent_commits_converge_to_one_entry(tmp_path: Path) -> None:
    raw = response()
    cache = make_cache(tmp_path)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)

    def commit() -> object:
        return cache.commit(transport(), derivation, "what is the answer?", raw)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result(timeout=5) for future in [executor.submit(commit), executor.submit(commit)]]

    assert results[0] == results[1]
    entry = cache.v2_root / transport().request_key[:2] / transport().request_key
    assert len(list(entry.rglob("transport-manifest.json"))) == 1
    assert len(list(entry.rglob("derivation-manifest.json"))) == 1


def test_byte_different_same_epoch_response_is_a_visible_conflict(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)
    cache.commit(transport(), derivation, "what is the answer?", response())

    with pytest.raises(RetrievalCacheConflictError, match="raw authority"):
        cache.commit(transport(), derivation, "what is the answer?", response(first_text="different"))
    assert list((cache.v2_root / "conflicts" / transport().request_key).iterdir())
    assert [row.docid for row in cache.lookup(transport(), derivation, "what is the answer?").hits] == [
        "doc-a",
        "doc-b",
    ]


def test_complete_attempt_recovers_after_publication_crash(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)
    cache.publication_hook = lambda phase: (_ for _ in ()).throw(
        RuntimeError("crash")
    ) if phase == "transport_raw" else None

    with pytest.raises(RuntimeError, match="crash"):
        cache.commit(transport(), derivation, "what is the answer?", response())
    assert list((cache.v2_root / "attempts" / transport().request_key).iterdir())
    assert not (cache.v2_root / transport().request_key[:2] / transport().request_key / "transport-manifest.json").exists()

    cache.publication_hook = None
    recovered = cache.lookup(transport(), derivation, "what is the answer?")
    assert recovered is not None
    assert recovered.raw_sha256 == hashlib.sha256(response()).hexdigest()


@pytest.mark.parametrize("crash_phase", ["transport_raw", "transport_raw_linked"])
def test_process_death_staging_recovers_only_from_sealed_attempt(
    tmp_path: Path,
    crash_phase: str,
) -> None:
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_hard_crash_with_sealed_attempt_and_transport_staging,
        args=(str(tmp_path), crash_phase),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 73
    cache = make_cache(tmp_path)
    identity = transport()
    entry = cache._entry_path(identity.request_key)
    stale = list(entry.parent.glob(f".{entry.name}.*.tmp"))
    assert len(stale) == 1

    recovered = cache.lookup(
        identity,
        DerivationIdentity.from_normalizer(cache.normalizer),
        "what is the answer?",
    )

    assert recovered is not None
    assert recovered.raw_response == response()
    assert not stale[0].exists()
    assert (entry / "transport-manifest.json").is_file()


def test_uncommitted_attempt_is_ignored(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    request_key = transport().request_key
    attempt = cache.v2_root / "attempts" / request_key / "uncommitted"
    attempt.mkdir(parents=True)
    (attempt / "request.json").write_text("{}")
    (attempt / "response.bin").write_bytes(response())

    with pytest.raises(RetrievalCacheMiss):
        cache.lookup(
            transport(),
            DerivationIdentity.from_normalizer(cache.normalizer),
            "what is the answer?",
            offline=True,
        )


def test_different_complete_attempt_success_hashes_conflict(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    identity = transport()
    query_text = "what is the answer?"
    cache.v2_root.mkdir(parents=True, exist_ok=True)
    cache._write_attempt_locked(identity, query_text, response())
    cache._write_attempt_locked(identity, query_text, response(first_text="other"))

    with pytest.raises(RetrievalCacheConflictError, match="complete attempts"):
        cache.lookup(
            identity,
            DerivationIdentity.from_normalizer(cache.normalizer),
            query_text,
        )


def test_partial_or_corrupt_complete_entries_fail_closed(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)
    cache.commit(transport(), derivation, "what is the answer?", response())
    entry = cache.v2_root / transport().request_key[:2] / transport().request_key

    (entry / "raw.body.gz").write_bytes(b"not gzip")
    with pytest.raises(RetrievalCacheIntegrityError):
        cache.lookup(transport(), derivation, "what is the answer?")

    (entry / "raw.body.gz").write_bytes(gzip.compress(response(), mtime=0))
    (entry / "transport-manifest.json").unlink()
    recovered = cache.lookup(transport(), derivation, "what is the answer?")
    assert recovered is not None
    assert (entry / "transport-manifest.json").exists()


def test_legacy_promotion_is_explicit_and_never_deletes_source(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    legacy_key = "03ad3a6e3c534274"
    legacy = (
        tmp_path
        / f"topic-31__original__climbmix_bm25__{legacy_key}.json"
    )
    raw = response()
    legacy.write_bytes(raw)
    sidecar = legacy.with_suffix(".meta.json")
    sidecar.write_bytes(
        json.dumps(
            {
                "cache_key": legacy_key,
                "hits": 2,
                "index": "climbmix-400b",
                "index_url": transport().endpoint_identity,
                "query": "what is the answer?",
                "rate_policy": {
                    "burst": 1,
                    "min_interval_seconds": 1.0,
                    "per_host": True,
                },
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "retriever_name": "climbmix_bm25",
                "retriever_type": "pyserini_remote",
                "topic_id": "topic-31",
                "variant_name": "original",
            },
            sort_keys=True,
        ).encode()
    )
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)

    receipt = cache.promote_legacy(
        legacy,
        transport(),
        derivation,
        "what is the answer?",
        sidecar_path=sidecar,
        operator_attestation={"operator": "test", "corpus_epoch": transport().corpus_epoch},
    )

    assert receipt.request_key == transport().request_key
    assert receipt.request_key != legacy_key
    assert legacy.read_bytes() == raw
    assert sidecar.exists()
    assert receipt.receipt_path.exists()


@pytest.mark.parametrize("tamper", ["filename", "cache_key", "rate_policy"])
def test_legacy_promotion_rejects_tampered_production_sidecar(
    tmp_path: Path,
    tamper: str,
) -> None:
    expected_key = "03ad3a6e3c534274"
    wrong_key = "1111111111111111"
    source_key = wrong_key if tamper in {"filename", "cache_key"} else expected_key
    sidecar_key = wrong_key if tamper == "cache_key" else expected_key
    raw = response()
    source = (
        tmp_path
        / f"topic-31__original__climbmix_bm25__{source_key}.json"
    )
    source.write_bytes(raw)
    rate_policy: dict[str, object] = {
        "burst": 1,
        "min_interval_seconds": 1.0,
        "per_host": True,
    }
    if tamper == "rate_policy":
        rate_policy["scope"] = "host"
    sidecar = source.with_suffix(".meta.json")
    sidecar.write_bytes(
        json.dumps(
            {
                "cache_key": sidecar_key,
                "hits": 2,
                "index": "climbmix-400b",
                "index_url": transport().endpoint_identity,
                "query": "what is the answer?",
                "rate_policy": rate_policy,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "retriever_name": "climbmix_bm25",
                "retriever_type": "pyserini_remote",
                "topic_id": "topic-31",
                "variant_name": "original",
            },
            sort_keys=True,
        ).encode()
    )
    cache = make_cache(tmp_path)

    with pytest.raises(RetrievalCacheIntegrityError, match="legacy"):
        cache.promote_legacy(
            source,
            transport(),
            DerivationIdentity.from_normalizer(cache.normalizer),
            "what is the answer?",
            sidecar_path=sidecar,
            operator_attestation={
                "operator": "test",
                "corpus_epoch": transport().corpus_epoch,
            },
        )


@pytest.mark.parametrize("epoch", ["", " ", "unspecified", "UNSPECIFIED"])
def test_transport_identity_rejects_missing_or_synthetic_corpus_epoch(epoch: str) -> None:
    with pytest.raises(ValueError, match="corpus_epoch"):
        TransportIdentity.from_query(
            query_text="q",
            index_id="climbmix-400b",
            endpoint_identity="https://pyserini.example/search",
            corpus_epoch=epoch,
            hits=1,
        )


@pytest.mark.parametrize("index_id", ["", "unknown-index", "unspecified"])
def test_transport_identity_rejects_fabricated_index_id(index_id: str) -> None:
    with pytest.raises(ValueError, match="index_id"):
        TransportIdentity.from_query(
            query_text="q",
            index_id=index_id,
            endpoint_identity="https://pyserini.example/search",
            corpus_epoch="epoch-1",
            hits=1,
        )


def test_exact_doc_string_normalizer_preserves_unicode_and_whitespace() -> None:
    body = "  Café\n\t\u202fexact  "
    parsed = OrganizerTextNormalizer().parse(
        json.dumps(
            {
                "api": "v1",
                "index": "climbmix-400b",
                "query": {"text": "what is the answer?"},
                "candidates": [
                    {"doc": body, "docid": "a", "rank": 1, "score": 1.0}
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    assert parsed[0].text == body


@pytest.mark.parametrize(
    "payload",
    [
        {"candidates": []},
        {"api": "v1", "index": "climbmix-400b", "extra": True, "candidates": []},
    ],
)
def test_exact_doc_string_normalizer_requires_actual_top_level_contract(
    payload: dict[str, object],
) -> None:
    with pytest.raises(RetrievalCacheIntegrityError):
        OrganizerTextNormalizer().parse(json.dumps(payload).encode())


def test_cached_commit_requires_exact_authority_bytes(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    identity = transport()
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)

    with pytest.raises(TypeError, match="exact response bytes"):
        cache.commit(identity, derivation, "what is the answer?", {"candidates": []})

    class ResponseObject:
        body = response()

    with pytest.raises(TypeError, match="exact response bytes"):
        cache.commit(identity, derivation, "what is the answer?", ResponseObject())


@pytest.mark.parametrize(
    "phase",
    [
        "transport_raw_linked",
        "transport_manifest_linked",
        "derived_hits_linked",
        "derived_manifest_linked",
    ],
)
def test_partial_publication_is_finishable_from_sealed_attempt(
    tmp_path: Path, phase: str
) -> None:
    cache = make_cache(tmp_path)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)
    cache.publication_hook = lambda current: (
        (_ for _ in ()).throw(RuntimeError("publication crash"))
        if current == phase
        else None
    )

    with pytest.raises(RuntimeError, match="publication crash"):
        cache.commit(transport(), derivation, "what is the answer?", response())

    cache.publication_hook = None
    recovered = cache.lookup(
        transport(), derivation, "what is the answer?", offline=True
    )
    assert recovered.raw_response == response()


def test_cache_artifacts_use_private_modes(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    previous_umask = os.umask(0)
    try:
        cache.commit(
            transport(),
            DerivationIdentity.from_normalizer(cache.normalizer),
            "what is the answer?",
            response(),
        )
        with pytest.raises(RetrievalCacheConflictError):
            cache.commit(
                transport(),
                DerivationIdentity.from_normalizer(cache.normalizer),
                "what is the answer?",
                response(first_text="conflicting body"),
            )
    finally:
        os.umask(previous_umask)

    for path in cache.v2_root.rglob("*"):
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            assert mode == 0o700, path
        elif path.name.endswith(".lock"):
            continue
        else:
            assert mode == 0o600, path

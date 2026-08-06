from __future__ import annotations

from hashlib import sha256
import json
import math

import pytest


def _model_identity() -> dict[str, object]:
    return {
        "backend": "sentence-transformers",
        "embedding_representation": "l2_normalized_float",
        "local_files_only": True,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "revision-1",
        "score_kind": "normalized_vector_cosine",
    }


def _identity(texts=("first", "second")):
    from trec_rag.similarity_cache import build_similarity_cache_identity

    return build_similarity_cache_identity(model_identity=_model_identity(), texts=texts)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def test_similarity_cache_round_trip_uses_finite_hex_without_raw_text(tmp_path) -> None:
    """Catches decimal/NaN serialization or portable entries retaining candidate text."""
    from trec_rag.similarity_cache import SimilarityCache

    cache = SimilarityCache(tmp_path / "cache")
    identity = _identity()
    matrix = ((1.0, -0.0), (0.125, 1.0))

    cache.store(identity, matrix)

    assert cache.load(identity) == matrix
    source = cache.entry_path(identity).read_text()
    entry = json.loads(source)
    assert entry["matrix_hex"] == [
        ["0x1.0000000000000p+0", "-0x0.0p+0"],
        ["0x1.0000000000000p-3", "0x1.0000000000000p+0"],
    ]
    assert "first" not in source
    assert "second" not in source


def test_similarity_cache_identity_binds_model_revision_and_ordered_text_hashes() -> None:
    """Catches matrix reuse across model revisions, text changes, or reordered rows."""
    identity = _identity()
    revised_model = dict(_model_identity(), model_revision="revision-2")
    from trec_rag.similarity_cache import build_similarity_cache_identity

    revised = build_similarity_cache_identity(
        model_identity=revised_model,
        texts=("first", "second"),
    )
    reordered = _identity(("second", "first"))
    changed = _identity(("first", "changed"))

    assert identity.text_sha256s == (
        sha256(b"first").hexdigest(),
        sha256(b"second").hexdigest(),
    )
    assert len({identity.cache_key, revised.cache_key, reordered.cache_key, changed.cache_key}) == 4


def test_similarity_cache_missing_read_is_pure(tmp_path) -> None:
    """Catches an offline matrix miss creating a directory, lock, or placeholder."""
    from trec_rag.similarity_cache import SimilarityCache, SimilarityCacheMiss

    root = tmp_path / "absent"
    with pytest.raises(SimilarityCacheMiss, match="similarity cache miss"):
        SimilarityCache(root).load(_identity())

    assert not root.exists()


def test_similarity_cache_rejects_nonfinite_malformed_and_conflicting_entries(
    tmp_path,
) -> None:
    """Catches accepting non-finite or replacing malformed immutable matrices."""
    from trec_rag.similarity_cache import SimilarityCache, SimilarityCacheIntegrityError

    cache = SimilarityCache(tmp_path / "cache")
    identity = _identity()
    with pytest.raises(ValueError, match="finite"):
        cache.store(identity, ((1.0, math.nan), (0.0, 1.0)))
    assert not (tmp_path / "cache").exists()

    cache.store(identity, ((1.0, 0.0), (0.0, 1.0)))
    path = cache.entry_path(identity)
    entry = json.loads(path.read_bytes())
    entry["matrix_hex"][0][0] = "nan"
    entry["matrix_sha256"] = sha256(_canonical(entry["matrix_hex"])).hexdigest()
    path.write_bytes(_canonical(entry) + b"\n")

    with pytest.raises(SimilarityCacheIntegrityError, match="similarity cache"):
        cache.load(identity)
    with pytest.raises(SimilarityCacheIntegrityError, match="similarity cache"):
        cache.store(identity, ((1.0, 0.0), (0.0, 1.0)))

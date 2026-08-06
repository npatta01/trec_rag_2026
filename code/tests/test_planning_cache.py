from __future__ import annotations

from dataclasses import replace
import json

import pytest


def _identity():
    from trec_rag.planning_cache import build_planning_cache_identity

    return build_planning_cache_identity(
        request_body=b'{"model":"pinned-model","messages":[]}',
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="pinned-model",
        prompt_version="planning-prompt-v1",
        schema_version="planning-schema-v1",
        topic_id="topic-1",
        narrative="Exact organizer narrative.",
    )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "subnarrative_queries_v1",
        "topic_id": "topic-1",
        "subnarratives": [
            {"subnarrative": "One facet", "bm25_queries": ["one query"]}
        ],
    }


def test_planning_cache_round_trip_persists_only_safe_validated_payload(tmp_path) -> None:
    """Catches retaining raw provider responses or narrative text in a portable entry."""
    from trec_rag.planning_cache import PlanningCache

    cache = PlanningCache(tmp_path / "cache")
    identity = _identity()

    cache.store(identity, _payload())

    assert cache.load(identity) == _payload()
    entry = json.loads(cache.entry_path(identity).read_bytes())
    assert entry["identity"] == identity.as_dict()
    assert entry["payload"] == _payload()
    assert "Exact organizer narrative." not in cache.entry_path(identity).read_text()
    assert "response" not in entry


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("endpoint", "https://other.example/chat/completions"),
        ("model", "other-model"),
        ("prompt_version", "planning-prompt-v2"),
        ("schema_version", "planning-schema-v2"),
        ("topic_id", "topic-2"),
        ("narrative_sha256", "0" * 64),
        ("request_body_sha256", "1" * 64),
    ),
)
def test_planning_cache_key_changes_for_every_exact_identity_field(
    field: str,
    value: str,
) -> None:
    """Catches cache reuse after any request, route, revision, or narrative change."""
    identity = _identity()

    changed = replace(identity, **{field: value})

    assert changed.cache_key != identity.cache_key


def test_planning_cache_missing_read_is_pure(tmp_path) -> None:
    """Catches an offline miss creating cache directories, locks, or placeholders."""
    from trec_rag.planning_cache import PlanningCache, PlanningCacheMiss

    root = tmp_path / "absent-cache"
    cache = PlanningCache(root)

    with pytest.raises(PlanningCacheMiss, match="planning cache miss"):
        cache.load(_identity())

    assert not root.exists()


def test_planning_cache_rejects_malformed_and_conflicting_entries(tmp_path) -> None:
    """Catches accepting or replacing corrupt immutable planning state."""
    from trec_rag.planning_cache import PlanningCache, PlanningCacheIntegrityError

    cache = PlanningCache(tmp_path / "cache")
    identity = _identity()
    cache.store(identity, _payload())
    path = cache.entry_path(identity)
    path.write_bytes(path.read_bytes().replace(b'"topic-1"', b'"topic-X"', 1))

    with pytest.raises(PlanningCacheIntegrityError, match="planning cache"):
        cache.load(identity)
    with pytest.raises(PlanningCacheIntegrityError, match="planning cache"):
        cache.store(identity, _payload())


def test_cached_facet_planning_reuses_validated_plan_without_backend_construction(
    tmp_path,
) -> None:
    """Catches a cache hit constructing a credentialed hosted backend."""
    from trec_rag.facet_extraction import extract_facets
    from trec_rag.topics import Topic

    topic = Topic("topic-1", "Private title", "Exact organizer narrative.")

    class Backend:
        def extract(self, received):
            assert received == topic
            return _payload()

    online_stats: dict[str, int] = {}
    first = extract_facets(
        topic,
        planning_cache_root=tmp_path / "cache",
        backend_factory=Backend,
        cache_stats=online_stats,
    )
    offline_stats: dict[str, int] = {}
    second = extract_facets(
        topic,
        planning_cache_root=tmp_path / "cache",
        cache_only=True,
        backend_factory=lambda: pytest.fail("cache hit must stay backend-free"),
        cache_stats=offline_stats,
    )

    assert first == second
    assert online_stats == {
        "cache_hits": 0,
        "cache_misses": 1,
        "backend_calls": 1,
        "provider_calls": 1,
    }
    assert offline_stats == {
        "cache_hits": 1,
        "cache_misses": 0,
        "backend_calls": 0,
        "provider_calls": 0,
    }
    persisted = next((tmp_path / "cache").rglob("*.json")).read_text()
    assert "Private title" not in persisted
    assert "provider" not in persisted


def test_cached_facet_planning_offline_miss_is_pure_and_lazy(tmp_path) -> None:
    """Catches a cache-only planning miss creating state or constructing a backend."""
    from trec_rag.facet_extraction import extract_facets
    from trec_rag.planning_cache import PlanningCacheMiss
    from trec_rag.topics import Topic

    root = tmp_path / "absent"
    stats: dict[str, int] = {}
    with pytest.raises(PlanningCacheMiss, match="planning cache miss"):
        extract_facets(
            Topic("topic-1", "Private title", "Exact organizer narrative."),
            planning_cache_root=root,
            cache_only=True,
            backend_factory=lambda: pytest.fail("offline miss must stay backend-free"),
            cache_stats=stats,
        )

    assert stats == {
        "cache_hits": 0,
        "cache_misses": 1,
        "backend_calls": 0,
        "provider_calls": 0,
    }
    assert not root.exists()


def test_cached_facet_planning_revalidates_semantics_before_backend_construction(
    tmp_path,
) -> None:
    """Catches trusting hash-consistent but semantically invalid cached plans."""
    from trec_rag.facet_extraction import extract_facets, planning_cache_identity
    from trec_rag.planning_cache import PlanningCache, PlanningCacheIntegrityError
    from trec_rag.topics import Topic

    topic = Topic("topic-1", "Private title", "Exact organizer narrative.")
    invalid = _payload()
    invalid["subnarratives"][0]["bm25_queries"] = ["unsafe AND query"]
    cache = PlanningCache(tmp_path / "cache")
    cache.store(planning_cache_identity(topic), invalid)

    with pytest.raises(PlanningCacheIntegrityError, match="validated planning payload"):
        extract_facets(
            topic,
            planning_cache_root=tmp_path / "cache",
            cache_only=True,
            backend_factory=lambda: pytest.fail("invalid cache must not construct backend"),
        )

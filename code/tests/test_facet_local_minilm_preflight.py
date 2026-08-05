from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import trec_rag.facet_local_minilm_preflight as module


class _Tokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[str]:
        assert add_special_tokens is False
        assert truncation is False
        return text.split()

    def decode(
        self,
        token_ids: list[str],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(token_ids)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is True
        return 3


class _BoundedLookupCache:
    path = Path("/tmp/test-score-cache.sqlite3")

    def __init__(self) -> None:
        self.lookup_calls: list[tuple[object, ...]] = []

    @property
    def scores(self) -> dict[str, float]:
        raise AssertionError("full score-table materialization is not allowed")

    def cache_key(self, *, query_text: str, text: str) -> str:
        return module._score_cache_key(query_text, text)

    def lookup_many(self, pairs: object) -> list[float | None]:
        materialized = tuple(pairs)  # type: ignore[arg-type]
        self.lookup_calls.append(materialized)
        return [None] * len(materialized)


def _candidate() -> dict[str, object]:
    query = "find evidence"
    text = "one two three"
    digest = lambda value: sha256(value.encode("utf-8")).hexdigest()
    return {
        "topic_id": "1",
        "family": "facet",
        "variant": "v1",
        "rank": 1,
        "document_id": "d1",
        "query": query,
        "query_sha256": digest(query),
        "text": text,
        "text_sha256": digest(text),
    }


def test_preflight_uses_bounded_cache_lookup_for_window_hits() -> None:
    cache = _BoundedLookupCache()

    plan = module.build_preflight([_candidate()], _Tokenizer(), cache)

    assert [row.cache_hit for row in plan.windows] == [False]
    assert len(cache.lookup_calls) == 1
    assert len(cache.lookup_calls[0]) == 1

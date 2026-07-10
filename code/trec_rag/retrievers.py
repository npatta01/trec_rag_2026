"""Retriever adapters and candidate normalization for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    normalize_candidates,
)


class Retriever(Protocol):
    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        ...


@dataclass
class RetrieverCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    bypasses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "bypasses": self.bypasses,
        }


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "part"


def cache_path(
    topic_id: str,
    variant_name: str,
    retriever_name: str,
    request_key: str | None = None,
) -> Path:
    base = f"{_safe_part(topic_id)}__{_safe_part(variant_name)}__{_safe_part(retriever_name)}"
    if request_key:
        return Path(f"{base}__{_safe_part(request_key)}.json")
    return Path(f"{base}.json")


def request_cache_key(
    config: RetrieverConfig,
    query: QueryVariant,
    *,
    index_url: str,
) -> str:
    fingerprint = {
        "retriever_name": config.name,
        "retriever_type": config.type,
        "index": config.index,
        "index_url": index_url,
        "hits": config.hits,
        "query_text": query.query_text,
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def normalize_retrieved_candidates(
    response: dict[str, object],
    *,
    query: QueryVariant,
    retriever_name: str,
) -> list[RetrievedCandidate]:
    candidates: list[RetrievedCandidate] = []
    for row in normalize_candidates(response):
        docid = row.get("docid")
        if not docid or not str(docid).strip():
            raise ValueError(f"{query.topic_id}/{query.variant_name}: candidate missing docid")
        candidates.append(
            RetrievedCandidate(
                topic_id=query.topic_id,
                variant_name=query.variant_name,
                retriever_name=retriever_name,
                query_text=query.query_text,
                docid=str(docid),
                rank=int(row.get("rank") or len(candidates) + 1),
                score=float(row.get("score") or 0.0),
                text=str(row.get("text") or ""),
            )
        )
    return candidates


class PyseriniRemoteRetriever:
    def __init__(
        self,
        config: RetrieverConfig,
        *,
        cache_dir: Path,
        client: RemotePyseriniClient | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.client = client or RemotePyseriniClient(_remote_config(config))
        self.cache_stats = RetrieverCacheStats()
        self.cache_hit_artifacts: list[dict[str, object]] = []

    def cache_summary(self) -> dict[str, object]:
        requests = self.cache_stats.hits + self.cache_stats.misses + self.cache_stats.bypasses
        return {
            "enabled": self.config.cache,
            "requests": requests,
            "hit_artifacts": self.cache_hit_artifacts,
            **self.cache_stats.as_dict(),
        }

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        request_key = request_cache_key(
            self.config,
            query,
            index_url=self.client.config.index_url,
        )
        cache_file = self.cache_dir / cache_path(
            query.topic_id,
            query.variant_name,
            self.config.name,
            request_key,
        )
        if self.config.cache and cache_file.exists():
            cache_bytes = cache_file.read_bytes()
            payload = json.loads(cache_bytes)
            response = payload.get("response")
            if not isinstance(response, dict):
                raise ValueError(f"cache file missing response object: {cache_file}")
            candidates = normalize_retrieved_candidates(
                response,
                query=query,
                retriever_name=self.config.name,
            )
            self.cache_stats.hits += 1
            self.cache_hit_artifacts.append(
                {
                    "file": cache_file.name,
                    "sha256": hashlib.sha256(cache_bytes).hexdigest(),
                    "candidate_count": len(candidates),
                }
            )
            return candidates
        if self.config.cache:
            self.cache_stats.misses += 1
        else:
            self.cache_stats.bypasses += 1
        response = self.client.search(query.query_text)
        if self.config.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    {
                        "cache_key": request_key,
                        "query": query.query_text,
                        "topic_id": query.topic_id,
                        "variant_name": query.variant_name,
                        "retriever_name": self.config.name,
                        "retriever_type": self.config.type,
                        "index": self.config.index,
                        "index_url": self.client.config.index_url,
                        "hits": self.config.hits,
                        "response": response,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.cache_stats.writes += 1
        return normalize_retrieved_candidates(response, query=query, retriever_name=self.config.name)


def _remote_config(config: RetrieverConfig) -> RemotePyseriniConfig:
    env = dict(os.environ)
    if config.index:
        index_url = env.get("INDEX_URL")
        if index_url:
            url_index = _index_from_search_url(index_url)
            if url_index != config.index:
                raise ValueError(
                    "INDEX_URL conflicts with retrievers[].index "
                    f"({url_index or 'unknown'} != {config.index})"
                )
    remote = RemotePyseriniConfig.from_env(env)
    return replace(remote, hits=config.hits)


def pyserini_factory(config: RetrieverConfig, cache_dir: Path) -> PyseriniRemoteRetriever:
    return PyseriniRemoteRetriever(config, cache_dir=cache_dir)


def _index_from_search_url(index_url: str) -> str | None:
    path_parts = [
        urllib.parse.unquote(part)
        for part in urllib.parse.urlparse(index_url.rstrip("?")).path.strip("/").split("/")
        if part
    ]
    if len(path_parts) >= 3 and path_parts[-3] == "v1" and path_parts[-1] == "search":
        return path_parts[-2]
    return None

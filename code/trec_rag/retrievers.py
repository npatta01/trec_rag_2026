"""Retriever adapters and candidate normalization for the RAG pipeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
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


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "part"


def cache_path(topic_id: str, variant_name: str, retriever_name: str) -> Path:
    return Path(
        f"{_safe_part(topic_id)}__{_safe_part(variant_name)}__{_safe_part(retriever_name)}.json"
    )


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

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        response = self.client.search(query.query_text)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / cache_path(query.topic_id, query.variant_name, self.config.name)).write_text(
            json.dumps(
                {
                    "query": query.query_text,
                    "topic_id": query.topic_id,
                    "variant_name": query.variant_name,
                    "retriever_name": self.config.name,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return normalize_retrieved_candidates(response, query=query, retriever_name=self.config.name)


def _remote_config(config: RetrieverConfig) -> RemotePyseriniConfig:
    env = dict(os.environ)
    if config.index and not env.get("INDEX_URL"):
        env.setdefault("PYSERINI_INDEX", config.index)
    remote = RemotePyseriniConfig.from_env(env)
    return replace(remote, hits=config.hits)


def pyserini_factory(config: RetrieverConfig, cache_dir: Path) -> PyseriniRemoteRetriever:
    return PyseriniRemoteRetriever(config, cache_dir=cache_dir)

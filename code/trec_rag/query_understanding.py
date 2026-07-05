"""Query-understanding strategies for RAG pipeline experiments."""

from __future__ import annotations

from typing import Iterable

from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


def build_query_variants(topic: Topic, *, variant_configs: Iterable[dict[str, str]]) -> list[QueryVariant]:
    variants: list[QueryVariant] = []
    for config in variant_configs:
        variant_type = config["type"]
        if variant_type != "original_topic":
            raise ValueError(f"unknown query variant type: {variant_type}")
        variants.append(
            QueryVariant(
                topic_id=topic.id,
                variant_name=config["name"],
                query_text=" ".join(topic.narrative.split()),
                source_type=variant_type,
            )
        )
    return variants

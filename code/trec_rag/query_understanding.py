"""Query-understanding strategies for RAG pipeline experiments."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


class FacetGenerator(Protocol):
    def generate(
        self,
        topic: Topic,
        *,
        variant_name: str,
        max_facets: int,
        cache: bool,
    ) -> list[dict[str, str]]:
        ...


def _config_value(config: object, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def build_query_variants(
    topic: Topic,
    *,
    variant_configs: Iterable[object],
    facet_generator: FacetGenerator | None = None,
) -> list[QueryVariant]:
    variants: list[QueryVariant] = []
    for config in variant_configs:
        variant_name = str(_config_value(config, "name")).strip()
        variant_type = str(_config_value(config, "type")).strip()
        if variant_type == "original_topic":
            variants.append(
                QueryVariant(
                    topic_id=topic.id,
                    variant_name=variant_name,
                    query_text=_normalized_text(topic.narrative),
                    source_type=variant_type,
                )
            )
        elif variant_type == "title":
            variants.append(
                QueryVariant(
                    topic_id=topic.id,
                    variant_name=variant_name,
                    query_text=_normalized_text(topic.title),
                    source_type=variant_type,
                )
            )
        elif variant_type == "llm_facets":
            if facet_generator is None:
                raise ValueError("llm_facets query variant requires a facet generator")
            facet_rows = facet_generator.generate(
                topic,
                variant_name=variant_name,
                max_facets=int(_config_value(config, "max_facets", 8)),
                cache=bool(_config_value(config, "cache", True)),
            )
            for row in facet_rows:
                query_text = _normalized_text(str(row.get("search_query") or ""))
                if query_text:
                    variants.append(
                        QueryVariant(
                            topic_id=topic.id,
                            variant_name=variant_name,
                            query_text=query_text,
                            source_type="llm_facet",
                        )
                    )
        else:
            raise ValueError(f"unknown query variant type: {variant_type}")
    return variants

"""LiteLLM-backed facet query generation."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from trec_rag.topics import Topic


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        ...


class UrllibJsonTransport:
    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            message = f"LiteLLM facet request failed: HTTP {exc.code} {exc.reason}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LiteLLM facet request failed: {exc}") from exc


class LiteLLMFacetGenerator:
    """Generate anchored BM25 facet queries through LiteLLM's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        transport: JsonTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LITELLM_BASE_URL") or "http://localhost:4000/v1").rstrip("/")
        self.model = model or os.getenv("LITELLM_MODEL") or "qwen-local"
        self.api_key = api_key if api_key is not None else os.getenv("LITELLM_API_KEY")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.transport = transport or UrllibJsonTransport()

    def generate(
        self,
        topic: Topic,
        *,
        variant_name: str,
        max_facets: int,
        cache: bool = True,
    ) -> list[dict[str, str]]:
        cache_file = self._cache_file(topic, variant_name, max_facets)
        if cache and cache_file and cache_file.exists():
            return self._parse_cached(cache_file, max_facets)

        response = self.transport.post_json(
            f"{self.base_url}/chat/completions",
            self._request_payload(topic, max_facets),
            self._headers(),
            self.timeout,
        )
        content = self._response_content(response)
        facets = self._parse_facets_json(content, max_facets)

        if cache and cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    {
                        "topic_id": topic.id,
                        "variant_name": variant_name,
                        "model": self.model,
                        "base_url": self.base_url,
                        "max_facets": max_facets,
                        "facets": facets,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return facets

    def _request_payload(self, topic: Topic, max_facets: int) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON only. Generate anchored BM25 search facets for a TREC RAG "
                        "narrative query. Each facet must be a short search query grounded in the title "
                        "or narrative, not a broad paraphrase."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title: {topic.title}\n"
                        f"Narrative: {topic.narrative}\n"
                        f"Return 5 to {max_facets} facets as "
                        '{"facets":[{"search_query":"..."}]}.'
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _cache_file(self, topic: Topic, variant_name: str, max_facets: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key_payload = json.dumps(
            {
                "topic_id": topic.id,
                "title": topic.title,
                "narrative": topic.narrative,
                "variant_name": variant_name,
                "model": self.model,
                "base_url": self.base_url,
                "max_facets": max_facets,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:16]
        safe_topic_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in topic.id)
        return self.cache_dir / f"{safe_topic_id}__{variant_name}__{digest}.json"

    def _parse_cached(self, cache_file: Path, max_facets: int) -> list[dict[str, str]]:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"facet cache is not a JSON object: {cache_file}")
        return self._normalize_facets(payload.get("facets"), max_facets)

    def _response_content(self, response: dict[str, object]) -> str:
        try:
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")
            message = choices[0]["message"]  # type: ignore[index]
            content = message["content"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise ValueError("LiteLLM response did not include choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LiteLLM response content is empty")
        return content

    def _parse_facets_json(self, content: str, max_facets: int) -> list[dict[str, str]]:
        text = self._strip_code_fence(content)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LiteLLM facet response must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("LiteLLM facet response must be a JSON object")
        return self._normalize_facets(payload.get("facets"), max_facets)

    def _normalize_facets(self, value: Any, max_facets: int) -> list[dict[str, str]]:
        if not isinstance(value, list):
            raise ValueError("LiteLLM facet response must include a facets list")
        facets: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            query = str(item.get("search_query") or item.get("query") or "").strip()
            query = " ".join(query.split())
            if query:
                facets.append({"search_query": query})
            if len(facets) >= max_facets:
                break
        if not facets:
            raise ValueError("LiteLLM facet response did not include any usable search_query values")
        return facets

    def _strip_code_fence(self, content: str) -> str:
        text = content.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

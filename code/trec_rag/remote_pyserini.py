"""Small client helpers for the hosted Pyserini search API."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_INDEX = "climbmix-400b"
DEFAULT_BASE_URL = "http://api.castorini.uwaterloo.ca"
DEFAULT_QUERIES = (
    "I'm seeking to understand the environmental and health impacts of e-waste and other "
    "waste, along with how proper recycling benefits sustainability and local economies. "
    "Could you explain the risks of improper waste management, recent recycling "
    "innovations, and practical steps individuals and businesses can take for responsible "
    "waste handling?"
)
DEFAULT_QUERY_SOURCE = "rag25-topics-dev.tsv topic 31"


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "AGENTS.md").exists():
            return candidate
    return current


def shared_checkout_root(repo_root: Path) -> Path | None:
    git_file = repo_root / ".git"
    if not git_file.is_file():
        return None
    git_pointer = git_file.read_text(encoding="utf-8").strip()
    if not git_pointer.startswith("gitdir:"):
        return None
    git_dir = Path(git_pointer.split(":", 1)[1].strip()).expanduser()
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    if git_dir.parent.name != "worktrees":
        return None
    return git_dir.parent.parent.parent


def load_dotenv(
    path: Path,
    *,
    override: bool = False,
    protected_keys: set[str] | None = None,
) -> None:
    if not path.exists():
        return
    protected_keys = protected_keys or set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if override and key not in protected_keys:
            os.environ[key] = value.strip().strip('"').strip("'")
        else:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def load_repo_env(repo_root: Path) -> None:
    protected_keys = set(os.environ)
    roots = []
    shared_root = shared_checkout_root(repo_root)
    if shared_root and shared_root != repo_root:
        roots.append(shared_root)
    roots.append(repo_root)
    for root in roots:
        load_dotenv(root / ".env", override=True, protected_keys=protected_keys)
        load_dotenv(root / ".env.local", override=True, protected_keys=protected_keys)


def _env_get(env: Mapping[str, str] | None, name: str) -> str | None:
    return (env or os.environ).get(name)


def env_int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    raw_value = _env_get(env, name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def env_queries(name: str, default: str, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw_value = _env_get(env, name) or default
    return tuple(query.strip() for query in raw_value.split(";") if query.strip())


@dataclass(frozen=True)
class RemotePyseriniConfig:
    index_url: str
    api_token: str | None
    hits: int
    queries: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RemotePyseriniConfig":
        index = (_env_get(env, "PYSERINI_INDEX") or DEFAULT_INDEX).strip() or DEFAULT_INDEX
        base_url = (_env_get(env, "PYSERINI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        default_index_url = f"{base_url}/v1/{urllib.parse.quote(index)}/search"
        return cls(
            index_url=(_env_get(env, "INDEX_URL") or default_index_url).rstrip("?"),
            api_token=_env_get(env, "PYSERINI_API_TOKEN"),
            hits=env_int("EXTERNAL_PYSERINI_HITS", 5, env),
            queries=env_queries("SAMPLE_QUERIES", DEFAULT_QUERIES, env),
        )


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        try:
            return extract_text(json.loads(payload))
        except json.JSONDecodeError:
            return " ".join(payload.split())
    if isinstance(payload, dict):
        for key in ("contents", "text", "body", "passage", "abstract"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        return " ".join(text for text in (extract_text(value) for value in payload.values()) if text)
    if isinstance(payload, list):
        return " ".join(text for text in (extract_text(value) for value in payload) if text)
    return ""


def normalize_candidates(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("candidates") or response.get("hits") or response.get("results") or []
    normalized = []
    for rank, row in enumerate(rows, start=1):
        doc_payload = row.get("doc") or row.get("contents") or row
        text = extract_text(doc_payload)
        normalized.append(
            {
                "rank": int(row.get("rank") or rank),
                "docid": row.get("docid") or row.get("id") or row.get("_id"),
                "score": row.get("score"),
                "text": text,
                "text_length": len(text),
            }
        )
    return normalized


class RemotePyseriniClient:
    def __init__(
        self,
        config: RemotePyseriniConfig,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: int = 30,
    ) -> None:
        self.config = config
        self.opener = opener
        self.timeout = timeout

    def search(self, query: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({"query": query, "hits": str(self.config.hits)})
        request = urllib.request.Request(
            f"{self.config.index_url}?{params}",
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token
                    else {}
                ),
            },
        )
        with self.opener(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def candidates(self, query: str) -> list[dict[str, Any]]:
        return normalize_candidates(self.search(query))

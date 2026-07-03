"""Typed environment helpers."""

from __future__ import annotations

import os
from typing import Mapping


def env_get(env: Mapping[str, str] | None, name: str) -> str | None:
    return (env or os.environ).get(name)


def env_int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    raw_value = env_get(env, name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def env_queries(name: str, default: str, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw_value = env_get(env, name) or default
    return tuple(query.strip() for query in raw_value.split(";") if query.strip())

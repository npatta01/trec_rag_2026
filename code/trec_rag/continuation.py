"""Inspect, resume, or discard a pending hosted-retrieval continuation.

New state is kept under ``topic-state/<topic-id>``.  The old root-level names
remain readable for the explicit recovery command so old tickets are never
silently treated as active v2 cache entries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

from filelock import FileLock

from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant
from trec_rag.repo_env import find_repo_root, load_repo_env, repo_cache_root
from trec_rag.retrievers import (
    KNOWN_TOPIC_LEASE_STATES,
    CONTINUATION_COMPLETION_SCHEMA_VERSION,
    PyseriniRemoteRetriever,
    TOPIC_COMPLETION_NAME,
    TOPIC_IN_PROGRESS_NAME,
    TOPIC_STATE_DIRNAME,
    TOPIC_TICKET_NAME,
    request_cache_key,
)

TICKET_NAME = "continuation-ticket.json"
IN_PROGRESS_NAME = "continuation-in-progress.json"


def cache_directory(root: Path | None = None) -> Path:
    resolved = Path(root) if root is not None else find_repo_root()
    return repo_cache_root(resolved) / "retrieval" / "pyserini_remote"


def topic_state_directory(cache_dir: Path, topic_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(topic_id)).strip("._") or "topic"
    return Path(cache_dir) / TOPIC_STATE_DIRNAME / safe


def _state_root(cache_dir: Path, topic_id: str | None) -> tuple[Path, bool, str | None]:
    cache_dir = Path(cache_dir)
    if topic_id is not None:
        return topic_state_directory(cache_dir, topic_id), True, str(topic_id)
    if (cache_dir / TICKET_NAME).exists() or (cache_dir / IN_PROGRESS_NAME).exists():
        return cache_dir, False, None
    candidates = sorted(
        path
        for path in (cache_dir / TOPIC_STATE_DIRNAME).glob("*")
        if path.is_dir()
        and ((path / TOPIC_TICKET_NAME).exists() or (path / TICKET_NAME).exists())
    )
    if len(candidates) == 1:
        return candidates[0], True, candidates[0].name
    return cache_dir, False, None


def _pending_topics(cache_dir: Path) -> list[str]:
    root = Path(cache_dir) / TOPIC_STATE_DIRNAME
    return sorted(
        path.name
        for path in root.glob("*")
        if path.is_dir()
        and ((path / TOPIC_TICKET_NAME).exists() or (path / TICKET_NAME).exists())
    )


def _paths(cache_dir: Path, topic_id: str | None) -> tuple[Path, Path, bool, str | None]:
    root, scoped, resolved_topic = _state_root(Path(cache_dir), topic_id)
    if scoped:
        return (
            root / TOPIC_TICKET_NAME,
            root / TOPIC_IN_PROGRESS_NAME,
            True,
            resolved_topic,
        )
    return root / TICKET_NAME, root / IN_PROGRESS_NAME, False, resolved_topic


def describe(cache_dir: Path, topic_id: str | None = None) -> dict[str, object]:
    """Report pending state for one topic, or the legacy root state."""
    if topic_id is None:
        topics = _pending_topics(Path(cache_dir))
        if len(topics) > 1:
            return {
                "blocked": True,
                "state": "multiple_pending",
                "topics": topics,
            }
    ticket_file, in_progress, scoped, resolved_topic = _paths(cache_dir, topic_id)
    if not ticket_file.exists():
        report: dict[str, object] = {
            "blocked": in_progress.exists(),
            "state": "in_progress" if in_progress.exists() else "clear",
        }
        if resolved_topic is not None:
            report["topic_id"] = resolved_topic
        return report
    ticket = json.loads(ticket_file.read_text(encoding="utf-8"))
    not_before = float(ticket.get("not_before_unix", 0.0))
    report = {
        "blocked": True,
        "state": "pending",
        "ticket": ticket.get("ticket"),
        "query": ticket.get("query"),
        "index": ticket.get("index"),
        "hits": ticket.get("hits"),
        "failure_type": ticket.get("failure_type"),
        "retry_after_seconds": ticket.get("retry_after_seconds"),
        "seconds_until_eligible": max(0.0, not_before - time.time()),
        "eligible_now": time.time() >= not_before,
    }
    if resolved_topic is not None or ticket.get("topic_id") is not None:
        report["topic_id"] = ticket.get("topic_id", resolved_topic)
    return report


def resume(cache_dir: Path, topic_id: str | None = None) -> dict[str, object]:
    """Reissue exactly one pending request and clear its topic lease."""
    if topic_id is None and len(_pending_topics(Path(cache_dir))) > 1:
        raise SystemExit("topic id is required when multiple continuation topics are pending")
    ticket_file, _in_progress, scoped, resolved_topic = _paths(cache_dir, topic_id)
    if not ticket_file.exists():
        raise SystemExit("no pending continuation ticket to resume")
    ticket = json.loads(ticket_file.read_text(encoding="utf-8"))
    if time.time() < float(ticket.get("not_before_unix", 0.0)):
        raise SystemExit("the recorded not-before time has not elapsed yet")

    effective_topic = str(ticket.get("topic_id") or resolved_topic or "continuation")
    corpus_epoch = ticket.get("corpus_epoch")
    if not isinstance(corpus_epoch, str) or not corpus_epoch.strip() or corpus_epoch.lower() in {
        "unspecified",
        "unknown",
    }:
        raise SystemExit("pending continuation is missing an explicit corpus epoch")
    config = RetrieverConfig(
        name="deepagent_climbmix",
        type="pyserini_remote",
        query_variants=(str(ticket.get("variant_name", "followup")),),
        hits=int(ticket["hits"]),
        index=str(ticket["index"]),
        corpus_epoch=corpus_epoch,
        cache=True,
    )
    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=Path(cache_dir),
        continuation_ticket=str(ticket["ticket"]),
        corpus_epoch=corpus_epoch,
    )
    query = QueryVariant(
        topic_id=effective_topic,
        variant_name=str(ticket.get("variant_name", "followup")),
        query_text=str(ticket["query"]),
        source_type="agent",
    )
    derived = request_cache_key(
        config,
        query,
        index_url=str(ticket["index_url"]),
        corpus_epoch=corpus_epoch,
    )
    if derived != ticket.get("request_key", ticket.get("cache_key")):
        raise SystemExit(
            "the pending request cannot be rebuilt from this configuration; "
            "discard it instead of issuing a different request"
        )
    candidates = retriever.retrieve(query)
    return {
        "resumed": True,
        "candidates": len(candidates),
        "state": describe(cache_dir, topic_id=effective_topic)["state"],
    }


def discard(cache_dir: Path, topic_id: str | None = None) -> dict[str, object]:
    """Drop one pending ticket without making a hosted call."""
    if topic_id is None and len(_pending_topics(Path(cache_dir))) > 1:
        raise SystemExit("topic id is required when multiple continuation topics are pending")
    ticket_file, in_progress, scoped, resolved_topic = _paths(cache_dir, topic_id)
    state_root = ticket_file.parent
    state_root.mkdir(parents=True, exist_ok=True)
    state_lock = FileLock(str(state_root / "state.lock"))
    removed: list[str] = []
    with state_lock:
        completion = state_root / TOPIC_COMPLETION_NAME
        if completion.exists():
            try:
                marker = json.loads(completion.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SystemExit("cannot safely discard invalid completion marker") from exc
            if (
                not isinstance(marker, dict)
                or marker.get("schema_version") != CONTINUATION_COMPLETION_SCHEMA_VERSION
                or marker.get("state") != "completed"
            ):
                raise SystemExit("cannot safely discard unknown completion marker state")
        if in_progress.exists():
            try:
                progress = json.loads(in_progress.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SystemExit("cannot safely discard invalid retrieval lease state") from exc
            if not isinstance(progress, dict):
                raise SystemExit("cannot safely discard invalid retrieval lease state")
            lease_state = progress.get("lease_state")
            if lease_state not in KNOWN_TOPIC_LEASE_STATES:
                raise SystemExit(
                    f"cannot safely discard unknown retrieval lease state: {lease_state!r}"
                )
            try:
                lease_expires = float(progress.get("lease_expires_unix", 0.0))
            except (TypeError, ValueError) as exc:
                raise SystemExit("cannot safely discard invalid retrieval lease state") from exc
            if not math.isfinite(lease_expires):
                raise SystemExit("cannot safely discard invalid retrieval lease state")
            if (
                lease_expires > time.time()
                and lease_state in {"active_recovery", "active_transport"}
            ):
                raise SystemExit(
                    f"cannot discard topic with active {lease_state} lease"
                )
        for path in (ticket_file, in_progress):
            if path.exists():
                removed.append(path.name)
                path.unlink()
    report = {
        "discarded": removed,
        "state": describe(cache_dir, topic_id=topic_id)["state"],
    }
    if resolved_topic is not None:
        report["topic_id"] = resolved_topic
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-id", help="inspect or recover one topic shard")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume",
        action="store_true",
        help="retry the pending request; makes one hosted call",
    )
    action.add_argument(
        "--discard",
        action="store_true",
        help="delete the pending ticket without retrying it",
    )
    arguments = parser.parse_args(argv)

    root = find_repo_root()
    load_repo_env(root)
    cache_dir = cache_directory(root)

    if arguments.resume:
        report = resume(cache_dir, topic_id=arguments.topic_id)
    elif arguments.discard:
        report = discard(cache_dir, topic_id=arguments.topic_id)
    else:
        report = describe(cache_dir, topic_id=arguments.topic_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Inspect, resume, or discard a pending hosted-retrieval continuation ticket.

A throttled ClimbMix request writes a ticket and refuses every later request
until that exact request is retried. The ticket records the query, so recovering
never requires remembering what was in flight when a run was interrupted.

    python -m trec_rag.continuation              # show what is pending
    python -m trec_rag.continuation --resume     # retry it, then unblock
    python -m trec_rag.continuation --discard    # drop it without retrying
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant
from trec_rag.repo_env import find_repo_root, load_repo_env, repo_cache_root
from trec_rag.retrievers import PyseriniRemoteRetriever, request_cache_key

TICKET_NAME = "continuation-ticket.json"
IN_PROGRESS_NAME = "continuation-in-progress.json"


def cache_directory(root: Path | None = None) -> Path:
    resolved = Path(root) if root is not None else find_repo_root()
    return repo_cache_root(resolved) / "retrieval" / "pyserini_remote"


def describe(cache_dir: Path) -> dict[str, object]:
    """Report what, if anything, is blocking hosted retrieval."""
    ticket_file = cache_dir / TICKET_NAME
    in_progress = cache_dir / IN_PROGRESS_NAME
    if not ticket_file.exists():
        return {
            "blocked": in_progress.exists(),
            "state": "in_progress" if in_progress.exists() else "clear",
        }
    ticket = json.loads(ticket_file.read_text(encoding="utf-8"))
    not_before = float(ticket.get("not_before_unix", 0.0))
    return {
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


def resume(cache_dir: Path) -> dict[str, object]:
    """Reissue exactly the pending request so the retriever unblocks."""
    ticket_file = cache_dir / TICKET_NAME
    if not ticket_file.exists():
        raise SystemExit("no pending continuation ticket to resume")
    ticket = json.loads(ticket_file.read_text(encoding="utf-8"))
    if time.time() < float(ticket.get("not_before_unix", 0.0)):
        raise SystemExit("the recorded not-before time has not elapsed yet")

    config = RetrieverConfig(
        name="deepagent_climbmix",
        type="pyserini_remote",
        query_variants=("original", "followup"),
        hits=int(ticket["hits"]),
        index=str(ticket["index"]),
        cache=True,
    )
    os.environ["PYSERINI_CONTINUATION_TICKET"] = str(ticket["ticket"])
    retriever = PyseriniRemoteRetriever(config, cache_dir=cache_dir)
    query = QueryVariant(
        topic_id="continuation",
        variant_name="followup",
        query_text=str(ticket["query"]),
        source_type="agent",
    )
    derived = request_cache_key(
        config, query, index_url=retriever.client.config.index_url
    )
    if derived != ticket.get("cache_key"):
        raise SystemExit(
            "the pending request cannot be rebuilt from this configuration; "
            "discard it instead of issuing a different request"
        )
    candidates = retriever.retrieve(query)
    return {
        "resumed": True,
        "candidates": len(candidates),
        "state": describe(cache_dir)["state"],
    }


def discard(cache_dir: Path) -> dict[str, object]:
    """Drop the pending ticket without spending a hosted call."""
    removed = [
        name
        for name in (TICKET_NAME, IN_PROGRESS_NAME)
        if (cache_dir / name).exists()
    ]
    for name in removed:
        (cache_dir / name).unlink()
    return {"discarded": removed, "state": describe(cache_dir)["state"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        report = resume(cache_dir)
    elif arguments.discard:
        report = discard(cache_dir)
    else:
        report = describe(cache_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

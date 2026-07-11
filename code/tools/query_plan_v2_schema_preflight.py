#!/usr/bin/env python3
"""Compile exact planner schemas inside the pinned local vLLM image.

This performs schema/linter work only.  It does not contact the HTTP server or
schedule model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from trec_rag.query_plan_v2_cli import (
    DIAGNOSTIC_TOPIC_IDS,
    SYNTHETIC_SMOKE_TOPIC,
)
from trec_rag.query_planner import (
    V2_PROMPT_VERSION,
    V2_SCHEMA_VERSION,
    query_plan_v2_json_schema,
    tokenize_narrative,
)
from trec_rag.query_schema_compat import (
    find_vllm_xgrammar_unsupported_features,
)
from trec_rag.repo_env import find_repo_root
from trec_rag.topics import Topic, load_topics


PINNED_IMAGE_DIGEST = (
    "sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)
COMPILER_PROGRAM = r"""
import importlib.metadata
import json
import sys

import vllm
from llguidance import JsonCompiler
from xgrammar import Grammar

schema = json.load(sys.stdin)
Grammar.from_json_schema(schema, strict_mode=True)
JsonCompiler().compile(json.dumps(schema), check=True)
print(json.dumps({
    "vllm": vllm.__version__,
    "xgrammar": importlib.metadata.version("xgrammar"),
    "llguidance": importlib.metadata.version("llguidance"),
    "xgrammar_strict": "pass",
    "llguidance_check": "pass"
}, sort_keys=True))
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _schema_sha256(schema: object) -> str:
    return hashlib.sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )


def _container_identity(container: str) -> dict[str, object]:
    inspected = json.loads(_run(["podman", "inspect", container]).stdout)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise ValueError("podman inspect did not return exactly one container")
    row = inspected[0]
    image_id = row.get("Image")
    config = row.get("Config")
    if not isinstance(image_id, str) or not isinstance(config, dict):
        raise ValueError("container inspection lacks image/config identity")
    image_rows = json.loads(_run(["podman", "image", "inspect", image_id]).stdout)
    if not isinstance(image_rows, list) or len(image_rows) != 1:
        raise ValueError("podman image inspect did not return exactly one image")
    image = image_rows[0]
    digest = image.get("Digest")
    if digest != PINNED_IMAGE_DIGEST:
        raise ValueError(
            f"container image digest mismatch: {digest!r} != {PINNED_IMAGE_DIGEST!r}"
        )
    return {
        "container_name": container,
        "image_name": config.get("Image"),
        "image_id": image_id,
        "image_digest": digest,
        "launch_command": config.get("Cmd"),
    }


def _compile_schema(container: str, schema: dict[str, object]) -> dict[str, object]:
    result = _run(
        ["podman", "exec", "-i", container, "python", "-c", COMPILER_PROGRAM],
        input_text=json.dumps(schema, ensure_ascii=False, sort_keys=True),
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("compiler result is not a JSON object")
    return value


def _topics(topics_path: Path, topic_format: str) -> list[Topic]:
    by_id = {
        topic.id: topic
        for topic in load_topics(topics_path, topic_format=topic_format)
    }
    missing = [topic_id for topic_id in DIAGNOSTIC_TOPIC_IDS if topic_id not in by_id]
    if missing:
        raise ValueError("missing diagnostic topics: " + ", ".join(missing))
    return [SYNTHETIC_SMOKE_TOPIC] + [
        by_id[topic_id] for topic_id in DIAGNOSTIC_TOPIC_IDS
    ]


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    compiler_versions: dict[str, object] | None = None
    for topic in _topics(args.topics, args.topic_format):
        tape = tokenize_narrative(topic.narrative)
        schema = query_plan_v2_json_schema(
            topic_id=topic.id,
            token_count=tape.token_count,
        )
        issues = find_vllm_xgrammar_unsupported_features(schema)
        if issues:
            raise ValueError(
                f"schema {topic.id} has unsupported features: "
                + "; ".join(issue.json_path for issue in issues)
            )
        compiled = _compile_schema(args.container, schema)
        if compiler_versions is None:
            compiler_versions = compiled
        elif compiled != compiler_versions:
            raise ValueError("compiler identity/result changed within preflight")
        rows.append(
            {
                "topic_id": topic.id,
                "narrative_sha256": tape.narrative_sha256,
                "token_count": tape.token_count,
                "schema_sha256": _schema_sha256(schema),
                "unsupported_feature_count": 0,
                "xgrammar_strict": compiled.get("xgrammar_strict"),
                "llguidance_check": compiled.get("llguidance_check"),
            }
        )
    return {
        "manifest_version": "query_plan_v2_schema_preflight_v1",
        "created_at": _utc_now(),
        "execution_boundary": "offline_compiler_only_no_http_or_inference",
        "schema_version": V2_SCHEMA_VERSION,
        "prompt_version": V2_PROMPT_VERSION,
        "required_server_backend": "xgrammar",
        "container": _container_identity(args.container),
        "compiler": compiler_versions,
        "schemas": rows,
        "all_passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = find_repo_root(Path.cwd())
    parser = argparse.ArgumentParser(
        description="Compile exact v2 planner schemas without model inference."
    )
    parser.add_argument("--container", default="trec-rag-gpt-oss")
    parser.add_argument(
        "--topics",
        type=Path,
        default=(
            repo_root
            / "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
        ),
    )
    parser.add_argument("--topic-format", choices=("tsv", "jsonl"), default="tsv")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_root
            / "reports/experiments/query_planner_v2_schema_compat_v2_1/compiler_manifest_xgrammar.json"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

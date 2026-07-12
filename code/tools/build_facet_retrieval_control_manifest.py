#!/usr/bin/env python3
"""Create the frozen facet retrieval-control pilot manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trec_rag.facet_retrieval_control_manifest import build_control_manifest


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_control_manifest(args.r1)
    _write_create_only(args.output, manifest.to_dict())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the exact 31-stream source boundary for facet-local MiniLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from trec_rag.facet_local_minilm_manifest import (
    build_facet_local_manifest,
    write_facet_local_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r1-cache", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    if args.manifest_output.exists():
        raise FileExistsError(
            f"create-only output already exists: {args.manifest_output}"
        )
    manifest = build_facet_local_manifest(
        r1_manifest_path=args.r1_manifest,
        prior_freeze_path=args.prior_freeze,
        base_run=args.base_run,
        base_cache=args.base_cache,
        r1_run=args.r1_run,
        r1_cache=args.r1_cache,
        source_output=args.source_output,
    )
    write_facet_local_manifest(args.manifest_output, manifest)
    print(
        f"froze {len(manifest.streams)} streams and "
        f"{manifest.candidate_rows} candidate rows"
    )


if __name__ == "__main__":
    main()

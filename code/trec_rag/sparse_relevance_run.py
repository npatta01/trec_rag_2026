"""Preflight and execute the matched R0/R1 rate-limited retrieval pilot."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from trec_rag.det_sparse_ledger import (
    RawTransportResponse,
    RetrievalLedger,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTransport,
)
from trec_rag.remote_client import rate_limited_session
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.sparse_relevance_manifest import (
    ManifestPair,
    RepairManifest,
    load_repair_manifest,
    validate_manifest_pair,
)


INDEX_ID = "climbmix-400b"
RETRIEVER_VERSION = "pyserini_remote_raw_first_v1"


def build_requests(
    manifest: RepairManifest, *, endpoint: str
) -> tuple[RetrievalRequest, ...]:
    return tuple(
        RetrievalRequest.from_query(
            topic_id=stream.topic_id,
            variant_name=(
                f"sparse_relevance_v1:{manifest.renderer}:{stream.stream_id}"
            ),
            query_text=stream.query,
            retriever_version=RETRIEVER_VERSION,
            index_url=endpoint,
            index_id=INDEX_ID,
            hits=100,
            analyzer_fingerprint_sha256=manifest.analyzer_fingerprint_sha256,
        )
        for stream in manifest.streams
    )


def preflight_pair(
    pair: ManifestPair,
    *,
    endpoint: str,
    shared_cache_dir: Path,
    probe_dir: Path,
) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for manifest in (pair.r0, pair.r1):
        ledger = RetrievalLedger(
            Path(probe_dir) / manifest.renderer,
            shared_cache_dir=shared_cache_dir,
            max_calls=22,
            max_calls_per_topic=9,
        )
        requests = build_requests(manifest, endpoint=endpoint)
        cache_hits = sum(ledger.has_verified_cache(request) for request in requests)
        report[manifest.renderer] = {
            "planned": len(requests),
            "verified_cache_hits": cache_hits,
            "external_calls": len(requests) - cache_hits,
        }
    report["total"] = {
        key: report["R0"][key] + report["R1"][key]
        for key in ("planned", "verified_cache_hits", "external_calls")
    }
    if report["R0"]["external_calls"] > 22 or report["R1"]["external_calls"] > 22:
        raise ValueError("renderer external-call ceiling exceeded")
    return report


def execute_renderer(
    manifest: RepairManifest,
    *,
    ledger: RetrievalLedger,
    transport: RetrievalTransport,
    endpoint: str,
    progress: Callable[[int, int, RetrievalRequest, RetrievalResult], None] | None = None,
) -> tuple[RetrievalResult, ...]:
    requests = build_requests(manifest, endpoint=endpoint)
    if len(requests) > manifest.max_external_requests:
        raise ValueError("renderer request plan exceeds frozen ceiling")
    results: list[RetrievalResult] = []
    for index, request in enumerate(requests, start=1):
        result = ledger.retrieve(request, transport)
        results.append(result)
        if progress is not None:
            progress(index, len(requests), request, result)
    validation = ledger.validate_run()
    if validation.failures or validation.pending:
        raise ValueError("renderer ledger is not complete")
    if validation.planned_requests != len(requests):
        raise ValueError("renderer ledger request count differs from manifest")
    return tuple(results)


class RateLimitedLedgerTransport:
    """One-attempt requests transport that lets the ledger capture raw bytes first."""

    def __init__(self, config: RemotePyseriniConfig, *, timeout: int = 30) -> None:
        self.config = config
        self.timeout = timeout
        self.session = rate_limited_session(config)

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        if request.identity.index_url != self.config.index_url:
            raise ValueError("request endpoint differs from rate-limited transport")
        started = time.monotonic()
        params = {"query": request.query_text, "hits": str(request.identity.hits)}
        if request.identity.bm25_k1 is not None:
            params["k1"] = str(float(request.identity.bm25_k1))
            params["b"] = str(float(request.identity.bm25_b))
        response = self.session.get(
            self.config.index_url,
            params=params,
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token
                    else {}
                ),
            },
            timeout=self.timeout,
            allow_redirects=False,
        )
        return RawTransportResponse(
            status=response.status_code,
            headers=dict(response.headers),
            body=response.content,
            elapsed_seconds=time.monotonic() - started,
        )


def build_live_config(
    *,
    endpoint: str,
    api_token: str | None,
    limiter_state: Path,
    min_interval_seconds: float,
) -> RemotePyseriniConfig:
    if min_interval_seconds < 3.0:
        raise ValueError("live sparse-relevance pacing cannot be below three seconds")
    return RemotePyseriniConfig(
        index_url=endpoint,
        api_token=api_token,
        hits=100,
        queries=(),
        min_interval_seconds=min_interval_seconds,
        burst=1,
        limiter_state_path=limiter_state,
    )


def _progress(index, count, request, result):
    source = "cache" if result.cache_hit else "external"
    print(
        f"{index:02d}/{count} {request.identity.topic_id} "
        f"{request.identity.variant_name} {source} {len(result.candidates)}",
        flush=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limiter-state", type=Path, required=True)
    parser.add_argument("--min-interval-seconds", type=float, default=3.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    pair = validate_manifest_pair(
        load_repair_manifest(args.r0), load_repair_manifest(args.r1)
    )
    preflight = preflight_pair(
        pair,
        endpoint=args.endpoint,
        shared_cache_dir=args.shared_cache,
        probe_dir=args.output / "preflight-probe",
    )
    _write_json(args.output / "preflight.json", preflight)
    print(json.dumps(preflight, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    config = build_live_config(
        endpoint=args.endpoint,
        api_token=os.environ.get("PYSERINI_API_TOKEN"),
        min_interval_seconds=args.min_interval_seconds,
        limiter_state=args.limiter_state,
    )
    transport = RateLimitedLedgerTransport(config)
    final: dict[str, object] = {"preflight": preflight, "renderers": {}}
    for manifest in (pair.r0, pair.r1):
        ledger = RetrievalLedger(
            args.output / manifest.renderer / "ledger",
            shared_cache_dir=args.shared_cache,
            max_calls=22,
            max_calls_per_topic=9,
        )
        results = execute_renderer(
            manifest,
            ledger=ledger,
            transport=transport,
            endpoint=args.endpoint,
            progress=_progress,
        )
        report = ledger.validate_run()
        final["renderers"][manifest.renderer] = {
            **asdict(report),
            "planned_requests": report.planned_requests,
            "result_count": len(results),
        }
    _write_json(args.output / "retrieval_summary.json", final)
    print("VALIDATION " + json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

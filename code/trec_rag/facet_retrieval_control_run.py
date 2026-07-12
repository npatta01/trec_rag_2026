"""Create-only, rate-limited runner for the frozen facet retrieval controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.parse
from pathlib import Path
from typing import Mapping, Sequence

import requests

from .det_sparse_ledger import (
    RawTransportResponse,
    RetrievalLedger,
    RetrievalLedgerError,
    RetrievalRequest,
    RetrievalTransport,
)
from .facet_retrieval_control_manifest import (
    MAX_EXTERNAL_REQUESTS,
    MIN_INTERVAL_SECONDS,
    PROTECTED_TOPIC_IDS,
    ControlManifest,
    _validate_manifest as _validate_exact_manifest,
    build_control_manifest,
    load_control_manifest,
)
from .remote_client import rate_limited_session
from .remote_config import RemotePyseriniConfig


RETRIEVER_VERSION = "pyserini_remote_bm25_controls_v1"
INDEX_ID = "climbmix-400b"
PREFLIGHT_SCHEMA_VERSION = "facet-retrieval-control-preflight-v1"
SUMMARY_SCHEMA_VERSION = "facet-retrieval-control-summary-v1"
_CONTROL_ARM_IDS = ("W0", "W1", "W2")
_TIMEOUT_SECONDS = 30.0


def _reject_protected(manifest: ControlManifest) -> None:
    for stream in manifest.streams:
        if stream.topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {stream.topic_id} is forbidden")


def _validate_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("endpoint must be non-empty text")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint must be an HTTP(S) URL without userinfo, query, or fragment"
        )
    return endpoint


def _manifest_sha256(manifest: ControlManifest) -> str:
    encoded = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runner_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON artifact without replacing any existing evidence."""

    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _reject_existing_outputs(run_dir: Path) -> None:
    for name in ("preflight.json", "retrieval_summary.json"):
        path = Path(run_dir) / name
        if path.exists():
            raise FileExistsError(f"create-only output already exists: {path}")


def build_control_requests(
    manifest: ControlManifest,
    *,
    endpoint: str,
) -> tuple[RetrievalRequest, ...]:
    """Build the exact W0/W1/W2 request identities, never B0."""

    _reject_protected(manifest)
    endpoint = _validate_endpoint(endpoint)
    planned = tuple(
        (stream, arm)
        for stream in manifest.streams
        for arm in stream.arms
        if arm.arm_id in _CONTROL_ARM_IDS and arm.external
    )
    if len(planned) != MAX_EXTERNAL_REQUESTS:
        raise ValueError(
            "control run requires exactly 12 planned W0/W1/W2 requests; "
            f"found {len(planned)}"
        )
    _validate_exact_manifest(manifest)
    if any(
        tuple(arm.arm_id for arm in stream.arms if arm.external)
        != _CONTROL_ARM_IDS
        for stream in manifest.streams
    ):
        raise ValueError("control run may build only W0/W1/W2 external requests")

    return tuple(
        RetrievalRequest.from_query(
            topic_id=stream.topic_id,
            variant_name=f"facet_control_v1:{arm.arm_id}:{stream.stream_id}",
            query_text=stream.reweighted_query,
            index_url=endpoint,
            index_id=INDEX_ID,
            hits=manifest.hits,
            analyzer_fingerprint_sha256=manifest.analyzer_fingerprint_sha256,
            retriever_version=RETRIEVER_VERSION,
            bm25_k1=arm.k1,
            bm25_b=arm.b,
        )
        for stream, arm in planned
    )


def build_control_live_config(
    endpoint: str,
    api_token: str | None,
    limiter_state_path: Path,
    min_interval_seconds: float,
) -> RemotePyseriniConfig:
    """Build the only live configuration accepted by the control transport."""

    endpoint = _validate_endpoint(endpoint)
    if (
        isinstance(min_interval_seconds, bool)
        or not isinstance(min_interval_seconds, (int, float))
        or not math.isfinite(float(min_interval_seconds))
        or float(min_interval_seconds) < MIN_INTERVAL_SECONDS
    ):
        raise ValueError("control retrieval requires at least ten seconds between starts")
    if api_token is not None and (not isinstance(api_token, str) or not api_token):
        raise ValueError("api_token must be non-empty text or None")
    return RemotePyseriniConfig(
        index_url=endpoint,
        api_token=api_token,
        hits=100,
        queries=(),
        min_interval_seconds=float(min_interval_seconds),
        burst=1,
        limiter_state_path=Path(limiter_state_path),
    )


class RateLimitedControlTransport:
    """Send one no-retry/no-redirect GET through the persistent limiter session."""

    transport_version = "facet_control_rate_limited_http_v1"
    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        config: RemotePyseriniConfig,
        *,
        allowed_requests: Sequence[RetrievalRequest],
        session: requests.Session | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        if config.hits != 100 or config.burst != 1:
            raise ValueError("control transport requires hits=100 and burst=1")
        if config.min_interval_seconds < MIN_INTERVAL_SECONDS:
            raise ValueError("control transport requires at least ten seconds between starts")
        if timeout_seconds != _TIMEOUT_SECONDS:
            raise ValueError("control transport timeout is frozen at 30 seconds")
        self.config = config
        self.endpoint_url = _validate_endpoint(config.index_url)
        self.timeout_seconds = timeout_seconds
        allowed_by_key = {
            request.identity.request_key: request for request in allowed_requests
        }
        canonical_requests = build_control_requests(
            build_control_manifest(),
            endpoint=self.endpoint_url,
        )
        canonical_by_key = {
            request.identity.request_key: request for request in canonical_requests
        }
        if (
            len(allowed_requests) != len(canonical_requests)
            or allowed_by_key != canonical_by_key
        ):
            raise ValueError(
                "control transport allowlist differs from the canonical 12 requests"
            )
        self._allowed_requests = allowed_by_key
        self.session = session if session is not None else rate_limited_session(config)

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        identity = request.identity
        if self._allowed_requests.get(identity.request_key) != request:
            raise ValueError("request is not in the frozen control allowlist")
        if identity.topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {identity.topic_id} is forbidden")
        if identity.index_url != self.endpoint_url:
            raise ValueError("request endpoint differs from the control transport")
        if (
            identity.index_id != INDEX_ID
            or identity.hits != 100
            or identity.retriever_version != RETRIEVER_VERSION
        ):
            raise ValueError("request identity differs from the frozen control protocol")
        if identity.bm25_k1 is None or identity.bm25_b is None:
            raise ValueError("control requests require explicit k1 and b")

        started = time.monotonic()
        response = self.session.get(
            self.endpoint_url,
            params={
                "query": request.query_text,
                "hits": "100",
                "k1": str(float(identity.bm25_k1)),
                "b": str(float(identity.bm25_b)),
            },
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token is not None
                    else {}
                ),
            },
            timeout=self.timeout_seconds,
            allow_redirects=False,
        )
        elapsed = getattr(response, "elapsed", None)
        elapsed_seconds = (
            float(elapsed.total_seconds())
            if elapsed is not None and hasattr(elapsed, "total_seconds")
            else time.monotonic() - started
        )
        body = response.content
        if not isinstance(body, bytes):
            body = bytes(body)
        return RawTransportResponse(
            status=int(response.status_code),
            headers=dict(response.headers),
            body=body,
            elapsed_seconds=elapsed_seconds,
        )


def preflight_control(
    manifest: ControlManifest,
    ledger: RetrievalLedger,
    endpoint: str,
) -> dict[str, object]:
    """Probe exact cache identities without recording planned invocations."""

    _reject_protected(manifest)
    _validate_exact_manifest(manifest)
    requests_to_run = build_control_requests(manifest, endpoint=endpoint)
    cache_status = tuple(
        ledger.has_verified_cache(request) for request in requests_to_run
    )
    existing_external_attempts = ledger.call_count()
    projected_external_attempts = (
        existing_external_attempts + len(requests_to_run) - sum(cache_status)
    )
    if projected_external_attempts > MAX_EXTERNAL_REQUESTS:
        raise ValueError(
            "control run would exceed the hard ceiling of 12 external attempts; "
            f"projected {projected_external_attempts}"
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "endpoint": endpoint,
        "manifest_sha256": _manifest_sha256(manifest),
        "runner_code_sha256": _runner_code_sha256(),
        "planned_requests": len(requests_to_run),
        "verified_cache_hits": sum(cache_status),
        "existing_external_attempts": existing_external_attempts,
        "projected_external_attempts": projected_external_attempts,
        "max_external_attempts": MAX_EXTERNAL_REQUESTS,
        "requests": [
            {
                "request_key": request.identity.request_key,
                "identity": request.identity.canonical_dict(),
                "query_text": request.query_text,
                "verified_cache": verified_cache,
            }
            for request, verified_cache in zip(
                requests_to_run,
                cache_status,
                strict=True,
            )
        ],
    }


def _execute_preflighted(
    manifest: ControlManifest,
    ledger: RetrievalLedger,
    transport: RetrievalTransport,
    endpoint: str,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    for request in build_control_requests(manifest, endpoint=endpoint):
        # Deliberately uncaught: one failed attempt stops the run immediately.
        ledger.retrieve(request, transport)

    report = ledger.validate_run()
    complete = (
        report.failures == 0
        and report.pending == 0
        and report.planned_requests == MAX_EXTERNAL_REQUESTS
    )
    if not complete:
        raise RetrievalLedgerError(
            "control run is incomplete: expected exactly 12 planned invocations "
            "with zero failures and pending attempts"
        )
    preflight_path = ledger.run_dir / "preflight.json"
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "complete": True,
        "manifest_sha256": preflight["manifest_sha256"],
        "runner_code_sha256": preflight["runner_code_sha256"],
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "planned_requests": report.planned_requests,
        "external_calls": report.external_calls,
        "successes": report.successes,
        "failures": report.failures,
        "pending": report.pending,
        "cache_hits": report.cache_hits,
        "per_topic_external_calls": report.per_topic_external_calls,
    }
    _create_json(ledger.run_dir / "retrieval_summary.json", summary)
    return summary


def execute_control(
    manifest: ControlManifest,
    ledger: RetrievalLedger,
    transport: RetrievalTransport,
    endpoint: str,
) -> dict[str, object]:
    """Execute the 12 planned invocations, stopping on the first exception."""

    _reject_protected(manifest)
    _reject_existing_outputs(ledger.run_dir)
    preflight_path = ledger.run_dir / "preflight.json"
    preflight = preflight_control(manifest, ledger, endpoint)
    _create_json(preflight_path, preflight)
    return _execute_preflighted(manifest, ledger, transport, endpoint, preflight)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limiter-state", type=Path, required=True)
    parser.add_argument("--min-interval-seconds", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_control_manifest(args.manifest)
    # This gate intentionally precedes ledger/cache writes and session construction.
    _reject_protected(manifest)
    config = build_control_live_config(
        args.endpoint,
        os.environ.get("PYSERINI_API_TOKEN") or None,
        args.limiter_state,
        args.min_interval_seconds,
    )
    _reject_existing_outputs(args.output)
    ledger = RetrievalLedger(
        args.output,
        shared_cache_dir=args.shared_cache,
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=6,
        min_results=50,
        required_text_results=50,
    )
    preflight = preflight_control(manifest, ledger, args.endpoint)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    # Constructing the real rate-limited session happens only after every preflight gate.
    allowed_requests = build_control_requests(manifest, endpoint=args.endpoint)
    transport = RateLimitedControlTransport(
        config,
        allowed_requests=allowed_requests,
    )
    _create_json(ledger.run_dir / "preflight.json", preflight)
    summary = _execute_preflighted(
        manifest,
        ledger,
        transport,
        args.endpoint,
        preflight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

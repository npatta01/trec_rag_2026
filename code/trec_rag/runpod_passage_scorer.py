"""Strict Runpod Flash client for batched Mixedbread passage scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import re
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SCORE_REPRESENTATION,
    ScoreCacheContext,
)


REQUEST_SCHEMA_VERSION = "runpod_passage_score_request_v1"
RESPONSE_SCHEMA_VERSION = "runpod_passage_score_response_v1"
IDENTITY_SCHEMA_VERSION = "runpod_passage_score_identity_v1"
MIXEDBREAD_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
MIXEDBREAD_REVISION = DEFAULT_MODEL_REVISION
REMOTE_BACKEND = "runpod-flash-sentence-transformers-cross-encoder"
REMOTE_BACKEND_VERSION = DEFAULT_BACKEND_VERSION
REMOTE_DEVICE_FAMILY = "nvidia-geforce-rtx-4090"
REMOTE_DEVICE = f"runpod:{REMOTE_DEVICE_FAMILY}"
REMOTE_MODEL_BATCH_SIZE = 16
REMOTE_IMPLEMENTATION_VERSION = 1
MAX_LENGTH = 1024
INPUT_POLICY = "topic_passage_query_text_v1"
SCORE_KIND = "topic_passage_relevance_v1"
MAX_REQUEST_PASSAGES = 256
MAX_REQUEST_BYTES = 6 * 1024 * 1024
_TORCH_VERSION = "2.9.1"
_TRANSFORMERS_VERSION = "5.13.0"
_FIXED_BATCH_POLICY = f"cross-encoder-predict-batch-size-{REMOTE_MODEL_BATCH_SIZE}"
_SAFE_ENDPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ACTIVE_JOB_STATES = frozenset({"IN_QUEUE", "IN_PROGRESS", "RUNNING"})
_FAILED_JOB_STATES = frozenset({"FAILED", "CANCELLED", "TIMED_OUT"})


class RemotePassageScorerError(RuntimeError):
    """A safe failure at the remote scoring boundary."""


class QueueJobClient(Protocol):
    def run(self, request: Mapping[str, object]) -> Mapping[str, object]: ...


JsonTransport = Callable[..., Mapping[str, object]]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("remote passage scorer values must be canonical JSON") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def remote_endpoint_identity() -> dict[str, object]:
    """Return the fixed identity that the Flash worker must echo exactly."""
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "backend": REMOTE_BACKEND,
        "backend_version": REMOTE_BACKEND_VERSION,
        "model": MIXEDBREAD_MODEL,
        "model_revision": MIXEDBREAD_REVISION,
        "max_length": MAX_LENGTH,
        "score_kind": SCORE_KIND,
        "score_representation": DEFAULT_SCORE_REPRESENTATION,
        "inference_dtype": DEFAULT_INFERENCE_DTYPE,
        "input_policy": INPUT_POLICY,
        "device_family": REMOTE_DEVICE_FAMILY,
        "fixed_batch_policy": _FIXED_BATCH_POLICY,
        "model_batch_size": REMOTE_MODEL_BATCH_SIZE,
        "torch_version": _TORCH_VERSION,
        "transformers_version": _TRANSFORMERS_VERSION,
        "implementation_version": REMOTE_IMPLEMENTATION_VERSION,
    }


def remote_endpoint_identity_sha256(
    identity: Mapping[str, object] | None = None,
) -> str:
    selected = remote_endpoint_identity() if identity is None else dict(identity)
    return _sha256_json(selected)


def remote_score_cache_context() -> ScoreCacheContext:
    """Use a CUDA-specific namespace that cannot mix with local ROCm scores."""
    return ScoreCacheContext(
        backend=REMOTE_BACKEND,
        backend_version=REMOTE_BACKEND_VERSION,
        model=MIXEDBREAD_MODEL,
        model_revision=MIXEDBREAD_REVISION,
        max_length=MAX_LENGTH,
        requested_max_length=MAX_LENGTH,
        score_kind=SCORE_KIND,
        score_representation=DEFAULT_SCORE_REPRESENTATION,
        inference_dtype=DEFAULT_INFERENCE_DTYPE,
        input_policy=INPUT_POLICY,
        scoring_contract=(
            "runpod-passage-score-v1:"
            + remote_endpoint_identity_sha256()
        ),
        transformers_version=_TRANSFORMERS_VERSION,
        torch_version=_TORCH_VERSION,
        device_family=REMOTE_DEVICE_FAMILY,
        fixed_batch_policy=_FIXED_BATCH_POLICY,
    )


def _request_batch_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("request_batch_size must be an integer")
    if value <= 0 or value > MAX_REQUEST_PASSAGES:
        raise ValueError(f"request_batch_size must be between 1 and {MAX_REQUEST_PASSAGES}")
    return value


def remote_scorer_identity(request_batch_size: int) -> dict[str, object]:
    size = _request_batch_size(request_batch_size)
    context = remote_score_cache_context()
    identity = remote_endpoint_identity()
    return {
        "backend": identity["backend"],
        "backend_version": identity["backend_version"],
        "model": identity["model"],
        "model_revision": identity["model_revision"],
        "score_representation": identity["score_representation"],
        "inference_dtype": identity["inference_dtype"],
        "max_length": identity["max_length"],
        "batch_size": identity["model_batch_size"],
        "device": REMOTE_DEVICE,
        "input_policy": identity["input_policy"],
        "implementation_version": 2,
        "request_batch_size": size,
        "endpoint_identity_sha256": remote_endpoint_identity_sha256(identity),
        "cache_context_sha256": context.context_sha256,
    }


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _retry_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_retries must be an integer")
    if value < 0 or value > 5:
        raise ValueError("max_retries must be between 0 and 5")
    return value


def _nonblank_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _content_id(query_text: str, passage_text: str) -> str:
    return _sha256_json(
        {
            "identity_sha256": remote_endpoint_identity_sha256(),
            "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "passage_sha256": hashlib.sha256(passage_text.encode("utf-8")).hexdigest(),
        }
    )


def _build_score_request(
    pairs: Sequence[tuple[str, str]],
    *,
    request_batch_size: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    rows = tuple(pairs)
    if len(rows) > request_batch_size or len(rows) > MAX_REQUEST_PASSAGES:
        raise ValueError(
            f"remote passage score request may contain at most {request_batch_size} passages"
        )
    if not rows:
        return {}, ()
    normalized: list[tuple[str, str]] = []
    for pair in rows:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("remote passage scorer pairs must contain query/passage pairs")
        query_text = _nonblank_text(pair[0], name="query text")
        passage_text = _nonblank_text(pair[1], name="passage text")
        normalized.append((query_text, passage_text))
    queries = {query_text for query_text, _passage_text in normalized}
    if len(queries) != 1:
        raise ValueError("one remote score request must contain exactly one query")
    query_text = normalized[0][0]
    passages = [
        {"content_id": _content_id(query_text, passage_text), "text": passage_text}
        for _query_text, passage_text in normalized
    ]
    content_ids = tuple(str(row["content_id"]) for row in passages)
    if len(set(content_ids)) != len(content_ids):
        raise ValueError("remote passage score request contains duplicate content identities")
    request_without_id: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "expected_identity_sha256": remote_endpoint_identity_sha256(),
        "query": query_text,
        "passages": passages,
    }
    request = {
        **request_without_id,
        "request_id": _sha256_json(request_without_id),
    }
    if len(_canonical_json(request)) > MAX_REQUEST_BYTES:
        raise ValueError("remote passage score request must not exceed 6 MiB")
    return request, content_ids


def _finite_remote_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemotePassageScorerError("remote score must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise RemotePassageScorerError("remote score must be a finite real number")
    return result


def _validate_score_response(
    response: Mapping[str, object],
    *,
    request: Mapping[str, object],
    content_ids: Sequence[str],
) -> tuple[float, ...]:
    if not isinstance(response, Mapping):
        raise RemotePassageScorerError("remote score response must be a mapping")
    if response.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise RemotePassageScorerError("remote score response schema differs")
    if response.get("request_id") != request.get("request_id"):
        raise RemotePassageScorerError("remote score response request identity differs")
    identity = response.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != remote_endpoint_identity():
        raise RemotePassageScorerError("remote score endpoint identity differs")
    digest = response.get("identity_sha256")
    if digest != remote_endpoint_identity_sha256(identity):
        raise RemotePassageScorerError("remote score endpoint identity digest differs")
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(content_ids):
        raise RemotePassageScorerError("remote score result count differs")
    actual_ids: list[str] = []
    scores: list[float] = []
    for row in results:
        if not isinstance(row, Mapping):
            raise RemotePassageScorerError("remote score results must be mappings")
        content_id = row.get("content_id")
        if not isinstance(content_id, str):
            raise RemotePassageScorerError("remote score result identities differ")
        actual_ids.append(content_id)
        scores.append(_finite_remote_score(row.get("score")))
    if tuple(actual_ids) != tuple(content_ids):
        raise RemotePassageScorerError("remote score result identities differ")
    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "passage_count",
        "model_batches",
    }:
        raise RemotePassageScorerError("remote score diagnostics differ")
    passage_count = diagnostics.get("passage_count")
    model_batches = diagnostics.get("model_batches")
    if (
        isinstance(passage_count, bool)
        or not isinstance(passage_count, int)
        or isinstance(model_batches, bool)
        or not isinstance(model_batches, int)
        or passage_count != len(content_ids)
        or model_batches != math.ceil(len(content_ids) / REMOTE_MODEL_BATCH_SIZE)
    ):
        raise RemotePassageScorerError("remote score diagnostics differ")
    return tuple(scores)


def _urllib_json_transport(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object] | None,
    timeout_seconds: float,
) -> Mapping[str, object]:
    body = None if payload is None else _canonical_json(payload)
    request = Request(url, data=body, headers=dict(headers), method=method)
    with urlopen(request, timeout=timeout_seconds) as response:
        try:
            decoded = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RemotePassageScorerError("Runpod queue returned invalid JSON") from None
    if not isinstance(decoded, Mapping):
        raise RemotePassageScorerError("Runpod queue response must be a mapping")
    return decoded


class RunpodQueueJobClient:
    """Submit one queue job and poll it without exposing remote response bodies."""

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        *,
        timeout_seconds: int | float = 900,
        max_retries: int = 3,
        poll_interval_seconds: int | float = 1,
        transport: JsonTransport = _urllib_json_transport,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(endpoint_id, str) or _SAFE_ENDPOINT_ID.fullmatch(endpoint_id) is None:
            raise ValueError("endpoint_id must be a safe Runpod endpoint identity")
        _nonblank_text(api_key, name="Runpod API key")
        if not callable(transport) or not callable(sleep) or not callable(monotonic):
            raise TypeError("queue client transport, sleep, and monotonic must be callable")
        self.endpoint_id = endpoint_id
        self._api_key = api_key
        self.timeout_seconds = _positive_number(timeout_seconds, name="timeout_seconds")
        if self.timeout_seconds < 5:
            raise ValueError("timeout_seconds must be at least 5")
        self.execution_timeout_ms = math.ceil(self.timeout_seconds * 1_000)
        self.max_retries = _retry_count(max_retries)
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or float(poll_interval_seconds) < 0
        ):
            raise ValueError("poll_interval_seconds must be a finite non-negative number")
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self._base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        deadline: float,
        retry_ambiguous_failures: bool,
    ) -> Mapping[str, object]:
        for attempt in range(self.max_retries + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise RemotePassageScorerError("Runpod queue job timed out")
            try:
                response = self._transport(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    payload=payload,
                    timeout_seconds=remaining,
                )
            except HTTPError as exc:
                retryable = exc.code == 429 or (
                    retry_ambiguous_failures and 500 <= exc.code < 600
                )
                if not retryable or attempt >= self.max_retries:
                    if not retry_ambiguous_failures and 500 <= exc.code < 600:
                        raise RemotePassageScorerError(
                            "Runpod queue submission outcome is unknown"
                        ) from None
                    raise RemotePassageScorerError(
                        f"Runpod queue request failed with HTTP {exc.code}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                self._sleep(max(0.0, min(delay, 60.0)))
                continue
            except (TimeoutError, URLError, OSError) as exc:
                if not retry_ambiguous_failures:
                    raise RemotePassageScorerError(
                        "Runpod queue submission outcome is unknown"
                    ) from None
                if attempt >= self.max_retries:
                    raise RemotePassageScorerError("Runpod queue transport failed") from exc
                self._sleep(float(min(2**attempt, 60)))
                continue
            if not isinstance(response, Mapping):
                raise RemotePassageScorerError("Runpod queue response must be a mapping")
            return response
        raise AssertionError("bounded retry loop did not return or raise")

    @staticmethod
    def _completed_output(response: Mapping[str, object]) -> Mapping[str, object] | None:
        status = response.get("status")
        if status == "COMPLETED":
            output = response.get("output")
            if not isinstance(output, Mapping):
                raise RemotePassageScorerError("completed Runpod job has invalid output")
            return output
        if status in _FAILED_JOB_STATES:
            raise RemotePassageScorerError(f"Runpod queue job ended with status {status}")
        if status not in _ACTIVE_JOB_STATES:
            raise RemotePassageScorerError("Runpod queue job returned an unknown status")
        return None

    def run(self, request: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(request, Mapping) or not request:
            raise ValueError("Runpod queue request must be a non-empty mapping")
        deadline = self._monotonic() + self.timeout_seconds
        response = self._request(
            method="POST",
            url=f"{self._base_url}/run",
            payload={
                "input": {"request": dict(request)},
                "policy": {"executionTimeout": self.execution_timeout_ms},
            },
            deadline=deadline,
            retry_ambiguous_failures=False,
        )
        output = self._completed_output(response)
        if output is not None:
            return output
        job_id = response.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RemotePassageScorerError("Runpod queue submission omitted the job identity")
        while True:
            if self.poll_interval_seconds:
                self._sleep(self.poll_interval_seconds)
            response = self._request(
                method="GET",
                url=f"{self._base_url}/status/{job_id}",
                payload=None,
                deadline=deadline,
                retry_ambiguous_failures=True,
            )
            output = self._completed_output(response)
            if output is not None:
                return output


class RunpodFlashPassagePredictor:
    """Validate and score one ordered query/passage batch through Runpod."""

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        *,
        request_batch_size: int = MAX_REQUEST_PASSAGES,
        timeout_seconds: int = 900,
        max_retries: int = 3,
        job_client: QueueJobClient | None = None,
    ) -> None:
        self.request_batch_size = _request_batch_size(request_batch_size)
        if job_client is None:
            self._job_client: QueueJobClient = RunpodQueueJobClient(
                endpoint_id,
                api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        else:
            if not callable(getattr(job_client, "run", None)):
                raise TypeError("job_client must expose a callable run method")
            self._job_client = job_client

    @property
    def cache_context(self) -> ScoreCacheContext:
        return remote_score_cache_context()

    @property
    def identity(self) -> dict[str, object]:
        return remote_scorer_identity(self.request_batch_size)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        request, content_ids = _build_score_request(
            pairs,
            request_batch_size=self.request_batch_size,
        )
        if not content_ids:
            return ()
        response = self._job_client.run(request)
        return _validate_score_response(
            response,
            request=request,
            content_ids=content_ids,
        )


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "INPUT_POLICY",
    "MAX_LENGTH",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_PASSAGES",
    "MIXEDBREAD_MODEL",
    "MIXEDBREAD_REVISION",
    "REMOTE_BACKEND",
    "REMOTE_DEVICE_FAMILY",
    "REMOTE_MODEL_BATCH_SIZE",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "RemotePassageScorerError",
    "RunpodFlashPassagePredictor",
    "RunpodQueueJobClient",
    "remote_endpoint_identity",
    "remote_endpoint_identity_sha256",
    "remote_score_cache_context",
    "remote_scorer_identity",
]

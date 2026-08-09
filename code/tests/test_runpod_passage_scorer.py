from __future__ import annotations

from collections.abc import Mapping
from email.message import Message
import io
import math
from urllib.error import HTTPError

import pytest

from trec_rag.runpod_passage_scorer import (
    MAX_REQUEST_BYTES,
    MAX_REQUEST_PASSAGES,
    REMOTE_DEVICE_FAMILY,
    RemotePassageScorerError,
    RunpodFlashPassagePredictor,
    RunpodQueueJobClient,
    remote_endpoint_identity,
    remote_endpoint_identity_sha256,
    remote_score_cache_context,
    remote_scorer_identity,
)


class EchoCompletedJobClient:
    """Return a complete endpoint-shaped response for the received request."""

    def __init__(
        self,
        scores: tuple[object, ...],
        *,
        reverse_results: bool = False,
        identity: Mapping[str, object] | None = None,
    ) -> None:
        self.scores = scores
        self.reverse_results = reverse_results
        self.identity = dict(identity or remote_endpoint_identity())
        self.requests: list[dict[str, object]] = []

    def run(self, request: Mapping[str, object]) -> Mapping[str, object]:
        copied = dict(request)
        self.requests.append(copied)
        passages = copied["passages"]
        assert isinstance(passages, list)
        results = [
            {"content_id": passage["content_id"], "score": score}
            for passage, score in zip(passages, self.scores, strict=True)
        ]
        if self.reverse_results:
            results.reverse()
        return {
            "schema_version": "runpod_passage_score_response_v1",
            "request_id": copied["request_id"],
            "identity": self.identity,
            "identity_sha256": remote_endpoint_identity_sha256(self.identity),
            "results": results,
            "diagnostics": {
                "passage_count": len(results),
                "model_batches": int(bool(results)),
            },
        }


class ScriptedTransport:
    def __init__(self, responses: list[Mapping[str, object] | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object] | None,
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _predictor(
    scores: tuple[object, ...],
    *,
    request_batch_size: int = 256,
    reverse_results: bool = False,
    identity: Mapping[str, object] | None = None,
) -> tuple[RunpodFlashPassagePredictor, EchoCompletedJobClient]:
    jobs = EchoCompletedJobClient(
        scores,
        reverse_results=reverse_results,
        identity=identity,
    )
    predictor = RunpodFlashPassagePredictor(
        endpoint_id="endpoint-1",
        api_key="secret-value",
        request_batch_size=request_batch_size,
        timeout_seconds=30,
        max_retries=2,
        job_client=jobs,
    )
    return predictor, jobs


def test_predictor_preserves_order_and_uses_content_addressed_request_identity() -> None:
    predictor, jobs = _predictor((0.75, -0.25), request_batch_size=2)

    scores = predictor.predict((("query", "first"), ("query", "second")))

    assert scores == (0.75, -0.25)
    request = jobs.requests[0]
    assert request["query"] == "query"
    assert request["expected_identity_sha256"] == remote_endpoint_identity_sha256()
    passages = request["passages"]
    assert isinstance(passages, list)
    assert [row["text"] for row in passages] == ["first", "second"]
    assert len({row["content_id"] for row in passages}) == 2
    assert isinstance(request["request_id"], str)
    assert len(request["request_id"]) == 64


def test_predictor_rejects_reordered_result_identities() -> None:
    predictor, _jobs = _predictor((0.75, -0.25), reverse_results=True)

    with pytest.raises(RemotePassageScorerError, match="result identities"):
        predictor.predict((("query", "first"), ("query", "second")))


def test_predictor_rejects_endpoint_identity_mismatch() -> None:
    wrong_identity = remote_endpoint_identity()
    wrong_identity["device_family"] = "unexpected-gpu"
    predictor, _jobs = _predictor((0.75,), identity=wrong_identity)

    with pytest.raises(RemotePassageScorerError, match="identity"):
        predictor.predict((("query", "first"),))


@pytest.mark.parametrize("score", [True, math.nan, math.inf, "0.2"])
def test_predictor_rejects_invalid_scores(score: object) -> None:
    predictor, _jobs = _predictor((score,))

    with pytest.raises(RemotePassageScorerError, match="finite real"):
        predictor.predict((("query", "passage"),))


def test_predictor_rejects_mixed_queries_and_request_count_ceiling() -> None:
    predictor, _jobs = _predictor((), request_batch_size=MAX_REQUEST_PASSAGES)

    with pytest.raises(ValueError, match="one query"):
        predictor.predict((("first query", "passage"), ("second query", "passage")))
    with pytest.raises(ValueError, match="at most 256"):
        predictor.predict(
            tuple(("query", f"passage-{index}") for index in range(MAX_REQUEST_PASSAGES + 1))
        )


def test_predictor_rejects_request_larger_than_six_mibibytes() -> None:
    predictor, _jobs = _predictor((0.0,))

    with pytest.raises(ValueError, match="6 MiB"):
        predictor.predict((("query", "x" * MAX_REQUEST_BYTES),))


def test_remote_cache_context_is_cuda_specific_and_excludes_request_batch_size() -> None:
    context = remote_score_cache_context()

    assert context.device_family == REMOTE_DEVICE_FAMILY
    assert context.fixed_batch_policy == "cross-encoder-predict-batch-size-16"
    assert context.context_sha256 == remote_score_cache_context().context_sha256
    first = remote_scorer_identity(64)
    second = remote_scorer_identity(256)
    assert first["request_batch_size"] == 64
    assert second["request_batch_size"] == 256
    assert first["cache_context_sha256"] == second["cache_context_sha256"]


def test_queue_client_wraps_input_and_polls_until_completed() -> None:
    transport = ScriptedTransport(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"id": "job-1", "status": "COMPLETED", "output": {"ok": True}},
        ]
    )
    client = RunpodQueueJobClient(
        "endpoint-1",
        "secret-value",
        timeout_seconds=30,
        max_retries=2,
        poll_interval_seconds=0,
        transport=transport,
        sleep=lambda _seconds: None,
    )

    output = client.run({"schema_version": "runpod_passage_score_request_v1"})

    assert output == {"ok": True}
    assert transport.calls[0]["method"] == "POST"
    assert transport.calls[0]["url"] == "https://api.runpod.ai/v2/endpoint-1/run"
    assert transport.calls[0]["payload"] == {
        "input": {"request": {"schema_version": "runpod_passage_score_request_v1"}}
    }
    assert transport.calls[1]["method"] == "GET"
    assert transport.calls[1]["url"] == (
        "https://api.runpod.ai/v2/endpoint-1/status/job-1"
    )


def test_queue_client_retries_http_429_without_exposing_body_or_key() -> None:
    headers = Message()
    headers["Retry-After"] = "0"
    failure = HTTPError(
        "https://api.runpod.ai/v2/endpoint-1/run",
        429,
        "rate limited",
        headers,
        io.BytesIO(b"private response body"),
    )
    transport = ScriptedTransport(
        [
            failure,
            {"id": "job-1", "status": "IN_QUEUE"},
            {"id": "job-1", "status": "COMPLETED", "output": {"ok": True}},
        ]
    )
    sleeps: list[float] = []
    client = RunpodQueueJobClient(
        "endpoint-1",
        "secret-value",
        timeout_seconds=30,
        max_retries=2,
        poll_interval_seconds=0,
        transport=transport,
        sleep=sleeps.append,
    )

    assert client.run({"request_id": "request-1"}) == {"ok": True}
    assert sleeps == [0.0]
    assert len(transport.calls) == 3


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "TIMED_OUT"])
def test_queue_client_fails_closed_without_exposing_job_error(status: str) -> None:
    transport = ScriptedTransport(
        [{"id": "job-1", "status": status, "error": "private response body"}]
    )
    client = RunpodQueueJobClient(
        "endpoint-1",
        "secret-value",
        timeout_seconds=30,
        max_retries=0,
        poll_interval_seconds=0,
        transport=transport,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RemotePassageScorerError, match=status) as captured:
        client.run({"request_id": "request-1"})
    message = str(captured.value)
    assert "private response body" not in message
    assert "secret-value" not in message

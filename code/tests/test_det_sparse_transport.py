from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest
import trec_rag.det_sparse_transport as transport_module

from trec_rag.det_sparse_config import (
    ANALYZER_FINGERPRINT_SHA256,
    RETRIEVAL_ENDPOINT_URL,
    load_det_sparse_config,
)
from trec_rag.det_sparse_ledger import RetrievalRequest
from trec_rag.det_sparse_transport import (
    FrozenHttpsRetrievalTransport,
    validate_transport_binding,
)
from trec_rag.det_sparse_run import _validate_formal_runtime_boundary


REPO_ROOT = Path(__file__).resolve().parents[2]


class Response:
    def __init__(self):
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"candidates":[]}'

    def getcode(self):
        return 200


def _request(endpoint=RETRIEVAL_ENDPOINT_URL):
    return RetrievalRequest.from_query(
        topic_id="200",
        variant_name="original",
        query_text="privacy controls?",
        index_url=endpoint,
        index_id="climbmix-400b",
        hits=100,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
    )


def test_one_shot_transport_uses_exact_https_origin_query_and_token():
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return Response()

    transport = FrozenHttpsRetrievalTransport(
        api_token="secret-test-token",
        opener=opener,
    )
    result = transport(_request())

    assert result.status == 200
    assert len(calls) == 1
    request, timeout = calls[0]
    parsed = urllib.parse.urlsplit(request.full_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == RETRIEVAL_ENDPOINT_URL
    assert urllib.parse.parse_qs(parsed.query) == {
        "query": ["privacy controls?"],
        "hits": ["100"],
    }
    assert request.get_header("Authorization") == "Bearer secret-test-token"
    assert timeout == 30.0
    validate_transport_binding(transport, RETRIEVAL_ENDPOINT_URL)


def test_transport_rejects_wrong_origin_before_opening():
    calls = []
    transport = FrozenHttpsRetrievalTransport(
        api_token=None,
        opener=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="approved HTTPS endpoint"):
        transport(_request("https://wrong.example/v1/climbmix-400b/search"))
    assert calls == []


def test_default_network_transport_is_hard_closed_before_opener(monkeypatch):
    calls = []

    def opener(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("opener must remain untouched")

    monkeypatch.setattr(transport_module, "_default_open", opener)
    transport = FrozenHttpsRetrievalTransport(api_token=None)
    assert transport.formal_network_ready is True

    with pytest.raises(PermissionError, match="hard-closed"):
        transport(_request())
    assert calls == []


def test_binding_rejects_unattested_callable():
    with pytest.raises(ValueError, match="one-shot HTTPS"):
        validate_transport_binding(lambda _request: None, RETRIEVAL_ENDPOINT_URL)


@pytest.mark.parametrize("mutate_after_init", [False, True])
def test_runtime_boundary_rejects_injected_or_mutated_opener(mutate_after_init):
    config = load_det_sparse_config(REPO_ROOT / "configs" / "det_sparse_v1.yaml")
    if mutate_after_init:
        transport = FrozenHttpsRetrievalTransport(api_token=None)
        transport._opener = lambda *_args, **_kwargs: Response()
    else:
        transport = FrozenHttpsRetrievalTransport(
            api_token=None,
            opener=lambda *_args, **_kwargs: Response(),
        )

    assert transport.formal_network_ready is False
    with pytest.raises(ValueError, match="non-injected HTTPS transport"):
        _validate_formal_runtime_boundary(config, object(), transport)

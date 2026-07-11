"""Single-origin, one-attempt HTTPS transport for an authorized sparse pilot."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from trec_rag.det_sparse_config import RETRIEVAL_ENDPOINT_URL
from trec_rag.det_sparse_ledger import RawTransportResponse, RetrievalRequest


TRANSPORT_VERSION = "det_sparse_https_one_shot_no_redirect_v1"


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _default_open(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout)


class FrozenHttpsRetrievalTransport:
    """Issue one GET to the exact approved HTTPS URL, with no retry/redirect."""

    transport_version = TRANSPORT_VERSION
    endpoint_url = RETRIEVAL_ENDPOINT_URL
    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        *,
        api_token: str | None,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if timeout_seconds != 30.0:
            raise ValueError("formal transport timeout is frozen at 30 seconds")
        if api_token is not None and (not isinstance(api_token, str) or not api_token):
            raise ValueError("api_token must be non-empty text or None")
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds
        self._opener = opener or _default_open

    @property
    def formal_network_ready(self) -> bool:
        return self._opener is _default_open

    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        if self.formal_network_ready:
            raise PermissionError(
                "formal HTTPS transport is hard-closed while the hosted index "
                "revision is unknown"
            )
        identity = request.identity
        if identity.index_url != self.endpoint_url:
            raise ValueError("transport request differs from the approved HTTPS endpoint")
        if identity.index_id != "climbmix-400b" or identity.hits != 100:
            raise ValueError("transport request index/depth differs from frozen protocol")
        query = urllib.parse.urlencode(
            {"query": request.query_text, "hits": str(identity.hits)}
        )
        url = f"{self.endpoint_url}?{query}"
        http_request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self._api_token}"}
                    if self._api_token is not None
                    else {}
                ),
            },
        )
        started = time.monotonic()
        try:
            response = self._opener(http_request, timeout=self._timeout_seconds)
            with response:
                body = response.read()
                status = int(response.getcode())
                headers = dict(response.headers.items())
        except urllib.error.HTTPError as error:
            # HTTPError is still the sole server response. Preserve its exact
            # bytes so the raw-first ledger can reject it after durable capture.
            with error:
                body = error.read()
                status = int(error.code)
                headers = dict(error.headers.items()) if error.headers else {}
        return RawTransportResponse(
            status=status,
            headers=headers,
            body=body,
            elapsed_seconds=time.monotonic() - started,
        )


def validate_transport_binding(transport: object, endpoint_url: str) -> None:
    """Fail before a reservation unless the transport advertises frozen behavior."""

    if endpoint_url != RETRIEVAL_ENDPOINT_URL:
        raise ValueError("executor endpoint differs from the approved HTTPS endpoint")
    if (
        getattr(transport, "transport_version", None) != TRANSPORT_VERSION
        or getattr(transport, "endpoint_url", None) != endpoint_url
        or getattr(transport, "one_shot_no_retry", None) is not True
        or getattr(transport, "redirects_allowed", None) is not False
    ):
        raise ValueError("retrieval transport is not bound to frozen one-shot HTTPS behavior")

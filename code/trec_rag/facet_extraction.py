"""Fail-closed model boundary for experimental narrative facet extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import ipaddress
import json
import os
import re
import socket
import threading
import time
from typing import Protocol
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


__all__ = [
    "FacetRequest",
    "FacetPlanningResult",
    "FacetResponse",
    "GeneratedQueryPlan",
    "MAX_COMPLETION_TOKENS",
    "MAX_RESPONSE_BYTES",
    "OPENROUTER_DEEPSEEK_ALIAS",
    "OPENROUTER_DEEPSEEK_MODEL",
    "OpenRouterDeepSeekFacetBackend",
    "OpenRouterFacetReceipt",
    "OpenAICompatibleFacetBackend",
    "REQUEST_TIMEOUT_SECONDS",
    "StructuredFacetBackend",
    "Subnarrative",
    "extract_facets",
    "parse_generated_query_plan",
    "plan_facet_queries",
    "render_facet_queries",
    "BackendReply",
]


SCHEMA_VERSION = "subnarrative_queries_v1"
PROMPT_VERSION = "subnarrative_query_generator_v1"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_COMPLETION_TOKENS = 4096
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEEPSEEK_ALIAS = "deepseek/deepseek-v4-flash"
OPENROUTER_DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash-20260423"
_HTTP_WORKER_SLOTS = threading.BoundedSemaphore(4)
MAX_SUBNARRATIVES = 8
MAX_BM25_QUERIES = 3
MAX_SUBNARRATIVE_LENGTH = 500
MAX_BM25_QUERY_LENGTH = 500
_QUERY_OPERATOR = re.compile(
    r'\b(?:AND|OR|NOT)\b|\b[A-Za-z][A-Za-z0-9_.-]*:|&&|\|\||[!(){}\[\]^~*?"/\\\\]|(?:^|\s)[+-](?=\S)',
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


@dataclass(frozen=True)
class FacetRequest:
    """One bounded OpenAI-compatible chat-completions request."""

    url: str
    body: bytes
    headers: dict[str, str]
    timeout: float


@dataclass(frozen=True)
class FacetResponse:
    """One received HTTP response, kept only long enough to validate it."""

    status: int
    body: bytes


@dataclass(frozen=True)
class BackendReply:
    """Shared typed reply returned by injected structured-model backends."""

    content: bytes
    response_body: bytes
    status: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class Subnarrative:
    """One generated semantic scope and its ordered BM25 query lanes."""

    topic_id: str
    subnarrative_id: str
    text: str
    bm25_queries: tuple[str, ...]
    semantic_query_sha256: str = field(init=False)
    bm25_query_sha256s: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_query_sha256",
            sha256(self.text.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(
            self,
            "bm25_query_sha256s",
            tuple(sha256(query.encode("utf-8")).hexdigest() for query in self.bm25_queries),
        )


@dataclass(frozen=True)
class GeneratedQueryPlan:
    """Validated model output with Python-derived stable subnarrative IDs."""

    topic_id: str
    subnarratives: tuple[Subnarrative, ...]


@dataclass(frozen=True)
class FacetPlanningResult:
    queries: tuple[QueryVariant, ...]
    used_fallback: bool
    error: str | None
    plan: GeneratedQueryPlan | None = None
    subnarratives: tuple[Subnarrative, ...] = ()


class _PlanValidationError(ValueError):
    """Raised internally when untrusted generated output is unsafe to use."""


def parse_generated_query_plan(payload: object, topic: Topic) -> GeneratedQueryPlan:
    """Parse exactly one strict ``subnarrative_queries_v1`` payload."""
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    root = _mapping(payload, "generated query plan")
    _require_keys(root, {"schema_version", "topic_id", "subnarratives"}, "generated query plan")
    if root["schema_version"] != SCHEMA_VERSION:
        raise _PlanValidationError("generated query plan schema version is invalid")
    if root["topic_id"] != topic.id:
        raise _PlanValidationError("plan topic ID does not match topic ID")

    rows = _list(root["subnarratives"], "subnarratives", 1, MAX_SUBNARRATIVES)
    records: list[Subnarrative] = []
    seen_texts: set[str] = set()
    for index, row in enumerate(rows, start=1):
        item = _mapping(row, f"subnarrative {index}")
        _require_keys(item, {"subnarrative", "bm25_queries"}, f"subnarrative {index}")
        text = _text(item["subnarrative"], "subnarrative", MAX_SUBNARRATIVE_LENGTH)
        text_key = _identity(text)
        if text_key in seen_texts:
            raise _PlanValidationError("subnarratives must be unique")
        seen_texts.add(text_key)
        queries = tuple(
            _bm25_query(value, f"subnarrative {index} BM25 query")
            for value in _list(item["bm25_queries"], "BM25 queries", 1, MAX_BM25_QUERIES)
        )
        seen_queries: set[str] = set()
        for query in queries:
            query_key = _identity(query)
            if query_key in seen_queries:
                raise _PlanValidationError("BM25 queries must be unique")
            seen_queries.add(query_key)
        records.append(Subnarrative(topic.id, f"subnarrative-{index}", text, queries))
    return GeneratedQueryPlan(topic.id, tuple(records))


def render_facet_queries(topic: Topic, plan: GeneratedQueryPlan) -> FacetPlanningResult:
    """Render the original lane followed by the plan's BM25 lanes."""
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    try:
        return _render_canonical_plan(topic, _canonicalize_plan(plan, topic))
    except _PlanValidationError as exc:
        return _fallback(topic, str(exc))


def plan_facet_queries(topic: Topic, payload: object) -> FacetPlanningResult:
    """Parse and render untrusted output, returning original-only fallback on failure."""
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    try:
        plan = parse_generated_query_plan(payload, topic)
        return _render_canonical_plan(topic, plan)
    except _PlanValidationError as exc:
        return _fallback(topic, str(exc))


def _canonicalize_plan(plan: GeneratedQueryPlan, topic: Topic) -> GeneratedQueryPlan:
    """Re-parse public plan objects so direct construction cannot bypass admission."""
    if not isinstance(plan, GeneratedQueryPlan):
        raise _PlanValidationError("plan must be a GeneratedQueryPlan")
    if not isinstance(plan.subnarratives, tuple):
        raise _PlanValidationError("plan subnarratives must be canonical")

    rows: list[dict[str, object]] = []
    for subnarrative in plan.subnarratives:
        if not isinstance(subnarrative, Subnarrative) or not isinstance(subnarrative.bm25_queries, tuple):
            raise _PlanValidationError("plan subnarratives must be canonical")
        rows.append(
            {
                "subnarrative": subnarrative.text,
                "bm25_queries": list(subnarrative.bm25_queries),
            }
        )
    canonical = parse_generated_query_plan(
        {
            "schema_version": SCHEMA_VERSION,
            "topic_id": plan.topic_id,
            "subnarratives": rows,
        },
        topic,
    )
    if plan != canonical:
        raise _PlanValidationError("plan is not canonical")
    return canonical


def _render_canonical_plan(topic: Topic, plan: GeneratedQueryPlan) -> FacetPlanningResult:
    queries = [QueryVariant(topic.id, "original", topic.narrative, "original_topic")]
    for subnarrative in plan.subnarratives:
        for index, query in enumerate(subnarrative.bm25_queries, start=1):
            queries.append(
                QueryVariant(
                    topic.id,
                    f"facet:{subnarrative.subnarrative_id}:q{index}",
                    query,
                    "generated_subnarrative_bm25",
                )
            )
    return FacetPlanningResult(tuple(queries), False, None, plan, plan.subnarratives)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _PlanValidationError(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, object], required: set[str], name: str) -> None:
    if set(value) != required:
        raise _PlanValidationError(f"{name} has unexpected or missing fields")


def _list(value: object, name: str, minimum: int, maximum: int) -> list[object]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise _PlanValidationError(f"{name} must contain between {minimum} and {maximum} items")
    return value


def _text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise _PlanValidationError(f"{name} must be non-empty and bounded")
    if value != value.strip():
        raise _PlanValidationError(f"{name} must not contain surrounding whitespace")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise _PlanValidationError(f"{name} contains control characters")
    return value


def _bm25_query(value: object, name: str) -> str:
    query = _text(value, name, MAX_BM25_QUERY_LENGTH)
    if _QUERY_OPERATOR.search(query):
        raise _PlanValidationError(f"{name} contains query operators")
    return query


def _identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


@dataclass(frozen=True)
class OpenRouterFacetReceipt:
    """Secret-free inputs and outcomes for an injectable local receipt writer."""

    request_body: bytes
    response_body: bytes | None
    response_sha256: str | None
    status: int | None
    started_at: str
    finished_at: str
    latency_seconds: float
    requested_model: str
    public_alias: str
    prompt_version: str
    response_model: str | None
    provider: str | None
    finish_reason: str | None
    usage: Mapping[str, object] | None
    validation_outcome: str
    error: str | None


@dataclass(frozen=True)
class _OpenRouterCompletion:
    content: str
    response_model: str
    provider: str | None
    finish_reason: str
    usage: Mapping[str, object]


class StructuredFacetBackend(Protocol):
    """A source of one untrusted structured plan for a topic narrative."""

    def extract(self, topic: Topic) -> object: ...


class OpenRouterDeepSeekFacetBackend:
    """Fixed, bounded OpenRouter route for experimental DeepSeek extraction."""

    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: object | None = None,
        receipt_hook: object | None = None,
    ) -> None:
        environment = os.environ if environ is None else environ
        api_key = environment.get("OPENROUTER_API_KEY")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set to non-empty text")
        if receipt_hook is not None and not callable(receipt_hook):
            raise TypeError("receipt_hook must be callable")
        self._api_key = api_key
        self._transport = transport if transport is not None else _UrllibFacetTransport()
        self._receipt_hook = receipt_hook

    def extract(self, topic: Topic) -> object:
        if not isinstance(topic, Topic):
            raise TypeError("topic must be a Topic")
        payload = _openrouter_request_payload(topic)
        request = FacetRequest(
            url=_chat_completions_url(OPENROUTER_BASE_URL),
            body=json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        response: FacetResponse | None = None
        completion: _OpenRouterCompletion | None = None
        stage = "transport"
        try:
            response = self._send(request)
            stage = "envelope"
            if not 200 <= response.status < 300:
                raise ValueError(f"OpenRouter returned HTTP {response.status}")
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise ValueError("OpenRouter response exceeded byte cap")
            envelope = _decode_json(response.body, "OpenRouter chat-completions envelope")
            completion = _openrouter_completion(envelope)
            stage = "semantic"
            payload = _decode_json(
                completion.content.encode("utf-8"), "generated query plan"
            )
            validated = plan_facet_queries(topic, payload)
            if validated.used_fallback:
                raise ValueError(validated.error or "generated query plan is invalid")
        except Exception as exc:
            sanitized_error = _redact_api_key(str(exc), self._api_key)
            self._emit_receipt(
                _openrouter_receipt(
                    request=request,
                    response=response,
                    completion=completion,
                    started_at=started_at,
                    started=started,
                    validation_outcome=f"{stage}_error",
                    error=sanitized_error,
                    api_key=self._api_key,
                )
            )
            if sanitized_error == str(exc):
                raise
            raise RuntimeError(sanitized_error) from None
        self._emit_receipt(
            _openrouter_receipt(
                request=request,
                response=response,
                completion=completion,
                started_at=started_at,
                started=started,
                validation_outcome="valid",
                error=None,
                api_key=self._api_key,
            )
        )
        return payload

    def _send(self, request: FacetRequest) -> FacetResponse:
        send = getattr(self._transport, "send", None)
        if not callable(send):
            raise TypeError("transport must provide send(request)")
        response = send(request)
        if not isinstance(response, FacetResponse):
            raise TypeError("transport must return a FacetResponse")
        return response

    def _emit_receipt(self, receipt: OpenRouterFacetReceipt) -> None:
        if self._receipt_hook is not None:
            self._receipt_hook(receipt)


class OpenAICompatibleFacetBackend:
    """Minimal OpenAI chat-completions adapter with strict response handling."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: object | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be explicit non-empty text")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be explicit non-empty text")
        self._endpoint_url = _chat_completions_url(base_url)
        if api_key and not _credentialed_endpoint_is_safe(self._endpoint_url):
            raise ValueError("credentials require HTTPS or a loopback HTTP host")
        self._model = model
        self._api_key = api_key
        self._transport = transport if transport is not None else _UrllibFacetTransport()

    def extract(self, topic: Topic) -> object:
        if not isinstance(topic, Topic):
            raise TypeError("topic must be a Topic")
        payload = _request_payload(topic, self._model)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = FacetRequest(
            url=self._endpoint_url,
            body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response = self._send(request)
        if not 200 <= response.status < 300:
            raise ValueError(f"facet backend returned HTTP {response.status}")
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise ValueError("facet backend response exceeded byte cap")
        envelope = _decode_json(response.body, "chat-completions envelope")
        content = _response_content(envelope)
        return _decode_json(content.encode("utf-8"), "facet plan")

    def _send(self, request: FacetRequest) -> FacetResponse:
        send = getattr(self._transport, "send", None)
        if not callable(send):
            raise TypeError("transport must provide send(request)")
        response = send(request)
        if not isinstance(response, FacetResponse):
            raise TypeError("transport must return a FacetResponse")
        return response


def _chat_completions_url(base_url: str) -> str:
    if base_url != base_url.strip() or any(
        character.isspace() or ord(character) < 32 or 127 <= ord(character) <= 159
        for character in base_url
    ):
        raise ValueError("base_url must not contain raw whitespace or control characters")
    if "?" in base_url or "#" in base_url:
        raise ValueError("base_url must not contain a query or fragment")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url is not a valid absolute URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url scheme must be http or https")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("base_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not include credentials")
    if parsed.netloc.endswith(":") and port is None:
        raise ValueError("base_url contains an invalid port")
    path = parsed.path.rstrip("/") + "/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _credentialed_endpoint_is_safe(endpoint_url: str) -> bool:
    parsed = urllib.parse.urlsplit(endpoint_url)
    if parsed.scheme == "https":
        return True
    host = parsed.hostname
    if parsed.scheme != "http" or host is None:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def extract_facets(topic: Topic, backend: StructuredFacetBackend) -> FacetPlanningResult:
    """Return validated facets, or preserve exactly the original query on failure."""
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    try:
        payload = backend.extract(topic)
    except Exception as exc:
        message = str(exc)
        return _fallback(topic, message if message.strip() else type(exc).__name__)
    return plan_facet_queries(topic, payload)


def _fallback(topic: Topic, error: str) -> FacetPlanningResult:
    return FacetPlanningResult(
        (QueryVariant(topic.id, "original", topic.narrative, "original_topic"),), True, error
    )


class _UrllibFacetTransport:
    def send(self, request: FacetRequest) -> FacetResponse:
        deadline = time.monotonic() + request.timeout
        return _call_before_deadline(
            lambda: self._send_before_deadline(request, deadline), deadline
        )

    @staticmethod
    def _send_before_deadline(
        request: FacetRequest,
        deadline: float,
    ) -> FacetResponse:
        raw_request = urllib.request.Request(
            request.url, data=request.body, headers=request.headers, method="POST"
        )
        try:
            with _NO_REDIRECT_OPENER.open(raw_request, timeout=_remaining(deadline)) as response:
                return FacetResponse(
                    status=int(response.status),
                    body=_read_bounded_body(response, deadline),
                )
        except urllib.error.HTTPError as exc:
            with exc:
                return FacetResponse(
                    status=exc.code,
                    body=_read_bounded_body(exc, deadline),
                )


def _call_before_deadline(call: object, deadline: float) -> FacetResponse:
    """Bound urllib's connect/header work, whose socket timeout is per read."""
    if not callable(call):
        raise TypeError("deadline call must be callable")
    worker_slots = _HTTP_WORKER_SLOTS
    if not worker_slots.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("facet backend request exceeded total deadline")
    if time.monotonic() >= deadline:
        worker_slots.release()
        raise TimeoutError("facet backend request exceeded total deadline")
    condition = threading.Condition()
    state: dict[str, object] = {"accepting": True}

    def run() -> None:
        try:
            outcome: object = call()
        except BaseException as exc:
            outcome = exc
        with condition:
            if state["accepting"]:
                state["outcome"] = outcome
                condition.notify()
        worker_slots.release()

    try:
        threading.Thread(target=run, name="facet-http-request", daemon=True).start()
    except BaseException:
        worker_slots.release()
        raise
    with condition:
        while "outcome" not in state:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state["accepting"] = False
                raise TimeoutError("facet backend request exceeded total deadline")
            condition.wait(remaining)
        state["accepting"] = False
        outcome = state["outcome"]
        if time.monotonic() >= deadline:
            raise TimeoutError("facet backend request exceeded total deadline")
    if isinstance(outcome, BaseException):
        raise outcome
    if not isinstance(outcome, FacetResponse):
        raise TypeError("deadline call must return a FacetResponse")
    return outcome

def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("facet backend request exceeded total deadline")
    return remaining


def _read_bounded_body(response: object, deadline: float) -> bytes:
    response_socket = _response_socket(response)
    read = getattr(response, "read1", None)
    if response_socket is None or not callable(read):
        raise RuntimeError("facet backend response cannot enforce total deadline")
    chunks: list[bytes] = []
    size = 0
    while size <= MAX_RESPONSE_BYTES:
        remaining = _remaining(deadline)
        response_socket.settimeout(remaining)
        try:
            chunk = read(min(65_536, MAX_RESPONSE_BYTES + 1 - size))
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError("facet backend request exceeded total deadline") from exc
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def _response_socket(response: object) -> object | None:
    candidates = [response]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        possible_socket = getattr(candidate, "_sock", None)
        if callable(getattr(possible_socket, "settimeout", None)):
            return possible_socket
        for attribute in ("fp", "raw"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                candidates.append(nested)
    return None


def _decode_json(body: bytes, label: str) -> object:
    try:
        decoded = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} was not valid strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _openrouter_completion(envelope: object) -> _OpenRouterCompletion:
    envelope = _strict_object(
        envelope,
        "OpenRouter chat-completions envelope",
        required={"id", "object", "created", "model", "choices", "usage"},
        optional={
            "provider",
            "service_tier",
            "system_fingerprint",
            "openrouter_metadata",
        },
    )
    if not isinstance(envelope["id"], str) or not envelope["id"]:
        raise ValueError("OpenRouter chat-completions envelope.id must be non-empty text")
    if envelope["object"] != "chat.completion":
        raise ValueError("OpenRouter chat-completions envelope.object is invalid")
    if type(envelope["created"]) is not int:
        raise ValueError("OpenRouter chat-completions envelope.created must be an integer")
    response_model = envelope["model"]
    if not isinstance(response_model, str) or not response_model:
        raise ValueError("OpenRouter chat-completions envelope.model must be non-empty text")
    _nullable_fields(envelope, ("service_tier", "system_fingerprint"), str)
    provider = envelope.get("provider")
    if provider is not None and (not isinstance(provider, str) or not provider):
        raise ValueError("OpenRouter chat-completions envelope.provider has an invalid type")
    metadata = envelope.get("openrouter_metadata")
    if metadata is not None:
        _validate_openrouter_metadata(metadata)
        if provider is None:
            provider = _selected_openrouter_provider(metadata)
    usage = _openrouter_usage(envelope["usage"])
    choices = envelope["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("OpenRouter chat-completions envelope must contain one choice")
    choice = _strict_object(
        choices[0],
        "OpenRouter chat-completions choice",
        required={"index", "message", "finish_reason"},
        optional={"logprobs", "native_finish_reason"},
    )
    if type(choice["index"]) is not int:
        raise ValueError("OpenRouter chat-completions choice.index must be an integer")
    _nullable_fields(choice, ("logprobs",), Mapping)
    _nullable_fields(choice, ("native_finish_reason",), str)
    finish_reason = choice["finish_reason"]
    if finish_reason != "stop":
        raise ValueError("OpenRouter chat-completions response did not finish with stop")
    message = _strict_object(
        choice["message"],
        "OpenRouter chat-completions message",
        required={"role", "content"},
        optional={
            "annotations",
            "audio",
            "images",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "refusal",
            "tool_calls",
        },
    )
    if message["role"] != "assistant":
        raise ValueError("OpenRouter response did not contain an assistant message")
    _nullable_fields(message, ("refusal", "reasoning", "reasoning_content"), str)
    _nullable_fields(
        message,
        ("annotations", "images", "reasoning_details", "tool_calls"),
        list,
    )
    _nullable_fields(message, ("audio",), Mapping)
    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter response content was empty")
    return _OpenRouterCompletion(
        content=content,
        response_model=response_model,
        provider=provider,
        finish_reason=finish_reason,
        usage=usage,
    )


def _openrouter_usage(value: object) -> Mapping[str, object]:
    usage = _strict_object(
        value,
        "OpenRouter usage",
        required={"prompt_tokens", "completion_tokens", "total_tokens"},
        optional={
            "completion_tokens_details",
            "cost",
            "cost_details",
            "is_byok",
            "prompt_tokens_details",
            "server_tool_use_details",
        },
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage[key]) is not int or usage[key] < 0:
            raise ValueError(f"OpenRouter usage.{key} must be a non-negative integer")
    if "cost" in usage and usage["cost"] is not None and (
        isinstance(usage["cost"], bool)
        or not isinstance(usage["cost"], (int, float))
    ):
        raise ValueError("OpenRouter usage.cost must be a number or null")
    if "is_byok" in usage and usage["is_byok"] is not None and type(
        usage["is_byok"]
    ) is not bool:
        raise ValueError("OpenRouter usage.is_byok must be a boolean or null")
    _nullable_fields(
        usage,
        (
            "completion_tokens_details",
            "cost_details",
            "prompt_tokens_details",
            "server_tool_use_details",
        ),
        Mapping,
    )
    return usage


def _validate_openrouter_metadata(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("OpenRouter metadata must be an object")
    _nullable_fields(value, ("requested", "strategy", "region", "summary"), str)
    if "attempt" in value and type(value["attempt"]) is not int:
        raise ValueError("OpenRouter metadata.attempt must be an integer")
    if "is_byok" in value and type(value["is_byok"]) is not bool:
        raise ValueError("OpenRouter metadata.is_byok must be a boolean")
    _nullable_fields(value, ("endpoints", "params"), Mapping)
    _nullable_fields(value, ("attempts", "pipeline"), list)


def _selected_openrouter_provider(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    endpoints = value.get("endpoints")
    if not isinstance(endpoints, Mapping):
        return None
    available = endpoints.get("available")
    if not isinstance(available, list):
        return None
    for endpoint in available:
        if not isinstance(endpoint, Mapping) or endpoint.get("selected") is not True:
            continue
        provider = endpoint.get("provider")
        if isinstance(provider, str) and provider:
            return provider
    return None


def _openrouter_receipt(
    *,
    request: FacetRequest,
    response: FacetResponse | None,
    completion: _OpenRouterCompletion | None,
    started_at: str,
    started: float,
    validation_outcome: str,
    error: str | None,
    api_key: str,
) -> OpenRouterFacetReceipt:
    response_body = None if response is None else response.body
    api_key_bytes = api_key.encode("utf-8")
    reflected_credential = (
        response_body is not None
        and (
            api_key_bytes in response_body
            or _raw_json_escapes_contain_credential(response_body, api_key)
            or _decoded_json_contains_credential(response_body, api_key)
        )
    )
    return OpenRouterFacetReceipt(
        request_body=_redact_api_key(request.body, api_key),
        response_body=(
            None
            if reflected_credential
            else response_body
        ),
        response_sha256=(
            None if response_body is None else sha256(response_body).hexdigest()
        ),
        status=None if response is None else response.status,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        latency_seconds=max(0.0, time.monotonic() - started),
        requested_model=OPENROUTER_DEEPSEEK_MODEL,
        public_alias=OPENROUTER_DEEPSEEK_ALIAS,
        prompt_version=PROMPT_VERSION,
        response_model=(
            None
            if completion is None
            else _redact_api_key(completion.response_model, api_key)
        ),
        provider=(
            None if completion is None else _redact_api_key(completion.provider, api_key)
        ),
        finish_reason=(
            None
            if completion is None
            else _redact_api_key(completion.finish_reason, api_key)
        ),
        usage=(
            None if completion is None else _redact_api_key(completion.usage, api_key)
        ),
        validation_outcome=validation_outcome,
        error=_redact_api_key(error, api_key),
    )


def _decoded_json_contains_credential(body: bytes, credential: str) -> bool:
    """Detect reflected credentials after JSON escape decoding without masking errors."""
    try:
        decoded = _decode_json(body, "OpenRouter credential-reflection envelope")
    except ValueError:
        return False
    return _contains_credential(decoded, credential)


def _raw_json_escapes_contain_credential(body: bytes, credential: str) -> bool:
    """Conservatively scan JSON Unicode escapes even when the body is malformed."""

    def decode_escape(match: re.Match[bytes]) -> bytes:
        codepoint = int(match.group(1), 16)
        try:
            return chr(codepoint).encode("utf-8")
        except UnicodeEncodeError:
            return b""

    normalized = re.sub(rb"\\u([0-9a-fA-F]{4})", decode_escape, body)
    return credential.encode("utf-8") in normalized


def _contains_credential(value: object, credential: str) -> bool:
    if isinstance(value, str):
        return credential in value
    if isinstance(value, Mapping):
        return any(
            _contains_credential(key, credential)
            or _contains_credential(item, credential)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_credential(item, credential) for item in value)
    return False


def _redact_api_key(value: object, api_key: str) -> object:
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]")
    if isinstance(value, bytes):
        return value.replace(api_key.encode("utf-8"), b"[REDACTED]")
    if isinstance(value, Mapping):
        return {
            _redact_api_key(key, api_key): _redact_api_key(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_api_key(item, api_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_api_key(item, api_key) for item in value)
    return value


def _response_content(envelope: object) -> str:
    envelope = _strict_object(
        envelope,
        "chat-completions envelope",
        required={"id", "object", "created", "model", "choices"},
        optional={
            "service_tier",
            "system_fingerprint",
            "usage",
            "prompt_logprobs",
            "prompt_token_ids",
            "prompt_text",
            "kv_transfer_params",
        },
    )
    if not isinstance(envelope["id"], str) or not envelope["id"]:
        raise ValueError("chat-completions envelope.id must be non-empty text")
    if envelope["object"] != "chat.completion":
        raise ValueError("chat-completions envelope.object is invalid")
    if type(envelope["created"]) is not int:
        raise ValueError("chat-completions envelope.created must be an integer")
    if not isinstance(envelope["model"], str) or not envelope["model"]:
        raise ValueError("chat-completions envelope.model must be non-empty text")
    _nullable_fields(envelope, ("service_tier", "system_fingerprint", "prompt_text"), str)
    _nullable_fields(envelope, ("usage", "kv_transfer_params"), Mapping)
    _nullable_fields(envelope, ("prompt_logprobs",), list)
    _nullable_integer_array(envelope, "prompt_token_ids")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("chat-completions envelope must contain one choice")
    choice = _strict_object(
        choices[0],
        "chat-completions choice",
        required={"index", "message", "finish_reason"},
        optional={"logprobs", "stop_reason", "token_ids", "routed_experts"},
    )
    if type(choice["index"]) is not int:
        raise ValueError("chat-completions choice.index must be an integer")
    _nullable_fields(choice, ("logprobs",), Mapping)
    stop_reason = choice.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str) and type(stop_reason) is not int:
        raise ValueError("chat-completions choice.stop_reason has an invalid type")
    _nullable_integer_array(choice, "token_ids")
    _nullable_fields(choice, ("routed_experts",), list)
    if choice["finish_reason"] != "stop":
        raise ValueError("chat-completions response did not finish with stop")
    message = _strict_object(
        choice["message"],
        "chat-completions message",
        required={"role", "content"},
        optional={
            "refusal",
            "annotations",
            "audio",
            "function_call",
            "reasoning",
            "reasoning_content",
            "tool_calls",
        },
    )
    if message["role"] != "assistant":
        raise ValueError("chat-completions response did not contain an assistant message")
    _nullable_fields(message, ("refusal", "reasoning", "reasoning_content"), str)
    _nullable_fields(message, ("annotations", "tool_calls"), list)
    _nullable_fields(message, ("audio", "function_call"), Mapping)
    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat-completions response content was empty")
    return content


def _strict_object(
    value: object,
    label: str,
    *,
    required: set[str],
    optional: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        raise ValueError(f"{label} has unexpected or missing fields")
    return value


def _nullable_fields(
    value: Mapping[str, object],
    keys: tuple[str, ...],
    expected: type[object],
) -> None:
    if any(key in value and value[key] is not None and not isinstance(value[key], expected) for key in keys):
        raise ValueError("chat-completions response field has an invalid type")


def _nullable_integer_array(value: Mapping[str, object], key: str) -> None:
    if key not in value or value[key] is None:
        return
    items = value[key]
    if not isinstance(items, list) or any(type(item) is not int for item in items):
        raise ValueError("chat-completions response integer array has an invalid type")


def _request_payload(topic: Topic, model: str) -> dict[str, object]:
    schema = _json_schema(topic.id)
    return {
        "model": model,
        "messages": [
            {"role": "developer", "content": _developer_prompt(schema)},
            {
                "role": "user",
                "content": _topic_user_message(topic),
            },
        ],
        "max_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_VERSION, "strict": True, "schema": schema},
        },
    }


def _openrouter_request_payload(topic: Topic) -> dict[str, object]:
    schema = _json_schema(topic.id)
    return {
        "model": OPENROUTER_DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _developer_prompt(schema)},
            {"role": "user", "content": _topic_user_message(topic)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_VERSION,
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False},
        "temperature": 0,
        "seed": 0,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }


def _topic_user_message(topic: Topic) -> str:
    return (
        f"<topic_id>{topic.id}</topic_id>\n"
        f"<narrative>\n{topic.narrative}\n</narrative>\n"
    )


def _json_schema(topic_id: str) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "topic_id", "subnarratives"],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "topic_id": {"type": "string", "const": topic_id},
            "subnarratives": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subnarrative", "bm25_queries"],
                    "properties": {
                        "subnarrative": {
                            "type": "string", "minLength": 1, "maxLength": 500
                        },
                        "bm25_queries": {
                            "type": "array", "minItems": 1, "maxItems": 3,
                            "items": {
                                "type": "string", "minLength": 1, "maxLength": 500
                            },
                        },
                    },
                },
            },
        },
    }


_DEVELOPER_PROMPT = """# Generated subnarrative query task

Plan retrieval over a very large passage collection using only the supplied
official topic narrative. Do not answer the topic. Return only the
schema-constrained JSON plan.

Rules:
1. Generate one to eight concise subnarratives that together preserve the
   narrative's requested subjects, relations, comparisons, constraints, and
   uncertainty. A subnarrative may paraphrase or synthesize the information need.
2. Give each subnarrative one to three plain BM25 queries. Queries must be useful
   lexical alternatives, not answers, and must not contain query operators or
   field syntax.
3. Use only the supplied narrative. Do not return identifiers or quote-location
   bookkeeping. Python derives stable identifiers from the returned order.
"""


def _developer_prompt(schema: Mapping[str, object]) -> str:
    return (
        f"# Prompt version: {PROMPT_VERSION}\n\n"
        + _DEVELOPER_PROMPT
        + "\n# Exact response schema\n\n"
        + json.dumps(
            schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    )

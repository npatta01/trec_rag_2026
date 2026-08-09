"""Scale-to-zero Runpod Flash worker for strict Mixedbread passage scoring."""

from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume


model_cache = NetworkVolume(
    name="trec-rag-mixedbread-model-cache-v1",
    size=50,
    datacenter=DataCenter.US_GA_2,
)


@Endpoint(
    name="trec-rag-mixedbread-passage-scorer-v1",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 3),
    idle_timeout=900,
    max_concurrency=1,
    flashboot=True,
    execution_timeout_ms=900_000,
    datacenter=DataCenter.US_GA_2,
    volume=model_cache,
    env={
        "HF_HUB_CACHE": "/runpod-volume/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    },
    dependencies=[
        "numpy==2.3.3",
        "sentence-transformers==5.6.0",
        "torch==2.9.1",
        "transformers==5.13.0",
    ],
)
async def score_batch(request: dict) -> dict:
    """Validate and score one query with up to 256 ordered passages."""

    import hashlib
    import json
    import math

    request_schema = "runpod_passage_score_request_v1"
    response_schema = "runpod_passage_score_response_v1"
    identity_schema = "runpod_passage_score_identity_v1"
    model_name = "mixedbread-ai/mxbai-rerank-base-v2"
    model_revision = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
    model_batch_size = 16
    max_request_passages = 256
    max_request_bytes = 6 * 1024 * 1024
    endpoint_identity = {
        "schema_version": identity_schema,
        "backend": "runpod-flash-sentence-transformers-cross-encoder",
        "backend_version": "5.6.0",
        "model": model_name,
        "model_revision": model_revision,
        "max_length": 1024,
        "score_kind": "topic_passage_relevance_v1",
        "score_representation": "raw_logits",
        "inference_dtype": "bfloat16",
        "input_policy": "topic_passage_query_text_v1",
        "device_family": "nvidia-geforce-rtx-4090",
        "fixed_batch_policy": "cross-encoder-predict-batch-size-16",
        "model_batch_size": model_batch_size,
        "torch_version": "2.9.1",
        "transformers_version": "5.13.0",
        "implementation_version": 1,
    }

    def canonical_json(value: object) -> bytes:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("request must contain canonical JSON values") from exc

    def sha256_json(value: object) -> str:
        return hashlib.sha256(canonical_json(value)).hexdigest()

    def content_id(query_text: str, passage_text: str) -> str:
        return sha256_json(
            {
                "identity_sha256": sha256_json(endpoint_identity),
                "query_sha256": hashlib.sha256(
                    query_text.encode("utf-8")
                ).hexdigest(),
                "passage_sha256": hashlib.sha256(
                    passage_text.encode("utf-8")
                ).hexdigest(),
            }
        )

    if not isinstance(request, dict):
        raise ValueError("request must be a mapping")
    if len(canonical_json(request)) > max_request_bytes:
        raise ValueError("request must not exceed 6 MiB")
    required_fields = {
        "schema_version",
        "request_id",
        "expected_identity_sha256",
        "query",
        "passages",
    }
    if set(request) != required_fields:
        raise ValueError("request fields differ from the scoring contract")
    if request["schema_version"] != request_schema:
        raise ValueError("request schema differs from the scoring contract")
    expected_identity_sha256 = sha256_json(endpoint_identity)
    if request["expected_identity_sha256"] != expected_identity_sha256:
        raise ValueError("request endpoint identity differs")
    query = request["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("request query must be nonblank text")
    passages = request["passages"]
    if (
        not isinstance(passages, list)
        or not passages
        or len(passages) > max_request_passages
    ):
        raise ValueError("request passages must contain between 1 and 256 rows")
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for passage in passages:
        if not isinstance(passage, dict) or set(passage) != {"content_id", "text"}:
            raise ValueError("request passages differ from the scoring contract")
        selected_content_id = passage["content_id"]
        text = passage["text"]
        if not isinstance(selected_content_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("request passage identity and text must be nonblank")
        if selected_content_id != content_id(query, text):
            raise ValueError("request passage content identity differs")
        if selected_content_id in seen:
            raise ValueError("request passages repeat a content identity")
        seen.add(selected_content_id)
        normalized.append((selected_content_id, text))
    request_without_id = {
        "schema_version": request["schema_version"],
        "expected_identity_sha256": request["expected_identity_sha256"],
        "query": query,
        "passages": passages,
    }
    if request["request_id"] != sha256_json(request_without_id):
        raise ValueError("request_id differs from request content")

    global _MODEL
    try:
        model = _MODEL
    except NameError:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            model_name,
            revision=model_revision,
            max_length=1024,
            device="cuda",
            local_files_only=False,
        )
        parameter = next(model.model.parameters())
        if str(parameter.dtype) != "torch.bfloat16":
            raise RuntimeError("model dtype differs from the scoring identity")
        _MODEL = model
    parameter = next(model.model.parameters())
    if str(parameter.dtype) != "torch.bfloat16":
        raise RuntimeError("model dtype differs from the scoring identity")

    def identity_activation(value):
        return value

    predicted = model.predict(
        [(query, text) for _selected_content_id, text in normalized],
        batch_size=model_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        activation_fn=identity_activation,
    )
    if hasattr(predicted, "detach"):
        predicted = predicted.detach().cpu()
    if hasattr(predicted, "tolist"):
        predicted = predicted.tolist()
    if isinstance(predicted, (int, float)) and not isinstance(predicted, bool):
        predicted = [predicted]
    if not isinstance(predicted, (list, tuple)) or len(predicted) != len(normalized):
        raise ValueError("model score count differs from request passages")
    scores: list[float] = []
    for value in predicted:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("model scores must be finite real numbers")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("model scores must be finite real numbers")
        scores.append(score)
    return {
        "schema_version": response_schema,
        "request_id": request["request_id"],
        "identity": endpoint_identity,
        "identity_sha256": expected_identity_sha256,
        "results": [
            {"content_id": selected_content_id, "score": score}
            for (selected_content_id, _text), score in zip(
                normalized,
                scores,
                strict=True,
            )
        ],
        "diagnostics": {
            "passage_count": len(scores),
            "model_batches": math.ceil(len(scores) / model_batch_size),
        },
    }

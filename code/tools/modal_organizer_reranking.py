"""Run the organizer Qwen pointwise -> FIRST listwise cascade on Modal.

The tracked benchmark YAML is copied into the image and is the only runtime
configuration source. GPU stages are split into fixed, non-retrying calls;
their combined decorator timeouts, CPU staging timeout, and prior experiment
spend are validated to remain below the YAML's ten-dollar ceiling.
"""

from __future__ import annotations

from pathlib import Path
import sys

import modal


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


APP_NAME = "trec-rag-pointwise-listwise-benchmark-v1"
SOURCE_VOLUME_NAME = "trec-rag-raw-logit-hits1000-20260710"
RESULT_VOLUME_NAME = "trec-rag-organizer-rerank-20260801"
HF_VOLUME_NAME = "hf-cache"
CONFIG_LOCAL = REPO_ROOT / "configs" / "rag25_pointwise_listwise_ndcg_v1.yaml"
CONFIG_REMOTE = "/root/rag25_pointwise_listwise_ndcg_v1.yaml"

SOURCE_ROOT = "/source"
RESULT_ROOT = "/results"
HF_ROOT = "/hf-cache"
HF_HUB_ROOT = f"{HF_ROOT}/hub"

SMOKE_TIMEOUT_SECONDS = 10
POINTWISE_TIMEOUT_SECONDS = 10
LISTWISE_TIMEOUT_SECONDS = 400
STAGING_TIMEOUT_SECONDS = 10


app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub==1.22.0",
        "PyYAML==6.0.3",
        "transformers==5.13.0",
        "vllm==0.24.0",
    )
    .add_local_python_source("trec_rag.organizer_reranking", copy=True)
    .add_local_file(str(CONFIG_LOCAL), CONFIG_REMOTE, copy=True)
    .env(
        {
            "HF_HOME": HF_ROOT,
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


def _settings() -> dict[str, object]:
    import yaml

    return yaml.safe_load(Path(CONFIG_REMOTE).read_text(encoding="utf-8"))


def _cost_guard(config: dict[str, object]) -> dict[str, float]:
    budget = config["modal_budget"]
    execution = config["modal_execution"]
    if execution["app_name"] != APP_NAME:
        raise ValueError("Modal app name differs from the tracked config")
    if execution["source_volume"] != SOURCE_VOLUME_NAME:
        raise ValueError("Modal source volume differs from the tracked config")
    if execution["result_volume"] != RESULT_VOLUME_NAME:
        raise ValueError("Modal result volume differs from the tracked config")
    if execution["huggingface_cache_volume"] != HF_VOLUME_NAME:
        raise ValueError("Modal Hugging Face volume differs from the tracked config")
    if int(budget["smoke_timeout_seconds"]) != SMOKE_TIMEOUT_SECONDS:
        raise ValueError("smoke timeout differs from the tracked budget")
    if int(execution["pointwise_timeout_seconds"]) != POINTWISE_TIMEOUT_SECONDS:
        raise ValueError("pointwise timeout differs from the tracked budget")
    if int(execution["listwise_timeout_seconds"]) != LISTWISE_TIMEOUT_SECONDS:
        raise ValueError("listwise timeout differs from the tracked budget")
    if int(budget["staging_timeout_seconds"]) != STAGING_TIMEOUT_SECONDS:
        raise ValueError("staging timeout differs from the tracked budget")

    gpu_rate = (
        float(budget["a100_80gb_usd_per_second"])
        + float(budget["gpu_cpu_cores"]) * float(budget["cpu_usd_per_core_second"])
        + float(budget["gpu_memory_gib"]) * float(budget["memory_usd_per_gib_second"])
    )
    staging_rate = (
        float(budget["staging_cpu_cores"]) * float(budget["cpu_usd_per_core_second"])
        + float(budget["staging_memory_gib"]) * float(budget["memory_usd_per_gib_second"])
    )
    incremental_maximum = gpu_rate * (
        SMOKE_TIMEOUT_SECONDS + POINTWISE_TIMEOUT_SECONDS + LISTWISE_TIMEOUT_SECONDS
    ) + staging_rate * STAGING_TIMEOUT_SECONDS
    prior_spend = float(budget["prior_spend_usd"])
    maximum = prior_spend + incremental_maximum
    if maximum >= float(budget["maximum_usd"]):
        raise ValueError(
            f"worst-case configured Modal cost ${maximum:.2f} is not below "
            f"${float(budget['maximum_usd']):.2f}"
        )
    return {
        "gpu_usd_per_second": gpu_rate,
        "staging_usd_per_second": staging_rate,
        "prior_spend_usd": prior_spend,
        "incremental_worst_case_usd": incremental_maximum,
        "worst_case_usd": maximum,
    }


def _run_root(config: dict[str, object]) -> Path:
    return Path(RESULT_ROOT) / "runs" / str(config["modal_execution"]["run_id"])


def _load_topic_candidates(config: dict[str, object], topic_id: str):
    import json

    from trec_rag.organizer_reranking import (
        RerankCandidate,
        retrieval_document_text,
        retrieval_query_text,
    )

    cache_dir = Path(SOURCE_ROOT) / str(config["modal_execution"]["source_cache_dir"])
    paths = sorted(cache_dir.glob(f"{topic_id}__original__climbmix_bm25__*.json"))
    if len(paths) != 1:
        raise ValueError(f"topic {topic_id}: expected exactly one source cache; found {len(paths)}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if "response" in payload:
        payload = payload["response"]
    query = retrieval_query_text(payload)
    raw_candidates = payload["candidates"]
    expected = int(config["modal_execution"]["expected_candidates_per_topic"])
    if len(raw_candidates) != expected:
        raise ValueError(f"topic {topic_id}: expected {expected} candidates")
    rows = [
        RerankCandidate(
            topic_id=topic_id,
            query_text=query,
            docid=str(row["docid"]),
            bm25_rank=int(row.get("rank") or rank),
            bm25_score=float(row["score"]),
            text=retrieval_document_text(row["doc"]),
        )
        for rank, row in enumerate(raw_candidates, start=1)
    ]
    if [row.bm25_rank for row in rows] != list(range(1, expected + 1)):
        raise ValueError(f"topic {topic_id}: source ranks are not contiguous")
    if len({row.docid for row in rows}) != expected:
        raise ValueError(f"topic {topic_id}: duplicate source documents")
    return rows


def _topic_ids(config: dict[str, object]) -> list[str]:
    cache_dir = Path(SOURCE_ROOT) / str(config["modal_execution"]["source_cache_dir"])
    topic_ids = sorted(
        {path.name.split("__", 1)[0] for path in cache_dir.glob("*__original__climbmix_bm25__*.json")},
        key=int,
    )
    if len(topic_ids) != int(config["modal_execution"]["expected_topics"]):
        raise ValueError("source cache does not contain the configured topic population")
    return topic_ids


def _build_llm(
    config: dict[str, object],
    *,
    model: str,
    max_model_len: int,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
):
    from vllm import LLM

    execution = config["modal_execution"]
    return LLM(
        model=model,
        download_dir=HF_HUB_ROOT,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=float(execution["vllm_gpu_memory_utilization"]),
        max_num_seqs=max_num_seqs or int(execution["vllm_max_num_seqs"]),
        max_num_batched_tokens=max_num_batched_tokens
        or int(execution["vllm_max_num_batched_tokens"]),
        enable_prefix_caching=True,
        trust_remote_code=False,
        disable_log_stats=True,
    )


def _label_logprobs(output, label_ids: list[int]) -> dict[int, float]:
    values = output.outputs[0].logprobs[0]
    missing = [token_id for token_id in label_ids if token_id not in values]
    if missing:
        raise ValueError(f"vLLM output omitted requested label token ids: {missing}")
    return {token_id: float(values[token_id].logprob) for token_id in label_ids}


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=STAGING_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={HF_ROOT: hf_volume, RESULT_ROOT: result_volume},
)
def stage_models() -> dict[str, object]:
    """Download both released BF16 checkpoints without billing GPU time."""

    import json
    import time

    from huggingface_hub import snapshot_download

    config = _settings()
    costs = _cost_guard(config)
    run_root = _run_root(config)
    receipt_path = run_root / "model_staging_receipt.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    snapshots = {}
    for model in (
        str(config["organizer_pointwise"]["model"]),
        str(config["organizer_listwise"]["model"]),
    ):
        snapshot = Path(snapshot_download(repo_id=model, cache_dir=HF_HUB_ROOT))
        snapshots[model] = {
            "snapshot": str(snapshot),
            "revision": snapshot.name,
        }
        hf_volume.commit()
    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": "organizer-model-staging-v1",
        "snapshots": snapshots,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": elapsed * costs["staging_usd_per_second"],
        "worst_case_total_usd": costs["worst_case_usd"],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_volume.commit()
    return receipt


def _pointwise_inference(*, smoke: bool) -> dict[str, object]:
    import json
    import math
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from trec_rag.organizer_reranking import (
        pointwise_score_row,
        qwen_label_token_ids,
        qwen_pointwise_batch_token_ids,
    )

    config = _settings()
    costs = _cost_guard(config)
    run_root = _run_root(config)
    stage_path = run_root / "model_staging_receipt.json"
    if not stage_path.exists():
        raise RuntimeError("models must be staged before GPU inference")
    stage_receipt = json.loads(stage_path.read_text(encoding="utf-8"))
    pointwise = config["organizer_pointwise"]
    model_name = str(pointwise["model"])
    revision = str(stage_receipt["snapshots"][model_name]["revision"])
    output_path = run_root / (
        "qwen3_reranker_8b_bf16_smoke_scores.jsonl"
        if smoke
        else "qwen3_reranker_8b_bf16_scores.jsonl"
    )
    receipt_path = run_root / ("pointwise_smoke_receipt.json" if smoke else "pointwise_receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to repeat an existing pointwise GPU stage")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if not smoke:
        smoke_receipt = json.loads((run_root / "pointwise_smoke_receipt.json").read_text(encoding="utf-8"))
        if not smoke_receipt.get("approved_for_full_run"):
            raise RuntimeError("pointwise smoke did not approve the full run")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=HF_HUB_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    yes_id, no_id = qwen_label_token_ids(tokenizer)
    # The configured limit applies to the complete reranker prompt. Reserve one
    # additional position for the generated yes/no label.
    llm = _build_llm(
        config,
        model=model_name,
        max_model_len=int(pointwise["max_length"]) + 1,
    )
    model_ready = time.perf_counter()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=[yes_id, no_id],
    )
    topic_ids = _topic_ids(config)
    if smoke:
        topic_ids = [str(config["modal_execution"]["smoke_topic_id"])]
    row_count = 0
    prompt_tokens = 0
    inference_seconds = 0.0
    run_root.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w", encoding="utf-8", newline="\n") as sink:
        for topic_id in topic_ids:
            rows = _load_topic_candidates(config, topic_id)
            if smoke:
                rows = rows[: int(config["modal_execution"]["smoke_candidate_count"])]
            token_ids = qwen_pointwise_batch_token_ids(
                tokenizer,
                query=rows[0].query_text,
                documents=[row.text for row in rows],
                max_length=int(pointwise["max_length"]),
                instruction=str(pointwise["instruction"]),
            )
            infer_started = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": value} for value in token_ids],
                sampling,
                use_tqdm=False,
            )
            inference_seconds += time.perf_counter() - infer_started
            if len(outputs) != len(rows):
                raise ValueError("vLLM pointwise output count changed")
            for candidate, output, input_ids in zip(rows, outputs, token_ids, strict=True):
                logits = _label_logprobs(output, [yes_id, no_id])
                maximum = max(logits.values())
                yes_exp = math.exp(logits[yes_id] - maximum)
                no_exp = math.exp(logits[no_id] - maximum)
                score = yes_exp / (yes_exp + no_exp)
                record = pointwise_score_row(
                    candidate,
                    score=score,
                    model=model_name,
                    model_revision=revision,
                    dtype=str(pointwise["dtype"]),
                    max_length=int(pointwise["max_length"]),
                    prompt_tokens=len(input_ids),
                    backend="vllm-offline-modal-a100-80gb",
                )
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            row_count += len(rows)
            prompt_tokens += sum(map(len, token_ids))
            result_volume.commit()
    partial_path.replace(output_path)
    result_volume.commit()
    elapsed = time.perf_counter() - started
    tokens_per_second = prompt_tokens / inference_seconds
    projected_seconds = float(config["modal_execution"]["expected_total_prompt_tokens"]) / tokens_per_second
    pointwise_timeout = int(config["modal_execution"]["pointwise_timeout_seconds"])
    receipt = {
        "schema_version": "organizer-pointwise-runtime-v1",
        "stage": "smoke" if smoke else "full",
        "model": model_name,
        "model_revision": revision,
        "dtype": pointwise["dtype"],
        "gpu": "A100-80GB",
        "rows": row_count,
        "prompt_tokens": prompt_tokens,
        "model_load_seconds": model_ready - started,
        "inference_seconds": inference_seconds,
        "elapsed_seconds": elapsed,
        "prompt_tokens_per_second": tokens_per_second,
        "projected_full_inference_seconds": projected_seconds,
        "approved_for_full_run": bool(
            not smoke or projected_seconds + (model_ready - started) < pointwise_timeout
        ),
        "estimated_cost_usd": elapsed * costs["gpu_usd_per_second"],
        "worst_case_total_usd": costs["worst_case_usd"],
        "output": output_path.name,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_volume.commit()
    return receipt


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=32768,
    timeout=SMOKE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={SOURCE_ROOT: source_volume, RESULT_ROOT: result_volume, HF_ROOT: hf_volume},
)
def pointwise_smoke() -> dict[str, object]:
    return _pointwise_inference(smoke=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=32768,
    timeout=POINTWISE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={SOURCE_ROOT: source_volume, RESULT_ROOT: result_volume, HF_ROOT: hf_volume},
)
def pointwise_full() -> dict[str, object]:
    return _pointwise_inference(smoke=False)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=32768,
    timeout=LISTWISE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={SOURCE_ROOT: source_volume, RESULT_ROOT: result_volume, HF_ROOT: hf_volume},
)
def listwise_full() -> dict[str, object]:
    """FIRST-rerank the exact Mixedbread top 100 in nine batched rounds."""

    import json
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from trec_rag.organizer_reranking import (
        LISTWISE_SCHEMA,
        LISTWISE_SEED_SCHEMA,
        first_label_token_ids,
        first_permutation_from_logprobs,
        first_prompt_token_ids,
        sha256_canonical_text,
        sha256_text,
        sliding_window_bounds,
    )

    config = _settings()
    costs = _cost_guard(config)
    run_root = _run_root(config)
    stage_path = run_root / "model_staging_receipt.json"
    seed_path = run_root / str(config["modal_execution"]["listwise_seed_filename"])
    if not stage_path.exists() or not seed_path.exists():
        raise RuntimeError("model staging and the Mixedbread top-100 seed are required")
    output_path = run_root / "first_qwen3_8b_bf16_mixedbread_top100.jsonl"
    receipt_path = run_root / "listwise_receipt.json"
    runtime_path = run_root / "runtime_receipt.json"
    if output_path.exists() or receipt_path.exists() or runtime_path.exists():
        raise FileExistsError("refusing to repeat an existing listwise GPU stage")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    topic_ids = _topic_ids(config)
    settings = config["organizer_listwise"]
    candidate_depth = int(settings["candidate_depth"])
    source_by_topic = {topic_id: _load_topic_candidates(config, topic_id) for topic_id in topic_ids}
    source_maps = {
        topic_id: {row.docid: row for row in rows}
        for topic_id, rows in source_by_topic.items()
    }
    seed_rows: dict[str, list[tuple[int, str]]] = {}
    with seed_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if row.get("schema_version") != LISTWISE_SEED_SCHEMA:
                raise ValueError(f"{seed_path}:{line_number}: seed schema mismatch")
            topic_id = str(row["topic_id"])
            docid = str(row["docid"])
            candidate = source_maps.get(topic_id, {}).get(docid)
            if candidate is None:
                raise ValueError(f"{seed_path}:{line_number}: seed candidate is not in BM25")
            if row.get("query_sha256") != sha256_text(candidate.query_text):
                raise ValueError(f"{seed_path}:{line_number}: seed query identity mismatch")
            if row.get("text_sha256") != sha256_canonical_text(candidate.text):
                raise ValueError(f"{seed_path}:{line_number}: seed text identity mismatch")
            seed_rows.setdefault(topic_id, []).append((int(row["rank"]), docid))
    if set(seed_rows) != set(topic_ids):
        raise ValueError("Mixedbread seed topics differ from the retrieval topics")
    orders = {}
    seed_rank = {}
    for topic_id in topic_ids:
        rows = sorted(seed_rows[topic_id])
        if [rank for rank, _docid in rows] != list(range(1, candidate_depth + 1)):
            raise ValueError(f"topic {topic_id}: Mixedbread seed ranks are incomplete")
        docids = [docid for _rank, docid in rows]
        if len(set(docids)) != candidate_depth:
            raise ValueError(f"topic {topic_id}: Mixedbread seed contains duplicates")
        orders[topic_id] = [source_maps[topic_id][docid] for docid in docids]
        seed_rank.update({(topic_id, docid): rank for rank, docid in rows})

    started = time.perf_counter()
    model_name = str(settings["model"])
    stage_receipt = json.loads(stage_path.read_text(encoding="utf-8"))
    revision = str(stage_receipt["snapshots"][model_name]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=HF_HUB_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    label_ids = first_label_token_ids(tokenizer, int(settings["window_size"]))
    # FIRST also emits one label token after its configured prompt context.
    llm = _build_llm(
        config,
        model=model_name,
        max_model_len=int(settings["context_size"]) + 1,
        max_num_seqs=int(settings["vllm_max_num_seqs"]),
        max_num_batched_tokens=int(settings["vllm_max_num_batched_tokens"]),
    )
    model_ready = time.perf_counter()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(label_ids),
        logprob_token_ids=label_ids,
    )
    inference_seconds = 0.0
    prompt_tokens = 0
    passage_words_used: list[int] = []
    for start, end in sliding_window_bounds(
        candidate_depth,
        window_size=int(settings["window_size"]),
        stride=int(settings["stride"]),
    ):
        prompts = []
        for topic_id in topic_ids:
            window = orders[topic_id][start:end]
            prompt, words_used = first_prompt_token_ids(
                tokenizer,
                query=window[0].query_text,
                documents=[row.text for row in window],
                context_size=int(settings["context_size"]),
                max_passage_words=int(settings["max_passage_words"]),
            )
            prompts.append({"prompt_token_ids": prompt})
            passage_words_used.append(words_used)
            prompt_tokens += len(prompt)
        infer_started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        inference_seconds += time.perf_counter() - infer_started
        for topic_id, output in zip(topic_ids, outputs, strict=True):
            window = orders[topic_id][start:end]
            permutation = first_permutation_from_logprobs(
                label_ids,
                _label_logprobs(output, label_ids),
            )
            orders[topic_id][start:end] = [window[index] for index in permutation]

    with partial_path.open("w", encoding="utf-8", newline="\n") as sink:
        for topic_id in topic_ids:
            for rank, row in enumerate(orders[topic_id], start=1):
                sink.write(
                    json.dumps(
                        {
                            "schema_version": LISTWISE_SCHEMA,
                            "topic_id": topic_id,
                            "docid": row.docid,
                            "rank": rank,
                            "seed_rank": seed_rank[(topic_id, row.docid)],
                            "seed_ranker": "mixedbread_coverage_aware_pointwise",
                            "model": model_name,
                            "model_revision": revision,
                            "dtype": settings["dtype"],
                            "candidate_depth": candidate_depth,
                            "window_size": int(settings["window_size"]),
                            "stride": int(settings["stride"]),
                            "context_size": int(settings["context_size"]),
                            "backend": "vllm-offline-modal-a100-80gb-first-token-logits",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    partial_path.replace(output_path)
    result_volume.commit()
    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": "organizer-listwise-runtime-v1",
        "model": model_name,
        "model_revision": revision,
        "dtype": settings["dtype"],
        "gpu": "A100-80GB",
        "topics": len(topic_ids),
        "rows": len(topic_ids) * candidate_depth,
        "windows": len(topic_ids)
        * len(
            sliding_window_bounds(
                candidate_depth,
                window_size=int(settings["window_size"]),
                stride=int(settings["stride"]),
            )
        ),
        "prompt_tokens": prompt_tokens,
        "minimum_passage_words_used": min(passage_words_used),
        "model_load_seconds": model_ready - started,
        "inference_seconds": inference_seconds,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": elapsed * costs["gpu_usd_per_second"],
        "seed": seed_path.name,
        "output": output_path.name,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    smoke_receipt = json.loads((run_root / "pointwise_smoke_receipt.json").read_text(encoding="utf-8"))
    staging_receipt = json.loads(stage_path.read_text(encoding="utf-8"))
    estimated_total = costs["prior_spend_usd"] + sum(
        float(value["estimated_cost_usd"])
        for value in (staging_receipt, smoke_receipt, receipt)
    )
    if estimated_total >= float(config["modal_budget"]["maximum_usd"]):
        raise RuntimeError("measured cost estimate reached the configured hard cap")
    runtime = {
        "schema_version": "organizer-cascade-runtime-v1",
        "staging": staging_receipt,
        "budget_gate_smoke": smoke_receipt,
        "listwise": receipt,
        "estimated_total_cost_usd": estimated_total,
        "prior_spend_estimate_usd": costs["prior_spend_usd"],
        "incremental_worst_case_usd": costs["incremental_worst_case_usd"],
        "configured_worst_case_usd": costs["worst_case_usd"],
        "hard_cap_usd": float(config["modal_budget"]["maximum_usd"]),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_volume.commit()
    return runtime

# Runpod Flash Batch Passage Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Runpod Flash batch backend for agentic Mixedbread passage scoring that scales from zero to three GPU workers and preserves the repository's canonical scored-pair cache guarantees.

**Architecture:** A pure runner-side module owns the remote score identity, request/response contract, and queue HTTP transport. `MixedbreadPassageScorer` accepts that module as a prediction backend while retaining local inference as its default. A self-contained Flash function endpoint uses a persistent Hugging Face cache and one worker-global CrossEncoder; strict agentic config selects the backend without placing endpoint IDs or API keys in tracked files.

**Tech Stack:** Python 3.12, `urllib`, SQLite `GlobalScoreCache`, `sentence-transformers==5.6.0`, `torch==2.9.1`, Runpod Flash 1.19-compatible decorators, pytest.

## Global Constraints

- Pin `mixedbread-ai/mxbai-rerank-base-v2` revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`.
- Return raw logits using identity activation, bfloat16 inference, and maximum length 1,024.
- Use a distinct remote CUDA `ScoreCacheContext`; never write remote scores under the local ROCm passage-cache identity.
- Send no more than 256 passages or 6 MiB of canonical request JSON per job.
- Configure Flash with `workers=(0, 3)`, `idle_timeout=900`, `max_concurrency=1`, FlashBoot, and one 24 GB Ada GPU per worker.
- Persist model files on a Runpod Network Volume; keep result scores in local `GlobalScoreCache` only.
- Do not log raw narratives, queries, passages, API keys, endpoint responses, or authorization headers.
- Do not change the checked-in canonical agentic config or invoke/deploy paid Runpod infrastructure in this implementation.

---

### Task 1: Remote score contract and queue client

**Files:**
- Create: `code/trec_rag/runpod_passage_scorer.py`
- Create: `code/tests/test_runpod_passage_scorer.py`

**Interfaces:**
- Produces: `remote_score_cache_context() -> ScoreCacheContext`.
- Produces: `remote_scorer_identity(request_batch_size: int) -> dict[str, object]`.
- Produces: `RunpodQueueJobClient(endpoint_id: str, api_key: str, timeout_seconds: int, max_retries: int)` with `run(request) -> Mapping[str, object]`.
- Produces: `RunpodFlashPassagePredictor(endpoint_id: str, api_key: str, request_batch_size: int, timeout_seconds: int, max_retries: int, job_client: QueueJobClient | None = None).predict(pairs) -> Sequence[float]`, plus `.cache_context` and `.identity`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_predictor_preserves_order_and_accepts_only_matching_identity():
    predictor = RunpodFlashPassagePredictor(
        endpoint_id="endpoint-1",
        api_key="secret",
        request_batch_size=2,
        job_client=CompletedJobClient(scores=(0.75, -0.25)),
    )
    assert predictor.predict((("query", "first"), ("query", "second"))) == (0.75, -0.25)

def test_predictor_rejects_reordered_ids_without_returning_partial_scores():
    with pytest.raises(RemotePassageScorerError, match="result identities"):
        predictor.predict((("query", "first"), ("query", "second")))

@pytest.mark.parametrize("score", [True, math.nan, math.inf, "0.2"])
def test_predictor_rejects_invalid_scores(score):
    predictor = predictor_returning_scores((score,))
    with pytest.raises(RemotePassageScorerError, match="finite real"):
        predictor.predict((("query", "passage"),))

def test_request_rejects_more_than_256_passages_and_six_mibibytes():
    predictor = predictor_returning_scores(())
    with pytest.raises(ValueError, match="at most 256"):
        predictor.predict(tuple(("query", str(index)) for index in range(257)))
    with pytest.raises(ValueError, match="6 MiB"):
        predictor.predict((("query", "x" * (6 * 1024 * 1024)),))
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest code/tests/test_runpod_passage_scorer.py -q`

Expected: collection fails because `trec_rag.runpod_passage_scorer` does not exist.

- [ ] **Step 3: Implement the identity and contract**

Create immutable constants for schema version, model identity, device family, internal microbatch size 16, request count 256, and request bytes 6 MiB. Build content IDs and the idempotency ID from canonical SHA-256 JSON. Validate one shared nonblank query, nonblank passages, unique ordered IDs, exact response identity digest, exact result count/order, and finite non-Boolean scores.

- [ ] **Step 4: Add failing queue-transport tests**

```python
def test_queue_client_submits_wrapped_input_and_polls_to_completion():
    transport = ScriptedJsonTransport([
        {"id": "job-1", "status": "IN_QUEUE"},
        {"id": "job-1", "status": "COMPLETED", "output": {"ok": True}},
    ])
    client = RunpodQueueJobClient("endpoint-1", "secret", transport=transport, poll_interval=0)
    assert client.run({"schema_version": "runpod_passage_score_request_v1"}) == {"ok": True}

def test_queue_client_retries_429_without_exposing_response_body():
    transport = ScriptedJsonTransport([HttpFailure(429, "private body"), {"id": "job-1", "status": "COMPLETED", "output": {"ok": True}}])
    client = RunpodQueueJobClient("endpoint-1", "secret", transport=transport, poll_interval=0)
    assert client.run({"schema_version": "runpod_passage_score_request_v1"}) == {"ok": True}

def test_queue_client_fails_closed_on_failed_job():
    transport = ScriptedJsonTransport([{"id": "job-1", "status": "FAILED", "error": "private body"}])
    client = RunpodQueueJobClient("endpoint-1", "secret", transport=transport, poll_interval=0)
    with pytest.raises(RemotePassageScorerError, match="FAILED") as captured:
        client.run({"schema_version": "runpod_passage_score_request_v1"})
    assert "private body" not in str(captured.value)
```

- [ ] **Step 5: Run transport tests and verify RED**

Expected: tests fail because queue submission, polling, and bounded retry behavior are absent.

- [ ] **Step 6: Implement the queue client and predictor**

Use `POST https://api.runpod.ai/v2/{endpoint_id}/run` with `{"input": {"request": request}}`, then poll `GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}`. Retry only timeout/connection failures, HTTP 429, and HTTP 5xx with bounded backoff. Reject failed/cancelled/timed-out jobs and malformed JSON without including body text or credentials in exceptions.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add code/trec_rag/runpod_passage_scorer.py code/tests/test_runpod_passage_scorer.py
git commit -m "feat: add Runpod passage score client"
```

### Task 2: Cache-aware Mixedbread backend seam

**Files:**
- Modify: `code/trec_rag/mixedbread_passage_scorer.py`
- Modify: `code/tests/test_mixedbread_passage_scorer.py`

**Interfaces:**
- Consumes: a backend exposing `predict(pairs)`, `cache_context`, and `identity`.
- Produces: `MixedbreadPassageScorer(score_cache_root, device="auto", model_loader=load_pinned_cross_encoder, batch_size=8, read_only=False, prediction_backend=None, request_batch_size=None)` while preserving `rank(query_text, chunks)`.

- [ ] **Step 1: Write failing remote-backend scorer tests**

```python
def test_remote_backend_uses_distinct_context_and_batches_only_cache_misses(tmp_path):
    backend = FakeRemoteBackend(scores=(0.7, 0.8))
    scorer = MixedbreadPassageScorer(
        tmp_path,
        prediction_backend=backend,
        request_batch_size=256,
    )
    rows = scorer.rank("query", chunks("first", "second"))
    assert [row.relevance_score for row in rows] == [0.7, 0.8]
    assert scorer.score_cache.context_sha256 == backend.cache_context.context_sha256

def test_remote_cache_hit_never_invokes_backend(tmp_path):
    seed_backend = FakeRemoteBackend(scores=(0.7,))
    MixedbreadPassageScorer(tmp_path, prediction_backend=seed_backend).rank("query", chunks("cached"))
    replay_backend = FailingRemoteBackend()
    rows = MixedbreadPassageScorer(tmp_path, prediction_backend=replay_backend).rank("query", chunks("cached"))
    assert rows[0].relevance_score == 0.7

def test_remote_failure_commits_no_partial_scores(tmp_path):
    backend = FakeRemoteBackend(scores=(0.7,), failure=RuntimeError("remote failed"))
    scorer = MixedbreadPassageScorer(tmp_path, prediction_backend=backend)
    with pytest.raises(RuntimeError, match="remote failed"):
        scorer.rank("query", chunks("first", "second"))
    assert scorer.score_cache.scores == {}
```

The mutation caught is accidentally retaining the local score context or contacting Runpod for a cache hit.

- [ ] **Step 2: Run focused scorer tests and verify RED**

Run: `PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest code/tests/test_mixedbread_passage_scorer.py -q`

Expected: new constructor arguments are rejected.

- [ ] **Step 3: Implement the prediction seam**

Keep the current local model loader and local identity unchanged when no backend is supplied. For a backend, use its cache context and identity, claim up to `request_batch_size` misses, call its `predict`, validate through `_model_scores`, and let `GlobalScoreCache` atomically commit only a complete batch. Continue heartbeat leases while network inference is pending.

- [ ] **Step 4: Run local and remote scorer tests and verify GREEN**

Run the focused scorer command. Expected: existing local tests and new remote tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add code/trec_rag/mixedbread_passage_scorer.py code/tests/test_mixedbread_passage_scorer.py
git commit -m "feat: support remote passage prediction backends"
```

### Task 3: Strict agentic config and production runner wiring

**Files:**
- Modify: `code/trec_rag/agentic_retrieval_config.py`
- Modify: `code/trec_rag/competition_agentic_retrieval.py`
- Modify: `code/trec_rag/competition_retrieval.py`
- Modify: `code/tests/test_agentic_retrieval_config.py`
- Modify: `code/tests/test_competition_agentic_retrieval.py`

**Interfaces:**
- Produces: optional `passage.scoring` with `backend`, `endpoint_id_env`, `api_key_env`, `request_batch_size`, `timeout_seconds`, and `max_retries`.
- Canonical omission resolves to local scoring.
- Remote config resolves endpoint ID and API key from named environment variables during preflight.

- [ ] **Step 1: Write failing strict-config tests**

```python
def test_config_accepts_remote_flash_scoring_without_embedding_endpoint_id(tmp_path):
    config = load_agentic_retrieval_config(remote_config_path)
    assert config.passage.scoring.backend == "runpod_flash"
    assert config.passage.scoring.endpoint_id_env == "RUNPOD_PASSAGE_ENDPOINT_ID"
    assert config.passage.scoring.request_batch_size == 256

@pytest.mark.parametrize("replacement", ["request_batch_size: 0", "request_batch_size: 257"])
def test_remote_scoring_rejects_invalid_request_batch_size(tmp_path, replacement):
    text = remote_config_text().replace("request_batch_size: 256", replacement)
    with pytest.raises(ValueError, match="request_batch_size"):
        load_agentic_retrieval_config(write_config(tmp_path, text))

def test_local_config_rejects_remote_only_fields(tmp_path):
    text = local_config_text().replace("backend: local", "backend: local\n    endpoint_id_env: RUNPOD_PASSAGE_ENDPOINT_ID")
    with pytest.raises(ValueError, match="remote-only"):
        load_agentic_retrieval_config(write_config(tmp_path, text))
```

- [ ] **Step 2: Run config tests and verify RED**

Expected: `passage.scoring` is rejected as an unknown field.

- [ ] **Step 3: Implement normalized scoring settings**

Default omitted scoring to local. Require exact `runpod_flash` fields for remote mode, uppercase safe environment-variable names, request size 1-256, timeout 1-3600 seconds, and retries 0-5. Include the normalized scoring block in the resolved run-plan payload so create/resume cannot drift.

- [ ] **Step 4: Write failing runner tests**

```python
def test_remote_preflight_requires_endpoint_and_api_key_environment_names():
    with pytest.raises(AgenticRunnerError, match="remote passage scorer"):
        run_agentic_retrieval(remote_config, environ=base_secrets_without_runpod)

def test_production_executor_builds_remote_predictor_and_expected_identity(monkeypatch):
    constructed = install_fake_remote_predictor(monkeypatch)
    execute_one_production_topic(remote_config)
    assert constructed == {"endpoint_id": "endpoint-1", "request_batch_size": 256}

def test_topic_passage_search_accepts_configured_remote_scorer_identity():
    search = _build_topic_passage_search(topic, scorer=remote_scorer, configured_scorer_identity=remote_scorer_identity(256), **search_kwargs)
    assert search.identity["scorer"] == remote_scorer_identity(256)
```

- [ ] **Step 5: Run runner tests and verify RED**

Expected: remote environment preflight and scorer construction are absent, and passage-search identity rejects the remote scorer.

- [ ] **Step 6: Implement production wiring**

Validate the configured environment variables before topic attempts are allocated. Build `RunpodFlashPassagePredictor` only in remote mode, inject it into `MixedbreadPassageScorer`, and pass the independently configured remote scorer identity into `_build_topic_passage_search`. Preserve every local and offline-cache-only path.

- [ ] **Step 7: Run Task 3 tests and verify GREEN**

Run: `PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest code/tests/test_agentic_retrieval_config.py code/tests/test_competition_agentic_retrieval.py code/tests/test_competition_topic_dispatch.py -q`

Expected: all focused config, runner, and existing topic concurrency tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add code/trec_rag/agentic_retrieval_config.py code/trec_rag/competition_agentic_retrieval.py code/trec_rag/competition_retrieval.py code/tests/test_agentic_retrieval_config.py code/tests/test_competition_agentic_retrieval.py
git commit -m "feat: wire Flash scoring into agentic retrieval"
```

### Task 4: Self-contained Flash endpoint

**Files:**
- Create: `code/tools/runpod_flash_passage_scorer/main.py`
- Create: `code/tools/runpod_flash_passage_scorer/README.md`
- Create: `code/tests/test_runpod_flash_passage_endpoint.py`

**Interfaces:**
- Produces queue function `score_batch(request: dict) -> dict`.
- Raw request envelope: `{"input": {"request": <request-v1>}}`.
- Worker configuration: RTX 4090, zero-to-three workers, 900-second idle timeout, one concurrent request, persistent model volume.

- [ ] **Step 1: Write failing endpoint tests with a fake decorator**

```python
def test_endpoint_configuration_scales_to_zero_with_generous_cooldown(endpoint_module):
    assert endpoint_module.ENDPOINT_TEST_CONFIG == {
        "workers": (0, 3),
        "idle_timeout": 900,
        "max_concurrency": 1,
    }

def test_endpoint_scores_ordered_passages_and_returns_matching_identity(endpoint_module):
    endpoint_module._MODEL = FakeBfloat16Model([0.75, -0.25])
    response = asyncio.run(endpoint_module.score_batch(valid_request()))
    assert [row["score"] for row in response["results"]] == [0.75, -0.25]

def test_endpoint_reuses_worker_global_model(endpoint_module):
    loader = install_fake_cross_encoder(endpoint_module)
    asyncio.run(endpoint_module.score_batch(valid_request()))
    asyncio.run(endpoint_module.score_batch(valid_request()))
    assert loader.load_count == 1

def test_endpoint_rejects_malformed_or_oversized_requests(endpoint_module):
    malformed = valid_request()
    malformed["passages"] = []
    with pytest.raises(ValueError, match="passages"):
        asyncio.run(endpoint_module.score_batch(malformed))
    oversized = valid_request(text="x" * (6 * 1024 * 1024))
    with pytest.raises(ValueError, match="6 MiB"):
        asyncio.run(endpoint_module.score_batch(oversized))
```

- [ ] **Step 2: Run endpoint tests and verify RED**

Run: `PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest code/tests/test_runpod_flash_passage_endpoint.py -q`

Expected: endpoint module is absent.

- [ ] **Step 3: Implement the Flash endpoint**

Use `@Endpoint` function form and the documented worker-global `_MODEL` lazy cache. Pin dependencies, volume/data center, model revision, bfloat16 dtype validation, raw-logit activation, and internal microbatch size 16. Keep all remote-executed imports, constants, validation, and helper logic inside the function body so `flash dev` works when only the body is shipped. Emit only counts, timings, IDs, and the fixed identity.

- [ ] **Step 4: Document development and teardown**

Document `uv tool install --python 3.12 runpod-flash`, `flash login`, running `flash dev` from the focused endpoint directory, the double-wrapped request body, `flash deploy`, obtaining the endpoint ID, and `flash app delete` teardown. State that `flash dev` and deployment are paid remote actions requiring explicit scope.

- [ ] **Step 5: Run endpoint tests and syntax validation**

Run the endpoint pytest command, then:

```bash
/home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m py_compile code/tools/runpod_flash_passage_scorer/main.py
```

Expected: tests pass and compilation exits zero without importing Flash.

- [ ] **Step 6: Commit Task 4**

```bash
git add code/tools/runpod_flash_passage_scorer code/tests/test_runpod_flash_passage_endpoint.py
git commit -m "feat: add Flash passage scoring worker"
```

### Task 5: Operator documentation and verification

**Files:**
- Modify: `code/trec_rag/README.md`
- Modify: `docs/superpowers/plans/2026-08-09-runpod-flash-batch-passage-scoring.md`

**Interfaces:**
- Produces a copyable ignored-config block and local test commands.
- Does not add a remote block to `configs/rag26_competition_agentic_retrieval_v1.yaml`.

- [ ] **Step 1: Add operator instructions**

Explain the three cache layers, local-versus-remote score identity, required environment names, ignored smoke-config example, request and worker batch sizes, scale-to-zero cooldown, benchmark metrics, and paid-action boundary.

- [ ] **Step 2: Run the focused suite**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest \
  code/tests/test_runpod_passage_scorer.py \
  code/tests/test_mixedbread_passage_scorer.py \
  code/tests/test_agentic_retrieval_config.py \
  code/tests/test_competition_agentic_retrieval.py \
  code/tests/test_competition_topic_dispatch.py \
  code/tests/test_runpod_flash_passage_endpoint.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run broader regression and static checks**

```bash
PYTHONPATH=code /home/npatta01/data/competitions/trec_rag_2026/.venv/bin/python -m pytest code/tests -q
git diff --check
```

Record any pre-existing environment failures separately from failures caused by this branch.

- [ ] **Step 4: Independently review substantive changes**

Review the branch diff for secret leakage, remote/local cache identity mixing, malformed-response acceptance, unbounded retries, accidental canonical-config changes, and missing teardown instructions. Fix every confirmed finding and rerun affected tests.

- [ ] **Step 5: Mark this plan with verification evidence and commit**

```bash
git add code/trec_rag/README.md docs/superpowers/plans/2026-08-09-runpod-flash-batch-passage-scoring.md
git commit -m "docs: explain Flash passage scoring workflow"
```

The implementation is complete when local tests pass, the worktree contains no uncommitted requested changes, and the final handoff clearly states that no paid Runpod endpoint was invoked or deployed.

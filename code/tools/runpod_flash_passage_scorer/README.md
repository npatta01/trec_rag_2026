# Runpod Flash passage scorer

This directory contains the scale-to-zero GPU worker used by the optional
agentic passage-scoring backend. It scores ordered
`(query, passage)` pairs with the repository-pinned Mixedbread CrossEncoder and
returns raw logits. It does not own topic state or the canonical score cache.

## Cost boundary

`flash dev`, `flash deploy`, and requests to a deployed endpoint use paid
Runpod infrastructure. Do not run them until the topic/input scope has been
explicitly authorized. Importing, compiling, and running the repository tests
does not provision Runpod infrastructure.

## Worker policy

- GPU: NVIDIA RTX 4090 (24 GB)
- workers: zero to three
- idle timeout: 900 seconds
- concurrent jobs per worker: one
- model microbatch: 16 pairs
- request ceiling: 256 passages and 6 MiB of canonical JSON
- execution timeout: 900 seconds
- model cache: `trec-rag-mixedbread-model-cache-v1`, mounted at
  `/runpod-volume`

The worker scales to zero after the cooldown. The Network Volume remains and
lets a later cold worker reuse the pinned Hugging Face snapshot. Runpod charges
for the volume independently of active GPU workers.

## Local validation

From the repository root:

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_runpod_flash_passage_endpoint.py \
  code/tests/test_runpod_passage_scorer.py -q

.venv/bin/python -m py_compile \
  code/tools/runpod_flash_passage_scorer/main.py
```

The endpoint test substitutes a local decorator and fake bfloat16 model. It
validates the real worker body without importing Flash or contacting Runpod.

## Remote development after authorization

Install and authenticate the Flash CLI once:

```bash
uv tool install --python 3.12 runpod-flash
flash login
```

Run development from this focused directory so Flash scans only this worker:

```bash
cd code/tools/runpod_flash_passage_scorer
flash dev > /tmp/trec-rag-flash-passage-scorer.log 2>&1 &
```

Read the actual local port from the log. A queue request goes to
`/main/runsync` and is double-wrapped:

```json
{
  "input": {
    "request": {
      "schema_version": "runpod_passage_score_request_v1",
      "request_id": "<canonical-request-sha256>",
      "expected_identity_sha256": "<endpoint-identity-sha256>",
      "query": "<query>",
      "passages": [
        {"content_id": "<canonical-pair-sha256>", "text": "<passage>"}
      ]
    }
  }
}
```

Use `trec_rag.runpod_passage_scorer.RunpodFlashPassagePredictor` to construct
valid content/request identities; do not hand-author production requests.

After the development endpoint passes a real external request, deploy it:

```bash
flash deploy
flash env list
flash env get <environment-name>
```

Put the resulting endpoint ID and API key only in the environment used by the
ignored smoke config:

```bash
export RUNPOD_PASSAGE_ENDPOINT_ID='<endpoint-id>'
export RUNPOD_API_KEY='<api-key>'
```

Never put either value in tracked YAML or logs.

## Teardown

After an authorized benchmark, identify the Flash app and remove it:

```bash
flash app list
flash app delete <app-name>
```

Deleting the app removes its endpoints but not necessarily the persistent
Network Volume. Inspect the volume separately before deleting it because it
contains the reusable model snapshot.

## Cache layers

1. The crawler's `GlobalScoreCache` is authoritative for scored pairs. Its
   hits bypass Runpod entirely.
2. The Runpod Network Volume caches model files across cold workers.
3. The worker-global `_MODEL` keeps one loaded CrossEncoder for the warm
   worker's lifetime.

Remote CUDA scores use a separate cache context from local ROCm scores. They
must not be copied into the local namespace without a separately reviewed
parity and promotion procedure.

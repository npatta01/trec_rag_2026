# RAG25 Topic 213 local non-agentic run

Date: 2026-08-05

Branch: `codex/rag25-topic213-nonagentic-local`

Implementation base: `70186faafffe2a3bc67a090361b9f87a2c024479`

## Outcome

The existing full-passage, non-agentic map/reduce workflow completed locally for
RAG25 development topic `213`. The run used the frozen Topic 213 evidence packet
and the local `Qwen/Qwen3-4B-Instruct-2507` model identity. It made no hosted
generation or retrieval calls.

The final output was regenerated from an internally consistent set of prior
local, content-addressed Qwen checkpoints. The runner re-read every cached
completion and applied its normal structured-output validation before publishing
the result. A short fresh inference attempt was retained only under ignored
`tmp/` storage and was not mixed into the final run because its wording changed
downstream request hashes.

## Scope and provenance

- Topic source:
  `trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv`
- Topic: `213` (Korean War)
- Evidence packet: `tmp/topic213-evidence-handover-v1/passages.jsonl`
- Evidence packet SHA-256:
  `4a2d91d4dd697823ccbcff01654673f4b7a2fdd77e3267dc4f029b9528845637`
- Evidence manifest SHA-256:
  `d93fbfd14ded43d428abd9e15936f45015ecd808afa6ccd126c3ac1a928e8201`
- Data submodule revision:
  `1de1b22ac7f9936be7e42c9e70d576cc9cb83770`
- Config:
  `configs/rag25_topic213_full_documents_qwen_local_v1.yaml`
- Runner: `trec_rag.topic213_response_experiment`
- Model identity: `Qwen/Qwen3-4B-Instruct-2507`
- Serving runtime: vLLM `0.24.0`
- Temperature: `0.0`

The workflow consumes all available passages in a non-agentic sequence:
evidence mapping, per-subnarrative reduction, section generation, support audit,
and post-freeze nugget evaluation. Organizer nuggets remain unavailable during
generation.

## Input accounting

| Item | Count |
| --- | ---: |
| Eligible documents | 173 |
| Processed documents | 171 |
| Unavailable documents | 2 |
| Processed passages | 1,478 |
| Evidence assignments | 1,011 |
| Processed subnarratives | 10 |
| Passage batches | 70 |

The two unavailable documents were `shard_00001_10555` and
`shard_00004_63642`. They were already absent from the frozen evidence packet
because their original provider calls failed; the local run did not fabricate
replacement evidence.

## Results

| Metric | Result |
| --- | ---: |
| Generated answer claims | 50 |
| Supported claims | 42 |
| Partially supported claims | 6 |
| Unsupported claims | 2 |
| Contradicted claims | 0 |
| Citation coverage | 1.000 |
| Strict nugget coverage | 0.520 |
| Partial-credit nugget coverage | 0.630 |
| Vital strict coverage | 0.556 |
| Response words | 1,883 |

This is a development-data experiment, not an organizer-ready submission. In
particular, the response exceeds the TREC RAG 2026 1,024-word ceiling, and the
automated Qwen support and nugget judgments have not been manually adjudicated.

## Validation

The focused runner test passed:

```text
10 passed in 0.03s
```

The regenerated artifacts matched the recorded experiment hashes exactly:

| Artifact | SHA-256 |
| --- | --- |
| `response_generation.json` | `9d4260d1e89744dec00f9d664a9e5316608e2e13ad04e61dfc420add3ae8af90` |
| `generated_response.md` | `e6e270433985cec75659aabdd9913a7c8f085df1c5c97d22b01f31c36f877797` |
| `claim_support_audit.jsonl` | `566e77205bf7b0109143bf0c076595b3e18af23edfa20765a760cbab20e9ee39` |
| `nugget_comparison.jsonl` | `8ca6fcb08369d0206df2bfaddec0880d3082ef2c7811246bb55585172cc836a7` |
| `metrics.json` | `cf33f77fe699f3cbb4833051497c5499890c9f074ef4beb75ac3966d8f42bf42` |
| `evaluation_report.md` | `44d4a10c90152b6691c788446718049be19cbe80ea12e022db04f3ef396dab84` |

The tracked worktree was clean before this summary was added. Private outputs,
checkpoints, server logs, the copied `.env`, and the evidence packet remain in
ignored locations and are not part of this change.

## Local command

With the configured OpenAI-compatible `qwen-local` endpoint available:

```powershell
$env:PYTHONPATH = "code"
C:\dev\trec_rag\.venv\Scripts\python.exe `
  -m trec_rag.topic213_response_experiment `
  --config configs\rag25_topic213_full_documents_qwen_local_v1.yaml
```

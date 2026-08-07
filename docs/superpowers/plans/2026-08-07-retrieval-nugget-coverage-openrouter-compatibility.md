# Retrieval Nugget Coverage OpenRouter Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore strict OpenRouter execution for Retrieval Nugget Coverage, make `openai/gpt-5.6-sol` the planner and judge default, and complete the private 22-topic 2025 evaluation.

**Architecture:** Keep the existing one-topic, two-stage evaluator and change only its provider request identity: omit unsupported `temperature`, advance the evaluator schema to v2, and update both default model identities. Persist new evaluations under an explicit private v2 namespace so the failed v1 topic-14 input remains an immutable audit artifact.

**Tech Stack:** Python 3.12, pytest, OpenRouter Chat Completions with strict JSON Schema, authenticated generation handoff artifacts, Markdown skill and README contract tests.

## Global Constraints

- Remove `temperature` from planner and judge provider requests.
- Keep `provider.require_parameters: true` and `provider.data_collection: "deny"`.
- Keep strict JSON Schema, `seed: 0`, `reasoning: {"enabled": false}`, `stream: false`, bounded transport retries, and zero semantic retries.
- Set both defaults to `openai/gpt-5.6-sol`.
- Set the evaluator identity to `retrieval_nugget_coverage_v2`; planner prompt v4 and judge prompt v2 remain unchanged.
- Never reuse, delete, or reinterpret the failed v1 topic-14 namespace.
- Send only narrative, frozen obligation plan, and canonical nugget text to the explicitly approved OpenRouter model.
- Never send passages or run retrieval, reranking, generation, qrels, or gold-nugget evaluation.
- Keep input, responses, work directories, and reports private and outside git.
- Stop the multi-topic run after any failed probe or incomplete receipt.

---

### Task 1: Provider request and evaluator identity

**Files:**
- Modify: `code/tests/test_retrieval_nugget_coverage.py:1-1205`
- Modify: `code/trec_rag/retrieval_nugget_coverage.py:40-65,1294-1314`

**Interfaces:**
- Consumes: `CoverageRunConfig`, `OpenRouterCoverageBackend`, `render_planner_request`, and the existing `_request_payload` serializer.
- Produces: v2 evaluator identity, `openai/gpt-5.6-sol` defaults, and a strict provider payload without `temperature`.

- [ ] **Step 1: Write failing request and default-identity assertions**

In `test_openrouter_backend_uses_strict_schema_and_redacts_credentials`, replace the old temperature assertion with an omission assertion while retaining every safety assertion:

```python
    assert body["provider"] == {"require_parameters": True, "data_collection": "deny"}
    assert body["reasoning"] == {"enabled": False}
    assert "temperature" not in body
    assert body["seed"] == 0
    assert body["stream"] is False
    assert body["max_tokens"] == 8192
```

Add this focused defaults test near `_identity`:

```python
def test_current_default_identity_uses_sol_and_v2_schema(tmp_path: Path) -> None:
    config = CoverageRunConfig(tmp_path / "handoff.json", "topic-defaults")

    assert coverage_module.EVALUATOR_SCHEMA_VERSION == "retrieval_nugget_coverage_v2"
    assert config.planner_model == "openai/gpt-5.6-sol"
    assert config.judge_model == "openai/gpt-5.6-sol"
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py::test_current_default_identity_uses_sol_and_v2_schema \
  code/tests/test_retrieval_nugget_coverage.py::test_openrouter_backend_uses_strict_schema_and_redacts_credentials \
  -q
```

Expected: both tests fail because the production identity is v1/GPT-5 and the serialized request still contains `temperature`.

- [ ] **Step 3: Implement the minimal identity and request change**

Change only these constants:

```python
EVALUATOR_SCHEMA_VERSION = "retrieval_nugget_coverage_v2"
DEFAULT_PLANNER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-sol"
```

Make `_request_payload` return the same mapping without the temperature entry:

```python
    return {
        "model": request.model,
        "messages": [dict(message) for message in request.messages],
        "max_tokens": request.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_schema_name,
                "strict": True,
                "schema": request.response_schema,
            },
        },
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False},
        "seed": 0,
        "stream": False,
    }
```

- [ ] **Step 4: Run the focused and complete evaluator suites and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py::test_current_default_identity_uses_sol_and_v2_schema \
  code/tests/test_retrieval_nugget_coverage.py::test_openrouter_backend_uses_strict_schema_and_redacts_credentials \
  -q

.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -q
```

Expected: the focused tests and the complete evaluator test module pass with no hosted calls.

- [ ] **Step 5: Commit Task 1**

```bash
git add code/trec_rag/retrieval_nugget_coverage.py code/tests/test_retrieval_nugget_coverage.py
git commit -m "Fix coverage OpenRouter request identity"
```

### Task 2: Skill and README contract

**Files:**
- Modify: `code/tests/test_retrieval_nugget_coverage_skill.py:20-55`
- Modify: `.agents/skills/trec-rag-competition-debug-report/SKILL.md:145-190`
- Modify: `code/trec_rag/README.md:1780-1850`

**Interfaces:**
- Consumes: the Task 1 defaults and v2 provider request behavior.
- Produces: a year-neutral user workflow that truthfully states the Sol defaults and supported deterministic controls.

- [ ] **Step 1: Write failing skill-contract assertions**

Extend `test_retrieval_nugget_coverage_route_documents_the_cli_contract`:

```python
    assert "defaults are `openai/gpt-5.6-sol` for each" in section
```

The existing year-specific isolation test remains unchanged and must continue to reject `2025`, gold, and qrels inside the skill route.

- [ ] **Step 2: Run the skill test and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_skill.py::test_retrieval_nugget_coverage_route_documents_the_cli_contract \
  -q
```

Expected: FAIL because the skill still names `openai/gpt-5`.

- [ ] **Step 3: Update the skill and README**

In the skill route, replace the defaults sentence with:

```markdown
Before egress in an authorized path, state the provider (OpenRouter), the
planner and judge model identities (defaults are `openai/gpt-5.6-sol` for each,
or the exact `--planner-model` and `--judge-model` overrides), and that the
maximum of two hosted calls is one narrative-only planner call followed by one
all-nugget judge call.
```

In the README provider-contract paragraph, replace the obsolete temperature statement with:

```markdown
Both structured OpenRouter requests require provider parameter support, deny
provider data collection, disable reasoning, omit unsupported sampling
temperature, use seed zero, and set `stream=false`; semantic retries are
disabled.
```

Change the README assumption heading/body from Version 1 to Version 2 while leaving the limitation itself unchanged.

- [ ] **Step 4: Run skill and neighboring regression suites and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_retrieval_nugget_coverage.py \
  -q
```

Expected: all selected tests pass, including the year-neutral route checks.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  .agents/skills/trec-rag-competition-debug-report/SKILL.md \
  code/trec_rag/README.md \
  code/tests/test_retrieval_nugget_coverage_skill.py
git commit -m "Document Sol coverage evaluation defaults"
```

### Task 3: Verification and private 22-topic execution

**Files:**
- Read: `outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/generation_handoff_manifest.json` in the shared checkout
- Create privately: `outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/retrieval_nugget_coverage_v2/<topic>/` in the shared checkout
- Create privately: `/tmp/retrieval_nugget_coverage_2025_summary.json`

**Interfaces:**
- Consumes: the Task 1 evaluator, Task 2 skill contract, the authenticated 22-topic handoff, and the ignored `.env` credential.
- Produces: 22 validated private v2 manifests plus a sanitized local score summary suitable for an inline user report.

- [ ] **Step 1: Run static and full local verification**

Run:

```bash
.venv/bin/python -m compileall -q code/trec_rag/retrieval_nugget_coverage.py
.venv/bin/python -m trec_rag.retrieval_nugget_coverage --help
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_retrieval_nugget_coverage_skill.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_generation_handoff.py \
  -q
.venv/bin/python -m pytest -q
git diff --check origin/master...HEAD
git status --short --branch --ignore-submodules=all
```

Expected: compile/help exit zero, all tests pass, diff check passes, and only intentional branch commits differ from master.

- [ ] **Step 2: Cache-check a fresh v2 topic-14 namespace**

Run:

```bash
.venv/bin/python -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest /home/npatta01/data/competitions/trec_rag_2026/outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/generation_handoff_manifest.json \
  --topic 14 \
  --work-dir /home/npatta01/data/competitions/trec_rag_2026/outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/retrieval_nugget_coverage_v2/14
```

Expected: safe cache error naming missing planner and judge stages, zero hosted calls, and no new v2 work directory.

- [ ] **Step 3: Run and validate the authorized topic-14 probe**

Load `.env` into the process without printing it and run:

```bash
set -a
source .env
set +a
.venv/bin/python -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest /home/npatta01/data/competitions/trec_rag_2026/outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/generation_handoff_manifest.json \
  --topic 14 \
  --work-dir /home/npatta01/data/competitions/trec_rag_2026/outputs/nonagentic-rag25-dev-all22-replay-p4-20260807/retrieval_nugget_coverage_v2/14 \
  --allow-hosted-calls
```

Require a receipt with `status: complete`, `topic_id: "14"`, `hosted_calls: 2`, no reused stages, and five artifact hashes. Validate the manifest-last bundle with a cache-only resume against the same work directory and require `hosted_calls: 0` plus reused planner and judge stages.

- [ ] **Step 4: Run the remaining 21 topics sequentially**

For topic IDs `31 37 58 72 84 144 161 200 213 219 224 225 233 273 300 407 477 499 515 707 897`, run the same cache-only create then authorized create sequence, substituting the topic ID in both `--topic` and the terminal work-directory component. Require each hosted receipt to be complete with exactly two hosted calls before advancing. Never run two hosted commands concurrently.

- [ ] **Step 5: Validate every bundle and derive the sanitized summary**

Run a cache-only `--mode resume` for every v2 work directory and require complete receipts with zero hosted calls. Build `/tmp/retrieval_nugget_coverage_2025_summary.json` from receipt/report scalar fields only:

```json
{
  "topic_count": 22,
  "topics": [
    {
      "topic_id": "14",
      "obligation_count": 0,
      "nugget_count": 0,
      "required_coverage": 0.0,
      "strict_full_rate": 0.0,
      "supplemental_coverage": null
    }
  ]
}
```

The numeric zeros above describe the summary schema, not expected results; populate them from each validated receipt. Compute macro means from the 22 scalar topic scores and report min/max topics without copying narrative, nugget, judgment rationale, provider events, hashes, or credentials.

- [ ] **Step 6: Final clean-state check and handoff**

Run:

```bash
git status --short --branch --ignore-submodules=all
git diff --check origin/master...HEAD
```

Expected: the tracked worktree is clean. Report the evaluated topic count, hosted-call total, complete-manifest count, aggregate scores, score range, limitations, and private artifact paths inline. Do not publish or serve any report.

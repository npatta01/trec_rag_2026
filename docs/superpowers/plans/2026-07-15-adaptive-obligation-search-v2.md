# Adaptive Obligation Search V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ('- [ ]') syntax for tracking.

**Goal:** Implement and verify the inference-free portion of adaptive-obligation search v2, ending with an exact 48-job proposal preflight and a rendered explanation artifact.

**Architecture:** Rebuild deterministic parent/fold evidence units from the authenticated four-topic contract and MiniLM scores under a new v2 identity. Freeze compact O1-only proposal jobs, guarded append-only model and retrieval ledgers, opposite-fold validation, and focused BM25 request planning. No Qwen generation, BM25 request, new MiniLM inference, qrels access, or promotion occurs in this plan.

**Tech Stack:** Python 3.12, pytest, existing JSON/JSONL canonicalization helpers, local Hugging Face tokenizer files without model loading, existing persistent requests rate limiter and raw-first ledger patterns, self-contained HTML, and headless Chrome/Playwright when available.

## Global Constraints

- Work only in '/home/npatta01/.codex/worktrees/41f9/trec_rag_2026' on branch 'codex/structured-query-planner'.
- Preserve immutable v2.1 artifacts, sealed baseline rankings, and discovery v1. Do not modify 'adaptive_evidence_discovery.py', 'adaptive_evidence_local_model.py', or any v1 runtime artifact.
- Keep unrelated untracked 'sparse_relevance*' files untouched and out of every commit.
- Hard-reject topic IDs '144', '213', '224', '407', and '515' before source, cache, tokenizer, model, endpoint, qrels, or output access.
- Pilot topic IDs are exactly '219', '72', '300', and '84'.
- Authenticated inputs contain exactly 8,114 documents, four broad obligations, 24 O0 obligations, 98,053 score windows, 96,911 unique pairs, and 28 score shards.
- V2 output root is create-only: 'outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/'.
- Model ID is 'Qwen/Qwen3-4B-Instruct-2507', revision 'cdbee75f17c01a7cc42f958dc650907174af0554'.
- Proposal jobs are exactly 48 primary jobs. Primary output ceiling is 256 tokens; the only permitted retry ceiling is 512 tokens for a completion truncated at the 256-token ceiling.
- Proposal preflight records a primary-call count of 48, retry-call ceiling of 48, and worst-case proposal-call ceiling of 96.
- Proposal schema emits only O1 records. It never emits N1 answer nuggets.
- Accepted O1 limits are one per O0 parent and four per topic.
- Focused BM25 request rendering is ordered parent anchor terms + complete O0 text + O1 label, with exact duplicate phrases removed after first occurrence.
- Retrieval depth is 1,000 from one request per accepted O1. Maximum accepted requests are 16. Initial transport retry count is zero, timeout is 120 seconds, and the persistent limiter allows at most one request start per three seconds.
- The semantic MiniLM query, planned for a later approved stage, is unchanged narrative + complete O0 + accepted O1.
- No raw BM25 or MiniLM score crosses query boundaries.
- No model weights may load. A locally materialized tokenizer may load only during proposal preflight; record 'tokenizer_load_count=1', 'model_load_count=0', and 'inference_count=0'.
- Do not create any approval artifact. Execution commands must reject unless a later, separately supplied approval binds the exact preflight SHA-256.
- Do not make network, retrieval, hosted inference, paid, Qwen generation, MiniLM inference, qrels, organizer-nugget, reference-answer, or review-label calls.
- Use process-local 'TMPDIR=/var/tmp' for Python commands.

---

## File Map

- 'code/trec_rag/adaptive_obligation_v2_contract.py': source authentication, protected-topic firewall, deterministic units, fold reservoirs, create-only contract, and verifier.
- 'code/trec_rag/adaptive_obligation_v2_propose.py': compact O1 schema, proposal messages/jobs, tokenizer-only preflight, receipts, and guarded proposal entrypoint.
- 'code/trec_rag/adaptive_obligation_v2_ledger.py': append-only attempt records, raw-first completion capture, truncation classification, and immutable replay verification.
- 'code/trec_rag/adaptive_obligation_v2_local_model.py': separately versioned local Qwen adapter that cannot construct or generate without an approved proposal preflight.
- 'code/trec_rag/adaptive_obligation_v2_validate.py': deterministic proposal checks, opposite-fold validator jobs, finite decisions, acceptance, and caps.
- 'code/trec_rag/adaptive_obligation_v2_retrieve.py': focused query rendering, request identities, cache audit, raw-first retrieval ledger interface, and guarded transport.
- 'code/trec_rag/build_adaptive_obligation_v2_report.py': verified report loader and self-contained HTML renderer.
- 'code/tests/test_adaptive_obligation_v2_*.py': focused component tests.
- 'reports/experiments/adaptive_obligation_search_v2/report.html': rendered preflight/status artifact.

---

### Task 1: Freeze deterministic v2 evidence units and reservoirs

**Files:**
- Create: 'code/trec_rag/adaptive_obligation_v2_contract.py'
- Create: 'code/tests/test_adaptive_obligation_v2_contract.py'
- Create at runtime: 'outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract/'

**Interfaces:**
- Consumes: authenticated Task 1 contract and Task 3 base scores from 'adaptive_evidence_ranker_v1'.
- Produces: 'reject_protected_before_access(...)', 'split_exact_units(...)', 'build_v2_contract(...)', 'verify_v2_contract(...)', 'manifest.json', 'parents.jsonl', 'reservoirs.jsonl', 'units.jsonl', and 'receipt.json'.

- [ ] **Step 1: Write failing protected-access, unit, reservoir, and seal tests**

~~~python
def test_protected_topic_fails_before_any_loader() -> None:
    touched = []
    with pytest.raises(ValueError, match="protected"):
        reject_protected_before_access(
            ["144"],
            source_loader=lambda: touched.append("source"),
        )
    assert touched == []


def test_unit_ids_bind_exact_source_span() -> None:
    units = split_exact_units(
        topic_id="219",
        parent_id="219-positive",
        fold=0,
        document_id="d1",
        window_id="w1",
        text="Technology helps access. It can also exclude people.",
    )
    assert [row["text"] for row in units] == [
        "Technology helps access.",
        "It can also exclude people.",
    ]
    assert all(row["text"] in "Technology helps access. It can also exclude people." for row in units)
    assert len({row["unit_id"] for row in units}) == 2


def test_contract_has_exact_parent_fold_reservoirs() -> None:
    contract = build_v2_contract(_authenticated_fixture())
    assert len(contract["parents"]) == 24
    assert len(contract["reservoirs"]) == 48
    assert all(row["document_count"] == 10 for row in contract["reservoirs"])
    assert all(len(set(row["document_ids"])) == 10 for row in contract["reservoirs"])


def test_v1_discovery_is_never_a_contract_source() -> None:
    contract = build_v2_contract(_authenticated_fixture())
    assert "discovery" not in contract["source_bindings"]
~~~

- [ ] **Step 2: Run the tests and verify the missing module**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py -q
~~~

Expected: collection fails with 'ModuleNotFoundError' for 'adaptive_obligation_v2_contract'.

- [ ] **Step 3: Implement exact units and deterministic top-ten reservoirs**

~~~python
import hashlib
import re

PILOT_TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
EXPECTED_DOCUMENT_COUNT = 8_114
EXPECTED_O0_COUNT = 24
EXPECTED_RESERVOIR_COUNT = 48
RESERVOIR_DOCUMENT_LIMIT = 10


_UNIT_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+[^\n]+"
    r"|[^.!?\n]+(?:[.!?]+|$)"
)


def sentence_and_list_item_spans(text):
    spans = []
    for match in _UNIT_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_exact_units(*, topic_id, parent_id, fold, document_id, window_id, text):
    spans = sentence_and_list_item_spans(text)
    return [
        {
            "schema_version": "adaptive-obligation-v2-unit-v1",
            "unit_id": canonical_sha256({
                "topic_id": topic_id,
                "parent_id": parent_id,
                "fold": fold,
                "document_id": document_id,
                "window_id": window_id,
                "start": start,
                "end": end,
                "text": text[start:end],
            }),
            "topic_id": topic_id,
            "parent_id": parent_id,
            "fold": fold,
            "document_id": document_id,
            "window_id": window_id,
            "start": start,
            "end": end,
            "text": text[start:end],
            "text_sha256": sha256_text(text[start:end]),
        }
        for start, end in spans
    ]
~~~

Reservoir selection must reduce scores within one '(topic, parent, document)' queue, sort by descending finite MiniLM score then earliest span, retain distinct documents, and take exactly ten per fold. Fail closed when any parent/fold has fewer than ten. Validate every unit span against its immutable window text. Reject protected topics and exact-topic/count mismatches before reading the large document or score-shard bodies.

Write the five artifacts to a sibling staging directory, fsync them, verify their row/byte/SHA-256 bindings, and atomically rename only when the final destination is absent. Reject existing, partial, unexpected, non-regular, or symlinked roots and children.

- [ ] **Step 4: Run focused and compatibility tests, then build and verify the canonical contract**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_evidence_contract.py \
  code/tests/test_adaptive_evidence_score.py \
  code/tests/test_adaptive_evidence_discovery.py \
  code/tests/test_adaptive_evidence_rank.py -q

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.adaptive_obligation_v2_contract build \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --scores outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/base \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.adaptive_obligation_v2_contract verify \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract
~~~

Expected: 24 parents, 48 reservoirs, 480 distinct reservoir document slots, exact units with valid spans, protected count zero, v1 absent from source bindings, and all network/model/inference/qrels counters zero.

- [ ] **Step 5: Commit Task 1**

~~~bash
git add code/trec_rag/adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_obligation_v2_contract.py
git commit -m "Freeze adaptive obligation v2 evidence units"
~~~

---

### Task 2: Freeze compact proposal jobs and the inference-free preflight

**Files:**
- Create: 'code/trec_rag/adaptive_obligation_v2_propose.py'
- Create: 'code/tests/test_adaptive_obligation_v2_propose.py'
- Create at runtime: 'outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight/'

**Interfaces:**
- Consumes: verified Task 1 v2 contract and the local Qwen snapshot/tokenizer.
- Produces: 'PROPOSAL_SCHEMA', 'render_proposal_messages(...)', 'build_proposal_jobs(...)', 'build_proposal_preflight(...)', 'verify_proposal_preflight(...)', 'jobs.jsonl', 'schema.json', 'prompt.json', and 'receipt.json'.

- [ ] **Step 1: Write failing compact-schema and preflight tests**

~~~python
def test_schema_is_one_o1_or_unsupported() -> None:
    assert PROPOSAL_SCHEMA["additionalProperties"] is False
    assert set(PROPOSAL_SCHEMA["properties"]["status"]["enum"]) == {
        "SUPPORTED", "UNSUPPORTED"
    }
    assert "n1" not in json.dumps(PROPOSAL_SCHEMA).casefold()


def test_jobs_are_exactly_parent_by_fold() -> None:
    jobs = build_proposal_jobs(_contract_fixture())
    assert len(jobs) == 48
    assert len({row["job_id"] for row in jobs}) == 48
    assert {(row["parent_id"], row["fold"]) for row in jobs} == _expected_pairs()


def test_preflight_never_constructs_a_model(tmp_path: Path) -> None:
    touched = []
    receipt = build_proposal_preflight(
        _contract_fixture(),
        tokenizer=_fake_tokenizer(),
        model_factory=lambda: touched.append("model"),
        output_dir=tmp_path / "preflight",
    )
    assert touched == []
    assert receipt["primary_call_count"] == 48
    assert receipt["retry_call_ceiling"] == 48
    assert receipt["worst_case_call_ceiling"] == 96
    assert receipt["tokenizer_load_count"] == 1
    assert receipt["model_load_count"] == 0
    assert receipt["inference_count"] == 0
~~~

- [ ] **Step 2: Run the tests and verify the missing module**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_propose.py -q
~~~

Expected: collection fails for the missing proposal module.

- [ ] **Step 3: Implement the compact schema, messages, jobs, and receipt**

~~~python
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PRIMARY_MAX_NEW_TOKENS = 256
RETRY_MAX_NEW_TOKENS = 512
PRIMARY_JOB_COUNT = 48

PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reason_code", "o1"],
    "properties": {
        "status": {"enum": ["SUPPORTED", "UNSUPPORTED"]},
        "reason_code": {
            "enum": [
                "SUPPORTED",
                "NO_ABSTRACT_CHILD",
                "ONLY_ANSWER_FACTS",
                "INSUFFICIENT_SCOPE",
            ]
        },
        "o1": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "scope_rationale", "support_unit_ids"],
                    "properties": {
                        "label": {"type": "string", "minLength": 3, "maxLength": 120},
                        "scope_rationale": {"type": "string", "minLength": 3, "maxLength": 240},
                        "support_unit_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                    },
                },
            ]
        },
    },
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "SUPPORTED"}}},
            "then": {
                "properties": {
                    "reason_code": {"const": "SUPPORTED"},
                    "o1": {"type": "object"},
                }
            },
            "else": {
                "properties": {
                    "reason_code": {
                        "enum": [
                            "NO_ABSTRACT_CHILD",
                            "ONLY_ANSWER_FACTS",
                            "INSUFFICIENT_SCOPE",
                        ]
                    },
                    "o1": {"type": "null"},
                }
            },
        }
    ],
}
~~~

Messages must contain the unchanged narrative, complete O0, one fold's exact unit records, and the response schema. The instructions require one abstract child need or 'UNSUPPORTED', forbid outside knowledge and answer facts, and require support unit IDs. Bind messages, schema, job identity, contract receipt, model snapshot inventory, tokenizer files, prompt token counts, output ceilings, code hashes, and safety counters.

No 'execute' action may occur in this task. The CLI exposes only 'build-preflight' and 'verify-preflight'.

- [ ] **Step 4: Run tests, build the canonical 48-job preflight, and verify it**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_obligation_v2_propose.py -q

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.adaptive_obligation_v2_propose build-preflight \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.adaptive_obligation_v2_propose verify-preflight \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight
~~~

Expected: exactly 48 jobs; primary ceiling 48, retry ceiling 48, worst-case 96; locally verified tokenizer/model snapshot identity; model loads zero; inference/network/retrieval/qrels/paid counters zero.

- [ ] **Step 5: Commit Task 2**

~~~bash
git add code/trec_rag/adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_propose.py
git commit -m "Add adaptive obligation v2 proposal preflight"
~~~

---

### Task 3: Add the guarded append-only proposal ledger without running it

**Files:**
- Create: 'code/trec_rag/adaptive_obligation_v2_ledger.py'
- Create: 'code/trec_rag/adaptive_obligation_v2_local_model.py'
- Modify: 'code/trec_rag/adaptive_obligation_v2_propose.py'
- Create: 'code/tests/test_adaptive_obligation_v2_ledger.py'
- Create: 'code/tests/test_adaptive_obligation_v2_local_model.py'
- Modify: 'code/tests/test_adaptive_obligation_v2_propose.py'

**Interfaces:**
- Produces: 'AppendOnlyAttemptLedger', 'classify_completion(...)', 'V2LocalJsonModel', and 'execute_proposals(...)'.
- No runtime proposal output is created in this plan.

- [ ] **Step 1: Write failing approval, raw-first, crash, and retry tests**

~~~python
def test_execute_rejects_before_model_or_output_without_approval(tmp_path: Path) -> None:
    touched = []
    with pytest.raises(PermissionError, match="proposal inference approval required"):
        execute_proposals(
            preflight_dir=tmp_path / "missing",
            approval_path=tmp_path / "missing-approval.json",
            output_dir=tmp_path / "out",
            model_factory=lambda: touched.append("model"),
        )
    assert touched == []
    assert not (tmp_path / "out").exists()


def test_attempt_start_precedes_model_call(tmp_path: Path) -> None:
    observed = []
    ledger = AppendOnlyAttemptLedger(tmp_path / "ledger")
    def generate(_messages, _schema, _max_new_tokens):
        observed.extend(ledger.read_events())
        return b'{"status":"UNSUPPORTED","reason_code":"NO_ABSTRACT_CHILD","o1":null}'
    ledger.run_attempt(_attempt(), generate)
    assert observed[-1]["state"] == "started"


def test_only_ceiling_truncation_gets_one_retry(tmp_path: Path) -> None:
    calls = []
    result = run_job_with_retry(
        _job(),
        generate=lambda ceiling: calls.append(ceiling) or (
            b'{"status":' if ceiling == 256 else _valid_unsupported_bytes()
        ),
    )
    assert calls == [256, 512]
    assert result["status"] == "UNSUPPORTED"


def test_schema_error_is_not_retried() -> None:
    calls = []
    with pytest.raises(ValueError, match="schema"):
        run_job_with_retry(
            _job(),
            generate=lambda ceiling: calls.append(ceiling) or b'{"status":"BAD"}',
        )
    assert calls == [256]
~~~

- [ ] **Step 2: Run tests and verify the missing ledger/model interfaces**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_ledger.py \
  code/tests/test_adaptive_obligation_v2_local_model.py \
  code/tests/test_adaptive_obligation_v2_propose.py -q
~~~

Expected: failures for missing ledger, local-model, and guarded execution symbols.

- [ ] **Step 3: Implement immutable attempt records and the separately gated model adapter**

~~~python
@dataclass(frozen=True)
class AttemptSpec:
    stage: str
    job_id: str
    attempt_ordinal: int
    request_sha256: str
    max_new_tokens: int


def verify_inference_approval(approval, preflight):
    required = {
        "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
        "stage": "proposal",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
    }
    if any(approval.get(name) != value for name, value in required.items()):
        raise PermissionError("proposal inference approval required")
    if approval.get("approved") is not True:
        raise PermissionError("proposal inference approval required")
    return dict(approval)


class V2LocalJsonModel:
    def __init__(self, *, approval, preflight):
        verify_inference_approval(approval, preflight)
        self._runtime = load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def generate(self, messages, schema, *, max_new_tokens):
        return self._runtime.generate_json_bytes(
            messages,
            schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
~~~

'verify_inference_approval' must authenticate a separately supplied create-only approval record whose preflight SHA-256, model, revision, stage, primary count, and retry ceiling match exactly. Missing, malformed, symlinked, or wrong-stage approval fails before model/tokenizer/output access.

'load_pinned_local_runtime' must use 'AutoTokenizer.from_pretrained' and
'AutoModelForCausalLM.from_pretrained' on the exact local snapshot with
'local_files_only=True', 'trust_remote_code=False', 'use_safetensors=True',
'torch_dtype=torch.bfloat16', evaluation mode, and the verified ROCm device.
The runtime returns both raw completion bytes and exact output-token count.
'classify_completion' parses exactly one JSON value, validates
'PROPOSAL_SCHEMA', and labels a parse failure 'truncated_at_ceiling' only when
the output-token count equals the current ceiling; every other parse or schema
failure is terminal.

The ledger writes and fsyncs a 'started' event before calling the model, writes and fsyncs raw completion bytes before parsing, then appends one terminal event. Reopening verifies the complete hash chain and rejects deletion, reordering, duplicate ordinals, extra artifacts, or a started attempt without its preserved state. Only an actual completion that reached 256 tokens and ended as incomplete JSON may retry once at 512. Never retry semantic or schema failures.

Add an 'execute' CLI subcommand but do not create an approval record and do not invoke it. Its unapproved smoke test must fail before model loading and output creation.

- [ ] **Step 4: Run focused and compatibility tests; prove unapproved execution is inert**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_ledger.py \
  code/tests/test_adaptive_obligation_v2_local_model.py -q

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.adaptive_obligation_v2_propose execute \
  --preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight \
  --approval outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/NO_APPROVAL.json \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposals
~~~

Expected: tests pass. The CLI exits nonzero with 'proposal inference approval required'; model loads, inference calls, and new output files remain zero.

- [ ] **Step 5: Commit Task 3**

~~~bash
git add code/trec_rag/adaptive_obligation_v2_ledger.py \
  code/trec_rag/adaptive_obligation_v2_local_model.py \
  code/trec_rag/adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_ledger.py \
  code/tests/test_adaptive_obligation_v2_local_model.py \
  code/tests/test_adaptive_obligation_v2_propose.py
git commit -m "Guard adaptive obligation v2 proposal execution"
~~~

---

### Task 4: Implement opposite-fold validation and deterministic acceptance

**Files:**
- Create: 'code/trec_rag/adaptive_obligation_v2_validate.py'
- Create: 'code/tests/test_adaptive_obligation_v2_validate.py'

**Interfaces:**
- Produces: 'validate_proposal_record(...)', 'build_validation_jobs(...)', 'render_validation_messages(...)', 'accept_validated_o1(...)', and 'build_validation_preflight(...)'.
- No canonical validation preflight or inference runs until proposals exist.

- [ ] **Step 1: Write failing cross-fold, decision, and cap tests**

~~~python
def test_validation_uses_only_opposite_fold_units() -> None:
    jobs = build_validation_jobs(_supported_proposals(), _contract_fixture())
    assert all(job["validation_fold"] != job["proposal_fold"] for job in jobs)
    assert all(
        unit["fold"] == job["validation_fold"]
        for job in jobs
        for unit in job["units"]
    )


def test_same_document_cannot_cross_validate() -> None:
    decision = validate_semantic_decision(
        _proposal(document_id="d1"),
        _supported_decision(unit_id="opposite-unit"),
        units={"opposite-unit": _unit(document_id="d1", fold=1)},
    )
    assert decision["accepted"] is False
    assert "distinct_document" in decision["reasons"]


def test_decisions_are_finite() -> None:
    assert set(VALIDATION_DECISIONS) == {
        "SUPPORTED",
        "NO_EVIDENCE",
        "OUT_OF_SCOPE",
        "ANSWER_FACT",
        "DUPLICATE_O0",
        "WRONG_DOMAIN",
    }


def test_acceptance_caps_one_parent_and_four_topic() -> None:
    accepted = accept_validated_o1(_six_validated_rows_one_topic())
    assert len(accepted) == 4
    assert len({row["parent_id"] for row in accepted}) == 4
~~~

- [ ] **Step 2: Run tests and verify the missing validator**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_validate.py -q
~~~

Expected: collection fails for the missing validation module.

- [ ] **Step 3: Implement deterministic validation, compact semantic jobs, and acceptance**

~~~python
import re

VALIDATION_DECISIONS = (
    "SUPPORTED",
    "NO_EVIDENCE",
    "OUT_OF_SCOPE",
    "ANSWER_FACT",
    "DUPLICATE_O0",
    "WRONG_DOMAIN",
)


def normalize_label(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def acceptance_key(row):
    return (
        -len(set(row["support_document_ids"])),
        int(row["parent_manifest_order"]),
        normalize_label(row["label"]),
        row["proposal_id"],
    )
~~~

Reject unknown unit IDs, source-fold validation units, protected topics, source identity mismatch, copied support text as a label, exact/normalized O0 duplicates, invalid finite codes, and same-document cross-validation. The validator prompt sees narrative, complete O0, proposed label, and opposite-fold units only; it never sees proposing units or rationale.

Validation preflight is two-stage: it can be built only after a complete authenticated proposal receipt. It records exact surviving job count 'V', primary calls 'V', retry ceiling 'V', worst-case '2V', and all zero execution counters. Implement and fixture-test the guarded runner using Task 3's ledger, but do not create a canonical preflight or execute it in this plan.

- [ ] **Step 4: Run Task 1-4 v2 tests and compatibility tests**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_ledger.py \
  code/tests/test_adaptive_obligation_v2_local_model.py \
  code/tests/test_adaptive_obligation_v2_validate.py \
  code/tests/test_adaptive_evidence_discovery.py \
  code/tests/test_adaptive_evidence_rank.py -q
~~~

Expected: all tests pass; no canonical proposal, validation, model, network, or qrels artifact is created.

- [ ] **Step 5: Commit Task 4**

~~~bash
git add code/trec_rag/adaptive_obligation_v2_validate.py \
  code/tests/test_adaptive_obligation_v2_validate.py
git commit -m "Add adaptive obligation v2 cross-fold validation"
~~~

---

### Task 5: Implement focused BM25 request planning and offline cache audit

**Files:**
- Create: 'code/trec_rag/adaptive_obligation_v2_retrieve.py'
- Create: 'code/tests/test_adaptive_obligation_v2_retrieve.py'

**Interfaces:**
- Produces: 'render_o1_bm25_query(...)', 'build_retrieval_jobs(...)', 'audit_retrieval_cache(...)', 'verify_retrieval_preflight(...)', and a guarded injected 'execute_retrieval(...)'.
- No canonical accepted-O1 input or external request exists in this plan.

- [ ] **Step 1: Write failing query, budget, cache, and guard tests**

~~~python
def test_bm25_query_is_focused_and_deduplicated() -> None:
    query = render_o1_bm25_query(
        anchor_terms=["deforestation", "Amazon rainforest"],
        parent_text="Effects of deforestation on the Amazon rainforest.",
        o1_label="Indigenous community displacement",
        narrative="This broad narrative must not be copied into BM25.",
    )
    assert query == (
        "deforestation Amazon rainforest Effects of deforestation on the "
        "Amazon rainforest. Indigenous community displacement"
    )
    assert "broad narrative" not in query


def test_jobs_use_one_hits_1000_request_per_accepted_o1() -> None:
    jobs = build_retrieval_jobs(_accepted_o1_rows(16), endpoint="https://example.test")
    assert len(jobs) == 16
    assert all(job["hits"] == 1000 for job in jobs)
    assert all(job["timeout_seconds"] == 120 for job in jobs)
    assert all(job["transport_retry_count"] == 0 for job in jobs)


def test_seventeenth_request_fails_before_cache_or_transport() -> None:
    touched = []
    with pytest.raises(ValueError, match="16"):
        audit_retrieval_cache(
            _accepted_o1_rows(17),
            cache_loader=lambda: touched.append("cache"),
        )
    assert touched == []


def test_execute_requires_separate_retrieval_approval(tmp_path: Path) -> None:
    touched = []
    with pytest.raises(PermissionError, match="retrieval approval required"):
        execute_retrieval(
            preflight_dir=tmp_path / "preflight",
            approval_path=tmp_path / "missing.json",
            transport_factory=lambda: touched.append("transport"),
        )
    assert touched == []
~~~

- [ ] **Step 2: Run tests and verify the missing retrieval planner**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_retrieve.py -q
~~~

Expected: collection fails for the missing retrieval module.

- [ ] **Step 3: Implement focused queries, exact identities, audit, and raw-first guard**

~~~python
RETRIEVAL_HITS = 1_000
MAX_ACCEPTED_O1_PER_TOPIC = 4
MAX_RETRIEVAL_REQUESTS = 16
TIMEOUT_SECONDS = 120
TRANSPORT_RETRY_COUNT = 0
REQUEST_START_INTERVAL_SECONDS = 3


def remove_exact_duplicate_phrases(parts):
    output = []
    seen = set()
    for part in parts:
        normalized = " ".join(str(part).split()).casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(" ".join(str(part).split()))
    return output


def render_o1_bm25_query(*, anchor_terms, parent_text, o1_label, narrative):
    del narrative
    parts = [*anchor_terms, parent_text, o1_label]
    return " ".join(remove_exact_duplicate_phrases(parts))
~~~

Request identity binds topic, parent, accepted O1, exact query text/hash, endpoint, index, retriever version, 'hits=1000', timeout, retry count, and rate-limiter identity. The audit must check the shared exact-identity cache without opening transport and report exact hits, misses, expected raw rows, disk estimate, primary external attempts, and maximum external attempts.

Reuse the tracked persistent 'requests' rate limiter and raw-first ledger patterns, not untracked sparse-relevance modules. The injected executor must append an attempt before transport, store raw bytes before JSON parsing, normalize only a valid response, preserve every provenance event, and reject unapproved execution before cache mutation or session construction. Do not call the real executor in this plan.

- [ ] **Step 4: Run focused tests and a fixture-only zero-network cache audit**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_retrieve.py \
  code/tests/test_remote_pyserini.py \
  code/tests/test_det_sparse_ledger.py -q
~~~

Expected: tests pass; fixture audit reports at most 16 requests and 16,000 raw rows; real network and qrels counters remain zero. No canonical retrieval preflight is created because accepted O1 records do not yet exist.

- [ ] **Step 5: Commit Task 5**

~~~bash
git add code/trec_rag/adaptive_obligation_v2_retrieve.py \
  code/tests/test_adaptive_obligation_v2_retrieve.py
git commit -m "Add adaptive obligation v2 retrieval planning"
~~~

---

### Task 6: Render and verify the inference-free v2 status report

**Files:**
- Create: 'code/trec_rag/build_adaptive_obligation_v2_report.py'
- Create: 'code/tests/test_build_adaptive_obligation_v2_report.py'
- Create: 'reports/experiments/adaptive_obligation_search_v2/report.html'

**Interfaces:**
- Consumes: verified v2 contract, verified proposal preflight, sealed NARRATIVE/FIXED-O0 ranking receipt, and terminal v1 receipt.
- Produces: 'build_report_payload(...)', 'render_report_html(...)', and a self-contained HTML report.

- [ ] **Step 1: Write failing report truthfulness and accessibility tests**

~~~python
def test_report_states_exact_current_status() -> None:
    payload = build_report_payload(_verified_sources())
    assert payload["status"] == "proposal_preflight_ready"
    assert payload["proposal_jobs"] == 48
    assert payload["proposal_worst_case_calls"] == 96
    assert payload["retrieval_hits_per_accepted_o1"] == 1000
    assert payload["maximum_retrieval_requests"] == 16
    assert payload["qwen_calls_completed"] == 0
    assert payload["retrieval_calls_completed"] == 0
    assert payload["qrels_opened"] is False


def test_report_never_claims_adaptive_results() -> None:
    html = render_report_html(build_report_payload(_verified_sources()))
    assert "No adaptive relevance result exists yet" in html
    assert "proposal_preflight_ready" in html
    assert "ADAPTIVE-V2 improved" not in html


def test_report_has_accessible_structure() -> None:
    html = render_report_html(build_report_payload(_verified_sources()))
    assert '<main id="main-content">' in html
    assert 'aria-label="V2 stage status"' in html
    assert "@media (max-width: 720px)" in html
~~~

- [ ] **Step 2: Run tests and verify the missing report builder**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_build_adaptive_obligation_v2_report.py -q
~~~

Expected: collection fails for the missing report builder.

- [ ] **Step 3: Implement the source-backed self-contained report**

The report must lead with:

~~~text
Current result: the fixed baselines are sealed; adaptive v2 has not run.
What is ready: 48 exact proposal jobs and a worst-case 96-call proposal ceiling.
What needs separate approval: Qwen proposals, opposite-fold validation,
up to 16 hits=1000 BM25 requests, O1 MiniLM scoring, and qrels.
~~~

~~~python
def build_report_payload(sources):
    contract = verify_v2_contract(sources["contract"])
    proposal = verify_proposal_preflight(sources["proposal_preflight"])
    baselines = verify_baseline_rankings(sources["baseline_rankings"])
    terminal_v1 = verify_discovery_terminal(sources["v1_discovery"])
    return {
        "status": "proposal_preflight_ready",
        "parents": contract["parent_count"],
        "reservoirs": contract["reservoir_count"],
        "proposal_jobs": proposal["primary_call_count"],
        "proposal_worst_case_calls": proposal["worst_case_call_ceiling"],
        "retrieval_hits_per_accepted_o1": 1000,
        "maximum_retrieval_requests": 16,
        "baseline_documents": baselines["document_count"],
        "v1_status": terminal_v1["status"],
        "qwen_calls_completed": 0,
        "retrieval_calls_completed": 0,
        "qrels_opened": False,
    }


def render_report_html(payload):
    status = html.escape(str(payload["status"]))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Adaptive obligation search v2</title>"
        "<style>body{margin:0;font:16px/1.5 system-ui;color:#172033;"
        "background:#f5f7fb}main{max-width:1100px;margin:auto;padding:24px}"
        ".card{background:white;border:1px solid #d8deea;border-radius:12px;"
        "padding:18px;margin:16px 0}table{width:100%;border-collapse:collapse}"
        "th,td{padding:8px;border-bottom:1px solid #dde3ee;text-align:left}"
        "@media (max-width:720px){main{padding:12px}.scroll{overflow-x:auto}}"
        "</style></head><body><main id=\"main-content\">"
        "<h1>Adaptive obligation search v2</h1>"
        "<section class=\"card\" aria-label=\"V2 stage status\">"
        "<h2>Current result</h2><p>No adaptive relevance result exists yet.</p>"
        "<p>Status: <strong>" + status + "</strong></p></section>"
        "<section class=\"card\"><h2>Approval gates</h2>"
        "<p>Qwen, BM25 retrieval, MiniLM inference, and qrels remain closed.</p>"
        "</section></main></body></html>"
    )
~~~

Include a responsive pipeline diagram, exact counts/hashes, the focused BM25 versus semantic MiniLM query distinction, top-100/top-1,000 from one response, retry ceilings, failure states, and links/paths to local verified sources. Clearly define O0, O1, fold, raw-first, and query-local scoring. Use no external JavaScript, font, stylesheet, analytics, or network resource.

- [ ] **Step 4: Run all v2 tests, build the report, and inspect desktop/mobile rendering**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_contract.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_ledger.py \
  code/tests/test_adaptive_obligation_v2_local_model.py \
  code/tests/test_adaptive_obligation_v2_validate.py \
  code/tests/test_adaptive_obligation_v2_retrieve.py \
  code/tests/test_build_adaptive_obligation_v2_report.py -q

TMPDIR=/var/tmp .venv/bin/python -m trec_rag.build_adaptive_obligation_v2_report \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract \
  --proposal-preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight \
  --baseline-rankings outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings \
  --v1-discovery outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery \
  --output reports/experiments/adaptive_obligation_search_v2/report.html
~~~

Use headless Chrome or Playwright at 1440x900 and 390x844. Verify no horizontal overflow, clipped tables, unreadable text, keyboard-inaccessible disclosure, broken local references, or console errors. Keep screenshots in ignored scratch only.

- [ ] **Step 5: Run final repository verification for this implementation boundary**

Run:

~~~bash
TMPDIR=/var/tmp .venv/bin/python -m pytest code/tests -q
TMPDIR=/var/tmp .venv/bin/python -m compileall -q code/trec_rag code/tests
git diff --check
git status --short
~~~

Expected: all tracked tests pass; unrelated untracked sparse-relevance files remain untouched; canonical v2 status is proposal-preflight-ready; no model, inference, retrieval, network, qrels, or paid call occurred.

- [ ] **Step 6: Commit Task 6**

~~~bash
git add code/trec_rag/build_adaptive_obligation_v2_report.py \
  code/tests/test_build_adaptive_obligation_v2_report.py \
  reports/experiments/adaptive_obligation_search_v2/report.html
git commit -m "Render adaptive obligation v2 preflight report"
~~~

---

## Final Verification and Handoff

Before claiming this plan complete:

1. Independently verify the v2 contract and proposal preflight.
2. Confirm exact counts: 24 parents, 48 reservoirs, 48 proposal jobs, primary 48, retry ceiling 48, worst-case 96.
3. Confirm v1 terminal receipt and both 8,114-row baseline rankings still verify unchanged.
4. Confirm the v2 proposal output, validation preflight, accepted O1 output, retrieval preflight/output, O1 score output, adaptive ranking, and qrels/evaluation artifacts do not exist.
5. Confirm all safety counters are zero and no approval file exists.
6. Run a broad whole-branch code/spec review with the exact diff package.
7. Use 'superpowers:finishing-a-development-branch' only after all task reviews and the final review are clean.

The handoff must present the rendered HTML and the exact proposal preflight. It must ask for a separate proposal-inference approval before any Qwen call.

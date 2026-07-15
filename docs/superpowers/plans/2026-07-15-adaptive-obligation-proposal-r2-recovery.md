# Adaptive Obligation Proposal R2 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the immutable failed R1 proposal attempt, freeze a tail-reinforced 48-job R2 proposal preflight, and stop before R2 approval or inference.

**Architecture:** A focused incident module authenticates and records the aborted R1 run without modifying its ledger. Separate R2 proposal and local-model modules keep the R1 verifier intact while versioning prompt, job, preflight, approval, and proposal receipt identities on top of the stable ledger schema. A new report shows the failed R1 call and inference-free R2 readiness; downstream validation stays explicitly unavailable pending a separate end-to-end provenance plan.

**Tech Stack:** Python 3.12, pytest, canonical JSON/JSONL, SHA-256, existing append-only ledger, local Hugging Face tokenizer/model snapshot, ROCm runtime adapter, and the canonical Data Analytics portable HTML renderer.

## Global Constraints

- Work only in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on branch `codex/structured-query-planner`.
- Preserve the R1 preflight, approval, ledger, events, and raw completion byte-for-byte.
- R1 contains one terminal `schema_error`, one attempted job, and 47 uncalled jobs. Never seal, resume, retry, reclassify, truncate, normalize, or accept that output.
- R2 pilot topics are exactly `219`, `72`, `300`, and `84`.
- Reject protected topics `144`, `213`, `224`, `407`, and `515` before source, cache, tokenizer, model, endpoint, qrels, or output access.
- Keep the accepted proposal schema unchanged: label maximum 120 characters and rationale maximum 240 characters.
- R2 generation targets are stricter guidance: label at most 80 characters and 10 words; rationale exactly one sentence at most 160 characters and 25 words.
- Preserve model `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, deterministic decoding, 256 primary output tokens, and one 512-token retry only for incomplete JSON at the 256-token ceiling.
- Freeze exactly 48 R2 primary jobs, a retry ceiling of 48, and a worst-case ceiling of 96.
- Use new create-only destinations `proposal_run_r1_incident/`, `proposal_preflight_r2/`, `proposal_approval_r2.json`, `proposal_ledger_r2/`, and `proposals_r2/`. Freeze the absolute ledger and proposal destinations inside the R2 preflight; bind the ledger again in approval.
- Do not create `proposal_approval_r2.json`, construct model weights, run R2 inference, make retrieval/network/paid calls, open qrels, or touch known-five topics in this plan.
- Do not modify the stable ledger schema or stage vocabulary: R2 uses ledger anchor schema `adaptive-obligation-v2-run-anchor-v1` and attempt stage `proposal`; its distinct job, preflight, approval, and receipt hashes provide authenticated R2 identity.
- Keep R2 validation unavailable in this recovery plan. Versioning proposal provenance through the validation preflight, approval, execution, finalizer, and receipt is a separate subsystem and receives its own plan only after a complete R2 proposal inventory exists.
- Use `TMPDIR=/var/tmp` for Python commands.
- Leave unrelated untracked `sparse_relevance*` files untouched and out of every commit.

## File Map

- Create `code/trec_rag/adaptive_obligation_v2_proposal_incident.py`: authenticate and freeze the external R1 incident receipt.
- Create `code/tests/test_adaptive_obligation_v2_proposal_incident.py`: incident TDD and real-fixture verification.
- Create `code/trec_rag/adaptive_obligation_v2_propose_r2.py`: R2 prompt, jobs, preflight, approval boundary, executor, and proposal finalizer.
- Create `code/tests/test_adaptive_obligation_v2_propose_r2.py`: R2 prompt/preflight/executor/finalizer tests.
- Create `code/trec_rag/adaptive_obligation_v2_local_model_r2.py`: approval-gated R2 facade over the existing pinned runtime.
- Create `code/tests/test_adaptive_obligation_v2_local_model_r2.py`: R2 model-boundary tests.
- Create `code/trec_rag/build_adaptive_obligation_v2_r2_report.py`: verified R1-incident/R2-preflight report builder.
- Create `code/tests/test_build_adaptive_obligation_v2_r2_report.py`: report truth-boundary and rendering tests.
- Create `reports/experiments/adaptive_obligation_search_v2_proposal_r2/artifact.json` and `report.html`: rendered status artifact.

---

### Task 1: Freeze the immutable R1 incident receipt

**Files:**
- Create: `code/trec_rag/adaptive_obligation_v2_proposal_incident.py`
- Create: `code/tests/test_adaptive_obligation_v2_proposal_incident.py`
- Create at runtime: `outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident/receipt.json`

**Interfaces:**
- Consumes: `verify_proposal_preflight(...)`, `_capture_inference_approval(...)`, `_build_run_anchor(...)`, `classify_completion(...)`, and the canonical R1 proposal schema. It deliberately does not reopen the failed ledger through `AppendOnlyAttemptLedger`.
- Produces: `build_r1_incident_receipt(...) -> dict[str, object]`, `publish_r1_incident(...) -> dict[str, object]`, and `verify_r1_incident(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for exact incident replay and immutability**

```python
def test_incident_replays_exact_terminal_r1_failure(r1_failure_fixture) -> None:
    receipt = build_r1_incident_receipt(**r1_failure_fixture.paths)
    assert receipt["schema_version"] == "adaptive-obligation-v2-proposal-incident-r1"
    assert receipt["status"] == "aborted"
    assert receipt["attempted_job_count"] == 1
    assert receipt["terminal_schema_error_count"] == 1
    assert receipt["uncalled_job_count"] == 47
    assert receipt["reason_code"] == "scope_rationale_length_exceeded"
    assert receipt["observed_rationale_characters"] == 245
    assert receipt["accepted_rationale_maximum"] == 240


@pytest.mark.parametrize("mutation", ["approval", "anchor", "event", "raw"])
def test_incident_rejects_changed_r1_bytes(r1_failure_fixture, mutation) -> None:
    r1_failure_fixture.mutate(mutation)
    with pytest.raises(ValueError, match="R1 incident"):
        build_r1_incident_receipt(**r1_failure_fixture.paths)


def test_incident_never_appends_or_seals_r1_ledger(r1_failure_fixture) -> None:
    before = r1_failure_fixture.snapshot_ledger()
    build_r1_incident_receipt(**r1_failure_fixture.paths)
    assert r1_failure_fixture.snapshot_ledger() == before


@pytest.mark.parametrize("mutation", ["completion", "extra_root", "extra_raw", "lock"])
def test_incident_rejects_noncanonical_failed_ledger_inventory(
    r1_failure_fixture, mutation
) -> None:
    r1_failure_fixture.mutate_ledger_inventory(mutation)
    with pytest.raises(ValueError, match="R1 failed ledger"):
        build_r1_incident_receipt(**r1_failure_fixture.paths)


def test_sole_defect_check_rejects_compound_invalid_245_character_result() -> None:
    raw, allowed_ids = _compound_invalid_245_character_result()
    with pytest.raises(ValueError, match="sole schema defect"):
        _prove_rationale_length_is_sole_defect(raw, allowed_ids)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py -q
```

Expected: collection fails because `adaptive_obligation_v2_proposal_incident` does not exist.

- [ ] **Step 3: Implement descriptor-safe incident replay and create-only publication**

```python
INCIDENT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-incident-r1"
EXPECTED_JOB_COUNT = 48
R1_FAILED_LEDGER_ROOT_NAMES = frozenset(
    {"events.jsonl", "head.json", "anchor.json", "mutation.lock", "raw"}
)
R1_RAW_BYTES = 546
R1_RAW_SHA256 = "683174f0ca3353b8dc901ec889974d83b2c02bac9cc79d00db6a08f26e4bb667"
R1_OUTPUT_TOKENS = 186


def _prove_rationale_length_is_sole_defect(
    raw: bytes, allowed_support_unit_ids: Sequence[str]
) -> str:
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("o1"), dict):
        raise ValueError("rationale length is not the sole schema defect")
    rationale = value["o1"].get("scope_rationale")
    if not isinstance(rationale, str) or len(rationale) != 245:
        raise ValueError("rationale length is not the sole schema defect")
    original = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=R1_OUTPUT_TOKENS,
        max_new_tokens=PRIMARY_MAX_NEW_TOKENS,
        allowed_support_unit_ids=allowed_support_unit_ids,
    )
    diagnostic_value = deepcopy(value)
    diagnostic_value["o1"]["scope_rationale"] = "Within the supported scope."
    diagnostic_raw = _compact_bytes(diagnostic_value).rstrip(b"\n")
    diagnostic = classify_completion(
        diagnostic_raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=R1_OUTPUT_TOKENS,
        max_new_tokens=PRIMARY_MAX_NEW_TOKENS,
        allowed_support_unit_ids=allowed_support_unit_ids,
    )
    if (
        original.get("classification") != "schema_error"
        or diagnostic.get("classification") != "valid"
    ):
        raise ValueError("rationale length is not the sole schema defect")
    return rationale


def build_r1_incident_receipt(
    *, preflight_dir: Path, approval_path: Path, ledger_dir: Path
) -> dict[str, object]:
    approval = _capture_inference_approval(approval_path)
    captured = _capture_and_verify_preflight(
        preflight_dir,
        expected_receipt_sha256=str(approval.value["preflight_sha256"]),
    )
    anchor = _build_run_anchor(
        captured.jobs,
        preflight_sha256=captured.receipt_sha256,
        approval_sha256=approval.sha256,
    )
    failed = _capture_failed_r1_ledger(ledger_dir)
    if failed["anchor"] != anchor:
        raise ValueError("R1 incident anchor differs")
    events = failed["events"]
    if len(events) != 2:
        raise ValueError("R1 incident event count differs")
    started, terminal = events
    if (
        started.get("state") != "started"
        or terminal.get("state") != "terminal"
        or terminal.get("classification") != "schema_error"
        or started.get("job_id") != captured.jobs[0]["job_id"]
        or terminal.get("job_id") != captured.jobs[0]["job_id"]
    ):
        raise ValueError("R1 incident terminal state differs")
    raw = failed["raw"]
    if (
        terminal.get("raw_bytes") != R1_RAW_BYTES
        or terminal.get("raw_sha256") != R1_RAW_SHA256
        or terminal.get("output_token_count") != R1_OUTPUT_TOKENS
        or len(raw) != R1_RAW_BYTES
        or sha256(raw).hexdigest() != R1_RAW_SHA256
    ):
        raise ValueError("R1 incident raw identity differs")
    rationale = _prove_rationale_length_is_sole_defect(
        raw, captured.jobs[0]["input_unit_ids"]
    )
    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "status": "aborted",
        "reason_code": "scope_rationale_length_exceeded",
        "attempted_job_count": 1,
        "terminal_schema_error_count": 1,
        "uncalled_job_count": 47,
        "observed_rationale_characters": len(rationale),
        "accepted_rationale_maximum": 240,
        "output_token_count": R1_OUTPUT_TOKENS,
        "preflight": _file_binding(preflight_dir / "receipt.json"),
        "approval": _file_binding(approval_path),
        "ledger": _failed_ledger_binding(ledger_dir, failed),
        "raw_completion": _raw_binding(terminal, raw),
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "qrels_opened": False,
    }
```

`_capture_failed_r1_ledger` is a standalone read-only verifier; it must not call
`AppendOnlyAttemptLedger`, whose reopen contract correctly requires a sealed
completion. Open the ledger root once with `O_DIRECTORY|O_NOFOLLOW`, require
exactly `R1_FAILED_LEDGER_ROOT_NAMES`, require a zero-byte regular single-link
`mutation.lock`, require exactly one regular single-link file under `raw/`, and
capture stable canonical bytes for anchor, head, events, and raw. Recompute both
event hashes and their previous-hash chain, require head count/hash equality,
bind each event request to the matching anchor attempt, and reject any
`completion.json` or extra entry. `_file_binding`, `_failed_ledger_binding`, and
`_raw_binding` return `{path, bytes, sha256}` records using the already captured
bytes. Publish only `receipt.json` through a sibling staging directory, fsync
it, and rename without replacement. `verify_r1_incident` rebuilds the receipt
and compares canonical bytes.

- [ ] **Step 4: Run focused and R1 compatibility tests**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_ledger.py -q
```

Expected: all tests pass; the synthetic R1 ledger remains unsealed and unchanged.

- [ ] **Step 5: Commit Task 1**

```bash
git add code/trec_rag/adaptive_obligation_v2_proposal_incident.py \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py
git commit -m "Record adaptive proposal R1 incident"
```

---

### Task 2: Freeze the tail-reinforced R2 prompt and preflight

**Files:**
- Create: `code/trec_rag/adaptive_obligation_v2_propose_r2.py`
- Create: `code/tests/test_adaptive_obligation_v2_propose_r2.py`
- Create at runtime: `outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2/`

**Interfaces:**
- Consumes: verified v2 contract, unchanged `PROPOSAL_SCHEMA`, pinned model snapshot, and local tokenizer.
- Produces: `R2_OUTPUT_CONTRACT`, `render_r2_proposal_messages(...)`, `build_r2_proposal_jobs(...)`, `_build_authenticated_r2_preflight(...)`, production-only `build_r2_proposal_preflight(*, contract_dir, output_dir, ledger_dir, proposal_dir)`, `publish_r2_proposal_preflight(...)`, and `verify_r2_proposal_preflight(...)`.

- [ ] **Step 1: Write failing prompt-tail, boundary, and preflight tests**

```python
def test_r2_output_contract_is_last_user_payload_field() -> None:
    messages = render_r2_proposal_messages(
        _parent(), _reservoir(), _evidence_units()
    )
    payload = json.loads(messages[1]["content"])
    assert list(payload)[-1] == "output_contract"
    assert payload["output_contract"] == {
        "return": "exactly one JSON object and no prose",
        "label": {"max_unicode_characters": 80, "max_words": 10},
        "scope_rationale": {
            "exact_sentences": 1,
            "max_unicode_characters": 160,
            "max_words": 25,
        },
        "support_unit_ids_for_supported": {
            "minimum_items": 1,
            "maximum_items": 2,
            "source": "supplied evidence_units only",
        },
        "self_check": "silently verify every limit before emitting JSON",
        "on_failure": "return a schema-valid UNSUPPORTED object",
    }
    assert messages[1]["content"].endswith(
        '"output_contract":' + _ordered_compact(R2_OUTPUT_CONTRACT) + "}"
    )


def test_r2_preserves_schema_and_all_48_coverage_boundaries() -> None:
    r1 = build_proposal_jobs(_contract_fixture())
    r2 = build_r2_proposal_jobs(_contract_fixture())
    assert len(r2) == 48
    assert [(j["topic_id"], j["parent_id"], j["fold"]) for j in r2] == [
        (j["topic_id"], j["parent_id"], j["fold"]) for j in r1
    ]
    assert [j["input_unit_ids"] for j in r2] == [j["input_unit_ids"] for j in r1]
    assert R2_PROPOSAL_SCHEMA == PROPOSAL_SCHEMA


def test_r2_preflight_is_tokenizer_only_and_versioned() -> None:
    receipt = _build_authenticated_r2_preflight(
        _authenticated_contract_fixture(),
        tokenizer=_fake_tokenizer(),
        model_snapshot=_snapshot_fixture(),
        **_absolute_destination_bindings(),
    )
    assert receipt["schema_version"] == "adaptive-obligation-v2-proposal-preflight-r2"
    assert receipt["prompt_revision"] == "tail-contract-r2"
    assert receipt["job_count"] == 48
    assert receipt["tokenizer_load_count"] == 1
    assert receipt["model_load_count"] == 0
    assert receipt["inference_count"] == 0


def test_public_preflight_has_no_tokenizer_or_model_injection() -> None:
    assert list(inspect.signature(build_r2_proposal_preflight).parameters) == [
        "contract_dir",
        "output_dir",
        "ledger_dir",
        "proposal_dir",
    ]


def test_r2_rejects_protected_topic_before_tokenizer_access(
    tmp_path, monkeypatch
) -> None:
    touched = []
    contract_dir = _published_contract_with_topic(tmp_path, "144")
    monkeypatch.setattr(
        r2_module,
        "_load_pinned_tokenizer_after_auth",
        lambda **_kwargs: touched.append("tokenizer"),
    )
    with pytest.raises(ValueError, match="protected topic"):
        build_r2_proposal_preflight(
            contract_dir=contract_dir,
            **_absolute_destination_bindings(tmp_path),
        )
    assert touched == []


def test_r2_preflight_rejects_relative_destinations(tmp_path) -> None:
    for changed in ("ledger_dir", "proposal_dir"):
        bindings = _absolute_destination_bindings(tmp_path)
        bindings[changed] = Path("relative-output")
        with pytest.raises(ValueError, match="absolute safe destination"):
            _build_authenticated_r2_preflight(
                _authenticated_contract_fixture(),
                tokenizer=_fake_tokenizer(),
                model_snapshot=_snapshot_fixture(),
                **bindings,
            )


@pytest.mark.parametrize("changed", ["ledger_dir", "proposal_dir"])
@pytest.mark.parametrize("leaf_kind", ["directory", "file", "symlink"])
def test_r2_preflight_requires_absent_create_only_destination_leaves(
    tmp_path, changed, leaf_kind
) -> None:
    bindings = _absolute_destination_bindings(tmp_path)
    _create_destination_leaf(bindings[changed], leaf_kind)
    with pytest.raises(FileExistsError, match="create-only destination exists"):
        _build_authenticated_r2_preflight(
            _authenticated_contract_fixture(),
            tokenizer=_fake_tokenizer(),
            model_snapshot=_snapshot_fixture(),
            **bindings,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_propose_r2.py -q
```

Expected: collection fails because `adaptive_obligation_v2_propose_r2` does not exist.

- [ ] **Step 3: Implement the ordered R2 payload and versioned artifacts**

```python
R2_PREFLIGHT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-preflight-r2"
R2_JOB_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-job-r2"
R2_PROMPT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-prompt-r2"
R2_PROMPT_REVISION = "tail-contract-r2"
R2_PROPOSAL_SCHEMA = deepcopy(PROPOSAL_SCHEMA)
R2_SYSTEM_INSTRUCTIONS = _PROPOSAL_INSTRUCTIONS + """
Before emitting JSON, check the final output_contract in the user payload.
Apply its exact label and scope_rationale limits. If you cannot satisfy that
contract from the supplied evidence, return a schema-valid UNSUPPORTED object."""
R2_OUTPUT_CONTRACT = {
    "return": "exactly one JSON object and no prose",
    "label": {"max_unicode_characters": 80, "max_words": 10},
    "scope_rationale": {
        "exact_sentences": 1,
        "max_unicode_characters": 160,
        "max_words": 25,
    },
    "support_unit_ids_for_supported": {
        "minimum_items": 1,
        "maximum_items": 2,
        "source": "supplied evidence_units only",
    },
    "self_check": "silently verify every limit before emitting JSON",
    "on_failure": "return a schema-valid UNSUPPORTED object",
}


def _ordered_compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def render_r2_proposal_messages(parent, reservoir, units):
    text = str(parent["text"])
    query = str(parent["query"])
    suffix = f"\n\nExplicit obligation:\n{text}"
    if not query.endswith(suffix):
        raise ValueError("parent query does not preserve the complete O0 suffix")
    payload = {
        "narrative": query[:-len(suffix)],
        "parent_o0": dict(parent),
        "source_fold": reservoir["fold"],
        "reservoir_id": reservoir["reservoir_id"],
        "evidence_units": [dict(unit) for unit in units],
        "response_json_schema": R2_PROPOSAL_SCHEMA,
        "output_contract": R2_OUTPUT_CONTRACT,
    }
    return [
        {"role": "system", "content": R2_SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
        },
    ]
```

Build 48 jobs from the authenticated contract rather than transforming R1
jobs. `_build_authenticated_r2_preflight` is the underscore-only unit-test seam
and accepts already authenticated sources plus a tokenizer. The public builder
first authenticates the contract and model snapshot, rejects protected topics,
verifies absolute `ledger_dir` and `proposal_dir` strings, requires both leaf
paths to be absent even when they are broken symlinks, and verifies their
existing parents by descriptor walk without following symlinks, then loads the pinned
tokenizer internally and calls the private builder. Freeze those exact absolute
destinations with `jobs.jsonl`, `schema.json`, `prompt.json`, and `receipt.json`
in a create-only directory. Publication writes a sibling staging directory,
fsyncs it, runs `verify_r2_proposal_preflight` against the staged contents, and
only then performs the atomic no-replace rename and parent fsync. A failed
staging verification removes staging and cannot strand the canonical leaf. It
may not publish a receipt produced with the private fake-tokenizer seam. The verifier
reconstructs the contract-derived jobs, checks every prompt token count with
the pinned tokenizer, verifies the model snapshot and tokenizer file inventory,
and rejects R1 schema versions.

- [ ] **Step 4: Run R2 and R1 compatibility tests**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_propose_r2.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_contract.py -q
```

Expected: all pass; the existing R1 canonical preflight verifies byte-identically.

- [ ] **Step 5: Commit Task 2**

```bash
git add code/trec_rag/adaptive_obligation_v2_propose_r2.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py
git commit -m "Add adaptive proposal R2 preflight"
```

---

### Task 3: Add the separately gated R2 executor and finalizer

**Files:**
- Create: `code/trec_rag/adaptive_obligation_v2_local_model_r2.py`
- Create: `code/tests/test_adaptive_obligation_v2_local_model_r2.py`
- Modify: `code/trec_rag/adaptive_obligation_v2_propose_r2.py`
- Modify: `code/tests/test_adaptive_obligation_v2_propose_r2.py`

**Interfaces:**
- Produces: `verify_r2_inference_approval(...)`, `R2LocalJsonModel`, `run_r2_job_with_retry(...)`, `_execute_authenticated_r2_jobs(...)`, `execute_r2_proposals(...)`, `finalize_r2_proposal_inventory(...)`, and `load_authenticated_r2_proposal_inventory(...)`.
- Reuses: `load_pinned_local_runtime(...)`, `AppendOnlyAttemptLedger`, `AttemptSpec`, and the unchanged proposal classifier.

- [ ] **Step 1: Write failing approval, fail-fast, retry, and sealed-finalizer tests**

```python
def test_r1_approval_cannot_authorize_r2(tmp_path, monkeypatch) -> None:
    touched = []
    monkeypatch.setattr(
        r2_module,
        "_load_r2_model_after_approval",
        lambda **_kwargs: touched.append("model"),
    )
    with pytest.raises(PermissionError, match="R2 proposal approval required"):
        execute_r2_proposals(
            preflight_dir=tmp_path / "r2-preflight",
            approval_path=_r1_approval(tmp_path),
            ledger_dir=tmp_path / "r2-ledger",
        )
    assert touched == []
    assert not (tmp_path / "r2-ledger").exists()


def test_r2_approval_binds_absolute_ledger_destination(tmp_path) -> None:
    preflight = _r2_preflight_fixture()
    approval = _r2_approval(preflight, tmp_path / "approved-ledger")
    with pytest.raises(PermissionError, match="ledger"):
        verify_r2_inference_approval(
            approval,
            preflight,
            ledger_dir=tmp_path / "different-ledger",
        )


def test_r2_approval_rejects_relative_or_symlinked_ledger_destination(
    tmp_path,
) -> None:
    preflight = _r2_preflight_fixture()
    with pytest.raises(PermissionError, match="absolute safe ledger"):
        verify_r2_inference_approval(
            _r2_approval(preflight, Path("relative-ledger")),
            preflight,
            ledger_dir=Path("relative-ledger"),
        )


def test_first_schema_error_leaves_47_jobs_uncalled(tmp_path) -> None:
    model = _sequence_model([_too_long_rationale_completion()])
    with pytest.raises(ValueError, match="scope_rationale"):
        _execute_authenticated_r2_jobs(
            _r2_preflight_fixture(), ledger=_real_r2_ledger(tmp_path), model=model
        )
    assert model.calls == 1


def test_only_exact_ceiling_incomplete_json_retries(tmp_path) -> None:
    model = _sequence_model([
        (_incomplete_json_completion(), 256),
        (_valid_completion(), 143),
    ])
    result = run_r2_job_with_retry(
        _r2_job_fixture(), generate=model.generate, ledger=_real_r2_ledger(tmp_path)
    )
    assert result == _valid_completion_value()
    assert model.calls == 2


def test_incomplete_json_below_ceiling_is_terminal(tmp_path) -> None:
    model = _sequence_model([(_incomplete_json_completion(), 255)])
    with pytest.raises(ValueError, match="JSON"):
        run_r2_job_with_retry(
            _r2_job_fixture(), generate=model.generate,
            ledger=_real_r2_ledger(tmp_path),
        )
    assert model.calls == 1


def test_public_executor_exposes_no_runtime_injection() -> None:
    assert list(inspect.signature(execute_r2_proposals).parameters) == [
        "preflight_dir",
        "approval_path",
        "ledger_dir",
    ]


def test_r2_model_rejects_duplicate_message_hash_before_runtime_load(
    tmp_path, monkeypatch
) -> None:
    touched = []
    preflight = _r2_preflight_fixture_with_duplicate_messages()
    monkeypatch.setattr(
        local_model_r2,
        "load_pinned_local_runtime",
        lambda **_kwargs: touched.append("runtime"),
    )
    with pytest.raises(ValueError, match="unique frozen message hashes"):
        R2LocalJsonModel(
            approval=_r2_approval(preflight, tmp_path / "ledger"),
            preflight=preflight,
            ledger_dir=tmp_path / "ledger",
        )
    assert touched == []


def test_r2_model_passes_exact_frozen_prompt_count(fake_runtime, approved_model) -> None:
    job = approved_model.preflight["jobs"][0]
    approved_model.generate(
        job["messages"], R2_PROPOSAL_SCHEMA, max_new_tokens=256
    )
    assert fake_runtime.calls[0]["expected_prompt_tokens"] == (
        job["prompt_token_count"]
    )


def test_complete_fake_r2_run_seals_48_results(tmp_path) -> None:
    result = _execute_authenticated_r2_jobs(
        _r2_preflight_fixture(),
        ledger=_real_r2_ledger(tmp_path),
        model=_valid_fake_model(),
    )
    assert result["job_count"] == 48
    assert len(result["results"]) == 48
    assert _reopen_r2_ledger(tmp_path).read_sealed_results()["results"] == (
        result["results"]
    )


def test_finalizer_rejects_nonfrozen_proposal_destination(tmp_path) -> None:
    frozen = _complete_r2_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="frozen proposal destination"):
        finalize_r2_proposal_inventory(
            preflight_dir=frozen.preflight_dir,
            approval_path=frozen.approval_path,
            ledger_dir=frozen.ledger_dir,
            output_dir=tmp_path / "different-proposals-r2",
        )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_local_model_r2.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py -q
```

Expected: failures for missing R2 approval, model, executor, and finalizer interfaces.

- [ ] **Step 3: Implement exact R2 approval and runtime boundaries**

```python
R2_APPROVAL_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-approval-r2"


def verify_r2_inference_approval(approval, preflight, *, ledger_dir: Path):
    if not ledger_dir.is_absolute() or preflight.get("ledger_dir") != str(ledger_dir):
        raise PermissionError("R2 proposal requires an absolute safe ledger destination")
    _verify_existing_parent_without_symlinks(ledger_dir)
    required = {
        "schema_version": R2_APPROVAL_SCHEMA_VERSION,
        "stage": "proposal_r2",
        "preflight_sha256": preflight["receipt_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "ledger_dir": str(ledger_dir),
        "approved": True,
    }
    if not isinstance(approval, Mapping) or dict(approval) != required:
        raise PermissionError("R2 proposal approval required")
    return dict(approval)


class R2LocalJsonModel:
    def __init__(self, *, approval, preflight, ledger_dir: Path) -> None:
        verify_r2_inference_approval(
            approval, preflight, ledger_dir=ledger_dir
        )
        prompt_counts = {}
        for job in preflight["jobs"]:
            messages_sha256 = canonical_sha256(job["messages"])
            prompt_token_count = job["prompt_token_count"]
            if (
                messages_sha256 in prompt_counts
                or type(prompt_token_count) is not int
                or prompt_token_count <= 0
            ):
                raise ValueError("R2 requires unique frozen message hashes and counts")
            prompt_counts[messages_sha256] = prompt_token_count
        self._prompt_counts = prompt_counts
        self._runtime = load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            model_manifest=preflight["model_snapshot"],
        )

    def generate(self, messages, schema, *, max_new_tokens):
        if canonical_sha256(schema) != canonical_sha256(R2_PROPOSAL_SCHEMA):
            raise ValueError("R2 proposal schema differs")
        prompt_token_count = self._prompt_counts.get(canonical_sha256(messages))
        if prompt_token_count is None:
            raise ValueError("R2 proposal messages differ from frozen jobs")
        return self._runtime.generate_json_bytes(
            messages,
            schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            expected_prompt_tokens=prompt_token_count,
        )


def _load_r2_model_after_approval(*, approval, preflight, ledger_dir):
    return R2LocalJsonModel(
        approval=approval,
        preflight=preflight,
        ledger_dir=ledger_dir,
    )


def run_r2_job_with_retry(job, *, generate, ledger):
    if R2_PROPOSAL_SCHEMA != PROPOSAL_SCHEMA:
        raise RuntimeError("R2 accepted proposal schema differs from R1")
    return run_job_with_retry(job, generate=generate, ledger=ledger)


def _execute_authenticated_r2_jobs(preflight, *, ledger, model):
    generate_method = getattr(model, "generate", None)
    if not callable(generate_method):
        raise TypeError("R2 proposal model must expose generate")
    results = []
    for job in preflight["jobs"]:
        def generate(ceiling, *, frozen_job=job):
            return generate_method(
                frozen_job["messages"],
                R2_PROPOSAL_SCHEMA,
                max_new_tokens=ceiling,
            )
        results.append(
            run_r2_job_with_retry(job, generate=generate, ledger=ledger)
        )
    completion = ledger.seal_completion()
    return {
        "status": "complete",
        "job_count": len(results),
        "results": results,
        "event_count": len(ledger.read_events()),
        "completion": completion,
    }


def execute_r2_proposals(*, preflight_dir, approval_path, ledger_dir):
    captured_approval = _capture_r2_inference_approval(approval_path)
    captured_preflight = _capture_and_verify_r2_preflight(
        preflight_dir,
        expected_receipt_sha256=str(
            captured_approval.value["preflight_sha256"]
        ),
    )
    preflight = {
        **captured_preflight.receipt,
        "receipt_sha256": captured_preflight.receipt_sha256,
        "jobs": captured_preflight.jobs,
    }
    verify_r2_inference_approval(
        captured_approval.value,
        preflight,
        ledger_dir=ledger_dir,
    )
    anchor = _build_run_anchor(
        captured_preflight.jobs,
        preflight_sha256=captured_preflight.receipt_sha256,
        approval_sha256=captured_approval.sha256,
    )
    ledger = AppendOnlyAttemptLedger(
        ledger_dir,
        expected_anchor=anchor,
        create_only=True,
    )
    model = _load_r2_model_after_approval(
        approval=captured_approval.value,
        preflight=preflight,
        ledger_dir=ledger_dir,
    )
    return _execute_authenticated_r2_jobs(preflight, ledger=ledger, model=model)
```

`_capture_r2_inference_approval` first performs a descriptor-safe, no-symlink
read and rejects every schema/stage except R2 before any preflight or output
access. `_capture_and_verify_r2_preflight` reconstructs the 48 frozen jobs and
returns the same captured-receipt shape used above. Reuse `_build_run_anchor`
unchanged: the stable ledger schema remains
`adaptive-obligation-v2-run-anchor-v1`, attempt stage remains `proposal`, and
the R2 job inventory, preflight, and approval hashes prevent substitution. The
approval stage remains `proposal_r2`; it is an approval-domain value, not a
ledger stage. The private model loader is the only test seam; the public
signature is fixed by the test. The inherited retry classifier remains
unchanged.

The R2 finalizer reopens only a sealed R2 ledger, binds every result to its R2
job and exact support units, requires its output directory to equal the frozen
absolute `proposal_dir`, and publishes `proposals.jsonl` plus `receipt.json`
under proposal receipt schema `adaptive-obligation-v2-proposal-receipt-r2`.

- [ ] **Step 4: Run R2, ledger, local-model, and R1 compatibility tests**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_propose_r2.py \
  code/tests/test_adaptive_obligation_v2_local_model_r2.py \
  code/tests/test_adaptive_obligation_v2_propose.py \
  code/tests/test_adaptive_obligation_v2_local_model.py \
  code/tests/test_adaptive_obligation_v2_ledger.py -q
```

Expected: all tests pass; no real tokenizer/model/GPU/network/qrels action occurs.

- [ ] **Step 5: Commit Task 3**

```bash
git add code/trec_rag/adaptive_obligation_v2_propose_r2.py \
  code/trec_rag/adaptive_obligation_v2_local_model_r2.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py \
  code/tests/test_adaptive_obligation_v2_local_model_r2.py
git commit -m "Guard adaptive proposal R2 execution"
```

---

### Task 4: Build verified R1 incident and R2 preflight artifacts

**Files:**
- Create runtime: `outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident/receipt.json`
- Create runtime: `outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2/`

**Interfaces:**
- Consumes: Tasks 1-3 public functions and exposes their safe CLI actions.
- Produces: exact verified R1 incident and inference-free R2 preflight receipts.

- [ ] **Step 1: Add CLI smoke tests for incident and R2 preflight actions**

```python
def test_r2_cli_has_no_approval_creation_or_implicit_execute_action() -> None:
    actions = set(r2_module._parser()._subparsers._group_actions[0].choices)
    assert actions == {
        "build-preflight",
        "verify-preflight",
        "execute",
        "finalize",
    }
    assert "approve" not in actions
```

- [ ] **Step 2: Run the CLI tests and verify they fail before parser completion**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py -k cli -q
```

Expected: the R2 parser smoke test fails until the exact safe action set exists.

- [ ] **Step 3: Complete the two safe CLI parsers**

```python
# adaptive_obligation_v2_proposal_incident.py
def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        action = actions.add_parser(name)
        action.add_argument("--preflight", type=Path, required=True)
        action.add_argument("--approval", type=Path, required=True)
        action.add_argument("--ledger", type=Path, required=True)
        action.add_argument("--output", type=Path, required=True)
    return parser


# adaptive_obligation_v2_propose_r2.py
def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    build = actions.add_parser("build-preflight")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ledger-destination", type=Path, required=True)
    build.add_argument("--proposal-destination", type=Path, required=True)
    verify = actions.add_parser("verify-preflight")
    verify.add_argument("--output", type=Path, required=True)
    execute = actions.add_parser("execute")
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--approval", type=Path, required=True)
    execute.add_argument("--ledger", type=Path, required=True)
    finalize = actions.add_parser("finalize")
    finalize.add_argument("--preflight", type=Path, required=True)
    finalize.add_argument("--approval", type=Path, required=True)
    finalize.add_argument("--ledger", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser
```

The R2 CLI never creates approval bytes. `execute` must fail before
preflight/model/output access when the separately supplied R2 approval is
absent.

- [ ] **Step 4: Build and verify the canonical incident receipt**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m \
  trec_rag.adaptive_obligation_v2_proposal_incident build \
  --preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight \
  --approval outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_approval.json \
  --ledger outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_ledger \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident

TMPDIR=/var/tmp .venv/bin/python -m \
  trec_rag.adaptive_obligation_v2_proposal_incident verify \
  --preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight \
  --approval outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_approval.json \
  --ledger outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_ledger \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident
```

Expected: one attempted job, one terminal schema error, 47 uncalled jobs, and
unchanged R1 ledger hashes.

- [ ] **Step 5: Build and verify the tokenizer-only R2 preflight**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m \
  trec_rag.adaptive_obligation_v2_propose_r2 build-preflight \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2 \
  --ledger-destination "$PWD/outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_ledger_r2" \
  --proposal-destination "$PWD/outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposals_r2"

TMPDIR=/var/tmp .venv/bin/python -m \
  trec_rag.adaptive_obligation_v2_propose_r2 verify-preflight \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2
```

Expected: 48 jobs, 48 primary calls, 48 retry ceiling, 96 worst case, exact
token counts, tokenizer load one, model/inference/network/retrieval/qrels zero,
and a receipt SHA-256 different from R1.

- [ ] **Step 6: Prove R2 inference remains unapproved and inert**

Run:

```bash
test ! -e outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_approval_r2.json
test ! -e outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_ledger_r2
TMPDIR=/var/tmp .venv/bin/python-rocm -m \
  trec_rag.adaptive_obligation_v2_propose_r2 execute \
  --preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2 \
  --approval outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_approval_r2.json \
  --ledger "$PWD/outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_ledger_r2"
```

Expected: nonzero exit with `R2 proposal approval required`; no model load or
`proposal_ledger_r2` creation.

---

### Task 5: Render the R1 incident and R2 readiness report

**Files:**
- Create: `code/trec_rag/build_adaptive_obligation_v2_r2_report.py`
- Create: `code/tests/test_build_adaptive_obligation_v2_r2_report.py`
- Create: `reports/experiments/adaptive_obligation_search_v2_proposal_r2/artifact.json`
- Create: `reports/experiments/adaptive_obligation_search_v2_proposal_r2/report.html`

**Interfaces:**
- Consumes: verified contract, sealed baselines, R1 incident receipt, R2 preflight receipt, plan, design, and implementation hashes.
- Produces: a validated portable report that says R1 failed after one call, R2 is preflight-ready, and no adaptive relevance result exists.

- [ ] **Step 1: Write failing report truth-boundary and rendering tests**

```python
def test_report_distinguishes_r1_failure_from_r2_readiness() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    text = json.dumps(artifact)
    assert "R1 stopped after one schema-invalid call" in text
    assert "R2 proposal inference has not run" in text
    assert "R2 validation integration is intentionally unavailable" in text
    assert "No adaptive relevance result exists" in text
    assert '"calls_completed": 1' in text
    assert '"r2_calls_completed": 0' in text


def test_report_records_exact_r2_preflight_hash_and_counts() -> None:
    artifact = build_r2_report_artifact(_verified_sources())
    rows = artifact["snapshot"]["datasets"]["proposal_runs"]
    assert rows[1]["preflight_sha256"] == _r2_receipt_sha256()
    assert rows[1]["primary_call_count"] == 48
    assert rows[1]["worst_case_call_ceiling"] == 96


def test_report_refuses_any_r2_approval_or_ledger() -> None:
    sources = _verified_sources()
    sources["r2_approval_present"] = True
    with pytest.raises(ValueError, match="R2 inference must remain unopened"):
        build_r2_report_artifact(sources)
```

- [ ] **Step 2: Run report tests and verify RED**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_build_adaptive_obligation_v2_r2_report.py -q
```

Expected: collection fails because the R2 report builder does not exist.

- [ ] **Step 3: Implement the verified report payload and artifact**

Build separate source-backed sections for:

- current answer: R1 stopped safely; R2 is ready but unrun;
- exact R1 failure evidence and immutable hashes;
- exact R2 prompt contract, job counts, token counts, and approval boundary;
- pipeline state through validation/retrieval/reranking/qrels;
- an explicit boundary that R2 proposal output cannot enter validation until a
  separate end-to-end revision/provenance plan is implemented;
- an operator-estimated 2-3 hour R2 runtime clearly labeled as an estimate based
  on the observed approximately three-minute 54,402-token R1 call, not as a
  receipt-verified metric; and
- the next action: approve only the exact R2 preflight and ledger destination.

Use the canonical Data Analytics portable artifact manifest with at least one
binary stage-readiness chart, source-backed metrics, exact tables, and no false
relevance comparison.

- [ ] **Step 4: Run all focused report and compatibility tests**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest \
  code/tests/test_build_adaptive_obligation_v2_r2_report.py \
  code/tests/test_build_adaptive_obligation_v2_report.py \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py -q
```

Expected: all pass and the original report files remain byte-identical.

- [ ] **Step 5: Build and render the versioned report**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m \
  trec_rag.build_adaptive_obligation_v2_r2_report \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/contract \
  --baseline-rankings outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings \
  --r1-incident outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_run_r1_incident \
  --r2-preflight outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2/proposal_preflight_r2 \
  --artifact reports/experiments/adaptive_obligation_search_v2_proposal_r2/artifact.json \
  --output reports/experiments/adaptive_obligation_search_v2_proposal_r2/report.html
```

Expected: renderer validation, packaging, source interaction, desktop 1440px,
and mobile 390px checks pass.

- [ ] **Step 6: Run final verification and obtain independent review**

Run:

```bash
TMPDIR=/var/tmp .venv/bin/python -m pytest -q \
  code/tests/test_adaptive_obligation_v2_proposal_incident.py \
  code/tests/test_adaptive_obligation_v2_propose_r2.py \
  code/tests/test_adaptive_obligation_v2_local_model_r2.py \
  code/tests/test_adaptive_obligation_v2_validate.py \
  code/tests/test_adaptive_obligation_v2_retrieve.py \
  code/tests/test_build_adaptive_obligation_v2_r2_report.py \
  -k 'not real_task4_acceptance_freezes_authenticated_retrieval_inputs and not validation_finalizer_rejects_forged_upstream_hashes_before_ledger'

TMPDIR=/var/tmp .venv/bin/python -m compileall -q code/trec_rag
git diff --check
```

Ask an independent advisor to review the exact implementation range, R1
immutability, R2 prompt salience, approval isolation, retry policy, explicit
validation deferral, and rendered claims. Resolve every blocking finding before
handoff.

- [ ] **Step 7: Commit Task 5**

```bash
git add code/trec_rag/build_adaptive_obligation_v2_r2_report.py \
  code/tests/test_build_adaptive_obligation_v2_r2_report.py \
  reports/experiments/adaptive_obligation_search_v2_proposal_r2/artifact.json \
  reports/experiments/adaptive_obligation_search_v2_proposal_r2/report.html
git commit -m "Render adaptive proposal R2 readiness"
```

## Final Handoff Gate

Report the exact R2 preflight receipt SHA-256, min/max/total prompt-token counts,
48/48/96 call ceilings, estimated 2-3 hour local runtime, model/revision, and
absolute `proposal_ledger_r2` destination. Confirm that R2 approval, ledger,
model inference, validation, BM25, MiniLM, and qrels are absent. Ask for one new
approval bound to those exact values before creating approval bytes or making
the first R2 model call.

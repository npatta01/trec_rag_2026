# Full-Union Adaptive Evidence Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-topic, full-union passage ranker that preserves all 8,114 candidate rows, discovers bounded corpus-derived sub-obligations, selects novel evidence with weight-free token fairness, and renders a source-backed diagnostic report.

**Architecture:** Freeze a qrels-blind contract over the existing accepted union, score query-local passage queues with the pinned MiniLM model, and run one cross-fitted local Qwen discovery pass. Build complete NARRATIVE, FIXED-O0, ADAPTIVE, and non-promoting COMPOSITE continuations; extract exact-span evidence cards; seal all packets before qrels and review labels are joined; then evaluate and render the existing report.

**Tech Stack:** Python 3.12, pytest, PyTorch ROCm, Transformers, the existing global score cache, JSON/JSONL immutable artifacts, SQLite-backed portable report data, and headless Google Chrome for desktop/mobile rendering checks.

## Global Constraints

- Work only in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve immutable v2.1 and all existing sealed deep-facet artifacts.
- Hard-reject topic IDs `144`, `213`, `224`, `407`, and `515` before any source, model, score, join, evaluation, or report access.
- Pilot topics are exactly `219`, `72`, `300`, and `84`; they are post-qrels diagnostic topics, not fresh confirmation.
- Input population is exactly 8,114 accepted topic-document rows; every document must remain represented in every complete continuation.
- No 100, 500, 1,000, or other document-count eligibility cutoff.
- No new retrieval, query rewrite, dense primary retrieval, network call, model download, hosted inference, or paid call in this plan.
- MiniLM is `cross-encoder/ms-marco-MiniLM-L6-v2` revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`.
- Corpus discovery uses local `Qwen/Qwen3-4B-Instruct-2507` revision `cdbee75f17c01a7cc42f958dc650907174af0554`, temperature `0`, seed `0`, and frozen JSON schemas.
- Cross-encoder scores order only their own query queue; raw or normalized scores never cross query boundaries.
- Explicit obligations `O0` always precede corpus-derived `O1`; `O1` never receives recurring equal allocation with `O0`.
- Run one discovery pass only; no recursion, new search, self-selected goal, or model-selected stopping rule.
- Evidence packets are nested complete prefixes at 8,000, 16,000, and 32,000 Qwen-token evidence budgets.
- Qrels, organizer nuggets, reference answers, and review labels stay unread until continuations, packets, cards, hashes, and the freeze seal exist.
- Use create-only output under `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/`.
- Keep unrelated untracked sparse-relevance files untouched and out of every commit.

---

## File map

- `code/trec_rag/adaptive_evidence_contract.py`: source binding, protected-topic checks, O0 records, fold assignment, canonical hashes, and create-only manifest.
- `code/trec_rag/adaptive_evidence_score.py`: query-local score populations, window planning, cache audit, ROCm scoring, resumable shards, and receipts.
- `code/trec_rag/adaptive_evidence_local_model.py`: pinned local Qwen loader and schema-constrained JSON generation shared by discovery and evidence cards.
- `code/trec_rag/adaptive_evidence_discovery.py`: fold reservoirs, O1/N1 proposals, opposite-fold corroboration, repeated-phrase control, and frozen discovery records.
- `code/trec_rag/adaptive_evidence_rank.py`: qualification, duplicate/source rules, queue construction, coverage floors, token-deficit fill, packet prefixes, and complete continuations.
- `code/trec_rag/adaptive_evidence_cards.py`: exact-span evidence atomization, card verification, and blinded review slots.
- `code/trec_rag/adaptive_evidence_evaluate.py`: freeze seal, qrels firewall, packet/card metrics, review-label joins, and advancement decision.
- `code/trec_rag/build_deep_facet_candidate_report.py`: verified adaptive-result loader plus answer-first report sections, tables, and charts.
- `code/tests/test_adaptive_evidence_*.py`: focused tests matching each module.
- `reports/experiments/deep_facet_candidate_pilot_v1/report.html`: rebuilt self-contained report.
- `reports/experiments/deep_facet_candidate_pilot_v1/adaptive_evidence_advisor_review.md`: independent method/findings review.

---

### Task 1: Freeze the full-union contract and explicit obligations

**Files:**
- Create: `code/trec_rag/adaptive_evidence_contract.py`
- Create: `code/tests/test_adaptive_evidence_contract.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract/`

**Interfaces:**
- Consumes: `reports/experiments/deep_facet_candidate_pilot_v1/manifest.json`, `outputs/rag25_deep_facet_candidates_v1/gate_v1/gates.json`, and `outputs/rag25_deep_facet_candidates_v1/gate_v1/u_accepted.jsonl`.
- Produces: `canonical_sha256(value) -> str`, `document_fold(topic_id, document_id) -> int`, `render_obligation_query(narrative, obligation) -> str`, `build_contract(manifest, gates, union_rows) -> dict[str, object]`, and create-only `manifest.json`, `obligations.jsonl`, `documents.jsonl`, `folds.jsonl`, `summary.json`.

- [ ] **Step 1: Write failing contract tests**

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trec_rag.adaptive_evidence_contract import (
    build_contract,
    document_fold,
    reject_protected_before_access,
    render_obligation_query,
)


def _manifest() -> dict[str, object]:
    return {
        "topics": [{"topic_id": "219", "query": "full narrative"}],
        "facets": [{
            "topic_id": "219",
            "facet_id": "219-positive",
            "manifest_order": 0,
            "obligation": "Positive effects on society.",
            "anchor_terms": ["technology"],
            "relation_terms": ["positive", "society"],
            "wrong_domain_patterns": ["technology stock"],
        }],
    }


def _rows() -> list[dict[str, object]]:
    return [{
        "topic_id": "219",
        "document_id": "d1",
        "text": "technology can affect society",
        "text_sha256": hashlib.sha256(b"technology can affect society").hexdigest(),
        "union_order": 1,
        "provenance": [{"family": "facet", "facet_id": "219-positive", "rank": 1}],
    }]


def _gates() -> dict[str, object]:
    return {"gates": [{"facet_id": "219-positive", "accepted": True}]}


def test_contract_tethers_o0_to_full_narrative_and_assigns_fold() -> None:
    contract = build_contract(
        _manifest(), _gates(), _rows(), expected_population=1, expected_o0=1,
    )
    o0 = contract["obligations"][1]
    assert contract["obligations"][0]["kind"] == "broad"
    assert o0["kind"] == "o0"
    assert o0["query"] == "full narrative\n\nExplicit obligation:\nPositive effects on society."
    assert contract["documents"][0]["fold"] == document_fold("219", "d1")


def test_protected_topic_fails_before_loader() -> None:
    touched = False
    def loader() -> object:
        nonlocal touched
        touched = True
        return object()
    with pytest.raises(ValueError, match="protected topic 144"):
        reject_protected_before_access(["144"], loader)
    assert touched is False


def test_fold_is_sha256_mod_two() -> None:
    expected = int(hashlib.sha256(b"219\0d1").hexdigest(), 16) % 2
    assert document_fold("219", "d1") == expected
```

- [ ] **Step 2: Run the focused tests and verify the import failure**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_contract.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trec_rag.adaptive_evidence_contract'`.

- [ ] **Step 3: Implement the contract module**

```python
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PILOT_TOPIC_IDS = ("219", "72", "300", "84")
SCHEMA_VERSION = "adaptive-evidence-contract-v1"


def canonical_sha256(value: object) -> str:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def reject_protected_before_access(topic_ids: Iterable[str], loader: Callable[[], T]) -> T:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    return loader()


def document_fold(topic_id: str, document_id: str) -> int:
    if str(topic_id) in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    digest = hashlib.sha256(f"{topic_id}\0{document_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def render_obligation_query(narrative: str, obligation: str) -> str:
    narrative, obligation = narrative.strip(), obligation.strip()
    if not narrative or not obligation:
        raise ValueError("narrative and obligation must be nonempty")
    return f"{narrative}\n\nExplicit obligation:\n{obligation}"


def build_contract(
    manifest: Mapping[str, object],
    gates: Mapping[str, object],
    union_rows: Sequence[Mapping[str, object]],
    *,
    expected_population: int = 8114,
    expected_o0: int = 24,
) -> dict[str, object]:
    topics = {str(row["topic_id"]): row for row in manifest["topics"]}  # type: ignore[index]
    accepted = {
        str(row["facet_id"])
        for row in gates["gates"]  # type: ignore[index]
        if row.get("accepted") is True
    }
    if len(accepted) != expected_o0:
        raise ValueError(f"accepted facet set must contain exactly {expected_o0} O0 obligations")
    obligations: list[dict[str, object]] = []
    for topic_id in topics:
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        narrative = str(topics[topic_id]["query"])
        obligations.append({
            "topic_id": topic_id,
            "obligation_id": f"{topic_id}:broad",
            "kind": "broad",
            "parent_id": None,
            "manifest_order": -1,
            "text": narrative,
            "query": narrative,
        })
    for facet in manifest["facets"]:  # type: ignore[index]
        if str(facet["facet_id"]) not in accepted:
            continue
        topic_id = str(facet["topic_id"])
        text = str(facet["obligation"])
        obligations.append({
            "topic_id": topic_id,
            "obligation_id": str(facet["facet_id"]),
            "source_facet_id": str(facet["facet_id"]),
            "kind": "o0",
            "parent_id": None,
            "manifest_order": int(facet["manifest_order"]),
            "text": text,
            "query": render_obligation_query(str(topics[topic_id]["query"]), text),
            "anchor_terms": list(facet["anchor_terms"]),
            "relation_terms": list(facet["relation_terms"]),
            "wrong_domain_patterns": list(facet["wrong_domain_patterns"]),
        })
    documents = []
    for row in union_rows:
        topic_id, document_id = str(row["topic_id"]), str(row["document_id"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        text = str(row["text"])
        if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("accepted-union text hash mismatch")
        documents.append({**dict(row), "fold": document_fold(topic_id, document_id)})
    if len(documents) != expected_population:
        raise ValueError(f"accepted population must contain exactly {expected_population} rows")
    return {"schema_version": SCHEMA_VERSION, "obligations": obligations, "documents": documents}
```

Add create-only canonical JSON/JSONL writers and a `create` CLI that writes hashes for every output file into `summary.json`; reject overwrite and any topic set other than the four pilot topics.

- [ ] **Step 4: Run contract tests and materialize the contract**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_contract.py -q
.venv/bin/python -m trec_rag.adaptive_evidence_contract create \
  --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json \
  --gates outputs/rag25_deep_facet_candidates_v1/gate_v1/gates.json \
  --union outputs/rag25_deep_facet_candidates_v1/gate_v1/u_accepted.jsonl \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract
```

Expected: tests pass; CLI prints `status=complete documents=8114 o0=24 protected=0 qrels_opened=false`.

- [ ] **Step 5: Commit the contract**

```bash
git add code/trec_rag/adaptive_evidence_contract.py code/tests/test_adaptive_evidence_contract.py
git commit -m "Add adaptive evidence contract"
```

---

### Task 2: Audit and preflight query-local MiniLM score coverage

**Files:**
- Create: `code/trec_rag/adaptive_evidence_score.py`
- Create: `code/tests/test_adaptive_evidence_score.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/preflight/`

**Interfaces:**
- Consumes: Task 1 contract records, existing MiniLM receipts/cache, and later accepted O1 records.
- Produces: `build_score_candidates(contract, derived=()) -> list[dict]`, `build_score_preflight(candidates, tokenizer, cache_lookup) -> dict`, `score_window_rows(rows, predict, cache_get, cache_add) -> list[dict]`, and create-only candidates/windows/preflight artifacts.

**Approved boundary resolution:** Task 2 defines and tests `score_window_rows` as a pure injected cache/predict orchestration function, but its CLI never calls it and performs zero inference. Task 3 supplies the authenticated ROCm predictor plus resumable shard orchestration around this interface.

- [ ] **Step 1: Write failing population and cache-identity tests**

```python
def test_score_populations_keep_broad_universal_and_o0_parent_local() -> None:
    rows = build_score_candidates(_contract())
    identities = {(row["document_id"], row["obligation_id"]) for row in rows}
    assert ("original-only", "219:broad") in identities
    assert ("facet-doc", "219:broad") in identities
    assert ("facet-doc", "219-positive") in identities
    assert ("original-only", "219-positive") not in identities
    assert all(row["query"].startswith("full narrative") for row in rows)


def test_o1_scores_only_its_parent_population() -> None:
    derived = [{
        "topic_id": "219", "obligation_id": "219-positive:o1:access",
        "kind": "o1", "parent_id": "219-positive", "text": "accessibility effects",
        "query": (
            "full narrative\n\nExplicit obligation:\nPositive effects on society."
            "\n\nCorpus-derived sub-obligation:\naccessibility effects"
        ),
    }]
    rows = build_score_candidates(_contract(), derived=derived)
    o1 = [row for row in rows if row["obligation_id"].endswith(":access")]
    assert {row["document_id"] for row in o1} == {"facet-doc"}


def test_preflight_records_exact_hits_misses_and_never_reads_qrels() -> None:
    preflight = build_score_preflight(
        build_score_candidates(_contract()),
        _Tokenizer(),
        cache_lookup=lambda query, text: 0.25 if text == "cached" else None,
    )
    assert preflight["qrels_opened"] is False
    assert preflight["summary"]["candidate_count"] > 0
    assert preflight["summary"]["unique_pair_count"] == (
        preflight["summary"]["cache_hit_count"]
        + preflight["summary"]["cache_miss_count"]
    )


def test_score_window_rows_deduplicates_pairs_and_restores_all_windows() -> None:
    rows = [_window("w1", "q", "same"), _window("w2", "q", "same")]
    calls: list[tuple[str, str]] = []
    scores = score_window_rows(
        rows,
        cache_get=lambda q, t: None,
        cache_add=lambda pairs: calls.extend((q, t) for q, t, _ in pairs),
        predict=lambda pairs: [0.75 for _ in pairs],
    )
    assert calls == [("q", "same")]
    assert [row["window_id"] for row in scores] == ["w1", "w2"]
    assert {row["score"] for row in scores} == {0.75}
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_score.py -q
```

Expected: collection fails because `adaptive_evidence_score` does not exist.

- [ ] **Step 3: Implement score populations and preflight**

```python
def _generated_by(document: Mapping[str, object], facet_id: str) -> bool:
    return any(
        row.get("family") == "facet" and row.get("facet_id") == facet_id
        for row in document["provenance"]  # type: ignore[index]
    )


def build_score_candidates(
    contract: Mapping[str, object],
    *,
    derived: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    obligations = [*contract["obligations"], *derived]  # type: ignore[index]
    by_id = {str(row["obligation_id"]): row for row in obligations}
    output: list[dict[str, object]] = []
    for document in contract["documents"]:  # type: ignore[index]
        topic_id = str(document["topic_id"])
        for obligation in obligations:
            if str(obligation["topic_id"]) != topic_id:
                continue
            kind = str(obligation["kind"])
            if kind == "o0" and not _generated_by(document, str(obligation["source_facet_id"])):
                continue
            if kind == "o1":
                parent = by_id[str(obligation["parent_id"])]
                if not _generated_by(document, str(parent["source_facet_id"])):
                    continue
            query = str(obligation["query"])
            text = str(document["text"])
            output.append({
                "topic_id": topic_id,
                "obligation_id": str(obligation["obligation_id"]),
                "family": kind,
                "variant": str(obligation["obligation_id"]),
                "rank": int(document["union_order"]),
                "document_id": str(document["document_id"]),
                "query": query,
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "fold": int(document["fold"]),
            })
    return sorted(output, key=lambda row: (row["topic_id"], row["obligation_id"], row["rank"]))
```

Use `facet_local_minilm_preflight.build_window_plan` to preserve the authenticated 512/192/256/64/max-32 window contract and its coverage fraction. Build a complete coverage matrix grouped by topic, obligation, document, window, cache hit, and cache miss. Bind model/tokenizer receipts and code hashes before any inference.

- [ ] **Step 4: Run tests and create the first score preflight**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_score.py -q
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_score preflight \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/preflight
```

Expected: tests pass; preflight prints exact document/window/pair/cache counts, projected ROCm runtime, `network=false`, `qrels_opened=false`, and `$0` external cost. Do not infer yet.

- [ ] **Step 5: Commit score planning**

```bash
git add code/trec_rag/adaptive_evidence_score.py code/tests/test_adaptive_evidence_score.py
git commit -m "Add adaptive evidence score preflight"
```

---

### Task 3: Complete MiniLM scoring locally and resumably

**Files:**
- Modify: `code/trec_rag/adaptive_evidence_score.py`
- Modify: `code/tests/test_adaptive_evidence_score.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/base/`

**Interfaces:**
- Consumes: Task 2 preflight, Task 2 `score_window_rows`, and the existing global MiniLM score cache.
- Produces: `verify_completed_shard(path, receipt) -> int`, `run_local_scoring(preflight_dir, output_dir, cache_root) -> dict`, the authenticated ROCm predictor adapter, topic-obligation score shards, and `receipt.json`.

- [ ] **Step 1: Add failing resumability and completeness tests**

```python
def test_resume_accepts_only_hash_matching_complete_shards(tmp_path: Path) -> None:
    shard = tmp_path / "219__broad.jsonl"
    shard.write_text('{"window_id":"w1","score":0.5}\n', encoding="utf-8")
    receipt = {"rows": 1, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
    assert verify_completed_shard(shard, receipt) == 1
    shard.write_text('{"window_id":"w1","score":0.6}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="shard hash"):
        verify_completed_shard(shard, receipt)
```

- [ ] **Step 2: Run tests and verify the new orchestration functions fail**

Run: `.venv/bin/python -m pytest code/tests/test_adaptive_evidence_score.py -q`

Expected: failures name `verify_completed_shard` and `run_local_scoring`; the already-tested Task 2 `score_window_rows` remains green.

- [ ] **Step 3: Implement shard verification, resumable orchestration, and the ROCm adapter**

```python
def verify_completed_shard(path: Path, receipt: Mapping[str, object]) -> int:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError("completed shard hash differs from its receipt")
    rows = sum(1 for line in payload.splitlines() if line)
    if rows != receipt["rows"]:
        raise ValueError("completed shard row count differs from its receipt")
    return rows
```

Use the Task 2 `score_window_rows` function unchanged. The production `predict` adapter must load the authenticated MiniLM snapshot with `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`, float32, `eval()`, and ROCm `cuda`. Write one create-only shard per topic-obligation; on restart, verify and skip only complete hash-matching shards. Poll long-running scoring sessions rather than blocking silently.

- [ ] **Step 4: Run tests, execute base scoring, and verify the receipt**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_score.py -q
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_score score \
  --preflight outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/preflight \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/base
.venv/bin/python -m trec_rag.adaptive_evidence_score verify \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/base
```

Expected: all tests pass; every frozen window has one score; receipt reports local ROCm device, pinned model revision, complete shard hashes, zero network/paid calls, and `qrels_opened=false`.

- [ ] **Step 5: Commit the scorer**

```bash
git add code/trec_rag/adaptive_evidence_score.py code/tests/test_adaptive_evidence_score.py
git commit -m "Add resumable adaptive MiniLM scoring"
```

---

### Task 4: Discover and cross-validate O1/N1 records once

**Files:**
- Create: `code/trec_rag/adaptive_evidence_local_model.py`
- Create: `code/trec_rag/adaptive_evidence_discovery.py`
- Create: `code/tests/test_adaptive_evidence_discovery.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery/`

**Interfaces:**
- Consumes: contract, base MiniLM scores, and exact top-ten per-O0/fold reservoirs.
- Produces: `LocalJsonModel.generate(messages, schema, max_new_tokens) -> dict`, `build_reservoirs(...)`, `validate_proposal(...)`, `merge_nuggets(...)`, and proposed/validated/rejected/frozen O1/N1 artifacts.

- [ ] **Step 1: Write failing discovery-boundary tests**

```python
def test_reservoir_has_ten_distinct_docs_per_parent_fold() -> None:
    rows = build_reservoirs(_o0(), _passages(30), limit=10)
    assert len(rows[("219-positive", 0)]) == 10
    assert len({row["document_id"] for row in rows[("219-positive", 0)]}) == 10


def test_answer_fact_cannot_become_o1() -> None:
    proposal = _proposal(label="COVID-19 caused a 42 percent increase", kind="o1")
    decision = validate_proposal(proposal, _parent(), opposite_fold_support=_support())
    assert decision["accepted"] is False
    assert "candidate_answer" in decision["reasons"]


def test_o1_requires_distinct_opposite_fold_document() -> None:
    decision = validate_proposal(_proposal(), _parent(), opposite_fold_support=[])
    assert decision["accepted"] is False
    assert "cross_fold_support" in decision["reasons"]


def test_nugget_merging_preserves_singletons_and_merges_jaccard_080() -> None:
    merged = merge_nuggets([
        _nugget("a b c d e f g h i"),
        _nugget("a b c d e f g h j"),
        _nugget("rare x"),
    ])
    assert len(merged) == 2
    assert any(row["singleton"] is True for row in merged)
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `.venv/bin/python -m pytest code/tests/test_adaptive_evidence_discovery.py -q`

Expected: import failure for the two discovery modules.

- [ ] **Step 3: Implement the pinned local JSON model adapter**

```python
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION


class LocalJsonModel:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16,
        ).eval().to("cuda")

    def generate(self, messages, schema, *, max_new_tokens=1200):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        value = json.loads(completion)
        validate_json_schema(value, schema)
        return value
```

The frozen discovery prompt must say: preserve the parent subject/population/relation; propose abstract O1 categories separately from specific N1 facts; quote exact substrings from supplied passages; never use outside knowledge; return `unsupported` rather than inventing evidence. Validate every returned span with `span in passage_text`.

- [ ] **Step 4: Implement deterministic cross-fold acceptance and the no-LLM control**

```python
def validate_proposal(proposal, parent, *, opposite_fold_support):
    reasons: list[str] = []
    if proposal["parent_id"] != parent["obligation_id"]:
        reasons.append("parent")
    if any(field not in proposal for field in ("label", "scope_rationale", "support_span")):
        reasons.append("schema")
    if _looks_like_answer_fact(str(proposal["label"])):
        reasons.append("candidate_answer")
    distinct = {str(row["document_id"]) for row in opposite_fold_support if row["qualified"] is True}
    if not distinct or proposal["document_id"] in distinct:
        reasons.append("cross_fold_support")
    if not _parent_scope_preserved(proposal, parent):
        reasons.append("scope")
    return {"accepted": not reasons, "reasons": sorted(set(reasons))}
```

The repeated-phrase control emits contiguous two-to-five analyzed content-token phrases found in at least two nonduplicate documents across folds. Every accepted O1 record gets a deterministic query containing the full narrative, its complete parent O0 text, and the O1 label under a `Corpus-derived sub-obligation` heading. Rank accepted O1 lexicographically and freeze at most one per parent and four per topic. Run discovery exactly once; refuse an existing output directory.

- [ ] **Step 5: Run tests, preflight Qwen, cross-validate proposals, and score accepted O1**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_discovery.py -q
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_discovery preflight \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --scores outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/base \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_discovery propose \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_score provisional-o1 \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --proposals outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery/proposed_o1.jsonl \
  --opposite-fold-only \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/provisional-o1
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_discovery validate \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery \
  --opposite-fold-scores outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/provisional-o1
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_score accepted-o1 \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --accepted outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery/accepted_o1.jsonl \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring/o1
```

Expected: tests pass; discovery receipt records exactly one proposal pass, two folds, pinned Qwen, no external calls, all exact spans, opposite-fold acceptance/rejection reasons, and zero qrels access. Provisional O1 scores use only the opposite fold; accepted O1 scoring then covers each full parent population.

- [ ] **Step 6: Commit discovery**

```bash
git add code/trec_rag/adaptive_evidence_local_model.py code/trec_rag/adaptive_evidence_discovery.py code/tests/test_adaptive_evidence_discovery.py code/trec_rag/adaptive_evidence_score.py code/tests/test_adaptive_evidence_score.py
git commit -m "Add cross-fitted evidence discovery"
```

---

### Task 5: Build weight-free complete continuations and nested packets

**Files:**
- Create: `code/trec_rag/adaptive_evidence_rank.py`
- Create: `code/tests/test_adaptive_evidence_rank.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings/`

**Interfaces:**
- Consumes: contract, base/O1 scores, frozen discovery records, Qwen token counts, and provenance.
- Produces: `qualify_passage(...)`, `qualified_deficit_round_robin(...)`, `packet_prefix(...)`, `build_arm_continuation(...)`, complete arm JSONL, and packet JSONL.

- [ ] **Step 1: Write failing fairness, novelty, and completeness tests**

```python
def test_coverage_floor_is_broad_then_all_o0_then_o1() -> None:
    rows = build_arm_continuation(_fixture(), arm="ADAPTIVE")
    assert [row["primary_obligation"] for row in rows[:4]] == [
        "219:broad", "219-positive", "219-negative", "219-positive:o1:access"
    ]


def test_deficit_fill_uses_fewest_tokens_and_prefers_unseen_nugget() -> None:
    selected = qualified_deficit_round_robin(_queues(), initial=_coverage_floor())
    assert selected[0]["primary_obligation"] == "219-negative"
    assert selected[0]["nugget_id"] == "new-negative-nugget"


def test_packets_are_nested_and_continuation_represents_every_document() -> None:
    continuation = build_arm_continuation(_fixture(), arm="ADAPTIVE")
    p8 = packet_prefix(continuation, 8_000)
    p16 = packet_prefix(continuation, 16_000)
    p32 = packet_prefix(continuation, 32_000)
    assert p16[:len(p8)] == p8
    assert p32[:len(p16)] == p16
    assert {row["document_id"] for row in continuation} == _all_document_ids()


def test_source_warning_cannot_reject_and_o1_never_recurs() -> None:
    qualified = qualify_passage(_warning_passage(), _o0(), selected=[])
    assert qualified["qualified"] is True
    continuation = build_arm_continuation(_fixture(), arm="ADAPTIVE")
    assert sum(row["primary_obligation"].endswith(":o1:access") for row in continuation[:20]) == 1
```

- [ ] **Step 2: Run tests and verify missing ranker**

Run: `.venv/bin/python -m pytest code/tests/test_adaptive_evidence_rank.py -q`

Expected: import failure for `adaptive_evidence_rank`.

- [ ] **Step 3: Implement qualification and token-deficit selection**

```python
def qualified_deficit_round_robin(queues, *, initial):
    selected = list(initial)
    assigned = defaultdict(int)
    for row in selected:
        assigned[row["primary_obligation"]] += int(row["evidence_tokens"])
    active = {key: list(rows) for key, rows in queues.items() if rows}
    while active:
        obligation_id = min(
            active,
            key=lambda key: (
                assigned[key],
                0 if key.endswith(":broad") else 1,
                active[key][0]["manifest_order"],
                key,
            ),
        )
        candidate = next_qualified_novel(active[obligation_id], obligation_id, selected)
        if candidate is None:
            del active[obligation_id]
            continue
        selected.append(candidate)
        assigned[obligation_id] += int(candidate["evidence_tokens"])
    return selected


def packet_prefix(rows, token_budget):
    selected, used = [], 0
    for row in rows:
        cost = int(row["evidence_tokens"])
        if used + cost > token_budget:
            break
        selected.append(row)
        used += cost
    return selected
```

Qualification must emit reasons and permit `unsupported` when a nonempty queue lacks anchor/relation/domain-coherent exact support. Apply Jaccard `>=0.80` as a hard finite-packet duplicate deferral, unseen host before reused host within an obligation, and at most three finite-packet passages per document. Deferred rows remain in deterministic tail order. Implement NARRATIVE, FIXED-O0, ADAPTIVE, and the exact commit-`1aacea7` COMPOSITE sensitivity; only the first three are eligible for decisions.

Every ranked passage record must include one `primary_obligation` and the complete deterministic `supported_obligations` list. These fields are frozen here and reused by packet construction, cards, and blinded review; downstream stages may validate them but cannot infer or rewrite them.

- [ ] **Step 4: Run tests, build all continuations, and verify complete eligibility**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_rank.py -q
.venv/bin/python -m trec_rag.adaptive_evidence_rank build \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --scores outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/scoring \
  --discovery outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/discovery \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings
.venv/bin/python -m trec_rag.adaptive_evidence_rank verify \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings
```

Expected: all tests pass; every arm has nested packet prefixes; every complete continuation covers the same 8,114 topic-document identities; COMPOSITE is marked `decision_eligible=false`.

- [ ] **Step 5: Commit ranking**

```bash
git add code/trec_rag/adaptive_evidence_rank.py code/tests/test_adaptive_evidence_rank.py
git commit -m "Add novelty-aware evidence ranking"
```

---

### Task 6: Atomize packets into exact-span cards and blind review slots

**Files:**
- Create: `code/trec_rag/adaptive_evidence_cards.py`
- Create: `code/tests/test_adaptive_evidence_cards.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/cards/`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/review/`

**Interfaces:**
- Consumes: Task 5 packets and Task 4 local JSON model adapter.
- Produces: `validate_card(card, passage)`, `build_cards(...)`, `build_review_slots(...)`, exact-span cards, 72 deterministic arm-O0 slots, and blinded packet.

- [ ] **Step 1: Write failing card and blinding tests**

```python
def test_card_requires_exact_span_and_source_identity() -> None:
    passage = {"window_text": "Technology can improve access.", "document_id": "d1", "window_id": "w1"}
    card = {"claim": "improved access", "support_span": "improve access", "document_id": "d1", "window_id": "w1"}
    assert validate_card(card, passage)["valid"] is True
    card["support_span"] = "not present"
    assert validate_card(card, passage)["valid"] is False


def test_review_has_72_slots_and_unsupported_sentinels() -> None:
    slots = build_review_slots(_three_packets(), _twenty_four_o0(), budget=16_000)
    assert len(slots) == 72
    assert {(row["arm"], row["obligation_id"]) for row in slots} == _expected_pairs()
    assert any(row["status"] == "unsupported" for row in slots)


def test_blinding_hides_arm_and_is_deterministic() -> None:
    first = blind_slots(_slots(), seed="adaptive-evidence-v1")
    second = blind_slots(list(reversed(_slots())), seed="adaptive-evidence-v1")
    assert first == second
    assert all("arm" not in row for row in first if row["status"] == "passage")
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `.venv/bin/python -m pytest code/tests/test_adaptive_evidence_cards.py -q`

Expected: import failure for `adaptive_evidence_cards`.

- [ ] **Step 3: Implement exact-span cards and review slots**

```python
def validate_card(card, passage):
    reasons = []
    if card.get("document_id") != passage.get("document_id") or card.get("window_id") != passage.get("window_id"):
        reasons.append("source_identity")
    span = str(card.get("support_span", ""))
    if not span or span not in str(passage.get("window_text", "")):
        reasons.append("exact_span")
    if not card.get("obligation_id") or not card.get("nugget_id"):
        reasons.append("coverage_identity")
    return {"valid": not reasons, "reasons": sorted(reasons)}


def build_review_slots(packets, obligations, *, budget=16_000):
    slots = []
    for arm in ("NARRATIVE", "FIXED-O0", "ADAPTIVE"):
        packet = packets[(arm, budget)]
        for obligation in obligations:
            match = next((row for row in packet if obligation["obligation_id"] in row["supported_obligations"]), None)
            slots.append({
                "arm": arm,
                "obligation_id": obligation["obligation_id"],
                "status": "passage" if match else "unsupported",
                "passage": match,
            })
    return slots
```

Use the pinned Qwen adapter with a separate frozen atomization schema. It may only split a selected passage into atomic claims and exact support spans; it cannot add facts. Preserve raw packet passages. Reject invalid cards rather than repairing spans fuzzily.

- [ ] **Step 4: Run tests and create the unlabeled blinded review packet**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_cards.py -q
.venv/bin/python-rocm -m trec_rag.adaptive_evidence_cards build \
  --rankings outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/rankings \
  --contract outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/contract \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/cards
.venv/bin/python -m trec_rag.adaptive_evidence_cards review-packet \
  --cards outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/cards \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/review
```

Expected: tests pass; all cards have exact spans/source IDs; review manifest contains 72 arm-obligation slots, hides arms in the reviewer view, and contains no qrels or labels. Do not send the packet to reviewers until Task 7 has created and verified the freeze seal.

- [ ] **Step 5: Commit cards and review tooling**

```bash
git add code/trec_rag/adaptive_evidence_cards.py code/tests/test_adaptive_evidence_cards.py
git commit -m "Add adaptive evidence cards and review"
```

---

### Task 7: Seal packets, open qrels once, and evaluate mechanisms

**Files:**
- Create: `code/trec_rag/adaptive_evidence_evaluate.py`
- Create: `code/tests/test_adaptive_evidence_evaluate.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/freeze/`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/evaluation/`

**Interfaces:**
- Consumes before freeze: contract, discovery, scores, continuations, packets, cards, and the unlabeled blinded review packet. Consumes after verified freeze: completed blinded labels and projected qrels.
- Produces: `create_freeze(...)`, `verify_freeze(...)`, `evaluate_packet(...)`, `decide(...)`, and sealed metrics/decision/summary.

- [ ] **Step 1: Write failing firewall, metrics, and decision tests**

```python
def test_qrels_loader_is_not_called_without_verified_freeze(tmp_path: Path) -> None:
    touched = {"qrels": False, "labels": False}
    def qrels_loader():
        touched["qrels"] = True
        return {}
    def labels_loader():
        touched["labels"] = True
        return []
    with pytest.raises(ValueError, match="freeze seal"):
        evaluate(
            tmp_path / "missing-freeze",
            qrels_loader=qrels_loader,
            labels_loader=labels_loader,
        )
    assert touched == {"qrels": False, "labels": False}


def test_packet_metrics_count_distinct_adjudicated_nuggets_per_tokens() -> None:
    metrics = evaluate_packet(_packet(), _qrels(), _labels())
    assert metrics["explicit_o0_coverage"] == 1.0
    assert metrics["distinct_supported_nuggets"] == 3
    assert metrics["distinct_supported_nuggets_per_10000_tokens"] == pytest.approx(6.0)


def test_decision_rejects_o0_regression_or_bad_o1() -> None:
    evidence = _passing_evidence()
    evidence["per_topic"]["219"]["ADAPTIVE"]["explicit_o0_coverage"] = 0.5
    decision = decide(evidence)
    assert decision["advance_to_fresh_validation"] is False
    assert "no_o0_topic_regression" in decision["failed_guards"]
```

- [ ] **Step 2: Run tests and verify missing evaluator**

Run: `.venv/bin/python -m pytest code/tests/test_adaptive_evidence_evaluate.py -q`

Expected: import failure for `adaptive_evidence_evaluate`.

- [ ] **Step 3: Implement seal verification, qrels firewall, metrics, and decision**

```python
def evaluate(freeze_dir, *, qrels_loader, labels_loader):
    seal = verify_freeze(freeze_dir)
    if (
        seal["status"] != "sealed"
        or seal["qrels_opened"] is not False
        or seal["labels_opened"] is not False
    ):
        raise ValueError("verified freeze seal is required before qrels access")
    qrels = qrels_loader()
    labels = labels_loader()
    return build_evidence_metrics(load_frozen_packets(freeze_dir), qrels, labels)


def decide(evidence):
    guards = {
        "no_o0_topic_regression": all(
            topic["ADAPTIVE"]["explicit_o0_coverage"] >= topic["FIXED-O0"]["explicit_o0_coverage"]
            for topic in evidence["per_topic"].values()
        ),
        "all_o1_parent_compatible": evidence["aggregate"]["invalid_o1_count"] == 0,
        "support_precision_within_005": evidence["aggregate"]["adaptive_support_precision"] + 0.05 >= evidence["aggregate"]["fixed_support_precision"],
        "three_topics_gain_nugget": evidence["aggregate"]["topics_with_nugget_gain"] >= 3,
        "novelty_per_token_improves": evidence["aggregate"]["adaptive_nuggets_per_token"] > evidence["aggregate"]["fixed_nuggets_per_token"],
        "redundancy_not_higher": evidence["aggregate"]["adaptive_redundancy"] <= evidence["aggregate"]["fixed_redundancy"],
        "original_retention_090": evidence["aggregate"]["adaptive_original_retention"] >= 0.90 * evidence["aggregate"]["fixed_original_retention"],
        "review_agreement_interpretable": evidence["aggregate"]["review_agreement_interpretable"] is True,
    }
    return {
        "advance_to_fresh_validation": all(guards.values()),
        "production_promotion_authorized": False,
        "guards": guards,
        "failed_guards": [name for name, passed in guards.items() if not passed],
    }
```

Metrics must separate automatic from adjudicated novelty, document qrel gain from passage support, and candidate discovery from selection. Include explicit O0/O1 coverage, exact-span validity, direct-support precision, wrong-domain/source-warning/host-redundancy rates, original and novel-facet retention, tokens per supported obligation/nugget, qrel grades 2/3/4, judged rate, and continuity nDCG diagnostics.

- [ ] **Step 4: Run tests, create the freeze, and verify it before any labels or qrels**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_adaptive_evidence_evaluate.py -q
.venv/bin/python -m trec_rag.adaptive_evidence_evaluate freeze \
  --root outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1 \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/freeze
.venv/bin/python -m trec_rag.adaptive_evidence_evaluate verify-freeze \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/freeze
```

Expected: tests pass; the freeze seal hashes the contract, discovery, scores, complete continuations, nested packets, cards, and unlabeled review packet. A fresh verification process reports `status=sealed`, `qrels_opened=false`, and `labels_opened=false`.

- [ ] **Step 5: Label the frozen packet independently, then open qrels once and evaluate**

Send the already-frozen blind packet to two independent advisors. Each reviewer labels direct support, mention-only, wrong domain, usable source, redundancy, supported obligations, exact-span support, and novelty without seeing arm identities or qrels. Save their immutable outputs separately, then adjudicate disagreements without changing any packet:

- `review/labels_reviewer_a.jsonl`
- `review/labels_reviewer_b.jsonl`
- `review/labels.jsonl`

Run:

```bash
.venv/bin/python -m trec_rag.adaptive_evidence_evaluate evaluate \
  --freeze outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/freeze \
  --qrels outputs/rag25_deep_facet_candidates_v1/evaluation_v1/qrels_projection.jsonl \
  --labels outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/review/labels.jsonl \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/evaluation
.venv/bin/python -m trec_rag.adaptive_evidence_evaluate verify \
  --output outputs/rag25_deep_facet_candidates_v1/adaptive_evidence_ranker_v1/evaluation
```

Expected: both reviewer files and their adjudication receipt are immutable; freeze verification precedes the first label and qrels read; evaluation reproduces every metric, agreement statistic, and failed/passed guard; decision always states `production_promotion_authorized=false`.

- [ ] **Step 6: Commit evaluation**

```bash
git add code/trec_rag/adaptive_evidence_evaluate.py code/tests/test_adaptive_evidence_evaluate.py
git commit -m "Evaluate adaptive evidence packets"
```

---

### Task 8: Obtain findings review and render the durable HTML report

**Files:**
- Modify: `code/trec_rag/build_deep_facet_candidate_report.py`
- Modify: `code/tests/test_build_deep_facet_candidate_report.py`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/report.html`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/artifact.json`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/report_data.sqlite`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/summary.json`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/README.md`
- Create: `reports/experiments/deep_facet_candidate_pilot_v1/adaptive_evidence_advisor_review.md`

**Interfaces:**
- Consumes: verified Task 7 metrics/decision/summary plus advisor findings review.
- Produces: `load_verified_adaptive(...)`, updated portable artifact/SQLite/HTML, screenshots for local verification only, and final recommendation.

- [ ] **Step 1: Add failing report-source and answer-first tests**

```python
def test_adaptive_loader_rejects_unsealed_or_mismatched_sources(tmp_path: Path) -> None:
    metrics, decision, summary, seal = _adaptive_sources(tmp_path)
    summary["metrics_sha256"] = "0" * 64
    _write(summary, tmp_path / "summary.json")
    with pytest.raises(ValueError, match="adaptive evidence source"):
        load_verified_adaptive(
            tmp_path / "metrics.json", tmp_path / "decision.json",
            tmp_path / "summary.json", tmp_path / "SEALED.json",
        )


def test_report_explains_obligations_novelty_and_failure_stage() -> None:
    artifact, summary = build_artifact(*_legacy_inputs(), adaptive=_adaptive_evidence())
    encoded = json.dumps(artifact)
    assert "Explicit obligations" in encoded
    assert "Corpus-derived sub-obligations" in encoded
    assert "Distinct supported nuggets" in encoded
    assert "Retrieval vs scoring vs selection" in encoded
    assert summary["adaptive_evidence"]["production_promotion_authorized"] is False
```

- [ ] **Step 2: Run report tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_build_deep_facet_candidate_report.py -q
```

Expected: failures name `load_verified_adaptive` and the missing adaptive report content.

- [ ] **Step 3: Implement verified loading and new report sections**

```python
def load_verified_adaptive(metrics_path, decision_path, summary_path, seal_path):
    summary = _read_object(summary_path, "adaptive summary")
    seal = _read_object(seal_path, "adaptive seal")
    if (
        summary.get("status") != "complete"
        or summary.get("post_qrels_diagnostic") is not True
        or summary.get("metrics_sha256") != _sha256(metrics_path)
        or summary.get("decision_sha256") != _sha256(decision_path)
        or seal.get("status") != "sealed"
        or seal.get("topic_ids") != ["219", "72", "300", "84"]
    ):
        raise ValueError("adaptive evidence source hash or completion state differs")
    return (
        _read_object(metrics_path, "adaptive metrics"),
        _read_object(decision_path, "adaptive decision"),
        summary,
        seal,
    )
```

Add answer-first blocks and compact tables/charts for: all-8,114 eligibility; 8k/16k/32k packet meaning; O0/O1/N1 hierarchy; automatic versus adjudicated novelty; per-topic direct support; source/host redundancy; original/novel facet retention; failure-stage diagnosis; and why final generation remains deferred. Preserve prior findings instead of replacing them.

- [ ] **Step 4: Ask an independent advisor to review frozen findings**

Provide the design, exact source hashes, discovery acceptance/rejections, packet metrics, representative blinded passages/cards, decision guards, and report recommendation. Ask whether the evidence correctly distinguishes candidate absence, MiniLM ordering, selector behavior, atomization, qrels incompleteness, and generation. Save the response verbatim to `adaptive_evidence_advisor_review.md`; make only evidence-backed report corrections and never retune packets.

- [ ] **Step 5: Rebuild and verify report artifacts**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_adaptive_evidence_contract.py \
  code/tests/test_adaptive_evidence_score.py \
  code/tests/test_adaptive_evidence_discovery.py \
  code/tests/test_adaptive_evidence_rank.py \
  code/tests/test_adaptive_evidence_cards.py \
  code/tests/test_adaptive_evidence_evaluate.py \
  code/tests/test_build_deep_facet_candidate_report.py -q
.venv/bin/python -m trec_rag.build_deep_facet_candidate_report
google-chrome --headless --disable-gpu --hide-scrollbars \
  --window-size=1440,1200 \
  --screenshot=/tmp/adaptive-evidence-desktop.png \
  "file://$PWD/reports/experiments/deep_facet_candidate_pilot_v1/report.html"
google-chrome --headless --disable-gpu --hide-scrollbars \
  --window-size=390,844 \
  --screenshot=/tmp/adaptive-evidence-mobile.png \
  "file://$PWD/reports/experiments/deep_facet_candidate_pilot_v1/report.html"
git diff --check
```

Expected: all targeted tests pass; report builder reproduces artifact, SQLite, HTML, and summary; both screenshots render without horizontal clipping or missing content; no external runtime dependency exists; diff check is clean.

- [ ] **Step 6: Render the report in the artifact viewer and commit scoped files**

Open the rebuilt `artifact.json` in the artifact viewer, confirm the ready snapshot and source-backed native charts/tables, and show the rendered HTML to the user. Then run:

```bash
git add \
  code/trec_rag/build_deep_facet_candidate_report.py \
  code/tests/test_build_deep_facet_candidate_report.py \
  reports/experiments/deep_facet_candidate_pilot_v1/report.html \
  reports/experiments/deep_facet_candidate_pilot_v1/artifact.json \
  reports/experiments/deep_facet_candidate_pilot_v1/report_data.sqlite \
  reports/experiments/deep_facet_candidate_pilot_v1/summary.json \
  reports/experiments/deep_facet_candidate_pilot_v1/README.md \
  reports/experiments/deep_facet_candidate_pilot_v1/adaptive_evidence_advisor_review.md
git commit -m "Report adaptive evidence ranker findings"
```

Expected: only the listed report files are staged; unrelated sparse-relevance files remain untracked and untouched.

---

## Final verification

- [ ] Run the complete targeted suite again from a clean process.
- [ ] Verify every create-only seal and receipt from contract through evaluation.
- [ ] Confirm all four continuations represent the same 8,114 unique topic-document identities.
- [ ] Confirm 8k is a prefix of 16k and 16k is a prefix of 32k for every topic and arm.
- [ ] Confirm qrels and labels were first opened only after the freeze seal.
- [ ] Confirm protected-topic counters are zero everywhere.
- [ ] Confirm all external/network/download/paid-call counters are zero.
- [ ] Inspect desktop and mobile screenshots and the artifact-viewer render.
- [ ] Run `git status --short` and verify only pre-existing unrelated untracked files remain.

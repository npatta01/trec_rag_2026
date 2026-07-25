# Topic 213 Evidence Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Topic 213 handover containing the five strongest reviewed ClimbMix documents and their supported claims for each of ten sub-narratives.

**Architecture:** A reusable Python module joins the pinned topic, nuggets, qrels, and authenticated accepted-union text into 173 eligible document records. A pinned local Mixedbread cross-encoder scores all 1,730 document/sub-narrative pairs for shortlisting; evidence reviewers inspect the leading candidates and produce sanitized final mappings consumed by deterministic JSON and Markdown renderers.

**Tech Stack:** Python 3.12.13, pytest, PyTorch/ROCm, Transformers, sentence-transformers, JSON/JSONL, YAML, SHA-256.

## Global Constraints

- Use Topic 213 and the pinned `trec-rag-data` submodule revision.
- Eligible documents have Codex UMBRELA qrel grade 2, 3, or 4; require exactly 173.
- Preserve the exact ten distinct `mapped_sub_narrative` strings from the released nuggets.
- Keep organizer qrel grade separate from derived sub-narrative support score.
- Score whole documents and omit passage offsets/locations.
- Require exactly five unique final documents per sub-narrative with support score 2 or 3.
- Do not track raw document text, credentials, absolute home paths, or model caches.
- Treat unjudged documents as unknown, not nonrelevant.
- Use the cached `mixedbread-ai/mxbai-rerank-base-v2` revision already pinned by the repository; do not download a new model.

---

### Task 1: Deterministic Topic 213 input and artifact contracts

**Files:**
- Create: `code/trec_rag/topic_evidence_handover.py`
- Create: `code/tests/test_topic_evidence_handover.py`

**Interfaces:**
- Consumes: topic TSV, nugget JSONL, qrels, accepted-union JSONL.
- Produces: `load_topic213_inputs(...) -> TopicEvidenceInputs`, `validate_reviewed_handover(...) -> None`, and `render_handover_markdown(...) -> str`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_load_topic213_inputs_preserves_population_and_subnarratives(fixture_paths):
    loaded = load_topic213_inputs(**fixture_paths)
    assert loaded.topic_id == "213"
    assert len(loaded.sub_narratives) == 10
    assert len(loaded.documents) == 3


def test_handover_requires_five_supported_unique_documents(sample_handover):
    sample_handover["sub_narratives"][0]["documents"][0]["support_score"] = 1
    with pytest.raises(ValueError, match="support_score"):
        validate_reviewed_handover(
            sample_handover,
            eligible_docids={f"doc-{index}" for index in range(1, 51)},
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py
```

Expected: collection fails because `trec_rag.topic_evidence_handover` does not exist.

- [ ] **Step 3: Implement typed loaders, validators, and Markdown rendering**

The module must:

- parse qrels with integer grades and retain only grades 2 through 4;
- extract exact unique `mapped_sub_narrative` values for Topic 213;
- join eligible docids to accepted-union `document_id` and `text`;
- reject missing text, duplicate document IDs, a population other than 173 in
  canonical mode, and final records violating the global constraints;
- render claims, scores, and document identifiers without raw document text.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add code/trec_rag/topic_evidence_handover.py code/tests/test_topic_evidence_handover.py
git commit -m "build topic evidence handover contracts"
```

---

### Task 2: Offline Mixedbread pairwise shortlisting

**Files:**
- Modify: `code/trec_rag/topic_evidence_handover.py`
- Modify: `code/tests/test_topic_evidence_handover.py`
- Create generated scratch: `outputs/rag25_topic213_evidence_handover_v1/shortlist.json`

**Interfaces:**
- Consumes: `TopicEvidenceInputs`.
- Produces: `score_sub_narrative_pairs(inputs, scorer, *, shortlist_depth=12) -> dict[str, object]` and CLI command `shortlist`.

- [ ] **Step 1: Write failing score-order and provenance tests**

```python
def test_shortlist_scores_every_pair_and_orders_descending(sample_inputs):
    result = score_sub_narrative_pairs(sample_inputs, deterministic_scorer, shortlist_depth=2)
    assert result["pair_count"] == len(sample_inputs.sub_narratives) * len(sample_inputs.documents)
    assert all(
        row["model_score"] >= next_row["model_score"]
        for rows in result["shortlists"].values()
        for row, next_row in pairwise(rows)
    )


def test_shortlist_keeps_qrel_grade_separate_from_model_score(sample_inputs):
    row = score_sub_narrative_pairs(sample_inputs, deterministic_scorer)["shortlists"]["SN1"][0]
    assert isinstance(row["topic_qrel_grade"], int)
    assert isinstance(row["model_score"], float)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py -k shortlist
```

Expected: tests fail because the scoring interface is missing.

- [ ] **Step 3: Implement injected scoring and the pinned-model CLI**

The CLI must score every one of the 1,730 pairs, chunk long documents through
the repository's existing chunking policy, retain the strongest chunk score
per pair, store only the leading 12 candidates per sub-narrative, and record:

```json
{
  "document_id": "shard_04810_73050",
  "topic_qrel_grade": 4,
  "model_score": 1.25,
  "model_rank": 1
}
```

The scratch shortlist may include document text for review but remains ignored
under `outputs/`; the tracked handover must not.

- [ ] **Step 4: Run tests, a two-document smoke run, and the full offline run**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py
.venv/bin/python -m trec_rag.topic_evidence_handover shortlist --limit-documents 2
.venv/bin/python-rocm -m trec_rag.topic_evidence_handover shortlist
```

Expected: the smoke run reports 20 pairs; the full run reports 1,730 pairs and
ten 12-document shortlists without network calls or model downloads.

- [ ] **Step 5: Commit Task 2**

```bash
git add code/trec_rag/topic_evidence_handover.py code/tests/test_topic_evidence_handover.py
git commit -m "score topic sub-narrative evidence"
```

---

### Task 3: Reviewed top-five handover and colleague documentation

**Files:**
- Create: `reports/experiments/rag25_topic213_evidence_handover_v1/README.md`
- Create: `reports/experiments/rag25_topic213_evidence_handover_v1/handover.json`
- Create: `reports/experiments/rag25_topic213_evidence_handover_v1/handover.md`
- Create: `reports/experiments/rag25_topic213_evidence_handover_v1/manifest.yaml`
- Modify: `code/tests/test_topic_evidence_handover.py`

**Interfaces:**
- Consumes: Task 2 shortlist and evidence-review JSON records.
- Produces: the sanitized durable colleague handover.

- [ ] **Step 1: Independently review the top candidates**

For every sub-narrative, inspect at least the leading eight model-ranked
documents. Assign `support_score` 0 through 3, name the concrete claims
supported by the whole document, and record a concise rationale. Select five
documents with `support_score >= 2`, preferring direct, detailed, and
non-redundant coverage.

- [ ] **Step 2: Write failing canonical artifact tests**

```python
def test_canonical_handover_has_ten_by_five_supported_documents():
    payload = json.loads(HANDOVER.read_text())
    validate_reviewed_handover(payload, eligible_docids=canonical_eligible_docids())
    assert len(payload["sub_narratives"]) == 10
    assert all(len(row["documents"]) == 5 for row in payload["sub_narratives"])


def test_canonical_handover_is_sanitized():
    text = HANDOVER.read_text() + MARKDOWN.read_text() + MANIFEST.read_text()
    assert "/home/" not in text
    assert "PYSERINI_API_TOKEN" not in text
    assert '"text":' not in HANDOVER.read_text()
```

- [ ] **Step 3: Run canonical tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py -k canonical
```

Expected: tests fail because the tracked handover files do not exist.

- [ ] **Step 4: Build and render the reviewed handover**

Write `handover.json`, render `handover.md` through the production renderer,
and write a manifest with pinned source revisions, model identity, exact
population counts, scoring definitions, limitations, and SHA-256 hashes.

- [ ] **Step 5: Verify the complete artifact**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_topic_evidence_handover.py
.venv/bin/python -m pytest -q
git diff --check
```

Expected: all focused and repository tests pass, all ten sub-narratives have
five valid documents, and no sanitization check fails.

- [ ] **Step 6: Commit Task 3**

```bash
git add reports/experiments/rag25_topic213_evidence_handover_v1 code/tests/test_topic_evidence_handover.py
git commit -m "curate topic 213 response evidence"
```

# Agentic Facet Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every coordinator-accepted evidence target before researcher handoffs so production agentic closeout cannot create a dangling facet reference.

**Architecture:** Preserve the existing researcher-proposes/coordinator-admits ownership model. DeepAgent finalization derives the referential closure of live nugget facet-or-need IDs and admits those canonical identities through `_ensure_ledger_facets` before projecting handoffs; the SQLite integrity guard remains the final authority.

**Tech Stack:** Python 3.12, pytest, `EvidenceCoverageState`, `DeepAgentRetriever`, `TopicRecordsBuilder`, SQLite.

## Global Constraints

- Do not change researcher or coordinator prompts.
- Do not change `EvidenceBundle`, `CandidateNugget`, or `ResearchTaskEnvelope` schemas.
- Do not change the SQLite schema or cache identities.
- Do not weaken `TopicRecordsBuilder` referential-integrity checks.
- Do not infer researcher authorship for a coordinator-admitted facet.
- Test the production merge sequence through the real SQLite-backed `TopicRecordsBuilder`.

---

### Task 1: Production-shaped regression and minimal finalization repair

**Files:**
- Modify: `code/tests/test_deepagent_retrieval.py`
- Modify: `code/trec_rag/deepagent_retrieval.py:1775-1785`

**Interfaces:**
- Consumes: `DeepAgentRetriever.retrieve(records, narrative)`, `_ensure_ledger_facets(facet_ids, query)`, and the immutable `EvidenceCoverageReport` returned by `coverage_state.report()`.
- Produces: the existing `AgentRetrievalResult`; no public interface or schema changes.

- [ ] **Step 1: Write the failing real-SQLite regression**

Add this test beside the existing direct-passage/direct-need durable-ledger tests:

```python
def test_coordinator_discovered_facet_is_admitted_before_researcher_handoff(
    tmp_path,
) -> None:
    store = DocumentStore(tmp_path / "objects")
    passage_search = _OffsetPassageSearch(store)
    records = TopicRecordsBuilder(
        tmp_path / "topic",
        passage_search.topic_id,
        store,
        run_id="run-1",
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_needs(toolset, "need-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="need-1",
                query="discover grounded facet",
            )
            passage = payload["passages"][0]
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_facets": [
                            {
                                "facet_id": "discovered-facet",
                                "need_ids": ["need-1"],
                                "dimension": "discovered",
                                "value": "Coordinator accepted discovered facet",
                                "origin": "snippet",
                                "origin_snippet_id": passage["passage_id"],
                            }
                        ],
                        "add_nuggets": [
                            {
                                "nugget_id": "discovered-nugget",
                                "text": passage_search.passage_text,
                                "need_ids": ["need-1"],
                                "facet_ids": ["discovered-facet"],
                                "evidence": [{"cite": passage["cite"]}],
                            }
                        ],
                    }
                )
            )
            assert set(update["accepted_ids"]) == {
                "discovered-facet",
                "discovered-nugget",
            }
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        passage_search=passage_search,
    ).retrieve(records, "narrative")

    snapshot = result.topic_snapshot
    facet = next(
        row for row in snapshot.facets
        if row.subnarrative_id == "discovered-facet"
    )
    assert facet.origin == "research_discovered"
    assert snapshot.researcher_handoffs[0].facet_updates == ()
    assert [
        (row.subnarrative_id, row.passage_id)
        for row in snapshot.researcher_evidence
    ] == [("discovered-facet", passage_search.passage_id)]
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_deepagent_retrieval.py::test_coordinator_discovered_facet_is_admitted_before_researcher_handoff \
  -q
```

Expected: FAIL at `TopicRecordsBuilder.add_researcher_handoff` with
`TopicRecordsIntegrityError: researcher evidence facet is not admitted`.

- [ ] **Step 3: Implement the minimal referential-closure admission**

Replace the direct-need-only block at the start of
`commit_researcher_handoffs` with:

```python
evidence_subnarrative_ids = tuple(
    dict.fromkeys(
        subnarrative_id
        for nugget in nuggets
        for subnarrative_id in (nugget.facet_ids or nugget.need_ids)
    )
)
if evidence_subnarrative_ids:
    _ensure_ledger_facets(
        evidence_subnarrative_ids,
        "Grounded researcher evidence",
    )
```

Do not change handoff creator attribution or `TopicRecordsBuilder`.

- [ ] **Step 4: Run the regression and verify GREEN**

Run the Step 2 command again.

Expected: PASS; the durable facet has origin `research_discovered`, evidence
references it, and the handoff has no invented `facet_updates` provenance.

- [ ] **Step 5: Run the focused neighboring tests**

```bash
.venv/bin/python -m pytest \
  code/tests/test_deepagent_retrieval.py::test_coordinator_discovered_facet_is_admitted_before_researcher_handoff \
  code/tests/test_deepagent_retrieval.py::test_direct_passage_citation_retains_exact_source_passage_and_offsets \
  code/tests/test_deepagent_retrieval.py::test_direct_need_citation_retains_exact_source_passage_and_offsets \
  code/tests/test_deepagent_retrieval.py::test_direct_multi_need_citation_admits_each_need_and_deduplicates_passage \
  code/tests/test_deepagent_retrieval.py::test_two_researchers_commit_only_their_evidence_and_discovered_facets \
  code/tests/test_deepagent_retrieval.py::test_shared_passage_discovered_facet_is_handed_off_only_by_its_creator \
  -q
```

Expected: six tests pass, including the new regression.

- [ ] **Step 6: Commit the repair**

```bash
git add code/tests/test_deepagent_retrieval.py code/trec_rag/deepagent_retrieval.py
git commit -m "fix: admit agentic facets before handoff"
```

### Task 2: Broader local verification

**Files:**
- Verify: `code/tests/test_deepagent_retrieval.py`
- Verify: `code/tests/test_deepagent_evidence.py`
- Verify: `code/tests/test_topic_records.py`
- Verify: `code/tests/test_deepagent_submission.py`
- Verify: `code/tests/test_retrieval_path_parity.py`

**Interfaces:**
- Consumes: the unchanged public DeepAgent, evidence-state, durable-ledger, submission, and shared-retrieval interfaces.
- Produces: verification evidence only.

- [ ] **Step 1: Run the related suites**

```bash
.venv/bin/python -m pytest \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_topic_records.py \
  code/tests/test_deepagent_submission.py \
  code/tests/test_retrieval_path_parity.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Check patch and repository hygiene**

```bash
git diff --check
git status --short
git submodule status --recursive
```

Expected: no whitespace errors, only intentional branch commits, and pinned
submodules without leading `+` or `-` markers.

### Task 3: Fresh live agentic validation

**Files:**
- Reuse privately: `/tmp/run_deepagent_e2e_7e9d3b9.py`, updated with the repair commit and a new run ID.
- Create privately: a new ignored `outputs/deepagent-e2e-*` directory.
- Create privately: new `/tmp/trec-rag-deepagent-cache-*` and model-runtime directories.

**Interfaces:**
- Consumes: one `rag2026-0` narrative, the checked-in retrieval smoke config copy, authenticated ClimbMix/Pyserini, local ROCm Mixedbread reranking, and OpenRouter DeepSeek.
- Produces: a sealed topic ledger, agentic retrieval result, TREC run, and manifest-last run receipt in a namespace never used by the failed run.

- [ ] **Step 1: Preflight the live run**

Verify the repair commit, clean tracked tree, pinned submodules, one selected
topic, fresh nonexistent output/cache/runtime paths, secret presence without
printing values, GPU access, and the pinned local model snapshot.

- [ ] **Step 2: Report scope before external calls**

Report one topic, zero cache reuse, expected hosted planning/research calls,
the actual default `ResearchBudgetConfig`, and all private output/cache paths.

- [ ] **Step 3: Run with the actual default budget**

Use `.venv/bin/python-rocm` with the fresh runtime and cache paths. Do not add a
researcher cap and do not reuse the failed run's 19 retrieval responses or
99,192 score records.

- [ ] **Step 4: Validate publication**

Require exit code zero, a sealed `records.sqlite3` plus manifest, a valid TREC
run with contiguous ranks, receipt hashes matching artifact bytes, no dangling
facet/evidence identities, unchanged fixed retrieval/RAG hashes, and a clean
tracked tree.

# Shared Topic-First Passage Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fixed facet retrieval and DeepAgent research use one topic-scoped implementation that retrieves 1,000 documents for each focused query, ranks passages from the complete returned pool with pinned Mixedbread, exposes the top 100 passages, and persists a citation-valid per-topic retrieval ledger that can be completed or explicitly incomplete.

**Architecture:** A deep `TopicPassageSearch` module owns organizer retrieval, exact document admission, deterministic passage geometry, cached Mixedbread scoring, and deterministic top-100 selection. Fixed and agentic orchestration become thin adapters over that interface; one sealed `records.sqlite3` per topic stores source bindings, query/candidate/passage provenance, researcher evidence, coverage, and completion state. A local process dispatcher executes independent topic jobs and publishes the global manifest only after every selected topic has produced a sealed receipt.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `concurrent.futures`, `hashlib`, `os`, `tempfile`, existing `DocumentStore`, `RetrievalCache`, `SemanticTextChunker`, `GlobalScoreCache`, pinned `mixedbread-ai/mxbai-rerank-base-v2`, and pytest.

## Current Code Map

- `code/trec_rag/facet_retrieval.py` retrieves 1,000 documents per lane but scores only the first 100, combines document and window scores, and selects 100 **documents**. This is the active fixed/non-agentic path.
- `code/trec_rag/deepagent_retrieval.py` retrieves 1,000 documents but retains and scores only the first 100; `code/trec_rag/deepagent_passages.py` returns 16 diversity-constrained passages. This is the active agentic path.
- `code/trec_rag/deepagent_snippets.py` and `code/trec_rag/facet_retrieval.py` contain separate Mixedbread/chunk/cache implementations.
- `code/trec_rag/competition_retrieval.py::_run_official` loops pending topics sequentially even though each topic already has a disjoint output directory and `records.sqlite3`.
- `code/trec_rag/document_store.py`, `code/trec_rag/retrieval_cache.py`, and `code/trec_rag/topic_records.py` already provide the exact-text CAS, immutable request cache, and sealed one-topic database foundations. Extend these foundations; do not create a second document store or a second database per topic.

## File Structure

- Create `code/trec_rag/topic_passage_search.py`: the deep shared query-to-top-passages module and its typed interface.
- Create `code/trec_rag/mixedbread_passage_scorer.py`: the one production Mixedbread adapter used by both orchestration modes.
- Create `code/trec_rag/topic_dispatch.py`: process-safe topic job/receipt dispatcher and deterministic source-order collection.
- Create `code/tests/test_topic_passage_search.py`: interface-level tests for retrieval, geometry, scoring, cache reuse, ordering, retries, and incomplete outcomes.
- Create `code/tests/test_mixedbread_passage_scorer.py`: no-model cache and batching tests for the shared production scorer.
- Create `code/tests/test_retrieval_path_parity.py`: fixed-versus-agentic parity tests through their public adapters.
- Create `code/tests/test_topic_dispatch.py`: two-topic concurrency, resume, worker failure, and manifest-last tests.
- Modify `code/trec_rag/remote_client.py` and `code/tests/test_remote_pyserini.py`: pin one organizer call deadline to 300 seconds.
- Modify `code/trec_rag/facet_pilot_config.py` and config tests in `code/tests/test_facet_pipeline_e2e.py`; create `configs/rag26_competition_retrieval_v2.yaml` and delete `configs/rag26_competition_retrieval_v1.yaml`: replace document rerank knobs with the passage contract and add local topic-worker count.
- Modify `code/trec_rag/facet_retrieval.py` and `code/tests/test_facet_retrieval.py`: make fixed lanes consume shared passage results and derive document export order from the same scores.
- Modify `code/trec_rag/deepagent_retrieval.py`, `code/trec_rag/deepagent_passages.py`, `code/tests/test_deepagent_retrieval.py`, and `code/tests/test_deepagent_passages.py`: route `search_passages` through the shared module and retire the separate depth-100/top-16 implementation.
- Modify `code/trec_rag/topic_records.py` and `code/tests/test_topic_records.py`: add query, retrieval candidate, scored passage, researcher evidence, facet coverage, and completion tables to the one-topic database.
- Modify `code/trec_rag/competition_retrieval.py` and its tests in `code/tests/test_facet_pipeline_e2e.py`: dispatch topic jobs and collect sealed receipts.
- Modify `code/trec_rag/retrieval_export.py` and `code/tests/test_retrieval_export.py` only when the active dirty changes have been preserved: include retrieval completion state in the private sealed handoff without changing organizer-facing TREC columns.
- Modify `code/trec_rag/README.md`: document the shared fixed/agentic contract, worker control, resume, and two-topic smoke.

## Global Constraints

- A focused query has one primary information need. A retained passage may support that primary subnarrative plus zero or more additional subnarratives.
- Each query makes one organizer request per attempt with `hits=1000`; bounded retry uses at most three attempts of the identical request. There is no query fallback, model fallback, or method fallback.
- Each organizer HTTP call has a 300-second deadline.
- Deterministically chunk every non-empty document returned in the depth-1,000 pool and Mixedbread-score every resulting passage. There is no document-prefix rerank depth.
- Return the global top 100 passages by `(-raw_logit, source_document_rank, passage_id)`. Do not impose the old top-16 limit or a hidden per-document cap.
- Keep document citations and exact source coordinates on every passage: `docid`, content SHA-256, source character offsets, source UTF-8 byte offsets, and source text SHA-256.
- The model-score cache may deduplicate mathematically identical `(query text, passage text, scorer identity)` computations. The topic ledger must separately bind that score to the full source passage identity and reject a conflicting binding.
- Store exact document content once in `cache/documents/v1`; topic databases reference content hashes and offsets. Never provide a global `docid -> text` lookup.
- Use exactly one `records.sqlite3` per topic. Creating/opening the topic workspace establishes topic identity; normal ledger mutation and snapshot methods do not accept `topic_id`.
- Researcher handoffs merge only when run ID and topic workspace match. Researchers validate their own cited passage handles before commit; the coordinator consumes one holistic topic snapshot and does not repeat retrieval or scoring.
- Persist useful validated work when a deadline, retry limit, researcher budget, or absent evidence prevents completion. Mark it `incomplete` with an enumerated stopping reason; never relabel it complete. Generation may consume an incomplete sealed handoff.
- A topic is the unit of execution, storage, retry, resume, and optional sharding. Local workers have disjoint topic directories and databases.
- Same topic/run publication is idempotent only for byte-identical sealed output. Different bytes are an integrity conflict; do not use unconstrained last-writer-wins.
- Export topics in official source order and publish the global manifest last.
- Existing cache/artifact schemas need no backward compatibility. New code rejects old schema identities; operators may delete old cache/artifact roots separately.
- Preserve all user-owned dirty work. In particular, do not overwrite concurrent edits in `competition_rag.py`, `retrieval_export.py`, canonical Nuggetizer files, or their tests; reconcile at the exact hunk if a planned task reaches them.
- Use test-driven development: demonstrate a focused RED failure before implementation, then run the focused GREEN command recorded in each task report.
- Do not start a full live retrieval, hosted generation, or full 2025 run under this plan. The live gate after implementation is an explicitly authorized two-topic smoke.

---

### Task 1: Shared Passage Search Interface and Deterministic Geometry

**Files:**
- Create: `code/trec_rag/topic_passage_search.py`
- Create: `code/tests/test_topic_passage_search.py`

**Interfaces:**
- Produces `PASSAGE_SEARCH_SCHEMA_VERSION = "topic-passage-search-v1"`.
- Produces `OrganizerRequestFailed(RuntimeError)` for a retryable organizer transport attempt and `PassageScoringFailed(RuntimeError)` for a classified model/runtime scoring failure. `search` catches only these two operational exceptions; schema, source-binding, score-shape, and programming errors remain loud integrity failures.
- Produces `PassageSearchPolicy(retrieval_depth=1000, passage_limit=100, max_attempts=3)`; all values are positive non-boolean integers and production values are pinned by config.
- Produces `FocusedQuery(query_id: str, text: str, primary_subnarrative_id: str, supporting_subnarrative_ids: tuple[str, ...] = ())`.
- Produces `SourceDocument(docid, content_sha256, source_rank, source_score, best_passage_id, best_passage_raw_logit)`; it contains no body text. The two best-passage fields are `None` only when a non-empty scored passage does not exist for that document.
- Produces `SourcePassage(passage_id, docid, content_sha256, source_rank, source_score, start_char, end_char, start_byte, end_byte, text_sha256, text, raw_logit, rank, score_cache_key)`.
- Produces `PassageSearchResult(query, status, stopping_reason, requested_documents, returned_documents, scored_documents, scored_passages, documents, passages, attempt_count, source_exhausted)`; `documents` holds the compact ordered depth-1,000 candidate closure and `passages` holds only the top 100 source-bound rows.
- Produces `TopicPassageSearch(topic_id, document_store, retriever, chunker, scorer, policy).search(query) -> PassageSearchResult`.
- Consumes a retriever adapter with `retrieve(QueryVariant, *, depth: int) -> Sequence[RetrievedCandidate]` and a scorer adapter with `identity`, `cache_key(query_text, passage_text)`, and `rank(query_text, chunks)`. The shared module passes its exact policy depth on every attempt, so the 1,000-document contract is explicit rather than inferred from external config.

- [x] **Step 1: Write failing validation and identity tests**

```python
def test_search_stops_after_three_identical_failed_attempts(tmp_path: Path) -> None:
    retriever = RecordingUnavailableRetriever()
    result = make_search(tmp_path, retriever=retriever).search(focused_query())
    assert result.status == "incomplete"
    assert result.stopping_reason == "retrieval_unavailable"
    assert result.attempt_count == 3
    assert retriever.queries == [focused_query().text] * 3

def test_focused_query_has_one_primary_and_deduplicated_supporting_facets() -> None:
    query = FocusedQuery("q-1", "focused text", "facet-a", ("facet-b", "facet-b"))
    assert query.primary_subnarrative_id == "facet-a"
    assert query.supporting_subnarrative_ids == ("facet-b",)
```

- [x] **Step 2: Run the focused RED tests**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_passage_search.py -q`

Expected: collection fails because `trec_rag.topic_passage_search` does not exist.

- [x] **Step 3: Implement exact passage identity and coordinate projection**

Use one query-independent but document-citation-specific passage ID so the same source span can be referenced by several facets without becoming ambiguous across duplicate-content docids:

```python
def passage_id(content_sha256: str, docid: str, chunk: TextChunk, chunker_identity: Mapping[str, object]) -> str:
    body = json.dumps(
        {
            "schema_version": PASSAGE_SEARCH_SCHEMA_VERSION,
            "content_sha256": content_sha256,
            "docid": docid,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest(),
            "chunker": dict(chunker_identity),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "p-" + sha256(body).hexdigest()
```

Compute byte offsets from the exact source body, not normalized scoring text:

```python
start_byte = len(document_text[: chunk.start_char].encode("utf-8"))
end_byte = len(document_text[: chunk.end_char].encode("utf-8"))
assert document_text[chunk.start_char:chunk.end_char] == chunk.text
```

- [x] **Step 4: Implement the deep `search` operation**

The method must:

1. construct one `QueryVariant` from the topic-scoped constructor identity and `FocusedQuery`;
2. retry only `OrganizerRequestFailed` for the identical retriever request and exact policy depth up to `max_attempts`;
3. sort candidates by source rank, reject conflicting duplicate `(docid, body)`, and retain at most `retrieval_depth`;
4. admit every exact body through `DocumentStore`;
5. chunk every non-empty retained body;
6. make one scorer call over the full chunk sequence;
7. validate one returned score per exact chunk;
8. build source-bound passages and sort by `(-raw_logit, source_rank, passage_id)`;
9. derive each compact source document's best passage from the complete scored set;
10. assign one-based passage ranks and return the first `passage_limit` rows.

Return `status="incomplete"` and an enumerated reason for exhausted `OrganizerRequestFailed`, one `PassageScoringFailed`, or `no_evidence`; retain validated prior attempt metadata but never invent passages. Do not catch `ValueError`, `TypeError`, `DocumentStoreIntegrityError`, conflicting docids, malformed scorer output, or other invariant failures. A successful fully scored non-empty response is `complete`; a successful response with fewer than 1,000 documents sets `source_exhausted=True` without hiding the returned count.

- [x] **Step 5: Add all-depth, global-top-100, Unicode, and deterministic-tie tests**

```python
def test_search_scores_passages_from_every_returned_document(tmp_path: Path) -> None:
    candidates = fake_candidates(1000, one_chunk_each=True)
    scorer = RecordingScorer(score=lambda chunk: float(chunk.document_id.removeprefix("d")))
    result = make_search(tmp_path, candidates, scorer).search(focused_query())
    assert result.returned_documents == 1000
    assert result.scored_documents == 1000
    assert result.scored_passages == 1000
    assert len(result.passages) == 100
    assert result.passages[0].docid == "d999"
    assert scorer.chunk_count == 1000

def test_passage_byte_offsets_are_from_exact_utf8_source(tmp_path: Path) -> None:
    result = make_search(tmp_path, [candidate("d1", "A café Ω end")], ConstantScorer()).search(focused_query())
    row = result.passages[0]
    assert row.text.encode("utf-8") == "A café Ω end".encode("utf-8")[row.start_byte:row.end_byte]
```

- [x] **Step 6: Run GREEN and commit only Task 1 files**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_passage_search.py -q`

Commit: `feat: add shared topic passage search`

---

### Task 2: One Cached Mixedbread Passage Scorer and 300-Second Organizer Deadline

**Files:**
- Create: `code/trec_rag/mixedbread_passage_scorer.py`
- Create: `code/tests/test_mixedbread_passage_scorer.py`
- Modify: `code/trec_rag/remote_client.py:106-132`
- Modify: `code/tests/test_remote_pyserini.py:140-175`

**Interfaces:**
- Produces `MixedbreadPassageScorer(score_cache_root: Path, device: str = "auto", model_loader: Callable = load_pinned_cross_encoder, batch_size: int = 8)` implementing the Task 1 scorer interface.
- Uses one `GlobalScoreCache` context with model `mixedbread-ai/mxbai-rerank-base-v2`, raw logits, the repository-pinned revision/backend/dtype, `max_length=1024`, and input policy `topic_passage_query_text_v1`.
- `cache_key(query_text, passage_text)` delegates to the exact `GlobalScoreCache` context.
- `rank(query_text, chunks)` returns rows in input order even when cache hits and misses are mixed.
- `RemotePyseriniClient(config, session=None, timeout=300)` is the production default; explicit test overrides remain supported.

- [x] **Step 1: Write RED tests proving cache reuse and input-order stability**

```python
def test_scorer_reuses_cached_passage_scores_without_loading_model(tmp_path: Path) -> None:
    first_model = FakeModel([0.1, 0.9])
    first = MixedbreadPassageScorer(tmp_path, model_loader=lambda **_: first_model)
    expected = first.rank("query", chunks("alpha", "beta"))
    replay = MixedbreadPassageScorer(
        tmp_path,
        model_loader=lambda **_: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    assert replay.rank("query", chunks("alpha", "beta")) == expected

def test_scorer_preserves_input_order_across_hits_and_misses(tmp_path: Path) -> None:
    scorer = MixedbreadPassageScorer(tmp_path, model_loader=fake_loader([0.4, 0.2]))
    rows = scorer.rank("query", chunks("second", "first"))
    assert [row.chunk.text for row in rows] == ["second", "first"]
```

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_mixedbread_passage_scorer.py code/tests/test_remote_pyserini.py -q`

Expected: import failure for the new scorer and the existing timeout assertion reports 30 instead of 300.

- [x] **Step 3: Implement the scorer by deepening existing cache mechanics**

Reuse `GlobalScoreCache.score_many`; do not add another cache database or JSON result cache. Lazily load the pinned cross encoder only when at least one pair misses. Validate all scores as finite floats and keep the full scorer identity JSON serializable.

- [x] **Step 4: Change the organizer default deadline to 300 seconds**

Change only the constructor default and its assertion:

```python
class RemotePyseriniClient:
    def __init__(self, config: RemotePyseriniConfig, session: requests.Session | None = None, timeout: int = 300) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
```

- [x] **Step 5: Run GREEN and the existing score-cache concurrency tests**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_mixedbread_passage_scorer.py code/tests/test_remote_pyserini.py code/tests/test_rerank_score_cache.py -q`

Commit: `refactor: share cached Mixedbread passage scoring`

---

### Task 3: Persist Shared Searches in the One Topic Database

**Files:**
- Modify: `code/trec_rag/topic_records.py:52,613-717,1316-1701`
- Modify: `code/tests/test_topic_records.py`
- Modify: `code/trec_rag/topic_passage_search.py`
- Modify: `code/tests/test_topic_passage_search.py`

**Interfaces:**
- Bumps the incompatible database identity to `TOPIC_RECORDS_SCHEMA_VERSION = "topic-records-v3"`.
- Adds `TopicRecordsBuilder.add_passage_search(result: PassageSearchResult) -> None` and `add_researcher_handoff(handoff: ResearcherHandoff) -> None`; neither accepts `topic_id`.
- Adds `TopicRecordsBuilder.add_facets(facets: Sequence[FacetRecord]) -> None`, where `FacetRecord(subnarrative_id, text, origin)` has `origin` equal to `initial` or `research_discovered`; every query and researcher evidence row references an admitted facet.
- Adds `TopicRecordsBuilder.set_completion(status: Literal["complete", "incomplete"], stopping_reason: str) -> None`.
- Adds `TopicRecordsBuilder.topic_snapshot() -> TopicEvidenceSnapshot` so the agentic coordinator can materialize the current unsealed ledger without opening another database or passing a topic ID.
- Adds `TopicRecords.passage_search(query_id: str) -> PassageSearchSnapshot` and `TopicRecords.topic_snapshot() -> TopicEvidenceSnapshot`; the latter is the coordinator's holistic single lookup.
- Adds `ResearcherHandoff(run_id, researcher_id, round_index, evidence, facet_updates)` and validates passage IDs against already stored source-bound passages.
- Topic/run identity is established once by `TopicRecordsBuilder(destination, topic_id, document_store, *, run_id)`.

- [x] **Step 1: Write RED schema and no-topic-argument tests**

```python
def test_topic_records_persists_search_without_repeating_document_text(tmp_path: Path) -> None:
    builder, store = make_builder(tmp_path, topic_id="t1", run_id="r1")
    result = complete_passage_result(document_text="exact body", passage_text="exact body")
    builder.add_passage_search(result)
    published = builder.publish({"run_id": "r1"})
    with open_published(published, store) as records:
        snapshot = records.passage_search("q1")
        assert snapshot.passages[0].text == "exact body"
    assert b"exact body" not in published.database_path.read_bytes()

def test_ordinary_ledger_methods_do_not_accept_topic_id() -> None:
    assert list(inspect.signature(TopicRecordsBuilder.add_passage_search).parameters) == ["self", "result"]
    assert list(inspect.signature(TopicRecords.topic_snapshot).parameters) == ["self"]
```

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_records.py -k 'passage_search or researcher_handoff or topic_snapshot or completion' -q`

Expected: failures because the v3 methods and tables do not exist.

- [x] **Step 3: Add normalized v3 tables**

Add strict foreign-key tables for:

```sql
CREATE TABLE query_identity(
  query_id TEXT PRIMARY KEY,
  query_text_sha256 TEXT NOT NULL,
  primary_subnarrative_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('complete','incomplete')),
  stopping_reason TEXT,
  attempt_count INTEGER NOT NULL,
  requested_documents INTEGER NOT NULL,
  returned_documents INTEGER NOT NULL,
  scored_documents INTEGER NOT NULL,
  scored_passages INTEGER NOT NULL,
  source_exhausted INTEGER NOT NULL CHECK(source_exhausted IN (0,1))
) STRICT;
CREATE TABLE retrieval_candidate(
  query_id TEXT NOT NULL,
  document_pk INTEGER NOT NULL,
  source_rank INTEGER NOT NULL,
  source_score REAL NOT NULL,
  best_passage_id TEXT,
  best_passage_raw_logit REAL,
  PRIMARY KEY(query_id, source_rank),
  UNIQUE(query_id, document_pk),
  FOREIGN KEY(query_id) REFERENCES query_identity(query_id),
  FOREIGN KEY(document_pk) REFERENCES document_binding(document_pk)
) STRICT;
CREATE TABLE query_passage(
  query_id TEXT NOT NULL,
  passage_pk INTEGER NOT NULL,
  raw_logit REAL NOT NULL,
  passage_rank INTEGER NOT NULL,
  score_cache_key TEXT NOT NULL,
  PRIMARY KEY(query_id, passage_pk),
  UNIQUE(query_id, passage_rank),
  FOREIGN KEY(query_id) REFERENCES query_identity(query_id),
  FOREIGN KEY(passage_pk) REFERENCES passage(passage_pk)
) STRICT;
CREATE TABLE researcher_handoff(
  researcher_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  round_index INTEGER NOT NULL,
  handoff_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE researcher_evidence(
  researcher_id TEXT NOT NULL,
  subnarrative_id TEXT NOT NULL,
  passage_pk INTEGER NOT NULL,
  relevance TEXT NOT NULL CHECK(relevance IN ('relevant','supporting')),
  PRIMARY KEY(researcher_id, subnarrative_id, passage_pk),
  FOREIGN KEY(researcher_id) REFERENCES researcher_handoff(researcher_id),
  FOREIGN KEY(passage_pk) REFERENCES passage(passage_pk)
) STRICT;
CREATE TABLE topic_completion(
  singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
  status TEXT NOT NULL CHECK(status IN ('complete','incomplete')),
  stopping_reason TEXT NOT NULL
) STRICT;
```

Reuse `document_binding`. In v3, redefine `passage` as query-independent source geometry: document foreign key, passage ID, source character/byte offsets, source text SHA-256, scoring text SHA-256, and chunker identity. Move `lane_id`, `query_id`, `cross_encoder_score`, and `cross_encoder_rank` out of `passage`; those are query-dependent and belong only in `query_passage`. Update the existing canonical candidate-passage writer and reader to join through `query_passage`. Do not duplicate passage text.

`passage.passage_id` is globally unique because its canonical identity includes `docid`; identical bytes under different citation docids remain distinct passage handles while the Mixedbread score cache still deduplicates identical `(query text, passage text, scorer identity)` computations. Persist `run_id` with `topic_id` in the database identity and sealed receipt. Persist researcher-discovered facet association in `researcher_facet_update` so a reopened handoff reconstructs its exact `facet_updates`. Exact v3 schema validation compares the declared table SQL as well as PRAGMA-visible columns/indexes/FKs, so removed `CHECK` constraints fail closed.

- [x] **Step 4: Implement transactional admission, handoff validation, and holistic snapshots**

`add_passage_search` validates every passage against exact CAS text and offsets before one transaction commits. `add_researcher_handoff` rejects a different run ID, unknown passage ID, conflicting repeated handoff, or invalid facet. `topic_snapshot` issues one deterministic read transaction and returns every facet, query, candidate, passage, researcher link, and completion row in stable order.

- [x] **Step 5: Add idempotence, conflict, incomplete, and reopen tests**

```python
def test_same_handoff_is_idempotent_but_changed_handoff_conflicts(topic_builder, stored_search) -> None:
    topic_builder.add_passage_search(stored_search)
    handoff = researcher_handoff(
        run_id="r1",
        researcher_id="worker-1",
        passage_id=stored_search.passages[0].passage_id,
    )
    topic_builder.add_researcher_handoff(handoff)
    topic_builder.add_researcher_handoff(handoff)
    with pytest.raises(TopicRecordsIntegrityError, match="handoff.*conflict"):
        topic_builder.add_researcher_handoff(replace(handoff, evidence=()))

def test_incomplete_topic_publishes_and_reopens_with_valid_passages(
    topic_builder, stored_search, document_store
) -> None:
    topic_builder.add_passage_search(stored_search)
    topic_builder.set_completion("incomplete", "budget_exhausted")
    published = topic_builder.publish({"run_id": "r1"})
    with open_published(published, document_store) as records:
        snapshot = records.topic_snapshot()
    assert snapshot.status == "incomplete"
    assert snapshot.stopping_reason == "budget_exhausted"
    assert snapshot.passages[0].passage_id == stored_search.passages[0].passage_id

def test_handoff_rejects_unknown_passage_and_wrong_run(topic_builder) -> None:
    unknown = researcher_handoff(run_id="r1", passage_id="p-" + "0" * 64)
    with pytest.raises(TopicRecordsIntegrityError, match="unknown passage"):
        topic_builder.add_researcher_handoff(unknown)
    wrong_run = researcher_handoff(run_id="another-run", passage_id="p-" + "0" * 64)
    with pytest.raises(TopicRecordsIntegrityError, match="run"):
        topic_builder.add_researcher_handoff(wrong_run)

def test_topic_snapshot_materializes_queries_passages_and_completion(
    topic_builder, stored_search
) -> None:
    topic_builder.add_passage_search(stored_search)
    topic_builder.set_completion("complete", "coverage_sufficient")
    snapshot = topic_builder.topic_snapshot()
    assert [row.query_id for row in snapshot.queries] == [stored_search.query.query_id]
    assert [row.passage_id for row in snapshot.passages] == [
        row.passage_id for row in stored_search.passages
    ]
    assert snapshot.status == "complete"
```

Define `topic_builder`, `stored_search`, `document_store`, `researcher_handoff`, and `open_published` as explicit fixtures/helpers at the top of `test_topic_records.py`; they construct one real temporary SQLite database and `DocumentStore`, not mocks. Assert the exact exception class and completion fields, not only that an exception occurred.

- [x] **Step 6: Run the complete TopicRecords and shared-search suites**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_records.py code/tests/test_topic_passage_search.py -q`

Commit: `feat: persist topic passage research ledger`

---

### Task 4: Move Fixed Facet Retrieval onto the Shared Passage Path

**Files:**
- Modify: `code/trec_rag/facet_pilot_config.py:18-198`
- Create: `configs/rag26_competition_retrieval_v2.yaml`
- Delete: `configs/rag26_competition_retrieval_v1.yaml`
- Modify: `code/trec_rag/facet_retrieval.py:338-1065`
- Modify: `code/trec_rag/competition_retrieval.py:496-837`
- Modify: `code/trec_rag/evidence_store.py` to require the topic run ID when constructing `TopicRecordsBuilder`.
- Modify: `code/tests/test_facet_retrieval.py`
- Create: `code/tests/test_competition_retrieval_v2.py` for v2 config/checkpoint integration coverage; leave the pre-existing dirty `code/tests/test_facet_pipeline_e2e.py` hunks untouched and run that suite as regression coverage only.
- Modify: `code/tests/test_evidence_pipeline_contract.py` to pass explicit fixture run IDs and prove the production candidate path receives `config.run_id`; leave the pre-existing dirty `code/tests/test_retrieval_export.py` hunks untouched and run that suite as regression coverage only.

**Interfaces:**
- Replaces config schema with `facet_pilot_config_v2`.
- Retrieval settings pin `documents_per_query: 1000`.
- Passage settings pin `passages_per_query: 100`, `chunk_max_characters: 3500`, `chunk_overlap_characters: 350`, and the Mixedbread identity.
- Fixed lane execution calls `TopicPassageSearch.search` exactly once per lane.
- Fixed active lanes remain exactly the original narrative plus one semantic subnarrative query per subnarrative. Stored planner BM25 suggestions do not silently become extra organizer calls in this task.
- `FacetRetrievalResult` carries the sealed `PassageSearchResult` for every lane.
- Organizer-facing document order is derived from the same passage scores: each document's best passage score, tie-broken by source rank then docid. No separate document model call or downstream subnarrative rescore is permitted.
- `generate_candidate_artifacts(..., run_id: str)` requires the already established experiment run ID; `_canonical_topic` passes `config.run_id`. No TopicRecords constructor call infers a run from topic ID or supplies a compatibility default.

- [x] **Step 1: Write RED config tests and reject v1**

```python
def test_v2_config_drives_depth_1000_and_top_100_behavior(config_path: Path) -> None:
    retriever = RecordingRetriever(fake_candidates(1000, one_chunk_each=True))
    result = run_v2_config_with_recording_adapters(config_path, retriever)
    assert retriever.requested_hits == [1000]
    assert result.lanes[0].passage_result.returned_documents == 1000
    assert result.lanes[0].passage_result.scored_documents == 1000
    assert len(result.lanes[0].passage_result.passages) == 100

def test_v1_config_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, schema_version="facet_pilot_config_v1")
    with pytest.raises(ValueError, match="facet_pilot_config_v2"):
        load_facet_pilot_config(path)
```

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_facet_retrieval.py code/tests/test_competition_retrieval_v2.py code/tests/test_facet_pipeline_e2e.py -k 'config or passage or rerank or lane' -q`

Expected: assertions expose the old v1 document-rerank fields and scorer calls.

- [x] **Step 3: Replace the lane scorer seam with shared passage search**

For each deterministic retrieval lane, construct:

```python
FocusedQuery(
    query_id=lane.retrieval_query.variant_name,
    text=lane.scoring_query.query_text,
    primary_subnarrative_id=lane.subnarrative_id or "original",
)
```

Call the topic-scoped search once. Project document rows from each `SourceDocument.best_passage_raw_logit`, which Task 1 derived from the complete scored passage set, not only the returned top 100. Store the top-100 passage rows unchanged for evidence selection and `TopicRecordsBuilder.add_passage_search`.

`_retrieve_topic` owns the shared search call and writes a sealed passage-first checkpoint. `_score_topic` becomes a validator/projector over that checkpoint and must not call the organizer retriever or Mixedbread again. A poison retriever/scorer in the score-phase test proves the second phase is cache-free validation, not an accidental replay.

At the canonical evidence boundary, add a required `run_id` keyword to `generate_candidate_artifacts` and pass `config.run_id` from `_canonical_topic`. Update direct offline fixtures to pass an explicit stable test run ID. This repairs the intentionally incompatible TopicRecords v3 constructor at every production call site without adding a fallback/default.

- [x] **Step 4: Delete duplicated fixed-path document/window scoring**

Remove production calls to `_score_document_rows`, `_score_window_rows`, `coverage_aware_long_doc_rank`, and `score_selected_documents`. Retain only any decoding helpers still required for old private artifacts until the v2 writer no longer imports them; do not let them remain reachable from a v2 run.

- [x] **Step 5: Add a poison-old-scorer test and all-1,000 test**

```python
def test_fixed_path_uses_shared_passage_search_and_never_old_document_scorer(
    topic, queries, subnarratives
) -> None:
    shared_search = RecordingTopicPassageSearch(complete_depth_1000_result())
    old = PoisonLaneScorer()
    result = run_facet_retrieval(
        topic,
        queries,
        passage_search=shared_search,
        subnarratives=subnarratives,
        legacy_scorer=old,
    )
    assert len(result.lanes[0].passage_result.passages) == 100

def test_fixed_path_scores_document_at_source_rank_1000(
    topic, queries, subnarratives
) -> None:
    shared_search = RecordingTopicPassageSearch(result_with_rank_1000_winner())
    result = run_facet_retrieval(
        topic,
        queries,
        passage_search=shared_search,
        subnarratives=subnarratives,
    )
    assert "rank-1000-passage" in {p.passage_id for p in result.lanes[0].passage_result.passages}
```

Define `RecordingTopicPassageSearch`, `complete_depth_1000_result`, and `result_with_rank_1000_winner` in `test_facet_retrieval.py` using the exact Task 1 dataclasses. `PoisonLaneScorer.score_lane` raises `AssertionError` so accidental legacy execution is unmistakable; assertions stay on the real fixed-path result rather than on the poison double.

- [x] **Step 6: Run fixed retrieval and checkpoint suites**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_facet_retrieval.py code/tests/test_competition_retrieval_v2.py code/tests/test_facet_pipeline_e2e.py code/tests/test_evidence_pipeline_contract.py code/tests/test_topic_records.py code/tests/test_retrieval_export.py -q`

Commit: `refactor: use shared passage search for fixed retrieval`

---

### Task 5: Move DeepAgent Researchers onto the Same Shared Passage Path

**Files:**
- Modify: `code/trec_rag/deepagent_retrieval.py:742-1314`
- Modify: `code/trec_rag/deepagent_passages.py`
- Modify: `code/tests/test_deepagent_retrieval.py`
- Modify: `code/tests/test_deepagent_passages.py`
- Create: `code/tests/test_retrieval_path_parity.py`

**Interfaces:**
- `DeepAgentRetriever` receives a `passage_search: TopicPassageSearch` adapter; production constructs the same Mixedbread scorer and cache identity as Task 4 after opening the topic worker.
- `retrieve(records: TopicRecordsBuilder, narrative: str) -> AgentRetrievalResult` receives an already topic/run-scoped ledger handle; no researcher or ledger method accepts `topic_id`.
- `search_passages` constructs one `FocusedQuery` from the research task and returns all 100 shared passage rows to the researcher-facing handle layer.
- Researcher evidence is validated against those handles and committed through `add_researcher_handoff`; the coordinator reads `topic_snapshot()`.
- Dynamic `add_facets` behavior remains agentic orchestration and does not enter the shared search module.

- [x] **Step 1: Write RED parity and agentic contract tests**

```python
def test_fixed_and_agentic_adapters_return_identical_passages_for_same_query(
    shared_search, focused_query
) -> None:
    dependencies = SharedFakeDependencies(search=shared_search)
    fixed = run_fixed_adapter(dependencies, focused_query)
    agentic = run_agentic_search_tool(dependencies, focused_query)
    assert [(p.passage_id, p.raw_logit, p.rank) for p in fixed.passages] == [
        (p.passage_id, p.raw_logit, p.rank) for p in agentic.passages
    ]

def test_agentic_search_returns_100_passages_and_scores_rank_1000(
    depth_1000_passage_result
) -> None:
    payload = invoke_search_passages(depth_1000_passage_result)
    assert len(payload["passages"]) == 100
    assert payload["documents_scored"] == 1000
    assert payload["documents_not_scored"] == 0
```

Define `SharedFakeDependencies`, `run_fixed_adapter`, and `run_agentic_search_tool` in `test_retrieval_path_parity.py`; each adapter receives the same injected `TopicPassageSearch` instance and returns its real Task 1 result. Define `depth_1000_passage_result` with 1,000 compact `SourceDocument` rows and 100 `SourcePassage` rows including a winner sourced from document rank 1,000.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_retrieval_path_parity.py code/tests/test_deepagent_passages.py code/tests/test_deepagent_retrieval.py -k 'passage or parity or rank_1000 or topic_records' -q`

Expected: parity fails because DeepAgent still uses `rerank_depth=100` and `top_k=16`.

- [x] **Step 3: Replace `score_document_pool`/`select_diverse_passages` in the tool**

`search_passages` delegates the query to `TopicPassageSearch.search`. Convert each returned `SourcePassage` to the existing citation-handle representation without changing text, docid, offsets, score, or rank. Grouping for handle registration may be by document, but grouping must not change global passage order or impose a cap.

- [x] **Step 4: Bind researcher evidence to the topic ledger**

After the existing evidence validator accepts a researcher's citations, write one `ResearcherHandoff` containing only passage IDs, facet IDs, relevance, researcher ID, round index, and run ID. Read one `TopicEvidenceSnapshot` for coordinator context. Do not rerun Mixedbread or revalidate model relevance in the coordinator.

- [x] **Step 5: Retire the duplicate passage policy**

Remove `pool_hits`, `rerank_depth`, `top_k`, `per_document_cap`, and `min_distinct_documents` from the active DeepAgent configuration. `deepagent_passages.py` may retain pure grouping helpers only if the handle layer still imports them; remove its scorer and selector implementation and their obsolete behavioral tests.

- [x] **Step 6: Preserve bounded incomplete results**

Map shared search failures and `ResearchBudget` exhaustion to the topic completion record. The agent result must return validated searches/evidence accumulated before the stop with `stopping_reason` set to `retrieval_unavailable`, `scoring_failed`, `budget_exhausted`, `hard_deadline`, `no_evidence`, or `evidence_validation_failed`. Production researchers cite `search_passages` rows directly; before completion, every live citation must exist in both the researcher-visible shared passage set and a successfully admitted durable handoff. Do not raise away a sealed incomplete topic; continue raising only integrity conflicts and programming errors.

- [x] **Step 7: Run agentic, evidence, budget, and parity suites**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_retrieval_path_parity.py code/tests/test_deepagent_passages.py code/tests/test_deepagent_retrieval.py code/tests/test_deepagent_evidence.py code/tests/test_deepagent_budget.py code/tests/test_topic_records.py -q`

Commit: `refactor: share passage retrieval with DeepAgent`

---

### Task 6: Topic-First Local Process Dispatcher

**Files:**
- Create: `code/trec_rag/topic_dispatch.py`
- Create: `code/tests/test_topic_dispatch.py`
- Modify: `code/trec_rag/facet_pilot_config.py`
- Modify: `configs/rag26_competition_retrieval_v2.yaml`
- Modify: `code/trec_rag/competition_retrieval.py:1486-1659`
- Create: `code/tests/test_competition_topic_dispatch.py` for runner/dispatcher integration; leave pre-existing dirty `code/tests/test_facet_pipeline_e2e.py` hunks untouched and run that file as regression coverage only.

**Interfaces:**
- Produces `TopicJob(topic_id, run_id, config_path, topic_root)` and path-free `TopicJobReceipt(topic_id, projection_manifest_sha256, status, stopping_reason)`.
- Produces `dispatch_topics(jobs, worker, *, max_workers) -> tuple[TopicJobReceipt, ...]` using local processes when `max_workers > 1` and an inline execution path when it equals one.
- Config adds `execution.topic_workers`, a positive integer; canonical checked-in local default is `2`.
- Worker inputs are serializable identities and paths, never live models, SQLite connections, `TopicRecords`, or validation capabilities.

- [x] **Step 1: Write RED two-topic overlap and source-order tests**

```python
def test_dispatch_runs_two_disjoint_topics_concurrently_and_returns_source_order(tmp_path: Path) -> None:
    receipts, intervals = run_two_blocking_jobs(tmp_path, max_workers=2)
    assert intervals_overlap(intervals["topic-b"], intervals["topic-a"])
    assert [row.topic_id for row in receipts] == ["topic-b", "topic-a"]

def test_dispatch_skips_sealed_topic_before_worker_construction(sealed_topic_job) -> None:
    def poison_worker(job: TopicJob) -> TopicJobReceipt:
        raise AssertionError(f"worker constructed for resumed topic {job.topic_id}")

    receipts = dispatch_topics([sealed_topic_job], poison_worker, max_workers=2)
    assert receipts[0].status == "complete"
```

The `sealed_topic_job` fixture writes a real manifest-last receipt using the same receipt writer read by `dispatch_topics`; do not patch the resume predicate.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_dispatch.py -q`

Expected: collection fails because `trec_rag.topic_dispatch` does not exist.

- [x] **Step 3: Implement serializable job dispatch and fail-closed receipt collection**

Use `ProcessPoolExecutor` for production multi-worker execution. Each process constructs its own retriever, scorer, document-store handle, and topic database under its assigned topic root. Parent code validates every returned sealed receipt from disk and restores official input order. A worker exception cancels no already-completed topic; the run omits the global manifest and reports the failed topic.

- [x] **Step 4: Add deterministic duplicate-publication and incomplete-topic tests**

Assert byte-identical duplicate receipts are accepted; different hashes for the same `(run_id, topic_id)` raise an integrity error. Assert an incomplete sealed topic is a valid receipt and is not rerun unless the operator explicitly starts a new run identity.

- [x] **Step 5: Replace the sequential loop in `_run_official`**

Build pending `TopicJob` values in official source order, call `dispatch_topics`, merge them with already validated resumed receipts, and call `export_retrieval_run` only after the complete selected receipt set exists. Never share the parent `MixedbreadPassageScorer` across workers.

- [x] **Step 6: Run dispatcher and fixed E2E tests**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_dispatch.py code/tests/test_competition_topic_dispatch.py code/tests/test_facet_pipeline_e2e.py code/tests/test_retrieval_export.py -q`

Commit: `feat: dispatch retrieval by topic`

---

### Task 7: Seal Completion State, Document the Contract, and Run Offline E2E

**Files:**
- Modify: `code/trec_rag/retrieval_export.py` only after reconciling its active dirty changes
- Modify: `code/tests/test_retrieval_export.py`
- Modify: `code/trec_rag/README.md`
- Modify: `docs/superpowers/specs/2026-08-04-topic-first-research-retrieval-design.md` only if implementation reveals a verified mismatch

**Interfaces:**
- Private topic projection and generation handoff include `retrieval_status`, `retrieval_stopping_reason`, and the topic snapshot seal.
- Organizer-facing TREC run rows remain document-citation rows and never embed passage text.
- Internal depth 1,000 and top-100 passage limits never become a fixed submission depth. The exporter chooses a narrative-specific `k` from deduplicated documents backed by validated retained evidence and never pads a topic.
- The global manifest records worker count, exact shared passage policy, scorer identity, per-topic status, and ordered topic receipt hashes.

- [x] **Step 1: Write RED complete/incomplete export tests**

```python
def test_incomplete_topic_projection_is_sealed_and_exportable_for_generation(
    incomplete_records, projection_fixture
) -> None:
    receipt = projection_fixture.project(incomplete_records)
    assert receipt.retrieval_status == "incomplete"
    handoff = projection_fixture.read_handoff(receipt)
    assert handoff["retrieval_stopping_reason"] == "budget_exhausted"

def test_global_manifest_is_not_published_when_any_receipt_is_missing(
    config, topics, receipts
) -> None:
    with pytest.raises(ValueError, match="all selected topic"):
        export_retrieval_run(config, topics, receipts[:-1], code_commit="a" * 40)
    assert not global_manifest_path(config).exists()

def test_retrieval_export_uses_variable_evidence_backed_depth_without_padding(
    topic_receipts,
) -> None:
    rows = export_fixture(topic_receipts)
    assert per_topic_depth(rows) == {"topic-a": 7, "topic-b": 23}
    assert all(row.docid in evidence_docids(row.topic_id) for row in rows)
```

`projection_fixture` uses the existing production `build_topic_projection` call with its validated config, topic, retriever identity, and canonical manifest bytes; only the opened `TopicRecords` handle varies. `incomplete_records` is a real sealed v3 database with `topic_completion=('incomplete', 'budget_exhausted')`.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_retrieval_export.py -k 'incomplete or retrieval_status or manifest or variable_depth or padding' -q`

- [x] **Step 3: Project completion fields without duplicating topic data**

Read the completion row through the already-open validated `TopicRecords` handle, include its semantic seal in the projection identity, and emit only compact fields plus existing evidence/citation rows. Do not put raw document bodies, provider responses, or the SQLite file into the handoff archive.

- [x] **Step 4: Update README commands and expected counts**

Document:

```bash
PYTHONPATH=code .venv/bin/python-rocm -m trec_rag.competition_retrieval \
  configs/local/two-topic-passage-v2.yaml --topic rag2026-0 --topic rag2026-1
```

State before execution: 2 topic workers, 1,000 organizer documents per focused query, all returned document passages scored, top 100 passages per query, one database per topic, cache-hit/miss counts, and no hosted generation calls.

- [x] **Step 5: Run the offline two-topic fake E2E and focused regression suite**

Run: `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_topic_passage_search.py code/tests/test_mixedbread_passage_scorer.py code/tests/test_retrieval_path_parity.py code/tests/test_topic_dispatch.py code/tests/test_facet_retrieval.py code/tests/test_facet_pipeline_e2e.py code/tests/test_deepagent_retrieval.py code/tests/test_topic_records.py code/tests/test_retrieval_export.py -q`

Expected: all pass with faked organizer/model adapters; the two topic intervals overlap; fixed and agentic passage rows are identical for the same query and inputs; no network request or model download occurs.

- [x] **Step 6: Run static old-path scans**

Run:

```bash
rg -n 'DEFAULT_RERANK_DEPTH = 100|DEFAULT_TOP_K = 16|select_diverse_passages|score_document_pool|round_robin_subnarrative_coverage' \
  code/trec_rag configs/rag26_competition_retrieval_v2.yaml
```

Expected: no active production call site; a historical comment or explicit legacy decoder must identify itself as non-v2 and be unreachable from the v2 config.

Run:

```bash
rg -n 'docid.*->.*text|dict\[str, str\].*document|documents: dict\[str, str\]' \
  code/trec_rag/topic_passage_search.py code/trec_rag/topic_records.py code/trec_rag/deepagent_retrieval.py
```

Expected: no global or cross-topic document-text map; invocation-local materialization may exist only behind a topic-scoped object and must validate conflicting bodies.

- [x] **Step 7: Commit the sealed handoff and documentation**

Commit: `docs: define shared topic passage retrieval workflow`

---

## Review and Live Gate

- [ ] Generate a task-scoped review package after every task and require both specification compliance and code-quality approval before continuing.
- [x] After Task 7, run one broad Sol reviewer over the complete diff, with special attention to all-1,000 scoring, cache identity, SQLite source binding, process isolation, incomplete-result semantics, and dirty-worktree preservation.
- [x] Address all Critical and Important findings and rerun their covering tests.
- [ ] Perform a secret-free preflight from a clean integration checkpoint; report selected topics, query count, expected organizer cache hits/misses, expected Mixedbread score hits/misses, worker count, and output roots.
- [ ] Ask for explicit authorization before the first live two-topic run. Do not run five topics or the full 2025 set under this implementation plan.

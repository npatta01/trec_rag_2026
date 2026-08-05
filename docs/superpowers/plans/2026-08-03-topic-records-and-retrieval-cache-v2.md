# Topic Records and Retrieval Cache v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated canonical candidate text with one sealed SQLite database per topic and a shared exact-text document store, prove one cached 2025 topic, then replace the active organizer response cache with immutable reference-only request entries.

**Architecture:** `DocumentStore` owns exact UTF-8 bodies by full SHA-256. `TopicRecords` owns topic-aware bindings, offset-only candidates, source reconstruction, semantic seals, and atomic SQLite publication. After that slice passes, `RetrievalCache` owns full-key immutable organizer request directories containing compressed raw authority plus ordered document references.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `hashlib`, `gzip`, `tempfile`, `os`, pytest, existing Pyserini client and facet evidence records.

## Current status (2026-08-03)

- TopicRecords, exact document storage, evidence reconstruction, and sealed
  per-topic retrieval projection are committed at `161cdb5`. The final Sol
  review found no Critical or Important issues; the direct/runner suite passed
  259 tests.
- The organizer RetrievalCache v2 has a post-advisor repair under repeat review.
  The review found three remaining skip paths: base-only projection migration
  bypassed the current eleven-field retriever identity, same-process threads
  could discard one another's live lease, and a cache hit after transport commit
  could strand continuation state. The current patch makes expected retriever
  identity mandatory at the public projection boundary; publishes an
  idempotent, hashed completion marker before cleanup; binds that marker to raw,
  derivation, and ordered-hit identity; rejects every live lease; and reconstructs
  a continuation endpoint from authenticated state without constructing a
  hosted client. The retrieval-cache/export/pipeline/continuation group passes
  308 tests offline. Promotion remains prohibited until repeat Sol review says
  READY. A privacy-safe audit of 165 archived pairs still has zero raw-hash,
  query, legacy-key, or filename-key mismatches across 165,000 candidates.
- The historical facet score promoter is **NOT READY** and will be replaced,
  not patched in place. Independent review proved that its document/window
  imports become directly readable before the manifest-last receipt, rollback
  can delete scores adopted by another source, caller-supplied whitespace-
  equivalent text can invent a raw preimage, and a miss can still reach model
  computation. The 144 MB production ledgers also peak at 763,132 KiB RSS
  before loading the exact-text closure. A three-model advisor panel approved
  one immutable all-22-topic SQLite bundle with separate document/window tables,
  streamed file-backed staging, CAS-only preimage authority, an independently
  expected authorization digest, atomic directory publication, and a read-only
  lookup-only capability whose miss is replay-fatal. The acceptance memory gate
  is at most 256 MiB RSS. The 138-lane authenticated closure dry-run must pass
  before this bundle is published.
- For the immediate five-topic gate, the user authorized abandoning old
  reranker-cache compatibility. The run will use a fresh isolated reranker
  cache and bounded local GPU recomputation. The immutable score bundle is
  deferred and must receive its own completed review before the 22-topic run.
- Live private-cache preflight (no hosted/model calls) verified that the sealed
  full-22 retrieval audits require 138 exact query hashes and all 138 have
  authenticated legacy response/sidecar pairs. Replaying all 138,000 candidate
  rows through the historical scoring normalizer reproduced rank, docid, score,
  and text hash with zero mismatches. Of those rows, 137,495 change when exact
  organizer text is projected to the historical whitespace scoring view, so
  exact source identity and scoring identity must remain separate.
- `ScoringView` now deterministically projects exact source text to
  `trec_rag_whitespace_v1` with source-character and UTF-8-byte coordinates.
  Document/window and sentence scorers retain exact public evidence while
  batching, caching, and model scoring use only the scoring view. The local
  evidence contract passes 52 tests and the downstream canonical/projection
  contract passes 211 tests; no network or model calls were made.
- Across all 165 archived responses, the shared exact-document CAS would hold
  133,425 unique bodies / 1,691,174,566 UTF-8 bytes for 165,000 references,
  with 31,575 duplicate references and no cross-response docid/body conflicts.
  The deterministic archive inventory SHA-256 is
  `2ccad901eb908ac14747bafd2069f7c8aa96ecaaa8cc30457ff47c73ce3bf81f`;
  the reviewed operator-archive epoch is
  `climbmix-400b-operator-archive-facet-2025-v2-sha256-2ccad901eb908ac14747bafd2069f7c8aa96ecaaa8cc30457ff47c73ce3bf81f`.
  It describes this authenticated operator archive cohort, not proof of one
  immutable organizer snapshot. All 165 authenticated pairs may be promoted,
  but only the selected 138 identities are authorized for the 22-topic replay;
  the remaining 27 lane identities are from the earlier five-topic plans and
  collapse to 24 unique surplus transport requests. The other three are
  cross-lane aliases: each surplus lane has the same exact query hash and
  authenticated response SHA-256 as one selected lane (two on topic 31 and one
  on topic 72). Thus `165 = 138 selected lane identities + 24 unique surplus
  transports + 3 byte-identical aliases`; the promotion receipt must preserve
  that classification and authorize only the selected set for replay.
- The existing topic-72 v2 storage smoke already demonstrates the physical
  optimization: `records.sqlite3` is 753,598,464 bytes and its 100-document
  logical exact-text closure is 1,293,981 bytes, versus 6,248,302,686 bytes for
  the old repeated-text candidate JSONL. The combined logical footprint is
  12.08% of the old artifact (5.12 GiB avoided for that topic), below the 20%
  gate. Semantic replay/equivalence is still required before Task 5 is complete.
- Existing v1 canonical checkpoints are retained but are not valid v2
  TopicRecords checkpoints. The migration advisor rejected both direct import
  of the 52 GB candidate streams and preservation of the collapsed-text
  handoff. The approved path is a fresh namespace that reconstructs exact
  organizer source from authenticated responses, derives
  `trec_rag_whitespace_v1` with a deterministic source-coordinate map, rebinds
  only authenticated legacy scores into that effective scoring context,
  regenerates TopicRecords and selections, and issues new canonical requests.
  The old false `trec_rag_raw_v2` label is retained only as declared legacy
  provenance and is never admitted as an effective raw-v2 score context.
  Topic 37 is the representative full gate after a five-topic offline mapping
  audit. The five-topic canonical budget is 32 new DeepSeek requests. No cache
  promotion, GPU scoring, or hosted generation will start before the cache
  review and scoring-view implementation gates close; existing artifacts will
  not be deleted as part of this run.
- The advisor approved byte-identical import of all 22 validated decomposition
  results into the fresh v3 namespace. Planning must not be rerun for this
  replay: the old plans define the authenticated 138-query request set and keep
  planner variation out of the retrieval/scoring comparison. Acceptance
  requires a manifest-last aggregate planning-seed receipt, exact source and
  destination byte hashes, authoritative narrative binding, an ordered plan and
  query digest, equality with the authenticated request set, and a poison
  backend proving zero planning invocations. A copied `result.json` without a
  verified receipt is invalid. The importer now recomputes a canonical source
  inventory digest over each ordered topic ID, source relative path, byte
  length, and source SHA-256; an invented but well-formed provenance digest
  fails closed. The initial v1 fixture passed 12 focused and 142 integration
  tests, but a subsequent production-shape probe found that it authenticated
  the wrong query layer: the 22 real
  decompositions contain 283 generated BM25 `QueryVariant` rows, while the
  organizer run uses exactly 138 retrieval lanes (22 exact originals plus 116
  full-subnarrative-text lanes). All 138 lane identities match authenticated
  archive entries with no missing lanes. Sol approved a v2 schema and shared
  pure lane projector; v1 receipts are now rejected as semantically obsolete.
  The v2 implementation passes 22 focused tests and 164 current
  planning/export/E2E tests. A temporary real-source probe imported and
  reverified all 22 decompositions with the planner poisoned, exactly 283
  planner variants, exactly 138 unique retrieval lanes, and zero planning
  calls; its ephemeral receipt SHA-256 was
  `d12544e4b42f062e007c23e4f83c54b3350826cb84eb35524636a8a5a2ba20a2`.
  It seals all three identities separately: 283 planner variants, 138 projected
  retrieval lanes, and the canonical plan payload. A post-implementation Sol
  review and run-manifest request-key binding remain open. The measured
  exact-source inventory SHA-256 is
  `b90cd86646f8b3bd3eae6a7dc75c83462184f39edad787e4ea60fa7c2b352897`;
  the rendered planner-variant-record digest is
  `a597a9a34862776f6b2cc26b5b759183d475b03895692a429b10c83ff3fe6c78`;
  the ordered retrieval-lane-record digest is
  `baae2c0e050c0113cd572b52beee20697dcf8075cefc7db23ed55f082d9522e7`,
  and the sorted 138-query-set digest is
  `a782aef60fc08cb19b1eb649707b7b942fb757dd0ac7f5c6b626a971d2fb884f`.
- The parallel generation track supplied reviewed aggregate checkpoint
  `a3fe571439185d1570e3f91d980076ed096e772a`. It contains topic-aware document
  binding, full generation/resume prompt identity, the selected-evidence
  controller, and resumed-projection verification. The final clean run commit
  must combine that history with this retrieval branch; do not copy the source
  worktree's unrelated untracked research note.

Tasks 1–4 below are retained as historical completed work and are not reopened
by this revision. Task 5 remains the representative gate; Task 5.5 is a
blocking implementation gate before RetrievalCache. Checked RED/GREEN steps
describe what happened at the time, not commands expected to fail at current
HEAD. Completion checkpoints are Task 1 `efc35ba`, Task 2 `fb4df6b`, Task 3
`dd993b6`, and Task 4 `863ac36`.

## Global Constraints

- Keep exact transport bytes, exact evidence UTF-8 bytes, and scoring-normalized text as three distinct identities.
- `DocumentStore` never normalizes whitespace or Unicode and has no `docid` lookup.
- Use one `records.sqlite3` per topic; no global SQLite database and no absolute content-store paths in database rows or seals.
- Resolve documents only through `(topic_id, docid) -> content_sha256 -> exact body`; conflicting content for one topic/docid fails.
- Never silently choose between conflicting same-topic docid bodies; keep the
  precise conservative policy as an advisor/test gate if source equivalence is
  not proven.
- Treat `ValidatedTopicRecords` as a non-serializable current-process
  capability bound to exact database/manifest bytes and the complete CAS
  closure. Never trust a disk validation marker after restart or rebind.
- Preserve unchanged organizer-facing TREC and `retrieval_with_text.jsonl.zip` semantics.
- Semantic equality uses canonical ordered row streams with `float.hex()`; SQLite byte equality across versions is not required.
- Publish document objects, topic databases, request cache entries, and completion manifests atomically without replacing contradictory state.
- Retain losslessly compressed exact organizer response bytes for the lifetime of retrieval-cache v2.
- Existing generated artifacts remain untouched; there is no old-checkpoint compatibility and no cleanup in this plan.
- Do not edit the user-owned dirty `code/trec_rag/competition_rag.py`, `code/tests/test_competition_rag.py`, non-agentic report, or non-agentic plan.
- Run live preflights and experiments from a clean detached worktree at the
  reviewed commit, with submodules initialized and the ignored environment
  copied from the shared checkout. The current worktree intentionally retains
  user-owned tracked generation changes and therefore cannot satisfy the
  runner's clean-worktree guard without altering their work.
- Do not start a broad live retrieval or generation run.

---

### Task 1: Exact UTF-8 `DocumentStore`

**Files:**
- Create: `code/trec_rag/document_store.py`
- Create: `code/tests/test_document_store.py`

**Interfaces:**
- Produces: `DOCUMENT_STORE_SCHEMA_VERSION = "document-store-v1"`.
- Produces: `DocumentStoreIntegrityError(RuntimeError)`.
- Produces: `DocumentReceipt(content_sha256: str, byte_count: int, character_count: int)`.
- Produces: `DocumentStore(root: Path).admit_text(text, expected_sha256=None)`, `.read_text(digest)`, and `.verify(digest)`.
- Physical layout is private: `<root>/sha256/<digest[:2]>/<digest>.utf8`.

- [x] **Step 1: Write failing exact-byte and topic-independent object tests**

```python
def test_document_store_preserves_exact_unicode_and_whitespace(tmp_path: Path) -> None:
    text = "Café\tline  one\nΩmega\n"
    store = DocumentStore(tmp_path / "objects")
    receipt = store.admit_text(text)
    assert receipt.content_sha256 == sha256(text.encode("utf-8")).hexdigest()
    assert receipt.byte_count == len(text.encode("utf-8"))
    assert receipt.character_count == len(text)
    assert store.read_text(receipt.content_sha256) == text

def test_document_store_rejects_expected_digest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(DocumentStoreIntegrityError, match="expected"):
        DocumentStore(tmp_path).admit_text("body", expected_sha256="0" * 64)
```

- [x] **Step 2: Run RED tests**

Run: `.venv/bin/python -m pytest code/tests/test_document_store.py -q`

Expected: collection fails because `trec_rag.document_store` does not exist.

- [x] **Step 3: Implement create-only atomic admission and verified reads**

Use a sibling `mkstemp`, write/flush/fsync exact UTF-8 bytes, then publish with
`os.link(temp, destination)` so an existing object is never replaced. On
`FileExistsError`, verify the existing object and return the same receipt.
Always unlink the temporary file. Validate lowercase 64-hex digests and reject
invalid UTF-8/corrupt content.

- [x] **Step 4: Add corruption and concurrent-idempotence tests**

```python
def test_document_store_rejects_corrupt_existing_object(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    receipt = store.admit_text("original")
    next(tmp_path.rglob("*.utf8")).write_bytes(b"changed")
    with pytest.raises(DocumentStoreIntegrityError, match="digest"):
        store.verify(receipt.content_sha256)

def test_document_store_identical_admission_is_idempotent(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    assert store.admit_text("same") == store.admit_text("same")
    assert len(list(tmp_path.rglob("*.utf8"))) == 1
```

- [x] **Step 5: Run GREEN tests and commit**

Run: `.venv/bin/python -m pytest code/tests/test_document_store.py -q`

Commit: `feat: add exact content-addressed document store`

---

### Task 2: Sealed per-topic candidate records

**Files:**
- Create: `code/trec_rag/topic_records.py`
- Create: `code/tests/test_topic_records.py`
- Use: `code/trec_rag/facet_evidence.py`

**Interfaces:**
- Consumes: `DocumentStore`, `ExtractiveCandidate`, `SelectionCandidate`, and `SubnarrativeContext`.
- Produces: `TOPIC_RECORDS_SCHEMA_VERSION = "topic-records-v1"` and `CANDIDATE_STAGE = "canonical-candidates-v1"`.
- Produces: `TopicRecordsIntegrityError(RuntimeError)`.
- Produces: `TopicRecordsReceipt(database_sha256, database_bytes, semantic_sha256, topic_id, document_sha256s, row_counts)`.
- Produces: `TopicRecordsBuilder(destination, topic_id, document_store)` with `bind_document`, `add_candidate`, and `publish(identity)`.
- Produces: `TopicRecords.open(database, manifest, expected_topic_id, document_store)`, `selection_pool`, `load_candidates`, and `validate_all_sources`.

- [x] **Step 1: Write failing topic binding tests**

Create two stores with the same `docid` under different topic IDs and different
bodies; both must resolve through their own binding. Attempting to bind a second
body to the same `(topic_id, docid)` must raise `TopicRecordsIntegrityError`.

- [x] **Step 2: Run RED binding tests**

Run: `.venv/bin/python -m pytest code/tests/test_topic_records.py -q`

Expected: collection fails because `trec_rag.topic_records` does not exist.

- [x] **Step 3: Implement strict schema and exact document bindings**

This historical step implemented the committed `topic-records-v1` schema, not
the v2 surrogate/passage-dictionary schema now specified in the design. It uses
`PRAGMA foreign_keys=ON`, `journal_mode=DELETE`, `synchronous=FULL`,
`temp_store=FILE`, and explicit transactions. `bind_document` admits the exact
text first, then inserts `document_binding` with a `(topic_id, docid)` primary
key and rejects conflicts. Task 5.5 owns the explicit v1-to-v2 production
boundary; only the isolated prototype continues to read v1.

- [x] **Step 4: Write failing candidate reconstruction tests**

Use `extract_document_candidates` on a source containing tabs, newlines, Café,
and Ωmega. Persist every returned candidate. Reopen the database from another
`DocumentStore` root containing a copied declared closure and assert exact
equality for candidate text, sentence spans, paragraph, neighboring contexts,
passage source text, character offsets, byte offsets, hashes, and scores.

- [x] **Step 5: Implement offset-only candidate tables and reconstruction**

Do not store candidate, sentence, paragraph, context, passage, or document text
in SQLite. Candidate start/end is the first/last evidence sentence range.
`candidate_span.role` is one of `evidence`, `matched_paragraph`,
`context_before`, or `context_after`. Reconstruct text by slicing the exact body
and verify UTF-8 byte offsets and hashes before returning domain records.

- [x] **Step 6: Write failing deterministic selection and semantic-seal tests**

Build equivalent databases in two temporary roots with candidates inserted in
opposite order. Assert equal semantic SHA-256 and equal `SelectionCandidate`
ordering for a fixed context/limit, while allowing database byte hashes to
differ. Change one score or offset and assert a different semantic seal or a
source-validation error.

- [x] **Step 7: Implement semantic hashing and selection query**

Hash every scoped table in explicit primary-key order. Encode floats with
`float.hex()`, nullable values with explicit tags, and all other values with
length prefixes. Exact groups use `(candidate_kind, text_sha256)`; rank each
group by score descending then candidate ID ascending; reconstruct and verify
only chosen groups before returning the precluster pool.

- [x] **Step 8: Write failing publication/integrity tests**

Cover database corruption, manifest corruption, wrong expected topic, missing
content object, bad foreign key, stale `-wal`/`-shm`, partial database without a
manifest, identical republish, and contradictory republish. Assert no complete
manifest names partial state.

- [x] **Step 9: Implement sealing and atomic publication**

Before publication run `foreign_key_check`, `integrity_check`, full source
validation, and semantic-seal recomputation. Close SQLite, reject WAL/SHM, hash
the final database, publish it without replacement, then atomically write the
records manifest. Idempotent publication requires equal semantic seal and
topic; contradictory publication raises without replacing the winner.

- [x] **Step 10: Run GREEN tests and commit**

Run: `.venv/bin/python -m pytest code/tests/test_topic_records.py code/tests/test_document_store.py -q`

Commit: `feat: add sealed per-topic candidate records`

---

### Task 3: Replace production candidate JSONL with `records.sqlite3`

**Files:**
- Modify: `code/trec_rag/evidence_store.py`
- Modify: `code/tests/test_evidence_pipeline_contract.py`
- Modify: `code/tests/test_canonical_nugget_contract.py`

**Interfaces:**
- Consumes: Task 2 `TopicRecordsBuilder` and `TopicRecords`.
- Changes: `CandidateArtifacts` fields become `records_path`, `manifest_path`, and `document_store_root`.
- Changes: `generate_candidate_artifacts(..., document_store_root: Path)` writes `<topic>/records.sqlite3` plus `canonical/records-manifest.json`.
- Changes: `select_evidence_artifacts` gets its deterministic pool directly from `TopicRecords`; `_CandidateSpill` is deleted.
- Changes: `load_validated_candidate_artifacts` opens topic records and reconstructs only requested candidates after validating all source relations.

- [x] **Step 1: Replace artifact contract expectations in tests and run RED**

Update focused tests to require no production `canonical/candidates.jsonl`, a
single topic database, offset reconstruction, empty fallback tables, stable
semantic seals, and semantic-tamper rejection. Preserve low-level
`write_candidate_jsonl` golden tests only as extraction serializer tests; they
must no longer represent the production checkpoint.

Run: `.venv/bin/python -m pytest code/tests/test_evidence_pipeline_contract.py code/tests/test_canonical_nugget_contract.py -q`

Expected: failures name the old `CandidateArtifacts` and candidate-file fields.

- [x] **Step 2: Route generation through `TopicRecordsBuilder`**

Audit requests as before. For each request, bind its exact source, extract
candidates with the existing scorer, and add records. Publish only after the
second-pass request digest matches. Store scorer/normalizer/source identities in
the candidate stage seal and records manifest.

- [x] **Step 3: Replace temporary spill and JSONL loader**

`select_evidence_artifacts` opens the records artifact with the expected topic
from contexts, obtains candidate/exact-group counts and deterministic pools,
then calls unchanged `select_subnarrative_candidates`. Selection manifests use:

```json
{
  "records_file": "records.sqlite3",
  "records_manifest_file": "records-manifest.json",
  "records_database_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "candidate_semantic_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

`load_validated_candidate_artifacts` verifies manifest/database/semantic/source
integrity before returning required records.

- [x] **Step 4: Remove production JSONL coupling and run GREEN**

Run: `.venv/bin/python -m pytest code/tests/test_document_store.py code/tests/test_topic_records.py code/tests/test_evidence_pipeline_contract.py code/tests/test_canonical_nugget_contract.py -q`

Commit: `refactor: store canonical candidates by topic`

---

### Task 4: Checkpoint and organizer-export integration

**Files:**
- Modify: `code/trec_rag/competition_retrieval.py`
- Modify: `code/trec_rag/retrieval_export.py`
- Modify: `code/trec_rag/competition_debug_report.py` only if focused tests prove it names old artifacts.
- Modify: `code/tests/test_retrieval_export.py`
- Modify: `code/tests/test_facet_pipeline_e2e.py`
- Modify: `code/tests/test_competition_debug_report.py` only if production behavior changed.

**Interfaces:**
- Produces: `document_store_dir(root_dir) -> repo_cache_root(root_dir) / "documents" / "v1"`.
- Canonical checkpoint artifacts become `records.sqlite3` and `canonical/records-manifest.json` plus existing handoff/selection/nugget files.
- Retrieval export derives the shared store root from `FacetPilotConfig.root_dir`, opens records with the expected topic ID, and validates selected evidence before unchanged organizer projection.

- [x] **Step 1: Update checkpoint/export fixtures and run RED**

Replace old candidate JSONL fixture construction with `TopicRecordsBuilder`.
Assert a wrong-topic database, missing CAS object, modified database, changed
semantic seal, and WAL/SHM are rejected. Retain literal expected official TREC
and ZIP-member bytes.

Run: `.venv/bin/python -m pytest code/tests/test_retrieval_export.py code/tests/test_facet_pipeline_e2e.py -q`

- [x] **Step 2: Update canonical artifact lists and dependency injection**

Pass `document_store_dir(config.root_dir)` into candidate generation. Change
canonical completion receipts and `_CANONICAL_ARTIFACTS` to the versioned topic
database/manifest contract. Do not touch RAG generation files.

- [x] **Step 3: Update retrieval export validation**

Validate selection-manifest records fields, open the database with
`expected_topic_id=topic.id`, reconstruct selected evidence from the shared
store, then build the existing `EvidenceBundle` organizer projections. Database
or CAS paths must not appear in official outputs.

- [x] **Step 4: Run focused and contract suites**

Run: `.venv/bin/python -m pytest code/tests/test_retrieval_export.py code/tests/test_facet_pipeline_e2e.py code/tests/test_competition_debug_report.py -q`

Commit: `refactor: export retrieval from topic records`

---

### Task 5: Representative cached 2025 topic gate

**Files:**
- Create ignored: `configs/local/topic-records-2025-smoke-v1.yaml`
- Create ignored: `outputs/topic-records-2025-smoke-v1/...`
- Update tracked evidence only: this plan's SDD report/ledger, not a private output artifact.

**Interfaces:**
- Consumes existing organizer retrieval, reranker, and canonical response caches.
- Produces one new topic namespace and a compact receipt containing counts,
  semantic seal, database bytes, old candidate bytes, ratio, cache hits/misses,
  hosted calls, and organizer output hashes. Do not copy private content into
  tracked files.

- [x] **Step 1: Preflight a numeric 2025 topic with complete reusable caches**

Select the smallest completed representative among topics 14, 31, 37, 58, and
72 by reading manifests only. Copy its prior config into `configs/local/` with a
new experiment ID/output directory. Confirm required secrets are present without
printing them, but require zero expected organizer/model cache misses.

Evidence: topic 72 was the smallest completed candidate artifact. Its seven
organizer retrieval queries and five canonical hosted responses are complete
cache hits; candidate sentence scoring replayed with zero model-loader calls.

- [ ] **Step 2: Run the new topic pipeline cache-only with miss-raising adapters**

Use the repository Python entrypoint and a one-topic selector. Inject adapters
that raise before organizer, hosted-model, or local-model execution on a cache
miss, because the explicit RetrievalCache `offline=True` interface is not built
until Task 6. Record wall time and peak disk use.

- [ ] **Step 3: Compare semantic outputs**

Compare selected subnarrative records, canonical nuggets, TREC rows, and
`retrieval_with_text.jsonl.zip` member content against the prior completed topic
after excluding intentionally versioned checkpoint metadata. Run records/CAS
integrity and copied-closure validation.

- [ ] **Step 4: Enforce the storage gate**

Require the new `records.sqlite3` plus its topic's unique CAS closure to be less
than 20% of the old `canonical/candidates.jsonl` size. “Closure” means the
logical bytes of every distinct referenced object, even when an object already
existed in the shared store. Report that logical closure, marginal CAS bytes
newly admitted by this run, and total shared-store bytes separately. If
equivalence or the storage gate fails, stop before Task 6 and diagnose
test-first.

Commit: no generated/private artifacts.

---

### Task 5.5: Blocking validation capability and per-document projection

Task 5.5 is inserted by the Sol advisor decision and must finish before
RetrievalCache. It is implementation work, not a documentation claim: do not
record code as complete until its tests and the representative gate pass.

**Files:**
- Modify: `code/trec_rag/topic_records.py`
- Modify: `code/trec_rag/facet_evidence.py`
- Modify: `code/trec_rag/evidence_store.py`
- Modify: `code/trec_rag/competition_retrieval.py`
- Modify: `code/trec_rag/retrieval_export.py`
- Modify: `code/tests/test_topic_records.py`
- Modify: `code/tests/test_evidence_pipeline_contract.py`
- Modify: `code/tests/test_retrieval_export.py`
- Modify: `code/tests/test_facet_pipeline_e2e.py`

**Interfaces:**
- Bumps: `TOPIC_RECORDS_SCHEMA_VERSION = "topic-records-v2"`.
- Produces: `ValidatedTopicRecords`, a non-serializable current-process proof.
- Changes: `TopicRecordsBuilder.publish(...)` returns a receipt plus validation
  session; `TopicRecords.open(..., validation_session=None)` deep-validates
  without one and exact-byte/CAS-rebinds with one.
- Carries: the session through candidate artifacts, selection, and sealed
  per-topic projection inside one topic worker. Only a serializable projection
  receipt reaches the parent/global exporter.

- [x] **Step 0: Capture the document-geometry prototype decision**

On a throwaway branch, benchmark a read-only per-document geometry index against
the sealed topic-72 v1 database. Require exactly one geometry build per bound
document, no private text in output, and an honest inventory of invariants that
do and do not match production validation. Record the branch commit and verdict
here; do not merge the prototype shell into production.

Evidence: isolated branch `prototype/topic-records-geometry-v1`, commit
`08f7d2c`. Production validation took 312.7100 s; indexed reconstruction took
13.7608 s (22.72x) and document-major validation took 11.8185 s (26.46x).
Both indexed paths built geometry exactly 100 times for 100 bound documents,
visited all 61,385 candidates / 128,395 spans / 1,109,800 passage links, and
reported zero mismatches. Peak process RSS was 1,217,540 KiB. The prototype is
the only v1-reader boundary and is not merged into production.

- [ ] **Step 1: Write failing tests for bounded derivation and capability binding**

Assert repeated candidates in one document derive byte offsets, scoring text,
paragraph maps, and sentence maps `O(documents)`, not `O(candidates)`. Assert
`ValidatedTopicRecords` is non-serializable, binds exact database bytes,
manifest bytes, validator version, and CAS closure, rejects changes to any of
them, and cannot be replaced by a serialized `source_validation` marker.

Required RED names include
`test_validate_sources_derives_geometry_once_per_bound_document`,
`test_shared_passage_is_validated_once_and_candidates_keep_compact_passage_links`,
`test_add_candidate_rejects_validly_formatted_forged_candidate_nugget_id`,
`test_validated_topic_records_is_non_pickleable_and_process_local`, and exact
database/manifest/CAS rebind-tamper tests. Count wrapper invocations across at
least two documents; do not rely on the existing `lru_cache(maxsize=1)` or test
only a malformed candidate-ID regex.

- [ ] **Step 2: Run RED tests**

Run: `.venv/bin/python -m pytest code/tests/test_topic_records.py -q`

Expected: the new derivation-count and current-process capability tests fail.

- [ ] **Step 3: Implement bounded validation and safe rebind**

Make the first validator direct and document-indexed. Extend/reuse the existing
per-document `_SourceValidationCache`, including
paragraph index maps and memoized sentence derivation. Stream database hashing.
Validate each unique passage once, then stream compact links, spans, and
candidates without constructing rich candidate objects solely for validation.
Recompute candidate nugget IDs from their natural identities. Close the
publish-time reopened connection. Add the explicit validation-session object
and pass it through selection and per-topic projection; same-session rebind
must use the capability rather than rerun deep reconstruction. Do not make a
disk marker an authorization to skip validation, and do not serialize the
session into the global exporter.

- [ ] **Step 4: Implement the v2 passage dictionary and compact links**

Move each complete passage provenance tuple, including score and rank, into a
`passage` dictionary keyed physically by a compact surrogate and uniquely by
document plus passage ID. Retain only candidate, document, ordinal, and passage
surrogates in `candidate_passage_link`. Enforce same-document ownership through
composite foreign keys on both link edges. Batch per-document builder writes,
project semantic seals through stable natural IDs rather than surrogate values,
and defer span normalization.

Add adversarial tests that attempt a cross-document link, duplicate
`(document, passage_id)`, changed surrogate allocation, and changed natural
identity. The first two must fail at the relational/publication boundary;
surrogate allocation must not change the semantic seal, while natural identity
must.

- [ ] **Step 5: Add the per-topic projection handoff test**

Build a validated topic and, inside the same process, hand its open records
handle/session to selection and `build_topic_projection`. Seal the per-topic
projection as `canonical/retrieval-projection.json` with
`canonical/retrieval-projection-manifest.json` published and receipted last,
then return only its serializable receipt. The global exporter must
accept those receipts, verify projection bytes, and concatenate without
reopening records. Assert exact selected evidence, downstream canonical nugget,
TREC, and full-text ZIP outputs. Rebinding with altered database, manifest, or
CAS bytes must fail closed; a race winner with different database bytes must be
validated rather than inheriting the loser's proof. A resumed topic must deep
open once to mint a fresh process-local session.

- [ ] **Step 6: Run GREEN tests and the representative gate**

Run focused topic-record tests plus the cache-only topic-72 gate. Gates are:
untrusted open <=30 seconds (target <=15), same-session rebind <=2 seconds,
exact selection/canonical/organizer outputs, and preserved storage reduction.
Record timings and output hashes in the SDD evidence only; do not claim code
exists here until verification supplies evidence.

---

### Task 6: Immutable normalized organizer request cache

**Files:**
- Create: `code/trec_rag/retrieval_cache.py`
- Create: `code/tests/test_retrieval_cache.py`
- Use: `code/trec_rag/document_store.py`
- Use: `code/trec_rag/det_sparse_ledger.py` semantics; do not duplicate its weaker text-bearing candidate format.

**Interfaces:**
- Produces: `RETRIEVAL_CACHE_SCHEMA_VERSION = "organizer-retrieval-cache-v2"`.
- Produces: `RETRIEVAL_TEXT_NORMALIZER_VERSION = "remote-extract-text-v1"`.
- Produces: `TransportIdentity` with `canonical_dict()` and full `request_key`.
- Produces: `DerivationIdentity` with full `derivation_key` beneath a transport entry.
- Produces: `CachedHit(rank, score, docid, content_sha256)` and `CachedRetrieval`.
- Produces: `RetrievalCache(root, document_store, normalizer).lookup(transport,
  derivation, ..., offline=False)` and `.commit(transport, derivation, ..., raw)`.

- [ ] **Step 1: Write failing transport/derivation, fail-closed, and exact-raw tests**

Use literal organizer-shaped responses. Assert remote-result fields change the
64-hex transport key, while topic/variant/retriever labels and local normalizer
versions do not. A normalizer bump rederives from byte-identical raw transport
with `external_calls == 0`. Assert `corpus_epoch` changes mint new keys, raw
gzip decompresses to byte-identical transport output, hits contain no text,
missing score/rank/docid fails closed, and offline misses fail before any
client method is called. Assert the exact transport manifest binds request key,
canonical identity, query hash, raw and gzip hashes/lengths; assert the exact
derivation manifest binds its parent request/raw hash, canonical derivation
identity, hit receipt/count/digest, and document closure. Altered or mismatched
parent/raw/derivation combinations must fail.

- [ ] **Step 2: Run RED cache tests**

Run: `.venv/bin/python -m pytest code/tests/test_retrieval_cache.py -q`

- [ ] **Step 3: Implement split identity, normalization, and validation**

Use an injected versioned normalizer with explicit field selection and no
coercing defaults. Admit exact UTF-8 bytes from one versioned evidence field;
never collapse whitespace, normalize Unicode, or concatenate arbitrary mapping
values. Keep scoring normalization as a separate identity. Store scores
semantically with `float.hex()` in the digest while retaining JSON numeric
values for consumers. On lookup, decompress/hash raw, reparse it, regenerate
the selected derivation, compare its semantic digest, and verify each document
object. Add `offline=True`; a miss must fail before network access.

- [ ] **Step 4: Write failing publication race/crash tests**

Two processes or threads committing byte-identical raw authority must converge
on one complete transport entry. Byte-different same-epoch raw responses must
conflict even when their normalized semantic hits are equal. Under an identical
raw parent, different derived semantic hits must raise and leave the winner
unchanged plus a private conflict entry. Inject failure before raw,
transport manifest, derived hits, derivation manifest, fsync, and each final
link; lookup must never treat partial state as complete. Two identical commits
must produce byte-identical gzip, and a new derivation must not rewrite the
transport manifest.

- [ ] **Step 5: Implement immutable directory publication**

Build under unique sibling directories, write and fsync the raw authority,
publish `raw.body.gz` with create-only `os.link`, and verify an existing race
winner byte-for-byte. Publish `transport-manifest.json` with a create-only link
last as its completion marker. Publish each immutable derivation independently with
`derivation-manifest.json` linked last, allowing local rederivation without
rewriting transport state. A derivation race loser must verify the identical
parent raw receipt plus exact hits receipt and semantic digest. Validate an
existing winner after races; an empty
pre-existing entry must not be adopted. Use
`<root>/conflicts/<request_key>/...` only for same-epoch contradictory completed
state; a changed `corpus_epoch` has a new transport key.

- [ ] **Step 6: Run GREEN tests and commit**

Run: `.venv/bin/python -m pytest code/tests/test_retrieval_cache.py code/tests/test_document_store.py -q`

Commit: `feat: add immutable organizer retrieval cache`

---

### Task 7: Migrate active retriever and cache bypasses to v2

**Files:**
- Modify: `code/trec_rag/retrievers.py`
- Modify: `code/trec_rag/remote_client.py` only to expose a stable normalizer adapter/version if required.
- Modify: `code/trec_rag/facet_retrieval.py`
- Modify: `code/trec_rag/continuation.py`
- Modify: `code/trec_rag/rerank_score_cache.py`
- Modify: `code/trec_rag/rerank_cache_promotion.py` if it derives old filenames.
- Modify: `code/tests/test_pipeline.py`
- Modify: `code/tests/test_remote_pyserini.py`
- Create: `docs/superpowers/retrieval-cache-v2-reader-inventory.md`
- Modify: focused rerank-cache tests discovered by `rg`.

**Interfaces:**
- `PyseriniRemoteRetriever` retains `retrieve(QueryVariant) -> list[RetrievedCandidate]` for scoring compatibility.
- Constructor accepts an injected `RetrievalCache`; production builders use
  `cache/retrieval/pyserini_remote/v2` and shared `cache/documents/v1`.
- `request_cache_key` returns the full transport v2 key; derivation uses the
  nested derivation key.
- Direct cache consumers use `RetrievalCache.lookup`. Enumerate every reader
  found by `rg` (including deep-facet candidate runners, all-topic retrieval,
  dev inputs, local MiniLM helpers, and config helpers) and either migrate it
  or explicitly scope it as a legacy reader. No silently inferred fallback is
  allowed, and the no-backward-compatibility stance remains.

- [ ] **Step 1: Update retriever tests to require v2 behavior and run RED**

Require split transport/derivation keys, corpus-epoch separation, one external
call on miss, zero on hit or local rederivation, no text in hits, durable exact
raw bytes, corrupt/partial/conflicting entry rejection, strict organizer field
parsing, explicit offline behavior, continuation semantics unchanged, and
`cache=False` bypass unchanged.

Run: `.venv/bin/python -m pytest code/tests/test_pipeline.py code/tests/test_remote_pyserini.py -q`

- [ ] **Step 2: Replace old cache read/write inside `retrieve`**

Perform v2 lookup before reserving an external call. Keep the existing explicit
continuation and transport-attempt ledger. After a successful raw response,
commit through `RetrievalCache`; materialize `RetrievedCandidate` values from
the returned references and document bodies.

- [ ] **Step 3: Remove active direct-reader bypasses**

Change `rerank_score_cache._load_cached_candidates` and promotion expectation
code to construct the same identity and call `RetrievalCache.lookup`. Add
`retrieval_text_normalizer_version` to the topic-records stage identity so a
normalizer bump is explicit. For same-topic docid/body disagreement, do not
silently pick text: fail conservatively unless an advisor-approved policy and
test prove equivalence. Do not silently fall back to legacy response files;
leave independently sealed all-topic sparse-ledger archives versioned and
unchanged, while listing their readers as explicit scope. Check in a bounded
reader inventory naming every old-layout read/write site, its migrated or
explicitly-legacy disposition, and an owner. Add a static guard test with a
small reviewed legacy allowlist so a new active fallback cannot silently
reintroduce v1 filenames.

- [ ] **Step 4: Namespace continuation and shared score-cache writes**

Namespace continuation tickets, ledgers, and in-progress markers per topic
shard. Give the marker an owner and expiring lease so a crashed process is
recoverable; keep only the cross-process rate limiter global. Add offline mode
so cache-only replay never reaches continuation/network code. Replace unsafe
shared JSONL appends with per-process parts (or an equivalent lock), and test
that concurrent reranker writers remain readable before process-per-topic
execution.

- [ ] **Step 5: Run focused retrieval/cache suites**

Run: `.venv/bin/python -m pytest code/tests/test_pipeline.py code/tests/test_remote_pyserini.py code/tests/test_facet_retrieval.py -q`

Commit: `refactor: use retrieval cache v2 in active pipeline`

---

### Task 7.25: Authenticated zero-call planning seed

**Files:**
- Create: a focused planning-seed module under `code/trec_rag/`.
- Create: its focused test module under `code/tests/`.
- Modify: `code/trec_rag/competition_retrieval.py` only at the preflight and
  run-manifest binding seams.

**Interfaces:**
- Imports the exact existing `decomposition/result.json` bytes create-only;
  never parses and reserializes the destination artifact.
- Validates each source through the reviewed production decomposition loader
  against the official topic ID and narrative bytes.
- Publishes one per-topic seed manifest last and one aggregate migration receipt
  last. The fresh v3 runner accepts a preexisting decomposition only when that
  receipt verifies and is bound into the run-level manifest.
- Records source run/config/commit provenance as declared provenance unless an
  independently authenticated hosted receipt exists; the result bytes alone do
  not prove the historical provider response.

- [ ] **Step 1: Write failing strict-validation and create-only tests**

Cover wrong topic/narrative, changed source or destination bytes, schema or
query-rendering incompatibility, rewritten destination JSON, missing per-topic
manifest, partial aggregate receipt, contradictory republish, and an existing
unreceipted `result.json`. Every case must fail closed without invoking a
planner.

- [ ] **Step 2: Implement per-topic seed receipts**

For each of exactly 22 topics record official narrative SHA-256, source and
destination relative paths/byte lengths/SHA-256 values, equality of source and
destination hashes, decomposition schema, fallback status, canonical plan
payload digest, and ordered query-record digest over variant, source, query
SHA-256, subnarrative ID, and lane order.

- [ ] **Step 3: Implement and verify the aggregate receipt**

Record the ordered-plan digest, unique query count `138` and digest, exact set
equality with authenticated required retrieval requests, validator code/module
identity, aggregate per-topic-manifest digest, and explicit counters
`planning_backend_invocations: 0` and `hosted_planning_calls: 0`. Bind the
aggregate receipt SHA-256 into the v3 run manifest.

- [ ] **Step 4: Prove zero-call loading**

Load all 22 seeded decompositions with a planning backend that raises on any
invocation. Require zero calls, exact 138-query equality, and byte equality for
all source/destination pairs. Any validation incompatibility stops the replay;
never rewrite an old plan merely to make it pass.

Commit: `refactor: authenticate zero-call planning seed`

---

### Task 7.5: Topic-first local process executor

**Files:**
- Modify: `code/trec_rag/competition_retrieval.py`
- Modify: the competition retrieval config schema/loader discovered by `rg`.
- Modify: `configs/rag26_competition_retrieval_v1.yaml` only after preserving
  its default behavior.
- Modify: `code/tests/test_facet_pipeline_e2e.py` for the runner boundary.

**Interfaces:**
- Produces a serializable `TopicJob` and `TopicJobReceipt`.
- Produces an importable dependency-factory descriptor (`module`, `qualname`,
  canonical scalar kwargs) for spawned workers.
- Adds local `execution.topic_workers` with default `1`; a CLI override may be
  added but must be recorded in the run manifest.
- Uses spawned child processes. Each child reloads config, constructs its own
  dependencies, owns one topic output root, and retains its validation session
  only until its per-topic projection is sealed.
- The job contract is topic-complete and infrastructure-neutral; no global GPU
  scoring worker or stage-first scheduler is introduced.
- Preserves arbitrary live adapters/callable factories when `topic_workers=1`;
  `topic_workers>1` rejects non-serializable adapters, lambdas, and closures
  before starting any child.

- [ ] **Step 1: Write failing two-topic process-isolation tests**

Run two invented topics with deterministic fake adapters and `topic_workers=2`.
Assert distinct child PIDs, disjoint topic roots, child-owned dependency
construction through a module-level fake factory descriptor, non-serialized
validation sessions, sealed projection receipts, official source-order merge,
and global manifest-last publication. Inject one
child failure and assert no global completion manifest is published.

- [ ] **Step 2: Implement the serializable topic job boundary**

The parent passes only config identity, topic ID, source commit, output
namespace, and the importable factory descriptor. The child reloads config,
resolves the descriptor, and runs the complete topic pipeline through
`build_topic_projection`, returning only receipt/metric scalars. A resumed topic
deep-validates once in its child to mint a new session. Equivalent-output races
may converge after byte/semantic verification; contradictory outputs fail.

- [ ] **Step 3: Add safe configurable local parallelism**

Use a spawn-context process executor. Preserve `topic_workers=1` behavior.
Do not share live retriever, model, SQLite connection, or validation-session
objects. Shared caches must already satisfy Task 7's immutable/partitioned write
contracts. Record worker count and per-topic timing/cache statistics in the run
receipt.

- [ ] **Step 4: Verify warm-cache and bounded-miss execution**

Run a two-topic cache-warm smoke with two workers and zero external calls. Then
exercise a tiny deterministic local-score miss fixture to prove two topic jobs
can own independent scorers without corrupting shared cache state. Local worker
count remains operator-selected according to GPU memory; there is no hard-coded
single scoring bottleneck.

Commit: `perf: execute retrieval topic first`

---

### Task 8: Final verification and independent review

**Files:**
- Modify: `code/trec_rag/README.md` with the v2 cache/topic artifact layout,
  validation command, and no-backward-compatibility note.
- Update: `docs/superpowers/plans/2026-08-03-topic-records-and-retrieval-cache-v2.md` with compact verification evidence only.

**Interfaces:**
- No new production interface.

- [ ] **Step 1: Run static and focused validation**

```bash
.venv/bin/python -m compileall -q code/trec_rag
.venv/bin/python -m pytest \
  code/tests/test_document_store.py \
  code/tests/test_topic_records.py \
  code/tests/test_evidence_pipeline_contract.py \
  code/tests/test_canonical_nugget_contract.py \
  code/tests/test_retrieval_export.py \
  code/tests/test_facet_pipeline_e2e.py \
  code/tests/test_retrieval_cache.py \
  code/tests/test_pipeline.py \
  code/tests/test_remote_pyserini.py \
  code/tests/test_facet_retrieval.py -q
```

- [ ] **Step 2: Run the complete non-GPU unit suite if focused tests pass**

Run: `.venv/bin/python -m pytest code/tests -q --ignore=code/tests/experiments/organizer_pi/test_cli.py`

If environment-only or unrelated pre-existing failures occur, record exact
commands and separate them from regressions; fix every in-scope regression.

- [ ] **Step 3: Verify git scope and secrets**

Confirm only this plan's production/tests/docs are staged, no generated outputs,
raw responses, databases, document bodies, `.env` files, or unrelated dirty
files are included, and submodules remain at recorded commits.

- [ ] **Step 4: Request Sol reviewer audit**

Review the full implementation against the design, advisor requirements,
concurrency/crash semantics, topic isolation, source reconstruction, organizer
output equivalence, and verification evidence. Fix Critical/Important findings
test-first and run a scoped re-review.

- [ ] **Step 5: Commit documentation and verification record**

Commit: `docs: document topic records and cache v2`

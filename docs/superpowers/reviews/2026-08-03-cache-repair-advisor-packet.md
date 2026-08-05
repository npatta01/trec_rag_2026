# Advisory question

Review two fail-closed repair designs before implementation: (A) organizer retrieval checkpoint/continuation repairs and (B) a redesign of historical facet score-ledger promotion. Are the proposed boundaries sufficient, and what exact invariants/tests are required before a clean offline 2025 facet replay?

## Desired outcome

Approve or correct a bounded implementation plan that Luna workers can implement in disjoint files, followed by an independent Sol code review. The result must permit a zero-network, zero-model replay from authenticated 2025 artifacts without making partial, stale, or invented state usable.

## User constraints and corrections

- The user explicitly requires advisor review of each bug and proposed solution before fixing, and another advisor/reviewer pass after implementation.
- The pipeline is topic-first and locally parallelizable by topic. Do not introduce a single GPU-worker bottleneck.
- TopicRecords uses one SQLite database per topic and a shared exact-document content-addressed store. Exact organizer text is authoritative; scoring uses `trec_rag_whitespace_v1` as a separate derived identity.
- Reuse caches wherever authenticated. Do not preserve backward compatibility merely to accept unsafe artifacts.
- Routine implementation should use inexpensive Luna workers; reserve Sol for consequential advice/review.
- Do not edit or overwrite the user-owned dirty `competition_rag.py`, its test, or the non-agentic report/plan.
- No network, hosted-model, or GPU calls are authorized by this advisory review.

## Current project state

- Repository and authoritative worktree: `/home/npatta01/.codex/worktrees/1caa/trec_rag_2026`
- Branch and HEAD: `codex/2025-deepseek-nuggetizer-flow` at `ba0ed13d915b2c35eafceb516ce436ef99ae7098`
- Dirty state: many scoped, uncommitted retrieval/storage changes plus four user-owned files that must not be staged or modified.
- Active workstreams: organizer RetrievalCache v2 repair; historical facet score migration; authenticated planning seed; later topic-first executor. Reviewed generation checkpoint `a3fe571439185d1570e3f91d980076ed096e772a` will be combined only after these gates close.

## Evidence

### E1 — Migration bypasses the current retriever identity

- Status: Verified
- Source: `code/trec_rag/competition_retrieval.py:771-781,1563-1641`; `code/trec_rag/retrieval_export.py:1008-1038,1389-1425`; independent Sol review and offline reproduction.
- Relevance: a base-only canonical checkpoint from corpus epoch A can be projected while the current configured retriever identifies epoch B.
- Compact result: migration topics are excluded from `fresh_pending_topics`; the retriever is constructed only for fresh/resumed topics; exact 11-field comparison is applied only to already-expanded resumed topics. The internal projection loader already accepts `expected_retriever_identity`, but public `build_topic_projection()` does not pass one. An offline test supplied `different-current-epoch` against a checkpoint containing `test-corpus-epoch`; projection still published.

### E2 — Same-process threads can discard a live continuation lease

- Status: Verified
- Source: `code/trec_rag/continuation.py:179-215`; independent Sol reproduction.
- Relevance: process-level ownership is insufficient when multiple threads/tasks use one process.
- Compact result: `discard()` rejects an unexpired `active_recovery`/`active_transport` lease only when the PID prefix differs. A future-expiring current-PID lease was deleted with its ticket.

### E3 — Recovery cache hit leaves a permanent pending ticket

- Status: Verified
- Source: `code/trec_rag/retrievers.py:316-359,592-612,760-805`; `code/trec_rag/continuation.py:145-175`; independent Sol reproduction.
- Relevance: a crash after cache commit but before `_finish_topic()` cannot self-heal.
- Compact result: `retrieve()` returns immediately on a verified cache hit. Ticket/progress cleanup happens only in `_retrieve_uncached()`. A retry returned `{resumed: true, state: pending}` and left both state files.

### E4 — Organizer cache otherwise passes focused tests

- Status: Verified
- Source: independent Sol review command.
- Relevance: favors narrow repairs rather than replacement.
- Compact result: 273 focused tests passed offline. Complete ordinary resume now compares all 11 retriever identity fields; `RetrievalCache` derives and binds its own parser/extractor/scoring normalizer identity. No network/model calls occurred.

### E5 — Score imports are live before the manifest-last receipt

- Status: Verified
- Source: `code/trec_rag/facet_score_ledger_promotion.py:810-975`; `code/trec_rag/rerank_score_cache.py:577,790-943`; independent Sol diagnostic.
- Relevance: the current design does not implement receipt-gated activation and is not crash/concurrency safe.
- Compact result: document and window imports commit separately before the external receipt. `GlobalScoreCache.lookup_many()` does not check that receipt. Killing after the first commit leaves readable scores and makes retry refuse the nonempty unreceipted cache. Removing a receipt still allowed lookup `[1.25]`. Rolling back import A after source B adopted the same score removed B's score while leaving B's import record.

### E6 — Migration-only score identity is descriptive, not enforced

- Status: Verified
- Source: `code/trec_rag/facet_score_ledger_promotion.py:470-507`; `code/trec_rag/rerank_score_cache.py:1047`; `code/trec_rag/facet_retrieval.py:818`; independent Sol poison callback.
- Relevance: historical rows can be laundered into an active scoring context and a miss can load the model.
- Compact result: unrecorded runtime fields are written as `legacy-unrecorded` only in receipt metadata; callers may supply an active scorer context. The cache API still exposes compute-on-miss. A poison callback was invoked on one missing pair.

### E7 — Raw preimages are not authenticated

- Status: Verified
- Source: `code/trec_rag/facet_score_ledger_promotion.py:122-145,500-575`; independent Sol mutation diagnostic.
- Relevance: exact source bytes and normalized scoring bytes must remain distinct authenticated identities.
- Compact result: caller supplies raw `document_text`; validation checks its derived normalized hash against the ledger but has no authenticated document-store/retrieval receipt. Replacing doubled whitespace/newlines with already-normalized text changed raw bytes while promotion still succeeded and receipted the invented raw identity.

### E8 — Score receipt/digest and importer-profile gaps

- Status: Verified
- Source: `code/trec_rag/facet_score_ledger_promotion.py:318-330,735-805`; `code/trec_rag/rerank_score_cache.py:1133+`; independent Sol diagnostics.
- Relevance: a reused receipt must bind all evidence, and the native legacy importer must remain disjoint from facet artifact-v2.
- Compact result: receipt reuse ignores manifest-last, producer commit, receipt path, zero-call evidence, and exact inserted/reused counts. Logical digest collapses by score key and omits source row identity. Adding both native and facet alias field vocabularies let the native importer accept a facet row. Production duplicates exercise distinct source row identities: 118 document duplicate-key groups and 915 window groups.

### E9 — Current score scan is not memory bounded

- Status: Verified
- Source: local `/usr/bin/time` measurement and `code/trec_rag/facet_score_ledger_promotion.py:336-430`; independent Sol review.
- Relevance: the redesign is also intended to remove data-storage and memory inefficiency.
- Compact result: two source ledgers total about 144 MB (23,314 document rows; 123,862 window rows), but validation peaks at `763132 KiB` RSS in 3.56 seconds before loading the exact-text closure. `_scan_source()` retains every row, a key map, and an unused canonical-row set; target conversion/import makes further copies.

### E10 — Production source identities and closure expectations are pinned

- Status: Verified
- Source: production metadata-only scans and reviewed design records.
- Relevance: file-backed migration can be exact and zero-call.
- Compact result: document source SHA-256 `7da82812a20ca50ca2dad3e6728850559ac7f5cb8e0d8518c8810999771a55ae`, 23,314 rows / 23,157 unique keys / 157 duplicates. Window source SHA-256 `11016b63f8520f0c1d0269b21e5c0fb925208e39199907750b14fcd9fe2d4eee`, 123,862 rows / 122,628 unique keys / 1,234 duplicates. No conflicting score groups were observed. The authenticated organizer archive supplies exact bodies through a shared SHA-256 CAS.

## Candidate choices or proposed plan

### A — Narrow organizer repair

1. Construct the lazy Pyserini retriever whenever any topic is fresh, migration-only, or resumed; construction identifies endpoint/index/epoch but does not issue a request.
2. Compute the complete 11-field expected identity once. Add required `expected_retriever_identity` to `build_topic_projection()` and pass it into the already-capable internal projection loader. `_run_topic()` must supply it for migration and fresh publication. Fail before any projection bytes are published on mismatch.
3. Change `discard()` to reject every unexpired `active_recovery` or `active_transport` lease, regardless of PID.
4. After a verified cache hit with a continuation token, call a new state-locked completion helper. It must authenticate ticket token and full request identity, enforce `not_before`, reject any unexpired active lease, validate stale/awaiting state, append a non-secret durable completion event, remove ticket/progress, and fsync the state directory. A missing/changed/already-consumed ticket fails; no hosted call occurs.
5. Tests: epoch mismatch migration publishes nothing; exact identity migration succeeds; same-PID and cross-PID live leases both block discard; expired lease can be discarded; cache-hit continuation clears stale state with zero client construction/calls; live competing recovery blocks cache-hit cleanup; token/request mismatch fails; crash points before/after cleanup converge safely.

Alternative for step 4: reserve a new recovery lease before cache lookup and pass that reservation into the miss path. This provides one ownership protocol but creates an attempt for a cache hit and requires a larger retrieval refactor.

### B — Replace mutable score-cache promotion with an immutable, receipt-gated migration bundle

1. Do not import historical rows into ordinary active `GlobalScoreCache` databases. Build a dedicated `facet-ledger-artifact-v2` bundle in a unique hidden staging directory under a lock scoped by the authorization digest.
2. Stream exact JSONL rows into one file-backed staging SQLite database (separate document/window tables or a `kind` discriminator), retaining source order/row identity and checking exact schema, historical key derivation, duplicates/conflicts, source SHA/count, and closure joins inside SQLite. Do not materialize all rows/text in Python.
3. Authenticate exact document preimages only through verified RetrievalCache v2/DocumentStore receipts and `(topic, lane, docid, rank, content_sha256)` references. The promoter reads CAS bodies by authenticated hash and derives `trec_rag_whitespace_v1`; callers cannot supply free-form raw text as authority.
4. Give the migration bundle a read-only API with lookup only. It has no compute callback, model loader, claim, add, or `score_many` surface. A miss raises before model/GPU code is reachable. Do not label the bundle as a current active scorer context; record absent historical runtime fields as `legacy-unrecorded` and effective normalization separately.
5. Complete and fsync the database and canonical receipt in staging, bind source ordered-row digests, duplicate classification, active closure, target row digests/counts, DB byte hash, producer/version identities, and authorization digest. Atomically rename the complete directory under the authorization lock. Consumers can open only through `open_verified_facet_score_bundle(receipt)`, which revalidates the receipt and DB before returning a non-serializable read capability. No ordinary cache path points into staging/final bundle, so unreceipted rows are unreachable.
6. Convergent writers validate an existing final bundle; contradictory output fails. Failed/abandoned unique staging directories are never read and need no rollback of shared scores. Do not use the current concurrency-unsafe `_rollback_score_import()` path.
7. Make native legacy import require its exact field set and explicitly reject facet-v2 names/aliases.
8. Tests: process death at each staging/publish point leaves no readable bundle; opening without/tampered receipt fails; whitespace-equivalent raw body with the wrong CAS hash fails; exact authenticated body succeeds; miss invokes no poison callback/import/model; full receipt field mutation fails; all source row identities and duplicate groups affect digests; two same-output writers converge; contradictory writers fail; production-sized fixture stays under a measured memory ceiling.

Open design choice: use one immutable SQLite bundle for all 22 topics (deduplicated score keys, read-only concurrent access) or one score bundle per topic. TopicRecords remains one SQLite per topic either way. The recommendation is one immutable 22-topic migration bundle because it is read-only after atomic publication, avoids write contention entirely, and preserves score-key deduplication; topic workers receive only a verified read capability/path descriptor and open their own read-only connection.

## Known unknowns

- Whether the historical score closure is exactly complete after rebinding against all 138 authenticated lanes; metadata and prior ledgers indicate it should be, but the new authenticated CAS-backed closure has not been executed.
- Whether the advisory panel prefers the narrow atomic cache-hit continuation helper or a unified reserve-before-lookup state-machine refactor.
- Whether the immutable score bundle should contain one SQLite database with a kind column or two databases inside one atomically published directory.
- The canonical live organizer corpus epoch is not proved by the old archive cohort. Offline replay can use the reviewed operator-archive epoch; a later live run needs an authoritative epoch supplied explicitly.

## Scope boundary

Read-only review. No implementation, artifact publication, network/model calls, or external actions.

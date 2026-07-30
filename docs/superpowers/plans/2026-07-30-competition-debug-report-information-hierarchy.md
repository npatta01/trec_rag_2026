# Competition Debug Report Information Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the narrative, subnarratives, and generated answer first; add an exact overall/per-subnarrative funnel; and make selected-document detail opt-in.

**Architecture:** Keep all artifact loading and validation unchanged. Add immutable, pure funnel projections beside the existing report records, then compose a new funnel stage from those projections and reorder only the HTML presentation boundary. Preserve every selected-document, reference, and diagnostic record behind native disclosures.

**Tech Stack:** Python 3.12, frozen dataclasses, deterministic standalone HTML/CSS/JavaScript, pytest, headless Google Chrome, uv with the existing `.venv`.

## Global Constraints

- Do not rerun retrieval, reranking, decomposition, canonicalization, or RAG generation.
- Do not change artifact schemas, organizer outputs, competition submission formats, or either organizer submodule.
- Keep source-derived strings HTML-escaped and out of executable JavaScript.
- Keep the report dependency-free, dark/light compatible, keyboard accessible, and free of page-level horizontal overflow.
- Preserve the existing topic, stage, citation, and selected-document anchor namespaces.
- Publish only to the already-authorized tailnet-only Tailscale Serve path after all verification passes.

## Completion Evidence

Completed on 2026-07-30.

### Auditable Safety Baseline

**Pre-run state, recorded immediately before the final post-run renderer invocation:**

- `git status --short --branch` printed only
  `## codex/competition-retrieval-rag...origin/master [ahead 68]`: the tracked
  worktree was clean at reviewed implementation commit
  `dd1672722d211df1f54b37a811279271ee5f5dba`.
- `git submodule status --recursive` recorded
  `trec-rag-data` at `a6255c10119a2984a874f46172d94045168ab1f3`
  and `trec-rag-skills` at `f281e88f61252662033c681df8b1ed2d0ceda97e`.
  Each submodule's `git status --porcelain=v1 --branch` printed only
  `## HEAD (no branch)`, so both were detached at the recorded organizer commit
  and had no modified or untracked files.
- Official input SHA-256 values were
  `17855f3ace3b662b67db4a6ff2bb31c7be857f26ff5be2b0d78644287cef5c71`
  for `r_output_trec_rag_2026.tsv` and
  `65d2efae3dd54d0428f9bf49ed9fb2666996453418128d6df2a400b89ea076af`
  for `rag_output_trec_rag_2026.jsonl`.
- `tailscale serve status` reported
  `https://npatta01-framework.tail481212.ts.net (tailnet only)` mapped to
  `/home/npatta01/codex-rendered`. `tailscale funnel status` showed only that
  same tailnet-only Serve mapping and no public Funnel mapping; Funnel was not
  configured.

**Post-run state, rechecked after rendering, testing, browser QA, portal copy,
and the evidence commit:**

- `git status --short --branch` again had no file entries and printed only
  `## codex/competition-retrieval-rag...origin/master [ahead 69]` at
  `e291ac304515e746091648c3dbae7ce40052ad7c`; the one-commit ahead-count change
  is the tracked completion-evidence commit, not an artifact or organizer
  change.
- Both submodule SHAs remained exactly
  `a6255c10119a2984a874f46172d94045168ab1f3` and
  `f281e88f61252662033c681df8b1ed2d0ceda97e`. Both submodule status commands
  again printed only `## HEAD (no branch)`, with no file entries.
- The retrieval TSV and RAG JSONL SHA-256 values remained exactly
  `17855f3ace3b662b67db4a6ff2bb31c7be857f26ff5be2b0d78644287cef5c71`
  and `65d2efae3dd54d0428f9bf49ed9fb2666996453418128d6df2a400b89ea076af`.
- Serve and Funnel status again showed only the authorized
  `tailnet only` Serve mapping. No public Funnel mapping appeared. Thus the
  organizer submodules, official inputs, and private-only exposure state were
  unchanged from the pre-run baseline.

- The post-run debug-report CLI regenerated the report from existing sealed
  artifacts only. Its receipt included `rag2026-0` and `rag2026-1`, reported
  `rag_included: true`, and matched the official retrieval TSV SHA-256
  `17855f3ace3b662b67db4a6ff2bb31c7be857f26ff5be2b0d78644287cef5c71`
  and RAG JSONL SHA-256
  `65d2efae3dd54d0428f9bf49ed9fb2666996453418128d6df2a400b89ea076af`.
- Full repository verification passed: 659 tests in 30.83 seconds.
- Real Chrome checks passed at 1440×1000 and an emulated 390×844 viewport:
  exactly one visible topic and one selected topic control; story stages in
  approved order; 38 citation links with zero broken targets; topic-2 citation
  navigation opened its owning topic and reference disclosure; ten direct
  selected-document rows plus one closed remaining-90 disclosure; and
  `scrollWidth == clientWidth` at both widths. Isolated light-mode screenshots
  of the story, funnel, and compact selected rows were inspected at both widths
  and retained only as ignored local scratch evidence.
- The exact overall funnels were verified as `709 → 100 → 800 → 2016 → 320 →
  148 → 71` for `rag2026-0` and `576 → 100 → 600 → 1374 → 240 → 90 → 60`
  for `rag2026-1`; all per-subnarrative rows also matched the stored records.
- The source report, mode-0600 rendered copy, and live HTTPS response are
  byte-identical at SHA-256
  `909fba5e5f00f16cee1166bd6dbf1dee746a2e8afbbd9a2d99645fe3dc8af7d4`;
  live HTTPS returned 200. The artifact is self-contained and no local `.env`
  secret value was embedded.
- Tailscale Serve remains `tailnet only`; no Funnel or public deployment was
  configured. Organizer submodules remained clean at `a6255c10119a2984a874f46172d94045168ab1f3`
  and `f281e88f61252662033c681df8b1ed2d0ceda97e`, and the official input hashes
  remained unchanged.

---

### Task 1: Add deterministic funnel projections and the funnel stage

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py` near `TopicReport` and `_render_topic`
- Test: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: `TopicReport`
- Produces: `_SubnarrativeFunnelReport`, `_FunnelReport`, `_project_funnel(topic: TopicReport) -> _FunnelReport`, and `_render_funnel_overview(topic: TopicReport, prefix: str) -> str`

- [ ] **Step 1: Write failing projection tests**

Add controlled replacements around the existing debug-run fixture so counts cannot pass accidentally:

```python
def test_funnel_projection_counts_overall_and_per_subnarrative(tmp_path: Path) -> None:
    config_path, _output = _write_debug_run(tmp_path)
    topic = load_debug_report_data(config_path).topics[0]
    projected = debug_report._project_funnel(topic)

    assert projected.facet_only_documents == len(topic.new_documents)
    assert projected.selected_documents == len(topic.selected_documents)
    assert projected.document_subnarrative_rankings == len(topic.passage_rankings)
    assert projected.stored_passages == sum(
        len(ranking.winning_passages) for ranking in topic.passage_rankings
    )
    assert projected.evidence_clusters == len(topic.evidence_clusters)
    assert projected.final_nuggets == len(topic.canonical_nuggets)
    assert projected.final_documents == len(topic.retrieval_output.documents)
    assert [row.subnarrative_id for row in projected.subnarratives] == [
        item.subnarrative_id for item in topic.subnarratives
    ]
```

Add a second test with two rankings for one DocID and one ranking for another so `ranked_documents == 2` while the ranking-record count is 3. Give the rankings different winning-passage tuple lengths and assert their sum. Add controlled cluster/nugget rows for both subnarratives and assert exact per-row counts.

- [ ] **Step 2: Run the projection tests and confirm RED**

Run:

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'funnel_projection' -q
```

Expected: failure because `_project_funnel` and its records do not exist.

- [ ] **Step 3: Implement the immutable projections**

Add these exact records and compute all values from validated tuples:

```python
@dataclass(frozen=True)
class _SubnarrativeFunnelReport:
    subnarrative_id: str
    subnarrative_text: str
    ranked_documents: int
    stored_passages: int
    evidence_clusters: int
    final_nuggets: int


@dataclass(frozen=True)
class _FunnelReport:
    facet_only_documents: int
    selected_documents: int
    document_subnarrative_rankings: int
    stored_passages: int
    evidence_clusters: int
    final_nuggets: int
    final_documents: int
    subnarratives: tuple[_SubnarrativeFunnelReport, ...]
```

`_project_funnel` must iterate `topic.subnarratives` for row order, count unique
`ranking.docid` values per subnarrative, sum stored winning-passage records, and
count matching clusters/nuggets. Do not aggregate logits, scores, or ranks.

- [ ] **Step 4: Add failing funnel-renderer contracts**

Assert that the new `stage-literal-rag2026-0-funnel-overview` contains seven
explicitly labeled count cards, a per-subnarrative table with the four approved
columns, and the exact projected integers. Add an original-only replacement
with no subnarratives and assert a clear fallback message instead of an empty
table. Assert “document × subnarrative rankings” appears so the pair count is
not mislabeled as unique documents.

- [ ] **Step 5: Render the funnel stage and make the focused tests GREEN**

Implement `_render_funnel_overview` with:

```html
<ol class="funnel-counts" aria-label="Overall retrieval and evidence funnel">
  <li class="funnel-count"><strong>…</strong><span>Selected documents</span></li>
</ol>
```

Use `_table` only for the narrow per-subnarrative comparison. Add responsive
`.funnel-counts` and `.funnel-count` CSS with `auto-fit`, `minmax`, and existing
color variables. Run the projection/renderer tests, then the full debug-report
module.

- [ ] **Step 6: Commit Task 1**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Add debug report funnel overview"
```

### Task 2: Put the story first and move technical answer context below it

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py` in `_render_topic`, `_render_narrative`, `_render_subnarratives`, and `_render_final_rag`
- Test: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: `_render_funnel_overview` from Task 1
- Produces: story-first stage order while preserving all existing stage IDs

- [ ] **Step 1: Add failing order and default-density tests**

Extract each stage position from rendered HTML and assert:

```python
expected = (
    "narrative", "subnarratives", "final-rag", "funnel-overview",
    "new-documents", "selected-documents", "top-passages",
    "final-selected-nuggets", "final-retrieval",
)
positions = [rendered.index(f'stage-literal-rag2026-0-{suffix}') for suffix in expected]
assert positions == sorted(positions)
```

Within `final-rag`, assert `Generated answer` and the first answer item precede
generation provenance and referenced documents. Within narrative, assert the
source seal is inside a closed disclosure. Within each subnarrative card,
assert text precedes a closed `Retrieval query and seal details` disclosure and
that literal queries/hashes remain inside it.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'story_stage_order or answer_precedes_provenance or concise_subnarrative' -q
```

- [ ] **Step 3: Reorder and simplify the story stages**

Change `_render_topic` to compose:

```python
stages = (
    _render_narrative(topic, prefix),
    _render_subnarratives(topic, prefix),
    _render_final_rag(topic, prefix),
    _render_funnel_overview(topic, prefix),
    _render_new_documents(topic, prefix),
    _render_selected_documents(topic, prefix),
    _render_passages(topic, prefix),
    _render_nuggets(topic, prefix),
    _render_retrieval(topic, prefix),
)
```

Move the narrative seal into a `technical-provenance` disclosure. Move BM25
query lists into the subnarrative technical disclosure. In `_render_final_rag`,
render answer items first, then a closed generation-provenance disclosure and a
closed referenced-documents disclosure. Keep citation reference-card anchors in
the DOM so citation links continue to resolve when their disclosure is closed.

- [ ] **Step 4: Run focused and complete report tests**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -q
```

Expected: all report tests pass; update only older order assertions that
conflict with the approved hierarchy, without weakening escaping, completeness,
or anchor contracts.

- [ ] **Step 5: Commit Task 2**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Put debug report story stages first"
```

### Task 3: Make selected documents compact by default

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py` in `_render_selected_documents` and `_render_selected_document_card`
- Test: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: existing `_selected_document_reason`, `_best_stored_passage`, `_membership_coverage`, and collision-safe topic anchors
- Produces: `_render_selected_document_disclosure(topic, item) -> str`

- [ ] **Step 1: Add failing compact-selection tests**

Build 12 selected documents by replacing the fixture's validated report data
and render it. Assert ten direct `selected-document-disclosure` records, one
closed `selected-document-remainder` summary labeled “Show remaining 2 selected
documents,” and all 12 stable selected-document IDs in the section.

For the first disclosure, split summary from body and assert the summary includes
rank, DocID, original/facet status, selected lane, and lane rank but excludes
passage text, membership score details, hashes, and the long selection reason.
Assert those complete values remain inside the disclosure body.

- [ ] **Step 2: Run the compact-selection test and confirm RED**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py \
  -k 'compact_selected_document' -q
```

- [ ] **Step 3: Implement nested native disclosures**

Change each selected record to:

```html
<li>
  <details class="selected-document-disclosure" id="selected-document-…">
    <summary><span class="card-rank">#1</span><code>docid</code><span>Original · original rank 1</span></summary>
    <div class="selected-document-detail">…existing reason, passage, metadata, provenance…</div>
  </details>
</li>
```

Render the first ten directly. Wrap the remaining list items in one closed
`selected-document-remainder` disclosure. Preserve native markers, 44px
targets, stored order, representative-passage caveat, bounded previews, and all
technical fields.

- [ ] **Step 4: Run focused, module, and full tests**

```bash
uv run --no-sync .venv/bin/python -m pytest \
  code/tests/test_competition_debug_report.py -q
uv run --no-sync .venv/bin/python -m pytest -q
```

- [ ] **Step 5: Commit Task 3**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Compact selected document diagnostics"
```

### Task 4: Regenerate and verify the real two-topic report

**Files/artifacts:**
- Source report: `/home/npatta01/data/competitions/trec_rag_2026/outputs/codex-pr24-two-topic-smoke/competition_debug_report.html`
- Private rendered copy: `/home/npatta01/codex-rendered/plans/trec-rag-2026-competition-debug-report.html`

**Interfaces:**
- Consumes: existing two-topic retrieval/RAG configs and all Tasks 1–3
- Produces: verified source and byte-identical tailnet-only rendered HTML

- [x] **Step 1: Record the safety baseline**

Record `git status`, both submodule SHAs/statuses, and SHA-256 for the official
retrieval TSV and RAG JSONL. Confirm Tailscale Serve says `tailnet only` and no
Funnel is configured.

- [x] **Step 2: Run the post-run CLI only**

```bash
uv run --no-sync .venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/local/rag26_competition_retrieval_two_topic_smoke.yaml \
  --rag-config configs/local/rag26_competition_rag_gpt_sol_two_topic_smoke.yaml
```

Assert the receipt reports two topic IDs, `rag_included: true`, and the expected
official input hashes. Do not invoke retrieval or generation CLIs.

- [x] **Step 3: Run fresh full verification**

```bash
uv run --no-sync .venv/bin/python -m pytest -q
```

Use headless Chrome at desktop and mobile widths to assert one visible topic,
the story-first anchor order, exact funnel counts, one selected topic tab, zero
broken citation targets, selected-document remainder behavior, citation
navigation within topic 2, and `scrollWidth == clientWidth`. Capture isolated
desktop/mobile screenshots of the story, funnel, and compact selected rows for
visual inspection.

- [x] **Step 4: Refresh and verify the private copy**

Install the verified source HTML with mode `0600` at the existing rendered
path. Confirm source, rendered file, and live HTTPS response have identical
SHA-256 values; live HTTPS returns 200; Tailscale Serve remains tailnet-only;
and the official retrieval/RAG hashes and organizer submodule states are
unchanged.

- [x] **Step 5: Record completion evidence**

Update this plan with the final test count, Chrome evidence, live SHA-256, and
tailnet-only status. Commit only tracked source/test/docs changes; do not commit
the generated private report or scratch screenshots.

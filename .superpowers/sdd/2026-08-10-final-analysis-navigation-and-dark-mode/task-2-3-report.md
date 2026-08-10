# Combined Task 2–3 report — final-analysis navigation and agent orientation

Recorded 2026-08-10. This report covers only the reader-facing final-analysis
navigation and completed-project orientation work. Task 1’s canonical dark-mode
CSS/render changes remain intact and were not rewritten. No portal files,
accepted organizer artifacts, private evaluation outputs, or provider state were
modified.

## Task 2 — three direct final-analysis links

The root artifact hub, report index, and canonical architecture source now use
the same three choices:

| Label | Destination | Access |
| --- | --- | --- |
| Retrieval Quality Analysis | `reports/2026-retrieval-nugget-coverage.html` from the root hub; `2026-retrieval-nugget-coverage.html` from `reports/` and the architecture report | Tracked/repository-relative |
| RAG Analysis: rag26-ss1 | `https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html` | Private / tailnet |
| RAG Analysis: rag26-ms1-final | `https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html` | Private / tailnet |

The root hub has a dedicated **Final analyses** card. The report index places
the same direct links before historical material. The architecture report adds
a **Final analysis reports** section to the canonical QMD and the generated
HTML; it does not introduce a comparison or side-by-side landing page.

The architecture source explains that the RAG pages are separate 119-topic
RAGDoll citation-support reports. RAGDoll measures citation support, not
official TREC correctness; qrel/gold metrics are unavailable, and the
evaluation did not influence accepted priority. Accepted files and recorded
priorities remain unchanged.

### Task 2 TDD evidence

The new contracts were written before the page changes. The RED run:

```text
node index.test.js
```

failed with `artifact hub should link reports/2026-retrieval-nugget-coverage.html`.
The corresponding report-index and architecture runs failed for their missing
Retrieval analysis target. After the three authored pages were updated, the
canonical source was rendered with:

```text
quarto render reports/2026-competition-architecture.qmd
```

The GREEN run passed all three suites:

```text
root artifact hub smoke test passed
reports index smoke test passed
2026 competition architecture smoke test passed
```

The architecture contract counts each analysis target once in source and
rendered output, checks the two nearby **Private / tailnet** labels, rejects
comparison/side-by-side targets, and checks the RAGDoll scope caveats.

Task 2 commit: `a2427fee` — `docs: link final retrieval and RAG analyses`.

## Task 3 — completed-project orientation

`AGENTS.md` now identifies:

- the tracked Retrieval Quality Analysis;
- the exact private tailnet URLs and run IDs `rag26-ss1` and `rag26-ms1-final`;
- the accepted single-pass and multi-stage RAG JSONLs as immutable evaluated
  inputs;
- `.agents/skills/trec-rag-competition-debug-report/SKILL.md` as the accepted
  input evaluation contract;
- the distinction between citation support and an official TREC score;
- unavailable qrel/gold metrics and the absence of priority influence; and
- the requirement that raw evaluation work stays outside git and the rendered
  portal.

### Task 3 TDD evidence

The orientation assertions were added before the `AGENTS.md` update. The RED
run failed on the first missing signal:

```text
Error: AGENTS.md should orient agents to final analyses:
reports/2026-retrieval-nugget-coverage.html
```

After the orientation section was added, the architecture suite passed, along
with the root and report-index suites. The Task 3 commit is:

```text
docs: orient agents to final analyses
```

This commit also contains this combined report so the implementation evidence
travels with the orientation change.

## Final verification

After both commits:

```text
quarto render reports/2026-competition-architecture.qmd
git diff --exit-code -- reports/2026-competition-architecture.html
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
git diff --check
```

All commands passed. The second command confirms the generated architecture
HTML is consistent with the canonical QMD after the final rerender. The three
Node suites report their smoke-test success, and `git diff --check` reports no
whitespace errors.

Static link/access checks confirm one occurrence of each exact analysis target
in each assigned navigation artifact, nearby **Private / tailnet** text for
both private RAG links, and no comparison or side-by-side report target. The
tailnet URLs are the verified destinations recorded by the accepted-evaluation
plan; this documentation task did not make network/provider calls or alter
the private portal.

The final tracked worktree is clean. Accepted organizer files and private
evaluation outputs remain untouched.

## Fix Round 1 — bind visible labels to exact destinations

The first review identified two contract gaps in the tests, while the authored
navigation and orientation content itself was already correct:

1. Root and report-index tests checked labels and URLs independently, so a
   swap of the `rag26-ss1` and `rag26-ms1-final` visible labels could pass.
2. The architecture test checked AGENTS signals independently, but did not
   bind each run ID to its complete private URL and accepted JSONL input path.

The fix changes only the three owned test files. Each HTML suite now extracts
the anchor for every exact href and requires its normalized visible text to
equal the expected label. The architecture suite applies the same binding to
both the QMD Markdown link and rendered HTML anchor. All suites include an
active fault-injection fixture that swaps the two RAG labels and asserts the
binding contract rejects it.

The AGENTS contract now asserts exact triples:

| Run ID | Private report URL | Accepted evaluated input |
| --- | --- | --- |
| `rag26-ss1` | `https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html` | `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl` |
| `rag26-ms1-final` | `https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html` | `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/multistage/rag_output_trec_rag_2026.jsonl` |

The AGENTS contract also injects and rejects a broken/swapped URL, removed
input, swapped input paths, and swapped run labels.

### Fix Round 1 TDD and mutation evidence

Before the stronger assertions, a legacy independent-label/URL harness accepted
the swapped fixture:

```text
legacy-independent-contract-accepted-root-swap=true
legacy-independent-contract-accepted-reports-swap=true
legacy-substring-agents-contract-accepted-broken-map=true
```

After the fix, the active mutation assertions and all normal suites pass:

```text
root artifact hub smoke test passed
reports index smoke test passed
2026 competition architecture smoke test passed
```

The final verification also reran:

```text
quarto render reports/2026-competition-architecture.qmd
git diff --exit-code -- reports/2026-competition-architecture.html
git diff --check
```

The canonical rerender completed and generated HTML remained byte-consistent;
whitespace checks passed. No authored content, portal file, accepted artifact,
private output, or provider state changed in this fix round.

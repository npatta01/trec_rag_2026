# Task 5 privacy-scan fix report

Recorded 2026-08-10. This report covers the renderer prerequisite only. The
accepted organizer files, retrieval inputs, shared judge cache, portal, and
all provider paths remained outside the change.

## Diagnosis and privacy boundary

The multi-stage cache-only report previously failed because a selected-evidence
passage was also a substring of valid generated answer/subnarrative text. The
full dynamic denylist was correct and the page contained no raw evidence field;
the collision came from scanning two different privacy classes in one string.
The generic scan with `denylist=()` was already clean for both real reports.

The fix preserves the complete bundle-derived denylist and the existing
fail-closed generic patterns. `write_report` now performs two checks:

1. The actual unredacted HTML is scanned with an empty dynamic denylist, so
   credentials, filesystem paths, hashes/digests, provider events, and private
   runtime fields remain forbidden even when they occur in an allowlisted answer
   or narrative node.
2. A scan-only DOM projection replaces only exact allowlisted presentation
   nodes (`blockquote.narrative`, `p.subnarrative-text`, `p.response-text`, and
   `q.judgment-statement`) with a per-scan unguessable sentinel. The complete
   dynamic denylist is scanned against that projection. Evidence in template
   chrome, attributes, provenance, metrics, or any other non-allowlisted node
   remains visible and fails closed.

The projection uses `HTMLParser`, class-token matching, preserved raw start
tags/entities, nested-tag tracking, and a fail-closed fallback to the original
page for malformed or unclosed markup. It does not globally replace strings,
so an evidence substring outside an explicitly allowlisted node cannot be
masked.

## TDD and regression evidence

The new `PrivacyTests.test_allowlisted_evidence_reuse_passes_but_template_leak_fails`
was written first and observed RED: the synthetic selected passage
`Shared selected evidence passage.` was rejected when placed in answer and
subnarrative text. The GREEN test then proved both sides of the boundary:

- `write_report` succeeds when the same dynamic evidence value is wholly inside
  the rendered answer and subnarrative nodes.
- Patching the renderer to append the identical value in a non-allowlisted
  `<aside>` makes `write_report` raise `ReportPrivacyError`.

The existing `test_dynamic_evidence_text_leak_is_rejected` remains unchanged
and passes. Focused verification:

```text
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py \
  -k 'allowlisted_evidence_reuse or dynamic_evidence_text_leak or clean_report_passes' -q
3 passed, 108 deselected

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py code/tests/test_ragdoll_io.py \
  code/tests/test_accepted_rag_evaluation.py -q
172 passed, 106 subtests passed
```

Compile and whitespace checks passed:

```text
.venv/bin/python -m compileall -q code/trec_rag/friendly_report.py \
  code/tests/test_offline_evaluation.py
git diff --check
```

## Real private cache-only render

Using the exact accepted single-pass and multi-stage inputs, the private
main-checkout retrieval config/handoff, the shared judge cache, and no
`--run-judge`, both reports were rendered and atomically written under:

```text
/home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/task5-privacy-fix-CuxbfS/
```

The directory is mode `0700`; each report is mode `0600`; no report or adjacent
private JSON/JSONL was copied to the portal. The authoritative receipts were:

| run | topics | tasks | hosted calls | cache hits / misses / writes | failed / conflicts | report bytes |
|---|---:|---:|---:|---:|---:|---:|
| `rag26-ss1` | 119 | 3,155 | 0 | 0 / 3,155 / 0 | 0 / 0 | 2,914,417 |
| `rag26-ms1-final` | 119 | 7,008 | 0 | 0 / 7,008 / 0 | 0 / 0 | 5,475,990 |

Both receipts retain the exact ordered topic scope (`rag2026-0` through
`rag2026-118`) and report `missing_judgments` equal to task count, as expected
for a cache-only inventory with no hosted judging. The important prerequisite
proved here is privacy-scanned write success, not completed evaluation.

## Self-review

- Full evidence values remain in `denylist_from_bundle`; no threshold was
  lowered and no evidence value was removed.
- Generic scanning still sees the actual page, including allowlisted text.
- Dynamic redaction is DOM/node scoped, selector exact, attribute preserving,
  and fail-closed on malformed markup.
- Atomic report replacement and `0600` output permissions are unchanged.
- The accepted RAG/Retrieval artifacts and shared cache were not modified by
  the code change; the real run was cache-only with zero hosted calls.

## Final full-suite verification

The repository suite was run from fresh temporary test state after the owned
files were committed. The one transient rerun before committing could not clone
the intentionally dirty source checkout; that is a test harness cleanliness
condition, not a code failure. The final clean-source result is recorded here
after commit:

```text
PYTHONPATH=code TMPDIR=<fresh-private-temp-dir> .venv/bin/python -m pytest -q
3234 passed, 19 skipped, 139 subtests passed
```

## Fix Round 1 — adversarial findings

Round 1 addressed all three important findings in the amended fix commit
recorded by the handoff:

1. `PrivacyDenylist` now separates always-forbidden identifiers (references,
   segment document IDs, and caller extras) from selected passage text. The
   identifier set is scanned against every raw, decoded, and browser-visible
   projection of the actual unredacted page, so a response containing
   `doc-original` is rejected even though response text is allowlisted. Passage
   values retain the complete existing length policy and are only collision-
   exempted in the explicit model projection.
2. The permissive HTML node parser was removed. The dynamic scan now deep-copies
   the validated `Presentation`, replaces only `narrative`, `subnarratives`,
   answer `text` (which also covers repeated judgment statement text), and
   renders that model through the same `render_report` function. Renderer tags,
   attributes, chrome, metrics, and template output remain intact. Regressions
   inject passage text in nested tags/attributes inside a response node and in
   a non-allowlisted template location; both fail closed.
3. Generic patterns now run over raw HTML, HTML-decoded HTML, and a
   browser-visible/plain projection. Block-level boundaries are retained so a
   preceding answer number cannot concatenate with a leaked `/home/...` path,
   while inline formatting such as `OPEN<strong>ROUTER</strong>` remains joined
   and is caught. Formatted-answer regressions cover credentials and paths;
   direct projection regressions cover credentials, paths, and hashes.

TDD RED/GREEN evidence for this round:

```text
RED: 6 new adversarial cases failed before the implementation (identifier in
      response, split raw/decoded/plain generic terms, and nested renderer leak).
GREEN: PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k 'privacy or generic_patterns_scan' -q
       10 passed, 104 deselected, 6 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py -q
114 passed, 112 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py code/tests/test_ragdoll_io.py \
  code/tests/test_accepted_rag_evaluation.py -q
175 passed, 110 subtests passed
```

The final Round 1 real cache-only replay used the same private root and again
succeeded for both runs with zero hosted calls and zero cache writes. Receipts
remain `3,155` and `7,008` tasks over the exact 119-topic order; both reports
remain mode `0600` and no portal copy exists. The committed-source full suite,
compile, diff, and final commit checks all pass.

Final clean-source full-suite result after this round:

```text
PYTHONPATH=code TMPDIR=<fresh-private-temp-dir> .venv/bin/python -m pytest -q
3237 passed, 19 skipped, 145 subtests passed
```

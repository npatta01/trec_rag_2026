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

## Fix Round 2 — raw-render sink-aware validation

Round 2 replaces the model-level authorization boundary with validation of the
actual HTML returned by `render_report`. The complete denylist remains intact,
including every selected passage and every always-forbidden reference/document
identifier. `write_report` now scans the actual page for identifiers and all
generic privacy patterns, then parses that same raw page a second time for the
passage collision check. The second projection preserves every raw start/end
tag and attribute and replaces only text data events in the exact renderer sinks:
`blockquote.narrative`, `p.subnarrative-text`, the pre-citation answer text in
`p.response-text`, and `q.judgment-statement`. Only renderer-known `strong` and
`br` descendants are permitted there; the direct `span.citations` subtree is
left unredacted. Evidence in an attribute, an unexpected descendant, a citation
subtree, or any other page location therefore remains visible to the denylist
scan. A malformed, mismatched, or unclosed page raises `ReportPrivacyError`
instead of guessing at a safe projection.

The same stdlib `HTMLParser` receives raw HTML with `convert_charrefs=False` for
the browser-visible projection. It decodes entities only in their text-event
handlers, tracks void elements and stack balance, preserves inline adjacency,
and inserts boundaries only for block elements and `br`. Raw and HTML-decoded
projections remain separate, so generic patterns and identifiers are checked
against raw markup, decoded attributes/text, faithful parser-visible text, and a
parser-backed compact projection that catches entity-encoded formatting without
parsing decoded HTML as markup.

Adversarial TDD coverage added in `test_offline_evaluation.py` proves:

- a renderer-time copy of `view.topics[0]["answer"][0]["text"]` into an
  `<aside>` attribute and nested `<strong>` is rejected; this is not a captured
  constant and is caught from the actual rendered page;
- a selected passage remains accepted only in the approved rendered answer and
  subnarrative text, while the same value in nested attributes or template
  chrome fails; the existing dynamic evidence leak rejection remains green;
- response-contained document identifiers are rejected even in allowlisted text;
- `OPEN<span title="&gt;">ROUTER</span>`, split `/home`, split hashes, split
  `doc-original`, and split passage values all fail; entity-encoded formatting,
  formatted answer credentials, and paths fail as well;
- block descendants, mismatched tags, unclosed tags, and passage text in the
  unredacted citation subtree fail closed.

Round 2 RED/GREEN and regression results:

```text
RED: the renderer-time answer-copy regression and quoted/split parser cases
     failed before the raw sink parser was implemented.
GREEN: PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py \
  -k 'privacy or generic_patterns_scan or dynamic_answer_copy' -q
       13 passed, 104 deselected, 11 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py -q
       117 passed, 117 subtests passed
```

Both real accepted-artifact renders were replayed privately with no
`--run-judge`. The output root was
`/home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/task5-privacy-fix-round2/`;
it was not copied to the portal. Both reports are mode `0600` and both
manifests retain the exact 119-topic order:

| run | topics | tasks | hosted calls | cache hits / misses / writes | failed / conflicts | report bytes |
|---|---:|---:|---:|---:|---:|---:|
| `rag26-ss1` | 119 | 3,155 | 0 | 0 / 3,155 / 0 | 0 / 0 | 2,914,417 |
| `rag26-ms1-final` | 119 | 7,008 | 0 | 0 / 7,008 / 0 | 0 / 0 | 5,475,990 |

The parser-backed write path succeeded for both single-pass and multi-stage
reports. No hosted calls or cache writes occurred.

Round 2 self-review:

- No passage value was removed and the existing `len(value) >= 6` policy was
  unchanged; identifiers and all generic patterns remain unredacted scans of
  the actual page.
- No global string replacement or decoded-before-parse operation authorizes a
  sink. Raw start/end tags and attributes are preserved, and parser failure is
  fail-closed.
- The renderer is invoked once per report; the dynamic projection is derived
  from that exact returned page, so copied answer values cannot evade the
  scan.
- Only the three owned tracked files changed; accepted artifacts, caches,
  private outputs, docs/hubs, AGENTS instructions, and portal files were not
  modified or published.

Final committed-source verification:

```text
TMPDIR=/tmp/trec-rag-round2-suite-final.yo6HFE \
  PYTHONPATH=code .venv/bin/python -m pytest -q
3240 passed, 19 skipped, 150 subtests passed in 1:53

PYTHONPATH=code .venv/bin/python -m py_compile \
  code/trec_rag/friendly_report.py code/tests/test_offline_evaluation.py
git diff --check
```

The owned files were committed as `850dd29c` (`fix: enforce sink-aware report
privacy`) after the clean-source run. The temporary suite directory was private
scratch only and was removed; no hosted call, cache write, portal copy, or
accepted-artifact change occurred.

## Fix Round 3 — permanent citation boundary and lexical completion

Round 3 closes the remaining parser gaps. Within each `p.response-text`, the
first direct `span.citations` permanently sets `citations_seen` until that `p`
closes. Once the citation subtree closes, only ordinary whitespace data may
occur; non-whitespace text, entities, comments, declarations, processing
instructions, or any start tag fail closed and are never redacted. A renderer
copy of the live `view.topics[0]["answer"][0]["text"]` inserted immediately
after the citation chrome is therefore rejected from the actual rendered page.
The state resets only when that response paragraph closes, preserving valid
multi-answer reports and zero-citation response paragraphs.

The parser now also checks raw lexical completion before and after
`HTMLParser.close()`, rejects literal `<` recovered as data, and validates raw
end tags against the strict `</tag>` grammar. Attributes, trailing slashes,
extra tokens, incomplete comments/declarations/processing instructions, and
unfinished tails cannot be silently normalized by the permissive stdlib parser.
Valid emitted doctype/comments, void elements, and normal start/end tags remain
accepted. No decoded-before-parse operation or global replacement was added.

Round 3 adversarial TDD coverage includes the exact review pages:

```text
<p>safe<
<p>safe<!--
<p>safe<!DOCTYPE
<p>safe<?pi
<p>safe</p extra>
<p>safe</p/>
```

It also covers a live post-citation answer copy, citation entities/comments/
tags, valid doctype/comment markup, and a valid zero-citation response.

RED/GREEN evidence:

```text
RED: the live post-citation renderer copy was accepted by the prior parser;
     the six lexical pages with permissive end tags/tails were not all rejected.
GREEN: PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py \
  -k 'post_citation or privacy_parser_fails_closed' -q
       1 passed, 117 deselected, 9 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py -q
       118 passed, 123 subtests passed

PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py code/tests/test_ragdoll_io.py \
  code/tests/test_accepted_rag_evaluation.py \
  code/tests/test_competition_cache_bundle_offline_replay.py -q
       194 passed, 123 subtests passed
```

Both accepted-artifact reports were replayed privately with no `--run-judge`
under:
`/home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/task5-privacy-fix-round3/`.
No report or adjacent private evaluation file was copied to the portal:

| run | topics | tasks | hosted calls | cache hits / misses / writes | failed / conflicts | report bytes |
|---|---:|---:|---:|---:|---:|---:|
| `rag26-ss1` | 119 | 3,155 | 0 | 0 / 3,155 / 0 | 0 / 0 | 2,914,417 |
| `rag26-ms1-final` | 119 | 7,008 | 0 | 0 / 7,008 / 0 | 0 / 0 | 5,475,990 |

Both reports are mode `0600`; manifests preserve the exact 119-topic order;
failed and conflicting judgments are zero; and cache writes are zero.

Round 3 self-review:

- The complete dynamic evidence denylist and existing length policy remain
  unchanged. Only approved pre-citation text is redacted; post-citation data
  is preserved or causes a fail-closed error.
- Per-response citation state prevents a later answer copy from being treated
  as allowlisted text while still admitting zero-citation answers.
- Raw tail checks and strict end-tag validation reject all six review pages
  without rejecting the real renderer's doctype/comments/void tags.
- The Round 2 implementation was committed as `850dd29c`; its report
  amendment was finalized in `97232e55`. Round 3's final amended commit and
  clean-source full-suite result are recorded below.

Final committed-source verification:

```text
TMPDIR=/tmp/trec-rag-round3-suite.F6Ujs9 \
  PYTHONPATH=code .venv/bin/python -m pytest -q
3241 passed, 19 skipped, 156 subtests passed in 1:51

PYTHONPATH=code .venv/bin/python -m py_compile \
  code/trec_rag/friendly_report.py code/tests/test_offline_evaluation.py
git diff --check
```

The implementation was committed as `f65de8cb` before this clean-source run;
the final report amendment is included in the amended commit below. The
private temporary suite directory was removed after verification. No hosted
calls, cache writes, accepted-artifact changes, or portal copies occurred.

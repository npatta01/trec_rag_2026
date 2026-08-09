# Luna Whole-Answer Splice Replay Results

**Date:** 2026-08-09
**Status:** completed two-topic shadow replay; reject as a replacement for the hardened Sol
revision flow.

## Decision

Do not replace the per-group omission audits and second Sol revision with one direct Luna splice.
The candidate was extremely cheap and safe, but it was not capable enough on the known-positive
topic. Keep the hardened bounded-revision flow and add the already-recommended cheap
operation-level accept/reject screen before deterministic application.

The replay made one good insertion on topic `233` and avoided that topic's known weak replacement.
On topic `499`, however, it made one redundant, only partially supported insertion and missed the
five meaningful edits recovered by the full flow. The negative probe passed; the positive probe
failed. That is enough to reject this simplification without tuning on the two topics.

## Tested flow

For each topic, the replay authenticated and reused the existing Luna blueprint and validated Sol
draft, then made exactly one new Luna medium-thinking call over the complete narrative, blueprint,
selected evidence, and indexed draft. Luna could return `keep_draft` or at most three atomic splice
operations. Existing local validators checked operation geometry, evidence aliases, citation
domain, answer structure, reference consistency, and the 1,024-word ceiling. Any invalid response
would have fallen back atomically to the draft.

The production runner and checked-in competition configs were unchanged. All generation and
evaluation artifacts remain ignored and private.

## Generation results

| Topic | Draft | Replay final | Replay action | Luna | Sol | Replay call cost |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `233` | 671 words / 38 objects | 709 / 39 | 1 insertion | 1 | 0 | `$0.001771100` |
| `499` | 665 words / 32 objects | 685 / 33 | 1 insertion | 1 | 0 | `$0.002692075` |
| **Total** |  |  | **2 insertions** | **2** | **0** | **`$0.004463175`** |

Both replay finals passed structural validation without fallback. Topic `233` retained 36 compact
references and topic `499` ended with 44. The one-call replay consumed 11,827 prompt tokens on
topic `233` and 18,638 on topic `499`.

## Like-for-like cost estimate

Because the experiment reused frozen planner/draft artifacts, the table below reconstructs what
each flow would have cost if run from the beginning using the provider-reported receipts from the
same topics.

| Flow | Topic `233` | Topic `499` | Two-topic total |
| --- | ---: | ---: | ---: |
| Luna plan + Sol draft only | `$0.154347520` | `$0.174093750` | `$0.328441270` |
| Plan + draft + direct Luna splice | `$0.156118620` | `$0.176785825` | `$0.332904445` |
| Hardened full flow | `$0.266484170` | `$0.354187875` | `$0.620672045` |

The direct-Luna version would save `$0.287767600`, or **46.4%**, versus the full flow. Almost all
of that saving comes from removing the second Sol call. Consolidating only the cheap Luna audits
would save about `$0.0071` across these two topics, which is not a meaningful cost lever.

## Post-hoc metrics

Gold nuggets and support judgments were opened only after both replay candidates were sealed; no
evaluation signal entered generation.

### Topic 233

| Metric | Draft | Direct Luna | Full flow |
| --- | ---: | ---: | ---: |
| Strict vital coverage | .666667 | .666667 | .700000 |
| Strict all coverage | .528302 | .547170 | .566038 |
| Partial vital coverage | .700000 | .733333 | .750000 |
| Partial all coverage | .594340 | .632075 | .632075 |
| Weighted first-citation P/R | .881579 | .884615 | .875000 |
| Weighted all-citation P/R | .868421 | .871795 | .856250 |
| Hard citation precision | .763158 | .769231 | .750000 |

The one insertion was materially relevant and fully supported. It improved broad coverage and
support slightly, and direct inspection found no visible seam. The direct-Luna result was safer
than the full flow on this negative probe, although it recovered fewer useful omissions.

### Topic 499

The coverage nuggetizer completed only two of seven windows before one DeepSeek V4 Flash call
stalled for more than fifteen minutes. The run was stopped instead of permitting up to four
fifteen-minute retries. No incomplete coverage score is reported. The completed citation-support
evaluation is sufficient to diagnose the actual insertion:

| Metric | Draft | Direct Luna | Full flow |
| --- | ---: | ---: | ---: |
| Weighted first-citation P/R | .671875 | .666667 | .708333 |
| Weighted all-citation P/R | .656250 | .651515 | .701389 |
| Hard citation precision | .375000 | .363636 | .444444 |
| FS / PS / NS citations | 15 / 31 / 3 | 15 / 32 / 3 | 20 / 30 / 3 |

The new citation was judged Partial Support, lowering each support metric. Direct inspection also
found the addition redundant with surviving draft material and stronger than its source on two
details. More importantly, the direct Luna call missed all five substantive edits that made the
full-flow final qualitatively preferable. This is a genuine capability loss, not merely evaluator
noise.

## Cost of evaluation

- Topic `233`: nugget coverage `$0.002700`; citation support `$0.000159`.
- Topic `499`: citation support `$0.000208`; the stopped partial nugget run recorded `$0.0009`
  across two completed calls and produced no score.

The evaluation delay was caused by one evaluator subprocess that did not complete within fifteen
minutes, not by the replay runner. It was terminated deliberately to keep this experiment bounded.

## Competition interpretation

The competition rewards more than nugget count: completeness must coexist with citation support,
coherence, and pairwise answer preference. Topic `233` shows that a conservative cheap pass can
avoid a bad edit. Topic `499` shows that it cannot reliably recognize and express the important
missing material needed to improve the full narrative.

The second Sol call therefore has evidence of meaningful value; its gains are not uniformly small.
The safer simplification is to keep the current planner, draft, audits, and bounded Sol revision,
then use one cheap whole-answer operation screen to reject weak or redundant operations. The
optional third Sol reservation remains validation repair only. If cost must be cut immediately,
use the validated planner-plus-draft output rather than pretending the direct Luna splice is a
reliable quality pass.

## Verification

- Both replay manifests sealed with one semantic-success Luna receipt and zero Sol calls.
- Source state, handoff digest, topic digest, and registered source-file hashes authenticated.
- Focused tests passed: 47 tests across the replay, splice, and blueprint modules.
- Ruff, Python compilation, and `git diff --check` passed before the live run.
- Organizer structure, citations, references, and word limit validated for both finals.
- Topic `233` produced complete nugget and support evaluations; topic `499` produced a complete
  50-task support evaluation and an explicitly incomplete, unscored nugget evaluation.
- A fresh repository-wide run reached 2,754 passed and 19 skipped, with 18 failures confined to
  linked-worktree environment checks: this worktree has no local `.venv`/`hf` executable, its
  shared virtual environment points RAGDoll at the main checkout, and the existing shared handoff
  makes two path-equality checks resolve outside the worktree. No failure exercised the replay
  module. These environment fixtures were not altered for this throwaway experiment.

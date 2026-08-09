# Bounded Narrative Revision Results

**Date:** 2026-08-08
**Status:** complete as a throwaway, three-topic shadow experiment; no production
hardening decision.

## Decision

The bounded post-draft audit and one revision improved the first-pass evaluator on
topics `233` and `300`, but the first pass regressed on `499`. A three-run repeat
diagnostic on `499` reversed that apparent conclusion often enough to invalidate
tuning against the first pass. The experiment does not establish a reliable quality
gate: the approach is plausible, but this sample and judge variance do not justify
productionization.

The next step is the smallest throwaway edit-operation prototype, followed by one
fresh sequential three-topic shadow batch with that new contract frozen across all
three topics. Do not harden the production runner yet.

## Verified paired results

Metrics are draft → final. `strict` requires full nugget support; `partial` is the
existing partial-credit score. `FS`, `PS`, and `NS` are aggregate semantic citation
support labels for Full, Partial, and No Support.

| Topic | Words | Answer objects | Strict vital | Strict all | Partial vital | Partial all | Support labels (FS / PS / NS) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `233` | 732 → 664 | 28 → 41 | 0.633333 → 0.700000 (+0.066667) | 0.528302 → 0.566038 (+0.037736) | 0.683333 → 0.716667 (+0.033334) | 0.566038 → 0.584906 (+0.018868) | 11 / 36 / 3 → 37 / 6 / 1 |
| `300` | 690 → 804 | 33 → 42 | 0.161290 → 0.387097 (+0.225807) | 0.212766 → 0.361702 (+0.148936) | 0.290323 → 0.467742 (+0.177419) | 0.382979 → 0.425532 (+0.042553) | 12 / 36 / 5 → 22 / 36 / 3 |
| `499` | 714 → 927 | 36 → 49 | 0.770833 → 0.625000 (−0.145833) | 0.692308 → 0.584615 (−0.107693) | 0.833333 → 0.760417 (−0.072916) | 0.784615 → 0.692308 (−0.092307) | 19 / 28 / 3 → 28 / 29 / 5 |

The same support-judge outputs gave weighted all-judged citation support of
`0.616071 → 0.914634` on `233`, `0.618687 → 0.724206` on `300`, and
`0.680556 → 0.734694` on `499`; hard support precision was respectively
`0.250000 → 0.853659`, `0.303030 → 0.476190`, and `0.444444 → 0.530612`.
These are aggregate metrics only; no raw answers, prompts, passages, gold text, or
individual judgments are reproduced here.

## Calls and exact spend

The Luna count is the durable manifest call count, including transport/request
outcomes. Sol counts are semantic reservations; no repair reservation was used.
The generation subtotal is the planner, draft, audit, and revision provider spend.
The evaluation subtotal is the post-hoc support/nuggetizer spend.

| Topic | Luna calls | Sol calls (total; draft / revision / repair) | Generation | Evaluation | Topic total |
|---:|---:|---:|---:|---:|---:|
| `233` | 6 (5 responses, 1 HTTP 400 rejection) | 2 / 1 / 1 / 0 | `$0.302288550` | `$0.021261000` | `$0.323549550` |
| `300` | 5 (5 responses) | 2 / 1 / 1 / 0 | `$0.349980200` | `$0.025048000` | `$0.375028200` |
| `499` | 8 (8 responses) | 2 / 1 / 1 / 0 | `$0.407830150` | `$0.029685000` | `$0.437515150` |
| **Initial trial** | **19** | **6** | **`$1.060098900`** | **`$0.075994000`** | **`$1.136092900`** |

Topic `233`'s zero-cost HTTP 400 was a pre-completion schema rejection: the nullable
`replacement_answer_index` property was missing from the strict schema's `required`
array. Commit `bd6bb10` fixed that contract, and the run resumed from the already-valid
planner and draft state without another Sol draft call.

The repeat `499` nuggetizer diagnostic cost exactly `$0.014623000`, giving total
experiment spend of **`$1.150715900`**. The repeat was post-hoc only and did not feed
back into audit, revision, or repair.

## Repeat-499 judge variance

The first run favored the draft on every listed metric. Runs two and three favored
the final overall; the displayed-precision `all` score in run three was tied. The
three-run means were:

| Metric | Draft mean | Final mean |
|---|---:|---:|
| Strict vital | 0.701389 | 0.715278 |
| Strict all | 0.641026 | 0.651282 |
| Partial vital | 0.798611 | 0.812500 |
| Partial all | 0.746154 | 0.735897 |

The run-level strict-vital / strict-all / partial-vital / partial-all values were:

| Repeat | Draft | Final |
|---:|---|---|
| 1 | 0.770833 / 0.692308 / 0.833333 / 0.784615 | 0.625000 / 0.584615 / 0.760417 / 0.692308 |
| 2 | 0.625000 / 0.584615 / 0.760417 / 0.700000 | 0.750000 / 0.676923 / 0.833333 / 0.761538 |
| 3 | 0.708333 / 0.646154 / 0.802083 / 0.753846 | 0.770833 / 0.692308 / 0.843750 / 0.753846 |

This variance invalidates tuning the prompts or thresholds to the first-pass `499`
regression. The correct conclusion is uncertainty, not that revision is harmful.

## Structural audit and shadow gate

The `499` trial had 36 draft answer objects and 49 final objects. Twenty-nine of the
36 draft sentence strings were replaced, the audit contained 17 cards, and one card
reported replacement index `37`, outside the valid zero-based draft range `0..35`.

The gold-free preservation prototype read only the draft/final submissions and audit
cards:

- `233`: allow, paraphrase-anchor margin `+0.062`;
- `300`: allow, margin `+0.000`;
- `499`: reject only the invalid replacement index; its anchors otherwise passed
  (margin `+0.100`).

Thus the structural contract is separable in this sample, but exact or paraphrase
anchors are not a quality discriminator. The durable gate should validate replacement
indices and use by-construction edit operations/deterministic patch application;
protected anchors are diagnostics and a fallback signal, not a gold-free quality
proof.

## Advisor consensus and next design

The independent design review converged on these constraints:

- validate every audit replacement index before applying an edit;
- prefer by-construction patch/edit operations with deterministic application over
  shadow inference from the final text, leaving untargeted draft objects unchanged;
- retain protected anchors as diagnostics and fallback checks;
- keep gold, qrels, nugget assignments, support judgments, and evaluator outputs out
  of generation-time inputs;
- use at most two routine Sol calls (draft and revision), with one optional Sol call
  only for deterministic validation repair.

First build only the throwaway edit-operation contract and deterministic applier. Then
run one fresh sequential three-topic shadow batch under those constraints. Do not
change the production runner or harden the prototype until that batch provides a
stable, judge-reproducible result.

## Provenance and privacy boundary

The aggregate values above were verified from the private trial manifests,
stage receipts, paired nuggetizer `run_metrics.csv` files, support metric rows,
support-label JSONL files, and usage CSVs under the three ignored bounded-revision
trial roots. The implementation and shadow gate never opened gold, qrels, passages,
or post-hoc evaluation data during generation. This committed report contains no raw
prompts, answers, passages, gold text, or individual judgments.

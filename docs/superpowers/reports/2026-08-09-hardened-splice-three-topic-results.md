# Hardened Bounded-Splice Three-Topic Results

**Date:** 2026-08-09
**Status:** completed sequential shadow batch; promising with an operation-level quality gap;
not promoted to the competition runner.

## Decision

Keep the hardened bounded-splice approach and make one focused change before a broader run:
accept or reject each proposed splice independently based on atomic citation support and whether a
replacement is stronger than the draft object it removes. Preserve the validated draft object when
a splice fails that check.

The frozen batch produced two clear qualitative improvements (`300` and `499`) and one mixed result
(`233`). Three of topic `233`'s four splices were useful and well grounded; the remaining replacement
combined weakly supported material, removed a clearer draft detail, and introduced redundancy. This
is a local operation-selection problem, not evidence that the whole revision design should be
discarded.

Automatic coverage results were mixed and noisy. Strict macro coverage fell slightly, while
partial-credit coverage was essentially flat. Semantic citation support improved in aggregate. The
qualitative review is decisive for topic `300`: its final answer only inserted three meaningful,
supported facts and removed nothing, so a lower one-pass nugget score cannot by itself represent a
real loss of content.

## Frozen batch and execution boundary

Topics `233`, `300`, and `499` were run sequentially under one frozen contract. All three had strict
full extraction coverage in the upstream diagnostic, so the experiment isolated generation behavior
rather than retrieval absence.

The corrected batch ran in the isolated worktree
`.worktrees/hardened-splice-three-topic` on branch
`codex/hardened-splice-three-topic`. The supported production runner and checked-in full-run configs
were unchanged. Private prompts, evidence, answers, judgments, and provider responses remain only in
ignored output directories.

| Topic | Narrative groups | Claim hints | Selected passages | Citation-domain docs |
| ---: | ---: | ---: | ---: | ---: |
| `233` | 4 | 37 | 188 | 143 |
| `300` | 4 | 80 | 186 | 126 |
| `499` | 7 | 86 | 309 | 223 |

Each topic used one Luna planner, one Luna audit per narrative group, one Sol draft, and one Sol
bounded revision. No corrected-batch topic needed the optional third Sol validation-repair call.

## Candidate shape and spend

| Topic | Draft words / objects | Final words / objects | Splices | Luna | Sol | Generation | Evaluation | Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `233` | 671 / 38 | 704 / 40 | 2 replace, 2 insert | 5 | 2 | `$0.266484170` | `$0.012666000` | `$0.279150170` |
| `300` | 829 / 42 | 897 / 45 | 3 insert | 5 | 2 | `$0.295567100` | `$0.020334000` | `$0.315901100` |
| `499` | 665 / 32 | 748 / 36 | 1 replace, 4 insert | 8 | 2 | `$0.354187875` | `$0.016073000` | `$0.370260875` |
| **Total** |  |  | **12 operations** | **18** | **6** | **`$0.916239145`** | **`$0.049073000`** | **`$0.965312145`** |

Every final answer validated against the organizer schema, citation domain, reference consistency,
and 1,024-word cap. The Luna count includes planning and per-group audits. Sol never exceeded the
two routine calls per topic, leaving the optional third reservation unused.

## Paired post-hoc metrics

The nuggetizer and citation-support judge used the same DeepSeek V4 Flash model and settings for
both arms. Gold nuggets and evaluator outputs were opened only after each final candidate was sealed;
they never fed back into generation.

### Nugget coverage

| Topic | Strict vital | Strict all | Partial vital | Partial all |
| ---: | ---: | ---: | ---: | ---: |
| `233` | .666667 → .700000 | .528302 → .566038 | .700000 → .750000 | .594340 → .632075 |
| `300` | .354839 → .290323 | .404255 → .297872 | .451613 → .435484 | .489362 → .414894 |
| `499` | .791667 → .791667 | .707692 → .723077 | .854167 → .843750 | .753846 → .792308 |
| **Macro** | **.604391 → .593997** | **.546750 → .528996** | **.668593 → .676411** | **.612516 → .613092** |

The strict macro deltas are `-0.010394` vital and `-0.017754` overall. Partial-credit deltas are
`+0.007818` vital and `+0.000576` overall. This misses the experiment's predeclared automatic
success gate, but it does not establish that the final answers are worse: topic `300` is an
insertion-only revision, and direct inspection found all three additions useful and supported.

### Citation support

Every answer object in both arms has at least one citation, so weighted precision and weighted
recall are identical in this batch. The shared values are shown below.

| Topic | Weighted P/R, first | Weighted P/R, all | Hard precision | Final FS / PS / NS |
| ---: | ---: | ---: | ---: | ---: |
| `233` | .881579 → .875000 | .868421 → .856250 | .763158 → .750000 | 31 / 11 / 2 |
| `300` | .809524 → .811111 | .809524 → .811111 | .619048 → .622222 | 29 / 28 / 0 |
| `499` | .671875 → .708333 | .656250 → .701389 | .375000 → .444444 | 20 / 30 / 3 |
| **Macro** | **.787659 → .798148** | **.778065 → .789583** | **.585735 → .605555** |  |

Macro support improved by `+0.010489` for the first citation, `+0.011518` across all citations, and
`+0.019820` for hard precision. The twelve new splice objects contributed fourteen citation tasks:

| Topic | New-object citation labels | Interpretation |
| ---: | --- | --- |
| `233` | 3 FS / 1 PS / 1 NS | one bad replacement; other three edits useful |
| `300` | 2 FS / 1 PS / 0 NS | all three additions meaningful |
| `499` | 5 FS / 1 PS / 0 NS | strong operation set overall |

## Qualitative answer judgment

The read-only Luna review examined the untouched narrative, paired answers, audit cards,
authenticated selected evidence, and support judgments. It did not use gold nuggets or qrels.

- **Topic `233`: change.** Both arms answer the complete narrative. The final adds four relevant
  areas, but its first replacement is only partially supported, loses a clearer draft detail, and
  overlaps later material. Retain the other three operations and reject or rewrite that one.
- **Topic `300`: final preferred.** The final still covers mitigation, adaptation, Antarctic
  protection, global policy actors, and the economics of action versus inaction. Its three
  insertions add a concrete technology result and two important economic caveats without deleting
  draft material or creating visible seams.
- **Topic `499`: final preferred.** Five substantive edits improve the arguments, definitions,
  cultural and religious context, international legal variation, and moral nuance. Five of six new
  citation links are fully supported; the remaining one is partial but paired with a second source
  that supports the claim.

The aggregate verdict is **change, not stop**: bounded splicing preserves the draft and can recover
meaningful omissions, but candidate validity is not enough to make every operation worth accepting.

## Competition interpretation

The competition is multi-objective. This batch directly measured nugget coverage and weighted
citation precision and recall, and used a blind qualitative review as a proxy for answer preference
and coherence. It did not produce an official anonymized battle judgment, so pairwise preference
remains unknown.

That distinction matters here. Optimizing the single stochastic nuggetizer pass would incorrectly
reject topic `300`, even though its final contains every draft object plus three grounded additions.
Conversely, accepting all structurally valid operations would keep topic `233`'s weak replacement.
The correct unit of judgment is therefore the splice operation inside the full-answer context, not
the whole answer's noisy score alone.

## Minimal next change

Add one cheap, evidence-only operation screen after Sol proposes the splice set and before local
application:

1. judge each new object against its own cited passages for full support and atomicity;
2. for replacements, compare the new object with the exact draft object it would remove and reject
   the operation if it loses a narrative-relevant detail or merely duplicates surviving prose;
3. apply only accepted operations, preserving every rejected draft object;
4. retain the existing deterministic final validator and optional third Sol call solely for
   malformed or structurally invalid candidates.

This is not another whole-answer rewrite. It is a bounded accept/reject pass that would have kept
the useful edits on all three topics while removing the observed topic-`233` failure. Validate that
screen on a small fresh batch before considering production integration.

## Diagnostic bug found before the frozen batch

An initial topic-`233` diagnostic was excluded from all tables above. Its valid splice candidate
became invalid because the experimental rebind path validated the assembled answer before removing
references orphaned by replacements. The optional third Sol repair could not fix that local
bookkeeping error, so the experiment correctly fell back to the draft.

Commit `c88f825` now applies the repository's safe reference normalization before validation. A
captured real candidate compacted from 44 to 41 references and validated without changing answer
text. The focused regression test failed before the fix and passed afterward; the corrected batch
then produced three genuine draft-to-final pairs. The excluded diagnostic cost `$0.347733850` and
is not part of the `$0.965312145` cohort total.

## Verification and privacy

- All three corrected candidates passed structural generation validation.
- Every support task completed with no missing, duplicate, or conflicting judgments.
- Focused prototype tests passed: 42 tests, plus Ruff, Python compilation, and diff checks.
- The generated outputs and evaluator artifacts are ignored and remain private.
- No production runner, checked-in competition config, retrieval artifact, or public artifact was
  changed.

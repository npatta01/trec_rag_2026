# Coverage-aware selected-evidence AG pilot

## Decision

Promote the coverage-aware strategy to Issue #52 Phase 2 on the remaining eight
matched development topics. Do not make it the production default yet.

Arm B improves every aggregate nugget metric and does so on three of four topics.
It also improves mean citation support, but that support result is not robust: Topic
72 supplies the entire positive macro delta. This is directional evidence from one
generation sample per arm, not a significance result.

## Controlled comparison

Both arms used Topics 31, 72, 200, and 225 from authenticated handoff digest
`451bd0478f5419e0f5471b170757fb641195ef42d3fde771914260077d03c7ad`.
Model, reasoning effort, evidence bytes, topic order, schema, normalization, word
trim, citation validation, and evaluation identity were fixed. The treatment changed
only `generation.strategy`:

- `baseline`: the historical checked-in prompt, byte-for-byte unchanged.
- `coverage_aware`: an ordered handoff-derived checklist, a supported 900-1,000-word
  target, focused citations, and one private pre-emission coverage/support audit in
  the same completion.

Generation never opened gold nuggets, coverage evaluator plans or reports, qrels,
the TREC run, the full-text ZIP, or retrieval candidates outside the handoff.

## Aggregate results

| Metric | Baseline | Coverage-aware | Delta |
|---|---:|---:|---:|
| Strict vital coverage | 0.3954 | **0.4237** | **+0.0283** |
| Strict all coverage | 0.3794 | **0.3974** | **+0.0180** |
| Vital coverage, partial credit | 0.4814 | **0.5170** | **+0.0356** |
| All coverage, partial credit | 0.4517 | **0.4813** | **+0.0296** |
| Weighted citation precision, first | 0.9077 | **0.9291** | **+0.0214** |
| Weighted citation precision, all | 0.9116 | **0.9291** | **+0.0175** |
| Hard citation precision | 0.8310 | **0.8974** | **+0.0665** |
| Mean answer words | 482.3 | **760.8** | +278.5 |
| Known cost | $1.9993 | $2.1624 | +$0.1631 |

Coverage denominators are 256 released nuggets per arm, including 191 vital
nuggets. Baseline labels were 92 Support, 34 Partial, and 130 Not Support;
coverage-aware labels were 99 Support, 37 Partial, and 120 Not Support. Both arms
had zero failed nugget assignments.

Citation denominators are 116 baseline tasks and 160 treatment tasks. Final label
counts were 97 Full / 17 Partial / 2 None for baseline and 144 Full / 10 Partial /
6 None for treatment. All expected citation tasks have one completed final judgment,
with zero unresolved failures and zero conflicting completed labels.

## Topic effects

| Topic | Strict vital A | Strict vital B | Delta | Weighted first A | Weighted first B | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | **0.5000** | 0.4167 | -0.0833 | **0.9667** | 0.9405 | -0.0262 |
| 72 | 0.2500 | **0.3056** | +0.0556 | 0.7500 | **0.9583** | +0.2083 |
| 200 | 0.5082 | **0.5902** | +0.0820 | **0.9143** | 0.8289 | -0.0853 |
| 225 | 0.3235 | **0.3824** | +0.0588 | **1.0000** | 0.9886 | -0.0114 |

The strict-vital gain survives removal of any one positive topic, so no single
improvement creates the coverage result. Topic 31 is a real counterexample: 227 more
words did not improve strict coverage. The next analysis should compare its omitted
and newly covered nuggets after generation, now that the outputs are sealed.

Citation support is less stable. Removing Topic 72 changes the weighted-first delta
from +0.0214 to about -0.0409. Topic 200 also gains the most strict coverage while
losing the most citation precision, showing that the coverage/support trade-off has
not disappeared.

## Mechanism and limits

The checklist increased total answer length by 58%, from 1,929 to 3,043 words, and
answer objects from 108 to 160. It did not actually reach the requested 900-1,000
words: treatment answers ranged from 681 to 847 words. More supported answer capacity
is the leading explanation for the coverage gain, but Topic 31 shows that length
alone is insufficient.

The pilot uses one stochastic generation per arm and four retrospectively selected
development topics. The cohort is full only against planner-derived obligations, not
organizer gold. Automated nugget and support judgments are suitable for paired
development comparisons, not claims about official scores or the 119 test topics.

## Reliability, cost, and incident

All eight organizer JSONL rows passed exact narrative, citation-domain, and 1,024-word
validation. Generation needed eight successful GPT-Sol calls, no semantic repairs,
and no transport retries. Four initial baseline requests were rejected with HTTP 401
before the key was corrected; they reported no usage or cost.

Known provider cost was $4.1618: $4.0926 generation, $0.0282 nugget assignment, and
$0.0410 citation support. One support task initially returned no assistant text and
completed on its authorized retry; that failed attempt reported no cost.

The pinned RAGDoll `support judge` command exposes `--dry-run` through its shared CLI
but does not check it in the support flow. The intended cache-only preflight therefore
executed the 116- and 160-task support batches. This was detected from raw receipts,
reported immediately, and no additional support expansion was run. Pi 0.73.1 was used
for both arms because the older handover's stated 0.83.0 package version is not
published in the current npm registry. These facts limit comparison to older records
but do not break the paired identity within this experiment.

## Next gate

Run the same frozen arms on Topics 14, 37, 161, 233, 300, 499, 707, and 897. Retain
the strategy only if the coverage delta remains positive without a material rise in
No Support citation rate, and report paired topic deltas rather than only a macro.

Private answers, selected passages, raw judge events, provider responses, gold data,
and local configs remain under ignored `outputs/`, `cache/`, and `configs/local/`.

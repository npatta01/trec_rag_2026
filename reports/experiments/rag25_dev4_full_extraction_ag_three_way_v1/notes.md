# Three-way selected-evidence answer-generation pilot

## Decision

Keep `coverage_aware` as the current competition choice. The new two-stage
`priority_aware` arm improves nugget coverage slightly, but its citation-support
regression is too large to accept.

The next priority-aware iteration should preserve the validated claim plan while
forcing one atomic claim and one strongest citation per answer object. It should
not add another planning model or expose evaluation data to generation.

## Controlled comparison

All three arms use Topics 31, 72, 200, and 225 from authenticated handoff digest
`451bd0478f5419e0f5471b170757fb641195ef42d3fde771914260077d03c7ad`.
The GPT-Sol writer, medium reasoning, selected evidence, topic order, organizer
schema, word limit, citation validator, and DeepSeek evaluation identity are
fixed. The arms differ only in answer-generation strategy:

- `baseline`: the historical one-shot prompt.
- `coverage_aware`: one-shot generation with a handoff-derived group checklist
  and one private coverage/support audit.
- `priority_aware`: DeepSeek Flash first extracts a bounded, evidence-linked
  essential/important/optional claim plan; GPT-Sol then writes from that validated
  plan and the same authenticated evidence.

Generation was frozen before released gold nuggets were opened. It never read
qrels, gold nuggets, RAGDoll results, the TREC run, the full-text ZIP, or retrieval
candidates outside the handoff.

## Aggregate results

| Metric | Baseline | Coverage-aware | Priority-aware | Priority vs coverage |
|---|---:|---:|---:|---:|
| Strict vital coverage | 0.3954 | 0.4237 | **0.4333** | **+0.0096** |
| Strict all coverage | 0.3794 | 0.3974 | **0.4040** | **+0.0066** |
| Vital coverage, partial credit | 0.4814 | 0.5170 | **0.5441** | **+0.0272** |
| All coverage, partial credit | 0.4517 | 0.4813 | **0.4980** | **+0.0167** |
| Weighted citation precision, first | 0.9077 | **0.9291** | 0.7971 | **-0.1320** |
| Weighted citation precision, all | 0.9116 | **0.9291** | 0.8007 | **-0.1284** |
| Hard citation precision | 0.8310 | **0.8974** | 0.6322 | **-0.2652** |
| Mean answer words | 482.3 | 760.8 | 633.3 | -127.5 |
| Known arm cost | $1.9993 | $2.1624 | $2.2822 | +$0.1198 |

Priority-aware produced 102 Support, 43 Partial, and 111 Not Support nugget
labels across the same 256 nuggets used by both prior arms. It led coverage-aware
on aggregate nugget metrics despite using 510 fewer words.

Citation quality moved in the opposite direction. Priority-aware produced 204
citation tasks for 132 answer objects, versus 160 for 160 coverage-aware objects.
Its labels were 108 Full, 91 Partial, and 5 None. Multiple passages were often
attached to one compound sentence, so an individual citation supported only part
of that sentence. This explains the large increase in Partial Support and the
drop in both weighted and hard precision.

## Topic effects

| Topic | Strict vital coverage-aware | Strict vital priority-aware | Delta | Weighted first coverage-aware | Weighted first priority-aware | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.4167 | 0.4167 | 0.0000 | **0.9405** | 0.8784 | -0.0621 |
| 72 | 0.3056 | **0.3472** | +0.0417 | **0.9583** | 0.7500 | -0.2083 |
| 200 | **0.5902** | 0.5574 | -0.0328 | **0.8289** | 0.6406 | -0.1883 |
| 225 | 0.3824 | **0.4118** | +0.0294 | **0.9886** | 0.9194 | -0.0693 |

The strict-vital gain occurs on two topics, one topic ties, and Topic 200
regresses. Citation precision regresses on every topic, so this is not a
single-topic artifact.

## Plan behavior

The validated plans retained all handoff groups and contained 128 claims: 93
essential, 30 important, and 5 optional. Per-topic plan sizes were 25, 38, 35,
and 30 claims for Topics 31, 72, 200, and 225 respectively.

The live probe exposed four planner integration failures that are now handled
locally and covered by tests:

1. The portable `json_object` route renamed `priority` to `classification` and
   omitted `group_id`; the prompt now states the exact item fields and shape.
2. DeepSeek reasoning exhausted 6,000 and then 12,000 output-token ceilings; the
   planner ceiling is 24,000.
3. Valid claims arrived out of priority/group order and with exact duplicates;
   validation now sorts and deduplicates deterministically.
4. One plan exceeded the global claim cap; validation now reserves each group's
   minimum quota, enforces per-group maxima, and fills remaining capacity in
   priority order.

Resume also revalidates completed raw planner responses before making another
hosted call. This recovered Topic 72 without repaying for the same plan.

## Reliability and cost

All four final JSONL rows passed exact narrative, selected-evidence citation
domain, zero-to-three citation, and 1,024-word validation. Answers ranged from
588 to 698 words. The frozen output SHA-256 is
`2071679c3d5fa51f36a5ed14baaec72e895fa7f415bf3d09d75a42128e8b2b70`.

The final priority-aware arm cost $2.2822: $2.2367 generation and planning,
$0.0127 nugget assignment, and $0.0328 citation support. Two discarded planner
probes cost another $0.5857, making the complete priority-aware development work
$2.8679, below the authorized $10 ceiling.

RAGDoll completed all 204 support tasks. One task initially returned no assistant
text and completed on resume. The validator now permits failed attempt history
only when exactly one completed, identity-matching final label exists; duplicate
completed labels still fail closed.

## Recommendation

Do not promote multi-stage priority-aware generation in its current form. Keep
coverage-aware as the best balanced RAG strategy on this cohort. The next
controlled arm should change only the priority writer contract:

- emit one atomic planned claim per answer object;
- cite one strongest linked document by default;
- allow a second citation only when no single passage supports the complete claim;
- preserve every group quota and essential claim before adding important claims;
- run the same four-topic gate, then the remaining eight matched topics only if
  citation support returns near the coverage-aware level.

This is a four-topic, one-sample development result. Automated nugget and support
judgments are useful for paired iteration, not a claim about official test scores.
NDCG is not applicable because retrieval and ranking were fixed upstream.

Private answers, selected passages, raw judge events, provider responses, gold
data, and local configs remain under ignored `outputs/`, `cache/`, and
`configs/local/`.

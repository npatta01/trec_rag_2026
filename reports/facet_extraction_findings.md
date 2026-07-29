# Generated subnarratives and canonical nuggets: findings

## Bottom line

The facet pipeline is ready as a **mechanics experiment**, not as a validated
retrieval or nugget-quality method. The supported runner preserves narrative-only
inputs, exact fallback, source provenance, deterministic artifact bytes, sealed
resume, and organizer-compatible export. Two development topics showed that
model-generated or paraphrased subnarratives can discover additional
known-relevant documents, but individual lanes were uneven and many documents
were unjudged.

The first canonical pilot produced 149 model claims across ten subnarratives
plus one 20-sentence extractive fallback after a timeout. An independent Codex
advisor estimated that 115/149 model claims were strictly supported, 117/149
were concrete and useful, and about 90/149 met both criteria. Roughly 60%
meeting both criteria is not production-ready. These are advisor judgments on
development-tuned outputs, not human labels or held-out organizer evaluation.
The counts were reconstructed in the Codex session recorded by
`b95cb2436ff0d4ad53622ab42b525ea75c4ae93b:reports/facet_extraction_findings.md`;
no per-claim advisor labels were saved, so they are directional and not a
reproducible evaluation artifact.

Terminology used below: **qrels** are relevance-judgment files; **R@k** is the
fraction of known-relevant documents found in the first *k* results; **CE** is
the cross-encoder reranker; **MMR** (maximal marginal relevance) balances
relevance and novelty; **RRF** (reciprocal rank fusion) combines ranked lists;
and **nDCG@10** is normalized discounted cumulative gain in the first ten
results. LLM means large language model.

## Supported experimental run

Use the single official interface:

```bash
.venv/bin/python -m trec_rag.official_run configs/facet_pilot_v1.yaml
```

The strict configuration fixes the experiment/run identity, topic source,
retrieval index and depth, reranker and candidate-pool depth, caches, selection
policy, evidence budget, claim limit, and supporting-document limit. The runner
performs five internal stages: planning, retrieval/reranking, extractive
evidence, canonicalization, and organizer export.

Only official topic IDs and untouched narratives are runtime inputs. Topic
titles, organizer subnarratives, organizer nuggets, and qrels are excluded from
both hosted model boundaries and every reusable runtime API. They belong only
in post-seal evaluation.

Planning transport, parsing, schema, or semantic failure retains exactly the
original narrative retrieval lane and emits empty downstream
evidence/canonical ledgers without a canonical model call. Canonical failure
retains deterministic exact extractive evidence. Resume revalidates the full
hash chain rather than trusting file existence, and the six conventional export
files are published under `outputs/<experiment.id>/`.

## Observed model-generated decompositions

The two saved planning calls used DeepSeek V4 Flash through OpenRouter with the
`subnarrative_query_generator_v1` prompt. The model received only the official
narrative. Python admitted ordered generated or paraphrased subnarratives and
derived their stable IDs. The one-to-three BM25 suggestions remain validated,
inactive plan metadata; retrieval uses the full subnarrative text.

After the `# Prompt version: subnarrative_query_generator_v1` header, the exact
instruction body sent before the generated JSON schema was:

> **Generated subnarrative query task**
>
> Plan retrieval over a very large passage collection using only the supplied
> official topic narrative. Do not answer the topic. Return only the
> schema-constrained JSON plan.
>
> Rules:
>
> 1. Generate one to eight concise subnarratives that together preserve the
> narrative's requested subjects, relations, comparisons, constraints, and
> uncertainty. A subnarrative may paraphrase or synthesize the information need.
> 2. Give each subnarrative one to three plain BM25 queries. Queries must be useful
> lexical alternatives, not answers, and must not contain query operators or
> field syntax.
> 3. Use only the supplied narrative. Do not return identifiers or quote-location
> bookkeeping. Python derives stable identifiers from the returned order.

The request then appended the exact `subnarrative_queries_v1` JSON schema and
the official topic ID and narrative. It supplied no title, organizer
subnarrative, organizer nugget, qrel, or retrieved document.

| Topic | Provider / model | Tokens | Cost | Admitted plan |
| ---: | --- | ---: | ---: | --- |
| 224 | Ionstream / DeepSeek V4 Flash | 809 | `$0.00016870` | 7 subnarratives / 21 inactive suggestions |
| 300 | Ionstream / DeepSeek V4 Flash | 698 | `$0.00011886` | 4 subnarratives / 12 inactive suggestions |
| **Total** | **two calls** | **1,507** | **`$0.00028756`** | **no replay calls** |

The original receipts recorded `semantic_error` because the validator treated
ordinary lower-case “and” and “or” as Boolean operators. The saved bodies were
revalidated after the narrow fix (`0c74b45`) without another model call.
Response SHA-256 values are
`76b3c79ae89fcaea28f580b2ec935815d154974ea6b78cb1ce72aaf6f295fbf6`
for Topic 224 and
`e266dfe1a4ebe5ffcea5d58d146e7e731dd785ca01627cac192a73abf441dd88`
for Topic 300.

### Topic 224 actual output

**Official narrative sent to the model**

> I want to understand why people immigrate or become refugees, the challenges
> they face, and how laws and different groups shape immigration policies.
> Additionally, I'm interested in how various countries and religions view
> immigrants, and what options migrant workers have to improve their lives.

| Generated or paraphrased subnarrative | BM25 query 1 | BM25 query 2 | BM25 query 3 |
| --- | --- | --- | --- |
| Reasons why people immigrate or become refugees | reasons for immigration | why people become refugees | causes of migration |
| Challenges faced by immigrants and refugees | challenges immigrants face | refugee difficulties | obstacles for migrants |
| How laws shape immigration policies | immigration laws | policies affecting immigration | legal framework for migration |
| Influence of different groups on immigration policy | groups influence immigration policy | interest groups immigration | advocacy immigration reform |
| Views of various countries on immigrants | country attitudes toward immigrants | national perspectives on immigration | how countries view refugees |
| Religious perspectives on immigrants and refugees | religion and immigration | religious views on refugees | faith based immigration attitudes |
| Options for migrant workers to improve their lives | migrant worker opportunities | improving life for migrant workers | migrant labor rights |

This separates motives, challenges, law, organized influence, national views,
religious views, and worker options. In the two-topic retrieval check, the full
subnarrative sentences were substantially stronger than the first short query
suggestions for this topic.

### Topic 300 actual output

**Official narrative sent to the model**

> I'm interested in learning about effective strategies to prevent and reduce
> global warming and climate change, including specific actions that can help
> regions like Antarctica. I’d also like to know what global measures can be
> taken and how the economic costs of addressing global warming compare to just
> dealing with its impacts.

| Generated or paraphrased subnarrative | BM25 query 1 | BM25 query 2 | BM25 query 3 |
| --- | --- | --- | --- |
| Effective strategies to prevent and reduce global warming and climate change | strategies to prevent global warming | effective climate change mitigation actions | ways to reduce global warming |
| Specific actions that can help regions like Antarctica | Antarctica climate change prevention actions | protecting Antarctica from global warming | specific measures for Antarctica climate |
| Global measures to address climate change | global measures to combat climate change | international climate change policies | worldwide actions to reduce global warming |
| Comparison of economic costs of addressing global warming versus dealing with its impacts | economic costs of mitigating global warming vs impacts | cost of climate change action versus inaction | comparing costs of prevention and adaptation to climate change |

This separates mitigation, Antarctica-specific action, global policy, and the
requested cost comparison. Its generated arms did not beat the original
narrative at the fixed 1,000-document budget. These two outputs are a plausible
decomposition smoke test, not proof of faithfulness, granularity, or retrieval
gain.

## Retrieval and extractive-evidence findings

At a fixed 1,000-document ranked budget, full subnarrative text improved
known-relevant recall for Topic 224 but lost to the original narrative for Topic
300. The qrels are incomplete pooled LLM judgments, so unjudged documents are
unknown rather than irrelevant.

| Topic | Retrieval arm | R@100 | R@500 | R@1000 | Full-pool recall / docs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 224 | Original narrative | 13.5% | 21.0% | 26.6% | 26.6% / 1,000 |
| 224 | Subnarrative text | 6.0% | 21.9% | **35.8%** | 60.3% / 6,298 |
| 224 | First short suggestion | 2.5% | 6.6% | 10.1% | 27.3% / 6,668 |
| 300 | Original narrative | 11.7% | 17.8% | **21.6%** | 21.6% / 1,000 |
| 300 | Subnarrative text | 4.4% | 13.1% | 17.6% | 28.3% / 3,907 |
| 300 | First short suggestion | 2.8% | 11.0% | 18.1% | 33.5% / 3,959 |

Independent native top-100 lanes also showed coverage beyond the original
lane. The generated-lane unions contained 662 documents for Topic 224 and 393
for Topic 300, with only 36 and 3 overlapping the respective original top 100.
Combined unions covered 233/653 and 154/635 Codex-judged relevant documents.
The high unjudged rate and unequal total retrieval cost prevent a precision or
ranker conclusion.

The cache-only evidence pilot processed 1,216 union documents and emitted
443,325 exact candidates:

| Topic | Lanes | Union docs | Selected passages | Complete sentences | Candidates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 8 | 726 | 2,068 | 39,213 | 299,145 |
| 300 | 5 | 490 | 1,511 | 32,995 | 144,180 |

Every candidate passed schema, stable-ID, finite-score, canonical-order,
document/subnarrative-hash, exact character/byte-span, source recovery, and
context-separation checks. Exact provenance worked, but high-ranked candidates
still included fragments such as `groups.`, questions, headings, and malformed
source concatenation. Source-exact evidence is a provenance safeguard, not
proof of relevance, entailment, novelty, or usefulness.

## Canonical pilot and quality boundary

The saved development run selected 120 evidence clusters per subnarrative and
attempted eleven one-shot DeepSeek calls. Ten completed; one exceeded the
120-second deadline and used exact extractive fallback. The completed calls used
173,722 prompt tokens and 6,069 completion tokens for recorded cost
`$0.024034016`.

An initially over-strict rule rejected any claim citing two sentences from the
same document. Offline revalidation removed that rule because the intended
contract limits supporting documents, not sentence count. Ten saved responses
were then admitted with zero additional hosted calls. The remaining timeout
fallback is visibly weaker because it retains raw selected sentences, including
possible fragments and questions.

Representative successful content included a migrant-worker claim grounded in
sources enumerating minimum wage, safe working conditions, and equal legal
treatment, and a climate claim grounded in a source quantifying about 0.5°C
avoided warming by 2050. Representative failures changed the actor, stitched
together a relationship absent from either cited sentence, or emitted
content-free statements. Alias validation proves that a source was cited; it
cannot prove that the claim follows from the source.

The saved run used the exact `canonical_nuggetizer_v1` instruction:

> Use only the supplied exact evidence quotations to write canonical claims
> for the one supplied subnarrative. Return zero to twenty atomic,
> independently checkable claims that directly respond to that subnarrative.
> Fewer claims are better than padding. You may paraphrase evidence into a
> claim, but do not copy, alter, or return evidence text. Return only each claim
> and one to three supplied evidence aliases. Every cited alias must directly
> support the claim, and cited aliases must come from distinct documents. Do
> not infer unsupported details or invent identifiers.

Each request also contained the official narrative as context, exactly one
generated subnarrative, and the selected exact evidence quotations under local
aliases. It contained no title, organizer subnarrative, organizer nugget, or
qrel. The later `canonical_nuggetizer_v2` contract removed only the
distinct-document requirement; the model was not rerun for that correction.

The 120-cluster run is **development-tuned**. Its budget was chosen after
inspection of an organizer-nugget embedding proxy, which is indirect evaluation
leakage. The supported configuration instead freezes a 40-cluster first-pass
policy. Neither budget is established as a production optimum.

## Cross-encoder score calibration conclusion

**Provenance.** The [pinned reranker config](../configs/rag25_bm25_mixedbread_rerank_v1.yaml)
fixes `mixedbread-ai/mxbai-rerank-base-v2` at revision
[`3ea9d4dffa7d12a4f366be8e275c349de9fc9865`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2/tree/3ea9d4dffa7d12a4f366be8e275c349de9fc9865),
Sentence-Transformers `5.6.0`, and `raw_logits`; the [comparison manifest](experiments/bm25_mixedbread_config_comparison_v1/manifest.yaml)
binds those settings to the retained score artifacts. The immutable model
[`LogitScore` configuration](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2/blob/3ea9d4dffa7d12a4f366be8e275c349de9fc9865/1_LogitScore/config.json)
selects token IDs 16 and 15, whose immutable
[`vocabulary`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2/blob/3ea9d4dffa7d12a4f366be8e275c349de9fc9865/vocab.json)
maps them to `"1"` and `"0"`; the pinned
[Sentence-Transformers implementation](https://github.com/UKPLab/sentence-transformers/blob/v5.6.0/sentence_transformers/cross_encoder/modules/logit_score.py)
defines their difference, and the local [score-cache runner](../code/trec_rag/rerank_score_cache.py)
passes an identity activation for `raw_logits`.

The pinned value is therefore the unactivated difference

```text
score(query, text) = logit("1") - logit("0")
```

It is an unbounded ranking score, not a calibrated probability. Upstream
evidence in this repository is ranking evidence: the [retained comparison
record](experiments/bm25_mixedbread_config_comparison_v1/manifest.yaml) is a
22-topic development nDCG@10 evaluation. It records no labeled,
topic-split sentence/subnarrative calibration set or validation procedure, so
it does not establish a portable absolute cutoff across generated
subnarratives or the local sentence-level distribution. A sigmoid changes the
numeric range, not the missing calibration evidence.

The actionable policy is therefore:

- rank raw scores only within each subnarrative;
- apply question, heading, fragment, and boilerplate filters independently of
  CE relevance;
- if a hard decision is required, label representative sentence/subnarrative
  pairs and split train, validation, and test by topic;
- tune the operating rule against an explicit objective, report per-topic
  uncertainty on held-out topics, and version it with the complete scorer and
  data identity;
- compare a query-local normalized relevance term with cosine novelty before
  revising MMR, because reciprocal rank quickly becomes much smaller than the
  novelty term.

## Next held-out evaluation

Promotion remains blocked on a precommitted, topic-split experiment:

1. Freeze the config, model identities, prompts, evidence policy, and evaluation
   plan without organizer data.
2. Run all stages and seal outputs before opening organizer nuggets or qrels.
3. Measure candidate recall, document-budget efficiency, claim-to-reference
   coverage, exact-evidence entailment, atomicity, redundancy, and failure rate.
4. Add blinded human review of decompositions, evidence, model claims, and
   extractive fallbacks.
5. Treat unjudged documents as unknown and keep retrieval cost visible.

No metric from the current two-topic, development-tuned pilot should be used as
unbiased promotion evidence.

## Independently reviewed facet references (evaluation-only)

The exact strings below are copied from the frozen
[`reports/experiments/all_topic_tethered_facet_validation_v1/facet_manifest.json`](experiments/all_topic_tethered_facet_validation_v1/facet_manifest.json).
Its accompanying [independent content review](experiments/all_topic_tethered_facet_validation_v1/facet_review.md)
states that the reviewer inspected the narratives, prompt, proposed manifest,
and validation design without opening qrels, nuggets, retrieved documents,
candidate results, metrics, model output, or web sources. The tracked record
does not identify the reviewer as human, so these are independently reviewed
references rather than human labels. They are useful for judging granularity
after a model run, but are not prompt material, runtime inputs, or a target
vocabulary for new topics.

| Topic | Representative reviewed facets |
| ---: | --- |
| 58 | `nuclear energy pros`; `nuclear energy accident risks Chernobyl`; `nuclear energy comparison fusion other energy forms` |
| 144 | `banks Silicon Valley Bank why fail`; `banks credit unions compare regulation`; `banks credit unions roles economic development` |
| 213 | `Korean War origins`; `Korean War why US involved Cold War strategy`; `Korean War different US presidents perceived conflict` |
| 224 | `people why immigrate become refugees`; `immigrants refugees challenges face`; `migrant workers options improve lives` |

## Historical evidence: not the current contract

Earlier prompt and output work remains useful evidence but does not describe the
supported runner:

| Attempt | Preserved result | Interpretation |
| --- | --- | --- |
| GPT-OSS V1 / `sparse_query_planner_v4` | Advisor summaries scored Topics 144, 213, and 224 at 8/12, 5/12, and 6/12. | None met the 10/12 gate; exact bodies are unavailable locally, so the diagnoses below are advisor summaries rather than quotations. |
| GPT-OSS V2 / `sparse_query_planner_v5` | HTTP 400: `Grammar error: Unimplemented keys: ["uniqueItems"]`. | Schema/transport failure, not model-quality evidence. |
| GPT-OSS V2.1 / `sparse_query_planner_v6` | A synthetic sensor request returned valid JSON but failed semantic admission. | Exact original-only fallback worked. |
| Earlier LiteLLM/Qwen | 22 topics, 165 generated facets, 18,700 retrieved rows, and 17,587 fused rows. | Recorded weighted-RRF nDCG@10 was `0.4294133095` versus `0.4139611685` for original BM25; both Recall@100 values were `0.1080879457`. Exact facet strings were not recovered and must not be reconstructed. |

The V1 advisor diagnosed concrete behavior: Topic 144 narrowed a general
bank-failure request to Silicon Valley Bank, dropped the credit-union side of
one query, and overbundled safety, trust, services, and regulation. Topic 213
collapsed origins, ending, US involvement, and Cold War rationale into one
mega-facet and introduced the absent date `1950-1953`. Topic 224 collapsed
immigrant and refugee scope and combined country and religious views. Two
failed outputs added further evidence: Topic 407 substituted `surged` for the
source word `soared`, built a non-contiguous source quote, and injected
candidate causes; Topic 515 stopped at the 6,000-token completion limit and,
even after a diagnostic closing bracket, still failed the frozen renderer's
minimum-content rule.

The V2.1 immutable raw response preserves this exact surviving facet object:

```json
{"facet_id":"fac_1","coverage_refs":["cov_1","cov_2"],"expansion_terms":[{"term":"mountain","relation":"common_variant","anchor_refs":["mountain_streams"]},{"term":"streams","relation":"common_variant","anchor_refs":["mountain_streams"]},{"term":"urban","relation":"common_variant","anchor_refs":["urban_canals"]}]}
```

The HTTP response was valid and ended with `stop`, but the plan omitted the
required global entity/topic anchor, so deterministic admission rejected it.
This is preserved transport and fallback evidence, not a quality success.

An **earlier stricter anchor contract** required generated facets to be directly
grounded in source wording and used deterministic rendering. It admitted three
of five plans and rejected two under those old rules. That historical contract
is superseded by the current model-generated/paraphrased subnarrative boundary,
so its Topic 224/300 rejections do not describe the present responses. Its
unequal candidate pools and mixed recall changes showed discovery potential but
did not justify global fusion or a final selector.

## Provenance

The two-topic evidence remains in the ignored local experiment tree. Relative
paths below are from the repository root; hashes identify the exact files:

- `outputs/subnarrative-bm25-224-300-v1/recall-eval-v1/bm25_metrics.json`
  SHA-256:
  `b5efea5ba352b89d002387b17ea3c2bc30f78b156ee86ef1b5c89a8f191f4947`;
- `outputs/subnarrative-bm25-224-300-v1/recall-eval-v1/crossencoder_metrics.json`
  SHA-256:
  `45a6cec37a4b7a64cb656e1f79aab36d98331150958d2f2b55eb07b84555c21a`;
- `outputs/subnarrative-bm25-224-300-v1/original-subnarrative-top100-v1/metrics.json`
  SHA-256:
  `0b176e8dbf4319c93138c76d8861afd35a3271d12aa6fdd5b86f64882e51ac69`;
- `outputs/subnarrative-bm25-224-300-v1/extractive-candidates-v1/candidates.jsonl`
  SHA-256:
  `326d9f3dcab8545350882cb09345ec28c2eb51fccb69e9b6365556f544f4b240`;
- `outputs/subnarrative-bm25-224-300-v1/canonical-nuggets-v1/canonical-nuggets-revalidated.jsonl`
  SHA-256:
  `e1b8e4141932b75ac622314e83617071bad88b435eb425a7cca85bb2018ae088`.

Recovered historical claims are traceable to immutable Git sources:

- `fc9261e752b2554fa43773c6a7f5633a60ebb3b1:reports/experiments/query_planner_gpt_oss_smoke_v1/advisor_review.md`;
- `fc9261e752b2554fa43773c6a7f5633a60ebb3b1:reports/experiments/query_planner_v2_synthetic_smoke_001/incident_evidence.json`;
- `1a879ac64a33969ff12cbae1299eea2e3a1b454b:reports/experiments/query_planner_v2_1_synthetic_smoke_001/artifacts/`;
- `b856e9773f0136b877281d4b57d494e79717d345:reports/experiments/rag25_bm25_original_litellm_facets_weighted_rrf_v1/`.

The local recovery audit found no exact V1 model bodies or early-Qwen facet
strings. Historical organizer material is not a runtime dependency.

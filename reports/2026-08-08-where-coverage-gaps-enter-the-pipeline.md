# Where Nugget Coverage Gaps Enter the Retrieval Pipeline

**Date:** 2026-08-08

**Status:** measured, 22 judged dev topics; root cause located in source code and confirmed against run artifacts

**Issue:** [#49 — Diagnose where nugget coverage gaps enter the retrieval pipeline](https://github.com/npatta01/trec_rag_2026/issues/49)

## Provenance and reproducibility boundary

This is a sanitized development diagnostic, not an official submission result.

The measured population is the private 2025 non-agentic all-22-topic replay at
`outputs/nonagentic-rag25-dev-all22-replay-p4-20260807`. It is the only run in
that directory carrying a `retrieval_nugget_coverage_v2/` evaluation; the three
sibling `nonagentic-rag25-dev-*` directories are single-topic runs with no
coverage evaluation.

Coverage judgments come from the frozen evaluator in
`code/trec_rag/retrieval_nugget_coverage.py`, whose per-topic artifacts
(`input.json`, `plan.json`, `judgments.json`, `report.json`, `manifest.json`)
are hash-bound. Topic IDs and obligation counts appear here with the
repository owner's explicit authorization. Narratives, retrieved passages,
generated claims, provider responses and topic-level evaluation content remain
private and are not reproduced.

A small number of content-free heading strings from the public MS MARCO v2.1
corpus appear as mechanism evidence. They carry no evaluation content and are
necessary to show the defect.

## Baseline

| | |
| --- | ---: |
| Topics | 22 |
| Obligations | 89 |
| Full | 72 |
| Partial | **17** |
| Unsupported | **0** |

The 17 partial obligations fall in ten topics: 58, 84, 144, 213, 219, 224, 273,
407, 477, 515. Topic 407 is the worst at `required_coverage` 0.50 with all three
of its obligations partial.

## Headline

**The gaps are overwhelmingly not retrieval failures.** In the great majority of
cases the missing information was retrieved, reached the selected documents, and
was then destroyed before the canonicalizer ever saw it. Exactly one obligation —
topic 58's "Bison Energy" — is a clean corpus miss, and it looks like an entity
absent from MS MARCO v2.1 rather than a query problem: the decomposition produced
a correctly targeted lane for it.

## The mechanism chain

### 1. Documents were segmented by line, not by paragraph

`_source_spans` in `code/trec_rag/facet_evidence.py` iterated
`source.splitlines()` and emitted one span per non-blank line, while every
downstream name called the result a paragraph.

Measured over topic 407's 100 selected documents (9,534 non-blank lines):

| | |
| --- | ---: |
| Lines under 40 characters | **60%** |
| Lines continuing mid-sentence onto the next line | **43%** |
| Median "paragraph" length | **11 characters** |

One decision produced two distinct defects. Standalone short lines — headings,
nav labels, list captions — became evidence candidates in their own right. And
hard-wrapped sentences were cut at the wrap, yielding truncated candidates.

### 2. Short strings outscore real sentences

Candidates are ranked by `mixedbread-ai/mxbai-rerank-base-v2`, which applies no
length normalization. A four-word heading that lexically mirrors the
subnarrative query therefore outscores every real sentence in the document.

Junk rate by position among the 4,960 selected cluster representatives:

| MMR position | junk rate |
| --- | ---: |
| 1 | **53.2%** |
| 2–4 | 47.0% |
| 5–8 | 31.0% |
| 9–16 | 20.6% |
| 33–40 | 13.6% |

Junk concentrates at the **top**, and correlates *positively* with reranker
score (47.5% junk among logits 11–12; minimum 14.7% at logits 6–7). This is the
opposite of the intuition that low-ranked candidates are the weak ones, and it
means the defect was costing the most valuable slots.

Run-wide, 19–21% of the 4,960 selected representatives are non-assertive text,
ranging from 11.9% to 42.5% by topic.

Subnarratives backing a **partial** obligation average 15.8% fragment slots
versus 8.3% for those backing a **full** one. Permutation test p = 0.0004;
p = 0.015 excluding topic 407; p = 0.030 shuffling labels within topic. The
effect survives both robustness checks but is weaker than the naive test
suggests.

### 3. The selection window sees 13% of the pool

`SelectionPolicy.__post_init__` (`code/trec_rag/facet_evidence.py`) sets

```python
limit = 10 * max(self.budgets) if self.precluster_limit is None else self.precluster_limit
```

and `competition_retrieval.py` constructs `SelectionPolicy(budgets=(selected_budget,))`
without ever passing `precluster_limit`. There is no YAML key for it. So
`nuggets.evidence_budget_per_subnarrative: 40` silently means a 400-candidate
window over a pool averaging 3,089 candidates per subnarrative — **13.1%**.

Verified independently: the maximum exact-group rank appearing among the kept 40
clusters is 380–408 across all subnarratives of all ten partial topics, a hard
cut at 400.

Because roughly a fifth of that window is consumed by non-assertive text, the
effective window on real evidence is smaller still.

### 4. Canonicalization is instructed to discard qualifiers

The claim-writing prompt is not ours. `nuggetizer_adapter.py` calls the
installed `nuggetizer` package's `_create_nugget_prompt`, which instructs:

> "Update the list of atomic nuggets of information **(1-12 words)**... Make
> sure there is **no redundant information**. Ensure the updated nugget list has
> at most `{creator_max_nuggets}` nuggets (can be less), **keeping only the most
> vital ones**."

A 1–12 word budget mechanically strips source attribution, population,
geography and date range — precisely what the judge recorded as missing on
topics 515, 407 and 273.

Separately, `MAX_CANONICAL_NUGGETS = 20` is hard-capped in two places
(`canonical_nuggets.py:45`, `facet_pilot_config.py:228`). It binds in **65 of
124 subnarratives (52%)**, and on those the canonicalizer leaves roughly 20 of
its 40 offered clusters uncited. Upstream's own default is 30.

## What this does not explain

Four obligations are unreachable by any downstream change, and no selection or
canonicalization fix recovers them.

Candidates for subnarrative *N* are mined only from documents in lane
`facet:subnarrative-N`'s top-100 passages, and within those only from paragraphs
intersecting the winning passage chunk (`facet_evidence.py`). Documents reached
only by the `original` lane are never mined, and most of a long article is never
mined.

That is what kills topic 144 (17 of 22 candidate quotes lie outside any mined
region), topic 213/f005 (12 of 15), topic 84 (best evidence at pool rank 727)
and the telehealth half of topic 219 (all ten relevant sentences at ranks
512–1546).

Two further obligations are not pipeline defects at all: topic 273's "how many
continents fit inside Africa" requires arithmetic no document states, and topic
58's "Bison Energy" appears absent from the corpus.

## Corrections made during the investigation

Recorded because they changed the conclusion, and because the first two answers
were wrong in ways that would have misdirected the fix.

- An initial regex-containment pass concluded that 9 of 17 gaps were "lost in
  the funnel". Reading the matched sentences showed the counts were inflated by
  incidental keyword hits; that specific number is not defensible.
- An initial reading blamed the 40-cluster budget. The binding constraint is the
  400-candidate precluster window, which is the *same dial* — raising
  `evidence_budget_per_subnarrative` raises both.
- Fragment filtering alone was measured to extend reach only from rank 400 to
  about 448, while most missing evidence sits at ranks 512–1846. Filtering
  improves slot quality; it is not the reach lever.
- Canonicalization runs at `temperature: 0` and is deterministic. The coverage
  **judge** is the stochastic stage (`seed: 0`, no temperature). Its noise floor
  has never been measured, so with only 17 partial obligations a two-item change
  is inside plausible noise.

## Judge stability

The coverage judge is a hosted call with `seed: 0` and no `temperature`, so its
run-to-run stability had to be measured before any coverage change could be
read as signal.

Method: the judge was re-run five times per topic on unchanged input, reusing
each topic's frozen `plan.json` so obligation IDs stay comparable and planner
variation cannot be mistaken for judge variation. 110 judge-only calls, no
planner calls, no failures.

| | |
| --- | ---: |
| Observations per obligation | 6 (baseline + 5 repeats) |
| Obligations that ever changed label | **1 of 89 (1.1%)** |
| Of the 17 baseline-partial obligations | **0** |
| Mean label flips per repeat | 0.6 of 89 (0.7%) |
| Topics whose `required_coverage` varied | 1 of 22 |

The single unstable obligation is topic 515 `f004-o001`, which is a genuine
coin flip (three `full`, three `partial`) and is **not** one of the 17 partials.
Its topic's `required_coverage` swings 0.750–0.875 as a result.

With zero flips across 85 partial-obligation observations, the rule of three
puts the 95% upper bound on the per-observation flip rate at 3.5%, i.e. at most
0.6 expected flips among the 17 partials per run. **An improvement of one
obligation is already outside the noise band, and two is comfortably so.**

This is a stronger result than expected and removes the main obstacle to
interpreting a re-scored run.

## Unknowns

- Whether correcting segmentation changes any coverage label is untested.
- Whether the larger paragraphs produced by the fix partially recover the
  "never mined" class above is plausible but unverified.

## Disposition

Segmentation has been rewritten onto a shared spaCy-backed segmenter
(`code/trec_rag/chunking.py`) used by both the competition and DeepAgent lanes,
with paragraphs derived from sentence boundaries and headings joined forward
into `exact_sentence_pair` candidates rather than admitted alone.

On topic 407's 100 selected documents:

| | Line-based | spaCy |
| --- | ---: | ---: |
| Units | 9,534 | 11,269 |
| Median length | 11 chars | **118 chars** |
| Under 40 characters | 60.4% | **13.2%** |
| Headings dropped | — | **0 of 1,040** |

An adversarial review of the change raised 16 findings, of which 9 survived
refutation; all 9 are fixed with regression tests. Two were reachable defects
that would have corrupted a re-scored run rather than merely degrading it: the
`attribute_ruler` pipeline component was excluded, leaving `token.pos_` empty so
that completeness silently collapsed to a punctuation test; and the pair
admission rule was duplicated in the builder and both validators, so every
heading-led pair the builder emitted would have been rejected, aborting the
candidate stage. The rule now lives once, in `pair_is_admissible`.

Test status: 2,700 passing. The single failure,
`test_ragdoll_resolves_to_the_repository_pinned_submodule`, is an artifact of
running the main checkout's virtualenv against worktree code and fails on
unmodified `master` too.

The change invalidates the sentence-score cache. Re-scoring all 383,078 cached
candidates measures at roughly 67 minutes on the local Radeon 8060S with
length-sorted batches of 64; BM25 and passage-score caches are unaffected.
Segmentation itself adds about 15 minutes across 22 topics.

The precluster window and the 20-nugget cap are **not** yet changed, and the
judge's noise floor is still unmeasured.

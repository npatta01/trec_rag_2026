# Coverage-gap diagnosis and post-retrieval fix plan

Date: 2026-08-08
Issue: [#49](https://github.com/npatta01/trec_rag_2026/issues/49)
Baseline run: 2025 non-agentic all-22-topic replay (`nonagentic-rag25-dev-all22-replay-p4-*`)

This document records mechanisms, code locations and measurements only. Narratives,
passages, generated claims and provider responses stay in the private run directory.

## Baseline

89 obligations across 22 topics: 72 full, 17 partial, 0 unsupported. Ten topics carry
the 17 partial obligations (58, 84, 144, 213, 219, 224, 273, 407, 477, 515).

## Constraint

Re-running BM25 or passage cross-encoder scoring is off the table before the deadline.
Everything below runs downstream of cached retrieval.

**This is cheaper than it looks.** All 383,078 sentence candidates across the 22 topics
already carry a persisted `sentence_score` in `<topic>/records.sqlite3` (`candidate`
table, 0 NULLs). `cache-operation-receipt.json` records a zero-miss replay
(`retrieval 6/0`, `passage_scores 27,956/0`, `sentence_scores 13,231/0`). Re-running
selection and canonicalization costs **zero GPU and zero BM25**; canonicalization for the
whole 22-topic run cost $0.053, so a re-run with a raised nugget cap is ~$0.11.

## Four verified mechanisms

### M1 — One dial silently controls two limits

`SelectionPolicy.__post_init__` (`code/trec_rag/facet_evidence.py:1190`):

```python
limit = 10 * max(self.budgets) if self.precluster_limit is None else self.precluster_limit
```

`competition_retrieval.py:1875` constructs `SelectionPolicy(budgets=(selected_budget,))`
and never passes `precluster_limit`. There is no YAML key for it — `facet_pilot_config.py`
rejects unknown `nuggets` keys.

So `evidence_budget_per_subnarrative: 40` means a **400-candidate precluster window**.

| precluster_limit | share of the ~3,089-candidate pool considered |
| ---: | ---: |
| 400 (current) | **13.1%** |
| 1000 | 32.9% |
| 2000 | 65.7% |
| 4000 | 99.7% |

Verified independently: the maximum exact-group rank appearing among the kept 40 clusters
is 380–408 across all subnarratives of all 10 partial topics — a hard cut at 400.

### M2 — Non-sentences win selection slots

`_sentences_in_paragraph` (`facet_evidence.py:527`) splits a paragraph on terminal `.?!`.
A paragraph with **no** terminal punctuation — heading, nav link, list item, table cell —
becomes one whole "sentence" and is admitted as an `exact_sentence` candidate. No
minimum-length or minimum-token gate exists anywhere.

Junk concentrates at the **top** of the ranking, because `mxbai-rerank-base-v2` applies no
length dilution: a 4-word heading that lexically mirrors the subnarrative query outscores
every real sentence in the document.

| MMR position | junk rate |
| --- | ---: |
| 1 | **53.2%** |
| 2–4 | 47.0% |
| 5–8 | 31.0% |
| 9–16 | 20.6% |
| 33–40 | 13.6% |

Run-wide, 19–21% of the 4,960 selected cluster representatives are non-assertive
(range 11.9%–42.5% by topic). A separate class of junk: bare `ecn1_<64 hex>` document-hash
tokens admitted as sentence candidates (184 in one topic-58 subnarrative pool alone).

Subnarratives backing a partial obligation average 15.8% fragment slots vs 8.3% for full
ones (permutation p=0.0004; p=0.015 excluding topic 407; p=0.030 shuffling within topic).

### M3 — The sentence splitter breaks on "U.S."

`_ABBREVIATIONS` (`facet_evidence.py:27`) omits `u.s`, `u.k`, `u.n`, `d.c`. `_is_terminal`
protects single letters, so the first period in `U.S.` is safe, but the second is treated
as a sentence boundary. Worst observed damage: 8 of 40 representatives truncated in one
topic-213 subnarrative.

### M4 — The canonicalization prompt is upstream and hostile to qualifiers

`nuggetizer_adapter.py:53` calls the installed `nuggetizer` package's
`_create_nugget_prompt`. Upstream text (`nuggetizer/models/nuggetizer.py:86`):

> "Update the list of atomic nuggets of information **(1-12 words)**... Make sure there is
> **no redundant information**. Ensure the updated nugget list has at most
> `{creator_max_nuggets}` nuggets (can be less), **keeping only the most vital ones**."

A 1–12 word budget mechanically strips source attribution, population, geography and date
range — exactly what the judge flagged as missing on topics 515, 407 and 273.

Separately, `MAX_CANONICAL_NUGGETS = 20` is hard-capped in **two** places
(`canonical_nuggets.py:45`, `facet_pilot_config.py:228`). It binds in **65 of 124
subnarratives (52%)**, and on those the canonicalizer leaves ~20 of its 40 offered clusters
completely uncited. Upstream's own default is 30.

Our controllable seam is the appended `grounded_contract` user message
(`nuggetizer_adapter.py:130`), not the upstream prompt.

## What is *not* fixable downstream

Candidates for subnarrative N are mined only from documents in lane
`facet:subnarrative-N`'s top-100 passages, and within those only from paragraphs
intersecting the winning passage chunk (`facet_evidence.py:872-905`). Documents reached
only by the `original` lane are never mined, and most of a long article is never mined.

No filtering, clustering, budget or prompt change reaches text that was never turned into
a scored candidate. This is what kills topic 144 (17 of 22 candidate quotes), topic
213/f005 (12 of 15), topic 84 (best evidence at rank 727) and the telehealth half of
topic 219 (all 10 sentences at ranks 512–1546).

## Ranked fixes

### F1 — Raise the budget dial (primary reach lever)

`configs/rag26_competition_retrieval_v2.yaml`: `nuggets.evidence_budget_per_subnarrative`
40 → 200, which raises the precluster window 400 → 2000 (13.1% → 65.7% of the pool) and
the MMR budget 40 → 200.

Must be paired with F2 or the extra clusters go uncited.

Effort: config only. Cost: local MiniLM similarity re-run + hosted canonicalization.

### F2 — Raise the nugget cap and preserve qualifiers

- `MAX_CANONICAL_NUGGETS` 20 → 40 (`canonical_nuggets.py:45`) and the matching
  `maximum=20` at `facet_pilot_config.py:228`.
- Extend the `grounded_contract` message (`nuggetizer_adapter.py:130`) to require source
  attribution and numeric qualifiers (population, geography, period, rate definition) be
  retained in `claim_text`.

Targets the confirmed canonicalization losses on 407/f001, 273/f001, 515/f003-o002.

Effort: small code change. Cost: ~$0.11 hosted, no GPU.

### F3 — Candidate admissibility gate

In `extract_document_candidates` (`facet_evidence.py:872`), before candidates are written:

```
drop if text matches ^ecn1_[0-9a-f]{40,}$
drop if token_count < 6
drop if no terminal .?! and token_count < 10
drop if span starts immediately after '\n' with a lowercase initial   # line-wrap clip
```

Apply it here, not at `select_subnarrative_candidates` (`facet_evidence.py:1411`) — that
seam does not backfill and would shrink the pool instead of deepening it.

Measured payoff on its own is modest: filtering short spans extends reach only from rank
400 to ~448 (topic 407: ~505). Its real value is slot *quality* and a cleaner generation
handoff. Targeted value is high on topics 224, 407/f002, 407/f003, 477, 213/f006.

### F4 — Add `u.s`, `u.k`, `u.n`, `d.c` to `_ABBREVIATIONS`

One line, `facet_evidence.py:27`. Bumps `SENTENCE_SPLITTER_VERSION`.

### Not worth doing

Topics 84, 144, 213/f005 and the telehealth half of 219 — evidence was never mined into a
candidate. Also do not "fix" MMR relevance normalisation as a junk remedy: junk correlates
*positively* with reranker score, so rescaling relevance does not remove it.

## Blast radius

`code_commit` (`git rev-parse HEAD`) is embedded in every phase checkpoint
(`competition_retrieval.py:2185`, `:1897`; resume at `:2364`). Any code edit invalidates
all four phase checkpoints for all 22 topics. This is nearly free given the zero-miss cache
— reranker cache keys are `sha256(context ‖ query_sha ‖ text_sha)` with no commit or config
component (`rerank_score_cache.py:245`).

F3 and F4 change candidate identity and therefore change `records.sqlite3` contents; F1 and
F2 do not.

## Verification protocol

1. Measure the judge noise floor first: re-run the coverage judge k times on unchanged
   baseline input. The judge is the stochastic stage (`seed: 0`, no `temperature`);
   canonicalization is `temperature: 0` and deterministic. With only 17 partials, a 2-item
   move is inside plausible noise.
2. Reuse baseline `plan.json` bytes verbatim across all variants and abort if `plan_sha256`
   differs — obligation IDs are position-derived.
3. Record the full label vector per variant, not just the targeted obligation.

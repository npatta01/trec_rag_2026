# Full-union facet-aware RAG evidence ranker design

Date: 2026-07-14

## Decision

Stop treating a single cross-encoder score sort as the final solution. Preserve
the complete accepted candidate union and build a deterministic passage-level
ranker that selects a compact, nonredundant, facet-covering evidence portfolio
for downstream RAG.

All 8,114 topic-document rows remain eligible. There is no 100, 500, 1,000, or
other document-count cutoff. Candidate eligibility and generator context are
separate concerns: the former is unlimited within the frozen union, while the
latter is evaluated at 8,000, 16,000, and 32,000 evidence tokens per topic.

The current four topics contain roughly 46.8 million raw document tokens in
aggregate, or 11.7 million per topic. A generator cannot consume that complete
text directly, so passage extraction and evidence budgeting are required even
though no candidate document is discarded before ranking.

This is a post-qrels mechanism diagnostic on topics `219`, `72`, `300`, and `84`.
It may identify a better evidence-selection mechanism, but it is not fresh
confirmatory evidence. Protected topic IDs `144`, `213`, `224`, `407`, and `515`
remain hard-rejected at every boundary.

## Why a ranker is required

A cross-encoder is a **score provider**. Given one query and one passage, it can
estimate local semantic relevance and reduce vocabulary mismatch. Sorting those
scores alone cannot answer the multi-facet selection problem:

- a passage can be highly relevant to one facet but weakly resemble the complete
  narrative;
- many high-scoring passages can repeat the same facet;
- a generic document can mention many query words without directly supporting
  any requested claim;
- a long document can expose more scoring windows without being more useful;
- useful evidence must fit a finite generator context.

The evidence ranker therefore combines cross-encoder signals with explicit facet
coverage, original-query relevance, domain coherence, source-quality warnings,
redundancy, retrieval provenance, and passage-token cost.

This design does not train a learned listwise ranker. Four already exposed topics
are insufficient training evidence and would invite qrels overfitting. A learned
ranker becomes a separate option only after the deterministic policy is validated
on fresh topics.

## Scope and success criterion

The experiment asks one practical question:

> Does the complete original-plus-facet candidate union contain enough usable
> passage evidence to build a better RAG context than narrative-only or simple
> score-only selection?

Success means the recommended selector improves supported facet coverage and
direct-support quality without losing broad original-query evidence, at one or
more frozen evidence-token budgets. Document nDCG remains diagnostic; it is not
the sole promotion target.

The deliverable is a RAG-ready evidence packet and a rendered analytical report.
The repository currently implements only placeholder answer generation, so this
experiment does not claim that a generated answer is good. Substantive generation,
citation verification, and answer-quality comparison follow in a separate design
after the selector passes.

## Frozen inputs

The experiment consumes existing immutable artifacts only:

- the complete `U_accepted` union from the deep-facet pilot;
- original and accepted-facet retrieval provenance;
- the sealed RRF, GLOBAL, DUAL, cascade, and Mixedbread rankings;
- existing MiniLM facet-local, narrative, and common-query window scores;
- existing Mixedbread residual scores;
- the already opened projected development qrels for diagnostic evaluation.

The input population is the set of all 8,114 accepted topic-document rows. Every
document ID, text hash, stream rank, score identity, and source seal must be
preserved. No new retrieval request, endpoint call, paid call, dense primary
retrieval, or query rewrite occurs in the initial audit.

Before any new local inference, the implementation must report exact score
coverage by topic, document, passage, query family, and model. Missing scores are
reported as missing; they are never imputed from another query or silently
treated as low relevance.

## Passage representation

Use the existing deterministic tokenizer and passage-window contract where its
sealed provenance is available:

- pair maximum: 512 tokens;
- query maximum: 192 tokens;
- minimum passage budget: 256 tokens;
- overlap: 64 passage tokens;
- retain the final non-empty short window;
- preserve document ID, character span, token span, and text hash.

The ranker may select at most three passages from one document. This is a
diversity safeguard, not a document-length judgment. A document may supply more
than one passage when the passages support different facets or provide
non-overlapping evidence. Multiple overlapping windows cannot inflate coverage.

Document length is recorded only as processing cost and for diagnostics. It is
not a positive or negative relevance feature.

## Score completion policy

### Audit first

Build a coverage matrix for every passage-query identity needed by the three
selection arms. Reuse exact cached scores only when model revision, tokenizer,
query hash, document hash, span, and aggregation identity match.

### Bounded local completion

If the audit shows that cached scores cannot give every accepted document a fair
opportunity, preflight a complete local MiniLM scoring pass over the missing
topic-passage pairs for:

- the unchanged full narrative; and
- every accepted facet belonging to that topic.

Use the already materialized
`cross-encoder/ms-marco-MiniLM-L6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`. There is no model download, network
request, paid inference, or change to candidate eligibility. Preflight freezes
the exact passage-pair count, cache hits, cache misses, estimated runtime, device,
and disk footprint before inference. Results are written as resumable,
create-only shards with complete score provenance.

MiniLM is preferred for exhaustive passage scoring because it is the cheaper
locally available semantic ranker. The existing Mixedbread scores remain a
stronger but incomplete global diagnostic signal; Mixedbread is not expanded to
the entire union unless a later preflight shows MiniLM passage quality is the
remaining bottleneck.

Raw scores from different queries are never compared. For each query independently,
convert passage scores to deterministic average-rank percentiles in `[0,1]`.

## Selection arms

Every arm begins with the same complete candidate union and the same passage
windows. Every arm emits a deterministic ordered evidence packet at 8k, 16k,
and 32k tokens, plus an auditable continuation over all remaining passages.

### NARRATIVE: broad-relevance control

Rank passages by the unchanged narrative MiniLM percentile. Apply the same
near-duplicate suppression and three-passages-per-document cap as the recommended
arm. This tests whether semantic reranking alone is enough.

### FACET-SCORE: decomposition control

Cycle through facets in frozen manifest order and take the best unselected
passage for each facet. Continue round-robin by descending within-facet percentile.
Apply domain coherence, duplicate suppression, and the per-document cap. This
tests facet-local scoring without a coverage-aware portfolio objective.

### PORTFOLIO: recommended evidence ranker

Select passages in two deterministic stages.

#### Stage 1: credible coverage floor

For each accepted facet in manifest order, select its strongest available
passage that:

- contains the topic subject anchor or an audited morphological variant;
- expresses the facet relation or an audited ordinary paraphrase;
- does not match a frozen wrong-domain rule;
- is not a near duplicate of already selected evidence; and
- does not exceed the three-passages-per-document cap.

If no passage qualifies, record the facet as unsupported. Do not fill it with a
generic or wrong-domain passage merely to claim coverage.

#### Stage 2: marginal evidence utility

Fill the remaining token budget greedily. For passage `p` and selected set `S`,
the ranking features are:

- `N(p)`: narrative MiniLM percentile;
- `F_i(p)`: facet-`i` MiniLM percentile;
- `C_i(S)`: strongest nonredundant support already selected for facet `i`;
- `M(p|S) = max_i F_i(p) * (1 - C_i(S))`: marginal uncovered-facet value;
- `A(p)`: deterministic subject, relation, population, and domain coherence;
- `R(p)`: normalized rank provenance from the original or facet stream;
- `Q(p)`: deterministic source-quality warning, never a rejection by itself;
- `D(p,S)`: maximum near-duplicate similarity to selected evidence;
- `T(p)`: passage tokens divided by the current evidence-token budget.

Use this frozen utility:

```text
U(p|S) = 0.40*M(p|S)
       + 0.25*N(p)
       + 0.15*A(p)
       + 0.10*R(p)
       + 0.10*max_i(F_i(p))
       - 0.20*D(p,S)
       - 0.05*Q(p)
       - 0.05*T(p)
```

Feature values are bounded to `[0,1]`. The source-quality term is `1` only when
the frozen warning patterns match. A warning can demote a passage but cannot
make it ineligible without independent coherence or direct-support failure.

The selected facet for a tied marginal value resolves by manifest order. Final
ties resolve by original-family presence, best family-balanced RRF rank, document
ID, and passage start offset. The weights are frozen before evaluation and are
not tuned on these four topics.

### MULTI-FACET sensitivity

Run one offline sensitivity arm that credits a passage for every facet whose
within-facet percentile is at least `0.90` and whose coherence checks pass. It
uses the PORTFOLIO selector otherwise. This tests whether genuine multi-facet
documents improve token efficiency.

Raw counts of facet streams retrieving a document never count as coverage.
Coverage requires passage-level semantic score plus coherence evidence.

## Redundancy and source-quality controls

Near-duplicate detection uses the existing lowercased, stemmed, non-stopword
token sets and token-set Jaccard:

- below `0.80`: no redundancy penalty;
- `0.80` to `1.00`: linear penalty from `0` to `1`;
- exact overlapping spans from the same document: ineligible after the first.

Frozen quality-warning families include essay-writing advice, dictionaries,
Scrabble/word-game pages, homework/templates, generic process pages, and known
wrong-domain populations such as canine results for human aggression or pet
health pages for human health questions. These patterns support diagnostics and
demotion. They cannot reject a passage alone.

## Evaluation

### Qrels-backed document diagnostics

At each token budget report:

- unique cited document count;
- binary and graded qrel gain captured;
- relevant documents introduced only by facets;
- original-query relevant evidence retained;
- judged rate;
- qrel grades `2`, `3`, and `4` separately;
- evidence gain per 10,000 tokens.

Also retain nDCG at document ranks 10, 100, 500, and 1,000 for continuity with
earlier experiments. These are ranking diagnostics, not the evidence selector's
promotion target.

### Blind passage audit

Create 72 arm-facet audit slots: one slot for each of the 24 accepted facets from
each of NARRATIVE, FACET-SCORE, and PORTFOLIO at the 16k-token budget. Fill a slot
with the first selected passage that the arm claims supports that facet. If the
arm makes no credible claim, preserve an explicit `unsupported` sentinel rather
than substituting an out-of-packet passage. Randomize the passage-bearing slots
deterministically and hide the arm; sentinel slots count as unsupported but are
not shown as passages. Review labels are:

- direct support;
- topical mention only;
- wrong domain or population;
- usable source quality;
- redundant with another selected passage;
- facet or facets supported.

The audit reports inter-reviewer agreement when a second reviewer is available.
It never exposes qrel labels in the review packet.

### Primary evidence metrics

- supported-facet coverage: fraction of accepted facets with direct support;
- worst-facet support: minimum direct-support count across facets;
- direct-support precision among selected passages;
- usable-source rate;
- wrong-domain rate;
- redundancy rate;
- multi-facet evidence rate;
- evidence tokens per supported facet;
- original-evidence retention;
- novel-facet-document retention.

## Promotion and failure diagnosis

Promote PORTFOLIO over NARRATIVE and FACET-SCORE only if all are true at either
16k or 32k tokens:

- supported-facet coverage improves by at least 10 percentage points over both;
- direct-support precision does not fall by more than 5 percentage points;
- original-query graded evidence retention is at least 90% of NARRATIVE;
- wrong-domain rate is no worse than either control;
- redundancy is lower than FACET-SCORE; and
- at least three of four topics do not regress in supported-facet coverage.

The failure reason determines the next action:

- **Facet candidates contain no supporting passage:** return to query repair or
  retrieval controls for that facet; reranking cannot create missing evidence.
- **Supporting passages exist but MiniLM scores them poorly:** compare a bounded
  identical-passage Mixedbread pass or another already available local ranker.
- **Scores are good but the selector misses passages:** revise the portfolio
  policy, not retrieval.
- **Evidence packets are good but generated answers are poor:** work on answer
  planning, citation-first generation, and post-generation verification.
- **Only exposed-qrels metrics improve:** do not generalize; validate once on
  fresh, non-protected topics.

No result is described merely as “B failed.” The report must identify which of
candidate discovery, passage scoring, portfolio selection, or downstream
generation is responsible.

## Output artifacts

Create a versioned output directory containing:

- input bindings and source seals;
- passage inventory and text hashes;
- score-coverage audit;
- optional local MiniLM preflight, shards, and scoring receipt;
- NARRATIVE, FACET-SCORE, PORTFOLIO, and MULTI-FACET packets at every budget;
- an all-passages continuation preserving complete eligibility;
- qrels-backed diagnostics;
- blinded review packet and completed labels;
- a mechanical promotion decision;
- advisor review of the frozen method and findings.

Update the standalone HTML report under
`reports/experiments/deep_facet_candidate_pilot_v1/`. It must explain, in plain
language:

- why the prior Mixedbread sort changed top-100 membership;
- why complete-union recall and evidence selection are different problems;
- why a cross-encoder score is not by itself a multi-facet ranker;
- how many facets receive direct support at each token budget;
- whether multi-facet documents save evidence tokens;
- whether failures come from retrieval, scoring, or selection;
- what remains before an end-to-end generated RAG answer can be judged.

The report must be rendered and verified at desktop and mobile viewports. It must
remain self-contained and accessible without an expiring localhost server.

## Safety and reproducibility

- Reject protected IDs in input reads, joins, scoring, evaluation, and reports.
- Do not access any new topic or qrels data in this post-qrels diagnostic.
- Use exact hashes for every query, document, passage, score, packet, and metric.
- Never compare raw BM25 or cross-encoder scores across query boundaries.
- Never discard a document from the complete accepted union.
- No recursive retrieval or agentic query repair occurs.
- No model download, hosted inference, paid call, or external request occurs.
- Local inference, if needed, uses a frozen preflight and resumable shards.
- Unrelated working-tree files are neither modified nor committed.

## Test requirements

Tests must verify:

- the input population contains exactly the sealed 8,114 topic-document rows;
- every input document remains represented in the continuation;
- protected topics fail before any source or score access;
- cached score identities match exact model, query, text, and span hashes;
- raw scores never cross query boundaries;
- score percentiles and ties are deterministic;
- each arm receives identical documents and passage windows;
- token budgets are enforced without a document-count eligibility limit;
- no document contributes more than three selected passages;
- overlap and near-duplicate behavior is deterministic;
- source-quality warnings cannot reject alone;
- multi-facet credit requires passage-level score and coherence evidence;
- complete-union eligibility is invariant across all arms;
- promotion decisions reproduce exactly from saved artifacts;
- report metrics and claims reproduce from sealed outputs;
- desktop and mobile report rendering pass accessibility checks.

## Deferred end-to-end RAG work

The current `code/trec_rag/generation.py` intentionally produces placeholder
output and cannot establish answer quality. After the evidence ranker passes, a
separate design will compare substantive citation-first generation over identical
evidence packets, including claim support, facet coverage, citation validity,
usefulness, and hallucination checks. This separation prevents a weak generator
from obscuring whether candidate discovery and evidence selection were fixed.

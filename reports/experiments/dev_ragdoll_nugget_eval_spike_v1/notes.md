# Dev RAGDoll nugget evaluation spike

This is the first answer-quality number produced in this repository. Everything before
it scored rankings against qrels; nothing scored generated text, and the 1,255 released
gold nuggets were referenced by no code at all.

**It validates wiring, not system quality.** Read the comparability section before
quoting any figure.

## What ran

1. `trec_rag.dev_rag_inputs` read the archived Pyserini responses for topics 58 and 213
   and emitted a six-column TREC run plus an organizer-shaped document JSONL.
2. `trec_rag.competition_rag` generated answers with `openai/gpt-5.6-sol`, unchanged from
   the merged competition path.
3. `trec_rag.ragdoll_io` derived a RAGDoll answers file and reshaped the gold nuggets.
4. `ragdoll nuggetizer eval` assigned the fixed gold nuggets and computed metrics.

## Results

| qid | strict_vital | strict_all | vital | all | nuggets | vital | words |
|---|---|---|---|---|---|---|---|
| 58 | 0.5714 | 0.5385 | 0.6857 | 0.6667 | 39 | 35 | 652 |
| 213 | 0.6296 | 0.7000 | 0.7778 | 0.8300 | 50 | 27 | 696 |
| **run** | **0.6005** | **0.6192** | **0.7317** | **0.7483** | 89 | 62 | — |

Label distribution: support 56 (62.9%), partial_support 23 (25.8%), not_support 10
(11.2%), failed 0. Judge cost: 9 calls, $0.1345 provider-reported.

## Integration gates

All passed, and each guards a failure that would otherwise look like a plausible score:

- **Non-empty qid join.** RAGDoll resolves ids from top-level `qid`/`topic_id`/`query_id`
  only and *skips* unmatched answers with a warning. Feeding submission rows directly
  yields empty metrics rather than an error, because our id lives in
  `metadata.narrative_id`. It cannot be moved: the organizer validator requires exactly
  three root keys, so a sidecar answers file is derived instead.
- **Non-degenerate labels.** An empty context short-circuits every nugget to
  `not_support` without any model call, which is indistinguishable from a bad answer.
- **`failed_count` zero.** Parse failures surface as `failed`, not as a low score.

## Reading the not_support cases

The ten misses are specific facts, not paraphrase failures: "Bison Energy includes
nuclear energy in their portfolios", "The Korean War became an election issue in 1952",
"Integration of black and white troops advanced during the Korean War". These look like
genuine coverage gaps from a BM25-only candidate pool. One gold nugget on topic 58,
"fusion energy is the current nuclear energy source", is questionable on its face and is
a reminder that gold nuggets are themselves model-assisted artifacts.

## Comparability

Do not compare these to published TREC numbers or to each other across configurations.

- **Two topics.** The per-topic spread is already 0.057 on strict_vital.
- **Raw BM25 ordering, no reranking.** The development archives predate the retriever
  provenance sidecar (`retrievers.py` raises `unverified cache missing provenance
  sidecar`), so they cannot be replayed through `trec_rag.pipeline`. Synthesizing
  sidecars would defeat that guard, so the archives were read directly instead. A real
  measurement needs a verified retrieval export.
- **Automated assignment scores above NIST manual assignment**, and `strict_vital` is the
  most fragile metric under full automation.
- The judge (`openai-codex/gpt-5.5`) differs from the generator (`openai/gpt-5.6-sol`),
  so no model graded its own writing.

## Next

Scale to all 22 development topics over a verified retrieval run, then add citation
support (`ragdoll support`) and arena battles (`ragdoll arena compare-all`, the primary
2026 metric).

## Citation support: answer-object granularity caps every citation at Partial

Running `ragdoll support judge` over the same two answers produced **31 out of 31
`Partial Support`** judgments — no Full Support, no No Support. Weighted precision and
recall are therefore 0.5 across the board and `hard_precision` is **0.0**.

That uniformity is the signature of a structural problem, not a scoring result, so it was
checked rather than reported.

The judge output is genuine (`raw_output` is the literal string `Partial Support`, not a
parse fallback). The cause is visible in the statements: the generator emits fused,
multi-claim answer objects behind a heading prefix, roughly 65 words and 3-4 sentences
each:

> How it works and is used: Commercial nuclear electricity currently relies mainly on
> fission. Neutrons split uranium nuclei in a controlled chain reaction...

No single cited document fully supports a compound like that, so each one caps at Partial.

### Controlled probe

Holding the cited document, the claim, and the judge model fixed, and changing only the
answer-object granularity:

| statement | label |
|---|---|
| `How it works and is used: Commercial nuclear electricity currently relies mainly on fission. Neutrons split...` | `PS` |
| `Commercial nuclear electricity currently relies mainly on fission.` | `FS` |

The structure alone moves the judgment. This matches the task reference, which asks
systems to "break prose into sentence-level answer objects" and permits headings only as
their own objects.

### Implication

The generator currently forfeits the entire Full-versus-Partial difference on citation
support for reasons unrelated to retrieval or evidence quality. Emitting one atomic claim
per answer object is a prompt-level change. It should be measured, not assumed: more
objects means more citations to get right, and the 1,024-word cap and the strict
profile's 1-3 citations per object both still apply.

Caveat: reference text was resolved offline from the local documents file and truncated
to the same 1,000-word view the generator saw, rather than resolved from the index. That
is stricter than organizer behaviour and can only understate support.

## A/B: fused versus atomic answer objects

Everything held fixed except the prompt profile — same topics, same archived retrieval, same
generator model and reasoning effort, same judge.

| | A `default` | B `atomic_claims` |
|---|---|---|
| answer objects (58 / 213) | 9 / 8 | 23 / 18 |
| words (58 / 213) | 652 / 696 | 587 / 444 |
| strict_vital | 0.6005 | **0.6333** |
| strict_all | 0.6192 | **0.6349** |
| vital | 0.7317 | 0.7325 |
| all | **0.7483** | 0.7454 |
| weighted precision, first citation | 0.500 | **0.709** |
| hard precision | 0.000 | **0.4395** |
| support labels | 0 FS / 31 PS / 0 NS | 26 FS / 45 PS / 2 NS |

**Atomic objects win on citation support decisively and cost nothing on nugget coverage.**
Weighted precision rises 0.21 and hard precision goes from zero to 0.44, because a
single-claim object can actually be fully supported by one cited document. Nugget scores
move within noise: strict_vital +0.033, `all` -0.003.

Arm B also spends *fewer* words (587 and 444 against 652 and 696), so it buys the support
improvement while leaving more of the 1,024-word budget unused. That headroom is the next
thing to exploit, since nugget scoring is pure recall with no length penalty.

### How much to trust this

The support difference is structural, understood, and independently demonstrated by the
controlled probe above, so it is unlikely to be an artifact. The nugget difference is
two topics wide and should be treated as noise until it is run over all 22.

Arm B also produced more references (14 per topic against 8 and 9), which means more
citations to get right. Two citations landed at No Support in arm B against none in arm A —
a small precision risk that grows with object count and is worth watching at scale.


> **Superseded.** The A/B section above rests on two topics. A four-topic paired run with a
> single judge (`dev4_paired_prompt_profile_v1`) reverses the coverage conclusion: atomic
> objects *lose* 0.065 strict_vital across all four topics while still winning citation
> support. Treat the support finding here as sound and the coverage finding as withdrawn.

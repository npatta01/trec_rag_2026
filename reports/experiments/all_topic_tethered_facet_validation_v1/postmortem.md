# Topic 31/300 retrieval-failure postmortem

Decision: retain canonical RRF. The bounded Topic 300 replay is recovery evidence, not a promotion result.

## Evidence boundary

Relevance means UMBRELA grade 2 or higher. Every unjudged (unknown) document remains separate from judged-below-2 evidence and is never described as nonrelevant. All replays are offline over frozen accepted-union candidates and features.

## RRF depth-20 capture diagnostic

This postmortem-only diagnostic derives from the authenticated RRF ranking and pinned qrels; it does not alter the sealed evaluation depths. RRF captures 316 / 12,984 known-relevant documents (2.43%) and 2,492 / 84,560 graded-gain units (2.95%) at depth 20.

## Topic 31 cutoff mechanics

Primary DUAL changes known-relevant capture by -7 at depth 1,000.

| Movement | Total | Known relevant (grade ≥2) | Judged below 2 | Unjudged (unknown) |
|---|---:|---:|---:|---:|
| Outgoing | 322 | 33 | 2 | 287 |
| Incoming | 322 | 26 | 0 | 296 |

## Topic 300 cutoff mechanics

Primary DUAL changes known-relevant capture by -3 at depth 1,000.

| Movement | Total | Known relevant (grade ≥2) | Judged below 2 | Unjudged (unknown) |
|---|---:|---:|---:|---:|
| Outgoing | 281 | 16 | 37 | 228 |
| Incoming | 281 | 13 | 11 | 257 |

### Facet retrieval-stream attribution (not DUAL selection coverage)

These memberships come from authenticated accepted-union retrieval provenance and the tracked facet manifest. They identify which query stream retrieved a moved candidate; they do not claim that the stream caused relevance or the rank change. A candidate may have more than one stream membership, while original-only candidates have none.

| Movement | Facet stream | Tracked query formulation | Memberships | Known relevant | Judged below 2 | Unjudged (unknown) | Per-stream rank buckets |
|---|---|---|---:|---:|---:|---:|---|
| Outgoing | `300-antarctica` (antarctica) | climate change actions help Antarctica | 26 | 0 | 1 | 25 | 1-50: 10, 101-150: 3, 51-100: 13 |
| Outgoing | `300-economic-costs` (economic costs) | global warming economic costs addressing compared dealing impacts | 32 | 0 | 0 | 32 | 1-50: 11, 101-150: 3, 51-100: 18 |
| Outgoing | `300-global-measures` (global measures) | global measures address global warming climate change | 30 | 3 | 27 | 0 | 1-50: 7, 51-100: 23 |
| Outgoing | `300-strategies` (strategies) | global warming climate change effective strategies prevent reduce | 14 | 5 | 8 | 1 | 1-50: 6, 101-150: 1, 51-100: 7 |
| Incoming | `300-antarctica` (antarctica) | climate change actions help Antarctica | 35 | 0 | 7 | 28 | 101-150: 22, 151-200: 13 |
| Incoming | `300-economic-costs` (economic costs) | global warming economic costs addressing compared dealing impacts | 32 | 2 | 0 | 30 | 101-150: 17, 151-200: 15 |
| Incoming | `300-global-measures` (global measures) | global measures address global warming climate change | 51 | 0 | 0 | 51 | 101-150: 28, 151-200: 23 |
| Incoming | `300-strategies` (strategies) | global warming climate change effective strategies prevent reduce | 54 | 1 | 0 | 53 | 101-150: 27, 151-200: 27 |

### DUAL selection-coverage attribution (not retrieval-stream provenance)

The greedy DUAL audit records which facet coverage state was credited when a candidate was selected. It does not identify the query stream that retrieved that candidate.

| Movement | DUAL coverage facet | Selected candidates |
|---|---|---:|
| Outgoing | `300-strategies` | 281 |
| Incoming | `300-antarctica` | 2 |
| Incoming | `300-economic-costs` | 2 |
| Incoming | `300-global-measures` | 1 |
| Incoming | `300-strategies` | 276 |

### Judgment-pool dependent facet-tail yield

These are known-relevant yields within the existing judgment pool; unjudged candidates remain unknown.

| Per-facet retrieval rank | Known-relevant yield |
|---|---:|
| 1-50 | 36.18% |
| 51-100 | 29.29% |
| 101-150 | 6.00% |
| 151-200 | 8.00% |

### Bounded offline recovery replay

Protected RRF prefix: 100

The `facet_rank_cap_100` arm changes known-relevant capture by +2 at depth 1,000. Original-stream candidates stay eligible; facet-only candidates require a facet retrieval rank within the cap. Deferred candidates return in canonical RRF order.

Complete accepted-union permutation checks: `RRF`=pass, `canonical_primary`=pass, `facet_rank_cap_100`=pass, `no_narrative_score`=pass.

The no-narrative-score arm changes known-relevant capture by +0. It is a post-hoc diagnostic and is not promotion-eligible. Method: fixed objective minus 0.15*N; no greedy replay from `RRF100-STATIC-DUAL-NR` (fixed zero inherited from diagnostic base arm); every remaining DUAL weight is unchanged.

## Authenticated provenance

- Ranking root: `6f4c35899f90c1d60324caf24bf8834d3482c5e8b9e785f8316ab0eec55fc305`
- Qrels SHA-256: `42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37`
- Retrieval root: `f2191c295f600d0243f4dcd2b1dc23a05b9fa8a7d9a4d6258983a0d3c9433aeb`
- Scoring root: `4f67107770b1cd35598adc75beeeafe205ba984f201cedf4b505c4d20515e990`
- Retrieval, inference, download, model-load, hosted, and paid calls during replay: 0.

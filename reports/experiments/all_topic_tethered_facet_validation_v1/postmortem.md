# Topic 31/300 retrieval-failure postmortem

Decision: retain canonical RRF. The bounded Topic 300 replay is recovery evidence, not a promotion result.

## Evidence boundary

Relevance means UMBRELA grade 2 or higher. Every unjudged (unknown) document remains separate from judged-below-2 evidence and is never described as nonrelevant. All replays are offline over frozen accepted-union candidates and features.

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
- Retrieval root: `f2191c295f600d0243f4dcd2b1dc23a05b9fa8a7d9a4d6258983a0d3c9433aeb`
- Scoring root: `4f67107770b1cd35598adc75beeeafe205ba984f201cedf4b505c4d20515e990`
- Retrieval, inference, download, model-load, hosted, and paid calls during replay: 0.

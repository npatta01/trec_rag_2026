# GPT-5.6 Sol coverage-repair experiment

This development experiment rebuilt the evidence ledger from the nugget-blind map-stage pool over
all 1,478 passages. It applied semantic deduplication, complexity-weighted facet quotas, up to three
covering citations per claim, pre-generation support repair, and post-generation audit repair.

| Run | Strict | Partial | Vital strict | Claims | Words | Excluded |
|---|---:|---:|---:|---:|---:|---:|
| Previous controlled GPT-5.6 Sol | 0.460 | 0.520 | 0.444 | 35 | 789 | 7 |
| Coverage-repair GPT-5.6 Sol | **0.540** | **0.630** | **0.630** | 48 | 941 | 0 |

## Change

- Strict: +0.080
- Partial credit: +0.110
- Vital strict: +0.185

Organizer nuggets were not loaded until the repaired submission was frozen. The quota design was
informed by earlier Topic 213 development error analysis, so these metrics are development-tuned and
must not be treated as held-out generalization.

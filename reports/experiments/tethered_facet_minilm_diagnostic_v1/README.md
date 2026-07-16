# Tethered facet MiniLM diagnostic v1

This directory is the create-only destination for the accessible diagnostic
report. The experiment is a **post-qrels diagnostic** over four fixed topics. It
performs **no new retrieval** and is **not production validation**.

The report consumes authenticated Task 1/2 receipts, the sealed Task 3
two-basket freeze, and bounded Task 4 metrics and diagnostic examples. To
authenticate qrels-derived grades and novel-document diagnostics, it also reads
only the exact historically anchored qrels projection already bound by Task 4.
It does not read original qrels or perform a new evaluation.

## Reproduce

Set paths to the already-created, local artifacts first:

```bash
TASK1=outputs/rag25_tethered_facet_minilm_v1/preflight_v1
TASK3=outputs/rag25_tethered_facet_minilm_v1/freeze_v1
TASK4=outputs/rag25_tethered_facet_minilm_v1/evaluation_v1
REPORT=reports/experiments/tethered_facet_minilm_diagnostic_v1
```

Task 1 is an offline tokenizer-only preflight:

```bash
.venv/bin/python -m trec_rag.tethered_facet_minilm_score preflight \
  --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json \
  --phase1 outputs/rag25_deep_facet_candidates_v1/phase1_v1 \
  --gate outputs/rag25_deep_facet_candidates_v1/gate_v1 \
  --output "$TASK1"
```

Task 2 is the only stage that uses the ROCm Python helper:

```bash
.venv/bin/python-rocm -m trec_rag.tethered_facet_minilm_score score \
  --preflight "$TASK1/preflight.json"
```

Task 3 loads the exact verified prior RRF/DUAL rankings, accepted facet
candidates and scores, and Task 2 scores, then creates the two-basket freeze:

```bash
.venv/bin/python -m trec_rag.tethered_facet_two_basket freeze \
  --deep-root outputs/rag25_deep_facet_candidates_v1 \
  --tethered "$TASK1" \
  --output "$TASK3"
```

Verify the new sealed freeze without changing it:

```bash
.venv/bin/python -m trec_rag.tethered_facet_two_basket verify \
  --freeze "$TASK3"
```

Task 4 replays the projection-only evaluation after the Task 3 verifier passes:

```bash
.venv/bin/python -m trec_rag.tethered_facet_evaluate \
  --freeze "$TASK3" \
  --prior-evaluation outputs/rag25_deep_facet_candidates_v1/evaluation_v1 \
  --output "$TASK4"
```

Finally, build the standalone HTML, JSON companions, and SQLite data source.
The destination may contain this tracked README and no generated report files:

```bash
.venv/bin/python -m trec_rag.build_tethered_facet_report \
  --task1-receipt "$TASK1/preflight.json" \
  --task2-receipt "$TASK1/scoring_receipt.json" \
  --task3-freeze "$TASK3" \
  --task4-evaluation "$TASK4" \
  --output "$REPORT"
```

All stages are create-only. Use a new output directory for a replay.

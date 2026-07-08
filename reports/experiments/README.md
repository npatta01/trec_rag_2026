# Experiment Trackers

This directory keeps durable experiment records.

- `<experiment_id>/`: source-of-truth folder for one experiment. Simple
  single-run folders should include `manifest.yaml`, `config.yaml`,
  `metrics.json`, `topic_scores.csv`, and `notes.md`. Multi-system comparison
  folders may use clearly named tables such as `system_scores.csv` or
  `topic_system_scores.csv` instead of the single-run `topic_scores.csv`.
- `runs.csv`: derived index with one row per experiment run, including
  run-level configuration, provenance, artifact paths, and aggregate metrics.
- `topic_scores.csv`: derived index with one row per
  `(experiment_id, topic_id)` metric record, plus an `overall` row for each
  experiment.

Use stable `experiment_id` values when possible. Future variants such as
different hit counts, query decomposition, dense retrieval, or reranking should
append new rows rather than overwrite old results.

Regenerate the global indexes from the per-experiment folders with:

```bash
PYTHONPATH=code uv run --with pyyaml python -m trec_rag.experiment_records \
  --experiments-dir reports/experiments
```

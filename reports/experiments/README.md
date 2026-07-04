# Experiment Trackers

This directory keeps append-friendly experiment records.

- `runs.csv`: one row per experiment run, with run-level configuration,
  provenance, artifact paths, and aggregate metrics.
- `topic_scores.csv`: one row per `(experiment_id, topic_id)` metric record,
  plus an `overall` row for each experiment.

Use stable `experiment_id` values when possible. Future variants such as
different hit counts, query decomposition, dense retrieval, or reranking should
append new rows rather than overwrite old results.

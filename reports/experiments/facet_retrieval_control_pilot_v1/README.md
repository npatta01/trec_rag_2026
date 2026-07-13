# Facet retrieval-control pilot v1

This directory freezes a four-stream, qrels-blind candidate-generation control
manifest derived from the exact prior R1 sparse-relevance manifest. Producing
the manifest makes no network call and does not alter the prior pilot.

The pilot covers `200/f07a`, `225/f02`, `225/f04`, and `707/f02`. Each stream
has the cached R1 baseline `B0` and three external reweighted arms: `W0`, `W1`,
and `W2`. The external-attempt ceiling is 12, retrieval depth is 100, and the
minimum interval between request starts is ten seconds.

Admission gates require the byte-pinned R1 source, its exact four baseline
queries, the fixed protected-topic set (`144`, `213`, `224`, `407`, `515`), no
new normalized query vocabulary, exact ordered BM25 settings, and exactly 12
external arms. The manifest generator is create-only; later retrieval work
must preserve these gates and freeze results before opening qrels.

No external retrieval, qrels access, model inference, reranking, or paid call
is performed while building this manifest.

## Portable report source

`code/trec_rag/build_facet_retrieval_control_report.py` builds the canonical
Data Analytics report source. It does not hand-author HTML and does not run
retrieval, read qrels, or call a reranker. Cross-encoder not run.

The builder consumes only saved, verified artifacts:

- this directory's `manifest.json`;
- the control freeze's `freeze.json`, `candidate_streams.json`, and
  `inspection.json`; and
- the post-freeze evaluation's `stream_evaluation.json`, `selection.json`,
  `evaluation.json`, and `decision.json`.

It validates the exact four-stream boundary, the 25 pre-qrels ranking
alternatives, the self-contained v2 candidate snapshot bindings, the four
original narrative hashes, the selected ranking references, O/F0/R1/R2
metrics, and the saved mechanical decision. Missing, corrupt, or inconsistent
decision inputs fail closed; the report never fills in a value. The bounded
snapshot includes only report-ready aggregate rows and twelve representative
selected-arm results. Raw candidate lists, qrels rows, endpoint details,
credentials, and protected topics are excluded.

The ready-state gate also recomputes the canonical Task 5 selection from every
saved arm record, including eligibility and the B0/W0/W1/W2 tie-break. Each
aggregate system metric must equal the exact arithmetic mean of its four
per-topic rows before the Task 5 decision rule is recomputed. The manifest and
freeze are authenticated against their loaded file bytes; the freeze root,
candidate-stream manifest, inspection, and all four evaluation JSON files are
checked using their canonical self-hash or bound hash contract. Manifest,
freeze, prior-freeze, selected-ranking, and qrels SHA-256 bindings must agree
across the loaded artifacts.

For each topic, the saved selection hash and the system-evaluation hash must
equal the exact semantic SHA-256 on its authenticated freeze ranking record.
The candidate snapshot is also closed in both directions: the hash of the
loaded `candidates.jsonl` bytes must equal both `candidate_file_sha256` in
`candidate_streams.json` and `candidates_sha256` in `freeze.json`. Candidate
bytes are used only for validation and never enter the portable report source.

Task 5 does not save an independent qrels attestation outside those four
self-hashed evaluation files. Because Task 6 must not reopen qrels, it can
prove that the saved qrels SHA-256 and name agree across intact evaluation
artifacts, but it cannot independently rehash an all-files-consistent qrels
claim. Task 7 must retain the already verified Task 5 evaluation directory as
the trust boundary rather than treating the report builder as a second qrels
verifier.

Task 7 creates `artifact.json` with:

```bash
.venv/bin/python -m trec_rag.build_facet_retrieval_control_report \
  --manifest reports/experiments/facet_retrieval_control_pilot_v1/manifest.json \
  --freeze-dir outputs/rag25_facet_retrieval_control_v1/freeze_v1 \
  --evaluation-dir outputs/rag25_facet_retrieval_control_v1/evaluation_v1 \
  --output reports/experiments/facet_retrieval_control_pilot_v1/artifact.json
```

The output is the exact top-level contract accepted by `validate_artifact`:
`surface`, `manifest`, `snapshot`, and `sources`, with `surface: report`. It
contains native markdown, chart, and table blocks plus canonical repo-relative
source provenance. Task 7 then runs the packaged Data Analytics delivery tool,
which revalidates the same artifact and generates the self-contained
`report.html` without a second report runtime:

```bash
node <DATA_ANALYTICS_PLUGIN_ROOT>/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input reports/experiments/facet_retrieval_control_pilot_v1/artifact.json \
  --output reports/experiments/facet_retrieval_control_pilot_v1/report.html
```

Task 6 deliberately does not create either final file. Task 7 owns generation,
portable packaging, and rendered desktop/mobile verification.

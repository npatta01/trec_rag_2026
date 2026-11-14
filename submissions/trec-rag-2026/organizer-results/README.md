# Organizer-released scores — TREC RAG 2026

The official per-narrative scores the track organizers released for team
NP Labs. These are **received artifacts**: do not edit, reformat, or
recompute them. Treat them like the accepted submission files beside them.

## Provenance

Both archives were emailed by Nour Jedidi (University of Waterloo), cc Lingwei
Gu, Jimmy Lin, and Ian Soboroff (NIST), to the team address:

| Received | Subject | Contents |
| --- | --- | --- |
| 2026-11-02 | TREC 2026 RAG Track — Retrieval Results | `retrieval/` |
| 2026-11-03 | TREC 2026 RAG Track — RAG Task Results | `rag/` |

Each `README.md` here is the organizers' own, unmodified. The `compute_*.py`
files are the organizers' reference scorers, kept for provenance only — they
cannot run, because the qrels and nuggets they need were not released.

## What is in the files

Filter `narrative-id` to `all` for one summary row per run; every other row is
a single narrative.

### `rag/` — 42 evaluated narratives

| File | Narratives | Columns |
| --- | --- | --- |
| `rag-bt-scores-batch1.tsv` | 17 | `bt_score`, `ci_lower`, `ci_upper` — `all` rows only |
| `rag-bt-scores-batch2.tsv` | 25 | same |
| `rag-sxs-counts-batch1.tsv` | 17 | `wins`, `ties`, `losses` |
| `rag-sxs-counts-batch2.tsv` | 25 | same |
| `rag-rubric-scores-batch1-human.tsv` | 17 | `vital_strict`, `all_strict` |
| `rag-nugget-scores-batch2-manual.tsv` | 25 | same |
| `rag-nugget-scores-batch2-automatic.tsv` | 25 | same |

Bradley–Terry ratings are fitted per batch across all judgments, so they have
no per-narrative value. `batch1-topics.txt` and `batch2-topics.txt` list the
exact narratives in each batch.

### `retrieval/` — 82 evaluated narratives

| File | Narratives | Columns |
| --- | --- | --- |
| `retrieval-document-level-scores.tsv` | 82 | `set_P`, `set_recall`, `num_ret` |
| `retrieval-nugget-level-scores-manual.tsv` | 25 | `gain`, `effort` |
| `retrieval-nugget-level-scores-automatic.tsv` | 25 | same |

## Boundaries that matter when quoting these numbers

- **Each collection is its own evaluation.** Different narrative sets,
  different items, different scales. A Batch-1 rubric score and a Batch-2
  nugget score are not comparable, and neither averages into the other.
- **`_min`, `_max`, and `_median` describe the whole field** — all 114
  evaluated runs across 27 teams, each run weighted equally. They are not
  NP Labs statistics, and they are not per-team.
- **Only full support earns credit.** Partial support scores zero in every
  strict measure.
- **`effort` and `num_ret` are cost, not quality.** Read them next to gain,
  never as a score.
- Nuggets and qrels were **not** released and are not in this directory.

## Where these numbers already appear

- [`reports/2026-organizer-score-explorer.html`](../../../reports/2026-organizer-score-explorer.html)
  renders every per-narrative row, filterable. Regenerate it with
  `python code/trec_rag/organizer_score_explorer.py .` after any change here.
- [`reports/2026-trec-rag-paper.tex`](../../../reports/2026-trec-rag-paper.tex)
  quotes the `all` rows in its two results tables. Those values were verified
  against these files and match exactly.

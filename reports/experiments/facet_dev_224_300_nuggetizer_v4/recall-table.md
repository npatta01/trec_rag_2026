# Live retrieval recall

Relevance is qrel score `>= 2`. Documents absent from a qrel are unknown, not
nonrelevant. The qrels are pooled LLM judgments and are not exhaustive
corpus-wide gold judgments.

`O` is original BM25; `S` is subnarrative-only round-robin; `U` is the
matched-arm `O ∪ S`; `A` is all-lane round-robin under one fixed total budget.

| Topic | Judge | Relevant | Depth | O | S | U | U docs | A |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 224 | Codex | 653 | 100 | 13.5% | 5.8% | 17.6% | 189 | 7.0% |
| 224 | Codex | 653 | 500 | 21.0% | 20.8% | 35.4% | 932 | 28.5% |
| 224 | Codex | 653 | 1000 | 26.6% | 33.5% | 48.5% | 1,823 | 39.2% |
| 224 | Ministral | 972 | 100 | 10.1% | 4.4% | 13.4% | 189 | 5.2% |
| 224 | Ministral | 972 | 500 | 16.0% | 16.8% | 28.2% | 932 | 22.2% |
| 224 | Ministral | 972 | 1000 | 21.0% | 26.1% | 38.4% | 1,823 | 31.1% |
| 224 | Qwen | 753 | 100 | 11.7% | 5.3% | 15.5% | 189 | 6.2% |
| 224 | Qwen | 753 | 500 | 18.6% | 20.2% | 33.3% | 932 | 26.3% |
| 224 | Qwen | 753 | 1000 | 24.2% | 31.7% | 45.3% | 1,823 | 37.2% |
| 300 | Codex | 635 | 100 | 11.7% | 3.9% | 15.6% | 198 | 5.2% |
| 300 | Codex | 635 | 500 | 17.8% | 12.3% | 28.3% | 973 | 23.9% |
| 300 | Codex | 635 | 1000 | 21.6% | 15.4% | 32.8% | 1,932 | 26.6% |
| 300 | Ministral | 1106 | 100 | 8.8% | 3.4% | 12.0% | 198 | 4.2% |
| 300 | Ministral | 1106 | 500 | 13.1% | 11.7% | 23.1% | 973 | 20.0% |
| 300 | Ministral | 1106 | 1000 | 15.9% | 14.1% | 26.7% | 1,932 | 22.4% |
| 300 | Qwen | 686 | 100 | 8.9% | 4.5% | 13.3% | 198 | 5.0% |
| 300 | Qwen | 686 | 500 | 13.8% | 14.1% | 25.9% | 973 | 22.3% |
| 300 | Qwen | 686 | 1000 | 16.8% | 17.6% | 29.9% | 1,932 | 25.5% |

## Full lane unions

| Topic | Judge | Original | Subnarratives | All union | Union documents |
|---:|---|---:|---:|---:|---:|
| 224 | Codex | 26.6% | 52.2% | 58.7% | 5,309 |
| 224 | Ministral | 21.0% | 41.6% | 47.1% | 5,309 |
| 224 | Qwen | 24.2% | 49.1% | 55.2% | 5,309 |
| 300 | Codex | 21.6% | 26.0% | 36.9% | 4,741 |
| 300 | Ministral | 15.9% | 23.1% | 31.4% | 4,741 |
| 300 | Qwen | 16.8% | 28.6% | 35.6% | 4,741 |

## Cross-encoder-selected 100-document pool

| Topic | Judge | Recall | Judged | Unjudged |
|---:|---|---:|---:|---:|
| 224 | Codex | 7.8% (51/653) | 55 | 45 |
| 224 | Ministral | 5.7% (55/972) | 55 | 45 |
| 224 | Qwen | 6.9% (52/753) | 55 | 45 |
| 300 | Codex | 6.1% (39/635) | 54 | 46 |
| 300 | Ministral | 4.8% (53/1106) | 54 | 46 |
| 300 | Qwen | 5.7% (39/686) | 54 | 46 |

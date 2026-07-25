# Topic 213 development inputs: source audit

**Scope.** This is a read-only audit of the official TREC-RAG repositories pinned by this checkout: [`trec-rag-data` at `1de1b22`](https://github.com/TREC-RAG/trec-rag-data/tree/1de1b22ac7f9936be7e42c9e70d576cc9cb83770) and [`trec-rag-skills` at `4bdbafb`](https://github.com/TREC-RAG/trec-rag-skills/tree/4bdbafb3861b7437dbf6194ff741a89dbc7bc77b). Counts below were computed directly from the released Codex qrels and nugget JSONL at those revisions.

## Authoritative topic

The complete released development narrative for `qid` **213** is:

> I'm looking into the Korean War to learn about its origins, how it ended, and why the US got involved, particularly in the context of Cold War strategy. I also want to understand its effects on US politics, any major errors that occurred, and how different presidents perceived the conflict.

Source: [`rag25-topics-dev.tsv`, line 10](https://github.com/TREC-RAG/trec-rag-data/blob/1de1b22ac7f9936be7e42c9e70d576cc9cb83770/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv#L10).

## Released answer-coverage targets

The organizer nugget release contains **50** topic-213 nuggets: **27 `vital`** and **23 `okay`**; **48 `post-edit`** and **2 `original`**. It maps them to these ten released coverage areas (count in parentheses):

1. `What triggered the Korean War?` (4)
2. `How did Cold War strategy influence US actions?` (6)
3. `What motivated US involvement in the Korean War?` (7)
4. `What major strategic or political mistakes were made during the war?` (5)
5. `What impact did the Korean War have on US politics?` (12)
6. `How did the Korean War conclude?` (8)
7. `How did US presidents differ in their views on the Korean War?` (1)
8. `New: How does the Korean War affect Korea` (1)
9. `New: How does the Korean War affect UN` (2)
10. `New: What motivated China involvement in te Korean War?` (4)

The labels, including `te`, are transcribed as released. The authoritative full nugget objects (text, mapped area, importance, and source) are the `qid` 213 JSONL record in [`rag25-dev-nuggets.jsonl`, line 2](https://github.com/TREC-RAG/trec-rag-data/blob/1de1b22ac7f9936be7e42c9e70d576cc9cb83770/trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl#L2).

For implementation planning, nuggets are coverage targets, not facts to copy into an answer: the official development-data guide says they diagnose answer coverage and **must not be used as citations or source evidence**; claims still need retrieved ClimbMix support. [Guide](https://github.com/TREC-RAG/trec-rag-skills/blob/4bdbafb3861b7437dbf6194ff741a89dbc7bc77b/skills/trec-rag-2026-track-guidelines/references/development-data.md)

## Recommended Codex projected qrels

Use the organizer-recommended [`rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels`](https://github.com/TREC-RAG/trec-rag-data/blob/1de1b22ac7f9936be7e42c9e70d576cc9cb83770/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels). It is standard TREC qrels: `topic_id 0 docid score`. It contains LLM-generated Umbrela judgments over a **pooled** set of ClimbMix documents, not exhaustive corpus-wide relevance labels. The README identifies this Codex file as the first released Codex qrels file for the **corrected narrative/sub-narrative formulation**; `v1` does not denote the older Qwen/Ministral formulation. [Organizer README](https://github.com/TREC-RAG/trec-rag-data/blob/1de1b22ac7f9936be7e42c9e70d576cc9cb83770/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/README.md)

| Grade | Organizer meaning | Topic 213 documents |
|---:|---|---:|
| 0 | Not relevant to the narrative | 629 |
| 1 | Related, but answers no sub-narrative | 375 |
| 2 | Sufficiently answers one sub-narrative | 122 |
| 3 | Sufficiently answers two to three sub-narratives | 25 |
| 4 | Sufficiently answers four or more sub-narratives | 26 |
| **Total** | **Judged topic-213 document IDs** | **1,177** |

So **173** of the topic-213 documents are grade 2 or above. The repository's current projected-qrels configurations use **grade >= 2** as the binary relevance threshold, while nDCG and grade-weighted diagnostics can retain the 0--4 scale. See [`rag25_bm25_full_query_v1.yaml`](../../../configs/rag25_bm25_full_query_v1.yaml) and [`evaluation.py`](../../../code/trec_rag/evaluation.py). Do not describe a document absent from the release as organizer-judged irrelevant: it is unjudged outside this pool.

## Nugget-to-document mapping: not released

No organizer nugget-to-ClimbMix-document mapping was found. Evidence:

- The complete official `trec-rag-data` tree at the pinned revision has one nugget data file and qrels files, but no nugget/document association file.
- Each nugget object has only `text`, `mapped_sub_narrative`, `importance`, and `source`; it has no document ID field.
- The official guide separates their roles: qrels join a topic to ClimbMix document IDs for retrieval diagnostics, whereas nuggets assess generated-answer coverage and are not evidence/citations.

Thus a qrels-positive document cannot be claimed to support a particular nugget without independently inspecting that document. This is a statement about the organizer's released 2026 development artifacts at the pinned revision, not a claim that no such mapping exists privately or may be released later.

## Provenance and reproducibility

- Data repository source paths: topic TSV; nugget JSONL; Codex qrels; and its [README](https://github.com/TREC-RAG/trec-rag-data/tree/1de1b22ac7f9936be7e42c9e70d576cc9cb83770/trec-rag-2026/development-data/rag25-dev-umbrela-qrels).
- Guideline source: [TREC RAG 2026 development data](https://github.com/TREC-RAG/trec-rag-skills/blob/4bdbafb3861b7437dbf6194ff741a89dbc7bc77b/skills/trec-rag-2026-track-guidelines/references/development-data.md).
- Method for counts: filter qrels records where first column is `213`, group by fourth column; parse the matching nugget JSON object and group `importance`, `source`, and `mapped_sub_narrative`.

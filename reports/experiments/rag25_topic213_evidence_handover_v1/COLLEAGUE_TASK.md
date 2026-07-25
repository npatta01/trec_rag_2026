# Colleague task: Korean War passage extraction and response generation

## Objective

Build a source-grounded answer to Topic 213 by extracting useful passages from
the supplied documents, converting those passages into concise supported
claims, and synthesizing the claims into a coherent response.

The organizer topic narrative is:

> I'm looking into the Korean War to learn about its origins, how it ended, and
> why the US got involved, particularly in the context of Cold War strategy. I
> also want to understand its effects on US politics, any major errors that
> occurred, and how different presidents perceived the conflict.

Use the ten exact organizer sub-narratives in
`colleague_subnarrative_candidates.jsonl` as the answer outline. Preserve their
wording even where the released labels contain typos.

## Input package

The two private extraction files are supplied alongside this task when it is
shared. In the source workspace they are under
`outputs/rag25_topic213_evidence_handover_v1/`:

- `colleague_full_corpus.jsonl`: 173 unique Topic 213 documents, each with
  `document_id`, organizer `topic_qrel_grade`, and full `text`.
- `colleague_subnarrative_candidates.jsonl`: one record for each of the ten
  sub-narratives, with the 12 highest-ranked candidate document IDs, their
  topic-level qrel grades, model ranks, and raw reranker logits.

The qrel grade measures relevance to the overall topic. It does **not** prove
that a document supports a particular sub-narrative. The model score only
orders candidates and is not a calibrated probability.

Start with the 12-document shortlist for each sub-narrative. Consult the
remaining documents in the 173-document corpus only when the shortlist is
insufficient or contradictory. This avoids sending all 173 documents through
the model ten separate times.

## Required work

For every sub-narrative:

1. Read all 12 shortlisted documents.
2. Extract the smallest self-contained passage that supports a concrete claim.
3. Record whether support is direct, partial, contradictory, or absent.
4. Convert supported passages into concise claims without adding facts.
5. Resolve duplicated and conflicting claims across documents.
6. Write a cited sub-narrative summary.

Then combine the ten summaries into one response to the full topic narrative.
Organize the response thematically or chronologically; do not answer as ten
disconnected mini-essays.

## Required outputs

Produce `passages.jsonl` with one record per reviewed passage:

```json
{
  "sub_narrative": "exact released label",
  "document_id": "shard_...",
  "topic_qrel_grade": 2,
  "model_rank": 1,
  "passage_text": "verbatim source excerpt",
  "supporting_claim": "careful paraphrase supported by the excerpt",
  "support": "direct",
  "notes": "qualification, contradiction, or source-quality concern"
}
```

Also produce:

- `subnarrative_summaries.md`: one cited synthesis per organizer label;
- `final_response.md`: the integrated answer, citing document IDs;
- `coverage.json`: document and claim counts, unsupported labels, conflicts,
  and any need to search below rank 12.

## Quality rules

- Every factual sentence in the summaries and final response must trace to at
  least one extracted passage and document ID.
- Preserve uncertainty and distinguish armistice from peace treaty.
- Do not treat repeated documents as independent corroboration.
- Prefer complementary evidence over five documents repeating the same point.
- Do not infer support from the qrel grade or model score.
- Flag dubious, partisan, or internally inconsistent sources rather than
  silently laundering their claims.

## Optional QA reference

After completing the first pass, compare the work against `handover.json`.
That file contains a separate reviewed top-five document/claim mapping for each
sub-narrative. Use it to find omissions or disagreements, not as ground truth
and not as the initial extraction input.

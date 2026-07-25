# Colleague task: Korean War passage extraction

## Objective

Extract concise verbatim evidence passages from every organizer-relevant Topic
213 document. The packet is intended for a colleague who will perform any
later claim writing or response synthesis; it does not contain generated
claims, summaries, or a final answer.

The organizer topic narrative is:

> I'm looking into the Korean War to learn about its origins, how it ended, and
> why the US got involved, particularly in the context of Cold War strategy. I
> also want to understand its effects on US politics, any major errors that
> occurred, and how different presidents perceived the conflict.

## Inputs

The private source corpus contains the 173 unique Topic 213 documents with
organizer qrel grade 2, 3, or 4. The ten labels are the exact released
sub-narratives, including their original quoting and typographical errors.

The qrel grade defines eligibility for this extraction pass. It is not a
sub-narrative support score and is intentionally omitted from the passage
packet.

## Completed output

`passages.jsonl` has one record per document with at least one extracted
passage:

```json
{
  "document_id": "shard_...",
  "evidence": [
    {
      "sub_narrative": "exact released label",
      "passages": ["verbatim source sentence or paragraph"]
    }
  ]
}
```

Unsupported label/document combinations are omitted. The packet contains no
qrel grades, model ranks, model scores, support scores, claims, summaries,
offsets, or machine-local paths.

## Extraction and verification

- Exactly one external `gpt-5.6-sol` call was attempted for each of the 173
  eligible documents, with all ten sub-narratives considered in that call.
- The model selected immutable source-passage identifiers. Final passage text
  was copied from the corresponding source document rather than generated.
- Every emitted passage was verified as an exact Unicode substring of its
  source after JSON decoding. No whitespace normalization, ellipsis, editing,
  or generated bridging text was allowed.
- Adjacent model-selected fragments were merged only when the intervening
  source characters were whitespace; the resulting span was then rechecked as
  an exact substring.
- The two initial calls failed provider-side schema validation before a model
  response. They were not retried, so the final packet contains 171 document
  records and the manifest records both failures.

## Scope boundary

The separately reviewed `handover.json` remains available as an optional
claim-level reference. Passage extraction did not modify or regenerate its
claims. Producing normalized claims, sub-narrative summaries, or a final
response is explicitly outside this completed task.

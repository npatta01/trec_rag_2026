# TREC RAG 2026 organizer data contracts

Researched 2026-07-29 from organizer-owned repositories only.

## Source priority and revisions

The current primary source is `TREC-RAG/trec-rag-data` `main` at `a6255c1`
(2026-07-29), which contains the released test topics, retrieval artifacts,
fixed-retrieval generator, and completed RAG outputs. The local submodule was
originally pinned at the earlier `1de1b22`; its `origin/main` was fetched for
this research without changing the checkout.

The current locally available `TREC-RAG/trec-rag-skills` `origin/main` is
`f281e88` (`v0.6.0`). The superproject still pins the much older `4bdbafb`
(`v0.2.0`). That pinned version says the input is a JSONL with `id`, `title`,
and `narrative`; the released organizer data and baseline instead use the
two-column TSV. The skill itself says newer official releases win when they
conflict (`trec-rag-skills@f281e88:skills/trec-rag-2026-track-guidelines/SKILL.md`,
lines 8-12). Therefore implementation decisions below prioritize the released
artifacts at `trec-rag-data@a6255c1`.

## Contract matrix

| Boundary | Canonical organizer artifact | Exact shape | Classification |
|---|---|---|---|
| Organizer -> Retrieval and RAG | `trec-rag-2026/test-data/trec_rag_2026_queries.tsv` | 119 headerless UTF-8 rows; `narrative_id<TAB>narrative` | Official shared task input |
| Retrieval -> organizer / fixed RAG | Retrieval run (`r_output_trec_rag_2026.tsv` for submission; released baselines use descriptive `.trec` names) | Six whitespace-separated fields: `query_id Q0 document_id rank score run_tag` | Official TREC run contract |
| Organizer baseline -> fixed RAG | `trec-rag-2026/baselines/retrieval/bm25_climbmix_top1000_with_text.jsonl.zip` | ZIP member `bm25_top1000.jsonl`; one query-bundled JSON object per topic with `query` and `candidates`; candidates carry full text | Officially released baseline input, but not a participant submission artifact |
| RAG -> organizer | `rag_output_trec_rag_2026.jsonl` (generator output basename is caller-selected; organizer examples are descriptively named) | One object per narrative with exactly `metadata`, `references`, `answer`; use the strict released-baseline intersection detailed below | Official answer-object contract |

## 1. Official narrative input

Canonical path and schema:

```text
trec-rag-2026/test-data/trec_rag_2026_queries.tsv

narrative_id<TAB>narrative
```

There are 119 rows, no header, and IDs `rag2026-0` through `rag2026-118`.
Preserve both fields exactly. There is no `title` field in the released input.
Parse on a tab, not arbitrary whitespace. The raw file's SHA-256 is
`72dc2fd358d3eeda973397ccd7a8775545b19a6deaefc67709167eee6a9f8a2c`.

Sources:

- `trec-rag-data@a6255c1:trec-rag-2026/test-data/trec_rag_2026_queries.tsv`,
  lines 1-119 (the released data).
- `trec-rag-skills@f281e88:skills/trec-rag-2026-track-guidelines/references/test-data.md`,
  lines 10-45 (path, filename, two-column headerless schema, preservation),
  47-58 (same input for both tasks), and 70-85 (count, identifiers, checksum).
- The released fixed-retrieval reader parses a two-column TSV in
  `trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/code/ragnarok_style_ag.py`,
  lines 137-153. Its `len(fields) < 2` behavior is implementation tolerance,
  not evidence for a third official field.

The stale pinned skill's `trec_rag_2026_queries.jsonl` and
`id`/`title`/`narrative` schema must not be used.

## 2. Retrieval run

The organizer format is the standard six-column TREC run:

```text
query_id Q0 document_id rank score run_tag
```

The released retrieval README states that shape at
`trec-rag-data@a6255c1:trec-rag-2026/baselines/retrieval/README.md`, lines
107-113. The task guideline names the submission output
`r_output_trec_rag_2026.tsv` and defines the corresponding fields as
`topic_id Q0 docid rank score run_id`
(`trec-rag-skills@f281e88:skills/trec-rag-2026-track-guidelines/references/retrieval-task.md`,
lines 92-118). Those names describe the same six positions.

Rules from the current task guideline:

- the first field is the exact narrative ID;
- `Q0` is literal;
- document IDs are ClimbMix IDs;
- ranks start at 1 per narrative and rows sort by ascending rank;
- scores are numeric and non-increasing within a narrative;
- the run tag/ID is stable;
- final depth is participant-chosen independently per narrative and must not be
  padded to a fixed cutoff (`retrieval-task.md`, lines 72-90 and 120-130).

This is participant output, not a run the organizers promise to provide for
every system. The organizers do, however, publish baseline run files. The
fixed-retrieval baseline intentionally accepts any six-column TREC run and
preserves its rank order
(`trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/code/README.md`, lines 1-26;
`ragnarok_style_ag.py`, lines 156-183).

Repository implication: keep the submission-compatible local basename
`r_output_trec_rag_2026.tsv`. It can feed answer generation directly; no
translation into a custom ranking schema is needed.

## 3. Full-text input to fixed-retrieval RAG

The current organizer data release now provides an actual full-text archive:

```text
trec-rag-2026/baselines/retrieval/
  bm25_climbmix_top1000_with_text.jsonl.zip
```

Its SHA-256 is
`cebbeb313065572ad69aaf3c9f311546bb09cf6d2085fc0160c81f9e6627c35c`
(`trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/code/README.md`, lines
36-52). It contains one JSONL member named `bm25_top1000.jsonl`, with one JSON
object per topic
(`trec-rag-data@a6255c1:trec-rag-2026/baselines/retrieval/README.md`, lines
107-115).

The released README describes each topic object as containing the query,
document IDs, ranks, scores, retrieval settings, and full text
(`retrieval/README.md`, lines 15-25). The released generator documents and
accepts the query-bundled candidate form:

```json
{
  "query": {"qid": "q1"},
  "candidates": [
    {"docid": "d1", "doc": "Document text."}
  ]
}
```

Source:
`trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/code/README.md`, lines
102-126. Its loader accepts `docid` (also `id`/`_id`) and text in `text`,
`doc`, `contents`, or `body`, retains only ranked docids, rejects missing text,
and detects conflicting duplicate text (`ragnarok_style_ag.py`, lines
186-238).

Important distinction: this ZIP is an **official organizer-provided baseline
artifact and reference interchange shape**, not a file participants submit to
the RAG task. The generator also accepts flat one-document-per-line JSONL and
ZIPs with another JSONL member name (`code/README.md`, lines 102-126). Thus the
organizer has released and used a standard-compatible full-text shape, but has
not mandated one universal archive basename for participant pipelines.

### Comparison with this repository

This repository's `retrieval_with_text.jsonl.zip` is structurally compatible:
it contains `retrieval_with_text.jsonl`, with one row per narrative shaped as
`query: {qid, text}` plus `candidates` containing `docid`, `rank`, `score`,
`doc`, `index`, and `stage`
(`code/trec_rag/retrieval_export.py`, lines 285-314, 345-360, and 1740-1750).
The organizer generator ignores the extra candidate fields and accepts the
`doc` text field.

Therefore the new generation path should consume this same query-bundled
candidate schema without conversion. Keep its local filename if desired, but
document it as the project's generated equivalent of the organizer's
`bm25_climbmix_top1000_with_text.jsonl.zip`, not as the canonical organizer
basename. Joining must use exact `docid`; the six-column run determines which
documents and order, while the ZIP supplies text. The released generator fails
if any ranked docid lacks text (`ragnarok_style_ag.py`, lines 211-238).

## 4. Final RAG answer JSONL

The released answer files each contain 119 objects. Their top-level shape is:

```json
{
  "metadata": {
    "team_id": "castorini",
    "narrative_id": "rag2026-0",
    "narrative": "...exact TSV narrative...",
    "run_id": "...",
    "run_desc": "..."
  },
  "references": ["shard_..."],
  "answer": [
    {"text": "One sentence or structural unit.", "citations": [0]}
  ]
}
```

Primary evidence:

- `trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/README.md`, lines 1-16,
  states that released records follow the current answer-object contract and
  identifies both 119-record outputs.
- The same README, lines 35-41 and 149-170, says the runner supplies the five
  official metadata fields and describes the fixed-retrieval baseline.
- `trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/code/ragnarok_style_ag.py`,
  lines 287-327, constructs and validates the exact shape.
- The released file
  `trec-rag-data@a6255c1:trec-rag-2026/baselines/rag/gpt-5.6-sol_medium_single-pass-rag_first-qwen3-8b-listwise-top100.jsonl`,
  line 1 (and all 119 records), demonstrates it concretely.

For maximum compatibility with what the organizer currently **produces and
validates**, use the strict released-baseline contract:

- exactly three root keys: `metadata`, `references`, `answer`;
- metadata exactly the five string fields above; narrative ID and text copy
  the official TSV;
- nonempty, unique ClimbMix `references`, all drawn from the selected TREC run;
- nonempty `answer`; each object has exactly nonempty `text` and `citations`;
- 1-3 unique zero-based integer citations per answer object, all in range;
- every reference is cited;
- total words across `answer[].text` is at most 1,024, counted with
  `str.split()`.

The executable checks are at `ragnarok_style_ag.py`, lines 301-327. The code
README summarizes the same checks at lines 81-100.

There is no `title`, `type`, or `prompt` field in released RAG metadata. The
older PR/pinned-guideline schema using those fields is incompatible with the
current released artifacts. `run_desc` is required by the released generator.

## Organizer-source conflict

`trec-rag-skills@f281e88` is newer than the pinned skill and correctly uses the
TSV and five metadata fields, but its `rag-task.md` is more permissive than the
released `trec-rag-data@a6255c1` reference validator: it permits extra metadata,
uncited references, empty citation arrays, and direct-docid citations
(`skills/trec-rag-2026-track-guidelines/references/rag-task.md`, lines 100-118
and 149-159). The released baseline code instead requires exact metadata,
all references used, and 1-3 integer citations.

Do not guess which permissive forms the eventual submission portal will accept.
The strict baseline form is a subset of the skill's permissive form and is
demonstrably used in the organizer's released outputs, so it is the safest
implementation target. This is a compatibility recommendation, not a claim
that the permissive skill text has been formally revoked.

## Required design corrections

1. Both paths read the official two-column TSV directly; no JSONL conversion
   and no synthetic title.
2. Retrieval emits the six-column organizer TREC run under
   `r_output_trec_rag_2026.tsv`.
3. Retrieval also emits the query-bundled full-text JSONL ZIP already used by
   this repository. Treat it as structurally compatible with the organizer's
   released baseline archive, while keeping the filename distinction clear.
4. Generation accepts: official TSV + six-column TREC run + query-bundled
   JSONL/ZIP full text, just like the released Ragnarok-style generator.
5. Generation emits the strict five-metadata-field answer-object JSONL above;
   require `run_desc`, and remove `title`, `type`, and `prompt` requirements.
6. Validate the strict intersection: integer citations, 1-3 per object,
   selected unique references all used, and exact 1,024-word cap.

## Remaining ambiguity

The organizer guideline says portal-specific upload procedures are not yet
specified (`trec-rag-skills@f281e88:skills/trec-rag-2026-track-guidelines/SKILL.md`,
lines 32-40). The released basenames and structures are sufficient to implement
and test both local paths, but final archive/upload rules must be rechecked
before submission.

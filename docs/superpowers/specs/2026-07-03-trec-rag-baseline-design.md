# TREC RAG Baseline Design

## Goal

Build a simple, reproducible baseline for TREC RAG 2026 that produces both required task outputs:

- Retrieval output: `r_output_trec_rag_2026.tsv`
- RAG output: `rag_output_trec_rag_2026.jsonl`

The baseline should use the official-style topic schema as its main input and support local development data for smoke testing.

## Scope

The first baseline will use Pyserini BM25 over the ClimbMix index, `climbmix-400b`, through the Pyserini REST API. It will retrieve the top 100 documents per topic by default.

This design does not include query rewriting, reranking, custom chunking, advanced generation, or official submission upload handling. Those can be added after the baseline is running and validated.

## Inputs

The primary pipeline input is a JSONL topic file named like the official input:

```json
{"id":"1","title":"Topic title","narrative":"Long topic narrative"}
```

For local development, a converter will transform the checked-in TSV topic files into this JSONL shape:

- `trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv`
- `trec-rag-data/trec-rag-2026/development-data/topics/research-rubrics-topics-dev.tsv`

For TSV-derived topics, the converter will preserve the TSV `qid` as `id`. It will derive a short `title` from the first words of the prompt or narrative and preserve the full TSV text as `narrative`.

## Components

### `scripts/convert_dev_topics.py`

Converts a development TSV file into official-style topic JSONL.

Responsibilities:

- Read `qid<TAB>text` topic rows.
- Emit one JSON object per line with `id`, `title`, and `narrative`.
- Preserve topic IDs exactly.
- Avoid modifying source data.

### `scripts/run_baseline.py`

Runs retrieval and automatic RAG output generation.

Responsibilities:

- Load official-style topic JSONL.
- Query the Pyserini REST API using the topic `title` as the default BM25 query.
- Request top 100 results from `climbmix-400b` unless a different depth is supplied.
- Cache raw search responses under `outputs/baseline/cache/`.
- Write the six-column retrieval runfile.
- Build a conservative automatic RAG object for each topic using retrieved document text.

### `scripts/validate_outputs.py`

Validates the generated retrieval and RAG outputs.

Responsibilities:

- Confirm every topic has retrieval rows and one RAG JSON object.
- Confirm retrieval rows have six whitespace-separated columns.
- Confirm ranks restart at 1 for each topic and are sorted ascending.
- Confirm scores are non-increasing within each topic.
- Confirm RAG JSONL is valid and has `metadata`, `references`, and `answer`.
- Confirm citation indices point into `references`.
- Confirm every reference is cited at least once.

## Retrieval Output

The retrieval output will be written to:

```text
outputs/baseline/r_output_trec_rag_2026.tsv
```

Each line will follow the official six-column format:

```text
topic_id Q0 docid rank score run_id
```

The default `run_id` will be `pyserini_climbmix_bm25_top100`.

## RAG Output

The RAG output will be written to:

```text
outputs/baseline/rag_output_trec_rag_2026.jsonl
```

The first implementation will use `metadata.type: "automatic"` because the script, not a manual per-topic agent review, creates the answer objects.

The automatic baseline will be intentionally conservative:

- Select a small evidence subset from top-ranked results that include usable document text.
- Write short sentence-level answer objects.
- Cite only selected ClimbMix document IDs.
- Avoid adding facts that are not present in the selected evidence.
- Produce an insufficiency sentence when retrieved evidence text is unavailable or too weak.

The RAG baseline is meant to be a valid starting point, not a strong final system.

## Authentication And Configuration

The pipeline will read local configuration from environment variables:

- `PYSERINI_API_TOKEN`: required for authenticated API calls.
- `INDEX_URL`: hosted Pyserini REST search endpoint.

The implementation must not print, log, commit, or write API tokens into generated outputs.

## Data Flow

1. Convert development TSV topics to official-style JSONL when needed.
2. Load topic JSONL.
3. For each topic, search ClimbMix using the topic title.
4. Cache the raw search response.
5. Write retrieval rows from returned document IDs, ranks, and scores.
6. Select evidence documents from retrieved candidates.
7. Write one cited RAG JSON object per topic.
8. Validate both outputs against local schema and consistency rules.

## Error Handling

The baseline should fail clearly when:

- The topic file is missing or malformed.
- `PYSERINI_API_TOKEN` is unavailable.
- The Pyserini REST API returns an authentication, index, timeout, or malformed-response error.
- A search response lacks candidate document IDs.
- Validation finds invalid output structure or citation indices.

Partial output files should not silently look successful. The script should exit non-zero when required topics fail.

## Testing

Implementation should include focused tests for:

- TSV-to-JSONL conversion.
- Retrieval runfile formatting and rank handling.
- RAG citation validation.
- Output validators catching malformed rows, missing topics, and bad citation indices.

Network calls to the Pyserini REST API should be isolated from unit tests. A small smoke test can be run manually against one or two topics when a local token is available.

## Future Extensions

After this baseline works, likely improvements include:

- Query rewriting from the narrative.
- Multi-query retrieval and fusion.
- Reranking.
- Better evidence selection.
- A stronger generation model or manual agent-written RAG baseline with `metadata.type: "manual"`.
- Offline retrieval evaluation against the included development qrels.

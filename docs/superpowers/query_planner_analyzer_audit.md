# Query-planner analyzer audit

Date: 2026-07-10

## Decision

`unicode_content_terms_v1` is **not equivalent** to the analyzer used by the
repository's BM25 retrieval path. It remains test-only and must not be described
as a count of terms retained by the hosted ClimbMix service.

Planner v2 resolves the model-comparison blocker by freezing a different
contract: `lucene_default_english_v1`, a local Lucene 10.4.0 reference
implementation of the inspected upstream Anserini `DefaultEnglishAnalyzer`
chain. It is preflighted and fingerprinted before model requests. Its counts
are explicitly **planner-budget reference tokens**, not verified hosted BM25
tokens. This makes the two planner arms comparable; later retrieval metrics on
the actual rendered strings, rather than this token gate, establish retrieval
effectiveness. No retrieval or qrels were used for this audit.

## Actual request path

The repository does not analyze query text before retrieval:

1. `PyseriniRemoteRetriever.retrieve` passes `query.query_text` to
   `RemotePyseriniClient.search` unchanged.
2. `RemotePyseriniClient.search` URL-encodes that string as the REST `query`
   parameter.
3. The hosted service's read-only `/` metadata identifies it as a Pyserini API
   aligned with Anserini. Its read-only `/openapi.yaml` exposes a text query
   but no analyzer-selection parameter, analyzer endpoint, package revision,
   or analyzer fingerprint.

Current upstream Pyserini passes a REST string query directly to
`LuceneSearcher.search`. `LuceneSearcher` constructs Anserini's
`SimpleSearcher`, which uses `IndexCollection.DEFAULT_ANALYZER`. In the pinned
upstream sources inspected below, that default is
`DefaultEnglishAnalyzer.newDefaultInstance()`:

- Pyserini server backend, commit
  [`e81bbf2`](https://github.com/castorini/pyserini/blob/e81bbf2cc89c83b327ad11bf460e8a9692b2bb83/pyserini/server/backend.py#L538)
- Pyserini Lucene searcher construction, commit
  [`e81bbf2`](https://github.com/castorini/pyserini/blob/e81bbf2cc89c83b327ad11bf460e8a9692b2bb83/pyserini/search/lucene/_searcher.py#L38-L56)
- Anserini `SimpleSearcher` default, commit
  [`bd93b89`](https://github.com/castorini/anserini/blob/bd93b899a1f34362b4146153c94a8eab14d9a7da/src/main/java/io/anserini/search/SimpleSearcher.java#L98-L115)
- Anserini default-analyzer declaration, commit
  [`bd93b89`](https://github.com/castorini/anserini/blob/bd93b899a1f34362b4146153c94a8eab14d9a7da/src/main/java/io/anserini/index/IndexCollection.java#L55)

The upstream default pipeline is:

```text
StandardTokenizer
  -> EnglishPossessiveFilter
  -> LowerCaseFilter
  -> StopFilter(Lucene English 33-word set)
  -> PorterStemFilter
```

Source: Anserini
[`DefaultEnglishAnalyzer`](https://github.com/castorini/anserini/blob/bd93b899a1f34362b4146153c94a8eab14d9a7da/src/main/java/io/anserini/analysis/DefaultEnglishAnalyzer.java#L56-L93).
Pyserini's
[`Analyzer` guide](https://github.com/castorini/pyserini/blob/e81bbf2cc89c83b327ad11bf460e8a9692b2bb83/docs/usage-analyzer.md)
also gives the concrete default example `City buses are running on time.` ->
`citi`, `buse`, `run`, `time`.

There is one important limit to this evidence: the hosted API does not expose
its installed Pyserini, Anserini, or Lucene revision or the index-time analyzer
configuration. The upstream default and the repository's existing behavioral
notes are consistent, but the service cannot currently provide a machine-
checkable analyzer identity. That missing identity is itself enough to reject
an equivalence claim.

## Exact mismatches

| Dimension | `unicode_content_terms_v1` | upstream BM25 default | Consequence |
|---|---|---|---|
| Stemming | none | Porter | `banks banking` counts as two unique planner tokens but one unique analyzed form, `bank`, for BM25. |
| Possessives | keeps an internal apostrophe | strips English possessive endings before stemming | `bank's` is counted differently from BM25 `bank`. |
| Hyphens | keeps an internal ASCII hyphen as one token | `StandardTokenizer` splits ordinary hyphenation | `open-source` is one planner token but two BM25 inputs (`open`, then stemmed `sourc`). |
| Stopwords | custom 53-word set | Lucene English 33-word set | The planner retains `no not such then there these will`, which BM25 removes; it removes the 27 words listed below that BM25 retains. |
| Unicode normalization | NFKC, then Python `casefold` | no NFKC char filter in this analyzer; Lucene lowercase filter | Compatibility characters and some non-ASCII case mappings can differ. |
| Unicode segmentation | Python word regex | Lucene UAX #29 `StandardTokenizer` | CJK ideographs, emoji, and some punctuation-bound words have different token boundaries or retention. |
| Multiplicity | returns first-occurrence unique terms | bag-of-words generator groups analyzed tokens and boosts by frequency | Unique-token budgeting can deduplicate, but actual BM25 query term weights still preserve repetition. |

The stopword-set differences are exact for the inspected versions:

- Planner-only stopwords (removed by the planner, retained by BM25):
  `been can could do does from had has have how i its like may should them were
  what when where whether which while who why would you`.
- Lucene-only stopwords (retained by the planner, removed by BM25):
  `no not such then there these will`.

The first list has 27 terms and the second has 7. The 26 shared terms do not
make the two analyzers equivalent.

## Where the mismatch changes planner behavior

`analyze_content_terms` currently controls all of the following:

- the narrative-size input to the global 25-percent expansion allowance;
- which rendered global terms count as newly added unique content tokens;
- the 5-to-25 unique-token facet gate;
- whether a single exact source span qualifies for the greater-than-25
  indivisible-span exception; and
- the `unique_content_tokens` recorded with each rendered query.

It does not alter the rendered query string. Therefore, the risk is asymmetric:
the planner may reject a query BM25 would see as within budget, or accept a
query BM25 would see as outside the intended budget, while the REST service
still analyzes the full rendered text using its own rules.

## Recommended frozen analyzer API

Freeze an analyzer abstraction before another model-planning run:

```python
@dataclass(frozen=True)
class AnalyzedQuery:
    tokens: tuple[str, ...]         # analyzer output, including repetitions
    unique_tokens: tuple[str, ...]  # stable first-occurrence order
    fingerprint: "AnalyzerFingerprint"

class QueryAnalyzer(Protocol):
    def analyze(self, text: str) -> AnalyzedQuery: ...
```

`AnalyzerFingerprint` should be serialized into plan provenance and cache
compatibility checks. At minimum it should contain:

- API implementation and package revisions (Pyserini, Anserini, Lucene);
- analyzer class;
- tokenizer and ordered filters;
- stemmer;
- stopword-set SHA-256;
- Unicode/tokenizer version if available;
- index name and a server/index configuration identifier; and
- an application-level analyzer contract version.

The preferred implementation is the exact Anserini analyzer, either through
Pyserini's `Analyzer(get_lucene_analyzer())` locally or through a read-only
hosted `/analyze` endpoint that returns tokens and the fingerprint. A Python
reimplementation should not be called equivalent until a conformance corpus
covers, at minimum, inflection, possessives, hyphens, all stopword-set
differences, compatibility Unicode, combining marks, CJK, emoji, acronyms,
decimals, and repeated terms, with zero token-sequence differences against the
same Anserini/Lucene revision used by the hosted index.

Until a hosted fingerprint or `/analyze` API exists, record the hosted index as
`hosted_climbmix_unknown_revision`, call local values planner-budget reference
tokens rather than hosted BM25 terms, and do not use the 5–25 gate as evidence
of hosted analyzer retention. `unicode_content_terms_v1` is allowed only in
unit tests; formal planner runs use the pinned Lucene reference sidecar.

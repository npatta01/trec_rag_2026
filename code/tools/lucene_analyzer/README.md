# Local Lucene query analyzer

This sidecar runs the exact analyzer chain used by Anserini's
`DefaultEnglishAnalyzer` against Lucene 10.4.0:

```text
StandardTokenizer
  -> EnglishPossessiveFilter
  -> LowerCaseFilter
  -> StopFilter(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)
  -> PorterStemFilter
```

It is read-only, binds to localhost through the container command, and exposes
`GET /health` plus `POST /analyze` with a UTF-8 `text/plain` body. Responses
include analyzed tokens with repetitions and a frozen analyzer fingerprint.

The hosted ClimbMix service does not expose its installed package or analyzer
fingerprint. This sidecar is therefore a pinned reference implementation of the
documented upstream default, not proof of the hosted deployment's exact build.
The run ledger must preserve that limitation and the hosted index identifier.

Use `code/tools/run_lucene_analyzer.sh` to download checksum-verified Lucene
jars, compile the server in a pinned Java container, and start it.

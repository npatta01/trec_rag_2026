# Query planner v2.1 synthetic smoke 001

Date: 2026-07-11

## Decision

**Mechanical failure; stop with no retry.** The versioned synthetic request
passed transport, compiler-manifest, live-runtime, analyzer, context, and JSON
schema gates. The local model returned complete schema-constrained JSON with
`finish_reason="stop"`, but Python rejected the plan because its global anchors
did not include an anchor typed as `entity` or `topic`.

This was the one explicitly authorized v2.1 synthetic first emission. It was
not rerun, repaired, or semantically tuned. The known five topics were not
requested, and the formal diagnostic remains NO-GO.

## Frozen boundary

- Run ID: `query_plan_v2_1_synthetic_smoke_001`
- Topic: built-in `synthetic_transport_v2_1`, outside the development topics
- Source commit: `ec412c4c1a7b8feb2800eb11829d1133590912d2`
- Source tree: `68b1511e4b16a9a609e765d81495d761499f5b2a`
- Schema: `query_plan_v2_1`
- Prompt: `sparse_query_planner_v6`
- Renderer: `deterministic_sparse_renderer_v3`
- Model: local `openai/gpt-oss-20b`, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`
- Served alias: `gpt-oss-local`
- vLLM: 0.24.0; XGrammar: 0.2.3, explicitly pinned backend
- Image digest:
  `sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`
- Analyzer fingerprint SHA-256:
  `f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4`
- Compiler manifest SHA-256:
  `cb96279f3f4919d1e6b1429b091bd286a951d03da346fcdad1114516aebdafbc`
- Context: 1,947 prompt + 5,400 maximum output = 7,347/8,192
- Execution: serial and loopback-only; zero retrieval, reranking, and paid API
  calls

## Mechanical outcome

- HTTP/schema transport: pass
- HTTP status: 200
- JSON parse: pass
- Schema/topic identity: pass
- Finish reason: `stop`
- Request elapsed time: 40.51 seconds
- Usage: 1,947 prompt + 423 completion = 2,370 local tokens
- Top-level outcome: `plan_validation_error`
- Validator message: `global anchors must include an entity or topic anchor`
- Hard safety failure: false
- Original-only fallback: exactly one
- Raw HTTP body: 2,451 bytes
- Raw body SHA-256:
  `9df74dd6d5da6b0ca09fb2c96cfc20441ffb5b17f82d4cb4a0dd23f81b883f4a`
- Raw base64/SHA round trip: pass
- Offline `--rebuild-only` integrity validation: pass; expected nonzero exit
  because the immutable run contains one failed outcome

The plan's semantics were not scored or used for prompt tuning. Only the
mechanical counts needed to establish that the schema response existed were
inspected: four anchors, two coverage items, one facet, and three global
expansion terms.

## Durable evidence

Byte-identical ledger copies are tracked under `artifacts/`:

- `_run.json`:
  `a705b8ab28d09d43e6e73c04929409e4dfe90badbe64775e0591605cb7a9dd9c`
- `outcome.json`:
  `6f1842f038e17d7464110dc8b4e3161abb93e87902a94aea1150a80b49f198ac`
- `raw_response.json`:
  `81a0a5be70491d9bc896eb1ac4af1943224189470665876a3fb3debdd0bbb3f4`
- `plans.manifest.json`:
  `c1d71b5568ab53ab09669c9b05d6b9e72e2061cb9174e73f4be0f8758a7cb150`

## Next gate

The IR advisor is reviewing this mechanical failure. No further model call is
authorized. In particular, this result does not justify spending bandwidth or
compute on a larger planner, and it does not authorize the known-five run.

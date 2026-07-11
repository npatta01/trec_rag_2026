# Query planner v2.1 schema compatibility preflight

Date: 2026-07-11

## Decision

The preserved synthetic-smoke 001 failure was isolated to decoder compatibility:
vLLM 0.24 rejected array `uniqueItems` before inference. The offline v2.1 patch
removes only that keyword. Python continues to reject duplicate ID references,
so accepted-plan invariants are unchanged.

Both delivered-boundary versions are bumped because the prompt embeds the exact
schema:

- payload schema: `query_plan_v2` -> `query_plan_v2_1`
- prompt: `sparse_query_planner_v5` -> `sparse_query_planner_v6`

Renderer `deterministic_sparse_renderer_v3`, tokenizer
`narrative_token_tape_v1`, and the Lucene reference-analyzer contract remain
unchanged.

## Offline-only validation

The reusable preflight is
`code/tools/query_plan_v2_schema_preflight.py`. It recursively rejects the
unsupported-feature set copied from the pinned vLLM 0.24 XGrammar backend with
JSON-path diagnostics:

- arrays: `uniqueItems`, `contains`, `minContains`, `maxContains`;
- number/integer: `multipleOf`;
- objects: `patternProperties`, `propertyNames`;
- string `format` values outside vLLM's supported set.

It then compiles every exact schema inside the pinned local image with:

- `xgrammar.Grammar.from_json_schema(..., strict_mode=True)`;
- `llguidance.JsonCompiler(...).compile(..., check=True)`.

No HTTP request, model inference, retrieval, reranking, or paid API call occurs.

The first compiler manifest, `compiler_manifest.json`, records the pre-restart
server whose structured-output backend was still the default `auto`. Results:

- vLLM: 0.24.0
- XGrammar: 0.2.3
- llguidance: 1.7.6
- pinned image digest:
  `sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`
- exact schemas compiled: new synthetic plus topics 144, 213, 224, 407, 515
- XGrammar strict passes: 6/6
- llguidance checks: 6/6
- unsupported-feature findings: 0

Tokenizer-only local context preflight also passed without inference. The new
synthetic prompt uses 1,947 tokens (845-token margin after the 5,400 cap); the
five known prompts use 2,530–2,672 tokens, leaving 120–262 tokens.

Unit tests also prove that Python rejects duplicate references without relying
on decoder-side `uniqueItems`.

## Remaining gate

After explicit user authorization, the same pinned image/model was restarted
with backend `xgrammar` and concurrency one. Runtime provenance is recorded in
`docs/superpowers/query_planner_v2_xgrammar_runtime.json`. The exact six-schema
preflight was repeated into `compiler_manifest_xgrammar.json`; all 6/6 strict
XGrammar and 6/6 llguidance checks pass. The runner additionally binds that
manifest to the live Podman image, launch command, port mapping, `/version`, and
`/v1/models` response before inference.

The authorization covers exactly one new synthetic smoke. The five-topic
diagnostic remains NO-GO.

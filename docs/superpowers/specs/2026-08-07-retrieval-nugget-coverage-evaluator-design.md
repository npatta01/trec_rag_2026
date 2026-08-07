# Retrieval Nugget Coverage Evaluator Design

**Date:** 2026-08-07
**Status:** Design approved; implementation not started.

## Objective

Add a small, year-neutral development evaluator that answers exactly one
question about one topic:

> How completely do this topic's canonical retrieval nuggets cover the answer
> obligations that a planner derived from this topic's narrative?

The output is a diagnostic score for a frozen, narrative-derived plan under a
versioned evaluator. It is not official TREC recall, not a nugget-level relevance
judgment, and not evidence that every relevant real-world fact was retrieved.

## What This Score Is Not

State these limits in the report artifact and in any summary of a result:

- It measures coverage of a **planner-derived** plan, not of ground truth. A
  different planner or prompt version yields a different denominator.
- It never opens selected passages, so it cannot detect a canonical nugget that
  misstates the passage it came from.
- A low score has at least three distinct causes it cannot separate: retrieval
  never found the information; selection dropped it; or canonicalization did not
  express it as a claim. Diagnosing which one requires the raw-retrieval work
  that is explicitly out of scope below.
- It is not comparable across evaluator schema versions, prompt versions, or
  planner/judge models. Every reported score carries those identities.

## Terminology

These three things are routinely confused. The implementation, prompts, schemas,
and skill text must use the full term wherever the distinction could be unclear.

- **Narrative** — the complete official information need for one topic, exactly
  as retrieval received it.
- **Selected passage** — exact corpus text chosen by retrieval. `AGENTS.md`
  names selected passages the factual authority. This evaluator never reads them.
- **Canonical retrieval nugget** — a short claim produced by *our* retrieval
  pipeline's canonicalization stage from selected passages, and carried into the
  sealed handoff as an advisory claim hint. This is the evaluator's only
  evidence representation.
- **Organizer gold nugget** — an organizer-produced evaluation nugget. This
  evaluator never reads, imports, or approximates one.
- **Answer obligation** — an evidence-neutral statement of information a good
  answer must address, e.g. "state the projected revenue figure and the
  assumptions behind it" — never an assertion of what that figure is.
- **Facet** — an ordered group of answer obligations.
- **Required obligation** — one whose planner-supplied narrative spans are exact
  substrings of the narrative.
- **Supplemental obligation** — a planner-inferred addition with no explicit
  narrative support.
- **Coverage judgment** — `full`, `partial`, or `unsupported` for one obligation
  against the supplied canonical retrieval nuggets.

## Core Assumption

Version 1 assumes canonical retrieval nuggets faithfully represent the selected
passages they were derived from. It does not reopen, ground, or re-judge those
passages. This assumption defines the evaluation boundary and must appear in the
report artifact.

Note the tension this creates with `AGENTS.md`, which records that selected
passages are factual authority and canonical claim hints are advisory. The
evaluator deliberately scores the advisory layer because that layer is cheap,
already sealed, and already hash-authenticated. Passage-level verification is a
separate, larger piece of work.

## Scope

### Included

- One narrative-only planner call per topic.
- One judge call per topic carrying the frozen plan and all canonical retrieval
  nuggets.
- Strict structured model output with local structural and semantic validation.
- Deterministic scoring for required and supplemental obligations.
- Hash-bound input, plan, judgment, report, and manifest artifacts in a private
  work directory, with create/resume modes.
- Injected planner and judge backends so tests need no hosted call.
- One repository module that owns the implementation and its CLI.
- A short retrieval-nugget-coverage route added to the existing competition
  debug-report skill.

### Excluded

- Any 2025-specific data, topic knowledge, dataset path, gold nugget, qrel, or
  calibration constant.
- A second planner, a planner ensemble, or a completeness critic.
- Selected-passage verification or nugget-faithfulness checking.
- Raw-retrieval inspection, missed-document diagnosis, or stage attribution.
- New searches, reranking, answer generation, or RAG answer evaluation.
- Cross-topic aggregation, trend tracking, or comparison reports.
- An HTML report.
- Any hosted call without an explicit execution flag.

A separate future calibration harness may score this generic evaluator's frozen
outputs against development gold. That harness is not part of this spec, this
module, or this skill route, and nothing in this evaluator may anticipate it.

## Repository Grounding

Decisions below are anchored to existing code, checked in this worktree:

| Fact | Source |
| --- | --- |
| The sealed handoff carries `topic_id`, `narrative`, `narrative_sha256`, and ordered `claim_hints` per topic | `code/trec_rag/generation_handoff.py:289` |
| Each `ClaimHint` carries `claim_id` and `text` for one canonical retrieval nugget, with unique `claim_id` per topic | `code/trec_rag/generation_handoff.py:232`, `code/trec_rag/generation_handoff.py:366` |
| The handoff projection emits one claim hint per validated canonical nugget, including extractive-fallback nuggets | `code/trec_rag/generation_handoff_export.py:371` |
| `load_generation_handoff` fully authenticates the manifest; `select_generation_topics` rejects unknown and duplicate topic IDs | `code/trec_rag/generation_handoff.py:760`, `code/trec_rag/generation_handoff.py:870` |
| `claim_hints` may legitimately be empty for a topic | `code/trec_rag/generation_handoff.py:317` |
| Compact aliases resolved locally, with the model schema constrained by `enum`, are the established pattern for referring to evidence | `code/trec_rag/canonical_nuggets.py:111`, `code/trec_rag/canonical_nuggets.py:692` |
| Injected structured backends use a `Protocol` plus a shared `BackendReply` record | `code/trec_rag/facet_extraction.py:94`, `code/trec_rag/facet_extraction.py:321` |
| Cross-module reuse of the OpenRouter transport helpers is already an accepted pattern | `code/trec_rag/canonical_nuggets.py:28` |
| Post-run diagnostics take CLI flags with repeated `--topic`, not a YAML config | `code/trec_rag/competition_debug_report.py` |
| Tests live in `code/tests/` with `pythonpath = ["code"]` | `pyproject.toml` |
| `outputs/` and `cache/` are ignored, so run-local artifacts stay private | `.gitignore` |

## Input Boundary

**Decision: the evaluator's only artifact input is the sealed generation handoff
manifest plus one topic ID.**

An earlier draft of this design read `<topic_root>/canonical/canonical-nuggets.jsonl`
directly. That is rejected. Reading the raw canonical artifact would require
re-implementing the retrieval config load, the canonical checkpoint hash chain,
per-subnarrative result-state semantics (`complete`, `empty`,
`fallback_extractive`), and evidence decoding — roughly the authentication block
already present at `code/trec_rag/competition_debug_report.py:1298`. The handoff
gives the narrative and every canonical retrieval nugget from one already
authenticated file.

The CLI therefore takes:

- `--handoff-manifest PATH` — a sealed `generation_handoff_manifest.json`.
- `--topic TOPIC_ID` — required, exactly one per invocation. Cross-topic
  aggregation is out of scope, so a repeated flag would only add namespace and
  resume complexity.
- `--work-dir PATH` — defaults to
  `<manifest parent>/retrieval_nugget_coverage/<topic_id>`, which lands under the
  ignored run output tree.
- `--planner-model NAME`, `--judge-model NAME` — model identities bound into the
  manifest.
- `--mode create|resume` — default `create`.
- `--allow-hosted-calls` — off by default.

There is no YAML config file. The competition runners use YAML because they own
long-lived pipeline state; this is a post-run diagnostic, and the debug-report
CLI is the correct precedent.

From the selected topic the evaluator reads exactly:

- `topic.narrative` and `topic.narrative_sha256`;
- for each `ClaimHint`, in handoff order, its `claim_id` and `text`; and
- `handoff.manifest_sha256`, for provenance binding only.

It reads no passage text, `group_id`, `kind`, `evidence_ids`, docid, rank,
query, qrel, gold label, or generation output. `group_id` and `kind` are withheld
deliberately: including them would let the judge reason about pipeline structure
instead of claim content.

Each nugget receives a local alias `n001`, `n002`, … in handoff order. Only the
alias and the claim text reach a model; `claim_id` is resolved locally. This
mirrors the existing evidence-alias pattern, keeps request bytes down, and makes
judge ID validation a plain `enum` check.

## Module and CLI Boundary

One module owns everything:

- `code/trec_rag/retrieval_nugget_coverage.py` — typed records, prompt
  rendering, response validation, hashing, scoring, orchestration, `argparse`
  entry point, and `if __name__ == "__main__"`. Every comparable post-run tool in
  `code/trec_rag/` keeps its CLI in the same module; a separate `_cli` module
  would be a new, unjustified convention.
- `code/tests/test_retrieval_nugget_coverage.py` — all tests, using injected
  fake backends.
- `code/tests/test_retrieval_nugget_coverage_skill.py` — text assertions on the
  new skill route, following `code/tests/test_competition_debug_report_skill.py`.
- `.agents/skills/trec-rag-competition-debug-report/SKILL.md` — the added route.
- `code/trec_rag/README.md` — a short section describing inputs, outputs, and
  validation, as `AGENTS.md` requires beside new code.

Invocation:

```bash
.venv/bin/python -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest OUTPUT_DIR/generation_handoff_manifest.json \
  --topic TOPIC_ID
```

The module may import the existing OpenRouter transport and completion helpers
from `trec_rag.facet_extraction`, as `canonical_nuggets.py` already does.
Evaluator schemas, prompts, scoring, and persistence stay owned here.

### Core API

Path-independent, so the scoring core is reusable and testable without any
competition artifact:

```python
def evaluate_nugget_coverage(
    *,
    narrative: str,
    nuggets: Sequence[CoverageNugget],   # (nugget_id, text), input order preserved
    planner: CoveragePlannerBackend,     # Protocol: plan(narrative) -> BackendReply
    judge: CoverageJudgeBackend,         # Protocol: judge(payload) -> BackendReply
    identity: EvaluatorIdentity,
) -> CoverageReport: ...
```

A thin adapter turns one `GenerationTopic` into `(narrative, nuggets)`. The
adapter is the only part that knows about the handoff.

## Planner Contract

The planner receives the narrative and fixed instructions. Nothing else — no
nuggets, no topic ID, no pipeline vocabulary.

Response schema `retrieval_nugget_plan_v1`:

```json
{
  "schema_version": "retrieval_nugget_plan_v1",
  "facets": [
    {
      "title": "…",
      "obligations": [
        {
          "requirement": "…",
          "support_test": "…",
          "kind": "required_explicit",
          "narrative_spans": ["…"]
        }
      ]
    }
  ],
  "unmapped_narrative_spans": ["…"]
}
```

**Identifiers are assigned locally, not by the model.** After validation the
evaluator numbers facets `f001…` and obligations `f001-o001…` by position. This
removes duplicate-ID, format, and ordering failure classes entirely.

Validation, all fail-closed:

- Exact key sets at every level; no extra or missing keys.
- 1–12 facets; 1–8 obligations per facet; at most 40 obligations in total.
- `title`, `requirement`, `support_test`: non-empty, stripped, no control
  characters, at most 300 characters.
- `kind` is `required_explicit` or `supplemental_inferred`.
- `required_explicit` requires at least one `narrative_span`, and every span must
  be a non-empty exact substring of the narrative.
- `supplemental_inferred` requires `narrative_spans` to be empty. A supplemental
  obligation may never claim explicit narrative support.
- Every entry in `unmapped_narrative_spans` must be an exact narrative substring;
  the list may be empty.
- At least one `required_explicit` obligation must exist, otherwise the plan is
  rejected as unscoreable rather than reported as a zero.

There is exactly one planner call. After validation the evaluator serializes the
plan with locally assigned IDs to canonical JSON, takes its SHA-256, and freezes
it. Resume never replaces a valid frozen plan with a new completion.

`unmapped_narrative_spans` is carried into the report as a planner-coverage
diagnostic. It never affects the score.

## Judge Contract

The judge receives the narrative, the complete frozen plan with its assigned IDs,
and every canonical retrieval nugget as `alias: text` in handoff order. It
receives no retrieval, passage, or pipeline metadata.

Response schema `retrieval_nugget_judgment_v1`:

```json
{
  "schema_version": "retrieval_nugget_judgment_v1",
  "judgments": [
    {
      "obligation_id": "f001-o001",
      "label": "partial",
      "supporting_nugget_aliases": ["n003"],
      "missing_elements": "…"
    }
  ]
}
```

Validation, all fail-closed:

- Exactly one judgment per obligation ID: no missing, no duplicate, no unknown.
- `label` is `full`, `partial`, or `unsupported`.
- Aliases are `enum`-constrained to the supplied set, unique within a judgment.
- `full` and `partial` require at least one supporting alias.
- `unsupported` requires an empty alias list.
- `full` requires `missing_elements` to be the empty string; `partial` and
  `unsupported` require non-empty `missing_elements` of at most 300 characters,
  stripped and free of control characters.

Version 1 is one request with a 1,000,000-byte ceiling on the rendered payload,
matching `MAX_CANONICAL_REQUEST_BYTES` in `canonical_nuggets.py`. If the nuggets
do not fit, the evaluator fails with an explicit size error. It never truncates,
ranks, samples, or shards silently. Sharding is a later extension only if real
inputs exceed the bound.

## Scoring

Label values: `full = 1.0`, `partial = 0.5`, `unsupported = 0.0`.

A facet is a **required facet** if it contains at least one required obligation.
Facet-level kinds do not exist; requiredness lives only on obligations, so there
is no way for a facet kind and its obligations' kinds to contradict each other.

Primary metric — **required coverage**:

1. For each required facet, average the values of its required obligations.
2. Average those facet scores equally.

Macro-averaging over facets stops a facet the planner happened to split finely
from dominating the topic score.

Also reported:

- `strict_full_rate` — required obligations labeled `full` divided by all
  required obligations.
- `supplemental_coverage` — flat average over every supplemental obligation
  across all facets; `null` when there are none. Never mixed into required
  coverage.
- Label counts, per-facet scores, and every per-obligation judgment with its
  resolved supporting `claim_id`s.
- `unmapped_narrative_spans` from the plan.
- Canonical retrieval nuggets cited by no judgment, as a count and alias list.

Uncited nuggets are a diagnostic only and never reduce coverage — an answer
obligation set is not required to consume everything retrieval found. Scores are
computed over the plan's fixed order and stored at full float precision in the
artifact; the stdout receipt rounds to four decimals.

## Persistence, Resume, and Receipts

The private work directory holds canonical JSON artifacts, written in this order:

1. `input.json` — evaluator schema version, topic ID, handoff `manifest_sha256`,
   `narrative_sha256`, and the alias table of `(alias, claim_id, text_sha256)`.
   **No narrative or nugget text is written here.**
2. `plan.json` — the frozen plan with assigned IDs, planner identity, request
   SHA-256, and `plan_sha256`. This artifact does contain requirement text and
   the narrative spans the planner quoted.
3. `judgments.json` — validated judgments, judge identity, and request SHA-256.
4. `report.json` — scores, per-obligation rows, and diagnostics.
5. `manifest.json` — written last, binding every artifact SHA-256, the evaluator
   schema and prompt versions, model identities, the truthful
   `completed_stages` count, and safe provider metadata only. The receipt keeps
   separate current-run `hosted_calls` and `reused_stages` values.

Manifest-last ordering matches the retrieval runner's receipt convention, so a
present manifest means every earlier artifact is complete.

`create` refuses a non-empty work directory. `resume` re-validates every present
artifact and reuses only complete, valid stages. Any change to the narrative,
the nugget set or its order, a prompt contract, a model identity, or a schema
version invalidates the matching request identity and fails rather than mixing
states. There is no `overwrite` mode in version 1; deleting the directory is an
explicit user action.

The authenticated narrative and canonical nugget text are retained byte-for-byte,
including multiline, surrounding whitespace, and legitimate Unicode format
characters such as U+200D (ZWJ), and their SHA-256 values are computed over
those exact strings. Empty or whitespace-only text and unsafe C0/Cc control
characters are rejected at the handoff boundary; newline, carriage return, and
tab are permitted. Model-output text remains strict: it must be trimmed,
non-empty, and free of controls where the planner or judge contract requires
it. Topic IDs are rejected before any work-directory path is resolved or
created when they contain path separators, absolute-path syntax, or `.`/`..`.

Resume performs an artifact-order preflight before loading a stage or invoking a
backend. A later artifact cannot exist while an earlier required artifact is
missing, and persisted nugget IDs are checked against the authenticated handoff
before judge validation; either condition is a persistence error. Before a
persisted planner or judge stage is reused, its request digest must be a
lowercase 64-hex SHA-256 and must equal the digest recomputed from the current
v3/v2 serialized request contract, respectively.

Without `--allow-hosted-calls` the run is cache-only: if a required stage is
missing it fails and names which stages would need a hosted call, without making
one.

The CLI prints one small JSON receipt with status, topic ID, artifact directory,
artifact hashes, actual hosted-call and current-run reuse counts, obligation and
nugget counts, and the aggregate scores. It never prints narrative text, nugget
text, requirement text, narrative spans, or credentials. The manifest uses the
truthful `completed_stages` count for the sealed planner/judge stages; the
receipt's `hosted_calls` counts only calls made by the current invocation and
`reused_stages` lists only stages reused by that invocation. Cache-only CLI
errors safely identify exactly the missing planner stage, judge stage, or both.

Both structured OpenRouter requests set `provider.require_parameters=true`,
`provider.data_collection="deny"`, `reasoning.enabled=false`,
`temperature=0`, `seed=0`, and `stream=false`; planner and judge prompt
identities are bumped whenever these request or prompt contracts change. The
completion budget is 8192 tokens to cover the bounded planner output. The
provider metadata stored in memory and persistence is filtered through the same
allowlist, and semantic retries remain zero.

## Failure Handling

Fail closed, with a typed `NuggetCoverageError`, on:

- an unreadable, unauthenticated, or schema-mismatched handoff manifest;
- an unknown topic ID;
- a topic with an empty narrative or zero canonical retrieval nuggets;
- malformed, truncated, or extra model output, at either stage;
- a planner span that is not an exact narrative substring;
- a supplemental obligation carrying narrative spans;
- a plan with no required obligation;
- a missing, duplicated, or unknown obligation judgment;
- an unknown supporting alias, or a label/alias/`missing_elements` contradiction;
- a rendered request over the byte ceiling;
- any artifact hash or resume-identity mismatch;
- credential material reflected in a provider response.

The CLI catches this error, prints a safe JSON error object with the failing
stage and reason, and exits non-zero.

Transport-level retries follow the repository's existing bounded transient-error
policy. There are **no semantic repair calls**: a malformed completion fails and
leaves its safe receipt for diagnosis. `AGENTS.md` requires transport and
semantic retry limits to stay separate, and version 1 sets the semantic limit to
zero.

## Skill Routing and Authorization

The existing `trec-rag-competition-debug-report` skill already advertises itself
for requests to "evaluate a completed run", so a request for retrieval nugget
coverage would otherwise route into RAGDoll evaluation. The skill therefore gains
one short, year-neutral retrieval-nugget-coverage section — no example topic IDs,
dataset paths, gold, or qrels — that routes only to this CLI. This skill edit is
the only non-code surface in scope.

Default invocation is cache-only and makes no hosted call. When planner or judge
state is missing, the skill must state the provider, the two model identities,
the maximum of two calls, and the payload categories — one narrative, the derived
frozen plan, and canonical retrieval nugget text — then ask once for explicit
evaluation authorization, unless the user's own request already explicitly asked
to evaluate nugget coverage.

That authorization covers only the planner and judge calls for the named topic.
It does not authorize retrieval, reranking, generation, passage egress, a
different provider or model, other topics, publication, or serving.

## Testing

All tests use injected fake backends. None needs network access, secrets, a GPU,
or corpus text.

1. Handoff adapter: narrative and ordered nuggets extracted from a fixture
   `GenerationTopic`; alias assignment is stable; empty `claim_hints` fails.
2. Planner validation: exact-key enforcement, bounds, exact-substring spans,
   supplemental-with-spans rejection, and the no-required-obligation rejection.
3. Local ID assignment and frozen-plan hashing are deterministic and
   order-stable.
4. Judge validation: completeness, duplicates, unknown obligation IDs, unknown
   aliases, and every label/alias/`missing_elements` contradiction.
5. Scoring: required-facet macro average, strict full rate, separate supplemental
   score, `null` supplemental when absent, and uncited-nugget diagnostics.
6. Request-size ceiling produces an explicit error and no hosted call.
7. `create` refuses an existing namespace; `resume` reproduces the report with
   zero backend calls; a changed narrative, nugget, prompt version, or model
   identity is rejected on resume.
8. Cache-only default makes zero backend calls and names the missing stages.
9. Receipt redaction: no narrative, nugget, requirement, span, or credential text
   on stdout; artifacts land under the private work directory.
10. Skill text: the new route contains no year-specific data, gold, or qrels, and
    states its hosted-call authorization requirement.

One fixture topic drives the CLI end to end with fake planner and judge backends.

## Acceptance Criteria

Version 1 is complete when:

- one command evaluates one topic's canonical retrieval nuggets from a sealed
  handoff manifest;
- the default run makes zero hosted calls and a resumed run reuses valid stages;
- every obligation has exactly one validated coverage judgment;
- required coverage, strict full rate, and supplemental coverage reproduce
  exactly from the persisted judgments;
- the manifest binds the handoff hash, narrative hash, ordered nugget hashes,
  prompts, models, and schema versions;
- receipts leak no narrative, nugget, requirement, or span text;
- the report artifact carries the core assumption and the "What This Score Is
  Not" limits;
- the skill route is year-neutral and calls only this CLI;
- `code/trec_rag/README.md` documents inputs, outputs, and validation; and
- `.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py
  code/tests/test_retrieval_nugget_coverage_skill.py` passes with no hosted
  service.

## Deferred

Not now, and not to be anticipated in version 1 code:

- Passage-level verification of canonical retrieval nuggets.
- Raw-retrieval diagnosis that separates retrieval, selection, and
  canonicalization failures.
- Judge request sharding.
- Cross-topic aggregation or an HTML report.
- A calibration harness that scores this evaluator against development gold.

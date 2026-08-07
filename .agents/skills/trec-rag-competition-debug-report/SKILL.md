---
name: trec-rag-competition-debug-report
description: Use when a user asks to inspect, explain, visualize, audit, debug, or evaluate a completed TREC RAG competition retrieval or RAG run.
---

# TREC RAG Competition Debug Report

Use the repository's post-run report CLI as the single report implementation. This workflow reads completed artifacts, creates a private standalone HTML explanation, and — once explicitly authorized — evaluates every applicable metric through the repository-pinned RAGDoll integration. Requests to execute an incomplete run belong to the competition retrieval or RAG workflow.

Never run retrieval, reranking, answer generation, unrelated models, or unrelated hosted APIs. Inputs, the raw report, judgments, raw provider events, and metrics must remain private and outside git: never copy, serve, or publish them, and never expose them through a public listener.

Every command below runs from the environment `code/tools/setup_env.sh` produces. The pinned `ragdoll/` submodule is a declared dependency of this repository, so no `PYTHONPATH` or other environment override is needed or supported. If a `ragdoll` import fails, the submodule is uninitialized or the environment is unsynced: run `git submodule update --init --recursive` and re-run the setup script rather than adding a path override.

## Explicit Evaluation Authorization

RAGDoll evaluation sends private answer-statement and authenticated citation text to a hosted judge. Only an explicit invocation authorizes that egress. Exactly these count as explicit evaluation authorization:

- The user names this skill — `$trec-rag-competition-debug-report`, `/trec-rag-competition-debug-report`, or "Competition Debug Report" — for a completed run.
- The user asks, in their own words, to evaluate, score, judge, or run RAGDoll on a completed run.

Nothing else authorizes it. A request that only asks to inspect, explain, visualize, audit, or debug a run — including one this skill was matched to automatically — authorizes local reading and the private HTML report and nothing more. Produce the report, state that evaluation would send statement and citation text to the pinned hosted judge, and ask once before any judge call.

With explicit evaluation authorization, that one invocation covers the whole evaluation. Send only the prepared answer-statement and authenticated selected-evidence citation text that RAGDoll requires, to the repository-pinned provider and model, and do not ask again between calls, batches, or resumes. State the provider, model, task count, and payload categories before calling, without revealing secrets. If the user says quota or authentication was restored, retry the same resumable RAGDoll command without asking again.

Standing authorization stops at that boundary. Ask first before retrieval, reranking, answer generation, a different provider or model, arbitrary corpus uploads, topics outside the user's stated scope, publication, serving, or any sharing change.

## Workflow

1. Use the supplied repository root, or walk upward from the current directory to the first root with the repository markers `pyproject.toml` and `code/trec_rag/competition_debug_report.py`; if none exists, ask for the repository path. Use supplied config paths. Otherwise, use the sole compatible completed retrieval run and, when requested, its matching standard RAG config. If multiple completed retrieval runs remain genuinely ambiguous, list them concisely and ask one short question for the retrieval-config path; do not guess, run every candidate, or ask for information that repository evidence already resolves.
2. Confirm the configured run is complete by checking that its sealed retrieval export exists and that the configured RAG output exists when a RAG config is included. If either requested run is incomplete, stop and identify the missing post-run artifact.
3. Invoke only the repository CLI. Pass each requested topic with a separate `--topic`; omit topic flags to include all exported topics.

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --topic TOPIC_ID
```

Omit `--rag-config` for retrieval-only reports. Omit `--topic` when the user requests every topic. Let the CLI choose its private default output unless the user explicitly requests an in-repository HTML path.

## Examples

Retrieval-only:

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml
```

Retrieval plus RAG:

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --rag-config configs/rag26_competition_rag_gpt_sol_v2.yaml
```

4. Read the single JSON receipt from stdout. Confirm it contains `schema_version`, an absolute `output_path`, `topic_ids`, `rag_included`, and `source_sha256s`, and that its topics and RAG status match the request.

## Evaluated Friendly Report

The full response-and-judgment report is one repository CLI. Both configs are required; the retrieval-only command above is a different, raw artifact and must not be presented as this report.

```bash
.venv/bin/python \
  -m trec_rag.competition_evaluation_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --topic TOPIC_ID \
  --work-dir WORK_DIR \
  --output REPORT_HTML
```

Omit `--topic` to evaluate every completed topic in declared order. Add `--qrels` or `--gold-nuggets` only when they match the selected topics. Everything else resolves from the configs and the authenticated manifests they name; never pass or hardcode an output directory, topic count, or expected label.

Without `--run-judge` the command makes no hosted calls: it reuses validated cache entries and reports anything unjudged as explicitly unavailable. Pass `--run-judge` only with explicit evaluation authorization, and only to fill validated cache misses.

### Cache first, then probe one task, then resume

Always start without `--run-judge`. Judging may already be complete from an earlier run, and a probe would then be a needless hosted call.

```bash
.venv/bin/python \
  -m trec_rag.competition_evaluation_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --work-dir WORK_DIR --cache-dir CACHE_DIR \
  --output REPORT_HTML
```

If that receipt reports `fully_judged: true`, you are done: `hosted_calls` is `0` and the report is complete. Do not probe.

If judgments are missing and explicit evaluation authorization applies, record the cache-only receipt — specifically `completed_judgments` and `missing_judgments` — then probe exactly one hosted call against the **same** `--cache-dir`:

```bash
.venv/bin/python \
  -m trec_rag.competition_evaluation_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --work-dir WORK_DIR --cache-dir CACHE_DIR \
  --output REPORT_HTML \
  --run-judge --judge-limit 1
```

The probe succeeded when the receipt shows `hosted_calls: 1`, `failed_judgments: 0`, `conflicting_judgments: 0`, and `completed_judgments` **greater than the cache-only run's** value. Do not require `completed_judgments` to equal 1: earlier cache hits may already be counted, and because the cache is keyed by the effective judge request, one newly cached identity can legitimately complete more than one task at once.

If the probe reports `failed_judgments` or `conflicting_judgments` above `0`, or `completed_judgments` did not increase, stop and report it. The limit means the remaining quota is untouched; issuing the unlimited run without fixing the cause would spend it on the same failure.

If the probe receipt is already `fully_judged: true`, finish there. Otherwise rerun the identical command against the same `--cache-dir` with `--judge-limit` removed:

```bash
.venv/bin/python \
  -m trec_rag.competition_evaluation_report \
  --retrieval-config RETRIEVAL_CONFIG \
  --rag-config RAG_CONFIG \
  --work-dir WORK_DIR --cache-dir CACHE_DIR \
  --output REPORT_HTML \
  --run-judge
```

Require `reused_from_cache >= 1` (the probe was reused, not re-sent), `failed_judgments: 0`, `conflicting_judgments: 0`, and `hosted_calls` equal to the remaining misses rather than the full task count. `--judge-limit` requires `--run-judge` and must be a positive number. Do not run concurrent judging commands against the same cache directory.

The command writes a private `evaluation_manifest.json` bundle and renders the HTML from it. Read the receipt and confirm `hosted_calls`, the cache counters, the topic order, and `fully_judged` before reporting results. A complete cache hit must show `hosted_calls: 0`.

The judge cache is shared across runs and keyed only by the effective judge request, so a differently named run with the same statements, evidence, prompt contract, RAGDoll revision, and judge settings reuses it. Cache entries are created only by this workflow after a completed current judge call. Do not import or attest legacy judgment files.

Nugget coverage stays unavailable in this workflow even when gold nuggets are supplied, because it also requires completed nugget assignments, which this command does not produce. Supplying `--gold-nuggets` only sharpens the recorded reason.

## Retrieval Nugget Coverage

Use the year-neutral Retrieval Nugget Coverage evaluator for one narrative and
its ordered canonical retrieval nugget text from an authenticated handoff. It
is a separate diagnostic from the completed-run report above: it never runs
retrieval, reranking, or generation, and it never receives selected passages.

The default is cache-only and makes zero hosted calls. Cache-only resume makes
zero hosted calls; the `resume` mode, after validating cached planner and judge
stages, may locally publish missing derived `report.json`/`manifest.json`
artifacts. Only a fresh cache-only create is write-free. For a fresh namespace,
start with exactly this read-only `create` command; it writes no state and
reports which planner or judge stages are missing:

```bash
.venv/bin/python \
  -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest HANDOFF_MANIFEST \
  --topic TOPIC_ID
```

If the user's request explicitly asks to evaluate, score, or judge retrieval
nugget coverage, it is already explicit authorization for only these planner
and judge calls for the named topic, OpenRouter provider, and stated (or
default) model identities; do not ask again. A request that only asks to
inspect, explain, debug, or audit does not authorize hosted calls, so ask once
when a stage is missing. Before egress in an authorized path, state the
provider (OpenRouter), the planner and judge model identities (defaults are
`openai/gpt-5.6-sol` for each, or the exact `--planner-model` and
`--judge-model` overrides), and that the maximum of two hosted calls is one
narrative-only planner call followed by one all-nugget judge call. The payload
categories are the one narrative, the derived frozen plan, and canonical
retrieval nugget text. Then rerun the identical command with the one opt-in
flag:

```bash
.venv/bin/python \
  -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest HANDOFF_MANIFEST \
  --topic TOPIC_ID \
  --allow-hosted-calls
```

For an existing or partial work directory, use the same two-command
cache-first sequence with `--work-dir WORK_DIR --mode resume` on both commands:

```bash
.venv/bin/python \
  -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest HANDOFF_MANIFEST \
  --topic TOPIC_ID \
  --work-dir WORK_DIR \
  --mode resume

.venv/bin/python \
  -m trec_rag.retrieval_nugget_coverage \
  --handoff-manifest HANDOFF_MANIFEST \
  --topic TOPIC_ID \
  --work-dir WORK_DIR \
  --mode resume \
  --allow-hosted-calls
```

The second command in either branch is identical to the first except for
`--allow-hosted-calls`.

That authorization covers only this named topic, provider, and the stated
planner/judge identities. Never use this route to run retrieval, reranking,
generation, passage egress, another topic, another model or provider, or any
publication or serving action. Keep the handoff, private work directory, model
responses, and report outside git and do not expose them through a listener.

## RAGDoll Evaluation

Run this section only with explicit evaluation authorization.

5. When the report includes a completed RAG run, inspect the pinned `ragdoll` submodule and the repository's `trec_rag.ragdoll_io` integration. Record the RAGDoll version and commit. Reuse only sidecars whose source hashes, topic scope, handoff identity, and generation identity validate; otherwise derive fresh sidecars in a private temporary directory.
6. Inventory labels before scoring:
   - Compute qrels-based retrieval metrics only with matching target-topic qrels.
   - Compute nugget coverage only with released target-topic gold nuggets and complete assignments. Never treat generated canonical claims as gold.
   - Compute citation support from the authenticated selected-evidence handoff. Do not stop after materializing tasks.
   - Run rubric or Arena metrics only when their required criteria, systems, and judgments exist.
7. Materialize the exact citation-support tasks and verify one task per citation. If complete validated judgments do not already exist, run the pinned `ragdoll support judge` privately. Probe one task first, then resume the remaining tasks into the same output. Preserve raw events and cache privately. A completed row must contain exactly one `FS`, `PS`, or `NS` label.
8. Rerun the repository adapter with the judgments so it fails closed on missing, duplicate, altered, or out-of-scope tasks. Assemble assignments, run `ragdoll support metrics`, and report all supported per-topic values plus macro aggregates. Leave metrics requiring absent inputs explicitly unavailable; never substitute zero.
9. Validate hashes, exact topic order, judgment counts, label counts, metric bounds, and the raw report receipt. Return absolute private artifact paths and exact commands. Clearly separate retrieval metrics from answer/RAG metrics.

## Friendly Derivative

The raw debug HTML contains private corpus text, generated claims, answers, and document identifiers and must never be served. If the user explicitly requests a browser view, create a separate privacy-reviewed derivative that omits raw corpus passages, document identifiers, credentials, and provider events; serve it only through an already-authorized private tailnet mapping.

`trec_rag.competition_evaluation_report` is the authoritative renderer. Do not hand-write a one-off renderer or patch a previously rendered page in place. It already fails closed:

- Reject duplicate identities in every identity-keyed input — submissions, per-topic diagnostics, metric rows, and judgments — instead of overwriting them.
- Fail on a citation that does not resolve to a listed reference. Never substitute a placeholder label.
- Generate every count, aggregate, metric, and status claim in the page from the same validated inputs on each run, so changed judgments can never leave a summary stale. Derive structural claims such as reference coverage; do not publish unchecked literals.
- Read precision and recall as separate fields. Show one value only after confirming they are equal.
- Scan the assembled page for document identifiers, corpus text, credentials, provider events, temporary paths, and long hashes before writing it.

Present it accessibly:

- Display each citation's `FS`, `PS`, or `NS` label inline beside the response text while retaining a separate detailed judgment list. Label multiple citations to the same statement independently.
- Give inline labels real text, including a visually hidden spelled-out phrase, rather than an `aria-label` on a generic span.
- Keep heading levels contiguous so each topic section is reachable by heading navigation.
- Meet WCAG AA contrast for badges in light and dark mode, keep badge text readable, and never rely on color alone.
- Reset text and border colors as well as backgrounds in print styles so printing from dark mode stays legible.
- Collapse long subnarrative lists by default so the response and judgment results remain easy to reach.
- Define candidate documents, submitted documents, evidence passages, answer references, and statement–citation judgments as distinct units. State that these counts are pipeline diagnostics, not relevance judgments.

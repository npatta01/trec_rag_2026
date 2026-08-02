# Handover: `claude/ragdoll-eval-spike` (PR #33)

Written 2026-08-02. Deadline **August 8th** (the current vendored skill says 8th; an older
copy said 7th, so plan against the 7th if you want slack).

## State

- Branch `claude/ragdoll-eval-spike`, 24 commits ahead of `origin/master`, pushed, working tree
  clean. PR #33 is OPEN, MERGEABLE, CLEAN. `origin/master` (including the evidence-bundle
  PR #34) is already merged in.
- Tests: **840 passed, 19 skipped**, one pre-existing failure.
  ```bash
  PYTHONPATH=code .venv/bin/python -m pytest -q \
    --ignore=code/tests/tracing/test_phoenix_export.py \
    --ignore=code/tests/experiments/organizer_pi/test_cli.py
  ```
  The excluded two, plus `test_package_direction`, fail only because
  `code/tools/setup_env.sh` does not sync the `observability` dependency group, so
  `opentelemetry` is absent. Pre-existing on master, unrelated to this branch.

## What this branch does

**1. Answer-quality evaluation, which the repo never had.** `castorini/RAGDoll` is added as the
`ragdoll` submodule, pinned at `1f06719`, driven through the already-installed `pi` CLI
(v0.83.0 at `~/.nvm/versions/node/v26.4.0/bin/pi`, not on the default PATH). Judge is
`openrouter/deepseek/deepseek-v4-flash`, which costs cents.

```bash
export PATH="/home/npatta01/.nvm/versions/node/v26.4.0/bin:$PATH"
export PI_TELEMETRY=0
cd ragdoll && uv run ragdoll nuggetizer eval \
  --nuggets-file <gold-reshaped>.jsonl --answers-file <answers>.jsonl \
  --output-dir <out> --model openrouter/deepseek/deepseek-v4-flash --thinking minimal
```

`code/trec_rag/ragdoll_io.py` derives RAGDoll inputs from a submission JSONL.
`code/trec_rag/dev_rag_inputs.py` builds dev-topic inputs from archived Pyserini responses.
Both are evaluation-only and cannot affect a submission. See `code/trec_rag/README.md`.

**2. A measurably better generation config.** `configs/rag26_competition_rag_gpt_sol_v1.yaml`
now selects `prompt_profile: focused_citations_tail` and `max_tokens: 12000`.

| | coverage (strict_vital) | strict_all | wp_first | hard precision |
|---|---|---|---|---|
| what a run used at PR open | 0.588 | 0.569 | 0.692 | 0.481 |
| current config | **0.630** | **0.609** | **0.747** | **0.586** |

Four development topics (58, 72, 144, 200), released gold nuggets, one judge throughout.
Full record in `reports/experiments/dev4_paired_prompt_profile_v1/`.

**3. Deterministic post-processors** in `competition_rag.py`, applied between
`build_submission_record` and validation: `trim_to_word_limit` then
`normalize_generated_record`. These enforce the organizer word cap and all-cited rule in code
rather than asking the prompt for them, which took uncited-reference generation failures to
zero.

## Key findings worth not relitigating

- **Answer structure dominates citation support.** Full Support rate by citations per object:
  72.5% at one, 27.9% at two, 11.1% at three. The old "cite every reference" instruction forced
  two or three citations onto nearly every object and manufactured the errors.
- **Instruction placement matters, and proximity beats channel.** With 100 documents inserted,
  the contract sits ~120k tokens above the generation point. Restating it *after* the question
  gains Sol +0.042 and DeepSeek +0.050 strict_vital. Moving it into the *system message*
  instead measured **worse** (0.509 vs 0.578).
- **DeepSeek `deepseek-v4-flash-0731` is the iteration vehicle, Sol is the submission.**
  $0.0135 vs $0.7115 per topic. On the same prompt Sol leads coverage by 0.081. DeepSeek does
  beat Sol on citation support (hard precision 0.678 vs 0.481), so it is not uniformly worse.
  A second config exists at `configs/rag26_competition_rag_deepseek_v1.yaml` in case the track
  accepts multiple runs; the guidelines state no limit but also do not specify upload
  procedures, so that is unverified.
- **Rejected, do not retry:** DeepSeek's documented in-prompt JSON example (compliance 59% →
  26%); `structured_output: json_object` for DeepSeek (3 of 4 topics failed with malformed
  shapes, so `require_parameters` is load-bearing); `effort: max` (no better than `high`);
  Gemini 3 Flash and Kimi as generators (both emitted docids outside the retrieval pool);
  GLM 4.7 (null content, 0/4).
- Each prompt profile in `PROMPT_PROFILES` carries its measured figure and status in a comment.

## The blocking item: the exhaustive audit came back NO-GO with 63 findings

Captured in `docs/superpowers/reviews/2026-08-02-exhaustive-codex-audit.md`. **Only findings
57-63 were captured**; 1-56 scrolled out of stdout and must be recovered by re-running:

```bash
cd <worktree>
codex exec --skip-git-repo-check "Exhaustive final audit. Review git diff origin/master...HEAD.
Do not stop at the first two or three findings; I want a COMPLETE enumeration of every defect
in the areas this branch touched so they can all be fixed in one pass. Prefer a long exhaustive
list over a short prioritised one, include low severity, and say explicitly if an area is clean.
Audit exhaustively: (A) competition_rag.py submission path - prompt profile registry,
render_prompt, system_prompt_for, complete_json request construction and structured_output modes
and finish_reason allowlist and retries and redaction, trim_to_word_limit and
normalize_generated_record and their interaction, _generation_identity and
_enforce_generation_identity, load_queries and load_trec_run and load_documents including the
topic_ids binding, _saved_record and resume and the atomic publish gate and locking,
validate_submission_record and _validate_generated_submission_record against
trec-rag-data/trec-rag-2026/baselines/rag/code/ragnarok_style_ag.py; (B) ragdoll_io.py;
(C) dev_rag_inputs.py; (D) both configs in configs/; (E) the tests themselves, including
assertions that do not test what they claim and fixtures that mask defects. For each finding
give file:line, severity, a concrete failing scenario, and the minimal fix. Then a final
go/no-go."
```

**The most important known finding, number 57: the topic-binding fix in `af6fc59` is
insufficient.** `load_documents(topic_ids=...)` excludes rows for unselected topics but does not
bind a document to the topic that ranked it. Two *selected* topics sharing a docid still allow
one topic's evidence to ground the other's answer. The fix is a per-topic `docid -> text`
mapping rather than a flat map. This is the third revision of the same defect, so test it with
two selected topics sharing a docid, not one selected and one unselected.

Other captured findings: overwrite deletes state before validating inputs (medium);
`ragdoll_io` trusts `metadata.narrative` as the question rather than the canonical topics file
(medium); rank gaps accepted despite the stated strict contract (low); a malformed unselected
topic aborts a selected-topic run (low); `OpenRouterJsonGenerator` accepts
`transport_max_attempts=0` and reaches an "unreachable" assertion (low); encoded-secret test
coverage weaker than the test names claim (low).

## The pattern to watch

Five review rounds, each found real defects, and **four of five were introduced by the previous
round's fix**:

1. Post-processors removed validation that already existed (invalid citations silently pruned).
2. The resume identity omitted `max_tokens`, right as `max_tokens` was raised.
3. The `finish_reason` check was a denylist rejecting only `length`, so `error` and
   `content_filter` were published.
4. Fixing that changed the acceptance policy without bumping `identity_version`.

Every guard added to `competition_rag.py` has interacted with an existing one. The file is
**+444 lines** on the only path that produces a submission. Treat further changes there with
suspicion and prefer verification over addition.

## Open work, in the order I would do it

1. **Recover findings 1-56 by re-running the audit, then fix all 63 in one pass**, starting
   with finding 57. Then re-run the two-topic smoke below, because those fixes touch
   `load_documents`. Do not merge before this is done: the current verdict is NO-GO.
2. **Re-run the two-topic smoke** and confirm it still passes the organizers' own validator:
   ```bash
   PYTHONPATH=code .venv/bin/python -m trec_rag.competition_rag \
     --config configs/local/smoke_b80_sol_tail.yaml
   ```
   It points at `/home/npatta01/.codex/worktrees/3822/trec_rag_2026/outputs/facet-deepseek-b80-v1-two-topic-smoke/`.
   Validate with `trec-rag-data/trec-rag-2026/baselines/rag/code/ragnarok_style_ag.py`'s
   `validate()`, not only ours. Last run: both records passed, $1.02, 45 and 49 answer objects
   at 1016 and 1000 words.
3. **Merge PR #33** once the audit is clean.
4. **The 119-topic production run has never happened and is the largest unretired risk.**
   Generation has never run past 22 topics. `outputs/facet-deepseek-b40-v1/`, which the
   committed Sol config names, **does not exist** — full-scale facet retrieval was never
   produced. Either run retrieval for all 119 topics, or point the config at the organizers'
   released retrieval, which is verified loadable:
   `trec-rag-data/trec-rag-2026/baselines/retrieval/first_qwen3_8b_listwise_reranked_qwen3_8b_pointwise_top100.trec`
   plus `bm25_climbmix_top1000_with_text.jsonl.zip` (member `bm25_top1000.jsonl`), which loads
   119 topics, 11,900 rows, 11,782 documents, zero missing. The organizers' own RAG baseline
   used exactly that retrieval. This is a strategy decision for Nidhin, not a technical one.
5. Expect **2-4 resume rounds** on a full run; the atomic gate blocks publishing, not progress,
   and every good row persists in `work/rows/`. Never use `overwrite` to recover; only `resume`.

## Standing caveats on every number above

- **Four topics.** Margins of 0.02-0.05 are inside the noise; generation variance alone once
  produced 9 versus 28 answer objects for one config.
- Citations are judged against the evidence each generator read, while the organizers resolve
  references from the index, that is full documents. Judging every arm against full document
  text is the realistic measurement and is untested.
- Automated nugget assignment runs about +0.10 absolute above NIST manual assignment, and
  `strict_vital` is the most fragile metric under full automation. Treat all figures as
  relative.

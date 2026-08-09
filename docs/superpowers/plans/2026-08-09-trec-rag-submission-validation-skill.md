# TREC RAG 2026 Submission Validation Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable skill that validates TREC RAG 2026 Retrieval TSV and RAG JSONL submission artifacts with the correct task-specific validators.

**Architecture:** Create one skill in the `trec-rag-skills` submodule with a concise routing guide and a standard-library Python CLI. Retrieval validation is implemented locally and deterministically; RAG validation delegates to `autojudge-base>=0.4.3`, using a compatible local installation or an isolated `uv` environment. The existing track-guidelines skill remains the canonical specification.

**Tech Stack:** Python 3.12 standard library, `unittest`, `autojudge-base>=0.4.3`, `uv`, Agent Skills metadata.

## Global Constraints

- Create `trec-rag-skills/skills/validate-trec-rag-2026-submissions/`; do not expand the existing track-guidelines skill.
- Require `trec-rag-2026-track-guidelines` as canonical background and do not duplicate its full task specification.
- Never treat a Retrieval TSV as RAG JSONL or present a synthetic empty-answer RAG adapter as authoritative Retrieval validation.
- Retrieval validation must use only the Python standard library and must not require network access.
- RAG validation must use `autojudge-base>=0.4.3` with `--spec rag26`.
- Accept official two-column topics TSV or AutoJudge Request JSONL.
- Do not read qrels, gold nuggets, RAGDoll scores, corpus text, raw provider responses, `.env` files, or credentials.
- Do not impose fixed/equal/max Retrieval depth, padding, topic order, or an unstated run-ID length limit.
- Do not modify submission artifacts; temporary topic conversion must use an automatically cleaned temporary directory.
- Preserve unrelated superproject and submodule changes. Commit only files owned by this work.

---

## File Map

- Create `trec-rag-skills/skills/validate-trec-rag-2026-submissions/SKILL.md`: trigger metadata, validator-routing workflow, quick reference, result interpretation, and common mistakes.
- Create `trec-rag-skills/skills/validate-trec-rag-2026-submissions/agents/openai.yaml`: Codex UI metadata generated from the finished skill.
- Create `trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py`: reusable Retrieval validator, AutoJudge launcher, topics adapter, multi-artifact CLI, and summary/exit-code logic.
- Create `trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests/test_validate_submission.py`: hermetic unit tests and an opt-in real AutoJudge integration test using temporary fixtures.
- Modify `docs/superpowers/plans/2026-08-09-trec-rag-submission-validation-skill.md`: record RED/GREEN skill-test evidence and completed verification commands.

---

### Task 1: Capture the Skill RED Baseline and Initialize the Skill

**Files:**
- Modify: `docs/superpowers/plans/2026-08-09-trec-rag-submission-validation-skill.md`
- Create via initializer: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/SKILL.md`
- Create via initializer: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/agents/openai.yaml`
- Create directory via initializer: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts/`

**Interfaces:**
- Consumes: the real organizer-facing Retrieval TSV, official topics TSV, an available RAG JSONL, and the organizer AutoJudge command from the user-provided screenshot.
- Produces: verbatim baseline behavior recorded under a `## Skill TDD Evidence` section in this plan, plus an initialized but not yet deployed skill directory.

- [x] **Step 1: Run a fresh-context baseline application scenario without the new skill**

  Give one Luna xhigh agent only these inputs: the official topics TSV, one valid full Retrieval TSV, one malformed/incomplete RAG JSONL, and the instruction to validate both quickly using `autojudge-base>=0.4.3`. Require read-only behavior and a concrete verdict. Do not provide the intended routing rule or this design.

- [x] **Step 2: Verify RED and record the exact gap**

  Record the agent's exact commands, verdicts, and rationale in this plan. RED is established if it sends Retrieval TSV to `report_tool`, omits Retrieval rank/score checks, reconstructs an ad hoc validator, cannot use the TSV topics with AutoJudge, or cannot produce one repeatable command for both tasks. If it succeeds fully, record the duplicated work or missing reusable interface that still motivates the deterministic skill.

- [x] **Step 3: Create an implementation branch inside the submodule**

  Run:

  ```bash
  git -C trec-rag-skills switch -c codex/validate-trec-rag-2026-submissions
  git -C trec-rag-skills status --short
  ```

  Expected: the new branch is active and the submodule worktree is clean.

- [x] **Step 4: Initialize the skill with the official scaffold tool**

  Run:

  ```bash
  python /home/npatta01/.codex/skills/.system/skill-creator/scripts/init_skill.py \
    validate-trec-rag-2026-submissions \
    --path trec-rag-skills/skills \
    --resources scripts \
    --interface 'display_name=Validate TREC RAG 2026' \
    --interface 'short_description=Validate Retrieval TSV and RAG JSONL submissions' \
    --interface 'default_prompt=Use $validate-trec-rag-2026-submissions to validate my TREC RAG 2026 Retrieval and RAG submission files.'
  ```

  Expected: the skill folder, template `SKILL.md`, `agents/openai.yaml`, and empty `scripts/` directory are created. Do not treat the template as implemented or commit it yet.

- [x] **Step 5: Commit only the RED evidence update in the superproject**

  ```bash
  git add docs/superpowers/plans/2026-08-09-trec-rag-submission-validation-skill.md
  git diff --cached --check
  git commit -m "Record validation skill baseline"
  ```

---

### Task 2: Implement Retrieval Validation Test-First

**Files:**
- Create: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests/test_validate_submission.py`
- Create: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py`

**Interfaces:**
- Consumes: `Path` objects for a topics file and one Retrieval runfile.
- Produces: `load_topics(path: Path) -> tuple[Topic, ...]`, `validate_retrieval(path: Path, topics: tuple[Topic, ...]) -> ArtifactResult`, immutable `Topic`, `Finding`, and `ArtifactResult` records, and statuses `pass`, `pass-with-warnings`, or `fail`.

- [x] **Step 1: Write failing Retrieval tests before the script exists**

  Create tests that load the future script with `importlib.util.spec_from_file_location`. Use temporary files and begin with this valid variable-depth case:

  ```python
  def test_retrieval_accepts_variable_depth_and_reports_counts(self):
      topics = self.write_topics_tsv(
          ("rag2026-0", "First narrative"),
          ("rag2026-1", "Second narrative"),
      )
      run = self.write_text(
          "run.tsv",
          "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
          "rag2026-0 Q0 shard_00002_2 2 1.0 run-a\n"
          "rag2026-1 Q0 shard_00003_3 1 9.0 run-a\n",
      )

      result = validator.validate_retrieval(run, validator.load_topics(topics))

      self.assertEqual("pass", result.status)
      self.assertEqual(3, result.row_count)
      self.assertEqual(2, result.topic_count)
      self.assertEqual((1, 2), (result.depth_min, result.depth_max))
  ```

  Add table-driven cases for malformed column count, wrong `Q0`, missing and extra topics, rank not starting at 1, rank gaps, duplicate topic/document pair, increasing score, `nan`/`inf`, invalid ClimbMix ID, conflicting run IDs, invalid UTF-8, and an empty run. Add a positive case showing that blank lines are ignored rather than interpreted as submission rows.

- [x] **Step 2: Run the tests and verify RED**

  Run:

  ```bash
  python -m unittest discover \
    -s trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests \
    -p 'test_*.py' -v
  ```

  Expected: FAIL because `scripts/validate_submission.py` does not exist. Fix test-loader mistakes until the failure is specifically the missing production script.

- [x] **Step 3: Implement minimal immutable result types and topics loading**

  Add these stable shapes:

  ```python
  @dataclass(frozen=True)
  class Topic:
      topic_id: str
      narrative: str

  @dataclass(frozen=True)
  class Finding:
      message: str
      line_number: int | None = None
      topic_id: str | None = None

  @dataclass(frozen=True)
  class ArtifactResult:
      task: Literal["retrieval", "rag"]
      path: Path
      status: Literal["pass", "pass-with-warnings", "fail"]
      row_count: int | None
      topic_count: int | None
      depth_min: int | None
      depth_max: int | None
      findings: tuple[Finding, ...]
      detail: str = ""
  ```

  `load_topics` must accept the exact two-column TSV or Request JSONL with `request_id` and `title`, reject duplicates/blank identities, and preserve source order.

- [x] **Step 4: Implement the Retrieval parser and validator**

  Implement `validate_retrieval` as one file snapshot read. Collect violations instead of failing at the first line. Maintain per-topic last rank, last score, and document set. Require `math.isfinite(score)` and `re.fullmatch(r"shard_\d+_\d+", docid)`. Compare the final topic set with the supplied topics and calculate depths only from structurally accepted rows.

- [x] **Step 5: Run Retrieval tests and verify GREEN**

  Run the Task 2 test command. Expected: all Retrieval tests pass with no network access.

- [x] **Step 6: Commit the Retrieval validator slice in the submodule**

  ```bash
  git -C trec-rag-skills add \
    skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
    skills/validate-trec-rag-2026-submissions/tests/test_validate_submission.py
  git -C trec-rag-skills diff --cached --check
  git -C trec-rag-skills commit -m "Validate TREC RAG 2026 retrieval runs"
  ```

---

### Task 3: Add RAG AutoJudge Routing and the Combined CLI Test-First

**Files:**
- Modify: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests/test_validate_submission.py`
- Modify: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py`

**Interfaces:**
- Consumes: repeatable `--retrieval PATH`, repeatable `--rag PATH`, one `--topics PATH`, and optional `--strict-rag`.
- Produces: `Runner = Callable[..., subprocess.CompletedProcess[str]]`, `prepare_autojudge_topics(topics_path: Path, destination: Path) -> Path`, `build_autojudge_command(strict: bool) -> tuple[str, ...]`, `validate_rag(path: Path, topics_path: Path, *, strict: bool, runner: Runner = subprocess.run) -> ArtifactResult`, and `main(argv: Sequence[str] | None = None) -> int`.

- [x] **Step 1: Write failing RAG routing and CLI tests**

  Add a recording runner that returns `subprocess.CompletedProcess` without invoking the network. Verify:

  ```python
  def test_rag_uses_rag26_check_and_converted_tsv_topics(self):
      result = validator.validate_rag(
          self.valid_rag,
          self.topics_tsv,
          strict=False,
          runner=self.recording_runner,
      )

      command = self.recorded_commands[0]
      self.assertIn("autojudge_base.report_tool", command)
      self.assertIn("check", command)
      self.assertIn("--spec", command)
      self.assertIn("rag26", command)
      self.assertNotIn("--strict", command)
      self.assertEqual("pass", result.status)
  ```

  Also test `--strict-rag`, compatible local-package selection, isolated `uv` fallback, missing dependency failure, AutoJudge exit 255, warning detection, passthrough JSONL topics, repeated inputs, combined Retrieval+RAG summary, and nonzero worst-result exit status.

- [x] **Step 2: Run the focused new tests and verify RED**

  Run the Task 2 unittest command. Expected: the existing Retrieval tests pass and new RAG/CLI tests fail because the RAG functions and CLI flags are absent.

- [x] **Step 3: Implement topics conversion and AutoJudge command selection**

  Parse the installed distribution version with `importlib.metadata.version("autojudge-base")`; accept numeric versions at or above `(0, 4, 3)`. When unavailable or older, require `shutil.which("uv")` and construct:

  ```python
  (
      "uv", "run", "--isolated", "--no-project",
      "--with", "autojudge-base>=0.4.3",
      "python", "-m", "autojudge_base.report_tool", "check",
  )
  ```

  Convert TSV topics to Request JSONL in a `TemporaryDirectory`; pass Request JSONL through unchanged after validating its topic identities.

- [x] **Step 4: Implement RAG status mapping and the combined CLI**

  Run AutoJudge once per RAG artifact with `--spec rag26 --topics NORMALIZED_TOPICS` and optional `--strict`. Preserve its stdout/stderr in `ArtifactResult.detail`. Map return code zero plus a `SMELL` block to `pass-with-warnings`, zero without smells to `pass`, and any nonzero result to `fail`.

  The CLI must validate every requested file, print one concise result line per artifact followed by validator detail, and return 1 if any result failed. `argparse` must reject an invocation with neither `--retrieval` nor `--rag`.

- [x] **Step 5: Add and run a real isolated AutoJudge integration test**

  Before changing production behavior for this test, add a `unittest.skipUnless(shutil.which("uv"), "uv required")` case using temporary one-topic fixtures:

  ```python
  valid_report = {
      "metadata": {
          "team_id": "test-team",
          "narrative_id": "rag2026-0",
          "narrative": "First narrative",
          "run_id": "test-run",
          "run_desc": "Validator integration fixture",
      },
      "references": ["shard_00001_1"],
      "answer": [{"text": "Supported fact.", "citations": [0]}],
  }
  ```

  Run the full unittest command. Expected: the fixture returns `pass`, and a second fixture with a changed narrative returns `fail` from real AutoJudge.

- [x] **Step 6: Run the script against real local artifacts**

  ```bash
  python trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
    --topics trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv \
    --retrieval outputs/retrieval-baseline-candidate-core-v1/final-v3/final/runs/narrative/r_output_trec_rag_2026.tsv \
    --rag outputs/rag26_competition_rag_gpt_sol_v2/rag_output_trec_rag_2026.jsonl
  ```

  Expected: Retrieval passes with 119 topics and 4,246 rows; the known incomplete RAG artifact fails. The combined command returns nonzero because the worst result is a failure.

- [x] **Step 7: Commit the RAG/CLI slice in the submodule**

  ```bash
  git -C trec-rag-skills add \
    skills/validate-trec-rag-2026-submissions/scripts/validate_submission.py \
    skills/validate-trec-rag-2026-submissions/tests/test_validate_submission.py
  git -C trec-rag-skills diff --cached --check
  git -C trec-rag-skills commit -m "Validate TREC RAG 2026 RAG reports"
  ```

---

### Task 4: Write and Validate the Skill Instructions

**Files:**
- Modify: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/SKILL.md`
- Modify: `trec-rag-skills/skills/validate-trec-rag-2026-submissions/agents/openai.yaml`

**Interfaces:**
- Consumes: baseline failures from Task 1 and the stable CLI from Tasks 2–3.
- Produces: a discoverable skill under 500 words whose quick-reference commands invoke the bundled script and whose UI metadata names `$validate-trec-rag-2026-submissions`.

- [x] **Step 1: Replace the scaffold with the minimal skill that addresses RED**

  Use this frontmatter trigger:

  ```yaml
  ---
  name: validate-trec-rag-2026-submissions
  description: Use when checking, preflighting, auditing, or diagnosing TREC RAG 2026 Retrieval TSV or RAG JSONL submission artifacts before delivery.
  ---
  ```

  The body must contain: core principle, required `trec-rag-2026-track-guidelines` background, the task-routing rule, one combined command, a quick-reference table, result interpretation, privacy boundaries, repository-specific additive checks, and common mistakes derived from the baseline. Use a positive command recipe rather than a long prohibition list.

- [x] **Step 2: Regenerate UI metadata from the finished skill**

  ```bash
  python /home/npatta01/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions \
    --interface 'display_name=Validate TREC RAG 2026' \
    --interface 'short_description=Validate Retrieval TSV and RAG JSONL submissions' \
    --interface 'default_prompt=Use $validate-trec-rag-2026-submissions to validate my TREC RAG 2026 Retrieval and RAG submission files.'
  ```

- [x] **Step 3: Run structural and content quality checks**

  ```bash
  python /home/npatta01/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions
  wc -w trec-rag-skills/skills/validate-trec-rag-2026-submissions/SKILL.md
  rg -n 'TBD|TODO|FIXME|PLACEHOLDER' \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions
  ```

  Expected: quick validation succeeds, `SKILL.md` is below 500 words, and the placeholder scan has no matches.

- [x] **Step 4: Commit the deployable skill in the submodule**

  ```bash
  git -C trec-rag-skills add \
    skills/validate-trec-rag-2026-submissions/SKILL.md \
    skills/validate-trec-rag-2026-submissions/agents/openai.yaml
  git -C trec-rag-skills diff --cached --check
  git -C trec-rag-skills commit -m "Add TREC RAG 2026 validation skill"
  ```

---

### Task 5: Forward-Test, Review, and Publish the Versioned Result Locally

**Files:**
- Modify if a test exposes a gap: files under `trec-rag-skills/skills/validate-trec-rag-2026-submissions/`
- Modify: `docs/superpowers/plans/2026-08-09-trec-rag-submission-validation-skill.md`
- Modify: `trec-rag-skills` submodule pointer in the superproject index

**Interfaces:**
- Consumes: the finished skill, raw local Retrieval/RAG artifacts, official topics, and all test results.
- Produces: forward-test evidence, independent review findings resolved test-first, a final submodule commit, and one superproject commit recording the submodule pointer and plan evidence.

- [x] **Step 1: Forward-test a fresh agent with the finished skill**

  Give one fresh Luna xhigh agent the skill path, official topics TSV, valid full Retrieval run, and known incomplete RAG output. Require read-only combined validation. Success requires correct task routing, Retrieval pass evidence, RAG failure evidence, no qrels/gold access, and no claim that AutoJudge directly validated Retrieval ranks/scores.

- [x] **Step 2: Record GREEN evidence and close any discovered gap test-first**

  Append the fresh agent's commands and outcome under `## Skill TDD Evidence` in this plan. If it misroutes, overclaims, or accesses prohibited data, add a failing unit/application scenario first, then minimally revise the script or `SKILL.md` and repeat the same forward test.

- [x] **Step 3: Run the complete verification suite fresh**

  ```bash
  python -m unittest discover \
    -s trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests \
    -p 'test_*.py' -v
  python /home/npatta01/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions
  python -m compileall -q \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions/scripts \
    trec-rag-skills/skills/validate-trec-rag-2026-submissions/tests
  git -C trec-rag-skills diff --check
  ```

  Re-run the combined real-artifact command from Task 3 and confirm the expected mixed pass/fail result.

- [x] **Step 4: Request independent code review**

  Dispatch one `sol_reviewer` with ownership limited to reviewing the new skill directory and its tests. Ask for correctness, security/privacy, false pass/fail risk, AutoJudge version handling, exit semantics, and missing verification. Do not send submission contents; provide repository paths only.

- [x] **Step 5: Resolve review findings and commit final submodule fixes**

  For every accepted behavior fix, write and observe a failing test before editing production code. Then run the complete suite and commit only the new skill files:

  ```bash
  git -C trec-rag-skills add skills/validate-trec-rag-2026-submissions
  git -C trec-rag-skills diff --cached --check
  git -C trec-rag-skills commit -m "Harden TREC RAG submission validation"
  ```

  Skip this commit if review requires no changes.

- [x] **Step 6: Record final evidence and commit the superproject pointer**

  Update this plan's checkboxes and evidence with exact command outputs. Then stage only the plan and submodule pointer:

  ```bash
  git add docs/superpowers/plans/2026-08-09-trec-rag-submission-validation-skill.md trec-rag-skills
  git diff --cached --check
  git diff --cached --submodule=log --stat
  git commit -m "Add TREC RAG submission validation skill"
  ```

- [x] **Step 7: Final handoff**

  Report the skill path, supported commands, Retrieval/RAG validation evidence, submodule commit, superproject commit, and any warnings. Do not push, open a PR, or publish artifacts without separate authorization.

## Skill TDD Evidence

### RED baseline — 2026-08-09

A fresh Luna xhigh agent received only the official topics TSV, one valid full Retrieval TSV, one malformed/incomplete RAG JSONL, and the organizer AutoJudge requirement. It remained read-only and reached the correct high-level verdicts, but only by reconstructing two one-off mechanisms:

- It wrote an inline standard-library Retrieval parser to check six fields, `Q0`, exact 119-topic coverage, `shard_*_*` IDs, contiguous ranks, finite/non-increasing scores, and duplicates. That run passed with 4,246 rows and depths 1–121.
- It manually converted the official topics TSV to temporary AutoJudge Request JSONL, then invoked isolated AutoJudge 0.4.4. The RAG file failed with a raw `JSONDecodeError` before submission checks ran.

The baseline therefore demonstrated the target gap without a new skill: correct validation required ad hoc code and manual format adaptation, there was no stable combined command, and dependency/parser failures were not normalized into per-artifact results. The new skill must preserve the correct task routing while making it deterministic and repeatable.

### GREEN forward test — 2026-08-09

A fresh Luna xhigh agent received only the finished skill path and the same three artifact paths. It followed the skill read-only and ran the bundled combined command. The process returned 1 because the native Retrieval check passed with 4,246 rows, all 119 topics, and depths 1–121, while organizer AutoJudge rejected the incomplete RAG JSONL with `JSONDecodeError`. The agent explicitly distinguished the two validators, did not claim AutoJudge checked Retrieval ranks or scores, and reported no access to qrels, gold nuggets, RAGDoll scores, corpus text, provider responses, environment files, or credentials.

The forward test found one presentation gap: the normalized malformed-JSONL finding did not surface AutoJudge's available line and column. A focused regression test was observed failing, then the wrapper was changed to include that location while retaining the complete AutoJudge detail. The focused test passed after the change.

### Final verification and review — 2026-08-09

The final fresh verification run completed with these results:

- `unittest discover`: 35 tests passed, including real isolated AutoJudge acceptance, narrative-mismatch rejection, and exact-whitespace rejection.
- `quick_validate.py`: `Skill is valid!`
- `compileall`: completed successfully using an isolated temporary bytecode cache.
- `git -C trec-rag-skills diff --check`: no errors.
- Real combined preflight: Retrieval `PASS` with 4,246 rows, 119 topics, and depths 1–121; RAG `FAIL` with normalized `JSONDecodeError` at line 1, column 1; expected combined exit status 1.

Independent Sol review initially found Retrieval I/O exceptions that could abort a batch, loss of exact narrative whitespace during TSV conversion, and incomplete compatible-version selection for PEP 440 post/local AutoJudge releases. Each was reproduced with a failing test and fixed. A focused rereview then found and reproduced hyphenated prerelease handling; the comparator and regression matrix were tightened. The final rereview reported no remaining actionable findings. The hardening changes are committed in submodule commit `825573c`.

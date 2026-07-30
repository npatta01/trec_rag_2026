# Tracing Package Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move reusable Phoenix tracing into `trec_rag.tracing` and isolate the runnable organizer Pi reproduction harness under `trec_rag.experiments.organizer_pi` without changing trace schemas or rendering.

**Architecture:** Reusable immutable records, OpenAI presentation, and Phoenix export form a deep `trec_rag.tracing` module. The organizer Pi experiment depends on that module and owns all source-specific input preparation, event parsing, prompt reconstruction, tool schemas, and CLI behavior; the dependency never points back toward experiments.

**Tech Stack:** Python 3.10+, pytest, OpenTelemetry/OpenInference, Arize Phoenix OTEL, setuptools, uv.

## Global Constraints

- `trec_rag.tracing` must not import from `trec_rag.experiments`.
- Move implementations rather than copying them; every record and function has one owner.
- Do not retain compatibility modules at the old top-level paths.
- Preserve trace bundle schemas, span topology, strict JSON behavior, secret handling, receipts, and hosted Phoenix rendering.
- Preserve the Pi harness as runnable experimental code without tracking native or generated private artifacts.
- Importing trace models or OpenAI helpers must not require optional Phoenix libraries.

---

### Task 1: Establish the reusable tracing module

**Files:**
- Create: `code/trec_rag/tracing/__init__.py`
- Move: `code/trec_rag/pi_trace_models.py` to `code/trec_rag/tracing/models.py`
- Move: `code/trec_rag/openai_trace_semantics.py` to `code/trec_rag/tracing/openai_semantics.py`
- Move: `code/trec_rag/phoenix_trace_export.py` to `code/trec_rag/tracing/phoenix_export.py`
- Create: `code/tests/tracing/__init__.py`
- Move: `code/tests/test_openai_trace_semantics.py` to `code/tests/tracing/test_openai_semantics.py`
- Move: `code/tests/test_phoenix_trace_export.py` to `code/tests/tracing/test_phoenix_export.py`
- Create: `code/tests/tracing/test_package_direction.py`

**Interfaces:**
- Produces: `trec_rag.tracing.models.{SpanSpec,TraceSpec,TraceBundle,read_trace_bundle,write_trace_bundle}`.
- Produces: `trec_rag.tracing.openai_semantics` normalization and OpenInference attribute helpers.
- Produces: `trec_rag.tracing.phoenix_export.{PhoenixSettings,ExportReceipt,SecretStr,export_trace_bundle}`.

- [ ] **Step 1: Add a failing dependency-direction test**

```python
from pathlib import Path


def test_reusable_tracing_does_not_import_experiments():
    tracing_root = Path(__file__).parents[2] / "trec_rag" / "tracing"
    sources = "\n".join(path.read_text() for path in tracing_root.glob("*.py"))
    assert "trec_rag.experiments" not in sources
```

- [ ] **Step 2: Run the new path before moving code**

Run: `.venv/bin/python -m pytest code/tests/tracing/test_package_direction.py -q`

Expected: FAIL because the new test path and tracing package do not exist yet.

- [ ] **Step 3: Move the three reusable implementations and update internal imports**

Use these exact import directions:

```python
from trec_rag.tracing.models import SpanSpec, TraceBundle
from trec_rag.tracing.openai_semantics import openai_llm_attributes
```

Keep `tracing/__init__.py` minimal so importing the package does not eagerly load the optional exporter.

- [ ] **Step 4: Move reusable tests and update imports**

Replace old imports with `trec_rag.tracing.models`, `trec_rag.tracing.openai_semantics`, and `trec_rag.tracing.phoenix_export`. Preserve every assertion.

- [ ] **Step 5: Run reusable module tests**

Run: `.venv/bin/python -m pytest code/tests/tracing -q`

Expected: all reusable tracing tests pass, including the dependency-direction assertion.

- [ ] **Step 6: Commit the reusable module**

```bash
git add code/trec_rag/tracing code/tests/tracing code/trec_rag/pi_trace_models.py code/trec_rag/openai_trace_semantics.py code/trec_rag/phoenix_trace_export.py code/tests/test_openai_trace_semantics.py code/tests/test_phoenix_trace_export.py
git commit -m "Extract reusable tracing package"
```

### Task 2: Isolate the organizer Pi experiment

**Files:**
- Create: `code/trec_rag/experiments/__init__.py`
- Create: `code/trec_rag/experiments/organizer_pi/__init__.py`
- Move: `code/trec_rag/organizer_pi_inputs.py` to `code/trec_rag/experiments/organizer_pi/inputs.py`
- Move: `code/trec_rag/pi_event_trace.py` to `code/trec_rag/experiments/organizer_pi/event_trace.py`
- Move: `code/trec_rag/organizer_pi_trace.py` to `code/trec_rag/experiments/organizer_pi/cli.py`
- Create: `code/tests/experiments/__init__.py`
- Create: `code/tests/experiments/organizer_pi/__init__.py`
- Move: `code/tests/test_organizer_pi_inputs.py` to `code/tests/experiments/organizer_pi/test_inputs.py`
- Move: `code/tests/test_pi_event_trace.py` to `code/tests/experiments/organizer_pi/test_event_trace.py`
- Move: `code/tests/test_organizer_pi_trace.py` to `code/tests/experiments/organizer_pi/test_cli.py`

**Interfaces:**
- Consumes: reusable trace records and exporter interface from Task 1.
- Produces: `trec_rag.experiments.organizer_pi.inputs.{OrganizerTopic,select_topic}`.
- Produces: `trec_rag.experiments.organizer_pi.event_trace.{load_pi_events,build_piika_trace,build_fixed_trace}`.
- Produces: module command `python -m trec_rag.experiments.organizer_pi.cli {build,export}`.

- [ ] **Step 1: Move experiment implementations without aliases**

Update source-specific imports to:

```python
from trec_rag.experiments.organizer_pi.inputs import OrganizerTopic, select_topic
from trec_rag.experiments.organizer_pi.event_trace import load_pi_events
from trec_rag.tracing.models import read_trace_bundle, write_trace_bundle
from trec_rag.tracing.phoenix_export import export_trace_bundle
```

Move `piika_tool_schemas` from the reusable semantic module into `event_trace.py`; pass the resulting captured schemas into generic OpenAI normalization through the existing `SpanSpec` input.

- [ ] **Step 2: Move experiment tests and update imports**

Preserve all existing input validation, timing reconstruction, reasoning-child, CLI, strict serialization, and failure assertions. Import only through the new experiment and tracing paths.

- [ ] **Step 3: Assert old modules are gone**

Add to `test_package_direction.py`:

```python
import importlib.util
import pytest


@pytest.mark.parametrize("name", [
    "trec_rag.pi_trace_models",
    "trec_rag.openai_trace_semantics",
    "trec_rag.phoenix_trace_export",
    "trec_rag.organizer_pi_inputs",
    "trec_rag.pi_event_trace",
    "trec_rag.organizer_pi_trace",
])
def test_old_top_level_trace_modules_are_removed(name):
    assert importlib.util.find_spec(name) is None
```

- [ ] **Step 4: Run all moved tests together**

Run: `.venv/bin/python -m pytest code/tests/tracing code/tests/experiments/organizer_pi -q`

Expected: the same 102 behavioral tests plus package-layout tests pass.

- [ ] **Step 5: Smoke-test CLI discovery**

Run: `.venv/bin/python -m trec_rag.experiments.organizer_pi.cli --help`

Expected: exit 0 with the existing `build` and `export` subcommands.

- [ ] **Step 6: Commit the experiment module**

```bash
git add code/trec_rag/experiments code/tests/experiments code/tests/tracing/test_package_direction.py code/trec_rag/organizer_pi_inputs.py code/trec_rag/pi_event_trace.py code/trec_rag/organizer_pi_trace.py code/tests/test_organizer_pi_inputs.py code/tests/test_pi_event_trace.py code/tests/test_organizer_pi_trace.py
git commit -m "Isolate organizer Pi experiment"
```

### Task 3: Update documentation and package metadata

**Files:**
- Modify: `code/trec_rag/README.md`
- Modify: `docs/superpowers/specs/2026-07-29-organizer-pi-phoenix-traces-design.md`
- Modify: `docs/superpowers/specs/2026-07-30-openai-shaped-phoenix-traces-design.md`
- Modify: `docs/superpowers/specs/2026-07-30-fixed-reasoning-child-spans-design.md`
- Modify: `docs/superpowers/plans/2026-07-29-organizer-pi-phoenix-traces.md`
- Modify: `docs/superpowers/plans/2026-07-30-openai-shaped-phoenix-traces.md`
- Modify: `docs/superpowers/plans/2026-07-30-fixed-reasoning-child-spans.md`
- Inspect: `pyproject.toml`
- Inspect: `uv.lock`

**Interfaces:**
- Consumes: final paths and CLI from Tasks 1 and 2.
- Produces: accurate reviewer-facing commands and module descriptions.

- [ ] **Step 1: Rewrite the README around reusable and experimental modules**

Document `trec_rag.tracing` first. Label `trec_rag.experiments.organizer_pi` as a temporary reproduction harness, and replace every CLI example with:

```bash
.venv/bin/python -m trec_rag.experiments.organizer_pi.cli build ...
.venv/bin/python -m trec_rag.experiments.organizer_pi.cli export ...
```

- [ ] **Step 2: Correct historical design and plan path references**

Replace obsolete current module paths with their final locations while retaining the historical decisions and validation evidence. Do not rewrite unrelated plan content.

- [ ] **Step 3: Verify package discovery and dependency metadata**

Run: `.venv/bin/python -c "import trec_rag.tracing.models; import trec_rag.experiments.organizer_pi.cli"`

Expected: exit 0. `pyproject.toml` and `uv.lock` need no path-specific changes because setuptools package discovery is recursive and dependency versions are unchanged.

- [ ] **Step 4: Scan for obsolete executable/import paths**

Run: `rg -n "trec_rag\.(pi_trace_models|openai_trace_semantics|phoenix_trace_export|organizer_pi_inputs|pi_event_trace|organizer_pi_trace)" code docs/superpowers`

Expected: no active Python or README reference; any intentionally retained historical prose must clearly identify an old path rather than instruct its use.

- [ ] **Step 5: Commit documentation updates**

```bash
git add code/trec_rag/README.md docs/superpowers
git commit -m "Document tracing package boundaries"
```

### Task 4: Verify and update the pull request

**Files:**
- Inspect: all files changed from `origin/master`
- Update: GitHub PR #29 title/body if needed

**Interfaces:**
- Consumes: completed package split.
- Produces: reviewed branch and accurate draft PR.

- [ ] **Step 1: Run focused tests and compilation**

```bash
.venv/bin/python -m pytest code/tests/tracing code/tests/experiments/organizer_pi -q
.venv/bin/python -m compileall -q code/trec_rag/tracing code/trec_rag/experiments/organizer_pi
```

Expected: all focused tests pass and compilation exits 0.

- [ ] **Step 2: Run repository verification**

```bash
.venv/bin/python -m pytest -q
git diff --check origin/master...HEAD
```

Expected: no new test failures. Existing failures caused only by absent ignored/external sealed fixtures must be reported separately.

- [ ] **Step 3: Verify dependency direction and tracked-secret safety**

Run the package-direction test, search tracing sources for experiment imports, and scan tracked regular files for configured Phoenix credential values without printing secrets.

Expected: no reverse dependency and zero credential matches.

- [ ] **Step 4: Review the complete diff**

Confirm that every old implementation was moved, no compatibility aliases or generated artifacts were added, and only the experiment module contains organizer Pi reconstruction.

- [ ] **Step 5: Push and refresh PR #29**

Push `codex/phoenix-reasoning-traces`, update the draft PR summary to lead with the reusable/experimental split, and verify the PR base, head, state, and URL through GitHub.

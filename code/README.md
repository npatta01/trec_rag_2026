# Code

Reusable code for this repository should live here.

Use this folder for scripts, parsers, evaluators, and small utilities that are
useful across reports or experiments. Keep one-off downloads, logs, temporary
HTML, and scratch data outside this folder.

When adding code, include:

- expected inputs
- produced outputs
- how to run it
- how to validate it

## Topic 213 GPT-5.6 Sol one-shot run

`trec_rag.topic213_gpt_sol_oneshot` reads the frozen full-evidence response and
support audit, sends the 42 supported claims to `openai/gpt-5.6-sol` once through
OpenRouter, and uses local Qwen through LiteLLM for support auditing and nugget
evaluation. It writes organizer JSONL, readable output, audits, metrics, a
pre-nugget freeze seal, and a SHA-256 manifest under the configured output and
report directories.

Run it from the repository root:

```powershell
$env:PYTHONPATH='code'
C:\dev\trec_rag\.venv\Scripts\python.exe -X utf8 -m trec_rag.topic213_gpt_sol_oneshot --config configs/rag25_topic213_full_documents_trec26_gpt_5_6_sol_oneshot_v1.yaml
```

Validate the implementation with:

```powershell
C:\dev\trec_rag\.venv\Scripts\python.exe -X utf8 -m pytest code/tests/test_topic213_gpt_sol_oneshot.py -q
```


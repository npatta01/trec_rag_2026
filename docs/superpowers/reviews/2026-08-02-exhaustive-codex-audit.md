# Exhaustive Codex audit of `claude/ragdoll-eval-spike`

Run 2026-08-02 via `codex exec` against `git diff origin/master...HEAD`, asking for a complete
enumeration rather than a prioritised shortlist. Five prior rounds each returned one or two
findings, and four of five fixes introduced the next round's defect, so the goal was to fix
everything in one pass.

**Verdict: NO-GO. The audit reported 63 findings.**

**Only findings 57-63 were captured below.** Findings 1-56 were emitted earlier in the run and
scrolled out of the captured stdout. Re-run the audit to recover them; the prompt used is in
`docs/superpowers/handover-2026-08-02-ragdoll-eval-spike.md`.

## The one that matters most

Finding 57 states the topic-binding fix committed in `af6fc59` is **insufficient**.
`load_documents(topic_ids=...)` excludes rows for *unselected* topics, but does not bind each
document to the topic that ranked it. If topic A ranks `doc-X` and `doc-X` appears only in
topic B's row, and both topics are selected, topic B's text is still used for topic A. The
minimal fix is a per-topic `docid -> text` mapping, or validating each ranked docid against its
owning query row, rather than a flat map.

This is the third revision of the same defect, so verify any fix with a test that pairs two
selected topics sharing a docid, not only one selected and one unselected.

## Captured findings, verbatim

```
Minimal fix: validate only selected topics, or document and test the intentional whole-file validation behavior.

61. **Low — direct `OpenRouterJsonGenerator` construction permits zero attempts.**  
[competition_rag.py:820](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:820)

Scenario: `transport_max_attempts=0` skips the retry loop and reaches the “unreachable” assertion.

Minimal fix: validate all constructor parameters, especially positive attempts, timeout, and token limits.

62. **Medium — evaluation trusts submission metadata as the question.**  
[ragdoll_io.py:73](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/ragdoll_io.py:73)

Scenario: a record has a valid topic ID but a forged `metadata.narrative`; the CLI uses that forged text when constructing the gold-nugget evaluation row.

Minimal fix: load official narratives from the canonical topics file, or validate metadata narratives against it.

63. **Low — encoded-secret coverage is weaker than the test names suggest.**  
[test_competition_rag.py:2168](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/tests/test_competition_rag.py:2168)

The non-JSON path omits the body before redaction, while the JSON case does not exercise the deepest mixed-encoding paths.

Minimal fix: explicitly assert persistence safety for parsed JSON envelopes containing layered encoded secrets.

The previous “clean” statement about generation topic binding should therefore be read as: the selected-topic filter exists, but the binding is not sufficient. The final verdict remains **NO-GO**.
tokens used
522,586
Addendum from the completed spec review: these findings should be added to the audit above.

57. **High — topic binding is still too weak in the generation path.**  
[competition_rag.py:554](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:554), [competition_rag.py:1620](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:1620)

`topic_ids` filters to a selected set, but does not bind each document to the topic that ranked it.

Scenario: topic A ranks `doc-X`; `doc-X` appears only in topic B’s selected document row. The loader accepts topic B’s text for topic A.

Minimal fix: pass a per-topic `docid → text` mapping or validate each ranked docid against its owning query row.

58. **Medium — overwrite deletes valid state before validating new inputs.**  
[competition_rag.py:1607](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:1607)

Scenario: an overwrite run has a typo in the queries/run/documents path. Existing output and work state are deleted before the typo is detected.

Minimal fix: validate and digest all inputs first, then clear state only after validation succeeds.

59. **Low — rank gaps are accepted despite the branch’s stated strict-run contract.**  
[competition_rag.py:490](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:490)

Scenario: ranks `1,3` are treated as valid adjacent results. This conflicts with the dev builder’s claim that rank gaps are rejected.

Minimal fix: either require contiguous ranks and update the test, or remove the strict-contract claim.

60. **Low — malformed unselected topics can abort a selected-topic run.**  
[competition_rag.py:491](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:491)

Scenario: the requested topic is valid, but an unrelated topic has duplicate ranks or a non-finite score; the entire selected-topic generation fails.

Minimal fix: validate only selected topics, or document and test the intentional whole-file validation behavior.

61. **Low — direct `OpenRouterJsonGenerator` construction permits zero attempts.**  
[competition_rag.py:820](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/competition_rag.py:820)

Scenario: `transport_max_attempts=0` skips the retry loop and reaches the “unreachable” assertion.

Minimal fix: validate all constructor parameters, especially positive attempts, timeout, and token limits.

62. **Medium — evaluation trusts submission metadata as the question.**  
[ragdoll_io.py:73](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/trec_rag/ragdoll_io.py:73)

Scenario: a record has a valid topic ID but a forged `metadata.narrative`; the CLI uses that forged text when constructing the gold-nugget evaluation row.

Minimal fix: load official narratives from the canonical topics file, or validate metadata narratives against it.

63. **Low — encoded-secret coverage is weaker than the test names suggest.**  
[test_competition_rag.py:2168](/home/npatta01/data/competitions/trec_rag_2026/.claude/worktrees/codex-review-gaps-45fe07/code/tests/test_competition_rag.py:2168)

The non-JSON path omits the body before redaction, while the JSON case does not exercise the deepest mixed-encoding paths.

Minimal fix: explicitly assert persistence safety for parsed JSON envelopes containing layered encoded secrets.

The previous “clean” statement about generation topic binding should therefore be read as: the selected-topic filter exists, but the binding is not sufficient. The final verdict remains **NO-GO**.
```

# Luna Operation Screen: Three-Topic Result

## Verdict

Promote the decisions-only Luna screen into the hardened generation design, after a normal
integration pass. The frozen v2 replay passed its intended behavioral probes: it rejected the
known weak topic-`233` replacement, retained every topic-`300` insertion, and removed the one
replacement on topic `499` that would have erased a distinct religious argument. It made no new
Sol calls and did not rewrite answer prose.

This verdict is based on the resulting prose and evidence, not only metric movement. The screen is
valuable because it selectively prevents information loss while preserving useful additions.

## Flow Tested

```text
authenticated draft + frozen Sol operations + narrative + selected evidence
                               |
                      one Luna-medium screen
                               |
             strict five-gate decision for every operation
                               |
                  deterministic accepted-subset apply
                               |
             organizer + citation-domain validation (local)
```

Luna could only return booleans for support, atomicity, materiality, redundancy, and replacement
safety. Local code accepted an operation only when all five were true. Invalid output or invalid
assembly would have preserved the original validated draft.

## Sealed v2 Results

| Topic | Operations kept | Words | Weighted first support | Weighted all support | Hard support | Judgment |
|---|---:|---:|---:|---:|---:|---|
| `233` | 1 / 4 | 686 | 0.884615 | 0.871795 | 0.769231 | Correctly removed unsafe/redundant edits; retained supported longitudinal context. |
| `300` | 3 / 3 | 897 | 0.811111 | 0.811111 | 0.622222 | Retained all three meaningful insertion-only additions. |
| `499` | 4 / 5 | 753 | 0.708333 | 0.694444 | 0.444444 | Preserved four useful additions and rejected a replacement that lost a distinct argument. |

All three outputs passed the existing strict organizer profile, exact-hint citation validation,
selected-evidence citation-domain validation, the 1,024-word cap, and authenticated receipt checks.
There was no fallback.

The support metrics required no new judge calls. Every screened statement/document pair existed in
the already judged union of its authenticated draft and full bounded-revision final; there were
zero missing pairs and zero conflicting cached labels.

## Comparison With the Unscreened Alternatives

| Topic | Candidate | Weighted first | Weighted all | Hard |
|---|---|---:|---:|---:|
| `233` | Draft | 0.881579 | 0.868421 | 0.763158 |
|  | Unscreened Sol operations | 0.875000 | 0.856250 | 0.750000 |
|  | **Luna-screened** | **0.884615** | **0.871795** | **0.769231** |
| `300` | Draft | 0.809524 | 0.809524 | 0.619048 |
|  | Unscreened Sol operations / **Luna-screened** | **0.811111** | **0.811111** | **0.622222** |
| `499` | Draft | 0.671875 | 0.656250 | 0.375000 |
|  | Unscreened Sol operations | 0.708333 | **0.701389** | 0.444444 |
|  | **Luna-screened** | **0.708333** | 0.694444 | **0.444444** |

Topic `499` illustrates why the decision is not score-only. Keeping the original two-citation
religious explanation slightly lowers the all-citation average relative to the proposed
replacement, but preserves a narrative-relevant sanctity-of-life rationale that the replacement
would delete. First-citation and hard support are unchanged.

The slow nuggetizer was not rerun. Its existing source-run scores are stochastic and even moved
down on topic `300` when supported sentences were added, so another expensive pass would not be a
sound gate for this bounded iteration. The retained additions were instead checked directly
against the full narrative and selected passages.

## Calls and Cost

| Run | Model calls | Provider cost |
|---|---:|---:|
| Initial topic-`233` diagnostic | 1 Luna, 0 Sol | $0.002224 |
| Frozen v2 topics `233`, `300`, `499` | 3 Luna, 0 Sol | $0.007576 |
| **Entire operation-screen experiment** | **4 Luna, 0 Sol** | **$0.009800** |

The initial call exposed an ambiguity: a replacement and a neighboring insertion could be judged
inconsistently against the unchanged draft. The only prompt revision instructed Luna to select a
coherent candidate subset and judge redundancy/replacement safety in that resulting draft. The v2
prompt was then frozen for all three reported topics.

## Scope and Next Action

The reusable strict screen and authenticated replay are implemented on the isolated
`codex/hardened-splice-three-topic` branch. Production competition runners and checked-in configs
remain unchanged. The next change should integrate the small decision contract at the existing
post-revision boundary; the replay harness itself should remain experimental rather than becoming
another production pipeline.

## Verification Boundary

The change-focused suite passes, but the repository-wide suite is not green in this linked
worktree. A fresh full run produced `2763 passed, 19 skipped, 18 failed`. The failures are outside
the operation-screen code and fall into three pre-existing environment/path groups:

- 2 checked-in RAG config assertions compare an absolute main-checkout handoff path with the
  linked-worktree retrieval path;
- 1 portability assertion resolves the installed RAGDoll package from the main checkout instead
  of this worktree's pinned submodule path;
- 15 retrieval-cache wrapper tests lack a worktree-local `hf` command or copy a non-portable
  `.venv/bin/python` link into temporary checkouts.

Because the full integration gate is red, keep the worktree and branch; do not open the PR until
those repository environment failures are resolved or explicitly accepted as the baseline.

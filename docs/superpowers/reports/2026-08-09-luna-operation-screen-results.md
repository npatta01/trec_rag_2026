# Luna Operation Screen: Three-Topic Result

## Verdict

The decisions-only Luna screen is now promoted into the opt-in hardened bounded-revision path.
The frozen v2 replay passed its intended behavioral probes: it rejected the
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

## Frozen Scope

The reusable strict screen now runs at the existing post-revision boundary in
`narrative_blueprint_trial`. It adds at most one Luna-medium call, never adds a Sol call, and falls
back to the validated draft on any screen semantic or local-validation problem. The trial contract
was bumped so older state cannot resume under the new call graph. The authenticated replay remains
an experimental diagnostic tool. Production `competition_rag.py` and checked-in competition
configs remain unchanged. This is the frozen best attempt; no further AG prompt or model iteration
is part of this release pass.

## Verification Boundary

The linked-worktree portability gaps were resolved without changing generation behavior: the
worktree-local environment was installed from the repository setup script, private `.cache/`
contents are ignored, and checked-in config tests now assert the authenticated handoff identity
without depending on which checkout contains the existing private artifact.

- Focused operation-screen/blueprint checks: `33 passed`.
- Retrieval portability/shard checks: `49 passed`.
- Representative topic-`233` dry run: at most 6 Luna calls (planner + 4 group audits + screen),
  at most 3 Sol reservations, and 0 provider calls during the dry run.
- Fresh repository suite: `2781 passed, 19 skipped, 60 subtests passed` in `118.83s`.
- Static checks: changed implementation files pass Ruff; `git diff --check` passes.

The integration and verification pass made zero hosted model calls and produced no new topic
answers or evaluator judgments.

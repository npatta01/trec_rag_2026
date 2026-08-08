# Bounded Narrative Revision Experiment Design

## Objective

Test whether a post-draft, evidence-only omission audit followed by one bounded revision improves
full-narrative nugget coverage under the 1,024-word organizer cap. Run exactly three development
topics sequentially, inspect the paired draft-to-final gaps, and do not productionize the approach
from this experiment alone.

## Question

The earlier blueprint trials showed that pre-draft claim selection changes which details are
omitted but does not reliably reduce omissions. This experiment asks whether auditing the actual
draft against the complete narrative and authenticated selected evidence makes the second Sol
completion materially better than the first.

## Scope

The experiment uses topics `233`, `300`, and `499`, in that order. All three have strict full
retrieval-nugget coverage (`1.0`) in the frozen extraction evaluator, so generation is tested where
the upstream canonical surface already covers every planner-derived narrative obligation. They
provide increasing synthesis pressure:

| Topic | Narrative groups | Claim hints | Selected passages | Citation-domain documents |
| ---: | ---: | ---: | ---: | ---: |
| `233` | 4 | 37 | 188 | 143 |
| `300` | 4 | 80 | 186 | 126 |
| `499` | 7 | 86 | 309 | 223 |

Run one topic to completion before starting the next. Freeze the prompts and contracts across all
three topics; findings from an earlier topic may be recorded but must not change later-topic prompts
inside this experiment.

## Model and Call Policy

- Planning: one GPT-5.6 Luna call at `medium` reasoning per topic.
- Drafting: one GPT-5.6 Sol call per topic.
- Audit: one GPT-5.6 Luna call at `medium` reasoning per authenticated generated group after the
  draft exists. This means 4, 4, and 7 audit calls for the three topics. Cheap audit calls are not
  capped at three, but their count and cost are recorded.
- Revision: one GPT-5.6 Sol call per topic.
- Repair: at most one additional GPT-5.6 Sol call, only when the candidate fails deterministic
  validation. It cannot be used for another quality revision.
- Hard ceiling: no more than three Sol semantic reservations per topic across all attempts and
  resumes. A malformed or rejected semantic response still consumes its reservation. Transport
  attempts that fail before a semantic response are counted separately and cannot change the
  semantic-reservation ceiling.
- Evaluation: use the existing inexpensive RAGDoll/DeepSeek path only after the topic's final
  candidate is sealed.

External model calls may receive the narrative, advisory hints, selected passages, draft, and
locally derived aliases for this experiment. The user has already approved these scoped external
API calls.

## Data Boundary

Planning, drafting, auditing, revision, and repair may consume only the authenticated generation
handoff and artifacts derived from it. Selected passages are factual authority; claim hints are
advisory.

Before the final answer is sealed, the experiment must not open or pass to any model:

- the organizer TREC run;
- the full-text ZIP;
- qrels or gold nuggets;
- RAGDoll assignments or scores;
- prior generated answers or prior evaluations for the selected topic.

RAGDoll is post-hoc only. There is no feedback edge from RAGDoll or gold data to audit, revision,
or repair.

## Pipeline

1. **Luna plan:** derive full-narrative obligations and allocate the available word budget. Claim
   aliases are non-exclusive focus suggestions; they never prune hints, passages, or citations.
2. **Sol draft:** write an answer from every advisory hint and every selected passage, using the
   plan as an attention and allocation guide.
3. **Draft preflight:** normalize only mechanically safe fields, validate the draft, and retain a
   valid copy as the fallback candidate.
4. **Group audits:** for each authenticated generated group, give Luna that group's text, linked
   hints and selected passages, the untouched full narrative, and the draft. Each audit returns a
   bounded list of evidence-backed missing-detail cards.
5. **Audit merge:** deterministically validate aliases, deduplicate overlapping cards, rank them by
   narrative importance and specificity, and cap the merged revision brief.
6. **Sol revision:** return a complete replacement answer. Prefer replacing generic, redundant, or
   lower-value prose with audited details instead of appending material. Stay within 1,024 words.
7. **Final validation:** perform safe normalization and deterministic validation. If valid, seal the
   revision without a third Sol call.
8. **Optional Sol repair:** if final validation fails, provide Sol the candidate, exact validation
   errors, and the same authenticated evidence. Ask for the smallest complete-answer repair, then
   validate once more.
9. **Seal or fall back:** if repair fails, use the latest earlier validated candidate. If none
   exists, mark the topic failed and resumable. Never make a fourth Sol call.
10. **Post-hoc evaluation:** evaluate both the retained first draft and sealed final candidate so
    the experiment obtains a paired draft-to-final comparison without spending another writer
    call.

If the first draft itself fails preflight, consume the repair allowance immediately and skip the
quality-revision Sol call for that topic. This preserves the hard ceiling and prevents an invalid
candidate from entering the audit stage.

## Audit Card Contract

Each cheap facet audit returns zero or more cards containing:

- the narrative obligation or facet alias;
- a concise missing claim or detail;
- one or more authenticated evidence aliases that fully support it;
- importance: `must`, `should`, or `could`;
- omission type: missing, too generic, incomplete enumeration, missing quantity/example,
  unbalanced tradeoff, or redundant-space replacement;
- a concise reason the detail is necessary for the full narrative;
- an optional answer-object index whose lower-value content could be replaced.

The audit cannot invent identifiers, request new retrieval, assign citations outside the supplied
evidence, or demand inclusion merely because a detail is interesting. Empty audit output is valid.

## Validation and Repair

Safe local normalization may attach fixed metadata, remove duplicate citations, rebuild the
reference list from actually used allowed document IDs, and remap citation indexes. It must not
change answer text or factual meaning.

The deterministic validator rejects:

- malformed or truncated JSON and unexpected fields;
- empty answer text or an empty answer list;
- more than 1,024 whitespace-separated answer words;
- answer objects without one to three unique citations;
- citations outside the authenticated selected-evidence domain;
- duplicated, uncited, or inconsistent references after normalization;
- exact near-verbatim claim hints paired with document IDs outside that hint's linked evidence.

An over-limit revision is a repair failure, not permission to silently drop the tail. The repair
prompt receives only the exact validator errors, the invalid candidate, and the original allowed
inputs. It returns a complete answer; local code then reruns the same validator. Semantic quality
concerns such as incomplete narrative coverage or weak citation support belong to the cheap audit
and post-hoc evaluation, not this deterministic repair trigger.

## Evaluation and Cost Ledger

For the first draft and final candidate, record:

- strict and partial-credit vital nugget coverage;
- strict and partial-credit all-nugget coverage;
- total words and answer-object count;
- structural citation validity;
- semantic citation-support results from the existing support judge, using the same judge model,
  prompt, and settings for both candidates;
- nuggets gained, lost, or changed from partial to full between draft and final.

For every hosted call, record the stage, model, reasoning effort, semantic completion status,
input/output/reasoning tokens when available, latency, and provider-reported or calculated cost.
Report transport failures separately from semantic calls. Aggregate both per-topic and three-topic
totals.

## Success and Interpretation

The approach is promising if the three-topic macro strict-vital and strict-all coverage do not
decrease, at least two topics improve on one of those strict metrics, and no final answer regresses
materially in semantic citation support. Partial-credit movement, word use, cost, and the exact
gained/lost nuggets explain the result; they do not override a strict-metric regression.

If the support judge cannot complete after its existing bounded retry policy, mark that topic's
support comparison unavailable and do not declare the overall experiment successful.

Regardless of outcome, finish with a gap taxonomy:

- evidence existed and the draft omitted it, then revision recovered it;
- evidence existed but revision still omitted or compressed it;
- audit proposed a detail but revision failed to preserve it;
- the selected evidence did not support the gold nugget;
- citation support or answer-contract failure;
- ambiguous.

Use that taxonomy to brainstorm the next change. Do not tune prompts between the three topics or
promote this prototype into the production competition runner in the same experiment.

## Artifacts and Privacy

Keep prompts, selected passages, provider responses, generated answers, per-nugget judgments,
and cost ledgers under ignored experiment output directories. Commit only reusable prototype code
and aggregate, privacy-reviewed findings. Do not push, publish, or copy private artifacts into a
rendered-report directory.

## Verification Scope

Keep verification proportionate to this prototype:

- one dry run proving the state transitions and Sol-call ceiling;
- targeted syntax/lint checks for touched prototype files;
- deterministic validation of every generated candidate;
- the three sequential live topics and their paired post-hoc metrics.

Do not build a broad production test suite until the experiment identifies a winning behavior.

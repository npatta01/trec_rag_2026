# Narrative Blueprint Generation Design

## Objective

Add an experimental, configuration-controlled generation strategy that plans coverage of the complete official narrative before writing the cited answer. The strategy is intended to convert unused answer-word budget into better nugget coverage without weakening citation support.

The first experiment is limited to RAG25 development topics `31`, `72`, and `200` over the existing sealed selected-evidence handoff. It is a mechanism test, not evidence for an all-topic competition run.

## Decision

Use a two-stage `narrative_blueprint_v1` strategy:

1. One Sol completion creates a compact narrative blueprint from the official narrative, generated subnarrative text, and existing Nuggetizer claim hints.
2. The existing Sol writer receives the validated blueprint and a deterministic evidence projection derived only from the authenticated handoff.

The planner does not see passage text, document IDs, evidence IDs, qrels, gold nuggets, RAGDoll judgments, or prior answers. Selected passages remain factual authority. Claim hints remain advisory routing aids.

The existing one-shot strategy remains the default and its prompt behavior does not change.

## User Constraints

- Do not run all topics. Run only topics `31`, `72`, and `200` in this experiment.
- Use at most three Sol semantic completions per topic.
- Multi-pass generation is acceptable when it improves the answer.
- The complete official narrative controls coverage. Generated subnarratives are retrieval structure, not independent output requirements.
- Reuse the existing per-subnarrative Nuggetizer claim hints. Do not add another claim-card model pass.
- A cheaper model may be explored later, but the first paired trial uses Sol for both planning and writing so the mechanism is not confounded by a model change.

## Evidence Behind the Design

### Baseline

The existing selected-evidence Sol baseline produced:

| Topic | Answer words | Strict vital | Strict all | Weighted precision, first citation |
|---|---:|---:|---:|---:|
| 31 | 351 | 0.458333 | 0.470588 | 0.842105 |
| 72 | 408 | 0.277778 | 0.244444 | 0.976190 |
| 200 | 649 | 0.524590 | 0.477273 | 0.863636 |
| Macro | — | 0.420234 | 0.397435 | 0.893977 |

The answers use only 34–63% of the 1,024-word allowance. Their 85 completed citation judgments contain 58 Full Support, 27 Partial Support, and zero No Support labels.

The selected evidence itself has macro strict-vital `0.546828` and strict-all `0.522178`, but requires 3,647–7,761 rendered words per topic. This is an availability diagnostic, not an attainable answer score.

### Compact planner size

With local short aliases and no passage text, cryptographic IDs, or evidence links, the three planner inputs are approximately 1,951–3,018 tokens. Pass 1 is therefore a prioritization problem rather than a context-window problem.

### Hint-link audit

A deterministic audit of the authenticated handoff found:

| Topic | Hint-linked passages | Hint-linked passage words | Baseline judged citations outside any hint-linked docid | Full-Support citations outside any hint-linked docid |
|---|---:|---:|---:|---:|
| 31 | 95/244 (38.9%) | 2,170/4,225 (51.4%) | 0/23 | 0/13 |
| 72 | 154/448 (34.4%) | 4,202/9,095 (46.2%) | 0/21 | 0/20 |
| 200 | 139/302 (46.0%) | 3,335/6,632 (50.3%) | 2/41 | 1/25 |

Claim hints route to useful documents, but they cover only about half the selected passage words. Hard-pruning every obligation to claim-linked passages would therefore create a material evidence-orphan risk.

## Advisor Review

Codex Sol xhigh, Claude Fable, and Gemini Pro High independently returned `proceed with corrections` on one frozen repository-grounded packet. Their common conclusions were:

- keep the compact narrative-first planner;
- do not give the planner passage text;
- protect against incomplete claim-to-passage coverage;
- allow disjoint narrative anchors and normalize them deterministically;
- require at least one `must` obligation and evidence for every obligation;
- preserve obligation-to-claim-to-passage mapping for the writer;
- persist the validated plan and reuse it on resume;
- count semantic completions separately from transport attempts and prohibit a fourth Sol completion;
- treat the target word allocation as available space, not a padding quota;
- use the three topics only as a mechanism test.

## Architecture

### Strategy seam

Add an optional generation setting:

```yaml
generation:
  strategy: narrative_blueprint_v1
```

Supported values are:

- `selected_evidence_one_shot_v1` — current behavior and default when omitted;
- `narrative_blueprint_v1` — experimental planner plus narrowed writer.

The strategy seam belongs in `competition_rag.py`, while blueprint construction and validation live in a focused `narrative_blueprint.py` module. The latter is a deep module: callers supply an authenticated `GenerationTopic` and receive either a validated blueprint/projection or a precise validation error. Alias creation, normalization, resolution, widening, hashing, and rendering remain inside the module.

### Pass 1 input

The planner prompt contains, in this order:

1. The untouched official narrative.
2. Instructions to derive obligations from that narrative rather than treating generated groups as mandatory.
3. Generated group aliases and group text.
4. Claim aliases and advisory claim text nested under each group.

Aliases are deterministic and local to the topic:

- groups: `g001`, `g002`, … in handoff order;
- claims: `c001`, `c002`, … in handoff order.

The prompt contains no authoritative passage text or authority-bearing identifiers.

### Pass 1 output

The strict provider schema returns exactly:

```json
{
  "obligations": [
    {
      "label": "Short advisory description",
      "narrative_spans": ["exact phrase one", "exact phrase two"],
      "priority": "must",
      "answer_mode": "explain",
      "target_words": 180,
      "selected_claim_aliases": ["c004", "c011", "c018"]
    }
  ]
}
```

`answer_mode` is one of `describe`, `explain`, `compare`, `evaluate`, `recommend`, or `enumerate`.

### Pass 1 local validation

Validation is deterministic and fail-closed:

- exactly 3–8 obligations;
- at least one obligation has priority `must`;
- priorities are only `must`, `should`, or `could`;
- every label is nonempty;
- every obligation has 1–4 nonempty narrative spans;
- each span matches a substring of the official narrative after Unicode NFKC normalization, case folding, and whitespace collapse;
- disjoint spans are permitted so a composite obligation can cite separated phrases;
- duplicate normalized span sets are rejected;
- every `target_words` value is a positive integer;
- the sum of target allocations is 850–950 words;
- every obligation selects at least one known claim alias;
- aliases within an obligation are unique;
- every selected claim belongs to a known handoff group and links to authenticated evidence.

The label is advisory. It cannot add factual authority or override the anchored narrative text.

No planner retry is allowed in the first experiment. A malformed or invalid plan consumes the planner reservation and fails that topic closed.

### Deterministic evidence projection

For each obligation:

- always include the authoritative passages linked by its selected claims;
- if priority is `must`, additionally include every selected passage from every group represented by those claims;
- if priority is `should` or `could`, do not widen beyond claim-linked passages.

This rule incorporates the measured hint-link audit: must-level narrative needs retain group-level recovery paths, while lower-priority content remains compact.

The module preserves stable mappings:

```text
obligation → selected claim aliases → authenticated claim IDs
           → linked/widened evidence IDs → allowed ClimbMix docids
```

The global writer evidence catalog is the stable handoff-order union of the per-obligation evidence sets. Passages shared by multiple obligations are rendered once, and each obligation names its evidence aliases.

### Pass 2 writer input

The writer prompt contains:

- the official narrative;
- the validated obligation list, priorities, approximate word allocations, claim aliases, and evidence aliases;
- the selected advisory claim hints;
- the deduplicated authoritative passage catalog;
- the existing organizer output and citation instructions.

The writer must cover `must` obligations first, then `should`, and use `could` only when room remains. The 850–950-word allocation is planning capacity, not a required minimum; the prompt explicitly forbids padding and repetition. The hard organizer ceiling remains 1,024 whitespace-separated words.

The writer may cite only docids in the derived projection. Existing exact-hint citation checks continue to apply.

## Persistence and Resume

For each topic in blueprint mode, persist a validated state record under the experiment's dedicated `work/` directory. The state binds:

- schema and prompt-contract version;
- topic ID and authenticated topic context hash;
- model/request identity inherited from the generation identity;
- planner prompt hash;
- validated obligations;
- alias-to-claim mapping;
- per-obligation evidence IDs;
- derived citation domain;
- writer prompt hash.

Resume authenticates this record and reuses it. It never spends another planner completion for a topic with valid persisted state.

Persist a per-topic semantic-call ledger using reservation-before-request semantics. A reservation counts even if the process crashes before receiving or saving the provider response. The ledger distinguishes:

- one planner semantic reservation maximum;
- two writer semantic reservations maximum;
- transport attempts, which remain governed separately by the HTTP retry policy.

A topic can therefore never exceed three Sol semantic reservations across resumes. There is no separate fourth repair pass.

## Error Handling

- Invalid planner schema or semantics: persist a sanitized failure receipt and fail the topic closed.
- Planner reservation exists without a valid blueprint after interruption: refuse another planner call and report that the one-call planning budget was consumed.
- Writer validation failure: use at most the remaining writer semantic reservation and append the existing targeted retry instruction.
- Exhausted three-call ledger: fail closed without invoking the provider.
- Invalid or mismatched persisted blueprint: refuse resume and require a new experiment ID; do not silently regenerate or mix identities.
- One topic failure prevents publishing a partial organizer JSONL, while preserving per-topic state for diagnosis.
- Provider responses, passage text, and evaluation artifacts remain private and outside git.

## Configuration and Identity

The optional strategy is part of strict config parsing and generation identity. Changing it invalidates resume. Identity also binds the blueprint prompt/schema versions and the strategy-specific prompt hashes.

Checked-in canonical competition configs remain one-shot and continue selecting all 119 test narratives. The trial uses a new ignored `configs/local/` file with only topics `31`, `72`, and `200` and a unique output namespace.

## Testing

Tests follow red-green-refactor and exercise behavior through public module seams.

### Blueprint module tests

- compact rendering contains narrative/groups/hints but no passage text, docids, evidence IDs, or cryptographic IDs;
- alias ordering is deterministic;
- normalized disjoint span validation accepts cosmetic Unicode/case/whitespace differences but rejects text absent from the narrative;
- obligation count, priorities, modes, labels, budgets, required claims, alias uniqueness, and at-least-one-must rules fail closed;
- must obligations widen to full selected groups;
- should/could obligations retain only claim-linked passages;
- evidence union is stable and deduplicated while obligation mappings are preserved;
- derived citation domain contains only selected projection docids;
- persisted state detects any topic, prompt, plan, or projection mismatch.

### Runner integration tests

- omitted strategy preserves existing one-shot calls and prompts;
- blueprint mode calls planner once and writer once on success;
- writer validation failure permits exactly one writer retry;
- invalid planner output produces no writer call;
- resume with a valid blueprint but no answer reuses the blueprint and calls only the writer;
- resume after a consumed planner reservation without valid state makes no provider call;
- exhausted writer reservations prevent further resume calls;
- no blueprint-mode execution can reserve more than three semantic completions per topic;
- generation identity changes when strategy or blueprint prompt contracts change;
- organizer output validation and exact-hint citation validation remain enforced against the derived citation domain.

Run targeted blueprint/competition tests first, then the full relevant generation and handoff test set.

## Three-Topic Trial

### Pre-run disclosure

- selected topics: 3 (`31`, `72`, `200`);
- retrieval/reranking/canonicalization: fully reused from the sealed handoff, zero expected calls;
- expected Sol semantic reservations: 2 per topic;
- hard maximum Sol semantic reservations: 3 per topic, 9 total;
- transport retries: separately counted and reported;
- output: a new unique ignored `outputs/` namespace;
- no all-topic run and no parallel competition run.

### Validation and evaluation

After generation:

1. Validate the exact topic IDs, narratives, organizer schema, citation domain, citation cardinality, and 1,024-word ceiling.
2. Confirm every topic's call ledger has at most three semantic reservations.
3. Run the same development-only RAGDoll nugget assignment and selected-passage citation-support evaluation used for the paired baseline.
4. Report per-topic and macro coverage, answer words, Full/Partial/No Support counts, weighted citation support, evidence-retention counts, and actual semantic/transport attempts.

### Mechanism-success gate

Promote only to a broader development validation—not to an all-topic run—when all conditions hold:

- macro strict-vital improves by at least `0.05` over `0.420234`;
- macro strict-all improves by at least `0.05` over `0.397435`;
- topic `72` improves on both coverage measures;
- topics `72` and `200` each capture at least 40% of their measured selected-evidence headroom on both measures;
- no topic regresses by more than `0.02` on either coverage measure;
- both macro weighted first-citation precision and macro weighted all-judged-citation precision are at least `0.85` and no more than `0.03` below their paired baseline values (`0.893977` and `0.884541`, respectively);
- No Support is at most 5% of completed judgments;
- every output passes organizer and authenticated-handoff validation;
- no topic exceeds three Sol semantic reservations.

Small metric changes are interpreted cautiously because there are only three topics and automated development labels are incomplete. The report must trace newly covered nuggets back to blueprint obligations before attributing gains to the mechanism.

## Explicit Non-Goals

- No all-topic generation run.
- No change to retrieval, reranking, canonicalization, or the handoff schema.
- No new claim-card or Nuggetizer model pass.
- No generation access to qrels, gold nuggets, RAGDoll output, the TREC run, or the full-text ZIP.
- No planner model comparison in the first trial.
- No length-only one-shot control without separate authorization; if the blueprint succeeds, that cheaper control is the next causal test.
- No automatic promotion based on this three-topic sample.

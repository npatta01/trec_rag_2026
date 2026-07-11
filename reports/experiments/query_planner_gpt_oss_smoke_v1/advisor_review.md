# Advisor review: gpt-oss-20b sparse query planner

## Decision

**NO-GO** for the frozen local `gpt-oss-20b` planner. This configuration must
not be promoted to retrieval.

The formal gate required at least four of five mechanically valid first-pass
plans, five of five after at most one repair, no semantic field scored zero,
and a mean semantic score of at least 10/12. The replacement run produced
three mechanically valid plans and two preserved failures. The three
scorable plans had zero accepts and averaged 6.33/12.

## Locked-rubric scores

| Topic | A | B | C | D | E | F | Total | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 144 | 2 | 1 | 1 | 2 | 1 | 1 | 8/12 | Revise |
| 213 | 2 | 0 | 0 | 2 | 1 | 0 | 5/12 | Reject; hard-rule violation |
| 224 | 1 | 1 | 0 | 2 | 1 | 1 | 6/12 | Reject |

The advisor applied this rubric without inspecting qrels, retrieval results,
reranker output, or downstream evaluation.

## Topic findings

### 144

- All requested needs appear.
- The bank-failure facet narrows the requested general mechanism, with Silicon
  Valley Bank as an example, to Silicon Valley Bank alone.
- The economic-development query loses the explicit credit-union side.
- Safety/trust, services, and regulation are overbundled into one facet.
- Expansions are safe, but the global expansion contributes no new analyzed
  tokens after deterministic deduplication.

### 213

- All requested needs appear, but origins, ending, US involvement, and Cold
  War rationale are collapsed into one mega-facet.
- `Korean War (1950-1953)` introduces a date and number absent from the
  narrative. The frozen prompt explicitly prohibited new facts, dates, and
  numbers, so this is a hard-rule violation even though the fact is true.
- Three explicit requests are incorrectly marked supporting rather than core.
- Facet expansions are empty and most global variants are redundant.

### 224

- Immigration motives and refugee causes/status are collapsed into one facet.
- Challenges are described and expanded as immigrant-only even though the
  request also concerns refugees.
- Country and religious views are also combined despite differing likely
  evidence.
- Expansions are mostly paraphrases; the global query adds only `policy`.

## Failed-output diagnosis

### 407

The first validator error was not a capitalization issue. The model substituted
`surged` for the exact narrative word `soared`. After correcting that only in
memory, a second error appeared: another purported span skipped the intervening
text `the housing market and`, constructing a non-contiguous quote.

Even after diagnostic in-memory span correction, the plan remained
semantically unacceptable. It injected candidate causes such as `economic
indicators`, `supply constraints`, `inflationary pressures`, `commodity
prices`, and `inflation`; bundled the REIT and public-housing-demolition
interventions; and used redundant six-term facet expansions.

### 515

The response stopped at exactly 6,000 completion tokens with
`finish_reason=length`. Appending only the missing `]}` made it parse for
diagnosis, and its six information needs were semantically well separated.
It would nevertheless fail rendering: the first facet has only four unique
content tokens, below the frozen minimum of five. Three explicit requested
facets were also mislabeled supporting.

## Recommendation

Do not spend the single formal repair on casefold canonicalization plus a
larger output budget. Those changes do not repair topic 407, and a 6,400-token
budget merely exposes topic 515's next validator failure.

For the next benchmark:

1. Use numbered exact token/span references and let Python resolve source text;
   never ask the model to recopy narrative spans.
2. Derive every explicit request's priority as core in code.
3. Tighten expansion caps and numeric/factual safety checks.
4. Benchmark a stronger planner under the same blinded rubric before running
   retrieval.

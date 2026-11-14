# SPEC — TREC RAG 2026, the whole flow

Picked storyline: **the entire flow, forward** — query → subnarratives → queries
→ search → rerank → select → nuggets → RAG → the cited answer. The other two
("Follow the citation", "Two writers") stay in `STORYLINES.md`.

```text
GOAL:    the viewer can say, in their own words, how one request becomes a
         cited answer — and that every sentence points at a passage that was
         actually retrieved.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps
TIME:    87 s x 30 = 2610 frames. The stages are the clock; the bed follows them.
```

## LOOK

The medium is the talk deck's own printing: **flat editorial figures on the
drawings' cream plate**. Not a palette — the same rules the deck's SVGs follow.

- **ground** `#fdf1da`, flat, 1.5% fibre grain, no gradient anywhere
- **ink** `#07394a`; **muted ink** `#3d5c66` (4.9:1 on the ground)
- **panel** `#fffaf0`, hairline `rgba(7,57,74,0.18)` at 1px, radius 12 px
- **depth** one offset only: `0 2px 0 rgba(7,57,74,0.06)`. No blur, no vignette.
- **accents, one stage each, never decorative**
  - retrieval `#0d6f84` — subnarratives, queries, search, rerank, select
  - advisory `#855a06` — claims (the hint lane)
  - generation `#9c4514` — Sol, drafting
  - verdict `#39702f` — Luna's screen, and the citation that holds
- **display** Inter 900, 64 px, tracking −2%
- **body** Inter 400 / 700, 30 px (over the 28 px floor at 1080)
- **stage label** Inter 700, 22 px, +6% tracking, uppercase — the deck's eyebrow
- **strokes** 2.6 px, square caps, the deck's own weight

## OBJECTS

Each is drawn so it reads without its label.

- **request sheet** — a panel holding the narrative's real words, hairline
  border, the way the deck prints a narrative. 980 px wide.
- **angle strips** — 8 strips, 1120 x 52 px, each a subnarrative in its own
  words, each with the deck's question glyph at its left.
- **query blocks** — rectangles at radius 6 px with the real query string in
  Inter 400 26 px. Rectangles, never pills.
- **document field** — 1,000 thin vertical bars, 3 px wide, 2 px apart, height
  by BM25 rank. A mass you cannot count, which is the point.
- **passage blocks** — 180 x 44 px with a score bar along the bottom edge.
- **claim rows** — 2 px dashed edge, advisory amber, because a claim is a hint.
- **Sol / Luna columns** — two labelled columns, generation and verdict, each
  with its own stack of operations.
- **answer sheet** — the real answer text with superscript citation markers that
  light when their passage is traced.

## MOTION

| Role   | Curve           | Frames |
| ------ | --------------- | -----: |
| enter  | `power2.out`    |     14 |
| move   | `power3.inOut`  |     24 |
| exit   | `power2.in`     |     10 |
| camera | `power2.inOut`  |     36 |
| stagger between siblings | — |   4 |

One hero move per shot; secondary moves travel 50–70% as far and start 5 frames
later. Nothing drifts after it lands.

## SHOTS

| # | Frames | Hero move | What is read | Hands off by |
| - | ------ | --------- | ------------ | ------------ |
| 0 | 0–240 | the masthead rule draws under the track's name | TREC RAG 2026, what the track asks, and the team | the request arriving on the same paper |
| 1 | 240–510 | the request sheet settles, its words already there | the real narrative, then "One request. 119 of them." | the sheet's text splitting along its seams |
| 2 | 210–480 | the sentence splits into 8 angle strips, staggered | "It asks eight separate things" · 1–8 subnarratives | each strip growing a query block |
| 3 | 480–720 | query blocks extrude from each strip | "Each angle becomes 1–3 BM25 queries" · at most 25 searches | the queries firing into the field |
| 4 | 720–960 | the document field floods in under them | "Up to 1,000 documents per query" | the field cropping to one query's share |
| 5 | 960–1200 | documents cut into passages, scores sort them | "A document becomes passages, scored against the query" · ≤100 per query | the low scores dropping out |
| 6 | 1200–1410 | near-duplicates collapse, MMR spreads the rest | "cosine ≥0.92 drops, MMR λ 0.7 spreads" · ≤40 selected | 40 blocks squaring into a sealed set |
| 7 | 1410–1650 | 20 dashed claim rows rise beside the passages | "≤20 claims — a hint about what to cover, never cited" | the sealed set sliding into generation |
| 8 | 1650–1980 | Sol drafts; Luna's audit marks one group; one edit refused | "Sol writes. Luna checks it against the same passages." · 7–12 calls | the two columns collapsing into one sheet |
| 9 | 1980–2220 | citation markers light in order along the answer | "Every sentence points at a passage that was retrieved." | hold the last frame 2 s |

## SOUND

One retrieved catalog bed of 98 s under an 87 s film, so it enters at 10.5 s:
the track's quiet pad plays under the title card and its first onset lands at
5.7 s, as the card gives way to the request. Volume 0.8, fading in over 1.4 s
and out over the last 4.5 s. Its peak lands at 56 s, on the boundary between
the claims and the writing, and it tails from 78.5 s so the closing line sits
in the decay. No effects: this film is cards arriving on paper, and three
borrowed hits would read as a template rather than as causes on screen.

## BANNED

The default list in full, and specifically: a dark void; glow, bloom, neon,
particles; gradients and glass; a serif display or italic accent words on this
cream (the deck has neither); "01 / 02" numbered labels; monospace caps labels;
pill chips; everything centred; text that moves while it is read; corner
brackets or HUD readouts; three cards in a row; a logo in a corner. Also banned
for this film: any number the deck or the paper does not state, and any claim
that this approach is novel.

## QA

Snapshot and check:

- **f 60** — the request's real words are legible at 30 px, nothing centred.
- **f 400** — eight strips, each with its own words, none clipped.
- **f 840** — the document field reads as a mass, not countable bars.
- **f 1100** — a passage's score bar is visible and sorted.
- **f 1500** — claim rows are dashed and passages are not.
- **f 1900** — the refused edit is legibly refused, not just greyed.
- **f 2150** — every citation marker is lit and the last line holds.

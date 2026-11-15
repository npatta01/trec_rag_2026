# Three storylines — TREC RAG 2026 approach video

## The one thing

A TREC RAG system has to answer a hard, many-part question **and** keep every
sentence it writes traceable to a real retrieved passage. Our approach splits
one request into angles, searches each, narrows thousands of documents to a few
dozen passages, writes from only those, and then has a second model audit the
draft against the same evidence.

**What the viewer remembers:** every sentence in the answer points at a passage
that was actually retrieved — and a second pass catches the ones that don't.

**The spine (one real piece of data):** `rag2026-18`, verbatim from the track —
*six weeks of temporary housing in New York City for a family of three.* It
travels from the first frame to the last.

Facts, all from the deck and the paper: 1–8 subnarratives, 1–3 BM25 queries each
and at most 25 searches per topic; ≤1,000 documents per query; ≤100 passages per
query after reranking; ≤40 selected; ≤20 claims; 1,024-word ceiling; 7–12 model
calls per narrative in multi-stage; 1,167 calls over the run.

---

## A. Eight angles *(the safe one)*

**Device:** one diagram that grows, carrying the worked example.
**Form:** one frame that rebuilds — the composition never cuts away, it
rearranges.

The request is typed once, plain, on paper. It splits into eight angles — *what
options exist, what does it really cost, is it safe for a child* — each angle
sprouting its own queries, and the frame keeps reorganising as the funnel
narrows: documents to passages to selected passages to claims. The viewer feels
the width of the search and then the squeeze.

**First two seconds:** the request's own words, typing on, then the cursor stops
and the sentence quietly splits along its seams into eight strips.

**Beats:** request → it splits into angles → queries fan out → 1,000s of
documents arrive → rerank squeezes to 100 → select to 40 → nuggetize to 20 →
the answer writes itself from them → one line.

**Look:** the deck's own cream page `#fdf1da`, deep teal ink `#07394a`, Inter,
the deck's stage hues on the lanes. Flat editorial, no gloss.

**Sound:** a light pulse per lane as it opens, one low settle on each narrowing.

---

## B. Follow the citation *(recommended)*

**Device:** the result first — the finished answer plays, then one sentence is
pulled and the camera travels back to where it came from.
**Form:** one continuous take that becomes an endless zoom.

The answer is already written and scrolling. The camera settles on one sentence
about illegal short-term rentals, and its citation marker lights. Then the
sentence is lifted off the page and we fall *backwards* through the pipeline —
into the claim that suggested it, the passage that supports it, the reranker
that scored that passage, the query that found it, the angle that asked it, and
finally the original request. We land back on the request, and the whole chain
snaps forward again in one second.

**Why I'd pick this:** it dramatises the one thing instead of describing it. A
stranger who watches it can say what the system guarantees.

**First two seconds:** the finished answer, mid-scroll, already citing — then
everything stops and one sentence stays lit while the rest dims.

**Beats:** finished answer → one sentence lights → fall back through claim,
passage, score, query, angle → land on the request → snap forward → one line.

**Look:** same cream and teal, but the fall is through depth — each stage a
plane the camera passes, the lit sentence the one constant on screen.

**Sound:** a long descending tone through the fall, a single click as each plane
passes, silence at the request, one chord on the snap forward.

---

## C. Two writers, one desk *(the bold one)*

**Device:** a character does the work — two of them. Sol drafts; Luna audits.
**Form:** split screen that collapses to one.

Left pane: Sol writes the answer from the sealed evidence, fast, confident.
Right pane: Luna reads it back line by line against the same passages, and marks
one claim that the evidence does not carry. Sol proposes three edits; Luna
screens them; two pass, one is refused and the draft stands. The panes collapse
into the single finished answer.

It is the only storyline that shows the *disagreement* — the part that is
actually novel in the submission and that the single-pass run does not have.

**First two seconds:** two cursors blinking on the same paragraph, in two panes,
out of step.

**Beats:** sealed evidence → Sol drafts → Luna audits, one group at a time → a
claim fails → three edits proposed → one refused → panes collapse → one line.

**Look:** cream page split by a hairline; Sol's side in the generation violet,
Luna's in the advisory amber; marks and strikes drawn on, never typed.

**Sound:** two typing textures that never sync, a held note under the audit, one
clean stamp on the refusal.

---

## Assumptions (say the word to change any)

- **Length:** 45–60 s, which is where the explainer shape lands for this many
  stages.
- **Aspect:** 1920×1080, 30 fps, to sit in a card on the project page.
- **Words:** on-screen titles only, no voiceover. The deck is silent and a
  narrated cut needs a voice from `/media-use`; happy to add one.
- **Music:** free catalog bed plus sound on the visible actions. Nothing paid.
- **Everything on screen** comes from the deck's figures and the paper's
  numbers. No invented values.

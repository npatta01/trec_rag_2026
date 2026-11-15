# TREC RAG 2026 Final Architecture Figures

These nine SVGs are reusable views of the final TREC RAG 2026 submission
architecture. Their geometry is **conceptual**, not a measurement of throughput,
cost, confidence, or relative volume. They contain aggregate provenance and no
test narrative, corpus passage, prompt, provider response, or generated answer.

Every figure is self-contained: it has a theme-aware light/dark palette,
visible labels, an accessible title and description, and redundant shape/line
semantics so color is never the only carrier of meaning. Solid arrows carry
data or evidence. Dashed arrows carry advisory, fallback, or validation flow.

| Figure | Reader question | Primary source | Text equivalent |
|---|---|---|---|
| `01-whole-system.svg` | How does one frozen source become five accepted files? | `submissions/trec-rag-2026/SUBMISSION_LEDGER.md`; both bundle READMEs | Bounded source retrieval freezes one authenticated artifact. A cache-first branch emits three Retrieval TSVs; a sealed-evidence branch emits two RAG JSONLs. |
| `02-frozen-source-retrieval.svg` | What produced the common source? | `configs/rag26_competition_retrieval_v2.yaml`; `competition_retrieval.py`; `facet_extraction.py` | The original narrative and bounded planned lanes cross Pyserini BM25, document chunking, Mixedbread passage scoring, evidence selection, and advisory hint generation before the source freezes. |
| `03-bounded-deepseek-planning.svg` | Why is source planning bounded? | `facet_extraction.py`; final Retrieval architecture | One DeepSeek call may add zero to eight subnarratives with one to three BM25 queries each. The original lane survives and search never loops back to planning. |
| `04-variable-depth-candidate-core.svg` | How is final Retrieval depth selected? | `retrieval_candidate_core.py`; final Retrieval architecture | Each authenticated lane applies a robust median/MAD gate. Their inclusive union gives a shared narrative-specific core of 1–121 documents, with an original-lane fallback. |
| `05-targeted-scoring-and-runs.svg` | How do three runs share one document set? | `retrieval_baseline_runs.py`; Retrieval metadata | Cache-first Mixedbread scoring completes a targeted candidate-by-semantic-unit matrix. Combo, breadth, and narrative heads reorder identical per-narrative sets. |
| `06-generation-handoff-contract.svg` | What is allowed to enter RAG? | `generation_handoff.py`; RAG bundle README | A canonical handoff carries exact narratives, selected passages, advisory hints, allowed citation IDs, receipts, and hashes. Retrieval TSVs, the ZIP, qrels, gold nuggets, and RAGDoll scores are barred. |
| `07-single-pass-rag.svg` | How does the single-pass run work? | `competition_rag.py`; RAG metadata | Sol receives the complete selected-evidence context. Local schema, citation, exact-hint, and word-limit validation permits one bounded semantic retry before publication. |
| `08-multistage-rag.svg` | How does the multi-stage run bound planning and revision? | `competition_rag_multistage.py`; `narrative_blueprint.py`; `bounded_splice.py`; `operation_screen.py` | Luna plans, audits, and screens; Sol drafts and proposes a bounded splice. Local code applies only safe operations or preserves the validated draft. |
| `09-accepted-submissions.svg` | What proves the exact files are delivery-ready? | submission ledger; validation skill; bundle metadata | Three Retrieval files pass structural validation, two RAG files pass organizer AutoJudge, all hashes match, and Evalbase accepted all five. |

## Visual grammar

- Teal rectangles: source or submission Retrieval.
- Indigo rectangles: RAG Generation.
- Amber fills or dashed lines: advisory or bounded control flow.
- Green capsules: accepted or validated completion.
- Red bars: forbidden input.
- Folded-corner documents: durable artifacts.
- Diamonds: selection or validation gates.
- Cylinders: searched or cached stores.

Human-readable names stay in the diagrams. Exact model, configuration, run,
and artifact identifiers are spelled out in the guide's implementation notes.

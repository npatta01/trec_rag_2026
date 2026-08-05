# TREC RAG 2026 Architecture Figures

These nine SVGs are reusable views of the supported competition path at commit
`7e9d3b9`. Their geometry is **conceptual**, not a measurement of throughput or
relative volume. The guide's extreme-heat narrative is **fictional** and does
not reproduce a TREC test topic, corpus passage, or model answer.

Every figure is self-contained: it has a theme-aware light/dark palette, visible
labels, an accessible title and description, and redundant shape/line semantics
so color is never the only carrier of meaning. Solid arrows carry data or
evidence. Dashed arrows carry advisory, retry, or control flow.

| Figure | Reader question | Primary source | Text equivalent |
|---|---|---|---|
| `01-whole-system.svg` | What is the complete supported path? | `AGENTS.md`; `code/trec_rag/README.md`; `code/trec_rag/competition_retrieval.py`; `code/trec_rag/competition_rag.py` | Narrative enters Retrieval; only an authenticated evidence handoff enters Generation; validated answers become organizer RAG JSONL, while the run and ZIP remain separate Retrieval outputs. |
| `02-retrieval-system.svg` | How does Retrieval widen and narrow? | `configs/rag26_competition_retrieval_v2.yaml`; `code/trec_rag/facet_extraction.py`; `code/trec_rag/competition_retrieval.py` | The original narrative supplies one lane. Up to eight subnarratives supply one to three BM25 queries each: up to 24 planned query lanes and 25 total searches. Every query independently retrieves at most 1,000 documents, ranks chunks, and retains at most 100 passages before evidence selection. |
| `03-bounded-deepseek-planning.svg` | In what sense is planning agentic? | `code/trec_rag/facet_extraction.py`; `code/trec_rag/competition_retrieval.py` | One strict DeepSeek call may add zero to eight subnarratives, each with one to three BM25 queries. The untouched original lane always remains; failure falls back to it, with no open-ended search loop. |
| `04-per-query-candidate-accounting.svg` | What do the 1,000 and 100 ceilings count? | `configs/rag26_competition_retrieval_v2.yaml`; `code/trec_rag/shared_passage_retrieval.py` | For one original or planned BM25 query, at most 1,000 documents become overlapping chunks, Mixedbread scores them as passages, and one global top list retains at most 100 passages. The process repeats per query lane. |
| `05-evidence-and-nuggetizer.svg` | Which material is factual authority? | `configs/rag26_competition_retrieval_v2.yaml`; `code/trec_rag/evidence_local.py`; `code/trec_rag/canonical_nuggets.py` | Local exact-span selection yields up to 40 factual evidence items per subnarrative. Nuggetizer may produce up to 20 advisory hints with up to three supporting documents each; failure leaves extractive evidence intact. |
| `06-generation-handoff-contract.svg` | What may cross into Generation? | `code/trec_rag/generation_handoff.py`; `code/trec_rag/competition_rag.py` | The canonical, hash-authenticated handoff carries narrative, evidence, hints, citation IDs, and receipts. The run, ZIP, qrels, gold nuggets, and RAGDoll scores are barred. |
| `07-sol-generation.svg` | What does Sol receive and return? | `configs/rag26_competition_rag_gpt_sol_v2.yaml`; `code/trec_rag/competition_rag.py` | GPT-5.6 Sol receives a deterministic projection of the handoff, uses medium reasoning and strict structured output under a 12,000-token ceiling, and returns answer objects citing raw allowed document IDs. |
| `08-validation-and-retries.svg` | Why are there two retry limits? | `configs/rag26_competition_rag_gpt_sol_v2.yaml`; `code/trec_rag/competition_rag.py` | Each hosted request has at most three transport attempts. A topic has at most two semantic attempts around generation and local validation, followed by deterministic citation-index conversion. |
| `09-organizer-output-split.svg` | Which stage publishes which organizer artifact? | `code/trec_rag/retrieval_export.py`; `code/trec_rag/competition_retrieval.py`; `code/trec_rag/competition_rag.py` | Retrieval publishes the evidence run, full-text ZIP, and receipt plus the private handoff; Generation consumes only the handoff and publishes organizer RAG JSONL. |

## Visual grammar

- Teal rectangles: Retrieval work.
- Indigo rectangles: Generation work.
- Amber fills or dashed lines: advisory or bounded control flow.
- Red bars: forbidden input or failed path.
- Folded-corner documents: durable artifacts.
- Diamonds: validation gates.
- Cylinders: searched stores.

Human-readable names stay in the diagrams. Exact model and configuration
identifiers are spelled out in the guide's implementation notes.

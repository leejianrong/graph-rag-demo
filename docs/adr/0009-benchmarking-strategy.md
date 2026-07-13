# ADR-0009 — Benchmarking strategy

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

We want to benchmark the pipeline's "capability" (Q30) while burning as few LLM
credits as possible (Q33), with a preference for non-LLM evaluation (Q32). Two
distinct things can be measured:

- **End-to-end retrieval / answer quality** — does the system answer multi-hop
  questions correctly? (The thing Graph RAG is *for*.)
- **Extraction quality** — are NER / coref / EL correct in isolation?

## Decision

**Dual track, with end-to-end as primary.**

**1. End-to-end (primary).** Use a **multi-hop QA dataset**, ingest its provided
context paragraphs as the document corpus, build the graph, then query.
- **Dataset:** **2WikiMultihopQA** — **confirmed choice**. Its evidence is
  expressed as `(entity, relation, entity)` reasoning paths, which mirror the
  knowledge graph and make both construction and retrieval evaluation natural.
  (HotpotQA remains a possible fallback for tooling; MuSiQue if we later want a
  harder, shortcut-resistant set.)
- **Subset:** a fixed **~100–200 questions** (cost + speed).
- **Metrics (all non-LLM):**
  - **Supporting-fact precision / recall / F1** — did retrieval surface the gold
    evidence sentences? Directly scores the retrieval mode.
  - **Answer Exact-Match + token-F1** — using the top-ranked entity node as the
    prediction (retrieval mode), or an LLM-synthesized string (synthesis mode).
    Either way the *scoring* is non-LLM (string comparison vs. gold). The
    prediction is matched against the node's `name` **and its `aliases`**, under
    standard answer normalization (lowercase, strip articles/punctuation/extra
    whitespace — the usual HotpotQA/2Wiki normalization), so a differently
    phrased-but-correct entity is not unfairly scored as wrong.

**2. Extraction (secondary, lightweight).** A sanity check rather than a full
study: run NER in isolation against a small standard set (e.g. CoNLL-2003) or a
small hand-labeled sample from the corpus; spot-check EL merges manually. No
gold corpus-local EL benchmark exists, so this stays lightweight.

**Reproducibility.** Corpus-local entity linking is order-sensitive
([ADR-0004](./0004-corpus-local-entity-linking.md)) — the first document to
mention an entity seeds its canonical record. For benchmark runs, **fix the
ingestion order and EL thresholds** so the constructed graph (and therefore the
scores) are deterministic and comparable across runs.

**Cost controls (Q33, all agreed):** cache every LLM call
([ADR-0008](./0008-llm-abstraction-caching-structured-output.md)); reuse the
pre-built graph so only query-time calls (if any) recur; use the cheaper model
for extraction; run the fixed small subset.

## Consequences

- Primary metrics are **non-LLM and standard**, matching the stated preference.
- Because retrieval mode is free and results are cached, the benchmark can be
  re-run repeatedly at ~no cost.
- Entity-answer questions score cleanly in non-LLM mode; descriptive answers are
  better measured in the optional synthesis mode (still scored non-LLM via
  EM/F1).
- Extraction quality is sanity-checked, not exhaustively measured — a
  deliberate scope choice for a demo.

# ADR-0007 — Graph RAG query: non-LLM retrieval, optional LLM synthesis

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

The pipeline needs a question → answer path (required for benchmarking). A goal
is to avoid the LLM at query time if possible. This requires separating two
things that "answering" conflates:

- **Retrieval** — finding the relevant entities, relationships, and supporting
  sentences for a question. Can be done **fully without an LLM** using
  embeddings + graph traversal.
- **Synthesis** — turning that retrieved evidence into fluent natural-language
  prose. This essentially **needs an LLM** (or brittle templating) for
  multi-hop answers.

## Decision

Build two modes; default to the non-LLM one.

**Retrieval mode (default, no LLM, deterministic, free):**
1. Embed the question with a **local sentence-transformer** (same model family
   as EL — reused).
2. **Vector-anchor** seed nodes via Elasticsearch kNN over **entity embeddings**
   (`ES-Entities`) and **passage/sentence embeddings** (`ES-Documents`).
3. **Expand** k hops from the seed entities in Neo4j to gather the connected
   subgraph.
4. Return a **ranked subgraph + supporting sentences with provenance**. For
   entity-answer questions, the **predicted answer is the top-ranked candidate
   entity node**.

**Answer-synthesis mode (optional, gated LLM, off by default):** feed the
retrieved subgraph + sentences to the LLM to produce prose. Kept separate so the
core path stays free and deterministic.

**Vector storage:** Elasticsearch `dense_vector` fields (Q25) — entity vectors
in `ES-Entities` (shared with EL), passage/sentence vectors in `ES-Documents`.
No separate vector database.

**Entry points (reconciles Q26 + Q39):** two, deliberately different:
- **Ingestion** — a FastAPI endpoint uploads a file to MinIO and **publishes the
  Kafka trigger**; the Kafka consumer remains the first stage of the ingestion
  pipeline.
- **Query** — a synchronous FastAPI `/query` REST endpoint that runs retrieval
  directly (read-only; no Kafka — queries are interactive, not a pipeline).

## Consequences

- Query-time retrieval is **fully local and $0** in API tokens.
- **Honest limitation:** non-LLM retrieval answers **entity-typed questions**
  well (top-ranked node), but **descriptive/explanatory** answers need the
  optional synthesis mode. Benchmark design accounts for this
  ([ADR-0009](./0009-benchmarking-strategy.md)).
- Reusing Elasticsearch for vectors avoids adding another datastore.
- Clear split: Kafka drives ingestion; REST drives queries.

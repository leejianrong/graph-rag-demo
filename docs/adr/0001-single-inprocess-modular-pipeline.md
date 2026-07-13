# ADR-0001 — Single in-process, modular pipeline consumer

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

The pipeline processes one document through five stages: read → NER →
coreference → entity linking → knowledge-graph build. Two topologies were
considered:

- **Microservices** — each stage its own Kafka consumer, chained by
  intermediate topics (`ner.done`, `coref.done`, …). Independently scalable and
  retryable, but many moving parts to run and debug locally.
- **Single in-process consumer** — one service consumes the trigger topic and
  runs all stages in-process for each document.

The project is a local, single-user demo run via Docker Compose (no
multi-tenancy, no public deployment). Simplicity of local operation and
debuggability outweigh independent scaling for now.

## Decision

Implement the pipeline as a **single in-process consumer**. Each stage is a
**separate, swappable module** behind a common stage interface (e.g.
`stages/ner.py`, `stages/coref.py`, `stages/entity_linking.py`,
`stages/kg_build.py`), so stages are unit-testable in isolation and can later be
promoted to independent services without rewriting their logic.

Supporting choices:

- The Kafka trigger message carries only `{bucket, objectKey}`. Within the
  single process, stages hand off **in-memory** (Python objects), not via
  Elasticsearch round-trips. Elasticsearch writes happen at defined
  **checkpoints**, not between every stage:
  - the **raw document text** is written to `ES-Documents` **at ingestion, before
    processing** (per `REQS.md`) — the record is created up front keyed by the
    deterministic document ID;
  - that same record is **enriched in place at the entity-linking checkpoint**
    with NER mentions + coref clusters + per-document EL result (and, later,
    passage/sentence vectors) — the enrichment fields are computed in-memory
    through the stages and persisted together at this one checkpoint, not written
    incrementally per stage;
  - canonical entities are upserted to `ES-Entities` during entity linking;
  - the graph is written to Neo4j at the KG-build stage.
- **Error handling:** log-and-drop per document (no dead-letter queue) for the
  demo.
- **Idempotency:** a deterministic document ID derived from
  `{bucket}/{objectKey}`; reprocessing overwrites prior results.

## Consequences

- Simplest possible local run; the whole flow is traceable in one process.
- A mid-pipeline failure discards the whole document — acceptable because runs
  are cheap and idempotently repeatable.
- No independent per-stage scaling — acceptable now; the modular boundaries keep
  the microservices path open for a future iteration.
- The modular interface is load-bearing: stages must not share hidden state.

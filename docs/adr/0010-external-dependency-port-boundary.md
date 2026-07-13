# ADR-0010 — External-dependency port boundary & fakes-first testing seam

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

The pipeline and query service depend on several external systems — MinIO,
Elasticsearch (two indices), Neo4j, the LLM provider, and a local embedding
model. Two things need to be true for this demo to stay maintainable and
testable:

1. The core logic (orchestration, entity linking, KG-build, retrieval) must be
   **exercisable without Docker** — a fast, deterministic, **$0** test suite that
   makes no live LLM calls and touches no real network.
2. The design must keep the modular boundaries of [ADR-0001](./0001-single-inprocess-modular-pipeline.md)
   honest, so a stage can later be promoted to its own service without rewriting
   its logic.

This decision was made and documented in [`PRD.md`](../PRD.md) (Implementation +
Testing Decisions) but had no ADR of its own; Step E extracts it here so the
decision register is complete.

## Decision

Put **everything outside the pipeline's control behind a narrow interface**
(Python `Protocol`), constructor-injected into the orchestrator and query
service. Real adapters wrap the live services; **in-memory fakes** back the fast
suite. This port boundary is the **single primary seam** the system is tested at.

**The six ports:**

- `ObjectStore` — read/write a document's bytes in MinIO given `{bucket, objectKey}`.
- `DocumentStore` — read/write the `ES-Documents` record (raw text at ingest;
  NER mentions + coref cluster map + per-document EL result + passage/sentence
  vectors written at the EL checkpoint — see [ADR-0001](./0001-single-inprocess-modular-pipeline.md)).
- `EntityStore` — upsert canonical entities and run blocking + kNN search over
  `ES-Entities` (entity vectors, shared between EL and query).
- `GraphStore` — write triples/nodes/edges and run k-hop traversal in Neo4j.
- `LLMClient` — the provider-agnostic client of [ADR-0008](./0008-llm-abstraction-caching-structured-output.md)
  (LiteLLM + response cache + Pydantic-validated structured output).
- `Embedder` — the local sentence-transformer of [ADR-0004](./0004-corpus-local-entity-linking.md)/[ADR-0007](./0007-graph-rag-query-retrieval.md).

**Testing strategy anchored on the seam:**

- **Fast suite (default, $0, no Docker):** both entry points run fully in-process
  against fakes. Ingestion drives `process_document({bucket, objectKey})` and
  asserts the written `ES-Documents` record, upserted `ES-Entities` canonical
  entities, and Neo4j triples with provenance. Query drives the retrieval
  function behind `/query` against pre-seeded fakes and asserts the ranked
  subgraph, supporting sentences, and top-ranked entity answer. The Kafka
  consumer loop and FastAPI endpoints are thin and exercised *through* the seam.
- **Fake `LLMClient` determinism:** returns canned structured responses; the
  prompt-hash response cache ([ADR-0008](./0008-llm-abstraction-caching-structured-output.md))
  doubles as the fixture store, so coref/KG-build/synthesis are deterministic and
  free.
- **Supplementary unit seams** only where internal logic is non-trivial: EL merge
  decision (ADR-0004), provenance offset resolution (ADR-0006), answer
  normalization + EM/F1 (ADR-0009), document ID + idempotency (ADR-0001).
- **Thin real-container contract layer:** a deliberately small set of
  integration tests runs the real adapters against real services (docker-compose
  / testcontainers) — enough to prove each real adapter behaves like its fake (a
  "contract test" per port). Slower; excluded from the fast pre-push loop; gates
  the real adapters, not pipeline logic.

## Consequences

- The whole system is testable without Docker, deterministically, at $0 — the
  stated cost/testability goals hold.
- Ports are the reuse seam that makes shared collaborators single instances
  (`LLMClient`+cache, `Embedder`) rather than per-stage copies — directly
  supporting the caching / reuse decisions (ADR-0004, ADR-0007, ADR-0008).
- Swapping a real service (e.g. a different object store) means writing one
  adapter, not touching pipeline logic.
- Small overhead: every external call goes through an interface + needs a fake.
  Acceptable — it is what makes the fast suite the prior art all future stage
  tests follow (greenfield repo; no in-repo prior art).
- Keeps the microservices path of ADR-0001 open: a port-backed stage can move
  behind a network boundary without changing its callers.

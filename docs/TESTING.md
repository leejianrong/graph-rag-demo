# TESTING — Graph RAG Demo

> Test-strategy audit for the sliced plan. **Consolidates** the testing decisions
> already accepted in [`PRD`](./PRD.md) §Testing Decisions + [`ADR-0010`](./adr/0010-external-dependency-port-boundary.md)
> and **audits** the per-slice Test Plans in [`SLICES.md`](./SLICES.md), flagging
> gaps + suggested fixes. Invents no new strategy; where a test can't be written
> yet it says why and what unblocks it.

---

## 1. What makes a good test here

Assert on **external behavior at a seam** — the document record, canonical
entities, and graph produced by ingesting a document; the ranked subgraph /
supporting sentences / top answer returned by a query; the benchmark scores for a
fixed input — **not** on internal call sequences or private structure. Tests must
be **deterministic** and cost **$0**: no live LLM calls, no wall-clock or network
dependence in the fast suite.

## 2. The one primary seam — the external-dependency port boundary

Per [ADR-0010](./adr/0010-external-dependency-port-boundary.md), all six ports
(`ObjectStore`, `DocumentStore`, `EntityStore`, `GraphStore`, `LLMClient`,
`Embedder`) have in-memory **fakes**. Both entry points run fully in-process
against fakes — no Docker in the fast suite:

- **Ingestion:** drive `process_document({bucket, objectKey})` end-to-end against
  fakes; assert the written `ES-Documents` record (NER mentions+spans, coref
  cluster map, per-doc EL result), the upserted `ES-Entities` canonical entities,
  and the Neo4j triples with provenance. The Kafka consumer loop + FastAPI
  `/ingest` are thin and exercised *through* this seam.
- **Query:** drive the retrieval function behind `/query` against pre-seeded fake
  `EntityStore`/`DocumentStore`/`GraphStore`; assert the ranked subgraph,
  supporting sentences + provenance, and the top-ranked entity answer. HTTP
  contract via FastAPI `TestClient` at the same seam.

**LLM determinism:** the fake `LLMClient` returns canned structured responses; the
`sha256(model+prompt+params)` response cache doubles as the fixture store, so
coref/KG-build/synthesis are deterministic and free. NER is local + deterministic.

## 3. Test tiers

| Tier | Runs against | Scope | In pre-push loop? |
|------|-------------|-------|:-----------------:|
| **Fast suite** | in-memory fakes | all pipeline + query logic, both entry points, unit seams | ✅ yes ($0, no Docker) |
| **Integration / contract** | real adapters via docker-compose / testcontainers | prove each real adapter behaves like its fake (one contract test per port) | ❌ no (slower; gates adapters, not logic) |
| **Benchmark smoke** | small fixture corpus | end-to-end scores are stable + reproducible | ❌ no (separate target) |

The contract layer is deliberately **small** — enough to prove real `EntityStore`
kNN, `GraphStore` Cypher traversal, `DocumentStore`, and `ObjectStore` behave like
their fakes. It gates the real adapters, not pipeline logic.

## 4. Supplementary unit seams (only where internal logic is non-trivial)

- **EL merge decision** (ADR-0004): blocking + similarity + threshold →
  merge-vs-create-new; gated tie-breaker + NIL paths; order-sensitivity explicit.
- **Provenance offset resolution** (ADR-0006): LLM cites a sentence index → spaCy
  segmentation resolves `char_start/char_end` to the correct sentence.
- **Answer normalization + EM/F1** (ADR-0009): standard normalization; match vs
  `name` + `aliases`; differently-phrased-but-correct entity scores correct.
- **Document ID + idempotency** (ADR-0001): deterministic ID from
  `{bucket}/{objectKey}`; reprocessing overwrites, not duplicates.
- **Cache-key stability** (ADR-0008): same `model+prompt+params` → same key;
  changing any → new key.

## 5. Per-slice test-plan audit

Each slice's three-tier plan (End-to-End / Integration / Unit) lives in
[`SLICES.md`](./SLICES.md). Audit of coverage + gaps:

| Slice | E2E | Integration | Unit | Audit |
|-------|:---:|:-----------:|:----:|-------|
| **V1** | ✅ | ✅ MinIO/ES/Kafka | ✅ doc-id | Solid. |
| **V2** | ✅ | ⚠️ none | ✅ | NER is local → no real-service integration test. **Fix:** add a CI **model-availability smoke test** (spaCy `en_core_web_trf` loads) so a missing model fails fast, not mid-pipeline. |
| **V3** | ✅ | ✅ (opt-in) | ✅ cache/schema | Solid; real-provider test stays opt-in. |
| **V4** | ✅ | ✅ ES kNN | ✅ merge boundary | Solid — the heart of R3. Ensure fixtures pin **ingestion order** (order-sensitive). |
| **V5** | ✅ | ✅ Neo4j | ✅ offset res. | **Gap:** no test that **re-ingesting a document overwrites its prior triples/edges** (idempotency at the graph, not just ES). **Fix:** add an E2E re-ingest assertion to V5. |
| **V6** | ✅ | ✅ | ⚠️ ranking | The **subgraph ranking function unit test is blocked on B4** (function not yet pinned). **Fix:** define the ranking function in the V6 spec first, then the unit test asserts its ordering; until then E2E only asserts "top node is correct" on a fixture where ordering is unambiguous. |
| **V7** | ✅ | ✅ (opt-in) | ✅ gate-off | Solid. Only the gated-**on** path is exercised; the gated-off default is the primary assertion. |
| **V8** | ✅ | ✅ (opt-in, slow) | ✅ EM/F1 + SF-P/R/F1 | Solid. Reproducibility rests on B2 + B8 being fixed. |

## 6. Cross-cutting gaps & suggested fixes

1. **Graph re-ingest idempotency (V5)** — the strongest gap. R1.5 idempotency is
   tested at ES (V1) but not at Neo4j. A doc re-ingested must **replace** its
   edges, not duplicate them. → add to V5 E2E.
2. **Ranking function testability (V6)** — B4 must be pinned before its unit test
   is meaningful. → sequence: pin B4 in the V6 spec, then write the test.
3. **spaCy model presence (V2)** — a missing or wrong spaCy model is a silent
   environment failure. → model-availability smoke test in CI.
4. **Gated paths (EL tie-breaker, NIL; V4)** — only the off-by-default behavior is
   tested (correct for v1). When those ship, add gated-on tests. Tracked, not a
   v1 gap.
5. **Whole-pipeline real-stack smoke** — the only place all real adapters run
   together is V8's opt-in integration run. That doubles as the end-to-end
   real-stack smoke; no separate one needed.

## 7. Benchmark determinism (ADR-0009)

With **fixed ingestion order** (B8), **fixed EL thresholds** (B2), and a **warm
response cache**, a run over the fixed subset is reproducible and itself
smoke-testable (small fixture corpus → stable scores). Scoring is non-LLM
throughout (supporting-fact P/R/F1; answer EM/token-F1 vs `name`+`aliases` under
standard normalization).

## 8. Greenfield note

No in-repo prior art (greenfield). The fakes-based fast suite established at V1 is
**the pattern all later stage tests follow** — highest seam first (the port
boundary), dropping to a unit seam only for the non-trivial internal logic in §4.
Real-adapter contract tests follow the standard testcontainers pattern.

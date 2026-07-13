# SPECS BACKLOG — Graph RAG Demo

> Entry point for the **specs phase** (`build-plan-specs`): the per-slice
> implementation plans to write, plus the parameters and assumptions that were
> *deliberately left open* through shaping and must be pinned during specs.
> Source of truth for slice definitions is [`SLICES.md`](./SLICES.md); the "why"
> is in [`adr/`](./adr/) and [`CONTEXT.md`](./CONTEXT.md).

---

## ⚠️ Decide these first — deferred parameters & carried assumptions

These are **not frozen decisions**. They were held open on purpose (tuning knobs
whose values only matter at implementation, and grilling-stage `ASSUMED` defaults
never explicitly confirmed). Pin each one while writing the slice it belongs to —
do not treat the "current default" as settled.

| # | Item | Current default | Pin during | Notes |
|---|------|-----------------|------------|-------|
| **B1** | **Embedding model** | `bge-small-en-v1.5` (local sentence-transformer), reused for EL + query anchoring | V4 (first use) | `ASSUMED` (Q41). Confirm the model + dimension; it sets the ES `dense_vector` mapping and is hard to change after ingestion. |
| **B2** | **EL similarity threshold** | none set | V4 | Governs merge-vs-create-new. Order-sensitive (ADR-0004); must be **fixed** for benchmark reproducibility (ADR-0009). Needs a value + how it's calibrated. |
| **B3** | **k-hop expansion depth** | none set | V6 | How many hops the retriever expands from seed nodes in Neo4j. Trades recall vs. subgraph noise. |
| **B4** | **Subgraph ranking function** | "ranked subgraph" (unspecified) | V6 | The concrete scoring that orders candidate nodes/paths and picks the top-entity answer. Load-bearing for answer EM/F1 — spell out the formula. |
| **B5** | **kNN seeding params** | anchor on both entity + passage/sentence vectors (Q42) | V6 | `ASSUMED`. Number of seeds (top-k) per index, and how entity-seeds vs. passage-seeds are combined. |
| **B6** | **Per-stage LLM model config** | `gpt-4o-mini` for coref + KG-build; fuller model for synthesis (ADR-0008) | V3 (coref), V5 (KG-build), V7 (synth) | Confirm exact model ids + params; the cache key is `sha256(model+prompt+params)`, so these are frozen once benchmarking starts. |
| **B7** | **Closed predicate set** | ~12 starter predicates + `RELATED_TO` fallback (ADR-0006, Q45) | V5 | `ASSUMED` as a starter set; confirm/extend before building the corpus, since predicates shape multi-hop queries. |
| **B8** | **Benchmark corpus + subset** | 2WikiMultihopQA context paragraphs; fixed ~100–200 questions (ADR-0009, Q43) | V8 | `ASSUMED` corpus construction. Pick the exact split/subset + the fixed ingestion order (reproducibility, ADR-0009). |

**Rule of thumb:** anything feeding the response cache key (B6), the vector
mappings (B1), or the constructed graph (B2, B7, B8) is expensive to change after
ingestion — decide those deliberately, not by default.

---

## Slice implementation plans

Per-slice detail (Implementation notes + 3-tier Test Plan) is **folded into
[`SLICES.md`](./SLICES.md)** — one scannable doc, no separate `SLICE-V*.md` files
(streamlining, WORKFLOW-PAINPOINTS #22). Technical architecture is in
[`ARCHITECTURE.md`](./ARCHITECTURE.md); the test-strategy audit in
[`TESTING.md`](./TESTING.md).

| Slice | Delivers | Pins params |
|-------|----------|-------------|
| V1 | Walking skeleton: ingest → stored raw doc | — |
| V2 | NER (typed mentions + spans) | — |
| V3 | Coreference + LLM client/cache | B6 |
| V4 | Entity linking (cross-doc unification) + EL checkpoint | B1, B2 |
| V5 | Graph build (triples + provenance) | B6, B7 |
| V6 | Retrieval (non-LLM multi-hop) | B3, B4, B5 |
| V7 | Gated prose synthesis | B6 |
| V8 | Benchmark harness + metrics | B8 |

**Build order:** V1→V6 is the critical path to the core goal (R0); V7 and V8
hang off V6. The A2 port seam (ADR-0010) is established in V1 and reused by every
later slice. Tickets for these become epic + issue cards in Step I via
`/project-manager-kanban`.

---

## Not in v1 (tracked, deferred — per PRD "Out of Scope")

Frontend/visualization UI · external KB linking · cross-doc coreference (handled
by EL) · microservices / per-stage topics · dead-letter queue · self-hosting
models · auth / multi-tenancy / public deploy · non-English · large-scale
throughput (10⁴–10⁵ docs). Gated-off mechanisms that ship present-but-disabled:
NIL/unlinked retention (R3.6) and the LLM EL tie-breaker (R3.4).

---
shaping: true
---

# SLICES — Graph RAG Demo

> The implementation plan: **Shape A** ([`SHAPING.md`](./SHAPING.md)), detailed
> in [`BREADBOARD.md`](./BREADBOARD.md), broken into ordered **vertical slices**.
> Each slice is a thin end-to-end increment that ends in **observable output**
> and builds on the ones before it. Ground truth for slice definitions and their
> per-slice affordances.
>
> **Built via inline fallback** — the standalone `/breadboarding` slicing recipe
> is unavailable (WORKFLOW-PAINPOINTS #13); this follows `/shaping`'s slicing
> principle: *every slice must end in a demo, not a horizontal layer.*
>
> **"Demo-able" without a frontend.** The UI is out of scope (R7.6), so a slice's
> demo is an *observable artifact*: an HTTP JSON response, an Elasticsearch
> record, a Neo4j Cypher result, or printed benchmark metrics. Every slice's
> acceptance is asserted at the **port seam** (A2) with in-memory fakes — the
> fast, $0 suite from the PRD Testing Decisions — and demoed against the real
> Docker Compose stack.

---

## Why this order (walking skeleton → enrich per stage)

The pipeline is inherently sequential (`read → NER → coref → EL → KG-build`), and
A1 persists at **checkpoints**. So the natural vertical slicing is a **walking
skeleton first** (V1: entry → storage, zero enrichment) then **one stage per
slice**, each adding an inspectable checkpoint. This is still vertical — every
slice runs entry-to-observable-output through the port seam — it just grows the
enrichment depth rather than the breadth of the flow. A "trivial end-to-end graph
in one slice" was rejected: it would require minimal versions of *all* stages up
front, which is a fat first slice, not a thin one.

## Slice map

| Slice | Delivers | New affordances | Demo artifact | Depends on |
|-------|----------|-----------------|---------------|------------|
| **V1** | Walking skeleton: ingest → stored document | U1, N1, N2, N3, N4, N5, N10, N11, N19, N20, N21 | ES-Documents record (text + doc_id) | — |
| **V2** | NER: typed mentions + spans | N6 | Record gains mentions+spans+sentences | V1 |
| **V3** | Coreference: doc-level entities | N7, N14 | Record gains coref cluster map | V2 |
| **V4** | Entity linking: cross-doc unification | N8, N12, N15 | ES-Entities canonical records | V3 |
| **V5** | Graph build: triples + provenance | N9, N13 | Neo4j nodes + provenance edges | V4 |
| **V6** | Retrieval: multi-hop answer (non-LLM) | U2, U3, N16 | `/query` JSON: subgraph + sentences + answer | V5 |
| **V7** | Gated prose synthesis (optional) | N17 | `/query?synthesize=true` prose | V6 |
| **V8** | Benchmark harness + metrics | U4, N18 | Printed P/R/F1 + EM/token-F1 | V6 |

Linear V1→V6 is the critical path to the core goal (R0). V7 and V8 both hang off
V6 and can be built in either order.

---

## V1 — Walking skeleton: ingest → stored document

**Goal:** stand up the full entry→checkpoint path with no enrichment, and
establish the port seam + Docker Compose so every later slice plugs in.

| Affordance | Role in this slice |
|-----------|--------------------|
| U1 `POST /ingest` | Accept a file upload |
| N1 Ingestion handler | Write bytes to MinIO, publish `{bucket, objectKey}` |
| N2 / N3 Kafka topic + consumer | Deliver the trigger, drive `process_document` |
| N4 Orchestrator (shell) | Compute deterministic `document_id`; log-and-drop; create the ES-Documents record with **raw text** at ingestion (before processing) |
| N5 Read stage | Fetch bytes via `ObjectStore` |
| N10 `ObjectStore` (MinIO) | Bytes in/out |
| N11 `DocumentStore` (ES-Documents) | Bare document record |
| N19 / N20 / N21 | `.env` config, logging seam, Docker Compose bring-up |

**Demo:** `docker compose up`; `curl -F file=@a.md /ingest` → returns
`document_id`; the file is in MinIO and a bare ES-Documents record exists.
Re-ingesting the same file **overwrites** (same ID, no duplicate).

**Acceptance (seam):** drive `process_document({bucket, objectKey})` against fake
`ObjectStore`/`DocumentStore`; assert the written record + deterministic ID +
idempotent overwrite; a failing doc is logged and dropped without wedging.

**Proves:** R1.1, R1.2, R1.3, R1.5, R1.6, R1.7, R7.1, R7.2, R7.4, R7.5 + A2 seam.

> ✅ **Write model (Q46, resolved):** raw text is written to ES-Documents **at
> ingestion, before processing**; NER/coref/EL enrichment is held **in-memory**
> through the stages and persisted into that same record at the **entity-linking
> checkpoint** (V4) — not written incrementally per stage. So V1 creates the
> record (raw text); V2/V3 compute enrichment in-memory and are demoed via the
> orchestrator's returned result at the port seam; V4 writes the enriched record.
> (ADR-0001.)

---

## V2 — NER: typed mentions + spans

**Goal:** first real enrichment — local, $0.

| Affordance | Role |
|-----------|------|
| N6 NER stage | spaCy `en_core_web_trf` → typed mentions + char spans + sentence segmentation, in the same pass; orchestrator (N4) runs it and carries the result **in-memory** (persisted at the V4 EL checkpoint, per the write model above) |

**Demo:** ingest a document → the orchestrator's returned result shows
curated-type mentions (PERSON/ORG/LOCATION/DATE/EVENT/NORP/[PRODUCT]) each with
`char_start/end`, plus the sentence segmentation. (Not yet persisted to
ES-Documents — that lands at V4.)

**Acceptance (seam):** assert mentions, types, spans, and sentence boundaries on
the orchestrator's returned NER result for a fixed input; NER is deterministic
and calls no LLM.

**Proves:** R2.1, R2.2, R2.3, R2.5.

---

## V3 — Coreference: doc-level entities

**Goal:** collapse within-document references; introduce the LLM client + cache.

| Affordance | Role |
|-----------|------|
| N7 Coref stage | LLM cluster map (non-destructive) → doc-level entities, carried **in-memory** (persisted at the V4 EL checkpoint) |
| N14 `LLMClient` + cache | First LLM use: LiteLLM, per-stage model, `sha256` response cache, Pydantic-validated structured output + retry |

**Demo:** ingest a doc with pronouns/repeats → the orchestrator result shows a
cluster map grouping mentions to a chosen in-doc canonical; a second identical run
is a cache hit (observably $0 / no API call).

**Acceptance (seam):** fake `LLMClient` returns canned clusters (cache doubles as
fixture); assert cluster map shape, non-destructive original text, valid spans;
assert cache-key stability (`model+prompt+params`).

**Proves:** R2.4, R6.1, R6.2, R6.3.

---

## V4 — Entity linking: cross-document unification

**Goal:** the corpus-local dedup that turns the same real-world entity into one
node — the heart of R3.

| Affordance | Role |
|-----------|------|
| N8 Entity-linking stage | Block by type+normalized name → score by embedding sim over mention-in-context → merge above threshold / else create-new; gated tie-break + NIL both off; **this is the EL checkpoint** — enrich the V1 ES-Documents record in place with NER + coref + per-doc EL, and upsert canonical entities → ES-Entities |
| N12 `EntityStore` (ES-Entities) | Upsert canonical entities; blocking + kNN over entity `dense_vector`s |
| N15 `Embedder` | Local `bge-small-en-v1.5`, shared with query later |

**Demo:** ingest **two** docs naming the same entity differently → **one**
canonical entity in ES-Entities (merge); ingest a doc with a genuinely new entity
→ a **new** canonical record (create-new). `canonical_id` is stable and reused.

**Acceptance (seam):** the ADR-0004 unit seam — same entity across two docs
merges to one canonical ID; a new entity creates one; **order-sensitivity is
explicit in fixtures** (first mention seeds the canonical record); gated paths
stay off by default.

**Proves:** R3.1, R3.2, R3.3, R3.4, R3.5 (+ R3.6 mechanism present, gated off);
R6.4 (fixed thresholds → deterministic).

---

## V5 — Graph build: triples + provenance in Neo4j

**Goal:** materialize the knowledge graph — the thing retrieval traverses.

| Affordance | Role |
|-----------|------|
| N9 KG-build stage | LLM emits `(subject_id, predicate, object_id)` over canonical IDs; closed ~12-predicate set + `RELATED_TO` fallback keeping `raw_predicate`; DATE → edge qualifier; resolve `char_start/end` from N6 segmentation; attach edge provenance; write at checkpoint |
| N13 `GraphStore` (Neo4j) | Multi-label `:Entity:Type` nodes `{canonical_id, name, type, aliases}`; provenance-carrying edges |

**Demo:** ingest a small corpus → Cypher over Neo4j returns multi-label nodes and
edges whose properties include `source_doc_id`, `sentence_index`,
`source_sentence`, `raw_predicate`, `confidence`; a rare relation shows
`RELATED_TO` + preserved `raw_predicate`; a dated fact shows the date as an edge
qualifier (no DATE node).

**Acceptance (seam):** drive ingestion end-to-end against fakes; assert emitted
triples reference canonical IDs (not strings), predicate mapping + fallback,
node/edge model, and the **provenance offset resolution** unit seam (LLM cites a
sentence index → spaCy segmentation resolves offsets to the correct sentence).

**Proves:** R0 (graph exists), R4.1, R4.2, R4.3, R4.4, R4.5, R4.6.

---

## V6 — Retrieval: multi-hop answer (non-LLM)

**Goal:** the payoff — answer a multi-hop, cross-document question with a
connected result, no LLM, free.

| Affordance | Role |
|-----------|------|
| U2 `POST /query` | Accept `{question, synthesize?}` |
| N16 Query retriever | Embed question (N15) → ES kNN seed on entity + passage/sentence vectors (N12/N11) → expand k hops in Neo4j (N13) → rank subgraph + supporting sentences; entity-typed answer = top-ranked node |
| U3 Query response | JSON: predicted answer, ranked subgraph, supporting sentences, per-edge provenance |

**Demo:** `POST /query` a multi-hop question ("how is X connected to Y?") →
response returns the **connected path/subgraph** (not isolated passages), the
supporting sentences with provenance, and a concrete top-entity answer — with no
LLM call.

**Acceptance (seam):** drive the retrieval function behind `/query` against
pre-seeded fake `EntityStore`/`DocumentStore`/`GraphStore`; assert ranked
subgraph, supporting sentences + provenance, and the top-ranked entity answer;
HTTP contract checked via FastAPI `TestClient` at the same seam.

**Proves:** R0, R5.1, R5.2, R5.3, R5.4, R5.5.

---

## V7 — Gated prose synthesis (optional)

**Goal:** opt-in descriptive answers; core path stays free by default.

| Affordance | Role |
|-----------|------|
| N17 Answer synthesizer (gated) | When `synthesize=true`, feed the V6 subgraph + sentences to the LLM (N14) for prose; **off by default** |

**Demo:** `POST /query {synthesize:true}` → prose answer grounded in the same
retrieved evidence; without the flag, the response is identical to V6 (no LLM).

**Acceptance (seam):** fake `LLMClient`; assert prose is produced only when gated
on and that the default path invokes no LLM.

**Proves:** R5.6.

---

## V8 — Benchmark harness + metrics

**Goal:** measure the capability, reproducibly, at ~$0.

| Affordance | Role |
|-----------|------|
| N18 Benchmark harness | Ingest 2WikiMultihopQA context paragraphs as corpus (fixed order) via V1's path; run the fixed ~100–200 subset through V6 retrieval; score non-LLM metrics; reuse warm cache + pre-built graph |
| U4 Benchmark CLI | Run a subset; print metrics |

**Demo:** `benchmark run --subset small` → prints supporting-fact P/R/F1 and
answer EM/token-F1 (scored vs `name` + `aliases` under standard normalization);
a second run reuses the graph + cache and is observably ~$0.

**Acceptance (seam):** small fixture corpus → stable scores across runs (fixed
ingestion order + fixed EL thresholds + warm cache); the **answer
normalization + EM/F1** unit seam (differently-phrased-but-correct entity scores
as correct); scoring uses no LLM.

**Proves:** R8.1, R8.2, R8.3, R8.4, R8.5, R8.6, R6.1, R6.4.

---

## Coverage check — every R lands in a slice

| Requirement chunk | Slice(s) |
|-------------------|----------|
| R1 Ingestion & orchestration | V1 |
| R2 Entity extraction | V2 (NER), V3 (coref) |
| R3 Cross-doc unification | V4 |
| R4 Graph + provenance | V5 |
| R5 Retrieval | V6 (+ V7 for R5.6) |
| R6 Cost/determinism/reproducibility | V3 (cache/structured), V4 (thresholds), V8 (reproducibility) |
| R7 Local run & standards | V1 (compose, config, logging, Python/uv); R7.6/R7.7 are `Out` boundaries |
| R8 Benchmark | V8 |
| R0 Core goal (end-to-end multi-hop) | Realized at V6; measured at V8 |

All Core-goal and Must-have requirements are covered by V1–V6 + V8; the two
Nice-to-haves that are their own mechanism (R5.6, R8.5) land in V7 and V8. R3.6
(NIL retention) ships as gated-off mechanism within V4, per scope.

---

## Next process step

Shaping (Step C) is complete after this slice breakdown. Per the
build-plan-product flow: **D (breadboarding refinement) → E (extract ADRs & final
consistency)**, then implementation planning per slice (V1-plan.md, …) in the
specs phase.

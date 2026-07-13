---
shaping: true
---

# SHAPING — Graph RAG Demo

> The working document for Step C. Ground truth for **requirements (R)**,
> **shapes**, **fit checks**, and (later) the **breadboard**. The "why" lives in
> [`FRAME.md`](./FRAME.md); settled decisions live in [`PRD.md`](./PRD.md),
> [`CONTEXT.md`](./CONTEXT.md) (glossary + D1–D9), and [`adr/0001–0009`](./adr/).
>
> **R states what's *needed* (the problem/constraint space), not how it's
> satisfied.** Satisfaction is shown only in a fit check (R × shape). Because
> the nine ADRs already fix most mechanisms, expect most R to be Must-have and
> later satisfied — R here is the yardstick the shapes are measured against, not
> an invitation to re-decide settled work.

**Status legend:** `Core goal` · `Must-have` · `Nice-to-have` · `Out` (tracked
but not pursued) · `Undecided` / `Leaning yes/no` (still open).

---

## Requirements (R)

| ID | Requirement | Status |
|----|-------------|--------|
| **R0** | **Turn a corpus of unstructured text (Markdown/plain text) into a queryable knowledge graph that answers multi-hop, cross-document questions.** | **Core goal** |
| **R1** | **Hands-off ingestion & orchestration** | Must-have |
| R1.1 | A single upload action stores a document and kicks off its processing, without manually touching the queue or object storage. | Must-have |
| R1.2 | The trigger to process a document carries only a locator (`{bucket, objectKey}`); document state is read from storage, not from the message. | Must-have |
| R1.3 | All processing steps run for a document as one traceable pass that can be debugged locally. | Must-have |
| R1.4 | Each processing step is an independently testable, swappable unit (no hidden shared state). | Must-have |
| R1.5 | Re-submitting the same document overwrites its prior results (stable, deterministic identity) rather than duplicating. | Must-have |
| R1.6 | A failed document is isolated (logged and dropped) and never wedges the pipeline; it can simply be re-triggered. | Must-have |
| R1.7 | Durable state is inspectable after processing without excessive intermediate storage round-trips. | Must-have |
| **R2** | **Faithful, low-cost entity extraction** | Must-have |
| R2.1 | Identify and type the entity mentions in each document (people, organizations, locations, dates, events, …). | Must-have |
| R2.2 | Restrict entity types to a curated set so the graph stays clean and multi-hop paths remain readable. | Must-have |
| R2.3 | Retain character spans for every mention (for downstream alignment, provenance, and later UI highlighting). | Must-have |
| R2.4 | Resolve within-document coreference so pronouns/repeated references collapse to one doc-level entity, non-destructively. | Must-have |
| R2.5 | The high-volume, per-token extraction work costs ~$0 (no per-token API spend). | Must-have |
| **R3** | **Cross-document entity unification** | Core goal |
| R3.1 | The same real-world entity mentioned across different documents becomes one graph node. | Core goal |
| R3.2 | Unification treats the corpus itself as the authority — it must not depend on an external/hosted knowledge base. | Must-have |
| R3.3 | A genuinely new entity is always captured (a create-new path is always available). | Must-have |
| R3.4 | Core unification runs locally at ~$0; higher-quality disambiguation is opt-in and off by default. | Must-have |
| R3.5 | One stable entity identity serves as both the dedup/merge key and the graph node identity. | Must-have |
| R3.6 | Optionally retain low-confidence / unlinked entities. | Nice-to-have |
| **R4** | **Readable, provenance-carrying knowledge graph** | Must-have |
| R4.1 | Facts are stored as subject–predicate–object relationships grounded in unified entity identities, not raw strings. | Must-have |
| R4.2 | Relationships use a consistent, bounded vocabulary with an open fallback, so rare relations are never silently lost. | Must-have |
| R4.3 | Entity types are first-class nodes; date is a relationship qualifier, not its own node (keeps paths readable). | Must-have |
| R4.4 | Entities can be queried uniformly across all types or narrowed to one type. | Must-have |
| R4.5 | Every fact/edge records provenance — source document + exact source sentence — so any answer is traceable to its origin. | Core goal |
| R4.6 | Provenance character offsets are resolved reliably (not by asking the LLM to count characters). | Must-have |
| **R5** | **Connected multi-hop retrieval with traceable evidence** | Core goal |
| R5.1 | Questions are asked through a synchronous, read-only query interface (interactive, not queue-driven). | Must-have |
| R5.2 | The default retrieval path is fully non-LLM, deterministic, and free. | Core goal |
| R5.3 | Retrieval returns a *connected* multi-hop result (a subgraph/path), not just independently-matching passages. | Core goal |
| R5.4 | The response includes the supporting sentences with provenance, so the user can see *why* an answer was produced. | Must-have |
| R5.5 | For an entity-typed question, a concrete answer is produced without invoking an LLM. | Must-have |
| R5.6 | Optional prose answer-synthesis over the retrieved evidence, gated and off by default. | Nice-to-have |
| **R6** | **Cost control, determinism & reproducibility** | Must-have |
| R6.1 | Every LLM call is cached so repeated pipeline runs and benchmark re-runs cost ~$0. | Must-have |
| R6.2 | LLM outputs are structured and validated so parsing stays reliable across providers. | Must-have |
| R6.3 | The LLM provider/model is swappable via config; a cheap model handles extraction, a fuller model only where needed. | Must-have |
| R6.4 | Given fixed inputs and settings, ingestion and retrieval are reproducible (order- and threshold-stable). | Must-have |
| **R7** | **Local, single-command operation & code standards** | Must-have |
| R7.1 | The entire stack runs locally and comes up with a single command; no hosted or authenticated dependency. | Core goal |
| R7.2 | Secrets and config are supplied via environment, kept out of code. | Must-have |
| R7.3 | LLM inference is external API only; no self-hosting of models. | Must-have |
| R7.4 | Modular Python 3.12, dependencies managed with `uv`, docstrings + type hints. | Must-have |
| R7.5 | Observability via basic logging behind a seam (structured/JSON logging swappable later). | Must-have |
| R7.6 | No authentication, no multi-tenancy, no public deployment. | Out |
| R7.7 | English only. | Out |
| **R8** | **Measured multi-hop capability** | Must-have |
| R8.1 | Primary benchmark is end-to-end multi-hop QA on a standard dataset (2WikiMultihopQA), ingesting its context paragraphs as the corpus. | Must-have |
| R8.2 | A fixed, small question subset (~100–200) keeps runs fast and cheap. | Must-have |
| R8.3 | Metrics are non-LLM: supporting-fact precision/recall/F1 and answer Exact-Match/token-F1. | Must-have |
| R8.4 | Answer scoring matches against the entity `name` **and** its `aliases` under standard normalization, so a differently-phrased-but-correct answer isn't penalized. | Must-have |
| R8.5 | A lightweight secondary extraction sanity check (NER vs. a small labeled set; manual EL-merge spot-checks). | Nice-to-have |
| R8.6 | Benchmark re-runs cost ~$0 (cache + reuse the pre-built graph + cheap extraction model + small subset). | Must-have |

**Chunking note:** 9 top-level requirements (R0 + R1–R8), at the skill's cap.
The 49 PRD user stories collapse into these families; sub-requirements carry the
detail. Traceability to the PRD stories:

- **R1** ← stories 1–8 (ingestion/orchestration)
- **R2** ← stories 9–14 (NER + coreference)
- **R3** ← stories 15–21 (entity linking)
- **R4** ← stories 22–28 (KG builder + graph model)
- **R5** ← stories 29–34 (query/retrieval)
- **R6** ← stories 35–38 (LLM abstraction/caching/structured output)
- **R7** ← stories 46–49 (ops/config/observability) + REQS non-goals
- **R8** ← stories 39–45 (benchmarking)

**Deferred / Out (tracked, not pursued in v1 — per PRD "Out of Scope"):**
frontend/visualization UI (Svelte+Vite), coreference *as* the cross-doc
mechanism (handled by R3 instead), microservices / per-stage topics, dead-letter
queue, self-hosting models, auth/multi-tenancy/public deploy (R7.6), non-English
(R7.7), and large-scale throughput (10⁴–10⁵ docs). NIL/unlinked retention (R3.6)
and the LLM EL tie-breaker (R3.4) ship gated off.

---

## Shapes

**Only one shape.** The nine Accepted ADRs already fix the architecture, so there
is no live bake-off — C3 records the decided approach as a single
**shape-of-record** (Shape A) and runs one confirming fit check, rather than
reconstructing rejected alternatives (see WORKFLOW-PAINPOINTS #16 for why). Each
part is a vertical slice (mechanism + the data it owns), traced to its ADR.

### A: ADR-decided Graph RAG architecture

| Part | Mechanism | Flag |
|------|-----------|:----:|
| **A1** | **Ingestion entry & orchestration** (ADR-0001) — FastAPI upload → write bytes to MinIO → publish `{bucket, objectKey}` Kafka trigger. Thin Kafka consumer resolves a trigger to `process_document({bucket, objectKey})`. Deterministic **document ID** from `{bucket}/{objectKey}` (reprocess overwrites). Log-and-drop on failure (no DLQ). In-memory stage handoff; ES/Neo4j writes only at checkpoints. | |
| **A2** | **External-dependency port seam** (ADR-0010) — six `Protocol` ports (`ObjectStore`, `DocumentStore`, `EntityStore`, `GraphStore`, `LLMClient`, `Embedder`), constructor-injected into orchestrator + query service. Real adapters wrap live services; in-memory fakes back the fast suite. The one primary test seam. | |
| **A3** | **Local NER stage** (ADR-0002) — spaCy `en_core_web_trf` (fallback `en_core_web_lg`). Curated types (PERSON, ORG, LOCATION=GPE+LOC, DATE, EVENT, NORP, opt. PRODUCT). Character spans per mention + sentence segmentation in the same pass. No LLM. | |
| **A4** | **Within-document coreference stage** (ADR-0003) — LLM-backed; emits a non-destructive **cluster map** (mention → chosen in-doc canonical). Clusters become the **doc-level entities** handed to EL. | |
| **A5** | **Corpus-local entity-linking stage** (ADR-0004) — block by type + normalized name → score by sentence-transformer similarity over mention-in-context → merge above threshold / else create-new (always on). Gated LLM tie-breaker + gated NIL path, both off by default. `canonical_id` = merge key = node identity. Owns writes: per-doc EL → ES-Documents; canonical entities → ES-Entities. | |
| **A6** | **Storage split** (ADR-0005) — one ES cluster, two indices. `ES-Documents`: text, NER mentions+spans, coref map, per-doc EL, passage/sentence `dense_vector`s. `ES-Entities`: one canonical entity per record + entity `dense_vector`. Backs `DocumentStore` + `EntityStore` ports. | |
| **A7** | **KG builder + Neo4j model** (ADR-0006) — schema-guided LLM emits `(subject_id, predicate, object_id)` over canonical IDs. Closed ~12-predicate set + `RELATED_TO` fallback keeping `raw_predicate`. Multi-label `:Entity:Type` nodes `{canonical_id, name, type, aliases}`; DATE = edge qualifier, not a node. Edge provenance `{source_doc_id, sentence_index, source_sentence, raw_predicate, confidence}`; LLM cites sentence index, spaCy resolves `char_start/char_end`. Backs `GraphStore`. | |
| **A8** | **Shared local embedder** (ADR-0004/0007) — one sentence-transformer (`bge-small-en-v1.5`) reused for EL scoring *and* query-time anchoring; embeds both canonical entities and passages/sentences. Backs `Embedder`. | |
| **A9** | **Query / retrieval service** (ADR-0007) — synchronous read-only FastAPI `/query`, no Kafka. Default non-LLM path: embed question → ES kNN over entity + passage/sentence vectors → expand k hops in Neo4j → return ranked subgraph + supporting sentences w/ provenance; entity-typed answer = top-ranked node. Optional gated LLM answer-synthesis, off by default. | |
| **A10** | **LLM client + cache + structured output** (ADR-0008) — LiteLLM, per-stage model config (`gpt-4o-mini` extraction; fuller model for synthesis). Persistent cache keyed `sha256(model+prompt+params)`; implicit invalidation. Pydantic-validated structured output with retry. Backs `LLMClient`. | |
| **A11** | **Benchmark harness** (ADR-0009) — ingest 2WikiMultihopQA context paragraphs as corpus; fixed ~100–200 question subset. Non-LLM metrics: supporting-fact P/R/F1; answer EM/token-F1 vs `name`+`aliases` under standard normalization. Fixed ingestion order + fixed EL thresholds; warm-cache reuse of the pre-built graph. | |
| **A12** | **Ops, config & standards** (PRD/REQS) — one Docker Compose (Kafka, MinIO, ES cluster w/ two indices, Neo4j, pipeline/API service). `.env`/env-var config. Basic `logging` behind a seam. Python 3.12, `uv`, modular layout, docstrings + type hints. | |

**No flagged unknowns (⚠️).** Every part's *mechanism* is concretely specified by
its ADR. Remaining choices (k-hop depth, EL similarity threshold values, exact
subgraph ranking function, embedding-model swap) are **tuning parameters**, not
flagged unknowns — the "how" is understood; the values get set in specs (Step F).

### Fit Check: R × A (single confirming pass)

Because A *is* the decided architecture, this is a confirming pass, not a
decision — no ❌ is expected. `Out` rows (R7.6, R7.7) are scope boundaries A
honors by *not* building them; ✅ means "boundary respected." Full sub-
requirement rollup below; every row maps to the Part(s)/ADR that satisfy it.

| Req | Requirement | Status | A |
|-----|-------------|--------|---|
| R0 | Turn a text corpus into a queryable KG answering multi-hop, cross-document questions | Core goal | ✅ |
| R1.1 | Single upload action stores a doc and kicks off processing | Must-have | ✅ |
| R1.2 | Trigger carries only `{bucket, objectKey}`; state read from storage | Must-have | ✅ |
| R1.3 | All steps run for a doc as one traceable, locally-debuggable pass | Must-have | ✅ |
| R1.4 | Each step independently testable & swappable (no hidden shared state) | Must-have | ✅ |
| R1.5 | Re-submitting the same doc overwrites prior results, never duplicates | Must-have | ✅ |
| R1.6 | A failed doc is isolated (log-and-drop), never wedges the pipeline | Must-have | ✅ |
| R1.7 | Durable state inspectable without excessive intermediate round-trips | Must-have | ✅ |
| R2.1 | Identify and type entity mentions | Must-have | ✅ |
| R2.2 | Curated type set for a clean, readable graph | Must-have | ✅ |
| R2.3 | Retain character spans for every mention | Must-have | ✅ |
| R2.4 | Resolve within-document coreference non-destructively | Must-have | ✅ |
| R2.5 | High-volume per-token extraction costs ~$0 | Must-have | ✅ |
| R3.1 | Same real-world entity across docs → one graph node | Core goal | ✅ |
| R3.2 | Corpus as authority — no external/hosted KB dependency | Must-have | ✅ |
| R3.3 | A genuinely new entity is always captured | Must-have | ✅ |
| R3.4 | Core unification local at ~$0; better disambiguation opt-in, off by default | Must-have | ✅ |
| R3.5 | One identity = dedup/merge key = graph node identity | Must-have | ✅ |
| R3.6 | Optionally retain low-confidence / unlinked entities | Nice-to-have | ✅ |
| R4.1 | Facts as S-P-O grounded in unified entity IDs, not raw strings | Must-have | ✅ |
| R4.2 | Bounded relationship vocabulary + open fallback (no lost relations) | Must-have | ✅ |
| R4.3 | Entity types are nodes; date is an edge qualifier | Must-have | ✅ |
| R4.4 | Entities queryable uniformly or by type | Must-have | ✅ |
| R4.5 | Every edge records provenance (source doc + exact sentence) | Core goal | ✅ |
| R4.6 | Provenance offsets resolved reliably (not by the LLM) | Must-have | ✅ |
| R5.1 | Synchronous, read-only query interface (not queue-driven) | Must-have | ✅ |
| R5.2 | Default retrieval fully non-LLM, deterministic, free | Core goal | ✅ |
| R5.3 | Retrieval returns a connected multi-hop result, not isolated passages | Core goal | ✅ |
| R5.4 | Response includes supporting sentences with provenance | Must-have | ✅ |
| R5.5 | Entity-typed question → concrete answer without an LLM | Must-have | ✅ |
| R5.6 | Optional gated prose answer-synthesis, off by default | Nice-to-have | ✅ |
| R6.1 | Every LLM call cached → repeated/benchmark re-runs ~$0 | Must-have | ✅ |
| R6.2 | Structured, validated LLM output → reliable parsing | Must-have | ✅ |
| R6.3 | Provider/model swappable; cheap extraction model, fuller only where needed | Must-have | ✅ |
| R6.4 | Fixed inputs+settings → reproducible ingestion & retrieval | Must-have | ✅ |
| R7.1 | Whole stack runs locally, one-command bring-up; no hosted dependency | Core goal | ✅ |
| R7.2 | Secrets/config via environment, out of code | Must-have | ✅ |
| R7.3 | LLM inference external API only; no self-hosting | Must-have | ✅ |
| R7.4 | Modular Python 3.12, `uv`, docstrings + type hints | Must-have | ✅ |
| R7.5 | Basic logging behind a seam | Must-have | ✅ |
| R7.6 | No auth / multi-tenancy / public deployment | Out | ✅ |
| R7.7 | English only | Out | ✅ |
| R8.1 | Primary benchmark: end-to-end multi-hop QA on 2WikiMultihopQA | Must-have | ✅ |
| R8.2 | Fixed small subset (~100–200 questions) | Must-have | ✅ |
| R8.3 | Non-LLM metrics: supporting-fact P/R/F1, answer EM/token-F1 | Must-have | ✅ |
| R8.4 | Answer scoring vs `name`+`aliases` under standard normalization | Must-have | ✅ |
| R8.5 | Lightweight secondary extraction sanity check | Nice-to-have | ✅ |
| R8.6 | Benchmark re-runs ~$0 (cache + reuse graph + cheap model + subset) | Must-have | ✅ |

**Notes (which Part/ADR satisfies each chunk):**
- **R1** ← A1 (orchestration, doc ID, log-and-drop, checkpoints) + A2 (swappable units via the port seam). ADR-0001.
- **R2** ← A3 (NER types, spans, $0) + A4 (within-doc coref). ADR-0002/0003.
- **R3** ← A5 (block/score/merge/create-new, gated tie-breaker+NIL, canonical_id) + A6 (ES-Entities substrate) + A8 (local embedder → $0). ADR-0004.
- **R4** ← A7 (triples over canonical IDs, closed+fallback predicates, node/edge model, provenance, offset resolution). ADR-0006.
- **R5** ← A9 (sync `/query`, non-LLM kNN + k-hop, ranked subgraph + provenance, top-node answer, gated synthesis) + A6/A8 (vectors). ADR-0007.
- **R6** ← A10 (LiteLLM, cache, structured output) + A11 (fixed order/thresholds → reproducibility). ADR-0008/0009.
- **R7** ← A12 (Docker Compose, .env, logging seam, Python 3.12/uv). R7.6/R7.7 are boundaries A respects by omission; R7.3 satisfied via A10's external-API client.
- **R8** ← A11 (2WikiMultihopQA, subset, non-LLM metrics, normalization, cost controls). ADR-0009.

**Result:** A passes all Core-goal and Must-have requirements and covers the
Nice-to-haves as gated/optional mechanisms; `Out` boundaries are respected. No
❌, no ⚠️ — the shape is complete and ready to **Detail** (C4).

---

## Detail A

**Done — see [`BREADBOARD.md`](./BREADBOARD.md).** Shape A is detailed there into
4 UI affordances (U1–U4), 21 Non-UI affordances (N1–N21, including the six ports),
and a Place-grouped Mermaid wiring diagram rendered from those tables. Built via
inline fallback (the standalone `/breadboarding` skill is absent — WORKFLOW-
PAINPOINTS #13). The breadboard also surfaces 6 candidate slice boundaries for C5.

## Slices

**Done (C5) — see [`SLICES.md`](./SLICES.md).** 8 vertical slices (V1 walking
skeleton → V6 multi-hop retrieval on the critical path; V7 + V8 off V6). Every
Core-goal/Must-have requirement is covered; each slice ends in an observable
artifact and is asserted at the A2 port seam. Shaping (Step C) is complete.

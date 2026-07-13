# PRD — Graph RAG Demo

> Product requirements for a locally-run Graph RAG pipeline: Kafka-triggered
> ingestion of text/Markdown → local spaCy **NER** → LLM within-document
> **coreference** → **corpus-local entity linking** (no external KB) →
> **knowledge-graph build** into Neo4j, with a non-LLM **vector-anchored +
> graph-expansion** retrieval path (optional gated LLM answer synthesis),
> benchmarked on 2WikiMultihopQA.
>
> This PRD synthesizes the settled decisions. It does **not** re-open them.
> Sources of truth: [`REQS.md`](./REQS.md), [`CONTEXT.md`](./CONTEXT.md)
> (glossary + decision register D1–D9), and [`docs/adr/0001–0009`](./adr/).
> Terms in **bold** are defined in the CONTEXT glossary and used with that
> exact meaning throughout.

---

## Problem Statement

I monitor large amounts of unstructured text — news sources, documents — and I
need to find connections between entities (people, organizations, places,
events) that are *spread across different documents*. Plain keyword or
single-document search can't answer the questions I actually care about, which
are **multi-hop** and **cross-document**: "how is X connected to Y", where the
link only exists by chaining facts that appear in separate files. Standard
vector RAG retrieves individual passages that mention the query terms but misses
the connected path between them, so the big picture is lost.

Separately, I'm a software engineer learning Graph RAG, and I want a working,
inspectable, locally-runnable system that demonstrates the full pipeline —
ingestion through knowledge-graph construction through retrieval — and lets me
**benchmark** its multi-hop question-answering capability with standard,
reproducible metrics, all without burning LLM API credits.

## Solution

A single, modular Graph RAG **pipeline** that runs entirely on my machine via
Docker Compose. I drop a Markdown or plain-text **document** into object storage
and publish a **trigger message**; the pipeline reads it, extracts typed entity
**mentions** locally (**NER**), groups co-referring mentions within the document
(**coreference**), resolves each **doc-level entity** to a **canonical entity**
in a **corpus-local entity store** (**entity linking** — the mechanism that lets
the same real-world entity from different documents become one graph node),
extracts **triples** grounded in canonical entity IDs, and writes them into a
Neo4j **knowledge graph** with edge-level **provenance**.

I then ask questions through a synchronous REST `/query` endpoint. By default,
retrieval is **fully non-LLM and free**: embed the question, vector-anchor onto
seed entities/passages, expand k hops in the graph, and return a ranked subgraph
plus supporting sentences with provenance (the predicted answer for
entity-typed questions is the top-ranked entity node). An optional, gated LLM
**answer-synthesis** mode turns that evidence into prose when I want it.

Finally, a benchmark harness ingests the **2WikiMultihopQA** context paragraphs
as the corpus, builds the graph, runs a fixed question subset, and reports
standard non-LLM metrics (supporting-fact P/R/F1, answer EM/token-F1), re-runnable
at ~$0 thanks to response caching.

## User Stories

### Ingestion & pipeline orchestration

1. As an operator, I want to upload a text/Markdown file through a single API
   call that stores it in object storage and publishes a **trigger message**,
   so that I can kick off ingestion without touching Kafka or MinIO directly.
2. As an operator, I want the **trigger message** to carry only
   `{bucket, objectKey}`, so that the queue stays simple and all document state
   lives in storage, not in the message.
3. As an operator, I want the pipeline to consume a trigger and run all five
   **stages** (read → NER → coreference → entity linking → KG-build) in one
   in-process pass for that document, so that the whole flow is traceable in a
   single process and easy to debug locally.
4. As a developer, I want each **stage** to be a separate, swappable module
   behind a common interface, so that I can unit-test stages in isolation and
   later promote any stage to its own service without rewriting its logic.
5. As an operator, I want a deterministic **document ID** derived from
   `{bucket}/{objectKey}`, so that re-publishing the same file overwrites its
   prior results rather than creating duplicates.
6. As an operator, I want a failed document to be logged and dropped (no
   dead-letter queue) for this local demo, so that one bad document never wedges
   the pipeline, and I can simply re-trigger it.
7. As an operator, I want the raw document text written to ES-Documents **at
   ingestion (before processing)**, then that same record enriched in place at
   defined **checkpoints** (NER mentions + coref clusters + per-document EL result
   at the entity-linking checkpoint; canonical entities during entity linking;
   graph at KG-build), so that the original is durably captured up front while
   in-between stage handoff stays in-memory and I avoid needless round-trips.
8. As an operator, I want a seed/loader path (the ingestion endpoint) that
   uploads a file to MinIO and publishes the Kafka trigger, so that I can drive
   the demo end-to-end from empty stores.

### Named entity recognition (NER)

9. As a cost-conscious operator, I want **NER** performed locally with spaCy
   (`en_core_web_trf`), so that this high-volume per-token stage costs **$0** in
   API tokens.
10. As a graph consumer, I want NER limited to a curated type set — PERSON, ORG,
    LOCATION (merging GPE+LOC), DATE, EVENT, NORP, and optionally PRODUCT — so
    that the resulting graph is cleaner and multi-hop paths are more readable.
11. As a downstream stage, I want every **mention** to retain character offsets
    (spans), so that coreference can align mentions, provenance can cite exact
    sentence/char ranges, and dedup/UI highlighting are possible later.
12. As an operator, I want spaCy's sentence segmentation produced in the same
    NER pass, so that the **provenance** model has trustworthy sentence
    boundaries without a second parse.

### Coreference resolution

13. As a graph consumer, I want **coreference** resolved *within a single
    document* (LLM-backed), so that pronouns and repeated references collapse to
    one **doc-level entity** while cross-document identity is deferred to entity
    linking.
14. As a downstream stage, I want coreference output as a **cluster map**
    (original text preserved; each cluster maps mentions to a chosen canonical
    in-document mention), so that it is non-destructive, keeps spans valid, and
    feeds both EL and provenance.

### Entity linking (corpus-local)

15. As a graph consumer, I want each **doc-level entity** matched against a
    **corpus-local entity store** (no external KB), so that the same real-world
    entity mentioned in different documents becomes one **canonical entity** and
    therefore one graph node.
16. As an operator, I want entity linking to block candidates by entity type +
    normalized name, then score with local sentence-transformer embedding
    similarity over the mention-in-context, so that resolution runs locally and
    costs **$0**.
17. As an operator, I want above-threshold matches merged into the existing
    **canonical entity** and everything else to **create a new canonical
    entity** (create-new is the normal, always-on path), so that every genuinely
    new entity is captured.
18. As a cost-conscious operator, I want the LLM tie-breaker for borderline
    matches to be **gated off by default**, so that the core path stays free and
    I opt in only when I want higher-quality disambiguation.
19. As an operator, I want an optional, gated "unlinked/NIL entity" path for
    mentions that can't be confidently linked, so that I can switch on retaining
    them within the corpus in a later iteration without it being on by default.
20. As a developer, I want the **canonical entity ID** to be the merge key and
    the graph node identity, so that the EL store and the knowledge graph stay
    consistent and multi-hop traversal is reliable.
21. As an operator, I want per-document EL results stored with the document in
    **ES-Documents** and deduplicated canonical entities stored in
    **ES-Entities**, so that the two very different shapes/write-patterns stay
    cleanly separated.

### Knowledge-graph builder & graph model

22. As a graph consumer, I want the KG-builder to receive the document text
    **and that document's canonical linked entities** and emit **triples**
    `(subject_id, predicate, object_id)` referencing canonical entity IDs (not
    raw strings), so that edges are grounded in the dedup store.
23. As a graph consumer, I want a closed **predicate** set (~12: `LOCATED_IN`,
    `PART_OF`, `MEMBER_OF`, `WORKS_FOR`, `HAS_ROLE`, `FOUNDED`, `OWNS`,
    `PRODUCES`, `PARTICIPATED_IN`, `OCCURRED_ON`, `AFFILIATED_WITH`, and
    `RELATED_TO` as fallback), so that multi-hop queries and benchmarking are
    consistent.
24. As a graph consumer, I want an open fallback that emits `RELATED_TO` and
    preserves the model's original phrasing in a `raw_predicate` edge property,
    so that rare relations are never silently lost.
25. As a graph consumer, I want first-class entity types (PERSON, ORG, LOCATION,
    EVENT, PRODUCT, NORP) modeled as nodes and DATE modeled as an **edge
    attribute/qualifier**, so that the graph isn't cluttered by date nodes and
    multi-hop paths stay readable.
26. As a Neo4j user, I want multi-label nodes — a shared `:Entity` label plus a
    type label (`:Entity:Person`, `:Entity:Organization`, …) with properties
    `canonical_id`, `name`, `type`, `aliases` — so that I can query all entities
    uniformly or one type via its label with per-label indexes.
27. As an analyst, I want every edge to record **provenance** —
    `source_doc_id`, `sentence_index`, `source_sentence` text, `raw_predicate`,
    and `confidence` — so that any answer can be traced back to the exact source
    sentence.
28. As a developer, I want the LLM to cite only the **sentence index** per
    triple and our own spaCy segmentation to resolve `char_start`/`char_end`, so
    that offsets are trustworthy (the model cannot count characters reliably).

### Graph RAG query / retrieval

29. As an analyst, I want a synchronous, read-only REST `/query` endpoint
    (no Kafka — queries are interactive, not a pipeline), so that I can ask
    questions and drive the benchmark programmatically.
30. As a cost-conscious analyst, I want default retrieval to be **fully non-LLM,
    deterministic, and free**: embed the question with the local
    sentence-transformer, vector-anchor seed nodes via ES kNN over **entity
    embeddings** (ES-Entities) and **passage/sentence embeddings**
    (ES-Documents), then expand k hops in Neo4j.
31. As an analyst, I want the retrieval response to be a ranked subgraph plus
    supporting sentences with provenance, so that I can see *why* an answer was
    produced and trace evidence.
32. As an analyst asking an entity-typed question, I want the predicted answer to
    be the top-ranked candidate entity node, so that I get a concrete answer
    without invoking an LLM.
33. As an analyst wanting prose, I want an optional, **gated** LLM
    answer-synthesis mode that consumes the retrieved subgraph + sentences, so
    that descriptive/explanatory answers are possible while the core path stays
    free by default.
34. As a developer, I want vectors stored in Elasticsearch `dense_vector` fields
    (entity vectors in ES-Entities shared with EL; passage/sentence vectors in
    ES-Documents), so that no separate vector database is introduced.

### LLM abstraction, caching, structured output

35. As a cost-conscious operator, I want a provider-agnostic LLM client (LiteLLM)
    with per-stage model config, so that I can use `gpt-4o-mini` for high-volume
    extraction and a fuller model only for optional synthesis, and swap in any
    OpenAI-compatible provider (incl. DeepSeek) via `.env`.
36. As a cost-conscious operator, I want a persistent response cache keyed by
    `sha256(model + prompt + params)`, so that repeated pipeline runs and
    benchmark re-runs hit the cache and cost **nothing**.
37. As a developer, I want LLM calls to request structured output validated
    against Pydantic models (coref clusters, links, triples) with retry on parse
    failure, so that parsing stays reliable across providers with varying JSON
    support.
38. As a developer, I want cache invalidation to be implicit — changing the
    prompt or model changes the key — so that improvements naturally bypass stale
    entries.

### Benchmarking

39. As an evaluator, I want end-to-end multi-hop QA as the **primary** benchmark
    on **2WikiMultihopQA**, ingesting its provided context paragraphs as the
    corpus, so that I measure the thing Graph RAG is *for* on a dataset whose
    `(entity, relation, entity)` evidence mirrors the knowledge graph.
40. As an evaluator, I want a fixed subset of ~100–200 questions, so that runs
    are fast and cheap.
41. As an evaluator, I want **supporting-fact precision/recall/F1**, so that I
    can score whether retrieval surfaced the gold evidence sentences.
42. As an evaluator, I want **answer Exact-Match + token-F1** scored by string
    comparison against the gold answer, matched against the node's `name` **and
    its `aliases`** under standard normalization (lowercase; strip
    articles/punctuation/extra whitespace), so that a differently-phrased-but-
    correct entity isn't unfairly marked wrong — and the scoring itself uses no
    LLM.
43. As an evaluator, I want a lightweight, secondary **extraction sanity check**
    (NER against a small standard/hand-labeled set; manual EL-merge spot-checks),
    so that I get a feel for extraction quality without a full study or a
    non-existent corpus-local EL gold set.
44. As an evaluator, I want reproducible benchmark runs via **fixed ingestion
    order and fixed EL thresholds**, so that the order-sensitive corpus-local
    graph (and therefore the scores) are deterministic and comparable across
    runs.
45. As a cost-conscious evaluator, I want cost controls — cache every LLM call,
    reuse the pre-built graph, use the cheaper extraction model, run the fixed
    small subset — so that I can re-run the benchmark repeatedly at ~$0.

### Operations, config, observability

46. As an operator, I want the full stack (Kafka, MinIO, one Elasticsearch
    cluster with two indices, Neo4j, and the pipeline/API service) brought up
    with a single Docker Compose, so that I can run everything locally.
47. As an operator, I want OpenAI/DeepSeek keys and service endpoints configured
    via `.env`/environment variables, so that secrets stay out of code and
    providers are swappable.
48. As an operator, I want basic Python `logging` behind a seam, so that the demo
    is simple now but structured/JSON logging can be swapped in later.
49. As a developer, I want the codebase to be modular Python 3.12 managed with
    `uv`, with docstrings and type hints, so that it follows the stated coding
    standards and stays maintainable.

## Implementation Decisions

Grounded in the accepted decision register (D1–D9) and ADRs 0001–0009. No file
paths or code beyond the small interface/shape sketches noted as such.

### Topology & orchestration (D1 / ADR-0001)

- **Single in-process pipeline consumer**, not microservices. One service
  consumes the trigger topic and runs all **stages** for each document.
- Each **stage** is a separate, swappable module behind a common stage
  interface (`read`, `ner`, `coref`, `entity_linking`, `kg_build`). Stages must
  not share hidden state — the modular boundary is load-bearing.
- Stage handoff is **in-memory** (Python objects). Elasticsearch/Neo4j writes
  occur at defined **checkpoints**: raw text → ES-Documents **at ingestion
  (before processing)**; that record enriched in place (NER + coref + per-doc EL)
  → ES-Documents at the entity-linking checkpoint; canonical entities →
  ES-Entities during entity linking; graph → Neo4j at KG-build. (ADR-0001.)
- **Error handling:** log-and-drop per document (no DLQ).
- **Idempotency:** deterministic **document ID** from `{bucket}/{objectKey}`;
  reprocessing overwrites.
- The **Kafka consumer loop is a thin driver** that resolves a trigger to a
  `process_document({bucket, objectKey})` call — the orchestrator is decoupled
  from Kafka so it can be driven directly in tests.

### External-dependency port boundary (primary architectural seam) — ADR-0010

Everything outside the pipeline's control sits behind a narrow interface
(Python `Protocol`), constructor-injected into the orchestrator and query
service. Real adapters wrap the live services; in-memory fakes back the fast
test suite. The set of ports:

- `ObjectStore` — read a document's bytes from MinIO (S3-compatible) given
  `{bucket, objectKey}`.
- `DocumentStore` — read/write the **ES-Documents** record (text + NER mentions
  + coref cluster map + per-document EL result + passage/sentence vectors).
- `EntityStore` — upsert canonical entities and run blocking + kNN search over
  **ES-Entities** (entity vectors live here, shared between EL and query).
- `GraphStore` — write triples/nodes/edges and run k-hop traversal in Neo4j.
- `LLMClient` — the provider-agnostic client (below).
- `Embedder` — the local sentence-transformer producing vectors.

This is the single seam the whole system is tested at; see Testing Decisions.

### NER (D2 / ADR-0002)

- Local spaCy, transformer pipeline `en_core_web_trf` (fall back to
  `en_core_web_lg` where CPU speed matters). No LLM.
- Curated types: PERSON, ORG, LOCATION (merge GPE+LOC), DATE, EVENT, NORP,
  optionally PRODUCT. Character **spans** retained for every mention. Sentence
  segmentation produced in the same pass for provenance.

### Coreference (D3 / ADR-0003)

- Within-document only, LLM-backed. Output is a **cluster map** (original text
  preserved; mention → chosen canonical in-document mention). Each document's
  clusters become the **doc-level entities** handed to EL.

### Entity linking (D4 / ADR-0004)

- Corpus-local resolution against **ES-Entities**, no external KB:
  1. **Block** by entity type + normalized name.
  2. **Score** with local sentence-transformer embedding similarity over the
     mention-in-context.
  3. **Merge** above threshold into the existing **canonical entity**; otherwise
     **create a new canonical entity** (always-on normal path).
  4. **Optional gated LLM tie-breaker** for borderline matches — off by default.
- **Optional gated NIL/unlinked-entity path** — implemented later, switchable,
  off by default.
- **Canonical entity ID = merge key = graph node identity.**
- Per-document EL result → ES-Documents; canonical entities → ES-Entities.
- Disambiguation is heuristic (type + context embeddings) and **order-sensitive**
  — the first document mentioning an entity seeds its canonical record.

### Storage split (D5 / ADR-0005)

- **Two indices in one Elasticsearch cluster.**
  - `ES-Documents`: one record per document — original text, NER mentions (with
    spans), coref cluster map, per-document EL result, passage/sentence
    `dense_vector`s.
  - `ES-Entities`: one record per deduplicated canonical entity, keyed by
    canonical entity ID, with an entity `dense_vector`; the substrate EL
    blocks/searches against and the query-side seeds on.

### KG builder & Neo4j model (D6 / ADR-0006)

- Schema-guided extraction: LLM (default `gpt-4o-mini`) receives document text +
  that document's canonical linked entities and emits triples
  `(subject_id, predicate, object_id)` over **canonical entity IDs**.
- **Closed predicate set + open fallback:** map to the closest of the ~12
  starter predicates; else `RELATED_TO` with the original phrase kept in
  `raw_predicate`.
- **Node model:** multi-label `:Entity` + type label; properties `canonical_id`,
  `name`, `type`, `aliases`. First-class node types: PERSON, ORG, LOCATION,
  EVENT, PRODUCT, NORP. **DATE is an edge attribute/qualifier, not a node.**
- **Edge provenance:** `source_doc_id`, `sentence_index`, `source_sentence`,
  `raw_predicate`, `confidence`. LLM cites the **sentence index** only; our
  spaCy segmentation resolves `char_start`/`char_end`.

  Triple shape (from the domain model — the decision-bearing part, not code):

  ```
  Triple {
    subject_id: canonical_entity_id
    predicate:  <one of closed set> | "RELATED_TO"
    object_id:  canonical_entity_id
    date_qualifier?: normalized date        # DATE as attribute, not node
    provenance: {
      source_doc_id, sentence_index, source_sentence,
      raw_predicate, confidence, char_start, char_end   # offsets resolved locally
    }
  }
  ```

### Query / retrieval (D7 / ADR-0007)

- **Two entry points, deliberately different.** Ingestion = FastAPI upload →
  MinIO → publish Kafka trigger (Kafka remains the first pipeline stage). Query =
  synchronous FastAPI `/query`, read-only, no Kafka.
- **Retrieval mode (default, no LLM, deterministic, free):** embed question →
  ES kNN over entity + passage/sentence vectors → expand k hops in Neo4j →
  return ranked subgraph + supporting sentences with provenance. Entity-typed
  answer = top-ranked candidate entity node.
- **Answer-synthesis mode (optional, gated LLM, off by default):** feed the
  retrieved subgraph + sentences to the LLM for prose.

### LLM abstraction, caching, structured output (D8 / ADR-0008)

- Provider-agnostic client via **LiteLLM**; per-stage model config. Defaults:
  `gpt-4o-mini` for coref + KG-build; fuller model reserved for synthesis. Any
  OpenAI-compatible endpoint (incl. DeepSeek) swappable via `.env`.
- **Persistent response cache** keyed by `sha256(model + prompt + params)`;
  invalidation is implicit.
- **Structured output** validated against Pydantic models with retry on parse
  failure.

### Embedding (assumed default, ADR-0004/0007)

- One local sentence-transformer (e.g. `bge-small-en-v1.5`) **reused** for both
  EL matching and query-time vector anchoring. Vector anchoring targets **both**
  canonical entities and passages/sentences.

### Benchmarking (D9 / ADR-0009)

- **Dual-track, end-to-end primary.** Dataset: **2WikiMultihopQA**, ingesting
  its provided context paragraphs as the corpus; fixed **~100–200 question**
  subset. Metrics (all non-LLM): supporting-fact P/R/F1; answer EM + token-F1
  vs. node `name` + `aliases` under standard normalization.
- **Secondary extraction sanity check:** NER vs. a small standard/hand-labeled
  set; manual EL-merge spot-checks.
- **Reproducibility:** fixed ingestion order + fixed EL thresholds.
- **Cost controls:** cache all LLM calls; reuse pre-built graph; cheaper
  extraction model; fixed small subset.

### Config, ops, standards (settled details)

- Full stack via one **Docker Compose**: Kafka, MinIO, one Elasticsearch cluster
  (two indices), Neo4j, pipeline/API service.
- Secrets/config via `.env` / environment variables.
- Basic Python `logging` behind a seam (structured logging swappable later).
- Python 3.12, dependencies via **`uv`**, modular layout, docstrings + type
  hints.

## Testing Decisions

**What makes a good test here:** it asserts on *external behavior at a seam* —
the document record, canonical entities, and graph produced by ingesting a
document; the ranked subgraph / supporting sentences / top-ranked answer
returned by a query; the benchmark scores for a fixed input — **not** on
internal call sequences or private structure. Tests must be deterministic and
cost **$0**: no live LLM calls, no reliance on wall-clock or external network in
the fast suite.

**The one primary seam — the external-dependency port boundary** (confirmed
with the product owner). All six ports (`ObjectStore`, `DocumentStore`,
`EntityStore`, `GraphStore`, `LLMClient`, `Embedder`) have in-memory fakes.
Because everything outside the pipeline's control crosses this boundary, both
entry points run fully in-process with no Docker in the fast suite:

- **Ingestion (entry point A):** drive `process_document({bucket, objectKey})`
  end-to-end against fakes; assert the written ES-Documents record (NER mentions
  + spans, coref cluster map, per-document EL result), the upserted ES-Entities
  canonical entities, and the Neo4j triples with provenance. The Kafka consumer
  loop and the FastAPI ingestion endpoint are thin and exercised *through* this
  seam, not directly.
- **Query (entry point B):** drive the retrieval function behind `/query`
  against a pre-seeded fake `EntityStore`/`DocumentStore`/`GraphStore`; assert
  the ranked subgraph, supporting sentences + provenance, and the top-ranked
  entity answer. The HTTP contract is checked via FastAPI `TestClient` at the
  same seam.

**LLM determinism:** the fake `LLMClient` returns canned structured responses;
the prompt-hash response cache (ADR-0008) doubles as a fixture so
coref/KG-build/synthesis are deterministic and free. NER is local and
deterministic already.

**Supplementary unit seams — only where internal logic is non-trivial:**

- **EL merge decision** (ADR-0004): blocking + similarity + threshold →
  merge-vs-create-new; plus the gated LLM tie-breaker and gated NIL/unlinked
  paths. Behavior asserted: same entity across two documents merges to one
  canonical ID; a genuinely new entity creates a new one; order-sensitivity is
  explicit in fixtures.
- **Provenance offset resolution** (ADR-0006): LLM cites a sentence index → our
  spaCy segmentation resolves `char_start`/`char_end`. Assert offsets map to the
  correct source sentence.
- **Answer normalization + EM/F1** (ADR-0009): standard normalization; matches
  against `name` + `aliases`; correct-but-differently-phrased entity scores as
  correct.
- **Document ID + idempotency** (ADR-0001): deterministic ID from
  `{bucket}/{objectKey}`; reprocessing overwrites rather than duplicates.

**Thin real-container integration layer** (confirmed: fakes + a thin real layer).
A deliberately *small* set of integration tests runs the real adapters against
real services via docker-compose/testcontainers — enough to prove the real
`EntityStore` kNN, `GraphStore` Cypher traversal, `DocumentStore`, and
`ObjectStore` behave like their fakes (a "contract test" per port). These are
slower and excluded from the fast pre-push loop; they gate the real adapters, not
pipeline logic.

**Benchmark determinism:** with fixed ingestion order, fixed EL thresholds, and
a warm response cache, a benchmark run over the fixed subset is reproducible and
can itself be smoke-tested (small fixture corpus → stable scores).

**Prior art:** none in-repo — this is greenfield. The seams are designed so that
the fakes-based suite is the prior art all future stage tests follow (highest
seam first; drop to a unit seam only for the non-trivial internal logic listed
above). Real-adapter contract tests follow the standard testcontainers pattern.

## Out of Scope

- **Frontend / visualization** — the Svelte+Vite UI for browsing ES-Documents /
  ES-Entities and interactive Neo4j graph viz is deferred to a later phase.
- **External knowledge base** — no Wikidata/Wikipedia linking; the corpus is its
  own authority (ADR-0004).
- **Cross-document coreference** — handled entirely by entity linking, not coref
  (ADR-0003).
- **Microservices / per-stage Kafka topics / independent scaling** — modular
  boundaries keep this path open but it's not built now (ADR-0001).
- **Dead-letter queue / retry orchestration** — log-and-drop for the demo.
- **Self-hosting LLM models** — external API only (vLLM path not built).
- **Authentication, multi-tenancy, public deployment.**
- **Non-English languages.**
- **Large-scale throughput (10⁴–10⁵ docs)** — the demo targets tens–hundreds of
  news-article-sized docs (~400–1500 words); the design stays flexible for
  future scale-up, but batching/throughput work is not in v1.
- **NIL/unlinked-entity retention** and the **LLM EL tie-breaker** ship gated
  off; enabling/tuning them is a later concern.
- **Exhaustive extraction benchmarking** — extraction is a lightweight sanity
  check only; no full NER/coref/EL study, no corpus-local EL gold set.
- **Full answer-synthesis evaluation as the primary metric** — synthesis mode is
  optional; the primary benchmark is the non-LLM retrieval path.

## Further Notes

- **Corpus sizing / cost estimate (demo):** news-article-sized documents,
  ~400–1500 words (~500–2000 tokens) each; ~100–300 documents for a substantial
  graph. Only coref + KG-build call the LLM per document (on `gpt-4o-mini`), so
  per-document extraction cost is small; the response cache makes re-ingestion
  and benchmark re-runs ~$0.
- **Honest limitation (by design):** the default non-LLM retrieval answers
  **entity-typed** questions well (top-ranked node) but not
  **descriptive/explanatory** ones — those need the optional gated synthesis
  mode. The benchmark design accounts for this (ADR-0007, ADR-0009).
- **Order-sensitivity is real:** corpus-local EL seeds a canonical record from
  the first mention, so benchmark comparability depends on fixed ingestion order
  + thresholds (ADR-0009).
- **A2 consistency items are resolved** and folded into the ADRs: persistence
  model (in-memory handoff + checkpoint writes), provenance (LLM cites sentence
  index; offsets resolved locally), node-vs-attribute (DATE is an edge
  attribute), answer scoring (EM/F1 vs. name + aliases), and benchmark
  reproducibility (fixed order + thresholds).
- **Q40 confirmed:** non-LLM retrieval is the default answer path; prose
  synthesis is the optional gated LLM mode. Q41–Q45 assumed defaults (embedding
  model, dual entity+passage anchoring, benchmark corpus = provided context
  paragraphs, LiteLLM, ~12-predicate starter set) are carried into this PRD; any
  can be revisited in specs without reopening an ADR.
- **Next process step:** C (shaping) → D (breadboarding) → E (extract ADRs &
  final consistency), per the build-plan-product flow.

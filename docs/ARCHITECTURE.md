# ARCHITECTURE — Graph RAG Demo

> Technical architecture reference. This doc **consolidates** the decisions
> already accepted in [`docs/adr/0001–0010`](./adr/), the [`PRD`](./PRD.md), and
> [`CONTEXT`](./CONTEXT.md) (glossary + decision register D1–D10). It invents
> nothing new. Terms in **bold** carry their CONTEXT-glossary meaning. Open
> tuning parameters are listed in [§9](#9-open-technical-parameters) and stay
> deferred to [`SPECS-BACKLOG`](./SPECS-BACKLOG.md) B1–B8 — no values are pinned
> here.

---

## 1. Overview

A locally-run **Graph RAG** pipeline, brought up entirely on one machine via a
single **Docker Compose**. Ingestion is **Kafka-triggered**: an operator uploads
a text/Markdown **document**, it lands in object storage, and a **trigger
message** (`{bucket, objectKey}`) drives a single in-process **pipeline** that
runs `read → NER → coreference → entity linking → knowledge-graph build`. **NER**
is local (spaCy), **coreference** is LLM-backed and within-document, **entity
linking** is corpus-local (no external KB) and unifies the same real-world entity
across documents into one **canonical entity**, and the **KG-build** stage writes
**triples** with edge-level **provenance** into a Neo4j **knowledge graph**.
Questions are answered through a synchronous read-only REST `/query` endpoint
whose default path is **non-LLM, deterministic, and free**: embed the question,
vector-anchor onto seed entities/passages, expand k hops in the graph, and return
a ranked subgraph plus supporting sentences (top-ranked entity node = predicted
answer). An optional **gated LLM answer-synthesis** mode turns that evidence into
prose. Capability is benchmarked on **2WikiMultihopQA** with non-LLM metrics.
(ADR-0001 through ADR-0010.)

## 2. System topology

Two deliberately different entry points into shared stores (ADR-0001, ADR-0007):

| Entry point | Driver | Nature | Path |
|-------------|--------|--------|------|
| **Ingestion** | **Kafka** | Asynchronous pipeline | `POST /ingest` → MinIO + publish trigger → **single in-process consumer** runs all 5 stages for the document |
| **Query** | **REST** | Synchronous, read-only, interactive (no Kafka) | `POST /query` → retrieval runs directly against the stores |

- The **pipeline** is a **single in-process consumer**, not microservices — one
  service consumes the trigger topic and runs every **stage** in-process, handing
  off in-memory. Chosen for local simplicity and debuggability; the modular stage
  boundaries keep a future microservices split open (ADR-0001). The consumer runs in a
  thread of the API process by default (`INGEST_MODE=inline`, job status in memory). With
  `INGEST_MODE=publish` (what both compose files set, GRA-31) the API only publishes and
  separate `graph-rag-worker` processes (the compose `worker` service, same image) consume,
  so ingest load does not starve queries. Job status then lives in Elasticsearch
  (`JOBS_INDEX`, `EsJobStore`) so the API reads, and streams, what the workers write. Workers
  share one consumer group; the trigger topic has `INGEST_PARTITIONS` partitions (default 4)
  and the publisher keys a trigger by its source, so one document stays in order on one worker
  while different documents spread. Scale with `docker compose up -d --scale worker=N`, useful
  up to the partition count. The files the API and workers both write (`USAGE_PATH`, `TENANT_LLM_KEYS_PATH`) sit on a `shared-state` volume mounted by both; the usage file is safe for several processes (a file lock around each read-modify-write). Documents and queries are counted in the API and LLM tokens in the worker, into the same file. A worker serves `GET /healthz` on `WORKER_HEALTH_PORT` (8001),
  200 while its consume loop runs, and on SIGTERM finishes the document in hand, commits and exits.
- The **Kafka consumer loop is a thin driver**: it resolves one trigger to a
  `process_document({bucket, objectKey})` call, so the orchestrator is decoupled
  from Kafka and can be driven directly in tests (ADR-0001, ADR-0010).
- The **query service** is a synchronous FastAPI endpoint that runs retrieval
  read-only; queries are interactive, not a pipeline, so they never touch Kafka
  (ADR-0007).
- One FastAPI service exposes both `/ingest` and `/query`.

## 3. Component / port architecture

Everything outside the pipeline's control sits behind a narrow Python `Protocol`
**port**, constructor-injected into the orchestrator and query service. Real
adapters wrap live services in production; in-memory **fakes** back the fast,
`$0`, no-Docker test suite. This port boundary is the **single primary seam** the
whole system is tested at, and the reuse seam that keeps shared collaborators
(`LLMClient`+cache, `Embedder`) single instances rather than per-stage copies
(ADR-0010).

| Port | Real adapter | Responsibility |
|------|--------------|----------------|
| `ObjectStore` | **MinIO** (S3-compatible) | Read/write a document's bytes given `{bucket, objectKey}` |
| `DocumentStore` | **Elasticsearch** `ES-Documents` | Read/write the per-document record: raw text, NER mentions+spans, coref cluster map, per-doc EL result, passage/sentence `dense_vector`s |
| `EntityStore` | **Elasticsearch** `ES-Entities` | Upsert canonical entities; run blocking + kNN search over entity `dense_vector`s (shared by EL and query) |
| `GraphStore` | **Neo4j** | Write multi-label nodes + provenance-carrying edges; run k-hop traversal |
| `LLMClient` | **LiteLLM** (+ response cache) | Provider-agnostic calls; per-stage model config; `sha256(model+prompt+params)` cache; Pydantic-validated structured output + retry |
| `Embedder` | **sentence-transformer** (`bge-small-en-v1.5`) | Local embeddings: mention-in-context, canonical entities, passages/sentences, query text |

The six ports map to breadboard affordances N10–N15. The `Embedder` (N15) is
shared by the ingestion path (EL) and the query path; `LLMClient`+cache (N14) is
shared across coref, KG-build, and synthesis.

## 4. Pipeline stages & data flow

The pipeline runs five swappable **stages** behind a common interface (stages
must not share hidden state — the modular boundary is load-bearing). Stage
handoff is **in-memory** (Python objects); Elasticsearch/Neo4j writes happen only
at defined **checkpoints**, not between every stage (ADR-0001).

```
read → NER → coreference → entity linking → knowledge-graph build
```

| Stage | Does | Persistence |
|-------|------|-------------|
| **read** | Fetch document bytes via `ObjectStore` | **Raw text → `ES-Documents` at ingestion, before processing** — the record is created up front, keyed by the deterministic document ID |
| **NER** | spaCy `en_core_web_trf`: typed mentions + char spans + sentence segmentation, one pass | in-memory |
| **coreference** | LLM within-doc cluster map → **doc-level entities** | in-memory |
| **entity linking** | Block by type+normalized name → score by embedding similarity → merge/create-new | **EL checkpoint:** the `ES-Documents` record is **enriched in place** with NER mentions + coref clusters + per-doc EL result (+ passage/sentence vectors); canonical entities upserted to `ES-Entities` |
| **KG-build** | LLM emits triples over canonical IDs; resolve offsets; attach edge provenance | Graph written to **Neo4j** |

Checkpoint / persistence model (ADR-0001):

- The **raw document text** is written to `ES-Documents` **at ingestion, before
  processing** (the record is created up front, keyed by the deterministic
  **document ID** from `{bucket}/{objectKey}`).
- That **same record is enriched in place at the entity-linking checkpoint** with
  NER mentions + coref cluster map + per-document EL result (and passage/sentence
  vectors) — computed in-memory through the stages and persisted together at this
  one checkpoint, not incrementally per stage.
- **Canonical entities** are upserted to `ES-Entities` during entity linking.
- The **graph** is written to Neo4j at KG-build.
- **Idempotency:** the deterministic document ID means reprocessing **overwrites**
  prior results. **Error handling:** log-and-drop per document (no DLQ); a failed
  document is discarded and can be re-triggered.

## 5. Data model

### (a) `ES-Documents` record (per document) — ADR-0005, ADR-0001

- Original **text**
- **NER mentions** with character spans (type + `char_start`/`char_end`)
- **Coreference cluster map** (mention → chosen canonical in-document mention;
  original text preserved)
- **Per-document EL result** (which doc-level entity resolved to which canonical
  entity ID)
- **Passage/sentence `dense_vector`s** (query-side seeding)

### (b) `ES-Entities` canonical entity record (per deduplicated entity) — ADR-0004, ADR-0005

- `canonical_id` (the **merge key** / graph node identity)
- `name`
- `type`
- `aliases`
- entity **`dense_vector`** (the substrate EL blocks/searches against and the
  query side seeds on)

### (c) Neo4j graph model — ADR-0002, ADR-0006

**Nodes** — multi-label: a shared `:Entity` label **plus** a type label, so all
entities are queryable via `:Entity` or one type via its label (per-label
indexes). Properties: `{canonical_id, name, type, aliases}`. One node per
**canonical entity ID** — this is what lets the same entity from different
documents become one node and enables cross-document multi-hop reasoning.

- First-class node types: `Person`, `Organization`, `Location`, `Event`,
  `Product`, **`Norp`** (`:Entity:Norp` — nationalities/religious/political
  groups; a first-class node per ADR-0006, participating in `AFFILIATED_WITH` /
  `MEMBER_OF` relations).
- **`DATE` is an edge qualifier, not a node** — modeled as a date qualifier /
  attribute on the relevant edge, to keep the graph readable for multi-hop paths.
- **Edges also carry `valid_from` / `valid_to`** (GRA-42): `YYYY-MM-DD`, from inclusive to exclusive, `None`
  when unknown. `valid_from` is the `document_date` the caller gave `/ingest` or `/v1/remember`. `valid_to` is set
  by supersession (GRA-43, `graph_rag/supersession.py`): a newer dated document's `WORKS_FOR`, `LOCATED_IN`,
  `HAS_ROLE` or `GOVERNED_BY` edge ends an older same-subject one with a different object, and the older edge
  records `superseded_by` so deleting or re-ingesting the newer document reopens it.
- **Duplicate entities merge when a later document links them** (GRA-30): linking is order-sensitive, so two documents
  can leave one real entity as two canonical entities. When one coref cluster in a later document has surface forms
  that already belong to two or more entities of the same type, the linker folds them into the lowest `canonical_id`
  (`EntityLinkingStage._unify`): names and aliases are unioned in the entity store, `GraphStore.merge_nodes` moves
  the dropped nodes' edges onto the survivor (provenance and validity untouched; an edge that would become a loop,
  or one the survivor already has, is dropped; `graph_rag/merging.py`), and other documents' `el_result` links are
  renamed. A cluster's other names are also kept as aliases when it creates or merges one entity, so the result does
  not depend on which document arrives first: the same entities and edges come out, though the surviving id can
  differ. A merge is not undone by deleting the document that caused it, and there is no split yet (GRA-51).
- **A changed source replaces its old contribution** (GRA-27): ingesting the same source with different bytes swaps
  the document's record, sentences, mentions and edges for the new text's, and removes entities only the old text
  linked to (from the graph and the entity store) unless another document still does. Identical bytes are skipped
  by `/ingest` before any of this runs.

**Edges** — carry a predicate from the closed **~12-predicate set** with a
`RELATED_TO` open fallback:

`LOCATED_IN`, `PART_OF`, `MEMBER_OF`, `WORKS_FOR`, `HAS_ROLE`, `FOUNDED`, `OWNS`,
`PRODUCES`, `PARTICIPATED_IN`, `OCCURRED_ON`, `AFFILIATED_WITH`, `GOVERNED_BY`, `RELATED_TO`
(fallback). When nothing fits, the edge is `RELATED_TO` and the model's original
phrase is preserved in the `raw_predicate` property — nothing is lost.

**Edge provenance** (load-bearing for traceable answers): `{source_doc_id,
sentence_index, source_sentence, raw_predicate, confidence, char_start,
char_end}`. The LLM cites only the **sentence index** per triple; `char_start`/
`char_end` are resolved by **our own spaCy sentence segmentation** (ADR-0002),
not by the LLM.

Triples reference **canonical entity IDs** (not raw strings), grounding the graph
in the EL store.

## 6. Retrieval architecture

Two modes; default to the non-LLM one (ADR-0007).

**Retrieval mode (default — no LLM, deterministic, `$0`):**

1. **Embed the question** with the local sentence-transformer (same model reused
   from EL).
2. **Vector-anchor** seed nodes via Elasticsearch kNN over **entity embeddings**
   (`ES-Entities`) and **passage/sentence embeddings** (`ES-Documents`).
3. **Expand k hops** from the seed entities in Neo4j to gather the connected
   subgraph.
4. Return a **ranked subgraph + supporting sentences with provenance**. For
   entity-typed questions, the **predicted answer is the top-ranked candidate
   entity node**.

**Answer-synthesis mode (optional, gated LLM, off by default):** feed the
retrieved subgraph + supporting sentences to the LLM (`/query` with
`synthesize=true`) to produce prose. Kept separate so the core path stays free
and deterministic.

Vectors live in Elasticsearch `dense_vector` fields (entity vectors in
`ES-Entities`, shared with EL; passage/sentence vectors in `ES-Documents`) — no
separate vector database.

**Honest limitation (by design):** non-LLM retrieval answers entity-typed
questions well (top node); descriptive/explanatory answers need the optional
synthesis mode (ADR-0007, ADR-0009).

## 7. LLM & embedding infrastructure

**LLM** (ADR-0008): a **provider-agnostic client via LiteLLM** — the
provider/model is a **config choice per stage**, and any OpenAI-compatible
endpoint (incl. DeepSeek) is swappable via `.env`.

- **Per-stage model config:** `gpt-4o-mini` default for the high-volume
  extraction stages (coref, KG-build); a fuller model reserved for the optional
  answer-synthesis mode.
- **Response cache:** persistent, keyed by `sha256(model + prompt + params)`.
  Repeated pipeline runs and benchmark re-runs hit the cache and cost nothing;
  cache invalidation is implicit (changing prompt/model changes the key).
- **Structured output:** JSON validated against **Pydantic** models (coref
  clusters, links, triples) with retry on parse failure.
- Only coref, KG-build, the optional EL tie-breaker, and optional synthesis call
  the LLM; NER and core EL are local (`$0`).

**Embedding** (ADR-0004, ADR-0007): **one local sentence-transformer**
(`bge-small-en-v1.5`, subject to B1 confirmation) **reused** for both EL matching
(mention-in-context, canonical entities) and query-time vector anchoring
(question, passages/sentences).

## 8. Deployment / tech stack

- **One Docker Compose** brings up: **Kafka**, **MinIO**, **one Elasticsearch
  cluster** (holding the two indices `ES-Documents` + `ES-Entities`), **Neo4j**,
  and the **pipeline/API service** — a single command to run everything locally
  (ADR-0005, PRD).
- **Python 3.12**, dependencies managed with **`uv`**, modular layout, docstrings
  + type hints.
- **Config/secrets** via `.env` / environment variables (OpenAI/DeepSeek key,
  service endpoints, per-stage models, EL thresholds, k-hop depth) — secrets stay
  out of code, providers are swappable.
- **Logging** via basic Python `logging` behind a seam; `LOG_FORMAT=json` writes one object per line
  with the trace id (`graph_rag/logging.py`). Content stays out of logs (`docs/LOG-POLICY.md`).
- **Metrics and traces** (GRA-56, `graph_rag/observability.py`), both off by default. `METRICS_ENABLED`
  serves Prometheus text at `/metrics` on the API and on the worker's health port: request count and
  latency by route *template* and status, ingest stage duration and failures, documents by outcome, LLM
  calls, duration and tokens by model. Tenant is not a label (unbounded, and usage already counts it);
  with `REQUIRE_AUTH` on, `/metrics` needs `METRICS_TOKEN`, never a tenant key. Traces use the
  OpenTelemetry API (a no-op until `OTEL_EXPORTER_OTLP_ENDPOINT` is set and the `otel` extra is installed,
  which the image has): the API puts a W3C `traceparent` on the ingest trigger and the worker continues
  it, so one trace runs API accept, then `ingest.<stage>` (read, entity recognition, coreference, entity
  linking, knowledge graph, which is the graph write), with an `llm.call` span under the stage that made it.
  A Grafana dashboard, Prometheus and Jaeger come from `docker-compose.observability.yml`.
- **Testing seam** (ADR-0010): fast suite runs both entry points fully in-process
  against fakes ($0, no Docker); a thin real-container contract layer
  (docker-compose / testcontainers) proves each real adapter behaves like its
  fake.

## 9. Open technical parameters

These are **not pinned here.** They are deferred tuning knobs / carried
assumptions tracked in [`SPECS-BACKLOG`](./SPECS-BACKLOG.md) B1–B8, each to be
pinned **during implementation** of the slice it belongs to — not by treating the
current default as settled.

| # | Parameter | Pin during |
|---|-----------|------------|
| B1 | Embedding model + dimension (confirm `bge-small-en-v1.5`; sets the `dense_vector` mapping) | V4 |
| B2 | EL similarity threshold (merge-vs-create-new; fixed for benchmark reproducibility) | V4 |
| B3 | k-hop expansion depth (recall vs. subgraph noise) | V6 |
| B4 | Subgraph ranking function (orders candidate nodes/paths; picks the top-entity answer) | V6 |
| B5 | kNN seeding params (top-k per index; how entity-seeds vs. passage-seeds combine) | V6 |
| B6 | Per-stage LLM model ids + params (feeds the `sha256` cache key) | V3 / V5 / V7 |
| B7 | Closed predicate set finalization (extend beyond the ~12 starter set) | V5 |
| B8 | Benchmark corpus + subset (exact 2WikiMultihopQA split/subset + fixed ingestion order) | V8 |

**Genuine architecture gaps not covered by the ADRs:** none identified. All
structural decisions (topology, ports, stages, checkpoints, data models,
retrieval, LLM/embedding infra, deployment) are settled in ADR-0001–0010; every
remaining open item above is a tuning value or a carried assumption, not a
missing architectural decision.

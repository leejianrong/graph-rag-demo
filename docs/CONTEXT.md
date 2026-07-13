# CONTEXT — Graph RAG Demo

Shared language and decision register for the project. This is the distilled
source of truth produced during the grilling step: the **glossary** fixes what
each term means, and the **decision register** records what we've settled (with
links to the ADRs that hold the full rationale).

- Raw idea: [`REQS.md`](./REQS.md)
- Raw answer log: [`ANSWERS.md`](./ANSWERS.md)
- Open questions: [`QUESTIONS.md`](./QUESTIONS.md)
- Decision records: [`docs/adr/`](./adr/)

---

## Glossary

**Document** — one ingested text file (Markdown or plain text), the unit of
processing. Identified by a deterministic **document ID** derived from its
`{bucket}/{objectKey}`.

**Trigger message** — the Kafka message that kicks off ingestion for one
document. Carries only `{bucket, objectKey}`; all other state is read from /
written to Elasticsearch.

**Pipeline** — the ordered processing of a single document:
`read → NER → coreference → entity linking → knowledge-graph build`. Runs as a
**single in-process consumer**, with each stage a separate, swappable module.

**Stage** — one step of the pipeline, implemented behind a common interface so
stages can be tested in isolation and later promoted to separate services.

**NER (Named Entity Recognition)** — local extraction of typed entity
**mentions** from text (spaCy). Produces surface strings, types, and character
spans. No LLM, no API cost.

**Mention** — a specific span of text referring to an entity (e.g. the words
"Elon Musk" at chars 0–9). Many mentions may refer to one entity.

**Coreference resolution** — grouping mentions **within a single document** that
refer to the same thing (including pronouns), producing a **cluster map**
(mention → canonical in-document mention). LLM-backed.

**Doc-level entity** — an entity as understood *within one document*: one coref
cluster, with a chosen representative name and type.

**Entity linking (EL) / entity resolution** — matching each doc-level entity to
the **corpus-local entity store**: either merge into an existing **canonical
entity** or **create a new one**. Local-first (blocking + embedding similarity);
optional gated LLM tie-breaker for borderline matches.

**Canonical entity** — the deduplicated, corpus-wide record of a real-world
entity, living in the `ES-Entities` index. Its ID is the **merge key** that
lets the same entity from different documents become one graph node.

**Corpus-local entity store** — the set of canonical entities discovered so far
across the ingested corpus. There is **no external knowledge base** (no
Wikidata/Wikipedia); the corpus is its own authority.

**ES-Documents** — Elasticsearch index holding each original document plus its
per-document processing results (entities, coref clusters, EL results).

**ES-Entities** — Elasticsearch index holding the deduplicated canonical
entities (the corpus-local store) used for entity resolution.

**Knowledge graph** — the Neo4j graph of **nodes** (canonical entities) and
**edges** (relationships). Built from **triples** (subject–predicate–object).

**Triple** — a subject–predicate–object statement extracted from a document,
grounded in canonical entity IDs and carrying **provenance**.

**Provenance** — the origin of a triple/edge: source document ID and sentence /
character range, so answers can be traced back to their source.

**Graph RAG** — retrieval that traverses the knowledge graph (not just matching
text chunks) so an LLM can answer multi-hop, cross-document questions.

**LLM** — external API model (OpenAI). Reserved for coref, KG-building,
query-time answering, and the optional EL tie-breaker. NER and core EL are
local. *(Model selection — see decision register — leans mini for extraction,
full model for answering.)*

---

## Decision register

Status values: **Accepted** · **Proposed** · **Open** (still being grilled).

| ID | Decision | Status | ADR |
|----|----------|--------|-----|
| D1 | Single in-process, modular pipeline consumer (not microservices) | Accepted | [ADR-0001](./adr/0001-single-inprocess-modular-pipeline.md) |
| D2 | Local NER via spaCy (`en_core_web_trf`); curated entity-type set; keep spans | Accepted | [ADR-0002](./adr/0002-local-ner-with-spacy.md) |
| D3 | Within-document coreference producing a cluster map; cross-doc identity deferred to EL | Accepted | [ADR-0003](./adr/0003-within-document-coreference.md) |
| D4 | Corpus-local entity linking (no external KB); local-first, optional gated LLM tie-break; create-new always on | Accepted | [ADR-0004](./adr/0004-corpus-local-entity-linking.md) |
| D5 | Elasticsearch split into `ES-Documents` + `ES-Entities` (two indices, one cluster) | Accepted | [ADR-0005](./adr/0005-elasticsearch-index-split.md) |
| D6 | KG builder: schema-guided triples, closed predicate set + open fallback; node identity = canonical entity ID; multi-label Neo4j nodes; edge-level provenance | Accepted | [ADR-0006](./adr/0006-knowledge-graph-builder-and-model.md) |
| D7 | Graph RAG query: non-LLM vector-anchored + graph-expansion retrieval (default), optional gated LLM synthesis; ES vectors + Neo4j; Kafka=ingestion, REST=query | Accepted | [ADR-0007](./adr/0007-graph-rag-query-retrieval.md) |
| D8 | Provider-agnostic LLM client (per-stage model config), prompt-hash response cache, structured output + Pydantic validation | Accepted | [ADR-0008](./adr/0008-llm-abstraction-caching-structured-output.md) |
| D9 | Benchmarking: dual-track (end-to-end primary + extraction sanity check); 2WikiMultihopQA subset; non-LLM metrics (supporting-fact P/R/F1, answer EM/F1) | Accepted | [ADR-0009](./adr/0009-benchmarking-strategy.md) |
| D10 | External-dependency port boundary (6 `Protocol` ports) as the single DI/test seam; fakes-first $0 fast suite + thin real-container contract layer | Accepted | [ADR-0010](./adr/0010-external-dependency-port-boundary.md) |

### Settled details (not warranting their own ADR yet)

- **Trigger payload:** Kafka message carries only `{bucket, objectKey}`.
- **Error handling:** log-and-drop per document for the local demo (no DLQ).
- **Idempotency:** deterministic document ID from `{bucket}/{objectKey}`;
  reprocessing **overwrites**.
- **Chunking:** whole-document processing; internal sentence segmentation for
  provenance; chunk only if a document exceeds the model context window.
- **Corpus sizing (demo):** news-article-sized docs, ~400–1500 words
  (~500–2000 tokens); ~100–300 docs for a substantial graph. Designed to scale
  to 10⁴–10⁵ docs later (out of scope now).
- **Language:** English only.
- **Coref output:** cluster map (original text preserved), not inline rewrite.
- **EL confidence gate:** below-threshold mentions **create a new entity**
  (always-on normal path); optional LLM tie-breaker is gated off by default.
- **LLM cost lean:** `gpt-4o-mini` for high-volume extraction (coref,
  KG-build); fuller model reserved for the optional answer-synthesis mode.
  Provider swappable (incl. DeepSeek) — see D8.
- **Entry points:** one FastAPI service exposes **ingestion** (upload → MinIO →
  publish Kafka trigger) and **query** (`/query`, synchronous, read-only, no
  Kafka). Kafka remains the first stage of the *ingestion* pipeline.
- **Provenance:** every graph edge records source doc + sentence/char range +
  source sentence text.
- **Logging:** basic Python `logging` for the demo, behind a seam so structured
  / JSON logging can be swapped in later.
- **Config/secrets:** `.env` / environment variables (OpenAI/DeepSeek key,
  service endpoints) read by the service.
- **Frontend:** deferred — out of scope for v1 (browse ES + interactive Neo4j
  graph viz come in a later phase).

### Assumed defaults (correct me if wrong)

These I've taken as sensible defaults to keep moving; flag any to revisit:

- **Embedding model:** a local sentence-transformer (e.g. `bge-small-en-v1.5`),
  **reused** for both EL matching and query-time vector anchoring.
- **Vector anchoring target:** embed **both** canonical entities and
  passages/sentences, so queries can seed on entities *and* evidence text.
- **Benchmark corpus:** ingest the QA dataset's **provided context paragraphs**
  as the document corpus (standard setup).
- **LLM abstraction:** use **LiteLLM** as the provider-agnostic client.
- **Closed predicate set (starter):** `LOCATED_IN, PART_OF, MEMBER_OF,
  WORKS_FOR, HAS_ROLE, FOUNDED, OWNS, PRODUCES, PARTICIPATED_IN, OCCURRED_ON,
  AFFILIATED_WITH, RELATED_TO` (fallback). Extensible.

---

## Step A status — COMPLETE

Grilling questions A–K answered; A2 inconsistency check done. The five A2 items
are resolved and folded into the ADRs:

1. Persistence model — in-memory stage handoff + ES writes at checkpoints
   (ADR-0001).
2. Provenance — LLM cites sentence index only; offsets resolved by our spaCy
   segmentation (ADR-0006).
3. Node vs. attribute — `PERSON/ORG/LOCATION/EVENT/PRODUCT/NORP` are nodes
   (`NORP` added in Step E, Q47); `DATE` is an edge attribute (ADR-0002,
   ADR-0006).
4. Answer scoring — EM/F1 against node `name` + `aliases` with standard
   normalization (ADR-0009).
5. Benchmark reproducibility — fixed ingestion order + EL thresholds (ADR-0009).

## Process status

- **Step A (grilling + domain modeling): COMPLETE** — see above (D1–D9, ADRs).
- **Step B (PRD): COMPLETE** — [`PRD.md`](./PRD.md).
- **Step C (shaping): COMPLETE** — [`FRAME.md`](./FRAME.md),
  [`SHAPING.md`](./SHAPING.md), [`BREADBOARD.md`](./BREADBOARD.md),
  [`SLICES.md`](./SLICES.md). No spike needed (Shape A had no flagged unknowns).
- **Step E (extract ADRs & consistency): COMPLETE** — Q46–Q48 resolved:
  Q46 raw text persisted at ingest + enriched at EL checkpoint (ADR-0001
  reworded); Q47 `NORP` is a first-class node (ADR-0002/0006 updated);
  Q48 `ADR-0010`/`D10` created for the external-dependency port boundary +
  testing seam. All shaping docs realigned.
- Step D (breadboarding) is redundant here — `/shaping` already produced the
  breadboard + slices (see WORKFLOW-PAINPOINTS #17).

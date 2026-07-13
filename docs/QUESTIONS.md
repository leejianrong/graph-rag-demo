# QUESTIONS — Graph RAG Demo (grilling log)

Working log for the grilling step of `build-plan-product`. Questions are dumped
up front and added as we go, so you can answer several per turn.

## Legend

- **Priority** — `P0` blocks the architecture (must resolve before PRD) ·
  `P1` important, shapes a stage · `P2` nice to pin down, can default.
- **Status** — `OPEN` · `ANSWERED` · `DEFERRED` · `ASSUMED` (I picked a
  sensible default; correct me if wrong).

Once answered, decisions graduate into `CONTEXT.md` (glossary + decision
register) and, where architectural, into `docs/adr/*.md`.

---

## A. Pipeline orchestration & topology (P0)

- **Q1 · P0 · OPEN** — Is each stage (read → NER → coref → EL → KG-build) a
  **separate service/consumer** wired together by intermediate Kafka topics, or
  **one consumer** that runs all stages in-process for a document? (Separate =
  more moving parts but independently scalable/retryable; monolith = simplest
  for a local demo.)
- **Q2 · P0 · OPEN** — If separate services: how do stages hand off — one Kafka
  topic **per stage boundary** (e.g. `raw`, `ner-done`, `coref-done`…), or a
  single topic where messages carry a "stage" field?
- **Q3 · P1 · OPEN** — What travels **on the Kafka message** vs. what gets
  fetched from storage? You said the trigger carries `{bucket, objectKey}`. For
  downstream stages, do we pass a **document ID** and have each stage re-read
  state from Elasticsearch, or pass the growing payload along in the message?
- **Q4 · P1 · OPEN** — **Error handling**: if a stage fails (bad LLM response,
  timeout), do we want retries + a dead-letter topic, or is "log and drop" fine
  for a local demo?
- **Q5 · P2 · OPEN** — **Idempotency**: if the same `{bucket, objectKey}` is
  published twice, should we reprocess (overwrite) or skip if already ingested?

## B. Document handling & chunking (P0)

- **Q6 · P0 · OPEN** — Are documents processed **whole**, or **chunked** first
  (by paragraph / token window)? This matters a lot: LLM context limits, coref
  quality (needs enough surrounding text), and how triples get provenance.
- **Q7 · P1 · OPEN** — Roughly how **large** is a typical document, and what
  **scale** overall — tens, hundreds, thousands, or millions of docs? (Sets
  whether we care about batching/throughput or just correctness for the demo.)
- **Q8 · P2 · OPEN** — **Language**: English only, or must we handle other
  languages? (Affects NER/coref model choice.)

## C. Named Entity Recognition (P0)

- **Q9 · P0 · OPEN** — What performs NER? You listed the LLM as used by coref /
  EL / KG-build but **not** NER. Is NER done **locally** (e.g. spaCy /
  GLiNER — no API cost), or also via the LLM? (Local NER is the standard,
  cheaper choice and keeps the "burn no credits" goal.)
- **Q10 · P1 · OPEN** — Entity type set: the standard `PERSON, ORG, LOC, DATE,
  QUANTITY` (+ maybe `EVENT`, `MISC`), or a fixed custom set you care about?
- **Q11 · P2 · OPEN** — Do we keep **character offsets / spans** for each entity
  mention (needed for good coref + provenance), or just the surface strings?

## D. Coreference resolution (P1)

- **Q12 · P0 · OPEN** — Coref scope: **within a single document** only, or also
  **cross-document** (same person mentioned in two files)? Cross-doc coreference
  is much harder and usually handled at the entity-linking / graph-merge step
  instead. Recommend within-doc for coref; cross-doc handled by EL.
- **Q13 · P1 · OPEN** — Output form: rewrite the text with pronouns **resolved
  inline**, or keep the original text + a **cluster map** (mention → canonical
  mention)? The cluster map is more useful downstream.

## E. Entity Linking (P0 — biggest open area)

- **Q14 · P0 · OPEN** — **What is the "Entities" knowledge base we link
  against?** Two very different designs:
  - (a) **External public KB** — link to Wikidata/Wikipedia (your examples use
    Wikipedia URLs). Needs a local Wikidata/Wikipedia dump or live API.
  - (b) **Corpus-local entity store** — the ES "Entities" index is the set of
    entities *we've discovered so far in the ingested corpus*, and EL means
    "match this mention to an existing corpus entity, or create a new one."
  - (c) **Both** — link to Wikidata when possible for a canonical ID, and also
    maintain the corpus-local store keyed by that ID.
  Which is it? (This decision drives the whole EL + graph-merge design.)
- **Q15 · P0 · OPEN** — You asked where the EL output JSON of linked clusters
  should be saved: **ES-Entities, ES-Documents, or both?** (My lean: the
  per-document EL result belongs with the document record in ES-Documents; the
  deduped canonical entities live in ES-Entities. Confirm.)
- **Q16 · P1 · OPEN** — If linking to Wikidata: do you want a **local dump**
  (heavy, but no per-call cost and fully offline) or **live API** calls
  (simple, rate-limited, needs internet)? For a local demo, live Wikidata API
  for candidate lookup + LLM to disambiguate is cheap and simple.
- **Q17 · P1 · OPEN** — When a mention can't be confidently linked, do we
  **create a new "unlinked" entity** (NIL) keyed within the corpus, or drop it?

## F. Knowledge-graph builder (P0)

- **Q18 · P0 · OPEN** — Relation/predicate vocabulary: **open** (LLM emits
  free-text predicates like "located in", "works for"), or a **closed schema**
  of allowed relation types? Open is easier to start; closed gives cleaner
  multi-hop queries and benchmarking.
- **Q19 · P0 · OPEN** — **Node identity / dedup** — this is what lets multi-hop
  reasoning span disjoint documents. Do we merge nodes across documents by the
  **entity-linking canonical ID** (e.g. Wikidata QID)? If EL is corpus-local,
  merge by the corpus entity ID? Confirm the merge key.
- **Q20 · P1 · OPEN** — What does the KG-builder **read from ES** — Documents,
  Entities, or both? (You flagged this. My lean: it reads the document text +
  that document's linked-entity clusters, so it can ground triples in real
  entity IDs. Confirm.)
- **Q21 · P1 · OPEN** — **Provenance**: should each triple/edge record which
  document (and ideally sentence/offset) it came from? Strongly recommended for
  a "connect disjoint documents" use case and for trustworthy answers.
- **Q22 · P2 · OPEN** — Neo4j node model: one generic `:Entity` label with a
  `type` property, or distinct labels per type (`:Person`, `:Organization`…)?

## G. Graph RAG query / retrieval side (P0 — scope question)

- **Q23 · P0 · OPEN** — **Is the query side in scope for the build**, or does
  the pipeline stop at "knowledge graph populated"? Your "What it should do"
  describes Graph RAG retrieval conceptually, but the last concrete stage listed
  is the KG builder. Benchmarking implies you need a **question → answer** path.
  Confirm we're building the retrieval + LLM-answer step too.
- **Q24 · P1 · OPEN** — If in scope, retrieval strategy:
  - (a) **LLM-generated Cypher** — turn the question into a graph query, run it,
    feed results to the LLM to synthesize an answer.
  - (b) **Vector-anchored + graph expansion** — embed the question, find seed
    nodes/chunks (via ES vector search), then traverse neighbours in Neo4j.
  - (c) **Community-summary / GraphRAG-style** — precompute community summaries
    (à la Microsoft GraphRAG) for global questions.
  Which flavour(s)? (a) or (b) are the usual demo choices.
- **Q25 · P1 · OPEN** — Do you want **embeddings/vector search** at all (over
  chunks and/or entities), using Elasticsearch's vector capabilities — i.e. a
  **hybrid** graph+vector RAG — or pure graph traversal?
- **Q26 · P2 · OPEN** — Query interface: a REST API endpoint, a CLI, and/or the
  frontend? Minimum for benchmarking is a programmatic function/endpoint.

## H. LLM usage & cost control (P1)

- **Q27 · P1 · OPEN** — Confirm **GPT-4o via OpenAI API** as the single LLM for
  coref, EL, and KG-build. Any interest in a cheaper model (e.g. GPT-4o-mini)
  for the high-volume stages to save credits?
- **Q28 · P1 · OPEN** — **Response caching**: cache LLM responses keyed by
  (prompt hash) so re-running the pipeline / benchmarks on the same data costs
  nothing? (Recommended — directly serves your "don't burn credits" goal.)
- **Q29 · P2 · OPEN** — Structured output: rely on the model's **JSON / function
  calling** mode for coref clusters, links, and triples? (Recommended for
  reliable parsing.)

## I. Benchmarking (P1)

- **Q30 · P0 · OPEN** — What are you benchmarking — **retrieval/answer quality**
  (multi-hop QA accuracy) and/or **pipeline extraction quality** (NER/EL
  correctness)? What does "capability" mean to you here?
- **Q31 · P1 · OPEN** — Which **dataset**? Standard multi-hop QA sets fit Graph
  RAG well: **HotpotQA**, **2WikiMultihopQA**, **MuSiQue**, or the Microsoft
  **GraphRAG** eval style. Or your own hand-built Q&A over a small corpus?
- **Q32 · P1 · OPEN** — Evaluation method: **exact-match / F1** against gold
  answers, or **LLM-as-judge**? (LLM-judge is flexible but costs credits — tie
  to the caching decision.)
- **Q33 · P1 · OPEN** — **Cost-control strategy for benchmarking** (your
  nice-to-have): my recommended combo is (1) cache all LLM calls, (2) run on a
  **small fixed subset** (e.g. 50–100 questions), (3) use a cheaper model for
  extraction, (4) reuse a pre-built graph so you only pay for query-time calls.
  Agree / adjust?

## J. Frontend & visualization (P2)

- **Q34 · P2 · OPEN** — Is the Svelte+Vite frontend **in scope for v1**, or a
  later add-on? (You marked it nice-to-have.)
- **Q35 · P2 · OPEN** — If yes: which views — browse ES Documents, browse ES
  Entities, and an interactive **Neo4j graph** visualization? Any preference for
  the graph viz library, or leave to implementation?

## K. Non-functional / cross-cutting (P1)

- **Q36 · P1 · OPEN** — **Two Elasticsearch instances or two indices in one
  cluster?** For a local demo, **two indices in a single ES container** is far
  simpler and lighter. Any reason to isolate them into separate instances?
- **Q37 · P2 · OPEN** — Config & secrets: OpenAI key + service endpoints via
  **`.env` / environment variables** read by each service? (Standard.)
- **Q38 · P2 · OPEN** — Observability: any need for structured logging / a way to
  trace one document through all stages, or is basic logging enough for the demo?
- **Q39 · P2 · OPEN** — How is data **first loaded** into MinIO + Kafka to kick
  things off — a small "seed/loader" script that uploads files and publishes the
  Kafka messages? (Needed to actually run the demo end-to-end.)

---

## Questions added during grilling (F–K follow-ups)

- **Q40 · P0 · ANSWERED (confirm)** — "Answer without an LLM": *retrieval* is
  fully non-LLM (embed → ES kNN → Neo4j expansion); *prose synthesis* needs the
  optional gated LLM. Default answer for entity-typed questions = top-ranked
  node. → Needs your explicit OK (see CONTEXT.md open items).
- **Q41 · P1 · ASSUMED** — Embedding model: local `bge-small-en-v1.5`, reused
  for EL + query anchoring.
- **Q42 · P1 · ASSUMED** — Vector anchoring over **both** entities and
  passages/sentences.
- **Q43 · P1 · ASSUMED** — Benchmark corpus = the QA dataset's provided context
  paragraphs, ingested as documents.
- **Q44 · P2 · ASSUMED** — LLM abstraction via **LiteLLM**.
- **Q45 · P2 · ASSUMED** — Starter closed predicate set of ~12 relation types
  (see ADR-0006 / CONTEXT.md).

## Answered / decisions

All questions A–K answered. Decisions distilled into
[`CONTEXT.md`](./CONTEXT.md) (register D1–D9) and
[`docs/adr/0001–0009`](./adr/). One confirmation outstanding (Q40).

---

## Step E — post-shaping consistency review

_Raised by the agent running Step E (grill-with-docs, recovered method — the
skill is a broken one-liner, see WORKFLOW-PAINPOINTS #19). Cross-checked
FRAME/SHAPING/BREADBOARD/SLICES against REQS/PRD/CONTEXT/ADRs. The shaping
outputs are consistent with the ADRs on all major points; the items below were
the exceptions + one ADR to extract. **All three resolved** (Q46/Q47/Q48
answered) and rippled through the ADRs, CONTEXT register, PRD, and shaping docs._

- **Q46 · P1 · ANSWERED → (a)** — **When is the raw document persisted to ES-Documents?**
  *Decision: raw text written at ingestion (before processing); NER/coref/EL
  enrichment held in-memory and persisted into the same record at the EL
  checkpoint. ADR-0001 reworded; SLICES V1–V4 + BREADBOARD N5 realigned.*
  `REQS.md` says the original text is "saved to Elasticsearch (Documents)
  **before any processing**." But `ADR-0001` says the document record
  (text + NER + coref + per-doc EL) is written to ES-Documents in a **single
  checkpoint after entity linking** — implying it is *not* separately persisted
  pre-processing. These two readings conflict.
  - **(a)** write raw text at ingestion **and** enrich/append at the EL
    checkpoint (two writes — honors REQS's "before processing" + keeps the
    enrichment checkpoint). *Recommended.*
  - **(b)** single write after EL (strict ADR-0001); raw text lives only in
    MinIO until then.
  **Knock-on to `SLICES.md`:** slices V1–V3 currently read as if ES-Documents is
  written incrementally per stage. Under (b), V1's demo is "bytes in MinIO +
  trigger consumed" and V2/V3 observe stage output in-memory/logs, not via ES
  writes. Whichever we pick, ADR-0001's wording and SLICES V1–V3 need a one-line
  align. (Flagged inline in `SLICES.md`.)

- **Q47 · P2 · ANSWERED → (a)** — **What is NORP's fate in the graph?**
  *Decision: NORP is a first-class node (`:Entity:Norp`). ADR-0002 & ADR-0006
  node lists updated; CONTEXT A2-item-3 + PRD stories 25 / node model updated.* `ADR-0002` extracts
  `NORP` as an NER type, but `ADR-0006`'s first-class node types are
  `PERSON, ORG, LOCATION, EVENT, PRODUCT` (with `DATE` as an edge qualifier).
  `NORP` is named in neither list — so it is extracted but its graph treatment is
  unspecified.
  - **(a)** promote `NORP` to a first-class node type;
  - **(b)** model it as an edge attribute/qualifier (like `DATE`);
  - **(c)** extract-only — used for mention context but not materialized in the
    graph.
  Affects `ADR-0006` node model + `SHAPING.md` A7 + `BREADBOARD.md` N9.

- **Q48 · P1 · ANSWERED → yes** — **Extract `ADR-0010` for the external-dependency port
  boundary + fakes-first testing seam?**
  *Decision: created [`ADR-0010`](./adr/0010-external-dependency-port-boundary.md)
  and added `D10` to the CONTEXT register; PRD port-boundary heading references it.* This is a load-bearing architectural
  decision (six `Protocol` ports — `ObjectStore`, `DocumentStore`, `EntityStore`,
  `GraphStore`, `LLMClient`, `Embedder` — as the single DI/test seam, with
  in-memory fakes for the fast $0 suite + a thin real-container contract layer).
  It currently lives **only in `PRD.md`** (Implementation + Testing Decisions),
  with **no ADR**, unlike all nine other decisions. Step E is "extract ADRs," so:
  propose promoting it to **`ADR-0010`** and adding **`D10`** to the
  `CONTEXT.md` register. OK to create? *Recommended — it's already decided in the
  PRD; this only records it at ADR level for consistency.*

### Resolved during Step E (no confirmation needed)

- **E-fix-1** — `CONTEXT.md` status footer said "Next process step: B (PRD)",
  which was stale (B and C are done). Updated to reflect Step C complete →
  Step E in progress. (Safe, factual.)

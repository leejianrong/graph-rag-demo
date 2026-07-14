# CLAUDE.md — Graph RAG Demo (agent brief)

A locally-run **Graph RAG** pipeline brought up via one Docker Compose. Ingestion is
Kafka-triggered (`POST /ingest` → MinIO + trigger → in-process consumer runs
`read → NER → coref → entity-linking → KG-build`); querying is a synchronous REST
`/query`. Python 3.12, `uv`, modular layout, type hints + docstrings throughout.

## Build status (honest)

**V1 (walking skeleton) — DONE.** Foundation contract (ports + fakes + models +
config + logging + ids), MinIO `ObjectStore`, ES `DocumentStore`, Kafka trigger
publisher/consumer, FastAPI `POST /ingest` + `GET /health`, orchestrator, `main.py`
composition root, `docker-compose.yml`.

**V2 (NER) — LANDED.** First real enrichment: `graph_rag/stages/ner.py` — a
constructor-injected `NerStage` seam with a real `SpacyNerStage` (one spaCy pass →
curated-type mentions + char spans + sentence segmentation; `GPE`+`LOC`→`LOCATION`;
model from `Settings.ner_model`, trf→lg→sm fallback) and a `FakeNerStage` for the
fast suite. `Orchestrator.process_document` now returns a `PipelineResult` carrying
the raw `DocumentRecord` plus in-memory `mentions`/`sentences` — **NOT persisted to
ES** (that lands at the V4 EL checkpoint; the ES write model is unchanged, raw text
only).

**V3 (coref + LLM client/cache) — LANDED.** First LLM use. `graph_rag/adapters/llm_client.py`
— `LiteLLMClient` (provider-agnostic via LiteLLM): `complete()` + `structured(prompt, schema)`
(Pydantic-validated JSON with retry, default 2), a persistent response cache keyed
`sha256(model+prompt+params)` under `.cache/llm/` (a cache hit never calls the provider,
observably `$0`), and an **injectable backend seam** (`completion_fn`, default lazy
`litellm.completion`) so tests count provider calls with no network/key. `graph_rag/stages/coref.py`
— a constructor-injected `CorefStage` seam with a real `LLMCorefStage` (LLM structured output →
`ClusterMap`) and a `FakeCorefStage`. `Orchestrator.process_document` now runs
read→NER→**coref** and returns a `PipelineResult` also carrying `coref_clusters` (a
non-destructive within-doc cluster map, mention surface forms → in-doc canonical) — still
**NOT persisted to ES** (V4 EL checkpoint). `FakeLLMClient` now returns canned STRUCTURED
responses + counts `.calls`. Per-stage model from `Settings` (coref pins B6 = `gpt-4o-mini`);
API key strictly from env.

**V4 (entity linking + EL checkpoint) — LANDED.** The corpus-local cross-document
unification that turns the same entity into one canonical record (the heart of R3,
ADR-0004/0005). `graph_rag/stages/entity_linking.py` — a constructor-injected
`ELStage` seam with the real `EntityLinkingStage` over the `EntityStore` + `Embedder`
ports; per doc-level entity (derived from coref clusters, else lone mentions) it
**embeds mention-in-context → blocks (type + normalized name) + kNN → scores by
cosine → merges** above `Settings.el_threshold` (B2 = 0.82) **or create-news**
(the always-on path). `canonical_id` is deterministic + stable
(`e-sha256("el:{type}:{normalized_name}")`), so a re-ingest of the same corpus
merges rather than duplicates; linking is **order-sensitive** (first doc seeds the
canonical name + vector). Alongside the real `SentenceTransformerEmbedder` +
`EsEntityStore` adapters (Agent A). **The EL checkpoint now persists the enriched
`ES-Documents` record** — `Orchestrator.process_document` runs
read→NER→coref→**EL**, then enriches the SAME `DocumentRecord` in place (raw text +
`mentions` + `coref_clusters` + `el_result` + `sentence_vectors`) and re-upserts it
(a 2nd idempotent write, overwriting the raw record), **and upserts the canonical
entities to `ES-Entities`**. The EL stage is **opt-in via injection** (absent → the
raw-only V1–V3 write model is preserved). The gated **LLM tie-breaker + NIL
retention are wired but OFF by default** (`el_tiebreaker_enabled`/`el_nil_enabled`),
so the default EL path is deterministic and `$0` (no LLM call).

**V5 (KG-build + graph checkpoint) — LANDED.** The knowledge graph is materialized
(realizes R0 — the graph exists; ADR-0006). `graph_rag/stages/kg_build.py` — a
constructor-injected `KgStage` seam with the real `KgBuildStage` over the
`LLMClient` port: it hands the model the doc's **canonical id↔name/type map** and
emits triples `(subject_id, predicate, object_id, sentence_index, date?, confidence?)`
(structured output → `TripleList`/`LLMTriple`) whose subject/object are **canonical
entity IDs**, never surface strings. Per triple: the raw predicate maps to the
**closed ~12-predicate set** via `map_predicate`, falling back to `RELATED_TO` with
the original phrase preserved on `EdgeProvenance.raw_predicate`; **char offsets are
resolved from OUR spaCy sentence segmentation** (the LLM cites only a
`sentence_index`; an out-of-range index is logged-and-skipped), filling
`source_sentence` + `char_start`/`char_end`; **DATE is an edge qualifier**
(`Triple.date`), never a node; a triple referencing an unknown canonical id is
dropped. `KgBuildStage.from_settings` builds a `LiteLLMClient` on
`Settings.kg_build_model` (own model, shared cache); the fast suite injects it over
`FakeLLMClient` canned triples. `Orchestrator.__init__` gains opt-in
`graph_store` + `kg_build_stage` (like EL — `None` skips, so V1–V4 tests are
unaffected). After the EL checkpoint the shell runs the **graph checkpoint**:
`upsert_entities` (multi-label `:Entity:Type` nodes, idempotent by `canonical_id`)
→ `delete_document_edges(document_id)` → `write_triples` — the **delete-then-write**
so RE-INGESTING a document REPLACES its edges rather than duplicating (closes the
TESTING graph-idempotency gap; nodes stay shared/idempotent). Triples are carried
in-memory on `PipelineResult.triples` (they live in **Neo4j**, not ES). Alongside
the real `Neo4jGraphStore` adapter (Agent A). Whole body stays inside log-and-drop
(a KG-build failure drops the doc, never wedges the loop). V6–V8 (retrieval →
benchmark) are **not built**.

> **Trust the code over the docs.** `docs/` (ARCHITECTURE, SLICES, TESTING, ADRs)
> is the design intent; where code and docs disagree, the code on this branch is
> the truth. Read the actual `graph_rag/ports.py` contract before coding against it.

## Exact commands

```bash
# Install (frozen — resolve the exact locked versions)
uv sync --frozen --extra dev

# Fast suite — the $0 gate: fakes only, NO Docker, no model, no LLM provider.
# Run this before every push.
uv run pytest -m "not contract and not model and not llm"

# Contract suite — real adapters via testcontainers. Needs a running Docker daemon.
uv run pytest -m contract

# Model suite — real spaCy NER. Fetch the model once, then run (NOT in the fast gate).
make models   # == python -m spacy download en_core_web_sm  (make models-trf for trf)
uv run pytest -m model

# Lint + format check
uv run ruff check .
uv run ruff format --check .

# Bring the whole stack up locally (Kafka + MinIO + Elasticsearch + service)
docker compose up
```

Run the whole stack in one command; a newcomer/agent runs `docker compose up` first.

## Testing philosophy — fakes-first at the port seam (ADR-0010)

- **One primary seam:** the six external-dependency ports in `graph_rag/ports.py`
  (`ObjectStore`, `DocumentStore`, `EntityStore`, `GraphStore`, `LLMClient`,
  `Embedder`) plus the `TriggerPublisher` messaging seam. Everything is
  constructor-injected; nothing constructs a live client inside request/pipeline code.
- **Fast suite runs against in-memory fakes** (`graph_rag/fakes.py`, exposed as
  fixtures in `tests/conftest.py`) — deterministic, `$0`, no Docker.
- **Contract layer** (`tests/contract/`, marked `contract`) proves each real
  adapter behaves like its fake — one contract test per port — via testcontainers.
  It gates the adapters, not the pipeline logic, and is excluded from the fast gate.
- Assert on **external behaviour at the seam**, never on internal call order. See
  `docs/TESTING.md`.

## Layered gates (by cost)

| Layer | Command | Where |
|-------|---------|-------|
| lint/format | `ruff check .` + `ruff format --check .` | pre-push + CI |
| fast ($0, no infra) | `uv run pytest -m "not contract and not model and not llm"` | pre-push + CI (**required gate**) |
| model (spaCy, no Docker) | `make models` + `uv run pytest -m model` | CI only (separate job) |
| llm (real provider) | `uv run pytest -m llm` | CI only (opt-in, NOT required) |
| contract (Docker) | `uv run pytest -m contract` | CI only (separate job) |

The `model` marker gates the real `SpacyNerStage` (loads `en_core_web_sm`, incl. a
model-availability smoke test). The `llm` marker gates the opt-in real-provider
`LiteLLMClient` test (V3): it needs an API key and **skips cleanly without one**, so
its CI job is NOT a required gate and holds no secret. Both need something the fast
gate lacks (a model / an API key), so both are **kept out of the fast pre-push gate** —
CI runs each in its own job. The fast gate stays model-free and LLM-free because the
orchestrator injects `FakeNerStage` + `FakeCorefStage` (or `FakeLLMClient`).

**Never let a slow check gate a local push.** The pre-push hook mirrors only the
cheap CI jobs.

### Pre-push hook (install once)

```bash
ln -sf ../../scripts/pre-push .git/hooks/pre-push   # version-controlled hook
```

It runs `ruff check .` + `uv run pytest -m "not contract and not model and not llm" -q`.
Escape hatch for a
rare scoped exception (docs-only, hotfix): `git push --no-verify` — CI is still the
real gate.

## Branch / PR conventions

- **Branch per vertical slice** off fresh `main` (e.g. `feat/v1-walking-skeleton`).
- **PR-only, protected `main`** — no direct pushes; merge after CI is green.
- Parallelize implementation only across **provably-disjoint file sets**; serialize
  the landing so `main` stays reviewable. Use worktrees for parallel work.
- Keep slices small and reversible; risky changes ship behind a flag defaulting off.

## Where things are

- `graph_rag/ports.py` — the port Protocols (the seam).
- `graph_rag/fakes.py` — in-memory fakes backing the fast suite.
- `graph_rag/models.py` — `IngestTrigger`, `DocumentRecord`, `Mention`/`Sentence`,
  `CorefCluster`/`ClusterMap`, `PipelineResult` (Pydantic v2).
- `graph_rag/config.py` — env-driven `Settings` / `get_settings()`.
- `graph_rag/adapters/` — real adapters (MinIO `ObjectStore`, ES `DocumentStore`,
  LiteLLM `LLMClient`).
- `graph_rag/stages/` — injected enrichment stages: `ner.py` (`NerStage`), `coref.py`
  (`CorefStage`).
- `graph_rag/messaging/` — Kafka trigger publisher + thin consumer.
- `graph_rag/api.py` — FastAPI `create_app(object_store, publisher, settings)`.
- `tests/e2e/` fast E2E · `tests/contract/` real-adapter contract · `tests/unit/` units ·
  `tests/model/` spaCy-model · `tests/llm/` opt-in real-provider.
- Design: `docs/ARCHITECTURE.md`, `docs/SLICES.md`, `docs/TESTING.md`, `docs/adr/`.

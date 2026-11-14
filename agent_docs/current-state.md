# Current state (what exists beyond V1 to V8)

Moved out of the root agent brief to keep it short. `build-status.md` covers how the eight pipeline slices work; this
file covers everything built after them. The code on `main` is the truth; where this and the code disagree, fix this.
For the open work and known quality gaps, read `docs/HANDOFF.md` and the pandan board (`GRA-*`, board 9).

## Surfaces

- **`/v1` agent API** (`graph_rag/agent_api.py`, `graph_rag/lookup.py`; contract `docs/api/agent-api.openapi.yaml`,
  ADR-0014): `remember`, `recall`, `forget`, `lookup`, plus `GET /v1/me` (the tenant a key opens; the UI signs in with
  it), `/v1/entities` browsing, `/v1/keys` self-service key management. Errors are `{"error": {"code", "message"}}`.
  Blank text is a 422 (GRA-164, GRA-167). `GET /jobs/{id}` gives job state, with `entities` and `edges` null until the
  job is done (GRA-150); `GET /jobs/{id}/events` streams the same progress as server-sent events.
- **Entity curation** (`graph_rag/curation_api.py`, `merge_records.py`, `merging.py`, `splitting.py`):
  `POST /v1/entities/{id}/merge` folds entities into one and records it; `.../split` undoes a recorded merge
  (GRA-144, GRA-145; ADR-0014 addendum).
- **CLI** `graphrag` (`graph_rag/cli.py`, extra `cli`), **MCP** `graph-rag-mcp` (`graph_rag/mcp_server.py`, extra `mcp`),
  **operator** `graph-rag-admin` (export, delete, usage per tenant), `graph-rag-keys` (API keys, per-tenant LLM keys),
  `graph-rag-worker` (separate ingest consumer), `graph-rag-offline` (the API over in-memory stores and heuristic
  stages, no Docker or LLM; the quickstart and the browser test use it).
- **Web UI** (`ui/`, Vite + Svelte + TypeScript): sign-in, ingest console, query playground, graph explorer, key
  management. Served at `/` by the API process (`graph_rag/ui_static.py`, `UI_DIR`) with a strict CSP and self-hosted
  fonts. The key lives in browser memory only. The client is generated from `docs/openapi.json` (`make openapi`).
- **Generator UI** (`kggen ui`, `make ui`): a different thing, the synthetic data generator's page built on indah.

## Pipeline behaviour added after V8

- **Entity linking name guard** (GRA-157): a cosine merge also needs a shared distinctive token or an acronym. Known
  hole: Acme Logistics and Northwind Logistics would merge on the shared token "Logistics" (GRA-172, read from the code, not run).
- **Answer selection** (GRA-161) and `answer_entities` for a second ask (GRA-165): ADR-0014, "How the answer is chosen".
- **`ACQUIRED` predicate** and a `role` property on edges (`HAS_ROLE` title), GRA-162 (`graph_rag/predicates.py`).
- **Time**: `document_date` on ingest, `valid_from`/`valid_to` on facts, `as_of` on recall; a newer document ends an older
  fact for single-valued predicates (`graph_rag/supersession.py`). Temporal edges are otherwise not modelled (GRA-171).
- **Tenancy and auth** (ADR-0015): the four store ports take a `Scope` first (`graph_rag/scope.py`); `REQUIRE_AUTH`
  makes the API key name the tenant (`graph_rag/auth.py`); `/v1`'s `namespace` is the part of a scope a caller sets.
  `REQUIRE_TENANT_LLM_KEY` (GRA-25) makes a tenant use its own LLM key, stored encrypted (`graph_rag/tenant_keys.py`).
  `METER_USAGE` (GRA-58) counts documents, queries and LLM tokens per tenant in a file.
- **Ingest queue** (ADR-0016): Valkey / Redis Streams by default (`TRIGGER_BACKEND=redis`), Kafka optional
  (`--profile kafka`). `graph_rag/messaging/factory.py` builds either. `INGEST_MODE=publish` (set by both compose files)
  makes the API publish only, with `graph-rag-worker` consuming and job status in Elasticsearch (`EsJobStore`); the
  default `inline` keeps a consumer thread and in-memory jobs.
- **LLM gate**: ingest (`/ingest`, `/v1/remember`) is refused with 503 `llm_not_configured` when neither `OPENAI_API_KEY`
  nor `OPENAI_API_BASE` is set (`graph_rag/llm_config.py`); the offline server and fakes skip the check.
- **Observability** (GRA-56, off by default): `/metrics` (`METRICS_ENABLED`, `METRICS_TOKEN` with auth on), OTLP spans
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, `traceparent` on the trigger, no tenant, key or text in a label or span.
  Dashboard in `observability/` (`docker-compose.observability.yml`).

## Bench and generator

- `datagen/` (`kggen`): seeded world, documents, questions, scenarios, `kggen export`; ADR-0012. `datagen/fixtures/` and
  `datagen/examples/` are committed outputs (their own READMEs).
- `bench/` (`kgbench`): extraction, linking, retrieval (BM25, dense, graph-rag), order, sweeps, typed answers, LLM judge,
  `real` (2Wiki slice), `longmemeval`, `ingest-cost` (GRA-55), `gate` (floors in `bench/kgbench/gate.json`, run by the fast
  suite; `--update` after an intended change) and `nightly`. Results are in `docs/results/`, method in `docs/BENCH.md`.
- `build_offline_components(llm_stages=True)` runs the real coref and KG-build stages over in-memory stores; point
  LiteLLM at semula with `OPENAI_API_BASE` (workflow in `docs/HANDOFF.md`).
- The four-document replay (`tests/model/test_e2e_four_docs.py`, GRA-163, GRA-168) runs in the CI `model` job with
  `en_core_web_trf`.

## Decisions without a migration

ADR-0017 (GRA-54) chose Postgres with pgvector as the direction for the four stateful stores. A prototype of the store
ports is in `spikes/pg_stores/` (outside the product, the type check and the fast suite). The build is GRA-169.

## Not built

A Python SDK (GRA-40), a staging environment (GRA-57), a real-stack quality run at scale (GRA-15), GPU use (GRA-177).

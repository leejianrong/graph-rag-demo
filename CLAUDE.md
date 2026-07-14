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
only). V3–V8 (coref → benchmark) are **not built**.

> **Trust the code over the docs.** `docs/` (ARCHITECTURE, SLICES, TESTING, ADRs)
> is the design intent; where code and docs disagree, the code on this branch is
> the truth. Read the actual `graph_rag/ports.py` contract before coding against it.

## Exact commands

```bash
# Install (frozen — resolve the exact locked versions)
uv sync --frozen --extra dev

# Fast suite — the $0 gate: fakes only, NO Docker. Run this before every push.
uv run pytest -m "not contract"

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
| fast ($0, no infra) | `uv run pytest -m "not contract and not model"` | pre-push + CI (**required gate**) |
| model (spaCy, no Docker) | `make models` + `uv run pytest -m model` | CI only (separate job) |
| contract (Docker) | `uv run pytest -m contract` | CI only (separate job) |

The `model` marker gates the real `SpacyNerStage` (loads `en_core_web_sm`, incl. a
model-availability smoke test). It needs a downloaded model, so it is **kept out of
the fast pre-push gate** — CI runs it in its own job. The fast gate stays model-free
because the orchestrator injects `FakeNerStage`.

**Never let a slow check gate a local push.** The pre-push hook mirrors only the
cheap CI jobs.

### Pre-push hook (install once)

```bash
ln -sf ../../scripts/pre-push .git/hooks/pre-push   # version-controlled hook
```

It runs `ruff check .` + `uv run pytest -m "not contract" -q`. Escape hatch for a
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
- `graph_rag/models.py` — `IngestTrigger`, `DocumentRecord` (Pydantic v2).
- `graph_rag/config.py` — env-driven `Settings` / `get_settings()`.
- `graph_rag/adapters/` — real adapters (MinIO `ObjectStore`, ES `DocumentStore`).
- `graph_rag/messaging/` — Kafka trigger publisher + thin consumer.
- `graph_rag/api.py` — FastAPI `create_app(object_store, publisher, settings)`.
- `tests/e2e/` fast E2E · `tests/contract/` real-adapter contract · `tests/unit/` units.
- Design: `docs/ARCHITECTURE.md`, `docs/SLICES.md`, `docs/TESTING.md`, `docs/adr/`.

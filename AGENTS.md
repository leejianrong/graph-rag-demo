# AGENTS.md — Graph RAG (agent brief)

This is the single canonical agent brief; `CLAUDE.md` imports it. Keep it short — put depth in
`agent_docs/` and `docs/`.

## What and why

A **Graph RAG** pipeline: documents → knowledge graph → multi-hop `/query`. Ingestion is queue-triggered
(`POST /ingest` → MinIO + trigger on a Valkey stream by default, Kafka via `TRIGGER_BACKEND=kafka`, ADR-0016 → consumer runs `read → NER → coref → entity linking →
KG-build`); querying is a synchronous REST `/query` with a **$0, deterministic, no-LLM default path**.
Python 3.12, `uv`, type hints + docstrings, everything constructor-injected at the port seam.

Direction (2026-11): turning the demo into a product — a memory/knowledge store for agents and developers
(API, MCP, CLI, Vite + Svelte UI). The roadmap lives on **pandan board 9 (`GRA-*`)**, not in the docs; use the `pandan` skill, refer to cards by ref (`GRA-85`).

**Build status:** V1–V8 are done and the pipeline is feature-complete ([`agent_docs/build-status.md`](agent_docs/build-status.md)).
Everything after that is in [`agent_docs/current-state.md`](agent_docs/current-state.md): the `/v1` API (and SSE job events), tenancy and
auth, the Valkey queue and workers, the CLI, MCP server and Svelte UI, entity merge/split, the generator (`datagen/`), the bench
(`bench/`, ADR-0011; results in [`docs/BENCH.md`](docs/BENCH.md)) and its quality gate (`kgbench gate`, floors in `bench/kgbench/gate.json`,
run by the fast suite; `--update` after an intended change). **Open:** the Python SDK, a staging environment, real-stack quality at
scale, and graph-rag not yet beating BM25 or dense on `hard-small`. A person or agent on a new machine starts at
**[`docs/HANDOFF.md`](docs/HANDOFF.md)** (setup, state, gaps, and every file that is not in git).

> **Trust the code over the docs.** `docs/` is design intent; where they disagree, the code is the truth and
> the doc should be fixed in the same PR. Read `graph_rag/ports.py` before coding against a port.

## Hard rules (these have bitten before)

- **Commit with the GitHub noreply email** (`85485699+leejianrong@users.noreply.github.com`). Pushes with the
  private address are rejected. The repo-local git config is already set; check it in a fresh clone.
- **`main` is protected and PR-only, for admins too.** No direct pushes, linear history, conversation
  resolution. Squash-merge once `lint`, `fast`, `contract`, `secret-scan` and `image` are green on the exact pushed SHA, never on stale
  green checks; `dep-audit` is advisory but should be green. Never `--admin`-merge.
- **Never let a slow check gate a local push.** The pre-push hook is the cheap mirror of CI only.
- **`datagen/` (`kggen`) must never import `graph_rag`, `kgbench` or `litellm`** (enforced by
  `datagen/tests/test_isolation.py`); it owns its vocabulary. **`bench/` may import `graph_rag` only via ports
  and public entry points** (`bench/tests/test_import_boundary.py`). Gold must never come from graph-rag's own extraction.
- **Plain `uv sync` prunes the heavy extras** (torch, sentence-transformers). Use
  `uv sync --frozen --extra dev --extra embed` locally if you run the model suite.
- **`bench/` and `datagen/` are uv workspace members**, so the Dockerfile must `COPY` both `pyproject.toml`
  files before `uv pip install --no-deps .` (uv parses the workspace). Keep them when touching the image.

## Commands

Bare `make` lists every target (`make check` = lint + format check + fast suite; `make ui` starts the generator UI).

```bash
uv sync --frozen --extra dev                                  # install (frozen; installs kggen and kgbench too)
uv run pytest -m "not contract and not model and not llm"     # FAST gate: fakes only, no Docker/model/LLM
uv run ruff check . && uv run ruff format --check . && uv run mypy   # lint + format + types
uv run pytest -m contract                                     # real adapters via testcontainers (Docker)
make models && uv run pytest -m model                         # real spaCy NER (not in the fast gate); trf replay needs `make models-trf`
uv run pytest -m llm                                          # opt-in real provider; skips without a key
docker compose up                                             # whole stack (or: make up)
make demo-offline | demo-live | demo | benchmark              # see `make` targets
uv run kggen world generate --preset medium                   # a ground-truth world + stats (datagen)
uv run kggen questions generate --preset small | docs generate --preset small   # questions / template documents
uv run kggen fixtures write   # regenerate the committed small/medium sets (datagen/fixtures)
uv run kggen scenario generate --preset small   # update/delete/reorder events + the expected final graph
uv run kggen export --spec spec.json --out dataset.zip   # the whole dataset as one reproducible zip
```

Run the fast gate and ruff before every push. Install the hook once:
`ln -sf ../../scripts/pre-push .git/hooks/pre-push`.

## Testing

Fakes-first at the port seam (ADR-0010): the six ports in `graph_rag/ports.py` (`ObjectStore`, `DocumentStore`,
`EntityStore`, `GraphStore`, `LLMClient`, `Embedder`) plus `TriggerPublisher`. The fast suite uses the in-memory
fakes in `graph_rag/fakes.py`; contract tests prove each real adapter behaves like its fake. Assert external
behaviour at the seam, never internal call order. See `docs/TESTING.md`. Markers: `contract`, `model`, `llm`,
`benchmark`, `e2e`. `bench/tests` and `datagen/tests` run in the fast suite.

## Workflow

- Branch per slice off fresh `main`; stage by explicit path, not `git add -A`, while a background job writes into the repo; small, reversible; risky changes ship behind a flag defaulting off.
- Parallelise only across provably disjoint file sets; use worktrees; serialise the landing.
- Update docs, README and this brief in the same PR that changes what they describe.

## Where things are (only the non-obvious)

- `ui/` — the product web UI (Vite + Svelte + TS; `make ui-install | ui-dev | ui-build`, CI job `ui`; not `make ui`, which is the generator UI). Its API client is generated from `docs/openapi.json`; after an API change run `make openapi` (a fast test fails on drift). One file per view in `ui/src/views/`. The API process serves the build at `/` (`graph_rag/ui_static.py`, `UI_DIR`, default `/app/ui-dist` in the image, none if absent); `make ui-e2e` runs the Playwright test against `graph-rag-offline` (CI job `e2e`).
- `graph_rag/` — the pipeline; `datagen/kggen/` — the standalone data generator (CLI `kggen`); `bench/kgbench/` — the evaluation bench (CLI `kgbench`).
- `docs/adr/` — the "why" (ADR-0011 is the bench, ADR-0012 the standalone generator); `docs/ARCHITECTURE.md`, `docs/TESTING.md`.
- `agent_docs/build-status.md` — how each V1–V8 slice works in detail.
- `docs/adr/README.md` indexes ADRs 0001–0017; `docs/openapi.json` is generated (`make openapi`).
- Real corpus lives under gitignored `datasets/`; the LLM response cache is `.cache/llm/` (gitignored). `docs/DATA-PROVENANCE.md`
  says where every ignored or external file comes from. Caches under `bench/kgbench/data/` need `git add -f` (`data/` is ignored).

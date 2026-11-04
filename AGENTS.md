# AGENTS.md — Graph RAG (agent brief)

This is the single canonical agent brief; `CLAUDE.md` imports it. Keep it short — put depth in
`agent_docs/` and `docs/`.

## What and why

A **Graph RAG** pipeline: documents → knowledge graph → multi-hop `/query`. Ingestion is Kafka-triggered
(`POST /ingest` → MinIO + trigger → in-process consumer runs `read → NER → coref → entity linking →
KG-build`); querying is a synchronous REST `/query` with a **$0, deterministic, no-LLM default path**.
Python 3.12, `uv`, type hints + docstrings, everything constructor-injected at the port seam.

Direction (2026-11): turning the demo into a product — a memory/knowledge store for agents and developers
(API, MCP, CLI, then a Vite + Svelte UI), with a standalone synthetic data generator (`datagen/`, ADR-0012) and an in-repo evaluation bench. The roadmap lives on
**pandan board 9 (`GRA-*`)**, not in the docs; use the `pandan` skill, refer to cards by ref (`GRA-85`).

**Build status:** V1–V8 are done and the pipeline is feature-complete ([`agent_docs/build-status.md`](agent_docs/build-status.md)).
The generator (`datagen/`, `kggen`) and its indah UI (`kggen ui`, extra `kggen[ui]`) are done; the UI's graph view needs
indah 0.12, unreleased, so it runs against the local indah checkout (`make ui` picks it up). The bench (`bench/`, ADR-0011) is
partly built: extraction and linking scoring (`kgbench extract`, `kgbench linking`) are done, and retrieval scoring (`kgbench.retrieval`, defines the `Retriever` interface) is built; a BM25 baseline (`kgbench.baselines`) is built; the dense baseline and a graph-rag adapter are still to build.

> **Trust the code over the docs.** `docs/` is design intent; where they disagree, the code is the truth and
> the doc should be fixed in the same PR. Read `graph_rag/ports.py` before coding against a port.

## Hard rules (these have bitten before)

- **Commit with the GitHub noreply email** (`85485699+leejianrong@users.noreply.github.com`). Pushes with the
  private address are rejected. The repo-local git config is already set; check it in a fresh clone.
- **`main` is protected and PR-only.** Merge (squash) once the required checks `lint`, `fast`, `contract`,
  `secret-scan` are green. `dep-audit` is advisory but should be green too. Never merge on stale green checks.
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
uv run ruff check . && uv run ruff format --check .           # lint + format
uv run pytest -m contract                                     # real adapters via testcontainers (Docker)
make models && uv run pytest -m model                         # real spaCy NER (not in the fast gate)
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

- Branch per slice off fresh `main`; small, reversible; risky changes ship behind a flag defaulting off.
- Parallelise only across provably disjoint file sets; use worktrees; serialise the landing.
- Update docs, README and this brief in the same PR that changes what they describe.

## Where things are (only the non-obvious)

- `graph_rag/` — the pipeline; `datagen/kggen/` — the standalone data generator (CLI `kggen`); `bench/kgbench/` — the evaluation bench (CLI `kgbench`).
- `docs/adr/` — the "why" (ADR-0011 is the bench, ADR-0012 the standalone generator); `docs/ARCHITECTURE.md`, `docs/TESTING.md`.
- `agent_docs/build-status.md` — how each V1–V8 slice works in detail.
- Real corpus lives under gitignored `datasets/`; the LLM response cache is `.cache/llm/` (gitignored).

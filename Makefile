# One-command dev loop (dev-playbook #17). See docs/ARCHITECTURE.md §8.
# Bare `make` prints this help; it never starts anything.
.DEFAULT_GOAL := help
.PHONY: help up down logs demo demo-live demo-offline check-site test benchmark test-benchmark contract model models models-trf embed-model lint types fmt check ui ui-install ui-dev ui-build ui-e2e openapi world fixtures extract linking order sweep compare gate nightly longmemeval

# The generator UI uses a local indah checkout when one exists, else PyPI indah (0.12 and later have
# the Graph component; 0.13.0 is on PyPI). Override: INDAH=/path/to/indah-python-ui, or INDAH= for PyPI.
INDAH ?= $(wildcard $(HOME)/projects/abang-ai/indah-python-ui)
PRESET ?= medium

help: ## Show this help.
	@echo "Targets (override variables like PRESET=small or PORT=8765):"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Bring up the whole local stack (Valkey/MinIO/ES/app), building the app image.
up: ## Bring up the whole stack (Valkey, MinIO, ES, Neo4j, app).
	docker compose up --build

# Tear the stack down.
down: ## Stop the stack.
	docker compose down

# Tail the app logs.
logs: ## Tail the app logs.
	docker compose logs -f app

# End-to-end demo over the real running stack (needs `make up` + OPENAI_API_KEY):
# ingests the bundled supply-chain corpus, waits for the graph, runs the multi-hop
# query and prints the connected subgraph. Add SYNTHESIZE=1 for a prose answer.
demo: ## Multi-hop demo against the running stack (needs OPENAI_API_KEY).
	uv run python -m graph_rag.demo --http http://localhost:8000 $(if $(SYNTHESIZE),--synthesize,)

# Self-sufficient real-stack demo: bring the WHOLE stack up (build + start), block
# on `--wait` until every service — including the app healthcheck — is healthy, then
# ingest the bundled corpus, build the graph and run the multi-hop query. Needs
# Docker + OPENAI_API_KEY in .env (coref + KG-build call an LLM). Leaves the stack UP
# afterwards: explore Neo4j at http://localhost:7474, re-run the query cheaply with
# `make demo`, or `make down` to stop. Add SYNTHESIZE=1 for a prose answer.
demo-live: ## Start the whole stack, then run the demo (needs Docker + key).
	docker compose up -d --build --wait
	uv run python -m graph_rag.demo --http http://localhost:8000 $(if $(SYNTHESIZE),--synthesize,)

# Same multi-hop demo with NO Docker, NO model, NO API key — the deterministic
# offline heuristic pipeline over the same bundled corpus ($0).
demo-offline: ## Multi-hop demo with no Docker, model or key.
	uv run python -m graph_rag.demo

# Fast suite — in-memory fakes, $0, no Docker, no model. The primary pre-push gate.
test: ## Fast suite: fakes only, no Docker/model/key (the pre-push gate).
	uv run pytest -m "not contract and not model and not llm" -q

# Run the V8 benchmark and print the metrics scorecard (EM / token-F1 /
# supporting-fact P·R·F1). Offline by default: no Docker, no model, no key ($0).
# Point at the real corpus with DATASET=..., or add REAL=1 for the real stack.
benchmark: ## V8 benchmark scorecard, offline (DATASET=... or REAL=1).
	uv run kgbench run --subset small $(if $(DATASET),--dataset $(DATASET),) $(if $(REAL),--real,)

# Opt-in whole-pipeline benchmark SMOKE TEST over the real stack (V8 pytest
# marker; skips cleanly without infra). This is the test, not the scorecard.
test-benchmark: ## Whole-pipeline benchmark smoke test over the real stack.
	uv run pytest -m benchmark

# Contract suite — real adapters via testcontainers (needs Docker).
contract: ## Contract suite: real adapters via testcontainers (Docker).
	uv run pytest -m contract

# Model-backed NER suite — loads a real spaCy model (run `make models` first).
# NOT part of the pre-push gate; runs in a dedicated CI job.
model: ## Model-backed NER suite (run `make models` first).
	uv run pytest -m model

# Fetch the small spaCy model used by the tests + trf fallback ($0, no extras).
models: ## Download en_core_web_sm (the model suite and the bench use it).
	uv run python -m spacy download en_core_web_sm

# Fetch the transformer model for the real stack and the four-document replay (needs the `trf` extra; about 400 MB).
models-trf: ## Download en_core_web_trf (trf extra; the four-document replay and the default NER_MODEL).
	uv run --extra trf python -m spacy download en_core_web_trf

# Fetch the sentence-transformer embedding model (V4, B1; needs the `embed` extra).
# Also warms it for the `model`-marked embedder test.
embed-model: ## Download the embedding model (embed extra).
	uv run --extra embed python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Validate the GitHub Pages landing page (site/index.html parses + every in-page
# anchor resolves) before it ships. Mirrors the landing-page CI job; stdlib only.
check-site: ## Validate the GitHub Pages landing page.
	python scripts/check_landing_page.py

# Lint.
lint: ## Ruff lint.
	uv run ruff check .

types: ## Type-check graph_rag, kggen and kgbench with mypy (baseline config in pyproject.toml).
	uv run mypy

# Auto-format.
fmt: ## Ruff auto-format.
	uv run ruff format .

# Lint, format check and the fast suite: what the pre-push hook and CI's cheap jobs run.
check: ## Lint + format check + types + fast suite (run before pushing).
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	$(MAKE) test

# Synthetic data generator (datagen/, kggen). No Docker, model or key needed.
# The UI picks the first free port itself; PORT=8765 pins one.
ui: ## Start the synthetic-data generator UI (kggen ui; not the product UI, see ui-dev). PORT=... to pin a port.
	@echo "kggen ui with $(if $(INDAH),local indah at $(INDAH),PyPI indah (the graph view needs indah 0.12 or later))"
	uv run $(if $(INDAH),--with-editable $(INDAH),--with indah) kggen ui $(if $(PORT),--port $(PORT),)

# The product web UI (ui/, Vite + Svelte). Needs Node 22+; the API client is generated from docs/openapi.json.
ui-install: ## Install the web UI's npm dependencies (ui/).
	cd ui && npm ci

ui-dev: ## Web UI dev server; proxies the API at http://localhost:8000 (API_URL=... to change).
	cd ui && npm run dev

ui-build: ## Type-check and build the web UI into ui/dist.
	cd ui && npm run check && npm run build

ui-e2e: ## Browser test: sign in, ingest, query, sign out over graph-rag-offline (needs `npx playwright install chromium`).
	cd ui && npm run build && npm run e2e

openapi: ## Regenerate docs/openapi.json from the app (the UI client is generated from it).
	uv run python scripts/gen_openapi.py

world: ## Generate a world and print its stats vs the spec (PRESET=tiny|small|medium|large|stress).
	uv run kggen world stats --preset $(PRESET)

fixtures: ## Regenerate the committed small and medium fixtures (datagen/fixtures).
	uv run kggen fixtures write

extract: ## Score extraction (mentions, entities, triples) against generated gold (PRESET=...).
	uv run kgbench extract --preset $(PRESET)

linking: ## Score entity linking (B-cubed) per difficulty setting (PRESET=...).
	uv run kgbench linking --preset $(PRESET)

longmemeval: ## LongMemEval-style chat memory, per question type (DATA=path/to/longmemeval.json; default: bundled mini fixture, offline).
	uv run kgbench longmemeval $(if $(DATA),--data $(DATA),)

order: ## Ingest one corpus in shuffled orders and report the spread (PRESET=..., offline).
	uv run kgbench order --preset $(PRESET)

# bge needs the embed extra; the default sweep takes over an hour with bge. KNOB=avg_degree for one knob.
sweep: ## Vary one world knob at a time and score every system (KNOB=all|n_entities|avg_degree|..., EMBEDDER=bge|fake).
	uv run --extra embed kgbench sweep --knob $(or $(KNOB),all) --preset small --set n_entities=500 --embedder $(or $(EMBEDDER),bge) --out docs/results/sweeps

nightly: ## Answer EM/F1 vs the stored baseline (what the nightly workflow runs, offline).
	uv run kgbench nightly

gate: ## Fail if graph-rag scores below the floors in bench/kgbench/gate.json (offline, ~15 s).
	uv run kgbench gate

# bge needs the embed extra and `make embed-model`; EMBEDDER=fake is fast and for checking only.
compare: ## Compare BM25, dense and graph-rag retrieval over seeds (PRESET, EMBEDDER=bge|fake).
	uv run --extra embed kgbench compare --preset $(PRESET) --embedder $(or $(EMBEDDER),bge)

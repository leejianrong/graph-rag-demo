# One-command dev loop (dev-playbook #16). See docs/ARCHITECTURE.md §8.
.PHONY: up down logs demo demo-live demo-offline check-site test benchmark test-benchmark contract model models models-trf embed-model lint fmt

# Bring up the whole local stack (Kafka/MinIO/ES/app), building the app image.
up:
	docker compose up --build

# Tear the stack down.
down:
	docker compose down

# Tail the app logs.
logs:
	docker compose logs -f app

# End-to-end demo over the real running stack (needs `make up` + OPENAI_API_KEY):
# ingests the bundled supply-chain corpus, waits for the graph, runs the multi-hop
# query and prints the connected subgraph. Add SYNTHESIZE=1 for a prose answer.
demo:
	uv run python -m graph_rag.demo --http http://localhost:8000 $(if $(SYNTHESIZE),--synthesize,)

# Self-sufficient real-stack demo: bring the WHOLE stack up (build + start), block
# on `--wait` until every service — including the app healthcheck — is healthy, then
# ingest the bundled corpus, build the graph and run the multi-hop query. Needs
# Docker + OPENAI_API_KEY in .env (coref + KG-build call an LLM). Leaves the stack UP
# afterwards: explore Neo4j at http://localhost:7474, re-run the query cheaply with
# `make demo`, or `make down` to stop. Add SYNTHESIZE=1 for a prose answer.
demo-live:
	docker compose up -d --build --wait
	uv run python -m graph_rag.demo --http http://localhost:8000 $(if $(SYNTHESIZE),--synthesize,)

# Same multi-hop demo with NO Docker, NO model, NO API key — the deterministic
# offline heuristic pipeline over the same bundled corpus ($0).
demo-offline:
	uv run python -m graph_rag.demo

# Fast suite — in-memory fakes, $0, no Docker, no model. The primary pre-push gate.
test:
	uv run pytest -m "not contract and not model and not benchmark"

# Run the V8 benchmark and print the metrics scorecard (EM / token-F1 /
# supporting-fact P·R·F1). Offline by default: no Docker, no model, no key ($0).
# Point at the real corpus with DATASET=..., or add REAL=1 for the real stack.
benchmark:
	uv run benchmark run --subset small $(if $(DATASET),--dataset $(DATASET),) $(if $(REAL),--real,)

# Opt-in whole-pipeline benchmark SMOKE TEST over the real stack (V8 pytest
# marker; skips cleanly without infra). This is the test, not the scorecard.
test-benchmark:
	uv run pytest -m benchmark

# Contract suite — real adapters via testcontainers (needs Docker).
contract:
	uv run pytest -m contract

# Model-backed NER suite — loads a real spaCy model (run `make models` first).
# NOT part of the pre-push gate; runs in a dedicated CI job.
model:
	uv run pytest -m model

# Fetch the small spaCy model used by the tests + trf fallback ($0, no extras).
models:
	uv run python -m spacy download en_core_web_sm

# Fetch the transformer model for the real stack (needs the `trf` extra).
models-trf:
	uv run --extra trf python -m spacy download en_core_web_trf

# Fetch the sentence-transformer embedding model (V4, B1; needs the `embed` extra).
# Also warms it for the `model`-marked embedder test.
embed-model:
	uv run --extra embed python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Validate the GitHub Pages landing page (site/index.html parses + every in-page
# anchor resolves) before it ships. Mirrors the landing-page CI job; stdlib only.
check-site:
	python scripts/check_landing_page.py

# Lint.
lint:
	uv run ruff check .

# Auto-format.
fmt:
	uv run ruff format .

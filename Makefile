# One-command dev loop (dev-playbook #16). See docs/ARCHITECTURE.md §8.
.PHONY: up down logs test benchmark contract model models models-trf embed-model lint fmt

# Bring up the whole local stack (Kafka/MinIO/ES/app), building the app image.
up:
	docker compose up --build

# Tear the stack down.
down:
	docker compose down

# Tail the app logs.
logs:
	docker compose logs -f app

# Fast suite — in-memory fakes, $0, no Docker, no model. The primary pre-push gate.
test:
	uv run pytest -m "not contract and not model and not benchmark"

# Opt-in whole-pipeline benchmark smoke over the real stack (V8; skips without infra).
benchmark:
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

# Lint.
lint:
	uv run ruff check .

# Auto-format.
fmt:
	uv run ruff format .

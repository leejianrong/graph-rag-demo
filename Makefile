# One-command dev loop (dev-playbook #16). See docs/ARCHITECTURE.md §8.
.PHONY: up down logs test contract lint fmt

# Bring up the whole local stack (Kafka/MinIO/ES/app), building the app image.
up:
	docker compose up --build

# Tear the stack down.
down:
	docker compose down

# Tail the app logs.
logs:
	docker compose logs -f app

# Fast suite — in-memory fakes, $0, no Docker. The primary pre-push gate.
test:
	uv run pytest -m "not contract"

# Contract suite — real adapters via testcontainers (needs Docker).
contract:
	uv run pytest -m contract

# Lint.
lint:
	uv run ruff check .

# Auto-format.
fmt:
	uv run ruff format .

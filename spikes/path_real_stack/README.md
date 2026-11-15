# path_real_stack: GRA-170 real-text check of the chain search

Scores whole-path@5 for BM25, dense, graph-rag's supporting sentences, the plain graph view and the path view
(`PATH_RETRIEVAL`) on real 2Wiki questions through the real stores. Retrieval only: it needs the Elasticsearch,
Neo4j and MinIO that `kgbench run --real` filled, and re-uses them. Not wired into the product, not in the mypy
file list, not collected by the fast suite. Results: `docs/results/path-retrieval.md`.

```bash
# the stores, under your own project name (the 50-question corpus replays from the committed cache)
docker compose -p grsmoke --env-file .env.example up -d elasticsearch minio neo4j
export ELASTICSEARCH_URL=http://localhost:9200 MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin \
  MINIO_SECRET_KEY=minioadmin MINIO_SECURE=false NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j \
  NEO4J_PASSWORD=devpassword OPENAI_API_BASE=http://127.0.0.1:1/v1 OPENAI_API_KEY=unused \
  COREF_MODEL=openai/claude-haiku KG_BUILD_MODEL=openai/claude-haiku \
  LLM_CACHE_DIR=$PWD/bench/kgbench/data/llm-cache-2wiki-medium50
uv run --extra embed kgbench run --dataset datasets/2wiki_dev.json --subset medium --limit 50 --real   # replays
uv run --extra embed python -m spikes.path_real_stack.run --data datasets/2wiki_dev.json --limit 50
```

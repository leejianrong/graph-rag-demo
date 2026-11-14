# Data provenance: what is not in git, and how to get it back

Everything a developer relies on that a fresh `git clone` does not give you, plus the large committed files that
are generated. Checked on 2026-11-14 with `git status --ignored`, `.gitignore`, the Makefile, the compose files
and the code. Where something could not be verified it says so.

## Gitignored and external

| What | Where | Where it comes from, and how to recreate it |
|---|---|---|
| Python environment | `.venv/` | `uv sync --frozen --extra dev` (add `--extra embed`, `--extra trf` as needed). About 6.6 GB with the embed extra on the author's machine. spaCy model packages are not in `uv.lock`, so a plain `uv sync` removes them along with extras you did not name: re-run `make models` after one. |
| spaCy `en_core_web_sm` | `.venv/lib/python3.12/site-packages/en_core_web_sm` | `make models` (downloads the 3.8.0 wheel from the explosion/spacy-models GitHub release). 15 MB, measured. Used by the model suite, the bench's spaCy pipelines and the Docker image's fallback. |
| spaCy `en_core_web_trf` | same place | `make models-trf` (runs with `--extra trf`, which pulls spacy-transformers and torch). About 400 MB (the CI comment's figure; not re-measured). It is the default `NER_MODEL` and the only model the four-document replay accepts. CI pins 3.8.0 and caches the wheel in `~/.cache/spacy-models`. |
| bge embedding model | `~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5` (or `$HF_HOME`) | `make embed-model` (needs the `embed` extra); it is also fetched on first use. 129 MB, measured. `EMBED_MODEL` and `EMBED_DIM` pin the Elasticsearch vector mapping, so changing them means re-ingesting. CI sets `HF_HUB_DISABLE_XET=1` because the Hugging Face xet endpoint was unreliable. |
| 2WikiMultihopQA slice | `datasets/2wiki-dev-200.json` (855,746 bytes) | See "The 2Wiki slice" below. |
| LLM response cache | `.cache/llm/` (`LLM_CACHE_DIR`); per tenant under `.cache/llm/tenants/<tenant>/` | Written by `LiteLLMClient` on every uncached LLM call. Nothing to restore: delete it to force real calls (`rm -rf .cache/llm`). The key is `sha256(model + prompt + params)` (`graph_rag/adapters/llm_client.py`, `cache_key`), so a different model string, a different prompt (a coref or kg-build prompt edit, the predicate list) or a different NER model (the coref prompt embeds the mentions) misses and calls the provider. Absent on a fresh clone. |
| Environment file | `.env` | `cp .env.example .env`. Every variable is read by `graph_rag/config.py` and listed in `.env.example` with its default; the secrets are `OPENAI_API_KEY`, `NEO4J_PASSWORD`, `MINIO_*`, `ELASTICSEARCH_PASSWORD`, `REDIS_PASSWORD`, `METRICS_TOKEN`, `TENANT_LLM_KEY_SECRET`. Never commit it. The full list is under "Variables" below. |
| Service state files | `.data/` (`api-keys.json`, `tenant-llm-keys.json`, `usage.json`) | Written by `graph-rag-keys` and the service; gitignored. Recreate keys with `graph-rag-keys create TENANT`. |
| MCP registration | `.mcp.json` (gitignored; absent here) | Not needed. Register this repo's server with `claude mcp add graph-rag --env GRAPH_RAG_URL=http://localhost:8000 -- uv run --directory "$PWD" graph-rag-mcp` (`docs/QUICKSTART.md`). |
| Web UI dependencies and build | `ui/node_modules/`, `ui/dist/`, `ui/src/api/schema.d.ts`, `ui/playwright-report/`, `ui/test-results/` | `make ui-install` (`npm ci`), `make ui-build`; `schema.d.ts` is generated from `docs/openapi.json` by `npm run gen:api`, which build, check, test and dev run first. Playwright's browser is in `~/.cache/ms-playwright`: `cd ui && npx playwright install chromium`. |
| indah, for the generator UI | `~/projects/abang-ai/indah-python-ui` (clone of `https://github.com/leejianrong/indah.git`) | `make ui` uses this path when it exists (`INDAH=/other/path` overrides, `INDAH=` forces PyPI). It is no longer required: the graph view needs indah 0.12 or later, and 0.13.0 is on PyPI (the lock has it, and `hasattr(indah, "Graph")` is true for it). The old note that 0.12 is unreleased is wrong. `datagen/pyproject.toml` still says `indah>=0.11`; that floor would let an older install lose the graph view. |
| Benchmark output | none | Bench output goes where `--out` says; the published results are under `docs/results/`, each with its command at the top. |
| Docker images | local image store | `make up` builds `graph-rag-demo:local` (7.05 GB here: torch, spaCy trf, bge). It pulls `elasticsearch:8.13.4`, `neo4j:5.20`, `valkey/valkey:8.1.10`, `chainguard/minio:latest`, and with `--profile kafka` `apache/kafka:3.7.0`. semula is `ghcr.io/leejianrong/semula`. |
| Docker volumes | Docker | Compose creates `minio-data`, `neo4j-data`, `valkey-data` and `shared-state` (usage and tenant LLM keys for the API and workers). **Elasticsearch has no volume**, so its documents and entities are lost when the container is removed (`docker compose down`) while MinIO and Neo4j keep theirs. Compose prefixes volume names with the project name. `docker compose down -v` removes them all. |
| The ingest-cost run | scratch | Needs its own containers (`docker compose -p gra55cost up -d --wait elasticsearch minio neo4j`); the exact commands are at the bottom of `docs/results/ingest-cost.md`. |

## The 2Wiki slice

`datasets/2wiki-dev-200.json` is not committed and has no download script of its own: `kgbench real --fetch N` writes it.
The code is `bench/kgbench/real.py` (`fetch`), which pages through the Hugging Face datasets server:

```
https://datasets-server.huggingface.co/rows?dataset=framolfese/2WikiMultihopQA&config=default&split=validation&offset=<n>&length=<=100>
```

It takes the first 200 rows of the `validation` split and rewrites them into the 2Wiki shape (`_id`, `type`, `question`,
`answer`, `supporting_facts`, `context`). The command from `docs/BENCH.md`:

```bash
uv run --extra embed kgbench real --data datasets/2wiki-dev-200.json --fetch 200 --embedder bge
```

That also runs the whole comparison, which needs `make embed-model` and takes a while. To fetch only:

```bash
uv run python -c "from pathlib import Path; from kgbench.real import fetch; fetch(200, Path('datasets/2wiki-dev-200.json'))"
```

Verified on 2026-11-14: running `fetch(200, ...)` fresh produced a file byte-identical to the author's copy
(sha256 `c267a128614f0fa0023cd185ea72c64c5e2d44645a0bfb8bcc52a5ffa00e3c18`). The mirror is a live service, so a later
run could differ; compare the hash. The mirror states no licence, which is why the file is not committed. `--fetch`
overwrites an existing file. The original 2WikiMultihopQA is at `https://github.com/Alab-NII/2wikimultihop`.
The bundled fixture `bench/kgbench/data/wiki2_mini.json` (4 KB) is committed and is what `make benchmark` and the nightly use.

## Committed caches (recorded LLM answers)

These are committed so a clone replays the bench and one test with no endpoint and no key. All came from Claude
Haiku answering through semula (a recording OpenAI-compatible server; see "The semula workflow" in `docs/HANDOFF.md`), so
there was no provider spend. The Haiku version was not recorded.

| Directory | Files | Model key | What it replays | Re-record |
|---|---|---|---|---|
| `bench/kgbench/data/llm-cache-hard-tiny/` | 128 | `openai/claude-haiku` | coref and kg-build for `datagen/examples/hard-tiny` | `kgbench hard --dir datagen/examples/hard-tiny --llm-cache bench/kgbench/data/llm-cache-hard-tiny --llm-base-url http://127.0.0.1:<port>/v1` against semula in record mode |
| `bench/kgbench/data/llm-cache-hard-small/` | 288 | `openai/claude-haiku` | coref and kg-build for `hard-small`, with spaCy `en_core_web_sm` as NER (the prompts embed its mentions) | the command in `docs/results/hard-small-pipelines.md`, plus `--llm-base-url` |
| `tests/model/four_docs/llm-cache/` | 8 | `openai/semula` | the four-document replay, with `en_core_web_trf` | `docs/TESTING.md` ("To re-record"); `tests/model/four_docs/` also holds the documents and questions |
| `bench/kgbench/data/{critique-cache-hard-tiny, faithfulness-cache-hard-small, reader-cache-hard-small, judge-cache, judge-cache-hard, judge-scaled-cache, judge-writer-cache-hard}/` | 180, 175, 175, 69, 99, 168, 102 | `claude-haiku` | the judge, critique, faithfulness and reader runs in `docs/results/` | the `kgbench` subcommand named in each result file's header (`critique`, `faithfulness`, `reader`, `judge`, `judge-scaled`) with the bench's `--base-url` flag pointed at semula |
| `datagen/examples/hard-small/cache/`, `hard-tiny/cache/`, `datagen/examples/llm-cache-tiny/` | 145, 26, 22 | `claude-haiku` | the LLM-written documents of those sets | each folder's `README.md` (`kggen docs generate --renderer llm --model M --cache-dir DIR --base-url URL`) |

`git add -f` is needed only under `bench/kgbench/data/` (the `data/` rule in `.gitignore` covers it; the fixtures there were
force-added). The `tests/model/four_docs/` and `datagen/examples/` caches are not ignored, so a plain `git add` takes them.
(The older note that all of them need `-f` is wrong.) A re-recorded cache changes its files, so read the new
answers before committing: they are a model's.

## Large or generated files that are committed

| File | Size | What it is | Regenerate |
|---|---|---|---|
| `datagen/fixtures/medium/documents.json`, `world.json`; `small/*` | 1.1 MB, 480 KB; 136 KB, 64 KB | seeded generator output | `make fixtures`; a test fails if they drift |
| `datagen/examples/hard-small/` (`documents.json` 224 KB, `questions.json` 228 KB, `world.json` 72 KB, `cache/`) | 1.2 MB | an LLM-written set | its `README.md` |
| `docs/openapi.json` | 72 KB | the API's OpenAPI document; the UI client is generated from it | `make openapi`; a fast test fails when stale |
| `docs/results/*` | 456 KB | bench output (markdown, JSON, SVG charts); each file's first lines give its command | the command in the file |
| `docs/assets/kggen-ui.png` | 192 KB | a screenshot for the README | by hand |
| `site/index.html` | 68 KB | the GitHub Pages walkthrough, hand-written and self-contained; `scripts/check_landing_page.py` checks it, `pages.yml` publishes it | by hand |
| `uv.lock` (848 KB), `ui/package-lock.json` (72 KB) | | lockfiles | `uv lock`, `npm install` |

Nothing else in the tracked tree is large (the biggest tracked file is 1.1 MB).

## Variables

Every field of `Settings` in `graph_rag/config.py`; the environment variable is the name upper-cased, and `.env` is
read too. Defaults are the code's own (generated from it on 2026-11-14). Compose overrides some of them for its
network (hostnames, `INGEST_MODE=publish`, `ALLOW_OPEN_BIND=true`). Read outside `Settings`: `OPENAI_API_BASE` (LiteLLM
and `graph_rag/llm_config.py`), `HOST` and `PORT` (`graph-rag-offline`, default `127.0.0.1:8000`), `GRAPH_RAG_URL` and
`GRAPH_RAG_API_KEY` (the CLI and MCP server), `OTEL_SDK_DISABLED` (set to true when no OTLP endpoint is configured),
`INGEST_PARTITIONS` and `VALKEY_PASSWORD` (compose only), `API_URL` (the UI dev proxy).

| Variable | Default | Meaning |
|---|---|---|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO address (S3 API) |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO user (secret) |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO password (secret) |
| `MINIO_SECURE` | `False` | Use TLS to MinIO |
| `MINIO_BUCKET` | `documents` | Bucket for raw uploads |
| `MAX_REMEMBER_BYTES` | `1000000` | Largest `/v1/remember` text, bytes (413 above) |
| `MAX_UPLOAD_BYTES` | `10000000` | Largest `/ingest` upload, bytes (413 above) |
| `RATE_LIMIT_PER_MINUTE` | `600` | Requests per key per minute with auth on (429 past it); 0 turns off |
| `FAILED_KEY_LIMIT_PER_MINUTE` | `20` | Failed requests per address per minute with auth on; 0 turns off |
| `REQUIRE_AUTH` | `False` | Every endpoint but `/health` needs a bearer key; the key names the tenant |
| `BIND_HOST` | `0.0.0.0` | Address the service listens on |
| `ALLOW_OPEN_BIND` | `False` | Allow a non-loopback bind with auth off (compose sets it) |
| `ALLOW_DEFAULT_CREDENTIALS` | `False` | With auth on, allow the local-dev store passwords anyway (staging) |
| `API_KEYS_PATH` | `.data/api-keys.json` | File holding hashed API keys (`graph-rag-keys`) |
| `MAX_KEYS_PER_TENANT` | `10` | Active keys one tenant may hold |
| `KEY_CREATE_LIMIT_PER_MINUTE` | `5` | Keys one tenant may create per minute |
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200` | Elasticsearch address |
| `ELASTICSEARCH_USER` | `elastic` | Elasticsearch user, used when a password is set |
| `ELASTICSEARCH_PASSWORD` | `""` | Elasticsearch password (secret); empty means open local dev |
| `DOCUMENTS_INDEX` | `documents` | Documents index name |
| `ENTITIES_INDEX` | `entities` | Entities index name |
| `TRIGGER_BACKEND` | `redis` | `redis` (Valkey or Redis Streams) or `kafka` |
| `REDIS_URL` | `redis://valkey:6379/0` | Valkey / Redis address |
| `REDIS_PASSWORD` | `""` | Valkey password (secret); the hosted file requires one |
| `REDIS_STREAM` | `ingest-triggers` | Stream name |
| `REDIS_GROUP` | `graph-rag-ingest` | Consumer group |
| `REDIS_RECLAIM_IDLE_SECONDS` | `900` | A trigger a dead worker held is reclaimed after this long |
| `REDIS_LOCK_SECONDS` | `300` | Per-document lock lifetime; keep below the reclaim time |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka address (kafka backend only) |
| `INGEST_TRIGGER_TOPIC` | `ingest-triggers` | Kafka topic (kafka backend only) |
| `INGEST_MODE` | `inline` | `inline`: the API consumes; `publish`: the API only publishes and `graph-rag-worker` consumes (job status then in Elasticsearch) |
| `JOBS_INDEX` | `jobs` | Index for job status in publish mode |
| `MERGES_INDEX` | `entity-merges` | Index for recorded entity merges |
| `WORKER_HEALTH_PORT` | `8001` | The worker's `/healthz` and `/metrics` port |
| `NER_MODEL` | `en_core_web_trf` | spaCy model; falls back to `en_core_web_lg`, then `en_core_web_sm` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model; pins the vector mapping |
| `EMBED_DIM` | `384` | Its dimension (384 for bge-small) |
| `EL_THRESHOLD` | `0.82` | Cosine merge threshold in entity linking |
| `EL_KNN_TOP_K` | `5` | Extra nearest-vector candidates per entity |
| `EL_TIEBREAKER_ENABLED` | `False` | LLM tie-breaker for near-threshold links (off keeps linking $0) |
| `EL_NIL_ENABLED` | `False` | Keep very-low-confidence entities as NIL |
| `COREF_MODEL` | `gpt-4o-mini` | LiteLLM model string for coreference |
| `KG_BUILD_MODEL` | `gpt-4o-mini` | LiteLLM model string for fact extraction |
| `SYNTHESIS_MODEL` | `gpt-4o` | Model for `synthesize=true` prose answers only |
| `LLM_CACHE_DIR` | `.cache/llm` | Response cache directory (per tenant below it) |
| `LLM_MAX_RETRIES` | `2` | Structured-output retries |
| `OPENAI_API_KEY` | unset | Provider key (secret); ingest needs it or `OPENAI_API_BASE` |
| `REQUIRE_TENANT_LLM_KEY` | `False` | A named tenant must bring its own LLM key |
| `TENANT_LLM_KEYS_PATH` | `.data/tenant-llm-keys.json` | File of encrypted tenant keys |
| `TENANT_LLM_KEY_SECRET` | `""` | Fernet key that encrypts them (secret) |
| `METER_USAGE` | `False` | Count documents, queries and LLM tokens per tenant |
| `USAGE_PATH` | `.data/usage.json` | The usage file |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j bolt address |
| `NEO4J_USER` | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | `devpassword` | Neo4j password (secret; local-dev default) |
| `KHOP_DEPTH` | `2` | Graph expansion depth at query time |
| `SEED_TOP_K_ENTITIES` | `5` | Entity seeds per query |
| `SEED_TOP_K_SENTENCES` | `5` | Sentence seeds per query |
| `RANK_WEIGHT_SEED` | `0.7` | Ranking weight on seed similarity |
| `RANK_WEIGHT_PROXIMITY` | `0.3` | Ranking weight on hop proximity |
| `UI_DIR` | `/app/ui-dist` | Built web UI to serve at `/`; empty or no `index.html` means API only |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `METRICS_ENABLED` | `False` | Serve Prometheus `/metrics` |
| `METRICS_TOKEN` | `""` | Bearer token for `/metrics` with auth on (secret) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `""` | OTLP/HTTP collector for traces; empty means none |

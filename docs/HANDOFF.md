# Handoff: starting on a fresh machine

For a person or a coding agent picking this repo up with nothing else to go on. Written 2026-11-14 against `main`.
Each claim was checked against the code, the Makefile, `pyproject.toml`, the compose files or CI, and the cheap
commands were run. Anything marked "not tested" or "inferred" was not run here. If this and the code disagree, the code wins:
fix this file in the same PR.

## What this is

A Graph RAG service: documents go in, a pipeline (`read → NER → coreference → entity linking → knowledge-graph build`)
writes a graph into Neo4j, and `/query` or `/v1/recall` answers multi-hop questions from it, by default with no LLM
call. It is being turned into a memory store for agents (HTTP `/v1`, an MCP server, a CLI, a Svelte web UI). A standalone
synthetic data generator (`datagen/`, `kggen`) and an evaluation bench (`bench/`, `kgbench`) measure it against gold the
pipeline never produces.

Read in this order: `AGENTS.md` (the rules), `agent_docs/current-state.md` (what exists), this file, `docs/BENCH.md`
(what the numbers say), `graph_rag/ports.py` (before touching a port), then the ADRs you need (`docs/adr/README.md`;
0014 to 0017 are the recent ones).

## Prerequisites and setup

You need Python 3.12 (`.python-version`), [uv](https://docs.astral.sh/uv/) (0.9.16 here), Node 22 or later for `ui/` (CI and the
Dockerfile use 24; 24.19 here), and Docker with Compose for the contract suite, the stack and semula. The `gh` CLI is used for
PRs.

```bash
git clone https://github.com/leejianrong/graph-rag-demo && cd graph-rag-demo
git config user.email   # must be 85485699+leejianrong@users.noreply.github.com, or pushes are rejected
ln -sf ../../scripts/pre-push .git/hooks/pre-push          # ruff, mypy and the fast suite before each push

uv sync --frozen --extra dev                               # enough for the fast suite, the bench offline and the offline server
make ui-install                                            # npm ci in ui/ (only if you work on the web UI)
```

Extras (`pyproject.toml`): `dev` (pytest, ruff, mypy, testcontainers, fakeredis, the `mcp` package, and the `kggen` and `kgbench`
workspace members), `embed` (sentence-transformers and torch, for the bge embedder), `trf` (spacy-transformers and torch, for
`en_core_web_trf`), `cli` (httpx, for `graphrag`), `mcp` (httpx and mcp, for `graph-rag-mcp`), `otel` (the trace exporter; the image installs
it). `ui` is not a root extra: it belongs to `kggen` (indah) and `make ui` supplies it. Combine as needed:

```bash
uv sync --frozen --extra dev --extra embed --extra trf --extra cli --extra mcp
```

A plain `uv sync` is exact: it removes extras you did not name and the spaCy model packages (they are not in `uv.lock`). After
one, re-run `make models`. `uv run` does not prune.

Models (details and sizes in `docs/DATA-PROVENANCE.md`):

| Command | Installs | Used by |
|---|---|---|
| `make models` | `en_core_web_sm` (15 MB) | `pytest -m model` NER tests, the bench's `--spacy en_core_web_sm` pipelines (hard-small), the image's fallback |
| `make models-trf` | `en_core_web_trf` 3.8.0 (about 400 MB, needs `--extra trf`) | the default `NER_MODEL` of the real stack, the four-document replay test, the Docker image |
| `make embed-model` | `BAAI/bge-small-en-v1.5` (129 MB, needs `--extra embed`) | the real embedder (entity linking, query seeds), the bench's `--embedder bge`, `pytest -m model` embedder tests |

The offline server, the fast suite and the offline demo need none of them: they use a deterministic hashing embedder and
rule-based NER (`graph_rag/wiring.py`).

## Running the tests

| Suite | Command | Needs | Notes |
|---|---|---|---|
| Fast (the gate) | `make test`, or `uv run pytest -m "not contract and not model and not llm and not benchmark" -q` | `--extra dev` | Fakes only. 1203 passed and 1 skipped in 82 s on 2026-11-14. Includes `bench/tests`, `datagen/tests`, the quality gate (`kgbench gate`) and the check that `docs/openapi.json` is current. |
| Lint, types | `uv run ruff check . && uv run ruff format --check . && uv run mypy` | `--extra dev` | `make check` runs these and the fast suite. |
| Contract | `make contract` | Docker | testcontainers start MinIO, Elasticsearch, Neo4j, Valkey and Kafka. Proves each real adapter behaves like its fake. A required CI check. Not run for this handoff. |
| Model | `make models && uv run pytest -m model` | `--extra embed`; `--extra trf` and `make models-trf` for the four-document replay | Tests skip cleanly when a model is missing. I ran `tests/model/test_spacy_ner.py` (5 passed); the rest run in CI. |
| LLM | `uv run pytest -m llm` | `OPENAI_API_KEY` | Skips without a key. |
| Benchmark smoke | `make test-benchmark` | the real stack and a key | Skips without them. |
| pytest `e2e` marker | part of the fast suite (`tests/e2e/`) | nothing | In-process runs through the port seam. Not the browser test. |
| Web UI unit | `cd ui && npm test` | `make ui-install` | vitest, 90 passed here. |
| Web UI browser | `make ui-e2e` | `npx playwright install chromium` once | Builds, then Playwright against `graph-rag-offline`. Not run here; CI job `e2e` runs it. |
| Landing page | `make check-site` | nothing | Parses `site/index.html` and checks its anchors. There is no link checker for the docs. |

## Running the system

Offline, no Docker, model or key (rule-based extraction, in-memory, forgets everything on exit):

```bash
uv sync --frozen --extra dev
uv run graph-rag-offline                       # http://127.0.0.1:8000 ; HOST and PORT change that
make demo-offline                              # the bundled supply-chain demo, in process (run here, it works)
UI_DIR=ui/dist uv run graph-rag-offline        # also serves the web UI, after `make ui-build`
```

Full stack: `cp .env.example .env`, set `OPENAI_API_KEY` (or `OPENAI_API_BASE`; see semula below), then `make up`
(`docker compose up --build`). It starts Elasticsearch, Neo4j, MinIO, Valkey (the ingest queue), the API on :8000 and a `worker`;
the API serves the UI at `/`. Scale workers with `docker compose up -d --scale worker=3`. Kafka instead of Valkey:
`TRIGGER_BACKEND=kafka docker compose --profile kafka up`. `make demo-live` brings it up and runs the demo; `make demo` runs the
demo against a running stack; `make down` stops it. The first build is slow and the image is 7 GB (torch, trf, bge). Ingest is
refused with 503 `llm_not_configured` until an LLM is configured; reading never needs one. Hosted deployment is a different
file (`docker-compose.hosted.yml`, `docs/THREAT-MODEL.md`); observability is `docker-compose.observability.yml`.

Elasticsearch has no volume in `docker-compose.yml`, so `docker compose down` loses its documents and entities while Neo4j and MinIO
keep theirs.

Generator and bench (no Docker or key): `make ui` (generator UI), `make world PRESET=small`, `uv run kggen --help`, `uv run kgbench --help`,
`make benchmark`, `make gate`. `make compare` and `make sweep` need `--extra embed` and `make embed-model`.

## CPU and GPU

Nothing here needs a GPU, and no code in `graph_rag/` mentions one. What could use one:

- **The bge embedder** (`graph_rag/adapters/embedder.py`) calls `SentenceTransformer(name)` and `model.encode(...)` with no device
  argument, so sentence-transformers picks CUDA when torch can see one. Inferred from the code, not tested.
- **The transformer NER** (`graph_rag/stages/ner.py`) calls `spacy.load(name)` and never `spacy.prefer_gpu()` or
  `require_gpu()`, so `en_core_web_trf` runs on CPU even on a GPU machine. Using the GPU would need a code change and
  spaCy's CUDA support (cupy), which is not in the dependencies. Not tested.
- Everything else (the rule-based stages, Elasticsearch, Neo4j, the ranking, the LLM calls) is CPU or remote.

CUDA torch with uv, not tested here (this machine has no GPU and no `nvidia-smi`). `uv.lock` pins torch 2.15.0 from PyPI with
the `nvidia-*` CUDA 13 libraries as dependencies (nvidia-cuda-runtime 13.2.86, cudnn-cu13 9.26), so `uv sync --extra embed` on
Linux x86_64 should already give a CUDA 13 build if the driver supports it. Check it first:

```bash
uv sync --frozen --extra dev --extra embed
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False` on a machine with a working driver, or the driver is too old for CUDA 13, point torch at PyTorch's index for
your CUDA version, using uv's documented pattern (add to `pyproject.toml`, re-lock, then sync; pick the index that matches
your driver, `cu126` is only an example):

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

Do not commit a changed `uv.lock` for one machine's GPU. Card GRA-177 is the job of verifying all this on the new machine and
recording per-document time next to the CPU figure; correct this section when it is done. CPU figures to compare against
(`docs/results/ingest-cost.md`): 0.56 s per document wall time, 2.29 s CPU time, with spaCy `en_core_web_sm`, bge, Elasticsearch
and Neo4j and the LLM answers replayed (so no provider latency), on a WSL2 machine with no GPU. `make compare` took about 8
minutes on a 4 vCPU cloud machine; the default `make sweep` takes over an hour; `kgbench split` took 28 minutes.

## The roadmap: pandan board 9

The roadmap is on a [pandan](https://simple-kanban-jian.fly.dev) board, not in the docs: board 9, key `GRA`. Refer to cards by ref
(`GRA-170`). The `pandan` CLI is installed at `~/.local/bin/pandan` here (the repo does not install it; log in with
`pandan login`, see `pandan login --help`), and the `pandan` skill describes the workflow.

```bash
pandan list --board 9 --column todo        # open cards
pandan get GRA-170                         # a card's why, do and done-when
pandan list --board 9 --column done --limit 10
pandan board use 9                         # pin the board
pandan move GRA-170 in_progress            # then one branch and one PR per card
```

On 2026-11-14 the board had 15 open cards and none in progress: GRA-15 (real-stack 2Wiki medium run), GRA-18 (publish benchmark results),
GRA-40 (Python SDK from the OpenAPI file), GRA-57 (staging), GRA-66 (interview agent builders), GRA-158 (one name under two entity
types), GRA-169 (build the Postgres backend of ADR-0017), GRA-170 (beat BM25 and dense on hard-small), GRA-171 (model when a fact was true),
GRA-172 (Acme Logistics versus Northwind Logistics), GRA-173 (pin GitHub Actions to SHAs), GRA-174 (first nightly run), GRA-175
(coref schema for singletons), GRA-176 (reprice with the measured ingest cost), GRA-177 (benchmark and stack on a GPU machine).
Read `GRA-170`, `GRA-15` and `GRA-177` first. GRA-174 is out of date: the nightly has already run (below).

## Where things stand

**Done.** The pipeline V1 to V8; the `/v1` API with server-sent job events at `GET /jobs/{id}/events`; tenancy and API keys
(ADR-0014, ADR-0015; off by default); a Valkey stream as the default ingest queue with separate workers (ADR-0016, Kafka optional);
CLI, MCP server, and the Svelte UI served by the API with a strict CSP and bundled fonts; entity merge and split; the entity-linking
name guard (GRA-157); answer selection and `answer_entities` (GRA-161, GRA-165); the `ACQUIRED` predicate and the `role` edge property
(GRA-162); job counts null until the job is done (GRA-150); 422 on empty text (GRA-164, GRA-167); `docs/openapi.json` generated
by `make openapi`; the four-document replay with `en_core_web_trf` in the CI `model` job (GRA-163, GRA-168); `kgbench ingest-cost`
(GRA-55); ADR-0017 with a Postgres prototype in `spikes/pg_stores/` (GRA-54); research notes in `docs/research/`.

**Open.** The items on the board above. Also: no SDK, no staging environment, nothing migrated to Postgres.

**Known quality gaps.**

- Retrieval does not beat BM25 or dense. On `hard-small` whole-path@5 is 0.24 for BM25 and dense and 0.21 for graph-rag
  (`docs/results/hard-small-pipelines.md`); on a 200-question 2Wiki slice dense scores 26%, graph-rag 23 to 24% and BM25 16%
  (`docs/results/real-2wiki-200.md`). At 3 and 4 hops nothing recovers whole paths. GRA-170.
- Real-LLM quality at scale is unmeasured. The largest run with real coref and graph build is 145 short documents through Haiku;
  the 2Wiki run uses the offline heuristics. GRA-15.
- Time is only partly modelled: a newer dated document ends an older fact for single-valued predicates, but a job change in undated
  text leaves two employers. GRA-171.
- "Acme Logistics" and "Northwind Logistics" would still merge: the GRA-157 guard (`_names_compatible` in
  `graph_rag/stages/entity_linking.py`) lets a cosine merge through when the two names share any distinctive token, and "Logistics"
  is one. This comes from card GRA-172 and a read of the code; I did not run it. The four-document test uses "Northwind Shipping", so
  it keeps that pair apart and does not cover this one. GRA-172.
- The nightly benchmark has run on schedule twice (2026-11-12 and 2026-11-13, both green), but its real-provider step was
  skipped because the repo has no `OPENAI_API_KEY` secret, and its baseline is five questions. The earlier belief that it had
  never run is wrong; GRA-174 is stale.
- Offline graph-rag is a floor: heuristic extraction emits only `RELATED_TO`.

## Hard rules and CI gates

From `AGENTS.md`, restated because they bite.

- Commit with `85485699+leejianrong@users.noreply.github.com`. No attribution lines in commits or PR text.
- `main` is protected and PR-only, for admins too. Branch off fresh `main`, one branch per card, stage by explicit path (not
  `git add -A`) while a background job writes into the repo. Never `--admin`-merge.
- Required checks (read from the branch protection on 2026-11-14): `lint`, `fast`, `contract`, `secret-scan`, `image`. Squash-merge
  with `gh pr merge --squash --delete-branch` only when all five are COMPLETED and green on the exact pushed SHA, never on stale checks.
- Also run, not required: `ui`, `e2e`, `landing-page`, `model`, `llm`, `benchmark`, and `dep-audit` (advisory: it is
  `continue-on-error`, but keep it green). The nightly workflow is separate.
- Never let a slow check gate a local push; the pre-push hook is the cheap mirror of CI only.
- `datagen/` must never import `graph_rag`, `kgbench` or `litellm`; `bench/` may use `graph_rag` only through ports and public entry points;
  gold never comes from graph-rag's own extraction. Tests enforce both.
- Docs change in the same PR as the behaviour they describe. A change to live behaviour ships behind a flag that defaults off.
- `bench/` and `datagen/` are workspace members: keep both `pyproject.toml` `COPY` lines in the Dockerfile.
- Do not touch Docker containers or volumes that are not yours; name your own (the examples use `grsmoke*` and `graph-rag-semula`).

## The semula workflow

semula (`https://github.com/leejianrong/semula`; docs at `https://leejianrong.github.io/semula/`; local checkout
`~/projects/abang-ai/semula`; image `ghcr.io/leejianrong/semula`; board `SEM`) is a mock LLM server that speaks the OpenAI chat API.
In record mode a prompt with no stored answer is parked, and a coding agent (Claude Code or Codex) writes the answer, so
recording costs a subscription and not API spend. Every answer is stored and replayed for free. This repo uses it to record the
LLM stages (coreference, fact extraction) and the bench's judges and writers. The model that answered was Claude Haiku (version
not recorded).

Record. Pick a free port and your own container name:

```bash
docker run -d --name graph-rag-semula -p 127.0.0.1:18083:8080 \
  -e SEMULA_MODE=record -e SEMULA_RECORD_TIMEOUT=900 \
  -v graph-rag-semula-data:/data ghcr.io/leejianrong/semula
```

`SEMULA_MODE=record` makes the default project park unknown prompts; `SEMULA_RECORD_TIMEOUT` is how long a parked call waits (semula's
default is 300 seconds; the four-document re-record in `docs/TESTING.md` uses 900 because a long prompt takes an agent a while). Point
the pipeline at it with `OPENAI_API_BASE=http://127.0.0.1:18083/v1` (and any `OPENAI_API_KEY`; the bench supplies `semula-local`) and the model
strings `openai/claude-haiku` (bench default, `--llm-model`) or `openai/semula` (the four-document test). The bench commands take
`--llm-base-url` (`kgbench hard`, `split`) or `--base-url` (`judge`, `critique`, `faithfulness`, `reader`, `judge-scaled`); omit it to
replay.

Answer. The agent connects once and then loops: `claude mcp add --transport http semula http://127.0.0.1:18083/mcp/`, then it calls
`wait_for_pending`, reads the request and calls `answer_pending`. Without MCP, the CLI does the same (inside the container, or the
standalone binary with `--url`):

```bash
docker exec graph-rag-semula semula pending --wait 60 --json     # parked prompts
docker exec graph-rag-semula semula show <id>
docker exec graph-rag-semula semula answer <id> --text "..."     # or --file answer.json, or --stdin
```

For coreference and fact extraction the prompt carries a JSON schema and the answer must be the JSON alone, not prose around it
(the client validates it and retries up to `LLM_MAX_RETRIES`, and never caches an invalid reply). semula's own agent playbook
(`docs/agents/PLAYBOOK.md` in its repo; a Claude Code skill is in `docs/agents/claude-code-skill/`) covers answer formats.

Replay and clean up. Run the same command again with no endpoint set: every answer now comes from the repo's cache and the run
asserts zero provider calls. Read the new answers before committing, since they are a model's. Then
`docker rm -f graph-rag-semula && docker volume rm graph-rag-semula-data`.

Two caches, and why a model change invalidates them. semula stores the answers it was given; its README says a stored answer comes
back even with a different model name or a new timestamp in the prompt. The pipeline's own client also caches, on disk, under
`.cache/llm/` (`LLM_CACHE_DIR`), keyed `sha256(model + prompt + params)` (`graph_rag/adapters/llm_client.py`). That key changes with the
model string, so `COREF_MODEL` or `KG_BUILD_MODEL` changing misses every entry. It also changes with any prompt edit (a coref or
kg-build prompt, the predicate list) and with the NER model, because the coref prompt embeds the mentions NER found. The cache is
per tenant: the default tenant uses the root, any other uses `.cache/llm/tenants/<tenant>/`, so identical text in two tenants is two
calls (GRA-139). Consequences: a client-cache hit never reaches semula, so to see traffic in semula, or to replay for real, clear
the cache first (`rm -rf .cache/llm`); and the committed caches only replay with the NER model, embedder, model string and prompts
they were recorded with (`docs/DATA-PROVENANCE.md` lists each). When you change a prompt on purpose, expect the committed caches
for it to miss, and re-record them.

## Things that are not in git

Short version; the full table (with commands, sizes and what was verified) is `docs/DATA-PROVENANCE.md`.

- `datasets/2wiki-dev-200.json`: rebuilt by `kgbench real --fetch 200` from the Hugging Face mirror `framolfese/2WikiMultihopQA`; a fresh fetch was
  byte-identical to the author's copy on 2026-11-14.
- `.env`: copy `.env.example`. `.data/` (API keys, usage, tenant keys) is created at run time and is not gitignored: stage by path.
- `.cache/llm/`: the response cache; rebuilt by running the pipeline; delete to force real calls.
- Models: `make models`, `make models-trf`, `make embed-model`. `.venv/`: `uv sync`. `ui/node_modules`, `ui/dist`: `make ui-install`, `make ui-build`.
- `.mcp.json`: not needed; `.mcp.json.example` is a stale leftover from another project.
- The indah checkout for `make ui` is optional now (indah 0.13.0 is on PyPI).
- Docker images and volumes; semula's data volume.
- Committed but easy to miss: recorded LLM caches under `bench/kgbench/data/` (these need `git add -f`, `data/` is ignored),
  `tests/model/four_docs/llm-cache/`, `datagen/examples/*/cache/`.

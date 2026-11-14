# Ingest cost per 1,000 documents

`uv run --extra embed kgbench ingest-cost --dir datagen/examples/hard-small --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm --embedder bge --price claude-haiku-4.5=1,5 claude-haiku-5.5=0.10,0.50 --stack 25 50 100 145`

145 documents, 43 words each on average. Scaled linearly to 1,000.

## LLM tokens and dollars

Tokens are counted from the prompts the two LLM stages build and the answers recorded for them, with OpenAI's o200k_base tokenizer (an approximation for other models). Dollars are tokens times the price shown, not billed.

| stage | requests | prompt tokens | answer tokens |
|---|---|---|---|
| coref | 1,000 | 538,531 | 59,469 |
| kg_build | 986 | 1,033,697 | 95,152 |

The documents' own text is 55,483 tokens per 1,000, 4% of the prompt tokens (it goes into both stages' prompts); the rest is the instructions and entity list each request carries. These documents are short, so a longer one costs more than shown.

Per 1,000 documents, uncached (every request goes to the provider):

| model | $ per million in / out | $ per 1,000 documents |
|---|---|---|
| claude-haiku-4.5 (list price) | 1.0 / 5.0 | $2.35 |
| claude-haiku-5.5 (list price) | 0.1 / 0.5 | $0.23 |

Cached: $0, because a hit never reaches the provider. This replay was served entirely from the cache (zero provider calls), which is what a re-run of identical prompts gets.

## Time per document, everything except the LLM

Real spaCy, bge, Elasticsearch and Neo4j; the LLM answers replayed from the cache, so this leaves out the provider's latency (not measured). One worker, one document at a time.

Wall time per document: mean 0.56s, median 0.50s, 95th percentile 0.91s; 1,000 documents take about 9.4 minutes.

CPU time of the ingesting process, summed over its threads (spaCy, bge, linking; the stores' own work is in other processes): mean 2.29s, median 2.23s per document, so it exceeds the wall time.

The host's one-minute load average was 5 before the run and 6 after. It is a shared machine, so read the wall times as one noisy measurement.

## Storage

Elasticsearch is merged to one segment per index and flushed before each reading. The slope is a least-squares line through every checkpoint, so it leaves out each store's fixed size; a store whose size stops growing (entities are shared across documents) has no meaningful slope.

| documents | raw text | elasticsearch documents | elasticsearch entities | neo4j nodes | neo4j edges |
|---|---|---|---|---|---|
| 25 | 7 KB | 0.93 MB | 0.59 MB | 62 | 44 |
| 50 | 14 KB | 1.89 MB | 1.18 MB | 110 | 99 |
| 100 | 27 KB | 3.72 MB | 1.87 MB | 159 | 205 |
| 145 | 38 KB | 4.99 MB | 2.16 MB | 210 | 263 |
| per 1,000 (slope) | 0.26 MB | 34.05 MB | 12.83 MB |  |  |

## GPU versus CPU, per component (GRA-177, 2026-11-14)

Not the stack run above. Each component timed alone on the 145 `hard-small` documents, one document at a time, after a 5-document
warm-up, on a WSL2 machine with an RTX 3080 Ti Laptop GPU (16 GB), torch 2.11.0+cu128, spaCy 3.8.18, load average about 2. The
locked torch 2.15.0+cu132 crashes on this machine, so these numbers are from the cu128 build; see `docs/HANDOFF.md`. The CPU
rows are this machine's CPU, not the machine behind the 0.56 s figure.

| component | CPU mean (median, p95) | GPU mean (median, p95) | speed-up |
|---|---|---|---|
| bge-small-en-v1.5, one document per call | 21.8 ms (17.3, 33.3) | 9.4 ms (8.7, 12.8) | 2.3x |
| bge-small-en-v1.5, 32 documents per call | 11.3 ms | 1.0 ms | 11x |
| `en_core_web_trf` | 49.0 ms (45.6, 69.3) | 14.0 ms (12.9, 16.2) | 3.5x |
| `en_core_web_sm` (the model behind the stack figure) | 7.0 ms (7.0, 9.4) | not run | |

Read it as follows. The pipeline embeds one document at a time, so the one-document row is the one that applies; batching is a
possible change, not what runs today. The GPU runs of `en_core_web_trf` used cupy through `spacy.require_gpu()`, which the product
code does not call, so the product is still CPU for NER. The stack's 0.56 s per document is mostly not these components (spaCy
`sm` plus bge is about 29 ms of CPU here), so a GPU would not move that figure much; it matters for `en_core_web_trf`, the
product default, and for large batches. No end-to-end GPU ingest was run, and the LLM time is not in any of this.

## What is measured and what is computed

| figure | status |
|---|---|
| Tokens per stage | Computed: the exact prompts and the recorded answers, counted with tiktoken's `o200k_base`. Anthropic's tokenizer differs, so treat the counts as an estimate. |
| Dollars | Computed, not billed: tokens times the list price. No provider was called. Prices are Anthropic's list prices from https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-14 (base input / output per million tokens: Haiku 4.5 $1 / $5; Haiku 5.5 $0.10 / $0.50 for prompts up to 100,000 tokens), with no batch discount and no prompt caching. The recorded answers came from a Haiku agent behind semula and the version is not recorded, so both rows are shown. |
| Wall time per document | Measured, on the real stack (spaCy `en_core_web_sm`, bge-small-en-v1.5, Elasticsearch 8.13.4, Neo4j 5.20, MinIO) in Docker on one CPU-only WSL2 machine, one document at a time. It is everything except the LLM: a replay has no provider latency, which is the largest part of a live run and was not measured. Two runs from empty stores gave a mean of 0.57 s and 0.56 s. |
| Storage | Measured: Elasticsearch segment bytes after a force-merge, and Neo4j node and edge counts. Neo4j's own file size was not measured. |
| Scaling to 1,000 | Linear extrapolation from 145 documents. |

## Caveats

- **Short documents.** Each is about 43 words. A longer one adds its text to both prompts and more entities to the entity list, so tokens grow faster than the document does. The documents' text is only 4% of the prompt tokens; the rest is instructions and the entity list each request carries.
- **One corpus, one world.** `datagen/examples/hard-small`, 145 LLM-written documents, 180 entities. Entities are shared across documents, so the entity index and the graph grow more slowly than the document count; their slopes are not a per-1,000 figure to rely on. The Elasticsearch documents index (dense vectors and text) is the steady one.
- **The cache must match the run.** The cache key is the prompt, so the replay only works with the same spaCy model and embedder it was recorded with (`--spacy en_core_web_sm --embedder bge`). A mismatch drops documents and the command fails rather than reporting.
- **Ingest only.** The default `/query` path has no LLM, so it spends no tokens.

## Re-run

Needs the `embed` extra, `make models` and three containers (Docker compose project name is arbitrary):

```bash
docker compose -p gra55cost up -d --wait elasticsearch minio neo4j
export ELASTICSEARCH_URL=http://localhost:9200 MINIO_ENDPOINT=localhost:9000 NEO4J_URI=bolt://localhost:7687
uv run --extra embed kgbench ingest-cost --llm-cache bench/kgbench/data/llm-cache-hard-small \
  --spacy en_core_web_sm --embedder bge --price claude-haiku-4.5=1,5 claude-haiku-5.5=0.10,0.50 \
  --stack 25 50 100 145 --out /tmp/ingest-cost.md
docker compose -p gra55cost down -v
```

The tables above are that output; the sections after them are written by hand, so `--out` here goes to a scratch file. Leave out `--stack` for the tokens and dollars only (about 30 seconds, no containers). Start from empty volumes for each stack run.

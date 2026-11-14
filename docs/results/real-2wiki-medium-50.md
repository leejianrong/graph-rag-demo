# Real stack on 2WikiMultihopQA: the first 50 medium questions (GRA-15)

Run on 2026-11-14 and 2026-11-15. This is a 50-question slice of the `medium` subset (200 questions), not the whole subset; the reason is under "Scope". Raw scores are in `real-2wiki-medium-50.json`.

## Setup

- Questions: the first 50 of the `medium` subset (a sorted-by-id prefix) of `datasets/2wiki_dev.json`, which is 56 MB and has the dev split's full rows. This is not the `2wiki-dev-200.json` file that `docs/DATA-PROVENANCE.md` describes, and the source of this copy is not recorded in the repo. Its SHA-256 starts `48b9bdc69654dc58`.
- Corpus: 436 documents (one per unique paragraph title across those 50 questions), ingested in the harness's fixed order.
- Pipeline: the real stack's components. spaCy `en_core_web_trf` for NER (on CPU), `BAAI/bge-small-en-v1.5` on the GPU, Elasticsearch, Neo4j and MinIO in Docker, entity linking at the default threshold.
- LLM stages (coreference and graph build): a Claude Haiku agent answering through semula in record mode, model string `openai/claude-haiku`. No hosted API was called; the repo's `.env` key was not used.
- Hardware: WSL2, one RTX 3080 Ti Laptop GPU, torch 2.11.0+cu128 (see `docs/HANDOFF.md` on why not the locked torch).

## Results (50 questions)

Retrieval is scored per question against the gold supporting sentences and averaged. Answer scores are exact match and token F1.

| pipeline | supporting-fact precision | recall | F1 | exact match | token F1 |
|---|---|---|---|---|---|
| offline (heuristic NER, no LLM stages) | 0.060 | 0.135 | 0.082 | 0.00 | 0.01 |
| real stack (trf NER, Haiku coref and graph build) | 0.248 | 0.547 | 0.335 | 0.02 | 0.02 |

Both rows are the same 50 questions and the same 436 documents. The offline row is `kgbench run --subset medium --limit 50` with no `--real`.

- **The LLM stages lift supporting-fact F1 from 0.08 to 0.34.** That is large, but it is one slice of 50 questions from the start of the sorted split, one run, and one recording model, and I did not compute intervals for the retrieval figures.
- **Exact match is 1 of 50** (2%, 95% Wilson interval 0.4% to 10.5%), token F1 0.02. I do not read this as a quality figure for graph-rag. The default path returns an entity from the retrieved subgraph with no answer synthesis. The per-question output shows entity names where the gold is `yes`, `no` or a date, for example `pred='Brazil' gold=['yes']`. I did not check how the answer entity is chosen. The earlier 10-question calibration run scored 0.00 on both.
- **These figures are not comparable with `docs/results/real-2wiki-200.md`.** That report scores whole-path@5 at sentence level over a different file and a fixed top 5. This one scores the harness's supporting-fact precision and recall over what the query returns.

## Cost and time

- **LLM calls:** 859 in total, 100% answered through semula and then cached: 336 coreference and 327 graph-build prompts in this run (663), plus 196 recorded in a 10-question calibration run on the same prefix, replayed here from the cache.
- **Dollars billed: 0.** No provider was called. Tokens were not counted, so there is no list-price estimate; `kgbench ingest-cost` does that for a cache and was not run on this one.
- **Wall time of the recording run:** 7,633 s (2.1 hours), of which 6,598 s was waiting for the answering agent, about 10 s per call. The sequential ingest is what makes it slow: each prompt waits on an agent turn, and a second agent cannot help because entity linking depends on order.
- **Agent tokens** (reported by the harness, a subscription cost and not an API bill): about 669k for the first 196 answers and 375k for the last 663. At that rate the full 200 questions is about 1,450 documents and roughly 2,900 prompts, so about 4 to 5 hours of recording. My earlier estimate of 8.5 hours and 9 to 10M tokens, from the calibration run alone, was too high on tokens.
- **Replay:** 312 s, 859 cache hits, 0 provider calls, scores identical to the recording run.

## Entity linking merges different people

I asked the answering agent to report entity IDs that seemed to stand for more than one real entity. I checked 9 of the IDs it named against Elasticsearch after the run. Six are clear over-merges, where one entity holds names of different people or places:

| entity | name | aliases it absorbed |
|---|---|---|
| `e-9c777467ecb454f1` | Maurice de Prendergast | Maurice, Count of Saxony; de Saxe |
| `e-fadc9350544f3dd1` | Sir Henry Curwen | Nicholas Curwen; Joan Curwen |
| `e-1edb4224157519e8` | Frederick Calvert, 6th Baron Baltimore | Frederick, Hereditary Prince of Baden-Durlach |
| `e-d7fc819dcb09188d` | Rajaraja Chola I | Arinjaya Chola; Sundara Chola |
| `e-0fd2ee116e806849` | St John | three different Barons St John of Bletso |
| `e-8fa56783f1325a82` | Santa Catarina (LOCATION) | four rivers |

Three more (`Maria`, `Anna`, `Mary`) are bare given names shared across documents; `Mary` carries the alias "Mary, Queen of Scots", which may be right.

What this does not show: how often it happens. The agent listed collisions it noticed while answering, so this is a set of examples, not a rate. The merges share a token (a given name or a surname), which is the hole GRA-172 describes for "Acme Logistics" and "Northwind Logistics". I have not confirmed that the GRA-157 name guard is what lets these through. This bears on GRA-170: a merged entity creates paths that do not exist and hides the ones that do.

## Quality of the recorded answers

The answers are a model's and some are wrong. The agent reported these, and I have not read every answer:

- At least seven coreference clusters contain a form that is not in the text (lowercase `he` and `she`, `The film`, `the countess`) or attach `Henry` to William Henry where it only occurs in "Henry's Law". One returned empty clusters although a pronoun refers to the subject, and one omitted its own canonical form from its members.
- A few `HAS_ROLE` edges point at a place or person, where the rule says organisation. Some birthplace and origin edges are `LOCATED_IN` or `RELATED_TO` at confidence 0.5 or 0.6.
- Two earlier answers put non-dates in date fields.

Haiku through an agent loop is not the same as Haiku through the API, and the agent answered each prompt without seeing the others. Treat the retrieval gain above as an upper-bound-style estimate of what an LLM builder adds here, not a measured production figure.

## Scope and what is not done

- **Only 50 of the 200 medium questions.** The card asked for the medium subset. I stopped at 50 because the calibration run put the full set at 8 or more hours of sequential recording, and the owner chose 50 over the full set. The remaining 150 questions need about 3 to 4 more hours and roughly 2,000 more prompts, and the first 50 replay for free as a prefix.
- No second run, seed or embedder, and no intervals on the retrieval figures.
- Not run: graph-rag on the real stack with a hosted model, and any comparison with BM25 or dense on these same 50 questions.

## Repeat it

The recorded answers are committed in `bench/kgbench/data/llm-cache-2wiki-medium50/` (859 files, 3.5 MB). Replay needs the same NER model, embedder, model string and prompts, and empty stores. These start my own services; use your own project name:

```bash
docker compose -p grsmoke --env-file .env.example up -d elasticsearch minio neo4j
export OPENAI_API_BASE=http://127.0.0.1:1/v1 OPENAI_API_KEY=unused   # a dead endpoint: any provider call fails
export COREF_MODEL=openai/claude-haiku KG_BUILD_MODEL=openai/claude-haiku
export LLM_CACHE_DIR=$PWD/bench/kgbench/data/llm-cache-2wiki-medium50
export ELASTICSEARCH_URL=http://localhost:9200 MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=minioadmin \
  MINIO_SECRET_KEY=minioadmin MINIO_SECURE=false NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=devpassword
uv run --extra embed kgbench run --dataset datasets/2wiki_dev.json --subset medium --limit 50 --real --json out.json
```

The cache is keyed by the prompt, so it also holds only if `en_core_web_trf` and bge produce the same mentions and entity order. To record more questions, start semula as described in `docs/HANDOFF.md`, point `OPENAI_API_BASE` at it, raise `--limit`, and have a Haiku agent answer the parked prompts. Run the command from a directory without a `.env`, or set the variables above explicitly: litellm loads a repo `.env` into the environment on import.

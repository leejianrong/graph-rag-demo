# Retrieval comparison: small worlds

Can a graph beat keyword and dense retrieval on multi-hop questions? Every system answers the same generated questions; a question counts as solved at k when every fact on its path is expressed by one of the top 5 sentences.

## Setup

- Preset `small`, world seeds [0, 1, 2], one world per seed (150 entities, 300 facts, 117 documents, 300 sentences in the first).
- 100 questions per seed, equal weight on each hop count; 300 in total.
- Embedder for the dense baseline and graph-rag: bge-small-en-v1.5 (local sentence-transformers).
- Reproduce: `PATH_RETRIEVAL=true uv run --extra embed kgbench compare --preset small --seeds 0 1 2 --questions 100 --embedder bge --k 5 --out docs/results/path-retrieval-small-worlds.md`

## path@5 / recall@5 (mean over seeds)

| hops | BM25 | Dense | graph-rag passages | graph-rag graph | graph-rag path |
|---|---|---|---|---|---|
| all | 0.25 / 0.50 | 0.25 / 0.50 | 0.25 / 0.50 | 0.28 / 0.53 | 0.35 / 0.53 |
| 1 | 0.95 / 0.95 | 0.99 / 0.99 | 0.99 / 0.99 | 0.96 / 0.96 | 0.89 / 0.89 |
| 2 | 0.04 / 0.48 | 0.01 / 0.47 | 0.01 / 0.47 | 0.16 / 0.55 | 0.40 / 0.61 |
| 3 | 0.00 / 0.32 | 0.00 / 0.32 | 0.00 / 0.32 | 0.00 / 0.35 | 0.09 / 0.39 |
| 4 | 0.00 / 0.23 | 0.00 / 0.23 | 0.00 / 0.23 | 0.00 / 0.25 | 0.01 / 0.25 |

## path@5: lowest to highest seed

| hops | BM25 | Dense | graph-rag passages | graph-rag graph | graph-rag path |
|---|---|---|---|---|---|
| all | 0.23 to 0.26 | 0.24 to 0.26 | 0.24 to 0.26 | 0.27 to 0.29 | 0.34 to 0.36 |
| 1 | 0.92 to 1.00 | 0.96 to 1.00 | 0.96 to 1.00 | 0.92 to 1.00 | 0.84 to 0.92 |
| 2 | 0.00 to 0.08 | 0.00 to 0.04 | 0.00 to 0.04 | 0.12 to 0.20 | 0.36 to 0.44 |
| 3 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.08 to 0.12 |
| 4 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.04 |

## Cost and latency

Wall time covers building the index and answering every question, on CPU with no GPU. The systems share one embedding cache, so the times are not a like-for-like speed comparison. LLM calls and dollars are totals over the seeds; none of these systems reaches a provider.

| system | seconds per seed (mean) | LLM calls | LLM dollars |
|---|---|---|---|
| BM25 | 0.1 | 0 | $0.00 |
| Dense | 9.2 | 0 | $0.00 |
| graph-rag passages | 113.3 | 0 | $0.00 |
| graph-rag graph | 2.8 | 0 | $0.00 |
| graph-rag path | 2.6 | 0 | $0.00 |

## Caveats

- The text is rendered from templates: one fact per sentence, made-up names, no paraphrase. That is easier than real prose in some ways (names are unambiguous tokens) and unlike it in others, so public sets stay in the picture.
- graph-rag here is the offline pipeline: heuristic entity detection, merge-by-name linking and a co-occurrence graph whose only relation is RELATED_TO. It is a floor, not the real stack (spaCy, an LLM-built graph), and it was not tuned.
- graph-rag passages are a nearest-neighbour search over sentence vectors, so they behave like the dense baseline by construction; graph-rag graph adds the sentences behind the retrieved subgraph's edges.
- A question is solved at k only if every fact on its path is expressed by one of the top k sentences. Questions come from the generator, never from graph-rag's own extraction.

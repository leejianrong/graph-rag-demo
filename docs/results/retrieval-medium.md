# Retrieval comparison: medium worlds

Can a graph beat keyword and dense retrieval on multi-hop questions? Every system answers the same generated questions; a question counts as solved at k when every fact on its path is expressed by one of the top 5 sentences.

## Setup

- Preset `medium`, world seeds [0, 1, 2], one world per seed (1000 entities, 2500 facts, 817 documents, 2500 sentences in the first).
- 200 questions per seed, equal weight on each hop count; 600 in total.
- Embedder for the dense baseline and graph-rag: bge-small-en-v1.5 (local sentence-transformers).
- Reproduce: `uv run --extra embed kgbench compare --preset medium --seeds 0 1 2 --questions 200 --embedder bge --k 5 --out docs/results/retrieval-medium.md`

## path@5 / recall@5 (mean over seeds)

| hops | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|
| all | 0.25 / 0.50 | 0.24 / 0.50 | 0.24 / 0.50 | 0.26 / 0.50 |
| 1 | 0.99 / 0.99 | 0.98 / 0.98 | 0.98 / 0.98 | 0.94 / 0.94 |
| 2 | 0.01 / 0.48 | 0.00 / 0.48 | 0.00 / 0.48 | 0.11 / 0.50 |
| 3 | 0.00 / 0.30 | 0.00 / 0.30 | 0.00 / 0.30 | 0.00 / 0.31 |
| 4 | 0.00 / 0.23 | 0.00 / 0.23 | 0.00 / 0.23 | 0.00 / 0.24 |

## path@5: lowest to highest seed

| hops | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|
| all | 0.24 to 0.26 | 0.24 to 0.25 | 0.24 to 0.25 | 0.24 to 0.28 |
| 1 | 0.98 to 1.00 | 0.96 to 1.00 | 0.96 to 1.00 | 0.88 to 1.00 |
| 2 | 0.00 to 0.02 | 0.00 to 0.00 | 0.00 to 0.00 | 0.08 to 0.16 |
| 3 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 |
| 4 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 | 0.00 to 0.00 |

## Caveats

- The text is rendered from templates: one fact per sentence, made-up names, no paraphrase. That is easier than real prose in some ways (names are unambiguous tokens) and unlike it in others, so public sets stay in the picture.
- graph-rag here is the offline pipeline: heuristic entity detection, merge-by-name linking and a co-occurrence graph whose only relation is RELATED_TO. It is a floor, not the real stack (spaCy, an LLM-built graph), and it was not tuned.
- graph-rag passages are a nearest-neighbour search over sentence vectors, so they behave like the dense baseline by construction; graph-rag graph adds the sentences behind the retrieved subgraph's edges.
- A question is solved at k only if every fact on its path is expressed by one of the top k sentences. Questions come from the generator, never from graph-rag's own extraction.

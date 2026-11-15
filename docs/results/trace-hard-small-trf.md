# Where multi-hop paths are lost

`uv run --extra embed kgbench trace --dir datagen/examples/hard-small --pipeline llm+spacy --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small-trf --spacy en_core_web_trf`

The built graph has 211 nodes and 324 edges. Counts are questions; a question needs every fact of its path. The graph view is scored at top 5.

| hops | n | every fact has an edge | edges join the right entities | start entity seeded | path in subgraph | graph top k | dense top k | BM25 top k |
|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 75 | 74 | 74 | 72 | 65 | 67 | 67 |
| 2 | 75 | 75 | 74 | 75 | 65 | 5 | 5 | 2 |
| 3 | 75 | 75 | 75 | 73 | 45 | 1 | 0 | 2 |
| 4 | 75 | 75 | 71 | 71 | 22 | 0 | 0 | 0 |
| all | 300 | 300 | 294 | 293 | 204 | 71 | 72 | 71 |

## First stage where a path is lost

| stage | 1 | 2 | 3 | 4 | all |
|---|---|---|---|---|---|
| solved (whole path in the top k) | 65 | 5 | 1 | 0 | 71 |
| 1 extraction or graph build: a path fact has no edge from its sentence | 0 | 0 | 0 | 0 | 0 |
| 2 linking: a path fact's edge joins the wrong entities | 0 | 1 | 0 | 4 | 5 |
| 3 seeding: the start entity is not among the seeds | 0 | 0 | 2 | 2 | 4 |
| 4 traversal: the correct edges are not in the subgraph | 2 | 9 | 29 | 47 | 87 |
| 5 ranking: the path is in the subgraph but not in the top k | 8 | 60 | 43 | 22 | 133 |

## Why a path fact has no edge

Each of the 221 distinct facts on a gold path, counted once.

| what happened to the fact | facts |
|---|---|
| right edge | 210 |
| edges from the sentence, none between the two | 11 |

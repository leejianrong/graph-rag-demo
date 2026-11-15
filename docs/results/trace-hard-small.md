# Where multi-hop paths are lost

`uv run --extra embed kgbench trace --dir datagen/examples/hard-small --pipeline llm+spacy --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm`

The built graph has 210 nodes and 263 edges. Counts are questions; a question needs every fact of its path. The graph view is scored at top 5.

| hops | n | every fact has an edge | edges join the right entities | start entity seeded | path in subgraph | graph top k | dense top k | BM25 top k |
|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 61 | 61 | 70 | 59 | 53 | 67 | 67 |
| 2 | 75 | 45 | 44 | 72 | 37 | 7 | 5 | 2 |
| 3 | 75 | 45 | 45 | 70 | 18 | 1 | 0 | 2 |
| 4 | 75 | 38 | 37 | 66 | 9 | 0 | 0 | 0 |
| all | 300 | 189 | 187 | 278 | 123 | 61 | 72 | 71 |

## First stage where a path is lost

| stage | 1 | 2 | 3 | 4 | all |
|---|---|---|---|---|---|
| solved (whole path in the top k) | 53 | 7 | 1 | 0 | 61 |
| 1 extraction or graph build: a path fact has no edge from its sentence | 13 | 30 | 30 | 37 | 110 |
| 2 linking: a path fact's edge joins the wrong entities | 0 | 1 | 0 | 1 | 2 |
| 3 seeding: the start entity is not among the seeds | 1 | 0 | 1 | 1 | 3 |
| 4 traversal: the correct edges are not in the subgraph | 2 | 7 | 26 | 27 | 62 |
| 5 ranking: the path is in the subgraph but not in the top k | 6 | 30 | 17 | 9 | 62 |

## Why a path fact has no edge

Each of the 221 distinct facts on a gold path, counted once.

| what happened to the fact | facts |
|---|---|
| right edge | 176 |
| no edge: the NER missed an endpoint | 35 |
| no edge: both found, the builder emitted nothing | 7 |
| edges from the sentence, none between the two | 2 |
| no edge: right entities joined, another sentence cited | 1 |

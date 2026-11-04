# Ingest order sensitivity: medium world, ambiguous ambiguity

Entity linking can depend on what the pipeline has already seen, and incremental ingest does not control arrival order. The same corpus was ingested 7 times, once as generated and 6 times in shuffled orders, each into a fresh pipeline, and scored against the generator's gold.

## Setup

- Preset `medium` with the `ambiguous` ambiguity setting, world seed 0, 1002 documents.
- Orders: as generated, then shuffle seeds [1, 2, 3, 4, 5, 6].
- Reproduce: `uv run kgbench order --preset medium --difficulty ambiguous --orders 6 --out docs/results/order-medium.md --json docs/results/order-medium.json`

## Result

Every metric came out the same in every order.

| metric | mean | std | min | max |
|---|---|---|---|---|
| canonical entities | 1227.000 | 0.000 | 1227.000 | 1227.000 |
| linking precision | 0.926 | 0.000 | 0.926 | 0.926 |
| linking recall | 0.767 | 0.000 | 0.767 | 0.767 |
| linking F1 | 0.839 | 0.000 | 0.839 | 0.839 |
| linking F1 (homonym) | 0.734 | 0.000 | 0.734 | 0.734 |
| linking F1 (alias) | 0.673 | 0.000 | 0.673 | 0.673 |
| linking F1 (plain) | 0.937 | 0.000 | 0.937 | 0.937 |
| entities F1 | 0.789 | 0.000 | 0.789 | 0.789 |
| triples F1 | 0.000 | 0.000 | 0.000 | 0.000 |
| edges F1 | 0.722 | 0.000 | 0.722 | 0.722 |

## What this does and does not show

- This is the offline pipeline: heuristic entity detection and linking that merges by name. Merging by name does not depend on what came first, so no spread is expected here, and none is the finding. It shows the harness reports no order effect when there is none (the tests show it reports one when there is).
- The real stack (spaCy, an LLM-built graph, linking against a persistent entity store) was not run. Its linking may well depend on order; run `kgbench order --real` with the services up to measure that.
- Scores depend only on which mentions are grouped, not on canonical ids. Entity alignment breaks vote ties by entity id, so the scorer adds no order effect of its own (it used to: ties went to whichever mention arrived first).
